"""Skipping optimizer steps whose gradient is not finite, for every model and every way of training.

One batch with a NaN or inf in its gradient, applied as it is, puts NaN into the weights (a full
fine-tune) or the adapter and the optimizer's moments (a LoRA), and the run is over without any
sign until the loss turns NaN. With `skip_nonfinite_steps = true` the step that would do that is
dropped instead: the weights, the optimizer state and the learning rate schedule stay as they were,
the step is counted and logged, and the run goes on with the next batch.

The check is on the gradient norm DeepSpeed's engine step already needs (utils/patches.py): the
norm is summed over every pipeline stage and averaged over the data-parallel ranks, so every rank
holds the same number and takes the same branch, with no extra agreement step. A skipped step uses
DeepSpeed's own protocol for a step that overflowed: after `optimizer.step()` the engine reads
`optimizer.overflow`, and when it is true it does not advance the learning rate scheduler and counts
a skipped step. Here the optimizer's `step` is replaced by a no-op for that one call and
`overflow` is set, and both are put back afterwards. The gradients are cleared by the engine as for
any step.

What it does not catch: a non-finite loss with a finite gradient, and weights that were already
non-finite. `train.py --nan_check` finds the layer and the images for those.

Not available with `gradient_release`: the gradients are freed during backward, so none is left to
check when the step runs.
"""

from contextlib import contextmanager

import torch

ENABLE = 'skip_nonfinite_steps'
PATIENCE = 'nonfinite_patience'
DEFAULT_PATIENCE = 8


class NonFiniteStepsError(RuntimeError):
    pass


def apply_step_guard_config_defaults(config):
    """Validate the two top-level keys and write their defaults into config."""
    enabled = config.setdefault(ENABLE, False)
    if not isinstance(enabled, bool):
        raise ValueError(f'{ENABLE} must be true or false, got {enabled!r}')
    patience = config.setdefault(PATIENCE, DEFAULT_PATIENCE)
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ValueError(f'{PATIENCE} must be a whole number >= 0 (0 never stops the run), got {patience!r}')
    optimizer = config.get('optimizer')
    if enabled and isinstance(optimizer, dict) and optimizer.get('gradient_release', False):
        raise ValueError(
            f'{ENABLE} needs the gradients at the end of the step, and gradient_release frees them during '
            'backward. Turn one of them off; GenericOptim has its own per-parameter skip_invalid_grads.')


class NonFiniteGuard:
    """Decides, for one engine step at a time, whether the step is applied, and keeps the counts.

    `step_skipped` is the verdict of the last step, `consecutive` the skipped steps in a row and
    `skipped` the total since the process started (a resumed run starts again from 0).
    """

    def __init__(self, patience=DEFAULT_PATIENCE):
        self.patience = patience
        self.consecutive = 0
        self.skipped = 0
        self.step_skipped = False

    @staticmethod
    def is_finite(norm):
        return bool(torch.isfinite(norm).all())

    @contextmanager
    def _no_step(self, optimizer):
        had_overflow = hasattr(optimizer, 'overflow')
        previous = getattr(optimizer, 'overflow', None)
        # An instance-level step can already exist: the LR scheduler wraps the optimizer's step to count
        # calls, and the Prodigy pipeline sync replaces it. Put that one back, not the class's.
        had_instance_step = 'step' in vars(optimizer)
        instance_step = vars(optimizer).get('step')
        optimizer.step = lambda *args, **kwargs: None
        optimizer.overflow = True
        try:
            yield
        finally:
            if had_instance_step:
                optimizer.step = instance_step
            else:
                del optimizer.step
            if had_overflow:
                optimizer.overflow = previous
            else:
                del optimizer.overflow

    def run_step(self, optimizer, norm, take_step):
        """Call take_step() (the engine's own step) as it is when `norm` is finite, and with the
        optimizer's step suppressed when it is not. Returns what take_step returns."""
        if self.is_finite(norm):
            self.step_skipped = False
            self.consecutive = 0
            return take_step()
        self.step_skipped = True
        self.consecutive += 1
        self.skipped += 1
        with self._no_step(optimizer):
            return take_step()

    def check_patience(self):
        """Raise after `patience` skipped steps in a row; call it after the step, on every rank."""
        if self.patience and self.consecutive >= self.patience:
            raise NonFiniteStepsError(
                f'{self.consecutive} optimizer steps in a row had a non-finite gradient and were skipped '
                f'({self.skipped} in all). Something in the model, the data or the learning rate produces '
                'NaN or inf every step; skipping cannot help. Run once with --nan_check to find the layer '
                f'and the images, or raise {PATIENCE} (0 never stops).')

    def log_scalars(self):
        return {'train/nonfinite_skipped': float(self.skipped)}
