"""Distillation toward the model the run started from (MOSS-Image), for adapters and full fine-tunes.

    loss = L(v, target) + lambda(sigma, step) * L(v, v_teacher)

`v` is the student's velocity, `target` the flow-matching target of the batch and `v_teacher` the
velocity of the frozen starting model for the same noisy latent, noise level and text embedding. L is
the loss the run already uses (MSE, or huber_delta / smooth_l1_beta when set), masked and averaged
the same way. The second term is a regulariser: it keeps what the model predicts close to what it
predicted before fine-tuning, wherever lambda is above 0, and so limits how far a small dataset can
drag it (KohakuMossTrainer's distillation; its data loss and distillation weights are 1 and
`loss_weight` here).

lambda = loss_weight * shape(sigma) * decay(step), the schedule helpers of the anima refiner's
teacher guidance (models/teacher_guidance.py), reused as they are:

    shape = 'constant'                       1 at every noise level (the default)
    shape = 'sigmoid'                        sigmoid((sigma - t_mid) / width): more at high noise
    decay = 'none'                           1 for the whole run (the default)
    decay = 'linear' | 'cosine'              falls from 1 to 0 over decay_steps optimizer steps

The teacher is the starting weights, which differs by training mode (models/moss_image.py): for an
adapter it is the same blocks with the adapter switched off, for a full fine-tune a frozen copy of
the model as loaded. Both are read from the model file the config names, so a resumed run distills
toward that file's weights, not toward the checkpoint it resumed.

Kept free of training-stack imports so the rules and the loss can be unit-tested on CPU.
"""

import math

import torch
import torch.nn.functional as F

from models.teacher_guidance import DECAYS, SHAPES, lambda_at

KEY = 'teacher'
_KEYS = ('loss_weight', 'shape', 't_mid', 'width', 'decay', 'decay_steps')


class DistillConfig:
    """The validated [teacher] table. Attribute names are the ones models/teacher_guidance.py's
    schedule helpers read."""

    def __init__(self, loss_weight, shape, t_mid, width, decay, decay_steps):
        self.loss_weight = loss_weight
        self.shape = shape
        self.t_mid = t_mid
        self.width = width
        self.decay = decay
        self.decay_steps = decay_steps

    @property
    def enabled(self):
        return self.loss_weight > 0


def _number(table, key, default):
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'[{KEY}] {key} must be a finite number, got {value!r}')
    return float(value)


def parse_distill_config(config):
    """The [teacher] table of a MOSS-Image config as a DistillConfig, or None when it is absent or
    its loss_weight is 0. Raises on a key it does not know, a value out of range, or a setting
    distillation cannot run with (adapters on several pipeline stages, block swapping, compile)."""
    table = config.get(KEY)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError(f'[{KEY}] must be a table, got {table!r}')
    unknown = sorted(set(table) - set(_KEYS))
    if unknown:
        raise ValueError(f'[{KEY}] has unknown keys {unknown}; known keys: {", ".join(_KEYS)}. '
                         'The teacher is always the model the run starts from, so there is no path to set.')
    loss_weight = _number(table, 'loss_weight', 0.0)
    if loss_weight < 0:
        raise ValueError(f'[{KEY}] loss_weight must be >= 0, got {loss_weight}')
    if loss_weight == 0:
        return None

    shape = table.get('shape', 'constant')
    if shape not in SHAPES:
        raise ValueError(f'[{KEY}] shape must be one of {SHAPES}, got {shape!r}')
    t_mid = _number(table, 't_mid', 0.5)
    if not 0.0 <= t_mid <= 1.0:
        raise ValueError(f'[{KEY}] t_mid must be in [0, 1], got {t_mid}')
    width = _number(table, 'width', 0.15)
    if width <= 0:
        raise ValueError(f'[{KEY}] width must be > 0, got {width}')
    decay = table.get('decay', 'none')
    if decay not in DECAYS:
        raise ValueError(f'[{KEY}] decay must be one of {DECAYS}, got {decay!r}')
    decay_steps = table.get('decay_steps', 0)
    if isinstance(decay_steps, bool) or not isinstance(decay_steps, int) or decay_steps < 0:
        raise ValueError(f'[{KEY}] decay_steps must be a whole number >= 0, got {decay_steps!r}')
    if decay != 'none' and decay_steps <= 0:
        raise ValueError(f'[{KEY}] decay = {decay!r} needs decay_steps > 0 (the optimizer steps over which '
                         "lambda falls to 0); use decay = 'none' to keep the term on for the whole run")

    if config.get('compile'):
        raise ValueError(f'[{KEY}] does not work with compile = true: the teacher runs the model\'s layers outside '
                         'the compiled pipeline. Set compile = false, or remove [teacher].')
    if config.get('blocks_to_swap', 0):
        raise ValueError(f'[{KEY}] cannot be combined with blocks_to_swap: block swapping already moves the '
                         'blocks between CPU and GPU every step, and the teacher is a second full forward.')
    if isinstance(config.get('adapter'), dict) and config['adapter'].get('init_from_existing'):
        raise ValueError(f'[{KEY}] cannot be combined with [adapter] init_from_existing: the teacher is the model with '
                         'the adapter switched off, which is the base model, not the base plus the adapter the run '
                         'starts from, so the term would pull the first steps away from the starting adapter. '
                         'Remove one of them.')
    if 'adapter' in config and int(config.get('pipeline_stages', 1)) != 1:
        raise ValueError(f'[{KEY}] with an [adapter] needs pipeline_stages = 1: the teacher is the same blocks '
                         'with the adapter switched off, which only a stage that holds the whole model can run. '
                         'Use data parallelism across the GPUs, or train a full fine-tune (which keeps a frozen '
                         'copy on the first stage).')
    return DistillConfig(loss_weight, shape, t_mid, width, decay, int(decay_steps))


def distill_warnings(distill, config):
    """Advisory messages for a valid but risky configuration, printed once at startup."""
    messages = []
    if distill.decay == 'none':
        messages.append(
            f"[{KEY}] decay = 'none': the term never fades, so the model is held near its starting predictions "
            'for the whole run. That is the point of a regulariser, but it also caps how far the fine-tune can '
            'move; use decay = \'linear\' or \'cosine\' with decay_steps to let it go.')
    return messages


def elementwise_loss(output, target, config):
    """The data loss's per-element form: huber_delta / smooth_l1_beta when set, else MSE."""
    if 'huber_delta' in config:
        return F.huber_loss(output, target, reduction='none', delta=config['huber_delta'])
    if 'smooth_l1_beta' in config:
        return F.smooth_l1_loss(output, target, reduction='none', beta=config['smooth_l1_beta'])
    return F.mse_loss(output, target, reduction='none')


def distill_loss(output, teacher, sigma, mask, step, distill, config):
    """lambda * L(output, teacher), averaged over every element like the data loss.

    Returns (term, mean lambda). `teacher` is detached; `sigma` has one value per sample and `mask`
    is the data loss's (empty for none).
    """
    with torch.autocast('cuda', enabled=False):
        output = output.to(torch.float32)
        teacher = teacher.detach().to(output.device, torch.float32)
        if teacher.shape != output.shape:
            raise RuntimeError(f'The teacher velocity {tuple(teacher.shape)} does not match the prediction '
                               f'{tuple(output.shape)}')
        sigma = sigma.to(output.device, torch.float32).reshape(-1, 1)
        lam = lambda_at(sigma, step, distill)
        loss = elementwise_loss(output, teacher, config)
        if mask.numel() > 0:
            loss = loss * mask.to(output.device, torch.float32)
        lam_b = lam.reshape(-1, *([1] * (loss.ndim - 1)))
        return (loss * lam_b).mean(), lam.mean()
