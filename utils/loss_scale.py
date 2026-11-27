"""Dynamic loss scaling for float16 compute (MOSS-Image with [model] compute_dtype = 'float16').

A float16 backward pass underflows the small gradients of a diffusion loss, so the loss is scaled up
before backward and the gradients are scaled back down before the step:

    loss' = loss * S            (the forward value is untouched, only the gradient is scaled)
    grad  = grad' / S           (before clipping, the norm and the optimizer step)

S is dynamic. A step whose gradient has a NaN or inf (the scale was too high, or a float16 overflow)
is dropped and S is cut by `backoff_factor`; after `growth_interval` steps in a row without one, S is
raised by `growth_factor`. A dropped step leaves the weights, the optimizer state and the learning rate
schedule as they were, as in utils/step_guard.py. Once S is down to `min_scale` a non-finite gradient is
no longer an effect of the scale and goes to the non-finite step guard, which stops the run after
`nonfinite_patience` of them in a row.

    [loss_scale]               # optional; the defaults below apply when compute_dtype = 'float16'
    init_scale = 65536.0
    growth_factor = 2.0
    backoff_factor = 0.5
    growth_interval = 2000
    min_scale = 1.0

Every rank decides from the same gradient norm (summed over the pipeline stages and averaged over the
data-parallel ranks, utils/patches.py), so the scales stay equal without any extra communication.
Kept free of training-stack imports so it can be unit-tested on CPU.
"""

import math

import torch

from utils.common import warn_ignored

KEY = 'loss_scale'
DEFAULTS = {
    'init_scale': 65536.0,
    'growth_factor': 2.0,
    'backoff_factor': 0.5,
    'growth_interval': 2000,
    'min_scale': 1.0,
}
FLOAT16_NAMES = ('float16', 'fp16')
# The names [model] compute_dtype takes (models/moss_image.py DTYPE_NAMES), as torch dtypes.
COMPUTE_DTYPES = {'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16, 'float16': torch.float16,
                  'fp16': torch.float16, 'float32': torch.float32, 'fp32': torch.float32}


def compute_dtype_of(model_config):
    """[model] compute_dtype as a torch dtype, or None when it is absent or not a dtype name."""
    value = model_config.get('compute_dtype')
    if isinstance(value, torch.dtype):
        return value
    return COMPUTE_DTYPES.get(str(value).lower()) if value is not None else None


def computes_in_float16(model_config):
    """Whether [model] asks for float16 compute (compute_dtype, as a name or a torch dtype)."""
    value = model_config.get('compute_dtype', '')
    return value == torch.float16 or str(value).lower() in FLOAT16_NAMES


def refuse_compute_dtype_off_moss(model_config):
    """[model] compute_dtype and fp16_fix are read by MOSS-Image only; on another model they would be ignored."""
    if model_config.get('type') == 'moss_image':
        return
    for key in ('compute_dtype', 'fp16_fix', 'fp16_fix_scale'):
        if key in model_config:
            raise ValueError(f"[model] {key} only applies to model type 'moss_image', not {model_config.get('type')!r}: "
                             "only MOSS-Image computes in a dtype other than the one its weights are stored in. "
                             "Remove it.")


def _number(table, name, low, strict=False, high=None):
    value = table[name]
    bad = (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           or value < low or (strict and value == low) or (high is not None and value >= high))
    if bad:
        wanted = f"a number {'>' if strict else '>='} {low}" + (f' and below {high}' if high is not None else '')
        raise ValueError(f'{KEY}.{name} must be {wanted}, got {value!r}')


def apply_loss_scale_config_defaults(config):
    """Validate config['loss_scale'] and fill in its defaults.

    The table is optional when [model] compute_dtype is float16 (the defaults apply); with any other
    compute dtype it is ignored with a warning, since there is nothing to scale. Returns the spec, or None when the
    run does not compute in float16. gradient_release is refused: it frees each gradient during
    backward, so there is nothing left to unscale when the step runs. The non-finite step guard is
    switched on (skip_nonfinite_steps = true), which drops the steps the scale gives up on; an explicit
    false is refused.
    """
    model_config = config.get('model', {})
    if not computes_in_float16(model_config):
        if KEY in config:
            warn_ignored(f'[{KEY}] is ignored: it applies with [model] compute_dtype = "float16"; a bfloat16 or '
                         'float32 backward needs no loss scaling.')
            del config[KEY]
        return None
    table = config.get(KEY, {})
    if not isinstance(table, dict):
        raise ValueError(f'{KEY} must be a table, got {table!r}')
    unknown = sorted(set(table) - set(DEFAULTS))
    if unknown:
        raise ValueError(f'{KEY} has unknown fields {unknown}; valid: {sorted(DEFAULTS)}')
    table = {**DEFAULTS, **table}
    _number(table, 'init_scale', 1.0)
    _number(table, 'growth_factor', 1.0, strict=True)
    _number(table, 'backoff_factor', 0.0, strict=True, high=1.0)
    _number(table, 'min_scale', 0.0, strict=True)
    value = table['growth_interval']
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{KEY}.growth_interval must be a whole number >= 1, got {value!r}')
    if table['min_scale'] > table['init_scale']:
        raise ValueError(f"{KEY}.min_scale ({table['min_scale']!r}) is above init_scale ({table['init_scale']!r})")
    optimizer = config.get('optimizer')
    if isinstance(optimizer, dict) and optimizer.get('gradient_release', False):
        raise ValueError('compute_dtype = "float16" needs the gradients at the end of the step to unscale them, '
                         'and gradient_release frees them during backward. Turn one of them off.')
    if isinstance(optimizer, dict) and str(optimizer.get('type', '')).lower() == 'offload':
        raise ValueError("compute_dtype = 'float16' scales the gradients until the engine step, but the CPU-offload "
                         "optimizer (type = 'offload') copies each gradient to the host during backward, before it "
                         'is unscaled, and would step on the scaled one. Use another optimizer.')
    adapter = config.get('adapter')
    if isinstance(adapter, dict) and (adapter.get('dtype') == torch.float16
                                      or str(adapter.get('dtype', '')).lower() in FLOAT16_NAMES):
        raise ValueError("[adapter] dtype = 'float16' with compute_dtype = 'float16': the gradients of float16 "
                         'parameters are unscaled in float16, where the small ones the loss scale protected '
                         "flush to zero. Leave the adapter dtype unset (float32 over a float16 base) or use 'float32'.")
    if config.get('skip_nonfinite_steps') is False:
        raise ValueError("compute_dtype = 'float16' scales the loss and drops the steps whose gradient overflows, "
                         'which is what skip_nonfinite_steps does: remove skip_nonfinite_steps = false.')
    config['skip_nonfinite_steps'] = True
    config[KEY] = table
    return table


def default_adapter_dtype(model_config):
    """The dtype an [adapter] trains in unless it says: the weights' own, except over a float16 MOSS-Image
    base, where float16 holds the optimizer's small updates badly (and its gradients cannot be unscaled in
    float16) and the adapter trains in float32, the layer casting it per call. Other models keep the weights'
    dtype."""
    dtype = model_config['dtype']
    if model_config.get('type') == 'moss_image' and str(dtype).lower() in FLOAT16_NAMES:
        return 'float32'
    return dtype


class _ScaleGradient(torch.autograd.Function):
    """Identity forward; the gradient on the way back is multiplied by the scaler's current scale."""

    @staticmethod
    def forward(ctx, loss, scaler):
        ctx.scaler = scaler
        return loss.view_as(loss)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scaler.scale, None


class LossScaler:
    """The dynamic scale and its three jobs: scale the loss's gradient, unscale the gradients at the
    step, and move the scale after the step.

    `scale` is a Python float, the same on every rank. `overflows` counts the steps dropped because
    the gradient was not finite while the scale could still fall.
    """

    def __init__(self, init_scale=DEFAULTS['init_scale'], growth_factor=DEFAULTS['growth_factor'],
                 backoff_factor=DEFAULTS['backoff_factor'], growth_interval=DEFAULTS['growth_interval'],
                 min_scale=DEFAULTS['min_scale']):
        self.scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self.clean_steps = 0
        self.overflows = 0
        self.last_overflow_scale = None

    @classmethod
    def from_spec(cls, spec):
        return cls(**{name: spec[name] for name in DEFAULTS})

    def scale_loss(self, loss):
        """The loss with its gradient scaled; the value is the loss itself."""
        return _ScaleGradient.apply(loss, self)

    @torch.no_grad()
    def unscale_(self, parameters):
        """Divide every gradient by the current scale, in place."""
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return
        inverse = 1.0 / self.scale
        by_kind = {}
        for g in grads:
            by_kind.setdefault((g.device, g.dtype), []).append(g)
        for group in by_kind.values():
            torch._foreach_mul_(group, inverse)

    def can_fall(self):
        return self.scale > self.min_scale

    def update(self, found_inf):
        """Move the scale after a step: cut it on a non-finite gradient, raise it after
        growth_interval clean steps. Returns True when the step is to be dropped because of the scale
        (a non-finite gradient while the scale could still fall)."""
        if found_inf:
            if not self.can_fall():
                self.clean_steps = 0
                return False
            self.last_overflow_scale = self.scale
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self.clean_steps = 0
            self.overflows += 1
            return True
        self.clean_steps += 1
        if self.clean_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self.clean_steps = 0
        return False

    def log_scalars(self):
        return {'train/loss_scale': float(self.scale), 'train/loss_scale_overflows': float(self.overflows)}

    def state_dict(self):
        return {'scale': self.scale, 'clean_steps': self.clean_steps, 'overflows': self.overflows}

    def load_state_dict(self, state):
        self.scale = float(state['scale'])
        self.clean_steps = int(state['clean_steps'])
        self.overflows = int(state['overflows'])
