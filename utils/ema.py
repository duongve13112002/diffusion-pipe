"""Exponential moving average of the trained weights, for any model (opt-in with ema_decay).

    ema_decay = 0.9999        # turns EMA on
    ema_update_every = 1      # optimizer steps between updates (decay is compounded to match)
    ema_start_step = 0        # until this step the average just follows the weights
    ema_device = 'cpu'        # 'cpu' or 'gpu': where the average lives
    ema_dtype = 'float32'     # or 'bfloat16': half the memory, stochastically rounded

Every trainable parameter gets a shadow copy, updated after each optimizer step as
shadow = decay * shadow + (1 - decay) * weight. Frozen parameters have no shadow: their EMA is
the weight itself. Each save then also writes `<save dir>/ema/`, in exactly the layout of the
normal save (the model's own save_model / save_adapter), so it loads anywhere the normal one
does. Training checkpoints carry the shadow too, so a resumed run continues the same average.

The shadow is kept on data-parallel rank 0 of each pipeline stage only, the ranks that save:
every data-parallel replica holds identical weights. It is fp32 by default because a 0.9999
decay moves the average by 1e-4 of the gap to the weights per step, under half a bf16 step
unless that gap is about 20x the weights' own magnitude: rounded to nearest, a bf16 average
would in practice never leave its starting weights. ema_dtype = 'bfloat16'
therefore rounds each update stochastically (up or down with the probabilities that keep its
expectation exact), so it does follow the weights, with rounding noise of about one bf16 step,
the precision of bf16 weights themselves. fp32 takes 4 bytes per parameter (12 GB for a 3B full
fine-tune), bf16 2; hence ema_device = 'cpu' by default. With it, each update copies the weights
to the host, so ema_update_every > 1 keeps that off the critical path on large models. An
adapter's average is small enough for 'gpu'.
"""

from pathlib import Path

import torch

DEVICES = ('cpu', 'gpu')
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
EMA_SUBDIR = 'ema'
_OPTION_KEYS = ('ema_update_every', 'ema_start_step', 'ema_device', 'ema_dtype')


def apply_ema_config_defaults(config):
    """Validate the ema_* options and fill their defaults. Returns whether EMA is on."""
    if 'ema_decay' not in config:
        stray = [k for k in _OPTION_KEYS if k in config]
        if stray:
            raise ValueError(f'{stray} set but ema_decay is not, so EMA is off. Set ema_decay (e.g. 0.9999) '
                             'to train with EMA, or remove them.')
        return False
    decay = config['ema_decay']
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < decay < 1:
        raise ValueError(f'ema_decay must be a number strictly between 0 and 1, got {decay!r}')
    config['ema_decay'] = float(decay)
    every = config.setdefault('ema_update_every', 1)
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(f'ema_update_every must be an integer >= 1, got {every!r}')
    start = config.setdefault('ema_start_step', 0)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(f'ema_start_step must be an integer >= 0, got {start!r}')
    device = config.setdefault('ema_device', 'cpu')
    if device not in DEVICES:
        raise ValueError(f'ema_device must be one of {DEVICES}, got {device!r}')
    dtype = config.setdefault('ema_dtype', 'float32')
    if dtype not in DTYPES:
        raise ValueError(f'ema_dtype must be one of {tuple(DTYPES)}, got {dtype!r}')
    return True


def to_bfloat16_stochastic(x):
    """fp32 -> bf16, rounding up or down at random with the probabilities that make the
    result equal x in expectation. bf16 is the top 16 bits of fp32: add uniform noise below
    them and truncate."""
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def saved_key(param, is_adapter):
    """The key the saver writes this parameter under (None: the saver skips it)."""
    key = getattr(param, 'original_name', None)
    if key is not None and is_adapter:
        key = key.replace('.default', '').replace('.modules_to_save', '')
    return key


class ParameterEMA:
    """fp32 moving average of the trainable parameters among `named_parameters`."""

    def __init__(self, named_parameters, decay, update_every=1, start_step=0, device='cpu', is_adapter=False,
                 dtype='float32'):
        self.decay = float(decay)
        self.dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
        self.update_every = int(update_every)
        self.start_step = int(start_step)
        self.is_adapter = is_adapter
        if device == 'gpu':
            self.device = torch.device('cuda', torch.cuda.current_device())
        else:
            self.device = torch.device('cpu')
        self.params = {}
        self.shadow = {}
        for _, p in named_parameters:
            if not p.requires_grad:
                continue
            key = saved_key(p, is_adapter)
            if key is None or key in self.params:
                continue
            self.params[key] = p
            self.shadow[key] = p.detach().to(self.device, torch.float32, copy=True).to(self.dtype)
        self.num_updates = 0

    def __len__(self):
        return len(self.shadow)

    def num_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.shadow.values())

    @torch.no_grad()
    def update(self, step):
        """Call after the optimizer step `step` (1-based)."""
        if step <= self.start_step:
            for key, p in self.params.items():
                self.shadow[key].copy_(p.detach())
            return
        if (step - self.start_step) % self.update_every:
            return
        # update_every steps of `decay`, compounded into one.
        weight = 1.0 - self.decay ** self.update_every
        for key, p in self.params.items():
            shadow = self.shadow[key]
            # Moved in the parameter's own (narrower) dtype, widened where the shadow lives.
            weights = p.detach().to(shadow.device).to(torch.float32)
            if shadow.dtype == torch.float32:
                shadow.lerp_(weights, weight)
            else:
                shadow.copy_(to_bfloat16_stochastic(shadow.float().lerp_(weights, weight)))
        self.num_updates += 1

    def averaged(self, key, param):
        """The EMA of `param`, in its own dtype on the CPU: the shadow, or the weight itself
        for a parameter that is not trained."""
        shadow = self.shadow.get(key)
        source = param.detach() if shadow is None else shadow
        return source.to(device='cpu', dtype=param.dtype)

    def state_dict(self):
        return {'shadow': {k: v.cpu() for k, v in self.shadow.items()}, 'num_updates': self.num_updates,
                'decay': self.decay}

    def load_state_dict(self, state):
        """Restore the shadows found in `state`. Returns the keys it had none for, which keep
        their current value (the weights, when EMA was switched on after the checkpoint)."""
        saved = state['shadow']
        missing = []
        for key, shadow in self.shadow.items():
            if key in saved:
                # Across an ema_dtype change too: fp32 -> bf16 rounds to nearest, once.
                shadow.copy_(saved[key])
            else:
                missing.append(key)
        self.num_updates = int(state.get('num_updates', 0))
        return missing


def checkpoint_file(save_root, tag, stage_id):
    return Path(save_root) / tag / f'ema_stage_{stage_id:02d}.pt'


def save_ema_checkpoint(ema, save_root, tag, stage_id):
    torch.save(ema.state_dict(), checkpoint_file(save_root, tag, stage_id))


def load_ema_checkpoint(ema, save_root, tag):
    """Load this rank's shadows from every stage file of checkpoint `tag`, by name, so a run
    may resume with a different pipeline split. Returns (files found, keys not found)."""
    files = sorted((Path(save_root) / tag).glob('ema_stage_*.pt'))
    if not files:
        return 0, list(ema.shadow)
    merged = {'shadow': {}, 'num_updates': 0}
    for path in files:
        # mmap: a stage only reads the tensors it holds, not every stage's average.
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        merged['shadow'].update(state['shadow'])
        merged['num_updates'] = max(merged['num_updates'], int(state.get('num_updates', 0)))
    return len(files), ema.load_state_dict(merged)
