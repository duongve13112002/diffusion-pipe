"""Exponential moving average of the trained weights, for any model (opt-in with ema_decay).

    ema_decay = 0.9999        # turns EMA on
    ema_update_every = 1      # optimizer steps between updates (decay is compounded to match)
    ema_start_step = 0        # until this step the average just follows the weights
    ema_device = 'cpu'        # 'cpu' or 'gpu': where the average lives
    ema_dtype = 'float32'     # or 'bfloat16': half the memory, stochastically rounded
    ema_init_path = '...'     # start the average from a saved EMA instead of the weights

ema_init_path takes a model or adapter save (a directory, e.g. a previous run's ema/, or the
.safetensors inside it) and starts the average there, so fine-tuning from a run's raw weights
keeps that run's average rather than restarting it from those weights. It must be a save of
the run being trained: every trained parameter, by name and shape (an adapter save for an
adapter run, a full save for a full fine-tune), or training stops with an error. A resumed run
uses its checkpoint's average instead.

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
fine-tune), bf16 2; hence ema_device = 'cpu' by default. Updates then stay off the training's
critical path: an fp32 average copies the weights into a pinned host buffer without blocking
(ordered on the CUDA stream, so it captures exactly this step's weights) and a background thread
averages them while the next step trains, which takes another copy of the trained weights in
host RAM (6 GB for 3B bf16 weights). A bf16 average is updated on the GPU instead (random
numbers are slow on the CPU), the shadow copied up and back down one tensor at a time, again
without blocking the host. An adapter's average is small enough for 'gpu'.
"""

import threading
from pathlib import Path

import torch

from utils.common import torch_save_atomic
from utils.dora import DORA_SCALE, saved_key as dora_saved_key

DEVICES = ('cpu', 'gpu')
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
EMA_SUBDIR = 'ema'
_OPTION_KEYS = ('ema_update_every', 'ema_start_step', 'ema_device', 'ema_dtype', 'ema_init_path')


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
    init_path = config.get('ema_init_path')
    if init_path is not None:
        if not isinstance(init_path, str) or not init_path:
            raise ValueError(f'ema_init_path must be a path, got {init_path!r}')
        if start > 0:
            raise ValueError('ema_init_path and ema_start_step > 0 contradict each other: before ema_start_step '
                             'the average is the weights. Remove one of them.')
        resolve_ema_init_file(init_path)
    return True


# Trained parameters a save writes to a file of their own (name prefix -> files, relative to the
# save directory, holding them without the prefix): Iris's REPA projector.
_COMPANION_FILES = {'repa.': ('repa_projector.safetensors', 'repa/repa_projector.safetensors')}


def resolve_ema_init_file(path):
    """The .safetensors file of a model or adapter save: `path` itself, or the one file in it."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f'ema_init_path {str(path)!r} does not exist')
    for name in ('model.safetensors', 'adapter_model.safetensors'):
        if (path / name).is_file():
            return path / name
    files = sorted(path.glob('*.safetensors'))
    if len(files) != 1:
        raise ValueError(f'ema_init_path {str(path)!r}: expected model.safetensors, adapter_model.safetensors or '
                         f'a single .safetensors file, found {[f.name for f in files]}')
    return files[0]


def load_ema_weights(ema, path):
    """Start `ema`'s average from the save at `path` (see resolve_ema_init_file). Every trained
    parameter must be there, matched by name (exactly, or by its longest dotted suffix when the
    save adds a prefix such as an adapter's 'diffusion_model.') with the same shape; otherwise
    ValueError and nothing is changed. Returns the number of tensors loaded."""
    from contextlib import ExitStack
    from safetensors import safe_open
    ema.wait()
    file = resolve_ema_init_file(path)
    with ExitStack() as stack:
        handles = {'': stack.enter_context(safe_open(str(file), framework='pt', device='cpu'))}
        for prefix, names in _COMPANION_FILES.items():
            for name in names:
                if (file.parent / name).is_file():
                    handles[prefix] = stack.enter_context(
                        safe_open(str(file.parent / name), framework='pt', device='cpu'))
                    break
        by_suffix = {}
        for prefix, handle in handles.items():
            for name in handle.keys():
                parts = (prefix + name).split('.')
                for i in range(len(parts)):
                    by_suffix.setdefault('.'.join(parts[i:]), []).append((prefix, name))

        plan, missing, wrong_shape = {}, [], []
        for key, shadow in ema.shadow.items():
            # A DoRA magnitude vector (out,) is saved as <module>.dora_scale (out, 1).
            found = by_suffix.get(key, []) or by_suffix.get(dora_saved_key(key), [])
            exact = [(prefix, name) for prefix, name in found if prefix + name == key]
            found = exact or found
            if not found:
                missing.append(key)
                continue
            if len(found) > 1:
                raise ValueError(f'ema_init_path {file}: {key} matches several tensors: '
                                 f'{[prefix + name for prefix, name in found]}')
            prefix, name = found[0]
            shape = list(handles[prefix].get_slice(name).get_shape())
            if name.endswith(DORA_SCALE) and shape == [*shadow.shape, 1]:
                shape = shape[:-1]
            if shape != list(shadow.shape):
                wrong_shape.append(f'{prefix + name} {shape} (trained {key} is {list(shadow.shape)})')
            plan[key] = found[0]
        if missing or wrong_shape:
            saved = [prefix + name for prefix, handle in handles.items() for name in handle.keys()]
            adapter_save = any('lora_' in name or 'lokr_' in name for name in saved)
            kind = ''
            if adapter_save and not ema.is_adapter:
                kind = ' It is an adapter save, but this run is a full fine-tune.'
            elif ema.is_adapter and not adapter_save:
                kind = ' It is a full model save, but this run trains an adapter.'
            problems = []
            if missing:
                problems.append(f'{len(missing)} of {len(ema.shadow)} trained parameters are not in it '
                                f'(e.g. {missing[:3]})')
            if wrong_shape:
                problems.append(f'{len(wrong_shape)} have another shape (e.g. {wrong_shape[:3]})')
            raise ValueError(f'ema_init_path {file} is not a save of the model being trained: '
                             f'{"; ".join(problems)}.{kind} Point it at the ema/ directory (or its .safetensors) '
                             'of a run with this model and these adapter settings.')
        for key, (prefix, name) in plan.items():
            ema.shadow[key].copy_(handles[prefix].get_tensor(name).to(torch.float32).reshape(ema.shadow[key].shape))
    if plan:
        ema.following = False
    return len(plan)


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


def _pinned(tensor):
    """`tensor` in page-locked memory, so copies to and from a GPU can run without blocking.
    Unpinned (copies then block) where that is not available."""
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


class ParameterEMA:
    """Moving average (fp32 or stochastically rounded bf16) of the trainable parameters among
    `named_parameters`."""

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
        # Until start_step the average is the weights themselves: averaged() returns them, and
        # the shadow is set from them once, at start_step, instead of copied every step.
        self.following = self.start_step > 0
        self._offload_ready = False
        self._staging = {}
        self._pending = None    # CUDA event: the last update's copies are done
        self._worker = None     # thread finishing the last fp32 update on the CPU
        self._error = None

    def __len__(self):
        return len(self.shadow)

    def num_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.shadow.values())

    def _offloaded(self):
        """Whether the shadow is in host RAM while the weights are on a GPU."""
        return self.device.type == 'cpu' and any(p.is_cuda for p in self.params.values())

    def _prepare_offload(self):
        if self._offload_ready:
            return
        for key, p in self.params.items():
            if self.dtype == torch.float32:
                self._staging[key] = _pinned(torch.empty(p.shape, dtype=p.dtype))
            else:
                self.shadow[key] = _pinned(self.shadow[key])
        self._offload_ready = True

    def _record_pending(self):
        if torch.cuda.is_available() and any(p.is_cuda for p in self.params.values()):
            self._pending = torch.cuda.Event()
            self._pending.record()

    def wait(self):
        """Block until the last update has landed in the shadow. Everything that reads or
        writes the shadow calls this first."""
        if self._worker is not None:
            self._worker.join()
            self._worker = None
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError('EMA update failed') from error

    @torch.no_grad()
    def update(self, step):
        """Call after the optimizer step `step` (1-based)."""
        self.wait()
        if step <= self.start_step or self.following:
            self.following = step < self.start_step
            if not self.following:
                # start_step (or a resume from before it): the average starts at the weights.
                if self._offloaded():
                    self._prepare_offload()
                for key, p in self.params.items():
                    self.shadow[key].copy_(p.detach(), non_blocking=True)
                self._record_pending()
            return
        if (step - self.start_step) % self.update_every:
            return
        # update_every steps of `decay`, compounded into one.
        weight = 1.0 - self.decay ** self.update_every
        if self._offloaded():
            self._prepare_offload()
            if self.dtype == torch.float32:
                self._update_in_background(weight)
            else:
                self._update_where_the_weights_are(weight)
            self._record_pending()
        else:
            self._update_where_the_weights_are(weight)
        self.num_updates += 1

    def _update_where_the_weights_are(self, weight):
        for key, p in self.params.items():
            shadow, weights = self.shadow[key], p.detach()
            if shadow.dtype == torch.float32 and shadow.device == weights.device:
                shadow.lerp_(weights.to(torch.float32), weight)
                continue
            # Copied up (if offloaded), averaged in fp32 and rounded beside the weights, copied back.
            average = shadow.to(weights.device, non_blocking=True).to(torch.float32).lerp_(
                weights.to(torch.float32), weight)
            if shadow.dtype == torch.bfloat16:
                average = to_bfloat16_stochastic(average)
            shadow.copy_(average, non_blocking=True)

    def _update_in_background(self, weight):
        # Stream-ordered copies: they read this step's weights whatever the host does next.
        for key, p in self.params.items():
            self._staging[key].copy_(p.detach(), non_blocking=True)
        copied = None
        if torch.cuda.is_available() and any(p.is_cuda for p in self.params.values()):
            copied = torch.cuda.Event()
            copied.record()

        def average():
            try:
                if copied is not None:
                    copied.synchronize()
                with torch.no_grad():
                    for key, weights in self._staging.items():
                        self.shadow[key].lerp_(weights.to(torch.float32), weight)
            except BaseException as e:  # re-raised by wait() on the training thread
                self._error = e

        self._worker = threading.Thread(target=average, name='ema-update', daemon=True)
        self._worker.start()

    def averaged(self, key, param):
        """The EMA of `param`, in its own dtype on the CPU: the shadow, or the weight itself
        for a parameter that is not trained (or before start_step)."""
        self.wait()
        shadow = None if self.following else self.shadow.get(key)
        source = param.detach() if shadow is None else shadow
        return source.to(device='cpu', dtype=param.dtype)

    def state_dict(self):
        self.wait()
        return {'shadow': {k: v.cpu() for k, v in self.shadow.items()}, 'num_updates': self.num_updates,
                'decay': self.decay, 'following': self.following}

    def load_state_dict(self, state):
        """Restore the shadows found in `state`. Returns the keys it had none for, which keep
        their current value (the weights, when EMA was switched on after the checkpoint)."""
        self.wait()
        saved = state['shadow']
        missing = []
        for key, shadow in self.shadow.items():
            if key in saved:
                # Across an ema_dtype change too: fp32 -> bf16 rounds to nearest, once.
                shadow.copy_(saved[key])
            else:
                missing.append(key)
        self.num_updates = int(state.get('num_updates', 0))
        self.following = bool(state.get('following', False))
        return missing


def checkpoint_file(save_root, tag, stage_id):
    return Path(save_root) / tag / f'ema_stage_{stage_id:02d}.pt'


def save_ema_checkpoint(ema, save_root, tag, stage_id):
    path = checkpoint_file(save_root, tag, stage_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_save_atomic(ema.state_dict(), path)


def load_ema_checkpoint(ema, save_root, tag):
    """Load this rank's shadows from every stage file of checkpoint `tag`, by name, so a run
    may resume with a different pipeline split. Returns (files found, keys not found)."""
    files = sorted((Path(save_root) / tag).glob('ema_stage_*.pt'))
    if not files:
        return 0, list(ema.shadow)
    merged = {'shadow': {}, 'num_updates': 0, 'following': False}
    for path in files:
        # mmap: a stage only reads the tensors it holds, not every stage's average.
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        merged['shadow'].update(state['shadow'])
        merged['num_updates'] = max(merged['num_updates'], int(state.get('num_updates', 0)))
        merged['following'] = merged['following'] or bool(state.get('following', False))
    return len(files), ema.load_state_dict(merged)
