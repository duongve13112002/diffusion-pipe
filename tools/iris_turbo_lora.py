"""Iris-3B Turbo as a LoRA: few-step distillation of an Iris model on one GPU or several.

    torchrun --standalone --nproc_per_node=1 tools/iris_turbo_lora.py --config examples/iris/turbo/lora.toml
    torchrun --standalone --nproc_per_node=8 tools/iris_turbo_lora.py --config examples/iris/turbo/lora.toml
    python tools/iris_turbo_lora.py --config examples/iris/turbo/lora.toml     # one GPU, no torchrun

The objective is Iris's own TDM distillation (iris3b.distill.objective and sampler, the code
behind the released Iris-3B Turbo), but the student and the fake score network are two LoRA
adapters on ONE frozen copy of the teacher, instead of two more full copies of the model:

    adapters off      -> the teacher (the model you distill, frozen)
    'student' adapter -> the few-step student, trained against the teacher's CFG target
    'fake' adapter    -> the fake score network, trained online on the student's samples
    'ema' adapter     -> the student's exponential moving average, used for previews and saves

So a run needs the base weights once (bf16 by default, 6 GB), the text encoder and the small
adapters with their optimizer states: an estimated 24-28 GB of GPU memory, or 16-20 GB with
text_encoder = 'offload' (docs/iris_turbo.md). With several GPUs every
rank holds the whole model and the adapters' gradients are averaged across ranks after each
update, which multiplies the batch (data parallel). Teacher and fake share the same frozen base,
so the bf16 base cannot bias their difference.

Output, in output_dir:
    adapters/step_XXXXXXXX/   the student in diffusion-pipe's adapter format
                              (adapter_model.safetensors + adapter_config.json) plus distill.yaml,
                              the few-step schedule. Merge it into the teacher with
                              tools/iris_merge_adapter.py and sample the result with Iris's
                              scripts/sample.py, which reads distill.yaml.
    checkpoints/step_XXXXXXXX.pt  training state (adapters, EMA, optimizers, prompt cursor, RNG),
                              written atomically. A restart with the same command continues from
                              the newest readable one; a finished run exits at once.
    samples/step_XXXXXXXX/    previews of sample_prompts with the EMA student, if any are set.

Every setting is in the TOML config (see examples/iris/turbo/lora.toml); --set key=value
overrides one, e.g. --set output_dir=/pvc/turbo_lora --set max_steps=3000.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from pathlib import Path

import toml
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'submodules' / 'iris-3b' / 'src'))
sys.path.insert(0, str(ROOT))

FORMAT_VERSION = 1
CHECKPOINT_RE = re.compile(r'^step_(\d{8})\.pt$')
DEFAULT_RESOLUTIONS = [[1024, 1024], [752, 1344], [896, 1344], [896, 1200], [1200, 896], [1344, 896], [1344, 752]]
# Every linear layer except the text adapter (y_embedder): the trunk blocks, the pixel stage, the
# embedders, the shared modulation and the final layer. Entries are module class names or
# top-level IrisDiT attribute names.
DEFAULT_TARGETS = ['s_embedder', 't_embedder', 'modulation_cores', 'MMDiTBlock', 'SingleStreamBlock',
                   'pixel_embedder', 'PiTBlock', 'final_layer']
# Modules diffusion-pipe keeps in high precision when the transformer is cast lower (models/iris.py).
KEEP_IN_HIGH_PRECISION = ['s_embedder', 't_embedder', 'y_embedder', 'y_pos_embedding', 'modulation_cores',
                          'pixel_embedder', 'pixel_blocks', 'final_layer', 'repa']
DTYPES = {'bfloat16': torch.bfloat16, 'float32': torch.float32, 'float16': torch.float16}


class ConfigError(ValueError):
    pass


# ---- config -------------------------------------------------------------------------------------

@dataclass
class AdapterConfig:
    rank: int = 64
    alpha: float = 64.0
    dropout: float = 0.0
    fake_rank: int = 64       # the fake network's adapter
    fake_alpha: float = 64.0
    target_modules: list = field(default_factory=lambda: list(DEFAULT_TARGETS))


@dataclass
class LoraDistillConfig:
    teacher: str = ''          # exported Iris directory (model.safetensors + config.yaml) or training .pth
    prompts_file: str = ''     # UTF-8, one prompt per line (tools/iris_turbo_prompts.py)
    output_dir: str = ''
    # the student's schedule: noise levels its steps start at, from 1 (Iris's released 4 steps)
    sigmas: list = field(default_factory=lambda: [1.0, 0.938, 0.833, 0.625])
    cfg_scale: float = 3.5     # teacher guidance distilled into the student
    sigma_min: float = 0.01
    resolutions: list = field(default_factory=lambda: copy.deepcopy(DEFAULT_RESOLUTIONS))
    batch_size: int = 1        # per GPU and micro-batch
    gradient_accumulation_steps: int = 1
    max_steps: int = 2000      # student updates
    fake_updates: int = 2      # fake updates after every student update
    learning_rate: float = 2e-5
    fake_learning_rate: float = 1e-4
    betas: list = field(default_factory=lambda: [0.0, 0.999])
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ema_decay: float = 0.99    # per student update; 0 turns the EMA off
    seed: int = 0
    base_dtype: str = 'bfloat16'           # the frozen base; the adapters are always float32
    activation_checkpointing: str = 'full'  # none | full | selective_op | selective_layer
    text_encoder: str = 'gpu'  # gpu: resident; offload: kept in CPU memory, moved to the GPU to encode
    save_every: int = 100
    keep_last: int = 2         # training checkpoints kept; the exported adapters are all kept
    save_dtype: str = 'bfloat16'
    log_every: int = 10
    sample_every: int = 100
    sample_prompts: list = field(default_factory=list)
    sample_size: list = field(default_factory=lambda: [1024, 1024])
    sample_seed: int = 0
    init_from: str = ''        # warm start: a checkpoint of another LoRA run (its EMA student and fake)
    nccl_timeout_minutes: int = 60
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    def validate(self):
        problems = []
        for name in ('teacher', 'prompts_file', 'output_dir'):
            if not getattr(self, name):
                problems.append(f'{name} is required')
        s = self.sigmas
        if not s or s[0] != 1.0 or any(not 0 < x <= 1 for x in s) or any(a <= b for a, b in zip(s, s[1:])):
            problems.append('sigmas must start at 1.0 and decrease strictly inside (0, 1]')
        elif not 0 < self.sigma_min < s[-1]:
            problems.append('sigma_min must be in (0, last sigma)')
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 1:
            problems.append('cfg_scale must be >= 1')
        if not self.resolutions or any(len(r) != 2 or min(r) < 16 for r in self.resolutions):
            problems.append('resolutions must be a list of [height, width]')
        for name in ('batch_size', 'gradient_accumulation_steps', 'max_steps', 'save_every', 'log_every'):
            if getattr(self, name) < 1:
                problems.append(f'{name} must be >= 1')
        for name in ('fake_updates', 'sample_every', 'keep_last'):
            if getattr(self, name) < 0:
                problems.append(f'{name} must be >= 0')
        if self.fake_updates < 1:
            problems.append('fake_updates must be >= 1: the fake network has to follow the student')
        if not 0 <= self.ema_decay < 1:
            problems.append('ema_decay must be in [0, 1)')
        if self.keep_last == 1:
            problems.append('keep_last = 1 leaves nothing to resume from if a crash tears the newest checkpoint; '
                            'use 2 or more (0 keeps all)')
        if self.base_dtype not in DTYPES or self.save_dtype not in DTYPES:
            problems.append(f'base_dtype and save_dtype must be one of {sorted(DTYPES)}')
        if self.text_encoder not in ('gpu', 'offload'):
            problems.append("text_encoder must be 'gpu' or 'offload'")
        if self.activation_checkpointing not in ('none', 'full', 'selective_op', 'selective_layer'):
            problems.append('activation_checkpointing must be none, full, selective_op or selective_layer')
        if len(self.sample_size) != 2:
            problems.append('sample_size must be [height, width]')
        a = self.adapter
        if a.rank < 1 or a.fake_rank < 1:
            problems.append('adapter ranks must be >= 1')
        if not 0 <= a.dropout < 1:
            problems.append('adapter.dropout must be in [0, 1)')
        if not a.target_modules:
            problems.append('adapter.target_modules is empty')
        if problems:
            raise ConfigError('invalid config:\n  ' + '\n  '.join(problems))


# Settings a resume may change: where things are, how often to save, log and preview, the step
# sizes (written into the restored optimizers), the run length and memory placement.
RESUME_MUTABLE = {'teacher', 'prompts_file', 'output_dir', 'max_steps', 'save_every', 'keep_last', 'log_every',
                  'sample_every', 'sample_prompts', 'sample_size', 'sample_seed', 'learning_rate',
                  'fake_learning_rate', 'max_grad_norm', 'init_from', 'text_encoder', 'activation_checkpointing',
                  'save_dtype', 'nccl_timeout_minutes'}


def contract(cfg_dict):
    return {k: v for k, v in cfg_dict.items() if k not in RESUME_MUTABLE}


def parse_value(text):
    try:
        return toml.loads(f'v = {text}')['v']
    except toml.TomlDecodeError:
        return text


def load_config(path, overrides=()):
    raw = toml.load(str(path))
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ConfigError(f'--set expects key=value, got {item!r}')
        target = raw
        *parents, leaf = key.strip().split('.')
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = parse_value(value.strip())
    adapter = raw.pop('adapter', {}) or {}
    known = {f.name for f in fields(LoraDistillConfig)}
    unknown = sorted(set(raw) - known)
    adapter_known = {f.name for f in fields(AdapterConfig)}
    unknown += [f'adapter.{k}' for k in sorted(set(adapter) - adapter_known)]
    if unknown:
        raise ConfigError(f'unknown config keys: {", ".join(unknown)}')
    cfg = LoraDistillConfig(**raw, adapter=AdapterConfig(**adapter))
    cfg.validate()
    return cfg


# ---- distributed helpers ------------------------------------------------------------------------

def rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def log(message):
    if rank() == 0:
        print(f'[iris_turbo_lora] {message}', flush=True)


def barrier():
    if world_size() > 1:
        torch.distributed.barrier()


def broadcast_object(value):
    if world_size() == 1:
        return value
    box = [value]
    torch.distributed.broadcast_object_list(box, src=0)
    return box[0]


def all_ok(error):
    """Raise on every rank if any rank failed, before anyone enters another collective."""
    if world_size() == 1:
        if error is not None:
            raise error
        return
    errors = [None] * world_size()
    torch.distributed.all_gather_object(errors, None if error is None else f'{type(error).__name__}: {error}')
    failed = [f'rank {i}: {e}' for i, e in enumerate(errors) if e is not None]
    if failed:
        raise RuntimeError('; '.join(failed))


# ---- LoRA ---------------------------------------------------------------------------------------

_ACTIVE = {'name': None}


@contextmanager
def adapter(name):
    """Run the model as `name` ('student', 'fake', 'ema') or, with None, as the frozen teacher.

    Activation checkpointing recomputes forwards during backward, so a backward must run under
    the same adapter as the forward that built its graph.
    """
    previous = _ACTIVE['name']
    _ACTIVE['name'] = name
    try:
        yield
    finally:
        _ACTIVE['name'] = previous


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus named low-rank adapters; the active one (if any) is added.

    Parameter names follow PEFT (`<path>.lora_A.<adapter>.weight`), so an adapter saves under
    the keys diffusion-pipe and tools/iris_merge_adapter.py use.
    """

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.scaling = {}
        # Iris reads Linear attributes (e.g. a patch embedder's out_features), as PEFT allows
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def add(self, name, rank, alpha, dropout=0.0):
        device = self.base_layer.weight.device
        a = nn.Linear(self.base_layer.in_features, rank, bias=False, device=device, dtype=torch.float32)
        b = nn.Linear(rank, self.base_layer.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))  # as PEFT
        nn.init.zeros_(b.weight)
        self.lora_A[name], self.lora_B[name] = a, b
        self.lora_dropout[name] = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling[name] = alpha / rank

    def forward(self, x):
        out = self.base_layer(x)
        name = _ACTIVE['name']
        if name is None or name not in self.lora_A:
            return out
        a = self.lora_A[name]
        delta = self.lora_B[name](a(self.lora_dropout[name](x.to(a.weight.dtype))))
        return out + (delta * self.scaling[name]).to(out.dtype)


def target_paths(model, targets):
    """Paths of the nn.Linear layers inside modules whose class or top-level name is in targets."""
    paths = set()
    for name, module in model.named_modules():
        top = name.split('.')[0]
        if type(module).__name__ in targets or (name == top and name in targets):
            for sub_name, sub in module.named_modules(prefix=name):
                if isinstance(sub, nn.Linear):
                    paths.add(sub_name)
    unknown = [t for t in targets if t not in {type(m).__name__ for m in model.modules()}
               and t not in dict(model.named_children())]
    if unknown:
        raise ConfigError(f'adapter.target_modules {unknown} match nothing in this Iris model')
    if any(p.split('.')[0] == 'y_embedder' for p in paths):
        log('note: the text adapter (y_embedder) is targeted')
    return sorted(paths)


def inject(model, paths):
    layers = {}
    for path in paths:
        parent_name, _, child = path.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = LoRALinear(getattr(parent, child))
        setattr(parent, child, layer)
        layers[path] = layer
    return layers


def adapter_params(layers, name):
    params = []
    for layer in layers.values():
        params += [layer.lora_A[name].weight, layer.lora_B[name].weight]
    return params


def adapter_state(layers, name, dtype=None):
    """{PEFT key without the adapter name: tensor}, e.g. 'blocks.0.attn.qkv.lora_A.weight'."""
    state = {}
    for path, layer in layers.items():
        for part in ('lora_A', 'lora_B'):
            tensor = getattr(layer, part)[name].weight.detach()
            state[f'{path}.{part}.weight'] = tensor.to(dtype) if dtype else tensor.clone()
    return state


def load_adapter_state(layers, name, state):
    expected = {f'{path}.{part}.weight' for path in layers for part in ('lora_A', 'lora_B')}
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ConfigError(f'adapter {name!r} does not match this model and targets '
                          f'(missing {missing[:3]}, unexpected {extra[:3]})')
    with torch.no_grad():
        for path, layer in layers.items():
            for part in ('lora_A', 'lora_B'):
                param = getattr(layer, part)[name].weight
                tensor = state[f'{path}.{part}.weight']
                if tensor.shape != param.shape:
                    raise ConfigError(f'adapter {name!r}: {path}.{part} has shape {tuple(tensor.shape)}, '
                                      f'this run {tuple(param.shape)} (a different rank?)')
                param.copy_(tensor.to(param.device, param.dtype))


# ---- teacher ------------------------------------------------------------------------------------

def resolve_teacher(teacher):
    """A local teacher path; a Hugging Face repo id is downloaded (model.safetensors + config.yaml)."""
    path = Path(teacher)
    if path.exists():
        return path
    if re.fullmatch(r'[\w.-]+/[\w.-]+', teacher):
        from huggingface_hub import hf_hub_download
        if rank() == 0:
            for name in ('config.yaml', 'model.safetensors'):
                hf_hub_download(teacher, name)
        barrier()
        return Path(hf_hub_download(teacher, 'model.safetensors')).parent
    raise ConfigError(f'teacher {teacher} is neither a local path nor a Hugging Face repo id')


def file_digest(path, chunk=64 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_teacher(teacher, device, base_dtype):
    """(IrisDiT on device with the teacher's weights, inference config, raw config, identity)."""
    from omegaconf import OmegaConf
    from iris3b.config import INFERENCE_SECTIONS, inference_config
    from iris3b.models.dit import IrisDiT
    from tools.iris_turbo_distill import PreflightError, check_teacher

    try:
        raw, _ = check_teacher(str(teacher))
    except PreflightError as error:
        raise ConfigError(str(error)) from None
    base = inference_config(raw)
    with torch.device('meta'):
        model = IrisDiT(base.model)
    high = tuple(KEEP_IN_HIGH_PRECISION)

    def dtype_for(key):
        return torch.float32 if key.split('.')[0] in high or base_dtype == torch.float32 else base_dtype

    teacher = Path(teacher)
    if teacher.is_dir():
        # Streamed tensor by tensor onto the device: every rank loading the whole file into host
        # memory first would need 12 GB of RAM per GPU.
        from safetensors import safe_open
        state = {}
        with safe_open(str(teacher / 'model.safetensors'), framework='pt', device='cpu') as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(device=device, dtype=dtype_for(key))
        identity = file_digest(teacher / 'model.safetensors') if rank() == 0 else None
    else:
        from iris3b.distill.trainer import tensor_content_hash
        from iris3b.sampling import load_for_inference
        _, weights = load_for_inference(teacher)
        identity = tensor_content_hash(weights) if rank() == 0 else None
        state = {k: v.to(device=device, dtype=dtype_for(k)) for k, v in weights.items()}
        del weights
    state = {k: v for k, v in state.items() if not k.startswith('repa.')}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    for buffer_name, buffer in list(model.named_buffers()):
        if buffer.device.type == 'meta':
            raise ConfigError(f'teacher buffer {buffer_name} was not initialised')
    model.requires_grad_(False)
    base_config = {name: asdict(getattr(base, name)) for name in INFERENCE_SECTIONS}
    identity = broadcast_object(identity)
    identity = {'weights_sha256': identity,
                'config_sha256': hashlib.sha256(json.dumps(base_config, sort_keys=True).encode()).hexdigest()}
    return model, base, OmegaConf.to_container(OmegaConf.create(raw)), identity


# ---- checkpoints --------------------------------------------------------------------------------

def checkpoint_files(directory):
    found = []
    for path in Path(directory).glob('step_*.pt'):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def read_checkpoint(path):
    return torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)


def newest_checkpoint(directory):
    """(step, path, payload) of the newest checkpoint that loads; unreadable ones become *.torn."""
    for step, path in reversed(checkpoint_files(directory)):
        try:
            payload = read_checkpoint(path)
            if payload.get('format_version') != FORMAT_VERSION or 'student' not in payload:
                raise ValueError('not a checkpoint of this tool')
            return step, path, payload
        except Exception as error:
            torn = path.with_name(path.name + '.torn')
            log(f'WARNING: {path} is unreadable ({type(error).__name__}: {error}), probably cut off by a crash; '
                f'renamed to {torn.name}, resuming from the one before')
            os.replace(path, torn)
    return None


def fsync_dir(directory):
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def save_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{uuid.uuid4().hex}')
    try:
        with open(tmp, 'wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def remove_temporaries(directory):
    for path in Path(directory).glob('.*.tmp-*'):
        if path.is_dir():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def prune(directory, keep_last):
    if keep_last <= 0:
        return
    for _, path in checkpoint_files(directory)[:-keep_last]:
        try:
            path.unlink()
        except OSError as error:
            log(f'WARNING: could not delete old checkpoint {path}: {error}')


def check_resume(payload, cfg, world, prompts_sha):
    """What a resume needs besides the teacher: same GPU count, prompts and training settings."""
    if payload['world_size'] != world:
        raise ConfigError(f'the checkpoint was written by {payload["world_size"]} processes, this run has '
                          f'{world}; resume with the same GPU count')
    if payload['prompt_state']['sha256'] != prompts_sha:
        raise ConfigError('prompts_file content changed since this run started; restore it or start a new '
                          'output_dir')
    old, new = contract(payload['config']), contract(asdict(cfg))
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    if changed:
        raise ConfigError(f'settings changed since this run started: {", ".join(changed)}. Only '
                          f'{", ".join(sorted(RESUME_MUTABLE))} may change on a resume; use a new output_dir.')


# ---- trainer ------------------------------------------------------------------------------------

class LoraDistiller:
    def __init__(self, cfg, *, model, base, raw_config, identity, layers, text_encoder, prompts, device):
        self.cfg = cfg
        self.model, self.base, self.raw_config, self.identity = model, base, raw_config, identity
        self.layers, self.text_encoder, self.prompts, self.device = layers, text_encoder, prompts, device
        self.rank, self.world = rank(), world_size()
        self.prediction = base.flow.prediction
        self.num_train_timesteps = base.flow.num_train_timesteps
        self.in_channels = base.model.in_channels
        self.autocast_device = device.type
        self.student = adapter_params(layers, 'student')
        self.fake = adapter_params(layers, 'fake')
        self.ema = adapter_params(layers, 'ema') if cfg.ema_decay > 0 else None
        kwargs = dict(betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(self.student, lr=cfg.learning_rate, **kwargs)
        self.fake_optimizer = torch.optim.AdamW(self.fake, lr=cfg.fake_learning_rate, **kwargs)
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed * 1000003 + self.rank)
        self.step = 0
        self.fake_steps = 0
        null = text_encoder.null('')
        self.null = (null.embeddings.to(device=device, dtype=torch.bfloat16), null.mask.to(device))
        self.output = Path(cfg.output_dir)
        self.sample_conditioning = None

    # -- text --
    def encode(self, prompts):
        """Prompt encodings kept in bf16 (the encoder's own dtype) until a micro-batch uses them."""
        if self.cfg.text_encoder == 'offload':
            self.text_encoder.to(self.device)
        try:
            out = []
            for start in range(0, len(prompts), 16):
                encoding = self.text_encoder.encode(prompts[start:start + 16])
                out.append((encoding.embeddings.to(self.device, torch.bfloat16), encoding.mask.to(self.device)))
        finally:
            if self.cfg.text_encoder == 'offload':
                self.text_encoder.to('cpu')
        return torch.cat([e for e, _ in out]), torch.cat([m for _, m in out])

    def conditioning(self, embeddings, mask):
        n = embeddings.shape[0]
        null_embeddings, null_mask = self.null
        return {
            'cond': (embeddings.float(), mask),
            'uncond': (null_embeddings.float().expand(n, *([-1] * (null_embeddings.ndim - 1))),
                       null_mask.expand(n, -1)),
        }

    # -- model calls --
    def model_fn(self, conditioning, name, *, uncond=False, train=False):
        embeddings, mask = conditioning['uncond' if uncond else 'cond']

        def call(x, t_model):
            self.model.train(train)
            with adapter(name), torch.autocast(device_type=self.autocast_device, dtype=torch.bfloat16):
                output = self.model(x, t_model, embeddings, y_mask=mask).x
            return output.float()
        return call

    def draw(self):
        """(step index, bucket) for one micro-batch; rank 0's draw, so every rank runs one graph depth."""
        choice = torch.stack([
            torch.randint(len(self.cfg.sigmas), (), device=self.device, generator=self.generator),
            torch.randint(len(self.cfg.resolutions), (), device=self.device, generator=self.generator),
        ])
        if self.world > 1:
            torch.distributed.broadcast(choice, src=0)
        return (int(v) for v in choice.tolist())

    def generate(self, conditioning, *, differentiable):
        from iris3b.distill.sampler import backward_simulate
        sigmas = self.cfg.sigmas
        step, bucket = self.draw()
        height, width = self.cfg.resolutions[bucket]
        noise = torch.randn(conditioning['cond'][0].shape[0], self.in_channels, height, width,
                            device=self.device, dtype=torch.float32, generator=self.generator)
        fn = self.model_fn(conditioning, 'student', train=differentiable)
        with torch.enable_grad() if differentiable else torch.no_grad():
            generated, sigma, state = backward_simulate(
                fn, noise, sigmas, step_index=step, prediction=self.prediction,
                num_train_timesteps=self.num_train_timesteps)
        return generated, sigma, state, sigmas[step + 1] if step + 1 < len(sigmas) else 0.0

    # -- updates --
    def require_finite(self, value, label):
        finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32, device=self.device)
        if self.world > 1:
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f'non-finite {label} on at least one rank; no update was applied. Lower the '
                                     'learning rates or resume from an earlier checkpoint.')

    def apply(self, params, optimizer):
        """Average gradients over ranks, clip, step. Returns the gradient norm."""
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        if self.world > 1:
            flat = torch.cat([g.reshape(-1) for g in grads])
            torch.distributed.all_reduce(flat)
            flat /= self.world
            offset = 0
            for p, g in zip(params, grads):
                p.grad = flat[offset:offset + g.numel()].view_as(g)
                offset += g.numel()
        else:
            for p, g in zip(params, grads):
                p.grad = g
        norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0
                                              else float('inf'))
        self.require_finite(norm, 'gradient norm')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return norm.detach()

    def student_update(self, batches):
        from iris3b.distill.objective import tdm_generator_loss
        accum = len(batches)
        totals = {}
        for conditioning in batches:
            generated, sigma, state, sigma_next = self.generate(conditioning, differentiable=True)
            result = tdm_generator_loss(
                generated, state, sigma, sigma_next,
                self.model_fn(conditioning, None),
                self.model_fn(conditioning, None, uncond=True),
                self.model_fn(conditioning, 'fake'),
                self._dmd(), prediction=self.prediction, num_train_timesteps=self.num_train_timesteps,
                generator=self.generator)
            self.require_finite(result.loss, 'student loss')
            self.model.train(True)
            with adapter('student'):
                (result.loss / accum).backward()
            for key, value in {'generator_loss': result.loss, **result.metrics}.items():
                totals[key] = totals.get(key, 0.0) + value.detach() / accum
        totals['generator_grad_norm'] = self.apply(self.student, self.optimizer)
        if self.ema is not None:
            with torch.no_grad():
                torch._foreach_lerp_(self.ema, self.student, 1.0 - self.cfg.ema_decay)
        self.step += 1
        return totals

    def fake_update(self, batches):
        from iris3b.distill.objective import tdm_fake_loss
        accum = len(batches)
        total = 0.0
        for conditioning in batches:
            generated, sigma, state, sigma_next = self.generate(conditioning, differentiable=False)
            loss = tdm_fake_loss(
                generated, state, sigma, sigma_next, self.model_fn(conditioning, 'fake', train=True),
                self._dmd(), prediction=self.prediction, num_train_timesteps=self.num_train_timesteps,
                generator=self.generator)
            self.require_finite(loss, 'fake loss')
            self.model.train(True)
            with adapter('fake'):
                (loss / accum).backward()
            total = total + loss.detach() / accum
        norm = self.apply(self.fake, self.fake_optimizer)
        self.fake_steps += 1
        return {'fake_loss': total, 'fake_grad_norm': norm}

    def _dmd(self):
        from iris3b.distill.config import DMDConfig
        return DMDConfig(sigmas=list(self.cfg.sigmas), cfg_scale=self.cfg.cfg_scale, sigma_min=self.cfg.sigma_min)

    def cycle(self):
        """One student update and its fake updates, each over gradient_accumulation_steps micro-batches,
        each micro-batch with fresh prompts (Iris's order: student first)."""
        cfg = self.cfg
        accum, n = cfg.gradient_accumulation_steps, cfg.batch_size
        groups = 1 + cfg.fake_updates
        prompts = [self.prompts.take(n, self.rank, self.world) for _ in range(groups * accum)]
        if cfg.text_encoder == 'offload':
            embeddings, mask = self.encode([p for batch in prompts for p in batch])
            encoded = [(embeddings[i * n:(i + 1) * n], mask[i * n:(i + 1) * n]) for i in range(len(prompts))]
        else:
            encoded = None

        def batch(i):
            e, m = encoded[i] if encoded is not None else self.encode(prompts[i])
            return self.conditioning(e, m)

        metrics = self.student_update([batch(i) for i in range(accum)])
        fake_rows = [self.fake_update([batch(accum * (1 + u) + i) for i in range(accum)])
                     for u in range(cfg.fake_updates)]
        for key in fake_rows[0]:
            metrics[key] = torch.stack([torch.as_tensor(row[key]) for row in fake_rows]).mean()
        return metrics

    # -- previews --
    @torch.no_grad()
    def preview(self, step):
        cfg = self.cfg
        if not cfg.sample_prompts:
            return
        if self.sample_conditioning is None:
            self.sample_conditioning = self.encode(list(cfg.sample_prompts))
        embeddings, mask = self.sample_conditioning
        name = 'ema' if self.ema is not None else 'student'
        height, width = cfg.sample_size
        out_dir = self.output / 'samples' / f'step_{step:08d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for index in range(self.rank, len(cfg.sample_prompts), self.world):
            g = torch.Generator(device=self.device).manual_seed(cfg.sample_seed + index)
            x = torch.randn(1, self.in_channels, height, width, device=self.device, generator=g)
            conditioning = self.conditioning(embeddings[index:index + 1], mask[index:index + 1])
            fn = self.model_fn(conditioning, name)
            sigma = torch.full((1,), cfg.sigmas[0], device=self.device)
            from iris3b.distill.objective import predict_x0
            for i in range(len(cfg.sigmas)):
                clean = predict_x0(fn, x, sigma, self.prediction, self.num_train_timesteps)
                if i + 1 < len(cfg.sigmas):
                    s = sigma.view(-1, 1, 1, 1)
                    implied = (x - (1 - s) * clean) / s
                    sigma = torch.full_like(sigma, cfg.sigmas[i + 1])
                    x = (1 - sigma.view(-1, 1, 1, 1)) * clean + sigma.view(-1, 1, 1, 1) * implied
            image = ((clean[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(image).save(out_dir / f'{index:03d}.png')
        if self.rank == 0:
            (out_dir / 'prompts.txt').write_text('\n'.join(f'{i:03d}: {p}' for i, p in enumerate(cfg.sample_prompts))
                                                 + '\n', encoding='utf-8')
        barrier()

    # -- saving --
    def export_adapter(self, step):
        """The (EMA) student in diffusion-pipe's adapter format, plus distill.yaml, written atomically."""
        if self.rank != 0:
            return
        import peft
        import safetensors.torch
        from omegaconf import OmegaConf
        import shutil
        cfg = self.cfg
        final = self.output / 'adapters' / f'step_{step:08d}'
        tmp = final.with_name(f'.{final.name}.tmp-{uuid.uuid4().hex}')
        tmp.mkdir(parents=True)
        try:
            name = 'ema' if self.ema is not None else 'student'
            state = adapter_state(self.layers, name, DTYPES[cfg.save_dtype])
            state = {'diffusion_model.' + k: v.contiguous().cpu() for k, v in state.items()}
            safetensors.torch.save_file(state, str(tmp / 'adapter_model.safetensors'), metadata={'format': 'pt'})
            peft.LoraConfig(r=cfg.adapter.rank, lora_alpha=cfg.adapter.alpha, lora_dropout=0.0, bias='none',
                            target_modules=sorted(self.layers)).save_pretrained(str(tmp))
            height, width = cfg.resolutions[0]
            OmegaConf.save(OmegaConf.create({
                'dmd': {'sigmas': list(cfg.sigmas), 'cfg_scale': cfg.cfg_scale, 'sigma_min': cfg.sigma_min},
                'height': height, 'width': width,
                'lora_teacher': str(cfg.teacher), 'lora_teacher_weights_sha256': self.identity['weights_sha256'],
                'lora_step': step, 'lora_weights': name,
            }), str(tmp / 'distill.yaml'))
            for path in tmp.iterdir():
                with open(path, 'rb') as f:
                    os.fsync(f.fileno())
            if final.is_dir():
                # left by an attempt that crashed before its checkpoint; this run's state replaces it
                shutil.rmtree(final)
            os.replace(tmp, final)
            fsync_dir(final.parent)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def save(self):
        """The exported adapter first, then the training checkpoint: a checkpoint never exists
        without its adapter, so a restart never has to re-export one."""
        step = self.step
        self.export_adapter(step)
        generators = [None] * self.world
        state = self.generator.get_state()
        if self.world > 1:
            torch.distributed.all_gather_object(generators, state.cpu())
        else:
            generators = [state.cpu()]
        error = None
        if self.rank == 0:
            try:
                payload = {
                    'format_version': FORMAT_VERSION,
                    'step': step, 'fake_steps': self.fake_steps,
                    'config': asdict(self.cfg), 'world_size': self.world,
                    'teacher_identity': self.identity, 'prompt_state': self.prompts.state_dict(),
                    'student': adapter_state(self.layers, 'student'),
                    'fake': adapter_state(self.layers, 'fake'),
                    'ema': adapter_state(self.layers, 'ema') if self.ema is not None else None,
                    'optimizer': self.optimizer.state_dict(), 'fake_optimizer': self.fake_optimizer.state_dict(),
                    'generators': generators,
                }
                checkpoints = self.output / 'checkpoints'
                save_atomic(payload, checkpoints / f'step_{step:08d}.pt')
                prune(checkpoints, self.cfg.keep_last)
            except Exception as exc:
                error = exc
        all_ok(error)
        log(f'saved step {step}: adapters/step_{step:08d}')

    def load(self, payload):
        check_resume(payload, self.cfg, self.world, self.prompts.sha256)
        if payload['teacher_identity'] != self.identity:
            raise ConfigError('the teacher differs from the one this run started with (weights or config)')
        load_adapter_state(self.layers, 'student', payload['student'])
        load_adapter_state(self.layers, 'fake', payload['fake'])
        if self.ema is not None:
            load_adapter_state(self.layers, 'ema', payload['ema'])
        self.optimizer.load_state_dict(payload['optimizer'])
        self.fake_optimizer.load_state_dict(payload['fake_optimizer'])
        for optimizer, lr in ((self.optimizer, self.cfg.learning_rate), (self.fake_optimizer, self.cfg.fake_learning_rate)):
            for group in optimizer.param_groups:
                group['lr'] = lr
        self.generator.set_state(payload['generators'][self.rank].to(self.generator.get_state().device))
        self.prompts.cursor = payload['prompt_state']['cursor']
        self.step, self.fake_steps = payload['step'], payload['fake_steps']

    def warm_start(self, payload):
        """init_from: the student from another run's EMA (or raw student), the fake network as it was."""
        student = payload['ema'] if payload.get('ema') is not None else payload['student']
        load_adapter_state(self.layers, 'student', student)
        load_adapter_state(self.layers, 'fake', payload['fake'])
        if self.ema is not None:
            load_adapter_state(self.layers, 'ema', student)

    # -- loop --
    def train(self):
        cfg = self.cfg
        rows, started, start_step = [], time.time(), self.step
        while self.step < cfg.max_steps:
            metrics = self.cycle()
            names = sorted(metrics)
            values = torch.stack([torch.as_tensor(metrics[k], device=self.device).float() for k in names])
            if self.world > 1:
                torch.distributed.all_reduce(values)
                values /= self.world
            step = self.step
            if self.rank == 0 and (step % cfg.log_every == 0 or step == 1 or step == cfg.max_steps):
                elapsed = time.time() - started
                per_step = elapsed / max(step - start_step, 1)
                row = dict(zip(names, values.cpu().tolist()))
                row.update(step=step, seconds_per_step=round(per_step, 3),
                           eta_minutes=round(per_step * (cfg.max_steps - step) / 60, 1),
                           prompt_cursor=self.prompts.cursor)
                print(json.dumps(row, sort_keys=True), flush=True)
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                self.save()
            if cfg.sample_every and (step % cfg.sample_every == 0 or step == cfg.max_steps):
                self.preview(step)


# ---- setup --------------------------------------------------------------------------------------

def build_text_encoder(base, cfg, device):
    """Iris's text encoder for the teacher's text_encoder section (a test can replace this)."""
    import iris3b.text  # noqa: F401  (registers the encoders)
    from iris3b.registry import TEXT_ENCODERS
    te_cfg = copy.deepcopy(base.text_encoder)
    # the empty-prompt cache goes with the run, not into the working directory
    te_cfg.null_embed_dir = str(Path(cfg.output_dir) / 'cache')
    placement = device if cfg.text_encoder == 'gpu' else torch.device('cpu')
    encoder = None
    # rank 0 first: it downloads the encoder and writes the empty-prompt cache the others read
    for turn in (0, 1):
        if (rank() == 0) == (turn == 0):
            encoder = TEXT_ENCODERS.build(te_cfg.name, te_cfg, placement)
            if cfg.text_encoder == 'offload':
                encoder.to(device)
                encoder.null('')
                encoder.to('cpu')
            else:
                encoder.null('')
        barrier()
    return encoder


def setup_device(cfg):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if torch.cuda.is_available():
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend, timeout=timedelta(minutes=cfg.nccl_timeout_minutes))
    return device


def build(cfg, device):
    """Everything up to the first update: model and adapters, text encoder, prompts, resume state."""
    from iris3b.distill.trainer import PromptStream
    output = Path(cfg.output_dir)
    error, decision = None, None
    if rank() == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            (output / 'checkpoints').mkdir(exist_ok=True)
            for directory in (output / 'checkpoints', output / 'adapters'):
                remove_temporaries(directory)
            found = newest_checkpoint(output / 'checkpoints')
            decision = {'resume': str(found[1]) if found else None, 'step': found[0] if found else 0}
            if found:
                # before the teacher loads, so a wrong restart fails in seconds
                check_resume(found[2], cfg, world_size(), PromptStream.from_file(cfg.prompts_file).sha256)
            if found is None and cfg.init_from:
                read_checkpoint(cfg.init_from)['student']  # readable, and one of ours
        except Exception as exc:
            error = exc
    all_ok(error)
    decision = broadcast_object(decision)
    if decision['resume'] and decision['step'] >= cfg.max_steps:
        log(f'{decision["resume"]} already reached max_steps={cfg.max_steps}; nothing to do '
            f'(raise max_steps to train further). The adapter is in {output / "adapters"}.')
        return None

    teacher = resolve_teacher(cfg.teacher)
    model, base, raw, identity = load_teacher(teacher, device, DTYPES[cfg.base_dtype])
    for h, w in cfg.resolutions + [cfg.sample_size]:
        if h % base.model.patch_size or w % base.model.patch_size:
            raise ConfigError(f'every resolution must be divisible by the patch size {base.model.patch_size}, '
                              f'got {h}x{w}')
    model.activation_checkpointing = cfg.activation_checkpointing
    layers = inject(model, target_paths(model, cfg.adapter.target_modules))
    # the same seed on every rank, and rank 0's values broadcast anyway: every replica starts equal
    torch.manual_seed(cfg.seed)
    a = cfg.adapter
    for layer in layers.values():
        layer.add('student', a.rank, a.alpha, a.dropout)
        layer.add('fake', a.fake_rank, a.fake_alpha, a.dropout)
        if cfg.ema_decay > 0:
            layer.add('ema', a.rank, a.alpha)
    for layer in layers.values():
        if cfg.ema_decay > 0:
            layer.lora_A['ema'].weight.data.copy_(layer.lora_A['student'].weight.data)
            layer.lora_B['ema'].weight.data.copy_(layer.lora_B['student'].weight.data)
        for adapter_dict in (layer.lora_A, layer.lora_B):
            for module in adapter_dict.values():
                module.weight.requires_grad_(False)
    names = ['student', 'fake'] + (['ema'] if cfg.ema_decay > 0 else [])
    if world_size() > 1:
        for name in names:
            for param in adapter_params(layers, name):
                torch.distributed.broadcast(param.data, src=0)
    for param in adapter_params(layers, 'student') + adapter_params(layers, 'fake'):
        param.requires_grad_(True)
    count = sum(p.numel() for p in adapter_params(layers, 'student'))
    log(f'teacher {cfg.teacher}: {len(layers)} linear layers adapted, {count / 1e6:.1f}M parameters per adapter')

    prompts = PromptStream.from_file(cfg.prompts_file)
    text_encoder = build_text_encoder(base, cfg, device)
    trainer = LoraDistiller(cfg, model=model, base=base, raw_config=raw, identity=identity, layers=layers,
                            text_encoder=text_encoder, prompts=prompts, device=device)
    if decision['resume']:
        error = None
        try:
            trainer.load(read_checkpoint(decision['resume']))
        except Exception as exc:
            error = exc
        all_ok(error)
        log(f'resumed from {decision["resume"]} at step {trainer.step}')
    elif cfg.init_from:
        trainer.warm_start(read_checkpoint(cfg.init_from))
        log(f'warm start from {cfg.init_from}')
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='TOML config (examples/iris/turbo/lora.toml).')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a config key, e.g. --set max_steps=3000 or --set adapter.rank=32.')
    parser.add_argument('--allow-cudnn-attention', action='store_true',
                        help="Keep PyTorch's cuDNN attention kernel (its backward returned NaN on "
                             'H200 / torch 2.12 / cuDNN 9.20).')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if not args.allow_cudnn_attention and hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    device = setup_device(cfg)
    random.seed(cfg.seed + rank())
    try:
        trainer = build(cfg, device)
        if trainer is None:
            return None
        if rank() == 0:
            Path(cfg.output_dir, 'config.toml').write_text(toml.dumps(asdict(cfg)), encoding='utf-8')
            log(f'global batch {cfg.batch_size * cfg.gradient_accumulation_steps * world_size()} '
                f'({cfg.batch_size} x {cfg.gradient_accumulation_steps} accumulation x {world_size()} GPUs), '
                f'{cfg.fake_updates} fake updates per student update, {cfg.max_steps} student updates')
        if trainer.step == 0 and cfg.sample_prompts and cfg.sample_every:
            trainer.preview(0)  # the starting point: the teacher sampled in the student's few steps
        trainer.train()
        return trainer
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except ConfigError as error:
        raise SystemExit(f'[iris_turbo_lora] {error}')
