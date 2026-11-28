"""partition_method = 'memory': the pipeline stages are balanced by the bytes their layers hold, not by trainable
parameter counts.

DeepSpeed's 'parameters' counts only parameters that require grad, by number of elements. A frozen text encoder
counts 0, a frozen base under LoRA counts 0, and a block kept in fp8 weighs as much as one in bfloat16. Here a layer
weighs its parameters and buffers in bytes (numel x element size, so the dtype is what counts), plus what training
adds for its trainable parameters (gradients and optimizer states, `trainable_factor` times their bytes).

Weights that live outside the layers' parameters (a text encoder a layer holds in a list, one the data loader runs on
the first stage) are added to the first layer by the model's `first_stage_extra_bytes()`. That is known only on
the ranks that loaded it, so every rank shares its numbers and the largest is taken: all ranks then split alike.
"""
import math
import os

import torch

from utils.common import all_ok, warn_ignored

DEFAULT_TRAINABLE_FACTOR = 3.0
GIB = 1024 ** 3


def check_partition_config(config):
    """Checks the top-level partition keys of 'memory' (before the process group exists). Keys that do not apply
    (another partition_method, one pipeline stage) are ignored with a warning, not refused."""
    method = str(config.get('partition_method', 'parameters')).lower()
    active = method == 'memory' and int(config.get('pipeline_stages', 1)) > 1
    if method == 'memory' and not active:
        warn_ignored("partition_method = 'memory' with pipeline_stages = 1: there is nothing to split, ignored.")
    if active and 'partition_split' in config:
        warn_ignored("partition_split only applies with partition_method = 'manual'; ignored with 'memory'.")
    for key in ('partition_extra_first_stage_gb', 'partition_trainable_factor'):
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a number >= 0, got {value!r}')
        if not active:
            warn_ignored(f"{key} only applies with partition_method = 'memory' and pipeline_stages > 1; ignored.")


def tensors_bytes(tensors):
    seen = set()
    total = 0
    for t in tensors:
        if id(t) in seen:
            continue
        seen.add(id(t))
        # nbytes, not numel x element_size: a quantized tensor (ComfyUI's fp8 / nvfp4 weights) reports its logical
        # size through those two, and its real storage through nbytes.
        total += int(t.nbytes)
    return total


def module_weight_bytes(module):
    """Parameters and buffers of `module` in bytes, a tensor shared inside it once."""
    return tensors_bytes((*module.parameters(), *module.buffers()))


def text_encoder_bytes(text_encoder):
    """The weights of a text encoder in bytes: a module, or a lazily loaded ComfyUI one once it is loaded (its
    patcher knows its size). 0 for one that is not loaded on this rank."""
    if text_encoder is None:
        return 0
    if isinstance(text_encoder, torch.nn.Module):
        return module_weight_bytes(text_encoder)
    patcher = getattr(getattr(text_encoder, '_model', text_encoder), 'patcher', None)
    size = getattr(patcher, 'model_size', None)
    return int(size()) if callable(size) else 0


def layer_weight_bytes(layer, trainable_factor=0.0):
    """The bytes of one layer: its weights, plus `trainable_factor` times those of its trainable parameters (their
    gradients and optimizer states). Anything that is not a module (a function layer) weighs 0."""
    if not isinstance(layer, torch.nn.Module):
        return 0
    total = module_weight_bytes(layer)
    if trainable_factor:
        total += int(trainable_factor * tensors_bytes(p for p in layer.parameters() if p.requires_grad))
    return int(total)


def split_by_weights(weights, num_stages):
    """Stage boundaries (without the 0 and the end) that balance `weights` over contiguous stages."""
    from deepspeed.runtime.utils import partition_balanced
    return [int(b) for b in partition_balanced([float(w) for w in weights], num_stages)[1:-1]]


def stage_bytes(weights, split):
    bounds = [0, *[int(b) for b in split], len(weights)]
    return [int(sum(weights[bounds[i]:bounds[i + 1]])) for i in range(len(bounds) - 1)]


def teacher_split(layers, num_stages):
    """The teacher's boundaries by the bytes of its layers. Every rank holds the whole teacher, so this needs no
    exchange (and must not have one: it runs one rank at a time)."""
    return split_by_weights([layer_weight_bytes(layer) for layer in layers], num_stages)


def agree(local_weights, local_extra, error=None):
    """Every rank's (weights, extra) -> the largest of each, the same on all ranks. A failure on any rank raises
    on all of them. Must be called on every rank."""
    import torch.distributed as torch_dist
    local = None if error is not None else ([int(w) for w in local_weights], int(local_extra))
    if torch_dist.is_available() and torch_dist.is_initialized() and torch_dist.get_world_size() > 1:
        gathered = [None] * torch_dist.get_world_size()
        torch_dist.all_gather_object(gathered, (None if error is None else f'{type(error).__name__}: {error}', local))
        failed = [f'rank {i}: {e}' for i, (e, _) in enumerate(gathered) if e is not None]
        if failed:
            raise RuntimeError('; '.join(failed)) from error
        values = [v for _, v in gathered]
        count = len(local[0])
        if any(len(v[0]) != count for v in values):
            raise RuntimeError("partition_method = 'memory': the ranks built different numbers of layers")
        return [max(v[0][i] for v in values) for i in range(count)], max(v[1] for v in values)
    all_ok(error)
    return local


def loader_encoders_bytes(model, loader_encoders):
    """The text encoders the data loader runs on the first stage (utils/text_encoding.py): their bytes where they are
    loaded, and where they are not yet (a ComfyUI encoder is read only when the loader is bound, after the engine
    exists) the size of their files, which `model.text_encoder_file_bytes()` knows."""
    if not loader_encoders:
        return 0
    loaded = sum(text_encoder_bytes(te) for te in loader_encoders)
    if any(text_encoder_bytes(te) == 0 for te in loader_encoders):
        loaded = max(loaded, int(model.text_encoder_file_bytes()))
    return loaded


def text_files_bytes(paths):
    """The summed sizes of text encoder checkpoint files (a directory: its .safetensors / .bin files); 0 for
    anything that is not there."""
    total = 0
    for path in ([paths] if isinstance(paths, str) else list(paths)):
        path = os.path.expanduser(str(path))
        if os.path.isdir(path):
            total += sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files
                         if f.endswith(('.safetensors', '.bin', '.pt', '.pth')))
        elif os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def plan(layers, model, num_stages, trainable_factor, extra_first_stage_gb, loader_encoders=()):
    """(split, report) for the student's pipeline. Must be called on every rank, before the engine.
    `loader_encoders`: the text encoders the data loader runs on the first stage (none when the layers or the
    model's own hook account for them)."""
    error = None
    weights, extra = [], 0
    try:
        weights = [layer_weight_bytes(layer, trainable_factor) for layer in layers]
        extra = int(model.first_stage_extra_bytes()) if model is not None else 0
        extra += loader_encoders_bytes(model, loader_encoders)
    except Exception as e:
        error = e
    weights, extra = agree(weights, extra, error)
    extra += int(float(extra_first_stage_gb) * GIB)
    if weights:
        weights[0] += extra
    split = split_by_weights(weights, num_stages)
    return split, report(weights, split, extra)


def report(weights, split, extra):
    bounds = [0, *[int(b) for b in split], len(weights)]
    parts = []
    for i, size in enumerate(stage_bytes(weights, split)):
        parts.append(f'stage {i}: {bounds[i + 1] - bounds[i]} layers, {size / GIB:.1f} GiB')
    note = f' (first stage text encoder and extra: {extra / GIB:.1f} GiB)' if extra else ''
    return "partition_method = 'memory'" + note + ' -> ' + '; '.join(parts)


PARTS_FILE = 'partition.json'


def read_saved_parts(run_dir, tag):
    """The stage boundaries a checkpoint was saved with (its partition.json), or None for one saved before they
    were recorded or without a readable record."""
    import json
    try:
        with open(os.path.join(run_dir, tag, PARTS_FILE)) as f:
            parts = json.load(f)['parts']
        return [int(b) for b in parts]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_resumed_partition(saved, current, reset_optimizer=False):
    """The optimizer state of a checkpoint is per stage, so it fits only the stage boundaries it was saved with
    (the layers' weights are per layer and load under any split). `partition_method = 'memory'` moves them
    when the dtype, the fp8 setting, the text encoder or its keys change. Raises on a mismatch unless the
    optimizer is reset; a checkpoint from before this was recorded (`saved` None) is not checked."""
    if saved is None or current is None or reset_optimizer or [int(b) for b in saved] == [int(b) for b in current]:
        return
    raise RuntimeError(
        f'The checkpoint was saved with the pipeline stage boundaries {[int(b) for b in saved]}, this run has '
        f'{[int(b) for b in current]}: its optimizer state is per stage and would be given to the wrong layers. '
        f"Set partition_method = 'manual' and partition_split = {[int(b) for b in saved][1:-1]} to keep the old "
        'split, or pass --reset_optimizer to start the optimizer fresh.')
