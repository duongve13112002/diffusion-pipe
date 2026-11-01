"""Resume checks on a DeepSpeed pipeline checkpoint, kept free of training-stack imports so they
can be unit-tested on CPU."""

import re
from pathlib import Path

LAYER_FILE = re.compile(r'layer_(\d+)-.*model_states\.pt$')
# One per pipeline stage: DeepSpeed's model-parallel rank is the stage of a PipelineModule. The
# optimizer state sits in it (or, with ZeRO, in zero_pp_rank_<dp>_mp_rank_<stage>_* files).
STAGE_FILE = re.compile(r'(?:zero_pp_rank_\d+_)?mp_rank_(\d+)_model_states\.pt$')


def check_checkpoint_files(run_dir, tag, pipeline_stages, loads_optimizer=True):
    """Refuse, before anything is loaded or cached, a checkpoint this run cannot resume from by its
    files alone: none of its layer files, an empty file (a save that did not finish writing), or
    other pipeline stages than `pipeline_stages`. DeepSpeed would otherwise get as far as
    load_checkpoint, after caching and loading the model, and fail there on a missing
    mp_rank_<stage> file or an optimizer state of other parameters. What needs the model's layers
    (check_checkpoint_layers) or the optimizer is checked when those exist. loads_optimizer
    False (--reset_optimizer) leaves another number of stages to DeepSpeed, as before: the
    optimizer state is what each stage's file holds that the other layout cannot use.
    """
    directory = Path(run_dir) / tag
    files = [path for path in directory.iterdir() if path.is_file()]
    if not any(LAYER_FILE.match(path.name) for path in files):
        raise RuntimeError(f'The checkpoint {directory} has no layer files (layer_<index>-*model_states.pt): '
                           'its save did not finish, or it is not a diffusion-pipe training checkpoint')
    if empty := sorted(path.name for path in files if path.name.endswith('.pt') and path.stat().st_size == 0):
        raise RuntimeError(f'The checkpoint {directory} has empty files, so its save did not finish: '
                           f'{", ".join(empty[:5])}{" ..." if len(empty) > 5 else ""}. Resume from an earlier '
                           'checkpoint (--resume_from_checkpoint <run>, after pointing its `latest` at one)')
    stages = {int(match.group(1)) for path in files if (match := STAGE_FILE.match(path.name))}
    if loads_optimizer and stages != set(range(pipeline_stages)):
        saved = f'{max(stages) + 1} pipeline stage(s)' if stages == set(range(len(stages))) and stages \
            else f'stage files {describe(stages)}'
        raise RuntimeError(f'The checkpoint {directory} was saved with {saved}; this run has pipeline_stages = '
                           f'{pipeline_stages}. Each stage keeps its own optimizer state, so a checkpoint resumes '
                           'only with the pipeline_stages it was saved with. Set it back, resume with '
                           '--reset_optimizer (a fresh optimizer state), or start a new run from its saved '
                           'model (transformer_path, or [adapter] init_from_existing).')


def check_checkpoint_layers(run_dir, layers, added_in_front=0, tag=None):
    """Refuse a checkpoint saved from another pipeline layout than `layers`; returns how many
    layers to shift its files by when loading (0, or added_in_front).

    DeepSpeed saves each layer that has a state dict as layer_<index>-*model_states.pt and loads
    each layer from the file of its index, and train.py loads non-strictly. A layout with a layer
    added or removed in front would load every layer from its neighbour's file: blocks with the
    same keys take each other's weights without an error. added_in_front: the model put that many
    layers without parameters in front of its old layout, so a checkpoint of the old layout loads
    shifted by that many (its optimizer state, which follows the parameters, is unaffected).
    Reads checkpoint `tag`, by default the one `latest` names; an OSError is left to propagate (a
    volume problem, not a mismatch).
    """
    run_dir = Path(run_dir)
    if tag is None:
        latest = run_dir / 'latest'
        if not latest.exists():
            return 0
        tag = latest.read_text().strip()
    found = set()
    for path in (run_dir / tag).iterdir():
        if match := LAYER_FILE.match(path.name):
            found.add(int(match.group(1)))
    expected = {i for i, layer in enumerate(layers) if has_file(layer)}
    if found == expected:
        return 0
    front = layers[:added_in_front]
    if (added_in_front and all(next(layer.parameters(), None) is None for layer in front)
            and found == {i - added_in_front for i in expected if i >= added_in_front}):
        return added_in_front
    names = [type(layer).__name__ for layer in layers]
    raise RuntimeError(
        f'The checkpoint {run_dir / tag} was saved from another pipeline layout: it holds layer '
        f'files {describe(found)}, this run builds layers {describe(expected)} ({names[0]} first, '
        f'{len(names)} in all). Loading it would put layers\' weights into their neighbours. Was '
        'it saved with other settings that change the layers (another number of blocks, a part '
        'switched on or off) or by other code? Restore them, or start a new run from its saved '
        'model (transformer_path, or [adapter] init_from_existing).')


def has_file(layer):
    """Whether DeepSpeed saves a file for this entry of the layer list: modules do, and a
    LayerSpec, which is built into one; plain functions and a TiedLayerSpec with a forward_fn
    (which becomes a function) do not."""
    kind = type(layer).__name__
    if kind == 'TiedLayerSpec':
        return layer.forward_fn is None
    return kind == 'LayerSpec' or hasattr(layer, 'state_dict')


def shift_checkpoint_layers(pipeline_module, shift):
    """Load layer i from the file of layer i - shift (the first layers, without parameters, from
    the first file, whose keys they ignore), until restored with the returned function."""
    original = pipeline_module.ckpt_layer_path_list

    def shifted(ckpt_dir, local_layer_idx):
        start = pipeline_module._local_start
        return original(ckpt_dir, max(local_layer_idx + start - shift, 0) - start)

    pipeline_module.ckpt_layer_path_list = shifted
    return lambda: delattr(pipeline_module, 'ckpt_layer_path_list')


def describe(indices):
    if not indices:
        return 'none'
    indices = sorted(indices)
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f'{indices[0]}-{indices[-1]}'
    return ', '.join(map(str, indices))
