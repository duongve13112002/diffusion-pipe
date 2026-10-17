"""Resume checks on a DeepSpeed pipeline checkpoint, kept free of training-stack imports so they
can be unit-tested on CPU."""

import re
from pathlib import Path

LAYER_FILE = re.compile(r'layer_(\d+)-.*model_states\.pt$')


def check_checkpoint_layers(run_dir, layers, added_in_front=0):
    """Refuse a checkpoint saved from another pipeline layout than `layers`; returns how many
    layers to shift its files by when loading (0, or added_in_front).

    DeepSpeed saves each layer that has a state dict as layer_<index>-*model_states.pt and loads
    each layer from the file of its index, and train.py loads non-strictly. A layout with a layer
    added or removed in front would load every layer from its neighbour's file: blocks with the
    same keys take each other's weights without an error. added_in_front: the model put that many
    layers without parameters in front of its old layout, so a checkpoint of the old layout loads
    shifted by that many (its optimizer state, which follows the parameters, is unaffected).
    Reads the checkpoint `latest` names; an OSError is left to propagate (a volume problem, not a
    mismatch).
    """
    run_dir = Path(run_dir)
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
