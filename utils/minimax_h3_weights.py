"""MiniMax-H3-Image transformer weights, in the three layouts they are published in, as ComfyUI's
MiniMaxH3ImageModel takes them.

comfy     One .safetensors file as ComfyUI loads it (Comfy-Org/MiniMax-H3-Image): contiguous
          [q_all; k_all; v_all] qkv, fc1 = [gate; value], the six PDD heads baked into final_layer.video_out
          (rows of head 0, then each later head minus head 0) and the sigma grid in the file's metadata.
official  The transformer/ shards of MiniMaxAI/MiniMax-H3-Image: ComfyUI's key names, but the qkv rows of
          each head interleaved (head 0: q k v, head 1: q k v, ...), one base final_layer.video_out and the
          six full heads in pdd_fused_heads.safetensors at the repository root.
diffusers transformer_diffusers/: to_q / to_k / to_v, fc1 = [value; gate], the six full heads as
          pdd_proj_out.N, no rope.inv_freq (recomputed from the config's rope_theta).

load_checkpoint returns the state dict in the ComfyUI layout plus the metadata ComfyUI's loader reads.
"""

import glob
import json
import os
import re

import safetensors
import torch

import comfy.utils

LAYOUTS = ('comfy', 'official', 'diffusers')
# pdd_config.json of MiniMaxAI/MiniMax-H3-Image
DEFAULT_PDD_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
DEFAULT_ROPE_THETA = 10000.0
DEFAULT_ROPE_FREQ_DIM = 16
OFFICIAL_HEADS_FILE = 'pdd_fused_heads.safetensors'

DIFFUSERS_RENAMES = [
    (r'^transformer_blocks\.(\d+)\.', r'blocks.\1.'),
    (r'^token_refiner\.refiner_blocks\.(\d+)\.', r'token_refiner.blocks.\1.'),
    (r'\.attn\.norm_q\.', '.attn.q_norm.'),
    (r'\.attn\.norm_k\.', '.attn.k_norm.'),
    (r'\.attn\.to_out\.0\.', '.attn.out_proj.'),
    (r'\.ff\.net\.2\.', '.mlp.fc2.'),
    (r'^time_embedder\.linear_1\.', 'time_embedder.proj_in.'),
    (r'^time_embedder\.linear_2\.', 'time_embedder.proj_out.'),
    (r'^proj_in\.', 'video_patch_proj.'),
    (r'^context_embedder\.', 'condition_proj.'),
    (r'^norm_out\.norm\.', 'final_layer.norm.'),
    (r'^norm_out\.linear\.', 'final_layer.adaln_proj.linear.'),
    (r'^proj_out\.', 'final_layer.video_out.'),
]


def reorder_interleaved_qkv(weight, heads, head_dim):
    """[head0: q k v, head1: q k v, ...] rows -> [q_all; k_all; v_all]."""
    if weight.shape[0] != heads * 3 * head_dim:
        raise ValueError(f'qkv weight has {weight.shape[0]} rows, expected {heads} heads * 3 * {head_dim}')
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    return torch.cat([part.reshape(heads * head_dim, *weight.shape[1:]) for part in grouped.split(head_dim, dim=1)], dim=0)


def pdd_bank(weights, biases):
    """Full heads [n, out, in] / [n, out] -> the stacked form ComfyUI's final layer reads: head 0, then
    each later head minus head 0."""
    weights, biases = weights.float(), biases.float()
    bank_weight = torch.cat([weights[0:1], weights[1:] - weights[0:1]]).flatten(0, 1).contiguous()
    bank_bias = torch.cat([biases[0:1], biases[1:] - biases[0:1]]).flatten(0, 1).contiguous()
    return bank_weight, bank_bias


def unstack_pdd_bank(weight, bias, steps):
    """The inverse of pdd_bank: the full heads [steps, out, in] / [steps, out]."""
    rows = weight.float().reshape(steps, -1, weight.shape[1])
    brows = bias.float().reshape(steps, -1)
    return (torch.cat([rows[0:1], rows[1:] + rows[0:1]]), torch.cat([brows[0:1], brows[1:] + brows[0:1]]))


def rope_inv_freq(theta=DEFAULT_ROPE_THETA, freq_dim=DEFAULT_ROPE_FREQ_DIM):
    return 1.0 / theta ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float64) / (2 * freq_dim)).float()


def diffusers_key(key):
    for pattern, replacement in DIFFUSERS_RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def diffusers_to_comfy(sd, rope_theta=DEFAULT_ROPE_THETA, rope_freq_dim=DEFAULT_ROPE_FREQ_DIM):
    """A diffusers state dict (transformer_diffusers/) -> the ComfyUI layout. The PDD heads stay as
    pdd_proj_out.N.* for the caller to stack."""
    out = {}
    parts = {}
    for key, tensor in sd.items():
        if key.startswith('pdd_proj_out.'):
            out[key] = tensor
            continue
        match = re.fullmatch(r'(.*)\.attn\.to_([qkv])\.weight', key)
        if match:
            parts.setdefault(match.group(1), {})[match.group(2)] = tensor
            continue
        if key.endswith('.ff.net.0.proj.weight'):
            # diffusers' SwiGLU is [value; gate], ComfyUI's fc1 [gate; value]
            value, gate = tensor.chunk(2, dim=0)
            out[diffusers_key(key.replace('.ff.net.0.proj.weight', '.mlp.fc1.weight'))] = torch.cat([gate, value], dim=0)
            continue
        out[diffusers_key(key)] = tensor
    for prefix, qkv in parts.items():
        if set(qkv) != {'q', 'k', 'v'}:
            raise ValueError(f'{prefix} lacks one of to_q / to_k / to_v')
        out[diffusers_key(prefix + '.attn.qkv_proj.weight')] = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
    if 'rope.inv_freq' not in out:
        out['rope.inv_freq'] = rope_inv_freq(rope_theta, rope_freq_dim)
    return out


def official_to_comfy(sd):
    """The official transformer/ shards -> the ComfyUI layout: only the qkv rows move."""
    head_dim = sd['blocks.0.attn.q_norm.weight'].shape[0]
    out = {}
    for key, tensor in sd.items():
        if key.endswith('.attn.qkv_proj.weight'):
            tensor = reorder_interleaved_qkv(tensor, tensor.shape[0] // (3 * head_dim), head_dim)
        out[key] = tensor
    return out


def _read_index(path):
    with open(path) as f:
        weight_map = json.load(f)['weight_map']
    directory = os.path.dirname(os.path.abspath(path))
    return sorted({os.path.join(directory, name) for name in weight_map.values()})


def _find_shards(directory):
    for name in ('model.safetensors.index.json', 'diffusion_pytorch_model.safetensors.index.json'):
        if os.path.exists(os.path.join(directory, name)):
            return _read_index(os.path.join(directory, name))
    files = sorted(glob.glob(os.path.join(directory, '*.safetensors')))
    if not files:
        raise FileNotFoundError(f'no .safetensors files in {directory}')
    return files


def checkpoint_files(path):
    """(layout hint or None, shard files, directory holding the transformer) for a diffusion_model path: a
    .safetensors file, a shard index (.json), a directory of shards, or a repository root with a transformer/
    (or transformer_diffusers/) folder."""
    path = str(path)
    if os.path.isdir(path):
        for sub, hint in (('transformer', 'official'), ('transformer_diffusers', 'diffusers')):
            if os.path.isdir(os.path.join(path, sub)):
                return hint, _find_shards(os.path.join(path, sub)), os.path.join(path, sub)
        return None, _find_shards(path), path
    if path.endswith('.json'):
        files = _read_index(path)
        return None, files, os.path.dirname(os.path.abspath(path))
    if os.path.isfile(path):
        return None, [path], os.path.dirname(os.path.abspath(path))
    raise FileNotFoundError(f'diffusion_model {path} does not exist')


def _shard_keys(files):
    keys = {}
    for file in files:
        with safetensors.safe_open(file, framework='pt', device='cpu') as f:
            for key in f.keys():
                keys[key] = file
    return keys


def detect_layout(hint, files, keys, directory, forced=None):
    if forced is not None:
        if forced not in LAYOUTS:
            raise ValueError(f"[model] diffusion_model_format must be one of {LAYOUTS}, got {forced!r}")
        return forced
    if any(key.startswith('transformer_blocks.') for key in keys):
        return 'diffusers'
    if hint is not None:
        return hint
    config_path = os.path.join(directory, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            if json.load(f).get('_class_name') == 'MiniMaxH3DiTModel':
                return 'official'
    if len(files) == 1:
        return 'comfy'
    # The official shards and a sharded ComfyUI file have the same names and shapes: only the qkv row order differs,
    # so a guess would silently scramble the attention.
    if find_official_heads(directory) is not None:
        return 'official'
    raise ValueError(f'cannot tell whether the {len(files)} shards of {directory} are the official layout (interleaved qkv) or '
                     f"ComfyUI's: set [model] diffusion_model_format = 'official' or 'comfy'")


def check_is_image_model(keys):
    if any(key.startswith(('audio_patch_proj.', 'audio_proj_in.', 'final_layer.audio_out.')) or '.audio_' in key for key in keys):
        raise ValueError("this transformer has an audio stream: it is MiniMax H3 (video and audio), not MiniMax-H3-Image. "
                         "Use type = 'minimax_h3'")
    diffusers = any(k.startswith('transformer_blocks.') for k in keys)
    required = ('proj_in.weight', 'context_embedder.weight') if diffusers else \
        ('video_patch_proj.weight', 'condition_proj.weight', 'final_layer.video_out.weight')
    for name in required:
        if name not in keys:
            raise ValueError(f'this is not a MiniMax-H3-Image transformer: {name} is missing')


def find_pdd_sigmas(directory, *others):
    """The sigma grid of a repository's pdd_config.json (next to or above the transformer folder, or in one of
    the other folders), or None."""
    for folder in (directory, os.path.dirname(directory), *others):
        path = os.path.join(folder, 'pdd_config.json')
        if os.path.exists(path):
            with open(path) as f:
                return [float(s) for s in json.load(f)['sigmas']]
    return None


def read_pdd_config(directory):
    """find_pdd_sigmas, else the released grid."""
    return find_pdd_sigmas(directory) or list(DEFAULT_PDD_SIGMAS)


def find_official_heads(directory, explicit=None):
    if explicit is not None:
        return explicit
    for folder in (directory, os.path.dirname(directory)):
        path = os.path.join(folder, OFFICIAL_HEADS_FILE)
        if os.path.exists(path):
            return path
    return None


def transformer_metadata(steps, sigmas, base=None):
    """The metadata ComfyUI's loader reads; `base` is a file's own metadata, whose other keys are kept."""
    if len(sigmas) != steps + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0:
        raise ValueError(f'{steps} PDD heads need a grid of {steps + 1} sigmas from 1.0 to 0.0 (pdd_config.json next to the '
                         f'transformer or the heads file), got {sigmas}')
    config = json.loads((base or {}).get('config') or '{}')
    config['transformer'] = {**config.get('transformer', {}), 'pdd_steps': steps, 'pdd_sigmas': sigmas}
    return {**(base or {}), 'config': json.dumps(config), 'format': 'pt'}


def load_checkpoint(path, forced_layout=None, heads_path=None):
    """(state dict in the ComfyUI layout, metadata for ComfyUI's loader, layout name, number of PDD heads).

    heads_path: the official layout's pdd_fused_heads.safetensors when it is not at the repository root."""
    hint, files, directory = checkpoint_files(path)
    keys = _shard_keys(files)
    layout = detect_layout(hint, files, keys, directory, forced_layout)
    check_is_image_model(keys)
    sd, metadata = {}, {}
    for file in files:
        shard, shard_metadata = comfy.utils.load_torch_file(file, return_metadata=True)
        sd.update(shard)
        metadata.update(shard_metadata or {})
    sigmas = None
    if layout == 'comfy':
        steps = 1
        if 'config' in metadata:
            transformer = json.loads(metadata['config']).get('transformer', {})
            steps, sigmas = transformer.get('pdd_steps', 1), transformer.get('pdd_sigmas')
        if sigmas is None:
            # A file without the grid (a single-head export, say): ComfyUI's model needs one to be built.
            sigmas = find_pdd_sigmas(directory) or (list(DEFAULT_PDD_SIGMAS) if steps == len(DEFAULT_PDD_SIGMAS) - 1 else [1.0, 0.0])
        return sd, transformer_metadata(steps, sigmas, metadata), layout, steps

    if layout == 'diffusers':
        theta, freq_dim = DEFAULT_ROPE_THETA, DEFAULT_ROPE_FREQ_DIM
        config_path = os.path.join(directory, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            theta, freq_dim = config.get('rope_theta', theta), config.get('rope_freq_dim', freq_dim)
        sd = diffusers_to_comfy(sd, theta, freq_dim)
        heads = sorted({int(m.group(1)) for k in sd if (m := re.match(r'pdd_proj_out\.(\d+)\.weight$', k))})
        if heads:
            if heads != list(range(len(heads))):
                raise ValueError(f'pdd_proj_out heads are not numbered 0..n-1: {heads}')
            weight = torch.stack([sd.pop(f'pdd_proj_out.{i}.weight') for i in heads])
            bias = torch.stack([sd.pop(f'pdd_proj_out.{i}.bias') for i in heads])
            sd['final_layer.video_out.weight'], sd['final_layer.video_out.bias'] = pdd_bank(weight, bias)
            steps = len(heads)
            sigmas = find_pdd_sigmas(directory) or (list(DEFAULT_PDD_SIGMAS) if steps == len(DEFAULT_PDD_SIGMAS) - 1 else None)
        else:
            steps = 1
        if steps > 1 and sigmas is None:
            raise ValueError(f'{steps} PDD heads and no pdd_config.json with their sigma grid')
        return sd, transformer_metadata(steps, sigmas or [1.0, 0.0]), layout, steps

    sd = official_to_comfy(sd)
    heads_file = find_official_heads(directory, heads_path)
    if heads_file is None:
        # Only the base head: it trains, but the six step heads a preview samples with are missing.
        return sd, transformer_metadata(1, [1.0, 0.0]), layout, 1
    with safetensors.safe_open(heads_file, framework='pt', device='cpu') as f:
        if set(f.keys()) != {'video_out.weight', 'video_out.bias'}:
            raise ValueError(f'{heads_file} must hold video_out.weight and video_out.bias, got {sorted(f.keys())}')
        weight, bias = f.get_tensor('video_out.weight'), f.get_tensor('video_out.bias')
    steps = weight.shape[0]
    sd['final_layer.video_out.weight'], sd['final_layer.video_out.bias'] = pdd_bank(weight, bias)
    sigmas = find_pdd_sigmas(directory, os.path.dirname(os.path.abspath(heads_file))) or list(DEFAULT_PDD_SIGMAS)
    return sd, transformer_metadata(steps, sigmas), layout, steps


def inspect_checkpoint(path, forced_layout=None, heads_path=None):
    """What load_checkpoint would find, from the headers alone: (layout, number of PDD heads or None when
    only a base head is there). Raises for a transformer this pipeline cannot train."""
    hint, files, directory = checkpoint_files(path)
    keys = _shard_keys(files)
    layout = detect_layout(hint, files, keys, directory, forced_layout)
    check_is_image_model(keys)
    if layout == 'diffusers':
        return layout, len({k for k in keys if k.startswith('pdd_proj_out.') and k.endswith('.weight')}) or 1
    if layout == 'official':
        heads_file = find_official_heads(directory, heads_path)
        if heads_file is None:
            return layout, 1
        with safetensors.safe_open(heads_file, framework='pt', device='cpu') as f:
            return layout, f.get_slice('video_out.weight').get_shape()[0]
    with safetensors.safe_open(files[0], framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
    if 'config' in metadata:
        return layout, json.loads(metadata['config']).get('transformer', {}).get('pdd_steps', 1)
    return layout, 1


VAE_DECODER_HEAD_DIM = 64
TEXT_ENCODER_LAYERS = 50


def interleave_qkv(q, k, v, heads, head_dim):
    """The inverse of reorder_interleaved_qkv: separate q / k / v -> rows [head0: q k v, head1: q k v, ...]."""
    parts = [t.reshape(heads, head_dim, *t.shape[1:]) for t in (q, k, v)]
    return torch.cat(parts, dim=1).reshape(heads * 3 * head_dim, *q.shape[1:])


def is_diffusers_vae(keys):
    return any(key.startswith('encoder.down_blocks.') for key in keys)


def diffusers_vae_to_comfy(sd):
    """The vae_diffusers/ weights -> the layout of vae/ (and of ComfyUI's loader): CompVis names for the
    encoder, x_embedder, a fused per-head-interleaved to_qkv, w1 = [gate; up]."""
    out, parts = {}, {}
    for key, tensor in sd.items():
        match = re.fullmatch(r'(.*)\.attn\.to_([qkv])\.(weight|bias)', key)
        if match:
            parts.setdefault((match.group(1), match.group(3)), {})[match.group(2)] = tensor
            continue
        new_key = key
        match = re.match(r'encoder\.down_blocks\.(\d+)\.(.*)$', key)
        if match:
            rest = match.group(2).replace('resnets.', 'block.', 1).replace('conv_shortcut.', 'nin_shortcut.', 1)
            rest = rest.replace('downsamplers.0.', 'downsample.', 1)
            new_key = f'encoder.down.{match.group(1)}.{rest}'
        new_key = new_key.replace('decoder.proj_in.', 'decoder.x_embedder.')
        new_key = new_key.replace('.attn.to_out.0.', '.attn.to_out.')
        new_key = new_key.replace('.ff.net.2.', '.ff.w2.')
        if '.ff.net.0.proj.' in new_key:
            new_key = new_key.replace('.ff.net.0.proj.', '.ff.w1.')
            up, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat([gate, up], dim=0)
        out[new_key] = tensor
    for (prefix, suffix), qkv in parts.items():
        if set(qkv) != {'q', 'k', 'v'}:
            raise ValueError(f'{prefix}.attn lacks one of to_q / to_k / to_v ({suffix})')
        heads = qkv['q'].shape[0] // VAE_DECODER_HEAD_DIM
        out[f'{prefix}.attn.to_qkv.{suffix}'] = interleave_qkv(qkv['q'], qkv['k'], qkv['v'], heads, VAE_DECODER_HEAD_DIM)
    return out


def vae_file(path):
    """The weights file of a VAE given as a file or as a folder (vae/ or vae_diffusers/)."""
    path = str(path)
    if os.path.isdir(path):
        path = os.path.join(path, 'diffusion_pytorch_model.safetensors')
    return path


def load_vae_state(path):
    sd = comfy.utils.load_torch_file(vae_file(path))
    return diffusers_vae_to_comfy(sd) if is_diffusers_vae(sd) else sd


def text_encoder_key(key, layers=TEXT_ENCODER_LAYERS):
    """The ComfyUI key of a Hugging Face Qwen3-VL key, or None for what MiniMax-H3 leaves out (the layers past
    the 50th, the final norm and the LM head: it conditions on the unnormalized state after layer 50)."""
    match = re.match(r'model\.language_model\.layers\.(\d+)\.(.*)$', key)
    if match:
        return f'model.layers.{match.group(1)}.{match.group(2)}' if int(match.group(1)) < layers else None
    if key == 'model.language_model.embed_tokens.weight':
        return 'model.embed_tokens.weight'
    if key.startswith('model.visual.'):
        return key[len('model.'):]
    return None


def text_encoder_files(paths):
    """The weight files of a text encoder given as a file, a list of files, a shard index or a folder."""
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for path in paths:
        path = str(path)
        if os.path.isdir(path):
            files.extend(_find_shards(path))
        elif path.endswith('.json'):
            files.extend(_read_index(path))
        else:
            files.append(path)
    return files


def load_text_encoder_state(paths):
    """The ComfyUI-layout state dict of a Hugging Face Qwen3-VL-32B text_encoder/ (all 64 layers), cut to the
    50 MiniMax-H3 reads; None when the files are already ComfyUI's (the standard loader takes those)."""
    files = text_encoder_files(paths)
    keys = _shard_keys(files)
    if 'model.language_model.embed_tokens.weight' not in keys:
        return None
    sd = {}
    for key, file in keys.items():
        new_key = text_encoder_key(key)
        if new_key is not None:
            with safetensors.safe_open(file, framework='pt', device='cpu') as f:
                sd[new_key] = f.get_tensor(key)
    if f'model.layers.{TEXT_ENCODER_LAYERS - 1}.self_attn.q_proj.weight' not in sd:
        raise ValueError(f'the text encoder has fewer than {TEXT_ENCODER_LAYERS} layers')
    return sd
