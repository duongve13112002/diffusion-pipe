"""MOSS-Image (OpenMOSS-Team/MOSS-Image-Base, -SFT): a 5B single-stream text-to-image DiT.

The transformer is ComfyUI's own MOSS-Image model (comfy.ldm.moss_image.model.MOSSImage), so what
trains here is what ComfyUI runs, and its state-dict names are what ComfyUI and diffusers load. It
is built with q, k and v as separate projections whatever the checkpoint's layout, so adapters
always have that layout. Its forward is cut into pipeline layers below; the blocks run their own
forward, on ComfyUI's differentiable path (models/base.py sets model_management.in_training). Two
things in that forward shape the cut:

  - Every block reads its adaLN parameters from one shared Linear (shared_block_modulation). It
    runs once, in InitialLayer, and its output travels down the pipeline with the hidden states.
  - The model evaluates each sample at its own text length: text tokens are never padded inside
    the transformer, and the reference encodes every prompt alone for the same reason (Qwen3.5's
    gated-delta layers are not padding invariant). The layers below keep that: samples of equal
    text length run together, others run separately, and the padding between them only exists
    between layers.

The text encoder is Qwen3.5-9B (last hidden state, MOSS's chat template, at most 4096 tokens),
run through ComfyUI without its vision tower. The VAE is FLUX.2's, normalised by
comfy.latent_formats.Flux2 exactly as the reference pipeline does.

Weights come either from a diffusers directory (diffusers_path, which supplies the transformer,
the text encoder and the VAE), from single files (transformer_path, text_encoder_path, vae_path,
ComfyUI or diffusers layout), or a mix: a component path overrides the same component of
diffusers_path.
"""

import json
import logging
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from models.base import ComfyPipeline, ModelWrapper, PreprocessMediaFile, make_contiguous
from utils.common import is_main_process
from utils.offloading import ModelOffloader
import comfy.latent_formats
import comfy.lora
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.sd1_clip
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils
import comfy.ldm.moss_image.model
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.moss_image.model import modulated_rms_norm
from comfy.text_encoders import qwen35


NUM_TRAIN_TIMESTEPS = 1000
# ComfyUI's MOSS-Image fixes these two; every released checkpoint uses them.
FREQ_DIM = 256
ROPE_BASE = 10000.0
MAX_TEXT_TOKENS = 4096
TEXT_DIM = 4096
LATENT_CHANNELS = 128
# Settings of the reference pipeline (pipeline_moss_image.py / the model cards).
SAMPLE_DEFAULTS = {
    'sample_steps': 50,
    'sample_cfg': 3.0,
    'sample_shift': 5.0,
    'sample_negative_prompt': '',
    'sample_uncond_skip_layers': [22],
    'sample_slg_layers': [],
    'sample_slg_scale': 0.0,
    'sample_slg_start': 0.0,
    'sample_slg_stop': 1.0,
    'sample_seed': 0,
}
# Final pretraining interval of the technical report (Table 6, eq. 4).
NOISE_DEFAULTS = {
    'timestep_sample_method': 'logit_normal',
    'logit_mean': -1.09,
    'logit_std': 0.95,
    'sigma_cutoff': 0.95,
    'uniform_fraction': 0.01,
    'shift': 1.0,
    'min_t': 0.0,
    'max_t': 1.0,
}
# Everything outside the 42 blocks. Kept at `dtype` when diffusion_model_dtype is lower
# precision; the community quantizations leave the same modules unquantized.
KEEP_IN_HIGH_PRECISION = [
    'video_embedder', 'text_embedder', 'time_embedding', 'shared_block_modulation',
    'final_norm_video', 'final_linear_video', 'video_head_modulation',
]
# Keys every MOSS-Image transformer has; a state dict without them is something else.
REQUIRED_TRANSFORMER_KEYS = (
    'video_embedder.bias', 'text_embedder.proj.weight', 'text_embedder.blocks.0.qkv.weight',
    'shared_block_modulation.weight', 'layers.0.q_norm.weight', 'layers.0.proj.ffn_up_gate.weight',
    'final_linear_video.weight', 'video_head_modulation.weight', 'time_embedding.0.bias',
)
DTYPE_NAMES = {
    'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16, 'float32': torch.float32, 'fp32': torch.float32,
    'float8_e4m3fn': torch.float8_e4m3fn, 'float8': torch.float8_e4m3fn, 'float8_e5m2': torch.float8_e5m2,
}


def _dtype(value, key):
    if isinstance(value, torch.dtype):
        return value
    if value not in DTYPE_NAMES:
        raise ValueError(f'{key} must be one of {sorted(DTYPE_NAMES)}, got {value!r}')
    return DTYPE_NAMES[value]


# ---- weights on disk -------------------------------------------------------------------------------

def read_component(path, stem):
    """State dict and metadata of a diffusers component directory or a single safetensors file.

    A directory holds either `{stem}.safetensors` or shards listed in
    `{stem}.safetensors.index.json`; every shard has to contain exactly the keys the index assigns
    it, so a partly downloaded component fails here instead of loading with missing weights.
    """
    path = Path(path)
    if path.is_file():
        return comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    if not path.is_dir():
        raise FileNotFoundError(f'{path} does not exist')
    index = path / f'{stem}.safetensors.index.json'
    single = path / f'{stem}.safetensors'
    if not index.is_file():
        if single.is_file():
            return comfy.utils.load_torch_file(str(single), safe_load=True, return_metadata=True)
        candidates = sorted(path.glob('*.safetensors'))
        if len(candidates) == 1:
            return comfy.utils.load_torch_file(str(candidates[0]), safe_load=True, return_metadata=True)
        raise FileNotFoundError(
            f'{path} has neither {index.name} nor {single.name}'
            + (f', and {len(candidates)} other .safetensors files to choose from' if candidates else ''))
    mapping = json.loads(index.read_text())['weight_map']
    state = {}
    metadata = None
    root = path.resolve()
    for filename in sorted(set(mapping.values())):
        shard = (path / filename).resolve()
        if root not in shard.parents:
            raise ValueError(f'Shard {filename} in {index} points outside {path}')
        values, shard_metadata = comfy.utils.load_torch_file(str(shard), safe_load=True, return_metadata=True)
        expected = {key for key, file in mapping.items() if file == filename}
        if set(values) != expected:
            missing, extra = sorted(expected - set(values)), sorted(set(values) - expected)
            raise ValueError(
                f'{shard} does not match {index.name}: missing {missing[:3]}, unexpected {extra[:3]}. '
                'The download is probably incomplete.')
        state.update(values)
        metadata = metadata or shard_metadata
    return state, metadata


TRANSFORMER_PREFIXES = ('', 'model.diffusion_model.', 'diffusion_model.', 'transformer.')


def _transformer_prefix(state):
    for prefix in TRANSFORMER_PREFIXES:
        if prefix + 'shared_block_modulation.weight' in state:
            return prefix
    return ''


def transformer_state(state, metadata):
    """The transformer's keys without a checkpoint prefix, with old quantization formats converted.

    The prefix is the one under which the transformer's own keys sit, and only keys under it are
    kept, as ComfyUI does: a full checkpoint may carry a VAE or text encoder beside it. Quantized
    layers are found as ComfyUI finds them, whether a checkpoint's quantization metadata names
    them with the prefix or without it.
    """
    prefix = _transformer_prefix(state)
    state, metadata = comfy.utils.convert_old_quants(dict(state), prefix, metadata=metadata)
    if not prefix:
        return split_fused_qkv(state), metadata
    stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    for k, v in state.items():
        if (k.endswith('.comfy_quant') and not k.startswith(prefix)
                and k[:-len('comfy_quant')] + 'weight' in stripped):
            stripped[k] = v
    return split_fused_qkv(stripped), metadata


# Quantization formats whose layers can be cut by output rows: row-major weights with one scale
# per tensor or per row. Block-scaled formats (nvfp4, mxfp8) keep their scales swizzled.
ROW_SPLITTABLE_QUANT_FORMATS = ('float8_e4m3fn', 'float8_e5m2', 'int8_tensorwise')
_FUSED_QKV = re.compile(r'layers\.\d+\.proj\.qkv\.weight')


def split_fused_qkv(state):
    """state with each block's fused proj.qkv cut into proj.q, proj.k and proj.v.

    Comfy-Org/MOSS-Image and ai-toolkit's saves fuse q, k and v into one Linear (rows q, k, v);
    this pipeline builds ComfyUI's model with them apart, so adapters have one layout whatever
    the checkpoint's (ComfyUI maps q/k/v adapters onto a fused file too). Cutting by rows is exact
    for plain weights and for quantized ones with a scale per tensor or per row (int8_tensorwise
    with convrot rotates the input dimension, which q, k and v share).
    """
    fused = sorted(k[:-len('.weight')] for k in state if _FUSED_QKV.fullmatch(k))
    if not fused:
        return state
    state = dict(state)
    for base in fused:
        block = base[:-len('.qkv')]
        both = [f'{block}.{p}.weight' for p in 'qkv' if f'{block}.{p}.weight' in state]
        if both:
            raise ValueError(f'The MOSS-Image checkpoint has both {base}.weight and {both[0]}')
        if (quant := state.get(base + '.comfy_quant')) is not None:
            layout = json.loads(bytes(quant.tolist()).decode('utf-8')).get('format')
            if layout not in ROW_SPLITTABLE_QUANT_FORMATS:
                raise ValueError(
                    f'{base} is quantized as {layout!r}, which cannot be split into q, k and v '
                    f'(only {", ".join(ROW_SPLITTABLE_QUANT_FORMATS)} can). Use an unquantized or '
                    'one of those checkpoints.')
        rows = state[base + '.weight'].shape[0]
        if rows % 3:
            raise ValueError(f'{base}.weight has {rows} rows, not three equal parts q, k and v')
        for key in [k for k in state if k.startswith(base + '.')]:
            value = state.pop(key)
            suffix = key[len(base):]
            if suffix == '.comfy_quant' or value.ndim == 0 or value.shape[0] != rows:
                parts = (value,) * 3  # format description, per-tensor scale
            else:
                parts = value.chunk(3, 0)  # views: load_state_dict copies them into the model
            for name, part in zip('qkv', parts):
                state[f'{block}.{name}{suffix}'] = part
    return state


# The block's fused qkv under PEFT/ComfyUI names (diffusion_model.layers.0.proj.qkv.lora_A.weight)
# and kohya ones (lora_unet_layers_0_proj_qkv.lora_down.weight).
_FUSED_QKV_ADAPTER = re.compile(r'(.*layers[._]\d+[._]proj[._])qkv\.(.+)')


def split_fused_qkv_adapter(sd, source):
    """An adapter for the fused proj.qkv (ai-toolkit's MOSS-Image LoRAs) as one for q, k and v.

    The rows of the fused update B @ A are q, k and v, so each gets A, its third of B and the
    same alpha: exact. A full-weight diff is cut the same way. Other adapter kinds (LoKr, LoHa,
    DoRA) do not split by rows and are refused. A diffusers-style 'transformer.' prefix becomes
    ComfyUI's 'diffusion_model.'.

    Returns the adapter and, for each of its keys, the key of the file it came from.
    """
    out, origin = {}, {}
    for key, value in sd.items():
        name = 'diffusion_model.' + key[len('transformer.'):] if key.startswith('transformer.') else key
        match = _FUSED_QKV_ADAPTER.fullmatch(name)
        if match is None:
            pieces = [(name, value)]
        else:
            head, rest = match.groups()
            if rest in ('lora_A.weight', 'lora_down.weight', 'alpha'):
                parts = (value,) * 3
            elif rest in ('lora_B.weight', 'lora_up.weight', 'diff', 'diff_b'):
                if value.shape[0] % 3:
                    raise ValueError(f'{source}: {key} has {value.shape[0]} rows, not three equal parts')
                parts = value.chunk(3, 0)
            else:
                raise ValueError(
                    f'{source}: {key} adapts the fused proj.qkv in a form that cannot be split into q, k '
                    'and v (only LoRA and full-weight diffs can)')
            pieces = [(f'{head}{p}.{rest}', part) for p, part in zip('qkv', parts)]
        for new, part in pieces:
            if new in out:
                raise ValueError(f'{source}: {key} and {origin[new]} both adapt {new}')
            out[new], origin[new] = part, key
    return out, origin


class _RecordingLogging:
    """The logging module as comfy.lora and comfy.sd see it while an adapter merges.

    They only log the adapter keys they could not place; recording the calls themselves does not
    depend on logger levels, filters or logging.disable.
    """

    MARKERS = ('lora key not loaded: ', 'NOT LOADED ')

    def __init__(self):
        self.keys = []

    def __getattr__(self, name):
        return getattr(logging, name)

    def warning(self, msg, *args, **kwargs):
        text = str(msg) % args if args else str(msg)
        for marker in self.MARKERS:
            if text.startswith(marker):
                self.keys.append(text[len(marker):])
        logging.warning(msg, *args, **kwargs)


def merge_adapter(patcher, adapter_path):
    """patcher with the adapter file at adapter_path added at strength 1; every key must apply."""
    sd = comfy.utils.load_torch_file(str(adapter_path), safe_load=True)
    sd, origin = split_fused_qkv_adapter(sd, adapter_path)
    recorder = _RecordingLogging()
    modules = (comfy.lora, comfy.sd)
    saved = [module.logging for module in modules]
    for module in modules:
        module.logging = recorder
    try:
        patcher, _ = comfy.sd.load_lora_for_models(patcher, None, sd, 1.0, 0.0)
    finally:
        for module, value in zip(modules, saved):
            module.logging = value
    if recorder.keys:
        keys = sorted({origin.get(key, key) for key in recorder.keys})
        raise ValueError(f'{adapter_path}: {len(keys)} adapter keys match no MOSS-Image weight, e.g. {keys[:3]}')
    return patcher


def transformer_unet_config(state, rope_base=ROPE_BASE):
    """The transformer's diffusers config (MOSSImageTransformer's arguments), read from its weights.

    rope_base is the one argument the weights do not determine; it comes from config.json.
    """
    missing = [key for key in REQUIRED_TRANSFORMER_KEYS if key not in state]
    if missing:
        raise ValueError(f'Not a MOSS-Image transformer checkpoint; missing {missing}')
    dim = state['video_embedder.bias'].shape[0]
    head_dim = state['layers.0.q_norm.weight'].shape[0]
    num_layers = 0
    while f'layers.{num_layers}.proj.q.weight' in state or f'layers.{num_layers}.norm1.weight' in state:
        num_layers += 1
    return {
        'dim': dim,
        'num_heads': dim // head_dim,
        'num_layers': num_layers,
        'ffn_dim': state['layers.0.proj.ffn_up_gate.weight'].shape[0] // 2,
        'latent_channels': state['final_linear_video.weight'].shape[0],
        'text_dim': state['text_embedder.proj.weight'].shape[1],
        'freq_dim': state['time_embedding.0.weight'].shape[1],
        'rope_base': float(rope_base),
    }


def comfy_unet_config(unet_config):
    """ComfyUI's MOSSImage arguments for a transformer of this diffusers config.

    ComfyUI's model has a fixed timestep embedding width and RoPE base, and splits each head's
    rotary dimensions over four axes; a config outside that is refused rather than run wrongly.
    """
    wrong = []
    if unet_config['freq_dim'] != FREQ_DIM:
        wrong.append(f'freq_dim = {unet_config["freq_dim"]} (ComfyUI\'s MOSS-Image uses {FREQ_DIM})')
    if unet_config['rope_base'] != ROPE_BASE:
        wrong.append(f'rope_base = {unet_config["rope_base"]} (ComfyUI\'s MOSS-Image uses {ROPE_BASE:g})')
    dim, heads = unet_config['dim'], unet_config['num_heads']
    if dim % heads or (dim // heads) % 8:
        wrong.append(f'dim = {dim} with num_heads = {heads} (the head size must be a multiple of 8)')
    if wrong:
        raise ValueError('This MOSS-Image transformer cannot run as ComfyUI\'s MOSS-Image: ' + ', '.join(wrong))
    return {
        'image_model': 'moss_image',
        'dim': dim,
        'ffn_dim': unet_config['ffn_dim'],
        'num_heads': heads,
        'num_layers': unet_config['num_layers'],
        'text_dim': unet_config['text_dim'],
        'latent_channels': unet_config['latent_channels'],
        # Always q, k and v apart (split_fused_qkv), so adapters have one layout.
        'fused_qkv': False,
    }


def read_transformer_config(path):
    """config.json of a diffusers transformer directory ({} when there is none).

    Refuses a config.json written for another model.
    """
    config_file = Path(path) / 'config.json'
    if not config_file.is_file():
        return {}
    config = json.loads(config_file.read_text())
    if config.get('_class_name') not in (None, 'MOSSImageTransformer'):
        raise ValueError(f'{config_file} describes a {config.get("_class_name")}, not a MOSS-Image transformer')
    rope_base = config.get('rope_base', ROPE_BASE)
    if not isinstance(rope_base, (int, float)) or isinstance(rope_base, bool) or not rope_base > 0:
        raise ValueError(f'{config_file}: rope_base must be a positive number, got {rope_base!r}')
    return config


def check_declared_config(declared, unet_config, source):
    """A config.json must describe the weights next to it."""
    wrong = [f'{key} = {declared[key]} (the weights say {unet_config[key]})'
             for key in unet_config if key in declared and key != 'rope_base' and declared[key] != unet_config[key]]
    if wrong:
        raise ValueError(f'{source} does not match its weights: ' + ', '.join(wrong))


# Keys of a checkpoint the model does not load: the RoPE frequencies (a derived constant ComfyUI
# recomputes; diffusers keeps it as a buffer) and Comfy-Org's int8 files' attention choice, an
# int8 attention kernel picked for inference. Training runs ComfyUI's regular attention.
_NOT_LOADED = re.compile(r'rope\.bands|.*\.comfy_attention\.config')

# What ComfyUI's checkpoint detection (model_detection) assumes of a MOSS-Image transformer: a
# file of another shape trains here, but ComfyUI's Load Diffusion Model cannot load what it saves.
COMFY_DETECTED = {'head size': 128, 'text_dim': 4096, 'latent_channels': 128}
RELEASED_NUM_LAYERS = 42


def model_keys(comfy_config):
    """The state dict keys ComfyUI's MOSSImage of this config has, built without any memory.

    ComfyUI's mixed-precision operations do not report a quantized layer whose weight is missing
    (nor its bias), so missing keys are found from these rather than from load_state_dict.
    """
    model = comfy.ldm.moss_image.model.MOSSImage(**comfy_config, device='meta', operations=comfy.ops.disable_weight_init)
    return set(model.state_dict())


def warn_about_shape(unet_config, declared, source):
    detected = {'head size': unet_config['dim'] // unet_config['num_heads'],
                'text_dim': unet_config['text_dim'], 'latent_channels': unet_config['latent_channels']}
    odd = [f'{key} = {detected[key]} (ComfyUI expects {value})'
           for key, value in COMFY_DETECTED.items() if detected[key] != value]
    if odd:
        logging.warning(f'{source}: ComfyUI cannot detect a MOSS-Image transformer with '
                        f'{", ".join(odd)}, so its Load Diffusion Model will not load what this run saves')
    # A single file has no record of its block count: the count ends at the first block missing.
    if unet_config['num_layers'] != RELEASED_NUM_LAYERS and 'num_layers' not in declared:
        logging.warning(f'{source} has {unet_config["num_layers"]} blocks; the released MOSS-Image '
                        f'transformers have {RELEASED_NUM_LAYERS}. If this file should have more, it is truncated.')


def load_transformer_patcher(state, metadata, dtype, load_device=None, offload_device=None, declared=None):
    """ModelPatcher for a MOSS-Image transformer state dict.

    comfy.sd.load_diffusion_model_state_dict, with the checkpoint normalised first: any prefix
    removed, a fused proj.qkv split, a diffusers config.json honoured. Comfy quantized checkpoints
    (fp8 scaled, comfy-kitchen formats) go through ComfyUI's own mixed-precision operations, as
    they do in ComfyUI.

    declared is the transformer directory's config.json, if any: checked against the weights,
    and the source of rope_base.
    """
    declared = declared or {}
    state, metadata = transformer_state(state, metadata)
    state = {k: v for k, v in state.items() if not _NOT_LOADED.fullmatch(k)}
    unet_config = transformer_unet_config(state, declared.get('rope_base', ROPE_BASE))
    check_declared_config(declared, unet_config, 'The transformer config.json')
    comfy_config = comfy_unet_config(unet_config)
    missing = sorted(model_keys(comfy_config) - state.keys())
    if missing:
        raise RuntimeError(f'The MOSS-Image checkpoint is missing {len(missing)} weights, e.g. {missing[:5]}')
    warn_about_shape(unet_config, declared, 'The MOSS-Image transformer')
    config = comfy.supported_models.MOSSImage(comfy_config)
    config.quant_config = comfy.utils.detect_layer_quantization(state, '')
    if load_device is None:
        load_device = comfy.model_management.get_torch_device()
    if offload_device is None:
        offload_device = comfy.model_management.unet_offload_device()
    manual_cast = comfy.model_management.unet_manual_cast(
        None if config.quant_config else dtype, load_device, config.supported_inference_dtypes)
    config.set_inference_dtype(dtype, manual_cast, device=load_device)
    model = config.get_model(state)
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    missing, unexpected = model.diffusion_model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f'The MOSS-Image checkpoint is missing {len(missing)} weights, e.g. {missing[:5]}')
    # Quantized layers' scales are consumed by the comfy ops, so anything left over is not the model's
    # (a block past the first one missing, a stray key), quantized or not.
    if unexpected:
        raise RuntimeError(f'Unexpected keys in the MOSS-Image checkpoint, e.g. {unexpected[:5]}')
    return patcher


def default_rope_bands(head_dim, base=ROPE_BASE):
    """The diffusers model's rope.bands buffer: base^(-i / n) for its n = head_dim / 8 bands, in
    fp32 (ComfyUI's EmbedND computes the same frequencies)."""
    num_bands = head_dim // 8
    return 1 / (base ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))


# ---- text encoder ----------------------------------------------------------------------------------

def text_encoder_state(state):
    """A Qwen3.5-9B state dict in the layout ComfyUI's Qwen35 expects, without the vision tower.

    Accepted layouts: the MOSS-Image text_encoder folder (transformers Qwen3_5Model:
    language_model.*, visual.*), the full VLM (model.language_model.*, model.visual.*, lm_head.*),
    and ComfyUI's own (model.*).
    """
    state = qwen35_vlm_layout(state)
    detected = comfy.sd.detect_te_model(state)
    if detected != comfy.sd.TEModel.QWEN35_9B:
        raise ValueError(f'MOSS-Image needs a Qwen3.5-9B text encoder; this checkpoint looks like {detected}')
    return qwen35_comfy_layout(state)


def qwen35_vlm_layout(state):
    """Any accepted layout -> the full VLM's (model.language_model.*), which ComfyUI detects."""
    state = comfy.utils.state_dict_prefix_replace(
        state, {'language_model.': 'model.language_model.', 'visual.': 'model.visual.'})
    if 'model.layers.0.linear_attn.A_log' in state and 'model.language_model.layers.0.linear_attn.A_log' not in state:
        state = comfy.utils.state_dict_prefix_replace(state, {'model.': 'model.language_model.'})
        state = comfy.utils.state_dict_prefix_replace(state, {'model.language_model.visual.': 'model.visual.'})
    return state


def qwen35_comfy_layout(state):
    """The VLM layout -> what ComfyUI's Qwen35 loads (model.*), text tower only."""
    state = comfy.utils.state_dict_prefix_replace(state, {'model.language_model.': 'model.'})
    drop = ('model.visual.', 'visual.', 'lm_head.', 'model.lm_head.', 'mtp.')
    return {k: v for k, v in state.items() if not k.startswith(drop)}


class MossImageTokenizer(comfy.sd1_clip.SD1Tokenizer):
    """Tokens of MOSS-Image's chat template, at most MAX_TEXT_TOKENS of them, recording how long
    each prompt was before truncation.

    The same token ids as ComfyUI's MOSSImageTokenizer for any caption that is not itself a chat
    transcript (ComfyUI passes text starting with <|im_start|> through as it is); ComfyUI does not
    truncate, the reference pipeline does (max_sequence_length 4096).
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        def tokenizer(*args, **kwargs):
            return qwen35.Qwen35Tokenizer(*args, **kwargs, embedding_size=TEXT_DIM, embedding_key='qwen35_9b')

        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name='qwen35_9b', tokenizer=tokenizer)
        self.last_untruncated_length = None

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        # The chat template trims the prompt (render_content(...)|trim, which is str.strip()).
        text = f'<|im_start|>user\n{text.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n'
        tokens = super().tokenize_with_weights(text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        self.last_untruncated_length = len(tokens['qwen35_9b'][0])
        tokens['qwen35_9b'] = [tokens['qwen35_9b'][0][:MAX_TEXT_TOKENS]]
        return tokens


class Qwen35TextOnly(qwen35.Qwen35):
    """Qwen3.5-9B without the vision tower: MOSS-Image only reads text, and the tower is ~0.9 GB."""
    model_type = 'qwen35_9b'

    def __init__(self, config_dict, dtype, device, operations):
        nn.Module.__init__(self)
        config = qwen35._make_config(self.model_type, config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = qwen35.Qwen35Transformer(config, device=device, dtype=dtype, ops=operations)
        self.mtp = None
        self.visual = None
        self.dtype = dtype

    def preprocess_embed(self, embed, device):
        if embed['type'] == 'image':
            raise ValueError('The MOSS-Image text encoder is loaded without its vision tower')
        return None, None


class MossImageClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device='cpu', dtype=None, model_options={}):
        super().__init__(
            device=device, dtype=dtype, layer='last', textmodel_json_config={'lm_head': False},
            model_class=Qwen35TextOnly, special_tokens={'pad': 248044}, enable_attention_masks=True,
            return_attention_masks=True, model_options=model_options)


def text_encoder_class(dtype_llama=None, llama_quantization_metadata=None):
    class MossImageTEModel(comfy.sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = {**model_options, 'quantization_metadata': llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype, name='qwen35_9b', clip_model=MossImageClipModel,
                             model_options=model_options)
    return MossImageTEModel


def load_text_encoder(path):
    state, metadata = read_component(path, 'model')
    state, metadata = comfy.utils.convert_old_quants(state, model_prefix='', metadata=metadata)
    state = text_encoder_state(state)
    target = comfy.supported_models_base.ClipTarget(MossImageTokenizer, text_encoder_class(**comfy.sd.llama_detect([state])))
    return comfy.sd.CLIP(target, tokenizer_data={}, parameters=comfy.utils.calculate_parameters(state),
                         state_dict=[state], model_options={}, disable_dynamic=True)


# ---- noise levels ----------------------------------------------------------------------------------

def training_grid(total=NUM_TRAIN_TIMESTEPS, shift=1.0, dtype=torch.float64):
    """Noise levels of the training grid, ascending: sigma_j for j = 1..total.

    The reference scheduler's grid(): linspace(1, shift/(total+shift-1), total), shifted. At
    shift 1 that is j/total, which is written out exactly.
    """
    if shift == 1.0:
        return torch.arange(1, total + 1, dtype=dtype) / total
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, total, dtype=dtype)
    return (shift * values / (1 + (shift - 1) * values)).flip(0)


def inference_sigmas(steps, shift, training_shift=1.0, total=NUM_TRAIN_TIMESTEPS):
    """MOSSImageScheduler.set_timesteps: a shifted schedule snapped to the training grid (fp32)."""
    if steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError('Steps and shifts must be positive.')

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.unique_consecutive(training[nearest])


class NoiseLevelSampler:
    """The discrete noise-level distribution of the report's final pretraining interval (eq. 4).

    With Z ~ N(mu, s^2) and S = 1 - sigmoid(Z), level j of the T-point grid has probability
        p_j = (1 - rho) * (F(a j/T) - F(a (j-1)/T)) / F(a) + rho / T,   F(x) = Pr(S <= x),
    i.e. a logit-normal truncated at the cutoff a and stretched over the whole grid, mixed with a
    discrete uniform. timestep_sample_method = 'uniform' is the discrete uniform alone. min_t and
    max_t restrict the support to grid levels inside [min_t, max_t] and renormalise, so the shape
    of the distribution is kept.
    """

    def __init__(self, method, logit_mean, logit_std, sigma_cutoff, uniform_fraction, shift, min_t, max_t,
                 total=NUM_TRAIN_TIMESTEPS):
        if method not in ('logit_normal', 'uniform'):
            raise ValueError(f"timestep_sample_method must be 'logit_normal' or 'uniform', got {method!r}")
        if not logit_std > 0:
            raise ValueError(f'logit_std must be positive, got {logit_std}')
        if not 0 < sigma_cutoff <= 1:
            raise ValueError(f'sigma_cutoff must be in (0, 1], got {sigma_cutoff}')
        if not 0 <= uniform_fraction <= 1:
            raise ValueError(f'uniform_fraction must be in [0, 1], got {uniform_fraction}')
        if not shift > 0:
            raise ValueError(f'shift must be positive, got {shift}')
        if not 0 <= min_t <= max_t <= 1:
            raise ValueError(f'Need 0 <= min_t <= max_t <= 1, got min_t={min_t}, max_t={max_t}')
        self.sigmas = training_grid(total, shift)
        if method == 'uniform':
            p = torch.full((total,), 1.0 / total, dtype=torch.float64)
        else:
            edges = sigma_cutoff * torch.arange(total + 1, dtype=torch.float64) / total
            with torch.no_grad():
                cdf = 1 - torch.special.ndtr((torch.log((1 - edges) / edges) - logit_mean) / logit_std)
            p = (cdf[1:] - cdf[:-1]) / cdf[-1]
            p = (1 - uniform_fraction) * p + uniform_fraction / total
        keep = (self.sigmas >= min_t - 1e-12) & (self.sigmas <= max_t + 1e-12)
        if not keep.any():
            raise ValueError(f'min_t={min_t}, max_t={max_t} leave no noise level of the training grid')
        p = torch.where(keep, p, torch.zeros_like(p))
        if p.sum() <= 0:
            raise ValueError(f'min_t={min_t}, max_t={max_t} only keep noise levels of probability zero')
        self.probabilities = p / p.sum()
        self.cdf = torch.cumsum(self.probabilities, 0)
        self.last = int(torch.nonzero(self.probabilities > 0).max())

    def sample(self, bs, quantile=None, generator=None):
        if quantile is not None:
            u = torch.full((bs,), float(quantile), dtype=torch.float64)
        else:
            u = torch.rand(bs, dtype=torch.float64, generator=generator)
        # right=True: u lands in the bin whose cdf interval [cdf[i-1], cdf[i]) contains it, which
        # skips the empty bins min_t / max_t created.
        idx = torch.searchsorted(self.cdf, u, right=True).clamp(max=self.last)
        return self.sigmas[idx].float()


# ---- the pipeline ----------------------------------------------------------------------------------

class LayerState:
    """What every pipeline layer shares and is not a module: the compute dtype, the layers to skip
    while sampling, and a few RoPE tables (they depend only on the image grid and text length).

    rope is the model's rope method; its fp32 tables stay fp32, as in ComfyUI's forward.
    """

    def __init__(self, rope, compute_dtype):
        self.rope_fn = rope
        self.compute_dtype = compute_dtype
        self.skip_layers = frozenset()
        self._rope = OrderedDict()

    def rope(self, height, width, text_length, device):
        key = (height, width, text_length, str(device))
        if key not in self._rope:
            with torch.no_grad():
                table = self.rope_fn(height, width, text_length, device)
            self._rope[key] = table
            while len(self._rope) > 8:
                self._rope.popitem(last=False)
        return self._rope[key]


class MossImagePipeline(ComfyPipeline):
    name = 'moss_image'
    checkpointable_layers = ['InitialLayer', 'TransformerLayer']
    adapter_target_modules = ['Block']
    keep_in_high_precision = KEEP_IN_HIGH_PRECISION
    spatial_compression = 16
    channels = LATENT_CHANNELS
    pixels_round_to_multiple = 16
    vae_config_keys = ('diffusers_path', 'vae_path')
    # JSON layout captions in MOSS-Image's prompt schema get their boxes rewritten per bucket:
    # see utils/layout_captions.py.
    layout_caption_format = 'moss_image'

    def __init__(self, config):
        # ComfyPipeline.__init__ is not called: it expects `vae` and `text_encoders` entries, and
        # CommonPipeline.__init__ builds a CUDA scheduler this model does not use.
        self.config = config
        self.model_config = config['model']
        self.dtype = _dtype(self.model_config.get('dtype', 'bfloat16'), 'dtype')
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('MOSS-Image runs in bfloat16 or float32; set dtype to one of them')
        self.model_config['dtype'] = self.dtype
        self._check_keys()
        self.latent_format = comfy.latent_formats.Flux2()
        self.text_dim = TEXT_DIM

        self.transformer_path = self._component('transformer_path', 'transformer')
        self.vae_path = self._component('vae_path', 'vae')
        self.text_encoder_path = self._component('text_encoder_path', 'text_encoder')
        if self.model_config.get('diffusers_path') and is_main_process():
            print(f'MOSS-Image: transformer from {self.transformer_path}, text encoder from '
                  f'{self.text_encoder_path}, VAE from {self.vae_path}')

        if adapter_targets := self.model_config.get('adapter_target_modules', None):
            self.adapter_target_modules = list(adapter_targets)
        self.on_caption_overflow = self.model_config.get('on_caption_overflow', 'warn')
        if self.on_caption_overflow not in ('warn', 'silent', 'error'):
            raise ValueError(f"on_caption_overflow must be 'warn', 'silent' or 'error', got {self.on_caption_overflow!r}")

        noise = {key: self.model_config.get(key, default) for key, default in NOISE_DEFAULTS.items()}
        self.noise_sampler = NoiseLevelSampler(
            noise['timestep_sample_method'], float(noise['logit_mean']), float(noise['logit_std']),
            float(noise['sigma_cutoff']), float(noise['uniform_fraction']), float(noise['shift']),
            float(noise['min_t']), float(noise['max_t']))

        def load_vae():
            vae = comfy.sd.VAE(sd=read_component(self.vae_path, 'diffusion_pytorch_model')[0], dtype=torch.float32)
            vae.throw_exception_if_invalid()
            return vae

        self.vae = ModelWrapper(load_vae)
        self.text_encoders = [ModelWrapper(lambda: load_text_encoder(self.text_encoder_path))]
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.diffusion_model = None
        self.layer_state = None
        self.conds = self.unconds = None

    def _check_keys(self):
        mc = self.model_config
        refused = {
            'flux_shift': 'MOSS-Image trains at a fixed shift; set shift instead',
            'sigmoid_scale': 'logit_std sets the spread of the logit-normal noise sampler',
            'vae': 'use vae_path (or diffusers_path)',
            'text_encoders': 'use text_encoder_path (or diffusers_path)',
            'diffusion_model': 'use transformer_path (or diffusers_path)',
            'llm_path': 'use text_encoder_path (or diffusers_path)',
            'text_encoder_fp8': 'a quantized text encoder is loaded from its own file; cached embeddings free it anyway',
            'transformer_dtype': 'use diffusion_model_dtype',
        }
        bad = [key for key in refused if key in mc]
        if bad:
            raise ValueError('Config keys not used by MOSS-Image: '
                             + '; '.join(f'{key} ({refused[key]})' for key in bad))
        if mc.get('cache_text_embeddings', True) is not True:
            raise ValueError(
                'MOSS-Image needs cache_text_embeddings = true: its 9B text encoder is run once '
                'per caption while caching and freed before training.')
        method = mc.get('timestep_sample_method', NOISE_DEFAULTS['timestep_sample_method'])
        if method == 'uniform':
            unused = [key for key in ('logit_mean', 'logit_std', 'sigma_cutoff', 'uniform_fraction') if key in mc]
            if unused:
                raise ValueError(f"{', '.join(unused)} only apply to timestep_sample_method = 'logit_normal'")
        if (dmd := mc.get('diffusion_model_dtype')) is not None:
            dmd = _dtype(dmd, 'diffusion_model_dtype')
            mc['diffusion_model_dtype'] = dmd
            if dmd.itemsize == 1 and 'adapter' not in self.config:
                raise ValueError('diffusion_model_dtype is fp8, which needs an [adapter]: a full fine-tune '
                                 'cannot update fp8 weights.')

    def _component(self, key, subfolder):
        if path := self.model_config.get(key):
            return str(path)
        root = self.model_config.get('diffusers_path')
        if not root:
            raise ValueError(f'Set diffusers_path, or {key} for the {subfolder}')
        path = Path(root) / subfolder
        if not path.is_dir():
            raise FileNotFoundError(f'{path} does not exist. Set {key} if the {subfolder} lives elsewhere.')
        return str(path)

    # ---- identities ------------------------------------------------------------------------------

    def text_encoder_cache_key(self, i):
        return f'moss_image|{self.text_encoder_path}|qwen35_9b|last|{MAX_TEXT_TOKENS}|trimmed-v1|prompt-strip-v1'

    # ---- models ----------------------------------------------------------------------------------

    def load_diffusion_model(self):
        path = self.transformer_path
        declared = read_transformer_config(path) if Path(path).is_dir() else {}
        state, metadata = read_component(path, 'diffusion_pytorch_model')
        load_device = comfy.model_management.get_torch_device()
        patcher = load_transformer_patcher(state, metadata, self.dtype, load_device=load_device, declared=declared)
        del state

        merging_adapter = False
        for adapter_path in self.model_config.get('merge_adapters', []):
            merging_adapter = True
            if is_main_process():
                print(f'Merging adapter {adapter_path}')
            adapter_path = Path(adapter_path)
            if adapter_path.is_dir():
                adapter_path = adapter_path / 'adapter_model.safetensors'
            patcher = merge_adapter(patcher, adapter_path)

        # Every ComfyUI layer then casts its weight to the compute dtype per call (differentiably;
        # the weights of a diffusion_model_dtype fp8 model stay fp8 in memory).
        patcher.set_model_compute_dtype(self.dtype)
        with torch.no_grad():
            # Patched where the weights are, on the CPU: merging on the GPU in lowvram mode, as
            # ComfyPipeline does, would leave LowVramPatch weight functions on the layers it
            # did not fit, re-applying the adapter on every call.
            patcher.patch_model()
        self.diffusion_model = patcher.model.diffusion_model
        del patcher

        if diffusion_model_dtype := self.model_config.get('diffusion_model_dtype', None):
            self.dequantize(self.diffusion_model, diffusion_model_dtype)

        self.diffusion_model.train()
        for name, p in self.diffusion_model.named_parameters():
            p.original_name = name
            p.requires_grad_(True)
        if self.patch_quantized_modules(self.diffusion_model) and 'adapter' not in self.config:
            raise ValueError(f'{self.transformer_path} is a quantized checkpoint: you are trying to full finetune '
                             'a quantized model, which will not work. Train an [adapter], or load an unquantized checkpoint.')
        self.layer_state = LayerState(self.diffusion_model.rope, self.dtype)

    def get_call_vae_fn(self, vae):
        def fn(images):
            # fp32 pixels into the fp32 VAE: the reference encodes without a bf16 round trip.
            return {'latents': self.vae_encode(images.to('cuda', torch.float32))}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        @torch.inference_mode()
        def fn(captions, is_video):
            embeds = []
            for caption in captions:
                embeds.append(self.encode_prompt(text_encoder, caption))
            # Trimmed to the real tokens; get_conds pads a batch back.
            return {'prompt_embeds': embeds}
        return fn

    def encode_prompt(self, text_encoder, caption):
        # One prompt per forward and no padding, as the reference does: Qwen3.5's gated-delta
        # layers would see the pad tokens of a padded batch.
        tokens = text_encoder.tokenize(caption)
        tokenizer = text_encoder.tokenizer
        if tokenizer.last_untruncated_length > MAX_TEXT_TOKENS:
            message = (f'Caption is {tokenizer.last_untruncated_length} tokens with the chat template, '
                       f'truncated to {MAX_TEXT_TOKENS}: {caption[:120]!r}...')
            if self.on_caption_overflow == 'error':
                raise ValueError(message)
            if self.on_caption_overflow == 'warn':
                print(f'WARNING: {message}')
        out = text_encoder.encode_from_tokens(tokens, return_dict=True)
        cond = out['cond']
        mask = out.get('attention_mask')
        if mask is not None and not bool(mask.bool().all()):
            raise RuntimeError('The text encoder padded a single prompt; expected an all-ones mask')
        if cond.shape[0] != 1 or cond.shape[-1] != self.text_dim:
            raise RuntimeError(f'Expected one [1, L, {self.text_dim}] text embedding, got {tuple(cond.shape)}')
        return cond[0].to(self.dtype)

    def get_conds(self, inputs):
        embeds = inputs['prompt_embeds']
        if torch.is_tensor(embeds):
            embeds = list(embeds)
        length = max(e.shape[0] for e in embeds)
        text = embeds[0].new_zeros((len(embeds), length, embeds[0].shape[-1]))
        mask = torch.zeros((len(embeds), length), dtype=torch.bool)
        for i, e in enumerate(embeds):
            if e.shape[0] == 0:
                raise ValueError('An empty text embedding reached MOSS-Image; every prompt has at least the template')
            text[i, :e.shape[0]] = e
            mask[i, :e.shape[0]] = True
        return text, mask

    # ---- adapters and saving ---------------------------------------------------------------------

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError('Use merge_adapters in [model] to merge adapters into MOSS-Image')

    def save_model(self, save_dir, sd):
        """diffusers transformer layout: config.json + diffusion_pytorch_model.safetensors.

        The same file is a ComfyUI diffusion model (Load Diffusion Model reads it) and a
        transformer_path for this pipeline, and the folder loads with diffusers, whose model keeps
        the RoPE frequencies as a buffer (rope.bands).
        """
        sd = dict(sd)
        if 'rope.bands' not in sd:
            sd['rope.bands'] = default_rope_bands(sd['layers.0.q_norm.weight'].shape[0])
        unet_config = transformer_unet_config(sd)
        config = {'_class_name': 'MOSSImageTransformer', **unet_config}
        safetensors.torch.save_file(sd, Path(save_dir) / 'diffusion_pytorch_model.safetensors', metadata={'format': 'pt'})
        (Path(save_dir) / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    # ---- data ------------------------------------------------------------------------------------

    def model_specific_dataset_config_validation(self, dataset_config):
        for directory in dataset_config.get('directory', []):
            if directory.get('control_path', None) is not None:
                raise NotImplementedError('MOSS-Image is a text-to-image model; control/edit datasets are not supported')

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        text, text_mask = self.get_conds(inputs)
        bs, _, h, w = latents.shape
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask.float(), size=(h, w), mode='nearest-exact')
        sigma = self.noise_sampler.sample(bs, timestep_quantile).to(latents.device)
        noise = torch.randn_like(latents)
        s = sigma.view(-1, 1, 1, 1)
        noisy = (1 - s) * latents + s * noise
        target = noise - latents
        # ComfyUI's MOSS-Image takes sigma itself; its timestep embedding scales it by 1000.
        return (noisy, sigma, text, text_mask), (target, mask)

    # ---- pipeline layers -------------------------------------------------------------------------

    def to_layers(self):
        model = self.diffusion_model
        state = self.layer_state
        layers = [InitialLayer(model, state)]
        for i, block in enumerate(model.layers):
            layers.append(TransformerLayer(block, i, self.offloader, state, len(model.layers)))
        layers.append(FinalLayer(model))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        model = self.diffusion_model
        blocks = model.layers
        num_blocks = len(blocks)
        assert blocks_to_swap <= num_blocks - 2, (
            f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.')
        self.offloader = ModelOffloader(
            'Block', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'),
            self.config['reentrant_activation_checkpointing'])
        model.layers = None
        model.to('cuda')
        model.layers = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()

    # ---- sampling (--test_sample) ----------------------------------------------------------------

    def sample_setting(self, key):
        value = self.model_config.get(key, SAMPLE_DEFAULTS[key])
        return list(value) if isinstance(value, (list, tuple)) else value

    @torch.no_grad()
    def prepare_sample_test(self, prompt, negative_prompt=None, cfg=None, device='cuda'):
        """Encode the preview prompts. The guidance comes from the sample_* keys (the reference
        pipeline's settings by default), not from `cfg`: train.py passes cfg=1 for every model."""
        if negative_prompt is None:
            negative_prompt = self.sample_setting('sample_negative_prompt')
        self.sample_cfg = float(self.sample_setting('sample_cfg'))
        te = self.text_encoders[0]
        te.load_model_if_needed()
        embeds = [self.encode_prompt(te, prompt)]
        self.conds = tuple(t.to(device) for t in self.get_conds({'prompt_embeds': embeds}))
        if self.sample_cfg != 1:
            embeds = [self.encode_prompt(te, negative_prompt)]
            self.unconds = tuple(t.to(device) for t in self.get_conds({'prompt_embeds': embeds}))
        comfy.model_management.unload_all_models()

    def _skip_set(self, key):
        layers = self.sample_setting(key) or []
        num_layers = len(self.diffusion_model.layers)
        for layer in layers:
            if type(layer) is not int or not 0 <= layer < num_layers:
                raise ValueError(f'{key} entries must be block indices in [0, {num_layers}), got {layer!r}')
        return frozenset(layers)

    def _predict(self, x, t, conds, skip=frozenset()):
        self.layer_state.skip_layers = skip
        try:
            return self.pipeline_model((x, t, *conds)).float()
        finally:
            self.layer_state.skip_layers = frozenset()

    @torch.no_grad()
    def sample(self, w=1024, h=1024, steps=None, seed=None, noise=None, decode=True):
        """One image with the reference pipeline's sampler: Euler over MOSSImageScheduler's snapped
        sigmas, CFG in fp32 whose negative branch skips sample_uncond_skip_layers, and optional
        skip-layer guidance, active for sample_slg_start <= step/steps < sample_slg_stop."""
        steps = int(steps or self.sample_setting('sample_steps'))
        seed = self.sample_setting('sample_seed') if seed is None else seed
        cfg = self.sample_cfg
        uncond_skip = self._skip_set('sample_uncond_skip_layers')
        slg_layers = self._skip_set('sample_slg_layers')
        slg_scale = float(self.sample_setting('sample_slg_scale'))
        slg_start = float(self.sample_setting('sample_slg_start'))
        slg_stop = float(self.sample_setting('sample_slg_stop'))
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError('Need 0 <= sample_slg_start <= sample_slg_stop <= 1')
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError('sample_slg_scale must be finite and nonnegative')
        h, w = h // 16 * 16, w // 16 * 16
        device = self.conds[0].device
        if noise is None:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, LATENT_CHANNELS, h // 16, w // 16), device=device, dtype=torch.float32,
                                generator=generator)
        x = noise.to(device=device, dtype=torch.float32)
        sigmas = inference_sigmas(steps, float(self.sample_setting('sample_shift'))).to(device)
        num_steps = len(sigmas)
        for i in tqdm(range(num_steps), desc='Sampling'):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < num_steps else torch.zeros_like(sigma)
            t = sigma.view(1).float()
            active = slg_start <= i / num_steps < slg_stop
            conditional = self._predict(x, t, self.conds)
            prediction = conditional
            if cfg != 1:
                uncond = self._predict(x, t, self.unconds, uncond_skip if active else frozenset())
                prediction = uncond + cfg * (conditional - uncond)
            if slg_layers and slg_scale > 0 and active:
                skipped = self._predict(x, t, self.conds, slg_layers)
                if cfg == 1:
                    prediction = skipped
                prediction = prediction + slg_scale * (conditional - skipped)
            x = x + (sigma_next - sigma) * prediction
        if not decode:
            return x
        self.vae.load_model_if_needed()
        img = self.vae_decode(x)
        comfy.model_management.unload_all_models()
        return img


def _length_groups(lengths):
    """{text length: [sample indices]} in first-seen order."""
    groups = OrderedDict()
    for i, n in enumerate(lengths):
        groups.setdefault(n, []).append(i)
    return groups


def _assemble(pieces, batch, total):
    """Samples back into one right-padded (batch, total, dim) tensor, in batch order."""
    rows = [None] * batch
    for indices, values in pieces:
        for j, i in enumerate(indices):
            rows[i] = values[j:j + 1]
    return torch.cat([F.pad(row, (0, 0, 0, total - row.shape[1])) for row in rows])


def _tie(out, *unused):
    """`out` with a zero-weight dependency on tensors this layer does not otherwise consume.

    Under pipeline parallelism every floating tensor a stage receives is given requires_grad, and
    DeepSpeed asserts each one gets a gradient to send back; the zero term gives it one.
    """
    for t in unused:
        if torch.is_floating_point(t) and t.requires_grad:
            out = out + t.reshape(-1)[:1].sum().to(out.dtype) * 0
    return out


class InitialLayer(nn.Module):
    """MOSSImage._forward up to the blocks: the time embedding and both modulations, the image
    tokens, and the text refined at each sample's own length."""

    def __init__(self, model, state):
        super().__init__()
        self.video_embedder = model.video_embedder
        self.text_embedder = model.text_embedder
        self.time_embedding = model.time_embedding
        self.shared_block_modulation = model.shared_block_modulation
        self.video_head_modulation = model.video_head_modulation
        self.latent_channels = model.latent_channels
        self.state = state

    @torch.compiler.disable
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, sigma, text, text_mask = inputs
        batch, channels, height, width = x.shape
        if channels != self.latent_channels:
            raise ValueError(f'Expected {self.latent_channels} latent channels, got {channels}')
        device = x.device
        dtype = self.state.compute_dtype
        text_mask = text_mask.bool()
        lengths = text_mask.sum(1)
        expected = torch.arange(text_mask.shape[1], device=device)[None] < lengths[:, None]
        if not torch.equal(text_mask, expected):
            raise ValueError('The text mask must mark a prefix of each row (right padding)')
        lengths_list = lengths.tolist()
        if min(lengths_list) == 0:
            raise ValueError('Each sample needs at least one valid text token.')

        # ComfyUI hands the model its inputs in the compute dtype (BaseModel._apply_model).
        sigma = sigma.to(device=device, dtype=torch.float32).reshape(-1).expand(batch)
        vec = self.time_embedding(timestep_embedding(sigma, FREQ_DIM).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1))
        head = self.video_head_modulation(vec).unflatten(-1, (2, -1))

        hw = height * width
        video = self.video_embedder(x.to(dtype).flatten(2).transpose(1, 2))
        text = text.to(dtype)
        pieces = []
        for n, indices in _length_groups(lengths_list).items():
            pe = self.state.rope(height, width, n, device)
            refined = self.text_embedder(text[indices, :n], pe[:, hw:])
            pieces.append((indices, torch.cat((video[indices], refined), 1)))
        hidden = _assemble(pieces, batch, hw + max(lengths_list))
        sizes = torch.tensor([height, width], device=device)
        return make_contiguous(hidden, mod, head, lengths, sizes)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader, state, num_blocks):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader
        self.state = state
        self.transformer_options = {'total_blocks': num_blocks, 'block_type': 'single', 'block_index': block_idx}

    def forward(self, inputs):
        hidden, mod, head, lengths, sizes = inputs
        self.offloader.wait_for_block(self.block_idx)
        if self.block_idx not in self.state.skip_layers:
            height, width = sizes.tolist()
            hw = height * width
            lengths_list = lengths.tolist()
            batch, total = hidden.shape[:2]
            groups = _length_groups(lengths_list)
            pieces = []
            for n, indices in groups.items():
                pe = self.state.rope(height, width, n, hidden.device)
                rows = hidden[:, :hw + n] if len(indices) == batch else hidden[indices, :hw + n]
                rows_mod = mod if len(indices) == batch else mod[indices]
                # MOSSImage._forward's mod: six (batch, 1, dim) tensors.
                out = self.block(rows, rows_mod.unsqueeze(2).unbind(1), pe, self.transformer_options)
                pieces.append((indices, out))
            if len(groups) == 1 and hw + lengths_list[0] == total:
                hidden = pieces[0][1]
            else:
                hidden = _assemble(pieces, batch, total)
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(hidden, mod, head, lengths, sizes)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_norm_video = model.final_norm_video
        self.final_linear_video = model.final_linear_video

    def forward(self, inputs):
        hidden, mod, head, lengths, sizes = inputs
        height, width = sizes.tolist()
        shift, scale = head.unsqueeze(2).unbind(1)
        x = modulated_rms_norm(hidden[:, :height * width], self.final_norm_video, scale, shift)
        x = self.final_linear_video(x)
        out = x.transpose(1, 2).reshape(x.shape[0], -1, height, width)
        return _tie(out, mod)
