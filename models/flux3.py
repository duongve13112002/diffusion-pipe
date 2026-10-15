"""FLUX.3 Klein Base (black-forest-labs/flux-3-image-klein-base) through ComfyUI.

The model has three token streams: the image being generated, the reference images
("image_cond", up to ten of them) and the text. Each stream first runs through its own
five blocks, attending only to itself, then the joint sequence [image, references, text]
runs through 28 shared blocks in which every span is modulated by its own adaLN. The
image is modulated at t, references and text at t = 0.

ComfyUI's forward is written for inference (in-place updates, a preallocated joint
sequence, fused kernels), so training runs its own functional forward over the same
modules. It reproduces ComfyUI's arithmetic, so a LoRA trained here behaves the same in
ComfyUI. Two things ComfyUI does not need are added:

  - Batches of captions of different lengths. FLUX.3 attends to every text token,
    padding included: ComfyUI pads each prompt to a multiple of 80 tokens and the DiT
    sees those pads. A batch is padded further to its longest caption; the extra
    positions are masked out of attention, so each sample sees exactly what it would see
    alone.
  - Training the edit branch without touching text-to-image (edit_isolation):
      'image_cond'        trains only the modules that exist for the references (their
                          input projection, their five blocks, their modulations). A
                          text-to-image forward never reaches them, so it stays
                          bit-identical, and the adapter is an ordinary LoRA.
      'reference_tokens'  also trains the 28 shared blocks, but applies the adapter only
                          to the rows of the reference tokens. Without references there
                          are no such rows, so text-to-image is unchanged. ComfyUI needs
                          tools/comfyui_flux3_reference_lora (a ComfyUI node) to apply it
                          this way.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch

from models.base import ComfyPipeline, ModelWrapper, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, is_main_process, empty_cuda_cache
from utils.offloading import ModelOffloader
import comfy.latent_formats
import comfy.model_management
import comfy.sd
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import _apply_rope1


MAX_REFERENCES = 10
TEXT_PAD_ID = 151643
TEXT_PAD_MULTIPLE = 80
TEXT_CLIP_NAME = 'qwen3vl_4b'

EDIT_ISOLATION_MODES = ('none', 'image_cond', 'reference_tokens')
# Every module a forward without reference images never runs.
IMAGE_COND_PREFIXES = (
    'emb_in.image_cond.',
    'content_mode_blocks.image_cond.',
    'early_stream_modulations.image_cond.',
    'single_stream_modulations.image_cond.',
)
# Saved key prefix of the adapter weights that apply only to reference-token rows. Not
# 'diffusion_model.', so ComfyUI's ordinary LoRA loader skips them (and says so) instead of
# applying them to every token.
REFERENCE_ROWS_PREFIX = 'flux3_ref_rows.'
ADAPTER_METADATA_KEY = 'flux3_edit_isolation'

# [model] keys this model reads. Anything else is refused by name.
MODEL_KEYS = {
    'type', 'diffusion_model', 'vae', 'text_encoders', 'dtype', 'diffusion_model_dtype',
    'merge_adapters', 'cache_text_embeddings', 'timestep_sample_method', 'sigmoid_scale',
    'shift', 'flux_shift', 'edit_isolation',
    # train.py writes guidance = 1.0 into every config; FLUX.3 has no guidance embedding.
    'guidance',
}


def is_image_cond_name(name):
    return name.startswith(IMAGE_COND_PREFIXES)


def control_paths_of(directory_config):
    paths = directory_config.get('control_path', None)
    if paths is None:
        return []
    return [paths] if isinstance(paths, (str, Path)) else list(paths)


def text_embedding_files(path):
    """The safetensors files of a text encoder: one file, or every shard in a directory
    (the Hugging Face layout black-forest-labs ships under text_encoder/)."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.safetensors'))
        if not files:
            raise RuntimeError(f'No .safetensors files in text encoder directory {path}')
        return files
    if not path.is_file():
        raise RuntimeError(f'Text encoder path does not exist: {path}')
    return [path]


def unfuse_qkv(diffusion_model):
    """Split fused qkv projections (ComfyUI's repacked checkpoints) into q/k/v, the layout of
    BFL's checkpoint. LoRA keys then name q_proj/k_proj/v_proj, which ComfyUI maps onto
    either layout."""
    unfused = 0
    for name, module in diffusion_model.named_modules():
        if not getattr(module, 'fused_qkv', False):
            continue
        qkv = module.qkv
        weight = qkv.weight
        if weight.__class__.__name__ == 'QuantizedTensor':
            raise ValueError(
                f'{name}.qkv is a quantized fused projection. Set diffusion_model_dtype to dequantize '
                'it, or train from the BFL checkpoint (separate q/k/v).')
        hidden = weight.shape[1]
        for i, proj in enumerate(('q_proj', 'k_proj', 'v_proj')):
            linear = type(qkv)(hidden, hidden, bias=False, device=weight.device, dtype=weight.dtype)
            linear.weight = nn.Parameter(weight.detach()[i * hidden:(i + 1) * hidden].clone(),
                                         requires_grad=weight.requires_grad)
            for attr in ('comfy_cast_weights',):
                if hasattr(qkv, attr):
                    setattr(linear, attr, getattr(qkv, attr))
            setattr(module, proj, linear)
        del module.qkv
        module.fused_qkv = False
        unfused += 1
    return unfused


class Flux3Pipeline(ComfyPipeline):
    name = 'flux3'
    checkpointable_layers = ['EarlyLayer', 'SingleLayer']
    adapter_target_modules = ['Block']
    keep_in_high_precision = ['emb_in', 'txt_in', 'time_in', 'final_layer', 'stream_modulations']
    spatial_compression = 16
    channels = 128
    max_control_images = MAX_REFERENCES
    # The text encoder (when not cached) is held outside the pipeline modules, see
    # TextEncoderLayer, so block swapping cannot strand it on the CPU.
    block_swap_supports_uncached_text_embeddings = True
    vae_config_keys = ('vae',)

    def __init__(self, config):
        model_config = config['model']
        # 20480 values per text token (~6.5 MB per 160-token caption in bf16): encoding on the
        # fly is the default, as for Iris. train.py reads this key with a default of True, so
        # the default has to be written into the config.
        model_config.setdefault('cache_text_embeddings', False)
        unknown = sorted(set(model_config) - MODEL_KEYS)
        if unknown:
            raise ValueError(f'[model] keys not used by flux3: {", ".join(unknown)}. '
                             f'Accepted keys: {", ".join(sorted(MODEL_KEYS))}.')
        if float(model_config.get('guidance', 1.0)) != 1.0:
            raise ValueError('guidance is not used by flux3: Klein Base has no guidance embedding '
                             '(it is sampled with real CFG). Remove it.')
        if model_config.get('timestep_sample_method', 'logit_normal') not in ('logit_normal', 'uniform'):
            raise ValueError("timestep_sample_method must be 'logit_normal' or 'uniform', "
                             f"got {model_config['timestep_sample_method']!r}")
        if 'shift' in model_config and model_config.get('flux_shift', False):
            raise ValueError('Set shift or flux_shift, not both.')

        te_configs = model_config.get('text_encoders', None)
        if not te_configs or len(te_configs) != 1:
            raise ValueError("flux3 takes exactly one text encoder: text_encoders = [{path = '.../text_encoder', "
                             "type = 'flux3'}]")
        te_config = te_configs[0]
        if str(te_config.get('type', '')).lower() != 'flux3':
            raise ValueError(f"The flux3 text encoder needs type = 'flux3', got {te_config.get('type')!r}")
        unknown_te = sorted(set(te_config) - {'path', 'paths', 'type'})
        if unknown_te:
            raise ValueError(f'text_encoders keys not used by flux3: {", ".join(unknown_te)}')

        self.edit_isolation = model_config.get('edit_isolation', 'none')
        if self.edit_isolation not in EDIT_ISOLATION_MODES:
            raise ValueError(f'edit_isolation must be one of {EDIT_ISOLATION_MODES}, got {self.edit_isolation!r}')
        if self.edit_isolation == 'reference_tokens' and 'adapter' not in config:
            raise ValueError("edit_isolation = 'reference_tokens' restricts an adapter to the reference-token rows "
                             'of the shared blocks, which a full fine-tune cannot do. Add an [adapter], or use '
                             "edit_isolation = 'image_cond' to fine-tune only the reference modules.")
        if self.edit_isolation == 'image_cond' and config.get('pipeline_stages', 1) > 1:
            raise ValueError("edit_isolation = 'image_cond' trains only the reference-image modules, which all sit "
                             'in the first layers, so later pipeline stages would have nothing to train. Use '
                             'pipeline_stages = 1 (data parallel) instead.')

        super().__init__(config)
        self.cache_text_embeddings = self.model_config['cache_text_embeddings']
        self.latent_format = comfy.latent_formats.Flux2()

        paths = te_config.get('paths', te_config.get('path', None))
        if paths is None:
            raise ValueError('need text encoder path in config')
        if isinstance(paths, (str, Path)):
            paths = [paths]
        te_files = [f for p in paths for f in text_embedding_files(p)]

        def load_text_encoder():
            sd = {}
            for f in te_files:
                sd.update(comfy.utils.load_torch_file(str(f), safe_load=True))
            return comfy.sd.load_text_encoder_state_dicts([sd], clip_type=comfy.sd.CLIPType.FLUX3, disable_dynamic=True)

        self.text_encoders = [ModelWrapper(load_text_encoder)]

        self._train_blocks = None
        self.offloader_early = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    # ---- loading ---------------------------------------------------------------------------

    def load_diffusion_model(self):
        for adapter_path in self.model_config.get('merge_adapters', []):
            sd = safetensors.torch.load_file(adapter_path) if str(adapter_path).endswith('.safetensors') else {}
            if any(k.startswith(REFERENCE_ROWS_PREFIX) for k in sd):
                raise ValueError(f'{adapter_path} was trained with edit_isolation = reference_tokens: it applies '
                                 'only to reference-token rows, which merging into the weights cannot express.')
        super().load_diffusion_model()
        self.prepare_diffusion_model()
        if not self.cache_text_embeddings:
            # The VAE is only needed while caching latents, which is done by now. (With cached
            # text embeddings train.py calls free_vae_and_te, which also drops the text encoder.)
            self.vae = None
            comfy.model_management.unload_all_models()
            empty_cuda_cache()

    def prepare_diffusion_model(self):
        """Checks and layout changes on the loaded ComfyUI model, before any adapter."""
        diffusion_model = self.diffusion_model
        if not hasattr(diffusion_model, 'single_stream_modulations') or not hasattr(diffusion_model, 'content_mode_blocks'):
            raise ValueError(f'{self.model_config["diffusion_model"]} is not a FLUX.3 checkpoint')
        if getattr(diffusion_model, 'flow_map', False):
            raise ValueError(
                f'{self.model_config["diffusion_model"]} is the step-distilled FLUX.3 Klein (a flow map, '
                'ref_time_in). Train FLUX.3 Klein Base (flux-3-image-klein-base) instead; a LoRA trained on '
                'Base also loads on the distilled model.')
        if unfuse_qkv(diffusion_model) and is_main_process():
            print('flux3: split fused qkv projections into q_proj/k_proj/v_proj')
        for name, p in diffusion_model.named_parameters():
            p.original_name = name
            p.requires_grad_(True)
        if self.edit_isolation == 'image_cond' and 'adapter' not in self.config:
            # Full fine-tune of the reference modules only.
            for name, p in diffusion_model.named_parameters():
                p.requires_grad_(is_image_cond_name(name))

    def free_vae_and_te(self):
        if self.cache_text_embeddings:
            super().free_vae_and_te()

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return self.text_encoders
        return []

    # ---- adapters --------------------------------------------------------------------------

    def get_target_modules(self, target_model):
        # Restricted to the reference rows, the last shared block's adapter reaches the loss only
        # through the keys and values the image attends to: on q, the output and the MLP it would
        # never train, so it is left out.
        last = f'single_blocks.{len(target_model.single_blocks) - 1}.'
        untrainable = {last + linear for linear in ('q_proj', 'attn_out', 'mlp_in', 'mlp_out')}
        targets = set()
        for name, module in target_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.edit_isolation == 'image_cond':
                if is_image_cond_name(name + '.'):
                    targets.add(name)
            elif self.edit_isolation == 'reference_tokens':
                if is_image_cond_name(name + '.') or (name.startswith('single_blocks.') and name not in untrainable):
                    targets.add(name)
            elif name.startswith(('content_mode_blocks.', 'txt_mode_blocks.', 'single_blocks.')):
                targets.add(name)
        return sorted(targets)

    def adapter_rows_restricted(self):
        return self.edit_isolation == 'reference_tokens'

    def save_adapter(self, save_dir, sd):
        self.peft_config.save_pretrained(save_dir)
        out = {}
        for k, v in sd.items():
            if self.adapter_rows_restricted() and k.startswith('single_blocks.'):
                out[REFERENCE_ROWS_PREFIX + k] = v
            else:
                out['diffusion_model.' + k] = v
        metadata = {'format': 'pt', ADAPTER_METADATA_KEY: self.edit_isolation}
        safetensors.torch.save_file(out, save_dir / 'adapter_model.safetensors', metadata=metadata)

    def load_adapter_weights(self, adapter_path):
        for f in Path(adapter_path).glob('*.safetensors'):
            with safetensors.safe_open(str(f), 'pt') as handle:
                trained = (handle.metadata() or {}).get(ADAPTER_METADATA_KEY, None)
                rows = any(k.startswith(REFERENCE_ROWS_PREFIX) for k in handle.keys())
            if trained is None:
                trained = 'reference_tokens' if rows else None
            if trained is not None and trained != self.edit_isolation:
                raise ValueError(f'{f} was trained with edit_isolation = {trained!r}, this run uses '
                                 f'{self.edit_isolation!r}. Set the same edit_isolation to continue training it.')

        def rename_key(k):
            return k[len(REFERENCE_ROWS_PREFIX):] if k.startswith(REFERENCE_ROWS_PREFIX) else k
        self.load_adapter_weights_into(self.diffusion_model, adapter_path, rename_key=rename_key)

    # ---- dataset ---------------------------------------------------------------------------

    def model_specific_dataset_config_validation(self, dataset_config):
        for directory in dataset_config.get('directory', []):
            paths = control_paths_of(directory)
            if len(paths) > MAX_REFERENCES:
                raise ValueError(f'{directory.get("path")}: FLUX.3 takes at most {MAX_REFERENCES} reference images, '
                                 f'control_path lists {len(paths)}')
            if self.edit_isolation != 'none' and not paths:
                raise ValueError(
                    f'{directory.get("path")} has no control_path, but edit_isolation = {self.edit_isolation!r} '
                    'trains only what reference images reach: a text-to-image sample would train nothing. '
                    "Remove the directory, give it control images, or set edit_isolation = 'none'.")

    def get_call_vae_fn(self, vae):
        def encode(images):
            return self.vae_encode(images.to(comfy.model_management.get_torch_device(), self.dtype))

        def fn(tensor, control_tensor=None):
            result = {'latents': encode(tensor)}
            if control_tensor is not None:
                # One reference is a tensor; several (a list of control_path directories) a list.
                controls = control_tensor if isinstance(control_tensor, (list, tuple)) else [control_tensor]
                for i, control in enumerate(controls):
                    result[control_latents_key(i)] = encode(control)
            return result
        return fn

    # ---- text ------------------------------------------------------------------------------

    def tokenize(self, captions):
        """Token ids of each caption as ComfyUI prompts FLUX.3 (chat template, right padded to a
        multiple of 80), padded further to the longest in the batch. Returns (ids, lengths):
        lengths are each caption's own padded length, the text the DiT sees for it."""
        te = self.text_encoders[0]
        te.load_model_if_needed()
        rows = []
        for caption in captions:
            tokens = te.tokenize(caption)[TEXT_CLIP_NAME]
            if len(tokens) != 1:
                raise RuntimeError(f'Expected one token sequence per caption, got {len(tokens)}')
            rows.append([int(t[0]) for t in tokens[0]])
        return pad_token_rows(rows)

    def encode_token_ids(self, ids, device=None):
        """FLUX.3 text features [B, L, 20480] for padded token ids, computed as ComfyUI does."""
        te = self.text_encoders[0]
        te.load_model_if_needed()
        inner = getattr(te.cond_stage_model, te.cond_stage_model.clip)
        if device is None:
            te.load_model()
            te.cond_stage_model.set_clip_options({'execution_device': te.patcher.load_device})
        try:
            z = inner.encode(ids.tolist())[0]
        finally:
            if device is None:
                te.cond_stage_model.reset_clip_options()
        return z.movedim(1, 2).flatten(2)

    def get_call_text_encoder_fn(self, text_encoder):
        @torch.inference_mode()
        def fn(captions, is_video):
            ids, lengths = self.tokenize(captions)
            embeds = self.encode_token_ids(ids).to(self.dtype)
            # Stored trimmed to each caption's own length; prepare_inputs pads the batch again.
            return {'text_embeds': [embeds[i, :n].cpu() for i, n in enumerate(lengths.tolist())]}
        return fn

    # ---- training inputs -------------------------------------------------------------------

    def sample_timesteps(self, bs, h, w, device, timestep_quantile=None):
        method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        else:
            dist = torch.distributions.uniform.Uniform(0, 1)
        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)
        if method == 'logit_normal':
            t = torch.sigmoid(t * self.model_config.get('sigmoid_scale', 1.0))
        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            # One token per latent pixel here (the latents are already 2x2-patched).
            mu = get_lin_function(y1=0.5, y2=1.15)(h * w)
            t = time_shift(mu, 1.0, t)
        return t

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        bs, c, h, w = latents.shape
        device = latents.device

        if self.cache_text_embeddings:
            text, text_lengths = pad_text_embeddings(inputs['text_embeds'])
        else:
            text, text_lengths = self.tokenize(inputs['caption'])

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')

        t = self.sample_timesteps(bs, h, w, device, timestep_quantile)
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        references = []
        for i in range(MAX_REFERENCES):
            key = control_latents_key(i)
            if key not in inputs:
                break
            references.append(inputs[key].float())

        return (noisy_latents, t, text, text_lengths, *references), (target, mask)

    # ---- layers ----------------------------------------------------------------------------

    def train_blocks(self):
        """Flux3Block wrappers around every ComfyUI Block, made once. Block swapping hooks these
        wrappers, so the same objects must reach both enable_block_swap and to_layers."""
        if self._train_blocks is None:
            dm = self.diffusion_model
            self._train_blocks = {
                'image': [Flux3Block(b) for b in dm.content_mode_blocks['image']],
                'image_cond': [Flux3Block(b) for b in dm.content_mode_blocks['image_cond']],
                'txt': [Flux3Block(b) for b in dm.txt_mode_blocks],
                'single': [Flux3Block(b, rows_restricted=self.adapter_rows_restricted()) for b in dm.single_blocks],
            }
        return self._train_blocks

    def early_offload_order(self):
        blocks = self.train_blocks()
        return blocks['image'] + blocks['txt']

    def to_layers(self):
        dm = self.diffusion_model
        blocks = self.train_blocks()
        text_encoder = None if self.cache_text_embeddings else self
        layers = [TextEncoderLayer(text_encoder, self.dtype), InitialLayer(dm, self.dtype)]
        early = self.early_offload_order()
        for mode in ('image', 'txt'):
            for i, block in enumerate(blocks[mode]):
                layers.append(EarlyLayer(mode, block, self.offloader_early, early.index(block)))
        for block in blocks['image_cond']:
            layers.append(EarlyLayer('image_cond', block, None, None))
        layers.append(ConcatLayer())
        for i, block in enumerate(blocks['single']):
            layers.append(SingleLayer(block, self.offloader_single, i))
        layers.append(FinalLayer(dm))
        return layers

    # ---- block swap ------------------------------------------------------------------------

    def enable_block_swap(self, blocks_to_swap):
        dm = self.diffusion_model
        blocks = self.train_blocks()
        early = self.early_offload_order()
        single = blocks['single']
        max_single = len(single) - 2
        max_early = len(early) - 2
        single_to_swap = min(blocks_to_swap, max_single)
        early_to_swap = blocks_to_swap - single_to_swap
        assert early_to_swap <= max_early, (
            f'Cannot swap {blocks_to_swap} blocks: at most {max_single} shared blocks and {max_early} '
            'image/text blocks can be swapped.')
        self.offloader_single = ModelOffloader(
            'Flux3SingleBlock', single, len(single), single_to_swap, True, torch.device('cuda'),
            self.config['reentrant_activation_checkpointing'])
        self.offloader_early = ModelOffloader(
            'Flux3EarlyBlock', early, len(early), early_to_swap, True, torch.device('cuda'),
            self.config['reentrant_activation_checkpointing'])
        single_blocks, content_blocks, txt_blocks = dm.single_blocks, dm.content_mode_blocks, dm.txt_mode_blocks
        dm.single_blocks = None
        dm.content_mode_blocks = None
        dm.txt_mode_blocks = None
        dm.to('cuda')
        content_blocks['image_cond'].to('cuda')
        dm.single_blocks = single_blocks
        dm.content_mode_blocks = content_blocks
        dm.txt_mode_blocks = txt_blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {single_to_swap} shared blocks and {early_to_swap} image/text blocks.')

    def prepare_block_swap_training(self):
        for offloader in (self.offloader_early, self.offloader_single):
            offloader.enable_block_swap()
            offloader.set_forward_only(False)
            offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        for offloader in (self.offloader_early, self.offloader_single):
            if disable_block_swap:
                offloader.disable_block_swap()
            offloader.set_forward_only(True)
            offloader.prepare_block_devices_before_forward()


def control_latents_key(i):
    return 'control_latents' if i == 0 else f'control_latents_{i + 1}'


def pad_token_rows(rows):
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.int64)
    ids = torch.full((len(rows), int(lengths.max())), TEXT_PAD_ID, dtype=torch.int64)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.int64)
    return ids, lengths


def pad_text_embeddings(embeds):
    if torch.is_tensor(embeds):
        embeds = list(embeds)
    lengths = torch.tensor([e.shape[0] for e in embeds], dtype=torch.int64)
    out = embeds[0].new_zeros((len(embeds), int(lengths.max()), embeds[0].shape[-1]))
    for i, e in enumerate(embeds):
        out[i, :e.shape[0]] = e
    return out, lengths


def text_key_mask(text_lengths, text_len, prefix_len=0):
    """Boolean key mask [B, prefix_len + text_len] that hides each sample's text positions past
    its own length, or None when no sample has any."""
    if bool((text_lengths >= text_len).all()):
        return None
    positions = torch.arange(text_len, device=text_lengths.device)
    mask = positions[None, :] < text_lengths[:, None]
    if prefix_len:
        mask = torch.cat([mask.new_ones(mask.shape[0], prefix_len), mask], dim=1)
    return mask


class ModulationOut:
    __slots__ = ('shift', 'scale', 'gate')

    def __init__(self, shift, scale, gate):
        self.shift, self.scale, self.gate = shift, scale, gate


def modulation_tensors(mod):
    return (mod.shift, mod.scale, mod.gate)


class Flux3Block(nn.Module):
    """Functional training forward of a ComfyUI FLUX.3 Block (comfy/ldm/flux3/model.py).

    Same arithmetic, without the in-place updates. With rows_restricted, adapters on the
    block's linears apply only to the rows in adapter_rows (the reference tokens); every
    other row goes through the frozen base layer.
    """

    def __init__(self, block, rows_restricted=False):
        super().__init__()
        self.block = block
        self.rows_restricted = rows_restricted

    def linear(self, module, x, adapter_rows):
        if not self.rows_restricted or not hasattr(module, 'get_base_layer'):
            return module(x)
        base = module.get_base_layer()
        if adapter_rows is None or adapter_rows[0] == adapter_rows[1]:
            return base(x)
        start, end = adapter_rows
        parts = []
        if start > 0:
            parts.append(base(x[:, :start]))
        parts.append(module(x[:, start:end]))
        if end < x.shape[1]:
            parts.append(base(x[:, end:]))
        return torch.cat(parts, dim=1)

    def forward(self, x, spans, pe, key_mask=None, adapter_rows=None):
        block = self.block
        B, N, _ = x.shape
        heads = block.num_heads
        normed = block.pre_norm(x)
        x_mod = torch.cat([torch.addcmul(mod.shift, normed[:, start:end], 1 + mod.scale)
                           for start, end, mod in spans], dim=1)

        q = self.linear(block.q_proj, x_mod, adapter_rows)
        k = self.linear(block.k_proj, x_mod, adapter_rows)
        v = self.linear(block.v_proj, x_mod, adapter_rows)
        mlp = self.linear(block.mlp_in, x_mod, adapter_rows)
        del x_mod
        q = _apply_rope1(block.norm.query_norm(q.view(B, N, heads, -1)), pe)
        k = _apply_rope1(block.norm.key_norm(k.view(B, N, heads, -1)), pe)
        v = v.view(B, N, heads, -1)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        gate, up = mlp.chunk(2, dim=-1)
        out = self.linear(block.attn_out, attn, adapter_rows) + self.linear(block.mlp_out, F.silu(gate) * up, adapter_rows)
        return torch.cat([torch.addcmul(x[:, start:end], out[:, start:end], mod.gate)
                          for start, end, mod in spans], dim=1)


# ---- pipeline layers -----------------------------------------------------------------------
#
# After InitialLayer the activations are one flat tuple:
#   dims [H, W, R, L], text_lengths, pe, vec_img, img, txt,
#   early_img (shift, scale, gate), early_txt (3), single_img (3), single_txt (3),
#   and only when there are references: refs, early_ref (3), single_ref (3).
# After ConcatLayer:
#   dims, text_lengths, pe, vec_img, x, single_img (3), single_txt (3) [, single_ref (3)]

N_EARLY_FIXED = 18
N_JOINT_FIXED = 11


def unpack_early(inputs):
    dims, text_lengths, pe, vec_img, img, txt = inputs[:6]
    mods = inputs[6:18]
    state = {
        'dims': dims, 'text_lengths': text_lengths, 'pe': pe, 'vec_img': vec_img, 'img': img, 'txt': txt,
        'early_image': ModulationOut(*mods[0:3]), 'early_txt': ModulationOut(*mods[3:6]),
        'single_image': ModulationOut(*mods[6:9]), 'single_txt': ModulationOut(*mods[9:12]),
        'refs': None,
    }
    if len(inputs) > N_EARLY_FIXED:
        state['refs'] = inputs[18]
        state['early_image_cond'] = ModulationOut(*inputs[19:22])
        state['single_image_cond'] = ModulationOut(*inputs[22:25])
    return state


def pack_early(state):
    out = [state['dims'], state['text_lengths'], state['pe'], state['vec_img'], state['img'], state['txt'],
           *modulation_tensors(state['early_image']), *modulation_tensors(state['early_txt']),
           *modulation_tensors(state['single_image']), *modulation_tensors(state['single_txt'])]
    if state['refs'] is not None:
        out += [state['refs'], *modulation_tensors(state['early_image_cond']),
                *modulation_tensors(state['single_image_cond'])]
    return make_contiguous(*out)


class TextEncoderLayer(nn.Module):
    """Runs the FLUX.3 text encoder on token ids when text embeddings are not cached. The
    pipeline (and through it the encoder) is held in a list so its weights are not
    parameters of the pipeline module: never trained, saved or swapped."""

    def __init__(self, pipeline, dtype):
        super().__init__()
        self.pipeline = [pipeline]
        self.dtype = dtype
        self.device = None

    def forward(self, inputs):
        x, t, text, text_lengths, *refs = inputs
        pipeline = self.pipeline[0]
        if not torch.is_floating_point(text):
            if pipeline is None:
                raise RuntimeError('Token ids reached the pipeline but text embeddings were meant to be cached')
            te = pipeline.text_encoders[0]
            te.load_model_if_needed()
            if self.device != text.device:
                te.cond_stage_model.to(text.device)
                self.device = text.device
            with torch.no_grad(), torch.autocast(text.device.type, enabled=False):
                text = pipeline.encode_token_ids(text, device=text.device).to(self.dtype)
        return make_contiguous(x, t, text, text_lengths, *refs)


class InitialLayer(nn.Module):
    def __init__(self, model, dtype):
        super().__init__()
        self.emb_in = model.emb_in
        self.txt_in = model.txt_in
        self.time_in = model.time_in
        self.early_stream_modulations = model.early_stream_modulations
        self.single_stream_modulations = model.single_stream_modulations
        self.model = [model]
        self.dtype = dtype

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, t, text, text_lengths, *refs = inputs
        model = self.model[0]
        B, _, H, W = x.shape
        device = x.device

        # As ComfyUI: the time embedding is taken in the compute dtype; references and text sit at t = 0.
        vec = self.time_in(timestep_embedding(torch.cat([t, t.new_zeros(1)]).to(self.dtype), 256))
        vec_img, vec_zero = vec[:-1], vec[-1:]

        img = self.emb_in['image'](x.flatten(2).transpose(1, 2))
        txt = self.txt_in(text)
        ids = [model.image_ids(H, W, 0, device)]
        ref_tokens = None
        if refs:
            ref_tokens = self.emb_in['image_cond'](torch.cat([r.flatten(2).transpose(1, 2) for r in refs], dim=1))
            ids.append(model.ref_ids([r.shape[-2:] for r in refs], device))
        ids.append(model.text_ids(txt.shape[1], device))
        pe = model.rope(torch.cat(ids))
        pe.requires_grad_(True)  # work around pipeline parallelism issue

        R = 0 if ref_tokens is None else ref_tokens.shape[1]
        dims = torch.tensor([H, W, R, txt.shape[1]], dtype=torch.int64, device=device)
        state = {
            'dims': dims, 'text_lengths': text_lengths.to(device), 'pe': pe, 'vec_img': vec_img,
            'img': img, 'txt': txt,
            'early_image': self.early_stream_modulations['image'](vec_img)[0],
            'early_txt': self.early_stream_modulations['txt'](vec_zero)[0],
            'single_image': self.single_stream_modulations['image'](vec_img)[0],
            'single_txt': self.single_stream_modulations['txt'](vec_zero)[0],
            'refs': ref_tokens,
        }
        if ref_tokens is not None:
            state['early_image_cond'] = self.early_stream_modulations['image_cond'](vec_zero)[0]
            state['single_image_cond'] = self.single_stream_modulations['image_cond'](vec_zero)[0]
        return pack_early(state)


class EarlyLayer(nn.Module):
    """One of the five per-stream blocks of the image, the text or the references."""

    def __init__(self, mode, block, offloader, block_idx):
        super().__init__()
        self.mode = mode
        self.block = block
        self.offloader = offloader
        self.block_idx = block_idx

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        state = unpack_early(inputs)
        H, W, R, L = state['dims'].tolist()
        HW = H * W
        pe = state['pe']
        key_mask = None
        if self.mode == 'image':
            key, pe = 'img', pe[:, :HW]
        elif self.mode == 'txt':
            key, pe = 'txt', pe[:, HW + R:]
            key_mask = text_key_mask(state['text_lengths'], L)
        else:
            if state['refs'] is None:
                return inputs
            key, pe = 'refs', pe[:, HW:HW + R]
        tokens = state[key]
        mod = state['early_' + ('image' if self.mode == 'image' else self.mode)]
        if self.offloader is not None:
            self.offloader.wait_for_block(self.block_idx)
        state[key] = self.block(tokens, [(0, tokens.shape[1], mod)], pe, key_mask)
        if self.offloader is not None:
            self.offloader.submit_move_blocks_forward(self.block_idx)
        return pack_early(state)


class ConcatLayer(nn.Module):
    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        state = unpack_early(inputs)
        parts = [state['img']] + ([state['refs']] if state['refs'] is not None else []) + [state['txt']]
        out = [state['dims'], state['text_lengths'], state['pe'], state['vec_img'], torch.cat(parts, dim=1),
               *modulation_tensors(state['single_image']), *modulation_tensors(state['single_txt'])]
        if state['refs'] is not None:
            out += modulation_tensors(state['single_image_cond'])
        return make_contiguous(*out)


def joint_spans(inputs):
    dims, text_lengths, pe, vec_img, x = inputs[:5]
    H, W, R, L = dims.tolist()
    HW = H * W
    spans = [(0, HW, ModulationOut(*inputs[5:8]))]
    if len(inputs) > N_JOINT_FIXED:
        spans.append((HW, HW + R, ModulationOut(*inputs[11:14])))
    spans.append((HW + R, HW + R + L, ModulationOut(*inputs[8:11])))
    return spans, (HW, HW + R), text_key_mask(text_lengths, L, prefix_len=HW + R)


class SingleLayer(nn.Module):
    """One of the 28 shared blocks over [image, references, text]."""

    def __init__(self, block, offloader, block_idx):
        super().__init__()
        self.block = block
        self.offloader = offloader
        self.block_idx = block_idx

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        spans, ref_rows, key_mask = joint_spans(inputs)
        self.offloader.wait_for_block(self.block_idx)
        x = self.block(inputs[4], spans, inputs[2], key_mask, adapter_rows=ref_rows)
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(*inputs[:4], x, *inputs[5:])


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        dims, _, _, vec_img, x = inputs[:5]
        H, W = dims[0].item(), dims[1].item()
        out = self.final_layer['image'](x[:, :H * W], vec_img)
        return out.transpose(1, 2).reshape(x.shape[0], -1, H, W)
