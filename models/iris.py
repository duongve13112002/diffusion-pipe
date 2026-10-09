"""Iris-3B: a pixel-space text-to-image diffusion transformer (speridlabs/iris-3b).

The model has no VAE. It predicts a rectified-flow velocity directly over RGB pixels in [-1, 1],
so what the dataset hands this pipeline as 'latents' is the image itself, stored as uint8 (the
exact bytes PIL decoded, so nothing is lost by storing it that way). Two ways to get there:

  - cache_pixels = false (the default): nothing is cached. The dataset reads and resizes every
    image when it is drawn, in the DataLoader worker, so a dataset never costs more disk than
    the images themselves.
  - cache_pixels = true: the resized pixels are written to the dataset cache once, exactly like
    any other model's latents, and read back from there.

The text encoder is the Qwen3-VL-4B language stack. Iris reads twelve of its hidden layers per
token through a learned layerwise adapter, which makes cached embeddings unusually large (12 x
2560 per token), so encoding on the fly is the default here too: cache_text_embeddings = false.

The submodule's IrisDiT is used as is. Its forward is cut into pipeline layers below, and the
one place that needed more than slicing is the modulation: every patch block reads its adaLN
parameters from a core shared by the whole trunk. A pipeline stage cannot reach a module that
lives on another stage, so the cores run once, in InitialLayer, and their output travels down
the pipeline with the hidden states. Each block's adaLN module is retargeted to read its own
slice of that tensor and add its own bias, which is exactly what it computed before.
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/iris-3b/src'))

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, is_main_process, iterate_safetensors
from utils.offloading import ModelOffloader

from iris3b.config import RepaConfig, inference_config
from iris3b.flow.schedule import FlowSchedule, resolution_shift
from iris3b.flow.solver import FlowDPMSolver
from iris3b.models.blocks.single_stream import SingleStreamBlock
from iris3b.models.dit import IrisDiT
from iris3b.nn.modulation import SharedCoreBias, SharedCoreModulation


# Same template as iris3b.text.qwen3_vl. Copied rather than imported so a change upstream shows
# up as a test failure (test/test_iris.py compares the two) instead of silently changing what a
# cached embedding means.
PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_REPO_ID = 'speridlabs/iris-3b'

# Everything outside the patch trunk stays in the base dtype when transformer_dtype is lower
# precision: embedders, the text adapter, the shared modulation cores and the pixel head.
KEEP_IN_HIGH_PRECISION = [
    's_embedder', 't_embedder', 'y_embedder', 'y_pos_embedding', 'modulation_cores',
    'pixel_embedder', 'pixel_blocks', 'final_layer', 'repa',
]


def _resolve_file(path_or_repo, filename, required=True):
    """A file inside a local directory, or fetched from a Hugging Face model repo."""
    path = Path(path_or_repo)
    if path.is_dir():
        candidate = path / filename
        if candidate.exists():
            return candidate
        if required:
            raise FileNotFoundError(f'{candidate} does not exist')
        return None
    if path.exists():
        # A file was given. Only the weights can be a bare file; config.yaml is looked up beside it.
        if filename.endswith('.safetensors'):
            return path
        candidate = path.parent / filename
        return candidate if candidate.exists() else None
    if re.fullmatch(r'[\w.-]+/[\w.-]+', str(path_or_repo)):
        from huggingface_hub import hf_hub_download
        try:
            return Path(hf_hub_download(str(path_or_repo), filename))
        except Exception:
            if required:
                raise
            return None
    raise FileNotFoundError(f'{path_or_repo} is neither a local path nor a Hugging Face repo id')


def load_iris_config(transformer_path):
    """The Iris inference config (model, text_encoder, flow) that belongs to a checkpoint."""
    from omegaconf import OmegaConf
    config_file = _resolve_file(transformer_path, 'config.yaml', required=False)
    raw = {} if config_file is None else OmegaConf.to_container(OmegaConf.load(str(config_file)))
    return inference_config(raw), raw


class _PackedCoreBias(SharedCoreBias):
    """SharedCoreBias reading the core output InitialLayer already computed.

    The block passes its `cond` argument straight to this module. Here that argument is the
    packed tensor [silu(t_emb) | core_0(cond) | core_1(cond) ...], and this block's slice of it
    plus the block's own bias is the same m_i(c) = core(c) + b_i the original computed.
    """

    def forward(self, packed):
        start = self._packed_offset
        return packed[..., start:start + self.bias.shape[0]] + self.bias


class _PackedCoreLowRank(SharedCoreModulation):
    """SharedCoreModulation reading the core output from the packed tensor (shared_lowrank)."""

    def forward(self, packed):
        start = self._packed_offset
        cond = packed[..., :self._cond_dim]
        return packed[..., start:start + self._packed_width] + self.adaln_up(self.down(cond))


def pack_modulation(transformer):
    """Retarget every block's shared adaLN module at the packed tensor.

    Returns the cores in packing order. Parameter names are untouched: only the instances'
    classes change, so checkpoints, adapters and the saver all see the original IrisDiT names.
    Idempotent, which matters because to_layers() can run more than once.
    """
    dim = transformer.cfg.hidden_size
    cores = list(transformer.modulation_cores.values())
    offsets = {id(core): dim + 6 * dim * i for i, core in enumerate(cores)}
    for module in transformer.modules():
        if isinstance(module, (SharedCoreBias, SharedCoreModulation)):
            offset = offsets[id(module.core)]
            if isinstance(module, SharedCoreBias):
                module.__class__ = _PackedCoreBias
            else:
                module.__class__ = _PackedCoreLowRank
                module._cond_dim = dim
                module._packed_width = 6 * dim
            module._packed_offset = offset
    return cores


def load_repa_projector(repa, state_dict):
    """Load REPA projector weights saved under any of the names they travel under.

    diffusion-pipe saves REPALoss parameter names ('projector.0.weight'); an Iris training
    checkpoint holds them under the model as 'repa.projector.0.weight'; a bare projector
    state dict has '0.weight'. Strict: a projector for another teacher dim must not load.
    """
    state_dict = {re.sub(r'^repa\.', '', k): v for k, v in state_dict.items()}
    state_dict = {re.sub(r'^projector\.', '', k): v for k, v in state_dict.items()}
    repa.projector.load_state_dict(state_dict, strict=True)


class IrisTextEncoder(nn.Module):
    """Qwen3-VL language stack producing Iris's [B, max_length, layers, dim] text states.

    Mirrors iris3b.text.qwen3_vl.Qwen3VLTextEncoder token for token: the chat template prefix,
    the caption truncated to max_length minus the suffix, the suffix appended after truncation
    so the assistant turn can never be cut, pad up to prefix + max_length, and the selected
    1-based hidden states sliced to the max_length positions after the prefix and zeroed at pad.

    Tokenizing and encoding are separate so that prepare_inputs can do the former on CPU in the
    dataloader process and the pipeline can run the latter on the GPU.
    """

    def __init__(self, decoder, tokenizer, hidden_layers, max_length):
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.hidden_layers = tuple(hidden_layers)
        self.max_length = max_length
        prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        suffix_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
        self.prefix_ids = torch.tensor(prefix_ids, dtype=torch.long)
        self.suffix_ids = torch.tensor(suffix_ids, dtype=torch.long)
        self.prefix_len = len(prefix_ids)
        self.suffix_len = len(suffix_ids)
        self.caption_budget = max_length - self.suffix_len
        if self.caption_budget < 1:
            raise ValueError(f'max_text_length={max_length} leaves no room for a caption')
        if self.hidden_layers and tuple(sorted(set(self.hidden_layers))) != self.hidden_layers:
            raise ValueError('text encoder hidden_layers must be sorted and unique')
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self.pad_id = int(pad_id)
        self.overflow_rows = 0
        self._warned_overflow = False
        self.decoder.requires_grad_(False)
        self.decoder.eval()

    @classmethod
    def from_pretrained(cls, path, dtype, hidden_layers, max_length, attn_implementation='sdpa'):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=dtype, attn_implementation=attn_implementation)
        decoder = model.get_decoder()
        del model
        num_layers = decoder.config.num_hidden_layers
        if hidden_layers and (hidden_layers[0] < 1 or hidden_layers[-1] > num_layers):
            raise ValueError(f'hidden_layers must be 1-based indices in [1, {num_layers}], got {hidden_layers}')
        return cls(decoder, tokenizer, hidden_layers, max_length)

    def tokenize(self, captions):
        caption_ids = self.tokenizer(list(captions), add_special_tokens=False)['input_ids']
        start = self.prefix_len
        stop = start + self.max_length
        input_ids = torch.full((len(caption_ids), stop), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(caption_ids), stop), dtype=torch.long)
        input_ids[:, :start] = self.prefix_ids
        for row, ids in enumerate(caption_ids):
            n = min(len(ids), self.caption_budget)
            if len(ids) > self.caption_budget:
                self.overflow_rows += 1
                if not self._warned_overflow and is_main_process():
                    self._warned_overflow = True
                    print(f'WARNING: a caption tokenized to {len(ids)} tokens and was truncated to '
                          f'{self.caption_budget} (max_text_length={self.max_length} minus the '
                          f'{self.suffix_len} chat-template tokens). Shown once.')
            if n:
                input_ids[row, start:start + n] = torch.tensor(ids[:n], dtype=torch.long)
            input_ids[row, start + n:start + n + self.suffix_len] = self.suffix_ids
            attention_mask[row, :start + n + self.suffix_len] = 1
        return input_ids, attention_mask

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        device = next(self.decoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        start = self.prefix_len
        stop = start + self.max_length
        output = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=bool(self.hidden_layers),
        )
        mask = attention_mask[:, start:stop].to(torch.int64)
        if self.hidden_layers:
            embeddings = torch.stack(
                [output.hidden_states[layer][:, start:stop] for layer in self.hidden_layers], dim=2)
            embeddings = embeddings * mask[:, :, None, None].to(embeddings.dtype)
        else:
            embeddings = output.last_hidden_state[:, start:stop]
            embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        return embeddings, mask

    def encode(self, captions):
        return self(*self.tokenize(captions))


class PixelCacheEncoder:
    """Stands in for a VAE: the 'latents' of a pixel-space model are the pixels.

    Not an nn.Module on purpose. DatasetManager moves modules to CUDA before calling their
    encode function and treats anything else as a lazily loaded ComfyUI wrapper, which is the
    contract this needs: nothing to load, nothing to move.
    """

    def load_model_if_needed(self):
        pass

    def to(self, *args, **kwargs):
        # DatasetManager moves every other submodel to the CPU before running a text encoder.
        return self


def pixels_to_uint8(tensor):
    """[0, 1] float pixels as decoded by PreprocessMediaFile -> uint8. Exact for PIL-decoded
    images, whose values are already k/255."""
    return (tensor.float() * 255).round_().clamp_(0, 255).to(torch.uint8)


class IrisPipeline(BasePipeline):
    name = 'iris'
    framerate = None
    checkpointable_layers = ['InitialLayer', 'TransformerLayer', 'PiTLayer']
    adapter_target_modules = ['MMDiTBlock', 'SingleStreamBlock']
    # Set per instance from cache_pixels. When true the dataset reads images at __getitem__
    # time through get_preprocess_media_file_fn() and get_call_vae_fn(), so both have to be
    # cheap and CPU-only, which they are here.
    load_media_on_the_fly = True

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        # Iris defaults to encoding text on the fly. train.py reads this key with a default of
        # True in several places, so the default has to be written into the config, not just
        # assumed here.
        self.model_config.setdefault('cache_text_embeddings', False)
        self.cache_text_embeddings = self.model_config['cache_text_embeddings']
        self.cache_pixels = self.model_config.get('cache_pixels', False)
        self.load_media_on_the_fly = not self.cache_pixels
        # A resident text encoder is held outside the pipeline modules (see TextEncoderLayer),
        # so it follows the batch to the GPU by itself and block swapping is safe with it.
        self.block_swap_supports_uncached_text_embeddings = True

        self.transformer_path = self.model_config.get('transformer_path', DEFAULT_REPO_ID)
        self.iris_config, self.iris_raw_config = load_iris_config(self.transformer_path)
        model_cfg = self.iris_config.model
        # What save_model writes back: a training-time attention backend (fa3, say) must not
        # end up in the exported config, where it would bind inference to that GPU.
        self._checkpoint_attn_backend = model_cfg.attn_backend
        if attn_backend := self.model_config.get('attn_backend', None):
            model_cfg.attn_backend = attn_backend
        self.patch_size = model_cfg.patch_size
        self.pixels_round_to_multiple = self.patch_size
        te_cfg = self.iris_config.text_encoder
        self.max_text_length = self.model_config.get('max_text_length', te_cfg.max_length)
        if self.max_text_length > model_cfg.text_len:
            raise ValueError(f'max_text_length={self.max_text_length} exceeds the model text_len={model_cfg.text_len}')
        if self.max_text_length != te_cfg.max_length and is_main_process():
            # The trunk's joint attention does not mask pad text positions, so the number of
            # pads is part of what the model sees, not just a truncation limit.
            print(f'WARNING: max_text_length={self.max_text_length} differs from the checkpoint '
                  f'text_encoder.max_length={te_cfg.max_length}. Iris attends to pad text positions, '
                  'so outputs change even for short captions; sample with the same length.')
        self.hidden_layers = tuple(te_cfg.hidden_layers)
        self.text_encoder_path = self.model_config.get('text_encoder_path', te_cfg.pretrained)

        if adapter_targets := self.model_config.get('adapter_target_modules', None):
            self.adapter_target_modules = list(adapter_targets)

        flow_cfg = self.iris_config.flow
        self.timestep_sample_method = self.model_config.get('timestep_sample_method', flow_cfg.timestep_sampler)
        if self.timestep_sample_method not in ('logit_normal', 'uniform'):
            raise ValueError(f'timestep_sample_method must be logit_normal or uniform, got {self.timestep_sample_method!r}')
        self.shift = float(self.model_config.get('shift', self.iris_config.flow.shift))
        self.shift_law = self.model_config.get('shift_law', flow_cfg.shift_law)
        if self.shift_law not in ('none', 'sd3', 'flux'):
            raise ValueError(f'shift_law must be none, sd3 or flux, got {self.shift_law!r}')
        self.num_train_timesteps = self.iris_config.flow.num_train_timesteps
        if self.iris_config.flow.prediction != 'v':
            raise NotImplementedError('Only velocity-prediction Iris checkpoints are supported')
        self._schedules = {}

        self.repa_weight = float(self.model_config.get('repa_weight', 0.0))
        self.repa_layer = int(self.model_config.get('repa_layer', model_cfg.repa_layer))
        if self.repa_weight > 0 and not 1 <= self.repa_layer <= model_cfg.depth:
            raise ValueError(f'repa_layer must be in [1, {model_cfg.depth}], got {self.repa_layer}')
        self.repa = None
        self._last_repa_loss = None
        self._checkpoint_repa_state = None

        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        self.text_encoder = None
        if self.model_config.get('load_text_encoder', True):
            self.text_encoder = IrisTextEncoder.from_pretrained(
                self.text_encoder_path,
                self._text_encoder_dtype(te_cfg),
                self.hidden_layers,
                self.max_text_length,
                attn_implementation=self.model_config.get('text_encoder_attn_implementation', te_cfg.attn_implementation),
            )
        self.pixel_encoder = PixelCacheEncoder()
        self.transformer = None

    def _text_encoder_dtype(self, te_cfg):
        dtype = self.model_config.get('text_encoder_dtype', te_cfg.dtype)
        if isinstance(dtype, str):
            dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
        return dtype

    # ---- identities ------------------------------------------------------------------------------

    def text_encoder_cache_key(self, i):
        return '|'.join(str(x) for x in (
            self.text_encoder_path, self.hidden_layers, self.max_text_length,
            self._text_encoder_dtype(self.iris_config.text_encoder), 'trimmed-v1'))

    # ---- model loading ---------------------------------------------------------------------------

    def _schedule(self, shift):
        if shift not in self._schedules:
            self._schedules[shift] = FlowSchedule(self.num_train_timesteps, shift)
        return self._schedules[shift]

    def build_transformer(self, state_dict_iter=None):
        """IrisDiT on CPU, weights from the checkpoint (or freshly initialised when None)."""
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)
        if state_dict_iter is None:
            transformer = IrisDiT(self.iris_config.model).to(dtype)
            return transformer
        self._checkpoint_repa_state = {}
        with init_empty_weights():
            transformer = IrisDiT(self.iris_config.model)
        expected = dict(transformer.named_parameters())
        loaded = set()
        for key, tensor in state_dict_iter:
            key = re.sub(r'^(model\.diffusion_model|diffusion_model|transformer|net)\.', '', key)
            if key not in expected:
                if key.startswith('repa.'):
                    self._checkpoint_repa_state[key] = tensor
                    continue
                raise RuntimeError(f'Unexpected key {key} in the Iris checkpoint')
            p = expected[key]
            if tuple(tensor.shape) != tuple(p.shape):
                raise RuntimeError(f'Shape mismatch for {key}: checkpoint {tuple(tensor.shape)}, model {tuple(p.shape)}')
            dtype_to_use = dtype if (any(k in key for k in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
            set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=tensor)
            loaded.add(key)
        missing = sorted(set(expected) - loaded)
        if missing:
            raise RuntimeError(f'The Iris checkpoint is missing {len(missing)} parameters, e.g. {missing[:5]}')
        return transformer

    def load_diffusion_model(self):
        weights = _resolve_file(self.transformer_path, 'model.safetensors')
        transformer = self.build_transformer(iterate_safetensors(weights))
        self.transformer = transformer
        for adapter_path in self.model_config.get('merge_adapters', []):
            # diffusion-pipe save directories (adapter_config.json + adapter_model.safetensors);
            # a path to the .safetensors inside one is accepted too.
            adapter_path = Path(adapter_path)
            if adapter_path.is_file():
                adapter_path = adapter_path.parent
            if is_main_process():
                print(f'Merging adapter {adapter_path}')
            self.load_and_fuse_adapter(str(adapter_path))
        self._finish_transformer(transformer)

    def _finish_transformer(self, transformer):
        cfg = transformer.cfg
        if cfg.modulation not in ('per_block', 'shared_bias', 'shared_lowrank'):
            raise NotImplementedError(f'modulation={cfg.modulation} is not supported')
        last = transformer.blocks[-1]
        if isinstance(last, SingleStreamBlock):
            # The model discards the last block's text output. For a single-stream block that
            # path owns no parameters, so skipping it changes nothing but the wasted compute:
            # the image rows are computed exactly as before (text still supplies keys/values).
            last.text_out = False
        transformer.train()
        self.transformer = transformer
        for name, p in transformer.named_parameters():
            p.original_name = name
        if self.repa_weight > 0:
            self.repa = self._build_repa(cfg.hidden_size)

    def _build_repa(self, student_dim):
        from iris3b.repa import REPALoss
        keys = {
            'repa_variant': 'variant', 'repa_teacher_source': 'teacher_source',
            'repa_teacher_hub': 'teacher_hub', 'repa_teacher': 'teacher',
            'repa_teacher_weights': 'teacher_weights', 'repa_teacher_dim': 'teacher_dim',
            'repa_proj_hidden_dim': 'proj_hidden_dim', 'repa_teacher_image_size': 'teacher_image_size',
            'repa_teacher_patch_size': 'teacher_patch_size',
            'repa_teacher_match_student': 'teacher_match_student',
            'repa_spatial_norm_gamma': 'spatial_norm_gamma',
        }
        kwargs = {dst: self.model_config[src] for src, dst in keys.items() if src in self.model_config}
        repa = REPALoss(RepaConfig(weight=self.repa_weight, **kwargs), student_dim)
        if not repa.available:
            raise RuntimeError(
                'repa_weight > 0 but the REPA teacher could not be loaded (see the warning above). '
                'Check repa_teacher_hub / repa_teacher, or set repa_weight = 0.')
        state = None
        if path := self.model_config.get('repa_projector_path', None):
            state = safetensors.torch.load_file(path)
        elif self._checkpoint_repa_state:
            # An Iris training export that still carries its projector: continue from it.
            state = self._checkpoint_repa_state
        if state is not None:
            load_repa_projector(repa, state)
        self._checkpoint_repa_state = None
        repa.projector.to(self.model_config['dtype'])
        for name, p in repa.named_parameters():
            p.original_name = 'repa.' + name
        return repa

    def get_vae(self):
        return self.pixel_encoder

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return [self.text_encoder]
        return []

    def free_vae_and_te(self):
        if self.cache_text_embeddings:
            self.text_encoder = None

    # ---- adapters / saving -----------------------------------------------------------------------

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        if self.repa is not None:
            # Trained densely next to the adapter; never part of it.
            for p in self.repa.parameters():
                p.requires_grad_(True)

    def _split_repa(self, state_dict):
        repa = {}
        for k in [k for k in state_dict if k.startswith('repa.')]:
            repa[k[len('repa.'):]] = state_dict.pop(k)
        return repa

    def save_adapter(self, save_dir, peft_state_dict):
        repa = self._split_repa(peft_state_dict)
        self.peft_config.save_pretrained(save_dir)
        peft_state_dict = {'diffusion_model.' + k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})
        if repa:
            # Subdirectory: load_adapter_weights globs '*.safetensors' in save_dir and refuses
            # more than one file.
            (save_dir / 'repa').mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(repa, save_dir / 'repa' / 'repa_projector.safetensors', metadata={'format': 'pt'})

    def load_adapter_weights(self, adapter_path):
        super().load_adapter_weights(adapter_path)
        repa_file = Path(adapter_path) / 'repa' / 'repa_projector.safetensors'
        if self.repa is not None and repa_file.exists():
            load_repa_projector(self.repa, safetensors.torch.load_file(repa_file))

    def save_model(self, save_dir, state_dict):
        """Iris's own exported layout: model.safetensors with IrisDiT keys plus config.yaml.

        iris3b.sampling.load_for_inference(save_dir) reads it back unchanged.
        """
        from omegaconf import OmegaConf
        from iris3b.config import INFERENCE_SECTIONS
        repa = self._split_repa(state_dict)
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})
        config = OmegaConf.to_container(OmegaConf.structured(self.iris_config))
        config = {k: config[k] for k in INFERENCE_SECTIONS}
        config['model']['attn_backend'] = self._checkpoint_attn_backend
        # Settings a sampler reads that training may have overridden.
        config['flow']['shift'] = self.shift
        config['flow']['shift_law'] = self.shift_law
        config['text_encoder']['max_length'] = self.max_text_length
        OmegaConf.save(OmegaConf.create(config), str(save_dir / 'config.yaml'))
        if repa:
            safetensors.torch.save_file(repa, save_dir / 'repa_projector.safetensors', metadata={'format': 'pt'})

    # ---- data ------------------------------------------------------------------------------------

    def model_specific_dataset_config_validation(self, dataset_config):
        for directory in dataset_config.get('directory', []):
            if directory.get('control_path', None) is not None:
                raise NotImplementedError('Iris is a text-to-image model; control/edit datasets are not supported')

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=False,
            round_height=self.patch_size,
            round_width=self.patch_size,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            return {'latents': pixels_to_uint8(tensor)}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            embeddings, mask = text_encoder.encode(captions)
            lengths = mask.sum(dim=1).tolist()
            # Stored trimmed to the real tokens: twelve hidden layers per token make the padded
            # [300, 12, 2560] tensor ~18 MB per caption. prepare_inputs pads it back.
            return {'prompt_embeds': [embeddings[i, :n] for i, n in enumerate(lengths)]}
        return fn

    def _pad_text_embeddings(self, prompt_embeds):
        if torch.is_tensor(prompt_embeds):
            prompt_embeds = list(prompt_embeds)
        bs = len(prompt_embeds)
        first = prompt_embeds[0]
        y = torch.zeros((bs, self.max_text_length, *first.shape[1:]), dtype=first.dtype)
        y_mask = torch.zeros((bs, self.max_text_length), dtype=torch.int64)
        for i, emb in enumerate(prompt_embeds):
            n = emb.shape[0]
            y[i, :n] = emb
            y_mask[i, :n] = 1
        return y, y_mask

    def _sample_timestep_idx(self, bs, quantile=None):
        n = self.num_train_timesteps
        if self.timestep_sample_method == 'logit_normal':
            mean = float(self.model_config.get('logit_mean', self.iris_config.flow.logit_mean))
            std = float(self.model_config.get('logit_std', self.iris_config.flow.logit_std))
            if quantile is not None:
                z = torch.distributions.Normal(mean, std).icdf(torch.full((bs,), float(quantile)))
            else:
                z = torch.randn(bs) * std + mean
            u = torch.sigmoid(z)
        else:
            u = torch.full((bs,), float(quantile)) if quantile is not None else torch.rand(bs)
        min_t = float(self.model_config.get('min_t', 0.0))
        max_t = float(self.model_config.get('max_t', 1.0))
        u = min_t + u * (max_t - min_t)
        return (u * n).long().clamp(0, n - 1)

    def _shift_for(self, height, width):
        tokens = (height // self.patch_size) * (width // self.patch_size)
        return resolution_shift(tokens, self.shift_law, self.shift, self.iris_config.flow.shift_base_tokens)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        pixels = inputs['latents']
        if pixels.dtype == torch.uint8:
            x0 = pixels.float() / 127.5 - 1.0
        else:
            x0 = pixels.float() * 2.0 - 1.0
        mask = inputs['mask']
        bs, _, h, w = x0.shape

        if self.cache_text_embeddings:
            text = self._pad_text_embeddings(inputs['prompt_embeds'])
        else:
            text = self.text_encoder.tokenize(inputs['caption'])

        if mask is not None:
            mask = mask.unsqueeze(1)  # (bs, 1, h, w)
            if mask.shape[-2:] != (h, w):
                # A mask batch fill synthesised from the size bucket, before the pixels were
                # rounded to the patch size. Real masks are already at pixel resolution.
                mask = F.interpolate(mask.float(), size=(h, w), mode='nearest-exact')

        idx = self._sample_timestep_idx(bs, timestep_quantile)
        schedule = self._schedule(self._shift_for(h, w))
        noise = torch.randn_like(x0)
        x_t, t_model = schedule.add_noise(x0, noise, idx)
        target = noise - x0

        features = (x_t, t_model, *text)
        if self.repa is not None:
            # uint8, so it is not part of the pipeline's gradient exchange (only the teacher,
            # under no_grad, ever reads it).
            features += (pixels if pixels.dtype == torch.uint8 else pixels_to_uint8(pixels),)
        return features, (target, mask)

    # ---- pipeline --------------------------------------------------------------------------------

    def to_layers(self):
        transformer = self.transformer
        pack_modulation(transformer)
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        dtype = self.model_config['dtype']
        layers = [TextEncoderLayer(text_encoder), InitialLayer(transformer, dtype)]
        num_double = transformer.cfg.dual_depth
        for i, block in enumerate(transformer.blocks):
            if i < num_double:
                layers.append(TransformerLayer(transformer, block, i, self.offloader_double))
            else:
                layers.append(TransformerLayer(transformer, block, i - num_double, self.offloader_single))
            if self.repa is not None and i + 1 == self.repa_layer:
                layers.append(RepaLayer(self.repa, self))
        if transformer.pixel_blocks is not None:
            layers.append(PixelEmbedLayer(transformer))
            for block in transformer.pixel_blocks:
                layers.append(PiTLayer(transformer, block))
        layers.append(FinalLayer(transformer, returns_repa_loss=self.repa is not None))
        return layers

    def get_param_groups(self, parameters):
        lr = self.config['optimizer']['lr']
        group_lrs = {
            'y_embedder': self.model_config.get('text_adapter_lr', lr),
            'pixel': self.model_config.get('pixel_head_lr', lr),
            'repa': self.model_config.get('repa_lr', lr),
            'base': lr,
        }
        groups = {k: [] for k in group_lrs}
        for p in parameters:
            name = p.original_name
            if name.startswith('repa.'):
                groups['repa'].append(p)
            elif name.startswith('y_embedder.') or name.startswith('y_pos_embedding'):
                groups['y_embedder'].append(p)
            elif name.startswith(('pixel_embedder.', 'pixel_blocks.', 'final_layer.')):
                groups['pixel'].append(p)
            else:
                groups['base'].append(p)
        param_groups = []
        for key, params in groups.items():
            if group_lrs[key] == 0:
                for p in params:
                    p.requires_grad_(False)
            elif params:
                param_groups.append({'params': params, 'lr': group_lrs[key]})
        if is_main_process():
            print('Iris param groups: ' + ', '.join(f'{k}: {len(v)} tensors @ lr={group_lrs[k]}' for k, v in groups.items()))
        return param_groups

    def get_loss_fn(self):
        repa_weight = self.repa_weight

        def loss_fn(output, label):
            target, mask = label
            repa_loss = None
            if isinstance(output, (tuple, list)):
                output, repa_loss = output
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'huber_delta' in self.config:
                    loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
                elif 'smooth_l1_beta' in self.config:
                    loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                if mask.numel() > 0:
                    loss = loss * mask.to(output.device, torch.float32)
                loss = loss.mean()
                if repa_loss is not None and repa_loss.numel() > 0:
                    loss = loss + repa_weight * repa_loss.to(output.device, torch.float32).sum()
            return loss
        return loss_fn

    def get_extra_log_scalars(self):
        if self._last_repa_loss is None:
            return {}
        return {'train/repa_loss': float(self._last_repa_loss)}

    # ---- block swap ------------------------------------------------------------------------------

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_double = transformer.cfg.dual_depth
        double_blocks = blocks[:num_double]
        single_blocks = blocks[num_double:]
        # A dual-stream block holds twice the parameters of a single-stream one, so a third of
        # the request goes to the dual half, and the swapper only ever exchanges blocks of the
        # same class. A trunk with only one kind of block takes the whole request there.
        max_double = max(len(double_blocks) - 2, 0)
        max_single = max(len(single_blocks) - 2, 0)
        if not single_blocks:
            double_to_swap = blocks_to_swap
        elif not double_blocks:
            double_to_swap = 0
        else:
            double_to_swap = min(blocks_to_swap // 3, max_double)
        single_to_swap = blocks_to_swap - double_to_swap
        assert double_to_swap <= max_double and single_to_swap <= max_single, (
            f'Cannot swap {blocks_to_swap} blocks: at most {max_double} dual-stream and '
            f'{max_single} single-stream blocks can be swapped.')
        self.offloader_double = ModelOffloader(
            'MMDiTBlock', list(double_blocks), len(double_blocks), double_to_swap, True,
            torch.device('cuda'), self.config['reentrant_activation_checkpointing'])
        self.offloader_single = ModelOffloader(
            'SingleStreamBlock', list(single_blocks), len(single_blocks), single_to_swap, True,
            torch.device('cuda'), self.config['reentrant_activation_checkpointing'])
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        if self.repa is not None:
            self.repa.to('cuda')
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {double_to_swap} dual-stream and {single_to_swap} single-stream blocks.')

    def prepare_block_swap_training(self):
        for offloader in (self.offloader_double, self.offloader_single):
            offloader.enable_block_swap()
            offloader.set_forward_only(False)
            offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        for offloader in (self.offloader_double, self.offloader_single):
            if disable_block_swap:
                offloader.disable_block_swap()
            offloader.set_forward_only(True)
            offloader.prepare_block_devices_before_forward()

    # ---- sampling (--test_sample) ----------------------------------------------------------------

    @torch.no_grad()
    def prepare_sample_test(self, prompt, negative_prompt=None, cfg=None, device='cuda'):
        sample_cfg = self.iris_config.sample
        if negative_prompt is None:
            negative_prompt = sample_cfg.negative_prompt
        if cfg is None:
            cfg = sample_cfg.cfg_scale
        te = self.text_encoder
        te_device = next(te.parameters()).device
        te.to(device)
        self.conds = te.encode([prompt])
        self.unconds = te.encode([negative_prompt]) if cfg != 1 else None
        if self.cache_text_embeddings:
            te.to(te_device)
        self.sample_cfg = cfg

    def get_conds(self, inputs):
        return self._pad_text_embeddings(inputs['prompt_embeds'])

    @torch.no_grad()
    def sample(self, w=512, h=512, steps=None, seed=0, noise=None):
        """One image through the pipeline layers with Iris's own solver and settings.

        Same integrator, CFG batch layout, cfg_interval and shift as iris3b.sampling.generate,
        so a sample here is what Iris's scripts/sample.py draws from the same weights.
        """
        sample_cfg = self.iris_config.sample
        steps = steps or self.model_config.get('sample_steps', sample_cfg.steps)
        p = self.patch_size
        h, w = h // p * p, w // p * p
        y, y_mask = self.conds
        device = y.device
        if noise is None:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, 3, h, w), device=device, generator=generator)
        z = noise.to(device=device, dtype=torch.float32)

        def model_fn(x, t_model, y_batch):
            if y_batch is y:
                text = (y, y_mask)
            else:
                # The solver's CFG batch is cat([uncond, cond]); the mask follows suit.
                text = (y_batch, torch.cat([self.unconds[1], y_mask]))
            inputs = (x, t_model, *text)
            if self.repa is not None:
                # Placeholder for the clean image; RepaLayer ignores it under no_grad.
                inputs += (torch.zeros((x.shape[0], 1, 1, 1), dtype=torch.uint8, device=x.device),)
            out = self.pipeline_model(inputs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.float()

        solver = FlowDPMSolver(model_fn, num_timesteps=self.num_train_timesteps, cfg_scale=self.sample_cfg,
                               cfg_interval=tuple(sample_cfg.cfg_interval))
        uncond = self.unconds[0] if self.unconds is not None else None
        x = solver.sample(z, y, uncond, steps=steps, order=sample_cfg.order, shift=self._shift_for(h, w))
        img = (x.clamp(-1, 1) + 1) / 2
        return img.permute(0, 2, 3, 1)


def _tie(out, *unused):
    """`out` with a zero-weight dependency on tensors this layer does not otherwise consume.

    Under pipeline parallelism every floating tensor a stage receives is given requires_grad,
    and DeepSpeed asserts each one gets a gradient to send back. A tensor that was only
    carried along (or whose last consumer was the previous layer) would trip that assert at a
    stage boundary placed right before this layer. The zero term gives it a zero gradient and
    changes nothing else.
    """
    for t in unused:
        if torch.is_floating_point(t) and t.requires_grad:
            out = out + t.reshape(-1)[:1].sum().to(out.dtype) * 0
    return out


def _grid_marker(height, width, patch_size, device):
    """The patch grid, carried as the shape of a tiny uint8 tensor.

    Layers after InitialLayer only need the image size, never the noisy image itself (except
    the pixel head, which receives it separately). A shape survives a pipeline stage boundary
    without a host sync, and a non-floating tensor is not part of the gradient exchange.
    """
    return torch.zeros((height // patch_size, width // patch_size), dtype=torch.uint8, device=device)


class TextEncoderLayer(nn.Module):
    """Runs the frozen text encoder when embeddings are not cached; a passthrough otherwise.

    The encoder is held in a list so it is not registered: it never trains, so keeping it out of
    the pipeline module keeps it out of DeepSpeed's checkpoints, its parameter-count partitioning
    and the block-swap device shuffle. It is moved to the batch's device on first use instead.
    Not a checkpointable layer: recomputing a 4B frozen forward to save its small output would be
    a bad trade.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = [text_encoder] if text_encoder is not None else []

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, t, text_a, text_b, *extra = inputs
        if torch.is_floating_point(text_a):
            y, y_mask = text_a, text_b
        else:
            if not self.text_encoder:
                raise RuntimeError('Got token ids but no text encoder is loaded')
            te = self.text_encoder[0]
            if next(te.parameters()).device != x.device:
                te.to(x.device)
            with torch.no_grad():
                y, y_mask = te(text_a, text_b)
        outputs = make_contiguous(x, t, y, y_mask, *extra)
        # Marked here, in the one layer that is never checkpointed, so InitialLayer's inputs
        # already require grad when a checkpoint wrapper looks at them. Reentrant checkpointing
        # decides from its inputs alone whether the wrapped layer's parameters get gradients.
        for tensor in outputs:
            if torch.is_floating_point(tensor) and not tensor.requires_grad:
                tensor.requires_grad_(True)
        return outputs


class InitialLayer(nn.Module):
    """Patch/time/text embedding and the shared modulation cores.

    Output: (s, y, packed, t_emb, grid, [x if the model has a pixel head], *extra).
    """

    def __init__(self, model, dtype):
        super().__init__()
        self.s_embedder = model.s_embedder
        self.t_embedder = model.t_embedder
        self.y_embedder = model.y_embedder
        if model.y_pos_embedding is not None:
            self.y_pos_embedding = model.y_pos_embedding
        else:
            self.y_pos_embedding = None
        self.modulation_cores = model.modulation_cores
        self.model = [model]
        self.dtype = dtype

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # TextEncoderLayer already marked the floating inputs as requiring grad. Nothing here
        # changes an input's metadata: non-reentrant checkpointing recomputes this layer and
        # rejects a recomputation that sees different tensors than the first pass did.
        x, t, y, y_mask, *extra = inputs
        cfg = self.model[0].cfg
        p = cfg.patch_size
        if x.shape[-2] % p or x.shape[-1] % p:
            raise ValueError(f'input {tuple(x.shape[-2:])} is not divisible by patch_size {p}')
        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
        s = self.s_embedder(patches)
        t_emb = self.t_embedder(t)
        cond = F.silu(t_emb)
        y = y[:, :cfg.text_len].to(self.dtype)
        y_mask = y_mask[:, :cfg.text_len]
        if cfg.text_adapter != 'linear':
            y = self.y_embedder(y, y_mask)
        else:
            y = self.y_embedder(y)
        if self.y_pos_embedding is not None:
            y = y + self.y_pos_embedding[:, :y.shape[1]].to(y.dtype)
        packed = torch.cat([cond] + [core(cond) for core in self.modulation_cores.values()], dim=-1)
        grid = _grid_marker(x.shape[-2], x.shape[-1], p, x.device)
        pixel_input = (x,) if self.model[0].pixel_blocks is not None else ()
        return make_contiguous(s, y, packed, t_emb, grid, *pixel_input, *extra)


class TransformerLayer(nn.Module):
    def __init__(self, model, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        s, y, packed, t_emb, grid, *rest = inputs
        model = self.model[0]
        rope_img = model._fetch_rope_img(tuple(grid.shape), s.device)
        rope_txt = model._fetch_rope_txt(y.shape[1], s.device)
        self.offloader.wait_for_block(self.block_idx)
        s, y = self.block(s, y, packed, rope_img, rope_txt)
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(s, y, packed, t_emb, grid, *rest)


class RepaLayer(nn.Module):
    """Representation alignment of the patch tokens after block `repa_layer` (Iris stage 1).

    Consumes the clean image (uint8, appended by prepare_inputs) and replaces it with this
    batch's REPA loss, which rides the remaining layers to the loss function. Skipped in eval and
    under no_grad (sampling), so eval losses stay the plain flow loss.
    """

    def __init__(self, repa, pipeline):
        super().__init__()
        self.repa = repa
        self.pipeline = [pipeline]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        *head, clean = inputs
        s = head[0]
        grid = head[4]
        if self.training and torch.is_grad_enabled():
            x0 = clean.float() / 127.5 - 1.0
            # Iris runs the teacher and projector outside autocast (TrainModel.forward).
            with torch.autocast('cuda', enabled=False):
                loss = self.repa(x0, s, tuple(grid.shape)).float().reshape(1)
            self.pipeline[0]._last_repa_loss = loss.detach()
        else:
            loss = torch.zeros(1, device=s.device, dtype=torch.float32)
        return make_contiguous(*head, loss)


class PixelEmbedLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pixel_embedder = model.pixel_embedder
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        s, y, packed, t_emb, grid, x, *extra = inputs
        s = F.silu(t_emb + s)  # timestep re-fused into every patch token
        s_cond = s.reshape(-1, s.shape[-1])
        pixels = _tie(self.pixel_embedder(x), y, packed)
        return make_contiguous(pixels, s_cond, grid, *extra)


class PiTLayer(nn.Module):
    def __init__(self, model, block):
        super().__init__()
        self.block = block
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        pixels, s_cond, grid, *extra = inputs
        model = self.model[0]
        grid_hw = tuple(grid.shape)
        rope_pix = model._fetch_rope_pix(grid_hw, pixels.device)
        pixels = self.block(pixels, s_cond, rope_pix, grid_hw)
        return make_contiguous(pixels, s_cond, grid, *extra)


class FinalLayer(nn.Module):
    def __init__(self, model, returns_repa_loss=False):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]
        self.returns_repa_loss = returns_repa_loss

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        model = self.model[0]
        p = model.cfg.patch_size
        if model.pixel_blocks is None:
            s, y, packed, t_emb, grid, *extra = inputs
            s = F.silu(t_emb + s)
            folded = _tie(self.final_layer(s), y, packed).transpose(1, 2)
            batch = s.shape[0]
        else:
            pixels, s_cond, grid, *extra = inputs
            gh, gw = grid.shape
            n_patches = gh * gw
            batch = pixels.shape[0] // n_patches
            out = _tie(self.final_layer(pixels), s_cond)  # [B*L, p*p, C]
            folded = out.reshape(batch, n_patches, p * p, -1).permute(0, 3, 2, 1)
            folded = folded.reshape(batch, -1, n_patches)
        height, width = grid.shape[0] * p, grid.shape[1] * p
        out = F.fold(folded, output_size=(height, width), kernel_size=p, stride=p)
        if self.returns_repa_loss:
            return out, extra[-1]
        return out
