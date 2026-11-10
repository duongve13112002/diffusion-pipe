"""MiniMax-H3-Image (MiniMaxAI): text to image and editing with the H3 transformer, no audio stream.

The transformer is ComfyUI's MiniMaxH3ImageModel (comfy.ldm.minimax). Its text encoder is the 50-layer Qwen3-VL-32B of
MiniMax H3 (ComfyUI loads it with type 'minimax'), and its VAE is the H3 encoder with a 2D ViT decoder
(MiniMaxH3ImageVAE). The official model is distilled: it samples in 6 steps without CFG on the sigma grid of
pdd_config.json, one output head per step (the PDD head bank, final_layer.video_out).

Weights come in the layouts of utils/minimax_h3_weights.py: ComfyUI's single file, the official transformer/ shards
(with pdd_fused_heads.safetensors) or transformer_diffusers/; the VAE as vae/, vae_diffusers/ or ComfyUI's file; the text
encoder as ComfyUI's file or the Hugging Face text_encoder/ shards.

Training: flow matching on the velocity noise - latent. Head 0 of the bank is the trainable weight and every other head
of a checkpoint is that head plus a fixed offset, so a full fine-tune moves all six together and adapters leave them
alone. A sample trains through the head of the grid interval its noise level falls in, as sampling applies it
(train_pdd_head = 0 trains through head 0 alone). Sampling previews run the six heads on the checkpoint's sigma grid.

Editing (a dataset directory with control_path): as ComfyUI's MiniMaxH3ImageConditioning node builds it. The caption is
preceded by "Picture 1: " and the control image shown to the text encoder at a 1024 short edge; the vision block's
tokens are tagged 0 (they take the video modulation), the rest 1. The control image's latent (cropped to the target's
bucket) is the reference, packed after the text, noise-augmented to t = 0.999, pinned at max(t, 0.999).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from tqdm import tqdm

from models.base import PreprocessMediaFile, convert_to_rgb, make_contiguous
from models.minimax_h3 import MinimaxH3Pipeline, TransformerLayer
from utils import minimax_h3_weights as weights
from utils.common import AUTOCAST_DTYPE, is_main_process
import comfy.model_management
import comfy.sd
import comfy.text_encoders.minimax
import comfy.utils
import comfy.ldm.minimax.vae
from comfy.ldm.minimax.model import (
    PackedLayout, VISUAL_COND_TIMESTEP, patchify_video, unpatchify_video, rope_rotation_table,
)

# ComfyUI comfy_extras/nodes_minimax_h3.py: the text encoder sees reference images at a 1024 short edge, sides /32.
VISION_SHORT_EDGE = 1024
CANVAS_MULTIPLE = 32
PICTURE_LABEL = 'Picture %d: '
# A common flow shift for images of this size. The checkpoint's own sampling grid (pdd_config.json) is not a shifted
# uniform grid, so this is a default, not a measured optimum.
DEFAULT_SHIFT = 3.0
TRAIN_HEADS = ('sigma', 0)
HEAD_LR_KEYS = ('head_lr', 'patch_embed_lr', 'text_projection_lr')
LR_KEYS = HEAD_LR_KEYS + ('blocks_lr', 'token_refiner_lr')
REFUSED_MODEL_KEYS = {
    'audio_vae': 'MiniMax-H3-Image has no audio stream: it trains images only',
    'cfg': 'MiniMax-H3-Image is guidance-distilled and samples without CFG; CFG-augmented training does not apply',
}


def size_short_edge(width, height, low, high):
    """ComfyUI's _size_short_edge: aspect-preserving scale putting the short edge in [low, high], sides /32."""
    scale = min(max(min(width, height), low), high) / min(width, height)
    return tuple(max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE) for side in (width, height))


def load_vision_image(path):
    """The control image ([1, H, W, 3] in 0..1) at the size MiniMaxH3ImageConditioning shows it to the text encoder.
    Decoded as the dataset decodes the same file into the reference latent (convert_to_rgb: no EXIF rotation,
    transparency onto white), so the encoder and the latent see one image."""
    with Image.open(path) as img:
        img = convert_to_rgb(img)
        image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]
    size = size_short_edge(image.shape[2], image.shape[1], VISION_SHORT_EDGE, VISION_SHORT_EDGE)
    if (image.shape[2], image.shape[1]) == size:
        return image
    samples = comfy.utils.common_upscale(image.movedim(-1, 1), size[0], size[1], 'lanczos', 'disabled')
    return samples.movedim(1, -1)


class MiniMaxH3ImagePipeline(MinimaxH3Pipeline):
    name = 'minimax_h3_image'
    is_video_vae = False
    vae_latent_keys = ('latents', 'control_latents')
    # fp32 islands of the checkpoint: save_dtype leaves them unrounded (rounding head 0 moves all six heads)
    save_float32_prefixes = ('final_layer.video_out.', 'video_patch_proj.', 'time_embedder.')

    def __init__(self, config):
        model_config = config['model']
        for key, reason in REFUSED_MODEL_KEYS.items():
            if key in model_config:
                raise ValueError(f'[model] {key} does not apply to minimax_h3_image: {reason}. Remove it.')
        text_encoders = model_config.get('text_encoders', [])
        if len(text_encoders) != 1 or str(text_encoders[0].get('type', '')).lower() != 'minimax':
            raise ValueError("minimax_h3_image needs one text encoder, the MiniMax H3 Qwen3-VL-32B (ComfyUI's file, or the "
                             "text_encoder/ shards of MiniMaxAI/MiniMax-H3-Image) with type = 'minimax'")
        if model_config.get('train_pdd_head', 'sigma') not in TRAIN_HEADS:
            raise ValueError(f"[model] train_pdd_head must be 'sigma' or 0, got {model_config['train_pdd_head']!r}")
        if 'diffusion_model_format' in model_config and model_config['diffusion_model_format'] not in weights.LAYOUTS:
            raise ValueError(f"[model] diffusion_model_format must be one of {weights.LAYOUTS}, "
                             f"got {model_config['diffusion_model_format']!r}")
        if 'adapter' in config and (keys := [k for k in HEAD_LR_KEYS if k in model_config]):
            raise ValueError(f'[model] {", ".join(keys)} do not apply to adapter training: they set the learning rate of '
                             'weights an adapter leaves frozen. Remove them (blocks_lr and token_refiner_lr do apply).')
        if not any(k in model_config for k in ('shift', 'image_shift', 'flux_shift')):
            model_config['shift'] = DEFAULT_SHIFT
        super().__init__(config)
        self.is_video_vae = False
        self.pdd_offsets = None

    def refuse_unsupported_checkpoint(self):
        layout, heads = weights.inspect_checkpoint(self.model_config['diffusion_model'], self.model_config.get('diffusion_model_format'),
                                                   self.model_config.get('pdd_heads'))
        if heads == 1 and 'train_pdd_head' in self.model_config:
            raise ValueError('[model] train_pdd_head does not apply: this diffusion_model has one output head, not the PDD bank '
                             'of six. Remove it.')
        if 'pdd_heads' in self.model_config and layout != 'official':
            raise ValueError(f"[model] pdd_heads applies to the official transformer/ layout only; this diffusion_model is "
                             f"in the {layout} layout, which carries its heads itself. Remove pdd_heads.")

    def model_specific_dataset_config_validation(self, dataset_config):
        directories = dataset_config.get('directory', [])
        for where, config in [('the dataset', dataset_config)] + [(f"[[directory]] {d.get('path')}", d) for d in directories]:
            frame_buckets = config.get('frame_buckets', [1])
            if any(f != 1 for f in frame_buckets):
                raise ValueError(f'minimax_h3_image trains images only; frame_buckets must be [1], got {frame_buckets} in {where}')
        for directory in directories:
            if any(k.startswith('control_path_') for k in directory):
                raise ValueError('minimax_h3_image takes one reference image per sample (control_path); '
                                 'several references (control_path_N) are not supported yet')
            # A dropped caption trains on the one cached empty-caption embedding, which has no control
            # image; ComfyUI never pairs that with a reference.
            if 'caption_dropout_rate' in directory:
                dropout, where = directory['caption_dropout_rate'], 'caption_dropout_rate in it'
            elif 'caption_dropout_rate' in dataset_config:
                dropout, where = dataset_config['caption_dropout_rate'], 'the dataset caption_dropout_rate'
            else:
                dropout, where = self.config.get('uncond_fraction', 0), 'uncond_fraction of the training config'
            if 'control_path' in directory and dropout:
                raise ValueError(f"caption dropout ({where}) does not apply to the edit directory "
                                 f"{directory.get('path')} of minimax_h3_image: a dropped caption would lose its control "
                                 'image. Set caption_dropout_rate = 0 in that [[directory]].')

    def text_encoder_cache_key(self, i):
        return 'minimax_h3_image|picture_label=' + PICTURE_LABEL

    def load_text_encoder(self, paths, clip_type):
        if isinstance(paths, str):
            paths = [paths]
        sd = weights.load_text_encoder_state(paths)
        if sd is None:
            return super().load_text_encoder(paths, clip_type)
        return comfy.sd.load_text_encoder_state_dicts([sd], clip_type=clip_type, disable_dynamic=True)

    def load_vae(self, dtype):
        sd = weights.load_vae_state(self.model_config['vae'])
        vae = comfy.sd.VAE(sd=sd, dtype=dtype)
        vae.throw_exception_if_invalid()
        if not isinstance(vae.first_stage_model, comfy.ldm.minimax.vae.MiniMaxH3ImageVAE):
            raise ValueError(f'{self.model_config["vae"]} is not the MiniMax-H3-Image VAE (its decoder is a 2D ViT with 16 '
                             f'heads; the video VAE and the P-H3-Image one are other models)')
        return vae

    def load_model_patcher(self, model_options):
        path = self.model_config['diffusion_model']
        sd, metadata, layout, steps = weights.load_checkpoint(
            path, self.model_config.get('diffusion_model_format'), self.model_config.get('pdd_heads'))
        if is_main_process():
            print(f'MiniMax-H3-Image transformer: {layout} layout, {steps} PDD head(s)')
        patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=True)
        if patcher is None:
            raise RuntimeError(f'Could not detect the model type of {path}')
        return patcher

    def load_diffusion_model(self):
        super().load_diffusion_model()
        self.split_pdd_head()

    def split_pdd_head(self):
        """Head 0 of the bank becomes the trainable final_layer.video_out; the offsets of the later heads, which a
        sampling preview adds to it, are a frozen buffer."""
        final = self.diffusion_model.final_layer
        steps = final.pdd_steps
        if steps == 1:
            return
        out = final.video_out
        dim = final.video_dim
        weight = out.weight.data.reshape(steps, dim, -1)
        bias = out.bias.data.reshape(steps, dim)
        final.register_buffer('pdd_offset_weight', weight[1:].clone())
        final.register_buffer('pdd_offset_bias', bias[1:].clone())
        out.weight = nn.Parameter(weight[0].clone())
        out.bias = nn.Parameter(bias[0].clone())
        out.out_features = dim
        for name, p in (('weight', out.weight), ('bias', out.bias)):
            p.original_name = f'final_layer.video_out.{name}'
            p.requires_grad_(True)
        self.pdd_offsets = (final.pdd_offset_weight.detach().cpu(), final.pdd_offset_bias.detach().cpu())

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False, round_height=32, round_width=32)

    def encode_images(self, images, vae=None):
        """[B, C, H, W] (or [B, C, 1, H, W]) pixels in 0..1 -> [B, 24, 1, h, w] latents. One image per call: the VAE's
        posterior sample draws its noise from seed 42 for the shape it is given, so in a batch every image but the
        first would get other noise than ComfyUI gives it."""
        vae = self.vae if vae is None else vae
        if images.ndim == 5:
            assert images.shape[2] == 1, images.shape
            images = images.squeeze(2)
        images = images.to('cuda', self.comfy_vae_pixel_dtype)
        latents = [vae.encode(images[i:i + 1].movedim(1, -1)) for i in range(images.shape[0])]
        return self.latent_format.process_in(torch.cat(latents))

    def vae_encode(self, img, vae=None):
        return self.encode_images(img, vae)

    def get_call_vae_fn(self, vae):
        def fn(images, control_images=None):
            result = {'latents': self.encode_images(images, vae)}
            if control_images is not None:
                result['control_latents'] = self.encode_images(control_images, vae)
            return result
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        batched = super().get_call_text_encoder_fn(text_encoder)
        te_idx = self.text_encoders.index(text_encoder)

        def check_encoder():
            if getattr(text_encoder.cond_stage_model, 'clip', None) != 'qwen3vl_32b':
                raise ValueError('The text encoder is not the MiniMax H3 one (Qwen3-VL-32B cut to 50 layers)')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_file: list[str] | None = None):
            check_encoder()
            if any(is_video):
                raise ValueError('minimax_h3_image trains images only')
            if control_file is None or all(path is None for path in control_file):
                # Text to image: the captions of a batch together, padded and masked; every token is text.
                result = batched(captions, is_video)
                result[f'token_tags_{te_idx}'] = torch.ones_like(result[f'attention_mask_{te_idx}'], dtype=torch.long)
                return result
            # Edit: one caption at a time with its control image, as ComfyUI encodes it. The vision blocks of a
            # batch differ in length, so the rows are a list.
            embeds, masks, tags = [], [], []
            for caption, path in zip(captions, control_file):
                images = [] if path is None else [load_vision_image(path)]
                tokens = text_encoder.tokenize(caption, images=images, picture_label=PICTURE_LABEL)
                cond, extra = text_encoder.encode_from_tokens_scheduled(tokens)[0]
                assert cond.shape[0] == 1, cond.shape
                n = cond.shape[1]
                token_tags = extra.get('minimax_token_tags')
                token_tags = torch.ones(n, dtype=torch.long) if token_tags is None else token_tags.view(-1).long()
                assert token_tags.shape[0] == n, (token_tags.shape, n)
                embeds.append(cond[0].to(self.dtype))
                masks.append(torch.ones(n, dtype=torch.int64))
                tags.append(token_tags)
            return {
                f'text_embeds_{te_idx}': embeds,
                f'attention_mask_{te_idx}': masks,
                f'token_tags_{te_idx}': tags,
            }

        return fn

    def get_conds(self, inputs):
        text_embeds, attention_mask = super().get_conds(inputs)
        token_tags = inputs.get('token_tags_0')
        if token_tags is None:
            token_tags = torch.ones(attention_mask.shape, dtype=torch.long)
        else:
            n = text_embeds.shape[1]
            # Padding takes the text modality; it is masked out as a key.
            token_tags = torch.stack([torch.cat([u.long(), u.new_ones(n - u.shape[0], dtype=torch.long)]) for u in token_tags])
        assert token_tags.shape == attention_mask.shape, (token_tags.shape, attention_mask.shape)
        return text_embeds, attention_mask, token_tags.to(text_embeds.device)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        bs, c, f, h, w = latents.shape
        if f != 1:
            raise ValueError(f'minimax_h3_image trains images only, got latents with {f} frames')
        device = latents.device

        t = self.sample_timesteps(bs, f, h, w, device, timestep_quantile)
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)

        text_embeds, attention_mask, token_tags = self.get_conds(inputs)
        model_inputs = (noisy_latents, t, text_embeds, attention_mask, token_tags)
        control_latents = inputs.get('control_latents')
        if control_latents is not None:
            control_latents = control_latents.float()
            if control_latents.shape != latents.shape:
                raise ValueError(f'The control latents {tuple(control_latents.shape)} and the latents '
                                 f'{tuple(latents.shape)} do not match')
            # ComfyUI's visual condition noise augmentation.
            ref = VISUAL_COND_TIMESTEP * control_latents + (1.0 - VISUAL_COND_TIMESTEP) * torch.randn_like(control_latents)
            model_inputs = (*model_inputs, ref)
        return model_inputs, (target, mask)

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            return self.elementwise_loss(output, target, mask).mean()
        return loss_fn

    def get_param_groups(self, parameters):
        if not any(k in self.model_config for k in LR_KEYS):
            return super().get_param_groups(parameters)
        lr = self.config['optimizer']['lr']
        group_lrs = {
            'blocks': self.model_config.get('blocks_lr', lr),
            'token_refiner': self.model_config.get('token_refiner_lr', lr),
            'text_projection': self.model_config.get('text_projection_lr', lr),
            'patch_embed': self.model_config.get('patch_embed_lr', lr),
            'head': self.model_config.get('head_lr', lr),
        }
        groups = {k: [] for k in group_lrs}
        other = []
        for p in parameters:
            name = p.original_name
            if 'token_refiner' in name:
                groups['token_refiner'].append(p)
            elif 'condition_proj' in name:
                groups['text_projection'].append(p)
            elif 'video_patch_proj' in name:
                groups['patch_embed'].append(p)
            elif 'final_layer' in name:
                groups['head'].append(p)
            elif 'blocks.' in name:
                groups['blocks'].append(p)
            else:
                other.append(p)
        param_groups = []
        for key, params in groups.items():
            if group_lrs[key] == 0:
                for p in params:
                    p.requires_grad_(False)
            elif params:
                param_groups.append({'params': params, 'lr': group_lrs[key]})
        if other:
            param_groups.append({'params': other, 'lr': lr})
        if is_main_process():
            print('MiniMax-H3-Image param groups: ' + ', '.join(f'{k}: {len(v)} tensors @ lr={group_lrs[k]}' for k, v in groups.items()))
        return param_groups

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [ImageInitialLayer(diffusion_model, self.model_config.get('train_pdd_head', 'sigma') == 'sigma')]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        self.final_layer_module = ImageFinalLayer(diffusion_model)
        layers.append(self.final_layer_module)
        return layers

    def save_model(self, save_dir, sd):
        """A full fine-tune in ComfyUI's single-file layout: the PDD bank stacked again (the trained head 0 and the
        frozen offsets), the buffers ComfyUI reads (rope.inv_freq, adaln_t_table), and the sigma grid as metadata."""
        import safetensors.torch
        final = self.diffusion_model.final_layer
        sd = dict(sd)
        steps = final.pdd_steps
        if steps > 1:
            offset_weight, offset_bias = self.pdd_offsets
            sd['final_layer.video_out.weight'] = torch.cat([sd['final_layer.video_out.weight'].float().cpu(), offset_weight.flatten(0, 1)])
            sd['final_layer.video_out.bias'] = torch.cat([sd['final_layer.video_out.bias'].float().cpu(), offset_bias.flatten(0, 1)])
        for name, buffer in self.diffusion_model.named_buffers():
            if name.startswith('final_layer.pdd_offset'):
                continue
            sd[name] = buffer.detach().cpu()
        metadata = weights.transformer_metadata(steps, self.diffusion_model.pdd_sigmas)
        safetensors.torch.save_file(sd, save_dir / 'model.safetensors', metadata=metadata)

    @torch.no_grad()
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=1):
        te = self.text_encoders[0]
        te.load_model_if_needed()
        inputs = self.get_call_text_encoder_fn(te)([prompt], [False])
        self.conds = tuple(x.cuda() for x in self.get_conds(inputs))
        comfy.model_management.unload_all_models()

    @torch.no_grad()
    def sample(self, w=1024, h=1024, seed=0):
        """Text to image on the checkpoint's sigma grid (Euler, no CFG, one PDD head per step), as ComfyUI samples it."""
        h, w = h // 32 * 32, w // 32 * 32
        generator = torch.Generator(device='cuda').manual_seed(seed)
        x = torch.randn((1, self.channels, 1, h // 16, w // 16), device='cuda', generator=generator)
        sigmas = self.diffusion_model.pdd_sigmas
        heads = self.diffusion_model.final_layer.pdd_steps
        try:
            for i, (sigma, sigma_next) in enumerate(tqdm(list(zip(sigmas[:-1], sigmas[1:])), desc='Sampling')):
                self.final_layer_module.head = i if heads > 1 else 0
                t = torch.tensor([sigma], device='cuda')
                v = self.pipeline_model((x, t, *self.conds))
                x = x + (sigma_next - sigma) * v.float()
        finally:
            self.final_layer_module.head = None
        self.vae.load_model_if_needed()
        img = self.vae_decode(x)
        comfy.model_management.unload_all_models()
        if img.ndim == 5:
            img = img.squeeze(1)
        return img


class ImageInitialLayer(nn.Module):
    """Text, reference image and target image packed into one sequence, with their modulation rows and rope table."""

    def __init__(self, model, head_by_sigma=False):
        super().__init__()
        # With a PDD bank each sample trains through the head of the grid interval its noise level falls in, as sampling
        # applies it; otherwise head 0.
        self.pdd_grid = list(model.pdd_sigmas) if head_by_sigma and model.final_layer.pdd_steps > 1 else None
        if model.use_adaln_curves:
            self.adaln_t_table = model.adaln_t_table
        else:
            self.time_embedder = model.time_embedder
        self.video_patch_proj = model.video_patch_proj
        self.condition_proj = model.condition_proj
        self.rope = model.rope
        self.token_refiner = model.token_refiner
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def make_packed_sequence(self, video_x, t, context, context_mask, token_tags, ref_latent=None):
        assert video_x.shape[0] == 1  # one sample, with a batch dimension
        device = video_x.device
        dtype = context.dtype  # compute dtype
        refs = None
        if ref_latent is not None:
            refs = [{'kind': 'image', 'latent_h': ref_latent.shape[-2], 'latent_w': ref_latent.shape[-1]}]
        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        text_len = context.shape[1]
        # The captions of a batch are padded to the longest. ComfyUI never pads a prompt, so the sample is built as
        # it would be alone: the refiner sees only the real tokens, and the rope positions are those of the unpadded
        # layout. The padding rows go at the end of the text span, masked out as keys.
        real_text = context_mask.view(-1).to(torch.bool).clone()
        if not real_text.any():
            # An empty caption is one pad token, masked by the encoder; ComfyUI runs it as a one-token context.
            real_text[0] = True
        n_text = int(real_text.sum())
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, 0, refs=refs)

        # Training t is the noise level; the model is told 1 - t. Reference rows sit at max(1 - t, 0.999).
        t_v = float(1.0 - t.flatten()[0].float())
        t_ref = max(t_v, VISUAL_COND_TIMESTEP)
        unique_t = [t_v] + ([t_ref] if refs else [])
        kind_row = {'text': 0, 'video': 0, 'ref_img': 1}
        seg_tag = {'video': 0, 'ref_img': 0}  # text takes its tags from token_tags
        tags = token_tags.view(-1).tolist()
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = kind_row[kind] * 3
            if kind == 'text':
                # the vision tokens of the presentation carry the video modality: one segment per tag run
                assert len(tags) == b - a, (len(tags), b - a)
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        all_rows = video_rows
        if ref_latent is not None:
            # already noise-augmented by the pipeline (prepare_inputs)
            ref_rows = patchify_video(ref_latent.to(torch.float32), self.patch_size)
            img_update = layout.img_update.to(device)
            all_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_rows[~img_update] = ref_rows
            all_rows[img_update] = video_rows
        video_embed = self.video_patch_proj(all_rows).to(dtype)

        text_states = context[:, real_text]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states), transformer_options={})
        text_states = text_states[0]

        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        attention_mask = torch.ones((layout.seq_len,), dtype=context_mask.dtype, device=device)
        offset = 0
        for a, b, kind in layout.segments:
            if kind == 'text':
                h[a:a + n_text] = text_states.to(dtype)
                h[a + n_text:b] = 0
                attention_mask[a:a + n_text] = True
                attention_mask[a + n_text:b] = False
            else:
                h[a:b] = video_embed[offset:offset + b - a]
                offset += b - a

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        position_ids = layout.position_ids
        if n_text != text_len:
            unpadded = PackedLayout(n_text, latent_t, lat_h, lat_w, 0, refs=refs).position_ids
            assert unpadded.shape[0] == layout.seq_len - (text_len - n_text)
            # the padding rows take the positions after the real text; being masked, they reach no one
            pad_positions = position_ids[n_text:text_len]
            position_ids = torch.cat([unpadded[:n_text], pad_positions, unpadded[n_text:]])
        rope_freqs = rope_rotation_table(self.rope_freqs(position_ids, device), dtype)

        va, vb, _ = layout.segments[-1]  # the target image is the last segment
        mod_segments = torch.tensor(mod_segments, dtype=torch.int32, device=device)
        head = 0
        if self.pdd_grid is not None:
            # the interval [grid[k + 1], grid[k]] holds the noise level; the last interval takes the end point
            head = sum(1 for g in self.pdd_grid[1:-1] if float(t.flatten()[0]) <= g)
        extra_ints = torch.tensor([va, vb, kind_row['video'], latent_t, lat_h, lat_w, head], dtype=torch.int32, device=device)
        return h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints

    def make_layer_inputs(self, video_x, t, context, context_mask, token_tags, *ref):
        first_extra_ints = None
        heads = []
        h_list, attention_mask_list, t_emb_list, rope_freqs_list, segments_list = [], [], [], [], []
        for items in zip(video_x, t, context, context_mask, token_tags, *ref):
            h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints = self.make_packed_sequence(*(x.unsqueeze(0) for x in items))
            if first_extra_ints is None:
                first_extra_ints = extra_ints
            else:
                # ImageFinalLayer reads the target span and timestep row of the first sample for all.
                assert (extra_ints[:6] == first_extra_ints[:6]).all()
            heads.append(extra_ints[6:])
            segments_list.append(mod_segments)
            h_list.append(h)
            attention_mask_list.append(attention_mask)
            t_emb_list.append(t_emb)
            rope_freqs_list.append(rope_freqs)

        if all(s.shape == segments_list[0].shape and (s == segments_list[0]).all() for s in segments_list[1:]):
            mod_segments = segments_list[0]
        else:
            # Samples whose segments differ get one list each, [batch, segments, 3]; the shorter ones are padded
            # with empty segments (start = stop), which modulate nothing.
            n = max(s.shape[0] for s in segments_list)
            mod_segments = torch.zeros((len(segments_list), n, 3), dtype=segments_list[0].dtype, device=segments_list[0].device)
            for i, s in enumerate(segments_list):
                mod_segments[i, :s.shape[0]] = s
                mod_segments[i, s.shape[0]:, :2] = s[-1, 1]

        h = torch.stack(h_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)[:, None, None, :]  # 4D bool for SDPA
        t_emb = torch.stack(t_emb_list)
        rope_freqs = torch.cat(rope_freqs_list, dim=0)
        # the shared layout, then each sample's PDD head
        extra_ints = torch.cat([first_extra_ints[:6], *heads])
        outputs = make_contiguous(h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        # (video, t, text, text mask, token tags[, reference latents])
        video_x, t, context, context_mask, token_tags, *ref = inputs
        return self.make_layer_inputs(video_x, t, context, context_mask, token_tags, *ref)


class ImageFinalLayer(nn.Module):
    # The PDD head every sample of a forward uses: a sampling preview walks 0..n-1. None: each sample's own, as the
    # initial layer chose it from its noise level.
    head = None

    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints = inputs
        va, vb, row, latent_t, lat_h, lat_w, *heads = extra_ints.tolist()
        final = self.final_layer
        shift, scale = final.adaln_proj(t_emb)
        hv = final.modulate(h, shift, scale, (va, vb, row))
        out = final.video_out
        if self.head is not None:
            heads = [self.head] * hv.shape[0]
        assert len(heads) == hv.shape[0], (len(heads), hv.shape[0])
        # The head is the checkpoint's fp32 island: no autocast. Head k is head 0 plus the offset of the bank.
        with torch.autocast('cuda', enabled=False):
            weights = [out.weight if k == 0 else out.weight + final.pdd_offset_weight[k - 1] for k in heads]
            biases = [out.bias if k == 0 else out.bias + final.pdd_offset_bias[k - 1] for k in heads]
            if all(k == heads[0] for k in heads):
                v = F.linear(hv.float(), weights[0].float(), biases[0].float())
            else:
                v = torch.einsum('bsd,bod->bso', hv.float(), torch.stack(weights).float()) + torch.stack(biases).float()[:, None]
        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        return -video_out
