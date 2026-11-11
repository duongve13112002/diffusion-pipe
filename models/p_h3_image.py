"""P-H3-Image (PrunaAI): a MiniMax H3 fine-tune for text to image and image editing.

The transformer is MiniMax H3's (ComfyUI's comfy.ldm.minimax), sampled in 8 steps without CFG at
ModelSamplingMiniMaxH3 1.0 / 1.0, so the audio stream sits at the image's sigma. Its text encoder is
Qwen3-VL-4B with a text connector (ComfyUI loads it with type 'minimax'), and its VAE is the H3
encoder with a single-frame decoder (PH3ImageVAE).

Text to image: images and captions, as for MiniMax H3, with the two audio latents a still image
spans (noise at the image's sigma, no loss).

Editing (a dataset directory with control_path, and control_path_2, ..., control_path_9 for further reference
images): as ComfyUI's PH3ImageConditioning node builds it. The caption gets "Edit <Picture 1>. " in front unless it
already names the picture to edit (edit_prompt_prefix = false leaves every caption as it is), and every image is shown
to the text encoder at <= 1 MP. The vision blocks' tokens are tagged 0 (they take the video modulation), the rest 1; the
tags travel with the embeddings. The latent of the first image (control_path, the one edited; cropped to the target's
bucket, so it stays aligned with the target) and of each further one (scaled within the target's pixel area, its own
aspect ratio, sides a multiple of 32, as ComfyUI sizes them) are the references, packed after the text in the order of
the pictures, noise-augmented to t = 0.999, pinned at max(t, 0.999) like ComfyUI's minimax_refs. Samples with references
of different sizes share a batch: their sequences are padded at the end and the padding is masked.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.base import PreprocessMediaFile, convert_to_rgb
from models.minimax_h3 import MinimaxH3Pipeline, InitialLayer, TransformerLayer, FinalLayer, pack_references
from models.minimax_h3_image import control_key
from utils.common import AUTOCAST_DTYPE
import comfy.model_management
import comfy.sd
import comfy.text_encoders.minimax
import comfy.utils
import comfy.ldm.minimax.vae

# The checkpoint's sampling grid (8 steps).
SAMPLE_SIGMAS = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]
# ComfyUI comfy_extras/nodes_minimax_h3.py: reference images reach Qwen at <= 1 MP, sides /32.
VISION_PIXELS = 1000000
CANVAS_MULTIPLE = 32
# ComfyUI's PH3ImageConditioning takes images image_1 ... image_9.
MAX_REFERENCES = 9
CONTROL_KEYS = tuple(control_key(n) for n in range(MAX_REFERENCES))
EDIT_PROMPT = 'Edit <Picture 1>. {}'
# An instruction that already names the picture to edit, or an edit brief from the prompt
# upsampler, is encoded as it is.
EDIT_VERBATIM = re.compile(r'\s*(Edit <Picture \d+>|subject_definitions:)')
# A still image spans two audio latents (ComfyUI's _av_latent(w, h, 1, 2)).
AUDIO_LATENTS = 2
AUDIO_CHANNELS = 32
# Keys of MiniMax H3 that have no meaning here.
REFUSED_MODEL_KEYS = {
    'audio_vae': 'P-H3-Image has no audio VAE: it trains images only',
    'cfg': 'P-H3-Image is step- and guidance-distilled and samples without CFG; CFG-augmented training does not apply',
}


class PH3ImageClipModel(comfy.text_encoders.minimax.PH3ImageClipModel):
    # Attention masks on, as models/minimax_h3.py turns them on for H3: a batch of captions is
    # padded to its longest, and training needs to know where each caption ends.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_attention_masks = True
        self.return_attention_masks = True

# PH3ImageTEModel looks the class up by this name when it builds the encoder.
comfy.text_encoders.minimax.PH3ImageClipModel = PH3ImageClipModel


def size_within_area(width, height, pixels):
    """ComfyUI's _size_within_area: aspect-preserving scale (down only) to `pixels`, sides /32."""
    scale = min(1.0, (pixels / (width * height)) ** 0.5)
    return tuple(max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE) for side in (width, height))


def load_within_area(path, pixels):
    """The image ([1, H, W, 3] in 0..1) scaled as ComfyUI's _resize_to scales it for a pixel area (down only, the aspect
    ratio kept up to the sides being a multiple of 32, the whole image stretched with lanczos, untouched when already
    there). Decoded as the dataset decodes the same file (convert_to_rgb: no EXIF rotation, transparency onto white), so
    the encoder's picture and the reference latent see one image; for an RGB file without EXIF rotation that is what
    ComfyUI's LoadImage gives."""
    with Image.open(path) as img:
        img = convert_to_rgb(img)
        image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]
    size = size_within_area(image.shape[2], image.shape[1], pixels)
    if (image.shape[2], image.shape[1]) == size:
        return image
    samples = comfy.utils.common_upscale(image.movedim(-1, 1), size[0], size[1], 'lanczos', 'disabled')
    return samples.movedim(1, -1)


def load_vision_image(path):
    """The image at the size PH3ImageConditioning shows it to the text encoder."""
    return load_within_area(path, VISION_PIXELS)


class ImagePreprocess(PreprocessMediaFile):
    def preprocess_control(self, path, size_bucket, index=0):
        """Reference `index` (0: control_path) of a sample whose target has the size bucket (width, height, frames). The
        first is cropped to the bucket, as the target is, so the edit stays aligned with it. The others are scaled within the
        target's pixel area with their own aspect ratio, as ComfyUI sizes the images after the first. [3, H, W] in 0..1."""
        if index == 0:
            return self((None, path), None, size_bucket)[0][0]
        return load_within_area(path, size_bucket[0] * size_bucket[1])[0].movedim(-1, 0)


def edit_prompt(caption):
    return caption.strip() if EDIT_VERBATIM.match(caption) else EDIT_PROMPT.format(caption)


class PH3ImagePipeline(MinimaxH3Pipeline):
    name = 'p_h3_image'
    is_video_vae = False

    def __init__(self, config):
        model_config = config['model']
        for key, reason in REFUSED_MODEL_KEYS.items():
            if key in model_config:
                raise ValueError(f'[model] {key} does not apply to p_h3_image: {reason}. Remove it.')
        text_encoders = model_config.get('text_encoders', [])
        if len(text_encoders) != 1 or str(text_encoders[0].get('type', '')).lower() != 'minimax':
            raise ValueError("p_h3_image needs one text encoder, the P-H3-Image Qwen3-VL-4B file with type = 'minimax'")
        edit_prompt_prefix = model_config.get('edit_prompt_prefix', True)
        if not isinstance(edit_prompt_prefix, bool):
            raise ValueError(f'[model] edit_prompt_prefix must be true or false, got {edit_prompt_prefix!r}')
        super().__init__(config)
        self.is_video_vae = False
        self.edit_prompt_prefix = edit_prompt_prefix

    max_control_images = MAX_REFERENCES

    def model_specific_dataset_config_validation(self, dataset_config):
        directories = dataset_config.get('directory', [])
        for where, config in [('the dataset', dataset_config)] + [(f"[[directory]] {d.get('path')}", d) for d in directories]:
            frame_buckets = config.get('frame_buckets', [1])
            if any(f != 1 for f in frame_buckets):
                raise ValueError(f'p_h3_image trains images only; frame_buckets must be [1], got {frame_buckets} in {where}')
        for directory in directories:
            # A dropped caption trains on the one cached empty-caption embedding, which has no
            # control image; ComfyUI never pairs that with a reference ("Edit <Picture 1>. " and
            # the image's vision tokens are what an empty edit prompt encodes to).
            # utils/dataset.py's order: the directory's rate, the dataset's, then the training
            # config's uncond_fraction.
            if 'caption_dropout_rate' in directory:
                dropout, where = directory['caption_dropout_rate'], 'caption_dropout_rate in it'
            elif 'caption_dropout_rate' in dataset_config:
                dropout, where = dataset_config['caption_dropout_rate'], 'the dataset caption_dropout_rate'
            else:
                dropout, where = self.config.get('uncond_fraction', 0), 'uncond_fraction of the training config'
            if 'control_path' in directory and dropout:
                raise ValueError(f"caption dropout ({where}) does not apply to the edit directory "
                                 f"{directory.get('path')} of p_h3_image: a dropped caption would lose its control "
                                 'image. Set caption_dropout_rate = 0 in that [[directory]].')

    def text_encoder_cache_key(self, i):
        # How a caption with a control image is presented to the encoder.
        return f'p_h3_image|edit_prompt_prefix={self.edit_prompt_prefix}'

    def load_vae(self, dtype):
        sd = comfy.utils.load_torch_file(self.model_config['vae'])
        vae = comfy.sd.VAE(sd=sd, dtype=dtype)
        vae.throw_exception_if_invalid()
        if not isinstance(vae.first_stage_model, comfy.ldm.minimax.vae.PH3ImageVAE):
            raise ValueError(f'{self.model_config["vae"]} is not the P-H3-Image VAE (its decoder decodes single '
                             f'frames; the MiniMax H3 video VAE does not)')
        return vae

    def load_diffusion_model(self):
        super().load_diffusion_model()
        # ModelSamplingMiniMaxH3 at 1.0 / 1.0: the audio sits at the image's sigma.
        self.diffusion_model.sigma_shift_video = 1.0
        self.diffusion_model.sigma_shift_audio = 1.0

    def get_preprocess_media_file_fn(self):
        return ImagePreprocess(self.config, support_video=False, round_height=32, round_width=32)

    def encode_images(self, images, vae=None):
        """[B, C, H, W] (or [B, C, 1, H, W]) pixels in 0..1 -> [B, 24, 1, h, w] latents. One image
        per call: the VAE's posterior sample draws its noise from seed 42 for the shape it is given,
        so in a batch every image but the first would get other noise than ComfyUI gives it."""
        vae = self.vae if vae is None else vae
        if images.ndim == 5:
            assert images.shape[2] == 1, images.shape
            images = images.squeeze(2)
        images = images.to('cuda', self.comfy_vae_pixel_dtype)
        latents = [vae.encode(images[i:i + 1].movedim(1, -1)) for i in range(images.shape[0])]
        return self.latent_format.process_in(torch.cat(latents))

    def vae_encode(self, img, vae=None):
        return self.encode_images(img, vae)

    vae_latent_keys = ('latents',) + CONTROL_KEYS

    def get_call_vae_fn(self, vae):
        def fn(images, control_images=None):
            """control_images: one list per sample of its reference images, [3, H, W] (or [3, 1, H, W]) each, sizes may
            differ, or [B, 3, H, W] for one reference per sample. The latents are one list per reference, an entry per
            sample."""
            result = {'latents': self.encode_images(images, vae)}
            if control_images is not None:
                if torch.is_tensor(control_images):
                    control_images = [[image] for image in control_images]
                for n in range(len(control_images[0])):
                    result[control_key(n)] = [self.encode_images(sample[n][None], vae)[0] for sample in control_images]
            return result
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        batched = super().get_call_text_encoder_fn(text_encoder)
        te_idx = self.text_encoders.index(text_encoder)

        def check_encoder():
            if getattr(text_encoder.cond_stage_model, 'clip', None) != 'qwen3vl_4b':
                raise ValueError('The text encoder is not the P-H3-Image one (Qwen3-VL-4B with its text connector)')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_file: list[str | list[str]] | None = None):
            check_encoder()
            if any(is_video):
                raise ValueError('p_h3_image trains images only')
            if control_file is None or all(path is None for path in control_file):
                # Text to image: the captions of a batch together, padded and masked; every token
                # is text.
                result = batched(captions, is_video)
                result[f'token_tags_{te_idx}'] = torch.ones_like(result[f'attention_mask_{te_idx}'], dtype=torch.long)
                return result
            # Edit: one caption at a time with its control image, as ComfyUI encodes it. The vision
            # blocks of a batch differ in length, so the rows are a list.
            embeds, masks, tags = [], [], []
            for caption, paths in zip(captions, control_file):
                # one path, or the list of a sample's reference images
                paths = [] if paths is None else [paths] if isinstance(paths, str) else list(paths)
                images = [load_vision_image(path) for path in paths]
                if images and self.edit_prompt_prefix:
                    caption = edit_prompt(caption)
                tokens = text_encoder.tokenize(caption, images=images)
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
            raise ValueError(f'p_h3_image trains images only, got latents with {f} frames')
        device = latents.device

        t = self.sample_timesteps(bs, f, h, w, device, timestep_quantile)
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        # The two audio latents of a still: ComfyUI's empty latent (zeros) noised at the image's
        # sigma (shifts 1.0 / 1.0). Their output has no loss.
        noisy_audio = t.view(-1, 1, 1, 1) * torch.randn((bs, AUDIO_CHANNELS, 2, AUDIO_LATENTS), device=device)
        valid_audio = torch.ones(bs, dtype=torch.bool, device=device)

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)

        text_embeds, attention_mask, token_tags = self.get_conds(inputs)
        model_inputs = (noisy_latents, noisy_audio, valid_audio, t, text_embeds, attention_mask, token_tags)
        keys = [key for key in CONTROL_KEYS if key in inputs]
        if keys:
            # the first reference is the image to edit, cropped to the target's bucket
            for i, control in enumerate(inputs['control_latents']):
                if control.shape != latents[i].shape:
                    raise ValueError(f'The control latents {tuple(control.shape)} and the latents '
                                     f'{tuple(latents[i].shape)} do not match')
            model_inputs = (*model_inputs, *pack_references([inputs[key] for key in keys], bs))
        return model_inputs, (target, mask)

    def get_loss_fn(self):
        def loss_fn(outputs, label):
            output = outputs[0]  # the audio output is not trained
            target, mask = label
            return self.elementwise_loss(output, target, mask).mean()
        return loss_fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [PH3InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model, 1.0))
        return layers

    @torch.no_grad()
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=1):
        te = self.text_encoders[0]
        te.load_model_if_needed()
        inputs = self.get_call_text_encoder_fn(te)([prompt], [False])
        self.conds = tuple(x.cuda() for x in self.get_conds(inputs))
        comfy.model_management.unload_all_models()

    @torch.no_grad()
    def sample(self, w=1024, h=1024, seed=0):
        """Text to image with the checkpoint's 8 sigmas (Euler, no CFG), as ComfyUI samples it."""
        h, w = h // 32 * 32, w // 32 * 32
        generator = torch.Generator(device='cuda').manual_seed(seed)
        x = torch.randn((1, self.channels, 1, h // 16, w // 16), device='cuda', generator=generator)
        audio = torch.randn((1, AUDIO_CHANNELS, 2, AUDIO_LATENTS), device='cuda', generator=generator)
        valid_audio = torch.ones(1, dtype=torch.bool, device='cuda')
        sigmas = SAMPLE_SIGMAS
        for sigma, sigma_next in tqdm(list(zip(sigmas[:-1], sigmas[1:])), desc='Sampling'):
            t = torch.tensor([sigma], device='cuda')
            v, v_audio = self.pipeline_model((x, audio, valid_audio, t, *self.conds))
            x = x + (sigma_next - sigma) * v.float()
            audio = audio + (sigma_next - sigma) * v_audio.float()
        self.vae.load_model_if_needed()
        img = self.vae_decode(x)
        comfy.model_management.unload_all_models()
        if img.ndim == 5:
            img = img.squeeze(1)
        return img


class PH3InitialLayer(InitialLayer):
    fixed_timestep_rows = True

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        # (video, audio, valid_audio, t, text, text mask, token tags[, reference patch rows, their latent sizes])
        video_x, audio_x, valid_audio, t, context, context_mask, token_tags, *ref = inputs
        return self.make_layer_inputs(video_x, audio_x, valid_audio, t, context, context_mask, token_tags, *ref)
