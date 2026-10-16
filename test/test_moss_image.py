"""MOSS-Image pipeline (models/moss_image.py) against ComfyUI's own MOSS-Image, on CPU.

The reference is comfy.ldm.moss_image.model.MOSSImage's own forward, the module ComfyUI runs, on
tiny randomly initialised models, so a mismatch is a bug here rather than a tolerance question.
Where the Hugging Face repo's pipeline_moss_image.py is available locally, the original
MOSSImageTransformer and MOSSImagePipeline are checked too.
"""

import json
import logging
import math
import re
from pathlib import Path

import pytest
import safetensors.torch
import torch

import comfy.model_management
import comfy.ops
from comfy.ldm.moss_image.model import MOSSImage

from models import moss_image as mi

TEXT_DIM = 48
TINY = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=TEXT_DIM, latent_channels=128)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    # deepspeed may be installed without a process group; these tests are rank 0.
    import models.base
    for module in (mi, models.base):
        monkeypatch.setattr(module, 'is_main_process', lambda: True)


def tiny_reference(seed=0, dtype=torch.float32, **overrides):
    torch.manual_seed(seed)
    kwargs = {**TINY, **overrides}
    model = MOSSImage(**kwargs, dtype=dtype, operations=comfy.ops.manual_cast)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and 'norm' in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
    return model


def tiny_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def hf_state(model):
    """The state dict of the diffusers MOSSImageTransformer: ComfyUI's plus its rope.bands buffer."""
    return {**tiny_state(model), 'rope.bands': mi.default_rope_bands(model.layers[0].q_norm.weight.shape[0])}


def reference_forward(model, x, sigma, text, mask, transformer_options=None, inference=False):
    """ComfyUI's forward on each sample at its own text length, as ComfyUI runs MOSS-Image (it
    never pads a cond). inference=True takes the comfy-kitchen kernels instead of the
    differentiable path diffusion-pipe trains on."""
    saved = comfy.model_management.in_training
    comfy.model_management.in_training = not inference
    try:
        sigma = sigma.reshape(-1).expand(x.shape[0])
        return torch.cat([
            model(x[i:i + 1], sigma[i:i + 1], text[i:i + 1, mask[i].bool()],
                  transformer_options=dict(transformer_options or {}))
            for i in range(x.shape[0])])
    finally:
        comfy.model_management.in_training = saved


class Runner:
    """What DeepSpeed's PipelineModule does on one stage: the layers in order."""

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def bare_pipeline(model_config=None, config=None):
    pipe = mi.MossImagePipeline.__new__(mi.MossImagePipeline)
    pipe.config = config or {}
    pipe.model_config = model_config or {}
    pipe.dtype = torch.float32
    pipe.text_dim = TEXT_DIM
    pipe.offloader = mi.ModelOffloader('dummy', [], 0, 0, True, torch.device('cpu'), False, debug=False)
    return pipe


def loaded_pipeline(reference, dtype=torch.float32):
    patcher = mi.load_transformer_patcher(tiny_state(reference), None, dtype)
    pipe = bare_pipeline()
    pipe.dtype = dtype
    pipe.diffusion_model = patcher.model.diffusion_model
    pipe.layer_state = mi.LayerState(pipe.diffusion_model.rope, dtype)
    pipe.pipeline_model = Runner(pipe.to_layers())
    return pipe


def text_batch(lengths, dim=TEXT_DIM, seed=1):
    torch.manual_seed(seed)
    embeds = [torch.randn(n, dim) for n in lengths]
    pipe = bare_pipeline()
    return embeds, pipe.get_conds({'prompt_embeds': embeds})


@pytest.mark.parametrize('inference', [False, True], ids=['training path', 'inference kernels'])
@pytest.mark.parametrize('lengths', [[7], [5, 9, 5], [6, 6]])
def test_layers_match_the_reference_forward(lengths, inference):
    reference = tiny_reference()
    pipe = loaded_pipeline(reference)
    _, (text, mask) = text_batch(lengths)
    batch = len(lengths)
    x = torch.randn(batch, 128, 4, 6)
    t = torch.linspace(0.003, 0.997, batch)
    with torch.no_grad():
        expected = reference_forward(reference, x, t, text, mask, inference=inference)
        got = pipe.pipeline_model((x, t, text, mask))
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('lengths', [[7], [5, 9, 5]])
def test_layers_match_the_original_transformer(lengths):
    """Against the Hugging Face repo's MOSSImageTransformer (timesteps on 0..1000, a text mask)."""
    module = reference_pipeline_module()
    reference = tiny_reference()
    hf = module.MOSSImageTransformer(**TINY).eval()
    hf.load_state_dict(hf_state(reference), strict=True)
    pipe = loaded_pipeline(reference)
    _, (text, mask) = text_batch(lengths)
    batch = len(lengths)
    x = torch.randn(batch, 128, 4, 6)
    t = torch.linspace(0.003, 0.997, batch)
    with torch.no_grad():
        expected = hf(x, t * 1000, text, encoder_attention_mask=mask).sample
        got = pipe.pipeline_model((x, t, text, mask))
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


# ---- text encoder ------------------------------------------------------------------------------

TINY_TEXT = dict(
    hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
    linear_value_head_dim=16, linear_conv_kernel_dim=4, vocab_size=248320,
)


def tiny_hf_text_encoder(seed=0):
    from transformers import Qwen3_5Config, Qwen3_5Model
    torch.manual_seed(seed)
    text_config = dict(TINY_TEXT, layer_types=['linear_attention'] * 3 + ['full_attention'],
                       rope_parameters={'rope_type': 'default', 'rope_theta': 10000000.0,
                                        'partial_rotary_factor': 0.25, 'mrope_interleaved': True,
                                        'mrope_section': [11, 11, 10]})
    vision_config = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    model = Qwen3_5Model(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'norm' in name and p.ndim == 1:
                p.copy_(0.1 * torch.randn_like(p))
            elif p.ndim > 1:
                p.copy_(0.05 * torch.randn_like(p))
    return model


def tiny_comfy_text_encoder(state):
    import comfy.sd
    import comfy.sd1_clip
    import comfy.supported_models_base

    overrides = dict(TINY_TEXT, lm_head=False, layer_types=['linear_attention'] * 3 + ['full_attention'])
    overrides.pop('linear_conv_kernel_dim')

    class TinyClip(comfy.sd1_clip.SDClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            super().__init__(device=device, dtype=dtype, layer='last', textmodel_json_config=overrides,
                             model_class=mi.Qwen35TextOnly, special_tokens={'pad': 248044},
                             enable_attention_masks=True, return_attention_masks=True, model_options=model_options)

    class TinyTE(comfy.sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            super().__init__(device=device, dtype=torch.float32, name='qwen35_9b', clip_model=TinyClip,
                             model_options=model_options)

    target = comfy.supported_models_base.ClipTarget(mi.MossImageTokenizer, TinyTE)
    return comfy.sd.CLIP(target, state_dict=[state], disable_dynamic=True)


def test_text_encoder_matches_transformers():
    hf = tiny_hf_text_encoder()
    hf_state = {k: v.clone() for k, v in hf.state_dict().items()}
    assert any(k.startswith('visual.') for k in hf_state) and any(k.startswith('language_model.') for k in hf_state)
    state = mi.qwen35_comfy_layout(mi.qwen35_vlm_layout(hf_state))
    assert not any(k.startswith(('visual.', 'model.visual.')) for k in state)
    clip = tiny_comfy_text_encoder(state)
    assert clip.cond_stage_model.qwen35_9b.transformer.visual is None

    pipe = bare_pipeline({'dtype': torch.float32})
    pipe.on_caption_overflow = 'warn'
    pipe.text_dim = 64
    for caption in ['a red fox in the snow', '{"summary": {"caption": "x"}, "items": []}']:
        got = pipe.encode_prompt(clip, caption)
        ids = [token for token, _ in clip.tokenize(caption)['qwen35_9b'][0]]
        with torch.no_grad():
            expected = hf(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]
        assert got.shape == expected.shape == (len(ids), 64)
        torch.testing.assert_close(got.float(), expected, rtol=2e-4, atol=2e-4)


def test_prompt_template_matches_the_chat_template():
    tokenizer_dir = Path('/home/user/moss-weights/Base/tokenizer')
    if not tokenizer_dir.is_dir():
        pytest.skip('needs the MOSS-Image tokenizer folder')
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    ours = mi.MossImageTokenizer()
    # The template trims the prompt (render_content(...)|trim); so must we.
    for prompt in ['a cat', 'Ảnh một con mèo, "quoted" (weighted:1.5) text', '{"items": [{"bbox": [1, 2, 3, 4]}]}',
                   '  a cat\n', '\n\ta dog, ', '   ']:
        formatted = hf.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        expected = hf(formatted)['input_ids']
        got = [token for token, _ in ours.tokenize_with_weights(prompt)['qwen35_9b'][0]]
        assert got == expected, prompt


def test_long_prompts_are_truncated_with_a_warning(capsys):
    tokenizer = mi.MossImageTokenizer()
    caption = 'word ' * 5000
    tokens = tokenizer.tokenize_with_weights(caption)['qwen35_9b'][0]
    assert len(tokens) == mi.MAX_TEXT_TOKENS
    assert tokenizer.last_untruncated_length > mi.MAX_TEXT_TOKENS


def test_text_encoder_layouts_are_detected():
    # Shapes are what detection reads; values are irrelevant.
    def layout(prefix):
        return {
            f'{prefix}layers.0.linear_attn.A_log': torch.zeros(32),
            f'{prefix}layers.0.input_layernorm.weight': torch.zeros(4096),
            f'{prefix}embed_tokens.weight': torch.zeros(4, 4096),
        }
    for prefix, vision in [('language_model.', 'visual.'), ('model.language_model.', 'model.visual.'), ('model.', 'visual.')]:
        state = {**layout(prefix), f'{vision}blocks.0.attn.qkv.weight': torch.zeros(1), 'lm_head.weight': torch.zeros(1)}
        out = mi.text_encoder_state(state)
        assert sorted(out) == ['model.embed_tokens.weight', 'model.layers.0.input_layernorm.weight',
                               'model.layers.0.linear_attn.A_log'], prefix
    small = layout('language_model.')
    small['language_model.layers.0.input_layernorm.weight'] = torch.zeros(2560)
    with pytest.raises(ValueError, match='Qwen3.5-9B'):
        mi.text_encoder_state(small)


# ---- noise levels and the inference schedule ---------------------------------------------------

REFERENCE_PIPELINE = Path('/home/user/MOSS-Image-hf/Base/pipeline_moss_image.py')


def reference_pipeline_module():
    if not REFERENCE_PIPELINE.is_file():
        pytest.skip('needs the reference pipeline_moss_image.py from the MOSS-Image Hugging Face repo')
    import importlib.util
    spec = importlib.util.spec_from_file_location('reference_pipeline_moss_image', REFERENCE_PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_sigmas_match_the_reference_values():
    sigmas = mi.inference_sigmas(50, 5.0)
    assert len(sigmas) == 50
    torch.testing.assert_close(sigmas[:5], torch.tensor([1.0, 0.996, 0.992, 0.987, 0.983]), rtol=0, atol=1e-6)
    torch.testing.assert_close(sigmas[-3:], torch.tensor([0.193, 0.115, 0.024]), rtol=0, atol=1e-6)
    # Every sigma is a point of the training grid.
    grid = mi.training_grid(dtype=torch.float32)
    assert all(torch.isclose(grid, s, atol=1e-6).any() for s in sigmas)


@pytest.mark.parametrize('steps,shift', [(50, 5.0), (28, 3.0), (4, 1.0), (1000, 5.0)])
def test_inference_sigmas_match_moss_image_scheduler(steps, shift):
    reference = reference_pipeline_module().MOSSImageScheduler(shift=shift)
    reference.set_timesteps(steps)
    expected = reference.sigmas[:-1]
    torch.testing.assert_close(mi.inference_sigmas(steps, shift), expected, rtol=0, atol=0)
    torch.testing.assert_close(mi.inference_sigmas(steps, shift) * 1000, reference.timesteps, rtol=0, atol=0)


def monte_carlo_levels(n, mu, s, cutoff, rho, total=1000, seed=0):
    """Eq. 4 sampled the way it is defined: S = 1 - sigmoid(Z), rejected above the cutoff,
    j = ceil(T S / a), and with probability rho a uniform level instead."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(4 * n, generator=g, dtype=torch.float64) * s + mu
    S = 1 - torch.sigmoid(z)
    S = S[S <= cutoff][:n]
    j = torch.ceil(total * S / cutoff).clamp(1, total).long()
    uniform = torch.rand(n, generator=g) < rho
    j[uniform] = torch.randint(1, total + 1, (int(uniform.sum()),), generator=g)
    return j


def test_noise_levels_follow_the_reports_distribution():
    sampler = mi.NoiseLevelSampler('logit_normal', -1.09, 0.95, 0.95, 0.01, 1.0, 0.0, 1.0)
    p = sampler.probabilities
    assert p.shape == (1000,) and math.isclose(float(p.sum()), 1.0, rel_tol=1e-12)
    j = monte_carlo_levels(400_000, -1.09, 0.95, 0.95, 0.01)
    empirical = torch.bincount(j - 1, minlength=1000).double() / len(j)
    # Compared on 20 bins of 50 levels: each bin holds thousands of samples.
    torch.testing.assert_close(empirical.view(20, 50).sum(1), p.view(20, 50).sum(1), rtol=0, atol=3e-3)
    sigmas = sampler.sample(200_000, generator=torch.Generator().manual_seed(1)).double()
    levels = torch.round(sigmas * 1000).long()
    assert torch.allclose(sigmas, levels.double() / 1000, atol=1e-7)
    drawn = torch.bincount(levels - 1, minlength=1000).double() / len(levels)
    torch.testing.assert_close(drawn.view(20, 50).sum(1), p.view(20, 50).sum(1), rtol=0, atol=3e-3)
    # The cutoff stretches the truncated logit-normal over the whole grid: the top level is
    # reachable (by the logit-normal part, not only the uniform one).
    assert float(p[-1]) > 0.01 / 1000


def test_noise_level_options():
    uniform = mi.NoiseLevelSampler('uniform', 0, 1, 1, 0, 1.0, 0.0, 1.0)
    torch.testing.assert_close(uniform.probabilities, torch.full((1000,), 1e-3, dtype=torch.float64))
    bounded = mi.NoiseLevelSampler('logit_normal', -1.09, 0.95, 0.95, 0.01, 1.0, 0.25, 0.5)
    drawn = bounded.sample(10_000)
    assert float(drawn.min()) >= 0.25 and float(drawn.max()) <= 0.5
    assert float(bounded.sample(1, quantile=0.0)) == 0.25 and float(bounded.sample(1, quantile=1.0)) == 0.5
    quantiles = [float(bounded.sample(1, quantile=q)) for q in (0.1, 0.5, 0.9)]
    assert quantiles == sorted(quantiles)
    shifted = mi.NoiseLevelSampler('uniform', 0, 1, 1, 0, 3.0, 0.0, 1.0)
    torch.testing.assert_close(shifted.sigmas.float(), reference_grid_or_formula(3.0))
    for bad in [dict(min_t=0.6, max_t=0.5), dict(shift=0.0), dict(sigma_cutoff=0.0)]:
        kwargs = dict(method='logit_normal', logit_mean=0.0, logit_std=1.0, sigma_cutoff=1.0,
                      uniform_fraction=0.0, shift=1.0, min_t=0.0, max_t=1.0)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            mi.NoiseLevelSampler(**kwargs)


def reference_grid_or_formula(shift, total=1000):
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, total, dtype=torch.float64)
    return (shift * values / (1 + (shift - 1) * values)).flip(0).float()


def test_prepare_inputs_builds_the_flow_matching_target():
    pipe = bare_pipeline()
    pipe.noise_sampler = mi.NoiseLevelSampler('logit_normal', -1.09, 0.95, 0.95, 0.01, 1.0, 0.0, 1.0)
    latents = torch.randn(2, 128, 3, 5)
    embeds = [torch.randn(4, TEXT_DIM), torch.randn(7, TEXT_DIM)]
    torch.manual_seed(5)
    (noisy, t, text, mask), (target, loss_mask) = pipe.prepare_inputs(
        {'latents': latents, 'mask': None, 'prompt_embeds': embeds}, timestep_quantile=0.5)
    sigma = t
    torch.manual_seed(5)
    noise = torch.randn_like(latents)
    s = sigma.view(-1, 1, 1, 1)
    torch.testing.assert_close(noisy, (1 - s) * latents + s * noise)
    torch.testing.assert_close(target, noise - latents)
    assert text.shape == (2, 7, TEXT_DIM) and mask.tolist() == [[True] * 4 + [False] * 3, [True] * 7]
    assert loss_mask is None
    assert float(sigma[0]) == float(pipe.noise_sampler.sample(1, quantile=0.5))


# ---- sampling ----------------------------------------------------------------------------------

def reference_velocity(reference, x, t, conds, skip=()):
    """ComfyUI's forward with blocks skipped the way SkipLayerGuidanceDiTSimple skips them."""
    replace = {('single_block', i): (lambda args, extra: {'img': args['img']}) for i in skip}
    options = {'patches_replace': {'dit': replace}} if replace else {}
    return reference_forward(reference, x, t, conds[0], conds[1], transformer_options=options).float()


def reference_loop(reference, noise, conds, unconds, steps, shift, cfg, uncond_skip, slg_layers, slg_scale,
                   slg_start, slg_stop):
    """MOSSImagePipeline.__call__'s loop, written out against ComfyUI's model."""
    scheduler_sigmas = mi.inference_sigmas(steps, shift).tolist() + [0.0]
    num_steps = len(scheduler_sigmas) - 1
    x = noise.clone()
    for i in range(num_steps):
        t = torch.tensor([scheduler_sigmas[i]], dtype=torch.float32)
        active = slg_start <= i / num_steps < slg_stop
        prediction = reference_velocity(reference, x, t, conds)
        conditional = prediction
        if cfg != 1:
            uncond = reference_velocity(reference, x, t, unconds, uncond_skip if active else ())
            prediction = uncond + cfg * (prediction - uncond)
        if slg_layers and slg_scale > 0 and active:
            skipped = reference_velocity(reference, x, t, conds, slg_layers)
            if cfg == 1:
                prediction = skipped
            prediction = prediction + slg_scale * (conditional - skipped)
        x = x + (scheduler_sigmas[i + 1] - scheduler_sigmas[i]) * prediction
    return x


SAMPLE_CASES = [
    dict(sample_cfg=3.0, sample_uncond_skip_layers=[1]),
    dict(sample_cfg=3.0, sample_uncond_skip_layers=[], sample_negative_prompt='x'),
    dict(sample_cfg=2.0, sample_uncond_skip_layers=[0, 2], sample_slg_layers=[1], sample_slg_scale=1.5,
         sample_slg_start=0.2, sample_slg_stop=0.7),
    dict(sample_cfg=1.0, sample_uncond_skip_layers=[], sample_slg_layers=[2], sample_slg_scale=2.0),
    dict(sample_cfg=1.0, sample_uncond_skip_layers=[]),
]


def sampling_pipeline(reference, settings, steps=6, shift=5.0):
    pipe = loaded_pipeline(reference)
    pipe.model_config = dict(settings, sample_steps=steps, sample_shift=shift)
    pipe.sample_cfg = float(pipe.sample_setting('sample_cfg'))
    _, pipe.conds = text_batch([5], seed=3)
    _, pipe.unconds = text_batch([3], seed=4)
    return pipe


@pytest.mark.parametrize('settings', SAMPLE_CASES)
def test_sample_matches_the_reference_loop(settings):
    reference = tiny_reference()
    pipe = sampling_pipeline(reference, settings)
    noise = torch.randn(1, 128, 3, 4, generator=torch.Generator().manual_seed(7))
    got = pipe.sample(w=64, h=48, noise=noise, decode=False)
    s = {**mi.SAMPLE_DEFAULTS, **settings}
    with torch.no_grad():
        expected = reference_loop(reference, noise, pipe.conds, pipe.unconds, 6, 5.0, s['sample_cfg'],
                                  s['sample_uncond_skip_layers'], s['sample_slg_layers'], s['sample_slg_scale'],
                                  s['sample_slg_start'], s['sample_slg_stop'])
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    assert pipe.layer_state.skip_layers == frozenset()


@pytest.mark.parametrize('settings', SAMPLE_CASES[:4])
def test_sample_matches_moss_image_pipeline(settings):
    """Against the Hugging Face repo's MOSSImagePipeline and scheduler (local reference files)."""
    module = reference_pipeline_module()
    reference = tiny_reference()
    hf = module.MOSSImageTransformer(**TINY).eval()
    hf.load_state_dict(hf_state(reference), strict=True)
    pipeline = module.MOSSImagePipeline(hf, module.MOSSImageScheduler(shift=5.0))
    pipeline.set_progress_bar_config(disable=True)
    pipe = sampling_pipeline(reference, settings)
    noise = torch.randn(1, 128, 3, 4, generator=torch.Generator().manual_seed(7))
    s = {**mi.SAMPLE_DEFAULTS, **settings}
    with torch.no_grad():
        expected = pipeline(
            prompt_embeds=pipe.conds[0], prompt_attention_mask=pipe.conds[1],
            negative_prompt_embeds=pipe.unconds[0], negative_prompt_attention_mask=pipe.unconds[1],
            height=48, width=64, num_inference_steps=6, guidance_scale=s['sample_cfg'], latents=noise,
            uncond_skip_layers=s['sample_uncond_skip_layers'], slg_layers=s['sample_slg_layers'],
            slg_scale=s['sample_slg_scale'], slg_start=s['sample_slg_start'], slg_stop=s['sample_slg_stop'],
            output_type='latent').images
    got = pipe.sample(w=64, h=48, noise=noise, decode=False)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_sample_settings_are_validated():
    pipe = sampling_pipeline(tiny_reference(), dict(sample_uncond_skip_layers=[3]))
    with pytest.raises(ValueError, match='sample_uncond_skip_layers'):
        pipe.sample(w=32, h=32, decode=False)
    pipe = sampling_pipeline(tiny_reference(), dict(sample_uncond_skip_layers=[True]))
    with pytest.raises(ValueError, match='sample_uncond_skip_layers'):
        pipe.sample(w=32, h=32, decode=False)
    pipe = sampling_pipeline(tiny_reference(), dict(sample_uncond_skip_layers=[1], sample_slg_start=0.8,
                                                    sample_slg_stop=0.5))
    with pytest.raises(ValueError, match='slg_start'):
        pipe.sample(w=32, h=32, decode=False)


# ---- adapters, merging and saving --------------------------------------------------------------

LORA = dict(type='lora', rank=4, alpha=4, dropout=0.0, dtype=torch.float32)


def saved_adapter_state(pipe):
    """What utils/saver.py collects for save_adapter."""
    return {p.original_name.replace('.default', '').replace('.modules_to_save', ''): p.detach().clone()
            for p in pipe.diffusion_model.parameters() if p.requires_grad}


def trained_lora_pipeline(reference, adapter=LORA, seed=11):
    pipe = loaded_pipeline(reference)
    for p in pipe.diffusion_model.parameters():
        p.requires_grad_(False)
    pipe.configure_adapter(dict(adapter))
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in pipe.diffusion_model.named_parameters():
            if p.requires_grad:
                p.copy_(0.2 * torch.randn_like(p))
    pipe.pipeline_model = Runner(pipe.to_layers())
    return pipe


def forward_inputs(lengths=(5, 8), seed=2, dim=TEXT_DIM):
    _, (text, mask) = text_batch(list(lengths), dim=dim, seed=seed)
    torch.manual_seed(seed)
    x = torch.randn(len(lengths), 128, 3, 5)
    t = torch.tensor([0.25, 0.9][:len(lengths)])
    return x, t, text, mask


def test_lora_targets_every_linear_of_every_block():
    pipe = trained_lora_pipeline(tiny_reference())
    targets = set(pipe.peft_config.target_modules)
    trained = {name.split('.lora_')[0] for name in saved_adapter_state(pipe)}
    per_block = {name.split('.', 2)[2] for name in trained}
    assert len(trained) == 3 * len(per_block) and all(name.startswith('layers.') for name in trained)
    assert {'proj.q', 'proj.k', 'proj.v', 'proj.o'} <= per_block
    assert targets == trained


# ComfyUI detects MOSS-Image by its keys and assumes 128-wide heads and a 4096-wide text encoder,
# so files meant for comfy.sd's own loader need a model of that shape.
COMFY_TINY = dict(dim=256, ffn_dim=128, num_heads=2, num_layers=2, text_dim=4096, latent_channels=128)


def comfy_loaded(state):
    """The ModelPatcher ComfyUI's Load Diffusion Model makes of a state dict."""
    import comfy.sd
    patcher = comfy.sd.load_diffusion_model_state_dict(dict(state), model_options={'dtype': torch.float32})
    assert type(patcher.model).__name__ == 'MOSSImage'
    return patcher


@pytest.mark.parametrize('layout', ['split', 'fused'])
@pytest.mark.parametrize('adapter', [LORA, dict(type='lokr', rank=4, alpha=4, decompose_factor=-1, rank_dropout=0.0,
                                                 dtype=torch.float32)], ids=['lora', 'lokr'])
def test_saved_adapter_loads_in_comfyui_and_round_trips(adapter, layout, tmp_path):
    import comfy.sd
    reference = tiny_reference(**COMFY_TINY)
    pipe = trained_lora_pipeline(reference, adapter)
    inputs = forward_inputs(dim=COMFY_TINY['text_dim'])
    with torch.no_grad():
        trained = pipe.pipeline_model(inputs)
        untrained = Runner(loaded_pipeline(reference).to_layers())(inputs)
    assert (trained - untrained).abs().max() > 1e-2

    state = saved_adapter_state(pipe)
    pipe.save_adapter(tmp_path, state)
    saved = safetensors.torch.load_file(tmp_path / 'adapter_model.safetensors')
    assert saved and all(key.startswith('diffusion_model.layers.') for key in saved)

    # ComfyUI: its own loader on the base weights, in either layout (the Comfy-Org repack fuses
    # q, k and v), with the saved file applied as a LoRA.
    base = tiny_state(reference) if layout == 'split' else fused_qkv(tiny_state(reference))
    patcher = comfy_loaded(base)
    assert patcher.model.diffusion_model.layers[0].fused_qkv == (layout == 'fused')
    patched, _ = comfy.sd.load_lora_for_models(patcher, None, saved, 1.0, 0.0)
    modules = {k.split('.lora_')[0].split('.lokr_')[0] for k in saved}
    # One patch per adapted module: q, k and v land on slices of the fused weight.
    assert sum(len(v) for v in patched.patches.values()) == len(modules)
    patched.patch_model()
    with torch.no_grad():
        comfy_out = reference_forward(patched.model.diffusion_model, *inputs)
    torch.testing.assert_close(comfy_out, trained, rtol=1e-4, atol=1e-4)

    # diffusion-pipe: the same file loaded back into a fresh adapter (resume / init_from_existing).
    fresh = trained_lora_pipeline(reference, adapter, seed=99)
    fresh.load_adapter_weights(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(fresh.pipeline_model(inputs), trained, rtol=0, atol=0)


def write_transformer_dir(state, path):
    path.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state, path / 'diffusion_pytorch_model.safetensors')
    config = {'_class_name': 'MOSSImageTransformer', **mi.transformer_unet_config(state)}
    (path / 'config.json').write_text(json.dumps(config))
    return path


def pipeline_from_transformer(path, model_config=None, adapter=None):
    pipe = bare_pipeline(dict(model_config or {}), {'adapter': adapter} if adapter else {})
    pipe.transformer_path = str(path)
    pipe.load_diffusion_model()
    pipe.pipeline_model = Runner(pipe.to_layers())
    return pipe


def test_merge_adapters_bakes_the_saved_adapter_in(tmp_path):
    reference = tiny_reference()
    pipe = trained_lora_pipeline(reference)
    inputs = forward_inputs()
    with torch.no_grad():
        trained = pipe.pipeline_model(inputs)
    (tmp_path / 'adapter').mkdir()
    pipe.save_adapter(tmp_path / 'adapter', saved_adapter_state(pipe))
    transformer = write_transformer_dir(tiny_state(reference), tmp_path / 'transformer')
    merged = pipeline_from_transformer(transformer, {'merge_adapters': [str(tmp_path / 'adapter')]})
    with torch.no_grad():
        torch.testing.assert_close(merged.pipeline_model(inputs), trained, rtol=1e-4, atol=1e-4)
    # The adapter is in the weights, not applied per call (a full fine-tune can continue from them).
    assert all(type(p) is torch.nn.Parameter and p.requires_grad for p in merged.diffusion_model.parameters())
    assert not any(getattr(m, 'weight_function', None) for m in merged.diffusion_model.modules())


@pytest.mark.parametrize('prefix', ['', 'model.diffusion_model.', 'diffusion_model.'])
def test_save_model_round_trips_and_single_files_load(prefix, tmp_path):
    reference = tiny_reference()
    state = tiny_state(reference)
    single = tmp_path / 'moss.safetensors'
    safetensors.torch.save_file({prefix + k: v for k, v in state.items()}, single)
    pipe = pipeline_from_transformer(single)
    inputs = forward_inputs()
    with torch.no_grad():
        expected = reference_forward(reference, *inputs)
        torch.testing.assert_close(pipe.pipeline_model(inputs), expected, rtol=1e-5, atol=1e-5)
    out = tmp_path / 'saved'
    out.mkdir()
    pipe.save_model(out, {k: v.detach() for k, v in pipe.diffusion_model.state_dict().items()})
    config = json.loads((out / 'config.json').read_text())
    assert config['_class_name'] == 'MOSSImageTransformer' and config['num_layers'] == 3
    assert config['freq_dim'] == 256 and config['rope_base'] == 10000.0
    again = pipeline_from_transformer(out)
    with torch.no_grad():
        torch.testing.assert_close(again.pipeline_model(inputs), expected, rtol=1e-5, atol=1e-5)


def test_a_saved_full_fine_tune_loads_in_comfyui_and_diffusers(tmp_path):
    reference = tiny_reference(**COMFY_TINY)
    pipe = loaded_pipeline(reference)
    with torch.no_grad():
        for p in pipe.diffusion_model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    inputs = forward_inputs(dim=COMFY_TINY['text_dim'])
    with torch.no_grad():
        expected = pipe.pipeline_model(inputs)
    pipe.save_model(tmp_path, {k: v.detach() for k, v in pipe.diffusion_model.state_dict().items()})
    saved = safetensors.torch.load_file(tmp_path / 'diffusion_pytorch_model.safetensors')
    torch.testing.assert_close(saved['rope.bands'], mi.default_rope_bands(128))
    with torch.no_grad():
        comfy_out = reference_forward(comfy_loaded(saved).model.diffusion_model, *inputs)
    torch.testing.assert_close(comfy_out, expected, rtol=1e-5, atol=1e-5)

    module = reference_pipeline_module()
    hf = module.MOSSImageTransformer.from_pretrained(str(tmp_path)).eval()
    torch.testing.assert_close(hf.rope.bands, saved['rope.bands'])
    x, t, text, mask = inputs
    with torch.no_grad():
        torch.testing.assert_close(hf(x, t * 1000, text, encoder_attention_mask=mask).sample, expected,
                                   rtol=1e-4, atol=1e-4)


# ---- gradients ---------------------------------------------------------------------------------

def assert_grad_close(got, expected, name):
    # Summation order differs from the reference's per-sample loop: compare to the gradient's scale.
    scale = float(expected.abs().max())
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5 * scale, msg=lambda m: f'{name}: {m}')


def reference_gradients(reference, inputs, weight):
    x, t, text, mask = inputs
    reference.zero_grad()
    out = reference_forward(reference, x, t, text, mask)
    (out * weight).sum().backward()
    return {n: p.grad.clone() for n, p in reference.named_parameters()}


def test_full_fine_tune_gradients_match_the_reference():
    reference = tiny_reference()
    pipe = loaded_pipeline(reference)
    for p in pipe.diffusion_model.parameters():
        p.requires_grad_(True)
    inputs = forward_inputs()
    weight = torch.randn(2, 128, 3, 5, generator=torch.Generator().manual_seed(3))
    expected = reference_gradients(reference, inputs, weight)
    out = pipe.pipeline_model(tuple(t.clone() for t in inputs))
    (out * weight).sum().backward()
    got = {n: p.grad for n, p in pipe.diffusion_model.named_parameters()}
    assert got.keys() == expected.keys()
    for name in expected:
        assert got[name] is not None, name
        assert_grad_close(got[name], expected[name], name)
    assert all(float(g.abs().max()) > 0 for g in expected.values())


def test_lora_gradients_reach_only_the_adapter():
    pipe = trained_lora_pipeline(tiny_reference())
    out = pipe.pipeline_model(forward_inputs())
    out.square().mean().backward()
    for name, p in pipe.diffusion_model.named_parameters():
        if 'lora_' in name:
            assert p.grad is not None and float(p.grad.abs().max()) > 0, name
        else:
            assert not p.requires_grad and p.grad is None, name


def split_run(layers, inputs, cut, weight):
    """Two pipeline stages as DeepSpeed runs them: the activations crossing the cut are detached,
    float ones given requires_grad, and every float input of stage 2 must receive a gradient."""
    first = Runner(layers[:cut])(inputs)
    received = tuple(t.detach().requires_grad_(torch.is_floating_point(t)) for t in first)
    out = Runner(layers[cut:])(received)
    (out * weight).sum().backward()
    for i, t in enumerate(received):
        if torch.is_floating_point(t):
            assert t.grad is not None, f'stage-2 input {i} got no gradient'
    torch.autograd.backward([t for t in first if t.requires_grad],
                            [r.grad for r, t in zip(received, first) if t.requires_grad])
    return out


@pytest.mark.parametrize('cut', [1, 2, 4])
def test_pipeline_stages_pass_gradients_across_the_cut(cut):
    reference = tiny_reference()
    pipe = loaded_pipeline(reference)
    for p in pipe.diffusion_model.parameters():
        p.requires_grad_(True)
    inputs = forward_inputs()
    weight = torch.randn(2, 128, 3, 5, generator=torch.Generator().manual_seed(3))
    expected = reference_gradients(reference, inputs, weight)
    split_run(pipe.to_layers(), tuple(t.clone() for t in inputs), cut, weight)
    for name, p in pipe.diffusion_model.named_parameters():
        assert_grad_close(p.grad, expected[name], name)


def test_activation_checkpointing_recomputes_the_same_gradients():
    from torch.utils.checkpoint import checkpoint
    reference = tiny_reference()
    pipe = loaded_pipeline(reference)
    for p in pipe.diffusion_model.parameters():
        p.requires_grad_(True)
    inputs = forward_inputs()
    weight = torch.randn(2, 128, 3, 5, generator=torch.Generator().manual_seed(3))
    expected = reference_gradients(reference, inputs, weight)
    x = tuple(t.clone() for t in inputs)
    for layer in pipe.to_layers():
        if type(layer).__name__ in pipe.checkpointable_layers:
            x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
    (x * weight).sum().backward()
    for name, p in pipe.diffusion_model.named_parameters():
        assert_grad_close(p.grad, expected[name], name)


# ---- low precision -----------------------------------------------------------------------------

def fp8_rounded_reference(reference, keep):
    """The reference with every weight the pipeline stores in fp8 rounded through fp8."""
    rounded = tiny_reference()
    rounded.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for name, p in rounded.named_parameters():
            if p.ndim > 1 and not any(k in name for k in keep):
                p.copy_(p.to(torch.float8_e4m3fn).float())
    return rounded


def test_fp8_diffusion_model_dtype_matches_fp8_rounded_weights(tmp_path):
    reference = tiny_reference()
    transformer = write_transformer_dir(tiny_state(reference), tmp_path / 'transformer')
    pipe = pipeline_from_transformer(transformer, {'diffusion_model_dtype': torch.float8_e4m3fn}, adapter=LORA)
    stored = {n: p.dtype for n, p in pipe.diffusion_model.named_parameters()}
    assert stored['layers.0.proj.q.weight'] == torch.float8_e4m3fn
    assert stored['layers.0.norm1.weight'] == torch.float32
    for module in mi.KEEP_IN_HIGH_PRECISION:
        assert all(dtype == torch.float32 for n, dtype in stored.items() if n.startswith(module + '.')), module
    expected_model = fp8_rounded_reference(reference, mi.KEEP_IN_HIGH_PRECISION)
    inputs = forward_inputs()
    with torch.no_grad():
        expected = reference_forward(expected_model, *inputs)
        torch.testing.assert_close(pipe.pipeline_model(inputs), expected, rtol=1e-5, atol=1e-5)
    # A LoRA trains on top of the fp8 weights.
    pipe.configure_adapter(dict(LORA))
    pipe.pipeline_model = Runner(pipe.to_layers())
    pipe.pipeline_model(forward_inputs()).square().mean().backward()
    lora = [p for n, p in pipe.diffusion_model.named_parameters() if 'lora_A' in n]
    assert lora and all(p.grad is not None for p in lora)


def fused_qkv(state):
    """The Comfy-Org/MOSS-Image layout (also ai-toolkit's saves): each block's q, k and v fused into proj.qkv."""
    state = dict(state)
    for key in [k for k in state if re.fullmatch(r'layers\.\d+\.proj\.q\.weight', k)]:
        base = key[:-len('.q.weight')]
        state[base + '.qkv.weight'] = torch.cat([state.pop(f'{base}.{p}.weight') for p in 'qkv'], 0)
    return state


@pytest.mark.parametrize('prefix', ['', 'model.diffusion_model.'])
def test_fused_qkv_checkpoints_load(prefix, tmp_path):
    reference = tiny_reference()
    path = tmp_path / 'moss_image_base_bf16.safetensors'
    safetensors.torch.save_file({prefix + k: v for k, v in fused_qkv(tiny_state(reference)).items()}, path)
    pipe = pipeline_from_transformer(path)
    assert 'layers.0.proj.q.weight' in pipe.diffusion_model.state_dict()
    inputs = forward_inputs()
    with torch.no_grad():
        expected = reference_forward(reference, *inputs)
        torch.testing.assert_close(pipe.pipeline_model(inputs), expected, rtol=1e-5, atol=1e-5)


def int8_rows(w):
    scale = w.abs().amax(1, keepdim=True) / 127.0
    return torch.round(w / scale).clamp(-127, 127).to(torch.int8), scale.float()


def quant_metadata(layers):
    return {'_quantization_metadata': json.dumps({'format_version': '1.0', 'layers': layers})}


@pytest.mark.parametrize('convrot', [False, True])
def test_fused_qkv_int8_checkpoints_load_like_split_ones(convrot, tmp_path):
    # Comfy-Org/MOSS-Image's int8_convrot files: int8 weights with one scale per row.
    pytest.importorskip('comfy_kitchen')
    reference = tiny_reference()
    conf = {'format': 'int8_tensorwise', **({'convrot': True, 'convrot_groupsize': 16} if convrot else {})}
    split, fused = tiny_state(reference), fused_qkv(tiny_state(reference))
    split_layers, fused_layers = {}, {}
    for i in range(TINY['num_layers']):
        base = f'layers.{i}.proj'
        weight, scale = int8_rows(fused[base + '.qkv.weight'])
        fused[base + '.qkv.weight'], fused[base + '.qkv.weight_scale'] = weight, scale
        fused_layers[base + '.qkv'] = conf
        # Comfy-Org's int8 files also pick a quantized attention kernel, which has no backward.
        fused[f'layers.{i}.comfy_attention.config'] = torch.tensor(
            list(json.dumps({'attention': 'comfy_kitchen_int8'}).encode()), dtype=torch.uint8)
        for name, w, s in zip('qkv', weight.chunk(3), scale.chunk(3)):
            split[f'{base}.{name}.weight'], split[f'{base}.{name}.weight_scale'] = w.clone(), s.clone()
            split_layers[f'{base}.{name}'] = conf
    outputs = []
    for name, state, layers in (('split', split, split_layers), ('fused', fused, fused_layers)):
        path = tmp_path / f'{name}.safetensors'
        safetensors.torch.save_file(state, path, metadata=quant_metadata(layers))
        pipe = pipeline_from_transformer(path, adapter=LORA)
        assert type(pipe.diffusion_model.get_submodule('layers.1.proj.k').weight).__name__ == 'QuantizedTensor'
        assert all(block.comfy_attention.config is None for block in pipe.diffusion_model.layers)
        with torch.no_grad():
            outputs.append(pipe.pipeline_model(forward_inputs()))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    if not convrot:  # the weights were not rotated, so only the plain format matches the reference
        inputs = forward_inputs()
        with torch.no_grad():
            expected = reference_forward(reference, *inputs)
        torch.testing.assert_close(outputs[1], expected, rtol=5e-2, atol=5e-2)


def test_fused_qkv_in_a_block_scaled_format_is_refused(tmp_path):
    state = fused_qkv(tiny_state(tiny_reference()))
    state['layers.0.proj.qkv.comfy_quant'] = torch.tensor(list(json.dumps({'format': 'nvfp4'}).encode()),
                                                          dtype=torch.uint8)
    with pytest.raises(ValueError, match="'nvfp4', which cannot be split"):
        mi.transformer_state(state, {})


FUSED_LORA_KEYS = {
    'comfy': lambda module: 'diffusion_model.' + module,
    'diffusers': lambda module: 'transformer.' + module,
    'kohya': lambda module: 'lora_unet_' + module.replace('.', '_'),
}


def fused_lora(reference, names=('lora_A.weight', 'lora_B.weight'), alpha=None, style='comfy', rank=4, seed=3):
    """A LoRA on every block's fused proj.qkv, and the split weights it should give.

    The defaults are what ai-toolkit saves for MOSS-Image: PEFT names, the comfy prefix, no alpha.
    """
    torch.manual_seed(seed)
    dim, name_of = TINY['dim'], FUSED_LORA_KEYS[style]
    scale = 1.0 if alpha is None else alpha / rank
    adapter, expected = {}, tiny_state(reference)
    for i in range(TINY['num_layers']):
        a, b = 0.1 * torch.randn(rank, dim), 0.1 * torch.randn(3 * dim, rank)
        key = name_of(f'layers.{i}.proj.qkv')
        adapter.update({f'{key}.{names[0]}': a, f'{key}.{names[1]}': b})
        if alpha is not None:
            adapter[f'{key}.alpha'] = torch.tensor(alpha)
        for name, part in zip('qkv', (scale * (b @ a)).chunk(3)):
            expected[f'layers.{i}.proj.{name}.weight'] = expected[f'layers.{i}.proj.{name}.weight'] + part
    # A module both layouts share is adapted as it is.
    a, b = 0.1 * torch.randn(rank, dim), 0.1 * torch.randn(3 * dim, rank)
    key = name_of('text_embedder.blocks.0.qkv')
    adapter.update({f'{key}.{names[0]}': a, f'{key}.{names[1]}': b})
    expected['text_embedder.blocks.0.qkv.weight'] = expected['text_embedder.blocks.0.qkv.weight'] + b @ a
    return adapter, expected


def merged_pipeline(reference, adapter, tmp_path, transformer=None, train=None):
    safetensors.torch.save_file(adapter, tmp_path / 'adapter.safetensors')
    if transformer is None:
        transformer = write_transformer_dir(tiny_state(reference), tmp_path / 'transformer')
    return pipeline_from_transformer(transformer, {'merge_adapters': [str(tmp_path / 'adapter.safetensors')]},
                                     adapter=train)


def assert_adapted(pipe, expected):
    got = pipe.diffusion_model.state_dict()
    for key in [k for k in expected if '.proj.' in k or k.startswith('text_embedder.blocks.0.qkv')]:
        torch.testing.assert_close(got[key].float(), expected[key].float(), rtol=1e-5, atol=1e-6, msg=key)


@pytest.mark.parametrize('names,alpha,style', [
    (('lora_A.weight', 'lora_B.weight'), None, 'comfy'),  # what ai-toolkit saves
    (('lora_A.weight', 'lora_B.weight'), 2.0, 'comfy'),
    (('lora_down.weight', 'lora_up.weight'), 2.0, 'comfy'),
    (('lora_A.weight', 'lora_B.weight'), None, 'diffusers'),
    (('lora_down.weight', 'lora_up.weight'), 8.0, 'kohya'),
])
def test_merge_adapters_splits_fused_qkv_loras(names, alpha, style, tmp_path):
    # ai-toolkit trains MOSS-Image LoRAs on the fused proj.qkv.
    reference = tiny_reference()
    adapter, expected = fused_lora(reference, names, alpha, style)
    assert_adapted(merged_pipeline(reference, adapter, tmp_path), expected)


def test_merge_adapters_splits_a_fused_qkv_diff(tmp_path):
    reference = tiny_reference()
    torch.manual_seed(4)
    diff = 0.01 * torch.randn(3 * TINY['dim'], TINY['dim'])
    expected = tiny_state(reference)
    for name, part in zip('qkv', diff.chunk(3)):
        expected[f'layers.1.proj.{name}.weight'] = expected[f'layers.1.proj.{name}.weight'] + part
    pipe = merged_pipeline(reference, {'diffusion_model.layers.1.proj.qkv.diff': diff}, tmp_path)
    assert_adapted(pipe, {k: v for k, v in expected.items() if k.startswith('layers.1.')})


def test_merge_adapters_onto_a_fused_fp8_base_matches_a_split_one(tmp_path):
    pytest.importorskip('comfy_kitchen')
    reference = tiny_reference()
    split, fused = tiny_state(reference), fused_qkv(tiny_state(reference))
    split_layers, fused_layers = {}, {}
    for i in range(TINY['num_layers']):
        base = f'layers.{i}.proj'
        scale = fused[base + '.qkv.weight'].abs().max() / 448.0
        fused[base + '.qkv.weight'] = (fused[base + '.qkv.weight'] / scale).to(torch.float8_e4m3fn)
        fused[base + '.qkv.weight_scale'] = scale.float()
        fused_layers[base + '.qkv'] = {'format': 'float8_e4m3fn'}
        for name, w in zip('qkv', fused[base + '.qkv.weight'].chunk(3)):
            split[f'{base}.{name}.weight'], split[f'{base}.{name}.weight_scale'] = w.clone(), scale.float().clone()
            split_layers[f'{base}.{name}'] = {'format': 'float8_e4m3fn'}
    adapter, _ = fused_lora(reference)
    outputs = []
    for name, state, layers in (('split', split, split_layers), ('fused', fused, fused_layers)):
        (tmp_path / name).mkdir()
        path = tmp_path / name / 'moss.safetensors'
        safetensors.torch.save_file(state, path, metadata=quant_metadata(layers))
        pipe = merged_pipeline(reference, adapter, tmp_path / name, transformer=path, train=LORA)
        with torch.no_grad():
            outputs.append(pipe.pipeline_model(forward_inputs()))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


def test_a_checkpoint_with_both_fused_and_split_qkv_is_refused():
    state = fused_qkv(tiny_state(tiny_reference()))
    state['layers.2.proj.k.weight'] = torch.zeros(TINY['dim'], TINY['dim'])
    with pytest.raises(ValueError, match=r'both layers\.2\.proj\.qkv\.weight and layers\.2\.proj\.k\.weight'):
        mi.transformer_state(state, {})


@pytest.mark.parametrize('order', ['fused first', 'split first'])
def test_an_adapter_with_both_fused_and_split_qkv_is_refused(order, tmp_path):
    fused = {'diffusion_model.layers.0.proj.qkv.lora_A.weight': torch.zeros(4, 64),
             'diffusion_model.layers.0.proj.qkv.lora_B.weight': torch.zeros(192, 4)}
    split = {'diffusion_model.layers.0.proj.v.lora_A.weight': torch.zeros(4, 64)}
    adapter = {**fused, **split} if order == 'fused first' else {**split, **fused}
    with pytest.raises(ValueError, match='both adapt diffusion_model.layers.0.proj.v.lora_A.weight'):
        mi.split_fused_qkv_adapter(adapter, 'adapter.safetensors')


def test_unmatched_adapter_keys_are_refused_whatever_logging_does(tmp_path):
    import logging

    class DropAll(logging.Filter):
        def filter(self, record):
            return False

    adapter = {'transformer.layers.0.proj.nothing.lora_A.weight': torch.zeros(4, 64),
               'transformer.layers.0.proj.nothing.lora_B.weight': torch.zeros(64, 4)}
    root, drop = logging.getLogger(), DropAll()
    level = root.level
    root.setLevel(logging.CRITICAL)
    root.addFilter(drop)
    logging.disable(logging.CRITICAL)
    try:
        # The file's own key names, not the renamed ones.
        with pytest.raises(ValueError, match=r"transformer\.layers\.0\.proj\.nothing\.lora_A\.weight"):
            merged_pipeline(tiny_reference(), adapter, tmp_path)
    finally:
        logging.disable(logging.NOTSET)
        root.removeFilter(drop)
        root.setLevel(level)


@pytest.mark.parametrize('adapter,message', [
    ({'diffusion_model.layers.0.proj.nothing.lora_A.weight': torch.zeros(4, 64),
      'diffusion_model.layers.0.proj.nothing.lora_B.weight': torch.zeros(64, 4)}, 'match no MOSS-Image weight'),
    ({'diffusion_model.layers.0.proj.qkv.lokr_w1': torch.zeros(3, 3),
      'diffusion_model.layers.0.proj.qkv.lokr_w2': torch.zeros(64, 64)}, 'cannot be split'),
])
def test_merge_adapters_refuses_what_it_cannot_apply(adapter, message, tmp_path):
    safetensors.torch.save_file(adapter, tmp_path / 'adapter.safetensors')
    transformer = write_transformer_dir(tiny_state(tiny_reference()), tmp_path / 'transformer')
    with pytest.raises(ValueError, match=message):
        pipeline_from_transformer(transformer, {'merge_adapters': [str(tmp_path / 'adapter.safetensors')]})


def quantized_state(reference, layers, prefix='', style='metadata'):
    """A ComfyUI fp8-scaled checkpoint of the reference.

    style 'metadata': comfy quant format v1, the layers named without the prefix;
    'metadata-prefixed': the same with the prefix in the layer names; 'scaled_fp8': the older
    format, a scaled_fp8 marker and .scale_weight tensors.
    """
    state = tiny_state(reference)
    quant = {}
    for layer in layers:
        w = state[layer + '.weight']
        scale = w.abs().max() / 448.0
        state[layer + '.weight'] = (w / scale).to(torch.float8_e4m3fn)
        state[layer + ('.scale_weight' if style == 'scaled_fp8' else '.weight_scale')] = scale.float()
        quant[(prefix if style == 'metadata-prefixed' else '') + layer] = {'format': 'float8_e4m3fn'}
    if style == 'scaled_fp8':
        state['scaled_fp8'] = torch.empty(0, dtype=torch.float8_e4m3fn)
    state = {prefix + k: v for k, v in state.items()}
    if style == 'scaled_fp8':
        return state, {}
    return state, {'_quantization_metadata': json.dumps({'format_version': '1.0', 'layers': quant})}


def dequantized_reference(reference, state, layers):
    expected = tiny_reference()
    expected.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for layer in layers:
            module = expected.get_submodule(layer)
            scale = state.get(layer + '.weight_scale', state.get(layer + '.scale_weight'))
            module.weight.copy_(state[layer + '.weight'].float() * scale)
    return expected


QUANTIZED = ['layers.0.proj.q', 'layers.1.proj.ffn_up_gate', 'layers.2.proj.o', 'text_embedder.blocks.0.qkv']


@pytest.mark.parametrize('prefix,style', [
    ('', 'metadata'), ('', 'scaled_fp8'), ('model.diffusion_model.', 'metadata'),
    ('model.diffusion_model.', 'metadata-prefixed'), ('model.diffusion_model.', 'scaled_fp8'),
])
def test_comfy_quantized_checkpoints_load_and_train_a_lora(prefix, style, tmp_path):
    pytest.importorskip('comfy_kitchen')
    reference = tiny_reference()
    state, metadata = quantized_state(reference, QUANTIZED, prefix, style)
    path = tmp_path / 'moss_fp8.safetensors'
    safetensors.torch.save_file(state, path, metadata=metadata)
    pipe = pipeline_from_transformer(path, adapter=LORA)
    for layer in QUANTIZED:
        assert type(pipe.diffusion_model.get_submodule(layer).weight).__name__ == 'QuantizedTensor', layer
    unprefixed = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    expected_model = dequantized_reference(reference, unprefixed, QUANTIZED)
    inputs = forward_inputs()
    with torch.no_grad():
        expected = reference_forward(expected_model, *inputs)
        torch.testing.assert_close(pipe.pipeline_model(inputs), expected, rtol=1e-4, atol=1e-4)
    for p in pipe.diffusion_model.parameters():
        p.requires_grad_(False)
    pipe.configure_adapter(dict(LORA))
    targets = set(pipe.peft_config.target_modules)
    assert {'layers.0.proj.q', 'layers.1.proj.ffn_up_gate', 'layers.2.proj.o'} <= targets
    pipe.pipeline_model = Runner(pipe.to_layers())
    pipe.pipeline_model(forward_inputs()).square().mean().backward()
    grads = {n: p.grad for n, p in pipe.diffusion_model.named_parameters() if 'lora_A' in n}
    assert grads and all(g is not None for g in grads.values())


def test_a_quantized_checkpoint_refuses_a_full_fine_tune(tmp_path):
    pytest.importorskip('comfy_kitchen')
    state, metadata = quantized_state(tiny_reference(), QUANTIZED)
    path = tmp_path / 'moss_fp8.safetensors'
    safetensors.torch.save_file(state, path, metadata=metadata)
    with pytest.raises(ValueError, match='full finetune a quantized model'):
        pipeline_from_transformer(path)


def break_block_1(state):
    for key in [k for k in state if k.startswith('layers.1.')]:
        del state[key]


@pytest.mark.parametrize('damage,message', [
    # ComfyUI's mixed-precision Linear does not report a missing weight, quantized or not.
    (lambda state: state.pop('layers.0.proj.q.weight'), r'missing 1 weights.*layers\.0\.proj\.q\.weight'),
    (lambda state: state.pop('layers.0.proj.k.weight'), r'missing 1 weights.*layers\.0\.proj\.k\.weight'),
    # A middle block gone: the blocks after it would be dropped and the model run with one block.
    (break_block_1, r'Unexpected keys.*layers\.2\.'),
    (lambda state: state.__setitem__('layers.0.proj.qq.weight', torch.zeros(4, 4)), r'Unexpected keys.*proj\.qq'),
])
def test_a_damaged_quantized_checkpoint_is_refused(damage, message, tmp_path):
    pytest.importorskip('comfy_kitchen')
    state, metadata = quantized_state(tiny_reference(), QUANTIZED)
    damage(state)
    path = tmp_path / 'moss_fp8.safetensors'
    safetensors.torch.save_file(state, path, metadata=metadata)
    with pytest.raises(RuntimeError, match=message):
        pipeline_from_transformer(path, adapter=LORA)


def test_shapes_comfyui_cannot_detect_and_short_files_are_warned_about(tmp_path, caplog):
    path = tmp_path / 'moss.safetensors'
    safetensors.torch.save_file(tiny_state(tiny_reference()), path)
    with caplog.at_level(logging.WARNING):
        pipeline_from_transformer(path, adapter=LORA)
    text = caplog.text
    assert 'head size = 32 (ComfyUI expects 128)' in text and f'text_dim = {TEXT_DIM}' in text
    assert 'latent_channels' not in text
    assert 'has 3 blocks' in text
    # A config.json records the block count, so a directory is not warned about.
    caplog.clear()
    directory = tmp_path / 'transformer'
    write_transformer_dir(tiny_state(tiny_reference()), directory)
    with caplog.at_level(logging.WARNING):
        pipeline_from_transformer(directory, adapter=LORA)
    assert 'blocks' not in caplog.text and 'head size = 32' in caplog.text


def test_bfloat16_matches_the_reference_in_bfloat16():
    reference = tiny_reference()
    pipe = loaded_pipeline(reference, dtype=torch.bfloat16)
    reference_bf16 = tiny_reference(dtype=torch.bfloat16)
    reference_bf16.load_state_dict(reference.state_dict())
    inputs = forward_inputs()
    x, t, text, mask = inputs
    with torch.no_grad():
        expected = reference_forward(reference_bf16, x.bfloat16(), t, text.bfloat16(), mask).float()
        got = pipe.pipeline_model((x.bfloat16(), t, text.bfloat16(), mask)).float()
    # Same bf16 arithmetic; only the batching (per sample vs per length group) differs.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    full = reference_forward(reference, x, t, text, mask)
    assert (got - full).abs().mean() < 0.05 * full.abs().mean()


# ---- configuration -----------------------------------------------------------------------------

def diffusers_dir(tmp_path):
    for sub in ('transformer', 'vae', 'text_encoder'):
        (tmp_path / 'MOSS-Image' / sub).mkdir(parents=True)
    return tmp_path / 'MOSS-Image'


def make_pipeline(model, extra=None):
    return mi.MossImagePipeline({'model': {'type': 'moss_image', **model}, **(extra or {})})


def test_diffusers_path_finds_every_component(tmp_path):
    root = diffusers_dir(tmp_path)
    pipe = make_pipeline({'diffusers_path': str(root)})
    assert pipe.transformer_path == str(root / 'transformer')
    assert pipe.vae_path == str(root / 'vae')
    assert pipe.text_encoder_path == str(root / 'text_encoder')
    assert pipe.dtype == torch.bfloat16
    single = tmp_path / 'moss_fp8.safetensors'
    pipe = make_pipeline({'diffusers_path': str(root), 'transformer_path': str(single)})
    assert pipe.transformer_path == str(single) and pipe.vae_path == str(root / 'vae')
    pipe = make_pipeline({'transformer_path': 'a', 'vae_path': 'b', 'text_encoder_path': 'c'})
    assert (pipe.transformer_path, pipe.vae_path, pipe.text_encoder_path) == ('a', 'b', 'c')
    with pytest.raises(ValueError, match='text_encoder_path'):
        make_pipeline({'transformer_path': 'a', 'vae_path': 'b'})
    with pytest.raises(FileNotFoundError, match='vae_path'):
        (root / 'vae').rmdir()
        make_pipeline({'diffusers_path': str(root)})


@pytest.mark.parametrize('model,match', [
    ({'flux_shift': True}, 'flux_shift'),
    ({'diffusion_model': 'x'}, 'diffusion_model'),
    ({'vae': 'x'}, 'vae'),
    ({'text_encoders': []}, 'text_encoders'),
    ({'llm_path': 'x'}, 'llm_path'),
    ({'transformer_dtype': 'float8'}, 'transformer_dtype'),
    ({'text_encoder_fp8': True}, 'text_encoder_fp8'),
    ({'sigmoid_scale': 1.0}, 'sigmoid_scale'),
    ({'cache_text_embeddings': False}, 'cache_text_embeddings'),
    ({'timestep_sample_method': 'uniform', 'logit_mean': 0.0}, 'logit_mean'),
    ({'timestep_sample_method': 'gamma'}, 'timestep_sample_method'),
    ({'dtype': torch.float16}, 'bfloat16 or float32'),
    ({'diffusion_model_dtype': 'float8'}, r'\[adapter\]'),
    ({'on_caption_overflow': 'drop'}, 'on_caption_overflow'),
])
def test_config_keys_that_do_not_apply_are_refused(model, match, tmp_path):
    root = diffusers_dir(tmp_path)
    with pytest.raises(ValueError, match=match):
        make_pipeline({'diffusers_path': str(root), **model})


def test_noise_settings_reach_the_sampler(tmp_path):
    root = diffusers_dir(tmp_path)
    pipe = make_pipeline({'diffusers_path': str(root), 'logit_mean': 0.0, 'logit_std': 1.0, 'sigma_cutoff': 1.0,
                          'uniform_fraction': 0.0, 'min_t': 0.1, 'max_t': 0.9})
    expected = mi.NoiseLevelSampler('logit_normal', 0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.9)
    torch.testing.assert_close(pipe.noise_sampler.probabilities, expected.probabilities)


def test_transformer_config_must_describe_its_weights(tmp_path):
    reference = tiny_reference()
    transformer = write_transformer_dir(tiny_state(reference), tmp_path / 'transformer')
    config = json.loads((transformer / 'config.json').read_text())
    (transformer / 'config.json').write_text(json.dumps({**config, 'num_layers': 42}))
    with pytest.raises(ValueError, match='num_layers = 42'):
        pipeline_from_transformer(transformer)
    (transformer / 'config.json').write_text(json.dumps({**config, '_class_name': 'FluxTransformer2DModel'}))
    with pytest.raises(ValueError, match='FluxTransformer2DModel'):
        pipeline_from_transformer(transformer)
    # ComfyUI's model has a fixed RoPE base: another one is refused, not run wrongly.
    (transformer / 'config.json').write_text(json.dumps({**config, 'rope_base': 500.0}))
    with pytest.raises(ValueError, match=r'rope_base = 500\.0'):
        pipeline_from_transformer(transformer)
    (transformer / 'config.json').write_text(json.dumps({**config, 'rope_base': 10000}))
    pipeline_from_transformer(transformer)


@pytest.mark.parametrize('change,match', [
    (dict(freq_dim=32), 'freq_dim = 32'),
    (dict(rope_base=500.0), 'rope_base = 500'),
    (dict(dim=60, num_heads=5), 'head size must be a multiple of 8'),
])
def test_transformers_comfyui_cannot_run_are_refused(change, match):
    config = mi.transformer_unet_config(tiny_state(tiny_reference()))
    assert mi.comfy_unet_config(config)['fused_qkv'] is False
    with pytest.raises(ValueError, match=match):
        mi.comfy_unet_config({**config, **change})


def test_control_datasets_are_refused():
    with pytest.raises(NotImplementedError, match='control'):
        bare_pipeline().model_specific_dataset_config_validation({'directory': [{'path': 'a', 'control_path': 'b'}]})


# ---- VAE ---------------------------------------------------------------------------------------

LOCAL_VAE = Path('/home/user/moss-weights/Base/vae')


@pytest.fixture(scope='module')
def vaes():
    if not (LOCAL_VAE / 'diffusion_pytorch_model.safetensors').is_file():
        pytest.skip('needs the MOSS-Image VAE weights')
    import comfy.sd
    from diffusers import AutoencoderKLFlux2
    reference = AutoencoderKLFlux2.from_pretrained(LOCAL_VAE, torch_dtype=torch.float32).eval()
    comfy_vae = comfy.sd.VAE(sd=mi.read_component(str(LOCAL_VAE), 'diffusion_pytorch_model')[0], dtype=torch.float32)
    pipe = bare_pipeline()
    pipe.vae = comfy_vae
    pipe.latent_format = comfy.latent_formats.Flux2()
    return reference, pipe


def reference_latent_to_vae(reference, z):
    """MOSSImagePipeline.__call__'s decode path, up to vae.decode."""
    bn = reference.bn
    eps = reference.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def reference_encode(reference, images):
    """The inverse of the decode path: posterior mean, 2x2 patches, batch-norm statistics."""
    z = reference.encode(images * 2 - 1).latent_dist.mode()
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    bn = reference.bn
    eps = reference.config.batch_norm_eps
    return (z - bn.running_mean.view(1, -1, 1, 1)) / (bn.running_var + eps).sqrt().view(1, -1, 1, 1)


def test_vae_decode_matches_the_reference_pipeline(vaes):
    reference, pipe = vaes
    z = torch.randn(1, 128, 4, 6, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = reference.decode(reference_latent_to_vae(reference, z)).sample.clamp(-1, 1)
        got = pipe.vae_decode(z).movedim(-1, 1)
    torch.testing.assert_close(got * 2 - 1, expected, rtol=0, atol=2e-3)


def test_vae_encode_inverts_the_reference_decode(vaes):
    reference, pipe = vaes
    torch.manual_seed(0)
    images = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        expected = reference_encode(reference, images)
        got = pipe.vae_encode(images)
    assert got.shape == (1, 128, 4, 6)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('name', ['moss_image_lora.toml', 'moss_image_full_8gpu.toml'])
def test_example_configs_are_accepted(name):
    import toml
    root = Path(__file__).resolve().parent.parent
    config = toml.load(root / 'examples' / 'moss_image' / name)
    assert config['model']['type'] == 'moss_image'
    assert (root / config['dataset']).is_file()
    toml.load(root / config['dataset'])
    pipe = mi.MossImagePipeline.__new__(mi.MossImagePipeline)
    pipe.config, pipe.model_config = config, config['model']
    pipe._check_keys()
    assert config.get('keep_last_n_checkpoints', 2) >= 2
