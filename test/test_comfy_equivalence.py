"""Each ComfyUI-backed pipeline's layers against ComfyUI's own forward of the model, on CPU.

The pipelines in models/ re-implement the forward of a ComfyUI model as a chain of pipeline
layers (to_layers()). When a ComfyUI update changes the model's internals, a copy that no longer
matches either crashes (Ideogram4's RoPE table became a matrix) or trains a different model
without saying so. These tests run the pipeline's layers and ComfyUI's forward on the same tiny,
randomly initialised model on ComfyUI's training path (models/base.py sets
model_management.in_training), compare the outputs and run a backward through the layers, so a
ComfyUI bump that breaks a pipeline fails here.

MOSS-Image has its own, more thorough tests in test_moss_image.py.
"""

import pytest
import torch

import comfy.model_management
import comfy.ops

from utils.offloading import ModelOffloader


@pytest.fixture(autouse=True)
def training_path():
    saved = comfy.model_management.in_training
    comfy.model_management.in_training = True
    yield
    comfy.model_management.in_training = saved


@pytest.fixture(autouse=True)
def pytorch_attention(monkeypatch):
    """Attention as on the GPUs the pipelines train on: PyTorch's SDPA. Without CUDA, ComfyUI
    picks attention_sub_quad, which adds a mask to the scores (attn_scores += mask), so a bool
    padding mask adds 0 or 1 instead of masking: every padding test would fail for a reason
    training never sees. Modules that imported optimized_attention by name are patched too, and
    those imported later read the patched attribute."""
    import sys
    import comfy.ldm.modules.attention as attention
    sub_quad, sdpa = attention.attention_sub_quad, attention.attention_pytorch
    for module in list(sys.modules.values()):
        for name in ('optimized_attention', 'optimized_attention_masked'):
            if getattr(module, name, None) is sub_quad:
                monkeypatch.setattr(module, name, sdpa)


def randomize(model, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and 'norm' in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.05 * torch.randn_like(p))
    return model


def bare_pipeline(cls, diffusion_model, model_config=None):
    pipe = cls.__new__(cls)
    pipe.config = {'model': model_config or {}, 'reentrant_activation_checkpointing': False}
    pipe.model_config = pipe.config['model']
    pipe.dtype = torch.float32
    pipe.diffusion_model = diffusion_model
    pipe.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cpu'), False, debug=False)
    return pipe


def run_layers(layers, inputs):
    for layer in layers:
        inputs = layer(inputs)
    return inputs


def check_backward(layers, inputs, model, autocast_dtype=None, unused=()):
    """A backward through the layers reaches every block's weights (an in-place op on a tensor
    autograd saved, or a detached input, fails or leaves them without a gradient).

    autocast_dtype runs the forward under CPU autocast, as training runs it under CUDA autocast:
    ComfyUI's flux DoubleStreamBlock adds into its residual stream in place (img += ...) after
    a norm saved it, which autograd only accepts because autocast hands the norm an fp32 copy
    of the bf16 stream."""
    model.zero_grad(set_to_none=True)
    layer_norm = torch.nn.functional.layer_norm

    def cuda_autocast_layer_norm(x, normalized_shape, weight=None, bias=None, *args, **kwargs):
        # CUDA autocast runs layer_norm in fp32, weights included (CPU autocast leaves it in bf16)
        weight = weight.float() if weight is not None else None
        bias = bias.float() if bias is not None else None
        return layer_norm(x.float(), normalized_shape, weight, bias, *args, **kwargs)

    try:
        if autocast_dtype is not None:
            torch.nn.functional.layer_norm = cuda_autocast_layer_norm
        with torch.autocast('cpu', dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None):
            out = run_layers(layers, inputs)
    finally:
        torch.nn.functional.layer_norm = layer_norm
    out.float().square().mean().backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None and name not in unused]
    assert not missing, f'no gradient reached {missing[:5]}'
    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f'non-finite gradients in {bad[:5]}'


def assert_close(actual, expected, atol=1e-5, rtol=1e-4):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def tiny_ideogram4():
    from comfy.ldm.ideogram4.model import Ideogram4Transformer2DModel
    model = Ideogram4Transformer2DModel(
        in_channels=128, num_layers=2, num_attention_heads=2, attention_head_dim=32, intermediate_size=64,
        adaln_dim=16, llm_features_dim=24, rope_theta=5000000, mrope_section=(6, 5, 5),
        dtype=torch.float32, operations=comfy.ops.manual_cast)
    return randomize(model)


def ideogram4_inputs(bs=2, text_len=7, h=4, w=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bs, 128, h, w, generator=g)
    t = torch.rand(bs, generator=g)
    text = torch.randn(bs, text_len, 24, generator=g)
    mask = torch.ones(bs, text_len, dtype=torch.bool)
    mask[1:, text_len - 3:] = False  # right padding on the second sample
    text[~mask] = 0
    return x, t, text, mask


def test_ideogram4_layers_match_comfyui():
    from models import ideogram4
    model = tiny_ideogram4()
    pipe = bare_pipeline(ideogram4.Ideogram4Pipeline, model)
    layers = pipe.to_layers()
    x, t, text, mask = ideogram4_inputs()
    expected = model(x, t, context=text, attention_mask=mask)
    actual = run_layers(layers, (x, t, text, mask))
    assert_close(actual, expected)
    check_backward(layers, (x, t, text, mask), model)


def tiny_z_image():
    from comfy.ldm.lumina.model import NextDiT
    model = NextDiT(
        # dim >= 256: Z-Image's adaLN takes min(dim, 256) features from a 256-wide timestep embedding
        patch_size=2, in_channels=16, dim=256, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
        multiple_of=16, ffn_dim_multiplier=1.0, qk_norm=True, cap_feat_dim=24, axes_dims=(32, 48, 48),
        axes_lens=(1536, 512, 512), rope_theta=256.0, z_image_modulation=True, time_scale=1000.0,
        pad_tokens_multiple=32, dtype=torch.float32, operations=comfy.ops.manual_cast)
    return randomize(model)


def test_z_image_layers_match_comfyui():
    # ComfyUI pads the caption to a multiple of 32 with cap_pad_token and runs no mask; the
    # pipeline masks padding instead (it batches captions of different lengths). The two agree
    # for a caption of 32 tokens.
    from models import z_image
    model = tiny_z_image()
    pipe = bare_pipeline(z_image.ZImagePipeline, model)
    layers = pipe.to_layers()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 16, 8, 12, generator=g)
    t = torch.rand(2, generator=g)
    text = torch.randn(2, 32, 24, generator=g)
    mask = torch.ones(2, 32, dtype=torch.bool)
    expected = model(x, t, text, 32)
    actual = run_layers(layers, (x, t, text, mask))
    # ComfyUI returns noise - x0; the pipeline trains on x0 - noise (target = latents - noise).
    assert_close(actual, -expected)
    # cap_pad_token pads captions in ComfyUI; the pipeline masks the padding instead.
    check_backward(layers, (x, t, text, mask), model, unused=('cap_pad_token',))


def test_z_image_padding_matches_comfyui():
    """Captions of a batch padded to the longest train as ComfyUI runs each alone: padded with
    cap_pad_token to a multiple of 32 (which the blocks attend to), the image positions after
    that, and the batch's extra padding masked out."""
    from models import z_image
    model = tiny_z_image()
    layers = bare_pipeline(z_image.ZImagePipeline, model).to_layers()
    g = torch.Generator().manual_seed(2)
    lengths = [40, 7, 32]
    x = torch.randn(3, 16, 8, 12, generator=g)
    t = torch.rand(3, generator=g)
    text = torch.randn(3, max(lengths), 24, generator=g)
    mask = torch.arange(text.shape[1])[None] < torch.tensor(lengths)[:, None]
    text[~mask] = 0
    actual = run_layers(layers, (x, t, text, mask))
    for i, n in enumerate(lengths):
        expected = model(x[i:i + 1], t[i:i + 1], text[i:i + 1, :n], n)
        assert_close(actual[i:i + 1], -expected)
    check_backward(layers, (x, t, text, mask), model)


def test_z_image_refuses_other_nextdit_models():
    from models import z_image
    model = tiny_z_image()
    model.masked_pad_multiple = 32
    with pytest.raises(NotImplementedError, match='Ming-Image'):
        bare_pipeline(z_image.ZImagePipeline, model).to_layers()
    model.masked_pad_multiple = None
    model.siglip_embedder = torch.nn.Identity()
    with pytest.raises(NotImplementedError, match='Omni'):
        bare_pipeline(z_image.ZImagePipeline, model).to_layers()


def tiny_krea2():
    from comfy.ldm.krea2.model import SingleStreamDiT
    model = SingleStreamDiT(
        features=64, tdim=32, txtdim=16, heads=4, kvheads=2, multiplier=2, layers=2, patch=2, channels=16,
        txtlayers=2, txtheads=2, txtkvheads=2, dtype=torch.float32, operations=comfy.ops.manual_cast)
    return randomize(model)


def test_krea2_layers_match_comfyui():
    # ComfyUI runs the blocks without a mask (one prompt at a time); with no padding the
    # pipeline's mask attends everywhere too.
    from models import krea2
    model = tiny_krea2()
    pipe = bare_pipeline(krea2.Krea2Pipeline, model)
    layers = pipe.to_layers()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 16, 1, 8, 12, generator=g)
    t = torch.rand(2, generator=g)
    text = torch.randn(2, 9, 2 * 16, generator=g)
    mask = torch.ones(2, 9, dtype=torch.bool)
    expected = model(x, t, text)
    actual = run_layers(layers, (x, t, text, mask))
    assert_close(actual, expected)
    check_backward(layers, (x, t, text, mask), model)


def test_krea2_padding_does_not_reach_the_image():
    from models import krea2
    model = tiny_krea2()
    layers = bare_pipeline(krea2.Krea2Pipeline, model).to_layers()
    g = torch.Generator().manual_seed(2)
    lengths = [9, 5]
    x = torch.randn(2, 16, 1, 8, 12, generator=g)
    t = torch.rand(2, generator=g)
    text = torch.randn(2, 9, 2 * 16, generator=g)
    mask = torch.arange(9)[None] < torch.tensor(lengths)[:, None]
    text[~mask] = 0
    actual = run_layers(layers, (x, t, text, mask))
    for i, n in enumerate(lengths):
        assert_close(actual[i:i + 1], model(x[i:i + 1], t[i:i + 1], text[i:i + 1, :n]))


def tiny_flux2(guidance_embed):
    from comfy.ldm.flux.model import Flux
    model = Flux(
        image_model='flux2', in_channels=32, out_channels=32, vec_in_dim=None, context_in_dim=24, hidden_size=64,
        mlp_ratio=3.0, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[8, 8, 8, 8], theta=2000,
        patch_size=1, qkv_bias=False, guidance_embed=guidance_embed, txt_ids_dims=[3], global_modulation=True,
        mlp_silu_act=True, ops_bias=False, default_ref_method='index', ref_index_scale=10.0,
        dtype=torch.float32, operations=comfy.ops.manual_cast)
    return randomize(model)


@pytest.mark.parametrize('guidance_embed', [False, True], ids=['klein', 'dev'])
@pytest.mark.parametrize('control', [False, True], ids=['t2i', 'edit'])
def test_flux2_layers_match_comfyui(guidance_embed, control):
    from models import flux2
    model = tiny_flux2(guidance_embed)
    pipe = bare_pipeline(flux2.Flux2Pipeline, model)
    pipe.offloader_double = pipe.offloader_single = pipe.offloader
    layers = pipe.to_layers()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 32, 6, 5, generator=g)
    t = torch.rand(2, generator=g)
    text = torch.randn(2, 9, 24, generator=g)
    guidance = torch.ones(2)
    inputs = (x, t, text, guidance, torch.ones(2, 9, dtype=torch.bool))
    kwargs = {}
    if control:
        ref = torch.randn(2, 32, 6, 5, generator=g)
        inputs += (ref,)
        kwargs['ref_latents'] = [ref]
    expected = model(x, t, text, guidance=guidance, **kwargs)
    actual = run_layers(layers, inputs)
    assert_close(actual, expected)
    check_backward(layers, inputs, model, autocast_dtype=torch.bfloat16)


@pytest.mark.parametrize('is_32b', [False, True], ids=['klein', '32b'])
def test_flux2_text_matches_comfyui_inference(is_32b):
    """Cached captions of different lengths, batched, train as ComfyUI runs each prompt alone.
    Klein: the Qwen3 tokenizer pads to 512 tokens (the encoder's padding rows are real hidden
    states) and the model attends to them. 32B: the Mistral tokenizer does not pad and ComfyUI's
    model_base.Flux2.extra_conds prepends zeros up to 512 tokens. A caption over 512 tokens is
    kept whole; padding added to batch the captions is masked out."""
    from models import flux2
    model = tiny_flux2(guidance_embed=is_32b)
    pipe = bare_pipeline(flux2.Flux2Pipeline, model)
    pipe.is_32b = is_32b
    pipe.offloader_double = pipe.offloader_single = pipe.offloader
    layers = pipe.to_layers()
    g = torch.Generator().manual_seed(2)
    lengths = [530, 20, 9]
    cached = [530, 530 if is_32b else 512, 9 if is_32b else 530]
    embeds = [torch.randn(c, 24, generator=g) for c in cached]
    masks = [(torch.arange(c) < n).long() for c, n in zip(cached, lengths)]
    for e, m in zip(embeds, masks):
        if is_32b:
            e[m == 0] = 0
    x = torch.randn(3, 32, 4, 5, generator=g)
    t = torch.rand(3, generator=g)
    inputs = (x, t, *pipe.get_conds({'text_embeds_0': embeds, 'attention_mask_0': masks}))
    actual = run_layers(layers, inputs)
    for i, n in enumerate(lengths):
        if is_32b:
            context = torch.nn.functional.pad(embeds[i][:n], (0, 0, max(512 - n, 0), 0))
        else:
            context = embeds[i][:max(512, n)]
        expected = model(x[i:i + 1], t[i:i + 1], context[None], guidance=torch.ones(1))
        assert_close(actual[i:i + 1], expected)


def tiny_ernie_image():
    from comfy.ldm.ernie.model import ErnieImageModel
    model = ErnieImageModel(
        hidden_size=64, num_attention_heads=2, num_layers=2, ffn_hidden_size=96, in_channels=32, out_channels=32,
        patch_size=1, text_in_dim=24, rope_theta=256, rope_axes_dim=(8, 12, 12),
        dtype=torch.float32, operations=comfy.ops.manual_cast)
    return randomize(model)


def test_ernie_image_layers_match_comfyui():
    # ComfyUI runs without a text mask; with no padding the pipeline's mask attends everywhere.
    from models import ernie_image
    model = tiny_ernie_image()
    pipe = bare_pipeline(ernie_image.ErnieImagePipeline, model)
    layers = pipe.to_layers()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 32, 6, 5, generator=g)
    t = torch.rand(2, generator=g) * 1000
    text = torch.randn(2, 9, 24, generator=g)
    mask = torch.ones(2, 9, dtype=torch.bool)
    expected = model(x, t, text)
    actual = run_layers(layers, (x, t, text, mask))
    assert_close(actual, expected)
    check_backward(layers, (x, t, text, mask), model)


def test_ernie_image_padding_does_not_reach_the_image():
    from models import ernie_image
    model = tiny_ernie_image()
    layers = bare_pipeline(ernie_image.ErnieImagePipeline, model).to_layers()
    g = torch.Generator().manual_seed(2)
    lengths = [9, 5]
    x = torch.randn(2, 32, 6, 5, generator=g)
    t = torch.rand(2, generator=g) * 1000
    text = torch.randn(2, 9, 24, generator=g)
    mask = torch.arange(9)[None] < torch.tensor(lengths)[:, None]
    text[~mask] = 0
    actual = run_layers(layers, (x, t, text, mask))
    for i, n in enumerate(lengths):
        assert_close(actual[i:i + 1], model(x[i:i + 1], t[i:i + 1], text[i:i + 1, :n]))


def test_ernie_image_caption_padding_is_masked():
    # base.py pads the captions of a batch to the longest. Run ComfyUI's own padding and mask
    # code with the Ministral settings ComfyUI declares (tokenizer: pad id 11 on the left;
    # encoder: pad id 0, start 1) after the pipeline's fix: the padding must be masked out and
    # the real tokens keep the positions they have alone.
    import types
    from comfy import sd1_clip
    from models.ernie_image import pad_captions_on_the_right
    tokenizer = types.SimpleNamespace(pad_left=True, pad_token=11)
    clip = types.SimpleNamespace(special_tokens={'start': 1, 'pad': 0})
    te = types.SimpleNamespace(tokenizer=types.SimpleNamespace(clip='m', m=tokenizer),
                               cond_stage_model=types.SimpleNamespace(clip='m', m=clip))
    assert pad_captions_on_the_right(te) is te

    tokens = [(1, 1.0, 0), (500, 1.0, 1), (501, 1.0, 2)]
    sd1_clip.SDTokenizer.pad_tokens(tokenizer, tokens, 3)
    embedding = torch.nn.Embedding(600, 4)
    clip.transformer = types.SimpleNamespace(
        get_input_embeddings=lambda: lambda ids, out_dtype=None: embedding(ids).to(out_dtype))
    _, attention_mask, *_ = sd1_clip.SDClipModel.process_tokens(clip, [[t for t, _, _ in tokens]], 'cpu')
    assert [t for t, _, _ in tokens][:3] == [1, 500, 501]
    assert attention_mask.tolist() == [[1, 1, 1, 0, 0, 0]]

    # Caches built before hold left-padded embeddings: keep_text_embedding_cache must not keep them.
    from models.ernie_image import ErnieImagePipeline
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    assert pipe.text_encoder_cache_key(0) and pipe.text_encoder_requires_identity(0)


def tiny_qwen_image21():
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
    model = QwenImage21Transformer2DModel(
        in_channels=32, out_channels=32, num_layers=2, attention_head_dim=16, num_attention_heads=2,
        context_in_dim=24, mlp_ratio=3, axes_dims_rope=(4, 6, 6), dtype=torch.float32,
        operations=comfy.ops.manual_cast)
    return randomize(model)


def test_qwen_image21_layers_match_comfyui():
    # The pipeline masks text padding (ComfyUI never pads a prompt); without padding the masks
    # are ComfyUI's: causal text, image rows attending to everything.
    from models import qwen_image21
    model = tiny_qwen_image21()
    pipe = bare_pipeline(qwen_image21.QwenImage21Pipeline, model)
    layers = pipe.to_layers()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 32, 6, 5, generator=g)
    t = torch.rand(2, generator=g)
    text = torch.randn(2, 9, 24, generator=g)
    mask = torch.ones(2, 9, dtype=torch.bool)
    expected = model(x, t, text)
    actual = run_layers(layers, (x, t, text, mask))
    assert_close(actual, expected)
    check_backward(layers, (x, t, text, mask), model)


def test_qwen_image21_padding_does_not_reach_the_image():
    # A padded caption gives the same image output as the caption alone.
    from models import qwen_image21
    model = tiny_qwen_image21()
    layers = bare_pipeline(qwen_image21.QwenImage21Pipeline, model).to_layers()
    g = torch.Generator().manual_seed(2)
    lengths = [9, 6]
    x = torch.randn(2, 32, 6, 5, generator=g)
    t = torch.rand(2, generator=g)
    padded = torch.randn(2, 9, 24, generator=g)
    mask = torch.arange(9)[None] < torch.tensor(lengths)[:, None]
    actual = run_layers(layers, (x, t, padded, mask))
    for i, n in enumerate(lengths):
        expected = model(x[i:i + 1], t[i:i + 1], padded[i:i + 1, :n])
        assert_close(actual[i:i + 1], expected)


def test_qwen_image21_empty_caption_is_encoded_as_comfyui_encodes_it(monkeypatch):
    # TextEncodeQwenImage21 tokenizes with prevent_empty_text=True: an empty prompt becomes ' '.
    from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
    from models import base, qwen_image21
    tokenizer = QwenImage21Tokenizer()
    comfy_empty = tokenizer.tokenize_with_weights('', prevent_empty_text=True)
    assert comfy_empty != tokenizer.tokenize_with_weights('')

    def call_text_encoder_fn(self, text_encoder):
        return lambda captions, is_video: [tokenizer.tokenize_with_weights(c) for c in captions]

    monkeypatch.setattr(base.ComfyPipeline, 'get_call_text_encoder_fn', call_text_encoder_fn)
    pipe = bare_pipeline(qwen_image21.QwenImage21Pipeline, tiny_qwen_image21())
    empty, caption = pipe.get_call_text_encoder_fn(None)(['', 'a cat'], [False, False])
    assert empty == comfy_empty
    assert caption == tokenizer.tokenize_with_weights('a cat')
    # Caches built before encode empty captions as '': keep_text_embedding_cache must not keep them.
    assert pipe.text_encoder_cache_key(0) and pipe.text_encoder_requires_identity(0)


@pytest.mark.parametrize('module, cls, limit', [
    ('z_image', 'ZImagePipeline', 512), ('krea2', 'Krea2Pipeline', 546), ('flux2', 'Flux2Pipeline', 512),
    ('ideogram4', 'Ideogram4Pipeline', 2048), ('ltx2', 'LTX2Pipeline', 1024),
    ('hunyuan_video_15', 'HunyuanVideo15Pipeline', 1108), ('ernie_image', 'ErnieImagePipeline', 2048)])
def test_captions_longer_than_the_official_pipeline_keeps_are_warned_about_once(module, cls, limit, capsys):
    import importlib
    pipe_cls = getattr(importlib.import_module(f'models.{module}'), cls)
    pipe = pipe_cls.__new__(pipe_cls)
    pipe.warn_long_captions([limit, 3])
    assert capsys.readouterr().out == ''
    pipe.warn_long_captions([4, limit + 1])
    assert f'keeps only {limit}' in capsys.readouterr().out
    pipe.warn_long_captions([limit + 7])
    assert capsys.readouterr().out == ''


class RecordingHunyuanVideo15TextEncoder:
    """ComfyUI's HunyuanVideo 1.5 tokenizer in front of a stand-in encoder that records the
    tokens and returns zeros of the right shapes."""

    def __init__(self):
        from comfy.text_encoders.hunyuan_video import HunyuanVideo15Tokenizer
        self.tokenizer = HunyuanVideo15Tokenizer()
        self.tokenizer.clip = 'qwen25_7b'
        self.calls = []

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, False, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        self.calls.append(tokens)
        qwen = tokens['qwen25_7b']
        extra = {'attention_mask': torch.ones(len(qwen), len(qwen[0]), dtype=torch.int64)}
        if 'byt5' in tokens:
            extra['conditioning_byt5small'] = torch.zeros(len(tokens['byt5']), len(tokens['byt5'][0]), 1472)
        return [[torch.zeros(len(qwen), len(qwen[0]), 8), extra]]


def hunyuan_video_15_text_pipeline(model_config=None):
    from models import hunyuan_video_15
    pipe = bare_pipeline(hunyuan_video_15.HunyuanVideo15Pipeline, None, model_config)
    te = RecordingHunyuanVideo15TextEncoder()
    pipe.text_encoders = [te]
    return pipe, te, pipe.get_call_text_encoder_fn(te)


def test_hunyuan_video_15_byt5_mask_covers_only_a_captions_own_tokens():
    # Each batch pads its captions to its longest; the next batch must not inherit that length.
    pipe, te, fn = hunyuan_video_15_text_pipeline()
    fn(['a sign that says "' + 'x' * 60 + '"', 'a dog'], [False, False])
    alone = fn(['a sign that says "hi"'], [False])
    own_length = len(te.tokenizer.byt5.tokenize_with_weights('Text "hi". ')[0])
    assert alone['byt5_attention_mask'].sum().item() == own_length
    assert alone['byt5_attention_mask'].shape[1] == own_length


def test_hunyuan_video_15_official_prompt_template():
    from models import hunyuan_video_15
    pipe, te, fn = hunyuan_video_15_text_pipeline({'prompt_template': 'official'})
    caption = 'a sign that says “OPEN” and "OPEN" and ‘closed’ and "sale"'
    fn([caption, caption], [False, True])
    tokens = te.calls[-1]
    qwen = te.tokenizer.qwen25_7b
    image_template = hunyuan_video_15.OFFICIAL_IMAGE_TEMPLATE.format(caption)
    video_template = hunyuan_video_15.OFFICIAL_VIDEO_TEMPLATE.format(caption)
    for row, text in zip(tokens['qwen25_7b'], [image_template, video_template]):
        ids = [t for t, *_ in row if t != 151643]
        assert ids == [t for t, *_ in qwen.tokenize_with_weights(text, disable_weights=True)[0]]
    # In order of appearance, de-duplicated, without ‘...’: as the official pipeline extracts them.
    expected = te.tokenizer.byt5.tokenize_with_weights('Text "OPEN". Text "sale". ')[0]
    assert [t for t, *_ in tokens['byt5'][0]] == [t for t, *_ in expected]
    comfyui_pipe = hunyuan_video_15_text_pipeline()[0]
    assert pipe.text_encoder_cache_key(0) != comfyui_pipe.text_encoder_cache_key(0)
    # Caches built before the option hold ComfyUI's template; ComfyUI's keeps them valid.
    assert pipe.text_encoder_requires_identity(0) and not comfyui_pipe.text_encoder_requires_identity(0)

    _, te, fn = hunyuan_video_15_text_pipeline()
    fn([caption], [False])
    assert te.calls[-1]['qwen25_7b'][0] != tokens['qwen25_7b'][0]


def test_hunyuan_video_15_official_byt5_is_cut_at_256_tokens():
    _, te, fn = hunyuan_video_15_text_pipeline({'prompt_template': 'official'})
    out = fn(['a sign that says "' + 'x' * 400 + '"'], [False])
    row = te.calls[-1]['byt5'][0]
    assert len(row) == 256 and row[-1][0] == 1
    assert out['byt5_attention_mask'].sum().item() == 256


def test_hunyuan_video_15_refuses_an_unknown_prompt_template():
    from models import hunyuan_video_15
    pipe = hunyuan_video_15.HunyuanVideo15Pipeline.__new__(hunyuan_video_15.HunyuanVideo15Pipeline)
    with pytest.raises(ValueError, match='prompt_template'):
        hunyuan_video_15.HunyuanVideo15Pipeline.__init__(pipe, {'model': {'dtype': torch.float32, 'prompt_template': 'diffusers'}})


def write_checkpoint_header(path, names, metadata=None, shapes=None):
    # As ComfyUI files store them: under model.diffusion_model., with a few other tensors.
    import safetensors.torch
    tensors = {f'model.diffusion_model.{name}': torch.zeros((shapes or {}).get(name, (1,))) for name in names}
    tensors.update({f'model.diffusion_model.filler_{i}': torch.zeros(1) for i in range(6)})
    tensors['vae.decoder.weight'] = torch.zeros(1)
    safetensors.torch.save_file(tensors, str(path), metadata=metadata)
    return str(path)


Z_IMAGE = {'cap_embedder.1.weight': (3840, 1)}


@pytest.mark.parametrize('module, cls, names, metadata, shapes, refused', [
    ('z_image', 'ZImagePipeline', ['cap_embedder.1.weight', 'cap_pad_token'], None, Z_IMAGE, None),
    ('z_image', 'ZImagePipeline', ['cap_embedder.1.weight', 'cap_pad_token', '__ming_image__'], None, Z_IMAGE, 'Ming-Image'),
    ('z_image', 'ZImagePipeline', ['cap_embedder.1.weight', 'cap_pad_token'],
     {'config': '{"transformer": {"image_model": "ming_image"}}'}, Z_IMAGE, 'Ming-Image'),
    ('z_image', 'ZImagePipeline', ['cap_embedder.1.weight'], None, Z_IMAGE, 'Ming-Image'),
    ('z_image', 'ZImagePipeline', ['cap_embedder.1.weight', 'dec_net.cond_embed.weight'], None, Z_IMAGE, 'Pixel-space'),
    ('z_image', 'ZImagePipeline', ['cap_embedder.1.weight', 'cap_pad_token', 'siglip_embedder.0.weight'], None, Z_IMAGE, 'Omni'),
    ('hunyuan_video_15', 'HunyuanVideo15Pipeline', ['img_in.proj.weight'], None, None, None),
    ('hunyuan_video_15', 'HunyuanVideo15Pipeline', ['guidance_in.in_layer.weight'], None, None, 'guidance distilled'),
    ('hunyuan_video_15', 'HunyuanVideo15Pipeline', ['time_r_in.in_layer.weight'], None, None, 'MeanFlow'),
    ('minimax_h3', 'MinimaxH3Pipeline', ['blocks.0.weight'], {'config': '{"transformer": {"pdd_steps": 1}}'}, None, None),
    ('minimax_h3', 'MinimaxH3Pipeline', ['blocks.0.weight'], {'config': '{"transformer": {"pdd_steps": 4}}'}, None, 'PDD'),
    ('ltx2', 'LTX2Pipeline', ['transformer_blocks.0.prompt_scale_shift_table'], None, None, None),
    ('ltx2', 'LTX2Pipeline', ['transformer_blocks.0.scale_shift_table'], None, None, 'Only LTX 2.3'),
])
def test_unsupported_checkpoints_are_refused_before_the_dataset_is_cached(
        tmp_path, module, cls, names, metadata, shapes, refused):
    # train.py caches the dataset before it loads the model: a refusal from to_layers() came
    # after hours of caching. The pipeline reads the checkpoint's header when it is created.
    import importlib
    pipe_cls = getattr(importlib.import_module(f'models.{module}'), cls)
    path = write_checkpoint_header(tmp_path / 'model.safetensors', names, metadata, shapes)
    pipe = bare_pipeline(pipe_cls, None, {'diffusion_model': path})
    if refused is None:
        pipe.refuse_unsupported_checkpoint()
    else:
        with pytest.raises(NotImplementedError, match=refused):
            pipe.refuse_unsupported_checkpoint()


def tiny_hunyuan_video_15():
    from comfy.ldm.hunyuan_video.model import HunyuanVideo
    model = HunyuanVideo(
        in_channels=16, out_channels=16, vec_in_dim=None, context_in_dim=24, hidden_size=64, mlp_ratio=2.0,
        num_heads=2, depth=2, depth_single_blocks=0, axes_dim=[8, 12, 12], theta=256, patch_size=[1, 2, 2],
        qkv_bias=True, guidance_embed=False, byt5=True, meanflow=False, use_cond_type_embedding=True,
        vision_in_dim=None, meanflow_sum=False, dtype=torch.float32, operations=comfy.ops.manual_cast)
    return randomize(model)


def hunyuan_video_15_inputs(text_lengths, byt5_lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    bs = len(text_lengths)
    x = torch.randn(bs, 16, 2, 4, 6, generator=g)
    t = torch.rand(bs, generator=g) * 1000
    text = torch.randn(bs, max(text_lengths), 24, generator=g)
    text_mask = torch.arange(text.shape[1])[None] < torch.tensor(text_lengths)[:, None]
    byt5 = torch.randn(bs, max(byt5_lengths), 1472, generator=g)
    byt5_mask = torch.arange(byt5.shape[1])[None] < torch.tensor(byt5_lengths)[:, None]
    text[~text_mask] = 0  # what prepare_inputs pads with
    byt5[~byt5_mask] = 0
    return x, t, text, text_mask, byt5, byt5_mask


def test_hunyuan_video_15_layers_match_comfyui():
    from models import hunyuan_video_15
    model = tiny_hunyuan_video_15()
    layers = bare_pipeline(hunyuan_video_15.HunyuanVideo15Pipeline, model).to_layers()
    x, t, text, text_mask, byt5, byt5_mask = hunyuan_video_15_inputs([9, 9], [5, 5])
    expected = model(x, t, text, txt_byt5=byt5)
    actual = run_layers(layers, (x, t, text, text_mask, byt5, byt5_mask))
    assert_close(actual, expected)
    # With no single blocks after it (as in HunyuanVideo 1.5), the last double block's text output
    # is dropped: its text-only weights get no gradient, in ComfyUI too.
    last = f'double_blocks.{len(model.double_blocks) - 1}.'
    unused = [n for n, _ in model.named_parameters()
              if n.startswith(last + 'txt_attn.proj') or n.startswith(last + 'txt_mlp')]
    check_backward(layers, (x, t, text, text_mask, byt5, byt5_mask), model, autocast_dtype=torch.bfloat16,
                   unused=unused)


def test_hunyuan_video_15_padding_does_not_reach_the_video():
    """A caption padded to the longest of its batch trains like the caption alone, which is what
    ComfyUI runs at inference (one prompt, no padding): the padding is out of the token
    refiner's pooled context and attention, and out of the blocks' attention."""
    from models import hunyuan_video_15
    model = tiny_hunyuan_video_15()
    layers = bare_pipeline(hunyuan_video_15.HunyuanVideo15Pipeline, model).to_layers()
    text_lengths, byt5_lengths = [9, 5], [3, 6]
    x, t, text, text_mask, byt5, byt5_mask = hunyuan_video_15_inputs(text_lengths, byt5_lengths)
    actual = run_layers(layers, (x, t, text, text_mask, byt5, byt5_mask))
    for i, (n, m) in enumerate(zip(text_lengths, byt5_lengths)):
        expected = model(x[i:i + 1], t[i:i + 1], text[i:i + 1, :n], txt_byt5=byt5[i:i + 1, :m])
        assert_close(actual[i:i + 1], expected)


def tiny_ltx2(caption_proj_before_connector=True, **kwargs):
    # LTX 2.3, the only version the pipeline trains: cross-attention adaLN and gated attention
    from comfy.ldm.lightricks.av_model import LTXAVModel
    config = dict(
        in_channels=16, audio_in_channels=128, cross_attention_dim=64, audio_cross_attention_dim=32,
        attention_head_dim=32, audio_attention_head_dim=16, num_attention_heads=2, audio_num_attention_heads=2,
        caption_channels=48, num_layers=2, connector_attention_head_dim=32, connector_num_attention_heads=2,
        audio_connector_attention_head_dim=16, audio_connector_num_attention_heads=2,
        connector_num_layers=1, cross_attention_adaln=True, apply_gated_attention=True,
        caption_proj_before_connector=caption_proj_before_connector, caption_projection_first_linear=True,
        dtype=torch.float32,
        operations=comfy.ops.manual_cast)
    config.update(kwargs)
    return randomize(LTXAVModel(**config))


def ltx2_pipeline(model):
    from models import ltx2
    pipe = bare_pipeline(ltx2.LTX2Pipeline, model)
    pipe.framerate = 25
    return pipe


def ltx2_inputs(text_lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    bs = len(text_lengths)
    vx = torch.randn(bs, 16, 2, 4, 6, generator=g)
    ax = torch.zeros(bs, 0)
    t = torch.rand(bs, generator=g)
    text = torch.randn(bs, max(text_lengths), 48, generator=g)
    mask = torch.arange(text.shape[1])[None] < torch.tensor(text_lengths)[:, None]
    text[~mask] = 0
    return vx, ax, t, text, mask


def test_ltx2_refuses_ltx2_0():
    with pytest.raises(NotImplementedError, match='LTX 2.3'):
        ltx2_pipeline(tiny_ltx2(cross_attention_adaln=False)).to_layers()


def test_ltx2_layers_match_comfyui():
    model = tiny_ltx2()
    layers = ltx2_pipeline(model).to_layers()
    vx, ax, t, text, mask = ltx2_inputs([7, 7])
    # ComfyUI runs the connectors when it builds the conditioning (model_base), not in the forward.
    context = model.preprocess_text_embeds(text, unprocessed=True)
    expected = model([vx], t, context, attention_mask=None, frame_rate=25)
    actual = run_layers(layers, (vx, ax, t, text, mask))
    assert_close(actual, expected)
    # Video only (the pipeline does not train audio yet): the audio stream and the attention
    # between the streams do not run.
    unused = [n for n, _ in model.named_parameters()
              if 'audio' in n or n.startswith('av_ca_') or '.audio_to_video' in n or '.video_to_audio' in n
              or 'a2v' in n or 'v2a' in n]
    check_backward(layers, (vx, ax, t, text, mask), model, unused=unused)


def test_ltx2_padding_does_not_reach_the_video():
    """A caption padded to the longest of its batch trains like the caption alone (ComfyUI's
    inference): the connectors replace the padding with their registers."""
    model = tiny_ltx2()
    layers = ltx2_pipeline(model).to_layers()
    lengths = [7, 4]
    vx, ax, t, text, mask = ltx2_inputs(lengths)
    actual = run_layers(layers, (vx, ax, t, text, mask))
    for i, n in enumerate(lengths):
        context = model.preprocess_text_embeds(text[i:i + 1, :n], unprocessed=True)
        expected = model([vx[i:i + 1]], t[i:i + 1], context, attention_mask=None, frame_rate=25)
        assert_close(actual[i:i + 1], expected)


def test_ltx2_cached_captions_train_as_comfyui_runs_them():
    """Captions as the cache holds them: the Gemma tokenizer pads each on the left to 1024
    tokens, and the batch pads on the right to its longest. ComfyUI drops the padding and the
    connectors append registers up to a multiple of 128 covering max(1024, tokens), so a
    caption over 1024 tokens gets more registers than the short one, which masks them out."""
    model = tiny_ltx2()
    layers = ltx2_pipeline(model).to_layers()
    lengths = [1100, 4]
    vx, ax, t, _, _ = ltx2_inputs(lengths)
    g = torch.Generator().manual_seed(3)
    text = torch.zeros(2, 1100, 48)
    mask = torch.zeros(2, 1100, dtype=torch.bool)
    text[0] = torch.randn(1100, 48, generator=g)
    mask[0] = True
    text[1, 1020:1024] = torch.randn(4, 48, generator=g)
    mask[1, 1020:1024] = True
    actual = run_layers(layers, (vx, ax, t, text, mask))
    for i in range(2):
        context = model.preprocess_text_embeds(text[i:i + 1, mask[i]], unprocessed=True)
        expected = model([vx[i:i + 1]], t[i:i + 1], context, attention_mask=None, frame_rate=25)
        assert_close(actual[i:i + 1], expected)


def test_ltx2_video_to_audio_sees_the_video_before_audio_to_video():
    """models/ltx2.py replaces BasicAVTransformerBlock.forward (ComfyUI's adds in place, which
    autograd refuses). As in Lightricks' ltx-core, video-to-audio attention reads the video
    tokens from before audio-to-video attention updated them, so the audio output does not
    depend on whether audio-to-video attention ran. (ComfyUI master reads the updated ones.)"""
    import models.ltx2  # noqa: F401  (installs the patched forward)
    model = tiny_ltx2()
    block = model.transformer_blocks[0]
    g = torch.Generator().manual_seed(3)
    bs, nv, na = 2, 12, 5
    vx = torch.randn(bs, nv, 64, generator=g)
    ax = torch.randn(bs, na, 32, generator=g)
    ts = lambda n, d, k: torch.randn(bs, n, k * d, generator=g)  # noqa: E731
    kwargs = dict(
        v_context=torch.randn(bs, 3, 64, generator=g), a_context=torch.randn(bs, 3, 32, generator=g),
        v_timestep=ts(nv, 64, block.scale_shift_table.shape[0]), a_timestep=ts(na, 32, block.audio_scale_shift_table.shape[0]),
        v_cross_scale_shift_timestep=ts(nv, 64, 4), a_cross_scale_shift_timestep=ts(na, 32, 4),
        v_cross_gate_timestep=ts(nv, 64, 1), a_cross_gate_timestep=ts(na, 32, 1),
        v_prompt_timestep=ts(1, 64, 2), a_prompt_timestep=ts(1, 32, 2))
    with torch.no_grad():
        _, ax_both = block((vx.clone(), ax.clone()), transformer_options={}, **kwargs)
        _, ax_v2a_only = block((vx.clone(), ax.clone()), transformer_options={'a2v_cross_attn': False}, **kwargs)
    assert_close(ax_both, ax_v2a_only)


MINIMAX_CONFIG = dict(
    hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
    ffn_hidden_size=96, latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=40,
    timestep_input_dim=32, time_embed_hidden_size=48, time_embed_dim=32, rope_inv_freq_len=4)


def pristine_minimax_module():
    """A fresh copy of comfy/ldm/minimax/model.py: models/minimax_h3.py replaces some of its
    classes and functions in place with batched, out-of-place versions."""
    import importlib.util
    import comfy.ldm.minimax.model as patched
    spec = importlib.util.spec_from_file_location('_pristine_minimax_model', patched.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_minimax_pair():
    """(the model built with the pipeline's patched classes, the same weights in ComfyUI's own)"""
    import models.minimax_h3  # noqa: F401  (installs the patches)
    import comfy.ldm.minimax.model
    model = comfy.ldm.minimax.model.MiniMaxH3Model(**MINIMAX_CONFIG, dtype=torch.float32, operations=comfy.ops.manual_cast)
    randomize(model)
    with torch.no_grad():
        model.rope.inv_freq.copy_(1.0 / 100.0 ** (torch.arange(4, dtype=torch.float32) / 4))
    reference = pristine_minimax_module().MiniMaxH3Model(**MINIMAX_CONFIG, dtype=torch.float32, operations=comfy.ops.manual_cast)
    reference.load_state_dict(model.state_dict())
    return model, reference


def minimax_pipeline(model):
    from models import minimax_h3
    pipe = bare_pipeline(minimax_h3.MinimaxH3Pipeline, model)
    pipe.cfg = 1.0
    return pipe


def minimax_inputs(text_lengths, audio_t=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    bs = len(text_lengths)
    video = torch.randn(bs, 24, 2, 4, 6, generator=g)
    audio = torch.randn(bs, 32, 2, audio_t, generator=g)
    valid_audio = torch.tensor([audio_t > 0] * bs)
    t = torch.rand(bs, generator=g)
    text = torch.randn(bs, max(text_lengths), 40, generator=g)
    mask = torch.arange(text.shape[1])[None] < torch.tensor(text_lengths)[:, None]
    text[~mask] = 0
    return video, audio, valid_audio, t, text, mask


@pytest.mark.parametrize('audio_t', [0, 3], ids=['video', 'video+audio'])
def test_minimax_h3_layers_match_comfyui(audio_t):
    model, reference = tiny_minimax_pair()
    layers = minimax_pipeline(model).to_layers()
    inputs = minimax_inputs([6, 6], audio_t=audio_t)
    video, audio, valid_audio, t, text, mask = inputs
    actual_video, actual_audio = run_layers(layers, inputs)
    for i in range(video.shape[0]):
        expected_video, expected_audio = reference([video[i:i + 1], audio[i:i + 1]], t[i:i + 1] * 1000, text[i:i + 1])
        assert_close(actual_video[i:i + 1], expected_video)
        assert_close(actual_audio[i:i + 1], expected_audio)

    model.zero_grad(set_to_none=True)
    actual_video, actual_audio = run_layers(layers, inputs)
    (actual_video.square().mean() + actual_audio.square().mean()).backward()
    for name, p in model.blocks.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_minimax_h3_padding_does_not_reach_the_video():
    """A caption padded to the longest of its batch trains like the caption alone (ComfyUI's
    inference): the padding is masked and does not move the video and audio positions."""
    model, reference = tiny_minimax_pair()
    layers = minimax_pipeline(model).to_layers()
    lengths = [6, 3]
    inputs = minimax_inputs(lengths)
    video, audio, valid_audio, t, text, mask = inputs
    actual_video, actual_audio = run_layers(layers, inputs)
    for i, n in enumerate(lengths):
        expected_video, expected_audio = reference([video[i:i + 1], audio[i:i + 1]], t[i:i + 1] * 1000, text[i:i + 1, :n])
        assert_close(actual_video[i:i + 1], expected_video)
        assert_close(actual_audio[i:i + 1], expected_audio)


@pytest.mark.parametrize('with_other', [False, True], ids=['alone', 'batched'])
def test_minimax_h3_empty_caption_matches_comfyui(with_other):
    """An empty caption (caption dropout, the uncond of cfg > 1) is one pad token that the
    encoder masks out. ComfyUI runs it as a one-token context and attends to it."""
    model, reference = tiny_minimax_pair()
    layers = minimax_pipeline(model).to_layers()
    video, audio, valid_audio, t, text, mask = minimax_inputs([6, 6])
    pad = torch.randn(40, generator=torch.Generator().manual_seed(4))
    text[1] = 0
    text[1, 0] = pad
    mask[1] = False
    first = 0 if with_other else 1
    actual_video, actual_audio = run_layers(
        layers, (video[first:], audio[first:], valid_audio[first:], t[first:], text[first:], mask[first:]))
    expected_video, expected_audio = reference([video[1:2], audio[1:2]], t[1:2] * 1000, pad[None, None])
    assert_close(actual_video[-1:], expected_video)
    assert_close(actual_audio[-1:], expected_audio)


def test_minimax_h3_builds_with_the_current_comfyui_signature():
    """ComfyUI passes pdd_steps to FinalLayer; the pipeline's replacement must take it (and
    refuse the PDD head bank it does not train)."""
    import models.minimax_h3  # noqa: F401
    import comfy.ldm.minimax.model
    comfy.ldm.minimax.model.MiniMaxH3Model(**MINIMAX_CONFIG, pdd_steps=1, dtype=torch.float32, operations=comfy.ops.manual_cast)
    with pytest.raises(NotImplementedError, match='PDD'):
        comfy.ldm.minimax.model.MiniMaxH3Model(**MINIMAX_CONFIG, pdd_steps=4, dtype=torch.float32, operations=comfy.ops.manual_cast)
