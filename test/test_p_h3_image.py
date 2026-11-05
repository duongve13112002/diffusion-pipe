"""P-H3-Image (models/p_h3_image.py) on CPU.

The layers against ComfyUI's own MiniMax-H3 forward with P-H3-Image's conditioning (a reference
image, token tags, shifts 1.0 / 1.0, two audio latents), the text encoder function against
ComfyUI's PH3ImageConditioning node on a tiny Qwen3-VL-4B, the inputs and the loss, the config
refusals, and a training step through DeepSpeed's pipeline engine in gloo process groups.
"""

import ast
import dataclasses
import json
import math
import re
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from PIL import Image

import comfy.model_management
import comfy.ops

from test.test_comfy_equivalence import (  # noqa: F401  (training_path and pytorch_attention are autouse fixtures)
    MINIMAX_CONFIG, assert_close, bare_pipeline, pristine_minimax_module, randomize, run_layers,
    training_path, pytorch_attention,
)

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
SHIFTS = {'minimax_h3_sigma_shift_video': 1.0, 'minimax_h3_sigma_shift_audio': 1.0}
LATENT_H, LATENT_W = 4, 6


def tiny_pair():
    """(the model with the pipeline's patched classes and P-H3-Image's shifts, ComfyUI's own)"""
    import models.p_h3_image  # noqa: F401  (installs the patches)
    import comfy.ldm.minimax.model
    model = comfy.ldm.minimax.model.MiniMaxH3Model(**MINIMAX_CONFIG, dtype=torch.float32, operations=comfy.ops.manual_cast)
    randomize(model)
    with torch.no_grad():
        model.rope.inv_freq.copy_(1.0 / 100.0 ** (torch.arange(4, dtype=torch.float32) / 4))
    model.sigma_shift_video = model.sigma_shift_audio = 1.0
    reference = pristine_minimax_module().MiniMaxH3Model(**MINIMAX_CONFIG, dtype=torch.float32, operations=comfy.ops.manual_cast)
    reference.load_state_dict(model.state_dict())
    return model, reference


def ph3_pipeline(model, model_config=None):
    from models import p_h3_image
    import comfy.latent_formats
    pipe = bare_pipeline(p_h3_image.PH3ImagePipeline, model, model_config)
    pipe.cfg = 1.0
    pipe.latent_format = comfy.latent_formats.MiniMaxH3Video()
    pipe.edit_prompt_prefix = True
    return pipe


def comfy_augmented(ref, seed=0):
    """ComfyUI's _cond_video_rows noise augmentation of a clean reference latent, as a latent:
    what prepare_inputs hands the layers."""
    from comfy.ldm.minimax.model import patchify_video, unpatchify_video, VISUAL_COND_TIMESTEP
    rows = patchify_video(ref.float())
    noise = torch.randn(rows.shape, generator=torch.Generator('cpu').manual_seed(seed))
    rows = VISUAL_COND_TIMESTEP * rows + (1.0 - VISUAL_COND_TIMESTEP) * noise
    out = unpatchify_video(rows, 1, ref.shape[-2] // 2, ref.shape[-1] // 2, ref.shape[1])
    assert torch.equal(patchify_video(out), rows)
    return out


def reference_forward(reference, video, audio, t, text, tags=None, ref=None):
    payload = {'seed': 0}
    if tags is not None:
        payload['text_token_tags'] = tags
    if ref is not None:
        payload['refs'] = [{'kind': 'image', 'latent_h': ref.shape[-2], 'latent_w': ref.shape[-1]}]
        payload['cond_video_latents'] = [ref]
    return reference([video, audio], t * 1000, text, transformer_options=dict(SHIFTS), minimax_payload=payload)


def tags_with_vision(n, start, length):
    tags = torch.ones(n, dtype=torch.long)
    tags[start:start + length] = 0
    return tags


def edit_inputs(lengths, vision, t, seed=1):
    """Batch of edit samples: captions of `lengths` tokens whose vision block is vision[i] =
    (start, length), padded to the longest; reference latents clean and augmented."""
    g = torch.Generator().manual_seed(seed)
    bs = len(lengths)
    video = torch.randn(bs, 24, 1, LATENT_H, LATENT_W, generator=g)
    audio = torch.randn(bs, 32, 2, 2, generator=g)
    text = torch.randn(bs, max(lengths), 40, generator=g)
    mask = torch.arange(text.shape[1])[None] < torch.tensor(lengths)[:, None]
    text[~mask] = 0
    tags = torch.ones(bs, max(lengths), dtype=torch.long)
    for i, (n, (start, length)) in enumerate(zip(lengths, vision)):
        tags[i, :n] = tags_with_vision(n, start, length)
    ref = torch.randn(bs, 24, 1, LATENT_H, LATENT_W, generator=g)
    augmented = torch.cat([comfy_augmented(ref[i:i + 1]) for i in range(bs)])
    return video, audio, torch.ones(bs, dtype=torch.bool), torch.tensor(t), text, mask, tags, ref, augmented


def test_text_to_image_layers_match_comfyui():
    """Padded captions of different lengths, all tags 1 (text only), one latent frame and the
    two audio latents of a still: each sample as ComfyUI runs it alone."""
    model, reference = tiny_pair()
    layers = ph3_pipeline(model).to_layers()
    lengths = [6, 3]
    video, audio, valid, t, text, mask, tags, _, _ = edit_inputs(lengths, [(0, 0), (0, 0)], [0.3, 0.8])
    actual_video, actual_audio = run_layers(layers, (video, audio, valid, t, text, mask, tags))
    for i, n in enumerate(lengths):
        expected_video, expected_audio = reference_forward(reference, video[i:i + 1], audio[i:i + 1], t[i:i + 1], text[i:i + 1, :n])
        assert_close(actual_video[i:i + 1], expected_video)
        assert_close(actual_audio[i:i + 1], expected_audio)


@pytest.mark.parametrize('t', [[0.3, 0.7], [0.3, 0.0005]], ids=['ordinary', 'sigma-below-ref-pin'])
def test_edit_layers_match_comfyui(t):
    """A reference image and the token tags of its vision block, in a batch whose samples have
    different caption lengths and vision blocks of different lengths (so different adaLN segments
    per sample). At sigma < 0.001 the image's timestep passes the reference's pin of 0.999 and
    ComfyUI merges their timestep rows; the layers keep a row per stream with the same values."""
    model, reference = tiny_pair()
    layers = ph3_pipeline(model).to_layers()
    lengths, vision = [9, 6], [(2, 4), (1, 2)]
    video, audio, valid, tt, text, mask, tags, ref, augmented = edit_inputs(lengths, vision, t)
    inputs = (video, audio, valid, tt, text, mask, tags, augmented)
    first = layers[0](inputs)
    assert first[3].ndim == 3, 'the samples have different segments: one list per sample'
    actual_video, actual_audio = run_layers(layers, inputs)
    for i, n in enumerate(lengths):
        expected_video, expected_audio = reference_forward(
            reference, video[i:i + 1], audio[i:i + 1], tt[i:i + 1], text[i:i + 1, :n], tags[i, :n], ref[i:i + 1])
        assert_close(actual_video[i:i + 1], expected_video)
        assert_close(actual_audio[i:i + 1], expected_audio)

    model.zero_grad(set_to_none=True)
    actual_video, _ = run_layers(layers, inputs)
    actual_video.square().mean().backward()
    for name, p in model.blocks.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_same_segments_stay_shared():
    """Samples whose segments agree keep one shared list (the batched modulation path)."""
    model, _ = tiny_pair()
    layers = ph3_pipeline(model).to_layers()
    video, audio, valid, t, text, mask, tags, _, augmented = edit_inputs([7, 5], [(1, 3), (1, 3)], [0.2, 0.0005])
    first = layers[0]((video, audio, valid, t, text, mask, tags, augmented))
    assert first[3].ndim == 2


def test_the_tags_change_the_output():
    """The token tags reach the model: the vision block's tokens take the video modality."""
    model, _ = tiny_pair()
    layers = ph3_pipeline(model).to_layers()
    video, audio, valid, t, text, mask, tags, _, augmented = edit_inputs([7], [(1, 3)], [0.4])
    with_tags, _ = run_layers(layers, (video, audio, valid, t, text, mask, tags, augmented))
    without, _ = run_layers(layers, (video, audio, valid, t, text, mask, torch.ones_like(tags), augmented))
    assert not torch.allclose(with_tags, without)


class StubVAE:
    """The P-H3-Image VAE's encode as far as batching matters: a seed-42 posterior sample drawn
    for the shape it is given, so a batch encodes its later images with other noise."""

    def __init__(self):
        self.batch_sizes = []

    def encode(self, pixels):
        self.batch_sizes.append(pixels.shape[0])
        b, h, w, _ = pixels.shape
        noise = torch.randn((b, 24, 1, h // 16, w // 16), generator=torch.Generator().manual_seed(42))
        return pixels.mean(dim=(1, 2, 3)).view(b, 1, 1, 1, 1) + noise


def _cpu_to(original):
    # The pipeline moves pixels to 'cuda'; the tests run on CPU.
    def to(self, *args, **kwargs):
        args = tuple('cpu' if a == 'cuda' else a for a in args)
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original(self, *args, **kwargs)
    return to


def test_images_are_encoded_one_at_a_time(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'to', _cpu_to(torch.Tensor.to))
    pipe = ph3_pipeline(None, {'dtype': torch.float32})
    vae = StubVAE()
    images = torch.rand(3, 3, 64, 96)
    result = pipe.get_call_vae_fn(vae)(images, images.flip(0))
    assert vae.batch_sizes == [1] * 6
    alone = pipe.latent_format.process_in(StubVAE().encode(images[2:3].movedim(1, -1)))
    assert_close(result['latents'][2:3], alone)
    assert result['control_latents'].shape == (3, 24, 1, 4, 6)
    # 5D input (a one-frame bucket) as the dataset hands it
    assert_close(pipe.encode_images(images[:, :, None], StubVAE()), result['latents'])


def test_prepare_inputs():
    torch.manual_seed(0)
    pipe = ph3_pipeline(None, {'timestep_sample_method': 'uniform'})
    bs = 3
    latents = torch.randn(bs, 24, 1, LATENT_H, LATENT_W)
    control = torch.randn(bs, 24, 1, LATENT_H, LATENT_W)
    text = [torch.randn(n, 40) for n in (5, 7, 2)]
    tags = [tags_with_vision(n, 1, 1) for n in (5, 7, 2)]
    inputs = {'latents': latents, 'mask': None, 'control_latents': control, 'text_embeds_0': text,
              'attention_mask_0': [torch.ones(n, dtype=torch.int64) for n in (5, 7, 2)], 'token_tags_0': tags}
    (noisy, noisy_audio, valid, t, text_embeds, attention_mask, token_tags, ref), (target, mask) = pipe.prepare_inputs(inputs)
    assert noisy.shape == latents.shape and target.shape == latents.shape
    assert noisy_audio.shape == (bs, 32, 2, 2) and valid.all()
    assert text_embeds.shape == (bs, 7, 40) and attention_mask.dtype == torch.bool
    assert token_tags.tolist()[2] == [1, 0, 1, 1, 1, 1, 1], 'padding takes tag 1'
    # the reference keeps 99.9% of the clean latent
    assert (ref - 0.999 * control).abs().max() < 0.01 and not torch.equal(ref, control)
    # the audio is silence noised at the image's sigma
    assert torch.allclose(noisy_audio.flatten(1).std(1), t, rtol=0.6)
    assert mask is None

    with pytest.raises(ValueError, match='do not match'):
        pipe.prepare_inputs({**inputs, 'control_latents': control[..., :2]})
    with pytest.raises(ValueError, match='images only'):
        pipe.prepare_inputs({**inputs, 'latents': torch.randn(bs, 24, 2, LATENT_H, LATENT_W)})
    (*model_inputs,), _ = pipe.prepare_inputs({k: v for k, v in inputs.items() if k != 'control_latents'})
    assert len(model_inputs) == 7


def test_the_loss_ignores_the_audio():
    pipe = ph3_pipeline(None)
    loss_fn = pipe.get_loss_fn()
    output, target = torch.randn(2, 24, 1, 4, 6), torch.randn(2, 24, 1, 4, 6)
    a = loss_fn((output, torch.randn(2, 32, 2, 2)), (target, None))
    b = loss_fn((output, torch.randn(2, 32, 2, 2)), (target, None))
    assert a == b == torch.nn.functional.mse_loss(output, target)
    mask = torch.zeros(2, 1, 1, 4, 6)
    mask[0] = 1
    assert_close(loss_fn((output, None), (target, mask)), torch.nn.functional.mse_loss(output[0], target[0]) / 2)


@pytest.mark.parametrize('key, value, match', [
    ('cfg', 2.0, 'cfg does not apply'),
    ('audio_vae', '/x.safetensors', 'audio_vae does not apply'),
])
def test_keys_that_do_not_apply_are_refused(key, value, match):
    from models.p_h3_image import PH3ImagePipeline
    config = {'model': {key: value, 'text_encoders': [{'path': 'x', 'type': 'minimax'}]}}
    with pytest.raises(ValueError, match=match):
        PH3ImagePipeline(config)


@pytest.mark.parametrize('text_encoders', [[], [{'path': 'x', 'type': 'qwen_image'}], [{'path': 'x', 'type': 'minimax'}] * 2])
def test_the_text_encoder_must_be_the_minimax_one(text_encoders):
    from models.p_h3_image import PH3ImagePipeline
    with pytest.raises(ValueError, match='one text encoder'):
        PH3ImagePipeline({'model': {'text_encoders': text_encoders}})


@pytest.mark.parametrize('dataset, match', [
    ({'frame_buckets': [1, 33]}, 'images only'),
    ({'directory': [{'path': 'x', 'frame_buckets': [1, 33]}]}, r'images only.*\[\[directory\]\] x'),
    ({'directory': [{'path': 'x', 'control_path': 'c', 'control_path_2': 'd'}]}, 'control_path_N'),
    ({'directory': [{'path': 'x', 'control_path': 'c', 'caption_dropout_rate': 0.1}]}, 'caption_dropout_rate does not apply'),
    ({'caption_dropout_rate': 0.1, 'directory': [{'path': 'x', 'control_path': 'c'}]}, 'caption_dropout_rate does not apply'),
])
def test_dataset_refusals(dataset, match):
    pipe = ph3_pipeline(None)
    with pytest.raises(ValueError, match=match):
        pipe.model_specific_dataset_config_validation(dataset)
    pipe.model_specific_dataset_config_validation({'frame_buckets': [1], 'directory': [{'path': 'x', 'control_path': 'c'}]})
    # caption dropout stays allowed for text to image, and an edit directory can turn it off
    pipe.model_specific_dataset_config_validation({'caption_dropout_rate': 0.1, 'directory': [
        {'path': 't2i'}, {'path': 'edit', 'control_path': 'c', 'caption_dropout_rate': 0}]})


def test_load_vae_refuses_the_h3_video_vae(monkeypatch):
    import comfy.sd
    import comfy.utils
    import comfy.ldm.minimax.vae as minimax_vae
    pipe = ph3_pipeline(None, {'vae': '/vae.safetensors'})
    monkeypatch.setattr(comfy.utils, 'load_torch_file', lambda path: {})

    class FakeVAE:
        def __init__(self, sd, dtype):
            self.first_stage_model = model

        def throw_exception_if_invalid(self):
            pass

    monkeypatch.setattr(comfy.sd, 'VAE', FakeVAE)
    model = object.__new__(minimax_vae.MiniMaxH3VideoVAE)
    with pytest.raises(ValueError, match='not the P-H3-Image VAE'):
        pipe.load_vae(torch.float32)
    model = object.__new__(minimax_vae.PH3ImageVAE)
    assert pipe.load_vae(torch.float32).first_stage_model is model


# The text encoder: a Qwen3-VL-4B with 36 tiny layers (the connector reads hidden states 13 to 36).

@pytest.fixture
def tiny_clip(monkeypatch):
    import models.p_h3_image  # noqa: F401  (attention masks on)
    import comfy.sd
    import comfy.supported_models_base
    import comfy.text_encoders.llama as llama
    import comfy.text_encoders.minimax as minimax
    import comfy.text_encoders.qwen3vl as qwen3vl

    @dataclasses.dataclass
    class Tiny(llama.Qwen3VL_4BConfig):
        hidden_size: int = 32
        intermediate_size: int = 48
        num_hidden_layers: int = 36
        num_attention_heads: int = 2
        num_key_value_heads: int = 1
        head_dim = 16

    monkeypatch.setitem(qwen3vl.QWEN3VL_CONFIGS, 'qwen3vl_4b', Tiny)
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION, 'qwen3vl_4b', dict(hidden_size=32, intermediate_size=48, depth=2, deepstack_visual_indexes=[0, 1]))
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION_COMMON, 'num_heads', 2)
    target = comfy.supported_models_base.ClipTarget(minimax.PH3ImageTokenizer, minimax.te(te_model=minimax.PH3ImageTEModel))
    clip = comfy.sd.CLIP(target, model_options={'dtype': torch.float32}, disable_dynamic=True)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in clip.cond_stage_model.state_dict().items():
            if p.is_floating_point():
                if 'std' in name or 'norm' in name:
                    p.copy_(1 + 0.1 * torch.rand(p.shape, generator=g))
                else:
                    p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return clip


def _node_namespace():
    import os
    import comfy.model_management
    import comfy.nested_tensor
    import comfy.utils
    import comfy.ldm.minimax.model as minimax_model
    path = os.path.join(os.path.dirname(minimax_model.__file__), '..', '..', '..', 'comfy_extras', 'nodes_minimax_h3.py')
    tree = ast.parse(open(path).read())
    wanted = {'_resize', '_size_within_area', '_resize_to', '_av_latent'}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and (t.id.startswith('PH3_') or t.id == 'CANVAS_MULTIPLE') for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'PH3ImageConditioning':
            execute = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
            execute.decorator_list = []
            execute.returns = None
            body.append(execute)
    module = ast.Module(body=body, type_ignores=[])
    namespace = {
        'math': math, 're': re, 'torch': torch, 'comfy': comfy,
        'node_helpers': types.SimpleNamespace(conditioning_set_values=lambda cond, values: [[c[0], {**c[1], **values}] for c in cond]),
        'io': types.SimpleNamespace(NodeOutput=lambda *outputs: outputs),
    }
    exec(compile(module, path, 'exec'), namespace)
    assert {'execute', '_size_within_area', '_resize_to'} <= set(namespace)
    return namespace


def write_image(path, width, height, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(path)
    return str(path)


def load_comfy_image(path):
    with Image.open(path) as img:
        return torch.from_numpy(np.array(img.convert('RGB')).astype(np.float32) / 255.0)[None]


class NodeVAE:
    def encode(self, image):
        return torch.zeros(1, 24, 1, image.shape[1] // 16, image.shape[2] // 16)


def te_pipeline(clip, edit_prompt_prefix=True):
    pipe = ph3_pipeline(None)
    pipe.text_encoders = [clip]
    pipe.edit_prompt_prefix = edit_prompt_prefix
    return pipe


def test_size_within_area_matches_comfyui():
    from models.p_h3_image import size_within_area
    ns = _node_namespace()
    for size in [(4000, 3000), (1000, 1000), (1001, 999), (640, 2400), (31, 20), (5000, 32)]:
        assert size_within_area(*size, 1000000) == ns['_size_within_area'](*size, 1000000)


@pytest.mark.parametrize('caption, prefixed', [
    ('make the sky red', True),
    ('Edit <Picture 1>. make the sky red', False),
    ('  subject_definitions: a cat', False),
])
def test_edit_captions_are_encoded_as_comfyui_encodes_them(tiny_clip, tmp_path, caption, prefixed):
    """Our text encoder function against the PH3ImageConditioning node: the caption with
    "Edit <Picture 1>. " in front unless it names the picture, the control image at <= 1 MP, the
    embeddings and the token tags of its vision block."""
    ns = _node_namespace()
    pipe = te_pipeline(tiny_clip)
    fn = pipe.get_call_text_encoder_fn(tiny_clip)
    paths = [write_image(tmp_path / 'a.png', 1280, 960, 0), write_image(tmp_path / 'b.png', 200, 320, 1)]
    result = fn([caption, caption], [False, False], paths)
    for i, path in enumerate(paths):
        (cond, latent) = ns['execute'](None, tiny_clip, caption, 0, 0, vae=NodeVAE(), images={'image_1': load_comfy_image(path)})
        expected, extra = cond[0]
        assert_close(result['text_embeds_0'][i], expected[0])
        assert torch.equal(result['token_tags_0'][i], extra['minimax_token_tags'].long())
        assert result['attention_mask_0'][i].all() and result['attention_mask_0'][i].shape[0] == expected.shape[1]
        assert (result['token_tags_0'][i] == 0).any()
    # the larger image is shown at 1 MP, the smaller one as it is: different vision blocks
    assert result['token_tags_0'][0].shape != result['token_tags_0'][1].shape

    if prefixed:
        unprefixed = te_pipeline(tiny_clip, edit_prompt_prefix=False).get_call_text_encoder_fn(tiny_clip)(
            [caption], [False], paths[:1])
        assert unprefixed['text_embeds_0'][0].shape[0] < result['text_embeds_0'][0].shape[0]


def test_text_to_image_captions_are_batched(tiny_clip):
    """Without control images the captions are encoded together, padded; each row's real tokens
    are the caption encoded alone (as the node encodes it), its tags all 1."""
    ns = _node_namespace()
    pipe = te_pipeline(tiny_clip)
    fn = pipe.get_call_text_encoder_fn(tiny_clip)
    captions = ['a red fox in the snow at dawn', 'a cat']
    result = fn(captions, [False, False], None)
    assert fn(captions, [False, False], [None, None])['text_embeds_0'].shape == result['text_embeds_0'].shape
    lengths = result['attention_mask_0'].sum(1).tolist()
    assert lengths[0] > lengths[1] and result['token_tags_0'].eq(1).all()
    for i, caption in enumerate(captions):
        (cond, _) = ns['execute'](None, tiny_clip, caption, 0, 0)
        expected, _ = cond[0]
        assert expected.shape[1] == lengths[i]
        assert_close(result['text_embeds_0'][i, :lengths[i]], expected[0], atol=1e-4)


def test_another_text_encoder_is_refused(tiny_clip, monkeypatch):
    pipe = te_pipeline(tiny_clip)
    fn = pipe.get_call_text_encoder_fn(tiny_clip)
    monkeypatch.setattr(tiny_clip.cond_stage_model, 'clip', 'qwen3vl_32b')
    with pytest.raises(ValueError, match='not the P-H3-Image one'):
        fn(['a cat'], [False], None)


# A training step through DeepSpeed's pipeline engine in real gloo process groups.

def _step_inputs():
    lengths, vision = [9, 6], [(2, 4), (1, 2)]
    video, audio, valid, t, text, mask, tags, _, augmented = edit_inputs(lengths, vision, [0.3, 0.0005], seed=3)
    target = torch.randn(video.shape, generator=torch.Generator().manual_seed(5))
    return (video, audio, valid, t, text, mask, tags, augmented), (target, None)


def _trained_weights(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


LR = 0.1


def _expected_weights():
    model, _ = tiny_pair()
    pipe = ph3_pipeline(model)
    layers = pipe.to_layers()
    inputs, label = _step_inputs()
    loss = pipe.get_loss_fn()(run_layers(layers, inputs), label)
    loss.backward()
    return {name: (p - LR * p.grad if p.grad is not None else p).detach() for name, p in model.named_parameters()}


def _sdpa_everywhere():
    # What the pytorch_attention fixture does, in a spawned process.
    import sys
    import comfy.ldm.modules.attention as attention
    for module in list(sys.modules.values()):
        for name in ('optimized_attention', 'optimized_attention_masked'):
            if getattr(module, name, None) is attention.attention_sub_quad:
                setattr(module, name, attention.attention_pytorch)


def _engine_worker(rank, world_size, num_stages, init_file, out_dir):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    comfy.model_management.in_training = True
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world_size, verbose=False)
    from utils.patches import apply_patches
    from utils.pipeline import ManualPipelineModule
    apply_patches()
    model, _ = tiny_pair()
    _sdpa_everywhere()
    pipe = ph3_pipeline(model)
    layers = pipe.to_layers()
    module = ManualPipelineModule(layers=layers, num_stages=num_stages, partition_method='uniform',
                                  manual_partition_split=[len(layers) / num_stages], loss_fn=pipe.get_loss_fn(), dynamic_shape=True)
    params = [p for p in module.parameters() if p.requires_grad]
    engine, *_ = deepspeed.initialize(
        model=module, model_parameters=params, optimizer=torch.optim.SGD(params, lr=LR),
        config={'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1,
                'gradient_clipping': 0.0, 'steps_per_print': 10**9})
    engine._support_torch_style_backward = True
    data_parallel = world_size // num_stages
    # Through split_batch as the dataset hands batches to the engine: the missing mask becomes an
    # empty tensor.
    from utils.dataset import split_batch
    engine.train_batch(iter(list(split_batch(_step_inputs(), 1))))
    if engine.grid.get_data_parallel_rank() == 0:
        weights = {}
        for name, p in module.named_parameters():
            weights[name] = p.detach().tolist()
        with open(f'{out_dir}/{rank}.json', 'w') as f:
            json.dump({'weights': weights, 'data_parallel': data_parallel}, f)


@pytest.mark.parametrize('world_size, num_stages', [(2, 2), (2, 1)], ids=['pp2', 'dp2'])
def test_a_training_step_through_the_pipeline_engine(world_size, num_stages, tmp_path):
    """The P-H3-Image layers split over pipeline stages pass their tensors (the per-sample adaLN
    segments, the token tags, the reference latents) from stage to stage and step as one
    process does; with data parallelism every rank steps on the same gradient."""
    mp.spawn(_engine_worker, args=(world_size, num_stages, str(tmp_path / 'init'), str(tmp_path)), nprocs=world_size)
    expected = _expected_weights()
    # ManualPipelineModule names a parameter by its layer's index: map the model's names onto them.
    model, _ = tiny_pair()
    layers = ph3_pipeline(model).to_layers()
    names = {}
    for i, layer in enumerate(layers):
        for name, p in layer.named_parameters():
            for model_name, q in model.named_parameters():
                if q is p:
                    names[f'{i}.{name}'] = model_name
    trained = {}
    for path in tmp_path.glob('*.json'):
        trained.update(json.loads(path.read_text())['weights'])
    assert trained
    covered = set()
    for name, value in trained.items():
        model_name = names[name]
        covered.add(model_name)
        torch.testing.assert_close(torch.tensor(value), expected[model_name], atol=2e-5, rtol=1e-4, msg=model_name)
    assert covered == set(expected), sorted(set(expected) - covered)[:5]


def test_the_example_configs_are_accepted(monkeypatch):
    import toml
    import models.minimax_h3
    from models.p_h3_image import PH3ImagePipeline

    def base_init(self, config):
        self.config, self.model_config = config, config['model']

    monkeypatch.setattr(models.minimax_h3.MinimaxH3Pipeline, '__init__', base_init)
    pipe = PH3ImagePipeline(toml.load(EXAMPLES / 'p_h3_image.toml'))
    assert pipe.edit_prompt_prefix is True
    pipe.model_specific_dataset_config_validation(toml.load(EXAMPLES / 'p_h3_image_dataset.toml'))


def test_edit_and_text_to_image_directories_never_share_a_batch(monkeypatch):
    """Directories with and without control images in one size bucket: a batch is collated by the
    keys of its first example, so a mixed one lost control_latents or failed on them."""
    import utils.dataset as dataset_module

    class SizeBucket:
        def __init__(self, size_bucket, name):
            self.size_bucket, self.name = size_bucket, name

    class Directory:
        def __init__(self, name, control_path, size_buckets):
            self.control_path = control_path
            self.buckets = [SizeBucket(s, name) for s in size_buckets]

        def get_size_bucket_datasets(self):
            return self.buckets

    class Bucket:
        def __init__(self, datasets):
            self.names = sorted(d.name for d in datasets)

        def post_init(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

    monkeypatch.setattr(dataset_module, 'ConcatenatedBatchedDataset', Bucket)
    ds = object.__new__(dataset_module.Dataset)
    ds.batch_fill = None
    ds.dataset_config = {}
    ds._report_batch_fill = lambda: None
    ds.directory_datasets = [
        Directory('t2i', None, [(1.0, 512, 512, 1), (1.5, 608, 416, 1)]),
        Directory('edit', '/control', [(1.0, 512, 512, 1)]),
        Directory('t2i_2', None, [(1.0, 512, 512, 1)]),
    ]
    ds.post_init(0, 1, {None: 1}, 1, {None: 1})
    assert sorted(b.names for b in ds.buckets) == [['edit'], ['t2i'], ['t2i', 't2i_2']]


def test_the_vision_image_is_decoded_as_the_reference_latent_is(tmp_path):
    """The control image reaches the text encoder decoded as the dataset decodes it into the
    reference latent: transparency onto white, no EXIF rotation (a phone photo shown rotated to
    the encoder but not in the latent would teach the edit on a mismatched pair)."""
    from models.base import convert_crop_and_resize
    from models.p_h3_image import load_vision_image
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, (64, 96, 4), dtype=np.uint8)
    rgba[:16, :, 3] = 0
    Image.fromarray(rgba, 'RGBA').save(tmp_path / 'alpha.png')
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    Image.fromarray(rgba[..., :3]).save(tmp_path / 'rotated.jpg', exif=exif, quality=95)
    for name in ('alpha.png', 'rotated.jpg'):
        with Image.open(tmp_path / name) as img:
            expected = np.array(convert_crop_and_resize(img, img.size)).astype(np.float32) / 255.0
        vision = load_vision_image(tmp_path / name)
        assert vision.shape == (1, 64, 96, 3), name
        assert torch.equal(vision[0], torch.from_numpy(expected)), name
    assert load_vision_image(tmp_path / 'alpha.png')[0, :16].eq(1).all()
