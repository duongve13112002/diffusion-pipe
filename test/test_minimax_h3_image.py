"""MiniMax-H3-Image (models/minimax_h3_image.py, utils/minimax_h3_weights.py) on CPU.

The three weight layouts (comfy, official, diffusers) loaded back to ComfyUI's, the layers against ComfyUI's own
MiniMaxH3ImageModel forward (text to image and editing, every PDD head), the head split and the export that stacks
the heads again, the text encoder function against ComfyUI's MiniMaxH3ImageConditioning node on a tiny Qwen3-VL-32B,
the inputs, the loss, the config refusals, and a training step through DeepSpeed's pipeline engine in gloo process
groups.
"""

import dataclasses
import json
import re
import types
from pathlib import Path

import numpy as np
import pytest
import safetensors
import safetensors.torch
import torch
import torch.multiprocessing as mp
from PIL import Image

import comfy.ops

from test.test_comfy_equivalence import (  # noqa: F401  (training_path and pytorch_attention are autouse fixtures)
    MINIMAX_CONFIG, assert_close, bare_pipeline, pristine_minimax_module, randomize, run_layers,
    training_path, pytorch_attention,
)
from test.test_p_h3_image import (
    NodeVAE, _cpu_to, _node_namespace, _sdpa_everywhere, comfy_augmented, load_comfy_image, tags_with_vision, write_image,
)

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
IMAGE_CONFIG = {k: v for k, v in MINIMAX_CONFIG.items() if k != 'audio_latents_dim'}
SIGMAS = {1: [1.0, 0.0], 3: [1.0, 0.6, 0.3, 0.0]}
LATENT_H, LATENT_W = 4, 6
ROPE_THETA, ROPE_FREQ_DIM = 100.0, 4  # the tiny model's rope.inv_freq (see tiny_pair)


def tiny_pair(steps=1):
    """(the model with the pipeline's patched classes, ComfyUI's own with the same weights)"""
    import models.minimax_h3_image  # noqa: F401  (installs the patches)
    import comfy.ldm.minimax.model
    args = dict(pdd_sigmas=SIGMAS[steps], pdd_steps=steps, **IMAGE_CONFIG, dtype=torch.float32, operations=comfy.ops.manual_cast)
    model = comfy.ldm.minimax.model.MiniMaxH3ImageModel(**args)
    randomize(model)
    with torch.no_grad():
        model.rope.inv_freq.copy_(1.0 / ROPE_THETA ** (torch.arange(ROPE_FREQ_DIM, dtype=torch.float32) / ROPE_FREQ_DIM))
    reference = pristine_minimax_module().MiniMaxH3ImageModel(**args)
    reference.load_state_dict(model.state_dict())
    return model, reference


def image_pipeline(model, model_config=None, config=None):
    from models import minimax_h3_image
    import comfy.latent_formats
    pipe = bare_pipeline(minimax_h3_image.MiniMaxH3ImagePipeline, model, model_config)
    pipe.config.update(config or {})
    pipe.latent_format = comfy.latent_formats.MiniMaxH3Video()
    pipe.pdd_offsets = None
    pipe.is_video_vae = False
    return pipe


def name_parameters(model):
    for name, p in model.named_parameters():
        p.original_name = name


def reference_forward(reference, video, t, text, tags=None, ref=None, sample_sigmas=None):
    payload = {'seed': 0}
    if tags is not None:
        payload['text_token_tags'] = tags
    if ref is not None:
        payload['refs'] = [{'kind': 'image', 'latent_h': ref.shape[-2], 'latent_w': ref.shape[-1]}]
        payload['cond_video_latents'] = [ref]
    options = {} if sample_sigmas is None else {'sample_sigmas': torch.tensor(sample_sigmas)}
    return reference(video, t * 1000, text, transformer_options=options, minimax_payload=payload)


def image_inputs(lengths, vision, t, seed=1):
    """Batch of samples: captions of `lengths` tokens whose vision block is vision[i] = (start, length), padded to the
    longest; reference latents clean and noise-augmented as prepare_inputs makes them."""
    g = torch.Generator().manual_seed(seed)
    bs = len(lengths)
    video = torch.randn(bs, 24, 1, LATENT_H, LATENT_W, generator=g)
    text = torch.randn(bs, max(lengths), 40, generator=g)
    mask = torch.arange(text.shape[1])[None] < torch.tensor(lengths)[:, None]
    text[~mask] = 0
    tags = torch.ones(bs, max(lengths), dtype=torch.long)
    for i, (n, (start, length)) in enumerate(zip(lengths, vision)):
        tags[i, :n] = tags_with_vision(n, start, length)
    ref = torch.randn(bs, 24, 1, LATENT_H, LATENT_W, generator=g)
    augmented = torch.cat([comfy_augmented(ref[i:i + 1]) for i in range(bs)])
    return video, torch.tensor(t), text, mask, tags, ref, augmented


def expected_outputs(reference, video, t, text, mask, tags, ref, sample_sigmas=None):
    outputs = []
    for i in range(video.shape[0]):
        n = int(mask[i].sum())
        outputs.append(reference_forward(reference, video[i:i + 1], t[i:i + 1], text[i:i + 1, :n],
                                         None if tags is None else tags[i, :n], None if ref is None else ref[i:i + 1], sample_sigmas))
    return torch.cat(outputs)


def test_text_to_image_layers_match_comfyui():
    """Padded captions of different lengths, every token text: each sample as ComfyUI runs it alone."""
    model, reference = tiny_pair()
    layers = image_pipeline(model).to_layers()
    lengths = [6, 3]
    video, t, text, mask, tags, _, _ = image_inputs(lengths, [(0, 0), (0, 0)], [0.3, 0.8])
    actual = run_layers(layers, (video, t, text, mask, tags))
    assert_close(actual, expected_outputs(reference, video, t, text, mask, None, None))


@pytest.mark.parametrize('t', [[0.3, 0.7], [0.3, 0.0005]], ids=['ordinary', 'sigma-below-ref-pin'])
def test_edit_layers_match_comfyui(t):
    """A reference image and the token tags of its vision block, in a batch whose samples have different caption
    lengths and vision blocks of different lengths (so different adaLN segments per sample). At sigma < 0.001 the
    image's timestep passes the reference's pin of 0.999 and ComfyUI merges their timestep rows; the layers keep a row
    for each with the same values."""
    model, reference = tiny_pair()
    layers = image_pipeline(model).to_layers()
    video, tt, text, mask, tags, ref, augmented = image_inputs([9, 6], [(2, 4), (1, 2)], t)
    inputs = (video, tt, text, mask, tags, augmented)
    assert layers[0](inputs)[3].ndim == 3, 'the samples have different segments: one list per sample'
    actual = run_layers(layers, inputs)
    assert_close(actual, expected_outputs(reference, video, tt, text, mask, tags, ref))

    model.zero_grad(set_to_none=True)
    run_layers(layers, inputs).square().mean().backward()
    for name, p in model.blocks.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.final_layer.video_out.weight.grad is not None


def test_same_segments_stay_shared():
    model, _ = tiny_pair()
    layers = image_pipeline(model).to_layers()
    video, t, text, mask, tags, _, augmented = image_inputs([7, 5], [(1, 3), (1, 3)], [0.2, 0.0005])
    assert layers[0]((video, t, text, mask, tags, augmented))[3].ndim == 2


def test_the_tags_change_the_output():
    model, _ = tiny_pair()
    layers = image_pipeline(model).to_layers()
    video, t, text, mask, tags, _, augmented = image_inputs([7], [(1, 3)], [0.4])
    with_tags = run_layers(layers, (video, t, text, mask, tags, augmented))
    without = run_layers(layers, (video, t, text, mask, torch.ones_like(tags), augmented))
    assert not torch.allclose(with_tags, without)


def test_the_reference_changes_the_output():
    model, _ = tiny_pair()
    layers = image_pipeline(model).to_layers()
    video, t, text, mask, tags, _, augmented = image_inputs([7], [(1, 3)], [0.4])
    with_ref = run_layers(layers, (video, t, text, mask, tags, augmented))
    without = run_layers(layers, (video, t, text, mask, tags))
    assert not torch.allclose(with_ref, without)


def full_heads(model):
    """The PDD heads of a model with the bank baked into final_layer.video_out: [n, out, in], [n, out]."""
    out = model.final_layer.video_out
    n = model.final_layer.pdd_steps
    rows, brows = out.weight.detach().reshape(n, -1, out.weight.shape[1]), out.bias.detach().reshape(n, -1)
    return torch.cat([rows[:1], rows[1:] + rows[:1]]), torch.cat([brows[:1], brows[1:] + brows[:1]])


def test_the_head_split_keeps_every_head():
    """Head 0 trains; the later heads are head 0 plus a frozen offset the previews add: every step of the grid
    gives ComfyUI's output for that step."""
    model, reference = tiny_pair(3)
    heads_before = full_heads(model)
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    out = model.final_layer.video_out
    assert out.weight.shape == (96, 64) and out.bias.shape == (96,) and out.out_features == 96
    assert out.weight.requires_grad and out.bias.requires_grad
    assert not model.final_layer.pdd_offset_weight.requires_grad
    assert model.final_layer.pdd_offset_weight.shape == (2, 96, 64)
    assert 'final_layer.pdd_offset_weight' not in dict(model.named_parameters())
    assert_close(out.weight, heads_before[0][0])

    layers = pipe.to_layers()
    video, t, text, mask, tags, ref, augmented = image_inputs([8, 5], [(1, 3), (1, 2)], [0.5, 0.5])
    outputs = []
    for head in range(3):
        pipe.final_layer_module.head = head
        t_head = torch.full((2,), SIGMAS[3][head])
        actual = run_layers(layers, (video, t_head, text, mask, tags, augmented))
        assert_close(actual, expected_outputs(reference, video, t_head, text, mask, tags, ref, sample_sigmas=SIGMAS[3]), atol=2e-5)
        outputs.append(actual)
    # the heads differ, or the loop above proves nothing
    assert not torch.allclose(outputs[0], outputs[1]) and not torch.allclose(outputs[1], outputs[2])


def test_a_single_head_checkpoint_is_not_split():
    model, _ = tiny_pair(1)
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    assert pipe.pdd_offsets is None and not hasattr(model.final_layer, 'pdd_offset_weight')
    assert model.final_layer.video_out.weight.shape == (96, 64)


def test_export_stacks_the_heads_again(tmp_path):
    """save_model writes ComfyUI's single file: the trained head 0 with the frozen offsets under it, the buffers
    ComfyUI reads, the sigma grid in the metadata. Only head 0 moves with training."""
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(3)
    bank = {k: v.clone() for k, v in model.state_dict().items()}
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    name_parameters(model)
    with torch.no_grad():
        model.final_layer.video_out.weight.add_(0.5)
    sd = {p.original_name: p.detach() for p in model.parameters()}
    assert not any('pdd_offset' in k for k in sd)
    pipe.save_model(tmp_path, sd)

    saved = safetensors.torch.load_file(tmp_path / 'model.safetensors')
    with safetensors.safe_open(tmp_path / 'model.safetensors', framework='pt') as f:
        metadata = f.metadata()
    assert json.loads(metadata['config'])['transformer'] == {'pdd_steps': 3, 'pdd_sigmas': SIGMAS[3]}
    assert set(saved) == set(bank), set(saved) ^ set(bank)
    torch.testing.assert_close(saved['rope.inv_freq'], bank['rope.inv_freq'])
    weight = saved['final_layer.video_out.weight']
    assert weight.shape == (288, 64) and weight.dtype == torch.float32
    assert_close(weight[:96], bank['final_layer.video_out.weight'][:96] + 0.5)
    assert_close(weight[96:], bank['final_layer.video_out.weight'][96:])
    assert_close(saved['final_layer.video_out.bias'], bank['final_layer.video_out.bias'])
    assert_close(saved['blocks.0.attn.qkv_proj.weight'], bank['blocks.0.attn.qkv_proj.weight'])
    # and it loads back as a comfy-layout file
    loaded, loaded_metadata, layout, steps = weights.load_checkpoint(str(tmp_path / 'model.safetensors'))
    assert (layout, steps) == ('comfy', 3) and json.loads(loaded_metadata['config'])['transformer']['pdd_sigmas'] == SIGMAS[3]
    assert set(loaded) == set(saved)


def test_export_of_a_single_head_checkpoint(tmp_path):
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(1)
    bank = {k: v.clone() for k, v in model.state_dict().items()}
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    name_parameters(model)
    pipe.save_model(tmp_path, {p.original_name: p.detach() for p in model.parameters()})
    saved = safetensors.torch.load_file(tmp_path / 'model.safetensors')
    assert set(saved) == set(bank)
    _, _, layout, steps = weights.load_checkpoint(str(tmp_path / 'model.safetensors'))
    assert (layout, steps) == ('comfy', 1)


# Weight layouts, written the way MiniMaxAI published them from the ComfyUI layout, by hand.

def interleaved(qkv, heads=2, head_dim=32):
    q, k, v = qkv.chunk(3, dim=0)
    return torch.stack([part.reshape(heads, head_dim, -1) for part in (q, k, v)], dim=1).reshape(3 * heads * head_dim, -1)


def save_shards(directory, sd, name='model', shards=2):
    directory.mkdir(parents=True, exist_ok=True)
    keys = sorted(sd)
    weight_map = {}
    for i in range(shards):
        file = f'{name}-0000{i + 1}-of-0000{shards}.safetensors'
        part = {k: sd[k].contiguous() for k in keys[i::shards]}
        safetensors.torch.save_file(part, directory / file)
        weight_map.update({k: file for k in part})
    (directory / f'{name}.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}))


def split_bank(sd, steps):
    weight = sd['final_layer.video_out.weight'].reshape(steps, 96, 64)
    bias = sd['final_layer.video_out.bias'].reshape(steps, 96)
    return torch.cat([weight[:1], weight[1:] + weight[:1]]), torch.cat([bias[:1], bias[1:] + bias[:1]])


def write_comfy(path, model, steps):
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    metadata = {'format': 'pt', 'config': json.dumps({'transformer': {'pdd_steps': steps, 'pdd_sigmas': SIGMAS[steps]}})}
    safetensors.torch.save_file(sd, path / 'comfy.safetensors', metadata=metadata)
    return sd


def write_official(root, model, steps, heads=True):
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    weight, bias = split_bank(sd, steps)
    official = {}
    for key, tensor in sd.items():
        official[key] = interleaved(tensor) if key.endswith('.attn.qkv_proj.weight') else tensor
    official['final_layer.video_out.weight'], official['final_layer.video_out.bias'] = weight[0].clone(), bias[0].clone()
    save_shards(root / 'transformer', official)
    (root / 'transformer' / 'config.json').write_text(json.dumps({'_class_name': 'MiniMaxH3DiTModel'}))
    if heads:
        safetensors.torch.save_file({'video_out.weight': weight.contiguous(), 'video_out.bias': bias.contiguous()}, root / 'pdd_fused_heads.safetensors')
        (root / 'pdd_config.json').write_text(json.dumps({'sigmas': SIGMAS[steps]}))
    return sd


DIFFUSERS_NAMES = [
    (r'^blocks\.(\d+)\.', r'transformer_blocks.\1.'),
    (r'^token_refiner\.blocks\.(\d+)\.', r'token_refiner.refiner_blocks.\1.'),
    (r'\.attn\.q_norm\.', '.attn.norm_q.'),
    (r'\.attn\.k_norm\.', '.attn.norm_k.'),
    (r'\.attn\.out_proj\.', '.attn.to_out.0.'),
    (r'\.mlp\.fc2\.', '.ff.net.2.'),
    (r'^time_embedder\.proj_in\.', 'time_embedder.linear_1.'),
    (r'^time_embedder\.proj_out\.', 'time_embedder.linear_2.'),
    (r'^video_patch_proj\.', 'proj_in.'),
    (r'^condition_proj\.', 'context_embedder.'),
    (r'^final_layer\.norm\.', 'norm_out.norm.'),
    (r'^final_layer\.adaln_proj\.linear\.', 'norm_out.linear.'),
]


def write_diffusers(root, model, steps, heads=True):
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    weight, bias = split_bank(sd, steps)
    out = {}
    for key, tensor in sd.items():
        if key in ('rope.inv_freq', 'final_layer.video_out.weight', 'final_layer.video_out.bias'):
            continue
        if key.endswith('.attn.qkv_proj.weight'):
            base = key[:-len('.attn.qkv_proj.weight')]
            for pattern, replacement in DIFFUSERS_NAMES[:2]:
                base = re.sub(pattern, replacement, base + '.').rstrip('.')
            for name, part in zip('qkv', tensor.chunk(3, dim=0)):
                out[f'{base}.attn.to_{name}.weight'] = part
            continue
        if key.endswith('.mlp.fc1.weight'):
            gate, value = tensor.chunk(2, dim=0)
            key, tensor = key.replace('.mlp.fc1.', '.ff.net.0.proj.'), torch.cat([value, gate])
        for pattern, replacement in DIFFUSERS_NAMES:
            key = re.sub(pattern, replacement, key)
        out[key] = tensor
    for i in range(steps):
        out[f'pdd_proj_out.{i}.weight'], out[f'pdd_proj_out.{i}.bias'] = weight[i].clone(), bias[i].clone()
    save_shards(root / 'transformer_diffusers', out, name='diffusion_pytorch_model')
    (root / 'transformer_diffusers' / 'config.json').write_text(json.dumps({'rope_theta': ROPE_THETA, 'rope_freq_dim': ROPE_FREQ_DIM}))
    if heads:
        (root / 'pdd_config.json').write_text(json.dumps({'sigmas': SIGMAS[steps]}))
    return sd


def assert_same_state(loaded, expected):
    assert set(loaded) == set(expected), (sorted(set(loaded) - set(expected))[:4], sorted(set(expected) - set(loaded))[:4])
    for key in expected:
        torch.testing.assert_close(loaded[key].float(), expected[key].float(), atol=1e-6, rtol=1e-5, msg=key)


@pytest.mark.parametrize('steps', [3, 1])
def test_the_comfy_layout_loads_as_it_is(tmp_path, steps):
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(steps)
    expected = write_comfy(tmp_path, model, steps)
    sd, metadata, layout, n = weights.load_checkpoint(str(tmp_path / 'comfy.safetensors'))
    assert (layout, n) == ('comfy', steps)
    assert_same_state(sd, expected)
    assert json.loads(metadata['config'])['transformer'] == {'pdd_steps': steps, 'pdd_sigmas': SIGMAS[steps]}
    assert weights.inspect_checkpoint(str(tmp_path / 'comfy.safetensors')) == ('comfy', steps)


@pytest.mark.parametrize('how', ['root', 'folder', 'index', 'forced'])
def test_the_official_layout_loads_as_comfys(tmp_path, how):
    """Interleaved qkv rows reordered, the six heads of pdd_fused_heads.safetensors stacked into the bank, the grid
    from pdd_config.json."""
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(3)
    expected = write_official(tmp_path, model, 3)
    path = {'root': tmp_path, 'folder': tmp_path / 'transformer',
            'index': tmp_path / 'transformer' / 'model.safetensors.index.json', 'forced': tmp_path / 'transformer'}[how]
    forced = 'official' if how == 'forced' else None
    sd, metadata, layout, steps = weights.load_checkpoint(str(path), forced)
    assert (layout, steps) == ('official', 3)
    assert_same_state(sd, expected)
    assert json.loads(metadata['config'])['transformer'] == {'pdd_steps': 3, 'pdd_sigmas': SIGMAS[3]}
    assert weights.inspect_checkpoint(str(path), forced) == ('official', 3)


def test_the_official_layout_without_the_heads_file_is_one_head(tmp_path):
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(3)
    expected = write_official(tmp_path, model, 3, heads=False)
    sd, metadata, layout, steps = weights.load_checkpoint(str(tmp_path / 'transformer'))
    assert (layout, steps) == ('official', 1)
    assert sd['final_layer.video_out.weight'].shape == (96, 64)
    assert_close(sd['final_layer.video_out.weight'], expected['final_layer.video_out.weight'][:96])
    assert weights.inspect_checkpoint(str(tmp_path / 'transformer')) == ('official', 1)
    # an explicit pdd_heads file stands in for the repository root's
    heads = tmp_path / 'elsewhere'
    heads.mkdir()
    write_official(heads, model, 3)
    (heads / 'transformer').rename(heads / 'unused')
    sd, _, _, steps = weights.load_checkpoint(str(tmp_path / 'transformer'), None, str(heads / 'pdd_fused_heads.safetensors'))
    assert steps == 3 and sd['final_layer.video_out.weight'].shape == (288, 64)
    assert_same_state({k: v for k, v in sd.items() if 'video_out' in k}, {k: v for k, v in expected.items() if 'video_out' in k})


@pytest.mark.parametrize('how', ['root', 'folder'])
def test_the_diffusers_layout_loads_as_comfys(tmp_path, how):
    """to_q / to_k / to_v concatenated, fc1 halves swapped back, the renames undone, the full heads stacked into the
    bank, rope.inv_freq recomputed from the config."""
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(3)
    expected = write_diffusers(tmp_path, model, 3)
    path = tmp_path if how == 'root' else tmp_path / 'transformer_diffusers'
    sd, metadata, layout, steps = weights.load_checkpoint(str(path))
    assert (layout, steps) == ('diffusers', 3)
    assert_same_state(sd, expected)
    assert json.loads(metadata['config'])['transformer'] == {'pdd_steps': 3, 'pdd_sigmas': SIGMAS[3]}
    assert weights.inspect_checkpoint(str(path)) == ('diffusers', 3)


def test_the_diffusers_fixture_uses_the_published_names(tmp_path):
    """Guard for the fixture itself: the qkv parts are named as diffusers names them (transformer_blocks.N,
    token_refiner.refiner_blocks.N), so the loader's renaming of the concatenated key is exercised."""
    model, _ = tiny_pair(1)
    write_diffusers(tmp_path, model, 1)
    with safetensors.safe_open(next((tmp_path / 'transformer_diffusers').glob('*0001*.safetensors')), framework='pt') as f:
        keys = set(f.keys())
    with safetensors.safe_open(next((tmp_path / 'transformer_diffusers').glob('*0002*.safetensors')), framework='pt') as f:
        keys |= set(f.keys())
    assert 'transformer_blocks.0.attn.to_q.weight' in keys and 'token_refiner.refiner_blocks.0.attn.to_k.weight' in keys
    assert not any(k.startswith(('blocks.', 'token_refiner.blocks.')) for k in keys)


def test_a_sharded_checkpoint_of_unknown_layout_is_not_guessed(tmp_path):
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(1)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    save_shards(tmp_path / 'shards', sd)
    with pytest.raises(ValueError, match="diffusion_model_format"):
        weights.load_checkpoint(str(tmp_path / 'shards'))
    loaded, _, layout, _ = weights.load_checkpoint(str(tmp_path / 'shards'), 'comfy')
    assert layout == 'comfy'
    assert_same_state(loaded, sd)


def test_the_file_metadata_is_kept(tmp_path):
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(1)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    config = {'transformer': {'other_key': 7}, 'unet': {'x': 1}}
    safetensors.torch.save_file(sd, tmp_path / 'm.safetensors', metadata={'config': json.dumps(config), 'note': 'kept'})
    _, metadata, _, _ = weights.load_checkpoint(str(tmp_path / 'm.safetensors'))
    merged = json.loads(metadata['config'])
    assert merged['transformer'] == {'other_key': 7, 'pdd_steps': 1, 'pdd_sigmas': [1.0, 0.0]} and merged['unet'] == {'x': 1}
    assert metadata['note'] == 'kept'


def test_a_diffusers_audio_transformer_is_refused(tmp_path):
    from utils import minimax_h3_weights as weights
    safetensors.torch.save_file({'transformer_blocks.0.norm1.weight': torch.zeros(2), 'proj_in.weight': torch.zeros(2, 2),
                                 'context_embedder.weight': torch.zeros(2, 2), 'audio_proj_in.weight': torch.zeros(2, 2)}, tmp_path / 'a.safetensors')
    with pytest.raises(ValueError, match='minimax_h3'):
        weights.inspect_checkpoint(str(tmp_path / 'a.safetensors'))


def test_a_diffusers_checkpoint_of_other_heads_needs_its_grid(tmp_path):
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair(3)
    write_diffusers(tmp_path, model, 3, heads=False)
    with pytest.raises(ValueError, match='pdd_config.json'):
        weights.load_checkpoint(str(tmp_path))


def test_the_loaded_layouts_run_to_the_same_output(tmp_path):
    """The three layouts of one model drive ComfyUI's forward to one result: the loaders' output builds the model."""
    from utils import minimax_h3_weights as weights
    import models.minimax_h3_image  # noqa: F401
    import comfy.ldm.minimax.model
    model, reference = tiny_pair(3)
    write_official(tmp_path / 'o', model, 3)
    write_diffusers(tmp_path / 'd', model, 3)
    video, t, text, mask, tags, ref, _ = image_inputs([6], [(1, 2)], [SIGMAS[3][1]])
    expected = expected_outputs(reference, video, t, text, mask, tags, ref, sample_sigmas=SIGMAS[3])
    for sub in ('o', 'd'):
        sd, metadata, _, steps = weights.load_checkpoint(str(tmp_path / sub))
        rebuilt = pristine_minimax_module().MiniMaxH3ImageModel(
            pdd_sigmas=SIGMAS[steps], pdd_steps=steps, **IMAGE_CONFIG, dtype=torch.float32, operations=comfy.ops.manual_cast)
        rebuilt.load_state_dict(sd)
        assert_close(expected_outputs(rebuilt, video, t, text, mask, tags, ref, sample_sigmas=SIGMAS[3]), expected, atol=2e-5)


def test_pdd_bank_is_what_comfyui_reads():
    from utils import minimax_h3_weights as weights
    comfy_model = pristine_minimax_module()
    g = torch.Generator().manual_seed(0)
    heads, biases = torch.randn(3, 5, 7, generator=g), torch.randn(3, 5, generator=g)
    bank_weight, bank_bias = weights.pdd_bank(heads, biases)
    assert bank_weight.shape == (15, 7) and bank_bias.shape == (15,)
    head = comfy.ops.manual_cast.Linear(7, 15)
    head.weight.data, head.bias.data = bank_weight, bank_bias
    h = torch.randn(4, 7, generator=g)
    for i in range(3):
        out = comfy_model._pdd_head(head, h, SIGMAS[3], i, i + 1)
        torch.testing.assert_close(out, torch.nn.functional.linear(h, heads[i], biases[i]), atol=1e-5, rtol=1e-5)
    unstacked = weights.unstack_pdd_bank(bank_weight, bank_bias, 3)
    torch.testing.assert_close(unstacked[0], heads, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(unstacked[1], biases, atol=1e-6, rtol=1e-6)


def test_interleaved_qkv_roundtrip():
    from utils import minimax_h3_weights as weights
    qkv = torch.randn(3 * 4 * 8, 5)
    q, k, v = qkv.chunk(3)
    inter = weights.interleave_qkv(q, k, v, 4, 8)
    assert torch.equal(inter, interleaved(qkv, 4, 8))
    assert torch.equal(weights.reorder_interleaved_qkv(inter, 4, 8), qkv)
    with pytest.raises(ValueError, match='rows'):
        weights.reorder_interleaved_qkv(qkv, 5, 8)


def test_rope_inv_freq_matches_the_model():
    from utils import minimax_h3_weights as weights
    model, _ = tiny_pair()
    torch.testing.assert_close(weights.rope_inv_freq(ROPE_THETA, ROPE_FREQ_DIM), model.rope.inv_freq, atol=1e-7, rtol=1e-6)


def test_an_audio_transformer_is_refused(tmp_path):
    from utils import minimax_h3_weights as weights
    safetensors.torch.save_file({'audio_patch_proj.weight': torch.zeros(2, 2), 'video_patch_proj.weight': torch.zeros(2, 2)}, tmp_path / 'm.safetensors')
    with pytest.raises(ValueError, match="minimax_h3"):
        weights.load_checkpoint(str(tmp_path / 'm.safetensors'))
    safetensors.torch.save_file({'x': torch.zeros(2, 2)}, tmp_path / 'other.safetensors')
    with pytest.raises(ValueError, match='not a MiniMax-H3-Image transformer'):
        weights.inspect_checkpoint(str(tmp_path / 'other.safetensors'))
    with pytest.raises(FileNotFoundError):
        weights.load_checkpoint(str(tmp_path / 'missing.safetensors'))


def test_the_pipeline_refuses_pdd_heads_outside_the_official_layout(tmp_path):
    model, _ = tiny_pair(3)
    write_comfy(tmp_path, model, 3)
    pipe = image_pipeline(None, {'diffusion_model': str(tmp_path / 'comfy.safetensors'), 'pdd_heads': 'h.safetensors'})
    with pytest.raises(ValueError, match='pdd_heads applies to the official'):
        pipe.refuse_unsupported_checkpoint()
    del pipe.model_config['pdd_heads']
    pipe.refuse_unsupported_checkpoint()
    root = tmp_path / 'repo'
    write_official(root, model, 3)
    image_pipeline(None, {'diffusion_model': str(root), 'pdd_heads': str(root / 'pdd_fused_heads.safetensors')}).refuse_unsupported_checkpoint()


def test_the_vae_of_diffusers_is_converted_to_comfys():
    from utils import minimax_h3_weights as weights
    g = torch.Generator().manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, generator=g)
    q, k, v = rand(128, 6), rand(128, 6), rand(128, 6)
    qb, kb, vb = rand(128), rand(128), rand(128)
    up, gate = rand(10, 6), rand(10, 6)
    sd = {
        'encoder.down_blocks.1.resnets.0.conv_shortcut.weight': rand(2),
        'encoder.down_blocks.1.resnets.0.conv1.weight': rand(2),
        'encoder.down_blocks.1.downsamplers.0.conv.weight': rand(2),
        'encoder.conv_in.weight': rand(2),
        'decoder.proj_in.weight': rand(2),
        'decoder.blocks.0.attn.to_q.weight': q, 'decoder.blocks.0.attn.to_k.weight': k, 'decoder.blocks.0.attn.to_v.weight': v,
        'decoder.blocks.0.attn.to_q.bias': qb, 'decoder.blocks.0.attn.to_k.bias': kb, 'decoder.blocks.0.attn.to_v.bias': vb,
        'decoder.blocks.0.attn.to_out.0.weight': rand(2),
        'decoder.blocks.0.ff.net.0.proj.weight': torch.cat([up, gate]),
        'decoder.blocks.0.ff.net.2.weight': rand(2),
    }
    assert weights.is_diffusers_vae(sd) and not weights.is_diffusers_vae({'encoder.down.0.block.0.conv1.weight': rand(1)})
    out = weights.diffusers_vae_to_comfy(sd)
    assert set(out) == {
        'encoder.down.1.block.0.nin_shortcut.weight', 'encoder.down.1.block.0.conv1.weight', 'encoder.down.1.downsample.conv.weight',
        'encoder.conv_in.weight', 'decoder.x_embedder.weight', 'decoder.blocks.0.attn.to_qkv.weight', 'decoder.blocks.0.attn.to_qkv.bias',
        'decoder.blocks.0.attn.to_out.weight', 'decoder.blocks.0.ff.w1.weight', 'decoder.blocks.0.ff.w2.weight'}
    qkv = out['decoder.blocks.0.attn.to_qkv.weight']
    assert qkv.shape == (384, 6)
    # two heads of 64: [h0 q, h0 k, h0 v, h1 q, h1 k, h1 v]
    for head in range(2):
        for j, part in enumerate((q, k, v)):
            assert torch.equal(qkv[(head * 3 + j) * 64:(head * 3 + j + 1) * 64], part[head * 64:(head + 1) * 64])
    assert torch.equal(out['decoder.blocks.0.attn.to_qkv.bias'][64:128], kb[:64])
    assert torch.equal(out['decoder.blocks.0.ff.w1.weight'], torch.cat([gate, up]))
    with pytest.raises(ValueError, match='lacks'):
        weights.diffusers_vae_to_comfy({'decoder.blocks.0.attn.to_q.weight': q})


def test_the_text_encoder_of_hugging_face_is_cut_and_renamed(tmp_path):
    from utils import minimax_h3_weights as weights
    layers = weights.TEXT_ENCODER_LAYERS
    sd = {
        'model.language_model.embed_tokens.weight': torch.zeros(3, 2),
        'model.language_model.norm.weight': torch.zeros(2),
        'lm_head.weight': torch.zeros(3, 2),
        'model.visual.blocks.0.attn.qkv.weight': torch.ones(2, 2),
    }
    for i in (0, layers - 1, layers, layers + 13):
        sd[f'model.language_model.layers.{i}.self_attn.q_proj.weight'] = torch.full((2, 2), float(i))
    shards = tmp_path / 'text_encoder'
    save_shards(shards, sd)
    result = weights.load_text_encoder_state([str(shards)])
    assert set(result) == {'model.embed_tokens.weight', 'visual.blocks.0.attn.qkv.weight',
                           'model.layers.0.self_attn.q_proj.weight', f'model.layers.{layers - 1}.self_attn.q_proj.weight'}
    assert result[f'model.layers.{layers - 1}.self_attn.q_proj.weight'][0, 0] == layers - 1
    # ComfyUI's own file goes to the standard loader
    safetensors.torch.save_file({'model.embed_tokens.weight': torch.zeros(3, 2)}, tmp_path / 'comfy_te.safetensors')
    assert weights.load_text_encoder_state([str(tmp_path / 'comfy_te.safetensors')]) is None
    # too few layers
    short = tmp_path / 'short'
    save_shards(short, {k: v for k, v in sd.items() if 'layers.' not in k or '.layers.0.' in k})
    with pytest.raises(ValueError, match='fewer than 50 layers'):
        weights.load_text_encoder_state([str(short)])


def test_prepare_inputs():
    torch.manual_seed(0)
    pipe = image_pipeline(None, {'timestep_sample_method': 'uniform'})
    bs = 3
    latents = torch.randn(bs, 24, 1, LATENT_H, LATENT_W)
    control = torch.randn(bs, 24, 1, LATENT_H, LATENT_W)
    lengths = (5, 7, 2)
    inputs = {'latents': latents, 'mask': None, 'control_latents': control,
              'text_embeds_0': [torch.randn(n, 40) for n in lengths],
              'attention_mask_0': [torch.ones(n, dtype=torch.int64) for n in lengths],
              'token_tags_0': [tags_with_vision(n, 1, 1) for n in lengths]}
    (noisy, t, text_embeds, attention_mask, token_tags, ref), (target, mask) = pipe.prepare_inputs(inputs)
    assert noisy.shape == latents.shape and target.shape == latents.shape and t.shape == (bs,)
    assert torch.allclose(noisy, (1 - t.view(-1, 1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1, 1) * (target + latents))
    assert text_embeds.shape == (bs, 7, 40) and attention_mask.dtype == torch.bool
    assert token_tags.tolist()[2] == [1, 0, 1, 1, 1, 1, 1], 'padding takes tag 1'
    assert (ref - 0.999 * control).abs().max() < 0.01 and not torch.equal(ref, control)
    assert mask is None

    with pytest.raises(ValueError, match='do not match'):
        pipe.prepare_inputs({**inputs, 'control_latents': control[..., :2]})
    with pytest.raises(ValueError, match='images only'):
        pipe.prepare_inputs({**inputs, 'latents': torch.randn(bs, 24, 2, LATENT_H, LATENT_W)})
    model_inputs, _ = pipe.prepare_inputs({k: v for k, v in inputs.items() if k != 'control_latents'})
    assert len(model_inputs) == 5
    # without token tags every token is text
    no_tags = {k: v for k, v in inputs.items() if k != 'token_tags_0'}
    (_, _, _, _, token_tags, _), _ = pipe.prepare_inputs(no_tags)
    assert token_tags.eq(1).all()


def test_the_loss():
    pipe = image_pipeline(None)
    loss_fn = pipe.get_loss_fn()
    output, target = torch.randn(2, 24, 1, 4, 6), torch.randn(2, 24, 1, 4, 6)
    assert loss_fn(output, (target, None)) == torch.nn.functional.mse_loss(output, target)
    assert loss_fn(output, (target, torch.empty(0))) == torch.nn.functional.mse_loss(output, target)
    mask = torch.zeros(2, 1, 1, 4, 6)
    mask[0] = 1
    assert_close(loss_fn(output, (target, mask)), torch.nn.functional.mse_loss(output[0], target[0]) / 2)


class StubVAE:
    """The VAE's encode as far as batching matters: a seed-42 posterior sample drawn for the shape it is given."""

    def __init__(self):
        self.batch_sizes = []

    def encode(self, pixels):
        self.batch_sizes.append(pixels.shape[0])
        b, h, w, _ = pixels.shape
        noise = torch.randn((b, 24, 1, h // 16, w // 16), generator=torch.Generator().manual_seed(42))
        return pixels.mean(dim=(1, 2, 3)).view(b, 1, 1, 1, 1) + noise


def test_images_are_encoded_one_at_a_time(monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'to', _cpu_to(torch.Tensor.to))
    pipe = image_pipeline(None, {'dtype': torch.float32})
    pipe.__dict__['_vae_dtypes'] = types.SimpleNamespace(encode=None)
    vae = StubVAE()
    images = torch.rand(3, 3, 64, 96)
    result = pipe.get_call_vae_fn(vae)(images, images.flip(0))
    assert vae.batch_sizes == [1] * 6
    alone = pipe.latent_format.process_in(StubVAE().encode(images[2:3].movedim(1, -1)))
    assert_close(result['latents'][2:3], alone)
    assert result['control_latents'].shape == (3, 24, 1, 4, 6)
    assert_close(pipe.encode_images(images[:, :, None], StubVAE()), result['latents'])
    assert 'control_latents' not in pipe.get_call_vae_fn(StubVAE())(images)


def test_the_vision_image_is_decoded_as_the_reference_latent_is(tmp_path):
    from models.minimax_h3_image import load_vision_image
    from models.base import convert_to_rgb
    rgba = Image.new('RGBA', (1200, 1100), (255, 0, 0, 0))
    rgba.putpixel((3, 3), (0, 0, 255, 255))
    path = tmp_path / 'a.png'
    rgba.save(path)
    image = load_vision_image(str(path))
    assert image.shape[0] == 1 and image.shape[3] == 3 and min(image.shape[1:3]) == 1024
    assert image.shape[1] % 32 == 0 and image.shape[2] % 32 == 0
    # transparency goes onto white as convert_to_rgb does it, not onto black
    assert image[0, 1000, 1000].min() > 0.9
    assert np.array(convert_to_rgb(Image.open(path)))[500, 500].min() == 255


def test_size_short_edge_matches_comfyui():
    from models.minimax_h3_image import size_short_edge
    ns = _node_namespace('MiniMaxH3ImageConditioning')
    for size in [(4000, 3000), (1000, 1000), (1001, 999), (640, 2400), (31, 20), (5000, 32), (3000, 800)]:
        for low, high in [(1024, 1024), (1024, 2048)]:
            assert size_short_edge(*size, low, high) == ns['_size_short_edge'](*size, low, high)


@pytest.mark.parametrize('key, value, match', [
    ('cfg', 2.0, 'cfg does not apply'),
    ('audio_vae', '/x.safetensors', 'audio_vae does not apply'),
])
def test_keys_that_do_not_apply_are_refused(key, value, match):
    from models.minimax_h3_image import MiniMaxH3ImagePipeline
    config = {'model': {key: value, 'text_encoders': [{'path': 'x', 'type': 'minimax'}]}}
    with pytest.raises(ValueError, match=match):
        MiniMaxH3ImagePipeline(config)


@pytest.mark.parametrize('text_encoders', [[], [{'path': 'x', 'type': 'qwen_image'}], [{'path': 'x', 'type': 'minimax'}] * 2])
def test_the_text_encoder_must_be_the_minimax_one(text_encoders):
    from models.minimax_h3_image import MiniMaxH3ImagePipeline
    with pytest.raises(ValueError, match='one text encoder'):
        MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders}})


def test_the_format_and_the_lr_keys_are_checked():
    from models.minimax_h3_image import MiniMaxH3ImagePipeline
    text_encoders = [{'path': 'x', 'type': 'minimax'}]
    with pytest.raises(ValueError, match='diffusion_model_format must be one of'):
        MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders, 'diffusion_model_format': 'gguf'}})
    with pytest.raises(ValueError, match='head_lr, text_projection_lr do not apply to adapter training'):
        MiniMaxH3ImagePipeline({'adapter': {'type': 'lora'}, 'model': {'text_encoders': text_encoders, 'head_lr': 1e-4, 'text_projection_lr': 1e-4}})


def test_the_default_shift_is_the_grid_of_the_checkpoint(monkeypatch):
    import models.minimax_h3
    from models.minimax_h3_image import MiniMaxH3ImagePipeline

    def base_init(self, config):
        self.config, self.model_config = config, config['model']

    monkeypatch.setattr(models.minimax_h3.MinimaxH3Pipeline, '__init__', base_init)
    text_encoders = [{'path': 'x', 'type': 'minimax'}]
    assert MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders}}).model_config['shift'] == 3.0
    assert 'shift' not in MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders, 'image_shift': 2.0}}).model_config
    assert MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders, 'shift': 1.0}}).model_config['shift'] == 1.0


@pytest.mark.parametrize('dataset, match', [
    ({'frame_buckets': [1, 33]}, 'images only'),
    ({'directory': [{'path': 'x', 'frame_buckets': [1, 33]}]}, r'images only.*\[\[directory\]\] x'),
    ({'directory': [{'path': 'x', 'control_path': 'c', 'control_path_2': 'd'}]}, 'control_path_N'),
    ({'directory': [{'path': 'x', 'control_path': 'c', 'caption_dropout_rate': 0.1}]}, 'caption_dropout_rate in it'),
    ({'caption_dropout_rate': 0.1, 'directory': [{'path': 'x', 'control_path': 'c'}]}, 'the dataset caption_dropout_rate'),
])
def test_dataset_refusals(dataset, match):
    pipe = image_pipeline(None)
    with pytest.raises(ValueError, match=match):
        pipe.model_specific_dataset_config_validation(dataset)
    pipe.model_specific_dataset_config_validation({'frame_buckets': [1], 'directory': [{'path': 'x', 'control_path': 'c'}]})
    pipe.model_specific_dataset_config_validation({'caption_dropout_rate': 0.1, 'directory': [
        {'path': 't2i'}, {'path': 'edit', 'control_path': 'c', 'caption_dropout_rate': 0}]})


def test_uncond_fraction_reaches_edit_directories_too():
    pipe = image_pipeline(None)
    pipe.config['uncond_fraction'] = 0.1
    with pytest.raises(ValueError, match='uncond_fraction of the training config'):
        pipe.model_specific_dataset_config_validation({'directory': [{'path': 'x', 'control_path': 'c'}]})
    pipe.model_specific_dataset_config_validation({'directory': [{'path': 'x', 'control_path': 'c', 'caption_dropout_rate': 0}]})
    pipe.model_specific_dataset_config_validation({'directory': [{'path': 't2i'}]})


def test_load_vae_takes_only_the_h3_image_vae(monkeypatch):
    import comfy.sd
    import comfy.utils
    import comfy.ldm.minimax.vae as minimax_vae
    pipe = image_pipeline(None, {'vae': '/vae.safetensors'})
    monkeypatch.setattr(comfy.utils, 'load_torch_file', lambda path: {})

    class FakeVAE:
        def __init__(self, sd, dtype):
            self.first_stage_model = model

        def throw_exception_if_invalid(self):
            pass

    monkeypatch.setattr(comfy.sd, 'VAE', FakeVAE)
    for other in (minimax_vae.MiniMaxH3VideoVAE, minimax_vae.PH3ImageVAE):
        model = object.__new__(other)
        with pytest.raises(ValueError, match='not the MiniMax-H3-Image VAE'):
            pipe.load_vae(torch.float32)
    model = object.__new__(minimax_vae.MiniMaxH3ImageVAE)
    assert pipe.load_vae(torch.float32).first_stage_model is model


def test_param_groups(monkeypatch):
    import models.minimax_h3_image
    monkeypatch.setattr(models.minimax_h3_image, 'is_main_process', lambda: False)
    model, _ = tiny_pair(3)
    pipe = image_pipeline(model, {'blocks_lr': 2e-5, 'head_lr': 0, 'text_projection_lr': 3e-5}, {'optimizer': {'lr': 1e-4}})
    pipe.split_pdd_head()
    name_parameters(model)
    parameters = list(model.parameters())
    groups = pipe.get_param_groups(parameters)
    by_lr = {}
    for g in groups:
        by_lr.setdefault(g['lr'], set()).update(p.original_name for p in g['params'])
    assert set(by_lr) == {2e-5, 3e-5, 1e-4}
    assert by_lr[3e-5] == {'condition_proj.weight', 'condition_proj.bias'}
    assert all(n.startswith('blocks.') for n in by_lr[2e-5])
    assert any('token_refiner' in n for n in by_lr[1e-4]) and any(n.startswith('time_embedder') for n in by_lr[1e-4])
    frozen = {p.original_name for p in parameters if not p.requires_grad}
    assert 'final_layer.video_out.weight' in frozen and 'final_layer.norm.weight' in frozen
    assert sum(len(g['params']) for g in groups) + len(frozen) == len(parameters)

    plain = image_pipeline(model, None, {'optimizer': {'lr': 1e-4}})
    assert plain.get_param_groups(parameters) == [{'params': parameters}]


# The text encoder: a Qwen3-VL-32B with 3 tiny layers; MiniMax-H3 conditions on the unnormalized last hidden state.

@pytest.fixture
def tiny_clip(monkeypatch):
    import models.minimax_h3_image  # noqa: F401  (attention masks on)
    import comfy.sd
    import comfy.supported_models_base
    import comfy.text_encoders.llama as llama
    import comfy.text_encoders.minimax as minimax
    import comfy.text_encoders.qwen3vl as qwen3vl

    @dataclasses.dataclass
    class Tiny(llama.Qwen3VL_32BConfig):
        hidden_size: int = 32
        intermediate_size: int = 48
        num_hidden_layers: int = 3
        num_attention_heads: int = 2
        num_key_value_heads: int = 1
        head_dim = 16

    monkeypatch.setitem(qwen3vl.QWEN3VL_CONFIGS, 'qwen3vl_32b', Tiny)
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION, 'qwen3vl_32b', dict(hidden_size=32, intermediate_size=48, depth=2, deepstack_visual_indexes=[0, 1]))
    monkeypatch.setitem(qwen3vl.QWEN3VL_VISION_COMMON, 'num_heads', 2)
    target = comfy.supported_models_base.ClipTarget(minimax.MiniMaxH3Tokenizer, minimax.te())
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


def te_pipeline(clip):
    pipe = image_pipeline(None)
    pipe.text_encoders = [clip]
    return pipe


def test_edit_captions_are_encoded_as_comfyui_encodes_them(tiny_clip, tmp_path):
    """Our text encoder function against the MiniMaxH3ImageConditioning node: "Picture 1: " and the control image
    at a 1024 short edge in front of the caption, the embeddings and the token tags of its vision block."""
    ns = _node_namespace('MiniMaxH3ImageConditioning')
    pipe = te_pipeline(tiny_clip)
    fn = pipe.get_call_text_encoder_fn(tiny_clip)
    caption = 'make the sky red'
    paths = [write_image(tmp_path / 'a.png', 1280, 1100, 0), write_image(tmp_path / 'b.png', 300, 1200, 1)]
    result = fn([caption, caption], [False, False], paths)
    for i, path in enumerate(paths):
        cond, _ = ns['execute'](None, tiny_clip, caption, 0, 0, vae=NodeVAE(), images={'image_1': load_comfy_image(path)})
        expected, extra = cond[0]
        assert_close(result['text_embeds_0'][i], expected[0], atol=1e-4)
        assert torch.equal(result['token_tags_0'][i], extra['minimax_token_tags'].long())
        assert result['attention_mask_0'][i].all() and result['attention_mask_0'][i].shape[0] == expected.shape[1]
        assert (result['token_tags_0'][i] == 0).any() and (result['token_tags_0'][i] == 1).any()
    # the vision blocks differ in length with the images' shapes
    assert result['token_tags_0'][0].shape != result['token_tags_0'][1].shape
    # a sample without a control image in the batch is text to image
    mixed = fn([caption, caption], [False, False], [paths[0], None])
    assert mixed['token_tags_0'][1].eq(1).all()


def test_text_to_image_captions_are_batched(tiny_clip):
    """Without control images the captions are encoded together, padded; each row's real tokens are the caption
    encoded alone, as the node encodes it, its tags all 1."""
    ns = _node_namespace('MiniMaxH3ImageConditioning')
    fn = te_pipeline(tiny_clip).get_call_text_encoder_fn(tiny_clip)
    captions = ['a red fox in the snow at dawn', 'a cat']
    result = fn(captions, [False, False], None)
    assert fn(captions, [False, False], [None, None])['text_embeds_0'].shape == result['text_embeds_0'].shape
    lengths = result['attention_mask_0'].sum(1).tolist()
    assert lengths[0] > lengths[1] and result['token_tags_0'].eq(1).all()
    for i, caption in enumerate(captions):
        cond, _ = ns['execute'](None, tiny_clip, caption, 0, 0)
        expected, _ = cond[0]
        assert expected.shape[1] == lengths[i]
        assert_close(result['text_embeds_0'][i, :lengths[i]], expected[0], atol=1e-4)


def test_another_text_encoder_or_a_video_is_refused(tiny_clip, monkeypatch):
    pipe = te_pipeline(tiny_clip)
    fn = pipe.get_call_text_encoder_fn(tiny_clip)
    with pytest.raises(ValueError, match='images only'):
        fn(['a cat'], [True], None)
    monkeypatch.setattr(tiny_clip.cond_stage_model, 'clip', 'qwen3vl_4b')
    with pytest.raises(ValueError, match='not the MiniMax H3 one'):
        fn(['a cat'], [False], None)


def test_the_conditioning_cache_key_names_the_picture_label():
    assert 'Picture %d: ' in image_pipeline(None).text_encoder_cache_key(0)


# A training step through DeepSpeed's pipeline engine in real gloo process groups.

def _step_inputs():
    video, t, text, mask, tags, _, augmented = image_inputs([9, 6], [(2, 4), (1, 2)], [0.3, 0.0005], seed=3)
    target = torch.randn(video.shape, generator=torch.Generator().manual_seed(5))
    return (video, t, text, mask, tags, augmented), (target, None)


LR = 0.1


def _split_model():
    model, _ = tiny_pair(3)
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    return model, pipe


def _expected_weights():
    model, pipe = _split_model()
    layers = pipe.to_layers()
    inputs, label = _step_inputs()
    loss = pipe.get_loss_fn()(run_layers(layers, inputs), label)
    loss.backward()
    return {name: (p - LR * p.grad if p.grad is not None else p).detach() for name, p in model.named_parameters()}


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
    model, pipe = _split_model()
    _sdpa_everywhere()
    layers = pipe.to_layers()
    module = ManualPipelineModule(layers=layers, num_stages=num_stages, partition_method='uniform',
                                  manual_partition_split=[len(layers) / num_stages], loss_fn=pipe.get_loss_fn(), dynamic_shape=True)
    params = [p for p in module.parameters() if p.requires_grad]
    engine, *_ = deepspeed.initialize(
        model=module, model_parameters=params, optimizer=torch.optim.SGD(params, lr=LR),
        config={'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1,
                'gradient_clipping': 0.0, 'steps_per_print': 10**9})
    engine._support_torch_style_backward = True
    # Through split_batch as the dataset hands batches to the engine: the missing mask becomes an empty tensor.
    from utils.dataset import split_batch
    engine.train_batch(iter(list(split_batch(_step_inputs(), 1))))
    if engine.grid.get_data_parallel_rank() == 0:
        weights = {name: p.detach().tolist() for name, p in module.named_parameters()}
        with open(f'{out_dir}/{rank}.json', 'w') as f:
            json.dump({'weights': weights}, f)


@pytest.mark.parametrize('world_size, num_stages', [(2, 2), (2, 1)], ids=['pp2', 'dp2'])
def test_a_training_step_through_the_pipeline_engine(world_size, num_stages, tmp_path):
    """The layers split over pipeline stages pass their tensors (the per-sample adaLN segments, the reference
    latents) from stage to stage and step as one process does; with data parallelism every rank steps on the same
    gradient. The head the layers train is the split head 0."""
    mp.spawn(_engine_worker, args=(world_size, num_stages, str(tmp_path / 'init'), str(tmp_path)), nprocs=world_size)
    expected = _expected_weights()
    # ManualPipelineModule names a parameter by its layer's index: map the model's names onto them.
    model, pipe = _split_model()
    names = {}
    for i, layer in enumerate(pipe.to_layers()):
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
    assert 'final_layer.video_out.weight' in covered


def test_the_example_configs_are_accepted(monkeypatch):
    import toml
    import models.minimax_h3
    from models.minimax_h3_image import MiniMaxH3ImagePipeline

    def base_init(self, config):
        self.config, self.model_config = config, config['model']

    monkeypatch.setattr(models.minimax_h3.MinimaxH3Pipeline, '__init__', base_init)
    config = toml.load(EXAMPLES / 'minimax_h3_image.toml')
    pipe = MiniMaxH3ImagePipeline(config)
    assert config['model']['type'] == 'minimax_h3_image'
    pipe.model_specific_dataset_config_validation(toml.load(EXAMPLES / 'minimax_h3_image_dataset.toml'))


def test_save_dtype_leaves_the_float32_islands_unrounded(tmp_path):
    """save_dtype = bfloat16 rounds the blocks but not the output head (rounding head 0 would move every head, the
    offsets being relative to it) nor the patch projection and time embedder."""
    from utils.saver import convert_state_dict_dtype
    model, _ = tiny_pair(3)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    name_parameters(model)
    sd = {p.original_name: p.detach() for p in model.parameters()}
    convert_state_dict_dtype(sd, torch.bfloat16, pipe.save_float32_prefixes)
    assert sd['blocks.0.attn.qkv_proj.weight'].dtype == torch.bfloat16
    for name in ('final_layer.video_out.weight', 'video_patch_proj.weight', 'time_embedder.proj_in.weight'):
        assert sd[name].dtype == torch.float32, name
    pipe.save_model(tmp_path, sd)
    saved = safetensors.torch.load_file(tmp_path / 'model.safetensors')
    assert torch.equal(saved['final_layer.video_out.weight'], original['final_layer.video_out.weight'])
    assert torch.equal(saved['video_patch_proj.weight'], original['video_patch_proj.weight'])
    assert saved['blocks.0.attn.qkv_proj.weight'].dtype == torch.bfloat16
    # an unlisted model rounds everything, as before
    plain = {'a.weight': torch.randn(2, 2)}
    convert_state_dict_dtype(plain, torch.bfloat16)
    assert plain['a.weight'].dtype == torch.bfloat16


def test_each_sample_trains_through_the_head_of_its_noise_interval():
    """The grid is [1, .6, .3, 0]: a sample at noise .8 sits in head 0's interval, .5 and .59 in head 1's, .2 in
    head 2's (a step from .6 down uses head 1, as ComfyUI picks it). The heads of one batch differ per sample."""
    model, _ = tiny_pair(3)
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    layers = pipe.to_layers()
    video, t, text, mask, tags, _, augmented = image_inputs([6, 6, 6, 6], [(1, 2)] * 4, [0.8, 0.5, 0.2, 0.59])
    inputs = (video, t, text, mask, tags, augmented)
    assert layers[0](inputs)[5].tolist()[6:] == [0, 1, 2, 1]
    by_sigma = run_layers(layers, inputs)
    forced = []
    for head in range(3):
        pipe.final_layer_module.head = head
        forced.append(run_layers(layers, inputs))
    pipe.final_layer_module.head = None
    for i, head in enumerate([0, 1, 2, 1]):
        assert_close(by_sigma[i], forced[head][i])
    assert not torch.allclose(forced[0][1], forced[1][1])

    # train_pdd_head = 0: head 0 whatever the noise
    head0 = image_pipeline(model, {'train_pdd_head': 0}).to_layers()
    assert head0[0](inputs)[5].tolist()[6:] == [0, 0, 0, 0]
    assert_close(run_layers(head0, inputs), forced[0])
    # one head: nothing to pick
    single, _ = tiny_pair(1)
    assert image_pipeline(single).to_layers()[0](inputs)[5].tolist()[6:] == [0, 0, 0, 0]


def test_the_gradient_of_a_later_head_reaches_head_0():
    model, _ = tiny_pair(3)
    pipe = image_pipeline(model)
    pipe.split_pdd_head()
    layers = pipe.to_layers()
    video, t, text, mask, tags, _, augmented = image_inputs([6], [(1, 2)], [0.2])
    run_layers(layers, (video, t, text, mask, tags, augmented)).square().mean().backward()
    assert model.final_layer.video_out.weight.grad.abs().sum() > 0
    assert model.final_layer.pdd_offset_weight.grad is None


def test_train_pdd_head_is_checked(tmp_path):
    from models.minimax_h3_image import MiniMaxH3ImagePipeline
    with pytest.raises(ValueError, match="train_pdd_head must be 'sigma' or 0"):
        MiniMaxH3ImagePipeline({'model': {'text_encoders': [{'path': 'x', 'type': 'minimax'}], 'train_pdd_head': 2}})
    model, _ = tiny_pair(1)
    write_comfy(tmp_path, model, 1)
    pipe = image_pipeline(None, {'diffusion_model': str(tmp_path / 'comfy.safetensors'), 'train_pdd_head': 0})
    with pytest.raises(ValueError, match='train_pdd_head does not apply'):
        pipe.refuse_unsupported_checkpoint()
    three, _ = tiny_pair(3)
    write_comfy(tmp_path / '..', three, 3)
    image_pipeline(None, {'diffusion_model': str(tmp_path / '..' / 'comfy.safetensors'), 'train_pdd_head': 0}).refuse_unsupported_checkpoint()


def test_pdd_grid_timesteps_are_the_samplers_sigmas():
    """timestep_sample_method = 'pdd_grid': the noise levels are the grid points the sampler runs, each through its own
    head, including the float32 value of a grid point (which sits above the double)."""
    model, _ = tiny_pair(3)
    pipe = image_pipeline(model, {'timestep_sample_method': 'pdd_grid'})
    pipe.split_pdd_head()
    torch.manual_seed(0)
    t = pipe.sample_timesteps(600, 1, LATENT_H, LATENT_W, 'cpu')
    assert set(t.tolist()) == {torch.tensor(g).item() for g in SIGMAS[3][:-1]}
    assert all(abs(t.eq(torch.tensor(g)).float().mean() - 1 / 3) < 0.08 for g in SIGMAS[3][:-1])
    assert pipe.sample_timesteps(2, 1, 4, 6, 'cpu', timestep_quantile=0.5).tolist() == [torch.tensor(0.6).item()] * 2
    assert pipe.sample_timesteps(1, 1, 4, 6, 'cpu', timestep_quantile=0.99).tolist() == [torch.tensor(0.3).item()]
    layers = pipe.to_layers()
    video, _, text, mask, tags, _, augmented = image_inputs([6, 6, 6], [(1, 2)] * 3, [1.0, 0.6, 0.3])
    sigmas = torch.tensor([torch.tensor(g).item() for g in SIGMAS[3][:-1]])
    assert layers[0]((video, sigmas, text, mask, tags, augmented))[5].tolist()[6:] == [0, 1, 2]
    # the other methods are untouched
    other = image_pipeline(model, {'timestep_sample_method': 'uniform'})
    assert not set(other.sample_timesteps(50, 1, 4, 6, 'cpu').tolist()) <= set(t.tolist())


def test_pdd_grid_replaces_the_continuous_sampling_keys(tmp_path):
    from models.minimax_h3_image import MiniMaxH3ImagePipeline
    text_encoders = [{'path': 'x', 'type': 'minimax'}]
    with pytest.raises(ValueError, match="shift, sigmoid_scale do not apply with timestep_sample_method = 'pdd_grid'"):
        MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders, 'timestep_sample_method': 'pdd_grid', 'shift': 3, 'sigmoid_scale': 1.0}})
    with pytest.raises(ValueError, match='timestep_sample_method must be one of'):
        MiniMaxH3ImagePipeline({'model': {'text_encoders': text_encoders, 'timestep_sample_method': 'sigmoid'}})
    model, _ = tiny_pair(1)
    write_comfy(tmp_path, model, 1)
    pipe = image_pipeline(None, {'diffusion_model': str(tmp_path / 'comfy.safetensors'), 'timestep_sample_method': 'pdd_grid'})
    with pytest.raises(ValueError, match="'pdd_grid' needs the PDD bank"):
        pipe.refuse_unsupported_checkpoint()
