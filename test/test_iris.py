"""Iris-3B pipeline (models/iris.py) against the submodule's own reference code, on CPU.

Every check compares diffusion-pipe's pipeline against iris3b itself -- the layer stack
against IrisDiT.forward, the text encoder against Qwen3VLTextEncoder, the saved checkpoint
against iris3b.sampling.load_for_inference -- on tiny randomly initialised models, so a
mismatch is a bug here rather than a tolerance question.
"""

import multiprocessing
import queue
import re
import sys
import threading
import types
from pathlib import Path

import pytest
import safetensors.torch
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'submodules' / 'iris-3b' / 'src'))

from iris3b.config import RepaConfig, inference_config  # noqa: E402
from iris3b.flow.schedule import FlowSchedule, resolution_shift  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402

from models import iris  # noqa: E402
from utils import dataset as dataset_util  # noqa: E402
from utils.oplora import OPLoRAProjector  # noqa: E402

TEXT_DIM = 32
TEXT_LEN = 16
HIDDEN_LAYERS = [1, 2, 4]


def tiny_raw_config(**model_overrides):
    pixel = model_overrides.pop('pixel', {})
    model = dict(hidden_size=64, depth=4, dual_depth=2, num_heads=4, num_kv_heads=2, patch_size=4,
                 text_dim=TEXT_DIM, text_len=TEXT_LEN, text_lap_num_layers=len(HIDDEN_LAYERS),
                 text_lap_num_heads=4, adaln_zero_init=False, repa_layer=2,
                 pixel=dict(hidden_size=8, attn_hidden_size=32, num_heads=4, depth=2, **pixel))
    model.update(model_overrides)
    return {'model': model,
            'text_encoder': dict(dim=TEXT_DIM, max_length=TEXT_LEN, hidden_layers=HIDDEN_LAYERS),
            'flow': dict(shift=3.0)}


def write_checkpoint(directory, raw=None, seed=0):
    """A random IrisDiT exported in Iris's own layout. Returns the reference model."""
    raw = raw or tiny_raw_config()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(raw), str(directory / 'config.yaml'))
    torch.manual_seed(seed)
    ref = IrisDiT(inference_config(raw).model)
    with torch.no_grad():
        # Default init zeroes the head and some biases; perturb so every path carries signal.
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    safetensors.torch.save_file({k: v.contiguous() for k, v in ref.state_dict().items()},
                                str(directory / 'model.safetensors'))
    ref.eval()
    return ref


def make_pipeline(ckpt_dir, text_encoder_path=None, **model_config):
    config = {
        'model': {'type': 'iris', 'transformer_path': str(ckpt_dir), 'dtype': torch.float32,
                  **model_config},
        'optimizer': {'lr': 1e-4},
        'reentrant_activation_checkpointing': False,
    }
    if text_encoder_path is None:
        config['model'].setdefault('load_text_encoder', False)
    else:
        config['model']['text_encoder_path'] = str(text_encoder_path)
    pipe = iris.IrisPipeline(config)
    return pipe


def run_layers(layers, inputs):
    out = inputs
    for layer in layers:
        out = layer(out)
    return out


def random_text(batch, lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(batch, TEXT_LEN, len(HIDDEN_LAYERS), TEXT_DIM, generator=g)
    y_mask = torch.zeros(batch, TEXT_LEN, dtype=torch.long)
    for i, n in enumerate(lengths):
        y_mask[i, :n] = 1
    return y * y_mask[:, :, None, None], y_mask


def random_inputs(batch=2, height=32, width=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 3, height, width, generator=g)
    t = torch.tensor([100.0, 750.0][:batch])
    y, y_mask = random_text(batch, [5, 11][:batch])
    return x, t, y, y_mask


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    # The real deepspeed may be installed without a process group; these tests are rank 0.
    import models.base
    for module in (iris, models.base, dataset_util):
        if hasattr(module, 'is_main_process'):
            monkeypatch.setattr(module, 'is_main_process', lambda: True)


@pytest.fixture
def ckpt(tmp_path):
    return tmp_path / 'ckpt', write_checkpoint(tmp_path / 'ckpt')


@pytest.fixture(scope='module')
def tiny_qwen(tmp_path_factory):
    """A 4-layer Qwen3-VL saved with the real Qwen3 tokenizer, so the chat template tokenizes
    exactly as it does for the 4B model."""
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen3VLForConditionalGeneration'):
        pytest.skip('transformers has no Qwen3-VL')
    path = tmp_path_factory.mktemp('tiny_qwen3_vl')
    tok = transformers.AutoTokenizer.from_pretrained(str(ROOT / 'configs' / 'qwen3_06b'))
    cfg = transformers.Qwen3VLConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=TEXT_DIM, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [2, 1, 1], 'mrope_interleaved': True}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=TEXT_DIM,
                           deepstack_visual_indexes=[0]),
    )
    torch.manual_seed(0)
    transformers.Qwen3VLForConditionalGeneration(cfg).save_pretrained(str(path))
    tok.save_pretrained(str(path))
    return path


# ---- forward / backward parity ------------------------------------------------------------------

LAYER_VARIANTS = {
    'shared_bias': {},
    'per_block': {'modulation': 'per_block'},
    'shared_lowrank': {'modulation': 'shared_lowrank', 'modulation_rank': 8},
    'no_pixel_head': {'pixel': {'enabled': False}},
    'adaln_zero_init': {'adaln_zero_init': True},
}


@pytest.mark.parametrize('variant', list(LAYER_VARIANTS))
def test_layer_stack_matches_reference_forward_and_backward(tmp_path, variant):
    raw = tiny_raw_config(**{k: (dict(v) if isinstance(v, dict) else v) for k, v in LAYER_VARIANTS[variant].items()})
    ref = write_checkpoint(tmp_path / 'ckpt', raw)
    pipe = make_pipeline(tmp_path / 'ckpt', cache_text_embeddings=True)
    pipe.load_diffusion_model()
    layers = pipe.to_layers()
    names = [type(layer).__name__ for layer in layers]
    assert names[:2] == ['TextEncoderLayer', 'InitialLayer']
    assert names.count('TransformerLayer') == raw['model']['depth']
    assert names[-1] == 'FinalLayer'
    for layer in layers:
        layer.eval()

    x, t, y, y_mask = random_inputs()
    ref.train()  # same mode as the layers' parameters; no dropout anywhere in IrisDiT
    ref.activation_checkpointing = 'none'
    expected = ref(x, t, y, y_mask=y_mask).x
    got = run_layers(layers, (x, t, y, y_mask))
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)

    # Full fine-tune: every parameter, the shared cores included, gets the reference gradient.
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (got * weight).sum().backward()
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    checked = 0
    for name, p in pipe.transformer.named_parameters():
        assert p.grad is not None, f'{name} received no gradient through the layer stack'
        torch.testing.assert_close(p.grad, ref_grads[name], rtol=1e-4, atol=1e-5, msg=lambda m: f"{name}: {m}")
        checked += 1
    assert checked == len(ref_grads)


def test_to_layers_is_idempotent(ckpt):
    path, ref = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    pipe.to_layers()
    layers = pipe.to_layers()
    x, t, y, y_mask = random_inputs()
    with torch.no_grad():
        torch.testing.assert_close(run_layers(layers, (x, t, y, y_mask)), ref(x, t, y, y_mask=y_mask).x,
                                   rtol=0, atol=1e-6)


def test_last_single_stream_block_skips_only_dead_text_output(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    assert pipe.transformer.blocks[-1].text_out is False
    assert all(getattr(b, 'text_out', True) for b in pipe.transformer.blocks[:-1])


def test_rejects_resolution_not_divisible_by_patch(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    layers = pipe.to_layers()
    x, t, y, y_mask = random_inputs(height=30)
    with pytest.raises(ValueError, match='not divisible'):
        run_layers(layers, (x, t, y, y_mask))


# ---- checkpoint loading / saving ----------------------------------------------------------------

def test_build_transformer_strips_prefixes_and_rejects_bad_checkpoints(ckpt):
    path, ref = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    sd = ref.state_dict()
    prefixed = [('model.diffusion_model.' + k, v) for k, v in sd.items()] + [('repa.projector.weight', torch.zeros(1))]
    model = pipe.build_transformer(iter(prefixed))
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, sd[k])

    missing = [(k, v) for k, v in sd.items() if not k.startswith('final_layer')]
    with pytest.raises(RuntimeError, match='missing'):
        pipe.build_transformer(iter(missing))
    with pytest.raises(RuntimeError, match='Unexpected key'):
        pipe.build_transformer(iter(list(sd.items()) + [('bogus.weight', torch.zeros(1))]))
    wrong = dict(sd)
    wrong['final_layer.linear.bias'] = torch.zeros(5)
    with pytest.raises(RuntimeError, match='Shape mismatch'):
        pipe.build_transformer(iter(wrong.items()))


def test_transformer_dtype_keeps_embedders_cores_and_head_in_base_dtype(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True, transformer_dtype=torch.bfloat16)
    pipe.load_diffusion_model()
    for name, p in pipe.transformer.named_parameters():
        high = p.ndim == 1 or any(k in name for k in iris.KEEP_IN_HIGH_PRECISION)
        assert p.dtype == (torch.float32 if high else torch.bfloat16), name
    assert any(p.dtype == torch.bfloat16 for p in pipe.transformer.blocks.parameters())


def test_full_fine_tune_save_reads_back_with_iris_loader(ckpt, tmp_path):
    from iris3b.sampling import load_for_inference
    path, ref = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    layers = pipe.to_layers()
    with torch.no_grad():
        for p in pipe.transformer.parameters():
            p.add_(0.01)
    # What utils/saver.py collects: every pipeline parameter under its original_name.
    pipeline_params = {p for layer in layers for p in layer.parameters()}
    assert pipeline_params == set(pipe.transformer.parameters())
    state_dict = {p.original_name: p.detach() for p in pipeline_params}
    out = tmp_path / 'saved'
    out.mkdir()
    pipe.save_model(out, state_dict)

    raw, weights = load_for_inference(out)
    reloaded = IrisDiT(inference_config(raw).model)
    reloaded.load_state_dict(weights, strict=True)
    reloaded.eval()
    pipe.transformer.eval()
    x, t, y, y_mask = random_inputs()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x, t, y, y_mask=y_mask).x, run_layers(layers, (x, t, y, y_mask)),
                                   rtol=0, atol=1e-6)
    # And diffusion-pipe can resume from it.
    again = make_pipeline(out, cache_text_embeddings=True)
    assert again.iris_config.model == pipe.iris_config.model
    again.load_diffusion_model()


# ---- adapters -----------------------------------------------------------------------------------

ADAPTERS = {
    'lora': {'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0, 'dtype': torch.float32},
    'lokr': {'type': 'lokr', 'rank': 4, 'alpha': 4, 'decompose_factor': -1, 'rank_dropout': 0.0,
             'dtype': torch.float32},
}


def saved_adapter_state(pipe):
    """What utils/saver.py hands save_adapter."""
    return {p.original_name.replace('.default', '').replace('.modules_to_save', ''): p.detach().clone()
            for p in pipe.transformer.parameters() if p.requires_grad}


@pytest.mark.parametrize('kind', list(ADAPTERS))
def test_adapter_trains_saves_reloads_and_merges(ckpt, tmp_path, kind):
    sys.path.insert(0, str(ROOT))
    from tools import iris_merge_adapter

    path, ref = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    pipe.configure_adapter(dict(ADAPTERS[kind]))
    trainable = [n for n, p in pipe.transformer.named_parameters() if p.requires_grad]
    assert trainable and all(('lora_' in n or 'lokr_' in n) for n in trainable)
    assert all(n.startswith('blocks.') for n in trainable), 'adapters target only the trunk blocks'
    # Both block types are adapted.
    assert any(n.startswith('blocks.0.') for n in trainable)
    assert any(n.startswith(f'blocks.{len(pipe.transformer.blocks) - 1}.') for n in trainable)

    layers = pipe.to_layers()
    x, t, y, y_mask = random_inputs()
    out = run_layers(layers, (x, t, y, y_mask))
    out.square().mean().backward()
    # Zero-initialised factors (LoRA's B, LoKr's w1) are the ones that get gradient first.
    for n, p in pipe.transformer.named_parameters():
        if p.requires_grad and re.search(r'lora_B|lokr_w1(_a)?\.', n):
            assert p.grad is not None and p.grad.abs().sum() > 0, n
    with torch.no_grad():
        for n, p in pipe.transformer.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)
        expected = run_layers(layers, (x, t, y, y_mask))
    assert not torch.allclose(expected, ref(x, t, y, y_mask=y_mask).x, atol=1e-4)

    save_dir = tmp_path / 'adapter'
    save_dir.mkdir()
    sd = saved_adapter_state(pipe)
    pipe.save_adapter(save_dir, sd)
    saved = safetensors.torch.load_file(str(save_dir / 'adapter_model.safetensors'))
    assert saved and all(k.startswith('diffusion_model.blocks.') for k in saved)

    # Resume path: a fresh pipeline loads it back into its own adapter.
    other = make_pipeline(path, cache_text_embeddings=True)
    other.load_diffusion_model()
    other.configure_adapter(dict(ADAPTERS[kind]))
    other.load_adapter_weights(save_dir)
    with torch.no_grad():
        torch.testing.assert_close(run_layers(other.to_layers(), (x, t, y, y_mask)), expected, rtol=0, atol=1e-6)

    # load_and_fuse_adapter (merge_adapters config option).
    fused = make_pipeline(path, cache_text_embeddings=True)
    fused.load_diffusion_model()
    fused.load_and_fuse_adapter(str(save_dir))
    with torch.no_grad():
        torch.testing.assert_close(run_layers(fused.to_layers(), (x, t, y, y_mask)), expected, rtol=1e-5, atol=1e-5)

    # The standalone merge tool yields a plain Iris checkpoint with the same output.
    merged, config_file, _ = iris_merge_adapter.merge(str(path), str(save_dir))
    assert set(merged) == set(ref.state_dict())
    plain = IrisDiT(inference_config(OmegaConf.to_container(OmegaConf.load(config_file))).model)
    plain.load_state_dict(merged, strict=True)
    plain.eval()
    with torch.no_grad():
        torch.testing.assert_close(plain(x, t, y, y_mask=y_mask).x, expected, rtol=1e-5, atol=1e-5)
        half, _, _ = iris_merge_adapter.merge(str(path), str(save_dir), strength=0.5)
        base = ref.state_dict()
        for k, v in half.items():
            torch.testing.assert_close(v - base[k], (merged[k] - base[k]) * 0.5, rtol=1e-4, atol=1e-6, msg=k)


def test_merge_tool_rejects_adapter_for_a_different_model(ckpt, tmp_path):
    from tools import iris_merge_adapter
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    pipe.configure_adapter(dict(ADAPTERS['lora']))
    save_dir = tmp_path / 'adapter'
    save_dir.mkdir()
    sd = saved_adapter_state(pipe)
    sd = {k: v for k, v in sd.items() if not k.startswith('blocks.0.')}
    pipe.save_adapter(save_dir, sd)
    with pytest.raises(RuntimeError, match='no weights'):
        iris_merge_adapter.merge(str(path), str(save_dir))


def test_oplora_projects_iris_adapter(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    pipe.configure_adapter(dict(ADAPTERS['lora']))
    root = torch.nn.ModuleList(pipe.to_layers())
    with torch.no_grad():
        for n, p in pipe.transformer.named_parameters():
            if 'lora_B' in n:
                p.normal_()
    projector = OPLoRAProjector.build(root, rank=2, full_svd=True, base_seed=0, exclude_names=pipe.oplora_exclude_names)
    num_linear = sum(1 for n, m in pipe.transformer.named_modules() if hasattr(m, 'lora_A'))
    assert len(projector._entries) == num_linear > 0
    assert projector.max_residual() > 1e-3
    projector.project()
    assert projector.max_residual() < 1e-4


# ---- text encoder -------------------------------------------------------------------------------

def test_prompt_template_matches_submodule():
    from iris3b.text import qwen3_vl
    assert iris.PROMPT_PREFIX == qwen3_vl._PROMPT_PREFIX
    assert iris.PROMPT_SUFFIX == qwen3_vl._PROMPT_SUFFIX


def test_text_encoder_matches_iris_encoder(tiny_qwen):
    from iris3b.config import TextEncoderConfig
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
    reference = Qwen3VLTextEncoder(TextEncoderConfig(pretrained=str(tiny_qwen), dim=TEXT_DIM, max_length=TEXT_LEN,
                                                     hidden_layers=HIDDEN_LAYERS, dtype='float32'))
    ours = iris.IrisTextEncoder.from_pretrained(str(tiny_qwen), torch.float32, HIDDEN_LAYERS, TEXT_LEN)
    captions = ['a cat', '', 'an extremely long caption ' * 10, 'a red bicycle leaning on a wall']
    expected = reference.encode(captions)
    emb, mask = ours.encode(captions)
    assert mask.dtype == torch.int64
    torch.testing.assert_close(mask, expected.mask.to(torch.int64))
    torch.testing.assert_close(emb, expected.embeddings.float(), rtol=1e-5, atol=1e-5)
    assert ours.overflow_rows == 1
    assert int(mask[2].sum()) == TEXT_LEN  # truncated caption still ends with the full suffix


def test_cached_embeddings_are_trimmed_and_pad_back_exactly(ckpt, tiny_qwen):
    path, _ = ckpt
    pipe = make_pipeline(path, text_encoder_path=tiny_qwen, cache_text_embeddings=True)
    te = pipe.get_text_encoders()[0]
    captions = ['a cat', 'two dogs on a sofa']
    emb, mask = te.encode(captions)
    stored = pipe.get_call_text_encoder_fn(te)(captions, [False, False])['prompt_embeds']
    assert [s.shape[0] for s in stored] == mask.sum(1).tolist()
    y, y_mask = pipe._pad_text_embeddings(stored)
    torch.testing.assert_close(y, emb)
    torch.testing.assert_close(y_mask, mask)


def test_on_the_fly_text_encoding_matches_cached_through_the_layers(ckpt, tiny_qwen):
    path, _ = ckpt
    captions = ['a cat', 'two dogs on a sofa']
    pixels = (torch.rand(2, 3, 32, 48) * 255).round().to(torch.uint8)

    outputs = []
    for cached in (False, True):
        pipe = make_pipeline(path, text_encoder_path=tiny_qwen, cache_text_embeddings=cached)
        assert (pipe.get_text_encoders() != []) == cached
        pipe.load_diffusion_model()
        layers = pipe.to_layers()
        for layer in layers:
            layer.eval()
        inputs = {'latents': pixels, 'mask': None, 'caption': captions}
        if cached:
            fn = pipe.get_call_text_encoder_fn(pipe.text_encoder)
            inputs['prompt_embeds'] = fn(captions, [False, False])['prompt_embeds']
        torch.manual_seed(0)
        features, label = pipe.prepare_inputs(inputs, timestep_quantile=0.5)
        if not cached:
            assert features[2].dtype == torch.long  # token ids, encoded by TextEncoderLayer
        with torch.no_grad():
            outputs.append(run_layers(layers, features))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)


def test_default_is_on_the_fly_text_and_pixels(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path)
    assert pipe.model_config['cache_text_embeddings'] is False
    assert pipe.get_text_encoders() == []
    assert pipe.load_media_on_the_fly is True
    assert make_pipeline(path, cache_pixels=True).load_media_on_the_fly is False


# ---- flow objective -----------------------------------------------------------------------------

@pytest.mark.parametrize('shift_law', ['none', 'sd3', 'flux'])
def test_prepare_inputs_matches_iris_flow_schedule(ckpt, shift_law):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True, shift_law=shift_law)
    pixels = (torch.rand(2, 3, 32, 48) * 255).round().to(torch.uint8)
    y, y_mask = random_text(2, [3, 7])
    prompt_embeds = [y[0, :3], y[1, :7]]
    mask = torch.ones(2, 32, 48, dtype=torch.float16)
    torch.manual_seed(123)
    features, (target, label_mask) = pipe.prepare_inputs(
        {'latents': pixels, 'mask': mask, 'prompt_embeds': prompt_embeds}, timestep_quantile=0.3)
    x_t, t_model, y_out, y_mask_out = features
    torch.testing.assert_close(y_out, y)
    torch.testing.assert_close(y_mask_out, y_mask)
    assert label_mask.shape == (2, 1, 32, 48)

    x0 = pixels.float() / 127.5 - 1
    assert x0.min() >= -1 and x0.max() <= 1
    shift = resolution_shift(8 * 12, shift_law, 3.0, pipe.iris_config.flow.shift_base_tokens)
    schedule = FlowSchedule(1000, shift)
    u = torch.sigmoid(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.3)))
    idx = torch.full((2,), int(u * 1000))
    torch.manual_seed(123)
    noise = torch.randn_like(x0)
    expected_xt, expected_t = schedule.add_noise(x0, noise, idx)
    torch.testing.assert_close(x_t, expected_xt)
    torch.testing.assert_close(t_model, expected_t)
    torch.testing.assert_close(target, noise - x0)


def test_timestep_sampling_matches_iris_sampler_distribution(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    torch.manual_seed(0)
    ours = pipe._sample_timestep_idx(20000)
    from iris3b.flow.timesteps import logit_normal
    theirs = logit_normal(20000, 1000, generator=torch.Generator().manual_seed(1))
    assert ours.min() >= 0 and ours.max() <= 999
    assert abs(ours.float().mean() - theirs.float().mean()) < 10
    assert abs(ours.float().std() - theirs.float().std()) < 10

    uniform = make_pipeline(path, cache_text_embeddings=True, timestep_sample_method='uniform',
                            min_t=0.2, max_t=0.4)
    idx = uniform._sample_timestep_idx(5000)
    assert idx.min() >= 200 and idx.max() < 400


def test_loss_fn_masks_and_adds_repa(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    loss_fn = pipe.get_loss_fn()
    out = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    mse = ((out - target) ** 2).mean()
    torch.testing.assert_close(loss_fn(out, (target, torch.tensor([]))), mse)
    mask = torch.zeros(2, 1, 8, 8)
    mask[0] = 1
    torch.testing.assert_close(loss_fn(out, (target, mask)), (((out - target) ** 2) * mask).mean())
    pipe.repa_weight = 0.5
    loss_fn = pipe.get_loss_fn()
    torch.testing.assert_close(loss_fn((out, torch.tensor([2.0])), (target, torch.tensor([]))), mse + 1.0)


def test_param_groups(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True, text_adapter_lr=0, pixel_head_lr=3e-4)
    pipe.load_diffusion_model()
    params = list(pipe.transformer.parameters())
    groups = pipe.get_param_groups(params)
    by_lr = {g['lr']: g['params'] for g in groups}
    assert set(by_lr) == {1e-4, 3e-4}
    assert all(p.original_name.startswith(('pixel_', 'final_layer')) for p in by_lr[3e-4])
    frozen = [p for p in params if not p.requires_grad]
    assert frozen and all(p.original_name.startswith(('y_embedder', 'y_pos_embedding')) for p in frozen)
    grouped = sum(len(g['params']) for g in groups)
    assert grouped + len(frozen) == len(params)


# ---- REPA -----------------------------------------------------------------------------------------

class FakeTeacher(torch.nn.Module):
    """Stands in for DINOv2: patch-14 conv tokens, returned the way the hub model does."""

    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, dim, kernel_size=14, stride=14)

    def forward_features(self, x):
        return {'x_norm_patchtokens': self.proj(x).flatten(2).transpose(1, 2)}


@pytest.fixture
def fake_hub(monkeypatch):
    def load(repo, model, **kwargs):
        torch.manual_seed(5)
        return FakeTeacher(24)
    monkeypatch.setattr(torch.hub, 'load', load)


def test_repa_trains_projector_and_matches_iris_loss(ckpt, fake_hub, tmp_path):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True, repa_weight=0.5, repa_teacher_dim=24,
                         repa_teacher_match_student=True)
    pipe.load_diffusion_model()
    assert pipe.repa is not None and pipe.repa_layer == 2
    layers = pipe.to_layers()
    names = [type(layer).__name__ for layer in layers]
    assert names.index('RepaLayer') == 2 + pipe.repa_layer

    pixels = (torch.rand(2, 3, 32, 48) * 255).round().to(torch.uint8)
    y, y_mask = random_text(2, [3, 7])
    features, label = pipe.prepare_inputs(
        {'latents': pixels, 'mask': None, 'prompt_embeds': [y[0, :3], y[1, :7]]}, timestep_quantile=0.5)
    assert len(features) == 5
    out = run_layers(layers, features)
    assert isinstance(out, tuple) and out[1].shape == (1,)

    # The same alignment loss Iris computes from the captured block output.
    ref = IrisDiT(pipe.iris_config.model)
    ref.load_state_dict({k: v.detach() for k, v in pipe.transformer.state_dict().items()})
    ref_out = ref(features[0], features[1], features[2], capture=(2,), y_mask=features[3])
    expected_repa = pipe.repa(features[4], ref_out.features[2], (8, 12))
    torch.testing.assert_close(out[1][0], expected_repa.float())
    torch.testing.assert_close(out[0], ref_out.x)
    assert pipe.get_extra_log_scalars()['train/repa_loss'] == pytest.approx(float(expected_repa))

    loss = pipe.get_loss_fn()(out, (label[0], torch.tensor([])))
    loss.backward()
    assert all(p.grad is not None for p in pipe.repa.parameters())
    assert all(p.original_name.startswith('repa.') for p in pipe.repa.parameters())

    # Eval: no teacher call, loss is the plain flow loss.
    for layer in layers:
        layer.eval()
    with torch.no_grad():
        out = run_layers(layers, features)
    assert float(out[1]) == 0.0

    # Full save writes the projector beside, not inside, model.safetensors.
    sd = {p.original_name: p.detach() for layer in layers for p in layer.parameters()}
    assert any(k.startswith('repa.') for k in sd)
    save = tmp_path / 'full'
    save.mkdir()
    pipe.save_model(save, dict(sd))
    assert not any(k.startswith('repa.') for k in safetensors.torch.load_file(str(save / 'model.safetensors')))
    assert (save / 'repa_projector.safetensors').exists()


def test_repa_with_adapter_keeps_projector_out_of_adapter_file(ckpt, fake_hub, tmp_path):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True, repa_weight=0.5, repa_teacher_dim=24)
    pipe.load_diffusion_model()
    pipe.configure_adapter(dict(ADAPTERS['lora']))
    assert all(p.requires_grad for p in pipe.repa.parameters())
    layers = pipe.to_layers()
    with torch.no_grad():
        for p in pipe.repa.parameters():
            p.add_(1.0)
    sd = {p.original_name.replace('.default', ''): p.detach().clone()
          for layer in layers for p in layer.parameters() if p.requires_grad}
    save = tmp_path / 'adapter'
    save.mkdir()
    pipe.save_adapter(save, sd)
    assert not any('repa' in k for k in safetensors.torch.load_file(str(save / 'adapter_model.safetensors')))
    other = make_pipeline(path, cache_text_embeddings=True, repa_weight=0.5, repa_teacher_dim=24)
    other.load_diffusion_model()
    other.configure_adapter(dict(ADAPTERS['lora']))
    other.load_adapter_weights(save)
    for a, b in zip(pipe.repa.parameters(), other.repa.parameters()):
        torch.testing.assert_close(a, b)


def test_repa_without_teacher_fails_loudly(ckpt, monkeypatch):
    path, _ = ckpt

    def load(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr(torch.hub, 'load', load)
    pipe = make_pipeline(path, cache_text_embeddings=True, repa_weight=0.5)
    with pytest.raises(RuntimeError, match='teacher could not be loaded'):
        pipe.load_diffusion_model()


def test_repa_off_by_default(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    assert pipe.repa is None
    assert 'RepaLayer' not in [type(layer).__name__ for layer in pipe.to_layers()]


# ---- dataset end to end ---------------------------------------------------------------------------

class LocalPool:
    def __init__(self, count, initializer, args):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(count, initializer=initializer, initargs=args)

    def imap(self, function, iterable):
        return self.executor.map(function, iterable)

    def close(self):
        self.executor.shutdown(wait=True)


LOCAL_WORKERS = types.SimpleNamespace(
    Manager=lambda: types.SimpleNamespace(Queue=queue.Queue), Pool=LocalPool, Pipe=multiprocessing.Pipe)


@pytest.fixture
def local_workers(monkeypatch):
    monkeypatch.setattr(dataset_util, 'NUM_PROC', 1)
    monkeypatch.setattr(dataset_util, 'mp', LOCAL_WORKERS)


def make_image_dir(path, count=4):
    path.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    for i in range(count):
        array = (torch.rand(40, 56, 3, generator=g) * 255).to(torch.uint8).numpy()
        Image.fromarray(array).save(path / f'{i}.png')
        (path / f'{i}.txt').write_text(f'caption number {i}')
    return path


def run_caching(model, ds, dataset_config):
    """DatasetManager.cache without its process and device handling: _cache_fn runs in a thread
    on its own Dataset object (the forked copy, in the real thing), and the main thread answers
    its tasks with DatasetManager._handle_task."""
    worker_ds = dataset_util.Dataset(dataset_config, model)
    manager = dataset_util.DatasetManager(model)
    # Nothing to move between devices on CPU; _handle_task treats non-modules as lazy loaders.
    manager.submodels = [types.SimpleNamespace(load_model_if_needed=lambda: None) for _ in manager.submodels]
    q = queue.Queue()
    errors = []

    def worker():
        try:
            dataset_util._cache_fn([worker_ds], q, model.get_preprocess_media_file_fn(), len(manager.text_encoders),
                                   False, False, 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            q.put(None)

    thread = threading.Thread(target=worker)
    thread.start()
    while (task := q.get()) is not None:
        manager._handle_task(task)
    thread.join()
    if errors:
        raise errors[0]
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(None, trust_cache=True)
    for i in range(1, len(manager.text_encoders) + 1):
        ds.cache_text_embeddings(None, i)


@pytest.mark.parametrize('cache_pixels', [False, True])
@pytest.mark.parametrize('cache_text', [False, True])
@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
def test_dataset_to_layers_end_to_end(ckpt, tiny_qwen, tmp_path, local_workers, cache_pixels, cache_text,
                                      reuse_metadata_cache):
    path, _ = ckpt
    images = make_image_dir(tmp_path / 'images')
    pipe = make_pipeline(path, text_encoder_path=tiny_qwen, cache_text_embeddings=cache_text,
                         cache_pixels=cache_pixels)
    pipe.load_diffusion_model()
    dataset_config = {
        'resolutions': [40], 'enable_ar_bucket': False,
        'reuse_metadata_cache': reuse_metadata_cache,
        'directory': [{'path': str(images), 'num_repeats': 1, 'size_buckets': [[48, 32, 1]]}],
    }
    ds = dataset_util.Dataset(dataset_config, pipe)
    run_caching(pipe, ds, dataset_config)
    ds.post_init(0, 1, {None: 2}, 1, {None: 2})
    assert len(ds) == 2

    preprocess = pipe.get_preprocess_media_file_fn()
    seen = set()
    layers = pipe.to_layers()
    for k in range(len(ds)):
        batch = ds[k]
        assert batch['latents'].dtype == torch.uint8
        assert batch['latents'].shape == (2, 3, 32, 48)
        # Pixels are byte-exact with the resize PreprocessMediaFile does, cached or not.
        for spec, pixels in zip(batch['image_spec'], batch['latents']):
            tensor = preprocess(tuple(spec), None, (48, 32, 1))[0][0]
            assert torch.equal(pixels, iris.pixels_to_uint8(tensor))
            seen.add(Path(spec[1]).name)
        if cache_text:
            assert 'prompt_embeds' in batch
        else:
            assert all(c.startswith('caption number') for c in batch['caption'])
        features, label = pipe.prepare_inputs(batch)
        out = run_layers(layers, features)
        assert out.shape == (2, 3, 32, 48)
        loss = pipe.get_loss_fn()(out, (label[0], torch.tensor([])))
        assert torch.isfinite(loss)
    assert seen == {f'{i}.png' for i in range(4)}
    cache_root = images / 'cache'
    latent_files = [p for p in cache_root.rglob('*') if p.is_file() and 'latent' in p.as_posix()]
    if not cache_pixels:
        assert not latent_files, latent_files


def test_on_the_fly_dataset_rejects_control_datasets(ckpt):
    path, _ = ckpt
    pipe = make_pipeline(path, cache_text_embeddings=True)
    with pytest.raises(NotImplementedError):
        pipe.model_specific_dataset_config_validation({'directory': [{'path': 'x', 'control_path': 'y'}]})


# ---- block swap -----------------------------------------------------------------------------------

def test_block_swap_split(ckpt, monkeypatch):
    path, _ = ckpt
    raw = tiny_raw_config(depth=10, dual_depth=4)
    write_checkpoint(path, raw)
    pipe = make_pipeline(path, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    created = []

    class FakeOffloader:
        def __init__(self, name, blocks, num_blocks, to_swap, *args, **kwargs):
            created.append((name, num_blocks, to_swap))

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(iris, 'ModelOffloader', FakeOffloader)
    monkeypatch.setattr(pipe.transformer, 'to', lambda *a, **k: pipe.transformer)
    pipe.enable_block_swap(6)
    assert created == [('MMDiTBlock', 4, 2), ('SingleStreamBlock', 6, 4)]
    assert pipe.transformer.blocks is not None
    with pytest.raises(AssertionError):
        pipe.enable_block_swap(9)


# ---- sampling -------------------------------------------------------------------------------------

@pytest.mark.parametrize('cfg_scale', [1.0, 3.0])
def test_sample_matches_iris_generate(ckpt, tiny_qwen, cfg_scale):
    from iris3b.config import TextEncoderConfig
    from iris3b.sampling import generate
    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder
    path, ref = ckpt
    pipe = make_pipeline(path, text_encoder_path=tiny_qwen)
    pipe.load_diffusion_model()
    layers = pipe.to_layers()
    pipe.pipeline_model = lambda inputs: run_layers(layers, inputs)
    pipe.prepare_sample_test('a cat on a mat', negative_prompt='blurry', cfg=cfg_scale, device='cpu')
    noise = torch.randn(1, 3, 32, 48, generator=torch.Generator().manual_seed(3))
    got = pipe.sample(w=48, h=32, steps=4, noise=noise)
    assert got.shape == (1, 32, 48, 3)
    assert got.min() >= 0 and got.max() <= 1

    te = Qwen3VLTextEncoder(TextEncoderConfig(pretrained=str(tiny_qwen), dim=TEXT_DIM, max_length=TEXT_LEN,
                                              hidden_layers=HIDDEN_LAYERS, dtype='float32'), device='cpu')
    expected = generate(ref, te, ['a cat on a mat'], 32, 48, steps=4, cfg_scale=cfg_scale,
                        shift=pipe.iris_config.flow.shift, negative_prompt='blurry', device='cpu', noise=noise)
    torch.testing.assert_close(got, ((expected + 1) / 2).permute(0, 2, 3, 1), rtol=1e-4, atol=1e-4)


def test_repa_projector_loads_from_iris_checkpoint(tmp_path, fake_hub):
    raw = tiny_raw_config()
    ref = write_checkpoint(tmp_path / 'ckpt', raw)
    sd = {k: v.contiguous() for k, v in ref.state_dict().items()}
    from iris3b.repa import REPALoss
    projector = REPALoss(RepaConfig(teacher_dim=24), 64).projector
    for k, v in projector.state_dict().items():
        sd['repa.projector.' + k] = torch.randn_like(v)
    safetensors.torch.save_file(sd, str(tmp_path / 'ckpt' / 'model.safetensors'))
    pipe = make_pipeline(tmp_path / 'ckpt', cache_text_embeddings=True, repa_weight=0.5, repa_teacher_dim=24)
    pipe.load_diffusion_model()
    for k, v in pipe.repa.projector.state_dict().items():
        torch.testing.assert_close(v, sd['repa.projector.' + k])
    # repa_projector_path wins over the checkpoint's own projector, in any of its key layouts.
    bare = {k: torch.zeros_like(v) for k, v in projector.state_dict().items()}
    safetensors.torch.save_file(bare, str(tmp_path / 'proj.safetensors'))
    pipe = make_pipeline(tmp_path / 'ckpt', cache_text_embeddings=True, repa_weight=0.5, repa_teacher_dim=24,
                         repa_projector_path=str(tmp_path / 'proj.safetensors'))
    pipe.load_diffusion_model()
    assert all(float(v.abs().sum()) == 0 for v in pipe.repa.projector.state_dict().values())
