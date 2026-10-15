"""FLUX.3 Klein Base pipeline (models/flux3.py) against ComfyUI's own FLUX.3 code, on CPU.

The checks run a tiny randomly initialised comfy.ldm.flux3.model.Flux3 (hidden size 128, one
head: the rope axes fix the head size at 128) through diffusion-pipe's layer stack and compare
it with ComfyUI's forward on the same weights, so a mismatch is a bug in the pipeline rather
than a tolerance question. The text encoder is a stand-in for Qwen3-VL behind the real FLUX.3
tokenizer.
"""

import math
import multiprocessing
import queue
import sys
import threading
import types
from pathlib import Path

import pytest
import safetensors.torch
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'submodules' / 'ComfyUI'))

from models import flux3  # noqa: E402
import comfy.ldm.flux3.model as comfy_flux3  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.sd1_clip  # noqa: E402
import comfy.text_encoders.flux  # noqa: E402
from utils import dataset as dataset_util  # noqa: E402
from utils.oplora import OPLoRAProjector  # noqa: E402

sys.path.insert(0, str(ROOT / 'tools'))
import comfyui_flux3_reference_lora as node  # noqa: E402

HIDDEN = 128
TEXT_LAYERS, TEXT_DIM = 8, 2560


def make_model(seed=0, fused_qkv=False, flow_map=False, depth=2, single=3):
    torch.manual_seed(seed)
    model = comfy_flux3.Flux3(hidden_size=HIDDEN, num_heads=1, depth=depth, depth_single_blocks=single,
                              fused_qkv=fused_qkv, flow_map=flow_map, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / math.sqrt(p.shape[1]))
    return model.eval()


def write_checkpoint(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    return path


class FakeQwen:
    """Stands in for Flux3Qwen3VLModel.encode: per-layer features of each token, [B, 8, L, 2560]."""

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.freqs = torch.rand(TEXT_LAYERS, 1, TEXT_DIM, generator=g, dtype=torch.float64) * 1e-3
        self.calls = []

    def encode(self, tokens):
        self.calls.append(tokens)
        ids = torch.tensor(tokens, dtype=torch.float64)
        z = torch.sin(ids[:, None, :, None] * self.freqs[None] + torch.arange(ids.shape[1])[None, None, :, None] * 0.01)
        return z.float(), None


class FakeCondStageModel(nn.Module):
    clip = 'qwen3vl_4b'

    def __init__(self):
        super().__init__()
        self.qwen3vl_4b_impl = FakeQwen()
        self.options = []

    def __getattr__(self, name):
        if name == 'qwen3vl_4b':
            return self.qwen3vl_4b_impl
        return super().__getattr__(name)

    def set_clip_options(self, options):
        self.options.append(options)

    def reset_clip_options(self):
        self.options.append(None)


class FakeTextEncoder:
    """The parts of ComfyUI's CLIP object the pipeline uses: the real FLUX.3 tokenizer and a
    stand-in for Qwen3-VL."""

    def __init__(self):
        self.tokenizer = comfy.text_encoders.flux.Flux3Tokenizer()
        self.cond_stage_model = FakeCondStageModel()
        self.patcher = types.SimpleNamespace(load_device=torch.device('cpu'))

    def load_model_if_needed(self):
        pass

    def load_model(self):
        pass

    def tokenize(self, text):
        return self.tokenizer.tokenize_with_weights(text)


class FakeVAE:
    """2x2-patched 16x down-sampling with 128 channels, like the FLUX.2 VAE's latent layout."""

    def load_model_if_needed(self):
        pass

    def encode(self, pixels):
        x = pixels.movedim(-1, 1).float()
        x = torch.nn.functional.pixel_unshuffle(x, 16)  # [B, 3*256, H/16, W/16]
        return x[:, :128] * 0.5


def make_pipeline(tmp_path, ckpt=None, adapter=None, **model_config):
    te = tmp_path / 'te' / 'model-00001-of-00002.safetensors'
    te.parent.mkdir(parents=True, exist_ok=True)
    te.touch()
    config = {
        'model': {'type': 'flux3', 'diffusion_model': str(ckpt or tmp_path / 'missing.safetensors'),
                  'vae': str(tmp_path / 'vae.safetensors'), 'dtype': torch.float32,
                  'text_encoders': [{'path': str(te.parent), 'type': 'flux3'}], 'guidance': 1.0,
                  **model_config},
        'optimizer': {'lr': 1e-4},
        'reentrant_activation_checkpointing': False,
    }
    if adapter is not None:
        config['adapter'] = adapter
    pipe = flux3.Flux3Pipeline(config)
    pipe.text_encoders = [FakeTextEncoder()]
    pipe.vae = FakeVAE()
    return pipe


def attach(pipe, model):
    pipe.diffusion_model = model
    pipe.prepare_diffusion_model()
    return pipe


def lora(rank=4, **kwargs):
    return {'type': 'lora', 'rank': rank, 'alpha': rank, 'dropout': 0.0, 'dtype': torch.float32, **kwargs}


def lokr(rank=4):
    return {'type': 'lokr', 'rank': rank, 'alpha': rank, 'decompose_factor': -1, 'rank_dropout': 0.0,
            'dtype': torch.float32}


def randomize_adapter(pipe, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in pipe.diffusion_model.named_parameters():
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def run_layers(layers, inputs):
    out = inputs
    for layer in layers:
        out = layer(out)
    return out


def text_embeds(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, TEXT_LAYERS * TEXT_DIM, generator=g) * 0.5 for n in lengths]


def latents(batch, h=3, w=4, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 128, h, w, generator=g)


def ref_latents(batch, shapes, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, 128, h, w, generator=g) for h, w in shapes]


def features(x, t, embeds, refs=()):
    text, lengths = flux3.pad_text_embeddings(embeds)
    return (x, t, text, lengths, *refs)


@torch.no_grad()
def comfy_forward(model, x, t, context, refs=()):
    return model(x.clone(), t, context=context.clone(), ref_latents=[r.clone() for r in refs] or None)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    import models.base
    for module in (flux3, models.base, dataset_util):
        if hasattr(module, 'is_main_process'):
            monkeypatch.setattr(module, 'is_main_process', lambda: True)


@pytest.fixture(autouse=True)
def cpu_scheduler(monkeypatch):
    # CommonPipeline sets up its sampling scheduler on CUDA.
    import models.base
    scheduler = models.base.FlowMatchEulerDiscreteScheduler
    original = scheduler.set_timesteps
    monkeypatch.setattr(scheduler, 'set_timesteps', lambda self, *a, device=None, **k: original(self, *a, device='cpu', **k))


@pytest.fixture(autouse=True)
def training_mode():
    # train.py runs with ComfyUI's training flag set: the eager reference arithmetic.
    old = comfy.model_management.in_training
    comfy.model_management.in_training = True
    yield
    comfy.model_management.in_training = old


# ---- forward against ComfyUI ---------------------------------------------------------------------

@pytest.mark.parametrize('ref_shapes', [(), ((3, 4),), ((2, 2), (3, 3), (1, 4))])
@pytest.mark.parametrize('batch', [1, 2])
def test_layers_match_comfyui_forward(tmp_path, ref_shapes, batch):
    model = make_model()
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), model)
    x = latents(batch)
    t = torch.tensor([0.3, 0.85][:batch])
    embeds = text_embeds([80] * batch)
    refs = ref_latents(batch, ref_shapes)
    with torch.no_grad():
        out = run_layers(pipe.to_layers(), features(x, t, embeds, refs))
    expected = comfy_forward(model, x, t, torch.stack(embeds), refs)
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected)


def test_checkpoint_loads_through_comfyui(tmp_path):
    """The real loader: ComfyUI detects the tiny checkpoint as FLUX.3 and the pipeline trains it."""
    model = make_model()
    ckpt = write_checkpoint(tmp_path / 'flux3.safetensors', model)
    pipe = make_pipeline(tmp_path, ckpt, cache_text_embeddings=True)
    pipe.load_diffusion_model()
    assert type(pipe.diffusion_model).__name__ == 'Flux3'
    assert all(p.requires_grad for p in pipe.diffusion_model.parameters())
    x, t, embeds = latents(1), torch.tensor([0.5]), text_embeds([80])
    refs = ref_latents(1, [(2, 3)])
    with torch.no_grad():
        out = run_layers(pipe.to_layers(), features(x, t, embeds, refs))
    torch.testing.assert_close(out, comfy_forward(model, x, t, torch.stack(embeds), refs))


@pytest.mark.parametrize('ref_shapes', [(), ((2, 3), (3, 2))])
def test_batched_mixed_text_lengths_equal_per_sample(tmp_path, ref_shapes):
    """FLUX.3 attends to every text token: the extra padding of a batch must be invisible."""
    model = make_model()
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), model)
    layers = pipe.to_layers()
    x = latents(3)
    t = torch.tensor([0.2, 0.5, 0.9])
    embeds = text_embeds([80, 240, 160])
    refs = ref_latents(3, ref_shapes)
    with torch.no_grad():
        batched = run_layers(layers, features(x, t, embeds, refs))
        for i in range(3):
            alone = run_layers(layers, features(x[i:i + 1], t[i:i + 1], embeds[i:i + 1], [r[i:i + 1] for r in refs]))
            torch.testing.assert_close(batched[i:i + 1], alone)
            torch.testing.assert_close(alone, comfy_forward(model, x[i:i + 1], t[i:i + 1], embeds[i][None],
                                                            [r[i:i + 1] for r in refs]))


def test_diffusion_model_dtype_keeps_embeddings_and_modulations_in_high_precision(tmp_path):
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model())
    pipe.dequantize(pipe.diffusion_model, torch.float16)
    for name, p in pipe.diffusion_model.named_parameters():
        high = p.ndim == 1 or name.startswith(('emb_in.', 'txt_in.', 'time_in.', 'final_layer.',
                                               'early_stream_modulations.', 'single_stream_modulations.'))
        assert p.dtype == (torch.float32 if high else torch.float16), name


def test_fused_qkv_checkpoint_is_split(tmp_path):
    fused = make_model(fused_qkv=True)
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model(fused_qkv=True))
    for module in pipe.diffusion_model.modules():
        if isinstance(module, comfy_flux3.Block):
            assert not module.fused_qkv and not hasattr(module, 'qkv')
            assert module.q_proj.weight.shape == (HIDDEN, HIDDEN)
    names = {n for n, _ in pipe.diffusion_model.named_parameters()}
    assert 'single_blocks.0.k_proj.weight' in names and not any('.qkv.' in n for n in names)
    x, t, embeds, refs = latents(1), torch.tensor([0.4]), text_embeds([80]), ref_latents(1, [(2, 2)])
    with torch.no_grad():
        out = run_layers(pipe.to_layers(), features(x, t, embeds, refs))
    torch.testing.assert_close(out, comfy_forward(fused, x, t, torch.stack(embeds), refs))


def test_backward_reaches_every_parameter(tmp_path):
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model())
    x, t, embeds, refs = latents(2), torch.tensor([0.3, 0.7]), text_embeds([80, 160]), ref_latents(2, [(2, 2)])
    out = run_layers(pipe.to_layers(), features(x, t, embeds, refs))
    out.square().mean().backward()
    missing = [n for n, p in pipe.diffusion_model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert missing == []


@pytest.mark.parametrize('ref_shapes', [[], [(2, 2), (1, 3)]], ids=['t2i', 'edit'])
def test_layers_do_not_read_their_sizes_back_from_the_device(tmp_path, monkeypatch, ref_shapes):
    """Reading a CUDA tensor waits for every queued kernel. The sizes are known when the first
    layer builds them, so no later layer of the stage reads them back."""
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model())
    layers = pipe.to_layers()
    inputs = features(latents(2), torch.tensor([0.3, 0.7]), text_embeds([80, 160]), ref_latents(2, ref_shapes))
    with torch.no_grad():
        start = run_layers(layers[:2], inputs)
        reads = []
        for name in ('tolist', 'item'):
            original = getattr(torch.Tensor, name)
            monkeypatch.setattr(torch.Tensor, name, lambda t, *a, _o=original, _n=name, **k: (reads.append(_n), _o(t, *a, **k))[1])
        run_layers(layers[2:], start)
    assert reads == []


@pytest.mark.parametrize('ref_shapes', [(), ((2, 2), (1, 3))])
@pytest.mark.parametrize('adapter', [None, lora()], ids=['full', 'lora'])
def test_every_stage_boundary_gets_its_gradients(tmp_path, adapter, ref_shapes):
    """Under pipeline parallelism DeepSpeed asserts that every floating point tensor a stage
    received has a gradient after its backward. Split the layers at every possible point and
    check that, as _exec_send_grads would."""
    pipe = attach(make_pipeline(tmp_path, adapter=adapter, cache_text_embeddings=True), make_model())
    if adapter is not None:
        pipe.configure_adapter(adapter)
    layers = pipe.to_layers()
    inputs = features(latents(2), torch.tensor([0.3, 0.7]), text_embeds([80, 160]), ref_latents(2, ref_shapes))
    with torch.no_grad():
        expected = run_layers(layers, inputs)
    for k in range(1, len(layers)):
        with torch.no_grad():
            sent = run_layers(layers[:k], inputs)
        received = tuple(t.detach().clone().requires_grad_(t.is_floating_point()) for t in sent)
        out = run_layers(layers[k:], received)
        torch.testing.assert_close(out, expected)
        out.square().mean().backward()
        missing = [i for i, t in enumerate(received) if t.is_floating_point() and t.grad is None]
        assert missing == [], f'split before layer {k} ({type(layers[k]).__name__}): inputs {missing} get no gradient'


# ---- text encoder --------------------------------------------------------------------------------

def comfy_conditioning(fake_te, caption):
    """What ComfyUI's Flux3TEModel.encode_token_weights returns for one caption, run over the
    stand-in encoder."""
    inner = comfy.sd1_clip.SDClipModel.__new__(comfy.sd1_clip.SDClipModel)
    nn.Module.__init__(inner)
    inner.special_tokens = {'pad': flux3.TEXT_PAD_ID}
    inner.encode = fake_te.cond_stage_model.qwen3vl_4b.encode
    outer = comfy.text_encoders.flux.Flux3TEModel.__new__(comfy.text_encoders.flux.Flux3TEModel)
    nn.Module.__init__(outer)
    outer.clip_name = outer.clip = 'qwen3vl_4b'
    outer.qwen3vl_4b = inner
    return outer.encode_token_weights(fake_te.tokenize(caption))[0]


def test_text_features_match_comfyui_conditioning(tmp_path):
    pipe = make_pipeline(tmp_path)
    te = pipe.text_encoders[0]
    captions = ['a cat', 'a very long caption ' * 30, '']
    ids, lengths = pipe.tokenize(captions)
    assert lengths.tolist() == [80, 160, 80]
    assert (ids[0, 80:] == flux3.TEXT_PAD_ID).all() and (ids[2, 80:] == flux3.TEXT_PAD_ID).all()
    z = pipe.encode_token_ids(ids)
    for i, caption in enumerate(captions):
        expected = comfy_conditioning(te, caption)
        assert expected.shape == (1, lengths[i], TEXT_LAYERS * TEXT_DIM)
        torch.testing.assert_close(z[i:i + 1, :lengths[i]], expected)
    # set_clip_options / reset_clip_options pair up around the encode.
    assert te.cond_stage_model.options[-1] is None


def test_tokenizing_does_not_load_the_text_encoder(tmp_path):
    pipe = make_pipeline(tmp_path)

    def refuse():
        raise AssertionError('the text encoder was loaded to tokenize')
    pipe.text_encoders[0].load_model_if_needed = refuse
    ids, lengths = pipe.tokenize(['a cat'])
    expected = [t for t, _ in pipe.text_encoders[0].tokenize('a cat')['qwen3vl_4b'][0]]
    assert ids[0].tolist() == expected and lengths.tolist() == [80]


@pytest.mark.parametrize('allow', [False, True])
def test_cudnn_attention_is_off_unless_allowed(tmp_path, monkeypatch, allow):
    calls = []
    monkeypatch.setattr(torch.backends.cuda, 'enable_cudnn_sdp', lambda enabled: calls.append(enabled))
    make_pipeline(tmp_path, allow_cudnn_attention=allow)
    assert calls == ([] if allow else [False])


def test_cached_text_embeddings_are_trimmed(tmp_path):
    pipe = make_pipeline(tmp_path, cache_text_embeddings=True)
    out = pipe.get_call_text_encoder_fn(pipe.text_encoders[0])(['a cat', 'word ' * 120], [False, False])
    assert [e.shape for e in out['text_embeds']] == [(80, 20480), (160, 20480)]


@pytest.mark.parametrize('ref_shapes', [(), ((2, 2),)])
def test_on_the_fly_text_matches_cached(tmp_path, ref_shapes):
    model = make_model()
    on_the_fly = attach(make_pipeline(tmp_path, cache_text_embeddings=False), model)
    assert on_the_fly.get_text_encoders() == []
    cached = attach(make_pipeline(tmp_path, cache_text_embeddings=True), model)
    captions = ['short', 'long ' * 100]
    embeds = cached.get_call_text_encoder_fn(cached.text_encoders[0])(captions, [False, False])['text_embeds']
    x, t, refs = latents(2), torch.tensor([0.3, 0.6]), ref_latents(2, ref_shapes)
    torch.manual_seed(0)
    batch = {'latents': x, 'mask': None, 'caption': captions,
             **{flux3.control_latents_key(i): r for i, r in enumerate(refs)}}
    feats, _ = on_the_fly.prepare_inputs(batch)
    assert not torch.is_floating_point(feats[2])
    torch.manual_seed(0)
    cached_feats, _ = cached.prepare_inputs({**batch, 'text_embeds': embeds})
    with torch.no_grad():
        a = run_layers(on_the_fly.to_layers(), feats)
        b = run_layers(cached.to_layers(), cached_feats)
    torch.testing.assert_close(a, b)


# ---- training inputs -----------------------------------------------------------------------------

def test_prepare_inputs_collects_references_in_order(tmp_path):
    pipe = make_pipeline(tmp_path, cache_text_embeddings=True)
    refs = ref_latents(2, [(2, 2), (3, 3), (1, 1)])
    batch = {'latents': latents(2), 'mask': None, 'text_embeds': text_embeds([80, 80]),
             **{flux3.control_latents_key(i): r for i, r in enumerate(refs)}}
    (noisy, t, text, lengths, *got), (target, mask) = pipe.prepare_inputs(batch)
    assert len(got) == 3 and all(torch.equal(a, b) for a, b in zip(got, refs))
    torch.testing.assert_close(target, (noisy - (1 - t.view(-1, 1, 1, 1)) * batch['latents']) / t.view(-1, 1, 1, 1)
                               - batch['latents'], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('indices', [list(range(11)), [0, 2]], ids=['eleven', 'gap'])
def test_prepare_inputs_refuses_references_it_would_drop(tmp_path, indices):
    pipe = make_pipeline(tmp_path, cache_text_embeddings=True)
    refs = ref_latents(1, [(1, 1)] * len(indices))
    batch = {'latents': latents(1), 'mask': None, 'text_embeds': text_embeds([80]),
             **{flux3.control_latents_key(i): r for i, r in zip(indices, refs)}}
    with pytest.raises(ValueError, match='reference images in order'):
        pipe.prepare_inputs(batch)


def test_vae_fn_keys(tmp_path):
    pipe = make_pipeline(tmp_path)
    fn = pipe.get_call_vae_fn(pipe.vae)
    image = torch.rand(1, 3, 32, 48) * 2 - 1
    assert set(fn(image)) == {'latents'}
    assert set(fn(image, image)) == {'latents', 'control_latents'}
    out = fn(image, [image, image * 0.5, -image])
    assert set(out) == {'latents', 'control_latents', 'control_latents_2', 'control_latents_3'}
    assert out['latents'].shape == (1, 128, 2, 3)
    torch.testing.assert_close(out['control_latents_2'], out['latents'] * 0.5)


@pytest.mark.parametrize('method', ['logit_normal', 'uniform'])
def test_timestep_shift(tmp_path, method):
    pipe = make_pipeline(tmp_path, timestep_sample_method=method, shift=3.0)
    t = pipe.sample_timesteps(4096, 4, 4, 'cpu')
    assert 0 <= t.min() and t.max() <= 1 and t.mean() > 0.6
    flux = make_pipeline(tmp_path, flux_shift=True)
    assert flux.sample_timesteps(8, 64, 64, 'cpu', timestep_quantile=0.5).allclose(
        torch.full((8,), 1 / (1 + math.exp(-flux3.get_lin_function(y1=0.5, y2=1.15)(64 * 64)))))


# ---- adapters and edit isolation -----------------------------------------------------------------

def t2i_and_edit_inputs(batch=2):
    x, t, embeds = latents(batch), torch.tensor([0.3, 0.7][:batch]), text_embeds([80, 160][:batch])
    return features(x, t, embeds), features(x, t, embeds, ref_latents(batch, [(2, 2), (1, 3)]))


@pytest.mark.parametrize('mode, prefixes', [
    ('none', ('content_mode_blocks.', 'txt_mode_blocks.', 'single_blocks.')),
    ('image_cond', flux3.IMAGE_COND_PREFIXES),
    ('reference_tokens', flux3.IMAGE_COND_PREFIXES + ('single_blocks.',)),
])
def test_adapter_targets(tmp_path, mode, prefixes):
    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True, edit_isolation=mode), make_model())
    targets = pipe.get_target_modules(pipe.diffusion_model)
    assert targets and all((name + '.').startswith(prefixes) for name in targets)
    for prefix in prefixes:
        assert any((name + '.').startswith(prefix) for name in targets), prefix
    last = 'single_blocks.2.'
    if mode == 'reference_tokens':
        # Only the keys and values of the last shared block's reference rows reach the image.
        assert sorted(n for n in targets if n.startswith(last)) == [last + 'k_proj', last + 'v_proj']
    elif mode == 'none':
        assert {last + n for n in ('q_proj', 'k_proj', 'v_proj', 'mlp_in', 'attn_out', 'mlp_out')} <= set(targets)
    if mode == 'image_cond':
        assert 'emb_in.image_cond' in targets and 'early_stream_modulations.image_cond.lin' in targets
        assert not any(name.startswith(('single_blocks.', 'emb_in.image', 'txt_in')) and 'image_cond' not in name
                       for name in targets)


def test_text_to_image_datasets_leave_the_reference_blocks_out(tmp_path):
    """Without any reference image the reference blocks never run: an adapter there would never
    train. One being continued with init_from_existing keeps them."""
    def targets(**adapter_kwargs):
        pipe = attach(make_pipeline(tmp_path, adapter=lora(**adapter_kwargs), cache_text_embeddings=True),
                      make_model())
        pipe.model_specific_dataset_config_validation({'directory': [{'path': 'a'}]})
        return pipe, pipe.get_target_modules(pipe.diffusion_model)

    def is_ref(name):
        return name.startswith('content_mode_blocks.image_cond.')

    pipe, plain = targets()
    assert plain and not any(is_ref(n) for n in plain)
    pipe.model_specific_dataset_config_validation({'directory': [{'path': 'b', 'control_path': 'c'}]})
    assert any(is_ref(n) for n in pipe.get_target_modules(pipe.diffusion_model))

    # An adapter trained with references, continued on text-to-image data.
    mixed = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True), make_model())
    mixed.model_specific_dataset_config_validation({'directory': [{'path': 'b', 'control_path': 'c'}]})
    mixed.configure_adapter(lora())
    randomize_adapter(mixed)
    save_dir = tmp_path / 'mixed'
    save_dir.mkdir()
    sd = saved_adapter_state(mixed)
    mixed.save_adapter(save_dir, sd)
    pipe, resumed = targets(init_from_existing=str(save_dir))
    assert any(is_ref(n) for n in resumed)
    pipe.configure_adapter(lora(init_from_existing=str(save_dir)))
    pipe.load_adapter_weights(save_dir)
    for k, v in saved_adapter_state(pipe).items():
        assert torch.equal(v, sd[k]), k


@pytest.mark.parametrize('saved_with_references', [True, False, None], ids=['with', 'without', 'no-save'])
@pytest.mark.parametrize('data_has_references', [True, False])
def test_a_resumed_run_keeps_the_reference_blocks_it_started_with(tmp_path, saved_with_references,
                                                                   data_has_references):
    """The checkpoint's optimizer state holds exactly the adapters the run started with, so a
    resumed run follows its saved adapters even when the data now says otherwise."""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    if saved_with_references is not None:
        first = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True), make_model())
        directory = {'path': 'a', 'control_path': 'c'} if saved_with_references else {'path': 'a'}
        first.model_specific_dataset_config_validation({'directory': [directory]})
        first.configure_adapter(lora())
        (run_dir / 'step5').mkdir()
        first.save_adapter(run_dir / 'step5', saved_adapter_state(first))

    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True), make_model())
    pipe.resume_run_dir = run_dir
    directory = {'path': 'b', 'control_path': 'c'} if data_has_references else {'path': 'b'}
    pipe.model_specific_dataset_config_validation({'directory': [directory]})
    targets = pipe.get_target_modules(pipe.diffusion_model)
    expected = data_has_references if saved_with_references is None else saved_with_references
    assert any(n.startswith('content_mode_blocks.image_cond.') for n in targets) == expected


@pytest.mark.parametrize('adapter', [lora(), lokr()], ids=['lora', 'lokr'])
@pytest.mark.parametrize('mode', ['image_cond', 'reference_tokens'])
def test_isolation_leaves_text_to_image_unchanged(tmp_path, adapter, mode):
    model = make_model()
    pipe = attach(make_pipeline(tmp_path, adapter=adapter, cache_text_embeddings=True, edit_isolation=mode), model)
    t2i, edit = t2i_and_edit_inputs()
    with torch.no_grad():
        base_t2i = run_layers(pipe.to_layers(), t2i)
        base_edit = run_layers(pipe.to_layers(), edit)
    pipe.configure_adapter(adapter)
    randomize_adapter(pipe)
    layers = pipe.to_layers()
    with torch.no_grad():
        assert torch.equal(run_layers(layers, t2i), base_t2i)
        assert not torch.allclose(run_layers(layers, edit), base_edit, atol=1e-4)

    # Every adapter parameter trains on an edit sample, and none on a text-to-image one.
    out = run_layers(layers, edit)
    out.square().mean().backward()
    trainable = {n: p for n, p in pipe.diffusion_model.named_parameters() if p.requires_grad}
    assert trainable and all(('lora_' in n or 'lokr_' in n) for n in trainable)
    dead = [n for n, p in trainable.items() if p.grad is None or not p.grad.abs().sum() > 0]
    assert dead == []
    for p in trainable.values():
        p.grad = None
    run_layers(layers, t2i).square().mean().backward()
    assert all(p.grad is None or not p.grad.any() for p in trainable.values())


def test_reference_tokens_adapter_touches_only_reference_rows(tmp_path):
    """In the shared blocks the image and text rows go through the base weights, the reference
    rows through the adapter."""
    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True,
                                edit_isolation='reference_tokens'), make_model())
    pipe.configure_adapter(lora())
    randomize_adapter(pipe)
    block = pipe.train_blocks()['single'][0]
    linear = block.block.q_proj
    x = torch.randn(2, 10, HIDDEN)
    out = block.linear(linear, x, (3, 7))
    base = linear.get_base_layer()
    torch.testing.assert_close(out[:, :3], base(x[:, :3]))
    torch.testing.assert_close(out[:, 7:], base(x[:, 7:]))
    torch.testing.assert_close(out[:, 3:7], linear(x[:, 3:7]))
    assert not torch.allclose(out[:, 3:7], base(x[:, 3:7]))
    assert torch.equal(block.linear(linear, x, (5, 5)), base(x))


def test_mode_none_adapter_changes_text_to_image(tmp_path):
    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True), make_model())
    t2i, _ = t2i_and_edit_inputs()
    with torch.no_grad():
        before = run_layers(pipe.to_layers(), t2i)
    pipe.configure_adapter(lora())
    randomize_adapter(pipe)
    out = run_layers(pipe.to_layers(), t2i)
    assert not torch.allclose(out, before, atol=1e-4)
    out.square().mean().backward()
    dead = [n for n, p in pipe.diffusion_model.named_parameters()
            if p.requires_grad and 'image_cond' not in n and (p.grad is None or not p.grad.any())]
    assert dead == []


def test_image_cond_full_finetune_freezes_the_rest(tmp_path):
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True, edit_isolation='image_cond'), make_model())
    trainable = [n for n, p in pipe.diffusion_model.named_parameters() if p.requires_grad]
    assert trainable and all(flux3.is_image_cond_name(n) for n in trainable)
    assert any(n.startswith('content_mode_blocks.image_cond.') for n in trainable)
    t2i, edit = t2i_and_edit_inputs()
    run_layers(pipe.to_layers(), edit).square().mean().backward()
    assert all(p.grad is not None and p.grad.any() for n, p in pipe.diffusion_model.named_parameters() if p.requires_grad)


def test_oplora_builds_on_flux3_lora(tmp_path):
    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True), make_model())
    pipe.configure_adapter(lora())
    projector = OPLoRAProjector.build(pipe.diffusion_model, rank=2, exclude_names=pipe.oplora_exclude_names)
    assert 'layers' in projector.describe() or projector.describe()
    projector.project()


def saved_adapter_state(pipe):
    # utils/saver.py: trainable parameters, PEFT's '.default' segment removed.
    return {n.replace('.default', ''): p.detach().clone()
            for n, p in pipe.diffusion_model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize('adapter', [lora(), lokr()], ids=['lora', 'lokr'])
def test_reference_tokens_adapter_round_trip(tmp_path, adapter):
    pipe = attach(make_pipeline(tmp_path, adapter=adapter, cache_text_embeddings=True,
                                edit_isolation='reference_tokens'), make_model())
    pipe.configure_adapter(adapter)
    randomize_adapter(pipe)
    save_dir = tmp_path / 'adapter'
    save_dir.mkdir()
    sd = saved_adapter_state(pipe)
    pipe.save_adapter(save_dir, sd)
    path = save_dir / 'adapter_model.safetensors'
    assert sorted(p.name for p in save_dir.iterdir()) == ['adapter_config.json', 'adapter_model.safetensors']
    saved = safetensors.torch.load_file(str(path))
    with safetensors.safe_open(str(path), 'pt') as f:
        assert f.metadata()[flux3.ADAPTER_METADATA_KEY] == 'reference_tokens'
    single = [k for k in saved if 'single_blocks.' in k]
    assert single and all(k.startswith(flux3.REFERENCE_ROWS_PREFIX + 'single_blocks.') for k in single)
    assert all(k.startswith('diffusion_model.') for k in saved if k not in single)
    assert any(k.startswith('diffusion_model.content_mode_blocks.image_cond.') for k in saved)

    other = attach(make_pipeline(tmp_path, adapter=adapter, cache_text_embeddings=True,
                                 edit_isolation='reference_tokens'), make_model())
    other.configure_adapter(adapter)
    other.load_adapter_weights(save_dir)
    for k, v in saved_adapter_state(other).items():
        assert torch.equal(v, sd[k]), k

    for mode in ('none', 'image_cond'):
        plain = attach(make_pipeline(tmp_path, adapter=adapter, cache_text_embeddings=True, edit_isolation=mode),
                       make_model())
        plain.configure_adapter(adapter)
        with pytest.raises(ValueError, match="trained with edit_isolation = 'reference_tokens'"):
            plain.load_adapter_weights(save_dir)


@pytest.mark.parametrize('suffix', ['.safetensors', '.pt'])
def test_merging_a_reference_tokens_adapter_is_refused(tmp_path, suffix):
    path = tmp_path / f'ref{suffix}'
    sd = {flux3.REFERENCE_ROWS_PREFIX + 'single_blocks.0.q_proj.lora_A.weight': torch.zeros(1)}
    if suffix == '.pt':
        torch.save(sd, str(path))
    else:
        safetensors.torch.save_file(sd, str(path))
    pipe = make_pipeline(tmp_path, merge_adapters=[str(path)])
    with pytest.raises(ValueError, match='reference_tokens'):
        pipe.load_diffusion_model()


# ---- ComfyUI node --------------------------------------------------------------------------------

@pytest.mark.parametrize('adapter', [lora(), lokr()], ids=['lora', 'lokr'])
@pytest.mark.parametrize('fused_qkv', [False, True])
@pytest.mark.parametrize('in_training', [True, False])
def test_comfyui_node_matches_training_forward(tmp_path, adapter, fused_qkv, in_training):
    """The node applied to ComfyUI's model gives what the pipeline trained, with references and
    without them (where it must equal the plain model)."""
    base = make_model(fused_qkv=fused_qkv)
    ckpt = write_checkpoint(tmp_path / 'flux3.safetensors', base)
    pipe = attach(make_pipeline(tmp_path, adapter=adapter, cache_text_embeddings=True,
                                edit_isolation='reference_tokens'), make_model(fused_qkv=fused_qkv))
    pipe.configure_adapter(adapter)
    randomize_adapter(pipe)
    save_dir = tmp_path / 'adapter'
    save_dir.mkdir()
    pipe.save_adapter(save_dir, saved_adapter_state(pipe))
    lora_sd = safetensors.torch.load_file(str(save_dir / 'adapter_model.safetensors'))

    patcher = comfy.sd.load_diffusion_model(str(ckpt), model_options={'dtype': torch.float32})
    assert patcher.model.diffusion_model.single_blocks[0].fused_qkv == fused_qkv
    patched = node.apply_reference_lora(patcher, lora_sd, 1.0)
    patched.patch_model()
    try:
        comfy.model_management.in_training = in_training
        x, t, embeds = latents(1), torch.tensor([0.6]), text_embeds([80])
        refs = ref_latents(1, [(2, 2), (1, 3)])
        dm = patched.model.diffusion_model
        got_edit = comfy_forward(dm, x, t, embeds[0][None], refs)
        got_t2i = comfy_forward(dm, x, t, embeds[0][None])
        plain_t2i = comfy_forward(base, x, t, embeds[0][None])
    finally:
        comfy.model_management.in_training = True
        patched.unpatch_model()
    with torch.no_grad():
        want_edit = run_layers(pipe.to_layers(), features(x, t, embeds, refs))
    tol = {} if in_training else {'rtol': 1e-4, 'atol': 1e-4}
    torch.testing.assert_close(got_edit, want_edit, **tol)
    torch.testing.assert_close(got_t2i, plain_t2i, **tol)


def test_comfyui_node_stacks_two_reference_loras(tmp_path):
    """A reference LoRA loaded after another keeps the first one's shared-block changes: a LoRA
    and its negation (lora_B * -1) cancel out."""
    base = make_model()
    ckpt = write_checkpoint(tmp_path / 'flux3.safetensors', base)
    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True,
                                edit_isolation='reference_tokens'), make_model())
    pipe.configure_adapter(lora())
    randomize_adapter(pipe)
    save_dir = tmp_path / 'adapter'
    save_dir.mkdir()
    pipe.save_adapter(save_dir, saved_adapter_state(pipe))
    sd = safetensors.torch.load_file(str(save_dir / 'adapter_model.safetensors'))
    negated = {k: -v if 'lora_B' in k else v for k, v in sd.items()}
    assert any(k.startswith(flux3.REFERENCE_ROWS_PREFIX) for k in sd)
    x, t, embeds, refs = latents(1), torch.tensor([0.6]), text_embeds([80]), ref_latents(1, [(2, 2)])

    def run(*loras):
        patcher = comfy.sd.load_diffusion_model(str(ckpt), model_options={'dtype': torch.float32})
        for lora_sd in loras:
            patcher = node.apply_reference_lora(patcher, lora_sd, 1.0)
        patcher.patch_model()
        try:
            return comfy_forward(patcher.model.diffusion_model, x, t, embeds[0][None], refs)
        finally:
            patcher.unpatch_model()

    plain = run()
    assert not torch.allclose(run(sd), plain, atol=1e-4)
    torch.testing.assert_close(run(sd, negated), plain, rtol=1e-4, atol=1e-4)


def test_node_splits_keys():
    sd = {'diffusion_model.emb_in.image_cond.lora_A.weight': torch.zeros(1),
          flux3.REFERENCE_ROWS_PREFIX + 'single_blocks.3.mlp_in.lora_B.weight': torch.ones(1)}
    ordinary, rows = node.split_reference_rows(sd)
    assert list(ordinary) == ['diffusion_model.emb_in.image_cond.lora_A.weight']
    assert list(rows) == [3] and list(rows[3]) == ['mlp_in'] and list(rows[3]['mlp_in']) == ['lora_B.weight']
    with pytest.raises(ValueError):
        node.split_reference_rows({flux3.REFERENCE_ROWS_PREFIX + 'single_blocks.0.norm.weight': torch.zeros(1)})
    assert node.REFERENCE_ROWS_PREFIX == flux3.REFERENCE_ROWS_PREFIX


# ---- config ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('overrides, adapter, extra, match', [
    ({'blocks': 3}, lora(), {}, 'blocks'),
    ({'guidance': 3.5}, lora(), {}, 'guidance'),
    ({'timestep_sample_method': 'cosine'}, lora(), {}, 'timestep_sample_method'),
    ({'shift': 3.0, 'flux_shift': True}, lora(), {}, 'shift or flux_shift'),
    ({'edit_isolation': 'text'}, lora(), {}, 'edit_isolation'),
    ({'edit_isolation': 'reference_tokens'}, None, {}, 'full fine-tune'),
    ({'edit_isolation': 'image_cond'}, lora(), {'pipeline_stages': 2}, 'pipeline_stages'),
])
def test_config_refusals(tmp_path, overrides, adapter, extra, match):
    te = tmp_path / 'te.safetensors'
    te.touch()
    config = {'model': {'type': 'flux3', 'diffusion_model': 'x', 'vae': 'v', 'dtype': torch.float32,
                        'text_encoders': [{'path': str(te), 'type': 'flux3'}], **overrides}, **extra}
    if adapter:
        config['adapter'] = adapter
    with pytest.raises(ValueError, match=match):
        flux3.Flux3Pipeline(config)


@pytest.mark.parametrize('te_configs, match', [
    ([], 'exactly one'),
    ([{'path': 'TE', 'type': 'flux3'}, {'path': 'TE', 'type': 'flux3'}], 'exactly one'),
    ([{'path': 'TE', 'type': 'qwen_image'}], "type = 'flux3'"),
    ([{'path': 'TE', 'type': 'flux3', 'llm_layers': 3}], 'llm_layers'),
])
def test_text_encoder_config_refusals(tmp_path, te_configs, match):
    te = tmp_path / 'te.safetensors'
    te.touch()
    te_configs = [{k: (str(te) if v == 'TE' else v) for k, v in c.items()} for c in te_configs]
    config = {'model': {'type': 'flux3', 'diffusion_model': 'x', 'vae': 'v', 'dtype': torch.float32,
                        'text_encoders': te_configs}}
    with pytest.raises(ValueError, match=match):
        flux3.Flux3Pipeline(config)


def test_text_encoder_directory_shards(tmp_path):
    shards = tmp_path / 'text_encoder'
    shards.mkdir()
    with pytest.raises(RuntimeError, match='No .safetensors'):
        flux3.text_embedding_files(shards)
    for name in ('model-00002-of-00002.safetensors', 'model-00001-of-00002.safetensors'):
        (shards / name).touch()
    assert [p.name for p in flux3.text_embedding_files(shards)] == [
        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
    with pytest.raises(RuntimeError, match='does not exist'):
        flux3.text_embedding_files(tmp_path / 'nope')


def test_cache_text_embeddings_defaults_off(tmp_path):
    pipe = make_pipeline(tmp_path)
    assert pipe.cache_text_embeddings is False and pipe.model_config['cache_text_embeddings'] is False


def test_distilled_klein_is_refused(tmp_path):
    with pytest.raises(ValueError, match='Klein Base'):
        attach(make_pipeline(tmp_path), make_model(flow_map=True))


def test_dataset_validation(tmp_path):
    pipe = make_pipeline(tmp_path)
    pipe.model_specific_dataset_config_validation({'directory': [{'path': 'a'}, {'path': 'b', 'control_path': 'c'}]})
    with pytest.raises(ValueError, match='at most 10'):
        pipe.model_specific_dataset_config_validation({'directory': [{'path': 'a', 'control_path': ['c'] * 11}]})
    isolated = make_pipeline(tmp_path, adapter=lora(), edit_isolation='image_cond')
    with pytest.raises(ValueError, match='no control_path'):
        isolated.model_specific_dataset_config_validation({'directory': [{'path': 'a'}]})
    with pytest.raises(ValueError, match='at most 10'):
        dataset_util.validate_control_paths(pipe, {'directory': [{'path': 'a', 'control_path': ['c'] * 11}]})
    with pytest.raises(ValueError, match='empty list'):
        dataset_util.validate_control_paths(pipe, {'directory': [{'path': 'a', 'control_path': []}]})


# ---- block swap ----------------------------------------------------------------------------------

def test_block_swap_split(tmp_path, monkeypatch):
    model = make_model(depth=5, single=6)
    pipe = attach(make_pipeline(tmp_path, adapter=lora(), cache_text_embeddings=True), model)
    created = []

    class FakeOffloader:
        def __init__(self, name, blocks, num_blocks, to_swap, *args, **kwargs):
            created.append((name, num_blocks, to_swap, blocks))

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(flux3, 'ModelOffloader', FakeOffloader)
    monkeypatch.setattr(model, 'to', lambda *a, **k: model)
    monkeypatch.setattr(model.content_mode_blocks['image_cond'], 'to', lambda *a, **k: None)
    pipe.config['reentrant_activation_checkpointing'] = False
    pipe.enable_block_swap(7)
    (single_name, n_single, single_swap, single_blocks), (early_name, n_early, early_swap, early_blocks) = created
    assert (n_single, single_swap, n_early, early_swap) == (6, 4, 10, 3)
    blocks = pipe.train_blocks()
    assert single_blocks is blocks['single'] and early_blocks == blocks['image'] + blocks['txt']
    # The offloaders hook the very wrappers the layers call.
    layers = pipe.to_layers()
    assert [l.block for l in layers if isinstance(l, flux3.SingleLayer)] == blocks['single']
    assert model.single_blocks is not None and model.content_mode_blocks is not None
    with pytest.raises(AssertionError):
        pipe.enable_block_swap(4 + 8 + 1)


# ---- dataset end to end --------------------------------------------------------------------------

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


def make_image_dir(path, count=4, seed=0, captions=True):
    path.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    for i in range(count):
        array = (torch.rand(40, 56, 3, generator=g) * 255).to(torch.uint8).numpy()
        Image.fromarray(array).save(path / f'{i}.png')
        if captions:
            (path / f'{i}.txt').write_text(f'caption number {i}')
    return path


def run_caching(model, ds, dataset_config):
    """DatasetManager.cache without its processes and devices (see test_iris.py)."""
    worker_ds = dataset_util.Dataset(dataset_config, model)
    manager = dataset_util.DatasetManager(model)
    q = queue.Queue()
    errors = []

    def worker():
        try:
            dataset_util._cache_fn([worker_ds], q, model.get_preprocess_media_file_fn(), len(manager.text_encoders),
                                   False, False, 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
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


@pytest.mark.parametrize('n_refs', [0, 1, 3])
@pytest.mark.parametrize('cache_text', [False, True])
def test_dataset_to_layers_end_to_end(tmp_path, local_workers, n_refs, cache_text):
    images = make_image_dir(tmp_path / 'images')
    controls = [make_image_dir(tmp_path / f'control{i}', seed=10 + i, captions=False) for i in range(n_refs)]
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=cache_text), make_model())
    directory = {'path': str(images), 'num_repeats': 1, 'size_buckets': [[48, 32, 1]]}
    if n_refs == 1:
        directory['control_path'] = str(controls[0])
    elif n_refs:
        directory['control_path'] = [str(c) for c in controls]
    dataset_config = {'resolutions': [40], 'enable_ar_bucket': False, 'directory': [directory]}
    ds = dataset_util.Dataset(dataset_config, pipe)
    run_caching(pipe, ds, dataset_config)
    ds.post_init(0, 1, {None: 2}, 1, {None: 2})
    assert len(ds) == 2

    vae_fn = pipe.get_call_vae_fn(pipe.vae)
    preprocess = pipe.get_preprocess_media_file_fn()
    layers = pipe.to_layers()
    for k in range(len(ds)):
        batch = ds[k]
        keys = sorted(k for k in batch if k.startswith('control_latents'))
        assert keys == sorted(flux3.control_latents_key(i) for i in range(n_refs))
        # Each reference is the encoding of its own directory's image, in control_path order.
        for spec, j in zip(batch['image_spec'], range(2)):
            stem = Path(spec[1]).name
            for i, control in enumerate(controls):
                pixels = preprocess((None, str(control / stem)), None, (48, 32, 1))[0][0]
                expected = vae_fn(pixels[None])['latents'][0]
                torch.testing.assert_close(batch[flux3.control_latents_key(i)][j], expected.to(batch['latents'].dtype))
        if cache_text:
            assert 'text_embeds' in batch
        else:
            assert all(c.startswith('caption number') for c in batch['caption'])
        feats, label = pipe.prepare_inputs(batch)
        assert len(feats) == 4 + n_refs
        out = run_layers(layers, feats)
        assert out.shape == (2, 128, 2, 3)
        loss = pipe.get_loss_fn()(out, (label[0], torch.tensor([])))
        assert torch.isfinite(loss)


def test_directories_with_different_references_never_share_a_batch(tmp_path, local_workers):
    """Text-to-image, one-reference and two-reference directories in the same size bucket: a
    batch is collated from its first example's keys, so mixing them would drop references."""
    images = {n: make_image_dir(tmp_path / n, seed=s) for n, s in (('t2i', 0), ('one', 1), ('two', 2))}
    controls = [make_image_dir(tmp_path / f'control{i}', seed=10 + i, captions=False) for i in range(2)]
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model())
    bucket = {'num_repeats': 1, 'size_buckets': [[48, 32, 1]]}
    dataset_config = {'resolutions': [40], 'enable_ar_bucket': False, 'directory': [
        {'path': str(images['t2i']), **bucket},
        {'path': str(images['one']), 'control_path': str(controls[0]), **bucket},
        {'path': str(images['two']), 'control_path': [str(c) for c in controls], **bucket},
    ]}
    ds = dataset_util.Dataset(dataset_config, pipe)
    run_caching(pipe, ds, dataset_config)
    ds.post_init(0, 1, {None: 4}, 1, {None: 4})
    assert len(ds) == 3
    seen = {}
    for k in range(len(ds)):
        batch = ds[k]
        directories = {Path(spec[1]).parent.name for spec in batch['image_spec']}
        assert len(directories) == 1, directories
        refs = sum(key.startswith('control_latents') for key in batch)
        seen[directories.pop()] = refs
        assert len(pipe.prepare_inputs(batch)[0]) == 4 + refs
    assert seen == {'t2i': 0, 'one': 1, 'two': 2}


@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
def test_adding_a_control_directory_to_a_cached_dataset(tmp_path, local_workers, reuse_metadata_cache):
    """A second control directory added after caching: the snapshot cache (default) starts
    over; trusted metadata cached without it is refused rather than silently used."""
    images = make_image_dir(tmp_path / 'images')
    controls = [make_image_dir(tmp_path / f'control{i}', seed=10 + i, captions=False) for i in range(2)]
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model())

    def config(control_path):
        return {'resolutions': [40], 'enable_ar_bucket': False, 'reuse_metadata_cache': reuse_metadata_cache,
                'directory': [{'path': str(images), 'num_repeats': 1, 'size_buckets': [[48, 32, 1]],
                               'control_path': control_path}]}

    first = config(str(controls[0]))
    run_caching(pipe, dataset_util.Dataset(first, pipe), first)
    second = config([str(c) for c in controls])
    ds = dataset_util.Dataset(second, pipe)
    if reuse_metadata_cache:
        run_caching(pipe, ds, second)
        ds.post_init(0, 1, {None: 2}, 1, {None: 2})
        assert all('control_latents_2' in ds[k] for k in range(len(ds)))
    else:
        with pytest.raises(RuntimeError, match='regenerate_cache'):
            ds.cache_metadata(trust_cache=True)


def test_trusted_metadata_with_other_control_directories_is_refused(tmp_path, local_workers):
    """The same number of control directories, swapped: the column check alone would pass and
    train with the old reference order."""
    images = make_image_dir(tmp_path / 'images')
    controls = [make_image_dir(tmp_path / f'control{i}', seed=10 + i, captions=False) for i in range(2)]
    pipe = attach(make_pipeline(tmp_path, cache_text_embeddings=True), make_model())

    def config(control_path):
        return {'resolutions': [40], 'enable_ar_bucket': False, 'reuse_metadata_cache': False,
                'directory': [{'path': str(images), 'num_repeats': 1, 'size_buckets': [[48, 32, 1]],
                               'control_path': control_path}]}

    first = config([str(c) for c in controls])
    run_caching(pipe, dataset_util.Dataset(first, pipe), first)
    swapped = config([str(c) for c in reversed(controls)])
    with pytest.raises(RuntimeError, match='control_file from .*control1.*regenerate_cache'):
        dataset_util.Dataset(swapped, pipe).cache_metadata(trust_cache=True)
    dataset_util.Dataset(first, pipe).cache_metadata(trust_cache=True)


@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
def test_one_image_directory_with_two_control_configurations_in_a_run(tmp_path, reuse_metadata_cache):
    """Training and eval on the same images with different references: with one legacy cache
    per directory they would overwrite each other's caches, so that is refused."""
    images = make_image_dir(tmp_path / 'images')
    controls = [make_image_dir(tmp_path / f'control{i}', seed=10 + i, captions=False) for i in range(2)]
    pipe = make_pipeline(tmp_path, cache_text_embeddings=True)

    def dataset(control_path):
        config = {'resolutions': [40], 'enable_ar_bucket': False, 'reuse_metadata_cache': reuse_metadata_cache,
                  'directory': [{'path': str(images), 'num_repeats': 1, 'control_path': control_path}]}
        return dataset_util.Dataset(config, pipe)

    manager = dataset_util.DatasetManager.__new__(dataset_util.DatasetManager)
    manager.datasets = []
    manager.register(dataset(str(controls[0])))
    manager.register(dataset(str(controls[0])))
    if reuse_metadata_cache:
        manager.register(dataset(str(controls[1])))
    else:
        with pytest.raises(ValueError, match='reuse_metadata_cache = true'):
            manager.register(dataset(str(controls[1])))


def test_collate_refuses_examples_with_different_features(tmp_path):
    pipe = make_pipeline(tmp_path)
    ds = dataset_util.Dataset.__new__(dataset_util.Dataset)
    examples = [{'latents': torch.zeros(1), 'mask': None}, {'latents': torch.zeros(1), 'mask': None,
                                                            'control_latents': torch.zeros(1)}]
    with pytest.raises(RuntimeError, match='different features'):
        ds._collate(examples)


def test_single_control_geometry_unchanged(tmp_path):
    """A single control_path keeps the snapshot geometry it had before lists were accepted."""
    images = make_image_dir(tmp_path / 'images')
    control = make_image_dir(tmp_path / 'control', captions=False)

    def directory(control_path):
        return dataset_util.DirectoryDataset({'path': str(images), 'control_path': control_path, 'reuse_metadata_cache': False},
                                             {'resolutions': [40]}, 'flux3', skip_dataset_validation=True)

    single = directory(str(control))
    geometry = single.snapshot_geometry()
    assert 'extra_control_paths' not in geometry and single.control_columns() == ['control_file']
    assert directory([str(control)]).snapshot_geometry() == geometry
    two = directory([str(control), str(tmp_path / 'images')])
    assert two.control_columns() == ['control_file', 'control_file_2']
    assert two.snapshot_geometry()['extra_control_paths'] == [str(tmp_path / 'images')]
    assert two.snapshot_geometry() != geometry


# ---- example configs -----------------------------------------------------------------------------

@pytest.mark.parametrize('name, dataset_name', [
    ('flux3_klein_base_lora.toml', 'flux3_dataset.toml'),
    ('flux3_klein_base_edit_only.toml', 'flux3_edit_dataset.toml'),
])
def test_example_configs_are_accepted(tmp_path, name, dataset_name):
    import toml
    examples = ROOT / 'examples' / 'flux3'
    config = toml.load(examples / name)
    assert config['dataset'] == f'examples/flux3/{dataset_name}'
    te = tmp_path / 'text_encoder'
    te.mkdir()
    (te / 'model-00001-of-00002.safetensors').touch()
    config['model']['text_encoders'][0]['path'] = str(te)
    config['model']['dtype'] = torch.bfloat16
    config['model']['guidance'] = 1.0  # train.py's default
    config['reentrant_activation_checkpointing'] = False
    pipe = flux3.Flux3Pipeline(config)
    dataset_config = toml.load(examples / dataset_name)
    dataset_util.validate_control_paths(pipe, dataset_config)
    pipe.model_specific_dataset_config_validation(dataset_config)
    if pipe.edit_isolation != 'none':
        # The text-to-image dataset is refused in the isolated modes.
        with pytest.raises(ValueError, match='no control_path'):
            pipe.model_specific_dataset_config_validation(toml.load(examples / 'flux3_dataset.toml'))
