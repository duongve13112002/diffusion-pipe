import inspect
import types

import pytest
import torch

from utils import vae_dtype as vd


def dtypes(**model_config):
    return vd.LatentDtypes(model_config)


def test_parse_and_defaults():
    assert not dtypes().active
    assert dtypes().store is None
    assert dtypes(vae_dtype='fp16').encode == torch.float16
    assert dtypes(vae_dtype='fp16').store == torch.float16
    both = dtypes(vae_dtype='float32', latent_cache_dtype='bf16')
    assert (both.encode, both.cache, both.store) == (torch.float32, torch.bfloat16, torch.bfloat16)
    only_cache = dtypes(latent_cache_dtype='float16')
    assert only_cache.active and only_cache.encode is None and only_cache.store == torch.float16
    assert vd.LatentDtypes({}, default_encode='float16').encode == torch.float16
    assert vd.LatentDtypes({'vae_dtype': 'float32'}, default_encode='float16').encode == torch.float32


@pytest.mark.parametrize('config', [{'vae_dtype': 'float8'}, {'latent_cache_dtype': 'half'}, {'vae_dtype': 16}])
def test_parse_rejects_unknown_dtypes(config):
    with pytest.raises(ValueError, match='must be one of'):
        dtypes(**config)


def test_cache_key():
    assert dtypes().cache_key() == ''
    assert dtypes(vae_dtype='float32').cache_key() == '|vae_dtype=float32'
    assert dtypes(vae_dtype='float32').cache_key(float32_is_default=True) == ''
    assert dtypes(vae_dtype='float16').cache_key(float32_is_default=True) == '|vae_dtype=float16'
    assert dtypes(vae_dtype='float16', latent_cache_dtype='float16').cache_key() == '|vae_dtype=float16'
    assert dtypes(vae_dtype='float32', latent_cache_dtype='bfloat16').cache_key(float32_is_default=True) == \
        '|latent_cache_dtype=bfloat16'
    assert dtypes(latent_cache_dtype='float16').cache_key() == '|latent_cache_dtype=float16'
    # Storing float32 after a float16 encode is not the default float16 cache, even for a model
    # whose float32 caches had no suffix.
    assert dtypes(vae_dtype='float16', latent_cache_dtype='float32').cache_key(float32_is_default=True) == \
        '|vae_dtype=float16|latent_cache_dtype=float32'
    # The same pair in another order is another cache.
    assert dtypes(vae_dtype='float16', latent_cache_dtype='bfloat16').cache_key() != \
        dtypes(vae_dtype='bfloat16', latent_cache_dtype='float16').cache_key()


def test_refuse_names_the_keys():
    vd.refuse({'dtype': 'bfloat16'}, 'Iris', 'it caches pixels')
    with pytest.raises(ValueError, match='vae_dtype, latent_cache_dtype.*Iris.*caches pixels'):
        vd.refuse({'vae_dtype': 'float16', 'latent_cache_dtype': 'float16'}, 'Iris', 'it caches pixels')


def fake_fn(calls, overflow_at=1e4):
    """Encodes `pixels * 1000` in float16 (overflowing for big pixels), like a VAE running in float16."""
    def fn(pixels, control=None, audio=None):
        calls.append((pixels.clone(), control, audio))
        return {'latents': (pixels * 1000).to(torch.float16), 'other': 'kept'}
    return fn


def fallback_for(calls, tried):
    def make(dtype):
        tried.append(dtype)

        def fn(pixels, control=None, audio=None):
            calls.append((pixels.clone(), control, audio))
            return {'latents': (pixels * 1000).to(dtype), 'other': 'kept'}
        return fn
    return make


def test_store_only_rounds_to_the_cache_dtype():
    wrapped = vd.wrap_call_vae_fn(lambda x: {'latents': x.float() * 1.0001, 'meta': 1}, dtypes(latent_cache_dtype='bf16'),
                                  ('latents',), lambda dtype: pytest.fail('no encoding again without vae_dtype'))
    out = wrapped(torch.ones(2, 3, 4))
    assert out['latents'].dtype == torch.bfloat16 and out['meta'] == 1
    assert torch.equal(out['latents'], (torch.ones(2, 3, 4) * 1.0001).to(torch.bfloat16))


def test_non_finite_rows_are_encoded_again_in_a_wider_dtype():
    calls, again, tried = [], [], []
    wrapped = vd.wrap_call_vae_fn(fake_fn(calls), dtypes(vae_dtype='float16'), ('latents',), fallback_for(again, tried))
    pixels = torch.tensor([0.5, 100.0, 0.25, 90.0]).view(4, 1, 1, 1)
    audio = ['a', 'b', 'c', 'd']
    out = wrapped(pixels, audio=audio)
    assert tried == [torch.bfloat16]
    # Only the two overflowing rows (and their audio) went through the fallback.
    assert torch.equal(again[0][0].flatten(), torch.tensor([100.0, 90.0]))
    assert again[0][2] == ['b', 'd']
    assert out['other'] == 'kept'
    # 100000 does not fit in float16, so the batch is cached in float32 (the storage warning). The
    # two rows that were encoded again hold what bfloat16 gave.
    assert out['latents'].dtype == torch.float32
    rows = torch.tensor([100000.0, 90000.0]).to(torch.bfloat16).float()
    assert torch.equal(out['latents'].flatten(), torch.stack([torch.tensor(500.0), rows[0], torch.tensor(250.0), rows[1]]))


def test_rows_that_fit_stay_in_the_storage_dtype():
    calls, again, tried = [], [], []
    wrapped = vd.wrap_call_vae_fn(fake_fn(calls), dtypes(vae_dtype='float16', latent_cache_dtype='bfloat16'),
                                  ('latents',), fallback_for(again, tried))
    out = wrapped(torch.tensor([0.5, 100.0]).view(2, 1, 1, 1))
    assert out['latents'].dtype == torch.bfloat16
    assert torch.equal(out['latents'].flatten().float(), torch.tensor([500.0, 100000.0]).to(torch.bfloat16).float())


def test_all_rows_bad_is_encoded_again_as_one_batch_and_walks_the_chain():
    again, tried = [], []

    def fn(pixels):
        return {'latents': torch.full((len(pixels), 2), float('inf'))}

    def make(dtype):
        tried.append(dtype)
        return lambda pixels: {'latents': torch.full((len(pixels), 2), float('inf') if dtype == torch.bfloat16 else 7.0)}

    wrapped = vd.wrap_call_vae_fn(fn, dtypes(vae_dtype='float16'), ('latents',), make)
    out = wrapped(torch.zeros(3, 1))
    assert tried == [torch.bfloat16, torch.float32]
    assert out['latents'].dtype == torch.float16 and (out['latents'] == 7.0).all()


def test_non_finite_in_the_widest_dtype_raises():
    wrapped = vd.wrap_call_vae_fn(lambda x: {'latents': torch.full((len(x), 2), float('nan'))},
                                  dtypes(vae_dtype='float32'), ('latents',), lambda dtype: pytest.fail('nothing wider'))
    with pytest.raises(RuntimeError, match='non-finite latents even in float32'):
        wrapped(torch.zeros(2, 1))


def test_non_finite_without_vae_dtype_is_not_ours_to_fix():
    wrapped = vd.wrap_call_vae_fn(lambda x: {'latents': torch.full((len(x), 2), float('nan'))},
                                  dtypes(latent_cache_dtype='float16'), ('latents',), lambda dtype: pytest.fail('no'))
    assert torch.isnan(wrapped(torch.zeros(2, 1))['latents']).all()


def test_every_key_is_cast_and_checked_and_others_untouched():
    again, tried = [], []

    def fn(x, control=None):
        return {'latents': torch.ones(len(x), 2), 'control_latents': torch.ones(len(x), 2) * torch.tensor([[1.0], [float('inf')]]),
                'clip_context': torch.ones(len(x), 3, dtype=torch.float32)}

    def make(dtype):
        tried.append(dtype)
        return lambda x, control=None: {'latents': torch.ones(len(x), 2) * 2, 'control_latents': torch.ones(len(x), 2) * 3,
                                        'clip_context': torch.zeros(len(x), 3)}

    wrapped = vd.wrap_call_vae_fn(fn, dtypes(vae_dtype='bfloat16'), ('latents', 'control_latents'), make)
    out = wrapped(torch.zeros(2, 1), torch.zeros(2, 1))
    assert out['latents'][0].eq(1).all() and out['latents'][1].eq(2).all()
    assert out['control_latents'][0].eq(1).all() and out['control_latents'][1].eq(3).all()
    assert out['latents'].dtype == out['control_latents'].dtype == torch.bfloat16
    # Not a VAE output: left as it came, and not replaced by the fallback's.
    assert out['clip_context'].dtype == torch.float32 and out['clip_context'].eq(1).all()


def test_signature_is_kept_for_the_audio_check():
    wrapped = vd.wrap_call_vae_fn(fake_fn([]), dtypes(vae_dtype='float32'), ('latents',), lambda dtype: None)
    assert 'audio' in inspect.signature(wrapped).parameters


class FakePipeline:
    """CommonPipeline on a bare object: the shared vae_dtype plumbing of every model."""

    @staticmethod
    def make(model_config, **attrs):
        from models.base import CommonPipeline

        class Pipeline(CommonPipeline):
            vae_config_keys = attrs.pop('vae_config_keys', ())

            def get_call_vae_fn(self, vae):
                return lambda x: {'latents': x}

        for name, value in attrs.items():
            setattr(Pipeline, name, value)
        pipe = Pipeline.__new__(Pipeline)
        pipe.model_config = model_config
        return pipe


def test_a_config_without_the_keys_changes_nothing():
    pipe = FakePipeline.make({'dtype': 'bfloat16'}, vae_config_keys=('vae_path',))
    assert pipe.vae_cache_key() == '|bfloat16'
    assert not pipe.vae_identity_required()
    assert not pipe.vae_dtypes.active
    fn = pipe.get_cached_vae_fn(None)
    assert not hasattr(fn, '__wrapped__') and fn(5) == {'latents': 5}


def test_a_model_without_vae_config_keys_records_the_dtypes_and_requires_them():
    # No identity before: recorded only when a key is set, and then a cache that records none
    # (every cache made before the keys existed) is not reused.
    assert FakePipeline.make({}).vae_cache_key() == ''
    pipe = FakePipeline.make({'latent_cache_dtype': 'float16'})
    assert pipe.vae_cache_key() == '|latent_cache_dtype=float16'
    assert pipe.vae_identity_required()
    assert FakePipeline.make({'vae_dtype': 'float32'}).vae_identity_required()


def test_the_cache_key_follows_the_settings():
    keys = {FakePipeline.make({'vae_config_keys': 1, **cfg}, vae_config_keys=('vae_path',)).vae_cache_key()
            for cfg in ({}, {'vae_dtype': 'float32'}, {'vae_dtype': 'float16'}, {'latent_cache_dtype': 'float16'},
                        {'vae_dtype': 'float32', 'latent_cache_dtype': 'float16'})}
    assert len(keys) == 5


def test_float32_is_the_default_only_for_a_model_that_says_so():
    plain = FakePipeline.make({'vae_dtype': 'float32'}, vae_config_keys=('vae_path',))
    assert plain.vae_cache_key().endswith('|vae_dtype=float32')
    legacy = FakePipeline.make({'vae_dtype': 'float32'}, vae_config_keys=('vae_path',), vae_float32_is_default=True)
    assert 'vae_dtype' not in legacy.vae_cache_key()


def test_a_model_without_latents_refuses_the_keys_naming_them():
    pipe = FakePipeline.make({'vae_dtype': 'float16', 'dtype': 'bfloat16'}, vae_dtype_refusal='it caches pixels')
    with pytest.raises(ValueError, match='vae_dtype in .model. do not apply to Pipeline: it caches pixels'):
        pipe.vae_dtypes
    FakePipeline.make({'dtype': 'bfloat16'}, vae_dtype_refusal='it caches pixels').vae_dtypes


def test_a_wrong_value_is_refused_by_the_property():
    with pytest.raises(ValueError, match='latent_cache_dtype must be one of'):
        FakePipeline.make({'latent_cache_dtype': 'float8'}).vae_dtypes


def test_the_fallback_vae_is_loaded_once_per_dtype():
    loaded = []
    pipe = FakePipeline.make({'vae_dtype': 'float16'})
    pipe.load_vae = lambda dtype: loaded.append(dtype) or types.SimpleNamespace(dtype=dtype)
    first = pipe.fallback_call_vae_fn(torch.float32)
    assert pipe.fallback_call_vae_fn(torch.float32) is first
    pipe.fallback_call_vae_fn(torch.bfloat16)
    assert loaded == [torch.float32, torch.bfloat16]


def test_pixels_reach_a_comfyui_vae_in_float32_once_vae_dtype_is_set():
    from models.base import ComfyPipeline
    seen = []

    class Images:
        def to(self, device, dtype):
            seen.append(dtype)
            return torch.zeros(1, 3, 8, 8)

    for config, expected in (({}, torch.bfloat16), ({'vae_dtype': 'float16'}, torch.float32),
                             ({'latent_cache_dtype': 'float16'}, torch.bfloat16)):
        pipe = ComfyPipeline.__new__(ComfyPipeline)
        pipe.model_config, pipe.dtype = config, torch.bfloat16
        pipe.latent_format = types.SimpleNamespace(process_in=lambda x: x)
        vae = types.SimpleNamespace(encode=lambda pixels: pixels)
        pipe.get_call_vae_fn(vae)(Images())
        assert seen[-1] == expected


def test_sdxl_loads_its_vae_from_the_checkpoint_at_full_precision(tmp_path):
    import diffusers
    from safetensors.torch import save_file
    from models import sdxl

    # SDXL's layout (the conversion map is written for four blocks of two resnets), made tiny.
    config = dict(in_channels=3, out_channels=3, down_block_types=['DownEncoderBlock2D'] * 4,
                  up_block_types=['UpDecoderBlock2D'] * 4, block_out_channels=[4, 4, 8, 8], layers_per_block=2,
                  latent_channels=4, norm_num_groups=2, sample_size=32, scaling_factor=0.13025)
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKL(**config)
    ldm = {'first_stage_model.' + k: v.contiguous() for k, v in sdxl.convert_vae_state_dict(vae.state_dict()).items()}
    ldm['model.diffusion_model.unrelated'] = torch.zeros(3)
    path = tmp_path / 'checkpoint.safetensors'
    save_file(ldm, str(path))
    pipe = types.SimpleNamespace(model_config={'checkpoint_path': str(path)}, diffusers_pipeline=types.SimpleNamespace(vae=vae))
    loaded = sdxl.SDXLPipeline.load_vae(pipe, torch.float32)
    assert loaded.dtype == torch.float32 and not loaded.training
    for name, value in vae.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    assert sdxl.SDXLPipeline.load_vae(pipe, torch.bfloat16).dtype == torch.bfloat16
    assert loaded.config.scaling_factor == 0.13025


from models.base import CommonPipeline  # noqa: E402
from test.test_cached_only import CountingModel, build  # noqa: E402
from test.test_text_encoding_on_the_fly import CAPTIONS, local_workers, make_images  # noqa: E402,F401

REUSE = pytest.mark.parametrize('reuse_metadata_cache', [True, False])


class DtypeModel(CountingModel):
    """The fake dataset model with the real vae_dtype / latent_cache_dtype plumbing."""
    vae_config_keys = ('vae_path',)
    vae_float32_is_default = False
    default_vae_dtype = None
    vae_latent_keys = ('latents',)
    vae_dtype_refusal = None
    vae_dtypes = CommonPipeline.vae_dtypes
    vae_cache_key = CommonPipeline.vae_cache_key
    vae_identity_required = CommonPipeline.vae_identity_required
    get_cached_vae_fn = CommonPipeline.get_cached_vae_fn
    fallback_call_vae_fn = CommonPipeline.fallback_call_vae_fn

    def __init__(self, **model_config):
        super().__init__()
        self.model_config = {'vae_path': 'vae.safetensors', 'dtype': 'bfloat16', **model_config}


def latents_of(ds):
    ds.post_init(0, 1, {None: 1}, 1, {None: 1})
    return [ds[k]['latents'] for k in range(len(ds))]


@REUSE
def test_the_cache_holds_the_latent_cache_dtype(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', CAPTIONS)
    ds, _ = build(images, model=DtypeModel(latent_cache_dtype='float16'),
                  reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    latents = latents_of(ds)
    assert len(latents) == len(CAPTIONS) and all(x.dtype == torch.float16 for x in latents)


@REUSE
def test_a_cache_made_without_the_keys_is_rebuilt_once_they_are_set(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', CAPTIONS)
    _, plain = build(images, model=DtypeModel(), reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert plain.vae_calls > 0
    # Nothing set: the cache is found again and nothing is encoded.
    _, again = build(images, model=DtypeModel(), reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert again.vae_calls == 0
    # The cache records no dtype, so it is not reused for latents in float16.
    ds, rebuilt = build(images, model=DtypeModel(latent_cache_dtype='float16'),
                        reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert rebuilt.vae_calls > 0
    assert all(x.dtype == torch.float16 for x in latents_of(ds))
    _, again = build(images, model=DtypeModel(latent_cache_dtype='float16'),
                     reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert again.vae_calls == 0
    # Another dtype is another cache.
    _, other = build(images, model=DtypeModel(latent_cache_dtype='bfloat16'),
                     reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert other.vae_calls > 0


def test_going_back_to_the_unset_cache_finds_it_with_reused_metadata(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS)
    build(images, model=DtypeModel(), reuse_metadata_cache=True, restrict=False)
    build(images, model=DtypeModel(latent_cache_dtype='float16'), reuse_metadata_cache=True, restrict=False)
    _, back = build(images, model=DtypeModel(), reuse_metadata_cache=True, restrict=False)
    assert back.vae_calls == 0


def test_every_latent_a_model_caches_is_checked():
    """The entries a get_call_vae_fn returns that the VAE produced must be listed, else an
    overflowing control image is cached non-finite."""
    from models.flux import FluxPipeline
    from models.flux2 import Flux2Pipeline
    from models.qwen_image import QwenImagePipeline
    for cls in (FluxPipeline, Flux2Pipeline, QwenImagePipeline):
        assert 'control_latents' in cls.vae_latent_keys


class NoConfigKeysModel(DtypeModel):
    """A model that declares nothing about its VAE: its caches record no identity at all."""
    vae_config_keys = ()


@REUSE
def test_a_cache_with_no_recorded_identity_is_rebuilt_only_once_a_key_is_set(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', CAPTIONS)
    _, plain = build(images, model=NoConfigKeysModel(), reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert plain.vae_calls > 0
    _, again = build(images, model=NoConfigKeysModel(), reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert again.vae_calls == 0
    ds, rebuilt = build(images, model=NoConfigKeysModel(latent_cache_dtype='float16'),
                        reuse_metadata_cache=reuse_metadata_cache, restrict=False)
    assert rebuilt.vae_calls > 0
    assert all(x.dtype == torch.float16 for x in latents_of(ds))
