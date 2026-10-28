"""Tests for DoRA support (utils/dora.py, and its use in the saver and the adapter loader).

They run on CPU with a small peft model, no DeepSpeed.
"""

import peft
import pytest
import safetensors.torch
import torch
from torch import nn

import models.base as base
from utils.dora import apply_dora_config_defaults, saved_key, saved_value


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


def make_model(use_dora, seed=0):
    torch.manual_seed(0)
    tiny = Tiny()
    torch.manual_seed(seed)
    config = peft.LoraConfig(r=2, lora_alpha=2, target_modules=['proj'], use_dora=use_dora)
    return peft.get_peft_model(tiny, config)


def save_like_saver(model, path):
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = name.replace('.default', '').replace('.modules_to_save', '')
            state[saved_key(key)] = saved_value(key, p.detach().clone())
    safetensors.torch.save_file(state, str(path / 'adapter_model.safetensors'))
    return state


class TestConfig:
    def test_default_off(self):
        cfg = {'type': 'lora', 'rank': 8}
        apply_dora_config_defaults(cfg)
        assert cfg['use_dora'] is False

    def test_enabled(self):
        cfg = {'type': 'lora', 'rank': 8, 'use_dora': True}
        apply_dora_config_defaults(cfg)
        assert cfg['use_dora'] is True

    def test_must_be_bool(self):
        with pytest.raises(ValueError, match='use_dora'):
            apply_dora_config_defaults({'type': 'lora', 'use_dora': 'yes'})

    def test_float8_base_weights_are_refused(self):
        from utils.dora import check_model_dtypes
        for key in ('transformer_dtype', 'diffusion_model_dtype'):
            with pytest.raises(ValueError, match=f'use_dora cannot be combined with {key}'):
                check_model_dtypes({'use_dora': True}, {key: torch.float8_e4m3fn})
        check_model_dtypes({'use_dora': False}, {'transformer_dtype': torch.float8_e4m3fn})
        check_model_dtypes({'use_dora': True}, {'transformer_dtype': torch.bfloat16})

    def test_refused_for_lokr(self):
        with pytest.raises(ValueError, match="'lokr'"):
            apply_dora_config_defaults({'type': 'lokr', 'use_dora': False})


class TestSavedForm:
    def test_names_and_shapes(self):
        assert saved_key('a.proj.lora_magnitude_vector.weight') == 'a.proj.dora_scale'
        assert saved_key('a.proj.lora_A.weight') == 'a.proj.lora_A.weight'
        assert saved_value('a.proj.lora_magnitude_vector.weight', torch.ones(4)).shape == (4, 1)
        assert saved_value('a.proj.lora_A.weight', torch.ones(2, 6)).shape == (2, 6)

    def test_saved_adapter_has_dora_scale_and_no_magnitude_vector(self, tmp_path):
        state = save_like_saver(make_model(True), tmp_path)
        key = 'base_model.model.proj.dora_scale'
        assert key in state
        assert state[key].shape == (4, 1)
        assert not any('lora_magnitude_vector' in k for k in state)

    def test_plain_lora_has_no_dora_scale(self, tmp_path):
        state = save_like_saver(make_model(False), tmp_path)
        assert not any('dora_scale' in k for k in state)


class TestTraining:
    def test_magnitude_gets_gradient_and_changes(self):
        model = make_model(True)
        before = model.base_model.model.proj.lora_magnitude_vector['default'].weight.detach().clone()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        for _ in range(3):
            loss = model(torch.randn(8, 6)).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        after = model.base_model.model.proj.lora_magnitude_vector['default'].weight
        assert not torch.equal(before, after)


class TestRoundTrip:
    def test_save_then_load_is_exact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, 'is_main_process', lambda: False)
        trained = make_model(True, seed=1)
        with torch.no_grad():
            for p in trained.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
        save_like_saver(trained, tmp_path)
        fresh = make_model(True, seed=2)
        loader().load_adapter_weights_into(fresh, tmp_path)
        x = torch.randn(5, 6)
        assert torch.equal(trained(x), fresh(x))

    def test_plain_lora_into_a_dora_model_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, 'is_main_process', lambda: False)
        save_like_saver(make_model(False, seed=1), tmp_path)
        with pytest.raises(ValueError, match='use_dora = true.*plain LoRA'):
            loader().load_adapter_weights_into(make_model(True, seed=2), tmp_path)

    def test_plain_lora_into_a_plain_model_still_loads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, 'is_main_process', lambda: False)
        trained = make_model(False, seed=1)
        with torch.no_grad():
            for p in trained.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
        save_like_saver(trained, tmp_path)
        fresh = make_model(False, seed=2)
        loader().load_adapter_weights_into(fresh, tmp_path)
        x = torch.randn(5, 6)
        assert torch.equal(trained(x), fresh(x))


def loader():
    cls = [c for c in vars(base).values() if isinstance(c, type) and hasattr(c, 'load_adapter_weights_into')][0]
    return object.__new__(cls)


class TestWithOPLoRA:
    def test_oplora_is_refused(self):
        from utils.oplora import apply_oplora_config_defaults
        with pytest.raises(ValueError, match='oplora cannot be combined with use_dora'):
            apply_oplora_config_defaults({'type': 'lora', 'use_dora': True, 'oplora': True, 'oplora_rank': 4})
        apply_oplora_config_defaults({'type': 'lora', 'use_dora': True, 'oplora': False})

    def test_dora_moves_the_subspace_oplora_protects(self):
        # Why oplora is refused: with the A/B factors projected exactly, the trained magnitude
        # still changes W's top singular block.
        from utils.oplora import OPLoRAProjector
        torch.manual_seed(0)
        tiny = nn.Sequential()
        tiny.proj = nn.Linear(16, 12, bias=False)
        w = tiny.proj.weight.detach().clone()
        model = peft.get_peft_model(
            tiny, peft.LoraConfig(r=4, lora_alpha=4, target_modules=['proj'], use_dora=True))
        projector = OPLoRAProjector.build(model, rank=3, full_svd=True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.05)
        for _ in range(20):
            loss = (model(torch.randn(8, 16)) - 1).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            projector.project()
        assert projector.max_residual() < 1e-4
        merged = model.merge_and_unload().proj.weight.detach()
        u, _, vh = torch.linalg.svd(w)
        block = u[:, :3].T @ (merged - w) @ vh[:3].T
        assert block.abs().max() > 1e-2



def tiny_sdxl(seed=0):
    import diffusers
    import transformers
    from models.sdxl import SDXLPipeline
    torch.manual_seed(seed)
    unet = diffusers.UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16), down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=8, attention_head_dim=2,
        norm_num_groups=4)
    te_config = transformers.CLIPTextConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        vocab_size=32, max_position_embeddings=8, projection_dim=8)
    pipe = object.__new__(SDXLPipeline)
    pipe.unet = unet
    pipe.text_encoder = transformers.CLIPTextModel(te_config)
    pipe.text_encoder_2 = transformers.CLIPTextModelWithProjection(te_config)
    pipe.config = {'adapter': {
        'type': 'lora', 'rank': 2, 'alpha': 2, 'dropout': 0.0, 'dtype': torch.float32, 'use_dora': True}}
    return pipe


class TestSDXL:
    def test_save_and_resume_a_dora_adapter(self, tmp_path):
        pipe = tiny_sdxl()
        pipe.configure_adapter(dict(pipe.config['adapter']))
        sd = {}
        with torch.no_grad():
            for module in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
                for p in module.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p))
                        key = saved_key(p.original_name.replace('.default', ''))
                        sd[key] = saved_value(p.original_name.replace('.default', ''), p.detach().clone())
        pipe.save_adapter(tmp_path, dict(sd))
        saved = safetensors.torch.load_file(str(tmp_path / 'lora.safetensors'))
        scales = [k for k in saved if k.endswith('.dora_scale')]
        downs = [k for k in saved if k.endswith('.lora_down.weight')]
        assert len(scales) == len(downs) > 0
        for k in scales:
            assert k[:-len('.dora_scale')] + '.lora_down.weight' in saved
            assert saved[k].ndim == 2 and saved[k].shape[1] == 1

        fresh = tiny_sdxl()
        fresh.config['adapter']['init_from_existing'] = str(tmp_path)
        fresh.load_adapter_weights(str(tmp_path))
        for module, other in ((pipe.unet, fresh.unet), (pipe.text_encoder, fresh.text_encoder),
                              (pipe.text_encoder_2, fresh.text_encoder_2)):
            a = {n: p for n, p in module.named_parameters() if p.requires_grad}
            b = {n: p for n, p in other.named_parameters() if p.requires_grad}
            assert a.keys() == b.keys()
            for n in a:
                assert torch.equal(a[n], b[n]), n


    def test_plain_lora_with_use_dora_is_refused(self, tmp_path):
        import diffusers
        pipe = tiny_sdxl()
        pipe.config['adapter']['use_dora'] = False
        pipe.configure_adapter(dict(pipe.config['adapter']))
        sd = {p.original_name.replace('.default', ''): p.detach().clone()
              for module in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2)
              for p in module.parameters() if p.requires_grad}
        pipe.save_adapter(tmp_path, sd)
        fresh = tiny_sdxl()
        fresh.config['adapter']['init_from_existing'] = str(tmp_path)
        with pytest.raises(ValueError, match='use_dora = true.*plain LoRA'):
            fresh.load_adapter_weights(str(tmp_path))

    def test_dora_file_without_use_dora_is_refused(self, tmp_path):
        pipe = tiny_sdxl()
        pipe.configure_adapter(dict(pipe.config['adapter']))
        sd = {}
        for module in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
            for p in module.parameters():
                if p.requires_grad:
                    key = p.original_name.replace('.default', '')
                    sd[saved_key(key)] = saved_value(key, p.detach().clone())
        pipe.save_adapter(tmp_path, sd)
        fresh = tiny_sdxl()
        fresh.config['adapter']['use_dora'] = False
        fresh.config['adapter']['init_from_existing'] = str(tmp_path)
        with pytest.raises(ValueError, match='DoRA adapter.*use_dora'):
            fresh.load_adapter_weights(str(tmp_path))


def tiny_flux(seed=0):
    import diffusers
    from models.flux import FluxPipeline as WrappedFluxPipeline
    torch.manual_seed(seed)
    pipe = object.__new__(WrappedFluxPipeline)
    pipe.__dict__['transformer'] = diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=8, pooled_projection_dim=8, axes_dims_rope=[2, 2, 4])
    pipe.__dict__['diffusers_pipeline'] = diffusers.FluxPipeline
    pipe.__dict__['config'] = {'adapter': {
        'type': 'lora', 'rank': 2, 'alpha': 2, 'dropout': 0.0, 'dtype': torch.float32, 'use_dora': True}}
    return pipe


class TestFlux:
    def test_save_and_resume_a_dora_adapter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, 'is_main_process', lambda: False)
        pipe = tiny_flux()
        pipe.configure_adapter(dict(pipe.config['adapter']))
        sd = {}
        with torch.no_grad():
            for name, p in pipe.transformer.named_parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
                    key = p.original_name.replace('.default', '')
                    sd[saved_key(key)] = saved_value(key, p.detach().clone())
        pipe.save_adapter(tmp_path, dict(sd))
        files = list(tmp_path.glob('*.safetensors'))
        saved = safetensors.torch.load_file(str(files[0]))
        assert any(k.endswith('.dora_scale') for k in saved)

        fresh = tiny_flux()
        fresh.config['adapter']['init_from_existing'] = str(tmp_path)
        fresh.load_adapter_weights(str(tmp_path))
        a = {n: p for n, p in pipe.transformer.named_parameters() if p.requires_grad}
        b = {n: p for n, p in fresh.transformer.named_parameters() if p.requires_grad}
        assert a.keys() == b.keys() and a
        for n in a:
            assert torch.equal(a[n], b[n]), n

    def test_a_named_file_loads_with_other_safetensors_beside_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, 'is_main_process', lambda: False)
        pipe = tiny_flux()
        pipe.configure_adapter(dict(pipe.config['adapter']))
        sd = {}
        with torch.no_grad():
            for name, p in pipe.transformer.named_parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
                    key = p.original_name.replace('.default', '')
                    sd[saved_key(key)] = saved_value(key, p.detach().clone())
        pipe.save_adapter(tmp_path, dict(sd))
        file = list(tmp_path.glob('*.safetensors'))[0]
        safetensors.torch.save_file({'other': torch.zeros(1)}, str(tmp_path / 'another_lora.safetensors'))
        fresh = tiny_flux()
        fresh.config['adapter']['init_from_existing'] = str(file)
        fresh.load_adapter_weights(str(file))
        a = {n: p for n, p in pipe.transformer.named_parameters() if p.requires_grad}
        b = {n: p for n, p in fresh.transformer.named_parameters() if p.requires_grad}
        for n in a:
            assert torch.equal(a[n], b[n]), n

    def test_plain_lora_with_use_dora_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base, 'is_main_process', lambda: False)
        pipe = tiny_flux()
        pipe.config['adapter']['use_dora'] = False
        pipe.configure_adapter(dict(pipe.config['adapter']))
        sd = {p.original_name.replace('.default', ''): p.detach().clone()
              for p in pipe.transformer.parameters() if p.requires_grad}
        pipe.save_adapter(tmp_path, sd)
        fresh = tiny_flux()
        fresh.config['adapter']['init_from_existing'] = str(tmp_path)
        with pytest.raises(ValueError, match='use_dora = true.*plain LoRA'):
            fresh.load_adapter_weights(str(tmp_path))


class TestIrisMerge:
    def test_merge_tool_loads_a_dora_adapter(self, tmp_path):
        from tools import iris_merge_adapter
        trained = make_model(True, seed=1)
        with torch.no_grad():
            for p in trained.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
        trained.peft_config['default'].save_pretrained(tmp_path)
        state = {}
        for name, p in trained.named_parameters():
            if p.requires_grad:
                key = name.replace('base_model.model.', '', 1).replace('.default', '')
                state['diffusion_model.' + saved_key(key)] = saved_value(key, p.detach().clone())
        safetensors.torch.save_file(state, str(tmp_path / 'adapter_model.safetensors'))

        torch.manual_seed(0)
        fresh = Tiny()
        peft_model = iris_merge_adapter.apply_adapter(fresh, tmp_path)
        x = torch.randn(5, 6)
        assert torch.allclose(trained(x), peft_model(x), atol=1e-6)

    def test_strength_is_refused_for_dora(self, tmp_path):
        from tools import iris_merge_adapter
        trained = make_model(True)
        trained.peft_config['default'].save_pretrained(tmp_path)
        state = {'diffusion_model.proj.dora_scale': torch.ones(4, 1)}
        safetensors.torch.save_file(state, str(tmp_path / 'adapter_model.safetensors'))
        with pytest.raises(ValueError, match='DoRA'):
            iris_merge_adapter.apply_adapter(Tiny(), tmp_path, strength=0.5)
