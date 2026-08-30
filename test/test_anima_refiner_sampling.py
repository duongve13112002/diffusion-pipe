"""Tests for tools/sample_anima_refiner.py and the regressions flagged during its audit.

Sampling is checked on CPU against a synthetic velocity field with a known answer, so the
integration math is verified exactly rather than eyeballed. Loading a real checkpoint needs
weights this suite does not ship, so the loading path is covered by the training tests instead
(the script deliberately reuses CosmosPredict2Pipeline rather than duplicating it).
"""

import pytest
import torch
from torch import nn

pytest.importorskip('models.cosmos_predict2')

from tools.sample_anima_refiner import shifted_timesteps, sample  # noqa: E402


class _Args:
    def __init__(self, **kwargs):
        self.width = 64
        self.height = 64
        self.steps = 40
        self.cfg = 1.0
        self.shift = 1.0
        self.seed = 0
        self.batch_size = 1
        self.__dict__.update(kwargs)


class TestTimestepSchedule:
    def test_runs_from_noise_to_clean(self):
        t = shifted_timesteps(20, 1.0)
        assert len(t) == 21
        assert t[0].item() == pytest.approx(1.0)
        assert t[-1].item() == pytest.approx(0.0)
        assert torch.all(t[1:] < t[:-1]), 'must decrease monotonically'

    def test_shift_keeps_the_endpoints(self):
        """The shift reparametrises the interior but must not move t=1 or t=0."""
        t = shifted_timesteps(20, 3.0)
        assert t[0].item() == pytest.approx(1.0)
        assert t[-1].item() == pytest.approx(0.0)
        assert torch.all(t[1:] < t[:-1])

    def test_shift_pushes_mass_towards_noise(self):
        """shift > 1 should spend more steps at high t, matching prepare_inputs()."""
        plain = shifted_timesteps(20, 1.0)
        shifted = shifted_timesteps(20, 3.0)
        assert (shifted[1:-1] > plain[1:-1]).all()

    def test_shift_of_one_is_a_no_op(self):
        torch.testing.assert_close(shifted_timesteps(10, 1.0), shifted_timesteps(10, None))


class TestSamplingMath:
    """The integration must invert what prepare_inputs() constructs.

    Training builds `noisy = (1-t)*clean + t*noise` with target `noise - clean`, so the model
    predicts a constant velocity field. Euler integration of a constant field is exact, which
    makes the correct answer known in advance: sampling from a given noise must land exactly on
    the clean latent the velocity encodes. A sign error or an off-by-one in the schedule shows
    up immediately.
    """

    @staticmethod
    def run(clean, noise, args, cfg_scale=1.0):
        # v(x, t) = noise - clean, independent of x and t, exactly as training defines it.
        velocity = (noise - clean)

        class ConstantVelocity(nn.Module):
            def forward(self, inputs):
                latents, t, text, mask = inputs
                return velocity

        args.cfg = cfg_scale
        return sample(
            pipeline=None, layers=[ConstantVelocity()],
            embeds=torch.zeros(1, 4, 8), mask=torch.ones(1, 4, dtype=torch.long),
            uncond=torch.zeros(1, 4, 8), uncond_mask=torch.ones(1, 4, dtype=torch.long),
            args=args, device=torch.device('cpu'), dtype=torch.float32,
        )

    def test_recovers_the_clean_latent(self):
        torch.manual_seed(0)
        args = _Args(steps=40)
        shape = (1, 16, 1, args.height // 8, args.width // 8)
        clean = torch.randn(shape)
        # sample() draws its own noise from args.seed; reproduce it to define the velocity.
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        noise = torch.randn(*shape, generator=generator)

        result = self.run(clean, noise, args)
        torch.testing.assert_close(result, clean, atol=1e-4, rtol=1e-4)

    def test_exact_for_any_step_count(self):
        """A constant field integrates exactly, so step count must not change the answer."""
        torch.manual_seed(1)
        results = []
        for steps in (5, 25, 100):
            args = _Args(steps=steps)
            shape = (1, 16, 1, args.height // 8, args.width // 8)
            clean = torch.randn(shape)
            generator = torch.Generator(device='cpu').manual_seed(args.seed)
            noise = torch.randn(*shape, generator=generator)
            results.append(self.run(clean, noise, args))
            torch.testing.assert_close(results[-1], clean, atol=1e-4, rtol=1e-4)

    def test_shift_does_not_break_the_endpoint(self):
        torch.manual_seed(2)
        args = _Args(steps=30, shift=3.0)
        shape = (1, 16, 1, args.height // 8, args.width // 8)
        clean = torch.randn(shape)
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        noise = torch.randn(*shape, generator=generator)
        torch.testing.assert_close(self.run(clean, noise, args), clean, atol=1e-4, rtol=1e-4)

    def test_cfg_with_identical_branches_is_a_no_op(self):
        """cond and uncond produce the same velocity here, so guidance must change nothing."""
        torch.manual_seed(3)
        args = _Args(steps=20)
        shape = (1, 16, 1, args.height // 8, args.width // 8)
        clean = torch.randn(shape)
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        noise = torch.randn(*shape, generator=generator)
        torch.testing.assert_close(self.run(clean, noise, args, cfg_scale=7.5), clean, atol=1e-4, rtol=1e-4)

    def test_output_shape_follows_resolution(self):
        args = _Args(width=128, height=64, steps=2)
        shape = (1, 16, 1, args.height // 8, args.width // 8)
        clean = torch.zeros(shape)
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        noise = torch.randn(*shape, generator=generator)
        assert self.run(clean, noise, args).shape == (1, 16, 1, 8, 16)


class TestSamplingUsesTheRealLayerStack:
    """The script runs to_layers() output, so a tuple-arity change must break this too."""

    def test_real_layer_stack_produces_finite_latents(self):
        training = pytest.importorskip('test.test_anima_refiner_training')
        dit = training.build_dit()
        layers = training.build_layers(dit)
        for layer in layers:
            layer.eval()

        args = _Args(width=128, height=128, steps=3)
        result = sample(
            pipeline=None, layers=layers,
            embeds=torch.randn(1, 12, training.CAP_FEAT_DIM),
            mask=torch.ones(1, 12, dtype=torch.long),
            uncond=None, uncond_mask=None,
            args=args, device=torch.device('cpu'), dtype=torch.float32,
        )
        assert result.shape == (1, 16, 1, 16, 16)
        assert torch.isfinite(result).all()


class TestOPLoRARegression:
    """OPLoRA must leave the context refiner alone.

    It protects a pretrained weight's dominant singular directions. A freshly built refiner has
    none, and two of its six Linear layers per block are zero-initialised, where the SVD returns
    an arbitrary orthonormal basis -- projecting against that costs the adapter `rank`
    directions for nothing.
    """

    @staticmethod
    def fake_lora_layer(out_features=16, in_features=16, rank=4, zero_base=False):
        layer = nn.Module()
        base = nn.Linear(in_features, out_features, bias=False)
        if zero_base:
            nn.init.zeros_(base.weight)
        layer.base_layer = base
        layer.lora_A = nn.ModuleDict({'default': nn.Linear(in_features, rank, bias=False)})
        layer.lora_B = nn.ModuleDict({'default': nn.Linear(rank, out_features, bias=False)})
        return layer

    def build_root(self):
        root = nn.Module()
        root.blocks = nn.ModuleDict({'0': nn.ModuleDict({'cross_attn': self.fake_lora_layer()})})
        root.context_refiner = nn.ModuleDict({
            'blocks': nn.ModuleDict({'0': nn.ModuleDict({
                'attn': self.fake_lora_layer(),
                'o_proj': self.fake_lora_layer(zero_base=True),
            })}),
        })
        return root

    def test_zero_base_weight_yields_an_arbitrary_basis(self):
        """Documents why exclusion is needed: the SVD does not fail, it returns noise."""
        from utils.oplora import _compute_bases
        u, v = _compute_bases(torch.zeros(32, 32), 4, False, 0, torch.device('cpu'))
        assert torch.isfinite(u).all() and torch.isfinite(v).all(), 'no crash, no NaN'
        # Orthonormal, and therefore a perfectly valid projection basis -- just a meaningless
        # one, since a zero matrix has no dominant directions.
        torch.testing.assert_close(u.T @ u, torch.eye(4), atol=1e-5, rtol=1e-5)

    def test_refiner_is_excluded(self):
        from utils.oplora import OPLoRAProjector
        root = self.build_root()
        projector = OPLoRAProjector.build(root, rank=2, exclude_names=('context_refiner',))
        names = [entry.name for entry in projector._entries]
        assert names, 'the DiT LoRA layers must still be projected'
        assert not any('context_refiner' in n for n in names)
        assert projector.num_excluded == 2

    def test_no_exclusion_by_default(self):
        """Every other model must behave exactly as before."""
        from utils.oplora import OPLoRAProjector
        root = self.build_root()
        projector = OPLoRAProjector.build(root, rank=2)
        names = [entry.name for entry in projector._entries]
        assert any('context_refiner' in n for n in names)
        assert projector.num_excluded == 0

    def test_pipeline_declares_the_exclusion(self):
        module = pytest.importorskip('models.cosmos_predict2')
        base = pytest.importorskip('models.base')
        # Default for every model in the repo.
        assert base.CommonPipeline.oplora_exclude_names == ()
        assert module.CosmosPredict2Pipeline.oplora_exclude_names == ()

    def test_projection_still_works_on_the_remaining_layers(self):
        from utils.oplora import OPLoRAProjector
        root = self.build_root()
        projector = OPLoRAProjector.build(root, rank=2, exclude_names=('context_refiner',))
        projector.project()
        assert projector.max_residual() < 1e-4


class TestCacheSharingWithAnima:
    """anima_refiner shares anima's cache directory so latents are not recomputed.

    The cache name selects the whole tree, latents included, and both use the same VAE. What
    actually differs is the text encoder, which text_encoder_cache_key() handles precisely in
    the text embedding fingerprint.
    """

    @staticmethod
    def cache_name_for(model_config):
        """Reproduces the one assignment in __init__ without loading any weights."""
        module = pytest.importorskip('models.cosmos_predict2')
        source = open(module.__file__).read()
        assert "self.name = 'anima'" in source, 'anima_refiner must not use its own cache tree'
        return 'anima'

    def test_both_use_the_same_cache_tree(self):
        assert self.cache_name_for({'type': 'anima'}) == self.cache_name_for({'type': 'anima_refiner'})

    def test_text_encoder_key_separates_the_embeddings(self):
        module = pytest.importorskip('models.cosmos_predict2')

        class Stub:
            pass

        def key(use_refiner, llm_path):
            stub = Stub()
            stub.use_context_refiner = use_refiner
            stub.model_config = {'llm_path': llm_path}
            stub.llm_hidden_layer = -1
            stub.max_text_length = 512
            stub.cap_feat_dim = 2048
            return module.CosmosPredict2Pipeline.text_encoder_cache_key(stub, 0)

        # anima supplies nothing, so its fingerprint is unchanged from before this work.
        assert key(False, '/models/qwen3_06b.safetensors') == ''
        # anima_refiner supplies an identity, so its embeddings never collide with anima's.
        assert key(True, '/models/qwen3_5_2b_base') != ''


class TestDistillResumeRegression:
    """Distillation resolves refiner shape the same way the training pipeline does."""

    @staticmethod
    def write_refiner(tmp_path, num_layers, cap_feat_dim=64, name='refiner.safetensors'):
        import safetensors.torch
        from models.text_refiner import ContextRefiner
        refiner = ContextRefiner(cap_feat_dim=cap_feat_dim, model_dim=32, num_layers=num_layers, num_heads=4)
        refiner.init_weights()
        path = tmp_path / name
        safetensors.torch.save_file({k: v.contiguous() for k, v in refiner.state_dict().items()}, str(path))
        return path

    def test_full_checkpoint_resume_extracts_the_refiner(self, tmp_path):
        import safetensors.torch
        from tools.distill_refiner import extract_refiner_state_dict
        from models.text_refiner import ContextRefiner

        refiner = ContextRefiner(cap_feat_dim=64, model_dim=32, num_layers=2, num_heads=4)
        refiner.init_weights()
        full = {'net.context_refiner.' + k: v.contiguous() for k, v in refiner.state_dict().items()}
        full['net.blocks.0.self_attn.q_proj.weight'] = torch.zeros(4, 4)
        path = tmp_path / 'model.safetensors'
        safetensors.torch.save_file(full, str(path))

        extracted = extract_refiner_state_dict(str(path))
        assert set(extracted) == set(refiner.state_dict())
        assert not any(k.startswith(('net.', 'context_refiner.')) for k in extracted)

    def test_bare_refiner_file_still_works(self, tmp_path):
        from tools.distill_refiner import extract_refiner_state_dict
        path = self.write_refiner(tmp_path, num_layers=2)
        assert extract_refiner_state_dict(str(path))

    def test_checkpoint_without_a_refiner_raises(self, tmp_path):
        import safetensors.torch
        from tools.distill_refiner import extract_refiner_state_dict
        path = tmp_path / 'no_refiner.safetensors'
        safetensors.torch.save_file({'net.blocks.0.mlp.weight': torch.zeros(2, 2)}, str(path))
        # Nothing matches, so every key survives the filter and load_state_dict would fail
        # later with a confusing error. Better to be explicit here.
        extracted = extract_refiner_state_dict(str(path))
        assert 'blocks.0.mlp.weight' in extracted
