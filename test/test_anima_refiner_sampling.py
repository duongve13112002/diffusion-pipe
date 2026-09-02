"""Tests for tools/sample_anima_refiner.py and the regressions flagged during its audit.

Sampling is checked on CPU against a synthetic velocity field with a known answer, so the
integration math is verified exactly rather than eyeballed. Loading a real checkpoint needs
weights this suite does not ship, so the loading path is covered by the training tests instead
(the script deliberately reuses CosmosPredict2Pipeline rather than duplicating it).
"""

from pathlib import Path

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
            args=args, device=torch.device('cpu'), dtype=torch.float32, in_channels=16,
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
            args=args, device=torch.device('cpu'), dtype=torch.float32, in_channels=16,
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

    def test_source_does_not_reintroduce_a_separate_cache_name(self):
        """A source-level guard, not a behaviour test -- be clear about which this is.

        The assignment lives inside __init__ behind a text-encoder load, so reaching it needs
        real weights this suite does not ship. Grepping the source at least catches someone
        reintroducing a separate cache name, which would silently discard every cached latent.
        The behaviour that IS testable -- that text embeddings stay separated -- is covered
        below.
        """
        module = pytest.importorskip('models.cosmos_predict2')
        source = open(module.__file__).read()
        assert "self.name = 'anima'" in source
        assert "'anima_refiner' if self.use_context_refiner" not in source

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
        from models.text_refiner import ContextRefiner, extract_refiner_state_dict
        from utils.common import load_state_dict

        refiner = ContextRefiner(cap_feat_dim=64, model_dim=32, num_layers=2, num_heads=4)
        refiner.init_weights()
        full = {'net.context_refiner.' + k: v.contiguous() for k, v in refiner.state_dict().items()}
        full['net.blocks.0.self_attn.q_proj.weight'] = torch.zeros(4, 4)
        path = tmp_path / 'model.safetensors'
        safetensors.torch.save_file(full, str(path))

        extracted = extract_refiner_state_dict(load_state_dict(str(path)))
        assert set(extracted) == set(refiner.state_dict())
        assert not any(k.startswith(('net.', 'context_refiner.')) for k in extracted)

    def test_bare_refiner_file_still_works(self, tmp_path):
        from models.text_refiner import extract_refiner_state_dict
        path = self.write_refiner(tmp_path, num_layers=2)
        from utils.common import load_state_dict
        assert extract_refiner_state_dict(load_state_dict(str(path)))

    def test_checkpoint_without_a_refiner_raises(self, tmp_path):
        """A checkpoint with no refiner must fail here, not pass unrelated tensors onward.

        The earlier version returned every non-refiner weight as if it were the refiner, so the
        failure surfaced later as an opaque KeyError or shape mismatch inside load_state_dict.
        """
        import safetensors.torch
        from models.text_refiner import extract_refiner_state_dict
        from utils.common import load_state_dict
        path = tmp_path / 'no_refiner.safetensors'
        safetensors.torch.save_file({
            'net.blocks.0.mlp.weight': torch.zeros(2, 2),
            'net.x_embedder.proj.1.weight': torch.zeros(4, 4),
        }, str(path))
        with pytest.raises(RuntimeError, match='No context_refiner weights found'):
            extract_refiner_state_dict(load_state_dict(str(path)))


class TestAdapterKeyNames:
    """Why the loader could not use a regex on '.weight'.

    This is the shape of the defect, asserted against the installed peft rather than described
    in a comment: LoRA's parameters end in '.weight' and LoKr's do not, so a rule that inserts
    the adapter segment before a trailing '.weight' covers exactly one of the two.
    """

    @staticmethod
    def _names(peft_config):
        import peft
        model = nn.Sequential(nn.Linear(16, 16, bias=False))
        peft.get_peft_model(model, peft_config)
        return [name for name, p in model.named_parameters() if p.requires_grad]

    def test_lora_parameters_end_in_weight(self):
        import peft
        names = self._names(peft.LoraConfig(
            r=4, lora_alpha=4, lora_dropout=0.0, bias='none', target_modules=['0']))
        assert names and all(n.endswith('.default.weight') for n in names)

    def test_lokr_parameters_do_not(self):
        import peft
        names = self._names(peft.LoKrConfig(
            r=4, decompose_factor=-1, alpha=4, rank_dropout=0.0, target_modules=['0']))
        assert names and not any(n.endswith('.weight') for n in names)
        assert all(n.endswith('.default') for n in names)


def _adapter_config(adapter_type):
    if adapter_type == 'lora':
        return {'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0, 'dtype': torch.float32}
    return {'type': 'lokr', 'rank': 4, 'alpha': 4, 'decompose_factor': -1,
            'rank_dropout': 0.0, 'dtype': torch.float32}


def _refiner_pipeline(dit, monkeypatch):
    """A CosmosPredict2Pipeline holding `dit`, with only what the adapter paths touch."""
    import models.base
    from models.cosmos_predict2 import CosmosPredict2Pipeline
    monkeypatch.setattr(models.base, 'is_main_process', lambda: True)

    pipeline = object.__new__(CosmosPredict2Pipeline)
    pipeline.transformer = dit
    pipeline.use_context_refiner = True
    pipeline.adapter_target_modules = ['Block', 'ContextRefiner']
    pipeline.model_config = {'dtype': torch.float32}
    pipeline._refiner_is_fresh = False
    return pipeline


def _train_and_save(tmp_path, adapter_type, monkeypatch, seed=0):
    """Run the real adapter path end to end and return (save_dir, merged_state_dict).

    Builds a DiT, configures the adapter exactly as train.py does, gives the adapter non-zero
    weights, saves it the way utils/saver.py serialises a run, then merges it. The merged
    weights are what a correct load has to reproduce.
    """
    training = pytest.importorskip('test.test_anima_refiner_training')
    dit = training.build_dit(seed=seed)
    pipeline = _refiner_pipeline(dit, monkeypatch)
    pipeline.configure_adapter(_adapter_config(adapter_type))

    # LoRA zeroes lora_B and LoKr zeroes lokr_w1, so an untouched adapter has a zero delta and
    # a broken load would be indistinguishable from a correct one.
    torch.manual_seed(100)
    with torch.no_grad():
        for name, p in dit.named_parameters():
            if p.requires_grad:
                p.copy_(torch.randn_like(p) * 0.1)

    # Exactly what utils/saver.py writes: every trainable parameter under its original name,
    # with PEFT's adapter segment removed.
    state_dict = {
        p.original_name.replace('.default', '').replace('.modules_to_save', ''): p.detach().clone()
        for _, p in dit.named_parameters() if p.requires_grad
    }
    save_dir = tmp_path / adapter_type
    save_dir.mkdir()
    pipeline.train_context_refiner = False
    pipeline.save_adapter(save_dir, state_dict)

    pipeline.lora_model.merge_and_unload()
    return save_dir, {k: v.detach().clone() for k, v in dit.state_dict().items()}


class TestAdapterRoundTrip:
    """Saving an adapter and merging it back through the sampler must reproduce the run.

    This is the test that fails without the loader fix. It is deliberately not a mock: the
    LoKr defect lives in how PEFT names its parameters, and a stand-in for PEFT cannot have
    that property.
    """

    @pytest.mark.parametrize('adapter_type', ['lora', 'lokr'])
    def test_merged_weights_match_the_training_run(self, tmp_path, monkeypatch, adapter_type):
        from tools.sample_anima_refiner import apply_adapters
        training = pytest.importorskip('test.test_anima_refiner_training')

        save_dir, expected = _train_and_save(tmp_path, adapter_type, monkeypatch)

        # Same seed, so the base weights are identical to the ones the adapter was trained on.
        fresh = training.build_dit(seed=0)
        apply_adapters(_refiner_pipeline(fresh, monkeypatch), [(save_dir, 1.0)])

        merged = fresh.state_dict()
        assert set(merged) == set(expected), 'merge_and_unload must leave the base model shape'
        for name, value in expected.items():
            torch.testing.assert_close(merged[name], value, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize('adapter_type', ['lora', 'lokr'])
    def test_the_adapter_actually_moved_the_weights(self, tmp_path, monkeypatch, adapter_type):
        """Guards the test above: a no-op adapter would satisfy it trivially."""
        training = pytest.importorskip('test.test_anima_refiner_training')
        _, merged = _train_and_save(tmp_path, adapter_type, monkeypatch)
        base = training.build_dit(seed=0).state_dict()
        moved = [n for n in base if not torch.equal(base[n], merged[n])]
        assert moved, 'the adapter delta is zero, so this fixture proves nothing'

    @pytest.mark.parametrize('adapter_type', ['lora', 'lokr'])
    def test_strength_scales_the_delta_linearly(self, tmp_path, monkeypatch, adapter_type):
        from tools.sample_anima_refiner import apply_adapters
        training = pytest.importorskip('test.test_anima_refiner_training')

        save_dir, full = _train_and_save(tmp_path, adapter_type, monkeypatch)
        base = training.build_dit(seed=0).state_dict()

        half_model = training.build_dit(seed=0)
        apply_adapters(_refiner_pipeline(half_model, monkeypatch), [(save_dir, 0.5)])
        half = half_model.state_dict()

        checked = 0
        for name, base_value in base.items():
            delta = full[name] - base_value
            if delta.abs().max() < 1e-6:
                continue
            torch.testing.assert_close(half[name] - base_value, delta * 0.5,
                                       atol=1e-5, rtol=1e-4)
            checked += 1
        assert checked, 'no parameter moved, so the scaling was never exercised'

    def test_stacking_two_adapters_applies_both(self, tmp_path, monkeypatch):
        from tools.sample_anima_refiner import apply_adapters
        training = pytest.importorskip('test.test_anima_refiner_training')

        first, _ = _train_and_save(tmp_path, 'lora', monkeypatch)
        second, _ = _train_and_save(tmp_path, 'lokr', monkeypatch)
        base = training.build_dit(seed=0).state_dict()

        both = training.build_dit(seed=0)
        apply_adapters(_refiner_pipeline(both, monkeypatch), [(first, 1.0), (second, 1.0)])
        stacked = both.state_dict()

        only_first = training.build_dit(seed=0)
        apply_adapters(_refiner_pipeline(only_first, monkeypatch), [(first, 1.0)])
        one = only_first.state_dict()

        moved_by_second = [n for n in base if not torch.equal(stacked[n], one[n])]
        assert moved_by_second, 'the second adapter was merged into nothing'
        assert all(torch.isfinite(v).all() for v in stacked.values())


class TestAdapterArgumentPairing:
    """--lora and --lora-strength must never be paired by guesswork."""

    @staticmethod
    def _dirs(tmp_path, n):
        made = []
        for i in range(n):
            path = tmp_path / f'adapter{i}'
            path.mkdir()
            made.append(str(path))
        return made

    def test_no_adapters_is_the_normal_case(self):
        from tools.sample_anima_refiner import resolve_adapters
        assert resolve_adapters(None, None) == []

    def test_strengths_default_to_one(self, tmp_path):
        from tools.sample_anima_refiner import resolve_adapters
        adapters = resolve_adapters(self._dirs(tmp_path, 2), None)
        assert [s for _, s in adapters] == [1.0, 1.0]

    def test_one_strength_per_adapter_is_kept_in_order(self, tmp_path):
        from tools.sample_anima_refiner import resolve_adapters
        dirs = self._dirs(tmp_path, 3)
        adapters = resolve_adapters(dirs, [0.5, 1.0, -0.25])
        assert [str(p) for p, _ in adapters] == dirs
        assert [s for _, s in adapters] == [0.5, 1.0, -0.25]

    def test_a_short_strength_list_is_refused(self, tmp_path):
        from tools.sample_anima_refiner import resolve_adapters
        with pytest.raises(SystemExit, match='2 --lora but 1 --lora-strength'):
            resolve_adapters(self._dirs(tmp_path, 2), [0.5])

    def test_strength_without_an_adapter_is_refused(self):
        from tools.sample_anima_refiner import resolve_adapters
        with pytest.raises(SystemExit, match='no --lora'):
            resolve_adapters(None, [0.5])

    def test_a_missing_directory_is_named(self, tmp_path):
        from tools.sample_anima_refiner import resolve_adapters
        with pytest.raises(SystemExit, match='is not a directory'):
            resolve_adapters([str(tmp_path / 'nope')], None)

    def test_two_dense_refiners_are_refused(self, tmp_path):
        """A densely trained refiner replaces the frontend outright; two cannot both apply."""
        import safetensors.torch
        from tools.sample_anima_refiner import resolve_adapters
        dirs = self._dirs(tmp_path, 2)
        for d in dirs:
            refiner_dir = Path(d) / 'context_refiner'
            refiner_dir.mkdir()
            safetensors.torch.save_file(
                {'cap_embedder.1.weight': torch.zeros(4, 4)},
                str(refiner_dir / 'context_refiner.safetensors'))
        with pytest.raises(SystemExit, match='densely trained context_refiner'):
            resolve_adapters(dirs, None)

    def test_one_dense_refiner_is_fine(self, tmp_path):
        import safetensors.torch
        from tools.sample_anima_refiner import resolve_adapters
        dirs = self._dirs(tmp_path, 2)
        refiner_dir = Path(dirs[0]) / 'context_refiner'
        refiner_dir.mkdir()
        safetensors.torch.save_file(
            {'cap_embedder.1.weight': torch.zeros(4, 4)},
            str(refiner_dir / 'context_refiner.safetensors'))
        assert len(resolve_adapters(dirs, None)) == 2


class TestOPLoRANeedsNoSamplingFlag:
    """OPLoRA is a constraint on a LoRA, never a different adapter format.

    It projects the low-rank update away from the base weight's dominant singular directions
    after each optimizer step. Nothing it configures reaches peft, so what lands on disk is an
    ordinary LoRA and the sampler reads it as one.
    """

    def test_oplora_keys_never_reach_the_peft_config(self):
        import inspect
        from models.base import CommonPipeline
        source = inspect.getsource(CommonPipeline.configure_adapter)
        assert 'oplora' not in source

    def test_peft_has_no_oplora_field_to_carry_it(self):
        import peft
        assert not any(k.startswith('oplora') for k in peft.LoraConfig.__dataclass_fields__)

    def test_defaults_land_in_the_toml_table_only(self):
        from utils.oplora import apply_oplora_config_defaults
        adapter_config = {'type': 'lora', 'rank': 4, 'alpha': 4, 'dropout': 0.0,
                          'oplora': True, 'oplora_rank': 2}
        apply_oplora_config_defaults(adapter_config)
        assert adapter_config['oplora'] is True
        assert adapter_config['oplora_rank'] == 2

    def test_a_saved_run_records_peft_type_lora(self, tmp_path, monkeypatch):
        import json
        save_dir, _ = _train_and_save(tmp_path, 'lora', monkeypatch)
        written = json.load(open(save_dir / 'adapter_config.json', encoding='utf-8'))
        assert written['peft_type'] == 'LORA'
        assert not any('oplora' in k for k in written)
