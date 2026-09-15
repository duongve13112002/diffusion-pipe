"""Teacher-guided training: the schedule, the blend, and what must be refused.

The feature mixes the ground-truth flow-matching target toward a frozen stock Anima's velocity
for the same latent, timestep and caption, by a weight that rises with the noise level and
decays over the run. See docs/anima_refiner/teacher-guided-training.md.

The tests that matter most here are the ones covering things that fail *silently*:

  * lambda is exactly an interpolation weight only because mixing squared losses equals mixing
    targets. TestBlendIsTheSameAsMixingLosses pins that algebra, since the implementation relies
    on it to skip the loss function entirely.
  * eval must not see the teacher, or the eval metric drifts as the decay runs and a change in
    the loss definition looks like a change in the model.
  * the teacher may share the student's DiT only when that DiT provably cannot move AND is
    provably the same checkpoint. Share it wrongly and the model becomes its own teacher: zero
    teacher signal, no error anywhere.
"""

import pytest
import torch
from torch import nn

pytest.importorskip('models.cosmos_predict2')

import models.cosmos_predict2 as cp2  # noqa: E402
from models.teacher_guidance import (  # noqa: E402
    DISABLED,
    TeacherConfig,
    TeacherGuide,
    blend_target,
    decay_at,
    lambda_at,
    shape_at,
    validate_teacher_config,
)


def cfg(**kwargs):
    base = dict(loss_weight=1.0, transformer_path='t.safetensors', llm_path='llm',
                shape='sigmoid', t_mid=0.5, width=0.15, decay='none', decay_steps=0)
    base.update(kwargs)
    return TeacherConfig(**base)


class TestShape:
    def test_it_rises_with_the_noise_level(self):
        t = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
        s = shape_at(t, cfg())
        assert torch.all(s[1:] > s[:-1]), (
            'the teacher must be weighted more at higher noise, where the caption determines '
            'the target and one ground-truth draw is a noisy estimate of it'
        )

    def test_it_is_a_half_at_the_midpoint(self):
        assert shape_at(torch.tensor([[0.5]]), cfg(t_mid=0.5)).item() == pytest.approx(0.5)
        assert shape_at(torch.tensor([[0.65]]), cfg(t_mid=0.65)).item() == pytest.approx(0.5)

    def test_t_mid_moves_the_boundary(self):
        t = torch.tensor([[0.6]])
        assert shape_at(t, cfg(t_mid=0.65)).item() < shape_at(t, cfg(t_mid=0.5)).item(), (
            'raising t_mid must hand the low-noise band back to the ground truth, which is what '
            'keeps a style fine tune learning its own texture rather than stock Anima\'s'
        )

    def test_width_controls_sharpness(self):
        t = torch.tensor([[0.9]])
        assert shape_at(t, cfg(width=0.05)).item() > shape_at(t, cfg(width=0.5)).item()

    def test_constant_removes_the_timestep_dependence(self):
        t = torch.tensor([[0.1], [0.9]])
        s = shape_at(t, cfg(shape='constant'))
        assert torch.equal(s, torch.ones_like(t))


class TestDecay:
    def test_none_never_fades(self):
        c = cfg(decay='none')
        assert decay_at(0, c) == 1.0
        assert decay_at(10_000, c) == 1.0

    @pytest.mark.parametrize('kind', ['linear', 'cosine'])
    def test_it_starts_at_one_and_reaches_zero(self, kind):
        c = cfg(decay=kind, decay_steps=100)
        assert decay_at(0, c) == pytest.approx(1.0)
        assert decay_at(100, c) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize('kind', ['linear', 'cosine'])
    def test_it_stays_at_zero_past_the_end(self, kind):
        # The run does not stop at decay_steps. Going negative would invert the term into a
        # push AWAY from the teacher, which is not a schedule anyone asked for.
        c = cfg(decay=kind, decay_steps=100)
        assert decay_at(500, c) == pytest.approx(0.0, abs=1e-12)

    def test_it_is_monotone(self):
        c = cfg(decay='cosine', decay_steps=100)
        values = [decay_at(s, c) for s in range(0, 101, 10)]
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


class TestLambda:
    def test_loss_weight_is_the_ceiling(self):
        t = torch.tensor([[0.999]])
        lam = lambda_at(t, 0, cfg(loss_weight=0.4))
        assert 0 < lam.item() <= 0.4

    def test_a_disabled_config_is_always_zero(self):
        t = torch.tensor([[0.1], [0.9]])
        assert torch.equal(lambda_at(t, 0, DISABLED), torch.zeros_like(t))

    def test_it_combines_both_schedules(self):
        t = torch.tensor([[0.9]])
        c = cfg(decay='linear', decay_steps=100)
        early, late = lambda_at(t, 0, c).item(), lambda_at(t, 90, c).item()
        assert late == pytest.approx(early * 0.1, rel=1e-6)


class TestBlend:
    def test_zero_lambda_is_exactly_the_ground_truth(self):
        gt, teacher = torch.randn(2, 3, 1, 4, 4), torch.randn(2, 3, 1, 4, 4)
        out = blend_target(gt, teacher, torch.zeros(2, 1))
        assert torch.allclose(out, gt)

    def test_unit_lambda_is_exactly_the_teacher(self):
        gt, teacher = torch.randn(2, 3, 1, 4, 4), torch.randn(2, 3, 1, 4, 4)
        out = blend_target(gt, teacher, torch.ones(2, 1))
        assert torch.allclose(out, teacher)

    def test_lambda_is_per_sample(self):
        """Each sample carries its own timestep, so it must carry its own lambda."""
        gt, teacher = torch.randn(2, 3, 1, 4, 4), torch.randn(2, 3, 1, 4, 4)
        out = blend_target(gt, teacher, torch.tensor([[0.0], [1.0]]))
        assert torch.allclose(out[0], gt[0])
        assert torch.allclose(out[1], teacher[1])


class TestBlendIsTheSameAsMixingLosses:
    """The identity the whole implementation rests on.

    Mixing the two squared-error losses has the same gradient as regressing onto the mixed
    target, because the prediction terms collect. That is why the teacher term needs no change
    to get_loss_fn, no extra tensor through the pipeline, and no normalising of two terms whose
    printed values differ by ~5x at high noise.
    """

    def test_the_gradients_agree(self):
        torch.manual_seed(0)
        gt, teacher = torch.randn(4, 8), torch.randn(4, 8)
        lam_value = 0.37

        pred_a = torch.randn(4, 8, requires_grad=True)
        mixed_losses = ((1 - lam_value) * (pred_a - gt).pow(2)
                        + lam_value * (pred_a - teacher).pow(2)).mean()
        mixed_losses.backward()

        pred_b = pred_a.detach().clone().requires_grad_(True)
        target = blend_target(gt, teacher, torch.full((4, 1), lam_value))
        (pred_b - target).pow(2).mean().backward()

        assert torch.allclose(pred_a.grad, pred_b.grad, atol=1e-6)


class TestValidation:
    @staticmethod
    def base_config(**overrides):
        config = {
            'pipeline_stages': 1,
            'teacher': {
                'loss_weight': 1.0,
                'transformer_path': 'anima.safetensors',
                'llm_path': 'qwen3-0.6b',
                'decay': 'linear',
                'decay_steps': 1000,
            },
        }
        config['teacher'].update(overrides.pop('teacher', {}))
        config.update(overrides)
        return config

    def test_absent_table_is_disabled(self):
        assert validate_teacher_config({}, True, False) is DISABLED

    def test_zero_weight_is_disabled(self):
        config = self.base_config(teacher={'loss_weight': 0.0})
        assert validate_teacher_config(config, True, False) is DISABLED

    def test_other_models_are_untouched_when_the_table_is_absent(self):
        """The rule from lessons.md: adding a feature must not change an existing model."""
        for use_refiner in (True, False):
            assert validate_teacher_config({'model': {}}, use_refiner, True) is DISABLED

    def test_it_refuses_a_model_without_a_second_text_frontend(self):
        with pytest.raises(ValueError, match='anima_refiner'):
            validate_teacher_config(self.base_config(), False, False)

    def test_it_refuses_cached_text_embeddings(self):
        with pytest.raises(ValueError, match='cache_text_embeddings = false'):
            validate_teacher_config(self.base_config(), True, True)

    def test_it_refuses_split_pipelines(self):
        with pytest.raises(ValueError, match='pipeline_stages = 1'):
            validate_teacher_config(self.base_config(pipeline_stages=2), True, False)

    def test_it_refuses_block_swapping(self):
        with pytest.raises(ValueError, match='blocks_to_swap'):
            validate_teacher_config(self.base_config(blocks_to_swap=4), True, False)

    @pytest.mark.parametrize('missing', ['transformer_path', 'llm_path'])
    def test_it_requires_both_teacher_paths(self, missing):
        config = self.base_config()
        del config['teacher'][missing]
        with pytest.raises(ValueError, match=missing):
            validate_teacher_config(config, True, False)

    def test_it_requires_decay_steps_when_decaying(self):
        config = self.base_config(teacher={'decay': 'linear', 'decay_steps': 0})
        with pytest.raises(ValueError, match='decay_steps'):
            validate_teacher_config(config, True, False)

    def test_decay_none_needs_no_decay_steps(self):
        config = self.base_config(teacher={'decay': 'none', 'decay_steps': 0})
        assert validate_teacher_config(config, True, False).enabled

    @pytest.mark.parametrize('bad', [
        {'shape': 'linear'}, {'t_mid': 1.5}, {'width': 0.0}, {'decay': 'exponential'},
        {'loss_weight': -1.0},
    ])
    def test_it_rejects_out_of_range_settings(self, bad):
        with pytest.raises(ValueError):
            validate_teacher_config(self.base_config(teacher=bad), True, False)

    def test_defaults(self):
        config = self.base_config(teacher={'decay': 'none'})
        resolved = validate_teacher_config(config, True, False)
        assert resolved.shape == 'sigmoid'
        assert resolved.t_mid == 0.5
        assert resolved.width == 0.15


class TestSharingTheStudentsDit:
    """A shared DiT is an optimisation worth 3.5-4 GB, and a silent bug if taken wrongly."""

    @staticmethod
    def frozen(**model_config):
        stub = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
        stub.config = {'optimizer': {'lr': 1e-4}}
        stub.config.update(model_config.pop('_config', {}))
        stub.model_config = dict(model_config)
        stub.use_context_refiner = True
        return cp2.CosmosPredict2Pipeline._dit_is_frozen(stub)

    def test_a_refiner_only_config_may_share(self):
        assert self.frozen(base_lr=0, self_attn_lr=0, cross_attn_lr=0, mlp_lr=0, mod_lr=0,
                           llm_adapter_lr=0, refiner_lr=1e-4)

    def test_a_training_cross_attention_may_not(self):
        assert not self.frozen(base_lr=0, self_attn_lr=0, cross_attn_lr=1e-5, mlp_lr=0,
                               mod_lr=0, llm_adapter_lr=0, refiner_lr=1e-4)

    def test_a_full_fine_tune_may_not(self):
        assert not self.frozen(refiner_lr=1e-4)

    def test_an_adapter_disqualifies_sharing_whatever_the_rates_say(self):
        # A LoRA wraps the DiT's own Linear layers, so the shared module stops being the stock
        # weights from the first optimizer step even though every listed rate is 0.
        assert not self.frozen(
            base_lr=0, self_attn_lr=0, cross_attn_lr=0, mlp_lr=0, mod_lr=0, llm_adapter_lr=0,
            refiner_lr=0, _config={'adapter': {'type': 'lora', 'rank': 16}},
        )

    def test_a_training_refiner_does_not_block_sharing(self):
        # The teacher's text frontend is the llm_adapter; it never reads context_refiner, so a
        # refiner under training cannot change the teacher's prediction.
        assert self.frozen(base_lr=0, self_attn_lr=0, cross_attn_lr=0, mlp_lr=0, mod_lr=0,
                           llm_adapter_lr=0, refiner_lr=1e-3)


class _FakeTokenizer:
    """Deterministic ids, so a caption maps to the same tokens for both frontends."""

    def __init__(self, vocab=64):
        self.vocab = vocab

    def __call__(self, prompts, return_tensors=None, truncation=None, padding=None,
                 max_length=8):
        import transformers
        ids, masks = [], []
        for prompt in prompts:
            real = [(abs(hash(tok)) % (self.vocab - 1)) + 1 for tok in prompt.split()][:max_length]
            pad = max_length - len(real)
            ids.append(real + [0] * pad)
            masks.append([1] * len(real) + [0] * pad)
        return transformers.BatchEncoding({
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(masks),
        })


class _FakeLLM(nn.Module):
    """Stands in for Qwen3-0.6B: the teacher's text encoder, frozen and deterministic."""

    def __init__(self, vocab=64, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.device = torch.device('cpu')

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        hidden = self.embed(input_ids)
        # hidden_states[-1] == last_hidden_state, which is the relationship llm_hidden_layer
        # indexes into. A stand-in without it would make llm_hidden_layer untestable.
        return type('Out', (), {
            'last_hidden_state': hidden,
            'hidden_states': (torch.zeros_like(hidden), hidden),
        })()


LATENT_CHANNELS = 16


def build_dit(crossattn_dim=64, num_blocks=1, n_refiner_layers=None, cap_feat_dim=None, seed=0):
    """A MiniTrainDIT small enough for CPU. Head dim must stay above 2: rope3d splits it three
    ways and raises ZeroDivisionError on a dim of exactly 2."""
    from models.cosmos_predict2_modeling import MiniTrainDIT
    torch.manual_seed(seed)
    kwargs = dict(
        max_img_h=32, max_img_w=32, max_frames=4,
        in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS,
        patch_spatial=2, patch_temporal=1, model_channels=64, concat_padding_mask=True,
        crossattn_emb_channels=crossattn_dim, pos_emb_cls='rope3d', pos_emb_learnable=True,
        num_blocks=num_blocks, num_heads=4, use_adaln_lora=True, adaln_lora_dim=16,
    )
    if cap_feat_dim is not None:
        kwargs.update(cap_feat_dim=cap_feat_dim, n_refiner_layers=n_refiner_layers)
    dit = MiniTrainDIT(**kwargs)
    if cap_feat_dim is not None:
        dit.context_refiner.init_weights()
    for name, p in dit.named_parameters():
        p.original_name = name
    return dit


def build_teacher_guide(dit, crossattn_dim, shares_dit=False, seed=1):
    from models.llm_adapter import LLMAdapter
    torch.manual_seed(seed)
    adapter = LLMAdapter(source_dim=32, target_dim=crossattn_dim, model_dim=crossattn_dim,
                         num_layers=1, num_heads=4, self_attn=True)
    return TeacherGuide(
        dit=dit, llm_adapter=adapter, text_encoder=_FakeLLM(), tokenizer=_FakeTokenizer(),
        t5_tokenizer=_FakeTokenizer(), max_text_length=8, shares_dit=shares_dit,
    )


class TestTeacherVelocity:
    """The teacher's forward: frozen, deterministic, and it must not disturb a shared DiT."""

    @staticmethod
    def build_dit(crossattn_dim=64, num_blocks=1):
        return build_dit(crossattn_dim=crossattn_dim, num_blocks=num_blocks)

    def test_it_returns_a_velocity_shaped_like_the_latent(self):
        dit = self.build_dit()
        guide = build_teacher_guide(dit, 64)
        x = torch.randn(2, LATENT_CHANNELS, 1, 8, 8)
        v = guide.velocity(x, torch.tensor([[0.3], [0.8]]), ['a cat', 'a dog'])
        assert v.shape == x.shape

    def test_it_carries_no_gradient(self):
        dit = self.build_dit()
        guide = build_teacher_guide(dit, 64)
        v = guide.velocity(torch.randn(1, LATENT_CHANNELS, 1, 8, 8), torch.tensor([[0.5]]), ['a cat'])
        assert not v.requires_grad, 'the teacher is a constant target, never a path to optimise'

    def test_it_is_deterministic_for_the_same_inputs(self):
        dit = self.build_dit()
        guide = build_teacher_guide(dit, 64)
        x, t, caps = torch.randn(1, LATENT_CHANNELS, 1, 8, 8), torch.tensor([[0.5]]), ['a cat']
        assert torch.allclose(guide.velocity(x, t, caps), guide.velocity(x, t, caps))

    def test_different_captions_give_different_velocities(self):
        # If they did not, the term would carry no information about the text at all.
        dit = self.build_dit()
        guide = build_teacher_guide(dit, 64)
        x, t = torch.randn(1, LATENT_CHANNELS, 1, 8, 8), torch.tensor([[0.8]])
        assert not torch.allclose(
            guide.velocity(x, t, ['a small cat']), guide.velocity(x, t, ['a large truck']))

    def test_it_leaves_a_shared_dit_in_training_mode(self):
        """The shared DiT is the student's, and train.py put it in train mode."""
        dit = self.build_dit()
        dit.train()
        guide = build_teacher_guide(dit, 64, shares_dit=True)
        guide.velocity(torch.randn(1, LATENT_CHANNELS, 1, 8, 8), torch.tensor([[0.5]]), ['a cat'])
        assert dit.training

    def test_the_shared_dit_is_not_registered_as_a_submodule(self):
        """Registering it would move it, save it, and hand its parameters to an optimizer."""
        dit = self.build_dit()
        guide = build_teacher_guide(dit, 64, shares_dit=True)
        assert all(module is not dit for module in guide.modules())


class TestTheShippedExampleConfig:
    """An example config that would be refused at startup is documentation that lies."""

    @staticmethod
    def load():
        import toml
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        return toml.load(repo / 'examples' / 'anima_refiner' / 'teacher_guided.toml')

    def test_it_validates(self):
        config = self.load()
        resolved = validate_teacher_config(
            config, config['model']['type'] == 'anima_refiner',
            config['model'].get('cache_text_embeddings', True))
        assert resolved.enabled

    def test_it_sets_everything_the_feature_requires(self):
        config = self.load()
        assert config['model']['cache_text_embeddings'] is False
        assert config['pipeline_stages'] == 1
        assert 'blocks_to_swap' not in config

    def test_it_is_a_refiner_only_configuration_so_the_dit_is_shareable(self):
        # The example advertises that no second DiT is loaded. That claim is only true while
        # every DiT rate stays 0 and no [adapter] table appears.
        config = self.load()
        model_config = config['model']
        assert 'adapter' not in config
        for key in ('base_lr', 'self_attn_lr', 'cross_attn_lr', 'mlp_lr', 'mod_lr'):
            assert model_config[key] == 0, f'{key} is non-zero, so the example needs a teacher copy'
        assert model_config['refiner_lr'] > 0

    def test_its_decay_is_finite(self):
        config = self.load()
        assert config['teacher']['decay'] != 'none'
        assert config['teacher']['decay_steps'] > 0
