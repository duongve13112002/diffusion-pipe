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
    provably the same weights. Share it wrongly and the model becomes its own teacher: zero
    teacher signal, no error anywhere. Paths are only a fast path to that answer, never the
    answer itself.
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


class TestTheDocConfigSnippetMatchesTheCode:
    """A config snippet in the docs is something people copy. It has to be loadable.

    The specific trap this guards: cache_text_embeddings is read from [model], so a snippet
    showing it at the top level parses fine, is silently ignored, and the run is then refused
    for having caching on -- with an error pointing at a line the user already wrote.
    """

    @staticmethod
    def snippets():
        import re
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        text = (repo / 'docs' / 'anima_refiner' / 'teacher-guided-training.md').read_text('utf-8')
        return [block for block in re.findall(r'```toml\n(.*?)```', text, re.DOTALL)]

    def test_there_is_a_snippet_to_check(self):
        assert self.snippets()

    def test_every_snippet_puts_cache_text_embeddings_under_model(self):
        import toml
        for snippet in self.snippets():
            parsed = toml.loads(snippet)
            assert 'cache_text_embeddings' not in parsed, (
                'cache_text_embeddings is shown at the top level, where the pipeline never '
                'looks for it; it belongs under [model]'
            )
            if 'model' in parsed:
                assert parsed['model'].get('cache_text_embeddings') is False

    def test_every_snippet_validates(self):
        import toml
        for snippet in self.snippets():
            parsed = toml.loads(snippet)
            if 'teacher' not in parsed:
                continue
            resolved = validate_teacher_config(
                parsed, True, parsed.get('model', {}).get('cache_text_embeddings', True))
            assert resolved.enabled


class TestSharingIsDecidedByWeightsNotPaths:
    """Whether the student's DiT IS the teacher's is a fact about weights.

    Comparing checkpoint paths is only a proxy for it, and one that answers "different" for two
    copies of a single file -- which would cost 3.5-4 GB to no purpose. These pin the real check.
    """

    @staticmethod
    def pipeline_with(dit):
        pipe = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
        pipe.transformer = dit
        return pipe

    def test_identical_weights_match(self):
        dit = build_dit(crossattn_dim=64, num_blocks=1, seed=0)
        state_dict = {n: p.detach().clone() for n, p in dit.named_parameters()}
        assert self.pipeline_with(dit)._dit_weights_match(state_dict)

    def test_a_single_changed_tensor_does_not_match(self):
        dit = build_dit(crossattn_dim=64, num_blocks=1, seed=0)
        state_dict = {n: p.detach().clone() for n, p in dit.named_parameters()}
        key = next(iter(state_dict))
        state_dict[key] = state_dict[key] + 1e-3
        assert not self.pipeline_with(dit)._dit_weights_match(state_dict), (
            'a model that differs anywhere is a different model; the teacher needs its own copy'
        )

    def test_a_differently_seeded_model_does_not_match(self):
        dit = build_dit(crossattn_dim=64, num_blocks=1, seed=0)
        other = build_dit(crossattn_dim=64, num_blocks=1, seed=7)
        state_dict = {n: p.detach().clone() for n, p in other.named_parameters()}
        assert not self.pipeline_with(dit)._dit_weights_match(state_dict)

    def test_a_missing_tensor_does_not_match(self):
        dit = build_dit(crossattn_dim=64, num_blocks=1, seed=0)
        state_dict = {n: p.detach().clone() for n, p in dit.named_parameters()}
        del state_dict[next(iter(state_dict))]
        assert not self.pipeline_with(dit)._dit_weights_match(state_dict)

    def test_the_refiner_is_not_part_of_the_comparison(self):
        """A stock Anima checkpoint has no context_refiner.* keys, and should not need them.

        The teacher never runs the refiner, so its absence from the teacher's checkpoint says
        nothing about whether the two DiTs are the same.
        """
        dit = build_dit(crossattn_dim=64, num_blocks=1, n_refiner_layers=2, cap_feat_dim=32)
        state_dict = {n: p.detach().clone() for n, p in dit.named_parameters()
                      if not n.startswith('context_refiner.')}
        assert self.pipeline_with(dit)._dit_weights_match(state_dict)

    def test_a_trained_refiner_does_not_stop_the_dit_matching(self):
        dit = build_dit(crossattn_dim=64, num_blocks=1, n_refiner_layers=2, cap_feat_dim=32)
        state_dict = {n: p.detach().clone() for n, p in dit.named_parameters()
                      if not n.startswith('context_refiner.')}
        with torch.no_grad():
            dit.context_refiner.cap_embedder[1].weight.add_(0.5)
        assert self.pipeline_with(dit)._dit_weights_match(state_dict)


class TestTheStudentAndTeacherAreDifferentModelTypes:
    """The normal case: student is anima_refiner, teacher is stock anima, one checkpoint file.

    The two carry different text frontends by definition -- the student a context_refiner and no
    llm_adapter, the teacher an llm_adapter and no context_refiner -- so a naive comparison would
    always call them different and always load a redundant DiT. What has to match is the DiT
    body, which is the same MiniTrainDIT in both and the whole reason one can be swapped for the
    other.
    """

    @staticmethod
    def refiner_student_and_anima_checkpoint(seed=0):
        student = build_dit(crossattn_dim=64, num_blocks=1, n_refiner_layers=2,
                            cap_feat_dim=32, seed=seed)
        # What a stock Anima file holds: the same DiT body, plus an llm_adapter, minus a refiner.
        state_dict = {n: p.detach().clone() for n, p in student.named_parameters()
                      if not n.startswith('context_refiner.')}
        state_dict['llm_adapter.out_proj.weight'] = torch.randn(64, 64)
        state_dict['llm_adapter.embed.weight'] = torch.randn(32128, 64)
        pipe = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
        pipe.transformer = student
        return pipe, state_dict

    def test_the_two_frontends_do_not_stop_the_dit_matching(self):
        pipe, state_dict = self.refiner_student_and_anima_checkpoint()
        assert pipe._dit_weights_match(state_dict), (
            'an anima_refiner student and an anima teacher from one file share a DiT body; '
            'their differing text frontends are exactly what the loss measures, not a reason '
            'to load the body twice'
        )

    def test_a_difference_in_the_body_still_fails(self):
        pipe, state_dict = self.refiner_student_and_anima_checkpoint()
        key = 'blocks.0.self_attn.q_proj.weight'
        state_dict[key] = state_dict[key] + 1e-3
        assert not pipe._dit_weights_match(state_dict)

    def test_the_teachers_adapter_is_never_compared(self):
        """The student has no llm_adapter at all, so it cannot be part of the comparison."""
        pipe, state_dict = self.refiner_student_and_anima_checkpoint()
        state_dict['llm_adapter.out_proj.weight'] = torch.randn(64, 64)
        assert pipe._dit_weights_match(state_dict)


class TestBothTypesShareOneCacheName:
    """anima and anima_refiner deliberately write under the same cache tree name."""

    def test_the_name_is_shared(self):
        # The name selects the whole cache tree, latents included, and the two use the same VAE.
        # A separate name would discard latents that are still perfectly valid -- by far the
        # expensive half. What differs is the text encoder, and that lives in the text-embedding
        # fingerprint instead, via text_encoder_cache_key.
        source = (cp2.__file__)
        with open(source, encoding='utf-8') as f:
            text = f.read()
        assert "self.name = 'anima'" in text, (
            'anima_refiner must keep sharing the cache name with anima, or every switch between '
            'them re-encodes the entire dataset through an unchanged VAE'
        )

    def test_only_the_refiner_contributes_a_text_encoder_cache_key(self):
        """anima returns '' so that adding the key never moved an existing install's cache."""
        pipe = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
        pipe.use_context_refiner = False
        assert pipe.text_encoder_cache_key(0) == ''

        pipe.use_context_refiner = True
        pipe.model_config = {'llm_path': '/models/Qwen3.5-2B-Base'}
        pipe.llm_hidden_layer = -1
        pipe.max_text_length = 512
        pipe.cap_feat_dim = 2048
        key = pipe.text_encoder_cache_key(0)
        assert key and 'Qwen3.5-2B-Base' in key


class TestAMismatchedTeacherShapeIsRefused:
    """A separately loaded teacher is the only case where the two DiTs can disagree on shape.

    A shared one is the student's module, so it cannot. A copied one comes from whatever file
    the user named, and the two velocities are mixed elementwise.
    """

    def test_a_differing_channel_count_raises(self):
        v_gt = torch.randn(2, 16, 1, 8, 8)
        with pytest.raises(RuntimeError, match='does not match'):
            blend_target(v_gt, torch.randn(2, 4, 1, 8, 8), torch.full((2, 1), 0.5))

    def test_a_single_channel_teacher_raises_instead_of_broadcasting(self):
        """The dangerous one: (B,1,T,H,W) broadcasts cleanly across (B,16,T,H,W).

        Left to torch, that builds a full-sized target out of one channel of prediction, with
        no error and a loss that looks entirely healthy.
        """
        v_gt = torch.randn(2, 16, 1, 8, 8)
        with pytest.raises(RuntimeError, match='does not match'):
            blend_target(v_gt, torch.randn(2, 1, 1, 8, 8), torch.full((2, 1), 0.5))

    def test_a_differing_resolution_raises(self):
        v_gt = torch.randn(2, 16, 1, 8, 8)
        with pytest.raises(RuntimeError, match='does not match'):
            blend_target(v_gt, torch.randn(2, 16, 1, 16, 16), torch.full((2, 1), 0.5))

    def test_matching_shapes_still_blend(self):
        v_gt = torch.randn(2, 16, 1, 8, 8)
        out = blend_target(v_gt, torch.randn(2, 16, 1, 8, 8), torch.zeros(2, 1))
        assert torch.allclose(out, v_gt)


class TestTheTeacherForwardMatchesTheStudentsDtypeHandling:
    """Regression: a bf16 multi-GPU run died in the teacher's first Linear.

        RuntimeError: expected mat1 and mat2 to have the same dtype, but got:
        float != c10::BFloat16                     (x_embedder.proj, rank 0/1/5)

    prepare_embedded_sequence concatenates the padding mask onto the latents, and torch.cat
    PROMOTES. velocity() had cast the latents to the DiT's dtype but built the mask from the
    original float32 tensor, so the cat dragged the latents back to float32 and the bf16
    x_embedder rejected them.

    The cast was the mistake, not just the mask. A DiT loaded with transformer_dtype has no
    single dtype -- KEEP_IN_HIGH_PRECISION leaves x_embedder and final_layer wider than the
    blocks -- so there is nothing correct to cast to. The student never casts either: every
    pipeline layer carries @torch.autocast and is handed float32. The teacher now does the same.
    """

    @staticmethod
    def capture_dit_inputs():
        """A stand-in DiT recording the dtypes its forward is handed."""
        seen = {}

        class RecordingDiT(nn.Module):
            in_channels = out_channels = 16

            def forward(self, x, t, crossattn_emb, fps=None, padding_mask=None):
                seen['latents'] = x.dtype
                seen['padding_mask'] = padding_mask.dtype
                seen['timesteps'] = t.dtype
                return torch.zeros(x.shape[0], 16, 1, x.shape[3], x.shape[4], dtype=x.dtype)

        return RecordingDiT(), seen

    def run(self, latents):
        dit, seen = self.capture_dit_inputs()
        guide = build_teacher_guide(dit, 64)
        guide.max_text_length = 8
        guide.velocity(latents, torch.tensor([[0.5]]), ['a cat'])
        return seen

    def test_the_padding_mask_shares_the_latents_dtype(self):
        seen = self.run(torch.randn(1, 16, 1, 8, 8))
        assert seen['padding_mask'] == seen['latents'], (
            'the mask is concatenated onto the latents and torch.cat promotes, so a mask in a '
            'different dtype silently changes what reaches the DiT'
        )

    def test_the_latents_are_not_cast_before_the_forward(self):
        """Handed through as they arrive, exactly as the student's layer stack receives them."""
        latents = torch.randn(1, 16, 1, 8, 8)
        assert self.run(latents)['latents'] == latents.dtype

    def test_it_holds_for_a_half_precision_latent_too(self):
        latents = torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16)
        seen = self.run(latents)
        assert seen['latents'] == torch.bfloat16
        assert seen['padding_mask'] == torch.bfloat16
