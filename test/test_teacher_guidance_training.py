"""Teacher-guided training, end to end on a toy model.

Runs real optimizer steps through the same layer stack CosmosPredict2Pipeline.to_layers()
produces, with a real frozen teacher (a second MiniTrainDIT plus an LLMAdapter), on CPU. It
covers the parts that only appear once the pieces are connected:

  * the real prepare_inputs blends, and skips the teacher during eval,
  * training converges with the teacher on, and only the refiner moves,
  * lambda actually steers the target -- toward the teacher at high noise, toward the ground
    truth at low noise -- and the decay schedule really does hand the run back to the data.

Run it as a script for a printed simulation. PYTHONPATH picks up the CPU shims that
test/conftest.py installs under pytest but that a bare script never sees:

    PYTHONPATH=test/childenv python -m test.test_teacher_guidance_training

The toy teacher is a differently initialised DiT, not a better model, so "the loss falls" here
means the optimisation is wired up correctly. It is not evidence about image quality, and
nothing on this branch has been run on a GPU.
"""

import pytest
import torch
from torch import nn

pytest.importorskip('models.cosmos_predict2')

import models.cosmos_predict2 as cp2  # noqa: E402
from models.cosmos_predict2 import (  # noqa: E402
    ContextRefinerLayer,
    FinalLayer,
    InitialLayer,
    TransformerLayer,
)
from models.teacher_guidance import TeacherConfig, lambda_at  # noqa: E402
from test.test_teacher_guidance import (  # noqa: E402
    LATENT_CHANNELS,
    _FakeLLM,
    _FakeTokenizer,
    build_dit,
    build_teacher_guide,
)

CAP_FEAT_DIM = 32
CROSSATTN_DIM = 64
MAX_TEXT_LENGTH = 8

CAPTIONS = [
    '1girl solo blue eyes',
    'a large red truck',
    'cherry blossom outdoors',
    'a small black cat',
]


class _NullOffloader:
    def wait_for_block(self, idx):
        pass

    def submit_move_blocks_forward(self, idx):
        pass


def build_layers(dit, text_encoder):
    """Mirrors CosmosPredict2Pipeline.to_layers() for the refiner architecture.

    The text encoder is resident in the layer stack because teacher guidance requires
    cache_text_embeddings = false, so prepare_inputs hands the stack token ids rather than
    embeddings and InitialLayer runs the encoder itself.
    """
    layers = [InitialLayer(dit, text_encoder, True, None),
              ContextRefinerLayer(dit.context_refiner)]
    for i, block in enumerate(dit.blocks):
        layers.append(TransformerLayer(block, i, _NullOffloader()))
    layers.append(FinalLayer(dit))
    return layers


def run_layers(layers, inputs):
    for layer in layers:
        inputs = layer(inputs)
    return inputs


def teacher_cfg(**kwargs):
    base = dict(loss_weight=1.0, transformer_path='teacher.safetensors', llm_path='llm',
                shape='sigmoid', t_mid=0.5, width=0.15, decay='none', decay_steps=0)
    base.update(kwargs)
    return TeacherConfig(**base)


def build_pipeline(teacher=None, cfg=None, step=0, shift=None):
    """A CosmosPredict2Pipeline carrying only what prepare_inputs reads.

    Built with __new__ rather than __init__ so the test needs no checkpoints, no VAE and no
    real LLM -- the same approach the existing anima_refiner tests take for get_param_groups.
    """
    pipe = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
    pipe.model_config = {'dtype': torch.float32, 'timestep_sample_method': 'uniform'}
    if shift is not None:
        pipe.model_config['shift'] = shift
    pipe.config = {'optimizer': {'lr': 1e-4}}
    pipe.cache_text_embeddings = False
    pipe.use_context_refiner = True
    pipe.max_text_length = MAX_TEXT_LENGTH
    # The student tokenizes through this whether or not a teacher is attached; the two runs
    # compared in these tests must see identical student input.
    pipe.tokenizer = _FakeTokenizer()
    pipe.teacher = teacher
    pipe.teacher_cfg = cfg if cfg is not None else teacher_cfg()
    pipe._step_source = lambda: step
    # Mirrors what __init__ sets. Deliberately not defaulted with getattr in the pipeline: a
    # missing attribute there would mean __init__ had not run, which is worth an exception
    # rather than a quietly empty log.
    pipe._last_teacher_lambda = None
    pipe._teacher_decay_announced = False
    return pipe


def toy_batch(batch_size=4, size=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        'latents': torch.randn(batch_size, LATENT_CHANNELS, 1, size, size, generator=generator),
        'mask': None,
        'caption': CAPTIONS[:batch_size],
    }


def build_student_and_teacher(num_blocks=2, seed=0):
    student = build_dit(crossattn_dim=CROSSATTN_DIM, num_blocks=num_blocks,
                        n_refiner_layers=2, cap_feat_dim=CAP_FEAT_DIM, seed=seed)
    student.train()
    torch.manual_seed(seed + 50)
    student_llm = _FakeLLM(dim=CAP_FEAT_DIM).requires_grad_(False)
    # A differently seeded DiT, so the teacher's velocity is genuinely a different function.
    teacher_dit = build_dit(crossattn_dim=CROSSATTN_DIM, num_blocks=num_blocks, seed=seed + 100)
    guide = build_teacher_guide(teacher_dit, CROSSATTN_DIM)
    guide.max_text_length = MAX_TEXT_LENGTH
    guide.device = torch.device('cpu')
    guide._placed = True
    return student, student_llm, guide


class TestPrepareInputs:
    """The real method, with a real teacher attached."""

    def test_the_label_keeps_its_shape(self):
        """The blend happens in the target, so nothing extra travels through the pipeline."""
        student, student_llm, guide = build_student_and_teacher()
        pipe = build_pipeline(teacher=guide)
        _, label = pipe.prepare_inputs(toy_batch())
        assert len(label) == 2, (
            'the label must stay (target, mask): mixing squared losses equals mixing targets, '
            'which is what lets get_loss_fn and the pipeline plumbing stay untouched'
        )

    def test_the_target_moves_when_the_teacher_is_on(self):
        batch = toy_batch()
        student, student_llm, guide = build_student_and_teacher()

        torch.manual_seed(7)
        _, (with_teacher, _) = build_pipeline(teacher=guide).prepare_inputs(batch)
        torch.manual_seed(7)
        _, (without, _) = build_pipeline(teacher=None).prepare_inputs(batch)

        assert not torch.allclose(with_teacher, without)

    def test_eval_never_sees_the_teacher(self):
        """timestep_quantile means eval, and eval must stay on the pure ground-truth loss.

        Otherwise the eval metric drifts as the decay schedule runs, and a change in the loss
        definition is indistinguishable from a change in the model.
        """
        batch = toy_batch()
        student, student_llm, guide = build_student_and_teacher()

        torch.manual_seed(7)
        _, (evaluated, _) = build_pipeline(teacher=guide).prepare_inputs(
            batch, timestep_quantile=0.5)
        torch.manual_seed(7)
        _, (plain, _) = build_pipeline(teacher=None).prepare_inputs(batch, timestep_quantile=0.5)

        assert torch.allclose(evaluated, plain)

    def test_a_fully_decayed_schedule_is_ordinary_training(self):
        batch = toy_batch()
        student, student_llm, guide = build_student_and_teacher()
        cfg = teacher_cfg(decay='linear', decay_steps=100)

        torch.manual_seed(7)
        _, (decayed, _) = build_pipeline(teacher=guide, cfg=cfg, step=100).prepare_inputs(batch)
        torch.manual_seed(7)
        _, (plain, _) = build_pipeline(teacher=None).prepare_inputs(batch)

        assert torch.allclose(decayed, plain), (
            'once lambda reaches zero the run must be bit-identical to ground-truth training'
        )


class TestLambdaSteersTheTarget:
    """What the schedule is for: the teacher at high noise, the data at low noise."""

    @staticmethod
    def targets_at(t_value, guide, batch):
        """Ground truth, teacher velocity and blended target at one fixed timestep."""
        latents = batch['latents'].float()
        t = torch.full((latents.shape[0], 1), t_value)
        noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(3))
        t_exp = t.view(-1, 1, 1, 1, 1)
        noisy = (1 - t_exp) * latents + t_exp * noise
        v_gt = noise - latents
        v_teacher = guide.velocity(noisy, t, batch['caption'])
        lam = lambda_at(t, 0, teacher_cfg())
        blended = (1 - lam.view(-1, 1, 1, 1, 1)) * v_gt + lam.view(-1, 1, 1, 1, 1) * v_teacher
        return v_gt, v_teacher, blended

    def test_at_high_noise_the_target_follows_the_teacher(self):
        batch = toy_batch()
        _, _, guide = build_student_and_teacher()
        v_gt, v_teacher, blended = self.targets_at(0.95, guide, batch)
        assert (blended - v_teacher).norm() < (blended - v_gt).norm()

    def test_at_low_noise_the_target_follows_the_ground_truth(self):
        batch = toy_batch()
        _, _, guide = build_student_and_teacher()
        v_gt, v_teacher, blended = self.targets_at(0.05, guide, batch)
        assert (blended - v_gt).norm() < (blended - v_teacher).norm(), (
            'low noise is where texture is learned, and it must come from the dataset rather '
            'than from stock Anima'
        )


def train(steps=60, lr=3e-3, teacher=None, cfg=None, seed=0, freeze_dit=True, record=None):
    """A real training loop over the real layer stack. Returns the loss history."""
    student, student_llm, guide = build_student_and_teacher(seed=seed)
    if teacher is False:
        guide = None

    if freeze_dit:
        for name, p in student.named_parameters():
            p.requires_grad_('context_refiner' in name)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    layers = build_layers(student, student_llm)
    batch = toy_batch(seed=seed)
    losses = []

    for step in range(steps):
        pipe = build_pipeline(teacher=guide, cfg=cfg, step=step)
        torch.manual_seed(1000 + step)  # same noise and timesteps across compared runs
        features, (target, _) = pipe.prepare_inputs(batch)
        output = run_layers(layers, features)
        loss = nn.functional.mse_loss(output.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if record is not None:
            # Zero when no teacher is attached: reporting the schedule's value for a run that
            # never applies it is how a reader concludes the term was on when it was not.
            if guide is None:
                lam = 0.0
            else:
                lam = float(lambda_at(features[1].detach(), step, pipe.teacher_cfg).mean())
            record.append((step, lam, loss.item()))

    return student, losses


def evaluate_ground_truth(student, student_llm, seed=0, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Loss against the pure ground truth, which is the only metric comparable across runs.

    Training loss is not: with the teacher on, the target is a pre-averaged velocity and so a
    strictly easier one, and the number falls for that reason alone. This is the same argument
    that makes eval skip the teacher entirely.
    """
    layers = build_layers(student, student_llm)
    batch = toy_batch(seed=seed)
    pipe = build_pipeline(teacher=None)
    total = 0.0
    with torch.no_grad():
        for quantile in quantiles:
            torch.manual_seed(4242)
            features, (target, _) = pipe.prepare_inputs(batch, timestep_quantile=quantile)
            output = run_layers(layers, features)
            total += nn.functional.mse_loss(output.float(), target.float()).item()
    return total / len(quantiles)


class TestToyTrainingRun:
    def test_the_loss_falls_with_the_teacher_on(self):
        _, losses = train()
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f'loss did not fall: {early:.4f} -> {late:.4f}'

    def test_only_the_refiner_moves_when_the_dit_is_frozen(self):
        student, _, _ = build_student_and_teacher()
        before = {n: p.detach().clone() for n, p in student.named_parameters()}
        trained, _ = train()
        moved = [n for n, p in trained.named_parameters()
                 if not torch.equal(p.detach(), before[n])]
        assert moved, 'nothing trained at all'
        assert all('context_refiner' in n for n in moved), (
            f'the DiT moved under a refiner-only configuration: {moved[:5]}'
        )

    def test_the_teacher_changes_where_training_goes(self):
        with_teacher, _ = train(seed=0)
        without, _ = train(seed=0, teacher=False)
        differences = [
            (a - b).abs().max().item()
            for (_, a), (_, b) in zip(with_teacher.named_parameters(), without.named_parameters())
        ]
        assert max(differences) > 1e-6, (
            'training with the teacher landed on the same weights as without it, so the term '
            'is not reaching the optimizer'
        )

    def test_a_decaying_run_ends_closer_to_plain_training(self):
        """The point of the decay: the run should finish as ordinary ground-truth training."""
        constant, _ = train(cfg=teacher_cfg(decay='none'))
        decaying, _ = train(cfg=teacher_cfg(decay='linear', decay_steps=30))
        plain, _ = train(teacher=False)

        def distance(a, b):
            return sum(((pa - pb) ** 2).sum().item()
                       for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()))

        assert distance(decaying, plain) < distance(constant, plain)


def main():
    torch.manual_seed(0)
    print('Toy teacher-guided training: 2-block MiniTrainDIT, 2-layer refiner, CPU.')
    print('DiT frozen, refiner training, teacher = a separately initialised frozen DiT.')
    print('Every run sees the same data, the same timesteps and the same noise.\n')

    summary = []
    for label, cfg in (
        ('teacher off', None),
        ("decay = 'none'", teacher_cfg(decay='none')),
        ("decay = 'linear', 30 steps", teacher_cfg(decay='linear', decay_steps=30)),
    ):
        record = []
        student, losses = train(cfg=cfg, teacher=False if cfg is None else None, record=record)
        _, student_llm, _ = build_student_and_teacher(seed=0)
        print(f'--- {label} ---')
        print(f"{'step':>6} {'mean lambda':>12} {'train loss':>12}")
        for step, lam, loss in record[::10]:
            print(f'{step:6d} {lam:12.3f} {loss:12.4f}')
        gt = evaluate_ground_truth(student, student_llm)
        print(f"{'final':>6} {record[-1][1]:12.3f} {losses[-1]:12.4f}")
        print(f'  ground-truth eval (lambda = 0): {gt:.4f}\n')
        summary.append((label, losses[-1], gt))

    print('Reading this correctly matters:')
    print('  * TRAIN LOSS IS NOT COMPARABLE ACROSS THESE RUNS. With the teacher on, the target')
    print('    is a pre-averaged velocity and therefore an easier one, so the number falls for')
    print('    that reason alone. Compare the ground-truth eval column instead -- which is')
    print('    exactly why eval skips the teacher.')
    print('  * The decaying run ends at the teacher-off training loss, because by then it IS')
    print('    ground-truth training. That is the schedule working, not the run degrading.\n')
    print(f"{'run':>28} {'train loss':>12} {'gt eval':>10}")
    for label, final, gt in summary:
        print(f'{label:>28} {final:12.4f} {gt:10.4f}')
    print('\nThe toy teacher is a different random DiT, not a better model, so no ranking here')
    print('says anything about image quality. It shows the term is wired up and the schedule')
    print('does what it claims.')


if __name__ == '__main__':
    main()


class TestItIsVisibleThatTheTeacherIsOn:
    """A startup line proves the teacher loaded. It does not prove it is still contributing.

    Without a per-step number, a run whose decay reached zero an hour ago looks exactly like one
    training at full teacher weight.
    """

    def test_nothing_is_logged_when_there_is_no_teacher(self):
        pipe = build_pipeline(teacher=None)
        pipe.prepare_inputs(toy_batch())
        assert pipe.get_extra_log_scalars() == {}, (
            'a model with the feature off must add no scalars at all'
        )

    def test_lambda_is_reported_after_a_training_batch(self):
        _, _, guide = build_student_and_teacher()
        pipe = build_pipeline(teacher=guide)
        pipe.prepare_inputs(toy_batch())
        scalars = pipe.get_extra_log_scalars()
        assert 'train/teacher_lambda' in scalars
        assert 0 < scalars['train/teacher_lambda'] <= 1

    def test_it_reports_zero_once_the_schedule_has_decayed(self):
        """The number has to keep being reported after decay, not stop being reported.

        A metric that vanishes is indistinguishable from a logger that broke.
        """
        _, _, guide = build_student_and_teacher()
        cfg = teacher_cfg(decay='linear', decay_steps=10)
        pipe = build_pipeline(teacher=guide, cfg=cfg, step=10)
        pipe.prepare_inputs(toy_batch())
        assert pipe.get_extra_log_scalars()['train/teacher_lambda'] == 0.0

    def test_eval_does_not_overwrite_the_training_number(self):
        """Eval runs prepare_inputs too, and reports lambda = 0 by construction."""
        _, _, guide = build_student_and_teacher()
        pipe = build_pipeline(teacher=guide)
        pipe.prepare_inputs(toy_batch())
        during_training = pipe.get_extra_log_scalars()['train/teacher_lambda']
        pipe.prepare_inputs(toy_batch(), timestep_quantile=0.5)
        assert pipe.get_extra_log_scalars()['train/teacher_lambda'] == during_training

    def test_full_decay_is_announced_once(self, capsys):
        # No patching of is_main_process: should_announce() treats "no distributed backend" as
        # a single process, so prepare_inputs never needs one just to print.
        _, _, guide = build_student_and_teacher()
        cfg = teacher_cfg(decay='linear', decay_steps=10)
        pipe = build_pipeline(teacher=guide, cfg=cfg, step=10)
        pipe.prepare_inputs(toy_batch())
        pipe.prepare_inputs(toy_batch())
        printed = capsys.readouterr().out
        assert 'fully decayed' in printed
        assert printed.count('fully decayed') == 1, 'announced once, not on every batch'
