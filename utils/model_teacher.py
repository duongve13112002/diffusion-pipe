"""Distillation toward a frozen teacher, for every model but MOSS-Image and anima_refiner.

    loss = L(student, target) + lambda(step) * L(student, teacher)

The second term is the model's own loss function with the teacher's output in place of the target, so the
model's masking, huber_delta / smooth_l1_beta and weights apply to it as they do to the data loss. lambda =
loss_weight * decay(step) (the decay helpers of models/teacher_guidance.py, as in utils/distillation.py).

    [teacher]
    loss_weight = 0.5
    decay = 'cosine'        # 'none' (default), 'linear' or 'cosine'
    decay_steps = 20000
    path = '/models/other_checkpoint'   # optional; the same key [model] names the weights with

Without `path` the teacher is the model the run starts from, as loaded (before an adapter or fp8 training is
applied to it). With one, it is another checkpoint of the same architecture, read the way [model] reads its own.
The text encoders, the VAE and the latent space are the student's: the teacher sees the same noisy latents,
timesteps and text embeddings.

How it runs. The batch's features are what prepare_inputs makes for the first pipeline stage; the teacher (a second
chain of layers from the model's own to_layers(), no gradients, eval mode) runs them and its output reaches the last
stage the way the target does, appended to the label (utils/dataset.py PipelineDataLoader). With one pipeline stage
the loader runs it (ModelTeacher.run) on the student's GPU. With several, TeacherPipeline runs it as a second,
forward-only DeepSpeed pipeline with the student's stages, so each GPU holds 1/stages of it, and puts its output in
the label before the student's train_batch. No layer has to pass an extra tensor on either way.

MOSS-Image has its own implementation (utils/distillation.py: noise-level dependent lambda, the teacher inside
the first layer). anima_refiner has two forms: a [teacher] that names transformer_path / llm_path distills from a
stock Anima by mixing the target (models/teacher_guidance.py); one that names neither is this teacher, toward the
refiner's own starting weights like any other model.

Kept free of training-stack imports so it can be unit-tested on CPU.
"""

import math

import torch

from models.teacher_guidance import DECAYS, decay_at, is_anima_teacher
from utils import fp8_training
from utils.common import warn_ignored
from utils.distillation import DistillConfig, distill_warnings

KEY = 'teacher'
_KEYS = ('loss_weight', 'decay', 'decay_steps', 'path')
_NOISE_LEVEL_KEYS = ('shape', 't_mid', 'width')


class ModelTeacherConfig(DistillConfig):
    def __init__(self, loss_weight, decay, decay_steps, path):
        super().__init__(loss_weight, 'constant', 0.5, 0.15, decay, decay_steps)
        self.path = path


def _number(table, key, default):
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'[{KEY}] {key} must be a finite number, got {value!r}')
    return float(value)


def parse_model_teacher_config(config):
    """The validated [teacher] table as a ModelTeacherConfig, or None when it does not apply: no table,
    loss_weight 0, or a model type that reads [teacher] itself. Raises on an unknown key, a value out of
    range, or a setting the teacher cannot run with (block swapping)."""
    table = config.get(KEY)
    if table is None:
        return None
    model_type = config.get('model', {}).get('type')
    if model_type == 'moss_image':
        return None
    if model_type == 'anima_refiner' and is_anima_teacher(table):
        # A stock Anima as the teacher: the refiner's own [teacher], which mixes the target.
        return None
    if not isinstance(table, dict):
        raise ValueError(f'[{KEY}] must be a table, got {table!r}')
    noise = [key for key in _NOISE_LEVEL_KEYS if key in table]
    if noise:
        raise ValueError(f'[{KEY}] {", ".join(noise)}: a loss weight that follows the noise level needs the noise '
                         'level of each sample, which only MOSS-Image hands to its loss. Here lambda depends on the '
                         "step alone: remove them (keys: " + ', '.join(_KEYS) + ').')
    unknown = sorted(set(table) - set(_KEYS))
    if unknown:
        raise ValueError(f'[{KEY}] has unknown keys {unknown}; known keys: {", ".join(_KEYS)}')
    loss_weight = _number(table, 'loss_weight', 0.0)
    if loss_weight < 0:
        raise ValueError(f'[{KEY}] loss_weight must be >= 0, got {loss_weight}')
    if loss_weight == 0:
        return None
    decay = table.get('decay', 'none')
    if decay not in DECAYS:
        raise ValueError(f'[{KEY}] decay must be one of {DECAYS}, got {decay!r}')
    decay_steps = table.get('decay_steps', 0)
    if isinstance(decay_steps, bool) or not isinstance(decay_steps, int) or decay_steps < 0:
        raise ValueError(f'[{KEY}] decay_steps must be a whole number >= 0, got {decay_steps!r}')
    if decay == 'none' and decay_steps > 0:
        warn_ignored(f"[{KEY}] decay_steps is ignored with decay = 'none' (nothing falls): set decay = 'linear' "
                     "or 'cosine' for it to apply.")
        decay_steps = 0
    if decay != 'none' and decay_steps <= 0:
        raise ValueError(f'[{KEY}] decay = {decay!r} needs decay_steps > 0 (the optimizer steps over which '
                         "lambda falls to 0); use decay = 'none' to keep the term on for the whole run")
    path = table.get('path')
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError(f'[{KEY}] path must be a non-empty string, got {path!r}')
    if config.get('blocks_to_swap', 0):
        raise ValueError(f'[{KEY}] cannot be combined with blocks_to_swap: block swapping already moves the '
                         'blocks between CPU and GPU every step, and the teacher is a second full forward.')
    adapter = config.get('adapter')
    if path is None and isinstance(adapter, dict) and adapter.get('init_from_existing'):
        raise ValueError(f'[{KEY}] cannot be combined with [adapter] init_from_existing unless [{KEY}] sets a path: '
                         'the teacher is then the model with no adapter, not the base plus the adapter the run '
                         'starts from, so the term would pull the first steps away from the starting adapter.')
    return ModelTeacherConfig(loss_weight, decay, int(decay_steps), path)


def warnings_for(teacher_config, config):
    return distill_warnings(teacher_config, config)


class ModelTeacher:
    """The teacher on one rank: its layers (first pipeline stage only), the step, and lambda.

    `layers` is None on a rank that does not run it. `select` turns the chain's output into the tensors that
    stand in for the label's targets, in the label's order (CommonPipeline.teacher_targets_of).
    """

    def __init__(self, config, layers=None, select=None, device=None):
        self.config = config
        self.layers = layers
        self.select = select
        self.device = device
        self.step = 1

    def set_step(self, step):
        self.step = step

    def lam(self):
        return self.config.loss_weight * decay_at(self.step, self.config)

    def finished(self):
        """Whether the decay has reached zero for good: the teacher's output no longer reaches the loss."""
        return self.config.decay != 'none' and self.step >= self.config.decay_steps

    @torch.no_grad()
    def run(self, features):
        """The teacher's output for one micro batch of features: a list of tensors, one per target."""
        if self.layers is None:
            raise RuntimeError('This rank holds no teacher layers: only the first pipeline stage runs the teacher')
        x = tuple(t.to(self.device) if torch.is_tensor(t) and self.device is not None else t for t in features)
        # The student's fp8 attention sites count the calls of its own blocks; the teacher is plain bf16.
        with fp8_training.disabled():
            for layer in self.layers:
                x = layer(x)
        outputs = self.select(x)
        return [t.detach() for t in outputs]


def on_first_stage(pipeline_stages):
    """Whether this rank is on the first pipeline stage (DeepSpeed numbers ranks stage-major)."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() // max(1, torch.distributed.get_world_size() // int(pipeline_stages)) == 0


def fit_trailing_dim(output, target):
    """`output` cut or zero-padded along its last dimension to the target's, when that is the only way they
    differ (MiniMax-H3 pads the audio target of a batch to its longest clip while each clip's audio output keeps
    its own length, and its loss crops the target to the output's). Anything else is returned unchanged."""
    # The batch dimension is not compared: the output is one micro batch, the target the whole batch.
    if (output.ndim != target.ndim or output.ndim < 2 or output.shape[1:-1] != target.shape[1:-1]
            or output.shape[-1] == target.shape[-1]):
        return output
    if output.shape[-1] > target.shape[-1]:
        return output[..., :target.shape[-1]]
    return torch.nn.functional.pad(output, (0, target.shape[-1] - output.shape[-1]))


def build_teacher(model, config, device, runs_teacher=True, all_ranks_run=True, place=True):
    """The ModelTeacher of this rank: its layers are built from a frozen copy of the model's denoiser
    (CommonPipeline.make_teacher_layers), frozen, in eval mode, and put on `device` when `place` is true.
    With place false the layers stay where they were built (CPU) for a TeacherPipeline, whose engine moves
    each stage's layers to its GPU. A rank that does not run the teacher (runs_teacher false) builds nothing;
    then all_ranks_run false tells the loaders not to wait for the others (one_at_a_time)."""
    teacher = ModelTeacher(config, select=model.teacher_targets_of, device=device)
    if not runs_teacher:
        return teacher
    import contextlib
    from utils.common import barriers_off
    with contextlib.nullcontext() if all_ranks_run else barriers_off():
        layers = model.make_teacher_layers(config.path)
    for layer in layers:
        # A layer can be a plain function (AuraFlow's concat_hidden_states).
        if isinstance(layer, torch.nn.Module):
            layer.requires_grad_(False)
            layer.eval()
            if place:
                layer.to(device)
    teacher.layers = layers
    return teacher


class TeacherPipeline:
    """The teacher as a second, forward-only pipeline engine with the student's number of stages.

    The first-stage teacher (ModelTeacher.run in the data loader) holds the whole teacher on the first stage's
    GPUs. Here the teacher's layers are split over the stages like the student's, so every GPU holds
    1/stages of it. Every step, before the student's train_batch, all ranks run the teacher engine over the
    step's micro batches (a collective: every rank calls attach, in the same order on every rank); the last stage
    then has the teacher's output of each micro batch and puts it in that micro batch's label, which is what
    ModelTeacher.run's output joins in the loader, so the loss function is the same.
    """

    def __init__(self, teacher, engine, label_targets=None, keep_float32=False):
        """`label_targets`: how many leading entries of a label are targets the teacher's output stands in for
        (default: all but the mask, the last); the output joins the label after its last entry. `keep_float32`:
        leave the teacher's output in float32 instead of casting it to its target's dtype (MOSS-Image's loss)."""
        self.teacher = teacher
        self.engine = engine
        self.label_targets = label_targets
        self.keep_float32 = keep_float32

    def attach(self, items):
        """`items`: the step's micro batches [(features, label)] on the first and last stage, None on the others.
        Returns them with the teacher's targets appended to each label on the last stage (zeros once lambda has
        decayed to 0, without running the engine: every rank knows the step)."""
        engine, teacher = self.engine, self.teacher
        last = engine.is_last_stage()
        if teacher.finished():
            if not last:
                return items
            return [(features, (*label, *[torch.zeros_like(t) for t in label[:self._count(label)]]))
                    for features, label in items]
        engine.reset_activation_shape()
        with fp8_training.disabled():
            outputs = engine.eval_batch(iter(items) if items is not None else None, compute_loss=False,
                                        reduce_output=None, bcast_loss=False)
        if not last:
            return items
        if len(outputs) != len(items):
            raise RuntimeError(f'[teacher]: the teacher pipeline returned {len(outputs)} outputs for '
                               f'{len(items)} micro batches')
        attached = []
        for output, (features, label) in zip(outputs, items):
            targets = label[:self._count(label)]
            if not torch.is_tensor(output):
                output = tuple(output)
            selected = teacher.select(output)
            if len(selected) != len(targets):
                raise RuntimeError(f'[teacher]: the teacher gives {len(selected)} output tensors and the label '
                                   f'has {len(targets)} targets')
            teacher_targets = []
            for out, target in zip(selected, targets):
                out = fit_trailing_dim(out.detach(), target)
                out = out.to(target.device, torch.float32 if self.keep_float32 else target.dtype)
                if out.shape != target.shape:
                    raise RuntimeError(f'[teacher]: the teacher output {tuple(out.shape)} does not match the '
                                       f'target {tuple(target.shape)} it stands in for')
                teacher_targets.append(out)
            attached.append((features, (*label, *teacher_targets)))
        return attached

    def _count(self, label):
        return len(label) - 1 if self.label_targets is None else self.label_targets


def wrap_loss_fn(inner, teacher):
    """The model's loss plus lambda times the same loss toward the teacher.

    A training batch's label is (*targets, mask, *teacher_targets) (PipelineDataLoader); an evaluation batch's
    is the plain (*targets, mask), and runs without gradients, which is how the two are told apart.
    """
    def loss_fn(output, label):
        if not torch.is_grad_enabled():
            return inner(output, label)
        if len(label) < 3 or len(label) % 2 == 0:
            raise RuntimeError(f'[{KEY}] is on and this is a training batch, but its label has {len(label)} entries '
                               'and so carries no teacher targets (a label is (*targets, mask, *teacher targets))')
        count = (len(label) - 1) // 2
        data_label = tuple(label[:count + 1])
        teacher_label = (*label[count + 1:], label[count])
        loss = inner(output, data_label)
        lam = teacher.lam()
        if lam == 0:
            return loss
        return loss + lam * inner(output, teacher_label)
    return loss_fn
