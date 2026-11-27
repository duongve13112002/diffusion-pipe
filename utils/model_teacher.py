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

import gc
import math

import torch

from models.teacher_guidance import DECAYS, decay_at, is_anima_teacher
from utils import fp8_training
from utils.common import all_ok, warn_ignored
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
    """The teacher on one rank: its layers (all of them with one pipeline stage, where the loader runs them;
    with several the TeacherPipeline's engine holds them), the step, and lambda.

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
            raise RuntimeError('This rank holds no teacher layers (with several pipeline stages the teacher runs in the '
                               'TeacherPipeline, not here)')
        x = tuple(t.to(self.device) if torch.is_tensor(t) and self.device is not None else t for t in features)
        # The student's fp8 attention sites count the calls of its own blocks; the teacher is plain bf16.
        with fp8_training.disabled():
            for layer in self.layers:
                x = layer(x)
        outputs = self.select(x)
        return [t.detach() for t in outputs]

    def release(self, protect=None):
        """Free the teacher's weights once it has finished (lambda is 0 for good): they are this teacher's own
        copies, and nothing reads them again. `protect`: the student's module, whose weights a teacher layer may
        share (a text encoder both hold) and are kept."""
        if self.layers is not None:
            empty_weights(self.layers, owned_ids(protect))
            self.layers = None
            free_gpu_cache()


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


def teacher_partition(layers, num_stages, partition_method, partition_split):
    """(partition_method, partition_split) for the teacher pipeline's module. The student's 'parameters' method
    weighs a layer by its trainable parameters, and every teacher layer is frozen (all weights 0, so nearly all
    layers land on stage 0): the teacher is balanced by its parameter counts instead. 'uniform' and an explicit
    'manual' split are the student's own (the same layer list)."""
    method = str(partition_method).lower()
    if method == 'uniform' or (method == 'manual' and partition_split is not None):
        return partition_method, partition_split
    return 'manual', balanced_split(layers, num_stages)


def balanced_split(layers, num_stages):
    """Stage boundaries (without the 0 and the end) that balance the layers' parameter counts."""
    from deepspeed.runtime.utils import partition_balanced
    weights = [sum(p.numel() for p in layer.parameters()) if isinstance(layer, torch.nn.Module) else 0
               for layer in layers]
    return [int(b) for b in partition_balanced(weights, num_stages)[1:-1]]


def release_foreign_weights(layers, module, protect=None):
    """Free the weights of the teacher layers that are not `module`'s (the other stages'). Several layers hold a
    hidden reference to the whole copied model (a list, so it is not a parameter of theirs), which keeps the
    other stages' blocks alive in host memory for the whole run. `protect`: the student's module, whose weights a
    teacher layer may share (a text encoder both hold) and are kept."""
    kept = {id(t) for t in (*module.parameters(), *module.buffers())} | owned_ids(protect)
    empty_weights(layers, kept)


def empty_weights(layers, kept=()):
    """Replace the parameters and buffers of `layers` (modules; anything else is skipped) that are not in `kept`
    (ids) by empty ones. The modules' entries are replaced rather than emptied in place: a tensor subclass (a
    quantized weight) keeps its payload in attributes that `.data = ...` does not drop."""
    for layer in layers:
        if not isinstance(layer, torch.nn.Module):
            continue
        for m in layer.modules():
            for name, p in list(m._parameters.items()):
                if p is not None and id(p) not in kept:
                    m._parameters[name] = torch.nn.Parameter(torch.empty(0, dtype=p.dtype), requires_grad=False)
            for name, b in list(m._buffers.items()):
                if b is not None and id(b) not in kept:
                    m._buffers[name] = torch.empty(0, dtype=b.dtype)


def owned_ids(module):
    if module is None:
        return set()
    return {id(t) for t in (*module.parameters(), *module.buffers())}


def free_gpu_cache():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _release_engine_buffers(engine):
    """What the teacher engine keeps after a forward (two micro batches' inputs, outputs and the received
    activations on every stage) would sit on the GPU through the student's step."""
    for buffers in getattr(engine, 'pipe_buffers', {}).values():
        for i in range(len(buffers)):
            buffers[i] = None
    engine.pipe_recv_buf = None
    engine.fwd_outputs = []
    engine.loss = None
    engine.total_loss = None
    engine.total_additional_losses = None


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
        layers = model.make_teacher_layers(getattr(config, 'path', None))
    for layer in layers:
        # A layer can be a plain function (AuraFlow's concat_hidden_states).
        if isinstance(layer, torch.nn.Module):
            layer.requires_grad_(False)
            layer.eval()
            if place:
                layer.to(device)
    teacher.layers = layers
    return teacher


class LabelHooks:
    """How a TeacherPipeline reads a micro batch and puts the teacher's output into its label. The default
    appends the output after the label's last entry (the loss then reads it there). A model whose label carries
    more than the loss reads (the Anima teacher's own text tokens) or whose teacher mixes into the target
    subclasses it.

    `label_targets`: how many leading label entries the teacher's output stands in for (default: all but the
    mask, the last). `keep_float32`: leave the output in float32 instead of the target's dtype."""

    def __init__(self, label_targets=None, keep_float32=False):
        self.label_targets = label_targets
        self.keep_float32 = keep_float32

    def inputs(self, features, label):
        """The features the teacher engine's first stage reads for this micro batch."""
        return features

    def targets(self, label):
        """The label entries the teacher's outputs stand in for, one per output tensor."""
        return label[:len(label) - 1 if self.label_targets is None else self.label_targets]

    def with_teacher(self, label, teacher_outputs):
        return (*label, *teacher_outputs)

    def without_teacher(self, label):
        """The label once lambda has decayed to 0 for good: no teacher output is computed."""
        return (*label, *[torch.zeros_like(t) for t in self.targets(label)])


class TeacherPipeline:
    """The teacher as a second, forward-only pipeline engine with the student's number of stages.

    The teacher's layers are split over the stages like the student's, so every GPU holds 1/stages of it (with
    one stage ModelTeacher.run in the data loader runs it on the student's GPU). Every step, before the student's train_batch, all ranks run the teacher engine over the
    step's micro batches (a collective: every rank calls attach, in the same order on every rank); the last stage
    then has the teacher's output of each micro batch and puts it in that micro batch's label, which is what
    ModelTeacher.run's output joins in the loader, so the loss function is the same.
    """

    def __init__(self, teacher, engine, hooks=None, protect=None):
        self.teacher = teacher
        self.engine = engine
        self.protect = protect
        self.hooks = hooks or LabelHooks()
        if hasattr(self.hooks, 'bind_step'):
            self.hooks.bind_step(lambda: teacher.step)
        self.released = False

    def attach(self, items):
        """`items`: the step's micro batches [(features, label)] on the first and last stage, None on the others.
        Returns them with the teacher's targets appended to each label on the last stage (zeros once lambda has
        decayed to 0, without running the engine: every rank knows the step)."""
        engine, teacher = self.engine, self.teacher
        last = engine.is_last_stage()
        if teacher.finished():
            if not self.released:
                # every rank takes this step together: the engine's stage of the teacher is of no use any more
                empty_weights([engine.module], owned_ids(self.protect))
                # The engine keeps every Parameter it was given (a dict of names), which would hold their storage
                getattr(engine, 'param_names', {}).clear()
                self.released = True
                free_gpu_cache()
            if not last:
                return items
            return [(features, self.hooks.without_teacher(label)) for features, label in items]
        # eval_batch never clears the totals the last stage adds every forward to (only train_batch does), so a
        # micro batch of another shape than the first would fail to add into them.
        engine.total_loss = None
        engine.total_additional_losses = None
        engine.reset_activation_shape()
        with fp8_training.disabled():
            teacher_items = (None if items is None else
                             [(self.hooks.inputs(features, label), label) for features, label in items])
            outputs = engine.eval_batch(iter(teacher_items) if items is not None else None, compute_loss=False,
                                        reduce_output=None, bcast_loss=False)
        _release_engine_buffers(engine)
        attached, error = items, None
        if last:
            try:
                attached = self._attach_outputs(outputs, items)
            except Exception as e:
                error = e
        # The last stage alone can fail here; every rank stops with it instead of waiting in the student's step.
        all_ok(error)
        return attached

    def _attach_outputs(self, outputs, items):
        teacher = self.teacher
        if len(outputs) != len(items):
            raise RuntimeError(f'[teacher]: the teacher pipeline returned {len(outputs)} outputs for '
                               f'{len(items)} micro batches')
        attached = []
        for output, (features, label) in zip(outputs, items):
            targets = self.hooks.targets(label)
            if not torch.is_tensor(output):
                output = tuple(output)
            selected = teacher.select(output)
            if len(selected) != len(targets):
                raise RuntimeError(f'[teacher]: the teacher gives {len(selected)} output tensors and the label '
                                   f'has {len(targets)} targets')
            teacher_targets = []
            for out, target in zip(selected, targets):
                out = fit_trailing_dim(out.detach(), target)
                out = out.to(target.device, torch.float32 if self.hooks.keep_float32 else target.dtype)
                if out.shape != target.shape:
                    raise RuntimeError(f'[teacher]: the teacher output {tuple(out.shape)} does not match the '
                                       f'target {tuple(target.shape)} it stands in for')
                teacher_targets.append(out)
            attached.append((features, self.hooks.with_teacher(label, teacher_targets)))
        return attached


def label_hooks_for(model, own_teacher):
    """The label hooks of the teacher pipeline. `own_teacher`: the model's own [teacher] (own_teacher_config)
    drives it, so the model says how its label carries the teacher; the generic [teacher] of any model, anima
    included, is the plain LabelHooks."""
    hooks = model.teacher_label_hooks() if own_teacher and hasattr(model, 'teacher_label_hooks') else None
    if hooks is not None:
        return hooks
    return LabelHooks(label_targets=getattr(model, 'teacher_label_targets', None) if own_teacher else None,
                      keep_float32=getattr(model, 'teacher_output_float32', False) if own_teacher else False)


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
