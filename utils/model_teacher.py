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

How it runs. The batch's features are what prepare_inputs makes for the first pipeline stage. There the teacher
(a second chain of layers from the model's own to_layers(), no gradients, eval mode) runs them, and its output
travels to the last stage the way the target does (utils/dataset.py PipelineDataLoader), appended to the label.
Any number of pipeline stages therefore works, and no layer has to pass an extra tensor on. The first-stage
ranks hold the whole teacher on their GPU; the others hold none.

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
        raise ValueError(f"[{KEY}] decay_steps does not apply with decay = 'none' (nothing falls): set decay = 'linear' "
                         "or 'cosine', or remove decay_steps")
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


def build_teacher(model, config, device, runs_teacher, all_ranks_run=True):
    """The ModelTeacher of this rank. On a rank that runs it (the first pipeline stage) the layers are built
    from a frozen copy of the model's denoiser (CommonPipeline.make_teacher_layers) and placed on `device`.
    When only some ranks run it (all_ranks_run False) the loaders do not wait for the others (one_at_a_time):
    the ranks that do not run it never reach them."""
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
            layer.to(device)
    teacher.layers = layers
    return teacher


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
