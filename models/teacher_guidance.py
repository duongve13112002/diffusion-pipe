# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teacher-guided training for anima_refiner.

Mixes the ground-truth flow-matching target with the velocity a frozen stock Anima predicts
from the same latent, the same timestep and the same caption -- differing only in the text
frontend. See docs/anima_refiner/teacher-guided-training.md for the argument; this module is
the schedule and the teacher forward, and nothing else.

    target = (1 - lam) * v_gt + lam * v_teacher
    lam    = loss_weight * shape(t) * decay(step)

Two things about lam are worth keeping in mind while reading:

  * For squared error, mixing the two LOSSES is algebraically the same as regressing onto the
    mixed TARGET above, because the prediction terms collect. So lam is exactly an interpolation
    weight, the two terms need no normalising, and this module builds the mixed target directly
    rather than computing two losses. .audit/exp_teacher_lambda.py measures this.
  * huber_delta / smooth_l1_beta break that identity, because they clip large residuals and the
    ground-truth residual is the large one. The pipeline still forms a mixed target -- the
    alternative, two clipped losses, distorts lam further -- but lam is approximate there, which
    teacher_warnings says once at startup.

Everything here is inert unless [teacher] loss_weight > 0.
"""

import math

import torch
from torch import nn


# 'constant' exists for ablation: it removes the timestep dependence without removing the term.
SHAPES = ('sigmoid', 'constant')
DECAYS = ('linear', 'cosine', 'none')


class TeacherConfig:
    """Validated [teacher] table. Construct through validate_teacher_config()."""

    def __init__(self, loss_weight, transformer_path, llm_path, shape, t_mid, width,
                 decay, decay_steps):
        self.loss_weight = loss_weight
        self.transformer_path = transformer_path
        self.llm_path = llm_path
        self.shape = shape
        self.t_mid = t_mid
        self.width = width
        self.decay = decay
        self.decay_steps = decay_steps

    @property
    def enabled(self):
        return self.loss_weight > 0


DISABLED = TeacherConfig(0.0, None, None, 'sigmoid', 0.5, 0.15, 'none', 0)


ANIMA_KEYS = ('transformer_path', 'llm_path')


def is_anima_teacher(table):
    """Whether a [teacher] table asks for a stock Anima as the teacher (it names its transformer or its text
    encoder). Without either key the table is the generic one: distillation toward the model's own starting
    weights (utils/model_teacher.py)."""
    return isinstance(table, dict) and any(key in table for key in ANIMA_KEYS)


def validate_teacher_config(config, use_context_refiner, cache_text_embeddings):
    """Read and check the [teacher] table, failing at startup rather than mid-run.

    Returns DISABLED when the table is absent or loss_weight is 0, in which case nothing else in
    this module is ever reached and no teacher is loaded.
    """
    table = config.get('teacher', None)
    if table is None:
        return DISABLED

    loss_weight = float(table.get('loss_weight', 0.0))
    if loss_weight < 0:
        raise ValueError(f'[teacher] loss_weight must be >= 0, got {loss_weight}')
    if loss_weight == 0:
        return DISABLED

    if not use_context_refiner:
        # The refiner's teacher term compares two text frontends, and only anima_refiner has a second one. Any other
        # type reads [teacher] as distillation toward a frozen copy of the model (utils/model_teacher.py).
        return DISABLED
    if not is_anima_teacher(table):
        # anima_refiner with no Anima named: the generic teacher, the refiner's own starting weights.
        return DISABLED
    if 'path' in table:
        raise ValueError('[teacher] path is the teacher of the generic distillation, the model\'s own weights or '
                         'another checkpoint of it. With transformer_path / llm_path the teacher is a stock Anima: '
                         'remove path, or remove transformer_path and llm_path to distill from the refiner itself.')
    if cache_text_embeddings:
        # The teacher and the student have to read one augmentation draw of one caption. Caching
        # freezes the draw, and a mismatch there changes no shape and no magnitude -- the loss
        # would compare two captions instead of two text frontends and look entirely healthy.
        raise ValueError(
            '[teacher] requires cache_text_embeddings = false.\n'
            '  The teacher and the student must encode the SAME augmented caption, drawn once. '
            'Cached embeddings freeze that draw, and a mismatch is invisible in the loss.\n'
            '  Set cache_text_embeddings = false in the config. VAE latents are cached either '
            'way; only the text half runs in the loop.'
        )

    if config.get('blocks_to_swap', 0):
        raise ValueError(
            '[teacher] cannot be combined with blocks_to_swap.\n'
            "  Block swapping is already moving the student's blocks between CPU and GPU every "
            'step; a second full DiT forward per step thrashes it.'
        )

    for key in ('transformer_path', 'llm_path'):
        if not table.get(key, None):
            raise ValueError(
                f'[teacher] {key} is required when loss_weight > 0. transformer_path must be a '
                'stock Anima checkpoint (the only one carrying an llm_adapter), and llm_path '
                'the text encoder that adapter was trained against -- Qwen3-0.6B-Base, NOT the '
                'student\'s encoder.'
            )

    shape = table.get('shape', 'sigmoid')
    if shape not in SHAPES:
        raise ValueError(f'[teacher] shape must be one of {SHAPES}, got {shape!r}')

    t_mid = float(table.get('t_mid', 0.5))
    if not 0.0 <= t_mid <= 1.0:
        raise ValueError(f'[teacher] t_mid must be in [0, 1], got {t_mid}')

    width = float(table.get('width', 0.15))
    if width <= 0:
        raise ValueError(f'[teacher] width must be > 0, got {width}')

    decay = table.get('decay', 'linear')
    if decay not in DECAYS:
        raise ValueError(f'[teacher] decay must be one of {DECAYS}, got {decay!r}')

    decay_steps = int(table.get('decay_steps', 0))
    if decay != 'none' and decay_steps <= 0:
        # Deliberately not defaulted to "the run length": prepare_inputs cannot see the run
        # length, and a schedule that silently means something different per config is worse
        # than one the user states.
        raise ValueError(
            f'[teacher] decay = {decay!r} requires decay_steps > 0 (the number of optimizer '
            'steps over which lambda falls to zero). Set decay = \'none\' to keep the teacher '
            'on for the whole run -- but read the ceiling section of '
            'docs/anima_refiner/teacher-guided-training.md first: a teacher that never decays '
            'pins the student to Qwen3-0.6B.'
        )

    return TeacherConfig(loss_weight, table['transformer_path'], table['llm_path'],
                         shape, t_mid, width, decay, decay_steps)


def should_announce():
    """Whether this process should print an informational message.

    utils.common.is_main_process() asserts a DeepSpeed backend is initialised, which is true
    during training and not true in a tool, a test, or anything driving prepare_inputs directly.
    No backend means one process, and one process is the main one -- the same single-process
    semantics test/conftest.py's stub implements. Printing must never be the reason a code path
    that would otherwise work raises.
    """
    try:
        from utils.common import is_main_process
        return is_main_process()
    except Exception:
        return True


def teacher_warnings(cfg, config):
    """Advisory messages for a valid but risky configuration.

    Separate from validate_teacher_config because validation runs in the pipeline constructor,
    which tools reach on a CPU box with no distributed backend -- and is_main_process() asserts
    one is initialised. Validation stays pure; this is called from the training path only.
    """
    messages = []
    if cfg.decay == 'none':
        messages.append(
            "WARNING: [teacher] decay = 'none'. The teacher term never fades, so the student is "
            "held at the teacher's ceiling (Qwen3-0.6B) for the whole run. This is rarely what "
            'you want -- see docs/anima_refiner/teacher-guided-training.md.'
        )
    if 'huber_delta' in config or 'smooth_l1_beta' in config:
        messages.append(
            'WARNING: huber_delta/smooth_l1_beta is set together with [teacher]. Both clip '
            'large residuals, and the ground-truth residual is the large one, so lambda is '
            'approximate rather than exact -- measured drifting from a nominal 0.5 to about '
            '0.585 at t = 0.9. See .audit/exp_teacher_lambda.py.'
        )
    return messages


def shape_at(t, cfg):
    """Timestep shape of lambda, in [0, 1]. t is the timestep AFTER shift, so it is the real
    noise level and no shift correction belongs here.

    Rising in t: the teacher is wanted where the caption determines the target and one
    ground-truth draw is a noisy estimate of it, not where the latent is nearly a finished image
    and the texture being learned should come from the dataset.
    """
    if cfg.shape == 'constant':
        return torch.ones_like(t)
    return torch.sigmoid((t - cfg.t_mid) / cfg.width)


def decay_at(step, cfg):
    """Training-progress factor of lambda, in [0, 1]. A plain float, not per sample."""
    if cfg.decay == 'none':
        return 1.0
    if cfg.decay_steps <= 0:
        return 0.0
    progress = min(max(step / cfg.decay_steps, 0.0), 1.0)
    if cfg.decay == 'linear':
        return 1.0 - progress
    # cosine
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lambda_at(t, step, cfg):
    """Per-sample lambda for a batch of timesteps.

    Args:
        t: (B, 1) timesteps, after shift.
        step: optimizer step, for the decay schedule.
        cfg: TeacherConfig.

    Returns:
        (B, 1) lambda in [0, loss_weight], broadcastable against a latent.
    """
    if not cfg.enabled:
        return torch.zeros_like(t)
    return cfg.loss_weight * shape_at(t, cfg) * decay_at(step, cfg)


def blend_target(v_gt, v_teacher, lam):
    """The mixed target lambda interpolates toward.

    lam arrives as (B, 1) and the targets as (B, C, T, H, W), so it is reshaped to broadcast on
    the batch dimension alone -- every element of one sample shares that sample's timestep.

    The shapes are checked rather than left to broadcasting. Two velocities of different channel
    counts mostly raise, but a teacher emitting a single channel would broadcast across all of
    the student's instead, producing a full-sized target built from one channel of prediction
    and no error anywhere.
    """
    if v_teacher.shape != v_gt.shape:
        raise RuntimeError(
            f'Teacher velocity {tuple(v_teacher.shape)} does not match the ground-truth target '
            f'{tuple(v_gt.shape)}. The teacher must predict the same latent shape as the '
            'student, which means the same VAE and the same channel count.'
        )
    lam = lam.reshape(lam.shape[0], *([1] * (v_gt.ndim - 1))).to(v_gt.device, v_gt.dtype)
    return (1.0 - lam) * v_gt + lam * v_teacher.to(v_gt.device, v_gt.dtype)


class TeacherGuide(nn.Module):
    """A frozen stock Anima, held for its velocity prediction only.

    Never wrapped by DeepSpeed, never given an optimizer, never checkpointed: it is a constant
    function of (x_t, t, caption). Its forward runs under no_grad inside prepare_inputs, on the
    main process, which is where prepare_inputs already runs (utils/dataset.py).

    `dit` may be the student's own transformer. That is correct exactly when every DiT learning
    rate is zero and no adapter targets the DiT, because then the student's DiT is bit-identical
    to the teacher's for the whole run and a second copy would be 3.5-4 GB of duplicate weights.
    The pipeline decides that from the resolved learning rates and passes `shares_dit` so the
    decision is visible here rather than inferred.
    """

    def __init__(self, dit, llm_adapter, text_encoder, tokenizer, t5_tokenizer,
                 max_text_length, shares_dit=False):
        super().__init__()
        self.dit = [dit]  # not a submodule: it must not be registered, moved or saved with this
        self.llm_adapter = llm_adapter
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_text_length = max_text_length
        self.shares_dit = shares_dit
        self.requires_grad_(False)
        self.eval()

    def to_device(self, device, dtype=None):
        self.llm_adapter.to(device, dtype)
        self.text_encoder.to(device, dtype)
        if not self.shares_dit:
            self.dit[0].to(device, dtype)
        return self

    def tokenize(self, captions):
        """The teacher's own token tensors for a batch of captions, on the CPU: the LLM's (ids, mask) and the T5
        tokenizer's (ids, mask) the llm_adapter's queries read. The teacher pipeline carries them in the label."""
        from models.cosmos_predict2 import _tokenize

        source = _tokenize(self.tokenizer, captions, self.max_text_length)
        target = _tokenize(self.t5_tokenizer, captions, self.max_text_length)
        return source.input_ids, source.attention_mask, target.input_ids, target.attention_mask

    def release(self):
        """Free the teacher's weights once lambda is 0 for good. Its LLM and llm_adapter are its own; the DiT only
        when it is not the student's."""
        from utils.model_teacher import empty_weights, free_gpu_cache
        if getattr(self, 'released', False):
            return
        self.released = True
        modules = [self.llm_adapter, self.text_encoder]
        if not self.shares_dit:
            modules.append(self.dit[0])
        empty_weights(modules)
        free_gpu_cache()

    @torch.no_grad()
    def text_features(self, captions, device):
        """LLMAdapter output for a batch of caption strings: (B, max_text_length, 1024).

        The caption strings must be the SAME objects the student tokenizes -- one augmentation
        draw, handed to both. This takes them already drawn; it never re-reads or re-augments.
        """
        from models.cosmos_predict2 import _compute_text_embeddings, _tokenize

        source = _tokenize(self.tokenizer, captions, self.max_text_length)
        target = _tokenize(self.t5_tokenizer, captions, self.max_text_length)
        source_ids = source.input_ids.to(device)
        source_mask = source.attention_mask.to(device)
        target_ids = target.input_ids.to(device)
        target_mask = target.attention_mask.to(device)

        hidden = _compute_text_embeddings(self.text_encoder, source_ids, source_mask)
        feats = self.llm_adapter(
            source_hidden_states=hidden,
            target_input_ids=target_ids,
            target_attention_mask=target_mask,
            source_attention_mask=source_mask,
        )
        # Matches LLMAdapterLayer: padded positions carry nothing.
        feats[~target_mask.bool()] = 0
        return feats

    @torch.no_grad()
    def velocity(self, noisy_latents, t, captions):
        """The frozen DiT's velocity for this latent, timestep and caption.

        Identical inputs to the student's forward in every respect except the text frontend.
        That is the whole point: anything else that differed would land in the loss as though it
        were text-frontend error.
        """
        import utils.common

        device = noisy_latents.device
        dit = self.dit[0]

        # The padding mask is concatenated onto the latents inside prepare_embedded_sequence,
        # and torch.cat PROMOTES: a mask in a wider dtype silently drags the latents up with it,
        # and the result then meets the DiT's own weights and raises at the first Linear. So it
        # is built from the tensor it will be concatenated with and from nothing else.
        padding_mask = torch.zeros(
            noisy_latents.shape[0], 1, noisy_latents.shape[3], noisy_latents.shape[4],
            dtype=noisy_latents.dtype, device=device,
        )

        # Run under the same autocast every pipeline layer is decorated with, rather than
        # casting the inputs by hand. Hand-casting has to pick one dtype, and a DiT loaded with
        # transformer_dtype does not have one -- KEEP_IN_HIGH_PRECISION leaves x_embedder and
        # final_layer wider than the blocks. Autocast is also what the student's forward does,
        # so this keeps the two numerically comparable, which is the entire point of the term.
        autocast_dtype = utils.common.AUTOCAST_DTYPE
        was_training = dit.training
        dit.eval()
        try:
            with torch.autocast('cuda', dtype=autocast_dtype,
                                enabled=torch.cuda.is_available() and autocast_dtype is not None):
                # Inside the autocast, because the student computes its text features inside one
                # too -- InitialLayer and ContextRefinerLayer carry the same decorator.
                feats = self.text_features(captions, device)
                out = dit(noisy_latents, t.reshape(-1), feats, padding_mask=padding_mask)
        finally:
            # The student's DiT may be the same module, and train.py put it in train mode.
            if was_training:
                dit.train()
        return out.to(noisy_latents.dtype)


class AnimaLabelHooks:
    """The teacher pipeline's view of the Anima teacher (utils/model_teacher.py TeacherPipeline).

    With several pipeline stages prepare_inputs does not mix the target itself: its label for a training batch is
    (target, t, source_ids, source_mask, t5_ids, t5_mask, mask), which the loader broadcasts from the first stage
    to the last. The teacher pipeline reads the noisy latent from the features and (t, the teacher's own tokens)
    from the label, and the last stage mixes the teacher's velocity into the target with this step's lambda,
    leaving the label (mixed target, mask) the loss function reads as ever.
    """
    keep_float32 = False

    def __init__(self, cfg, current_step):
        self.cfg = cfg
        self.current_step = current_step

    def bind_step(self, current_step):
        """The step the teacher pipeline decides `finished()` from, so lambda and the decision agree."""
        self.current_step = current_step

    def inputs(self, features, label):
        noisy_latents, t, *tokens, _mask = (features[0], *label[1:])
        return (noisy_latents, t, *tokens)

    def targets(self, label):
        return (label[0],)

    def with_teacher(self, label, teacher_outputs):
        target, t, mask = label[0], label[1], label[-1]
        lam = lambda_at(t.to(target.device, torch.float32), self.current_step(), self.cfg)
        return (blend_target(target, teacher_outputs[0], lam), mask)

    def without_teacher(self, label):
        return (label[0], label[-1])
