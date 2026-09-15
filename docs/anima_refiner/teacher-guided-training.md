# Teacher-guided training

**Status: proposal. None of this is implemented.** No config key described here is read by any
code in the repo today. This document is the design to argue with before it gets built.

Distillation today lives in `tools/distill_refiner.py`: captions only, no images, no VAE, no
diffusion loss. This proposes the other half — keeping the teacher around during ordinary
diffusion training in `train.py`, and blending its prediction with the ground-truth target by
timestep.

## What this is, in one formula

Ordinary training compares the model's velocity against the ground-truth one:

```
L = ‖v_student − v_gt‖²                  v_gt = noise − latents
```

This proposes a second target, from a frozen stock Anima given the *same* `x_t`, the *same* `t`
and the *same* caption — differing only in the text frontend:

```
L = (1 − λ) · ‖v_student − v_gt‖²  +  λ · ‖v_student − v_teacher‖²

λ = λ(t, step)
```

Everything hangs on λ. The rest of this document is where its shape comes from.

## Why this is not the rollout that already exists

[denoising-rollout.md](./denoising-rollout.md) implements
[Scaling Down Text Encoders (CVPR 2025)](https://arxiv.org/html/2503.19897v1) faithfully, and
that paper's objective is

> ℒ_vision = 𝔼‖μ_θ(x_t, t, ω_teacher(p)) − μ_θ(x_t, t, ω_student(p))‖²

— teacher only, no ground truth, and `x_t` taken from a rollout out of pure noise because that
stage has no images.

Ordinary training *does* have images, so `x_t` is the real thing rather than a walk from noise,
and the ground-truth target exists and is free. The proposal is the hybrid neither has: real
`x_t`, both targets, mixed by noise level. No paper found does this.

## Why the teacher should win at high noise and lose at low noise

Two independent arguments land on the same schedule.

### 1. Text only matters at high noise

[eDiff-I](https://arxiv.org/abs/2211.01324) found that early in generation the model leans hard
on the prompt, and late in generation text conditioning is *almost entirely ignored*. That
finding is the reason they split one model into noise-level experts at all. Later
cross-attention analyses report the same shape: cross-attention dominates the first steps and
its influence falls away in the later, low-noise ones.

The refiner reaches the DiT only through cross-attention. So at low `t` there is little text
gradient to be had from either target, and what the teacher has to say is least relevant exactly
there.

### 2. Bias against variance

Decompose the ground-truth target: `v_gt = E[v | x_t, c] + residual`. In expectation, the
ground-truth loss and a *perfect* teacher's loss have the **same minimiser** — they differ only
in the variance of the gradient.

That residual grows with `t` **for data on a manifold**, which is the case that matters and not
a general fact. At `t → 1` the latent is pure noise, `x_0` is unrecoverable, and the caption is
the only handle on it, so one ground-truth draw is a very noisy estimate of the caption's mean.
At `t → 0` the latent is nearly a real image: `x_0` is readable off the manifold and the
off-manifold part divided by a small `t` recovers the noise, so both halves of `noise − latents`
are nearly determined. Note this does **not** hold for a Gaussian toy model, where the residual
is symmetric in `t` and peaks at `0.5` — a Gaussian has no manifold to read `x_0` off. It is an
argument about real latents, and it has not been measured on any.

The teacher is a pre-denoised target: low variance everywhere. It is also not perfect, and what
it gets wrong is **bias** — Qwen3-0.6B's conditioning, not Qwen3.5-2B's.

So λ trades bias against variance, and the variance it is buying down is largest at high `t`.
Both arguments, one about what the model attends to and one about the estimator, give the same
answer.

### Why not simply λ ∝ t

At `t → 1` every text frontend produces nearly the same velocity, because the prediction is
dominated by the noise rather than the conditioning. The teacher signal is real but barely
discriminative there. The informative band is the **upper middle**, not the top, so the shape
should be a smoothstep with a movable midpoint rather than a monotone power of `t`:

```
shape(t) = sigmoid((t − t_mid) / width)        t_mid = 0.5, width = 0.15
```

`t` at the loss is the timestep *after* `shift` has been applied (`prepare_inputs` warps `t` and
then builds `x_t` from the warped value), so it is the true noise level and `shape(t)` needs no
shift correction. It does interact with `timestep_sample_method = 'logit_normal'`, which already
concentrates samples near the middle: `shape` reweights that density, it does not replace it.

## What λ means, exactly

Worth stating precisely, because the loss *values* invite a wrong reading. Every number in this
section comes from `.audit/exp_teacher_lambda.py`, which reproduces them in a few seconds.

For squared error, mixing the two losses is **algebraically identical** to regressing onto a
mixed target:

```
(1−λ)‖v_s − v_gt‖² + λ‖v_s − v_T‖²      has the same gradient as
‖v_s − [(1−λ)·v_gt + λ·v_T]‖²           up to a constant
```

because the `v_s` terms collect. So λ is exactly the interpolation weight between the two
targets — nothing rescales it, and nothing needs normalising. Simulated over 800 residual draws,
the expected gradient of the mixed loss has cosine **0.99999–0.9993** against the mixed-target
reference, across `t` from 0.1 to 0.9.

This is worth pinning down because the two logged numbers look wildly unbalanced and suggest
otherwise. With the residual model above, `D(v_s, v_gt)` runs about **5× larger** than
`D(v_s, v_T)` at `t = 0.9`. That gap is almost entirely the irreducible residual, which is
**zero-mean**: it inflates the number printed in the log and contributes nothing to the expected
gradient. Do not normalise the terms to make them look comparable — that would break the
identity above and make λ mean something else.

What the gap *does* cost is gradient noise. Per-draw gradient deviation relative to the mean
gradient grows from about **0.1× at `t = 0.1` to 1.1× at `t = 0.9`** — at high noise a single
sample's gradient is as large as the signal it carries. That is the mechanism this whole feature
runs on, stated quantitatively: the teacher is a pre-averaged target, so it hands the refiner at
high `t` the signal that the ground truth only delivers buried in an equal amount of noise.

**The exception is Huber and smooth-L1.** Both clip large residuals, so the ground-truth term
saturates at high `t` while the teacher term does not, and the identity breaks. Solving for the
λ whose target-blend gradient best matches the real one: a nominal `0.5` behaves like **0.497 at
`t = 0.1` and 0.585 at `t = 0.9`** with `huber_delta = 1.0`. The drift favours the teacher —
clipping suppresses the noisy ground-truth term — so it is mild and in a safe direction, but
with `huber_delta` or `smooth_l1_beta` set, λ is approximate rather than exact.

## The teacher is a ceiling, so λ must also decay with training

[training.md](./training.md) already says it about the distillation stage:

> Distilling perfectly would reproduce the 0.6B model's information content and throw away
> everything the larger encoder knows. It is a warm start.

That applies with more force here, because this term can run for a whole training run rather
than a warm-up. A teacher term that never decays pins the student to Qwen3-0.6B's ceiling —
which destroys the entire reason for replacing the text encoder.

So λ carries a second schedule:

```
λ(t, step) = loss_weight · shape(t) · decay(step)
```

with `decay` falling to zero over the run. The run then *ends* as ordinary ground-truth
training, and the teacher is what got it there quickly. This is the difference between
"distillation" and "diffusion training that was warm-started by distillation", and only the
second one is wanted.

## It applies to whatever the learning rates say

This is not a mode. It is not tied to `refiner_only`. The same term is available to every
configuration, and **what it means changes with which learning rates are non-zero** — which is
also what decides whether a second DiT has to be resident.

| DiT learning rates | Teacher's DiT | What the term actually is | Extra VRAM |
|---|---|---|---|
| all zero (`refiner_only`) | *is* the student's DiT | pure text-frontend distillation: isolates refiner error from the DiT's own error | none |
| `cross_attn_lr` ≠ 0 | frozen copy | distillation, plus an anchor on stock Anima's cross-attention behaviour | ~3.5–4 GB bf16 |
| full fine tune, LoRA, LoKr | frozen copy | mostly an **anti-forgetting regulariser** pulling the model back toward stock Anima | ~3.5–4 GB bf16 |

The first row is the one worth understanding. When every DiT learning rate is zero and no
adapter targets the DiT, the student's DiT and the teacher's are the same weights, so there is
no second copy to hold — one extra `no_grad` forward through the module already resident. Only
the `llm_adapter` has to come from `teacher.transformer_path`, because a checkpoint that has
been through the refiner path no longer carries one.

Resolution is automatic, from the learning rates as `get_param_groups` resolves them, and it is
verified rather than assumed: sharing is taken only when every DiT group is frozen. Anything
else loads the independent copy.

The bottom two rows deserve a warning. Once the student's DiT moves, the teacher's velocity
stops being an achievable target and becomes a pull back toward where the model started. That is
genuinely useful — it is a principled anti-forgetting term, and forgetting is the documented
failure mode of opening the DiT too early. But with `decay = 'none'` it is also a hard ceiling
on a full fine tune, which can then never improve past stock Anima. **The longer the run and the
more of the DiT that trains, the more the decay schedule matters.**

## Where the teacher's text features come from

This follows `cache_text_embeddings`, the setting that already decides the same question for the
student. There is no separate switch.

### `cache_text_embeddings = true`

Cache the teacher's `LLMAdapter` output the way student embeddings are already cached: a new
prefix under the same `cache_dir`, with its own fingerprint covering the teacher's identity
(`teacher.llm_path`, `teacher.transformer_path`, `max_text_length`).

Per [lessons.md](./lessons.md), this must invalidate nothing that already exists — it is a new
cache, so latents and the student's embeddings are untouched, and a fingerprint argument is
appended only when it is non-empty.

The teacher's sequence is indexed by **T5** tokens, so the cached tensor is
`(max_text_length, 1024)` — about 1 MB per caption in bf16, against the 2 MB the student's
2048-wide embeddings already cost.

Once cached, **Qwen3-0.6B, the `LLMAdapter` and the T5 tokenizer are all droppable.** Only the
teacher's DiT stays, and only when the table above says a copy is needed.

### `cache_text_embeddings = false`

Both text frontends stay resident and run every step, on **the same caption string**.

This is a correctness requirement, not a convenience. On-the-fly mode exists precisely so that
tag shuffling and `tag_dropout_rate` are re-drawn on every access — so the augmented caption has
to be produced **once** and then handed to both tokenizers. Draw it twice and the teacher is
being asked about a different prompt than the student, and the loss is comparing two captions
rather than two text frontends. Nothing about the loss would look wrong; it would just quietly
optimise the wrong thing.

Resident cost in this mode: Qwen3-0.6B (~1.2 GB bf16) plus the `LLMAdapter`, on top of whatever
the DiT row of the table above requires.

## How it fits the pipeline

The plumbing already exists, which is the main reason this is worth building.

- `prepare_inputs` runs in the **main process**, not a dataloader worker
  (`utils/dataset.py:2229`), so a GPU teacher forward there is possible.
- The label is already a variable-length list of targets: `*target_list, mask = label`
  (`utils/dataset.py:2230`), and each element is broadcast from stage 0 to the last stage by
  `_broadcast_target`. `models/minimax_h3.py:420` already ships `(target, audio_target, mask)`.

So the change is the precedented shape, not a new mechanism:

```python
# prepare_inputs
return (noisy_latents, t, *conds), (target, teacher_v, t, mask)

# get_loss_fn
target, teacher_v, t, mask = label
lam = lambda_schedule(t, step)
loss = (1 - lam) * D(output, target) + lam * D(output, teacher_v)
```

`t` has to travel in the label because `loss_fn` receives only `(output, label)` and needs it to
evaluate λ. It is `(B, 1)`; the cost is nothing next to the latent-shaped `teacher_v`.

`D` is whatever loss the config already selects — `get_loss_fn` supports `huber_delta` and
`smooth_l1_beta` as well as MSE. Both terms use the same one; hardcoding MSE for the teacher
term would silently change what `huber_delta` means for a run.

### The mask applies to both terms

Non-negotiable, and there is history: the 2026-09-03 entry in the README records two loss
functions that computed terms without the mask, so masked-out samples reached the optimizer and
a fifth of a padded step's gradient came from padding. The teacher term must take the same mask
and the same batch-fill `G/G_real` weighting as the ground-truth term.

## Configuration

A new top-level `[teacher]` table, gated exactly the way `[rollout]` is: **`loss_weight = 0`
disables the feature entirely**, including loading anything.

```toml
[teacher]
loss_weight = 1.0

# Stock Anima -- the only checkpoint that carries an llm_adapter, so the only possible teacher.
transformer_path = '/data2/imagegen_models/comfyui-models/anima-preview.safetensors'
# The encoder the llm_adapter was trained against. NOT the student's encoder.
llm_path = '/data2/imagegen_models/Qwen3-0.6B-Base'

shape = 'sigmoid'   # 'sigmoid' | 'constant'
t_mid = 0.5
width = 0.15

decay = 'linear'    # 'linear' | 'cosine' | 'none'
decay_steps = 0     # 0 = the whole run
```

| Key | Default | Meaning |
|---|---|---|
| `loss_weight` | `0.0` | λ's ceiling. **0 disables the feature**, loads nothing, allocates nothing |
| `transformer_path` | — | Stock Anima. Required when enabled |
| `llm_path` | — | Teacher's text encoder. Required when enabled |
| `shape` | `'sigmoid'` | Timestep shape. `'constant'` removes the timestep dependence, for ablation |
| `t_mid`, `width` | `0.5`, `0.15` | Midpoint and softness of the smoothstep |
| `decay` | `'linear'` | How λ falls over the run. `'none'` keeps the teacher on forever — see the ceiling section |
| `decay_steps` | `0` | Steps to decay over; 0 means the run length |

**`teacher.llm_path` is the easy mistake**, and [training.md](./training.md) already flags the
same trap for distillation: the teacher has to be reproduced exactly as it was trained, or the
target the student chases is not one the DiT can read.

## What has to be refused at startup

- **`pipeline_stages > 1`.** The teacher forward happens where `prepare_inputs` runs, so the
  whole teacher DiT would sit on stage 0 while the student is split across stages — the memory
  imbalance defeats the point of splitting. `pipeline_stages = 1` with data parallelism across
  GPUs is the supported path, which is the same constraint `train.py:809` already imposes on
  refiner-only runs.
- **`blocks_to_swap` > 0 together with a separate teacher DiT.** Block swapping is moving the
  student's blocks between CPU and GPU already; a second full forward per step thrashes it.
- **A teacher checkpoint with no `llm_adapter.*` keys.** That is not a teacher, and the failure
  should name the file rather than surface as a missing-key error later.

## Eval should stay on the ground truth

Eval loss should be reported with **λ = 0**, always.

Otherwise the eval metric moves as `decay` runs, and a number that drifts because the *loss
definition* changed is indistinguishable from one that drifts because the model changed. Holding
eval on the pure ground-truth loss keeps it comparable across the decay schedule, across
`loss_weight` settings, and against runs with the feature off entirely.

Training logs the two terms separately, and λ alongside them — the same reasoning
[denoising-rollout.md](./denoising-rollout.md) gives for its own terms: there is no reason to
assume two different MSEs are comparable in magnitude, and a term that dominates should be
visible rather than buried in a sum.

## Limitations and open questions

- **Nothing here is measured.** No GPU run, no image, no ablation. Consistent with the rest of
  this branch, and stated rather than implied.
- **`huber_delta` / `smooth_l1_beta` distort λ; MSE does not.** See the section above. The
  distortion is modest at `delta = 1.0` (0.5 nominal reading as ~0.59 at `t = 0.9`) and it is in
  the teacher's favour, so it is a caveat rather than a blocker — but λ stops being exactly what
  it says, and a run that changes `huber_delta` also silently changes its mixing schedule.
- **`t_mid` and `width` are reasoned, not swept.** The arguments above fix the *direction* of
  the schedule confidently and its *shape* only loosely.
- **No CFG.** The teacher's velocity is the conditional prediction; sampling uses guidance. The
  rollout's `guidance_scale` handles the unconditional branch for its own objective and nothing
  equivalent is proposed here.
- **The anti-forgetting reading of the bottom two table rows is an argument, not a result.** It
  is a plausible and well-motivated use, and it is not what the term was designed for.
- **Interaction with OPLoRA is unexamined.** OPLoRA already exists to reduce catastrophic
  forgetting; whether a teacher term on top is complementary or redundant is untested.
