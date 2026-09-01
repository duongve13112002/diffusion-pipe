# The four distillation objectives

Distillation trains the `ContextRefiner` to emit what Anima's `LLMAdapter` already emits. There
are four ways it measures whether that is happening, and **they are summed, not chosen between**.
Turning one on does not turn another off.

This page is the honest comparison: what each one can see, what it cannot, and what it costs.

## The shape of the problem

The teacher's sequence is indexed by **T5** tokens, because the `LLMAdapter` embeds T5 token ids
as its query sequence. The student's is indexed by **Qwen BPE** tokens. Position `i` means
different things on each side, so a position-wise comparison is meaningless — it runs without
error and plateaus forever.

Every objective below is a way around that. The first three compare quantities that do not
depend on how the text is indexed; the fourth sidesteps the question entirely by comparing what
the model *does* with the features rather than the features themselves.

## 1. Cross-attention probe — the base

**Config:** none. Always on. `[probe] num_queries` and `[probe] num_blocks` shape it.

Push a fixed set of random query vectors through the DiT's own frozen cross-attention with the
teacher's features, then with the student's, and compare the outputs.

```
teacher_feats ──┐
                ├──> frozen cross_attn(probe_queries, context=·) ──> compare
student_feats ──┘
```

The queries are **random** because the cross-attention weights are frozen and shared by both
paths, so the comparison is fair whatever they are; and **fixed** because a resampled probe would
make the objective non-stationary — the loss would move when the probe moved, not only when the
student improved.

**What it does well.** It is the cheapest objective, it gives dense signal (every text position
contributes), and it is the only one of the four that would work at all with nothing else. It is
also the only one whose validity depends on the tokenization mismatch being sidestepped, which is
the whole reason this design exists.

**What it cannot see.** Cross-attention output is a weighted sum over text positions. A sum does
not care which token holds which content.

> **Example.** Teacher, for `"a black cat"`, puts *black* in one position and *cat* in another.
> A student that puts "the average of black and cat" in *both* positions produces nearly the
> same sum — and satisfies this objective. Worse, a student that produces the same average for
> `"a black cat"` and `"a white dog"` is barely penalised, because the loss never compares two
> captions to each other.

**Measured.** The generalization gap is a clean function of `num_queries / head_dim`: at 0.25 the
held-out probe error is 2.8× the training-probe error, at 0.5 it is 1.63×, at 2.0 it is 1.16×.
`num_blocks` recovers most of it, because each block projects the queries differently — 8 blocks
bring the 0.5 case down to 1.14×. Hence the default of `2 × head_dim` queries, and **never fewer
than 4 blocks**.

## 2. Pooled mean

**Config:** `pooled_loss_weight`, default `0.1`.

Compare the length-normalised mean of the two feature sets directly. It normalises by the padded
length rather than the mask length, because a mask-normalised mean would disagree with the
attention term about what a "position" is.

**What it does well.** Essentially free, and it stabilises the early steps when the refiner is
still random and the probe term is noisy.

**What it cannot see.** One vector for a whole caption. It is the weakest of the four by a
distance.

**Measured, and this is the fair part.** It helps only when the probe is starved: held-out error
0.597 with it versus 0.609 without, at `num_queries / head_dim = 0.25`. At an adequate probe
count it is a wash — 0.2855 either way. It does reduce the pooled error it targets (0.071 vs
0.077), which is what it is for.

**Verdict.** Nearly harmless, nearly useless at the shipped settings. It is kept because it costs
nothing and because it is genuinely useful in the probe-starved regime the defaults no longer sit
in.

## 3. Relational (RKD distance-wise)

**Config:** `relational_loss_weight`, default `1.0`.

Match the teacher's **pairwise distance structure** within the batch. If two captions are far
apart for the teacher, they must be far apart for the student.

```
teacher:  d(A,B) = 3.0   d(A,C) = 0.5   d(B,C) = 3.2
student:  d(A,B) = 2.9   d(A,C) = 0.6   d(B,C) = 3.1     ← satisfied
student:  d(A,B) = 0.1   d(A,C) = 0.1   d(B,C) = 0.1     ← penalised heavily
```

**What it does well.** It prices exactly what objective 1 cannot see, and it costs nothing —
both feature sets already exist for the probe. This is the direct remedy for the mode collapse
that [Scaling Down Text Encoders (CVPR 2025)](https://arxiv.org/html/2503.19897v1) reports for
naive text-encoder distillation, where "rat", "cat" and "man" come back as identical embeddings.

> **Example.** `"mèo"`, `"chó"`, `"xe"`. The teacher puts cat and dog near each other and car far
> away. A student that collapses all three to one point sails past objective 1 and is penalised
> immediately here.

**One deliberate deviation from the textbook.** RKD normalises each side by its own mean distance,
which makes the loss scale-invariant — and uniform shrinkage toward a centroid *is* a scale
change, so that formulation scores **0.0000 at every collapse fraction from 25% to 90%**. Both
sides are divided by the *teacher's* mean here instead: 0.0277 at 25% collapsed, 0.1106 at 50%,
0.4396 at 100%.

**What it cannot see.** Only *relations*, never *content*. A student that preserves every pairwise
distance while placing the whole constellation in the wrong region satisfies it completely. It
also only sees within one batch, so a small batch gives it few pairs to work with.

**Diagnostic.** The progress bar's `spread` figure is the mean pairwise cosine distance between
the batch's student features, shown against the teacher's. It should track the teacher's number;
falling toward zero is collapse in progress.

## 4. Denoising rollout

**Config:** `[rollout] loss_weight`, default `0.0` — **off**. See
[denoising-rollout.md](./denoising-rollout.md).

Walk the frozen DiT from pure noise toward clean, driven by the **teacher's** prediction, and
compare what the DiT predicts from the teacher's features against what it predicts from the
student's, at points along that path.

**What it does well.** It measures agreement at the quantity the DiT actually produces, on inputs
the DiT itself produced. It is the only objective of the four that constrains *content* rather
than a summary of it, and it is the published remedy the other three are approximations of.

**What it costs.** The whole DiT stays resident, roughly 3.5–4 GB in bf16 for Anima, taking this
stage from about 6 GB to 10–12 GB — plus one full backward graph per `loss_points`, doubled again
with guidance. It is off by default because that ~6 GB figure is what makes distillation runnable
on modest hardware.

**What it cannot fix.** There is an **irreducible floor of roughly 0.3 relative RMS**, caused
purely by the teacher and student having different token counts: a student sequence cannot
reproduce a teacher sequence's output for every possible query. **A plateau in the distillation
loss is expected, not a failure.**

## Side by side

| | Cost | Sees collapse? | Constrains content? | Default |
| --- | --- | --- | --- | --- |
| 1. Probe | Low | ✗ | Permutation-invariant moments only | always on |
| 2. Pooled | ~0 | ✗ | Very weakly | `0.1` |
| 3. Relational | ~0 | ✓ | ✗ (relations only) | `1.0` |
| 4. Rollout | **High** | ✓ | ✓ | `0.0` (off) |

**How to choose.** Start with 1–3, which is what the shipped configs do. Watch `spread` in the
progress bar. Turn on 4 when `spread` shows the student's captions collapsing toward each other
despite objective 3 — that is the signal it is for, and the only reason to pay its price.

Each weighted term is logged separately, so a term that dominates the sum is visible rather than
buried in it. That matters here: the probe is an MSE over cross-attention outputs and the rollout
is an MSE over velocities, and there is no reason to assume the two are comparable in magnitude.

## Configuration that applies to every objective

None of these is specific to one objective. Changing them changes all four.

| Key | Meaning |
| --- | --- |
| `epochs` / `steps` | How long to train. Alternatives; setting both is refused |
| `batch_size`, `gradient_accumulation_steps` | Per rank. The relational term sees pairs only within one micro batch |
| `[optimizer]` or flat `lr` / `betas` / `weight_decay` | Alternatives, never both |
| `precision` | The trainable refiner: `fp32`, `bf16-mixed`, `fp16-mixed`, `bf16-full` |
| `dtype` | The **frozen** modules, independent of `precision` |
| `max_grad_norm` | Gradient clipping |
| `warmup_steps`, `lr_scheduler` | Shares `utils/lr_schedule.py` with `train.py` |
| `distributed_strategy` | `ddp`, `zero1`, `zero2`. ZeRO needs 2+ ranks |
| `save_every_n_epochs` / `save_every`, `keep_last_n_checkpoints`, `save_full_model` | Checkpointing |
| `log_every` | Progress-bar interval |
| `seed` | Model seed; the caption and rollout streams are rank-offset from it |
| `resume_from` | Restores weights, optimizer, scheduler, step and the augmentation RNG |
| `max_text_length` | Padded token length. **Should stay 512** — see design-notes.md |
| `dataset` / `caption_corpus` / `captions` | Caption source. Alternatives |

## A note on the two `steps`

`[distill] steps` is the number of optimizer updates for the whole run. `[rollout] steps` is the
number of points on the denoising trajectory, per micro batch. Different tables, different
meanings, no interaction — but the names are easy to conflate when skimming.
