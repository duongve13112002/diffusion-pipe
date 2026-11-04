# Training options: step guard, weight decay target, LoRA+, cosine floor, distillation

Five opt-in settings ported from KohakuMossTrainer. Four work for every model; distillation is
MOSS-Image's. All are off unless the config turns them on, so a config that does not mention them trains
exactly as before. Where each came from, and what was decided along the way, is in
[the port plan](note/kohakumoss-port-plan-and-recommended-configs.md).

| Setting | Where | Models | Training types |
| --- | --- | --- | --- |
| [`skip_nonfinite_steps`](#skipping-steps-with-a-non-finite-gradient) | top level | all | full fine-tune, LoRA, OPLoRA, LoKr, DoRA |
| [`weight_decay_target`, `weight_decay_to_one`](#weight-decay-toward-the-starting-weights) | top level | all | full fine-tune |
| [`lora_plus_ratio`](#lora) | `[adapter]` | all | LoRA, OPLoRA, DoRA (not LoKr) |
| [`lr_scheduler_min_ratio`](#cosine-and-linear-floor) | top level | all | all |
| [`[teacher]`](#distillation-toward-the-starting-model-moss-image) | top level table | MOSS-Image | LoRA, LoKr, OPLoRA, full fine-tune |

Each setting is written to run the same on one GPU, several GPUs and several nodes, and the CPU tests
cover the single-process and the data-parallel cases with a real DeepSpeed engine. None of it has been run
on a GPU, with a real MOSS-Image model, with pipeline stages or across nodes yet; the notes under each
setting say what it needs from a multi-process run, and `tools/test_nonfinite_guard_gpu.py` and
`tools/test_distillation_gpu.py` are the checks to run on a GPU machine.

## Skipping steps with a non-finite gradient

```toml
skip_nonfinite_steps = true   # default false
nonfinite_patience = 8        # stop after this many skipped steps in a row; 0 never stops
```

One batch whose gradient holds a NaN or inf, applied as it is, puts NaN into the weights (a full
fine-tune) or the adapter and the optimizer's moments (a LoRA), and the run is over with no sign until the
loss turns NaN. With the guard on, that step is dropped instead: the weights, the optimizer state and the
learning-rate schedule stay as they were, the step is counted (`train/nonfinite_skipped`) and the run goes on
with the next batch.

- The check is on the gradient norm the engine step already computes. The norm is summed over the pipeline
  stages and averaged over the data-parallel ranks, so every rank holds the same number and takes the same
  branch: nothing extra to agree on, on one node or many. With `gradient_clipping > 0` the norm is the one
  clipping uses (computed once); with clipping off it is computed once for the guard.
- A skipped step uses DeepSpeed's own protocol for an overflowed step, so the schedule does not advance and
  the engine counts it in `skipped_steps`. EMA, outlier decay and the weight-decay target do not run on it, and
  its loss is left out of the epoch average.
- After `nonfinite_patience` skipped steps in a row the run stops with an error naming `--nan_check`:
  skipping cannot cure something that produces NaN every step.
- It does not catch a non-finite loss with a finite gradient, or weights that were already non-finite.
- Refused with `gradient_release` (the gradients are freed during backward; GenericOptim has its own
  per-parameter `skip_invalid_grads`).

## Weight decay toward the starting weights

```toml
weight_decay_target = 'start'                      # default 'zero'
weight_decay_to_one = '^layers\.41\.(q_norm|k_norm)\.weight$'   # default '' (none)
```

AdamW's decay pulls every weight toward 0 a little each step: `p -= lr * wd * p`. For a pretrained model, 0
is not where the weights belong; decay steadily erodes what was learned. With `weight_decay_target = 'start'`
the pull is toward the weights the run started from instead:

    p -= lr * wd * (p - a)        a = the weights at launch

so weight decay still limits how far training can drift, but now it limits the drift away from the pretrained
model rather than shrinking it. `weight_decay_to_one` names parameters (a regular expression matched against
the parameter name, as saved) that decay toward 1 instead of the start: the norm gains of the last block, in
KohakuMoss's MOSS recipe.

- Full fine-tune only; refused next to an `[adapter]`, `gradient_release` or `blocks_to_swap`. Needs `adamw`, `adamw_optimi`,
  `adamw8bit` or `genericoptim` (whose decay is exactly `lr * wd * p`, Muon groups included) and
  `weight_decay > 0`. `stableadamw` (its decay is divided by the update's RMS) and `decouple_lr = true` are
  refused for that reason.
- The optimizer's own decay is switched off for the tensors this takes over, so nothing decays twice. A checkpoint
  saved by such a run holds those groups with a decay of 0 and remembers the decay they had; resuming it
  with the feature turned off puts that decay back. Which
  tensors: the ones `train.py` already gives weight decay (two or more dimensions), plus the ones
  `weight_decay_to_one` matches (which must be exempt from the optimizer's decay, i.e. 1D).
- The anchors `a` are a copy of the launch weights, in their own dtype, on each GPU: 2 bytes per bf16
  parameter. They are taken before a resume loads a checkpoint, so they are the weights of the model file
  the config names.
- The update is far below one bf16 step, so bf16 weights are updated in fp32 and rounded back at random
  (unbiased), from a generator seeded by the step alone: data-parallel replicas and nodes draw the same
  numbers and stay identical. No collective runs during training; the only one is a single `all_reduce`
  at startup that counts the tensors taken over on every stage.
- The run stops at startup when nothing at all is taken over, and when `weight_decay_to_one` is set but
  matches no trained parameter on any stage (a mistyped regular expression). An `lr * wd` above 1
  (the pull would pass its target) is refused at startup, on every rank at once, from the highest learning
  rate the schedule reaches.

## LoRA+

```toml
[adapter]
type = 'lora'
rank = 32
lora_plus_ratio = 16     # default 1.0 (plain LoRA); the paper uses 2 to 16
```

In `W + B @ A`, A starts random and B at zero. Hayou et al. (2024) show that one learning rate for both leaves
B under-trained on wide layers and that B should learn faster by a fixed ratio. Every `lora_B` tensor gets
`ratio` times the learning rate its group has; everything else is unchanged. The schedule, warmup, weight
decay split and OPLoRA projection act per group as before, so B follows the same curve at `ratio` times the
height with a constant, linear or `lr_scheduler_min_ratio` schedule. A plain `cosine` ends at an absolute
`1e-6` for every group, so A and B both end there and the ratio fades towards the end; set
`lr_scheduler_min_ratio` to keep it all the way. The groups it adds are named `<name>+B` when the model
names its groups.

Refused: LoKr (no A/B pair), Prodigy, Automagic and GenericOptim `automagic` (they adapt their own rates and would ignore the ratio), an `[optimizer]` without `lr`, `gradient_release`, `force_constant_lr` (it sets every group to one rate and
would undo the ratio). A resumed run keeps the learning rates its optimizer saved, and turning LoRA+ on or off
for an existing checkpoint changes the number of optimizer groups, which PyTorch refuses: use
`--reset_optimizer`. `train/lr` logs the first group (the A side).

## Cosine and linear floor

```toml
lr_scheduler = 'cosine'          # or 'linear'
lr_scheduler_min_ratio = 0.1     # end at 10% of each group's own lr
```

Without it the cosine ends at an absolute `1e-6` and the linear schedule at 0. With a ratio in `[0, 1)` both
end at that fraction of each parameter group's own learning rate, so groups with different rates keep their
ratios (KohakuMoss: `SCHEDULE min_value = 0.1`). With a warmup, the decay spans the steps after it, so it reaches
the floor on the last step of the run (without this key, a warmup shifts the cosine and the run ends before
the schedule does, as it always has). Only `cosine` and `linear` take it; any other scheduler
refuses the key.

## Distillation toward the starting model (MOSS-Image)

```toml
[teacher]
loss_weight = 0.5      # 0 or absent: off
shape = 'constant'     # 'sigmoid' weights high noise more (t_mid, width)
decay = 'none'         # 'linear' or 'cosine' fall to 0 over decay_steps
decay_steps = 0
```

    loss = L(v, target) + lambda(sigma, step) * L(v, v_teacher)

`v` is the model's velocity, `v_teacher` the velocity the *starting* model predicts for the same noisy
latent, noise level and text embedding, and `L` the loss the run already uses (MSE, or `huber_delta` /
`smooth_l1_beta`). `lambda = loss_weight * shape(sigma) * decay(step)`, the schedule helpers of the anima
refiner's teacher guidance. The second term keeps what the model predicts close to what it predicted
before fine-tuning, which limits how far a small dataset can drag it. The step log adds
`train/distill_loss` and `train/distill_lambda`; `train/loss` is their sum with `train/flow_loss`.

- The teacher is always the model the run starts from. LoRA/LoKr/OPLoRA: the same blocks with the adapter
  switched off, nothing to copy, one extra forward per micro-batch; needs `pipeline_stages = 1`. Full
  fine-tune: a frozen copy of the model as loaded, on the GPU of every rank of the first pipeline stage (the
  size of the model: about 10 GB in bf16, worked out from the parameter count), one extra forward per
  micro-batch on that stage, which should make the first stage the slower one when there are several (an
  expectation, not yet measured on a GPU). A resumed full fine-tune distills toward the model
  file's weights, not toward the checkpoint it resumed.
- The teacher forward runs in `TextEncoderLayer`, which is never activation-checkpointed, so it is not
  recomputed in backward; it sees the same embeddings the student does (cached or encoded on the fly) and
  its velocity travels with the activations to the loss. Training batches only.
- Refused: `compile = true`, `blocks_to_swap`, an `[adapter]` with `pipeline_stages > 1`, an `[adapter]` with
  `init_from_existing` (the teacher would be the base model, not the model the run starts from), unknown keys.
  `decay = 'none'` is warned about: the term never fades, so the model is held near its start for the whole
  run, which is the point of a regulariser but caps how far the fine-tune can move.
- Not part of KohakuMoss's default recipes, and no measured image-quality benefit was found; judge it from
  your own samples.
