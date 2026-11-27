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
| [`[teacher]`](#distillation-toward-a-teacher-every-other-model) | top level table | every other model | LoRA, LoKr, DoRA, OPLoRA, full fine-tune |

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

- The teacher is the model the run starts from (or another transformer, see `path`). With one pipeline stage:
  LoRA/LoKr/OPLoRA use the same blocks with the adapter switched off, nothing to copy, one extra forward per
  micro-batch; a full fine-tune keeps a frozen copy of the model as loaded on the GPU. With `pipeline_stages > 1`
  (adapter or full fine-tune) the teacher is a frozen copy split over the stages as a second, forward-only
  pipeline, exactly as for the other models below: every GPU holds 1/stages of it, and the velocity reaches the
  loss through the label instead of travelling with the activations. A resumed full fine-tune distills toward the
  model file's weights, not toward the checkpoint it resumed.
- With one stage the teacher forward runs in `TextEncoderLayer`, which is never activation-checkpointed, so it is
  not recomputed in backward; it sees the same embeddings the student does (cached or encoded on the fly) and
  its velocity travels with the activations to the loss. Training batches only.
- Refused: `compile = true`, `blocks_to_swap`, an `[adapter]` with
  `init_from_existing` (the teacher would be the base model, not the model the run starts from), unknown keys.
  `decay = 'none'` is warned about: the term never fades, so the model is held near its start for the whole
  run, which is the point of a regulariser but caps how far the fine-tune can move.
- Not part of KohakuMoss's default recipes, and no measured image-quality benefit was found; judge it from
  your own samples.

MOSS-Image also takes `path` (see below): the teacher can be another transformer instead of the starting model. With a
`path` the teacher is always a full copy of that transformer (also under an `[adapter]`), and any number of
pipeline stages works; the `init_from_existing` and adapter-with-stages refusals above apply only without it.

## Distillation toward a teacher (every other model)

```toml
[teacher]
loss_weight = 0.5      # 0 or absent: off
decay = 'cosine'       # 'none' (default), 'linear' or 'cosine'
decay_steps = 20000    # optimizer steps over which lambda falls to 0
path = '/models/other_checkpoint'   # optional: another checkpoint of the same model
```

    loss = L(student, target) + lambda(step) * L(student, teacher)

`L` is the model's own loss function, so its masking, `huber_delta` / `smooth_l1_beta` and weights apply to the
second term as they do to the first, and `lambda = loss_weight * decay(step)`. This is for every model but
MOSS-Image (which has its own table above) and works with LoRA, LoKr, DoRA, OPLoRA and full fine-tuning,
with one GPU, several, several nodes and any `pipeline_stages`.

- Without `path` the teacher is the model the run starts from, as loaded (before an adapter, `[fp8_training]`
  or any training touches it). With `path` it is another checkpoint of the *same architecture*, read the way
  `[model]` reads its own weights: the same key (`transformer_path`, `diffusion_model`, `diffusers_path` for
  SD3 / OmniGen2 / HiDream, `single_file_path` for LTX-Video, `checkpoint_path` for SDXL, ...), format and dtype
  handling. Adapters the student's base is merged with (`merge_adapters`, `fuse_adapters`) are not applied to a
  `path` teacher: it is the finished model (nor the student's `context_refiner_path` / `llm_adapter_path` of
  an anima, nor the `pdd_heads` / `diffusion_model_format` of a MiniMax-H3 transformer). The text encoders, the VAE and the latent space are the student's
  (SDXL aside: its text encoders train with the UNet, so its teacher is a whole second pipeline).
- The teacher is a second copy of the model's own layers (no gradients, eval mode). It runs on the same inputs
  as the student (the noisy latents, timesteps and text embeddings the batch carries; `cache_text_embeddings`
  true or false makes no difference: with false the loader encodes the captions before the batch is prepared,
  once, and the student and the teacher both read the result), and its output joins the label of the last stage
  with the targets, so no layer needs to pass anything on.
  - `pipeline_stages = 1`: the teacher sits on the same GPU as the student, one extra forward per micro batch.
  - `pipeline_stages > 1`: the teacher is a second, forward-only DeepSpeed pipeline with the student's number of
    stages (no optimizer, no gradients). Every GPU holds 1/stages of the teacher, not the whole of it on the first
    stage. Each step, before the student's `train_batch`, every rank runs the teacher over the step's micro
    batches; the last stage then has the teacher's output of each micro batch. The teacher forward is therefore
    sequential with the student's step (it fills the same GPUs one after the other) rather than overlapped.
    Host memory while it is built: every rank reads and copies the whole teacher on the CPU, so with the default
    `parameters` or `uniform` partition they do it one at a time on a node and each frees the stages it does not
    run at once (peak: the node's ranks x the student, plus one teacher). With a `manual` `partition_split` the
    split is only known once the student's layers exist, so all ranks build it together and free it afterwards.
    The default `partition_method = 'parameters'` weighs trainable parameters, which a frozen teacher has none of,
    so the teacher is balanced by its parameter counts instead (`uniform` and a `manual` `partition_split` are the
    student's own; `memory` balances it by the bytes of its layers). Each rank builds the teacher's layers on the CPU first, then frees the other stages' weights:
    the CPU RAM peaks by about the size of the weights in `dtype` on every rank while loading. Neither the speed
    nor the VRAM has been measured on a GPU.
  Once `decay` has brought lambda to 0 (step `decay_steps`) the teacher no longer runs: the loss ignores it from
  there on, and the label carries zeros; the teacher's weights are freed from the GPU at that point, on every path (the generic
  teacher, MOSS-Image's and anima_refiner's stock Anima, with one stage or several); a layer or module it shares
  with the student (an adapter run's blocks, a DiT the stock teacher shares) is kept, and the forward does not run.
  A run resumed from a checkpoint whose next step is already at or past `decay_steps` does not load the teacher at
  all (it is switched off before anything is built). The teacher always runs in bf16, also next to the student's
  `[fp8_training]`. MOSS-Image with `cache_text_embeddings = false` encodes each caption once per
  step: the teacher's text layer leaves the embedding on stage 0 and the student's takes it. Under `[fp8_training]` the teacher runs in bf16, and MOSS-Image's full
  fine-tune teacher (without `path`) is copied from the model before fp8 touches it.
- Evaluation batches (`[eval_datasets]`) report the plain loss, without the distillation term. The step log has
  the combined `train/loss`.
- `[teacher] path` and an anima_refiner: a table that names `transformer_path` or `llm_path` is the older
  form, a stock Anima mixed into the target (`docs/anima_refiner/teacher-guided-training.md`). One that names
  neither is this teacher, toward the refiner's own starting weights; a freshly initialised refiner cannot be
  its own teacher (random weights), and `path` cannot be combined with the Anima keys.
- Refused, naming the key: `shape`, `t_mid` and `width` (a loss weight that follows the noise level needs the
  noise level of each sample, which only MOSS-Image hands to its loss), unknown keys, `blocks_to_swap`, an
  `[adapter]` with `init_from_existing` unless `path` is set, `decay` other than `none` without `decay_steps`, `decay_steps` with `decay = 'none'`,
  and Iris with `repa_weight > 0`. `decay = 'none'` is warned about: the term then never fades.
- Wan with `cache_text_embeddings = false`: the umt5 encoder now always encodes in eval mode. DeepSpeed's `train()`
  had left its dropout on, so the student trained on dropout-noised embeddings (and a teacher sharing the encoder
  would have drawn other masks). This changes Wan runs with on-the-fly text whether or not `[teacher]` is on.
- Not run on a GPU or with real weights yet. Not measured: the speed cost of the extra forward, the memory, whether
  distillation helps image quality on your data. Judge it from your samples.

## Writing a model's layers for `pipeline_stages > 1`

DeepSpeed's pipeline engine runs backward on every floating tensor a stage sends on, and the last stage asserts that
every floating tensor it received got a gradient. Two rules follow for the tuples that `to_layers()` passes
between layers (a stage can start at any layer):

- A floating tensor a layer builds (a rope table, a float mask, zeros) and returns without a graph must be marked
  `requires_grad_(True)` where it is made, or travel as a bool or integer tensor and be converted inside the block.
  Complex tensors (some rope tables) are not floating for this purpose and need nothing.
- A floating tensor that a stage receives but none of its layers reads through a differentiable op (an entry a
  layer drops from the tuple, audio entries of a video model with audio off) needs nothing from the model:
  `ManualPipelineModule.forward` (utils/pipeline.py) ties every received floating tensor that requires grad to the
  stage's output with `models.base.tie_unused`, a term that is an exact 0 (no value changes, a NaN in the tied tensor
  stays out) and gives the tensor a zero gradient. It adds no communication, so it holds for any number of stages on
  one node or several.

With a reentrant checkpoint (`reentrant_activation_checkpointing = true`, or `activation_checkpointing = 'unsloth'`),
`ManualPipelineModule.forward` also marks the first stage's floating inputs as requiring grad: such a checkpoint gives
its outputs a graph only when an input requires grad, and DeepSpeed's own marking does not run because this repo hands
the checkpointing options to the module instead of the DeepSpeed config.

`test/test_pipeline_stage_grads.py` pins the first rule for the models that needed it and the tie itself;
`test/test_pipeline_engine.py` runs the second in real DeepSpeed process groups (a dropped tensor, a tensor the last stage
ignores).
