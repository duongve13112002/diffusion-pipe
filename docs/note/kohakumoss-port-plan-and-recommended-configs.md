# Plan: non-finite step guard, weight decay toward the start weights, LoRA+, distillation, and recommended configs

Date: 2026-11-02

Status: the plan as it was approved (2026-11-02), then built. Section 7 lists what was built and every place the
code differs from this plan; where a sentence below is overtaken, section 7 is right.

Question: after [kohakumoss-features-missing-in-diffusion-pipe.md](kohakumoss-features-missing-in-diffusion-pipe.md),
which of the missing KohakuMossTrainer features to build, how, and which LoRA and full fine-tune
settings KohakuMossTrainer recommends, mapped onto diffusion-pipe keys.

## Decisions taken so far

| Item | Decision | Scope asked for |
| --- | --- | --- |
| Non-finite step guard | Build | Every model, full fine-tune, LoRA (including OPLoRA), LoKr and any other adapter |
| Weight decay toward the start weights, LoRA+ | Build | Not MOSS only; see "Scope" below |
| Distillation toward a frozen teacher | Build, for LoRA and for full fine-tune | See "Scope" below |
| Samples while training, packed token batching, extra data sources | Dropped | |
| Checkpoint every N minutes | Already exists (`checkpoint_every_n_minutes`) | |
| Per-rank RNG in checkpoints | Dropped: the training loop is never seeded, so there is nothing to reproduce | |
| Partial activation checkpointing, LoRA on the text refiner and modulation | Dropped (code complexity against an unproven gain) | |
| p_full, sigma weighting, l1 loss | Dropped: not in KohakuMoss's recipes, no evidence for a fine-tune (see "Evidence for the knobs") | |

Scope. The guard lives in the DeepSpeed engine patches, so it is model independent by
construction. Weight decay toward the start weights is optimizer-side and model independent. LoRA+ is
applied where `configure_adapter` builds the PEFT adapter (`models/base.py`), which every model uses.
Distillation, `p_full`, sigma weighting and l1 each touch a model's `prepare_inputs` and loss, which
differ per model (the MOSS-Image one returns `(target, mask)` as its label and encodes text inside a
pipeline layer). The plan builds them for MOSS-Image first behind a shared helper and lists what
another model needs; it does not claim they are free for every model.

## Evidence for the knobs

What the KohakuMossTrainer recipes themselves use (clone at commit b06ac60):

| Knob | KohakuMoss default | In its recommended LoRA / full configs? |
| --- | --- | --- |
| Non-finite handling | Always on: `NONFINITE_PATIENCE = 8` (`scripts/train/moss.py:233`), stop after 8 non-finite steps at the loss-scale floor | Yes, it is part of the base setup. Here it is `skip_nonfinite_steps`, off by default at your request |
| `WD_TARGET` | `"zero"` | No. `algo_full.py` sets `WD_RULES = "moss"` only. The docs describe `"start"` as keeping the pretrained weights from shrinking, and give no measured result |
| `LORAPLUS_RATIO` | 1.0 (off) | No |
| `WEIGHTING` | `"none"` ("the MOSS recipe") | No |
| `LOSS` | `"mse"` | No |
| `p_full` | 0.0 in the final-stage schedule; 0.01 in the two earliest stages of the tech-report schedule | No. `base_lora.py` sets `TIMESTEP_KWARGS` without `p_full` |
| `TEACHER`, `DISTILL_WEIGHT` | off | No |
| `STREAM_LOSS`, `OUTLIER_DECAY` | off | No (they are documented for a heal run, `configs/heal`) |

So none of these is part of KohakuMoss's default recipe. They are options. `p_full` only matters when
reproducing the early pretraining stages, which a fine-tune does not do; the plan below leaves
`p_full`, sigma weighting and l1 out unless you ask for them, and builds the others as opt-in
keys that default to the current behaviour.

## 1. Non-finite step guard

Goal: a step whose gradient contains NaN or inf is skipped (weights, optimizer state and the
learning-rate schedule untouched), counted, logged, and the run stops after N in a row.

### Hook point (verified in DeepSpeed 0.18.4, the pinned version)

- `train.py` runs DeepSpeed's fp32 path: it does not enable DeepSpeed fp16 or bf16 (comment at
  `train.py:1007`). In `DeepSpeedEngine._take_model_step` (`engine.py:2569`) that path calls
  `clip_fp32_gradients()` when `gradient_clipping > 0`, then `self.optimizer.step()`, then zeroes the
  gradients with `self.zero_grad()` (sets `param.grad = None`).
- DeepSpeed already has a skipped-step protocol: after `step()` it reads
  `getattr(self.optimizer, 'overflow', False)`. When true, `_step_applied` is False,
  `skipped_steps` is incremented and `lr_scheduler.step()` is not called (`engine.py:2614-2623`).
  `global_steps` still advances. `engine.was_step_applied()` (`engine.py:2020`) is the public reader.
- `utils/patches.py` already replaces `_take_model_step` (`_take_model_step_measured`) and
  `clip_fp32_gradients` for every model, and already has `grad_norm()`, which sums over the pipeline
  and averages over data-parallel ranks, so every rank gets the same number.

### Design

1. In `_take_model_step_measured`, when the guard is on, compute the global gradient norm with
   `grad_norm` before calling the original (one collective). Keep it in `self.last_grad_norm` and let
   `_clip_fp32_gradients` reuse it instead of computing it a second time.
2. If the norm is not finite: replace `self.optimizer.step` with a no-op for this call and set
   `self.optimizer.overflow = True`, call the original (so DeepSpeed zeroes the gradients, counts the
   skip and leaves the scheduler alone), then restore both. If finite, nothing changes. Every rank
   takes the same branch because the norm is identical on every rank.
3. `train.py`, after `train_batch`: read `model_engine.was_step_applied()`. When false: skip the
   post-step work that assumes an update happened (`oplora_projector.project()`, `outlier_decay.apply`,
   the new weight decay toward the start weights, `ema.update`), add 1 to a consecutive counter, log a
   line with the step, the loss and the count, and log `train/nonfinite_skipped` (total). When true,
   reset the counter.
4. A consecutive-skips limit (`nonfinite_patience`) raises on every rank at once, since all ranks
   agree. A skipped step still counts as a step for `max_steps`, epochs and the saver, so a run that
   skips cannot loop forever without the patience stop.
5. Only the gradient is checked. A non-finite loss with a finite gradient, or non-finite weights that
   were already there, are not caught; `--nan_check` still finds the layer and images (the guard cannot
   name the batch, because the pipeline has consumed it when the step runs).

Why not the existing `GenericOptim(skip_invalid_grads=True)`: it skips per parameter, only inside that
one optimizer, and leaves the schedule, EMA and the other optimizers untouched.

### Config keys (top level)

| Key | Default | Meaning |
| --- | --- | --- |
| `skip_nonfinite_steps` | `false` (you chose off) | Turn the guard on or off |
| `nonfinite_patience` | `8` | Stop after this many skipped steps in a row; `0` never stops |

Refused with `gradient_release` (gradients are freed during backward, so there is none to check at
step time; the error says so and points to `skip_invalid_grads`).

### Tests

- CPU: a fake engine and optimizer (no DeepSpeed, as `test/conftest.py` already stubs it): a finite norm
  steps normally; a NaN or inf norm leaves `step` uncalled, sets and clears `overflow`, restores the
  original `step`; the norm is computed once; consecutive counter, reset and patience stop; the post-step
  hooks are skipped on a skipped step.
- GPU (`tools/test_nonfinite_guard_gpu.py`, one or many GPUs, no code edits): run `train.py` for a few
  steps through a small launcher that poisons one gradient with a parameter hook on a chosen step, and
  check that the weights and optimizer state are bit-identical across that step, the schedule did not
  advance, the next step is normal, and `nonfinite_patience` stops a run poisoned on every step. Run it
  for a LoRA, a LoKr, an OPLoRA and a full fine-tune config.

### Risk

`was_step_applied()` and `optimizer.overflow` are DeepSpeed internals that the patch relies on; a
DeepSpeed upgrade needs the check that `docs/note/upstream-api-drift-audit.md` describes.

## 2. Weight decay toward the start weights (`weight_decay_target = 'start'`)

KohakuMoss: `p -= lr * wd * (p - a)` with `a` a bf16 copy of the launch weights, 2 bytes per decayed
parameter, full fine-tune only, gated by the step's non-finite flag (`training.md`, "Weight decay target").

Design for diffusion-pipe: reuse `utils/outlier_decay.py`'s machinery (the learning rate read before
`train_batch`, the seeded bf16 stochastic rounding, the post-step application), and set the optimizer's
own `weight_decay` to 0 for the parameters it takes over, so decay is not applied twice. A step update of
`lr * wd * (p - a)` is far below one bf16 step, which is why the stochastic rounding is needed.

- Key: top-level `weight_decay_target = 'zero'` (default, today's behaviour) or `'start'`.
- Refused with `[adapter]`, `gradient_release`, `blocks_to_swap`; when `weight_decay` is 0; and with an
  optimizer whose decay cannot be switched off per group (checked per optimizer type at startup).
- Resume: the anchor must be the original start weights, not the resumed ones. Nothing extra is stored in
  the checkpoint; the anchor is an in-memory copy of the weights as loaded from the model file, taken before
  a resume loads a checkpoint (see section 7).
- The same skip flag as the guard gates it.
- Tests: the update equals the formula on fp32; on bf16 the expected value is matched on average; the
  optimizer's decay is 0 for the taken-over groups; refusals.

## 3. LoRA+ (`lora_plus_ratio`)

KohakuMoss multiplies the learning rate of each algorithm's zero-initialised factors (LoCon `lora_up`,
LoKr `lokr_w2` / `lokr_w2_b`, ...). Default 1.0 (off).

diffusion-pipe: `configure_adapter` builds PEFT `LoraConfig` / `LoKrConfig`. Plan: LoRA first, where the
zero-initialised factor is `lora_B`; a parameter group split by name, with the ratio applied to those
parameters' learning rate after the model's own learning-rate groups. LoKr is left out until the
zero-initialised PEFT factor names are confirmed in the installed PEFT (the code at `base.py:451`
re-initialises LoKr, so the original init cannot be assumed); a LoKr config with the key set is refused,
saying so. OPLoRA is a LoRA, so it is covered.

- Key: `[adapter] lora_plus_ratio = 1.0` (1.0 changes nothing).
- Interaction to check: `train.py` splits every group into weight-decay and no-decay groups with
  `pg.copy()`, which keeps the group's `lr`; the plan adds a test that the multiplied rate survives it, and
  that the cosine scheduler (built from the group rates) scales it.
- Tests: group membership by name; rates; refusal for LoKr and for a full fine-tune.

## 4. Distillation toward a frozen teacher (LoRA and full fine-tune)

Loss, as KohakuMoss defines it: `hard_weight * L(v, target) + distill_weight * L(v, v_teacher)`, the
teacher being a frozen DiT given the student's own noisy input, timestep and text embedding, under
no_grad. diffusion-pipe has an existing version only for the anima refiner
(`models/teacher_guidance.py`).

### What can be reused from the anima refiner

Reused: the schedule helpers (`shape_at`, `decay_at`, `lambda_at`), the shape check in `blend_target`,
the validation pattern and the warnings. Not reused: `TeacherGuide` (it holds a second text frontend)
and the placement of the teacher forward.

### Why the placement differs for MOSS-Image

The anima refiner runs the teacher in `prepare_inputs`, where it has the caption. MOSS-Image does not
have the text embedding there: strings cannot cross pipeline stages, so `prepare_inputs` returns token
ids and `TextEncoderLayer` (stage 0, a pipeline layer) encodes them. The teacher forward therefore
has to run inside the pipeline, after the text is encoded, on stage 0. The plan put it in
`InitialLayer`, on the belief that `InitialLayer` is not activation-checkpointed. **That was wrong:**
`checkpointable_layers = ['InitialLayer', 'TransformerLayer']`, so a teacher forward there would run
again in backward. It was built in `TextEncoderLayer`, which is never checkpointed (section 7). The
teacher velocity is passed on in the layer tuple next to the stream vector, and the loss on the last
stage reads it.

The loss function also needs the per-sample sigma for the teacher schedule. The plan adds sigma to the
label: `(target, sigma, mask)` (built that way round, see section 7), MOSS-Image only, used by evaluation too since it shares `prepare_inputs`.

### Teacher per training mode

| Mode | Teacher | Extra memory | Notes |
| --- | --- | --- | --- |
| LoRA, LoKr, OPLoRA | The same blocks with the adapter switched off (`PeftModel.disable_adapter()` is a context manager in the current PEFT release, which `requirements.txt` does not pin; `configure_adapter` returns a `peft.get_peft_model` object, `base.py:451`). With `pipeline_stages = 1` the whole DiT is on one GPU | None for weights; one extra forward per micro-batch | `pipeline_stages > 1` would need a full DiT copy on stage 0, so it is refused with a message |
| Full fine-tune | A frozen copy of the starting DiT, loaded separately | About 10 GB per GPU in bf16 (5 GB with a float8 teacher) on every GPU of stage 0 | With `pipeline_stages = 2` the copy and its full forward sit on stage 0 while stage 1 waits; stage 0 becomes the slower stage |

Cost of the forward, from KohakuMoss's own description ("about a quarter of a checkpointed full-finetune
step"): with pipeline stages the teacher forward is not split across stages, so I expect stage 0 to take
roughly half again as long. That is reasoning, not a measurement; the GPU script below measures it.

### Hazards found in the code

- The stream-penalty forward hooks (`_write_tap`) fire on every forward through a block. A teacher
  forward through the same blocks would write into the stream statistics. The teacher pass must run with
  the collection switched off (a flag on `LayerState`), and a test must prove the stream vector is
  identical with and without distillation.
- `blocks_to_swap` moves blocks between devices every step; a second full forward thrashes it, as the
  anima validation already says. Refused.
- `cache_text_embeddings` true is fine (the teacher reads the cached embedding); false is the default
  and works through the text layer.
- `compile = true` is already refused with stream penalties; the same check covers it.

### Keys

```
[teacher]
loss_weight = 0.0        # distill weight; 0 turns it off. The data loss keeps weight 1.
transformer_path = ...   # full fine-tune only; defaults to the model's own starting file
dtype = 'bfloat16'       # or 'float8' for the full fine-tune copy
shape = 'constant'       # reuses the anima helpers; 'sigmoid' needs t_mid and width
decay = 'none'           # 'linear' | 'cosine' with decay_steps
```

The anima doc warns that a teacher that never decays holds the student at the teacher's level. With the
base model as teacher that is the intent, but it caps what the fine-tune can learn, so the plan warns
once at startup when `decay = 'none'` and `loss_weight` is above a small value.

### Tests

CPU: the loss equals the formula for mse, huber and l1; the hooks stay silent during the teacher pass;
adapter-off output differs from adapter-on once the adapter is non-zero and equals the base output exactly;
refusals. GPU (`tools/test_distillation_gpu.py`): a few steps with a LoRA, then a full fine-tune on one and
two pipeline stages, reporting the step time with and without the teacher and the peak memory per stage.

### Open risk

No measured image-quality result for distillation was found in the KohakuMoss pages that were read, so the
plan builds it as an opt-in whose benefit has to be judged from your own samples.

## 5. Recommended configs

Source: `configs/moss/_base/base_lora.py`, `base_full.py`, `algo/algo_full.py`, `docs/guides/full-finetune.md`,
`docs/performance/training-5090.md`, the quality configs. The mapping below says what each KohakuMoss
value is in diffusion-pipe, what already matches, and what differs.

### LoRA

| KohakuMoss | Value | diffusion-pipe key | State |
| --- | --- | --- | --- |
| `ALGO`, `DIM`, `ALPHA`, `PRESET` | lora, 32, 32, `attn-mlp` | `[adapter] type = 'lora'`, `rank = 32` (alpha is forced equal to rank by `train.py`); targets: every linear of the 42 blocks | Matches. `attn-mlp` is the attention and feed-forward linears; diffusion-pipe's default targets every linear in the block, so check that the set is the same before assuming equality |
| `OPTIMIZER`, `LR`, `WEIGHT_DECAY` | adamw, 1e-4, 0.01 | `[optimizer] type = 'adamw_optimi'`, `lr = 1e-4`, `weight_decay = 0.01` | Matches the shipped example for these three. KohakuMoss's AdamW betas are `(0.9, 0.95)` (`scripts/train/moss.py:140`); the built example uses them, while the other shipped examples use `[0.9, 0.99]` |
| `SCHEDULE` | cosine, warmup 100, floor 10% of the lr | `lr_scheduler = 'cosine'`, `warmup_steps = 100` | Built as `lr_scheduler_min_ratio = 0.1` (section 7); without it the floor is the absolute `1e-6` (`utils/lr_schedule.py`). The decay spans the steps after the warmup, so it ends at exactly the floor on the last step |
| `GRAD_CLIP` | 1.0 | `gradient_clipping = 1.0` | Matches |
| `TIMESTEP = "moss"`, mu -1.09, s 0.95, shift 1.0, cutoff 0.95, uniform_mix 0.01 | | `timestep_sample_method = 'logit_normal'`, `logit_mean = -1.09`, `logit_std = 0.95`, `sigma_cutoff = 0.95`, `uniform_fraction = 0.01`, `shift = 1.0` | Matches (these are the shipped defaults) |
| `TEXT_DROP` | 0.1 | `caption_dropout_rate = 0.1` in the dataset file | Matches |
| `TARGET_AREA`, resize | 1 MP | `resolutions = [1024]` with aspect-ratio buckets | Equivalent |
| `STEPS` | 4000 | `epochs` (the scheduler length is `epochs * steps_per_epoch`, `train.py:1027`; `max_steps` only stops early) | Set `epochs` so the run ends near 4000 optimizer steps; do not rely on `max_steps`, or the cosine will not reach its end |
| Step size | `base_lora.py`: `MAX_TOKENS = 12288`, `GRAD_ACC = 1`; the 5090 recipes (`recipes_lora_5090.py`, `recipes_lora_ddp_4x5090.py`): about 64 images of 4608 tokens | `micro_batch_size_per_gpu * gradient_accumulation_steps * data-parallel ranks` | Differs: the built LoRA example gives 2 * 2 * 1 = 4 images per step. The 64 images belong to the 5090 recipes only, and nothing read says the learning rate was tuned for them or what to do with a smaller batch, so this is a choice to make, not a rule |
| `PRECISION = bf16` | | `dtype = 'bfloat16'` | Matches |
| `STREAM_LOSS`, distillation | off | Leave off | KohakuMoss does not enable them in its LoRA config |

File `examples/moss_image/moss_image_lora_kohakumoss.toml`: the shipped example with the table
above applied and comments naming where each number came from.

### Full fine-tune

| KohakuMoss | Value | diffusion-pipe key | State |
| --- | --- | --- | --- |
| `OPTIMIZER`, `LR` | Muon (2-D weights) plus AdamW, 5e-5. This is the `base_full.py` / `algo_full.py` setting; the runnable `recipes_full_*` recipes use `OPTIMIZER = "adamw"` and `EMA = 0.0` | `[optimizer] type = 'genericoptim'`, `muon = true`, `lr = 5e-5` | Only partly the same Muon. Shared: the `0.2 * sqrt(max(rows, cols))` scale (`optim/optimizers/muon.py`; `generic_optim.py:507-519`), the Newton-Schulz coefficients and 5 steps. Different: KohakuMoss uses Nesterov momentum 0.95, splits the fused `ffn_up_gate` and keeps some tensors off Muon; `GenericOptim` uses the EMA of `betas[0]` without Nesterov and puts Muon on every decayed 2-D weight with both dimensions at least 32. So the learning rate is a starting point, not a proven transfer. The shipped example's default is AdamW at 1e-5, five times lower |
| `WEIGHT_DECAY`, `WD_RULES = "moss"` | 0.01; none on biases, norms, 1-D, embedders (also the 2-D `video_embedder`, `text_embedder.proj` and `time_embedding.*`); last block's q/k norm decays toward 1 | `weight_decay = 0.01`; `train.py` already exempts 1-D parameters and `llm_adapter.embed` | Partly. Decay-to-one is built as `weight_decay_to_one`. The 2-D `video_embedder`, `text_embedder.proj` and `time_embedding` weights are exempt in KohakuMoss and decayed here; that difference is left as it is |
| `SCHEDULE` | cosine, warmup 500, 10% floor | `lr_scheduler = 'cosine'`, `warmup_steps = 500`, `lr_scheduler_min_ratio = 0.1` | Built, as above |
| `EMA` | 0.9999 | `ema_decay = 0.9999` | Exists (`utils/ema.py`) |
| `GRAD_CLIP` | 1.0 | `gradient_clipping = 1.0` | Matches |
| Timestep, caption dropout, 1 MP | as LoRA | as LoRA | Matches |
| `STREAM_LOSS`, `OUTLIER_DECAY` | off in the base recipe; documented for a heal run | `[model.stream_loss.*]`, `[outlier_decay]` (already implemented, commit 02f9977) | Leave commented, as KohakuMoss does |
| `WD_TARGET` | `"zero"` by default | `weight_decay_target` (section 2) | Opt-in |
| Layout | one to four GPUs, FSDP2 or DDP with weight streaming | 8 GPUs, 2 pipeline stages * 4 data-parallel, bf16 weights, activation checkpointing | Different engine; the shipped 8-GPU example stays |

File `examples/moss_image/moss_image_full_kohakumoss.toml`: the 8-GPU example with Muon at 5e-5,
warmup 500, cosine, `ema_decay = 0.9999`, and the guard keys.

Open points about the full fine-tune numbers: the 5090 full fine-tune recipe (`recipes_full_5090.py`) steps on
36,864 tokens times `GRAD_ACC = 8`, i.e. 294,912 tokens per step; `base_full.py` and the other recipes differ.
The 8-GPU example's 32 images per step is far below the 5090 figure, and nothing read says what that does
to the learning rate.

## 6. Order of work

1. The guard (section 1) and its tests; GPU script.
2. `weight_decay_target = 'start'` (section 2).
3. `lora_plus_ratio` for LoRA (section 3).
4. Distillation for LoRA, then for the full fine-tune (section 4).
5. The two example configs and the documentation (README, `docs/supported_models.md`).

Each step: CPU tests, a GPU script for what a CPU cannot show, an entry in `implementation-notes.html`, a
commit.

## Questions for you

1. Default of `skip_nonfinite_steps`: on for every model (a behaviour change for existing runs) or off
   until a config turns it on?
2. Weight decay toward the start weights, LoRA+ and distillation: generic for every model as far as the
   code allows, or MOSS-Image only for now? (Weight decay and LoRA+ are generic in this plan; distillation
   needs per-model work.)
3. Add a `min` floor for the cosine schedule (KohakuMoss: 10% of the lr), and the decay-to-one rule for
   the last block's q/k norm? Both are small and both are in the recommended full fine-tune config.
4. Batch size in the LoRA config: follow KohakuMoss (64 images per step) or keep a smaller default?

## 7. What was built, and where it differs from this plan

Everything in sections 1 to 5 was built; the code, tests and docs are in the commit that adds this section.
User-facing description: [docs/training-options.md](../training-options.md). The differences, so that the
text above is not mistaken for the code:

| Item | Plan said | Built |
| --- | --- | --- |
| Teacher placement | In `InitialLayer`, "not activation-checkpointed" | In `TextEncoderLayer`. `InitialLayer` **is** in `checkpointable_layers`, so a teacher forward there would have run twice. `TextEncoderLayer` is never checkpointed. The teacher velocity rides the layer tuple (after the stream vector, if there is one), is tied to each layer's output so it has a graph (DeepSpeed backpropagates every floating output of a stage), and `FinalLayer` returns it to the loss |
| Label | `(target, mask, sigma)` | `(target, sigma, mask)`: `utils/dataset.py` broadcasts everything before the last element from the first stage to the last (`*target_list, mask = label`), and sigma comes from the first stage's draw like the target. Only with `[teacher]` on; otherwise the label is `(target, mask)` as before |
| `[teacher]` keys | `loss_weight`, `transformer_path`, `dtype`, `shape`, `decay` | `loss_weight`, `shape`, `t_mid`, `width`, `decay`, `decay_steps`. No `transformer_path` and no `dtype`: the teacher is always the model as the run loaded it (merged adapters included), a frozen copy in the model's own dtype; a separate file would have meant a second loading path |
| Loss | `hard_weight * L + distill_weight * L` | `L(v, target) + lambda * L(v, v_teacher)`, `lambda = loss_weight * shape(sigma) * decay(step)` (the anima helpers `shape_at`, `decay_at`, `lambda_at`, reused). Two losses, not a mixed target, so `huber_delta` and `smooth_l1_beta` need no caveat |
| LoRA teacher | `PeftModel.disable_adapter()` | The same, wrapped so `requires_grad` is put back afterwards: PEFT's `enable_adapters` sets it on every adapter weight again, which would thaw a weight `train.py` froze at a zero learning rate |
| Stream-penalty hooks during the teacher pass | A flag on `LayerState` | Not needed: the taps return at once when the owning layer is not inside its own forward (`_writes is None`), and the teacher pass runs in another layer, with a `LayerState` of its own. A test proves the stream vector is identical with and without a teacher |
| Weight decay target, optimizers | `adamw`, `adamw_optimi`, `stableadamw`, `adamw8bit` | Also `genericoptim`, which the recommended full fine-tune uses (Muon). Checked against the real optimizer: its decay is exactly `lr * wd * p` with the group's `lr`, Muon groups included; a test keeps it so |
| LoRA+ | Applied where `configure_adapter` builds the adapter (`models/base.py`) | Applied to the parameter groups in `train.py` (names containing `.lora_B.`), which is where the learning rate lives and covers every model. LoKr is refused rather than guessed at (the zero-initialised factor names were not confirmed) |
| Weight decay anchor | Re-read from the model file at startup | An in-memory copy of the weights as loaded, taken before a resume loads a checkpoint, so it is the same weights without a second loading path. `blocks_to_swap` is refused (the copy would sit on another device than the swapped weight) |
| Guard | A flag inside the engine step | `utils/patches.py` computes the gradient norm once, reuses it for clipping, and `utils/step_guard.py` suppresses the optimizer step through DeepSpeed's own `optimizer.overflow` protocol. Tested in a real DeepSpeed engine on CPU |
| `p_full`, sigma weighting, l1 | Open | Not built (section "Evidence for the knobs") |

Questions answered by the user: the guard defaults to off; weight decay target and LoRA+ are generic with a
config switch; the cosine floor and decay-to-one were wanted; the LoRA example keeps the small batch.

Not verified (no GPU was available): the real 5B model; DeepSpeed pipeline stages with the extra tuple element;
bf16 on CUDA; multi-node. `tools/test_nonfinite_guard_gpu.py` and `tools/test_distillation_gpu.py` are the
scripts to run, and the teacher's step time and memory on a two-stage full fine-tune are estimates until they
are run.
