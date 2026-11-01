# KohakuMossTrainer vs diffusion-pipe: MOSS-Image training compared

Date: 2026-11-01

Update (2026-11-02): several differences listed here (the non-finite step guard, weight decay toward the start
weights, LoRA+, the cosine floor, distillation) were closed afterwards; see section 7 of
[kohakumoss-port-plan-and-recommended-configs.md](kohakumoss-port-plan-and-recommended-configs.md). The tables
below describe the state when this note was written.

Question: the `docs/concepts/stream-healing.md` page of KohakuMossTrainer describes a problem with
MOSS-Image finetunes (massive residual-stream channels and sparse weight outliers). Does the
diffusion-pipe MOSS-Image trainer handle it, and what else differs between the two training scripts?

Sources read:

- diffusion-pipe, branch `feat/moss-image`, commit `70b8468`: `train.py`, `models/moss_image.py`,
  `models/base.py`, `utils/optimizer_factory.py`, `utils/ema.py`, `examples/moss_image/*.toml`,
  `docs/supported_models.md#moss-image`, ComfyUI's `comfy/ldm/moss_image/model.py`.
- KohakuMossTrainer (https://github.com/KohakuBlueleaf/KohakuMossTrainer), commit `b06ac60`:
  `src/kohakumoss/training/stream_loss.py`, `training/optim/outlier.py`, `model/block.py`,
  `training/step.py`, `training/build/loop.py`, `diffusion/timestep/moss.py`, and
  `docs/concepts/{training,stream-healing,numerics,adapters,data}.md`.

"Code" below means the claim was checked in source. "Doc" means it comes from the KohakuMoss docs
only and was not traced in its code.

## Answer to the original question

diffusion-pipe does not handle the problem. A search of the repository finds no stream penalty, no
hook on block outputs, no outlier decay and no use of the words massive channel or outlier outside
text-encoder embedding caching (`utils/embedding_dtype.py`). The loss is the generic MSE of
`models/base.py` (`get_loss_fn`). The only related knobs are `weight_decay = 0.01` (plain
decoupled decay, which KohakuMoss describes as shrinking the whole model and leaving the
concentration in place) and the per-part learning rates (`attn_lr`, `ffn_lr`, `norm_lr`,
`text_refiner_lr`, `modulation_lr`, `embed_head_lr`, `block_lr_scale`).

## Where KohakuMoss hooks in (code)

- `StreamPenalty` (`training/stream_loss.py`) registers a forward hook on every joint block for the
  `concentration`, `peak` and `rms` terms. The `write` term needs the attention write and the FFN
  write separately, so `JointBlock` (`model/block.py`) calls `self._tap_attn(out, x, r)` and
  `self._tap_ffn(out, x, 1.0)` inside `attn_part` / `ffn_part`. The write is `out - x * r`.
- The penalty is added to the task loss in `training/step.py` (`_composite`): `loss + penalty *
  share`, `share` being the micro-batch's part of the step's loss tokens.
- `OutlierDecay` (`training/optim/outlier.py`) runs after the optimizer step on each targeted
  parameter: `p -= lr * rate * sign(p) * relu(|p| - cap * r0)`, `r0` being the tensor's RMS at the
  start of the run (`start_refs`).
- KohakuMoss validates both for a narrow setup: `STREAM_LOSS` only with `PARALLEL` `none` or `ddp`
  (`stream_loss_problems`); `OUTLIER_DECAY` only for a full finetune, an `adamw` / `adamw8bit` /
  `adamw8bit_kernel` / `muon` optimizer, device-stepped fp32 weights and no FSDP (`outlier_problems`).

## What this means for porting to diffusion-pipe (code)

1. Block structure. ComfyUI's `Block.forward` (`comfy/ldm/moss_image/model.py`) does both residual
   adds inline (`x = residual(x, self.proj["o"](out), mod, 2, ref)` and the FFN one), so there is
   no module boundary at the attention write. A tap needs either a change in the ComfyUI
   submodule, or forward hooks on `proj["o"]` / `proj["ffn_down"]` whose outputs are multiplied by
   the gate (`mod[2]`, `mod[5]`) that `TransformerLayer` in `models/moss_image.py` already builds.
   The second option was not tried.
2. Pipeline parallelism. diffusion-pipe's full finetune example uses `pipeline_stages = 2`, and the
   loss is computed on the last stage only. KohakuMoss itself refuses `STREAM_LOSS` under its
   pipeline layouts, so a stream penalty across stages would be new work.
3. Weight precision. diffusion-pipe trains bf16 weights; `optimi.AdamW` applies Kahan summation
   automatically for bf16 parameters (checked in its signature and docstring, `kahan_sum=None`).
   The decay step is about 1.2e-4 of the excess per step at lambda 6 and lr 2e-5 (KohakuMoss
   doc), below bf16 resolution, which is why KohakuMoss requires fp32 device-stepped weights.
4. Checkpointing. KohakuMoss's tap runs in every grad-enabled forward so that a checkpoint
   recompute saves what the forward saved (`_tap`, `torch.is_grad_enabled()`). diffusion-pipe uses
   activation checkpointing per layer, which would need the same care.

## Other differences between the two trainers

| Area | diffusion-pipe | KohakuMossTrainer | Verified |
| --- | --- | --- | --- |
| Transformer | ComfyUI's native `comfy.ldm.moss_image` (differentiable path) | its own `kohakumoss.model` (`JointBlock`, fused ops, own Triton / TileLang kernels, per-device autotune, `COMPILE` modes) | code / doc |
| Objective | velocity MSE; `huber_delta` or `smooth_l1_beta` optional (`models/base.py`) | `LOSS` mse / huber / l1, `WEIGHTING` (x0, eps), `HARD_WEIGHT` | code / doc |
| Distillation | none for MOSS | `TEACHER`: frozen DiT velocity at the student's inputs, weighted by `DISTILL_WEIGHT` | doc |
| Loss normalization | per micro-batch mean; a step's micro-batches come from one size bucket, so the image-token count is the same; `batch_fill_strategy` and `_batch_weight_scale` fix padded samples (`docs/note/batch-fill-strategies.md`) | per image token over the global token count of the step, all-reduced across ranks | code / doc |
| Timestep sampler | one sampler: the report's final-stage distribution (mean -1.09, std 0.95, cutoff 0.95, 1% uniform) on a 1000-step grid, `shift`, `min_t` / `max_t`, `clean_end_share` with an epoch schedule | the same pmf formula; also `p_full` (mass moved to sigma 1) and `schedule="tech_report"`, which switches parameters by step through the report's four stage-3 intervals | code |
| Precision | bf16 or fp32 only (`models/moss_image.py` raises for others) | bf16 or fp16; `FP16_FIX` (`online`, `fp32_residual`, or a weight fold) because the base stream peaks near 62k, close to fp16's limit; fp32 master weights for trained parameters | code / doc |
| Optimizers | `adamw`, `adamw8bit`, `adamw_optimi`, `stableadamw`, `adamw8bitkahan`, `offload`, `automagic`, `genericoptim` (Muon option), pytorch_optimizer names | `adamw`, `adamw16` (rounding nearest / stochastic / kahan), `adamw8bit`, `adamw8bit_kernel`, `muon`, `prodigy`, `schedule_free` | code / doc |
| Weight decay | plain `weight_decay` | `WD_RULES="moss"` (no decay on norms, 1-D tensors, embedders; decay-to-one on the last block's q/k norm), `WD_TARGET="start"` (decay toward the start weights) | doc |
| Step safety | DeepSpeed gradient clipping | non-finite step skipped and counted, `NONFINITE_PATIENCE`, repro dump, `STEP_SYNC="deferred"`, fused in-backward step | doc |
| Learning rate per part | `attn_lr`, `ffn_lr`, `norm_lr`, `text_refiner_lr`, `modulation_lr`, `embed_head_lr`, `block_lr_scale` | no per-part knob in the 165-knob list of `scripts/train/moss.py`; `LORAPLUS_RATIO` for adapters | code |
| Adapters | `lora`, `lokr`, OPLoRA; DoRA helpers in `utils/dora.py` (not read in detail) | LyCORIS `lora`, `loha`, `lokr`, `oft`, `dora`, `full`; T-LoRA masks, DyLoRA, LoRA+, module dropout | code / doc |
| EMA | `ema_decay`, `ema_update_every`, `ema_start_step`, `ema_device`, `ema_dtype` (bf16 with stochastic rounding), `ema_init_path` | several decays, power EMA (EDM2) with a post-hoc `sigma_rel` fit | code / doc |
| Parallelism | DeepSpeed pipeline stages x data parallel | `none`, `ddp`, `fsdp`, `pipeline` (torch.distributed), `threaded`; Linux only for the pipelines | doc |
| Memory | block swap (adapters only), fp8 base (adapters only), activation checkpointing | memory profiles: weight streaming, host fp32 masters, host optimizer state, gradient and activation offload, `FUSED_STEP`, int8 / fp8 / mxfp8 base | doc |
| Data | aspect-ratio buckets, global batches per size bucket, latent and text-embedding caches (text cache in float8 optional) | token batching (`MAX_TOKENS`) on packed varlen sequences, token drop, online cache write-back, KohakuClip shards, GBC datasets | doc |
| Validation and samples | eval loss at timestep quantiles, `--test_sample` | fixed-noise validation per sigma, sample grids during training | doc |
| Resume | atomic save, `keep_last_n_checkpoints` | atomic save, one RNG file per rank, so a resumed run draws the same data, sigma and noise | doc |
| Stream and weight healing | none | `STREAM_LOSS` (`concentration`, `peak`, `rms`, `write`), `OUTLIER_DECAY` | code |

## Open points

- KohakuMoss's doc says the full run with the combined `write` + `OUTLIER_DECAY` configuration is
  still pending, so there are no measured results for the combination yet.
- The ComfyUI block exposes no module boundary at the attention write; the hook-on-projection
  alternative above was only read, not run. This machine is CPU-only, so nothing here was trained.
