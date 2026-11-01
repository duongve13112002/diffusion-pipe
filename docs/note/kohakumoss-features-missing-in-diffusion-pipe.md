# KohakuMossTrainer features that diffusion-pipe does not have (MOSS-Image training)

Date: 2026-11-02

Update: the non-finite step guard, weight decay toward the start weights (with the decay-to-one rule), LoRA+,
the cosine floor and distillation listed as missing below were built afterwards; see section 7 of
[kohakumoss-port-plan-and-recommended-configs.md](kohakumoss-port-plan-and-recommended-configs.md). The tables
below describe the state when this note was written.

Question: which MOSS-Image training features of KohakuMossTrainer are missing from diffusion-pipe,
how hard each would be to add, and what it would bring. Research only, no code was written for it.
Scope agreed with the maintainer: MOSS-Image training only (not inference, per-GPU kernels or
KohakuMoss's config tooling); each feature rated; no single priority, so the list is grouped by
kind (image quality, training stability, VRAM and speed, adapters, data, operations).

It extends [kohakumoss-vs-diffusion-pipe-moss-training.md](kohakumoss-vs-diffusion-pipe-moss-training.md),
which compared the two trainers area by area. This note goes feature by feature and rates the gap.

## Sources and method

- KohakuMossTrainer commit `b06ac60`: every knob of `scripts/train/moss.py` (all ~165 global
  settings, read in full) and `docs/concepts/{training,adapters,data,profiles}.md`,
  `docs/internals/quant-kernels.md`, `docs/performance/training-b300.md`.
- diffusion-pipe, branch `feat/moss-image` at the commit that added the stream penalties and
  outlier decay (`02f9977`): `train.py`, `models/moss_image.py`, `models/base.py`,
  `utils/{dataset,saver,ema,captions,nan_check,optimizer_factory,lr_schedule,step_stats}.py`,
  `optimizers/generic_optim.py`, `README.md`, `docs/supported_models.md`.
- Each "missing" was checked by searching the code for the feature's name and its likely
  synonyms. "Doc" marks a KohakuMoss behaviour taken from its documentation only, not traced in its
  code. No GPU was available, so no speed or VRAM figure here was measured by me: the numbers quoted
  come from KohakuMoss's own benchmark pages and are attributed.

Effort scale: S = a few functions and tests, M = touches several modules or the pipeline layers,
L = new subsystem or needs kernels / a data-pipeline redesign.

## Already ported in this repository

`concentration`, `rms` and `write` stream penalties and `OUTLIER_DECAY` (commit `02f9977`, see
[stream-healing-implementation-plan.md](stream-healing-implementation-plan.md)). Not ported from that
area: the `peak` term (it guards fp16 headroom, diffusion-pipe trains bf16/fp32) and
`STREAM_LOSS_PROBE` (listed below).

## Image quality

| Feature | KohakuMoss | diffusion-pipe today | Effort | What it would bring |
| --- | --- | --- | --- | --- |
| Distillation toward a frozen teacher | `TEACHER`, `DISTILL_WEIGHT`, `DISTILL_WEIGHTING`, `TEACHER_PRECISION`, `TEACHER_STREAM`: loss `HARD_WEIGHT*L(v, target) + DISTILL_WEIGHT*L(v, v_teacher)`; teacher forward under no_grad at the student's own inputs; only `PARALLEL` none / ddp (doc) | None for MOSS-Image. A teacher-guided mix exists only for the anima_refiner (`models/teacher_guidance.py`) | M for LoRA (teacher = the same model with the adapter off, so no second copy); for a full fine-tune the teacher must be a frozen copy of the starting weights (about 10 GB in bf16 per GPU, less in fp8), and with pipeline stages it would sit whole on the first stage, where `prepare_inputs` runs (the anima_refiner path refuses `pipeline_stages != 1` for this memory reason). Whether it fits next to the full fine-tune state was not measured | An anchor to the starting model's function while the data loss moves it (the KohakuMoss doc's description). I did not find a measured image-quality result for it in the pages I read |
| Other losses and sigma weighting | `LOSS` = mse / huber / l1; `WEIGHTING` = none / x0 / eps; `HARD_WEIGHT`; per-dataset `loss_weight` | mse, `huber_delta`, `smooth_l1_beta`; no l1, no sigma weighting, no per-dataset weight | S to M (the loss function does not receive sigma, so the label tuple has to carry it) | Choice of what the fine-tune is penalised on; no evidence in the docs that any beats the default `none` recipe (the MOSS recipe) |
| Timestep schedule extras | `p_full` (probability mass moved to sigma 1) and `schedule = "tech_report"` (the four stage-3 parameter sets switched by step) | One sampler: the report's final-stage distribution, `shift`, `min_t`, `max_t`, `clean_end_share` by epoch | S (the sampler is small and the step now reaches the model through `set_training_step`) | Reproduces the report's later stages and lets a run put weight on pure-noise steps |
| EMA variants | List of decays, power-function EMA (EDM2) with `sigma_rel`, a snapshot at every checkpoint and `kohakumoss.training.ema.posthoc` to rebuild an average of any width afterwards | One `ema_decay`, fixed during the run | M (snapshots plus a least-squares post-hoc script) | Choosing the averaging width after training instead of re-running (doc) |
| Weight decay toward the start weights, MOSS decay rules | `WD_TARGET = "start"` (`p -= lr*wd*(p - a)`, `a` the launch weights as a bf16 copy), `WD_RULES = "moss"` (no decay on norms, 1-D tensors, embedders; decay-to-one on the last block's q/k norm) | Plain decay; weight decay off only for 1-D tensors and `llm_adapter.embed` in `train.py` | S to M; the start copy costs 2 bytes per decayed parameter per stage | Full fine-tunes drift less from the base (doc: the stock decay shrinks the pretrained weights each step) |
| LoRA+ | `LORAPLUS_RATIO`: multiplies the lr of each algorithm's zero-initialised factors | Not present | S (parameter groups by the `lora_B` / `lokr` factor names) | Faster adapter convergence in the LoRA+ paper's setting; KohakuMoss leaves it off by default (1.0) |
| Gradient probe | `STREAM_LOSS_PROBE`: per-term gradient norms and cosines against the task gradient, to set penalty weights | Not present | L (separate `autograd.grad` per term over a retained graph; KohakuMoss restricts it to `PARALLEL = "none"`) | A way to set the new stream-penalty weights from measurements instead of by analogy |

## Training stability

| Feature | KohakuMoss | diffusion-pipe today | Effort | What it would bring |
| --- | --- | --- | --- | --- |
| Non-finite step guard | A step with a non-finite loss or gradient norm is skipped and counted; `NONFINITE_PATIENCE` stops the run after N in a row; on the deferred path the flag stays on the device | No whole-step guard in `train.py`. `--nan_check` finds the first NaN but is slow and meant for debugging; `GenericOptim(skip_invalid_grads=True)` skips per parameter (default off) | M | One bad batch no longer writes NaN into the weights of a long run |
| Instability dumps | `instability/step_X.json` on a non-finite step or a loss above 5x its running average; `repro_step.pt` holds the failing micro-batch | None | M | A replayable record of the batch that blew up |
| Exact resume | One `rng.<rank>.pt` per rank: a resumed run draws the same data, sigma and noise | The checkpoint's `client_state` holds `step`, `examples`, the loader position and the loss average; no RNG state is saved. The training loop also never seeds the generators (the only `manual_seed` calls in `train.py` are in `evaluate`), so noise and timesteps differ between runs anyway | S to M (per-rank RNG file in the checkpoint) | Reproducibility only; no effect on quality. Low value here: without a training seed there is no reference run to reproduce |
| Write retries on checkpoint save | Every checkpoint write retried for up to 600 s; `FINAL_CHECKPOINT`; a failed export or sample does not stop the run (time-based saving is `SAVE_MINUTES`) | Time-based saving already exists (`checkpoint_every_n_minutes`, `utils/saver.py`), next to epochs / steps / examples; saves are atomic with `keep_last_n_checkpoints`. I found no retry of a failed write | S | Robustness on shared or flaky storage |
| Gradient reduction dtype | `DDP_REDUCE_DTYPE` fp32 / param / bf16 | `communication_data_type` follows the adapter or model dtype | S | Control over the precision of the gradient sums |

## VRAM and speed

| Feature | KohakuMoss | diffusion-pipe today | Effort | What it would bring |
| --- | --- | --- | --- | --- |
| Low-precision matmul and scaled quantized base | `BASE_PRECISION` = int8 / fp8 / fp8_compute / mxfp8 / quant, `FP8_RECIPE`, `FP8_DGRAD`, `INT8_COMPUTE`: a 16-bit file is quantized on load and the matmuls themselves run in fp8 or int8; `fp8_compute` / `mxfp8` also work for a full fine-tune by re-quantizing the fp8 copies after every step | Storing the base in fp8 for adapter training already exists: `diffusion_model_dtype` for MOSS-Image (refused for a full fine-tune in `_check_keys`), `transformer_dtype` in e.g. `models/flux.py` and `models/qwen_image.py`; Comfy int8 / fp8 checkpoints load through ComfyUI's mixed-precision ops. I did not check whether the matmul on those paths runs in low precision or the weights are upcast per call | L for the compute path and the full fine-tune; the idea is model-independent (a Linear swap) but KohakuMoss's kernels are tied to its own MOSS implementation | The VRAM saving (about 1 byte per weight) is largely already available for adapters. What is missing is fast low-precision compute and scaled formats. Speed depends on the card, by KohakuMoss's own measurements: RTX 5090 DiT sampling 0.875 s/step in bf16, 0.567 with `fp8_compute` row, 0.483 with mxfp8; B300 training GEMMs 1.2x to 1.6x with mxfp8, which the doc says is 10 to 20 percent of a merged step; on Ada (RTX 4090, L4) mxfp8 is slower than bf16 and `row` fp8 is the format to use |
| Full fine-tune on one 24 GB card | `STREAM = "swap"`: fp32 masters on the host, weight streaming, host optimizer state (`OPTIM_STATE`), gradient offload, `FUSED_STEP` (optimizer step inside backward); needs about 70 GiB of host RAM (doc) | Block swap is for adapters only; `optimizer.type = 'offload'` (torchao CPU offload) and `gradient_release` exist; full fine-tune examples use 8 GPUs with 2 pipeline stages | L | Full fine-tuning on a single consumer card. Not needed if the target is the 8-GPU server |
| Checkpointing granularity | `CHECKPOINT` = none / block / sub-block / selective / `partial:<fraction>`; activation offload `ACT_OFFLOAD` | `activation_checkpointing` true or `'unsloth'` (which offloads the saved inputs to CPU); no partial fraction | S for a fraction of the blocks, M for sub-block | Trading recompute for VRAM in finer steps |
| Packed variable-length token batching | `BATCHING = "tokens"` / `"tokens_ffd"`, `MAX_TOKENS`, `OVERLONG` (`truncate_text`, `drop_tokens`), mixed sizes in one batch, no padding | Aspect-ratio buckets, one size per batch; text lengths in a batch are run as separate groups inside the layer | L (dataset, collator and attention path) | No padding, and a token budget per GPU instead of a sample count |
| Dedicated encoder GPUs | `ENCODE_DEVICES`: text encoder and VAE in encoder processes feeding the training ranks | On-the-fly text encoding runs on the first stage's GPUs; latents are cached ahead | L | Training GPUs spend no time or memory on the encoders when captions are encoded online |
| Other parallel layouts | FSDP2, `threaded` pipeline, torch pipeline schedules (1f1b, interleaved, zero-bubble, dualpipe) | DeepSpeed pipeline plus data parallel; no ZeRO / FSDP | L | Different memory / throughput trade-offs; not clearly a gain over the current layout |
| Compile modes, per-arch kernels, memory profile catalog and `auto` planner | `COMPILE` block / model, Triton and TileLang varlen kernels with tuning tables, 240-entry profile catalog, `kohakumoss-doctor` | `compile` (whole pipeline, dynamic) | L, and tied to KohakuMoss's own model implementation | Out of the agreed scope beyond this mention |

## Adapters

| Feature | KohakuMoss | diffusion-pipe today | Effort | What it would bring |
| --- | --- | --- | --- | --- |
| More adapter algorithms | lora / locon / loha / lokr / oft / boft / glora / ia3 / dylora / tlora | lora, lokr (anything else raises `NotImplementedError` in `train.py`), plus DoRA and OPLoRA for lora | M for algorithms PEFT already has (the work is the ComfyUI-loadable save format); L for T-LoRA and DyLoRA (not in PEFT) | Different capacity / regularisation trade-offs |
| Target presets | `attn`, `attn-mlp`, `full-block`, `all`, plus `+text-refiner`, `+modulation`, `+embed`, or a LyCORIS preset file | `adapter_target_modules` (class names), default every linear of the 42 blocks; text refiner, modulation and embed linears not targeted | S to M | LoRA on the conditioning path and the text refiner |
| Adapter extras | `ADAPTER_MODE` bypass / merge, `DORA_DETACH`, `TRAIN_NORMS`, input and module dropout, `TRAIN_TEXT_REFINER` | rank dropout (LoKr), `dropout` (LoRA); in a full fine-tune the text refiner trains unless `text_refiner_lr = 0` | S each | Small |

## Data

| Feature | KohakuMoss | diffusion-pipe today | Effort | What it would bring |
| --- | --- | --- | --- | --- |
| Dataset sources | kohya repeat folders, folder, jsonl, parquet, webdataset, `moss_shard`, GBC graph datasets, mixed with float `weight` | Directories with same-name `.txt` captions only (I found no jsonl, parquet or webdataset reader in `utils/dataset.py`) | M per source | Training from sharded or tabular data without unpacking it |
| Caption templates | `tags` with fixed prefix and `%%` groups, group dropout, `nl`, `mix` (natural language with probability `p_nl`), `json` with field dropout, `gbc` | `tag_dropout_rate`, `shuffle_tags`, prefix tags, multi-line captions, line filters, `caption_dropout_rate` | M | Richer caption augmentation |
| Fractional dataset weights, per-dataset loss weight | `weight` (a float; the integer part is repeats, the fraction a probability), `loss_weight` | `num_repeats`, integer | S | Mixing proportions without duplicating data |
| Flip and random crop | `FLIP`, `CROP = "random"` (needs an online latent cache) | I found no flip option; crops follow the aspect-ratio buckets | S to M | Augmentation |
| Online cache write-back and fp8 cache | `LATENT_CACHE = "online"`: encode while training and write the results back; cache dtype fp8 (about 7 percent max relative error, half the bytes of bf16) | Caching is a pass before training (`--cache_only`, `use_cached_*_only` to train on a partial cache) | M | Start training before the whole dataset is encoded |
| Catalog and rank balancing | Catalog cache of the scan, `DATA_BALANCE` (batches dealt to ranks by estimated cost) | Metadata cache and scan options (`metadata_read_threads`, `reuse_metadata_cache`) | M | Less idle time on mixed-size data |

## Operations and observability

| Feature | KohakuMoss | diffusion-pipe today | Effort | What it would bring |
| --- | --- | --- | --- | --- |
| Samples while training | `SAMPLE_EVERY`, prompts, seeds, CFG, skip layers, sample sheet, `SAMPLE_AT_START`; fixed-noise validation per sigma. Not available in its pipeline layouts (doc) | `--test_sample` writes one image and exits; eval loss at nine timestep quantiles during training | M with one stage; L across pipeline stages, where `sample()` (which calls the whole `pipeline_model`) cannot run | The most direct view of what a run is doing |
| Throughput, MFU, VRAM peaks, profiler windows | `METRICS_LEVEL`, `TRACE_STEPS`, `PROFILE_STEPS` | `train/grad_norm`, `grad_clipped`, `update_norm`, `update_ratio`, `loss_avg`; Tensorboard and wandb | S to M | Spotting slow or wasteful runs |
| Export options | ComfyUI file with fused or split q/k/v (`SAVE_FUSED_QKV`), merged DiT of an adapter (`SAVE_MERGED`), diffusers keys, exports inside every checkpoint | Adapters in ComfyUI format and diffusers transformer directories for a full fine-tune; fused q/k/v files load but are saved split; merging goes the other way (`merge_adapters` at load) | S to M | Convenience |
| Platform | Windows and Linux, XPU, sm70 up | Linux with CUDA and DeepSpeed | Not a feature to port | Noted for completeness |

## Suggested order (judgement, not measured)

For a user who cares about quality, stability and VRAM equally, in order of benefit against effort:

1. Non-finite step guard with a patience limit and an instability dump (stability, M): a full
   fine-tune that hits one NaN currently has no whole-step protection.
2. Samples while training for single-stage runs (quality feedback, M).
3. The small objective and optimizer knobs: sigma weighting and l1 loss, `p_full`, LoRA+,
   `WD_TARGET = "start"` (S each; the last matters for full fine-tunes).
4. Nothing from exact resume: `checkpoint_every_n_minutes` already exists and the per-rank RNG has no quality effect (the loop is unseeded).
5. Distillation toward the starting model (M to L; the biggest quality feature, but pipeline
   stages make it hard, and the KohakuMoss pages I read give no measured quality result).
6. Fast low-precision matmul (L; the fp8 base storage for adapters already exists) and, only if
   single-card full fine-tuning matters, the host-master streaming path (L).
7. Packed token batching and extra data sources (L / M each), when the dataset or GPU budget calls
   for them.

## Open points

- I did not find a measured image-quality result for distillation, LoRA+ or the weight-decay
  targets in the KohakuMoss pages I read, so the "what it would bring" column states what the docs
  say each does, not that it improves images.
- Whether ComfyUI's quantized-checkpoint path inside diffusion-pipe runs low-precision matmuls or
  dequantizes per call was not checked; that decides how much of the fp8 speed gain is already
  available for adapters.
- Features that exist only on the diffusion-pipe side (DeepSpeed pipeline parallelism, OPLoRA,
  block swap, aspect-ratio bucket tooling, caption filtering) were left out: this note lists gaps
  in one direction only.
