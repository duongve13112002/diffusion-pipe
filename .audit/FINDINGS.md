# Anima Refiner audit — findings at session interruption

Baseline `fd00a5e` -> HEAD `d895ad8`. Working tree has NO tracked source changes.
Submodule pointers identical to baseline.

**STATUS: INCOMPLETE. No final verdict issued.** See "WHAT REMAINS".

## CRITICAL

### C-1 ComfyUI-backed models crash at dataset construction (regression, unrelated to Anima)
`utils/dataset.py:1345` `vae_identity=model.vae_cache_key()` — unconditional.
AST-proven: `vae_cache_key` is defined ONLY on `BasePipeline` (`models/base.py:389`).
`ComfyPipeline` (`base.py:595`) inherits `CommonPipeline` (`base.py:216`), which lacks it.
`train.py:442` constructs `Dataset` for every model => AttributeError before training starts.
Affected (8): z_image, hunyuan_video_15, flux2, ernie_image, ltx2, ideogram4, krea2, minimax_h3.
Proof of oversight: the sibling call at `dataset.py:1441` IS guarded with
`getattr(self.model, 'text_encoder_cache_key', lambda _i: '')`.
Introduced by `3b71586`; the guarded pattern already existed from the earlier `758e449`.
Test coverage: ZERO — no test constructs the top-level `Dataset` (14 construct `DirectoryDataset`
directly, bypassing line 1345).
FIX: move `vae_config_keys` / `vae_cache_key` / `text_encoder_cache_key` / `text_encoder_identity`
from `BasePipeline` up to `CommonPipeline`; add a test building `Dataset` with a ComfyPipeline.

## HIGH

### H-1 ZeRO + grad_accum + guided rollout: uncond branch gradient is N times too large
PROVEN NUMERICALLY on a real DeepSpeed 0.18.4 engine (`.audit/exp_zero_scaling.py`):
gas=4, ZeRO-1, gloo -> grad via `engine.forward` = 1.0x reference; grad via BARE module = 4.0x.
Mechanism: DeepSpeed does NOT scale inside `engine.backward()`. `engine.py:2490` is explicitly
commented `# Used only for return value`. The 1/N is a hook on the OUTPUT of `engine.forward()`:
`engine.py:2237-2243` -> `_backward_prologue_per_tensor` -> `engine.py:2362-2366` `grad / gas`.
`tools/distill_refiner.py:1627` calls the bare `refiner(...)` (not `train_module`) deliberately
(comment at :1616 explains avoiding a second hook manager) — which also bypasses the only scaling.
Trigger: `distributed_strategy` = zero1|zero2 AND grad_accum > 1 AND `[rollout] loss_weight` > 0
AND `guidance_scale` > 1.
BLAST RADIUS: no shipped config triggers it — all four use `ddp` and `guidance_scale = 0.0`.
DDP is immune (divides the whole scalar once, `distill_refiner.py:958-963`).
FIX: register a `/N` hook on `student_uncond` under ZeRO, or route both branches identically.

## MEDIUM (each independently confirmed by the orchestrator)

- **M-1 Cache identity permanently inert on pre-existing installs.** `utils/dataset.py:151`
  `return cache` fires BEFORE `cache.write_manifest()` at `:206`, so an already-complete cache
  never gains a manifest. `utils/cache.py:121` claims it "gets a manifest the next time it is
  written in full" — false. An existing anima install that swaps `llm_path` silently reuses the
  old encoder's embeddings, exactly what `d895ad8` set out to prevent.
  FIX: `write_manifest()` before the early return.
- **M-2 Unconditional text-embedding cache carries no identity.** `dataset.py:1293` passes
  `identity=identity`; the uncond call at `:1298-1307` omits it. For anima
  (`text_encoder_cache_key == ''`) the uncond fingerprint is constant across all runs, so swapping
  `llm_path` rebuilds conditional embeddings but serves the OLD encoder's unconditional one.
  FIX: pass `identity=` (one line).
- **M-3 Stable checkpoint names written non-atomically.** `shutil.copy2` at
  `distill_refiner.py:1696, 1705, 1719` truncates the destination in place — the exact hazard
  `_save_file_atomically` (`:1801`) exists to prevent for the tagged file. Every shipped config
  points at the stable name. FIX: copy to `.tmp` then `os.replace`.
- **M-4 Nothing pins resumed weights to resumed optimizer state.** `refiner_provenance`
  (`:1774-1800`) records llm_path/hidden_layer/cap_feat_dim/max_text_length but NO step. With M-3,
  a crash between copies leaves weights at step N with optimizer state at step M and the resume
  succeeds silently. FIX: record step in metadata, compare on load.
- **M-5 Refiner precision asymmetry between the two saved artefacts.** `save_full_model`
  (`:1764`) writes the refiner `.to(dtype)` (bf16 default); `save_refiner` (`:1811`) writes
  `.float()` with a docstring arguing bf16 "threw away sixteen mantissa bits ... on every resume".
  Both files are valid `resume_from` sources (`build_student:441`).
- **M-6 One-time latent-cache invalidation for EVERY model in the repo.** Baseline appended
  `dataset._fingerprint`; HEAD appends a SHA over selected columns (`dataset.py:74-88`) and
  `cache_latents` passes `latent_columns`. `keep_latent_cache` defaults False (`dataset.py:392`)
  => "[CACHE] Fingerprint changed, deleting existing cache files". Loud, but undocumented.
  FIX: document in README and `examples/dataset.toml`.

## LOW

- **L-1 Two more tests CANNOT FAIL** (`d895ad8` fixed two others; these remain):
  `test_distill_refiner.py:141-151` `test_setup_distributed_single_process` — the second
  parametrisation makes `env` truthy, so `if not env:` executes ZERO assertions.
  `test_distill_refiner.py:153-159` `test_the_probe_seed_is_not_rank_offset` — asserts
  `'manual_seed(seed)' in src`, which still matches line 1310 even if the probe seed at line 1395
  became `seed + rank`. It cannot detect the inversion it names, and that invariant is the one
  whose violation makes every rank optimise a different objective.
- **L-2** `keep_last_n_checkpoints` is validated in the distiller but silently ignored in
  `train.py` (documented at `examples/main_example.toml:90`; `utils/saver.py:66` does
  `if not keep: return`).
- **L-3** The corpus caption-settings warning (`distill_refiner.py:296-306`) fires only when NONE
  of the five keys is restated, so restating one and forgetting `prefix_tag_caption` is silent.
  The shipped corpus configs do restate correctly.
- **L-4** `models/base.py:145` `tar_f.extractfile(str(spec[1]))` is an unfixed instance of the
  documented "as_posix for archive members" lesson. PRE-EXISTING at `fd00a5e` (byte-identical),
  so NOT a regression of this branch. `utils/dataset.py:1153` does it correctly.

## REFUTED / CLEARED — do not re-report

- **R-1 REFUTED:** "relational_loss NaNs at batch_size > 25 via the cdist zero diagonal."
  Measured: at n=26/48 the implementation does switch (`CdistBackward0` -> composite) and the
  diagonal stops being exactly 0 (5.5e-3), but gradients are FINITE at unit scale, at /512 scale,
  and even with a FORCED exactly-zero diagonal. No discontinuity across n=25/26.
- **R-2 CLEARED:** `RotaryEmbedding.inv_freq` (non-persistent buffer, not covered by
  `init_weights`). `accelerate.init_empty_weights` defaults `include_buffers=False`
  (`big_modeling.py:91`, resolved from `ACCELERATE_INIT_INCLUDE_BUFFERS`). Documented
  (`upstream-api-drift-audit.md:82`) AND guarded by construction in
  `tools/check_vendored_apis.py:246-266`.
- **R-3 CLEARED:** the ZeRO accumulation-boundary bug (`50730bc`) is genuinely fixed.
  `engine.py:2529-2533` honours the manual override; `micro_steps` advances only in `step()`
  (`:2762`). The local pattern matches deepspeed's own docstring. No double clipping
  (`gradient_clipping` delegated) and no double scheduler step.
- **R-4 CLEARED:** caption/embedding ORDER pairing. `dataset.py:432-437` shuffles
  (index, caption) PAIRS; premise verified — `flatten_captions` (`:228-239`) iterates metadata order.
- **R-5 CLEARED:** the `create_lr_scheduler` `T_max` quirk is byte-identical to `fd00a5e` —
  pre-existing, documented, deliberately left. NOT a regression.
- **R-6 CLEARED:** no config key in any of the four shipped `distill*.toml` is unread
  (23/33/29/39 keys checked, all referenced).
- **R-7 CLEARED:** shared checkpoint retention (`utils/saver.py`) is integer-sorted, always keeps
  the newest, rank-0 only, off by default. No regression for existing users.
- **R-8 CLEARED:** CFG is textbook — `distill_refiner.py:500`
  `v = v_uncond + g*(v_cond - v_uncond)`; the dangerous `0 < g <= 1` band is refused at `:1291`.
  Matches the sampler.
- **R-9 CLEARED:** the rollout is teacher-owned, fully under `no_grad`, the student is evaluated
  at the SAME (x,t) against the stored teacher velocity, the Euler sign is correct, and the lowest
  t is 0.125 at steps=8 exactly as documented.

## Environment note

An earlier full-suite run hard-crashed (Windows access violation in `torch.optim.adam`) at
`test_stage1_freeze_leaves_dit_untouched`. That test PASSES in isolation. Free RAM was ~1.6 GB
with three subagents live on an 8 GB box. Attributed to memory pressure, NOT a code defect.
Toolchain matches CLAUDE.md: torch 2.13.0+cpu, Python 3.12.10, deepspeed 0.18.4 — and deepspeed
is REAL here, not shimmed (conftest's stub disables itself when the package is installed).

## WHAT REMAINS

1. ~~Canonical full pytest run.~~ **DONE: 490 passed, 1 skipped, 843s, exit 0** — exactly the
   figure CLAUDE.md documents. This confirms the earlier access violation was memory pressure,
   not a code defect. Note what it does NOT prove: the green suite does not catch C-1 (no test
   constructs the top-level `Dataset`) or H-1 (the ZeRO tests use a fake engine or compute the
   loss purely from the engine output). Log: `.audit/pytest_full.log`.
2. **Adversarial falsification pass** (was killed mid-run): attack C-1 and H-1, and hunt the
   under-covered surfaces — `tools/sample_anima_refiner.py`, the `train.py` anima_refiner path
   (`get_param_groups`, `to_layers`, `ContextRefinerLayer`, `save_adapter`/`load_adapter_weights`,
   pipeline-stage split), `utils/oplora.py` `exclude_names`, `utils/captions.py` round-trip fidelity.
3. **Reported by Opus C, NOT yet orchestrator-verified:** ZeRO fp32 master weights and dynamic
   loss scale are not checkpointed; `bf16-full` under DDP gives bf16 params AND bf16 Adam moments
   with no master copy and no Kahan guard by default; `protect_tag` covers only the tag just
   written, so a second run into a populated `output_dir` prunes its own fresh checkpoints;
   ZeRO stage 2 has no real-engine coverage.
4. **Reported by Opus B, NOT yet orchestrator-verified:** the relational term's magnitude is
   batch-size dependent ((B-1)/B, diagonal and duplicated pairs inside a `mean` reduction); the
   rollout's default `guidance_scale = 0` means the trajectory is not the CFG inference
   trajectory; rollout resolution 256 vs sampler default 1024; probe `num_heads` hardcoded to 16;
   `sample_anima_refiner.py` hardcodes 16 latent channels; two stale doc claims
   (`design-notes.md` `num_queries = 64` vs code 256; `denoising-rollout.md` "only the summed loss
   is logged" — per-term logging now exists).

## GPU-only (cannot be settled on this box)

- The fp16-mixed GradScaler path: `torch.amp.GradScaler('cuda')` self-disables on CPU and
  `Precision.autocast` returns `nullcontext` off-CUDA, so every fp16-mixed test runs the fp32 path.
- Whether fused CUDA SDPA really returns NaN for a fully-masked row (the premise of
  `allow_fully_masked_rows`).
- Multi-GPU: DDP `no_sync` across real ranks, the ZeRO shard save/load round trip
  (`tools/test_zero_resume_gpu.py`), and `tools/test_rollout_multirank.py`.
- Loss-term balance at production scale, and sampling quality. No training run, no multi-GPU run
  and no image has ever been produced from this branch — stated in the branch's own docs.

---

# FIX STATUS (2026-09-02)

| ID | Severity | Status | Where |
| --- | --- | --- | --- |
| C-1 ComfyPipeline crash | CRITICAL | FIXED | cache-identity members moved to `CommonPipeline` (`models/base.py`); tests in `test_dataset_smoke.py::TestVaeCacheKeyIsDeclaredPerModel` |
| H-1 ZeRO side-branch scaling | HIGH | FIXED | `strategy.scale_side_branch()` in `tools/distill_refiner.py`; proved on a real engine (1.0x vs 4.0x); tests in `TestSideBranchAccumulationScaling` |
| M-1 manifest never written for a complete cache | MEDIUM | FIXED | `utils/dataset.py` writes the manifest before the early return; test `test_an_already_complete_cache_still_acquires_a_manifest` |
| M-2 uncond cache had no identity | MEDIUM | FIXED | `identity=` threaded through; test `test_the_unconditional_cache_records_the_same_identity` |
| M-3 non-atomic stable checkpoint names | MEDIUM | FIXED | `_copy_atomically()`; test `test_the_stable_name_is_never_left_truncated` |
| M-4 weights/optimizer step not pinned | MEDIUM | FIXED | step recorded in safetensors metadata, checked on resume; tests in `TestCheckpointHalvesStayTogether` |
| M-5 refiner rounded to bf16 in the full model | MEDIUM | FIXED | `save_full_model` writes the refiner fp32; test `TestFullModelKeepsTheRefinerInFp32` |
| M-6 undocumented one-time latent-cache rebuild | MEDIUM | FIXED (docs) | `examples/dataset.toml` and README "Recent changes" |
| L-1 two tests that could not fail | LOW | FIXED | both now assert on something an inversion breaks; verified by mutating the probe seed |
| L-2 `keep_last_n_checkpoints` unvalidated in train.py | LOW | FIXED | `set_config_defaults` raises on a non-positive value |
| L-3 corpus warning silenced by partial restatement | LOW | FIXED | second branch warns about a missing `prefix_tag_caption`; tests in `TestCorpusCaptionSettingsWarning` |
| L-4 `models/base.py:145` `extractfile(str(...))` | LOW | NOT FIXED | PRE-EXISTING at `fd00a5e`, unchanged by this branch. Windows-only, and the branch is not the owner. Left deliberately rather than widening the diff. |
| R-1 cdist NaN at batch > 25 | — | REFUTED, no change | measured finite at unit scale, /512 scale, and with a forced exactly-zero diagonal |

## Still open — reported by specialists, NOT independently verified, NOT fixed

Fixing these blind would be guessing. Each needs the evidence step named beside it.

1. ZeRO does not checkpoint the fp32 master weights or the dynamic loss scale, so a resumed
   `bf16-full`/`fp16-mixed` ZeRO run restarts its masters from bit16 values.
   Settle with: a 2-GPU resume comparing masters before and after (`tools/test_zero_resume_gpu.py`).
2. `bf16-full` under DDP gives bf16 params AND bf16 Adam moments with no master copy, while the
   numerically milder `fp16-full` is refused outright. The shipped 4-GPU configs pair it with
   `AdamW8bitKahan`, which is the compensation; the default `adamw` has none.
   Settle with: 200 steps bf16 vs fp32 on a toy Linear, counting parameters that never moved.
3. `protect_tag` shields only the tag just written, so a second run into a populated `output_dir`
   prunes its own fresh checkpoints. Settle with: a retention test across two simulated runs.
4. ZeRO stage 2 has no real-engine coverage (the real-engine test hardcodes stage 1).
5. The relational term's magnitude scales as (B-1)/B because the `mean` reduction includes the
   zero diagonal and counts each pair twice, so `relational_loss_weight = 1.0` does not mean the
   same thing at batch 8 and batch 48.
6. Rollout fidelity gaps: the default `guidance_scale = 0` walks a non-CFG trajectory while
   sampling uses `--cfg 5`; the rollout runs at 256px while the sampler defaults to 1024px.
7. Minor hardcoding: probe `num_heads` defaults to a literal 16 rather than the checkpoint's head
   count; `tools/sample_anima_refiner.py` hardcodes 16 latent channels.
8. Two stale doc claims: `design-notes.md` documents `num_queries = 64` (code uses 256);
   `denoising-rollout.md` says only the summed loss is logged (per-term logging now exists).

## Second pass — previously-open items now closed

| Item | Status | Where |
| --- | --- | --- |
| `bf16-full` + non-Kahan optimizer | FIXED (warning + docs) | `validate_config_early` warns; `distill.toml`, both 4-GPU configs and `docs/anima_refiner/README.md` explain that `adamw8bit` and `adamw8bitkahan` save the SAME memory and only Kahan addresses bf16 parameter rounding. Tests: `TestBf16FullNeedsKahan` |
| `protect_tag` pruning the current run's own checkpoints | FIXED | `prune_distill_checkpoints` sorts this run's tags after every tag it did not write, so the OLDEST run is pruned. First attempt (protect-only) was wrong: it pruned nothing and the directory grew without bound. Test: `test_a_second_run_prunes_the_old_checkpoints_not_its_own` |
| probe `num_heads` hardcoded 16 | FIXED | defaults to `teacher_dit_config['num_heads']` (16/40/20 by width) |
| sampler hardcoded 16 latent channels | FIXED | `sample()` takes `in_channels`; `main` passes `pipeline.transformer.in_channels` |
| `design-notes.md` `num_queries = 64` | FIXED | now documents `2 * head_dim` and why it is tied to `head_dim` |
| `denoising-rollout.md` "only the summed loss is logged" | FIXED | per-term logging exists; text corrected |

Note: the sampler fix was caught by the suite — reaching through `pipeline.transformer` broke six
tests that pass `pipeline=None`. Making `in_channels` an explicit parameter is both testable and
free of the hardcoded fallback.

## Still open after the second pass — need hardware or a measurement

1. ZeRO does not checkpoint fp32 master weights or the dynamic loss scale. Needs
   `tools/test_zero_resume_gpu.py` on 2 GPUs; writing masters back onto DeepSpeed's flat
   partitions is not safe to guess at.
2. ZeRO stage 2 has no real-engine coverage (the real-engine test hardcodes stage 1).
3. Relational term magnitude scales as (B-1)/B (zero diagonal + each pair counted twice in a
   `mean` reduction). MEASURED on CPU with a constant per-pair disagreement: the raw loss varies
   about 2x between B=8 and B=48, while (B-1)/B predicts only 1.12x -- so the diagonal is real
   algebraically but is NOT the dominant batch-size effect, and the rest is sampling noise that
   settles by B>=16. Not changed: the correction would be ~12% and would move every existing
   run's loss balance for no measured gain.
4. Rollout fidelity: default `guidance_scale = 0` walks a non-CFG trajectory while sampling uses
   `--cfg 5`; rollout runs at 256px while the sampler defaults to 1024px. Both are design
   choices with a cost, not defects.
5. `models/base.py:145` `extractfile(str(spec[1]))` — pre-existing at `fd00a5e`, left out of
   this diff deliberately.

---

# GPU validation plan

Smallest experiments that would retire the most remaining risk, in priority order. None of these
needs a full training run.

## P1 — Does the ZeRO side-branch fix hold across real ranks?
Test: 2 GPUs, `distributed_strategy = 'zero1'`, `gradient_accumulation_steps = 4`,
`[rollout] loss_weight = 1.0`, `guidance_scale = 5.0`, ~50 steps. Repeat with
`distributed_strategy = 'ddp'`, same seed and data.
Why GPU: `deepspeed.initialize` needs a real accelerator for multi-rank; the fix is proved only
on a single-rank CPU engine.
Expected: the two loss curves track each other closely. Before the fix they would diverge, the
ZeRO run behaving as if the uncond branch had 4x the learning rate.
Failure signature: ZeRO loss falls faster early then plateaus higher, or grad norm is
systematically larger under ZeRO.

## P2 — ZeRO resume actually restores what it claims
Test: `deepspeed --num_gpus=2 tools/test_zero_resume_gpu.py`, then a real 200-step run with a
mid-run restart, comparing weights and loss either side of the restart.
Why GPU: the premise (`initialize` replaces param_groups with a rank-local flat fp32 partition)
is confirmed at source level, but the round trip through `load_state_dict` onto those partitions
is not.
Expected: loss continues smoothly across the restart. This also settles open item 1 (fp32 master
weights not being checkpointed) — if masters are lost, a `bf16-full` ZeRO resume shows a visible
step in the loss.

## P3 — fp16-mixed actually works
Test: 30 steps at `precision = 'fp16-mixed'` on one GPU, watching for skipped optimizer steps and
a stable loss scale.
Why GPU: `torch.amp.GradScaler('cuda')` self-disables without CUDA and `Precision.autocast`
returns a null context off-CUDA, so every fp16-mixed test on the dev box silently runs fp32.
Failure signature: loss scale collapsing toward 1, or every step skipped.

## P4 — Is `allow_fully_masked_rows` load-bearing?
Test: one CUDA `F.scaled_dot_product_attention` call in bf16 with an all-False mask row, with and
without the guard.
Why GPU: the guard exists because fused CUDA backends are believed to return NaN where the CPU
math backend returns zeros. On CPU the guard can be shown to be a no-op but never shown to be
necessary.

## P5 — bf16-full without Kahan really does stall
Test: 200 steps at `precision = 'bf16-full'` with `type = 'adamw'`, then the same with
`adamw8bitkahan`, same seed. Count parameters bit-identical to their initial value.
Why GPU: not strictly required, but the magnitude at real parameter scale is the thing worth
knowing. This validates the new warning rather than the code.

## P6 — Rollout memory at the shipped settings
Test: `[rollout] loss_points = 2` then 4 at `batch_size = 48`, watching peak VRAM.
Why GPU: the derivation says `loss_points` full-DiT backward graphs are live at once, doubled
with guidance. The doc's memory figures are estimates derived from the DiT's shape, never
measured.
