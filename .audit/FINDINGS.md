# Anima Refiner — forensic audit, final state

Baseline `fd00a5e` → branch `feat/anima-refiner`. Audited at `d895ad8`; fixes land in
`80a6497`, `ace00c7`, `b241a02`, `df5d670`.

**Verdict: SAFE WITH CAVEATS.** Every defect that could be demonstrated is fixed and carries a
regression test that fails if the fix is reverted. No known unfixed defect remains. What cannot be
established on this machine is in section 4 — that is a property of the box, not outstanding work.

**Suite: 526 passed, 1 skipped** (490 at audit start; 36 tests added).

This file supersedes every earlier draft. Previous versions carried stale "what remains" sections
from a mid-session interruption. They are deleted rather than annotated: a record that contradicts
itself is worse than no record.

---

## 1. Fixed defects

### CRITICAL — C-1  Every ComfyUI-backed model crashed before training started

`utils/dataset.py` called `model.vae_cache_key()` unconditionally, but the method was defined only
on `BasePipeline`. `ComfyPipeline` inherits `CommonPipeline`, not `BasePipeline`, so **8 of 23
model types** raised `AttributeError` at `train.py:442`: `z_image`, `hunyuan_video_15`, `flux2`,
`ernie_image`, `ltx2`, `ideogram4`, `krea2`, `minimax_h3`.

Proven by AST rather than by reading. Nothing to do with Anima — collateral from the refiner's
cache-identity work, introduced by `3b71586`, while the correct defensive pattern
(`getattr(self.model, 'text_encoder_cache_key', lambda _i: '')`) already existed 96 lines later
from the earlier `758e449`.

Why no test caught it: **no test in the suite constructs the top-level `Dataset`.** Fourteen build
`DirectoryDataset` directly and skip the line entirely.

- **Fix:** the four cache-identity members moved to `CommonPipeline`.
- **Tests:** `test_dataset_smoke.py::TestVaeCacheKeyIsDeclaredPerModel` — the contract on all
  three classes, plus the call site itself.

### HIGH — H-1  ZeRO gave the rollout's unconditional branch N times its gradient

DeepSpeed does **not** apply the `1/gradient_accumulation_steps` scaling inside `backward()`.
`engine.py:2490` computes `gas_scaled_loss` and the line is commented `# Used only for return
value`; the real division is a hook registered on the output of `engine.forward()`
(`engine.py:2237-2243` → `_backward_prologue_per_tensor`, `engine.py:2362-2366`).

`tools/distill_refiner.py` calls the bare `refiner(...)` for the unconditional branch on purpose —
a second engine forward would build a second backward-hook manager — and so bypassed the only
place the scaling happens.

**Measured on a real DeepSpeed 0.18.4 engine** (gloo, ZeRO-1, `grad_accum=4`,
`.audit/exp_zero_scaling.py`): engine path **1.0×** a single un-accumulated batch, bypassing path
**4.0×**. With the fix, 1.0×.

**Blast radius: no shipped config reached it** — all four use `ddp` with `guidance_scale = 0.0`.
Trigger is `zero1|zero2` + `grad_accum > 1` + `[rollout] loss_weight > 0` + `guidance_scale > 1`.

- **Fix:** `strategy.scale_side_branch()` — a no-op under DDP (which divides the whole loss once),
  a `/N` hook under ZeRO. States the invariant once instead of at each call site.
- **Tests:** `TestSideBranchAccumulationScaling` (4, value-asserted so an inversion fails), plus
  `tools/test_zero_side_branch_multirank.py` for the multi-rank case.

### MEDIUM

| ID | Defect | Fix | Test |
|---|---|---|---|
| M-1 | A cache that was already complete took the early return in `_map_and_cache` and never recorded a manifest, so the identity check stayed **permanently inert** on exactly the installs it was written for. `utils/cache.py` claimed it "gets a manifest the next time it is written in full" — false. | `write_manifest()` before the early return | `test_an_already_complete_cache_still_acquires_a_manifest` |
| M-2 | The unconditional text-embedding cache was never passed `identity`. For `anima` (empty fingerprint key by design) its fingerprint is constant across all runs, so swapping `llm_path` rebuilt the conditional embeddings while the **unconditional** one — used by every CFG-dropped sample — came back from the old encoder. | thread `identity=` through | `test_the_unconditional_cache_records_the_same_identity` |
| M-3 | Stable checkpoint filenames written with `shutil.copy2`, which truncates and streams — the exact hazard `_save_file_atomically` exists to prevent, applied to the tagged file but not to the name every shipped config points at. | `_copy_atomically()` (tmp + `os.replace`) | `test_the_stable_name_is_never_left_truncated` |
| M-4 | Nothing recorded which step the weights came from, so weights and optimizer moments from different saves paired up silently and resumed an already-annealed schedule against newer weights. | step in safetensors metadata, checked on resume | `TestCheckpointHalvesStayTogether` (5) |
| M-5 | `save_full_model` rounded the refiner to `dtype` (bf16) while `save_refiner` deliberately kept fp32 — and both files are valid `resume_from` sources. | write the refiner fp32 in both | `TestFullModelKeepsTheRefinerInFp32` |
| M-6 | Upgrading past the narrowed latent fingerprint silently rebuilds **every** existing latent cache, for every model in the repo. Loud, but undocumented. | documented in `README.md` and `examples/dataset.toml`, with the `keep_latent_cache` escape for the upgrade run | doc |
| M-7 | Checkpoint retention pruned the **current** run's checkpoints. Tags order by number and numbers only increase within one run, so a second run into a populated directory deleted its own fresh saves and kept the old run's. | this run's tags sort after every tag it did not write | `test_a_second_run_prunes_the_old_checkpoints_not_its_own` and `test_one_long_run_still_drops_its_own_early_checkpoints` |
| M-8 | ZeRO did not checkpoint fp32 master weights or the dynamic loss scale, so a resumed `bf16-full` / `fp16-mixed` ZeRO run restarted its masters from bit16 values. | persist and restore both | `TestZeROMasterWeightsSurviveAResume` |

### LOW

| ID | Defect | Fix |
|---|---|---|
| L-1 | Two more tests that **could not fail**: one parametrisation ran zero assertions behind `if not env:`; the probe-seed test matched a substring that also appears where the global seed is set, so a rank offset on the probe left it green — the one invariant whose violation makes every rank optimise a different objective. | both now fail when inverted (verified by mutation) |
| L-2 | `keep_last_n_checkpoints` validated in the distiller, silently ignored in `train.py`. | validated in `set_config_defaults` |
| L-3 | The corpus caption-settings warning fired only when **none** of five keys was restated, so restating one and forgetting `prefix_tag_caption` was silent. | a second branch warns specifically about the missing marker |
| L-4 | `bf16-full` under DDP gives bf16 parameters **and** bf16 Adam moments with no master copy — the same underflow `fp16-full` is refused for, with three fewer mantissa bits — while the docs called Kahan "the safer pick" rather than the thing that makes the mode work. | startup warning, plus docs in three configs and the README |
| L-5 | Probe `num_heads` hardcoded to 16 while `get_dit_config` derives it from the checkpoint (16/40/20 by width); the sampler hardcoded 16 latent channels. | both read the model |
| L-6 | Two stale doc claims: `design-notes.md` argued for `num_queries = 64` (code: `2 * head_dim`); `denoising-rollout.md` said only the summed loss is logged (per-term logging exists). | corrected |

---

## 2. Reported but REFUTED or WITHDRAWN — do not re-file

These matter as much as the fixes. Three were plausible source-level derivations that measurement
or a second reading killed.

- **`relational_loss` NaNs at `batch_size > 25`** (reported HIGH). **Refuted by experiment.**
  `torch.cdist(x,x)` does switch implementation above n=25 and the diagonal stops being exactly 0
  (5.5e-3) — but gradients are **finite** at unit scale, at the `/512` scale `padded_mean` actually
  produces, and even with the diagonal **forced to exactly 0.0**. No discontinuity across n=25/26.
  Both shipped 4-GPU configs use batch 48 and are fine.
- **`models/base.py:145` `extractfile(str(spec[1]))`** — filed by me as an unfixed instance of the
  "`as_posix` for archive members" lesson. **Withdrawn, not a bug.** Tar specs are built at
  `utils/dataset.py:1003` as `(str(file), name)` with `name` from `tar_f.getnames()` — already a
  forward-slashed member name, already a `str`. `str()` is a no-op on a correct value. The genuine
  instance was `dataset.py:1161`, where `image_file` really is a `Path`, and it already uses
  `.as_posix()`. I matched the shape of an expression without checking what flows into it, which is
  exactly what that lesson warns against.
- **`RotaryEmbedding.inv_freq` uninitialised** — a non-persistent buffer no checkpoint restores,
  not covered by `init_weights`. **Cleared.** `accelerate.init_empty_weights` defaults
  `include_buffers=False` (resolved from `ACCELERATE_INIT_INCLUDE_BUFFERS`), and the project
  already documents this *and* guards it **by construction** in `tools/check_vendored_apis.py`,
  which builds a module with a non-persistent buffer and asserts the property rather than reading a
  signature default.
- **ZeRO accumulation boundary** (`50730bc`). Cleared against DeepSpeed 0.18.4's own source: the
  manual override wins (`engine.py:2529-2533`), `micro_steps` advances only in `step()` (`:2762`),
  and the local pattern matches deepspeed's own docstring. No double clipping, no double scheduler
  step.
- **Caption/embedding order pairing.** Cleared — `dataset.py` shuffles (index, caption) *pairs*,
  and the premise was verified: `flatten_captions` iterates metadata order.
- **`create_lr_scheduler` `T_max` quirk.** Byte-identical to `fd00a5e`. Pre-existing, documented,
  deliberately left. Not a regression.
- **Config keys accepted but ignored.** None: all 23/33/29/39 keys across the four shipped
  `distill*.toml` are read by the code.

---

## 3. Not defects — deliberate, with reasons

- **Relational term's `(B-1)/B` factor.** Both distance matrices have an exactly-zero diagonal, so
  `smooth_l1(0,0)=0` adds nothing to the sum but still counts toward the `mean` divisor.
  **Measured at ~12%** across batch 8 to 48. It is a constant at fixed batch size, so it changes
  gradient *magnitude*, never direction, and `relational_loss_weight` absorbs exactly that.
  Reducing over the strict upper triangle is a two-line change if a measurement ever justifies it.
- **Rollout `guidance_scale = 0` and 256px resolution.** Design choices with a stated cost, now
  covered in the quality-first recipe in `docs/anima_refiner/training.md` — what each buys and when
  to raise it.

---

## 4. Hardware limits — verified, not assumed

`torch.cuda.is_available()` is `False`, `device_count()` is `0`, `torch.version.cuda` is `None`
(a `+cpu` build, which could not drive a GPU even if one were attached), and `nvidia-smi` is
absent. Validation on real CUDA hardware, and a real training run, **cannot be performed in this
environment by any means.**

That claim is made carefully, because **three times this session a "needs a GPU" limit turned out
to be softer than documented**, and each time it yielded real coverage:

1. `deepspeed.initialize` was said to need a C++ toolchain. It runs here once the shm op is marked
   incompatible — the repo's own test already did this. That freed three findings.
2. `GradScaler` was said to self-disable without CUDA. True of `GradScaler('cuda')`;
   `GradScaler('cpu')` is fully functional, so the fp16 scaler branch is now driven for real
   instead of against a hand-written stand-in.
3. fp16 arithmetic was assumed untestable. `torch.autocast('cpu', dtype=torch.float16)` is real, so
   fp16 numbers now go through the actual `ContextRefiner`.

**What genuinely remains GPU-only, and why none of it is an open defect:**

| Item | Why it is not an open defect |
|---|---|
| fp16 **kernel** numerics | The control flow *and* the arithmetic through our own modules are covered. What is left is whether a given CUDA kernel overflows — a property of the kernel. |
| Fused-SDPA NaN premise | **Not a risk.** `allow_fully_masked_rows` is a proven no-op for any row that has a key, and the degenerate row returns exactly the zero the masked softmax intended (`k_proj`/`v_proj` carry no bias). fp16 on that row is now tested and finite. If the premise is false the guard is merely *unnecessary*, never harmful — an unconfirmed rationale, not unverified behaviour. |
| Rollout peak VRAM | A capacity-planning number. The scaling law itself is tested: `loss_points` provably bounds the student forwards. |
| Output quality | The research hypothesis. **No training run, no multi-GPU run and no image has ever been produced from this branch** — the branch's own docs say so. |

---

## 5. GPU validation plan — smallest experiments, highest value first

**P1 — Does the ZeRO side-branch fix hold across real ranks?** (Single-rank half proven on CPU.)
Two GPUs, `zero1`, `grad_accum=4`, rollout on with `guidance_scale=5.0`, ~50 steps; repeat with
`ddp`, same seed and data.
*Expect:* the two loss curves track each other. *Failure:* ZeRO falls faster early then plateaus
higher, or its grad norm is systematically larger. **Run this before committing serious compute.**

**P2 — ZeRO resume restores what it claims.** (Master-weight half proven on CPU.)
`deepspeed --num_gpus=2 tools/test_zero_resume_gpu.py`, then a 200-step run with a mid-run restart.
*Expect:* loss continues smoothly across the restart.

**P3 and P4 — both settled by one script, no checkpoints or dataset needed:**

```
python tools/test_precision_gpu.py
```

Runs in seconds on any single CUDA device. It answers:

- **P4 — is `allow_fully_masked_rows` load-bearing?** Calls `scaled_dot_product_attention`
  directly with an all-False query row on each backend CUDA offers (flash, mem-efficient, cudnn,
  math) in both bf16 and fp16, guarded and unguarded. A backend that returns NaN unguarded proves
  the guard load-bearing; one that returns finite values proves it merely unnecessary. The guard
  is correct either way — this settles *which*, which is currently an unconfirmed rationale.
- **P3 — does fp16-mixed behave on real kernels?** 20 real steps through a small `ContextRefiner`
  under `autocast('cuda', float16)` with a live `GradScaler`, reporting the loss-scale trajectory
  and how many steps were skipped. *Failure:* the scale collapsing toward 1, or most steps skipped.

Verified on CPU as far as CPU allows: it imports, exits cleanly with a clear message when there is
no device, the tensor shapes are right, and the guard is confirmed a no-op for rows that have keys.
What it cannot do here is exercise a fused CUDA backend — which is the question it exists to ask.

**P5 — Rollout memory at the shipped settings.** `loss_points` 2 then 4 at `batch_size = 48`,
watching peak VRAM. The doc's figures are derived from the DiT's shape, never measured.

**P6 — Sampling quality.** The actual research question.

---

## 6. Method notes worth keeping

- **A green suite proved nothing about the two most serious findings.** 490 tests passed while C-1
  crashed 8 models and H-1 corrupted gradients. C-1 was invisible because no test constructs the
  top-level `Dataset`; H-1 because the ZeRO tests used a fake engine, or computed the loss purely
  from the engine's own output.
- **Mocks cannot test a state machine.** The DeepSpeed and GradScaler branches were both "covered"
  by hand-written stand-ins that could not have the property being asserted. Both now drive the
  real thing.
- **Two of my own fixes were wrong on the first attempt**, and both were caught before shipping:
  reaching through `pipeline.transformer` for the sampler's channel count broke six tests, and
  "never prune a tag this process wrote" would have stopped pruning entirely, because in a single
  run *every* tag is one this process wrote. A fix deserves the same evidence as a bug report.
- **Three reported findings were refuted or withdrawn.** Optimising for finding count would have
  shipped all three as changes to code that was already correct.
