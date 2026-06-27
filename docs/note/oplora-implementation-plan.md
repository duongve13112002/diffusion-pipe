# OPLoRA implementation plan (LoRA, diffusion-pipe)

Date: 2026-06-27
Prompt: Plan, in detail, how to add OPLoRA (orthogonal-projection LoRA) to diffusion-pipe for LoRA training, before writing any code. Full-parameter fine-tuning is out of scope for this plan. Background and method comparison: [oplora-and-full-model-anti-forgetting.md](oplora-and-full-model-anti-forgetting.md).

## 1. Goal and scope

Add an opt-in OPLoRA mode that, after every optimizer step, projects each trainable LoRA factor into the orthogonal complement of the top-k singular subspace of its frozen base weight, so the base model's top-k singular triples stay unchanged.

In scope: LoRA adapters (peft `LoraConfig`) on the diffusion transformer, for both backends (`BasePipeline` and `ComfyPipeline`), single and multi-GPU, with and without block swap.

Out of scope: LoKr adapters, full-parameter fine-tuning, text-encoder training (already unsupported in this repo).

## 2. Method recap

For a frozen base weight `W` with thin SVD `W = U S V^T`, keep the top-k left/right singular vectors `U_k` (out x k) and `V_k` (in x k). peft stores a LoRA layer as `lora_A` (r x in, the "down") and `lora_B` (out x r, the "up"), with `dW = lora_B @ lora_A`. After each optimizer step, apply:

```
up   = lora_B.weight            # (out, r)
down = lora_A.weight            # (r, in)
up_proj   = up   - U_k @ (U_k.T @ up)      # remove top-k left components
down_proj = down - (down @ V_k) @ V_k.T    # remove top-k right components
```

Then `U_k.T @ up_proj = 0` and `down_proj @ V_k = 0`, so `W + lora_B' @ lora_A'` keeps `W`'s top-k singular triples exactly. The shapes match peft's convention, so the mapping is direct.

## 3. Where it hooks into the codebase

- Adapter is built at [train.py:523](../../train.py) via `model.configure_adapter(adapter_config)`, which calls `CommonPipeline.configure_adapter` in [models/base.py:216](../../models/base.py). That function discovers target `nn.Linear` modules from each model's `adapter_target_modules` and wraps them with peft. The base weights are frozen there; the LoRA factors become the trainable params.
- The optimizer step happens inside `model_engine.train_batch(iterator)` at [train.py:908](../../train.py) (DeepSpeed pipeline engine runs forward, backward, and step internally). Immediately after this call is the correct place to apply the projection, because OPLoRA must project after the weights have been updated.
- Block swap keeps `lora` params resident on GPU during the step ([utils/offloading.py:56](../../utils/offloading.py) and `weights_to_device` at line 116 skip `lora`), so the per-step projection only ever touches GPU-resident LoRA params plus the resident bases `U_k`, `V_k`.

## 4. Design

### 4.1 New module: `utils/oplora.py`

A small, dependency-light helper, holding an `OPLoRAProjector`:

- `OPLoRAProjector.build(root_module, rank, use_full_svd)`:
  - Iterate `root_module.named_modules()` and select peft LoRA layers (the modules that own `base_layer`, `lora_A`, `lora_B` for the active adapter). Confirm the exact peft attribute names against the installed peft version during implementation rather than assuming.
  - For each selected layer on this process: read the base weight `W = base_layer.weight`, dequantize and move it to the compute device for the SVD if needed, compute `U_k` and `V_k` for the configured `rank`, and store `(lora_A_param, lora_B_param, U_k, V_k)`. Keep `U_k`, `V_k` resident on the param's device in a small dtype-compatible buffer.
  - Compute the bases with torch built-ins rather than the GaLore projector. Verified: the repo's `approximate_svd` ([optimizers/projectors/approx_svd.py](../../optimizers/projectors/approx_svd.py)) needs the optional `fast_hadamard_transform` package (not in requirements, raises `NotImplementedError` otherwise), and `get_orthogonal_matrix` returns `Vh`, not `V`. So use `torch.linalg.svd(W, full_matrices=False)` for the full path and `torch.svd_lowrank(W, q=rank, niter=...)` for the randomized path (`torch.svd_lowrank` returns `V` directly as `in x rank` and needs no extra dependency). `U_k = U[:, :rank]` (out x rank); `V_k` is the `in x rank` right factor.
- `OPLoRAProjector.project()`:
  - For each stored entry, run the two in-place `.data` projections shown above, in `torch.no_grad()`. Match the in-place `.data` update style already used for gradient release in [optimizers/gradient_release.py](../../optimizers/gradient_release.py) so autograd does not flag it.
- The projector is built from `model_engine.module` after `deepspeed.initialize`, so each rank only sees and projects the LoRA layers on its own pipeline stage. This makes the projection rank-local with no cross-rank communication, and naturally handles pipeline sharding.

### 4.2 Configuration

Extend the `[adapter]` table parsed in [train.py](../../train.py) (`set_config_defaults` validates the adapter block around line 115):

- `oplora` (bool, default false): enable the projection. LoRA adapters only.
- `oplora_rank` (int, required when `oplora = true`): size of the protected top-k subspace per base weight.
- `oplora_full_svd` (bool, default false): use full SVD instead of randomized SVD when building the bases.

Validation rules (fail fast, no silent fallback):
- `oplora = true` with adapter `type != 'lora'` (e.g. `lokr`) raises a clear error.
- `oplora = true` without `oplora_rank` raises a clear error.
- `oplora_rank` larger than `min(out, in)` of any target weight raises a clear error naming the layer.

### 4.3 Lifecycle

- Build the projector once, after `deepspeed.initialize` and after any block-swap setup, before the training loop. Bases come from the frozen base weights, so nothing about OPLoRA needs to be checkpointed; on resume it is simply rebuilt (same behavior the reference OPLoRA documents).
- Call `projector.project()` right after [train.py:908](../../train.py) on every step, on every rank.
- Saving is unaffected: `save_adapter` writes the current `lora_A`/`lora_B`, which are already projected.

## 5. Key technical decisions and edge cases

- Quantized base weights (ComfyPipeline `fp8_scaled` path): `ComfyPipeline.dequantize` in [models/base.py:528](../../models/base.py) already dequantizes before training; ensure the SVD runs on a real floating-point base weight, dequantizing a copy at build time if a quantized tensor is still present.
- Block swap: bases may sit on CPU during training, but `U_k`/`V_k` are precomputed and resident, and LoRA params stay on GPU, so the per-step projection never has to wait on a swap. Build-time SVD may temporarily move a base weight to the compute device, then leave it as it was.
- Pipeline sharding: build from `model_engine.module` so each rank only handles its own layers. Confirm during implementation how the peft-wrapped layers appear inside the `ManualPipelineModule` (see [utils/pipeline.py](../../utils/pipeline.py) and each model's `to_layers`).
- dtype: compute the SVD in float32 for stability, then store `U_k`/`V_k` in a dtype that matches the LoRA param dtype (adapter dtype, set in `configure_adapter`). The projection itself is cheap and can be done in float32 then cast back.
- Fused vs split qkv: in peft each wrapped `nn.Linear` is projected independently, so a fused qkv `Linear` is handled correctly by SVD of its full weight. If any model packs multiple logical projections that must be treated separately, detect and log it at build time rather than silently mis-projecting. Verify per model from `adapter_target_modules`.
- Optimizer momentum is intentionally not projected (matches the reference). The per-step projection removes any drift the optimizer reintroduces into the protected subspace.
- Randomized SVD leakage means orthogonality is approximate, not exact (the reference reports a small residual). Offer `oplora_full_svd` for users who want the exact bases.

## 6. Multi-device and parallelism

diffusion-pipe runs hybrid data- and pipeline-parallel training through DeepSpeed (README "Parallelism"): `pipeline_stages` splits one model across GPUs, and the remaining GPUs replicate that model for data parallelism, coordinated through `model_engine.grid` (`get_data_parallel_world_size`, `get_data_parallel_rank`, `pp_group`; see [train.py:622](../../train.py) and [train.py:812](../../train.py)). OPLoRA must stay correct and consistent in every configuration the repo supports: single GPU, pure pipeline parallel, pure data parallel, and the hybrid mix.

- Single GPU: the degenerate case (one stage, one replica). No special handling.
- Pipeline parallel: build the projector from `model_engine.module` after `deepspeed.initialize`. Each rank materializes only its own stage's layers, so building from `model_engine.module` makes each rank compute and store `U_k`/`V_k` for exactly the LoRA layers it owns. Pipeline sharding is handled for free and no rank wastes SVD on layers it does not hold.
- Data parallel: the projection is rank-local and needs no collective. DeepSpeed keeps replicas in sync by all-reducing gradients, so after each synchronized optimizer step every replica of a given LoRA param holds identical weights. The frozen base weight is identical on every replica, so the bases `U_k`/`V_k` are identical, so each rank applies an identical projection and the replicas stay consistent. This is unlike Rank-1 EWC, which must all-reduce its Fisher direction.
- Determinism requirement (the one real multi-device pitfall): data-parallel consistency depends on every replica deriving the same `U_k`/`V_k` for the same layer. Full SVD (`torch.linalg.svd`) is deterministic and therefore automatically consistent. Randomized/approximate SVD draws random projections, which would differ per rank and silently desynchronize replicas. The build step must seed the randomized SVD deterministically per layer (a fixed base seed combined with the layer's stable name), or compute the bases on one rank and broadcast them. A test must cover this.
- ZeRO is not used here: the DeepSpeed config in [train.py:420](../../train.py) sets no `zero_optimization`, so pipeline training runs at ZeRO stage 0 and LoRA params are not sharded across ranks; each rank can project its params wholesale. If ZeRO param partitioning were ever enabled, the projector would need to gather each param before projecting. Add this as a guard/assert rather than silently mis-projecting.
- Block swap is only allowed with `pipeline_stages = 1` ([train.py:568](../../train.py)), and gradient release requires data-parallel world size 1 ([train.py:704](../../train.py)). OPLoRA does not change these constraints; it just has to coexist with each (build-time base access under block swap, post-step projection after the gradient-release hooks have run).

## 7. Implementation tasks

1. Add `utils/oplora.py` with `OPLoRAProjector` (`build`, `project`) and the validation helpers.
2. Verify peft LoRA attribute names and the `svd_projector`/`approx_svd` signatures against the installed versions; adapt the helper accordingly.
3. Parse and validate the new `[adapter]` keys in `set_config_defaults` and wherever the adapter block is read.
4. Build the projector after `deepspeed.initialize` from `model_engine.module`, and call `project()` after [train.py:908](../../train.py) on every rank.
5. Add the deterministic-seed path for randomized SVD (Section 6) and a ZeRO-partition guard.
6. Tests (see Section 8).
7. Docs (see Section 9).

## 8. Test plan (CPU-friendly)

The dev machine is CPU-only, so all of these run on CPU without DeepSpeed.

- Unit test `test/test_oplora.py`:
  - Random `W` (e.g. 32x24). Build `U_k`, `V_k` with full SVD. Random `up`, `down`. After projection assert `||U_k.T @ up_proj||` and `||down_proj @ V_k||` are at machine-eps level.
  - Assert the top-k singular values of `W + up_proj @ down_proj` equal those of `W` within tolerance.
- Smoke test `test/test_oplora_smoke.py`:
  - Build a tiny `nn.Linear` stack, wrap it with a peft `LoraConfig`, build `OPLoRAProjector` from the wrapped module, run a few steps of a real optimizer on random data, call `project()` each step, and assert the invariants from the unit test still hold and nothing raises. No DeepSpeed, no GPU.
- Data-parallel determinism test (covers Section 6): build the projector twice with randomized SVD on the same base weight, standing in for two data-parallel replicas, and assert the bases and the projected params are identical. This exercises the determinism requirement on CPU without needing multiple GPUs. Optionally run it as two `gloo` processes to mirror the real DP layout.
- Validation tests: `oplora = true` with `type = 'lokr'`, with missing `oplora_rank`, and with too-large `oplora_rank` each raise the expected error.

## 9. GPU verification script (for the maintainer's server)

A small standalone script `tools/test_oplora_gpu.py` (or a documented short config) that, on a GPU box:
- loads a small real model in LoRA mode with `oplora = true`, runs `--cache_only` then a handful of training steps, and checks the loss is finite and that for a sampled layer `U_k.T @ lora_B` stays near zero across steps.
- ideally also a 2-GPU smoke (pipeline_stages=1 data-parallel, and pipeline_stages=2) to confirm the projection keeps replicas consistent and pipeline stages each project their own layers.
- Intended to be run by the maintainer; it must not be part of the CPU test suite.

## 10. Docs to update once implemented

- `docs/supported_models.md` or a new `docs/anti_forgetting.md`: document the `oplora`, `oplora_rank`, `oplora_full_svd` options and the LoRA-only restriction.
- `examples/main_example.toml`: add commented `[adapter]` keys.
- `README.md`: one line under features / recent changes.
- `requirements.txt`: only if a new dependency is required (not expected; SVD is already available via torch and the existing projector utilities).

## 11. Open questions and risks

- Exact peft attribute names and whether multiple adapters can ever be active here (the repo uses a single `default` adapter). Confirm before coding.
- Whether any target module is a packed projection that needs splitting before SVD (per-model check).
- Performance on the largest models: storing `U_k`/`V_k` per target layer scales with layer count and rank. Measure on the GPU box and document the cost, mirroring the reference's note.
- Interaction with `gradient_release` (per-param optimizer hook in [train.py:741](../../train.py)): the step still completes before the post-`train_batch` projection, so it should compose, but verify.
