# OPLoRA: orthogonal projection LoRA

OPLoRA reduces catastrophic forgetting during LoRA fine-tuning. It projects every LoRA update onto the orthogonal complement of the base weight's top-k singular subspace, so the base model's most important directions are left unchanged while the adapter still learns the new task in the remaining subspace.

For a frozen base weight `W = U S V^T`, OPLoRA keeps the top-k left and right singular vectors `U_k` and `V_k`. After each optimizer step it corrects the two LoRA factors (peft stores them as `lora_A`, shape `(rank, in)`, and `lora_B`, shape `(out, rank)`):

```
lora_B <- lora_B - U_k (U_k^T lora_B)
lora_A <- lora_A - (lora_A V_k) V_k^T
```

After this, `U_k^T lora_B = 0` and `lora_A V_k = 0`, so the effective weight `W + lora_B lora_A` keeps `W`'s top-k singular triples exactly. This is a hard guarantee about each weight's top-k subspace, not a guarantee that the whole model output is unchanged: the adapter is still free to change behaviour in the orthogonal complement, which is where it learns.

There is no teacher model and no extra forward pass. The bases `U_k`, `V_k` are computed once at startup by an SVD of each base weight; the per-step projection is a few small matmuls on the LoRA factors.

This is based on "OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning" (arXiv:2510.13003).

## Scope

- LoRA adapters only. It operates on the LoRA factors, so it is rejected for LoKr and for full fine-tuning.
- Works for both the Diffusers-backed and ComfyUI-backed models, since it acts at the peft adapter level.
- Works on single GPU, pipeline parallel, data parallel, and the hybrid mix. The projection is applied per rank to the LoRA layers that rank owns, and needs no extra cross-GPU communication.

## Configuration

Add these to the `[adapter]` table (only when `type = 'lora'`):

```toml
[adapter]
type = 'lora'
rank = 32
dtype = 'bfloat16'
oplora = true
oplora_rank = 16
# optional:
oplora_full_svd = false
oplora_seed = 0
```

- `oplora` (bool, default `false`): enable the projection.
- `oplora_rank` (int, required when `oplora = true`): size of the protected top-k singular subspace per base weight. It is independent of the LoRA `rank`. Larger preserves more base knowledge but leaves the adapter less room to learn. It must not exceed the smallest dimension of any target weight.
- `oplora_full_svd` (bool, default `false`): use exact full SVD instead of the default fast randomized SVD when building the bases. Full SVD is slower at startup but exact.
- `oplora_seed` (int, default `0`): seed for the randomized SVD. It is kept deterministic so that data-parallel replicas build identical bases and stay consistent. You normally do not need to change it.

## Cost and notes

- Startup: one SVD per target weight. The default randomized SVD keeps this fast; `oplora_full_svd = true` is exact but slower on large models.
- Memory: the bases `U_k` (`out x rank`) and `V_k` (`in x rank`) are kept resident per target layer. Per layer this is small, but it scales with the number of target layers and `oplora_rank`, so budget for it on very large models.
- The randomized SVD leaves a tiny residual, so orthogonality is approximate rather than exact; use `oplora_full_svd = true` if you want the exact bases.
- The projection is computed in float32 but written back in the LoRA dtype. With `bfloat16` LoRA weights the leftover overlap is at bf16 rounding level rather than float32 noise, which is expected and harmless.
- On resume the bases are rebuilt from the (unchanged) base weights, so nothing OPLoRA-specific needs to be checkpointed.

## Verifying

- CPU tests: `python -m pytest test/test_oplora.py`.
- GPU sanity check (run on a CUDA machine): `python tools/test_oplora_gpu.py`, or `torchrun --nproc_per_node=2 tools/test_oplora_gpu.py` to also check data-parallel consistency.
