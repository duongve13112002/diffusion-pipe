# PyTorch SDPA kernel order and activation checkpointing

2026-11-01. Prompted by a MOSS-Image training run with `[model] attention = 'sdpa'` and activation
checkpointing that stopped in the first backward with `torch.utils.checkpoint.CheckpointError`,
while the same run with `attention = 'flash4'` trained.

## The error

```
CheckpointError: Recomputed values for the following tensors have different metadata than during the forward pass.
tensor at position 28: saved {'shape': [1, 24, 97, 1], float32}  recomputed {'shape': [1, 24, 97], float32}
tensor at position 30: saved {'shape': [], int64}                 recomputed {'shape': [2], uint64}
tensor at position 31: saved {'shape': [], int64}                 recomputed {'shape': [], uint64}
```

Each PyTorch SDPA kernel saves its own tensors for the backward. From the operator schemas
(`torch.ops.aten.<op>.default._schema`):

| Kernel | Saved for the backward |
| --- | --- |
| `_scaled_dot_product_cudnn_attention` | `logsumexp` `[B, H, L, 1]`, `philox_seed` and `philox_offset` (int64 scalars) |
| `_scaled_dot_product_flash_attention` | `logsumexp` `[B, H, L]`, `rng_state` (uint64 `[2]`), `unused` (uint64 scalar) |

So the forward ran cuDNN attention and its recomputation in the backward ran flash. `[1, 24, 97]`
is MOSS-Image's text refiner (`TextEmbedder`, 24 heads) on a 97-token caption, inside
`InitialLayer`, which `checkpointable_layers` checkpoints.

## Why the kernel changes between the two calls

Two pieces of code each work as intended on their own.

**PyTorch sets its kernel order lazily.** In `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`
(v2.9.0 to v2.15.0 and `main`), `select_sdp_backend` calls `priority_order()`, which on its first
call in the process (`priority_order_init_`) replaces the global order with cuDNN first when
`check_prefer_cudnn_attention()` holds:

- v2.9: cuDNN 9.9 or newer, compute capability 9.0 or 10.0; off with `TORCH_CUDNN_SDPA_PREFERRED=0`.
- v2.10 and later: runtime cuDNN newer than 9.15 (`cudnn_version > 91500`), compute capability 9.x
  or 10.x (minor 0 or 3 on CUDA < 13 builds); off with `TORCH_CUDNN_SDPA_DEPRIORITIZED=1`.
- v2.8 and earlier: no lazy change (`priority_order()` returns the context's order).

The Python getter `torch._C._get_sdp_priority_order` (`THPModule_sDPPriorityOrder` in
`torch/csrc/Module.cpp`) reads the context directly and does not trigger that first-call change.
The comment above `priority_order_init_` describes the getter as initialising the order; the code
does not do that.

**ComfyUI wraps SDPA in a priority context.** `comfy/ops.py` (same in ComfyUI 7cebca5 and 7c4390a)
runs every call with 128K query elements or more inside
`sdpa_kernel([FLASH, CUDNN, EFFICIENT, MATH], set_priority=True)`. `torch.nn.attention.sdpa_kernel`
reads the order on entry (`_cur_sdpa_kernel_backends(with_priority=True)`) and writes it back on exit.

When the first SDPA call of a process is one of ComfyUI's:

1. On entry, ComfyUI saves the default order (flash first) and sets its own (flash first).
2. The call picks a kernel. PyTorch's one-time change overwrites the order with cuDNN first, so
   this call runs cuDNN.
3. On exit, ComfyUI writes back the order it saved, flash first. PyTorch never changes it again.
4. The recomputation of that call in the backward runs flash: different saved tensors, and
   `torch.utils.checkpoint` stops the run.

Only the first call of each process is affected, which is why the logs of rank 1 and rank 5 show
different caption lengths (140 and 97): each rank's own first caption.

## When it shows up

All of these at once:

- PyTorch 2.9 or later, on a GPU and cuDNN where PyTorch prefers cuDNN (see above);
- a model whose attention runs through ComfyUI (`ernie_image`, `flux2`, `hunyuan_video_15`,
  `ideogram4`, `krea2`, `ltx2`, `minimax_h3`, `moss_image`, `qwen_image21`, `z_image`), or any other
  first SDPA call inside a priority context;
- activation checkpointing on, with that first call inside a checkpointed layer;
- nothing called SDPA earlier in the training process. Caching runs the VAE and text encoders in
  the training processes (`DatasetManager._handle_task`), so a run that still had media or captions
  to encode settled the order before training without anyone noticing. A run whose caches were
  complete (resumed, `--trust_cache`, an unchanged dataset) did not. This is why a commit that
  trained before can stop here with no change to the attention code.

`attention = 'flash3'` / `'flash4'` replace `torch.nn.functional.scaled_dot_product_attention`, so
the model's calls never reach PyTorch's kernel selection. Their fallback calls (float32, arbitrary
masks) still do, so they are exposed to the same problem.

## The fix

`utils/attention.py:settle_sdpa_priority`, called by `train.py` right after
`torch.cuda.set_device`, before caching and before the model loads. It makes one small SDPA call
outside any context, so PyTorch's one-time change happens there, then:

- `[model] sdpa_prefer_cudnn = false` (default): writes back the order from before the change
  (flash, efficient, math, cuDNN). That is the order every call after the first already ran with,
  so training is unchanged apart from the first call no longer differing.
- `[model] sdpa_prefer_cudnn = true`: keeps PyTorch's order (cuDNN first where PyTorch prefers it).
  Calls outside ComfyUI's context (smaller inputs, non-ComfyUI models) then run cuDNN.

A side effect: in a training run that still has media or captions to cache, the VAE and text
encoders now encode with the settled order (flash first by default) where cuDNN may have run
before. The cache keys are unchanged, so nothing is re-encoded; new entries can differ from
cuDNN-encoded ones in the last bits. `--cache_only` runs do not settle the order.

Either way the order is settled before any model call, so ComfyUI's context saves and restores the
real order and a call and its recomputation pick the same kernel. Calls inside ComfyUI's context
keep ComfyUI's order (flash first) in both settings.

`sdpa_prefer_cudnn = false` was chosen as the default because it is what the runs that worked were
training with, and because cuDNN attention's backward gave NaN gradients on Iris (H200, torch 2.12,
cuDNN 9.20; see `docs/supported_models.md`, `allow_cudnn_attention`).

The order is read and written with `torch._C._get_sdp_priority_order` and
`torch._C._set_sdp_priority_order`, the same private functions `torch.nn.attention.sdpa_kernel`
uses. Where they do not exist (PyTorch without priority orders) or CUDA is not available, the
function does nothing.

Alternatives considered:

- Disable cuDNN SDPA, as Iris does (`torch.backends.cuda.enable_cudnn_sdp(False)`). Does not work
  for ComfyUI models: `sdpa_kernel` enables every backend in its list on entry, cuDNN included.
- Set `TORCH_CUDNN_SDPA_DEPRIORITIZED=1` from code. The variable's name and meaning changed between
  PyTorch releases (v2.9 reads `TORCH_CUDNN_SDPA_PREFERRED`), and it is read once, at the first
  call, so it only works if set before anything calls SDPA. Still useful as a launch-time
  workaround on PyTorch 2.10 and later.

## The other PyTorch SDPA settings

README "PyTorch SDPA backends and their settings" lists what each backend (flash, cuDNN,
efficient, math) accepts and on which GPUs, the global switches (`torch.backends.cuda.enable_*_sdp`),
`sdpa_kernel`, the environment variables per PyTorch version, PyTorch's FA3 / FA4 prototype
(`torch.nn.attention.activate_flash_attention_impl`, 2.10+ / 2.11+), and the keys of this repo
that choose among them. Note that cuDNN attention is refused while
`torch.use_deterministic_algorithms(True)` is on, so deterministic runs never hit this problem.

## Checking it on a GPU

`python tools/check_sdpa_priority.py` runs the problem in fresh processes (the change is once per
process): a layer whose first SDPA call goes through `comfy.ops.scaled_dot_product_attention` under
`torch.utils.checkpoint`, unsettled and settled both ways. It reports the kernels of the forward and
the recomputation from the profiler. The unsettled scenario fails only where PyTorch prefers cuDNN;
the settled ones must pass everywhere. `test/test_attention.py` reproduces the mechanism on CPU with
a stand-in for PyTorch's selection state driven through the real `sdpa_kernel`.

## Sources

- PyTorch `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` at v2.8.0, v2.9.0, v2.10.0,
  v2.11.0 to v2.15.0 and `main` (`priority_order`, `check_prefer_cudnn_attention`,
  `select_sdp_backend`): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
- PyTorch `torch/csrc/Module.cpp` (`THPModule_sDPPriorityOrder`).
- `torch/nn/attention/__init__.py` (`sdpa_kernel`, `_sdpa_kernel`, `_cur_sdpa_kernel_backends`), torch 2.15.0.
- ComfyUI `comfy/ops.py` (`scaled_dot_product_attention`, `SDPA_BACKEND_PRIORITY`) at 7cebca5 and 7c4390a.
