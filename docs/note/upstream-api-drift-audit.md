# Auditing code that depends on upstream internals

- Date: 2026-06-29, extended 2026-07-29
- Prompt: A real training run of Ideogram4 crashed with
  `TypeError: Ideogram4EmbedScalar.forward() missing 1 required positional argument: 'dtype'`
  after the ComfyUI submodule was bumped. The crash was not in our code per se — it was a
  signature drift in a vendored submodule that our code did not follow. ComfyUI is only the most
  active example: **every** submodule in `submodules/` is a third-party library that one or more
  models import from and call directly, so the same drift can come from any of them. This note
  records why the bug class exists and the procedure to re-audit after **any** submodule pin changes.
- Extension (2026-07-29): the same bug class hit twice more, this time from **pip dependencies**
  rather than submodules — bitsandbytes 0.50 broke `optimizers/adamw_8bit.py` with
  `KeyError: 'percentile_clipping'`, and torch 2.13 broke `utils/reduction.py` by removing
  `torch._namedtensor_internals.check_serializing_named_tensor`. The trigger is not "a submodule
  moved", it is "we copied or subclassed someone else's internals and they changed". This note now
  covers both.

## Why this class of bug exists

The models do not always run the upstream model's own `forward()`. To split a model across
pipeline-parallel stages (or to strip in-place ops, handle masks, etc.), a model file imports
classes and functions from its backing submodule and calls them directly — re-implementing or
borrowing pieces of the upstream forward pass. Concretely this happens three ways:

- **Direct imports** of submodule functions/classes used inside our forward (e.g. ComfyUI's
  `timestep_embedding`, `rms_norm`, `ModulationOut`; a backbone block class).
- **Grabbed leaf submodules** pulled off the loaded model and called inside the `to_layers()`
  wrappers (`InitialLayer` / `TransformerLayer` / `FinalLayer`).
- **Monkey-patches / overrides** that replace an upstream method (ltx2 replaces
  `BasicAVTransformerBlock.forward`; `base.py` replaces `ClipTokenWeightEncoder.encode_token_weights`).

When the submodule pin moves and an imported/called symbol gains a required argument, changes its
return arity, renames an attribute, or makes an optional argument required, our code keeps calling
the old shape and breaks at runtime — usually deep in a forward pass, often for just one model.

## The core audit principle

> Our code copies or calls into the submodule's internals, so **the way we call a symbol must match
> the way the submodule's current code expects to be called.** Compare the two; any divergence is a
> suspected bug.

That is exactly how the Ideogram4 bug reads: ComfyUI's native forward calls
`self.t_embedding(t, dtype=x.dtype)`, but our `InitialLayer.forward` called `self.t_embedding(t)`.

## Which submodule backs which model

When a submodule changes, audit the models that depend on it (and only those). The mapping comes
from the `sys.path.insert(..., 'submodules/<X>')` lines plus the `comfy` imports:

| Submodule | Dependent code |
| --- | --- |
| `ComfyUI` | `models/base.py` and the `ComfyPipeline` models: `z_image`, `ltx2`, `hunyuan_video_15`, `flux2`, `ernie_image`, `krea2`, `ideogram4`. Also `chroma` (uses `comfy.ldm.flux.layers.timestep_embedding`). |
| `flow` | `models/chroma.py` |
| `Cosmos` | `models/cosmos.py` |
| `HiDream` | `models/hidream.py` |
| `HunyuanImage-2.1` | `models/hunyuan_image.py` |
| `HunyuanVideo` | `models/hunyuan_video.py`, `utils/patches.py` |
| `LTX_Video` | `models/ltx_video.py` |
| `Lumina_2` | `models/lumina_2.py` |
| `OmniGen2` | `models/omnigen2.py` |

Re-derive this map (don't trust it blindly) with:

```
grep -rnE "sys\.path.*submodules/|^(from|import) .*comfy" models/ utils/ train.py
```

## Which pip dependency backs which file

Submodules are the obvious case because they are pinned in-tree, but `requirements.txt` is
mostly unpinned, so a plain `pip install -U` moves these the same way. Only the files below
copy or subclass a dependency's **internals**; everything else uses public APIs and is not part
of this audit.

| Dependency | Dependent code | What it borrows |
| --- | --- | --- |
| `torch` | `utils/reduction.py` | A copy of `torch/multiprocessing/reductions.py` with `multiprocessing` swapped for the third-party `multiprocess` library, so HF Datasets workers can pass CUDA tensors over queues. Reaches into private symbols (`torch._utils._rebuild_tensor`, `torch._storage_classes`, `torch._nested_view_from_buffer_copy`, ...). |
| `torch` | `utils/patches.py` | `torch._inductor.runtime.triton_heuristics`, plus it registers the reductions above. |
| `bitsandbytes` | `optimizers/adamw_8bit.py` | A re-implementation of `Optimizer2State.update_step` that adds Kahan summation, so it depends on the exact keys of `get_config()` and on the positional order of `functional.optimizer_update_32bit` / `optimizer_update_8bit_blockwise`. |
| `deepspeed` | `tools/distill_refiner.py` (`save_training_state`, `load_training_state`, `ZeroStrategy`) | No import of a private symbol, but a hard dependency on a **behaviour**: `deepspeed.initialize` mutates the client optimizer in place, replacing each param group's parameter list with one flat rank-local fp32 partition (`deepspeed/runtime/zero/stage_1_and_2.py`, the `param_group['params'] = [self.single_partition_of_fp32_groups[i]]` assignment). The whole per-rank shard design follows from that. Also depends on `set_gradient_accumulation_boundary` driving the accumulation window, and on `get_global_grad_norm` returning `None` before the first boundary. |
| `transformers` | `models/cosmos_predict2.py` (`_compute_text_embeddings`, `_load_llm_from_single_file`, the `llm_path` branches), `tools/distill_refiner.py` (`build_teacher`, `build_student`) | Not a vendored copy, but not a stable contract either: concrete attribute paths (`full_model.model.language_model`, `llm_config.text_config.hidden_size`), direct indexing of the `output_hidden_states` tuple (`llm_hidden_layer` selects a layer by position), and architecture-specific classes (`Qwen3Config`, `Qwen3ForCausalLM`, `AutoModelForImageTextToText`). A wrong hidden-states length or ordering picks the wrong layer instead of raising. |
| `accelerate` | every model using `init_empty_weights`, especially `models/text_refiner.py` | `ContextRefiner.init_weights` materialises every parameter by hand because they arrive on the meta device, and deliberately does **not** materialise `RotaryEmbedding.inv_freq`, because `init_empty_weights` leaves buffers real. That default is resolved from `ACCELERATE_INIT_INCLUDE_BUFFERS` rather than written in the signature, so it is checked by construction. Also `set_module_tensor_to_device(device=, dtype=, value=)`. |

Re-derive this map with:

```
grep -rnE "^from torch\._|^import torch\._|Copied from|bitsandbytes\.optim\." models/ utils/ optimizers/ train.py
```

The automated guard for these is:

```
python tools/check_vendored_apis.py
```

It checks every private torch symbol `utils/reduction.py` needs, that the module still imports,
and the bitsandbytes surface `adamw_8bit.py` re-implements (including the positional order of the
two update kernels). It also covers the `transformers` and `accelerate` surface the anima_refiner
text encoder depends on, including two behavioural checks that no import test would catch: that
`output_hidden_states` still yields `num_hidden_layers + 1` tensors ending in `last_hidden_state`,
and that `init_empty_weights` still leaves buffers off the meta device. Each dependency's half is
skipped rather than failed when it is not installed, so the torch half still runs anywhere.

### Most of the deepspeed behaviour dependency *can* be checked on a CPU box

`tools/check_vendored_apis.py` covers torch and bitsandbytes by importing them and comparing
signatures. It cannot cover the deepspeed row above, because the claim is about what
`deepspeed.initialize` *does* rather than what it accepts.

This section used to say that could not be checked without a GPU, on the grounds that
`initialize` JIT-builds `deepspeed_shm_comm` and needs a compiler. That is only true of the
default path. Marking the shm op incompatible before calling `initialize` is the supported way to
skip it:

```python
for name in list(deepspeed.ops.__compatible_ops__):
    if 'shm' in name.lower():
        deepspeed.ops.__compatible_ops__[name] = False
```

With that, a real engine runs on CPU over gloo, at one rank or two. The claim was load-bearing in
the wrong direction: it is what made three separate checks get deferred as "needs hardware" when
they did not. What actually needs a GPU is narrower than it looked — anything about CUDA kernels,
fused attention numerics, or real device memory.

Two CPU checks now exist and are worth running after a deepspeed upgrade:

```
PYTHONPATH=test/childenv python tools/test_zero_side_branch_multirank.py
```

Two gloo ranks and a real ZeRO engine. It asserts the property
`DeepSpeedZeROStrategy.scale_side_branch` exists for: deepspeed applies its
`1/gradient_accumulation_steps` scaling through a hook on the output of its *own* forward and
never inside `backward()`, so a forward that bypasses the engine bypasses the scaling and
contributes `gradient_accumulation_steps` times its intended gradient. If a future deepspeed moves
the division into `backward()`, the "bypassing path is NOT scaled" check fails and names it —
at which point the hook would be double-scaling and must be removed.

`test/test_distill_refiner.py::TestZeROAccumulationBoundaryForReal` is the other, covering stages
1 and 2 against a real engine inside the normal suite.

`tools/test_zero_resume_gpu.py` remains for the genuinely device-bound part. Run it on a two-rank
GPU box after any deepspeed upgrade:

```
deepspeed --num_gpus=2 tools/test_zero_resume_gpu.py
```

Its first assertion is the premise itself — that each param group holds exactly one tensor after
`initialize`, and that the ranks' partitions differ. If deepspeed ever stops partitioning the
client optimizer this way, that assertion fails first and names the reason, rather than the
failure surfacing as a silently unloadable checkpoint hours into a run. Which is how it surfaced
the first time.

## Procedure (run on every submodule pin change or dependency upgrade)

1. Find which submodule commits changed: `git submodule status` and `git diff <old>..<new>` on the
   submodule gitlink. Only audit models that depend on a changed submodule.
2. For each dependent model, find every cross-boundary call and check it against the submodule's
   **current** code:
   - Direct imports: `grep -nE "^(from|import) " models/<m>.py` (filtered to the submodule), then
     verify each imported function/class signature.
   - Grabbed leaf submodules called in the `to_layers()` wrappers: locate the matching upstream
     class and diff our leaf calls against how the upstream `forward()` calls the same leaves.
     Watch for new required args, `out_dtype=`, `transformer_options=`, and changes to **return
     arity** that we unpack positionally.
   - Monkey-patched/overridden methods: the replacement's signature must match the method it
     replaces, and every attribute/method it calls must still exist with the same shape.
   - Loading / adapter / VAE / text-encode APIs the model uses (e.g. for ComfyUI:
     `comfy.sd.load_clip`, `load_diffusion_model`, `load_checkpoint_guess_config`,
     `load_lora_for_models`, `comfy.sd.VAE.__init__`, `comfy.utils.load_torch_file`). This covers
     the full-fine-tune, LoRA-merge, and caching paths, not just LoRA forward.
3. For ComfyUI, run the automated guard on a machine where the submodule imports (training box; the
   CPU dev box is missing runtime deps such as `comfy_aimdo`):

   ```
   python tools/check_comfy_signatures.py
   ```

   It encodes the ComfyUI contracts verified by hand and exits non-zero on drift. The other
   submodules back one model each and are lower-churn, so they use the manual import-diff above; add
   an equivalent guard script for one if it ever becomes a frequent source of drift.
4. Fix any divergence by making our call match the submodule's current native call. Note which
   models actually needed a change so the fix can be GPU-verified on the affected model. If the
   upstream change was intentional, update the guard script (for ComfyUI) in the same commit.

## Cross-cutting couplings worth remembering

- **`transformer_options` is always omitted** by our ComfyUI block/refiner calls and is optional
  (default `{}`) in every ComfyUI block today. If ComfyUI ever makes it required, **all**
  ComfyUI-backed models break at once. The guard script asserts it stays optional.
- **Return arity** of ltx2's `_process_input` (3), `_prepare_timestep` (3), `_prepare_context` (2)
  is unpacked positionally; an upstream refactor that changes these is a silent breakage.
- `comfy_aimdo` is a current ComfyUI runtime dependency (imported by `comfy.model_management`); a
  bare environment cannot import ComfyUI without it. Other submodules have their own runtime deps.

## Result of the 2026-06-29 audit (ComfyUI pinned at `0ba903bd`)

- z_image, hunyuan_video_15, flux2, ernie_image, krea2, ltx2: all leaf/block/loading calls match
  the current ComfyUI. No change needed.
- ideogram4: `t_embedding` was fixed upstream of this audit (`self.t_embedding(t, dtype=h.dtype)`).
  This audit found one more divergence — `embed_image_indicator(...)` was missing the
  `out_dtype=h.dtype` that ComfyUI's native forward passes. It is not a crash (`out_dtype` is
  optional on `comfy.ops`'s `Embedding`), but it can desync the embedding output dtype under
  fp8 / mixed-precision, so it was aligned to match ComfyUI.
- The non-ComfyUI submodules were not changed in this update, so their models (chroma, cosmos,
  hidream, hunyuan_image, hunyuan_video, ltx_video, lumina_2, omnigen2) were not re-audited here;
  audit them when their submodule pin moves.

## Result of the 2026-07-29 audit (bitsandbytes 0.50.0, torch 2.13.0)

Two crashes were reported from a training box that had upgraded both dependencies.

**bitsandbytes 0.50.0** — the reported crash was `KeyError: 'percentile_clipping'` at
`optimizers/adamw_8bit.py:30`. Diffing our `update_step` against 0.50.0's showed the single
reported error was hiding three more breakages right behind it:

- `get_config()` no longer emits `percentile_clipping` (the reported crash).
- `get_config()` no longer emits `block_wise` either, so the next two branches would `KeyError`.
- `functional.percentile_clipping` was removed outright.
- `functional.optimizer_update_8bit` (the non-blockwise 8-bit kernel) was removed; 0.50 keeps
  only the blockwise path, and `AdamW8bit.__init__` no longer accepts either keyword.

The positional order of `optimizer_update_32bit` and `optimizer_update_8bit_blockwise` did
**not** change, so the Kahan summation itself was unaffected. The fix reads both keys with
`config.get(...)` and defaults (`percentile_clipping=100`, `block_wise=True`), which reproduces
0.50's behaviour on 0.50 and keeps 0.49.x working, since `requirements.txt` leaves bitsandbytes
unpinned and other users may still be on the old release.

**torch 2.13.0** — `utils/reduction.py` failed at import because
`torch._namedtensor_internals.check_serializing_named_tensor` was removed. The helper is five
lines and its semantics are stable, so it was inlined into `utils/reduction.py` rather than
re-imported from another private location. The file depends on 14 other private torch symbols;
`tools/check_vendored_apis.py` now enumerates all of them so the next removal is caught before
a training run hits it. Those symbols were verified present on torch 2.12.1; they still need one
run of the guard on the 2.13 box to confirm nothing else went missing.

**Lesson worth keeping:** in both the ComfyUI case and this one, the single error message the
user saw was one of several divergences. Fixing only the line in the traceback leaves the rest to
surface one training run at a time, so always diff the whole contact surface, not just the crash
site.
