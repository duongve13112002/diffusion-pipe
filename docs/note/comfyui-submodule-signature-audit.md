# Auditing ComfyUI-backed models after a submodule update

- Date: 2026-06-29
- Prompt: A real training run of Ideogram4 crashed with
  `TypeError: Ideogram4EmbedScalar.forward() missing 1 required positional argument: 'dtype'`
  after the ComfyUI submodule was bumped. The crash was not in our code per se — it was a
  signature drift in ComfyUI that our re-implemented forward pass did not follow. This note
  records why that whole class of bug happens and the exact procedure to catch it every time
  the `submodules/ComfyUI` pin changes.

## Why this class of bug exists

The ComfyUI-backed models do **not** run ComfyUI's own `forward()`. To split a model across
pipeline-parallel stages, each `models/*.py` re-implements the forward pass: it grabs ComfyUI
**leaf submodules** off the loaded model (e.g. `self.t_embedding = model.t_embedding`) and calls
them directly inside the `InitialLayer` / `TransformerLayer` / `FinalLayer` wrappers produced by
`to_layers()`. Some models go further and **monkey-patch** a ComfyUI method (ltx2 replaces
`BasicAVTransformerBlock.forward` to strip in-place ops) or **override** text-encoder methods
(`base.py` replaces `ClipTokenWeightEncoder.encode_token_weights`).

Every one of those is a copy of, or a direct call into, ComfyUI internals. When the ComfyUI pin
moves and a leaf gains a required argument, changes its return arity, renames an attribute, or
makes an optional argument required, the re-implementation keeps calling the old shape and breaks
at runtime — usually deep inside a forward pass, often only for one model.

## The core audit principle

> diffusion-pipe copies ComfyUI's forward logic, so **the way it calls a leaf must match the way
> ComfyUI's own `forward()` calls that same leaf.** Compare the two call sites; any divergence is a
> suspected bug.

That is exactly how the Ideogram4 bug reads: ComfyUI's native forward calls
`self.t_embedding(t, dtype=x.dtype)`, but our `InitialLayer.forward` called `self.t_embedding(t)`.

## Procedure (run on every `submodules/ComfyUI` change)

1. List the ComfyUI-backed models. They are the `ComfyPipeline` subclasses:
   `grep -rl "ComfyPipeline" models/` → currently `z_image`, `ltx2`, `hunyuan_video_15`, `flux2`,
   `ernie_image`, `krea2`, `ideogram4`. Also check any non-ComfyPipeline model that still imports
   from `comfy` (e.g. `chroma` uses `comfy.ldm.flux.layers.timestep_embedding`).
2. For each model, find every cross-boundary call and check it against the **current** ComfyUI:
   - Direct comfy imports: `grep -nE "^(from|import) .*comfy" models/<m>.py`, then verify each
     imported function/class signature (`timestep_embedding`, `precompute_freqs_cis`, `rms_norm`,
     `pad_to_patch_size`, `ModulationOut`, ...).
   - Grabbed leaf submodules called in the `to_layers()` wrappers (`InitialLayer`,
     `TransformerLayer`, `FinalLayer`): locate the matching ComfyUI model class under
     `submodules/ComfyUI/comfy/ldm/<arch>/model.py` and diff our leaf calls against how ComfyUI's
     own `forward()` / `forward_orig()` calls the same leaves. Watch for: new required args,
     `out_dtype=`, `transformer_options=`, and changes to **return arity** that we unpack.
   - Monkey-patched methods (ltx2 `BasicAVTransformerBlock.forward`, the text-encoder overrides in
     `base.py`): the patched function's signature must match the method it replaces, and every
     attribute/method it calls must still exist with the same shape.
   - Loading / adapter / VAE / text-encoder APIs: `comfy.sd.load_clip`,
     `comfy.sd.load_diffusion_model`, `comfy.sd.load_checkpoint_guess_config`,
     `comfy.sd.load_lora_for_models`, `comfy.sd.VAE.__init__`, `comfy.utils.load_torch_file`,
     `comfy.sd1_clip.gen_empty_tokens`. These cover the full-fine-tune, LoRA-merge, and caching
     paths, not just LoRA forward.
3. Run the automated guard on a machine where ComfyUI imports (training box; the CPU dev box is
   missing ComfyUI runtime deps such as `comfy_aimdo`):

   ```
   python tools/check_comfy_signatures.py
   ```

   It encodes the contracts verified by hand. A non-zero exit means ComfyUI drifted — fix the
   matching model code, and if the ComfyUI change was intentional, update the check in the same
   commit so the script keeps reflecting reality.
4. Fix any divergence by making our call match ComfyUI's current native call. Note which models
   genuinely needed a change so the fix can be GPU-verified on the affected model.

## Cross-cutting couplings worth remembering

- **`transformer_options` is always omitted** by our block/refiner calls and is optional (default
  `{}`) in every ComfyUI block today. If ComfyUI ever makes it required, **all** ComfyUI-backed
  models break at once. The guard script asserts it stays optional.
- **Return arity** of ltx2's `_process_input` (3), `_prepare_timestep` (3), `_prepare_context` (2)
  is unpacked positionally; a ComfyUI refactor that changes these is a silent breakage.
- `comfy_aimdo` is a current ComfyUI runtime dependency (imported by `comfy.model_management`); a
  bare environment cannot import ComfyUI without it.

## Result of the 2026-06-29 audit (ComfyUI pinned at `0ba903bd`)

- z_image, hunyuan_video_15, flux2, ernie_image, krea2, ltx2: all leaf/block/loading calls match
  the current ComfyUI. No change needed.
- ideogram4: `t_embedding` was fixed upstream of this audit (`self.t_embedding(t, dtype=h.dtype)`).
  This audit found one more divergence — `embed_image_indicator(...)` was missing the
  `out_dtype=h.dtype` that ComfyUI's native forward passes. It is not a crash (`out_dtype` is
  optional on `comfy.ops`'s `Embedding`), but it can desync the embedding output dtype under
  fp8 / mixed-precision, so it was aligned to match ComfyUI.
