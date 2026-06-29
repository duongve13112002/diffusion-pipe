# Auditing submodule-backed models after a submodule update

- Date: 2026-06-29
- Prompt: A real training run of Ideogram4 crashed with
  `TypeError: Ideogram4EmbedScalar.forward() missing 1 required positional argument: 'dtype'`
  after the ComfyUI submodule was bumped. The crash was not in our code per se — it was a
  signature drift in a vendored submodule that our code did not follow. ComfyUI is only the most
  active example: **every** submodule in `submodules/` is a third-party library that one or more
  models import from and call directly, so the same drift can come from any of them. This note
  records why the bug class exists and the procedure to re-audit after **any** submodule pin changes.

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

## Procedure (run on every submodule pin change)

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
