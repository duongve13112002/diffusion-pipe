# Stream healing for MOSS-Image: implementation plan

Date: 2026-11-01

Task: add the residual-stream penalties and the outlier decay of KohakuMossTrainer's
`docs/concepts/stream-healing.md` to diffusion-pipe, for both LoRA and full fine-tunes.
Background and the source references are in
[kohakumoss-vs-diffusion-pipe-moss-training.md](kohakumoss-vs-diffusion-pipe-moss-training.md).
Status: implemented (`utils/stream_loss.py`, `utils/outlier_decay.py`, `models/moss_image.py`,
`train.py`), checked on CPU only. The measured facts below come from reading this repository's code
(commit `70b8468` plus the note commit) and the installed DeepSpeed 0.18.4. Where the code differs
from this plan: the vector the layers carry holds per-block shares (concentration, the RMS term and
value, the two write shares) rather than finished terms, so the write hinge and its schedule are
formed in the loss function and the layers do not need the step; logged names are
`train/flow_loss`, `train/stream_penalty`, `train/penalty_<term>`, `train/write_share_<part>`,
`train/write_target` and `train/stream_rms`.

## Scope and decisions

| Piece | LoRA | Full fine-tune | Decision |
| --- | --- | --- | --- |
| `STREAM_LOSS` term `concentration` | yes | yes | implement |
| `STREAM_LOSS` term `rms` | yes | yes | implement; per-sample RMS, no sigma weighting |
| `STREAM_LOSS` term `write` | yes | yes | implement; the main term of the doc's recommended configuration |
| `STREAM_LOSS` term `peak` | no | no | skip: it exists for fp16 headroom (default cap 32768 against fp16's 65504), and diffusion-pipe trains bf16 or fp32 only |
| `OUTLIER_DECAY` | no | yes | implement; it decays base weights, which a LoRA run never changes, so a config with `[adapter]` is refused |

Nothing here has been shown to improve image quality: KohakuMoss measured the `concentration` +
`rms` finetune and a toy model of the combination, and lists the full run of `write` +
`OUTLIER_DECAY` as pending. Every part is therefore opt-in and off by default.

## Config

Stream penalties are MOSS-Image specific, so they live in `[model]`:

```toml
[model.stream_loss.write]
weight = 5e-3
target = 0.5        # final hinge share, required
anneal = 2000       # steps from share 1 down to target
start = 0
k = 32
parts = 'both'      # 'attn', 'ffn' or 'both'
dilute = false      # false: shrink-only (the total in the denominator is detached)

[model.stream_loss.rms]
weight = 5e-5
target = 1.0

[model.stream_loss.concentration]
weight = 1e-2
k = 32
dilute = true
```

`weight` is required for a present term (0 turns it off); unknown terms and fields are refused by
name; `k` must not exceed the model width (checked once the model is loaded).

The outlier decay is not model specific and goes in a top-level table, refused unless the run is
a full fine-tune:

```toml
[outlier_decay]
rate = 6.0
cap = 16.0
cap_start = 64.0
cap_anneal = 10000
warmup = 1000
targets = ''        # regex over the parameter names, empty = every 2-D trained weight
exclude = ''
include_1d = false
```

All defaults and validation live in `utils/stream_loss.py` and `utils/outlier_decay.py`, called from
`set_config_defaults` in `train.py` so a typo fails before any model loads, and unit-testable
without the training stack (the pattern of `utils/oplora.py` and `utils/ema.py`).

## Stream penalties

### Where the energies come from

ComfyUI's `Block.forward` (`comfy/ldm/moss_image/model.py`) does both residual adds inline:
`x = residual(x, self.proj["o"](out), mod, 2, ref)` and the same for `proj["ffn_down"]` with gate
`mod[5]`. In training `gated_residual` is `torch.addcmul(x, y, gate)`, so a sublayer's write is
exactly `y * gate`, `y` being the projection output.

- `linear_input_act` (`comfy/ops.py`) calls `linear(...)` when `comfy.model_management.in_training`,
  so a forward hook on `block.proj['o']` and `block.proj['ffn_down']` fires for both sublayers.
- `train.py` injects the adapter (`model.configure_adapter`, line 690) before `model.to_layers()`
  (line 754), so the hooked modules are PEFT's wrappers and the captured output includes the
  LoRA delta.
- The gates are `mod[2]` and `mod[5]`, which `TransformerLayer.forward` already builds with
  `rows_mod.unsqueeze(2).unbind(1)` (six tensors of shape `(batch, 1, dim)`).

So no change to the ComfyUI submodule is needed. The hook computes the channel energy of
`y * gate` in fp32 and adds it to a per-layer accumulator; the energies of the length groups
(`_length_groups`) add up before the shares are taken, as KohakuMoss sums over the micro-batch.

### Getting the penalty to the loss across pipeline stages

The loss is computed on the last stage only. diffusion-pipe already has an auxiliary loss that
crosses stages: Iris's REPA (`models/iris.py`, `RepaLayer`, `get_loss_fn`, `get_extra_log_scalars`).
The same mechanism is used:

1. `InitialLayer` appends a float32 vector `stream` of fixed length to the tuple
   `(hidden, mod, head, lengths, sizes)`, tied to `hidden` (as `_tie` does) so it has a gradient path.
2. Each `TransformerLayer` adds its block's contributions to `stream` and passes it on. The layer
   returns the same 6-tuple shape whether or not it computes anything, so DeepSpeed's buffer
   shapes do not change between train and eval.
3. `FinalLayer` returns `(out, stream)` while a penalty is active and a plain tensor otherwise
   (sampling calls `self.pipeline_model(...).float()` in `_predict`).
4. The model's `get_loss_fn` wraps the base loss (huber, smooth L1, mask and batch-fill weight
   stay exactly as they are), unpacks the tuple, and adds `sum(weight_i * stream[i] / count_i)`.
5. Because the value rides the layers instead of a side channel, DeepSpeed's per-layer activation
   checkpointing recomputes it with the layer, with no special handling.

DeepSpeed facts checked in `deepspeed/runtime/pipe/engine.py`: `train_batch` calls
`self.module.train()` and `eval_batch` calls `self.module.eval()`; the loss function receives the
last stage's whole output tuple. The penalty is active when the layer is in training mode and
sampling is not running (`LayerState` gets a `sampling` flag set in `_predict`, next to
`skip_layers`), not when `torch.is_grad_enabled()`: with reentrant checkpointing the first
forward runs under no_grad and its value is what reaches the loss.

`stream` slots (all float32, one vector): `concentration`, `rms`, `write` (attached to the graph),
and `write_share_attn`, `write_share_ffn`, `stream_rms` (detached, for logging). The per-layer
sums are divided by the block count, or by blocks times tapped parts for `write`, at the loss.

### Terms (per block, `x` the block output, `D` the width, `T_i` the valid tokens of sample `i`)

- `concentration`: top-`k` share of the channel energy of `x` over the micro-batch, mean over blocks.
- `rms`: per sample `R_i = ||x_i|| / sqrt(T_i * D)` (valid tokens are the image tokens plus that
  sample's text length; the padding rows are zero), term `mean_i (R_i / target - 1)^2`, mean over
  blocks.
- `write`: top-`k` share of each sublayer write's channel energy, hinge
  `relu(share - tau(step))^2`, mean over blocks and tapped parts. `tau` is 1 before `start`, then
  linear to `target` over `anneal` steps. With `dilute = false` the total in the denominator is
  detached.
- Top-`k` shares use `topk` on the fp32 energy vector, as in KohakuMoss (`top_shares`).

The step for `tau` is the optimizer step (`train.py`'s `step`, restored on resume), passed with a
new `model.set_training_step(step)` call before `train_batch`, next to the existing
`set_training_epoch` (`utils/dataset.py` line 4714). Every pipeline process makes the call.

### Metrics

`MossImagePipeline.get_extra_log_scalars` (called on every rank every step, `train.py` line 1181)
all-reduces sums and a count recorded by the last stage's loss function, the Iris pattern:
`train/flow_loss`, `train/stream_penalty`, and per term the unweighted value, the two write shares,
the write target and the stream RMS. `train/loss` stays the total the optimizer minimised.

### Refused combinations

`compile = true` (`pipeline_model.compile(dynamic=True)` would trace the hooks and the accumulator
state; not tried), and a `k` above the model width.

## Outlier decay

After every `train_batch`, before `step_stats.after_step()` and `ema.update` (the position of
`oplora_projector.project()`, so the step statistics and the EMA see the decayed weights):

    p -= lr * rate(t) * sign(p) * relu(|p| - cap(t) * r0)

- `r0` is each tensor's RMS from the weights as loaded at launch. It is taken right before the
  resume block in `train.py` (line 1033), when `model_engine.load_checkpoint` has not yet replaced
  them, so a resumed run keeps the original `r0` as long as the config's `transformer_path` is the
  one it started from.
- Targets: the trained parameters of this pipeline stage (`requires_grad` after the optimizer
  froze zero-lr groups), 2-D unless `include_1d`, selected by the `targets` / `exclude` regexes on
  `original_name`, the name the saver and the EMA use.
- Learning rate: each parameter group's lr as it was for the step (read before `train_batch`,
  because DeepSpeed's scheduler advances inside it).
- Schedules as in KohakuMoss: `cap(t)` falls geometrically from `cap_start` to `cap` over
  `cap_anneal` steps; `rate(t)` ramps linearly from 0 over `warmup` steps; `t = step - 1`.
- bf16 weights: `lr * rate * excess` is about 1e-4 of the excess, below one bf16 step, so rounding
  to nearest would never move anything. The update is computed in fp32 and rounded back with the
  stochastic rounding of `utils/ema.py` (`to_bfloat16_stochastic`), which keeps its expectation
  exact and leaves an untouched element (exactly representable in bf16) unchanged. Weights that
  are fp32 are written directly. The optimizer's Kahan buffer (`optimi.AdamW` applies it
  automatically to bf16 parameters, `kahan_sum=None`) is not adjusted: the decay changes the stored
  weight, and the error this adds is below one bf16 step per element and unbiased.
- Memory: a transient fp32 copy of one tensor at a time.
- Metrics `train/outlier_over` (elements above the cap) and `train/outlier_max` (the largest
  `|p| / (cap * r0)`), computed on the device every step and reduced only on logged steps, from
  data-parallel rank 0 of each stage.

## Tests

CPU, in the repo's pytest suite (the baseline `test/test_moss_image.py` runs 170 passed, 16 skipped):

- `test/test_stream_loss.py`: config defaults and every refusal; `top_shares` with and without
  `dilute`; `write_target` schedule; each term against a direct reference computation; the
  tiny-MOSS pipeline (`TINY` in `test_moss_image.py`) with the penalty on: the carried vector
  equals the terms recomputed from a plain ComfyUI forward with its own hooks, gradients reach the
  LoRA parameters and the base parameters, outputs are unchanged when the weights are zero, eval
  and sampling still return a plain tensor and a loss equal to the flow loss, the tuple structure
  is the same in train and eval, and the hinge is zero at share <= target.
- `test/test_outlier_decay.py`: update formula, both schedules, regex selection, the bf16
  stochastic rounding (unbiased over many trials, untouched elements bit-identical), group lr
  handling including lr 0, refusal with `[adapter]`, `r0` from the launch weights.
- The existing suites must stay green.

GPU, standalone like `tools/test_oplora_gpu.py`: `tools/test_stream_healing_gpu.py`, one script that
runs the real `train.py` for a few steps with both features on and checks the logged metrics,
using every visible GPU through `deepspeed --num_gpus` with `pipeline_stages` taken from the GPU
count, so single and multi GPU need no edits.

## Docs

`docs/supported_models.md#moss-image` (config keys), commented blocks in
`examples/moss_image/moss_image_lora.toml` and `moss_image_full_8gpu.toml`, an entry in
`implementation-notes.html`, this note, and the index.

## Risks and open points

- The hook-on-projection approach is verified by reading the code; the CPU test with a tiny MOSS
  model is what proves it against ComfyUI's own forward.
- Nothing here is tuned. KohakuMoss's weights (`write` 5e-3, `rms` 5e-5, `rate` 6) are for its
  10k-step full fine-tune at lr 2e-5; a LoRA run at lr 1e-4 may need other values.
- No GPU is available where this is written, so the training run and the multi-GPU path are
  checked only by the script above on the maintainer's machine.
