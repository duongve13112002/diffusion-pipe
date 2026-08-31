# Lumina2 training: diffusion-pipe vs sd-scripts

Date: 2026-07-02
Prompted by: user request to check whether the Lumina2 training implementation in this repo (`models/lumina_2.py`) matches the Lumina2 implementation in the `sd-scripts` tree that was dropped into the repo root (untracked, not a submodule).

Sources read:
- `models/lumina_2.py` (diffusion-pipe)
- `sd-scripts/lumina_train_network.py`, `sd-scripts/library/lumina_train_util.py`, `sd-scripts/library/strategy_lumina.py`, `sd-scripts/library/lumina_models.py`, `sd-scripts/networks/lora_lumina.py`
- `models/base.py` (loss_fn), `sd-scripts/train_network.py` (process_batch / loss)

## TL;DR

Same model family, same flow-matching math, same LoRA target (`JointTransformerBlock`) — but two structurally different training harnesses. The per-sample math (timestep sampling, noise formula, LoRA target modules) is close to identical; everything around it (how the model is split, how the loop is driven, how config is expressed) is not, because the two repos parallelize training differently.

| Aspect | diffusion-pipe | sd-scripts |
| --- | --- | --- |
| Parallelism | DeepSpeed pipeline parallelism (`to_layers()` splits the model into `nn.Module` stages passed between GPUs) | `accelerate` (DDP / single GPU), optional block-swap offloading (`enable_block_swap`) |
| Model class used | `Lumina_2.models.model.NextDiT_2B_GQA_patch2_Adaln_Refiner` (external submodule/package) | `library.lumina_models.NextDiT` (vendored copy in-repo) |
| Text encoder | Gemma-2-2B, loaded from `configs/gemma_2_2b` + `--llm_path` safetensors | Gemma-2-2B, loaded via `transformers` `Gemma2Model`, path from `--gemma2` |
| Config format | TOML (`[model]` table) | argparse / CLI flags (or a config file consumed by argparse) |
| Training entry | `train.py` dispatch → `Lumina2Pipeline` | `lumina_train_network.py` → `LuminaNetworkTrainer(train_network.NetworkTrainer)` |

## 1. Model loading

Both load the same three components — VAE (Flux-style AutoencoderKL), Gemma-2-2B text encoder, NextDiT transformer — but diffusion-pipe loads everything eagerly on CPU with `init_empty_weights()` + `set_module_tensor_to_device()` so it can later be sharded across pipeline stages ([lumina_2.py:43-87](../../models/lumina_2.py#L43-L87)). sd-scripts loads through `lumina_util.load_lumina_model` / `load_gemma2` / `load_ae` ([lumina_train_network.py:54-88](../../sd-scripts/lumina_train_network.py#L54-L88)) with optional fp8 casting and block-swap for the transformer — mechanisms diffusion-pipe doesn't need because pipeline parallelism already keeps only a shard of the model resident per GPU.

## 2. Text tokenization / encoding

Nearly identical in substance:
- Both tokenize with `AutoTokenizer.from_pretrained` for Gemma-2, `padding_side='right'`, `max_length=256` (diffusion-pipe hardcodes 256; sd-scripts defaults to 256 via `--gemma2_max_token_length`).
- Both take `hidden_states[-2]` (second-to-last hidden layer) as the caption embedding — see [lumina_2.py:129-134](../../models/lumina_2.py#L129-L134) vs [strategy_lumina.py:118-125](../../sd-scripts/library/strategy_lumina.py#L118-L125).
- sd-scripts additionally prepends a configurable `--system_prompt` + `<Prompt Start>` sentinel ([strategy_lumina.py:37-41](../../sd-scripts/library/strategy_lumina.py#L37-L41)); diffusion-pipe has no system-prompt concept — it tokenizes the caption as-is.
- diffusion-pipe pads to `max_length` only; sd-scripts additionally pads to a multiple of 8 (`pad_to_multiple_of=8`).

Both cache text-encoder outputs to disk ahead of training (diffusion-pipe via the generic `utils/cache.py` path shared by all models; sd-scripts via `LuminaTextEncoderOutputsCachingStrategy`, a Lumina-specific subclass of its `strategy_base` caching framework).

## 3. Timestep sampling / noise formula

Both implement rectified-flow training (`x_t = (1-t) * x + t * noise`, `target = x - noise`), but express the shift function differently:

**diffusion-pipe** ([lumina_2.py:149-177](../../models/lumina_2.py#L149-L177)): samples `t` from `logit_normal` (sigmoid of a standard normal, optionally scaled by `sigmoid_scale`) or `uniform`, then applies either a fixed `shift` config value or the resolution-dependent `lumina_shift` (`get_lin_function` + `time_shift`, identical formula to Flux's resolution-dependent shift). Noise is added directly in `prepare_inputs`, no `noise_scheduler` object involved — the DeepSpeed pipeline doesn't have one.

**sd-scripts** ([lumina_train_util.py:808-874](../../sd-scripts/library/lumina_train_util.py#L808-L874) `get_noisy_model_input_and_timesteps`): supports more sampling modes (`uniform`, `sigmoid`, `shift`, `nextdit_shift`, `flux_shift`, plus SD3-style `logit_normal`/`mode`/density weighting via `compute_density_for_timestep_sampling`), goes through a real `FlowMatchEulerDiscreteScheduler` object, and additionally supports `--ip_noise_gamma` (input-perturbation noise). The `nextdit_shift` mode is the same `get_lin_function`/`time_shift` pair diffusion-pipe uses for `lumina_shift`.

Net: sd-scripts exposes strictly more timestep-sampling knobs (it's the more actively-developed, general-purpose trainer supporting many architectures with shared flags); diffusion-pipe implements the subset it needs directly in the pipeline, with the same underlying shift math for the resolution-aware case.

The `t` sign convention also matches once you track it through: diffusion-pipe stores `t` as "noise fraction" internally and flips it (`1 - t`) right before calling the model ([lumina_2.py:179-180](../../models/lumina_2.py#L179-L180)); sd-scripts keeps `timesteps` in `[0, 1000]` scale and does the same flip inline in `call_dit` (`t=1 - timesteps / 1000`, [lumina_train_network.py:276](../../sd-scripts/lumina_train_network.py#L276)) and again in `denoise` for sampling ([lumina_train_util.py:653](../../sd-scripts/library/lumina_train_util.py#L653)). Same convention, just different scale (`[0,1]` vs `[0,1000]`).

## 4. Forward pass / model call

Both ultimately call the same `NextDiT.forward(x, t, cap_feats, cap_mask)` signature — the vendored model architectures are functionally the same network (patchify, RoPE position ids, `context_refiner`/`noise_refiner`, joint transformer blocks, `final_layer`/unpatchify).

The difference is *how* that forward pass is invoked:
- diffusion-pipe splits `NextDiT` itself into pipeline-parallel stages (`InitialLayer` does embedding + patchify + refiners, then one `TransformerLayer` per block, then `FinalLayer` does the output projection + unpatchify — see `to_layers()` at [lumina_2.py:182-188](../../models/lumina_2.py#L182-L188)). Each stage forward is wrapped in `@torch.autocast('cuda', dtype=AUTOCAST_DTYPE)` and its outputs must be made contiguous, fixed-shape tensors (`make_contiguous`) because DeepSpeed pipeline parallelism sends activations between GPU stages and needs consistent shapes — note the explicit comment that `max_seq_len` is intentionally *not* allowed to vary per micro-batch for this reason ([lumina_2.py:235-236](../../models/lumina_2.py#L235-L236)), a constraint sd-scripts doesn't have since the whole model lives in one process.
- sd-scripts calls the model as one opaque function inside `call_dit` under `accelerator.autocast()`, with no manual layer splitting (`train_network.py`'s generic training loop handles gradient accumulation, DDP, and optional block-swap instead of layer sharding).

## 5. Differential output preservation ("prior preservation")

sd-scripts has an anti-forgetting feature baked into `get_noise_pred_and_target` — when a batch item is tagged `diff_output_preservation` in `custom_attributes`, it re-runs the un-LoRA'd model (`network.set_multiplier(0.0)`) to get a prior prediction and substitutes it as the target ([lumina_train_network.py:295-322](../../sd-scripts/lumina_train_network.py#L295-L322)). diffusion-pipe has no equivalent in `lumina_2.py`; the closest concept in this repo is OPLoRA (`utils/oplora.py`), which is a different mechanism (orthogonal projection of LoRA updates, not per-batch dual forward passes) — see `docs/oplora.md`.

## 6. Loss computation

- diffusion-pipe: generic `loss_fn` shared by all models in `models/base.py` (`CommonPipeline`/`BasePipeline`) — MSE (or Huber/smooth-L1) between `output` and `target`, multiplied by an optional per-pixel mask, no per-timestep weighting is applied for Lumina2 specifically ([base.py:363-379](../../models/base.py#L363-L379)).
- sd-scripts: `conditional_loss` (same MSE/Huber/L2 choice) but additionally multiplies by a `weighting` term when `model_prediction_type == "sigma_scaled"` (SD3-style loss reweighting, `compute_loss_weighting_for_sd3`), and applies mask via a separate `apply_masked_loss` helper before reducing to a per-sample scalar ([train_network.py:461-486](../../sd-scripts/train_network.py#L461-L486)). Lumina2's default `model_prediction_type` is `"raw"`, so this weighting path is inactive unless explicitly configured — meaning the *default* loss math is the same as diffusion-pipe's (mask-only weighting), but sd-scripts has the extra knob available.

## 7. LoRA / adapter targeting

Both target the same transformer block class:
- diffusion-pipe: `adapter_target_modules = ['JointTransformerBlock']` ([lumina_2.py:41](../../models/lumina_2.py#L41)), applied generically through `peft.get_peft_model` in `models/base.py`'s `configure_adapter`.
- sd-scripts: `LUMINA_TARGET_REPLACE_MODULE = ["JointTransformerBlock", "FinalLayer"]` in `networks/lora_lumina.py:471`, plus an optional, separate LoRA on the Gemma-2 text encoder (`TEXT_ENCODER_TARGET_REPLACE_MODULE`, gated by `network_train_unet_only`).

Differences: sd-scripts additionally LoRA-fies `FinalLayer` and can train a LoRA on the text encoder; diffusion-pipe only targets `JointTransformerBlock` and never trains the text encoder (text embeddings are always pre-cached to disk in this repo — see CLAUDE.md's "training text encoders is not supported"). diffusion-pipe uses `peft`'s standard LoRA/LoKr implementation (config-driven, matches Linear submodules by name inside the target block classes) rather than sd-scripts' own hand-rolled `LoRAModule`/`LoRANetwork` classes.

## Conclusion

The core per-sample math for Lumina2 — flow-matching noise formula, resolution-aware time-shift, Gemma-2 caption encoding, `JointTransformerBlock` LoRA targeting — is effectively the same between the two repos; diffusion-pipe's is a trimmed-down subset of sd-scripts' more configurable version (fewer timestep-sampling modes, no differential-output-preservation, no text-encoder LoRA, no per-sample loss weighting for the default prediction type). The structural implementation is not the same: diffusion-pipe restructures the model itself into pipeline-parallel stages for DeepSpeed (with a hard constraint that per-batch sequence length must be fixed across the pipeline), while sd-scripts drives the whole model as one unit through `accelerate`, trading pipeline parallelism for a much larger surface of CLI-configurable training options (fp8, block-swap offload, differential output preservation, SD3-style loss weighting).
