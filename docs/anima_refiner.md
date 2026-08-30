# Anima Refiner

`anima_refiner` is Anima with its text frontend replaced. Anima routes captions through an
`LLMAdapter`; this architecture routes them through a `ContextRefiner` instead, the frontend
Lumina 2 and Z-Image use.

## Why

Anima's DiT is Cosmos-Predict2's. In ComfyUI the class is literally
`class Anima(MiniTrainDIT)`, and the only thing it adds is the adapter:

```python
class Anima(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter(...)
```

Cosmos-Predict2 consumes old-T5 encoder output directly, with no adapter at all. The
`LLMAdapter` exists to let a Qwen3 LLM drive a DiT that was built T5-native. It embeds T5
token ids as a query sequence and cross-attends into the LLM's hidden states, which papers
over three genuine mismatches:

| | old T5-XXL (T5-11B) | Qwen3 |
|---|---|---|
| Direction | bidirectional encoder | causal decoder |
| Trained for | span corruption / seq2seq | next-token prediction |
| Tokenizer | sentencepiece, 32128 | BPE, 151936 |

Note what the mismatch is *not*: old T5-XXL has `d_model = 1024` (an unusually narrow model
with `d_ff = 65536`), and Qwen3-0.6B has `hidden_size = 1024`. The dimensions already agree.
A six-layer cross-attention module with a 32128-entry embedding table was built anyway, which
is the clearest evidence that the gap is representational rather than dimensional.

`anima_refiner` drops that scaffolding:

* no T5 tokenizer, no 32128-entry embedding table (~33M parameters of dead weight),
* captions are tokenized once instead of twice,
* output positions are indexed by real semantic tokens rather than T5 subwords.

What it keeps is the part that matters. `ContextRefiner` is `cap_embedder` (an RMSNorm
followed by a linear, exactly Lumina 2's recipe) plus bidirectional self-attention blocks.
The bidirectionality is not optional: causal LLMs only give each position left context, and
hybrid linear-attention models such as Qwen3.5 cannot even be made bidirectional by flipping
an attention mask, because their linear-attention layers are recurrences with no attention
matrix to unmask. The refiner blocks restore that mixing.

The default is six refiner layers rather than Lumina's two. Lumina trains its refiner jointly
with the DiT, so the DiT adapts to meet it halfway; here the DiT is usually frozen and the
refiner carries the whole distribution gap on its own.

## Choosing `llm_hidden_layer`

Lumina 2 takes `hidden_states[-2]`. Copying that blindly is a mistake for Qwen3.5-2B.

`hidden_states[i]` is the output of layer `i-1`, and Qwen3.5-2B interleaves attention types
(`full_attention_interval = 4`), putting full-attention layers at 3, 7, 11, 15, 19 and 23 of
24. So:

* `hidden_states[-1]` is the output of layer 23, a **full-attention** layer,
* `hidden_states[-2]` is the output of layer 22, a **linear-attention** layer.

The default here is `-1`. `20` (the output of layer 19) is the other candidate worth sweeping.
Anima itself uses the final hidden state, but it can afford to: its adapter compensates.

## Stages

The ordering matters more than any individual setting. Unfreezing the DiT while the refiner
still emits noise is what causes catastrophic forgetting: loss is highest at the start, so
gradients are largest exactly when the signal is least meaningful. Train the refiner against a
frozen DiT first, so it adapts one-way; only then open the cross-attention, by which point the
gradients arriving there are already small.

### Stage 0 (optional, cheap): distil from Anima's adapter

```
python -m tools.distill_refiner --config examples/anima_refiner_distill.toml
```

Captions only. No images, no VAE, no diffusion, roughly 6GB of VRAM. It teaches the new
refiner to reproduce what Anima's existing `llm_adapter` already emits, which is by definition
what the frozen DiT knows how to read.

**Why the loss is measured at the cross-attention output.** The obvious objective, a
position-wise MSE between teacher and student features, does not work. Both sides are
`(B, L, 1024)` so the shapes match and the code runs — but the teacher's positions are indexed
by T5 tokens and the student's by the source LLM's tokens, so position `i` means different
things on each side. The loss falls a little, then plateaus, with no error to explain why.
Cross-attention is a weighted sum over text positions and does not depend on their indexing,
so pushing both feature sets through the DiT's own frozen cross-attention and comparing there
sidesteps the mismatch and optimises exactly the quantity the DiT consumes.

This is a warm start, not a finished model. Distilling perfectly would reproduce the *0.6B*
model's information content and discard everything the larger encoder knows. Follow it with a
diffusion-loss stage.

### Stage 1: refiner only, DiT frozen

`examples/anima_refiner_stage1.toml`

```toml
base_lr       = 0
self_attn_lr  = 0
cross_attn_lr = 0
mlp_lr        = 0
mod_lr        = 0
refiner_lr    = 1e-4
```

`base_lr` covers everything not matched by a more specific group (`x_embedder`, `t_embedder`,
positional embeddings, `final_layer`). Any group with `lr = 0` has `requires_grad_(False)`
applied and is left out of the optimizer.

### Stage 2: open cross-attention key/value

`examples/anima_refiner_stage2.toml`

A frozen DiT can only understand the vocabulary the previous frontend spoke. If the point of
the change is to exploit what a stronger text encoder knows, the cross-attention has to learn
to read it — with Stage 1 finished, the best case for a fully frozen DiT is parity with base
Anima, not an improvement.

Only `k_proj` and `v_proj` matter: they project the text side, which is what changed. `q_proj`
and `output_proj` work on the image side. `get_param_groups` matches on `.cross_attn`, so
`cross_attn_lr` covers all four; to restrict it to k/v, split the group in
`models/cosmos_predict2.py`:

```python
elif '.cross_attn.k_proj' in name or '.cross_attn.v_proj' in name:
    cross_attn_kv_params.append(p)
```

Keep `cross_attn_lr` roughly 10x below `refiner_lr`. Structural knowledge lives in
`self_attn`, `mlp` and `final_layer`; leaving those frozen protects it.

### Stage 3: LoRA / LoKr / full fine tune

`examples/anima_refiner_lora.toml`, `anima_refiner_lokr.toml`,
`anima_refiner_full_finetune.toml`.

LoRA targets `Block` and `RefinerBlock`. A low-rank update needs something to build on, so
either point `context_refiner_path` at a trained refiner, or set `train_context_refiner = true`
in `[adapter]`, which excludes the refiner from the adapter, trains it densely alongside, and
saves it as `context_refiner.safetensors` ready to feed back through `context_refiner_path`.

Block swapping requires an `[adapter]` block (`train.py` asserts this), so it is available for
LoRA and LoKr but not for the frozen-DiT stages or a full fine tune. Use `pipeline_stages`
across GPUs there instead.

## Loading the text encoder

Three ways, all through `llm_path`:

```toml
# 1. A full Transformers folder. Vision-language models are detected via config.text_config;
#    only the language tower is kept, so the vision tower never costs memory.
llm_path = '/models/Qwen3.5-2B'

# 2. A single safetensors file. Architecture and tokenizer come from a config directory.
#    configs/qwen3_5_2b ships with the repo and is the default.
llm_path = '/models/qwen3_5_2b.safetensors'
#llm_config_path = 'configs/qwen3_5_2b'

# 3. Same, but download the config and tokenizer from the Hub instead.
llm_path = '/models/qwen3_5_2b.safetensors'
llm_repo_id = 'Qwen/Qwen3.5-2B'
```

Checkpoint keys are matched after stripping `model.language_model.`, `language_model.` or
`model.`, so files exported from either the full model or the text tower alone both work. A
checkpoint that does not match the config raises rather than loading partially.

## Text embedding cache

`cache_name` keys the cached embeddings, and **nothing detects a stale cache**. Changing the
text encoder, `llm_hidden_layer` or `max_text_length` all change what gets cached; change
`cache_name` at the same time or old embeddings are silently reused.

At `max_text_length = 512`, 3M images cache to roughly 6TB in bf16 at 2048 dims. Set
`cache_text_embeddings = false` to run the text encoder inside the training loop instead: no
disk cost, about 4.5GB more VRAM, slower steps.

`padding='max_length'` is mandatory, not a default worth changing. Pipeline parallelism sends
fixed-size tensors between stages, and `lumina_2.py` and `z_image.py` both carry comments
about having disabled dynamic padding for this reason. Short captions therefore cost the same
as long ones, so `max_text_length` is worth setting to fit the data.

## Config reference

| Key | Default | Meaning |
|---|---|---|
| `type` | — | `'anima_refiner'` |
| `llm_path` | — | Transformers folder or single safetensors file |
| `llm_config_path` | `configs/qwen3_5_2b` | config/tokenizer dir when `llm_path` is a file |
| `llm_repo_id` | — | Hub repo to fetch config/tokenizer from instead |
| `llm_hidden_layer` | `None` (last) | index into `hidden_states` |
| `n_refiner_layers` | `6` | refiner blocks |
| `max_text_length` | `512` | padded token length |
| `context_refiner_path` | — | trained refiner weights to load |
| `cache_name` | `'anima_refiner'` | text embedding cache key |
| `cache_text_embeddings` | `true` | cache to disk, or run the encoder inline |
| `base_lr` | optimizer `lr` | lr for otherwise unmatched parameters |
| `refiner_lr` | `base_lr` | lr for the refiner |
| `train_context_refiner` | `false` | in `[adapter]`: train the refiner densely |

## Tests

```
pytest test/test_anima_refiner.py test/test_anima_refiner_training.py
```

CPU only, no downloads. `test_anima_refiner_training.py` runs real optimisation steps through
the same layer stack `to_layers()` builds, and checks the invariants pipeline parallelism
depends on: constant shapes across micro batches, and every layer boundary being a valid split
point. To exercise DeepSpeed itself, run `test/debug_deepspeed_init.py` under
`deepspeed --num_gpus=2`.

`test/conftest.py` stubs `comfy_aimdo` and forces ComfyUI to CPU, but only when the real
package is missing and CUDA is unavailable — in a real training environment it does nothing.
