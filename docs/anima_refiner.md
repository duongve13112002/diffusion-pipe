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

Only `anima_refiner` reads any of the options documented here. `cosmos_predict2` and `anima`
keep exactly the config surface and behaviour they had before this architecture existed.

## The modes

These are *configurations*, not a fixed pipeline. They differ only in which learning rates are
non-zero; every one of them loads the model through the same path, so they can be run in any
order and any one's output can feed any other.

| Mode | Trains | Config |
|---|---|---|
| `distill` | refiner, against Anima's LLMAdapter, captions only | `examples/anima_refiner_distill.toml` |
| `refiner_only` | refiner, diffusion loss, DiT frozen | `examples/anima_refiner_refiner_only.toml` |
| `refiner_crossattn` | refiner + cross-attention, diffusion loss | `examples/anima_refiner_refiner_crossattn.toml` |

Plus `anima_refiner_lora.toml`, `anima_refiner_lokr.toml` and
`anima_refiner_full_finetune.toml` for the adapter and full fine tune variants, and
`anima_refiner_onfly_lora.toml` + `anima_refiner_dataset.toml` for a worked example with
on-the-fly text embeddings and per-access caption augmentation.

All of them use `adamw8bitkahan`. Kahan summation is what makes an 8-bit optimizer usable with
`bfloat16` weights: bf16 has 8 bits of mantissa, so a small update rounds away entirely, and the
compensation term keeps it. Plain `adamw8bit` loses those updates silently.

### On-the-fly text embeddings

`cache_text_embeddings = false` keeps the text encoder resident and runs it every step. It costs
VRAM and time, and buys caption augmentation that is re-drawn on every access rather than frozen
into the cache.

The difference is not cosmetic. With cached embeddings, `cache_shuffle_num` fixes how many tag
orders and dropout draws exist for the entire run; `tag_dropout_rate` with `cache_shuffle_num = 1`
gives a *single* permanent draw, which deletes those tags rather than augmenting them. With
on-the-fly embeddings there is no such ceiling.

The dataset code adapts automatically. When the model caches no text embeddings, the metadata
keeps captions exactly as they are on disk — marker included — and `__getitem__` strips the
marker, shuffles and drops tags on each access. Epoch length is unchanged: `cache_shuffle_num`
still decides how many entries a caption contributes, those entries just differ every pass. VAE
latents are cached either way; that is the expensive half and it does not depend on the caption.

Ordering still matters *for results*, even though nothing enforces it. Unfreezing the DiT
while the refiner still emits noise is what causes catastrophic forgetting: loss is highest at
the start, so gradients are largest exactly when the signal is least meaningful. Training the
refiner against a frozen DiT first lets it adapt one-way; opening the cross-attention
afterwards means the gradients arriving there are already small. That is a recommendation, not
a constraint the code imposes.

## Which file holds what

There are two save formats, and `context_refiner.safetensors` is produced by only two modes.

| Mode | Produces | Where the refiner lives |
|---|---|---|
| `distill` | `context_refiner.safetensors` | that file (refiner **only**) |
| `distill` with `save_full_model = true` | both, plus `model.safetensors` | inside it, under `net.context_refiner.*` |
| `refiner_only` / `refiner_crossattn` / full fine tune | `model.safetensors` | inside it, under `net.context_refiner.*` |
| LoRA / LoKr | `adapter_model.safetensors` | inside it, as adapter tensors |
| LoRA / LoKr with `train_context_refiner = true` | both, plus `context_refiner.safetensors` | the separate file |

`context_refiner.safetensors` contains **nothing but the refiner** — roughly 77M parameters
(`cap_embedder`, the refiner blocks, `norm_out`). No DiT, no VAE, no LLM, no optimizer state.
The LLM is never part of any saved file; it is always loaded separately via `llm_path`.

A complete model is assembled from independent pieces:

```
transformer_path          DiT: blocks, self-attention, cross-attention, and
                          (once trained) the refiner
      + llm_path          the text encoder, always separate
      + context_refiner_path    OPTIONAL, only when the refiner is in its own file
```

Most of the time `context_refiner_path` is unset, because the refiner is already inside
`transformer_path`. That is the normal case, not a missing step.

## Checkpoint loading

One rule: **whatever weights you point at are the weights that get used.** Nothing is
re-initialised from Anima when real weights exist.

```
                    transformer_path
                            |
              does it contain context_refiner.* ?
                   /                        \
                 yes                         no
                  |                           |
      derive n_refiner_layers and      build a fresh refiner from
      cap_feat_dim FROM THE WEIGHTS    n_refiner_layers / llm_path
      (config that disagrees = error)            |
                  \                             /
                   \                           /
                    is context_refiner_path set?
                       /                \
                     yes                 no
                      |                   |
             it WINS, with a         use what the
             loud warning            checkpoint had
```

The shape is derived from the weights rather than the config on purpose. The loading loop
skips parameter names absent from the checkpoint, so a checkpoint holding more refiner layers
than the config asked for would quietly lose the surplus — a silently wrong model with no
error anywhere. Deriving the shape removes that failure mode, and a config that contradicts
the weights raises instead.

`context_refiner_path` winning over the checkpoint is deliberate: that file was named
explicitly. Because it is also an easy way to continue training from the wrong starting point,
it prints a warning naming both files every time it happens.

## The text encoder

Three ways to supply it, all through `llm_path`:

```toml
# 1. A full Transformers folder. Vision-language models are detected via config.text_config;
#    only the language tower is kept, so the vision tower never costs memory.
llm_path = '/models/Qwen3.5-2B-Base'

# 2. A single safetensors file. Architecture and tokenizer come from a config directory.
#    configs/qwen3_5_2b_base ships with the repo and is the default.
llm_path = '/models/qwen3_5_2b_base.safetensors'
#llm_config_path = 'configs/qwen3_5_2b_base'

# 3. Same, but download the config and tokenizer from the Hub instead.
llm_path = '/models/qwen3_5_2b_base.safetensors'
llm_repo_id = 'Qwen/Qwen3.5-2B-Base'
```

Use the **Base** model, not Instruct, matching how Anima uses Qwen3-0.6B-Base. The two share a
byte-identical `config.json`, so the architecture is the same either way; what differs is the
tokenizer, where Instruct declares `eos = <|im_end|>` against Base's `<|endoftext|>`.

Checkpoint keys are matched after stripping `model.language_model.`, `language_model.` or
`model.`, so files exported from either the full model or the text tower alone both work. A
checkpoint that does not match the config raises rather than loading partially.

### Choosing `llm_hidden_layer`

Lumina 2 takes `hidden_states[-2]`. Copying that blindly is a mistake for Qwen3.5-2B-Base.

`hidden_states[i]` is the output of layer `i-1`, and Qwen3.5 interleaves attention types
(`full_attention_interval = 4`), putting full-attention layers at 3, 7, 11, 15, 19 and 23 of
24. So:

* `hidden_states[-1]` is the output of layer 23, a **full-attention** layer,
* `hidden_states[-2]` is the output of layer 22, a **linear-attention** layer.

The default here is `-1`. `20` (the output of layer 19) is the other candidate worth sweeping.
Anima itself uses the final hidden state, but it can afford to: its adapter compensates.

## Caching

Latents and text embeddings share a cache directory but are fingerprinted **separately**. The
text embedding fingerprint includes the identity of the text encoder — `llm_path`,
`llm_hidden_layer`, `max_text_length`, the hidden size — so:

* changing the text encoder invalidates only the text embeddings,
* **latents are never touched**, which matters because they are by far the more expensive half
  and the VAE has not changed.

This is automatic. There is nothing to configure and no cache name to keep track of. Models
other than `anima_refiner` supply no such identity, so their cache fingerprints are exactly
what they were before.

`anima_refiner` also shares `anima`'s cache directory on purpose. The directory name selects
the whole tree, latents included, and the two use the same VAE — a separate name would
discard latents that are still valid. So **latents cached by an `anima` run are reused as
is**. The one caveat: latents are not fingerprinted by VAE, so pointing `anima_refiner` at a
*different* VAE than the run that filled the cache would silently reuse the wrong ones. Pass
`--regenerate_cache` when changing VAE.

At `max_text_length = 512`, 3M images cache to roughly 6TB of text embeddings in bf16 at 2048
dims. Set `cache_text_embeddings = false` to run the text encoder inside the training loop
instead: no disk cost, about 4.5GB more VRAM, slower steps.

`padding='max_length'` is mandatory, not a default worth changing. Pipeline parallelism sends
fixed-size tensors between stages, and `lumina_2.py` and `z_image.py` both carry comments
about having disabled dynamic padding for this reason. Short captions therefore cost the same
as long ones, so `max_text_length` is worth setting to fit the data.

## Distillation

```
python -m tools.distill_refiner --config examples/anima_refiner_distill.toml
```

Captions only. No images, no VAE, no diffusion, roughly 6GB of VRAM. It teaches the new
refiner to reproduce what Anima's existing `llm_adapter` already emits, which is by definition
what the frozen DiT knows how to read.

The teacher is always a **stock Anima checkpoint** — it is the only thing that has an
`llm_adapter`. It is entirely separate from the model being trained, which is what lets you
come back and distil again later: point `student.resume_from` at any refiner you already have,
including one inside a full `model.safetensors` saved by another mode. When resuming, the
refiner's layer count comes from those weights, not from `n_refiner_layers`, matching how the
training pipeline resolves it.

By default this writes `context_refiner.safetensors`, the refiner alone. Set
`save_full_model = true` to also write a complete `model.safetensors` — the teacher's DiT
with the refiner attached and `llm_adapter` dropped — so distillation produces the same kind
of artefact every other mode does. The DiT weights are untouched, so it is exactly equivalent
to pairing the Anima checkpoint with `context_refiner_path`.

### Where the captions come from

Set exactly one of three sources under `[distill]`:

| Key | What it is |
| --- | --- |
| `dataset` | The ordinary `dataset.toml`, resolved by the same rules `DirectoryDataset` applies (`captions.json` first, then a matching `.txt`, then `skip_empty_caption`) with the same `caption_prefix`. Images are enumerated but never opened. |
| `caption_corpus` | A file from `tools/export_caption_corpus.py`: every caption of that same dataset, flattened once. |
| `captions` | A bare file of one caption per line, or a directory of `.txt` files, for when there is no `dataset.toml` to hand. |

`num_repeats` is ignored unless `apply_num_repeats = true`: repeats rebalance how often
*images* are sampled, which means little for captions alone.

### The caption corpus

```
python tools/export_caption_corpus.py --dataset examples/dataset.toml --output captions.jsonl
```

Reading captions straight from a `dataset.toml` stays exactly in step with the diffusion
stages, but it walks every media file and opens every tar on every run, purely to discover
which captions exist. For a few million images that is worth doing once. The corpus is a cache
of that walk, not a different source of truth — the same strings come out either way.

Format follows the extension:

| | |
| --- | --- |
| `.jsonl` | One JSON object per line. Prefer this: it streams, needs no quoting rules, and a corrupt line is obviously a corrupt line. |
| `.csv` | The same fields as columns, for spreadsheets. |
| `.txt` | One caption per line, with newlines and backslashes escaped so a caption containing a line break survives the round trip. |

`--dedupe` collapses identical captions and records how many times each occurred; reading
expands the count back out, so the sampling distribution is unchanged either way. A `.txt`
corpus has nowhere to put a count, so it repeats the line instead.

Captions are written **exactly as the dataset holds them**: tag marker included, no
`caption_prefix`, no shuffling. Those are training-time concerns, which keeps one corpus usable
by runs configured differently and lets every epoch see a fresh variant.

The prefix in particular must *not* be baked in. Training builds a caption as
`caption_prefix + augment(strip_marker(raw))`, so a stored `"anime, Special: red, blue"` no
longer starts with the marker: distillation would fail to strip it, silently skip augmentation,
and train the marker as though it were a tag.

The exporter prints the markers it found and the exact `prefix_tag_caption` line to add under
`[distill]`. A dataset that annotates its directories differently is fine — `prefix_tag_caption`
accepts a list, which is the only way a flat corpus can represent several markers.

`num_repeats` is not a corpus field. The dataset applies it per directory as
`int(len(directory_captions) * num_repeats)`; stored per caption, any value below 1 would floor
to zero and delete captions outright. Use `--apply-num-repeats` at export, which expands it with
the dataset's own semantics. Setting `apply_num_repeats` under `[distill]` with a corpus source
warns and does nothing.

### Caption augmentation

`shuffle_tags` / `cache_shuffle_num`, `cache_shuffle_delimiter`, `tag_dropout_rate` and
`prefix_tag_caption` (see [Multi-caption and tag augmentation](../README.md#multi-caption-and-tag-augmentation))
all work here. Two differences from the diffusion stages:

- **It is applied per sample, not baked into a cache.** Distillation re-embeds the text every
  step, so there is no embedding cache to invalidate and no reason to freeze a fixed set of
  variants. Each epoch sees a different tag order and a different dropout draw.
- **Settings fall back to the `dataset.toml`.** Anything left unset under `[distill]` is read
  from the dataset config's top level when `dataset` is the source, so a distillation run
  matches the diffusion stages without restating them. Only top-level keys: a batch here is a
  flat sample of captions with no directory attached, so per-directory overrides must be
  restated under `[distill]` if they matter.

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
diffusion-loss mode.

## Learning rates

Every mode uses the same parameter groups; `lr = 0` freezes a group and drops it from the
optimizer entirely.

```toml
base_lr       = 0      # x_embedder, t_embedder, positional embeddings, final_layer
self_attn_lr  = 0
cross_attn_lr = 0
mlp_lr        = 0
mod_lr        = 0
refiner_lr    = 1e-4   # the only thing that trains in refiner_only
```

`refiner_crossattn` raises `cross_attn_lr` off zero. A frozen DiT can only understand the
vocabulary the previous frontend spoke, so if the point is to exploit what a stronger text
encoder knows, the cross-attention has to learn to read it — with the refiner already trained,
the best case for a fully frozen DiT is parity with base Anima, not an improvement.

Only `k_proj` and `v_proj` really matter there: they project the text side, which is what
changed. `q_proj` and `output_proj` work on the image side. `get_param_groups` matches on
`.cross_attn`, so `cross_attn_lr` covers all four; to restrict it to k/v, split the group in
`models/cosmos_predict2.py`:

```python
elif '.cross_attn.k_proj' in name or '.cross_attn.v_proj' in name:
    cross_attn_kv_params.append(p)
```

Keep `cross_attn_lr` roughly 10x below `refiner_lr`. Structural knowledge lives in
`self_attn`, `mlp` and `final_layer`; leaving those frozen protects it.

## LoRA and LoKr

LoRA targets `Block` and `ContextRefiner` -- the latter rather than `RefinerBlock`, because
`get_target_modules` only walks the matched module's own Linears, and `cap_embedder` hangs
off `ContextRefiner`. A low-rank update needs something to build on, so
either point `context_refiner_path` at a trained refiner, or set `train_context_refiner = true`
in `[adapter]`, which excludes the refiner from the adapter, trains it densely alongside, and
saves it as `context_refiner/context_refiner.safetensors` -- in a subdirectory, because
`load_adapter_weights` globs `*.safetensors` in the save dir and raises on more than one
match, so a second file beside the adapter would break `init_from_existing`.

Block swapping requires an `[adapter]` block (`train.py` asserts this), so it is available for
LoRA and LoKr but not for the frozen-DiT modes or a full fine tune. Use `pipeline_stages`
across GPUs there instead.

## OPLoRA

OPLoRA works as before, on the DiT's LoRA layers. The refiner is excluded from it: OPLoRA
projects a low-rank update away from the top-k singular subspace of the **pretrained** base
weight, and a newly built refiner has no pretrained weights to protect. Two of its six Linear
layers per block are zero-initialised as well, where that subspace is whatever arbitrary
orthonormal basis the SVD happens to return — projecting against it would cost the adapter
`rank` directions for nothing. This does not fail loudly, which is why it is worth stating.

The exclusion is per-model (`oplora_exclude_names`), empty for every other model, so nothing
else changes.

## Activation checkpointing

`activation_checkpointing = true` uses `torch.utils.checkpoint` with `use_reentrant=False`.
Unsloth is used only when `activation_checkpointing = 'unsloth'` is written explicitly; there
is no automatic fallback to it. All the example configs here use `true`.

## Sampling

```
python -m tools.sample_anima_refiner \
    --config examples/anima_refiner_refiner_only.toml \
    --prompt '1girl, solo, blue eyes' \
    --steps 30 --cfg 5 --output out.png
```

ComfyUI's `Anima` class has no `context_refiner`, so a model trained here cannot be sampled
there yet. This script fills the gap. It reads the `[model]` table of a training config and
loads through `CosmosPredict2Pipeline` — the same class and the same checkpoint-resolution
rules training uses — then runs the layer stack `to_layers()` builds. There is no second copy
of the loading logic to drift out of sync.

Sampling is rectified-flow Euler, inverting what `prepare_inputs()` constructs: training
builds `noisy = (1-t)*clean + t*noise` with target `noise - clean`, so the model predicts a
velocity that is integrated from `t=1` down to `t=0`. `--shift` applies the same
reparametrisation the training timestep sampling uses, defaulting to the config's `shift`.

## Config reference

Every key below is read only when `type = 'anima_refiner'`.

| Key | Default | Meaning |
|---|---|---|
| `type` | — | `'anima_refiner'` |
| `transformer_path` | — | DiT checkpoint; carries the refiner once one has been trained |
| `llm_path` | — | Transformers folder or single safetensors file |
| `llm_config_path` | `configs/qwen3_5_2b_base` | config/tokenizer dir when `llm_path` is a file |
| `llm_repo_id` | — | Hub repo to fetch config/tokenizer from instead |
| `llm_hidden_layer` | `None` (last) | index into `hidden_states` |
| `n_refiner_layers` | `6` | refiner blocks. Used only when building a fresh refiner; **omit it** when the checkpoint carries one, because a value that disagrees with the weights raises |
| `max_text_length` | `512` | padded token length |
| `context_refiner_path` | — | separate refiner file; overrides the checkpoint's, with a warning |
| `cache_text_embeddings` | `true` | cache to disk, or run the encoder inline |
| `base_lr` | optimizer `lr` | lr for otherwise unmatched parameters |
| `refiner_lr` | `base_lr` | lr for the refiner |
| `train_context_refiner` | `false` | in `[adapter]`: train the refiner densely |

### Distillation config (`[distill]`)

| Key | Default | Meaning |
|---|---|---|
| `dataset` | — | a `dataset.toml`; captions read through the training rules |
| `caption_corpus` | — | a file from `tools/export_caption_corpus.py` |
| `caption_corpus_format` | from extension | `jsonl`, `csv` or `txt` |
| `captions` | — | bare file of one caption per line, or a directory of `.txt` |
| `apply_num_repeats` | `false` | honour the dataset's `num_repeats`; `dataset` source only |
| `shuffle_tags` / `cache_shuffle_num` | from `dataset.toml` | shuffle tag order, per sample |
| `cache_shuffle_delimiter` | `', '` | tag separator |
| `tag_dropout_rate` | from `dataset.toml` | drop each tag with this probability, per sample |
| `prefix_tag_caption` | markers found in the `dataset.toml` | string or list of markers for tag captions; matched case-insensitively and stripped before training |
| `caption_prefix` | from `dataset.toml` | prepended after the marker is stripped |

Exactly one of `dataset`, `caption_corpus` and `captions` may be set; setting more than one is
an error rather than a silent precedence rule.

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
