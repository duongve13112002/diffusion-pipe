# From Anima to Anima Refiner, step by step

One recommended path, start to finish. It takes a stock Anima checkpoint and a folder of images
and ends with a sampled image from a model whose text encoder is Qwen3.5-2B-Base instead of the
T5-shaped `LLMAdapter`.

The modes are configurations, not a pipeline the code enforces — any one's output can feed any
other, in any order. The order below is recommended because of *results*, not because anything
breaks otherwise. Where a step can be skipped, it says so.

See [README.md](./README.md) for the architecture and the full config reference.

## What you need before starting

| | |
|---|---|
| A stock Anima checkpoint | The only thing that has an `llm_adapter`, so it is the only possible distillation teacher |
| Qwen3.5-2B-Base | Base, not Instruct. `Qwen/Qwen3.5-2B-Base` |
| A Cosmos-Predict2 / Qwen-Image VAE | `vae_path` |
| Images with captions | `.txt` sidecars or a `captions.json` per folder |
| `configs/qwen3_5_2b_base/` | Ships with the repo; only needed if `llm_path` is a single safetensors file rather than a Transformers folder |

Every command below runs from the repo root.

## Step 0: point the configs at your files

Copy `examples/anima_refiner/` somewhere and edit the paths. Every config needs `vae_path` and
`llm_path`; `transformer_path` changes at each step and is called out below.

Start from `examples/anima_refiner/dataset.toml`, which has three directories and the caption
settings already filled in. Set `path` on each `[[directory]]` and delete the ones you do not
need.

## Step 1: distil the refiner from Anima's adapter

```
python -m tools.distill_refiner --config examples/anima_refiner/distill.toml
```

Three paths matter here. The teacher is the whole stock Anima setup, so it needs both halves:

| Key | Value |
|---|---|
| `teacher.transformer_path` | the stock Anima checkpoint |
| `teacher.llm_path` | Qwen3-0.6B — the encoder Anima's `llm_adapter` was trained against, not the new one |
| `student.llm_path` | Qwen3.5-2B-Base |

Getting `teacher.llm_path` wrong is the easy mistake: the teacher has to be reproduced exactly as
it was trained, or the target the student is chasing is not the one the DiT can read.

Captions only. No images, no VAE, no diffusion. It teaches a freshly
initialised `ContextRefiner` to reproduce what Anima's `llm_adapter` already emits, which is by
definition what the frozen DiT knows how to read.

**Produces** `context_refiner.safetensors` — the refiner alone, about 77M parameters. Set
`save_full_model = true` to also get a `model.safetensors` with the refiner attached and
`llm_adapter` dropped, which is the same thing as pairing the Anima checkpoint with
`context_refiner_path`.

**Skip it if** you already have a trained refiner from anywhere. Step 2 can start from a random
init; it will just take much longer to get anywhere, because the loss starts high and every step
needs a full DiT forward pass.

**Do not expect a finished model.** Distilling perfectly would reproduce the 0.6B model's
information content and throw away everything the larger encoder knows. It is a warm start.

### Faster distillation on a large dataset

Distillation needs captions and nothing else, but reading them from a `dataset.toml` walks every
image file and opens every tar to find them. At a few million images that is worth doing once:

```
python tools/export_caption_corpus.py \
    --dataset examples/anima_refiner/dataset.toml \
    --output captions.jsonl
```

Then set `caption_corpus = 'captions.jsonl'` under `[distill]` instead of `dataset`. The same
captions come out either way. If your dataset uses `prefix_tag_caption`, the exporter prints the
exact line to add — the markers stay in the corpus on purpose, and whatever reads the file has
to strip them.

## Step 2: train the refiner against the frozen DiT

```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 \
    train.py --deepspeed --config examples/anima_refiner/refiner_only.toml
```

Set `transformer_path` to the Anima checkpoint and `context_refiner_path` to the
`context_refiner.safetensors` from step 1. If step 1 was run with `save_full_model = true`, point
`transformer_path` at that `model.safetensors` and leave `context_refiner_path` unset.

This is the first step with a diffusion loss, so it is the first that needs images and a VAE. The
DiT is frozen: only `cap_embedder` and the refiner blocks move.

**Produces** `model.safetensors` with the refiner inside it, under `net.context_refiner.*`.

**Why before step 3.** Unfreezing the DiT while the refiner still emits noise is what causes
catastrophic forgetting. Loss is highest at the start, so gradients are largest exactly when the
signal is least meaningful. Letting the refiner adapt one-way first means the gradients reaching
the DiT later are already small.

**How to know it is working.** Loss should fall well below where it started and flatten. Sample
from it (step 5) — the images will be poor, but they should be *images*, not noise. If they are
noise, the refiner is not yet producing something the cross-attention can read, and step 3 will
make it worse rather than better.

## Step 3: open the cross-attention

```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 \
    train.py --deepspeed --config examples/anima_refiner/refiner_crossattn.toml
```

Set `transformer_path` to the `model.safetensors` from step 2. Leave `context_refiner_path`
unset — the refiner is already in there, and pointing at an older file would override it (with a
warning).

Now the DiT's cross-attention trains too, so the two halves can meet instead of the refiner doing
all the adapting. Self-attention and the feed-forward blocks stay frozen.

**Produces** `model.safetensors`. This is the checkpoint to keep. Everything after this point is
optional.

**Skip it if** you only want a small adaptation and step 2 already samples acceptably.

## Step 4: style or subject training

At this point it is an ordinary model. Pick one:

| Config | Use it for |
|---|---|
| `lora.toml` | The usual choice. Small, composable |
| `onfly_lora.toml` | Same, with on-the-fly text embeddings and per-access caption augmentation. See below |
| `lokr.toml` | Higher capacity than LoRA at similar size |
| `full_finetune.toml` | Every DiT parameter. Needs the most VRAM |

Set `transformer_path` to the `model.safetensors` from step 3.

A LoRA is a low-rank update to weights that already exist, so it has nothing to build on if the
refiner is still freshly initialised. Coming from step 3 that is not a concern. Starting from a
stock Anima checkpoint instead, set `train_context_refiner = true` in `[adapter]`, which trains
the refiner densely alongside the adapter and saves it separately under `context_refiner/`.

**Produces** `adapter_model.safetensors`.

### When to use on-the-fly text embeddings

`cache_text_embeddings = false` keeps the text encoder resident and runs it every step. It costs
VRAM and time. What it buys is caption augmentation re-drawn on every access instead of frozen
into the cache.

That matters if you use `tag_dropout_rate`. With cached embeddings, `cache_shuffle_num` fixes how
many tag orders and dropout draws exist for the whole run — and `tag_dropout_rate` with
`cache_shuffle_num = 0` gives a *single* permanent draw, which deletes those tags rather than
augmenting them. The code warns when you do that.

The alternative, if you want to keep cached embeddings: raise `cache_shuffle_num` so several
draws are cached, and set `caption_sampling = "random_per_epoch"` so a different one is used each
epoch. VAE latents are cached either way; that is the expensive half and it does not depend on
the caption.

## Step 5: sample

```
python -m tools.sample_anima_refiner \
    --config examples/anima_refiner/refiner_crossattn.toml \
    --prompt '1girl, solo, blue eyes' \
    --steps 30 --cfg 5 --output out.png
```

ComfyUI's `Anima` class has no `context_refiner`, so a model trained here cannot be sampled there
yet. This script reads the `[model]` table of any training config and loads through the same
pipeline class training uses, so there is no second copy of the loading logic to drift.

Point `--config` at whichever step's config matches the checkpoint you want to sample. For a LoRA
run, the adapter is applied on top of that config's `transformer_path`.

## Where the refiner lives at each step

| After | File | Refiner is |
|---|---|---|
| Step 1 | `context_refiner.safetensors` | that file, alone |
| Step 1 with `save_full_model` | also `model.safetensors` | inside it, `net.context_refiner.*` |
| Step 2, 3, full fine tune | `model.safetensors` | inside it, `net.context_refiner.*` |
| Step 4 LoRA / LoKr | `adapter_model.safetensors` | inside it, as adapter tensors |
| Step 4 with `train_context_refiner` | also `context_refiner/context_refiner.safetensors` | the separate file |

One rule governs loading: **whatever weights you point at are the weights that get used.**
Nothing is re-initialised from Anima when real weights exist. `context_refiner_path` overrides a
refiner already inside `transformer_path`, and says so in a warning.

Most of the time `context_refiner_path` is unset, because the refiner is already inside
`transformer_path`. That is the normal case, not a missing step.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `n_refiner_layers` raises on startup | The checkpoint carries a refiner and its layer count disagrees with the config. Remove the key; the count is read from the weights |
| Samples are noise after step 2 | The refiner has not converged. Do not proceed to step 3 — more distillation or more step-2 steps first |
| Quality collapses during step 3 | Step 2 was too short. The DiT is being pulled by a refiner that still emits noise |
| Warning about `context_refiner_path` overriding | You passed an older refiner file alongside a checkpoint that already has one. Usually you want to drop the path |
| Tag dropout seems to do nothing useful | `cache_shuffle_num = 0` with cached embeddings gives one frozen draw. See step 4 |
| Captions look wrong in distillation | If the dataset uses `prefix_tag_caption`, a corpus keeps the markers and `[distill]` must be told about them |
