# Lessons from adding Anima Refiner

Mistakes made while building the `anima_refiner` architecture, written down so the next
change to this codebase does not repeat them. Every item below is something that actually
shipped or was caught mid-implementation, not a hypothetical.

## Adding a model must not change any existing model

`models/cosmos_predict2.py` serves three model types (`cosmos_predict2`, `anima`,
`anima_refiner`), and the dataset/caching code is shared by every model in the repo. Any new
option belongs behind a check on the model type, not in a shared default.

Mistakes actually made while adding `anima_refiner`, all of which shipped before being caught:

- **`base_lr` and `max_text_length` were read unconditionally**, so `anima` and
  `cosmos_predict2` silently gained config keys nobody asked for. Now gated on
  `use_context_refiner`.
- **The VLM detection branch was added to the shared `llm_path` directory path**, changing how
  `anima` would load a vision-language model. Now gated too.
- **A new fingerprint argument was nearly added unconditionally** to
  `_cache_text_embeddings`, which would have invalidated the text embedding cache of *every*
  model in the repo. `Hasher.hash([i])` and `Hasher.hash([i, ''])` differ. Extra fingerprint
  args must only be appended when they are actually non-empty.

Before committing a change to a shared file, diff it and check every modified line against the
other models that use it.

## Caching: latents and text embeddings are separate, keep them that way

`DirectoryDataset.cache_dir` is `<dataset>/cache/<model.name>`, and **both** latents
(`latents_` prefix) and text embeddings (`text_embeddings_{i}_` prefix) live under it.

A `cache_name` config key was added to invalidate stale text embeddings when the text encoder
changed. That was wrong: it changed `model.name`, which moves the *whole* tree, so switching
text encoder also threw away the VAE latents — by far the more expensive half, for a component
that had not changed.

The right hook is the per-cache fingerprint (`new_fingerprint_args` in `_map_and_cache`), which
is already separate for latents and text embeddings. Invalidate the narrowest thing that
actually changed.

## Derive model shape from weights, never from config

`load_diffusion_model` loads with `if name not in state_dict: continue`. That means a config
declaring *fewer* layers than the checkpoint holds will build the smaller model and **silently
drop the surplus weights** — no warning, no error, just a quietly wrong model. This was a real
bug in the first version of `anima_refiner`: a 4-layer refiner checkpoint loaded under the
default `n_refiner_layers = 6` lost 24 tensors.

When a checkpoint carries a component, derive that component's shape from the weights and treat
a contradicting config as an error. `get_dit_config()` already does this for the DiT; follow it.

## Checkpoint priority

Whatever weights the user points at are the weights used. Never re-initialise from a base model
when real weights exist. When two sources overlap (a component file plus a full checkpoint that
also contains that component), pick one deliberately, document it, and **warn every time it
happens** — silently choosing is how someone ends up training from the wrong starting point for
a week.

## Don't invent formats that already exist

`tools/distill_refiner.py` originally took its own caption format (a text file or a folder of
`.txt`). Every other training mode used `dataset.toml`. A tool that needs captions should read
`dataset.toml` through `utils.dataset.enumerate_captions()` so it sees the same captions
training will see — same `captions.json`/`.txt` resolution, same `caption_prefix`, same tag
shuffling. Skip only the parts that genuinely do not apply (this tool never opens an image).

## Example configs are documentation

All example configs shipped with `activation_checkpointing = 'unsloth'`, copied from another
model's example, when the user had not asked for it and the repo default is `False`. Two things
follow:

- Don't copy settings between example configs without checking they are wanted.
- `activation_checkpointing = true` uses `torch.utils.checkpoint` with `use_reentrant=False`;
  `'unsloth'` is only ever used when written explicitly. There is no automatic fallback.

Also: a hard-coded path from one example to another (`stage2` pointing at `stage1`'s output)
implies a one-way dependency the code does not have. Say so explicitly when the order is a
recommendation rather than a constraint.

## Base vs Instruct models

Use the **Base** model when the encoder is a pretrained LM, not an instruction-tuned one. For
Qwen3.5-2B the `config.json` of Base and Instruct is byte-identical, so the architecture needs
no code change and the mistake is invisible in the model code — but the bundled tokenizer
differs (`eos = <|endoftext|>` vs `<|im_end|>`). Download config/tokenizer from the exact repo
the weights come from.

## Docs must say when a file exists

`context_refiner.safetensors` is produced by only two of the six modes; the rest embed the
refiner inside `model.safetensors`. Documenting the option without saying when the file exists
led to real confusion about whether a step had been missed. For any optional artefact, document
which runs produce it and which do not.

## Testing on a CPU-only machine

`test/conftest.py` stubs `comfy_aimdo` and forces ComfyUI to CPU, but **only** when the real
package is missing and CUDA is unavailable, so it does nothing in a real training environment.
Without it `models/*.py` cannot even be imported off-GPU.

`pytest test/` runs everything on CPU with no downloads. Multi-GPU behaviour is covered
indirectly by asserting the invariants pipeline parallelism depends on (constant tensor shapes
across micro batches; every layer boundary being a valid split point). Real DeepSpeed execution
needs `deepspeed --num_gpus=2 --module test.debug_deepspeed_init`.

## Git

Commits carry no Claude attribution: no `Co-Authored-By: Claude ...`, no `Claude-Session:`
trailer.
