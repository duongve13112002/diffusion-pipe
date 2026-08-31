# Design notes

Why the distillation stage is shaped the way it is, where the numbers came from, and what four
rounds of review found wrong. Written for whoever changes this next.

## Why distil at all, and why against the cross-attention output

The DiT is frozen and was trained to read whatever Anima's `LLMAdapter` emits. A freshly
initialised `ContextRefiner` emits something else entirely, so the first diffusion step sees a
huge loss and pushes large gradients through a text frontend that has no idea what it is doing
yet. That works eventually — it just costs images, a VAE encode and a full DiT forward for every
step of a long warm-up.

Distillation replaces that warm-up with a text-only objective: make the refiner emit what the
adapter already emits. Captions only, no images, no VAE, no diffusion.

The obvious way to compare them does not work. Both sides are `(B, L, 1024)`, so a position-wise
MSE runs without error — and plateaus. The teacher's sequence is indexed by **T5** tokens,
because the `LLMAdapter` embeds T5 token ids as its query sequence; the student's is indexed by
Qwen's own BPE tokens. Position `i` means different things on each side, and no amount of
training reconciles two different tokenizations position by position.

Cross-attention is a weighted sum over text positions, so its output does not depend on how the
text is indexed. Pushing both feature sets through the DiT's own frozen cross-attention and
comparing *there* sidesteps the mismatch, and it optimises exactly the quantity the DiT
consumes. Nothing else about the two sequences has to line up.

## Where the numbers come from

### `n_refiner_layers = 6`

From Lumina 2 and Z-Image, which is also where the whole `cap_embedder` + refiner-blocks recipe
comes from. Nothing here was invented; the point of the architecture is to be the frontend those
models already use.

It is a default, not a constraint, and it is used **only when building a fresh refiner**. Once a
checkpoint carries a refiner, the layer count is read from the weights, and a config value that
disagrees raises rather than silently dropping tensors — an earlier version of this branch
quietly discarded 24 tensors when a 4-layer checkpoint met the default of 6.

At production size (`cap_feat_dim=2048`, `model_dim=1024`, 6 layers, 16 heads) that is **77.64M
parameters**: 50.36M in the MLPs (64.9%), 25.17M in attention (32.4%), 2.10M in `cap_embedder`
(2.7%).

### `num_blocks` — two different things with the same name

**`dit_config["num_blocks"]`** is the DiT's depth and comes from the checkpoint, never from a
config here. It used to be hardcoded per `model_channels` (28 for the 2048-channel variant Anima
is); upstream now derives it by counting `blocks.N.` keys, which also means a refiner in the same
file cannot confuse it — its blocks are `context_refiner.blocks.N.`, verified as 28 and 6
counted separately.

**`[probe] num_blocks = 8`** is a distillation setting, and it is the one worth explaining. The
loss is measured through the DiT's frozen cross-attention, but there is no need to push through
all 28 of them: adjacent blocks give highly correlated signal, so a spread covers the same ground
for a fraction of the compute. The script takes 8 blocks evenly strided across the stack. Raise
it if you suspect later blocks read the text differently from earlier ones; lower it to trade
signal for speed.

### `num_queries = 64` (`[probe]`)

This one is a genuine judgement call, so here is the reasoning rather than a citation.

The loss is measured by pushing both feature sets through the frozen cross-attention modules.
Cross-attention needs a query sequence, and during distillation there is no image to supply one.
So the script uses a **fixed set of random query vectors** as a measuring stick:

- **Random** because any query set works. The cross-attention weights are frozen and shared by
  both paths, so the comparison is fair whatever the queries are.
- **Fixed** (seeded once, reused every step) because a resampled probe would make the objective
  non-stationary — the loss would move when the probe moved, not only when the student
  improved.
- **64** because matching the output for many independent random queries is a strong proxy for
  matching the key/value content itself, and 64 is enough independent directions to make
  accidental agreement unlikely while staying cheap. It is a hyperparameter, not a derived
  quantity: raise it if the loss looks too easy to satisfy.

An auxiliary term compares the length-normalised mean of the two feature sets directly. It uses
`padded_mean` — normalising by the same padded length the attention softmax does — because a
mask-normalised mean would disagree with the attention term about what a "position" is.

## What the review passes found, and why

Four rounds. Each found things the previous one missed, including in code the previous round had
read. That pattern is the most useful thing on this page.

### `init_weights()` that only set the overrides

The worst bug on the branch, and it was in the first commit. `ContextRefiner.init_weights()` was
written to run *after* `__init__`, so it set only the ten values that differ from PyTorch's
defaults. But the pipeline builds the module under `init_empty_weights()` and materialises every
parameter with `torch.empty`. The other eighteen kept allocator residue — measured up to 1e32,
sometimes NaN.

Two things hid it. The zeroed residual branches mean the garbage does not reach the output until
training moves them off zero, so a smoke test looks fine. And `tools/distill_refiner.py`
constructs the module normally before calling `init_weights()`, so the one path with a test
covering it was the one path that worked.

**Rule:** a function that runs where default init did not must cover every parameter. Delegate to
each submodule's `reset_parameters()` rather than hand-rolling the bounds.

### Caption settings that did not reach the cache path

The metadata cache is keyed by directory and model name; `--trust_cache` reuses it without
looking. Adding settings that change caption *text* meant a run that flipped one read back
captions built under the other — raw text with the tag marker intact going into the text
encoder, or `caption_prefix` applied twice and then shuffled into the middle of the tag list.

**Rule:** a setting that changes cached content belongs in the cache path. And the suffix must be
empty at the defaults, or every existing install loses its cache.

### An unseeded draw that wiped the VAE cache every run

The captions are a column of `metadata_dataset`, and the **latent** cache is keyed by that
dataset's fingerprint. `shuffle_captions` drew unseeded, so the captions differed on every
launch, and the entire dataset was re-encoded through the VAE on every single run. Pre-existing
for `cache_shuffle_num`; `tag_dropout_rate` inherited it.

Nothing was gained by the randomness: the variants are frozen into the text embedding cache
anyway, so redrawing them per run never produced fresh augmentation, only a rebuild.

**Rule:** anything that feeds a cache key must be reproducible.

### A claim asserted in docs before it was measured

The docs, three commit messages and a docstring all said "the same draw selects the caption text
and its embedding, so the two can never disagree." Measured: 8 of 9 rows disagreed. The
iteration-order builder shuffled each image's caption list and then used the post-shuffle
position to index an embedding cache built in the original order.

The bug pre-dated the branch. The claim did not — and a claim is what turns a latent bug into a
relied-upon one.

**Rule:** do not assert a property you have not measured, especially about code you did not
write.

### A green test suite that proved nothing

Extracting the caption helpers by *text range* silently moved `bucket_suffix`,
`dedup_and_sort` and `seed_from_hash` out of `utils/dataset.py`. Twelve call sites broke; every
training run for every model died at dataset construction. 182 tests passed, because no test in
the repo had ever constructed a `DirectoryDataset`.

**Rule:** move code by name, never by line range. And before trusting a suite on a change to
shared code, ask which test would fail if the change were wrong. If the answer is none, that is
the test to write.

## What is still unverified

No training run, no multi-GPU run, no image has ever been produced from this branch. Everything
is covered by CPU tests, including a real optimisation loop and sampling checked against a
synthetic velocity field where the exact answer is known, but that is not the same thing.

`utils/cache.py` opens sqlite with `autocommit=`, which needs Python 3.12, so `cache_latents`
cannot run on a 3.11 box at all — the iteration-order paths are tested through the pure function
`collapse_to_one_entry_per_image` and through the cache path derivation, not through a real
caching run.

`blocks_to_swap` with `cache_text_embeddings = false` is asserted against rather than fixed,
because verifying a device move needs a GPU.
