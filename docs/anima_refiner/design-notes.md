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

## What the probe objective does and does not constrain

Cross-attention output is a weighted sum over text positions, which is exactly what makes the
teacher/student comparison legal across two different tokenizations. It also bounds what the
objective can see. Expanding the softmax in the regime the padding creates, what the loss
actually compares is the sums over positions -- and a sum does not care which token holds which
content. A student that smears the same average content across all of its positions satisfies it
about as well as one that puts the right content in the right place.

That is not a hypothetical. *Scaling Down Text Encoders of Text-to-Image Diffusion Models*
(CVPR 2025) reports exactly this failure for naive text-encoder distillation: the student's
embedding space collapses, and "rat", "cat" and "man" come back as the same embedding.

Two things follow, and both are now in the code.

**A relational term.** `relational_loss` matches the teacher's *pairwise distance structure*
within the batch, which is Relational Knowledge Distillation's distance-wise loss. Collapse
destroys that structure by construction, so the term prices it directly while the probe loss
cannot see it at all. It costs nothing -- both feature sets already exist for the probe -- and
needs no images, so the stage keeps the property that justifies its existence.

One deviation from the textbook form is deliberate. RKD normalises each side by its own mean
distance, which makes the loss scale invariant; uniform shrinkage toward a centroid is exactly a
scale change, so that formulation scores 0.0000 at every collapse fraction from 25% to 90%.
Both sides are divided by the *teacher's* mean here instead, which keeps the comparison unit-free
while leaving shrinkage visible: 0.0277 at 25% collapsed, 0.1106 at 50%, 0.4396 at 100%.

**A diagnostic.** The progress bar reports `spread`, the mean pairwise cosine distance between
the batch's student features, against the teacher's. It should track the teacher's number.
Falling toward zero is collapse, and it is the only cheap way to see it happening.

### What is still not addressed

The published remedy goes further: push both feature sets through the frozen diffusion model and
compare its *predictions*, over a short denoising trajectory started from pure noise. That keeps
distillation image-free -- no VAE, no dataset of images -- which is why it fits here in
principle. It is not implemented, because `build_teacher` deliberately discards the DiT after
taking its cross-attention modules (`dit.blocks = None`), and keeping the whole DiT resident
would cost several GB and undo the ~6 GB figure this stage advertises. It is the right next step
if the relational term and the spread diagnostic show collapse is still happening.

There is also an irreducible floor, measured at roughly 0.3 relative RMS, that comes purely from
the teacher and student having different token counts: a student sequence cannot reproduce a
teacher sequence's output for every possible query. **A plateau in the distillation loss is
therefore expected behaviour, not a failure.**

## The unconditional embedding

An empty caption is not an edge case here: `uncond_fraction` produces them during training and
every CFG sample uses one. Qwen pads with its own eos and adds no bos, so `''` tokenizes to a
row whose attention mask is entirely zero.

That row is degenerate all the way down. `_compute_text_embeddings` zeroes every position, the
refiner masks its output to zero, and the DiT cross-attention carries no attention mask at all
-- it relies on padded keys being exactly zero, which they are, because `k_proj` has no bias.
So the output is identically zero for every query. Two consequences follow: those samples
deliver **no gradient to the refiner**, because the output does not depend on any of its
parameters, and at sampling time the frozen DiT is handed a context its original training never
produced.

Old T5, which this DiT was trained against, never emits that row. An empty string still yields
`</s>` -- one real token, with a real embedding. `_tokenize(..., keep_one_real_token=True)`
reproduces that by marking position 0 real, and it is applied in `_tokenize` rather than at each
call site so the caching path, the on-the-fly path and the sampler cannot drift apart. It is off
by default, because `anima` and `cosmos_predict2` share that helper and their T5 query sequence
already has the property.

## What is verified, and what is still not

Measured 2026-08-31 on Windows 11, Python 3.12.10, torch 2.13.0+cpu, no GPU. The suite is 320
passed / 1 skipped there.

### Verified since this section was first written

`cache_latents` now runs for real. `utils/cache.py` opens sqlite with `autocommit=`, which needs
Python 3.12, so on the 3.11 box this was written for the path could not execute at all, and the
iteration-order code was only ever reached through the pure helper
`collapse_to_one_entry_per_image`. `TestLatentCachingRunsForReal` in `test/test_dataset_smoke.py`
drives the real thing with a stub in place of the VAE — the only genuinely GPU-shaped part —
covering `_map_and_cache`, the sqlite cache and the iteration-order directory, including that a
second trusting pass lands on the same directory name.

Both drift guards run and pass here: 34/34 vendored-API checks against torch 2.13.0 and
bitsandbytes 0.50.0, and 26/26 ComfyUI signature checks. The ComfyUI one needs
`PYTHONPATH=test/childenv` so the submodule can import without a CUDA device.

### Still unverified, and why

No training run, no multi-GPU run, and no image has ever been produced from this branch. CPU
tests cover a real optimisation loop, and sampling is checked against a synthetic velocity field
where the exact answer is known, but that is not the same thing.

`deepspeed.initialize` now runs here, and finding the bug below is what it was worth. Two
things stand between a CPU-only Windows box and a live engine, both surmountable. The sdist
omits `bin/deepspeed.bat`, which its own `setup.py` lists for win32, so unpack the sdist, add
that file and `bin/ds_report.bat`, then build with `DS_BUILD_OPS=0` and `--no-build-isolation`.
`initialize` then tries to JIT-build the `deepspeed_shm_comm` op and needs MSVC `cl.exe`, which
`build_shm_op()` lets you skip:

```python
deepspeed.ops.__compatible_ops__['ShareMemCommBuilder'] = False   # before initialize
```

`test/test_distill_refiner.py::TestZeROAccumulationBoundaryForReal` uses that to drive a real
ZeRO-1 engine on CPU. It is what caught the accumulation-boundary bug, which every mocked test
in the file had passed straight over.

What is still unverified is what needs real hardware: no multi-rank run, no GPU run, no training
run, and no image has been produced from this branch.

`blocks_to_swap` with `cache_text_embeddings = false` is asserted against rather than fixed,
because verifying a device move needs a GPU.
