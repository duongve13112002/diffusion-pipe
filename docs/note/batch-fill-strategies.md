# Batch fill strategies

**Date:** 2026-09-03

**Prompted by:** the pipeline silently discards samples in three places when a bucket's sample
count is not a multiple of the global batch size. The request was for an opt-in way to complete
the batch instead, without ever putting the same image in a global batch twice.

Sources are file paths at the commit this note was written against; every claim below was either
read out of the code or measured on the Windows CPU box (Python 3.12.10, torch 2.13.0+cpu).

## The three places data was dropped

| Where | What it did |
| --- | --- |
| [`utils/dataset.py`](../../utils/dataset.py) `ConcatenatedBatchedDataset._make_divisible_by` | Truncates each size bucket's `iteration_order` to a whole number of global batches. A bucket smaller than one global batch is emptied, with a warning. |
| [`tools/distill_refiner.py`](../../tools/distill_refiner.py) `EpochSampler` | `steps_per_epoch = len(captions) // global_batch`, and the caption tail past `usable` is not returned. Fewer captions than one global batch raises. |
| `train.py` eval datasets | Goes through the same `ConcatenatedBatchedDataset.post_init`, with `eval_gradient_accumulation_steps`. Eval sets are small, so a whole bucket is easy to lose. |

The first is worse than a one-off loss. `Dataset.post_init` and
`ConcatenatedBatchedDataset.post_init` build their iteration orders once, with
`shuffle_with_seed(..., 0)`, and `PipelineDataLoader` creates its `DataLoader` with no shuffling.
So the order is identical in every epoch, and the samples past the cut are the *same* samples
every epoch, for the life of the run.

## Why the tail is extended rather than `__len__` being changed

`ConcatenatedBatchedDataset.__len__` is `len(self.iteration_order) // self.global_batch_size` and
`__getitem__` slices `iteration_order` directly. Appending to `iteration_order` until it is a
multiple of the global batch leaves both of those correct with no change, and it keeps one
representation of "what this epoch trains on" instead of two. Any scheme that left the order
short and synthesised extra entries inside `__getitem__` would need `__len__`, `__getitem__`, the
resume path's linear indexing (`SkipFirstNSampler`) and the duplicate check to agree about
something that is not written down anywhere.

## What "no duplicate image" does and does not promise

Only the appended tail is constrained. The first N entries are exactly what `drop` would have
produced, so a duplicate that was already there is still there.

That is not hypothetical, and it is not caused by `num_repeats`. With the default
`caption_sampling = 'all'`, an image with three captions produces three rows in
`SizeBucketDataset.iteration_order`, all with the same `latents_idx`.
`ConcatenatedBatchedDataset.post_init` shuffles `(dataset_idx, j)` pairs, and nothing stops two
rows of one image landing in the same global batch. Measured on a synthetic bucket of 10 images
with 2 captions each and a global batch of 16: the first batch already held 6 pairs of rows
sharing an image before any fill ran.

Constraining the whole order would mean permuting all N entries with a batch-aware algorithm,
which changes the training order of every run that turns this on and is a much larger surface to
get wrong. The decision was to bound the change to the tail and write the limitation down here.

## Loss normalisation

### The mask is the carrier

Masked training already exists: `PreprocessMediaFile.__call__` reads a mask image into a float16
`(H, W)` tensor, `Dataset._collate` stacks them, and all eight `get_loss_fn` implementations
multiply it into the loss elementwise before averaging. A sample whose mask is identically zero
therefore contributes nothing to the loss and nothing to the gradient — which is exactly what a
padded sample needs. Reusing that channel is what keeps padding out of eight loss functions.

Every model interpolates the mask to the latent size
(`F.interpolate(mask, size=(h, w), mode='nearest-exact')`, verified in all 23 model files), so a
constant mask survives the trip unchanged and the pixel-space shape of a synthesised mask does
not affect the result. `_collate` still prefers a real mask's shape when the batch has one, so a
synthesised mask can never disagree with a real one.

### Why the weight is `G / G_real` and not 1

`loss.mean()` divides by the element count *including* the elements that were multiplied by zero.
With a global batch of 8 holding 6 real samples and 2 padded:

```
loss = (Σ_6 ℓ + 0 + 0) / 8 = 0.75 × (mean over the real samples)
```

Adam is largely invariant to a constant gradient scale, but `gradient_clipping` compares against
an absolute norm, `weight_decay` does not shrink with it, the other optimizers in this repo
(Prodigy, Automagic, GenericOptim) react differently, and the logged loss is simply wrong. So the
real samples carry `G / G_real` and the padded ones carry 0.

### Why the constant is global, and the proof

`ConcatenatedBatchedDataset._batch_weight_scale` computes it over the whole global batch, not over
one rank's slice and not over one micro batch. Write `W` for the data-parallel world size, `M` for
`gradient_accumulation_steps`, `b` for the micro batch size, so `G = b · M · W`, and let `s`
be the shared constant.

One rank's loss is the mean over its `M` micro batches, each of which is a mean over its `b`
samples:

```
rank loss = (1/M) Σ_m (1/b) Σ_{i∈m} s · ℓ_i · [i is real]
          = (s / (b·M)) Σ_{i on this rank, real} ℓ_i
```

Data parallel averages the ranks' gradients, so:

```
total = (1/W) Σ_ranks (s / (b·M)) Σ_{i on rank, real} ℓ_i
      = (s / (b·M·W)) Σ_{all real} ℓ_i
      = (s / G) Σ_real ℓ_i
```

With `s = G / G_real` that is `(1/G_real) Σ_real ℓ_i` — the mean over the real samples, which is
what a batch of `G_real` samples would have produced. Note the `1/W` step: a derivation that
stops at one rank has `b·M = G/W`, not `G`, and looks wrong.

A per-micro-batch ratio `b / b_real` does not survive this. It equals the global constant only
when the padding is spread evenly across micro batches and ranks, and it divides by zero on a
micro batch that is entirely padding — which is legal and which
`TestNormalisationSurvivesMicroBatching` in [`test/test_batch_fill.py`](../../test/test_batch_fill.py)
drives directly.

### Why not `(loss * mask).sum() / mask.sum()`

That is the textbook masked mean and it would make the compensation unnecessary. It is not used
because it changes what a *spatial* mask means. Today a mask that is mostly black makes the loss
for that image smaller, and everyone who has tuned a learning rate against masked training has
tuned it against that behaviour. Folding the weight into the mask leaves that semantics alone:
with no padding the weight is 1.0 and every number is what it was.

One model already does divide by the mask mean: `models/ltx_video.py:216` computes
`loss.mul(mask).div(mask.mean())`. The weight cancels between numerator and denominator there, so
that model is correct either way — checked, not assumed.

### The two loss functions that needed fixing

Six of the eight applied the mask to everything they summed. Two did not.

`models/cosmos_predict2.py` adds a term per reduced scale under `multiscale_loss_weight`, and
those terms were computed with a plain `F.mse_loss` after the masked term had already been
reduced. `models/minimax_h3.py` calls `single_loss(audio_output, audio_target)` with no mask at
all for the audio branch.

Neither shows up in the loss value. Measured on a 3-real / 13-padded batch with
`multiscale_loss_weight = 0.5`, the loss was 0.155659 against a reference of 0.155521 — 0.1%,
indistinguishable from noise, because the padding is a copy of real samples and so has the same
error distribution. The gradient is where it shows: the 13 padded slots carried ‖grad‖ 0.0084
against 0.0335 for the three real ones, so roughly a fifth of the step was extra weight on the
three images the padding was copied from. After the fix the padded slots carry exactly 0.

`minimax_h3`'s audio branch takes the per-sample **maximum** of the spatial mask, since audio has
no spatial dimension to multiply against. That is exact for the cases that matter — padding's
mask is identically zero, and an unmasked sample's mask is a constant, so the maximum is that
constant. Two edge cases change: an image mask that never reaches 1 now attenuates the audio
loss, and an image masked out entirely now contributes no audio either.

## Distillation

`EpochSampler` was already cleaner than the dataset path: `epoch_order` is a pure function of
`(seed, epoch)` that shuffles the whole caption list and then takes a strided shard. Under `fill`,
`steps_per_epoch` rounds up, and the short final batch is completed from the *next* epoch's
permutation — which keeps the whole thing a function of `(seed, epoch)` and so keeps resume
exact.

Deduplication there is by caption **text**, not by index, and that is load-bearing.
`relational_loss` matches the teacher's pairwise distance structure specifically to punish
distances collapsing toward zero. Two identical caption strings in one batch produce a distance
of exactly zero on both sides, so the term would be teaching the model that two captions belong
on top of each other — the shape of the collapse it exists to prevent.

For the same reason, padded captions are **removed** from `relational_loss` and from the `spread`
diagnostic rather than weighted to zero. A masked repeat sits exactly on top of the caption it was
copied from; counting it drags the mean pairwise cosine distance down and makes the progress bar
report a collapse that is not happening.

The same caveat as the dataset side applies to the static part: a corpus that genuinely repeats a
caption still gets repeats inside the earlier batches of an epoch, because the shuffle does not
know about them. Measured on a corpus of 30 identical strings plus 70 unique ones: batches 0, 1,
2, 4 and 5 held a repeat, batch 6 — the one the fill builds — did not.

## What is not done, and why

- **Shrinking the batch instead of padding it.** Every rank must run the same number of steps and
  DeepSpeed pipeline parallelism expects a fixed micro-batch size, so a short final step would
  need coordination this does not have. `pad_masked` was chosen instead.
- **Rebalancing across aspect-ratio buckets.** Moving an image between buckets would invalidate
  its cached latents, which are the expensive half.
- **A batch-aware permutation of the whole order.** See "What no duplicate image does and does not
  promise" above.
- **`models/hunyuan_image.py`'s own `load_adapter_weights`** carries an unrelated copy of a
  key-rewriting rule that this work did not touch.

## Verification

Everything here is CPU-only. There has been no GPU run, no multi-rank run and no real training
run; the multi-rank cases are covered by constructing one `ConcatenatedBatchedDataset` per rank
and checking that the ranks compute the same global order and that their slices reassemble into
it, which is the property a real multi-GPU run depends on.

Tests: [`test/test_batch_fill.py`](../../test/test_batch_fill.py) and the
`TestEpochSamplerBatchFill`, `TestWeightedMse` and `TestPaddingIsExcludedFromRelationalAndSpread`
classes in [`test/test_distill_refiner.py`](../../test/test_distill_refiner.py).

The eight-loss-function test was checked against a reverted fix: with the two masking fixes
undone, exactly `CosmosPredict2Pipeline-ms` and `MinimaxH3Pipeline` fail and the other six stay
green. `models/cosmos.py` is skipped on this box because it imports `pynvml`, which is not
installed without a GPU.
