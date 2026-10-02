# Caption filtering and per-sample unconditional conditioning

Date: 2026-09-30; extended 2026-10-02

Update: default snapshot/profile mode supersedes the full-copy mode isolation described
below. Those details remain accurate for `reuse_metadata_cache = false`. See
[source-cache-profiles.md](source-cache-profiles.md) for read-only legacy tensor donation,
raw-caption snapshots, refresh boundaries and migration tests.

Task: add dataset-wide/per-directory `caption_dropout_rate` and `enable_remove_non_latin`
without changing image latents or source caption files.

Extension: add tri-state `require_non_latin_caption` to select whole media items using
all source caption lines. True includes a media item if any line matches; false excludes
the entire media item if any line matches; unset keeps the existing dataset. Alternatives
are never individually selected. Share the predicate between metadata and caption-only
enumeration before augmentation, including raw export for distillation. Exclude configured
annotation markers using the same rule as line removal.

This changes image-row membership, so the line-removal latent reuse guarantee does not
apply. Isolate enabled modes below the existing model cache root; preserve the legacy
path at None. Mode-specific caches cost extra disk and initial VAE/text encoding but avoid
destructive mode switching and incorrect reuse of equal-length, different subsets. Keep
normal source-edit/trusted-cache rules and existing identity/fingerprint checks within a
mode. Do not treat a flat caption corpus as if it retained image grouping information.
For enabled modes, record an image-spec digest to reject retained latents after an
equal-length selection change. Cache-only loads without a map function check recorded
content and counts; mismatches require an encoder rather than returning stale tensors.

Warm selected-cache tests also cover the grouped-metadata key check: size buckets use
three-element keys and must not be rejected by the two-element aspect-ratio check.

## Code paths inspected

- [`utils/captions.py`](../../utils/captions.py): marker stripping, tag augmentation, `.txt`
  resolution, and caption-only enumeration.
- [`utils/dataset.py`](../../utils/dataset.py): `DirectoryDataset` builds caption metadata;
  `SizeBucketDataset.__getitem__` chooses conditioning; `_cache_text_embeddings` caches every
  caption variant. `DirectoryDataset.cache_text_embeddings` already caches `""` separately.
- [`utils/cache.py`](../../utils/cache.py): fingerprint, identity and content-digest checks.
- [`train.py`](../../train.py): the existing training-wide `uncond_fraction` populates
  `utils.dataset.UNCOND_FRACTION`; the new setting overrides it only when declared.
- [`tools/export_caption_corpus.py`](../../tools/export_caption_corpus.py): exports raw captions
  using `enumerate_captions(..., apply_shuffle=False)`.

## Decisions

Whole-caption dropout belongs at sample access, independent of whether text is cached. Reuse
the existing unconditional-selection branch so the string, all cached encoder features and
live/hybrid encoders switch together. Do not pass the probability into caption expansion or
add it to `CAPTION_CACHE_SETTINGS`. An explicit directory zero must override a dataset-wide
rate and the legacy fallback; absence is therefore represented separately from zero.

Filtering is deterministic and belongs before tag augmentation/text encoding. The detector
matches the user's seven inclusive Unicode intervals exactly; extending it to more scripts
would change the agreed semantics. Strip annotation markers before testing, and apply a
training prefix only after a non-empty body survives. Preserve empty elements/variant counts
so filtering does not remove media rows just because they become unconditional samples.

Add only the boolean filter to the caption-settings suffix. It separates metadata and
iteration orders under `--trust_cache` while preserving the empty suffix at the default.
Image latents already fingerprint all metadata columns **except caption**, so neither new
setting needs a change to the latent-cache implementation. Existing blank multiline files
are retained with the filter enabled; including previously skipped media legitimately changes
image-row alignment and is outside the unchanged-image cache guarantee.

The online dictionary previously indexed raw captions using a cached-variant index and could
also pair an edited string with an old embedding. When any text encoder is cached, use its
matching metadata caption instead; only fully uncached models read/transform the live string.
This directly protects hybrid models that use cached features alongside live caption tokens.
For uncached online augmentation, map the expanded variant index back to its source caption
using the existing variants-per-caption count. Otherwise `cache_shuffle_num > 1` can index
past the raw list before filtering even runs. Both caption sampling modes exercise this path.

Keep raw corpus exports raw. Processed caption-only enumeration honors filtering and directory
overrides, but no dropout is frozen into enumeration. The separate flat text-only distillation
trainer retains its own configuration contract.

## Evidence and practical limits

[`test/test_caption_filter_dropout.py`](../../test/test_caption_filter_dropout.py) runs the real
metadata and disk-cache machinery against tiny images. Encoder outputs deliberately distinguish
every tested string from `""`, so an empty string paired with a conditional embedding fails.
Cache-reuse tests forbid encoder/VAE calls by passing `None` after the first cache build, and
compare actual latent shard bytes/fingerprints before and after filtering. Dropout-rate cache
reuse is checked both with and without `--trust_cache`.

The suite also checks old fallback behavior, explicit zero, exact Unicode endpoints, Latin
Extended, source-file bytes, full filtering, per-access draws, multiple encoders, shuffled
online cached variants and `random_per_epoch`. These CPU checks do not claim real GPU training
or model-quality verification. See [the user documentation](../caption-processing.md) for the
config and command examples.

Final CPU validation: **821 passed, 1 skipped** in 70.61 seconds on Python 3.12 / PyTorch
2.9.0+cpu. All **73 new caption tests** passed. The existing Cosmos batch-fill test was skipped
because its optional import requires `pynvml`; this is not a skip in the new caption coverage.
`pip check`, dataset TOML parsing, new documentation links and `git diff --check` also passed.
