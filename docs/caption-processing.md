# Whole-caption dropout and non-Latin line filtering

These settings belong in the **dataset TOML**, before the first `[[directory]]` for dataset-wide
defaults, or inside an individual `[[directory]]` to override those defaults. They apply to
diffusion training datasets regardless of whether text embeddings are cached or computed
on-the-fly. They do not modify source `.txt` files, `captions.json`, images, or model weights.

```toml
resolutions = [512]
caption_dropout_rate = 0.1
enable_remove_non_latin = true

[[directory]]
path = '/data/filtered'
# Inherits both settings above.

[[directory]]
path = '/data/multilingual'
caption_dropout_rate = 0.0
enable_remove_non_latin = false
```

## Whole-caption dropout

`caption_dropout_rate` accepts a finite number from `0` through `1`. Each sample access draws
again: `0.1` means a 10% probability of empty conditioning, not a permanently selected 10% of
files. The image and latent stay the same. At `1`, every accessed caption becomes exactly `""`,
even when `caption_prefix` is configured. At `0`, this feature does not consume RNG.

The effective probability is the first **declared** value in this order:

1. The current `[[directory]]`'s `caption_dropout_rate`.
2. The dataset TOML's top-level `caption_dropout_rate`.
3. The training TOML's existing `uncond_fraction`, defaulting to zero.

An explicit zero overrides the fallback. The old and new probabilities are never combined.
Use a separate eval dataset with `caption_dropout_rate = 0.0` when evaluating conditional loss
without dropout; absent settings keep the existing `uncond_fraction` behavior.

For cached text encoders, the same decision selects both the caption `""` and the existing
unconditional embedding **for every cached encoder**, including its associated masks/features.
These are encoder outputs for the empty string, not zeroed versions of a conditional tensor.
For on-the-fly or hybrid models, the live caption also becomes `""`, so their live encoders
receive the same conditioning. The decision is never baked into metadata or embedding caches.
Changing the dropout rate requires no recache, including when `--trust_cache` is not set.

This differs from `tag_dropout_rate`, which removes individual tags and always keeps at least
one tag. Its existing cache-time/runtime behavior remains unchanged.

## Line filtering

`enable_remove_non_latin` is a boolean, default `false`. With it enabled, processing is:

1. Remove the configured `prefix_tag_caption` annotation before checking the caption body.
2. Remove each entire line containing a character in any of the ranges below. A mixed line
   such as `blue eyes, 制服` is removed in full, including its Latin words.
3. Shuffle/drop tags according to the existing settings.
4. Add `caption_prefix` to a non-empty body. A body with no remaining content stays `""`.
5. At sample access, whole-caption dropout can replace the resulting conditioning with `""`.

| Range | Covered characters |
| --- | --- |
| U+4E00–U+9FFF | CJK Unified Ideographs |
| U+3400–U+4DBF | CJK Extension A |
| U+3040–U+30FF | Hiragana and Katakana |
| U+AC00–U+D7AF | Hangul syllables |
| U+0400–U+04FF | Cyrillic |
| U+0600–U+06FF | Arabic |
| U+0590–U+05FF | Hebrew |

This uses precisely the requested range detector, not language detection or a comprehensive
Unicode script classifier. Latin Extended remains valid (`Pokémon`, `résumé`, Vietnamese
letters); scripts outside these ranges also remain valid. Digits, punctuation and emoji alone
do not trigger removal. The training prefix is added after checking and is not itself filtered.

For `prefix_tag_caption = 'Special:'` and `caption_prefix = 'anime, '`:

| Source caption | Processed caption |
| --- | --- |
| `Special: blue eyes` | `anime, blue eyes` |
| `Special: 制服` | `""` |
| `a girl` followed by `blue eyes, 制服` followed by `outdoor` on separate lines | `anime, a girl` followed by `outdoor` |

With the normal `.txt` format, the file is one caption and only offending lines inside that
caption disappear. With `multiline_captions = true`, each original non-empty line remains a
sample; a removed line becomes an empty caption for that sample. Each `captions.json` list
element is processed independently. All-filtered captions remain samples rather than removing
their images. An existing blank `.txt` also produces an empty caption with filtering enabled,
including in multiline mode. Missing caption files still follow `skip_empty_caption`.

## Caches and online captions

| Change/path | Caption metadata / iteration order | Text embeddings | Image latents |
| --- | --- | --- | --- |
| Change `caption_dropout_rate` | Reused; no frozen dropout | Reused; select unconditional at access | Reused |
| Toggle `enable_remove_non_latin` | Separate caption-settings suffix, including under `--trust_cache` | Refresh when the caption metadata changes | Reuse when image rows, masks, controls, buckets and VAE are unchanged |
| On-the-fly text encoding | Caption is filtered before runtime augmentation/encoding | Encode the resulting string each access | Existing latent caching behavior |

The new filter suffix is empty at its default. Caption text is excluded from the latent
fingerprint. Changing filtering on a dataset with unchanged image rows therefore does not
re-encode its images through the VAE. Changes that actually add/remove media rows, change a
mask/control/bucket, or change the VAE still require compatible latents. For example, preserving
a previously skipped blank multiline file adds a media row that needs a latent.

An explicit `--regenerate_cache` still rebuilds all caches as it did before. It is not needed
merely to switch either of these new settings; omit it when reusing image latents.

`online_captions = true` continues to use the in-memory `captions.json` dictionary for models
with no cached text encoders. Raw captions there are filtered at access. When any encoder is
cached, the caption and its embedding come from the **same cached variant** instead. This
prevents live edits or cache-shuffled indices from desynchronizing cached and live encoders in
hybrid models. To apply source caption edits to cached embeddings, rebuild the caption/text
cache according to the existing source-edit rules.
For uncached online captions with `cache_shuffle_num > 1`, a sampled variant index is mapped
back to its original caption before applying the fresh augmentation; it is not used as an
index into the shorter raw caption list.

Caption-only enumeration with `apply_shuffle=True` uses the same deterministic filter and
per-directory inheritance. Whole-caption dropout is never applied during enumeration.
`export_caption_corpus.py` uses `apply_shuffle=False` and intentionally preserves raw caption
text/markers: it is a source corpus, not a frozen training augmentation. The new dataset
sampling settings do not configure the separate text-only distillation trainer.

## Validation

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 DIFFUSION_PIPE_NUM_PROC=1 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled \
python -m pytest test/test_caption_filter_dropout.py -q
```

The tests use actual dataset metadata, iteration orders and SQLite tensor shards; only VAE
and encoder forward calls use small deterministic tensor maps. They cover Unicode boundaries,
whole mixed-line removal, marker/prefix ordering, source preservation, empty captions,
global/directory precedence, legacy fallback, rate validation, per-access RNG, multiple cached
encoders, hybrid/online paths, both caption sampling modes, and cache reuse with and without
`--trust_cache`. GPU training with real model weights is not part of the CPU checks.
