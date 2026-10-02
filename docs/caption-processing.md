# Whole-caption dropout, line filtering and media selection

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

## Whole-media selection by source captions

`require_non_latin_caption` belongs in the dataset TOML, globally or per `[[directory]]`.
It accepts `true` or `false`; omitting it globally defaults to `None` internally and preserves
normal selection. Directories inherit the global value unless they declare an override.
TOML has no literal `None`/`null` value. This selects whole images/videos rather than editing
caption lines:

| Setting | Media selection |
| --- | --- |
| `true` | Keep a media item only if at least one source line in any alternative caption contains a detected non-Latin character. Keep all its alternative captions. |
| `false` | Exclude the entire media item if any source line in any alternative caption contains a detected non-Latin character. |
| Unset | Do not select media by script. |

For an image with `a girl`, `制服`, and `blue eyes` on separate lines, `true` keeps the
image and all three lines, while `false` drops the image. The same rule applies to a
`captions.json` list: one matching element selects/excludes the entire image and every
alternative caption, including the Latin ones. It also applies with `multiline_captions`
and `caption_sampling = 'random_per_epoch'`; a random Latin caption cannot let an excluded
image back into training. Missing captions still follow `skip_empty_caption`; an empty
caption has no detected non-Latin character.

Selection uses the seven Unicode ranges listed above. Vietnamese/Latin Extended, emoji,
and scripts outside those ranges (such as Thai) do not trigger it. Configured
`prefix_tag_caption` markers are excluded from each line's check, and `caption_prefix` is
not part of the source caption. Selection precedes line removal, tag augmentation and
whole-caption dropout. These settings remain independent: combining `true` with
`enable_remove_non_latin = true` first selects the multilingual images, then removes their
offending caption lines; captions can become empty under the existing rules.

```toml
resolutions = [512]
require_non_latin_caption = true

[[directory]]
path = '/data/multilingual'

[[directory]]
path = '/data/latin-only'
require_non_latin_caption = false
```

Unlike line removal, selection changes the media rows used to index latents. Enabled modes
therefore isolate all caches under `cache/<model>/require_non_latin_caption_true/` or
`cache/<model>/require_non_latin_caption_false/`. Metadata, conditional/unconditional text
embeddings, image latents, and iteration order all belong to that mode. The first run of
each enabled mode builds its own cache; this costs extra disk space and does not project
an existing full-dataset latent cache onto the subset. Switching modes preserves other
modes' caches. Unset retains the original `cache/<model>/` paths and fingerprint inputs.
`keep_latent_cache` cannot reuse another mode's rows. Compatible caches within a mode retain
their normal reuse behavior, including trusted grouped metadata.
Within enabled modes, an image-spec digest also prevents `keep_latent_cache` from retaining
the wrong latents if caption edits swap selected images while keeping the same row count.
Cache-only loading without an encoder refuses mismatched recorded content/row counts;
run caching with the encoder available to regenerate the affected cache.

With on-the-fly text encoding, the same source-caption selection happens before media
caching, and kept captions are encoded during training. With cached text, only the kept
images' captions are embedded. Selection is fixed when metadata is built, including with
`online_captions`; it is not re-drawn per step. After source caption edits, rebuild metadata
following the existing source-edit rules rather than trusting an old selection cache.

Caption-only enumeration applies the same whole-media rule even with `apply_shuffle=False`,
so raw corpus export and dataset-driven distillation see the same selected media. A corpus
already flattened into independent captions has lost image boundaries: re-export it from
the filtered dataset to change selection. Metadata logs report retained/skipped media
after selection and validation; an empty selected directory gets a warning. Caption-only
enumeration and export also report the exact number excluded by the script condition.

## Caches and online captions

| Change/path | Caption metadata / iteration order | Text embeddings | Image latents |
| --- | --- | --- | --- |
| Change `caption_dropout_rate` | Reused; no frozen dropout | Reused; select unconditional at access | Reused |
| Toggle `enable_remove_non_latin` | Separate caption-settings suffix, including under `--trust_cache` | Refresh when the caption metadata changes | Reuse when image rows, masks, controls, buckets and VAE are unchanged |
| Set/change `require_non_latin_caption` | Separate mode directory for selected media | Separate mode directory | Separate mode directory; first run caches the selected subset |
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

Whole-media selection tests use real metadata and tensor-cache paths with tiny CPU-only
encoder maps. Run inside the project venv; no model downloads or GPU allocation are needed:

```bash
mkdir -p .tmp .cache/pip
TMPDIR="$PWD/.tmp" TMP="$PWD/.tmp" TEMP="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.cache/pip" \
CUDA_VISIBLE_DEVICES=-1 \
.venv/bin/python -m pytest test/test_non_latin_dataset_selection.py -q --basetemp .tmp/pytest-selection
```

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
