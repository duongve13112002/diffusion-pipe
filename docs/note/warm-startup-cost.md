# What a fully cached launch reads again, and what it no longer does

2026-11-01. Prompted by a MOSS-Image dataset of about 10M images in about 100 `[[directory]]`
entries with `reuse_metadata_cache = false`: with every cache complete and `--trust_cache`, a launch
still spent a long time "putting the dataset into buckets and loading the VAE and text caches"
before training started.

## Where the time went

Measured on CPU with a synthetic directory of 100,000 images in 7 aspect ratio buckets, two
captions each, caches complete, timing one process's load (`DirectoryDataset.cache_metadata`,
`cache_latents`, `cache_text_embeddings`, as every rank runs them in `DatasetManager.cache`). The
benchmark script is not part of the repository; the numbers below are its output.

| Setting | Warm load, 100k images | Scaled to 10M |
| --- | --- | --- |
| `reuse_metadata_cache = true` (default) | 0.8 s (1.3 s with `post_init`) | about 2 min |
| `reuse_metadata_cache = false`, before | 55.9 s | about 1.5 h |
| `reuse_metadata_cache = false`, first launch after the change | 22.5 s | about 40 min, once |
| `reuse_metadata_cache = false`, later launches | 2.9 s | about 5 min |

Every rank pays this, and on rank 0's node the caching process pays it once more before the ranks
start loading.

With `reuse_metadata_cache = false`, the latent and text embedding caches are checked against a
fingerprint of the metadata (`_cache_fingerprint`) and, for text, a digest of the captions
(`_content_digest`). Both hashed whole columns on every launch:

- `list(dataset[column])` reads a column through the `datasets` library's lazy `Column`, which
  formats one row at a time: about 45 s of the 56 s above.
- `Hasher.update(values)` then pickles the whole Python list with dill: about 13 s per 100k rows.

Layout-caption models (MOSS-Image, Ideogram 4) also read every caption of every bucket at every
launch, on every rank: `_report_layout_captions` (only rank 0 prints the result) and the check in
`ARBucketDataset.cache_text_embeddings` for whether a bucket has layout captions.

## What changed

- `_column_values(dataset, column)` reads the column as one Arrow column
  (`select_columns([column]).with_format('arrow')[:].column(0).to_pylist()`), 20 to 40 times
  faster. The values are the same Python objects the row-by-row read returns (the `datasets`
  Python formatter itself calls `to_pylist`, and decodes only features that need decoding, which
  are left to the old path), so every fingerprint keeps its value and no cache is re-encoded.
  `test/test_digest_memo.py` checks the fingerprints against the old code on file-backed,
  shuffled, selected and in-memory datasets.
- `_memoized_digest` remembers a digest in `diffusion_pipe_digests.json` next to the dataset's
  Arrow files, keyed by the files' paths, sizes and modification times, the row selection
  (indices mapping), the length and the dataset's fingerprint. The fingerprint is what tells
  apart two contiguous ranges of one file (`select(range(a, b))` slices the table without an
  indices mapping) and a column changed in memory (`add_column` keeps the files); an independent
  review found both collided before it was part of the key. A regrouping writes new files, which miss. A dataset held in
  memory is not remembered unless it names the file-backed dataset it is a pure function of
  (`_digest_origin`, set for the flattened captions). A volume that cannot be written computes the
  digest every time, as before. The file keeps the newest 64 entries.
- The layout-caption counts are remembered the same way, and only rank 0 computes the report it
  prints.
- `TextEmbeddingDataset` and `_build_iteration_order` read `image_spec` with `_column_values`.

When a remembered digest is recomputed:

- the metadata is grouped again: a launch without `--trust_cache`, `--regenerate_cache`, a changed
  bucket setting or caption setting, or a stale grouping that `--trust_cache` refuses. Each writes
  new Arrow files, so the size or modification time differs and the digest is computed from the new
  rows. The cache checks then run exactly as before: a changed fingerprint still rebuilds the
  latent or text cache, and a changed caption digest still rebuilds the text cache.
- the row selection changes (another shuffle, another subset): the indices mapping is part of the key.
- `diffusion_pipe_digests.json` is deleted, or cannot be read.

Several GPUs and nodes: every process (each rank, and rank 0's caching process) reads and writes
the same memo file on the shared volume. A write goes to a temporary file named with the host,
pid and thread, then `os.replace`, so readers see a whole file; two writers racing can drop each
other's new entry, which only means it is computed again at a later launch. The memo is read
after the dataset's files were opened (`load_from_disk`), which on NFS revalidates their
attributes (close-to-open consistency), so another node's rewrite is seen by size and time.

Not detected: a file rewritten in place that keeps its size and its modification time to the
nanosecond. Nothing in this repository writes metadata that way (a regrouping writes new files).

A dataset with an output format set (numpy, torch) is read through the `datasets` library as
before, so its fingerprint is the one the old code computed; no caller sets one today.

What `--trust_cache` trusts is unchanged: it never rescanned the source images and captions, and
still does not. Edited source files are seen by a launch without `--trust_cache` (which groups the
metadata again, and so computes fresh digests) or with `--regenerate_cache`.

The first launch after the change still hashes once (on every rank at the same time, since none
has a remembered digest yet). The default `reuse_metadata_cache = true` path never hashed whole
columns and is unchanged.

## What is still linear in the dataset size

With `reuse_metadata_cache = false`, flattening the captions (`metadata_dataset.map(flatten_captions)`)
and building the text embedding index run on every launch: about 2.5 s per 100k images, so
about 4 minutes per process for 10M. Keeping the flattened captions on disk, as the layout path
already does (`_flattened_captions`), would remove the first; it changes which `_fingerprint` the
text cache is checked with only if the saved dataset keeps the in-memory one, which needs checking
before it is done.

## Measuring a real launch

`train.py` prints `[startup] <phase>: <seconds>` as each startup phase ends and a summary with the
slowest rank per phase before the first step (`utils/startup_timing.py`). The two dataset phases are
`caching round` (rank 0's node checking or filling the caches) and `every rank loading the cached
datasets`.
