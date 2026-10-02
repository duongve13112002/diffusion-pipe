# Source metadata snapshots and tensor profiles

Date: 2026-10-02. Task: change caption selection, line removal and runtime dropout without
rescanning a 30M-media dataset or encoding compatible images again, including with trust_cache.

The source contract is explicit: reuse_metadata_cache defaults to true and freezes original
captions and media metadata until regeneration. This also freezes online_captions. It does
not disable uncached text encoding or runtime augmentation. The opt-out retains legacy
source rescans and live online captions. Caption-only corpus enumeration remains separate.

Source geometry and caption configuration have separate directory keys. Runtime dropout,
sampling, repeats and optimizer/training settings are not tensor inputs. Selection profiles
refer to tensor shards by disk-backed input indexes; text keys include processed caption and
all media/bucket/control context. Encoder identity and declared encoding-key mismatches reject
donation; legacy donors lacking a required non-empty key are conservatively recached. Incomplete
donors are ignored rather than appended by unverifiable row position. Donors are read-only.

The critical abd164e migration detail is SizeBucketDataset's extra shuffle. On-disk grouped
metadata order is not tensor order. Saved iteration_order provides image_spec -> latents_idx;
the importer reconstructs that order and checks complete contiguous indices. Text donor
caption digests then validate the reconstructed caption order. This caught a real mock-test
failure before the migration was accepted. Tuple/list seed representation is normalized in
new profile mode so cold and warm bucket loading have identical ordering.

Old metadata does not store original dimensions or a complete geometry configuration. Import
assumes unchanged bucketing/media processing; unmatched attributes fall back to media reads.
Regenerate when changing source/bucketing without proven compatible metadata. Imports record
the first declared legacy geometry; later geometries cannot borrow those legacy tensors.
Mask/control migration conservatively rescans source metadata. Old sidecars
require one initial raw-text read. Imports/new profiles still cost O(N), and first grouping
retains the old Python grouping structures; no 30M startup/memory claim is made.

Refresh writes a new generation and publishes its source pointer. Main-rank dataset objects
constructed before the cache worker must reload that pointer. Tensors do not donate across
source generations because bytes can change at the same filename. Old snapshots are preserved
for other profiles/runs; removing a donor breaks its references and is not automatic cleanup.

Implementation: utils/cache_profiles.py and utils/dataset.py. Exact installed datasets APIs
(from_generator, map, sort, from_file, select, save_to_disk) were inspected before use. No
dependency/submodule upgrades are involved. Tests in test/test_cache_profiles.py generate
real historical-format mock metadata/SQLite/shards from git show abd164e, with deterministic
CPU tensor forwards and thread-based mock transport. GPU/full training qualification is
outside these cache/index tests. See docs/caption-processing.md for the server CPU command.
The historical matrix explicitly skips when the abd164e Git object is absent; current-format
tests remain available offline. Local validation included that object and ran all 213 cases.

Final local evidence: 1103 passed, 2 skipped in the full CPU suite (143.70 seconds), including
213 cache-profile tests. pip check, AST/TOML parsing and diff checks passed. Historical fixture
hashes were unchanged. Artifacts: .tmp/pytest-profiles-final-verified/abd164e_* and
.tmp/cache-profile-validation-final.xml. Native Linux multi-process caching and full GPU
training/30M-scale performance were not tested; serialization and CPU tensor/index correctness
do not substitute for those integration/performance checks.
