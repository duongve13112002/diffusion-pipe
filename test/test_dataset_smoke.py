"""Construct real dataset objects.

This file exists because 182 tests passed while `utils/dataset.py` was missing three names it
calls at runtime -- `bucket_suffix`, `dedup_and_sort`, `seed_from_hash`, accidentally moved into
`utils/captions.py`. Every training run was broken. The module still *imported* cleanly, and no
test in the repo had ever constructed a `DirectoryDataset`, `ARBucketDataset` or
`SizeBucketDataset`, so nothing noticed.

The lesson is narrow and worth keeping: an import test proves a module parses, not that it runs.
These tests build the real objects against a handful of real (tiny) images.
"""

import gc
import sys
from pathlib import Path

import pytest
import toml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import utils.dataset as dataset_module  # noqa: E402
from utils.dataset import (  # noqa: E402
    CAPTION_CACHE_SETTINGS,
    DirectoryDataset,
    caption_cache_suffix,
    bucket_suffix,
    collapse_to_one_entry_per_image,
    dedup_and_sort,
    seed_from_hash,
)


def make_image_dir(tmp_path, captions, size=(64, 64), multiline=False):
    d = tmp_path / 'imgs'
    d.mkdir(exist_ok=True)
    for name, text in captions.items():
        Image.new('RGB', size, (128, 128, 128)).save(d / f'{name}.png')
        (d / f'{name}.txt').write_text(text)
    return d


def cached_captions(ds):
    """Every caption list in the metadata, whichever bucketing path the config selected."""
    buckets = ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets
    return [c for bucket in buckets for c in bucket.metadata_dataset['caption']]


def directory_config(path, **overrides):
    config = {'path': str(path), 'resolutions': [64], 'frame_buckets': [1], 'num_repeats': 1}
    config.update(overrides)
    return config


class TestModuleLevelNames:
    """Names utils/dataset.py calls but does not define itself are the ones that go missing."""

    def test_helpers_are_importable_and_callable(self):
        assert seed_from_hash('anything') == seed_from_hash('anything')
        assert bucket_suffix((512, 512, 1)) == '512x512x1'
        assert bucket_suffix((1.0, 1)).startswith('1.0')
        assert list(dedup_and_sort([2.0, 1.0, 1.0])) == [1.0, 2.0]

    def test_caption_helpers_are_re_exported(self):
        # utils/captions.py is the definition site; utils/dataset.py must keep re-exporting.
        for name in ('shuffle_captions', 'enumerate_captions', 'preprocess_caption',
                     'read_caption_file', 'split_tag_prefix', 'drop_tags',
                     'CAPTIONS_JSON_FILE', 'NON_MEDIA_SUFFIXES'):
            assert hasattr(dataset_module, name), f'utils.dataset no longer exports {name}'

    def test_caption_module_holds_no_dataset_helpers(self):
        # The three above live in utils/dataset.py. If they drift back into utils/captions.py
        # they lose the imports their bodies need (numpy, hashlib, ROUND_DECIMAL_DIGITS).
        import utils.captions as captions_module
        for name in ('bucket_suffix', 'dedup_and_sort', 'seed_from_hash'):
            assert not hasattr(captions_module, name), (
                f'{name} belongs in utils/dataset.py; in utils/captions.py its body has no '
                f'numpy / hashlib / ROUND_DECIMAL_DIGITS in scope'
            )

    def test_every_global_utils_dataset_references_actually_resolves(self):
        """Catch the whole class of bug, not just the three names that hit us."""
        import ast
        import builtins
        source = (REPO / 'utils/dataset.py').read_text()
        tree = ast.parse(source)

        # Module-level names Python injects rather than the source assigning them.
        assigned = set(dir(builtins)) | {'__file__', '__name__', '__doc__', '__package__'}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assigned.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    assigned.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.arg):
                assigned.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                assigned.add(node.name)
            elif isinstance(node, ast.Global):
                assigned.update(node.names)

        missing = sorted({
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id not in assigned
        })
        assert not missing, f'utils/dataset.py references undefined names: {missing}'


class TestDirectoryDatasetConstruction:
    """The objects a training run actually builds, on real image files."""

    def test_constructs_and_buckets(self, tmp_path):
        d = make_image_dir(tmp_path, {'a': 'one', 'b': 'two'})
        ds = DirectoryDataset(directory_config(d), {'resolutions': [64]}, 'test_model', skip_dataset_validation=True)
        assert ds.path == d
        assert len(ds.ars) > 0

    def test_cache_metadata_runs_end_to_end(self, tmp_path):
        d = make_image_dir(tmp_path, {'a': 'one', 'b': 'two', 'c': 'three'})
        ds = DirectoryDataset(directory_config(d), {'resolutions': [64]}, 'test_model', skip_dataset_validation=True)
        ds.cache_metadata()
        captions = cached_captions(ds)
        assert len(captions) == 3
        assert sorted(c[0] for c in captions) == ['one', 'three', 'two']

    def test_multiline_captions_reach_the_metadata(self, tmp_path):
        d = make_image_dir(tmp_path, {'a': 'first\nsecond'})
        ds = DirectoryDataset(
            directory_config(d, multiline_captions=True), {'resolutions': [64]},
            'test_model', skip_dataset_validation=True,
        )
        ds.cache_metadata()
        captions = cached_captions(ds)
        assert sorted(captions[0]) == ['first', 'second'], captions

    def test_multiline_off_keeps_one_caption(self, tmp_path):
        d = make_image_dir(tmp_path, {'a': 'first\nsecond'})
        ds = DirectoryDataset(directory_config(d), {'resolutions': [64]}, 'test_model', skip_dataset_validation=True)
        ds.cache_metadata()
        captions = cached_captions(ds)
        assert captions[0] == ['first\nsecond']

    def test_tag_marker_is_stripped_before_the_metadata(self, tmp_path):
        d = make_image_dir(tmp_path, {'a': 'Special: x, y'})
        ds = DirectoryDataset(
            directory_config(d, prefix_tag_caption='Special: '), {'resolutions': [64]},
            'test_model', skip_dataset_validation=True,
        )
        ds.cache_metadata()
        captions = cached_captions(ds)
        assert captions[0] == ['x, y'], 'the marker must never reach the text encoder'


class TestCaptionSampling:
    """caption_sampling = 'random_per_epoch': one sample per image, caption drawn each pass."""

    def _dataset(self, tmp_path, captions_json, **overrides):
        import json
        d = tmp_path / 'imgs'
        d.mkdir(exist_ok=True)
        for name in captions_json:
            Image.new('RGB', (64, 64), (128, 128, 128)).save(d / name)
        (d / 'captions.json').write_text(json.dumps(captions_json))
        return DirectoryDataset(
            directory_config(d, **overrides), {'resolutions': [64]},
            'test_model', skip_dataset_validation=True,
        )

    def test_rejects_an_unknown_value(self, tmp_path):
        with pytest.raises(ValueError, match='caption_sampling'):
            self._dataset(tmp_path, {'a.png': ['x']}, caption_sampling='nonsense')

    def test_default_is_all(self, tmp_path):
        assert self._dataset(tmp_path, {'a.png': ['x']}).caption_sampling == 'all'

    def test_setting_reaches_the_size_bucket(self, tmp_path):
        ds = self._dataset(tmp_path, {'a.png': ['x']}, caption_sampling='random_per_epoch')
        assert ds.directory_config['caption_sampling'] == 'random_per_epoch'


class TestCollapseToOneEntryPerImage:
    """The iteration-order transform behind caption_sampling = 'random_per_epoch'.

    Tested directly rather than through cache_latents: utils/cache.py opens sqlite with
    autocommit=, which needs Python 3.12, so the caching path cannot run on 3.11.
    """

    def rows(self):
        return [
            (['spec', 'a.png'], 0, 'one', 0),
            (['spec', 'a.png'], 0, 'two', 1),
            (['spec', 'a.png'], 0, 'three', 2),
            (['spec', 'b.png'], 1, 'x', 0),
            (['spec', 'b.png'], 1, 'y', 1),
        ]

    def test_one_entry_per_image(self):
        out = collapse_to_one_entry_per_image(self.rows())
        assert len(out) == 2, '5 caption rows over 2 images collapse to 2 entries'

    def test_every_caption_is_kept(self):
        out = collapse_to_one_entry_per_image(self.rows())
        by_image = {tuple(spec): captions for spec, _, captions, _ in out}
        assert by_image[('spec', 'a.png')] == ['one', 'two', 'three']
        assert by_image[('spec', 'b.png')] == ['x', 'y']

    def test_caption_and_cache_index_stay_paired(self):
        # The pairing is the whole point: one draw must select the text and the embedding
        # together, or a model reading both disagrees with itself.
        out = collapse_to_one_entry_per_image(self.rows())
        for _, _, captions, numbers in out:
            assert len(captions) == len(numbers)
        by_image = {tuple(spec): numbers for spec, _, _, numbers in out}
        assert by_image[('spec', 'a.png')] == [0, 1, 2]
        assert by_image[('spec', 'b.png')] == [0, 1]

    def test_latents_index_is_preserved(self):
        out = collapse_to_one_entry_per_image(self.rows())
        assert {tuple(spec): idx for spec, idx, _, _ in out} == {
            ('spec', 'a.png'): 0, ('spec', 'b.png'): 1,
        }

    def test_single_caption_images_are_unaffected_in_content(self):
        rows = [(['spec', 'a.png'], 0, 'only', 0)]
        assert collapse_to_one_entry_per_image(rows) == [(['spec', 'a.png'], 0, ['only'], [0])]

    def test_empty_input(self):
        assert collapse_to_one_entry_per_image([]) == []


class TestRuntimeAugmentation:
    """When no text embeddings are cached, augmentation moves from cache time to __getitem__."""

    def _dataset(self, tmp_path, caption, caches_text_embeddings, **overrides):
        import json
        d = tmp_path / 'imgs'
        d.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), (128, 128, 128)).save(d / 'a.png')
        (d / 'captions.json').write_text(json.dumps({'a.png': caption}))
        ds = DirectoryDataset(
            directory_config(d, **overrides), {'resolutions': [64]}, 'test_model',
            skip_dataset_validation=True, caches_text_embeddings=caches_text_embeddings,
        )
        ds.cache_metadata()
        return ds

    def _captions(self, ds):
        buckets = ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets
        return [c for row in buckets[0].metadata_dataset['caption'] for c in row]

    def test_uncached_keeps_the_marker_so_tags_stay_distinguishable(self, tmp_path):
        ds = self._dataset(
            tmp_path, ['Special: a, b, c', 'Some prose, with a comma.'], False,
            prefix_tag_caption='Special:', cache_shuffle_num=1, tag_dropout_rate=0.1,
        )
        assert self._captions(ds) == ['Special: a, b, c', 'Some prose, with a comma.'], (
            'raw captions must survive to __getitem__, or prose cannot be told from tags there'
        )

    def test_cached_bakes_augmentation_and_strips_the_marker(self, tmp_path):
        ds = self._dataset(
            tmp_path, ['Special: a, b, c', 'Some prose, with a comma.'], True,
            prefix_tag_caption='Special:', cache_shuffle_num=1,
        )
        captions = self._captions(ds)
        assert not any('Special' in c for c in captions)
        assert sorted(captions[0].split(', ')) == ['a', 'b', 'c']
        assert captions[1] == 'Some prose, with a comma.', 'prose must not be shuffled'

    def test_epoch_length_is_unchanged_by_moving_augmentation(self, tmp_path):
        # cache_shuffle_num still expands to the same number of variants; they are just
        # augmented per access rather than frozen. Changing epoch length would silently
        # rescale every existing training schedule.
        lengths = {}
        for caches in (True, False):
            ds = self._dataset(
                tmp_path / str(caches), ['a, b, c'], caches, cache_shuffle_num=7,
            )
            lengths[caches] = len(self._captions(ds))
        assert lengths[True] == lengths[False] == 7, lengths

    def test_flag_is_off_when_no_augmentation_is_configured(self, tmp_path):
        ds = self._dataset(tmp_path, ['a, b'], False)
        assert not ds.augment_at_runtime

    def test_flag_is_off_when_the_model_caches(self, tmp_path):
        ds = self._dataset(tmp_path, ['a, b'], True, cache_shuffle_num=4, tag_dropout_rate=0.1)
        assert not ds.augment_at_runtime

    def test_flag_is_on_for_dropout_alone(self, tmp_path):
        ds = self._dataset(tmp_path, ['a, b'], False, tag_dropout_rate=0.1)
        assert ds.augment_at_runtime

    def test_no_frozen_dropout_warning_when_augmenting_at_runtime(self, tmp_path, caplog):
        # The warning is about a single frozen draw. That cannot happen per access.
        self._dataset(tmp_path, ['a, b, c'], False, tag_dropout_rate=0.5)
        assert 'permanent tag deletion' not in caplog.text


class TestCaptionCacheFingerprint:
    """Caption settings must reach the metadata cache path, or --trust_cache serves stale text.

    The metadata cache lives at a fixed path and is reused under --trust_cache. Flipping a
    setting that changes the caption text used to hand back captions built under the old one:
    raw text with its tag marker intact going straight into the text encoder, or caption_prefix
    applied a second time and then shuffled into the middle of the tag list.
    """

    def _dataset(self, tmp_path, caches_text_embeddings=True, **overrides):
        d = tmp_path / 'imgs'
        d.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), (128, 128, 128)).save(d / 'a.png')
        (d / 'a.txt').write_text('Special: red, blue, green')
        cfg = directory_config(d, prefix_tag_caption='Special:', caption_prefix='anime, ',
                               cache_shuffle_num=4)
        cfg.update(overrides)
        ds = DirectoryDataset(cfg, {'resolutions': [64]}, 'anima', skip_dataset_validation=True,
                              caches_text_embeddings=caches_text_embeddings)
        ds.cache_metadata(trust_cache=True)
        return ds

    def _captions(self, ds):
        buckets = ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets
        return [c for row in buckets[0].metadata_dataset['caption'] for c in row]

    def test_default_settings_keep_the_legacy_cache_paths(self, tmp_path):
        # Any suffix at the defaults would invalidate every cache in every existing install.
        assert caption_cache_suffix(dict(CAPTION_CACHE_SETTINGS)) == ''
        d = tmp_path / 'plain'
        d.mkdir()
        Image.new('RGB', (64, 64), (128, 128, 128)).save(d / 'a.png')
        (d / 'a.txt').write_text('a, b')
        ds = DirectoryDataset(
            directory_config(d, cache_shuffle_num=3, caption_prefix='anime, '),
            {'resolutions': [64]}, 'flux', skip_dataset_validation=True,
        )
        ds.cache_metadata()
        assert ds.caption_cache_suffix == ''
        written = sorted(p.name for p in (ds.cache_dir / 'metadata').glob('*'))
        assert 'metadata.arrow' in written and 'grouping_keys.json' in written

    def test_flipping_cache_text_embeddings_changes_the_suffix(self, tmp_path):
        a = self._dataset(tmp_path / 'a', caches_text_embeddings=False)
        b = self._dataset(tmp_path / 'b', caches_text_embeddings=True)
        assert a.caption_cache_suffix != b.caption_cache_suffix

    def test_stale_metadata_is_not_served_across_the_flip(self, tmp_path):
        # Same directory, same cache tree, --trust_cache on both runs.
        raw = self._captions(self._dataset(tmp_path, caches_text_embeddings=False))
        baked = self._captions(self._dataset(tmp_path, caches_text_embeddings=True))
        assert all(c.startswith('Special: ') for c in raw), raw
        assert not any('Special' in c for c in baked), baked
        assert all(c.startswith('anime, ') for c in baked), baked
        assert all(c.count('anime, ') == 1 for c in baked), 'caption_prefix applied twice'
        again = self._captions(self._dataset(tmp_path, caches_text_embeddings=False))
        assert again == raw, 'flipping back must not serve the baked captions'

    @pytest.mark.parametrize('setting,value', [
        ('prefix_tag_caption', 'Tags:'),
        ('tag_dropout_rate', 0.25),
        ('multiline_captions', True),
    ])
    def test_each_caption_setting_reaches_the_suffix(self, setting, value):
        base = dict(CAPTION_CACHE_SETTINGS)
        changed = dict(base, **{setting: value})
        assert caption_cache_suffix(changed) != caption_cache_suffix(base), setting

    def test_caption_sampling_separates_the_iteration_order(self, tmp_path):
        # iteration_order stores different COLUMNS per mode. Sharing a path meant a run that
        # flipped the mode read back the wrong ones and died on KeyError inside the dataloader.
        a = self._dataset(tmp_path / 'a', caption_sampling='all')
        b = self._dataset(tmp_path / 'b', caption_sampling='random_per_epoch')
        assert a.caption_sampling == 'all'
        assert b.caption_sampling == 'random_per_epoch'
        # The caption text is identical, so the caption suffix matches; the mode is what
        # separates the two iteration_order directories, appended after that suffix.
        assert a.caption_cache_suffix == b.caption_cache_suffix
        assert a.directory_config['caption_sampling'] != b.directory_config['caption_sampling']


class TestLatentCacheStability:
    """Caption augmentation must not invalidate the VAE latent cache.

    The captions are a column of metadata_dataset, and the latent cache is keyed by that
    dataset's fingerprint. Shuffling and dropout used to draw unseeded, so the captions -- and
    therefore the fingerprint -- differed on every launch, and the whole dataset was re-encoded
    through the VAE every single run. Nothing was gained by it: the variants are frozen into
    the text embedding cache anyway.
    """

    def _captions(self, tmp_path, **overrides):
        d = tmp_path / 'imgs'
        d.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), (128, 128, 128)).save(d / 'a.png')
        (d / 'a.txt').write_text('red, blue, green, yellow')
        ds = DirectoryDataset(directory_config(d, **overrides), {'resolutions': [64]},
                              'anima', skip_dataset_validation=True)
        ds.cache_metadata(regenerate_cache=True)
        buckets = ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets
        md = buckets[0].metadata_dataset
        captions = [c for row in md['caption'] for c in row]
        fingerprint = md._fingerprint
        # Read everything out, then let the arrow files go. A caller that builds the same
        # directory twice is standing in for two training launches, and a real second launch is
        # a fresh process holding no memory maps. Keeping the first build's maps open here is
        # not just untidy: Windows refuses to reopen a mapped .arrow file for writing, so the
        # second save_to_disk fails with EINVAL where Linux would have overwritten it.
        del md, buckets, ds
        gc.collect()
        return captions, fingerprint

    @pytest.mark.parametrize('overrides', [
        {},
        {'cache_shuffle_num': 4},
        {'cache_shuffle_num': 4, 'tag_dropout_rate': 0.3},
        {'tag_dropout_rate': 0.3},
    ])
    def test_metadata_is_reproducible(self, tmp_path, overrides):
        # Same directory, rebuilt twice: exactly what a second training launch does. Two
        # different directories would differ in the fingerprint's lineage regardless.
        a_caps, a_fp = self._captions(tmp_path, **overrides)
        b_caps, b_fp = self._captions(tmp_path, **overrides)
        assert a_caps == b_caps, f'{overrides} produced different captions on a second build'
        assert a_fp == b_fp, f'{overrides} moved the fingerprint the latent cache is keyed by'

    def test_the_variants_are_still_different_from_each_other(self, tmp_path):
        caps, _ = self._captions(tmp_path, cache_shuffle_num=8)
        assert len(caps) == 8
        assert len(set(caps)) > 1, 'seeding must not collapse the variants into one'

    def test_dropout_still_drops(self, tmp_path):
        caps, _ = self._captions(tmp_path, cache_shuffle_num=8, tag_dropout_rate=0.5)
        lengths = {len(c.split(', ')) for c in caps}
        assert min(lengths) < 4, f'nothing was dropped from a 4-tag caption: {caps}'
        assert min(lengths) >= 1, 'a caption was emptied'

    def test_different_images_get_different_draws(self, tmp_path):
        d = tmp_path / 'imgs'
        d.mkdir()
        for name in ('a', 'b', 'c', 'd'):
            Image.new('RGB', (64, 64), (128, 128, 128)).save(d / f'{name}.png')
            (d / f'{name}.txt').write_text('red, blue, green, yellow')
        ds = DirectoryDataset(directory_config(d, cache_shuffle_num=2), {'resolutions': [64]},
                              'anima', skip_dataset_validation=True)
        ds.cache_metadata(regenerate_cache=True)
        buckets = ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets
        caps = [c for row in buckets[0].metadata_dataset['caption'] for c in row]
        assert len(set(caps)) > 1, 'the seed must vary per image, not be one global constant'


def _stub_latents_map_fn(example, rank):
    """Stand in for the VAE call, returning one fixed-size latent per media item.

    Deliberately a module-level function, not a closure: the caching pool pickles it, and on a
    spawn platform a closure over the test's locals is far more fragile than a plain import.
    """
    import torch

    latents = []
    image_specs = []
    captions = []
    masks = []
    for image_spec, caption in zip(example['image_spec'], example['caption']):
        latents.append(torch.zeros(1, 4, 8, 8))
        image_specs.append(image_spec)
        captions.append(caption)
        masks.append(None)
    return {
        'latents': torch.cat(latents),
        'image_spec': image_specs,
        'caption': captions,
        'mask': masks,
    }


class TestLatentCachingRunsForReal:
    """Run cache_latents end to end, rather than around it.

    Everything else in this file stops at the metadata. The latent cache is the one part that
    goes through utils/cache.py, whose sqlite connection uses the autocommit= keyword added in
    Python 3.12 -- so on a 3.11 box this path could not execute at all, and the iteration-order
    code underneath it was only ever covered through the pure helper
    collapse_to_one_entry_per_image. These tests close that gap with a stub for the VAE, which
    is the only genuinely GPU-shaped part of the path.
    """

    def _dataset(self, tmp_path, n_images=2, **overrides):
        d = tmp_path / 'imgs'
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            Image.new('RGB', (64, 64), (10 * i, 20, 30)).save(d / f'{i}.png')
            (d / f'{i}.txt').write_text(f'caption number {i}')
        ds = DirectoryDataset(directory_config(d, **overrides), {'resolutions': [64]},
                              'anima', skip_dataset_validation=True)
        ds.cache_metadata(regenerate_cache=True)
        return ds

    def _cache(self, ds, **kwargs):
        """Cache latents and return the size-bucket datasets that actually hold them.

        With ar buckets in play the object cache_latents is called on is not the object that
        ends up owning latent_dataset: ARBucketDataset builds its size buckets inside the call
        and delegates to them, so the leaves only exist afterwards.
        """
        leaves = []
        for bucket in (ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets):
            bucket.cache_latents(_stub_latents_map_fn, **kwargs)
            if hasattr(bucket, 'get_size_bucket_datasets'):
                leaves.extend(bucket.get_size_bucket_datasets())
            else:
                leaves.append(bucket)
        return leaves

    def test_latents_are_cached_for_every_metadata_row(self, tmp_path):
        ds = self._dataset(tmp_path)
        leaves = self._cache(ds, regenerate_cache=True)
        assert leaves, 'no size bucket dataset was produced'
        for leaf in leaves:
            # cache_latents asserts this itself, but asserting it here is what makes a silent
            # change to that invariant show up as a test failure rather than a training crash.
            assert len(leaf.latent_dataset) == len(leaf.metadata_dataset)

    def test_the_sqlite_cache_is_actually_written(self, tmp_path):
        ds = self._dataset(tmp_path)
        leaves = self._cache(ds, regenerate_cache=True)
        latents_dir = leaves[0].cache_dir / 'latents'
        assert latents_dir.exists(), 'cache_latents produced no latents directory'
        assert any(latents_dir.iterdir()), 'the latents cache directory is empty'

    def test_iteration_order_is_built_and_reused(self, tmp_path):
        ds = self._dataset(tmp_path)
        leaves = self._cache(ds, regenerate_cache=True)
        cache_dir = leaves[0].cache_dir
        first = sorted(p.name for p in cache_dir.iterdir() if p.name.startswith('iteration_order'))
        assert first, 'no iteration_order directory was written'

        # A second pass that trusts the cache must not invent a different directory: the name
        # carries the caption settings, and a second launch has to land on the same one or the
        # cache is silently rebuilt every run.
        self._cache(ds, trust_cache=True)
        again = sorted(p.name for p in cache_dir.iterdir() if p.name.startswith('iteration_order'))
        assert again == first

    def test_every_image_reaches_the_latent_cache(self, tmp_path):
        ds = self._dataset(tmp_path, n_images=3)
        leaves = self._cache(ds, regenerate_cache=True)
        assert sum(len(leaf.latent_dataset) for leaf in leaves) == 3


class TestWindowsPathHandling:
    """Paths that only break where the separator differs from '/'.

    Four instances of this class were fixed once already; these cover the two that were still
    live afterwards. Both assertions are meaningful on Linux too -- they just cannot fail there.
    """

    def test_tar_members_in_a_subdirectory_are_extractable(self, tmp_path):
        # utils/dataset.py extracts by the raw member name. A tar name always uses forward
        # slashes; str(Path(name)) renders backslashes on Windows and extractfile matches
        # literally, so every member in a subdirectory raised KeyError during caching.
        import tarfile
        src = tmp_path / 'src'
        src.mkdir()
        (src / '000001.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'0' * 32)
        tar_path = tmp_path / 'shard.tar'
        with tarfile.TarFile(tar_path, 'w') as tar:
            tar.add(src / '000001.png', arcname='images/000001.png')

        with tarfile.TarFile(tar_path) as tar:
            name = tar.getnames()[0]
            assert name == 'images/000001.png'
            # as_posix() is what utils/dataset.py now uses.
            assert tar.extractfile(Path(name).as_posix()) is not None

    def test_captions_json_is_read_as_utf8(self, tmp_path):
        # open() without an encoding uses the locale codepage. On Windows that does not raise
        # on UTF-8 input -- it silently produces mojibake, which then gets cached and trained.
        import json
        d = tmp_path / 'imgs'
        d.mkdir()
        Image.new('RGB', (64, 64), (128, 128, 128)).save(d / 'a.png')
        caption = '1girl, \u65e5\u672c\u8a9e, caf\u00e9'
        (d / 'captions.json').write_text(
            json.dumps({'a.png': [caption]}, ensure_ascii=False), encoding='utf-8')

        ds = DirectoryDataset(directory_config(d), {'resolutions': [64]}, 'anima',
                              skip_dataset_validation=True)
        ds.cache_metadata(regenerate_cache=True)
        buckets = ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets
        found = [c for row in buckets[0].metadata_dataset['caption'] for c in row]
        assert found == [caption], f'caption was corrupted in transit: {found!r}'


class TestCaptionSettingsAreRecorded:
    """The settings that predate the cache suffix are baked into the cached captions.

    They are deliberately not part of the suffix -- that would move the cache path of every
    install already using them, discarding the latents keyed off it too. So they are recorded
    beside the cache and a change is reported instead of being served silently.
    """

    def _dataset(self, tmp_path, **overrides):
        d = tmp_path / 'imgs'
        d.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), (128, 128, 128)).save(d / 'a.png')
        (d / 'a.txt').write_text('red, blue, green')
        return DirectoryDataset(directory_config(d, **overrides), {'resolutions': [64]},
                                'anima', skip_dataset_validation=True)

    def test_the_settings_file_is_written_next_to_the_cache(self, tmp_path):
        ds = self._dataset(tmp_path, caption_prefix='anime, ')
        ds.cache_metadata(regenerate_cache=True)
        assert ds._caption_settings_file.exists()
        import json
        recorded = json.loads(ds._caption_settings_file.read_text(encoding='utf-8'))
        assert recorded['caption_prefix'] == 'anime, '

    def test_changing_a_recorded_setting_warns_when_the_cache_is_trusted(self, tmp_path, caplog):
        self._dataset(tmp_path, caption_prefix='anime, ').cache_metadata(regenerate_cache=True)
        gc.collect()

        second = self._dataset(tmp_path, caption_prefix='photo, ')
        with caplog.at_level('WARNING'):
            second.cache_metadata(trust_cache=True)
        assert 'caption_prefix' in caplog.text, (
            'a run that changed caption_prefix must be told the cached captions are the old ones'
        )

    def test_an_unchanged_setting_says_nothing(self, tmp_path, caplog):
        self._dataset(tmp_path, caption_prefix='anime, ').cache_metadata(regenerate_cache=True)
        gc.collect()

        second = self._dataset(tmp_path, caption_prefix='anime, ')
        with caplog.at_level('WARNING'):
            second.cache_metadata(trust_cache=True)
        assert 'caption_prefix' not in caplog.text

    def test_a_cache_with_no_settings_file_is_left_alone(self, tmp_path, caplog):
        # An upgrade must be seamless: nothing is known about a cache written before this
        # check existed, so nothing is claimed about it.
        ds = self._dataset(tmp_path, caption_prefix='anime, ')
        ds.cache_metadata(regenerate_cache=True)
        ds._caption_settings_file.unlink()
        gc.collect()

        second = self._dataset(tmp_path, caption_prefix='photo, ')
        with caplog.at_level('WARNING'):
            second.cache_metadata(trust_cache=True)
        assert 'caption_prefix' not in caplog.text


class TestLatentCacheIgnoresCaptions:
    """Latents depend on pixels, never on the text beside them.

    The latent cache was fingerprinted with the metadata dataset's own fingerprint, which covers
    every column -- captions included. Adding a caption_prefix, or changing tag dropout, moved it
    and re-encoded the whole dataset through a VAE that had not changed.
    """

    @staticmethod
    def _latent_fingerprint(captions, image_specs=(('a', 'b'), ('c', 'd'))):
        import datasets
        from datasets.fingerprint import Hasher
        ds = datasets.Dataset.from_dict({
            'image_spec': list(image_specs),
            'size_bucket': [(64, 64, 1)] * len(image_specs),
            'caption': captions,
        })
        hasher = Hasher()
        for column in sorted(c for c in ds.column_names if c != 'caption'):
            hasher.update(column)
            hasher.update(list(ds[column]))
        return hasher.hexdigest()

    def test_changing_the_captions_does_not_move_it(self):
        plain = self._latent_fingerprint([['red'], ['blue']])
        prefixed = self._latent_fingerprint([['anime, red'], ['anime, blue']])
        assert plain == prefixed, 'a caption change must not invalidate the VAE latent cache'

    def test_changing_the_images_does_move_it(self):
        # The decoupling must not go so far that a real change stops invalidating.
        plain = self._latent_fingerprint([['red'], ['blue']])
        other = self._latent_fingerprint([['red'], ['blue']], image_specs=(('a', 'b'), ('X', 'Y')))
        assert plain != other, 'a different set of images must invalidate the latent cache'

    def test_a_lazy_column_would_have_defeated_it(self):
        # Regression guard for the subtle part: dataset[column] returns a lazy Column holding a
        # reference to its parent, so hashing it drags the captions back in. Only list() works.
        import datasets
        from datasets.fingerprint import Hasher
        common = {'image_spec': [('a', 'b')], 'size_bucket': [(64, 64, 1)]}
        a = datasets.Dataset.from_dict({**common, 'caption': [['red']]})
        b = datasets.Dataset.from_dict({**common, 'caption': [['blue']]})
        assert Hasher.hash(list(a['image_spec'])) == Hasher.hash(list(b['image_spec']))
        assert Hasher.hash(a['image_spec']) != Hasher.hash(b['image_spec']), (
            'if this ever becomes equal, datasets stopped returning a lazy Column and the '
            'list() in _map_and_cache can be simplified'
        )


class TestCacheCompatibilityManifest:
    """A cache records what produced it, so an incompatible one is rebuilt rather than served.

    The identity is recorded in a manifest beside the cache, never folded into the fingerprint.
    That is the whole point: folding it in would move the cache path of every existing install
    and rebuild caches that are perfectly good. A cache with no manifest predates this check,
    so nothing is known about it and nothing is claimed.
    """

    @staticmethod
    def _cache(tmp_path, identity, keep=False):
        from utils.cache import Cache
        return Cache(str(tmp_path / 'latents'), 'fingerprint-a',
                     keep_on_fingerprint_change=keep, identity=identity)

    @classmethod
    def _closed_cache(cls, tmp_path, identity):
        """A cache from a previous run: written, then released.

        Production reaches this state by the process exiting. Windows will not delete a sqlite
        file while another handle holds it open, so a test that models two runs has to close
        the first one explicitly or the rebuild fails on a file lock rather than on logic.
        """
        cache = cls._cache(tmp_path, identity)
        cache.write_manifest()
        cache.con.close()
        return cache

    def test_a_cache_with_no_manifest_is_treated_as_compatible(self, tmp_path):
        cache = self._cache(tmp_path, identity='vae-A')
        assert cache.check_identity() == (True, '')
        assert not cache.manifest_file.exists(), 'nothing is claimed until contents are complete'

    def test_the_manifest_is_written_only_when_asked(self, tmp_path):
        cache = self._cache(tmp_path, identity='vae-A')
        cache.write_manifest()
        assert cache.manifest_file.exists()
        assert self._cache(tmp_path, identity='vae-A').check_identity()[0] is True

    def test_a_different_producer_is_incompatible(self, tmp_path):
        # Checked before construction: Cache.__init__ acts on the answer immediately and clears
        # the cache, manifest included, so asking afterwards finds no evidence either way.
        from utils.cache import Cache
        self._closed_cache(tmp_path, identity='vae-A')
        probe = object.__new__(Cache)
        probe.path = tmp_path / 'latents'
        probe.identity = 'vae-B'
        compatible, reason = probe.check_identity()
        assert compatible is False
        assert 'vae-A' in reason and 'vae-B' in reason

    def test_an_incompatible_cache_is_actually_rebuilt(self, tmp_path):
        # The behaviour that matters: opening it with a different producer wipes it.
        self._closed_cache(tmp_path, identity='vae-A')
        cache = self._cache(tmp_path, identity='vae-B')
        assert len(cache) == 0, 'an incompatible cache must be emptied, not served'
        assert not cache.manifest_file.exists(), (
            'the manifest describes contents that no longer exist and must go with them'
        )

    def test_keep_cannot_rescue_an_incompatible_cache(self, tmp_path):
        # keep_* means "do not recache unnecessarily", never "reuse anything". Latents produced
        # by a different VAE are wrong data, and no flag should let them through.
        self._closed_cache(tmp_path, identity='vae-A')
        cache = self._cache(tmp_path, identity='vae-B', keep=True)
        assert len(cache) == 0, 'keep_* must not make an incompatible cache usable'

    def test_a_model_that_declares_no_identity_is_unaffected(self, tmp_path):
        # Every model behaved this way before the manifest existed, and most still do.
        self._closed_cache(tmp_path, identity='vae-A')
        assert self._cache(tmp_path, identity='').check_identity() == (True, '')

    def test_an_unreadable_manifest_does_not_destroy_the_cache(self, tmp_path):
        cache = self._cache(tmp_path, identity='vae-A')
        cache.write_manifest()
        cache.manifest_file.write_text('{ this is not json', encoding='utf-8')
        assert self._cache(tmp_path, identity='vae-A').check_identity() == (True, '')


class TestVaeCacheKeyIsDeclaredPerModel:
    """There is no config key that names the VAE across models, so each declares its own."""

    def test_the_default_records_nothing(self):
        from models.base import BasePipeline
        stub = object.__new__(BasePipeline)
        stub.model_config = {'vae_path': '/models/vae.safetensors', 'dtype': 'bfloat16'}
        assert BasePipeline.vae_cache_key(stub) == '', (
            'a model that has not declared vae_config_keys must record no identity, or adding '
            'this would invalidate its existing caches'
        )

    def test_a_declared_key_produces_an_identity(self):
        from models.base import BasePipeline

        class Declared(BasePipeline):
            vae_config_keys = ('vae_path',)

        stub = object.__new__(Declared)
        stub.model_config = {'vae_path': '/models/vae.safetensors', 'dtype': 'bfloat16'}
        key = Declared.vae_cache_key(stub)
        assert '/models/vae.safetensors' in key and 'bfloat16' in key

    def test_changing_the_vae_changes_the_identity(self):
        from models.base import BasePipeline

        class Declared(BasePipeline):
            vae_config_keys = ('vae_path',)

        def key(path):
            stub = object.__new__(Declared)
            stub.model_config = {'vae_path': path, 'dtype': 'bfloat16'}
            return Declared.vae_cache_key(stub)

        assert key('/models/a.safetensors') != key('/models/b.safetensors')

    def test_every_declaration_names_a_key_the_model_actually_reads(self):
        # A typo would make the identity a constant empty string, silently disabling the check
        # for that model. Checked by reading the source rather than importing: several model
        # modules need GPU-only packages (flash_attn) that a CPU test box does not have.
        import ast as ast_module
        from pathlib import Path as P

        checked = 0
        for path in sorted(P('models').rglob('*.py')):
            tree = ast_module.parse(path.read_text(encoding='utf-8'))
            source = path.read_text(encoding='utf-8')
            for node in ast_module.walk(tree):
                if not isinstance(node, ast_module.Assign):
                    continue
                if not any(getattr(t, 'id', None) == 'vae_config_keys' for t in node.targets):
                    continue
                keys = ast_module.literal_eval(node.value)
                if not keys:
                    # models/base.py declares the empty default every model inherits.
                    continue
                for key in keys:
                    assert repr(key) in source or f'"{key}"' in source, (
                        f'{path} declares vae_config_keys={keys} but never mentions {key!r}, '
                        'so the identity would always be empty'
                    )
                checked += 1
        assert checked >= 10, f'expected the declarations to still be there, found {checked}'
