"""Construct real dataset objects.

This file exists because 182 tests passed while `utils/dataset.py` was missing three names it
calls at runtime -- `bucket_suffix`, `dedup_and_sort`, `seed_from_hash`, accidentally moved into
`utils/captions.py`. Every training run was broken. The module still *imported* cleanly, and no
test in the repo had ever constructed a `DirectoryDataset`, `ARBucketDataset` or
`SizeBucketDataset`, so nothing noticed.

The lesson is narrow and worth keeping: an import test proves a module parses, not that it runs.
These tests build the real objects against a handful of real (tiny) images.
"""

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
        return [c for row in md['caption'] for c in row], md._fingerprint

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
