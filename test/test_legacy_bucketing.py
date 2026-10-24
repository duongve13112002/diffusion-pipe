"""reuse_metadata_cache = false with --trust_cache: grouped metadata is reused only while the
buckets it was grouped by are still the dataset's buckets."""

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import utils.dataset as dataset_module  # noqa: E402
from utils.dataset import DirectoryDataset  # noqa: E402

SIZES = [(64, 32), (48, 32), (40, 32), (32, 32), (32, 40), (32, 48), (32, 64), (56, 32), (32, 56)]


@pytest.fixture(autouse=True)
def in_process(monkeypatch):
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)


def make_data(path):
    path.mkdir()
    for i, (width, height) in enumerate(SIZES):
        Image.new('RGB', (width, height), (i * 20, 0, 0)).save(path / f'{i}.png')
        (path / f'{i}.txt').write_text(f'tag{i}\nsecond line {i}')
    return path


def directory(path, num_ar_buckets=5, **settings):
    config = {'path': str(path), 'num_repeats': 1, 'shuffle_metadata': False,
              'reuse_metadata_cache': False, **settings}
    return DirectoryDataset(config, {'resolutions': [64], 'enable_ar_bucket': True, 'min_ar': 0.5, 'max_ar': 2.0,
                                     'num_ar_buckets': num_ar_buckets, 'frame_buckets': [1]},
                            'bucketing', skip_dataset_validation=True)


def grouping(ds):
    return sorted((round(bucket.ar_frames[0], 3), sorted(bucket.metadata_dataset['image_spec']))
                  for bucket in ds.ar_bucket_datasets)


def relative(grouping_):
    return [(ar, sorted(Path(spec[1]).name for spec in specs)) for ar, specs in grouping_]


def test_new_buckets_that_include_the_old_ones_regroup(tmp_path):
    """9 buckets from 0.5 to 2.0 include the 5 of before: every old key fits, yet the images
    belong in the finer buckets now."""
    data = make_data(tmp_path / 'data')
    directory(data, 5).cache_metadata(trust_cache=True)
    ds = directory(data, 9)
    ds.cache_metadata(trust_cache=True)
    expected = directory(make_data(tmp_path / 'fresh'), 9)
    expected.cache_metadata(trust_cache=True)
    assert relative(grouping(ds)) == relative(grouping(expected))
    assert len(ds.ar_bucket_datasets) > 5
    # Recorded: the next run loads it as it is.
    assert json.loads(ds._bucketing_file().read_text()) == ds._bucketing()
    assert directory(data, 9)._legacy_grouping_is_stale() is None


def test_metadata_grouped_before_the_record_by_buckets_that_no_longer_exist(tmp_path):
    data = make_data(tmp_path / 'data')
    old = directory(data, 5)
    old.cache_metadata(trust_cache=True)
    old._bucketing_file().unlink()  # as a cache written before the buckets were recorded
    ds = directory(data, 4)
    assert 'other buckets' in ds._legacy_grouping_is_stale()
    ds.cache_metadata(trust_cache=True)
    expected = directory(make_data(tmp_path / 'fresh'), 4)
    expected.cache_metadata(trust_cache=True)
    assert relative(grouping(ds)) == relative(grouping(expected))


def test_metadata_grouped_before_the_record_by_the_same_buckets_is_kept_and_recorded(tmp_path):
    data = make_data(tmp_path / 'data')
    old = directory(data, 5)
    old.cache_metadata(trust_cache=True)
    old._bucketing_file().unlink()
    ds = directory(data, 5)
    assert ds._legacy_grouping_is_stale() is None
    scanned = []
    original = DirectoryDataset._get_ungrouped_metadata
    ds._get_ungrouped_metadata = lambda *args, **kwargs: scanned.append(1) or original(ds, *args, **kwargs)
    ds.cache_metadata(trust_cache=True)
    assert not scanned
    assert json.loads(ds._bucketing_file().read_text()) == ds._bucketing()


def image_latents(batch, rank):
    values = [int(Path(spec[1]).stem) for spec in batch['image_spec']]
    return {'latents': torch.tensor(values).view(-1, 1), 'image_spec': batch['image_spec'],
            'mask': [None] * len(values), 'caption': batch['caption']}


def bucket_captions(ds):
    return sorted((Path(row['image_spec'][1]).name, tuple(row['caption']))
                  for bucket in ds.get_size_bucket_datasets() for row in bucket.metadata_dataset)


def test_each_caption_profile_keeps_its_own_captions_per_bucket(tmp_path):
    """Two caption settings of one directory share its cache directory. The per-resolution copy
    of an AR bucket's metadata carries the captions, so each profile needs its own."""
    data = make_data(tmp_path / 'data')
    first = directory(data, 5)
    first.cache_metadata(trust_cache=True)
    first.cache_latents(image_latents, trust_cache=True)
    multiline = directory(data, 5, multiline_captions=True)
    multiline.cache_metadata(trust_cache=True)
    multiline.cache_latents(image_latents, trust_cache=True)
    assert all(len(captions) == 2 for _, captions in bucket_captions(multiline))
    assert all(len(captions) == 1 for _, captions in bucket_captions(first))


def test_regrouping_drops_the_per_resolution_copies_of_the_old_buckets(tmp_path):
    """A bucket key of both groupings (1.0 is among 5 and 9 buckets) holds other images after
    the regrouping: the copy made for the old grouping must not be loaded again."""
    data = make_data(tmp_path / 'data')
    old = directory(data, 5)
    old.cache_metadata(trust_cache=True)
    old.cache_latents(image_latents, trust_cache=True)
    ds = directory(data, 9)
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(image_latents, trust_cache=True)
    for ar_bucket in ds.ar_bucket_datasets:
        expected = sorted(map(tuple, ar_bucket.metadata_dataset['image_spec']))
        for size_bucket in ar_bucket.get_size_bucket_datasets():
            assert sorted(map(tuple, size_bucket.metadata_dataset['image_spec'])) == expected


def test_an_interrupted_scan_is_not_trusted(tmp_path, monkeypatch):
    """A run with 9 buckets killed after its ungrouped metadata was saved: back on 5 buckets,
    whose record it replaced, --trust_cache must not load the 9-bucket rows."""
    data = make_data(tmp_path / 'data')
    directory(data, 5).cache_metadata(trust_cache=True)
    original = DirectoryDataset._get_ungrouped_metadata

    def killed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(DirectoryDataset, '_get_ungrouped_metadata', killed)
    with pytest.raises(KeyboardInterrupt):
        directory(data, 9).cache_metadata()
    monkeypatch.setattr(DirectoryDataset, '_get_ungrouped_metadata', original)
    ds = directory(data, 4)
    assert 'interrupted' in ds._legacy_grouping_is_stale()
    ds.cache_metadata(trust_cache=True)
    expected = directory(make_data(tmp_path / 'fresh'), 4)
    expected.cache_metadata(trust_cache=True)
    assert relative(grouping(ds)) == relative(grouping(expected))


def test_size_buckets_changed(tmp_path):
    data = make_data(tmp_path / 'data')
    old = DirectoryDataset({'path': str(data), 'num_repeats': 1, 'shuffle_metadata': False, 'reuse_metadata_cache': False},
                           {'size_buckets': [[64, 64, 1], [64, 32, 1]]}, 'sizes', skip_dataset_validation=True)
    old.cache_metadata(trust_cache=True)
    new = DirectoryDataset({'path': str(data), 'num_repeats': 1, 'shuffle_metadata': False, 'reuse_metadata_cache': False},
                           {'size_buckets': [[64, 64, 1], [64, 32, 1], [32, 64, 1]]}, 'sizes', skip_dataset_validation=True)
    assert 'other buckets' in new._legacy_grouping_is_stale()
    new.cache_metadata(trust_cache=True)
    assert sorted(tuple(bucket.size_bucket) for bucket in new.size_bucket_datasets) == [(32, 64, 1), (64, 32, 1), (64, 64, 1)]


def test_a_train_and_an_eval_dataset_with_other_buckets_on_one_directory_are_refused(tmp_path):
    from types import SimpleNamespace
    data = make_data(tmp_path / 'data')
    train = SimpleNamespace(directory_datasets=[directory(data, 5)])
    same = SimpleNamespace(directory_datasets=[directory(data, 5)])
    other_captions = SimpleNamespace(directory_datasets=[directory(data, 9, multiline_captions=True)])
    dataset_module.check_shared_legacy_caches([train, same, other_captions])
    with pytest.raises(ValueError, match='same caption settings and other buckets'):
        dataset_module.check_shared_legacy_caches([train, SimpleNamespace(directory_datasets=[directory(data, 9)])])


def test_a_regrouping_keeps_the_other_caption_profiles_caches(tmp_path):
    data = make_data(tmp_path / 'data')
    for settings in ({}, {'multiline_captions': True}):
        ds = directory(data, 5, **settings)
        ds.cache_metadata(trust_cache=True)
        ds.cache_latents(image_latents, trust_cache=True)
    multiline = directory(data, 5, multiline_captions=True)
    kept_orders = sorted(data.glob(f'cache/bucketing/cache_*/iteration_order{multiline.caption_cache_suffix}'))
    kept_copies = sorted(data.glob(f'cache/bucketing/ar_frames_*/metadata/metadata_*_captions{multiline.caption_cache_suffix}.arrow'))
    assert kept_orders and kept_copies
    directory(data, 9).cache_metadata(trust_cache=True)
    assert sorted(data.glob(f'cache/bucketing/cache_*/iteration_order{multiline.caption_cache_suffix}')) == kept_orders
    assert all(copy.exists() for copy in kept_copies)
    assert not list(data.glob('cache/bucketing/cache_*/iteration_order'))
    assert not list(data.glob('cache/bucketing/ar_frames_*/metadata/metadata_*_captions.arrow'))


def test_the_copy_name_every_profile_once_shared_is_not_loaded(tmp_path):
    """Written by another caption profile before the copies were named per profile."""
    data = make_data(tmp_path / 'data')
    other = directory(data, 5, multiline_captions=True)
    other.cache_metadata(trust_cache=True)
    other.cache_latents(image_latents, trust_cache=True)
    for copy in data.glob('cache/bucketing/ar_frames_*/metadata/metadata_*_captions_*.arrow'):
        copy.rename(copy.with_name(copy.name.split('_captions')[0] + '.arrow'))
    ds = directory(data, 5)
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(image_latents, trust_cache=True)
    assert all(len(captions) == 1 for _, captions in bucket_captions(ds))
    assert not [copy for copy in data.glob('cache/bucketing/ar_frames_*/metadata/metadata_*.arrow')
                if '_captions' not in copy.name]
