"""--cache_status: how many files have their latents and text embeddings cached, read without
encoding, clearing or writing anything; and the same count shown while caching."""

import datasets
import numpy as np
import pytest

import utils.dataset as dataset_util
from test.test_cached_only import CountingModel, full_caches, truncate
from test.test_text_encoding_on_the_fly import (  # noqa: F401  (local_workers is an autouse fixture)
    CAPTIONS, Model, dataset_config, local_workers, make_images, run_caching)

REUSE = pytest.mark.parametrize('reuse_metadata_cache', [True, False])
TWO_CAPTIONS = ['a\nb', 'c\nd', 'e\nf']


def status_of(images, model=None, encoders=2, trust_cache=True, **directory):
    model = CountingModel() if model is None else model
    ds = dataset_util.Dataset(dataset_config(images, **directory), model)
    ds.cache_metadata(trust_cache=trust_cache)
    return dataset_util.cache_status(ds, encoders), ds, model


def only_bucket(status):
    (directory,) = status
    (bucket,) = directory['buckets']
    return bucket


@REUSE
def test_everything_cached(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache)
    status, _, model = status_of(images, reuse_metadata_cache=reuse_metadata_cache)
    bucket = only_bucket(status)
    assert (bucket['files'], bucket['latents'], bucket['text'], bucket['notes']) == (6, 6, [(6, 0), (6, 0)], [])
    assert model.vae_calls == 0 and model.calls == []


@REUSE
def test_an_interrupted_cache_counts_the_files_it_finished(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache, TWO_CAPTIONS, multiline_captions=True)
    truncate(images, 'latents', 2)
    # Two captions to a file: 3 rows are one whole file and half of the next.
    truncate(images, 1, 3)
    truncate(images, 2, 4)
    status, _, _ = status_of(images, reuse_metadata_cache=reuse_metadata_cache, multiline_captions=True)
    bucket = only_bucket(status)
    assert bucket['files'] == 3 and bucket['latents'] == 2
    assert bucket['text'] == [(1, 1), (2, 0)]
    lines = dataset_util.describe_cache_status(status, 2)
    assert any('latents: 2 of 3 files (66.7%)' in line for line in lines)
    assert any('text encoder 1: 1 of 3 files (33.3%); 1 more have the embeddings of only some of their captions'
               in line for line in lines)
    assert any('text encoder 2: 2 of 3 files' in line and 'only some' not in line for line in lines)


@REUSE
def test_nothing_cached_is_zero_and_writes_nothing(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', CAPTIONS)
    status, ds, model = status_of(images, reuse_metadata_cache=reuse_metadata_cache)
    bucket = only_bucket(status)
    assert (bucket['files'], bucket['latents'], bucket['text'], bucket['notes']) == (6, 0, [(0, 0), (0, 0)], [])
    assert not list(images.rglob('metadata.db')) and not list(images.rglob('iteration_order*'))
    assert model.vae_calls == 0 and model.calls == []
    directory = ds.directory_datasets[0]
    assert not directory.use_cached_latents_only and not directory.use_cached_embeddings_only
    assert not getattr(directory, 'reading_cache_status', False)


def test_a_cache_for_other_inputs_is_reported_and_kept(tmp_path):
    images = full_caches(tmp_path, False)
    make_images(tmp_path / 'more', ['new one'])
    (images / 'extra.png').write_bytes((tmp_path / 'more' / '0.png').read_bytes())
    (images / 'extra.txt').write_text('new one')
    db = next((images / 'cache').rglob('latents/metadata.db'))
    before = db.stat().st_size
    status, _, _ = status_of(images, trust_cache=False, reuse_metadata_cache=False)
    bucket = only_bucket(status)
    assert bucket['files'] == 7 and bucket['latents'] is None
    assert any('latents not readable' in note and 'cannot be matched' in note for note in bucket['notes'])
    assert db.exists() and db.stat().st_size == before
    assert any('latents not readable' in line for line in dataset_util.describe_cache_status(status, 2))


@REUSE
def test_aspect_ratio_buckets(tmp_path, reuse_metadata_cache):
    ar = dict(size_buckets=None, enable_ar_bucket=True, ar_buckets=[1.0], resolutions=[32])
    images = full_caches(tmp_path, reuse_metadata_cache, **ar)
    truncate(images, 'latents', 4)
    truncate(images, 1, 3)
    status, _, _ = status_of(images, reuse_metadata_cache=reuse_metadata_cache, **ar)
    bucket = only_bucket(status)
    assert (bucket['files'], bucket['latents'], bucket['text']) == (6, 4, [(3, 0), (6, 0)])


def test_text_embeddings_encoded_while_training(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS)
    status, _, _ = status_of(images, model=CountingModel(cache_text_embeddings=False), encoders=0,
                             reuse_metadata_cache=False)
    assert status[0]['text_cached'] is False
    lines = dataset_util.describe_cache_status(status, 0)
    assert any('none cached, the captions are encoded while training' in line for line in lines)


class OnTheFlyModel(CountingModel):
    load_media_on_the_fly = True


@REUSE
@pytest.mark.parametrize('ar', [False, True])
def test_media_read_while_training_still_counts_the_text_embeddings(tmp_path, reuse_metadata_cache, ar):
    directory = (dict(size_buckets=None, enable_ar_bucket=True, ar_buckets=[1.0], resolutions=[32]) if ar else {})
    images = full_caches(tmp_path, reuse_metadata_cache, **directory)
    truncate(images, 1, 4)
    status, _, _ = status_of(images, model=OnTheFlyModel(), reuse_metadata_cache=reuse_metadata_cache, **directory)
    assert status[0]['latents_cached'] is False
    bucket = only_bucket(status)
    assert bucket['latents'] is None and bucket['text'] == [(4, 0), (6, 0)] and bucket['notes'] == []
    assert any('latents: none cached' in line for line in dataset_util.describe_cache_status(status, 2))


def test_files_are_counted_not_captions():
    spec = [[None, f'{i // 2}.png'] for i in range(7)]
    ends = dataset_util._file_ends(datasets.Dataset.from_dict({'image_spec': spec, 'caption': list('abcdefg')}))
    assert ends.tolist() == [2, 4, 6, 7]
    assert [dataset_util._files_done(ends, rows) for rows in (0, 1, 2, 3, 6, 7)] == [0, 0, 1, 1, 3, 4]
    assert dataset_util._file_ends(datasets.Dataset.from_dict({'caption': ['a']})) is None


def test_caching_shows_the_files_done(tmp_path, monkeypatch):
    shown = []

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.description = kwargs.get('desc')

        def set_postfix_str(self, text):
            shown.append((self.description, text))

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(dataset_util, 'tqdm', Progress)
    images = make_images(tmp_path / 'images', TWO_CAPTIONS)
    config = dataset_config(images, multiline_captions=True, reuse_metadata_cache=False)
    model = CountingModel()
    run_caching(model, dataset_util.Dataset(config, model), config)
    latents = [text for desc, text in shown if desc == 'latents']
    embeddings = [text for desc, text in shown if desc == 'text_embeddings_1']
    assert latents[0] == '0/3 files' and latents[-1] == '3/3 files'
    # Six captions, two to a file: the count moves a file at a time.
    assert embeddings[0] == '0/3 files' and embeddings[-1] == '3/3 files'
    assert '1/3 files' in embeddings and '2/3 files' in embeddings
