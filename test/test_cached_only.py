"""use_cached_latents_only / use_cached_embeddings_only: train on the samples whose cache exists,
encoding nothing more."""

import shutil
import sqlite3
import types

import numpy as np
import pytest
import torch

import utils.dataset as dataset_util
from test.test_text_encoding_on_the_fly import (  # noqa: F401  (local_workers is an autouse fixture)
    CAPTIONS, Model, dataset_config, local_workers, make_images, run_caching)

REUSE = pytest.mark.parametrize('reuse_metadata_cache', [True, False])


class CountingModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_calls = 0

    def get_call_vae_fn(self, vae):
        inner = super().get_call_vae_fn(vae)

        def fn(*args):
            self.vae_calls += 1
            return inner(*args)
        return fn

    def encoded_captions(self):
        return [captions for _, captions in self.calls if captions != ['']]


def build(images, model=None, restrict=True, **directory):
    """The datasets a run caches and loads; restrict like DatasetManager.cache does last."""
    model = CountingModel() if model is None else model
    config = dataset_config(images, **directory)
    ds = dataset_util.Dataset(config, model)
    run_caching(model, ds, config)
    if restrict:
        ds.restrict_to_cached()
    return ds, model


def trained_names(ds):
    ds.post_init(0, 1, {None: 1}, 1, {None: 1})
    return sorted(spec[1].rsplit('/', 1)[-1] for k in range(len(ds)) for spec in ds[k]['image_spec'])


def truncate(images, kind, keep):
    """The cache as an interrupted run leaves it: the first `keep` rows, and no profile reference
    (those are written once everything is encoded)."""
    root = images / 'cache'
    for db in list(root.rglob('metadata.db')):
        parts = db.relative_to(root).parts[:-1]
        if any('uncond' in p for p in parts):
            continue
        if not any(p.startswith('latent') if kind == 'latents' else p.startswith(f'text_embeddings_{kind}')
                   for p in parts):
            continue
        connection = sqlite3.connect(db)
        connection.execute('DELETE FROM items WHERE rowid > ?', (keep,))
        connection.commit()
        connection.close()
        if parts[-1].startswith('generated_'):
            parent = db.parents[1]
            (parent / 'references.json').unlink(missing_ok=True)
            for reference in parent.glob('references_*'):
                shutil.rmtree(reference)


def full_caches(tmp_path, reuse_metadata_cache, captions=CAPTIONS, **directory):
    images = make_images(tmp_path / 'images', captions)
    build(images, reuse_metadata_cache=reuse_metadata_cache, restrict=False, **directory)
    return images


def metadata_names(images, reuse_metadata_cache, **directory):
    """The images in the order the buckets cache them (a row's cache position is its index here)."""
    ds, _ = build(images, reuse_metadata_cache=reuse_metadata_cache, restrict=False, **directory)
    bucket = ds.directory_datasets[0].get_size_bucket_datasets()[0]
    return [spec[1].rsplit('/', 1)[-1] for spec in bucket.metadata_dataset['image_spec']]


@REUSE
def test_everything_cached_is_trained_as_before(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache)
    everything = trained_names(build(images, reuse_metadata_cache=reuse_metadata_cache)[0])
    flagged, model = build(images, reuse_metadata_cache=reuse_metadata_cache,
                           use_cached_latents_only=True, use_cached_embeddings_only=True)
    assert trained_names(flagged) == everything == sorted(f'{i}.png' for i in range(6))
    assert model.vae_calls == 0 and model.encoded_captions() == []


@REUSE
def test_latents_only_trains_on_the_images_with_latents(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache)
    order = metadata_names(images, reuse_metadata_cache)
    truncate(images, 'latents', 4)
    ds, model = build(images, reuse_metadata_cache=reuse_metadata_cache, use_cached_latents_only=True)
    assert trained_names(ds) == sorted(order[:4])
    assert model.vae_calls == 0
    # Without the option the missing latents are encoded, as ever.
    full, model = build(images, reuse_metadata_cache=reuse_metadata_cache)
    assert len(trained_names(full)) == 6 and model.vae_calls == 2


@REUSE
def test_embeddings_only_trains_on_the_captions_with_embeddings(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache)
    order = metadata_names(images, reuse_metadata_cache)
    truncate(images, 1, 5)
    truncate(images, 2, 3)
    ds, model = build(images, reuse_metadata_cache=reuse_metadata_cache, use_cached_embeddings_only=True)
    # An image needs the embeddings of every text encoder.
    assert trained_names(ds) == sorted(order[:3])
    assert model.encoded_captions() == [] and model.vae_calls == 0
    # The empty caption's embedding is encoded when missing: it is one caption.
    full, model = build(images, reuse_metadata_cache=reuse_metadata_cache)
    assert len(trained_names(full)) == 6 and len(model.encoded_captions()) > 0


@REUSE
def test_both_options_train_on_the_images_with_both(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache)
    order = metadata_names(images, reuse_metadata_cache)
    truncate(images, 'latents', 4)
    truncate(images, 1, 2)
    truncate(images, 2, 5)
    ds, model = build(images, reuse_metadata_cache=reuse_metadata_cache,
                      use_cached_latents_only=True, use_cached_embeddings_only=True)
    assert trained_names(ds) == sorted(order[:2])
    # Embeddings only: the latents are encoded for the rest, and the images the embeddings cover are trained.
    ds, model = build(images, reuse_metadata_cache=reuse_metadata_cache, use_cached_embeddings_only=True)
    assert trained_names(ds) == sorted(order[:2]) and model.vae_calls == 2


@REUSE
def test_the_cached_data_is_what_training_reads(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache)
    reference = {}
    full, _ = build(images, reuse_metadata_cache=reuse_metadata_cache)
    full.post_init(0, 1, {None: 1}, 1, {None: 1})
    for k in range(len(full)):
        batch = full[k]
        reference[batch['image_spec'][0][1].rsplit('/', 1)[-1]] = batch
    truncate(images, 'latents', 3)
    truncate(images, 1, 4)
    partial, _ = build(images, reuse_metadata_cache=reuse_metadata_cache,
                       use_cached_latents_only=True, use_cached_embeddings_only=True)
    partial.post_init(0, 1, {None: 1}, 1, {None: 1})
    assert len(partial) == 3
    for k in range(len(partial)):
        batch = partial[k]
        expected = reference[batch['image_spec'][0][1].rsplit('/', 1)[-1]]
        for key in ('latents', 'emb1', 'emb2'):
            assert torch.equal(batch[key], expected[key])


@REUSE
def test_nothing_cached_is_an_error(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', CAPTIONS)
    with pytest.raises(RuntimeError, match='nothing to train on'):
        build(images, reuse_metadata_cache=reuse_metadata_cache, use_cached_latents_only=True)


def test_a_cache_made_for_other_images_is_refused_not_cleared(tmp_path):
    images = full_caches(tmp_path, False)
    # One more image: the rows would no longer match the cache.
    make_images(tmp_path / 'more', ['new one'])
    shutil.copy(tmp_path / 'more' / '0.png', images / 'extra.png')
    (images / 'extra.txt').write_text('new one')
    db = next((images / 'cache').rglob('latents/metadata.db'))
    before = db.stat().st_size
    with pytest.raises(RuntimeError, match='cannot be matched'):
        build(images, reuse_metadata_cache=False, use_cached_latents_only=True)
    assert db.exists() and db.stat().st_size == before


@REUSE
def test_random_per_epoch_keeps_the_captions_that_have_embeddings(tmp_path, reuse_metadata_cache):
    images = full_caches(tmp_path, reuse_metadata_cache, ['a\nb\nc', 'd\ne\nf', 'g\nh\ni'],
                         multiline_captions=True, caption_sampling='random_per_epoch')
    truncate(images, 1, 4)
    truncate(images, 2, 4)
    ds, _ = build(images, reuse_metadata_cache=reuse_metadata_cache, use_cached_embeddings_only=True,
                  multiline_captions=True, caption_sampling='random_per_epoch')
    bucket = ds.directory_datasets[0].get_size_bucket_datasets()[0]
    cached = {(tuple(spec), n) for spec, numbers in zip(bucket.iteration_order['image_spec'],
                                                       bucket.iteration_order['caption_numbers'])
              for n in numbers}
    embeddings = bucket.text_embedding_datasets[0]
    assert cached and all(embeddings.has_text_embeddings(spec, n) for spec, n in cached)
    # 4 of the 9 captions are cached, so some images keep fewer than their 3.
    assert 0 < len(cached) <= 4 and len(bucket.iteration_order) < 3 + 1


def test_options_that_do_not_apply_are_refused(tmp_path):
    images = full_caches(tmp_path, True)
    with pytest.raises(ValueError, match='use_cached_embeddings_only does not apply'):
        build(images, model=CountingModel(cache_text_embeddings=False), use_cached_embeddings_only=True)
    with pytest.raises(ValueError, match='use_cached_latents_only must be true or false'):
        build(images, use_cached_latents_only='yes')
    config = dataset_config(images, use_cached_latents_only=True)
    ds = dataset_util.Dataset(config, CountingModel())
    ds.cache_metadata()
    with pytest.raises(ValueError, match='--regenerate_cache'):
        ds.cache_latents(None, regenerate_cache=True)


def test_ranks_use_the_samples_all_of_them_have(monkeypatch):
    import hashlib
    mine = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    theirs = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    other = {'keep': theirs}

    second = np.array([1, 0, 1], dtype=bool)
    calls = []

    def gather(out, obj, group=None):
        calls.append(obj)
        joined = np.concatenate([other['keep'], second])
        out[:] = [obj, np.packbits(joined) if isinstance(obj, np.ndarray)
                  else (hashlib.md5(joined.tobytes()).hexdigest(), other.get('lengths', [10, -1, 3]))]

    monkeypatch.setattr(dataset_util.torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(dataset_util.torch.distributed, 'get_world_size', lambda *a, **k: 2)
    monkeypatch.setattr(dataset_util.torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(dataset_util, 'dist', types.SimpleNamespace(get_world_group=lambda: None))
    agreed = dataset_util.agree_on_kept([mine, None, second])
    assert agreed[0].tolist() == (mine & theirs).tolist() and agreed[1] is None
    assert agreed[2].tolist() == second.tolist()
    # Ranks that agree keep the arrays as they are, after one gather.
    other['keep'] = mine
    calls.clear()
    masks = [mine, None, second]
    assert dataset_util.agree_on_kept(masks) is masks and len(calls) == 1
    # Ranks with other buckets fail together instead of waiting in a second gather.
    other.update(keep=theirs, lengths=[10, 3])
    calls.clear()
    with pytest.raises(RuntimeError, match='different datasets'):
        dataset_util.agree_on_kept([mine, None, second])
    assert len(calls) == 1


def test_the_samples_other_ranks_lack_are_dropped(tmp_path, monkeypatch):
    images = full_caches(tmp_path, True)
    order = metadata_names(images, True)
    truncate(images, 'latents', 4)

    def other_ranks_lack_the_first(masks):
        keep = masks[0].copy()
        keep[np.flatnonzero(keep)[0]] = False
        return [keep, *masks[1:]]

    monkeypatch.setattr(dataset_util, 'agree_on_kept', other_ranks_lack_the_first)
    ds, _ = build(images, reuse_metadata_cache=True, use_cached_latents_only=True)
    assert len(trained_names(ds)) == 3 and set(trained_names(ds)) < set(order[:4])


@REUSE
def test_aspect_ratio_buckets(tmp_path, reuse_metadata_cache):
    ar = dict(size_buckets=None, enable_ar_bucket=True, ar_buckets=[1.0], resolutions=[32])
    images = full_caches(tmp_path, reuse_metadata_cache, **ar)
    truncate(images, 'latents', 4)
    truncate(images, 1, 3)
    ds, model = build(images, reuse_metadata_cache=reuse_metadata_cache,
                      use_cached_latents_only=True, use_cached_embeddings_only=True, **ar)
    names = trained_names(ds)
    assert len(names) == 3 and model.vae_calls == 0 and model.encoded_captions() == []


def test_an_interrupted_cache_keeps_the_rows_it_committed(tmp_path, monkeypatch):
    from utils.cache import Cache
    from utils.cache_profiles import open_prefix_cache
    monkeypatch.setattr(Cache, 'commit_seconds', 0)
    cache = Cache(tmp_path / 'c', 'fp')
    for i in range(3):
        cache.add({'x': torch.full((4,), float(i))})
    # Killed: no finalize, and whatever was not committed is gone.
    cache.con.close()
    cache.shard_file.close()
    partial = open_prefix_cache(tmp_path / 'c', 'fp', 5)
    assert [partial.is_available(i) for i in range(5)] == [True, True, True, False, False]
    assert torch.equal(partial[2]['x'], torch.full((4,), 2.))
    partial.close()
    # A resumed run appends after them.
    cache = Cache(tmp_path / 'c', 'fp')
    assert len(cache) == 3
    for i in range(3, 5):
        cache.add({'x': torch.full((4,), float(i))})
    cache.finalize_current_shard()
    assert [cache[i]['x'][0].item() for i in range(5)] == [0, 1, 2, 3, 4]


class VaeModel(CountingModel):
    def __init__(self, vae='vae A', text_encoder='te A', **kwargs):
        super().__init__(**kwargs)
        self.vae_name, self.text_encoder_name = vae, text_encoder

    def vae_cache_key(self):
        return self.vae_name

    def text_encoder_identity(self, i):
        return f'{self.text_encoder_name} {i}'


def test_a_cache_made_by_another_encoder_is_refused(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS)
    build(images, model=VaeModel(), reuse_metadata_cache=False, restrict=False)
    truncate(images, 'latents', 4)
    truncate(images, 1, 3)
    with pytest.raises(RuntimeError, match="made by 'vae A'"):
        build(images, model=VaeModel(vae='vae B'), reuse_metadata_cache=False, use_cached_latents_only=True)
    with pytest.raises(RuntimeError, match="made by 'te A 1'"):
        build(images, model=VaeModel(text_encoder='te B'), reuse_metadata_cache=False,
              use_cached_embeddings_only=True)
    ds, _ = build(images, model=VaeModel(), reuse_metadata_cache=False,
                  use_cached_latents_only=True, use_cached_embeddings_only=True)
    assert len(trained_names(ds)) == 3


def test_an_interrupted_cache_records_its_encoder_from_the_start(tmp_path):
    from utils.cache import Cache
    cache = Cache(tmp_path / 'c', 'fp', identity='vae A')
    cache.add({'x': torch.zeros(4)})
    cache.commit_written()
    cache.con.close()
    cache.shard_file.close()
    # Resumed with another encoder: its rows are not appended to the other one's.
    assert len(Cache(tmp_path / 'c', 'fp', identity='vae B')) == 0


def test_kept_cache_with_an_edited_caption_is_refused(tmp_path):
    images = full_caches(tmp_path, False, keep_text_embedding_cache=True)
    first = sorted(images.glob('*.txt'))[0]
    first.write_text('an edited caption')
    with pytest.raises(RuntimeError, match='cannot be matched'):
        build(images, reuse_metadata_cache=False, keep_text_embedding_cache=True,
              use_cached_embeddings_only=True)


def test_a_cache_whose_writer_was_killed_mid_transaction_opens(tmp_path):
    import subprocess
    import sys
    from utils.cache_profiles import open_prefix_cache
    script = f'''
import os, sys, torch
sys.path.insert(0, {str(dataset_util.__file__.rsplit('/utils/', 1)[0])!r})
from utils.cache import Cache
Cache.commit_seconds = 10 ** 9
cache = Cache({str(tmp_path / 'c')!r}, 'fp')
for i in range(3):
    cache.add({{'x': torch.full((4,), float(i))}})
cache.commit_written()
cache.con.execute('PRAGMA cache_size = 1')
for i in range(3, 3000):
    cache.add({{'x': torch.full((4,), float(i))}})
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script], check=True)
    assert (tmp_path / 'c' / 'metadata.db-journal').exists()
    partial = open_prefix_cache(tmp_path / 'c', 'fp', 5000)
    assert partial.count == 3 and torch.equal(partial[2]['x'], torch.full((4,), 2.))
    partial.close()


def test_ranks_agree_on_the_captions_a_row_keeps(tmp_path, monkeypatch):
    kwargs = dict(multiline_captions=True, caption_sampling='random_per_epoch')
    images = full_caches(tmp_path, True, ['a\nb\nc', 'd\ne\nf', 'g\nh\ni'], **kwargs)

    def other_ranks_lack_the_first_caption(masks):
        keep = masks[0].copy()
        keep[0] = False
        return [keep, *masks[1:]]

    monkeypatch.setattr(dataset_util, 'agree_on_kept', other_ranks_lack_the_first_caption)
    ds, _ = build(images, reuse_metadata_cache=True, use_cached_embeddings_only=True, **kwargs)
    order = ds.directory_datasets[0].get_size_bucket_datasets()[0].iteration_order
    assert sorted(len(numbers) for numbers in order['caption_numbers']) == [2, 3, 3]
