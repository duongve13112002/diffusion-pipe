"""Migrate real abd164e-format mock shards, then switch caption configurations."""

import gc
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys
import types
import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import datasets
from PIL import Image

from utils import dataset as current
from utils.cache_profiles import tensor_profile
from test.test_caption_filter_dropout import text_map, assert_embedding


def image_latents(batch, rank):
    values = [int(Path(spec[1]).stem) for spec in batch['image_spec']]
    return {'latents': torch.tensor(values).view(-1, 1), 'image_spec': batch['image_spec'],
            'mask': [None] * len(values), 'caption': batch['caption']}


class LocalPool:
    """Run the tiny encoder mocks without importing a historical module in spawned workers."""

    def __init__(self, count, initializer, args):
        self.executor = ThreadPoolExecutor(count, initializer=initializer, initargs=args)

    def imap(self, function, iterable):
        return self.executor.map(function, iterable)

    def close(self):
        self.executor.shutdown(wait=True)


LOCAL_WORKERS = types.SimpleNamespace(
    Manager=lambda: types.SimpleNamespace(Queue=queue.Queue), Pool=LocalPool)


def close_directory(ds):
    for bucket in ds.get_size_bucket_datasets():
        for te in bucket.text_embedding_datasets:
            if hasattr(te, 'close'):
                te.close()
        for cache in [getattr(bucket, 'latent_dataset', None),
                      *[te.te_dataset for te in bucket.text_embedding_datasets],
                      *bucket.uncond_text_embeddings]:
            if cache is None:
                continue
            if hasattr(cache, 'close'):
                cache.close()
            else:
                cache.con.close()
                for handle in cache.open_files.values():
                    handle.close()
        bucket.iteration_order = None
    gc.collect()


@pytest.fixture(scope='module')
def legacy_module():
    root = Path(__file__).resolve().parents[1]
    module = types.ModuleType('abd164e_dataset_fixture')
    module.__file__ = str(root / 'utils/dataset.py')
    sys.modules[module.__name__] = module
    try:
        source = subprocess.check_output(['git', 'show', 'abd164e:utils/dataset.py'], cwd=root).decode('utf-8')
        cache_source = subprocess.check_output(['git', 'show', 'abd164e:utils/cache.py'], cwd=root).decode('utf-8')
    except subprocess.CalledProcessError:
        pytest.skip('Historical migration matrix requires the abd164e Git object (use a full-history checkout).')
    exec(compile(source, 'abd164e:utils/dataset.py', 'exec'), module.__dict__)
    cache = types.ModuleType('abd164e_cache_fixture')
    exec(compile(cache_source, 'abd164e:utils/cache.py', 'exec'), cache.__dict__)
    module.Cache = cache.Cache
    module.NUM_PROC = 1
    module.mp = LOCAL_WORKERS
    return module


@pytest.fixture(scope='module', params=[('json', 'size'), ('txt', 'size'), ('json', 'ar'), ('txt', 'ar')])
def old_cache(request, tmp_path_factory, legacy_module):
    source, geometry = request.param
    path = tmp_path_factory.mktemp(f'abd164e_{source}_{geometry}')
    captions = {'0.png': ['red\n制服', 'blue'], '1.png': ['outdoor', 'a boy'],
                '2.png': ['Привет'], '3.png': ['green']}
    for name, variants in captions.items():
        Image.new('RGB', (64, 64)).save(path / name)
        if source == 'txt':
            path.joinpath(Path(name).with_suffix('.txt')).write_text('\n'.join(variants), encoding='utf-8')
    if source == 'json':
        path.joinpath('captions.json').write_text(json.dumps(captions), encoding='utf-8')
    config = {'resolutions': [64]}
    directory = {'path': str(path), 'shuffle_metadata': False}
    if geometry == 'size':
        directory['size_buckets'] = [[64, 64, 1]]
    old = legacy_module.DirectoryDataset(directory, config, 'migration_test', skip_dataset_validation=True)
    old.cache_metadata()
    old.cache_latents(image_latents)
    old.cache_text_embeddings(text_map, 0)
    leaf = old.get_size_bucket_datasets()[0]
    original_captions = list(leaf.text_embedding_datasets[0].flattened_captions['caption'])
    encoder_dir = leaf.cache_dir if geometry == 'size' else old.ar_bucket_datasets[0].cache_dir
    manifest = json.loads((encoder_dir / 'text_embeddings_0/cache_manifest.json').read_text())
    assert manifest['content_digest'] == legacy_module.Hasher.hash(original_captions), original_captions
    encoder_metadata = leaf.metadata_dataset if geometry == 'size' else old.ar_bucket_datasets[0].metadata_dataset
    assert original_captions == [c for row in encoder_metadata for c in row['caption']], original_captions
    rebuilt = encoder_metadata.map(lambda b: {'caption': [c for cs in b['caption'] for c in cs]},
                                       batched=True, keep_in_memory=True,
                                       remove_columns=encoder_metadata.column_names)
    assert legacy_module.Hasher.hash(list(rebuilt['caption'])) == manifest['content_digest'], (
        [(type(c), repr(c)) for c in original_captions], [(type(c), repr(c)) for c in rebuilt['caption']])
    baseline = {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
                for file in path.joinpath('cache').rglob('*') if file.is_file()}
    close_directory(old)
    path.joinpath('fixture.json').write_text(json.dumps({'format_commit': 'abd164e', 'source': source,
                                                       'geometry': geometry,
                                                       'legacy_hashes': baseline}), encoding='utf-8')
    return path, baseline, source


@pytest.fixture(autouse=True)
def bounded_workers(monkeypatch):
    monkeypatch.setattr(current, 'NUM_PROC', 1)
    monkeypatch.setattr(current, 'mp', LOCAL_WORKERS)


def build(path, cached=True, **settings):
    directory = {'path': str(path), 'shuffle_metadata': False, **settings}
    if not path.joinpath('fixture.json').exists() or json.loads(path.joinpath('fixture.json').read_text())['geometry'] == 'size':
        directory.setdefault('size_buckets', [[64, 64, 1]])
    return current.DirectoryDataset(directory, {'resolutions': [64]}, 'migration_test',
                                    skip_dataset_validation=True, caches_text_embeddings=cached)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('requirement', [None, True, False])
@pytest.mark.parametrize('remove', [False, True])
@pytest.mark.parametrize('dropout', [0.0, 1.0])
@pytest.mark.parametrize('trust', [False, True])
def test_legacy_configuration_matrix(old_cache, monkeypatch, cached, requirement, remove, dropout, trust):
    path, hashes, source = old_cache
    ds = build(path, cached, require_non_latin_caption=requirement,
               enable_remove_non_latin=remove, caption_dropout_rate=dropout)
    # Migration and profile switches must not enumerate or open source media.
    monkeypatch.setattr(current.Image, 'open', lambda *a, **kw: pytest.fail('unexpected image scan'))
    ds.cache_metadata(trust_cache=trust)
    monkeypatch.setattr(ds, 'legacy_tensor_sources', ds.legacy_tensor_sources)
    ds.cache_latents(None, trust_cache=trust)
    if cached:
        ds.cache_text_embeddings(text_map if remove else None, 0)
    expected_ids = {0, 1, 2, 3} if requirement is None else ({0, 2} if requirement else {1, 3})
    selected_ids = set()
    for bucket in ds.get_size_bucket_datasets():
        for index in range(len(bucket)):
            item = bucket[index]
            assert item['image_spec'] == bucket.iteration_order[index % len(bucket.iteration_order)]['image_spec']
            selected_ids.add(item['latents'].item())
            assert item['latents'].item() == int(Path(item['image_spec'][1]).stem)
            if dropout:
                assert item['caption'] == ''
            elif remove:
                assert not current.caption_matches_non_latin_requirement([item['caption']], True)
            if cached:
                assert_embedding(item, item['caption'])
            else:
                assert not any(key.startswith('embedding_') for key in item)
                # A tiny online encoder consumes the exact runtime caption.
                encoded = text_map({'caption': [item['caption']]}, 0)
                assert encoded['embedding_0'][0, 1].item() == len(item['caption'])
    assert selected_ids == expected_ids
    assert all(hashlib.sha256(path.joinpath(file).read_bytes()).hexdigest() == digest
               for file, digest in hashes.items())
    close_directory(ds)


def test_warm_profile_needs_no_sources_or_encoders(old_cache, monkeypatch):
    path, _, _ = old_cache
    ds = build(path, require_non_latin_caption=True, enable_remove_non_latin=True)
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(None, trust_cache=True)
    ds.cache_text_embeddings(text_map, 0)
    close_directory(ds)
    again = build(path, require_non_latin_caption=True, enable_remove_non_latin=True,
                  caption_dropout_rate=1.0, num_repeats=2, caption_sampling='random_per_epoch')
    monkeypatch.setattr(again, '_load_source_snapshot', lambda: pytest.fail('warm metadata rebuilt'))
    monkeypatch.setattr(again, 'legacy_tensor_sources', lambda *a: pytest.fail('warm donor index rebuilt'))
    again.cache_metadata(trust_cache=True)
    again.cache_latents(None, trust_cache=True)
    again.cache_text_embeddings(None, 0)
    assert all(bucket[i]['caption'] == '' for bucket in again.get_size_bucket_datasets()
               for i in range(len(bucket)))
    close_directory(again)


def new_source(path):
    for index, caption in enumerate(['red\n制服', 'blue']):
        Image.new('RGB', (64, 64)).save(path / f'{index}.png')
        path.joinpath(f'{index}.txt').write_text(caption, encoding='utf-8')


def test_cold_cache_donates_latents_and_embeddings_to_new_selection(tmp_path):
    new_source(tmp_path)
    first = build(tmp_path)
    first.cache_metadata()
    first.cache_latents(image_latents)
    first.cache_text_embeddings(text_map, 0)
    close_directory(first)
    selected = build(tmp_path, require_non_latin_caption=False)
    selected.cache_metadata(trust_cache=True)
    selected.cache_latents(None, trust_cache=True)
    selected.cache_text_embeddings(None, 0)
    bucket = selected.get_size_bucket_datasets()[0]
    assert len(bucket) == 1 and bucket[0]['latents'].item() == 1
    assert_embedding(bucket[0], 'blue')
    close_directory(selected)


def test_selection_profiles_share_the_donor_lookup_index(tmp_path):
    new_source(tmp_path)
    first = build(tmp_path)
    first.cache_metadata()
    first.cache_latents(image_latents)
    first.cache_text_embeddings(text_map, 0)
    close_directory(first)
    indices = None
    for requirement in (False, True):
        ds = build(tmp_path, require_non_latin_caption=requirement)
        ds.cache_metadata(trust_cache=True)
        ds.cache_latents(None, trust_cache=True)
        ds.cache_text_embeddings(None, 0)
        current_indices = {file.name: file.stat().st_mtime_ns
                           for file in (ds.snapshot_dir / 'tensor_lookups').glob('*.npz')}
        if indices is not None:
            assert current_indices == indices
        indices = current_indices
        close_directory(ds)


def test_snapshot_dropout_switches_multiple_encoders_together(tmp_path):
    import functools
    new_source(tmp_path)
    ds = build(tmp_path, require_non_latin_caption=True, caption_dropout_rate=1.0)
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(image_latents, trust_cache=True)
    for encoder in (0, 1):
        ds.cache_text_embeddings(functools.partial(text_map, encoder=encoder), encoder)
    item = ds.get_size_bucket_datasets()[0][0]
    assert item['caption'] == ''
    for encoder in (0, 1):
        assert_embedding(item, '', encoder)
    close_directory(ds)


def test_bucket_can_reopen_readers_after_serialization(tmp_path):
    new_source(tmp_path)
    ds = build(tmp_path)
    ds.cache_metadata()
    ds.cache_latents(image_latents)
    ds.cache_text_embeddings(text_map, 0)
    bucket = ds.get_size_bucket_datasets()[0]
    expected = [bucket[i] for i in range(len(bucket))]
    restored = pickle.loads(pickle.dumps(ds))
    for index, item in enumerate(expected):
        actual = restored.get_size_bucket_datasets()[0][index]
        torch.testing.assert_close(actual['latents'], item['latents'])
        assert actual['caption'] == item['caption']
        assert_embedding(actual, actual['caption'])
    close_directory(restored)
    close_directory(ds)


def test_snapshot_stays_frozen_until_refresh_and_parent_observes_generation(tmp_path):
    new_source(tmp_path)
    initial = build(tmp_path, require_non_latin_caption=True)
    initial.cache_metadata()
    initial.cache_latents(image_latents)
    close_directory(initial)
    tmp_path.joinpath('0.txt').write_text('red', encoding='utf-8')
    tmp_path.joinpath('1.txt').write_text('制服', encoding='utf-8')
    parent = build(tmp_path, require_non_latin_caption=True)
    parent.cache_metadata(trust_cache=True)
    parent.cache_latents(None, trust_cache=True)
    assert parent.get_size_bucket_datasets()[0][0]['latents'].item() == 0
    close_directory(parent)
    worker = build(tmp_path, require_non_latin_caption=True)
    worker.cache_metadata(regenerate_cache=True, trust_cache=True)
    worker.cache_latents(image_latents, regenerate_cache=True, trust_cache=True)
    new_path = worker.cache_dir
    assert worker.get_size_bucket_datasets()[0][0]['latents'].item() == 1
    close_directory(worker)
    # Main-rank objects were created before the worker published the active generation.
    parent.size_bucket_datasets = []
    parent.cache_metadata(trust_cache=True)
    parent.cache_latents(None, trust_cache=True)
    assert parent.cache_dir == new_path
    assert parent.get_size_bucket_datasets()[0][0]['latents'].item() == 1
    close_directory(parent)


def test_new_resolution_cannot_reference_old_latents(tmp_path):
    new_source(tmp_path)
    first = build(tmp_path)
    first.cache_metadata()
    first.cache_latents(image_latents)
    close_directory(first)
    changed = build(tmp_path, size_buckets=[[96, 96, 1]])
    changed.cache_metadata(trust_cache=True)
    with pytest.raises(RuntimeError, match='need an encoder'):
        changed.cache_latents(None, trust_cache=True)
    changed.cache_latents(image_latents, trust_cache=True)
    assert tuple(changed.get_size_bucket_datasets()[0].size_bucket) == (96, 96, 1)
    close_directory(changed)


def test_changed_encoder_identity_requires_new_embeddings(tmp_path):
    new_source(tmp_path)
    first = build(tmp_path)
    first.cache_metadata()
    first.cache_latents(image_latents)
    first.cache_text_embeddings(text_map, 0, identity='encoder-A')
    close_directory(first)
    changed = build(tmp_path)
    changed.cache_metadata(trust_cache=True)
    changed.cache_latents(None, trust_cache=True)
    with pytest.raises(RuntimeError, match='need an encoder'):
        changed.cache_text_embeddings(None, 0, identity='encoder-B')
    changed.cache_text_embeddings(text_map, 0, identity='encoder-B')
    assert_embedding(changed.get_size_bucket_datasets()[0][0],
                     changed.get_size_bucket_datasets()[0][0]['caption'])
    close_directory(changed)


def test_changed_encoder_key_requires_new_embeddings_with_same_weights(tmp_path):
    new_source(tmp_path)
    first = build(tmp_path)
    first.cache_metadata()
    first.cache_latents(image_latents)
    first.cache_text_embeddings(text_map, 0, identity='same-weights', text_encoder_key='template-A')
    close_directory(first)
    changed = build(tmp_path)
    changed.cache_metadata(trust_cache=True)
    changed.cache_latents(None, trust_cache=True)
    with pytest.raises(RuntimeError, match='need an encoder'):
        changed.cache_text_embeddings(None, 0, identity='same-weights', text_encoder_key='template-B')
    changed.cache_text_embeddings(text_map, 0, identity='same-weights', text_encoder_key='template-B')
    close_directory(changed)


@pytest.mark.parametrize('cached', [False, True])
def test_runtime_augmentation_and_line_removal_use_snapshot_captions(tmp_path, cached):
    new_source(tmp_path)
    ds = build(tmp_path, cached, require_non_latin_caption=True, enable_remove_non_latin=True,
               cache_shuffle_num=2, caption_sampling='random_per_epoch', caption_prefix='anime, ')
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(image_latents, trust_cache=True)
    if cached:
        ds.cache_text_embeddings(text_map, 0)
    bucket = ds.get_size_bucket_datasets()[0]
    for _ in range(5):
        item = bucket[0]
        assert item['caption'] == 'anime, red'
        if cached:
            assert_embedding(item, item['caption'])
    close_directory(ds)


def test_incomplete_donor_warns_and_regenerates_without_overwriting_it(tmp_path, caplog):
    metadata = datasets.Dataset.from_dict({
        'image_spec': [[None, str(tmp_path / f'{i}.png')] for i in range(3)],
        'caption': [['one']] * 3, 'size_bucket': [[64, 64, 1]] * 3})
    donor = current._map_and_cache(metadata.select(range(2)), image_latents, tmp_path / 'old')
    donor.con.close()
    shard = donor.path / 'shard_0.bin'
    original = shard.read_bytes()
    profile = tensor_profile(metadata, image_latents, tmp_path / 'profile', ['image_spec', 'size_bucket'],
                             None, [(metadata, donor.path)], current._map_and_cache)
    assert '2 entries for 3 metadata rows' in caplog.text
    assert 'regenerated' in caplog.text
    assert [profile[i]['latents'].item() for i in range(3)] == [0, 1, 2]
    assert shard.read_bytes() == original
    profile.close()


@pytest.mark.parametrize('donor_change', ['legacy_rebuild', 'donor_deleted'])
def test_profile_rebinds_after_its_donor_changes(tmp_path, donor_change):
    """A reuse_metadata_cache = false run may clear the legacy cache a profile references."""
    import shutil

    def run(**settings):
        ds = build(tmp_path, **settings)
        ds.cache_metadata()
        ds.cache_latents(image_latents)
        ds.cache_text_embeddings(text_map, 0)
        bucket = ds.get_size_bucket_datasets()[0]
        items = [bucket[i] for i in range(len(bucket))]
        for item in items:
            assert item['latents'].item() == int(Path(item['image_spec'][1]).stem)
            assert_embedding(item, item['caption'])
        close_directory(ds)
        return sorted(item['latents'].item() for item in items)

    new_source(tmp_path)
    assert run(reuse_metadata_cache=False) == [0, 1]
    assert run() == [0, 1]
    if donor_change == 'legacy_rebuild':
        Image.new('RGB', (64, 64)).save(tmp_path / '2.png')
        tmp_path.joinpath('2.txt').write_text('green', encoding='utf-8')
        assert run(reuse_metadata_cache=False) == [0, 1, 2]
    else:
        shutil.rmtree(tmp_path / 'cache' / 'migration_test' / 'cache_64x64x1')
    # The snapshot stays frozen, so the new image is not selected, but the profile must not
    # fail on references into a cache that another run was entitled to rebuild.
    assert run() == [0, 1]


def test_legacy_trusted_warm_load_keeps_latents(tmp_path):
    """Grouping keys read back from JSON are lists; the bucket order must not depend on that."""
    encoded = []

    def counting_latents(batch, rank):
        encoded.extend(batch['image_spec'])
        return image_latents(batch, rank)

    new_source(tmp_path)
    for trust in (False, True, True, False, True):
        encoded.clear()
        ds = build(tmp_path, reuse_metadata_cache=False)
        ds.cache_metadata(trust_cache=trust)
        ds.cache_latents(counting_latents, trust_cache=trust)
        bucket = ds.get_size_bucket_datasets()[0]
        assert all(bucket[i]['latents'].item() == int(Path(bucket[i]['image_spec'][1]).stem)
                   for i in range(len(bucket)))
        close_directory(ds)
        if encoded and trust:
            pytest.fail(f'trusted rerun re-encoded {len(encoded)} images')


def test_training_reads_use_in_memory_indexes(tmp_path, monkeypatch):
    """Per-sample reads must not query SQLite: that is slow on network storage."""
    import utils.cache_profiles as profiles
    from utils.cache import Cache

    metadata = datasets.Dataset.from_dict({
        'image_spec': [[None, str(tmp_path / f'{i}.png')] for i in range(7)],
        'caption': [['c']] * 7, 'size_bucket': [[64, 64, 1]] * 7})
    donor = current._map_and_cache(metadata, image_latents, tmp_path / 'donor')
    donor.con.close()
    # Several shards, so the shard/offset arrays are exercised, not just shard 0.
    small = Cache(tmp_path / 'small', 'fp', shard_size_gb=1e-9)
    for i in range(7):
        small.add({'latents': torch.tensor([[i]])})
    small.finalize_current_shard()
    small.con.close()
    reader = profiles.ReadOnlyCache(tmp_path / 'small')
    expected = [reader[i]['latents'].item() for i in range(7)]
    reader.load_index()
    assert reader.con is None
    monkeypatch.setattr(profiles.sqlite3, 'connect', lambda *a, **k: pytest.fail('SQLite opened on read'))
    assert [reader[i]['latents'].item() for i in range(7)] == expected == list(range(7))
    reader.close()
    monkeypatch.undo()

    shuffled = metadata.shuffle(seed=3)
    profile = tensor_profile(shuffled, None, tmp_path / 'profile', ['image_spec', 'size_bucket'], None,
                             [(metadata, donor.path)], current._map_and_cache)
    flat = shuffled.map(lambda b: {'caption': [c for cs in b['caption'] for c in cs]}, batched=True,
                        keep_in_memory=True)
    text = profiles.IndexedTextEmbeddingDataset(profile, flat, tmp_path / 'index.sqlite')
    assert text._hashes is not None
    monkeypatch.setattr(profiles.sqlite3, 'connect', lambda *a, **k: pytest.fail('SQLite opened on read'))
    for i, row in enumerate(shuffled):
        stem = int(Path(row['image_spec'][1]).stem)
        assert profile[i]['latents'].item() == stem
        assert text.get_text_embeddings(tuple(row['image_spec']), 0)['latents'].item() == stem
    with pytest.raises(KeyError):
        text.get_text_embeddings((None, 'missing.png'), 0)
    with pytest.raises(KeyError):
        text.get_text_embeddings(tuple(shuffled[0]['image_spec']), 1)
    monkeypatch.undo()

    # A hash collision between two images falls back to the exact SQLite lookup.
    text.close()
    monkeypatch.setattr(profiles, '_image_hash', lambda image: 1)
    path = tmp_path / 'index.sqlite.lookup.npz'
    path.unlink()
    fallback = profiles.IndexedTextEmbeddingDataset(profile, flat, tmp_path / 'index.sqlite')
    assert fallback._hashes is None
    assert all(fallback.get_text_embeddings(tuple(row['image_spec']), 0)['latents'].item()
               == int(Path(row['image_spec'][1]).stem) for row in shuffled)
    fallback.close()
    profile.close()


class CountingConnection:
    def __init__(self, connection, counter):
        self.connection, self.counter = connection, counter

    def execute(self, *args):
        self.counter.append(args[0])
        return self.connection.execute(*args)

    def __getattr__(self, name):
        return getattr(self.connection, name)


@pytest.mark.parametrize('rows', [5, 40])
def test_profile_startup_queries_do_not_scale_with_rows(tmp_path, monkeypatch, rows):
    """Building a profile must not issue SQLite statements per row: each is a network round trip."""
    import sqlite3
    import utils.cache_profiles as profiles

    metadata = datasets.Dataset.from_dict({
        'image_spec': [[None, str(tmp_path / f'{i}.png')] for i in range(rows)],
        'caption': [[f'c{i}', f'd{i}'] for i in range(rows)], 'size_bucket': [[64, 64, 1]] * rows})
    donor = current._map_and_cache(metadata, image_latents, tmp_path / 'donor')
    donor.con.close()
    statements = []
    connect = sqlite3.connect
    monkeypatch.setattr(profiles.sqlite3, 'connect',
                        lambda *a, **k: CountingConnection(connect(*a, **k), statements))
    # A different selection and order: a new profile over the same donor.
    target = metadata.select(range(rows - 1, 0, -1))
    profile = tensor_profile(target, None, tmp_path / 'profile', ['image_spec', 'size_bucket'], None,
                             [(metadata, donor.path)], current._map_and_cache)
    flat = datasets.Dataset.from_dict({
        'image_spec': [row['image_spec'] for row in target for _ in row['caption']],
        'caption': [caption for row in target for caption in row['caption']]})
    # Positions stand in for embeddings: the second caption of each image follows its first.
    text = profiles.IndexedTextEmbeddingDataset(list(range(len(flat))), flat, tmp_path / 'index.sqlite')
    assert len(statements) <= 6, statements
    assert not (tmp_path / 'index.sqlite').exists()
    assert [profile[i]['latents'].item() for i in range(len(target))] == list(range(rows - 1, 0, -1))
    assert [text.get_text_embeddings(tuple(target[i]['image_spec']), 1) for i in range(len(target))] == [
        2 * i + 1 for i in range(len(target))]
    profile.close()


def test_digest_lookups_keep_the_legacy_duplicate_rules():
    from utils.cache_profiles import DigestIndex, RowsByImage

    # First value wins, as INSERT OR IGNORE kept it.
    index = DigestIndex(np.array([b'b', b'a', b'b'], dtype='S16'), [0, 1, 2])
    assert index.lookup(np.array([b'b', b'a', b'c'], dtype='S16')).tolist() == [0, 1, -1]
    assert DigestIndex(np.empty(0, dtype='S16'), []).lookup(np.array([b'a'], dtype='S16')).tolist() == [-1]
    # Last row wins across and within sources, as INSERT OR REPLACE kept it.
    first = datasets.Dataset.from_dict({'image_spec': [[None, 'x'], [None, 'y']], 'size_bucket': [[1], [2]]})
    second = datasets.Dataset.from_dict({'image_spec': [[None, 'x'], [None, 'x']], 'size_bucket': [[3], [4]]})
    rows = RowsByImage([first, second]).get([[None, 'y'], (None, 'x'), [None, 'z']])
    assert rows == [{'image_spec': [None, 'y'], 'size_bucket': [2]},
                    {'image_spec': [None, 'x'], 'size_bucket': [4]}, None]
    assert RowsByImage([]).get([[None, 'x']]) == [None]


def test_single_worker_mapping_does_not_spawn_transport(tmp_path, monkeypatch):
    metadata = datasets.Dataset.from_dict({'caption': ['one', 'two']})
    monkeypatch.setattr(current.mp, 'Manager', lambda: pytest.fail('single worker spawned manager'))
    cache = current._map_and_cache(metadata, text_map, tmp_path / 'single')
    assert len(cache) == 2
    cache.con.close()


def test_empty_selected_profile_is_safe(tmp_path):
    new_source(tmp_path)
    tmp_path.joinpath('0.txt').write_text('red', encoding='utf-8')
    ds = build(tmp_path, require_non_latin_caption=True)
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(None, trust_cache=True)
    assert ds.get_size_bucket_datasets() == []


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_reuse_metadata_cache_requires_boolean(tmp_path, value):
    with pytest.raises(ValueError, match='reuse_metadata_cache must be a boolean'):
        build(tmp_path, reuse_metadata_cache=value)
