"""--cache_status: how many files have their latents and text embeddings cached, read without
encoding, clearing or writing anything; and the same count shown while caching."""

import multiprocessing
import queue
import threading
import time
import types

import datasets
import numpy as np
import pytest
import torch
from deepspeed import comm as dist

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
    latents = [text.split(',')[0] for desc, text in shown if desc == 'latents']
    embeddings = [text.split(',')[0] for desc, text in shown if desc == 'text_embeddings_1']
    assert latents[0] == '0/3 files' and latents[-1] == '3/3 files'
    # Six captions, two to a file: the count moves a file at a time.
    assert embeddings[0] == '0/3 files' and embeddings[-1] == '3/3 files'
    assert '1/3 files' in embeddings and '2/3 files' in embeddings


def timing_of(shown, description):
    """(saving %, waiting %) of the last progress update of one cache."""
    text = [t for d, t in shown if d == description][-1]
    saving, waiting = (int(part.split()[1].rstrip('%')) for part in text.split(', ')[-2:])
    return saving, waiting


def test_caching_shows_where_the_time_goes(tmp_path, monkeypatch):
    from utils.cache import Cache

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
    add = Cache.add

    def slow_add(self, item):
        time.sleep(0.1)
        return add(self, item)

    monkeypatch.setattr(Cache, 'add', slow_add)
    # A writer that queues one row at most, so that its slowness holds the caching process up.
    writer = dataset_util._BackgroundWriter
    monkeypatch.setattr(dataset_util, '_BackgroundWriter', lambda cache, limit: writer(cache, limit_bytes=1))
    images = make_images(tmp_path / 'images', CAPTIONS)
    config = dataset_config(images, reuse_metadata_cache=False)
    model = CountingModel()
    run_caching(model, dataset_util.Dataset(config, model), config)
    # Writing a row takes 0.1 s here and encoding one almost nothing: the writer is the limit.
    for description in ('latents', 'text_embeddings_1', 'text_embeddings_2'):
        saving, waiting = timing_of(shown, description)
        assert saving >= 80 and saving + waiting in (99, 100, 101), (description, saving, waiting)
    assert [t for d, t in shown if d == 'latents'][-1].startswith('6/6 files, saving ')


class SlowCache:
    def __init__(self, delay=0.0, fail_at=None):
        self.rows = []
        self.delay = delay
        self.fail_at = fail_at

    def add(self, item):
        time.sleep(self.delay)
        if self.fail_at is not None and len(self.rows) == self.fail_at:
            raise OSError('disk full')
        self.rows.append(item)


def test_the_writer_writes_every_row_in_order():
    cache = SlowCache()
    writer = dataset_util._BackgroundWriter(cache)
    for i in range(50):
        writer.put({'x': torch.tensor([i])})
    writer.close()
    assert [row['x'].item() for row in cache.rows] == list(range(50))


def test_the_writer_stops_the_caller_at_its_limit():
    cache = SlowCache(delay=0.05)
    writer = dataset_util._BackgroundWriter(cache, limit_bytes=3 * 8)
    row = {'x': torch.zeros(1, dtype=torch.int64)}
    started = time.monotonic()
    for _ in range(8):
        writer.put({k: v.clone() for k, v in row.items()})
        assert writer.queued <= 3 * 8 + 8
    # With room for three rows the caller must have waited for the disk for most of the rest.
    assert time.monotonic() - started >= 0.05 * 3
    writer.close()
    assert len(cache.rows) == 8


def test_a_failed_write_reaches_the_caller():
    cache = SlowCache(fail_at=2)
    writer = dataset_util._BackgroundWriter(cache)
    with pytest.raises(RuntimeError, match='Writing the cache failed: OSError: disk full'):
        for i in range(100):
            writer.put({'x': torch.tensor([i])})
            time.sleep(0.01)
        writer.close()
    assert [row['x'].item() for row in cache.rows] == [0, 1]
    writer.abort()


def test_aborting_drops_the_rows_not_yet_written_and_keeps_a_prefix():
    cache = SlowCache(delay=0.05)
    writer = dataset_util._BackgroundWriter(cache)
    for i in range(20):
        writer.put({'x': torch.tensor([i])})
    while not cache.rows:
        time.sleep(0.01)
    writer.abort()
    written = [row['x'].item() for row in cache.rows]
    assert 0 < len(written) < 20 and written == list(range(len(written)))


class StreamingPool:
    """multiprocessing.Pool.imap takes its tasks from the iterable on a thread of its own."""

    feeders = []

    def __init__(self, count, initializer, args):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(count, initializer=initializer, initargs=args)

    def imap(self, function, iterable):
        futures = queue.Queue()

        def feed():
            for item in iterable:
                futures.put(self.executor.submit(function, item))
            futures.put(None)

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        self.feeders.append(feeder)

        def results():
            while (future := futures.get()) is not None:
                yield future.result()
        return results()

    def close(self):
        self.executor.shutdown(wait=True)

    def join(self):
        pass


def test_workers_do_not_run_far_ahead_of_the_cache(tmp_path, monkeypatch):
    """The rows the workers return wait in memory until they are written: how many batches are out
    at once is bounded, whatever the disk does."""
    monkeypatch.setattr(dataset_util, 'NUM_PROC', 2)
    monkeypatch.setattr(dataset_util.mp, 'Pool', StreamingPool)
    monkeypatch.setattr(dataset_util.mp, 'Manager', lambda: types.SimpleNamespace(
        Queue=queue.Queue, shutdown=lambda: None))
    started = consumed = highest = 0
    lock = threading.Lock()

    def map_fn(batch, rank):
        nonlocal started
        with lock:
            started += 1
        return {'x': [torch.zeros(1) for _ in batch['image_spec']]}

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def set_postfix_str(self, text):
            pass

        def __iter__(self):
            nonlocal consumed, highest
            for item in self.iterable:
                time.sleep(0.02)
                with lock:
                    consumed += 1
                    highest = max(highest, started - consumed)
                yield item

    monkeypatch.setattr(dataset_util, 'tqdm', Progress)
    rows = datasets.Dataset.from_dict({'image_spec': [[None, f'{i}.png'] for i in range(60)]})
    cache = dataset_util._map_and_cache(rows, map_fn, tmp_path / 'cache', caching_batch_size=1)
    assert len(cache) == 60 and consumed == 60
    # The window is 2 * NUM_PROC batches; a batch counts as started a moment before it is returned.
    assert highest <= 2 * 2 + 1, highest


def test_a_gpu_rank_reports_where_its_time_goes(capsys, monkeypatch):
    monkeypatch.setattr(dist, 'get_rank', lambda: 3)
    timing = dataset_util._RankTiming()
    timing.add('waiting for work', 1.0)
    timing.add('encoding', 6.0)
    timing.add('moving results', 3.0)
    timing.tasks = 5
    timing.since = time.monotonic() - 10.0
    timing.report()
    assert capsys.readouterr().out == ''
    timing.INTERVAL = 5
    timing.report()
    line = capsys.readouterr().out.strip()
    assert line.startswith('[CACHE] rank 3, last 10 s, 5 tasks: waiting for work 10%, encoding 60%, moving results 30%')
    assert 'remembering' not in line and timing.tasks == 0


def test_handling_a_task_adds_its_time_to_the_rank(tmp_path):
    model = CountingModel()
    manager = dataset_util.DatasetManager(model)
    dataset_util._RANK_TIMING.reset()
    pipe_parent, pipe_child = multiprocessing.Pipe(duplex=False)
    manager._handle_task((1, ['a red bus', 'two dogs'], [False, False], None, pipe_child))
    pipe_parent.recv()
    seconds = dataset_util._RANK_TIMING.seconds
    assert seconds['encoding'] > 0 and seconds['moving results'] > 0 and seconds['remembering rows'] > 0


def test_a_failing_worker_does_not_leave_the_pool_hanging(tmp_path, monkeypatch):
    """The pool's shutdown joins the thread that feeds its workers; that thread must not stay
    blocked on the window of batches in flight once the caching loop has failed."""
    monkeypatch.setattr(dataset_util, 'NUM_PROC', 2)
    monkeypatch.setattr(dataset_util.mp, 'Pool', StreamingPool)
    monkeypatch.setattr(dataset_util.mp, 'Manager', lambda: types.SimpleNamespace(
        Queue=queue.Queue, shutdown=lambda: None))
    StreamingPool.feeders = []

    def map_fn(batch, rank):
        if batch['image_spec'][0][1] == '3.png':
            raise ValueError('worker failed')
        return {'x': [torch.zeros(1) for _ in batch['image_spec']]}

    rows = datasets.Dataset.from_dict({'image_spec': [[None, f'{i}.png'] for i in range(60)]})
    with pytest.raises(ValueError, match='worker failed'):
        dataset_util._map_and_cache(rows, map_fn, tmp_path / 'cache', caching_batch_size=1)
    (feeder,) = StreamingPool.feeders
    feeder.join(5)
    assert not feeder.is_alive()


def test_rows_of_arrays_and_bytes_count_toward_the_writers_limit():
    row = {'a': np.zeros(10, dtype=np.float32), 'b': b'12345', 'c': [torch.zeros(2)]}
    assert dataset_util._item_bytes(row) == 40 + 5 + 8


@pytest.mark.parametrize('config, expected', [
    ({}, (512, 60)),
    ({'caching_write_queue_mb': 2048, 'caching_report_seconds': 0}, (2048, 0)),
    ({'caching_write_queue_mb': 0.5, 'caching_report_seconds': 2.5}, (0.5, 2.5)),
])
def test_caching_progress_settings(config, expected):
    assert dataset_util.caching_progress_settings(config) == expected


@pytest.mark.parametrize('key, value', [
    ('caching_write_queue_mb', 0), ('caching_write_queue_mb', -5), ('caching_write_queue_mb', 'big'),
    ('caching_write_queue_mb', True), ('caching_write_queue_mb', float('nan')),
    ('caching_report_seconds', -1), ('caching_report_seconds', '60'), ('caching_report_seconds', False),
])
def test_caching_progress_settings_are_checked(key, value):
    with pytest.raises(ValueError, match=key):
        dataset_util.caching_progress_settings({key: value})


def test_the_settings_reach_the_caching_process_and_the_ranks(monkeypatch):
    started = []

    class Process:
        def __init__(self, **kwargs):
            started.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(dataset_util.mp, 'Process', Process, raising=False)
    monkeypatch.setattr(dataset_util._RANK_TIMING, 'INTERVAL', 60, raising=False)
    manager = dataset_util.DatasetManager(CountingModel(), write_queue_mb=7, report_seconds=0)
    assert dataset_util._RANK_TIMING.INTERVAL == 0
    manager._start_caching_process(queue.Queue(), stage='latents')
    assert started[0]['kwargs'] == {'stage': 'latents', 'write_queue_mb': 7}
    monkeypatch.setattr(dataset_util, '_write_queue_bytes', 0)
    monkeypatch.setattr(dataset_util, '_cache_datasets', lambda *args, **kwargs: None)
    dataset_util._cache_fn([], queue.Queue(), None, 0, False, False, 1, write_queue_mb=3)
    assert dataset_util._write_queue_bytes == 3 << 20


def test_the_writer_gets_the_configured_limit(tmp_path, monkeypatch):
    limits = []
    writer = dataset_util._BackgroundWriter
    monkeypatch.setattr(dataset_util, '_BackgroundWriter', lambda cache, limit: limits.append(limit) or writer(cache, limit))
    monkeypatch.setattr(dataset_util, '_write_queue_bytes', 12345)
    rows = datasets.Dataset.from_dict({'image_spec': [[None, f'{i}.png'] for i in range(3)]})
    dataset_util._map_and_cache(rows, lambda batch, rank: {'x': [torch.zeros(1) for _ in batch['image_spec']]},
                                tmp_path / 'cache', caching_batch_size=1)
    assert limits == [12345]


def test_a_rank_with_the_report_off_prints_nothing(capsys, monkeypatch):
    monkeypatch.setattr(dist, 'get_rank', lambda: 0)
    timing = dataset_util._RankTiming()
    timing.INTERVAL = 0
    timing.add('encoding', 1.0)
    timing.tasks = 4
    timing.since = time.monotonic() - 1000
    timing.report()
    timing.report(force=True)
    assert capsys.readouterr().out == ''
