"""The metadata scan: one directory listing without a stat per file, and media headers and
caption files read on several threads. Both must give exactly what the plain scan gave."""

import errno
import multiprocessing
import os
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import utils.dataset as dataset_module  # noqa: E402
from utils.dataset import NON_MEDIA_SUFFIXES, DirectoryDataset  # noqa: E402


def directory(path, model, reuse_metadata_cache=False, framerate=None, **settings):
    config = {'path': str(path), 'num_repeats': 1, 'shuffle_metadata': False,
              'reuse_metadata_cache': reuse_metadata_cache, **settings}
    return DirectoryDataset(config, {'resolutions': [64], 'enable_ar_bucket': True, 'min_ar': 0.5, 'max_ar': 2.0,
                                     'num_ar_buckets': 5, 'frame_buckets': [1]}, model, framerate=framerate,
                            skip_dataset_validation=True)


def image(path, width, height):
    Image.new('RGB', (width, height), (width % 256, height % 256, 7)).save(path, format='PNG')


def old_listing(path, mask_path=None):
    """(image_spec, caption_file, mask_file) as the scan listed them before: Path.glob, a stat
    per entry and an exists() per caption."""
    files = sorted(path.glob('*'))
    masks = {p.stem: p for p in mask_path.glob('*') if p.is_file()} if mask_path is not None else {}
    rows = []
    for file in files:
        if not file.is_file() or file.suffix in NON_MEDIA_SUFFIXES:
            continue
        if file.suffix == '.tar':
            with tarfile.TarFile(file) as tar:
                specs = [(str(file), name) for name in tar.getnames()]
        else:
            specs = [(None, str(file))]
        for spec in specs:
            image_file = Path(spec[1])
            caption = image_file.with_suffix('.txt')
            rows.append((list(spec), str(caption) if os.path.exists(caption) else '',
                         str(masks[image_file.stem]) if image_file.stem in masks else None))
    return rows


def test_listing_matches_the_scan_with_a_stat_per_file(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('a', 'b', 'c', 'd', 'e', '.hidden', 'noext'):
        image(data / (f'{name}.png' if name != 'noext' else name), 64, 48)
    (data / 'a.txt').write_text('a caption')
    (data / '.hidden.txt').write_text('hidden caption')
    (data / 'elsewhere.txt').write_text('linked caption')
    (data / 'b.txt').symlink_to(data / 'elsewhere.txt')    # a symlinked caption
    (data / 'c.txt').symlink_to(data / 'missing.txt')      # a broken one: no caption
    (data / 'd.txt').mkdir()                                # exists() is true for a directory
    (data / 'linked.png').symlink_to(data / 'a.png')        # a symlinked image
    (data / 'gone.png').symlink_to(data / 'missing.png')    # a broken one: skipped
    (data / 'sub').mkdir()
    (data / 'meta.json').write_text('{}')
    with tarfile.open(data / 'pack.tar', 'w') as tar:
        member = tmp_path / 't.png'
        image(member, 32, 64)
        tar.add(member, arcname='inner/t.png')
    masks = tmp_path / 'masks'
    masks.mkdir()
    for name in ('a.png', 'a.jpg', 'e.png'):               # two masks of stem a: the later wins
        image(masks / name, 64, 48)

    def no_map(self):
        return lambda example: {'image_spec': [], 'mask_file': [], 'caption': [], 'ar_bucket': [],
                                'size_bucket': [], 'is_video': []}
    monkeypatch.setattr(DirectoryDataset, '_metadata_map_fn', no_map)
    ds = directory(data, 'listing', mask_path=str(masks))
    ds._get_ungrouped_metadata()
    import datasets
    listed = datasets.load_from_disk(str(ds.cache_dir / 'metadata' / 'metadata_intermediate'))
    rows = [(row['image_spec'], row['caption_file'], row['mask_file']) for row in listed]
    assert rows == old_listing(data, masks)
    assert {Path(spec[1]).name for spec, _, _ in rows} == {
        'a.png', 'b.png', 'c.png', 'd.png', 'e.png', '.hidden.png', 'noext', 'linked.png', 't.png'}


def make_dataset(path, count=40):
    path.mkdir()
    for i in range(count):
        image(path / f'{i:03d}.png', 32 + 8 * (i % 9), 32 + 8 * (i % 5))
        if i % 7 != 3:  # some images have no caption and are skipped
            (path / f'{i:03d}.txt').write_text(f'tag{i}, red, blue\nsecond line {i}' if i % 3 else f'tag{i}')
    with tarfile.open(path / 'pack.tar', 'w') as tar:
        for i in range(24):
            member = path.parent / f'member{i}.png'
            image(member, 32 + 16 * (i % 4), 32 + 8 * (i % 6))
            tar.add(member, arcname=f'm{i:02d}.png')
            member.unlink()


@pytest.mark.parametrize('reuse_metadata_cache', [False, True], ids=['legacy', 'snapshot'])
@pytest.mark.parametrize('num_proc', [1, 2])
def test_threads_give_exactly_the_rows_of_one_thread(tmp_path, monkeypatch, num_proc, reuse_metadata_cache):
    data = tmp_path / 'data'
    make_dataset(data)

    def scan(model, threads, proc):
        monkeypatch.setattr(dataset_module, 'METADATA_READ_THREADS', threads)
        monkeypatch.setattr(dataset_module, 'NUM_PROC', proc)
        ds = directory(data, model, reuse_metadata_cache=reuse_metadata_cache, cache_shuffle_num=2)
        if reuse_metadata_cache:
            return ds._load_source_snapshot().to_list()
        return ds._get_ungrouped_metadata().to_list()

    expected = scan('one', 1, 1)
    # The legacy scan skips media without a caption (the tar members have none); the snapshot
    # keeps them and leaves skipping to each caption profile.
    assert len(expected) == (40 + 24 if reuse_metadata_cache else 40 - 6)
    assert scan('many', 8, num_proc) == expected


def test_reads_run_on_several_threads(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    make_dataset(data, count=16)
    monkeypatch.setattr(dataset_module, 'METADATA_READ_THREADS', 4)
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)
    threads = set()
    open_image = Image.open

    def slow_open(*args, **kwargs):
        threads.add(threading.get_ident())
        time.sleep(0.05)
        return open_image(*args, **kwargs)
    monkeypatch.setattr(dataset_module.Image, 'open', slow_open)
    directory(data, 'threads')._get_ungrouped_metadata()
    assert len(threads) == 4


def _read_after_fork(queue):
    queue.put(dataset_module._read_in_threads(lambda x: x * 2, [1, 2]))


def test_a_forked_map_worker_gets_its_own_threads(monkeypatch):
    """A map worker forked from a process that already read in threads inherits the pool but
    none of its threads: reading there must not wait on them forever."""
    monkeypatch.setattr(dataset_module, 'METADATA_READ_THREADS', 4)

    def slow(x):
        time.sleep(0.05)
        return x + 1
    # Every thread busy at once, then idle: an inherited pool counts them as free and starts no
    # new one for the child's two reads.
    assert dataset_module._read_in_threads(slow, list(range(8))) == list(range(1, 9))
    time.sleep(0.2)
    context = multiprocessing.get_context('fork')
    queue = context.Queue()
    child = context.Process(target=_read_after_fork, args=(queue,))
    child.start()
    child.join(timeout=30)
    if child.is_alive():
        child.kill()
        pytest.fail('the forked process waited on threads it does not have')
    assert child.exitcode == 0 and queue.get(timeout=5) == [2, 4]


class _Entry:
    def __init__(self, error):
        self.error = error

    def is_file(self):
        raise OSError(self.error, os.strerror(self.error))


def test_a_volume_error_on_an_entry_is_raised_not_read_as_no_file():
    """A stale handle or an I/O error must not silently drop a file from the metadata, which
    --trust_cache would then keep; a symlink loop or a vanished target is "not a file", as for
    Path.is_file()."""
    for error in (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP):
        assert dataset_module._is_file(_Entry(error)) is False
    for error in (errno.EIO, errno.ESTALE, errno.EACCES):
        with pytest.raises(OSError):
            dataset_module._is_file(_Entry(error))


def test_a_caption_found_only_by_a_filesystem_that_ignores_case(tmp_path, monkeypatch):
    """CIFS or exFAT on Linux lists a.txt yet opens it as A.txt for A.png, which os.path.exists
    found for the old scan. Simulated by listing A.txt under a lower-case name."""
    data = tmp_path / 'data'
    data.mkdir()
    image(data / 'A.png', 64, 48)
    (data / 'A.txt').write_text('a caption')
    scandir = os.scandir

    class Lowered:
        def __init__(self, entry):
            self.entry = entry
            self.name = entry.name.lower() if entry.name.endswith('.txt') else entry.name

        def __getattr__(self, name):
            return getattr(self.entry, name)

    class Listing:
        def __init__(self, path):
            self.entries = scandir(path)

        def __enter__(self):
            return (Lowered(entry) for entry in self.entries.__enter__())

        def __exit__(self, *args):
            return self.entries.__exit__(*args)
    monkeypatch.setattr(dataset_module.os, 'scandir', Listing)
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)
    rows = directory(data, 'case')._get_ungrouped_metadata().to_list()
    assert [row['caption'] for row in rows] == [['a caption']]


def test_one_tarfile_per_tar_per_process(tmp_path, monkeypatch):
    """The threads share one TarFile per tar, not one each: every TarFile keeps the archive's
    member table and a descriptor open."""
    data = tmp_path / 'data'
    data.mkdir()
    for pack in ('p', 'q'):
        with tarfile.open(data / f'{pack}.tar', 'w') as tar:
            for i in range(24):
                member = tmp_path / 'member.png'
                image(member, 32 + 16 * (i % 4), 48)
                tar.add(member, arcname=f'{pack}{i:02d}.png')
    monkeypatch.setattr(dataset_module, 'METADATA_READ_THREADS', 8)
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)
    opened = []

    class Counting(tarfile.TarFile):
        def __init__(self, *args, **kwargs):
            opened.append(args[0] if args else kwargs.get('name'))
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(dataset_module.tarfile, 'TarFile', Counting)
    open_image = Image.open

    def slow_open(*args, **kwargs):
        time.sleep(0.01)
        return open_image(*args, **kwargs)
    monkeypatch.setattr(dataset_module.Image, 'open', slow_open)
    rows = directory(data, 'tars', skip_empty_caption=False)._get_ungrouped_metadata().to_list()
    assert len(rows) == 48
    # The listing opens each tar once, the reads once more.
    assert sorted(map(str, opened)) == sorted(2 * [str(data / 'p.tar'), str(data / 'q.tar')])


def test_one_mkv_decode_at_a_time(tmp_path, monkeypatch):
    """An .mkv is decoded whole to count its frames: threads must not hold several in memory."""
    data = tmp_path / 'data'
    data.mkdir()
    for i in range(8):
        (data / f'{i}.mkv').write_bytes(b'not decoded')
        (data / f'{i}.txt').write_text(f'video {i}')
    monkeypatch.setattr(dataset_module, 'METADATA_READ_THREADS', 8)
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)
    running = []
    most = []

    class Components:
        def __init__(self):
            import torch
            self.images = torch.zeros(1, 48, 64, 3)

    class Video:
        def __init__(self, file):
            pass

        def get_components(self):
            running.append(1)
            most.append(len(running))
            time.sleep(0.05)
            running.pop()
            return Components()

        def get_frame_rate(self):
            return 24
    monkeypatch.setattr(dataset_module.InputImpl, 'VideoFromFile', Video)
    rows = directory(data, 'mkv', framerate=24)._get_ungrouped_metadata().to_list()
    assert len(rows) == 8 and max(most) == 1
