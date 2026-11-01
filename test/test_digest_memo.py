"""utils/dataset.py: column reads and digests that every rank computes at launch.

The cache fingerprints must come out exactly as before (a changed fingerprint re-encodes a whole
cache), and a remembered digest must never outlive the files it describes.
"""

import json
import os

import datasets
import pytest
from datasets.fingerprint import Hasher

from utils import dataset as dataset_module


def rows(n=50):
    return {
        'image_spec': [[None if i % 3 else 'archive.tar', f'dir/{i}.png'] for i in range(n)],
        'caption': [[f'caption {i}', '{"summary": {"caption": "x"}}' if i % 5 == 0 else f'other {i}'] for i in range(n)],
        'size_bucket': [[64.0, 48.0 + (i % 4), 1] for i in range(n)],
        'mask_file': [None] * n,
        'ar_bucket': [[0.75 + 0.01 * (i % 7), 1.0] for i in range(n)],
        'is_video': [i % 2 == 0 for i in range(n)],
    }


@pytest.fixture
def on_disk(tmp_path):
    path = tmp_path / 'grouped_metadata'
    datasets.Dataset.from_dict(rows()).save_to_disk(str(path))
    return datasets.load_from_disk(str(path))


def old_fingerprint(dataset, args, columns):
    """_cache_fingerprint as it was, row by row: the value every existing cache records."""
    hasher = Hasher()
    for column in sorted(columns):
        hasher.update(column)
        hasher.update(list(dataset[column]))
    return Hasher.hash(list(args) + [hasher.hexdigest()])


def variants(dataset):
    return [dataset, dataset.shuffle(seed=7), dataset.select(range(3, 40, 2)),
            datasets.Dataset.from_dict(rows())]


def test_column_values_are_the_rows_values(on_disk):
    for dataset in variants(on_disk):
        for column in dataset.column_names:
            assert dataset_module._column_values(dataset, column) == list(dataset[column])


def test_fingerprints_are_unchanged(on_disk):
    columns = ['image_spec', 'size_bucket', 'mask_file', 'ar_bucket', 'is_video']
    for dataset in variants(on_disk):
        for _ in range(2):  # computed, then remembered
            assert dataset_module._cache_fingerprint(dataset, ['vae'], columns) == old_fingerprint(dataset, ['vae'], columns)
        assert dataset_module._content_digest(dataset, 'caption') == Hasher.hash(list(dataset['caption']))
        assert dataset_module._content_digest(dataset, 'caption', 10) == Hasher.hash(list(dataset['caption'])[:10])


def counting(value='digest'):
    calls = []

    def compute():
        calls.append(1)
        return value
    return compute, calls


def test_a_digest_is_remembered_with_the_files(on_disk):
    compute, calls = counting()
    assert dataset_module._memoized_digest(on_disk, ['what'], compute) == 'digest'
    assert dataset_module._memoized_digest(on_disk, ['what'], compute) == 'digest'
    assert len(calls) == 1
    memo = dataset_module._memo_path(on_disk)
    assert memo.parent == dataset_module.Path(on_disk.cache_files[0]['filename']).parent
    # Another question, or another row selection of the same files, is another entry.
    dataset_module._memoized_digest(on_disk, ['other'], compute)
    dataset_module._memoized_digest(on_disk.shuffle(seed=1), ['what'], compute)
    assert len(calls) == 3
    assert len(json.loads(memo.read_text())) == 3


def test_rewritten_files_are_not_served_an_old_digest(tmp_path):
    path = tmp_path / 'grouped_metadata'
    datasets.Dataset.from_dict(rows()).save_to_disk(str(path))
    first = datasets.load_from_disk(str(path))
    columns = ['image_spec', 'size_bucket']
    before = dataset_module._cache_fingerprint(first, [], columns)
    changed = rows()
    changed['image_spec'][0] = [None, 'dir/replaced.png']
    datasets.Dataset.from_dict(changed).save_to_disk(str(tmp_path / 'next'))
    # The same path, written again, as a regrouping does.
    for name in os.listdir(tmp_path / 'next'):
        os.replace(tmp_path / 'next' / name, path / name)
    second = datasets.load_from_disk(str(path))
    after = dataset_module._cache_fingerprint(second, [], columns)
    assert after != before
    assert after == old_fingerprint(second, [], columns)


def test_a_dataset_in_memory_is_never_remembered():
    compute, calls = counting()
    dataset = datasets.Dataset.from_dict(rows())
    for _ in range(2):
        dataset_module._memoized_digest(dataset, ['what'], compute)
    assert len(calls) == 2


def test_a_derived_dataset_is_remembered_under_its_origin(on_disk):
    compute, calls = counting()
    derived = on_disk.map(lambda batch: {'caption': [c[0] for c in batch['caption']]}, batched=True,
                          keep_in_memory=True, remove_columns=on_disk.column_names)
    derived._digest_origin = (on_disk, 'first_caption')
    for _ in range(2):
        dataset_module._memoized_digest(derived, ['what'], compute)
    assert len(calls) == 1
    assert dataset_module._memo_path(derived) == dataset_module._memo_path(on_disk)


def test_a_volume_that_cannot_be_written_still_answers(on_disk, monkeypatch):
    compute, calls = counting()

    def refuse(*args, **kwargs):
        raise OSError('read-only file system')
    monkeypatch.setattr(dataset_module.Path, 'write_text', refuse)
    for _ in range(2):
        assert dataset_module._memoized_digest(on_disk, ['what'], compute) == 'digest'
    assert len(calls) == 2
    assert not list(dataset_module._memo_path(on_disk).parent.glob('*.tmp'))


def test_a_corrupt_memo_is_ignored(on_disk):
    compute, calls = counting()
    dataset_module._memo_path(on_disk).write_text('not json')
    assert dataset_module._memoized_digest(on_disk, ['what'], compute) == 'digest'
    assert dataset_module._memoized_digest(on_disk, ['what'], compute) == 'digest'
    assert len(calls) == 1


def test_the_memo_keeps_its_newest_entries(on_disk, monkeypatch):
    monkeypatch.setattr(dataset_module, 'DIGEST_MEMO_ENTRIES', 3)
    for i in range(5):
        dataset_module._memoized_digest(on_disk, ['what', i], lambda i=i: str(i))
    assert sorted(json.loads(dataset_module._memo_path(on_disk).read_text()).values()) == ['2', '3', '4']


def test_layout_caption_counts(on_disk):
    assert dataset_module._layout_caption_counts(on_disk) == (50, 10)
    assert dataset_module._layout_caption_counts(on_disk.select(range(5))) == (5, 1)
    plain = datasets.Dataset.from_dict({'caption': [['a', 'b']] * 4})
    assert dataset_module._layout_caption_counts(plain) == (4, 0)


def test_two_ranges_of_one_file_are_told_apart(on_disk):
    # select() of a contiguous range slices the table without an indices mapping.
    first, second = on_disk.select(range(0, 10)), on_disk.select(range(5, 15))
    assert first._indices is None and second._indices is None
    for _ in range(2):
        assert dataset_module._content_digest(first, 'caption') == Hasher.hash(list(first['caption']))
        assert dataset_module._content_digest(second, 'caption') == Hasher.hash(list(second['caption']))


@pytest.mark.parametrize('kind', ['numpy', 'torch'])
def test_a_formatted_dataset_keeps_its_old_fingerprint(on_disk, kind):
    formatted = on_disk.with_format(kind)
    columns = ['is_video', 'ar_bucket']
    assert dataset_module._cache_fingerprint(formatted, [], columns) == old_fingerprint(formatted, [], columns)


def test_a_column_changed_in_memory_is_not_served_the_old_digest(on_disk):
    # remove_columns / add_column keep the dataset's files; its fingerprint follows the change.
    before = dataset_module._content_digest(on_disk, 'is_video')
    changed = on_disk.remove_columns('is_video').add_column('is_video', [True] * len(on_disk))
    assert changed.cache_files == on_disk.cache_files
    assert dataset_module._content_digest(changed, 'is_video') == Hasher.hash(list(changed['is_video'])) != before


def test_a_file_rewritten_to_the_same_size_misses(tmp_path):
    # Same bytes count and the same state.json (so the same fingerprint): only the modification
    # time tells the rewrite apart.
    path, other = tmp_path / 'grouped_metadata', tmp_path / 'other'
    rows_a = {'image_spec': [[None, f'dir/{i}.png'] for i in range(20)]}
    rows_b = {'image_spec': [[None, f'dir/{i}.jpg'] for i in range(20)]}
    datasets.Dataset.from_dict(rows_a).save_to_disk(str(path))
    datasets.Dataset.from_dict(rows_b).save_to_disk(str(other))
    first = datasets.load_from_disk(str(path))
    columns = ['image_spec']
    before = dataset_module._cache_fingerprint(first, [], columns)
    data = next(path.glob('data-*.arrow'))
    replacement = next(other.glob('data-*.arrow'))
    assert data.stat().st_size == replacement.stat().st_size
    stat = data.stat()
    os.replace(replacement, data)
    os.utime(data, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = datasets.load_from_disk(str(path))
    assert second._fingerprint == first._fingerprint
    after = dataset_module._cache_fingerprint(second, [], columns)
    assert after == old_fingerprint(second, [], columns) != before
