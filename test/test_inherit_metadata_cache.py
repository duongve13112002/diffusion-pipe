"""tools/inherit_metadata_cache.py: a model starting from another model's source snapshot ends up
with exactly the metadata its own scan would have produced, and nothing is inherited when it
would not."""

import json
import sys
from pathlib import Path

import datasets
import pytest
import toml
from datasets.fingerprint import Hasher
from PIL import Image

from utils import dataset as current
from test.test_cache_profiles import LOCAL_WORKERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import inherit_metadata_cache as tool  # noqa: E402


@pytest.fixture(autouse=True)
def bounded_workers(monkeypatch):
    monkeypatch.setattr(current, 'NUM_PROC', 1)
    monkeypatch.setattr(current, 'mp', LOCAL_WORKERS)


SIZES = [(64, 64), (96, 64), (64, 96), (128, 64), (80, 64)]
SIZE_BUCKETS = {'size_buckets': [[64, 64, 1], [96, 64, 1], [64, 96, 1]]}
AR_BUCKETS = {'enable_ar_bucket': True, 'resolutions': [64], 'min_ar': 0.5, 'max_ar': 2.0, 'num_ar_buckets': 5}


def make_source(path):
    path.mkdir(parents=True, exist_ok=True)
    for index, (w, h) in enumerate(SIZES):
        Image.new('RGB', (w, h)).save(path / f'{index}.png')
        path.joinpath(f'{index}.txt').write_text(f'tag{index}, red, blue', encoding='utf-8')


def directory(path, **settings):
    return {'path': str(path), 'shuffle_metadata': False, **settings}


def build(path, model, framerate, cached, geometry, **settings):
    return current.DirectoryDataset(directory(path, **settings), {'resolutions': [64], **geometry}, model,
                                    framerate=framerate, round_to_multiple=16,
                                    skip_dataset_validation=True, caches_text_embeddings=cached)


def cache_donor(path, geometry, **settings):
    """What an Anima run leaves behind: framerate 16, text embeddings cached."""
    donor = build(path, 'anima', 16, True, geometry, **settings)
    donor.cache_metadata()
    return donor


def write_toml(tmp_path, path, geometry, **settings):
    config = {'resolutions': [64], **geometry, 'directory': [directory(path, **settings)]}
    file = tmp_path / 'dataset.toml'
    file.write_text(toml.dumps(config))
    return file


def forbid_media_scan(monkeypatch):
    """Fail if the dataset enumerates and opens the files: the snapshot bootstrap does that
    through a reuse_metadata_cache = false DirectoryDataset."""
    original = current.DirectoryDataset._get_ungrouped_metadata

    def guarded(self, *args, **kwargs):
        if not self.reuse_metadata_cache:
            raise AssertionError('the inherited snapshot should make the media scan unnecessary')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(current.DirectoryDataset, '_get_ungrouped_metadata', guarded)


def grouped_rows(ds):
    """(bucket, image, captions) per grouped metadata row. AR datasets split into size buckets
    only when latents are cached, so their grouping is read at the AR level."""
    buckets = ([(tuple(b.size_bucket), b.metadata_dataset) for b in ds.size_bucket_datasets]
               if ds.use_size_buckets else
               [(tuple(float(x) for x in b.ar_frames), b.metadata_dataset) for b in ds.ar_bucket_datasets])
    return sorted((key, tuple(row['image_spec']), tuple(row['caption']))
                  for key, metadata in buckets for row in metadata)


@pytest.mark.parametrize('geometry', [SIZE_BUCKETS, AR_BUCKETS], ids=['size', 'ar'])
def test_inherited_snapshot_matches_a_fresh_scan_without_scanning(tmp_path, monkeypatch, geometry):
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, geometry)
    file = write_toml(tmp_path, data, geometry)
    assert tool.main(['--dataset', str(file)]) == 0

    # Reference: Iris scanning the same files itself (under another cache name).
    fresh = build(data, 'iris_fresh', None, False, geometry)
    fresh.cache_metadata()

    forbid_media_scan(monkeypatch)
    iris = build(data, 'iris', None, False, geometry)
    iris.cache_metadata()

    inherited = datasets.load_from_disk(str(iris.snapshot_path))
    scanned = datasets.load_from_disk(str(fresh.snapshot_path))
    key = lambda r: tuple(r['image_spec'])  # noqa: E731
    assert sorted(inherited.to_list(), key=key) == sorted(scanned.to_list(), key=key)
    assert grouped_rows(iris) == grouped_rows(fresh) and grouped_rows(iris)
    # The donor is untouched and still loads.
    again = build(data, 'anima', 16, True, geometry)
    again.cache_metadata(trust_cache=True)
    assert len(grouped_rows(again)) == len(SIZES)
    provenance = json.loads((iris.snapshot_dir / 'inherited_from.json').read_text())
    assert provenance['donor_model'] == 'anima' and provenance['rows'] == len(SIZES)


def test_iris_caption_settings_are_applied_to_the_inherited_raw_captions(tmp_path, monkeypatch):
    """The snapshot holds raw captions; the target's own caption settings still decide the text."""
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    file = write_toml(tmp_path, data, SIZE_BUCKETS)
    assert tool.main(['--dataset', str(file)]) == 0
    iris = build(data, 'iris', None, False, SIZE_BUCKETS, caption_prefix='photo, ')
    iris.cache_metadata()
    captions = [caption for *_, (caption,) in grouped_rows(iris)]
    assert captions and all(c.startswith('photo, tag') for c in captions)


def test_dry_run_writes_nothing(tmp_path):
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    file = write_toml(tmp_path, data, SIZE_BUCKETS)
    assert tool.main(['--dataset', str(file), '--dry-run']) == 0
    assert not list((data / 'cache' / 'iris').rglob('source_metadata'))


@pytest.mark.parametrize('change', ['resolution', 'frame_buckets', 'deleted', 'replaced', 'no_donor'])
def test_refuses_whenever_the_snapshot_would_differ(tmp_path, change, capsys):
    data = tmp_path / 'data'
    make_source(data)
    geometry = dict(SIZE_BUCKETS)
    if change != 'no_donor':
        cache_donor(data, geometry)
    settings = {}
    if change == 'resolution':
        geometry = {'size_buckets': [[64, 64, 1], [128, 64, 1]]}
    elif change == 'frame_buckets':
        settings['frame_buckets'] = [1, 33]
    elif change == 'deleted':
        for index in range(len(SIZES)):
            (data / f'{index}.png').unlink()
    elif change == 'replaced':
        for index in range(len(SIZES)):
            Image.new('RGB', (64, 128)).save(data / f'{index}.png')
    file = write_toml(tmp_path, data, geometry, **settings)
    assert tool.main(['--dataset', str(file)]) == 1
    assert '[ERROR]' in capsys.readouterr().out
    assert not list((data / 'cache' / 'iris').rglob('source_metadata'))


def test_existing_target_snapshot_is_kept_unless_forced(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    iris = build(data, 'iris', None, False, SIZE_BUCKETS)
    iris.cache_metadata()
    own = iris.snapshot_path
    file = write_toml(tmp_path, data, SIZE_BUCKETS)
    assert tool.main(['--dataset', str(file)]) == 0          # skipped, not an error
    assert iris.snapshot_path == own and not (iris.snapshot_dir / 'inherited_from.json').exists()

    assert tool.main(['--dataset', str(file), '--force']) == 0
    forbid_media_scan(monkeypatch)
    forced = build(data, 'iris', None, False, SIZE_BUCKETS)
    forced.cache_metadata()
    assert forced.snapshot_dir.name.startswith('inherited_') and own.exists()
    assert len(grouped_rows(forced)) == len(SIZES)


def test_geometry_hash_is_unchanged_by_the_refactor(tmp_path):
    """snapshot_geometry() must hash exactly like the inline dict it replaced, or every existing
    snapshot cache would be orphaned."""
    data = tmp_path / 'data'
    make_source(data)
    for geometry in (SIZE_BUCKETS, AR_BUCKETS):
        ds = build(data, 'anima', 16, True, geometry)
        old = {'size_buckets': ds.size_buckets.tolist() if ds.use_size_buckets else None,
               'ars': ds.ars.tolist(), 'frames': ds.frame_buckets.tolist(),
               'resolutions': ds.resolutions.tolist() if not ds.use_size_buckets else None,
               'round_to_multiple': ds.round_to_multiple, 'framerate': ds.framerate,
               'mask_path': str(ds.mask_path), 'control_path': str(ds.control_path),
               'default_mask_file': str(ds.default_mask_file), 'shuffle': ds.shuffle_metadata}
        assert ds.geometry_dir.name == Hasher.hash(old)
        assert Hasher.hash(ds.snapshot_geometry(framerate=None)) != ds.geometry_dir.name


def test_check_all_finds_a_missing_file_the_sample_skipped(tmp_path, capsys):
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    (data / '3.png').unlink()
    file = write_toml(tmp_path, data, SIZE_BUCKETS)
    # A one-file sample that happens to miss 3.png passes...
    assert tool.main(['--dataset', str(file), '--sample', '1', '--dry-run']) in (0, 1)
    sampled = capsys.readouterr().out
    # ...but checking every file never does.
    assert tool.main(['--dataset', str(file), '--sample', '0', '--check-all', '--dry-run']) == 1
    out = capsys.readouterr().out
    assert '--check-all: 1 of 5 recorded files are missing' in out and '3.png' in out
    assert '[READY]' in sampled or '[ERROR]' in sampled


def test_a_snapshot_published_by_another_pod_during_the_copy_is_kept(tmp_path, monkeypatch):
    """Two pods (or a pod and the target's own first scan) publishing at once: the loser
    leaves the winner's snapshot alone and no staging directory behind."""
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    file = write_toml(tmp_path, data, SIZE_BUCKETS)
    dataset_config = toml.load(file)
    plan = tool.plan_directory(dataset_config['directory'][0], dataset_config, 'anima', 'iris', 16)
    assert plan.status == 'ready'

    real_copytree = tool.shutil.copytree

    def copytree_while_another_pod_wins(src, dst, **kwargs):
        result = real_copytree(src, dst, **kwargs)
        real_copytree(src, plan.target.snapshot_path)       # the other pod finishes first
        (plan.target.geometry_dir / 'winner').write_text('')
        return result
    monkeypatch.setattr(tool.shutil, 'copytree', copytree_while_another_pod_wins)
    assert tool.apply_plan(plan, 'anima') is None
    geometry_dir = plan.target.geometry_dir
    assert (geometry_dir / 'winner').exists() and plan.target.snapshot_path.is_dir()
    assert not list(geometry_dir.glob('.source_metadata.*')) and not list(geometry_dir.glob('*.tmp'))


def test_falls_back_to_copying_where_hard_links_are_not_supported(tmp_path, monkeypatch):
    """SMB shares (Azure Files), separate mounts for donor and target, or a donor cache owned by
    another UID refuse os.link; the snapshot is copied instead and works the same."""
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    file = write_toml(tmp_path, data, SIZE_BUCKETS)

    def no_links(src, dst):
        raise OSError(18, 'Invalid cross-device link')
    monkeypatch.setattr(tool.os, 'link', no_links)
    assert tool.main(['--dataset', str(file)]) == 0
    forbid_media_scan(monkeypatch)
    iris = build(data, 'iris', None, False, SIZE_BUCKETS)
    iris.cache_metadata()
    assert len(grouped_rows(iris)) == len(SIZES)
    copied = next(iris.snapshot_path.glob('*.arrow'))
    assert copied.stat().st_nlink == 1


def test_refuses_when_the_volume_is_mounted_somewhere_else(tmp_path, capsys):
    """The snapshot records absolute paths. A PVC mounted at another mountPath than the one the
    donor ran with would point training at files that are not there, so it is refused."""
    old_mount = tmp_path / 'mnt_a' / 'data'
    make_source(old_mount)
    cache_donor(old_mount, SIZE_BUCKETS)
    new_mount = tmp_path / 'mnt_b' / 'data'
    new_mount.parent.mkdir()
    old_mount.rename(new_mount)
    file = write_toml(tmp_path, new_mount, SIZE_BUCKETS)
    assert tool.main(['--dataset', str(file)]) == 1
    assert 'no longer exist' in capsys.readouterr().out
    assert not list((new_mount / 'cache' / 'iris').rglob('source_metadata'))


def test_runs_as_a_script_from_anywhere(tmp_path):
    """Run as `python tools/inherit_metadata_cache.py` from another directory, in a fresh
    interpreter without the test shims: ComfyUI's regular utils/ package must not shadow
    diffusion-pipe's namespace utils/ ("No module named 'utils.dataset'")."""
    import os
    import subprocess
    script = Path(tool.__file__).resolve()
    code = (
        'import runpy, sys\n'
        f'sys.argv = [{str(script)!r}]\n'
        f'ns = runpy.run_path({str(script)!r}, run_name="inherit_tool")\n'
        'ns["_force_comfy_cpu_if_needed"]()\n'
        'import utils.dataset\n'
        'print("ok", utils.dataset.__file__)\n'
    )
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert str(script.parent.parent / 'utils' / 'dataset.py') in result.stdout


def test_missing_dataset_file_is_a_clear_error(tmp_path, capsys):
    with pytest.raises(SystemExit):
        tool.main(['--dataset', str(tmp_path / 'dataset_new')])
    assert 'no such file; pass the dataset .toml' in capsys.readouterr().err


def test_reports_progress_while_it_works(tmp_path, capsys):
    # On network storage a run takes minutes; every stage says what it is doing as it starts.
    data = tmp_path / 'data'
    make_source(data)
    cache_donor(data, SIZE_BUCKETS)
    file = write_toml(tmp_path, data, SIZE_BUCKETS)
    assert tool.main(['--dataset', str(file)]) == 0
    err = capsys.readouterr().err
    for stage in ('importing torch', 'directory 1/1', 'looking for the', 'loading the donor snapshot',
                  'checking', 'publishing the snapshot'):
        assert stage in err, stage
    assert any(line.startswith('[00:') and 'importing torch' in line for line in err.splitlines())
