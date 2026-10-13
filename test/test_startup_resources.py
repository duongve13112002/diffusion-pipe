"""train.py startup: a CPU share per GPU process, and an unwritable output_dir caught up front."""

import os

import pytest
import torch

from utils import common


def cgroup(tmp_path, v2=None, v1=None):
    root = tmp_path / 'cgroup'
    root.mkdir()
    if v2 is not None:
        (root / 'cpu.max').write_text(v2)
    if v1 is not None:
        (root / 'cpu').mkdir()
        (root / 'cpu' / 'cpu.cfs_quota_us').write_text(str(v1[0]))
        (root / 'cpu' / 'cpu.cfs_period_us').write_text(str(v1[1]))
    return root


@pytest.fixture
def host_cpus(monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(224)), raising=False)


@pytest.mark.parametrize('v2,v1,expected', [
    (None, None, 224),                    # no cgroup files
    ('max 100000\n', None, 224),          # v2, no limit
    ('3200000 100000\n', None, 32),       # v2, 32-CPU limit
    ('150000 100000\n', None, 1),         # v2, 1.5 CPUs
    (None, (-1, 100000), 224),            # v1, no limit
    (None, (6400000, 100000), 64),        # v1, 64-CPU limit
])
def test_usable_cpu_count_honours_the_container_limit(tmp_path, host_cpus, v2, v1, expected):
    assert common.usable_cpu_count(cgroup(tmp_path, v2, v1)) == expected


@pytest.fixture
def restore_threads(monkeypatch):
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def test_each_gpu_process_gets_its_share(tmp_path, host_cpus, restore_threads):
    assert common.limit_cpu_threads(8, cgroup(tmp_path, '3200000 100000')) == 4
    assert torch.get_num_threads() == 4 and os.environ['OMP_NUM_THREADS'] == '4'


def test_never_below_one_thread(tmp_path, host_cpus, restore_threads):
    assert common.limit_cpu_threads(8, cgroup(tmp_path, '200000 100000')) == 1


def test_omp_num_threads_set_by_the_user_wins(tmp_path, host_cpus, restore_threads, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '12')
    before = torch.get_num_threads()
    assert common.limit_cpu_threads(8, cgroup(tmp_path, '3200000 100000')) is None
    assert torch.get_num_threads() == before and os.environ['OMP_NUM_THREADS'] == '12'


def test_output_dir_is_created(tmp_path):
    common.ensure_writable_dir(tmp_path / 'runs' / 'iris', 'output_dir')
    assert (tmp_path / 'runs' / 'iris').is_dir()


@pytest.mark.skipif(os.geteuid() == 0, reason='root can write anywhere')
def test_unwritable_output_dir_stops_with_the_setting_named(tmp_path):
    locked = tmp_path / 'locked'
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(SystemExit, match="output_dir = .*locked/runs.*cannot be created"):
            common.ensure_writable_dir(locked / 'runs', 'output_dir')
    finally:
        locked.chmod(0o700)


def test_output_dir_that_cannot_be_created_stops_with_the_setting_named(tmp_path):
    blocker = tmp_path / 'file'
    blocker.write_text('')
    with pytest.raises(SystemExit, match="output_dir = .*file/runs.*cannot be created"):
        common.ensure_writable_dir(blocker / 'runs', 'output_dir')
