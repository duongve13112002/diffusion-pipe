"""Multi-node launches: local ranks from any launcher, and dataset caching across nodes.

The caching tests run real gloo process groups in which some ranks pretend to be on another
machine (another hostname); the work queue of global rank 0 lives behind a Unix socket in its
machine's /tmp, which such ranks cannot reach.
"""

import json
import types
import socket

import pytest
import torch.multiprocessing as mp

from utils.common import resolve_local_rank, set_rendezvous_env


@pytest.mark.parametrize('environ, arg, local_rank, local_size', [
    # deepspeed launcher: --local_rank, LOCAL_RANK and LOCAL_SIZE
    ({'WORLD_SIZE': '16', 'LOCAL_RANK': '3', 'LOCAL_SIZE': '8'}, 3, 3, '8'),
    # torchrun / PyTorchJob: no argument, LOCAL_WORLD_SIZE instead of LOCAL_SIZE
    ({'WORLD_SIZE': '16', 'LOCAL_RANK': '5', 'LOCAL_WORLD_SIZE': '8'}, -1, 5, '8'),
    # an old launcher passing only the argument
    ({'WORLD_SIZE': '8', 'LOCAL_SIZE': '8'}, 2, 2, '8'),
    # a single process started as `python train.py`
    ({}, -1, 0, '1'),
])
def test_the_local_rank_comes_from_whichever_launcher_set_it(environ, arg, local_rank, local_size):
    assert resolve_local_rank(arg, environ) == local_rank
    assert environ['LOCAL_RANK'] == str(local_rank) and environ['LOCAL_SIZE'] == local_size


@pytest.mark.parametrize('environ, arg, match', [
    ({'WORLD_SIZE': '16', 'LOCAL_RANK': '5'}, 4, 'disagrees with LOCAL_RANK=5'),
    ({'WORLD_SIZE': '16'}, -1, 'neither LOCAL_RANK nor --local_rank'),
])
def test_an_unknown_or_contradicting_local_rank_is_refused(environ, arg, match):
    with pytest.raises(ValueError, match=match):
        resolve_local_rank(arg, environ)


@pytest.mark.parametrize('environ, arg, port', [
    ({'MASTER_ADDR': 'pod-0', 'MASTER_PORT': '23456'}, None, '23456'),
    ({'MASTER_ADDR': 'pod-0', 'MASTER_PORT': '23456'}, 29600, '29600'),
    ({}, None, '29500'),
])
def test_the_launchers_master_port_is_kept_unless_one_is_given(environ, arg, port):
    addr = environ.get('MASTER_ADDR', 'localhost')
    set_rendezvous_env(arg, environ)
    assert (environ['MASTER_ADDR'], environ['MASTER_PORT']) == (addr, port)


HOSTS = ['node-a', 'node-a', 'node-b']


def _caching_process(queue, mode):
    """Stands in for _cache_fn, in the process global rank 0 starts for it."""
    import os
    import time
    if mode == 'killed':
        # Killed (out of memory, say) with no chance to end the queue.
        queue.put('task')
        os._exit(9)
    try:
        queue.put('boom' if mode == 'rank_failed' else 'task')
        if mode == 'process_failed':
            raise RuntimeError('caching process failed')
        if mode == 'rank_failed':
            # As a map worker would, wait for the result the failed rank never sends.
            time.sleep(600)
    finally:
        queue.put(None)


def _caching_worker(rank, init_file, out_dir, mode):
    from datetime import timedelta
    import deepspeed
    import multiprocess
    # As train.py does: the queue's proxies authenticate with this key in every process.
    multiprocess.current_process().authkey = b'afsaskgfdjh4'
    socket.gethostname = lambda: HOSTS[rank]
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=len(HOSTS), verbose=False, timeout=timedelta(seconds=120))
    from utils import dataset
    dataset.CACHING_POLL_SECONDS = 0.5

    if mode == 'connect_failed' and rank == 1:
        def refuse(payload):
            raise ConnectionRefusedError('cannot reach the queue')
        dataset.pickle = types.SimpleNamespace(loads=refuse)
    queue, manager, error = dataset._share_work_queue()
    process = None
    if rank == 0:
        if mode == 'start_failed':
            # As DatasetManager.cache passes a failed process.start() on.
            error = OSError('cannot start the caching process')
        else:
            process = multiprocess.Process(target=_caching_process, args=(queue, mode))
            process.start()
    tasks = []

    def handle_task(task):
        if task == 'boom':
            raise RuntimeError('task failed')
        tasks.append(task)

    result = {'has_queue': queue is not None}
    try:
        dataset._run_caching(queue, handle_task, process, 1, error)
        result['outcome'] = 'done'
    except RuntimeError as error:
        result['outcome'] = str(error)
    result['tasks'] = tasks
    deepspeed.comm.barrier()
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


@pytest.mark.parametrize('mode, error', [
    ('ok', None),
    ('process_failed', 'The dataset caching process failed with exit code 1'),
    ('killed', 'The dataset caching process ended with exit code 9 before finishing'),
    ('rank_failed', 'task failed'),
    ('start_failed', 'cannot start the caching process'),
    ('connect_failed', 'cannot reach the queue'),
])
def test_only_the_ranks_on_rank_0s_node_take_caching_work_and_all_fail_together(mode, error, tmp_path):
    mp.spawn(_caching_worker, args=(str(tmp_path / 'init'), str(tmp_path), mode), nprocs=len(HOSTS))
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(len(HOSTS))]
    # Ranks 0 and 1 share node-a and the queue; rank 2, on node-b, waits for them.
    assert [r['has_queue'] for r in results] == [True, mode != 'connect_failed', False]
    if error is None:
        assert sorted(results[0]['tasks'] + results[1]['tasks']) == ['task']
        assert [r['outcome'] for r in results] == ['done'] * 3
    else:
        # Every rank fails, with the error of the rank it came from, instead of waiting forever.
        for r in results:
            assert error in r['outcome'], r['outcome']
        if mode != 'connect_failed':
            # Rank 0 may finish the work before rank 1's abort reaches it, and tell node-b all went well:
            # node-b then fails with the others at the check after caching.
            assert 'Dataset caching failed on global rank 0' in results[2]['outcome']


def test_the_caching_process_ends_the_queue_when_it_fails():
    import queue as queue_module
    from utils import dataset

    class Broken:
        def cache_metadata(self, **kwargs):
            raise OSError('unreadable dataset')

    queue = queue_module.Queue()
    with pytest.raises(OSError, match='unreadable dataset'):
        dataset._cache_fn([Broken()], queue, None, 1, False, False, 1)
    assert queue.get_nowait() is None


def _loader(batches_per_epoch):
    from utils.dataset import PipelineDataLoader
    loader = PipelineDataLoader.__new__(PipelineDataLoader)
    loader.dataset = range(batches_per_epoch)
    loader.iter_called = False
    loader.skipped = None
    loader._set_dataset_epoch = lambda: None
    loader._create_dataloader = lambda skip_first_n_batches=0: setattr(loader, 'skipped', skip_first_n_batches)
    loader._pull_batches_from_dataloader = lambda: iter(())
    return loader


@pytest.mark.parametrize('saved, resumed, pulled, skipped', [
    # 2 nodes x 8 GPUs with pipeline_stages = 2 (DP 8), resumed on one node (DP 4): an epoch has
    # twice as many batches of half the size.
    (200, 400, 101, 200),
    (400, 200, 101, 50),
    # Another micro batch size changes the epoch's length in the same way.
    (300, 150, 241, 120),
    (200, 200, 101, 100),
    # Per-bucket rounding: 10 buckets of 3 batches become 10 batches. Late in the epoch, a count of
    # examples would point past its end.
    (30, 10, 30, 9),
    # Saved just after the epoch rolled over.
    (400, 200, 0, 0),
    (200, 200, 0, 0),
])
def test_the_dataloader_resumes_after_the_same_share_of_the_epoch(saved, resumed, pulled, skipped, monkeypatch):
    monkeypatch.setattr('utils.dataset.is_main_process', lambda: True)
    state = _loader(saved)
    state.epoch, state.num_batches_pulled = 3, pulled
    loader = _loader(resumed)
    loader.load_state_dict(state.state_dict())
    assert (loader.epoch, loader.skipped) == (3, skipped)
    assert loader.skipped < resumed


def test_an_old_checkpoint_without_the_epoch_length_resumes_as_before(monkeypatch):
    monkeypatch.setattr('utils.dataset.is_main_process', lambda: True)
    loader = _loader(50)
    loader.load_state_dict({'epoch': 2, 'num_batches_pulled': 101})
    assert (loader.epoch, loader.skipped) == (2, 100)


def test_a_bucket_is_shuffled_in_memory_with_the_fingerprint_its_caches_know(tmp_path):
    import datasets
    from utils.dataset import shuffle_in_memory
    datasets.Dataset.from_dict({'image_spec': [f'{i}.png' for i in range(50)]}).save_to_disk(str(tmp_path))
    files = sorted(p.name for p in tmp_path.iterdir())
    on_disk = datasets.load_from_disk(str(tmp_path))
    shuffled = shuffle_in_memory(on_disk, 1234)
    # Nothing written next to the dataset, which every rank of every node shuffles.
    assert sorted(p.name for p in tmp_path.iterdir()) == files
    expected = on_disk.shuffle(seed=1234)
    assert shuffled._fingerprint == expected._fingerprint
    assert shuffled['image_spec'] == expected['image_spec'] != on_disk['image_spec']


def _signal_worker(rank, init_file, out_dir, name):
    from pathlib import Path
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=2, verbose=False)
    from utils.saver import manual_save_signal
    # Each rank its own view of the output directory: only rank 0 sees the file yet, as a rank
    # that got there first, or a node whose view of the volume is behind, would not.
    save_root = Path(out_dir) / f'view{rank}'
    save_root.mkdir()
    if rank == 0:
        (save_root / name).touch()
    first = manual_save_signal(save_root)
    second = manual_save_signal(save_root)
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump({'first': first, 'second': second, 'left': sorted(p.name for p in save_root.iterdir())}, f)


@pytest.mark.parametrize('name, signal', [('save', [True, False]), ('save_quit', [True, True])])
def test_every_rank_acts_on_the_save_signal_rank_0_sees_once(name, signal, tmp_path):
    mp.spawn(_signal_worker, args=(str(tmp_path / 'init'), str(tmp_path), name), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    for result in results:
        assert result['first'] == signal and result['second'] == [False, False]
    assert results[0]['left'] == []


def _run(root, name, tag=None):
    run = root / name
    run.mkdir(parents=True)
    if tag:
        (run / tag).mkdir()
        (run / 'latest').write_text(tag + '\n')
    return run


def test_resuming_picks_the_newest_run_that_has_a_checkpoint(tmp_path, capsys):
    from utils.common import resume_run_dir
    _run(tmp_path, '20260101_00-00-00', 'global_step100')
    newest = _run(tmp_path, '20260102_00-00-00', 'global_step7')
    # A run that died before its first checkpoint, and a stray file.
    _run(tmp_path, '20260103_00-00-00')
    (tmp_path / 'train.log').write_text('')
    assert resume_run_dir(tmp_path, True) == (str(newest), 'global_step7')
    printed = capsys.readouterr().out
    # Which run, and that a newer one was passed over.
    assert f'Resuming {newest} from checkpoint global_step7' in printed
    assert 'newer directories without one are left as they are: 20260103_00-00-00\n' in printed
    assert resume_run_dir(tmp_path, '20260101_00-00-00') == (str(tmp_path / '20260101_00-00-00'), 'global_step100')
    assert 'WARNING' not in capsys.readouterr().out


@pytest.mark.parametrize('setup, resume, match', [
    (lambda root: None, True, 'no run directory in .* has a checkpoint'),
    (lambda root: _run(root, 'a'), True, 'no run directory in .* has a checkpoint'),
    (lambda root: _run(root, 'a', 'global_step1'), 'typo', 'Nothing to resume in .*typo'),
    (lambda root: (_run(root, 'a', 'global_step1') / 'latest').write_text('global_step2'), True,
     "names checkpoint 'global_step2', which is not in"),
])
def test_there_is_no_silent_new_run_when_nothing_can_be_resumed(setup, resume, match, tmp_path):
    from utils.common import resume_run_dir
    setup(tmp_path)
    with pytest.raises(ValueError, match=match):
        resume_run_dir(tmp_path, resume)


def test_resuming_from_an_output_dir_that_does_not_exist_is_refused(tmp_path):
    from utils.common import resume_run_dir
    with pytest.raises(ValueError, match='no run directory in .* has a checkpoint'):
        resume_run_dir(tmp_path / 'missing', True)


def test_a_save_missing_a_pipeline_stage_fails_instead_of_publishing_part_of_the_model(tmp_path, monkeypatch):
    import threading
    import time
    import torch
    from utils import saver
    monkeypatch.setattr(saver, 'STAGE_FILE_WAIT_SECONDS', 3)
    for stage in (0, 2):
        torch.save({f'w{stage}': torch.zeros(1)}, tmp_path / f'state_dict_{stage}.bin')
    # A file another node wrote shows up late: waited for.
    late = threading.Timer(0.5, lambda: torch.save({}, tmp_path / 'state_dict_1.bin'))
    late.start()
    assert [p.name for p in saver.stage_files(tmp_path, 3)] == [f'state_dict_{s}.bin' for s in range(3)]
    late.join()
    # One that never does: refused, rather than an export without that stage's layers.
    monkeypatch.setattr(saver, 'STAGE_FILE_WAIT_SECONDS', 0.5)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='state_dict_3.bin not found .* volume that every node shares'):
        saver.stage_files(tmp_path, 4)
    assert time.monotonic() - start < 5


def test_a_file_rank_0_wrote_must_show_up_on_every_node(tmp_path):
    import threading
    from utils.common import wait_until_visible
    path = tmp_path / 'train.toml'
    threading.Timer(0.3, path.touch).start()
    wait_until_visible(path, 'hint', seconds=5)
    with pytest.raises(RuntimeError, match='missing.toml, written by global rank 0, is not visible on .*: shared volume'):
        wait_until_visible(tmp_path / 'missing.toml', 'shared volume', seconds=0.5)
