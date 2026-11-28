"""Fixes from the audit of the launch glue: serialized loads, warnings, early checks, the resume record."""
import inspect
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from utils import common, memory_partition
from utils import model_teacher as mt
from utils.loss_scale import computes_in_float16

ROOT = Path(__file__).resolve().parent.parent


def run_workers(worker, args, nprocs, timeout=90):
    import time
    context = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout
    finished = False
    while not finished and time.monotonic() < deadline:
        finished = context.join(5)
    if not finished:
        for process in context.processes:
            process.terminate()
        pytest.fail('the ranks are still waiting on each other: a barrier only some of them reach')


def _init(rank, init_file, size):
    os.environ.setdefault('DS_ACCELERATOR', 'cpu')
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['LOCAL_SIZE'] = str(size)
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=size, verbose=False)


def _failing_rank_worker(rank, init_file, nested):
    _init(rank, init_file, 3)
    import torch.distributed as dist
    from utils.common import one_at_a_time
    order = []
    raised = None
    try:
        with one_at_a_time():
            order.append(rank)
            if rank == 1:
                if nested:
                    with one_at_a_time():
                        raise KeyError('inner')
                raise KeyError('body')
    except KeyError as e:
        raised = e
    # Every rank got through all the barriers: this one would hang otherwise.
    gathered = [None] * 3
    dist.all_gather_object(gathered, (rank, raised is not None))
    assert gathered == [(0, False), (1, True), (2, False)]
    assert order == [rank]
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize('nested', [False, True])
def test_a_rank_whose_turn_fails_still_passes_every_barrier_and_then_raises(tmp_path, nested):
    run_workers(_failing_rank_worker, (str(tmp_path / 'init'), nested), nprocs=3)


def test_serialized_per_node_is_a_plain_body_without_a_process_group(monkeypatch):
    monkeypatch.setattr(common.dist, 'is_initialized', lambda: False)
    assert isinstance(common.serialized_per_node(), type(common.nullcontext()))


def test_serialized_per_node_serializes_when_several_ranks_share_the_group(monkeypatch):
    monkeypatch.setattr(common.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(common.dist, 'get_world_size', lambda: 2)
    barriers = []
    monkeypatch.setattr(common.dist, 'barrier', lambda *a, **k: barriers.append(1))
    monkeypatch.setenv('LOCAL_SIZE', '2')
    monkeypatch.setenv('LOCAL_RANK', '0')
    with common.serialized_per_node():
        pass
    assert len(barriers) == 2
    monkeypatch.setattr(common.dist, 'get_world_size', lambda: 1)
    with common.serialized_per_node():
        pass
    assert len(barriers) == 2


def test_one_at_a_time_counter_is_back_to_zero_after_a_failure(monkeypatch):
    monkeypatch.setattr(common.dist, 'barrier', lambda *a, **k: None, raising=False)
    monkeypatch.setenv('LOCAL_SIZE', '1')
    monkeypatch.setenv('LOCAL_RANK', '0')
    with pytest.raises(KeyError):
        with common.one_at_a_time():
            raise KeyError('x')
    assert common._serialized == 0


@pytest.mark.parametrize('table, tag, skipped', [
    ({'loss_weight': 0.5, 'llm_path': 'x', 'decay_steps': 100}, 'global_step500', True),
    ({'loss_weight': 0.5, 'llm_path': 'x', 'decay_steps': 100}, 'global_step10', False),
    ({'loss_weight': 0.5, 'path': 'x', 'decay_steps': 100}, 'global_step500', False),
    ({'loss_weight': 0.5, 'decay': 'linear', 'decay_steps': 100.0}, 'global_step500', True),
    ({'loss_weight': 0.5, 'decay': 'linear', 'decay_steps': 100.5}, 'global_step99', True),
    ({'loss_weight': 0.5, 'decay': 'linear', 'decay_steps': True}, 'global_step500', False),
])
def test_skip_finished_teacher_follows_the_decay_the_table_defaults_to(table, tag, skipped):
    assert mt.skip_finished_teacher({'teacher': dict(table)}, tag) is skipped


def test_warn_ignored_prints_a_message_once(capsys, monkeypatch):
    monkeypatch.setattr(common, '_warned', set())
    monkeypatch.delenv('RANK', raising=False)
    common.warn_ignored('hello')
    common.warn_ignored('hello')
    common.warn_ignored('other')
    assert capsys.readouterr().out.count('WARNING:') == 2


def test_warn_ignored_is_silent_off_rank_zero(capsys, monkeypatch):
    monkeypatch.setattr(common, '_warned', set())
    monkeypatch.setenv('RANK', '3')
    common.warn_ignored('hello')
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('environ, rank', [
    ({}, 0), ({'RANK': '5'}, 5), ({'OMPI_COMM_WORLD_RANK': '2'}, 2), ({'PMI_RANK': '7'}, 7),
    ({'RANK': '', 'LOCAL_RANK': '1'}, 1),
])
def test_launcher_rank_reads_the_launchers_environment(monkeypatch, environ, rank):
    for name in ('RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    assert common.launcher_rank() == rank


@pytest.mark.parametrize('model, expected', [
    ({'type': 'moss_image', 'dtype': 'float16'}, True),
    ({'type': 'moss_image', 'dtype': 'bfloat16'}, False),
    ({'type': 'moss_image', 'dtype': 'float16', 'compute_dtype': 'bfloat16'}, False),
    ({'type': 'moss_image', 'dtype': 'bfloat16', 'compute_dtype': 'float16'}, True),
    ({'type': 'anima', 'dtype': 'float16'}, False),
])
def test_float16_compute_follows_the_weights_dtype_of_moss_image_without_compute_dtype(model, expected):
    assert computes_in_float16(model) is expected


def test_resolve_local_rank_always_sets_local_size(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    environ = {'WORLD_SIZE': '8', 'LOCAL_RANK': '2'}
    assert common.resolve_local_rank(-1, environ) == 2
    assert environ['LOCAL_SIZE'] == '4'
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    environ = {'WORLD_SIZE': '8', 'LOCAL_RANK': '0'}
    common.resolve_local_rank(-1, environ)
    assert environ['LOCAL_SIZE'] == '1'
    environ = {'WORLD_SIZE': '8', 'LOCAL_RANK': '0', 'LOCAL_WORLD_SIZE': '2'}
    common.resolve_local_rank(-1, environ)
    assert environ['LOCAL_SIZE'] == '2'


def test_read_saved_parts(tmp_path):
    tag = tmp_path / 'global_step3'
    tag.mkdir()
    assert memory_partition.read_saved_parts(tmp_path, 'global_step3') is None
    (tag / memory_partition.PARTS_FILE).write_text('{not json')
    assert memory_partition.read_saved_parts(tmp_path, 'global_step3') is None
    (tag / memory_partition.PARTS_FILE).write_text(json.dumps({'other': 1}))
    assert memory_partition.read_saved_parts(tmp_path, 'global_step3') is None
    (tag / memory_partition.PARTS_FILE).write_text(json.dumps({'parts': [0, 3, 9]}))
    assert memory_partition.read_saved_parts(tmp_path, 'global_step3') == [0, 3, 9]
    assert memory_partition.read_saved_parts(tmp_path, 'missing') is None


def _train_main():
    import train
    return inspect.getsource(train).split("if __name__ == '__main__':", 1)[1]


def test_train_py_checks_the_world_size_and_the_cache_only_options_before_the_heavy_work():
    source = (ROOT / 'train.py').read_text()
    assert source.index("dist.get_world_size() % int(config['pipeline_stages'])") < source.index('check_checkpoint_files(')
    assert "if args.cache_status and regenerate_cache:" in source
    assert source.index("if args.cache_status and regenerate_cache:") < source.index('dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache')


def test_a_resume_without_a_saved_ema_starts_the_average_from_ema_init_path():
    source = (ROOT / 'train.py').read_text()
    window = source.split('found, missing = load_ema_checkpoint(ema, run_dir, tag)', 1)[1].split('\n        if holds_ema:', 1)[0]
    assert "if config.get('ema_init_path') and found:" in window
    assert "load_ema_weights(ema, config['ema_init_path'])" in window


def test_the_stock_anima_teacher_is_built_one_rank_at_a_time():
    source = (ROOT / 'models' / 'cosmos_predict2.py').read_text()
    window = source.split('self.teacher = self._build_teacher(dtype)', 1)[0].rsplit('if self.teacher_cfg.enabled:', 1)[1]
    assert 'with serialized_per_node():' in window
    assert 'all_ok(error)' in source.split('self.teacher = self._build_teacher(dtype)', 1)[1][:200]
