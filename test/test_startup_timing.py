"""utils/startup_timing.py: train.py's startup phases, printed as they end and summed at the end."""

import pytest

from utils import startup_timing


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(startup_timing.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(startup_timing, '_start', None)
    monkeypatch.setattr(startup_timing, '_last', None)
    monkeypatch.setattr(startup_timing, '_phases', [])
    return now


def test_marks_before_begin_do_nothing(capsys):
    startup_timing.mark('anything')
    startup_timing.report()
    assert startup_timing.phases() == [] and capsys.readouterr().out == ''


def test_each_phase_is_timed_from_the_last_mark(clock, capsys):
    startup_timing.begin()
    clock[0] += 12.5
    startup_timing.mark('loading')
    clock[0] += 3.0
    startup_timing.mark('engine')
    assert startup_timing.phases() == [('loading', 12.5), ('engine', 3.0)]
    out = capsys.readouterr().out
    assert '[startup] loading: 12.5 s (12 s since launch)' in out
    assert '[startup] engine: 3.0 s (16 s since launch)' in out
    startup_timing.report()
    out = capsys.readouterr().out
    assert '[startup]   loading: 12.5 s\n' in out and '[startup]   total on rank 0: 15.5 s' in out
    assert '(slowest:' not in out


def test_the_slowest_rank_is_named(clock, monkeypatch, capsys):
    startup_timing.begin()
    clock[0] += 10.0
    startup_timing.mark('loading')
    monkeypatch.setattr(startup_timing.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(startup_timing.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(startup_timing.dist, 'get_world_size', lambda: 3)

    def gather(out, mine):
        out[:] = [mine, [('loading', 10.4)], [('loading', 31.0)]]
    monkeypatch.setattr(startup_timing.dist, 'all_gather_object', gather)
    capsys.readouterr()
    startup_timing.report()
    assert '[startup]   loading: 10.0 s (slowest: rank 2, 31.0 s)' in capsys.readouterr().out


def _report_on_rank(rank, world, port, out_dir):
    """One rank of a real gloo process group: the phases take longer on higher ranks."""
    import contextlib
    import io
    import os
    import torch.distributed as dist
    from utils import startup_timing as timing
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    clock = [0.0]
    timing.time.monotonic = lambda: clock[0]
    timing.begin()
    clock[0] += 5.0 + 20.0 * rank
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        timing.mark('loading')
        clock[0] += 1.0
        timing.mark('engine')
        timing.report()
    with open(os.path.join(out_dir, f'{rank}.txt'), 'w') as f:
        f.write(output.getvalue())
    dist.barrier()
    dist.destroy_process_group()


def test_report_gathers_every_rank_of_a_real_process_group(tmp_path):
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    mp.spawn(_report_on_rank, args=(3, port, str(tmp_path)), nprocs=3)
    rank0 = (tmp_path / '0.txt').read_text()
    assert '[startup]   loading: 5.0 s (slowest: rank 2, 45.0 s)' in rank0
    assert '[startup]   engine: 1.0 s\n' in rank0
    # Only rank 0 prints; every rank took part in the gather.
    assert (tmp_path / '1.txt').read_text() == '' and (tmp_path / '2.txt').read_text() == ''
