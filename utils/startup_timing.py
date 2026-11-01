"""Where train.py's startup time goes, from launch to the first training step.

train.py marks the end of each startup phase; each mark prints that phase's wall time on rank 0 as
it ends, and report() prints every phase once more with the slowest rank's time and which rank it
was, so a phase that only some ranks are slow in (a node reading from a slower volume, say) shows.
Marks are cheap and local; report() is one all_gather that every rank must reach.
"""

import time

import torch.distributed as dist

_start = None
_last = None
_phases = []


def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def begin():
    """Starts timing; the first phase runs from here."""
    global _start, _last
    _start = _last = time.monotonic()
    _phases.clear()


def mark(name):
    """Ends the current phase as `name` and starts the next one."""
    global _last
    if _start is None:
        return
    now = time.monotonic()
    seconds = now - _last
    _phases.append((name, seconds))
    _last = now
    if _is_main():
        print(f'[startup] {name}: {seconds:.1f} s ({now - _start:.0f} s since launch)', flush=True)


def phases():
    return list(_phases)


def report():
    """Prints every phase with this rank's time and the slowest rank's; every rank calls this."""
    if _start is None:
        return
    mine = phases()
    gathered = [mine]
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, mine)
    if not _is_main():
        return
    lines = ['[startup] time to the first step, rank 0 and the slowest rank per phase:']
    for i, (name, seconds) in enumerate(mine):
        times = [ranks[i][1] if i < len(ranks) and ranks[i][0] == name else None for ranks in gathered]
        known = [(t, rank) for rank, t in enumerate(times) if t is not None]
        slowest, rank = max(known)
        line = f'[startup]   {name}: {seconds:.1f} s'
        if len(known) > 1 and slowest > seconds + max(1.0, 0.1 * seconds):
            line += f' (slowest: rank {rank}, {slowest:.1f} s)'
        lines.append(line)
    lines.append(f'[startup]   total on rank 0: {sum(seconds for _, seconds in mine):.1f} s')
    print('\n'.join(lines), flush=True)
