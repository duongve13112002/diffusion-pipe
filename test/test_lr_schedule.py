"""Tests for utils/lr_schedule.py. CPU only, no DeepSpeed."""

import math

import pytest
import torch
from torch import nn

from utils.lr_schedule import create_lr_scheduler, validate_min_ratio


def _collect_lrs(scheduler, optimizer, steps):
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    return lrs


def _make_optimizer(lr=1.0):
    model = nn.Linear(2, 2)
    return torch.optim.SGD(model.parameters(), lr=lr)


def test_constant_keeps_lr_flat():
    opt = _make_optimizer(lr=0.5)
    scheduler = create_lr_scheduler(opt, 'constant', total_steps=10)
    lrs = _collect_lrs(scheduler, opt, 10)
    assert all(abs(lr - 0.5) < 1e-9 for lr in lrs)


def test_linear_decays_to_zero():
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'linear', total_steps=10)
    lrs = _collect_lrs(scheduler, opt, 11)
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(b <= a + 1e-9 for a, b in zip(lrs, lrs[1:]))


def test_cosine_decays_monotonically():
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'cosine', total_steps=10)
    lrs = _collect_lrs(scheduler, opt, 11)
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[-1] < 0.01
    assert all(b <= a + 1e-9 for a, b in zip(lrs, lrs[1:]))


def test_cosine_with_restarts_restarts():
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'cosine_with_restarts', total_steps=20, num_cycles=2)
    lrs = _collect_lrs(scheduler, opt, 20)
    # There must be at least one step where the LR jumps back up (a restart).
    increases = [b - a for a, b in zip(lrs, lrs[1:]) if b > a + 1e-6]
    assert len(increases) >= 1
    # The restart should bring the LR back near the base value.
    assert max(lrs[1:]) > 0.9


def test_cosine_with_restarts_one_cycle_matches_plain_cosine():
    opt_a = _make_optimizer(lr=1.0)
    sched_a = create_lr_scheduler(opt_a, 'cosine_with_restarts', total_steps=12, num_cycles=1)
    lrs_a = _collect_lrs(sched_a, opt_a, 12)

    opt_b = _make_optimizer(lr=1.0)
    sched_b = create_lr_scheduler(opt_b, 'cosine', total_steps=12)
    lrs_b = _collect_lrs(sched_b, opt_b, 12)

    for a, b in zip(lrs_a, lrs_b):
        assert a == pytest.approx(b, abs=1e-6)


def test_warmup_ramps_then_follows_schedule():
    opt = _make_optimizer(lr=1.0)
    warmup_steps = 5
    scheduler = create_lr_scheduler(opt, 'cosine', total_steps=20, warmup_steps=warmup_steps)
    lrs = _collect_lrs(scheduler, opt, 20)
    # Warmup phase ramps strictly up toward the base LR.
    assert lrs[0] < lrs[1] < lrs[warmup_steps - 1]
    assert lrs[warmup_steps - 1] < 1.0 + 1e-9
    # Peak is reached around the end of warmup.
    assert max(lrs) == pytest.approx(1.0, abs=1e-6)


def test_warmup_with_restarts_composes():
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'cosine_with_restarts', total_steps=20, warmup_steps=5, num_cycles=2)
    lrs = _collect_lrs(scheduler, opt, 20)
    # Warmup ramp at the start.
    assert lrs[0] < lrs[4]
    # A restart happens later in the cosine phase.
    increases_after_warmup = [b - a for a, b in zip(lrs[5:], lrs[6:]) if b > a + 1e-6]
    assert len(increases_after_warmup) >= 1


def test_unknown_scheduler_raises():
    opt = _make_optimizer()
    with pytest.raises(NotImplementedError, match='Unknown lr_scheduler'):
        create_lr_scheduler(opt, 'made_up', total_steps=10)


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_num_cycles_raises(bad):
    opt = _make_optimizer()
    with pytest.raises(ValueError, match='lr_scheduler_num_cycles'):
        create_lr_scheduler(opt, 'cosine_with_restarts', total_steps=10, num_cycles=bad)


def test_min_ratio_ends_cosine_at_a_fraction_of_each_groups_own_rate():
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD([{'params': [model.weight], 'lr': 1.0}, {'params': [model.bias], 'lr': 0.01}])
    scheduler = create_lr_scheduler(opt, 'cosine', total_steps=20, warmup_steps=5, min_ratio=0.1)
    seen = []
    for _ in range(40):
        seen.append([g['lr'] for g in opt.param_groups])
        opt.step()
        scheduler.step()
    assert seen[0][0] == pytest.approx(1 / 5) and seen[5][0] == pytest.approx(1.0)
    assert seen[-1][0] == pytest.approx(0.1) and seen[-1][1] == pytest.approx(0.001)
    # The decay spans the 15 steps after the warmup: the rate after the 20th step of the run is the floor.
    assert seen[20][0] == pytest.approx(0.1) and seen[19][0] > 0.1
    cosine = [row[0] for row in seen[5:21]]
    assert all(b <= a + 1e-9 for a, b in zip(cosine, cosine[1:]))
    assert seen[5 + 7][0] == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 7 / 15)))


@pytest.mark.parametrize('kind', ['cosine', 'linear'])
def test_min_ratio_is_reached_on_the_last_step_of_a_run_with_warmup(kind):
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, kind, total_steps=100, warmup_steps=20, min_ratio=0.1)
    lrs = _collect_lrs(scheduler, opt, 101)
    assert lrs[-1] == pytest.approx(0.1) and min(lrs[20:]) == pytest.approx(0.1)


def test_without_min_ratio_a_warmup_still_shifts_the_schedule_as_before():
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'linear', total_steps=100, warmup_steps=20)
    assert _collect_lrs(scheduler, opt, 101)[-1] == pytest.approx(0.2)


def test_min_ratio_linear_and_default_unchanged():
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'linear', total_steps=10, min_ratio=0.25)
    assert _collect_lrs(scheduler, opt, 11)[-1] == pytest.approx(0.25)
    opt = _make_optimizer(lr=1.0)
    scheduler = create_lr_scheduler(opt, 'cosine', total_steps=10, min_ratio=None)
    assert _collect_lrs(scheduler, opt, 11)[-1] == pytest.approx(1e-6, abs=1e-5)


@pytest.mark.parametrize('kind, ratio', [('constant', 0.1), ('cosine_with_restarts', 0.1), ('cosine', 1.0),
                                         ('cosine', -0.1), ('cosine', True), ('cosine', '0.1')])
def test_min_ratio_refusals(kind, ratio):
    with pytest.raises(ValueError, match='lr_scheduler_min_ratio'):
        create_lr_scheduler(_make_optimizer(), kind, total_steps=10, min_ratio=ratio)



def test_min_ratio_is_refused_next_to_force_constant_lr():
    validate_min_ratio(0.1, 'cosine', None)
    with pytest.raises(ValueError, match='force_constant_lr'):
        validate_min_ratio(0.1, 'cosine', 1e-4)
    validate_min_ratio(None, 'cosine', 1e-4)
