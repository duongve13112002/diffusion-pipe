"""Learning-rate scheduler construction.

Kept free of training-stack imports so it can be unit-tested on CPU. All step
counts here are optimizer (global) steps, which is the unit DeepSpeed advances
the scheduler in: it calls ``lr_scheduler.step()`` once per ``train_batch`` (after
gradient accumulation), so warmup and the schedule length are counted in global
steps, not micro-batches, and do not change with the data-parallel world size.
"""

import math

import torch


def validate_min_ratio(min_ratio, scheduler_type):
    """Raise ValueError for a bad lr_scheduler_min_ratio; None (not set) is fine for any scheduler."""
    if min_ratio is None:
        return
    if isinstance(min_ratio, bool) or not isinstance(min_ratio, (int, float)) or not 0 <= min_ratio < 1:
        raise ValueError(f'lr_scheduler_min_ratio must be a number in [0, 1), got {min_ratio!r}')
    if scheduler_type not in ('cosine', 'linear'):
        raise ValueError(f"lr_scheduler_min_ratio applies to lr_scheduler 'cosine' and 'linear', not {scheduler_type!r}")


def create_lr_scheduler(optimizer, scheduler_type, total_steps, warmup_steps=0, num_cycles=1, min_ratio=None):
    """Build the LR scheduler, optionally prefixed with a linear warmup.

    total_steps is the total number of optimizer steps (epochs * steps_per_epoch).
    num_cycles is only used by 'cosine_with_restarts' and controls how many cosine
    restarts happen over the run (1 reproduces a plain cosine decay).
    min_ratio (None or a number in [0, 1)) is where 'cosine' and 'linear' end, as a fraction of each
    parameter group's own learning rate (KohakuMossTrainer's SCHEDULE min_value: 0.1 ends at 10%).
    None keeps the schedules as they were: linear to 0, cosine to an absolute 1e-6.
    """
    validate_min_ratio(min_ratio, scheduler_type)
    # The schedule after the warmup. With a floor set it spans the steps left, so that it reaches the floor on
    # the last step; without one it keeps its earlier length (total_steps from the end of the warmup), so
    # existing runs train exactly as before.
    span = max(1, total_steps - warmup_steps) if min_ratio is not None else total_steps

    if scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=min_ratio or 0.0, total_iters=span)
    elif scheduler_type == 'cosine' and min_ratio is not None:
        total = span

        def factor(step):
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif scheduler_type == 'cosine_with_restarts':
        if isinstance(num_cycles, bool) or not isinstance(num_cycles, int) or num_cycles < 1:
            raise ValueError(f'lr_scheduler_num_cycles must be a positive integer, got {num_cycles!r}')
        restart_period = max(1, total_steps // num_cycles)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=restart_period, T_mult=1, eta_min=1e-6)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {scheduler_type}')

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps])
    return scheduler
