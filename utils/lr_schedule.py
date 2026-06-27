"""Learning-rate scheduler construction.

Kept free of training-stack imports so it can be unit-tested on CPU. All step
counts here are optimizer (global) steps, which is the unit DeepSpeed advances
the scheduler in: it calls ``lr_scheduler.step()`` once per ``train_batch`` (after
gradient accumulation), so warmup and the schedule length are counted in global
steps, not micro-batches, and do not change with the data-parallel world size.
"""

import torch


def create_lr_scheduler(optimizer, scheduler_type, total_steps, warmup_steps=0, num_cycles=1):
    """Build the LR scheduler, optionally prefixed with a linear warmup.

    total_steps is the total number of optimizer steps (epochs * steps_per_epoch).
    num_cycles is only used by 'cosine_with_restarts' and controls how many cosine
    restarts happen over the run (1 reproduces a plain cosine decay).
    """
    if scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
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
