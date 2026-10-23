"""How large each optimizer step is, for the training log.

train/grad_norm is the norm of the whole model's gradient before clipping (the one
gradient_clipping compares against), and train/grad_clipped is 1 on the steps it clipped, so its
average over a window is the share of clipped steps. train/update_norm is the norm of the change
the step made to the trained weights, and train/update_ratio that norm over the weights' norm
before the step: the share of the weights one step moves, which is what the learning rate
controls in the end.

Every rank calls before_step and after_step on the same steps: both norms are summed over the
pipeline's stages. Global rank 0 logs what after_step returns.
"""

import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from utils.patches import counts_in_norm


class StepStats:
    def __init__(self, engine, log_update_norm, gradient_release=False):
        self.engine = engine
        self.log_update_norm = log_update_norm
        self.max_norm = engine.gradient_clipping()
        # Gradient release steps each parameter during backward and frees its gradient: none is
        # left to measure when the step ends.
        self.measure_grad_norm = not gradient_release
        # Each weight once: a tied layer's copies on later stages are left out, as in the norm.
        self.params = [p for p in engine.module.parameters() if p.requires_grad and counts_in_norm(p, engine.mpu)]
        self.snapshot = None

    def before_step(self, measure):
        self.engine.measure_grad_norm = measure and self.measure_grad_norm
        self.engine.last_grad_norm = None
        if measure and self.log_update_norm:
            self.snapshot = [p.detach().clone() for p in self.params]

    def after_step(self):
        stats = {}
        norm = self.engine.last_grad_norm
        if norm is not None and self.measure_grad_norm:
            norm = float(norm.reshape(-1)[0])
            stats['train/grad_norm'] = norm
            if self.max_norm > 0:
                stats['train/grad_clipped'] = float(norm > self.max_norm)
        if self.snapshot is not None:
            snapshot, self.snapshot = self.snapshot, None
            sums = torch.zeros(2, dtype=torch.float64, device=get_accelerator().current_device_name())
            for p, old in zip(self.params, snapshot):
                old = old.float()
                # A swapped block's weights may have moved to another device since.
                new = p.detach().to(old.device, torch.float32)
                sums[0] += (new - old).square().sum().to(sums)
                sums[1] += old.square().sum().to(sums)
            dist.all_reduce(sums, group=self.engine.grid.get_model_parallel_group())
            update, weight = sums.sqrt().tolist()
            stats['train/update_norm'] = update
            stats['train/update_ratio'] = update / weight if weight > 0 else 0.0
        return stats
