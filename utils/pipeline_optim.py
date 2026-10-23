"""Optimizers under pipeline parallelism, where each stage builds its own over its own layers.

An optimizer that sums a statistic over all its parameters sees only one stage's share of the
model. GenericOptim's gradient norm and Prodigy's step size are all-reduced over the pipeline on
every step, so every stage must join that collective, including a stage with nothing to train.
"""

import math
import types
from collections import defaultdict

import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from utils.common import is_main_process

# Step size estimated from sums over all parameters, with no way to share them between stages
# here: each stage would train at its own rate. Prodigy is synchronised instead (sync_prodigy).
STAGE_LOCAL_OPTIMIZERS = ('dadaptadagrad', 'dadaptadam', 'dadaptadan', 'dadaptlion', 'dadaptsgd')


def pipeline_collective(optim_type_lower, gradient_release, num_stages):
    """The collective the stage optimizers run over the pipeline on each step: 'genericoptim',
    'prodigy' or None. Gradient release steps each parameter on its own and runs none."""
    if gradient_release:
        return None
    if optim_type_lower == 'genericoptim' or (optim_type_lower == 'prodigy' and num_stages > 1):
        return optim_type_lower
    return None


def check_pipeline_optimizer(optim_type, num_stages):
    if num_stages > 1 and optim_type.lower() in STAGE_LOCAL_OPTIMIZERS:
        raise ValueError(
            f"[optimizer] type = '{optim_type}' estimates its step size from all the parameters it "
            f'updates, and with pipeline_stages = {num_stages} each stage would estimate its own. '
            "Use pipeline_stages = 1, or type = 'Prodigy', which is synchronised across stages."
        )


class DummyOptimizer(torch.optim.Optimizer):
    """The optimizer of a pipeline stage without trainable parameters. It joins the stage
    optimizers' collective (pipeline_collective) with zeros, or they would wait for it forever."""

    def __init__(self, collective=None, group=None):
        self.state = defaultdict(dict)
        self.param_groups = []
        self.collective = collective
        self.group = group

    def step(self, closure=None):
        if self.collective == 'genericoptim':
            norm = get_accelerator().FloatTensor([0.0])
            dist.all_reduce(norm, op=dist.ReduceOp.SUM, group=self.group)
            # The whole model's gradient norm, logged from global rank 0 like GenericOptim's.
            self._grad_norm = norm[0].item() ** 0.5
        elif self.collective == 'prodigy':
            if not getattr(self, '_prodigy_state_shared', False):
                _share_prodigy_state([], self.group)
                self._prodigy_state_shared = True
            _all_reduce_prodigy_sums(torch.zeros(2, device=get_accelerator().current_device_name()), self.group)

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def _all_reduce_prodigy_sums(sums, group):
    sums = sums.float()
    dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    return sums


def _share_prodigy_state(groups, group):
    """Give every stage the same d, d_max and d_numerator before the first step of a run.

    A checkpoint saved before Prodigy was synchronised holds each stage's own estimate. With the
    default growth_rate = inf, d is the running maximum of d_hat, so each stage would keep its own
    for the rest of the run: take the largest, and the sum of the stage numerators, which is what
    the shared one is. Stages whose numerators already agree were synchronised (or have not
    stepped yet) and keep theirs. groups is empty on a stage with nothing to train.
    """
    device = get_accelerator().current_device_name()
    if groups:
        first, last = groups[0], groups[-1]
        numerator = first['d_numerator'].item() if 'd_numerator' in first else 0.0
        largest = torch.tensor([first['d'], last['d_max'], numerator, -numerator], dtype=torch.float64, device=device)
        total = torch.tensor([numerator], dtype=torch.float64, device=device)
    else:
        largest = torch.full((4,), -math.inf, dtype=torch.float64, device=device)
        total = torch.zeros(1, dtype=torch.float64, device=device)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    if not groups:
        return
    d, d_max, most, least = largest.tolist()
    legacy = most != -least
    if legacy:
        old = first['d_numerator']
        numerator = torch.tensor([total.item()], dtype=old.dtype, device=old.device)
        if is_main_process():
            print(f'WARNING: the checkpoint holds a Prodigy step size per pipeline stage (from before they were '
                  f'shared); continuing with the largest, d = {d:.3g}.')
    for g in groups:
        g['d'] = d
        g['d_max'] = d_max
        if legacy:
            g['d_numerator'] = numerator


def check_prodigy_learning_rates(optimizer, optim_type_lower, gradient_release, num_stages):
    """Prodigy steps every parameter group with the first group's lr (d * lr), so the learning
    rates a model gives its groups are not applied. With pipeline_stages > 1 each stage would use
    its own first group's, and weigh its share of the shared sums with it. Call on every rank."""
    if optim_type_lower != 'prodigy' or gradient_release:
        return
    lrs = sorted({g['lr'] for g in optimizer.param_groups})
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, lrs)
    lrs = sorted(set().union(*gathered))
    if len(lrs) <= 1:
        return
    if num_stages > 1:
        raise ValueError(
            f"[optimizer] type = 'Prodigy' steps every parameter group with one learning rate, but this model's "
            f'groups have {lrs}, and with pipeline_stages = {num_stages} each stage would take its own. Give '
            'every group the same learning rate, or use pipeline_stages = 1.'
        )
    if is_main_process():
        print(f"WARNING: [optimizer] type = 'Prodigy' steps every parameter group with the first group's "
              f'learning rate: the per-group learning rates {lrs} are not applied.')


def sync_prodigy(optimizer, group):
    """Make pytorch_optimizer's Prodigy estimate d over the whole pipeline instead of one stage.

    Prodigy.step sets d from d_numerator and d_de_nom, two sums over the parameters it owns.
    Within a step the new d is used only for the eps term of the update denominator (the update
    itself uses d_lr, taken from the previous d), so the step runs as is, then the stage's shares
    of the two sums are all-reduced and d, d_max and d_hat are recomputed from the totals exactly
    as Prodigy does. The differences from one Prodigy over the whole model: that eps term,
    d * eps against sqrt(exp_avg_sq), for that one step; and a stage whose own share of d_de_nom
    is 0 (every gradient it has seen was 0) skips that step, its weight decay included.
    Every group must have the same lr (check_prodigy_learning_rates).
    """
    inner_step = optimizer.step

    def step(self, closure=None):
        groups = self.param_groups
        if not getattr(self, '_prodigy_state_shared', False):
            _share_prodigy_state(groups, group)
            self._prodigy_state_shared = True
        first, last = groups[0], groups[-1]
        d, d_max = first['d'], last['d_max']
        previous = first['d_numerator'].clone() if 'd_numerator' in first else None
        loss = inner_step(closure)

        beta2 = first['betas'][1]
        beta3 = first['beta3'] if first['beta3'] is not None else math.sqrt(beta2)
        numerator = first['d_numerator']
        decayed = previous.to(numerator.device) * beta3 if previous is not None else torch.zeros_like(numerator)
        de_nom = sum(
            (self.state[p]['s'].abs().sum() for g in groups for p in g['params'] if p.grad is not None),
            torch.zeros((), device=numerator.device),
        )
        sums = _all_reduce_prodigy_sums(torch.stack([(numerator - decayed).reshape(()), de_nom.reshape(())]), group)
        numerator = (decayed + sums[0]).to(numerator.dtype)
        de_nom = sums[1:].to(numerator.dtype)

        if de_nom.item() == 0:
            for g in groups:
                g['d_numerator'] = numerator
            return loss
        d_hat = (last['d_coef'] * numerator / de_nom).item()
        if d == last['d0']:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * last['growth_rate'])
        for g in groups:
            g['d_numerator'] = numerator
            g['d_de_nom'] = de_nom
            g['d'] = d
            g['d_max'] = d_max
            g['d_hat'] = d_hat
        return loss

    # A bound method: torch's LR schedulers rewrap optimizer.step through its __func__.
    optimizer.step = types.MethodType(step, optimizer)
    return optimizer
