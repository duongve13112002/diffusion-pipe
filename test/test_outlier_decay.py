"""utils/outlier_decay.py: config validation, the two schedules, target selection, the update
itself in fp32 and bf16, and the learning rate it uses."""

import math

import pytest
import torch
from torch import nn

from utils.outlier_decay import (SECOND_COPY_OPTIMIZERS, OutlierDecay, apply_outlier_decay_config_defaults, cap_at,
                                 rate_at, round_to_bfloat16, targeted)


def make_spec(**overrides):
    config = {'outlier_decay': {'rate': 6.0, **overrides}}
    apply_outlier_decay_config_defaults(config)
    return config['outlier_decay']


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8, bias=True)
        self.b = nn.Linear(8, 8, bias=False)
        self.norm = nn.LayerNorm(8)


def optimizer_for(module, lr=1e-3, frozen=()):
    params = [p for n, p in module.named_parameters() if n not in frozen]
    return torch.optim.SGD(params, lr=lr)


def test_config_defaults_and_refusals():
    config = {}
    apply_outlier_decay_config_defaults(config)
    assert config == {}
    spec = make_spec()
    assert spec['cap'] == 16.0 and spec['cap_start'] == 0.0 and spec['warmup'] == 0 and spec['include_1d'] is False
    bad = [
        {'outlier_decay': {}},
        {'outlier_decay': {'rate': -1}},
        {'outlier_decay': {'rate': True}},
        {'outlier_decay': {'rate': 1, 'cap': 0}},
        {'outlier_decay': {'rate': 1, 'cap': 16, 'cap_start': 8}},
        {'outlier_decay': {'rate': 1, 'cap_anneal': -1}},
        {'outlier_decay': {'rate': 1, 'warmup': 1.5}},
        {'outlier_decay': {'rate': 1, 'targets': '('}},
        {'outlier_decay': {'rate': 1, 'include_1d': 'yes'}},
        {'outlier_decay': {'rate': 1, 'typo': 1}},
        {'outlier_decay': 3},
        {'outlier_decay': {'rate': 1}, 'adapter': {'type': 'lora'}},
    ]
    for config in bad:
        with pytest.raises(ValueError):
            apply_outlier_decay_config_defaults(config)
    # Rate 0 is "off" and fine next to an adapter.
    config = {'outlier_decay': {'rate': 0}, 'adapter': {'type': 'lora'}}
    apply_outlier_decay_config_defaults(config)


def test_schedules():
    spec = make_spec(cap=16, cap_start=64, cap_anneal=100, warmup=10)
    assert cap_at(spec, 0) == pytest.approx(64)
    assert cap_at(spec, 50) == pytest.approx(math.sqrt(64 * 16))
    assert cap_at(spec, 100) == 16 and cap_at(spec, 10_000) == 16
    assert rate_at(spec, 0) == pytest.approx(0.6)
    assert rate_at(spec, 4) == pytest.approx(3.0)
    assert rate_at(spec, 9) == 6.0 and rate_at(spec, 500) == 6.0
    flat = make_spec()
    assert cap_at(flat, 0) == 16.0 and rate_at(flat, 0) == 6.0


def test_target_selection():
    module = Stub()
    spec = make_spec()
    assert targeted(spec, 'a.weight', module.a.weight)
    assert not targeted(spec, 'a.bias', module.a.bias)
    assert targeted(make_spec(include_1d=True), 'a.bias', module.a.bias)
    assert targeted(make_spec(targets=r'^a\.'), 'a.weight', module.a.weight)
    assert not targeted(make_spec(targets=r'^a\.'), 'b.weight', module.b.weight)
    assert not targeted(make_spec(exclude='^b'), 'b.weight', module.b.weight)


def test_build_takes_the_stepped_trained_two_dimensional_tensors():
    module = Stub()
    module.b.weight.requires_grad_(False)
    optimizer = optimizer_for(module, frozen=('b.weight',))
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec())
    assert decay.names == ['a.weight']
    assert decay.r0[0] == pytest.approx(module.a.weight.detach().pow(2).mean().sqrt().item(), rel=1e-5)
    # A parameter the optimizer does not step is left alone even if it requires grad.
    module = Stub()
    optimizer = optimizer_for(module, frozen=('b.weight',))
    assert OutlierDecay.build(module.named_parameters(), optimizer, make_spec()).names == ['a.weight']
    # original_name is the name used for the regexes and the report.
    module = Stub()
    for name, p in module.named_parameters():
        p.original_name = 'layer.' + name
    decay = OutlierDecay.build(module.named_parameters(), optimizer_for(module), make_spec(targets='^layer.b'))
    assert decay.names == ['layer.b.weight']


def test_build_refuses_other_dtypes():
    module = Stub().half()
    with pytest.raises(ValueError, match='float32 and bfloat16'):
        OutlierDecay.build(module.named_parameters(), optimizer_for(module), make_spec())


def test_all_zero_tensor_is_skipped():
    module = Stub()
    nn.init.zeros_(module.a.weight)
    decay = OutlierDecay.build(module.named_parameters(), optimizer_for(module), make_spec())
    assert decay.names == ['b.weight']


def test_fp32_update_matches_the_formula():
    torch.manual_seed(0)
    module = Stub()
    optimizer = optimizer_for(module, lr=0.01)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=2.0))
    r0 = decay.r0[decay.names.index('a.weight')]
    with torch.no_grad():
        module.a.weight[0, 0] = 10 * r0
        module.a.weight[1, 1] = -7 * r0
    before = module.a.weight.detach().clone()
    decay.before_step()
    decay.apply(1)
    limit = 2.0 * r0
    expected = before - 0.01 * 6.0 * before.sign() * (before.abs() - limit).clamp_min(0)
    assert torch.allclose(module.a.weight, expected, rtol=1e-6, atol=1e-8)
    assert (module.a.weight.abs() < before.abs()).sum() == 2
    scalars = decay.log_scalars()
    assert scalars['train/outlier_over'] == 2
    assert scalars['train/outlier_max'] == pytest.approx(10 / 2.0, rel=1e-5)


def test_nothing_changes_below_the_cap_or_at_lr_zero():
    module = Stub()
    optimizer = optimizer_for(module, lr=0.01)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=100.0))
    before = [p.detach().clone() for p in module.parameters()]
    decay.before_step()
    decay.apply(1)
    assert all(torch.equal(a, b) for a, b in zip(before, module.parameters()))
    assert decay.log_scalars() == {'train/outlier_over': 0.0, 'train/outlier_max': 1.0}

    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=0.5))
    with torch.no_grad():
        module.a.weight[0, 0] = 100.0
    optimizer.param_groups[0]['lr'] = 0.0
    before = module.a.weight.detach().clone()
    decay.before_step()
    decay.apply(1)
    assert torch.equal(before, module.a.weight)


def test_warmup_scales_the_first_steps_and_lr_is_the_one_before_the_step():
    module = Stub()
    optimizer = optimizer_for(module, lr=0.01)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=1.0, warmup=10))
    r0 = decay.r0[decay.names.index('a.weight')]
    with torch.no_grad():
        module.a.weight[0, 0] = 20 * r0
    decay.before_step()
    optimizer.param_groups[0]['lr'] = 1.0  # what a scheduler step inside train_batch leaves behind
    before = module.a.weight[0, 0].item()
    decay.apply(1)
    # step 1: rate = 6 * 1/10, lr = 0.01 as recorded
    assert module.a.weight[0, 0].item() == pytest.approx(before - 0.01 * 0.6 * (before - r0), rel=1e-5)


def test_cap_anneal_changes_which_elements_are_over():
    module = Stub()
    optimizer = optimizer_for(module, lr=0.01)
    spec = make_spec(cap=2.0, cap_start=50.0, cap_anneal=100)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, spec)
    r0 = decay.r0[decay.names.index('a.weight')]
    with torch.no_grad():
        module.a.weight[0, 0] = 10 * r0
    decay.before_step()
    decay.apply(1)  # cap ~50 r0: nothing over
    assert decay.log_scalars()['train/outlier_over'] == 0
    decay.before_step()
    decay.apply(101)  # cap 2 r0
    assert decay.log_scalars()['train/outlier_over'] == 1


def test_round_to_bfloat16_is_unbiased_and_keeps_exact_values():
    generator = torch.Generator().manual_seed(0)
    exact = torch.tensor([1.0, -2.5, 0.0, 3.140625, 1e-3]).to(torch.bfloat16).float()
    assert torch.equal(round_to_bfloat16(exact, generator).float(), exact)
    x = torch.full((200_000,), 1.0 + 2 ** -10)  # a quarter of the way between two bf16 values
    out = round_to_bfloat16(x, generator).float()
    assert set(out.unique().tolist()) == {1.0, 1.0 + 2 ** -7}
    assert out.mean().item() == pytest.approx(x[0].item(), abs=2e-4)


def test_bf16_update_moves_by_the_expected_amount_though_smaller_than_one_step():
    torch.manual_seed(0)
    module = Stub().to(torch.bfloat16)
    optimizer = optimizer_for(module, lr=1e-3)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=1.0, seed=3))
    r0 = decay.r0[decay.names.index('a.weight')]
    with torch.no_grad():
        module.a.weight.fill_(1.0 * r0)
        module.a.weight[:, :4] = 16 * r0
    start = module.a.weight.detach().float().clone()
    steps = 400
    for step in range(1, steps + 1):
        decay.before_step()
        decay.apply(step)
    moved = (start - module.a.weight.float())[:, :4].mean().item()
    value = start[0, 0].item()
    # Closed form of the fp32 recursion e <- e (1 - lr rate) on the excess over r0.
    excess0 = value - r0
    wanted = excess0 * (1 - (1 - 1e-3 * 6.0) ** steps)
    assert moved == pytest.approx(wanted, rel=0.15)
    # the elements below the cap never moved
    assert torch.equal(module.a.weight.float()[:, 4:], start[:, 4:])


def test_same_seed_and_step_give_identical_weights_on_every_replica():
    outcomes = []
    for run in range(2):
        torch.manual_seed(0)
        module = Stub().to(torch.bfloat16)
        optimizer = optimizer_for(module, lr=1e-3)
        decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=1.0, seed=5))
        with torch.no_grad():
            module.a.weight[:, :4] *= 40
        # A replica's global generator is in a state of its own (data, noise, dropout draw from it):
        # the decay must not depend on it.
        torch.manual_seed(100 + run)
        for step in range(1, 6):
            torch.rand(7 + run)
            decay.before_step()
            decay.apply(step)
        outcomes.append(module.a.weight.detach().clone())
    assert torch.equal(*outcomes)


def test_r0_is_taken_at_build_time():
    module = Stub()
    optimizer = optimizer_for(module)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec())
    r0 = list(decay.r0)
    with torch.no_grad():
        for p in module.parameters():
            p.mul_(3.0)
    assert decay.r0 == r0


@pytest.mark.parametrize('optimizer, steps_at_group_lr', [
    ({'type': 'adamw'}, True), ({'type': 'AdamW_Optimi'}, True), ({'type': 'genericoptim'}, True),
    ({'type': 'offload'}, True),
    ({'type': 'genericoptim', 'automagic': True}, False), ({'type': 'prodigy'}, False),
    ({'type': 'automagic'}, False), ({'type': 'Lion'}, False), ({'type': 'SOAP'}, False),
])
def test_an_optimizer_whose_step_is_not_its_group_lr_needs_an_lr_of_the_decay_own(optimizer, steps_at_group_lr):
    config = {'outlier_decay': {'rate': 6.0}, 'optimizer': optimizer}
    if steps_at_group_lr:
        apply_outlier_decay_config_defaults(config)
    else:
        with pytest.raises(ValueError, match='needs lr'):
            apply_outlier_decay_config_defaults(config)
    # With an lr of its own every one of them is accepted, and so is rate 0 (off) without one.
    apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 6.0, 'lr': 1e-5}, 'optimizer': optimizer})
    apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 0}, 'optimizer': optimizer})


@pytest.mark.parametrize('name', sorted(SECOND_COPY_OPTIMIZERS))
def test_an_optimizer_that_keeps_a_second_copy_of_the_weights_is_refused_even_with_an_lr(name):
    config = {'outlier_decay': {'rate': 6.0, 'lr': 1e-5}, 'optimizer': {'type': name}}
    with pytest.raises(ValueError, match='cannot work with optimizer'):
        apply_outlier_decay_config_defaults(config)
    apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 0}, 'optimizer': {'type': name}})


def test_the_refused_names_are_real_pytorch_optimizer_classes():
    pytorch_optimizer = pytest.importorskip('pytorch_optimizer')
    known = {name.lower() for name in dir(pytorch_optimizer)}
    assert set(SECOND_COPY_OPTIMIZERS) <= known


@pytest.mark.parametrize('value', [0, -1e-5, True, 'x', float('nan'), float('inf')])
def test_lr_must_be_a_positive_number(value):
    with pytest.raises(ValueError, match='lr'):
        apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 6.0, 'lr': value}})


def test_gradient_release_is_refused():
    config = {'outlier_decay': {'rate': 6.0}, 'optimizer': {'type': 'adamw', 'gradient_release': True}}
    with pytest.raises(ValueError, match='gradient_release'):
        apply_outlier_decay_config_defaults(config)


def test_with_an_lr_the_step_size_is_that_lr_shaped_by_the_schedule():
    module = Stub()
    optimizer = optimizer_for(module, lr=1.0)
    group = optimizer.param_groups[0]
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(lr=1e-3))
    # The scheduler's start is initial_lr; the group has fallen to a quarter of it.
    group['initial_lr'], group['lr'] = 1.0, 0.25
    decay.before_step()
    assert decay._lrs[0] == pytest.approx(0.25e-3)
    # Without initial_lr the highest rate seen so far stands for the start.
    del group['initial_lr']
    group['lr'] = 0.5
    decay.before_step()
    assert decay._lrs[0] == pytest.approx(1e-3)
    group['lr'] = 0.125
    decay.before_step()
    assert decay._lrs[0] == pytest.approx(0.25e-3)
    group['lr'] = 0.0
    decay.before_step()
    assert decay._lrs[0] == 0.0
    # The startup check is on the lr of the decay, not on the group's.
    assert decay.peak_pull() == pytest.approx(1e-3 * 6.0)


def test_with_an_lr_each_group_keeps_its_ratio_to_the_largest_starting_rate():
    """A part trained at a tenth of the base rate decays at a tenth, as with AdamW; a frozen part does not
    decay; a group the decay does not touch does not set the base."""
    module = Stub()
    optimizer = torch.optim.SGD([
        {'params': [module.a.weight], 'lr': 1.0, 'initial_lr': 1.0},
        {'params': [module.b.weight], 'lr': 0.05, 'initial_lr': 0.1},
        {'params': [module.norm.weight], 'lr': 0.0, 'initial_lr': 0.0},
        {'params': [module.a.bias], 'lr': 50.0, 'initial_lr': 50.0},
    ])
    named = [('a.weight', module.a.weight), ('b.weight', module.b.weight), ('norm.weight', module.norm.weight)]
    decay = OutlierDecay.build(named, optimizer, make_spec(lr=1e-3, include_1d=True))
    decay.before_step()
    steps = dict(zip(decay.names, decay._lrs))
    assert steps['a.weight'] == pytest.approx(1e-3)
    # the second group is at half of its own start, which is a tenth of the base
    assert steps['b.weight'] == pytest.approx(0.05e-3)
    assert steps['norm.weight'] == 0.0


def test_the_cpu_offload_optimizers_host_and_device_weights_hold_the_same_bits_after_the_decay():
    module = Stub().bfloat16()
    optimizer = FakeOffloadOptimizer(module, lr=0.01)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=1.0, rate=20.0, include_1d=True))
    with torch.no_grad():
        for p in module.parameters():
            if p in optimizer.param_d2h_map:
                p.mul_(1.0 + torch.rand_like(p))
                optimizer.param_d2h_map[p].copy_(p)
    decay.before_step()
    decay.apply(1)
    for p, host in optimizer.param_d2h_map.items():
        assert torch.equal(p, host)
    assert decay.log_scalars()['train/outlier_over'] > 0


def test_prodigy_decays_at_the_lr_of_the_decay_and_not_at_its_own_multiplier():
    pytorch_optimizer = pytest.importorskip('pytorch_optimizer')
    module = Stub()
    optimizer = pytorch_optimizer.Prodigy(list(module.parameters()), lr=1.0)
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(rate=10.0, cap=1.0, lr=1e-3))
    r0 = decay.r0[decay.names.index('a.weight')]
    with torch.no_grad():
        module.a.weight[0, 0] = 20 * r0
    before = module.a.weight[0, 0].item()
    decay.require_match()
    decay.before_step()
    decay.apply(1)
    assert module.a.weight[0, 0].item() == pytest.approx(before - 1e-3 * 10.0 * (before - r0), rel=1e-5)


class FakeOffloadOptimizer:
    """The part of torchao's CPUOffloadOptimizer the decay sees: groups that hold host copies of the large
    weights next to the device weights of the small ones, and param_d2h_map from device to host."""

    def __init__(self, module, lr):
        self.param_d2h_map = {}
        host_groups, small = [], []
        for p in module.parameters():
            if p.numel() >= 32:
                host = p.detach().clone()
                self.param_d2h_map[p] = host
                host_groups.append({'params': [host], 'lr': lr})
            else:
                small.append(p)
        self.param_groups = [{'params': small, 'lr': lr}] + host_groups if small else host_groups


def test_the_cpu_offload_optimizers_host_copy_is_decayed_along_with_the_weight():
    module = Stub()
    optimizer = FakeOffloadOptimizer(module, lr=0.01)
    assert module.a.weight in optimizer.param_d2h_map and module.a.bias not in optimizer.param_d2h_map
    decay = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(cap=10.0, include_1d=True))
    # norm.bias starts at zero, which has no scale to measure outliers against.
    assert set(decay.names) == {'a.weight', 'a.bias', 'b.weight', 'norm.weight'}
    i = decay.names.index('a.weight')
    r0 = decay.r0[i]
    host = optimizer.param_d2h_map[module.a.weight]
    with torch.no_grad():
        module.a.weight[0, 0] = 20 * r0
        host[0, 0] = 20 * r0
    before = module.a.weight[0, 0].item()
    decay.before_step()
    decay.apply(1)
    expected = before - 0.01 * 6.0 * (before - 10 * r0)
    assert module.a.weight[0, 0].item() == pytest.approx(expected, rel=1e-5)
    assert host[0, 0].item() == pytest.approx(expected, rel=1e-5)
    assert decay.log_scalars()['train/outlier_over'] == 1.0


def test_a_weight_the_cpu_offload_optimizer_does_not_step_is_not_decayed():
    module = Stub()
    optimizer = FakeOffloadOptimizer(module, lr=0.01)
    other = nn.Linear(8, 8)
    decay = OutlierDecay.build(list(module.named_parameters()) + [('other.weight', other.weight)], optimizer, make_spec())
    assert 'other.weight' not in decay.names


def test_lr_times_rate_above_one_is_refused():
    module = Stub()
    decay = OutlierDecay.build(module.named_parameters(), optimizer_for(module, lr=0.5), make_spec(rate=6.0))
    with torch.no_grad():
        module.a.weight[0, 0] = 1000.0 * decay.r0[0]
    decay.before_step()
    with pytest.raises(ValueError, match='flips the sign'):
        decay.apply(1)


def test_a_rate_that_would_pass_the_cap_is_caught_at_startup_on_every_rank():
    module = Stub()
    decay = OutlierDecay.build(module.named_parameters(), optimizer_for(module, lr=0.5), make_spec(rate=6.0))
    assert decay.peak_pull() == pytest.approx(3.0)
    with pytest.raises(ValueError, match='above 1 for some tensor'):
        decay.require_match()
    OutlierDecay.build(module.named_parameters(), optimizer_for(module, lr=1e-3), make_spec(rate=6.0)).require_match()


def test_reset_metrics_forgets_the_last_steps_numbers():
    module = Stub()
    decay = OutlierDecay.build(module.named_parameters(), optimizer_for(module, lr=1e-3), make_spec(rate=6.0))
    with torch.no_grad():
        module.a.weight[0, 0] = 1000.0 * decay.r0[0]
    decay.before_step()
    decay.apply(1)
    assert decay.log_scalars()['train/outlier_max'] > 1.0
    decay.reset_metrics()
    assert decay.log_scalars() == {'train/outlier_over': 0.0, 'train/outlier_max': 1.0}


def test_require_match_raises_when_nothing_is_targeted():
    module = Stub()
    optimizer = optimizer_for(module)
    OutlierDecay.build(module.named_parameters(), optimizer, make_spec()).require_match()
    nothing = OutlierDecay.build(module.named_parameters(), optimizer, make_spec(targets='no_such_name'))
    assert len(nothing) == 0
    with pytest.raises(ValueError, match='no trained parameter of any pipeline stage'):
        nothing.require_match()



@pytest.mark.parametrize('fields', [{'cap_start': 64.0}, {'cap_anneal': 100}])
def test_cap_start_and_cap_anneal_come_as_a_pair(fields):
    with pytest.raises(ValueError, match='cap_start and .*cap_anneal'):
        apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 1.0, **fields}})
    apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 1.0, 'cap_start': 64.0, 'cap_anneal': 100}})
