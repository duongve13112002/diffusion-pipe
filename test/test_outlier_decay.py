"""utils/outlier_decay.py: config validation, the two schedules, target selection, the update
itself in fp32 and bf16, and the learning rate it uses."""

import math

import pytest
import torch
from torch import nn

from utils.outlier_decay import (OutlierDecay, apply_outlier_decay_config_defaults, cap_at, rate_at,
                                 round_to_bfloat16, targeted)


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


@pytest.mark.parametrize('optimizer, ok', [
    ({'type': 'adamw'}, True), ({'type': 'AdamW_Optimi'}, True), ({'type': 'genericoptim'}, True),
    ({'type': 'genericoptim', 'automagic': True}, False), ({'type': 'prodigy'}, False),
    ({'type': 'automagic'}, False), ({'type': 'offload'}, False), ({'type': 'Lion'}, False),
])
def test_only_optimizers_that_step_at_the_group_lr_are_accepted(optimizer, ok):
    config = {'outlier_decay': {'rate': 6.0}, 'optimizer': optimizer}
    if ok:
        apply_outlier_decay_config_defaults(config)
    else:
        with pytest.raises(ValueError, match='steps at its group'):
            apply_outlier_decay_config_defaults(config)
    # Rate 0 is off: no optimizer is refused.
    apply_outlier_decay_config_defaults({'outlier_decay': {'rate': 0}, 'optimizer': optimizer})


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
