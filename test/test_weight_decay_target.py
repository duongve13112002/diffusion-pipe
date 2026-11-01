"""utils/weight_decay_target.py: config validation, which tensors are taken over, the update in fp32
and bf16, the decay the optimizer must not apply twice, and a resume bringing it back."""

import math

import pytest
import torch
from torch import nn

from utils.weight_decay_target import (SAVED_DECAY, WeightDecayTarget, apply_weight_decay_target_config_defaults,
                                       restore_optimizer_decay, weight_decay_target_active)


def make_config(wd=0.1, **overrides):
    config = {'optimizer': {'type': 'adamw', 'lr': 1e-3, 'weight_decay': wd}, 'weight_decay_target': 'start', **overrides}
    apply_weight_decay_target_config_defaults(config)
    return config


class Stub(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.a = nn.Linear(width, width, bias=True)
        self.norm = nn.LayerNorm(width)


def split_optimizer(module, lr=1e-2, wd=0.1, optimizer_class=torch.optim.AdamW):
    """train.py's split: tensors with 2+ dimensions decay, the rest do not."""
    decayed = [p for p in module.parameters() if p.ndim > 1]
    plain = [p for p in module.parameters() if p.ndim <= 1]
    return optimizer_class([{'params': decayed, 'lr': lr, 'weight_decay': wd}, {'params': plain, 'lr': lr, 'weight_decay': 0.0}])


def named(module):
    for name, p in module.named_parameters():
        p.original_name = name
        yield name, p


def test_config_defaults():
    config = {}
    apply_weight_decay_target_config_defaults(config)
    assert config == {'weight_decay_target': 'zero', 'weight_decay_to_one': ''} and not weight_decay_target_active(config)
    assert weight_decay_target_active(make_config())
    assert weight_decay_target_active(make_config(weight_decay_target='zero', weight_decay_to_one='norm'))


@pytest.mark.parametrize('overrides', [
    {'weight_decay_target': 'one'}, {'weight_decay_to_one': 3}, {'weight_decay_to_one': '('},
    {'adapter': {'type': 'lora'}}, {'optimizer': {'type': 'adamw', 'weight_decay': 0.0}},
    {'optimizer': {'type': 'adamw'}}, {'optimizer': {'type': 'prodigy', 'weight_decay': 0.1}},
    {'optimizer': {'type': 'sgd', 'weight_decay': 0.1}},
    {'optimizer': {'type': 'adamw', 'weight_decay': 0.1, 'gradient_release': True}},
    {'blocks_to_swap': 4}, {'optimizer': {'type': 'stableadamw', 'weight_decay': 0.1}},
    {'optimizer': {'type': 'adamw_optimi', 'weight_decay': 0.1, 'decouple_lr': True}},
])
def test_config_refusals(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_off_changes_nothing_and_asks_nothing_of_the_optimizer():
    config = {'optimizer': {'type': 'prodigy'}, 'adapter': {'type': 'lora'}}
    apply_weight_decay_target_config_defaults(config)
    assert config['weight_decay_target'] == 'zero'


def test_build_takes_over_the_decayed_tensors_and_switches_the_optimizers_decay_off():
    module = Stub()
    optimizer = split_optimizer(module)
    decay = WeightDecayTarget.build(named(module), optimizer, make_config())
    assert decay.names == ['a.weight']
    assert optimizer.param_groups[0]['weight_decay'] == 0.0
    assert torch.equal(decay.anchors[0], module.a.weight.detach())
    assert decay.anchors[0] is not module.a.weight.detach()
    assert len(decay) == 1 and 'toward the launch weights' in decay.describe()


def test_to_one_takes_named_undecayed_tensors_and_refuses_decayed_ones():
    module = Stub()
    decay = WeightDecayTarget.build(named(module), split_optimizer(module),
                                    make_config(weight_decay_target='zero', weight_decay_to_one=r'^norm\.weight$'))
    assert decay.names == ['norm.weight'] and decay.anchors == [None]
    with pytest.raises(ValueError, match='already decays'):
        WeightDecayTarget.build(named(module), split_optimizer(module), make_config(weight_decay_to_one=r'a\.weight'))


def test_require_match():
    module = Stub()
    decay = WeightDecayTarget.build(named(module), split_optimizer(module),
                                    make_config(weight_decay_target='zero', weight_decay_to_one='no_such_name'))
    with pytest.raises(ValueError, match='is taken over'):
        decay.require_match()
    WeightDecayTarget.build(named(module), split_optimizer(module), make_config()).require_match()


def test_require_match_catches_a_regex_that_matches_nothing_even_with_target_start():
    module = Stub()
    decay = WeightDecayTarget.build(named(module), split_optimizer(module),
                                    make_config(weight_decay_to_one='no_such_name'))
    assert len(decay) > 0
    with pytest.raises(ValueError, match='matches no trained parameter'):
        decay.require_match()
    WeightDecayTarget.build(named(module), split_optimizer(module),
                            make_config(weight_decay_to_one=r'^norm\.weight$')).require_match()


def run(decay, steps, lr_set=None):
    for step in range(1, steps + 1):
        decay.before_step()
        decay.apply(step)


def test_fp32_update_matches_the_formula_toward_the_launch_weights_and_toward_one():
    torch.manual_seed(0)
    module = Stub()
    optimizer = split_optimizer(module, lr=1e-2, wd=0.1)
    decay = WeightDecayTarget.build(named(module), optimizer,
                                    make_config(weight_decay_to_one=r'^norm\.weight$'))
    start = module.a.weight.detach().clone()
    with torch.no_grad():
        module.a.weight.add_(0.5)
        module.norm.weight.fill_(3.0)
    moved = module.a.weight.detach().clone()
    decay.before_step()
    decay.apply(1)
    k = 1e-2 * 0.1
    torch.testing.assert_close(module.a.weight, moved - k * (moved - start))
    torch.testing.assert_close(module.norm.weight, torch.full((8,), 3.0 - k * 2.0))
    assert torch.equal(module.a.bias, module.a.bias) and decay.names == ['a.weight', 'norm.weight']
    run(decay, 2000)
    # Toward the launch weights, not toward 0: the distance to the start shrinks, the weights stay.
    assert float((module.a.weight - start).abs().max().detach()) == pytest.approx(0.5 * (1 - k) ** 2000, rel=1e-3)
    assert float(module.a.weight.abs().mean()) > 0.1 * float(start.abs().mean())
    assert float((module.norm.weight - 1.0).abs().max().detach()) == pytest.approx(2.0 * (1 - k) ** 2000, rel=1e-3)


def test_equals_the_optimizers_own_decay_when_the_anchor_is_zero():
    torch.manual_seed(1)
    reference = Stub()
    ours = Stub()
    ours.load_state_dict(reference.state_dict())
    ref_opt = split_optimizer(reference, lr=1e-3, wd=0.05)
    our_opt = split_optimizer(ours, lr=1e-3, wd=0.05)
    decay = WeightDecayTarget.build(named(ours), our_opt, make_config(wd=0.05))
    decay.anchors = [torch.zeros_like(a) for a in decay.anchors]
    for _ in range(5):
        grads = [torch.randn_like(p) for p in reference.parameters()]
        for p, q, g in zip(reference.parameters(), ours.parameters(), grads):
            p.grad, q.grad = g.clone(), g.clone()
        ref_opt.step()
        decay.before_step()
        our_opt.step()
        decay.apply(1)
    for p, q in zip(reference.parameters(), ours.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-6)


def test_bf16_update_moves_by_the_expected_amount_though_smaller_than_one_step():
    torch.manual_seed(0)
    module = Stub(width=64).to(torch.bfloat16)
    optimizer = split_optimizer(module, lr=1e-3, wd=0.01)
    decay = WeightDecayTarget.build(named(module), optimizer, make_config(wd=0.01))
    start = module.a.weight.detach().float().clone()
    with torch.no_grad():
        module.a.weight.add_(1.0)
    far = module.a.weight.detach().float().clone()
    steps = 400
    run(decay, steps)
    wanted = (far - start) * (1 - (1 - 1e-3 * 0.01) ** steps)
    got = far - module.a.weight.detach().float()
    assert abs(float(got.mean()) - float(wanted.mean())) < 0.1 * float(wanted.mean())
    assert module.a.weight.dtype == torch.bfloat16


def test_replicas_agree_whatever_their_global_generator_holds():
    outcomes = []
    for run_index in range(2):
        torch.manual_seed(0)
        module = Stub().to(torch.bfloat16)
        decay = WeightDecayTarget.build(named(module), split_optimizer(module, lr=1e-2, wd=0.1), make_config())
        with torch.no_grad():
            module.a.weight.mul_(3)
        torch.manual_seed(50 + run_index)
        for step in range(1, 6):
            torch.rand(3 + run_index)
            decay.before_step()
            decay.apply(step)
        outcomes.append(module.a.weight.detach().clone())
    assert torch.equal(*outcomes)


def test_a_step_with_lr_times_decay_above_one_is_refused():
    module = Stub()
    decay = WeightDecayTarget.build(named(module), split_optimizer(module, lr=20.0, wd=0.1), make_config())
    decay.before_step()
    with pytest.raises(ValueError, match='flips the sign'):
        decay.apply(1)


def test_zero_learning_rate_changes_nothing():
    module = Stub()
    decay = WeightDecayTarget.build(named(module), split_optimizer(module, lr=0.0), make_config())
    with torch.no_grad():
        module.a.weight.add_(1.0)
    before = module.a.weight.detach().clone()
    decay.before_step()
    decay.apply(1)
    assert torch.equal(module.a.weight, before)


def test_a_resume_brings_the_optimizers_decay_back_and_claim_groups_removes_it_again():
    module = Stub()
    optimizer = split_optimizer(module, wd=0.1)
    saved = optimizer.state_dict()
    decay = WeightDecayTarget.build(named(module), optimizer, make_config())
    assert optimizer.param_groups[0]['weight_decay'] == 0.0
    optimizer.load_state_dict(saved)
    assert optimizer.param_groups[0]['weight_decay'] == 0.1, 'the premise: loading restores the saved decay'
    decay.claim_groups()
    assert optimizer.param_groups[0]['weight_decay'] == 0.0


def test_a_checkpoint_saved_with_the_target_on_gets_its_decay_back_when_a_run_turns_it_off():
    module = Stub()
    optimizer = split_optimizer(module, wd=0.1)
    WeightDecayTarget.build(named(module), optimizer, make_config())
    saved = optimizer.state_dict()
    assert saved['param_groups'][0]['weight_decay'] == 0.0 and saved['param_groups'][0][SAVED_DECAY] == 0.1
    fresh = split_optimizer(Stub(), wd=0.05)
    fresh.load_state_dict(saved)
    assert fresh.param_groups[0]['weight_decay'] == 0.0
    assert restore_optimizer_decay(fresh) == 1
    assert fresh.param_groups[0]['weight_decay'] == 0.1 and SAVED_DECAY not in fresh.param_groups[0]
    assert restore_optimizer_decay(fresh) == 0


def test_a_learning_rate_that_would_pull_past_the_target_is_caught_at_startup_from_the_schedules_peak():
    module = Stub()
    optimizer = split_optimizer(module, lr=20.0, wd=0.1)
    decay = WeightDecayTarget.build(named(module), optimizer, make_config())
    assert decay.peak_pull() == pytest.approx(2.0)
    with pytest.raises(ValueError, match='above 1 for some tensor'):
        decay.require_match()
    # A warmup starts the group low; initial_lr is what the schedule reaches.
    low = split_optimizer(module, lr=1e-3, wd=0.1)
    for group in low.param_groups:
        group['initial_lr'] = 20.0
    with pytest.raises(ValueError, match='above 1 for some tensor'):
        WeightDecayTarget.build(named(module), low, make_config()).require_match()


def test_unsupported_dtype_is_refused():
    module = Stub().to(torch.float16)
    with pytest.raises(ValueError, match='float32 and bfloat16'):
        WeightDecayTarget.build(named(module), split_optimizer(module), make_config())


@pytest.mark.parametrize('muon', [False, True], ids=['adam', 'muon'])
def test_genericoptims_own_decay_is_the_decay_this_takes_over(muon):
    """The reason genericoptim is on the list: its weight decay is exactly p -= lr * wd * p, Muon groups
    included, so switching it off and applying lr * wd * (p - a) in its place changes nothing else."""
    from optimizers.generic_optim import GenericOptim
    config = make_config(optimizer={'type': 'genericoptim', 'weight_decay': 0.1})

    def one_step(weight_decay):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(64, 64))
        start = p.detach().clone()
        p.grad = torch.randn(64, 64)
        GenericOptim([{'params': [p], 'weight_decay': weight_decay, 'muon': muon}], lr=1e-2, mpu=None,
                     muon=muon).step()
        return p.detach().clone(), start

    decayed, start = one_step(0.1)
    plain, _ = one_step(0.0)
    torch.testing.assert_close(decayed - plain, -1e-2 * 0.1 * start, rtol=0, atol=1e-6)
    assert config['optimizer']['type'] == 'genericoptim'
