"""utils/lora_plus.py: the config keys, the group split, and the split inside an optimizer with the
weight-decay split of train.py and a schedule (CPU)."""

import pytest
import torch

from utils.lora_plus import (NAME, apply_lora_plus_config_defaults, is_lora_b, lora_plus_active,
                             split_lora_plus_groups)
from utils.lr_schedule import create_lr_scheduler


def _param(name, *shape):
    p = torch.nn.Parameter(torch.zeros(*shape))
    p.original_name = name
    return p


def _pair(layer):
    return (_param(f'layers.{layer}.q.lora_A.default.weight', 4, 8),
            _param(f'layers.{layer}.q.lora_B.default.weight', 8, 4))


def test_config_defaults_and_refusals():
    config = {'adapter': {'type': 'lora', 'rank': 4}, 'optimizer': {'type': 'adamw'}}
    apply_lora_plus_config_defaults(config)
    assert config['adapter'][NAME] == 1.0 and not lora_plus_active(config['adapter'])
    assert not lora_plus_active(None)

    ok = {'adapter': {'type': 'lora', NAME: 16}, 'optimizer': {'type': 'adamw', 'lr': 1e-4}}
    apply_lora_plus_config_defaults(ok)
    assert lora_plus_active(ok['adapter'])

    for bad in (0.5, 0, -2, True, '16', float('nan'), float('inf')):
        with pytest.raises(ValueError, match=NAME):
            apply_lora_plus_config_defaults({'adapter': {'type': 'lora', NAME: bad}})
    with pytest.raises(ValueError, match="'lora'"):
        apply_lora_plus_config_defaults({'adapter': {'type': 'lokr', NAME: 4}})
    with pytest.raises(ValueError, match='gradient_release'):
        apply_lora_plus_config_defaults({'adapter': {'type': 'lora', NAME: 4}, 'optimizer': {'gradient_release': True}})
    with pytest.raises(ValueError, match='force_constant_lr'):
        apply_lora_plus_config_defaults({'adapter': {'type': 'lora', NAME: 4}, 'force_constant_lr': 1e-4})
    # Off next to the things it would conflict with is fine.
    apply_lora_plus_config_defaults({'adapter': {'type': 'lokr'}, 'optimizer': {'gradient_release': True},
                                     'force_constant_lr': 1e-4})


def test_is_lora_b_matches_only_the_b_factor():
    assert is_lora_b('layers.0.q.lora_B.default.weight')
    assert not is_lora_b('layers.0.q.lora_A.default.weight')
    assert not is_lora_b('layers.0.q.lora_magnitude_vector.default.weight')
    assert not is_lora_b('layers.0.q.base_layer.weight')


def test_b_tensors_get_a_group_of_their_own_at_ratio_times_the_lr():
    a0, b0 = _pair(0)
    a1, b1 = _pair(1)
    out = split_lora_plus_groups([{'params': [a0, b0, a1, b1]}], 16, 1e-4)
    assert [g['params'] for g in out] == [[a0, a1], [b0, b1]]
    assert 'lr' not in out[0] and out[1]['lr'] == pytest.approx(1.6e-3)


def test_a_groups_own_lr_name_and_options_carry_over():
    a0, b0 = _pair(0)
    a1, b1 = _pair(1)
    other = _param('norm.weight', 8)
    groups = [{'params': [a0, b0], 'lr': 2e-5, 'name': 'early', 'weight_decay': 0.5},
              {'params': [other], 'lr': 3e-5, 'name': 'norm'},
              {'params': [b1, a1], 'lr': 4e-5}]
    out = split_lora_plus_groups(groups, 4, 1e-4)
    assert [(g.get('name'), g['lr'], g['params']) for g in out] == [
        ('early', 2e-5, [a0]), ('early+B', pytest.approx(8e-5), [b0]),
        ('norm', 3e-5, [other]),
        (None, 4e-5, [a1]), (None, pytest.approx(1.6e-4), [b1])]
    assert out[1]['weight_decay'] == 0.5
    # The input groups are not edited.
    assert groups[0]['params'] == [a0, b0] and groups[0]['lr'] == 2e-5


def test_a_group_of_only_b_tensors_has_no_empty_partner():
    _, b0 = _pair(0)
    out = split_lora_plus_groups([{'params': [b0], 'lr': 1e-5}], 2, 1e-4)
    assert len(out) == 1 and out[0]['lr'] == pytest.approx(2e-5)


def test_no_lora_b_is_an_error_not_a_silent_no_op():
    a0, _ = _pair(0)
    with pytest.raises(ValueError, match='found no lora_B'):
        split_lora_plus_groups([{'params': [a0]}], 16, 1e-4)


def test_inside_an_optimizer_with_the_weight_decay_split_and_a_schedule():
    a0, b0 = _pair(0)
    bias = _param('layers.0.q.base_layer.bias', 8)
    groups = split_lora_plus_groups([{'params': [a0, b0, bias]}], 8, 1e-3)
    # train.py's weight decay split: 1D tensors without decay, copying the group (its lr included).
    split = []
    for pg in groups:
        params = pg.pop('params')
        wd = [p for p in params if p.ndim != 1]
        no_wd = [p for p in params if p.ndim == 1]
        no_wd_group = pg.copy()
        pg['params'] = wd
        no_wd_group['params'] = no_wd
        no_wd_group['weight_decay'] = 0
        split += [g for g in (pg, no_wd_group) if g['params']]
    optimizer = torch.optim.AdamW(split, lr=1e-3, weight_decay=0.01)
    by_param = {id(p): g for g in optimizer.param_groups for p in g['params']}
    assert by_param[id(a0)]['lr'] == pytest.approx(1e-3) and by_param[id(b0)]['lr'] == pytest.approx(8e-3)
    assert by_param[id(bias)]['lr'] == pytest.approx(1e-3) and by_param[id(bias)]['weight_decay'] == 0
    assert by_param[id(b0)]['weight_decay'] == 0.01

    scheduler = create_lr_scheduler(optimizer, 'cosine', total_steps=100, warmup_steps=10, min_ratio=0.1)
    seen = set()
    for _ in range(100):
        a_lr, b_lr, bias_lr = (by_param[id(p)]['lr'] for p in (a0, b0, bias))
        assert b_lr == pytest.approx(8 * a_lr) and bias_lr == pytest.approx(a_lr)
        seen.add(round(a_lr, 12))
        optimizer.step()
        scheduler.step()
    # The schedule did move (warmup, then decay), and every group followed it relative to its own rate.
    assert len(seen) > 50 and max(seen) == pytest.approx(1e-3, rel=1e-6)


def test_lora_plus_refuses_adaptive_optimizers_and_a_missing_lr():
    for optimizer in ({'type': 'prodigy', 'lr': 1}, {'type': 'genericoptim', 'lr': 1e-4, 'automagic': True},
                      {'type': 'adamw'}):
        config = {'adapter': {'type': 'lora', 'lora_plus_ratio': 8}, 'optimizer': optimizer}
        with pytest.raises(ValueError):
            apply_lora_plus_config_defaults(config)
    apply_lora_plus_config_defaults({'adapter': {'type': 'lora', 'lora_plus_ratio': 8},
                                     'optimizer': {'type': 'adamw', 'lr': 1e-4}})
