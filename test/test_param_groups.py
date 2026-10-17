"""utils/param_groups.py and the gradient-release wrapper's param groups (CPU)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimizers.gradient_release import GradientReleaseOptimizerWrapper  # noqa: E402
from utils.param_groups import check_resumed_param_groups, group_names, split_2d_param_groups  # noqa: E402


def test_every_2d_group_gets_the_projection_settings():
    groups = [{'params': [torch.zeros(4, 4), torch.zeros(4)], 'lr': 1.0},
              {'params': [torch.zeros(3, 3)], 'lr': 2.0},
              {'params': [torch.zeros(2, 2), torch.zeros(2)], 'lr': 3.0}]
    kwargs = {'rank': 8, 'proj_type': 'std', 'update_proj_gap': 50, 'betas': (0.9, 0.99)}
    out = split_2d_param_groups(groups, kwargs)
    two_d = [g for g in out if g['params'] and g['params'][0].ndim == 2]
    assert [g['lr'] for g in two_d] == [1.0, 2.0, 3.0]
    assert all((g['rank'], g['proj_type'], g['update_proj_gap']) == (8, 'std', 50) for g in two_d)
    assert all('rank' not in g for g in out if g not in two_d)
    assert kwargs == {'betas': (0.9, 0.99)}


def test_group_names_must_match_on_resume():
    check_resumed_param_groups([None, None], [None, None])
    check_resumed_param_groups(['a #1', 'b #2'], ['a #1', 'b #2'])
    with pytest.raises(RuntimeError, match='--reset_optimizer'):
        check_resumed_param_groups(['a #1'], ['a #2'])
    assert group_names([{'params': [], 'name': 'x'}, {'params': []}]) == ['x', None]


def test_the_gradient_release_wrapper_takes_back_its_groups():
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    wrapper = GradientReleaseOptimizerWrapper([torch.optim.SGD([p], lr=0.1 * (i + 1)) for i, p in enumerate(params)])
    saved = wrapper.param_groups.copy()
    for opt in wrapper.optimizers:
        opt.param_groups = [dict(g, lr=9.0) for g in opt.param_groups]
    wrapper.param_groups = saved
    assert [g['lr'] for g in wrapper.param_groups] == pytest.approx([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match='one param group per wrapped optimizer group'):
        wrapper.param_groups = saved[:2]
