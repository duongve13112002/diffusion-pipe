"""Tests for AdamW8bitKahan (optimizers/adamw_8bit.py).

The optimizer re-implements bitsandbytes' Optimizer2State.update_step to add Kahan
summation, so it depends on the exact shape of bitsandbytes' config dict. bitsandbytes 0.50
dropped 'percentile_clipping' and 'block_wise' from get_config(), which raised KeyError
mid-training. These tests drive update_step with both the old and the new config shape.

They run on CPU and stub out the bitsandbytes CUDA kernels, so they do not need a GPU and
do not need the real bitsandbytes update to be callable.
"""

import pytest
import torch

bitsandbytes = pytest.importorskip('bitsandbytes', reason='bitsandbytes is not installed')

from optimizers.adamw_8bit import AdamW8bitKahan


class RecordingFunctional:
    """Stands in for bitsandbytes.functional, recording which update path was taken."""

    def __init__(self, with_percentile_clipping=False):
        self.calls = []
        if with_percentile_clipping:
            self.percentile_clipping = self._percentile_clipping

    def _percentile_clipping(self, grad, gnorm_vec, step, percentile):
        self.calls.append(('percentile_clipping', percentile))
        return 0.0, 0.0, 0.5

    def optimizer_update_32bit(self, *args, **kwargs):
        # Positional layout is (optimizer_name, g, p, state1, beta1, eps, step, lr, state2,
        # beta2, beta3, alpha, weight_decay, gnorm_scale, unorm_vec), so gnorm_scale is 13.
        self.calls.append(('update_32bit', args[13]))

    def optimizer_update_8bit(self, *args, **kwargs):
        self.calls.append(('update_8bit', kwargs['gnorm_scale']))

    def optimizer_update_8bit_blockwise(self, *args, **kwargs):
        self.calls.append(('update_8bit_blockwise', kwargs['gnorm_scale']))


def make_optimizer(config, functional, state1_dtype=torch.float32):
    """Build an AdamW8bitKahan without running bitsandbytes' __init__.

    Only the attributes update_step touches are populated, so the test exercises our
    re-implemented step rather than bitsandbytes' parameter management.
    """
    optimizer = AdamW8bitKahan.__new__(AdamW8bitKahan)
    optimizer.stabilize = False
    optimizer.optimizer_name = 'adam'
    optimizer.get_config = lambda gindex, pindex, group: config

    p = torch.nn.Parameter(torch.zeros(4, 4))
    p.grad = torch.full((4, 4), 0.1)

    state = {
        'step': 0,
        'shift': torch.zeros(4, 4),
        'state1': torch.zeros(4, 4, dtype=state1_dtype),
        'state2': torch.ones(4, 4),
        'gnorm_vec': torch.zeros(100),
        'unorm_vec': torch.zeros(1),
        'qmap1': torch.zeros(256),
        'qmap2': torch.zeros(256),
        'absmax1': torch.zeros(1),
        'absmax2': torch.zeros(1),
        'max1': torch.zeros(1),
        'max2': torch.zeros(1),
        'new_max1': torch.zeros(1),
        'new_max2': torch.zeros(1),
    }
    optimizer.state = {p: state}
    return optimizer, p


def base_config():
    """The keys bitsandbytes still provides in every version we support."""
    return {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01,
        'lr': 1e-4,
        'alpha': 0.0,
        'max_unorm': 0.0,
        'skip_zeros': False,
    }


def run_step(monkeypatch, config, state1_dtype=torch.float32, with_percentile_clipping=False):
    functional = RecordingFunctional(with_percentile_clipping=with_percentile_clipping)
    monkeypatch.setattr('optimizers.adamw_8bit.F', functional)
    optimizer, p = make_optimizer(config, functional, state1_dtype=state1_dtype)
    optimizer.update_step(None, p, 0, 0)
    return functional, optimizer, p


def test_new_bitsandbytes_config_does_not_raise(monkeypatch):
    """bitsandbytes 0.50 config: no 'percentile_clipping', no 'block_wise'."""
    functional, _, _ = run_step(monkeypatch, base_config())
    assert functional.calls == [('update_32bit', 1.0)]


def test_new_bitsandbytes_config_uses_blockwise_for_8bit_state(monkeypatch):
    """Without 'block_wise' the 8-bit path must default to blockwise, like bitsandbytes 0.50."""
    functional, _, _ = run_step(monkeypatch, base_config(), state1_dtype=torch.uint8)
    assert functional.calls == [('update_8bit_blockwise', 1.0)]


def test_old_bitsandbytes_config_still_clips(monkeypatch):
    """bitsandbytes 0.49 config: percentile clipping is honored when the key is present."""
    config = base_config()
    config['percentile_clipping'] = 5
    config['block_wise'] = True
    functional, _, _ = run_step(monkeypatch, config, with_percentile_clipping=True)
    assert functional.calls == [('percentile_clipping', 5), ('update_32bit', 0.5)]


def test_old_bitsandbytes_config_can_still_select_non_blockwise(monkeypatch):
    """The non-blockwise branch stays reachable on the bitsandbytes versions that have it."""
    config = base_config()
    config['percentile_clipping'] = 100
    config['block_wise'] = False
    functional, _, _ = run_step(monkeypatch, config, state1_dtype=torch.uint8)
    assert functional.calls == [('update_8bit', 1.0)]


def test_kahan_shift_is_applied(monkeypatch):
    """The Kahan compensation still runs: p moves by shift and shift keeps the residual."""
    config = base_config()
    functional = RecordingFunctional()

    def update_32bit(*args, **kwargs):
        # Emulate the kernel writing the update into the shift buffer.
        args[2].fill_(0.25)
        functional.calls.append(('update_32bit', args[13]))

    functional.optimizer_update_32bit = update_32bit
    monkeypatch.setattr('optimizers.adamw_8bit.F', functional)
    optimizer, p = make_optimizer(config, functional)

    optimizer.update_step(None, p, 0, 0)

    assert torch.allclose(p.data, torch.full((4, 4), 0.25))
    assert torch.allclose(optimizer.state[p]['shift'], torch.zeros(4, 4))
