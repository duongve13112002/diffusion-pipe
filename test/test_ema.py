"""utils/ema.py and its use by utils/saver.py: the average itself, its checkpoint round trip
(across a different pipeline split), and the ema/ save written beside every normal save."""

import types
from pathlib import Path

import pytest
import torch
from torch import nn

from utils import ema as ema_util
from utils import saver as saver_util
from utils.ema import ParameterEMA, apply_ema_config_defaults, load_ema_checkpoint, save_ema_checkpoint


def named(module, prefix=''):
    for name, p in module.named_parameters():
        p.original_name = prefix + name
    return list(module.named_parameters())


def test_config_defaults_and_validation():
    assert apply_ema_config_defaults({}) is False
    config = {'ema_decay': 0.999}
    assert apply_ema_config_defaults(config) is True
    assert config == {'ema_decay': 0.999, 'ema_update_every': 1, 'ema_start_step': 0, 'ema_device': 'cpu'}
    for bad in ({'ema_decay': 1.0}, {'ema_decay': 0}, {'ema_decay': True}, {'ema_decay': '0.9'},
                {'ema_decay': 0.9, 'ema_update_every': 0}, {'ema_decay': 0.9, 'ema_start_step': -1},
                {'ema_decay': 0.9, 'ema_device': 'cuda'}, {'ema_update_every': 10}):
        with pytest.raises(ValueError):
            apply_ema_config_defaults(dict(bad))


def test_update_is_the_exponential_average_of_the_weights():
    layer = nn.Linear(4, 3)
    ema = ParameterEMA(named(layer), decay=0.9)
    expected = {n: p.detach().clone() for n, p in layer.named_parameters()}
    for step in range(1, 6):
        with torch.no_grad():
            for p in layer.parameters():
                p.add_(torch.randn_like(p))
        ema.update(step)
        for n, p in layer.named_parameters():
            expected[n] = 0.9 * expected[n] + 0.1 * p.detach()
    for n in expected:
        torch.testing.assert_close(ema.shadow[n], expected[n])
    assert ema.num_updates == 5


def test_update_every_compounds_the_decay():
    """Every k steps with decay**k equals every step when the weights only move every k."""
    a, b = nn.Linear(3, 3), nn.Linear(3, 3)
    b.load_state_dict(a.state_dict())
    every = ParameterEMA(named(a), decay=0.9, update_every=3)
    each = ParameterEMA(named(b), decay=0.9)
    for step in range(1, 10):
        if step % 3 == 1:
            delta = [torch.randn_like(p) for p in a.parameters()]
            with torch.no_grad():
                for pa, pb, d in zip(a.parameters(), b.parameters(), delta):
                    pa.add_(d)
                    pb.add_(d)
        every.update(step)
        each.update(step)
    # The k-step EMA samples the weights once per window instead of every step, so it agrees
    # with the per-step one when the weights are constant within each window, as here.
    for key in each.shadow:
        torch.testing.assert_close(every.shadow[key], each.shadow[key])
    assert every.num_updates == 3


def test_start_step_follows_the_weights():
    layer = nn.Linear(2, 2)
    ema = ParameterEMA(named(layer), decay=0.99, start_step=3)
    for step in (1, 2, 3):
        with torch.no_grad():
            layer.weight.add_(1.0)
        ema.update(step)
        torch.testing.assert_close(ema.shadow['weight'], layer.weight.detach())
    assert ema.num_updates == 0


def test_only_trainable_parameters_get_a_shadow_and_keys_match_the_saver():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    params = named(model)
    model[0].weight.requires_grad_(False)
    model[1].weight.original_name = '1.lora_A.default.weight'
    ema = ParameterEMA(params, decay=0.9, is_adapter=True)
    assert set(ema.shadow) == {'0.bias', '1.lora_A.weight', '1.bias'}
    # A frozen weight's average is the weight itself.
    torch.testing.assert_close(ema.averaged('0.weight', model[0].weight), model[0].weight.detach())


def test_shadow_is_fp32_and_saved_in_the_parameter_dtype():
    layer = nn.Linear(4, 4).to(torch.bfloat16)
    ema = ParameterEMA(named(layer), decay=0.9999)
    assert all(t.dtype == torch.float32 for t in ema.shadow.values())
    before = ema.shadow['weight'].clone()
    with torch.no_grad():
        layer.weight.add_(1.0)
    ema.update(1)
    # 1e-4 of a step of 1: lost in bf16, kept in the fp32 shadow.
    torch.testing.assert_close(ema.shadow['weight'] - before, torch.full_like(before, 1e-4), rtol=1e-3, atol=1e-6)
    assert ema.averaged('weight', layer.weight).dtype == torch.bfloat16


def test_checkpoint_round_trip_across_a_different_pipeline_split(tmp_path):
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3))
    params = named(model)
    stage0 = ParameterEMA(params[:2], decay=0.5)
    stage1 = ParameterEMA(params[2:], decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    stage0.update(1)
    stage1.update(1)
    (tmp_path / 'global_step7').mkdir()
    save_ema_checkpoint(stage0, tmp_path, 'global_step7', 0)
    save_ema_checkpoint(stage1, tmp_path, 'global_step7', 1)

    # Resumed with a single stage: every shadow is found by name, whichever file it is in.
    fresh = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3))
    resumed = ParameterEMA(named(fresh), decay=0.5)
    found, missing = load_ema_checkpoint(resumed, tmp_path, 'global_step7')
    assert found == 2 and missing == []
    for key, value in {**stage0.shadow, **stage1.shadow}.items():
        torch.testing.assert_close(resumed.shadow[key], value)
    assert resumed.num_updates == 1

    # A checkpoint from before EMA was switched on: nothing to load, the weights are the start.
    (tmp_path / 'global_step3').mkdir()
    assert load_ema_checkpoint(resumed, tmp_path, 'global_step3') == (0, list(resumed.shadow))


class _Grid:
    def get_data_parallel_rank(self):
        return 0

    def get_pipe_parallel_rank(self):
        return 0


class _Engine:
    grid = _Grid()
    global_steps = 12

    def __init__(self):
        self.saved = []

    def save_checkpoint(self, save_dir, tag=None, **kwargs):
        Path(save_dir, tag).mkdir(parents=True)
        self.saved.append(tag)


class _Model:
    """Records what the saver hands to the model's own save functions."""

    def __init__(self):
        self.saves = {}

    def save_model(self, save_dir, state_dict):
        self.saves[Path(save_dir)] = state_dict

    save_adapter = save_model


@pytest.fixture
def no_barrier(monkeypatch):
    monkeypatch.setattr(saver_util, 'dist', types.SimpleNamespace(barrier=lambda: None))
    monkeypatch.setattr(saver_util, 'is_main_process', lambda: True)


@pytest.mark.parametrize('is_adapter', [False, True], ids=['full', 'adapter'])
def test_saver_writes_the_average_beside_the_weights(tmp_path, no_barrier, is_adapter):
    module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(torch.bfloat16)
    params = named(module)
    module[0].weight.requires_grad_(False)
    ema = ParameterEMA(params, decay=0.5, is_adapter=is_adapter)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(2.0)
    ema.update(1)

    config_file = tmp_path / 'train.toml'
    config_file.write_text('')
    model = _Model()
    engine = _Engine()
    saver = saver_util.Saver(types.SimpleNamespace(config=str(config_file)), {}, is_adapter, tmp_path / 'run', model,
                             types.SimpleNamespace(state_dict=lambda: {}), engine, module, ema=ema)
    saver.save_model('epoch1')
    raw, averaged = model.saves[tmp_path / 'run' / 'epoch1'], model.saves[tmp_path / 'run' / 'epoch1' / 'ema']
    assert set(raw) == set(averaged)
    for key, value in averaged.items():
        assert value.dtype == torch.bfloat16
        if key in ema.shadow:
            torch.testing.assert_close(value, ema.shadow[key].to(torch.bfloat16))
            assert not torch.equal(value, raw[key])
        else:
            torch.testing.assert_close(value, raw[key])     # frozen: the weight itself
    assert (tmp_path / 'run' / 'epoch1' / 'ema' / 'train.toml').exists()
    assert ('0.weight' in raw) == (not is_adapter)

    saver.save_checkpoint(5, 40)
    assert engine.saved == ['global_step12']
    assert ema_util.checkpoint_file(tmp_path / 'run', 'global_step12', 0).exists()


def test_saver_without_ema_is_unchanged(tmp_path, no_barrier):
    module = nn.Linear(2, 2)
    named(module)
    config_file = tmp_path / 'train.toml'
    config_file.write_text('')
    model = _Model()
    engine = _Engine()
    saver = saver_util.Saver(types.SimpleNamespace(config=str(config_file)), {}, False, tmp_path / 'run', model,
                             types.SimpleNamespace(state_dict=lambda: {}), engine, module)
    saver.save_model('step3')
    saver.save_checkpoint(3, 24)
    assert list(model.saves) == [tmp_path / 'run' / 'step3']
    assert not list((tmp_path / 'run').rglob('ema*'))
