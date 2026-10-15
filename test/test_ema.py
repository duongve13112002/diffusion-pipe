"""utils/ema.py and its use by utils/saver.py: the average itself, its checkpoint round trip
(across a different pipeline split), and the ema/ save written beside every normal save."""

import os
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from utils import ema as ema_util
from utils import saver as saver_util
from utils.ema import (ParameterEMA, apply_ema_config_defaults, load_ema_checkpoint, load_ema_weights,
                       save_ema_checkpoint, to_bfloat16_stochastic)


def named(module, prefix=''):
    for name, p in module.named_parameters():
        p.original_name = prefix + name
    return list(module.named_parameters())


def test_config_defaults_and_validation():
    assert apply_ema_config_defaults({}) is False
    config = {'ema_decay': 0.999}
    assert apply_ema_config_defaults(config) is True
    assert config == {'ema_decay': 0.999, 'ema_update_every': 1, 'ema_start_step': 0, 'ema_device': 'cpu',
                      'ema_dtype': 'float32'}
    for bad in ({'ema_decay': 1.0}, {'ema_decay': 0}, {'ema_decay': True}, {'ema_decay': '0.9'},
                {'ema_decay': 0.9, 'ema_update_every': 0}, {'ema_decay': 0.9, 'ema_start_step': -1},
                {'ema_decay': 0.9, 'ema_device': 'cuda'}, {'ema_decay': 0.9, 'ema_dtype': 'float16'},
                {'ema_update_every': 10}, {'ema_dtype': 'bfloat16'}):
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
    ema = ParameterEMA(named(layer), decay=0.5, start_step=3)
    for step in (1, 2, 3):
        with torch.no_grad():
            layer.weight.add_(1.0)
        ema.update(step)
        torch.testing.assert_close(ema.averaged('weight', layer.weight), layer.weight.detach())
    # Set from the weights once, at start_step, then averaged from there.
    torch.testing.assert_close(ema.shadow['weight'], layer.weight.detach())
    assert ema.num_updates == 0 and not ema.following
    start = layer.weight.detach().clone()
    with torch.no_grad():
        layer.weight.add_(2.0)
    ema.update(4)
    torch.testing.assert_close(ema.averaged('weight', layer.weight), start + 1.0)


def test_resume_from_a_checkpoint_saved_before_start_step(tmp_path):
    """The shadow saved during warmup is stale; the resumed run sets it from the weights."""
    layer = nn.Linear(2, 2)
    ema = ParameterEMA(named(layer), decay=0.5, start_step=10)
    ema.update(4)
    (tmp_path / 'global_step4').mkdir()
    save_ema_checkpoint(ema, tmp_path, 'global_step4', 0)
    with torch.no_grad():
        layer.weight.add_(5.0)
    # Resumed with a lower start_step: past it already, so the first update starts the average.
    resumed = ParameterEMA(named(layer), decay=0.5, start_step=2)
    resumed.shadow['weight'].zero_()
    assert load_ema_checkpoint(resumed, tmp_path, 'global_step4') == (1, [])
    assert resumed.following
    resumed.update(5)
    assert not resumed.following
    torch.testing.assert_close(resumed.shadow['weight'], layer.weight.detach())


def _offloaded(ema):
    """Exercise the GPU-weights paths on the CPU: the code is the same, minus CUDA events."""
    ema._offloaded = lambda: True
    return ema


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_offloaded_update_matches_the_plain_one(dtype):
    torch.manual_seed(0)
    a = nn.Linear(64, 64).to(torch.bfloat16)
    b = nn.Linear(64, 64).to(torch.bfloat16)
    b.load_state_dict(a.state_dict())
    plain = ParameterEMA(named(a), decay=0.9, dtype=dtype)
    offloaded = _offloaded(ParameterEMA(named(b), decay=0.9, dtype=dtype))
    for step in range(1, 6):
        delta = [torch.randn_like(p) for p in a.parameters()]
        with torch.no_grad():
            for pa, pb, d in zip(a.parameters(), b.parameters(), delta):
                pa.add_(d)
                pb.add_(d)
        plain.update(step)
        offloaded.update(step)
    if dtype == 'float32':
        assert offloaded._worker is not None          # still averaging in the background
    for key in plain.shadow:
        expected = plain.averaged(key, plain.params[key]).float()
        actual = offloaded.averaged(key, offloaded.params[key]).float()   # waits for it
        torch.testing.assert_close(actual, expected, atol=0.05 if dtype == 'bfloat16' else 0, rtol=0)
    assert offloaded._worker is None and offloaded.num_updates == 5


def test_background_update_reads_the_weights_of_its_own_step():
    """The weights are copied before update() returns: changing them afterwards (the next
    optimizer step) does not leak into that update."""
    layer = nn.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    ema = _offloaded(ParameterEMA(named(layer), decay=0.5))
    with torch.no_grad():
        layer.weight.fill_(3.0)
    ema.update(1)
    with torch.no_grad():
        layer.weight.fill_(100.0)
    torch.testing.assert_close(ema.averaged('weight', layer.weight), torch.full((4, 4), 2.0))


def test_background_update_errors_reach_the_training_loop():
    layer = nn.Linear(2, 2)
    ema = _offloaded(ParameterEMA(named(layer), decay=0.5))
    ema.shadow['weight'] = torch.zeros(3, 3)          # shape mismatch: lerp_ raises in the thread
    ema.update(1)
    with pytest.raises(RuntimeError, match='EMA update failed'):
        ema.wait()
    ema.wait()                                        # reported once


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
        Path(save_dir, tag).mkdir(parents=True, exist_ok=True)
        self.saved.append(tag)
        # What was on disk when DeepSpeed started writing (and, at its end, `latest`).
        self.files_before = sorted(p.name for p in Path(save_dir, tag).iterdir())


class _Model:
    """Writes what the saver hands to the model's own save functions, so a test reads it back
    from where the saver published it."""

    def save_model(self, save_dir, state_dict):
        torch.save(state_dict, Path(save_dir) / 'model.pt')

    save_adapter = save_model


def _saved(save_dir):
    return torch.load(Path(save_dir) / 'model.pt', weights_only=True)


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
    raw, averaged = _saved(tmp_path / 'run' / 'epoch1'), _saved(tmp_path / 'run' / 'epoch1' / 'ema')
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
    # Written before DeepSpeed's files: DeepSpeed writes `latest` before it returns, and a
    # crash after that must not leave `latest` naming a checkpoint without its average.
    assert engine.files_before == [ema_util.checkpoint_file(tmp_path / 'run', 'global_step12', 0).name]


def test_a_failed_average_write_stops_the_checkpoint(tmp_path, no_barrier, monkeypatch):
    module = nn.Linear(2, 2)
    ema = ParameterEMA(named(module), decay=0.5)
    config_file = tmp_path / 'train.toml'
    config_file.write_text('')
    engine = _Engine()
    saver = saver_util.Saver(types.SimpleNamespace(config=str(config_file)), {}, False, tmp_path / 'run', _Model(),
                             types.SimpleNamespace(state_dict=lambda: {}), engine, module, ema=ema)

    def full_volume(obj, path):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(ema_util, 'torch_save_atomic', full_volume)
    with pytest.raises(OSError, match='No space'):
        saver.save_checkpoint(5, 40)
    assert engine.saved == []


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
    assert sorted(p.name for p in (tmp_path / 'run').iterdir()) == ['global_step12', 'step3']
    assert sorted(p.name for p in (tmp_path / 'run' / 'step3').iterdir()) == ['model.pt', 'train.toml']
    assert not list((tmp_path / 'run').rglob('ema*'))


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(0)
    x = torch.full((200000,), 1.001)        # between the bf16 neighbours 1.0 and 1.0078125
    rounded = to_bfloat16_stochastic(x)
    assert set(rounded.unique().tolist()) == {1.0, 1.0078125}
    assert abs(rounded.float().mean().item() - 1.001) < 2e-4
    exact = torch.tensor([0.0, 1.0, -2.5, 3.140625])  # already bf16: unchanged
    assert torch.equal(to_bfloat16_stochastic(exact).float(), exact)


def test_bf16_average_follows_the_weights_where_rounding_to_nearest_would_not():
    """At decay 0.9999 each update is 1e-4 of the gap, under half a bf16 step unless the gap is
    ~20x the weights' magnitude: rounded to nearest, a bf16 average at 1 with weights at 2 never
    moves. Stochastic rounding keeps it on the fp32 average's course, within bf16 noise."""
    torch.manual_seed(0)
    layer = nn.Linear(256, 256, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    exact = ParameterEMA(named(layer), decay=0.9999)
    half = ParameterEMA(named(layer), decay=0.9999, dtype='bfloat16')
    assert half.shadow['weight'].dtype == torch.bfloat16 and half.num_bytes() * 2 == exact.num_bytes()
    with torch.no_grad():
        layer.weight.fill_(2.0)
    nearest = torch.ones(1, dtype=torch.bfloat16)
    for step in range(1, 2001):
        exact.update(step)
        half.update(step)
        nearest = nearest.float().lerp_(torch.full((1,), 2.0), 1e-4).to(torch.bfloat16)
    target = exact.shadow['weight'].mean().item()
    assert abs(target - (2 - 0.9999 ** 2000)) < 1e-4
    assert abs(half.shadow['weight'].float().mean().item() - target) < 0.005
    assert nearest.item() == 1.0


def test_checkpoint_loads_across_an_ema_dtype_change(tmp_path):
    layer = nn.Linear(3, 3)
    fp32 = ParameterEMA(named(layer), decay=0.5)
    (tmp_path / 'global_step1').mkdir()
    save_ema_checkpoint(fp32, tmp_path, 'global_step1', 0)
    bf16 = ParameterEMA(named(layer), decay=0.5, dtype='bfloat16')
    assert load_ema_checkpoint(bf16, tmp_path, 'global_step1') == (1, [])
    torch.testing.assert_close(bf16.shadow['weight'], fp32.shadow['weight'].to(torch.bfloat16))


def test_init_path_validation(tmp_path):
    import safetensors.torch
    safetensors.torch.save_file({'w': torch.zeros(1)}, str(tmp_path / 'model.safetensors'))
    config = {'ema_decay': 0.9, 'ema_init_path': str(tmp_path)}
    assert apply_ema_config_defaults(config) is True
    for bad in ({'ema_init_path': str(tmp_path / 'nope')}, {'ema_init_path': ''},
                {'ema_init_path': str(tmp_path), 'ema_start_step': 100}, {'ema_init_path': 3}):
        with pytest.raises(ValueError):
            apply_ema_config_defaults({'ema_decay': 0.9, **bad})
    with pytest.raises(ValueError):    # EMA off
        apply_ema_config_defaults({'ema_init_path': str(tmp_path)})
    (tmp_path / 'a.safetensors').write_bytes(b'')
    (tmp_path / 'model.safetensors').unlink()
    (tmp_path / 'b.safetensors').write_bytes(b'')
    with pytest.raises(ValueError, match='single'):
        apply_ema_config_defaults({'ema_decay': 0.9, 'ema_init_path': str(tmp_path)})


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_init_path_starts_the_average_from_a_saved_ema(tmp_path, dtype):
    import safetensors.torch
    model = nn.Sequential(nn.Linear(3, 3))
    params = named(model)
    model[0].bias.requires_grad_(False)        # frozen: no shadow, so the save need not have it
    safetensors.torch.save_file({'0.weight': torch.full((3, 3), 2.0, dtype=torch.bfloat16)},
                                str(tmp_path / 'model.safetensors'))
    ema = ParameterEMA(params, decay=0.5, dtype=dtype)
    assert load_ema_weights(ema, tmp_path) == 1
    torch.testing.assert_close(ema.averaged('0.weight', model[0].weight), torch.full((3, 3), 2.0))
    with torch.no_grad():
        model[0].weight.fill_(4.0)
    ema.update(1)
    torch.testing.assert_close(ema.averaged('0.weight', model[0].weight), torch.full((3, 3), 3.0))


def test_init_path_reads_an_adapter_save_with_its_prefix(tmp_path):
    import safetensors.torch
    model = nn.Sequential(nn.Linear(2, 2))
    params = named(model)
    model[0].weight.original_name = '0.lora_A.default.weight'
    ema = ParameterEMA(params, decay=0.9, is_adapter=True)
    safetensors.torch.save_file({'diffusion_model.0.lora_A.weight': torch.ones(2, 2),
                                 'diffusion_model.0.bias': torch.zeros(2)},
                                str(tmp_path / 'adapter_model.safetensors'))
    assert load_ema_weights(ema, tmp_path / 'adapter_model.safetensors') == 2
    torch.testing.assert_close(ema.shadow['0.lora_A.weight'], torch.ones(2, 2))


def test_init_path_reads_the_repa_projector_saved_beside_the_model(tmp_path):
    import safetensors.torch
    model = nn.ModuleDict({'blocks': nn.Linear(2, 2), 'repa': nn.Linear(2, 2)})
    params = named(model)
    ema = ParameterEMA(params, decay=0.9)
    safetensors.torch.save_file({'blocks.weight': torch.ones(2, 2), 'blocks.bias': torch.ones(2)},
                                str(tmp_path / 'model.safetensors'))
    safetensors.torch.save_file({'weight': torch.full((2, 2), 7.0), 'bias': torch.ones(2)},
                                str(tmp_path / 'repa_projector.safetensors'))
    assert load_ema_weights(ema, tmp_path) == 4
    torch.testing.assert_close(ema.shadow['repa.weight'], torch.full((2, 2), 7.0))


@pytest.mark.parametrize('case', ['missing', 'shape', 'adapter_for_full', 'full_for_adapter'])
def test_init_path_of_another_format_is_an_error_and_changes_nothing(tmp_path, case):
    import safetensors.torch
    model = nn.Sequential(nn.Linear(3, 3))
    params = named(model)
    is_adapter = case == 'full_for_adapter'
    if is_adapter:
        model[0].weight.original_name = '0.lora_A.default.weight'
    saved = {'missing': {'0.weight': torch.zeros(3, 3)},
             'shape': {'0.weight': torch.zeros(4, 3), '0.bias': torch.zeros(3)},
             'adapter_for_full': {'diffusion_model.0.lora_A.weight': torch.zeros(3, 3),
                                  'diffusion_model.0.lora_B.weight': torch.zeros(3, 3)},
             'full_for_adapter': {'0.weight': torch.zeros(3, 3), '0.bias': torch.zeros(3),
                                  '1.weight': torch.zeros(3, 3)}}[case]
    safetensors.torch.save_file(saved, str(tmp_path / 'model.safetensors'))
    ema = ParameterEMA(params, decay=0.9, is_adapter=is_adapter)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with pytest.raises(ValueError, match='not a save of the model being trained') as error:
        load_ema_weights(ema, tmp_path)
    if case == 'adapter_for_full':
        assert 'adapter save, but this run is a full fine-tune' in str(error.value)
    if case == 'full_for_adapter':
        assert 'full model save, but this run trains an adapter' in str(error.value)
    for key, value in before.items():
        torch.testing.assert_close(ema.shadow[key], value)


def test_init_path_ends_following_and_survives_a_checkpoint(tmp_path):
    import safetensors.torch
    model = nn.Sequential(nn.Linear(2, 2))
    params = named(model)
    safetensors.torch.save_file({'0.weight': torch.full((2, 2), 5.0), '0.bias': torch.zeros(2)},
                                str(tmp_path / 'model.safetensors'))
    ema = ParameterEMA(params, decay=0.9)
    load_ema_weights(ema, tmp_path)
    assert not ema.following
    (tmp_path / 'global_step1').mkdir()
    save_ema_checkpoint(ema, tmp_path, 'global_step1', 0)
    resumed = ParameterEMA(named(model), decay=0.9)
    load_ema_checkpoint(resumed, tmp_path, 'global_step1')
    torch.testing.assert_close(resumed.shadow['0.weight'], torch.full((2, 2), 5.0))


def _saver(tmp_path, model, module):
    config_file = tmp_path / 'train.toml'
    config_file.write_text('')
    return saver_util.Saver(types.SimpleNamespace(config=str(config_file)), {}, False, tmp_path / 'run', model,
                            types.SimpleNamespace(state_dict=lambda: {}), _Engine(), module)


def test_a_failed_save_leaves_the_previous_one(tmp_path, no_barrier):
    """The save directory is built under a temporary name and renamed into place: a save that
    fails part way leaves the previous save of that name whole, and the next run removes the
    leftover."""
    module = nn.Linear(2, 2)
    named(module)
    saver = _saver(tmp_path, _Model(), module)
    saver.save_model('step3')
    before = _saved(tmp_path / 'run' / 'step3')

    class Failing(_Model):
        def save_model(self, save_dir, state_dict):
            (Path(save_dir) / 'model.pt').write_bytes(b'torn')
            raise OSError('volume went away')

    with torch.no_grad():
        module.weight.add_(1.0)
    saver.model = Failing()
    with pytest.raises(OSError, match='volume went away'):
        saver.save_model('step3')
    after = _saved(tmp_path / 'run' / 'step3')
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)
    assert (tmp_path / 'run' / '.step3.partial').exists()

    saver = _saver(tmp_path, _Model(), module)
    assert not (tmp_path / 'run' / '.step3.partial').exists()
    saver.save_model('step3')
    assert torch.equal(_saved(tmp_path / 'run' / 'step3')['weight'], module.weight.detach())
    assert sorted(p.name for p in (tmp_path / 'run').iterdir()) == ['step3']


@pytest.mark.parametrize('in_ema', [False, True], ids=['save', 'ema'])
def test_a_save_moved_aside_but_not_replaced_goes_back(tmp_path, no_barrier, in_ema):
    """publish_dir moves the old save aside, then renames the new one in: a crash between the
    two renames leaves only `.name.old-*`, which the next run puts back rather than deletes."""
    module = nn.Linear(2, 2)
    named(module)
    saver = _saver(tmp_path, _Model(), module)
    saver.save_model('step3')
    run = tmp_path / 'run'
    if in_ema:
        (run / 'step3' / 'ema').mkdir()
        torch.save({'w': torch.ones(1)}, run / 'step3' / 'ema' / 'model.pt')
        os.replace(run / 'step3' / 'ema', run / 'step3' / '.ema.old-1234')
        moved = run / 'step3' / 'ema'
    else:
        os.replace(run / 'step3', run / '.step3.old-1234')
        moved = run / 'step3'
    (run / '.step4.partial').mkdir()

    _saver(tmp_path, _Model(), module)
    assert (moved / 'model.pt').exists()
    assert sorted(p.name for p in run.rglob('.*')) == []


def test_the_model_is_saved_before_the_checkpoint_of_the_same_epoch(tmp_path, no_barrier, monkeypatch):
    module = nn.Linear(2, 2)
    named(module)
    saver = _saver(tmp_path, _Model(), module)
    saver.config = {'save_every_n_epochs': 1, 'checkpoint_every_n_epochs': 1, 'epochs': 5}
    order = []
    monkeypatch.setattr(saver, 'save_model', lambda name: order.append('model'))
    monkeypatch.setattr(saver, 'save_checkpoint', lambda step, examples: order.append('checkpoint'))
    saver.train_dataloader = types.SimpleNamespace(epoch=2)
    saver.process_epoch(1, 10, 80)
    assert order == ['model', 'checkpoint']


def _failing_save_worker(rank, world_size, init_file, save_root, results):
    import torch.distributed as torch_dist
    torch_dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=world_size)
    saver_util.is_main_process = lambda: rank == 0     # DeepSpeed's comm is not set up here
    try:
        module = nn.Linear(2, 2)
        named(module)

        class Grid:
            def get_data_parallel_rank(self):
                return rank

            def get_pipe_parallel_rank(self):
                return 0

        class Failing(_Model):
            def save_model(self, save_dir, state_dict):
                raise OSError('volume went away')

        config_file = Path(save_root) / 'train.toml'
        if rank == 0:
            Path(save_root).mkdir(exist_ok=True)
            config_file.write_text('')
        torch_dist.barrier()
        engine = types.SimpleNamespace(grid=Grid(), global_steps=1)
        saver = saver_util.Saver(types.SimpleNamespace(config=str(config_file)), {}, False, Path(save_root) / 'run',
                                 Failing(), types.SimpleNamespace(state_dict=lambda: {}), engine, module)
        try:
            saver.save_model('step1')
            results[rank] = 'returned'
        except Exception as e:
            results[rank] = f'{type(e).__name__}: {e}'
    finally:
        torch_dist.destroy_process_group()


def test_a_save_failing_on_rank_0_stops_every_rank(tmp_path):
    """Without the shared error the other ranks return from the save and wait at the next
    collective until the process group times out."""
    import torch.multiprocessing as mp
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_failing_save_worker, args=(2, str(tmp_path / 'init'), str(tmp_path), results), nprocs=2,
                 join=True)
        results = dict(results)
    assert all('rank 0: OSError: volume went away' in results[r] for r in (0, 1)), results
