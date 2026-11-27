"""utils/loss_scale.py: the config, the scaled gradient, the unscale, and the dynamic scale."""

import pytest
import torch

from utils.loss_scale import (DEFAULTS, LossScaler, apply_loss_scale_config_defaults, compute_dtype_of,
                              computes_in_float16, default_adapter_dtype, refuse_compute_dtype_off_moss)


def config_with(**table):
    config = {'model': {'compute_dtype': 'float16'}, 'optimizer': {'type': 'adamw'}}
    if table:
        config['loss_scale'] = table
    return config


def test_defaults_apply_when_the_model_computes_in_float16():
    config = config_with()
    spec = apply_loss_scale_config_defaults(config)
    assert spec == DEFAULTS and config['loss_scale'] == DEFAULTS
    assert apply_loss_scale_config_defaults(config_with(init_scale=1024))['init_scale'] == 1024


@pytest.mark.parametrize('name', ['float16', 'fp16', 'FLOAT16'])
def test_float16_names(name):
    assert computes_in_float16({'compute_dtype': name})


@pytest.mark.parametrize('model', [{}, {'compute_dtype': 'bfloat16'}, {'dtype': 'float16'}])
def test_other_compute_dtypes_need_no_scale_and_ignore_the_table(model, capsys):
    assert apply_loss_scale_config_defaults({'model': model}) is None
    config = {'model': model, 'loss_scale': {'init_scale': 8}}
    assert apply_loss_scale_config_defaults(config) is None
    assert '[loss_scale] is ignored' in capsys.readouterr().out
    assert 'loss_scale' not in config


@pytest.mark.parametrize('table,message', [
    ({'initial': 1}, 'unknown fields'),
    ({'init_scale': 0.5}, 'init_scale'),
    ({'init_scale': True}, 'init_scale'),
    ({'growth_factor': 1.0}, 'growth_factor'),
    ({'backoff_factor': 1.0}, 'backoff_factor'),
    ({'backoff_factor': 0.0}, 'backoff_factor'),
    ({'growth_interval': 0}, 'growth_interval'),
    ({'growth_interval': 2.5}, 'growth_interval'),
    ({'min_scale': 0.0}, 'min_scale'),
    ({'min_scale': 1e9}, 'above init_scale'),
])
def test_refusals(table, message):
    with pytest.raises(ValueError, match=message):
        apply_loss_scale_config_defaults(config_with(**table))


def test_gradient_release_is_refused():
    config = config_with()
    config['optimizer']['gradient_release'] = True
    with pytest.raises(ValueError, match='gradient_release'):
        apply_loss_scale_config_defaults(config)


def test_the_loss_value_is_unchanged_and_its_gradient_follows_the_scale_at_backward_time():
    scaler = LossScaler(init_scale=8.0)
    w = torch.tensor([2.0, 3.0], requires_grad=True)
    loss = scaler.scale_loss((w * w).sum())
    assert loss.item() == 13.0
    scaler.scale = 32.0   # moved between forward and backward: backward reads it then
    loss.backward()
    assert torch.equal(w.grad, 32.0 * 2 * torch.tensor([2.0, 3.0]))


def test_unscale_divides_every_gradient_and_skips_parameters_without_one():
    scaler = LossScaler(init_scale=4.0)
    a = torch.nn.Parameter(torch.ones(3))
    b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    c = torch.nn.Parameter(torch.ones(1))
    a.grad = torch.full((3,), 8.0)
    b.grad = torch.full((2,), 4.0, dtype=torch.float64)
    scaler.unscale_([a, b, c])
    assert torch.equal(a.grad, torch.full((3,), 2.0))
    assert b.grad.dtype == torch.float64 and torch.equal(b.grad, torch.ones(2, dtype=torch.float64))
    assert c.grad is None
    scaler.unscale_([c])


def test_scaling_keeps_a_gradient_that_float16_would_flush_to_zero():
    x = torch.ones(4, 4)
    target = torch.zeros(4, 4)

    def gradient(scaler):
        w = torch.nn.Parameter(torch.ones(4, 4))
        y = x.half() @ w.half()
        # A loss this small gives float16's backward a gradient of 1e-9 per element, below its smallest
        # subnormal (6e-8): it rounds to zero without a scale.
        loss = ((y.float() - target) * 1e-9).sum()
        (scaler.scale_loss(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_([w])
        return w.grad

    assert gradient(None).abs().max() == 0
    scaled = gradient(LossScaler(init_scale=65536.0))
    assert torch.allclose(scaled, torch.full((4, 4), 4e-9), rtol=2e-2)


def test_the_scale_backs_off_on_a_non_finite_gradient_and_drops_the_step():
    scaler = LossScaler(init_scale=1024.0, backoff_factor=0.5)
    assert scaler.update(True) is True
    assert scaler.scale == 512.0 and scaler.overflows == 1 and scaler.clean_steps == 0


def test_the_scale_stops_at_the_floor_and_a_non_finite_gradient_there_is_not_the_scales_doing():
    scaler = LossScaler(init_scale=4.0, backoff_factor=0.5, min_scale=1.0)
    assert [scaler.update(True) for _ in range(2)] == [True, True]
    assert scaler.scale == 1.0 and not scaler.can_fall()
    assert scaler.update(True) is False and scaler.scale == 1.0 and scaler.overflows == 2


def test_the_scale_grows_after_growth_interval_clean_steps_and_an_overflow_restarts_the_count():
    scaler = LossScaler(init_scale=8.0, growth_factor=2.0, growth_interval=3)
    for _ in range(2):
        assert scaler.update(False) is False
    assert scaler.scale == 8.0
    scaler.update(True)
    assert scaler.scale == 4.0 and scaler.clean_steps == 0
    for _ in range(3):
        scaler.update(False)
    assert scaler.scale == 8.0 and scaler.clean_steps == 0


def test_state_round_trip_and_logged_scalars():
    scaler = LossScaler(init_scale=64.0, growth_interval=5)
    scaler.update(True)
    scaler.update(False)
    state = scaler.state_dict()
    other = LossScaler()
    other.load_state_dict(state)
    assert (other.scale, other.clean_steps, other.overflows) == (32.0, 1, 1)
    assert other.log_scalars() == {'train/loss_scale': 32.0, 'train/loss_scale_overflows': 1.0}


def test_a_torch_dtype_counts_as_float16_too():
    assert computes_in_float16({'compute_dtype': torch.float16})
    assert not computes_in_float16({'compute_dtype': torch.bfloat16})


@pytest.mark.parametrize('key', ['compute_dtype', 'fp16_fix', 'fp16_fix_scale'])
def test_the_precision_keys_are_refused_off_moss_image(key):
    refuse_compute_dtype_off_moss({'type': 'moss_image', key: 'x'})
    refuse_compute_dtype_off_moss({'type': 'flux'})
    with pytest.raises(ValueError, match=key):
        refuse_compute_dtype_off_moss({'type': 'flux', key: 'x'})


def test_the_guard_drops_a_scale_overflow_without_counting_it_towards_the_patience():
    from utils.step_guard import NonFiniteGuard
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    guard = NonFiniteGuard(patience=1)
    calls = []
    guard.run_step(optimizer, torch.tensor(float('inf')), lambda: calls.append('engine'), scale_overflow=True)
    assert calls == ['engine'] and guard.step_skipped and guard.scale_overflow
    assert guard.consecutive == 0 and guard.skipped == 0
    guard.check_patience()
    guard.run_step(optimizer, torch.tensor(1.0), lambda: None)
    assert not guard.step_skipped and not guard.scale_overflow
    guard.run_step(optimizer, torch.tensor(float('nan')), lambda: None)
    assert guard.step_skipped and not guard.scale_overflow and guard.consecutive == 1 and guard.skipped == 1


def test_the_engine_step_refuses_a_scaler_without_the_guard():
    import types
    from utils import patches
    engine = types.SimpleNamespace(loss_scaler=LossScaler(), nonfinite_guard=None)
    with pytest.raises(RuntimeError, match='non-finite step guard'):
        patches._take_model_step_measured(engine)


def _engine_worker(rank, clipping, init_file, out_path):
    import json
    import os
    os.environ.setdefault('DS_ACCELERATOR', 'cpu')
    os.environ['LOCAL_RANK'] = '0'
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=0, world_size=1, verbose=False)
    from deepspeed.runtime.engine import DeepSpeedEngine
    import utils.patches as patches
    from utils.step_guard import NonFiniteGuard

    DeepSpeedEngine._take_model_step = patches._take_model_step_measured
    DeepSpeedEngine.clip_fp32_gradients = patches._clip_fp32_gradients

    def build():
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer,
            config={'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
                    'gradient_clipping': clipping, 'steps_per_print': 1000})
        engine._support_torch_style_backward = True
        return model, engine

    def step(model, engine, scaler=None, poison=None):
        engine.zero_grad()
        loss = engine(torch.ones(1, 4) * 3).pow(2).sum()
        (scaler.scale_loss(loss) if scaler else loss).backward()
        if poison is not None:
            model.weight.grad[0, 0] = poison
        before = model.weight.detach().clone()
        engine._take_model_step(None)
        return before, model.weight.detach().clone()

    reference_model, reference_engine = build()
    reference_engine.nonfinite_guard = NonFiniteGuard()
    step(reference_model, reference_engine)

    model, engine = build()
    guard = NonFiniteGuard(patience=2)
    scaler = LossScaler(init_scale=1024.0, growth_interval=2, min_scale=256.0)
    engine.nonfinite_guard, engine.loss_scaler = guard, scaler
    results = {}
    step(model, engine, scaler)
    results['same_step_as_unscaled'] = bool(torch.allclose(model.weight, reference_model.weight, rtol=1e-6, atol=1e-7))
    results['after_clean'] = [scaler.scale, scaler.clean_steps, guard.step_skipped]
    before, after = step(model, engine, scaler, poison=float('inf'))
    results['overflow'] = {'weight_same': torch.equal(before, after), 'scale': scaler.scale,
                           'skipped': guard.step_skipped, 'scale_overflow': guard.scale_overflow,
                           'consecutive': guard.consecutive, 'overflows': scaler.overflows,
                           'applied': engine.was_step_applied()}
    step(model, engine, scaler)
    step(model, engine, scaler)
    results['grown'] = scaler.scale
    # down to the floor, then a non-finite gradient is the guard's
    for _ in range(3):
        step(model, engine, scaler, poison=float('nan'))
    results['floor'] = {'scale': scaler.scale, 'consecutive': guard.consecutive, 'overflows': scaler.overflows,
                        'scale_overflow': guard.scale_overflow}
    with open(out_path, 'w') as f:
        json.dump(results, f)


@pytest.mark.parametrize('clipping', [1.0, 0.0], ids=['clipping', 'no-clipping'])
def test_inside_a_real_deepspeed_engine(clipping, tmp_path):
    import json
    import torch.multiprocessing as mp
    out = tmp_path / 'out.json'
    mp.spawn(_engine_worker, args=(clipping, str(tmp_path / 'init'), str(out)), nprocs=1, join=True)
    results = json.loads(out.read_text())
    assert results['same_step_as_unscaled']
    assert results['after_clean'] == [1024.0, 1, False]
    assert results['overflow'] == {'weight_same': True, 'scale': 512.0, 'skipped': True, 'scale_overflow': True,
                                   'consecutive': 0, 'overflows': 1, 'applied': False}
    # two clean steps after the overflow: the scale doubles back
    assert results['grown'] == 1024.0
    # 1024 -> 512 -> 256 (the floor), then the third non-finite gradient counts against the run
    assert results['floor'] == {'scale': 256.0, 'consecutive': 1, 'overflows': 3, 'scale_overflow': False}


def test_float16_compute_switches_the_step_guard_on_and_refuses_it_off():
    config = config_with()
    apply_loss_scale_config_defaults(config)
    assert config['skip_nonfinite_steps'] is True
    config = config_with()
    config['skip_nonfinite_steps'] = True
    apply_loss_scale_config_defaults(config)
    config = config_with()
    config['skip_nonfinite_steps'] = False
    with pytest.raises(ValueError, match='skip_nonfinite_steps'):
        apply_loss_scale_config_defaults(config)
    other = {'model': {'compute_dtype': 'bfloat16'}}
    apply_loss_scale_config_defaults(other)
    assert 'skip_nonfinite_steps' not in other


@pytest.mark.parametrize('model,expected', [
    ({'type': 'moss_image', 'dtype': 'float16'}, 'float32'), ({'type': 'moss_image', 'dtype': 'fp16'}, 'float32'),
    ({'type': 'moss_image', 'dtype': 'bfloat16'}, 'bfloat16'), ({'type': 'moss_image', 'dtype': 'float32'}, 'float32'),
    ({'type': 'sdxl', 'dtype': 'float16'}, 'float16'), ({'type': 'flux', 'dtype': 'bfloat16'}, 'bfloat16'),
])
def test_an_adapter_over_a_float16_moss_image_base_trains_in_float32_and_other_models_are_unchanged(model, expected):
    assert default_adapter_dtype(model) == expected


@pytest.mark.parametrize('name,expected', [('float16', torch.float16), ('fp16', torch.float16),
                                           ('BF16', torch.bfloat16), ('fp32', torch.float32),
                                           (torch.float16, torch.float16), ('half', None), (None, None)])
def test_the_compute_dtype_aliases_the_model_takes_resolve_to_torch_dtypes(name, expected):
    assert compute_dtype_of({} if name is None else {'compute_dtype': name}) == expected


@pytest.mark.parametrize('dtype', ['float16', 'fp16', torch.float16])
def test_a_float16_adapter_is_refused_with_float16_compute(dtype):
    config = config_with()
    config['adapter'] = {'type': 'lora', 'dtype': dtype}
    with pytest.raises(ValueError, match=r'\[adapter\] dtype'):
        apply_loss_scale_config_defaults(config)
    config['adapter']['dtype'] = 'float32'
    apply_loss_scale_config_defaults(config)


def test_the_cpu_offload_optimizer_is_refused_with_float16_compute():
    config = config_with()
    config['optimizer'] = {'type': 'offload'}
    with pytest.raises(ValueError, match='CPU-offload'):
        apply_loss_scale_config_defaults(config)


def _two_rank_worker(rank, init_file, out_dir):
    import json
    import os
    os.environ.setdefault('DS_ACCELERATOR', 'cpu')
    os.environ['LOCAL_RANK'] = str(rank)
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=2, verbose=False)
    from deepspeed.runtime.engine import DeepSpeedEngine
    import utils.patches as patches
    from utils.step_guard import NonFiniteGuard

    DeepSpeedEngine._take_model_step = patches._take_model_step_measured
    DeepSpeedEngine.clip_fp32_gradients = patches._clip_fp32_gradients
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer,
        config={'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
                'gradient_clipping': 1.0, 'steps_per_print': 1000})
    engine._support_torch_style_backward = True
    guard = NonFiniteGuard(patience=2)
    scaler = LossScaler(init_scale=1024.0, min_scale=1.0)
    engine.nonfinite_guard, engine.loss_scaler = guard, scaler

    def step(poison_rank=None):
        engine.zero_grad()
        scaler.scale_loss(engine(torch.ones(1, 4) * (rank + 1)).pow(2).sum()).backward()
        if poison_rank == rank:
            model.weight.grad[0, 0] = float('inf')
        before = model.weight.detach().clone()
        engine._take_model_step(None)
        return {'moved': not torch.equal(before, model.weight.detach()), 'scale': scaler.scale,
                'skipped': guard.step_skipped, 'scale_overflow': guard.scale_overflow}

    results = [step(), step(poison_rank=1), step(), step(poison_rank=0)]
    with open(os.path.join(out_dir, f'{rank}.json'), 'w') as f:
        json.dump(results, f)


def test_every_rank_takes_the_same_overflow_decision_when_only_one_has_a_non_finite_gradient(tmp_path):
    import json
    import torch.multiprocessing as mp
    mp.spawn(_two_rank_worker, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    per_rank = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in (0, 1)]
    expected = [(True, 1024.0, False), (False, 512.0, True), (True, 512.0, False), (False, 256.0, True)]
    for results in per_rank:
        assert [(r['moved'], r['scale'], r['skipped']) for r in results] == expected
        assert all(r['scale_overflow'] == r['skipped'] for r in results)
