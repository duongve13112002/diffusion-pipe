"""utils/step_guard.py: the config keys, the guard's bookkeeping, and the whole thing inside a real
DeepSpeed engine (one gloo process on CPU) with the engine step patch of utils/patches.py."""

import json
import os

import pytest
import torch
import torch.multiprocessing as mp

from utils.step_guard import (DEFAULT_PATIENCE, NonFiniteGuard, NonFiniteStepsError,
                              apply_step_guard_config_defaults)


def test_config_defaults_and_refusals():
    config = {}
    apply_step_guard_config_defaults(config)
    assert config == {'skip_nonfinite_steps': False, 'nonfinite_patience': DEFAULT_PATIENCE}
    for bad in ({'skip_nonfinite_steps': 1}, {'nonfinite_patience': -1}, {'nonfinite_patience': True},
                {'nonfinite_patience': 1.5}):
        with pytest.raises(ValueError):
            apply_step_guard_config_defaults(dict(bad))
    config = {'skip_nonfinite_steps': True, 'optimizer': {'gradient_release': True}}
    with pytest.raises(ValueError, match='gradient_release'):
        apply_step_guard_config_defaults(config)
    # Off next to gradient_release is fine, and 0 means never stop.
    apply_step_guard_config_defaults({'optimizer': {'gradient_release': True}})
    config = {'skip_nonfinite_steps': True, 'nonfinite_patience': 0}
    apply_step_guard_config_defaults(config)


class CountingOptimizer(torch.optim.SGD):
    steps = 0

    def step(self, *args, **kwargs):
        type(self).steps += 1
        return super().step(*args, **kwargs)


def test_a_non_finite_norm_suppresses_the_step_and_puts_everything_back():
    CountingOptimizer.steps = 0
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = CountingOptimizer([param], lr=0.1)
    guard = NonFiniteGuard(patience=3)
    seen = []

    def take_step():
        seen.append(getattr(optimizer, 'overflow', False))
        optimizer.step()
        return 'done'

    param.grad = torch.ones(3)
    assert guard.run_step(optimizer, torch.tensor(1.0), take_step) == 'done'
    assert CountingOptimizer.steps == 1 and not guard.step_skipped and guard.consecutive == 0
    after_first = param.detach().clone()

    for bad in (float('nan'), float('inf'), -float('inf')):
        assert guard.run_step(optimizer, torch.tensor(bad), take_step) == 'done'
        assert guard.step_skipped
    assert CountingOptimizer.steps == 1, 'the optimizer stepped on a non-finite norm'
    assert torch.equal(param.detach(), after_first)
    assert seen == [False, True, True, True]
    assert guard.skipped == 3 and guard.consecutive == 3
    # The instance is as it was: the class's step, and no overflow flag.
    assert 'step' not in vars(optimizer) and not hasattr(optimizer, 'overflow')
    with pytest.raises(NonFiniteStepsError, match='3 optimizer steps in a row'):
        guard.check_patience()

    guard.run_step(optimizer, torch.tensor(2.0), take_step)
    assert guard.consecutive == 0 and guard.skipped == 3 and CountingOptimizer.steps == 2
    guard.check_patience()
    assert guard.log_scalars() == {'train/nonfinite_skipped': 3.0}


def test_overflow_flag_the_optimizer_already_had_is_restored():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    optimizer.overflow = False
    guard = NonFiniteGuard()
    guard.run_step(optimizer, torch.tensor(float('nan')), lambda: None)
    assert optimizer.overflow is False


def test_patience_zero_never_stops():
    guard = NonFiniteGuard(patience=0)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    for _ in range(50):
        guard.run_step(optimizer, torch.tensor(float('nan')), lambda: None)
    guard.check_patience()


def test_the_step_is_restored_even_when_the_engine_step_raises():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    guard = NonFiniteGuard()

    def boom():
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        guard.run_step(optimizer, torch.tensor(float('nan')), boom)
    assert 'step' not in vars(optimizer) and not hasattr(optimizer, 'overflow')


def _engine_worker(rank, clipping, init_file, out_path):
    os.environ.setdefault('DS_ACCELERATOR', 'cpu')
    os.environ['LOCAL_RANK'] = '0'
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=0, world_size=1, verbose=False)
    from deepspeed.runtime.engine import DeepSpeedEngine
    import utils.patches as patches

    calls = {'grad_norm': 0}
    original_grad_norm = patches.grad_norm

    def counting_grad_norm(*args, **kwargs):
        calls['grad_norm'] += 1
        return original_grad_norm(*args, **kwargs)

    patches.grad_norm = counting_grad_norm
    DeepSpeedEngine._take_model_step = patches._take_model_step_measured
    DeepSpeedEngine.clip_fp32_gradients = patches._clip_fp32_gradients

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config={'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
                'gradient_clipping': clipping, 'steps_per_print': 1000})
    engine._support_torch_style_backward = True
    guard = NonFiniteGuard(patience=2)
    engine.nonfinite_guard = guard

    def snapshot():
        state = optimizer.state.get(model.weight, {})
        momentum = state.get('momentum_buffer')
        return {'weight': model.weight.detach().clone(), 'bias': model.bias.detach().clone(),
                'momentum': None if momentum is None else momentum.clone(),
                'lr': scheduler.get_last_lr()[0]}

    def step(poison):
        engine.zero_grad()
        engine(torch.ones(1, 4)).sum().backward()
        if poison:
            model.weight.grad[0, 0] = poison
        calls['grad_norm'] = 0
        before = snapshot()
        engine._take_model_step(None)
        return before, snapshot(), calls['grad_norm']

    results = {}
    b, a, n = step(None)
    results['normal'] = {'moved': not torch.equal(b['weight'], a['weight']), 'lr_moved': b['lr'] != a['lr'],
                         'norm_calls': n, 'skipped': guard.step_skipped}
    b, a, n = step(float('nan'))
    results['nan'] = {'weight_same': torch.equal(b['weight'], a['weight']), 'bias_same': torch.equal(b['bias'], a['bias']),
                      'momentum_same': torch.equal(b['momentum'], a['momentum']), 'lr_same': b['lr'] == a['lr'],
                      'skipped': guard.step_skipped, 'norm_calls': n, 'engine_skipped': engine.skipped_steps,
                      'applied': engine.was_step_applied(), 'overflow_left': hasattr(optimizer, 'overflow'),
                      'grads_cleared': all(p.grad is None for p in model.parameters())}
    b, a, n = step(None)
    results['after'] = {'moved': not torch.equal(b['weight'], a['weight']), 'consecutive': guard.consecutive,
                        'applied': engine.was_step_applied()}
    step(float('inf'))
    step(float('nan'))
    try:
        guard.check_patience()
        results['patience'] = 'no error'
    except NonFiniteStepsError:
        results['patience'] = 'stopped'
    results['weights_finite'] = bool(torch.isfinite(model.weight).all())
    with open(out_path, 'w') as f:
        json.dump(results, f)


@pytest.mark.parametrize('clipping', [1.0, 0.0], ids=['clipping', 'no-clipping'])
def test_inside_a_real_deepspeed_engine(clipping, tmp_path):
    out = tmp_path / 'out.json'
    mp.spawn(_engine_worker, args=(clipping, str(tmp_path / 'init'), str(out)), nprocs=1, join=True)
    results = json.loads(out.read_text())
    assert results['normal'] == {'moved': True, 'lr_moved': True, 'norm_calls': 1, 'skipped': False}
    nan = results['nan']
    assert nan['weight_same'] and nan['bias_same'] and nan['momentum_same'] and nan['lr_same']
    assert nan['skipped'] and nan['norm_calls'] == 1 and nan['engine_skipped'] == 1
    assert nan['applied'] is False and nan['overflow_left'] is False and nan['grads_cleared']
    assert results['after'] == {'moved': True, 'consecutive': 0, 'applied': True}
    assert results['patience'] == 'stopped' and results['weights_finite']


def test_an_instance_level_step_survives_a_skipped_step():
    """The LR scheduler wraps optimizer.step on the instance, and the Prodigy pipeline sync replaces it:
    a skipped step has to put that one back, not remove it."""
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    wrapper = vars(optimizer)['step']
    calls = []
    guard = NonFiniteGuard()
    guard.run_step(optimizer, torch.tensor(float('inf')), lambda: calls.append('engine'))
    assert vars(optimizer)['step'] is wrapper and calls == ['engine']
    parameter.grad = torch.ones(1)
    optimizer.step()
    assert parameter.item() == pytest.approx(0.9) and scheduler is not None
