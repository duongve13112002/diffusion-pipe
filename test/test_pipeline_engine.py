"""DeepSpeed's PipelineEngine with diffusion-pipe's patches, in real gloo process groups on CPU.

A small chain of Linear layers is trained for one step with plain SGD; each stage's update is
compared with the gradient of the same chain computed by torch on one process.
"""

import json

import pytest
import torch
import torch.multiprocessing as mp

LAYERS = 4
WIDTH = 4
LR = 0.5


def _layers():
    torch.manual_seed(0)
    return [torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(LAYERS)]


def _micro_batches(n):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(1, WIDTH, generator=g), torch.randn(1, WIDTH, generator=g)) for _ in range(n)]


def _loss(output, target):
    return torch.nn.functional.mse_loss(output, target)


def _expected_weights(gas):
    """One SGD step on the mean loss over the micro-batches, as a single process computes it."""
    layers = _layers()
    model = torch.nn.Sequential(*layers)
    for x, y in _micro_batches(gas):
        (_loss(model(x), y) / gas).backward()
    return {str(i): (layer.weight - LR * layer.weight.grad).tolist() for i, layer in enumerate(layers)}


def _pipeline_worker(rank, world_size, num_stages, gas, init_file, out_dir):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world_size, verbose=False)
    from deepspeed.runtime.pipe.module import PipelineModule
    from utils.patches import apply_patches
    apply_patches()

    pipe = PipelineModule(layers=_layers(), num_stages=num_stages, loss_fn=_loss, partition_method='uniform')
    params = [p for p in pipe.parameters()]
    engine, *_ = deepspeed.initialize(
        model=pipe,
        model_parameters=params,
        optimizer=torch.optim.SGD(params, lr=LR),
        config={'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': gas,
                'gradient_clipping': 0.0, 'steps_per_print': 10**9},
    )
    # As train.py does after deepspeed.initialize.
    engine._support_torch_style_backward = True
    engine.train_batch(iter(_micro_batches(gas)))
    weights = {name.split('.')[0]: p.tolist() for name, p in pipe.named_parameters()}
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(weights, f)


@pytest.mark.parametrize('num_stages, gas', [(1, 2), (2, 1), (2, 2), (3, 3)], ids=['pp1-gas2', 'pp2-gas1', 'pp2-gas2', 'pp3-gas3'])
def test_every_stage_steps_on_the_mean_gradient_of_the_micro_batches(num_stages, gas, tmp_path):
    mp.spawn(_pipeline_worker, args=(num_stages, num_stages, gas, str(tmp_path / 'init'), str(tmp_path)), nprocs=num_stages)
    trained = {}
    for rank in range(num_stages):
        trained.update(json.loads((tmp_path / f'{rank}.json').read_text()))
    expected = _expected_weights(gas)
    assert sorted(trained) == sorted(expected)
    for layer in expected:
        assert trained[layer] == [pytest.approx(row, abs=1e-6) for row in expected[layer]], f'layer {layer}'


def _stats_worker(rank, num_stages, clipping, init_file, out_dir):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=num_stages, verbose=False)
    from deepspeed.runtime.pipe.module import PipelineModule
    from utils.patches import apply_patches
    from utils.step_stats import StepStats
    apply_patches()

    pipe = PipelineModule(layers=_layers(), num_stages=num_stages, loss_fn=_loss, partition_method='uniform')
    params = [p for p in pipe.parameters()]
    engine, *_ = deepspeed.initialize(
        model=pipe,
        model_parameters=params,
        optimizer=torch.optim.SGD(params, lr=LR),
        config={'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2,
                'gradient_clipping': clipping, 'steps_per_print': 10**9},
    )
    engine._support_torch_style_backward = True
    stats = StepStats(engine, log_update_norm=True)
    stats.before_step(True)
    engine.train_batch(iter(_micro_batches(2)))
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(stats.after_step(), f)


@pytest.mark.parametrize('num_stages, clipping', [(2, 0.0), (2, 1e-3), (1, 100.0), (3, 1e-3)],
                         ids=['pp2-no-clipping', 'pp2-clipped', 'pp1-not-clipped', 'pp3-clipped'])
def test_every_rank_logs_the_whole_models_gradient_and_update(num_stages, clipping, tmp_path):
    mp.spawn(_stats_worker, args=(num_stages, clipping, str(tmp_path / 'init'), str(tmp_path)), nprocs=num_stages)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(num_stages)]

    layers = _layers()
    model = torch.nn.Sequential(*layers)
    for x, y in _micro_batches(2):
        (_loss(model(x), y) / 2).backward()
    grad = torch.cat([layer.weight.grad.flatten() for layer in layers]).norm().item()
    weight = torch.cat([layer.weight.detach().flatten() for layer in layers]).norm().item()
    clipped = clipping > 0 and grad > clipping
    scale = clipping / (grad + 1e-6) if clipped else 1.0
    expected = {'train/grad_norm': grad, 'train/update_norm': LR * scale * grad,
                'train/update_ratio': LR * scale * grad / weight}
    if clipping > 0:
        expected['train/grad_clipped'] = float(clipped)
    # The comparison is not vacuous: the clipped cases do clip.
    assert clipped == (clipping == 1e-3)
    for result in results:
        assert result == pytest.approx(expected, rel=1e-4)
