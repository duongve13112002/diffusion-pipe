"""Optimizers under pipeline parallelism (utils/pipeline_optim.py), in real gloo process groups.

Each pipeline stage builds its own optimizer over its own layers. Prodigy must still estimate one
step size for the whole model, and a stage with nothing to train must still join the collectives
of the others.
"""

import json
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from utils.pipeline_optim import check_pipeline_optimizer, pipeline_collective

STEPS = 8
# Two parameters per stage; stage 1's gradients are much larger, so a per-stage d differs.
SHAPES = [(3, 2), (4,), (2, 2), (5,)]
STAGE_OF = [0, 0, 1, 1]
GRAD_SCALE = [1.0, 1.0, 30.0, 30.0]


def _params():
    g = torch.Generator().manual_seed(0)
    return [torch.nn.Parameter(torch.randn(shape, generator=g)) for shape in SHAPES]


def _set_grads(params, indices, step):
    # A steady direction with some noise, so Prodigy's d keeps growing.
    for p, i in zip(params, indices):
        direction = torch.randn(p.shape, generator=torch.Generator().manual_seed(i))
        noise = torch.randn(p.shape, generator=torch.Generator().manual_seed(100 * (step + 1) + i))
        p.grad = (direction + 0.3 * noise) * GRAD_SCALE[i]


def _prodigy(params, eps):
    from pytorch_optimizer import Prodigy
    optimizer = Prodigy(params, lr=1.0, eps=eps)
    # As train.py builds it, with an LR scheduler wrapping optimizer.step.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return optimizer, scheduler


def _single_process_prodigy(eps):
    params = _params()
    optimizer, scheduler = _prodigy(params, eps)
    ds = []
    for step in range(STEPS):
        _set_grads(params, range(len(params)), step)
        optimizer.step()
        scheduler.step()
        ds.append(optimizer.param_groups[0]['d'])
    return [p.flatten().tolist() for p in params], ds


def _init(rank, world_size, init_file):
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world_size, verbose=False, timeout=timedelta(seconds=60))
    from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipelineParallelGrid
    return PipelineParallelGrid(topology=PipeDataParallelTopology(num_pp=world_size, num_dp=1))


def _prodigy_worker(rank, eps, init_file, out_dir):
    from utils.pipeline_optim import sync_prodigy
    grid = _init(rank, 2, init_file)
    indices = [i for i, stage in enumerate(STAGE_OF) if stage == grid.get_stage_id()]
    params = [_params()[i] for i in indices]
    optimizer, scheduler = _prodigy(params, eps)
    sync_prodigy(optimizer, grid.get_model_parallel_group())
    ds = []
    for step in range(STEPS):
        _set_grads(params, indices, step)
        optimizer.step()
        scheduler.step()
        ds.append(optimizer.param_groups[0]['d'])
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump({'params': {i: p.flatten().tolist() for i, p in zip(indices, params)}, 'd': ds}, f)


@pytest.mark.parametrize('eps', [None, 1e-8], ids=['atan2', 'eps'])
def test_prodigy_estimates_one_step_size_for_the_whole_pipeline(eps, tmp_path):
    mp.spawn(_prodigy_worker, args=(eps, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    expected_params, expected_ds = _single_process_prodigy(eps)
    # d grows over the steps, so the comparison is not vacuous.
    assert expected_ds[-1] > 10 * expected_ds[0]
    for result in results:
        assert result['d'] == pytest.approx(expected_ds, rel=1e-5)
        for i, p in result['params'].items():
            assert p == pytest.approx(expected_params[int(i)], rel=1e-4, abs=1e-6), f'param {i}'


def _dummy_worker(rank, collective, init_file, out_dir):
    from utils.pipeline_optim import DummyOptimizer, sync_prodigy
    grid = _init(rank, 2, init_file)
    group = grid.get_model_parallel_group()
    result = {}
    if grid.get_stage_id() == 0:
        params = [_params()[i] for i in (0, 1)]
        if collective == 'genericoptim':
            from optimizers.generic_optim import GenericOptim
            optimizer = GenericOptim(params, lr=1e-3, mpu=grid)
        else:
            optimizer, _ = _prodigy(params, None)
            sync_prodigy(optimizer, group)
        for step in range(2):
            _set_grads(params, (0, 1), step)
            optimizer.step()
        result['params'] = [p.flatten().tolist() for p in params]
        if collective == 'prodigy':
            result['d'] = optimizer.param_groups[0]['d']
    else:
        # Stage 1 has nothing to train.
        optimizer = DummyOptimizer(collective, group)
        for step in range(2):
            optimizer.step()
    if collective == 'genericoptim':
        result['grad_norm'] = optimizer._grad_norm
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


@pytest.mark.parametrize('collective', ['genericoptim', 'prodigy'])
def test_a_stage_with_nothing_to_train_joins_the_other_stages_collective(collective, tmp_path):
    mp.spawn(_dummy_worker, args=(collective, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    if collective == 'genericoptim':
        params = _params()
        _set_grads(params, (0, 1), 1)
        norm = torch.cat([p.grad.flatten() for p in params[:2]]).norm().item()
        # Both stages report the norm of the gradients there are, stage 0's.
        assert results[0]['grad_norm'] == pytest.approx(norm, rel=1e-5)
        assert results[1]['grad_norm'] == pytest.approx(norm, rel=1e-5)
    else:
        # The zeros of the empty stage leave stage 0's Prodigy as it would be alone.
        params = [_params()[i] for i in (0, 1)]
        optimizer, _ = _prodigy(params, None)
        for step in range(2):
            _set_grads(params, (0, 1), step)
            optimizer.step()
        assert results[0]['d'] == pytest.approx(optimizer.param_groups[0]['d'], rel=1e-6)
        assert results[0]['params'] == [pytest.approx(p.flatten().tolist(), rel=1e-5) for p in params]


@pytest.mark.parametrize('optim_type, num_stages, refused', [
    ('DAdaptAdam', 2, True),
    ('DAdaptSGD', 3, True),
    ('DAdaptAdam', 1, False),
    ('Prodigy', 2, False),
    ('adamw', 4, False),
])
def test_optimizers_that_cannot_share_their_step_size_are_refused_with_pipeline_stages(optim_type, num_stages, refused):
    if refused:
        with pytest.raises(ValueError, match=f"type = '{optim_type}'.*pipeline_stages = {num_stages}"):
            check_pipeline_optimizer(optim_type, num_stages)
    else:
        check_pipeline_optimizer(optim_type, num_stages)


@pytest.mark.parametrize('optim_type, gradient_release, num_stages, collective', [
    ('genericoptim', False, 1, 'genericoptim'),
    ('genericoptim', False, 2, 'genericoptim'),
    ('genericoptim', True, 2, None),
    ('prodigy', False, 2, 'prodigy'),
    ('prodigy', False, 1, None),
    ('prodigy', True, 2, None),
    ('adamw', False, 2, None),
])
def test_the_collective_a_dummy_stage_joins(optim_type, gradient_release, num_stages, collective):
    assert pipeline_collective(optim_type, gradient_release, num_stages) == collective


def _resume_worker(rank, legacy, init_file, out_dir):
    from utils.pipeline_optim import sync_prodigy
    grid = _init(rank, 2, init_file)
    group = grid.get_model_parallel_group()
    indices = [i for i, stage in enumerate(STAGE_OF) if stage == grid.get_stage_id()]
    params = [_params()[i] for i in indices]
    optimizer, _ = _prodigy(params, None)
    if not legacy:
        sync_prodigy(optimizer, group)
    for step in range(3):
        _set_grads(params, indices, step)
        optimizer.step()
    saved_d = optimizer.param_groups[0]['d']
    saved_numerator = optimizer.param_groups[0]['d_numerator'].item()
    state = optimizer.state_dict()

    # Resumed: a new optimizer over the same parameters, loaded from the checkpoint.
    optimizer, _ = _prodigy(params, None)
    optimizer.load_state_dict(state)
    sync_prodigy(optimizer, group)
    for step in range(3, 5):
        _set_grads(params, indices, step)
        optimizer.step()
    first = optimizer.param_groups[0]
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump({'saved_d': saved_d, 'saved_numerator': saved_numerator, 'd': first['d'],
                   'numerator': first['d_numerator'].item(),
                   'params': {i: p.flatten().tolist() for i, p in zip(indices, params)}}, f)


def test_a_resumed_pipeline_prodigy_continues_as_if_never_stopped(tmp_path):
    mp.spawn(_resume_worker, args=(False, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    # One Prodigy over the whole model, five steps without a stop.
    params = _params()
    optimizer, _ = _prodigy(params, None)
    for step in range(5):
        _set_grads(params, range(len(params)), step)
        optimizer.step()
    for result in results:
        assert result['d'] == pytest.approx(optimizer.param_groups[0]['d'], rel=1e-5)
        assert result['numerator'] == pytest.approx(optimizer.param_groups[0]['d_numerator'].item(), rel=1e-5)
        for i, p in result['params'].items():
            assert p == pytest.approx(params[int(i)].flatten().tolist(), rel=1e-4, abs=1e-6), f'param {i}'


def test_a_checkpoint_with_a_step_size_per_stage_resumes_with_one(tmp_path):
    mp.spawn(_resume_worker, args=(True, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    # Saved before Prodigy was shared: each stage had its own d.
    assert results[0]['saved_d'] != pytest.approx(results[1]['saved_d'], rel=1e-3)
    assert results[0]['d'] == results[1]['d']
    assert results[0]['numerator'] == results[1]['numerator']
    assert results[0]['d'] >= max(r['saved_d'] for r in results)


def _learning_rate_worker(rank, lrs, num_stages, init_file, out_dir):
    import contextlib
    import io
    from pytorch_optimizer import Prodigy
    from utils.pipeline_optim import check_prodigy_learning_rates
    _init(rank, 2, init_file)
    params = _params()
    optimizer = Prodigy([{'params': [p], 'lr': lr} for p, lr in zip(params, lrs[rank])])
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            check_prodigy_learning_rates(optimizer, 'prodigy', False, num_stages)
        outcome = 'ok'
    except ValueError as e:
        outcome = str(e)
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump({'outcome': outcome, 'printed': printed.getvalue()}, f)


@pytest.mark.parametrize('lrs, num_stages, refused, warned', [
    ([[1.0], [1.0]], 2, False, False),
    # Each stage alone has one learning rate, but not the same one.
    ([[1.0], [0.5]], 2, True, False),
    # Data parallel: the ranks hold the same model.
    ([[1.0, 0.5], [1.0, 0.5]], 1, False, True),
])
def test_prodigy_with_per_group_learning_rates(lrs, num_stages, refused, warned, tmp_path):
    mp.spawn(_learning_rate_worker, args=(lrs, num_stages, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    for rank, result in enumerate(results):
        assert ('pipeline_stages = 2' in result['outcome']) == refused, result['outcome']
        assert ('not applied' in result['printed']) == (warned and rank == 0)
