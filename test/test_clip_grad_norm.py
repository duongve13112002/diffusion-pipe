"""utils.patches.clip_grad_norm_ under DeepSpeed's pipeline grid, in real gloo process groups.

DeepSpeed sums the squared gradient norms over the grid's model-parallel group, which for a
pipeline is the whole pipeline, but its copy counted only that group's rank 0 (stage 0): with
pipeline_stages > 1 every later stage's gradients were left out of the norm that clipping uses.
"""

import json
import math

import pytest
import torch
import torch.multiprocessing as mp


def _clip_worker(rank, world_size, num_pp, init_file, out_dir):
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world_size, verbose=False)
    from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipelineParallelGrid
    from deepspeed.utils import groups
    from utils.patches import clip_grad_norm_

    grid = PipelineParallelGrid(topology=PipeDataParallelTopology(num_pp=num_pp, num_dp=world_size // num_pp))
    groups.mpu = grid
    stage = grid.get_stage_id()

    def param(value, size=4):
        p = torch.nn.Parameter(torch.zeros(size))
        p.grad = torch.full((size,), float(value))
        return p

    # Each stage's own parameter; data-parallel replicas hold the same gradients, as after
    # DeepSpeed's all-reduce.
    params = [param(stage + 1)]
    # A frozen parameter has no gradient and is left out.
    params.append(torch.nn.Parameter(torch.zeros(3), requires_grad=False))
    # A tied layer on every stage: stage 0's copy counts, DeepSpeed marks the others.
    tied = param(5, size=1)
    if stage > 0:
        tied.ds_pipe_replicated = True
    params.append(tied)

    norm = clip_grad_norm_(params, max_norm=1.0, mpu=grid)
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump({'stage': stage, 'norm': norm.item(),
                   'grads': [p.grad.tolist() for p in params if p.grad is not None]}, f)


@pytest.mark.parametrize('world_size, num_pp', [(2, 2), (4, 2), (2, 1)], ids=['pp2', 'pp2-dp2', 'dp2'])
def test_the_clipping_norm_covers_every_pipeline_stage(world_size, num_pp, tmp_path):
    mp.spawn(_clip_worker, args=(world_size, num_pp, str(tmp_path / 'init'), str(tmp_path)), nprocs=world_size)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(world_size)]
    expected = math.sqrt(sum(4 * (stage + 1) ** 2 for stage in range(num_pp)) + 5 ** 2)
    coef = 1.0 / (expected + 1e-6)
    for result in results:
        assert result['norm'] == pytest.approx(expected, rel=1e-6)
        stage = result['stage']
        # Every stage is scaled by the one coefficient of the whole model.
        assert result['grads'] == [pytest.approx([(stage + 1) * coef] * 4, rel=1e-5), pytest.approx([5 * coef], rel=1e-5)]
