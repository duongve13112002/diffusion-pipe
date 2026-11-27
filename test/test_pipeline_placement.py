"""pipeline_placement: which global ranks make one pipeline (utils/placement.py), in real gloo process groups."""

import json

import pytest
import torch
import torch.multiprocessing as mp

from utils import placement

LAYERS = 4
WIDTH = 4
GAS = 2


def _layers():
    torch.manual_seed(0)
    return [torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(LAYERS)]


def _data():
    g = torch.Generator().manual_seed(3)
    return [(torch.randn(2, WIDTH, generator=g), (torch.randn(2, WIDTH, generator=g), torch.ones(1)))
            for _ in range(GAS)]


def _loss(output, label):
    return torch.nn.functional.mse_loss(output, label[0])


def _engine(placement_name, stages):
    import deepspeed
    from utils.pipeline import ManualPipelineModule
    pipe = ManualPipelineModule(layers=_layers(), num_stages=stages, placement=placement_name,
                                partition_method='uniform', loss_fn=_loss, dynamic_shape=True)
    params = list(pipe.parameters())
    config = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': GAS,
              'gradient_clipping': 0.0, 'steps_per_print': 10**9}
    engine, *_ = deepspeed.initialize(model=pipe, model_parameters=params,
                                      optimizer=torch.optim.SGD(params, lr=0.1), config=config)
    engine._support_torch_style_backward = True
    return engine


def _worker(rank, world, stages, placement_name, init_file, out_dir, ckpt_dir, mode):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world, verbose=False)
    from utils.patches import apply_patches
    apply_patches()
    engine = _engine(placement_name, stages)
    grid = engine.grid
    result = {'stage': engine.stage_id, 'data': grid.get_data_parallel_rank(),
              'stage0': grid.stage_to_global(0), 'last': grid.stage_to_global(stages - 1),
              'layers': [i for i, _ in enumerate(engine.module.forward_funcs)]}
    data = _data() if engine.is_first_stage() or engine.is_last_stage() else None
    if mode == 'resume':
        engine.load_checkpoint(ckpt_dir, tag='t', load_optimizer_states=True)
    loss = engine.train_batch(iter(data) if data is not None else None)
    result['loss'] = loss.item()
    result['weights'] = [p.detach().flatten().tolist() for p in engine.module.parameters()]
    if mode == 'save':
        engine.save_checkpoint(ckpt_dir, tag='t')
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


def _run(tmp_path, world, stages, placement_name, name, mode='plain', ckpt=None):
    out = tmp_path / name
    out.mkdir()
    mp.spawn(_worker, args=(world, stages, placement_name, str(tmp_path / f'init_{name}'), str(out),
                            str(ckpt or tmp_path / 'ckpt'), mode), nprocs=world)
    return [json.loads((out / f'{r}.json').read_text()) for r in range(world)]


def test_node_local_puts_the_stages_of_a_pipeline_on_consecutive_ranks():
    assert [placement.stage_of_rank(r, 8, 4, 'node_local') for r in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [placement.data_rank_of_rank(r, 8, 4, 'node_local') for r in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [placement.stage_of_rank(r, 8, 4, 'stage_major') for r in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [placement.data_rank_of_rank(r, 8, 4, 'stage_major') for r in range(8)] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert placement.stage_of_rank(5, 8, 1) == 0 and placement.data_rank_of_rank(5, 8, 1) == 5


def test_a_bad_placement_is_refused():
    with pytest.raises(ValueError, match='pipeline_placement'):
        placement.check_placement('node-local')


def test_two_nodes_of_eight_with_four_stages_keep_every_pipeline_on_one_node():
    topology = placement.pipeline_topology(4, 16, 'node_local')
    for replica in range(4):
        ranks = [topology.get_rank(data=replica, pipe=stage) for stage in range(4)]
        assert len({r // 8 for r in ranks}) == 1
    assert placement.pipeline_topology(4, 16, 'stage_major') is None
    with pytest.raises(RuntimeError, match='divide'):
        placement.pipeline_topology(3, 16, 'node_local')


def test_a_pipeline_that_does_not_fit_a_node_evenly_is_warned_about():
    assert placement.straddling_note(4, 16, 8) is None
    assert placement.straddling_note(16, 16, 8) is None
    assert placement.straddling_note(4, 4, 4) is None
    assert 'cross a node boundary' in placement.straddling_note(3, 24, 8)


@pytest.mark.parametrize('name', ['node_local', 'stage_major'])
def test_ranks_hold_the_stage_the_placement_says_and_training_gives_the_same_loss(name, tmp_path):
    results = _run(tmp_path, 4, 2, name, name)
    for rank, r in enumerate(results):
        assert r['stage'] == placement.stage_of_rank(rank, 4, 2, name)
        assert r['data'] == placement.data_rank_of_rank(rank, 4, 2, name)
    firsts = sorted(r['stage0'] for r in results)
    lasts = sorted(r['last'] for r in results)
    assert firsts == ([0, 0, 2, 2] if name == 'node_local' else [0, 0, 1, 1])
    assert lasts == ([1, 1, 3, 3] if name == 'node_local' else [2, 2, 3, 3])
    other = _run(tmp_path, 4, 2, 'stage_major' if name == 'node_local' else 'node_local', 'other')
    assert results[0]['loss'] == pytest.approx(other[0]['loss'], abs=1e-6)


def test_a_checkpoint_saved_with_one_placement_resumes_with_the_other(tmp_path):
    saved = _run(tmp_path, 4, 2, 'stage_major', 'saved', mode='save')
    resumed = _run(tmp_path, 4, 2, 'node_local', 'resumed', mode='resume', ckpt=tmp_path / 'ckpt')
    again = _run(tmp_path, 4, 2, 'stage_major', 'again', mode='resume', ckpt=tmp_path / 'ckpt')
    # a stage's weights are the same wherever it ran, so each (stage, data rank) matches across the placements
    def by_stage(results):
        return {(r['stage'], r['data']): r['weights'] for r in results}
    for key, weights in by_stage(resumed).items():
        assert weights == by_stage(again)[key]
    assert all(r['loss'] == pytest.approx(resumed[0]['loss'], abs=1e-6) for r in resumed)
