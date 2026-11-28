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


def test_the_gpus_of_a_node_come_from_either_launchers_variable(monkeypatch):
    monkeypatch.delenv('LOCAL_SIZE', raising=False)
    monkeypatch.delenv('LOCAL_WORLD_SIZE', raising=False)
    assert placement.local_world_size() == 0
    monkeypatch.setenv('LOCAL_SIZE', '8')
    assert placement.local_world_size() == 8
    monkeypatch.delenv('LOCAL_SIZE')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '4')
    assert placement.local_world_size() == 4


@pytest.mark.parametrize('how', ['keyword', 'positional'])
def test_the_placement_applies_whether_the_stages_are_given_by_keyword_or_position(how, monkeypatch):
    import utils.pipeline as pipeline_module
    seen = {}
    monkeypatch.setattr(pipeline_module.PipelineModule, '__init__', lambda self, *a, **kw: seen.update(kw))
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda *a, **k: 8)
    if how == 'keyword':
        pipeline_module.ManualPipelineModule(layers=[], num_stages=4, placement='node_local')
    else:
        pipeline_module.ManualPipelineModule([], 4, placement='node_local')
    assert seen['topology'].get_dim('pipe') == 4 and seen['topology'].axes == ['data', 'pipe']
    seen.clear()
    pipeline_module.ManualPipelineModule(layers=[], num_stages=4, placement='stage_major')
    assert 'topology' not in seen


def _relay_replicas_worker(rank, world, stages, placement_name, init_file, out_dir):
    """The text encoder relay and the label broadcast on a real engine, every replica with batches of its own."""
    import os
    import sys
    from pathlib import Path
    import deepspeed
    sys.path.insert(0, str(Path(__file__).parent))
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world, verbose=False)
    from utils.patches import apply_patches
    apply_patches()
    import utils.dataset as dataset_util
    from test_text_encoding_on_the_fly import CONTROL_FILE_KEY, IS_VIDEO_KEY, Model, TextEncodingOnTheFly
    engine = _engine(placement_name, stages)
    replica = engine.grid.get_data_parallel_rank()
    batches = [{'caption': [chr(97 + replica) * 3, chr(100 + replica) * 3], IS_VIDEO_KEY: [False, False],
                CONTROL_FILE_KEY: [None, None]} for _ in range(2)]
    model = Model(cache_text_embeddings=False)

    class Prepared:
        def prepare_inputs(self, batch, timestep_quantile=None):
            return (batch['emb1'],), (torch.full((2,), float(replica)), None)

    loader = dataset_util.PipelineDataLoader(batches, engine, 1, Prepared(), num_dataloader_workers=0,
                                             text_encoding=TextEncodingOnTheFly(model))
    result = {'stage': engine.stage_id, 'replica': replica, 'calls': len(model.calls), 'features': [], 'targets': []}
    if engine.is_first_stage() or engine.is_last_stage():
        for _ in batches:
            features, label = next(loader)
            result['features'].append(features[0].float().tolist())
            result['targets'].append(label[0].tolist())
    result['calls'] = len(model.calls)
    import json
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


@pytest.mark.parametrize('name', ['node_local', 'stage_major'])
def test_the_relay_pairs_each_replicas_first_and_last_stage_under_either_placement(name, tmp_path):
    mp.spawn(_relay_replicas_worker, args=(4, 2, name, str(tmp_path / 'init'), str(tmp_path)), nprocs=4)
    results = [json.loads((tmp_path / f'{r}.json').read_text()) for r in range(4)]
    for replica in (0, 1):
        first = next(r for r in results if r['replica'] == replica and r['stage'] == 0)
        last = next(r for r in results if r['replica'] == replica and r['stage'] == 1)
        assert first['calls'] > 0 and last['calls'] == 0
        assert first['features'] and last['features'] == first['features']
        assert last['targets'] == first['targets'] == [[float(replica)] * 2] * 2
    # the replicas really had different captions, so a crossed pair would have shown
    assert next(r for r in results if r['replica'] == 0 and r['stage'] == 0)['features'] != \
        next(r for r in results if r['replica'] == 1 and r['stage'] == 0)['features']


def _overwritten_imports(source):
    import ast
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'utils':
            imported.update(a.asname or a.name for a in node.names)
    assigned = {}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.NamedExpr)):
            targets = [node.target]
        for target in targets:
            for part in ([target] if isinstance(target, ast.Name) else getattr(target, 'elts', [])):
                if isinstance(part, ast.Name) and part.id in imported:
                    assigned.setdefault(part.id, []).append(part.lineno)
    return imported, assigned


def test_train_py_does_not_overwrite_a_name_it_imports_from_utils():
    # train.py is a script, nothing runs it in the tests: a local variable called like an imported function
    # (local_world_size, the GPUs per node) turned the call into "'int' object is not callable" on rank 0 at startup.
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / 'train.py').read_text()
    imported, assigned = _overwritten_imports(source)
    assert {'gpus_per_node', 'check_placement', 'memory_partition'} <= imported
    assert not assigned, assigned
    # the check sees the mistake it was written for
    broken = source.replace('gpus_per_node', 'local_world_size')
    assert 'local_world_size' in _overwritten_imports(broken)[1]
