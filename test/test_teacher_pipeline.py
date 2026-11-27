"""model_teacher.TeacherPipeline: the teacher as a second forward-only DeepSpeed pipeline engine, in real gloo
process groups on CPU. Its output on the last stage must be what the teacher's chain computes on one process, and
the student's step must see it in the loss."""

import json

import pytest
import torch
import torch.multiprocessing as mp

LAYERS = 4
WIDTH = 4
GAS = 3
LAM = 0.5


def _chain(seed):
    torch.manual_seed(seed)
    return [torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(LAYERS)]


def _micro_batches():
    g = torch.Generator().manual_seed(7)
    return [(torch.randn(2, WIDTH, generator=g), (torch.randn(2, WIDTH, generator=g), torch.ones(1)))
            for _ in range(GAS)]


def _inner(output, label):
    return torch.nn.functional.mse_loss(output, label[0])


def _worker(rank, world_size, num_stages, finished, init_file, out_dir):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world_size, verbose=False)
    from utils.patches import apply_patches
    apply_patches()
    from utils import model_teacher
    from utils.pipeline import ManualPipelineModule

    config = model_teacher.ModelTeacherConfig(LAM, 'none' if not finished else 'linear', 10, None)
    teacher = model_teacher.ModelTeacher(config, select=lambda out: (out,), device=torch.device('cpu'))
    teacher.set_step(100 if finished else 1)
    for layer in (teacher_layers := _chain(1)):
        layer.requires_grad_(False)
        layer.eval()
    teacher.layers = teacher_layers

    pipe = ManualPipelineModule(layers=_chain(0), num_stages=num_stages, partition_method='uniform',
                                loss_fn=model_teacher.wrap_loss_fn(_inner, teacher), dynamic_shape=True)
    params = list(pipe.parameters())
    ds_config = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': GAS,
                 'gradient_clipping': 0.0, 'steps_per_print': 10**9}
    engine, *_ = deepspeed.initialize(model=pipe, model_parameters=params,
                                      optimizer=torch.optim.SGD(params, lr=0.0), config=ds_config)
    engine._support_torch_style_backward = True
    # The student's default method weighs trainable parameters, which a frozen teacher has none of.
    method, split = model_teacher.teacher_partition(teacher.layers, num_stages, 'parameters', None)
    teacher_layers = list(teacher.layers)
    teacher_module = ManualPipelineModule(layers=teacher.layers, num_stages=num_stages, partition_method=method,
                                          manual_partition_split=split, loss_fn=None, dynamic_shape=True)
    parts = list(teacher_module.parts)
    teacher_engine, *_ = deepspeed.initialize(model=teacher_module, config=dict(ds_config))
    teacher_engine._support_torch_style_backward = True
    teacher_pipeline = model_teacher.TeacherPipeline(teacher, teacher_engine)

    model_teacher.release_foreign_weights(teacher_layers, teacher_engine.module)
    items = _micro_batches() if engine.is_first_stage() or engine.is_last_stage() else None
    attached = teacher_pipeline.attach(items)
    result = {'stage': engine.stage_id, 'parts': parts,
              'emptied': sum(1 for layer in teacher_layers if layer.weight.numel() == 0)}
    # Another step with micro batches of another shape: the engine keeps no totals from the first.
    wide = [(torch.randn(3, WIDTH), (torch.randn(3, WIDTH), torch.ones(1))) for _ in range(GAS)]
    again = teacher_pipeline.attach(wide if items is not None else None)
    if engine.is_last_stage():
        result['wide'] = [tuple(label[2].shape) for _, label in again]
    if engine.is_last_stage():
        result['teacher_targets'] = [[t.tolist() for t in label[2:]] for _, label in attached]
        result['label_len'] = [len(label) for _, label in attached]
    else:
        assert attached is items
    loss = engine.train_batch(iter(attached) if attached is not None else None)
    result['loss'] = loss.item()
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


def _expected():
    student, teacher = torch.nn.Sequential(*_chain(0)), torch.nn.Sequential(*_chain(1))
    targets, losses = [], []
    for x, label in _micro_batches():
        with torch.no_grad():
            t = teacher(x)
            out = student(x)
        targets.append(t.tolist())
        losses.append(_inner(out, label).item() + LAM * torch.nn.functional.mse_loss(out, t).item())
    return targets, sum(losses) / len(losses)


@pytest.mark.parametrize('world, stages', [(2, 2), (3, 3), (4, 2)], ids=['pp2', 'pp3', 'pp2-dp2'])
def test_the_last_stage_gets_the_teachers_output_and_the_loss_uses_it(world, stages, tmp_path):
    mp.spawn(_worker, args=(world, stages, False, str(tmp_path / 'init'), str(tmp_path)), nprocs=world)
    results = [json.loads((tmp_path / f'{r}.json').read_text()) for r in range(world)]
    targets, loss = _expected()
    last = [r for r in results if r['stage'] == stages - 1]
    assert len(last) == world // stages
    for r in last:
        assert r['label_len'] == [3] * GAS
        for got, want in zip(r['teacher_targets'], targets):
            assert got[0] == [pytest.approx(row, abs=1e-5) for row in want]
    for r in results:
        assert r['loss'] == pytest.approx(loss, abs=1e-5)
        # balanced by parameter counts, not piled on stage 0, and the other stages' weights are freed
        sizes = [b - a for a, b in zip(r['parts'], r['parts'][1:])]
        assert min(sizes) >= 1 and max(sizes) <= -(-LAYERS // stages)
        assert r['emptied'] == LAYERS - sizes[r['stage']]
    for r in last:
        assert r['wide'] == [[3, WIDTH]] * GAS


def test_a_finished_teacher_attaches_zeros_and_the_engine_does_not_run(tmp_path):
    mp.spawn(_worker, args=(2, 2, True, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{r}.json').read_text()) for r in range(2)]
    last = [r for r in results if r['stage'] == 1][0]
    assert last['label_len'] == [3] * GAS
    assert all(all(v == 0 for row in t[0] for v in row) for t in last['teacher_targets'])
