"""anima_refiner's stock-Anima [teacher] with pipeline_stages > 1: a teacher pipeline of the stock Anima's own layers,
the mixing of its velocity into the target on the last stage (models/teacher_guidance.py AnimaLabelHooks)."""

import json

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

pytest.importorskip('models.cosmos_predict2')

from models.teacher_guidance import AnimaLabelHooks  # noqa: E402
from test.test_teacher_guidance_training import (  # noqa: E402
    _NullOffloader, build_layers, build_pipeline, build_student_and_teacher, teacher_cfg, toy_batch)
from test.test_teacher_guidance import _FakeTokenizer  # noqa: E402

GAS = 2


def _batch():
    batch = toy_batch(batch_size=2 * GAS)
    batch['mask'] = torch.ones(2 * GAS, 16, 16)
    return batch


def _split(features, label):
    chunks = []
    for i in range(GAS):
        part = slice(2 * i, 2 * i + 2)
        chunks.append((tuple(t[part] for t in features), tuple(t[part] for t in label)))
    return chunks


def _pipe(stages):
    student, student_llm, guide = build_student_and_teacher()
    # Another vocabulary for the T5 side, so the two token pairs differ; a lambda that depends on the step
    guide.t5_tokenizer = _FakeTokenizer(vocab=17)
    pipe = build_pipeline(teacher=guide, cfg=teacher_cfg(decay='linear', decay_steps=10), step=4)
    pipe.config['pipeline_stages'] = stages
    pipe.offloader = _NullOffloader()
    return pipe, student, student_llm, guide


def _chain_velocity(pipe, hooks, features, label):
    x = hooks.inputs(features, label)
    with torch.no_grad():
        for layer in pipe.make_teacher_layers():
            x = layer(x)
    return x


def test_the_label_carries_t_and_the_teachers_tokens_only_with_several_stages_and_only_in_training():
    pipe, *_ = _pipe(2)
    assert pipe.teacher_in_pipeline() and pipe.own_teacher_config() is pipe.teacher_cfg
    torch.manual_seed(7)
    _, label = pipe.prepare_inputs(_batch())
    target, t, *tokens, mask = label
    assert len(tokens) == 4 and t.shape == (2 * GAS, 1) and target.shape[0] == 2 * GAS
    assert all(not torch.is_floating_point(x) for x in tokens)
    _, evaluated = pipe.prepare_inputs(_batch(), timestep_quantile=0.5)
    assert len(evaluated) == 2
    one, *_ = _pipe(1)
    assert not one.teacher_in_pipeline() and one.own_teacher_config() is None
    _, label = one.prepare_inputs(_batch())
    assert len(label) == 2


def test_the_teacher_layers_and_the_last_stages_mix_give_the_single_stage_target():
    single, *_ = _pipe(1)
    torch.manual_seed(7)
    _, (expected, _) = single.prepare_inputs(_batch())

    pipe, *_ = _pipe(2)
    torch.manual_seed(7)
    features, label = pipe.prepare_inputs(_batch())
    hooks = pipe.teacher_label_hooks()
    velocity = _chain_velocity(pipe, hooks, features, label)
    mixed, mask = hooks.with_teacher(label, [velocity])
    torch.testing.assert_close(mixed, expected, rtol=1e-4, atol=1e-5)
    assert mask is label[-1] and hooks.without_teacher(label)[0] is label[0]
    # the teacher's weights are frozen and it is eval-mode
    assert not any(p.requires_grad for layer in pipe.make_teacher_layers() for p in layer.parameters())


def _worker(rank, world_size, finished, init_file, out_dir):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world_size, verbose=False)
    from utils import model_teacher
    from utils.patches import apply_patches
    from utils.pipeline import ManualPipelineModule
    apply_patches()

    pipe, student, student_llm, guide = _pipe(world_size)
    if finished:
        pipe.teacher_cfg = teacher_cfg(decay='linear', decay_steps=10)
        pipe._step_source = lambda: 100
    teacher = model_teacher.build_teacher(pipe, pipe.own_teacher_config(), torch.device('cpu'), place=False)
    teacher.set_step(100 if finished else 4)
    teacher_layers = list(teacher.layers)
    hooks = pipe.teacher_label_hooks()

    layers = build_layers(student, student_llm)
    student_module = ManualPipelineModule(layers=layers, num_stages=world_size, partition_method='uniform',
                                          loss_fn=lambda out, label: F.mse_loss(out, label[0]), dynamic_shape=True)
    params = [p for p in student_module.parameters() if p.requires_grad]
    ds_config = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': GAS,
                 'gradient_clipping': 0.0, 'steps_per_print': 10**9}
    engine, *_ = deepspeed.initialize(model=student_module, model_parameters=params,
                                      optimizer=torch.optim.SGD(params, lr=0.0), config=ds_config)
    engine._support_torch_style_backward = True

    method, split = model_teacher.teacher_partition(teacher.layers, world_size, 'parameters', None)
    teacher_module = ManualPipelineModule(layers=teacher.layers, num_stages=world_size, partition_method=method,
                                          manual_partition_split=split, loss_fn=None, dynamic_shape=True)
    teacher_engine, *_ = deepspeed.initialize(model=teacher_module, config=dict(ds_config))
    model_teacher.release_foreign_weights(teacher_layers, teacher_engine.module)
    teacher_pipeline = model_teacher.TeacherPipeline(teacher, teacher_engine, hooks=hooks)

    torch.manual_seed(7)
    features, label = pipe.prepare_inputs(_batch())
    items = _split(features, label) if engine.is_first_stage() or engine.is_last_stage() else None
    attached = teacher_pipeline.attach(items)
    result = {'stage': engine.stage_id}
    if engine.is_last_stage():
        result['label_len'] = [len(lab) for _, lab in attached]
    loss = engine.train_batch(iter(attached) if attached is not None else None)
    result['loss'] = loss.item()
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


def _expected(finished):
    pipe, student, student_llm, guide = _pipe(1)
    if finished:
        pipe.teacher_cfg = teacher_cfg(decay='linear', decay_steps=10)
        pipe._step_source = lambda: 100
    torch.manual_seed(7)
    features, (target, mask) = pipe.prepare_inputs(_batch())
    layers = build_layers(student, student_llm)
    losses = []
    for (f, _), start in zip(_split(features, (target, mask)), range(0, 2 * GAS, 2)):
        x = f
        with torch.no_grad():
            for layer in layers:
                x = layer(x)
        losses.append(F.mse_loss(x, target[start:start + 2]).item())
    return sum(losses) / len(losses)


@pytest.mark.parametrize('finished', [False, True], ids=['on', 'decayed'])
def test_two_stage_loss_is_the_single_stage_loss_with_the_teacher_in_its_own_pipeline(finished, tmp_path):
    mp.spawn(_worker, args=(2, finished, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{r}.json').read_text()) for r in range(2)]
    expected = _expected(finished)
    assert [r for r in results if r['stage'] == 1][0]['label_len'] == [2] * GAS
    for r in results:
        assert r['loss'] == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_the_generic_teacher_of_an_anima_pipeline_uses_the_plain_label_hooks():
    from utils import model_teacher
    pipe, *_ = _pipe(2)
    assert isinstance(model_teacher.label_hooks_for(pipe, own_teacher=True), AnimaLabelHooks)
    # [teacher] without transformer_path / llm_path is the generic one, whatever the model
    generic = model_teacher.label_hooks_for(pipe, own_teacher=False)
    assert type(generic) is model_teacher.LabelHooks
    label = (torch.zeros(2, 1), torch.ones(2))
    features = (torch.zeros(1),)
    assert generic.inputs(features, label) is features
    assert len(generic.targets(label)) == 1 and generic.targets(label)[0] is label[0]
    # with the stock teacher off (one stage) the model gives no hooks of its own
    one, *_ = _pipe(1)
    assert one.teacher_label_hooks() is None
    assert type(model_teacher.label_hooks_for(one, own_teacher=True)) is model_teacher.LabelHooks


def test_the_deep_copy_for_the_teacher_pipeline_leaves_the_students_refiner_out():
    from models.cosmos_predict2 import _copy_without_refiner
    student, *_ = build_student_and_teacher()
    refiner = student.context_refiner
    copy_of = _copy_without_refiner(student)
    assert student.context_refiner is refiner and copy_of.context_refiner is None
    assert copy_of.blocks[0] is not student.blocks[0]
    torch.testing.assert_close(next(copy_of.blocks[0].parameters()), next(student.blocks[0].parameters()))


def test_the_hooks_take_lambdas_step_from_the_teacher_the_pipeline_decides_finished_from():
    from utils import model_teacher
    pipe, *_ = _pipe(2)
    hooks = pipe.teacher_label_hooks()
    teacher = model_teacher.ModelTeacher(model_teacher.ModelTeacherConfig(1.0, 'linear', 10, None))
    model_teacher.TeacherPipeline(teacher, engine=None, hooks=hooks)
    teacher.set_step(7)
    assert hooks.current_step() == 7
