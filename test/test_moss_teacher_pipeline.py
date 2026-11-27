"""MOSS-Image's [teacher] on several pipeline stages: the teacher pipeline (model_teacher.TeacherPipeline) over
the frozen copy's layers, with the velocity delivered to the loss through the label."""

import json

import pytest
import torch
import torch.multiprocessing as mp

from test.test_distillation import cfg, distilling_pipeline, label_with_sigma
from test.test_moss_image import forward_inputs, loaded_pipeline, reference_forward, single_process  # noqa: F401

GAS = 2


def _perturb(pipe):
    """The student drifts from its starting model, deterministically."""
    torch.manual_seed(3)
    with torch.no_grad():
        for p in pipe.diffusion_model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))


def _micro_batches():
    items = []
    for seed in range(GAS):
        features = forward_inputs(seed=2 + seed)
        torch.manual_seed(10 + seed)
        target = torch.randn(2, 128, 3, 5)
        items.append((features, (target, features[1], torch.empty(0))))
    return items


@pytest.mark.parametrize('lora', [False, True], ids=['full fine-tune', 'adapter'])
def test_the_teacher_layers_give_the_starting_models_velocity_and_the_loss_reads_it_from_the_label(lora, single_process):
    pipe, reference = distilling_pipeline(lora=lora, stages=2)
    assert pipe.teacher_in_pipeline() and pipe.own_teacher_config() is pipe.distill
    # The student's layers carry no teacher: it is a pipeline of its own.
    assert pipe.build_teacher() is None and not pipe.pipeline_model.layers[0].teacher
    if lora:
        # what load_diffusion_model keeps with several stages, adapter or not
        pipe.teacher_snapshot = loaded_pipeline(reference).diffusion_model
    layers = pipe.make_teacher_layers()
    assert pipe.teacher_snapshot is None and type(layers[0]).__name__ == 'TextEncoderLayer' and not layers[0].teacher
    features, label = _micro_batches()[0]
    x = features
    with torch.no_grad():
        for layer in layers:
            x = layer(x)
        expected = reference_forward(reference, *features).float()
    torch.testing.assert_close(x.float(), expected, rtol=1e-5, atol=1e-5)
    assert not any(p.requires_grad for layer in layers for p in layer.parameters())

    # The loss: a teacher velocity in the label's fourth entry is the one the carried form reads.
    pipe.training_step = 1
    prediction = torch.randn(2, 128, 3, 5)
    target, sigma, mask = label
    loss_fn = pipe.get_loss_fn()
    velocity = torch.randn_like(prediction)
    via_label = loss_fn(prediction, (target, sigma, mask, velocity))
    carried = loss_fn((prediction, velocity), (target, sigma, mask))
    torch.testing.assert_close(via_label, carried)
    without = loss_fn(prediction, (target, sigma, mask, torch.zeros_like(prediction)))
    assert not torch.allclose(via_label, without)


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

    distill = cfg(loss_weight=1.0, **({'decay': 'linear', 'decay_steps': 5} if finished else {}))
    pipe, _ = distilling_pipeline(distill=distill, stages=num_stages)
    _perturb(pipe)
    pipe.training_step = 100 if finished else 1
    teacher = model_teacher.build_teacher(pipe, distill, torch.device('cpu'), place=False)
    teacher.set_step(100 if finished else 1)

    student = ManualPipelineModule(layers=pipe.pipeline_model.layers, num_stages=num_stages,
                                   partition_method='uniform', loss_fn=pipe.get_loss_fn(), dynamic_shape=True)
    params = [p for p in student.parameters() if p.requires_grad]
    ds_config = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': GAS,
                 'gradient_clipping': 0.0, 'steps_per_print': 10**9}
    engine, *_ = deepspeed.initialize(model=student, model_parameters=params,
                                      optimizer=torch.optim.SGD(params, lr=0.0), config=ds_config)
    engine._support_torch_style_backward = True
    teacher_module = ManualPipelineModule(layers=teacher.layers, num_stages=num_stages, partition_method='uniform',
                                          loss_fn=None, dynamic_shape=True)
    teacher_engine, *_ = deepspeed.initialize(model=teacher_module, config=dict(ds_config))
    teacher_pipeline = model_teacher.TeacherPipeline(
        teacher, teacher_engine, hooks=model_teacher.LabelHooks(pipe.teacher_label_targets, pipe.teacher_output_float32))
    items = _micro_batches() if engine.is_first_stage() or engine.is_last_stage() else None
    attached = teacher_pipeline.attach(items)
    loss = engine.train_batch(iter(attached) if attached is not None else None)
    result = {'stage': engine.stage_id, 'loss': loss.item()}
    if engine.is_last_stage():
        result['label_len'] = [len(label) for _, label in attached]
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


def _expected(finished):
    distill = cfg(loss_weight=1.0, **({'decay': 'linear', 'decay_steps': 5} if finished else {}))
    pipe, _ = distilling_pipeline(distill=distill)
    _perturb(pipe)
    pipe.training_step = 100 if finished else 1
    loss_fn = pipe.get_loss_fn()
    losses = []
    for features, label in _micro_batches():
        out = pipe.pipeline_model(features)
        losses.append(loss_fn(out, label).item())
    return sum(losses) / len(losses)


@pytest.mark.parametrize('finished', [False, True], ids=['on', 'decayed'])
def test_two_stage_loss_is_the_one_stage_loss_with_the_teacher_in_its_own_pipeline(finished, tmp_path):
    mp.spawn(_worker, args=(2, 2, finished, str(tmp_path / 'init'), str(tmp_path)), nprocs=2)
    results = [json.loads((tmp_path / f'{r}.json').read_text()) for r in range(2)]
    expected = _expected(finished)
    last = [r for r in results if r['stage'] == 1][0]
    assert last['label_len'] == [4] * GAS
    for r in results:
        assert r['loss'] == pytest.approx(expected, rel=1e-4, abs=1e-5)
