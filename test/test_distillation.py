"""[teacher] distillation for MOSS-Image: utils/distillation.py (config and loss) and the teacher forward
that rides the pipeline layers of models/moss_image.py (CPU, a tiny model)."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models import moss_image as mi
from test.test_moss_image import (LORA, Runner, bare_pipeline, caption_batch, forward_inputs,  # noqa: F401
                                  loaded_pipeline, on_the_fly_pipelines, reference_forward, single_process,
                                  tiny_reference, trained_lora_pipeline)
from test.test_stream_loss import ALL, spec_of
from utils.distillation import (DistillConfig, distill_loss, distill_warnings, elementwise_loss,
                                parse_distill_config)
from utils.stream_loss import StreamLayout, StreamMetrics


def cfg(**table):
    config = parse_distill_config({'teacher': {'loss_weight': 0.5, **table}})
    assert config is not None
    return config


# configuration

def test_absent_or_zero_weight_is_off():
    assert parse_distill_config({}) is None
    assert parse_distill_config({'teacher': {}}) is None
    assert parse_distill_config({'teacher': {'loss_weight': 0}}) is None
    # An off table is not validated further.
    assert parse_distill_config({'teacher': {'loss_weight': 0, 'decay': 'linear'}}) is None


def test_defaults():
    config = cfg()
    assert (config.loss_weight, config.shape, config.decay, config.decay_steps) == (0.5, 'constant', 'none', 0)
    assert config.enabled
    assert isinstance(config, DistillConfig)


@pytest.mark.parametrize('table, match', [
    ({'loss_weight': -1}, 'loss_weight'),
    ({'loss_weight': True}, 'loss_weight'),
    ({'loss_weight': float('nan')}, 'loss_weight'),
    ({'loss_weight': '1'}, 'loss_weight'),
    ({'transformer_path': 'x.safetensors'}, 'unknown keys'),
    ({'shape': 'ramp'}, 'shape'),
    ({'t_mid': 1.5}, 't_mid'),
    ({'width': 0}, 'width'),
    ({'decay': 'exp'}, 'decay'),
    ({'decay': 'linear'}, 'decay_steps'),
    ({'decay': 'cosine', 'decay_steps': 0}, 'decay_steps'),
    ({'decay_steps': -3}, 'decay_steps'),
    ({'decay_steps': 2.5}, 'decay_steps'),
])
def test_bad_tables_are_refused(table, match):
    with pytest.raises(ValueError, match=match):
        parse_distill_config({'teacher': {'loss_weight': 0.5, **table}})
    with pytest.raises(ValueError, match='must be a table'):
        parse_distill_config({'teacher': 3})


def test_settings_distillation_cannot_run_with():
    table = {'loss_weight': 0.5}
    with pytest.raises(ValueError, match='compile'):
        parse_distill_config({'teacher': table, 'compile': True})
    with pytest.raises(ValueError, match='blocks_to_swap'):
        parse_distill_config({'teacher': table, 'blocks_to_swap': 4})
    with pytest.raises(ValueError, match='pipeline_stages = 1'):
        parse_distill_config({'teacher': table, 'adapter': {'type': 'lora'}, 'pipeline_stages': 2})
    # An adapter on one stage, and a full fine-tune on several, are the supported shapes.
    assert parse_distill_config({'teacher': table, 'adapter': {'type': 'lora'}, 'pipeline_stages': 1})
    assert parse_distill_config({'teacher': table, 'pipeline_stages': 2})


def test_a_teacher_that_never_fades_is_warned_about():
    assert 'never fades' in distill_warnings(cfg(), {})[0]
    assert distill_warnings(cfg(decay='linear', decay_steps=10), {}) == []


# the loss term

def test_the_term_is_lambda_times_the_data_loss_form():
    torch.manual_seed(0)
    output, teacher = torch.randn(3, 4, 2, 2), torch.randn(3, 4, 2, 2)
    sigma = torch.tensor([0.2, 0.5, 0.9])
    config = cfg(loss_weight=2.0)
    term, mean_lambda = distill_loss(output, teacher, sigma, torch.empty(0), 1, config, {})
    assert float(term) == pytest.approx(2.0 * float(F.mse_loss(output, teacher)), rel=1e-6)
    assert float(mean_lambda) == pytest.approx(2.0)
    # huber and smooth l1 follow the data loss's setting
    for key, value, expected in (('huber_delta', 0.5, F.huber_loss(output, teacher, delta=0.5)),
                                 ('smooth_l1_beta', 0.3, F.smooth_l1_loss(output, teacher, beta=0.3))):
        term, _ = distill_loss(output, teacher, sigma, torch.empty(0), 1, config, {key: value})
        assert float(term) == pytest.approx(2.0 * float(expected), rel=1e-6)
    assert elementwise_loss(output, teacher, {}).shape == output.shape


def test_per_sample_weights_follow_sigma_and_the_mask_applies():
    torch.manual_seed(1)
    output, teacher = torch.randn(3, 2, 2, 2), torch.randn(3, 2, 2, 2)
    sigma = torch.tensor([0.1, 0.5, 0.9])
    config = cfg(loss_weight=1.0, shape='sigmoid', t_mid=0.5, width=0.15)
    mask = torch.ones(3, 1, 2, 2)
    mask[0, :, 0] = 0
    term, mean_lambda = distill_loss(output, teacher, sigma, mask, 1, config, {})
    lam = torch.sigmoid((sigma - 0.5) / 0.15)
    expected = ((output - teacher) ** 2 * mask * lam.view(3, 1, 1, 1)).mean()
    assert float(term) == pytest.approx(float(expected), rel=1e-6)
    assert float(mean_lambda) == pytest.approx(float(lam.mean()), rel=1e-6)
    assert float(lam[0]) < 0.1 < 0.9 < float(lam[2])


@pytest.mark.parametrize('decay, steps, factors', [('linear', 100, [1.0, 0.5, 0.0, 0.0]),
                                                  ('cosine', 100, [1.0, 0.5, 0.0, 0.0])])
def test_the_decay_schedule(decay, steps, factors):
    output, teacher, sigma = torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1), torch.tensor([0.5])
    config = cfg(loss_weight=1.0, decay=decay, decay_steps=steps)
    got = [float(distill_loss(output, teacher, sigma, torch.empty(0), step, config, {})[0]) for step in (0, 50, 100, 150)]
    assert got == pytest.approx(factors, abs=1e-6)


def test_the_teacher_is_a_constant_and_a_wrong_shape_is_an_error():
    output = torch.randn(2, 3, 2, 2, requires_grad=True)
    teacher = torch.randn(2, 3, 2, 2, requires_grad=True)
    term, _ = distill_loss(output, teacher, torch.tensor([0.3, 0.6]), torch.empty(0), 1, cfg(), {})
    term.backward()
    assert output.grad is not None and teacher.grad is None
    with pytest.raises(RuntimeError, match='does not match'):
        distill_loss(output, torch.randn(2, 3, 2, 3), torch.tensor([0.3, 0.6]), torch.empty(0), 1, cfg(), {})


# the label

def test_the_label_carries_sigma_only_when_distilling():
    latents = torch.randn(2, 128, 3, 5)
    embeds = [torch.randn(4, mi.TEXT_DIM), torch.randn(7, mi.TEXT_DIM)]
    batch = {'latents': latents, 'mask': None, 'prompt_embeds': embeds}
    for distill in (None, cfg()):
        pipe = bare_pipeline()
        pipe.distill = distill
        pipe.noise_sampler = mi.NoiseLevelSampler('logit_normal', -1.09, 0.95, 0.95, 0.01, 1.0, 0.0, 1.0)
        (noisy, sigma, text, mask), label = pipe.prepare_inputs(batch, timestep_quantile=0.5)
        if distill is None:
            assert len(label) == 2
        else:
            target, label_sigma, loss_mask = label
            assert torch.equal(label_sigma, sigma) and label_sigma.shape == (2,) and loss_mask is None
            assert target.shape == latents.shape


# the teacher in the pipeline

def distilling_pipeline(lora=False, distill=None, stream=None, reference=None):
    """The tiny model with [teacher] on: layers built with the teacher, a Runner over them."""
    reference = reference or tiny_reference()
    pipe = trained_lora_pipeline(reference) if lora else loaded_pipeline(reference)
    if not lora:
        for p in pipe.diffusion_model.parameters():
            p.requires_grad_(True)
    pipe.distill = distill or cfg(loss_weight=1.0)
    pipe.config = {'adapter': dict(LORA)} if lora else {}
    pipe.training_step = 1
    if stream is not None:
        layout = StreamLayout(stream, len(pipe.diffusion_model.layers))
        pipe.layer_state.stream = layout
        pipe.stream_spec = stream
    names = pipe.layer_state.stream.metric_names() if stream is not None else ['flow_loss']
    pipe.stream_metrics = StreamMetrics(names + ['distill_loss', 'distill_lambda'])
    pipe.pipeline_model = Runner(pipe.to_layers())
    return pipe, reference


def label_with_sigma(prediction, sigma):
    torch.manual_seed(5)
    return torch.randn_like(prediction), sigma, torch.empty(0)


def test_the_teacher_is_not_a_parameter_of_any_layer():
    pipe, _ = distilling_pipeline()
    first = pipe.pipeline_model.layers[0]
    assert type(first) is mi.TextEncoderLayer and first.teacher
    assert list(first.parameters()) == [] and list(first.state_dict()) == []
    layer_params = {id(p) for layer in pipe.pipeline_model.layers for p in layer.parameters()}
    teacher_params = [p for layer in first.teacher[0].layers for p in layer.parameters()]
    assert teacher_params and not any(id(p) in layer_params for p in teacher_params)
    assert not any(p.requires_grad for p in teacher_params)


def test_off_by_default_and_on_the_other_stages(monkeypatch):
    pipe = loaded_pipeline(tiny_reference())
    assert pipe.build_teacher() is None
    pipe.distill = cfg()
    monkeypatch.setattr(pipe, 'on_first_stage', lambda: False)
    assert pipe.build_teacher() is None


def test_full_fine_tune_teacher_is_the_starting_model_after_the_student_moves():
    pipe, reference = distilling_pipeline()
    x, t, text, mask = forward_inputs()
    with torch.no_grad():
        expected = reference_forward(reference, x, t, text, mask).float()
    first_layer = pipe.pipeline_model.layers[0]
    out, teacher = pipe.pipeline_model((x, t, text, mask))
    torch.testing.assert_close(teacher, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.detach(), expected, rtol=1e-5, atol=1e-5)
    # The student trains; the copy does not follow it.
    with torch.no_grad():
        for p in pipe.diffusion_model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out2, teacher2 = pipe.pipeline_model((x, t, text, mask))
    assert not torch.allclose(out2.detach(), expected, atol=1e-3)
    torch.testing.assert_close(teacher2, expected, rtol=1e-5, atol=1e-5)
    assert teacher2.dtype == torch.float32 and teacher2.requires_grad  # a leaf with a graph, see below
    assert first_layer.teacher[0].layers[1].block is not pipe.pipeline_model.layers[2].block


def test_adapter_teacher_is_the_base_model_and_leaves_the_adapter_alone():
    pipe, reference = distilling_pipeline(lora=True)
    x, t, text, mask = forward_inputs()
    with torch.no_grad():
        base = reference_forward(reference, x, t, text, mask).float()
    # A weight train.py froze (a zero learning rate) must stay frozen across the teacher pass.
    frozen = next(p for n, p in pipe.diffusion_model.named_parameters() if 'lora_B' in n)
    frozen.requires_grad_(False)
    flags = {n: p.requires_grad for n, p in pipe.diffusion_model.named_parameters()}

    out, teacher = pipe.pipeline_model((x, t, text, mask))
    torch.testing.assert_close(teacher, base, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out.detach(), base, atol=1e-3), 'the adapter must change the student'
    assert {n: p.requires_grad for n, p in pipe.diffusion_model.named_parameters()} == flags
    assert not frozen.requires_grad

    # The student is what it was without a teacher in front of it, and the adapter is back on.
    student_only = Runner(pipe.pipeline_model.layers[1:])((x, t, text, mask))
    torch.testing.assert_close(out.detach(), student_only.detach())
    again, _ = pipe.pipeline_model((x, t, text, mask))
    torch.testing.assert_close(again.detach(), out.detach())


def test_loss_is_the_flow_loss_plus_lambda_times_the_distance_to_the_teacher():
    pipe, _ = distilling_pipeline(distill=cfg(loss_weight=0.7))
    loss_fn = pipe.get_loss_fn()
    x, t, text, mask = forward_inputs()
    output = pipe.pipeline_model((x, t, text, mask))
    out, teacher = output
    label = label_with_sigma(out, t)
    pipe.training_step = 3
    loss = loss_fn(output, label)
    flow = F.mse_loss(out, label[0])
    term = 0.7 * F.mse_loss(out, teacher)
    torch.testing.assert_close(loss, flow + term)
    scalars = pipe.get_extra_log_scalars()
    assert scalars['train/flow_loss'] == pytest.approx(float(flow), rel=1e-5)
    assert scalars['train/distill_loss'] == pytest.approx(float(term), rel=1e-5)
    assert scalars['train/distill_lambda'] == pytest.approx(0.7)
    # The data loss alone when the label has no sigma and the output is plain (eval).
    plain = loss_fn(out.detach(), (label[0], label[2]))
    torch.testing.assert_close(plain, flow.detach())


def test_the_gradient_is_the_gradient_of_the_formula_and_the_teacher_gets_none():
    pipe, _ = distilling_pipeline(distill=cfg(loss_weight=0.7))
    x, t, text, mask = forward_inputs()
    output = pipe.pipeline_model((x, t, text, mask))
    out, teacher = output
    pipe.get_loss_fn()(output, label_with_sigma(out, t)).backward()
    got = {n: p.grad.clone() for n, p in pipe.diffusion_model.named_parameters() if p.grad is not None}
    pipe.diffusion_model.zero_grad()

    output = pipe.pipeline_model((x, t, text, mask))
    out, teacher = output
    label = label_with_sigma(out, t)
    (F.mse_loss(out, label[0]) + 0.7 * F.mse_loss(out, teacher.detach())).backward()
    for name, p in pipe.diffusion_model.named_parameters():
        if p.grad is not None or name in got:
            torch.testing.assert_close(got[name], p.grad, rtol=1e-4, atol=1e-6 * float(p.grad.abs().max()))
    teacher_layers = pipe.pipeline_model.layers[0].teacher[0].layers
    assert all(p.grad is None for layer in teacher_layers for p in layer.parameters())


def test_checkpointing_gives_the_same_gradients():
    pipe, _ = distilling_pipeline(distill=cfg(loss_weight=0.7))
    inputs = forward_inputs()

    def loss_of(run):
        output = run(tuple(t.clone() for t in inputs))
        return pipe.get_loss_fn()(output, label_with_sigma(output[0], inputs[1]))

    loss_of(pipe.pipeline_model).backward()
    expected = {n: p.grad.clone() for n, p in pipe.diffusion_model.named_parameters() if p.grad is not None}
    pipe.diffusion_model.zero_grad()

    def checkpointed(x):
        for layer in pipe.pipeline_model.layers:
            x = checkpoint(layer, x, use_reentrant=False) if type(layer).__name__ in pipe.checkpointable_layers else layer(x)
        return x

    loss_of(checkpointed).backward()
    for name, grad in expected.items():
        torch.testing.assert_close(dict(pipe.diffusion_model.named_parameters())[name].grad, grad,
                                   rtol=1e-4, atol=1e-5 * float(grad.abs().max()))


@pytest.mark.parametrize('cut', [1, 2, 3, 4])
def test_stages_pass_the_teacher_across_the_cut(cut):
    pipe, _ = distilling_pipeline(distill=cfg(loss_weight=0.7))
    inputs = forward_inputs()
    output = pipe.pipeline_model(tuple(t.clone() for t in inputs))
    pipe.get_loss_fn()(output, label_with_sigma(output[0], inputs[1])).backward()
    expected = {n: p.grad.clone() for n, p in pipe.diffusion_model.named_parameters() if p.grad is not None}
    pipe.diffusion_model.zero_grad()

    layers = pipe.pipeline_model.layers
    first = Runner(layers[:cut])(tuple(t.clone() for t in inputs))
    received = tuple(t.detach().requires_grad_(torch.is_floating_point(t)) for t in first)
    output = Runner(layers[cut:])(received)
    pipe.get_loss_fn()(output, label_with_sigma(output[0], inputs[1])).backward()
    for i, t in enumerate(received):
        if torch.is_floating_point(t):
            assert t.grad is not None, f'stage-2 input {i} got no gradient: DeepSpeed would fail'
    floating = [i for i, t in enumerate(first) if torch.is_floating_point(t)]
    for i in floating:
        assert first[i].requires_grad, f'stage-1 output {i} has no graph: DeepSpeed would fail on it'
    torch.autograd.backward([first[i] for i in floating], [received[i].grad for i in floating])
    for name, grad in expected.items():
        torch.testing.assert_close(dict(pipe.diffusion_model.named_parameters())[name].grad, grad,
                                   rtol=1e-4, atol=1e-5 * float(grad.abs().max()))


@pytest.mark.parametrize('mode', ['eval', 'sampling', 'no_grad'])
def test_eval_and_sampling_run_no_teacher(mode):
    """Each guard on its own: the layer in eval mode, the sampling flag, and no autograd."""
    pipe, _ = distilling_pipeline()
    layers = pipe.pipeline_model.layers
    calls = []
    original = layers[0].teacher[0].velocity
    layers[0].teacher[0].velocity = lambda *args: calls.append(1) or original(*args)
    if mode == 'eval':
        for layer in layers:
            layer.train(False)
    elif mode == 'sampling':
        pipe.layer_state.sampling = True
    if mode == 'no_grad':
        with torch.no_grad():
            output = pipe.pipeline_model(forward_inputs())
    else:
        output = pipe.pipeline_model(forward_inputs())
    assert not calls and torch.is_tensor(output)
    # and a training batch does run it, once
    for layer in layers:
        layer.train(True)
    pipe.layer_state.sampling = False
    output = pipe.pipeline_model(forward_inputs())
    assert len(calls) == 1 and isinstance(output, tuple) and len(output) == 2


def test_the_stream_penalty_is_unchanged_by_the_teacher_pass():
    """The teacher runs through the same blocks in the adapter case; the forward hooks that tap the
    sublayer writes must not see it, and the stream vector must be what it is without a teacher."""
    reference = tiny_reference()
    inputs = forward_inputs()
    spec = spec_of()

    plain = trained_lora_pipeline(reference)
    layout = StreamLayout(spec, len(plain.diffusion_model.layers))
    plain.layer_state.stream = layout
    plain.pipeline_model = Runner(plain.to_layers())
    _, expected_stream = plain.pipeline_model(tuple(t.clone() for t in inputs))

    both, _ = distilling_pipeline(lora=True, stream=spec, reference=reference)
    out, stream, teacher = both.pipeline_model(tuple(t.clone() for t in inputs))
    torch.testing.assert_close(stream.detach(), expected_stream.detach())
    assert float(stream.detach().abs().sum()) > 0
    for layer in both.pipeline_model.layers:
        if type(layer).__name__ == 'TransformerLayer':
            assert layer._writes is None and layer._gates is None
    loss = both.get_loss_fn()(( out, stream, teacher), label_with_sigma(out, inputs[1]))
    assert torch.isfinite(loss)
    assert set(both.get_extra_log_scalars()) >= {'train/flow_loss', 'train/stream_penalty', 'train/distill_loss',
                                                'train/distill_lambda'}


def test_the_adapter_switch_restores_every_flag():
    pipe = trained_lora_pipeline(tiny_reference())
    params = list(pipe.lora_model.parameters())
    frozen = next(p for n, p in pipe.lora_model.named_parameters() if 'lora_A' in n)
    frozen.requires_grad_(False)
    before = [p.requires_grad for p in params]
    with mi.adapter_switched_off(pipe.lora_model):
        pass
    assert [p.requires_grad for p in params] == before
    with pytest.raises(RuntimeError, match='boom'):
        with mi.adapter_switched_off(pipe.lora_model):
            raise RuntimeError('boom')
    assert [p.requires_grad for p in params] == before


def test_the_teacher_reads_the_embeddings_the_text_layer_encodes_on_the_fly(monkeypatch):
    """cache_text_embeddings = false: the batch carries token ids, TextEncoderLayer encodes them, and the
    teacher gets the same embeddings the student does (the student starts equal to it)."""
    _, _, (cached, on_the_fly) = on_the_fly_pipelines(monkeypatch)
    inputs, _ = caption_batch(on_the_fly)
    assert not torch.is_floating_point(inputs[2])
    with torch.no_grad():
        expected = on_the_fly.pipeline_model(tuple(t.clone() for t in inputs))
    on_the_fly.distill = cfg(loss_weight=1.0)
    on_the_fly.config = {}
    on_the_fly.pipeline_model = Runner(on_the_fly.to_layers())
    for p in on_the_fly.diffusion_model.parameters():
        p.requires_grad_(True)
    out, teacher = on_the_fly.pipeline_model(tuple(t.clone() for t in inputs))
    torch.testing.assert_close(teacher, expected.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.detach(), expected, rtol=1e-5, atol=1e-5)


def test_a_teacher_cannot_follow_an_adapter_started_from_existing_weights():
    base = {'teacher': {'loss_weight': 0.5}, 'adapter': {'type': 'lora'}}
    assert parse_distill_config(base) is not None
    with pytest.raises(ValueError, match='init_from_existing'):
        parse_distill_config({**base, 'adapter': {'type': 'lora', 'init_from_existing': 'some_lora'}})
