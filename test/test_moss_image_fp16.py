"""MOSS-Image float16 compute (models/moss_image.py): the compute_dtype / fp16_fix options, the residual-stream
fold, and the loss scale on the loss, on CPU.

The fold is checked against the unfolded model: in float32 it must change nothing but the size of the stream
(outputs, stream penalty numbers and every gradient), and in float16 it must keep a stream that the unfolded
model overflows finite and close to the float32 result.
"""

import copy

import pytest
import torch

from models import moss_image as mi
from test.test_moss_image import (Runner, bare_pipeline, forward_inputs, single_process,  # noqa: F401
                                  tiny_reference, tiny_state)
from test.test_distillation import cfg, label_with_sigma
from test.test_stream_loss import label_for, spec_of
from utils.loss_scale import LossScaler
from utils.stream_loss import StreamLayout, StreamMetrics

FOLD = 1 / 16


def fold_pipeline(fold=None, compute=torch.float32, spec=None, state_edit=None, seed=0, distill=None):
    """The tiny model loaded as train does (storage float32, weights cast to the compute dtype per call),
    folded when `fold` is given, with the stream penalties when `spec` is."""
    reference = tiny_reference(seed)
    state = tiny_state(reference)
    if state_edit:
        state_edit(state)
    patcher = mi.load_transformer_patcher(state, None, torch.float32)
    patcher.set_model_compute_dtype(compute)
    with torch.no_grad():
        patcher.patch_model()
    pipe = bare_pipeline()
    pipe.dtype = torch.float32
    pipe.compute_dtype = compute
    pipe.fp16_fold = fold
    pipe.diffusion_model = patcher.model.diffusion_model
    for p in pipe.diffusion_model.parameters():
        p.requires_grad_(True)
    if fold is not None:
        mi.fold_stream(pipe.diffusion_model, fold)
    pipe.layer_state = mi.LayerState(pipe.diffusion_model.rope, compute, fold)
    names = ['flow_loss']
    if spec is not None:
        layout = StreamLayout(spec, len(pipe.diffusion_model.layers))
        pipe.layer_state.stream = layout
        pipe.stream_spec = spec
        names = layout.metric_names()
    if distill is not None:
        pipe.distill = distill
        pipe.config = {}
        # what load_diffusion_model keeps (after the fold) for a full fine-tune's teacher
        pipe.teacher_snapshot = copy.deepcopy(pipe.diffusion_model)
        names = names + ['distill_loss', 'distill_lambda']
    if spec is not None or distill is not None:
        pipe.stream_metrics = StreamMetrics(names)
    pipe.pipeline_model = Runner(pipe.to_layers())
    return pipe


def stream_peaks(pipe, inputs):
    """The largest |stream| after each transformer block."""
    peaks = []
    with torch.no_grad():
        for layer in pipe.pipeline_model.layers:
            inputs = layer(inputs)
            if isinstance(layer, mi.TransformerLayer):
                peaks.append(float(inputs[0].float().abs().max()))
    return peaks, inputs


def prediction(output):
    return output[0] if isinstance(output, tuple) else output


# configuration

def precision_of(model_config, config=None):
    pipe = bare_pipeline(dict(model_config), config)
    pipe.dtype = mi._dtype(pipe.model_config.get('dtype', 'bfloat16'), 'dtype')
    pipe._check_precision()
    return pipe


def test_the_compute_dtype_defaults_to_the_storage_dtype_and_nothing_is_folded():
    for dtype in ('bfloat16', 'float32'):
        pipe = precision_of({'dtype': dtype})
        assert pipe.compute_dtype == mi._dtype(dtype, 'dtype') and pipe.fp16_fold is None


def test_float16_compute_over_float32_storage_with_the_online_fold():
    pipe = precision_of({'dtype': 'float32', 'compute_dtype': 'float16', 'fp16_fix': 'online'})
    assert pipe.compute_dtype == torch.float16 and pipe.fp16_fold == 1 / 16
    assert pipe.model_config['fp16_fix_scale'] == 1 / 16
    assert precision_of({'compute_dtype': 'fp16', 'fp16_fix': 'online', 'fp16_fix_scale': 0.25}).fp16_fold == 0.25
    assert precision_of({'compute_dtype': 'fp16', 'fp16_fix': 'none'}).fp16_fold is None
    # float32 storage with bfloat16 compute is allowed too
    assert precision_of({'dtype': 'float32', 'compute_dtype': 'bfloat16'}).compute_dtype == torch.bfloat16


def test_float16_storage_is_for_an_adapter_over_a_float16_compute_base():
    pipe = precision_of({'dtype': 'float16', 'compute_dtype': 'float16'}, {'adapter': {}})
    assert pipe.dtype == torch.float16 and pipe.compute_dtype == torch.float16


@pytest.mark.parametrize('model,config,match', [
    ({'compute_dtype': 'float8'}, None, 'computes in'),
    ({'dtype': 'float16'}, None, 'stores the weights in float16'),
    ({'dtype': 'float16', 'compute_dtype': 'float16'}, None, 'stores the weights in float16'),
    ({'dtype': 'float16', 'compute_dtype': 'bfloat16'}, {'adapter': {}}, 'stores the weights in float16'),
    ({'dtype': 'float32', 'compute_dtype': 'float16', 'fp16_fix': 'weights'}, None, "'none' or 'online'"),
    ({'compute_dtype': 'float16', 'fp16_fix_scale': 0.5}, None, "only applies with fp16_fix = 'online'"),
    ({'dtype': 'bfloat16', 'fp16_fix': 'online'}, None, 'only applies with compute_dtype'),
    ({'compute_dtype': 'float16', 'fp16_fix': 'online', 'fp16_fix_scale': 0}, None, 'fp16_fix_scale'),
    ({'compute_dtype': 'float16', 'fp16_fix': 'online', 'fp16_fix_scale': 2.0}, None, 'fp16_fix_scale'),
    ({'compute_dtype': 'float16', 'fp16_fix': 'online', 'fp16_fix_scale': True}, None, 'fp16_fix_scale'),
    ({'compute_dtype': 'float16', 'fp16_fix': 'online', 'fp16_fix_scale': 'x'}, None, 'fp16_fix_scale'),
    ({'dtype': 'float32', 'compute_dtype': 'float16'}, {'fp8_training': {}}, r'\[fp8_training\]'),
])
def test_precision_refusals(model, config, match):
    with pytest.raises(ValueError, match=match):
        precision_of(model, config)


def test_float16_without_the_fold_warns_about_the_base_weights(capsys):
    precision_of({'dtype': 'float32', 'compute_dtype': 'float16'})
    assert 'fp16_fix' in capsys.readouterr().out
    precision_of({'dtype': 'float32', 'compute_dtype': 'float16', 'fp16_fix': 'online'})
    assert capsys.readouterr().out == ''


# the fold

def test_a_model_is_folded_once():
    pipe = fold_pipeline(FOLD)
    with pytest.raises(RuntimeError, match='already folded'):
        mi.fold_stream(pipe.diffusion_model, FOLD)


def test_only_the_eps_of_the_stream_readers_change():
    plain, folded = fold_pipeline(), fold_pipeline(FOLD)
    for block, folded_block in zip(plain.diffusion_model.layers, folded.diffusion_model.layers):
        for name in ('norm1', 'norm2'):
            assert getattr(folded_block, name).eps == pytest.approx(getattr(block, name).eps * FOLD ** 2)
        for name in ('q_norm', 'k_norm'):
            assert getattr(folded_block, name).eps == getattr(block, name).eps
    assert folded.diffusion_model.final_norm_video.eps == pytest.approx(
        plain.diffusion_model.final_norm_video.eps * FOLD ** 2)
    for name, p in plain.diffusion_model.named_parameters():
        assert torch.equal(p, dict(folded.diffusion_model.named_parameters())[name]), name


@pytest.mark.parametrize('fold', [FOLD, 0.25, 0.3])
@pytest.mark.parametrize('lengths', [(5, 8), (6, 6)], ids=['mixed lengths', 'one length'])
def test_the_fold_in_float32_changes_nothing_but_the_size_of_the_stream(fold, lengths):
    plain, folded = fold_pipeline(), fold_pipeline(fold)
    inputs = forward_inputs(lengths)
    plain_peaks, expected = stream_peaks(plain, inputs)
    folded_peaks, got = stream_peaks(folded, inputs)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)
    # The eps is a small term next to the stream's size, so the sizes follow the factor closely.
    for before, after in zip(plain_peaks, folded_peaks):
        assert after == pytest.approx(before * fold, rel=1e-3)


def test_the_stream_penalties_see_the_stream_at_its_true_size():
    plain, folded = fold_pipeline(spec=spec_of()), fold_pipeline(FOLD, spec=spec_of())
    inputs = forward_inputs()
    with torch.no_grad():
        expected = plain.pipeline_model(inputs)[1]
        got = folded.pipeline_model(inputs)[1]
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-6)


def test_the_loss_and_every_gradient_are_those_of_the_unfolded_model():
    plain, folded = fold_pipeline(spec=spec_of()), fold_pipeline(FOLD, spec=spec_of())
    inputs = forward_inputs()
    grads = []
    losses = []
    for pipe in (plain, folded):
        output = pipe.pipeline_model(tuple(t.clone() for t in inputs))
        loss = pipe.get_loss_fn()(output, label_for(output[0]))
        loss.backward()
        losses.append(float(loss))
        grads.append({n: p.grad.clone() for n, p in pipe.diffusion_model.named_parameters() if p.grad is not None})
    assert losses[1] == pytest.approx(losses[0], rel=1e-4)
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) > 20
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=2e-3, atol=1e-7, msg=lambda m: f'{name}: {m}')


def test_a_folded_teacher_gives_the_velocity_of_the_unfolded_one():
    """The frozen copy is made from the folded model and runs with the same fold, so its velocity (and the
    distillation loss against it) is the unfolded model's."""
    plain, folded = fold_pipeline(distill=cfg(loss_weight=1.0)), fold_pipeline(FOLD, distill=cfg(loss_weight=1.0))
    teachers = [pipe.pipeline_model.layers[0].teacher[0] for pipe in (plain, folded)]
    assert teachers[1].state.fold == FOLD and teachers[0].state.fold is None
    inputs = forward_inputs()
    results = []
    for pipe in (plain, folded):
        out, teacher = pipe.pipeline_model(tuple(t.clone() for t in inputs))
        label = label_with_sigma(out, inputs[1])
        pipe.training_step = 3
        results.append((out.detach(), teacher.detach(), float(pipe.get_loss_fn()((out, teacher), label))))
    torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-4, atol=1e-5)
    assert results[1][2] == pytest.approx(results[0][2], rel=1e-4)


def overflowing_stream(state):
    """Gates and FFN outputs large enough that the float32 stream passes float16's 65504."""
    d = state['layers.0.norm1.weight'].shape[0]
    state['shared_block_modulation.weight'].view(6, d, -1)[5] *= 1e4
    state['shared_block_modulation.bias'].view(6, d)[5] *= 1e4
    for name in state:
        if name.endswith('proj.ffn_down.weight'):
            state[name] *= 300


def test_the_fold_keeps_a_stream_that_overflows_float16_in_range():
    inputs = forward_inputs()
    reference_peaks, expected = stream_peaks(fold_pipeline(state_edit=overflowing_stream), inputs)
    assert max(reference_peaks) > 65504, 'the tiny model no longer overflows float16: retune overflowing_stream'
    plain_peaks, plain = stream_peaks(fold_pipeline(compute=torch.float16, state_edit=overflowing_stream), inputs)
    assert not torch.isfinite(plain).all()
    folded_peaks, got = stream_peaks(fold_pipeline(FOLD, torch.float16, state_edit=overflowing_stream), inputs)
    assert max(folded_peaks) < 65504 and torch.isfinite(got).all()
    scale = float(expected.abs().max())
    assert float((got.float() - expected).abs().max()) < 0.03 * scale


def test_float16_compute_matches_float32_when_nothing_overflows():
    inputs = forward_inputs()
    _, expected = stream_peaks(fold_pipeline(), inputs)
    for fold in (None, FOLD):
        _, got = stream_peaks(fold_pipeline(fold, torch.float16), inputs)
        assert got.dtype == torch.float16
        assert float((got.float() - expected).abs().max()) < 0.03 * float(expected.abs().max())


# the loss scale

def test_the_scaled_loss_has_the_same_value_and_scaled_gradients():
    plain, scaled = fold_pipeline(spec=spec_of()), fold_pipeline(spec=spec_of())
    scaled.loss_scaler = LossScaler(init_scale=64.0)
    inputs = forward_inputs()
    losses, grads = [], []
    for pipe in (plain, scaled):
        output = pipe.pipeline_model(tuple(t.clone() for t in inputs))
        loss = pipe.get_loss_fn()(output, label_for(output[0]))
        loss.backward()
        losses.append(float(loss))
        grads.append({n: p.grad.clone() for n, p in pipe.diffusion_model.named_parameters() if p.grad is not None})
    assert losses[0] == losses[1]
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name] * 64.0, rtol=1e-4, atol=1e-6,
                                   msg=lambda m: f'{name}: {m}')


def test_evaluation_is_not_scaled():
    pipe = fold_pipeline()
    pipe.loss_scaler = LossScaler(init_scale=64.0)
    with torch.no_grad():
        output = pipe.pipeline_model(forward_inputs())
        loss = pipe.get_loss_fn()(output, label_for(output))
    assert not loss.requires_grad


def test_the_logged_flow_loss_is_not_scaled():
    logged = []
    for scaler in (None, LossScaler(init_scale=64.0)):
        pipe = fold_pipeline(spec=spec_of())
        pipe.loss_scaler = scaler
        output = pipe.pipeline_model(forward_inputs())
        pipe.get_loss_fn()(output, label_for(output[0]))
        logged.append(pipe.get_extra_log_scalars())
    assert logged[0] == logged[1] and logged[0]['train/flow_loss'] > 0
