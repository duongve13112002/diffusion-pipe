"""utils/stream_loss.py and its use by models/moss_image.py (the stream penalties), on CPU.

The numbers the pipeline layers carry to the loss are checked against ComfyUI's own MOSS-Image
forward: forward hooks on its blocks and a recording stand-in for its gated_residual give the block
outputs and the sublayer writes, and the penalty terms are recomputed from them directly.
"""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import comfy.ldm.moss_image.model as comfy_moss
from models import moss_image as mi
from test.test_moss_image import (LORA, Runner, forward_inputs, loaded_pipeline, reference_forward,
                                  single_process, tiny_reference, trained_lora_pipeline)  # noqa: F401
from utils import stream_loss as sl
from utils.stream_loss import StreamLayout, StreamMetrics, parse_stream_loss, top_shares, write_target

ALL = {
    'concentration': {'weight': 0.5, 'k': 4},
    'rms': {'weight': 0.25, 'target': 0.7},
    'write': {'weight': 2.0, 'k': 4, 'target': 0.3, 'dilute': False},
}


def spec_of(**terms):
    return parse_stream_loss({'stream_loss': {name: dict(fields) for name, fields in (terms or ALL).items()}})


# configuration

def test_defaults_and_active_terms():
    assert parse_stream_loss({}) is None
    assert parse_stream_loss({'stream_loss': {}}) is None
    assert parse_stream_loss({'stream_loss': {'write': {'weight': 0, 'target': 0.5}}}) is None
    spec = parse_stream_loss({'stream_loss': {
        'write': {'weight': 1e-3, 'target': 0.5}, 'rms': {'weight': 1e-5}, 'concentration': {'weight': 0}}})
    assert set(spec) == {'write', 'rms'}
    assert spec['write'] == {'weight': 1e-3, 'target': 0.5, 'k': 32, 'parts': 'both', 'anneal': 0, 'start': 0,
                             'dilute': False}
    assert spec['rms'] == {'weight': 1e-5, 'target': 1.0}
    spec = parse_stream_loss({'stream_loss': {'concentration': {'weight': 1}}})
    assert spec['concentration'] == {'weight': 1, 'k': 32, 'dilute': True}


@pytest.mark.parametrize('table', [
    3,
    {'peak': {'weight': 1}},
    {'rms': 3},
    {'rms': {}},
    {'rms': {'weight': 1, 'typo': 2}},
    {'rms': {'weight': -1}},
    {'rms': {'weight': True}},
    {'rms': {'weight': 1, 'target': 0}},
    {'concentration': {'weight': 1, 'k': 0}},
    {'concentration': {'weight': 1, 'k': 2.5}},
    {'concentration': {'weight': 1, 'dilute': 1}},
    {'write': {'weight': 1}},
    {'write': {'weight': 1, 'target': 0}},
    {'write': {'weight': 1, 'target': 1.5}},
    {'write': {'weight': 1, 'target': 0.5, 'parts': 'mlp'}},
    {'write': {'weight': 1, 'target': 0.5, 'anneal': -1}},
    {'write': {'weight': 1, 'target': 0.5, 'start': 1.5}},
    {'write': {'weight': float('nan'), 'target': 0.5}},
])
def test_bad_tables_are_refused(table):
    with pytest.raises(ValueError):
        parse_stream_loss({'stream_loss': table})


def test_k_cannot_exceed_the_width():
    sl.check_width(spec_of(), 64)
    with pytest.raises(ValueError, match='exceeds the model width'):
        sl.check_width(spec_of(), 3)


def test_compile_is_refused():
    config = {'compile': True, 'model': {'type': 'moss_image', 'transformer_path': 'a', 'vae_path': 'b',
                                         'text_encoder_path': 'c',
                                         'stream_loss': {'rms': {'weight': 1}}}}
    with pytest.raises(ValueError, match='compile'):
        mi.MossImagePipeline(config)
    config['compile'] = False
    assert set(mi.MossImagePipeline(config).stream_spec) == {'rms'}


# the terms

def test_top_shares_and_dilute():
    energy = torch.tensor([[4.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    shares = top_shares(energy, 2)
    torch.testing.assert_close(shares.detach(), torch.tensor([0.7, 0.5]))
    shares.sum().backward()
    # diluted: raising a channel outside the top lowers the share
    assert energy.grad[0, 1] < 0 and energy.grad[0, 0] > 0
    energy.grad = None
    top_shares(energy, 2, dilute=False).sum().backward()
    # shrink-only: the total is a constant, so only the top channels carry gradient
    assert energy.grad[0, 1] == 0 and energy.grad[0, 3] == 0 and energy.grad[0, 0] > 0
    zero = top_shares(torch.zeros(1, 4), 2)
    assert torch.isfinite(zero).all()


def test_write_target_schedule():
    term = {'start': 10, 'anneal': 100, 'target': 0.5}
    assert write_target(term, 1) == 1.0 and write_target(term, 10) == 1.0
    assert write_target(term, 60) == pytest.approx(0.75)
    assert write_target(term, 110) == pytest.approx(0.5) and write_target(term, 10_000) == 0.5
    assert write_target({'start': 0, 'anneal': 0, 'target': 0.2}, 1) == pytest.approx(0.2)


# the numbers the layers carry, against ComfyUI's own forward

class Recorder:
    """The block outputs and the sublayer writes of ComfyUI's forward, summed over its calls."""

    def __init__(self, reference, monkeypatch):
        self.blocks = len(reference.layers)
        self.outputs = []        # per call: [block][tokens, D]
        self.writes = []         # per call: [(attn, ffn)] energies per block
        self._current = None
        self._pending = []
        for block in reference.layers:
            block.register_forward_hook(self._block)
        original = comfy_moss.gated_residual

        def recording(x, y, gate):
            self._pending.append(sl.channel_energy(y * gate))
            return original(x, y, gate)

        monkeypatch.setattr(comfy_moss, 'gated_residual', recording)

    def _block(self, module, inputs, output):
        self._current = [] if self._current is None else self._current
        self._current.append(output[0].detach())

    def run(self, reference, x, t, text, mask):
        outputs = []
        writes = {}
        for i in range(x.shape[0]):
            self._current, self._pending = None, []
            reference_forward(reference, x[i:i + 1], t[i:i + 1], text[i:i + 1], mask[i:i + 1])
            outputs.append(self._current)
            for block in range(self.blocks):
                for j, part in enumerate(('attn', 'ffn')):
                    e = self._pending[2 * block + j].detach()
                    writes[(block, part)] = writes.get((block, part), 0) + e
        return outputs, writes


def expected_vector(spec, layout, outputs, writes, dim):
    """What StreamLayout should hold, from per-sample block outputs [sample][block][T, D]."""
    blocks = layout.num_blocks
    v = torch.zeros(sl.GROUPS, blocks)
    for b in range(blocks):
        energy = sum(sl.channel_energy(sample[b]) for sample in outputs)
        if 'concentration' in spec:
            t = spec['concentration']
            v[sl.CONCENTRATION, b] = top_shares(energy[None], t['k'], t['dilute'])[0]
        if 'rms' in spec:
            rms = torch.stack([(sample[b].square().sum() / (sample[b].shape[0] * dim)).sqrt() for sample in outputs])
            v[sl.RMS_TERM, b] = (rms / spec['rms']['target'] - 1).square().mean()
            v[sl.RMS_VALUE, b] = rms.mean()
        if 'write' in spec:
            t = spec['write']
            for part in layout.write_parts:
                v[sl.SHARE_SLOT[part], b] = top_shares(writes[(b, part)][None], t['k'], t['dilute'])[0]
    return v.reshape(-1)


def stream_pipeline(spec, lora=False, reference=None):
    reference = reference or tiny_reference()
    pipe = trained_lora_pipeline(reference) if lora else loaded_pipeline(reference)
    if not lora:
        for p in pipe.diffusion_model.parameters():
            p.requires_grad_(True)
    layout = StreamLayout(spec, len(pipe.diffusion_model.layers))
    pipe.layer_state.stream = layout
    pipe.stream_metrics = StreamMetrics(layout.metric_names())
    pipe.training_step = 1
    pipe.pipeline_model = Runner(pipe.to_layers())
    return pipe, layout, reference


@pytest.mark.parametrize('lengths', [(5, 8), (6, 6)], ids=['mixed lengths', 'one length'])
@pytest.mark.parametrize('terms', [None, {'write': ALL['write']}, {'write': {**ALL['write'], 'parts': 'ffn'}},
                                   {'rms': ALL['rms']}, {'concentration': ALL['concentration']}],
                         ids=['all', 'write', 'write-ffn', 'rms', 'concentration'])
def test_the_carried_vector_matches_comfyuis_forward(lengths, terms, monkeypatch):
    spec = spec_of(**terms) if terms else spec_of()
    pipe, layout, reference = stream_pipeline(spec)
    x, t, text, mask = forward_inputs(lengths)
    recorder = Recorder(reference, monkeypatch)
    outputs, writes = recorder.run(reference, x, t, text, mask)
    prediction, stream = pipe.pipeline_model((x, t, text, mask))
    assert stream.shape == (layout.size,) and stream.dtype == torch.float32
    expected = expected_vector(spec, layout, outputs, writes, dim=64)
    torch.testing.assert_close(stream.detach(), expected, rtol=1e-4, atol=1e-6)
    # the prediction is the one the unpenalised model gives
    plain = loaded_pipeline(tiny_reference()).pipeline_model((x, t, text, mask))
    torch.testing.assert_close(prediction, plain, rtol=1e-5, atol=1e-5)


def test_penalty_is_the_weighted_terms():
    spec = spec_of()
    layout = StreamLayout(spec, 3)
    torch.manual_seed(0)
    stream = torch.rand(layout.size)
    total, terms, parts = layout.penalty(stream, step=1)
    v = stream.reshape(sl.GROUPS, 3)
    tau = write_target(spec['write'], 1)
    shares = torch.cat([v[sl.SHARE_ATTN], v[sl.SHARE_FFN]])
    expected = {
        'concentration': v[sl.CONCENTRATION].mean(),
        'rms': v[sl.RMS_TERM].mean(),
        'write': (shares - tau).clamp_min(0).square().mean(),
    }
    for name, value in expected.items():
        torch.testing.assert_close(terms[name], value)
    torch.testing.assert_close(total, sum(spec[n]['weight'] * expected[n] for n in expected))
    assert set(parts) == {'stream_rms', 'write_share_attn', 'write_share_ffn', 'write_target'}
    assert set(layout.metric_names()) == {'flow_loss', 'stream_penalty', 'penalty_concentration', 'penalty_rms',
                                          'penalty_write', 'stream_rms', 'write_share_attn', 'write_share_ffn',
                                          'write_target'}


def test_write_hinge_is_zero_at_or_below_the_target_share():
    spec = spec_of(write={'weight': 1.0, 'k': 2, 'target': 0.5})
    layout = StreamLayout(spec, 2)
    stream = torch.zeros(layout.size)
    stream[2 * sl.SHARE_ATTN:2 * sl.SHARE_ATTN + 2] = 0.4
    stream[2 * sl.SHARE_FFN:2 * sl.SHARE_FFN + 2] = 0.5
    total, terms, _ = layout.penalty(stream, step=10)
    assert float(total) == 0.0
    spec = spec_of(write={'weight': 1.0, 'k': 2, 'target': 0.5, 'anneal': 100, 'start': 10})
    total, _, parts = StreamLayout(spec, 2).penalty(stream, step=10)
    assert float(parts['write_target']) == 1.0 and float(total) == 0.0


# the loss function

def label_for(prediction):
    torch.manual_seed(5)
    return torch.randn_like(prediction), torch.empty(0)


def test_loss_is_the_flow_loss_plus_the_weighted_penalty():
    pipe, layout, _ = stream_pipeline(spec_of())
    loss_fn = pipe.get_loss_fn()
    inputs = forward_inputs()
    output = pipe.pipeline_model(inputs)
    label = label_for(output[0])
    loss = loss_fn(output, label)
    flow = torch.nn.functional.mse_loss(output[0], label[0])
    penalty, terms, parts = layout.penalty(output[1], step=1)
    torch.testing.assert_close(loss, flow + penalty)
    assert float(penalty) > 0
    scalars = pipe.get_extra_log_scalars()
    assert scalars['train/flow_loss'] == pytest.approx(float(flow), rel=1e-5)
    assert scalars['train/stream_penalty'] == pytest.approx(float(penalty), rel=1e-5)
    assert set(scalars) == {f'train/{n}' for n in layout.metric_names()}
    # reduce() starts the next step afresh
    assert pipe.get_extra_log_scalars() == {}


def test_step_reaches_the_hinge():
    spec = spec_of(write={'weight': 1.0, 'k': 4, 'target': 0.05, 'anneal': 100, 'start': 0})
    pipe, layout, _ = stream_pipeline(spec)
    loss_fn = pipe.get_loss_fn()
    output = pipe.pipeline_model(forward_inputs())
    label = label_for(output[0])
    pipe.training_step = 1
    early = float(loss_fn(output, label))
    pipe.get_extra_log_scalars()
    pipe.training_step = 100
    late = float(loss_fn(output, label))
    assert late > early
    assert pipe.get_extra_log_scalars()['train/write_target'] == pytest.approx(0.05)


@pytest.mark.parametrize('mode', ['eval', 'sampling'])
def test_eval_and_sampling_return_the_plain_prediction(mode):
    pipe, layout, _ = stream_pipeline(spec_of())
    inputs = forward_inputs()
    for layer in pipe.pipeline_model.layers:
        layer.train(False)
    if mode == 'sampling':
        for layer in pipe.pipeline_model.layers:
            layer.train(True)
        pipe.layer_state.sampling = True
    with torch.no_grad():
        output = pipe.pipeline_model(inputs)
    assert torch.is_tensor(output)
    loss_fn = pipe.get_loss_fn()
    label = label_for(output)
    torch.testing.assert_close(loss_fn(output, label), torch.nn.functional.mse_loss(output, label[0]))
    assert pipe.get_extra_log_scalars() == {}


def test_sampling_with_the_penalty_on_still_runs_and_resets_its_flag():
    pipe, layout, _ = stream_pipeline(spec_of())
    pipe.sample_cfg = 1.0
    pipe.conds = tuple(t for t in forward_inputs()[2:])
    x, t = forward_inputs()[:2]
    out = pipe._predict(x[:1], t[:1], tuple(c[:1] for c in pipe.conds))
    assert torch.is_tensor(out) and not pipe.layer_state.sampling


def test_without_stream_loss_nothing_changes():
    pipe = loaded_pipeline(tiny_reference())
    inputs = forward_inputs()
    first = Runner(pipe.to_layers()[:2])(inputs)
    assert len(first) == 5
    assert torch.is_tensor(pipe.pipeline_model(inputs))
    pipe.layer_state.stream = None
    assert pipe.get_loss_fn() is not None


# gradients

def penalty_loss(pipe, inputs):
    output = pipe.pipeline_model(inputs)
    return pipe.get_loss_fn()(output, label_for(output[0]))


def test_full_fine_tune_penalty_reaches_the_base_weights():
    pipe, layout, _ = stream_pipeline(spec_of())
    inputs = forward_inputs()
    output = pipe.pipeline_model(inputs)
    penalty = layout.penalty(output[1], 1)[0]
    penalty.backward()
    for name, p in pipe.diffusion_model.named_parameters():
        if 'layers.' in name and 'proj' in name:
            assert p.grad is not None and float(p.grad.abs().max()) > 0, name


def test_lora_penalty_reaches_only_the_adapter():
    pipe, layout, _ = stream_pipeline(spec_of(), lora=True)
    output = pipe.pipeline_model(forward_inputs())
    layout.penalty(output[1], 1)[0].backward()
    seen = 0
    for name, p in pipe.diffusion_model.named_parameters():
        if 'lora_' in name:
            seen += p.grad is not None and float(p.grad.abs().max()) > 0
        else:
            assert p.grad is None, name
    assert seen > 0


def test_write_includes_the_adapters_part():
    """A write measured at the wrapped projection moves when the adapter does."""
    spec = spec_of(write={'weight': 1.0, 'k': 4, 'target': 0.3})
    pipe, layout, _ = stream_pipeline(spec, lora=True)
    inputs = forward_inputs()
    before = pipe.pipeline_model(inputs)[1].detach().clone()
    with torch.no_grad():
        for name, p in pipe.diffusion_model.named_parameters():
            if 'lora_B' in name:
                p.mul_(0)
    after = pipe.pipeline_model(inputs)[1].detach()
    assert not torch.allclose(before, after)


def grads_of(pipe):
    return {n: p.grad.clone() for n, p in pipe.diffusion_model.named_parameters() if p.grad is not None}


def test_checkpointing_gives_the_same_penalty_gradients():
    spec = spec_of()
    pipe, layout, reference = stream_pipeline(spec)
    inputs = forward_inputs()
    penalty_loss(pipe, tuple(t.clone() for t in inputs)).backward()
    expected = grads_of(pipe)
    pipe.diffusion_model.zero_grad()
    x = tuple(t.clone() for t in inputs)
    for layer in pipe.pipeline_model.layers:
        if type(layer).__name__ in pipe.checkpointable_layers:
            x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
    output = x
    pipe.get_loss_fn()(output, label_for(output[0])).backward()
    got = grads_of(pipe)
    assert got.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(got[name], expected[name], rtol=1e-4, atol=1e-5 * float(expected[name].abs().max()))


@pytest.mark.parametrize('cut', [1, 2, 4])
def test_stages_pass_the_vector_across_the_cut(cut):
    spec = spec_of()
    pipe, layout, _ = stream_pipeline(spec)
    inputs = forward_inputs()
    penalty_loss(pipe, tuple(t.clone() for t in inputs)).backward()
    expected = grads_of(pipe)
    pipe.diffusion_model.zero_grad()

    layers = pipe.pipeline_model.layers
    first = Runner(layers[:cut])(tuple(t.clone() for t in inputs))
    received = tuple(t.detach().requires_grad_(torch.is_floating_point(t)) for t in first)
    output = Runner(layers[cut:])(received)
    pipe.get_loss_fn()(output, label_for(output[0])).backward()
    for i, t in enumerate(received):
        if torch.is_floating_point(t):
            assert t.grad is not None, f'stage-2 input {i} got no gradient'
    # DeepSpeed backpropagates every floating output of a stage, so each one must have a graph.
    floating = [i for i, t in enumerate(first) if torch.is_floating_point(t)]
    for i in floating:
        assert first[i].requires_grad, f'stage-1 output {i} has no graph: DeepSpeed would fail on it'
    torch.autograd.backward([first[i] for i in floating], [received[i].grad for i in floating])
    got = grads_of(pipe)
    for name in expected:
        torch.testing.assert_close(got[name], expected[name], rtol=1e-4, atol=1e-5 * float(expected[name].abs().max()))


def test_skipped_layers_leave_the_vector_alone_in_training_too():
    pipe, layout, _ = stream_pipeline(spec_of())
    pipe.layer_state.skip_layers = frozenset({1})
    output = pipe.pipeline_model(forward_inputs())
    v = output[1].detach().reshape(sl.GROUPS, layout.num_blocks)
    assert float(v[:, 1].abs().sum()) == 0 and float(v[:, 0].abs().sum()) > 0


# metrics

def test_stream_metrics_average_the_micro_batches():
    metrics = StreamMetrics(['a', 'b'])
    assert metrics.reduce() == {}
    metrics.record({'a': torch.tensor(1.0), 'b': 4.0})
    metrics.record({'a': torch.tensor(3.0), 'b': 2.0})
    assert metrics.reduce() == {'train/a': 2.0, 'train/b': 3.0}
    assert metrics.reduce() == {}


def test_hooks_are_registered_once_and_only_for_the_tapped_parts():
    spec = spec_of(write={'weight': 1.0, 'k': 4, 'target': 0.3, 'parts': 'attn'})
    pipe, layout, _ = stream_pipeline(spec)
    block = pipe.diffusion_model.layers[0]
    assert len(block.proj['o']._forward_hooks) == 1
    assert len(block.proj['ffn_down']._forward_hooks) == 0
    spec = spec_of(rms=ALL['rms'])
    pipe, layout, _ = stream_pipeline(spec)
    assert len(pipe.diffusion_model.layers[0].proj['o']._forward_hooks) == 0
