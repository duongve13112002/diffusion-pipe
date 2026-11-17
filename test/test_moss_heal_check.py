"""utils/moss_heal_check.py and tools/check_moss_heal.py on tiny MOSS-Image models, on CPU."""

import importlib.util
import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

import comfy.model_management

from test.test_moss_image import TEXT_DIM, tiny_reference, tiny_state, write_transformer_dir
from utils import moss_heal_check as hc

K = 2


@pytest.fixture(autouse=True)
def training_path():
    # The differentiable path diffusion-pipe trains on; the inference kernels are not for CPU.
    saved = comfy.model_management.in_training
    comfy.model_management.in_training = True
    yield
    comfy.model_management.in_training = saved


def inputs(n=2, size=4, length=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 128, size, size, generator=g), torch.randn(n, length, TEXT_DIM, generator=g))


def plant(model, block=1, channel=3, factor=3e5):
    """A massive channel: the FFN's write into one stream channel is made huge."""
    with torch.no_grad():
        model.layers[block].proj['ffn_down'].weight[channel] *= factor
    return model


def test_probe_reads_the_same_numbers_as_independent_hooks():
    model = tiny_reference()
    # Two samples at two noise levels, so the sums and the running peak run over several passes.
    latents, context = inputs(n=2)
    calls = []
    cursor = {}

    def pre(i):
        return lambda m, args: cursor.update({i: {'x': args[0].detach().clone(), 'mod': args[1]}})

    def post(i):
        return lambda m, args, out: cursor[i].update(out=out.detach().clone())

    def tap(i, name):
        return lambda m, args, out: cursor[i].update({name: out.detach().clone()})

    def new_call(m, args):
        calls.append({})
        cursor.clear()

    def keep(m, args, out):
        calls[-1].update(cursor)

    handles = [model.register_forward_pre_hook(new_call), model.register_forward_hook(keep)]
    for i, block in enumerate(model.layers):
        handles += [block.register_forward_pre_hook(pre(i)), block.register_forward_hook(post(i)),
                    block.proj['o'].register_forward_hook(tap(i, 'o')),
                    block.proj['ffn_down'].register_forward_hook(tap(i, 'down'))]
    blocks = hc.measure(model, latents, context, [0.2, 0.9], k=K, seed=3)
    for h in handles:
        h.remove()
    assert len(calls) == 4

    def energy_of(t):
        return t.reshape(-1, t.shape[-1]).square().sum(0)

    for i, record in enumerate(blocks):
        outs = [c[i]['out'] for c in calls]
        writes = {'attn': [c[i]['o'] * c[i]['mod'][2] for c in calls],
                  'ffn': [c[i]['down'] * c[i]['mod'][5] for c in calls]}
        for c in calls:
            # The gate indices the probe assumes: the block output is the input plus the two gated writes.
            torch.testing.assert_close(c[i]['out'], c[i]['x'] + c[i]['o'] * c[i]['mod'][2]
                                       + c[i]['down'] * c[i]['mod'][5], rtol=1e-4, atol=1e-5)
        assert record['peak'] == pytest.approx(max(float(o.abs().max()) for o in outs))
        assert min(float(o.abs().max()) for o in outs) < record['peak']
        assert record['rms'] == pytest.approx(float(torch.cat([o.flatten() for o in outs]).square().mean().sqrt()),
                                              rel=1e-5)
        for part in ('attn', 'ffn'):
            energy = sum(energy_of(w) for w in writes[part])
            assert record[f'write_{part}'] == pytest.approx(float(energy.topk(K).values.sum() / energy.sum()),
                                                            rel=1e-4)
            assert record[f'write_{part}_peak'] == pytest.approx(max(float(w.abs().max()) for w in writes[part]))
        energy = sum(energy_of(o) for o in outs)
        assert record['concentration'] == pytest.approx(float(energy.topk(K).values.sum() / energy.sum()), rel=1e-4)
        assert record['top_channel'] == int(energy.argmax())
        assert record['top_channel_share'] == pytest.approx(float(energy.max() / energy.sum()), rel=1e-4)


def test_measure_feeds_the_model_flow_matching_noised_latents_and_the_samples_own_text():
    model = tiny_reference()
    latents, context = inputs(n=2)
    fed = []
    handle = model.register_forward_pre_hook(lambda m, args: fed.append([a.detach().clone() for a in args]))
    hc.measure(model, latents, context, [0.25, 0.75], k=K, seed=5)
    handle.remove()
    generator = torch.Generator().manual_seed(5)
    assert len(fed) == 4
    for n in range(2):
        noise = torch.randn(latents[n].shape, generator=generator)
        for j, sigma in enumerate([0.25, 0.75]):
            x, t, text = fed[n * 2 + j]
            torch.testing.assert_close(x[0], (1 - sigma) * latents[n] + sigma * noise)
            assert t.tolist() == [sigma]
            torch.testing.assert_close(text[0], context[n])


def test_probe_detaches_its_hooks_and_counts_every_pass():
    model = tiny_reference()
    latents, context = inputs(n=2)
    probe = hc.StreamProbe(model.layers, K)
    with probe:
        pass
    assert all(len(block._forward_hooks) == 0 and len(block._forward_pre_hooks) == 0 for block in model.layers)
    assert all(len(block.proj['o']._forward_hooks) == 0 for block in model.layers)
    with pytest.raises(RuntimeError, match='no forward pass'):
        probe.blocks()
    hc.measure(model, latents, context, [0.2, 0.8], k=K)
    assert all(len(block._forward_hooks) == 0 for block in model.layers)
    probe = hc.StreamProbe(model.layers, K)
    with probe:
        for i in range(2):
            model(torch.randn(1, 128, 4, 4), torch.tensor([0.5]), context[i:i + 1])
    assert probe.calls == 2 and probe.count[0] == 2 * 1 * (16 + 5) * 64


def test_measure_is_deterministic_and_seeded():
    model = tiny_reference()
    latents, context = inputs()
    a = hc.measure(model, latents, context, [0.3, 0.9], k=K, seed=1)
    b = hc.measure(model, latents, context, [0.3, 0.9], k=K, seed=1)
    c = hc.measure(model, latents, context, [0.3, 0.9], k=K, seed=2)
    assert a == b
    assert a != c
    assert hc.text_dim_of(model) == TEXT_DIM


def test_a_planted_massive_channel_shows_in_every_stream_number():
    latents, context = inputs()
    clean = hc.summarize_stream(hc.measure(tiny_reference(), latents, context, [0.5], k=K))
    massive = hc.summarize_stream(hc.measure(plant(tiny_reference()), latents, context, [0.5], k=K))
    assert massive["peak"] > 3 * clean["peak"]
    assert massive['write_ffn_mean'] > clean['write_ffn_mean'] + 0.1
    assert massive['write_share_mean'] > clean['write_share_mean']
    assert massive['concentration_mean'] > clean['concentration_mean']
    assert massive['fp16_headroom'] < clean['fp16_headroom']
    blocks = hc.measure(plant(tiny_reference(), channel=3), latents, context, [0.5], k=K)
    assert blocks[2]['top_channel'] == 3
    assert blocks[1]['write_ffn'] > 0.9


def test_weight_report_finds_an_outlier_and_counts_the_elements_over_the_cap():
    clean = hc.weight_report(tiny_reference().layers, cap=16.0)
    model = tiny_reference()
    with torch.no_grad():
        model.layers[2].proj['o'].weight[5, 7] = 3.0
    report = hc.weight_report(model.layers, cap=16.0)
    row = {r['name']: r for r in report['tensors']}['layers.2.proj.o.weight']
    assert row['max'] == pytest.approx(3.0)
    assert row['ratio'] > 16 and row['over_cap'] == 1
    assert row['elements'] == 64 * 64
    summary = hc.summarize_weights(report)
    assert summary['ratio_max_name'] == 'layers.2.proj.o.weight'
    assert summary['over_cap'] == 1 and summary['tensors_over_cap'] == 1
    assert summary['over_cap_fraction'] == pytest.approx(1 / sum(r['elements'] for r in report['tensors']))
    clean_summary = hc.summarize_weights(clean)
    assert clean_summary['over_cap'] == 0 and clean_summary['ratio_max'] < 8
    # Only 2-D tensors of the blocks are looked at (the norms' gains are 1-D).
    assert all(r['name'].startswith('layers.') and '.proj.' in r['name'] for r in report['tensors'])
    assert len(report['tensors']) == 3 * 6


def test_weight_report_skips_what_is_not_plain_floating_point():
    model = tiny_reference()
    model.layers[0].proj['o'].weight.data = model.layers[0].proj['o'].weight.data.to(torch.float8_e4m3fn)
    report = hc.weight_report(model.layers)
    assert report['skipped'] == ['layers.0.proj.o.weight']
    assert 'layers.0.proj.o.weight' not in {r['name'] for r in report['tensors']}
    assert hc.summarize_weights(report)['skipped'] == 1


def test_a_zero_weight_does_not_divide_by_zero():
    model = tiny_reference()
    with torch.no_grad():
        model.layers[0].proj['o'].weight.zero_()
    row = {r['name']: r for r in hc.weight_report(model.layers)['tensors']}['layers.0.proj.o.weight']
    assert row['ratio'] == float('inf') and row['over_cap'] == 0


def stream(peak=1000.0, share=0.3):
    return {'peak': peak, 'write_share_mean': share}


def weights(ratio=10.0):
    return {'ratio_max': ratio}


def test_verdict_statuses_and_limits():
    limits = hc.Limits(peak=16000.0, write_share=0.6, weight_ratio=32.0)
    assert hc.verdict(weights(), stream(), limits)['status'] == 'healed'
    assert hc.verdict(weights(100.0), stream(), limits)['status'] == 'partly healed'
    assert hc.verdict(weights(100.0), stream(peak=62000.0), limits)['status'] == 'partly healed'
    assert hc.verdict(weights(100.0), stream(peak=62000.0, share=0.9), limits)['status'] == 'not healed'
    assert hc.verdict(None, None, limits)['status'] == 'unknown'
    # Weights only, or the stream only: the checks that were measured decide.
    assert hc.verdict(weights(), None, limits)['status'] == 'healed'
    assert hc.verdict(weights(99.0), None, limits)['status'] == 'not healed'
    assert hc.verdict(None, stream(peak=62000.0), limits)['status'] == 'partly healed'
    assert len(hc.verdict(None, stream(), limits)['checks']) == 2


def test_verdict_limits_are_inclusive_and_each_check_is_its_own():
    limits = hc.Limits(peak=100.0, write_share=0.5, weight_ratio=16.0)
    result = hc.verdict(weights(16.0), stream(peak=100.0, share=0.5), limits)
    assert result['status'] == 'healed'
    for w, s, failing in ((weights(16.5), stream(100.0, 0.5), 'weight'), (weights(16.0), stream(100.5, 0.5), 'peak'),
                          (weights(16.0), stream(100.0, 0.51), 'write share')):
        checks = hc.verdict(w, s, limits)['checks']
        failed = [c['name'] for c in checks if not c['ok']]
        assert len(failed) == 1 and failing in failed[0]


def test_default_limits_follow_float16_headroom_and_the_recipe():
    limits = hc.Limits()
    assert limits.peak == hc.FP16_MAX / 4
    # Base and SFT peak near 62,000: not healed by the stream peak; a stream far below float16's range is.
    assert not hc.verdict(None, stream(peak=62000.0))['checks'][0]['ok']
    assert hc.verdict(None, stream(peak=3000.0))['checks'][0]['ok']


def test_summaries_from_blocks():
    blocks = [{'block': 0, 'peak': 10.0, 'rms': 1.0, 'concentration': 0.2, 'write_attn': 0.1, 'write_ffn': 0.3},
              {'block': 1, 'peak': 40.0, 'rms': 3.0, 'concentration': 0.4, 'write_attn': 0.5, 'write_ffn': 0.7}]
    s = hc.summarize_stream(blocks)
    assert s['peak'] == 40.0 and s['peak_block'] == 1
    assert s['fp16_headroom'] == pytest.approx(hc.FP16_MAX / 40.0)
    assert s['rms_mean'] == 2.0 and s['concentration_mean'] == pytest.approx(0.3)
    assert s['write_share_mean'] == pytest.approx(0.4) and s['write_share_max'] == 0.7
    assert s['write_attn_mean'] == pytest.approx(0.3) and s['write_ffn_mean'] == pytest.approx(0.5)


def test_report_text_has_the_numbers_checks_verdict_and_reference():
    model = plant(tiny_reference())
    latents, context = inputs()
    blocks = hc.measure(model, latents, context, [0.5], k=K)
    s = hc.summarize_stream(blocks)
    w = hc.summarize_weights(hc.weight_report(model.layers))
    result = hc.verdict(w, s, hc.Limits(peak=1.0, write_share=0.01, weight_ratio=1.0))
    clean = tiny_reference()
    cs = hc.summarize_stream(hc.measure(clean, latents, context, [0.5], k=K))
    cw = hc.summarize_weights(hc.weight_report(clean.layers))
    text = hc.format_report('mine', w, s, result, blocks, ('base', cw, cs))
    assert text.startswith('mine') and 'verdict: NOT HEALED' in text and 'FAIL' in text
    assert 'reference' in text and 'per block' in text and f"{s['peak']:.4g}" in text
    assert f"{cs['peak']:.4g}" in text
    assert hc.format_report('m', None, s, result).count('weights') == 0


def load_tool():
    spec = importlib.util.spec_from_file_location('check_moss_heal', Path(__file__).resolve().parent.parent
                                                  / 'tools' / 'check_moss_heal.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def run_tool(tool, capsys, *argv):
    code = tool.main(['--device', 'cpu', '--dtype', 'float32', '--k', str(K), '--size', '4', '--text-len', '5',
                      '--samples', '1', '--sigmas', '0.5', *argv])
    return code, capsys.readouterr().out


def test_tool_end_to_end_on_a_directory_a_file_and_a_reference(tmp_path, capsys):
    tool = load_tool()
    healed_dir = write_transformer_dir(tiny_state(tiny_reference()), tmp_path / 'healed')
    bad_file = tmp_path / 'bad.safetensors'
    safetensors.torch.save_file(tiny_state(plant(tiny_reference())), bad_file)
    report = tmp_path / 'report.json'
    code, out = run_tool(tool, capsys, '--model', str(healed_dir), '--peak-limit', '1e6', '--write-share-limit', '0.99',
                         '--weight-ratio-limit', '100', '--json', str(report), '--per-block')
    assert code == 0 and 'verdict: HEALED' in out and 'note: random inputs' in out and 'per block' in out
    data = json.loads(report.read_text())
    assert data['verdict']['status'] == 'healed' and len(data['blocks']) == 3 and data['stream']['peak'] > 0
    code, out = run_tool(tool, capsys, '--model', str(bad_file), '--reference', str(healed_dir), '--peak-limit', '1.0',
                         '--write-share-limit', '0.5', '--weight-ratio-limit', '100')
    assert code == 1 and 'verdict: PARTLY HEALED' in out
    assert 'reference' in out and 'FAIL' in out
    assert 'FAIL  stream peak' in out


def test_tool_weights_only_runs_no_forward_pass(tmp_path, capsys):
    tool = load_tool()
    path = tmp_path / 'm.safetensors'
    safetensors.torch.save_file(tiny_state(tiny_reference()), path)
    code, out = run_tool(tool, capsys, '--model', str(path), '--weights-only', '--weight-ratio-limit', '100')
    assert code == 0 and 'stream:' not in out and 'weights' in out and 'random inputs' not in out


def test_tool_reads_real_inputs_and_refuses_bad_ones(tmp_path, capsys):
    tool = load_tool()
    path = tmp_path / 'm.safetensors'
    safetensors.torch.save_file(tiny_state(tiny_reference()), path)
    latents, context = inputs(n=2)
    good = tmp_path / 'inputs.safetensors'
    safetensors.torch.save_file({'latents': latents, 'context': context, 'sigmas': torch.tensor([0.25, 0.75])}, good)
    report = tmp_path / 'r.json'
    code, out = run_tool(tool, capsys, '--model', str(path), '--inputs', str(good), '--json', str(report))
    assert 'random inputs' not in out
    assert json.loads(report.read_text())['stream']['peak'] > 0
    bad = tmp_path / 'bad.safetensors'
    safetensors.torch.save_file({'latents': latents}, bad)
    with pytest.raises(SystemExit, match='no context'):
        tool.read_inputs(str(bad))
    safetensors.torch.save_file({'latents': latents, 'context': context[:1]}, bad)
    with pytest.raises(SystemExit, match='same N'):
        tool.read_inputs(str(bad))
    lat, ctx, sig = tool.read_inputs(str(good))
    assert sig == [0.25, 0.75] and lat.shape[0] == ctx.shape[0] == 2
