"""Is a MOSS-Image model healed? Measurements and a verdict (tools/check_moss_heal.py runs them).

A fine-tune of MOSS-Image can end up with a few "massive" channels that carry most of the residual
stream's energy and with weight outliers; the stream penalties and the outlier decay (utils/stream_loss.py,
utils/outlier_decay.py) are what heal that. This module measures the three things healing changes:

- the stream peak: the largest |value| any block writes to the residual stream. The Base and SFT weights
  peak near 62,000, close to float16's 65,504; a healed model peaks far below it, so float16 compute needs
  no fold (fp16_fix = 'online');
- the write shares: the share of each sublayer's write (the attention's and the FFN's, projection output
  times its gate) that its k largest channels hold, as `[model.stream_loss.write]` measures it. The
  hinge pushes it below `target` (0.5 in the MOSS-Image-heal recipe);
- the weight outliers: per 2-D weight of the blocks, the largest |w| over the tensor's RMS. The outlier decay
  pulls every element back under `cap` x RMS (the recipe's cap falls from 64 to 16; the MOSS-Image-heal card
  reports no element above 64x).

    probe = StreamProbe(model.layers, k=32)
    with probe:
        run the model on a few (latent, sigma, text) inputs, without gradients
    stream = summarize_stream(probe.blocks())
    weights = summarize_weights(weight_report(model.layers))
    result = verdict(weights, stream, Limits())

The limits are rules of thumb taken from the recipe's own settings, not measured on a population of models;
the report prints every number so the call can be made by eye, and `--reference` in the tool puts the
numbers of an unhealed model beside them. The stream numbers depend on the inputs: random ones only show
whether the weights can produce a large peak at all, real latents and encoded prompts are what to judge by.

Kept free of training-stack imports so it can be unit-tested on CPU.
"""

import math
from dataclasses import dataclass

import torch

from utils.stream_loss import channel_energy, top_shares

FP16_MAX = 65504.0
DEFAULT_K = 32
# (part, the block's projection whose output is the write, the index of its gate in the modulation)
WRITES = (('attn', 'o', 2), ('ffn', 'ffn_down', 5))


@dataclass
class Limits:
    """What a healed model has to stay under. peak: the stream peak (a quarter of float16's maximum; the
    MOSS-Image-heal card reports 3,083 against the base's 58,411); write_share: the mean top-k share of the
    sublayer writes (the recipe's target 0.5, plus room); weight_ratio: the largest |w| / RMS of any block
    weight (the card reports no element above 64x its tensor's RMS)."""
    peak: float = FP16_MAX / 4
    write_share: float = 0.6
    weight_ratio: float = 64.0


def _plain(weight):
    return weight.dtype.is_floating_point and weight.dtype.itemsize >= 2


class StreamProbe:
    """Records, for every block of `layers` (the ModuleList of ComfyUI's MOSS-Image Block), the stream it writes
    and the channel energies of its writes while it runs. Use as a context manager around the forward passes;
    every pass adds to the same sums, so the shares are over everything that ran.

    The blocks' forward is ComfyUI's: Block(x, mod, pe, ...), mod the six (batch, 1, dim) modulation tensors,
    the attention's write is proj.o's output times mod[2], the FFN's is proj.ffn_down's times mod[5].
    """

    def __init__(self, layers, k=DEFAULT_K):
        self.layers = list(layers)
        self.k = k
        self.handles = []
        self._mod = None
        self.reset()

    def reset(self):
        n = len(self.layers)
        self.peak = [0.0] * n
        self.squares = [0.0] * n
        self.count = [0] * n
        self.energy = [None] * n
        self.write_energy = [{part: None for part, _, _ in WRITES} for _ in range(n)]
        self.write_peak = [{part: 0.0 for part, _, _ in WRITES} for _ in range(n)]
        self.calls = 0

    def __enter__(self):
        for index, block in enumerate(self.layers):
            self.handles.append(block.register_forward_pre_hook(self._take_mod))
            self.handles.append(block.register_forward_hook(self._block_tap(index)))
            for part, name, gate in WRITES:
                self.handles.append(block.proj[name].register_forward_hook(self._write_tap(index, part, gate)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._mod = None
        return False

    def _take_mod(self, module, args):
        self._mod = args[1]

    @torch.no_grad()
    def _block_tap(self, index):
        def tap(module, args, output):
            hidden = output.detach()
            self.peak[index] = max(self.peak[index], float(hidden.abs().amax()))
            squares = torch.linalg.vector_norm(hidden.float()).square().item()
            self.squares[index] += squares
            self.count[index] += hidden.numel()
            energy = channel_energy(hidden)
            self.energy[index] = energy if self.energy[index] is None else self.energy[index] + energy
            if index == 0:
                self.calls += 1
        return tap

    @torch.no_grad()
    def _write_tap(self, index, part, gate):
        def tap(module, args, output):
            write = output.detach() * self._mod[gate].detach()
            self.write_peak[index][part] = max(self.write_peak[index][part], float(write.abs().amax()))
            energy = channel_energy(write)
            previous = self.write_energy[index][part]
            self.write_energy[index][part] = energy if previous is None else previous + energy
        return tap

    def blocks(self):
        """One dict per block: peak, rms, concentration (top-k share of the block output), the write shares and
        peaks, and the channel holding the most energy."""
        if self.calls == 0:
            raise RuntimeError('The probe saw no forward pass')
        out = []
        for i in range(len(self.layers)):
            energy = self.energy[i]
            record = {
                'block': i,
                'peak': self.peak[i],
                'rms': math.sqrt(self.squares[i] / self.count[i]),
                'concentration': float(top_shares(energy[None], min(self.k, energy.numel()))[0]),
                'top_channel': int(energy.argmax()),
                'top_channel_share': float(energy.max() / energy.sum().clamp_min(torch.finfo(torch.float32).tiny)),
            }
            for part, _, _ in WRITES:
                write = self.write_energy[i][part]
                record[f'write_{part}'] = float(top_shares(write[None], min(self.k, write.numel()))[0])
                record[f'write_{part}_peak'] = self.write_peak[i][part]
            out.append(record)
        return out


def summarize_stream(blocks):
    """The numbers the verdict and the report lead with, from StreamProbe.blocks()."""
    worst = max(blocks, key=lambda b: b['peak'])
    writes = [b[f'write_{part}'] for b in blocks for part, _, _ in WRITES]
    return {
        'peak': worst['peak'],
        'peak_block': worst['block'],
        'fp16_headroom': FP16_MAX / worst['peak'] if worst['peak'] > 0 else math.inf,
        'rms_mean': sum(b['rms'] for b in blocks) / len(blocks),
        'concentration_mean': sum(b['concentration'] for b in blocks) / len(blocks),
        'write_share_mean': sum(writes) / len(writes),
        'write_share_max': max(writes),
        'write_attn_mean': sum(b['write_attn'] for b in blocks) / len(blocks),
        'write_ffn_mean': sum(b['write_ffn'] for b in blocks) / len(blocks),
    }


@torch.no_grad()
def weight_report(layers, cap=16.0):
    """Per 2-D weight of the blocks: its RMS, its largest |w|, their ratio and how many elements lie above
    cap x RMS. Weights that are not plain floating point (quantized) are listed as skipped."""
    rows, skipped = [], []
    for index, block in enumerate(layers):
        for name, parameter in block.named_parameters():
            if parameter.ndim != 2:
                continue
            full = f'layers.{index}.{name}'
            if not _plain(parameter):
                skipped.append(full)
                continue
            w = parameter.detach().float()
            rms = w.square().mean().sqrt().item()
            peak = w.abs().max().item()
            rows.append({
                'name': full,
                'rms': rms,
                'max': peak,
                'ratio': peak / rms if rms > 0 else math.inf,
                'over_cap': int((w.abs() > cap * rms).sum()),
                'elements': w.numel(),
            })
    return {'cap': cap, 'tensors': rows, 'skipped': skipped}


def summarize_weights(report):
    rows = report['tensors']
    if not rows:
        return None
    worst = max(rows, key=lambda r: r['ratio'])
    total = sum(r['elements'] for r in rows)
    over = sum(r['over_cap'] for r in rows)
    ratios = sorted(r['ratio'] for r in rows)
    return {
        'ratio_max': worst['ratio'],
        'ratio_max_name': worst['name'],
        'ratio_median': ratios[len(ratios) // 2],
        'over_cap': over,
        'over_cap_fraction': over / total,
        'tensors': len(rows),
        'tensors_over_cap': sum(1 for r in rows if r['over_cap']),
        'skipped': len(report['skipped']),
    }


def verdict(weights, stream, limits=None):
    """{'status': 'healed' | 'partly healed' | 'not healed' | 'unknown', 'checks': [...]}.

    Each check is {name, value, limit, ok}, for the measurements that were taken (`weights` or `stream` may be
    None). healed: every check passes; not healed: none does.
    """
    limits = limits or Limits()
    checks = []
    if stream is not None:
        checks.append({'name': 'stream peak (float16 headroom)', 'value': stream['peak'], 'limit': limits.peak,
                       'ok': stream['peak'] <= limits.peak})
        checks.append({'name': 'write share, mean over blocks and sublayers', 'value': stream['write_share_mean'],
                       'limit': limits.write_share, 'ok': stream['write_share_mean'] <= limits.write_share})
    if weights is not None:
        checks.append({'name': 'largest weight |w| / RMS', 'value': weights['ratio_max'],
                       'limit': limits.weight_ratio, 'ok': weights['ratio_max'] <= limits.weight_ratio})
    if not checks:
        return {'status': 'unknown', 'checks': checks}
    passed = sum(c['ok'] for c in checks)
    status = 'healed' if passed == len(checks) else 'not healed' if passed == 0 else 'partly healed'
    return {'status': status, 'checks': checks}


def text_dim_of(model):
    return model.text_embedder.proj.weight.shape[1]


@torch.no_grad()
def measure(model, latents, context, sigmas, k=DEFAULT_K, seed=0, device=None):
    """StreamProbe's blocks() after running `model` (ComfyUI's MOSSImage) on every latent at every sigma.

    latents [N, C, H, W] are clean latents: each is mixed with seeded noise as flow matching does,
    x = (1 - sigma) latent + sigma noise. context [N, L, text_dim] is each sample's text embedding (one at a
    time, as ComfyUI runs MOSS-Image). Pass `device` to move the inputs there; the model is where it is.
    """
    generator = torch.Generator().manual_seed(seed)
    dtype = next(model.parameters()).dtype
    device = device or next(model.parameters()).device
    model.eval()
    probe = StreamProbe(model.layers, k)
    with probe:
        for i in range(latents.shape[0]):
            noise = torch.randn(latents[i].shape, generator=generator, dtype=torch.float32)
            for sigma in sigmas:
                x = (1 - sigma) * latents[i].float() + sigma * noise
                model(x[None].to(device, dtype), torch.tensor([float(sigma)], device=device, dtype=dtype),
                      context[i:i + 1].to(device, dtype))
    return probe.blocks()


def format_report(name, weights, stream, result, blocks=None, reference=None):
    """The text of one model's report. `reference` is (name, weights summary, stream summary) of another
    model to print beside it."""
    lines = [f'{name}']
    def row(label, value, ref=None, fmt='{:.4g}'):
        text = f'  {label:<44}{fmt.format(value):>12}'
        if ref is not None:
            text += f'   (reference {fmt.format(ref)})'
        lines.append(text)
    ref_w = reference[1] if reference else None
    ref_s = reference[2] if reference else None
    if stream is not None:
        lines.append(' stream:')
        row('peak |value| (any block)', stream['peak'], ref_s and ref_s['peak'])
        row('  in block', stream['peak_block'], ref_s and ref_s['peak_block'], '{:d}')
        row('  float16 headroom (65504 / peak)', stream['fp16_headroom'], ref_s and ref_s['fp16_headroom'])
        row('block RMS (mean)', stream['rms_mean'], ref_s and ref_s['rms_mean'])
        row('top-k share of the block output (mean)', stream['concentration_mean'],
            ref_s and ref_s['concentration_mean'])
        row('top-k share of the sublayer writes (mean)', stream['write_share_mean'],
            ref_s and ref_s['write_share_mean'])
        row('  attention writes', stream['write_attn_mean'], ref_s and ref_s['write_attn_mean'])
        row('  FFN writes', stream['write_ffn_mean'], ref_s and ref_s['write_ffn_mean'])
    if weights is not None:
        lines.append(' weights (the blocks\' 2-D tensors):')
        row('largest |w| / RMS', weights['ratio_max'], ref_w and ref_w['ratio_max'])
        lines.append(f"    in {weights['ratio_max_name']}")
        row('median |w| max / RMS', weights['ratio_median'], ref_w and ref_w['ratio_median'])
        row('elements above the cap x RMS', weights['over_cap'], ref_w and ref_w['over_cap'], '{:d}')
        row('  tensors with any', weights['tensors_over_cap'], ref_w and ref_w['tensors_over_cap'], '{:d}')
        if weights['skipped']:
            lines.append(f"    ({weights['skipped']} quantized tensors skipped)")
    lines.append(' checks:')
    for c in result['checks']:
        lines.append(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']}: {c['value']:.4g} (limit {c['limit']:.4g})")
    lines.append(f" verdict: {result['status'].upper()}")
    if blocks:
        lines.append(' per block: block  peak  rms  concentration  write_attn  write_ffn  top_channel')
        for b in blocks:
            lines.append(f"  {b['block']:>3}  {b['peak']:>10.4g}  {b['rms']:>8.4g}  {b['concentration']:>6.3f}"
                         f"  {b['write_attn']:>6.3f}  {b['write_ffn']:>6.3f}  {b['top_channel']}")
    return '\n'.join(lines)
