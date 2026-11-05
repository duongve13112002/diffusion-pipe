"""All-in-one check of [fp8_training] (NVIDIA Transformer Engine) on the GPU: run it, send back the report.

    python tools/check_fp8_training.py                  # everything, ~5-15 minutes on an H100 / H200
    python tools/check_fp8_training.py --quick          # smaller benchmarks
    python tools/check_fp8_training.py --out-dir /data/reports
    torchrun --nproc_per_node 2 tools/check_fp8_training.py --distributed   # several GPUs (and nodes)

Run from the diffusion-pipe checkout, in the environment you train with. It writes
fp8_training_check_<host>_<time>.txt (and a .json of the same results) as it goes, so a crash keeps
everything up to that point; send the .txt back. Nothing is downloaded and no model weights are
needed: every check uses random tensors.

What it checks:
  1. environment: Python, torch, CUDA, cuDNN, driver, GPU, Transformer Engine, and what
     utils/fp8_training.py's own start-up check says for recipe 'block' and 'current' and for
     attention;
  2. Linear layers (Fp8Linear, the layer train.py puts in the transformer blocks) for both recipes:
     output, input gradient and weight gradient against a float32 reference, next to bf16's own
     error; a frozen weight (adapters) with and without cache_frozen_weights; weights = 'direct'
     with a stored e4m3 weight; and that a call with a token count TE cannot take runs bf16;
  3. fp8 attention through the SDPA replacement and the flash-attn entry points: no mask, causal,
     key padding (also with masked keys in the middle), GQA, head sizes 64 / 128, packed varlen
     (thd, run padded before Blackwell); the bf16 calibration step, then
     fp8 output and gradients against float32; and that activation checkpointing's recomputation
     gives the same gradients as no checkpointing;
  4. speed of the Linear GEMMs (forward + backward) in bf16 and fp8 on the shapes of the models
     this repo trains, for full fine-tuning and for frozen weights;
  5. speed of attention forward + backward: PyTorch SDPA (bf16) against TE fp8;
  6. a small DiT trained for a few dozen steps in bf16, fp8 Linear, and fp8 Linear + attention,
     with LoRA and full fine-tuning, with activation checkpointing: step time (the speed-up you can
     expect in training) and the loss next to bf16's;
  7. torch.compile of a model with fp8 Linear layers;
  8. with --distributed under torchrun: data-parallel training over every rank (gradients
     all-reduced), the same call sites on every rank, and the cross-rank report.
"""

import argparse
import contextlib
import copy
import datetime
import importlib.metadata
import math
import os
import statistics
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402

import utils.common  # noqa: E402,F401  (claims the utils package before ComfyUI joins sys.path)
from utils import attention  # noqa: E402
from utils import fp8_training  # noqa: E402
from check_flash_attention import REAL_SDPA, Report, environment, rel_error, version  # noqa: E402

DEVICE = 'cuda'
BF16 = torch.bfloat16
# Relative errors (norm of the difference over the norm of the float32 reference) above these fail.
# e4m3 keeps 3 mantissa bits, so a GEMM's fp8 error is several times bf16's: a few percent.
LINEAR_LIMIT = 0.08
ATTENTION_LIMITS = {'out': 0.1, 'grad': 0.25}
# Activation checkpointing recomputes with the same scales: only kernel nondeterminism differs.
RECOMPUTE_LIMIT = 1e-2


def fp8_config(**table):
    config = {'model': {'type': 'check'}, 'adapter': {'type': 'lora'}, 'fp8_training': dict(table)}
    fp8_training.apply_fp8_training_config_defaults(config)
    return config


def fresh(**table):
    """fp8_training as a new process would have it, with the real TE backend kept."""
    backend = fp8_training._backend
    attention.uninstall()
    fp8_training.reset(backend)
    fp8_training.configure_settings(fp8_config(**table))
    attention.configure({'model': {'type': 'check'}})
    attention.install()


def gen(*shape, seed=0, dtype=BF16, scale=1.0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (torch.randn(*shape, generator=g, device=DEVICE, dtype=torch.float32) * scale).to(dtype)


def cuda_time(fn, iterations, warmup=3):
    """Median milliseconds of fn() on the GPU."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def check_environment(report):
    environment(report)
    info = {}
    for dist in ('transformer_engine', 'transformer_engine_torch', 'transformer_engine_cu12', 'transformer_engine_cu13',
                 'nvidia-cudnn-cu12', 'nvidia-cudnn-cu13'):
        info[f'package {dist}'] = version(dist)
    for key, value in info.items():
        report.line(f'  {key}: {value}')
    report.results['environment'].update(info)
    usable = {}
    for name, table in (("recipe 'block'", {}), ("recipe 'current'", {'recipe': 'current'}),
                        ('attention', {'recipe': 'current', 'attention': True})):
        try:
            fp8_training.check_environment(fp8_config(**table))
            usable[name] = None
        except Exception as error:
            usable[name] = f'{type(error).__name__}: {error}'
        report.line(f'  fp8_training start-up check, {name}: '
                    f'{"passes" if usable[name] is None else "refuses: " + usable[name]}')
        report.record({'check': f'start-up {name}', 'error': usable[name]})
    return usable


def linear_layer(recipe, fin, fout, mode='quantize', weight=None, scale_inv=None, cache=False, requires_grad=True):
    source = nn.Linear(fin, fout, device=DEVICE, dtype=BF16)
    with torch.no_grad():
        source.weight.copy_(gen(fout, fin, seed=1, scale=fin ** -0.5))
        source.bias.copy_(gen(fout, seed=2, scale=0.1))
    source.weight.requires_grad_(requires_grad)
    if weight is not None:
        source.weight = nn.Parameter(weight, requires_grad=False)
    return fp8_training.Fp8Linear(source, f'check {fin}->{fout}', recipe, mode, scale_inv=scale_inv, cache=cache)


def linear_reference(x, weight, bias, grad, train_weight):
    x = x.detach().float().requires_grad_()
    w = weight.detach().float().requires_grad_(train_weight)
    out = F.linear(x, w, bias.detach().float())
    out.backward(grad.float())
    return out.detach(), x.grad, w.grad if train_weight else None


def run_linear(layer, x, grad):
    x = x.detach().requires_grad_()
    if layer.weight.grad is not None:
        layer.weight.grad = None
    out = layer(x)
    out.backward(grad)
    return out.detach(), x.grad, layer.weight.grad


def compare_linear(report, case, layer, x, grad, reference_weight, train_weight, bf16=True):
    ref = linear_reference(x, reference_weight, layer.bias, grad, train_weight)
    before = fp8_training.counts()[0].get('fp8', 0)
    got = run_linear(layer, x, grad)
    ran_fp8 = fp8_training.counts()[0].get('fp8', 0) == before + 1
    names = ('out', 'dx', 'dw')
    errors = {name: rel_error(g, r) for name, g, r in zip(names, got, ref) if r is not None}
    line = f'  {case}: ' + ', '.join(f'{k} {v:.4f}' for k, v in errors.items())
    if bf16:
        plain = bf16_linear(x, reference_weight, layer.bias, grad, train_weight)
        line += '   (bf16 GEMM: ' + ', '.join(
            f'{k} {rel_error(g, r):.4f}' for k, g, r in zip(names, plain, ref) if r is not None) + ')'
    ok = ran_fp8 and all(v <= LINEAR_LIMIT for v in errors.values())
    if not ran_fp8:
        line += ' -- did NOT run fp8 (see the [fp8] lines above)'
    report.line(('PASS' if ok else 'FAIL') + line)
    report.record({'case': case, 'errors': errors, 'ran_fp8': ran_fp8}, ok)


def bf16_linear(x, weight, bias, grad, train_weight):
    x = x.detach().to(BF16).requires_grad_()
    w = weight.detach().to(BF16).requires_grad_(train_weight)
    out = F.linear(x, w, bias.detach().to(BF16))
    out.backward(grad.to(BF16))
    return out.detach(), x.grad, w.grad if train_weight else None


def check_linear(report, recipes):
    fresh()
    shapes = [(4096, 3072, 3072), (4096, 3072, 12288), (2048, 12288, 3072), (1024, 5120, 13824), (528, 1536, 8960)]
    for recipe in recipes:
        for tokens, fin, fout in shapes:
            layer = linear_layer(recipe, fin, fout)
            x, grad = gen(tokens, fin, seed=3), gen(tokens, fout, seed=4, scale=1e-3)
            compare_linear(report, f"{recipe}, full fine-tune, {tokens}x{fin}->{fout}", layer, x, grad,
                           layer.weight, True)
        tokens, fin, fout = 4096, 3072, 3072
        x, grad = gen(tokens, fin, seed=3), gen(tokens, fout, seed=4, scale=1e-3)
        for cache in (False, True):
            layer = linear_layer(recipe, fin, fout, cache=cache, requires_grad=False)
            # Twice: the second call is the one that reuses the cached fp8 weight.
            run_linear(layer, x, grad)
            compare_linear(report, f'{recipe}, frozen weight{", cached" if cache else ""}, {tokens}x{fin}->{fout}', layer,
                           x, grad, layer.weight, False, bf16=False)

    # weights = 'direct': a stored e4m3 weight with one scale, as ComfyUI's float8 checkpoints hold it.
    fin, fout = 3072, 3072
    value = gen(fout, fin, seed=1, scale=fin ** -0.5).float()
    scale = value.abs().amax() / torch.finfo(torch.float8_e4m3fn).max
    data = (value / scale).to(torch.float8_e4m3fn)
    layer = linear_layer('current', fin, fout, mode='direct', weight=data, scale_inv=scale.reshape(1))
    x, grad = gen(4096, fin, seed=3), gen(4096, fout, seed=4, scale=1e-3)
    compare_linear(report, "weights = 'direct', stored e4m3, 4096x3072->3072", layer, x, grad,
                   data.float() * scale, False, bf16=False)

    # 520 tokens (a multiple of 8, not 16): TE's weight-gradient GEMM refuses them, so they run bf16.
    for recipe in recipes:
        layer = linear_layer(recipe, 1536, 8960)
        counts = fp8_training.counts()[0]
        before, skipped = counts.get('fp8', 0), counts.get('calls whose token count is not a multiple of 16', 0)
        layer(gen(520, 1536).requires_grad_()).sum().backward()
        after = fp8_training.counts()[0]
        ok = after.get('fp8', 0) == before and after.get('calls whose token count is not a multiple of 16', 0) == skipped + 1
        report.line(f'{"PASS" if ok else "FAIL"}  {recipe}: 520 tokens (not a multiple of 16) run bf16 and are counted')
        report.record({'case': f'{recipe}, token count not a multiple of 16'}, ok)


class CheckBlock(nn.Module):
    """Stands for a transformer block: fp8 attention takes calls inside hooked blocks only."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def attention_reference(fn32, q, k, v, grad):
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    out = fn32(q, k, v)
    out.backward(grad.float())
    return out.detach(), q.grad, k.grad, v.grad


def attention_step(block, q, k, v, grad, checkpoint=False, training=True):
    q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
    with fp8_training.step_scope(training=training):
        if checkpoint:
            out = torch.utils.checkpoint.checkpoint(block, q, k, v, use_reentrant=False)
        else:
            out = block(q, k, v)
        out.backward(grad)
    return out.detach(), q.grad, k.grad, v.grad


def sdpa_bhsd(mask=None, causal=False, gqa=False):
    def fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=gqa)

    def fn32(q, k, v):
        m = mask
        return REAL_SDPA(q, k, v, attn_mask=m, is_causal=causal, enable_gqa=gqa)
    return fn, fn32


def varlen_case(lengths, causal):
    cu = F.pad(torch.tensor(lengths, device=DEVICE).cumsum(0), (1, 0)).to(torch.int32)
    longest = max(lengths)

    def per_sequence(q, k, v, sdpa):
        outs = []
        for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
            o = sdpa(*(t[a:b].transpose(0, 1)[None] for t in (q, k, v)), is_causal=causal)
            outs.append(o[0].transpose(0, 1))
        return torch.cat(outs)

    def fn(q, k, v):
        return attention.fp8_varlen(q, k, v, cu, cu, longest, longest, None, causal,
                                    lambda: per_sequence(q, k, v, REAL_SDPA))

    def fn32(q, k, v):
        return per_sequence(q, k, v, REAL_SDPA)
    return fn, fn32


def attention_cases():
    b, h, s = 2, 16, 1024
    keep = torch.ones(b, 1, 1, s, dtype=torch.bool, device=DEVICE)
    keep[1, ..., 700:] = False
    cases = []
    for d in (64, 128):
        fn, fn32 = sdpa_bhsd()
        cases.append((f'SDPA, no mask, head_dim {d}', fn, fn32, [(b, h, s, d)] * 3, (b, h, s, d)))
    fn, fn32 = sdpa_bhsd(causal=True)
    cases.append(('SDPA, is_causal', fn, fn32, [(b, h, s, 128)] * 3, (b, h, s, 128)))
    fn, fn32 = sdpa_bhsd(mask=keep)
    cases.append(('SDPA, key padding (bool mask)', fn, fn32, [(b, h, s, 128)] * 3, (b, h, s, 128)))
    # Padded text before the image tokens, as Qwen-Image, Z-Image and Krea 2 build their masks.
    holes = torch.ones(b, 1, 1, s, dtype=torch.bool, device=DEVICE)
    holes[0, ..., 100:300] = False
    fn, fn32 = sdpa_bhsd(mask=holes)
    cases.append(('SDPA, key padding with masked keys in the middle', fn, fn32, [(b, h, s, 128)] * 3, (b, h, s, 128)))
    fn, fn32 = sdpa_bhsd(gqa=True)
    cases.append(('SDPA, GQA 16 query / 4 key heads', fn, fn32, [(b, h, s, 128), (b, 4, s, 128), (b, 4, s, 128)],
                  (b, h, s, 128)))
    fn, fn32 = sdpa_bhsd()
    cases.append(('SDPA, cross attention 1024 queries / 512 keys', fn, fn32,
                  [(b, h, s, 128), (b, h, 512, 128), (b, h, 512, 128)], (b, h, s, 128)))
    lengths = [700, 1024, 333]
    for causal in (False, True):
        fn, fn32 = varlen_case(lengths, causal)
        shape = (sum(lengths), h, 128)
        cases.append((f'flash-attn varlen (thd){", causal" if causal else ""}', fn, fn32, [shape] * 3, shape))

    def dense(q, k, v):
        return attention.fp8_dense(q, k, v, None, False,
                                   lambda: REAL_SDPA(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2))

    def dense32(q, k, v):
        return REAL_SDPA(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
    cases.append(('flash-attn dense (bshd)', dense, dense32, [(b, s, h, 128)] * 3, (b, s, h, 128)))
    return cases


def check_attention(report):
    for case, fn, fn32, shapes, out_shape in attention_cases():
        try:
            fresh(attention=True)
            block = CheckBlock(fn)
            fp8_training._attention.hook_blocks(block, ['CheckBlock'])
            q, k, v = (gen(*shape, seed=10 + i) for i, shape in enumerate(shapes))
            grad = gen(*out_shape, seed=20, scale=1e-2)
            attention_step(block, q, k, v, grad)  # bf16 calibration step
            sites = list(fp8_training._attention.sites.values())
            active = bool(sites) and all(site.state == 'active' for site in sites)
            attention_step(block, q, k, v, grad)
            before = fp8_training.counts()[1].get('fp8', 0)
            got = attention_step(block, q, k, v, grad)
            ran_fp8 = fp8_training.counts()[1].get('fp8', 0) > before
            # Recomputation under activation checkpointing against a plain run with the same scales:
            # outside a training step TE does not update them (between two training steps it does,
            # and delayed scaling then moves the numbers by a few percent, as it should).
            plain = attention_step(block, q, k, v, grad, training=False)
            again = attention_step(block, q, k, v, grad, checkpoint=True, training=False)
            ref = attention_reference(fn32, q, k, v, grad)
            errors = {name: rel_error(g, r) for name, g, r in zip(('out', 'dq', 'dk', 'dv'), got, ref)}
            recompute = max(rel_error(a, g) for a, g in zip(again, plain))
            ok = (active and ran_fp8 and errors['out'] <= ATTENTION_LIMITS['out']
                  and all(errors[n] <= ATTENTION_LIMITS['grad'] for n in ('dq', 'dk', 'dv'))
                  and recompute <= RECOMPUTE_LIMIT)
            line = f'  {case}: ' + ', '.join(f'{k} {v:.4f}' for k, v in errors.items())
            line += f'; checkpointed vs not {recompute:.2e}'
            refused = [site.error for site in fp8_training._attention.sites.values() if site.error]
            if not ran_fp8 and refused and errors['out'] <= ATTENTION_LIMITS['out'] / 4:
                # TE cannot take this kind of call on this GPU / cuDNN: it runs bf16, as training does.
                ok = True
                line += ' -- runs bf16, TE cannot take it here: ' + refused[0].splitlines()[0][:300]
                report.line('BF16' + line)
                report.record({'case': case, 'errors': errors, 'ran_fp8': False, 'te_error': refused[0]}, ok)
                continue
            if not ran_fp8:
                line += ' -- did NOT run fp8: ' + str(fp8_training.counts()[1])
                if refused:
                    line += '; TE: ' + refused[0].splitlines()[0][:300]
            report.line(('PASS' if ok else 'FAIL') + line)
            report.record({'case': case, 'errors': errors, 'recompute': recompute, 'ran_fp8': ran_fp8,
                           'counts': fp8_training.counts()[1]}, ok)
        except Exception:
            report.line(f'FAIL  {case}: crashed\n' + traceback.format_exc())
            report.record({'case': case, 'crash': traceback.format_exc()}, False)


def bench_linear(report, recipes, quick, iterations):
    shapes = [(4096, 3072, 3072), (4096, 3072, 12288), (4096, 12288, 3072), (16384, 5120, 5120),
              (16384, 5120, 13824), (16384, 13824, 5120)]
    if quick:
        shapes = shapes[:3]
    report.line('  ms per forward+backward; speed-up over bf16 in brackets')
    for tokens, fin, fout in shapes:
        for train_weight in (True, False):
            x, grad = gen(tokens, fin, seed=3), gen(tokens, fout, seed=4, scale=1e-3)
            x.requires_grad_()
            weight = gen(fout, fin, seed=1, scale=fin ** -0.5).requires_grad_(train_weight)

            def plain():
                F.linear(x, weight).backward(grad)
            base = cuda_time(plain, iterations)
            line = f'  {tokens}x{fin}->{fout} {"full fine-tune" if train_weight else "frozen weight"}: bf16 {base:.2f}'
            entry = {'shape': (tokens, fin, fout), 'train_weight': train_weight, 'bf16': base}
            for recipe in recipes:
                for cache in ((False, True) if not train_weight else (False,)):
                    layer = linear_layer(recipe, fin, fout, cache=cache, requires_grad=train_weight)
                    layer.bias = None
                    try:
                        ms = cuda_time(lambda: layer(x).backward(grad), iterations)
                    except Exception as error:
                        line += f', {recipe}: {type(error).__name__}'
                        continue
                    name = recipe + (' cached' if cache else '')
                    line += f', {name} {ms:.2f} ({base / ms:.2f}x)'
                    entry[name] = ms
            report.line(line)
            report.record(entry)


def bench_attention(report, quick, iterations):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    shapes = [(1, 24, 4096, 128), (1, 24, 16384, 128), (2, 40, 8192, 128), (1, 24, 4096, 64)]
    if quick:
        shapes = shapes[:1] + shapes[-1:]
    report.line('  ms per forward+backward; TE fp8 speed-up over the fastest SDPA in brackets')
    for shape in shapes:
        q, k, v = (gen(*shape, seed=i).requires_grad_() for i in range(3))
        grad = gen(*shape, seed=5, scale=1e-2)
        times = {}
        for name, kernel in (('SDPA flash', SDPBackend.FLASH_ATTENTION), ('SDPA cuDNN', SDPBackend.CUDNN_ATTENTION)):
            try:
                with sdpa_kernel(kernel):
                    times[name] = cuda_time(lambda: REAL_SDPA(q, k, v).backward(grad), iterations)
            except Exception as error:
                times[name] = f'{type(error).__name__}'
        fresh(attention=True)
        block = CheckBlock(lambda q, k, v: F.scaled_dot_product_attention(q, k, v))
        fp8_training._attention.hook_blocks(block, ['CheckBlock'])
        try:
            attention_step(block, q, k, v, grad)

            def fp8():
                with fp8_training.step_scope(training=True):
                    block(q, k, v).backward(grad)
            times['TE fp8'] = cuda_time(fp8, iterations)
        except Exception as error:
            times['TE fp8'] = f'{type(error).__name__}: {error}'
        numbers = [t for n, t in times.items() if n != 'TE fp8' and isinstance(t, float)]
        line = f'  {shape}: ' + ', '.join(f'{n} {t:.2f}' if isinstance(t, float) else f'{n} {t}' for n, t in times.items())
        if numbers and isinstance(times['TE fp8'], float):
            line += f' ({min(numbers) / times["TE fp8"]:.2f}x)'
        report.line(line)
        report.record({'shape': shape, **times})


class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate='tanh'))


class TinyDiT(nn.Module):
    def __init__(self, dim, heads, depth, checkpointing):
        super().__init__()
        self.blocks = nn.ModuleList(DiTBlock(dim, heads) for _ in range(depth))
        self.checkpointing = checkpointing

    def forward(self, x):
        for block in self.blocks:
            if self.checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def build_model(base, variant, training):
    """A copy of `base` set up as train.py would: fp8 conversion before the adapter."""
    import peft
    model = copy.deepcopy(base)
    if variant != 'bf16':
        fresh(attention=variant == 'fp8 Linear + attention')
        fp8_training.convert(model, ['DiTBlock'], BF16)
        if variant == 'fp8 Linear + attention':
            fp8_training._attention.hook_blocks(model, ['DiTBlock'])
    else:
        attention.uninstall()
        fp8_training.reset(fp8_training._backend)
    if training == 'LoRA':
        config = peft.LoraConfig(r=64, lora_alpha=64, target_modules=['qkv', 'proj', 'fc1', 'fc2'])
        model = peft.get_peft_model(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    return model, torch.optim.AdamW(params, lr=1e-4 if training == 'LoRA' else 2e-5)


def train_steps(model, optimizer, batches, measure_from):
    losses, times = [], []
    for i, (x, target) in enumerate(batches):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with fp8_training.step_scope(training=True):
            loss = F.mse_loss(model(x).float(), target.float())
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        end.record()
        end.synchronize()
        losses.append(loss.item())
        if i >= measure_from:
            times.append(start.elapsed_time(end))
    return losses, statistics.median(times)


def check_training(report, quick, steps):
    dim, heads, depth, seq = (1536, 12, 2, 2048) if quick else (3072, 24, 4, 4096)
    report.line(f'  {depth} blocks of width {dim} ({heads} heads of 128), {seq} tokens, activation checkpointing; '
                f'{steps} steps; median step time from step 5 on')
    torch.manual_seed(0)
    base = TinyDiT(dim, heads, depth, checkpointing=True).to(DEVICE, BF16)
    # A target the model can learn: a fixed random linear map of the input.
    teacher = gen(dim, dim, seed=7, scale=dim ** -0.5)
    batches = []
    for i in range(steps):
        x = gen(1, seq, dim, seed=100 + i)
        batches.append((x, x @ teacher))
    for training in ('LoRA', 'full fine-tune'):
        results = {}
        for variant in ('bf16', 'fp8 Linear', 'fp8 Linear + attention'):
            try:
                model, optimizer = build_model(base, variant, training)
                results[variant] = train_steps(model, optimizer, batches, measure_from=5)
                del model, optimizer
            except Exception:
                report.line(f'FAIL  {training}, {variant}: crashed\n' + traceback.format_exc())
                report.record({'case': f'{training}, {variant}', 'crash': traceback.format_exc()}, False)
            torch.cuda.empty_cache()
        if 'bf16' not in results:
            continue
        losses, ms = results['bf16']
        tail = statistics.mean(losses[-5:])
        report.line(f'  {training}, bf16: {ms:.1f} ms/step, loss {losses[0]:.4f} -> {tail:.4f}')
        for variant, (v_losses, v_ms) in results.items():
            if variant == 'bf16':
                continue
            v_tail = statistics.mean(v_losses[-5:])
            finite = all(math.isfinite(x) for x in v_losses)
            # fp8 should learn as bf16 does: the end loss within 10% of bf16's.
            ok = finite and v_tail <= tail * 1.1 + 1e-4
            report.line(f'{"PASS" if ok else "FAIL"}  {training}, {variant}: {v_ms:.1f} ms/step ({ms / v_ms:.2f}x bf16), '
                        f'loss {v_losses[0]:.4f} -> {v_tail:.4f}')
            report.record({'case': f'{training}, {variant}', 'ms': v_ms, 'bf16_ms': ms, 'speedup': ms / v_ms,
                           'losses': v_losses, 'bf16_losses': losses}, ok)
        report.line(f'  counts of the last run: Linear {fp8_training.counts()[0]}, attention {fp8_training.counts()[1]}')


def check_compile(report):
    fresh()
    torch.manual_seed(0)
    model = TinyDiT(1024, 8, 1, checkpointing=False).to(DEVICE, BF16)
    reference = copy.deepcopy(model)
    fp8_training.convert(model, ['DiTBlock'], BF16)
    x = gen(1, 1024, 1024)
    try:
        compiled = torch.compile(model)
        out = compiled(x)
        out.float().square().mean().backward()
        error = rel_error(out, reference(x))
        ok = error <= LINEAR_LIMIT * 2
        report.line(f'{"PASS" if ok else "FAIL"}  torch.compile with fp8 Linear layers runs; output vs bf16 {error:.4f}')
        report.record({'case': 'compile', 'error': error}, ok)
    except Exception as error:
        report.line(f'FAIL  torch.compile with fp8 Linear layers: {type(error).__name__}: {error}'.splitlines()[0])
        report.line('  (set compile = false with [fp8_training], and send this report back)')
        report.record({'case': 'compile', 'crash': traceback.format_exc()}, False)


def check_distributed(report, steps=8):
    """Data parallel over every rank (gradients all-reduced, as DeepSpeed does)."""
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.manual_seed(0)
    model = TinyDiT(1024, 8, 2, checkpointing=True).to(DEVICE, BF16)
    fresh(attention=True)
    fp8_training.convert(model, ['DiTBlock'], BF16)
    fp8_training._attention.hook_blocks(model, ['DiTBlock'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for step in range(steps):
        x = gen(1, 1024, 1024, seed=1000 * rank + step)
        with fp8_training.step_scope(training=True):
            model(x).float().square().mean().backward()
        for p in model.parameters():
            torch.distributed.all_reduce(p.grad)
            p.grad /= world
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    fp8_training.report()
    sites = sorted((repr(key), site.state) for key, site in fp8_training._attention.sites.items())
    flat = torch.cat([p.detach().float().flatten() for p in model.parameters()])
    spread = flat.clone()
    torch.distributed.all_reduce(spread, op=torch.distributed.ReduceOp.MAX)
    low = flat.clone()
    torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN)
    drift = float((spread - low).abs().max())
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, {'sites': sites, 'counts': fp8_training.counts()})
    same_sites = all(g['sites'] == gathered[0]['sites'] for g in gathered)
    all_fp8 = all(g['counts'][1].get('fp8', 0) > 0 and g['counts'][0].get('fp8', 0) > 0 for g in gathered)
    ok = same_sites and all_fp8 and drift == 0
    if report is not None:
        report.line(f'{"PASS" if ok else "FAIL"}  {world} ranks, {steps} steps: same call sites on every rank '
                    f'{same_sites}, fp8 ran on every rank {all_fp8}, largest weight difference between ranks {drift}')
        report.record({'case': 'distributed', 'ranks': gathered, 'drift': drift}, ok)


def section(report, title, fn, *args):
    report.section(title)
    try:
        fn(report, *args)
    except Exception:
        report.line('  section crashed:\n' + traceback.format_exc())
        report.failures.append({'section': title})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default='.', help='Where the report goes (default: the current directory).')
    parser.add_argument('--quick', action='store_true', help='Smaller benchmarks and training run.')
    parser.add_argument('--iterations', type=int, default=20, help='Timed runs per benchmark.')
    parser.add_argument('--steps', type=int, default=40, help='Training steps per run in section 6.')
    parser.add_argument('--skip-benchmarks', action='store_true')
    parser.add_argument('--distributed', action='store_true',
                        help='Under torchrun: run the multi-GPU check (section 8) on every rank, then the rest on rank 0.')
    args = parser.parse_args(argv)

    distributed = args.distributed and int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = 0
    if distributed:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        torch.distributed.init_process_group('nccl')
        rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device(0)
    report = None
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        report = Report(args.out_dir, prefix='fp8_training_check')
        report.watch_warnings()
        report.line(f'fp8 training check, {datetime.datetime.now().isoformat(timespec="seconds")}; report: '
                    f'{report.txt_path}')
    try:
        if distributed:
            if report is not None:
                report.section('8. Several GPUs')
            usable = None
            try:
                fp8_training.check_environment(fp8_config(recipe='current', attention=True))
                usable = True
            except Exception as error:
                if report is not None:
                    report.line(f'  fp8 attention cannot run here ({error}); not checked')
            if usable:
                check_distributed(report)
            torch.distributed.barrier()
            if rank != 0:
                return 0
        usable = check_environment(report)
        if not torch.cuda.is_available():
            report.line('No CUDA GPU: nothing else to check.')
            return 1
        recipes = [r for r in fp8_training.RECIPES if usable[f"recipe '{r}'"] is None]
        if not recipes:
            report.line('  Neither recipe can run here: nothing else to check.')
            return 1
        fp8_training.check_environment(fp8_config(recipe=recipes[-1]))
        section(report, '2. Linear layers: error against float32', check_linear, recipes)
        if usable['attention'] is None:
            section(report, '3. fp8 attention: error against float32', check_attention)
        if not args.skip_benchmarks:
            section(report, '4. Linear speed (forward + backward)', bench_linear, recipes, args.quick, args.iterations)
            if usable['attention'] is None:
                section(report, '5. Attention speed (forward + backward)', bench_attention, args.quick, args.iterations)
        if usable['attention'] is None:
            section(report, '6. Training a small DiT', check_training, args.quick, args.steps)
        section(report, '7. torch.compile', check_compile)
        report.section('Summary')
        failed = len(report.failures)
        report.line(f'  {failed} failed check(s)' + (': ' + '; '.join(
            str(f.get('case') or f.get('check') or f.get('section')) for f in report.failures) if failed else ''))
        report.results['failures'] = report.failures
        if report.warnings:
            report.line(f'  {len(report.warnings)} distinct warning(s) were raised:')
            for text in report.warnings:
                report.line(f'    {text}')
            report.results['warnings'] = report.warnings
        report.line(f'  Send back: {report.txt_path}')
        return 1 if failed else 0
    finally:
        attention.uninstall()
        if report is not None:
            report.close()
        if distributed:
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
