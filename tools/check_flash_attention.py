"""All-in-one check of [model] attention = 'flash3' / 'flash4' on one GPU: run it, send back the report.

    python tools/check_flash_attention.py                 # everything, ~5-15 minutes on an H100/H200
                                                          # (more with FA4, which compiles each kernel first)
    python tools/check_flash_attention.py --quick         # smaller benchmarks
    python tools/check_flash_attention.py --out-dir /data/reports

Run from the diffusion-pipe checkout, in the environment you train with. It writes
flash_attention_check_<host>_<time>.txt (and a .json of the same results) as it goes, so a crash
keeps everything up to that point; send the .txt back. Nothing is downloaded and no model weights
are needed: every check uses random tensors.

What it checks, for FlashAttention-3 (flash_attn_3 / flash_attn_interface) and FlashAttention-4
(flash-attn-4: flash_attn.cute), whichever import:
  1. environment: Python, torch, CUDA, cuDNN, driver, GPU, and which flash-attn packages import;
  2. the SDPA replacement (utils/attention.py) on ~25 shapes and masks: that each call takes the
     route it should (FlashAttention, FlashAttention with key padding, or PyTorch SDPA with the
     expected reason), and that output and q/k/v gradients match a float32 reference as closely as
     PyTorch's own bf16 SDPA does;
  3. the helpers the flash-attn models use (varlen, unpadded, sliding window);
  4. a small transformer trained 40 steps with SDPA and with FlashAttention (with and without
     activation checkpointing): the losses must agree;
  5. deterministic mode: two backward passes give bitwise equal gradients;
  6. the models that call flash-attn directly (Wan, HiDream, HunyuanVideo, HunyuanImage, OmniGen2)
     plus Lumina 2, and Iris's fa3 / fa4 backends, where their modules import;
  7. speed and memory of forward+backward: PyTorch SDPA (its own pick, flash, cuDNN, efficient)
     against FA3 and FA4, on shapes of the models this repo trains.
"""

import argparse
import contextlib
import datetime
import importlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import utils.common  # noqa: E402,F401  (claims the utils package before ComfyUI joins sys.path)
from utils import attention  # noqa: E402

REAL_SDPA = F.scaled_dot_product_attention
FLASH = ('flash3', 'flash4')
DEVICE = 'cuda'


class Report:
    """Lines to the console and the .txt as they come (flushed), results to the .json at the end."""

    def __init__(self, out_dir):
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        base = os.path.join(out_dir, f'flash_attention_check_{socket.gethostname()}_{stamp}')
        self.txt_path, self.json_path = base + '.txt', base + '.json'
        self.file = open(self.txt_path, 'w')
        self.results = {'sections': {}}
        self.failures = []

    def line(self, text=''):
        print(text, flush=True)
        self.file.write(text + '\n')
        self.file.flush()

    def section(self, title):
        self.line()
        self.line(title)
        self.line('-' * len(title))
        self.current = self.results['sections'].setdefault(title, [])

    def record(self, entry, ok=None):
        self.current.append(entry)
        if ok is False:
            self.failures.append(entry)

    def watch_warnings(self):
        """Remembers each distinct warning the run raises (FA4's JIT, say), still printing it as usual."""
        self.warnings = {}
        shown = warnings.showwarning

        def record(message, category, filename, lineno, file=None, line=None):
            key = f'{category.__name__}: {str(message)[:400]}'
            self.warnings[key] = True
            shown(message, category, filename, lineno, file, line)

        warnings.showwarning = record
        warnings.simplefilter('default')

    def close(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.results, f, indent=1, default=str)
        self.file.close()


def version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def try_import(name, attr=None):
    try:
        module = importlib.import_module(name)
        if attr and not hasattr(module, attr):
            return f'imports, but has no {attr}'
        return 'ok'
    except Exception as error:
        return f'{type(error).__name__}: {error}'.splitlines()[0][:200]


def environment(report):
    report.section('1. Environment')
    info = {
        'host': socket.gethostname(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torch CUDA': torch.version.cuda,
        'cuDNN': torch.backends.cudnn.version(),
        'GPU count': torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['GPU'] = f'{props.name}, compute capability {props.major}.{props.minor}, ' \
                      f'{props.total_memory / 2**30:.0f} GiB, {props.multi_processor_count} SMs'
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name,clocks.max.sm,power.limit',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        info['nvidia-smi'] = smi.stdout.strip().splitlines()[0] if smi.stdout.strip() else smi.stderr.strip()
    except Exception as error:
        info['nvidia-smi'] = repr(error)
    for dist in ('flash_attn', 'flash_attn_3', 'flash-attn-4', 'nvidia-cutlass-dsl', 'quack-kernels', 'triton',
                 'deepspeed', 'diffusers', 'transformers', 'peft'):
        info[f'package {dist}'] = version(dist)
    info['import flash_attn (2)'] = try_import('flash_attn', 'flash_attn_varlen_func')
    info['import flash_attn_interface (3)'] = try_import('flash_attn_interface')
    info['import flash_attn_3.flash_attn_interface (3)'] = try_import('flash_attn_3.flash_attn_interface')
    info['import flash_attn.cute (4)'] = try_import('flash_attn.cute', 'flash_attn_varlen_func')
    try:
        info['diffusion-pipe commit'] = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                                       capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        info['diffusion-pipe local changes'] = dirty.replace('\n', '; ') or 'none'
    except Exception as error:
        info['diffusion-pipe commit'] = repr(error)
    for key, value in info.items():
        report.line(f'  {key}: {value}')
    report.results['environment'] = info


def use_backend(backend):
    """Switches utils.attention to a backend; returns None or the reason it cannot run here."""
    attention.uninstall()
    attention.reset_counts()
    try:
        attention.configure({'model': {'type': 'check', 'attention': backend}})
    except Exception as error:
        attention.configure({'model': {'type': 'check'}})
        return str(error)
    if backend == 'sdpa':
        attention._module = None
    return None


def rel_error(value, ref):
    value, ref = value.float(), ref.float()
    if not torch.isfinite(value).all():
        return math.inf
    # The floor keeps a reference that is ~0 (dq when every query sees one key) from blowing it up.
    return float((value - ref).norm() / ref.norm().clamp_min(1e-3 * ref.numel() ** 0.5))


def sdpa_reference(q, k, v, **kwargs):
    """float32 math SDPA: output and q/k/v gradients for the grad given in kwargs['grad']."""
    grad = kwargs.pop('grad')
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    mask = kwargs.get('attn_mask')
    if mask is not None and mask.is_floating_point():
        kwargs['attn_mask'] = mask.float()
    with sdpa_kernel(SDPBackend.MATH):
        out = REAL_SDPA(q, k, v, **kwargs)
    out.backward(grad.float())
    return out.detach(), q.grad, k.grad, v.grad


def run_sdpa(fn, q, k, v, grad, **kwargs):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = fn(q, k, v, **kwargs)
    out.backward(grad.to(out.dtype))
    return out.detach(), q.grad, k.grad, v.grad


def gen(shape, dtype, seed, layout='bhld'):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if layout == 'blhd':
        b, h, l, d = shape
        return torch.randn(b, l, h, d, generator=g, device=DEVICE).to(dtype).transpose(1, 2)
    return torch.randn(*shape, generator=g, device=DEVICE).to(dtype)


def key_padding(lengths, lk):
    return torch.arange(lk, device=DEVICE)[None, :] < torch.tensor(lengths, device=DEVICE)[:, None]


def correctness_cases():
    """(name, b, hq, hk, lq, lk, d, options, expected route): route is 'flash', 'padded' or a
    substring of the reason the call stays on SDPA."""
    bf = torch.bfloat16
    cases = [
        ('self-attention d128', 2, 8, 8, 1024, 1024, 128, {}, 'flash'),
        ('self-attention d64', 2, 8, 8, 1024, 1024, 64, {}, 'flash'),
        ('self-attention d96', 1, 8, 8, 512, 512, 96, {}, 'flash'),
        ('self-attention d192', 1, 4, 4, 512, 512, 192, {}, 'flash'),
        ('self-attention d256', 1, 4, 4, 512, 512, 256, {}, 'flash'),
        ('BLHD layout (transposed views)', 2, 8, 8, 1024, 1024, 128, {'layout': 'blhd'}, 'flash'),
        ('odd length 1337', 2, 8, 8, 1337, 1337, 128, {}, 'flash'),
        ('length 1', 2, 8, 8, 1, 1, 128, {}, 'flash'),
        ('GQA 20/5 heads, 1326 tokens (Iris)', 2, 20, 5, 1326, 1326, 128, {'enable_gqa': True}, 'flash'),
        ('cross-attention 1024 -> 77', 2, 8, 8, 1024, 77, 128, {}, 'flash'),
        ('cross-attention 77 -> 1024', 2, 8, 8, 77, 1024, 128, {}, 'flash'),
        ('causal, equal lengths', 2, 8, 8, 512, 512, 128, {'is_causal': True}, 'flash'),
        ('custom scale 0.05', 2, 8, 8, 512, 512, 128, {'scale': 0.05}, 'flash'),
        ('float16', 2, 8, 8, 1024, 1024, 128, {'dtype': torch.float16}, 'flash'),
        ('float32 under bf16 autocast', 2, 8, 8, 1024, 1024, 128, {'dtype': torch.float32, 'autocast': bf},
         'flash'),
        ('key padding, bool (B,1,1,Lk)', 4, 8, 8, 600, 600, 128, {'mask': ('bool', [600, 450, 300, 17])},
         'padded'),
        ('key padding, float -inf', 4, 8, 8, 600, 600, 128, {'mask': ('-inf', [600, 451, 300, 17])}, 'padded'),
        ('key padding, finfo.min (B,1,Lq,Lk)', 4, 8, 8, 600, 600, 128, {'mask': ('min-full', [600, 1, 333, 599])},
         'padded'),
        ('cross-attention, padded text 1024 -> 512', 2, 8, 8, 1024, 512, 128, {'mask': ('bool', [512, 77])},
         'padded'),
        ('key padding with GQA', 2, 20, 5, 700, 700, 128, {'mask': ('bool', [700, 123]), 'enable_gqa': True},
         'padded'),
        ('mask keeping every key', 2, 8, 8, 512, 512, 128, {'mask': ('bool', [512, 512])}, 'flash'),
        ('causal with Lq != Lk', 2, 8, 8, 256, 512, 128, {'is_causal': True}, 'different query and key lengths'),
        ('causal mask tensor', 2, 8, 8, 512, 512, 128, {'mask': ('causal', None)}, 'not key padding'),
        ('additive bias', 2, 8, 8, 512, 512, 128, {'mask': ('bias', None)}, 'bias values'),
        ('float32 without autocast', 2, 8, 8, 512, 512, 128, {'dtype': torch.float32}, 'float32 inputs'),
        ('head_dim 512 (VAE attention)', 1, 1, 1, 1024, 1024, 512, {}, 'head_dim 512'),
        ('batch item with no keys', 2, 8, 8, 128, 128, 128, {'mask': ('bool', [128, 0])}, 'hides every key'),
    ]
    return cases


def build_mask(kind, lengths, b, lq, lk, dtype):
    if kind == 'bool':
        return key_padding(lengths, lk)[:, None, None, :]
    if kind == '-inf':
        return torch.zeros(b, 1, 1, lk, device=DEVICE, dtype=dtype).masked_fill(
            ~key_padding(lengths, lk)[:, None, None, :], float('-inf'))
    if kind == 'min-full':
        keep = key_padding(lengths, lk)[:, None, None, :].expand(b, 1, lq, lk)
        return torch.zeros(b, 1, lq, lk, device=DEVICE, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)
    if kind == 'causal':
        return torch.ones(lq, lk, device=DEVICE, dtype=torch.bool).tril()
    if kind == 'bias':
        g = torch.Generator(device=DEVICE).manual_seed(7)
        return torch.randn(b, 1, lq, lk, generator=g, device=DEVICE).to(dtype)
    raise ValueError(kind)


def route_of(before, after, expected):
    gained = {key: after.get(key, 0) - before.get(key, 0) for key in after if after.get(key, 0) != before.get(key, 0)}
    if gained.get(attention.FLASH_CALL):
        return 'flash', gained
    if gained.get(attention.FLASH_PADDED):
        return 'padded', gained
    return ', '.join(gained) or 'nothing counted', gained


def check_replacement(report, backend):
    """Section 2: every case through the replaced F.scaled_dot_product_attention."""
    attention.install()
    for name, b, hq, hk, lq, lk, d, opts, expected in correctness_cases():
        if expected in ('flash', 'padded') and not attention.head_dim_supported(d):
            # FA4 on Blackwell takes fewer head sizes (README "FlashAttention"): SDPA by design.
            expected = f'head_dim {d}'
        entry = {'backend': backend, 'case': name, 'expected': expected}
        try:
            dtype = opts.get('dtype', torch.bfloat16)
            q = gen((b, hq, lq, d), dtype, 1, opts.get('layout', 'bhld'))
            k = gen((b, hk, lk, d), dtype, 2, opts.get('layout', 'bhld'))
            v = gen((b, hk, lk, d), dtype, 3, opts.get('layout', 'bhld'))
            grad = gen((b, hq, lq, d), dtype, 4)
            kwargs = {}
            for key in ('is_causal', 'scale', 'enable_gqa'):
                if key in opts:
                    kwargs[key] = opts[key]
            if 'mask' in opts:
                kind, lengths = opts['mask']
                kwargs['attn_mask'] = build_mask(kind, lengths, b, lq, lk,
                                                 opts.get('autocast', dtype) if dtype == torch.float32 else dtype)
            ref = sdpa_reference(q, k, v, grad=grad, **kwargs)
            autocast = torch.autocast(DEVICE, dtype=opts['autocast']) if 'autocast' in opts else contextlib.nullcontext()
            with autocast:
                before = attention.counts()
                mine = run_sdpa(F.scaled_dot_product_attention, q, k, v, grad, **kwargs)
                route, gained = route_of(before, attention.counts(), expected)
                theirs = run_sdpa(REAL_SDPA, q, k, v, grad, **kwargs)
            errors = [rel_error(m, r) for m, r in zip(mine, ref)]
            baseline = [rel_error(t, r) for t, r in zip(theirs, ref)]
            routed = route == expected if expected in ('flash', 'padded') else expected in route
            if expected == 'hides every key':
                # SDPA itself gives NaN for that batch item; only the route is checked.
                accurate = True
            else:
                accurate = all(e <= 2 * base + 2e-3 for e, base in zip(errors, baseline))
            ok = routed and accurate
            entry.update(route=route, counted=gained, errors_out_dq_dk_dv=errors, sdpa_bf16_errors=baseline, ok=ok)
            report.line(f'  [{"PASS" if ok else "FAIL"}] {name}: route {route}'
                        f'{"" if routed else f" (expected {expected})"}; rel. error out/dq/dk/dv '
                        + '/'.join(f'{e:.1e}' for e in errors) + ' (PyTorch bf16 SDPA '
                        + '/'.join(f'{e:.1e}' for e in baseline) + ')')
        except Exception as error:
            ok = False
            entry.update(ok=False, error=traceback.format_exc())
            report.line(f'  [FAIL] {name}: {type(error).__name__}: {error}'.splitlines()[0])
        report.record(entry, ok)
        torch.cuda.empty_cache()
    attention.uninstall()


def check_helpers(report, backend):
    """Section 3: flash_varlen / flash_unpadded / sliding window against float32 math."""
    checks = []
    b, s, h, d = 3, 700, 8, 128
    lengths = [700, 513, 60]
    keep = key_padding(lengths, s)

    def unpadded():
        q, k, v = (gen((b, s, h, d), torch.bfloat16, i) for i in range(3))
        out = attention.flash_unpadded(q, k, v, keep)
        ref = REAL_SDPA(*(t.float().transpose(1, 2) for t in (q, k, v)), attn_mask=keep[:, None, None, :])
        ref = ref.transpose(1, 2) * keep[:, :, None, None]
        zero = bool((out[~keep] == 0).all())
        return rel_error(out, ref), zero

    def varlen():
        total = sum(lengths)
        q, k, v = (gen((1, total, h, d), torch.bfloat16, 10 + i)[0] for i in range(3))
        cu = torch.tensor([0, 700, 1213, 1273], dtype=torch.int32, device=DEVICE)
        out = attention.flash_varlen(q, k, v, cu, cu, 700, 700, causal=True)
        ref = torch.cat([REAL_SDPA(*(t[a:z].float().transpose(0, 1) for t in (q, k, v)), is_causal=True)
                         .transpose(0, 1) for a, z in ((0, 700), (700, 1213), (1213, 1273))])
        return rel_error(out, ref), True

    def window():
        q, k, v = (gen((2, 512, h, d), torch.bfloat16, 20 + i) for i in range(3))
        out = attention.flash_dense(q, k, v, window_size=(64, 64))
        i = torch.arange(512, device=DEVICE)
        band = (i[None, :] - i[:, None]).abs() <= 64
        ref = REAL_SDPA(*(t.float().transpose(1, 2) for t in (q, k, v)), attn_mask=band).transpose(1, 2)
        return rel_error(out, ref), True

    for name, fn in (('flash_unpadded (HunyuanImage)', unpadded), ('flash_varlen causal (Wan / HunyuanVideo)', varlen),
                     ('flash_dense sliding window 64', window)):
        try:
            error, extra = fn()
            ok = error < 1e-2 and extra
            report.line(f'  [{"PASS" if ok else "FAIL"}] {name}: rel. error {error:.1e}'
                        + ('' if extra else ', padded rows not zero'))
            report.record({'backend': backend, 'check': name, 'error': error, 'ok': ok}, ok)
        except Exception as error:
            report.line(f'  [FAIL] {name}: {type(error).__name__}: {error}'.splitlines()[0])
            report.record({'backend': backend, 'check': name, 'error': traceback.format_exc(), 'ok': False}, False)


class TinyBlock(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = torch.nn.LayerNorm(dim), torch.nn.LayerNorm(dim)
        self.qkv, self.out = torch.nn.Linear(dim, 3 * dim), torch.nn.Linear(dim, dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, 4 * dim), torch.nn.GELU(), torch.nn.Linear(4 * dim, dim))

    def forward(self, x, mask):
        b, s, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, c // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        x = x + self.out(a.transpose(1, 2).reshape(b, s, c))
        return x + self.mlp(self.norm2(x))


def tiny_training(checkpointing, steps=40):
    torch.manual_seed(0)
    dim, heads, s, b = 512, 4, 384, 4
    blocks = torch.nn.ModuleList(TinyBlock(dim, heads) for _ in range(3)).to(DEVICE)
    opt = torch.optim.AdamW(blocks.parameters(), lr=3e-4)
    g = torch.Generator(device=DEVICE).manual_seed(1)
    x = torch.randn(b, s, dim, generator=g, device=DEVICE)
    target = torch.randn(b, s, dim, generator=g, device=DEVICE)
    mask = key_padding([384, 300, 200, 97], s)[:, None, None, :]
    losses = []
    for _ in range(steps):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            h = x
            for block in blocks:
                h = torch.utils.checkpoint.checkpoint(block, h, mask, use_reentrant=False) if checkpointing \
                    else block(h, mask)
            keep = mask[:, 0, 0, :, None]
            loss = ((h.float() - target) ** 2 * keep).sum() / keep.sum() / dim
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in blocks.parameters())
        if not finite:
            return losses, False
        opt.step()
        losses.append(loss.item())
    return losses, True


def check_training(report, backend, baselines):
    """Section 4: the same tiny run with SDPA and with FlashAttention."""
    for checkpointing in (False, True):
        name = f'tiny transformer, 40 AdamW steps, key padding, {"with" if checkpointing else "no"} checkpointing'
        try:
            if checkpointing not in baselines:
                attention.uninstall()
                baselines[checkpointing] = tiny_training(checkpointing)[0]
            attention.install()
            attention.reset_counts()
            losses, finite = tiny_training(checkpointing)
            counted = attention.counts()
            attention.uninstall()
            base = baselines[checkpointing]
            diff = max(abs(a - b) / abs(b) for a, b in zip(losses, base)) if len(losses) == len(base) else math.inf
            ok = finite and diff < 0.05 and counted.get(attention.FLASH_PADDED, 0) > 0
            report.line(f'  [{"PASS" if ok else "FAIL"}] {name}: loss {base[0]:.4f} -> SDPA {base[-1]:.4f}, '
                        f'{attention.label()} {losses[-1] if losses else float("nan"):.4f}; largest relative '
                        f'difference along the run {diff:.1e}; gradients {"finite" if finite else "NOT FINITE"}; '
                        f'calls {counted}')
            report.record({'backend': backend, 'check': name, 'sdpa_losses': base, 'losses': losses,
                           'max_rel_diff': diff, 'finite': finite, 'counted': counted, 'ok': ok}, ok)
        except Exception as error:
            attention.uninstall()
            report.line(f'  [FAIL] {name}: {type(error).__name__}: {error}'.splitlines()[0])
            report.record({'backend': backend, 'check': name, 'error': traceback.format_exc(), 'ok': False}, False)


def check_determinism(report, backend):
    """Section 5: deterministic backward, twice, bitwise."""
    name = 'deterministic backward (torch.use_deterministic_algorithms)'
    try:
        q, k, v = (gen((2, 8, 2048, 128), torch.bfloat16, 30 + i) for i in range(3))
        grad = gen((2, 8, 2048, 128), torch.bfloat16, 34)
        mask = key_padding([2048, 1500], 2048)[:, None, None, :]
        attention.install()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            first = run_sdpa(F.scaled_dot_product_attention, q, k, v, grad, attn_mask=mask)
            second = run_sdpa(F.scaled_dot_product_attention, q, k, v, grad, attn_mask=mask)
        finally:
            torch.use_deterministic_algorithms(False)
            attention.uninstall()
        same = all(torch.equal(a, b) for a, b in zip(first, second))
        report.line(f'  [{"PASS" if same else "FAIL"}] {name}: gradients {"identical" if same else "differ"}')
        report.record({'backend': backend, 'check': name, 'identical': same, 'ok': same}, same)
    except Exception as error:
        attention.uninstall()
        report.line(f'  [FAIL] {name}: {type(error).__name__}: {error}'.splitlines()[0])
        report.record({'backend': backend, 'check': name, 'error': traceback.format_exc(), 'ok': False}, False)


def model_checks():
    """(name, setup) pairs: setup() imports the model's module and returns run(), which calls the
    model's own attention and returns (output, float32 reference)."""

    def wan():
        from models.wan import attention as wan_attention
        b, lq, lk, h, d = 2, 1560, 512, 12, 128
        q, k, v = gen((b, lq, h, d), torch.bfloat16, 40), gen((b, lk, h, d), torch.bfloat16, 41), \
            gen((b, lk, h, d), torch.bfloat16, 42)
        k_lens = torch.tensor([512, 77], dtype=torch.int32, device=DEVICE)

        def run():
            out = wan_attention.flash_attention(q, k, v, k_lens=k_lens)
            ref = REAL_SDPA(*(t.float().transpose(1, 2) for t in (q, k, v)),
                            attn_mask=key_padding([512, 77], lk)[:, None, None, :]).transpose(1, 2)
            return out, ref
        return run

    def hidream():
        import models.hidream as hidream_model
        q, k, v = (gen((2, 1024, 8, 128), torch.bfloat16, 50 + i) for i in range(3))

        def run():
            out = hidream_model.hidream_attention_processor.attention(q, k, v)
            ref = REAL_SDPA(*(t.float().transpose(1, 2) for t in (q, k, v))).transpose(1, 2).flatten(-2)
            return out, ref
        return run

    def hunyuan_video():
        import models.hunyuan_video  # noqa: F401
        from hyvideo.modules import attenion
        b, s, h, d = 2, 1280, 8, 128
        q, k, v = (gen((b, s, h, d), torch.bfloat16, 60 + i) for i in range(3))
        lengths = [1280, 1100]
        cu = torch.tensor([0, 1280, 1280, 2380, 2560], dtype=torch.int32, device=DEVICE)

        def run():
            out = attenion.attention(q, k, v, mode='flash', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=s,
                                     max_seqlen_kv=s, batch_size=b)
            keep = key_padding(lengths, s)
            ref = REAL_SDPA(*(t.float().transpose(1, 2) for t in (q, k, v)),
                            attn_mask=keep[:, None, None, :]).transpose(1, 2).flatten(-2)
            return out * keep[..., None], ref * keep[..., None]
        return run

    def hunyuan_image():
        sys.path.insert(0, os.path.join(ROOT, 'submodules', 'HunyuanImage-2.1'))
        import models.hunyuan_image_modeling as modeling
        b, si, st, h, d = 2, 1024, 256, 8, 128
        img = [gen((b, si, h, d), torch.bfloat16, 70 + i) for i in range(3)]
        txt = [gen((b, st, h, d), torch.bfloat16, 80 + i) for i in range(3)]
        text_mask = key_padding([256, 40], st)

        def run():
            out = modeling.attention(*zip(img, txt), text_mask=text_mask)
            keep = torch.cat([torch.ones(b, si, dtype=torch.bool, device=DEVICE), text_mask], 1)
            ref = REAL_SDPA(*(torch.cat(p, 1).float().transpose(1, 2) for p in zip(img, txt)),
                            attn_mask=keep[:, None, None, :]).transpose(1, 2).flatten(-2) * keep[..., None]
            return out, ref
        return run

    def omnigen2():
        import models.omnigen2 as omnigen2_model
        processor = omnigen2_model.omnigen2_attention_processor
        if omnigen2_model._flash_attn_2_varlen is None:
            raise RuntimeError('flash-attn 2 is not installed: OmniGen2 uses its SDPA processor, which the SDPA '
                               'replacement covers (section 2)')
        q, k, v = (gen((1, 2000, 8, 128), torch.bfloat16, 90 + i)[0] for i in range(3))
        cu = torch.tensor([0, 1200, 2000], dtype=torch.int32, device=DEVICE)

        def run():
            out = processor.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1200,
                                                   max_seqlen_k=1200, dropout_p=0.0, causal=False,
                                                   softmax_scale=0.1)
            ref = torch.cat([REAL_SDPA(*(t[a:z].float().transpose(0, 1) for t in (q, k, v)), scale=0.1)
                             .transpose(0, 1) for a, z in ((0, 1200), (1200, 2000))])
            return out, ref
        return run

    def lumina_2():
        import models.lumina_2 as lumina_2_pipeline
        model = lumina_2_pipeline.lumina_2_model
        torch.manual_seed(0)
        attn = model.JointAttention(2304, 24, 8, qk_norm=True).to(DEVICE)
        b, s = 2, 1280
        x = gen((b, s, 2304), torch.float32, 110)
        x_mask = key_padding([1280, 900], s)
        angles = gen((b, s, 48), torch.float32, 111)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        def run():
            out = attn.to(torch.bfloat16)(x.to(torch.bfloat16), x_mask, freqs_cis)
            # Lumina's own float32 path: SDPA with the mask, keys and values repeated for GQA.
            ref = attn.float()(x, x_mask, freqs_cis)
            return out[x_mask], ref[x_mask]
        return run

    def iris():
        sys.path.insert(0, os.path.join(ROOT, 'submodules', 'iris-3b', 'src'))
        from iris3b.nn import attention as iris_attention
        q = gen((2, 20, 1326, 128), torch.bfloat16, 100)
        k, v = gen((2, 5, 1326, 128), torch.bfloat16, 101), gen((2, 5, 1326, 128), torch.bfloat16, 102)
        backend = {'flash3': 'fa3', 'flash4': 'fa4'}.get(attention._backend, 'sdpa')

        def run():
            out = iris_attention.scaled_dot_product(q, k, v, backend=backend, enable_gqa=True)
            ref = REAL_SDPA(q.float(), k.float(), v.float(), enable_gqa=True)
            return out, ref
        return run

    return [('Wan (flash_attention, k_lens)', wan), ('HiDream (attention)', hidream),
            ('HunyuanVideo (varlen, padding as its own sequence)', hunyuan_video),
            ('HunyuanImage (unpadded image + text)', hunyuan_image), ('OmniGen2 (flash-attn 2 processor)', omnigen2),
            ('Lumina 2 (JointAttention, 24/8 heads, padded)', lumina_2), ('Iris (attn_backend fa3 / fa4)', iris)]


def check_models(report, backends):
    """Section 6: each model's own attention call with every backend that runs here."""
    try:
        import models.base  # noqa: F401  (sets up sys.path for the submodules)
    except Exception as error:
        report.line(f'  models.base did not import, model checks skipped: {error}'.splitlines()[0])
        return
    for name, setup in model_checks():
        for backend in ['sdpa', *backends]:
            label = f'{name} with attention = {backend!r}'
            if use_backend(backend):
                continue
            # As train.py does after caching: the models' FlashAttention calls run from here on.
            attention.install()
            try:
                run = setup()
            except Exception as error:
                report.line(f'  [SKIP] {label}: {type(error).__name__}: {error}'.splitlines()[0][:300])
                report.record({'check': label, 'skipped': str(error)})
                break
            try:
                out, ref = run()
            except ImportError as error:
                # Without flash-attn 2, the model has no kernel for attention = 'sdpa' either.
                report.line(f'  [SKIP] {label}: {type(error).__name__}: {error}'.splitlines()[0][:300])
                report.record({'check': label, 'skipped': str(error)})
                continue
            except Exception as error:
                report.line(f'  [FAIL] {label}: {type(error).__name__}: {error}'.splitlines()[0][:300])
                report.record({'check': label, 'error': traceback.format_exc(), 'ok': False}, False)
                continue
            try:
                error = rel_error(out, ref)
                ok = error < 1e-2
                report.line(f'  [{"PASS" if ok else "FAIL"}] {label}: rel. error {error:.1e}')
                report.record({'check': label, 'error': error, 'ok': ok}, ok)
            except Exception as error:
                report.line(f'  [FAIL] {label}: {type(error).__name__}: {error}'.splitlines()[0][:300])
                report.record({'check': label, 'error': traceback.format_exc(), 'ok': False}, False)
    use_backend('sdpa')


def bench_shapes(quick):
    """(name, b, hq, hk, l, d, key lengths or None)."""
    shapes = [
        ('MOSS / Flux-like 512px: 1024 tokens', 16, 24, 24, 1024, 128, None),
        ('MOSS 1024px: 4096 tokens', 4, 24, 24, 4096, 128, None),
        ('Flux 1024px: 4096 + 512 text', 2, 24, 24, 4608, 128, None),
        ('Iris GQA 20/5, 1326 tokens', 16, 20, 5, 1326, 128, None),
        ('SDXL-like head_dim 64, 4096 tokens', 4, 10, 10, 4096, 64, None),
        ('key padding 2048 tokens (lengths 100% / 75% / 50% / 25%)', 4, 24, 24, 2048, 128, [2048, 1536, 1024, 512]),
    ]
    if not quick:
        shapes += [
            ('Wan 14B 480p 81 frames: 32760 tokens', 1, 40, 40, 32760, 128, None),
            ('HunyuanVideo 720p 33 frames: 32400 tokens', 1, 24, 24, 32400, 128, None),
            ('MOSS 2048px: 16384 tokens', 1, 24, 24, 16384, 128, None),
        ]
    return shapes


def time_fwd_bwd(fn, q, k, v, grad, kwargs, iterations):
    def step():
        qs, ks, vs = (t.detach().requires_grad_() for t in (q, k, v))
        out = fn(qs, ks, vs, **kwargs)
        out.backward(grad)

    torch.cuda.synchronize()
    start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    for _ in range(2):
        step()
    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()
    times = []
    for _ in range(iterations):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(begin.elapsed_time(end))
    times.sort()
    peak = (torch.cuda.max_memory_allocated() - base_memory) / 2**20
    return times[len(times) // 2], first, peak


def benchmarks(report, backends, quick, iterations):
    """Section 7: forward+backward time per kernel."""
    methods = [('SDPA, PyTorch picks', None), ('SDPA flash (FA2)', SDPBackend.FLASH_ATTENTION),
               ('SDPA cuDNN', SDPBackend.CUDNN_ATTENTION), ('SDPA efficient', SDPBackend.EFFICIENT_ATTENTION)]
    for name, b, hq, hk, l, d, lengths in bench_shapes(quick):
        report.line(f'  {name} (batch {b}, heads {hq}/{hk}, head_dim {d}):')
        q, k, v, grad = gen((b, hq, l, d), torch.bfloat16, 1), gen((b, hk, l, d), torch.bfloat16, 2), \
            gen((b, hk, l, d), torch.bfloat16, 3), gen((b, hq, l, d), torch.bfloat16, 4)
        kwargs = {'enable_gqa': True} if hq != hk else {}
        if lengths:
            kwargs['attn_mask'] = key_padding(lengths, l)[:, None, None, :]
            useful = sum(n * n for n in lengths) / b
        else:
            useful = l * l
        flops = 3.5 * 4 * b * hq * useful * d
        rows = []
        for method, kernel in methods:
            rows.append((method, lambda *a, kernel=kernel, **kw: _sdpa_with(kernel, *a, **kw)))
        for backend in backends:
            rows.append((attention.LABELS[backend], backend))
        results = {}
        for method, fn in rows:
            try:
                if isinstance(fn, str):
                    if use_backend(fn):
                        continue
                    attention.install()
                    fn = F.scaled_dot_product_attention
                ms, first, peak = time_fwd_bwd(fn, q, k, v, grad, kwargs, iterations)
                attention.uninstall()
                results[method] = ms
                extra = f', first call {first:.1f} s' if first > 1 else ''
                report.line(f'    {method:22s} {ms:8.2f} ms  {flops / ms / 1e9:6.0f} TFLOP/s  peak +{peak:7.0f} MiB'
                            f'{extra}')
                report.record({'shape': name, 'method': method, 'ms': ms, 'tflops': flops / ms / 1e9,
                               'peak_mib': peak, 'first_call_s': first})
            except Exception as error:
                attention.uninstall()
                report.line(f'    {method:22s} failed: {type(error).__name__}: {error}'.splitlines()[0][:200])
                report.record({'shape': name, 'method': method, 'error': str(error)})
            torch.cuda.empty_cache()
        baseline = results.get('SDPA, PyTorch picks')
        if baseline:
            speedups = ', '.join(f'{m} {baseline / t:.2f}x' for m, t in results.items() if m.startswith('Flash'))
            if speedups:
                report.line(f'    speed-up over SDPA (PyTorch picks): {speedups}')
        del q, k, v, grad
        torch.cuda.empty_cache()
    use_backend('sdpa')


def _sdpa_with(kernel, q, k, v, **kwargs):
    if kernel is None:
        return REAL_SDPA(q, k, v, **kwargs)
    with sdpa_kernel(kernel):
        return REAL_SDPA(q, k, v, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default='.', help='Where the report goes (default: the current directory).')
    parser.add_argument('--quick', action='store_true', help='Skip the video-length benchmarks.')
    parser.add_argument('--iterations', type=int, default=10, help='Timed forward+backward runs per kernel.')
    parser.add_argument('--skip-benchmarks', action='store_true')
    parser.add_argument('--skip-models', action='store_true')
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    report = Report(args.out_dir)
    report.watch_warnings()
    report.line(f'FlashAttention check, {datetime.datetime.now().isoformat(timespec="seconds")}; report: '
                f'{report.txt_path}')
    try:
        environment(report)
        if not torch.cuda.is_available():
            report.line('No CUDA GPU: nothing else to check.')
            return 1
        torch.cuda.set_device(0)
        backends = []
        report.section('Backends')
        for backend in FLASH:
            reason = use_backend(backend)
            report.line(f'  {attention.LABELS[backend]}: {"usable" if reason is None else "not usable: " + reason}')
            report.record({'backend': backend, 'usable': reason is None, 'reason': reason})
            if reason is None:
                backends.append(backend)
        use_backend('sdpa')
        if not backends:
            report.line('  Neither FlashAttention-3 nor -4 can run here; only the environment and the SDPA '
                        'benchmarks are reported.')
        baselines = {}
        for backend in backends:
            label = attention.LABELS[backend]
            for title, check in ((f'2. SDPA replacement with {label}', check_replacement),
                                 (f'3. Helpers with {label}', check_helpers),
                                 (f'4. Training with {label}', lambda r, b: check_training(r, b, baselines)),
                                 (f'5. Determinism with {label}', check_determinism)):
                report.section(title)
                use_backend(backend)
                try:
                    check(report, backend)
                except Exception:
                    report.line('  section crashed:\n' + traceback.format_exc())
                    report.failures.append({'section': title})
        if not args.skip_models:
            report.section('6. Models that call flash-attn directly, and Iris')
            try:
                check_models(report, backends)
            except Exception:
                report.line('  section crashed:\n' + traceback.format_exc())
        if not args.skip_benchmarks:
            report.section('7. Forward + backward speed and memory (bf16, median)')
            try:
                benchmarks(report, backends, args.quick, args.iterations)
            except Exception:
                report.line('  section crashed:\n' + traceback.format_exc())
        report.section('Summary')
        failed = len(report.failures)
        report.line(f'  {failed} failed check(s)' + (': ' + '; '.join(
            str(f.get('case') or f.get('check') or f.get('section')) + f' ({f.get("backend", "")})'
            for f in report.failures) if failed else ''))
        report.results['failures'] = report.failures
        if report.warnings:
            report.line(f'  {len(report.warnings)} distinct warning(s) were raised; the checks above say whether they '
                        'matter (a result is a PASS only when output and gradients match the reference):')
            for text in report.warnings:
                report.line(f'    {text}')
            report.results['warnings'] = report.warnings
        report.line(f'  Send back: {report.txt_path}')
        return 1 if failed else 0
    finally:
        attention.uninstall()
        report.close()


if __name__ == '__main__':
    sys.exit(main())
