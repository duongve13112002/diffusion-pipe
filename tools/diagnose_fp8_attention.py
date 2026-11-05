"""Is fp8 attention wrong because of Transformer Engine / cuDNN, or because of utils/fp8_training?

tools/check_fp8_training.py section 3 found, on H200 (TE 2.20.2, cuDNN 9.20): causal attention in
fp8 within limits, but every call without a mask that ran fp8 off by about 50% (output), and
head_dim 64 or a padding mask refused by TE. This runs TE's DotProductAttention on its own, the
way TE's documentation trains it (DelayedScaling, fp8_dpa, scales updated by TE after each step),
on the same inputs, and then with the scales utils/fp8_training sets before the first fp8 step.
If TE alone is as wrong, the library is; if only the primed run is, our code is.

    python tools/diagnose_fp8_attention.py            # on the GPU machine
    python tools/diagnose_fp8_attention.py --steps 8

Writes ./fp8_attention_diagnosis_<host>_<time>.txt; send that file back.
"""

import argparse
import datetime
import os
import platform
import socket
import sys
import traceback

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16


class Report:
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'w')

    def line(self, text=''):
        print(text, flush=True)
        self.file.write(text + '\n')
        self.file.flush()


def gen(*shape, seed, scale=1.0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    return (torch.randn(*shape, generator=g, device='cuda') * scale).to(BF16)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def reference(q, k, v, grad, causal, keep):
    """fp32 SDPA on (b, s, h, d) inputs; keep is a (b, s_kv) bool key mask or None."""
    q, k, v = (t.detach().float().requires_grad_() for t in (q, k, v))
    mask = None if keep is None else keep[:, None, None, :]
    out = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)), attn_mask=mask,
                                         is_causal=causal).transpose(1, 2)
    out.backward(grad.float())
    with torch.no_grad():
        scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * q.shape[-1] ** -0.5
        if causal:
            scores = scores.masked_fill(torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1), float('-inf'))
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        p = scores.softmax(-1)
    return (out.detach(), q.grad, k.grad, v.grad), p.amax().item(), p.float().median().item()


def recipe_for(te_recipe, history):
    return te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID, amax_history_len=history,
                                    amax_compute_algo='max', reduce_amax=False, fp8_dpa=True, fp8_mha=False)


def scales(dpa, meta_key, indexes):
    state = dpa.fp8_meta.get(meta_key)
    if state is None:
        return 'none'
    s = state.scale.float().tolist()
    a = state.amax_history.float().amax(0).tolist()
    return ', '.join(f'{name} scale {s[i]:.4g} amax {a[i]:.4g}' for name, i in indexes)


def run_case(report, te, te_recipe, name, b, h, s, d, mask_type, causal, keep, steps, prime, history):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (META_DO, META_DP, META_DQKV, META_O,
                                                                       META_QKV, META_S)
    q, k, v = (gen(b, s, h, d, seed=10 + i) for i in range(3))
    grad = gen(b, s, h, d, seed=20, scale=1e-2)
    ref, p_max, p_median = reference(q, k, v, grad, causal, keep)
    report.line(f'  {name}: (b, s, h, d) = {(b, s, h, d)}, softmax max {p_max:.3g}, median {p_median:.3g}, '
                f'|out| max {ref[0].abs().max().item():.3g}')
    recipe = recipe_for(te_recipe, history)
    try:
        dpa = te.DotProductAttention(h, d, attention_dropout=0.0, qkv_format='bshd', attn_mask_type=mask_type)
        attention_mask = None
        if keep is not None:
            padded = ~keep
            attention_mask = (torch.zeros(b, 1, 1, s, dtype=torch.bool, device='cuda'), padded[:, None, None, :])
        if prime:
            # What utils/fp8_training does: scales from a bf16 step's amaxes (S at 1) before the first fp8 call.
            with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
                dpa.init_fp8_metadata(num_gemms=3)
            fwd = dpa.fp8_meta['scaling_fwd']
            fmax = dpa.fp8_meta['fp8_max_fwd']
            qkv = max(t.abs().amax().float() for t in (q, k, v))
            for index, amax in ((META_QKV, qkv), (META_S, torch.tensor(1.0, device='cuda')), (META_O, ref[0].abs().amax())):
                fwd.scale[index] = fmax / amax.to(fwd.scale.device)
        for step in range(steps):
            qs, ks, vs = (t.detach().requires_grad_() for t in (q, k, v))
            with te.autocast(enabled=True, recipe=recipe):
                out = dpa(qs, ks, vs, attention_mask=attention_mask, attn_mask_type=mask_type)
            out = out.view(b, s, h, d)
            out.backward(grad)
            errors = [rel(out, ref[0]), rel(qs.grad, ref[1]), rel(ks.grad, ref[2]), rel(vs.grad, ref[3])]
            report.line(f'    step {step}: out {errors[0]:.4f}, dq {errors[1]:.4f}, dk {errors[2]:.4f}, dv {errors[3]:.4f}; '
                        + scales(dpa, 'scaling_fwd', (('QKV', META_QKV), ('S', META_S), ('O', META_O)))
                        + '; ' + scales(dpa, 'scaling_bwd', (('dO', META_DO), ('dP', META_DP), ('dQKV', META_DQKV))))
    except Exception as error:
        report.line(f'    TE refused or failed: {type(error).__name__}: {str(error).strip()[:2000]}')
        if os.environ.get('NVTE_DEBUG') != '1':
            report.line('    (rerun with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 for TE\'s backend choice)')
        report.line('    ' + traceback.format_exc().strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--history', type=int, default=16)
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    report = Report(f'fp8_attention_diagnosis_{socket.gethostname()}_{stamp}.txt')

    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    report.line(f'host {socket.gethostname()}, python {platform.python_version()}, torch {torch.__version__}, '
                f'CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, '
                f'TE {getattr(transformer_engine, "__version__", "?")}, GPU {torch.cuda.get_device_name()}')
    for key in ('NVTE_FUSED_ATTN', 'NVTE_FLASH_ATTN', 'NVTE_UnfusedDPA_Emulate_FP8', 'NVTE_FP8_DPA_BWD'):
        if key in os.environ:
            report.line(f'  {key}={os.environ[key]}')

    b, h, s = 2, 16, 1024
    keep = torch.ones(b, s, dtype=torch.bool, device='cuda')
    keep[1, 700:] = False
    cases = [
        ('no mask, head_dim 128', 128, 'no_mask', False, None),
        ('causal, head_dim 128', 128, 'causal', True, None),
        ('no mask, head_dim 64', 64, 'no_mask', False, None),
        ('key padding, head_dim 128', 128, 'padding', False, keep),
    ]
    for prime in (False, True):
        report.line()
        report.line('TE alone (scales start at 1, TE updates them after each step)' if not prime else
                    'TE with the scales utils/fp8_training sets before the first fp8 step')
        for name, d, mask_type, causal, mask in cases:
            torch.manual_seed(0)
            run_case(report, te, te_recipe, name, b, h, s, d, mask_type, causal, mask, args.steps, prime, args.history)
    report.line()
    report.line(f'Send back: ./{report.path}')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        sys.exit('needs a CUDA GPU')
    main()
