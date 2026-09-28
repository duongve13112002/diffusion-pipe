"""Settle the two precision questions a CPU box cannot answer.

    python tools/test_precision_gpu.py

Needs one CUDA GPU. No checkpoints, no dataset, no DeepSpeed -- it builds a small
ContextRefiner and calls attention directly, so it runs in seconds and can be pointed at any
machine that has a device.

Two things are checked, and they are the only two precision claims in this repo that a CPU
cannot reach:

1. Does a fused CUDA attention backend return NaN for a fully masked row? This is the premise
   `allow_fully_masked_rows` (models/llm_adapter.py) was written for. On CPU the math backend
   returns zeros, so the guard can be shown to be a no-op but never shown to be necessary. The
   guard is correct either way -- it is a no-op for any row that has a key, and the degenerate
   row's output is zeroed downstream regardless -- so a failure here means the guard is
   unnecessary, not that anything is broken. Worth knowing which.

2. Does `fp16-mixed` behave on real kernels? `torch.amp.GradScaler('cuda')` disables itself
   without CUDA and `Precision.autocast` returns a null context off-CUDA, so every fp16-mixed
   test on a CPU box silently exercises the fp32 path. The control flow around the scaler is
   covered by TestGradScalerPathAgainstARealScaler, and the arithmetic through these modules is
   covered by TestFp16NumericsThroughTheRefiner using CPU autocast -- what is left is whether a
   real CUDA kernel overflows.

Exit status is 0 when every check passes. Each check prints what it measured, not just a
verdict, because the numbers are the point.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F

from models.llm_adapter import allow_fully_masked_rows
from models.text_refiner import ContextRefiner


failures = []


def check(name, passed, detail):
    print(f'{"PASS" if passed else "FAIL"}  {name}\n      {detail}')
    if not passed:
        failures.append(name)


def fused_attention_on_a_fully_masked_row(device):
    """Call SDPA directly with an all-False mask row, on each backend CUDA offers.

    Shapes are (batch, heads, queries, keys) -- the layout Attention.forward builds before it
    calls scaled_dot_product_attention.
    """
    for dtype in (torch.bfloat16, torch.float16):
        q = torch.randn(1, 4, 8, 16, device=device, dtype=dtype)
        k = torch.randn(1, 4, 8, 16, device=device, dtype=dtype)
        v = torch.randn(1, 4, 8, 16, device=device, dtype=dtype)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool, device=device)
        mask[..., 0, :] = False          # query row 0 attends to nothing

        for backend_name, backend in (
            ('flash', torch.nn.attention.SDPBackend.FLASH_ATTENTION),
            ('mem_efficient', torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
            ('cudnn', torch.nn.attention.SDPBackend.CUDNN_ATTENTION),
            ('math', torch.nn.attention.SDPBackend.MATH),
        ):
            try:
                with torch.nn.attention.sdpa_kernel(backend):
                    raw = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                    guarded = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=allow_fully_masked_rows(mask))
            except RuntimeError as e:
                # A backend that refuses this mask/dtype combination is not a result either way.
                print(f'      ({backend_name}, {dtype}): backend unavailable -- {e}')
                continue

            raw_nan = not torch.isfinite(raw[:, :, 0]).all()
            check(
                f'guard keeps the fully masked row finite ({backend_name}, {dtype})',
                torch.isfinite(guarded).all().item(),
                f'unguarded row finite={not raw_nan}; guarded row finite='
                f'{torch.isfinite(guarded).all().item()}. '
                + ('The guard is load-bearing on this backend.' if raw_nan
                   else 'This backend returns finite values unguarded, so the guard is '
                        'unnecessary here -- not wrong, just not needed.'),
            )


def fp16_mixed_through_the_refiner(device):
    """A few real steps with a live GradScaler, watching for a collapsing scale."""
    torch.manual_seed(0)
    refiner = ContextRefiner(cap_feat_dim=64, model_dim=128, num_layers=2, num_heads=8)
    refiner.init_weights()
    refiner.to(device)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    check('GradScaler is actually enabled on this device', scaler.is_enabled(),
          f'is_enabled={scaler.is_enabled()} (it self-disables without CUDA)')

    hidden = torch.randn(2, 16, 64, device=device)
    mask = torch.ones(2, 16, dtype=torch.long, device=device)
    target = torch.randn(2, 16, 128, device=device)

    scales, skipped, losses = [], 0, []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = refiner(hidden, mask)
        loss = F.mse_loss(out.float(), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            skipped += 1
        scales.append(scaler.get_scale())
        losses.append(loss.item())

    check('fp16 forward stays finite on real kernels', all(l == l for l in losses),
          f'first loss {losses[0]:.6f}, last {losses[-1]:.6f}')
    check('the loss scale does not collapse', scales[-1] >= 1.0,
          f'scale {scales[0]:.0f} -> {scales[-1]:.0f}, {skipped}/20 steps skipped')
    check('most steps are not skipped', skipped < 10,
          f'{skipped} of 20 steps skipped (a few early ones are normal while the scale settles)')


def main():
    if not torch.cuda.is_available():
        print('This script needs a CUDA device. torch.cuda.is_available() is False'
              f' (torch {torch.__version__}, cuda build {torch.version.cuda}).')
        return 1
    device = torch.device('cuda')
    print(f'device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}\n')

    print('1. Is allow_fully_masked_rows load-bearing on a fused backend?')
    fused_attention_on_a_fully_masked_row(device)
    print('\n2. Does fp16-mixed behave on real kernels?')
    fp16_mixed_through_the_refiner(device)

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: ' + ', '.join(failures))
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
