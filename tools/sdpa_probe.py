"""Check PyTorch's attention kernels on the exact shapes of an Iris training step, on one GPU.

    python tools/sdpa_probe.py
    python tools/sdpa_probe.py --batch-size 64 --tokens 1326 1344 1315

Iris attention is grouped-query (20 query heads, 5 key/value heads, head_dim 128) over the text
and image tokens of a sample: text_len 300 + (height/16) * (width/16) patches, so a 608x432
bucket is 300 + 38*27 = 1326 tokens. For every kernel F.scaled_dot_product_attention can use
(flash, efficient, cudnn, math) and the one it picks by itself (default), this runs forward and
backward in bfloat16 with the training layout, and compares the output and the q/k/v gradients
with a float32 reference. A kernel that gives NaN or a large error here is a kernel bug at that
shape, independent of the model and the data: set [model] attn_backend to one that passes.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {
    'flash': SDPBackend.FLASH_ATTENTION,
    'efficient': SDPBackend.EFFICIENT_ATTENTION,
    'cudnn': SDPBackend.CUDNN_ATTENTION,
    'math': SDPBackend.MATH,
}
# How [model] attn_backend selects each kernel (see iris3b.nn.attention.scaled_dot_product).
CONFIG_NAME = {'default': "'sdpa'", 'flash': "'torch_flash'", 'cudnn': "'torch_cudnn'"}


def make_inputs(batch, tokens, heads, kv_heads, head_dim, gain, device, seed):
    """q, k, v as the Iris blocks hand them to attention: RMS-normalized per head (qk_norm),
    [B, heads, N, D] views of [B, N, heads, D] tensors."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rms_normed(n_heads):
        x = torch.randn(batch, tokens, n_heads, head_dim, generator=g, device=device)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * gain

    q = rms_normed(heads).to(torch.bfloat16).transpose(1, 2)
    k = rms_normed(kv_heads).to(torch.bfloat16).transpose(1, 2)
    v = torch.randn(batch, tokens, kv_heads, head_dim, generator=g, device=device).to(torch.bfloat16).transpose(1, 2)
    grad = torch.randn(batch, tokens, heads, head_dim, generator=g, device=device).to(torch.bfloat16).transpose(1, 2)
    return q, k, v, grad


def run(q, k, v, grad, backend):
    q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
    if backend == 'default':
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    else:
        with sdpa_kernel(BACKENDS[backend]):
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    out.backward(grad)
    return out.detach(), q.grad, k.grad, v.grad


def reference(q, k, v, grad, chunk):
    """float32 attention, a few samples at a time so the score matrix fits."""
    results = [[], [], [], []]
    group = q.shape[1] // k.shape[1]
    for start in range(0, q.shape[0], chunk):
        qs, ks, vs = (t[start:start + chunk].float().detach().requires_grad_() for t in (q, k, v))
        kr, vr = ks.repeat_interleave(group, dim=1), vs.repeat_interleave(group, dim=1)
        weights = torch.softmax(qs @ kr.transpose(-1, -2) * qs.shape[-1] ** -0.5, dim=-1)
        out = weights @ vr
        out.backward(grad[start:start + chunk].float())
        for store, value in zip(results, (out.detach(), qs.grad, ks.grad, vs.grad)):
            store.append(value)
        del weights, out
    return [torch.cat(r) for r in results]


def compare(name, value, ref):
    bad = int((~torch.isfinite(value)).sum())
    if bad:
        return f'{name} {bad} non-finite', False
    error = float((value.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    # bfloat16 rounding (math also runs its softmax in bfloat16) stays within a few percent; a broken
    # kernel is off by the size of the values themselves.
    return f'{name} err {error:.1e}', error < 0.2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=64, help='micro_batch_size_per_gpu (default 64).')
    parser.add_argument('--tokens', type=int, nargs='+', default=[1326, 1344, 1315],
                        help='Sequence lengths: 300 text + patches of the bucket (default: the 608x432, '
                             '576x464 and 560x464 buckets).')
    parser.add_argument('--heads', type=int, default=20)
    parser.add_argument('--kv-heads', type=int, default=5)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--gain', type=float, nargs='+', default=[1.0, 4.0],
                        help='Scale of the normalized q and k (the q_norm/k_norm weights); larger means sharper attention.')
    parser.add_argument('--backends', nargs='+', default=['default', *BACKENDS],
                        choices=['default', *BACKENDS])
    parser.add_argument('--reference-chunk', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, '
              f'{torch.cuda.get_device_name(device)}')
    else:
        print(f'torch {torch.__version__} on {device} (only math runs off CUDA)')
    failures, passed = [], set()
    for tokens in args.tokens:
        for gain in args.gain:
            q, k, v, grad = make_inputs(args.batch_size, tokens, args.heads, args.kv_heads, args.head_dim,
                                        gain, device, args.seed)
            ref = reference(q, k, v, grad, args.reference_chunk)
            print(f'\nbatch {args.batch_size}, {tokens} tokens, heads {args.heads}/{args.kv_heads}, '
                  f'head_dim {args.head_dim}, q/k gain {gain}:')
            outputs = {}
            for backend in args.backends:
                try:
                    outputs[backend] = run(q, k, v, grad, backend)
                except RuntimeError as error:
                    message = str(error).strip().splitlines()[0][:100]
                    print(f'  {backend:9s} unavailable: {message}')
                    continue
                verdicts = [compare(name, value, r) for name, value, r in
                            zip(('out', 'dq', 'dk', 'dv'), outputs[backend], ref)]
                ok = all(passed for _, passed in verdicts)
                print(f'  {backend:9s} {"ok  " if ok else "FAIL"}  ' + ', '.join(text for text, _ in verdicts))
                if ok:
                    passed.add(backend)
                else:
                    failures.append((tokens, gain, backend))
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if 'default' in outputs:
                picked = [b for b in BACKENDS if b in outputs
                          and all(torch.equal(a, b_) for a, b_ in zip(outputs['default'], outputs[b]))]
                print(f'  default uses: {", ".join(picked) or "none of the above bit for bit"}')
            del q, k, v, grad, ref, outputs
    print()
    if not failures:
        print('=> Every kernel is correct at these shapes: the NaN is not an attention kernel bug.')
        return 0
    for tokens, gain, backend in failures:
        print(f'=> {backend} FAILS at {tokens} tokens (gain {gain})')
    working = [b for b in CONFIG_NAME if b in passed and b not in {f[2] for f in failures}]
    if working:
        print('=> Set [model] attn_backend = ' + ' or '.join(CONFIG_NAME[b] for b in working)
              + ' (the kernels that passed).')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
