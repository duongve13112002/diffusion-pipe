"""A stand-in for FlashAttention-3's flash_attn_interface, for CPU tests.

Same arguments and semantics as the real one (fp16 / bf16 only, GQA, causal aligned bottom-right,
sliding windows, varlen through cu_seqlens, zero output for a row with nothing to attend to),
computed with plain torch math in float32. CALLS records each call.
"""

import torch

CALLS = []
# Older FA3 builds return (out, softmax_lse); the current one returns out alone.
RETURN_TUPLE = False


def reference(q, k, v, scale=None, causal=False, window_size=(-1, -1)):
    """One sequence: q (Lq, Hq, D), k / v (Lk, Hk, D) -> (Lq, Hq, D) in q's dtype."""
    lq, hq, d = q.shape
    lk, hk = k.shape[:2]
    assert hq % hk == 0, (hq, hk)
    k = k.repeat_interleave(hq // hk, 1)
    v = v.repeat_interleave(hq // hk, 1)
    scale = d ** -0.5 if scale is None else scale
    s = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * scale
    i = torch.arange(lq)[:, None] + (lk - lq)
    j = torch.arange(lk)[None, :]
    allowed = torch.ones(lq, lk, dtype=torch.bool)
    if causal:
        allowed &= j <= i
    left, right = window_size
    if left >= 0:
        allowed &= j >= i - left
    if right >= 0:
        allowed &= j <= i + right
    s = s.masked_fill(~allowed, float('-inf'))
    p = torch.softmax(s, -1).nan_to_num(0.0)
    return torch.einsum('hqk,khd->qhd', p, v.float()).to(q.dtype)


def check(q, k, v):
    assert q.dtype in (torch.float16, torch.bfloat16), q.dtype
    assert q.dtype == k.dtype == v.dtype, (q.dtype, k.dtype, v.dtype)
    assert q.shape[-1] <= 256 and q.shape[-1] % 8 == 0, q.shape
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]


def _out(out):
    return (out, torch.zeros(1)) if RETURN_TUPLE else out


def flash_attn_func(q, k, v, softmax_scale=None, causal=False, qv=None, q_descale=None, k_descale=None,
                    v_descale=None, window_size=(-1, -1), attention_chunk=0, softcap=0.0, num_splits=1,
                    pack_gqa=None, deterministic=False, sm_margin=0, return_attn_probs=False):
    check(q, k, v)
    assert q.dim() == 4 and q.shape[0] == k.shape[0]
    assert None not in tuple(window_size), f'FA3 writes "no limit" as -1: {window_size}'
    CALLS.append(('flash_attn_func', dict(shape=tuple(q.shape), causal=causal, scale=softmax_scale,
                                          deterministic=deterministic)))
    out = torch.stack([reference(q[b], k[b], v[b], softmax_scale, causal, window_size) for b in range(q.shape[0])])
    return _out(out)


def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size):
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert isinstance(max_seqlen_q, int) and isinstance(max_seqlen_k, int)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    assert len(cq) == len(ck) and cq[0] == ck[0] == 0
    assert cq[-1] == q.shape[0] and ck[-1] == k.shape[0] == v.shape[0], (cq[-1], q.shape, ck[-1], k.shape)
    outs = []
    for a, b, c, d in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
        assert 0 <= b - a <= max_seqlen_q and 0 <= d - c <= max_seqlen_k
        if b == a:
            continue
        if d == c:
            outs.append(q.new_zeros((b - a, q.shape[1], v.shape[2])))
        else:
            outs.append(reference(q[a:b], k[c:d], v[c:d], softmax_scale, causal, window_size))
    return torch.cat(outs)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_q=None,
                           seqused_k=None, softmax_scale=None, causal=False, qv=None, q_descale=None,
                           k_descale=None, v_descale=None, window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                           num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                           return_attn_probs=False):
    check(q, k, v)
    assert q.dim() == 3 and seqused_q is None and seqused_k is None
    assert None not in tuple(window_size), f'FA3 writes "no limit" as -1: {window_size}'
    CALLS.append(('flash_attn_varlen_func', dict(cu_seqlens_q=cu_seqlens_q.tolist(),
                                                 cu_seqlens_k=cu_seqlens_k.tolist(), causal=causal,
                                                 scale=softmax_scale, deterministic=deterministic,
                                                 window_size=tuple(window_size))))
    return _out(varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal,
                       window_size))
