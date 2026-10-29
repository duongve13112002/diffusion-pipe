"""A stand-in for flash-attn 2, for CPU tests: the functions the models call, computed with the
math of the fake FlashAttention-3 next to it. CALLS records each call."""

import flash_attn_interface as _fa3

CALLS = []


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
                    alibi_slopes=None, deterministic=False, return_attn_probs=False):
    assert not dropout_p
    _fa3.check(q, k, v)
    CALLS.append(('flash_attn_func', dict(shape=tuple(q.shape), causal=causal)))
    import torch
    return torch.stack([_fa3.reference(q[b], k[b], v[b], softmax_scale, causal, window_size)
                        for b in range(q.shape[0])])


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                           softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                           deterministic=False, return_attn_probs=False, block_table=None):
    assert not dropout_p
    _fa3.check(q, k, v)
    CALLS.append(('flash_attn_varlen_func', dict(cu_seqlens_q=cu_seqlens_q.tolist(), causal=causal)))
    return _fa3.varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k), softmax_scale,
                       causal, window_size)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False,
                                     window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False,
                                     return_attn_probs=False):
    assert not dropout_p
    q, k, v = qkv.unbind(1)
    _fa3.check(q, k, v)
    CALLS.append(('flash_attn_varlen_qkvpacked_func', dict(cu_seqlens=cu_seqlens.tolist(), causal=causal)))
    return _fa3.varlen(q, k, v, cu_seqlens, cu_seqlens, int(max_seqlen), int(max_seqlen), softmax_scale, causal,
                       window_size)
