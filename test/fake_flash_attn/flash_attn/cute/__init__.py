"""A stand-in for FlashAttention-4 (flash_attn.cute), for CPU tests: its argument order (qv fourth,
cu_seqlens by keyword), "no window limit" written as None and an (out, lse) return, with the math of
the fake FlashAttention-3. CALLS records each call."""

import torch

import flash_attn_interface as _fa3

CALLS = []


def _window(window_size):
    assert all(w is None or w >= 0 for w in window_size), f'FA4 writes "no limit" as None: {window_size}'
    return tuple(-1 if w is None else w for w in window_size)


def flash_attn_func(q, k, v, qv=None, gather_kv_indices=None, softmax_scale=None, causal=False,
                    window_size=(None, None), learnable_sink=None, softcap=0.0, num_splits=1, pack_gqa=None,
                    deterministic=False, score_mod=None, score_mod_bwd=None, mask_mod=None, aux_tensors=None,
                    aux_scalars=None, block_sparse_tensors=None, block_sparse_tensors_bwd=None, return_lse=False,
                    gather_bwd_recompute_p=False, gather_bwd_token_chunk=None):
    assert qv is None
    _fa3.check(q, k, v)
    window = _window(window_size)
    CALLS.append(('flash_attn_func', dict(shape=tuple(q.shape), causal=causal, scale=softmax_scale,
                                          deterministic=deterministic)))
    out = torch.stack([_fa3.reference(q[b], k[b], v[b], softmax_scale, causal, window) for b in range(q.shape[0])])
    return out, None


def flash_attn_varlen_func(q, k, v, qv=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None,
                           max_seqlen_k=None, min_seqlen_k=None, seqused_q=None, seqused_k=None,
                           gather_kv_indices=None, page_table=None, softmax_scale=None, causal=False,
                           window_size=(None, None), learnable_sink=None, softcap=0.0, num_splits=1, pack_gqa=None,
                           deterministic=False, score_mod=None, score_mod_bwd=None, mask_mod=None,
                           block_sparse_tensors=None, aux_tensors=None, aux_scalars=None, return_lse=False,
                           scheduler_metadata=None, seqlen_k_per_split=None, disable_scheduler_metadata=False,
                           gather_bwd_recompute_p=False, gather_bwd_token_chunk=None):
    assert qv is None and cu_seqlens_q is not None and cu_seqlens_k is not None
    _fa3.check(q, k, v)
    window = _window(window_size)
    CALLS.append(('flash_attn_varlen_func', dict(cu_seqlens_q=cu_seqlens_q.tolist(),
                                                 cu_seqlens_k=cu_seqlens_k.tolist(), causal=causal,
                                                 scale=softmax_scale, deterministic=deterministic,
                                                 window_size=window)))
    out = _fa3.varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window)
    return out, None
