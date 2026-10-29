# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None
# flash-attn-4 alone installs flash_attn/cute, which makes `import flash_attn` work without flash-attn 2.
FLASH_ATTN_2_AVAILABLE = hasattr(flash_attn, 'flash_attn_varlen_func')

from utils import attention as attention_util


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    version:        unused: [model] attention picks the kernel. 'flash3' / 'flash4' run
                    FlashAttention-3 / -4 (flash-attn 2 for the calls with dropout, which they lack); 'sdpa' runs
                    flash-attn 2, or PyTorch SDPA with a key-padding mask when it is not installed.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    use_flash = attention_util.flash_enabled() and not dropout_p
    if not use_flash and not FLASH_ATTN_2_AVAILABLE:
        return _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size,
                               dtype)
    assert attention_util.device_ok(q) and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)

    # apply attention
    if use_flash:
        # deterministic=False is Wan's default, not a choice: None follows torch.use_deterministic_algorithms.
        x = attention_util.flash_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k, lq, lk, softmax_scale=softmax_scale, causal=causal,
            window_size=window_size, deterministic=deterministic or None).unflatten(0, (b, lq))
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size, dtype):
    """flash_attention on PyTorch SDPA, for when flash-attn 2 is not installed: k_lens becomes a
    key-padding mask. Wan never passes q_lens or a sliding window."""
    if q_lens is not None or tuple(window_size) != (-1, -1):
        raise NotImplementedError('Wan without flash-attn: q_lens and window_size need flash-attn 2 or '
                                  "[model] attention = 'flash3' / 'flash4'")
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q, k, v = half(q), half(k), half(v)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None:
        lengths = k_lens.to(q.device)
        mask = (torch.arange(k.size(1), device=q.device)[None, :] < lengths[:, None])[:, None, None, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, dropout_p=dropout_p,
        is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # flash_attention picks the kernel, PyTorch SDPA with a key-padding mask included.
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
