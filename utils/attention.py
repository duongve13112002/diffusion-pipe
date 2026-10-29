"""Optional FlashAttention-3 / -4 for training: `[model] attention = 'flash3'` or `'flash4'`.

With either, PyTorch's scaled_dot_product_attention is replaced, after caching, by a function that
runs the chosen FlashAttention for every call it can take and PyTorch's own SDPA for the rest.
FlashAttention takes fp16 / bf16 inputs on the GPU, head_dim a multiple of 8 within the kernel's
range (see head_dim_supported), no dropout, and either no mask or a key-padding mask (one that
only says which keys of each batch item count); is_causal only when queries and keys have the same
length, since FlashAttention aligns the causal mask bottom-right and SDPA top-left. Every other
call (float32, an arbitrary mask or bias, dropout, ...) runs SDPA and is counted: the first of
each kind is printed when it happens, the totals after the first step and at the end.

flash3 is the `flash_attn_3` package (flash-attention's hopper/ directory): Hopper, plus Ampere /
Ada when built with those kernels. flash4 is the `flash-attn-4` package (flash_attn.cute, CuTe
DSL): Hopper and Blackwell, compiling its kernels the first time each configuration runs.

Models that call flash-attn directly (Wan, HiDream, HunyuanVideo, HunyuanImage, OmniGen2, Lumina 2) use
flash_dense / flash_varlen / flash_unpadded below when flash_enabled(); Iris maps the key onto its
own attn_backend. 'sdpa' (the default) changes nothing for SDPA models.
"""

import collections
import contextlib
import sys
import weakref

import torch
import torch.nn.functional as F

CHOICES = ('sdpa', 'flash3', 'flash4')
LABELS = {'flash3': 'FlashAttention-3', 'flash4': 'FlashAttention-4'}
PACKAGES = {
    'flash3': "the flash_attn_3 package (flash_attn_interface), built from flash-attention's hopper/ directory",
    'flash4': 'the flash-attn-4 package (flash_attn.cute): pip install flash-attn-4',
}
# A float mask entry at or below this counts as "masked out": exp() of it is 0 next to any real
# attention score, so dropping the key gives the same result.
MASKED_BELOW = -1e4

_backend = 'sdpa'
# The GPU's compute capability major, which decides FA4's head_dim range.
_arch = 9
_module = None
_original_sdpa = None
_counts = collections.Counter()
_reported_reasons = set()
_mask_cache = {}
# Inside bypass(): the replacement hands every call to PyTorch SDPA, uncounted.
_bypass = 0

FLASH_CALL = 'flash'
FLASH_PADDED = 'flash with key padding'


def attention_setting(config):
    """The checked `[model] attention` value; refuses combinations FlashAttention cannot run with."""
    model = config['model']
    value = model.get('attention', 'sdpa')
    if not isinstance(value, str) or value not in CHOICES:
        raise ValueError(f"[model] attention must be one of {', '.join(repr(c) for c in CHOICES)}, got {value!r}")
    if value != 'sdpa':
        if config.get('compile', False):
            raise ValueError(f"[model] attention = {value!r} cannot be combined with compile = true: torch.compile "
                             "would trace through the attention replacement and its mask checks; turn one off")
        if model['type'] == 'cosmos':
            raise ValueError(f"[model] attention = {value!r} does not apply to cosmos: Cosmos runs its attention "
                             "through Transformer Engine, not PyTorch SDPA or flash-attn")
    return value


def import_module(backend):
    """The FlashAttention module for a backend; ImportError when it is not installed."""
    if backend == 'flash3':
        try:
            import flash_attn_interface as module
        except ImportError:
            from flash_attn_3 import flash_attn_interface as module
        return module
    if backend == 'flash4':
        from flash_attn import cute as module
        if not hasattr(module, 'flash_attn_varlen_func'):
            raise ImportError('flash_attn.cute has no flash_attn_varlen_func')
        return module
    raise ValueError(f'no FlashAttention module for {backend!r}')


def installed(backend):
    try:
        import_module(backend)
    except ImportError:
        return False
    return True


def flash_module():
    """The active backend's module."""
    global _module
    if _module is None:
        _module = import_module(_backend)
    return _module


def label():
    return LABELS.get(_backend, 'PyTorch SDPA')


def configure(config):
    """Sets the backend for this process; checks the chosen FlashAttention imports and the GPU can
    train with it."""
    global _backend, _arch, _module
    value = attention_setting(config)
    if value != 'sdpa':
        name = LABELS[value]
        try:
            module = import_module(value)
        except ImportError as error:
            raise RuntimeError(f"[model] attention = {value!r} needs {name}: {PACKAGES[value]}, which did not "
                               f"import: {error}") from error
        if not torch.cuda.is_available():
            raise RuntimeError(f"[model] attention = {value!r} needs a CUDA GPU")
        major, minor = torch.cuda.get_device_capability()
        gpu = torch.cuda.get_device_name()
        if value == 'flash3' and major not in (8, 9):
            raise RuntimeError(f"[model] attention = 'flash3': {gpu} (compute capability {major}.{minor}) has no "
                               "FlashAttention-3 kernels; FA3 runs on Hopper (9.0) and, when built with them, on "
                               "Ampere / Ada (8.x). Use attention = 'sdpa' (or 'flash4' on Blackwell)")
        if value == 'flash4' and major not in (9, 10, 11, 12):
            raise RuntimeError(f"[model] attention = 'flash4': {gpu} (compute capability {major}.{minor}) cannot "
                               "train with FlashAttention-4, whose backward runs on Hopper (9.x) and Blackwell "
                               "(10.x - 12.x). Use attention = 'sdpa'")
        if value == 'flash3' and major == 8 and _is_main():
            print(f"[attention] {gpu} is not a Hopper GPU: FlashAttention-3 runs its Ampere kernels here (if "
                  "the build has them), which are no faster than flash-attn 2")
        _arch, _module = major, module
    _backend = value
    return value


def flash_enabled():
    """Whether FlashAttention runs: from install(), after caching, on. Before it (caching, and the
    --cache_only / --cache_status / --metadata_only runs) the models' own attention runs as with 'sdpa'."""
    return _original_sdpa is not None


def note_if_unused(model_name, used):
    """For models that picked FA3 by themselves when it was installed: say it no longer happens."""
    if _backend == 'sdpa' and installed('flash3') and _is_main():
        print(f"[attention] FlashAttention-3 is installed, but [model] attention = 'sdpa': {model_name} runs "
              f"{used}. Set attention = 'flash3' to train with FlashAttention-3")


def install():
    """Routes torch.nn.functional.scaled_dot_product_attention through FlashAttention."""
    global _original_sdpa
    if _backend == 'sdpa' or _original_sdpa is not None:
        return
    flash_module()
    _original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = scaled_dot_product_attention
    if _is_main():
        line = f'[attention] {label()} replaces PyTorch SDPA from here on; calls it cannot take run SDPA and are counted'
        if _backend == 'flash4':
            line += ' (FA4 compiles its kernels the first time each configuration runs: the first steps are slow)'
        print(line)
        _note_comfy_attention()


def _note_comfy_attention():
    # ComfyUI picks its attention when it is imported; with xformers, SageAttention or flash-attn
    # chosen there, the ComfyUI-based models never call PyTorch SDPA and the replacement misses them.
    comfy_attention = sys.modules.get('comfy.ldm.modules.attention')
    name = getattr(getattr(comfy_attention, 'optimized_attention', None), '__name__', '')
    if name in ('attention_xformers', 'attention_sage', 'attention_flash'):
        print(f'[attention] WARNING: ComfyUI runs its models\' attention with {name}, not PyTorch SDPA, so '
              f'{label()} does not reach the models built on ComfyUI. Uninstall '
              'xformers / sageattention, or train them with attention = \'sdpa\'')


def uninstall():
    global _original_sdpa
    if _original_sdpa is not None:
        F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
    _mask_cache.clear()


@contextlib.contextmanager
def bypass():
    """PyTorch SDPA for every call made inside, as if nothing were replaced (for --nan_check's
    replays with each SDPA kernel). The models' direct FlashAttention calls are not affected."""
    global _bypass
    _bypass += 1
    try:
        yield
    finally:
        _bypass -= 1


def counts():
    return dict(_counts)


def reset_counts():
    _counts.clear()
    _reported_reasons.clear()


def _is_main():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def report():
    """Prints how many attention calls ran FlashAttention and SDPA (and why), all ranks together."""
    if _original_sdpa is None:
        return
    totals = collections.Counter(_counts)
    if torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, dict(_counts))
        totals = collections.Counter()
        for c in gathered:
            totals.update(c)
    if not _is_main():
        return
    padded = totals.pop(FLASH_PADDED, 0)
    flash = totals.pop(FLASH_CALL, 0) + padded
    line = f'[attention] so far, all ranks: {label()} ran {flash} calls'
    if padded:
        line += f' ({padded} with key padding)'
    sdpa = sum(totals.values())
    if sdpa:
        reasons = ', '.join(f'{n} {reason}' for reason, n in totals.most_common())
        line += f'; PyTorch SDPA ran {sdpa}: {reasons}'
    else:
        line += '; PyTorch SDPA ran none'
    print(line)


def device_ok(t):
    return t.is_cuda


def head_dim_supported(d):
    """Whether the active FlashAttention trains with this head_dim (queries, keys and values alike)."""
    if d % 8 or d < 8:
        return False
    if _backend == 'flash4' and _arch in (10, 11):
        return d <= 128 or d == 256
    if _backend == 'flash4' and _arch == 12:
        return d <= 128
    return d <= 256


def _deterministic():
    return torch.are_deterministic_algorithms_enabled()


def _autocast_dtype(device_type):
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return None


def _half(*tensors):
    """The tensors in the dtype SDPA would run them in: autocast's, else their own."""
    dtype = _autocast_dtype(tensors[0].device.type)
    if dtype is None:
        return tensors
    return tuple(t.to(dtype) if t.is_floating_point() else t for t in tensors)


def _unwrap(out):
    # FA4 returns (out, lse); FA3 the output alone, or (out, softmax_lse) in older builds.
    return out[0] if isinstance(out, tuple) else out


def _check_half(q, k, v, where):
    for t in (q, k, v):
        if t.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f'{where}: {label()} takes float16 / bfloat16 inputs, got {t.dtype}')


def _window(window_size):
    # FA3 writes "no limit" as -1, FA4 as None.
    if _backend == 'flash4':
        return tuple(None if w is None or w < 0 else w for w in window_size)
    return tuple(-1 if w is None else w for w in window_size)


def _dense(q, k, v, softmax_scale, causal, window_size, deterministic):
    return _unwrap(flash_module().flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal,
                                                  window_size=_window(window_size), deterministic=deterministic))


def _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size,
            deterministic):
    return _unwrap(flash_module().flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k), softmax_scale=softmax_scale, causal=causal,
        window_size=_window(window_size), deterministic=deterministic))


def _check_deterministic(deterministic, where):
    """The deterministic flag to pass: the global setting unless given; refused where FA4 has no
    deterministic backward (the SDPA replacement runs SDPA there instead)."""
    if deterministic is None:
        deterministic = _deterministic()
    if deterministic and _backend == 'flash4' and _arch == 12:
        raise RuntimeError(f'{where}: FlashAttention-4 has no deterministic backward on this GPU (compute capability '
                           "12.x); turn deterministic algorithms off, or use [model] attention = 'sdpa'")
    return deterministic


def flash_dense(q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=None):
    """FlashAttention on (B, L, H, D) tensors; returns (B, Lq, H, D)."""
    q, k, v = _half(q, k, v)
    _check_half(q, k, v, 'flash_dense')
    deterministic = _check_deterministic(deterministic, 'flash_dense')
    return _dense(q, k, v, softmax_scale, causal, window_size, deterministic)


def flash_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                 softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=None):
    """FlashAttention on packed (total, H, D) tensors, with flash-attn 2's flash_attn_varlen_func arguments."""
    if dropout_p:
        raise ValueError(f'{label()} has no attention dropout (dropout_p={dropout_p})')
    q, k, v = _half(q, k, v)
    _check_half(q, k, v, 'flash_varlen')
    deterministic = _check_deterministic(deterministic, 'flash_varlen')
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal,
                   window_size, deterministic)


def flash_unpadded(q, k, v, keep, softmax_scale=None, causal=False, deterministic=None):
    """Self-attention on (B, S, H, D) where `keep` (B, S) marks the real tokens of both queries and
    keys, as flash-attn 2's unpad_input / pad_input do it: padded output rows are zero."""
    b, s = keep.shape
    keep = keep.bool()
    index = keep.flatten().nonzero().flatten()
    lengths = keep.sum(1, dtype=torch.int32)
    cu = F.pad(torch.cumsum(lengths, 0, dtype=torch.int32), (1, 0))
    longest = int(lengths.max())
    out = flash_varlen(q.flatten(0, 1)[index], k.flatten(0, 1)[index], v.flatten(0, 1)[index], cu, cu, longest,
                       longest, softmax_scale=softmax_scale, causal=causal, deterministic=deterministic)
    full = out.new_zeros((b * s, *out.shape[1:]))
    full = full.index_copy(0, index, out)
    return full.view(b, s, *out.shape[1:])


class _KeyPadding:
    """What a mask means to FlashAttention: no mask at all, or the keys each batch item keeps."""

    def __init__(self, batch=None, key=None, cu_seqlens_k=None, max_seqlen_k=None):
        self.batch, self.key = batch, key
        self.cu_seqlens_k, self.max_seqlen_k = cu_seqlens_k, max_seqlen_k

    @property
    def dense(self):
        return self.batch is None


def _analyse_mask(mask, b, h, lq, lk):
    """A _KeyPadding for a mask FlashAttention can run, else the reason it cannot (a str)."""
    if mask.requires_grad:
        return 'a mask that needs a gradient'
    if mask.dim() > 4:
        return 'a mask with more than 4 dimensions'
    m = mask.reshape((1,) * (4 - mask.dim()) + tuple(mask.shape))
    for size, full in zip(m.shape, (b, h, lq, lk)):
        if size not in (1, full):
            return 'a mask that does not broadcast to the attention'
    if m.dtype == torch.bool:
        keep = m
    elif m.is_floating_point():
        keep = m == 0
        if not bool((keep | (m <= MASKED_BELOW)).all()):
            return 'a float mask with bias values'
    else:
        return f'a {m.dtype} mask'
    row = keep[:, :1, :1, :]
    if (keep.shape[1] > 1 or keep.shape[2] > 1) and not bool((keep == row).all()):
        return 'a mask that is not key padding'
    keep = row.reshape(row.shape[0], row.shape[3]).expand(b, lk)
    lengths = keep.sum(1, dtype=torch.int32)
    shortest, longest, every = torch.stack([lengths.min(), lengths.max(), keep.all().int()]).tolist()
    if shortest == 0:
        return 'a mask that hides every key of a batch item'
    if every:
        return _KeyPadding()
    index = keep.flatten().nonzero().flatten()
    cu = F.pad(torch.cumsum(lengths, 0, dtype=torch.int32), (1, 0))
    return _KeyPadding(index // lk, index % lk, cu, longest)


def _mask_info(mask, b, h, lq, lk):
    """_analyse_mask, remembered for the life of the mask so each layer of a forward (and its
    recomputation in backward) checks a shared mask once."""
    root = mask._base if mask._base is not None else mask
    if torch.is_inference_mode_enabled():
        # What it computes would be inference tensors, which a later forward with autograd cannot
        # save for backward.
        return _analyse_mask(mask, b, h, lq, lk)
    try:
        version = root._version
    except RuntimeError:
        # Inference tensors keep no version counter, so nothing safe to key a cache entry on.
        return _analyse_mask(mask, b, h, lq, lk)
    # Only writes autograd sees bump _version; the storage address also catches `mask.data = other`.
    # Writes behind autograd's back into the same storage (mask.data[...] = ..., a collective into a
    # persistent buffer) are not seen: build masks per forward, as the models here do.
    key = (id(root), mask.untyped_storage().data_ptr(), mask.storage_offset(), tuple(mask.shape), mask.stride(),
           mask.dtype, b, h, lq, lk)
    entry = _mask_cache.get(key)
    if entry is not None and entry[0]() is root and entry[1] == version:
        return entry[2]
    info = _analyse_mask(mask, b, h, lq, lk)
    root_id = id(root)

    def forget(_, root_id=root_id):
        for k in [k for k in _mask_cache if k[0] == root_id]:
            _mask_cache.pop(k, None)

    _mask_cache[key] = (weakref.ref(root, forget), version, info)
    return info


def _note(reason, q):
    _counts[reason] += 1
    if reason not in _reported_reasons:
        _reported_reasons.add(reason)
        if _is_main():
            print(f'[attention] PyTorch SDPA runs a call {label()} cannot take: {reason} '
                  f'(q {tuple(q.shape)}, {q.dtype}); further ones are counted')


def _flash_reason(q, k, v, dropout_p, is_causal, enable_gqa, extra):
    """Why FlashAttention cannot run this call, or None."""
    if extra:
        return f'unknown arguments {sorted(extra)}'
    if any(type(t) is not torch.Tensor for t in (q, k, v)):
        return 'tensor subclasses'
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return 'inputs that are not 4-D'
    if not all(device_ok(t) for t in (q, k, v)) or not (q.device == k.device == v.device):
        return 'inputs not on one GPU'
    if dropout_p:
        return 'attention dropout'
    b, hq, lq, d = q.shape
    if k.shape[0] != b or v.shape[0] != b or k.shape[2] != v.shape[2] or k.shape[1] != v.shape[1]:
        return 'shapes that do not match'
    if k.shape[3] != d or v.shape[3] != d:
        return 'different query and value head sizes'
    if not head_dim_supported(d):
        return f'head_dim {d}'
    hk = k.shape[1]
    if hk != hq and not (enable_gqa and hq % hk == 0):
        return 'different numbers of query and key heads'
    if lq == 0 or k.shape[2] == 0:
        return 'empty inputs'
    if is_causal and lq != k.shape[2]:
        return 'is_causal with different query and key lengths'
    if _backend == 'flash4' and _arch == 12 and _deterministic():
        return 'deterministic mode (FA4 has no deterministic backward on this GPU)'
    return None


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False, **extra):
    """torch.nn.functional.scaled_dot_product_attention, run by FlashAttention where it can."""
    if _bypass:
        kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **extra)
        if enable_gqa:
            kwargs['enable_gqa'] = enable_gqa
        return _original_sdpa(query, key, value, **kwargs)
    reason = _flash_reason(query, key, value, dropout_p, is_causal, enable_gqa, extra)
    if reason is None and attn_mask is not None and is_causal:
        reason = 'is_causal with a mask'
    q, k, v = (query, key, value) if reason else _half(query, key, value)
    if reason is None and not (q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)):
        reason = f'{q.dtype} inputs' if q.dtype == k.dtype == v.dtype else 'inputs of different dtypes'
    info = None
    if reason is None and attn_mask is not None:
        b, h, lq, d = q.shape
        info = _mask_info(attn_mask, b, h, lq, k.shape[2])
        if isinstance(info, str):
            reason = info
    if reason:
        _note(reason, query)
        kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **extra)
        if enable_gqa:
            kwargs['enable_gqa'] = enable_gqa
        return _original_sdpa(query, key, value, **kwargs)

    deterministic = _deterministic()
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if info is None or info.dense:
        _counts[FLASH_CALL] += 1
        return _dense(qt, kt, vt, scale, is_causal, (-1, -1), deterministic).transpose(1, 2)

    _counts[FLASH_PADDED] += 1
    b, lq = qt.shape[:2]
    cu_q = torch.arange(0, (b + 1) * lq, lq, dtype=torch.int32, device=q.device)
    out = _varlen(qt.reshape(b * lq, *qt.shape[2:]), kt[info.batch, info.key], vt[info.batch, info.key], cu_q,
                  info.cu_seqlens_k, lq, info.max_seqlen_k, scale, False, (-1, -1), deterministic)
    return out.view(b, lq, *out.shape[1:]).transpose(1, 2)
