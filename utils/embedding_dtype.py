"""text_embedding_dtype: the dtype text embeddings are cached in.

'bfloat16', 'float16' and 'float32' cast every floating tensor. 'float8' stores float8_e4m3fn with a
scale per token: e4m3 tops out at 448, and text encoder hidden states have outliers in the hundreds
or thousands, so each token's vector (the last dimension) is divided by its own amax / 448, rounded
up to a power of two, before the cast.

An encoded entry `key` keeps the stored values under `key`, an empty tensor of the original dtype
under `key + DTYPE` and, for float8, the float32 scales under `key + SCALE` (the last dimension
reduced to 1), all with the batch as the first dimension so each cache row takes its slice.
Reading a row (decode_row) puts back the original dtype and drops the companions, so the model
gets what it got before.
"""

import torch

FLOAT8 = torch.float8_e4m3fn
FLOAT8_MAX = torch.finfo(FLOAT8).max
SCALE = '.stored_scale'
DTYPE = '.original_dtype'
TEXT_EMBEDDING_DTYPES = {
    'float8': FLOAT8,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}


def check_text_embedding_dtype(model_config, caches_text_embeddings):
    """The torch dtype text_embedding_dtype in [model] names, or None when it is unset."""
    value = model_config.get('text_embedding_dtype')
    if value is None:
        return None
    if value not in TEXT_EMBEDDING_DTYPES:
        raise ValueError(f"text_embedding_dtype must be one of {', '.join(TEXT_EMBEDDING_DTYPES)}, got {value!r}. "
                         'Remove it to cache text embeddings in the dtype the text encoder returns.')
    if not caches_text_embeddings:
        raise ValueError('text_embedding_dtype only applies to cached text embeddings, and this run encodes '
                         'captions on the fly (cache_text_embeddings = false). Remove text_embedding_dtype.')
    return TEXT_EMBEDDING_DTYPES[value]


def _store(x, dtype):
    """(stored tensor, float8 scale or None)."""
    if dtype == FLOAT8:
        # A power of two, so dividing and multiplying back are exact: only the float8 cast rounds,
        # and values float8 holds exactly (0 and 1 of a mask) come back unchanged.
        amax = x.float().abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(torch.ceil(torch.log2(amax / FLOAT8_MAX)))
        # 1 for a token of zeros, and for one whose scale underflows (amax below ~6e-43) or is not
        # finite: dividing by those gives NaN.
        scale = torch.where((amax > 0) & (scale > 0) & torch.isfinite(scale), scale, torch.ones_like(scale))
        return (x.float() / scale).to(FLOAT8), scale
    stored = x.to(dtype)
    if torch.isfinite(x).all() and not torch.isfinite(stored).all():
        raise ValueError(f'A text embedding does not fit in {dtype} (largest value {x.abs().max().item():g}). '
                         "Use text_embedding_dtype = 'bfloat16' or 'float8', which scale or keep the range.")
    return stored, None


def _encodable(value):
    return torch.is_tensor(value) and value.is_floating_point() and value.ndim >= 1


def encode_results(results, dtype):
    """A text encoder's batch results (batch first) with every floating tensor stored in `dtype`;
    lists of per-row tensors are stored row by row."""
    encoded = {}
    for key, value in results.items():
        if _encodable(value) and value.ndim >= 2:
            stored, scale = _store(value, dtype)
            encoded[key] = stored
            encoded[key + DTYPE] = torch.empty(len(value), 0, dtype=value.dtype)
            if scale is not None:
                encoded[key + SCALE] = scale
        elif isinstance(value, (list, tuple)) and value and all(_encodable(v) for v in value):
            pairs = [_store(v, dtype) for v in value]
            encoded[key] = [stored for stored, _ in pairs]
            encoded[key + DTYPE] = [torch.empty(0, dtype=v.dtype) for v in value]
            if dtype == FLOAT8:
                encoded[key + SCALE] = [scale for _, scale in pairs]
        else:
            encoded[key] = value
    return encoded


def decode_row(row):
    """A cached row with its stored entries back in their original dtype."""
    if not isinstance(row, dict) or not any(isinstance(k, str) and k.endswith(DTYPE) for k in row):
        return row
    decoded = {}
    for key, value in row.items():
        if key.endswith(SCALE) or key.endswith(DTYPE):
            continue
        if key + DTYPE in row:
            if key + SCALE in row:
                value = value.float() * row[key + SCALE]
            value = value.to(row[key + DTYPE].dtype)
        decoded[key] = value
    return decoded
