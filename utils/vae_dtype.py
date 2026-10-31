"""The precision a model's VAE encodes the training images in (`vae_dtype`) and the precision its
latents are cached in (`latent_cache_dtype`), for every model that caches latents.

Neither key set: nothing here runs and the model behaves, and caches, as before. `vae_dtype` alone
caches in `vae_dtype`; `latent_cache_dtype` alone encodes as before and rounds the latents to it
(encoded in float32, stored in float16 or bfloat16 to halve the disk).

A latent that comes out non-finite would be cached and trained on, so the images concerned are
encoded again in a wider dtype, and a batch whose latents overflow the storage dtype is cached in
float32 instead.
"""

import functools

import torch

DTYPES = {
    'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16, 'float32': torch.float32, 'fp32': torch.float32,
    'float16': torch.float16, 'fp16': torch.float16,
}
NAMES = {torch.bfloat16: 'bfloat16', torch.float32: 'float32', torch.float16: 'float16'}
# Images whose latents come out non-finite are encoded again in these, in order.
FALLBACKS = {torch.float16: (torch.bfloat16, torch.float32), torch.bfloat16: (torch.float32,), torch.float32: ()}
KEYS = ('vae_dtype', 'latent_cache_dtype')


def parse_dtype(model_config, key, default=None):
    value = model_config.get(key, default)
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        if value not in NAMES:
            raise ValueError(f'{key} must be one of {", ".join(sorted(DTYPES))}, got {value}')
        return value
    if not isinstance(value, str) or value not in DTYPES:
        raise ValueError(f'{key} must be one of {", ".join(sorted(DTYPES))}, got {value!r}')
    return DTYPES[value]


class LatentDtypes:
    """The two settings of a model.

    encode: the dtype the VAE runs in, None for the model's own (the `dtype` it is loaded in).
    cache: the dtype set by latent_cache_dtype, None when unset.
    store: the dtype the latents are cached in: `cache`, else `encode`, else None (as they come).
    """

    def __init__(self, model_config, default_encode=None):
        self.encode = parse_dtype(model_config, 'vae_dtype', default_encode)
        self.cache = parse_dtype(model_config, 'latent_cache_dtype')

    @property
    def store(self):
        return self.cache or self.encode

    @property
    def active(self):
        return self.encode is not None or self.cache is not None

    def cache_key(self, float32_is_default=False):
        """What a latent cache must record so that one made with other settings is rebuilt.

        float32_is_default: a model whose caches were all float32 before these keys existed (its
        VAE always encoded in float32): float32 then adds nothing, and its old caches stay valid.
        """
        key = ''
        if self.encode is not None and not (float32_is_default and self.encode == torch.float32):
            key += f'|vae_dtype={NAMES[self.encode]}'
        if self.cache is not None and self.cache != self.encode:
            key += f'|latent_cache_dtype={NAMES[self.cache]}'
        return key


def refuse(model_config, model_name, why):
    """Raises for the keys of a model they do not apply to, naming them."""
    present = [key for key in KEYS if key in model_config]
    if present:
        raise ValueError(f'{", ".join(present)} in [model] do not apply to {model_name}: {why}. Remove them.')


def _rows(tensor, batch):
    return torch.is_tensor(tensor) and tensor.ndim > 0 and tensor.shape[0] == batch


def _is_output(value):
    """A tensor, or a list with one tensor (or None) per image, as MiniMax's audio latents are."""
    return torch.is_tensor(value) or (
        isinstance(value, (list, tuple)) and any(torch.is_tensor(x) for x in value))


def _take(value, index, batch):
    if _rows(value, batch):
        return value[index.to(value.device)]
    if isinstance(value, (list, tuple)) and len(value) == batch:
        return type(value)(value[i] for i in index.tolist())
    return value


def _float(value):
    if torch.is_tensor(value):
        return value.float()
    return [x.float() if torch.is_tensor(x) else x for x in value]


def _to(value, dtype):
    if torch.is_tensor(value):
        return value.to(dtype)
    return [x.to(dtype) if torch.is_tensor(x) else x for x in value]


def _finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    return all(torch.isfinite(x).all() for x in value if torch.is_tensor(x))


def _bad_rows(value, batch, device):
    """Which images' entries are non-finite. All of them for an entry that is not per image."""
    if _rows(value, batch):
        return ~torch.isfinite(value.flatten(1)).all(1)
    if isinstance(value, (list, tuple)) and len(value) == batch:
        return torch.tensor([torch.is_tensor(x) and not bool(torch.isfinite(x).all()) for x in value], device=device)
    return torch.full((batch,), not _finite(value), device=device)


def wrap_call_vae_fn(fn, dtypes, keys, fallback_fn):
    """fn: a model's call_vae_fn. keys: the entries of its result that the VAE produced.
    fallback_fn(dtype): a call_vae_fn for the VAE loaded in dtype (only called after an overflow).

    The result has those entries in dtypes.store (dtypes must be active).
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        result = dict(fn(*args, **kwargs))
        present = [k for k in keys if _is_output(result.get(k))]
        if not present:
            return result
        # Worked on in float32, so a row encoded again in a wider dtype does not overflow here.
        work = {k: _float(result[k]) for k in present}
        tried = dtypes.encode
        if dtypes.encode is not None:
            tried = _encode_again(work, args, kwargs, dtypes.encode, fallback_fn)
            for k in present:
                if not _finite(work[k]):
                    raise RuntimeError(f'The VAE gave non-finite {k} even in {NAMES[tried]}: '
                                       'an image in this batch has non-finite pixels')
        store = dtypes.store
        for k in present:
            stored = _to(work[k], store)
            if _finite(work[k]) and not _finite(stored):
                print(f'WARNING: {k} do not fit in {NAMES[store]}; '
                      f'caching this batch of {len(work[k])} images in float32.')
                stored = work[k]
            result[k] = stored
        return result
    return wrapped


def _encode_again(work, args, kwargs, encode, fallback_fn):
    """Re-encodes the images whose entries of `work` are non-finite, in each wider dtype in turn.
    Returns the last dtype tried."""
    tried = encode
    first = next(iter(work.values()))
    batch = len(first)
    device = first.device if torch.is_tensor(first) else next(x.device for x in first if torch.is_tensor(x))
    for dtype in FALLBACKS[encode]:
        bad = torch.zeros(batch, dtype=torch.bool, device=device)
        for v in work.values():
            bad |= _bad_rows(v, batch, device)
        if not bad.any():
            break
        print(f'WARNING: the VAE in {NAMES[tried]} gave non-finite latents for {int(bad.sum())} of '
              f'{len(bad)} images; encoding them again in {NAMES[dtype]}. If this happens often, '
              f"set vae_dtype = '{NAMES[dtype]}' in [model].")
        index = bad.nonzero().flatten()
        redo = bool(bad.all())
        sub_args = args if redo else tuple(_take(a, index, batch) for a in args)
        sub_kwargs = kwargs if redo else {k: _take(v, index, batch) for k, v in kwargs.items()}
        again = fallback_fn(dtype)(*sub_args, **sub_kwargs)
        for k, v in work.items():
            new = _float(again[k])
            if redo:
                work[k] = new
            elif torch.is_tensor(v):
                v[index] = new.to(v.device)
            else:
                for j, i in enumerate(index.tolist()):
                    v[i] = new[j]
        tried = dtype
    return tried
