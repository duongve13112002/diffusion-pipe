"""[model] attention = 'flash3' / 'flash4': the SDPA replacement, the FlashAttention helpers and the
config checks, on CPU with the stand-in FlashAttention-3 and -4 in test/fake_flash_attn (plain torch
math, the same semantics and arguments).

Every routed call is compared, output and gradients, with PyTorch's own SDPA in float32. The
models that call flash-attn directly are checked in test_attention_models.py.
"""

import gc
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from utils import attention

REAL_SDPA = F.scaled_dot_product_attention
FAKE_DIR = Path(__file__).parent / 'fake_flash_attn'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fake(backend, monkeypatch):
    fake3 = load('fake_flash_attn_interface', FAKE_DIR / 'flash_attn_interface.py')
    if backend == 'flash3':
        return fake3
    # The fake FA4 borrows the fake FA3's math by importing it under its real name.
    monkeypatch.setitem(sys.modules, 'flash_attn_interface', fake3)
    return load('fake_flash_attn_cute', FAKE_DIR / 'flash_attn' / 'cute' / '__init__.py')


@pytest.fixture(params=['flash3', 'flash4'])
def fa3(request, monkeypatch):
    """The SDPA replacement installed with a fake FlashAttention-3 or -4 (both run every test)."""
    fake = load_fake(request.param, monkeypatch)
    monkeypatch.setattr(attention, '_module', fake)
    monkeypatch.setattr(attention, 'device_ok', lambda t: True)
    monkeypatch.setattr(attention, '_backend', request.param)
    monkeypatch.setattr(attention, '_arch', 9)
    attention.reset_counts()
    attention.install()
    yield fake
    attention.uninstall()
    attention.reset_counts()


def make(*shape, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype).requires_grad_()


def qkv(b=2, hq=4, hk=4, lq=16, lk=16, d=64, dtype=torch.bfloat16):
    return make(b, hq, lq, d, dtype=dtype, seed=1), make(b, hk, lk, d, dtype=dtype, seed=2), \
        make(b, hk, lk, d, dtype=dtype, seed=3)


def compare(q, k, v, *args, atol=3e-2, **kwargs):
    """The replaced SDPA against PyTorch's in float32: output and q / k / v gradients."""
    out = F.scaled_dot_product_attention(q, k, v, *args, **kwargs)
    ref_in = [t.detach().float().requires_grad_() for t in (q, k, v)]
    args = [a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args]
    if torch.is_tensor(kwargs.get('attn_mask')) and kwargs['attn_mask'].is_floating_point():
        kwargs = dict(kwargs, attn_mask=kwargs['attn_mask'].float())
    ref = REAL_SDPA(*ref_in, *args, **kwargs)
    assert out.shape == ref.shape
    grad = torch.randn(ref.shape, generator=torch.Generator().manual_seed(9))
    (out.float() * grad).sum().backward()
    (ref * grad).sum().backward()
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=0)
    for mine, theirs in zip((q, k, v), ref_in):
        torch.testing.assert_close(mine.grad.float(), theirs.grad, atol=2 * atol, rtol=0)
    return out


def key_padding(b, lk, lengths):
    return torch.arange(lk)[None, :] < torch.tensor(lengths)[:, None]


def test_dense_call_runs_fa3(fa3):
    out = compare(*qkv())
    assert out.dtype == torch.bfloat16
    assert [c[0] for c in fa3.CALLS] == ['flash_attn_func']
    assert attention.counts() == {attention.FLASH_CALL: 1}


def test_output_layout_matches_sdpa(fa3):
    q, k, v = qkv(b=2, hq=3, hk=3, lq=5, d=16)
    out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 5, 16)
    out.transpose(1, 2).reshape(2, 5, 48)


def test_gqa(fa3):
    compare(*qkv(hq=8, hk=2), enable_gqa=True)
    assert attention.counts() == {attention.FLASH_CALL: 1}


def test_gqa_without_enable_gqa_is_left_to_sdpa(fa3):
    with pytest.raises(RuntimeError):
        F.scaled_dot_product_attention(*qkv(hq=8, hk=2))
    assert attention.counts() == {'different numbers of query and key heads': 1}


def test_causal_same_length(fa3):
    compare(*qkv(), is_causal=True)
    assert fa3.CALLS[-1][1]['causal'] is True


def test_causal_different_lengths_runs_sdpa(fa3):
    # FA3 aligns the causal mask bottom-right, SDPA top-left: only equal lengths agree.
    compare(*qkv(lq=8, lk=16), is_causal=True)
    assert fa3.CALLS == []
    assert attention.counts() == {'is_causal with different query and key lengths': 1}


def test_cross_attention_lengths(fa3):
    compare(*qkv(lq=12, lk=7))
    assert attention.counts() == {attention.FLASH_CALL: 1}


def test_scale(fa3):
    compare(*qkv(), scale=0.3)
    assert fa3.CALLS[-1][1]['scale'] == 0.3


def test_bool_key_padding_runs_varlen(fa3):
    q, k, v = qkv(lq=10, lk=16)
    mask = key_padding(2, 16, [16, 9])[:, None, None, :]
    compare(q, k, v, attn_mask=mask)
    name, info = fa3.CALLS[-1]
    assert name == 'flash_attn_varlen_func'
    assert info['cu_seqlens_q'] == [0, 10, 20] and info['cu_seqlens_k'] == [0, 16, 25]
    assert attention.counts() == {attention.FLASH_PADDED: 1}
    # Padded keys get no gradient.
    assert torch.all(k.grad[1, :, 9:] == 0) and torch.all(v.grad[1, :, 9:] == 0)


def test_key_padding_with_holes(fa3):
    keep = torch.ones(2, 16, dtype=torch.bool)
    keep[0, 3:6] = False
    keep[1, ::2] = False
    compare(*qkv(), attn_mask=keep[:, None, None, :])
    assert fa3.CALLS[-1][1]['cu_seqlens_k'] == [0, 13, 21]


@pytest.mark.parametrize('fill', [float('-inf'), 'min', -1e4])
def test_float_key_padding(fa3, fill):
    q, k, v = qkv(lq=6, lk=16)
    keep = key_padding(2, 16, [11, 16])
    value = torch.finfo(q.dtype).min if fill == 'min' else fill
    # A full (B, 1, Lq, Lk) mask whose rows agree: still key padding.
    mask = torch.zeros(2, 1, 6, 16, dtype=q.dtype).masked_fill(~keep[:, None, None, :], value)
    compare(q, k, v, attn_mask=mask)
    assert fa3.CALLS[-1][1]['cu_seqlens_k'] == [0, 11, 27]


def test_mask_keeping_everything_runs_dense(fa3):
    compare(*qkv(), attn_mask=torch.ones(2, 1, 1, 16, dtype=torch.bool))
    assert [c[0] for c in fa3.CALLS] == ['flash_attn_func']
    assert attention.counts() == {attention.FLASH_CALL: 1}


def test_broadcast_mask_shapes(fa3):
    keep = key_padding(1, 16, [12])
    compare(*qkv(), attn_mask=keep)  # (1, Lk): the same keys for every batch item
    assert fa3.CALLS[-1][1]['cu_seqlens_k'] == [0, 12, 24]
    compare(*qkv(), attn_mask=key_padding(2, 16, [5, 16])[:, None, None, :].expand(2, 4, 16, 16))
    assert fa3.CALLS[-1][1]['cu_seqlens_k'] == [0, 5, 21]


def test_causal_mask_tensor_runs_sdpa(fa3):
    mask = torch.ones(16, 16, dtype=torch.bool).tril()
    compare(*qkv(), attn_mask=mask)
    assert fa3.CALLS == []
    assert attention.counts() == {'a mask that is not key padding': 1}


def test_bias_runs_sdpa(fa3):
    q, k, v = qkv()
    bias = torch.randn(2, 1, 16, 16).to(q.dtype)
    compare(q, k, v, attn_mask=bias)
    assert attention.counts() == {'a float mask with bias values': 1}


def test_per_head_padding_runs_sdpa(fa3):
    mask = torch.ones(2, 4, 1, 16, dtype=torch.bool)
    mask[0, 1, :, 4:] = False
    compare(*qkv(), attn_mask=mask)
    assert attention.counts() == {'a mask that is not key padding': 1}


def test_batch_item_with_no_keys_runs_sdpa(fa3):
    q, k, v = qkv()
    mask = key_padding(2, 16, [0, 16])[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(out, REAL_SDPA(q, k, v, attn_mask=mask), equal_nan=True)
    assert attention.counts() == {'a mask that hides every key of a batch item': 1}


def test_float32_runs_sdpa(fa3):
    compare(*qkv(dtype=torch.float32), atol=1e-5)
    assert attention.counts() == {'torch.float32 inputs': 1}


def test_autocast_float32_runs_fa3(fa3):
    q, k, v = qkv(dtype=torch.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = F.scaled_dot_product_attention(q, k, v)
    assert out.dtype == torch.bfloat16
    assert attention.counts() == {attention.FLASH_CALL: 1}
    torch.testing.assert_close(out.float(), REAL_SDPA(q, k, v), atol=3e-2, rtol=0)
    out.float().sum().backward()
    assert q.grad.dtype == torch.float32


def test_mixed_dtypes_run_sdpa(fa3):
    q, k, v = qkv()
    with pytest.raises(RuntimeError):
        F.scaled_dot_product_attention(q.float(), k, v)
    assert attention.counts() == {'inputs of different dtypes': 1}


def test_dropout_runs_sdpa(fa3):
    F.scaled_dot_product_attention(*qkv(), dropout_p=0.5)
    assert attention.counts() == {'attention dropout': 1}


@pytest.mark.parametrize('d', [264, 20])
def test_head_dim_runs_sdpa(fa3, d):
    compare(*qkv(b=1, hq=1, hk=1, lq=4, lk=4, d=d))
    assert attention.counts() == {f'head_dim {d}': 1}


def test_three_dim_inputs_run_sdpa(fa3):
    q, k, v = (make(4, 8, 32, seed=s) for s in range(3))
    compare(q, k, v)
    assert attention.counts() == {'inputs that are not 4-D': 1}


def test_not_on_gpu_runs_sdpa(fa3, monkeypatch):
    monkeypatch.setattr(attention, 'device_ok', lambda t: False)
    compare(*qkv())
    assert attention.counts() == {'inputs not on one GPU': 1}


def test_positional_arguments(fa3):
    compare(*qkv(), key_padding(2, 16, [16, 3])[:, None, None, :], 0.0, False)
    assert attention.counts() == {attention.FLASH_PADDED: 1}


def test_mask_is_checked_once_per_mask(fa3, monkeypatch):
    checked = []
    real = attention._analyse_mask
    monkeypatch.setattr(attention, '_analyse_mask', lambda *a: checked.append(1) or real(*a))
    mask = key_padding(2, 16, [16, 7])[:, None, None, :].clone()
    for _ in range(3):
        compare(*qkv(), attn_mask=mask)
    # A view of the same mask (OmniGen2 does mask.bool().view(...) in every layer) too.
    compare(*qkv(), attn_mask=mask.view(2, 1, 1, 16))
    assert len(checked) == 1
    # Changed in place: checked again, and the new padding is used.
    mask[0, ..., 12:] = False
    compare(*qkv(), attn_mask=mask)
    assert len(checked) == 2
    assert fa3.CALLS[-1][1]['cu_seqlens_k'] == [0, 12, 19]


def test_mask_cache_entry_goes_with_the_mask(fa3):
    mask = key_padding(2, 16, [16, 7])[:, None, None, :].clone()
    compare(*qkv(), attn_mask=mask)
    assert attention._mask_cache
    del mask
    gc.collect()
    assert not attention._mask_cache


def test_inference_mode_mask(fa3):
    with torch.inference_mode():
        mask = key_padding(2, 16, [16, 7])[:, None, None, :]
        q, k, v = (t.detach() for t in qkv())
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(out.float(), REAL_SDPA(q.float(), k.float(), v.float(), attn_mask=mask),
                               atol=3e-2, rtol=0)


def test_old_fa3_tuple_return(fa3):
    # FA4 returns (out, lse) anyway.
    fa3.RETURN_TUPLE = True
    compare(*qkv())
    compare(*qkv(), attn_mask=key_padding(2, 16, [16, 3])[:, None, None, :])


def test_deterministic_flag(fa3):
    torch.use_deterministic_algorithms(True)
    try:
        compare(*qkv())
    finally:
        torch.use_deterministic_algorithms(False)
    assert fa3.CALLS[-1][1]['deterministic'] is True
    compare(*qkv())
    assert fa3.CALLS[-1][1]['deterministic'] is False


def test_install_and_uninstall(monkeypatch):
    monkeypatch.setattr(attention, '_module', load_fake('flash3', monkeypatch))
    monkeypatch.setattr(attention, '_backend', 'sdpa')
    attention.install()
    assert F.scaled_dot_product_attention is REAL_SDPA
    monkeypatch.setattr(attention, '_backend', 'flash3')
    attention.install()
    try:
        assert F.scaled_dot_product_attention is attention.scaled_dot_product_attention
        attention.install()
        assert attention._original_sdpa is REAL_SDPA
    finally:
        attention.uninstall()
    assert F.scaled_dot_product_attention is REAL_SDPA


def test_report_and_first_notes(fa3, capsys):
    capsys.readouterr()
    compare(*qkv())
    compare(*qkv(), attn_mask=key_padding(2, 16, [16, 3])[:, None, None, :])
    for _ in range(2):
        compare(*qkv(dtype=torch.float32), atol=1e-5)
    notes = capsys.readouterr().out
    name = attention.label()
    assert name == {'flash3': 'FlashAttention-3', 'flash4': 'FlashAttention-4'}[attention._backend]
    assert notes.count(f'PyTorch SDPA runs a call {name} cannot take: torch.float32 inputs') == 1
    attention.report()
    line = capsys.readouterr().out
    assert f'{name} ran 2 calls (1 with key padding)' in line
    assert 'PyTorch SDPA ran 2: 2 torch.float32 inputs' in line


def test_report_without_install(capsys):
    attention.report()
    assert capsys.readouterr().out == ''


def model_config(**model):
    return {'model': {'type': 'flux', **model}}


def test_attention_setting():
    assert attention.attention_setting(model_config()) == 'sdpa'
    assert attention.attention_setting(model_config(attention='flash3')) == 'flash3'
    for bad in ('flash2', 'FLASH3', True, 3):
        with pytest.raises(ValueError, match=r'\[model\] attention must be one of'):
            attention.attention_setting(model_config(attention=bad))
    assert attention.attention_setting(model_config(attention='flash4')) == 'flash4'
    for flash in ('flash3', 'flash4'):
        with pytest.raises(ValueError, match='compile'):
            attention.attention_setting({**model_config(attention=flash), 'compile': True})
        with pytest.raises(ValueError, match='Transformer Engine'):
            attention.attention_setting({'model': {'type': 'cosmos', 'attention': flash}})
    assert attention.attention_setting({**model_config(), 'compile': True}) == 'sdpa'


def test_configure(monkeypatch):
    monkeypatch.setattr(attention, '_backend', 'sdpa')
    monkeypatch.setattr(attention, '_module', None)
    monkeypatch.setattr(attention, '_arch', 9)
    assert attention.configure(model_config()) == 'sdpa' and attention._backend == 'sdpa'

    def missing(backend):
        raise ImportError(f'no {backend}')

    monkeypatch.setattr(attention, 'import_module', missing)
    with pytest.raises(RuntimeError, match='flash_attn_3'):
        attention.configure(model_config(attention='flash3'))
    with pytest.raises(RuntimeError, match='pip install flash-attn-4'):
        attention.configure(model_config(attention='flash4'))
    monkeypatch.setattr(attention, 'import_module', lambda backend: backend)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA GPU'):
        attention.configure(model_config(attention='flash3'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'B200')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (10, 0))
    with pytest.raises(RuntimeError, match='no FlashAttention-3 kernels'):
        attention.configure(model_config(attention='flash3'))
    assert attention._backend == 'sdpa'
    assert attention.configure(model_config(attention='flash4')) == 'flash4'
    assert attention._backend == 'flash4' and attention._arch == 10 and attention._module == 'flash4'
    # Not before install(), which train.py calls after caching.
    assert not attention.flash_enabled()
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (8, 6))
    with pytest.raises(RuntimeError, match='cannot train with FlashAttention-4'):
        attention.configure(model_config(attention='flash4'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (9, 0))
    assert attention.configure(model_config(attention='flash3')) == 'flash3' and attention._module == 'flash3'


def test_fa3_unpadded(fa3):
    b, s, h, d = 2, 12, 2, 32
    q, k, v = (make(b, s, h, d, seed=i) for i in range(3))
    keep = key_padding(b, s, [12, 7])
    out = attention.flash_unpadded(q, k, v, keep)
    ref = REAL_SDPA(*(t.detach().float().transpose(1, 2) for t in (q, k, v)),
                    attn_mask=keep[:, None, None, :]).transpose(1, 2)
    ref = ref * keep[:, :, None, None]
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=0)
    assert torch.all(out[1, 7:] == 0)
    out.sum().backward()
    assert torch.all(q.grad[1, 7:] == 0) and torch.all(k.grad[1, 7:] == 0)


def test_fa3_helpers_refuse_what_fa3_lacks(fa3):
    q = make(6, 2, 32)
    cu = torch.tensor([0, 6], dtype=torch.int32)
    with pytest.raises(ValueError, match='dropout'):
        attention.flash_varlen(q, q, q, cu, cu, 6, 6, dropout_p=0.1)
    q32 = make(1, 6, 2, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match='float16 / bfloat16'):
        attention.flash_dense(q32, q32, q32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert attention.flash_dense(q32, q32, q32).dtype == torch.bfloat16


def test_wan_flash3_and_sdpa_paths(fa3, monkeypatch):
    from models.wan import attention as wan_attention
    b, lq, lk, h, d = 2, 10, 14, 2, 32
    q, k, v = make(b, lq, h, d, seed=1), make(b, lk, h, d, seed=2), make(b, lk, h, d, seed=3)
    k_lens = torch.tensor([14, 6], dtype=torch.int32)
    mask = key_padding(b, lk, [14, 6])[:, None, None, :]
    ref = REAL_SDPA(*(t.detach().float().transpose(1, 2) for t in (q, k, v)), attn_mask=mask).transpose(1, 2)

    out = wan_attention.flash_attention(q, k, v, k_lens=k_lens)
    assert fa3.CALLS[-1][0] == 'flash_attn_varlen_func'
    assert fa3.CALLS[-1][1]['cu_seqlens_k'] == [0, 14, 20]
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=0)

    # 'sdpa' without flash-attn 2: PyTorch SDPA with a key-padding mask, the same result.
    attention.uninstall()
    monkeypatch.setattr(attention, '_backend', 'sdpa')
    monkeypatch.setattr(wan_attention, 'FLASH_ATTN_2_AVAILABLE', False)
    calls = len(fa3.CALLS)
    out = wan_attention.flash_attention(q, k, v, k_lens=k_lens)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=0)
    out = wan_attention.attention(q, k, v, k_lens=k_lens)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=0)
    assert out.shape == (b, lq, h, d)
    assert len(fa3.CALLS) == calls


def test_wan_dropout_without_flash_attn_2(fa3, monkeypatch):
    from models.wan import attention as wan_attention
    monkeypatch.setattr(wan_attention, 'FLASH_ATTN_2_AVAILABLE', False)
    q = make(1, 4, 2, 32)
    wan_attention.flash_attention(q, q, q, dropout_p=0.1)
    assert attention.counts() == {'attention dropout': 1}


@pytest.mark.parametrize('arch, d, flash', [(9, 256, True), (9, 192, True), (10, 128, True), (10, 192, False),
                                            (10, 256, True), (12, 128, True), (12, 256, False)])
def test_fa4_head_dims_by_gpu(fa3, monkeypatch, arch, d, flash):
    monkeypatch.setattr(attention, '_backend', 'flash4')
    monkeypatch.setattr(attention, '_arch', arch)
    monkeypatch.setattr(attention, '_module', load_fake('flash4', monkeypatch))
    compare(*qkv(b=1, hq=1, hk=1, lq=4, lk=4, d=d))
    assert attention.counts() == ({attention.FLASH_CALL: 1} if flash else {f'head_dim {d}': 1})


def test_fa4_deterministic_on_sm120_runs_sdpa(fa3, monkeypatch):
    monkeypatch.setattr(attention, '_backend', 'flash4')
    monkeypatch.setattr(attention, '_arch', 12)
    monkeypatch.setattr(attention, '_module', load_fake('flash4', monkeypatch))
    torch.use_deterministic_algorithms(True)
    try:
        compare(*qkv())
    finally:
        torch.use_deterministic_algorithms(False)
    assert list(attention.counts()) == ['deterministic mode (FA4 has no deterministic backward on this GPU)']


def test_window_size_spelling(fa3):
    q = make(1, 8, 2, 32)
    attention.flash_dense(q, q, q, window_size=(2, 0))
    cu = torch.tensor([0, 8], dtype=torch.int32)
    attention.flash_varlen(q[0], q[0], q[0], cu, cu, 8, 8, window_size=(-1, -1))
    assert fa3.CALLS[-1][1]['window_size'] == (-1, -1)


def test_wan_detects_flash_attn_2_by_its_functions(monkeypatch):
    """flash-attn-4 alone installs flash_attn/cute: `import flash_attn` works, flash-attn 2 is absent."""
    import types
    from models.wan import attention as wan_attention
    monkeypatch.setitem(sys.modules, 'flash_attn', types.ModuleType('flash_attn'))
    spec = importlib.util.spec_from_file_location('wan_attention_probe', wan_attention.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.FLASH_ATTN_2_AVAILABLE is False


@pytest.mark.parametrize('model, expected', [
    ({}, None),
    ({'attn_backend': 'torch_flash'}, 'torch_flash'),
    ({'attention': 'flash3'}, 'fa3'),
    ({'attention': 'flash3', 'attn_backend': 'fa3'}, 'fa3'),
    ({'attention': 'flash4'}, 'fa4'),
    ({'attention': 'flash4', 'attn_backend': 'fa4'}, 'fa4'),
    ({'attention': 'sdpa'}, None),
    ({'attention': 'sdpa', 'attn_backend': 'torch_cudnn'}, 'torch_cudnn'),
    ({'attn_backend': 'fa3'}, 'fa3'),
])
def test_iris_attn_backend(model, expected):
    from models.iris import iris_attn_backend
    assert iris_attn_backend(model) == expected


@pytest.mark.parametrize('model', [{'attention': 'flash3', 'attn_backend': 'sdpa'},
                                   {'attention': 'flash3', 'attn_backend': 'fa4'},
                                   {'attention': 'flash4', 'attn_backend': 'fa3'},
                                   {'attention': 'sdpa', 'attn_backend': 'fa3'},
                                   {'attention': 'sdpa', 'attn_backend': 'fa4'}])
def test_iris_attn_backend_conflicts(model):
    from models.iris import iris_attn_backend
    with pytest.raises(ValueError, match='attn_backend'):
        iris_attn_backend(model)


@pytest.mark.parametrize('backend, capability, module_ok, error', [
    ('fa3', (9, 0), True, None),
    ('fa3', (8, 0), True, 'compute capability 8.0'),
    ('fa3', (9, 0), False, 'unavailable'),
    ('fa4', (10, 0), True, None),
    ('fa4', (12, 0), True, 'compute capability 12.0'),
    ('sdpa', (8, 0), False, None),
])
def test_iris_flash_backend_checked_before_caching(monkeypatch, backend, capability, module_ok, error):
    import models.iris as iris
    from iris3b.nn import attention as iris_attention

    def external(name):
        if not module_ok:
            raise RuntimeError(f"attention backend '{name}' is unavailable: cannot import it")
        return lambda *a, **kw: None

    monkeypatch.setattr(iris_attention, '_external_attention_func', external)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: capability)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'test GPU')
    if error is None:
        iris.check_iris_flash_backend(backend)
    else:
        with pytest.raises(RuntimeError, match=error):
            iris.check_iris_flash_backend(backend)


def test_nan_check_replays_bypass_the_replacement(fa3):
    from utils import nan_check
    kernels = dict(nan_check._attention_kernels())
    default = f'default kernel ({attention.label()} where it can)'
    assert list(kernels)[0] == default
    q, k, v = qkv()
    with kernels['PyTorch SDPA math kernel']():
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_padding(2, 16, [16, 3])[:, None, None, :])
    assert not fa3.CALLS and not attention.counts()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        ref = REAL_SDPA(q, k, v, attn_mask=key_padding(2, 16, [16, 3])[:, None, None, :])
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    with kernels[default]():
        F.scaled_dot_product_attention(q, k, v)
    assert [c[0] for c in fa3.CALLS] == ['flash_attn_func']


def test_bypass_nests():
    with attention.bypass():
        with attention.bypass():
            assert attention._bypass == 2
        assert attention._bypass == 1
    assert attention._bypass == 0


def test_mask_cache_sees_a_swapped_storage(fa3):
    mask = key_padding(2, 16, [16, 3])[:, None, None, :]
    compare(*qkv(), attn_mask=mask)
    # .data = ... keeps the tensor's id and _version: only the storage tells the masks apart.
    mask.data = key_padding(2, 16, [5, 16])[:, None, None, :]
    compare(*qkv(), attn_mask=mask)


def test_mask_analysed_in_inference_mode_is_not_kept(fa3):
    mask = key_padding(2, 16, [16, 3])[:, None, None, :]
    q, k, v = (t.detach() for t in qkv())
    with torch.inference_mode():
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert not attention._mask_cache
    # The same mask then trains: nothing from inference mode is saved for backward.
    compare(*qkv(), attn_mask=mask)


def test_helpers_refuse_deterministic_fa4_on_sm120(fa3, monkeypatch):
    monkeypatch.setattr(attention, '_backend', 'flash4')
    monkeypatch.setattr(attention, '_arch', 12)
    monkeypatch.setattr(attention, '_module', load_fake('flash4', monkeypatch))
    q = make(1, 6, 2, 32)
    cu = torch.tensor([0, 6], dtype=torch.int32)
    attention.flash_varlen(q[0], q[0], q[0], cu, cu, 6, 6)
    with pytest.raises(RuntimeError, match='no deterministic backward'):
        attention.flash_varlen(q[0], q[0], q[0], cu, cu, 6, 6, deterministic=True)
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(RuntimeError, match='no deterministic backward'):
            attention.flash_dense(q, q, q)
    finally:
        torch.use_deterministic_algorithms(False)


def test_wan_follows_deterministic_mode(fa3):
    from models.wan import attention as wan_attention
    q = make(1, 6, 2, 32)
    torch.use_deterministic_algorithms(True)
    try:
        wan_attention.flash_attention(q, q, q)
    finally:
        torch.use_deterministic_algorithms(False)
    assert fa3.CALLS[-1][1]['deterministic'] is True


def test_models_own_flash_calls_wait_for_install(monkeypatch):
    from models.wan import attention as wan_attention
    fake = load_fake('flash3', monkeypatch)
    monkeypatch.setattr(attention, '_module', fake)
    monkeypatch.setattr(attention, '_backend', 'flash3')
    monkeypatch.setattr(attention, 'device_ok', lambda t: True)
    monkeypatch.setattr(wan_attention, 'FLASH_ATTN_2_AVAILABLE', False)
    q = make(1, 6, 2, 32)
    # Configured but not installed (caching): Wan runs PyTorch SDPA.
    wan_attention.flash_attention(q, q, q)
    assert not fake.CALLS
    attention.install()
    try:
        wan_attention.flash_attention(q, q, q)
    finally:
        attention.uninstall()
    assert [c[0] for c in fake.CALLS] == ['flash_attn_varlen_func']


@pytest.mark.parametrize('name, warned', [('attention_xformers', True), ('attention_sage', True),
                                          ('attention_pytorch', False)])
def test_install_warns_when_comfyui_bypasses_sdpa(monkeypatch, capsys, name, warned):
    import types
    comfy_attention = types.ModuleType('comfy.ldm.modules.attention')
    comfy_attention.optimized_attention = types.FunctionType((lambda: None).__code__, {}, name)
    monkeypatch.setitem(sys.modules, 'comfy.ldm.modules.attention', comfy_attention)
    monkeypatch.setattr(attention, '_module', load_fake('flash3', monkeypatch))
    monkeypatch.setattr(attention, '_backend', 'flash3')
    attention.install()
    attention.uninstall()
    assert ('does not reach the models built on ComfyUI' in capsys.readouterr().out) == warned
