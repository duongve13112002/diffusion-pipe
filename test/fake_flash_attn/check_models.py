"""The models that call flash-attn directly, with [model] attention = 'flash3' / 'flash4' / 'sdpa'.

Run by test/test_attention_models.py with this directory on PYTHONPATH, so `flash_attn` (2),
`flash_attn.cute` (4) and `flash_attn_interface` (3) import the stand-ins here. Each check runs the
model's own attention call with every backend, asserts which kernel ran and that all agree with
PyTorch SDPA.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

import flash_attn
import flash_attn_interface
from flash_attn import cute as flash_attn_cute
from utils import attention

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FAKES = {'flash3': flash_attn_interface, 'flash4': flash_attn_cute}


def make(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(attention, 'device_ok', lambda t: True)
    for module in (flash_attn, flash_attn_interface, flash_attn_cute):
        module.CALLS.clear()
    yield


def use(monkeypatch, backend):
    """Sets the backend; returns the CALLS list of the kernel that should run."""
    monkeypatch.setattr(attention, '_backend', backend)
    monkeypatch.setattr(attention, '_module', FAKES.get(backend))
    # What install() sets: the models' own FlashAttention calls only run from then on.
    monkeypatch.setattr(attention, '_original_sdpa', F.scaled_dot_product_attention if backend in FAKES else None)
    return FAKES[backend].CALLS if backend in FAKES else flash_attn.CALLS


def close(a, b):
    torch.testing.assert_close(a.float(), b.float(), atol=3e-2, rtol=0)


def sdpa_bshd(q, k, v, keep=None):
    """PyTorch SDPA on (B, S, H, D), keys masked by keep (B, S)."""
    mask = None if keep is None else keep[:, None, None, :]
    out = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)), attn_mask=mask)
    return out.transpose(1, 2)


@pytest.mark.parametrize('backend', ['flash3', 'flash4', 'sdpa'])
def test_hidream(monkeypatch, backend):
    import models.hidream as hidream
    assert hidream.hidream_attention_processor.attention is hidream._hidream_attention
    calls = use(monkeypatch, backend)
    q, k, v = (make(2, 12, 2, 32, seed=i) for i in range(3))
    out = hidream.hidream_attention_processor.attention(q, k, v)
    assert calls and calls[-1][0] == 'flash_attn_func'
    assert out.shape == (2, 12, 64)
    close(out, sdpa_bshd(q, k, v).flatten(-2))


def test_hidream_without_flash_attn_2(monkeypatch):
    import models.hidream as hidream
    use(monkeypatch, 'sdpa')
    monkeypatch.setattr(hidream, 'flash_attn_2', None)
    q, k, v = (make(2, 12, 2, 32, seed=i) for i in range(3))
    close(hidream.hidream_attention_processor.attention(q, k, v), sdpa_bshd(q, k, v).flatten(-2))
    assert not flash_attn.CALLS and not flash_attn_interface.CALLS


@pytest.mark.parametrize('backend', ['flash3', 'flash4', 'sdpa'])
def test_hunyuan_video(monkeypatch, backend):
    import models.hunyuan_video  # noqa: F401  (installs the patch)
    from hyvideo.modules import attenion
    calls = use(monkeypatch, backend)
    b, s, h, d = 2, 10, 2, 32
    q, k, v = (make(b, s, h, d, seed=i) for i in range(3))
    # get_cu_seqlens: per batch item, the real tokens then the padding as a separate sequence.
    lengths = [10, 6]
    cu = torch.tensor([0, 10, 10, 16, 20], dtype=torch.int32)
    out = attenion.attention(q, k, v, mode='flash', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=s,
                             max_seqlen_kv=s, batch_size=b)
    assert calls[-1][0] == 'flash_attn_varlen_func' and calls[-1][1]['cu_seqlens_q'] == cu.tolist()
    assert out.shape == (b, s, h * d)
    keep = torch.arange(s)[None, :] < torch.tensor(lengths)[:, None]
    ref = sdpa_bshd(q, k, v, keep).flatten(-2)
    close(out[0], ref[0])
    close(out[1, :6], ref[1, :6])


@pytest.mark.parametrize('backend', ['flash3', 'flash4', 'sdpa'])
def test_hunyuan_image(monkeypatch, backend):
    sys.path.insert(0, os.path.join(ROOT, 'submodules', 'HunyuanImage-2.1'))
    import models.hunyuan_image_modeling as modeling
    from hyimage.models.hunyuan.modules import token_refiner
    assert token_refiner.flash_attn_no_pad is modeling.flash_attn_no_pad
    calls = use(monkeypatch, backend)
    b, si, st, h, d = 2, 8, 5, 2, 32
    img = [make(b, si, h, d, seed=i) for i in range(3)]
    txt = [make(b, st, h, d, seed=10 + i) for i in range(3)]
    text_mask = torch.arange(st)[None, :] < torch.tensor([5, 2])[:, None]
    out = modeling.attention(*zip(img, txt), text_mask=text_mask)
    expected = 'flash_attn_varlen_qkvpacked_func' if backend == 'sdpa' else 'flash_attn_varlen_func'
    assert calls[-1][0] == expected
    assert calls[-1][1].get('cu_seqlens', calls[-1][1].get('cu_seqlens_q')) == [0, 13, 23]
    keep = torch.cat([torch.ones(b, si, dtype=torch.bool), text_mask], 1)
    ref = sdpa_bshd(*(torch.cat(pair, 1) for pair in zip(img, txt)), keep).flatten(-2) * keep[..., None]
    assert out.shape == (b, si + st, h * d)
    close(out, ref)


@pytest.mark.parametrize('backend', ['flash3', 'flash4', 'sdpa'])
def test_omnigen2(monkeypatch, backend):
    import models.omnigen2 as omnigen2
    processor = omnigen2.omnigen2_attention_processor
    assert processor.flash_attn_varlen_func is omnigen2._varlen_attention
    calls = use(monkeypatch, backend)
    q, k, v = (make(9, 2, 32, seed=i) for i in range(3))
    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    out = processor.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=5,
                                           max_seqlen_k=5, dropout_p=0.0, causal=False, softmax_scale=0.2)
    assert calls[-1][0] == 'flash_attn_varlen_func'
    ref = torch.cat([F.scaled_dot_product_attention(*(t[a:b].float().transpose(0, 1) for t in (q, k, v)),
                                                    scale=0.2).transpose(0, 1) for a, b in ((0, 4), (4, 9))])
    close(out, ref)


@pytest.mark.parametrize('backend', ['flash3', 'flash4', 'sdpa'])
def test_wan(monkeypatch, backend):
    from models.wan import attention as wan_attention
    assert wan_attention.FLASH_ATTN_2_AVAILABLE
    calls = use(monkeypatch, backend)
    b, lq, lk, h, d = 2, 6, 9, 2, 32
    q, k, v = make(b, lq, h, d, seed=1), make(b, lk, h, d, seed=2), make(b, lk, h, d, seed=3)
    out = wan_attention.flash_attention(q, k, v, k_lens=torch.tensor([9, 4], dtype=torch.int32))
    assert calls[-1][0] == 'flash_attn_varlen_func' and calls[-1][1]['cu_seqlens_q'] == [0, 6, 12]
    keep = torch.arange(lk)[None, :] < torch.tensor([9, 4])[:, None]
    close(out, sdpa_bshd(q, k, v, keep))


def test_wan_dropout_with_flash3_runs_flash_attn_2(monkeypatch):
    from models.wan import attention as wan_attention
    use(monkeypatch, 'flash3')
    q = make(1, 4, 2, 32)
    monkeypatch.setattr(flash_attn, 'flash_attn_varlen_func',
                        lambda *a, **kw: flash_attn.CALLS.append(('dropout', kw['dropout_p'])) or q[0])
    wan_attention.flash_attention(q, q, q, dropout_p=0.1)
    assert flash_attn.CALLS == [('dropout', 0.1)] and not flash_attn_interface.CALLS


@pytest.mark.parametrize('backend', ['flash3', 'flash4', 'sdpa'])
def test_lumina_2(monkeypatch, backend):
    import models.lumina_2 as lumina_2
    model = lumina_2.lumina_2_model
    assert model.flash_attn_varlen_func is lumina_2._varlen_attention
    calls = use(monkeypatch, backend)
    torch.manual_seed(0)
    attn = model.JointAttention(64, 4, 2, qk_norm=True)
    b, s = 2, 7
    x = make(b, s, 64, seed=1).float()
    x_mask = torch.arange(s)[None, :] < torch.tensor([7, 4])[:, None]
    g = torch.Generator().manual_seed(2)
    freqs_cis = torch.polar(torch.ones(b, s, 8), torch.randn(b, s, 8, generator=g))
    out = attn.to(torch.bfloat16)(x.to(torch.bfloat16), x_mask, freqs_cis)
    assert calls[-1][0] == 'flash_attn_varlen_func' and calls[-1][1]['cu_seqlens_q'] == [0, 7, 11]
    # Lumina's float32 path: SDPA with the mask, keys and values repeated for GQA.
    ref = attn.float()(x, x_mask, freqs_cis)
    close(out[x_mask], ref[x_mask])
