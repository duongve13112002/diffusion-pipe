"""QwenImage21TransformerBlock must backpropagate in training.

ComfyUI 29e28e2 made the block's gated residual and fp16 clamp in-place. models/qwen_image21.py
restores the out-of-place versions while comfy.model_management.in_training is set; these
tests run a tiny block on CPU through that path.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../submodules/ComfyUI'))

import models.qwen_image21  # noqa: F401  (installs the training-time block forward)
import comfy.model_management
import comfy.ops
import comfy.ldm.qwen_image21.model as qi21


DIM, HEADS = 32, 2


def make_block():
    torch.manual_seed(0)
    block = qi21.QwenImage21TransformerBlock(DIM, HEADS, DIM // HEADS, operations=comfy.ops.disable_weight_init)
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return block


def attn_fn(q, k, v, heads, preferred_attention=None):
    q, k, v = q.take(), k.take(), v.take()
    B, N = q.shape[:2]
    out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).reshape(B, N, -1)


def inputs(prefix_len, n=6, leaf=True):
    torch.manual_seed(1)
    x = torch.randn(1, n, DIM, requires_grad=True)
    if not leaf:
        x = x * 1.0
    mod = tuple((torch.randn(1, 1, DIM) * 0.1, torch.randn(1, 1, DIM) * 0.1) for _ in range(4))
    pe = torch.zeros(1, n, 1, DIM // HEADS // 2, 2, 2)
    pe[..., 0, 0] = 1
    pe[..., 1, 1] = 1
    return x, (*mod, torch.zeros(1, 1, DIM)), pe


def reference(block, x, mod, pe, prefix_len):
    """The block written out by hand, out of place."""
    scale1, gate1, scale2, gate2, _ = mod

    def mnorm(norm, h, scale):
        out = norm(h)
        return torch.cat([out[:, :prefix_len] * (1 + scale[0]), out[:, prefix_len:] * (1 + scale[1])], dim=1)

    def residual(h, y, gate):
        return torch.cat([h[:, :prefix_len] + y[:, :prefix_len] * gate[0], h[:, prefix_len:] + y[:, prefix_len:] * gate[1]], dim=1)

    h = residual(x, block.attn(mnorm(block.img_norm1, x, scale1), pe, attn_fn, prefix_len), gate1)
    return residual(h, block.img_mlp(mnorm(block.img_norm2, h, scale2)), gate2)


@pytest.fixture(autouse=True)
def training_mode():
    old = comfy.model_management.in_training
    comfy.model_management.in_training = True
    yield
    comfy.model_management.in_training = old


@pytest.mark.parametrize('leaf', [True, False])
@pytest.mark.parametrize('prefix_len', [0, 2])
def test_backward_matches_out_of_place_reference(leaf, prefix_len):
    block = make_block()
    x, mod, pe = inputs(prefix_len, leaf=leaf)
    out = block(x, mod, pe, attn_fn, prefix_len)
    grads = torch.autograd.grad(out.square().sum(), list(block.parameters()))

    x2, mod2, pe2 = inputs(prefix_len, leaf=leaf)
    ref = reference(block, x2, mod2, pe2, prefix_len)
    ref_grads = torch.autograd.grad(ref.square().sum(), list(block.parameters()))
    torch.testing.assert_close(out, ref)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r)


def test_fp16_clamp_is_out_of_place():
    block = make_block().half()
    x, mod, pe = inputs(2)
    x = x.detach().half().requires_grad_(True)
    mod = tuple(tuple(t.half() for t in m) if isinstance(m, tuple) else m.half() for m in mod)
    out = block(x, mod, pe, attn_fn, 2)
    out.float().sum().backward()
    assert x.grad is not None
