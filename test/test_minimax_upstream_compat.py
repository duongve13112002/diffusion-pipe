"""Exercise the native MiniMax constructor against the training monkey patches."""

import pytest
import torch

from models.minimax_h3 import Attention, DiTBlock
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ops import disable_weight_init


@pytest.mark.parametrize('gate_compress', [False, True])
def test_native_model_builds_with_training_blocks(gate_compress):
    model = MiniMaxH3Model(
        hidden_size=8, num_layers=1, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=4, ffn_hidden_size=16,
        latents_dim=2, audio_latents_dim=2, patch_size=(1, 1, 1), text_dim=8,
        timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=8,
        rope_inv_freq_len=1, gate_compress=gate_compress,
        dtype=torch.float32, device='cpu', operations=disable_weight_init,
    )
    block = model.blocks[0]
    assert isinstance(block, DiTBlock)
    assert isinstance(block.attn, Attention)
    gate_key = 'blocks.0.attn.to_gate_compress.weight'
    assert (gate_key in model.state_dict()) == gate_compress
    if gate_compress:
        assert block.attn.to_gate_compress.weight.shape == (8, 8)

    # The gate is stored for VSA checkpoints; dense training still uses ordinary attention.
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(0.05)
    x = torch.randn(2, 3, 8, requires_grad=True)
    t_emb = torch.randn(2, 1, 8)
    output = block(x, t_emb, [(0, 3, 0)], rope_freqs=None)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(block.attn.qkv_proj.weight.grad).all()


def build_block():
    block = DiTBlock(8, 2, 4, 16, 8, 1e-6, 1e-6, dtype=torch.float32, device='cpu', operations=disable_weight_init)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(0.05)
    return block


def test_block_uses_an_attention_passed_by_comfyui():
    """ComfyUI hands a block its own (patched) attention as `attention=`, as in its block-replace path."""
    from comfy.ldm.modules.attention import ComfyAttention
    block = build_block()
    assert isinstance(block.attn.comfy_attention, ComfyAttention)
    x, t_emb = torch.randn(2, 3, 8), torch.randn(2, 1, 8)
    calls = []

    def patched(h, rope_freqs=None, transformer_options=None, attention_mask=None):
        calls.append(h.shape)
        return torch.zeros_like(h)

    patched_out = block(x, t_emb, [(0, 3, 0)], None, transformer_options={}, attention=patched)
    assert calls == [x.shape]
    # No attention: the block's own attention runs, so the two outputs differ.
    assert not torch.allclose(patched_out, block(x, t_emb, [(0, 3, 0)], None))


def test_attention_uses_the_regular_attention_not_the_checkpoints_choice(monkeypatch):
    import models.minimax_h3 as minimax_h3
    from comfy.ldm.modules.attention import AttentionTensorContainer
    attention = Attention(8, 2, 4, 1e-6, dtype=torch.float32, device='cpu', operations=disable_weight_init)
    seen = {}

    def spy(q, k, v, heads, **kwargs):
        seen.update(kwargs, containers=[type(t) for t in (q, k, v)])
        return torch.zeros(q.peek().shape[0], q.peek().shape[2], 8)

    monkeypatch.setattr(minimax_h3, 'optimized_attention', spy)
    attention(torch.randn(2, 3, 8))
    # an int8 convrot checkpoint's ComfyAttention config would pick an int8 attention kernel, which cannot train
    assert 'preferred_attention' not in seen
    assert seen['containers'] == [AttentionTensorContainer] * 3


def test_timestep_rows_are_per_kind_not_per_distinct_value():
    # t_v and t_a coincide at some sigmas (1.0). Rows deduplicated by value then number 2 in one sample and 3 in
    # another, and the batch's t_emb could not be stacked.
    import inspect
    import models.minimax_h3 as minimax_h3
    source = inspect.getsource(minimax_h3.InitialLayer.make_packed_sequence)
    assert 'unique_t' not in source
    assert 'kind_row = {kind: i for i, kind in enumerate(seg_t)}' in source
