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
