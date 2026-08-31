# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context refiner: the text frontend used by the anima_refiner architecture.

Anima feeds the DiT through an LLMAdapter (models/llm_adapter.py), which embeds T5 token
ids as a query sequence and cross-attends into the LLM hidden states. That design exists
because the DiT was inherited from Cosmos-Predict2, which consumes old-T5 encoder output.

This module replaces it with the frontend Lumina 2 and Z-Image use: project the LLM hidden
states with a norm + linear (cap_embedder), then refine them with a few bidirectional
self-attention blocks. The output sequence is indexed by the LLM's own tokenization, so the
T5 tokenizer and its 32128-entry embedding table are no longer needed.

The bidirectional self-attention matters: causal LLMs (and hybrid linear-attention models
such as Qwen3.5) only give each position left context, while the DiT cross-attention was
trained on bidirectional T5 encoder output. The refiner blocks restore that mixing.
"""

import torch
from torch import nn

from models.llm_adapter import RMSNorm, Attention, RotaryEmbedding


# Signature key of a bare refiner state dict, used to tell one apart from a full checkpoint.
_REFINER_MARKER = 'cap_embedder.1.weight'


def extract_refiner_state_dict(state_dict):
    """Pull refiner weights out of any of the three shapes they are stored in.

    Refiner weights reach the loader as a bare file from distillation, as
    `net.context_refiner.*` inside a full model checkpoint, or as `context_refiner.*` after the
    caller has already stripped `net.`. All three have to work wherever a refiner is accepted,
    so this lives in one place rather than being reimplemented per call site.

    Raises when the input holds no refiner at all, rather than passing unrelated tensors on to
    fail later with an opaque KeyError or shape mismatch.
    """
    stripped = {
        (k[len('net.'):] if k.startswith('net.') else k): v
        for k, v in state_dict.items()
    }
    refiner = {
        k[len('context_refiner.'):]: v
        for k, v in stripped.items() if k.startswith('context_refiner.')
    }
    if refiner:
        return refiner
    if _REFINER_MARKER in stripped:
        # Already a bare refiner state dict.
        return stripped
    raise RuntimeError(
        'No context_refiner weights found. Expected either a refiner file (keys like '
        f'{_REFINER_MARKER!r}) or a checkpoint containing context_refiner.* entries.'
    )


def reset_parameters(root):
    """Apply each submodule's own default initialisation, recursively.

    nn.Linear and friends carry reset_parameters(); RMSNorm here does not, and its weight is a
    scale that defaults to one. Delegating rather than hand-rolling kaiming bounds keeps this
    identical to what PyTorch would have done in __init__.
    """
    for module in root.modules():
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif callable(getattr(module, 'reset_parameters', None)):
            module.reset_parameters()


class RefinerBlock(nn.Module):
    """Bidirectional self-attention + MLP.

    Unlike models.llm_adapter.TransformerBlock there is no cross-attention: the LLM hidden
    states are the sequence being refined, not a separate context to attend into. The
    attention mask is padding-only, never causal, which is what restores bidirectionality.
    """

    def __init__(self, model_dim, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.norm_attn = RMSNorm(model_dim)
        self.attn = Attention(
            query_dim=model_dim,
            context_dim=model_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )
        self.norm_mlp = RMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, attention_mask=None, position_embeddings=None):
        # Attention() treats context=None as self attention.
        normed = self.norm_attn(x)
        x = x + self.attn(
            normed,
            mask=attention_mask,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        reset_parameters(self)
        # Zero the output of both residual branches so the block starts as the identity. The
        # refiner then begins life as a plain linear projection of the LLM hidden states,
        # which is a much more stable starting point than random blocks when the DiT it feeds
        # is frozen. The weights still receive gradients, so they train normally.
        nn.init.zeros_(self.attn.o_proj.weight)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)


class ContextRefiner(nn.Module):
    """cap_embedder + bidirectional refiner blocks.

    cap_embedder follows the Lumina 2 recipe (submodules/Lumina_2/models/model.py): an RMSNorm
    *before* the linear, which is what absorbs the scale and distribution mismatch between the
    source LLM's hidden states and the space the DiT cross-attention expects.

    Args:
        cap_feat_dim: hidden size of the source LLM (e.g. 2048 for Qwen3.5-2B-Base).
        model_dim: dimension the DiT cross-attention consumes (crossattn_emb_channels, 1024).
        num_layers: number of refiner blocks. Higher than Lumina's default of 2 because here
            the DiT is typically frozen, so the refiner carries the whole distribution gap.
    """

    def __init__(self, cap_feat_dim, model_dim, num_layers=6, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.cap_feat_dim = cap_feat_dim
        self.model_dim = model_dim
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim),
            nn.Linear(cap_feat_dim, model_dim, bias=True),
        )
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [RefinerBlock(model_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.norm_out = RMSNorm(model_dim)

    def init_weights(self):
        """Initialise every parameter, without relying on __init__ having run.

        The pipeline builds this module under init_empty_weights(), so its parameters arrive
        on the meta device and are materialised with torch.empty -- whatever the allocator
        last held. This method is the only thing that runs afterwards, so it has to cover
        every parameter, not just the ones whose values differ from the PyTorch defaults.
        """
        reset_parameters(self.cap_embedder)
        nn.init.trunc_normal_(self.cap_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.cap_embedder[1].bias)
        nn.init.ones_(self.norm_out.weight)
        for block in self.blocks:
            block.init_weights()

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, L, cap_feat_dim) hidden states from the source LLM.
            attention_mask: (B, L) padding mask, 1 for real tokens. Padded positions are
                masked out of attention and zeroed in the output, matching what the DiT
                cross-attention expects.

        Returns:
            (B, L, model_dim) features for the DiT cross-attention.
        """
        sdpa_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
            sdpa_mask = attention_mask
            if sdpa_mask.ndim == 2:
                sdpa_mask = sdpa_mask.unsqueeze(1).unsqueeze(1)
            # An empty caption tokenizes to an all-padding row, and that is not a corner case:
            # it is exactly the unconditional embedding every CFG run needs. A row with no
            # unmasked key makes the attention softmax degenerate; the CPU math backend returns
            # zeros, but the fused CUDA backends can return NaN, and NaN * 0 is still NaN, so
            # zeroing the padded output afterwards would not contain it -- it would poison the
            # gradients instead. Let such a row attend freely; its output is discarded anyway.
            empty_rows = ~sdpa_mask.any(dim=-1, keepdim=True)
            sdpa_mask = sdpa_mask | empty_rows

        x = self.cap_embedder(hidden_states)

        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        for block in self.blocks:
            x = block(x, attention_mask=sdpa_mask, position_embeddings=position_embeddings)

        x = self.norm_out(x)

        if attention_mask is not None:
            # Out-of-place so this stays safe under activation checkpointing.
            x = x * attention_mask.reshape(attention_mask.shape[0], -1, 1).to(x.dtype)
        return x
