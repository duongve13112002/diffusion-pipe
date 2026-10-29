"""flash-attn 2's padding helpers (pure torch there too)."""

import torch
import torch.nn.functional as F


def index_first_axis(x, indices):
    return x[indices]


def unpad_input(hidden_states, attention_mask):
    lengths = attention_mask.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(lengths, 0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return flat[indices], indices, cu_seqlens, int(lengths.max()), lengths


def pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    out = out.index_copy(0, indices, hidden_states)
    return out.view(batch, seqlen, *hidden_states.shape[1:])
