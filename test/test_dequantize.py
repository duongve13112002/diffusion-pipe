"""ComfyPipeline.dequantize: which weights diffusion_model_dtype leaves in high precision."""

import torch
from torch import nn

from models.base import ComfyPipeline


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_in = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.txt_in = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.ModuleDict({'attn': nn.Linear(4, 4)}) for _ in range(2)])
        self.final_layer = nn.ModuleDict({'linear': nn.Linear(4, 4)})


def test_keywords_match_the_full_path():
    """A keyword names a module whose weights sit in its children (time_in.0.weight); matched
    against the module's own name ('0.weight') it kept nothing but direct children."""
    pipe = object.__new__(ComfyPipeline)
    pipe.keep_in_high_precision = ['time_in', 'txt_in', 'final_layer']
    model = Toy()
    pipe.dequantize(model, torch.float16)
    for name, p in model.named_parameters():
        high = p.ndim == 1 or name.startswith(('time_in.', 'txt_in.', 'final_layer.'))
        assert p.dtype == (torch.float32 if high else torch.float16), name
    assert model.blocks[0]['attn'].weight.dtype == torch.float16
