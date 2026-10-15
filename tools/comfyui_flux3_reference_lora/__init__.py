"""ComfyUI custom node: load a FLUX.3 LoRA trained with edit_isolation = 'reference_tokens'.

Such a LoRA (diffusion-pipe, models/flux3.py) trains the 28 shared blocks, but applies there
only to the rows of the reference-image tokens. Its weights for those blocks are saved under
'flux3_ref_rows.' instead of 'diffusion_model.', so ComfyUI's ordinary LoRA loader skips them
(and logs them as not loaded) rather than applying them to every token. This node applies
them the way they were trained:

  - the 'diffusion_model.' weights (the reference-only modules) load as an ordinary LoRA;
  - each shared block's linears add the LoRA's change only on the reference rows. Without a
    reference image there are no such rows, so text-to-image is left exactly as it was.

Install: copy or symlink this directory into ComfyUI/custom_nodes/. The node is
"Load FLUX.3 Reference LoRA" (loaders), taking the model and a file from models/loras.
"""

import logging

import torch
import torch.nn.functional as F

import comfy.ldm.flux.layers
import comfy.lora
import comfy.ops
import comfy.sd
import comfy.utils
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

REFERENCE_ROWS_PREFIX = 'flux3_ref_rows.'
LINEARS = ('q_proj', 'k_proj', 'v_proj', 'mlp_in', 'attn_out', 'mlp_out')


def split_reference_rows(sd):
    """(ordinary LoRA state dict, {block index: {linear name: {suffix: tensor}}})."""
    ordinary, rows = {}, {}
    for key, value in sd.items():
        if not key.startswith(REFERENCE_ROWS_PREFIX):
            ordinary[key] = value
            continue
        parts = key[len(REFERENCE_ROWS_PREFIX):].split('.')
        if len(parts) < 4 or parts[0] != 'single_blocks' or parts[2] not in LINEARS:
            raise ValueError(f'Unexpected reference-row LoRA key: {key}')
        rows.setdefault(int(parts[1]), {}).setdefault(parts[2], {})['.'.join(parts[3:])] = value
    return ordinary, rows


class RowDelta:
    """The weight change of one linear, computed by ComfyUI's own LoRA/LoKr code, so the
    scaling conventions are exactly those of the ordinary loader."""

    def __init__(self, name, tensors, out_features, in_features, strength):
        lora = {f'{name}.{suffix}': t for suffix, t in tensors.items()}
        patches = comfy.lora.load_lora(lora, {name: name}, log_missing=False)
        if name not in patches:
            raise ValueError(f'Could not read the LoRA weights of {name}: {sorted(tensors)}')
        self.name = name
        self.adapter = patches[name]
        self.shape = (out_features, in_features)
        self.strength = strength

    def weight(self, device):
        zero = torch.zeros(self.shape, dtype=torch.float32, device=device)
        return comfy.lora.calculate_weight([(self.strength, self.adapter, 1.0, None, None)], zero, self.name)


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def make_block_forward(block, deltas):
    """A forward for a ComfyUI FLUX.3 Block (comfy/ldm/flux3/model.py) that adds each linear's
    LoRA change on the reference rows only. deltas: {linear name: RowDelta}; on a fused-qkv
    model the q/k/v changes are stacked onto qkv."""

    def linear(name, module, x, rows):
        out = module(x)
        delta = deltas.get(name)
        if delta is None or rows is None:
            return out
        start, end = rows
        if isinstance(delta, list):
            weight = torch.cat([d.weight(x.device) if d is not None else
                                torch.zeros(block.qkv.weight.shape[0] // 3, x.shape[-1], device=x.device)
                                for d in delta])
        else:
            weight = delta.weight(x.device)
        out[:, start:end] += F.linear(x[:, start:end], weight.to(x.dtype))
        return out

    def forward(x, spans, pe, transformer_options={}):
        # Spans are [image, references, text] with references, [image, text] without.
        rows = spans[1][:2] if len(spans) == 3 else None
        B, N, _ = x.shape
        mod = spans[0][2]
        x_mod = comfy.ldm.flux.layers.modulated_norm(x, block.pre_norm, mod.scale, mod.shift)
        for start, end, mod in spans[1:]:
            x_mod[:, start:end] = comfy.ldm.flux.layers.modulated_norm(x[:, start:end], block.pre_norm, mod.scale, mod.shift)

        if block.fused_qkv:
            q, k, v = linear('qkv', block.qkv, x_mod, rows).chunk(3, dim=-1)
        else:
            q = linear('q_proj', block.q_proj, x_mod, rows)
            k = linear('k_proj', block.k_proj, x_mod, rows)
            v = linear('v_proj', block.v_proj, x_mod, rows)
        mlp = linear('mlp_in', block.mlp_in, x_mod, rows)
        del x_mod
        q, k = block.norm(q.view(B, N, block.num_heads, -1), k.view(B, N, block.num_heads, -1), pe)
        q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v)
        attn = optimized_attention(q, k, v, block.num_heads, transformer_options=transformer_options,
                                   preferred_attention=block.comfy_attention)
        out = linear('attn_out', block.attn_out, attn, rows) + linear('mlp_out', block.mlp_out, _swiglu(mlp), rows)
        del mlp
        for start, end, mod in spans:
            x[:, start:end].addcmul_(out[:, start:end], mod.gate)
        return x

    return forward


def apply_reference_lora(model, sd, strength):
    """Return a clone of the ModelPatcher `model` with the reference-token LoRA `sd` applied."""
    ordinary, rows = split_reference_rows(sd)
    if ordinary:
        model, _ = comfy.sd.load_lora_for_models(model, None, ordinary, strength, 0)
    else:
        model = model.clone()
    diffusion_model = model.model.diffusion_model
    if not hasattr(diffusion_model, 'single_blocks') or not hasattr(diffusion_model, 'content_mode_blocks'):
        raise ValueError('This LoRA is for FLUX.3; the model is not a FLUX.3 model')
    for index, linears in rows.items():
        block = diffusion_model.single_blocks[index]
        deltas = {}
        for name, tensors in linears.items():
            key = f'single_blocks.{index}.{name}'
            if block.fused_qkv and name in ('q_proj', 'k_proj', 'v_proj'):
                hidden = block.qkv.weight.shape[1]
                deltas.setdefault('qkv', [None, None, None])[('q_proj', 'k_proj', 'v_proj').index(name)] = (
                    RowDelta(key, tensors, hidden, hidden, strength))
                continue
            module = getattr(block, name)
            deltas[name] = RowDelta(key, tensors, module.weight.shape[0], module.weight.shape[1], strength)
        model.add_object_patch(f'diffusion_model.single_blocks.{index}.forward', make_block_forward(block, deltas))
    logging.info(f'FLUX.3 reference LoRA: {len(ordinary)} ordinary tensors, reference-row weights on '
                 f'{len(rows)} shared blocks')
    return model


class LoadFlux3ReferenceLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {
            'model': ('MODEL',),
            'lora_name': (folder_paths.get_filename_list('loras'),),
            'strength': ('FLOAT', {'default': 1.0, 'min': -100.0, 'max': 100.0, 'step': 0.01}),
        }}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'loaders'

    def load(self, model, lora_name, strength):
        import folder_paths
        if strength == 0:
            return (model,)
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise('loras', lora_name), safe_load=True)
        return (apply_reference_lora(model, sd, strength),)


NODE_CLASS_MAPPINGS = {'LoadFlux3ReferenceLoRA': LoadFlux3ReferenceLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {'LoadFlux3ReferenceLoRA': 'Load FLUX.3 Reference LoRA'}
