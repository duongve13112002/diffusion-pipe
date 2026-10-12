"""End to end on CPU: tiny Iris (envy-ai ComfyUI-Iris model code) -> comfy-quants export with
each Iris config -> load through the node's own load_model (detect_layer_quantization, strict)
-> forward vs the bf16 source. Same key names and structure as Iris-3B, smaller widths, w2 kept
unaligned (682) like the real 6826.

    python validate_e2e.py WORK_DIR --comfyui PATH --custom-nodes PATH --comfy-quants PATH

--comfyui: a recent ComfyUI checkout (with comfy_kitchen installed); --custom-nodes: a folder
holding envy-ai's node as ComfyUI_Iris/ (its __init__.py may be emptied to skip node
registration); --comfy-quants: comfy-quants with iris.patch applied.
"""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('work', type=Path)
parser.add_argument('--comfyui', type=Path, required=True)
parser.add_argument('--custom-nodes', type=Path, required=True)
parser.add_argument('--comfy-quants', type=Path, required=True)
cli = parser.parse_args()
sys.path[:0] = [str(cli.comfyui), str(cli.custom_nodes), str(cli.comfy_quants / 'src')]

from comfy.cli_args import args
args.cpu = True
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import comfy.ops

from ComfyUI_Iris import integration
from ComfyUI_Iris.model import IrisConfig, IrisDiT

from comfy_quants.cli.main import main as cq_main
from comfy_quants.model_adapters import iris as cq_iris

TINY = dict(hidden_size=256, depth=3, dual_depth=1, num_heads=2, num_kv_heads=1, text_dim=256, text_len=8,
            text_lap_num_heads=8)
cq_iris._HIDDEN, cq_iris._NUM_KV_HEADS, cq_iris._MLP = 256, 1, int(2 * 4 * 256 / 3)
cq_iris._DUAL_BLOCKS, cq_iris._SINGLE_BLOCKS = 1, 2
integration.IrisConfig = lambda: IrisConfig(**TINY)

work = cli.work
work.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
model = IrisDiT(IrisConfig(**TINY), dtype=torch.float32, operations=comfy.ops.manual_cast)
sd = {}
for name, tensor in model.state_dict().items():
    sd[name] = (torch.randn_like(tensor) * (0.02 if tensor.ndim > 1 else 0.1)).to(torch.bfloat16)
src = work / 'model.safetensors'
save_file(sd, str(src))
print('source tensors', len(sd), 'w2 shape', sd['blocks.0.mlp_x.w2.weight'].shape)

# Every 2D block weight the checkpoint has must be declared by the contract.
from comfy_quants.model_adapters.base import ModelSource
_, graph = cq_iris.IrisAdapter().inspect(ModelSource(family='iris', model_id='x'))
declared = {t.name: t.shape for m in graph.modules for t in m.tensors}
undeclared = sorted(n for n, t in sd.items() if n.startswith('blocks.') and n.endswith('.weight') and t.ndim == 2
                    and n not in declared)
mismatched = {n: (s, list(sd[n].shape)) for n, s in declared.items() if list(sd[n].shape) != s}
assert not undeclared and not mismatched, (undeclared, mismatched)

configs = cli.comfy_quants / 'configs'
commands = {'fp8': 'export-model', 'fp8_e5m2': 'export-model', 'mxfp8': 'export-model-mxfp8',
            'nvfp4': 'export-model-nvfp4', 'int8_tensorwise': 'export-model-int8-tensorwise'}

x = torch.randn(1, 3, 64, 48)
timestep = torch.tensor([0.6])
states = torch.randn(1, 8, 12, 256)
mask = torch.ones(1, 8, dtype=torch.bool)


def run(path):
    patcher = integration.load_model(str(path), 'bfloat16')
    dm = patcher.model.diffusion_model
    with torch.no_grad():
        context = dm.preprocess_text(states.to(torch.bfloat16), mask)
        return dm(x.to(torch.bfloat16), timestep, context).float(), patcher


reference, _ = run(src)
results = {}
for key, command in commands.items():
    out = work / f'iris_{key}.safetensors'
    code = cq_main([command, '--config', str(configs / f'iris_3b_{key}.yaml'), '--source', str(src), '--out', str(out),
                    '--device', 'cpu', '--no-progress', '--json'])
    assert code == 0
    with safe_open(str(out), 'pt') as f:
        keys = list(f.keys())
        markers = sorted(k for k in keys if k.endswith('.comfy_quant'))
        w2_quantized = [k for k in markers if '.w2.' in k]
        text_quantized = [k for k in markers if '_y.' in k]
        kept_dtypes = {str(f.get_tensor(k).dtype) for k in keys if k.endswith('mlp_x.w2.weight')}
    output, patcher = run(out)
    rel = ((output - reference).norm() / reference.norm()).item()
    layer = patcher.model.diffusion_model.blocks[1].q_proj
    probe = torch.randn(4, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        exact = probe.float() @ sd['blocks.1.q_proj.weight'].float().t()
        got = layer(probe).float()
    layer_rel = ((got - exact).norm() / exact.norm()).item()
    results[key] = dict(quantized_layers=len(markers), w2_quantized=len(w2_quantized),
                        text_side_quantized=len(text_quantized), w2_dtype=sorted(kept_dtypes),
                        q_proj_weight=type(layer.weight).__name__ + ':' + str(getattr(layer.weight, '_layout_cls', '')),
                        rel_error_vs_bf16=round(rel, 4), q_proj_rel_error=round(layer_rel, 4), finite=bool(torch.isfinite(output).all()))
    print(key, json.dumps(results[key]))

expected_layers = 1 * 7 + 2 * 7
for key, r in results.items():
    assert r['quantized_layers'] == expected_layers, (key, r)
    assert r['w2_quantized'] == 0 and r['text_side_quantized'] == 0 and r['finite'], (key, r)
print('ALL OK')
