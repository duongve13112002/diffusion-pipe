# Iris-3B for comfy-quants

Adds Iris-3B to [Comfy-Org/comfy-quants](https://github.com/Comfy-Org/comfy-quants), so an
Iris checkpoint (the release, or a fine-tune / EMA saved by diffusion-pipe) can be exported in
the same formats as the stock models: FP8 E4M3, FP8 E5M2, MXFP8, NVFP4 and INT8 tensorwise.

| File | What |
| --- | --- |
| `iris.patch` | the whole change against comfy-quants `3d057e9` (adapter, configs, docs, tests) |
| `configs/iris_3b_*.yaml` | the five export configs (also inside the patch) |
| `iris_fp32_to_bf16.py` | casts the float32 release to bfloat16 before exporting (also inside the patch) |
| `validate_e2e.py` | CPU end-to-end check: export with every config, load through ComfyUI-Iris, compare to bf16 |

## Apply

```bash
git clone https://github.com/Comfy-Org/comfy-quants && cd comfy-quants
git checkout 3d057e9            # or a later commit: `git apply --3way` if it no longer applies cleanly
git apply /path/to/diffusion-pipe/tools/comfy_quants_iris/iris.patch
pip install -e .
python -m pytest tests/unit/test_iris_contract.py -q
```

The full guide is `docs/quantization/iris.md` in the patched tree. In short:

```bash
python scripts/external/iris_fp32_to_bf16.py iris-3b/model.safetensors iris-3b-bf16/model.safetensors
comfy-quants export-model --config configs/iris_3b_fp8.yaml \
  --source iris-3b-bf16/model.safetensors --out iris-3b_fp8.safetensors --device cuda:0 --json
```

Other formats use `export-model-mxfp8`, `export-model-nvfp4` or `export-model-int8-tensorwise`
with the matching config.

## Notes

- Stock ComfyUI has no Iris. The files load through
  [envy-ai/ComfyUI-Iris](https://github.com/envy-ai/ComfyUI-Iris), which builds the model from
  `comfy.ops` and detects the per-layer `comfy_quant` markers. Loaders that wrap the upstream
  code with plain `nn.Linear` can't read them.
- 168 Linears are quantized: the image side of the 8 dual blocks and all 16 single blocks
  (q/k/v, out proj, gate, MLP w1/w3). The text side, every `w2` (in-features 6826 is not
  block-aligned), the modulation cores, the text adapter, embedders and pixel head stay bf16.
- Expected size: ~4.6 GB for FP8 / MXFP8 / INT8 and ~4.0 GB for NVFP4, vs 6.0 GB bf16.
