#!/usr/bin/env python3
"""Cast the released Iris-3B checkpoint (float32 EMA) to bfloat16 before quantizing.

The exporters copy every tensor they do not quantize verbatim, so quantizing the float32
release directly would leave ~1.6B kept parameters (text side, w2, modulation cores, text
adapter, pixel head) in float32: a ~7.5 GB "FP8" file. Iris runs under bfloat16 autocast,
so a bfloat16 source loses nothing at inference; quantizing from it instead of float32
adds no measurable error (FP8/FP4 round far coarser than bfloat16).

    python scripts/external/iris_fp32_to_bf16.py iris-3b/model.safetensors iris-3b-bf16/model.safetensors

Copies ``config.yaml`` from beside the source. Tensors are read one at a time; peak host
memory is the bfloat16 output (~6 GB).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(source: Path, output: Path) -> dict[str, int]:
    if source.resolve() == output.resolve():
        raise SystemExit("output must differ from the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    counts = {"cast": 0, "kept": 0}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if tensor.dtype in (torch.float32, torch.float64, torch.float16):
                tensor = tensor.to(torch.bfloat16)
                counts["cast"] += 1
            else:
                counts["kept"] += 1
            tensors[name] = tensor.contiguous()
    save_file(tensors, str(output), metadata=metadata)
    config = source.parent / "config.yaml"
    if config.is_file() and not (output.parent / "config.yaml").exists():
        shutil.copy2(config, output.parent / "config.yaml")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Iris model.safetensors (float32)")
    parser.add_argument("output", type=Path, help="bfloat16 model.safetensors to write")
    args = parser.parse_args(argv)
    counts = convert(args.source, args.output)
    print(f"wrote {args.output}: {counts['cast']} tensors cast to bfloat16, {counts['kept']} copied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
