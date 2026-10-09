"""Fold an Iris LoRA or LoKr adapter trained with diffusion-pipe into the base weights.

The result is Iris's own exported layout -- model.safetensors with IrisDiT keys plus
config.yaml -- the same thing a full fine-tune saves, so iris3b.sampling.load_for_inference
and Iris's own sampling scripts read it unchanged.

Usage:
    python -m tools.iris_merge_adapter \\
        --model speridlabs/iris-3b \\
        --adapter /data/output/iris_lora/20260101_12-00-00/epoch10 \\
        --output /data/iris_merged \\
        --strength 1.0

--model is a local directory holding model.safetensors and config.yaml, or a Hugging Face
repo id. --adapter is a save directory written by diffusion-pipe (adapter_config.json plus
adapter_model.safetensors). The adapter type, rank and targets are read from the json.

Only iris3b, peft and safetensors are imported: nothing from diffusion-pipe's model stack,
so this runs on a machine without CUDA.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'submodules', 'iris-3b', 'src'))

import peft
import safetensors.torch

PREFIX_RE = re.compile(r'^(model\.diffusion_model|diffusion_model|transformer|net)\.')

# One factor of each adapter's delta, so scaling it scales the whole delta exactly once.
# LoRA: delta = B @ A. LoKr: delta = kron(w1, w2) with w1 either full or w1_a @ w1_b.
STRENGTH_FACTOR_RE = re.compile(r'\.(lora_B|lokr_w1|lokr_w1_a)(\.|$)')


def resolve_file(path_or_repo, filename, required=True):
    path = Path(path_or_repo)
    if path.is_dir():
        candidate = path / filename
        if candidate.exists():
            return candidate
        if required:
            raise FileNotFoundError(f'{candidate} does not exist')
        return None
    if path.exists():
        if filename.endswith('.safetensors'):
            return path
        candidate = path.parent / filename
        return candidate if candidate.exists() else None
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(str(path_or_repo), filename))
    except Exception:
        if required:
            raise
        return None


def load_base(model_path):
    """(IrisDiT with the checkpoint's weights in fp32, per-key original dtypes, raw config)."""
    from omegaconf import OmegaConf
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT

    config_file = resolve_file(model_path, 'config.yaml', required=False)
    raw = {} if config_file is None else OmegaConf.to_container(OmegaConf.load(str(config_file)))
    cfg = inference_config(raw)
    state_dict = safetensors.torch.load_file(str(resolve_file(model_path, 'model.safetensors')))
    state_dict = {PREFIX_RE.sub('', k): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('repa.')}
    dtypes = {k: v.dtype for k, v in state_dict.items()}
    model = IrisDiT(cfg.model)
    model.load_state_dict({k: v.float() for k, v in state_dict.items()}, strict=True)
    return model, dtypes, cfg, config_file


def apply_adapter(model, adapter_dir, strength=1.0):
    """Wrap `model` in the saved adapter and load its weights. Returns the PEFT model.

    The saver stored each trainable parameter under its own name with PEFT's '.default' and
    '.modules_to_save' segments removed and 'diffusion_model.' prepended, so the inverse is a
    lookup built from the wrapped model's parameter names.
    """
    adapter_dir = Path(adapter_dir)
    peft_config = peft.PeftConfig.from_pretrained(str(adapter_dir))
    peft_model = peft.get_peft_model(model, peft_config)
    by_saved_key = {
        name.replace('.default', '').replace('.modules_to_save', ''): name
        for name, _ in model.named_parameters()
    }
    adapter_sd = safetensors.torch.load_file(str(adapter_dir / 'adapter_model.safetensors'))
    renamed = {}
    for key, tensor in adapter_sd.items():
        key = PREFIX_RE.sub('', key)
        if key not in by_saved_key:
            raise RuntimeError(f'Adapter key {key} does not match any parameter of this Iris model')
        tensor = tensor.float()
        if strength != 1.0 and STRENGTH_FACTOR_RE.search(key):
            tensor = tensor * strength
        renamed[by_saved_key[key]] = tensor
    adapter_params = {name for name, _ in model.named_parameters()
                      if '.default' in name and any(s in name for s in ('lora_', 'lokr_'))}
    not_loaded = sorted(adapter_params - set(renamed))
    if not_loaded:
        raise RuntimeError(f'The adapter file has no weights for {len(not_loaded)} adapter parameters, '
                           f'e.g. {not_loaded[:3]}. Was it trained on a different model or config?')
    model.load_state_dict(renamed, strict=False)
    return peft_model


def merge(model_path, adapter_dir, strength=1.0):
    """(merged state dict in the base checkpoint's dtypes, raw config file or None, cfg)."""
    model, dtypes, cfg, config_file = load_base(model_path)
    merged = apply_adapter(model, adapter_dir, strength).merge_and_unload()
    state_dict = {k: v.detach().to(dtypes[k]).contiguous() for k, v in merged.state_dict().items()}
    return state_dict, config_file, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='speridlabs/iris-3b', help='Base checkpoint directory or HF repo id.')
    parser.add_argument('--adapter', required=True, help='diffusion-pipe adapter save directory.')
    parser.add_argument('--output', required=True, help='Directory to write model.safetensors and config.yaml to.')
    parser.add_argument('--strength', type=float, default=1.0, help='Adapter multiplier (default 1.0).')
    args = parser.parse_args()

    state_dict, config_file, cfg = merge(args.model, args.adapter, args.strength)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state_dict, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    if config_file is not None:
        shutil.copyfile(config_file, out / 'config.yaml')
    else:
        from omegaconf import OmegaConf
        from iris3b.config import INFERENCE_SECTIONS
        config = OmegaConf.to_container(OmegaConf.structured(cfg))
        OmegaConf.save(OmegaConf.create({k: config[k] for k in INFERENCE_SECTIONS}), str(out / 'config.yaml'))
    print(f'Wrote {len(state_dict)} tensors to {out / "model.safetensors"}')


if __name__ == '__main__':
    main()
