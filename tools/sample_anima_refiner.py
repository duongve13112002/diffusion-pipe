"""Generate images from an anima_refiner checkpoint.

diffusion-pipe is a training repo, and ComfyUI's `class Anima(MiniTrainDIT)` has no
context_refiner, so a model trained with this architecture cannot be sampled anywhere else
yet. This script fills that gap.

It deliberately loads the model through CosmosPredict2Pipeline -- the same class, the same
`[model]` config table and the same checkpoint-resolution rules training uses -- and runs the
same layer stack `to_layers()` builds. Anything that loads for training loads here identically,
and a discrepancy between the two would be a bug in one shared place rather than in a
sampling-only copy of the loading logic.

Usage:
    python -m tools.sample_anima_refiner \\
        --config examples/anima_refiner/refiner_only.toml \\
        --prompt '1girl, solo, blue eyes, looking at viewer' \\
        --steps 30 --cfg 5 --output out.png

Adapters are merged in before sampling, one --lora per save directory and an optional
--lora-strength each:

    python -m tools.sample_anima_refiner \\
        --config examples/anima_refiner/lora.toml \\
        --lora /data/output/anima_refiner_lora/20250101_12-00-00/epoch10 \\
        --lora-strength 0.8 \\
        --prompt '1girl, solo, blue eyes' --output out.png

LoRA, LoKr and OPLoRA all land here: OPLoRA constrains a LoRA while it trains and writes an
ordinary LoRA file, so it needs no separate flag.

The [model] table is read from the config; everything else in it (dataset, optimizer, learning
rates) is ignored. The [adapter] table is ignored too -- the adapter's own shape comes from the
adapter_config.json in its save directory, so a sample never depends on the config still
matching the run that produced it.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import toml
import torch
from PIL import Image

from utils.common import DTYPE_MAP

# models.cosmos_predict2 is imported lazily, inside build_pipeline, and that is load-bearing.
# Every pipeline layer is decorated `@torch.autocast('cuda', dtype=AUTOCAST_DTYPE)`, and the
# decorator captures utils.common.AUTOCAST_DTYPE at import time. train.py sets that global
# before it imports the model module; importing at the top here would capture the default of
# None, which torch.autocast resolves to float16 -- silently running a bfloat16 model under
# fp16 autocast on CUDA, where the 65504 ceiling makes overflow to inf/NaN a live risk. A
# CPU test suite cannot catch it, because autocast('cuda') is inert there.
cosmos_predict2 = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Training TOML; only its [model] table is used.')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--negative-prompt', default='', help='Only used when --cfg > 1.')
    parser.add_argument('--output', default='sample.png')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg', type=float, default=1.0, help='1.0 disables classifier-free guidance.')
    parser.add_argument('--shift', type=float, default=None,
                        help='Timestep shift. Defaults to the config, then 1.0 (no shift).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default=None, help='Overrides the config dtype.')
    parser.add_argument('--lora', action='append', metavar='DIR',
                        help='Save directory of a trained adapter, holding adapter_config.json '
                             'and adapter_model.safetensors. Repeat to stack several, merged in '
                             'the order given. LoRA, LoKr and OPLoRA runs are all read from '
                             'here: OPLoRA constrains a LoRA while it trains and writes an '
                             'ordinary LoRA, so it needs no flag of its own.')
    parser.add_argument('--lora-strength', action='append', type=float, metavar='S',
                        help='Multiplier for the --lora in the same position. Pass one per '
                             '--lora, or none at all for 1.0 each. 0 disables an adapter, above '
                             '1 overdrives it.')
    return parser.parse_args()


def resolve_adapters(lora_dirs, strengths):
    """Pair each --lora with its --lora-strength, or refuse to guess.

    Silently recycling one strength across several adapters, or padding a short list with 1.0,
    would apply a weight the user did not write. Both spellings a caller is likely to mean are
    accepted -- no strengths at all, or exactly one per adapter -- and everything else says so.
    """
    lora_dirs = lora_dirs or []
    strengths = strengths or []
    if not lora_dirs:
        if strengths:
            raise SystemExit('--lora-strength was given with no --lora for it to apply to.')
        return []
    if not strengths:
        strengths = [1.0] * len(lora_dirs)
    elif len(strengths) != len(lora_dirs):
        raise SystemExit(
            f'{len(lora_dirs)} --lora but {len(strengths)} --lora-strength. Pass one strength '
            'per adapter, in the same order, or none at all to use 1.0 for every one.'
        )

    adapters = []
    refiner_dirs = []
    for path, strength in zip(lora_dirs, strengths):
        path = Path(path)
        if not path.is_dir():
            raise SystemExit(
                f'--lora {path} is not a directory. Point it at a run\'s save directory, the '
                'one holding adapter_config.json and adapter_model.safetensors.'
            )
        if (path / 'context_refiner' / 'context_refiner.safetensors').exists():
            refiner_dirs.append(path)
        adapters.append((path, strength))

    if len(refiner_dirs) > 1:
        # A densely trained refiner is a whole replacement, not an increment, so stacking two
        # means the last one silently wins and the earlier adapter is left reading a frontend
        # it was never trained against.
        raise SystemExit(
            'More than one --lora carries a densely trained context_refiner:\n'
            + '\n'.join(f'    {p}' for p in refiner_dirs)
            + '\n  These replace the refiner outright rather than adding to it, so only one can '
              'apply. Pass just that one, or point context_refiner_path in the config at the '
              'refiner you want and use adapters that do not carry their own.'
        )
    return adapters


def apply_adapters(pipeline, adapters):
    """Inject, load, scale and merge each adapter into the base weights.

    Everything goes through the pipeline's own loaders rather than a sampling-only copy: peft
    rebuilds the exact target module list from the adapter_config.json the run wrote, and
    load_adapter_weights is the same method init_from_existing uses during training, including
    the part that restores a densely trained context_refiner from the run's context_refiner/
    subdirectory.

    Merging afterwards leaves plain nn.Linear behind, so to_layers() sees the model it would
    have seen with no adapter at all, and a second adapter can be injected on top of the first.
    """
    # peft reaches this process through models.base either way; importing here keeps the
    # careful import ordering at the top of this file untouched.
    import peft
    from peft.tuners.lora import LoraLayer
    from peft.tuners.lycoris_utils import LycorisLayer

    for path, strength in adapters:
        # PeftConfig dispatches on the peft_type in adapter_config.json, so one call covers
        # LoRA and LoKr, and the target modules are the run's own rather than re-derived here.
        peft_config = peft.PeftConfig.from_pretrained(str(path))
        peft_model = peft.get_peft_model(pipeline.transformer, peft_config)
        pipeline.load_adapter_weights(path)
        if strength != 1.0:
            # set_scale multiplies the delta the merge applies. Measured on peft 0.19.1: a
            # scale of 0.5 halves the merged delta for LoRA and for LoKr alike, so one code
            # path covers both instead of reaching into the factors by name.
            for module in pipeline.transformer.modules():
                if isinstance(module, (LoraLayer, LycorisLayer)):
                    module.set_scale('default', strength)
        peft_model.merge_and_unload()
        # .value, because PeftType is an enum and str() on it prints 'PeftType.LOKR'.
        print(f'Merged {peft_config.peft_type.value} adapter {path} at strength {strength}')


def build_pipeline(config_path, device, dtype_override, adapters=()):
    """Load through the training pipeline so the loading rules are shared, not duplicated."""
    global cosmos_predict2
    config = toml.load(config_path)
    model_config = dict(config['model'])  # never mutate the caller's parsed config
    if model_config.get('type') != 'anima_refiner':
        raise RuntimeError(f"Expected type = 'anima_refiner' in {config_path}, got {model_config.get('type')!r}")
    if dtype_override:
        model_config['dtype'] = dtype_override
    if isinstance(model_config['dtype'], str):
        model_config['dtype'] = DTYPE_MAP[model_config['dtype']]
    # transformer_dtype is a training memory optimisation; sampling wants the full precision.
    model_config.pop('transformer_dtype', None)
    # Text embeddings are computed here, never read from a training cache.
    model_config['cache_text_embeddings'] = False

    # Must happen before models.cosmos_predict2 is imported: see the note at the top.
    import utils.common
    utils.common.AUTOCAST_DTYPE = model_config['dtype']
    from models import cosmos_predict2 as _cosmos_predict2
    cosmos_predict2 = _cosmos_predict2

    pipeline = cosmos_predict2.CosmosPredict2Pipeline({'model': model_config})
    pipeline.load_diffusion_model()

    if adapters:
        if getattr(pipeline, '_refiner_is_fresh', False) and not any(
                (path / 'context_refiner' / 'context_refiner.safetensors').exists()
                for path, _ in adapters):
            # The same combination configure_adapter refuses during training, for the same
            # reason: the base refiner is random, was never saved, and came from the ambient
            # RNG stream, so it is not the one the adapter was trained against and cannot be.
            raise SystemExit(
                'The refiner in this config is freshly initialised and random, so an adapter '
                'trained on a real refiner has nothing to attach to and the samples would be '
                'noise.\n'
                '  Point transformer_path or context_refiner_path at a trained refiner, or '
                'pass a --lora from a run that set train_context_refiner, which carries its '
                'refiner in a context_refiner/ subdirectory.'
            )
        apply_adapters(pipeline, adapters)

    pipeline.transformer.eval().requires_grad_(False).to(device)
    pipeline.text_encoder.eval().requires_grad_(False).to(device)
    pipeline.vae.model.to(device)
    pipeline.vae.mean = pipeline.vae.mean.to(device)
    pipeline.vae.std = pipeline.vae.std.to(device)
    pipeline.vae.scale = [pipeline.vae.mean, 1.0 / pipeline.vae.std]
    return pipeline, model_config


@torch.no_grad()
def encode_prompt(pipeline, prompts, device):
    # keep_one_real_token: the negative prompt defaults to '', and an all-padding row would
    # hand the frozen DiT an all-zero context it was never trained on.
    batch_encoding = cosmos_predict2._tokenize(
        pipeline.tokenizer, prompts, pipeline.max_text_length, keep_one_real_token=True)
    embeds = cosmos_predict2._compute_text_embeddings(
        pipeline.text_encoder,
        batch_encoding.input_ids,
        batch_encoding.attention_mask,
        hidden_layer=pipeline.llm_hidden_layer,
    )
    return embeds.to(device), batch_encoding.attention_mask.to(device)


def shifted_timesteps(steps, shift):
    """Sampling schedule from t=1 (pure noise) to t=0, with the training-time shift applied.

    prepare_inputs() draws t, applies `t = (t*shift) / (1 + (shift-1)*t)`, and builds
    `noisy = (1-t)*latents + t*noise` with target `noise - latents`. So the model predicts the
    velocity from clean towards noise, and sampling integrates it backwards.
    """
    t = torch.linspace(1.0, 0.0, steps + 1)
    if shift and shift != 1.0:
        t = (t * shift) / (1 + (shift - 1) * t)
    return t


@torch.no_grad()
def sample(pipeline, layers, embeds, mask, uncond, uncond_mask, args, device, dtype,
           in_channels):
    latent_h = args.height // 8
    latent_w = args.width // 8
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    # in_channels is passed in rather than assumed: get_dit_config derives it from the
    # checkpoint's x_embedder, so a checkpoint that is not 16-channel would otherwise fail with
    # a shape error deep inside the first block. The rollout in tools/distill_refiner.py already
    # takes it from the config this way.
    x = torch.randn(args.batch_size, in_channels, 1, latent_h, latent_w, generator=generator).to(device=device, dtype=torch.float32)

    timesteps = shifted_timesteps(args.steps, args.shift)

    def velocity(latents, t_value, text, text_mask):
        # t must share the model dtype. On CUDA the layers' autocast hides a mismatch; on
        # CPU autocast('cuda') is inert and the first matmul raises.
        t = torch.full((latents.shape[0], 1), t_value, device=device, dtype=dtype)
        inputs = (latents.to(dtype), t, text.to(dtype), text_mask)
        for layer in layers:
            inputs = layer(inputs)
        return inputs.float()

    for i in range(args.steps):
        t_now, t_next = timesteps[i].item(), timesteps[i + 1].item()
        v = velocity(x, t_now, embeds, mask)
        if args.cfg > 1.0:
            v_uncond = velocity(x, t_now, uncond, uncond_mask)
            v = v_uncond + args.cfg * (v - v_uncond)
        # x moves from noise towards clean, so step along -v by the size of the t decrement.
        x = x + (t_next - t_now) * v

    return x


@torch.no_grad()
def decode(pipeline, latents, device):
    images = pipeline.vae.model.decode(latents.to(next(pipeline.vae.model.parameters()).dtype), pipeline.vae.scale)
    images = images.float().squeeze(2)          # (B, C, 1, H, W) -> (B, C, H, W)
    images = ((images + 1) / 2).clamp(0, 1)     # encode did tensor*2 - 1
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype('uint8')


def main():
    args = parse_args()
    device = torch.device(args.device)
    adapters = resolve_adapters(args.lora, args.lora_strength)
    pipeline, model_config = build_pipeline(args.config, device, args.dtype, adapters)
    dtype = model_config['dtype']
    if args.shift is None:
        # flux_shift is the other schedule prepare_inputs supports. Reading only `shift` meant a
        # model trained with flux_shift was sampled on an unshifted schedule, silently.
        if model_config.get('flux_shift', False):
            raise SystemExit(
                'This config sets flux_shift, whose schedule depends on the image resolution '
                'and is not implemented in this sampler. Pass --shift explicitly to sample on a '
                'fixed shift, and be aware it will not match training exactly.'
            )
        args.shift = model_config.get('shift', 1.0)

    print(f'Encoding prompt through {pipeline.cap_feat_dim}-dim text encoder '
          f'(hidden layer {pipeline.llm_hidden_layer}, {pipeline.max_text_length} tokens)')
    prompts = [args.prompt] * args.batch_size
    embeds, mask = encode_prompt(pipeline, prompts, device)
    uncond, uncond_mask = (None, None)
    if args.cfg > 1.0:
        uncond, uncond_mask = encode_prompt(pipeline, [args.negative_prompt] * args.batch_size, device)

    layers = pipeline.to_layers()
    for layer in layers:
        layer.to(device).eval()

    print(f'Sampling {args.steps} steps at {args.width}x{args.height}, cfg={args.cfg}, shift={args.shift}')
    latents = sample(pipeline, layers, embeds, mask, uncond, uncond_mask, args, device, dtype,
                     in_channels=pipeline.transformer.in_channels)
    images = decode(pipeline, latents, device)

    for i, image in enumerate(images):
        path = args.output if len(images) == 1 else f'{os.path.splitext(args.output)[0]}_{i}.png'
        Image.fromarray(image).save(path)
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
