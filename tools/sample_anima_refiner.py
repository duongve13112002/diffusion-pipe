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
        --config examples/anima_refiner_refiner_only.toml \\
        --prompt '1girl, solo, blue eyes, looking at viewer' \\
        --steps 30 --cfg 5 --output out.png

The [model] table is read from the config; everything else in it (dataset, optimizer, learning
rates) is ignored.
"""

import argparse
import os
import sys

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
    return parser.parse_args()


def build_pipeline(config_path, device, dtype_override):
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
    pipeline.transformer.eval().requires_grad_(False).to(device)
    pipeline.text_encoder.eval().requires_grad_(False).to(device)
    pipeline.vae.model.to(device)
    pipeline.vae.mean = pipeline.vae.mean.to(device)
    pipeline.vae.std = pipeline.vae.std.to(device)
    pipeline.vae.scale = [pipeline.vae.mean, 1.0 / pipeline.vae.std]
    return pipeline, model_config


@torch.no_grad()
def encode_prompt(pipeline, prompts, device):
    batch_encoding = cosmos_predict2._tokenize(pipeline.tokenizer, prompts, pipeline.max_text_length)
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
def sample(pipeline, layers, embeds, mask, uncond, uncond_mask, args, device, dtype):
    latent_h = args.height // 8
    latent_w = args.width // 8
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    x = torch.randn(args.batch_size, 16, 1, latent_h, latent_w, generator=generator).to(device=device, dtype=torch.float32)

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
    pipeline, model_config = build_pipeline(args.config, device, args.dtype)
    dtype = model_config['dtype']
    if args.shift is None:
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
    latents = sample(pipeline, layers, embeds, mask, uncond, uncond_mask, args, device, dtype)
    images = decode(pipeline, latents, device)

    for i, image in enumerate(images):
        path = args.output if len(images) == 1 else f'{os.path.splitext(args.output)[0]}_{i}.png'
        Image.fromarray(image).save(path)
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
