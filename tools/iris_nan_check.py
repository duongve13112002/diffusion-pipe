"""Find where an Iris training run's NaN comes from, on one GPU, with the run's own config.

    python tools/iris_nan_check.py --config iris_3b_full.toml
    python tools/iris_nan_check.py --config iris_3b_full.toml --batch-size 8 --steps 10

Four stages, each stopping at the first non-finite value it meets and saying where it is:
  1. checkpoint  every tensor in transformer_path's model.safetensors is finite;
  2. inputs      real images and captions from the dataset config: pixels and text embeddings;
  3. forward     the training layer stack, layer by layer (max |value| of each output), then
                 the loss and the backward pass (which parameters get a non-finite gradient);
  4. optimizer   --steps N optimizer steps on that same batch with the config's optimizer,
                 learning rate (no warmup: the full lr, the harshest case) and gradient
                 clipping, checking the loss and every weight after each step.
A NaN at stage 3 is the model or its inputs; one that first appears at stage 4 is the update.
Nothing is written. Needs the GPU memory of a training process at --batch-size (default 4).
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'submodules' / 'iris-3b' / 'src'))

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def say(message):
    print(message, flush=True)


def finite_report(name, tensor):
    """None when finite, else a one-line description."""
    import torch
    if not torch.is_floating_point(tensor):
        return None
    bad = ~torch.isfinite(tensor)
    if not bad.any():
        return None
    nan = int(torch.isnan(tensor).sum())
    return (f'{name}: {int(bad.sum())} of {tensor.numel()} values non-finite ({nan} NaN, '
            f'{int(bad.sum()) - nan} inf), shape {tuple(tensor.shape)}, dtype {tensor.dtype}')


def max_abs(tensor):
    import torch
    return float(tensor.detach().float().abs().max()) if torch.is_floating_point(tensor) and tensor.numel() else 0.0


def check_checkpoint(path):
    from safetensors import safe_open
    from models.iris import _resolve_file
    weights = _resolve_file(path, 'model.safetensors')
    say(f'[1/4] checkpoint {weights}')
    bad, count, dtypes = [], 0, {}
    with safe_open(str(weights), framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            count += 1
            dtypes[str(tensor.dtype)] = dtypes.get(str(tensor.dtype), 0) + 1
            if (report := finite_report(key, tensor.float() if tensor.is_floating_point() else tensor)):
                bad.append(report)
    say(f'      {count} tensors, dtypes {dtypes}')
    for line in bad[:10]:
        say('      NON-FINITE ' + line)
    return not bad


def dataset_samples(dataset_file, count, seed):
    """(image path, caption) pairs from the dataset config's directories."""
    import toml
    config = toml.load(dataset_file)
    found = []
    for directory in config.get('directory', []):
        root = Path(directory['path'])
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if root.is_dir() else []
        random.Random(seed).shuffle(files)
        for image in files[:count]:
            caption = image.with_suffix('.txt')
            found.append((image, caption.read_text(encoding='utf-8').strip() if caption.exists() else ''))
        if len(found) >= count:
            break
    return found[:count], config


def load_pixels(paths, resolution, multiple):
    """uint8 [B, 3, H, W] at the first image's aspect ratio and the dataset's resolution, like a
    size bucket: every image is center-cropped to that ratio and resized."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(paths[0]) as first:
        ratio = first.width / first.height
    width = max(multiple, round(math.sqrt(resolution * resolution * ratio) / multiple) * multiple)
    height = max(multiple, round(resolution * resolution / width / multiple) * multiple)
    batch = []
    for path in paths:
        with Image.open(path) as image:
            image = ImageOps.fit(image.convert('RGB'), (width, height), Image.Resampling.LANCZOS)
            batch.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    return torch.stack(batch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='The training config (.toml).')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=5, help='Optimizer steps for stage 4 (0 skips it).')
    parser.add_argument('--activation-checkpointing', action='store_true',
                        help='Checkpoint every layer after the text encoder, as activation_checkpointing = true '
                             "does in training: needed for the backward pass at the run's micro batch size.")
    parser.add_argument('--forward-only', action='store_true',
                        help="Forward pass only, without autograd: what the run's micro_batch_size_per_gpu "
                             'fits in memory without activation checkpointing. Skips backward and stage 4.')
    parser.add_argument('--resolution', type=int, default=None,
                        help="Side of the square-equivalent size (default: the dataset's first resolution).")
    parser.add_argument('--quantile', type=float, default=None,
                        help='Fix the timestep at this quantile of the sampler (default: sampled).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip-checkpoint-scan', action='store_true')
    args = parser.parse_args(argv)

    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(29500 + random.randint(0, 999)))
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')

    import toml
    import torch
    import deepspeed
    from utils.common import DTYPE_MAP
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl' if device.type == 'cuda' else 'gloo')

    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))
    model_config = config['model']
    if model_config.get('type') != 'iris':
        raise SystemExit(f"model.type is {model_config.get('type')!r}, not 'iris'")
    say(f"dtype {model_config['dtype']}, transformer_dtype {model_config.get('transformer_dtype', '(same)')}, "
        f"attn_backend {model_config.get('attn_backend', '(checkpoint default)')}, "
        f"optimizer {config['optimizer']}")
    model_config['dtype'] = DTYPE_MAP[model_config['dtype']]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]
    config.setdefault('reentrant_activation_checkpointing', False)

    from utils import common
    common.AUTOCAST_DTYPE = model_config['dtype']
    import models.iris as iris

    if not args.skip_checkpoint_scan and not check_checkpoint(model_config.get('transformer_path', iris.DEFAULT_REPO_ID)):
        say('=> The checkpoint itself holds NaN/inf. Re-download or re-export it.')
        return 1

    say('[2/4] inputs')
    samples, dataset_config = dataset_samples(config['dataset'], args.batch_size, args.seed)
    if not samples:
        raise SystemExit(f"no images found in the directories of {config['dataset']}")
    resolution = args.resolution or int(dataset_config.get('resolutions', [512])[0])
    pipe = iris.IrisPipeline(config)
    pixels = load_pixels([p for p, _ in samples], resolution, pipe.patch_size)
    captions = [c for _, c in samples]
    say(f'      {len(samples)} images at {pixels.shape[-1]}x{pixels.shape[-2]}, e.g. {samples[0][0]}')
    for (path, caption) in samples:
        say(f'      {path.name}: {len(caption)} chars{" (EMPTY caption)" if not caption else ""}')

    pipe.load_diffusion_model()
    layers = pipe.to_layers()
    torch.nn.ModuleList(layers).to(device)
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)

    torch.manual_seed(args.seed)
    batch = {'latents': pixels, 'mask': None, 'caption': captions}
    # Exactly what the training dataloader hands deepspeed (None becomes an empty tensor).
    from utils.dataset import split_batch
    (features, label), = split_batch(pipe.prepare_inputs(batch, timestep_quantile=args.quantile), 1)
    features = tuple(x.to(device) for x in features)
    target, mask = (x.to(device) for x in label)
    say(f'      timesteps (model units): {features[1].tolist()}')
    if pipe.text_encoder is not None and not pipe.cache_text_embeddings:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            y, y_mask = pipe.text_encoder(features[2], features[3])
        say(f'      text embeddings {tuple(y.shape)} {y.dtype}, max |y| {max_abs(y):.4g}, '
            f'tokens per caption {y_mask.sum(1).tolist()}')
        if (report := finite_report('text embeddings', y)):
            say('      NON-FINITE ' + report)
            say('=> The text encoder produces NaN/inf. Check text_encoder_path / text_encoder_dtype '
                '(bfloat16) / text_encoder_attn_implementation.')
            return 1

    say('[3/4] forward / backward, layer by layer')
    loss_fn = pipe.get_loss_fn()
    for layer in layers:
        layer.train()

    def run_layer(i, layer, out):
        if args.activation_checkpointing and i > 0 and torch.is_grad_enabled():
            # Training's checkpoint_func for activation_checkpointing = true; the text encoder
            # layer is never checkpointed there either.
            return torch.utils.checkpoint.checkpoint(layer, out, use_reentrant=False)
        return layer(out)

    def forward():
        out = features
        for i, layer in enumerate(layers):
            out = run_layer(i, layer, out)
            outputs = out if isinstance(out, tuple) else (out,)
            peak = max((max_abs(t) for t in outputs if torch.is_tensor(t)), default=0.0)
            say(f'      {i:3d} {type(layer).__name__:<18} max |out| {peak:.4g}')
            for j, t in enumerate(outputs):
                if torch.is_tensor(t) and (report := finite_report(f'layer {i} {type(layer).__name__} output {j}', t)):
                    if t.ndim and t.shape[0] == len(samples):
                        rows = [k for k in range(len(samples)) if not torch.isfinite(t[k]).all()]
                        report += f'; images {[samples[k][0].name for k in rows[:8]]}'
                    return None, report
        return out, None

    if args.forward_only:
        with torch.no_grad():
            out, report = forward()
    else:
        out, report = forward()
    if report:
        say('      NON-FINITE ' + report)
        say('=> The forward pass produces NaN/inf at the layer above, with finite weights and inputs.')
        return 1
    loss = loss_fn(out, (target, mask))
    say(f'      loss {loss.item():.6f}')
    if not torch.isfinite(loss):
        say('=> The loss is NaN/inf although every layer output is finite.')
        return 1
    if args.forward_only:
        say(f'=> The forward pass and the loss are finite at batch size {len(samples)}.')
        return 0
    loss.backward()
    params = [(n, p) for layer in layers for n, p in layer.named_parameters() if p.requires_grad]
    bad = [f'{getattr(p, "original_name", n)}' for n, p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        # Backward runs from the last layer to the first: the highest layer with a non-finite
        # gradient is where it started.
        bad_layers = [i for i, layer in enumerate(layers)
                      if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in layer.parameters())]
        say(f'      non-finite gradients in layers {bad_layers} (backward starts at the highest: '
            f'{bad_layers[-1]} {type(layers[bad_layers[-1]]).__name__})')
    norm = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for _, p in params if p.grad is not None))
    say(f'      gradient norm {norm:.4g} over {len(params)} tensors')
    if bad:
        say(f'      NON-FINITE gradient in {len(bad)} tensors, e.g. {bad[:5]}')
        say('=> The backward pass produces NaN/inf.')
        return 1

    if args.steps <= 0:
        say('=> Forward and backward are finite. Rerun with --steps N to test the optimizer.')
        return 0
    say(f'[4/4] {args.steps} optimizer steps at lr {config["optimizer"]["lr"]}')
    from utils.optimizer_factory import resolve_optimizer_class
    unique = {id(p): p for _, p in params}
    for p in unique.values():
        if not hasattr(p, 'original_name'):
            p.original_name = ''
    klass, opt_args, kwargs = resolve_optimizer_class(config['optimizer'])
    groups = pipe.get_param_groups(list(unique.values()))
    split = []
    for group in groups:
        decay = [p for p in group['params'] if p.ndim != 1]
        no_decay = [p for p in group['params'] if p.ndim == 1]
        if decay:
            split.append({**group, 'params': decay})
        if no_decay:
            split.append({**group, 'params': no_decay, 'weight_decay': 0})
    optimizer = klass(split, *opt_args, **kwargs)
    clip = config.get('gradient_clipping', None)
    names = {id(p): getattr(p, 'original_name', '') or n for n, p in params}
    for step in range(1, args.steps + 1):
        if step > 1:
            optimizer.zero_grad(set_to_none=True)
            out = features
            for i, layer in enumerate(layers):
                out = run_layer(i, layer, out)
            loss = loss_fn(out, (target, mask))
            loss.backward()
        grads = [p.grad for p in unique.values() if p.grad is not None]
        total = torch.nn.utils.clip_grad_norm_(list(unique.values()), clip) if clip else \
            torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads]))
        optimizer.step()
        broken = [names[id(p)] for p in unique.values() if not torch.isfinite(p).all()]
        say(f'      step {step}: loss {loss.item():.6f}, grad norm {float(total):.4g}')
        if broken:
            say(f'      NON-FINITE weights after step {step} in {len(broken)} tensors, e.g. {broken[:5]}')
            say(f"=> The optimizer update ({config['optimizer']['type']}) writes NaN/inf into the weights.")
            return 1
        if not torch.isfinite(loss):
            say(f'=> The loss turned NaN/inf at step {step} with finite weights.')
            return 1
    say('=> Everything stayed finite. Send the full output to compare against the training run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
