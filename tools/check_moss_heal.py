"""Is a MOSS-Image model healed? Measures the residual stream and the block weights and prints a verdict.

    python tools/check_moss_heal.py --model /models/MOSS-Image-heal
    python tools/check_moss_heal.py --model /models/my_finetune --reference /models/MOSS-Image-SFT
    python tools/check_moss_heal.py --model my.safetensors --weights-only          # no forward pass
    python tools/check_moss_heal.py --model /models/my_finetune --inputs samples.safetensors --json report.json

--model and --reference take what [model] transformer_path takes: a diffusers transformer directory or a
single safetensors file (a prefixed or fused-qkv ComfyUI checkpoint too). The reference is another model (the
unhealed one it came from) whose numbers are printed beside the model's.

What is measured (utils/moss_heal_check.py has the definitions):
  - stream peak: the largest |value| any block writes to the residual stream, and float16's headroom over it.
    Base and SFT peak near 62,000; a healed model peaks far below float16's 65,504;
  - write shares: the share of the attention's and the FFN's writes held by their 32 largest channels
    (what [model.stream_loss.write] pushes under its target);
  - weight outliers: the largest |w| over the RMS of each block weight (what [outlier_decay] pulls under its cap).

The stream numbers depend on the inputs. Without --inputs they are random latents and random text embeddings,
which show whether the weights can produce a large peak but not what real prompts do; for a verdict to rely
on, give --inputs: a safetensors file with `latents` [N, 128, H, W] (clean VAE latents) and `context` [N, L,
text_dim] (the text encoder's output), and optionally `sigmas` [S]. The limits (--peak-limit,
--write-share-limit, --weight-ratio-limit) are rules of thumb from the heal recipe and card, not measured on many
models. Exit status 0 when the verdict is healed, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SIGMAS = (0.1, 0.4, 0.7, 1.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='transformer directory or safetensors file to check')
    parser.add_argument('--reference', help='another model to print beside it (e.g. the unhealed one)')
    parser.add_argument('--inputs', help='safetensors with latents [N,128,H,W], context [N,L,text_dim], optional sigmas')
    parser.add_argument('--weights-only', action='store_true', help='measure the weights only, run nothing')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'],
                        help='the dtype the model is loaded and run in (default bfloat16)')
    parser.add_argument('--device', default=None, help="'cuda', 'cpu' (default: cuda when there is one)")
    parser.add_argument('--k', type=int, default=32, help='channels counted as the top (default 32)')
    parser.add_argument('--cap', type=float, default=16.0, help='weight outlier cap in RMS units (default 16)')
    parser.add_argument('--size', type=int, default=32, help='latent side of the random inputs (default 32 = 512 px)')
    parser.add_argument('--text-len', type=int, default=64, help='token count of the random text (default 64)')
    parser.add_argument('--samples', type=int, default=2, help='how many random inputs (default 2)')
    parser.add_argument('--sigmas', type=float, nargs='+', default=list(DEFAULT_SIGMAS),
                        help='noise levels to run each input at (default 0.1 0.4 0.7 1.0)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-block', action='store_true', help='print the per-block table')
    parser.add_argument('--peak-limit', type=float, default=None)
    parser.add_argument('--write-share-limit', type=float, default=None)
    parser.add_argument('--weight-ratio-limit', type=float, default=None)
    parser.add_argument('--json', help='write the numbers to this file as well')
    return parser.parse_args(argv)


def load_model(path, dtype, device):
    """ComfyUI's MOSSImage module of `path`, on `device`, in eval mode."""
    import torch
    from models import moss_image as mi
    import comfy.model_management
    declared = mi.read_transformer_config(path) if Path(path).is_dir() else {}
    state, metadata = mi.read_component(path, 'diffusion_pytorch_model')
    patcher = mi.load_transformer_patcher(state, metadata, dtype, load_device=torch.device(device),
                                          offload_device=torch.device('cpu'), declared=declared)
    del state
    patcher.set_model_compute_dtype(dtype)
    with torch.no_grad():
        patcher.patch_model()
    model = patcher.model.diffusion_model
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def read_inputs(path):
    import safetensors.torch
    tensors = safetensors.torch.load_file(path)
    missing = [name for name in ('latents', 'context') if name not in tensors]
    if missing:
        raise SystemExit(f'{path} has no {" or ".join(missing)} (found: {sorted(tensors)})')
    latents, context = tensors['latents'], tensors['context']
    if latents.ndim != 4 or context.ndim != 3 or latents.shape[0] != context.shape[0]:
        raise SystemExit(f'latents must be [N, C, H, W] and context [N, L, text_dim] with the same N, '
                         f'got {tuple(latents.shape)} and {tuple(context.shape)}')
    sigmas = [float(s) for s in tensors['sigmas'].flatten()] if 'sigmas' in tensors else list(DEFAULT_SIGMAS)
    return latents, context, sigmas


def check_model(path, args, inputs):
    """(weights summary, stream summary or None, blocks or None, weight report) of one model."""
    import torch
    from utils import moss_heal_check as hc
    dtype = getattr(torch, args.dtype)
    print(f'loading {path} ...', file=sys.stderr, flush=True)
    model = load_model(path, dtype, args.device)
    weights_report = hc.weight_report(model.layers, args.cap)
    weights = hc.summarize_weights(weights_report)
    stream = blocks = None
    if not args.weights_only:
        if inputs is None:
            generator = torch.Generator().manual_seed(args.seed)
            channels = model.video_embedder.weight.shape[1]
            latents = torch.randn(args.samples, channels, args.size, args.size, generator=generator)
            context = torch.randn(args.samples, args.text_len, hc.text_dim_of(model), generator=generator)
            sigmas = args.sigmas
        else:
            latents, context, sigmas = inputs
            sigmas = args.sigmas if args.sigmas != list(DEFAULT_SIGMAS) else sigmas
        import comfy.model_management
        saved = comfy.model_management.in_training
        # The differentiable path diffusion-pipe trains on; it also runs on CPU.
        comfy.model_management.in_training = True
        try:
            blocks = hc.measure(model, latents, context, sigmas, k=args.k, seed=args.seed, device=args.device)
        finally:
            comfy.model_management.in_training = saved
        stream = hc.summarize_stream(blocks)
    del model
    return weights, stream, blocks, weights_report


def main(argv=None):
    args = parse_args(argv)
    import torch
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu':
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
    from utils import moss_heal_check as hc
    limits = hc.Limits(**{name: value for name, value in (
        ('peak', args.peak_limit), ('write_share', args.write_share_limit),
        ('weight_ratio', args.weight_ratio_limit)) if value is not None})
    inputs = read_inputs(args.inputs) if args.inputs and not args.weights_only else None

    weights, stream, blocks, _ = check_model(args.model, args, inputs)
    reference = None
    if args.reference:
        r_weights, r_stream, _, _ = check_model(args.reference, args, inputs)
        reference = (args.reference, r_weights, r_stream)
    result = hc.verdict(weights, stream, limits)
    print(hc.format_report(args.model, weights, stream, result, blocks if args.per_block else None, reference))
    if stream is not None and inputs is None:
        print('\nnote: random inputs; give --inputs (real latents and encoded prompts) for a verdict to rely on.')
    if args.json:
        payload = {'model': args.model, 'weights': weights, 'stream': stream, 'blocks': blocks, 'verdict': result,
                   'limits': vars(limits) if hasattr(limits, '__dict__') else None}
        if reference:
            payload['reference'] = {'model': reference[0], 'weights': reference[1], 'stream': reference[2]}
        Path(args.json).write_text(json.dumps(payload, indent=2))
    return 0 if result['status'] == 'healed' else 1


if __name__ == '__main__':
    sys.exit(main())
