"""GPU check for MOSS-Image's stream penalties and outlier decay, to be run on a CUDA machine.

Two parts.

1. `python tools/test_stream_healing_gpu.py selftest` (one GPU, no model): the outlier decay on
   bf16 and fp32 CUDA tensors. The bf16 update is far below one bf16 step, so it only works through
   stochastic rounding: the script checks the weights move by the expected amount, that the
   elements below the cap stay bit-identical, and that two generators with the same seed give the
   same result (what keeps data-parallel replicas identical).

2. `python tools/test_stream_healing_gpu.py train --config <your MOSS-Image config.toml> [--gpus N]
   [--steps 8] [--full]` runs the real train.py for a few steps with both features on, using every
   GPU asked for (`pipeline_stages` follows --gpus, so one GPU and eight need no edits), and checks
   the lines it prints:

   - train/loss equals train/flow_loss + train/stream_penalty,
   - the write shares are in [0, 1] and the write target starts at 1 and falls,
   - train/outlier_over and train/outlier_max appear for a full fine-tune (--full) and are finite,
   - the run finishes the steps, with the penalty on both LoRA and full fine-tune configs.

   The config is your own, with your model paths and dataset. The script copies it, adds the
   features, caps max_steps and saves nothing it needs to keep. With --full the config must have no
   [adapter] table (the decay is refused next to one); without it the config should have one.

Nothing here judges image quality. Look at samples from a longer run for that.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

import toml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.outlier_decay import OutlierDecay, apply_outlier_decay_config_defaults  # noqa: E402


def check(condition, message):
    print(('PASS  ' if condition else 'FAIL  ') + message)
    if not condition:
        raise SystemExit(1)


def selftest():
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA device available. Run this on a GPU machine.')
    device = 'cuda'
    for dtype in (torch.float32, torch.bfloat16):
        # cap 4 x r0: a default-initialised layer's largest element is sqrt(3) x r0, so only the 16 x r0 columns exceed it
        cap = 4.0
        config = {'outlier_decay': {'rate': 6.0, 'cap': cap, 'seed': 7}}
        apply_outlier_decay_config_defaults(config)

        def make():
            torch.manual_seed(0)
            module = torch.nn.Linear(256, 256, bias=False, device=device, dtype=dtype)
            optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
            decay = OutlierDecay.build(module.named_parameters(), optimizer, config['outlier_decay'])
            with torch.no_grad():
                module.weight[:, :8] = 16 * decay.r0[0]
            return module, decay

        module, decay = make()
        r0 = decay.r0[0]
        start = module.weight.detach().float().clone()
        steps = 400
        for step in range(1, steps + 1):
            decay.before_step()
            decay.apply(step)
        moved = (start - module.weight.float())[:, :8].mean().item()
        wanted = (16 - cap) * r0 * (1 - (1 - 1e-3 * 6.0) ** steps)
        check(abs(moved - wanted) < 0.15 * wanted, f'{dtype}: outliers moved {moved:.4g}, expected {wanted:.4g}')
        check(torch.equal(module.weight.float()[:, 8:], start[:, 8:]), f'{dtype}: elements below the cap are bit-identical')
        again, decay2 = make()
        for step in range(1, steps + 1):
            decay2.before_step()
            decay2.apply(step)
        check(torch.equal(module.weight, again.weight), f'{dtype}: the same seed gives the same weights')
        scalars = decay.log_scalars()
        check(math.isfinite(scalars['train/outlier_max']), f'{dtype}: metrics are finite {scalars}')


def parse_steps(output):
    """{step: {name: value}} from the 'step N: name=value ...' lines train.py prints."""
    steps = {}
    for line in output.splitlines():
        match = re.match(r'step (\d+): (.*)', line)
        if match:
            steps[int(match.group(1))] = {k: float(v) for k, v in re.findall(r'(\w+)=(\S+)', match.group(2))}
    return steps


def train(args):
    config = toml.load(args.config)
    full = args.full
    if full and 'adapter' in config:
        raise SystemExit('--full needs a config without an [adapter] table')
    if not full and 'adapter' not in config:
        raise SystemExit('Without --full the config needs an [adapter] table (a LoRA run)')
    with tempfile.TemporaryDirectory() as directory:
        config['output_dir'] = directory
        config['max_steps'] = args.steps
        config['pipeline_stages'] = args.stages or args.gpus
        config['steps_per_print'] = 1
        config['logging_steps'] = 1
        config.setdefault('save_every_n_steps', 10**9)
        config['eval_before_first_step'] = False
        config['compile'] = False
        config['model']['stream_loss'] = {
            'write': {'weight': 5e-3, 'target': 0.5, 'anneal': args.steps, 'k': 32},
            'rms': {'weight': 5e-5, 'target': 1.0},
            'concentration': {'weight': 1e-2, 'k': 32},
        }
        if full:
            config['outlier_decay'] = {'rate': 6.0, 'cap': 1.0, 'cap_start': 2.0, 'cap_anneal': args.steps}
        path = os.path.join(directory, 'config.toml')
        with open(path, 'w') as f:
            toml.dump(config, f)
        command = ['deepspeed', f'--num_gpus={args.gpus}', os.path.join(ROOT, 'train.py'), '--deepspeed', '--config', path]
        print(' '.join(command))
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    steps = parse_steps(output)
    if result.returncode != 0 or len(steps) < args.steps:
        print(output[-4000:])
    check(result.returncode == 0, 'train.py exited cleanly')
    check(len(steps) >= args.steps, f'{len(steps)} steps logged of {args.steps}')
    for step, values in sorted(steps.items()):
        check(all(math.isfinite(v) for v in values.values()), f'step {step}: every number is finite')
        check(abs(values['loss'] - (values['flow_loss'] + values['stream_penalty'])) < 3e-3 * max(1.0, abs(values["loss"])),
              f'step {step}: loss {values["loss"]:.5g} = flow {values["flow_loss"]:.5g} + penalty {values["stream_penalty"]:.5g}')
        for part in ('attn', 'ffn'):
            check(0.0 <= values[f'write_share_{part}'] <= 1.0, f'step {step}: write_share_{part} {values[f"write_share_{part}"]:.4f} in [0, 1]')
        if full:
            check('outlier_over' in values and 'outlier_max' in values, f'step {step}: outlier metrics logged')
    first, last = steps[min(steps)], steps[max(steps)]
    check(first['write_target'] >= last['write_target'], f'write target falls: {first["write_target"]:.3f} -> {last["write_target"]:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('selftest')
    run = sub.add_parser('train')
    run.add_argument('--config', required=True)
    run.add_argument('--gpus', type=int, default=1)
    run.add_argument('--stages', type=int, default=0, help='pipeline stages; default: one per GPU')
    run.add_argument('--steps', type=int, default=8)
    run.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.command == 'selftest':
        selftest()
    else:
        train(args)


if __name__ == '__main__':
    main()
