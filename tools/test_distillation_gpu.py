"""GPU check for MOSS-Image's [teacher] distillation, to be run on a CUDA machine with a real model.

    python tools/test_distillation_gpu.py run --config <your MOSS-Image config.toml> [--gpus N]
        [--stages S] [--steps 6] [--full]

Runs the real train.py twice from a copy of your config, once without and once with
`[teacher] loss_weight = 0.5`, for a few steps each, using every GPU asked for (`pipeline_stages`
follows --gpus unless --stages is given; an [adapter] run needs one stage, so give --stages 1 and
the GPUs become data-parallel replicas). The teacher run is checked on the lines train.py prints:

  - train/loss equals train/flow_loss + train/distill_loss,
  - train/distill_lambda is 0.5 (the constant schedule),
  - the distill loss is finite, and for a LoRA run it is close to 0 at the first step (the adapter
    starts at zero, so the student still equals the teacher) and grows once the adapter moves,
  - every step is logged and the run exits cleanly.

It also prints the wall time of the two runs. That is a rough cost of the teacher forward, startup
included; for a clean number run more steps. Peak memory is not measured here: watch nvidia-smi on
the first-stage GPUs, where the full fine-tune keeps its frozen copy of the model.

The config is your own, with your model paths and dataset. With --full it must have no [adapter]
table; without --full it must have one. Nothing here judges image quality.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import toml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(condition, message):
    print(('PASS  ' if condition else 'FAIL  ') + message)
    if not condition:
        raise SystemExit(1)


def parse_steps(output):
    """{step: {name: value}} from the 'step N: name=value ...' lines train.py prints."""
    steps = {}
    for line in output.splitlines():
        match = re.match(r'step (\d+): (.*)', line)
        if match:
            steps[int(match.group(1))] = {k: float(v) for k, v in re.findall(r'(\w+)=(\S+)', match.group(2))}
    return steps


def launch(config, args, directory, name):
    config = dict(config)
    config['output_dir'] = os.path.join(directory, name)
    config['max_steps'] = args.steps
    config['pipeline_stages'] = args.stages or args.gpus
    config['steps_per_print'] = 1
    config['logging_steps'] = 1
    config.setdefault('save_every_n_steps', 10**9)
    config['eval_before_first_step'] = False
    config['compile'] = False
    path = os.path.join(directory, f'{name}.toml')
    with open(path, 'w') as f:
        toml.dump(config, f)
    command = ['deepspeed', f'--num_gpus={args.gpus}', os.path.join(ROOT, 'train.py'), '--deepspeed', '--config', path]
    print(' '.join(command))
    began = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    seconds = time.monotonic() - began
    output = result.stdout + result.stderr
    steps = parse_steps(output)
    if result.returncode != 0 or len(steps) < args.steps:
        print(output[-4000:])
    check(result.returncode == 0, f'{name}: train.py exited cleanly')
    check(len(steps) >= args.steps, f'{name}: {len(steps)} steps logged of {args.steps}')
    return steps, seconds


def run(args):
    config = toml.load(args.config)
    if args.full and 'adapter' in config:
        raise SystemExit('--full needs a config without an [adapter] table')
    if not args.full and 'adapter' not in config:
        raise SystemExit('Without --full the config needs an [adapter] table (a LoRA run)')
    if not args.full and (args.stages or args.gpus) != 1:
        raise SystemExit('A LoRA run with [teacher] needs one pipeline stage: pass --stages 1')
    config.pop('teacher', None)
    with tempfile.TemporaryDirectory() as directory:
        _, base_seconds = launch(config, args, directory, 'baseline')
        config['teacher'] = {'loss_weight': 0.5}
        steps, teacher_seconds = launch(config, args, directory, 'teacher')

    for step, values in sorted(steps.items()):
        check(all(math.isfinite(v) for v in values.values()), f'step {step}: every number is finite')
        for key in ('flow_loss', 'distill_loss', 'distill_lambda'):
            check(key in values, f'step {step}: {key} is logged')
        check(abs(values['loss'] - (values['flow_loss'] + values['distill_loss'])) < 3e-3 * max(1.0, abs(values['loss'])),
              f'step {step}: loss {values["loss"]:.5g} = flow {values["flow_loss"]:.5g} + distill {values["distill_loss"]:.5g}')
        check(abs(values['distill_lambda'] - 0.5) < 1e-4, f'step {step}: lambda {values["distill_lambda"]:.4f} is 0.5')
    ordered = [steps[s] for s in sorted(steps)]
    if not args.full:
        check(ordered[0]['distill_loss'] < 1e-3 * max(1.0, ordered[0]['flow_loss']),
              f'LoRA step 1: distill {ordered[0]["distill_loss"]:.3g} is ~0 (the adapter starts at zero)')
        check(ordered[-1]['distill_loss'] > ordered[0]['distill_loss'],
              f'distill grows as the adapter moves: {ordered[0]["distill_loss"]:.3g} -> {ordered[-1]["distill_loss"]:.3g}')
    print(f'wall time, startup included: {base_seconds:.0f}s without the teacher, {teacher_seconds:.0f}s with it')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', required=True)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--stages', type=int, default=0, help='pipeline stages; default: one per GPU')
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--full', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
