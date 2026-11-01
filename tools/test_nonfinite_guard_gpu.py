"""GPU check for the non-finite step guard (skip_nonfinite_steps), for any model and any way of
training: full fine-tune, LoRA, OPLoRA, LoKr.

    python tools/test_nonfinite_guard_gpu.py run --config <your config.toml> [--gpus N] [--stages S] [--steps 8]

It copies your config, turns the guard on, caps max_steps, and runs train.py through
tools/nonfinite_poison_launcher.py, which writes a NaN into one gradient on chosen steps. Two runs:

  1. NaN on steps 3 and 5: the run finishes, `step N: skipped` is printed for exactly those steps,
     train/update_norm is 0 on them and above 0 on the others, every other number is finite, and
     train/nonfinite_skipped ends at 2. The weights of a skipped step are not touched, which the zero
     update norm shows (the check needs log_update_norm, which the script turns on).
  2. NaN on every step with nonfinite_patience = 3: the run stops after the third skipped step with
     the message that says so.

`--gpus` and `--stages` work as in tools/test_stream_healing_gpu.py: pipeline_stages = --stages
(default --gpus); with more GPUs than stages the rest are data-parallel replicas, which must all skip
the same steps. Run it once per kind of config you train (LoRA, LoKr, OPLoRA, full fine-tune).
Nothing here judges image quality.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

import toml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(condition, message):
    print(('PASS  ' if condition else 'FAIL  ') + message)
    if not condition:
        raise SystemExit(1)


def parse_steps(output):
    steps = {}
    for line in output.splitlines():
        match = re.match(r'step (\d+): (.*)', line)
        if match and 'skipped, the gradient' not in line:
            steps[int(match.group(1))] = {k: float(v) for k, v in re.findall(r'(\w+)=(\S+)', match.group(2))}
    return steps


def launch(config_path, gpus, poison):
    command = ['deepspeed', f'--num_gpus={gpus}', os.path.join(ROOT, 'tools', 'nonfinite_poison_launcher.py'),
               '--deepspeed', '--config', config_path]
    print(' '.join(command), f'(POISON_STEPS={poison})')
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'POISON_STEPS': poison})


def prepared(args, directory, patience):
    config = toml.load(args.config)
    config['output_dir'] = directory
    config['max_steps'] = args.steps
    config['pipeline_stages'] = args.stages or args.gpus
    config['steps_per_print'] = 1
    config['logging_steps'] = 1
    config.setdefault('save_every_n_steps', 10**9)
    config['eval_before_first_step'] = False
    config['compile'] = False
    config['log_update_norm'] = True
    config['skip_nonfinite_steps'] = True
    config['nonfinite_patience'] = patience
    path = os.path.join(directory, f'config_{patience}.toml')
    with open(path, 'w') as f:
        toml.dump(config, f)
    return path


def run(args):
    with tempfile.TemporaryDirectory() as directory:
        result = launch(prepared(args, directory, 8), args.gpus, '3,5')
        output = result.stdout + result.stderr
        steps = parse_steps(output)
        if result.returncode != 0 or len(steps) < args.steps:
            print(output[-4000:])
        check(result.returncode == 0, 'run 1 exited cleanly')
        check(len(steps) >= args.steps, f'{len(steps)} steps logged of {args.steps}')
        skipped = sorted(int(n) for n in re.findall(r'step (\d+): skipped, the gradient is not finite', output))
        check(skipped == [3, 5], f'skipped steps are exactly the poisoned ones: {skipped}')
        for step, values in sorted(steps.items()):
            finite = {k: v for k, v in values.items() if k not in ('grad_norm', 'loss')}
            check(all(math.isfinite(v) for v in finite.values()), f'step {step}: every number but the poisoned norm and loss is finite')
            update = values.get('update_norm')
            if update is not None:
                check((update == 0.0) == (step in (3, 5)), f'step {step}: update_norm {update:.4g} ({"skipped" if step in (3, 5) else "applied"})')
        check(steps[args.steps]['nonfinite_skipped'] == 2.0, 'train/nonfinite_skipped ends at 2')

        result = launch(prepared(args, directory, 3), args.gpus, 'all')
        output = result.stdout + result.stderr
        check(result.returncode != 0, 'run 2 stopped')
        check('3 optimizer steps in a row had a non-finite gradient' in output, 'run 2 stopped with the patience message')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--config', required=True)
    runner.add_argument('--gpus', type=int, default=1)
    runner.add_argument('--stages', type=int, default=0, help='pipeline stages; default: one per GPU')
    runner.add_argument('--steps', type=int, default=8)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
