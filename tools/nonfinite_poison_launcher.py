"""Runs train.py with a NaN written into one gradient on chosen optimizer steps, to see the non-finite
step guard work in a real run (tools/test_nonfinite_guard_gpu.py starts it; it is not for training).

    deepspeed --num_gpus=N tools/nonfinite_poison_launcher.py --deepspeed --config c.toml

The steps come from the environment variable POISON_STEPS (comma separated engine step numbers from 1,
or 'all'). Only the first rank's first parameter with a gradient is poisoned, so the other ranks and
stages can only skip the step if the norm reaches them, which is what the guard relies on.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import utils.patches as patches  # noqa: E402

spec = os.environ.get('POISON_STEPS', '')
if not spec:
    raise SystemExit('Set POISON_STEPS (for example 3,5 or all)')
every = spec == 'all'
poison = set() if every else {int(item) for item in spec.split(',')}
original = patches._take_model_step_measured
count = {'step': 0}


def poisoned(self, *args, **kwargs):
    count['step'] += 1
    if (every or count['step'] in poison) and torch.distributed.get_rank() == 0:
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.view(-1)[0] = float('nan')
                break
    return original(self, *args, **kwargs)


patches._take_model_step_measured = poisoned
sys.argv[0] = os.path.join(ROOT, 'train.py')
runpy.run_path(sys.argv[0], run_name='__main__')
