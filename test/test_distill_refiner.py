"""Tests for the distillation script's gradient accumulation and distributed setup.

The script does not go through train.py, so it inherits none of DeepSpeed's machinery. These
cover the two things that machinery would otherwise have given it for free.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class TestGradientAccumulationEquivalence:
    """N micro batches of B must give the same gradient as one batch of N*B.

    That is what the 1/grad_accum scaling buys. Getting it wrong is silent: training still
    runs, the gradient is just N times too large, and the effective learning rate with it.
    """

    def _model(self, seed=0):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))

    def test_accumulated_gradient_matches_one_large_batch(self):
        torch.manual_seed(1)
        data = torch.randn(12, 8)
        target = torch.randn(12, 4)

        big = self._model()
        F.mse_loss(big(data), target).backward()
        big_grads = [p.grad.clone() for p in big.parameters()]

        accum = self._model()
        grad_accum = 3
        chunk = len(data) // grad_accum
        for i in range(grad_accum):
            sl = slice(i * chunk, (i + 1) * chunk)
            loss = F.mse_loss(accum(data[sl]), target[sl])
            (loss / grad_accum).backward()
        accum_grads = [p.grad.clone() for p in accum.parameters()]

        for a, b in zip(big_grads, accum_grads):
            assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()

    def test_without_the_scaling_the_gradient_is_n_times_too_large(self):
        # The failure mode this guards against, stated explicitly.
        torch.manual_seed(1)
        data, target = torch.randn(12, 8), torch.randn(12, 4)
        big = self._model()
        F.mse_loss(big(data), target).backward()

        unscaled = self._model()
        for i in range(3):
            sl = slice(i * 4, (i + 1) * 4)
            F.mse_loss(unscaled(data[sl]), target[sl]).backward()

        a = next(iter(big.parameters())).grad
        b = next(iter(unscaled.parameters())).grad
        assert not torch.allclose(a, b, atol=1e-6)
        assert torch.allclose(a * 3, b, atol=1e-5), 'expected exactly 3x without the scaling'


class TestScriptStructure:
    """Source-level guards: these paths need a GPU or a launcher to exercise for real."""

    def source(self):
        return (REPO / 'tools/distill_refiner.py').read_text()

    def test_grad_accum_scales_the_loss(self):
        assert '(loss / grad_accum).backward()' in self.source()

    def test_zero_grad_is_outside_the_micro_loop(self):
        src = self.source()
        zero = src.index('optimizer.zero_grad(set_to_none=True)')
        micro = src.index('for micro in range(grad_accum):')
        assert zero < micro, 'zero_grad must run before the micro batches, not inside them'

    def test_optimizer_steps_once_per_outer_step(self):
        src = self.source()
        assert src.count('optimizer.step()') == 1

    def test_ddp_no_sync_wraps_all_but_the_last_micro_batch(self):
        src = self.source()
        assert 'train_module.no_sync()' in src
        assert 'sync = (world_size == 1) or (micro == grad_accum - 1)' in src

    def test_saving_is_rank_gated(self):
        # Call sites only -- the def line matches the same substring.
        src = self.source()
        calls = [l for l in src.splitlines()
                 if ('save_refiner(refiner' in l or 'save_full_model(config' in l)
                 and not l.lstrip().startswith('def ')]
        assert calls, 'no save call sites found; did they get renamed?'
        for line in calls:
            indent = len(line) - len(line.lstrip())
            assert indent >= 8, f'save at top level of main(), not under a rank guard: {line!r}'

    def test_the_probe_seed_is_not_rank_offset(self):
        # Every rank must measure against the same queries, or the ranks optimise different
        # objectives and the all-reduce averages nonsense.
        src = self.source()
        assert 'manual_seed(seed)' in src
        assert 'random.seed(seed + rank)' in src, 'only the data stream should differ per rank'


@pytest.mark.skipif(sys.platform == 'win32', reason='gloo rendezvous differs on Windows')
class TestDistributedSmoke:
    """Two real processes over gloo, exercising the same no_sync + accumulation shape."""

    SCRIPT = '''
import os, sys, torch, contextlib
import torch.distributed as dist
import torch.nn.functional as F
rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
dist.init_process_group('gloo', rank=rank, world_size=world)
torch.manual_seed(0)
model = torch.nn.Linear(8, 4)
ddp = torch.nn.parallel.DistributedDataParallel(model)
torch.manual_seed(100 + rank)
data, target = torch.randn(6, 8), torch.randn(6, 4)
grad_accum = 3
opt = torch.optim.SGD(model.parameters(), lr=0.0)
opt.zero_grad(set_to_none=True)
for micro in range(grad_accum):
    sl = slice(micro * 2, (micro + 1) * 2)
    sync = (micro == grad_accum - 1)
    with contextlib.nullcontext() if sync else ddp.no_sync():
        (F.mse_loss(ddp(data[sl]), target[sl]) / grad_accum).backward()
g = next(iter(model.parameters())).grad
gathered = [torch.zeros_like(g) for _ in range(world)]
dist.all_gather(gathered, g)
assert torch.allclose(gathered[0], gathered[1], atol=1e-6), 'ranks disagree after all-reduce'
if rank == 0:
    print('RANKS_AGREE', float(g.abs().sum()))
dist.destroy_process_group()
'''

    def test_two_ranks_end_with_identical_gradients(self, tmp_path):
        script = tmp_path / 'ddp_smoke.py'
        script.write_text(self.SCRIPT)
        env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT='29517', WORLD_SIZE='2')
        procs = [
            subprocess.Popen([sys.executable, str(script)], env=dict(env, RANK=str(r)),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for r in range(2)
        ]
        outs = [p.communicate(timeout=180) for p in procs]
        for (out, err), p in zip(outs, procs):
            assert p.returncode == 0, err[-2000:]
        assert any('RANKS_AGREE' in out for out, _ in outs), outs
