"""Tests for the distillation script's gradient accumulation and distributed setup.

The script does not go through train.py, so it inherits none of DeepSpeed's machinery. These
cover the two things that machinery would otherwise have given it for free.
"""

import contextlib
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
        assert 'self.scaler.scale(loss / grad_accum).backward()' in self.source()

    def test_the_scaler_is_unscaled_before_clipping(self):
        # Clipping a scaled gradient compares max_grad_norm against an inflated norm, so it
        # would effectively never fire.
        src = self.source()
        unscale = src.index('self.scaler.unscale_(self.optimizer)')
        clip = src.index('clip_grad_norm_')
        assert unscale < clip

    def test_zero_grad_is_outside_the_micro_loop(self):
        src = self.source()
        zero = src.index('        strategy.zero_grad()')
        micro = src.index('for micro in range(grad_accum):')
        assert zero < micro, 'zero_grad must run before the micro batches, not inside them'

    def test_optimizer_steps_once_per_outer_step(self):
        # Routed through the GradScaler, which calls optimizer.step() itself (and skips it on a
        # non-finite gradient). Exactly one step site either way.
        src = self.source()
        assert src.count('self.scaler.step(self.optimizer)') == 1
        assert 'optimizer.step()' not in src, 'stepping around the scaler defeats fp16-mixed'

    def test_ddp_no_sync_wraps_all_but_the_last_micro_batch(self):
        src = self.source()
        assert 'self.module.no_sync()' in src
        assert 'if self.world_size == 1 or is_last:' in src

    def test_the_micro_loop_delegates_syncing_to_the_strategy(self):
        # The loop must not hardcode DDP's boundary rule: the ZeRO engine tracks its own.
        src = self.source()
        assert 'strategy.micro_batch_context(is_last=(micro == grad_accum - 1))' in src

    def test_contextlib_is_imported(self):
        # nullcontext is evaluated on the very first micro batch, so a missing import is an
        # immediate NameError rather than a rare path.
        src = self.source()
        assert 'import contextlib' in src
        assert 'contextlib.nullcontext()' in src


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

    def test_both_launchers_drive_the_same_path(self):
        # deepspeed's launcher and torchrun both export RANK/LOCAL_RANK/WORLD_SIZE, so the
        # script must read those and nothing launcher-specific.
        src = self.source()
        assert "os.environ['RANK']" in src
        assert "os.environ['WORLD_SIZE']" in src
        assert "os.environ.get('LOCAL_RANK'" in src
        for launcher_specific in ('OMPI_COMM_WORLD', 'SLURM_PROCID', 'deepspeed.init_distributed'):
            assert launcher_specific not in src, f'{launcher_specific} ties this to one launcher'

    @pytest.mark.parametrize('env,expected', [
        ({}, (0, 1, 0)),
        ({'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_RANK': '0'}, (0, 1, 0)),
    ])
    def test_setup_distributed_single_process(self, env, expected, monkeypatch):
        # Both cases assert. The env-set case used to sit behind `if not env:` and therefore
        # ran no assertion at all, passing whatever setup_distributed returned. The reason for
        # the guard was real -- with RANK set the function calls init_process_group -- so stub
        # the collective out instead of skipping the check.
        for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import tools.distill_refiner as distill
        monkeypatch.setattr(distill.dist, 'is_initialized', lambda: True)
        monkeypatch.setattr(
            distill.dist, 'init_process_group',
            lambda *a, **k: pytest.fail('a single-process launch must not build a process group'),
        )
        assert distill.setup_distributed() == expected

    def test_the_probe_seed_is_not_rank_offset(self):
        # Every rank must measure against the same queries, or the ranks optimise different
        # objectives and the all-reduce averages nonsense.
        #
        # Asserted on the probe's own line, not on 'manual_seed(seed)' anywhere in the file:
        # torch.manual_seed(seed) also appears where the global seed is set, so the looser
        # substring still matched after the probe generator was given a rank offset -- the
        # exact inversion this test names was invisible to it.
        src = self.source()
        assert "generator = torch.Generator(device='cpu').manual_seed(seed)\n" in src, (
            'the probe generator must be seeded without a rank offset'
        )
        assert 'random.seed(seed + rank)' in src, 'only the data stream should differ per rank'


class TestDistributedStrategySelection:
    """The strategy is a config choice, independent of the launcher."""

    def _config(self, **distill):
        return {'distill': {**distill}}

    def _refiner(self):
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))

    def _build(self, config, refiner=None):
        from tools.distill_refiner import build_strategy, resolve_precision
        refiner = refiner if refiner is not None else self._refiner()
        opt = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        return build_strategy(config, refiner, world_size=1, local_rank=0,
                              device=torch.device('cpu'), batch_size=2, grad_accum=2,
                              optimizer=opt, scheduler=sched,
                              precision=resolve_precision(config))

    def test_ddp_is_the_default(self):
        strategy = self._build(self._config())
        assert strategy.name == 'ddp'

    def test_unknown_strategy_is_rejected_by_name(self):
        with pytest.raises(RuntimeError, match='distributed_strategy'):
            self._build(self._config(distributed_strategy='fsdp'))

    def _zero(self, stage=1, world_size=2, **kw):
        from tools.distill_refiner import DeepSpeedZeROStrategy, resolve_precision
        refiner = kw.pop('refiner', None) or self._refiner()
        opt = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        precision = resolve_precision({'distill': {'distributed_strategy': 'zero1'}})
        return DeepSpeedZeROStrategy(refiner, stage, world_size, 16, 2, opt, sched, 1.0,
                                     precision)

    def test_zero3_is_refused_with_a_reason(self):
        # Not merely unimplemented: stage 3 shards the parameters, which breaks the save path.
        with pytest.raises(RuntimeError, match='stage 3'):
            self._zero(stage=3)

    def test_zero_on_a_single_rank_is_refused_before_deepspeed_sees_it(self):
        # Left to DeepSpeed this dies on `No module named 'mpi4py'` from its MPI discovery
        # path, which says nothing about the real problem.
        with pytest.raises(RuntimeError, match='world_size=1'):
            self._zero(world_size=1)

    def test_the_generated_config_is_valid_deepspeed(self, monkeypatch):
        """Parse the config we build with DeepSpeed's own parser, not by eyeballing the dict."""
        from deepspeed.runtime.config import DeepSpeedConfig
        import deepspeed

        captured = {}

        def fake_initialize(model=None, optimizer=None, lr_scheduler=None, config=None, **kw):
            captured['config'] = config
            captured['kwargs'] = kw
            return object(), optimizer, None, lr_scheduler

        monkeypatch.setattr(deepspeed, 'initialize', fake_initialize)
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
        self._zero(stage=1)

        parsed = DeepSpeedConfig(captured['config'])
        assert parsed.zero_config.stage == 1
        assert parsed.gradient_accumulation_steps == 2
        assert parsed.gradient_clipping == 1.0
        # The refiner trains in fp32. If the engine were to enable either of these it would cast
        # the master weights out from under the optimizer.
        assert not parsed.float16_config.enabled
        assert not parsed.bfloat16_config.enabled
        # The process group already exists; a second one must not be stood up.
        assert captured['kwargs']['dist_init_required'] is False

    def test_the_strategy_delegates_to_the_engine_and_does_not_touch_the_optimizer(self, monkeypatch):
        """backward/step/zero_grad must go through the engine, which owns accumulation."""
        import deepspeed

        calls = []

        class FakeEngine:
            def set_gradient_accumulation_boundary(self, is_boundary):
                # The strategy drives this per micro batch. DeepSpeed cannot work the boundary
                # out for itself under a loop that calls backward N times and step once, so a
                # fake engine that lacks this method no longer matches the real interface.
                calls.append(f'boundary={is_boundary}')

            def backward(self, loss):
                calls.append('backward')

            def step(self):
                calls.append('step')

            def get_global_grad_norm(self):
                return None  # what the engine returns before the first boundary

        def fake_initialize(model=None, optimizer=None, lr_scheduler=None, config=None, **kw):
            return FakeEngine(), optimizer, None, lr_scheduler

        monkeypatch.setattr(deepspeed, 'initialize', fake_initialize)
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
        strategy = self._zero(stage=2)

        # The boundary is driven from out here. The engine derives it from a counter that only
        # advances inside step(), so under this loop shape -- N backwards, one step -- it would
        # otherwise reach the boundary once every N outer steps instead of once per outer step.
        with strategy.micro_batch_context(is_last=False):
            strategy.backward(torch.tensor(1.0))
        strategy.zero_grad()  # must be a no-op, or gradients vanish mid-accumulation
        assert strategy.step() == 0.0  # None from the engine becomes 0.0, not a crash
        assert calls == ['boundary=False', 'backward', 'step'], (
            'the boundary must be set BEFORE the backward it describes'
        )

    def test_zero_does_not_rescale_the_loss_a_second_time(self):
        # DeepSpeed's engine.backward applies 1/gradient_accumulation_steps itself. Dividing
        # again here would silently halve (or worse) the effective learning rate.
        src = (REPO / 'tools/distill_refiner.py').read_text()
        body = src[src.index('class DeepSpeedZeROStrategy'):src.index('def build_strategy')]
        assert 'self.engine.backward(loss)' in body
        assert '/ grad_accum' not in body, 'the engine already scales; do not scale twice'
        assert 'clip_grad_norm_' not in body, 'gradient_clipping in the config already clips'

    def test_ddp_strategy_matches_one_large_batch(self):
        # The strategy wrapper must not change the accumulation arithmetic it replaced.
        torch.manual_seed(1)
        data, target = torch.randn(12, 8), torch.randn(12, 4)

        big = self._refiner()
        F.mse_loss(big(data), target).backward()

        accum = self._refiner()
        strategy = self._build(self._config(), refiner=accum)
        strategy.grad_accum = 3  # three micro batches of 4 below, not the 2 _build passes
        strategy.zero_grad()
        for i in range(3):
            sl = slice(i * 4, (i + 1) * 4)
            with strategy.micro_batch_context(is_last=(i == 2)):
                strategy.backward(F.mse_loss(accum(data[sl]), target[sl]))

        for a, b in zip(big.parameters(), accum.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=1e-6), (a.grad - b.grad).abs().max()

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


class TestPrecision:
    """`precision` controls the trainable refiner. `dtype` controls the frozen modules.

    Confusing the two is the whole reason this setting exists as its own name: the config has
    carried `dtype = 'bfloat16'` since the beginning while the refiner trained in fp32.
    """

    def _resolve(self, precision=None, strategy=None):
        from tools.distill_refiner import resolve_precision
        distill = {}
        if precision is not None:
            distill['precision'] = precision
        if strategy is not None:
            distill['distributed_strategy'] = strategy
        return resolve_precision({'distill': distill})

    def test_the_default_is_unchanged_fp32(self):
        p = self._resolve()
        assert (p.name, p.param_dtype, p.autocast_dtype, p.needs_scaler) == \
            ('fp32', torch.float32, None, False)
        assert p.deepspeed_section == {}, 'fp32 must add no mixed-precision section'

    def test_unknown_precision_is_rejected(self):
        with pytest.raises(RuntimeError, match='precision'):
            self._resolve('bf16')  # a plausible typo for bf16-mixed / bf16-full

    def test_fp16_full_is_refused_with_the_numeric_reason(self):
        with pytest.raises(RuntimeError, match='underflow'):
            self._resolve('fp16-full')

    def test_bf16_mixed_keeps_fp32_parameters_and_needs_no_scaler(self):
        p = self._resolve('bf16-mixed')
        assert p.param_dtype is torch.float32, 'mixed means fp32 master weights'
        assert p.autocast_dtype is torch.bfloat16
        assert not p.needs_scaler, 'bf16 has fp32 exponent range; a scaler would be theatre'

    def test_bf16_full_casts_the_parameters_under_ddp(self):
        p = self._resolve('bf16-full', 'ddp')
        assert p.param_dtype is torch.bfloat16
        assert p.autocast_dtype is None, 'autocast over bf16 parameters is a no-op'

    def test_fp16_mixed_uses_a_scaler_under_ddp_but_not_under_zero(self):
        ddp = self._resolve('fp16-mixed', 'ddp')
        assert ddp.needs_scaler and ddp.autocast_dtype is torch.float16
        assert ddp.deepspeed_section == {}

        zero = self._resolve('fp16-mixed', 'zero1')
        # The engine owns loss scaling; a GradScaler on top of it would scale twice.
        assert not zero.needs_scaler
        assert zero.autocast_dtype is None
        assert zero.deepspeed_section == {'fp16': {'enabled': True, 'loss_scale': 0}}

    @pytest.mark.parametrize('name,expected', [
        ('bf16-full', {'bf16': {'enabled': True}}),
        ('fp16-mixed', {'fp16': {'enabled': True, 'loss_scale': 0}}),
        ('bf16-mixed', {}),
        ('fp32', {}),
    ])
    def test_the_deepspeed_section_reaches_the_engine_config(self, name, expected, monkeypatch):
        """Resolve, then parse what the strategy actually hands DeepSpeed."""
        from deepspeed.runtime.config import DeepSpeedConfig
        from tools.distill_refiner import DeepSpeedZeROStrategy
        import deepspeed

        captured = {}

        def fake_initialize(model=None, optimizer=None, lr_scheduler=None, config=None, **kw):
            captured['config'] = config
            return object(), optimizer, None, lr_scheduler

        monkeypatch.setattr(deepspeed, 'initialize', fake_initialize)
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)

        precision = self._resolve(name, 'zero1')
        model = torch.nn.Linear(8, 4).to(precision.param_dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        DeepSpeedZeROStrategy(model, 1, 2, 16, 2, opt, sched, 1.0, precision)

        for key, section in expected.items():
            assert captured['config'][key] == section
        parsed = DeepSpeedConfig(captured['config'])
        assert parsed.float16_config.enabled == ('fp16' in expected)
        assert parsed.bfloat16_config.enabled == ('bf16' in expected)


class TestPrecisionEndToEnd:
    """Actually run a step in each mode and look at the dtypes and the gradients."""

    def _strategy(self, precision_name):
        from tools.distill_refiner import build_strategy, resolve_precision
        cfg = {'distill': {'precision': precision_name, 'max_grad_norm': 1.0}}
        precision = resolve_precision(cfg)
        torch.manual_seed(0)
        refiner = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(),
                                      torch.nn.Linear(32, 32))
        refiner.to(dtype=precision.param_dtype)
        opt = torch.optim.AdamW(refiner.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        strategy = build_strategy(cfg, refiner, 1, 0, torch.device('cpu'), 4, 2, opt, sched,
                                  precision)
        return refiner, strategy, precision

    @pytest.mark.parametrize('name', ['fp32', 'bf16-mixed', 'fp16-mixed', 'bf16-full'])
    def test_a_full_step_runs_and_moves_the_weights(self, name):
        refiner, strategy, precision = self._strategy(name)
        before = [p.detach().clone() for p in refiner.parameters()]

        # The frozen side stays bf16 regardless of `precision`, exactly as in the real loop.
        frozen = torch.nn.Linear(32, 32).to(torch.bfloat16).requires_grad_(False)
        hidden = torch.randn(4, 8, 32, dtype=torch.bfloat16)
        target = frozen(torch.randn(4, 8, 32, dtype=torch.bfloat16)).float()

        strategy.zero_grad()
        for micro in range(2):
            with strategy.micro_batch_context(is_last=(micro == 1)):
                with precision.autocast('cpu'):
                    feats = strategy.module(hidden.to(precision.param_dtype))
                pred = frozen(feats.to(torch.bfloat16))
                strategy.backward(F.mse_loss(pred.float(), target))

        for p in refiner.parameters():
            assert p.grad is not None, f'{name}: no gradient reached the refiner'
            assert torch.isfinite(p.grad).all(), f'{name}: non-finite gradient'
        grad_norm = strategy.step()
        assert grad_norm >= 0.0

        moved = any(not torch.equal(a, b) for a, b in zip(before, refiner.parameters()))
        assert moved, f'{name}: optimizer step left every weight unchanged'

    @pytest.mark.parametrize('name,expected', [
        ('fp32', torch.float32),
        ('bf16-mixed', torch.float32),
        ('fp16-mixed', torch.float32),
        ('bf16-full', torch.bfloat16),
    ])
    def test_parameter_dtype_matches_the_mode(self, name, expected):
        refiner, _, _ = self._strategy(name)
        for p in refiner.parameters():
            assert p.dtype is expected

    def test_optimizer_state_matches_the_parameter_dtype(self):
        # The reason precision is applied before the optimizer is built: AdamW sizes exp_avg
        # to the parameter dtype at construction.
        refiner, strategy, _ = self._strategy('bf16-full')
        strategy.zero_grad()
        out = strategy.module(torch.randn(2, 8, 32, dtype=torch.bfloat16))
        strategy.backward(out.float().pow(2).mean())
        strategy.step()
        state = strategy.optimizer.state[next(iter(refiner.parameters()))]
        assert state['exp_avg'].dtype is torch.bfloat16


class TestOptimizerAndScheduler:
    """Distillation reuses train.py's optimizer names and the repo's shared LR scheduler."""

    def _build(self, distill=None, optimizer=None, params=None):
        from tools.distill_refiner import build_optimizer
        torch.manual_seed(0)
        model = params if params is not None else torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))
        cfg = {'distill': distill or {}}
        if optimizer is not None:
            cfg['optimizer'] = optimizer
        return model, build_optimizer(cfg, model, is_main=False)

    def test_no_optimizer_table_reproduces_the_old_hardcoded_adamw(self):
        # The behaviour every existing distill config gets. It must not move.
        _, opt = self._build({'lr': 3e-4, 'betas': [0.8, 0.95], 'weight_decay': 0.05})
        assert isinstance(opt, torch.optim.AdamW)
        g = opt.param_groups[0]
        assert (g['lr'], tuple(g['betas']), g['weight_decay']) == (3e-4, (0.8, 0.95), 0.05)

    def test_both_forms_at_once_is_refused_not_half_merged(self):
        # The failure this prevents: an [optimizer] table holding only `type` inherits lr but
        # silently takes torch's defaults for betas and weight_decay, discarding the values
        # configured under [distill] with nothing said.
        with pytest.raises(RuntimeError, match='exactly one'):
            self._build({'lr': 3e-4, 'betas': [0.8, 0.95], 'weight_decay': 0.05},
                        optimizer={'type': 'adamw'})

    def test_the_error_names_the_offending_keys(self):
        with pytest.raises(RuntimeError, match='betas'):
            self._build({'betas': [0.8, 0.95]}, optimizer={'type': 'sgd', 'lr': 0.1})

    def test_scheduler_and_clipping_keys_do_not_trigger_the_conflict(self):
        # warmup_steps / lr_scheduler / max_grad_norm configure the schedule and clipping, not
        # the optimizer, so they stay under [distill] alongside an [optimizer] table.
        _, opt = self._build({'warmup_steps': 100, 'lr_scheduler': 'linear', 'max_grad_norm': 1.0},
                             optimizer={'type': 'sgd', 'lr': 0.1})
        assert isinstance(opt, torch.optim.SGD)

    def test_the_optimizer_table_selects_by_train_py_name(self):
        from utils.optimizer_factory import resolve_optimizer_class
        # sgd rather than an 8-bit optimizer: this asserts the wiring, not CUDA availability.
        _, opt = self._build(optimizer={'type': 'sgd', 'lr': 0.1, 'momentum': 0.9})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]['momentum'] == 0.9
        # And the same name resolves identically through the shared table train.py now uses.
        klass, _, _ = resolve_optimizer_class({'type': 'sgd', 'lr': 0.1})
        assert klass is torch.optim.SGD

    def test_gradient_release_is_refused(self):
        with pytest.raises(RuntimeError, match='gradient_release'):
            self._build(optimizer={'type': 'adamw', 'lr': 1e-4, 'gradient_release': True})

    def test_genericoptim_is_refused_with_the_reason(self):
        with pytest.raises(RuntimeError, match='mpu'):
            self._build(optimizer={'type': 'genericoptim', 'lr': 1e-4})

    def test_the_optimizer_table_carries_its_own_hyperparameters(self):
        _, opt = self._build(optimizer={'type': 'adamw', 'lr': 1e-5, 'betas': [0.8, 0.95],
                                        'weight_decay': 0.05})
        g = opt.param_groups[0]
        assert (g['lr'], tuple(g['betas']), g['weight_decay']) == (1e-5, (0.8, 0.95), 0.05)

    def test_weight_decay_split_is_off_by_default(self):
        _, opt = self._build({'weight_decay': 0.01})
        assert len(opt.param_groups) == 1, 'default must not change existing runs'

    def test_weight_decay_split_excludes_1d_parameters_when_enabled(self):
        model, opt = self._build({'weight_decay': 0.01, 'no_weight_decay_on_1d': True})
        assert len(opt.param_groups) == 2
        decay, no_decay = opt.param_groups
        assert no_decay['weight_decay'] == 0.0
        assert all(p.ndim > 1 for p in decay['params'])
        assert all(p.ndim == 1 for p in no_decay['params'])
        # Every parameter is in exactly one group, none dropped.
        assert len(decay['params']) + len(no_decay['params']) == len(list(model.parameters()))

    def test_scheduler_comes_from_the_shared_helper(self):
        from tools.distill_refiner import build_lr_scheduler
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        sched = build_lr_scheduler({'distill': {'lr_scheduler': 'linear', 'warmup_steps': 0}},
                                   opt, steps=10)
        lrs = []
        for _ in range(11):
            lrs.append(opt.param_groups[0]['lr'])
            opt.step()
            sched.step()
        assert lrs[0] == pytest.approx(1.0)
        assert lrs[-1] == pytest.approx(0.0, abs=1e-6)

    def test_warmup_then_cosine_is_still_the_default_shape(self):
        from tools.distill_refiner import build_lr_scheduler
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)
        sched = build_lr_scheduler({'distill': {'warmup_steps': 100}}, opt, steps=1000)
        lrs = []
        for _ in range(1000):
            lrs.append(opt.param_groups[0]['lr'])
            opt.step()
            sched.step()
        assert lrs[0] < lrs[50] < lrs[100], 'warmup must ramp up'
        assert lrs[100] == pytest.approx(1e-4, rel=1e-3), 'peak at the end of warmup'
        assert all(b <= a + 1e-12 for a, b in zip(lrs[100:], lrs[101:])), 'then decay'

    @pytest.mark.parametrize('name', ['constant', 'linear', 'cosine', 'cosine_with_restarts'])
    def test_every_shared_scheduler_name_is_reachable(self, name):
        # The point of the switch: three of these were unavailable before.
        from tools.distill_refiner import build_lr_scheduler
        opt = torch.optim.SGD(torch.nn.Linear(4, 4).parameters(), lr=1e-4)
        sched = build_lr_scheduler({'distill': {'lr_scheduler': name, 'warmup_steps': 10}},
                                   opt, steps=100)
        for _ in range(20):
            opt.step()
            sched.step()
        assert opt.param_groups[0]['lr'] > 0


class TestGradientClipping:
    """max_grad_norm must actually clip in every mode, by two different mechanisms.

    Under DDP this code calls clip_grad_norm_ itself; under ZeRO the engine does it from
    gradient_clipping in the config and this code must NOT clip again. Both need checking,
    because "the setting is present" and "the gradient is clipped" are different claims.
    """

    MAX_NORM = 1.0

    def _explode(self, precision_name):
        """One step with a deliberately enormous gradient, returning (reported, post-clip)."""
        from tools.distill_refiner import build_strategy, resolve_precision
        cfg = {'distill': {'precision': precision_name, 'max_grad_norm': self.MAX_NORM}}
        precision = resolve_precision(cfg)
        torch.manual_seed(0)
        model = torch.nn.Linear(128, 128).to(dtype=precision.param_dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0)  # lr 0: grads survive the step
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        strategy = build_strategy(cfg, model, 1, 0, torch.device('cpu'), 4, 1, opt, sched,
                                  precision)
        strategy.zero_grad()
        out = model(torch.randn(32, 128, dtype=precision.param_dtype) * 100)
        strategy.backward(out.float().pow(2).sum())
        # step() ends with zero_grad, which would erase what we came to measure.
        opt.zero_grad = lambda **kw: None
        reported = strategy.step()
        post = torch.cat([p.grad.flatten().float() for p in model.parameters()]).norm().item()
        return reported, post

    @pytest.mark.parametrize('name', ['fp32', 'bf16-mixed', 'fp16-mixed', 'bf16-full'])
    def test_ddp_clips_to_max_grad_norm(self, name):
        reported, post = self._explode(name)
        assert reported > 1e3, f'{name}: test is meaningless unless the gradient is large'
        # bf16-full accumulates the norm in bf16, so allow a little slack.
        assert post <= self.MAX_NORM * 1.02, f'{name}: not clipped, norm is {post}'

    @pytest.mark.parametrize('name', ['fp32', 'bf16-mixed', 'fp16-mixed', 'bf16-full'])
    def test_the_reported_norm_is_pre_clip(self, name):
        # Logging a post-clip norm would show a flat line at max_grad_norm and hide the signal.
        reported, post = self._explode(name)
        assert reported > post * 100

    @pytest.mark.parametrize('precision_name', ['fp32', 'bf16-full', 'fp16-mixed'])
    def test_zero_hands_clipping_to_the_engine_and_does_not_double_clip(self, precision_name,
                                                                       monkeypatch):
        from deepspeed.runtime.config import DeepSpeedConfig
        from tools.distill_refiner import DeepSpeedZeROStrategy, resolve_precision
        import deepspeed

        captured = {}

        def fake_initialize(model=None, optimizer=None, lr_scheduler=None, config=None, **kw):
            captured['config'] = config
            return object(), optimizer, None, lr_scheduler

        monkeypatch.setattr(deepspeed, 'initialize', fake_initialize)
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)

        precision = resolve_precision({'distill': {'precision': precision_name,
                                                   'distributed_strategy': 'zero1'}})
        model = torch.nn.Linear(8, 8).to(precision.param_dtype)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        DeepSpeedZeROStrategy(model, 1, 2, 16, 2, opt, sched, 0.7, precision)

        assert DeepSpeedConfig(captured['config']).gradient_clipping == 0.7


class TestZeROAccumulationBoundaryForReal:
    """Drive a real DeepSpeed engine, not a fake one.

    Every other ZeRO test in this file monkeypatches deepspeed.initialize away and asserts that
    the strategy CALLS backward and step. That is worth checking, but it cannot see whether the
    engine acts on those calls -- and it did not. DeepSpeed advances micro_steps inside step(),
    never inside backward(), and derives the accumulation boundary from that counter, so a loop
    that calls backward() N times and step() once reached the boundary every Nth OUTER step.
    With gradient_accumulation_steps = 4 only one optimizer update in six actually happened and
    the LR schedule never annealed, while the run completed and the progress bar filled.

    Nothing about that is visible to a mock, so this test uses the real engine.
    """

    @staticmethod
    def _init_engine(gas, stage=1):
        import torch.distributed as dist
        import deepspeed
        import deepspeed.ops

        # The shm comm op is JIT-compiled and needs a C++ toolchain, which a CPU-only test box
        # need not have. build_shm_op() checks this registry and returns None when the op is
        # marked incompatible, which is the supported way to skip it.
        for name in list(deepspeed.ops.__compatible_ops__):
            if 'shm' in name.lower():
                deepspeed.ops.__compatible_ops__[name] = False

        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29577')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.zeros_(model.weight)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        config = {
            'train_micro_batch_size_per_gpu': 2,
            'gradient_accumulation_steps': gas,
            'gradient_clipping': 1.0,
            'zero_optimization': {
                'stage': stage, 'contiguous_gradients': True, 'overlap_comm': True,
                # Small on purpose: DeepSpeed's 5e8-element default would allocate 2 GB here.
                'reduce_bucket_size': 4096, 'allgather_bucket_size': 4096,
            },
            'zero_allow_untested_optimizer': True,
            'steps_per_print': 10 ** 9,
            'wall_clock_breakdown': False,
        }
        engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=opt, lr_scheduler=sched, config=config,
            dist_init_required=False,
        )
        return engine, model

    def _count_updates(self, gas, outer_steps, set_boundary, stage=1):
        engine, model = self._init_engine(gas, stage)
        torch.manual_seed(1234)
        previous = model.weight.detach().clone()
        updates = 0
        for _ in range(outer_steps):
            # Exactly the shape of the loop in tools/distill_refiner.py: N backwards, one step.
            for micro in range(gas):
                if set_boundary:
                    engine.set_gradient_accumulation_boundary(micro == gas - 1)
                loss = ((engine(torch.randn(2, 4)) - 1.0) ** 2).mean()
                engine.backward(loss)
            engine.step()
            if not torch.equal(previous, model.weight.detach()):
                updates += 1
            previous = model.weight.detach().clone()
        return updates

    @pytest.mark.parametrize('stage', [1, 2])
    def test_the_boundary_must_be_set_or_most_updates_are_skipped(self, stage):
        """The regression itself: without the boundary call the engine skips updates."""
        pytest.importorskip('deepspeed')
        gas, outer = 4, 6
        assert self._count_updates(gas, outer, set_boundary=False, stage=stage) < outer, (
            'Expected the unfixed call pattern to skip optimizer updates. If this now passes, '
            'DeepSpeed changed how it derives the accumulation boundary and '
            'DeepSpeedZeROStrategy.micro_batch_context should be re-checked against it.'
        )

    @pytest.mark.parametrize('stage', [1, 2])
    def test_setting_the_boundary_gives_one_update_per_outer_step(self, stage):
        """Stage 2 takes a different backward path -- it reduces on every micro batch rather
        than only at the boundary -- so the boundary contract has to be checked against both."""
        pytest.importorskip('deepspeed')
        gas, outer = 4, 6
        assert self._count_updates(gas, outer, set_boundary=True, stage=stage) == outer

    def test_the_strategy_sets_the_boundary_on_every_micro_batch(self, monkeypatch):
        """The strategy must be the thing that calls it, not just the test."""
        import deepspeed
        from tools.distill_refiner import DeepSpeedZeROStrategy, resolve_precision

        calls = []

        class FakeEngine:
            def set_gradient_accumulation_boundary(self, is_boundary):
                calls.append(is_boundary)

        monkeypatch.setattr(
            deepspeed, 'initialize',
            lambda **kw: (FakeEngine(), kw['optimizer'], None, kw['lr_scheduler']),
        )
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)

        model = torch.nn.Linear(8, 8)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        strategy = DeepSpeedZeROStrategy(
            model, 1, 2, 16, 3, opt, sched, 1.0, resolve_precision({'distill': {}}))

        for micro in range(3):
            with strategy.micro_batch_context(is_last=(micro == 2)):
                pass

        assert calls == [False, False, True], (
            f'the boundary must be False until the last micro batch, got {calls}'
        )


class TestRelationalTermCatchesCollapse:
    """The probe loss compares captions one at a time, so it cannot see them merging.

    A student mapping every caption to the same feature satisfies a per-caption objective about
    as well as one that keeps them apart. That is the mode collapse reported for naive
    text-encoder distillation (CVPR 2025), and the relational term is what makes it costly.
    """

    @staticmethod
    def _collapsed(teacher, fraction):
        centre = teacher.mean(0, keepdim=True)
        return teacher * (1 - fraction) + centre * fraction

    def test_the_penalty_grows_with_collapse(self):
        from tools.distill_refiner import relational_loss
        torch.manual_seed(0)
        teacher = torch.randn(8, 32)
        losses = [float(relational_loss(self._collapsed(teacher, f), teacher))
                  for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert losses[0] < 1e-6, 'an exact match must cost nothing'
        assert all(a < b for a, b in zip(losses, losses[1:])), (
            f'the penalty must increase monotonically with collapse, got {losses}'
        )

    def test_uniform_shrinkage_is_visible(self):
        # The usual RKD formulation normalises each side by its own mean, which makes the loss
        # scale invariant -- and uniform shrinkage toward a centroid is exactly a scale change,
        # so collapse became invisible. Both sides share the teacher's scale for this reason.
        from tools.distill_refiner import relational_loss
        torch.manual_seed(0)
        teacher = torch.randn(8, 32)
        assert float(relational_loss(teacher * 0.25, teacher)) > 0.01

    def test_a_single_sample_has_no_structure_and_no_gradient_blow_up(self):
        from tools.distill_refiner import relational_loss
        one = torch.randn(1, 32, requires_grad=True)
        loss = relational_loss(one, torch.randn(1, 32))
        loss.backward()
        assert float(loss) == 0.0
        assert torch.isfinite(one.grad).all()

    def test_the_diagnostic_tracks_spread(self):
        from tools.distill_refiner import mean_pairwise_cosine_distance
        torch.manual_seed(0)
        teacher = torch.randn(8, 32)
        assert mean_pairwise_cosine_distance(teacher) > 0.5
        assert mean_pairwise_cosine_distance(self._collapsed(teacher, 1.0)) < 0.01


class TestDistillResumeIsComplete:
    """resume_from restored the weights and nothing else.

    Adam's moments restarted at zero and the LR schedule rebuilt from step 0, so a run resumed
    at 15,000 of 20,000 steps re-ran its warmup at peak learning rate and then the whole cosine
    again -- a visible regression that costs more than the interruption did.
    """

    def test_optimizer_scheduler_and_step_round_trip(self, tmp_path):
        from tools.distill_refiner import save_training_state, load_training_state

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 ** s)
        for _ in range(3):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            sched.step()

        refiner_path = tmp_path / 'context_refiner.safetensors'
        save_training_state(refiner_path, opt, sched, step=1234)

        fresh_model = torch.nn.Linear(4, 4)
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=0.1)
        fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lambda s: 0.5 ** s)
        step = load_training_state(refiner_path, fresh_opt, fresh_sched, is_main=False)

        assert step == 1234
        assert fresh_sched.get_last_lr() == sched.get_last_lr()
        assert fresh_opt.state_dict()['state'], 'the optimizer moments must come back non-empty'

    def test_a_missing_state_file_is_not_an_error(self, tmp_path):
        from tools.distill_refiner import load_training_state
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        # A refiner distilled before this existed must still be usable as a warm start.
        assert load_training_state(tmp_path / 'context_refiner.safetensors', opt, sched, False) == 0

    def test_the_refiner_is_saved_in_fp32(self, tmp_path):
        # dtype describes the frozen modules; the trainable refiner has always been fp32, and
        # saving through bf16 threw away sixteen mantissa bits on every periodic save.
        import safetensors.torch
        from tools.distill_refiner import save_refiner
        from models.text_refiner import ContextRefiner

        refiner = ContextRefiner(cap_feat_dim=16, model_dim=8, num_layers=1, num_heads=2)
        refiner.init_weights()
        path = tmp_path / 'context_refiner.safetensors'
        save_refiner(refiner, path, torch.bfloat16)

        loaded = safetensors.torch.load_file(str(path))
        assert all(v.dtype == torch.float32 for v in loaded.values())

    def test_provenance_is_recorded(self, tmp_path):
        import safetensors
        from tools.distill_refiner import save_refiner, refiner_provenance
        from models.text_refiner import ContextRefiner

        config = {'student': {'llm_path': '/models/Qwen3.5-2B-Base', 'llm_hidden_layer': 20}}
        refiner = ContextRefiner(cap_feat_dim=16, model_dim=8, num_layers=1, num_heads=2)
        refiner.init_weights()
        path = tmp_path / 'context_refiner.safetensors'
        save_refiner(refiner, path, torch.float32,
                     metadata=refiner_provenance(config, cap_feat_dim=16, max_text_length=512))

        with safetensors.safe_open(str(path), framework='pt') as f:
            metadata = f.metadata()
        assert metadata['llm_path'] == '/models/Qwen3.5-2B-Base'
        assert metadata['llm_hidden_layer'] == '20'
        assert metadata['max_text_length'] == '512'


class TestDenoisingRollout:
    """The rollout produces x_t on a real sampling trajectory, which this stage cannot otherwise get.

    Distillation has no images, so there is no x_0 and therefore no way to build
    x_t = (1-t)*x_0 + t*noise the way training normally does. The trajectory is what supplies
    plausible x_t at all.

    It is driven entirely by the teacher: the student never advances it and never sees its own
    output as input, so there is no error accumulation and none of the exposure-bias behaviour
    the word "rollout" usually implies. Built against a real MiniTrainDIT, small enough to run
    on CPU, because the point is that this works with the actual model.
    """

    DIT_CONFIG = dict(
        max_img_h=64, max_img_w=64, max_frames=8,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        model_channels=64, num_blocks=2, num_heads=4,
        crossattn_emb_channels=32,
        concat_padding_mask=True,
        pos_emb_cls='rope3d', pos_emb_learnable=True, pos_emb_interpolation='crop',
        min_fps=1, max_fps=30,
        use_adaln_lora=True, adaln_lora_dim=16,
    )

    @classmethod
    def _dit(cls):
        from models.cosmos_predict2_modeling import MiniTrainDIT
        torch.manual_seed(0)
        return MiniTrainDIT(**cls.DIT_CONFIG).eval().requires_grad_(False)

    @classmethod
    def _trajectory(cls, dit, steps=4, guidance_scale=0.0, uncond=None):
        from tools.distill_refiner import teacher_trajectory
        teacher = torch.randn(2, 8, cls.DIT_CONFIG['crossattn_emb_channels'])
        shape = (2, cls.DIT_CONFIG['in_channels'], 1, 16, 16)
        visited = teacher_trajectory(
            dit, teacher, uncond, shape, steps, guidance_scale,
            torch.Generator().manual_seed(0), torch.device('cpu'), torch.float32,
        )
        return teacher, visited

    def test_the_schedule_runs_from_pure_noise_downward(self):
        # t = 1 is pure noise and t = 0 is clean, which is the rectified-flow convention this
        # DiT was trained with -- not DDPM's, where the paper's Eq. 3 lives.
        _, visited = self._trajectory(self._dit(), steps=4)
        schedule = [round(float(t[0, 0]), 4) for _, t, _ in visited]
        assert schedule == [1.0, 0.75, 0.5, 0.25], schedule

    def test_the_trajectory_carries_no_gradient(self):
        # The whole walk is teacher-driven and frozen. If this ever starts carrying gradient,
        # the cost of `steps` stops being inference-only and the two knobs stop being
        # independent.
        _, visited = self._trajectory(self._dit(), steps=4)
        assert not any(x.requires_grad for x, _, _ in visited)

    def test_identical_features_cost_nothing(self):
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=4)
        loss = rollout_loss(dit, visited, teacher, teacher.clone(), None, None, 0.0, 4,
                            random.Random(0))
        assert float(loss) == 0.0

    def test_a_different_student_costs_something_and_has_gradient(self):
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=4)
        student = torch.randn_like(teacher).requires_grad_(True)
        loss = rollout_loss(dit, visited, teacher, student, None, None, 0.0, 2, random.Random(0))
        loss.backward()
        assert float(loss) > 0
        assert student.grad is not None and torch.isfinite(student.grad).all()
        assert float(student.grad.norm()) > 0, 'the student must receive gradient from the rollout'

    def test_loss_points_bounds_how_many_predictions_are_compared(self):
        from tools.distill_refiner import rollout_loss
        import random

        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=8)
        student = torch.randn_like(teacher)

        calls = []
        original = dit.forward

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        dit.forward = counting
        rollout_loss(dit, visited, teacher, student, None, None, 0.0, 3, random.Random(0))
        dit.forward = original
        # Three points, STUDENT only. The teacher's velocity at each point was already computed
        # while walking the trajectory and is carried in the triple, so recomputing it here
        # would be a full redundant DiT forward per point.
        assert len(calls) == 3, len(calls)

    def test_the_teacher_velocity_is_reused_not_recomputed(self):
        # The regression guard for that: the target must be the value teacher_trajectory
        # produced, bit for bit, not a fresh forward that merely agrees with it.
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=4)
        student = torch.randn_like(teacher)

        captured = []
        original = F.mse_loss

        def capturing(prediction, target, **kwargs):
            captured.append(target)
            return original(prediction, target, **kwargs)

        F.mse_loss = capturing
        try:
            rollout_loss(dit, visited, teacher, student, None, None, 0.0, 2, random.Random(0))
        finally:
            F.mse_loss = original

        cached = {id(v) for _, _, v in visited}
        assert captured, 'no loss term was computed'
        assert all(id(t) in cached for t in captured), (
            'the target must BE the trajectory velocity object, not an equal-valued recompute'
        )

    def test_more_points_than_the_trajectory_has_is_not_an_error(self):
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=2)
        student = torch.randn_like(teacher)
        loss = rollout_loss(dit, visited, teacher, student, None, None, 0.0, 99, random.Random(0))
        assert torch.isfinite(loss)

    def test_guidance_doubles_the_predictions_and_still_trains(self):
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        uncond = torch.randn(2, 8, self.DIT_CONFIG['crossattn_emb_channels'])
        teacher, visited = self._trajectory(dit, steps=4, guidance_scale=3.0, uncond=uncond)
        student = torch.randn_like(teacher).requires_grad_(True)
        student_uncond = torch.randn_like(uncond).requires_grad_(True)

        loss = rollout_loss(dit, visited, teacher, student, uncond, student_uncond, 3.0, 2,
                            random.Random(0))
        loss.backward()
        assert torch.isfinite(loss)
        # Both branches must receive gradient: the unconditional one is what every CFG sample
        # uses, and it is the branch an empty caption feeds.
        assert float(student.grad.norm()) > 0
        assert float(student_uncond.grad.norm()) > 0

    def test_guidance_zero_never_evaluates_the_unconditional_branch(self):
        # Off must mean off: no extra forward, and no cost for a feature nobody enabled.
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=4)
        student = torch.randn_like(teacher)

        calls = []
        original = dit.forward
        dit.forward = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        rollout_loss(dit, visited, teacher, student, torch.randn_like(teacher),
                     torch.randn_like(teacher), 0.0, 2, random.Random(0))
        dit.forward = original
        assert len(calls) == 2, f'expected 2 points, student only, got {len(calls)}'

    def test_guiding_one_side_only_is_refused(self):
        # _velocity drops guidance when its uncond argument is None, so a caller that supplies
        # one side and not the other would get a silently biased comparison: the teacher guided,
        # the student not.
        from tools.distill_refiner import rollout_loss
        import random
        dit = self._dit()
        teacher, visited = self._trajectory(dit, steps=2)
        student = torch.randn_like(teacher)
        with pytest.raises(RuntimeError, match='both sides or neither'):
            rollout_loss(dit, visited, teacher, student, torch.randn_like(teacher), None,
                         3.0, 1, random.Random(0))


class TestRolloutIsOffByDefault:
    """The feature must be invisible until someone asks for it."""

    def test_the_default_config_does_not_enable_it(self):
        import toml
        config = toml.load(REPO / 'examples/anima_refiner/distill.toml')
        assert config.get('rollout', {}).get('loss_weight', 0.0) == 0.0, (
            'the shipped config must leave the rollout off; it costs several GB of resident DiT'
        )

    def test_build_teacher_discards_the_dit_when_it_is_off(self):
        # Reading the source rather than running build_teacher, which needs real checkpoints.
        source = (REPO / 'tools/distill_refiner.py').read_text(encoding='utf-8')
        body = source[source.index('def build_teacher'):source.index('def build_student')]
        assert "rollout_config.get('loss_weight', 0.0) > 0" in body
        assert 'dit.blocks = None' in body, 'the off path must still throw the DiT away'


class TestEveryShippedDistillKeyIsRead:
    """A key in the wrong table is invisible, and a matching default hides it completely.

    relational_loss_weight shipped inside [student] while the code reads it from
    config['distill']. It was therefore ignored, and only the fact that its default was also 1.0
    kept behaviour correct -- anyone who changed it would have seen no effect at all. Checking
    that a key merely appears somewhere in the source would not have caught that. This checks
    the TABLE each key is read from.
    """

    # Every shipped distillation config, so a new one cannot arrive unprotected.
    CONFIGS = ('distill.toml', 'distill_rollout.toml',
               'distill_4gpu.toml', 'distill_rollout_4gpu.toml')
    # [optimizer] is forwarded wholesale to the optimizer constructor, so its keys are named by
    # the optimizer rather than by this script.
    SKIP_TABLES = {'optimizer'}

    @staticmethod
    def _reads_by_table():
        """Extract which (table, key) pairs tools/distill_refiner.py actually reads.

        Recognises the four shapes the script uses:
            config['distill'].get('steps', ...)
            config['teacher']['llm_path']
            config.get('probe', {}).get('num_queries', ...)
            rollout_config.get('steps', ...)      # via `rollout_config = config.get('rollout', {})`
            setting('shuffle_tags', ...)          # caption_augment_config's [distill]-first helper
        """
        import ast as ast_module
        source = (REPO / 'tools/distill_refiner.py').read_text(encoding='utf-8')
        tree = ast_module.parse(source)

        def table_of(node):
            """The config table `node` evaluates to, if it plainly is one."""
            # config['distill']
            if (isinstance(node, ast_module.Subscript)
                    and isinstance(node.value, ast_module.Name) and node.value.id == 'config'
                    and isinstance(node.slice, ast_module.Constant)):
                return node.slice.value
            # config.get('probe', {})
            if (isinstance(node, ast_module.Call)
                    and isinstance(node.func, ast_module.Attribute) and node.func.attr == 'get'
                    and isinstance(node.func.value, ast_module.Name)
                    and node.func.value.id == 'config'
                    and node.args and isinstance(node.args[0], ast_module.Constant)):
                return node.args[0].value
            return None

        # Names bound to a config table, e.g. rollout_config = config.get('rollout', {}).
        aliases = {}
        for node in ast_module.walk(tree):
            if isinstance(node, ast_module.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                table = table_of(node.value)
                if table is not None and isinstance(target, ast_module.Name):
                    aliases[target.id] = table

        reads = {}
        for node in ast_module.walk(tree):
            table = key = None
            # <table>.get('key', ...)
            if (isinstance(node, ast_module.Call)
                    and isinstance(node.func, ast_module.Attribute) and node.func.attr == 'get'
                    and node.args and isinstance(node.args[0], ast_module.Constant)):
                key = node.args[0].value
                base = node.func.value
                table = table_of(base)
                if table is None and isinstance(base, ast_module.Name):
                    table = aliases.get(base.id)
            # config['teacher']['llm_path'], and distill_config['dataset'] where
            # distill_config was bound to config['distill'] earlier.
            elif isinstance(node, ast_module.Subscript) and isinstance(node.slice, ast_module.Constant):
                table = table_of(node.value)
                if table is None and isinstance(node.value, ast_module.Name):
                    table = aliases.get(node.value.id)
                key = node.slice.value
            # caption_augment_config reads its keys through a closure, `setting(key, default)`,
            # which indexes distill_config with a variable -- invisible to the shapes above. That
            # blind spot covered the whole caption-augmentation family, so a corpus-driven config
            # restating them under [distill], which is the documented thing to do, read as four
            # keys the script never touches.
            if (table is None and isinstance(node, ast_module.Call)
                    and isinstance(node.func, ast_module.Name) and node.func.id == 'setting'
                    and node.args and isinstance(node.args[0], ast_module.Constant)):
                table, key = 'distill', node.args[0].value
            if table and isinstance(key, str):
                reads.setdefault(table, set()).add(key)
        return reads

    def test_the_extractor_finds_the_known_reads(self):
        # If this ever comes back empty the test below would pass vacuously.
        reads = self._reads_by_table()
        assert 'steps' in reads.get('distill', set())
        assert 'llm_path' in reads.get('teacher', set())
        assert 'num_blocks' in reads.get('probe', set())
        assert 'loss_weight' in reads.get('rollout', set()), (
            'the alias tracking for rollout_config = config.get("rollout", {}) broke'
        )
        assert 'shuffle_tags' in reads.get('distill', set()), (
            "the setting() tracking for caption_augment_config's keys broke"
        )
        assert 'dataset' in reads.get('distill', set()), (
            'the alias tracking for distill_config = config["distill"] broke'
        )

    @pytest.mark.parametrize('filename', CONFIGS)
    def test_every_key_is_read_from_the_table_it_sits_in(self, filename):
        import toml
        config = toml.load(REPO / 'examples/anima_refiner' / filename)
        reads = self._reads_by_table()

        unread = []
        for table, values in config.items():
            if table in self.SKIP_TABLES or not isinstance(values, dict):
                continue
            for key in values:
                if key not in reads.get(table, set()):
                    unread.append(f'[{table}] {key}')
        assert not unread, (
            f'{filename} sets keys that tools/distill_refiner.py never reads from that table: '
            f'{unread}. Either the key is in the wrong table, or it does nothing.'
        )

    @pytest.mark.parametrize('filename', CONFIGS)
    def test_commented_out_keys_are_in_the_right_table_too(self, filename):
        """A commented key is an instruction to the reader, and it can point at the wrong table.

        The guard above cannot see one: toml.load discards comments, so `#resume_from = ...`
        sitting under [distill] passes it while inviting the user to uncomment a line the code
        reads from [student]. Uncommenting it then starts from a random refiner at step 0, with
        no warning, and the interrupted run's checkpoints get overwritten by the fresh one.
        This repo has shipped a key in the wrong table twice; both 4-GPU configs shipped this
        one commented.
        """
        import re
        reads = self._reads_by_table()
        table = None
        misplaced = []
        for line in (REPO / 'examples/anima_refiner' / filename).read_text(
                encoding='utf-8').splitlines():
            stripped = line.strip()
            # A commented header opens a commented table: distill.toml documents the whole
            # optional [rollout] and [optimizer] tables that way. Without following it, every
            # key inside gets attributed to the last real table and the guard cries wolf.
            header = re.fullmatch(r'#?\s*\[([a-z_.]+)\]', stripped)
            if header:
                table = header.group(1)
                continue
            commented = re.match(r'#\s*([a-z_][a-z0-9_]*)\s*=', stripped)
            if not commented or table in self.SKIP_TABLES or table is None:
                continue
            key = commented.group(1)
            # Only flag a key this script reads from somewhere: a commented key naming nothing
            # the code knows is prose, not a misplacement.
            known = {t for t, keys in reads.items() if key in keys}
            if known and table not in known:
                misplaced.append(f'[{table}] #{key} -- read from {sorted(known)}')
        assert not misplaced, (
            f'{filename} has commented-out keys under the wrong table: {misplaced}. '
            'Uncommenting one would silently do nothing.'
        )


class _AttrDict(dict):
    """Stands in for BatchEncoding, which supports both d['k'] and d.k."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class TestUnconditionalSetup:
    """The guidance branch was the least-tested surface in the feature.

    Nothing exercised the empty-caption tokenization through both frontends, the asymmetry
    between them, or that the frozen LLM's hidden state is produced once. Extracting
    build_unconditional_features from main() is what made any of it reachable.
    """

    class _Tok:
        """A tokenizer whose '' is an all-padding row, like Qwen's."""

        def __init__(self, real_tokens=0):
            self.real_tokens = real_tokens

        def __call__(self, prompts, **kwargs):
            length = kwargs['max_length']
            mask = torch.zeros(len(prompts), length, dtype=torch.long)
            for i, prompt in enumerate(prompts):
                n = self.real_tokens if prompt == '' else 3
                mask[i, :n] = 1
            out = {'input_ids': torch.zeros(len(prompts), length, dtype=torch.long),
                   'attention_mask': mask}
            return _AttrDict(out)

    @staticmethod
    def _build(monkeypatch, teacher_real_tokens=1):
        import tools.distill_refiner as module

        # encode() would run a real LLM; the identity of the hidden state is not what is under
        # test here, only that it is produced once and carries the right mask.
        monkeypatch.setattr(module, 'encode',
                            lambda llm, ids, mask, layer: torch.ones(ids.shape[0], ids.shape[1], 4))

        def adapter(source_hidden_states, target_input_ids, target_attention_mask,
                    source_attention_mask):
            return torch.ones(target_input_ids.shape[0], target_input_ids.shape[1], 5)

        return module.build_unconditional_features(
            teacher_tok=TestUnconditionalSetup._Tok(real_tokens=0),
            t5_tokenizer=TestUnconditionalSetup._Tok(real_tokens=teacher_real_tokens),
            teacher_llm=None, llm_adapter=adapter,
            student_tok=TestUnconditionalSetup._Tok(real_tokens=0),
            student_llm=None, max_text_length=8, device=torch.device('cpu'),
            llm_hidden_layer=None,
        )

    def test_the_student_gets_one_real_token_for_an_empty_caption(self, monkeypatch):
        # Without this the refiner emits zeros and the frozen DiT sees a context its original
        # T5 training never produced -- and the sample delivers no gradient at all.
        _, _, student_mask, _ = self._build(monkeypatch)
        assert student_mask.sum().item() == 1
        assert student_mask[0, 0].item() == 1

    def test_the_teacher_is_deliberately_not_given_one(self, monkeypatch):
        # The teacher's query sequence is old T5's, which already yields </s> for ''. Adding a
        # token there would change what the LLMAdapter path has always produced.
        teacher_uncond, _, _, _ = self._build(monkeypatch, teacher_real_tokens=1)
        # The adapter output is masked by the T5 mask; one real T5 token means one live row.
        live = (teacher_uncond.abs().sum(dim=-1) > 0).sum().item()
        assert live == 1, f'expected the T5 </s> row only, got {live}'

    def test_padded_positions_of_the_teacher_feature_are_zeroed(self, monkeypatch):
        teacher_uncond, _, _, _ = self._build(monkeypatch, teacher_real_tokens=2)
        assert torch.equal(teacher_uncond[0, 2:], torch.zeros_like(teacher_uncond[0, 2:]))

    def test_everything_comes_back_with_batch_one(self, monkeypatch):
        # The caller expands to the real batch; returning anything else would silently broadcast.
        teacher_uncond, ids, mask, hidden = self._build(monkeypatch)
        assert teacher_uncond.shape[0] == 1
        assert ids.shape[0] == 1 and mask.shape[0] == 1 and hidden.shape[0] == 1

    def test_the_student_hidden_state_is_produced_once(self, monkeypatch):
        import tools.distill_refiner as module
        calls = []
        monkeypatch.setattr(module, 'encode', lambda llm, ids, mask, layer: (
            calls.append(1), torch.ones(ids.shape[0], ids.shape[1], 4))[1])
        module.build_unconditional_features(
            teacher_tok=self._Tok(0), t5_tokenizer=self._Tok(1), teacher_llm=None,
            llm_adapter=lambda **kw: torch.ones(1, 8, 5),
            student_tok=self._Tok(0), student_llm=None, max_text_length=8,
            device=torch.device('cpu'), llm_hidden_layer=None,
        )
        # Once for the teacher LLM, once for the student LLM. Not once per micro-batch.
        assert len(calls) == 2, len(calls)

    def test_it_runs_under_no_grad(self, monkeypatch):
        # A frozen constant must not drag an autograd graph into every step.
        _, _, _, hidden = self._build(monkeypatch)
        assert not hidden.requires_grad


class TestShiftedSchedule:
    """The trajectory has to be warped the way training and sampling warp t, or the x_t it
    visits are not where the model is actually asked to predict."""

    def test_shift_one_leaves_it_uniform(self):
        from tools.distill_refiner import shifted_schedule
        schedule = shifted_schedule(4, 1.0, torch.device('cpu'))
        assert torch.allclose(schedule, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))

    def test_the_endpoints_are_fixed(self):
        # t=1 must stay pure noise and t=0 must stay clean for any shift.
        from tools.distill_refiner import shifted_schedule
        for shift in (0.5, 1.0, 3.0, 7.0):
            schedule = shifted_schedule(6, shift, torch.device('cpu'))
            assert abs(float(schedule[0]) - 1.0) < 1e-6, shift
            assert abs(float(schedule[-1])) < 1e-6, shift

    def test_it_stays_monotonically_decreasing(self):
        from tools.distill_refiner import shifted_schedule
        for shift in (0.5, 1.0, 3.0, 7.0):
            schedule = shifted_schedule(8, shift, torch.device('cpu'))
            assert all(a > b for a, b in zip(schedule, schedule[1:])), shift

    def test_a_shift_above_one_concentrates_steps_at_the_noisy_end(self):
        # Which is the whole point: it matches prepare_inputs' t*shift / (1 + (shift-1)*t).
        from tools.distill_refiner import shifted_schedule
        uniform = shifted_schedule(8, 1.0, torch.device('cpu'))
        shifted = shifted_schedule(8, 3.0, torch.device('cpu'))
        assert float(shifted[4]) > float(uniform[4])

    def test_it_matches_the_training_transform(self):
        # The same formula prepare_inputs applies (models/cosmos_predict2.py).
        from tools.distill_refiner import shifted_schedule
        shift = 3.0
        schedule = shifted_schedule(5, shift, torch.device('cpu'))
        uniform = torch.linspace(1.0, 0.0, 6)
        expected = (uniform * shift) / (1 + (shift - 1) * uniform)
        assert torch.allclose(schedule, expected, atol=1e-6)


class TestHiddenLayerMinusOneIsNormalised:
    """-1 asks for the last hidden state, which is what None already means.

    Asking by index takes the output_hidden_states branch, which materialises every layer's
    output -- 25 tensors for a 24-layer model -- and then indexes one. At -1 that tensor IS
    last_hidden_state, so the whole allocation is thrown away: roughly 420 MB per forward at
    B=8, L=512, d=2048 in bf16. Every shipped anima_refiner config uses -1.
    """

    def test_minus_one_becomes_none(self):
        from models.cosmos_predict2 import normalise_hidden_layer
        assert normalise_hidden_layer(-1) is None

    def test_none_stays_none(self):
        from models.cosmos_predict2 import normalise_hidden_layer
        assert normalise_hidden_layer(None) is None

    def test_a_real_index_is_left_alone(self):
        # 20 is the other candidate the docs suggest sweeping, and it is a genuinely different
        # tensor -- a raw residual-stream value rather than a post-final-RMSNorm one.
        from models.cosmos_predict2 import normalise_hidden_layer
        assert normalise_hidden_layer(20) == 20
        assert normalise_hidden_layer(-2) == -2

    def test_the_expensive_branch_is_skipped(self):
        # The behavioural half: with the normalised value, _compute_text_embeddings must not ask
        # for hidden states at all.
        from models.cosmos_predict2 import _compute_text_embeddings, normalise_hidden_layer

        seen = {}

        class Encoder:
            device = torch.device('cpu')

            def __call__(self, input_ids=None, attention_mask=None, **kwargs):
                seen['output_hidden_states'] = kwargs.get('output_hidden_states', False)
                hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
                return type('Out', (), {'last_hidden_state': hidden,
                                        'hidden_states': [hidden] * 25})()

        ids = torch.zeros(1, 4, dtype=torch.long)
        mask = torch.ones(1, 4, dtype=torch.long)
        _compute_text_embeddings(Encoder(), ids, mask,
                                 hidden_layer=normalise_hidden_layer(-1))
        assert seen['output_hidden_states'] is False, (
            'llm_hidden_layer = -1 must not materialise every hidden state to index the last one'
        )


class TestConfigIsRefusedBeforeAnythingExpensive:
    """Every refusal used to fire after the dataset walk and three models were loaded.

    A typo in `precision` cost minutes and tens of GB of I/O per rank before being reported. The
    checks are pure config, so they run first now.
    """

    @staticmethod
    def _config(**distill):
        base = {'output_dir': '/tmp/out'}
        base.update(distill)
        return {'distill': base}

    def test_an_unknown_strategy_is_refused(self):
        from tools.distill_refiner import validate_config_early
        with pytest.raises(RuntimeError, match='distributed_strategy'):
            validate_config_early(self._config(distributed_strategy='zero9'), 2, False)

    def test_zero_on_one_rank_is_refused(self):
        from tools.distill_refiner import validate_config_early
        with pytest.raises(RuntimeError, match='world_size=1'):
            validate_config_early(self._config(distributed_strategy='zero1'), 1, False)

    def test_an_unknown_precision_is_refused(self):
        from tools.distill_refiner import validate_config_early
        with pytest.raises(RuntimeError, match='precision'):
            validate_config_early(self._config(precision='bf8'), 1, False)

    def test_fp16_full_is_refused(self):
        from tools.distill_refiner import validate_config_early
        with pytest.raises(RuntimeError, match='fp16-full'):
            validate_config_early(self._config(precision='fp16-full'), 1, False)

    def test_the_optimizer_table_conflicting_with_flat_keys_is_refused(self):
        from tools.distill_refiner import validate_config_early
        config = self._config(lr=1e-4)
        config['optimizer'] = {'type': 'adamw', 'lr': 1e-4}
        with pytest.raises(RuntimeError, match='alternative'):
            validate_config_early(config, 1, False)

    def test_gradient_release_is_refused(self):
        from tools.distill_refiner import validate_config_early
        config = self._config()
        config['optimizer'] = {'type': 'adamw', 'gradient_release': True}
        with pytest.raises(RuntimeError, match='gradient_release'):
            validate_config_early(config, 1, False)

    def test_offload_with_the_weight_decay_split_is_refused(self):
        # CPUOffloadOptimizer takes its inner class positionally and does not accept parameter
        # groups, which is exactly what no_weight_decay_on_1d produces.
        from tools.distill_refiner import validate_config_early
        config = self._config(no_weight_decay_on_1d=True)
        config['optimizer'] = {'type': 'offload'}
        with pytest.raises(RuntimeError, match='offload'):
            validate_config_early(config, 1, False)

    def test_a_valid_config_returns_the_precision(self):
        from tools.distill_refiner import validate_config_early
        precision = validate_config_early(self._config(precision='bf16-full'), 1, False)
        assert precision.name == 'bf16-full'

    def test_it_runs_before_the_captions_are_loaded(self):
        # The ordering is the entire point; a later refactor could undo it silently.
        source = (REPO / 'tools/distill_refiner.py').read_text(encoding='utf-8')
        body = source[source.index('def main('):]
        assert body.index('validate_config_early(') < body.index('load_captions_once('), (
            'validation must come before the dataset walk'
        )
        assert body.index('validate_config_early(') < body.index('build_teacher('), (
            'validation must come before the teacher is loaded'
        )


class TestCaptionsAreResolvedOncePerJob:
    """Eight ranks walking the same tree at startup is eight clients hammering one filesystem."""

    def test_a_single_process_does_not_broadcast(self, monkeypatch):
        import tools.distill_refiner as module
        called = []
        monkeypatch.setattr(module, 'load_captions', lambda config: ['a', 'b'])
        monkeypatch.setattr(module.dist, 'broadcast_object_list',
                            lambda *a, **k: called.append(1))
        assert module.load_captions_once({}, rank=0, world_size=1, is_main=True) == ['a', 'b']
        assert not called, 'a single process has nobody to broadcast to'

    def test_only_rank_zero_resolves_them(self, monkeypatch):
        import tools.distill_refiner as module
        walks = []

        def fake_load(config):
            walks.append(1)
            return ['a', 'b']

        def fake_broadcast(payload, src=0):
            payload[0] = ['a', 'b']       # what rank 0 sent

        monkeypatch.setattr(module, 'load_captions', fake_load)
        monkeypatch.setattr(module.dist, 'broadcast_object_list', fake_broadcast)

        assert module.load_captions_once({}, rank=1, world_size=4, is_main=False) == ['a', 'b']
        assert not walks, 'a non-zero rank must not walk the dataset itself'

    def test_a_failed_broadcast_is_reported_rather_than_returning_nothing(self, monkeypatch):
        import tools.distill_refiner as module
        monkeypatch.setattr(module, 'load_captions', lambda config: ['a'])
        monkeypatch.setattr(module.dist, 'broadcast_object_list', lambda payload, src=0: None)
        with pytest.raises(RuntimeError, match='received no captions'):
            module.load_captions_once({}, rank=2, world_size=4, is_main=False)


class TestEpochSampler:
    """An epoch has to mean one pass over every caption, or the word is decoration.

    The loop used to draw random.sample(captions, batch_size) per micro batch -- sampling with
    replacement across steps, so some captions appear many times before others appear once.
    """

    CAPTIONS = [f'caption {i}' for i in range(100)]

    def _samplers(self, world_size=4, batch_size=2, grad_accum=2, seed=42):
        from tools.distill_refiner import EpochSampler
        return [EpochSampler(self.CAPTIONS, batch_size, grad_accum, rank, world_size, seed)
                for rank in range(world_size)]

    def test_steps_per_epoch_drops_the_partial_tail(self):
        # Same rounding SizeBucketDataset does. Every rank must run the same number of steps or
        # a collective waits forever on one that finished early.
        samplers = self._samplers()
        assert samplers[0].global_batch == 16
        assert samplers[0].steps_per_epoch == 6      # 100 // 16, the last 4 dropped

    def test_the_ranks_do_not_overlap(self):
        shards = [s.epoch_order(0) for s in self._samplers()]
        flat = [caption for shard in shards for caption in shard]
        assert len(set(flat)) == len(flat), 'a caption appeared on two ranks in one epoch'

    def test_together_they_cover_the_usable_captions(self):
        shards = [s.epoch_order(0) for s in self._samplers()]
        flat = {caption for shard in shards for caption in shard}
        assert len(flat) == 6 * 16, 'an epoch must be one pass over every usable caption'

    def test_the_shards_are_equal_sized(self):
        shards = [s.epoch_order(0) for s in self._samplers()]
        assert len({len(shard) for shard in shards}) == 1, [len(s) for s in shards]

    def test_a_later_epoch_reshuffles(self):
        # Sharding a fixed order would pin each rank to the same captions forever, which is the
        # reason the shuffle happens before the shard rather than after.
        sampler = self._samplers()[0]
        assert sampler.epoch_order(0) != sampler.epoch_order(1)

    def test_every_rank_sees_new_captions_in_a_later_epoch(self):
        first = set(self._samplers()[2].epoch_order(0))
        second = set(self._samplers()[2].epoch_order(1))
        assert first != second, 'rank 2 got the same captions in both epochs'

    def test_it_is_deterministic(self):
        # Resuming mid-epoch relies on the order being a pure function of (seed, epoch) rather
        # than of how many draws have happened, so nothing about it needs saving.
        assert self._samplers()[1].epoch_order(3) == self._samplers()[1].epoch_order(3)

    def test_a_different_seed_gives_a_different_order(self):
        assert self._samplers(seed=42)[0].epoch_order(0) != self._samplers(seed=43)[0].epoch_order(0)

    def test_a_single_process_uses_everything_it_can(self):
        sampler = self._samplers(world_size=1, batch_size=10, grad_accum=1)[0]
        assert sampler.steps_per_epoch == 10
        assert len(sampler.epoch_order(0)) == 100

    def test_too_few_captions_is_refused_with_the_arithmetic(self):
        from tools.distill_refiner import EpochSampler
        with pytest.raises(RuntimeError, match='cannot fill one global batch'):
            EpochSampler(['only', 'three', 'captions'], batch_size=8, grad_accum=4,
                         rank=0, world_size=2, seed=0)


class TestEpochsAndStepsAreAlternatives:
    """Both are ways of saying how long to train. Setting both is a question with no answer."""

    def test_the_shipped_4gpu_configs_use_epochs(self):
        import toml
        for name in ('distill_4gpu.toml', 'distill_rollout_4gpu.toml'):
            config = toml.load(REPO / 'examples/anima_refiner' / name)['distill']
            assert 'epochs' in config, name
            assert 'steps' not in config, f'{name} sets both epochs and steps'

    def test_the_older_configs_still_use_steps(self):
        # Backward compatibility is the point: configs written before epochs existed must run
        # unchanged.
        import toml
        for name in ('distill.toml', 'distill_rollout.toml'):
            config = toml.load(REPO / 'examples/anima_refiner' / name)['distill']
            assert 'steps' in config, name
            assert 'epochs' not in config, name

    def test_setting_both_is_refused(self):
        source = (REPO / 'tools/distill_refiner.py').read_text(encoding='utf-8')
        assert 'sets both epochs and steps' in source

    def test_neither_still_defaults_to_steps(self):
        # A config predating this feature sets neither explicitly in some cases; it must not
        # start training for zero steps.
        source = (REPO / 'tools/distill_refiner.py').read_text(encoding='utf-8')
        assert 'if epochs is None and steps is None:' in source
        assert 'steps = 20000' in source


class TestResumeRestoresTheAugmentationStream:
    """The optimizer came back; the RNG did not, so a resumed run saw different augmentations."""

    @staticmethod
    def _optimizer():
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    def test_the_python_stream_continues_where_it_left_off(self, tmp_path):
        import random as py_random
        from tools.distill_refiner import save_training_state, load_training_state

        opt, sched = self._optimizer()
        py_random.seed(7)
        [py_random.random() for _ in range(5)]
        expected = [py_random.random() for _ in range(3)]

        py_random.seed(7)
        [py_random.random() for _ in range(5)]
        save_training_state(tmp_path / 'context_refiner.safetensors', opt, sched, 10)

        [py_random.random() for _ in range(100)]        # drift the stream
        fresh_opt, fresh_sched = self._optimizer()
        load_training_state(tmp_path / 'context_refiner.safetensors', fresh_opt, fresh_sched,
                            is_main=False)
        assert [py_random.random() for _ in range(3)] == expected

    def test_the_torch_stream_continues_too(self, tmp_path):
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        torch.manual_seed(11)
        expected = torch.randn(4)

        torch.manual_seed(11)
        save_training_state(tmp_path / 'context_refiner.safetensors', opt, sched, 10)
        torch.randn(100)
        fresh_opt, fresh_sched = self._optimizer()
        load_training_state(tmp_path / 'context_refiner.safetensors', fresh_opt, fresh_sched,
                            is_main=False)
        assert torch.equal(torch.randn(4), expected)

    def test_a_checkpoint_without_rng_state_still_resumes(self, tmp_path):
        # Written before the RNG was recorded. It cannot reproduce the augmentation, but it must
        # not refuse to load.
        from tools.distill_refiner import load_training_state, training_state_path
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner.safetensors'
        torch.save({'step': 42, 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict()},
                   training_state_path(path))
        fresh_opt, fresh_sched = self._optimizer()
        assert load_training_state(path, fresh_opt, fresh_sched, is_main=False) == 42


class TestResolveSchedule:
    """The step count and the sampler are one decision, and reading one without the other broke.

    The derivation used to sit inline in main() below the optimizer, so the LR scheduler was
    built holding total_steps=None whenever `epochs` was used. It constructed without complaint
    and raised TypeError at the end of warmup -- 500 steps into a four-GPU run.
    """

    CAPTIONS = [f'caption {i}' for i in range(1000)]

    def _resolve(self, epochs, steps, world_size=4, batch_size=2, grad_accum=2):
        from tools.distill_refiner import resolve_schedule
        return resolve_schedule(epochs, steps, self.CAPTIONS, batch_size, grad_accum,
                                rank=0, world_size=world_size, seed=42)

    def test_epochs_produces_an_integer_step_count(self):
        # The regression itself: with epochs set, `steps` must come back a usable number, never
        # the None it was passed in as.
        sampler, steps, _ = self._resolve(epochs=20, steps=None)
        assert isinstance(steps, int)
        assert steps == 20 * sampler.steps_per_epoch

    def test_the_step_count_it_returns_builds_a_working_scheduler(self):
        # The end the bug was actually felt at. total_steps=None survives construction, so
        # asserting the scheduler exists proves nothing -- it has to be stepped past warmup.
        import torch
        from utils.lr_schedule import create_lr_scheduler
        _, steps, _ = self._resolve(epochs=20, steps=None)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        warmup = 10
        scheduler = create_lr_scheduler(optimizer, 'cosine', total_steps=steps,
                                        warmup_steps=warmup)
        for _ in range(warmup + 5):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]['lr'] > 0

    def test_steps_passes_straight_through(self):
        _, steps, _ = self._resolve(epochs=None, steps=12345)
        assert steps == 12345

    def test_it_reports_the_conversion_both_ways(self):
        _, _, from_epochs = self._resolve(epochs=20, steps=None)
        assert 'epochs' in from_epochs and 'steps' in from_epochs
        _, _, from_steps = self._resolve(epochs=None, steps=12345)
        assert 'epochs' in from_steps and '12345' in from_steps

    def test_it_refuses_a_caption_set_that_cannot_fill_a_batch(self):
        from tools.distill_refiner import resolve_schedule
        with pytest.raises(RuntimeError, match='cannot fill one global batch'):
            resolve_schedule(20, None, ['one', 'two'], batch_size=8, grad_accum=4,
                             rank=0, world_size=2, seed=0)


class TestShardedTrainingState:
    """ZeRO partitions optimizer state across ranks, so rank 0's copy is not the state.

    deepspeed.initialize replaces the client optimizer's param_groups with one flat fp32
    partition per group (stage_1_and_2.py:473). Saving only rank 0's view wrote a file that
    looked valid and cost nothing, and the resume failed hours later with a torch ValueError
    about mismatched parameter groups.
    """

    @staticmethod
    def _optimizer():
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    def test_a_shard_is_named_apart_from_whole_state(self):
        from tools.distill_refiner import training_state_path
        weights = Path('/out/context_refiner_epoch7.safetensors')
        assert training_state_path(weights).name == 'distill_state_epoch7.pt'
        assert training_state_path(weights, rank=3).name == 'distill_state_epoch7_rank3.pt'

    def test_a_shard_round_trips(self, tmp_path):
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner_epoch1.safetensors'
        save_training_state(path, opt, sched, 40, rank=2, world_size=4)
        assert (tmp_path / 'distill_state_epoch1_rank2.pt').exists()
        fresh_opt, fresh_sched = self._optimizer()
        assert load_training_state(path, fresh_opt, fresh_sched, is_main=False,
                                   rank=2, world_size=4) == 40

    def test_resuming_into_a_different_world_size_is_refused(self, tmp_path):
        # A shard describes one partition of a particular world. Loading it into a job of
        # another size pairs Adam's moments with the wrong parameters, and nothing downstream
        # would notice.
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner_epoch1.safetensors'
        save_training_state(path, opt, sched, 40, rank=0, world_size=4)
        fresh_opt, fresh_sched = self._optimizer()
        with pytest.raises(RuntimeError, match='4-rank job'):
            load_training_state(path, fresh_opt, fresh_sched, is_main=False,
                                rank=0, world_size=8)

    def test_a_ddp_state_file_is_not_loaded_by_a_zero_resume(self, tmp_path):
        # Switching distributed_strategy between runs must not silently feed whole state to a
        # sharded optimizer; it resumes the weights only and says so.
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner_epoch1.safetensors'
        save_training_state(path, opt, sched, 40)
        fresh_opt, fresh_sched = self._optimizer()
        assert load_training_state(path, fresh_opt, fresh_sched, is_main=False,
                                   rank=0, world_size=4) == 0

    def test_ddp_state_is_unchanged(self, tmp_path):
        # Backward compatibility: rank=None must produce exactly the old filename and content.
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner.safetensors'
        save_training_state(path, opt, sched, 7)
        assert (tmp_path / 'distill_state.pt').exists()
        fresh_opt, fresh_sched = self._optimizer()
        assert load_training_state(path, fresh_opt, fresh_sched, is_main=False) == 7


class TestResumeStateCompleteness:
    """Everything a resumed run needs that is not the weights, and the pairings that got it wrong."""

    @staticmethod
    def _optimizer(lr=0.1):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    def test_a_tagged_full_model_finds_its_own_state_file(self):
        # save_full_model writes model_epoch7.safetensors and the docs offer it as a rollback
        # target. Matching only 'context_refiner' sent it to the untagged distill_state.pt --
        # epoch-7 weights paired with the newest moments and the newest step counter, silently.
        from tools.distill_refiner import training_state_path
        assert training_state_path(Path('/out/model_epoch7.safetensors')).name == \
            'distill_state_epoch7.pt'
        assert training_state_path(Path('/out/model_step900.safetensors')).name == \
            'distill_state_step900.pt'
        assert training_state_path(Path('/out/model.safetensors')).name == 'distill_state.pt'

    def test_the_tag_just_written_is_never_pruned(self, tmp_path):
        # Tag numbers only increase within one uninterrupted run. A second run into the same
        # output_dir, or a resume onto fewer ranks, can write a lower number than what is there
        # -- and the prune then deleted the checkpoint the run had just written, announcing it
        # as a routine eviction.
        from tools.distill_refiner import prune_distill_checkpoints
        for n in (4, 5, 6, 7):
            (tmp_path / f'context_refiner_epoch{n}.safetensors').write_bytes(b'w')
            (tmp_path / f'distill_state_epoch{n}.pt').write_bytes(b's')
        removed = [p.name for p in prune_distill_checkpoints(tmp_path, 3, protect_tag='_epoch4')]
        assert 'context_refiner_epoch4.safetensors' not in removed
        assert (tmp_path / 'context_refiner_epoch4.safetensors').exists()

    def test_without_the_guard_the_lowest_tag_still_goes(self, tmp_path):
        # The protection is scoped to the current tag, not a blanket "never prune the lowest".
        from tools.distill_refiner import prune_distill_checkpoints
        for n in (4, 5, 6, 7):
            (tmp_path / f'context_refiner_epoch{n}.safetensors').write_bytes(b'w')
        removed = [p.name for p in prune_distill_checkpoints(tmp_path, 3, protect_tag='_epoch7')]
        assert 'context_refiner_epoch4.safetensors' in removed

    def test_a_ddp_resume_does_not_push_rank0s_stream_onto_every_rank(self, tmp_path):
        # main() seeds random.seed(seed + rank) so the ranks draw different caption
        # augmentations. Only rank 0 writes a DDP state file; restoring it everywhere would
        # collapse that offset and correlate the augmentation noise for the rest of the run.
        import random as py_random
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner.safetensors'
        py_random.seed(0)
        save_training_state(path, opt, sched, 10)

        py_random.seed(999)                       # stand in for a non-zero rank's stream
        mine = py_random.getstate()
        fresh_opt, fresh_sched = self._optimizer()
        load_training_state(path, fresh_opt, fresh_sched, is_main=False, own_python_rng=False)
        assert py_random.getstate() == mine, 'rank 0 stream overwrote this rank'

    def test_the_owning_rank_still_gets_its_stream_back(self, tmp_path):
        import random as py_random
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        path = tmp_path / 'context_refiner.safetensors'
        py_random.seed(0)
        expected = [py_random.random() for _ in range(3)]
        py_random.seed(0)
        save_training_state(path, opt, sched, 10)
        py_random.seed(12345)
        fresh_opt, fresh_sched = self._optimizer()
        load_training_state(path, fresh_opt, fresh_sched, is_main=False, own_python_rng=True)
        assert [py_random.random() for _ in range(3)] == expected

    def test_the_rollout_streams_come_back(self, tmp_path):
        # Both are seeded apart from the global stream so that enabling the rollout does not
        # shift the caption augmentation; they need saving for the same reason it does.
        import random as py_random
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        generator = torch.Generator(device='cpu').manual_seed(7)
        rng = py_random.Random(11)
        path = tmp_path / 'context_refiner.safetensors'
        save_training_state(path, opt, sched, 10, rollout_generator=generator, rollout_rng=rng)
        expected_noise = torch.randn(4, generator=generator)
        expected_draw = rng.random()

        generator.manual_seed(999)
        drifted = py_random.Random(999)
        fresh_opt, fresh_sched = self._optimizer()
        load_training_state(path, fresh_opt, fresh_sched, is_main=False,
                            rollout_generator=generator, rollout_rng=drifted)
        assert torch.equal(torch.randn(4, generator=generator), expected_noise)
        assert drifted.random() == expected_draw

    def test_the_fp16_loss_scaler_comes_back(self, tmp_path):
        from tools.distill_refiner import save_training_state, load_training_state
        opt, sched = self._optimizer()
        scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=1024.0)
        path = tmp_path / 'context_refiner.safetensors'
        save_training_state(path, opt, sched, 10, scaler=scaler)
        fresh = torch.amp.GradScaler('cpu', enabled=True, init_scale=65536.0)
        fresh_opt, fresh_sched = self._optimizer()
        load_training_state(path, fresh_opt, fresh_sched, is_main=False, scaler=fresh)
        assert fresh.get_scale() == 1024.0

    def test_the_restored_learning_rate_is_live_immediately(self, tmp_path):
        # load_state_dict restores the schedule's position but leaves the optimizer holding the
        # LR it was built with until the next step() -- one step at the start of a warmup, which
        # is near zero.
        from utils.lr_schedule import create_lr_scheduler
        from tools.distill_refiner import save_training_state, load_training_state
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        sched = create_lr_scheduler(opt, 'cosine', total_steps=1000, warmup_steps=100)
        for _ in range(300):
            opt.step()
            sched.step()
        expected = opt.param_groups[0]['lr']
        path = tmp_path / 'context_refiner.safetensors'
        save_training_state(path, opt, sched, 300)

        fresh_model = torch.nn.Linear(2, 2)
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=5e-5)
        fresh_sched = create_lr_scheduler(fresh_opt, 'cosine', total_steps=1000, warmup_steps=100)
        load_training_state(path, fresh_opt, fresh_sched, is_main=False)
        assert fresh_opt.param_groups[0]['lr'] == pytest.approx(expected)


class TestWarmupAdvice:
    """The default warmup was chosen for a 20,000-step run; `epochs` derives the run length."""

    def test_a_warmup_longer_than_the_run_is_called_out(self):
        from tools.distill_refiner import warmup_advice
        message = warmup_advice(500, 400)
        assert message and 'never starts' in message

    def test_a_warmup_equal_to_the_run_is_still_broken(self):
        # SequentialLR's milestone is reached only by stepping past it, so equality is the
        # boundary case where the decay phase gets exactly zero steps.
        from tools.distill_refiner import warmup_advice
        assert warmup_advice(500, 500) is not None

    def test_a_large_fraction_is_called_out_with_a_number_to_use(self):
        # 20 epochs over 100k captions at a global batch of 1536 is 1300 steps, where the
        # default 500 is 38% of the schedule. Nothing errors; the run just spends a third of
        # itself ramping.
        from tools.distill_refiner import warmup_advice
        message = warmup_advice(500, 1300)
        assert message and '38%' in message and '65' in message

    def test_a_normal_warmup_says_nothing(self):
        from tools.distill_refiner import warmup_advice
        assert warmup_advice(500, 20000) is None


class TestSchedulerFollowsRealUpdates:
    """An overflowed fp16 step updates nothing, so it must not advance the LR schedule.

    GradScaler.step() silently skips the optimizer when it finds inf or nan, and update() lowers
    the loss scale when that happens. DeepSpeed gates its own scheduler on exactly this
    condition, so leaving the DDP path ungated made the two strategies walk different LR curves
    under the same `precision` setting.

    The scaler is stubbed because a real one disables itself without CUDA, which would make the
    skip path unreachable here. What is under test is this code's decision, not torch's.
    """

    class _Scaler:
        def __init__(self, scales):
            self.scales = list(scales)
            self.stepped = 0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.scales[0]

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            self.stepped += 1

        def update(self):
            # A real scaler backs the scale off when it skipped; mimic that by advancing.
            if len(self.scales) > 1:
                self.scales.pop(0)

    def _strategy(self, scales):
        from tools.distill_refiner import DDPStrategy
        model = torch.nn.Linear(4, 4)
        model(torch.randn(2, 4)).sum().backward()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        strategy = object.__new__(DDPStrategy)
        strategy.refiner = model
        strategy.optimizer = opt
        strategy.scheduler = sched
        strategy.max_grad_norm = 1.0
        strategy.scaler = self._Scaler(scales)
        return strategy, sched

    def test_a_skipped_step_does_not_advance_the_schedule(self):
        strategy, sched = self._strategy([65536.0, 32768.0])  # scale backed off: step was skipped
        strategy.step()
        assert sched.last_epoch == 0, 'the LR schedule advanced past a step that did not happen'

    def test_a_normal_step_still_advances_the_schedule(self):
        strategy, sched = self._strategy([65536.0])  # scale unchanged: step was applied
        strategy.step()
        assert sched.last_epoch == 1


class TestResumeReportsScheduleDrift:
    """DDP resume at a different global batch loads fine but does not line up. Say so.

    The optimizer state is replicated under DDP, so it loads correctly at any world size and
    refusing would block a legitimate resize. What does not carry across is the alignment: the
    saved step counts global batches, so a different global batch puts it elsewhere in the
    corpus, and under `epochs` the schedule's total moves with it.
    """

    def _round_trip(self, tmp_path, saved, current, capsys):
        from tools.distill_refiner import load_training_state, save_training_state
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        refiner_path = tmp_path / 'context_refiner.safetensors'
        save_training_state(refiner_path, opt, sched, 10, world_size=saved['world_size'],
                            batch_size=saved['batch_size'], grad_accum=saved['grad_accum'],
                            precision_name=saved['precision'])
        capsys.readouterr()
        step = load_training_state(refiner_path, opt, sched, is_main=True,
                                   world_size=current['world_size'],
                                   batch_size=current['batch_size'],
                                   grad_accum=current['grad_accum'],
                                   precision_name=current['precision'])
        return step, capsys.readouterr().out

    BASE = {'world_size': 2, 'batch_size': 8, 'grad_accum': 1, 'precision': 'bf16-mixed'}

    def test_an_unchanged_resume_says_nothing(self, tmp_path, capsys):
        step, out = self._round_trip(tmp_path, self.BASE, self.BASE, capsys)
        assert step == 10
        assert 'WARNING' not in out

    def test_a_changed_global_batch_warns_but_still_resumes(self, tmp_path, capsys):
        step, out = self._round_trip(tmp_path, self.BASE, {**self.BASE, 'world_size': 4}, capsys)
        assert step == 10, 'DDP state is replicated, so it must still load'
        assert 'world_size 2 -> 4' in out

    def test_a_changed_precision_warns(self, tmp_path, capsys):
        _, out = self._round_trip(tmp_path, self.BASE, {**self.BASE, 'precision': 'fp32'}, capsys)
        assert "'bf16-mixed'" in out and "'fp32'" in out


class TestBf16FullNeedsKahan:
    """bf16 parameters with no master copy need a compensated optimizer, or updates round away.

    Not refused, because it is a legitimate way to fit a larger batch and the shipped 4-GPU
    configs use it -- but they pair it with adamw8bitkahan, which is what makes it safe.
    """

    @staticmethod
    def _warn(distill, optimizer, capsys):
        from tools.distill_refiner import validate_config_early
        config = {'distill': dict(distill)}
        if optimizer is not None:
            config['optimizer'] = optimizer
        validate_config_early(config, world_size=1, is_main=True)
        return capsys.readouterr().out

    def test_bf16_full_with_the_default_optimizer_warns(self, capsys):
        out = self._warn({'precision': 'bf16-full'}, None, capsys)
        assert 'adamw8bitkahan' in out and 'WARNING' in out

    def test_bf16_full_with_plain_adamw8bit_warns(self, capsys):
        """Same memory as the Kahan variant, none of the compensation -- the trap."""
        out = self._warn({'precision': 'bf16-full'}, {'type': 'adamw8bit'}, capsys)
        assert 'adamw8bitkahan' in out and 'WARNING' in out

    def test_bf16_full_with_kahan_is_silent(self, capsys):
        out = self._warn({'precision': 'bf16-full'}, {'type': 'AdamW8bitKahan'}, capsys)
        assert 'WARNING' not in out

    def test_other_precisions_keep_their_fp32_parameters_and_do_not_warn(self, capsys):
        for name in ('fp32', 'bf16-mixed'):
            out = self._warn({'precision': name}, {'type': 'adamw8bit'}, capsys)
            assert 'WARNING' not in out, f'{name} keeps fp32 parameters; nothing rounds away'


class TestCorpusCaptionSettingsWarning:
    """A corpus inherits nothing, so anything not restated is silently off."""

    @staticmethod
    def _warn(distill_config, capsys):
        from tools.distill_refiner import caption_augment_config
        caption_augment_config({'distill': distill_config})
        return capsys.readouterr().out

    def test_restating_nothing_warns(self, capsys):
        out = self._warn({'caption_corpus': '/data/captions.jsonl'}, capsys)
        assert 'prefix_tag_caption' in out and 'WARNING' in out

    def test_restating_something_else_still_warns_about_the_marker(self, capsys):
        """The marker is the setting whose absence corrupts the caption, not just changes it."""
        out = self._warn(
            {'caption_corpus': '/data/captions.jsonl', 'caption_prefix': 'anime, '}, capsys)
        assert 'prefix_tag_caption' in out, (
            'restating one setting silenced the warning entirely, so a config that set '
            'caption_prefix and forgot the marker trained the marker as a tag with no warning'
        )

    def test_restating_the_marker_is_silent(self, capsys):
        out = self._warn(
            {'caption_corpus': '/data/captions.jsonl', 'prefix_tag_caption': 'Special: '}, capsys)
        assert 'WARNING' not in out

    def test_a_dataset_source_never_warns(self, capsys, tmp_path):
        dataset_toml = tmp_path / 'dataset.toml'
        dataset_toml.write_text('resolutions = [512]\n', encoding='utf-8')
        out = self._warn({'dataset': str(dataset_toml)}, capsys)
        assert 'WARNING' not in out


class TestGradScalerPathAgainstARealScaler:
    """Drive DDPStrategy.step()'s fp16 branch with a real GradScaler, not a fake one.

    Every fp16-mixed case elsewhere in this file runs the fp32 path on a CPU box, because
    `torch.amp.GradScaler('cuda')` disables itself when CUDA is absent and Precision.autocast
    returns a null context off-CUDA. The branch that unscales before clipping and gates the
    scheduler on the scale not dropping was therefore never executed here, and the tests that
    look like they cover it assert against a hand-written stand-in.

    A CPU GradScaler is fully functional in this torch, so the branch can be exercised for real.
    What remains CUDA-only is the numerics -- whether fp16 actually overflows on a given kernel --
    not this control flow.
    """

    @staticmethod
    def _strategy(max_grad_norm=1e9):
        from tools.distill_refiner import DDPStrategy
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        strategy = object.__new__(DDPStrategy)
        strategy.refiner = model
        strategy.optimizer = optimizer
        strategy.scheduler = scheduler
        strategy.max_grad_norm = max_grad_norm
        strategy.scaler = torch.amp.GradScaler('cpu', enabled=True)
        assert strategy.scaler.is_enabled(), 'this test is pointless with an inert scaler'
        return strategy, model, scheduler

    def test_the_reported_norm_is_unscaled(self):
        """unscale_ must precede clipping, or max_grad_norm is compared against a norm inflated
        by the loss scale and effectively never fires."""
        strategy, model, _ = self._strategy()
        x = torch.randn(2, 4)
        strategy.scaler.scale(model(x).sum()).backward()
        scale = strategy.scaler.get_scale()
        assert scale > 1.0

        # What is sitting in .grad right now is `scale` times the true gradient.
        scaled_norm = torch.linalg.vector_norm(
            torch.cat([p.grad.flatten() for p in model.parameters()])).item()
        reported = strategy.step()

        # step() unscales before clipping, so the norm it reports is the true one. Without the
        # unscale it would report `scaled_norm`, which is `scale` times larger.
        assert reported == pytest.approx(scaled_norm / scale, rel=1e-3), (
            f'reported {reported}, expected the unscaled {scaled_norm / scale} '
            f'(scaled norm {scaled_norm}, loss scale {scale})'
        )

    def test_an_overflowing_step_is_skipped_and_the_schedule_holds(self):
        """The real scaler's own skip behaviour, not a stand-in's."""
        strategy, model, scheduler = self._strategy()
        before = [p.detach().clone() for p in model.parameters()]
        strategy.scaler.scale(model(torch.randn(2, 4)).sum()).backward()
        # What an fp16 overflow produces.
        for p in model.parameters():
            p.grad[0, 0] = float('inf')

        scale_before = strategy.scaler.get_scale()
        strategy.step()

        assert strategy.scaler.get_scale() < scale_before, (
            'the real scaler should back the scale off after a non-finite gradient'
        )
        assert scheduler.last_epoch == 0, (
            'the LR schedule advanced past an optimizer step that never happened'
        )
        for p, was in zip(model.parameters(), before):
            assert torch.equal(p.detach(), was), 'weights moved on a skipped step'

    def test_a_finite_step_updates_and_advances(self):
        strategy, model, scheduler = self._strategy()
        before = [p.detach().clone() for p in model.parameters()]
        strategy.scaler.scale(model(torch.randn(2, 4)).sum()).backward()

        strategy.step()

        assert scheduler.last_epoch == 1, 'a real step must advance the schedule'
        assert any(not torch.equal(p.detach(), was)
                   for p, was in zip(model.parameters(), before)), 'weights did not move'


class TestZeROMasterWeightsSurviveAResume:
    """Under ZeRO the module is bit16 and the only full-precision weights are the optimizer's.

    deepspeed.initialize casts the module to bf16 and keeps fp32 masters in the flat partition it
    installs as the client optimizer's param_groups. save_refiner writes the module, so it writes
    the bit16 view, and optimizer.state_dict() carries the moments but never the parameter values.
    Without the masters in the payload a resumed bf16-full run silently restarts them from
    8-mantissa-bit values.
    """

    def test_the_masters_are_written_and_put_back(self, tmp_path):
        pytest.importorskip('deepspeed')
        from tools.distill_refiner import save_training_state, _restore_master_weights

        # Stand in for the optimizer deepspeed hands back: param_groups holding fp32 masters.
        master = torch.tensor([1.0000001, 2.0000002, 3.0000003], dtype=torch.float32)
        optimizer = torch.optim.SGD([torch.nn.Parameter(master.clone())], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

        refiner_path = tmp_path / 'context_refiner.safetensors'
        save_training_state(refiner_path, optimizer, scheduler, step=10, rank=0, world_size=1,
                            master_weights=True)

        state = torch.load(str(_state_file(tmp_path)), weights_only=False)
        assert state['master_weights'] is not None, 'the masters must be in the payload'

        # A fresh run: masters rebuilt from a bf16 round trip, which is what losing them costs.
        degraded = master.to(torch.bfloat16).to(torch.float32)
        assert not torch.equal(degraded, master), 'bf16 must actually lose bits here'
        optimizer.param_groups[0]['params'][0].data = degraded.clone()

        _restore_master_weights(state, optimizer, is_main=True)
        restored = optimizer.param_groups[0]['params'][0].detach()
        assert torch.equal(restored, master), (
            f'the fp32 masters were not restored: {restored.tolist()} != {master.tolist()}'
        )

    def test_a_checkpoint_without_masters_resumes_unchanged(self, tmp_path):
        """DDP and fp32 runs write nothing here, and must not be disturbed by the restore."""
        from tools.distill_refiner import _restore_master_weights
        value = torch.tensor([5.0, 6.0])
        optimizer = torch.optim.SGD([torch.nn.Parameter(value.clone())], lr=0.1)
        _restore_master_weights({'master_weights': None}, optimizer, is_main=True)
        assert torch.equal(optimizer.param_groups[0]['params'][0].detach(), value)

    def test_a_master_count_mismatch_warns_instead_of_corrupting(self, tmp_path, capsys):
        from tools.distill_refiner import _restore_master_weights
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
        _restore_master_weights({'master_weights': [torch.zeros(2), torch.zeros(2)]},
                                optimizer, is_main=True)
        assert 'WARNING' in capsys.readouterr().out

    def test_a_shape_mismatch_restores_nothing_at_all(self, capsys):
        """Partially restoring is worse than not restoring: some partitions would be from one
        step and the rest from another, with no error."""
        from tools.distill_refiner import _restore_master_weights
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.SGD([first, second], lr=0.1)
        # The first tensor matches and the second does not.
        _restore_master_weights({'master_weights': [torch.ones(2), torch.ones(9)]},
                                optimizer, is_main=True)
        assert 'WARNING' in capsys.readouterr().out
        assert torch.equal(first.detach(), torch.zeros(2)), (
            'the matching partition was written before the mismatch was noticed'
        )


def _state_file(directory):
    """The single distill_state_*.pt a rank-0 save writes into `directory`."""
    matches = sorted(directory.glob('distill_state*.pt'))
    assert len(matches) == 1, f'expected one state file, found {[p.name for p in matches]}'
    return matches[0]


class TestCheckpointHalvesStayTogether:
    """The weights and the optimizer state are two files; they must describe the same step."""

    @staticmethod
    def _write_refiner(path, step):
        import safetensors.torch
        metadata = {'format': 'pt'}
        if step is not None:
            metadata['step'] = str(step)
        safetensors.torch.save_file({'a': torch.zeros(2)}, str(path), metadata=metadata)

    def test_a_step_mismatch_is_refused(self, tmp_path):
        from tools.distill_refiner import _check_weights_match_training_state
        path = tmp_path / 'context_refiner.safetensors'
        self._write_refiner(path, 4000)
        with pytest.raises(RuntimeError, match='two halves of different checkpoints'):
            _check_weights_match_training_state(path, 2000)

    def test_matching_steps_are_accepted(self, tmp_path):
        from tools.distill_refiner import _check_weights_match_training_state
        path = tmp_path / 'context_refiner.safetensors'
        self._write_refiner(path, 2000)
        _check_weights_match_training_state(path, 2000)

    def test_a_file_without_a_recorded_step_claims_nothing(self, tmp_path):
        """Refiners predating this, and those from other modes, must still resume."""
        from tools.distill_refiner import _check_weights_match_training_state
        path = tmp_path / 'context_refiner.safetensors'
        self._write_refiner(path, None)
        _check_weights_match_training_state(path, 2000)

    def test_the_provenance_records_the_step_when_given_one(self):
        from tools.distill_refiner import refiner_provenance
        config = {'student': {'llm_path': '/llm', 'llm_hidden_layer': -1}}
        assert 'step' not in refiner_provenance(config, 2048, 512)
        assert refiner_provenance(config, 2048, 512, step=7)['step'] == '7'

    def test_the_stable_name_is_never_left_truncated(self, tmp_path):
        """A fixed filename rewritten in place has no fallback if the write is interrupted."""
        from tools.distill_refiner import _copy_atomically
        src, dst = tmp_path / 'tagged.bin', tmp_path / 'stable.bin'
        src.write_bytes(b'x' * 1024)
        dst.write_bytes(b'old')
        _copy_atomically(src, dst)
        assert dst.read_bytes() == b'x' * 1024
        assert not list(tmp_path.glob('*.tmp')), 'the temporary copy must be renamed, not left'


class TestFullModelKeepsTheRefinerInFp32:
    """Both artefacts a save writes are valid resume_from sources, so both keep the mantissa."""

    def test_the_refiner_inside_the_full_model_is_fp32(self, tmp_path):
        import safetensors.torch
        from tools.distill_refiner import save_full_model
        from models.text_refiner import ContextRefiner

        teacher = tmp_path / 'anima.safetensors'
        safetensors.torch.save_file(
            {'net.blocks.0.weight': torch.zeros(2, 2, dtype=torch.float32)},
            str(teacher), metadata={'format': 'pt'},
        )
        refiner = ContextRefiner(cap_feat_dim=8, model_dim=16, num_layers=1, num_heads=2)
        out = tmp_path / 'model.safetensors'
        save_full_model(teacher, refiner, out, torch.bfloat16)

        with safetensors.safe_open(str(out), framework='pt') as f:
            refiner_keys = [k for k in f.keys() if k.startswith('net.context_refiner.')]
            assert refiner_keys, 'the refiner must be embedded in the full model'
            for key in refiner_keys:
                assert f.get_tensor(key).dtype is torch.float32, (
                    f'{key} was written as {f.get_tensor(key).dtype}; save_refiner keeps the '
                    'refiner in fp32 for a reason that applies to this file too'
                )
            # The frozen DiT half still follows `dtype`.
            assert f.get_tensor('net.blocks.0.weight').dtype is torch.bfloat16


class TestSideBranchAccumulationScaling:
    """A forward that bypasses the engine must still get the 1/N accumulation scaling.

    DeepSpeed does not divide inside backward(). engine.py computes gas_scaled_loss only as a
    return value and applies the real division through a hook on the output of its own forward.
    The unconditional rollout branch calls the bare refiner on purpose -- a second engine
    forward would build a second backward hook manager -- so it bypasses the only scaling site
    and contributed grad_accum times its intended gradient. Measured against a real engine at
    grad_accum=4: engine path 1.0x a single un-accumulated batch, bypassing path 4.0x.

    These assert the values, not the presence of a line, so inverting either one fails.
    """

    @staticmethod
    def _grad_after(strategy, grad_accum):
        x = torch.ones(3, requires_grad=True)
        y = strategy.scale_side_branch(x * 2.0)
        # Stand in for grad_accum micro batches all contributing through the same tensor.
        (y.sum() * grad_accum).backward()
        return x.grad.clone()

    def test_zero_divides_a_bypassing_branch_by_grad_accum(self):
        from tools.distill_refiner import DeepSpeedZeROStrategy
        strategy = object.__new__(DeepSpeedZeROStrategy)
        strategy.grad_accum = 4
        grad = self._grad_after(strategy, 4)
        # Without the hook this would be 2 * 4 = 8 per element.
        assert torch.allclose(grad, torch.full((3,), 2.0)), (
            f'expected the side branch to be divided by grad_accum, got {grad.tolist()}'
        )

    def test_zero_leaves_a_bypassing_branch_alone_without_accumulation(self):
        from tools.distill_refiner import DeepSpeedZeROStrategy
        strategy = object.__new__(DeepSpeedZeROStrategy)
        strategy.grad_accum = 1
        grad = self._grad_after(strategy, 1)
        assert torch.allclose(grad, torch.full((3,), 2.0))

    def test_ddp_does_not_scale_the_side_branch_a_second_time(self):
        from tools.distill_refiner import DDPStrategy
        strategy = object.__new__(DDPStrategy)
        strategy.grad_accum = 4
        grad = self._grad_after(strategy, 4)
        # DDP divides the whole loss in backward(), so scaling here too would halve it twice.
        assert torch.allclose(grad, torch.full((3,), 8.0)), (
            f'DDP must leave the side branch alone, got {grad.tolist()}'
        )

    def test_both_strategies_expose_the_hook(self):
        from tools.distill_refiner import DDPStrategy, DeepSpeedZeROStrategy
        for cls in (DDPStrategy, DeepSpeedZeROStrategy):
            assert callable(getattr(cls, 'scale_side_branch', None)), (
                f'{cls.__name__} must answer scale_side_branch; the training loop calls it on '
                'whichever strategy is active'
            )


class TestNoSyncBoundaryBehaviour:
    """Exercise the accumulation boundary rather than grepping for it.

    TestScriptStructure asserts the two literals appear in the file, which cannot tell
    `is_last` from `not is_last` -- inverting the condition keeps both strings intact and
    would turn every micro batch into an all-reduce (or, worse, skip the one that matters).
    """

    class _Module:
        def __init__(self):
            self.no_sync_calls = 0

        def no_sync(self):
            self.no_sync_calls += 1
            return contextlib.nullcontext('no_sync')

    def _strategy(self, world_size):
        from tools.distill_refiner import DDPStrategy
        strategy = object.__new__(DDPStrategy)
        strategy.world_size = world_size
        strategy.module = self._Module()
        return strategy

    def test_the_last_micro_batch_syncs(self):
        strategy = self._strategy(2)
        with strategy.micro_batch_context(is_last=True) as marker:
            assert marker is None, 'the last micro batch must all-reduce, not be wrapped'
        assert strategy.module.no_sync_calls == 0

    def test_every_earlier_micro_batch_does_not_sync(self):
        strategy = self._strategy(2)
        with strategy.micro_batch_context(is_last=False) as marker:
            assert marker == 'no_sync', 'accumulating micro batches must skip the all-reduce'
        assert strategy.module.no_sync_calls == 1

    def test_a_single_process_never_calls_no_sync(self):
        """no_sync only exists on a DDP-wrapped module, and world size 1 leaves it unwrapped."""
        strategy = self._strategy(1)
        for is_last in (True, False):
            with strategy.micro_batch_context(is_last=is_last) as marker:
                assert marker is None
        assert strategy.module.no_sync_calls == 0
