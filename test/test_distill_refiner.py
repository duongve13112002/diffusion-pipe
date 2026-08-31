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
        for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from tools.distill_refiner import setup_distributed
        if not env:
            assert setup_distributed() == expected

    def test_the_probe_seed_is_not_rank_offset(self):
        # Every rank must measure against the same queries, or the ranks optimise different
        # objectives and the all-reduce averages nonsense.
        src = self.source()
        assert 'manual_seed(seed)' in src
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

        # No suppression from outside: the engine tracks its own accumulation boundary.
        with strategy.micro_batch_context(is_last=False):
            strategy.backward(torch.tensor(1.0))
        strategy.zero_grad()  # must be a no-op, or gradients vanish mid-accumulation
        assert strategy.step() == 0.0  # None from the engine becomes 0.0, not a crash
        assert calls == ['backward', 'step']

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
