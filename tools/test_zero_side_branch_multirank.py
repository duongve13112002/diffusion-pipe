"""Two real ranks, gloo, a real DeepSpeed ZeRO engine, gradient accumulation on.

    PYTHONPATH=test/childenv python tools/test_zero_side_branch_multirank.py

A standalone script rather than a pytest case, for the same reason as
tools/test_rollout_multirank.py: it spawns two processes, and two DeepSpeed engines on an 8 GB
box during a full suite run is not worth the risk. Tiny on purpose -- a 4-wide Linear standing
in for the refiner -- because what is under test is the wiring, not the model.

Checks the claim DeepSpeedZeROStrategy.scale_side_branch rests on, across real ranks rather than
one:

  - DeepSpeed applies its 1/gradient_accumulation_steps scaling through a hook on the output of
    its OWN forward, never inside backward(). So a forward that bypasses the engine -- the
    unconditional rollout branch, which calls the bare refiner to avoid building a second
    backward hook manager -- bypasses the only place the scaling happens.
  - With the hook the strategy installs, the bypassing path lands on the same gradient as the
    engine path, and both ranks agree after the all-reduce.

Prints PASS/FAIL per check and exits non-zero on any failure.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GAS = 4


def worker(rank, world_size, out):
    sys.path.insert(0, os.getcwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29656'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    import deepspeed
    import deepspeed.ops
    # The shm comm op is JIT-compiled and needs a C++ toolchain. build_shm_op() checks this
    # registry and returns None when the op is marked incompatible, which is the supported way
    # to skip it on a box that has no compiler.
    for name in list(deepspeed.ops.__compatible_ops__):
        if 'shm' in name.lower():
            deepspeed.ops.__compatible_ops__[name] = False

    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from tools.distill_refiner import DeepSpeedZeROStrategy

    torch.manual_seed(0)                       # identical model on every rank
    model = torch.nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    config = {
        'train_micro_batch_size_per_gpu': 2,
        'gradient_accumulation_steps': GAS,
        'gradient_clipping': 1.0,
        'zero_optimization': {
            'stage': 1, 'contiguous_gradients': False, 'overlap_comm': False,
            'reduce_bucket_size': 4096, 'allgather_bucket_size': 4096,
        },
        'zero_allow_untested_optimizer': True,
        'steps_per_print': 10 ** 9,
        'wall_clock_breakdown': False,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=config,
        dist_init_required=False,
    )

    strategy = object.__new__(DeepSpeedZeROStrategy)
    strategy.engine = engine
    strategy.grad_accum = GAS

    # The same input on both ranks, so the all-reduced gradient equals a single rank's and the
    # comparison against the un-accumulated reference stays exact.
    torch.manual_seed(7)
    x = torch.randn(2, 4)

    def accumulate(kind):
        engine.optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        for micro in range(GAS):
            engine.set_gradient_accumulation_boundary(False)
            if kind == 'engine':
                out_tensor = engine(x)
            else:
                out_tensor = model(x)
                if kind == 'bare_fixed':
                    out_tensor = strategy.scale_side_branch(out_tensor)
            engine.backward(out_tensor.sum())
        return model.weight.grad.detach().clone()

    g_engine = accumulate('engine')
    g_bare = accumulate('bare')
    g_fixed = accumulate('bare_fixed')

    # One un-accumulated batch through plain autograd: what N micro batches should average to.
    torch.manual_seed(0)
    reference_model = torch.nn.Linear(4, 4, bias=False)
    reference_model(x).sum().backward()
    g_ref = reference_model.weight.grad.detach().clone()

    # Do the two ranks agree on the fixed gradient?
    gathered = [torch.zeros_like(g_fixed) for _ in range(world_size)]
    dist.all_gather(gathered, g_fixed)
    ranks_agree = all(torch.allclose(gathered[0], g, atol=1e-6) for g in gathered)

    out[rank] = {
        'engine_ratio': float(g_engine.norm() / g_ref.norm()),
        'bare_ratio': float(g_bare.norm() / g_ref.norm()),
        'fixed_ratio': float(g_fixed.norm() / g_ref.norm()),
        'ranks_agree': ranks_agree,
    }
    dist.barrier()
    dist.destroy_process_group()


def main():
    world_size = 2
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(worker, args=(world_size, out), nprocs=world_size, join=True)

    results = [out[r] for r in range(world_size)]
    failures = []

    def check(name, ok, detail):
        print(f'{"PASS" if ok else "FAIL"}  {name}: {detail}')
        if not ok:
            failures.append(name)

    for rank, r in enumerate(results):
        check(f'rank {rank} engine path is 1/N scaled',
              abs(r['engine_ratio'] - 1.0) < 1e-4,
              f"ratio to a single un-accumulated batch = {r['engine_ratio']:.4f} (want 1.0)")
        check(f'rank {rank} bypassing path is NOT scaled by the engine',
              abs(r['bare_ratio'] - GAS) < 1e-4,
              f"ratio = {r['bare_ratio']:.4f} (want {GAS}.0 -- this is the bug the hook fixes)")
        check(f'rank {rank} scale_side_branch restores the 1/N',
              abs(r['fixed_ratio'] - 1.0) < 1e-4,
              f"ratio = {r['fixed_ratio']:.4f} (want 1.0)")
    check('both ranks agree on the fixed gradient',
          all(r['ranks_agree'] for r in results),
          'all_gather comparison across ranks')

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: {", ".join(failures)}')
        sys.exit(1)
    print('All checks passed.')


if __name__ == '__main__':
    main()
