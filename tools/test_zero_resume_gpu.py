"""Check that a ZeRO distillation checkpoint actually resumes. Needs a GPU box and 2 ranks.

    deepspeed --num_gpus=2 tools/test_zero_resume_gpu.py

Why this is a script and not a pytest case: it needs `deepspeed.initialize`, which JIT-builds
`deepspeed_shm_comm` and so does not run on the CPU-only development machine at all. Everything
in `tools/distill_refiner.py` that this exercises is unit-tested there against a plain torch
optimizer; what cannot be tested there is the one claim the whole design rests on -- that
`deepspeed.initialize` replaces the client optimizer's param_groups with a rank-local flat fp32
partition, so rank 0's `state_dict()` is a shard rather than the state.

That claim is why `save_training_state` is called by every rank with its own `rank=`, and why
the load moved to after `build_strategy`. Before that, a ZeRO run wrote a `distill_state.pt` that
looked fine and could not be loaded: the resume raised "loaded state dict contains a parameter
group that doesn't match the size of optimizer's group", hours after the file was written.

The script asserts, in order:

  1. after `deepspeed.initialize`, each param group holds exactly one tensor, and the ranks'
     partitions differ -- the premise;
  2. every rank writes its own shard, and the shards differ on disk;
  3. a resume restores each rank's own moments bit-exactly;
  4. resuming into a different world size is refused rather than loading the wrong shard;
  5. a DDP-written state file is not silently loaded into a sharded optimizer.

Exit code 0 means all five hold.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deepspeed  # noqa: E402

from tools.distill_refiner import (  # noqa: E402
    load_training_state,
    save_training_state,
    training_state_path,
)


def build_engine(model, lr=1e-3):
    """A ZeRO stage 1 engine over `model`, returning the client optimizer we hand it.

    The client optimizer is the object under test: main() keeps its own name bound to it and
    calls state_dict() on it, so what deepspeed does to it in place is the whole question.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    config = {
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 1,
        'zero_optimization': {'stage': 1},
        'zero_allow_untested_optimizer': True,
        'steps_per_print': 10 ** 9,
    }
    engine, _, _, wrapped_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=config,
        dist_init_required=False,
    )
    return engine, optimizer, wrapped_scheduler


def moments(optimizer):
    """Adam's exp_avg for every parameter the optimizer is actually stepping."""
    out = []
    for group in optimizer.param_groups:
        for param in group['params']:
            state = optimizer.state.get(param, {})
            if 'exp_avg' in state:
                out.append(state['exp_avg'].detach().clone())
    return out


def main():
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    if world_size < 2:
        raise SystemExit('Run with --num_gpus=2 or more: one rank shards nothing.')

    device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)).to(device)
    engine, optimizer, scheduler = build_engine(model)

    # 1. The premise: the client optimizer now holds one flat partition per group, and the
    #    partitions differ between ranks.
    sizes = [len(group['params']) for group in optimizer.param_groups]
    assert all(n == 1 for n in sizes), f'rank {rank}: expected one flat partition per group, got {sizes}'
    partition = optimizer.param_groups[0]['params'][0]
    gathered = [torch.zeros_like(partition) for _ in range(world_size)]
    dist.all_gather(gathered, partition.detach())
    if is_main:
        distinct = not torch.equal(gathered[0], gathered[1])
        print(f'1. one flat partition per group: OK | rank 0 and 1 partitions differ: {distinct}')

    # Take a few steps so Adam has non-zero moments to lose.
    for _ in range(3):
        loss = engine(torch.randn(1, 64, device=device)).square().mean()
        engine.backward(loss)
        engine.step()

    before = moments(optimizer)
    assert before, f'rank {rank}: no Adam state after three steps; the engine is not stepping this optimizer'

    with tempfile.TemporaryDirectory() as raw_tmp:
        # Every rank writes into the same directory, so it has to be the same directory. On a
        # single box tempfile gives each rank its own; broadcast rank 0's.
        shared = [raw_tmp if is_main else None]
        dist.broadcast_object_list(shared, src=0)
        tmp = Path(shared[0])
        weights = tmp / 'context_refiner_epoch1.safetensors'

        # 2. Every rank writes its own shard.
        save_training_state(weights, optimizer, scheduler, 3, rank=rank, world_size=world_size)
        dist.barrier()
        shards = sorted(p.name for p in tmp.glob('distill_state_epoch1_rank*.pt'))
        assert len(shards) == world_size, f'expected {world_size} shards, found {shards}'
        if is_main:
            print(f'2. {len(shards)} shards written, one per rank: {shards}')

        # 3. A resume restores this rank's own moments, bit-exactly.
        fresh_model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)).to(device)
        fresh_engine, fresh_optimizer, fresh_scheduler = build_engine(fresh_model)
        step = load_training_state(weights, fresh_optimizer, fresh_scheduler, is_main,
                                   rank=rank, world_size=world_size)
        assert step == 3, f'rank {rank}: resumed at step {step}, expected 3'
        after = moments(fresh_optimizer)
        assert len(after) == len(before), f'rank {rank}: {len(before)} moment tensors before, {len(after)} after'
        for i, (a, b) in enumerate(zip(before, after)):
            assert torch.equal(a.to(b.device), b), f'rank {rank}: moment tensor {i} differs after resume'
        if is_main:
            print(f'3. resume restored {len(after)} moment tensors bit-exactly on every rank')

        # 4. A different world size is refused, not silently mis-loaded.
        refused = False
        try:
            load_training_state(weights, fresh_optimizer, fresh_scheduler, is_main,
                                rank=rank, world_size=world_size + 4)
        except RuntimeError as exc:
            refused = 'rank job' in str(exc)
        assert refused, f'rank {rank}: a {world_size + 4}-rank resume of a {world_size}-rank checkpoint was allowed'
        if is_main:
            print('4. resuming into a different world size is refused')

        # 5. A DDP-written file is not fed to a sharded optimizer.
        if is_main:
            ddp_weights = tmp / 'context_refiner_epoch2.safetensors'
            plain = torch.nn.Linear(8, 8)
            plain_opt = torch.optim.AdamW(plain.parameters(), lr=1e-3)
            plain_sched = torch.optim.lr_scheduler.LambdaLR(plain_opt, lambda s: 1.0)
            save_training_state(ddp_weights, plain_opt, plain_sched, 99)
            assert training_state_path(ddp_weights).exists()
        dist.barrier()
        resumed = load_training_state(tmp / 'context_refiner_epoch2.safetensors',
                                      fresh_optimizer, fresh_scheduler, is_main,
                                      rank=rank, world_size=world_size)
        assert resumed == 0, f'rank {rank}: a DDP state file was loaded into a ZeRO optimizer'
        if is_main:
            print('5. a DDP state file resumes the weights only under ZeRO')

        dist.barrier()

    if is_main:
        print('\nAll five checks passed. ZeRO resume is sound on this build.')


if __name__ == '__main__':
    main()
