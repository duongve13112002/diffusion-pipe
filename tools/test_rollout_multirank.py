"""Two real ranks, gloo, DDP over the refiner, denoising rollout on.

    python tools/test_rollout_multirank.py

A standalone script rather than a pytest case: it spawns two processes, and the CPU suite
already skips its distributed smoke test on Windows. Tiny enough to run on an 8 GB box -- a
one-block DiT and a 32-wide Linear standing in for the refiner -- because what is under test is
the wiring, not the model.

Checks the claims docs/anima_refiner/denoising-rollout.md makes about multi-GPU that nothing
else verifies:
  - the frozen DiT is outside the DDP-wrapped module, so it contributes no gradients
  - the rank-offset generator makes each rank walk a DIFFERENT trajectory
  - DDP still all-reduces the refiner's gradients to identical values across ranks
"""
import os, sys, random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size, out):
    sys.path.insert(0, os.getcwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29655'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    from models.cosmos_predict2_modeling import MiniTrainDIT
    from tools.distill_refiner import teacher_trajectory, rollout_loss

    cfg = dict(max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16,
               patch_spatial=2, patch_temporal=1, model_channels=64, num_blocks=1, num_heads=4,
               crossattn_emb_channels=32, concat_padding_mask=True, pos_emb_cls='rope3d',
               pos_emb_learnable=True, pos_emb_interpolation='crop', min_fps=1, max_fps=30,
               use_adaln_lora=True, adaln_lora_dim=16)
    torch.manual_seed(0)                       # same frozen DiT on every rank
    dit = MiniTrainDIT(**cfg).eval().requires_grad_(False)

    torch.manual_seed(0)                       # same initial refiner on every rank
    refiner = torch.nn.Linear(32, 32)
    ddp = torch.nn.parallel.DistributedDataParallel(refiner)

    # The claim under test: the DiT is not part of what DDP wraps.
    wrapped = {id(p) for p in ddp.parameters()}
    dit_inside = any(id(p) in wrapped for p in dit.parameters())

    # Rank-offset, exactly as main() builds it.
    generator = torch.Generator().manual_seed(42 + 10_000 + rank)
    teacher = torch.randn(2, 8, 32)            # same captions here, to isolate the generator
    visited = teacher_trajectory(dit, teacher, None, (2, 16, 1, 16, 16), 3, 0.0,
                                 generator, torch.device('cpu'), torch.float32)
    first_x = visited[0][0]

    student = ddp(torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(7)))
    loss = rollout_loss(dit, visited, teacher, student, None, None, 0.0, 2, random.Random(rank))
    loss.backward()

    out[rank] = {
        'dit_inside_ddp': dit_inside,
        'first_x_checksum': float(first_x.sum()),
        'grad_checksum': float(refiner.weight.grad.sum()),
        'dit_grads': sum(1 for p in dit.parameters() if p.grad is not None),
    }
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(worker, args=(2, out), nprocs=2, join=True)

    r0, r1 = out[0], out[1]
    print('DiT inside the DDP module      :', r0['dit_inside_ddp'], r1['dit_inside_ddp'], ' <- must be False False')
    print('DiT params holding a gradient  :', r0['dit_grads'], r1['dit_grads'], ' <- must be 0 0')
    print(f"trajectory checksum rank0/rank1: {r0['first_x_checksum']:.4f} / {r1['first_x_checksum']:.4f}")
    print('  different trajectories       :', r0['first_x_checksum'] != r1['first_x_checksum'], ' <- must be True')
    print(f"refiner grad rank0/rank1       : {r0['grad_checksum']:.8f} / {r1['grad_checksum']:.8f}")
    print('  all-reduced to the same value:', abs(r0['grad_checksum'] - r1['grad_checksum']) < 1e-9, ' <- must be True')
