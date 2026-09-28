"""Audit evidence: where ZeRO keeps the full-precision weights under a bf16 section.

    PYTHONPATH=test/childenv python .audit/exp_zero_masters.py

Establishes the fact tools/distill_refiner.py's master-weight save/restore rests on, against a
real engine rather than by reading the source:

  - deepspeed.initialize casts the module to bf16, so refiner.state_dict() -- what save_refiner
    writes -- is the bit16 view, not the training weights.
  - the fp32 masters live in the optimizer's flat partition, which initialize installs as the
    client optimizer's param_groups.
  - optimizer.state_dict() carries the moments and the loss scaler but never the parameter
    values, so nothing else in the checkpoint holds them.
"""
import os, torch, torch.distributed as dist, deepspeed, deepspeed.ops
for name in list(deepspeed.ops.__compatible_ops__):
    if 'shm' in name.lower():
        deepspeed.ops.__compatible_ops__[name] = False
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29594')
os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
if not dist.is_initialized():
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

torch.manual_seed(0)
m = torch.nn.Linear(8, 8, bias=False)
opt = torch.optim.AdamW(m.parameters(), lr=0.1)
cfg = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1,
       'gradient_clipping': 1.0,
       'zero_optimization': {'stage': 1, 'contiguous_gradients': False, 'overlap_comm': False,
                             'reduce_bucket_size': 4096, 'allgather_bucket_size': 4096},
       'zero_allow_untested_optimizer': True, 'steps_per_print': 10**9,
       'wall_clock_breakdown': False,
       'bf16': {'enabled': True}}
eng, o, _, _ = deepspeed.initialize(model=m, optimizer=opt, config=cfg, dist_init_required=False)

print('module param dtype after initialize :', next(eng.module.parameters()).dtype)
print('state_dict() tensor dtype           :', next(iter(eng.module.state_dict().values())).dtype)
has = hasattr(o, 'single_partition_of_fp32_groups')
print('optimizer exposes fp32 masters      :', has)
if has:
    print('master partition dtype              :', o.single_partition_of_fp32_groups[0].dtype)
    print('master partition numel              :', o.single_partition_of_fp32_groups[0].numel())
print('optimizer.state_dict() top-level keys:', sorted(o.state_dict().keys())[:8])
print('cur_scale present                   :', hasattr(o, 'cur_scale'))
