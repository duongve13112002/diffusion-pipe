"""Audit evidence: DeepSpeed's 1/N accumulation scaling follows the engine forward.

    PYTHONPATH=test/childenv python .audit/exp_zero_scaling.py

Measures, on a real ZeRO engine at gradient_accumulation_steps=4, the gradient reaching a
parameter through three paths, as a ratio to one un-accumulated batch:

  - through engine(x)               -> 1.0x  (correctly averaged)
  - through the bare module         -> 4.0x  (no scaling: the bug)
  - bare module + the /N hook       -> 1.0x  (the fix)

The scaling is applied by a hook DeepSpeed registers on the output of its own forward, not inside
backward(), which is why a forward that bypasses the engine bypasses the scaling.
tools/test_zero_side_branch_multirank.py is the two-rank version of this.
"""
import os, torch, torch.distributed as dist, deepspeed, deepspeed.ops

for name in list(deepspeed.ops.__compatible_ops__):
    if 'shm' in name.lower():
        deepspeed.ops.__compatible_ops__[name] = False
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29591')
os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
if not dist.is_initialized():
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

GAS = 4

def build():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4, bias=False)
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    cfg = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': GAS,
           'gradient_clipping': 1.0,
           'zero_optimization': {'stage': 1, 'contiguous_gradients': False, 'overlap_comm': False,
                                 'reduce_bucket_size': 4096, 'allgather_bucket_size': 4096},
           'zero_allow_untested_optimizer': True, 'steps_per_print': 10**9,
           'wall_clock_breakdown': False}
    eng, _, _, _ = deepspeed.initialize(model=m, optimizer=opt, config=cfg,
                                        dist_init_required=False)
    return eng, m

def run(via_engine):
    eng, m = build()
    torch.manual_seed(7)
    x = torch.randn(2, 4)
    for micro in range(GAS):
        eng.set_gradient_accumulation_boundary(False)
        if via_engine == 'engine':
            out = eng(x)
        else:
            out = m(x)
            if via_engine == 'bare_fixed' and GAS > 1 and out.requires_grad:
                out.register_hook(lambda g, n=GAS: g / n)
        eng.backward(out.sum())
    g = m.weight.grad.detach().clone()
    return g

g_engine = run('engine')
g_bare   = run('bare')
g_fixed  = run('bare_fixed')
# Reference: one un-accumulated batch, plain autograd
torch.manual_seed(0)
ref = torch.nn.Linear(4, 4, bias=False)
torch.manual_seed(7); x = torch.randn(2, 4)
ref(x).sum().backward()
g_ref = ref.weight.grad.detach().clone()

print('GAS =', GAS)
print('||grad|| via engine forward :', float(g_engine.norm()))
print('||grad|| via BARE module    :', float(g_bare.norm()))
print('||grad|| single-batch ref   :', float(g_ref.norm()))
print('engine/ref ratio :', float(g_engine.norm() / g_ref.norm()))
print('bare/ref   ratio :', float(g_bare.norm()   / g_ref.norm()))
print('bare+FIX/ref ratio:', float(g_fixed.norm()  / g_ref.norm()))
