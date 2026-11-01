"""Check that an attention call and its activation-checkpointing recomputation run the same PyTorch
SDPA kernel, with and without utils.attention.settle_sdpa_priority.

    python tools/check_sdpa_priority.py
    python tools/check_sdpa_priority.py --gpus 0 1 --tokens 97 140 --heads 24 --head-dim 128

Run it from the diffusion-pipe checkout, in the environment you train with, on a GPU of the kind you
train on. No weights are needed. Each scenario runs in a fresh process, because PyTorch changes its
SDPA kernel order once per process, at the first call that picks a kernel:

  unsettled              nothing settles the order first (the code before the fix);
  sdpa_prefer_cudnn=false settle_sdpa_priority(False), the default;
  sdpa_prefer_cudnn=true  settle_sdpa_priority(True).

A scenario is one small attention layer whose first SDPA call goes through ComfyUI's
comfy.ops.scaled_dot_product_attention (its sdpa_kernel(..., set_priority=True) path, taken for
inputs of 128K elements or more) inside torch.utils.checkpoint, followed by a backward. It reports
the SDPA kernels the forward and the recomputation ran (from the profiler) and whether the backward
stopped with a CheckpointError. The unsettled scenario fails only where PyTorch changes the order
(Hopper / Blackwell with a recent cuDNN); the two settled ones must pass everywhere. The exit code is
1 when a settled scenario fails.

Scenarios are spread over the GPUs given (default: every visible GPU), one process per GPU at a time.
On AMD GPUs ComfyUI calls SDPA when it is imported, which settles the order before any scenario
runs, so the unsettled scenario cannot show the problem there.
"""

import argparse
import contextlib
import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SCENARIOS = {
    'unsettled': None,
    'sdpa_prefer_cudnn=false': False,
    'sdpa_prefer_cudnn=true': True,
}
# ComfyUI runs SDPA inside its priority context only from this many query elements on.
COMFY_CONTEXT_ELEMENTS = 1024 * 128


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', type=int, nargs='*', default=None,
                        help='GPU indices to spread the scenarios over (default: every visible GPU)')
    parser.add_argument('--tokens', type=int, nargs='+', default=[97],
                        help='sequence lengths to check, one scenario run each (default: 97, a MOSS-Image caption)')
    parser.add_argument('--heads', type=int, default=24, help='attention heads (default: 24, MOSS-Image)')
    parser.add_argument('--head-dim', type=int, default=128, help='head dimension (default: 128, MOSS-Image)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16',
                        help='activation dtype (default: bfloat16)')
    parser.add_argument('--timeout', type=float, default=600,
                        help='seconds a scenario may run before it is stopped and reported as failed (default: 600)')
    args = parser.parse_args(argv)
    for tokens in args.tokens:
        elements = args.batch_size * args.heads * tokens * args.head_dim
        if elements < COMFY_CONTEXT_ELEMENTS:
            parser.error(f'--tokens {tokens}: {elements} query elements; ComfyUI only uses its priority context '
                         f'from {COMFY_CONTEXT_ELEMENTS}, so this shape cannot show the problem')
    return args


# The forward op of each PyTorch SDPA kernel, as the profiler names it.
KERNEL_OPS = {
    'aten::_scaled_dot_product_flash_attention': 'flash',
    'aten::_scaled_dot_product_cudnn_attention': 'cudnn',
    'aten::_scaled_dot_product_efficient_attention': 'efficient',
    'aten::_scaled_dot_product_attention_math': 'math',
}


def attention_kernels(profile):
    """The SDPA kernels a profile recorded, in order."""
    return [KERNEL_OPS[event.name] for event in profile.events() if event.name in KERNEL_OPS]


def run_scenario(gpu, name, prefer_cudnn, tokens, args):
    """One scenario in this (fresh) process; returns a result dict."""
    sys.path.insert(0, ROOT)
    import torch
    import torch.utils.checkpoint
    torch.cuda.set_device(gpu)
    import utils.common  # noqa: F401  (claims the utils package before ComfyUI joins sys.path)
    import models.base  # noqa: F401  (puts ComfyUI on sys.path and sets it up as training does)
    import comfy.ops
    from utils import attention

    result = {'scenario': name, 'gpu': gpu, 'tokens': tokens,
              'device': f'{torch.cuda.get_device_name()} (sm{"".join(map(str, torch.cuda.get_device_capability()))})',
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    if prefer_cudnn is not None:
        attention.settle_sdpa_priority(prefer_cudnn)
    dtype = getattr(torch, args.dtype)
    dim = args.heads * args.head_dim
    generator = torch.Generator(device='cuda').manual_seed(0)
    qkv = torch.nn.Linear(dim, 3 * dim, device='cuda', dtype=dtype)
    x = torch.randn(args.batch_size, tokens, dim, device='cuda', dtype=dtype, generator=generator,
                    requires_grad=True)

    def layer(x):
        q, k, v = qkv(x).unflatten(-1, (3, args.heads, args.head_dim)).permute(2, 0, 3, 1, 4)
        return comfy.ops.scaled_dot_product_attention(q, k, v)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as forward:
        out = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
    result['forward'] = attention_kernels(forward)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as backward:
            out.float().square().mean().backward()
        result['recomputed'] = attention_kernels(backward)
        result['error'] = None
    except torch.utils.checkpoint.CheckpointError as error:
        result['recomputed'] = None
        result['error'] = f'CheckpointError: {str(error).splitlines()[0]}'
    result['order'] = attention._backend_names(torch._C._get_sdp_priority_order())
    return result


def worker(gpu, task, connection, args):
    """One scenario in a fresh process: PyTorch changes its kernel order once per process."""
    name, prefer_cudnn, tokens = task
    try:
        result = run_scenario(gpu, name, prefer_cudnn, tokens, args)
    except Exception:
        result = {'scenario': name, 'gpu': gpu, 'tokens': tokens, 'crash': traceback.format_exc()}
    connection.send(result)
    connection.close()


def run_all(tasks, gpus, args, target=worker):
    """Every task in a process of its own, at most one process per GPU at a time; a GPU takes the
    next task as soon as its process ends. Each process sends its result through a pipe of its own,
    so a result cannot be credited to another task. A process that ends without one (a crash in
    CUDA, say) or runs past args.timeout seconds is reported as failed, and its GPU is reused."""
    context = mp.get_context('spawn')
    todo, free, results = list(tasks), list(gpus), []
    running = {}  # process sentinel -> [process, receiving end, gpu, task, start time, result]
    while todo or running:
        while free and todo:
            gpu, task = free.pop(0), todo.pop(0)
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=target, args=(gpu, task, send, args))
            process.start()
            send.close()
            running[process.sentinel] = [process, receive, gpu, task, time.monotonic(), None]
        # The pipes too: a result larger than the pipe's buffer is only sent once it is read.
        waiting = list(running) + [entry[1] for entry in running.values() if entry[5] is None]
        ready = mp_connection.wait(waiting, timeout=5)
        for entry in running.values():
            if entry[5] is None and entry[1] in ready:
                try:
                    entry[5] = entry[1].recv()
                except EOFError:
                    pass
        now = time.monotonic()
        for sentinel in list(running):
            process, receive, gpu, task, started, result = running[sentinel]
            timed_out = process.exitcode is None and now - started > args.timeout
            if process.exitcode is None and not timed_out:
                continue
            if timed_out:
                process.terminate()
            process.join()
            if result is None and not timed_out and receive.poll():
                with contextlib.suppress(EOFError):
                    result = receive.recv()
            if result is None:
                name, _, tokens = task
                why = (f'it ran longer than --timeout {args.timeout:g} s' if timed_out
                       else f'the process exited with code {process.exitcode} and no result')
                result = {'scenario': name, 'gpu': gpu, 'tokens': tokens, 'crash': why}
            receive.close()
            results.append(result)
            running.pop(sentinel)
            free.append(gpu)
    return results


def main(argv=None):
    args = parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        sys.exit('This check needs a CUDA GPU.')
    gpus = args.gpus if args.gpus else list(range(torch.cuda.device_count()))
    tasks = [(name, prefer_cudnn, tokens) for tokens in args.tokens for name, prefer_cudnn in SCENARIOS.items()]
    results = run_all(tasks, gpus, args)

    failed = False
    first = next((r for r in results if 'crash' not in r), None)
    if first:
        print(f"{first['device']}, torch {first['torch']}, CUDA {first['cuda']}, cuDNN {first['cudnn']}")
    for result in sorted(results, key=lambda r: (r['tokens'], list(SCENARIOS).index(r['scenario']))):
        head = f"[{result['scenario']}] tokens {result['tokens']}, GPU {result['gpu']}:"
        if 'crash' in result:
            print(f'{head} the check itself failed:\n{result["crash"]}')
            failed = True
            continue
        kernels = f"forward {', '.join(result['forward']) or 'none'}"
        if result['recomputed'] is not None:
            kernels += f"; recomputed {', '.join(result['recomputed']) or 'none'}"
        verdict = result['error'] or 'OK'
        print(f"{head} {verdict} ({kernels}). Kernel order after: {result['order']}")
        if result['scenario'] != 'unsettled' and result['error']:
            failed = True
    unsettled = [r for r in results if r['scenario'] == 'unsettled' and 'crash' not in r]
    if unsettled and not any(r['error'] for r in unsettled):
        print('The unsettled scenario passed: PyTorch does not change its kernel order on this GPU and '
              'cuDNN, so the problem does not occur here.')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
