"""Guard against pip-dependency API drift in the code that vendors or forks upstream internals.

Two files in this repo do not call a public API. They copy or subclass a dependency's
internals, so a new release of that dependency silently breaks them at runtime:

- ``utils/reduction.py`` is a copy of ``torch/multiprocessing/reductions.py`` with
  ``multiprocessing`` swapped for the third-party ``multiprocess`` library. It reaches into
  private torch symbols (``torch._utils._rebuild_tensor``, ``torch._storage_classes``, ...).
  torch 2.13 removed ``torch._namedtensor_internals.check_serializing_named_tensor`` and the
  module stopped importing.
- ``optimizers/adamw_8bit.py`` re-implements bitsandbytes' ``Optimizer2State.update_step`` to
  add Kahan summation. bitsandbytes 0.50 removed percentile clipping and the non-blockwise
  8-bit path, and dropped both keys from ``get_config()``, which raised ``KeyError``.

This script checks every symbol those two files depend on against what is actually installed.
Run it after upgrading torch or bitsandbytes:

    python tools/check_vendored_apis.py

Exit code is non-zero if any check fails. bitsandbytes checks are skipped (not failed) when it
is not installed, so the torch half still runs on a CPU-only box. See
docs/note/upstream-api-drift-audit.md for the wider procedure and the dependency->file map.
"""

import importlib
import inspect
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def _resolve(root, dotted):
    """Walk a dotted attribute path, returning None if any step is missing."""
    obj = root
    for part in dotted.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def accepts(func, name):
    """True if func accepts a keyword argument `name` (explicitly or via **kwargs)."""
    params = inspect.signature(func).parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def positional_order(func):
    """Names of the parameters in declaration order, excluding self and *args/**kwargs."""
    out = []
    for name, p in inspect.signature(func).parameters.items():
        if name == 'self':
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        out.append(name)
    return out


def check_torch(record):
    """Every private/internal torch symbol utils/reduction.py depends on."""
    import torch

    # Private symbols. These carry no stability guarantee, which is exactly why they are listed.
    private_symbols = [
        '_utils._rebuild_tensor',
        '_utils._element_size',
        '_nested_view_from_buffer_copy',
        '_storage_classes',
        '_tensor_classes',
        'Storage._free_weak_ref',
        'Storage._expired',
        'UntypedStorage._new_shared_filename_cpu',
        'UntypedStorage._new_with_weak_ptr',
        'cuda._lazy_init',
        'cuda.Event.from_ipc_handle',
        'storage.TypedStorage',
        'utils.hooks.warn_if_has_hooks',
        'multiprocessing.get_sharing_strategy',
    ]
    for dotted in private_symbols:
        record(f'torch.{dotted} exists', _resolve(torch, dotted) is not None)

    record('torch._storage_classes is iterable', hasattr(torch._storage_classes, '__iter__'))
    record('torch._tensor_classes is iterable', hasattr(torch._tensor_classes, '__iter__'))

    # The nested-tensor rebuild path imports this module lazily inside a function.
    try:
        importlib.import_module('torch.nested._internal.nested_tensor')
        nested_ok = True
    except ImportError:
        nested_ok = False
    record('torch.nested._internal.nested_tensor importable', nested_ok)

    # torch 2.13 removed the named-tensor API outright -- has_names, names, rename and
    # refine_names are all gone. It used to back check_serializing_named_tensor, which torch
    # dropped from reduce_tensor; utils/reduction.py never called either, so there is nothing
    # here to keep in step and no check to make.

    # The real target: utils/reduction.py must import and register cleanly.
    try:
        from utils import reduction
        import_error = None
    except Exception as e:
        reduction = None
        import_error = e
    record(f'utils.reduction imports (error: {import_error})' if import_error else 'utils.reduction imports',
           reduction is not None)
    if reduction is not None:
        record('utils.reduction.init_reductions is callable', callable(getattr(reduction, 'init_reductions', None)))
        # torch used to guard reduce_tensor with check_serializing_named_tensor and dropped it;
        # 2.13 has no such symbol and neither does our copy, so there is nothing to keep in
        # step. What matters now is that every name we vendored still exists upstream with the
        # same signature, which is a comparison against torch rather than a call into ours.
        import inspect
        import torch.multiprocessing.reductions as upstream
        for name in ('reduce_tensor', 'rebuild_tensor', 'reduce_storage', 'reduce_typed_storage'):
            ours = getattr(reduction, name, None)
            theirs = getattr(upstream, name, None)
            if ours is None or theirs is None:
                record(f'torch.multiprocessing.reductions.{name} still exists', False)
                continue
            same = list(inspect.signature(ours).parameters) == list(inspect.signature(theirs).parameters)
            record(f'{name} signature matches torch', same)


def check_bitsandbytes(record):
    """The bitsandbytes surface optimizers/adamw_8bit.py re-implements."""
    import bitsandbytes
    import bitsandbytes.functional as F

    adamw8bit = _resolve(bitsandbytes, 'optim.AdamW8bit')
    record('bitsandbytes.optim.AdamW8bit exists', adamw8bit is not None)
    if adamw8bit is None:
        return

    for method in ('init_state', 'update_step', 'get_config', 'get_state_buffer'):
        record(f'AdamW8bit.{method} exists', callable(getattr(adamw8bit, method, None)))

    update_32bit = getattr(F, 'optimizer_update_32bit', None)
    record('functional.optimizer_update_32bit exists', callable(update_32bit))
    if callable(update_32bit):
        expected = [
            'optimizer_name', 'g', 'p', 'state1', 'beta1', 'eps', 'step', 'lr', 'state2',
            'beta2', 'beta3', 'alpha', 'weight_decay', 'gnorm_scale', 'unorm_vec',
            'max_unorm', 'skip_zeros',
        ]
        record('optimizer_update_32bit positional order unchanged',
               positional_order(update_32bit) == expected)

    update_blockwise = getattr(F, 'optimizer_update_8bit_blockwise', None)
    record('functional.optimizer_update_8bit_blockwise exists', callable(update_blockwise))
    if callable(update_blockwise):
        expected = [
            'optimizer_name', 'g', 'p', 'state1', 'state2', 'beta1', 'beta2', 'beta3', 'alpha',
            'eps', 'step', 'lr', 'qmap1', 'qmap2', 'absmax1', 'absmax2', 'weight_decay',
            'gnorm_scale', 'skip_zeros',
        ]
        record('optimizer_update_8bit_blockwise positional order unchanged',
               positional_order(update_blockwise) == expected)

    # Percentile clipping is optional, but it has to be all-or-nothing: our update_step only
    # reaches F.percentile_clipping when the config carries the key, and the config only
    # carries it when the constructor accepts it.
    ctor_has_pc = accepts(adamw8bit.__init__, 'percentile_clipping')
    func_has_pc = callable(getattr(F, 'percentile_clipping', None))
    record(f'percentile_clipping support is consistent (ctor={ctor_has_pc}, functional={func_has_pc})',
           ctor_has_pc == func_has_pc)

    # Same contract for the non-blockwise 8-bit path.
    ctor_has_bw = accepts(adamw8bit.__init__, 'block_wise')
    func_has_8bit = callable(getattr(F, 'optimizer_update_8bit', None))
    record(f'block_wise support is consistent (ctor={ctor_has_bw}, functional={func_has_8bit})',
           ctor_has_bw == func_has_8bit)


def run_checks():
    """Return (results, skipped) where results is a list of (description, ok)."""
    results = []
    skipped = []

    def record(description, condition):
        results.append((description, bool(condition)))
        return condition

    check_torch(record)
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        skipped.append(f'bitsandbytes checks skipped, import failed: {e}')
    else:
        check_bitsandbytes(record)

    return results, skipped


def main():
    import torch
    print(f'torch {torch.__version__}')
    try:
        import bitsandbytes
        print(f'bitsandbytes {bitsandbytes.__version__}')
    except Exception:
        print('bitsandbytes not importable')
    print()

    results, skipped = run_checks()
    for description, ok in results:
        print(f'[{"PASS" if ok else "FAIL"}] {description}')
    for message in skipped:
        print(f'[SKIP] {message}')

    failed = [d for d, ok in results if not ok]
    print()
    if failed:
        print(f'{len(failed)} vendored-API check(s) FAILED. The dependency drifted; fix the '
              'matching code (and update this script if the change is intended):')
        for d in failed:
            print(f'  - {d}')
        return 1
    print(f'All {len(results)} vendored-API checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
