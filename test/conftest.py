"""Test-environment shims so the model pipelines can be imported on a CPU-only machine.

models/base.py imports ComfyUI, which at import time pulls in comfy_aimdo (a GPU memory
management extension) and queries the current CUDA device. Neither is available on a plain
CPU box, so the pipeline modules cannot even be imported there.

DeepSpeed is the third: it does not install on a CPU-only Windows box, and utils/common.py
imports it at module scope, so without a stand-in not one test in the suite can be collected.

All three shims are conditional: if the real comfy_aimdo, CUDA or deepspeed is present, this
file changes nothing. That keeps the tests honest in a real training environment while still
letting the CPU-only tests run in CI or on a laptop.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types


def _stub_deepspeed():
    """Stand in for the deepspeed symbols the library imports, with single-process semantics.

    Only a handful of deepspeed's surface is reachable from a CPU test. Some of it has an
    unambiguous meaning when there is exactly one process -- rank 0, world size 1, a barrier
    that has nobody to wait for -- so that part is implemented, not faked.

    The collectives are deliberately different. send, recv, broadcast and all_reduce need a
    real peer, and a test that reaches one is testing distributed behaviour this shim cannot
    stand in for. Returning a plausible value there would let such a test pass while proving
    nothing, so they raise instead and say why.
    """
    if importlib.util.find_spec('deepspeed') is not None:
        return  # the real package is installed; leave it alone

    def _needs_real_backend(name):
        def fail(*args, **kwargs):
            raise RuntimeError(
                f'deepspeed.comm.{name} was called, but deepspeed is not installed and this '
                'test process is running single-process on CPU. This code path needs a real '
                'distributed backend: run it under deepspeed on a multi-GPU box instead of '
                'relying on the CPU test shim.'
            )
        return fail

    comm = types.ModuleType('deepspeed.comm.comm')
    comm.get_rank = lambda group=None: 0
    comm.get_world_size = lambda group=None: 1
    comm.get_world_group = lambda: None
    comm.is_initialized = lambda: False
    comm.barrier = lambda *args, **kwargs: None
    for _name in ('send', 'recv', 'broadcast', 'all_reduce', 'all_gather', 'reduce'):
        setattr(comm, _name, _needs_real_backend(_name))
    # 'from deepspeed import comm as dist' and 'import deepspeed.comm.comm as dist' have to
    # reach the same functions, so the package and its submodule are the same object.
    comm.comm = comm

    ds_logging = types.ModuleType('deepspeed.utils.logging')
    ds_logging.logger = logging.getLogger('deepspeed')

    ds_utils = types.ModuleType('deepspeed.utils')
    ds_utils.logging = ds_logging

    deepspeed = types.ModuleType('deepspeed')
    deepspeed.comm = comm
    deepspeed.utils = ds_utils

    modules = {
        'deepspeed': deepspeed,
        'deepspeed.comm': comm,
        'deepspeed.comm.comm': comm,
        'deepspeed.utils': ds_utils,
        'deepspeed.utils.logging': ds_logging,
    }
    # A module built with types.ModuleType has __spec__ = None, and importlib.util.find_spec
    # raises ValueError rather than returning None for such an entry in sys.modules.
    # transformers probes for deepspeed exactly that way, so every stub needs a real spec.
    # It still reports deepspeed as unavailable, because it also requires the package metadata
    # and there is no dist-info to find -- which is the answer we want.
    for name, module in modules.items():
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules.update(modules)


def _bootstrap_spawned_children():
    """Put test/childenv on PYTHONPATH so spawned subprocesses get the same shims.

    Hugging Face datasets maps with a process pool. On Windows those workers are spawned, not
    forked, so they start as bare interpreters that never see anything conftest did and cannot
    even import utils.dataset. test/childenv/sitecustomize.py fixes that up at their startup;
    this is what makes them find it. Prepended rather than assigned, so an existing PYTHONPATH
    still wins for everything else.
    """
    childenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'childenv')
    existing = os.environ.get('PYTHONPATH', '')
    entries = existing.split(os.pathsep) if existing else []
    if childenv not in entries:
        os.environ['PYTHONPATH'] = os.pathsep.join([childenv] + entries)


def _limit_dataset_workers():
    """Map datasets in-process where a worker has to be spawned rather than forked.

    utils/dataset.py maps with min(8, cpu_count()) workers, which is right for training on
    Linux, where fork makes a worker nearly free. Windows spawns instead, so each one re-imports
    torch, deepspeed and ComfyUI -- about 45 seconds of startup to map a handful of rows, which
    turns this suite from three minutes into well over ten.

    Only set where it applies, and never over a value the caller chose, so a Linux run still
    exercises the real multiprocess path.
    """
    if sys.platform != 'win32':
        return
    os.environ.setdefault('DIFFUSION_PIPE_NUM_PROC', '1')


_stub_deepspeed()
_bootstrap_spawned_children()
_limit_dataset_workers()

# Claim the 'utils' namespace package before ComfyUI's own utils/ directory can shadow it.
# ComfyUI gets prepended to sys.path by models/base.py, and diffusion-pipe's utils/ has no
# __init__.py, so without this 'import utils.common' resolves to the wrong directory.
import utils.common  # noqa: F401


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


class _StubModule(types.ModuleType):
    """Module whose every attribute is a permissive dummy class."""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return type(name, (), {'__init__': lambda self, *args, **kwargs: None})


class _StubFinder(importlib.abc.MetaPathFinder):
    """Resolves comfy_aimdo and any of its submodules to permissive stubs."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'comfy_aimdo' and not fullname.startswith('comfy_aimdo.'):
            return None
        spec = importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        spec.submodule_search_locations = []
        return spec


def _stub_comfy_aimdo():
    if importlib.util.find_spec('comfy_aimdo') is not None:
        return  # the real package is installed; leave it alone
    sys.meta_path.insert(0, _StubFinder())


def _force_comfy_cpu():
    import torch

    if torch.cuda.is_available():
        return
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'ComfyUI')))
    try:
        import comfy.cli_args
    except ImportError:
        return
    # ComfyUI calls torch.cuda.current_device() at import time unless it is told to use CPU.
    comfy.cli_args.args.cpu = True


_stub_comfy_aimdo()
_force_comfy_cpu()
