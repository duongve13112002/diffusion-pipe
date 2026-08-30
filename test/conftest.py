"""Test-environment shims so the model pipelines can be imported on a CPU-only machine.

models/base.py imports ComfyUI, which at import time pulls in comfy_aimdo (a GPU memory
management extension) and queries the current CUDA device. Neither is available on a plain
CPU box, so the pipeline modules cannot even be imported there.

Both shims are conditional: if the real comfy_aimdo is installed, or CUDA is available, this
file changes nothing. That keeps the tests honest in a real training environment while still
letting the CPU-only tests run in CI or on a laptop.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

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
