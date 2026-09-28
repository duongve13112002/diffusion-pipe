"""Applies the CPU-only shims inside processes that pytest never gets to configure.

test/conftest.py sets up the ComfyUI shims for the process pytest runs in, but Hugging Face
datasets maps with a process pool, and on Windows a pool worker is *spawned*: a fresh
interpreter that never imports conftest. It unpickles a reference to utils.dataset, which
pulls in ComfyUI, which queries the current CUDA device and dies. On Linux the workers are
forked and inherit the parent's already-shimmed state, which is why this only bites here.

Python imports sitecustomize automatically at interpreter startup, so putting this directory
on PYTHONPATH covers the children without the test code having to know how datasets spawns
them. conftest.py is what puts it there; nothing else should.

Everything is best-effort. This file runs before any application code in every process that
inherits the variable, so a failure here must never be what stops one from starting.
"""

import os
import sys


def _add_repo_root():
    # A spawned worker starts with only its own cwd-ish path entries, and diffusion-pipe's
    # utils/ is a namespace package with no __init__.py, so without the repo root it resolves
    # 'utils' to ComfyUI's own utils/ directory instead.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _claim_utils_namespace():
    """Pin diffusion-pipe's utils/ into sys.modules before ComfyUI's can take the name.

    ComfyUI ships a utils/ with an __init__.py, so it is a *regular* package. diffusion-pipe's
    has none, so it is a namespace package. A namespace package gathers portions from every
    sys.path entry, but a regular package ends the search outright -- so once ComfyUI's utils/
    is reachable and earlier on the path, utils.dataset stops existing.

    Importing a light module out of the real utils/ first settles it: sys.modules is consulted
    before the path is, and utils.captions pulls in nothing heavier than the standard library.
    """
    import utils.captions  # noqa: F401


def _force_comfy_cpu(repo_root):
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return
    # Appended, not inserted: ComfyUI must never come ahead of the repo root, for the reason
    # spelled out in _claim_utils_namespace.
    sys.path.append(os.path.join(repo_root, 'submodules', 'ComfyUI'))
    try:
        import comfy.cli_args
    except ImportError:
        return
    comfy.cli_args.args.cpu = True


try:
    _repo_root = _add_repo_root()
    _claim_utils_namespace()
    _force_comfy_cpu(_repo_root)
except Exception:
    pass
