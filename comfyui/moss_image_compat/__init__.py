"""ComfyUI custom node: the MOSS-Image node pack on a ComfyUI that predates it.

The pack (submodules/MOSS-Image/comfyui/moss-image) is written against a ComfyUI that has
comfy.storage and a fast_disk argument on ModelPatcher. ComfyUI 3216c62 has neither, so the pack
fails to import there. This node adds both, only where they are missing, and then loads the pack:

  comfy.storage   A stand-in reporting no fast disk (state_dict_fast_disk and friends). That is
                  what 3216c62 does for every model it loads; its global --fast-disk flag is
                  unaffected.
  fast_disk=...   Accepted and dropped by ModelPatcher and ModelPatcherDynamic.

On a ComfyUI that already has both, nothing is patched and this is the pack unchanged.

Install one of these into ComfyUI/custom_nodes, not both (they register the same nodes): this
directory, or the pack itself on a ComfyUI recent enough for it. The pack is found at, in order:

  1. the MOSS_IMAGE_NODE_PATH environment variable (the pack's directory);
  2. moss-image/ inside this directory (copy the pack here for a self-contained install);
  3. submodules/MOSS-Image/comfyui/moss-image of the diffusion-pipe checkout this directory is
     in (works through a symlink from custom_nodes).
"""

import functools
import importlib.util
import inspect
import logging
import os
import sys
import types
from pathlib import Path

PACK_MODULE = 'moss_image_compat_pack'
_COMPAT = '_moss_image_compat'


def pack_candidates():
    here = Path(__file__).resolve().parent
    candidates = []
    if os.environ.get('MOSS_IMAGE_NODE_PATH'):
        candidates.append(Path(os.environ['MOSS_IMAGE_NODE_PATH']).expanduser())
    candidates.append(here / 'moss-image')
    candidates.append(here.parent.parent / 'submodules' / 'MOSS-Image' / 'comfyui' / 'moss-image')
    return candidates


def find_pack():
    candidates = pack_candidates()
    if os.environ.get('MOSS_IMAGE_NODE_PATH'):
        # Named explicitly, so a wrong path is an error, not quietly replaced by another copy.
        if (candidates[0] / '__init__.py').is_file():
            return candidates[0].resolve()
        raise RuntimeError(f'MOSS_IMAGE_NODE_PATH={os.environ["MOSS_IMAGE_NODE_PATH"]} is not the MOSS-Image '
                           'node pack (no __init__.py there): point it at comfyui/moss-image of MOSS-Image.')
    for path in candidates:
        if (path / '__init__.py').is_file():
            return path.resolve()
    tried = '\n  '.join(str(p) for p in candidates)
    raise RuntimeError(
        'moss_image_compat could not find the MOSS-Image node pack. Tried:\n  ' + tried + '\nSet '
        'MOSS_IMAGE_NODE_PATH to its comfyui/moss-image directory, or copy that directory here as moss-image/.')


def storage_shim():
    """comfy.storage for a ComfyUI without one: no model is on a fast disk."""
    module = types.ModuleType('comfy.storage')
    module.__doc__ = 'moss_image_compat stand-in for comfy.storage: reports no fast disk.'
    module.fast_storage = lambda path: None
    module.annotate_state_dict = lambda state_dict, path: None
    module.model_fast_disk = lambda paths: False
    module.state_dict_fast_disk = lambda state_dict: False
    setattr(module, _COMPAT, True)
    return module


def install_storage(comfy_package):
    """Add comfy.storage if this ComfyUI has none. True if it was added."""
    if 'comfy.storage' in sys.modules or importlib.util.find_spec('comfy.storage') is not None:
        return False
    module = storage_shim()
    sys.modules['comfy.storage'] = module
    comfy_package.storage = module
    return True


def _accepts(function, name):
    parameters = inspect.signature(function).parameters
    return name in parameters or any(p.kind is p.VAR_KEYWORD for p in parameters.values())


def accept_fast_disk(cls):
    """Let cls take (and drop) fast_disk where its own __init__ / __new__ does not. True if patched."""
    patched = False
    init = cls.__dict__.get('__init__')
    if init is not None and not getattr(init, _COMPAT, False) and not _accepts(init, 'fast_disk'):
        @functools.wraps(init)
        def __init__(self, *args, fast_disk=False, **kwargs):
            init(self, *args, **kwargs)
        setattr(__init__, _COMPAT, True)
        cls.__init__ = __init__
        patched = True
    new = cls.__dict__.get('__new__')
    if new is not None:
        new = new.__func__ if isinstance(new, staticmethod) else new
        if not getattr(new, _COMPAT, False) and not _accepts(new, 'fast_disk'):
            @functools.wraps(new)
            def __new__(klass, *args, fast_disk=False, **kwargs):
                return new(klass, *args, **kwargs)
            setattr(__new__, _COMPAT, True)
            cls.__new__ = staticmethod(__new__)
            patched = True
    return patched


def install():
    """Patch what this ComfyUI lacks; returns the names of what was patched."""
    import comfy
    import comfy.model_patcher
    patched = []
    if install_storage(comfy):
        patched.append('comfy.storage')
    for name in ('ModelPatcher', 'ModelPatcherDynamic'):
        cls = getattr(comfy.model_patcher, name, None)
        if isinstance(cls, type) and accept_fast_disk(cls):
            patched.append(f'{name}(fast_disk=)')
    return patched


def load_pack(path=None):
    """Import the pack as a package (its modules import each other relatively)."""
    if PACK_MODULE in sys.modules:
        return sys.modules[PACK_MODULE]
    path = find_pack() if path is None else Path(path)
    spec = importlib.util.spec_from_file_location(PACK_MODULE, path / '__init__.py',
                                                  submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in [n for n in sys.modules if n == PACK_MODULE or n.startswith(PACK_MODULE + '.')]:
            del sys.modules[name]
        raise
    return module


_patched = install()
_pack = load_pack()
logging.info('moss_image_compat: MOSS-Image node pack from %s%s', Path(_pack.__file__).parent,
             f' (added for this ComfyUI: {", ".join(_patched)})' if _patched else '')

comfy_entrypoint = _pack.comfy_entrypoint

__all__ = ['comfy_entrypoint']
