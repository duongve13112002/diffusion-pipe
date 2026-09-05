"""Regression coverage for the ComfyUI/utils import-shadow bug.

models/base.py puts submodules/ComfyUI on sys.path so it can import comfy.*. ComfyUI ships its
own utils/ directory, and it has an __init__.py, making it a *regular* package; diffusion-pipe's
own utils/ has no __init__.py, making it a *namespace* package. Per PEP 420, a regular package
always wins a namespace package for the same top-level name, on every sys.path entry order --
there is no sys.path ordering that lets the namespace package win once both are reachable. The
only thing that avoids the collision is importing diffusion-pipe's utils.common before ComfyUI
ever joins sys.path, so the later 'from utils.common import ...' is served from sys.modules
instead of triggering a fresh path search.

train.py never hits this by accident, because it imports utils.dataset before importing any
models/*.py module. tools/distill_refiner.py imports models.cosmos_predict2 first, so before
this fix it failed on a real multi-GPU box with 'ModuleNotFoundError: No module named
utils.common' as soon as models/base.py's ComfyUI sys.path insert ran ahead of its own
utils.common import.

The subprocess tests below reproduce the real shadow with the project's actual files (not a
toy stand-in), which is possible without any GPU or ComfyUI shims because
submodules/ComfyUI/utils/__init__.py is empty -- the CUDA-touching code lives under comfy/,
never under ComfyUI's utils/. PYTHONPATH is stripped of test/childenv for these two subprocess
calls, because its sitecustomize.py itself pre-claims the utils namespace for spawned dataset
workers and would otherwise mask exactly the bug being reproduced.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(script):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    return subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, cwd=REPO, env=env,
    )


class TestModelsBaseImportOrder:
    def test_utils_common_is_imported_before_comfyui_joins_sys_path(self):
        tree = ast.parse((REPO / 'models' / 'base.py').read_text(encoding='utf-8'))
        utils_common_line = None
        comfy_insert_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'utils.common':
                utils_common_line = node.lineno
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'insert' and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == 'path'):
                source = ast.get_source_segment((REPO / 'models' / 'base.py').read_text(encoding='utf-8'), node)
                if source and 'ComfyUI' in source:
                    comfy_insert_line = node.lineno
        assert utils_common_line is not None, 'models/base.py no longer imports utils.common the expected way'
        assert comfy_insert_line is not None, 'models/base.py no longer inserts ComfyUI onto sys.path the expected way'
        assert utils_common_line < comfy_insert_line, (
            'utils.common must be imported before ComfyUI is added to sys.path, or a regular '
            'package (ComfyUI/utils/__init__.py) shadows the namespace package (utils/) and any '
            'entry point that imports a model before importing utils.* breaks with '
            "ModuleNotFoundError: No module named 'utils.common'"
        )

    def test_the_shadow_reproduces_with_the_real_files_in_the_broken_order(self):
        result = _run(
            "import sys, os\n"
            "sys.path.insert(0, os.path.join('submodules', 'ComfyUI'))\n"
            "from utils.common import is_main_process\n"
        )
        assert result.returncode != 0, 'expected the ComfyUI-first order to reproduce the shadow'
        assert 'ModuleNotFoundError' in result.stderr and 'utils.common' in result.stderr

    def test_the_shadow_is_avoided_with_the_real_files_in_the_fixed_order(self):
        result = _run(
            "from utils.common import is_main_process\n"
            "import sys, os\n"
            "sys.path.insert(0, os.path.join('submodules', 'ComfyUI'))\n"
            "print('OK')\n"
        )
        assert result.returncode == 0, f'expected the fixed order to succeed, stderr:\n{result.stderr}'
        assert 'OK' in result.stdout
