"""A missing torchaudio must say what to install, not fail from inside a vendored file.

torch, torchvision and torchaudio are all absent from requirements.txt on purpose: they have to
come from the same PyTorch index, chosen for the machine's CUDA version, so the README has the
user install them together. Missing one is therefore a normal install slip.

torchaudio is the slip that used to be worst. models/base.py is imported by every model in the
repo, and it needs torchaudio twice over -- once for its own video-audio resample, and again
because `import comfy.sd` pulls in comfy.ldm.lightricks.vae.audio_vae, which imports torchaudio
at module scope. So somebody training images, who will never touch an audio track, still lost
every model in the repo, and the error named a Lightricks VAE file they had no reason to have
heard of.

The fix is ordering plus a message: import torchaudio ahead of the comfy imports, guarded. These
tests pin both, because the value is entirely in the ordering -- a guard placed after the comfy
imports would never run.
"""

import ast
import importlib.abc
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASE_PY = REPO / 'models' / 'base.py'


def _torchaudio_guard_and_first_comfy_import(tree):
    """Line numbers of the guarded torchaudio import, and of the first comfy import."""
    guard_line = None
    first_comfy_line = None
    for node in tree.body:
        if guard_line is None and isinstance(node, ast.Try):
            imports_torchaudio = any(
                isinstance(stmt, ast.Import) and any(a.name == 'torchaudio' for a in stmt.names)
                for stmt in node.body
            )
            catches_missing_module = any(
                isinstance(h.type, ast.Name) and h.type.id == 'ModuleNotFoundError'
                for h in node.handlers
            )
            if imports_torchaudio and catches_missing_module:
                guard_line = node.lineno
        if first_comfy_line is None:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            if any(n == 'comfy' or n.startswith('comfy') for n in names):
                first_comfy_line = node.lineno
    return guard_line, first_comfy_line


class TestTheGuardIsWhereItHasToBe:
    def test_torchaudio_is_imported_under_a_guard_before_any_comfy_import(self):
        tree = ast.parse(BASE_PY.read_text(encoding='utf-8'))
        guard_line, first_comfy_line = _torchaudio_guard_and_first_comfy_import(tree)

        assert guard_line is not None, (
            'models/base.py no longer imports torchaudio inside a try/except ModuleNotFoundError'
        )
        assert first_comfy_line is not None, 'models/base.py no longer imports comfy at module scope'
        assert guard_line < first_comfy_line, (
            'the torchaudio guard must come before the comfy imports. comfy.sd imports '
            'comfy.ldm.lightricks.vae.audio_vae, which imports torchaudio at module scope, so a '
            'guard placed after it never runs and the user gets the vendored file\'s bare '
            'ModuleNotFoundError instead of the message naming the fix.'
        )


class _BlockTorchaudio(importlib.abc.MetaPathFinder):
    """Make `import torchaudio` fail the way an uninstalled torchaudio fails."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torchaudio' or fullname.startswith('torchaudio.'):
            raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)
        return None


class TestTheMessage:
    def test_it_names_the_package_and_the_install_command(self, monkeypatch):
        # Execute models/base.py into a throwaway module rather than reloading the real one, so a
        # failed import cannot leave sys.modules['models.base'] half-built for later tests. The
        # guard is above the comfy imports, so execution stops there and nothing heavyweight runs;
        # sys.path is copied because the file appends ComfyUI to it on the way past.
        monkeypatch.setattr(sys, 'path', list(sys.path))
        for name in [n for n in sys.modules if n == 'torchaudio' or n.startswith('torchaudio.')]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, 'meta_path', [_BlockTorchaudio()] + sys.meta_path)

        spec = importlib.util.spec_from_file_location('_models_base_torchaudio_probe', BASE_PY)
        module = importlib.util.module_from_spec(spec)

        with pytest.raises(ModuleNotFoundError) as excinfo:
            spec.loader.exec_module(module)

        message = str(excinfo.value)
        assert 'torchaudio' in message
        assert 'pip install torch torchvision torchaudio' in message
        assert 'README' in message
        # The original is kept as the cause, so the traceback still shows the real import error.
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
        assert excinfo.value.__cause__.name == 'torchaudio'
