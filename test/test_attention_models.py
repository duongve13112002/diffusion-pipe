"""Runs test/fake_flash_attn/check_models.py in a fresh process with the stand-in flash-attn 2 / 3 / 4
packages importable: HiDream imports flash-attn at module level, and the stand-ins must not leak
into the rest of the suite (Wan, for one, decides at import whether flash-attn 2 is there)."""

import os
import subprocess
import sys
from pathlib import Path


def test_models_with_fake_flash_attn(tmp_path):
    here = Path(__file__).parent
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(here / 'fake_flash_attn'), env.get('PYTHONPATH')]))
    log = tmp_path / 'log'
    with open(log, 'w') as f:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                 str(here / 'fake_flash_attn' / 'check_models.py')],
                                cwd=here.parent, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    output = log.read_text()
    assert result.returncode == 0, output[-6000:]
    assert ' passed' in output and 'failed' not in output
