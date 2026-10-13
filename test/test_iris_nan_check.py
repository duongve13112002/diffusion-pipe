"""tools/iris_nan_check.py names the stage a NaN comes from. Runs the tool in a subprocess: it
initialises a process group, which must not leak into the other tests."""

import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch
import toml
import torch
from PIL import Image

from test.test_iris import ROOT, tiny_qwen, write_checkpoint  # noqa: F401  (tiny_qwen is a fixture)

TOOL = ROOT / 'tools' / 'iris_nan_check.py'


def setup(tmp_path, qwen, poison=None):
    write_checkpoint(tmp_path / 'ckpt')
    if poison:
        weights = tmp_path / 'ckpt' / 'model.safetensors'
        state = safetensors.torch.load_file(str(weights))
        state[poison] = state[poison].clone()
        state[poison].view(-1)[0] = float('inf')
        safetensors.torch.save_file(state, str(weights))
    data = tmp_path / 'data'
    data.mkdir()
    for i in range(3):
        Image.new('RGB', (48, 32), (i * 60, 90, 200)).save(data / f'{i}.png')
        (data / f'{i}.txt').write_text(f'tag{i}, red')
    (tmp_path / 'dataset.toml').write_text(toml.dumps({'resolutions': [32], 'directory': [{'path': str(data)}]}))
    config = tmp_path / 'train.toml'
    config.write_text(toml.dumps({
        'output_dir': str(tmp_path / 'out'), 'dataset': str(tmp_path / 'dataset.toml'),
        'save_every_n_epochs': 1, 'gradient_clipping': 1.0,
        'model': {'type': 'iris', 'transformer_path': str(tmp_path / 'ckpt'), 'text_encoder_path': str(qwen),
                  'dtype': 'float32', 'text_encoder_dtype': 'float32'},
        'optimizer': {'type': 'adamw', 'lr': 1e-4, 'weight_decay': 0.01},
    }))
    return config


def run(config, *flags):
    code = ('import sys, runpy; sys.path.insert(0, %r); import test.conftest; '
            'sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")') % str(ROOT)
    result = subprocess.run([sys.executable, '-c', code, str(TOOL), '--config', str(config), '--device', 'cpu',
                             '--batch-size', '2', *flags], cwd=ROOT, capture_output=True, text=True, timeout=600)
    return result.returncode, result.stdout


def test_finite_run_goes_through_every_stage(tmp_path, tiny_qwen):  # noqa: F811
    code, out = run(setup(tmp_path, tiny_qwen), '--steps', '2')
    assert code == 0, out
    for stage in ('[1/4]', '[2/4]', '[3/4]', '[4/4]', 'FinalLayer', 'step 2: loss', 'Everything stayed finite'):
        assert stage in out


def test_non_finite_checkpoint_is_named(tmp_path, tiny_qwen):  # noqa: F811
    code, out = run(setup(tmp_path, tiny_qwen, poison='final_layer.linear.weight'))
    assert code == 1
    assert 'NON-FINITE final_layer.linear.weight' in out and 'checkpoint itself' in out


def test_forward_nan_is_located_at_its_layer(tmp_path, tiny_qwen):  # noqa: F811
    code, out = run(setup(tmp_path, tiny_qwen, poison='final_layer.linear.weight'), '--skip-checkpoint-scan')
    assert code == 1
    assert 'NON-FINITE layer' in out and 'FinalLayer' in out.split('NON-FINITE layer')[1].splitlines()[0]
    assert 'forward pass produces NaN/inf' in out


def test_forward_only_stops_after_the_loss(tmp_path, tiny_qwen):  # noqa: F811
    code, out = run(setup(tmp_path, tiny_qwen), '--forward-only')
    assert code == 0, out
    assert 'forward pass and the loss are finite at batch size 2' in out and '[4/4]' not in out


def test_activation_checkpointing_gives_the_same_loss_and_gradients(tmp_path, tiny_qwen):  # noqa: F811
    config = setup(tmp_path, tiny_qwen)
    plain = run(config, '--steps', '2')
    checkpointed = run(config, '--steps', '2', '--activation-checkpointing')
    assert plain[0] == checkpointed[0] == 0, checkpointed[1]
    pick = lambda out: [line for line in out.splitlines() if 'loss' in line or 'gradient norm' in line]  # noqa: E731
    assert pick(plain[1]) == pick(checkpointed[1]) and pick(plain[1])
