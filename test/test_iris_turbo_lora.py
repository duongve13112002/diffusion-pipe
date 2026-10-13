"""tools/iris_turbo_lora.py on CPU: tiny Iris teachers, a deterministic stand-in text encoder.

The objective itself is Iris's (iris3b.distill.objective), tested upstream; these tests cover what
the LoRA trainer adds: one shared base acting as teacher, student, fake and EMA through adapters,
gradients that stay right under activation checkpointing, exact resume, multi-process averaging,
and an export that tools/iris_merge_adapter.py folds into a model sampling like the student.

Run as a script, this file is the multi-process harness: `torchrun ... test_iris_turbo_lora.py
--config cfg.toml` trains with the stand-in encoder.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'submodules' / 'iris-3b' / 'src'))

from iris3b.text.base import TextEncoding  # noqa: E402

from tools import iris_turbo_lora as lora  # noqa: E402

TEXT_DIM, TEXT_LEN, HIDDEN_LAYERS = 32, 16, [1, 2, 4]


class StandInEncoder:
    """Deterministic per-prompt encodings in Iris's layerwise layout [B, L, layers, dim]."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.moves = 0

    def to(self, device):
        self.device = torch.device(device)
        self.moves += 1
        return self

    def _one(self, prompt):
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        n = 1 + seed % (TEXT_LEN - 1)
        mask = torch.zeros(TEXT_LEN, dtype=torch.long)
        mask[:n] = 1
        return torch.randn(TEXT_LEN, len(HIDDEN_LAYERS), TEXT_DIM, generator=g), mask

    def encode(self, prompts):
        rows = [self._one(p) for p in prompts]
        return TextEncoding(torch.stack([r[0] for r in rows]).to(self.device, torch.bfloat16),
                            torch.stack([r[1] for r in rows]).to(self.device))

    def null(self, negative_prompt=''):
        return self.encode([negative_prompt])


def stand_in(base, cfg, device):
    return StandInEncoder('cpu' if cfg.text_encoder == 'offload' else device)


def tiny_raw():
    return {'model': dict(hidden_size=64, depth=4, dual_depth=2, num_heads=4, num_kv_heads=2, patch_size=4,
                          text_dim=TEXT_DIM, text_len=TEXT_LEN, text_lap_num_layers=len(HIDDEN_LAYERS),
                          text_lap_num_heads=4, adaln_zero_init=False, repa_layer=2,
                          pixel=dict(hidden_size=8, attn_hidden_size=32, num_heads=4, depth=2)),
            'text_encoder': dict(dim=TEXT_DIM, max_length=TEXT_LEN, hidden_layers=HIDDEN_LAYERS),
            'flow': dict(shift=3.0)}


def write_teacher(directory, seed=0):
    from omegaconf import OmegaConf
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(tiny_raw()), str(directory / 'config.yaml'))
    torch.manual_seed(seed)
    ref = IrisDiT(inference_config(tiny_raw()).model)
    with torch.no_grad():
        for p in ref.parameters():  # default init zeroes the head and some biases
            p.add_(torch.randn_like(p) * 0.05)
    safetensors.torch.save_file({k: v.contiguous() for k, v in ref.state_dict().items()},
                                str(directory / 'model.safetensors'))
    return ref.eval()


CONFIG = '''teacher = "{teacher}"
prompts_file = "{prompts}"
output_dir = "{output}"
resolutions = [[32, 32], [32, 48]]
sample_size = [32, 32]
max_steps = 3
save_every = 2
log_every = 1
sample_every = 0
base_dtype = "float32"
[adapter]
rank = 4
alpha = 8
fake_rank = 4
fake_alpha = 8
'''


@pytest.fixture
def run(tmp_path, monkeypatch):
    """(write_config, train): a config in tmp_path and an in-process run of the tool."""
    write_teacher(tmp_path / 'teacher')
    (tmp_path / 'prompts.txt').write_text(''.join(f'prompt {i}\n' for i in range(23)), encoding='utf-8')
    monkeypatch.setattr(lora, 'build_text_encoder', stand_in)

    def config(name='cfg.toml', output='out', extra=''):
        path = tmp_path / name
        path.write_text(CONFIG.format(teacher=tmp_path / 'teacher', prompts=tmp_path / 'prompts.txt',
                                      output=tmp_path / output) + extra, encoding='utf-8')
        return path

    def train(*argv):
        return lora.main([str(a) for a in argv])
    return config, train


def adapter_file(directory):
    return safetensors.torch.load_file(str(directory / 'adapter_model.safetensors'))


# ---- adapters on one base -----------------------------------------------------------------------

def test_teacher_is_the_base_and_fresh_adapters_change_nothing(tmp_path):
    ref = write_teacher(tmp_path / 'teacher')
    model, base, _, identity = lora.load_teacher(tmp_path / 'teacher', torch.device('cpu'), torch.float32)
    layers = lora.inject(model, lora.target_paths(model, lora.DEFAULT_TARGETS))
    assert not any(p.split('.')[0] == 'y_embedder' for p in layers)  # the text adapter stays frozen
    torch.manual_seed(0)
    for layer in layers.values():
        layer.add('student', 4, 4)
    model.eval()
    g = torch.Generator().manual_seed(3)
    x, t = torch.randn(2, 3, 32, 32, generator=g), torch.tensor([300.0, 800.0])
    y = torch.randn(2, TEXT_LEN, len(HIDDEN_LAYERS), TEXT_DIM, generator=g)
    mask = torch.ones(2, TEXT_LEN, dtype=torch.long)
    with torch.no_grad():
        expected = ref(x, t, y, y_mask=mask).x
        with lora.adapter(None):
            teacher = model(x, t, y, y_mask=mask).x
        with lora.adapter('student'):
            student = model(x, t, y, y_mask=mask).x
        torch.testing.assert_close(teacher, expected)
        torch.testing.assert_close(student, expected)  # lora_B starts at zero
        for layer in layers.values():
            torch.nn.init.normal_(layer.lora_B['student'].weight, std=0.1)
        with lora.adapter('student'):
            assert not torch.allclose(model(x, t, y, y_mask=mask).x, expected, atol=1e-3)
        with lora.adapter(None):
            torch.testing.assert_close(model(x, t, y, y_mask=mask).x, expected)
    assert len(identity['weights_sha256']) == 64


def trained_after_one_cycle(tmp_path, monkeypatch, checkpointing):
    monkeypatch.setattr(lora, 'build_text_encoder', stand_in)
    cfg_path = tmp_path / f'{checkpointing}.toml'
    cfg_path.write_text(CONFIG.format(teacher=tmp_path / 'teacher', prompts=tmp_path / 'prompts.txt',
                                      output=tmp_path / checkpointing), encoding='utf-8')
    cfg = lora.load_config(cfg_path, [f'activation_checkpointing="{checkpointing}"', 'max_grad_norm=0'])
    trainer = lora.build(cfg, torch.device('cpu'))
    # non-zero student and fake deltas: a backward recomputed under the wrong adapter would differ
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in trainer.student + trainer.fake:
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    trainer.cycle()
    return [p.detach().clone() for p in trainer.student + trainer.fake]


def test_activation_checkpointing_recomputes_under_the_right_adapter(tmp_path, monkeypatch):
    write_teacher(tmp_path / 'teacher')
    (tmp_path / 'prompts.txt').write_text('a\nb\nc\nd\n', encoding='utf-8')
    plain = trained_after_one_cycle(tmp_path, monkeypatch, 'none')
    checkpointed = trained_after_one_cycle(tmp_path, monkeypatch, 'full')
    for a, b in zip(plain, checkpointed):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


# ---- runs ---------------------------------------------------------------------------------------

def test_run_exports_adapters_checkpoints_and_previews(run, tmp_path, capsys):
    config, train = run
    cfg = config(extra='')
    trainer = train('--config', cfg, '--set', 'sample_every=2', '--set', 'sample_prompts=["a cat", "a dog"]')
    out = tmp_path / 'out'
    assert trainer.step == 3 and trainer.fake_steps == 6
    assert trainer.prompts.cursor == 3 * 3  # (1 student + 2 fake updates) x batch 1, per student update
    assert sorted(p.name for p in (out / 'checkpoints').iterdir()) == ['step_00000002.pt', 'step_00000003.pt']
    adapters = out / 'adapters' / 'step_00000003'
    state = adapter_file(adapters)
    assert all(k.startswith('diffusion_model.') and k.endswith(('.lora_A.weight', '.lora_B.weight')) for k in state)
    assert all(v.dtype == torch.bfloat16 for v in state.values())
    config_json = json.loads((adapters / 'adapter_config.json').read_text())
    assert config_json['r'] == 4 and config_json['lora_alpha'] == 8
    from omegaconf import OmegaConf
    distill = OmegaConf.load(str(adapters / 'distill.yaml'))
    assert list(distill.dmd.sigmas) == [1.0, 0.938, 0.833, 0.625] and distill.lora_weights == 'ema'
    for step in (0, 2, 3):
        assert sorted(p.name for p in (out / 'samples' / f'step_{step:08d}').iterdir()) == \
            ['000.png', '001.png', 'prompts.txt']
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert [r['step'] for r in rows] == [1, 2, 3]
    assert all(torch.isfinite(torch.tensor(r['generator_loss'])) for r in rows)

    # a restart of a finished run does nothing
    assert train('--config', cfg) is None


def test_resume_continues_exactly(run, tmp_path):
    config, train = run
    straight = train('--config', config('a.toml', 'straight'), '--set', 'max_steps=4')
    train('--config', config('b.toml', 'resumed'), '--set', 'max_steps=2')
    resumed = train('--config', config('b.toml', 'resumed'), '--set', 'max_steps=4')
    assert resumed.step == 4 and resumed.prompts.cursor == straight.prompts.cursor
    for name in ('student', 'fake', 'ema'):
        for a, b in zip(lora.adapter_params(straight.layers, name), lora.adapter_params(resumed.layers, name)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = adapter_file(tmp_path / 'straight' / 'adapters' / 'step_00000004')
    b = adapter_file(tmp_path / 'resumed' / 'adapters' / 'step_00000004')
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resume_refuses_changed_settings_and_prompts(run, tmp_path):
    config, train = run
    cfg = config()
    train('--config', cfg, '--set', 'max_steps=2')
    with pytest.raises(lora.ConfigError, match='adapter'):
        train('--config', cfg, '--set', 'adapter.rank=8')
    with pytest.raises(lora.ConfigError, match='batch_size'):
        train('--config', cfg, '--set', 'batch_size=2')
    # allowed: run length, learning rates, previews
    assert train('--config', cfg, '--set', 'max_steps=3', '--set', 'learning_rate=1e-5').step == 3
    (tmp_path / 'prompts.txt').write_text('changed\n', encoding='utf-8')
    with pytest.raises(lora.ConfigError, match='prompts_file content changed'):
        train('--config', cfg, '--set', 'max_steps=4')


def test_torn_checkpoint_is_set_aside(run, tmp_path, capsys):
    config, train = run
    cfg = config()
    train('--config', cfg, '--set', 'max_steps=4', '--set', 'keep_last=0')
    checkpoints = tmp_path / 'out' / 'checkpoints'
    newest = checkpoints / 'step_00000004.pt'
    newest.write_bytes(newest.read_bytes()[:1000])
    trainer = train('--config', cfg, '--set', 'max_steps=4', '--set', 'keep_last=0')
    assert trainer.step == 4 and (checkpoints / 'step_00000004.pt.torn').exists()
    assert 'unreadable' in capsys.readouterr().out


def test_offload_and_bf16_base_and_accumulation(run, tmp_path):
    config, train = run
    trainer = train('--config', config(), '--set', 'text_encoder="offload"', '--set', 'base_dtype="bfloat16"',
                    '--set', 'gradient_accumulation_steps=2', '--set', 'max_steps=2', '--set', 'fake_updates=1')
    assert trainer.step == 2 and trainer.prompts.cursor == 2 * (1 + 1) * 2
    assert trainer.text_encoder.device.type == 'cpu' and trainer.text_encoder.moves >= 2
    blocks = trainer.model.blocks[0]
    linear = next(m for m in blocks.modules() if isinstance(m, lora.LoRALinear))
    assert linear.base_layer.weight.dtype == torch.bfloat16 and linear.lora_A['student'].weight.dtype == torch.float32
    assert trainer.model.final_layer.linear.base_layer.weight.dtype == torch.float32  # kept high precision
    assert all(torch.isfinite(p).all() for p in trainer.student + trainer.fake)


def test_init_from_warm_starts(run, tmp_path):
    config, train = run
    first = train('--config', config('a.toml', 'first'), '--set', 'max_steps=2')
    second = train('--config', config('b.toml', 'second'), '--set', 'max_steps=1',
                   '--set', f'init_from="{tmp_path / "first" / "checkpoints" / "step_00000002.pt"}"',
                   '--set', 'learning_rate=0', '--set', 'fake_learning_rate=0')
    for a, b in zip(lora.adapter_params(first.layers, 'ema'), lora.adapter_params(second.layers, 'student')):
        torch.testing.assert_close(a, b)
    assert second.step == 1


def test_config_errors(run, tmp_path):
    config, _ = run
    cfg = config()
    with pytest.raises(lora.ConfigError, match='unknown config keys: bogus'):
        lora.load_config(cfg, ['bogus=1'])
    with pytest.raises(lora.ConfigError, match='keep_last = 1'):
        lora.load_config(cfg, ['keep_last=1'])
    with pytest.raises(lora.ConfigError, match='sigmas'):
        lora.load_config(cfg, ['sigmas=[0.9, 0.5]'])
    assert lora.load_config(cfg, ['adapter.rank=8', 'output_dir=/x/y']).adapter.rank == 8
    assert lora.load_config(cfg, ['output_dir=/x/y']).output_dir == '/x/y'


def test_example_config_is_the_defaults():
    from dataclasses import asdict
    example = asdict(lora.load_config(ROOT / 'examples' / 'iris' / 'turbo' / 'lora.toml'))
    defaults = asdict(lora.LoraDistillConfig())
    assert {k for k in example if example[k] != defaults[k]} == {'teacher', 'prompts_file', 'output_dir',
                                                                 'sample_prompts'}


def test_exported_adapter_merges_into_the_student(run, tmp_path):
    config, train = run
    trainer = train('--config', config(), '--set', 'save_dtype="float32"')
    adapters = tmp_path / 'out' / 'adapters' / 'step_00000003'
    from tools import iris_merge_adapter
    iris_merge_adapter.main(['--model', str(tmp_path / 'teacher'), '--adapter', str(adapters),
                             '--output', str(tmp_path / 'merged')])
    assert (tmp_path / 'merged' / 'distill.yaml').read_text() == (adapters / 'distill.yaml').read_text()
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    merged = IrisDiT(inference_config(tiny_raw()).model)
    merged.load_state_dict(safetensors.torch.load_file(str(tmp_path / 'merged' / 'model.safetensors')))
    merged.eval()
    trainer.model.eval()
    g = torch.Generator().manual_seed(9)
    x, t = torch.randn(1, 3, 32, 32, generator=g), torch.tensor([500.0])
    y = torch.randn(1, TEXT_LEN, len(HIDDEN_LAYERS), TEXT_DIM, generator=g)
    mask = torch.ones(1, TEXT_LEN, dtype=torch.long)
    with torch.no_grad(), lora.adapter('ema'):
        torch.testing.assert_close(merged(x, t, y, y_mask=mask).x, trainer.model(x, t, y, y_mask=mask).x,
                                   rtol=1e-4, atol=1e-5)


# ---- several processes --------------------------------------------------------------------------

def test_two_processes_average_gradients(run, tmp_path):
    config, train = run
    cfg = config()
    env = {**os.environ, 'OMP_NUM_THREADS': '1'}
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               __file__, '--config', str(cfg), '--set', 'max_steps=2', '--dump', str(tmp_path / 'dump')]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    ranks = [torch.load(tmp_path / 'dump' / f'rank{r}.pt') for r in (0, 1)]
    assert ranks[0]['cursor'] == ranks[1]['cursor'] == 2 * 3 * 2  # 2 updates x 3 batches x 2 ranks
    for a, b in zip(ranks[0]['params'], ranks[1]['params']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)  # replicas stay identical
    assert not torch.equal(ranks[0]['params'][1], torch.zeros_like(ranks[0]['params'][1]))  # it trained
    payload = torch.load(tmp_path / 'out' / 'checkpoints' / 'step_00000002.pt', weights_only=False)
    assert payload['world_size'] == 2 and len(payload['generators']) == 2
    # one process cannot continue a two-process run
    with pytest.raises(lora.ConfigError, match='same GPU count'):
        train('--config', cfg, '--set', 'max_steps=3')


def harness(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump')
    args, rest = parser.parse_known_args(argv)
    lora.build_text_encoder = stand_in
    trainer = lora.main(rest)
    if args.dump and trainer is not None:
        Path(args.dump).mkdir(exist_ok=True)
        torch.save({'cursor': trainer.prompts.cursor,
                    'params': [p.detach().clone() for p in trainer.student + trainer.fake]},
                   Path(args.dump) / f'rank{int(os.environ.get("RANK", 0))}.pt')


if __name__ == '__main__':
    harness(sys.argv[1:])
