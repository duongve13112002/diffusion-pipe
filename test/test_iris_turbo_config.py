"""Iris-3B Turbo from diffusion-pipe configs (utils/iris_turbo.py), on CPU.

train.py hands a config with a [turbo] table to utils/iris_turbo.py, which translates it for the
LoRA trainer ([adapter] present) or Iris's full distillation trainer, and writes the prompts from
the dataset's captions. The trainers themselves are tested in test_iris_turbo*.py.
"""

import copy
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import toml
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test.test_iris_turbo_lora import stand_in, write_teacher  # noqa: E402
from tools import iris_turbo_lora as lora  # noqa: E402
from utils import iris_turbo  # noqa: E402

EXAMPLES = ROOT / 'examples' / 'iris' / 'turbo'


def example(name):
    return toml.loads((EXAMPLES / name).read_text(encoding='utf-8'))


def write_dataset(tmp_path, captions=('a red fox', 'a bowl of ramen', 'a lighthouse at night')):
    images = tmp_path / 'images'
    images.mkdir()
    for i, caption in enumerate(captions):
        (images / f'{i}.png').write_bytes(b'not decoded')
        (images / f'{i}.txt').write_text(caption, encoding='utf-8')
    path = tmp_path / 'dataset.toml'
    path.write_text(toml.dumps({'resolutions': [64], 'directory': [{'path': str(images), 'num_repeats': 2}]}),
                    encoding='utf-8')
    return path


def lora_config(tmp_path):
    return {
        'output_dir': str(tmp_path / 'out'),
        'dataset': str(write_dataset(tmp_path)),
        'micro_batch_size_per_gpu': 1,
        'max_steps': 3,
        'save_every_n_steps': 2,
        'steps_per_print': 1,
        'activation_checkpointing': True,
        'model': {'type': 'iris', 'transformer_path': str(tmp_path / 'teacher'), 'dtype': 'float32'},
        'adapter': {'type': 'lora', 'rank': 4, 'alpha': 8, 'fake_rank': 4, 'fake_alpha': 8, 'dtype': 'bfloat16'},
        'optimizer': {'type': 'adamw', 'lr': 2e-5, 'fake_lr': 1e-4, 'betas': [0.0, 0.999]},
        'turbo': {'resolutions': [[32, 32], [32, 48]], 'sample_size': [32, 32], 'sample_every': 0},
        'monitoring': {'enable_wandb': False},
    }


# ---- translation --------------------------------------------------------------------------------

def test_lora_example_is_the_trainers_defaults():
    which, settings, notes = iris_turbo.translate(example('iris_3b_turbo_lora.toml'))
    assert which == 'lora' and notes == []
    cfg = asdict(lora.config_from_dict({**settings, 'prompts_file': 'p'}))
    defaults = asdict(lora.LoraDistillConfig())
    assert {k for k in cfg if cfg[k] != defaults[k]} == {'teacher', 'prompts_file', 'output_dir', 'sample_prompts'}


@pytest.mark.parametrize('stage', ['1', '2'])
def test_full_examples_are_the_iris_recipe(stage):
    which, settings, notes = iris_turbo.translate(example(f'iris_3b_turbo_stage{stage}.toml'))
    assert which == 'full' and notes == []
    assert (settings['keep_last'], settings['nccl_timeout_minutes']) == (2, 120)
    fields = iris_turbo.check_full({**settings, 'prompts_file': 'p'})
    from iris3b.distill.config import DistillConfig
    ours = OmegaConf.to_container(OmegaConf.merge(OmegaConf.structured(DistillConfig), fields))
    yaml = next(EXAMPLES.glob(f'stage{stage}_*.yaml'))
    recipe = OmegaConf.to_container(OmegaConf.merge(OmegaConf.structured(DistillConfig), OmegaConf.load(yaml)))
    paths = {'teacher', 'prompts_file', 'work_dir', 'init_from'}
    assert {k for k in recipe if ours[k] != recipe[k]} <= paths
    assert ours['dmd'] == recipe['dmd'] and ours['batch_size'] == recipe['batch_size']


def test_key_mapping():
    config = example('iris_3b_turbo_stage1.toml')
    config.update(activation_checkpointing=False, gradient_clipping=0.5, steps_per_print=3)
    config['turbo'].update(keep_last=4, nccl_timeout_minutes=30)
    _, settings, _ = iris_turbo.translate(config)
    assert settings['activation_checkpointing'] == 'none' and settings['max_grad_norm'] == 0.5
    assert settings['log_every'] == 3 and settings['batch_size'] == 8 and settings['save_every'] == 100
    assert settings['learning_rate'] == 1e-6 and settings['fake_learning_rate'] == 5e-6
    assert settings['dmd']['sigmas'][-1] == 0.333 and 'sigmas' not in settings
    assert (settings['keep_last'], settings['nccl_timeout_minutes']) == (4, 30)
    config = example('iris_3b_turbo_lora.toml')
    config['model']['dtype'] = 'float32'
    config['activation_checkpointing'] = 'selective_op'
    _, settings, _ = iris_turbo.translate(config)
    assert settings['base_dtype'] == 'float32' and settings['activation_checkpointing'] == 'selective_op'
    assert settings['adapter'] == {'rank': 64, 'alpha': 64, 'fake_rank': 64, 'fake_alpha': 64}


def test_unused_settings_are_noted():
    config = example('iris_3b_turbo_stage1.toml')
    config['save_dtype'] = 'bfloat16'
    config['model']['dtype'] = 'bfloat16'
    _, _, notes = iris_turbo.translate(config)
    assert [n.split(':')[0] for n in notes] == ['save_dtype', '[model] dtype']


@pytest.mark.parametrize('name, edit, message', [
    ('lora', lambda c: c.update(epochs=10), 'not used by Turbo distillation: epochs'),
    ('lora', lambda c: c.update(warmup_steps=100, eval_every_n_epochs=1), 'eval_every_n_epochs, warmup_steps'),
    ('lora', lambda c: c.update(caching_write_queue_mb=256, caching_report_seconds=0),
     'caching_report_seconds, caching_write_queue_mb'),
    ('stage1', lambda c: c.update(gradient_accumulation_steps=4), 'no gradient accumulation'),
    ('stage1', lambda c: c.update(pipeline_stages=2), 'pipeline_stages must be 1'),
    ('lora', lambda c: c['model'].update(type='flux'), "type must be 'iris'"),
    ('lora', lambda c: c['model'].pop('transformer_path'), 'transformer_path is required'),
    ('lora', lambda c: c['model'].update(text_encoder_path='/x'), 'text_encoder_path is not supported'),
    ('lora', lambda c: c.pop('output_dir'), 'output_dir is required'),
    ('lora', lambda c: c['adapter'].update(type='lokr'), "type must be 'lora'"),
    ('lora', lambda c: c['adapter'].update(oplora=True), 'oplora is not supported'),
    ('lora', lambda c: c['optimizer'].update(type='adamw_optimi'), "set type = 'adamw'"),
    ('lora', lambda c: c['optimizer'].update(eps=1e-8), 'eps is not configurable'),
    ('lora', lambda c: c['turbo'].update(strategy='fsdp2'), 'strategy only applies to full Turbo'),
    ('stage1', lambda c: c['turbo'].update(text_encoder='offload'), 'text_encoder only applies to a Turbo LoRA'),
    ('lora', lambda c: c['turbo'].update(bogus=1), 'bogus is not a Turbo setting'),
    ('lora', lambda c: c.update(monitoring={'enable_wandb': True}), 'no wandb support'),
])
def test_config_errors(name, edit, message):
    config = example('iris_3b_turbo_lora.toml' if name == 'lora' else 'iris_3b_turbo_stage1.toml')
    edit(config)
    with pytest.raises(iris_turbo.TurboConfigError, match=message):
        iris_turbo.translate(config)


def test_trainer_validation_runs_before_anything_starts(tmp_path):
    config = example('iris_3b_turbo_lora.toml')
    config['turbo']['keep_last'] = 1
    with pytest.raises(iris_turbo.TurboConfigError, match='keep_last = 1'):
        iris_turbo.run(config)
    config = example('iris_3b_turbo_stage1.toml')
    config['turbo']['sigmas'] = [0.9, 0.5]
    with pytest.raises(iris_turbo.TurboConfigError, match='dmd.sigmas must start at pure noise'):
        iris_turbo.run(config)
    assert not Path(config['output_dir']).exists()


def test_prompts_sources(tmp_path):
    config = {'output_dir': str(tmp_path), 'dataset': 'd.toml', 'turbo': {}}
    assert iris_turbo.prompts_plan(config) == ('build', str(tmp_path / 'prompts.txt'), ['d.toml'], [])
    config['turbo']['extra_prompts'] = 'general.txt'
    assert iris_turbo.prompts_plan(config)[3] == ['general.txt']
    assert iris_turbo.prompts_plan({'output_dir': 'o', 'turbo': {'prompts_file': 'p.txt'}}) == ('file', 'p.txt')
    with pytest.raises(iris_turbo.TurboConfigError, match='not both'):
        iris_turbo.prompts_plan({**config, 'turbo': {'prompts_file': 'p.txt'}})
    with pytest.raises(iris_turbo.TurboConfigError, match='Turbo needs prompts'):
        iris_turbo.prompts_plan({'output_dir': 'o', 'turbo': {}})


# ---- runs ---------------------------------------------------------------------------------------

def test_lora_run_from_a_dataset(tmp_path, monkeypatch):
    write_teacher(tmp_path / 'teacher')
    monkeypatch.setattr(lora, 'build_text_encoder', stand_in)
    config = lora_config(tmp_path)
    trainer = iris_turbo.run(copy.deepcopy(config))
    out = tmp_path / 'out'
    prompts = (out / 'prompts.txt').read_text(encoding='utf-8').split('\n')[:-1]
    assert sorted(prompts) == sorted(['a red fox', 'a bowl of ramen', 'a lighthouse at night'] * 2)
    assert trainer.step == 3 and trainer.cfg.base_dtype == 'float32' and trainer.cfg.prompts_file == str(
        out / 'prompts.txt')
    assert (out / 'adapters' / 'step_00000003' / 'adapter_model.safetensors').is_file()
    assert sorted(p.name for p in (out / 'checkpoints').iterdir()) == ['step_00000002.pt', 'step_00000003.pt']
    # the same command again: the prompts are not rebuilt (a resume needs the same file) and the run is done
    stamp = (out / 'prompts.txt').stat().st_mtime_ns
    (tmp_path / 'images' / '0.txt').write_text('something else', encoding='utf-8')
    config['max_steps'] = 4
    trainer = iris_turbo.run(copy.deepcopy(config))
    assert trainer.step == 4 and (out / 'prompts.txt').stat().st_mtime_ns == stamp
    assert iris_turbo.run(copy.deepcopy(config)) is None


def test_the_lora_tool_takes_the_same_config(tmp_path, monkeypatch):
    write_teacher(tmp_path / 'teacher')
    monkeypatch.setattr(lora, 'build_text_encoder', stand_in)
    path = tmp_path / 'turbo.toml'
    path.write_text(toml.dumps(lora_config(tmp_path)), encoding='utf-8')
    trainer = lora.main(['--config', str(path), '--set', 'max_steps=2', '--set', 'adapter.rank=2'])
    assert trainer.step == 2 and trainer.cfg.adapter.rank == 2


def test_full_run_goes_to_iris_trainer(tmp_path, monkeypatch):
    write_teacher(tmp_path / 'teacher')
    import iris3b.distill.trainer as trainer
    ran = []

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            ran.append(self.cfg)

    monkeypatch.setattr(trainer, 'DistillationTrainer', FakeTrainer)
    config = example('iris_3b_turbo_stage1.toml')
    config.update(output_dir=str(tmp_path / 'stage1'), dataset=str(write_dataset(tmp_path)))
    config['model']['transformer_path'] = str(tmp_path / 'teacher')
    config['turbo']['keep_last'] = 3
    before = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        iris_turbo.run(config)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)
    cfg, = ran
    stage1 = tmp_path / 'stage1'
    assert (cfg.teacher, cfg.work_dir, cfg.prompts_file) == (str(tmp_path / 'teacher'), str(stage1),
                                                             str(stage1 / 'prompts.txt'))
    assert (cfg.batch_size, cfg.max_steps, cfg.strategy, cfg.dmd.sigmas[1]) == (8, 1600, 'fsdp2', 0.926)
    assert cfg.resume_from == ''
    assert len((stage1 / 'prompts.txt').read_text(encoding='utf-8').split('\n')[:-1]) == 6
    assert OmegaConf.load(stage1 / 'turbo_distill.yaml').teacher == str(tmp_path / 'teacher')


def test_full_run_passes_the_launcher_flags(tmp_path, monkeypatch):
    from tools import iris_turbo_distill as distill
    calls = []
    monkeypatch.setattr(distill, 'main', calls.append)
    config = example('iris_3b_turbo_stage2.toml')
    config['output_dir'] = str(tmp_path / 'stage2')
    config['turbo'].update(prompts_file=str(tmp_path / 'p.txt'), keep_last=0, nccl_timeout_minutes=240)
    iris_turbo.run(config)
    argv, = calls
    assert argv == ['--config', str(tmp_path / 'stage2' / 'turbo_distill.yaml'), '--auto-resume',
                    '--keep-last', '0', '--nccl-timeout-minutes', '240']
    written = OmegaConf.load(tmp_path / 'stage2' / 'turbo_distill.yaml')
    assert (written.prompts_file, written.init_weights, written.ema_decay) == (str(tmp_path / 'p.txt'), 'ema', 0.9995)
    assert 'keep_last' not in written and 'nccl_timeout_minutes' not in written


def test_a_prompts_failure_stops_the_run(tmp_path):
    empty = tmp_path / 'empty.txt'
    empty.write_text('\n', encoding='utf-8')
    config = example('iris_3b_turbo_stage1.toml')
    config.pop('dataset')
    config['output_dir'] = str(tmp_path / 'stage1')
    config['turbo']['extra_prompts'] = [str(empty)]
    with pytest.raises(iris_turbo.TurboConfigError, match='No non-empty prompts'):
        iris_turbo.run(config)


def test_train_py_hands_turbo_configs_over(tmp_path):
    config = example('iris_3b_turbo_lora.toml')
    config['epochs'] = 10
    path = tmp_path / 'turbo.toml'
    path.write_text(toml.dumps(config), encoding='utf-8')
    result = subprocess.run([sys.executable, 'train.py', '--config', str(path)], cwd=ROOT, capture_output=True,
                            text=True, timeout=600, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    assert result.returncode != 0
    assert '[iris_turbo] invalid Turbo config' in result.stderr and 'epochs' in result.stderr, result.stderr[-3000:]


def test_two_processes_share_the_prompts(tmp_path):
    write_teacher(tmp_path / 'teacher')
    path = tmp_path / 'turbo.toml'
    path.write_text(toml.dumps({**lora_config(tmp_path), 'max_steps': 2}), encoding='utf-8')
    harness = ROOT / 'test' / 'test_iris_turbo_lora.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               str(harness), '--config', str(path), '--dump', str(tmp_path / 'dump')]
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, 'OMP_NUM_THREADS': '1'},
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    ranks = [torch.load(tmp_path / 'dump' / f'rank{r}.pt') for r in (0, 1)]
    assert ranks[0]['cursor'] == ranks[1]['cursor'] == 2 * 3 * 2
    for a, b in zip(ranks[0]['params'], ranks[1]['params']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len((tmp_path / 'out' / 'prompts.txt').read_text(encoding='utf-8').split('\n')[:-1]) == 6
