"""Iris-3B Turbo tooling: tools/iris_turbo_prompts.py and tools/iris_turbo_distill.py, on CPU.

The prompts file is read back with Iris's own PromptStream, and a diffusion-pipe full fine-tune
save goes through Iris's TDM objective as the teacher, student and fake network, so a format
mismatch with iris3b.distill fails here rather than hours into a GPU run.
"""

import os
import sys
from pathlib import Path

import pytest
import safetensors.torch
import toml
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'submodules' / 'iris-3b' / 'src'))

from test.test_iris import (  # noqa: E402
    ADAPTERS, make_pipeline, random_text, saved_adapter_state, single_process, write_checkpoint,  # noqa: F401
)
from tools import iris_turbo_distill as distill  # noqa: E402
from tools import iris_turbo_prompts as prompts_tool  # noqa: E402

IRIS_CONFIGS = ROOT / 'submodules' / 'iris-3b' / 'configs' / 'iris3b'


# ---- prompts ------------------------------------------------------------------------------------

def write_image(path):
    path.write_bytes(b'not decoded')


@pytest.fixture
def dataset(tmp_path):
    tags = tmp_path / 'tags'
    tags.mkdir()
    for name, caption in [('a', 'Special: red, blue, green'), ('b', 'A cat on a mat.\nSecond line.'),
                          ('c', ''), ('d', 'one two\x85three')]:
        write_image(tags / f'{name}.png')
        (tags / f'{name}.txt').write_text(caption, encoding='utf-8')
    write_image(tags / 'nocaption.png')
    json_dir = tmp_path / 'json'
    json_dir.mkdir()
    write_image(json_dir / 'x.png')
    (json_dir / 'captions.json').write_text('{"x.png": ["first caption", "second caption"]}', encoding='utf-8')
    config = {
        'resolutions': [64],
        'shuffle_tags': True,
        'directory': [
            {'path': str(tags), 'num_repeats': 2, 'caption_prefix': 'anime, ', 'prefix_tag_caption': 'Special:',
             'tag_dropout_rate': 0.9, 'cache_shuffle_num': 4, 'caption_dropout_rate': 0.5},
            {'path': str(json_dir)},
        ],
    }
    path = tmp_path / 'dataset.toml'
    path.write_text(toml.dumps(config), encoding='utf-8')
    return path


def run_prompts_tool(dataset, out, *flags):
    prompts_tool.main(['--dataset', str(dataset), '--output', str(out), '--no-progress', *flags])
    return out.read_text(encoding='utf-8').split('\n')[:-1]


def test_prompts_are_training_captions_without_augmentation(dataset, tmp_path):
    lines = run_prompts_tool(dataset, tmp_path / 'prompts.txt', '--no-shuffle')
    # caption_prefix on, marker off, no shuffle/tag dropout, num_repeats 2 for the first
    # directory, the empty caption and the image without one dropped, lines joined.
    once = ['anime, red, blue, green', 'anime, A cat on a mat. Second line.', 'anime, one two three']
    assert lines == once + once + ['first caption', 'second caption']
    assert not any('Special' in line for line in lines)


def test_prompts_file_reads_back_with_iris_prompt_stream(dataset, tmp_path):
    from iris3b.distill.trainer import PromptStream
    out = tmp_path / 'prompts.txt'
    lines = run_prompts_tool(dataset, out)
    assert list(PromptStream.from_file(out).prompts) == lines
    assert len(lines) == 8


def test_prompts_are_shuffled_reproducibly(dataset, tmp_path):
    first = run_prompts_tool(dataset, tmp_path / 'a.txt')
    again = run_prompts_tool(dataset, tmp_path / 'b.txt')
    other = run_prompts_tool(dataset, tmp_path / 'c.txt', '--seed', '3')
    ordered = run_prompts_tool(dataset, tmp_path / 'd.txt', '--no-shuffle')
    assert first == again
    assert sorted(first) == sorted(ordered) == sorted(other)
    assert first != ordered or other != ordered


def test_prompts_ignore_num_repeats_dedupe_and_extra_prompts(dataset, tmp_path):
    assert len(run_prompts_tool(dataset, tmp_path / 'a.txt', '--ignore-num-repeats')) == 5
    assert len(run_prompts_tool(dataset, tmp_path / 'b.txt', '--dedupe')) == 5
    extra = tmp_path / 'extra.txt'
    extra.write_text('﻿a general prompt\n\n  another one  \n', encoding='utf-8')
    lines = run_prompts_tool(dataset, tmp_path / 'c.txt', '--extra-prompts', str(extra), '--no-shuffle')
    assert lines[-2:] == ['a general prompt', 'another one']


def test_prompts_write_is_atomic(dataset, tmp_path, monkeypatch):
    out = tmp_path / 'prompts.txt'
    out.write_text('previous\n', encoding='utf-8')

    def fail(_):
        raise OSError('volume detached')
    monkeypatch.setattr(prompts_tool.os, 'fsync', fail)
    with pytest.raises(OSError):
        run_prompts_tool(dataset, out)
    assert out.read_text(encoding='utf-8') == 'previous\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(['prompts.txt', 'dataset.toml', 'tags', 'json'])


def test_prompts_tool_refuses_to_write_nothing(tmp_path):
    empty = tmp_path / 'empty.txt'
    empty.write_text('\n\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        prompts_tool.main(['--extra-prompts', str(empty), '--output', str(tmp_path / 'out.txt')])
    assert not (tmp_path / 'out.txt').exists()


# ---- teacher ------------------------------------------------------------------------------------

@pytest.fixture
def fine_tune(tmp_path):
    """A diffusion-pipe full fine-tune save of a tiny Iris."""
    write_checkpoint(tmp_path / 'base')
    pipe = make_pipeline(tmp_path / 'base', cache_text_embeddings=True)
    pipe.load_diffusion_model()
    layers = pipe.to_layers()
    with torch.no_grad():
        for p in pipe.transformer.parameters():
            p.add_(0.01)
    state_dict = {p.original_name: p.detach() for layer in layers for p in layer.parameters()}
    out = tmp_path / 'epoch1'
    out.mkdir()
    pipe.save_model(out, state_dict)
    return out


def test_a_diffusion_pipe_save_is_a_valid_teacher(fine_tune):
    assert distill.check_teacher(str(fine_tune)) is not None
    weights = safetensors.torch.load_file(str(fine_tune / 'model.safetensors'))
    assert distill.parameter_count(str(fine_tune)) == sum(t.numel() for t in weights.values())


def test_a_diffusion_pipe_save_trains_through_iris_tdm(fine_tune):
    """The trainer's own loading (load_for_inference + strict IrisDiT load) and one student and
    fake-network loss with backward, on the saved weights."""
    from copy import deepcopy

    from iris3b.config import inference_config
    from iris3b.distill.config import DMDConfig
    from iris3b.distill.objective import tdm_fake_loss, tdm_generator_loss
    from iris3b.distill.sampler import backward_simulate
    from iris3b.models.dit import IrisDiT
    from iris3b.sampling import load_for_inference

    raw, weights = load_for_inference(str(fine_tune))
    base = inference_config(raw)
    student = IrisDiT(base.model)
    student.load_state_dict(weights, strict=True)
    teacher = deepcopy(student).requires_grad_(False).eval()
    fake = deepcopy(student)
    y, y_mask = random_text(2, [5, 11])
    null_y, null_mask = random_text(2, [1, 1], seed=2)

    def fn(model, text, mask):
        return lambda x, t: model(x, t, text, y_mask=mask).x.float()

    cfg = DMDConfig(sigmas=[1.0, 0.75, 0.5], cfg_scale=3.5, sigma_min=0.01)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(2, base.model.in_channels, 32, 32, generator=g)
    generated, sigma, state = backward_simulate(fn(student, y, y_mask), noise, cfg.sigmas, step_index=1,
                                                num_train_timesteps=base.flow.num_train_timesteps)
    result = tdm_generator_loss(generated, state, sigma, 0.5, fn(teacher, y, y_mask), fn(teacher, null_y, null_mask),
                                fn(fake, y, y_mask), cfg, num_train_timesteps=base.flow.num_train_timesteps,
                                generator=g)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad
               and p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    fake_loss = tdm_fake_loss(generated.detach(), state, sigma, 0.5, fn(fake, y, y_mask), cfg,
                              num_train_timesteps=base.flow.num_train_timesteps, generator=g)
    fake_loss.backward()
    assert torch.isfinite(fake_loss)


def test_teacher_checks_fail_fast(fine_tune, tmp_path):
    with pytest.raises(distill.PreflightError, match='does not exist'):
        distill.check_teacher(str(tmp_path / 'missing'))
    (fine_tune / 'distill.yaml').write_text('dmd: {sigmas: [1.0, 0.5]}\n')
    with pytest.raises(distill.PreflightError, match='Turbo'):
        distill.check_teacher(str(fine_tune))
    (fine_tune / 'distill.yaml').unlink()
    config = OmegaConf.load(fine_tune / 'config.yaml')
    config.model.attn_backend = 'fa3'
    OmegaConf.save(config, fine_tune / 'config.yaml')
    with pytest.raises(distill.PreflightError, match='attn_backend'):
        distill.check_teacher(str(fine_tune))
    (fine_tune / 'config.yaml').unlink()
    with pytest.raises(distill.PreflightError, match='config.yaml'):
        distill.check_teacher(str(fine_tune))


def test_training_on_a_turbo_checkpoint_warns(tmp_path, capsys):
    write_checkpoint(tmp_path / 'turbo')
    (tmp_path / 'turbo' / 'distill.yaml').write_text('dmd: {sigmas: [1.0, 0.5]}\n')
    make_pipeline(tmp_path / 'turbo', cache_text_embeddings=True)
    assert 'Turbo (distilled) checkpoint' in capsys.readouterr().out
    write_checkpoint(tmp_path / 'base')
    make_pipeline(tmp_path / 'base', cache_text_embeddings=True)
    assert 'Turbo' not in capsys.readouterr().out


def test_merging_onto_a_turbo_base_keeps_its_schedule(tmp_path):
    from tools import iris_merge_adapter
    write_checkpoint(tmp_path / 'turbo')
    (tmp_path / 'turbo' / 'distill.yaml').write_text('dmd: {sigmas: [1.0, 0.5]}\n')
    pipe = make_pipeline(tmp_path / 'turbo', cache_text_embeddings=True)
    pipe.load_diffusion_model()
    pipe.configure_adapter(dict(ADAPTERS['lora']))
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    pipe.save_adapter(adapter, saved_adapter_state(pipe))
    iris_merge_adapter.main(['--model', str(tmp_path / 'turbo'), '--adapter', str(adapter),
                             '--output', str(tmp_path / 'merged')])
    assert (tmp_path / 'merged' / 'distill.yaml').read_text() == 'dmd: {sigmas: [1.0, 0.5]}\n'


# ---- work dir and checkpoints ---------------------------------------------------------------------

def test_work_dir_check_passes_and_cleans_up(tmp_path):
    checkpoints = distill.check_work_dir(tmp_path / 'run')
    assert checkpoints == tmp_path / 'run' / 'checkpoints'
    assert list(checkpoints.iterdir()) == []


def test_work_dir_without_symlinks_is_refused(tmp_path, monkeypatch):
    def no_symlinks(self, target):
        raise OSError(95, 'Operation not supported')
    monkeypatch.setattr(Path, 'symlink_to', no_symlinks)
    with pytest.raises(distill.PreflightError, match='symlink'):
        distill.check_work_dir(tmp_path / 'run')
    assert list((tmp_path / 'run' / 'checkpoints').iterdir()) == []


def make_checkpoints(directory, steps, latest):
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (directory / f'step_{step:08d}.pth').write_bytes(b'x')
    (directory / 'latest.pth').symlink_to((directory / f'step_{latest:08d}.pth').resolve())


def test_pruning_keeps_the_newest_and_the_latest_target(tmp_path):
    directory = tmp_path / 'checkpoints'
    make_checkpoints(directory, [100, 200, 300, 400, 500], latest=300)
    removed = distill.prune_checkpoints(directory, keep_last=2)
    assert sorted(p.name for p in removed) == ['step_00000100.pth', 'step_00000200.pth']
    assert sorted(p.name for p in directory.iterdir()) == \
        ['latest.pth', 'step_00000300.pth', 'step_00000400.pth', 'step_00000500.pth']
    assert distill.prune_checkpoints(directory, keep_last=0) == []


def test_stale_temporaries_are_removed(tmp_path):
    directory = tmp_path / 'checkpoints'
    make_checkpoints(directory, [100], latest=100)
    (directory / 'step_00000200.pth.tmp').write_bytes(b'partial')
    (directory / 'latest.pth.tmp').symlink_to(directory / 'step_00000100.pth')
    removed = distill.remove_stale_temporaries(directory)
    assert sorted(p.name for p in removed) == ['latest.pth.tmp', 'step_00000200.pth.tmp']
    assert sorted(p.name for p in directory.iterdir()) == ['latest.pth', 'step_00000100.pth']


def test_pruning_hooks_into_iris_checkpoint_saves(tmp_path, monkeypatch):
    from iris3b.train import ckpt
    monkeypatch.setattr(ckpt, 'save_checkpoint', ckpt.save_checkpoint)
    distill.install_pruning(keep_last=2)
    directory = tmp_path / 'checkpoints'
    model = torch.nn.Linear(2, 2)
    for step in (100, 200, 300):
        # how DistillationState.save calls it
        from iris3b.train.ckpt import save_checkpoint
        save_checkpoint(directory / f'step_{step:08d}.pth', model.state_dict(), epoch=0, step=step)
    assert sorted(p.name for p in directory.iterdir()) == ['latest.pth', 'step_00000200.pth', 'step_00000300.pth']
    assert (directory / 'latest.pth').resolve() == (directory / 'step_00000300.pth').resolve()
    assert torch.load(directory / 'latest.pth', weights_only=False)['step'] == 300


@pytest.mark.parametrize('max_steps, save_every, keep_last, expected', [
    (1600, 100, 2, 3), (1600, 100, 0, 16), (500, 100, 0, 5), (450, 100, 0, 5), (100, 100, 5, 1),
])
def test_checkpoints_on_disk(max_steps, save_every, keep_last, expected):
    cfg = distill.load_config(IRIS_CONFIGS / 'turbo_stage1_6step.yaml',
                              ['teacher=t', 'prompts_file=p', f'max_steps={max_steps}', f'save_every={save_every}'])
    assert distill.checkpoints_on_disk(cfg, keep_last) == expected


# ---- launcher -------------------------------------------------------------------------------------

def test_launcher_turns_cudnn_off_and_runs_iris_trainer(fine_tune, tmp_path, monkeypatch):
    import iris3b.distill.trainer as trainer
    from iris3b.train import ckpt
    monkeypatch.setattr(ckpt, 'save_checkpoint', ckpt.save_checkpoint)
    ran = []

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            ran.append((self.cfg, torch.backends.cuda.cudnn_sdp_enabled()))

    monkeypatch.setattr(trainer, 'DistillationTrainer', FakeTrainer)
    prompts = tmp_path / 'prompts.txt'
    prompts.write_text('a prompt\n', encoding='utf-8')
    before = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        distill.main(['--config', str(IRIS_CONFIGS / 'turbo_stage1_6step.yaml'), f'teacher={fine_tune}',
                      f'prompts_file={prompts}', f'work_dir={tmp_path / "run"}', 'strategy=fsdp2'])
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)
    (cfg, cudnn), = ran
    assert cudnn is False
    assert (cfg.teacher, cfg.strategy, cfg.max_steps) == (str(fine_tune), 'fsdp2', 1600)
    assert (tmp_path / 'run' / 'checkpoints').is_dir()


def test_launcher_stops_before_training_on_a_bad_setup(fine_tune, tmp_path, monkeypatch):
    import iris3b.distill.trainer as trainer
    monkeypatch.setattr(trainer, 'DistillationTrainer', lambda cfg: pytest.fail('trainer must not start'))
    config = str(IRIS_CONFIGS / 'turbo_stage1_6step.yaml')
    with pytest.raises(SystemExit, match='prompts_file'):
        distill.main(['--config', config, f'teacher={fine_tune}', f'prompts_file={tmp_path / "none.txt"}',
                      f'work_dir={tmp_path / "run"}', '--allow-cudnn-attention'])
    prompts = tmp_path / 'prompts.txt'
    prompts.write_text('a prompt\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='init_from'):
        distill.main(['--config', str(IRIS_CONFIGS / 'turbo_stage2_4step.yaml'), f'teacher={fine_tune}',
                      f'prompts_file={prompts}', f'work_dir={tmp_path / "run"}', '--allow-cudnn-attention'])


def test_upstream_configs_load():
    for name in ('turbo_stage1_6step.yaml', 'turbo_stage2_4step.yaml'):
        cfg = distill.load_config(IRIS_CONFIGS / name, ['teacher=t', 'prompts_file=p'])
        assert cfg.betas == [0.0, 0.999]
    assert os.path.isfile(ROOT / 'submodules' / 'iris-3b' / 'scripts' / 'export_distilled.py')
