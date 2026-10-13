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


def test_prompts_drop_captions_that_are_not_valid_unicode():
    prompts, report = prompts_tool.build_prompts(['fine', 'bad \ud800 one'], shuffle=False)
    assert prompts == ['fine'] and report['unencodable'] == 1


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
    raw, params = distill.check_teacher(str(fine_tune))
    weights = safetensors.torch.load_file(str(fine_tune / 'model.safetensors'))
    assert params == sum(t.numel() for t in weights.values())
    assert raw['model']['attn_backend'] == 'sdpa'


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
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    fake_loss = tdm_fake_loss(generated.detach(), state, sigma, 0.5, fn(fake, y, y_mask), cfg,
                              num_train_timesteps=base.flow.num_train_timesteps, generator=g)
    fake_loss.backward()
    assert torch.isfinite(fake_loss)


def set_model_config(directory, **values):
    config = OmegaConf.load(directory / 'config.yaml')
    for key, value in values.items():
        config.model[key] = value
    OmegaConf.save(config, directory / 'config.yaml')


def test_teacher_checks_fail_fast(fine_tune, tmp_path):
    with pytest.raises(distill.PreflightError, match='does not exist'):
        distill.check_teacher(str(tmp_path / 'missing'))
    (fine_tune / 'distill.yaml').write_text('dmd: {sigmas: [1.0, 0.5]}\n')
    with pytest.raises(distill.PreflightError, match='Turbo'):
        distill.check_teacher(str(fine_tune))
    (fine_tune / 'distill.yaml').unlink()
    set_model_config(fine_tune, attn_backend='fa3')
    with pytest.raises(distill.PreflightError, match='attn_backend'):
        distill.check_teacher(str(fine_tune))
    (fine_tune / 'config.yaml').unlink()
    with pytest.raises(distill.PreflightError, match='config.yaml'):
        distill.check_teacher(str(fine_tune))


def test_teacher_config_edge_cases():
    distill.check_inference_config({'model': None}, 'teacher')
    distill.check_inference_config({'model': {'attn_backend': 'fa3', 'text_adapter': 'linear'}}, 'teacher')
    with pytest.raises(distill.PreflightError, match='task'):
        distill.check_inference_config({'task': {'name': 'depth'}}, 'teacher')
    with pytest.raises(distill.PreflightError, match='no Iris config'):
        distill.check_inference_config(None, 'teacher')


def test_pth_teachers(fine_tune, tmp_path):
    raw = OmegaConf.to_container(OmegaConf.load(fine_tune / 'config.yaml'))
    weights = safetensors.torch.load_file(str(fine_tune / 'model.safetensors'))
    training = tmp_path / 'training.pth'
    torch.save({'config': raw, 'state_dict': weights, 'state_dict_ema': weights}, training)
    _, params = distill.check_teacher(str(training))
    assert params == sum(t.numel() for t in weights.values())
    student = tmp_path / 'student.pth'
    torch.save({'config': raw, 'state_dict_ema': weights, 'distill_format_version': 1}, student)
    with pytest.raises(distill.PreflightError, match='Turbo student'):
        distill.check_teacher(str(student))
    torch.save({'state_dict': weights}, training)
    with pytest.raises(distill.PreflightError, match='no Iris config'):
        distill.check_teacher(str(training))


def test_training_on_a_turbo_checkpoint_warns(tmp_path, capsys):
    write_checkpoint(tmp_path / 'turbo')
    (tmp_path / 'turbo' / 'distill.yaml').write_text('dmd: {sigmas: [1.0, 0.5]}\n')
    make_pipeline(tmp_path / 'turbo', cache_text_embeddings=True)
    assert 'Turbo (distilled) checkpoint' in capsys.readouterr().out
    write_checkpoint(tmp_path / 'base')
    make_pipeline(tmp_path / 'base', cache_text_embeddings=True)
    assert 'Turbo' not in capsys.readouterr().out


def test_merge_keeps_a_turbo_schedule_and_drops_a_stale_one(tmp_path):
    from tools import iris_merge_adapter
    for name in ('turbo', 'base'):
        write_checkpoint(tmp_path / name)
    (tmp_path / 'turbo' / 'distill.yaml').write_text('dmd: {sigmas: [1.0, 0.5]}\n')
    pipe = make_pipeline(tmp_path / 'turbo', cache_text_embeddings=True)
    pipe.load_diffusion_model()
    pipe.configure_adapter(dict(ADAPTERS['lora']))
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    pipe.save_adapter(adapter, saved_adapter_state(pipe))
    out = tmp_path / 'merged'
    iris_merge_adapter.main(['--model', str(tmp_path / 'turbo'), '--adapter', str(adapter), '--output', str(out)])
    assert (out / 'distill.yaml').read_text() == 'dmd: {sigmas: [1.0, 0.5]}\n'
    # the same output directory, now merged onto a base without a schedule
    iris_merge_adapter.main(['--model', str(tmp_path / 'base'), '--adapter', str(adapter), '--output', str(out)])
    assert not (out / 'distill.yaml').exists()


# ---- prompts file and distillation checkpoints ------------------------------------------------

def test_prompts_file_checks(tmp_path):
    path = tmp_path / 'prompts.txt'
    with pytest.raises(distill.PreflightError, match='does not exist'):
        distill.check_prompts(path)
    path.write_bytes(b'\n  \n')
    with pytest.raises(distill.PreflightError, match='no prompts'):
        distill.check_prompts(path)
    path.write_bytes(b'caf\xe9\n')
    with pytest.raises(distill.PreflightError, match='UTF-8'):
        distill.check_prompts(path)
    path.write_bytes(b'a prompt\n')
    from iris3b.distill.trainer import PromptStream
    assert distill.check_prompts(path) == PromptStream.from_file(path).sha256


def write_distill_checkpoint(path, step, prompts_sha='sha', world_size=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'distill_format_version': 1, 'state_dict': {}, 'state_dict_ema': {}, 'state_dict_fake': {},
                'generator_updates': step, 'world_size': world_size,
                'prompt_state': {'sha256': prompts_sha, 'count': 1, 'cursor': 0}}, path)
    latest = path.parent / 'latest.pth'
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.resolve())


def stage_config(tmp_path, *overrides):
    return distill.load_config(IRIS_CONFIGS / 'turbo_stage1_6step.yaml',
                               ['teacher=t', 'prompts_file=p', f'work_dir={tmp_path / "run"}', *overrides])


def test_a_fresh_run_does_not_overwrite_checkpoints(tmp_path):
    checkpoints = tmp_path / 'run' / 'checkpoints'
    cfg = stage_config(tmp_path)
    assert distill.resolve_run(cfg, checkpoints, 'sha', auto_resume=False) == ''
    write_distill_checkpoint(checkpoints / 'step_00000100.pth', 100)
    write_distill_checkpoint(checkpoints / 'step_00000200.pth', 200)
    with pytest.raises(distill.PreflightError, match='start over'):
        distill.resolve_run(cfg, checkpoints, 'sha', auto_resume=False)
    assert distill.resolve_run(cfg, checkpoints, 'sha', auto_resume=True) == \
        str((checkpoints / 'step_00000200.pth').resolve())
    # latest.pth dangling (the volume mounted elsewhere): still the newest step file
    (checkpoints / 'latest.pth').unlink()
    (checkpoints / 'latest.pth').symlink_to('/elsewhere/step_00000200.pth')
    assert distill.resolve_run(cfg, checkpoints, 'sha', auto_resume=True) == \
        str((checkpoints / 'step_00000200.pth').resolve())
    with pytest.raises(distill.PreflightError, match='mounted at another path'):
        distill.resolve_run(stage_config(tmp_path, f'resume_from={checkpoints / "latest.pth"}'),
                            checkpoints, 'sha', auto_resume=False)


def test_auto_resume_skips_a_torn_checkpoint_and_a_lagging_latest(tmp_path):
    checkpoints = tmp_path / 'run' / 'checkpoints'
    cfg = stage_config(tmp_path)
    for step in (100, 200, 300):
        write_distill_checkpoint(checkpoints / f'step_{step:08d}.pth', step)
    # latest.pth one save behind (the pod died between Iris's rename and relink)
    (checkpoints / 'latest.pth').unlink()
    (checkpoints / 'latest.pth').symlink_to((checkpoints / 'step_00000200.pth').resolve())
    assert distill.resolve_run(cfg, checkpoints, 'sha', True) == str((checkpoints / 'step_00000300.pth').resolve())
    # the newest one zero-filled by a node crash
    size = (checkpoints / 'step_00000300.pth').stat().st_size
    (checkpoints / 'step_00000300.pth').write_bytes(bytes(size))
    assert distill.resolve_run(cfg, checkpoints, 'sha', True) == str((checkpoints / 'step_00000200.pth').resolve())
    assert (checkpoints / 'step_00000300.pth.torn').exists() and not (checkpoints / 'step_00000300.pth').exists()
    # an explicit resume_from of a torn file is a clear refusal, not a traceback
    (checkpoints / 'step_00000200.pth').write_bytes(b'cut')
    with pytest.raises(distill.PreflightError, match='cannot read'):
        distill.resolve_run(stage_config(tmp_path, f'resume_from={checkpoints / "step_00000200.pth"}'),
                            checkpoints, 'sha', False)


def test_resume_must_match_gpus_prompts_and_be_the_newest(tmp_path, monkeypatch):
    checkpoints = tmp_path / 'run' / 'checkpoints'
    write_distill_checkpoint(checkpoints / 'step_00000100.pth', 100)
    write_distill_checkpoint(checkpoints / 'step_00000200.pth', 200)
    resume = stage_config(tmp_path, f'resume_from={checkpoints / "step_00000200.pth"}')
    assert distill.resolve_run(resume, checkpoints, 'sha', auto_resume=False) == str(checkpoints / 'step_00000200.pth')
    with pytest.raises(distill.PreflightError, match='another prompts file'):
        distill.resolve_run(resume, checkpoints, 'other', auto_resume=False)
    monkeypatch.setenv('WORLD_SIZE', '8')
    with pytest.raises(distill.PreflightError, match='same GPU count'):
        distill.resolve_run(resume, checkpoints, 'sha', auto_resume=False)
    monkeypatch.delenv('WORLD_SIZE')
    older = stage_config(tmp_path, f'resume_from={checkpoints / "step_00000100.pth"}')
    with pytest.raises(distill.PreflightError, match='step_00000200.pth, newer'):
        distill.resolve_run(older, checkpoints, 'sha', auto_resume=False)


def test_init_from_must_be_a_distillation_checkpoint(tmp_path, fine_tune):
    good = tmp_path / 'stage1' / 'step_00001600.pth'
    write_distill_checkpoint(good, 1600)
    checkpoints = tmp_path / 'run' / 'checkpoints'
    assert distill.resolve_run(stage_config(tmp_path, f'init_from={good}'), checkpoints, 'sha', False) == ''
    with pytest.raises(distill.PreflightError, match='is not a file'):
        distill.resolve_run(stage_config(tmp_path, f'init_from={fine_tune}'), checkpoints, 'sha', False)
    plain = tmp_path / 'plain.pth'
    torch.save({'state_dict': {}}, plain)
    with pytest.raises(distill.PreflightError, match='not written by Iris distillation'):
        distill.resolve_run(stage_config(tmp_path, f'init_from={plain}'), checkpoints, 'sha', False)
    # Iris ignores init_from when resuming, so a missing one does not matter then
    write_distill_checkpoint(checkpoints / 'step_00000100.pth', 100)
    resume = stage_config(tmp_path, f'init_from={tmp_path / "gone.pth"}', f'resume_from={checkpoints / "latest.pth"}')
    assert distill.resolve_run(resume, checkpoints, 'sha', False)


# ---- storage, memory ------------------------------------------------------------------------------

def test_work_dir_check_passes_and_cleans_up(tmp_path):
    checkpoints = distill.check_work_dir(tmp_path / 'run')
    assert checkpoints == tmp_path / 'run' / 'checkpoints'
    assert list(checkpoints.iterdir()) == []
    assert distill.write_speed(checkpoints, size=1 << 20) > 0
    assert list(checkpoints.iterdir()) == []


def test_work_dir_without_symlinks_is_refused(tmp_path, monkeypatch):
    def no_symlinks(self, target):
        raise OSError(95, 'Operation not supported')
    monkeypatch.setattr(Path, 'symlink_to', no_symlinks)
    with pytest.raises(distill.PreflightError, match='symlink'):
        distill.check_work_dir(tmp_path / 'run')
    assert list((tmp_path / 'run' / 'checkpoints').iterdir()) == []


def test_work_dir_that_cannot_rename_over_a_file_is_refused(tmp_path, monkeypatch):
    original = Path.replace

    def no_overwrite(self, target):
        if Path(target).exists():
            raise OSError(17, 'File exists')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', no_overwrite)
    with pytest.raises(distill.PreflightError, match='rename over an existing entry'):
        distill.check_work_dir(tmp_path / 'run')


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


def test_stale_temporaries_and_broken_null_cache_are_removed(tmp_path):
    directory = tmp_path / 'checkpoints'
    make_checkpoints(directory, [100], latest=100)
    (directory / 'step_00000200.pth.tmp').write_bytes(b'partial')
    (directory / 'latest.pth.tmp').symlink_to(directory / 'step_00000100.pth')
    removed = distill.remove_stale_temporaries(directory)
    assert sorted(p.name for p in removed) == ['latest.pth.tmp', 'step_00000200.pth.tmp']
    assert sorted(p.name for p in directory.iterdir()) == ['latest.pth', 'step_00000100.pth']
    cache = tmp_path / 'null'
    cache.mkdir()
    torch.save({'embeddings': torch.zeros(1, 2), 'mask': torch.ones(1, 2)}, cache / 'null_embed_good.pth')
    (cache / 'null_embed_torn.pth').write_bytes(b'PK\x03\x04 cut off')
    assert [p.name for p in distill.remove_broken_null_cache(cache)] == ['null_embed_torn.pth']
    assert [p.name for p in cache.iterdir()] == ['null_embed_good.pth']


def fake_distillation_state(directory):
    """A DistillationState holding a tiny model, enough for its real save()."""
    import iris3b.distill.trainer as trainer
    from iris3b.distill.config import DistillConfig
    state = object.__new__(trainer.DistillationState)
    state.cfg = DistillConfig(teacher='t', prompts_file='p', fake_updates=1, batch_size=1)
    state.rank, state.world_size, state.sharded = 0, 1, False
    state.generator_updates = state.fake_updates = 0
    state.prompts = trainer.PromptStream(('a',), 'sha', 0)
    model = torch.nn.Linear(2, 2)
    state.student = state.fake = model
    state.ema = trainer.EMA(model, 0.9)
    state.optimizer = state.fake_optimizer = torch.optim.AdamW(model.parameters())
    state.device, state.generator = torch.device('cpu'), torch.Generator()
    state.base_config, state.teacher_identity = {}, {}
    return state


def test_pruning_hooks_into_iris_distillation_saves(tmp_path, monkeypatch):
    import iris3b.distill.trainer as trainer
    from iris3b.train import ckpt
    # The hook relies on DistillationState.save importing save_checkpoint at each call.
    assert not hasattr(trainer, 'save_checkpoint')
    monkeypatch.setattr(ckpt, 'save_checkpoint', ckpt.save_checkpoint)
    monkeypatch.setattr(trainer, '_capture_rng', lambda device, generator: {})
    distill.install_pruning(keep_last=2)
    state = fake_distillation_state(tmp_path)
    directory = tmp_path / 'checkpoints'
    for step in (100, 200, 300):
        state.generator_updates = state.fake_updates = step
        state.prompts.cursor = 2 * step
        state.save(directory / f'step_{step:08d}.pth')
    assert sorted(p.name for p in directory.iterdir()) == ['latest.pth', 'step_00000200.pth', 'step_00000300.pth']
    assert torch.load(directory / 'latest.pth', weights_only=False)['generator_updates'] == 300

    # a failed delete is reported, and the save that just succeeded stands
    original = Path.unlink

    def refuse(self, missing_ok=False):
        if self.name.startswith('step_'):
            raise PermissionError(13, 'Permission denied', str(self))
        return original(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, 'unlink', refuse)
    state.generator_updates = state.fake_updates = 400
    state.prompts.cursor = 800
    state.save(directory / 'step_00000400.pth')
    assert (directory / 'latest.pth').resolve() == (directory / 'step_00000400.pth').resolve()


@pytest.mark.parametrize('max_steps, save_every, keep_last, expected', [
    (1600, 100, 2, 3), (1600, 100, 0, 16), (500, 100, 0, 5), (450, 100, 0, 5), (100, 100, 5, 1),
])
def test_checkpoints_on_disk(max_steps, save_every, keep_last, expected, tmp_path):
    cfg = stage_config(tmp_path, f'max_steps={max_steps}', f'save_every={save_every}')
    assert distill.checkpoints_on_disk(cfg, keep_last) == expected


def test_memory_warnings(tmp_path, capsys, monkeypatch):
    fsdp2 = stage_config(tmp_path, 'strategy=fsdp2')
    distill.check_host_memory(fsdp2, checkpoint_size=84e9, limit=80e9)
    assert 'gathers each checkpoint' in capsys.readouterr().out
    distill.check_host_memory(fsdp2, checkpoint_size=84e9, limit=512e9)
    distill.check_host_memory(stage_config(tmp_path), checkpoint_size=84e9, limit=40e9)
    assert capsys.readouterr().out == ''
    def full(*args, **kwargs):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(distill, 'write_speed', full)
    distill.check_write_speed(tmp_path, 84e9, 120)
    assert 'No space left' in capsys.readouterr().out


# ---- launcher -------------------------------------------------------------------------------------

def run_launcher(monkeypatch, argv):
    import iris3b.distill.trainer as trainer
    from iris3b.train import ckpt
    monkeypatch.setattr(ckpt, 'save_checkpoint', ckpt.save_checkpoint)
    ran = []

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            ran.append((self.cfg, torch.backends.cuda.cudnn_sdp_enabled(), ckpt.save_checkpoint))

    monkeypatch.setattr(trainer, 'DistillationTrainer', FakeTrainer)
    before = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        distill.main(argv)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)
    return ran


def test_launcher_turns_cudnn_off_and_runs_iris_trainer(fine_tune, tmp_path, monkeypatch):
    from iris3b.train import ckpt
    original = ckpt.save_checkpoint
    prompts = tmp_path / 'prompts.txt'
    prompts.write_text('a prompt\n', encoding='utf-8')
    # overrides may come after the launcher's own options
    (cfg, cudnn, save), = run_launcher(monkeypatch, [
        '--config', str(IRIS_CONFIGS / 'turbo_stage1_6step.yaml'), f'teacher={fine_tune}', '--keep-last', '3',
        f'prompts_file={prompts}', f'work_dir={tmp_path / "run"}', 'strategy=fsdp2'])
    assert cudnn is False
    assert save is not original
    assert (cfg.teacher, cfg.prompts_file, cfg.strategy, cfg.resume_from) == (str(fine_tune), str(prompts), 'fsdp2', '')
    assert (tmp_path / 'run' / 'checkpoints').is_dir()


def test_launcher_auto_resumes(fine_tune, tmp_path, monkeypatch):
    prompts = tmp_path / 'prompts.txt'
    prompts.write_text('a prompt\n', encoding='utf-8')
    from iris3b.distill.trainer import PromptStream
    checkpoints = tmp_path / 'run' / 'checkpoints'
    write_distill_checkpoint(checkpoints / 'step_00000100.pth', 100, PromptStream.from_file(prompts).sha256)
    argv = ['--config', str(IRIS_CONFIGS / 'turbo_stage1_6step.yaml'), f'teacher={fine_tune}',
            f'prompts_file={prompts}', f'work_dir={tmp_path / "run"}', '--skip-write-test']
    with pytest.raises(SystemExit, match='start over'):
        run_launcher(monkeypatch, argv)
    (cfg, _, _), = run_launcher(monkeypatch, [*argv, '--auto-resume'])
    assert cfg.resume_from == str((checkpoints / 'step_00000100.pth').resolve())


def test_launcher_stops_before_training_on_a_bad_setup(fine_tune, tmp_path, monkeypatch):
    config = str(IRIS_CONFIGS / 'turbo_stage1_6step.yaml')
    with pytest.raises(SystemExit, match='prompts_file'):
        run_launcher(monkeypatch, ['--config', config, f'teacher={fine_tune}', f'prompts_file={tmp_path / "none.txt"}',
                                   f'work_dir={tmp_path / "run"}'])
    prompts = tmp_path / 'prompts.txt'
    prompts.write_text('a prompt\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='init_from'):
        run_launcher(monkeypatch, ['--config', str(IRIS_CONFIGS / 'turbo_stage2_4step.yaml'), f'teacher={fine_tune}',
                                   f'prompts_file={prompts}', f'work_dir={tmp_path / "run"}'])


def test_upstream_configs_load(tmp_path):
    for name in ('turbo_stage1_6step.yaml', 'turbo_stage2_4step.yaml'):
        cfg = distill.load_config(IRIS_CONFIGS / name, ['teacher=t', 'prompts_file=p'])
        assert cfg.betas == [0.0, 0.999]
    assert os.path.isfile(ROOT / 'submodules' / 'iris-3b' / 'scripts' / 'export_distilled.py')


def test_launcher_exits_cleanly_for_a_finished_run_and_unexpected_errors(fine_tune, tmp_path, monkeypatch, capsys):
    prompts = tmp_path / 'prompts.txt'
    prompts.write_text('a prompt\n', encoding='utf-8')
    from iris3b.distill.trainer import PromptStream
    checkpoints = tmp_path / 'run' / 'checkpoints'
    write_distill_checkpoint(checkpoints / 'step_00001600.pth', 1600, PromptStream.from_file(prompts).sha256)
    argv = ['--config', str(IRIS_CONFIGS / 'turbo_stage1_6step.yaml'), f'teacher={fine_tune}',
            f'prompts_file={prompts}', f'work_dir={tmp_path / "run"}', '--skip-write-test', '--auto-resume']
    assert run_launcher(monkeypatch, argv) == []
    assert 'already reached max_steps=1600' in capsys.readouterr().out
    (cfg, _, _), = run_launcher(monkeypatch, [*argv, 'max_steps=2000'])
    assert cfg.resume_from.endswith('step_00001600.pth')

    def broken(*args, **kwargs):
        raise KeyError('model')
    monkeypatch.setattr(distill, 'check_prompts', broken)
    with pytest.raises(SystemExit, match="preflight failed: KeyError: 'model'"):
        run_launcher(monkeypatch, argv)
    with pytest.raises(SystemExit):
        run_launcher(monkeypatch, [*argv, '--skip-preflight'])
