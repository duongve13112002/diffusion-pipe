"""Iris-3B Turbo from a diffusion-pipe config: train.py hands any config with a [turbo] table here.

    deepspeed --num_gpus=8 train.py --deepspeed --config examples/iris/turbo/iris_3b_turbo_lora.toml
    deepspeed --num_gpus=8 train.py --deepspeed --config examples/iris/turbo/iris_3b_turbo_stage1.toml

Turbo is distillation, not diffusion-pipe's training loop: the student learns from the teacher's
own few-step samples of text prompts, not from images, and it needs a fake score network and an
EMA beside it. So the config keeps diffusion-pipe's layout and names, and these trainers run it:

    [adapter] present -> a Turbo LoRA (tools/iris_turbo_lora.py): student, fake network and EMA
                         are LoRA adapters on one frozen copy of the model
    no [adapter]      -> full Turbo (tools/iris_turbo_distill.py, Iris's own iris3b.distill
                         trainer): the whole model is distilled, as Iris-3B Turbo was

Key mapping (docs/iris_turbo.md lists every key):

    output_dir                  -> the run directory (Iris's work_dir for full Turbo)
    dataset                     -> the prompts: the captions of this dataset.toml, written once
                                   to output_dir/prompts.txt (tools/iris_turbo_prompts.py)
    micro_batch_size_per_gpu    -> batch_size
    gradient_accumulation_steps -> LoRA only (Iris's full trainer has none)
    gradient_clipping           -> max_grad_norm
    activation_checkpointing    -> true = 'full', false = 'none', or a policy name
    max_steps                   -> student updates; save_every_n_steps -> save_every;
                                   steps_per_print -> log_every; save_dtype -> the adapter files
    [model] transformer_path    -> the teacher; [model] dtype -> the frozen base's dtype (LoRA)
    [adapter] rank / alpha / dropout / target_modules, plus fake_rank / fake_alpha
    [optimizer] lr / betas / weight_decay, plus fake_lr for the fake network
    [turbo]                     -> the distillation itself (sigmas, cfg_scale, fake_updates, ...)

Both trainers resume by themselves: the same command after an interruption continues from the
newest checkpoint in output_dir, and a finished run exits at once.
"""

import os
from datetime import timedelta
from pathlib import Path

import toml
import torch


class TurboConfigError(ValueError):
    pass


# Top-level diffusion-pipe keys and what they become.
TOP_LEVEL = {
    'output_dir': 'output_dir',
    'micro_batch_size_per_gpu': 'batch_size',
    'gradient_accumulation_steps': 'gradient_accumulation_steps',
    'gradient_clipping': 'max_grad_norm',
    'activation_checkpointing': 'activation_checkpointing',
    'save_dtype': 'save_dtype',
    'max_steps': 'max_steps',
    'save_every_n_steps': 'save_every',
    'steps_per_print': 'log_every',
}
# Read here, not passed on.
TOP_LEVEL_OWN = {'dataset', 'pipeline_stages', 'model', 'adapter', 'optimizer', 'turbo', 'monitoring'}
LORA_ONLY_TOP = {'gradient_accumulation_steps', 'save_dtype'}

TURBO_COMMON = {'sigmas', 'cfg_scale', 'sigma_min', 'resolutions', 'fake_updates', 'ema_decay', 'seed',
                'init_from', 'keep_last', 'nccl_timeout_minutes'}
TURBO_PROMPTS = {'prompts_file', 'extra_prompts'}
TURBO_LORA = {'text_encoder', 'sample_every', 'sample_prompts', 'sample_size', 'sample_seed'}
TURBO_FULL = {'strategy', 'init_weights', 'height', 'width'}
DMD_KEYS = {'sigmas', 'cfg_scale', 'sigma_min'}

ADAPTER_KEYS = {'rank': 'rank', 'alpha': 'alpha', 'dropout': 'dropout', 'fake_rank': 'fake_rank',
                'fake_alpha': 'fake_alpha', 'target_modules': 'target_modules'}
OPTIMIZER_KEYS = {'lr': 'learning_rate', 'fake_lr': 'fake_learning_rate', 'betas': 'betas',
                  'weight_decay': 'weight_decay'}

# What tools/iris_turbo_distill.py's own flags default to; [turbo] sets them for full Turbo.
FULL_KEEP_LAST, FULL_NCCL_TIMEOUT = 2, 120
LORA_NCCL_TIMEOUT = 60


def is_turbo(config):
    return 'turbo' in config


def mode(config):
    return 'lora' if 'adapter' in config else 'full'


def _unknown(table, allowed, where):
    extra = sorted(set(table) - set(allowed))
    return [f'{where}{k}' for k in extra]


def _checkpointing(value):
    if value is True:
        return 'full'
    if value is False:
        return 'none'
    return value


def translate(config):
    """(mode, settings, notes): the trainer's settings for a diffusion-pipe Turbo config.

    LoRA: a dict for tools/iris_turbo_lora.config_from_dict. Full: Iris's DistillConfig fields as
    a nested dict, plus 'keep_last' and 'nccl_timeout_minutes' for the launcher. `notes` lists
    keys accepted but not used, for the log.
    """
    which = mode(config)
    turbo = dict(config.get('turbo') or {})
    model = dict(config.get('model') or {})
    adapter = dict(config.get('adapter') or {}) if which == 'lora' else {}
    optimizer = dict(config.get('optimizer') or {})
    problems, notes = [], []

    allowed_top = set(TOP_LEVEL) | TOP_LEVEL_OWN
    if which == 'full':
        allowed_top -= LORA_ONLY_TOP
    unknown = _unknown(config, allowed_top, '')
    if which == 'full':
        if config.get('gradient_accumulation_steps', 1) != 1:
            problems.append('gradient_accumulation_steps: full Turbo runs Iris\'s distillation trainer, which has no '
                            'gradient accumulation (every update reads a fresh global batch); raise '
                            'micro_batch_size_per_gpu or the GPU count instead, or train a Turbo LoRA')
        unknown = [k for k in unknown if k != 'gradient_accumulation_steps']
        if 'save_dtype' in config:
            notes.append('save_dtype: full Turbo saves training checkpoints; pick the dtype when exporting '
                         '(export_distilled.py --dtype)')
            unknown = [k for k in unknown if k != 'save_dtype']
    if unknown:
        problems.append(f'not used by Turbo distillation: {", ".join(unknown)}. Turbo counts student updates '
                        '(max_steps, save_every_n_steps), not epochs, and has no eval set or warmup')
    if config.get('pipeline_stages', 1) != 1:
        problems.append('pipeline_stages must be 1: every GPU holds the whole model')
    monitoring = config.get('monitoring') or {}
    if monitoring.get('enable_wandb'):
        problems.append('[monitoring] enable_wandb: Turbo logs JSON lines to stdout and has no wandb support')

    if not config.get('output_dir'):
        problems.append('output_dir is required')
    if model.get('type') != 'iris':
        problems.append("[model] type must be 'iris': Turbo distills Iris-3B models")
    if not model.get('transformer_path'):
        problems.append('[model] transformer_path is required: the model to distill (the teacher)')
    model_allowed = {'type', 'transformer_path', 'dtype'}
    problems += [f'[model] {k} is not supported by Turbo distillation (the teacher\'s config.yaml decides it)'
                 for k in sorted(set(model) - model_allowed)]
    if which == 'full' and 'dtype' in model:
        notes.append("[model] dtype: full Turbo keeps Iris's own precision (FP32 weights, bf16 autocast)")

    if which == 'lora':
        if adapter.get('type', 'lora') != 'lora':
            problems.append("[adapter] type must be 'lora' for a Turbo LoRA")
        problems += [f'[adapter] {k} is not supported for a Turbo LoRA'
                     for k in _unknown(adapter, set(ADAPTER_KEYS) | {'type', 'dtype'}, '')]
        if 'dtype' in adapter:
            notes.append('[adapter] dtype: the adapters train in float32; save_dtype sets the saved files')

    if str(optimizer.get('type', 'adamw')).lower() != 'adamw':
        problems.append(f"[optimizer] type = {optimizer.get('type')!r}: Turbo trains with torch AdamW; "
                        "set type = 'adamw'")
    problems += [f'[optimizer] {k} is not configurable for Turbo'
                 for k in _unknown(optimizer, set(OPTIMIZER_KEYS) | {'type'}, '')]

    turbo_allowed = TURBO_COMMON | TURBO_PROMPTS | (TURBO_LORA if which == 'lora' else TURBO_FULL)
    for key in sorted(set(turbo) - turbo_allowed):
        if key in TURBO_LORA | TURBO_FULL:
            other = 'a Turbo LoRA ([adapter] set)' if key in TURBO_LORA else 'full Turbo (no [adapter])'
            problems.append(f'[turbo] {key} only applies to {other}')
        else:
            problems.append(f'[turbo] {key} is not a Turbo setting')

    if problems:
        raise TurboConfigError('invalid Turbo config:\n  ' + '\n  '.join(problems))

    settings = {TOP_LEVEL[k]: config[k] for k in TOP_LEVEL if k in config}
    if 'activation_checkpointing' in settings:
        settings['activation_checkpointing'] = _checkpointing(settings['activation_checkpointing'])
    settings['teacher'] = str(model['transformer_path'])
    settings.update({OPTIMIZER_KEYS[k]: v for k, v in optimizer.items() if k in OPTIMIZER_KEYS})
    for key in turbo_allowed - TURBO_PROMPTS:
        if key in turbo:
            settings[key] = turbo[key]

    if which == 'lora':
        if 'dtype' in model:
            settings['base_dtype'] = model['dtype']
        settings['adapter'] = {ADAPTER_KEYS[k]: v for k, v in adapter.items() if k in ADAPTER_KEYS}
        settings.setdefault('nccl_timeout_minutes', LORA_NCCL_TIMEOUT)
    else:
        settings['work_dir'] = settings.pop('output_dir', '')
        dmd = {k: settings.pop(k) for k in DMD_KEYS if k in settings}
        if dmd:
            settings['dmd'] = dmd
        settings.setdefault('keep_last', FULL_KEEP_LAST)
        settings.setdefault('nccl_timeout_minutes', FULL_NCCL_TIMEOUT)
    return which, settings, notes


def output_dir(config):
    return Path(config.get('output_dir') or '')


def prompts_plan(config):
    """('file', path) for a given prompts file, or ('build', path, datasets, extra prompt files)."""
    turbo = config.get('turbo') or {}
    datasets = config.get('dataset') or []
    datasets = [datasets] if isinstance(datasets, str) else list(datasets)
    extra = turbo.get('extra_prompts') or []
    extra = [extra] if isinstance(extra, str) else list(extra)
    if turbo.get('prompts_file'):
        if datasets or extra:
            raise TurboConfigError('set either [turbo] prompts_file or dataset / [turbo] extra_prompts, not both')
        return ('file', str(turbo['prompts_file']))
    if not datasets and not extra:
        raise TurboConfigError('Turbo needs prompts: set dataset (its captions become the prompts), '
                               '[turbo] extra_prompts, or [turbo] prompts_file')
    if not config.get('output_dir'):
        raise TurboConfigError('output_dir is required')
    return ('build', str(output_dir(config) / 'prompts.txt'), datasets, extra)


def build_prompts(plan, log):
    """Write output_dir/prompts.txt once. A resumed run must read the same file, so it is kept."""
    path = Path(plan[1])
    if path.is_file() and path.stat().st_size:
        log(f'using the prompts written at the start of this run: {path} (delete it to rebuild from the '
            'dataset, but only before a new run: a resume refuses a changed prompts file)')
        return
    from tools import iris_turbo_prompts
    argv = ['--output', str(path), '--no-progress']
    for dataset in plan[2]:
        argv += ['--dataset', dataset]
    for extra in plan[3]:
        argv += ['--extra-prompts', extra]
    path.parent.mkdir(parents=True, exist_ok=True)
    iris_turbo_prompts.main(argv)


def check_full(settings):
    """Iris's own validation of a full Turbo config, before anything loads."""
    from tools import iris_turbo_distill  # noqa: F401  (puts iris3b on the path)
    from omegaconf import OmegaConf
    from iris3b.distill.config import DistillConfig
    fields = {k: v for k, v in settings.items() if k not in ('keep_last', 'nccl_timeout_minutes')}
    try:
        cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DistillConfig), fields))
        cfg.validate()
    except Exception as error:
        raise TurboConfigError(f'invalid Turbo config: {error}') from error
    return fields


def _rank():
    return int(os.environ.get('RANK', '0'))


def _world():
    return int(os.environ.get('WORLD_SIZE', '1'))


def _log(message):
    if _rank() == 0:
        print(f'[iris_turbo] {message}', flush=True)


def _init_group(timeout_minutes):
    """The process group, created before the trainers so the ranks can wait for the prompts file."""
    if _world() == 1 or torch.distributed.is_initialized():
        return False
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        backend = 'nccl'
    else:
        backend = 'gloo'
    torch.distributed.init_process_group(backend, timeout=timedelta(minutes=timeout_minutes))
    return True


def _share(value):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        box = [value]
        torch.distributed.broadcast_object_list(box, src=0)
        return box[0]
    return value


def prepare(config, which, settings):
    """Rank 0 writes the prompts file (and full Turbo's YAML); every rank gets the paths or the error."""
    result = None
    if _rank() == 0:
        try:
            plan = prompts_plan(config)
            if plan[0] == 'build':
                build_prompts(plan, _log)
            result = {'prompts_file': plan[1]}
            if which == 'full':
                from omegaconf import OmegaConf
                fields = check_full({**settings, 'prompts_file': plan[1]})
                path = Path(fields['work_dir']) / 'turbo_distill.yaml'
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f'.{path.name}.tmp')
                OmegaConf.save(OmegaConf.create(fields), str(tmp))
                os.replace(tmp, path)
                result['yaml'] = str(path)
        except TurboConfigError as error:
            result = {'error': str(error)}
        except (Exception, SystemExit) as error:  # every rank must hear about it, not only rank 0
            import traceback
            traceback.print_exc()
            result = {'error': f'preparing the prompts failed: {type(error).__name__}: {error}'}
    result = _share(result)
    if 'error' in result:
        raise TurboConfigError(result['error'])
    return result


def run(config, allow_cudnn_attention=False):
    """Run the Turbo distillation a diffusion-pipe config with a [turbo] table describes."""
    which, settings, notes = translate(config)
    if which == 'full':
        check_full({**settings, 'prompts_file': '-'})
        return _run(config, which, settings, notes, allow_cudnn_attention)
    from tools import iris_turbo_lora as lora
    try:
        # every rank validates the same settings, so a bad value stops all of them before any work
        lora.config_from_dict({**settings, 'prompts_file': '-'})
        return _run(config, which, settings, notes, allow_cudnn_attention)
    except lora.ConfigError as error:
        raise TurboConfigError(str(error)) from error


def _run(config, which, settings, notes, allow_cudnn_attention):
    created = _init_group(settings['nccl_timeout_minutes'])
    try:
        for note in notes:
            _log(f'not used: {note}')
        prepared = prepare(config, which, settings)
        if which == 'lora':
            from tools import iris_turbo_lora as lora
            cfg = lora.config_from_dict({**settings, 'prompts_file': prepared['prompts_file']})
            _log(f'Turbo LoRA: {cfg.teacher} -> {cfg.output_dir} on {_world()} GPU(s)')
            return lora.run(cfg, allow_cudnn_attention)
        from tools import iris_turbo_distill as distill
        _log(f"full Turbo (Iris's distillation trainer): {settings['teacher']} -> {settings['work_dir']} "
             f'on {_world()} GPU(s)')
        argv = ['--config', prepared['yaml'], '--auto-resume', '--keep-last', str(settings['keep_last']),
                '--nccl-timeout-minutes', str(settings['nccl_timeout_minutes'])]
        if allow_cudnn_attention:
            argv.append('--allow-cudnn-attention')
        return distill.main(argv)
    finally:
        if created and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

