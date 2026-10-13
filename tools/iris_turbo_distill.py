"""Run Iris's Turbo distillation (TDM) with the checks a long run on shared storage needs.

    torchrun --nproc_per_node=8 tools/iris_turbo_distill.py \\
        --config submodules/iris-3b/configs/iris3b/turbo_stage1_6step.yaml \\
        teacher=/pvc/output/iris_full/<run>/epoch6 prompts_file=/pvc/turbo/prompts.txt \\
        work_dir=/pvc/turbo/stage1 strategy=fsdp2

This trains with Iris's own trainer (iris3b.distill, the code that produced Iris-3B Turbo) and
takes the same arguments as Iris's scripts/distill.py: a YAML config plus key=value overrides.
`teacher` can be a diffusion-pipe full fine-tune save directory or a LoRA merged with
tools/iris_merge_adapter.py; both are Iris's exported layout. Around the trainer it adds:

- PyTorch's cuDNN attention kernel is turned off, as diffusion-pipe's Iris training does: on
  torch 2.12 / cuDNN 9.20 / H200 its backward returns NaN for some real Iris activations.
  --allow-cudnn-attention keeps it.
- Before any weights load: the teacher is a text-to-image export and not a Turbo export, its
  attn_backend can run Iris's masked text adapter, the work directory accepts files and
  symlinks (Iris points latest.pth at the newest checkpoint with a symlink, which some network
  volumes refuse; without this check the run would fail at its first save), and there is room
  for the checkpoints. A checkpoint holds the FP32 student, EMA, fake network and both AdamW
  states, about 84 GB for Iris-3B.
- --keep-last N (default 2) deletes older step_*.pth after each save, never the one latest.pth
  points at. Iris keeps all of them: 16 saves, about 1.3 TB, for stage 1 at save_every 100.
  0 keeps every checkpoint.
- Leftover *.tmp files of a save that was interrupted (a pod evicted mid-write) are removed
  from work_dir/checkpoints at start; Iris never reads them.
- A warning when strategy=ddp likely does not fit the GPU: Iris states about 160 GB per GPU
  at batch_size 8, above an H200's 141 GB. strategy=fsdp2 shards student, fake network and
  optimizer states instead and keeps the global batch.

Export the result with Iris's script, which adds the distill.yaml that tells
scripts/sample.py to sample with the few-step schedule:

    python submodules/iris-3b/scripts/export_distilled.py <work_dir>/checkpoints/latest.pth <out> --dtype bf16
"""

import argparse
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

IRIS_ROOT = Path(__file__).resolve().parent.parent / 'submodules' / 'iris-3b'
sys.path.insert(0, str(IRIS_ROOT / 'src'))

import torch  # noqa: E402

CHECKPOINT_RE = re.compile(r'^step_(\d{8})\.pth$')
# FP32 copies per parameter in a checkpoint: student, EMA, fake network, and the two AdamW
# moments of both the student and the fake network.
FP32_COPIES_PER_CHECKPOINT = 7
# Iris: "ddp needs about 160 GB per GPU at batch 8". FP32 teacher, student, fake, EMA, two AdamW
# states and gradients are ~10 copies of 3B params (~120 GB); the rest scales with the batch.
DDP_STATIC_GB, DDP_PER_SAMPLE_GB = 120, 5
GB = 1e9


class PreflightError(RuntimeError):
    pass


def is_main_process():
    return int(os.environ.get('RANK', '0')) == 0


def log(message):
    if is_main_process():
        print(f'[iris_turbo_distill] {message}', flush=True)


def load_config(config_path, overrides):
    try:
        from iris3b.distill.config import load_distill_config
    except ImportError as error:
        raise PreflightError(
            f'iris3b.distill is missing ({error}). The submodule predates Iris-3B Turbo: run '
            '`git submodule update --init submodules/iris-3b`.') from error
    return load_distill_config(config_path, overrides)


def check_teacher(teacher):
    """The teacher's raw config (None for a training .pth) after the checks that fail fast."""
    from omegaconf import OmegaConf
    path = Path(teacher)
    if not path.exists():
        raise PreflightError(f'teacher {teacher} does not exist (it must be a local directory or .pth)')
    if path.is_file():
        return None
    for name in ('model.safetensors', 'config.yaml'):
        if not (path / name).is_file():
            raise PreflightError(
                f'teacher {teacher} has no {name}. Use a diffusion-pipe full fine-tune save directory, '
                'a LoRA merged with tools/iris_merge_adapter.py, or `hf download speridlabs/iris-3b`.')
    if (path / 'distill.yaml').is_file():
        raise PreflightError(
            f'teacher {teacher} is a Turbo (distilled) export. Distill from a text-to-image model: the '
            'base Iris-3B or your fine-tune of it.')
    raw = OmegaConf.to_container(OmegaConf.load(str(path / 'config.yaml')))
    if raw.get('task'):
        raise PreflightError(f'teacher {teacher} is an Iris task checkpoint ({raw["task"]}), not text-to-image')
    backend = raw.get('model', {}).get('attn_backend', 'sdpa')
    if backend != 'sdpa':
        raise PreflightError(
            f"teacher config.yaml sets model.attn_backend = {backend!r}. Iris applies it to every "
            "attention, including the text adapter's masked one, which only 'sdpa' can run, so the "
            f"first forward would fail. Set it to sdpa in {path / 'config.yaml'}.")
    return raw


def parameter_count(teacher):
    """Parameters in the teacher's model.safetensors from its header, or None."""
    path = Path(teacher)
    weights = path / 'model.safetensors' if path.is_dir() else None
    if weights is None or not weights.is_file():
        return None
    from safetensors import safe_open
    total = 0
    with safe_open(str(weights), framework='pt') as f:
        for key in f.keys():
            if not key.startswith('repa.'):
                total += math.prod(f.get_slice(key).get_shape())
    return total


def checkpoint_steps(directory):
    """[(step, path)] of step_XXXXXXXX.pth in `directory`, oldest first."""
    found = []
    for entry in Path(directory).iterdir() if Path(directory).is_dir() else []:
        match = CHECKPOINT_RE.match(entry.name)
        if match and entry.is_file() and not entry.is_symlink():
            found.append((int(match.group(1)), entry))
    return sorted(found)


def prune_checkpoints(directory, keep_last):
    """Delete all but the newest `keep_last` step_*.pth, never the target of latest.pth."""
    if keep_last <= 0:
        return []
    directory = Path(directory)
    latest = directory / 'latest.pth'
    protected = latest.resolve() if latest.is_symlink() else None
    removed = []
    for _, path in checkpoint_steps(directory)[:-keep_last]:
        if protected is not None and path.resolve() == protected:
            continue
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def remove_stale_temporaries(directory):
    """Delete scratch files an interrupted save left behind (`<name>.pth.tmp<tag>`)."""
    removed = []
    for entry in Path(directory).iterdir() if Path(directory).is_dir() else []:
        if '.pth.tmp' in entry.name and (entry.is_file() or entry.is_symlink()):
            entry.unlink(missing_ok=True)
            removed.append(entry)
    return removed


def check_work_dir(work_dir):
    """The checkpoint directory accepts a file, a symlink to it and an atomic rename over a link."""
    checkpoints = Path(work_dir) / 'checkpoints'
    try:
        checkpoints.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PreflightError(f'cannot create {checkpoints}: {error}') from error
    tag = uuid.uuid4().hex[:8]
    probe = checkpoints / f'.preflight-{tag}'
    link = checkpoints / f'.preflight-{tag}.link'
    renamed = checkpoints / f'.preflight-{tag}.latest'
    try:
        probe.write_bytes(b'ok')
        # what iris3b.train.ckpt._relink_latest does: symlink to an absolute path, then rename
        link.symlink_to(probe.resolve())
        link.replace(renamed)
        if renamed.read_bytes() != b'ok':
            raise OSError('the symlink does not resolve to its target')
    except OSError as error:
        raise PreflightError(
            f'{checkpoints} does not support what Iris checkpoints need (a file, a symlink to it and '
            f'a rename over the link): {error}. Some network volumes (object-storage FUSE mounts, '
            'SMB/Azure Files) refuse symlinks; put work_dir on a volume that allows them '
            '(block storage, NFS, Lustre).') from error
    finally:
        for path in (probe, link, renamed):
            if path.is_symlink() or path.exists():
                path.unlink()
    return checkpoints


def checkpoints_on_disk(cfg, keep_last):
    """How many checkpoints exist at once at worst: kept ones plus the one being written."""
    saves = cfg.max_steps // cfg.save_every + (1 if cfg.max_steps % cfg.save_every else 0)
    return saves if keep_last <= 0 else min(saves, keep_last + 1)


def check_space(cfg, checkpoints, keep_last, params):
    if params is None:
        return
    size = params * 4 * FP32_COPIES_PER_CHECKPOINT
    needed = size * checkpoints_on_disk(cfg, keep_last)
    existing = sum(path.stat().st_size for _, path in checkpoint_steps(checkpoints))
    free = shutil.disk_usage(checkpoints).free
    log(f'each checkpoint is about {size / GB:.0f} GB; this run needs up to {needed / GB:.0f} GB in '
        f'{checkpoints} ({free / GB:.0f} GB free, {existing / GB:.0f} GB of checkpoints already there)')
    # With pruning the checkpoints already there count against the kept ones; without it they
    # stay on top of the new ones (an overestimate on resume, where some are this run's own).
    if free + (existing if keep_last > 0 else 0) < needed:
        log('WARNING: not enough free space for the checkpoints this run will keep. Raise save_every, '
            'lower --keep-last, or grow the volume; a save that runs out of space stops the run.')


def check_memory(cfg):
    if cfg.strategy != 'ddp' or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / GB
    estimate = DDP_STATIC_GB + DDP_PER_SAMPLE_GB * cfg.batch_size
    if estimate > total:
        log(f'WARNING: strategy=ddp at batch_size {cfg.batch_size} needs about {estimate} GB per GPU '
            f'and this GPU has {total:.0f} GB. Pass strategy=fsdp2 (same global batch, sharded '
            'student, fake network and optimizer states) rather than lowering batch_size.')


def install_pruning(keep_last):
    """Prune after every checkpoint Iris writes (rank 0 is the only writer)."""
    if keep_last <= 0:
        return
    from iris3b.train import ckpt
    original = ckpt.save_checkpoint

    def save_checkpoint(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        for removed in prune_checkpoints(Path(path).parent, keep_last):
            print(f'[iris_turbo_distill] removed old checkpoint {removed}', flush=True)
        return result

    # The distillation trainer imports save_checkpoint from this module at each save.
    ckpt.save_checkpoint = save_checkpoint


def preflight(cfg, keep_last):
    raw = check_teacher(cfg.teacher)
    if not Path(cfg.prompts_file).is_file():
        raise PreflightError(f'prompts_file {cfg.prompts_file} does not exist; make it with tools/iris_turbo_prompts.py')
    for name in ('init_from', 'resume_from'):
        value = getattr(cfg, name)
        if value and not Path(value).exists():
            raise PreflightError(f'{name} {value} does not exist')
    check_memory(cfg)
    if not is_main_process():
        return
    checkpoints = check_work_dir(cfg.work_dir)
    for path in remove_stale_temporaries(checkpoints):
        log(f'removed {path}, left by an interrupted save')
    check_space(cfg, checkpoints, keep_last, parameter_count(cfg.teacher))
    if raw is not None and not (os.environ.get('HF_HOME') or os.environ.get('HF_HUB_CACHE')):
        encoder = raw.get('text_encoder', {}).get('pretrained', 'the Qwen3-VL text encoder')
        log(f'HF_HOME is not set: {encoder} downloads to ~/.cache/huggingface. In a pod whose home is '
            'not on the persistent volume, set HF_HOME to a directory on it to download only once.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help="An Iris distillation YAML (submodules/iris-3b/configs/iris3b/turbo_*.yaml).")
    parser.add_argument('overrides', nargs='*', metavar='KEY=VALUE', help='Overrides merged onto the YAML, as in Iris.')
    parser.add_argument('--keep-last', type=int, default=2,
                        help='Checkpoints to keep in work_dir/checkpoints (default 2; 0 keeps all, as Iris does).')
    parser.add_argument('--allow-cudnn-attention', action='store_true',
                        help="Leave PyTorch's cuDNN attention kernel on (it returns NaN gradients on some setups).")
    parser.add_argument('--skip-preflight', action='store_true', help='Run the trainer without the checks above.')
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if not args.allow_cudnn_attention and hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
        log("PyTorch's cuDNN attention kernel is off (--allow-cudnn-attention keeps it)")
    if not args.skip_preflight:
        try:
            preflight(cfg, args.keep_last)
        except PreflightError as error:
            raise SystemExit(f'[iris_turbo_distill] {error}') from None
    install_pruning(args.keep_last)

    from iris3b.distill.trainer import DistillationTrainer
    DistillationTrainer(cfg).run()


if __name__ == '__main__':
    main()
