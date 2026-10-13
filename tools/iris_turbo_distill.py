"""Run Iris's Turbo distillation (TDM) with the checks a long run on shared storage needs.

    torchrun --nproc_per_node=8 tools/iris_turbo_distill.py --auto-resume \\
        --config submodules/iris-3b/configs/iris3b/turbo_stage1_6step.yaml \\
        teacher=/pvc/out/iris_full/<run>/epoch6/ema prompts_file=/pvc/turbo/prompts.txt \\
        work_dir=/pvc/turbo/stage1 strategy=fsdp2

This trains with Iris's own trainer (iris3b.distill, the code that produced Iris-3B Turbo) and
takes the same arguments as Iris's scripts/distill.py: a YAML config plus key=value overrides,
which may come before or after this script's own options. `teacher` can be a diffusion-pipe
full fine-tune save directory (its ema/ directory when EMA was on), a LoRA merged with
tools/iris_merge_adapter.py, or an Iris training .pth. Around the trainer it adds:

- PyTorch's cuDNN attention kernel is turned off, as diffusion-pipe's Iris training does: on
  torch 2.12 / cuDNN 9.20 / H200 its backward returns NaN for some real Iris activations.
  --allow-cudnn-attention keeps it.
- The process group is created here with a 2 hour timeout (--nccl-timeout-minutes) instead of
  Iris's 30 minutes: rank 0 writes each ~84 GB checkpoint while the other ranks wait, and on a
  network volume slower than ~47 MB/s that alone exceeds 30 minutes.
- Checks on rank 0 before any weights load, whose outcome every rank shares. These stop the
  whole job at once with one message:
  - the teacher is a text-to-image model, not a Turbo export or student, and its attn_backend
    can run Iris's masked text adapter (sdpa, unless the text adapter is linear);
  - prompts_file is non-empty UTF-8; init_from and resume_from are readable distillation
    checkpoints, and a resume matches this GPU count and prompts file;
  - the checkpoint directory accepts a file, a symlink and a rename over an existing entry
    (Iris points latest.pth at the newest checkpoint that way; some network volumes refuse).
  These only warn: the write speed (timed with 1 GiB) against the timeout, free space, the
  pod's memory limit (fsdp2 gathers the whole checkpoint into rank 0's CPU memory to save it),
  /dev/shm and GPU memory.
- A work_dir that already holds checkpoints is not silently overwritten from step 0, which is
  what Iris does when a pod restarts with the same command. --auto-resume continues from the
  newest step_*.pth that loads instead, so the same command works for the first start and
  every restart. It does not rely on latest.pth, which lags one save behind when a pod dies
  between the rename and the relink, and dangles when the volume is mounted elsewhere. A
  step file left incomplete by a node crash is renamed to *.torn and skipped. A run that
  already reached max_steps exits at once instead of loading everything to save again.
- --keep-last N (default 2) deletes older step_*.pth after each save, never the one latest.pth
  points at. Iris keeps all of them: 16 saves, about 1.3 TB, for stage 1 at save_every 100.
  0 keeps every checkpoint. A failed delete is reported and does not stop training.
- Scratch files of an interrupted save and an unreadable empty-prompt cache are removed.

Export the result with Iris's script, which adds the distill.yaml that tells
scripts/sample.py to sample with the few-step schedule:

    python submodules/iris-3b/scripts/export_distilled.py <work_dir>/checkpoints/step_<N>.pth <out> --dtype bf16

(the step file itself: latest.pth is an absolute symlink and dangles under another mount path).
"""

import argparse
import hashlib
import math
import os
import re
import shutil
import sys
import time
import uuid
from datetime import timedelta
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
DDP_STATIC_GB, DDP_PER_SAMPLE_GB, MEMORY_MARGIN = 120, 5, 1.1
WRITE_TEST_BYTES = 1 << 30
# Where iris3b's Qwen3-VL encoder caches the empty-prompt embedding (text_encoder.null_embed_dir,
# which inference configs always leave at its default), relative to the working directory.
NULL_EMBED_DIR = Path('output/pretrained_models')
GB, GIB = 1e9, 1 << 30


class PreflightError(RuntimeError):
    pass


def rank():
    return int(os.environ.get('RANK', '0'))


def world_size():
    return int(os.environ.get('WORLD_SIZE', '1'))


def is_main_process():
    return rank() == 0


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


def load_payload(path):
    """A .pth's top-level dict; mmap reads only the pickle, not the tensors."""
    try:
        return torch.load(str(path), map_location='cpu', mmap=True, weights_only=False)
    except Exception as error:
        raise PreflightError(f'cannot read {path} ({type(error).__name__}: {error}); it may be incomplete, '
                             'e.g. written when a node crashed') from error


# ---- teacher ------------------------------------------------------------------------------------

def check_inference_config(raw, where):
    if not isinstance(raw, dict):
        raise PreflightError(f'{where} has no Iris config')
    if raw.get('task'):
        raise PreflightError(f'{where} is an Iris task checkpoint ({raw["task"]}), not text-to-image')
    model = raw.get('model') or {}
    backend = model.get('attn_backend', 'sdpa')
    if backend != 'sdpa' and model.get('text_adapter', 'lap_blocks2') != 'linear':
        raise PreflightError(
            f"{where} sets model.attn_backend = {backend!r}. Iris applies it to every attention, "
            "including the text adapter's masked one, which only 'sdpa' can run, so the first forward "
            "would fail. Set it to sdpa.")


def check_teacher(teacher):
    """(raw config, parameter count or None)."""
    from omegaconf import OmegaConf
    path = Path(teacher)
    if not path.exists():
        raise PreflightError(f'teacher {teacher} does not exist (it must be a local directory or .pth)')
    if path.is_file():
        payload = load_payload(path)
        if not isinstance(payload, dict):
            raise PreflightError(f'teacher {teacher} is not an Iris checkpoint')
        if 'distill_format_version' in payload:
            raise PreflightError(
                f'teacher {teacher} is a distillation checkpoint (a Turbo student). Distill from a '
                'text-to-image model: the base Iris-3B or your fine-tune of it.')
        check_inference_config(payload.get('config'), f'teacher {teacher}')
        state = payload.get('state_dict_ema') or payload.get('state_dict') or {}
        params = sum(t.numel() for k, t in state.items() if not k.startswith('repa.')) or None
        return payload['config'], params
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
    check_inference_config(raw, f'teacher {path / "config.yaml"}')
    return raw, parameter_count(path / 'model.safetensors')


def parameter_count(weights):
    """Parameters in a model.safetensors, from its header."""
    from safetensors import safe_open
    total = 0
    with safe_open(str(weights), framework='pt') as f:
        for key in f.keys():
            if not key.startswith('repa.'):
                total += math.prod(f.get_slice(key).get_shape())
    return total


# ---- prompts and distillation checkpoints -----------------------------------------------------

def check_prompts(path):
    """sha256 of the prompts file, as Iris records it for resume."""
    path = Path(path)
    if not path.is_file():
        raise PreflightError(f'prompts_file {path} does not exist; make it with tools/iris_turbo_prompts.py')
    content = path.read_bytes()
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError as error:
        raise PreflightError(f'prompts_file {path} is not UTF-8: {error}') from error
    if not any(line.strip() for line in text.splitlines()):
        raise PreflightError(f'prompts_file {path} has no prompts')
    return hashlib.sha256(content).hexdigest()


def check_distill_checkpoint(value, name):
    path = Path(value)
    if path.is_symlink() and not path.exists():
        raise PreflightError(
            f'{name} {value} is a symlink to {os.readlink(path)}, which does not exist here: the volume is '
            'mounted at another path than when it was written. Pass the step_XXXXXXXX.pth file itself.')
    if not path.is_file():
        raise PreflightError(f'{name} {value} does not exist or is not a file')
    payload = load_payload(path)
    if not isinstance(payload, dict) or 'distill_format_version' not in payload:
        raise PreflightError(f'{name} {value} was not written by Iris distillation')
    return payload


def checkpoint_steps(directory):
    """[(step, path)] of step_XXXXXXXX.pth in `directory`, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for entry in directory.iterdir():
        match = CHECKPOINT_RE.match(entry.name)
        if match and entry.is_file() and not entry.is_symlink():
            found.append((int(match.group(1)), entry))
    return sorted(found)


def newest_checkpoint(checkpoints):
    """What --auto-resume continues from: the newest step_*.pth in `checkpoints` that loads.

    Not latest.pth: Iris repoints it after renaming the step file into place, so a pod killed
    in between leaves it one save behind, and after a move of the volume it dangles. A step file
    that does not load (a node crash before the data reached the disk) is renamed to *.torn,
    so the resume and pruning stop seeing it.
    """
    for _, path in reversed(checkpoint_steps(checkpoints)):
        try:
            load_payload(path)
        except PreflightError as error:
            torn = path.with_name(path.name + '.torn')
            path.replace(torn)
            log(f'WARNING: {error}. Renamed it to {torn.name}; delete it once the run continues.')
            continue
        return path.resolve()
    return None


def resolve_run(cfg, checkpoints, prompts_sha, auto_resume):
    """The resume_from to use (or ''). Refuses to restart over, or next to, other checkpoints."""
    existing = checkpoint_steps(checkpoints)
    resume_from = cfg.resume_from
    if not resume_from and existing:
        if not auto_resume:
            raise PreflightError(
                f'{checkpoints} already holds {existing[-1][1].name}. Iris would start over from step 0 and '
                f'overwrite it. Pass resume_from={checkpoints / "latest.pth"} (or --auto-resume) to continue, '
                'or a new work_dir to start over.')
        newest = newest_checkpoint(checkpoints)
        if newest is None:
            log('--auto-resume: no readable checkpoint left; starting over')
            resume_from = ''
        else:
            resume_from = str(newest)
            log(f'--auto-resume: continuing from {resume_from}')
    if not resume_from:
        if cfg.init_from:
            payload = check_distill_checkpoint(cfg.init_from, 'init_from')
            key = 'state_dict_ema' if cfg.init_weights == 'ema' else 'state_dict'
            if key not in payload or 'state_dict_fake' not in payload:
                raise PreflightError(f'init_from {cfg.init_from} has no {key} or state_dict_fake')
        return ''
    payload = check_distill_checkpoint(resume_from, 'resume_from')
    if payload.get('world_size') != world_size():
        raise PreflightError(f'resume_from {resume_from} was written by {payload.get("world_size")} processes; '
                             f'this run has {world_size()}. Resume with the same GPU count.')
    if (payload.get('prompt_state') or {}).get('sha256') != prompts_sha:
        raise PreflightError(f'resume_from {resume_from} was trained on another prompts file; resume with the '
                             f'same {cfg.prompts_file} it used (do not regenerate it between runs).')
    step = payload.get('generator_updates', 0)
    newer = [path.name for found_step, path in checkpoint_steps(checkpoints) if found_step > step]
    if newer:
        raise PreflightError(
            f'{checkpoints} holds {", ".join(newer)}, newer than the resume point (step {step}): from a run '
            'that went further, or incomplete after a crash. Resuming would overwrite them one by one and '
            'pruning would keep the wrong ones: move them out of the directory first.')
    return resume_from


def finished(resume_from, cfg):
    """True when the checkpoint to resume already reached max_steps (a pod restarted after the end)."""
    return bool(resume_from) and load_payload(resume_from).get('generator_updates', 0) >= cfg.max_steps


# ---- storage, memory ----------------------------------------------------------------------------

def check_work_dir(work_dir):
    """The checkpoint directory accepts a file, a symlink to it and a rename over an existing entry."""
    checkpoints = Path(work_dir) / 'checkpoints'
    try:
        checkpoints.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PreflightError(f'cannot create {checkpoints}: {error}') from error
    tag = uuid.uuid4().hex[:8]
    probe = checkpoints / f'.preflight-{tag}'
    link = checkpoints / f'.preflight-{tag}.link'
    current = checkpoints / f'.preflight-{tag}.latest'
    try:
        probe.write_bytes(b'ok')
        current.write_bytes(b'old')
        # what iris3b.train.ckpt._relink_latest does from the second save on: symlink to an
        # absolute path, then rename it over the existing latest.pth
        link.symlink_to(probe.resolve())
        link.replace(current)
        if current.read_bytes() != b'ok':
            raise OSError('the renamed symlink does not resolve to its target')
    except OSError as error:
        raise PreflightError(
            f'{checkpoints} does not support what Iris checkpoints need (a file, a symlink to it and a '
            f'rename over an existing entry): {error}. Some network volumes refuse symlinks (e.g. '
            'mountpoint-s3, SMB mounted without mfsymlinks); put work_dir on one that allows them.') from error
    finally:
        for path in (probe, link, current):
            if path.is_symlink() or path.exists():
                path.unlink()
    return checkpoints


def write_speed(directory, size=WRITE_TEST_BYTES):
    """Bytes per second for an fsync'd write of `size` bytes into `directory`."""
    path = Path(directory) / f'.preflight-write-{uuid.uuid4().hex[:8]}'
    block = os.urandom(64 << 20)
    try:
        start = time.perf_counter()
        with open(path, 'wb') as f:
            written = 0
            while written < size:
                f.write(block[:size - written])
                written += min(len(block), size - written)
            f.flush()
            os.fsync(f.fileno())
        return size / max(time.perf_counter() - start, 1e-9)
    finally:
        path.unlink(missing_ok=True)


def remove_stale_temporaries(directory):
    """Delete scratch files an interrupted save left behind (`<name>.pth.tmp<tag>`)."""
    removed = []
    for entry in Path(directory).iterdir() if Path(directory).is_dir() else []:
        if '.pth.tmp' in entry.name and (entry.is_file() or entry.is_symlink()):
            entry.unlink(missing_ok=True)
            removed.append(entry)
    return removed


def remove_broken_null_cache(directory=NULL_EMBED_DIR):
    """Iris writes the empty-prompt cache without a rename; a pod killed mid-write leaves it torn."""
    removed = []
    for path in Path(directory).glob('null_embed_*.pth') if Path(directory).is_dir() else []:
        try:
            torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def checkpoints_on_disk(cfg, keep_last):
    """How many checkpoints exist at once at worst: kept ones plus the one being written."""
    saves = cfg.max_steps // cfg.save_every + (1 if cfg.max_steps % cfg.save_every else 0)
    return saves if keep_last <= 0 else min(saves, keep_last + 1)


def check_space(cfg, checkpoints, keep_last, checkpoint_size):
    needed = checkpoint_size * checkpoints_on_disk(cfg, keep_last)
    existing = sum(path.stat().st_size for _, path in checkpoint_steps(checkpoints))
    free = shutil.disk_usage(checkpoints).free
    log(f'each checkpoint is about {checkpoint_size / GB:.0f} GB; this run needs up to {needed / GB:.0f} GB '
        f'in {checkpoints} ({free / GB:.0f} GB free, {existing / GB:.0f} GB of checkpoints already there)')
    # With pruning the checkpoints already there count against the kept ones; without it they
    # stay on top of the new ones (an overestimate on resume, where some are this run's own).
    if free + (existing if keep_last > 0 else 0) < needed:
        log('WARNING: not enough free space for the checkpoints this run will keep. Raise save_every, '
            'lower --keep-last, or grow the volume; a save that runs out of space stops the run.')


def check_write_speed(checkpoints, checkpoint_size, timeout_minutes):
    try:
        speed = write_speed(checkpoints)
    except OSError as error:
        log(f'WARNING: a 1 GiB test write into {checkpoints} failed ({error}); checkpoints will fail the same way')
        return
    seconds = checkpoint_size / speed
    log(f'{checkpoints} writes {speed / 1e6:.0f} MB/s: a checkpoint takes about {seconds / 60:.1f} min '
        f'(process group timeout {timeout_minutes} min)')
    if seconds > 0.5 * timeout_minutes * 60:
        log('WARNING: the other ranks wait for rank 0 while it writes a checkpoint, and the job aborts when '
            'that exceeds the timeout. Raise --nccl-timeout-minutes or put work_dir on faster storage.')


def memory_limit():
    """The pod's (cgroup's) memory limit, capped by physical memory; None if unknown."""
    limits = []
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit() and int(value) < 1 << 60:
            limits.append(int(value))
        break
    try:
        limits.append(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        pass
    return min(limits) if limits else None


def check_host_memory(cfg, checkpoint_size, limit):
    """fsdp2 gathers each checkpoint into rank 0's CPU memory before writing it. (The teacher
    file each rank loads is memory-mapped, so the ranks share one page-cache copy of it.)"""
    if limit is None or cfg.strategy != 'fsdp2':
        return
    needed = checkpoint_size * MEMORY_MARGIN
    if needed > limit:
        log(f'WARNING: fsdp2 gathers each checkpoint into rank 0\'s CPU memory before writing it: about '
            f'{needed / GB:.0f} GB, above this pod\'s memory limit of {limit / GB:.0f} GB. Expect the pod '
            'to be OOM-killed at the first save; raise the limit.')


def check_shm(minimum=GIB):
    try:
        total = shutil.disk_usage('/dev/shm').total
    except OSError:
        return
    if total < minimum:
        log(f'WARNING: /dev/shm is {total / 2**20:.0f} MB (Kubernetes defaults to 64 MB); NCCL needs more. '
            'Mount a memory-backed emptyDir at /dev/shm.')


def check_gpu_memory(cfg):
    if cfg.strategy != 'ddp' or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / GB
    estimate = DDP_STATIC_GB + DDP_PER_SAMPLE_GB * cfg.batch_size
    if estimate * MEMORY_MARGIN > total:
        log(f'WARNING: strategy=ddp at batch_size {cfg.batch_size} needs about {estimate} GB per GPU '
            f'and this GPU has {total:.0f} GB. Pass strategy=fsdp2 (same global batch, sharded '
            'student, fake network and optimizer states) rather than lowering batch_size.')


# ---- pruning ------------------------------------------------------------------------------------

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


def install_pruning(keep_last):
    """Prune after every checkpoint Iris writes (rank 0 is the only writer)."""
    if keep_last <= 0:
        return
    from iris3b.train import ckpt
    original = ckpt.save_checkpoint

    def save_checkpoint(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        # The save itself succeeded; a failed delete must not fail it (Iris turns any error here
        # into a stop of the whole run).
        try:
            for removed in prune_checkpoints(Path(path).parent, keep_last):
                print(f'[iris_turbo_distill] removed old checkpoint {removed}', flush=True)
        except OSError as error:
            print(f'[iris_turbo_distill] WARNING: could not remove old checkpoints: {error}', flush=True)
        return result

    # The distillation trainer imports save_checkpoint from this module at each save.
    ckpt.save_checkpoint = save_checkpoint


# ---- launcher -----------------------------------------------------------------------------------

def preflight(cfg, args):
    """Rank 0's checks. Returns the decisions every rank applies: {'resume_from': ...}."""
    check_gpu_memory(cfg)
    raw, params = check_teacher(cfg.teacher)
    prompts_sha = check_prompts(cfg.prompts_file)
    checkpoints = check_work_dir(cfg.work_dir)
    for path in remove_stale_temporaries(checkpoints):
        log(f'removed {path}, left by an interrupted save')
    resume_from = resolve_run(cfg, checkpoints, prompts_sha, args.auto_resume)
    if finished(resume_from, cfg):
        return {'done': f'{resume_from} already reached max_steps={cfg.max_steps}; nothing to do. '
                        'Raise max_steps to train further.'}
    if params is not None:
        checkpoint_size = params * 4 * FP32_COPIES_PER_CHECKPOINT
        check_space(cfg, checkpoints, args.keep_last, checkpoint_size)
        if not args.skip_write_test:
            check_write_speed(checkpoints, checkpoint_size, args.nccl_timeout_minutes)
        check_host_memory(cfg, checkpoint_size, memory_limit())
    check_shm()
    for path in remove_broken_null_cache():
        log(f'removed {path}, an unreadable empty-prompt cache')
    if not (os.environ.get('HF_HOME') or os.environ.get('HF_HUB_CACHE')):
        encoder = (raw.get('text_encoder') or {}).get('pretrained', 'the Qwen3-VL text encoder')
        log(f'HF_HOME is not set: {encoder} downloads to ~/.cache/huggingface. In a pod whose home is '
            'not on the persistent volume, set HF_HOME to a directory on it and download it once '
            'before torchrun, so the ranks do not all download it at the same time.')
    return {'resume_from': resume_from}


def init_process_group(cfg, timeout_minutes):
    """Create the group Iris would, with a longer timeout. True if this created it."""
    if world_size() == 1 and cfg.strategy != 'fsdp2':
        return False
    if not torch.cuda.is_available() or torch.distributed.is_initialized():
        return False
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=timeout_minutes))
    return True


def share(decision):
    """Rank 0's decision, on every rank."""
    if torch.distributed.is_available() and torch.distributed.is_initialized() \
            and torch.distributed.get_world_size() > 1:
        box = [decision]
        torch.distributed.broadcast_object_list(box, src=0)
        return box[0]
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help="An Iris distillation YAML: examples/iris/turbo/stage*.yaml or submodules/iris-3b/configs/iris3b/turbo_*.yaml.")
    parser.add_argument('overrides', nargs='*', metavar='KEY=VALUE', help='Overrides merged onto the YAML, as in Iris.')
    parser.add_argument('--auto-resume', action='store_true',
                        help='Continue from work_dir/checkpoints when it holds checkpoints and resume_from is unset.')
    parser.add_argument('--keep-last', type=int, default=2,
                        help='Checkpoints to keep in work_dir/checkpoints (default 2; 0 keeps all, as Iris does).')
    parser.add_argument('--nccl-timeout-minutes', type=int, default=120,
                        help='Process group timeout (default 120; Iris uses 30).')
    parser.add_argument('--allow-cudnn-attention', action='store_true',
                        help="Leave PyTorch's cuDNN attention kernel on (it returns NaN gradients on some setups).")
    parser.add_argument('--skip-write-test', action='store_true', help='Do not time a 1 GiB write into work_dir.')
    parser.add_argument('--skip-preflight', action='store_true', help='Run the trainer without the checks above.')
    args = parser.parse_intermixed_args(argv)
    if args.skip_preflight and args.auto_resume:
        parser.error('--auto-resume is part of the preflight; it cannot be combined with --skip-preflight')

    try:
        cfg = load_config(args.config, args.overrides)
    except PreflightError as error:
        raise SystemExit(f'[iris_turbo_distill] {error}') from None
    if not args.allow_cudnn_attention and hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
        log("PyTorch's cuDNN attention kernel is off (--allow-cudnn-attention keeps it)")
    created = init_process_group(cfg, args.nccl_timeout_minutes)
    if world_size() > 1 and not torch.distributed.is_initialized():
        raise SystemExit(f'[iris_turbo_distill] WORLD_SIZE is {world_size()} but CUDA is not available, so no '
                         'process group can be created. Are the GPUs allocated to this pod?')
    try:
        if not args.skip_preflight:
            decision = None
            if is_main_process():
                try:
                    decision = preflight(cfg, args)
                except PreflightError as error:
                    decision = {'error': str(error)}
                except Exception as error:  # every rank must hear about it, not only rank 0
                    import traceback
                    traceback.print_exc()
                    decision = {'error': f'preflight failed: {type(error).__name__}: {error}'}
            decision = share(decision)
            if 'error' in decision:
                raise SystemExit(f'[iris_turbo_distill] {decision["error"]}' if is_main_process() else 1)
            if 'done' in decision:
                log(decision['done'])
                return
            cfg.resume_from = decision['resume_from']
        install_pruning(args.keep_last)

        from iris3b.distill.trainer import DistillationTrainer
        DistillationTrainer(cfg).run()
    finally:
        if created and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
