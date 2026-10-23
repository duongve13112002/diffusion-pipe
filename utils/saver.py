from pathlib import Path
import os
import shutil
import time
import sys

import torch
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.utils.logging import logger

from utils.common import all_ok, fsync_dir, is_main_process, torch_save_atomic, write_atomic
from utils.ema import EMA_SUBDIR, save_ema_checkpoint


def convert_state_dict_dtype(state_dict, dtype):
    for key, v in state_dict.items():
        state_dict[key] = v.to(device='cpu', dtype=dtype)


def publish_dir(staging, final):
    """Make the fully written directory `staging` appear as `final`: fsync its files, then rename.

    A directory already at `final` (the same name saved again, e.g. a manual save on a step that
    was just saved) is moved aside first and removed only once the new one is in place.
    """
    staging, final = Path(staging), Path(final)
    for root, _, files in os.walk(staging):
        for file in files:
            # Windows fsyncs (FlushFileBuffers) only a handle open for writing.
            fd = os.open(Path(root) / file, os.O_RDWR if os.name == 'nt' else os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fsync_dir(root)
    old = None
    if final.exists():
        old = final.with_name(f'.{final.name}.old-{os.getpid()}-{time.monotonic_ns()}')
        os.replace(final, old)
    os.replace(staging, final)
    fsync_dir(final.parent)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def remove_unpublished_saves(save_root):
    """Remove what interrupted saves left behind (see publish_dir): staging directories that were
    never renamed into place, and replaced directories that were not yet removed. Called before
    any save of this run starts, so none of them is in use."""
    save_root = Path(save_root)
    if not save_root.is_dir():
        return
    # publish_dir moves an existing save aside before renaming the new one in: a crash between
    # the two renames leaves only the old one, which goes back into place.
    for pattern in ('.*.old-*', f'*/.{EMA_SUBDIR}.old-*'):
        for old in sorted(save_root.glob(pattern), reverse=True):
            final = old.with_name(old.name[1:].split('.old-')[0])
            if old.is_dir() and not final.exists():
                logger.info(f'Restoring {final} from {old}')
                os.replace(old, final)
                fsync_dir(final.parent)
    for pattern in ('.*.partial', f'*/.{EMA_SUBDIR}.partial', '.*.old-*', f'*/.{EMA_SUBDIR}.old-*'):
        for path in save_root.glob(pattern):
            if path.is_dir():
                logger.info(f'Removing unfinished save {path}')
                shutil.rmtree(path, ignore_errors=True)
    # Files that write_atomic was writing when the process was killed (`latest`, the files of a
    # training checkpoint).
    for pattern in ('.*.tmp-*', '*/.*.tmp-*'):
        for path in save_root.glob(pattern):
            if path.is_file():
                logger.info(f'Removing unfinished file {path}')
                path.unlink(missing_ok=True)


def read_latest(save_root):
    """The tag `latest` names (the checkpoint a resume loads), or None when there is none yet."""
    try:
        return (Path(save_root) / 'latest').read_text().strip()
    except FileNotFoundError:
        return None


class _CheckpointWrites:
    """DeepSpeed's checkpoint engine for one save_checkpoint call. Each file is written under a
    temporary name and renamed into place, and a failed write is kept instead of raised:
    DeepSpeed has barriers after the writes, and a rank that raised before them would leave the
    others waiting there until the process group times out. Saver.save_checkpoint raises it on
    every rank once DeepSpeed returns."""

    def __init__(self, engine):
        self.engine = engine
        self.error = None

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _guard(self, write, *args, **kwargs):
        if self.error is not None:
            return
        try:
            write(*args, **kwargs)
        except Exception as e:
            self.error = e

    def makedirs(self, path, exist_ok=False):
        self._guard(self.engine.makedirs, path, exist_ok=exist_ok)

    def save(self, state_dict, path):
        if isinstance(self.engine, TorchCheckpointEngine):
            self._guard(torch_save_atomic, state_dict, path,
                        _use_new_zipfile_serialization=self.engine.zipfile_serialization)
        else:
            self._guard(self.engine.save, state_dict, path)


def prune_old_checkpoints(save_root, prefix, keep, dry_run=False):
    """Keep the `keep` highest-numbered `<prefix><N>` directories under save_root, delete the rest.

    Each kind is counted on its own, so keep=3 means three epoch* AND three step* AND three
    global_step*, not three in total. They are produced by different triggers at different rates
    and pruning them together would let a fast trigger evict a slow one.

    Ordered by the number in the name, not by mtime: a resumed run rewrites older directories'
    timestamps, and the number is what actually says which checkpoint is later.

    The newest is never deleted, nor the one `latest` names: that is the tag a resume loads, and
    deleting it turns a prunable cache into an unrecoverable run. A global_step* newer than
    `latest` is one whose save did not finish: it is neither counted nor removed here (its next
    save replaces it), so `keep` complete checkpoints remain.

    Returns the paths removed (or that would be, under dry_run).
    """
    if not keep or keep < 1:
        return []
    save_root = Path(save_root)
    if not save_root.is_dir():
        return []

    numbered = []
    for path in save_root.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix):]
        if suffix.isdigit():
            numbered.append((int(suffix), path))

    try:
        latest = read_latest(save_root)
    except OSError as e:
        # Not knowing which one a resume loads, delete none of them.
        logger.warning(f'keep_last_n_checkpoints: could not read {save_root / "latest"} ({e}); not pruning')
        return []
    if latest is not None and latest.startswith(prefix) and latest[len(prefix):].isdigit():
        numbered = [(n, path) for n, path in numbered if n <= int(latest[len(prefix):])]
    numbered.sort(key=lambda pair: pair[0])
    doomed = [path for _, path in numbered[:-keep] if path.name != latest] if len(numbered) > keep else []
    if not dry_run:
        for path in doomed:
            logger.info(f'keep_last_n_checkpoints: removing {path.name}')
            shutil.rmtree(path, ignore_errors=True)
    return doomed


# Each kind is counted separately, so a directory belongs to exactly one of these. The names
# do not collide despite the shared word: startswith is applied to the whole prefix, and
# 'global_step500' does not start with 'step'. Order here carries no meaning.
CHECKPOINT_PREFIXES = ('epoch', 'step', 'global_step')


def prune_all_checkpoint_kinds(config, save_root):
    """Apply keep_last_n_checkpoints to every kind that accumulates, each counted separately."""
    keep = config.get('keep_last_n_checkpoints', None)
    if not keep:
        return
    if not is_main_process():
        return
    for prefix in CHECKPOINT_PREFIXES:
        prune_old_checkpoints(save_root, prefix, keep)


last_checkpoint_time = None
def need_to_checkpoint(config, epoch=None):
    global last_checkpoint_time

    if epoch is not None:
        if 'checkpoint_every_n_epochs' in config and epoch % config['checkpoint_every_n_epochs'] == 0:
            last_checkpoint_time = time.time()
            return True
        else:
            return False

    if 'checkpoint_every_n_minutes' not in config:
        return False

    checkpoint = False
    # rank 0 tracks if we need to checkpoint, broadcasts to everyone else
    if is_main_process():
        current_time = time.time()
        if last_checkpoint_time is None:
            last_checkpoint_time = current_time
        elif (current_time - last_checkpoint_time) / 60 > config['checkpoint_every_n_minutes']:
            checkpoint = True
            last_checkpoint_time = current_time
    result = [checkpoint]
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


# How long the writer of a save looks for the other stages' files: a volume shared between nodes
# can show a file written on another node a little late.
STAGE_FILE_WAIT_SECONDS = 180


def stage_files(tmp_dir, num_stages):
    """The weights file of every pipeline stage in tmp_dir, waiting a little for late ones. A save
    missing a stage must fail rather than publish part of the model."""
    paths = [Path(tmp_dir) / f'state_dict_{stage}.bin' for stage in range(num_stages)]
    deadline = time.monotonic() + STAGE_FILE_WAIT_SECONDS
    while missing := [p for p in paths if not p.is_file()]:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f'{", ".join(p.name for p in missing)} not found in {tmp_dir}: the other pipeline stages '
                'wrote their weights there and it does not show them. output_dir must be on a volume that '
                'every node shares.'
            )
        time.sleep(1)
    return paths


def manual_save_signal(save_root):
    """(save, quit) from the signal files the user can create in save_root: `save`, or `save_quit`
    to save and quit. Rank 0 alone looks and tells the others. Ranks reach this at different times
    (pipeline stages, other nodes), and a file seen by only some of them would send those into the
    save's collectives while the others go on training."""
    signal = [(False, False)]
    if is_main_process():
        for name, quit in (('save', False), ('save_quit', True)):
            path = save_root / name
            if path.is_file():
                signal[0] = (True, quit)
                try:
                    os.remove(path)
                except OSError as e:
                    print(f'WARNING: could not remove the signal file {path}: {e}')
                break
    torch.distributed.broadcast_object_list(signal, src=0)
    return signal[0]


class Saver:
    def __init__(self, args, config, is_adapter, save_root, model, train_dataloader, model_engine, pipeline_model,
                 ema=None, loss_average=None):
        self.args = args
        # utils.step_stats.LossAverage, or None. When set, every checkpoint carries its state.
        self.loss_average = loss_average
        # utils.ema.ParameterEMA, or None. When set, every save also writes its average to
        # <save dir>/ema/ and every checkpoint carries it.
        self.ema = ema
        self.config = config
        self.is_adapter = is_adapter
        self.save_root = Path(save_root)
        self.model = model
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model
        if is_main_process():
            remove_unpublished_saves(self.save_root)

    def _save(self, name, ema, collect, write):
        """Gather the weights on data-parallel rank 0 of each stage, then write them from stage 0.

        The directory is built under a temporary name beside the real one and renamed into place
        once every file in it is on disk, so a crash or a network volume hiccup leaves either
        the previous save or none, never a torn one. Each step's failure is shared with every
        rank (all_ok) before the next collective, so a failed save stops the run instead of
        leaving the other ranks to wait out the process group timeout.
        """
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        if ema:
            save_dir = save_dir / EMA_SUBDIR
        staging = save_dir.parent / f'.{save_dir.name}.partial'
        tmp_dir = staging / 'tmp'
        writer = dp_id == 0 and stage_id == 0

        error = None
        if writer:
            try:
                # Left by a save that was interrupted: never published, so nothing to keep.
                if staging.exists():
                    shutil.rmtree(staging)
                os.makedirs(tmp_dir)
            except Exception as e:
                error = e
        all_ok(error)

        error = None
        if dp_id == 0:
            try:
                partial_state_dict = collect()
                if 'save_dtype' in self.config:
                    convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
                torch_save_atomic(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')
            except Exception as e:
                error = e
        all_ok(error)

        error = None
        if writer:
            try:
                state_dict = {}
                for path in stage_files(tmp_dir, self.model_engine.num_stages):
                    state_dict.update(torch.load(path, weights_only=True, map_location='cpu'))
                shutil.rmtree(tmp_dir)
                write(staging, state_dict)
                shutil.copy(self.args.config, staging)
                publish_dir(staging, save_dir)
            except Exception as e:
                error = e
        all_ok(error)

    def save_adapter(self, name, ema=False):
        def collect():
            partial_state_dict = {}
            for param_name, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, 'original_name'):
                        logger.warning(f'WARNING: parameter {param_name} requires_grad but does not have original_name. Not saving it.')
                        continue
                    # TODO: maybe this needs to change if we ever have non-lora adapters?
                    key = p.original_name.replace('.default', '').replace('.modules_to_save', '')
                    partial_state_dict[key] = self.ema.averaged(key, p) if ema else p.detach()
            return partial_state_dict
        self._save(name, ema, collect, self.model.save_adapter)

    def save_full_model(self, name, ema=False):
        def collect():
            # With BF16_Optimizer, we get pickle errors unless we do p.detach(). I have no idea why.
            if ema:
                return {p.original_name: self.ema.averaged(p.original_name, p)
                        for p in self.pipeline_model.parameters() if hasattr(p, 'original_name')}
            return {p.original_name: p.detach() for p in self.pipeline_model.parameters() if hasattr(p, 'original_name')}
        self._save(name, ema, collect, self.model.save_model)

    def save_model(self, name):
        if is_main_process():
            print(f'Saving model to directory {name}')
        save = self.save_adapter if self.is_adapter else self.save_full_model
        save(name)
        if self.ema is not None:
            save(name, ema=True)

    def save_checkpoint(self, step, examples):
        if self.model_engine.checkpoint_engine.is_decoupled():
            # Its files are written after save_checkpoint returns, so `latest` would come first.
            raise ValueError('A decoupled DeepSpeed checkpoint engine is not supported: remove the '
                             '"checkpoint" "writer" "decoupled" setting from the DeepSpeed config.')
        # DeepSpeed's own default tag, spelled out so the EMA file lands beside it.
        tag = f'global_step{self.model_engine.global_steps}'
        save_root = Path(self.save_root)

        # A directory of this tag that `latest` does not name is what a save cut short left;
        # DeepSpeed would write into it and keep its stale files.
        error = None
        if is_main_process():
            try:
                if (save_root / tag).exists() and read_latest(save_root) != tag:
                    logger.info(f'Removing unfinished checkpoint {save_root / tag}')
                    shutil.rmtree(save_root / tag)
            except Exception as e:
                error = e
        all_ok(error)

        if self.ema is not None:
            # The average lives on data-parallel rank 0 of each stage; each writes its own.
            error = None
            if self.model_engine.grid.get_data_parallel_rank() == 0:
                try:
                    save_ema_checkpoint(self.ema, self.save_root, tag, self.model_engine.grid.get_pipe_parallel_rank())
                except Exception as e:
                    error = e
            all_ok(error)

        writes = _CheckpointWrites(self.model_engine.checkpoint_engine)
        self.model_engine.checkpoint_engine = writes
        try:
            self.model_engine.save_checkpoint(
                self.save_root,
                tag=tag,
                client_state={
                    'step': step,
                    'examples': examples,
                    'custom_loader': self.train_dataloader.state_dict(),
                    **({'loss_average': self.loss_average.state_dict()} if self.loss_average is not None else {}),
                },
                save_latest=False,
                exclude_frozen_parameters=True
            )
        finally:
            self.model_engine.checkpoint_engine = writes.engine
        all_ok(writes.error)

        # Last, once every rank's files are in place: a resume loads the tag `latest` names, so
        # a save that fails or is cut short before this resumes from the previous checkpoint.
        error = None
        if is_main_process():
            try:
                write_atomic(save_root / 'latest', tag.encode())
            except Exception as e:
                error = e
        all_ok(error)

    def process_epoch(self, epoch, step, examples):
        checkpointed, saved = False, False
        if self.train_dataloader.epoch != epoch:
            # The model before the checkpoint: resuming from that checkpoint never skips a save.
            if 'save_every_n_epochs' in self.config and epoch % self.config['save_every_n_epochs'] == 0:
                self.save_model(f'epoch{epoch}')
                saved = True
            if need_to_checkpoint(self.config, epoch):
                self.save_checkpoint(step, examples)
                checkpointed = True
            if checkpointed or saved:
                # After both possible writes for this epoch, so a checkpoint and a model saved
                # on the same epoch are pruned in one pass.
                prune_all_checkpoint_kinds(self.config, self.save_root)
            epoch = self.train_dataloader.epoch
            if epoch > self.config['epochs']:
                return None, checkpointed, saved
            if is_main_process():
                print(f'Started new epoch: {epoch}')
        return epoch, checkpointed, saved

    def process_step(self, step, examples):
        checkpointed, saved = False, False
        # Look at some simple "signal files" the user can write to save and optionally quit manually
        should_manually_save, should_manually_quit = manual_save_signal(self.save_root)

        if 'save_every_n_steps' in self.config and step % self.config['save_every_n_steps'] == 0:
            self.save_model(f'step{step}')
            saved = True

        if need_to_checkpoint(self.config) or should_manually_save:
            self.save_checkpoint(step, examples)
            checkpointed = True

        if checkpointed or saved:
            prune_all_checkpoint_kinds(self.config, self.save_root)

        if should_manually_quit:
            print('Manually quitting')
            sys.exit()

        return checkpointed, saved
