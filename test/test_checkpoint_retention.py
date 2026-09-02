"""Retention deletes files, so its edges are worth pinning down.

`keep_last_n_checkpoints` is the only feature in this repo that removes data the user did not
ask it to remove. Every case below is one where getting it wrong costs a checkpoint: the newest
being pruned, one kind evicting another, or a resume state surviving without the weights it
belongs to.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.saver import prune_old_checkpoints, CHECKPOINT_PREFIXES  # noqa: E402


class TestTrainCheckpointRetention:
    """train.py's three kinds: epoch<N>/, step<N>/ and global_step<N>/."""

    @staticmethod
    def _populate(root, epochs=(), steps=(), global_steps=(), extra=()):
        for n in epochs:
            (root / f'epoch{n}').mkdir()
        for n in steps:
            (root / f'step{n}').mkdir()
        for n in global_steps:
            (root / f'global_step{n}').mkdir()
        for name in extra:
            (root / name).mkdir()

    def test_each_kind_is_counted_on_its_own(self, tmp_path):
        # The scenario the feature was asked for: four by epoch and five by step, keep 3, and
        # end with three of each -- not three in total.
        self._populate(tmp_path, epochs=(1, 2, 3, 4), steps=(100, 200, 300, 400, 500))
        prune_old_checkpoints(tmp_path, 'epoch', 3)
        prune_old_checkpoints(tmp_path, 'step', 3)
        left = sorted(p.name for p in tmp_path.iterdir())
        assert left == ['epoch2', 'epoch3', 'epoch4', 'step300', 'step400', 'step500']

    def test_the_newest_is_never_removed(self, tmp_path):
        # DeepSpeed's `latest` file names the global_step it resumes from. Pruning that one
        # turns a prunable cache into an unrecoverable run.
        self._populate(tmp_path, global_steps=(10, 20, 30))
        prune_old_checkpoints(tmp_path, 'global_step', 1)
        assert [p.name for p in tmp_path.iterdir()] == ['global_step30']

    def test_step_pruning_does_not_touch_global_step(self, tmp_path):
        # 'global_step100' does not start with 'step', so the two are distinct by construction.
        # If that ever stops being true, one kind starts evicting the other silently.
        self._populate(tmp_path, steps=(1, 2, 3, 4), global_steps=(1, 2, 3, 4))
        prune_old_checkpoints(tmp_path, 'step', 1)
        left = sorted(p.name for p in tmp_path.iterdir())
        assert left == ['global_step1', 'global_step2', 'global_step3', 'global_step4', 'step4']

    def test_ordering_is_by_number_not_by_name(self, tmp_path):
        # Lexicographic order would keep step9 over step100, which is backwards.
        self._populate(tmp_path, steps=(9, 100, 1000))
        prune_old_checkpoints(tmp_path, 'step', 1)
        assert [p.name for p in tmp_path.iterdir()] == ['step1000']

    def test_ordering_is_by_number_not_by_mtime(self, tmp_path):
        # A resumed run rewrites older directories' timestamps; the number is what says which
        # checkpoint is actually later.
        import os
        import time
        self._populate(tmp_path, epochs=(1, 2, 3))
        # Make the OLDEST look freshest.
        now = time.time()
        os.utime(tmp_path / 'epoch1', (now + 1000, now + 1000))
        prune_old_checkpoints(tmp_path, 'epoch', 1)
        assert [p.name for p in tmp_path.iterdir()] == ['epoch3']

    def test_unrelated_directories_are_left_alone(self, tmp_path):
        self._populate(tmp_path, epochs=(1, 2), extra=('samples', 'epochs_notes', 'epochX'))
        prune_old_checkpoints(tmp_path, 'epoch', 1)
        left = sorted(p.name for p in tmp_path.iterdir())
        assert left == ['epoch2', 'epochX', 'epochs_notes', 'samples'], left

    def test_a_file_is_not_mistaken_for_a_checkpoint(self, tmp_path):
        self._populate(tmp_path, epochs=(1, 2))
        (tmp_path / 'epoch99').write_text('a file, not a checkpoint directory', encoding='utf-8')
        prune_old_checkpoints(tmp_path, 'epoch', 1)
        assert (tmp_path / 'epoch99').is_file()
        assert (tmp_path / 'epoch2').is_dir()

    @pytest.mark.parametrize('keep', [None, 0])
    def test_unset_or_zero_keeps_everything(self, tmp_path, keep):
        # The default must be the behaviour this had before the feature existed.
        self._populate(tmp_path, epochs=(1, 2, 3))
        assert prune_old_checkpoints(tmp_path, 'epoch', keep) == []
        assert len(list(tmp_path.iterdir())) == 3

    def test_fewer_than_keep_removes_nothing(self, tmp_path):
        self._populate(tmp_path, epochs=(1, 2))
        assert prune_old_checkpoints(tmp_path, 'epoch', 5) == []

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert prune_old_checkpoints(tmp_path / 'nothing_here', 'epoch', 3) == []

    def test_dry_run_reports_without_deleting(self, tmp_path):
        self._populate(tmp_path, epochs=(1, 2, 3))
        doomed = prune_old_checkpoints(tmp_path, 'epoch', 1, dry_run=True)
        assert sorted(p.name for p in doomed) == ['epoch1', 'epoch2']
        assert len(list(tmp_path.iterdir())) == 3

    def test_all_three_kinds_are_covered(self):
        assert set(CHECKPOINT_PREFIXES) == {'epoch', 'step', 'global_step'}


class TestDistillCheckpointRetention:
    """Distillation tags each checkpoint and keeps its three files together."""

    @staticmethod
    def _populate(directory, kind, numbers, with_full_model=True):
        for n in numbers:
            (directory / f'context_refiner_{kind}{n}.safetensors').write_bytes(b'w')
            (directory / f'distill_state_{kind}{n}.pt').write_bytes(b's')
            if with_full_model:
                (directory / f'model_{kind}{n}.safetensors').write_bytes(b'm')

    def test_the_three_files_of_a_tag_go_together(self, tmp_path):
        # A surviving tag must be complete: weights without their optimizer state cannot resume,
        # and a refiner without the model it belongs in is half a checkpoint.
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 2, 3, 4))
        removed = sorted(p.name for p in prune_distill_checkpoints(tmp_path, 3))
        assert removed == ['context_refiner_epoch1.safetensors',
                           'distill_state_epoch1.pt',
                           'model_epoch1.safetensors']

    def test_a_second_run_prunes_the_old_checkpoints_not_its_own(self, tmp_path):
        """Tags only increase within one run, so the newest number is not the newest file.

        Protecting just the tag being written is not enough over a sequence of saves: the second
        run writes epoch1 (protected, nothing goes), then epoch2 -- at which point epoch1 is the
        lowest number present and is deleted. The run would finish holding the previous run's
        three checkpoints and one of its own.
        """
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (18, 19, 20), with_full_model=False)

        written = set()
        for n in (1, 2, 3):
            self._populate(tmp_path, 'epoch', (n,), with_full_model=False)
            written.add(f'_epoch{n}')
            prune_distill_checkpoints(tmp_path, 3, protect_tag=f'_epoch{n}',
                                      protect_tags=written)

        survivors = sorted(p.name for p in tmp_path.glob('context_refiner_epoch*.safetensors'))
        assert survivors == ['context_refiner_epoch1.safetensors',
                             'context_refiner_epoch2.safetensors',
                             'context_refiner_epoch3.safetensors'], (
            f'this run\'s own checkpoints were pruned in favour of the previous run\'s: {survivors}'
        )

    def test_one_long_run_still_drops_its_own_early_checkpoints(self, tmp_path):
        """protect_tags orders the candidates; it must not immunise them.

        Every tag in a single run is one this process wrote, so a rule that skipped anything it
        had written would prune nothing at all and the directory would grow without bound --
        which is the opposite of what keep_last_n_checkpoints is for.
        """
        from tools.distill_refiner import prune_distill_checkpoints
        written = set()
        for n in range(1, 8):
            self._populate(tmp_path, 'epoch', (n,), with_full_model=False)
            written.add(f'_epoch{n}')
            prune_distill_checkpoints(tmp_path, 3, protect_tag=f'_epoch{n}',
                                      protect_tags=written)

        survivors = sorted(int(p.stem[len('context_refiner_epoch'):])
                           for p in tmp_path.glob('context_refiner_epoch*.safetensors'))
        assert survivors == [5, 6, 7], f'keep=3 should leave the newest three, got {survivors}'

    def test_epoch_and_step_tags_are_counted_apart(self, tmp_path):
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 2, 3, 4), with_full_model=False)
        self._populate(tmp_path, 'step', (100, 200, 300, 400, 500), with_full_model=False)
        prune_distill_checkpoints(tmp_path, 3)
        weights = sorted(p.name for p in tmp_path.glob('context_refiner_*.safetensors'))
        assert weights == ['context_refiner_epoch2.safetensors',
                           'context_refiner_epoch3.safetensors',
                           'context_refiner_epoch4.safetensors',
                           'context_refiner_step300.safetensors',
                           'context_refiner_step400.safetensors',
                           'context_refiner_step500.safetensors']

    def test_the_untagged_names_are_never_pruned(self, tmp_path):
        # They are the stable names every config points at.
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 2, 3, 4), with_full_model=False)
        (tmp_path / 'context_refiner.safetensors').write_bytes(b'w')
        (tmp_path / 'distill_state.pt').write_bytes(b's')
        (tmp_path / 'model.safetensors').write_bytes(b'm')
        prune_distill_checkpoints(tmp_path, 1)
        for name in ('context_refiner.safetensors', 'distill_state.pt', 'model.safetensors'):
            assert (tmp_path / name).exists(), name

    def test_a_missing_companion_is_not_an_error(self, tmp_path):
        # save_full_model may be off, so model_*.safetensors need not exist.
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 2), with_full_model=False)
        removed = sorted(p.name for p in prune_distill_checkpoints(tmp_path, 1))
        assert removed == ['context_refiner_epoch1.safetensors', 'distill_state_epoch1.pt']

    def test_unset_keeps_everything(self, tmp_path):
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 2, 3))
        assert prune_distill_checkpoints(tmp_path, None) == []

    def test_the_state_file_pairs_with_its_own_weights(self, tmp_path):
        # Fixed naming would pair the newest optimizer moments with older weights on a resume
        # from an older tag.
        from tools.distill_refiner import training_state_path
        assert training_state_path(tmp_path / 'context_refiner_epoch7.safetensors').name == \
            'distill_state_epoch7.pt'
        assert training_state_path(tmp_path / 'context_refiner_step900.safetensors').name == \
            'distill_state_step900.pt'
        assert training_state_path(tmp_path / 'context_refiner.safetensors').name == \
            'distill_state.pt'


class TestDistillRetentionWithShardedState:
    """A ZeRO run's tag owns one state file per rank, not one in total."""

    @staticmethod
    def _populate(directory, kind, numbers, ranks):
        for n in numbers:
            (directory / f'context_refiner_{kind}{n}.safetensors').write_bytes(b'w')
            for r in range(ranks):
                (directory / f'distill_state_{kind}{n}_rank{r}.pt').write_bytes(b's')

    def test_every_rank_shard_of_a_pruned_tag_goes_with_it(self, tmp_path):
        # Leaving shards behind would grow the output directory without bound, which is the one
        # thing keep_last_n_checkpoints exists to stop.
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 2, 3), ranks=4)
        prune_distill_checkpoints(tmp_path, 2)
        assert sorted(p.name for p in tmp_path.glob('distill_state_epoch1*')) == []
        assert len(list(tmp_path.glob('distill_state_epoch3_rank*.pt'))) == 4

    def test_pruning_epoch1_does_not_delete_epoch10(self, tmp_path):
        # The reason the shard glob keys on the '_rank' separator: 'distill_state_epoch1*' also
        # matches distill_state_epoch10.pt, so the obvious glob deletes a checkpoint nine
        # epochs newer than the one being pruned.
        from tools.distill_refiner import prune_distill_checkpoints
        self._populate(tmp_path, 'epoch', (1, 10, 11), ranks=2)
        prune_distill_checkpoints(tmp_path, 2)
        assert not list(tmp_path.glob('distill_state_epoch1_rank*.pt'))
        assert len(list(tmp_path.glob('distill_state_epoch10_rank*.pt'))) == 2
        assert len(list(tmp_path.glob('distill_state_epoch11_rank*.pt'))) == 2
