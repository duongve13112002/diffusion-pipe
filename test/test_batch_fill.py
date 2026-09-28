"""Tests for batch_fill_strategy: filling a short final batch instead of dropping it.

Three things are being protected here, in descending order of importance.

The first is that nothing changes by default. `drop` has to produce the same iteration order it
produced before any of this existed, because every run anyone has done so far trained on those
samples in that order, and a config that says nothing must keep meaning what it meant.

The second is the invariant the feature exists for: no two real samples in one global batch may
come from the same image. That has to hold across gradient accumulation and across data
parallel ranks, since a "global batch" spans both -- which is why the multi-rank cases below
reassemble the ranks' slices and check the union rather than checking one rank.

The third is the loss arithmetic. A padded sample is masked to zero, and `.mean()` divides by
the padded count, so the real samples carry G/G_real to compensate. That constant is computed
over the whole global batch and shared by every micro batch and every rank, because both of
those average -- and a test that only ever looks at one micro batch cannot tell the difference
between a constant that is right and one that happens to work when the padding is spread
evenly. TestNormalisationSurvivesMicroBatching drives that case directly.
"""

import math

import numpy as np
import pytest
import torch

pytest.importorskip('datasets')
import datasets  # noqa: E402

import utils.common  # noqa: E402
from utils import dataset as dataset_util  # noqa: E402


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    """No process group here, and is_main_process only decides whether a warning prints."""
    monkeypatch.setattr(utils.common, 'is_main_process', lambda: True)
    monkeypatch.setattr(dataset_util, 'is_main_process', lambda: True)


class FakeSizeBucket:
    """The parts of SizeBucketDataset that ConcatenatedBatchedDataset reaches for.

    A real one needs a cached latent dataset on disk, which is covered by
    test_dataset_smoke.py. What batch fill actually reads is narrow: the size bucket, the
    iteration_order's latents_idx column, and __len__ (which is where num_repeats enters).
    """

    def __init__(self, num_images, captions_per_image=1, num_repeats=1,
                 size_bucket=(64, 64, 1), tag=''):
        self.size_bucket = size_bucket
        self.num_repeats = num_repeats
        latents = []
        for _ in range(captions_per_image):
            latents.extend(range(num_images))
        self.iteration_order = datasets.Dataset.from_dict({'latents_idx': latents})
        self.tag = tag

    def __len__(self):
        return int(len(self.iteration_order) * self.num_repeats)

    def __getitem__(self, idx):
        row = idx % len(self.iteration_order)
        return {'latents_idx': row, 'mask': None, 'caption': f'{self.tag}{row}'}


def build_bucket(sub_datasets, global_batch, world_size=1, rank=0, **fill):
    bucket = dataset_util.ConcatenatedBatchedDataset(sub_datasets)
    bucket.post_init({None: global_batch}, {None: global_batch}, rank, world_size,
                     batch_fill=dataset_util.resolve_batch_fill_config(fill))
    return bucket


def real_image_ids(bucket, start, size):
    return [bucket._image_id(bucket.iteration_order[k])
            for k in range(start, start + size) if bucket.sample_weights[k] > 0]


def real_row_ids(bucket, start, size):
    return [bucket._row_id(bucket.iteration_order[k])
            for k in range(start, start + size) if bucket.sample_weights[k] > 0]


def duplicate_image_batches(bucket, global_batch):
    """Global batches holding two real samples of the same image, and whether rows differ.

    A tier-2 fill puts the same image in twice under a different caption, which is allowed, so
    the row check is what separates that from a plain repeat.
    """
    offending = []
    for start in range(0, len(bucket.iteration_order), global_batch):
        images = real_image_ids(bucket, start, global_batch)
        rows = real_row_ids(bucket, start, global_batch)
        if len(set(images)) != len(images):
            offending.append((start, len(set(rows)) == len(rows)))
    return offending


class TestDropIsUnchanged:
    """The regression guard. Everything else is worth less than this one."""

    @pytest.mark.parametrize('num_images,global_batch', [(100, 64), (128, 64), (50, 64), (7, 3)])
    def test_default_matches_explicit_drop(self, num_images, global_batch):
        default = build_bucket([FakeSizeBucket(num_images)], global_batch)
        explicit = build_bucket([FakeSizeBucket(num_images)], global_batch,
                                batch_fill_strategy='drop')
        assert np.array_equal(default.iteration_order, explicit.iteration_order)

    @pytest.mark.parametrize('num_images,global_batch', [(100, 64), (128, 64), (7, 3)])
    def test_drop_truncates_to_a_whole_number_of_batches(self, num_images, global_batch):
        bucket = build_bucket([FakeSizeBucket(num_images)], global_batch)
        assert len(bucket.iteration_order) == (num_images // global_batch) * global_batch
        assert (bucket.sample_weights == 1.0).all()

    def test_a_bucket_smaller_than_one_batch_is_still_dropped(self):
        bucket = build_bucket([FakeSizeBucket(50)], 64)
        assert len(bucket) == 0

    def test_drop_attaches_no_weight_to_the_examples(self):
        bucket = build_bucket([FakeSizeBucket(100)], 64)
        for example in bucket[0]:
            assert dataset_util.SAMPLE_WEIGHT_KEY not in example


class TestFillCompletesTheLastBatch:
    def test_length_becomes_a_multiple_of_the_global_batch(self):
        bucket = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        assert len(bucket.iteration_order) == 128
        assert len(bucket) == 2

    def test_the_static_part_is_exactly_what_drop_would_have_kept(self):
        """Only the tail moves. Everything before it is the order 'drop' already produced."""
        filled = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        assert np.array_equal(filled.iteration_order[:100], filled._static_iteration_order)

    def test_no_sample_is_lost(self):
        filled = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        static = {tuple(row) for row in filled._static_iteration_order}
        assert static <= {tuple(row) for row in filled.iteration_order}

    def test_fill_is_a_no_op_when_the_count_already_divides(self):
        filled = build_bucket([FakeSizeBucket(128)], 64, batch_fill_strategy='fill')
        dropped = build_bucket([FakeSizeBucket(128)], 64)
        assert np.array_equal(filled.iteration_order, dropped.iteration_order)
        assert (filled.sample_weights == 1.0).all()

    def test_a_bucket_with_enough_images_needs_no_masking(self):
        filled = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        assert (filled.sample_weights > 0).all()
        assert filled.fill_report['num_added'] == 28
        assert filled.fill_report['num_masked'] == 0

    def test_no_repeated_image_among_the_real_samples(self):
        filled = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        assert duplicate_image_batches(filled, 64) == []


class TestTierOrder:
    """Tier 1 is a new image, tier 2 the same image with a different caption, tier 3 a repeat."""

    def test_tier_two_is_used_before_masking(self):
        # 10 images x 2 captions = 20 rows, global batch 16, so 12 slots are needed and only
        # 10 distinct images exist. Tier 2 has to supply the rest, unmasked.
        bucket = build_bucket([FakeSizeBucket(10, captions_per_image=2)], 16,
                              batch_fill_strategy='fill', min_real_fraction=0.1)
        assert bucket.fill_report['num_added'] == 12
        assert bucket.fill_report['num_masked'] == 0
        # Duplicated images are allowed here, but every one of them must be a different row,
        # which is what makes it a new (image, caption) pair rather than a repeat.
        for _, rows_distinct in duplicate_image_batches(bucket, 16):
            assert rows_distinct

    def test_num_repeats_copies_are_repeats_not_tier_two(self):
        """A num_repeats copy is the same row, so it must be masked rather than counted real."""
        bucket = build_bucket([FakeSizeBucket(10, num_repeats=3)], 16,
                              batch_fill_strategy='fill', min_real_fraction=0.1)
        # 30 rows, batch 16, so 2 slots needed; all 10 rows are already in the final batch.
        assert bucket.fill_report['num_added'] == 2
        assert bucket.fill_report['num_masked'] == 2

    def test_image_identity_includes_the_sub_dataset(self):
        """Two directories in one bucket both number their images from zero."""
        bucket = build_bucket([FakeSizeBucket(5, tag='A'), FakeSizeBucket(5, tag='B')], 8,
                              batch_fill_strategy='fill', min_real_fraction=0.1)
        assert bucket.fill_report['num_masked'] == 0
        assert duplicate_image_batches(bucket, 8) == []
        ids = real_image_ids(bucket, len(bucket.iteration_order) - 8, 8)
        assert len({d for d, _ in ids}) == 2, 'both sub-datasets should be represented'


class TestUndersizedBucket:
    def test_pad_masked_produces_one_batch(self):
        bucket = build_bucket([FakeSizeBucket(3)], 16, batch_fill_strategy='fill',
                              min_real_fraction=0.1)
        assert len(bucket) == 1
        assert int((bucket.sample_weights > 0).sum()) == 3
        assert int((bucket.sample_weights == 0).sum()) == 13

    def test_the_real_samples_carry_the_compensating_scale(self):
        bucket = build_bucket([FakeSizeBucket(3)], 16, batch_fill_strategy='fill',
                              min_real_fraction=0.1)
        assert bucket._batch_weight_scale(0) == pytest.approx(16 / 3)
        weights = [e[dataset_util.SAMPLE_WEIGHT_KEY] for e in bucket[0]]
        assert sorted(set(round(w, 6) for w in weights)) == [0.0, round(16 / 3, 6)]

    def test_drop_keeps_the_old_behaviour_for_this_case_only(self):
        bucket = build_bucket([FakeSizeBucket(3)], 16, batch_fill_strategy='fill',
                              undersized_bucket='drop')
        assert len(bucket) == 0

    def test_min_real_fraction_drops_a_bucket_that_is_mostly_padding(self):
        bucket = build_bucket([FakeSizeBucket(3)], 16, batch_fill_strategy='fill')
        assert len(bucket) == 0, '3/16 = 0.19 is below the default 0.25'

    def test_min_real_fraction_keeps_a_bucket_above_the_threshold(self):
        bucket = build_bucket([FakeSizeBucket(5)], 16, batch_fill_strategy='fill')
        assert len(bucket) == 1, '5/16 = 0.31 is above the default 0.25'


class TestEpochRotation:
    def test_the_tail_moves_and_the_static_part_does_not(self):
        bucket = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        first = bucket.iteration_order.copy()
        bucket.set_epoch(1)
        second = bucket.iteration_order.copy()
        assert len(first) == len(second)
        assert np.array_equal(first[:100], second[:100])
        assert not np.array_equal(first[100:], second[100:])

    def test_an_epoch_is_reproducible_from_its_number(self):
        """Resume depends on this: the order is a function of (seed, epoch), nothing else."""
        bucket = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        epoch_zero = bucket.iteration_order.copy()
        bucket.set_epoch(1)
        bucket.set_epoch(2)
        bucket.set_epoch(0)
        assert np.array_equal(bucket.iteration_order, epoch_zero)

    def test_the_invariant_holds_in_every_epoch(self):
        bucket = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill')
        for epoch in range(4):
            bucket.set_epoch(epoch)
            assert duplicate_image_batches(bucket, 64) == [], f'epoch {epoch}'

    def test_rotation_can_be_turned_off(self):
        bucket = build_bucket([FakeSizeBucket(100)], 64, batch_fill_strategy='fill',
                              fill_rotate_per_epoch=False)
        first = bucket.iteration_order.copy()
        bucket.set_epoch(5)
        assert np.array_equal(bucket.iteration_order, first)

    def test_drop_ignores_set_epoch_entirely(self):
        bucket = build_bucket([FakeSizeBucket(100)], 64)
        first = bucket.iteration_order.copy()
        bucket.set_epoch(3)
        assert np.array_equal(bucket.iteration_order, first)


class TestEveryRankAgrees:
    """post_init runs independently per rank and they synchronise only by computing the same
    thing. A rank that filled its tail differently would put different images in one batch and
    the duplicate check would pass on each rank while failing globally."""

    @pytest.mark.parametrize('world_size,grad_accum', [(1, 1), (1, 4), (2, 2), (4, 2)])
    def test_ranks_produce_the_same_global_order(self, world_size, grad_accum):
        global_batch = 8 * grad_accum * world_size
        buckets = [build_bucket([FakeSizeBucket(100)], global_batch, world_size=world_size,
                                rank=r, batch_fill_strategy='fill', min_real_fraction=0.1)
                   for r in range(world_size)]
        for other in buckets[1:]:
            assert np.array_equal(buckets[0].iteration_order, other.iteration_order)
            assert np.array_equal(buckets[0].sample_weights, other.sample_weights)

    @pytest.mark.parametrize('world_size,grad_accum', [(1, 1), (1, 4), (2, 2), (4, 2)])
    def test_the_ranks_slices_reassemble_into_the_global_batch(self, world_size, grad_accum):
        global_batch = 8 * grad_accum * world_size
        buckets = [build_bucket([FakeSizeBucket(100)], global_batch, world_size=world_size,
                                rank=r, batch_fill_strategy='fill', min_real_fraction=0.1)
                   for r in range(world_size)]
        for idx in range(len(buckets[0])):
            rows = []
            for bucket in buckets:
                rows.extend(e['latents_idx'] for e in bucket[idx])
            assert len(rows) == global_batch
            expected = [int(bucket.iteration_order[idx * global_batch + k][1])
                        for k in range(global_batch)
                        for bucket in [buckets[0]]]
            assert sorted(rows) == sorted(r % 100 for r in expected)

    @pytest.mark.parametrize('world_size,grad_accum', [(2, 2), (4, 2)])
    def test_no_image_repeats_across_ranks_within_a_batch(self, world_size, grad_accum):
        global_batch = 8 * grad_accum * world_size
        bucket = build_bucket([FakeSizeBucket(100)], global_batch, world_size=world_size,
                              rank=0, batch_fill_strategy='fill', min_real_fraction=0.1)
        assert duplicate_image_batches(bucket, global_batch) == []


class TestConfigValidation:
    @pytest.mark.parametrize('key,value', [
        ('batch_fill_strategy', 'fil'),
        ('batch_fill_strategy', True),
        ('undersized_bucket', 'mask'),
        ('fill_rotate_per_epoch', 'yes'),
        ('min_real_fraction', 1.5),
        ('min_real_fraction', -0.1),
        ('min_real_fraction', 'a quarter'),
    ])
    def test_a_bad_value_raises_rather_than_falling_back(self, key, value):
        with pytest.raises(ValueError, match=key):
            dataset_util.resolve_batch_fill_config({key: value})

    def test_defaults_are_the_old_behaviour(self):
        resolved = dataset_util.resolve_batch_fill_config({})
        assert resolved['batch_fill_strategy'] == 'drop'
        assert resolved['undersized_bucket'] == 'pad_masked'
        assert resolved['min_real_fraction'] == 0.25

    def test_eval_can_override_a_default_without_restating_the_table(self):
        resolved = dataset_util.resolve_batch_fill_config(
            {'batch_fill_strategy': 'fill'}, defaults={'fill_rotate_per_epoch': False})
        assert resolved['batch_fill_strategy'] == 'fill'
        assert resolved['fill_rotate_per_epoch'] is False

    def test_the_config_still_wins_over_an_overridden_default(self):
        resolved = dataset_util.resolve_batch_fill_config(
            {'fill_rotate_per_epoch': True}, defaults={'fill_rotate_per_epoch': False})
        assert resolved['fill_rotate_per_epoch'] is True

    def test_the_training_config_overrides_the_dataset_config(self):
        """Several training configs share one dataset TOML, so this layer has to exist."""
        resolved = dataset_util.resolve_batch_fill_config(
            {'batch_fill_strategy': 'drop'}, overrides={'batch_fill_strategy': 'fill'})
        assert resolved['batch_fill_strategy'] == 'fill'

    def test_an_override_is_validated_like_anything_else(self):
        with pytest.raises(ValueError, match='batch_fill_strategy'):
            dataset_util.resolve_batch_fill_config({}, overrides={'batch_fill_strategy': 'nope'})

    def test_unrelated_training_config_keys_are_ignored(self):
        resolved = dataset_util.resolve_batch_fill_config(
            {}, overrides={'epochs': 100, 'batch_fill_strategy': 'fill'})
        assert resolved['batch_fill_strategy'] == 'fill'
        assert 'epochs' not in resolved

    def test_rotation_with_drop_warns(self, capsys, monkeypatch):
        warned = []
        monkeypatch.setattr(dataset_util.logger, 'warning', lambda msg: warned.append(msg))
        dataset_util.resolve_batch_fill_config({'fill_rotate_per_epoch': True})
        assert warned and 'no effect' in warned[0]


class TestCollateCarriesTheWeightIntoTheMask:
    @staticmethod
    def collate(weights, masks=None, mask_shape=(64, 64)):
        examples = []
        for i, weight in enumerate(weights):
            example = {'latents': torch.zeros(2, 2), 'mask': None if masks is None else masks[i]}
            if weight is not None:
                example[dataset_util.SAMPLE_WEIGHT_KEY] = weight
            examples.append(example)
        stub = object.__new__(dataset_util.Dataset)
        return dataset_util.Dataset._collate(stub, examples, mask_shape=mask_shape)

    def test_an_unweighted_batch_keeps_mask_none(self):
        """The old path, byte for byte: no mask tensor is invented where there was none."""
        assert self.collate([None, None])['mask'] is None
        assert self.collate([1.0, 1.0])['mask'] is None

    def test_the_weight_key_never_reaches_the_model(self):
        batch = self.collate([2.0, 0.0])
        assert dataset_util.SAMPLE_WEIGHT_KEY not in batch

    def test_a_padded_batch_gets_a_mask_carrying_the_weights(self):
        batch = self.collate([2.0, 2.0, 0.0])
        mask = batch['mask']
        assert mask.shape == (3, 64, 64)
        assert mask[0].unique().tolist() == [2.0]
        assert mask[2].unique().tolist() == [0.0]

    def test_a_real_mask_is_scaled_rather_than_replaced(self):
        real = torch.full((8, 8), 0.5, dtype=torch.float16)
        batch = self.collate([2.0, 0.0], masks=[real, None])
        assert batch['mask'][0].unique().tolist() == [1.0]
        assert batch['mask'][1].unique().tolist() == [0.0]

    def test_a_real_masks_shape_wins_over_the_fallback(self):
        real = torch.ones((8, 8), dtype=torch.float16)
        batch = self.collate([2.0, 0.0], masks=[real, None], mask_shape=(64, 64))
        assert batch['mask'].shape == (2, 8, 8)


def masked_label(num_real, num_pad, ndim5=True, spatial=8):
    """A (target, mask) label shaped the way prepare_inputs hands one to a loss function."""
    total = num_real + num_pad
    weight = total / num_real
    shape = (total, 1, 1, spatial, spatial) if ndim5 else (total, 1, spatial, spatial)
    mask = torch.zeros(shape)
    mask[:num_real] = weight
    return mask, weight


def padded_batch(num_real, num_pad, channels=3, ndim5=True, spatial=8, seed=0):
    """Predictions and targets where the padding is a copy of a real sample, as fill produces."""
    torch.manual_seed(seed)
    total = num_real + num_pad
    shape = (total, channels, 1, spatial, spatial) if ndim5 else (total, channels, spatial, spatial)
    target = torch.randn(shape)
    output = target + 0.4 * torch.randn(shape)
    for k in range(num_real, total):
        output[k], target[k] = output[k % num_real], target[k % num_real]
    return output.requires_grad_(True), target


class TestEveryLossFunctionIgnoresPadding:
    """All eight get_loss_fn implementations, not just the one this feature was written for.

    A padded sample must contribute exactly zero gradient, and the loss must equal what the
    real samples alone would have produced. Two of these needed fixing to get there: the
    multiscale branch of cosmos_predict2 added an unmasked term at each reduced scale, and
    minimax_h3's audio branch was never masked at all. Neither shows up in the loss value --
    both were within 0.3% of correct -- so the gradient is what is asserted.
    """

    CASES = [
        ('models.base', 'BasePipeline', {}, {}),
        ('models.base', 'ComfyPipeline', {}, {}),
        ('models.cosmos_predict2', 'CosmosPredict2Pipeline', {'multiscale_loss_weight': None}, {}),
        # 128 latent gives side_length 1024, above the 921.6 threshold, so the multiscale
        # branch actually runs. At the default 8 it does not, and this case would pass without
        # ever reaching the code it exists to cover.
        ('models.cosmos_predict2', 'CosmosPredict2Pipeline', {'multiscale_loss_weight': 0.5},
         {'spatial': 128}),
        ('models.cosmos', 'CosmosPipeline', {}, {'weights_per_sigma': True}),
        ('models.sdxl', 'SDXLPipeline',
         {'min_snr_gamma': None, 'debiased_estimation_loss': None, 'v_pred': False,
          'scheduler': None},
         {'ndim5': False, 'timesteps': True}),
        ('models.ltx_video', 'LTXVideoPipeline', {}, {}),
        ('models.ltx2', 'LTX2Pipeline', {}, {}),
        ('models.minimax_h3', 'MinimaxH3Pipeline', {}, {'audio': True}),
    ]

    @pytest.mark.parametrize('module,cls_name,attrs,shape_kwargs', CASES,
                             ids=[f'{c[1]}{"-ms" if c[2].get("multiscale_loss_weight") else ""}'
                                  for c in CASES])
    def test_padding_contributes_no_gradient(self, module, cls_name, attrs, shape_kwargs):
        pytest.importorskip(module)
        import importlib
        cls = getattr(importlib.import_module(module), cls_name)

        stub = object.__new__(cls)
        stub.config = {}
        for key, value in attrs.items():
            setattr(stub, key, value)
        loss_fn = cls.get_loss_fn(stub)

        ndim5 = shape_kwargs.get('ndim5', True)
        spatial = shape_kwargs.get('spatial', 8)
        num_real, num_pad = 4, 2

        def evaluate(pad):
            # One draw, sliced. Regenerating at the smaller size would consume the RNG stream
            # differently and the two batches would not hold the same real samples, which
            # showed up as a 1.3% mismatch that looked like a masking bug and was not one.
            output, target = padded_batch(num_real, num_pad, ndim5=ndim5, spatial=spatial)
            real = num_real
            if pad:
                mask, _ = masked_label(num_real, num_pad, ndim5=ndim5, spatial=spatial)
            else:
                output = output.detach()[:num_real].requires_grad_(True)
                target = target[:num_real]
                mask = torch.ones((num_real, 1, 1, spatial, spatial) if ndim5
                                  else (num_real, 1, spatial, spatial))
            pad = num_pad if pad else 0
            wrapped = output
            if shape_kwargs.get('weights_per_sigma'):
                wrapped = (output, torch.full((real + pad, 1, 1, 1, 1), 0.7))
            if shape_kwargs.get('timesteps'):
                wrapped = (output, torch.zeros(real + pad))
            if shape_kwargs.get('audio'):
                # The audio prediction has to be a function of `output`, or the audio branch
                # contributes nothing to output.grad and this test cannot fail whatever the
                # loss does with it.
                audio = output.flatten(1)[:, :64].reshape(real + pad, 2, 32) * 0.5
                wrapped = (output, audio)
                torch.manual_seed(5)
                label = (target, torch.randn(real + pad, 2, 32), mask)
            else:
                label = (target, mask)
            return loss_fn(wrapped, label), output

        loss, output = evaluate(pad=True)
        loss.backward()
        pad_gradient = output.grad.flatten(1).norm(dim=1)[num_real:].sum().item()
        assert pad_gradient == 0.0, f'{cls_name} lets padding reach the optimizer'

        reference, _ = evaluate(pad=False)
        assert loss.item() == pytest.approx(reference.item(), rel=1e-3)


class TestNormalisationSurvivesMicroBatching:
    """The G/G_real constant must be global, not per micro batch.

    Deepspeed averages the micro batch losses, so a batch split into micro batches has to give
    the same answer as the unsplit one -- including when every padded sample lands in the same
    micro batch, which is the case a per-micro-batch ratio gets wrong and an evenly spread test
    never notices.
    """

    @staticmethod
    def loss_of(output, target, mask):
        loss = torch.nn.functional.mse_loss(output, target, reduction='none')
        return (loss * mask).mean()

    @pytest.mark.parametrize('num_micro_batches', [1, 2, 4])
    def test_the_average_over_micro_batches_is_the_mean_over_real_samples(self, num_micro_batches):
        num_real, num_pad = 6, 2
        output, target = padded_batch(num_real, num_pad)
        mask, _ = masked_label(num_real, num_pad)

        whole = self.loss_of(output, target, mask)
        pieces = [self.loss_of(o, t, m) for o, t, m in
                  zip(output.chunk(num_micro_batches), target.chunk(num_micro_batches),
                      mask.chunk(num_micro_batches))]
        averaged = sum(pieces) / num_micro_batches
        assert averaged.item() == pytest.approx(whole.item(), rel=1e-5)

        real_only = torch.nn.functional.mse_loss(output[:num_real], target[:num_real])
        assert whole.item() == pytest.approx(real_only.item(), rel=1e-5)

    def test_a_micro_batch_that_is_entirely_padding_is_finite(self):
        """Legal, and it must give zero rather than a division by zero."""
        output, target = padded_batch(4, 4)
        mask, _ = masked_label(4, 4)
        pieces = [self.loss_of(o, t, m) for o, t, m in
                  zip(output.chunk(2), target.chunk(2), mask.chunk(2))]
        assert pieces[1].item() == 0.0
        assert torch.isfinite(torch.stack(pieces)).all()

    def test_the_padding_may_be_distributed_unevenly(self):
        """Two micro batches, all the padding in one. The global constant handles it."""
        num_real, num_pad = 6, 2
        output, target = padded_batch(num_real, num_pad)
        mask, _ = masked_label(num_real, num_pad)
        # chunk(2) puts samples 0-3 in the first piece and 4-7 in the second, so the second
        # holds both padded samples and only two real ones.
        pieces = [self.loss_of(o, t, m) for o, t, m in
                  zip(output.chunk(2), target.chunk(2), mask.chunk(2))]
        averaged = sum(pieces) / 2
        real_only = torch.nn.functional.mse_loss(output[:num_real], target[:num_real])
        assert averaged.item() == pytest.approx(real_only.item(), rel=1e-5)
