"""text_embedding_dtype (utils/embedding_dtype.py) and max_cached_captions."""

import pytest
import torch

import utils.dataset as dataset_util
from utils.embedding_dtype import FLOAT8, SCALE, check_text_embedding_dtype, decode_row, encode_results
from test.test_text_encoding_on_the_fly import (  # noqa: F401  (local_workers is an autouse fixture)
    Model, batches, dataset_config, local_workers, make_images)


def rows(results, i):
    return {k: v[i] for k, v in results.items()}


class TestCodec:
    def test_round_trip_is_close_and_keeps_the_dtype(self):
        torch.manual_seed(0)
        x = torch.randn(4, 9, 32, dtype=torch.bfloat16)
        x[1, 3, 7] = 2500
        encoded = encode_results({'e': x}, FLOAT8)
        assert encoded['e'].dtype == torch.float8_e4m3fn
        for i in range(4):
            back = decode_row(rows(encoded, i))['e']
            assert back.dtype == torch.bfloat16 and back.shape == x[i].shape
            error = (back.float() - x[i].float()).abs() / x[i].float().abs().clamp_min(1e-2)
            assert error.max() < 0.07
        # The outlier is scaled, not clipped at e4m3's 448.
        assert abs(decode_row(rows(encoded, 1))['e'][3, 7].item() - 2500) / 2500 < 0.07

    def test_masks_ids_and_zero_rows_come_back_exactly(self):
        mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
        ids = torch.tensor([[5, 6, 7], [8, 9, 10]])
        zeros = torch.zeros(2, 3, 4)
        encoded = encode_results({'mask': mask, 'ids': ids, 'zeros': zeros}, FLOAT8)
        assert encoded['ids'] is ids and 'ids' + SCALE not in encoded
        for i in range(2):
            back = decode_row(rows(encoded, i))
            assert sorted(back) == ['ids', 'mask', 'zeros']
            assert torch.equal(back['mask'], mask[i]) and torch.equal(back['ids'], ids[i])
            assert torch.equal(back['zeros'], zeros[i])

    def test_lists_of_per_row_tensors(self):
        rows_in = [torch.randn(3, 8), torch.randn(5, 8)]
        encoded = encode_results({'e': rows_in, 'name': ['a', 'b']}, FLOAT8)
        assert encoded['name'] == ['a', 'b']
        for i in range(2):
            back = decode_row(rows(encoded, i))
            assert back['e'].shape == rows_in[i].shape and back['e'].dtype == torch.float32
            assert set(back) == {'e', 'name'}

    def test_a_row_stored_as_returned_is_returned_as_is(self):
        row = {'e': torch.ones(2)}
        assert decode_row(row) is row

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
    def test_a_cast_dtype_is_stored_and_read_back_in_the_original(self, dtype):
        x = torch.randn(2, 5, 8, dtype=torch.bfloat16)
        ids = torch.tensor([[1, 2], [3, 4]])
        encoded = encode_results({'e': x, 'ids': ids}, dtype)
        assert encoded['e'].dtype == dtype and 'e' + SCALE not in encoded and encoded['ids'] is ids
        for i in range(2):
            back = decode_row(rows(encoded, i))
            assert set(back) == {'e', 'ids'} and back['e'].dtype == torch.bfloat16
            # bfloat16 holds each value in all three, so nothing is rounded.
            assert torch.equal(back['e'], x[i])

    def test_values_float16_cannot_hold_are_refused(self):
        x = torch.full((1, 2, 3), 1e5)
        with pytest.raises(ValueError, match="does not fit in torch.float16"):
            encode_results({'e': x}, torch.float16)

    def test_config(self):
        assert check_text_embedding_dtype({}, True) is None
        for name, dtype in (('float8', FLOAT8), ('bfloat16', torch.bfloat16), ('float16', torch.float16),
                            ('float32', torch.float32)):
            assert check_text_embedding_dtype({'text_embedding_dtype': name}, True) == dtype
        with pytest.raises(ValueError, match='must be one of float8, bfloat16, float16, float32'):
            check_text_embedding_dtype({'text_embedding_dtype': 'fp8'}, True)
        with pytest.raises(ValueError, match='cache_text_embeddings = false'):
            check_text_embedding_dtype({'text_embedding_dtype': 'bfloat16'}, False)


class Float8Model(Model):
    def __init__(self, embedding_dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.model_config = {} if embedding_dtype is None else {'text_embedding_dtype': embedding_dtype}


@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
def test_cached_float8_embeddings_train_like_the_plain_ones(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', ['a red bus', 'two dogs', 'x y', 'sunset over water'])
    config = dataset_config(images, reuse_metadata_cache=reuse_metadata_cache)
    plain, _ = batches(Float8Model(), config)
    quantized, _ = batches(Float8Model('float8'), config)
    assert len(plain) == len(quantized) > 0
    for a, b in zip(plain, quantized):
        assert sorted(a) == sorted(b)
        for key in ('emb1', 'emb2'):
            xs, ys = (a[key], b[key]) if isinstance(a[key], list) else ([a[key]], [b[key]])
            for x, y in zip(xs, ys):
                assert y.dtype == x.dtype and y.shape == x.shape
                assert not torch.equal(x.float(), y.float()) or key == 'emb2'
                assert ((y.float() - x.float()).abs() <= x.float().abs() * 0.07 + 1e-6).all(), key
        assert torch.equal(a['len1'], b['len1'])
    # float32 holds every bfloat16 value: the batch is the plain one exactly.
    for a, b in zip(plain, batches(Float8Model('float32'), config)[0]):
        for key in ('emb1', 'emb2'):
            xs, ys = (a[key], b[key]) if isinstance(a[key], list) else ([a[key]], [b[key]])
            assert all(x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(xs, ys)), key
    # A cache of its own: switching back to the plain dtype finds the plain cache untouched.
    again, _ = batches(Float8Model(), config)
    for a, b in zip(plain, again):
        xs, ys = (a['emb1'], b['emb1']) if isinstance(a['emb1'], list) else ([a['emb1']], [b['emb1']])
        assert all(torch.equal(x, y) for x, y in zip(xs, ys))


def test_float8_with_captions_encoded_on_the_fly_is_refused(tmp_path):
    images = make_images(tmp_path / 'images', ['a red bus'])
    with pytest.raises(ValueError, match='Remove text_embedding_dtype'):
        batches(Float8Model('float8', cache_text_embeddings=False), dataset_config(images))


class TestMaxCachedCaptions:
    def test_limit_captions(self):
        captions = [f'c{i}' for i in range(6)]
        assert dataset_util.limit_captions(captions, None, 1) == captions
        assert dataset_util.limit_captions(captions, 6, 1) == captions
        assert dataset_util.limit_captions(captions, 9, 1) == captions
        picked = dataset_util.limit_captions(captions, 2, 1)
        assert len(picked) == 2 and picked == sorted(picked, key=captions.index)
        assert dataset_util.limit_captions(captions, 2, 1) == picked
        assert {tuple(dataset_util.limit_captions(captions, 2, s)) for s in range(20)} != {tuple(picked)}

    @pytest.mark.parametrize('value', [0, -1, True, 1.5, '2'])
    def test_refused(self, value):
        with pytest.raises(ValueError, match='max_cached_captions must be an integer >= 1'):
            dataset_util.check_max_cached_captions(value, 'dir')


def caption_lists(images, limit, reuse_metadata_cache, where='directory'):
    directory = {'multiline_captions': True, 'reuse_metadata_cache': reuse_metadata_cache}
    config = dataset_config(images, **directory)
    if limit is not None:
        (config['directory'][0] if where == 'directory' else config)['max_cached_captions'] = limit
    model = Float8Model()
    ds = dataset_util.Dataset(config, model)
    from test.test_text_encoding_on_the_fly import run_caching
    run_caching(model, ds, config)
    captions = {}
    for directory_dataset in ds.directory_datasets:
        for bucket in directory_dataset.bucket_datasets():
            for spec, caption in zip(bucket.metadata_dataset['image_spec'], bucket.metadata_dataset['caption']):
                captions[spec[1].rsplit('/', 1)[-1]] = sorted(caption)
    return captions, sum(len(c) for c in captions.values())


@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
def test_max_cached_captions_keeps_that_many_of_each_image(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', ['a\nb\nc\nd\ne', 'one', 'p\nq'])
    every, every_count = caption_lists(images, None, reuse_metadata_cache)
    assert every == {'0.png': ['a', 'b', 'c', 'd', 'e'], '1.png': ['one'], '2.png': ['p', 'q']}
    limited, limited_count = caption_lists(images, 2, reuse_metadata_cache)
    assert limited['1.png'] == ['one'] and limited['2.png'] == ['p', 'q']
    assert len(limited['0.png']) == 2 and set(limited['0.png']) < set(every['0.png'])
    assert limited_count < every_count
    # The same pick on every run, and from the top level as from the directory.
    assert caption_lists(images, 2, reuse_metadata_cache)[0] == limited
    assert caption_lists(images, 2, reuse_metadata_cache, where='top')[0] == limited
    # Removing the limit brings every caption back.
    assert caption_lists(images, None, reuse_metadata_cache)[0] == every


def test_both_metadata_paths_pick_the_same_captions(tmp_path):
    images = make_images(tmp_path / 'images', ['a\nb\nc\nd\ne\nf', 'g\nh\ni'])
    assert caption_lists(images, 2, True)[0] == caption_lists(images, 2, False)[0]
