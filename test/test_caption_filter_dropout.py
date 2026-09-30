"""Caption filtering and per-access dropout through the real dataset/cache paths.

Only the VAE and text encoders are replaced with small deterministic tensor maps. Metadata,
iteration order, SQLite shards, conditional/unconditional selection and config inheritance
are the production implementations. No model downloads or GPU are needed.
"""

import functools
import json
import random

import pytest
import torch
from PIL import Image

from utils import dataset as dataset_module
from utils.captions import enumerate_captions, has_non_latin_script, preprocess_caption, shuffle_captions
from utils.dataset import DirectoryDataset


def latent_map(batch, rank):
    count = len(batch['image_spec'])
    return {
        'latents': torch.full((count, 4, 8, 8), 7.0),
        'image_spec': batch['image_spec'],
        'caption': batch['caption'],
        'mask': [None] * count,
    }


def text_map(batch, rank, encoder=0):
    # Different captions must have different outputs, including the unconditional caption.
    return {
        f'embedding_{encoder}': torch.tensor([[sum(map(ord, c)), len(c)] for c in batch['caption']]),
        f'attention_mask_{encoder}': torch.tensor([[bool(c)] for c in batch['caption']]),
    }


@pytest.fixture(autouse=True)
def bounded_dataset_workers(monkeypatch):
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)
    monkeypatch.setattr(dataset_module, 'UNCOND_FRACTION', 0.0)


def build_directory(path, captions=None, *, global_settings=None, caches_text_embeddings=True,
                    trust_cache=False, **settings):
    path.mkdir(parents=True, exist_ok=True)
    if captions is not None:
        if isinstance(captions, list):
            Image.new('RGB', (64, 64), (31, 42, 53)).save(path / 'a.png')
            (path / 'captions.json').write_text(json.dumps({'a.png': captions}), encoding='utf-8')
        else:
            for name, caption in captions.items():
                Image.new('RGB', (64, 64), (31, 42, 53)).save(path / f'{name}.png')
                (path / f'{name}.txt').write_text(caption, encoding='utf-8')
    config = {'resolutions': [64], **(global_settings or {})}
    directory = {'path': str(path), 'size_buckets': [[64, 64, 1]], **settings}
    ds = DirectoryDataset(directory, config, 'caption_test', skip_dataset_validation=True,
                          caches_text_embeddings=caches_text_embeddings)
    ds.cache_metadata(trust_cache=trust_cache)
    return ds


def prepare(ds, *, encoders=0, latent_fn=latent_map, text_fn=text_map, trust_cache=False):
    ds.cache_latents(latent_fn, trust_cache=trust_cache)
    for encoder in range(encoders):
        ds.cache_text_embeddings(
            None if text_fn is None else functools.partial(text_fn, encoder=encoder), encoder)
    return ds.get_size_bucket_datasets()[0]


def assert_embedding(item, caption, encoder=0):
    torch.testing.assert_close(
        item[f'embedding_{encoder}'], torch.tensor([sum(map(ord, caption)), len(caption)]))
    assert bool(item[f'attention_mask_{encoder}'][0]) == bool(caption)


class TestScriptRanges:
    @pytest.mark.parametrize('lo,hi', [
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xAC00, 0xD7AF),
        (0x0400, 0x04FF), (0x0600, 0x06FF), (0x0590, 0x05FF),
    ])
    def test_boundaries_are_inclusive(self, lo, hi):
        assert has_non_latin_script('Latin ' + chr(lo))
        assert has_non_latin_script(chr(hi) + ' Latin')

    @pytest.mark.parametrize('text', ['', 'Pokémon, résumé, ñ, ü', 'Tiếng Việt', '123, 😺', 'ไทย'])
    def test_characters_outside_the_requested_ranges_are_kept(self, text):
        assert not has_non_latin_script(text)


class TestLineFiltering:
    @pytest.mark.parametrize('script', ['制服', 'ひらがな', 'カタカナ', '한글', 'Привет', 'مرحبا', 'שלום'])
    def test_remove_whole_mixed_line(self, script):
        caption = f'a girl\nblue eyes, {script}\noutdoor'
        assert preprocess_caption(caption, enable_remove_non_latin=True) == 'a girl\noutdoor'

    def test_marker_is_excluded_and_training_prefix_is_added_afterwards(self):
        assert preprocess_caption(
            '注: Pokémon, blue eyes', prefix_tag_caption='注:', caption_prefix='制服, ',
            enable_remove_non_latin=True) == '制服, Pokémon, blue eyes'

    def test_markers_on_each_line_are_excluded_from_the_script_check(self):
        assert preprocess_caption('注: red\n注: blue', prefix_tag_caption='注:',
                                  enable_remove_non_latin=True) == 'red\nblue'

    @pytest.mark.parametrize('caption', ['Special: 制服', '制服\nПривет', '', '  \r\n\t'])
    def test_filtered_empty_never_gets_a_training_prefix(self, caption):
        assert preprocess_caption(caption, prefix_tag_caption='Special:', caption_prefix='anime, ',
                                  enable_remove_non_latin=True) == ''

    def test_filter_precedes_shuffling_and_tag_dropout(self):
        assert preprocess_caption('Special: red, 制服, blue', prefix_tag_caption='Special:',
                                  shuffle=True, tag_dropout_rate=1.0,
                                  enable_remove_non_latin=True, rng=random.Random(1)) == ''

    def test_disabled_flag_preserves_old_text_and_prefix_behavior(self):
        assert preprocess_caption('制服', caption_prefix='anime, ') == 'anime, 制服'
        assert shuffle_captions([''], caption_prefix='anime, ') == ['anime, ']

    def test_variants_keep_the_same_cardinality(self):
        variants = shuffle_captions(['red\n制服', 'שלום'], count=4, caption_prefix='anime, ',
                                   enable_remove_non_latin=True)
        assert variants == ['anime, red'] * 4 + [''] * 4


class TestConfig:
    def test_global_values_reach_every_bucket(self, tmp_path):
        ds = build_directory(tmp_path, ['red\n制服'], global_settings={
            'enable_remove_non_latin': True, 'caption_dropout_rate': 0.4})
        bucket = ds.get_size_bucket_datasets()[0]
        assert bucket.enable_remove_non_latin is True
        assert bucket.caption_dropout_rate == 0.4
        assert bucket.metadata_dataset[0]['caption'] == ['red']

    def test_directory_can_disable_global_values(self, tmp_path):
        ds = build_directory(tmp_path, ['制服'], global_settings={
            'enable_remove_non_latin': True, 'caption_dropout_rate': 1.0},
            enable_remove_non_latin=False, caption_dropout_rate=0.0)
        bucket = prepare(ds)
        assert bucket.enable_remove_non_latin is False
        assert bucket[0]['caption'] == '制服'

    @pytest.mark.parametrize('rate', [-0.1, 1.1, float('nan'), float('inf'), True, '0.1', None])
    @pytest.mark.parametrize('scope', ['global', 'directory'])
    def test_invalid_rates_fail_before_caching(self, tmp_path, rate, scope):
        with pytest.raises(ValueError, match='caption_dropout_rate'):
            build_directory(tmp_path, ['red'],
                            global_settings={'caption_dropout_rate': rate} if scope == 'global' else {},
                            **({'caption_dropout_rate': rate} if scope == 'directory' else {}))

    @pytest.mark.parametrize('flag', [0, 1, 'true', None])
    def test_filter_requires_a_boolean(self, tmp_path, flag):
        with pytest.raises(ValueError, match='enable_remove_non_latin'):
            build_directory(tmp_path, ['red'], enable_remove_non_latin=flag)


class TestRuntimeDropout:
    @pytest.mark.parametrize('cached', [False, True])
    @pytest.mark.parametrize('online', [False, True])
    def test_redraw_on_every_access_and_switch_all_encoders(self, tmp_path, monkeypatch, cached, online):
        ds = build_directory(tmp_path, ['Special: red\n制服'], caches_text_embeddings=cached,
                             online_captions=online, prefix_tag_caption='Special:',
                             caption_prefix='anime, ', enable_remove_non_latin=True,
                             caption_dropout_rate=0.2)
        bucket = prepare(ds, encoders=2 if cached else 0)
        draws = iter([0.19, 0.2, 0.99, 0.0])
        monkeypatch.setattr(dataset_module.random, 'random', lambda: next(draws))
        for expected in ['', 'anime, red', 'anime, red', '']:
            item = bucket[0]
            assert item['caption'] == expected
            assert torch.all(item['latents'] == 7)
            if cached:
                assert_embedding(item, expected, 0)
                assert_embedding(item, expected, 1)
        # No extra draw from the legacy probability or cache handling.
        with pytest.raises(StopIteration):
            next(draws)

    @pytest.mark.parametrize('rate', [0.0, 1.0])
    def test_explicit_rate_overrides_legacy_probability(self, tmp_path, monkeypatch, rate):
        bucket = prepare(build_directory(tmp_path, ['red'], caption_dropout_rate=rate), encoders=1)
        monkeypatch.setattr(dataset_module, 'UNCOND_FRACTION', 1.0 - rate)
        assert bucket[0]['caption'] == ('' if rate else 'red')
        assert_embedding(bucket[0], '' if rate else 'red')

    def test_absent_rate_keeps_legacy_fallback(self, tmp_path, monkeypatch):
        bucket = prepare(build_directory(tmp_path, ['red']), encoders=1)
        monkeypatch.setattr(dataset_module, 'UNCOND_FRACTION', 1.0)
        item = bucket[0]
        assert item['caption'] == ''
        assert_embedding(item, '')

    def test_zero_rate_does_not_consume_rng(self, tmp_path, monkeypatch):
        bucket = prepare(build_directory(tmp_path, ['red'], caption_dropout_rate=0.0))

        def no_draw():
            raise AssertionError('disabled caption dropout must not consume RNG')

        monkeypatch.setattr(dataset_module.random, 'random', no_draw)
        assert bucket[0]['caption'] == 'red'

    def test_empty_conditioning_survives_full_tag_dropout(self, tmp_path):
        bucket = prepare(build_directory(
            tmp_path, ['Special: 制服'], caches_text_embeddings=False,
            cache_shuffle_num=2, tag_dropout_rate=1.0, prefix_tag_caption='Special:',
            caption_prefix='anime, ', enable_remove_non_latin=True))
        assert bucket._augment_at_runtime
        assert bucket[0]['caption'] == ''

    def test_cached_online_caption_uses_the_same_variant_as_its_embedding(self, tmp_path):
        ds = build_directory(tmp_path, ['Special: red, blue\n制服'], online_captions=True,
                             cache_shuffle_num=4, prefix_tag_caption='Special:',
                             caption_prefix='anime, ', enable_remove_non_latin=True)
        bucket = prepare(ds, encoders=2)
        # Live edits cannot be paired with embeddings from before that edit. Four cache
        # variants also cannot be indexed into the original one-element caption list.
        ds.captions_dict['a.png'] = ['a different live caption']
        for i in range(len(bucket)):
            item = bucket[i]
            assert 'different' not in item['caption']
            assert '制服' not in item['caption']
            assert_embedding(item, item['caption'], 0)
            assert_embedding(item, item['caption'], 1)

    def test_uncached_online_caption_is_filtered_at_access_time(self, tmp_path):
        ds = build_directory(tmp_path, ['old'], online_captions=True,
                             caches_text_embeddings=False, enable_remove_non_latin=True)
        bucket = prepare(ds)
        ds.captions_dict['a.png'] = ['fresh\n制服']
        assert bucket[0]['caption'] == 'fresh'
        ds.captions_dict['a.png'] = ['制服']
        assert bucket[0]['caption'] == ''

    @pytest.mark.parametrize('mode', ['all', 'random_per_epoch'])
    def test_uncached_online_variants_map_back_to_original_captions(self, tmp_path, monkeypatch, mode):
        ds = build_directory(tmp_path, ['Special: red, blue\n制服', 'outdoor\nПривет'],
                             caches_text_embeddings=False, online_captions=True,
                             enable_remove_non_latin=True, caption_sampling=mode,
                             cache_shuffle_num=4, prefix_tag_caption='Special:',
                             caption_prefix='anime, ')
        bucket = prepare(ds)
        if mode == 'random_per_epoch':
            picks = iter([0, 4] * 4)
            monkeypatch.setattr(dataset_module.random, 'randrange', lambda n: next(picks))
        results = []
        for i in range(8):
            caption = bucket[i % len(bucket)]['caption']
            assert caption.startswith('anime, ')
            assert not has_non_latin_script(caption)
            body = caption.removeprefix('anime, ')
            assert body == 'outdoor' or sorted(body.split(', ')) == ['blue', 'red']
            results.append(body)
        assert 'outdoor' in results
        assert any(c != 'outdoor' for c in results)

    @pytest.mark.parametrize('mode', ['all', 'random_per_epoch'])
    def test_multicaption_selection_stays_paired(self, tmp_path, mode):
        ds = build_directory(tmp_path, ['red\n制服', 'blue', 'Привет'],
                             enable_remove_non_latin=True, caption_sampling=mode,
                             caption_dropout_rate=0.5)
        bucket = prepare(ds, encoders=1)
        assert len(bucket) == (3 if mode == 'all' else 1)
        for i in range(30):
            item = bucket[i % len(bucket)]
            assert item['caption'] in ('red', 'blue', '')
            assert_embedding(item, item['caption'])


class TestMetadataAndCache:
    @pytest.mark.parametrize('multiline', [False, True])
    def test_sidecars_are_unchanged_and_filtered_empty_images_are_retained(self, tmp_path, multiline):
        originals = {'a': 'red\n制服\nPokémon', 'b': 'Привет\n制服', 'c': ''}
        ds = build_directory(tmp_path, originals, enable_remove_non_latin=True,
                             multiline_captions=multiline, caption_prefix='anime, ')
        bucket = prepare(ds, encoders=1)
        assert len(bucket.latent_dataset) == 3
        values = [bucket[i]['caption'] for i in range(len(bucket))]
        assert '' in values
        assert 'anime, ' not in values
        assert not any(has_non_latin_script(c) for c in values)
        for name, text in originals.items():
            assert (tmp_path / f'{name}.txt').read_text(encoding='utf-8') == text

    def test_json_is_unchanged(self, tmp_path):
        ds = build_directory(tmp_path, ['制服', 'blue'], enable_remove_non_latin=True)
        original = (tmp_path / 'captions.json').read_bytes()
        bucket = prepare(ds, encoders=1)
        assert {bucket[i]['caption'] for i in range(len(bucket))} == {'', 'blue'}
        assert (tmp_path / 'captions.json').read_bytes() == original

    @pytest.mark.parametrize('trust_cache', [False, True])
    def test_dropout_rate_changes_reuse_metadata_latents_and_text_cache(self, tmp_path, trust_cache):
        ds = build_directory(tmp_path, ['red'], caption_dropout_rate=0.0)
        first = prepare(ds, encoders=1)
        latent_path = first.cache_dir / 'latents' / 'shard_0.bin'
        text_path = first.cache_dir / 'text_embeddings_0' / 'shard_0.bin'
        before = (latent_path.read_bytes(), text_path.read_bytes())
        again = build_directory(tmp_path, caption_dropout_rate=1.0, trust_cache=trust_cache)
        assert again.caption_cache_suffix == ds.caption_cache_suffix == ''
        # None refuses to run an encoder; this only succeeds if the real caches are reused.
        second = prepare(again, encoders=1, latent_fn=None, text_fn=None, trust_cache=trust_cache)
        assert (latent_path.read_bytes(), text_path.read_bytes()) == before
        assert second.metadata_dataset[0]['caption'] == ['red'], 'dropout must not enter metadata'
        item = second[0]
        assert item['caption'] == ''
        assert_embedding(item, '')

    def test_filter_changes_text_cache_but_reuses_the_actual_latent_shard(self, tmp_path):
        ds = build_directory(tmp_path, ['red\n制服', 'שלום'], enable_remove_non_latin=False)
        first = prepare(ds, encoders=1)
        latent_path = first.cache_dir / 'latents' / 'shard_0.bin'
        before = latent_path.read_bytes()
        fingerprint = first.latent_dataset.fingerprint
        again = build_directory(tmp_path, enable_remove_non_latin=True, trust_cache=True)
        assert again.caption_cache_suffix != ds.caption_cache_suffix
        second = prepare(again, encoders=1, latent_fn=None, trust_cache=True)
        assert second.latent_dataset.fingerprint == fingerprint
        assert latent_path.read_bytes() == before
        assert second.metadata_dataset[0]['caption'] == ['red', '']
        for i in range(len(second)):
            item = second[i]
            assert_embedding(item, item['caption'])
        # Disabling filtering under --trust_cache must also restore the original captions.
        restored = build_directory(tmp_path, enable_remove_non_latin=False, trust_cache=True)
        third = prepare(restored, encoders=1, latent_fn=None, trust_cache=True)
        assert third.metadata_dataset[0]['caption'] == ['red\n制服', 'שלום']
        assert latent_path.read_bytes() == before

    @pytest.mark.parametrize('cached', [False, True])
    def test_filtered_empty_preserves_variant_and_latent_counts(self, tmp_path, cached):
        ds = build_directory(tmp_path, ['制服'], caches_text_embeddings=cached,
                             enable_remove_non_latin=True, cache_shuffle_num=4)
        bucket = prepare(ds, encoders=1 if cached else 0)
        assert len(bucket) == 4
        assert len(bucket.latent_dataset) == 1
        assert all(bucket[i]['caption'] == '' for i in range(4))


class TestCaptionOnlyEnumeration:
    def test_processed_enumeration_filters_but_never_freezes_whole_caption_dropout(self, tmp_path):
        ds = build_directory(tmp_path, {'a': 'red\n制服', 'b': 'Привет'}, global_settings={
            'enable_remove_non_latin': True, 'caption_dropout_rate': 1.0})
        config = {**ds.dataset_config, 'directory': [ds.directory_config]}
        assert sorted(enumerate_captions(config)) == ['', 'red']
        assert sorted(enumerate_captions(config, apply_shuffle=False)) == ['red\n制服', 'Привет']

    def test_enumeration_honors_directory_override(self, tmp_path):
        ds = build_directory(tmp_path, {'a': '制服'}, global_settings={'enable_remove_non_latin': True},
                             enable_remove_non_latin=False)
        config = {**ds.dataset_config, 'directory': [ds.directory_config]}
        assert enumerate_captions(config) == ['制服']
