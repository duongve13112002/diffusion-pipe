"""Whole-media script selection, including raw caption sources and isolated caches."""

import hashlib
import gc
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import datasets
import torch
from PIL import Image

from utils import dataset as dataset_module
from utils.caption_corpus import read_corpus
from utils.captions import (
    caption_matches_non_latin_requirement,
    enumerate_captions,
    validate_non_latin_caption_requirement,
)
from utils.dataset import DirectoryDataset
from test.test_caption_filter_dropout import text_map


@pytest.fixture(autouse=True)
def bounded_workers(monkeypatch):
    monkeypatch.setattr(dataset_module, 'NUM_PROC', 1)


def build(path, *, requirement=None, global_settings=None, **settings):
    config = {'resolutions': [64], **(global_settings or {})}
    directory = {'path': str(path), 'size_buckets': [[64, 64, 1]],
                 'shuffle_metadata': False, **settings}
    if requirement is not None:
        directory['require_non_latin_caption'] = requirement
    ds = DirectoryDataset(directory, config, 'selection_test', skip_dataset_validation=True)
    return ds


def source_files(path, source='json'):
    captions = {'mixed.png': ['a girl', 'blue eyes, 制服'], 'latin.png': ['a boy', 'outdoor']}
    for name in captions:
        Image.new('RGB', (64, 64)).save(path / name)
    if source == 'json':
        (path / 'captions.json').write_text(json.dumps(captions), encoding='utf-8')
    else:
        for name, variants in captions.items():
            (path / Path(name).with_suffix('.txt')).write_text('\n'.join(variants), encoding='utf-8')
    return captions


def rows(ds):
    return [row for bucket in ds.get_size_bucket_datasets() for row in bucket.metadata_dataset]


@pytest.mark.parametrize('captions,contains', [
    (['a girl', 'outdoor'], False),
    (['a girl\n制服\nblue eyes'], True),
    (['outdoor', 'Привет'], True),
    (['Tiếng Việt, Pokémon'], False),
    (['123 😺'], False),
    (['ไทย'], False),
    (['', '  '], False),
])
@pytest.mark.parametrize('requirement', [None, True, False])
def test_whole_item_truth_table(captions, contains, requirement):
    assert caption_matches_non_latin_requirement(captions, requirement) == (
        requirement is None or contains == requirement)


def test_annotation_markers_do_not_select_an_item():
    captions = ['注: a girl\n注: outdoor']
    assert caption_matches_non_latin_requirement(captions, False, '注:')
    assert not caption_matches_non_latin_requirement(captions, True, '注:')


@pytest.mark.parametrize('invalid', [0, 1, 'true', 'false', 'None', [], {}])
@pytest.mark.parametrize('scope', ['directory', 'global'])
def test_invalid_config_rejected_by_training_and_enumeration(tmp_path, invalid, scope):
    setting = {'require_non_latin_caption': invalid}
    with pytest.raises(ValueError, match='require_non_latin_caption'):
        build(tmp_path, global_settings=setting if scope == 'global' else {},
              **(setting if scope == 'directory' else {}))
    config = {'directory': [{'path': str(tmp_path), **(setting if scope == 'directory' else {})}],
              **(setting if scope == 'global' else {})}
    with pytest.raises(ValueError, match='require_non_latin_caption'):
        enumerate_captions(config)


@pytest.mark.parametrize('requirement', [None, True, False])
@pytest.mark.parametrize('source', ['json', 'txt', 'multiline'])
def test_metadata_selects_whole_media_and_preserves_all_captions(tmp_path, requirement, source):
    source_files(tmp_path, source)
    ds = build(tmp_path, requirement=requirement, multiline_captions=(source == 'multiline'))
    ds.cache_metadata()
    selected = rows(ds)
    expected = {'mixed.png', 'latin.png'} if requirement is None else {
        'mixed.png' if requirement else 'latin.png'}
    assert {Path(row['image_spec'][1]).name for row in selected} == expected
    for row in selected:
        if Path(row['image_spec'][1]).name == 'mixed.png':
            assert row['caption'] == (['a girl\nblue eyes, 制服'] if source == 'txt'
                                      else ['a girl', 'blue eyes, 制服'])


@pytest.mark.parametrize('runtime_augmentation', [False, True])
def test_selection_precedes_line_removal_and_runtime_augmentation(tmp_path, runtime_augmentation):
    source_files(tmp_path)
    # Construct directly to exercise both metadata and on-the-fly caption storage.
    config = {'resolutions': [64]}
    directory = {'path': str(tmp_path), 'size_buckets': [[64, 64, 1]],
                 'require_non_latin_caption': True, 'enable_remove_non_latin': True,
                 'caption_dropout_rate': 1.0, 'cache_shuffle_num': int(runtime_augmentation)}
    ds = DirectoryDataset(directory, config, 'selection_test', skip_dataset_validation=True,
                          caches_text_embeddings=not runtime_augmentation)
    ds.cache_metadata()
    assert len(rows(ds)) == 1
    assert Path(rows(ds)[0]['image_spec'][1]).name == 'mixed.png'
    assert rows(ds)[0]['caption'] == (['a girl', 'blue eyes, 制服'] if runtime_augmentation
                                     else ['a girl', ''])


def test_global_inheritance_and_false_directory_override(tmp_path):
    source_files(tmp_path)
    inherited = build(tmp_path, global_settings={'require_non_latin_caption': True})
    overridden = build(tmp_path, requirement=False,
                       global_settings={'require_non_latin_caption': True})
    for ds, expected in [(inherited, 'mixed.png'), (overridden, 'latin.png')]:
        ds.cache_metadata()
        assert [Path(row['image_spec'][1]).name for row in rows(ds)] == [expected]


def test_rejected_media_never_opened_and_empty_result_is_safe(tmp_path, monkeypatch):
    (tmp_path / 'bad.mp4').write_bytes(b'not a real video')
    (tmp_path / 'captions.json').write_text(json.dumps({'bad.mp4': ['a girl', '制服']}), encoding='utf-8')
    ds = build(tmp_path, requirement=False)
    warnings = []
    monkeypatch.setattr(dataset_module.logger, 'warning', warnings.append)
    monkeypatch.setattr(dataset_module.InputImpl, 'VideoFromFile',
                        lambda *args: pytest.fail('excluded video must not be opened'))
    ds.cache_metadata()
    assert rows(ds) == []
    assert any('no media remain' in message for message in warnings)
    assert json.loads(ds.grouping_keys_json_file.read_text()) == []


@pytest.mark.parametrize('requirement', [True, False])
@pytest.mark.parametrize('skip_empty', [True, False])
def test_missing_caption_respects_existing_skip_rule(tmp_path, requirement, skip_empty):
    Image.new('RGB', (64, 64)).save(tmp_path / 'missing.png')
    ds = build(tmp_path, requirement=requirement, skip_empty_caption=skip_empty)
    ds.cache_metadata()
    assert len(rows(ds)) == int(not requirement and not skip_empty)


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('requirement', [None, True, False])
def test_caption_only_enumeration_matches_media_selection(tmp_path, raw, requirement):
    captions = source_files(tmp_path)
    config = {'require_non_latin_caption': requirement,
              'directory': [{'path': str(tmp_path)}]}
    stats = {}
    result = enumerate_captions(config, apply_shuffle=not raw, stats=stats)
    expected = list(captions) if requirement is None else ['mixed.png' if requirement else 'latin.png']
    assert result == [caption for name in sorted(expected) for caption in captions[name]]
    assert stats.get('filtered', 0) == int(requirement is not None)


def test_raw_export_still_excludes_annotation_markers_from_selection(tmp_path):
    (tmp_path / 'source.txt').write_text('注: a girl\n注: outdoor', encoding='utf-8')
    config = {'require_non_latin_caption': False, 'prefix_tag_caption': '注:',
              'directory': [{'path': str(tmp_path)}]}
    assert enumerate_captions(config, apply_shuffle=False) == ['注: a girl\n注: outdoor']


def test_control_column_survives_a_rejected_row(tmp_path):
    source_files(tmp_path)
    ds = build(tmp_path, requirement=False, control_path=str(tmp_path))
    ds.cache_metadata()
    assert len(rows(ds)) == 1
    assert rows(ds)[0]['control_file'] == str(tmp_path / 'latin.png')


@pytest.mark.parametrize('requirement', [True, False])
def test_tar_members_use_the_same_whole_item_rule(tmp_path, requirement):
    captions = {'nested/mixed.png': ['a girl', '制服'], 'nested/latin.png': ['a boy']}
    pixels = io.BytesIO()
    Image.new('RGB', (64, 64)).save(pixels, format='PNG')
    data = pixels.getvalue()
    with tarfile.open(tmp_path / 'images.tar', 'w') as archive:
        for name in captions:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    (tmp_path / 'captions.json').write_text(json.dumps(captions), encoding='utf-8')
    ds = build(tmp_path, requirement=requirement)
    ds.cache_metadata()
    expected = 'nested/mixed.png' if requirement else 'nested/latin.png'
    assert [row['image_spec'][1] for row in rows(ds)] == [expected]
    config = {'require_non_latin_caption': requirement, 'directory': [{'path': str(tmp_path)}]}
    assert enumerate_captions(config, apply_shuffle=False) == captions[expected]


def test_export_cli_filters_whole_media_and_reports_counts(tmp_path):
    media = tmp_path / 'media'
    media.mkdir()
    source_files(media)
    config = tmp_path / 'dataset.toml'
    config.write_text('require_non_latin_caption = true\n[[directory]]\n'
                      f'path = "{media.as_posix()}"\n', encoding='utf-8')
    output = tmp_path / 'corpus.jsonl'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'tools/export_caption_corpus.py'),
               '--dataset', str(config), '--output', str(output), '--no-progress']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert 'Media files: 2 total' in result.stdout
    assert '1 excluded by require_non_latin_caption' in result.stdout
    assert read_corpus(output) == ['a girl', 'blue eyes, 制服']


def selection_latents(batch, rank):
    values = [int(Path(spec[1]).name == 'mixed.png') for spec in batch['image_spec']]
    return {'latents': torch.tensor(values).view(-1, 1), 'image_spec': batch['image_spec'],
            'caption': batch['caption'], 'mask': [None] * len(values)}


def test_modes_isolate_latent_and_text_caches_and_reuse_warm_mode(tmp_path, monkeypatch):
    source_files(tmp_path)
    modes = [build(tmp_path, requirement=value, keep_latent_cache=True,
                   keep_text_embedding_cache=True) for value in (None, True, False)]
    assert modes[0].cache_dir == tmp_path / 'cache' / 'selection_test'
    assert len({ds.cache_dir for ds in modes}) == 3
    snapshots = {}
    for ds in modes:
        ds.cache_metadata()
        ds.cache_latents(selection_latents)
        ds.cache_text_embeddings(text_map, 0)
        bucket = ds.get_size_bucket_datasets()[0]
        for index, row in enumerate(bucket.metadata_dataset):
            assert bucket.latent_dataset[index]['latents'].item() == int(
                Path(row['image_spec'][1]).name == 'mixed.png')
        for file in ds.cache_dir.rglob('shard_*.bin'):
            snapshots[file] = hashlib.sha256(file.read_bytes()).hexdigest()

    # Trusted warm metadata must load rather than overwrite memory-mapped Arrow files.
    warm = build(tmp_path, requirement=True, keep_latent_cache=True, keep_text_embedding_cache=True)
    monkeypatch.setattr(warm, '_group_metadata_and_save_to_disk',
                        lambda **kwargs: pytest.fail('warm metadata should be reused'))
    warm.cache_metadata(trust_cache=True)
    warm.cache_latents(None, trust_cache=True)
    warm.cache_text_embeddings(None, 0)
    assert [Path(row['image_spec'][1]).name for row in rows(warm)] == ['mixed.png']
    assert all(hashlib.sha256(file.read_bytes()).hexdigest() == digest
               for file, digest in snapshots.items())


def test_unset_keeps_legacy_cache_suffix(tmp_path):
    ds = build(tmp_path)
    assert ds.require_non_latin_caption is None
    assert ds.caption_cache_suffix == ''
    validate_non_latin_caption_requirement(None)


def test_kept_selected_latents_rebuild_when_equal_length_image_rows_change(tmp_path):
    source_files(tmp_path)
    ds = build(tmp_path, requirement=True, keep_latent_cache=True)
    ds.cache_metadata()
    ds.cache_latents(selection_latents)
    bucket = ds.get_size_bucket_datasets()[0]
    assert bucket.latent_dataset[0]['latents'].item() == 1
    # Model a subsequent run's updated selection without rewriting open metadata Arrow files.
    row = dict(bucket.metadata_dataset[0])
    row['image_spec'] = [None, str(tmp_path / 'latin.png')]
    row['caption'] = ['制服']
    bucket.metadata_dataset = datasets.Dataset.from_list([row])
    bucket.latent_dataset.con.close()
    for handle in bucket.latent_dataset.open_files.values():
        handle.close()
    bucket.iteration_order = None
    gc.collect()
    with pytest.raises(RuntimeError, match='cached rows do not match'):
        dataset_module._map_and_cache(
            bucket.metadata_dataset, None, bucket.cache_dir, cache_file_prefix='latents_',
            fingerprint_columns=[c for c in row if c != 'caption'],
            keep_on_fingerprint_change=True, content_column='image_spec')
    gc.collect()
    ds.cache_latents(selection_latents)
    assert bucket.latent_dataset[0]['latents'].item() == 0
    assert Path(bucket.iteration_order[0]['image_spec'][1]).name == 'latin.png'


def test_aspect_ratio_metadata_reuses_its_warm_selected_cache(tmp_path, monkeypatch):
    source_files(tmp_path)
    config = {'resolutions': [64], 'require_non_latin_caption': True}
    directory = {'path': str(tmp_path), 'enable_ar_bucket': False}
    first = DirectoryDataset(dict(directory), config, 'selection_test', skip_dataset_validation=True)
    first.cache_metadata()
    assert len(first.ar_bucket_datasets) == 1
    assert len(first.ar_bucket_datasets[0].metadata_dataset) == 1
    warm = DirectoryDataset(dict(directory), config, 'selection_test', skip_dataset_validation=True)
    monkeypatch.setattr(warm, '_group_metadata_and_save_to_disk',
                        lambda **kwargs: pytest.fail('warm aspect-ratio metadata should be reused'))
    warm.cache_metadata(trust_cache=True)
    assert Path(warm.ar_bucket_datasets[0].metadata_dataset[0]['image_spec'][1]).name == 'mixed.png'
