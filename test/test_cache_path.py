"""cache_path: the caches of a dataset directory live under another folder than <directory>/cache."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import utils.dataset as dataset_util  # noqa: E402
from test_text_encoding_on_the_fly import (CAPTIONS, Model, batches, dataset_config, local_workers,  # noqa: E402,F401
                                           make_images)


def test_the_default_is_the_cache_folder_of_the_directory(tmp_path):
    assert dataset_util.directory_cache_root(tmp_path / 'imgs', {}, {}) == tmp_path / 'imgs' / 'cache'


def test_cache_path_gives_each_directory_a_subfolder_of_its_own(tmp_path):
    a = dataset_util.directory_cache_root(tmp_path / 'one' / 'imgs', {}, {'cache_path': str(tmp_path / 'fast')})
    b = dataset_util.directory_cache_root(tmp_path / 'two' / 'imgs', {}, {'cache_path': str(tmp_path / 'fast')})
    assert a.parent == b.parent == tmp_path / 'fast'
    assert a != b and a.name.startswith('imgs_') and b.name.startswith('imgs_')
    # stable, and the directory's own setting wins over the top level one
    assert a == dataset_util.directory_cache_root(tmp_path / 'one' / 'imgs', {}, {'cache_path': str(tmp_path / 'fast')})
    own = dataset_util.directory_cache_root(tmp_path / 'one' / 'imgs', {'cache_path': str(tmp_path / 'ssd')},
                                            {'cache_path': str(tmp_path / 'fast')})
    assert own.parent == tmp_path / 'ssd'


@pytest.mark.parametrize('bad', ['', '  ', 3, ['a']])
def test_a_cache_path_that_is_not_a_folder_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match='cache_path'):
        dataset_util.directory_cache_root(tmp_path, {'cache_path': bad}, {})


@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
@pytest.mark.parametrize('level', ['directory', 'top'])
def test_every_cache_is_written_under_cache_path_and_none_beside_the_images(tmp_path, reuse_metadata_cache, level):
    images = make_images(tmp_path / 'images', CAPTIONS)
    elsewhere = tmp_path / 'elsewhere'
    config = dataset_config(images, reuse_metadata_cache=reuse_metadata_cache,
                            **({'cache_path': str(elsewhere)} if level == 'directory' else {}))
    if level == 'top':
        config['cache_path'] = str(elsewhere)
    result, _ = batches(Model(), config)
    assert not (images / 'cache').exists()
    root = dataset_util.directory_cache_root(images, config['directory'][0], config)
    assert root.parent == elsewhere and (root / 'fake').is_dir()
    assert list(root.rglob('text_embeddings_*')) and (list(root.rglob('latents')) or list(root.rglob('latent_profile_*')))
    # trained from the cache found there, which has what the default location would have
    default, _ = batches(Model(), dataset_config(make_images(tmp_path / 'other', CAPTIONS),
                                                 reuse_metadata_cache=reuse_metadata_cache))
    assert len(result) == len(default)

    def rows(batches_):
        return sorted(round(float(row.sum()), 3) for batch in batches_ for row in batch['latents'])

    assert rows(result) == rows(default)


def test_a_dataset_built_without_a_process_group_announces_the_cache_path(tmp_path, capsys, monkeypatch):
    """tools/inherit_metadata_cache.py builds datasets with no distributed backend."""
    images = make_images(tmp_path / 'images', CAPTIONS[:2])
    config = dataset_config(images, cache_path=str(tmp_path / 'elsewhere'))
    dataset_util._ANNOUNCED_CACHE_ROOTS.clear()

    def no_backend():
        raise AssertionError('DeepSpeed backend not set')

    monkeypatch.setattr(dataset_util, 'is_main_process', no_backend)
    dataset_util.Dataset(config, Model(cache_text_embeddings=False))
    assert 'caches go to' in capsys.readouterr().out


def test_a_symlinked_spelling_of_the_dataset_directory_shares_its_cache(tmp_path):
    real = tmp_path / 'real' / 'imgs'
    real.mkdir(parents=True)
    (tmp_path / 'link').symlink_to(tmp_path / 'real')
    config = {'cache_path': str(tmp_path / 'fast')}
    assert (dataset_util.directory_cache_root(real, {}, config)
            == dataset_util.directory_cache_root(tmp_path / 'link' / 'imgs', {}, config))
