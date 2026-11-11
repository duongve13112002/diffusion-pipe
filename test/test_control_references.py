"""control_path_2, control_path_3, ...: a directory's further reference images, for a model that takes several
(max_control_images). Each reference reaches the VAE function at the size the model gives it and the text encoders
by file name; models that take one refuse them."""

import sys
import types
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'test'))

import utils.dataset as dataset_util  # noqa: E402
from models.base import PreprocessMediaFile  # noqa: E402
from test_text_encoding_on_the_fly import (  # noqa: E402,F401
    CAPTIONS, Model, batches, dataset_config, encode_text, local_workers, make_images, run_caching,
)


class OwnSize(PreprocessMediaFile):
    """A reference at its file's own size, as a model that keeps them (minimax_h3_image) preprocesses it."""

    def preprocess_control(self, path, size_bucket=None):
        with Image.open(path) as img:
            width, height = img.size
        return self((None, path), None, (width, height, 1))[0][0]


class ReferenceModel(Model):
    max_control_images = 3

    def __init__(self, cache_text_embeddings=True):
        super().__init__(cache_text_embeddings, control=True)

    def get_preprocess_media_file_fn(self):
        return OwnSize({})

    def get_call_vae_fn(self, vae):
        def fn(images, control_images=None):
            result = {'latents': images.mean(dim=1, keepdim=True)}
            for n in range(len(control_images[0])):
                key = 'control_latents' if n == 0 else f'control_latents_{n + 1}'
                result[key] = [sample[n].mean(dim=0, keepdim=True) for sample in control_images]
            return result
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        number = text_encoder.number
        if number != 2:
            return super().get_call_text_encoder_fn(text_encoder)

        def fn(captions, is_video, control_file):
            control_file = [None] * len(captions) if control_file is None else control_file
            self.calls.append((number, list(captions), list(control_file)))
            result = encode_text(captions, is_video, number)
            result['control_names'] = [[] if c is None else [Path(f).parent.name + '/' + Path(f).name for f in ([c] if isinstance(c, str) else c)]
                                       for c in control_file]
            return result
        return fn


def make_controls(path, count, size):
    path.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(count):
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path / f'{i}.png')
    return path


@pytest.mark.parametrize('cache_text_embeddings', [True, False])
def test_every_reference_reaches_the_vae_and_the_text_encoder(tmp_path, cache_text_embeddings):
    images = make_images(tmp_path / 'images', CAPTIONS[:3])
    first = make_controls(tmp_path / 'ref1', 3, (32, 32))
    second = make_controls(tmp_path / 'ref2', 3, (64, 48))
    third = make_controls(tmp_path / 'ref3', 3, (16, 48))
    model = ReferenceModel(cache_text_embeddings)
    result, _ = batches(model, dataset_config(images, control_path=str(first), control_path_2=str(second), control_path_3=str(third)))
    batch = result[0]
    for key, shape in (('control_latents', (1, 32, 32)), ('control_latents_2', (1, 48, 64)), ('control_latents_3', (1, 48, 16))):
        assert batch[key].shape == (3, *shape), key
    names = [Path(spec[1]).name for spec in batch['image_spec']]
    assert batch['control_names'] == [[f'ref1/{n}', f'ref2/{n}', f'ref3/{n}'] for n in names]


def test_a_missing_reference_file_is_named(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS[:3])
    first = make_controls(tmp_path / 'ref1', 3, (32, 32))
    second = make_controls(tmp_path / 'ref2', 2, (32, 32))
    with pytest.raises(RuntimeError, match='No control_path_2 file exists for image'):
        batches(ReferenceModel(), dataset_config(images, control_path=str(first), control_path_2=str(second)))


@pytest.mark.parametrize('directory, match', [
    ({'control_path_2': 'x'}, 'control_path_2 needs control_path'),
    ({'control_path': 'x', 'control_path_3': 'y'}, 'without gaps'),
    ({'control_path': 'x', 'control_path_1': 'y'}, 'without gaps'),
    ({'control_path': 'x', 'control_path_02': 'y'}, 'without leading zeros'),
])
def test_the_numbering_has_no_gaps(directory, match):
    with pytest.raises(ValueError, match=match):
        dataset_util.extra_control_paths(directory)


def test_a_model_that_takes_one_control_image_refuses_the_others(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS[:3])
    first = make_controls(tmp_path / 'ref1', 3, (32, 32))
    config = dataset_config(images, control_path=str(first), control_path_2=str(first))
    with pytest.raises(ValueError, match='control_path_2 in the directory .*: fake takes one control image'):
        dataset_util.Dataset(config, Model(control=True))
    # a model that takes several still has its limit: ReferenceModel takes 3
    config = dataset_config(images, control_path=str(first), **{f'control_path_{n}': str(first) for n in (2, 3)})
    dataset_util.Dataset(config, ReferenceModel())
    config = dataset_config(images, control_path=str(first), **{f'control_path_{n}': str(first) for n in (2, 3, 4)})
    with pytest.raises(ValueError, match='control_path_2, control_path_3, control_path_4 in the directory .*: fake takes at most 3'):
        dataset_util.Dataset(config, ReferenceModel())


def test_directories_with_another_number_of_references_do_not_share_a_batch(tmp_path):
    one = make_images(tmp_path / 'one', CAPTIONS[:3])
    two = make_images(tmp_path / 'two', CAPTIONS[:3])
    ref1 = make_controls(tmp_path / 'ref1', 3, (32, 32))
    ref2 = make_controls(tmp_path / 'ref2', 3, (32, 32))
    config = dataset_config(one, control_path=str(ref1))
    config['directory'].append(dataset_config(two, control_path=str(ref1), control_path_2=str(ref2))['directory'][0])
    result, _ = batches(ReferenceModel(), config)
    assert len(result) == 2
    for batch in result:
        directories = {Path(spec[1]).parent.name for spec in batch['image_spec']}
        assert len(directories) == 1
        assert ('control_latents_2' in batch) == (directories == {'two'})



class CountingModel(ReferenceModel):
    """Counts the batches its VAE function encodes."""

    def __init__(self, sizing=None):
        super().__init__()
        self.control_sizing = sizing
        self.vae_calls = 0

    def get_call_vae_fn(self, vae):
        inner = super().get_call_vae_fn(vae)

        def fn(images, control_images=None):
            self.vae_calls += 1
            return inner(images, control_images) if control_images is not None else {'latents': images.mean(dim=1, keepdim=True)}
        return fn


def encoded_batches(model, config):
    ds = dataset_util.Dataset(config, model)
    run_caching(model, ds, config)
    return model.vae_calls


@pytest.mark.parametrize('reuse_metadata_cache', [False, True])
def test_a_cache_made_with_another_control_sizing_is_rebuilt(tmp_path, reuse_metadata_cache):
    """A directory with control images records how the model sizes them in its latent fingerprint: latents made with
    references at another size are not reused. A directory without control images keeps its cache."""
    edit = make_images(tmp_path / 'edit', CAPTIONS[:3])
    plain = make_images(tmp_path / 'plain', CAPTIONS[:3])
    control = make_controls(tmp_path / 'ref1', 3, (32, 32))
    edit_config = dataset_config(edit, control_path=str(control), reuse_metadata_cache=reuse_metadata_cache)
    plain_config = dataset_config(plain, reuse_metadata_cache=reuse_metadata_cache)

    assert encoded_batches(CountingModel(), edit_config) > 0
    assert encoded_batches(CountingModel(), plain_config) > 0
    assert encoded_batches(CountingModel(), edit_config) == 0, 'the same sizing reuses its cache'
    assert encoded_batches(CountingModel('own_size'), edit_config) > 0, 'another sizing rebuilds the edit cache'
    assert encoded_batches(CountingModel('own_size'), edit_config) == 0
    assert encoded_batches(CountingModel('own_size'), plain_config) == 0, 'a directory without control images is kept'


def directory_dataset(paths):
    return types.SimpleNamespace(path='dir', control_extra_paths=paths)


def metadata(extra=None):
    columns = {'control_file': ['a', 'b'], 'image_file': ['a', 'b']}
    if extra is not None:
        columns['control_extra_file'] = extra
    return datasets.Dataset.from_dict(columns)


def test_metadata_with_another_number_of_references_than_the_config_is_refused():
    check = dataset_util.check_control_columns
    check(metadata(), directory_dataset([]))
    check(metadata([['x'], ['y']]), directory_dataset([Path('r2')]))
    check(metadata([['x', 'z'], ['y', 'w']]), directory_dataset([Path('r2'), Path('r3')]))
    with pytest.raises(RuntimeError, match='has 0 further reference image'):
        check(metadata(), directory_dataset([Path('r2')]))
    with pytest.raises(RuntimeError, match='has 1 further reference image.*config has 0'):
        check(metadata([['x'], ['y']]), directory_dataset([]))
    with pytest.raises(RuntimeError, match='has 1 further reference image'):
        check(metadata([['x'], ['y']]), directory_dataset([Path('r2'), Path('r3')]))
    # a directory without control images has nothing to check
    check(datasets.Dataset.from_dict({'image_file': ['a']}), directory_dataset([]))
