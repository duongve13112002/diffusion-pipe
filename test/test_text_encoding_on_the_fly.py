"""cache_text_embeddings = false on a model without its own on-the-fly text encoding: the captions
of each batch are encoded with the model's text encoder functions in the training process
(utils/text_encoding.py), and the batch holds what the text embedding cache would have."""

import multiprocessing
import queue
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import utils.dataset as dataset_util  # noqa: E402
from utils import text_encoding  # noqa: E402
from utils.text_encoding import CONTROL_FILE_KEY, IS_VIDEO_KEY, TextEncodingOnTheFly, encodes_text_in_loader  # noqa: E402


class LocalPool:
    def __init__(self, count, initializer, args):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(count, initializer=initializer, initargs=args)

    def imap(self, function, iterable):
        return self.executor.map(function, iterable)

    def close(self):
        self.executor.shutdown(wait=True)


@pytest.fixture(autouse=True)
def local_workers(monkeypatch):
    monkeypatch.setattr(dataset_util, 'NUM_PROC', 1)
    monkeypatch.setattr(dataset_util, 'mp', types.SimpleNamespace(
        Manager=lambda: types.SimpleNamespace(Queue=queue.Queue), Pool=LocalPool, Pipe=multiprocessing.Pipe))
    monkeypatch.setattr(dataset_util, 'is_main_process', lambda: True)


def encode_text(captions, is_video, number):
    """Character codes padded to the longest caption of the group, as a real encoder pads."""
    longest = max(len(c) for c in captions)
    embeddings = torch.zeros(len(captions), longest, 2, dtype=torch.bfloat16)
    for i, caption in enumerate(captions):
        embeddings[i, :len(caption), 0] = torch.tensor([float(ord(ch)) for ch in caption])
        embeddings[i, :, 1] = number + 10 * float(is_video[i])
    return {f'emb{number}': embeddings, f'len{number}': torch.tensor([len(c) for c in captions])}


class Model:
    name = 'fake'
    framerate = None
    pixels_round_to_multiple = 16
    encodes_text_on_the_fly = False

    def __init__(self, cache_text_embeddings=True, control=False):
        self.config = {'model': {'cache_text_embeddings': cache_text_embeddings}}
        self.vae = types.SimpleNamespace(load_model_if_needed=lambda: None)
        self.text_encoders = [types.SimpleNamespace(load_model_if_needed=lambda: None, number=n) for n in (1, 2)]
        self.control = control
        self.calls = []

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    def vae_cache_key(self):
        return ''

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return self.text_encoders

    def get_preprocess_media_file_fn(self):
        from models.base import PreprocessMediaFile
        return PreprocessMediaFile({})

    def get_call_vae_fn(self, vae):
        if self.control:
            return lambda tensor, control: {'latents': tensor.mean(dim=1, keepdim=True),
                                            'control_latents': control.mean(dim=1, keepdim=True)}
        return lambda tensor: {'latents': tensor.mean(dim=1, keepdim=True)}

    def get_call_text_encoder_fn(self, text_encoder):
        number = text_encoder.number
        if self.control and number == 2:
            def fn(captions, is_video, control_file):
                control_file = [None] * len(captions) if control_file is None else control_file
                self.calls.append((number, list(captions), list(control_file)))
                result = encode_text(captions, is_video, number)
                result['control_name'] = [None if f is None else Path(f).name for f in control_file]
                return result
            return fn

        def fn(captions, is_video):
            self.calls.append((number, list(captions)))
            return encode_text(captions, is_video, number)
        return fn


def make_images(path, captions):
    path.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i, caption in enumerate(captions):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path / f'{i}.png')
        (path / f'{i}.txt').write_text(caption)
    return path


CAPTIONS = ['a', 'a red bus', 'two dogs', 'the quick brown fox jumps', 'x y', 'sunset over water']


def dataset_config(path, **directory):
    return {'resolutions': [32], 'enable_ar_bucket': False,
            'directory': [{'path': str(path), 'num_repeats': 1, 'size_buckets': [[32, 32, 1]], **directory}]}


def run_caching(model, ds, config):
    """DatasetManager.cache without its processes: _cache_fn runs in a thread and the main thread
    answers its tasks, as tests/test_iris.py does."""
    worker_ds = dataset_util.Dataset(config, model)
    manager = dataset_util.DatasetManager(model)
    q = queue.Queue()
    errors = []

    def worker():
        try:
            dataset_util._cache_fn([worker_ds], q, model.get_preprocess_media_file_fn(), len(manager.text_encoders),
                                   False, False, 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while (task := q.get()) is not None:
        manager._handle_task(task)
    thread.join()
    if errors:
        raise errors[0]
    ds.cache_metadata(trust_cache=True)
    ds.cache_latents(None, trust_cache=True)
    for i in range(1, len(manager.text_encoders) + 1):
        ds.cache_text_embeddings(None, i)
    return manager


def batches(model, config, caching_batch_size=1):
    ds = dataset_util.Dataset(config, model)
    manager = run_caching(model, ds, config)
    ds.post_init(0, 1, {None: 3}, 1, {None: 3})
    encoder = TextEncodingOnTheFly(model, caching_batch_size) if encodes_text_in_loader(model) else None
    result = []
    for k in range(len(ds)):
        batch = ds[k]
        result.append(batch if encoder is None else encoder(batch))
    return result, manager


def assert_same(a, b):
    assert sorted(a) == sorted(b)
    for key in a:
        if key == 'image_spec':
            assert [Path(s[1]).name for s in a[key]] == [Path(s[1]).name for s in b[key]]
        elif torch.is_tensor(a[key]):
            assert torch.is_tensor(b[key]) and a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key
        elif isinstance(a[key], list) and a[key] and torch.is_tensor(a[key][0]):
            assert len(a[key]) == len(b[key]), key
            for x, y in zip(a[key], b[key]):
                assert x.dtype == y.dtype and torch.equal(x, y), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize('reuse_metadata_cache', [True, False])
def test_a_batch_encoded_on_the_fly_is_the_cached_batch(tmp_path, reuse_metadata_cache):
    images = make_images(tmp_path / 'images', CAPTIONS)
    model = Model(cache_text_embeddings=False)
    on_the_fly, manager = batches(model, dataset_config(images, reuse_metadata_cache=reuse_metadata_cache))
    # Nothing cached for the text encoders: they encode the batches.
    assert manager.text_encoders == []
    assert not list(images.glob('cache/**/text_embeddings_*'))
    # The same directory, so the same metadata and order.
    cached, _ = batches(Model(), dataset_config(images, reuse_metadata_cache=reuse_metadata_cache))
    assert list(images.glob('cache/**/text_embeddings_*'))
    assert len(cached) == len(on_the_fly) == 2
    for a, b in zip(cached, on_the_fly):
        assert_same(a, b)
    # Captions of different lengths: rows of different shapes stay a list, as cached.
    assert isinstance(on_the_fly[0]['emb1'], list)
    # Every caption encoded alone, by each encoder, at caching_batch_size = 1; and the empty
    # caption once per encoder, for model.uncond_dict.
    assert sorted(len(captions) for _, captions in model.calls) == [1] * 14
    assert [captions for _, captions in model.calls].count(['']) == 2


def test_the_group_size_follows_caching_batch_size(tmp_path):
    model = Model(cache_text_embeddings=False)
    batches(model, dataset_config(make_images(tmp_path / 'images', CAPTIONS)), caching_batch_size=2)
    calls = [captions for _, captions in model.calls if captions != ['']]
    assert sorted(len(captions) for captions in calls) == [1, 1, 1, 1, 2, 2, 2, 2]


def test_rows_of_one_shape_are_stacked_and_copied_out_of_the_group():
    model = Model(cache_text_embeddings=False)
    encoder = TextEncodingOnTheFly(model, caching_batch_size=4)
    batch = {'caption': ['ab', 'cd', 'ef'], IS_VIDEO_KEY: [False, True, False], CONTROL_FILE_KEY: [None] * 3}
    batch = encoder(batch)
    assert IS_VIDEO_KEY not in batch and CONTROL_FILE_KEY not in batch
    assert batch['emb1'].shape == (3, 2, 2) and batch['emb1'].dtype == torch.bfloat16
    # is_video reaches the encoder per row.
    assert batch['emb1'][:, 0, 1].tolist() == [1.0, 11.0, 1.0]
    assert torch.equal(batch['len2'], torch.tensor([2, 2, 2]))


def test_a_row_does_not_keep_its_group_alive():
    rows = text_encoding._row_to_cpu(torch.zeros(4, 8)[1])
    assert rows.untyped_storage().nbytes() == 8 * 4


def test_the_control_file_reaches_an_encoder_that_takes_it(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS[:3])
    control = make_images(tmp_path / 'control', ['', '', ''])
    for caption in control.glob('*.txt'):
        caption.unlink()
    model = Model(cache_text_embeddings=False, control=True)
    result, _ = batches(model, dataset_config(images, control_path=str(control)))
    names = [Path(spec[1]).name for spec in result[0]['image_spec']]
    assert result[0]['control_name'] == names
    assert all(len(call) == 2 for call in model.calls if call[0] == 1)


def test_captions_are_augmented_per_sample_when_encoded_on_the_fly(tmp_path):
    images = make_images(tmp_path / 'images', CAPTIONS)
    on_the_fly = dataset_util.Dataset(dataset_config(images, cache_shuffle_num=2), Model(cache_text_embeddings=False))
    cached = dataset_util.Dataset(dataset_config(images, cache_shuffle_num=2), Model())
    assert on_the_fly.directory_datasets[0].augment_at_runtime
    assert not cached.directory_datasets[0].augment_at_runtime


def test_a_model_that_encodes_its_captions_itself_is_left_alone():
    model = Model(cache_text_embeddings=False)
    model.encodes_text_on_the_fly = True
    assert not encodes_text_in_loader(model)
    assert not encodes_text_in_loader(Model())


def test_the_loader_encodes_each_batch_before_prepare_inputs():
    seen = []

    class Prepared:
        def prepare_inputs(self, batch, timestep_quantile=None):
            seen.append(dict(batch))
            return (batch['emb1'],), (torch.zeros(len(batch['caption'])), None)

    batch = {'caption': ['ab', 'cd'], IS_VIDEO_KEY: [False, False], CONTROL_FILE_KEY: [None, None]}
    engine = types.SimpleNamespace(is_pipe_parallel=False)
    loader = dataset_util.PipelineDataLoader([batch], engine, 1, Prepared(), num_dataloader_workers=0,
                                             text_encoding=TextEncodingOnTheFly(Model(cache_text_embeddings=False)))
    next(iter(loader))
    assert 'emb1' in seen[0] and 'emb2' in seen[0] and IS_VIDEO_KEY not in seen[0]


def test_a_dropped_caption_is_encoded_as_caching_encodes_the_empty_one(tmp_path):
    """Not a video and no control image, as the cached empty caption."""
    images = make_images(tmp_path / 'images', CAPTIONS[:3])
    control = make_images(tmp_path / 'control', ['', '', ''])
    for caption in control.glob('*.txt'):
        caption.unlink()
    model = Model(cache_text_embeddings=False, control=True)
    result, _ = batches(model, dataset_config(images, control_path=str(control), caption_dropout_rate=1.0))
    assert result[0]['caption'] == ['', '', '']
    assert result[0]['control_name'] == [None] * 3


def test_the_empty_captions_embeddings_are_left_on_the_model():
    """Caching leaves them in model.uncond_dict, which MiniMax H3 reads with cfg > 1."""
    model = Model(cache_text_embeddings=False)
    encoder = TextEncodingOnTheFly(model)
    encoder({'caption': ['ab'], IS_VIDEO_KEY: [True], CONTROL_FILE_KEY: [None]})
    expected = {**encode_text([''], [False], 1), **encode_text([''], [False], 2)}
    assert sorted(model.uncond_dict) == sorted(expected)
    for key, value in expected.items():
        assert torch.equal(model.uncond_dict[key], value[0])


class Encoder(torch.nn.Module):
    """A torch text encoder built from its config: in train mode, with dropout."""

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)

    def to(self, *args, **kwargs):
        return self  # no CUDA here


def test_a_torch_text_encoder_runs_in_eval_mode():
    model = Model(cache_text_embeddings=False)
    model.text_encoders = [Encoder()]
    seen = []
    model.get_call_text_encoder_fn = lambda text_encoder: lambda captions, is_video: (
        seen.append(text_encoder.training) or {'emb': torch.ones(len(captions), 1)})
    TextEncodingOnTheFly(model)({'caption': ['ab'], IS_VIDEO_KEY: [False], CONTROL_FILE_KEY: [None]})
    assert seen and not any(seen)
