"""Parts of train.py's launch that no other test runs, because train.py is a script."""
import copy
import importlib.util
import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='module')
def train():
    spec = importlib.util.spec_from_file_location('train_script', ROOT / 'train.py')
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        # train.py parses its arguments when it is imported
        patch.setattr(sys, 'argv', ['train.py', '--config', 'x.toml'])
        spec.loader.exec_module(module)
    return module


def example_configs():
    import toml
    for path in sorted((ROOT / 'examples').rglob('*.toml')):
        config = toml.load(path)
        if 'model' in config and 'type' in config['model']:
            yield pytest.param(config, id=str(path.relative_to(ROOT / 'examples')))


@pytest.mark.parametrize('config', example_configs())
def test_set_config_defaults_runs_before_the_process_group_exists(train, config):
    # set_config_defaults runs before deepspeed.init_distributed(): it must not ask for the rank. A [teacher]
    # table made it print its warnings through is_main_process(), which raised on every model.
    config = copy.deepcopy(config)
    config.setdefault('save_every_n_epochs', 1)
    config['teacher'] = {'loss_weight': 1.0}
    try:
        train.set_config_defaults(config)
    except (ValueError, NotImplementedError, KeyError, FileNotFoundError):
        pass  # a refusal of this config (a model that cannot use [teacher], a path); not the failure tested here
    else:
        assert train.model_teacher.parse_model_teacher_config(config) is not None or config['model']['type'] in (
            'moss_image', 'anima_refiner')


def test_the_teacher_warnings_are_printed_after_init_distributed(train):
    source = inspect.getsource(train).split("if __name__ == '__main__':", 1)[1]
    assert source.index('deepspeed.init_distributed()') < source.index('teacher_warnings:')


def test_regenerate_cache_can_come_from_the_config(train):
    assert train.parser.parse_args(['--config', 'x.toml']).regenerate_cache is None
    assert train.parser.parse_args(['--config', 'x.toml', '--regenerate_cache']).regenerate_cache is True


def test_dump_dataset_hands_post_init_the_batch_size_dicts(train):
    source = inspect.getsource(train)
    assert 'train_data.post_init(0, 1, micro_batch_size_per_gpu, 1, image_micro_batch_size_per_gpu)' in source


def test_flux2_control_latents_use_the_latent_format():
    source = (ROOT / 'models' / 'flux2.py').read_text()
    assert 'model_patcher' not in source
    assert 'self.latent_format.process_in(control_latents)' in source
