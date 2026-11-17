"""examples/moss_image/moss_image_heal.toml goes through the validators of every feature it turns on, and
carries the values of the MOSS-Image-heal model card (KBlueLeaf/MOSS-Image-heal)."""

import json
from pathlib import Path

import toml

from models.moss_image import LearningRates
from utils.distillation import parse_distill_config
from utils.outlier_decay import apply_outlier_decay_config_defaults, cap_at, rate_at
from utils.stream_loss import parse_stream_loss

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'moss_image' / 'moss_image_heal.toml'


def load():
    return json.loads(json.dumps(toml.load(EXAMPLE)))


def test_the_stream_terms_are_the_cards():
    spec = parse_stream_loss(load()['model'])
    assert set(spec) == {'write', 'rms'}
    assert spec['write']['weight'] == 5e-3 and spec['write']['target'] == 0.5
    assert spec['rms']['weight'] == 5e-5


def test_the_outlier_decay_is_the_cards():
    config = load()
    apply_outlier_decay_config_defaults(config)
    spec = config['outlier_decay']
    assert spec['rate'] == 12.0
    assert cap_at(spec, 0) == 64.0 and cap_at(spec, 10000) == 16.0
    # The geometric midpoint of 64 -> 16 over 10k steps.
    assert abs(cap_at(spec, 5000) - 32.0) < 1e-9
    assert rate_at(spec, 1000) == 12.0 and rate_at(spec, 0) < 0.1
    # lr * rate stays below 1 at the highest rate of the schedule, which the decay checks at startup.
    assert config['optimizer']['lr'] * spec['rate'] < 1.0


def test_the_distillation_has_weight_one_toward_the_base():
    distill = parse_distill_config(load())
    assert distill is not None and distill.loss_weight == 1.0 and distill.decay == 'none'


def test_only_the_text_refiner_is_frozen():
    config = load()
    rates = LearningRates(config['model'], config['optimizer'])
    assert rates.part_lrs == {'text_refiner': 0.0}
    assert rates.base_lr == 1e-5
    assert config['lr_scheduler'] == 'cosine' and config['lr_scheduler_min_ratio'] == 0.01
    # 4 GPUs x batch 4 x 4 accumulation = 64 images per step.
    assert config['micro_batch_size_per_gpu'] * config['gradient_accumulation_steps'] * 4 == 64


def test_the_cards_float16_precision_is_valid_for_this_recipe():
    from utils.loss_scale import apply_loss_scale_config_defaults
    text = EXAMPLE.read_text()
    for line in ("#dtype = 'float32'", "#compute_dtype = 'float16'", "#fp16_fix = 'online'"):
        assert line in text
    config = load()
    config['model'].update({'dtype': 'float32', 'compute_dtype': 'float16', 'fp16_fix': 'online'})
    spec = apply_loss_scale_config_defaults(config)
    assert spec['init_scale'] == 65536.0 and config['skip_nonfinite_steps'] is True
