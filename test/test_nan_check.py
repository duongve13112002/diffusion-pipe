"""utils/nan_check.py (train.py --nan_check) names the first non-finite value and the batch."""

import pytest
import torch
from torch import nn

from utils.nan_check import NanCheck, NonFiniteError


class Scale(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, inputs):
        x, label = inputs
        return x * self.weight, label


class Pipeline(nn.Module):
    """The two things NanCheck reads from a deepspeed PipelineModule."""

    def __init__(self, *layers):
        super().__init__()
        self.forward_funcs = list(layers)
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)


class Model:
    def prepare_inputs(self, inputs):
        return (inputs['latents'], torch.zeros(1)), None


def setup(*layers):
    pipeline = Pipeline(*layers)
    model = Model()
    check = NanCheck(pipeline, model, rank=3)
    check.before_step(7)
    batch = {'latents': torch.ones(2, 3), 'caption': ['a b', ''],
             'image_spec': [(None, '/data/x/one.png'), (None, '/data/x/two.png')]}
    features, _ = model.prepare_inputs(batch)
    return pipeline, check, features


def run(pipeline, features):
    out = features
    for layer in pipeline.forward_funcs:
        out = layer(out)
    return out


def test_finite_training_passes_untouched():
    pipeline, check, features = setup(Scale(2.0), Scale(0.5))
    out = run(pipeline, features)
    out[0].sum().backward()
    check.after_step(7, 1.25)
    assert torch.equal(out[0], features[0])


def test_first_non_finite_layer_is_named_with_step_rank_and_images(capsys):
    pipeline, check, features = setup(Scale(1.0), Scale(float('inf')), Scale(0.0))
    with pytest.raises(NonFiniteError) as error:
        run(pipeline, (features[0] * 0, features[1]))  # 0 * inf = NaN in layer 1
    message = str(error.value)
    assert 'rank 3, step 7: layer 1 Scale output 0 is non-finite (its inputs were finite)' in message
    assert "images ['one.png', 'two.png']" in message and 'caption lengths (chars) [3, 0]' in message
    assert message in capsys.readouterr().out


def test_non_finite_gradient_is_named():
    pipeline, check, features = setup(Scale(1.0), Scale(1.0))
    out = run(pipeline, features)
    with pytest.raises(NonFiniteError, match=r'gradient of 1\.weight is non-finite'):
        (out[0].sum() * float('nan')).backward()


def test_weight_broken_by_the_update_is_named():
    pipeline, check, features = setup(Scale(1.0), Scale(1.0))
    with torch.no_grad():
        pipeline.forward_funcs[0].weight.fill_(float('nan'))
    with pytest.raises(NonFiniteError, match=r'weight 0\.weight became non-finite in the optimizer step'):
        check.after_step(7, 0.5)


def test_non_finite_loss_without_a_local_cause_is_reported():
    pipeline, check, features = setup(Scale(1.0))
    with pytest.raises(NonFiniteError, match='loss is nan although no layer output'):
        check.after_step(7, float('nan'))
