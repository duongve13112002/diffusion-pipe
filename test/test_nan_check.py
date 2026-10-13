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


def test_caption_lengths_in_tokens_with_the_model_tokenizer():
    from utils.nan_check import describe_batch

    def tokenizer(captions, add_special_tokens):
        assert add_special_tokens is False
        return {'input_ids': [c.split() for c in captions]}
    lines = describe_batch({'caption': ['a b c', '', 'word']}, tokenizer)
    assert 'caption lengths (chars) [5, 0, 4]' in lines
    assert 'caption lengths (tokens, before truncation) [3, 0, 1]' in lines


class SqrtLinear(nn.Module):
    """Finite forward; the gradient is infinite for a sample whose pre-activation is exactly 0."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, inputs):
        x, label = inputs
        return self.linear(x).abs().sqrt(), label


def test_non_finite_gradient_is_replayed_per_sample(capsys):
    pipeline, check, _ = setup(Scale(1.0), SqrtLinear(), Scale(1.0))
    latents = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]])  # sample 1 hits sqrt(0)
    check.batches, check.names = [['latents (2, 3)']], ['one.png', 'two.png']
    out = run(pipeline, (latents, torch.zeros(1)))
    with pytest.raises(NonFiniteError) as error:
        out[0].sum().backward()
    message = str(error.value)
    assert 'gradient of 1.linear.weight is non-finite' in message
    assert 'replay of layer 1 SqrtLinear with its recorded inputs:' in message
    assert "whole batch, default kernel: NaN/inf in ['linear.weight', 'input 0']" in message
    assert 'each sample alone, default kernel: 1 of 2 non-finite' in message
    assert "sample 1 two.png: NaN/inf in ['linear.weight', 'input 0']" in message


def test_replay_reports_a_gradient_that_arrives_non_finite():
    pipeline, check, features = setup(Scale(1.0), Scale(1.0))
    out = run(pipeline, features)
    with pytest.raises(NonFiniteError) as error:
        (out[0].sum() * float('nan')).backward()
    assert 'the gradient arriving at its outputs [0] is already non-finite' in str(error.value)
