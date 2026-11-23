"""A pipeline stage's backward (deepspeed/runtime/pipe/engine.py, _exec_backward_pass) runs over every floating
output of the stage's last layer, and a stage asserts that every floating tensor it received got a gradient
(_exec_send_grads). A constant a layer builds (rope tables, float masks, zeros) that stays in the tuple therefore has
to require grad (checked per model below), and a received tensor nothing reads is tied to the output with a zero
weight by ManualPipelineModule.forward (models/base.py tie_unused)."""

import ast
from pathlib import Path

import pytest
import torch

from deepspeed.pipe import PipelineModule

from models.base import tie_unused
from utils.pipeline import ManualPipelineModule

MODELS = Path(__file__).resolve().parent.parent / 'models'


def method_source(file, cls, method):
    tree = ast.parse((MODELS / file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return ast.get_source_segment((MODELS / file).read_text(), item)
    raise AssertionError(f'{file}: {cls}.{method} not found')


def test_tie_unused_changes_nothing_and_gives_the_unused_tensor_a_zero_gradient():
    out = torch.randn(3, 4, requires_grad=True)
    unused = torch.randn(5, 2, requires_grad=True)
    tied = tie_unused(out * 2, unused)
    assert torch.equal(tied, out * 2)
    tied.sum().backward()
    assert unused.grad is not None and not unused.grad.any()
    assert torch.equal(out.grad, torch.full_like(out, 2.0))


def test_tie_unused_skips_what_has_no_graph_or_is_not_floating():
    out = torch.randn(2, requires_grad=True) * 1.0
    no_graph = torch.randn(2)
    ints = torch.arange(3)
    assert tie_unused(out, no_graph, ints, None) is not None
    tied = tie_unused(out, no_graph, ints)
    tied.sum().backward()
    assert no_graph.grad is None
    assert tie_unused(out) is out
    plain = torch.randn(2)
    assert tie_unused(plain, no_graph, ints) is plain


def test_tie_unused_ties_the_first_floating_element_whatever_its_position():
    steps = torch.arange(3)
    video = torch.randn(2, 3, requires_grad=True)
    unused = torch.randn(4, requires_grad=True)
    result = tie_unused((steps, video * 1.0), unused)
    assert result[0] is steps
    result[1].sum().backward()
    assert unused.grad is not None and not unused.grad.any()
    ints = (steps, steps)
    assert tie_unused(ints, unused) is ints


def test_tie_unused_does_nothing_without_a_tensor_that_needs_grad_or_with_grad_disabled():
    out = (torch.randn(2, requires_grad=True) * 1.0, torch.arange(2))
    assert tie_unused(out, torch.randn(3), torch.arange(2), None) is out
    single = torch.randn(2, requires_grad=True) * 1.0
    assert tie_unused(single, torch.randn(3)) is single
    with torch.no_grad():
        assert tie_unused(out, torch.randn(3, requires_grad=True)) is out


def test_tie_unused_picks_an_output_that_requires_grad():
    no_graph = torch.randn(2)
    graph = torch.randn(3, requires_grad=True) * 1.0
    unused = torch.randn(4, requires_grad=True)
    result = tie_unused((no_graph, graph), unused)
    assert result[0] is no_graph and result[1].grad_fn is not None
    result[1].sum().backward()
    assert unused.grad is not None


def test_tie_unused_leaves_an_output_that_has_no_graph_alone():
    out = (torch.randn(2), torch.randn(3))
    assert tie_unused(out, torch.randn(4, requires_grad=True)) is out


def test_tie_unused_does_not_let_a_nan_or_inf_reach_the_output():
    out = torch.randn(3, requires_grad=True) * 1.0
    bad = torch.tensor([float('nan'), float('inf')], requires_grad=True)
    tied = tie_unused(out, bad, torch.tensor(float('inf'), requires_grad=True))
    assert torch.isfinite(tied).all()
    tied.sum().backward()


def test_tie_unused_reaches_a_scalar_and_a_non_contiguous_tensor():
    out = torch.randn(3, requires_grad=True) * 1.0
    scalar = torch.tensor(2.0, requires_grad=True)
    strided = torch.randn(4, 6, requires_grad=True)
    tie_unused(out, scalar, strided.t()).sum().backward()
    assert scalar.grad is not None and strided.grad is not None


def test_tie_unused_ties_the_first_element_of_a_list_or_tuple_output():
    video = torch.randn(2, 3, requires_grad=True)
    audio = torch.randn(2, 5, requires_grad=True)
    unused = torch.randn(4, requires_grad=True)
    for kind in (list, tuple):
        unused.grad = None
        result = tie_unused(kind([video * 1.0, audio * 1.0]), unused)
        assert type(result) is kind and len(result) == 2
        assert torch.equal(result[0], video) and result[1].shape == audio.shape
        result[0].sum().backward()
        assert unused.grad is not None and not unused.grad.any()


def test_tie_unused_handles_an_empty_tensor():
    out = torch.randn(2, requires_grad=True) * 1.0
    empty = torch.zeros(1, 0, requires_grad=True)
    tied = tie_unused(out, empty)
    tied.sum().backward()
    assert empty.grad is not None and empty.grad.shape == empty.shape


def test_ideogram4_does_not_carry_a_float_mask_or_a_graphless_rope_across_stages():
    initial = method_source('ideogram4.py', 'InitialLayer', 'forward')
    assert 'freqs_cis.requires_grad_(True)' in initial
    # the additive float mask is built in each block, from the bool mask that travels
    assert 'masked_fill_' not in initial and 'attn_mask = (segment_ids' in initial
    block = method_source('ideogram4.py', 'TransformerLayer', 'forward')
    assert 'masked_fill_(~attn_mask' in block and 'self.layer(h, additive_mask, freqs_cis, adaln_input)' in block
    assert 'return make_contiguous(h, attn_mask, adaln_input, sizes, freqs_cis)' in block


def test_qwen_image21_marks_the_constants_it_puts_in_the_tuple():
    initial = method_source('qwen_image21.py', 'InitialLayer', 'forward')
    assert 'pe.requires_grad_(True)' in initial and 'mod_zeros.requires_grad_(True)' in initial
    assert initial.index('pe.requires_grad_(True)') < initial.index('return make_contiguous')


def test_cosmos_marks_the_rope_and_the_extra_position_embedding():
    initial = method_source('cosmos.py', 'InitialLayer', 'forward')
    assert 'rope_emb_L_1_1_D.requires_grad_(True)' in initial
    assert 'extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.requires_grad_(True)' in initial
    assert initial.index('rope_emb_L_1_1_D.requires_grad_(True)') < initial.index('return make_contiguous')


def stage_output(received):
    """What a stage whose layers keep only the first of the received tensors returns."""
    return (received[0] * 2.0, received[3])


def run_stage(monkeypatch, inputs, forward=stage_output):
    stage = object.__new__(ManualPipelineModule)
    monkeypatch.setattr(PipelineModule, 'forward', lambda self, x: forward(x))
    return stage.forward(inputs)


def test_a_stage_gives_a_gradient_to_every_floating_tensor_it_received_but_did_not_read(monkeypatch):
    received = [torch.randn(2, 3, requires_grad=True), torch.randn(5, requires_grad=True),
                torch.randn(4, 1, requires_grad=True), torch.arange(3), torch.randn(7, requires_grad=True)]
    out = run_stage(monkeypatch, tuple(received), lambda x: (x[0] * 2.0, x[3]))
    assert torch.equal(out[0], received[0] * 2.0) and out[1] is received[3]
    out[0].sum().backward()
    for i in (1, 2, 4):
        assert received[i].grad is not None and not received[i].grad.any(), i
    assert received[0].grad is not None and received[0].grad.eq(2.0).all()


def test_a_stage_that_received_nothing_that_needs_grad_returns_what_its_layers_returned(monkeypatch):
    received = (torch.randn(2, 3), torch.randn(4))
    sentinel = (torch.randn(2), torch.arange(2))
    assert run_stage(monkeypatch, received, lambda x: sentinel) is sentinel
    with torch.no_grad():
        grad_inputs = (torch.randn(2, requires_grad=True),)
        assert run_stage(monkeypatch, grad_inputs, lambda x: sentinel) is sentinel


def test_a_stage_returning_one_tensor_is_tied_too(monkeypatch):
    received = (torch.randn(2, requires_grad=True), torch.randn(3, requires_grad=True))
    out = run_stage(monkeypatch, received, lambda x: x[0] * 3.0)
    assert torch.is_tensor(out)
    out.sum().backward()
    assert received[1].grad is not None
