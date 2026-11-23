"""A pipeline stage's backward (deepspeed/runtime/pipe/engine.py, _exec_backward_pass) runs over every floating
output of the stage's last layer, and the last stage asserts that every floating tensor it received got a gradient
(_exec_send_grads). A constant a layer builds (rope tables, float masks, zeros) that stays in the tuple therefore has
to require grad, and a received tensor nothing reads has to be tied to the output (models/base.py tie_unused)."""

import ast
from pathlib import Path

import pytest
import torch

from models.base import tie_unused

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


def test_ltx2_ties_the_audio_entries_it_does_not_read_with_audio_off():
    final = method_source('ltx2.py', 'FinalLayer', 'forward')
    assert 'tie_unused(out, ax,' in final
    for name in ('a_context', 'a_timestep', 'a_freq_cis', 'av_cross_video_freq_cis', 'av_cross_audio_freq_cis',
                 'av_ca_video_scale_shift_timestep', 'av_ca_audio_scale_shift_timestep',
                 'av_ca_a2v_gate_noise_timestep', 'av_ca_v2a_gate_noise_timestep'):
        assert name in final.split('tie_unused(out', 1)[1], name


TIED_IN_FINAL_LAYER = [
    ('chroma.py', 'FinalLayer', 'tie_unused(img, txt, pe)'),
    ('cosmos.py', 'FinalLayer', 'tie_unused(x0_pred, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, crossattn_emb)'),
    ('cosmos_predict2.py', 'FinalLayer', 'tie_unused(net_output_B_C_T_H_W, timesteps_B_T'),
    ('qwen_image21.py', 'FinalLayer', 'tie_unused(out, mod_zeros)'),
    ('sdxl.py', 'FinalLayer', 'tie_unused(self.conv_out(sample), emb, encoder_hidden_states)'),
    ('qwen_image.py', 'FinalLayer', 'tie_unused(output, encoder_hidden_states'),
    ('sd3.py', 'FinalLayer', 'tie_unused(output, *_)'),
    ('z_image.py', 'FinalLayer', 'tie_unused(x, freqs_cis)'),
    ('wan/wan.py', 'FinalLayer', 'tie_unused(torch.stack(x, dim=0), e0, context)'),
]


@pytest.mark.parametrize('file,cls,call', TIED_IN_FINAL_LAYER)
def test_final_layer_ties_the_floating_entries_no_layer_of_the_last_stage_may_read(file, cls, call):
    assert call in method_source(file, cls, 'forward')


def test_flux2_ties_the_double_stream_modulation_that_the_concat_layer_drops():
    concat = method_source('flux2.py', 'ConcatenateTxtImg', 'forward')
    assert concat.index('tie_unused(img, *vec)') < concat.index('vec = [mod_out.shift')
