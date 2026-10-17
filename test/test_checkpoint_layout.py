"""utils/checkpoint_layout.py: a resume refuses a checkpoint of another pipeline layout."""

import pytest
import torch
from torch import nn
from deepspeed.runtime.checkpoint_engine import TorchCheckpointEngine
from deepspeed.runtime.pipe import LayerSpec
from deepspeed.runtime.pipe.module import PipelineModule, TiedLayerSpec

from utils.checkpoint_layout import check_checkpoint_layers, shift_checkpoint_layers


def checkpoint(tmp_path, indices, tag='global_step7', ranks=('model_00',)):
    (tmp_path / tag).mkdir(parents=True)
    for i in indices:
        for rank in ranks:
            torch.save({}, tmp_path / tag / f'layer_{i:02d}-{rank}-model_states.pt')
    (tmp_path / tag / 'mp_rank_00_model_states.pt').write_bytes(b'')
    (tmp_path / 'latest').write_text(tag + '\n')
    return tmp_path


class TextEncoderLayer(nn.Module):
    pass


def layers(n):
    return [TextEncoderLayer()] + [nn.Linear(2, 2) for _ in range(n - 1)]


def test_the_same_layout_loads(tmp_path):
    check_checkpoint_layers(checkpoint(tmp_path, range(45), ranks=('model_00', 'model_01')), layers(45))


def test_a_layer_added_in_front_is_refused(tmp_path):
    run = checkpoint(tmp_path, range(44))
    with pytest.raises(RuntimeError, match=r'layer files 0-43, this run builds layers 0-44 \(TextEncoderLayer first, 45 in all\)'):
        check_checkpoint_layers(run, layers(45))
    with pytest.raises(RuntimeError, match='another pipeline layout'):
        check_checkpoint_layers(run, layers(43))


def test_the_old_layout_of_a_model_that_put_layers_in_front_loads_shifted(tmp_path):
    # A MOSS-Image checkpoint from before its text encoder layer: 44 layers, this run builds 45.
    run = checkpoint(tmp_path, range(44))
    assert check_checkpoint_layers(run, layers(45), added_in_front=1) == 1
    assert check_checkpoint_layers(checkpoint(tmp_path / 'new', range(45)), layers(45), added_in_front=1) == 0
    # Only parameterless layers, and only exactly that shift.
    with pytest.raises(RuntimeError, match='another pipeline layout'):
        check_checkpoint_layers(run, [nn.Linear(2, 2)] + layers(44), added_in_front=1)
    with pytest.raises(RuntimeError, match='another pipeline layout'):
        check_checkpoint_layers(run, layers(46), added_in_front=1)


class Grid:
    def get_slice_parallel_rank(self):
        return 0

    def get_slice_parallel_world_size(self):
        return 1


class Stage:
    """The parts of DeepSpeed's PipelineModule its load_state_dir uses, on one pipeline stage."""
    load_state_dir = PipelineModule.load_state_dir
    ckpt_layer_path_list = PipelineModule.ckpt_layer_path_list

    def __init__(self, forward_funcs, local_start=0):
        self.forward_funcs = forward_funcs
        self._local_start = local_start
        self._grid = Grid()

    def _synchronize_tied_weights(self):
        pass


@pytest.mark.parametrize('stage_start', [0, 2], ids=['first stage', 'later stage'])
def test_deepspeed_loads_the_shifted_files_into_the_right_layers(tmp_path, stage_start):
    torch.manual_seed(0)
    old = [nn.Linear(2, 3), nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 1)]
    run = tmp_path / 'global_step3'
    run.mkdir()
    for i, layer in enumerate(old):
        torch.save(layer.state_dict(), run / f'layer_{i:02d}-model_states.pt')
    new = [TextEncoderLayer()] + [type(layer)(layer.in_features, layer.out_features) for layer in old]
    stage = Stage(new[stage_start:], stage_start)
    restore = shift_checkpoint_layers(stage, 1)
    stage.load_state_dir(str(run), TorchCheckpointEngine(), strict=False)
    restore()
    for i, layer in enumerate(new):
        if i >= max(stage_start, 1):
            torch.testing.assert_close(layer.weight, old[i - 1].weight, rtol=0, atol=0)
    # Restored: the next load reads each layer's own file again.
    assert stage.ckpt_layer_path_list(str(run), 0)[0].endswith(f'layer_{stage_start:02d}-model_states.pt')


def test_functions_have_no_files_and_layer_specs_do(tmp_path):
    run = checkpoint(tmp_path, [0, 2, 3])
    check_checkpoint_layers(run, [nn.Linear(2, 2), lambda x: x, LayerSpec(nn.Linear, 2, 2), nn.ReLU()])
    # A tied layer with a forward_fn becomes a function in DeepSpeed's layer list.
    check_checkpoint_layers(run, [nn.Linear(2, 2), TiedLayerSpec('t', nn.Linear, 2, 2, forward_fn=lambda m, x: x),
                                  TiedLayerSpec('t', nn.Linear, 2, 2), nn.ReLU()])
    with pytest.raises(RuntimeError, match='0, 2, 3'):
        check_checkpoint_layers(run, [nn.Linear(2, 2), nn.ReLU(), LayerSpec(nn.Linear, 2, 2), nn.ReLU()])


def test_the_checkpoint_latest_names_is_the_one_checked(tmp_path):
    checkpoint(tmp_path, range(3), tag='global_step1')
    checkpoint(tmp_path, range(4), tag='global_step2')
    check_checkpoint_layers(tmp_path, layers(4))
    (tmp_path / 'latest').write_text('global_step1')
    with pytest.raises(RuntimeError, match='global_step1'):
        check_checkpoint_layers(tmp_path, layers(4))


def test_a_run_without_a_checkpoint_is_left_to_deepspeed(tmp_path):
    check_checkpoint_layers(tmp_path, layers(4))
