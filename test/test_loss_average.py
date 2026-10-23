"""train/loss_avg and train/epoch_loss (utils/step_stats.py LossAverage), and their state in the
training checkpoint."""

import types

import pytest
import torch
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

from utils import saver as saver_util
from utils.step_stats import LossAverage


def test_the_average_covers_the_last_window_of_steps():
    average = LossAverage(3)
    average.add(4.0)
    assert average.average() == 4.0
    for loss in (2.0, 6.0, 1.0):
        average.add(loss)
    assert average.average() == pytest.approx(3.0)      # 2, 6 and 1: the 4 has left the window


def test_the_epoch_average_covers_the_epoch_and_starts_again():
    average = LossAverage(2)
    for loss in (1.0, 2.0, 6.0):
        average.add(loss)
    assert average.end_epoch() == pytest.approx(3.0)
    average.add(5.0)
    assert average.end_epoch() == 5.0
    assert average.average() == pytest.approx(5.5)      # the window runs on across epochs


def test_a_resumed_run_goes_on_with_the_same_averages():
    average = LossAverage(3)
    for loss in (1.0, 2.0, 3.0, 4.0):
        average.add(loss)
    resumed = LossAverage(3)
    resumed.load_state_dict(average.state_dict())
    for run in (average, resumed):
        run.add(8.0)
    assert resumed.average() == average.average() == pytest.approx(5.0)
    assert resumed.end_epoch() == average.end_epoch() == pytest.approx(3.6)


def test_a_smaller_window_keeps_the_newest_losses():
    average = LossAverage(4)
    for loss in (1.0, 2.0, 3.0, 4.0):
        average.add(loss)
    resumed = LossAverage(2)
    resumed.load_state_dict(average.state_dict())
    assert resumed.average() == pytest.approx(3.5)
    resumed.add(10.0)
    assert resumed.average() == pytest.approx(7.0)


class _Engine:
    grid = types.SimpleNamespace(get_data_parallel_rank=lambda: 0, get_pipe_parallel_rank=lambda: 0)
    global_steps = 7

    def __init__(self):
        self.checkpoint_engine = TorchCheckpointEngine()

    def save_checkpoint(self, save_dir, tag=None, client_state=None, **kwargs):
        self.checkpoint_engine.makedirs(f'{save_dir}/{tag}', exist_ok=True)
        self.client_state = client_state


@pytest.mark.parametrize('kept', [True, False], ids=['loss_average', 'none'])
def test_the_checkpoint_carries_the_averages(tmp_path, monkeypatch, kept):
    monkeypatch.setattr(saver_util, 'is_main_process', lambda: True)
    config_file = tmp_path / 'train.toml'
    config_file.write_text('')
    average = LossAverage(2)
    for loss in (1.0, 3.0, 5.0):
        average.add(loss)
    engine = _Engine()
    saver = saver_util.Saver(types.SimpleNamespace(config=str(config_file)), {}, False, tmp_path / 'run', None,
                             types.SimpleNamespace(state_dict=lambda: {}), engine, torch.nn.Linear(1, 1),
                             loss_average=average if kept else None)
    saver.save_checkpoint(7, 56)
    assert engine.client_state['step'] == 7
    if not kept:
        assert 'loss_average' not in engine.client_state
        return
    resumed = LossAverage(2)
    resumed.load_state_dict(engine.client_state['loss_average'])
    assert resumed.average() == pytest.approx(4.0)
    assert resumed.end_epoch() == pytest.approx(3.0)
