"""partition_method = 'memory' (utils/memory_partition.py): stages balanced by the bytes of their layers."""

import json
import types

import pytest
import torch
import torch.multiprocessing as mp

from utils import memory_partition as mp_


def block(numel, dtype=torch.bfloat16, trainable=False):
    layer = torch.nn.Module()
    layer.w = torch.nn.Parameter(torch.zeros(numel, dtype=dtype), requires_grad=trainable)
    return layer


def test_a_layer_weighs_its_bytes_so_the_dtype_counts():
    assert mp_.layer_weight_bytes(block(16, torch.float8_e4m3fn)) == 16
    assert mp_.layer_weight_bytes(block(16, torch.bfloat16)) == 32
    assert mp_.layer_weight_bytes(block(16, torch.float32)) == 64


def test_frozen_weights_count_and_a_trainable_parameter_adds_its_gradient_and_states():
    frozen = block(16, trainable=False)
    trainable = block(16, trainable=True)
    assert mp_.layer_weight_bytes(frozen, 3.0) == 32
    assert mp_.layer_weight_bytes(trainable, 3.0) == 32 + 3 * 32
    assert mp_.layer_weight_bytes(trainable, 0.0) == 32


def test_a_tensor_shared_inside_a_layer_counts_once_and_a_function_layer_is_free():
    layer = torch.nn.Module()
    layer.a = torch.nn.Linear(4, 4, bias=False)
    layer.b = layer.a
    assert mp_.layer_weight_bytes(layer) == 64
    assert mp_.layer_weight_bytes(lambda x: x) == 0


def test_a_frozen_text_encoder_in_the_first_layer_takes_blocks_off_the_first_stage():
    # DeepSpeed's 'parameters' counts only trainable elements, so the frozen encoder weighs nothing there.
    from deepspeed.runtime.utils import partition_balanced
    first = block(4 * 100, trainable=False)
    first.encoder = torch.nn.Parameter(torch.zeros(8 * 100, dtype=torch.bfloat16), requires_grad=False)
    layers = [first] + [block(100, trainable=True) for _ in range(8)]
    trainable_only = [sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in layers]
    by_parameters = [int(b) for b in partition_balanced(trainable_only, 2)[1:-1]]
    split = mp_.split_by_weights([mp_.layer_weight_bytes(layer) for layer in layers], 2)
    # The encoder's 1600 bytes + the first layer's 800 against blocks of 200 bytes: stage 0 gets fewer layers.
    assert split[0] < by_parameters[0]
    weights = [mp_.layer_weight_bytes(layer) for layer in layers]
    assert max(mp_.stage_bytes(weights, split)) < max(mp_.stage_bytes(weights, by_parameters))


def test_fp8_blocks_are_lighter_than_bfloat16_ones_when_the_stages_are_balanced():
    layers = [block(100, torch.float8_e4m3fn) for _ in range(6)] + [block(100, torch.bfloat16) for _ in range(3)]
    weights = [mp_.layer_weight_bytes(layer) for layer in layers]
    split = mp_.split_by_weights(weights, 3)
    # 600 fp8 bytes + 600 bf16 bytes = 1200: 400 a stage, which is 4 fp8, 2 fp8 + 1 bf16, ... not 3 and 3 and 3.
    assert split != [3, 6]
    sizes = mp_.stage_bytes(weights, split)
    assert max(sizes) <= 600


def test_the_teacher_is_balanced_by_bytes_too():
    from utils import model_teacher
    layers = [block(100, torch.float8_e4m3fn) for _ in range(6)] + [block(100, torch.bfloat16) for _ in range(3)]
    method, split = model_teacher.teacher_partition(layers, 3, 'memory', [1, 2])
    assert method == 'manual' and split == mp_.teacher_split(layers, 3)
    assert split == mp_.split_by_weights([mp_.layer_weight_bytes(layer) for layer in layers], 3)
    method, split, parts = model_teacher.teacher_plan(layers, 3, 'memory')
    assert parts == [0, *split, 9]


def test_text_encoder_bytes_of_a_module_a_loaded_comfy_encoder_and_an_unloaded_one():
    assert mp_.text_encoder_bytes(torch.nn.Linear(4, 4, bias=False)) == 64
    assert mp_.text_encoder_bytes(None) == 0
    loaded = types.SimpleNamespace(_model=types.SimpleNamespace(patcher=types.SimpleNamespace(model_size=lambda: 1234)))
    assert mp_.text_encoder_bytes(loaded) == 1234
    assert mp_.text_encoder_bytes(types.SimpleNamespace(_model=None)) == 0


def test_the_models_say_how_much_text_encoder_the_first_stage_holds():
    from models.base import BasePipeline
    from models.iris import IrisPipeline
    from models.moss_image import MossImagePipeline
    encoder = torch.nn.Linear(4, 4, bias=False)
    # The loader's encoders are the plan's business (loader_encoders), not a model hook.
    assert BasePipeline.first_stage_extra_bytes(types.SimpleNamespace()) == 0
    iris = types.SimpleNamespace(cache_text_embeddings=False, text_encoder=types.SimpleNamespace(decoder=encoder))
    assert IrisPipeline.first_stage_extra_bytes(iris) == 64
    assert IrisPipeline.first_stage_extra_bytes(types.SimpleNamespace(cache_text_embeddings=True, text_encoder=None)) == 0
    comfy = types.SimpleNamespace(_model=types.SimpleNamespace(patcher=types.SimpleNamespace(model_size=lambda: 99)))
    assert MossImagePipeline.first_stage_extra_bytes(
        types.SimpleNamespace(cache_text_embeddings=False, text_encoders=[comfy])) == 99
    assert MossImagePipeline.first_stage_extra_bytes(
        types.SimpleNamespace(cache_text_embeddings=True, text_encoders=[comfy])) == 0


def test_a_quantized_tensor_weighs_its_real_storage_not_its_logical_size():
    from comfy_kitchen.tensor import QuantizedTensor
    weight = torch.randn(64, 64)
    quantized = QuantizedTensor.from_float(weight, 'TensorCoreFP8Layout')
    assert mp_.tensors_bytes([quantized]) == 64 * 64 == int(quantized.nbytes)
    assert quantized.numel() * quantized.element_size() > 64 * 64


def test_loader_encoders_are_counted_where_loaded_and_by_file_size_where_not(tmp_path):
    (tmp_path / 'te.safetensors').write_bytes(b'x' * 1000)
    (tmp_path / 'dir').mkdir()
    (tmp_path / 'dir' / 'a.safetensors').write_bytes(b'x' * 300)
    (tmp_path / 'dir' / 'notes.txt').write_bytes(b'x' * 50)
    assert mp_.text_files_bytes([str(tmp_path / 'te.safetensors'), str(tmp_path / 'dir'), str(tmp_path / 'none')]) == 1300
    model = types.SimpleNamespace(text_encoder_file_bytes=lambda: 1300)
    unloaded = types.SimpleNamespace(_model=None)
    loaded = types.SimpleNamespace(_model=types.SimpleNamespace(patcher=types.SimpleNamespace(model_size=lambda: 500)))
    assert mp_.loader_encoders_bytes(model, []) == 0
    assert mp_.loader_encoders_bytes(model, [unloaded]) == 1300
    assert mp_.loader_encoders_bytes(model, [loaded]) == 500
    module = torch.nn.Linear(4, 4, bias=False)
    assert mp_.loader_encoders_bytes(model, [module]) == 64


def test_the_plan_counts_the_loader_encoders_without_asking_the_model_for_its_text_encoders():
    # ComfyPipeline's free_vae_and_te() deletes `text_encoders` before the plan: the loader keeps its own list.
    model = types.SimpleNamespace(first_stage_extra_bytes=lambda: 0, text_encoder_file_bytes=lambda: 800)
    layers = [block(100, trainable=False) for _ in range(5)]
    split, text = mp_.plan(layers, model, 2, 0.0, 0, loader_encoders=[types.SimpleNamespace(_model=None)])
    assert split == mp_.split_by_weights([200 + 800] + [200] * 4, 2)


def test_a_resume_under_other_stage_boundaries_is_refused_unless_the_optimizer_is_reset():
    mp_.check_resumed_partition([0, 3, 9], [0, 3, 9])
    mp_.check_resumed_partition(None, [0, 3, 9])
    mp_.check_resumed_partition([0, 4, 9], [0, 3, 9], reset_optimizer=True)
    with pytest.raises(RuntimeError, match=r"partition_split = \[4\]"):
        mp_.check_resumed_partition([0, 4, 9], [0, 3, 9])


def test_train_py_and_the_saver_record_and_check_the_boundaries():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert 'memory_partition.PARTS_FILE' in (root / 'utils' / 'saver.py').read_text()
    train = (root / 'train.py').read_text()
    assert train.index('memory_partition.check_resumed_partition(') < train.index('load_checkpoint(')
    assert 'memory_partition.read_saved_parts(run_dir, resume_tag)' in train and "loader_encoders=list(text_encoding.text_encoders)" in train


@pytest.mark.parametrize('key, value', [('partition_extra_first_stage_gb', -1), ('partition_trainable_factor', 'x'),
                                        ('partition_extra_first_stage_gb', True),
                                        ('partition_trainable_factor', float('inf'))])
def test_a_bad_partition_number_is_refused(key, value):
    with pytest.raises(ValueError, match=key):
        mp_.check_partition_config({'partition_method': 'memory', 'pipeline_stages': 2, key: value})


def test_partition_keys_that_do_not_apply_are_ignored_with_a_warning(capsys):
    mp_.check_partition_config({'partition_method': 'parameters', 'pipeline_stages': 2,
                                'partition_extra_first_stage_gb': 2, 'partition_trainable_factor': 3})
    mp_.check_partition_config({'partition_method': 'memory', 'pipeline_stages': 1})
    mp_.check_partition_config({'partition_method': 'memory', 'pipeline_stages': 2, 'partition_split': [3]})
    out = capsys.readouterr().out
    assert out.count('WARNING') == 4 and 'partition_split only applies' in out
    assert 'partition_extra_first_stage_gb' in out and 'partition_trainable_factor' in out and 'pipeline_stages = 1' in out
    mp_.check_partition_config({'partition_method': 'memory', 'pipeline_stages': 2, 'partition_extra_first_stage_gb': 2})
    assert capsys.readouterr().out == ''


def test_without_a_process_group_the_plan_is_the_local_one():
    model = types.SimpleNamespace(first_stage_extra_bytes=lambda: 400)
    layers = [block(100, trainable=False) for _ in range(5)]
    split, text = mp_.plan(layers, model, 2, 3.0, 0)
    assert split == mp_.split_by_weights([400 + 200] + [200] * 4, 2)
    assert 'stage 0' in text and 'stage 1' in text


def test_extra_first_stage_gb_is_added_to_the_first_layer():
    model = types.SimpleNamespace(first_stage_extra_bytes=lambda: 0)
    layers = [block(1024 ** 2, torch.float8_e4m3fn) for _ in range(8)]
    plain, _ = mp_.plan(layers, model, 2, 0.0, 0)
    padded, _ = mp_.plan(layers, model, 2, 0.0, 0.004)
    assert padded[0] < plain[0]


def _plan_worker(rank, world, init_file, out_dir, fail_rank):
    import os
    import deepspeed
    os.environ['LOCAL_RANK'] = str(rank)
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=world, verbose=False)
    # Only rank 1 has the encoder (a lazily loaded ComfyUI one lives on the first stage's rank only).
    extra = 800 if rank == 1 else 0

    def hook():
        if rank == fail_rank:
            raise RuntimeError('no encoder')
        return extra

    model = types.SimpleNamespace(first_stage_extra_bytes=hook)
    layers = [block(100, trainable=False) for _ in range(9)]
    result = {}
    try:
        split, text = mp_.plan(layers, model, 3, 3.0, 0)
        from utils.pipeline import ManualPipelineModule
        pipe = ManualPipelineModule(layers=layers, num_stages=3, partition_method='manual',
                                    manual_partition_split=split, dynamic_shape=True)
        result = {'split': split, 'parts': pipe.parts, 'text': text}
    except Exception as e:
        result = {'error': str(e)}
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


def _run_plan(tmp_path, fail_rank=-1):
    mp.spawn(_plan_worker, args=(3, str(tmp_path / 'init'), str(tmp_path), fail_rank), nprocs=3)
    return [json.load(open(tmp_path / f'{r}.json')) for r in range(3)]


def test_ranks_that_know_different_text_encoder_sizes_still_split_alike_and_build_the_engine_module(tmp_path):
    results = _run_plan(tmp_path)
    assert all('error' not in r for r in results), results
    assert results[0] == results[1] == results[2]
    split = results[0]['split']
    assert results[0]['parts'] == [0, *split, 9]
    # the encoder's 800 bytes (rank 1 only) sit on layer 0, so stage 0 has fewer layers than a uniform split's 3
    assert split[0] < 3


def test_a_failure_on_one_rank_stops_every_rank_instead_of_leaving_the_others_waiting(tmp_path):
    results = _run_plan(tmp_path, fail_rank=2)
    assert all('rank 2' in r['error'] and 'no encoder' in r['error'] for r in results), results


def test_train_py_splits_the_student_by_the_plan_and_checks_the_keys_with_the_config():
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / 'train.py').read_text()
    assert 'memory_partition.check_partition_config(config)' in source
    start = source.index("str(partition_method).lower() == 'memory' and num_stages > 1")
    block = source[start:start + 900]
    assert 'memory_partition.plan(' in block and "student_method = 'manual'" in block
    assert 'partition_method=student_method' in source and 'manual_partition_split=student_split' in source
    # the teacher gets the configured method ('memory' too), not the student's manual one
    assert 'teacher.layers, num_stages, partition_method,' in source


def test_ltx2_estimates_its_text_encoder_from_its_own_config_key(tmp_path):
    from models.ltx2 import LTX2Pipeline
    (tmp_path / 'te.safetensors').write_bytes(b'x' * 700)
    pipe = types.SimpleNamespace(model_config={'text_encoder': str(tmp_path / 'te.safetensors'),
                                               'diffusion_model': str(tmp_path / 'huge.safetensors')})
    assert LTX2Pipeline.text_encoder_file_bytes(pipe) == 700
