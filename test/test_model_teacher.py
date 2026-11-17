"""[teacher] distillation for every model but MOSS-Image and anima_refiner (utils/model_teacher.py): the config, the
teacher chain built from a model's own to_layers(), the loss, and the way the teacher's output reaches the last
pipeline stage inside the label (utils/dataset.py PipelineDataLoader). CPU, a toy model."""

import ast
import copy
import textwrap
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from models.base import BasePipeline, ComfyPipeline
from utils import dataset as dataset_util
from utils import model_teacher as mt


def parse(**table):
    return mt.parse_model_teacher_config({'model': {'type': 'flux'}, 'teacher': {'loss_weight': 0.5, **table}})


# configuration

def test_absent_zero_and_own_implementations_are_off():
    assert mt.parse_model_teacher_config({'model': {'type': 'flux'}}) is None
    assert mt.parse_model_teacher_config({'model': {'type': 'flux'}, 'teacher': {'loss_weight': 0}}) is None
    # An off table is not validated further.
    assert mt.parse_model_teacher_config({'model': {'type': 'flux'},
                                          'teacher': {'loss_weight': 0, 'decay': 'bogus'}}) is None
    # MOSS-Image reads its own [teacher].
    assert mt.parse_model_teacher_config({'model': {'type': 'moss_image'},
                                          'teacher': {'loss_weight': 1, 'decay': 'bogus'}}) is None


def test_anima_refiner_has_two_forms():
    # Naming a stock Anima (its transformer or its text encoder) is the refiner's own teacher, which mixes the target.
    for key in ('transformer_path', 'llm_path'):
        table = {'loss_weight': 1, key: '/anima'}
        assert mt.parse_model_teacher_config({'model': {'type': 'anima_refiner'}, 'teacher': table}) is None
    # Naming neither is this one: the refiner's own starting weights, validated like any other model's table.
    config = mt.parse_model_teacher_config({'model': {'type': 'anima_refiner'},
                                            'teacher': {'loss_weight': 0.5, 'decay': 'linear', 'decay_steps': 10}})
    assert config is not None and config.decay == 'linear'
    with pytest.raises(ValueError, match='decay'):
        mt.parse_model_teacher_config({'model': {'type': 'anima_refiner'}, 'teacher': {'loss_weight': 1, 'decay': 'x'}})
    # Plain Anima has no second form: the table is always this one.
    assert mt.parse_model_teacher_config({'model': {'type': 'anima'}, 'teacher': {'loss_weight': 1}}) is not None


def test_defaults_and_fields():
    config = parse()
    assert (config.loss_weight, config.decay, config.decay_steps, config.path) == (0.5, 'none', 0, None)
    assert config.enabled
    config = parse(decay='cosine', decay_steps=100, path='/m/other.safetensors')
    assert (config.decay, config.decay_steps, config.path) == ('cosine', 100, '/m/other.safetensors')


@pytest.mark.parametrize('table, match', [
    ({'loss_weight': -1}, 'loss_weight'),
    ({'loss_weight': True}, 'loss_weight'),
    ({'loss_weight': float('inf')}, 'loss_weight'),
    ({'decay': 'exp'}, 'decay'),
    ({'decay': 'linear'}, 'decay_steps'),
    ({'decay_steps': -1}, 'decay_steps'),
    ({'decay_steps': 1.5}, 'decay_steps'),
    ({'path': ''}, 'path'),
    ({'path': 3}, 'path'),
    ({'transformer_path': 'x'}, 'unknown keys'),
    ({'llm_path': 'x'}, 'unknown keys'),
    ({'shape': 'sigmoid'}, 'noise level'),
    ({'t_mid': 0.5, 'width': 0.1}, 'noise level'),
])
def test_bad_tables_are_refused(table, match):
    with pytest.raises(ValueError, match=match):
        parse(**table)
    with pytest.raises(ValueError, match='must be a table'):
        mt.parse_model_teacher_config({'model': {'type': 'flux'}, 'teacher': 3})


def test_block_swap_and_starting_adapter_are_refused():
    base = {'model': {'type': 'flux'}, 'teacher': {'loss_weight': 1}}
    with pytest.raises(ValueError, match='blocks_to_swap'):
        mt.parse_model_teacher_config({**base, 'blocks_to_swap': 4})
    adapter = {'adapter': {'type': 'lora', 'init_from_existing': '/a'}}
    with pytest.raises(ValueError, match='init_from_existing'):
        mt.parse_model_teacher_config({**base, **adapter})
    # With a path the teacher is not the adapter-less base, so there is nothing for the term to fight.
    with_path = {**base, 'teacher': {'loss_weight': 1, 'path': '/m'}, **adapter}
    assert mt.parse_model_teacher_config(with_path).path == '/m'


def test_warning_for_a_term_that_never_fades():
    config = parse()
    assert any("decay = 'none'" in m for m in mt.warnings_for(config, {}))
    assert mt.warnings_for(parse(decay='linear', decay_steps=10), {}) == []


def test_lambda_follows_the_step():
    teacher = mt.ModelTeacher(parse(loss_weight=2.0, decay='linear', decay_steps=10))
    assert teacher.lam() == pytest.approx(2.0 * (1 - 1 / 10))
    teacher.set_step(5)
    assert teacher.lam() == pytest.approx(1.0)
    teacher.set_step(10)
    assert teacher.lam() == 0.0
    teacher.set_step(50)
    assert teacher.lam() == 0.0
    assert mt.ModelTeacher(parse(loss_weight=0.25)).lam() == 0.25


# a toy model: three layers over a tuple, the way the pipeline hands activations on

class Embed(nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.proj = proj

    def forward(self, inputs):
        x, t = inputs
        return self.proj(x) + t[:, None], t


class Block(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, inputs):
        x, t = inputs
        return torch.tanh(self.lin(x)), t


class Head(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, inputs):
        x, t = inputs
        return self.out(x)


class Denoiser(nn.Module):
    def __init__(self, width=6, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(4, width)
        self.lin = nn.Linear(width, width)
        self.out = nn.Linear(width, 4)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)


class ToyPipeline(BasePipeline):
    name = 'toy'

    def __init__(self, seed=0):
        self.transformer = Denoiser(seed=seed)
        self.config = {}
        self.final_layer_module = 'student'

    def to_layers(self):
        t = self.transformer
        self.final_layer_module = t.out
        return [Embed(t.proj), Block(t.lin), Head(t.out)]


def student_forward(pipeline, features):
    x = features
    for layer in pipeline.to_layers():
        x = layer(x)
    return x


def make_features(batch=4, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 4, generator=gen), torch.rand(batch, generator=gen)


def test_teacher_layers_are_an_independent_frozen_copy():
    pipeline = ToyPipeline()
    layers = pipeline.make_teacher_layers()
    # What to_layers() assigns on the pipeline during the build stays on the copy.
    assert pipeline.final_layer_module == 'student'
    features = make_features()
    expected = student_forward(pipeline, features)
    with torch.no_grad():
        got = features
        for layer in layers:
            got = layer(got)
    assert torch.equal(got, expected)
    # Training the student does not move the teacher.
    with torch.no_grad():
        for p in pipeline.transformer.parameters():
            p.add_(1.0)
    with torch.no_grad():
        again = features
        for layer in layers:
            again = layer(again)
    assert torch.equal(again, got)
    assert not torch.equal(student_forward(pipeline, features), got)
    teacher_params = {id(p) for layer in layers for p in layer.parameters()}
    assert not teacher_params & {id(p) for p in pipeline.transformer.parameters()}


def test_attributes_to_copy_are_the_pipelines_to_name():
    class Wrapped(ToyPipeline):
        teacher_denoiser_attrs = ('transformer', 'extra')

        def __init__(self):
            super().__init__()
            self.extra = nn.Linear(2, 2)

    pipeline = Wrapped()
    teacher = copy.copy(pipeline)
    layers = pipeline.make_teacher_layers()
    assert layers and pipeline.extra is teacher.extra
    assert ComfyPipeline.teacher_denoiser_attrs == ('diffusion_model',)
    assert BasePipeline.teacher_denoiser_attrs == ('transformer',)


def test_a_path_goes_to_the_models_own_loader_on_a_copy():
    class WithLoader(ToyPipeline):
        def load_teacher_denoiser(self, path):
            self.transformer = Denoiser(seed=int(path))

    pipeline = WithLoader(seed=0)
    layers = pipeline.make_teacher_layers('7')
    features = make_features()
    with torch.no_grad():
        got = features
        for layer in layers:
            got = layer(got)
    assert torch.equal(got, student_forward(WithLoader(seed=7), features))
    assert not torch.equal(got, student_forward(pipeline, features))


def test_a_model_without_a_loader_refuses_a_path():
    from models.base import CommonPipeline

    class Bare(CommonPipeline):
        name = 'bare'
        teacher_denoiser_attrs = ('transformer',)

        def __init__(self):
            self.transformer = Denoiser()

        def to_layers(self):
            return [self.transformer.lin]

    with pytest.raises(NotImplementedError, match='path is not available for bare'):
        Bare().make_teacher_layers('/some/path')


class Shell:
    """Stands for a diffusers pipeline, which load_diffusion_model assigns the transformer on."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.vae = 'the student vae'


class ShellPipeline(BasePipeline):
    name = 'shell'
    teacher_path_key = 'weights'

    def __init__(self, seed=0):
        self.model_config = {'weights': '0', 'merge_adapters': ['/a'], 'dtype': torch.float32}
        self.config = {'model': self.model_config, 'adapter': {'rank': 4, 'fuse_adapters': [{'path': '/b'}]}}
        self.diffusers_pipeline = Shell(Denoiser(seed=seed))
        self.loads = []

    def __getattr__(self, name):
        if name == 'diffusers_pipeline':
            raise AttributeError(name)
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        self.loads.append((self.model_config['weights'], 'merge_adapters' in self.model_config,
                           dict(self.config['adapter'])))
        self.diffusers_pipeline.transformer = Denoiser(seed=int(self.model_config['weights']))

    def to_layers(self):
        t = self.transformer
        return [Embed(t.proj), Block(t.lin), Head(t.out)]


def test_a_base_pipeline_reads_the_teacher_with_its_own_loader_and_leaves_the_student_alone():
    pipeline = ShellPipeline(seed=0)
    student_shell, student = pipeline.diffusers_pipeline, pipeline.transformer
    layers = pipeline.make_teacher_layers('7')
    features = make_features()
    with torch.no_grad():
        got = features
        for layer in layers:
            got = layer(got)
        expected = student_forward(ShellPipeline(seed=7), features)
        assert torch.equal(got, expected)
    # Nothing of the student moved: not its shell, its transformer, its config, nor the loads it did.
    assert pipeline.diffusers_pipeline is student_shell and pipeline.transformer is student
    assert student_shell.transformer is student
    assert pipeline.model_config == {'weights': '0', 'merge_adapters': ['/a'], 'dtype': torch.float32}
    assert pipeline.config['adapter'] == {'rank': 4, 'fuse_adapters': [{'path': '/b'}]}
    # `loads` is shared with the copy: it records the teacher's load only (the path, no merges, no fuse_adapters).
    assert pipeline.loads == [('7', False, {'rank': 4})]
    # The teacher's loader read the path under the model's own key, without the adapters the student merges.
    teacher_params = {id(p) for layer in layers for p in layer.parameters()}
    assert not teacher_params & {id(p) for p in student.parameters()}
    # Another call, another teacher; the student is still what it was.
    again = pipeline.make_teacher_layers('9')
    assert again[0] is not layers[0] and [load[0] for load in pipeline.loads] == ['7', '9']
    assert pipeline.transformer is student


def test_the_teacher_loader_gets_the_path_the_key_and_no_adapter_merges():
    seen = []

    class Recording(ShellPipeline):
        def load_diffusion_model(self):
            seen.append((self.model_config['weights'], 'merge_adapters' in self.model_config,
                         dict(self.config['adapter']), self.config['model'] is self.model_config))
            super().load_diffusion_model()

    Recording().make_teacher_layers('3')
    assert seen == [('3', False, {'rank': 4}, True)]


def test_the_default_key_and_shell_are_transformer_path_and_diffusers_pipeline():
    assert BasePipeline.teacher_path_key == 'transformer_path'
    assert BasePipeline.teacher_shell_attrs == ('diffusers_pipeline',)


def test_a_loader_that_assigns_on_a_shell_needs_it_named():
    class Inner:
        net = None

    class OnModel(BasePipeline):
        name = 'on_model'
        teacher_shell_attrs = ('model',)

        def __init__(self):
            self.model = Inner()
            self.model.net = Denoiser(seed=0)
            self.model_config = {'transformer_path': 'x'}
            self.config = {'model': self.model_config}
            self.transformer = self.model.net

        def load_diffusion_model(self):
            self.model.net = Denoiser(seed=int(self.model_config['transformer_path']))
            self.transformer = self.model.net

        def to_layers(self):
            t = self.transformer
            return [Embed(t.proj), Block(t.lin), Head(t.out)]

    pipeline = OnModel()
    student = pipeline.model.net
    pipeline.make_teacher_layers('5')
    assert pipeline.model.net is student and pipeline.transformer is student


def test_comfy_pipelines_load_the_teacher_with_their_own_loader():
    calls = []

    class Fake(ComfyPipeline):
        name = 'fake'

        def __init__(self):
            self.model_config = {'diffusion_model': '/student.safetensors', 'dtype': torch.float32}
            self.config = {'model': self.model_config}
            self.diffusion_model = nn.Linear(2, 2)

        def load_diffusion_model(self):
            calls.append(self.model_config['diffusion_model'])
            self.diffusion_model = nn.Linear(2, 2)

        def to_layers(self):
            return [self.diffusion_model]

    pipeline = Fake()
    student = pipeline.diffusion_model
    layers = pipeline.make_teacher_layers('/teacher.safetensors')
    assert calls == ['/teacher.safetensors']
    assert layers[0] is not student
    assert pipeline.diffusion_model is student
    assert pipeline.model_config['diffusion_model'] == '/student.safetensors'
    assert pipeline.config['model']['diffusion_model'] == '/student.safetensors'
    # Without a path it is a deep copy of the student, not a second load.
    layers = pipeline.make_teacher_layers()
    assert calls == ['/teacher.safetensors']
    assert layers[0] is not student and torch.equal(layers[0].weight, student.weight)


def test_default_selection_and_tuple_outputs():
    pipeline = ToyPipeline()
    out = torch.zeros(2)
    assert pipeline.teacher_targets_of(out) == (out,)
    with pytest.raises(NotImplementedError, match='teacher_targets_of'):
        pipeline.teacher_targets_of((out, out))
    assert pipeline.teacher_unsupported() is None


def test_build_teacher_places_layers_only_where_it_runs():
    pipeline = ToyPipeline()
    config = parse()
    elsewhere = mt.build_teacher(pipeline, config, torch.device('cpu'), runs_teacher=False)
    assert elsewhere.layers is None
    with pytest.raises(RuntimeError, match='first pipeline stage'):
        elsewhere.run(make_features())
    here = mt.build_teacher(pipeline, config, torch.device('cpu'), runs_teacher=True)
    assert all(not p.requires_grad for layer in here.layers for p in layer.parameters())
    assert all(not layer.training for layer in here.layers)
    outputs = here.run(make_features())
    assert len(outputs) == 1 and not outputs[0].requires_grad
    assert torch.equal(outputs[0], student_forward(pipeline, make_features()))


def test_on_first_stage_without_distributed_is_true():
    assert mt.on_first_stage(1) and mt.on_first_stage(4)


# the loss

def mse_loss_fn(output, label):
    target, mask = label
    loss = F.mse_loss(output.float(), target.float(), reduction='none')
    if mask.numel() > 0:
        loss = loss * mask
    return loss.mean()


def test_loss_is_the_data_loss_plus_lambda_times_the_loss_toward_the_teacher():
    teacher = mt.ModelTeacher(parse(loss_weight=0.5))
    loss_fn = mt.wrap_loss_fn(mse_loss_fn, teacher)
    output = torch.randn(4, 4, requires_grad=True)
    target, teacher_target = torch.randn(4, 4), torch.randn(4, 4)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])[:, None].expand(4, 4)
    loss = loss_fn(output, (target, mask, teacher_target))
    expected = mse_loss_fn(output, (target, mask)) + 0.5 * mse_loss_fn(output, (teacher_target, mask))
    assert torch.allclose(loss, expected)
    loss.backward()
    # Only the student's output carries a gradient.
    reference = output.detach().clone().requires_grad_(True)
    (mse_loss_fn(reference, (target, mask)) + 0.5 * mse_loss_fn(reference, (teacher_target, mask))).backward()
    assert torch.allclose(output.grad, reference.grad)


def test_evaluation_is_the_plain_loss():
    loss_fn = mt.wrap_loss_fn(mse_loss_fn, mt.ModelTeacher(parse()))
    output, target = torch.randn(3, 4), torch.randn(3, 4)
    empty = torch.tensor([])
    with torch.no_grad():
        assert torch.equal(loss_fn(output, (target, empty)), mse_loss_fn(output, (target, empty)))


def test_a_training_label_without_teacher_targets_is_an_error():
    loss_fn = mt.wrap_loss_fn(mse_loss_fn, mt.ModelTeacher(parse()))
    output, target = torch.randn(3, 4, requires_grad=True), torch.randn(3, 4)
    with pytest.raises(RuntimeError, match='no teacher targets'):
        loss_fn(output, (target, torch.tensor([])))


def test_a_faded_out_term_costs_nothing():
    teacher = mt.ModelTeacher(parse(decay='linear', decay_steps=4))
    teacher.set_step(4)
    calls = []

    def inner(output, label):
        calls.append(len(label))
        return mse_loss_fn(output, label)

    loss_fn = mt.wrap_loss_fn(inner, teacher)
    output = torch.randn(3, 4, requires_grad=True)
    loss_fn(output, (torch.randn(3, 4), torch.tensor([]), torch.randn(3, 4)))
    assert calls == [2]


def test_two_targets_give_two_teacher_targets_in_order():
    seen = []

    def inner(output, label):
        seen.append(label)
        return output.sum() * 0

    loss_fn = mt.wrap_loss_fn(inner, mt.ModelTeacher(parse()))
    a, b, mask, ta, tb = (torch.full((2,), float(i)) for i in range(5))
    loss_fn(torch.zeros(2, requires_grad=True), (a, b, mask, ta, tb))
    assert [tuple(x.tolist()) for x in seen[0]] == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert [tuple(x.tolist()) for x in seen[1]] == [(3.0, 3.0), (4.0, 4.0), (2.0, 2.0)]


# the data loader

class FakeDataset:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


class FakeModel:
    """prepare_inputs as a pipeline does it: features for the first layer, (target, mask) for the loss."""

    def __init__(self, noise_seed=0):
        self.noise_seed = noise_seed

    def prepare_inputs(self, batch, timestep_quantile=None):
        x = batch['x']
        t = batch['t']
        return (x, t), (x * 2.0, None)


class SingleProcessEngine:
    is_pipe_parallel = False


def make_loader(batches, gas, teacher, engine=None, num_batches=None):
    return dataset_util.PipelineDataLoader(
        FakeDataset(batches), engine or SingleProcessEngine(), gas, FakeModel(), num_dataloader_workers=0,
        teacher=teacher)


def batch_of(size, seed):
    x, t = make_features(size, seed)
    return {'x': x, 't': t}


def toy_teacher(pipeline, **table):
    return mt.build_teacher(pipeline, parse(**table), torch.device('cpu'), runs_teacher=True)


def test_the_label_of_a_training_batch_carries_the_teacher_output():
    pipeline = ToyPipeline()
    teacher = toy_teacher(pipeline)
    batch = batch_of(6, seed=3)
    loader = make_loader([batch], gas=3, teacher=teacher)
    pulled = [next(loader) for _ in range(3)]
    for i, (features, label) in enumerate(pulled):
        assert len(label) == 3
        target, mask, teacher_target = label
        rows = slice(2 * i, 2 * i + 2)
        assert torch.equal(target, batch['x'][rows] * 2.0)
        expected = student_forward(pipeline, (batch['x'][rows], batch['t'][rows]))
        assert torch.allclose(teacher_target, expected)
        assert teacher_target.dtype == target.dtype


def test_the_teacher_sees_micro_batches_not_the_whole_batch():
    pipeline = ToyPipeline()
    teacher = toy_teacher(pipeline)
    sizes = []
    original = teacher.run

    def run(features):
        sizes.append(features[0].shape[0])
        return original(features)

    teacher.run = run
    loader = make_loader([batch_of(8, seed=4)], gas=4, teacher=teacher)
    next(loader)
    assert sizes == [2, 2, 2, 2]


def test_without_a_teacher_the_label_is_unchanged():
    loader = make_loader([batch_of(4, seed=5)], gas=2, teacher=None)
    _, label = next(loader)
    assert len(label) == 2


def test_a_teacher_output_of_the_wrong_shape_or_count_stops_the_run():
    pipeline = ToyPipeline()
    teacher = toy_teacher(pipeline)
    teacher.select = lambda out: (out[:, None],)
    with pytest.raises(RuntimeError, match='does not match the target'):
        next(make_loader([batch_of(4, seed=6)], gas=2, teacher=teacher))
    teacher = toy_teacher(pipeline)
    teacher.select = lambda out: (out, out)
    with pytest.raises(RuntimeError, match='2 output tensors and the label has 1 targets'):
        next(make_loader([batch_of(4, seed=6)], gas=2, teacher=teacher))


def test_the_teacher_output_is_cast_to_the_target_dtype():
    pipeline = ToyPipeline()
    teacher = toy_teacher(pipeline)
    original = teacher.select
    teacher.select = lambda out: tuple(t.to(torch.bfloat16) for t in original(out))
    _, label = next(make_loader([batch_of(4, seed=7)], gas=2, teacher=teacher))
    assert label[2].dtype == label[0].dtype == torch.float32


# two pipeline stages: the first computes, the last receives

def _two_stage_worker(rank, init_file, out_dir):
    import os
    os.environ.setdefault('DS_ACCELERATOR', 'cpu')
    os.environ['LOCAL_RANK'] = str(rank)
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=2, verbose=False)
    dataset_util._broadcast_device = lambda: 'cpu'
    grid = types.SimpleNamespace(pp_group=[0, 1], stage_to_global=lambda stage: stage)
    engine = types.SimpleNamespace(is_pipe_parallel=True, num_stages=2, grid=grid,
                                   is_first_stage=lambda: rank == 0, is_last_stage=lambda: rank == 1)
    pipeline = ToyPipeline()
    teacher = mt.build_teacher(pipeline, parse(), torch.device('cpu'), runs_teacher=(rank == 0))
    loader = make_loader([batch_of(4, seed=8), batch_of(4, seed=9)], gas=2, teacher=teacher, engine=engine)
    results = []
    for _ in range(4):
        features, label = next(loader)
        results.append(label)
    torch.save({'labels': results}, f'{out_dir}/rank{rank}.pt')
    dist.barrier()
    dist.destroy_process_group()


def test_the_last_stage_receives_what_the_first_computed(tmp_path):
    mp.spawn(_two_stage_worker, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    first = torch.load(tmp_path / 'rank0.pt')['labels']
    last = torch.load(tmp_path / 'rank1.pt')['labels']
    assert len(first) == len(last) == 4
    pipeline = ToyPipeline()
    for (t0, m0, teacher0), (t1, m1, teacher1) in zip(first, last):
        assert torch.equal(t0, t1)
        assert torch.equal(teacher0, teacher1)
    # The teacher output equals the chain run on that micro batch's features.
    for index, seed in enumerate((8, 8, 9, 9)):
        batch = batch_of(4, seed)
        rows = slice(0, 2) if index % 2 == 0 else slice(2, 4)
        expected = student_forward(pipeline, (batch['x'][rows], batch['t'][rows]))
        assert torch.allclose(last[index][2], expected)


# robustness found while reading the other pipelines

def test_a_pipeline_that_forwards_unknown_attributes_can_build_a_teacher():
    class Proxy(ToyPipeline):
        # AuraFlow, Wan, HunyuanVideo and SDXL: attributes missing on the pipeline are looked up on diffusers_pipeline.
        def __init__(self):
            super().__init__()
            self.diffusers_pipeline = types.SimpleNamespace(marker='inner')

        def __getattr__(self, name):
            return getattr(self.diffusers_pipeline, name)

    pipeline = Proxy()
    assert pipeline.marker == 'inner'
    layers = pipeline.make_teacher_layers()
    assert len(layers) == 3


def test_a_plain_function_layer_is_fine():
    class WithFunction(ToyPipeline):
        def to_layers(self):
            layers = super().to_layers()
            return [layers[0], lambda inputs: inputs, *layers[1:]]

    pipeline = WithFunction()
    teacher = mt.build_teacher(pipeline, parse(), torch.device('cpu'), runs_teacher=True)
    assert torch.equal(teacher.run(make_features())[0], student_forward(ToyPipeline(), make_features()))


def test_features_are_moved_to_the_teachers_device():
    seen = []

    def layer(inputs):
        seen.append({t.device.type for t in inputs if torch.is_tensor(t)})
        return inputs[0]

    teacher = mt.ModelTeacher(parse(), layers=[layer], select=lambda out: (out,), device=torch.device('meta'))
    teacher.run((torch.zeros(2, 3), torch.zeros(2), None))
    assert seen == [{'meta'}]


def test_fit_trailing_dim_only_touches_a_differing_last_dimension():
    target = torch.zeros(2, 3, 5)
    longer = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    shorter = torch.ones(2, 3, 2)
    assert torch.equal(mt.fit_trailing_dim(longer, target), longer[..., :5])
    padded = mt.fit_trailing_dim(shorter, target)
    assert padded.shape == target.shape and torch.equal(padded[..., :2], shorter) and not padded[..., 2:].any()
    same = torch.ones(2, 3, 5)
    assert mt.fit_trailing_dim(same, target) is same
    other = torch.ones(2, 4, 7)
    assert mt.fit_trailing_dim(other, target) is other
    # One micro batch against the whole batch's target.
    assert mt.fit_trailing_dim(torch.ones(1, 3, 7), target).shape == (1, 3, 5)
    assert mt.fit_trailing_dim(torch.ones(2, 3), target).shape == (2, 3)


def test_a_teacher_audio_output_shorter_than_the_padded_audio_target_is_padded_for_the_label():
    pipeline = ToyPipeline()
    teacher = toy_teacher(pipeline)
    original = teacher.select
    teacher.select = lambda out: (original(out)[0][:, :3],)

    class AudioModel(FakeModel):
        def prepare_inputs(self, batch, timestep_quantile=None):
            features, (target, mask) = super().prepare_inputs(batch)
            return features, (F.pad(target, (0, 1)), mask)

    loader = dataset_util.PipelineDataLoader(FakeDataset([batch_of(4, seed=6)]), SingleProcessEngine(), 2, AudioModel(),
                                             num_dataloader_workers=0, teacher=teacher)
    # The teacher output is (B, 3) against a (B, 5) target: padded, not an error.
    _, label = next(loader)
    assert label[2].shape == label[0].shape and not label[2][:, 3:].any()


# every pipeline in the repo, read from the source (several cannot be imported without a GPU or flash_attn)

from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / 'models'


def pipeline_classes():
    classes = {}
    for path in sorted([*MODELS.glob('*.py'), *MODELS.glob('wan/*.py')]):
        tree = ast.parse(path.read_text())
        source = path.read_text()
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith('Pipeline'):
                methods = {n.name: ast.get_source_segment(source, n) for n in node.body
                           if isinstance(n, ast.FunctionDef)}
                attrs = None
                for n in node.body:
                    if (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'teacher_denoiser_attrs'
                                                          for t in n.targets)):
                        attrs = ast.literal_eval(n.value)
                consts = {}
                for n in node.body:
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                        try:
                            consts[n.targets[0].id] = ast.literal_eval(n.value)
                        except ValueError:
                            pass
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                classes[node.name] = {'bases': bases, 'methods': methods, 'attrs': attrs, 'consts': consts,
                                      'file': path.name}
    return classes


def resolve(classes, name, field):
    """The nearest definition of a method, or of teacher_denoiser_attrs, up the base classes."""
    node = classes.get(name)
    if node is None:
        return None
    if field == 'attrs':
        if node['attrs'] is not None:
            return node['attrs']
    elif field in node['methods']:
        return node['methods'][field]
    for base in node['bases']:
        found = resolve(classes, base, field)
        if found is not None:
            return found
    return None


MODEL_PIPELINES = sorted(name for name in pipeline_classes()
                         if name not in ('CommonPipeline', 'BasePipeline', 'ComfyPipeline', 'MossImagePipeline'))


def test_every_pipeline_was_found():
    assert len(MODEL_PIPELINES) >= 25
    assert {'FluxPipeline', 'WanPipeline', 'SDXLPipeline', 'QwenImage21Pipeline', 'PH3ImagePipeline'} <= set(MODEL_PIPELINES)


@pytest.mark.parametrize('name', MODEL_PIPELINES)
def test_to_layers_wraps_what_the_teacher_copies(name):
    classes = pipeline_classes()
    classes['BasePipeline'] = {'bases': [], 'methods': {}, 'attrs': ('transformer',), 'file': 'base.py'}
    classes['ComfyPipeline'] = {'bases': [], 'methods': {}, 'attrs': ('diffusion_model',), 'file': 'base.py'}
    attrs = resolve(classes, name, 'attrs')
    to_layers = resolve(classes, name, 'to_layers')
    assert attrs, f'{name} has no teacher_denoiser_attrs'
    assert to_layers, f'{name} has no to_layers'
    for attr in attrs:
        assert f'self.{attr}' in to_layers, f'{name}.to_layers() does not read self.{attr}, which the teacher copies'


def test_models_whose_layers_return_more_than_the_prediction_say_which_part_is_the_target():
    classes = pipeline_classes()
    defining = {name for name, node in classes.items() if 'teacher_targets_of' in node['methods']}
    assert defining - {'CommonPipeline'} == {'CosmosPipeline', 'SDXLPipeline', 'MinimaxH3Pipeline', 'PH3ImagePipeline',
                                              'MiniMaxH3ImagePipeline'}


def own_definition(classes, name, field):
    """The method as the nearest class below the base pipelines defines it, or None."""
    node = classes.get(name)
    if node is None or name in ('BasePipeline', 'CommonPipeline', 'ComfyPipeline'):
        return None
    if field in node['methods']:
        return node['methods'][field]
    for base in node['bases']:
        found = own_definition(classes, base, field)
        if found is not None:
            return found
    return None


def class_constant(classes, name, field, default):
    """A class attribute set to a literal, from the nearest class that sets it (the base pipelines' too)."""
    node = classes.get(name)
    if node is None:
        return default
    if field in node['consts']:
        return node['consts'][field]
    for base in node['bases']:
        value = class_constant(classes, base, field, None)
        if value is not None:
            return value
    return default


def is_base_pipeline(classes, name):
    node = classes.get(name)
    return node is not None and ('BasePipeline' in node['bases'] or any(is_base_pipeline(classes, b) for b in node['bases']))


BASE_MODELS = [n for n in MODEL_PIPELINES if is_base_pipeline(pipeline_classes(), n)]


def test_the_base_pipeline_models_were_found():
    assert {'FluxPipeline', 'WanPipeline', 'SDXLPipeline', 'IrisPipeline', 'HiDreamPipeline', 'Lumina2Pipeline',
            'CosmosPipeline', 'CosmosPredict2Pipeline', 'ChromaPipeline'} <= set(BASE_MODELS)
    assert 'QwenImage21Pipeline' not in BASE_MODELS and 'MossImagePipeline' not in BASE_MODELS


@pytest.mark.parametrize('name', BASE_MODELS)
def test_every_base_pipeline_model_can_read_a_teacher_path(name):
    classes = pipeline_classes()
    if own_definition(classes, name, 'load_teacher_denoiser'):
        return
    # Otherwise the generic loader runs the model's load_diffusion_model on a copy: that has to read the key
    # the model names, and every object it assigns the denoiser on has to be one the copy replaces.
    loader = own_definition(classes, name, 'load_diffusion_model')
    assert loader, f'{name} defines neither load_teacher_denoiser nor load_diffusion_model'
    key = class_constant(classes, name, 'teacher_path_key', 'transformer_path')
    shells = class_constant(classes, name, 'teacher_shell_attrs', ('diffusers_pipeline',))
    assert f"'{key}'" in loader, f'{name}.load_diffusion_model does not read [model] {key}'
    tree = ast.parse(textwrap.dedent(loader))
    for node in ast.walk(tree):
        for target in getattr(node, 'targets', []):
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name) and target.value.value.id == 'self'):
                assert target.value.attr in shells, (
                    f'{name}.load_diffusion_model assigns self.{target.value.attr}.{target.attr}, so '
                    f'{target.value.attr} has to be in teacher_shell_attrs (the teacher would change the student\'s)')


def test_models_that_cannot_reuse_load_diffusion_model_have_their_own_teacher_loader():
    classes = pipeline_classes()
    own = {n for n in BASE_MODELS if own_definition(classes, n, 'load_teacher_denoiser')}
    assert own == {'FluxPipeline', 'HiDreamPipeline', 'Lumina2Pipeline', 'SDXLPipeline', 'IrisPipeline',
                   'WanPipeline', 'CosmosPredict2Pipeline'}
    # The ones that read another attribute than the config's key set it first, then use the generic loader.
    for name in ('IrisPipeline', 'WanPipeline', 'CosmosPredict2Pipeline'):
        assert 'super().load_teacher_denoiser(path)' in own_definition(classes, name, 'load_teacher_denoiser')
    # The ones whose denoiser is read in the constructor, or next to a text encoder, build a clone of the shell
    # (or a whole pipeline) before assigning on it.
    for name in ('FluxPipeline', 'HiDreamPipeline'):
        assert 'shallow_clone(self.diffusers_pipeline)' in own_definition(classes, name, 'load_teacher_denoiser')
    # Their modules cannot be imported without a GPU: the lines that matter, read from the source.
    assert 'self.transformer_path = Path(path)' in own_definition(classes, 'WanPipeline', 'load_teacher_denoiser')
    lumina = own_definition(classes, 'Lumina2Pipeline', 'load_teacher_denoiser')
    assert 'self.transformer = self.load_transformer(path, self.text_encoder.config.hidden_size)' in lumina


def test_the_keys_that_are_not_transformer_path():
    classes = pipeline_classes()
    keys = {n: class_constant(classes, n, 'teacher_path_key', 'transformer_path') for n in BASE_MODELS}
    assert {n: k for n, k in keys.items() if k != 'transformer_path'} == {
        'OmniGen2Pipeline': 'diffusers_path', 'SD3Pipeline': 'diffusers_path', 'LTXVideoPipeline': 'single_file_path'}
    assert class_constant(classes, 'CosmosPipeline', 'teacher_shell_attrs', None) == ('model',)


def test_flux_reads_the_teacher_transformer_from_the_path_on_a_clone_of_the_pipeline(monkeypatch):
    from models import flux
    pipe = flux.FluxPipeline.__new__(flux.FluxPipeline)
    pipe.model_config = {'dtype': torch.float32, 'transformer_dtype': torch.bfloat16}
    student = nn.Sequential(nn.Linear(2, 2))
    shell = Shell(student)
    pipe.diffusers_pipeline = shell
    read = []

    def fake_load(self, path):
        read.append(path)
        return nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))

    monkeypatch.setattr(flux.FluxPipeline, 'load_transformer', fake_load)
    teacher = flux.shallow_clone(pipe)
    teacher.load_teacher_denoiser('/other.safetensors')
    assert read == ['/other.safetensors']
    assert pipe.diffusers_pipeline is shell and shell.transformer is student
    assert teacher.diffusers_pipeline is not shell and teacher.diffusers_pipeline.transformer is not student
    # The same dtype handling as the student's: the weights go to transformer_dtype, 1-D tensors stay as loaded.
    new = teacher.diffusers_pipeline.transformer
    assert new[0].weight.dtype == torch.bfloat16 and new[1].weight.dtype == torch.float32


def test_sdxl_reads_a_whole_second_pipeline_and_keeps_the_students_vae_and_scheduler(monkeypatch):
    from models import sdxl
    pipe = sdxl.SDXLPipeline.__new__(sdxl.SDXLPipeline)
    pipe.model_config = {'dtype': torch.float16}
    student = types.SimpleNamespace(vae='student vae', scheduler='student scheduler', unet='student unet')
    pipe.diffusers_pipeline = student
    calls = []

    class Loaded(types.SimpleNamespace):
        pass

    def fake_from_single_file(path, **kwargs):
        calls.append((path, kwargs))
        return Loaded(vae='teacher vae', scheduler='teacher scheduler', unet='teacher unet', tokenizer='tok',
                      tokenizer_2='tok2')

    monkeypatch.setattr(sdxl.diffusers.StableDiffusionXLPipeline, 'from_single_file', staticmethod(fake_from_single_file))
    teacher = pipe.__class__.__new__(pipe.__class__)
    teacher.__dict__.update(pipe.__dict__)
    teacher.load_teacher_denoiser('/other.safetensors')
    assert calls == [('/other.safetensors', {'torch_dtype': torch.float16, 'add_watermarker': False})]
    loaded = teacher.diffusers_pipeline
    assert loaded.unet == 'teacher unet' and loaded.vae == 'student vae' and loaded.scheduler == 'student scheduler'
    assert loaded.tokenizer_2 == loaded.tokenizer == 'tok'
    assert pipe.diffusers_pipeline is student and student.unet == 'student unet'


def test_iris_points_the_attribute_its_loader_reads_at_the_path(monkeypatch):
    # (Wan does the same with transformer_path; its module cannot be imported without a GPU, so the structural
    # test above checks it.)
    from models import base, iris
    seen = []
    monkeypatch.setattr(base.BasePipeline, 'load_teacher_denoiser', lambda self, path: seen.append(path))
    pipe = iris.IrisPipeline.__new__(iris.IrisPipeline)
    pipe.transformer_path = '/student'
    teacher = base.shallow_clone(pipe)
    teacher.load_teacher_denoiser('/other')
    assert teacher.transformer_path == '/other' and pipe.transformer_path == '/student' and seen == ['/other']


def test_a_teacher_path_without_a_trained_refiner_is_refused(monkeypatch):
    from models import base
    from models import cosmos_predict2 as cp2

    def fake_load(self, path):
        self._refiner_is_fresh = True

    monkeypatch.setattr(base.BasePipeline, 'load_teacher_denoiser', fake_load)
    pipe = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
    pipe.use_context_refiner = True
    pipe._refiner_is_fresh = False
    with pytest.raises(ValueError, match='no trained context_refiner'):
        pipe.load_teacher_denoiser('/other')
    # The student's own fresh refiner does not matter to a teacher read from a path.
    monkeypatch.setattr(base.BasePipeline, 'load_teacher_denoiser', lambda self, path: None)
    pipe._refiner_is_fresh = True
    pipe.load_teacher_denoiser('/other')
    assert pipe._refiner_is_fresh is False
    monkeypatch.setattr(base.CommonPipeline, 'make_teacher_layers', lambda self, path=None: 'layers')
    pipe._refiner_is_fresh = True
    assert pipe.make_teacher_layers('/other') == 'layers'


def test_a_fresh_refiner_cannot_be_its_own_teacher(monkeypatch):
    from models import base
    from models import cosmos_predict2 as cp2
    monkeypatch.setattr(base.CommonPipeline, 'make_teacher_layers', lambda self, path=None: 'layers')
    pipe = cp2.CosmosPredict2Pipeline.__new__(cp2.CosmosPredict2Pipeline)
    pipe.use_context_refiner = True
    pipe._refiner_is_fresh = True
    with pytest.raises(ValueError, match='freshly initialised context_refiner'):
        pipe.make_teacher_layers()
    pipe._refiner_is_fresh = False
    assert pipe.make_teacher_layers() == 'layers'
    # Plain Anima has no refiner to be fresh.
    pipe.use_context_refiner = False
    pipe._refiner_is_fresh = True
    assert pipe.make_teacher_layers() == 'layers'
