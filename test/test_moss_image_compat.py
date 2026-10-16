"""comfyui/moss_image_compat: the MOSS-Image node pack on ComfyUI 3216c62.

The fake classes below have 3216c62's ModelPatcher / ModelPatcherDynamic signatures. The
end-to-end test loads the node the way ComfyUI does, in a fresh process (it patches ComfyUI
classes), and builds a model with the pack's own loader; run at both ComfyUI pins.
"""

import importlib.util
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NODE = REPO / 'comfyui' / 'moss_image_compat' / '__init__.py'
PACK = REPO / 'submodules' / 'MOSS-Image' / 'comfyui' / 'moss-image'


@pytest.fixture
def compat(monkeypatch):
    """The node's helpers, without running its import-time install / pack load."""
    source = NODE.read_text(encoding='utf-8')
    head = source[:source.index('\n_patched = install()')]
    module = types.ModuleType('moss_image_compat_helpers')
    module.__file__ = str(NODE)
    exec(compile(head, str(NODE), 'exec'), module.__dict__)
    return module


def old_patchers():
    calls = []

    class ModelPatcher:
        def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
            calls.append(('init', type(self).__name__, model, load_device, offload_device, size, weight_inplace_update))

    class ModelPatcherDynamic(ModelPatcher):
        def __new__(cls, model=None, load_device=None, offload_device=None, size=0, weight_inplace_update=False):
            if load_device == 'cpu':
                return ModelPatcher(model, load_device, offload_device, size, weight_inplace_update)
            return super().__new__(cls)

        def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
            super().__init__(model, load_device, offload_device, size, weight_inplace_update)

    return ModelPatcher, ModelPatcherDynamic, calls


class TestFastDisk:
    def test_old_patchers_take_and_drop_fast_disk(self, compat):
        ModelPatcher, ModelPatcherDynamic, calls = old_patchers()
        with pytest.raises(TypeError):
            ModelPatcher('m', 'cuda', 'cpu', fast_disk=True)
        assert compat.accept_fast_disk(ModelPatcher)
        assert compat.accept_fast_disk(ModelPatcherDynamic)
        ModelPatcher('m', 'cuda', 'cpu', 5, weight_inplace_update=True, fast_disk=True)
        dynamic = ModelPatcherDynamic('m', 'cuda', 'cpu', fast_disk=False)
        rerouted = ModelPatcherDynamic('m', 'cpu', 'cpu', fast_disk=True)
        assert type(dynamic) is ModelPatcherDynamic and type(rerouted) is ModelPatcher
        assert calls == [('init', 'ModelPatcher', 'm', 'cuda', 'cpu', 5, True),
                         ('init', 'ModelPatcherDynamic', 'm', 'cuda', 'cpu', 0, False),
                         ('init', 'ModelPatcher', 'm', 'cpu', 'cpu', 0, False)]

    def test_patching_is_idempotent(self, compat):
        ModelPatcher, ModelPatcherDynamic, calls = old_patchers()
        assert compat.accept_fast_disk(ModelPatcherDynamic)
        init = ModelPatcherDynamic.__init__
        assert not compat.accept_fast_disk(ModelPatcherDynamic)
        assert ModelPatcherDynamic.__init__ is init

    def test_a_patcher_that_takes_fast_disk_is_left_alone(self, compat):
        class ModelPatcher:
            def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False,
                         fast_disk=False):
                self.fast_disk = fast_disk
        init = ModelPatcher.__init__
        assert not compat.accept_fast_disk(ModelPatcher)
        assert ModelPatcher.__init__ is init
        assert ModelPatcher('m', 'cuda', 'cpu', fast_disk=True).fast_disk is True


class TestStorage:
    def test_added_only_where_missing(self, compat, monkeypatch):
        package = types.ModuleType('comfy')
        monkeypatch.delitem(sys.modules, 'comfy.storage', raising=False)
        monkeypatch.setattr(compat.importlib.util, 'find_spec', lambda name: None)
        assert compat.install_storage(package)
        storage = sys.modules['comfy.storage']
        assert package.storage is storage
        assert storage.state_dict_fast_disk({'w': object()}) is False
        assert storage.model_fast_disk(['/x']) is False
        assert storage.fast_storage('/x') is None
        assert storage.annotate_state_dict({}, '/x') is None
        # Present now, so not added twice.
        assert not compat.install_storage(package)

    def test_a_real_comfy_storage_is_kept(self, compat, monkeypatch):
        monkeypatch.delitem(sys.modules, 'comfy.storage', raising=False)
        monkeypatch.setattr(compat.importlib.util, 'find_spec', lambda name: object())
        package = types.ModuleType('comfy')
        assert not compat.install_storage(package)
        assert not hasattr(package, 'storage')


class TestFindPack:
    def test_search_order(self, compat, monkeypatch, tmp_path):
        monkeypatch.delenv('MOSS_IMAGE_NODE_PATH', raising=False)
        here = NODE.parent
        assert compat.pack_candidates() == [here / 'moss-image', REPO / 'submodules/MOSS-Image/comfyui/moss-image']
        assert compat.find_pack() == PACK.resolve()
        env = tmp_path / 'pack'
        env.mkdir()
        (env / '__init__.py').write_text('')
        monkeypatch.setenv('MOSS_IMAGE_NODE_PATH', str(env))
        assert compat.find_pack() == env.resolve()

    def test_a_wrong_explicit_path_is_an_error(self, compat, monkeypatch, tmp_path):
        monkeypatch.setenv('MOSS_IMAGE_NODE_PATH', str(tmp_path / 'nowhere'))
        with pytest.raises(RuntimeError, match='MOSS_IMAGE_NODE_PATH=.*is not the MOSS-Image node pack'):
            compat.find_pack()

    def test_nothing_found_lists_what_was_tried(self, compat, monkeypatch, tmp_path):
        monkeypatch.delenv('MOSS_IMAGE_NODE_PATH', raising=False)
        monkeypatch.setattr(compat, 'pack_candidates', lambda: [tmp_path / 'a', tmp_path / 'b'])
        with pytest.raises(RuntimeError, match=r'(?s)Tried:.*a.*b.*MOSS_IMAGE_NODE_PATH'):
            compat.find_pack()


END_TO_END = textwrap.dedent('''
    import asyncio, importlib.util, sys
    sys.path.insert(0, {test!r})
    import conftest  # CPU-only ComfyUI import shims
    sys.path.insert(0, {comfy!r})
    import torch
    import comfy.model_patcher

    spec = importlib.util.spec_from_file_location('moss_image_compat', {node!r},
                                                  submodule_search_locations=[{node_dir!r}])
    node = importlib.util.module_from_spec(spec)
    sys.modules['moss_image_compat'] = node
    spec.loader.exec_module(node)
    print('PATCHED', sorted(node._patched))
    extension = asyncio.run(node.comfy_entrypoint())
    print('NODES', sorted(n.GET_SCHEMA().node_id for n in asyncio.run(extension.get_node_list())))

    import comfy.ops
    pack = sys.modules['moss_image_compat_pack']
    model_module = sys.modules['moss_image_compat_pack.model']
    loaders = sys.modules['moss_image_compat_pack.loaders']
    torch.manual_seed(0)
    reference = model_module.MOSSImageTransformer(dim=64, ffn_dim=96, num_heads=2, num_layers=2, text_dim=4096,
                                                  freq_dim=256, latent_channels=128, dtype=torch.float32,
                                                  operations=comfy.ops.manual_cast)
    with torch.no_grad():
        for p in reference.parameters():
            p.copy_(0.05 * torch.randn_like(p))
    state = {{k: v.clone() for k, v in reference.state_dict().items()}}
    patcher = loaders.load_model_state(dict(state), torch.float32, False)
    print('PATCHER', type(patcher).__name__)
    x = torch.randn(1, 128, 4, 4)
    t = torch.tensor([500.0])
    text = torch.randn(1, 5, 4096)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        expected = reference(x, t, text, attention_mask=mask)
        got = patcher.model.diffusion_model(x, t, text, attention_mask=mask)
    print('MAXDIFF', float((expected - got).abs().max()))
''')


def test_the_pack_loads_and_builds_a_model_through_the_node(tmp_path):
    script = tmp_path / 'run.py'
    script.write_text(END_TO_END.format(test=str(REPO / 'test'), comfy=str(REPO / 'submodules' / 'ComfyUI'),
                                        node=str(NODE), node_dir=str(NODE.parent)))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600,
                            cwd=str(tmp_path))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    lines = dict(line.split(' ', 1) for line in result.stdout.splitlines()
                 if line.split(' ', 1)[0] in ('PATCHED', 'NODES', 'PATCHER', 'MAXDIFF'))
    assert eval(lines['NODES']) == sorted(['MOSSImageUNETLoader', 'MOSSImageCLIPLoader', 'MOSSImageHFLoader',
                                           'MOSSImageHFDiTLoader', 'MOSSImageHFTextEncoderLoader',
                                           'MOSSImageHFVAELoader'])
    assert float(lines['MAXDIFF']) < 1e-5
    has_storage = (REPO / 'submodules' / 'ComfyUI' / 'comfy' / 'storage.py').exists()
    patched = eval(lines['PATCHED'])
    if has_storage:
        assert patched == []
    else:
        assert patched == ['ModelPatcher(fast_disk=)', 'ModelPatcherDynamic(fast_disk=)', 'comfy.storage']
