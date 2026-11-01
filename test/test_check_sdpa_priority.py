"""CPU checks of tools/check_sdpa_priority.py's own logic; the check itself needs a GPU."""

import importlib.util
import types
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parent.parent / 'tools' / 'check_sdpa_priority.py'


@pytest.fixture(scope='module')
def tool():
    spec = importlib.util.spec_from_file_location('check_sdpa_priority', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_are_a_moss_image_caption(tool):
    args = tool.parse_args([])
    assert (args.tokens, args.heads, args.head_dim, args.batch_size) == ([97], 24, 128, 1)


def test_shapes_comfyui_runs_outside_its_context_are_refused(tool):
    # 24 heads * 32 tokens * 128 = 98304 query elements: ComfyUI skips its priority context there.
    with pytest.raises(SystemExit):
        tool.parse_args(['--tokens', '32'])


def test_attention_kernels_reads_forward_ops_in_order(tool):
    names = ['aten::linear', 'aten::_scaled_dot_product_cudnn_attention', 'aten::scaled_dot_product_attention',
             'aten::_scaled_dot_product_flash_attention_backward', 'aten::_scaled_dot_product_flash_attention',
             'aten::_scaled_dot_product_attention_math']
    profile = types.SimpleNamespace(events=lambda: [types.SimpleNamespace(name=n) for n in names])
    assert tool.attention_kernels(profile) == ['cudnn', 'flash', 'math']


def fake_worker(gpu, task, connection, args):
    """Stands in for worker() in the spawned processes: the scenario name says how it behaves."""
    import os
    import time
    name, _, tokens = task
    if name == 'crash':
        os._exit(3)
    if name == 'silent':
        return
    if name == 'hang':
        time.sleep(120)
    payload = 'x' * (1 << 20) if name == 'large' else ''
    connection.send({'scenario': name, 'gpu': gpu, 'tokens': tokens, 'payload': payload})


def test_every_task_gets_one_result_on_one_gpu(tool):
    tasks = [(name, None, tokens) for tokens in (97, 140) for name in ('a', 'b', 'large')]
    args = types.SimpleNamespace(timeout=60)
    results = tool.run_all(tasks, [0, 1], args, target=fake_worker)
    assert sorted((r['scenario'], r['tokens']) for r in results) == sorted((t[0], t[2]) for t in tasks)
    assert all('crash' not in r for r in results) and {r['gpu'] for r in results} == {0, 1}
    assert len(next(r for r in results if r['scenario'] == 'large')['payload']) == 1 << 20


def test_failed_scenarios_are_reported_and_free_their_gpu(tool):
    tasks = [('crash', None, 97), ('silent', None, 97), ('hang', None, 97), ('after', None, 97)]
    args = types.SimpleNamespace(timeout=10)
    results = {r['scenario']: r for r in tool.run_all(tasks, [0], args, target=fake_worker)}
    assert results['crash']['crash'] == 'the process exited with code 3 and no result'
    assert results['silent']['crash'] == 'the process exited with code 0 and no result'
    assert results['hang']['crash'] == 'it ran longer than --timeout 10 s'
    assert 'crash' not in results['after']
