"""Multi-node launches: local ranks from any launcher, and dataset caching across nodes.

The caching tests run real gloo process groups in which some ranks pretend to be on another
machine (another hostname); a node's work queue lives behind a Unix socket in its machine's /tmp,
which the ranks of other machines cannot reach.
"""

import json
from collections import defaultdict
import types
import socket

import pytest
import torch.multiprocessing as mp

from utils.common import resolve_local_rank, set_rendezvous_env


@pytest.mark.parametrize('environ, arg, local_rank, local_size', [
    # deepspeed launcher: --local_rank, LOCAL_RANK and LOCAL_SIZE
    ({'WORLD_SIZE': '16', 'LOCAL_RANK': '3', 'LOCAL_SIZE': '8'}, 3, 3, '8'),
    # torchrun / PyTorchJob: no argument, LOCAL_WORLD_SIZE instead of LOCAL_SIZE
    ({'WORLD_SIZE': '16', 'LOCAL_RANK': '5', 'LOCAL_WORLD_SIZE': '8'}, -1, 5, '8'),
    # an old launcher passing only the argument
    ({'WORLD_SIZE': '8', 'LOCAL_SIZE': '8'}, 2, 2, '8'),
    # a single process started as `python train.py`
    ({}, -1, 0, '1'),
])
def test_the_local_rank_comes_from_whichever_launcher_set_it(environ, arg, local_rank, local_size):
    assert resolve_local_rank(arg, environ) == local_rank
    assert environ['LOCAL_RANK'] == str(local_rank) and environ['LOCAL_SIZE'] == local_size


@pytest.mark.parametrize('environ, arg, match', [
    ({'WORLD_SIZE': '16', 'LOCAL_RANK': '5'}, 4, 'disagrees with LOCAL_RANK=5'),
    ({'WORLD_SIZE': '16'}, -1, 'neither LOCAL_RANK nor --local_rank'),
])
def test_an_unknown_or_contradicting_local_rank_is_refused(environ, arg, match):
    with pytest.raises(ValueError, match=match):
        resolve_local_rank(arg, environ)


@pytest.mark.parametrize('environ, arg, port', [
    ({'MASTER_ADDR': 'pod-0', 'MASTER_PORT': '23456'}, None, '23456'),
    ({'MASTER_ADDR': 'pod-0', 'MASTER_PORT': '23456'}, 29600, '29600'),
    ({}, None, '29500'),
])
def test_the_launchers_master_port_is_kept_unless_one_is_given(environ, arg, port):
    addr = environ.get('MASTER_ADDR', 'localhost')
    set_rendezvous_env(arg, environ)
    assert (environ['MASTER_ADDR'], environ['MASTER_PORT']) == (addr, port)


HOSTS = ['node-a', 'node-a', 'node-b']


def _caching_process(queue, mode):
    """Stands in for _cache_fn, in the process global rank 0 starts for it."""
    import os
    import time
    if mode == 'killed':
        # Killed (out of memory, say) with no chance to end the queue.
        queue.put('task')
        os._exit(9)
    try:
        queue.put('boom' if mode == 'rank_failed' else 'task')
        if mode == 'process_failed':
            raise RuntimeError('caching process failed')
        if mode == 'rank_failed':
            # As a map worker would, wait for the result the failed rank never sends.
            time.sleep(600)
    finally:
        queue.put(None)


def _caching_worker(rank, init_file, out_dir, mode):
    from datetime import timedelta
    import deepspeed
    import multiprocess
    # As train.py does: the queue's proxies authenticate with this key in every process.
    multiprocess.current_process().authkey = b'afsaskgfdjh4'
    socket.gethostname = lambda: HOSTS[rank]
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=len(HOSTS), verbose=False, timeout=timedelta(seconds=120))
    from utils import dataset
    dataset.CACHING_POLL_SECONDS = 0.5

    if mode == 'connect_failed' and rank == 1:
        def refuse(payload):
            raise ConnectionRefusedError('cannot reach the queue')
        dataset.pickle = types.SimpleNamespace(loads=refuse)
    queue, manager, error = dataset._share_work_queue()
    process = None
    if rank == 0:
        if mode == 'start_failed':
            # As DatasetManager.cache passes a failed process.start() on.
            error = OSError('cannot start the caching process')
        else:
            process = multiprocess.Process(target=_caching_process, args=(queue, mode))
            process.start()
    tasks = []

    def handle_task(task):
        if task == 'boom':
            raise RuntimeError('task failed')
        tasks.append(task)

    result = {'has_queue': queue is not None}
    try:
        dataset._run_caching(queue, handle_task, process, 1, error)
        result['outcome'] = 'done'
    except RuntimeError as error:
        result['outcome'] = str(error)
    result['tasks'] = tasks
    deepspeed.comm.barrier()
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


@pytest.mark.parametrize('mode, error', [
    ('ok', None),
    ('process_failed', 'The dataset caching process failed with exit code 1'),
    ('killed', 'The dataset caching process ended with exit code 9 before finishing'),
    ('rank_failed', 'task failed'),
    ('start_failed', 'cannot start the caching process'),
    ('connect_failed', 'cannot reach the queue'),
])
def test_only_the_ranks_on_rank_0s_node_take_caching_work_and_all_fail_together(mode, error, tmp_path):
    mp.spawn(_caching_worker, args=(str(tmp_path / 'init'), str(tmp_path), mode), nprocs=len(HOSTS))
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(len(HOSTS))]
    # Ranks 0 and 1 share node-a and the queue; rank 2, on node-b, waits for them.
    assert [r['has_queue'] for r in results] == [True, mode != 'connect_failed', False]
    if error is None:
        assert sorted(results[0]['tasks'] + results[1]['tasks']) == ['task']
        assert [r['outcome'] for r in results] == ['done'] * 3
    else:
        # Every rank fails, with the error of the rank it came from, instead of waiting forever.
        for r in results:
            assert error in r['outcome'], r['outcome']
        if mode != 'connect_failed':
            # Rank 0 may finish the work before rank 1's abort reaches it, and tell node-b all went well:
            # node-b then fails with the others at the check after caching.
            assert 'Dataset caching failed on global rank 0' in results[2]['outcome']


NODE_HOSTS = ['node-a', 'node-a', 'node-b', 'node-b']


def _node_caching_process(queue, mode, host, out_dir):
    """Stands in for _cache_fn in stage 'tensors', in the process the lowest rank of each node
    starts for it."""
    import os
    import time
    if mode in ('claims', 'disagree'):
        from utils import dataset
        units = [f'unit {i}' for i in range(13 if mode == 'disagree' and host == 'node-b' else 12)]
        try:
            first = True
            while (index := dataset._claim(queue, 'tensors', units)) < len(units):
                queue.put(f'{host} took {index}')
                if first:
                    # Each node holds on to its first unit until the other has one too, so both
                    # nodes take part whatever their start times.
                    first = False
                    open(f'{out_dir}/{host}.claimed', 'w').close()
                    deadline = time.time() + 60
                    while not all(os.path.exists(f'{out_dir}/{h}.claimed') for h in ('node-a', 'node-b')):
                        assert time.time() < deadline
                        time.sleep(0.1)
        finally:
            queue.put(None)
        return
    failing = host == ('node-a' if mode == 'rank_failed' else 'node-b')
    if mode == 'silent' and failing:
        # Waits for the answer of a rank that has died, as a caching process would.
        queue.put(f'task from {host}')
        time.sleep(600)
    if mode == 'killed' and failing:
        queue.put(f'task from {host}')
        os._exit(9)
    try:
        queue.put('boom' if mode == 'rank_failed' and failing else f'task from {host}')
        if mode == 'process_failed' and failing:
            raise RuntimeError('caching process failed')
        if mode == 'rank_failed' and failing:
            time.sleep(600)
    finally:
        queue.put(None)


def _node_caching_worker(rank, init_file, out_dir, mode):
    from datetime import timedelta
    import deepspeed
    import multiprocess
    multiprocess.current_process().authkey = b'afsaskgfdjh4'
    socket.gethostname = lambda: NODE_HOSTS[rank]
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=len(NODE_HOSTS), verbose=False, timeout=timedelta(seconds=120))
    from utils import dataset
    dataset.CACHING_POLL_SECONDS = 0.5
    dataset.CACHING_HEARTBEAT_SECONDS = 0.5
    dataset.CACHING_SILENCE = timedelta(seconds=4)

    node, nodes, leader = dataset._node_layout()
    if mode == 'silent' and rank == 3:
        # Dead to the others: it never beats. (Really killing it would end mp.spawn.)
        dataset._Liveness._beat = lambda self: None
    if mode == 'connect_failed' and rank == 3:
        def refuse(payload):
            raise ConnectionRefusedError('cannot reach the queue')
        dataset.pickle = types.SimpleNamespace(loads=refuse)
    queue, manager, error = dataset._share_node_queues(leader)
    process = None
    if leader:
        if mode == 'start_failed' and node == 1:
            error = OSError('cannot start the caching process')
        else:
            process = multiprocess.Process(target=_node_caching_process, args=(queue, mode, NODE_HOSTS[rank], out_dir))
            process.start()
    tasks = []

    def handle_task(task):
        if task == 'boom':
            raise RuntimeError('task failed')
        tasks.append(task)

    result = {'layout': [node, nodes, leader], 'has_queue': queue is not None}
    try:
        dataset._run_caching_on_every_node(queue, dataset._with_claims(7, handle_task), process, 7, node, nodes, leader,
                                           error)
        result['outcome'] = 'done'
    except RuntimeError as e:
        result['outcome'] = str(e)
    result['tasks'] = tasks
    deepspeed.comm.barrier()
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


@pytest.mark.parametrize('mode, error', [
    ('ok', None),
    ('process_failed', 'The dataset caching process failed with exit code 1'),
    ('killed', 'The dataset caching process ended with exit code 9 before finishing'),
    ('rank_failed', 'task failed'),
    ('start_failed', 'cannot start the caching process'),
    ('connect_failed', 'cannot reach the queue'),
    ('claims', None),
    # Whichever node claims second finds the other's digest.
    ('disagree', 'sees other dataset tensors work than another node'),
    ('silent', 'Global rank 3 has given no sign of life'),
])
def test_every_node_takes_its_own_caching_work_and_all_fail_together(mode, error, tmp_path):
    mp.spawn(_node_caching_worker, args=(str(tmp_path / 'init'), str(tmp_path), mode), nprocs=len(NODE_HOSTS))
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(len(NODE_HOSTS))]
    assert [r['layout'] for r in results] == [[0, 2, True], [0, 2, False], [1, 2, True], [1, 2, False]]
    assert [r['has_queue'] for r in results] == [True, True, True, mode != 'connect_failed']
    # A node's tasks go to its own ranks only.
    taken = {}
    for node, ranks in (('node-a', (0, 1)), ('node-b', (2, 3))):
        tasks = [task for rank in ranks for task in results[rank]['tasks']]
        assert all(task.startswith(node) or task == f'task from {node}' for task in tasks), tasks
        taken[node] = [int(task.rsplit(' ', 1)[1]) for task in tasks if ' took ' in task]
    if mode == 'claims':
        # Both nodes drew from the one pool, and between them took every unit once.
        assert taken['node-a'] and taken['node-b']
        assert sorted(taken['node-a'] + taken['node-b']) == list(range(12))
        assert [r['outcome'] for r in results] == ['done'] * 4
    elif error is None:
        assert sorted(results[0]['tasks'] + results[1]['tasks']) == ['task from node-a']
        assert sorted(results[2]['tasks'] + results[3]['tasks']) == ['task from node-b']
        assert [r['outcome'] for r in results] == ['done'] * 4
    else:
        # Every rank of every node fails, with the error of the rank it came from.
        for r in results:
            assert error in r['outcome'], r['outcome']


class _Model:
    """Stands in for the model's VAE and text encoder wrappers, and records which node encoded
    each row."""
    name = 'fake'
    framerate = None
    pixels_round_to_multiple = 16

    def __init__(self):
        self.config = {}
        self.encoder = types.SimpleNamespace(load_model_if_needed=lambda: None)
        self.text_encoders = [types.SimpleNamespace(load_model_if_needed=lambda: None, number=n) for n in (1, 2)]

    def node(self, count):
        import torch
        return torch.full((count,), ord(socket.gethostname()[-1]) - ord('a'))

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    def vae_cache_key(self):
        return ''

    def get_vae(self):
        return self.encoder

    def get_text_encoders(self):
        return self.text_encoders

    def get_preprocess_media_file_fn(self):
        from models.base import PreprocessMediaFile
        return PreprocessMediaFile(self.config)

    def get_call_vae_fn(self, vae):
        return lambda tensor: {'latents': tensor.mean(dim=1, keepdim=True), 'vae_node': self.node(len(tensor))}

    def get_call_text_encoder_fn(self, text_encoder):
        """Two encoders, the first slow on node-a: node-b runs out of first encoder work while
        node-a is still on its first bucket, and would start the second encoder at once if it
        could."""
        import time
        import torch
        suffix = '' if text_encoder.number == 1 else str(text_encoder.number)

        def encode(captions, is_video):
            if text_encoder.number == 1 and socket.gethostname() == 'node-a':
                time.sleep(0.3)
            return {f'length{suffix}': torch.tensor([len(c) for c in captions]),
                    f'te{suffix}_node': self.node(len(captions)),
                    f'te{suffix}_time': torch.full((len(captions),), time.time(), dtype=torch.float64)}
        return encode


# Images per size bucket: (width, height) -> count.
BUCKET_IMAGES = {(32, 32): 5, (48, 32): 3, (32, 48): 2}


def _image_dir(path):
    import numpy as np
    from PIL import Image
    path.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for (w, h), count in BUCKET_IMAGES.items():
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path / f'{w}x{h}_{i}.png')
            (path / f'{w}x{h}_{i}.txt').write_text(f'{w} by {h}, number {i}')
    return path


# Two directories, so that metadata is drawn from a pool too.
DIRECTORIES = ('images', 'more')


def _dataset_config(root, reuse_metadata_cache):
    return {'resolutions': [32], 'enable_ar_bucket': False, 'reuse_metadata_cache': reuse_metadata_cache,
            'directory': [{'path': str(root / name), 'num_repeats': 1,
                           'size_buckets': [[w, h, 1] for w, h in BUCKET_IMAGES]} for name in DIRECTORIES]}


def _dataset_caching_worker(rank, init_file, out_dir, hosts, config, multi_node):
    from datetime import timedelta
    import deepspeed
    import multiprocess
    multiprocess.current_process().authkey = b'afsaskgfdjh4'
    socket.gethostname = lambda: hosts[rank]
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=len(hosts), verbose=False, timeout=timedelta(seconds=120))
    from utils import dataset
    dataset.CACHING_POLL_SECONDS = 0.5

    model = _Model()
    data = dataset.Dataset(config, model)
    manager = dataset.DatasetManager(model, multi_node=multi_node)
    manager.register(data)
    manager.cache(unload_models=False)

    rows = {}
    uncond = {}
    for directory in data.directory_datasets:
        for bucket in directory.bucket_datasets():
            for index in range(len(bucket.iteration_order)):
                item = bucket[index]
                rows[f'{directory.path.name}/{item["image_spec"][1].rsplit("/", 1)[-1]}'] = {
                    'bucket': list(bucket.size_bucket), 'latents': item['latents'].flatten().tolist(),
                    'vae_node': int(item['vae_node']), 'length': int(item['length']), 'te_node': int(item['te_node']),
                    'length2': int(item['length2']), 'te2_node': int(item['te2_node']),
                    'te_time': float(item['te_time']), 'te2_time': float(item['te2_time'])}
            first, second = (embeddings[0] for embeddings in bucket.uncond_text_embeddings)
            uncond[directory.path.name] = [int(first['length']), int(first['te_node']),
                                           int(second['length2']), int(second['te2_node'])]
    result = {'rows': rows, 'uncond': uncond}

    # A second run, on every node, finds every cache complete, whether one node or every node
    # made it: nothing to encode.
    again = dataset.DatasetManager(model, multi_node=True)
    again.register(dataset.Dataset(config, model))
    tasks = []
    handle_task = again._handle_task
    again._handle_task = lambda task: (tasks.append(task[0]), handle_task(task))
    again.cache(unload_models=False)
    result['tasks_again'] = len(tasks)
    deepspeed.comm.barrier()
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump(result, f)


def _cache_dataset(tmp_path, name, hosts, reuse_metadata_cache, multi_node):
    import shutil
    for directory in DIRECTORIES:
        shutil.copytree(tmp_path / 'source', tmp_path / name / directory)
    out = tmp_path / name / 'out'
    out.mkdir()
    mp.spawn(_dataset_caching_worker, nprocs=len(hosts),
             args=(str(tmp_path / name / 'init'), str(out), hosts, _dataset_config(tmp_path / name, reuse_metadata_cache),
                   multi_node))
    return [json.loads((out / f'{rank}.json').read_text()) for rank in range(len(hosts))]


@pytest.mark.parametrize('reuse_metadata_cache', [False, True])
def test_nodes_cache_buckets_drawn_from_a_pool_and_every_rank_reads_all(tmp_path, monkeypatch, reuse_metadata_cache):
    monkeypatch.setenv('DIFFUSION_PIPE_NUM_PROC', '1')
    _image_dir(tmp_path / 'source')
    hosts = ['node-a', 'node-a', 'node-b']
    one_node = _cache_dataset(tmp_path, 'one_node', hosts, reuse_metadata_cache, multi_node=False)
    two_nodes = _cache_dataset(tmp_path, 'two_nodes', hosts, reuse_metadata_cache, multi_node=True)

    def without_nodes(rows):
        return {name: {k: v for k, v in row.items() if not k.endswith(('_node', '_time'))}
                for name, row in rows.items()}

    expected = one_node[0]['rows']
    assert len(expected) == len(DIRECTORIES) * sum(BUCKET_IMAGES.values())
    assert {row[kind] for row in expected.values() for kind in ('vae_node', 'te_node', 'te2_node')} == {0}
    for result in one_node + two_nodes:
        # The same caches, whichever node encoded them, on every rank.
        assert without_nodes(result['rows']) == without_nodes(expected)
        # The empty caption of each directory is encoded on global rank 0's node.
        assert result['uncond'] == {name: [0, 0, 0, 0] for name in DIRECTORIES}
        assert result['tasks_again'] == 0
    # One text encoder's round ends on every node before the next encoder's starts: their buckets
    # share files, which no two nodes may work on at once.
    rows = two_nodes[0]['rows'].values()
    assert max(row['te_time'] for row in rows) < min(row['te2_time'] for row in rows)
    # A bucket's latents all come from the node that drew it from the latents pool, and its text
    # embeddings from the node that drew it from that encoder's pool.
    for kind in ('vae_node', 'te_node', 'te2_node'):
        nodes = defaultdict(set)
        for name, row in two_nodes[0]['rows'].items():
            nodes[name.split('/')[0], tuple(row['bucket'])].add(row[kind])
        assert len(nodes) == len(DIRECTORIES) * len(BUCKET_IMAGES)
        assert all(len(drawn_by) == 1 for drawn_by in nodes.values()), (kind, nodes)


def test_pools_go_largest_bucket_first_and_keep_what_shares_files_together(tmp_path):
    from pathlib import Path
    from utils.dataset import metadata_units, tensor_units

    legacy = types.SimpleNamespace(reuse_metadata_cache=False)

    def bucket(cache_dir, rows, resolutions=None, directory=legacy):
        b = types.SimpleNamespace(cache_dir=Path(cache_dir), metadata_dataset=range(rows), directory_dataset=directory)
        if resolutions is not None:
            b.resolutions = resolutions
        return b

    def datasets(*directories):
        return [types.SimpleNamespace(directory_datasets=[
            types.SimpleNamespace(path=Path(path), bucket_datasets=lambda b=buckets: b) for path, buckets in directories])]

    train = datasets(('/a', [bucket('/a/1', 10), bucket('/a/2', 6), bucket('/a/3', 5)]), ('/b', [bucket('/b/1', 4)]))
    assert tensor_units(train) == ['/a/1', '/a/2', '/a/3', '/b/1']
    # An eval dataset reading /a as well: its bucket /a/2 is one unit with the train one (12 rows,
    # now the largest), cached one after the other as on a single node. An aspect ratio bucket
    # counts once per resolution (9 rows). Ties go by name.
    evaluation = datasets(('/a', [bucket('/a/2', 6)]), ('/c', [bucket('/c/ar', 3, resolutions=[32, 64, 96])]),
                          ('/d', [bucket('/d/1', 4)]))
    assert tensor_units(train + evaluation) == ['/a/2', '/a/1', '/c/ar', '/a/3', '/b/1', '/d/1']
    # Metadata: one unit per source directory, in the order of the configs.
    units = metadata_units(train + evaluation)
    assert list(units) == ['/a', '/b', '/c', '/d']
    assert [len(directories) for directories in units.values()] == [2, 1, 1, 1]
    # One directory spelled two ways is one unit.
    (tmp_path / 'data').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'data')
    spelled = datasets((str(tmp_path / 'data'), []), (str(tmp_path / 'link'), []))
    assert list(metadata_units(spelled)) == [str(tmp_path / 'data')]
    # With reuse_metadata_cache, a bucket under two caption profiles of one snapshot is one unit:
    # a profile caches its latents from the other's.
    snapshot = types.SimpleNamespace(reuse_metadata_cache=True, snapshot_dir=Path('/s'))
    profiles = datasets(('/e', [bucket('/s/p1/cache_64x64x1', 5, directory=snapshot),
                                bucket('/s/p2/cache_64x64x1', 5, directory=snapshot),
                                bucket('/s/p2/cache_32x32x1', 7, directory=snapshot)]))
    assert tensor_units(profiles) == ['/s/cache_64x64x1', '/s/cache_32x32x1']


def test_the_caching_process_ends_the_queue_when_it_fails():
    import queue as queue_module
    from utils import dataset

    class Broken:
        def cache_metadata(self, **kwargs):
            raise OSError('unreadable dataset')

    queue = queue_module.Queue()
    with pytest.raises(OSError, match='unreadable dataset'):
        dataset._cache_fn([Broken()], queue, None, 1, False, False, 1)
    assert queue.get_nowait() is None


def _loader(batches_per_epoch):
    from utils.dataset import PipelineDataLoader
    loader = PipelineDataLoader.__new__(PipelineDataLoader)
    loader.dataset = range(batches_per_epoch)
    loader.iter_called = False
    loader.skipped = None
    loader._set_dataset_epoch = lambda: None
    loader._create_dataloader = lambda skip_first_n_batches=0: setattr(loader, 'skipped', skip_first_n_batches)
    loader._pull_batches_from_dataloader = lambda: iter(())
    return loader


@pytest.mark.parametrize('saved, resumed, pulled, skipped', [
    # 2 nodes x 8 GPUs with pipeline_stages = 2 (DP 8), resumed on one node (DP 4): an epoch has
    # twice as many batches of half the size.
    (200, 400, 101, 200),
    (400, 200, 101, 50),
    # Another micro batch size changes the epoch's length in the same way.
    (300, 150, 241, 120),
    (200, 200, 101, 100),
    # Per-bucket rounding: 10 buckets of 3 batches become 10 batches. Late in the epoch, a count of
    # examples would point past its end.
    (30, 10, 30, 9),
    # Saved just after the epoch rolled over.
    (400, 200, 0, 0),
    (200, 200, 0, 0),
])
def test_the_dataloader_resumes_after_the_same_share_of_the_epoch(saved, resumed, pulled, skipped, monkeypatch):
    monkeypatch.setattr('utils.dataset.is_main_process', lambda: True)
    state = _loader(saved)
    state.epoch, state.num_batches_pulled = 3, pulled
    loader = _loader(resumed)
    loader.load_state_dict(state.state_dict())
    assert (loader.epoch, loader.skipped) == (3, skipped)
    assert loader.skipped < resumed


def test_an_old_checkpoint_without_the_epoch_length_resumes_as_before(monkeypatch):
    monkeypatch.setattr('utils.dataset.is_main_process', lambda: True)
    loader = _loader(50)
    loader.load_state_dict({'epoch': 2, 'num_batches_pulled': 101})
    assert (loader.epoch, loader.skipped) == (2, 100)


def test_a_bucket_is_shuffled_in_memory_with_the_fingerprint_its_caches_know(tmp_path):
    import datasets
    from utils.dataset import shuffle_in_memory
    datasets.Dataset.from_dict({'image_spec': [f'{i}.png' for i in range(50)]}).save_to_disk(str(tmp_path))
    files = sorted(p.name for p in tmp_path.iterdir())
    on_disk = datasets.load_from_disk(str(tmp_path))
    shuffled = shuffle_in_memory(on_disk, 1234)
    # Nothing written next to the dataset, which every rank of every node shuffles.
    assert sorted(p.name for p in tmp_path.iterdir()) == files
    expected = on_disk.shuffle(seed=1234)
    assert shuffled._fingerprint == expected._fingerprint
    assert shuffled['image_spec'] == expected['image_spec'] != on_disk['image_spec']


def _signal_worker(rank, init_file, out_dir, name):
    from pathlib import Path
    import deepspeed
    deepspeed.init_distributed('gloo', auto_mpi_discovery=False, init_method=f'file://{init_file}',
                               rank=rank, world_size=2, verbose=False)
    from utils.saver import manual_save_signal
    # Each rank its own view of the output directory: only rank 0 sees the file yet, as a rank
    # that got there first, or a node whose view of the volume is behind, would not.
    save_root = Path(out_dir) / f'view{rank}'
    save_root.mkdir()
    if rank == 0:
        (save_root / name).touch()
    first = manual_save_signal(save_root)
    second = manual_save_signal(save_root)
    with open(f'{out_dir}/{rank}.json', 'w') as f:
        json.dump({'first': first, 'second': second, 'left': sorted(p.name for p in save_root.iterdir())}, f)


@pytest.mark.parametrize('name, signal', [('save', [True, False]), ('save_quit', [True, True])])
def test_every_rank_acts_on_the_save_signal_rank_0_sees_once(name, signal, tmp_path):
    mp.spawn(_signal_worker, args=(str(tmp_path / 'init'), str(tmp_path), name), nprocs=2)
    results = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    for result in results:
        assert result['first'] == signal and result['second'] == [False, False]
    assert results[0]['left'] == []


def _run(root, name, tag=None):
    run = root / name
    run.mkdir(parents=True)
    if tag:
        (run / tag).mkdir()
        (run / 'latest').write_text(tag + '\n')
    return run


def test_resuming_picks_the_newest_run_that_has_a_checkpoint(tmp_path, capsys):
    from utils.common import resume_run_dir
    _run(tmp_path, '20260101_00-00-00', 'global_step100')
    newest = _run(tmp_path, '20260102_00-00-00', 'global_step7')
    # A run that died before its first checkpoint, and a stray file.
    _run(tmp_path, '20260103_00-00-00')
    (tmp_path / 'train.log').write_text('')
    assert resume_run_dir(tmp_path, True) == (str(newest), 'global_step7')
    printed = capsys.readouterr().out
    # Which run, and that a newer one was passed over.
    assert f'Resuming {newest} from checkpoint global_step7' in printed
    assert 'newer directories without one are left as they are: 20260103_00-00-00\n' in printed
    assert resume_run_dir(tmp_path, '20260101_00-00-00') == (str(tmp_path / '20260101_00-00-00'), 'global_step100')
    assert 'WARNING' not in capsys.readouterr().out


@pytest.mark.parametrize('setup, resume, match', [
    (lambda root: None, True, 'no run directory in .* has a checkpoint'),
    (lambda root: _run(root, 'a'), True, 'no run directory in .* has a checkpoint'),
    (lambda root: _run(root, 'a', 'global_step1'), 'typo', 'Nothing to resume in .*typo'),
    (lambda root: (_run(root, 'a', 'global_step1') / 'latest').write_text('global_step2'), True,
     "names checkpoint 'global_step2', which is not in"),
])
def test_there_is_no_silent_new_run_when_nothing_can_be_resumed(setup, resume, match, tmp_path):
    from utils.common import resume_run_dir
    setup(tmp_path)
    with pytest.raises(ValueError, match=match):
        resume_run_dir(tmp_path, resume)


def test_resuming_from_an_output_dir_that_does_not_exist_is_refused(tmp_path):
    from utils.common import resume_run_dir
    with pytest.raises(ValueError, match='no run directory in .* has a checkpoint'):
        resume_run_dir(tmp_path / 'missing', True)


def test_a_save_missing_a_pipeline_stage_fails_instead_of_publishing_part_of_the_model(tmp_path, monkeypatch):
    import threading
    import time
    import torch
    from utils import saver
    monkeypatch.setattr(saver, 'STAGE_FILE_WAIT_SECONDS', 3)
    for stage in (0, 2):
        torch.save({f'w{stage}': torch.zeros(1)}, tmp_path / f'state_dict_{stage}.bin')
    # A file another node wrote shows up late: waited for.
    late = threading.Timer(0.5, lambda: torch.save({}, tmp_path / 'state_dict_1.bin'))
    late.start()
    assert [p.name for p in saver.stage_files(tmp_path, 3)] == [f'state_dict_{s}.bin' for s in range(3)]
    late.join()
    # One that never does: refused, rather than an export without that stage's layers.
    monkeypatch.setattr(saver, 'STAGE_FILE_WAIT_SECONDS', 0.5)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='state_dict_3.bin not found .* volume that every node shares'):
        saver.stage_files(tmp_path, 4)
    assert time.monotonic() - start < 5


def test_a_file_rank_0_wrote_must_show_up_on_every_node(tmp_path):
    import threading
    from utils.common import wait_until_visible
    path = tmp_path / 'train.toml'
    threading.Timer(0.3, path.touch).start()
    wait_until_visible(path, 'hint', seconds=5)
    with pytest.raises(RuntimeError, match='missing.toml, written by global rank 0, is not visible on .*: shared volume'):
        wait_until_visible(tmp_path / 'missing.toml', 'shared volume', seconds=0.5)
