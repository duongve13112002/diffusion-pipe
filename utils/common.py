from contextlib import contextmanager
import gc
import time
import math
from pathlib import Path
import os

import torch
import deepspeed.comm.comm as dist
import imageio
from safetensors import safe_open


DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float8': torch.float8_e4m3fn,
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
}
VIDEO_EXTENSIONS = set()
for x in imageio.config.video_extensions:
    VIDEO_EXTENSIONS.add(x.extension)
    VIDEO_EXTENSIONS.add(x.extension.upper())
AUTOCAST_DTYPE = None


def get_rank():
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@contextmanager
def zero_first():
    if not is_main_process():
        dist.barrier()
    yield
    if is_main_process():
        dist.barrier()


@contextmanager
def one_at_a_time():
    for i in range(int(os.environ['LOCAL_SIZE'])):
        if i == int(os.environ['LOCAL_RANK']):
            yield
        dist.barrier()


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def log_duration(name):
    start = time.time()
    try:
        yield
    finally:
        print(f'{name}: {time.time()-start:.3f}')


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_state_dict(path):
    path = str(path)
    if path.endswith('.safetensors'):
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, weights_only=True)
    for key in sd:
        if key.endswith('scale_input') or key.endswith('scale_weight'):
            raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
    return sd


def iterate_safetensors(path):
    path = Path(path)
    if path.is_dir():
        safetensors_files = list(path.glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise FileNotFoundError(f'Cound not find safetensors files in directory {path}')
    else:
        if path.suffix != '.safetensors':
            raise ValueError(f'Expected {path} to be a safetensors file')
        safetensors_files = [path]
    for filename in safetensors_files:
        with safe_open(str(filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith('scale_input') or key.endswith('scale_weight'):
                    raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
                yield key, f.get_tensor(key)


def round_to_nearest_multiple(x, multiple):
    return int(round(x / multiple) * multiple)


def round_down_to_multiple(x, multiple):
    return int((x // multiple) * multiple)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_t_distribution(model_config):
    timestep_sample_method = model_config.get('timestep_sample_method', 'logit_normal')

    if timestep_sample_method == 'logit_normal':
        dist = torch.distributions.normal.Normal(0, 1)
    elif timestep_sample_method == 'uniform':
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError()

    n_buckets = 10_000
    delta = 1 / n_buckets
    min_quantile = delta
    max_quantile = 1 - delta
    quantiles = torch.linspace(min_quantile, max_quantile, n_buckets)
    t = dist.icdf(quantiles)

    if timestep_sample_method == 'logit_normal':
        sigmoid_scale = model_config.get('sigmoid_scale', 1.0)
        t = t * sigmoid_scale
        t = torch.sigmoid(t)

    return t


def slice_t_distribution(t, min_t=0.0, max_t=1.0):
    start = torch.searchsorted(t, min_t).item()
    end = torch.searchsorted(t, max_t).item()
    return t[start:end]


def sample_t(t, batch_size, quantile=None):
    if quantile is not None:
        i = (torch.full((batch_size,), quantile) * len(t)).to(torch.int32)
    else:
        i = torch.randint(0, len(t), size=(batch_size,))
    return t[i]


def _cgroup_cpu_quota(root=Path('/sys/fs/cgroup')):
    """CPUs granted by a cgroup CPU quota (a Kubernetes CPU limit), or None without one."""
    try:
        quota, period = (root / 'cpu.max').read_text().split()[:2]  # cgroup v2
        return None if quota == 'max' else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int((root / 'cpu' / 'cpu.cfs_quota_us').read_text())  # cgroup v1
        period = int((root / 'cpu' / 'cpu.cfs_period_us').read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def usable_cpu_count(cgroup_root=Path('/sys/fs/cgroup')):
    """CPUs this process can actually run on. os.cpu_count() reports every core of the host, even
    in a container whose CPU limit or cpuset grants a fraction of them."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    quota = _cgroup_cpu_quota(cgroup_root)
    if quota is not None:
        count = min(count, max(1, int(quota)))
    return count


def limit_cpu_threads(local_world_size, cgroup_root=Path('/sys/fs/cgroup')):
    """Give each of the GPU processes on this machine its share of the CPUs.

    torch otherwise starts a thread per host core in every process, and the deepspeed launcher
    (unlike torchrun) does not set OMP_NUM_THREADS: 8 GPU processes on a 224-thread host run
    1800 threads over the CPUs the pod may use, and the CPU work of startup (loading and casting
    checkpoints, decoding) spends its time in the scheduler instead. An OMP_NUM_THREADS set by
    the user is left alone. Returns the thread count set, or None."""
    if os.environ.get('OMP_NUM_THREADS'):
        return None
    threads = max(1, usable_cpu_count(cgroup_root) // max(1, local_world_size))
    torch.set_num_threads(threads)
    # Inherited by the processes started later (dataset map workers, dataloader workers).
    os.environ['OMP_NUM_THREADS'] = str(threads)
    return threads


def ensure_writable_dir(path, setting):
    """Create `path` now, so a run whose output directory cannot be written stops before the
    dataset and model load instead of after."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise SystemExit(f'{setting} = {str(path)!r} cannot be created ({e.strerror}: {e.filename}). '
                         f'Point {setting} at a directory this user can write, e.g. on the PVC.') from None
    if not os.access(path, os.W_OK | os.X_OK):
        raise SystemExit(f'{setting} = {str(path)!r} exists but this user cannot write to it.')


def fsync_dir(path):
    """Make a rename or a new entry in `path` durable. Some filesystems refuse to open or fsync a
    directory; the rename itself has still happened, so that is not an error."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path, data):
    """Write bytes under a temporary name (not matching *.safetensors or *.pt, which loaders
    glob), fsync, rename, fsync the directory: a crash or a network volume hiccup never leaves a
    torn file under the real name."""
    def write(f):
        f.write(data)
    _write_atomic(path, write)


def torch_save_atomic(obj, path, **kwargs):
    """torch.save, with write_atomic's guarantees."""
    _write_atomic(path, lambda f: torch.save(obj, f, **kwargs))


def _write_atomic(path, write):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}')
    try:
        with open(tmp, 'wb') as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def all_ok(error):
    """Raise on every rank if any rank failed, before anyone enters another collective. Work done
    on one rank (saving, exporting) goes between two of these, so a failure there stops every
    rank instead of leaving the others waiting out the process group timeout."""
    import torch.distributed as torch_dist
    if not torch_dist.is_available() or not torch_dist.is_initialized() or torch_dist.get_world_size() == 1:
        if error is not None:
            raise error
        return
    errors = [None] * torch_dist.get_world_size()
    torch_dist.all_gather_object(errors, None if error is None else f'{type(error).__name__}: {error}')
    failed = [f'rank {i}: {e}' for i, e in enumerate(errors) if e is not None]
    if failed:
        if error is not None:
            raise RuntimeError('; '.join(failed)) from error
        raise RuntimeError('; '.join(failed))
