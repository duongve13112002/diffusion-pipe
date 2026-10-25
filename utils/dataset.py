from pathlib import Path
import os.path
import random
from collections import OrderedDict, defaultdict
import math
import os
import hashlib
import pickle
from queue import Empty
from datetime import timedelta
import json
import contextlib
import errno
import shutil
import socket
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import numpy as np
import torch
from torch import nn
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from datasets.fingerprint import Hasher
from PIL import Image
import imageio
import multiprocess as mp
from tqdm import tqdm
from comfy_api.latest import InputImpl

from utils.common import all_ok, fsync_dir, is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple
from utils.cache import Cache
from utils.cache_profiles import (align_legacy_metadata, IndexedTextEmbeddingDataset, RowsByImage,
                                  tensor_profile)
import comfy.model_management as mm


DEBUG = False
IMAGE_SIZE_ROUND_TO_MULTIPLE = 32
# Worker count for the dataset map pools. A forked worker is nearly free, but on a spawn
# platform each one re-imports torch, deepspeed and ComfyUI, which costs far more than
# mapping a small dataset. DIFFUSION_PIPE_NUM_PROC overrides the default; 1 maps in-process.
NUM_PROC = int(os.environ.get('DIFFUSION_PIPE_NUM_PROC', 0)) or min(8, os.cpu_count())
ROUND_DECIMAL_DIGITS = 3
# Threads per metadata map process that read media headers and caption files. On network storage
# the scan waits on a round trip per open, and many at once make it several times faster: with
# 3 ms per open, 5000 images took 38 s on one thread and 5.4 s on 16. On a local disk the reads
# are CPU work holding the GIL and threads slow it down (1.9 s on one thread, 5.2 s on 16), so
# set 1 there. Config: metadata_read_threads.
METADATA_READ_THREADS = 16

UNCOND_FRACTION = 0.0


_read_pool = None


def _read_in_threads(fn, items, threads=None):
    """fn over items, in order, on threads (default METADATA_READ_THREADS) of this process."""
    global _read_pool
    threads = METADATA_READ_THREADS if threads is None else threads
    if threads <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    # One pool per process: a map worker forked from a process that had one inherits it
    # without its threads.
    if _read_pool is None or _read_pool[:2] != (os.getpid(), threads):
        _read_pool = (os.getpid(), threads, ThreadPoolExecutor(threads, thread_name_prefix='metadata-read'))
    return list(_read_pool[2].map(fn, items))


def _rows_in_threads(row_fn, threads):
    """A batched map function running row_fn, which takes and returns a batch of one row, on
    each row of the batch in threads. The rows come out in the order row_fn would give them
    one at a time. threads is passed, not read in the worker: a spawned map worker never sees
    the configured METADATA_READ_THREADS."""
    def fn(batch):
        size = len(next(iter(batch.values())))
        results = _read_in_threads(row_fn, [{key: [values[i]] for key, values in batch.items()}
                                            for i in range(size)], threads)
        out = {key: [] for key in results[0]}
        for result in results:
            for key, values in result.items():
                out[key].extend(values)
        return out
    return fn


def _is_file(entry):
    """Path.is_file() for an os.scandir entry, without a stat unless the entry is a symlink."""
    try:
        return entry.is_file()
    except OSError as e:
        # The errors Path.is_file() reads as "not a file" (ELOOP: a symlink loop). Anything else,
        # a stale handle or an I/O error on the volume, must not silently drop the file.
        if e.errno in (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP):
            return False
        raise


# {(pid, key): Lock}. Keyed by pid: a forked map worker must not wait on a lock some thread of
# its parent held at the fork. dict.setdefault is atomic, so threads agree on one lock.
_process_locks = {}
# {(pid, tar path): TarFile}
_tarfiles = {}


def _process_lock(key):
    return _process_locks.setdefault((os.getpid(), key), threading.Lock())


def _shared_tarfile(path):
    """(TarFile, lock) for path, one per process. Reading a member moves the TarFile's file
    position, so the reader holds the lock. One TarFile per thread instead would keep a member
    table and a descriptor per thread for every tar."""
    lock = _process_lock(('tar', path))
    with lock:
        key = (os.getpid(), path)
        if key not in _tarfiles:
            _tarfiles[key] = tarfile.TarFile(path)
        return _tarfiles[key], lock


def _files_by_stem(directory):
    """{stem: path} of the files in directory, a later file of a stem replacing an earlier one,
    in the order Path.glob('*') lists them."""
    with os.scandir(directory) as entries:
        return {Path(entry.name).stem: directory / entry.name for entry in entries if _is_file(entry)}


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


from utils.captions import (
    CAPTIONS_JSON_FILE,
    NON_MEDIA_SUFFIXES,
    caption_matches_non_latin_requirement,
    drop_tags,
    enumerate_captions,
    preprocess_caption,
    read_caption_file,
    shuffle_captions,
    split_caption_lines,
    split_tag_prefix,
    validate_caption_dropout_rate,
    validate_non_latin_caption_requirement,
)
from utils.layout_captions import LAYOUT_METADATA_VERSION, LayoutCaptionRewriter, is_layout_caption
from utils.text_encoding import CONTROL_FILE_KEY, IS_VIDEO_KEY, encodes_text_in_loader


def size_bucket_matching_is_current(metadata_dir, suffix):
    """Whether the legacy grouped metadata in metadata_dir was grouped by the fixed size bucket
    matching: its marker records the grouped datasets' fingerprints, which follow their content,
    so a later regrouping by a checkout without the fix no longer matches it."""
    marker = metadata_dir / f'size_bucket_matching{suffix}.json'
    keys_file = metadata_dir / f'grouping_keys{suffix}.json'
    try:
        recorded = json.loads(marker.read_text()).get('grouped')
        keys = json.loads(keys_file.read_text())
        current = {bucket_suffix(key): json.loads(
            (metadata_dir / f'grouped_metadata_{bucket_suffix(key)}{suffix}' / 'state.json').read_text())['_fingerprint']
            for key in keys}
    except (FileNotFoundError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return recorded == current


def bucket_suffix(key):
    if len(key) == 2:
        # AR, frames
        return f'{key[0]:.{ROUND_DECIMAL_DIGITS}f}_{key[1]}'
    elif len(key) == 3:
        # width, height, frames
        return f'{key[0]}x{key[1]}x{key[2]}'
    elif len(key) == 4:
        # AR, width, height, frames
        return f'{key[0]:.{ROUND_DECIMAL_DIGITS}f}x{key[1]}x{key[2]}x{key[3]}'
    else:
        raise RuntimeError(f'Unexpected bucket: {key}')


def legacy_copy_tag(caption_cache_suffix):
    """Ends the name of an AR bucket's per-resolution metadata copy in a legacy cache."""
    return f'_captions{caption_cache_suffix}'


def legacy_iteration_order_names(caption_cache_suffix):
    return [f'iteration_order{caption_cache_suffix}' + ('' if mode == 'all' else f'_{mode}')
            for mode in CAPTION_SAMPLING_MODES]


def dedup_and_sort(values):
    values = set(round(x, ROUND_DECIMAL_DIGITS) for x in values)
    values = list(values)
    values.sort()
    return np.array(values)


def seed_from_hash(item):
    return int(hashlib.md5(str.encode(str(item))).hexdigest(), 16) % int(1e9)


# What to do when a size bucket's sample count is not a multiple of the global batch size.
# 'drop' predates all of this and stays the default everywhere, so an existing config trains on
# exactly the samples it trained on before.
BATCH_FILL_STRATEGIES = ('drop', 'fill')
# What to do for a bucket that cannot even fill one global batch, where repeating something is
# the only way to produce a batch at all.
UNDERSIZED_BUCKET_STRATEGIES = ('drop', 'pad_masked')

# Key ConcatenatedBatchedDataset attaches to an example to carry its loss weight to _collate.
# Leading underscore because it is internal plumbing between two methods of this file and must
# never reach prepare_inputs, which would pass it on to the model as if it were a feature.
SAMPLE_WEIGHT_KEY = '_sample_weight'

BATCH_FILL_DEFAULTS = {
    'batch_fill_strategy': 'drop',
    'fill_rotate_per_epoch': True,
    'undersized_bucket': 'pad_masked',
    'min_real_fraction': 0.25,
}


def resolve_batch_fill_config(dataset_config, defaults=None, overrides=None):
    """Read and check the four batch-fill keys, raising on anything unrecognised.

    A typo in a strategy name has to fail here rather than fall back to a default: the two
    values differ by whether data is silently dropped, and a run that quietly took the other
    branch is a run whose epoch composition nobody can reconstruct afterwards.

    Three layers, lowest first. `defaults` lets eval change a default without restating the
    table -- eval wants fill_rotate_per_epoch off, because a metric that moves between epochs
    for reasons unrelated to the model is worse than one computed on a fixed tail. Then the
    dataset config, which is where the setting belongs. Then `overrides` from the training
    config, which wins: several training configs share one dataset TOML, so without a layer
    above it there is no way to turn this on for one run without turning it on for all of them.
    """
    resolved = dict(BATCH_FILL_DEFAULTS)
    if defaults:
        resolved.update(defaults)
    for key in BATCH_FILL_DEFAULTS:
        if key in dataset_config:
            resolved[key] = dataset_config[key]
    if overrides:
        resolved.update({k: v for k, v in overrides.items() if k in BATCH_FILL_DEFAULTS})

    if resolved['batch_fill_strategy'] not in BATCH_FILL_STRATEGIES:
        raise ValueError(
            f"batch_fill_strategy must be one of {list(BATCH_FILL_STRATEGIES)}, got "
            f"{resolved['batch_fill_strategy']!r}"
        )
    if resolved['undersized_bucket'] not in UNDERSIZED_BUCKET_STRATEGIES:
        raise ValueError(
            f"undersized_bucket must be one of {list(UNDERSIZED_BUCKET_STRATEGIES)}, got "
            f"{resolved['undersized_bucket']!r}"
        )
    if not isinstance(resolved['fill_rotate_per_epoch'], bool):
        raise ValueError(
            f"fill_rotate_per_epoch must be a boolean, got {resolved['fill_rotate_per_epoch']!r}"
        )
    fraction = resolved['min_real_fraction']
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError(f'min_real_fraction must be a number, got {fraction!r}')
    if not 0 <= fraction <= 1:
        raise ValueError(f'min_real_fraction must be between 0 and 1, got {fraction}')

    rotate_was_set = ('fill_rotate_per_epoch' in dataset_config
                      or 'fill_rotate_per_epoch' in (overrides or {}))
    if resolved['batch_fill_strategy'] == 'drop' and rotate_was_set:
        if is_main_process():
            logger.warning(
                'fill_rotate_per_epoch is set but batch_fill_strategy is '
                f"{resolved['batch_fill_strategy']!r}, so it has no effect. It only rotates the "
                'samples used to fill a short final batch, and nothing is filled under drop.'
            )
    return resolved


def _content_digest(dataset, column, upto=None):
    """Hash of the first `upto` values of `column`, or all of them when upto is None.

    list(), not dataset[column], for the same reason the fingerprint code below spells it out:
    indexing returns a lazy Column holding a reference to its parent dataset.
    """
    values = list(dataset[column])
    return Hasher.hash(values if upto is None else values[:upto])


def _map_and_cache(dataset, map_fn, cache_dir, cache_file_prefix='', new_fingerprint_args=None,
                   regenerate_cache=False, caching_batch_size=1, fingerprint_columns=None,
                   keep_on_fingerprint_change=False, identity=None, content_column=None,
                   require_identity=False):
    new_fingerprint_args = [] if new_fingerprint_args is None else list(new_fingerprint_args)
    if fingerprint_columns is None:
        new_fingerprint_args.append(dataset._fingerprint)
    else:
        # Fingerprint only the columns this cache's contents actually depend on. The dataset's
        # own fingerprint covers every column, captions included, so anything that rewrites the
        # caption text -- a prefix, tag dropout, a different shuffle count -- moved the latent
        # fingerprint too and re-encoded the entire dataset through a VAE that had not changed.
        hasher = Hasher()
        for column in sorted(fingerprint_columns):
            hasher.update(column)
            # list(), not dataset[column]: indexing returns a lazy Column that holds a reference
            # to its parent dataset, so hashing it drags the whole dataset -- captions included
            # -- back into the digest and defeats the entire point of selecting columns.
            hasher.update(list(dataset[column]))
        new_fingerprint_args.append(hasher.hexdigest())
    new_fingerprint = Hasher.hash(new_fingerprint_args)
    if cache_file_prefix:
        cache_dir = cache_dir / cache_file_prefix.strip('_')

    cache = Cache(cache_dir, new_fingerprint, shard_size_gb=10,
                  keep_on_fingerprint_change=keep_on_fingerprint_change, identity=identity,
                  content_digest=_content_digest(dataset, content_column) if content_column else None,
                  require_identity=require_identity)

    if map_fn is None:
        # loading directly from cache without mapping
        assert new_fingerprint == cache.fingerprint
        if content_column is not None:
            recorded = cache.recorded_content_digest()
            if (len(cache) != len(dataset)
                    or (recorded is not None and recorded != _content_digest(dataset, content_column))):
                raise RuntimeError(
                    f'Cannot load {cache_dir} without an encoder: cached rows do not match '
                    f'the current {content_column}. Rebuild this cache with its map function.'
                )
        return cache

    if regenerate_cache:
        cache.clear()

    # Cache has either been cleared if fingerprint didn't match, or has some (maybe 0) existing items in it.

    # Skip existing items
    cache_size = len(cache)
    dataset_size = len(dataset)
    if cache.reused_after_fingerprint_change and cache_size != dataset_size:
        # keep_* can only preserve an old cache when every entry still lines up with the
        # current dataset. A different count proves that is not true. In particular, selecting
        # fewer rows used to leave a larger cache here and fail at the assertion below; adding
        # rows was allowed to append, but that still trusted an unverifiable old prefix. Rebuild
        # both directions and reserve incremental continuation for an interrupted cache whose
        # fingerprint has not changed.
        logger.warning(
            f'Existing cache has {cache_size} entries but the current dataset has '
            f'{dataset_size} rows after its fingerprint changed. The old cache cannot be '
            'reused safely; regenerating it.'
        )
        cache.clear()
        cache_size = len(cache)
    if content_column is not None and cache_size:
        # The entries about to be reused were computed from the caller's input as it was when
        # this cache was last completed. The count check above has already ruled out additions
        # and removals. Editing a row in place is still unsafe: the entry would be kept and
        # silently paired with the new text.
        recorded = cache.recorded_content_digest()
        if recorded is not None and recorded != _content_digest(dataset, content_column, cache_size):
            print(
                f'[CACHE] The {content_column} these {cache_size} cached entries were built '
                'from has changed, so they no longer describe this run. Rebuilding.'
            )
            cache.clear()
            cache_size = len(cache)
    assert cache_size <= dataset_size
    if cache_size == dataset_size:
        # Record the identity here too, not only after a map. A cache that is already complete
        # never reaches the write below, so without this it never acquires a manifest at all --
        # and check_identity treats a cache with no manifest as compatible. The protection
        # would therefore stay permanently inert on exactly the installs it was written for:
        # one that already has a warm cache, changes llm_path, and gets the old encoder's
        # embeddings back with no error. The contents are complete at this point, so claiming
        # them is true; write_manifest records nothing when there is nothing to record.
        cache.write_manifest()
        return cache
    dataset = dataset.select(range(cache_size, dataset_size), keep_in_memory=True)

    pool = None
    manager = None
    if NUM_PROC == 1:
        def wrapper(example):
            return map_fn(example, 0)
    else:
        # Let each worker process know its rank.
        manager = mp.Manager()
        id_queue = manager.Queue()

        def init(queue):
            global rank
            rank = queue.get()

        for i in range(NUM_PROC):
            id_queue.put(i)
        pool = mp.Pool(NUM_PROC, init, (id_queue,))

        def wrapper(example):
            global rank
            return map_fn(example, rank)

    # Tensor slices reference the entire memory of the original tensor, and everything would be pickled and stored
    # in cache, so we do this.
    def recursive_clone_tensors(obj):
        if torch.is_tensor(obj):
            return obj.clone()
        elif isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = recursive_clone_tensors(v)
            return obj
        elif isinstance(obj, (list, tuple)):
            return [recursive_clone_tensors(x) for x in obj]
        else:
            return obj

    def unbatch_iter(batch):
        length = len(next(iter(batch.values())))
        for i in range(length):
            result = {}
            for key in batch:
                result[key] = batch[key][i]
            yield recursive_clone_tensors(result)

    completed_batches = cache_size // caching_batch_size
    total_batches = dataset_size // caching_batch_size

    map_iter = (pool.imap(wrapper, dataset.iter(batch_size=caching_batch_size)) if pool
                else map(wrapper, dataset.iter(batch_size=caching_batch_size)))
    for batch in tqdm(map_iter, initial=completed_batches, total=total_batches):
        for example in unbatch_iter(batch):
            cache.add(example)

    if pool is not None:
        pool.close()
        pool.join()
        manager.shutdown()
    cache.finalize_current_shard()
    # Written only now: the manifest describes complete contents, so a run interrupted
    # part-way leaves no claim about what this cache holds.
    cache.write_manifest()
    return cache


class TextEmbeddingDataset:
    def __init__(self, te_dataset, flattened_captions):
        self.te_dataset = te_dataset
        self.flattened_captions = flattened_captions
        self.image_spec_to_te_idx = defaultdict(list)
        # TODO: maybe make this use Dataset object like the latents. But, you won't be caching text embeddings
        # when training on very large datasets, so perhaps it doesn't really matter.
        for i, image_spec in enumerate(flattened_captions['image_spec']):
            self.image_spec_to_te_idx[tuple(image_spec)].append(i)

    def get_text_embeddings(self, image_spec, caption_number):
        return self.te_dataset[self.image_spec_to_te_idx[image_spec][caption_number]]

    def get_caption(self, image_spec, caption_number):
        """The text these embeddings were encoded from."""
        return self.flattened_captions[self.image_spec_to_te_idx[image_spec][caption_number]]['caption']


def _save_dataset_atomically(dataset, path, replace=False):
    """save_to_disk under a temporary name, then renamed into place, so a run killed part-way
    (or a network volume that drops a write) never leaves a directory that looks complete.
    Without replace, a copy another process published first is kept."""
    path = Path(path)
    host = socket.gethostname()  # ranks of several nodes share a volume
    _remove_dead_temporaries(path, host)
    tag = f'{host}-{os.getpid()}'
    temporary = path.with_name(f'.{path.name}.partial-{tag}')
    if temporary.exists():
        shutil.rmtree(temporary)
    dataset.save_to_disk(str(temporary))
    for file in temporary.iterdir():
        if file.is_file():
            with open(file, 'rb') as handle:
                os.fsync(handle.fileno())
    fsync_dir(temporary)
    previous = None
    if replace and path.exists():
        # A directory cannot be renamed over a non-empty one. Interrupted here, the path is
        # missing and the next run builds it again.
        previous = path.with_name(f'.{path.name}.previous-{tag}')
        os.rename(path, previous)
    try:
        os.rename(temporary, path)
    except OSError:
        if not path.exists():
            raise
        shutil.rmtree(temporary)  # another process published the same contents first
    fsync_dir(path.parent)
    if previous is not None:
        # On NFS a file still open somewhere cannot be removed (it is renamed to .nfsXXXX and
        # the directory stays non-empty). What is left is removed by a later save.
        shutil.rmtree(previous, ignore_errors=True)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_dead_temporaries(path, host):
    """Remove what an interrupted _save_dataset_atomically of path on this host left behind:
    its partial copy, or the previous copy it had moved aside. Only for processes that are
    gone; another host's are left alone (their pids mean nothing here)."""
    for leftover in path.parent.glob(f'.{path.name}.*-{host}-*'):
        kind, _, pid = leftover.name[len(path.name) + 2:].rpartition('-')
        if kind not in (f'partial-{host}', f'previous-{host}') or not pid.isdigit():
            continue
        if int(pid) != os.getpid() and not _pid_alive(int(pid)):
            shutil.rmtree(leftover, ignore_errors=True)


# Flattened-caption paths this process already rebuilt for --regenerate_cache: each text
# encoder asks for the same one, and the earlier encoder's dataset still has it open.
_regenerated_paths = set()
# False in the caching rounds of the second and later text encoders on several nodes: the first
# encoder's round rebuilt every flattened caption dataset of this caching run already.
_regenerate_flattened = True


def _flattened_captions(metadata_dataset, flatten_captions, path, regenerate=False):
    """The flattened (and rewritten) captions, kept at path. Rewriting reads every layout
    caption's image header, so it is done once, not on every rank at every launch; regenerate
    (--regenerate_cache) does it again, for images replaced under the same name."""
    regenerate = regenerate and _regenerate_flattened and str(path) not in _regenerated_paths
    if path.exists() and not regenerate:
        return datasets.load_from_disk(str(path))
    flattened = metadata_dataset.map(flatten_captions, batched=True, keep_in_memory=True,
                                     remove_columns=metadata_dataset.column_names)
    _save_dataset_atomically(flattened, path, replace=regenerate)
    if regenerate:
        _regenerated_paths.add(str(path))
    return datasets.load_from_disk(str(path))


def _has_content_digest(cache_path):
    """Whether a tensor cache recorded the digest of the inputs it encoded (ReadOnlyCache reads
    the same field). An unreadable volume raises; a manifest that does not parse claims nothing."""
    manifest = Path(cache_path) / 'cache_manifest.json'
    if not manifest.exists():
        return False
    try:
        record = json.loads(manifest.read_text(encoding='utf-8'))
    except ValueError:
        return False
    return isinstance(record, dict) and bool(record.get('content_digest'))


def _cache_text_embeddings(metadata_dataset, map_fn, i, cache_dir, regenerate_cache,
                           caching_batch_size, text_encoder_key='',
                           keep_on_fingerprint_change=False, identity=None,
                           profile_sources=None, profile_lookup_dir=None, layout_rewriter=None,
                           require_identity=False):
    """layout_rewriter (a LayoutCaptionRewriter) rewrites each layout caption for its row's
    media and size bucket before it is encoded; the metadata must then have size_bucket set."""

    def flatten_captions(example):
        result = {key: [] for key in example}
        for i, captions in enumerate(example['caption']):
            for caption in captions:
                if layout_rewriter is not None:
                    caption = layout_rewriter(caption, example['image_spec'][i], example['size_bucket'][i],
                                              example['is_video'][i])
                result['caption'].append(caption)
                for key, value in example.items():
                    if key == 'caption':
                        continue
                    result[key].append(value[i])
        return result

    if profile_sources is not None:
        # The flattened text depends on the rewrite, so its files are named after it too.
        rewrite_tag = ('' if layout_rewriter is None else
                       '_' + Hasher.hash([layout_rewriter.cache_tag, layout_rewriter.caption_prefix]))
        flattened_captions = _flattened_captions(
            metadata_dataset, flatten_captions,
            cache_dir / f'flattened_captions_{metadata_dataset._fingerprint}{rewrite_tag}',
            regenerate=regenerate_cache and map_fn is not None)
        def donors():
            result = []
            for source, path in profile_sources():
                if len(source) and isinstance(source[0]['caption'], str):
                    result.append((source, path))
                    continue
                flattened = _flattened_captions(
                    source, flatten_captions, profile_lookup_dir / f'flattened_{source._fingerprint}{rewrite_tag}')
                if layout_rewriter is not None and not _has_content_digest(path) and any(
                        is_layout_caption(c) for captions in source['caption'] for c in captions):
                    # This donor's metadata holds the caption files' text, and the code that
                    # filled it may or may not have rewritten the layout ones before encoding.
                    # Only its recorded content digest tells which, and it has none.
                    print(f'[CACHE] {path} has layout captions and no content digest; not reusing it.')
                    continue
                result.append((flattened, path))
            return result
        te_dataset = tensor_profile(
            flattened_captions, map_fn,
            cache_dir / f'text_embeddings_{i}_{Hasher.hash([identity, text_encoder_key])}',
            sorted(flattened_captions.column_names), identity, donors, _map_and_cache,
            caching_batch_size=caching_batch_size, regenerate_cache=regenerate_cache,
            encoder_key=text_encoder_key, lookup_dir=profile_lookup_dir)
        return IndexedTextEmbeddingDataset(
            te_dataset, flattened_captions, cache_dir / f'caption_index_{metadata_dataset._fingerprint}.sqlite')
    if layout_rewriter is None:
        flattened_captions = metadata_dataset.map(flatten_captions, batched=True, keep_in_memory=True, remove_columns=metadata_dataset.column_names)
    else:
        rewrite_tag = Hasher.hash([layout_rewriter.cache_tag, layout_rewriter.caption_prefix])
        flattened_captions = _flattened_captions(
            metadata_dataset, flatten_captions,
            cache_dir / f'flattened_captions_{metadata_dataset._fingerprint}_{rewrite_tag}',
            regenerate=regenerate_cache and map_fn is not None)
    te_dataset = _map_and_cache(
        flattened_captions,
        map_fn,
        cache_dir,
        cache_file_prefix=f'text_embeddings_{i}_',
        keep_on_fingerprint_change=keep_on_fingerprint_change,
        identity=identity,
        require_identity=require_identity,
        # text_encoder_key identifies which text encoder produced these embeddings, and is
        # deliberately only in this fingerprint: latents are cached separately, so swapping
        # the text encoder must not invalidate them. Models that supply no key keep their
        # original fingerprint exactly, so existing caches stay valid.
        new_fingerprint_args=[i, text_encoder_key] if text_encoder_key else [i],
        regenerate_cache=regenerate_cache,
        caching_batch_size=caching_batch_size,
        # keep_text_embedding_cache tolerates a moved fingerprint, and for this cache the
        # caption text is what the fingerprint is made of. Without this the flag's only
        # reachable effect would be to reuse embeddings of captions that no longer exist.
        content_column='caption',
    )
    assert len(te_dataset) == len(flattened_captions)
    return TextEmbeddingDataset(te_dataset, flattened_captions)


CAPTION_SAMPLING_MODES = ('all', 'random_per_epoch')

# Settings that change the caption text stored in the metadata cache, with their defaults. The
# cache lives at a fixed path and is reused under --trust_cache, so without this a run that
# flips one of them reads back captions built under the old setting: raw text with its tag
# marker intact fed straight to the text encoder, or caption_prefix applied a second time.
#
# The suffix is empty when every one of them is at its default, so caches written before these
# settings existed stay valid. That is the same rule text_encoder_cache_key follows.
CAPTION_CACHE_SETTINGS = {
    'augment_at_runtime': False,
    'prefix_tag_caption': '',
    'tag_dropout_rate': 0.0,
    'multiline_captions': False,
    'enable_remove_non_latin': False,
    # Set for a model with a layout_caption_format: its JSON captions skip the tag operations
    # and keep caption_prefix for the per-bucket rewrite.
    'layout_captions': '',
}


# Caption settings that predate CAPTION_CACHE_SETTINGS. They change the cached caption text
# exactly as much as the ones above, but they are deliberately NOT part of the suffix: they
# already existed when the suffix was introduced, so putting them in it would move the cache
# path of every install that uses them. They are recorded alongside the cache instead, and a
# mismatch is reported rather than silently served.
LEGACY_CAPTION_SETTINGS = {
    'caption_prefix': '',
    'cache_shuffle_num': 0,
    'cache_shuffle_delimiter': ', ',
    'skip_empty_caption': True,
}


def caption_cache_suffix(settings):
    non_default = {k: v for k, v in settings.items() if v != CAPTION_CACHE_SETTINGS[k]}
    if not non_default:
        return ''
    digest = hashlib.md5(json.dumps(non_default, sort_keys=True, default=str).encode()).hexdigest()
    return '_' + digest[:12]


def collapse_to_one_entry_per_image(iteration_order_list):
    """Group (image_spec, latents_idx, caption, caption_number) rows by image.

    Used by caption_sampling = 'random_per_epoch'. Instead of one entry per caption, each image
    gets a single entry carrying all of its captions and their cache indices; __getitem__ draws
    one per access. An epoch becomes len(images) rather than len(images) * len(captions), and an
    image gets a different caption on each pass.

    Every caption's embedding is still cached -- selection is an index into that cache -- so
    this costs nothing extra to cache. Keeping the caption and its index together is what lets
    __getitem__ pick both with one draw, which models that read the text as well as the cached
    embedding (HiDream) depend on.
    """
    by_image = {}
    for image_spec, latents_idx, caption, caption_number in iteration_order_list:
        key = tuple(image_spec)
        if key not in by_image:
            by_image[key] = (image_spec, latents_idx, [], [])
        by_image[key][2].append(caption)
        by_image[key][3].append(caption_number)
    return list(by_image.values())


class OnTheFlyMediaDataset:
    """Stands in for a latent cache when the model reads its media at __getitem__ time.

    Row i is metadata row i decoded and resized now, with the same functions and the same
    output layout the caching path would have stored: {latents, mask, image_spec, caption}.
    Used by models with load_media_on_the_fly (iris), whose 'latents' are just the pixels,
    so caching them would only duplicate the dataset on disk.
    """

    def __init__(self, metadata_dataset, media_loader, encode_fn):
        if 'control_file' in metadata_dataset.column_names:
            raise NotImplementedError('Reading media on the fly does not support control/edit datasets')
        self.metadata_dataset = metadata_dataset.select_columns(['image_spec', 'mask_file', 'size_bucket', 'caption'])
        self.media_loader = media_loader
        self.encode_fn = encode_fn

    def __len__(self):
        return len(self.metadata_dataset)

    def __getitem__(self, idx):
        row = self.metadata_dataset[int(idx)]
        items = self.media_loader(row['image_spec'], row['mask_file'], row['size_bucket'])
        if len(items) != 1:
            raise RuntimeError(f'Expected exactly one item from {row["image_spec"]}, got {len(items)}')
        tensor, _audio, mask = items[0]
        latents = self.encode_fn(tensor.unsqueeze(0))['latents'][0]
        return {'latents': latents, 'mask': mask, 'image_spec': row['image_spec'], 'caption': row['caption']}


def shuffle_in_memory(dataset, seed):
    """dataset.shuffle(seed=seed), with the same fingerprint (the caches key on it), but without
    writing an indices cache file next to the dataset. Every rank of every node shuffles every
    bucket; on a shared volume they would write that file and rename it over each other, and over
    other nodes' memory maps of it."""
    # The fingerprint depends only on the source fingerprint and the arguments: take it from a
    # one-row stand-in, which datasets shuffles in memory anyway.
    stand_in = datasets.Dataset.from_dict({'row': [0]})
    stand_in._fingerprint = dataset._fingerprint
    fingerprint = stand_in.shuffle(seed=seed)._fingerprint
    return dataset.shuffle(seed=seed, keep_in_memory=True, new_fingerprint=fingerprint)


# The smallest unit of a dataset. Represents a single size bucket from a single folder of images
# and captions on disk. Not batched; returns individual items.
class SizeBucketDataset:
    def __init__(self, metadata_dataset, directory_config, size_bucket, cache_base, directory_dataset):
        # Shuffle deterministically based on size bucket, so that two resolutions of the same aspect ratio get different
        # orders, which mixes data better when training on multiple resolutions at once.
        # tuple(): seed_from_hash hashes str(), and a grouping key read back from
        # grouping_keys.json is a list. '[64, 64, 1]' and '(64, 64, 1)' give different orders,
        # so a trusted warm load would reorder the rows, move the latent fingerprint and
        # re-encode the whole bucket. A freshly grouped key is a tuple, as it always was.
        seed = seed_from_hash(tuple(size_bucket))
        metadata_dataset = shuffle_in_memory(metadata_dataset, seed)
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.size_bucket = size_bucket
        self.path = Path(self.directory_config['path'])
        self.cache_dir = cache_base / f'cache_{bucket_suffix(size_bucket)}'
        self.captions_dict = directory_dataset.captions_dict  # optional
        # Kept so the cache identity can be reached without threading it through both
        # SizeBucketDataset construction sites.
        self.directory_dataset = directory_dataset

        if len(size_bucket) == 4:
            # rename old folder name to the new one for convenience
            old_cache_dir = cache_base / f'cache_{bucket_suffix(size_bucket[1:])}'
            if old_cache_dir.exists() and not self.cache_dir.exists():
                old_cache_dir.rename(self.cache_dir)

        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_embedding_datasets = []
        self.uncond_text_embeddings = []
        self.encode_text_in_loader = directory_config.get('encode_text_in_loader', False)
        self.num_repeats = self.directory_config['num_repeats']
        self.shuffle_skip = max(directory_config.get('cache_shuffle_num', 0), 1) # Should be provided in DirectoryDataset
        # Only used on the online_captions path, where the caption is chosen at __getitem__ time
        # and so can still be augmented. On the cached path the embedding is already computed,
        # so shuffling and dropout have to happen at cache time instead (shuffle_captions).
        self.caption_sampling = directory_config.get('caption_sampling', 'all')
        self.online_shuffle = directory_config.get('cache_shuffle_num', 0) > 0 or directory_config.get('shuffle_tags', False)
        self.augment_at_runtime = directory_config.get('augment_at_runtime', False)
        self.caption_prefix = directory_config.get('caption_prefix', '')
        self.online_delimiter = directory_config.get('cache_shuffle_delimiter', ', ')
        self.prefix_tag_caption = directory_config.get('prefix_tag_caption', '')
        self.tag_dropout_rate = directory_config.get('tag_dropout_rate', 0.0)
        self.enable_remove_non_latin = directory_config.get('enable_remove_non_latin', False)
        self.caption_dropout_rate = directory_config.get('caption_dropout_rate', None)
        # Rewrites JSON layout captions for this bucket (utils/layout_captions.py); None for a
        # model without a layout_caption_format.
        self.layout_rewriter = getattr(directory_dataset, 'layout_rewriter', None)
        if self.num_repeats <= 0:
            raise ValueError(f'num_repeats must be >0, was {self.num_repeats}')

    @property
    def vae_identity(self):
        # Reached through the directory dataset so the two SizeBucketDataset construction sites
        # do not both have to grow an argument. Absent in tests that build one standalone.
        return getattr(self.directory_dataset, 'vae_identity', '')

    @property
    def text_embedding_identity(self):
        return getattr(self.directory_dataset, 'text_embedding_identity', '')

    def cache_latents(self, map_fn, regenerate_cache=False, trust_cache=False, caching_batch_size=1):
        on_the_fly_media = getattr(self.directory_dataset, 'on_the_fly_media', None)
        if on_the_fly_media is not None:
            # Nothing to cache: rows are decoded from the media file when drawn. The iteration
            # order is still built (and cached) exactly as below, indexed by the same metadata
            # rows the decoder reads, so everything downstream is unchanged.
            print(f'media read on the fly (nothing cached): {self.size_bucket}')
            self.latent_dataset = OnTheFlyMediaDataset(self.metadata_dataset, *on_the_fly_media)
            self._build_iteration_order(regenerate_cache, trust_cache)
            return
        print(f'caching latents: {self.size_bucket}')
        # Latents depend on the image, its mask, its control image and the size bucket -- never
        # on the caption, which is only carried alongside them. Fingerprinting the caption
        # column too meant every caption setting change wiped the whole VAE cache.
        latent_columns = [c for c in self.metadata_dataset.column_names if c != 'caption']
        if getattr(self.directory_dataset, 'reuse_metadata_cache', False):
            self.latent_dataset = tensor_profile(
                self.metadata_dataset, map_fn,
                self.cache_dir / f'latent_profile_{Hasher.hash(self.vae_identity)}',
                sorted(latent_columns), self.vae_identity,
                lambda: self.directory_dataset.legacy_tensor_sources(self.size_bucket, 'latents'),
                _map_and_cache, caching_batch_size=caching_batch_size,
                regenerate_cache=regenerate_cache,
                lookup_dir=self.directory_dataset.snapshot_dir / 'tensor_lookups')
            trust_cache = True
        else:
            self.latent_dataset = _map_and_cache(
                self.metadata_dataset, map_fn, self.cache_dir, cache_file_prefix='latents_',
                regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size,
                fingerprint_columns=latent_columns,
                keep_on_fingerprint_change=self.directory_config.get('keep_latent_cache', False),
                identity=self.vae_identity,
                content_column=('image_spec' if self.directory_config.get('require_non_latin_caption') is not None
                                else None))
        assert len(self.latent_dataset) == len(self.metadata_dataset), (len(self.latent_dataset), len(self.metadata_dataset))
        self._build_iteration_order(regenerate_cache, trust_cache)

    def _build_iteration_order(self, regenerate_cache, trust_cache):
        suffix = self.directory_config.get('caption_cache_suffix', '')
        if self.caption_sampling != 'all':
            suffix += '_' + self.caption_sampling
        iteration_order_cache_dir = self.cache_dir / f'iteration_order{suffix}'

        if regenerate_cache or not iteration_order_cache_dir.exists() or not trust_cache:
            print('Building iteration order')
            image_spec_to_latents_idx = {
                tuple(image_spec): i
                for i, image_spec in enumerate(self.metadata_dataset['image_spec'])
            }

            equal_num_captions = True
            num_captions = None
            for example in self.metadata_dataset.select_columns(['caption']):
                n = len(example['caption'])
                if num_captions is not None and n != num_captions:
                    equal_num_captions = False
                    break
                num_captions = n

            if equal_num_captions:
                # If all images have the same number of captions, set things up so we read (mostly) sequentially off disk. The metadata was already shuffled in the beginning.
                iteration_order_by_caption_num = [[] for _ in range(num_captions)]
                seed = 0
                for example in self.metadata_dataset.select_columns(['image_spec', 'caption']):
                    image_spec = example['image_spec']
                    # Shuffle the captions WITH their original positions. The text embedding
                    # cache is indexed by position in the unshuffled metadata order
                    # (_cache_text_embeddings flattens metadata_dataset['caption'] as it
                    # stands), so shuffling the strings alone and then using the post-shuffle
                    # position as caption_number hands back the embedding of a different
                    # caption. Harmless for a model that ignores the caption string, wrong for
                    # one that reads both -- HiDream tokenizes Llama3 from the live caption
                    # while using cached CLIP and T5 embeddings.
                    numbered = list(enumerate(example['caption']))
                    shuffle_with_seed(numbered, seed)
                    seed += 1
                    latents_idx = image_spec_to_latents_idx[tuple(image_spec)]
                    for i, (caption_number, caption) in enumerate(numbered):
                        iteration_order_by_caption_num[i].append((image_spec, latents_idx, caption, caption_number))
                iteration_order_list = []
                for l in iteration_order_by_caption_num:
                    iteration_order_list.extend(l)
            else:
                iteration_order_list = []
                for example in self.metadata_dataset.select_columns(['image_spec', 'caption']):
                    image_spec = example['image_spec']
                    captions = example['caption']
                    latents_idx = image_spec_to_latents_idx[tuple(image_spec)]
                    for i, caption in enumerate(captions):
                        iteration_order_list.append((image_spec, latents_idx, caption, i))
                shuffle_with_seed(iteration_order_list, 42)

            if self.caption_sampling == 'random_per_epoch':
                iteration_order_list = collapse_to_one_entry_per_image(iteration_order_list)

            iteration_order_dict = defaultdict(list)
            for image_spec, latents_idx, caption, caption_number in iteration_order_list:
                iteration_order_dict['image_spec'].append(image_spec)
                iteration_order_dict['latents_idx'].append(latents_idx)
                if self.caption_sampling == 'random_per_epoch':
                    iteration_order_dict['captions'].append(caption)
                    iteration_order_dict['caption_numbers'].append(caption_number)
                else:
                    iteration_order_dict['caption'].append(caption)
                    iteration_order_dict['caption_number'].append(caption_number)
            iteration_order = datasets.Dataset.from_dict(iteration_order_dict)
            iteration_order.save_to_disk(str(iteration_order_cache_dir))
            del iteration_order

        self.iteration_order = datasets.load_from_disk(str(iteration_order_cache_dir))

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1,
                              text_encoder_key='', keep_text_embedding_cache=False,
                              identity=None, require_identity=False):
        print(f'caching text embeddings: {self.size_bucket}')
        te_dataset = _cache_text_embeddings(
            self.metadata_dataset, map_fn, i, self.cache_dir, regenerate_cache,
            caching_batch_size, text_encoder_key,
            keep_on_fingerprint_change=keep_text_embedding_cache,
            identity=identity,
            require_identity=require_identity,
            profile_sources=((lambda: self.directory_dataset.legacy_tensor_sources(
                self.size_bucket, f'text_embeddings_{i}'))
                if getattr(self.directory_dataset, 'reuse_metadata_cache', False) else None),
            profile_lookup_dir=(self.directory_dataset.snapshot_dir / 'tensor_lookups'
                                if getattr(self.directory_dataset, 'reuse_metadata_cache', False) else None),
            layout_rewriter=self.layout_rewriter,
        )
        self.text_embedding_datasets.append(te_dataset)

    def _rewrite_layout(self, caption, image_spec):
        """A layout caption as this bucket trains it, for the paths that tokenize the caption
        string at __getitem__ time (the cached path rewrites before encoding)."""
        if self.layout_rewriter is None or not caption:
            return caption
        # (w, h, frames), also for an AR bucket's (ar, w, h, frames) naming key.
        size_bucket = tuple(self.size_bucket)[-3:]
        return self.layout_rewriter(caption, tuple(image_spec), size_bucket, size_bucket[-1] > 1)

    def add_text_embedding_dataset(self, te_dataset):
        self.text_embedding_datasets.append(te_dataset)

    @property
    def _augment_at_runtime(self):
        """Whether __getitem__ should shuffle and drop tags itself.

        Set by DirectoryDataset when the model caches no text embeddings: the caption string is
        then what gets tokenized every step, so a fresh draw per access costs nothing and beats
        the fixed variants baked in at cache time. In that case the metadata holds the captions
        raw -- marker intact -- so tag lists can still be told from prose here.

        A model that caches, or caches only some of its encoders as HiDream does, must never
        have its text re-augmented behind a frozen embedding, which is exactly what the
        text_embedding_datasets check below prevents.
        """
        return self.augment_at_runtime and not self.text_embedding_datasets

    def __getitem__(self, idx):
        idx = idx % len(self.iteration_order)
        entry = self.iteration_order[idx]

        ret = self.latent_dataset[entry['latents_idx']]

        if self.caption_sampling == 'random_per_epoch':
            # Every caption of this image is cached; pick which one this pass uses. The same
            # index selects both the text and the cached embedding, so the two never disagree
            # -- which matters for models like HiDream that read both.
            pick = random.randrange(len(entry['captions']))
            caption_number = entry['caption_numbers'][pick]
            entry_caption = entry['captions'][pick]
        else:
            caption_number = entry['caption_number']
            entry_caption = entry['caption']

        # A directory override (including explicit zero) wins over the dataset setting. Only
        # an absent setting falls back to the legacy training-wide probability; never draw twice.
        dropout_rate = UNCOND_FRACTION if self.caption_dropout_rate is None else self.caption_dropout_rate
        use_uncond = dropout_rate > 0 and random.random() < dropout_rate
        if use_uncond:
            caption = ''
        else:
            if self.captions_dict and not self.text_embedding_datasets:
                # A cached encoder fixes the text as well as its embedding. Reading an edited
                # live caption here would desynchronise hybrid models such as HiDream, and a
                # cache-shuffled caption index need not even exist in the raw caption list.
                tar_file, image_file = entry['image_spec']
                # Match how DirectoryDataset keys captions.json: a tar member by its full path
                # inside the archive, a plain file by basename. Using the full on-disk path for
                # both made every lookup miss on an ordinary image directory, so every caption
                # became ''.
                # os.path.basename, not a split on '/': a tar member always uses forward
                # slashes, but an on-disk path uses the platform separator, and splitting a
                # Windows path on '/' returns the whole path and misses every lookup.
                key = image_file if tar_file is not None else os.path.basename(image_file)
                if key in self.captions_dict:
                    # Metadata contains shuffle_skip variants per original caption, including
                    # on-the-fly variants. The dictionary is still the unexpanded source list.
                    raw_caption_number = caption_number // self.shuffle_skip
                    caption = self.captions_dict[key][raw_caption_number]
                    if self._augment_at_runtime:
                        caption = preprocess_caption(
                            caption,
                            delimiter=self.online_delimiter,
                            caption_prefix=self.caption_prefix,
                            prefix_tag_caption=self.prefix_tag_caption,
                            shuffle=self.online_shuffle,
                            tag_dropout_rate=self.tag_dropout_rate,
                            enable_remove_non_latin=self.enable_remove_non_latin,
                            layout_captions=self.layout_rewriter is not None,
                        )
                    else:
                        # Metadata already applied the deterministic parts, but the online
                        # dictionary holds raw captions. Strip/filter/prefix them once here.
                        caption = preprocess_caption(
                            caption,
                            caption_prefix=self.caption_prefix,
                            prefix_tag_caption=self.prefix_tag_caption,
                            enable_remove_non_latin=self.enable_remove_non_latin,
                            layout_captions=self.layout_rewriter is not None,
                        )
                    caption = self._rewrite_layout(caption, entry['image_spec'])
                else:
                    print(f'WARNING: image {key} did not have entry in captions_dict. Using empty caption.')
                    caption = ''
            elif self.text_embedding_datasets and self.layout_rewriter is not None:
                # The text the cached embeddings were encoded from: a layout caption rewritten
                # for this bucket, so a model that reads the caption too sees the same boxes.
                caption = self.text_embedding_datasets[0].get_caption(tuple(entry['image_spec']), caption_number)
            else:
                caption = entry_caption
                if self._augment_at_runtime:
                    # Nothing was cached, so this string is what the model tokenizes, every
                    # step. Augmenting per sample is free here and better than the fixed
                    # variants baked in at cache time. Models that do cache are excluded:
                    # re-augmenting text whose embedding is frozen would make the two disagree.
                    caption = preprocess_caption(
                        caption,
                        delimiter=self.online_delimiter,
                        caption_prefix=self.caption_prefix,
                        prefix_tag_caption=self.prefix_tag_caption,
                        shuffle=self.online_shuffle,
                        tag_dropout_rate=self.tag_dropout_rate,
                        enable_remove_non_latin=self.enable_remove_non_latin,
                        layout_captions=self.layout_rewriter is not None,
                    )
                if not self.text_embedding_datasets:
                    # Tokenized from this string: rewrite a layout caption for this bucket.
                    caption = self._rewrite_layout(caption, entry['image_spec'])

        for ds, uncond_ds in zip(self.text_embedding_datasets, self.uncond_text_embeddings):
            emb_dict = uncond_ds[0] if use_uncond else ds.get_text_embeddings(tuple(entry['image_spec']), caption_number)
            ret.update(emb_dict)
        ret['caption'] = caption
        if self.encode_text_in_loader:
            if use_uncond:
                # As caching encodes the empty caption: not a video, no control image.
                ret[IS_VIDEO_KEY], ret[CONTROL_FILE_KEY] = False, None
            else:
                # What caching passes to the text encoder with this row's caption. The latents
                # are cached row for row with the metadata.
                row = self.metadata_dataset[int(entry['latents_idx'])]
                assert tuple(row['image_spec']) == tuple(entry['image_spec']), (row['image_spec'], entry['image_spec'])
                ret[IS_VIDEO_KEY], ret[CONTROL_FILE_KEY] = row['is_video'], row.get('control_file')
        return ret

    def __len__(self):
        return int(len(self.iteration_order) * self.num_repeats)


# Logical concatenation of multiple SizeBucketDataset, for the same size bucket. It returns items
# as batches.
class ConcatenatedBatchedDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.post_init_called = False
        self.batch_fill = dict(BATCH_FILL_DEFAULTS)
        self.fill_report = None

    def post_init(self, global_batch_size: dict, global_batch_size_image: dict, data_parallel_rank: int, data_parallel_world_size: int, batch_fill=None):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        if batch_fill is not None:
            self.batch_fill = batch_fill
        iteration_order = []
        size_bucket = self.datasets[0].size_bucket
        for i, ds in enumerate(self.datasets):
            assert ds.size_bucket == size_bucket
            iteration_order.extend([i]*len(ds))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.datasets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = np.array(iteration_order)

        # size_bucket could be [ar, w, h, frame] or [w, h, frames]
        global_batch_size_dict = global_batch_size_image if size_bucket[-1] == 1 else global_batch_size
        if None in global_batch_size_dict:
            # single value
            self.global_batch_size = global_batch_size_dict[None]
        else:
            # per image size batch size
            bucket_size = math.sqrt(size_bucket[-2] * size_bucket[-3])
            min_diff = float('inf')
            for size, bs in global_batch_size_dict.items():
                diff = abs(size - bucket_size)
                if diff < min_diff:
                    min_diff = diff
                    self.global_batch_size = bs

        assert self.global_batch_size % self.data_parallel_world_size == 0
        # size_bucket is [w, h, frames] or [ar, w, h, frames]. Only used to name the bucket in
        # log lines and to shape a synthesised mask, never to compute a batch.
        self.size_bucket = size_bucket
        # (height, width) in pixels, matching what PreprocessMediaFile returns for a real mask.
        # Only ever the fallback in _collate -- a real mask in the batch supplies the shape --
        # and only the value matters downstream, since every model interpolates the mask to the
        # latent size before using it.
        self.mask_shape = (int(size_bucket[-2]), int(size_bucket[-3]))
        if self.batch_fill['batch_fill_strategy'] == 'fill':
            self._fill_to_multiple_of(self.global_batch_size)
        else:
            self._make_divisible_by(self.global_batch_size)
            self.sample_weights = np.ones(len(self.iteration_order), dtype=np.float32)
        self.batch_size = self.global_batch_size // self.data_parallel_world_size
        self.post_init_called = True

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order) // self.global_batch_size

    def __getitem__(self, idx):
        assert self.post_init_called
        block_start = idx * self.global_batch_size
        start_idx = block_start + self.data_parallel_rank * self.batch_size
        end_idx = start_idx + self.batch_size
        examples = [self.datasets[i.item()][j.item()] for i, j in self.iteration_order[start_idx : end_idx]]
        scale = self._batch_weight_scale(block_start)
        if scale is not None:
            for k, example in enumerate(examples):
                example[SAMPLE_WEIGHT_KEY] = scale * float(self.sample_weights[start_idx + k])
        return examples

    def _batch_weight_scale(self, block_start):
        """Loss weight for the real samples of the global batch starting at block_start.

        Computed over the WHOLE global batch, never over this rank's slice or one micro batch.
        loss.mean() divides by the element count including the zeroed padding, so the real
        samples have to be scaled back up by G/G_real to make the step behave like a smaller
        batch at the same learning rate. Deepspeed averages the micro batch losses and the data
        parallel ranks average their gradients, so one constant shared by every micro batch and
        every rank is what makes the arithmetic come out right no matter how unevenly the
        padding happens to be distributed. A per-micro-batch ratio would not.

        Returns None when every sample in the batch is real, which is every batch under 'drop'
        and every batch of a bucket that had enough data. That is deliberate: nothing is
        attached to the examples, so _collate stays on exactly the path it had before batch
        fill existed, mask = None included.
        """
        block = self.sample_weights[block_start : block_start + self.global_batch_size]
        num_real = int(np.count_nonzero(block))
        if num_real == len(block):
            return None
        if num_real == 0:
            # Never reached: a bucket with no real sample in a batch is dropped in
            # _fill_to_multiple_of. Guarded anyway, because the alternative is a division by
            # zero that surfaces as a NaN loss thousands of steps later.
            raise RuntimeError(f'size bucket {self.size_bucket} produced a global batch with no real sample')
        return len(block) / num_real

    def _make_divisible_by(self, n):
        new_length = (len(self.iteration_order) // n) * n
        self.iteration_order = self.iteration_order[:new_length]
        if new_length == 0 and is_main_process():
            logger.warning(f"size bucket {self.datasets[0].size_bucket} is being completely dropped because it doesn't have enough images")

    def _cache_row_identity(self):
        """Read each sub-dataset's latents_idx column once, for the duplicate check below.

        iteration_order is a datasets.Dataset backed by Arrow, and the fill has to ask "which
        image is this?" for potentially every entry of the bucket. Reading the column per entry
        is a million Arrow lookups on a real dataset and turns startup into minutes; reading it
        once is a list.

        list(), not the Column object, for the reason _content_digest spells out: indexing a
        datasets.Dataset hands back a lazy view holding a reference to its parent.
        """
        self._latents_idx = []
        self._num_rows = []
        for ds in self.datasets:
            self._latents_idx.append(list(ds.iteration_order['latents_idx']))
            self._num_rows.append(len(ds.iteration_order))

    def _row_id(self, entry):
        """(sub-dataset, row) -- one row of one SizeBucketDataset, so one (image, caption) pair.

        j indexes SizeBucketDataset, whose __len__ is len(iteration_order) * num_repeats and
        whose __getitem__ takes idx % len(iteration_order). So the num_repeats copies of a row
        all collapse to the same row here, which is what makes a copy count as a repeat rather
        than as a new sample.
        """
        dataset_idx = int(entry[0])
        return (dataset_idx, int(entry[1]) % self._num_rows[dataset_idx])

    def _image_id(self, entry):
        """(sub-dataset, latents_idx) -- one image.

        The sub-dataset index is not optional: several [[directory]] entries land in the same
        size bucket and latents_idx is local to each one, so image 0 of two directories would
        otherwise look like the same image.
        """
        dataset_idx, row = self._row_id(entry)
        return (dataset_idx, self._latents_idx[dataset_idx][row])

    def _fill_to_multiple_of(self, n):
        """Extend iteration_order so it is a multiple of n, instead of truncating to one.

        Only the tail is touched. The first N entries stay exactly what 'drop' would have
        produced, which is what keeps the change to the training order bounded and keeps the
        static part reproducible across epochs; the price is that a duplicate image already
        present inside the static part is left there, exactly as it is today. See
        docs/note/batch-fill-strategies.md.
        """
        self._static_iteration_order = self.iteration_order
        self._cache_row_identity()
        n_static = len(self.iteration_order)
        remainder = n_static % n

        if n_static == 0 or (n_static < n and self.batch_fill['undersized_bucket'] == 'drop'):
            self._dropped = True
            self._make_divisible_by(n)
            self.sample_weights = np.ones(len(self.iteration_order), dtype=np.float32)
            return

        self._dropped = False
        if remainder == 0:
            # Already a whole number of batches. Filling is a no-op, and saying so keeps the
            # startup report honest rather than reporting a fill of zero samples.
            self.sample_weights = np.ones(n_static, dtype=np.float32)
            self.fill_report = self._make_fill_report(n_static, 0, 0)
            return

        self._fill_tail(epoch=0)

    def _fill_tail(self, epoch):
        """Choose the entries that pad the final batch, for this epoch.

        Rebuilt from the static order every time rather than mutated in place, so calling this
        for epoch k gives the same answer whenever it is called. That is what resume needs:
        the order has to be a function of (seed, epoch), never of how many times anything has
        been drawn.
        """
        n = self.global_batch_size
        order = self._static_iteration_order
        n_static = len(order)
        remainder = n_static % n
        num_missing = n - remainder
        tail_start = n_static - remainder

        forbidden_images = set()
        used_rows = set()
        for k in range(tail_start, n_static):
            forbidden_images.add(self._image_id(order[k]))
            used_rows.add(self._row_id(order[k]))

        candidates = list(range(n_static))
        shuffle_with_seed(candidates, seed_from_hash((tuple(self.size_bucket), 'fill', epoch)))

        chosen = []
        weights = []
        # Tier 1: an image not already in this batch. A genuinely new sample.
        for k in candidates:
            if len(chosen) == num_missing:
                break
            image_id = self._image_id(order[k])
            if image_id in forbidden_images:
                continue
            chosen.append(k)
            weights.append(1.0)
            forbidden_images.add(image_id)
            used_rows.add(self._row_id(order[k]))
        # Tier 2: the same image under a different caption. Still a new (image, caption) pair,
        # so it is real compute and is not masked. Note this tier is empty by construction when
        # caption_sampling is 'random_per_epoch', because that collapses every image to one
        # row: a copy of that row would pick its caption at random at access time and there is
        # nothing guaranteeing a different one, so it belongs in tier 3, not here.
        if len(chosen) < num_missing:
            for k in candidates:
                if len(chosen) == num_missing:
                    break
                row_id = self._row_id(order[k])
                if row_id in used_rows:
                    continue
                chosen.append(k)
                weights.append(1.0)
                used_rows.add(row_id)

        num_real = remainder + len(chosen)
        # Tier 3: nothing left but an exact repeat. Masked to zero, so it costs GPU time and
        # teaches the model nothing -- which is why min_real_fraction exists to refuse a batch
        # that is mostly this.
        if len(chosen) < num_missing:
            fraction = num_real / n
            if fraction < self.batch_fill['min_real_fraction']:
                if is_main_process():
                    logger.warning(
                        f'size bucket {self.size_bucket} is being dropped: it can supply only '
                        f'{num_real} real samples for a global batch of {n} '
                        f'({fraction:.3f} < min_real_fraction {self.batch_fill["min_real_fraction"]}), '
                        f'so {n - num_real} of every batch would be masked-out padding.\n'
                        '  To keep it: lower micro_batch_size_per_gpu or '
                        'gradient_accumulation_steps, raise num_repeats, lower num_ar_buckets so '
                        'this bucket merges with a neighbour, or lower min_real_fraction.'
                    )
                self._dropped = True
                self.iteration_order = self._static_iteration_order[:0]
                self.sample_weights = np.ones(0, dtype=np.float32)
                return
            k = 0
            while len(chosen) < num_missing:
                chosen.append(candidates[k % n_static])
                weights.append(0.0)
                k += 1

        self.iteration_order = np.concatenate([order, order[chosen]])
        self.sample_weights = np.concatenate(
            [np.ones(n_static, dtype=np.float32), np.array(weights, dtype=np.float32)]
        )
        self.fill_report = self._make_fill_report(n_static, len(chosen), weights.count(0.0))

    def _make_fill_report(self, n_static, num_added, num_masked):
        n = self.global_batch_size
        return {
            'size_bucket': tuple(self.size_bucket),
            'num_samples': n_static,
            'global_batch_size': n,
            'batches_before': n_static // n,
            'batches_after': (n_static + num_added) // n,
            'num_added': num_added,
            'num_masked': num_masked,
        }

    def set_epoch(self, epoch):
        """Rotate which samples pad the final batch.

        Without this, the same `num_missing` images are seen twice in every epoch forever. On a
        large bucket that is a rounding error; on a small one it is not -- 100 samples at a
        global batch of 64 pads 28 of them, so 28% of the data is systematically oversampled.
        Only the padding moves: the static part stays put, so the change to the training order
        stays bounded by the size of the tail.
        """
        if not self.post_init_called or self._is_static():
            return
        before = len(self.iteration_order)
        self._fill_tail(epoch)
        # The count of tier-1 and tier-2 candidates depends on which images and rows exist, not
        # on the order they are visited in, so the length cannot move between epochs. Asserted
        # because Dataset.post_init built its own iteration order from this length and would
        # silently index off the end if it ever did.
        assert len(self.iteration_order) == before, (len(self.iteration_order), before)

    def _is_static(self):
        return (
            self.batch_fill['batch_fill_strategy'] != 'fill'
            or not self.batch_fill['fill_rotate_per_epoch']
            or getattr(self, '_dropped', False)
            or self.fill_report is None
            or self.fill_report['num_added'] == 0
        )


class ARBucketDataset:
    def __init__(self, ar_frames, resolutions, metadata_dataset, directory_config, cache_base, round_to_multiple, directory_dataset):
        self.ar_frames = ar_frames
        self.resolutions = resolutions
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.size_buckets = []
        self.path = Path(directory_config['path'])
        self.cache_base = cache_base
        self.cache_dir = cache_base / f'ar_frames_{bucket_suffix(self.ar_frames)}'
        self.round_to_multiple = round_to_multiple
        self.directory_dataset = directory_dataset
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_size_bucket_datasets(self):
        return self.size_buckets

    def cache_latents(self, map_fn, regenerate_cache=False, trust_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.ar_frames}')

        for res in self.resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_nearest_multiple(w, self.round_to_multiple)
            h = round_to_nearest_multiple(h, self.round_to_multiple)
            size_bucket = (w, h, self.ar_frames[1])
            # to make sure the directory has a unique name
            naming_size_bucket = (self.ar_frames[0],) + size_bucket
            mapped_path = self.cache_dir / f'metadata/metadata_{bucket_suffix(naming_size_bucket)}.arrow'
            if not getattr(self.directory_dataset, 'reuse_metadata_cache', False):
                # Per caption profile: the copy carries the captions, and the profiles of a legacy
                # directory share its cache directory. The name all profiles once shared may hold
                # another profile's captions.
                mapped_path.unlink(missing_ok=True)
                mapped_path = mapped_path.with_name(
                    f'{mapped_path.stem}{legacy_copy_tag(self.directory_config.get("caption_cache_suffix", ""))}.arrow')
            if (getattr(self.directory_dataset, 'reuse_metadata_cache', False)
                    and mapped_path.exists() and not regenerate_cache):
                metadata_with_size_bucket = datasets.Dataset.from_file(str(mapped_path))
            else:
                metadata_with_size_bucket = self.metadata_dataset.map(
                    lambda example: {'size_bucket': size_bucket},
                    cache_file_name=str(mapped_path),
                    load_from_cache_file=(not regenerate_cache and trust_cache),
                    desc='Adding size bucket')
            if getattr(self.directory_dataset, 'reuse_metadata_cache', False):
                metadata_with_size_bucket = datasets.Dataset.from_file(str(mapped_path))
            self.size_buckets.append(
                SizeBucketDataset(metadata_with_size_bucket, self.directory_config, naming_size_bucket, self.cache_base, self.directory_dataset)
            )

        for ds in self.size_buckets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, trust_cache=trust_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1,
                              text_encoder_key='', keep_text_embedding_cache=False,
                              identity=None, require_identity=False):
        if (getattr(self.directory_dataset, 'layout_rewriter', None) is not None
                and any(is_layout_caption(c) for captions in self.metadata_dataset['caption'] for c in captions)):
            # A layout caption is a different text at each resolution (its boxes follow the
            # crop), so each size bucket of this aspect ratio caches its own embeddings.
            for size_bucket_dataset in self.size_buckets:
                size_bucket_dataset.cache_text_embeddings(
                    map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size,
                    text_encoder_key=text_encoder_key, keep_text_embedding_cache=keep_text_embedding_cache,
                    identity=identity, require_identity=require_identity)
            return
        print(f'caching text embeddings: {self.ar_frames}')
        te_dataset = _cache_text_embeddings(
            self.metadata_dataset, map_fn, i, self.cache_dir, regenerate_cache,
            caching_batch_size, text_encoder_key,
            keep_on_fingerprint_change=keep_text_embedding_cache,
            identity=identity,
            require_identity=require_identity,
            profile_sources=((lambda: self.directory_dataset.legacy_tensor_sources(
                self.ar_frames, f'text_embeddings_{i}'))
                if getattr(self.directory_dataset, 'reuse_metadata_cache', False) else None),
            profile_lookup_dir=(self.directory_dataset.snapshot_dir / 'tensor_lookups'
                                if getattr(self.directory_dataset, 'reuse_metadata_cache', False) else None),
        )
        for size_bucket_dataset in self.size_buckets:
            size_bucket_dataset.add_text_embedding_dataset(te_dataset)


# {geometry directory: the snapshot --regenerate_cache made for it in this caching run}. None
# outside a run (cache_metadata called directly): every --regenerate_cache call makes its own.
_REFRESHED_SNAPSHOTS = None

# Default for DirectoryDataset.snapshot_geometry: use the dataset's own value. None is a real
# framerate (an image model has none), so it cannot be the default.
_OWN = object()


class DirectoryDataset:
    def __init__(self, directory_config, dataset_config, model_name, framerate=None,
                 round_to_multiple=32, skip_dataset_validation=False,
                 caches_text_embeddings=True, vae_identity='', on_the_fly_media=None,
                 layout_caption_format=None, encode_text_in_loader=False):
        self._set_defaults(directory_config, dataset_config)
        self.directory_config = directory_config
        # Each sample then carries what utils/text_encoding.py needs to encode its caption.
        self.directory_config['encode_text_in_loader'] = encode_text_in_loader
        # (media_loader, encode_fn) for a model that reads its media at __getitem__ time instead
        # of caching latents (see OnTheFlyMediaDataset). None for every other model.
        self.on_the_fly_media = on_the_fly_media
        self.dataset_config = dataset_config
        if not skip_dataset_validation:
            self.validate()
        self.model_name = model_name
        self.framerate = framerate
        self.round_to_multiple = round_to_multiple
        self.enable_ar_bucket = directory_config.get('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        # Configure directly from user-specified size buckets.
        self.size_buckets = directory_config.get('size_buckets', dataset_config.get('size_buckets', None))
        self.use_size_buckets = (self.size_buckets is not None)
        if self.use_size_buckets:
            # sort size bucket from longest frame length to shortest
            self.size_buckets.sort(key=lambda t: t[-1], reverse=True)
            self.size_buckets = np.array(self.size_buckets)
            self.size_bucket_datasets = []
        else:
            self.resolutions = self._process_user_provided_resolutions(
                directory_config.get('resolutions', dataset_config['resolutions'])
            )
            self.resolutions = dedup_and_sort(self.resolutions)
            self.ar_bucket_datasets = []
        self.shuffle = directory_config.get('cache_shuffle_num', dataset_config.get('cache_shuffle_num', 0))
        if self.shuffle == 0 and directory_config.get('shuffle_tags', dataset_config.get('shuffle_tags', False)):
            # Legacy spelling. This fixup used to live inside _metadata_map_fn's inner function,
            # which runs later and in a worker process, so every gate computed here missed it --
            # a config saying shuffle_tags got frozen shuffling while the identical config
            # saying cache_shuffle_num = 1 got per-access shuffling.
            self.shuffle = 1
        self.shuffle_metadata = directory_config['shuffle_metadata']
        self.directory_config['cache_shuffle_num'] = self.shuffle # Make accessible if it wasn't yet, for picking one out
        self.shuffle_delimiter = directory_config.get('cache_shuffle_delimiter', dataset_config.get('cache_shuffle_delimiter', ", "))
        self.path = Path(self.directory_config['path'])
        self.mask_path = Path(self.directory_config['mask_path']) if 'mask_path' in self.directory_config else None
        self.control_path = Path(self.directory_config['control_path']) if 'control_path' in self.directory_config else None
        # For testing. Default if a mask is missing.
        self.default_mask_file = Path(self.directory_config['default_mask_file']) if 'default_mask_file' in self.directory_config else None
        self.cache_dir = self.path / 'cache' / self.model_name
        self.require_non_latin_caption = directory_config['require_non_latin_caption']
        if self.require_non_latin_caption is not None:
            # Selection changes image rows, so isolate all row-indexed caches, not only captions.
            self.cache_dir /= f'require_non_latin_caption_{str(self.require_non_latin_caption).lower()}'
        # Recorded in the cache manifests, not in the fingerprints, so adding them invalidates
        # nothing. Empty means the model declares no identity and every cache stays valid.
        self.vae_identity = vae_identity
        self.text_embedding_identity = ''
        self.keep_text_embedding_cache = directory_config.get(
            'keep_text_embedding_cache', dataset_config.get('keep_text_embedding_cache', False))
        # Every metadata artefact derived from the caption text carries this suffix. It is
        # empty at the default settings, so caches written before they existed stay valid.
        self.caption_cache_suffix = caption_cache_suffix({
            'augment_at_runtime': (
                not caches_text_embeddings
                and (directory_config.get('cache_shuffle_num', dataset_config.get('cache_shuffle_num', 0)) > 0
                     or directory_config.get('shuffle_tags', dataset_config.get('shuffle_tags', False))
                     or directory_config.get('tag_dropout_rate', dataset_config.get('tag_dropout_rate', 0.0)) > 0)
            ),
            'prefix_tag_caption': directory_config.get('prefix_tag_caption', dataset_config.get('prefix_tag_caption', '')),
            'tag_dropout_rate': directory_config.get('tag_dropout_rate', dataset_config.get('tag_dropout_rate', 0.0)),
            'multiline_captions': directory_config.get('multiline_captions', dataset_config.get('multiline_captions', False)),
            'enable_remove_non_latin': directory_config['enable_remove_non_latin'],
            'layout_captions': (f'v{LAYOUT_METADATA_VERSION}-{layout_caption_format}'
                                if layout_caption_format else ''),
        })
        # JSON layout captions (utils/layout_captions.py), for a model that declares a format.
        self.layout_caption_format = layout_caption_format
        self.layout_rewriter = (LayoutCaptionRewriter(layout_caption_format, directory_config['caption_prefix'])
                                if layout_caption_format else None)
        self.grouping_keys_json_file = self.cache_dir / f'metadata/grouping_keys{self.caption_cache_suffix}.json'
        self.skip_empty_caption = directory_config.get('skip_empty_caption', dataset_config.get('skip_empty_caption', True))
        self.multiline_captions = directory_config.get('multiline_captions', dataset_config.get('multiline_captions', False))
        self.prefix_tag_caption = directory_config.get('prefix_tag_caption', dataset_config.get('prefix_tag_caption', ''))
        self.tag_dropout_rate = directory_config.get('tag_dropout_rate', dataset_config.get('tag_dropout_rate', 0.0))
        self.enable_remove_non_latin = directory_config['enable_remove_non_latin']
        self.caption_sampling = directory_config.get('caption_sampling', dataset_config.get('caption_sampling', 'all'))
        if self.caption_sampling not in CAPTION_SAMPLING_MODES:
            raise ValueError(
                f'caption_sampling must be one of {CAPTION_SAMPLING_MODES}, got '
                f'{self.caption_sampling!r}'
            )
        # Not gated on caption_sampling: random_per_epoch draws from the cached variants, so
        # with cache_shuffle_num = 0 there is still exactly one variant to draw from and the
        # dropped tags are still gone for good.
        if self.tag_dropout_rate > 0 and self.shuffle == 0 and caches_text_embeddings:
            logger.warning(
                f'{self.path}: tag_dropout_rate={self.tag_dropout_rate} with cache_shuffle_num=0 '
                f'produces ONE frozen variant per caption, so the dropped tags are gone for good '
                f'-- that is permanent tag deletion, not augmentation. Set cache_shuffle_num > 1 '
                f'to cache several draws, or caption_sampling = "random_per_epoch" to pick a '
                f'different one each epoch.'
            )
        self.directory_config['caption_sampling'] = self.caption_sampling
        # With no cached embeddings the caption string is what gets tokenized every step, so
        # augmentation moves to __getitem__ and the metadata keeps raw captions -- markers
        # intact, so tag captions can still be told from prose there.
        self.augment_at_runtime = (
            not caches_text_embeddings and (self.shuffle > 0 or self.tag_dropout_rate > 0)
        )
        self.directory_config['augment_at_runtime'] = self.augment_at_runtime
        self.directory_config['caption_cache_suffix'] = self.caption_cache_suffix
        # SizeBucketDataset needs these at __getitem__ time for the online_captions path.
        self.directory_config['prefix_tag_caption'] = self.prefix_tag_caption
        self.directory_config['tag_dropout_rate'] = self.tag_dropout_rate
        self.directory_config['cache_shuffle_delimiter'] = self.shuffle_delimiter

        if not self.path.exists() or not self.path.is_dir():
            raise RuntimeError(f'Invalid path: {self.path}')
        if self.mask_path is not None and (not self.mask_path.exists() or not self.mask_path.is_dir()):
            raise RuntimeError(f'Invalid mask_path: {self.mask_path}')
        if self.control_path is not None and (not self.control_path.exists() or not self.control_path.is_dir()):
            raise RuntimeError(f'Invalid control_path: {self.control_path}')
        if self.default_mask_file is not None and (not self.default_mask_file.exists() or not self.default_mask_file.is_file()):
            raise RuntimeError(f'Invalid default_mask_file: {self.default_mask_file}')

        if self.use_size_buckets:
            self.ars = np.array([w / h for w, h, _ in self.size_buckets])
        elif not self.enable_ar_bucket:
            self.ars = np.array([1.0])
        elif ars := self.directory_config.get('ar_buckets', self.dataset_config.get('ar_buckets', None)):
            self.ars = self._process_user_provided_ars(ars)
        else:
            min_ar = self.directory_config.get('min_ar', self.dataset_config['min_ar'])
            max_ar = self.directory_config.get('max_ar', self.dataset_config['max_ar'])
            num_ar_buckets = self.directory_config.get('num_ar_buckets', self.dataset_config['num_ar_buckets'])
            self.ars = np.geomspace(min_ar, max_ar, num=num_ar_buckets)
        self.ars = dedup_and_sort(self.ars)
        self.log_ars = np.log(self.ars)
        if self.use_size_buckets:
            # One per size bucket, in the same order: self.ars is deduplicated and sorted. Rounded
            # as self.ars is, so buckets the old matching got right keep every answer it gave.
            self.size_bucket_log_ars = np.log([round(w / h, ROUND_DECIMAL_DIGITS) for w, h, _ in self.size_buckets])
            # Before this was fixed, size buckets were matched against self.ars, which gave the
            # right bucket only when the buckets already came in increasing, distinct aspect ratios.
            self.size_buckets_matched_before = [round(w / h, ROUND_DECIMAL_DIGITS) for w, h, _ in self.size_buckets] == self.ars.tolist()
        frame_buckets = self.directory_config.get('frame_buckets', self.dataset_config.get('frame_buckets', [1]))
        if 1 not in frame_buckets:
            # always have an image bucket for convenience
            frame_buckets.append(1)
        frame_buckets.sort()
        self.frame_buckets = np.array(frame_buckets)

        online_captions = directory_config.get('online_captions', dataset_config.get('online_captions', False))
        self.reuse_metadata_cache = directory_config['reuse_metadata_cache']
        if online_captions and not self.reuse_metadata_cache:
            captions_json = self.path / CAPTIONS_JSON_FILE
            assert captions_json.exists()
            # encoding='utf-8' is not optional: open() defaults to the locale encoding, which
            # on Windows is a codepage that decodes UTF-8 captions into mojibake WITHOUT
            # raising. The corrupted text then reaches the cache and the text encoder.
            with open(captions_json, encoding='utf-8-sig') as f:
                self.captions_dict = json.load(f)
        else:
            self.captions_dict = None

        self.legacy_cache_dir = self.path / 'cache' / self.model_name
        if self.reuse_metadata_cache:
            self.geometry_dir = self.legacy_cache_dir / 'profiles' / Hasher.hash(self.snapshot_geometry())
            self.legacy_geometry_file = self.legacy_cache_dir / 'profiles' / 'legacy_geometry.json'
            self.import_legacy_geometry = True
            if self.legacy_geometry_file.exists():
                self.import_legacy_geometry = (
                    json.loads(self.legacy_geometry_file.read_text())['geometry'] == self.geometry_dir.name)
            self.snapshot_dir = self.geometry_dir
            active = self.geometry_dir / 'active_snapshot.json'
            if active.exists():
                self.snapshot_dir = self.geometry_dir / json.loads(active.read_text())['directory']
                if not self.snapshot_dir.resolve().is_relative_to(self.geometry_dir.resolve()):
                    raise ValueError('Source metadata generation must remain inside its cache root')
            self.snapshot_path = self.snapshot_dir / 'source_metadata'
            settings = {**self._legacy_caption_settings(),
                        'caption_settings': self.caption_cache_suffix,
                        'require_non_latin_caption': self.require_non_latin_caption}
            self.profile_key = Hasher.hash(settings)
            self.cache_dir = self.snapshot_dir / self.profile_key
            self.grouping_keys_json_file = self.cache_dir / 'metadata' / 'grouping_keys.json'
            self.cache_dir.joinpath('metadata').mkdir(parents=True, exist_ok=True)
            self._refresh_source = False

    def snapshot_geometry(self, framerate=_OWN, round_to_multiple=_OWN):
        """The settings that decide which source metadata snapshot this directory uses. Its hash
        names the profiles/<geometry> directory. The arguments override this dataset's own
        values, for tools/inherit_metadata_cache.py to find another model's snapshot."""
        geometry = {'size_buckets': self.size_buckets.tolist() if self.use_size_buckets else None,
                'ars': self.ars.tolist(), 'frames': self.frame_buckets.tolist(),
                'resolutions': self.resolutions.tolist() if not self.use_size_buckets else None,
                'round_to_multiple': self.round_to_multiple if round_to_multiple is _OWN else round_to_multiple,
                'framerate': self.framerate if framerate is _OWN else framerate,
                'mask_path': str(self.mask_path), 'control_path': str(self.control_path),
                'default_mask_file': str(self.default_mask_file), 'shuffle': self.shuffle_metadata}
        if self.use_size_buckets and not self.size_buckets_matched_before:
            # A snapshot of these size buckets from before the matching fix has media in the
            # wrong buckets, or none at all: make a new one.
            geometry['size_bucket_matching'] = 2
        return geometry

    def legacy_tensor_sources(self, bucket, cache_name):
        """Describe old tensor inputs without modifying their metadata or shard files."""
        bucket = tuple(bucket)
        grouping_key = (bucket[0], bucket[-1]) if len(bucket) == 4 else bucket
        prefix = f'grouped_metadata_{bucket_suffix(grouping_key)}'
        tensor_dir = (f'ar_frames_{bucket_suffix(bucket)}' if len(bucket) == 2
                      else f'cache_{bucket_suffix(bucket)}')
        sources = []
        profile_prefix = 'latent_profile_' if cache_name == 'latents' else cache_name + '_'
        for inputs in self.snapshot_dir.glob(f'*/{tensor_dir}/{profile_prefix}*/generated_*/inputs'):
            sources.append((datasets.load_from_disk(str(inputs)), inputs.parent))
        # A refreshed source may have changed bytes at unchanged filenames. Old inputs alone
        # cannot prove the old tensors still apply, so do not donate across refresh generations.
        if self.snapshot_dir != self.geometry_dir or not self.import_legacy_geometry:
            return sources
        for root in [self.legacy_cache_dir, *self.legacy_cache_dir.glob('require_non_latin_caption_*')]:
            path = root / tensor_dir / cache_name
            if not path.joinpath('metadata.db').exists():
                continue
            for candidate in root.joinpath('metadata').glob(prefix + '*'):
                if not candidate.is_dir():
                    continue
                metadata = datasets.load_from_disk(str(candidate))
                if len(bucket) == 4:
                    metadata = metadata.map(lambda row: {'size_bucket': list(bucket[1:])},
                                            keep_in_memory=True)
                if len(bucket) != 2:
                    suffix = candidate.name[len(prefix):]
                    orders = list((root / tensor_dir).glob('iteration_order' + suffix + '*'))
                    if not orders:
                        continue
                    try:
                        metadata = align_legacy_metadata(
                            metadata, orders[0], self.snapshot_dir / 'legacy_order' / tensor_dir)
                    except ValueError:
                        continue
                sources.append((metadata, path))
        return sources

    def _load_source_snapshot(self):
        if self.snapshot_path.exists():
            print(f'[CACHE] Using source metadata snapshot: {self.snapshot_path}')
            return datasets.load_from_disk(str(self.snapshot_path))
        print(f'[CACHE] No source metadata snapshot at {self.snapshot_path}: scanning {self.path} once '
              '(later runs reuse it). tools/inherit_metadata_cache.py can give this model another '
              "model's snapshot instead; run it with this same dataset config.")

        baseline_config = dict(self.directory_config)
        baseline_config.update(reuse_metadata_cache=False, require_non_latin_caption=None,
                               caption_prefix='', prefix_tag_caption='', cache_shuffle_num=0,
                               shuffle_tags=False, tag_dropout_rate=0.0, multiline_captions=False,
                               enable_remove_non_latin=False, skip_empty_caption=False,
                               online_captions=False)
        baseline = DirectoryDataset(baseline_config, self.dataset_config, self.model_name,
                                    self.framerate, self.round_to_multiple, True,
                                    caches_text_embeddings=True)
        baseline.cache_dir = self.snapshot_dir / 'bootstrap'
        baseline.cache_dir.joinpath('metadata').mkdir(parents=True, exist_ok=True)
        intermediate = self.legacy_cache_dir / 'metadata' / 'metadata_intermediate'
        legacy = []
        if (not self._refresh_source and self.import_legacy_geometry
                and self.mask_path is None and self.control_path is None and self.default_mask_file is None):
            for candidate in self.legacy_cache_dir.joinpath('metadata').glob('grouped_metadata_*'):
                if candidate.is_dir():
                    legacy.append(datasets.load_from_disk(str(candidate)))
        if not intermediate.exists() or not legacy:
            legacy = [baseline._get_ungrouped_metadata()]
            intermediate = baseline.cache_dir / 'metadata' / 'metadata_intermediate'

        # In RAM, not a SQLite table queried per row: that cost a lock and page-read round
        # trip per image on network storage.
        media = RowsByImage(legacy)
        raw = datasets.load_from_disk(str(intermediate))
        fallback = baseline._metadata_map_fn()

        def generate_snapshot():
            for batch in raw.iter(batch_size=1000):
                rows = [dict(zip(batch, values)) for values in zip(*batch.values())]
                # Caption files, and media the old metadata does not cover, are read in threads.
                olds = media.get([row['image_spec'] for row in rows])
                results = _read_in_threads(snapshot_row, list(zip(rows, olds)))
                yield from (result for result in results if result is not None)

        def snapshot_row(row_and_old):
            row, old = row_and_old
            captions = row.get('caption')
            missing = not captions and not row.get('caption_file')
            if captions is None and row.get('caption_file'):
                captions = read_caption_file(Path(row['caption_file']), False, True)
            compatible = old is not None and (
                self.size_buckets_matched_before and old['size_bucket'] in self.size_buckets.tolist()
                if self.use_size_buckets
                else old['ar_bucket'] is not None and old['ar_bucket'][0] in self.ars
                and old['ar_bucket'][1] in self.frame_buckets)
            if compatible and all(old.get(k) == row.get(k) for k in ('mask_file', 'control_file')):
                result = old
                result['caption'] = captions or ['']
            else:
                batch = {key: [value] for key, value in row.items()}
                batch['caption'] = [captions or ['']]
                mapped = fallback(batch)
                if not mapped['image_spec']:
                    return None
                result = {key: value[0] for key, value in mapped.items()}
            result['_caption_missing'] = missing
            result['_caption_sidecar'] = bool(row.get('caption_file'))
            return result

        iterator = iter(generate_snapshot())
        first = next(iterator, None)
        if first is None:
            snapshot = datasets.Dataset.from_dict({
                **{key: [] for key in legacy[0].column_names},
                '_caption_missing': [], '_caption_sidecar': []})
        else:
            def nonempty_snapshot():
                yield first
                yield from iterator

            snapshot = datasets.Dataset.from_generator(
                nonempty_snapshot, cache_dir=str(self.snapshot_dir / 'arrow'),
                fingerprint=Hasher.hash([str(self.snapshot_dir), raw._fingerprint]))
        snapshot.save_to_disk(str(self.snapshot_path))
        if self.legacy_cache_dir.joinpath('metadata').is_dir():
            try:
                with self.legacy_geometry_file.open('x', encoding='utf-8') as handle:
                    json.dump({'geometry': self.geometry_dir.name}, handle)
            except FileExistsError:
                pass
        active = self.geometry_dir / 'active_snapshot.json'
        temporary = active.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'directory': os.path.relpath(self.snapshot_dir, self.geometry_dir)}))
        os.replace(temporary, active)
        return datasets.load_from_disk(str(self.snapshot_path))

    def _profile_metadata(self):
        snapshot = self._load_source_snapshot()

        def process(batch):
            result = {key: [] for key in snapshot.column_names if not key.startswith('_caption_')}
            for index in range(len(batch['image_spec'])):
                captions = batch['caption'][index]
                if batch['_caption_missing'][index]:
                    captions = None
                elif batch['_caption_sidecar'][index] and self.multiline_captions:
                    captions = split_caption_lines(captions[0], True, self.enable_remove_non_latin,
                                                   self.layout_rewriter is not None)
                if not captions:
                    if self.skip_empty_caption:
                        continue
                    captions = ['']
                if not caption_matches_non_latin_requirement(
                        captions, self.require_non_latin_caption, self.prefix_tag_caption):
                    continue
                if self.augment_at_runtime:
                    captions = [c for c in captions for _ in range(max(self.shuffle, 1))]
                else:
                    captions = shuffle_captions(
                        captions, self.shuffle, self.shuffle_delimiter,
                        self.directory_config['caption_prefix'], self.prefix_tag_caption,
                        self.tag_dropout_rate, seed=seed_from_hash((self.path, batch['image_spec'][index])),
                        enable_remove_non_latin=self.enable_remove_non_latin,
                        layout_captions=self.layout_rewriter is not None)
                for key in result:
                    result[key].append(captions if key == 'caption' else batch[key][index])
            return result

        processed = snapshot.map(process, batched=True, remove_columns=snapshot.column_names,
                                 cache_file_name=str(self.cache_dir / 'metadata' / 'processed.arrow'),
                                 load_from_cache_file=True, desc='Applying caption profile')
        logger.info(f'{self.path}: caption profile retained {len(processed)} of {len(snapshot)} '
                    'source media rows. Sources are frozen until --regenerate_cache.')
        return processed

    def validate(self):
        resolutions = self.directory_config.get('resolutions', self.dataset_config.get('resolutions', []))
        if len(resolutions) > 3:
            if is_main_process():
                print('WARNING: You have set a lot of resolutions in the dataset config. Please read the comments in the example dataset.toml file,'
                      ' and make sure you understand what this setting does. If you still want to proceed with the current configuration,'
                      ' run the script with the --i_know_what_i_am_doing flag.')
            quit()

    def _legacy_caption_settings(self):
        return {
            'caption_prefix': self.directory_config.get('caption_prefix', ''),
            # The resolved count, so the legacy shuffle_tags spelling compares equal to the
            # cache_shuffle_num it means.
            'cache_shuffle_num': self.shuffle,
            'cache_shuffle_delimiter': self.directory_config.get('cache_shuffle_delimiter', ', '),
            'skip_empty_caption': self.skip_empty_caption,
        }

    @property
    def _caption_settings_file(self):
        return self.cache_dir / f'metadata/caption_settings{self.caption_cache_suffix}.json'

    def _record_caption_settings(self):
        path = self._caption_settings_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._legacy_caption_settings(), f, sort_keys=True)

    def _warn_if_caption_settings_changed(self):
        """Report a cache built under different caption settings, rather than serving it.

        Absent file means a cache written before this check existed. Nothing is known about it,
        so nothing is claimed -- staying quiet is what keeps the upgrade seamless.
        """
        path = self._caption_settings_file
        if not path.exists():
            return
        with open(path, encoding='utf-8') as f:
            previous = json.load(f)
        current = self._legacy_caption_settings()
        changed = {k: (previous.get(k, LEGACY_CAPTION_SETTINGS[k]), v)
                   for k, v in current.items() if previous.get(k, LEGACY_CAPTION_SETTINGS[k]) != v}
        if not changed:
            return
        lines = '\n'.join(f'    {k}: {was!r} -> {now!r}' for k, (was, now) in sorted(changed.items()))
        logger.warning(
            f'{self.path}: the cached captions were built with different settings, and '
            f'--trust_cache is reusing them as they are:\n{lines}\n'
            '  These settings are baked into the cached caption text, so this run will train on '
            'the OLD captions. Pass --regenerate_cache to rebuild them, or drop --trust_cache.'
        )

    def cache_metadata(self, regenerate_cache=False, trust_cache=False):
        if self.reuse_metadata_cache:
            if regenerate_cache:
                import uuid
                # One fresh snapshot per geometry: directories of the same source and geometry
                # (the same path under other caption settings) share it, and publishing a second
                # one would leave the first directory's profile behind in an inactive snapshot.
                refresh = 'refresh_' + uuid.uuid4().hex
                if _REFRESHED_SNAPSHOTS is not None:
                    refresh = _REFRESHED_SNAPSHOTS.setdefault(str(self.geometry_dir.resolve()), refresh)
                self.snapshot_dir = self.geometry_dir / refresh
                self.snapshot_path = self.snapshot_dir / 'source_metadata'
                self.cache_dir = self.snapshot_dir / self.profile_key
                self.cache_dir.joinpath('metadata').mkdir(parents=True, exist_ok=True)
                self.grouping_keys_json_file = self.cache_dir / 'metadata' / 'grouping_keys.json'
                self._refresh_source = True
            else:
                # The cache worker may have published a refreshed generation after this
                # object was constructed on a training rank.
                active = self.geometry_dir / 'active_snapshot.json'
                if active.exists():
                    directory = self.geometry_dir / json.loads(active.read_text())['directory']
                    if not directory.resolve().is_relative_to(self.geometry_dir.resolve()):
                        raise ValueError('Source metadata generation must remain inside its cache root')
                    self.snapshot_dir = directory
                    self.snapshot_path = directory / 'source_metadata'
                    self.cache_dir = directory / self.profile_key
                    self.cache_dir.joinpath('metadata').mkdir(parents=True, exist_ok=True)
                    self.grouping_keys_json_file = self.cache_dir / 'metadata' / 'grouping_keys.json'
            trust_cache = True
        elif trust_cache and (stale := self._legacy_grouping_is_stale()):
            # --trust_cache would serve the old grouping as it is: the media scan is redone, as
            # it is on every run without --trust_cache.
            print(f'{self.path}: {stale} Rebuilding it.')
            trust_cache = False
            self._remove_legacy_bucket_caches()
        def check_grouped_metadata():
            all_grouped_metadata_exists = False
            unique_grouping_keys = None
            if self.grouping_keys_json_file.exists():
                with open(self.grouping_keys_json_file) as f:
                    unique_grouping_keys = json.load(f)
                if self.use_size_buckets and not all(len(key) == 3 for key in unique_grouping_keys):
                    # Using size buckets but have AR keys.
                    return False, unique_grouping_keys
                elif not self.use_size_buckets and not all(len(key) == 2 for key in unique_grouping_keys):
                    # Using AR buckets but have size bucket keys
                    return False, unique_grouping_keys
                all_grouped_metadata_exists = all(
                    (self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(key)}{self.caption_cache_suffix}').exists()
                    for key in unique_grouping_keys
                )
            return all_grouped_metadata_exists, unique_grouping_keys

        # Check if all the grouped metadata datasets exist. If so, we can directly load them.
        all_grouped_metadata_exists, unique_grouping_keys = check_grouped_metadata()
        if regenerate_cache or not all_grouped_metadata_exists or not trust_cache:
            # Otherwise, need to compute the ungrouped metadata and then group.
            print('Grouped metadata is not cached. Computing ungrouped metadata and then grouping.')
            unique_grouping_keys = self._group_metadata_and_save_to_disk(regenerate_cache=regenerate_cache, trust_cache=trust_cache)
            self._record_caption_settings()
        else:
            print('Found grouped metadata cache. Directly loading it.')
            self._warn_if_caption_settings_changed()
            if not self.reuse_metadata_cache and not self._bucketing_file().exists():
                # Grouped before the buckets were recorded, by buckets that fit these: recorded
                # now, so a later change of the buckets is seen. A read-only cache still loads.
                try:
                    self._record_bucketing()
                except OSError as e:
                    logger.warning(f'{self.path}: could not record the buckets of the cached metadata: {e}')

        if self.require_non_latin_caption is not None and not unique_grouping_keys:
            logger.warning(
                f'{self.path}: no media remain after require_non_latin_caption='
                f'{self.require_non_latin_caption} and metadata validation.'
            )

        for grouping_key in unique_grouping_keys:
            grouped_cache_dir = self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(grouping_key)}{self.caption_cache_suffix}'
            print(f'Loading grouped metadata with grouping key {grouping_key}')
            metadata = datasets.load_from_disk(str(grouped_cache_dir))
            if self.use_size_buckets:
                assert len(grouping_key) == 3
                self.size_bucket_datasets.append(
                    SizeBucketDataset(
                        metadata,
                        self.directory_config,
                        # A key read back from grouping_keys.json is a list, and Dataset.post_init
                        # groups buckets in a dict keyed by it.
                        tuple(grouping_key),
                        self.cache_dir,
                        self,
                    )
                )
            else:
                self.ar_bucket_datasets.append(
                    ARBucketDataset(
                        grouping_key,
                        self.resolutions,
                        metadata,
                        self.directory_config,
                        self.cache_dir,
                        self.round_to_multiple,
                        self,
                    )
                )
        if self.layout_rewriter is not None:
            self._report_layout_captions()

    def _report_layout_captions(self):
        buckets = self.bucket_datasets()
        # Counted per image: cache_shuffle_num repeats each caption, which is not the user's count.
        total = layout = 0
        for bucket in buckets:
            for captions in bucket.metadata_dataset['caption']:
                total += 1
                layout += any(is_layout_caption(c) for c in captions)
        if layout and is_main_process():
            field = {'moss_image': 'summary.caption (gen_brief in the scene/element schema)',
                     'moss_image_centered': 'summary.caption (gen_brief in the scene/element schema)',
                     'ideogram4': 'high_level_description'}[self.layout_caption_format]
            print(f'{self.path}: {layout} of {total} images have {self.layout_caption_format} layout (JSON) captions: '
                  f'their boxes are rewritten for each bucket, caption_prefix goes into {field}, and tag '
                  f'shuffling, tag dropout, multiline splitting and non-Latin line removal skip them.')

    def _group_metadata_and_save_to_disk(self, regenerate_cache=False, trust_cache=False):
        if not self.reuse_metadata_cache:
            # Until the grouping is complete, the ungrouped metadata and the grouped datasets may
            # be from this scan or the last one: a run killed meanwhile must not be trusted.
            self._record_bucketing({'incomplete': True})
        metadata_dataset = self._get_ungrouped_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)
        grouped_metadata = defaultdict(lambda: defaultdict(list))
        unique_grouping_keys = set()
        for example in tqdm(metadata_dataset, desc='Grouping examples'):
            if self.use_size_buckets:
                grouping_key = tuple(example['size_bucket'])
            else:
                grouping_key = example['ar_bucket']
                grouping_key = (grouping_key[0], int(grouping_key[1]))
            unique_grouping_keys.add(grouping_key)
            d = grouped_metadata[grouping_key]
            for k, v in example.items():
                d[k].append(v)
        unique_grouping_keys = list(unique_grouping_keys)

        if self.use_size_buckets:
            fingerprints = {}
            for size_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                grouped_cache_dir = self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(size_bucket)}{self.caption_cache_suffix}'
                metadata.save_to_disk(str(grouped_cache_dir))
                fingerprints[bucket_suffix(size_bucket)] = metadata._fingerprint
        else:
            for ar_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                grouped_cache_dir = self.cache_dir / f'metadata/grouped_metadata_{bucket_suffix(ar_bucket)}{self.caption_cache_suffix}'
                metadata.save_to_disk(str(grouped_cache_dir))

        with open(self.grouping_keys_json_file, 'w') as f:
            json.dump(unique_grouping_keys, f)
        if self.use_size_buckets:
            marker = self.cache_dir / f'metadata/size_bucket_matching{self.caption_cache_suffix}.json'
            with open(marker, 'w') as f:
                json.dump({'size_bucket_matching': 2, 'grouped': fingerprints}, f)
        if not self.reuse_metadata_cache:
            self._record_bucketing()

        return unique_grouping_keys

    def _remove_legacy_bucket_caches(self):
        """This caption profile's caches made from a grouping: --trust_cache would load them over a
        new one. The buckets' iteration orders index their old rows, and the AR buckets' copies
        per resolution hold their old images. Removed before a regrouping, so an interrupted one
        removes them again."""
        for name in legacy_iteration_order_names(self.caption_cache_suffix):
            for iteration_order in self.cache_dir.glob(f'cache_*/{name}'):
                shutil.rmtree(iteration_order)
        tag = legacy_copy_tag(self.caption_cache_suffix)
        for copy in self.cache_dir.glob(f'ar_frames_*/metadata/metadata_*{tag}.arrow'):
            copy.unlink()

    def _bucketing(self):
        """What decides each image's bucket in the metadata scan: a video's frame count is
        converted to the model's framerate."""
        if self.use_size_buckets:
            return {'size_buckets': self.size_buckets.tolist(), 'framerate': self.framerate}
        return {'ars': self.ars.tolist(), 'frame_buckets': self.frame_buckets.tolist(), 'framerate': self.framerate}

    def _bucketing_file(self):
        return self.cache_dir / f'metadata/bucketing{self.caption_cache_suffix}.json'

    def _record_bucketing(self, record=None):
        path = self._bucketing_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.{socket.gethostname()}-{os.getpid()}.tmp')
        with open(temporary, 'w') as f:
            json.dump(self._bucketing() if record is None else record, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)

    def _legacy_grouping_is_stale(self):
        """Why the legacy grouped metadata does not fit this dataset's buckets, or None. Both the
        ungrouped metadata and the grouping hold each image's bucket, and --trust_cache loads them
        without looking."""
        metadata_dir = self.cache_dir / 'metadata'
        if self.use_size_buckets and not self.size_buckets_matched_before \
                and not size_bucket_matching_is_current(metadata_dir, self.caption_cache_suffix):
            # Grouped before the size bucket fix: some images are in the wrong bucket.
            return 'the cached metadata predates the size_buckets matching fix.'
        try:
            recorded = json.loads(self._bucketing_file().read_text())
        except FileNotFoundError:
            recorded = None
        except ValueError:
            return 'the record of the buckets the cached metadata was grouped by is unreadable.'
        if recorded is not None:
            if recorded == {'incomplete': True}:
                return 'the last metadata scan was interrupted.'
            if recorded != self._bucketing():
                return 'the cached metadata was grouped by other buckets than these.'
            return None
        # Grouped before the buckets were recorded: a bucket these settings never make shows
        # they changed. New buckets that happen to include all the old ones go unseen.
        try:
            keys = json.loads(self.grouping_keys_json_file.read_text())
        except (FileNotFoundError, ValueError):
            return None
        for key in keys:
            if self.use_size_buckets:
                fits = len(key) == 3 and list(key) in self.size_buckets.tolist()
            else:
                fits = (len(key) == 2 and key[1] in self.frame_buckets.tolist()
                        and bool(np.isclose(self.ars, key[0], rtol=0, atol=1e-6).any()))
            if not fits:
                return f'the cached metadata was grouped by other buckets than these ({key} is not one of them).'
        return None

    def _get_ungrouped_metadata(self, regenerate_cache=False, trust_cache=False):
        if self.reuse_metadata_cache:
            return self._profile_metadata()
        # This method caches some intermediate datasets so we don't have to enumerate all the files each time.
        metadata_cache_file_1 = self.cache_dir / 'metadata/metadata_intermediate'
        metadata_cache_file_2 = self.cache_dir / f'metadata/metadata{self.caption_cache_suffix}.arrow'

        if regenerate_cache or not metadata_cache_file_1.exists() or not trust_cache:
            print('Intermediate metadata is not cached. Enumerating all files.')
            # One listing, and no stat per file: os.scandir tells files apart and the caption
            # files are looked up in it. A stat per entry and an exists() per image, one after
            # the other, took hours for millions of images on network storage.
            files = []
            caption_names = {}
            with os.scandir(self.path) as entries:
                for entry in entries:
                    if entry.name.endswith('.txt'):
                        caption_names[entry.name] = entry.is_symlink()
                    if _is_file(entry) and Path(entry.name).suffix not in NON_MEDIA_SUFFIXES:
                        files.append(self.path / entry.name)
            # deterministic order
            files.sort()

            def caption_exists(caption_file):
                symlink = caption_names.get(caption_file.name)
                if symlink is None:
                    # A filesystem that ignores case (CIFS, exFAT, Windows, macOS) finds X.TXT
                    # for x.txt. This only costs a stat for media without a caption.
                    return os.path.exists(caption_file)
                return not symlink or os.path.exists(caption_file)

            # Mask can have any extension, it just needs to have the same stem as the image.
            mask_file_stems = _files_by_stem(self.mask_path) if self.mask_path is not None else {}
            control_file_stems = _files_by_stem(self.control_path) if self.control_path is not None else {}

            def process_file(file):
                if file.suffix != '.tar':
                    return [(None, str(file))]
                with tarfile.TarFile(file) as tar_f:
                    return [(str(file), name) for name in tar_f.getnames()]

            captions_json = self.path / CAPTIONS_JSON_FILE
            has_captions_json = captions_json.exists()

            image_specs = []
            caption_files = []
            mask_files = []
            control_files = []
            for file in tqdm(files):
                for image_spec in process_file(file):
                    image_file = Path(image_spec[1])
                    caption_file = image_file.with_suffix('.txt')
                    # A tar member's caption is looked for where it always was.
                    if has_captions_json or not (caption_exists(caption_file) if image_spec[0] is None
                                                 else os.path.exists(caption_file)):
                        caption_file = ''
                    image_specs.append(image_spec)
                    caption_files.append(str(caption_file))
                    # mask
                    if image_file.stem in mask_file_stems:
                        mask_files.append(str(mask_file_stems[image_file.stem]))
                    elif self.default_mask_file is not None:
                        mask_files.append(str(self.default_mask_file))
                    else:
                        if self.mask_path is not None:
                            logger.warning(f'No mask file was found for image {image_file}, not using mask.')
                        mask_files.append(None)
                    # control (e.g. Flux Kontext)
                    if self.control_path:
                        if image_file.stem not in control_file_stems:
                            raise RuntimeError(f'No control file exists for image {image_file}')
                        control_files.append(str(control_file_stems[image_file.stem]))
            assert len(image_specs) > 0, f'Directory {self.path} had no images/videos!'

            d = {'image_spec': image_specs, 'caption_file': caption_files, 'mask_file': mask_files}
            if self.control_path:
                d['control_file'] = control_files
            metadata_dataset = datasets.Dataset.from_dict(d)

            if captions_json.exists():
                print('Loading captions JSON')
                with open(captions_json, encoding='utf-8-sig') as f:
                    caption_data = json.load(f)

                def add_captions(example):
                    tar_file, image_file = example['image_spec']
                    if tar_file is None:
                        image_file = os.path.basename(image_file)
                    captions = caption_data.get(image_file, None)
                    if captions is None:
                        logger.warning(f'Image file {image_file} does not have an entry in captions.json')
                    else:
                        assert isinstance(captions, list), 'captions.json must contain lists of captions'
                    return {'caption': captions}

                metadata_dataset = metadata_dataset.map(
                    add_captions,
                    cache_file_name=str(self.cache_dir / 'metadata/metadata_with_captions.arrow'),
                    load_from_cache_file=(not regenerate_cache and trust_cache),
                    desc='Adding captions',
                )
                del caption_data

            # Shuffle the data. Use a deterministic seed, so the dataset is identical on all processes.
            # Seed is based on the hash of the directory path, so that if directories have the same set of images, they are shuffled differently.
            seed = seed_from_hash(self.path)
            if self.shuffle_metadata:
                metadata_dataset = metadata_dataset.shuffle(seed=seed)
            print('Saving intermediate metadata dataset.')
            metadata_dataset.save_to_disk(str(metadata_cache_file_1))
            # Need to delete and load from disk, or else the map() call below is extremely slow to launch worker processes
            # and they use huge amounts of memory. Probably because this dataset is in memory?
            del metadata_dataset

        print('Loading intermediate metadata dataset.')
        metadata_dataset = datasets.load_from_disk(str(metadata_cache_file_1))

        metadata_map_fn = _rows_in_threads(self._metadata_map_fn(), METADATA_READ_THREADS)
        print('Caching ungrouped metadata.')
        input_rows = len(metadata_dataset)
        metadata_dataset = metadata_dataset.map(
            metadata_map_fn,
            cache_file_name=str(metadata_cache_file_2),
            load_from_cache_file=(not regenerate_cache and trust_cache),
            batched=True,
            batch_size=8 * max(METADATA_READ_THREADS, 1),
            # None, not 1: datasets still starts a worker process for num_proc=1.
            num_proc=NUM_PROC if NUM_PROC > 1 else None,
            remove_columns=metadata_dataset.column_names,
        )
        if self.require_non_latin_caption is not None:
            logger.info(
                f'{self.path}: require_non_latin_caption={self.require_non_latin_caption}; '
                f'retained {len(metadata_dataset)} of {input_rows} media rows, '
                f'skipped {input_rows - len(metadata_dataset)} by caption selection or metadata validation.'
            )
        return metadata_dataset

    def _set_defaults(self, directory_config, dataset_config):
        directory_config.setdefault('reuse_metadata_cache', dataset_config.get('reuse_metadata_cache', True))
        if not isinstance(directory_config['reuse_metadata_cache'], bool):
            raise ValueError('reuse_metadata_cache must be a boolean')
        directory_config.setdefault('require_non_latin_caption', dataset_config.get('require_non_latin_caption', None))
        validate_non_latin_caption_requirement(directory_config['require_non_latin_caption'])
        directory_config.setdefault('enable_remove_non_latin', dataset_config.get('enable_remove_non_latin', False))
        if not isinstance(directory_config['enable_remove_non_latin'], bool):
            raise ValueError('enable_remove_non_latin must be a boolean')
        if 'caption_dropout_rate' not in directory_config and 'caption_dropout_rate' in dataset_config:
            directory_config['caption_dropout_rate'] = dataset_config['caption_dropout_rate']
        if 'caption_dropout_rate' in directory_config:
            validate_caption_dropout_rate(directory_config['caption_dropout_rate'])
        directory_config.setdefault('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        directory_config.setdefault('shuffle_tags', dataset_config.get('shuffle_tags', False))
        directory_config.setdefault('caption_prefix', dataset_config.get('caption_prefix', ''))
        directory_config.setdefault('num_repeats', dataset_config.get('num_repeats', 1))
        directory_config.setdefault('shuffle_metadata', dataset_config.get('shuffle_metadata', True))

    def _metadata_map_fn(self):
        def fn(example):
            empty_return = {'image_spec': [], 'mask_file': [], 'caption': [], 'ar_bucket': [], 'size_bucket': [], 'is_video': []}
            if self.control_path:
                empty_return['control_file'] = []
            # batch size always 1
            caption_file = example['caption_file'][0]
            image_spec = example['image_spec'][0]
            image_file = Path(image_spec[1])
            captions = None
            if 'caption' in example:
                # Already put in dataset from captions.json file.
                captions = example['caption'][0]
            if captions is None and caption_file:
                captions = read_caption_file(Path(caption_file), self.multiline_captions,
                                             self.enable_remove_non_latin,
                                             layout_captions=self.layout_rewriter is not None)
            if not captions:
                captions = None
            if captions is None:
                if self.skip_empty_caption:
                    logger.warning(f'Cound not find caption for {image_file}. Skipping image.')
                    return empty_return
                else:
                    logger.warning(f'Cound not find caption for {image_file}. Using empty caption.')
                    captions = ['']
            if not caption_matches_non_latin_requirement(
                    captions, self.require_non_latin_caption, self.prefix_tag_caption):
                logger.debug(f'Skipping {image_file}: require_non_latin_caption={self.require_non_latin_caption}')
                return empty_return
            if self.augment_at_runtime:
                # Keep the captions exactly as they are on disk, marker included, and expand to
                # the same number of variants cache_shuffle_num would have produced so the
                # epoch length is unchanged. __getitem__ augments each access, so the variants
                # differ every pass instead of being frozen here.
                captions = [c for c in captions for _ in range(max(self.shuffle, 1))]
            else:
                captions = shuffle_captions(
                    captions,
                    self.shuffle,
                    self.shuffle_delimiter,
                    self.directory_config['caption_prefix'],
                    self.prefix_tag_caption,
                    self.tag_dropout_rate,
                    # Seeded per image so the frozen variants and text-cache inputs are
                    # identical on every launch. Image latents fingerprint other columns only.
                    seed=seed_from_hash((self.path, image_spec)),
                    enable_remove_non_latin=self.enable_remove_non_latin,
                    layout_captions=self.layout_rewriter is not None,
                )
            if image_spec[0] is None:
                tar_f, lock = None, contextlib.nullcontext()
            else:
                tar_f, lock = _shared_tarfile(image_spec[0])
            # A tar member is read under its tar's lock, from extractfile to close.
            with lock:
                if tar_f is None:
                    filepath_or_file = str(image_file)
                else:
                    # as_posix(), not str(): a tar member name always uses forward slashes and
                    # extractfile matches it literally, so str() on a Path looks up a name with
                    # backslashes on Windows and raises KeyError for any member in a subdirectory.
                    filepath_or_file = tar_f.extractfile(image_file.as_posix())

                if image_file.suffix == '.webp':
                    # Make sure this this object stays alive so it doesn't close the file on us.
                    reader = imageio.get_reader(filepath_or_file)
                    frames = reader.get_length()
                    if frames > 1:
                        raise NotImplementedError('WebP videos are not supported.')
                try:
                    if image_file.suffix in VIDEO_EXTENSIONS:
                        comfy_video = InputImpl.VideoFromFile(filepath_or_file)
                        if image_file.suffix.lower() == '.mkv':
                            # Can't get number of frames (it would show as 0) unless we actually load the video tensor.
                            # One decode at a time per process, as before reads ran in threads:
                            # a whole video is in memory meanwhile.
                            with _process_lock('mkv'):
                                components = comfy_video.get_components()
                            source_frames, height, width, _ = components.images.shape
                        else:
                            width, height = comfy_video.get_dimensions()
                            source_frames = comfy_video.get_frame_count()
                        source_fps = float(comfy_video.get_frame_rate())

                        assert self.framerate is not None, "Need model framerate but don't have it. This shouldn't happen. Is the framerate attribute on the model set?"
                        frames = int(source_frames * self.framerate / source_fps)
                    else:
                        pil_img = Image.open(filepath_or_file)
                        width, height = pil_img.size
                        frames = 1
                except Exception as e:
                    logger.warning(f'Media file {image_file} could not be opened. Skipping. The exception was: {e}')
                    return empty_return
                finally:
                    if hasattr(filepath_or_file, 'close'):
                        filepath_or_file.close()

            is_video = (frames > 1)
            log_ar = np.log(width / height)

            if self.use_size_buckets:
                size_bucket = self._find_closest_size_bucket(log_ar, frames, is_video)
                if size_bucket is None:
                    print(f'video with frames={frames} is being skipped because it is too short')
                    return empty_return
                ar_bucket = None
            else:
                ar_bucket = self._find_closest_ar_bucket(log_ar, frames, is_video)
                if ar_bucket is None:
                    print(f'video with frames={frames} is being skipped because it is too short')
                    return empty_return
                size_bucket = None

            ret = {
                'image_spec': [image_spec],
                'mask_file': [example['mask_file'][0]],
                'caption': [captions],
                'ar_bucket': [ar_bucket],
                'size_bucket': [size_bucket],
                'is_video': [is_video],
            }
            if self.control_path:
                ret['control_file'] = [example['control_file'][0]]
            return ret

        return fn

    def _find_closest_ar_bucket(self, log_ar, frames, is_video):
        # Best AR bucket is the one with the smallest AR difference in log space.
        i = np.argmin(np.abs(log_ar - self.log_ars))
        # find closest frame bucket where the number of frames is greater than or equal to the bucket
        diffs = frames - self.frame_buckets
        positive_diffs = diffs[diffs >= 0]
        if len(positive_diffs) == 0:
            # video not long enough to find any valid frame bucket
            return None
        j = np.argmin(positive_diffs)
        if is_video and self.frame_buckets[j] == 1:
            # don't let video be mapped to the image frame bucket
            return None
        ar_bucket = (self.ars[i], self.frame_buckets[j])
        return ar_bucket

    def _find_closest_size_bucket(self, log_ar, frames, is_video):
        # Best AR bucket is the one with the smallest AR difference in log space.
        ar_diffs = np.abs(log_ar - self.size_bucket_log_ars)
        candidate_size_buckets = self.size_buckets[np.argsort(ar_diffs, kind='stable')]
        # Find closest size bucket where the number of frames is greater than or equal to the bucket.
        # self.size_buckets was already sorted longest -> shortest frame length
        found = False
        for size_bucket in candidate_size_buckets:
            if is_video and size_bucket[-1] == 1:
                # don't let video be mapped to the image frame bucket
                continue
            if frames >= size_bucket[-1]:
                found = True
                break
        if not found:
            # video not long enough to find any valid frame bucket
            return None
        return size_bucket

    def _process_user_provided_ars(self, ars):
        ar_buckets = []
        for ar in ars:
            if isinstance(ar, (tuple, list)):
                assert len(ar) == 2
                ar = ar[0] / ar[1]
            ar_buckets.append(ar)
        return ar_buckets

    def _process_user_provided_resolutions(self, resolutions):
        result = []
        for res in resolutions:
            if isinstance(res, (tuple, list)):
                assert len(res) == 2
                res = math.sqrt(res[0] * res[1])
            result.append(res)
        return result

    def get_size_bucket_datasets(self):
        if self.use_size_buckets:
            return self.size_bucket_datasets
        result = []
        for ar_bucket_dataset in self.ar_bucket_datasets:
            result.extend(ar_bucket_dataset.get_size_bucket_datasets())
        return result

    def bucket_datasets(self):
        """The buckets caching goes through one by one, each with caches of its own: a size
        bucket, or an aspect ratio bucket with all its resolutions."""
        return self.size_bucket_datasets if self.use_size_buckets else self.ar_bucket_datasets

    def cache_latents(self, map_fn, regenerate_cache=False, trust_cache=False, caching_batch_size=1, takes=None):
        """takes(bucket) picks the buckets to cache, None all of them: with several nodes caching,
        each takes the units of work it draws (see tensor_units)."""
        buckets = [ds for ds in self.bucket_datasets() if takes is None or takes(ds)]
        if buckets:
            print(f'caching latents: {self.path}')
        for ds in buckets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, trust_cache=trust_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1,
                              text_encoder_key='', keep_text_embedding_cache=False,
                              identity=None, require_identity=False, takes=None, uncond=True):
        """takes as in cache_latents; uncond=False leaves the empty caption's embedding, which
        the directory shares, to another node."""
        buckets = [ds for ds in self.bucket_datasets() if takes is None or takes(ds)]
        if buckets:
            print(f'caching text embeddings: {self.path}')
        for ds in buckets:
            ds.cache_text_embeddings(
                map_fn, i, regenerate_cache=regenerate_cache,
                caching_batch_size=caching_batch_size, text_encoder_key=text_encoder_key,
                keep_text_embedding_cache=self.keep_text_embedding_cache,
                identity=identity, require_identity=require_identity,
            )
        if not uncond:
            return
        # TODO: do this separately for is_video True and False for models that support it?
        empty_caption_ds = datasets.Dataset.from_dict({'caption': [''], 'is_video': [False], 'image_spec': [(None, None)]})
        if self.reuse_metadata_cache:
            uncond_text_embeddings_ds = tensor_profile(
                empty_caption_ds, map_fn,
                self.snapshot_dir / f'uncond_text_embeddings_{i}_{Hasher.hash([identity, text_encoder_key])}',
                sorted(empty_caption_ds.column_names), identity,
                ([(empty_caption_ds, self.legacy_cache_dir / f'uncond_text_embeddings_{i}')]
                 if self.snapshot_dir == self.geometry_dir and self.import_legacy_geometry else []),
                _map_and_cache, regenerate_cache=regenerate_cache,
                caching_batch_size=caching_batch_size, encoder_key=text_encoder_key,
                lookup_dir=self.snapshot_dir / 'tensor_lookups')
        else:
            uncond_text_embeddings_ds = _map_and_cache(
                empty_caption_ds,
                map_fn,
                cache_dir=self.cache_dir,
                cache_file_prefix=f'uncond_text_embeddings_{i}_',
                # Same reasoning as the conditional embeddings above: this is text encoder
                # output, so it must follow the text encoder rather than the latents. Left at
                # None without a key, preserving the original fingerprint.
                new_fingerprint_args=[text_encoder_key] if text_encoder_key else None,
                # Same identity as the conditional embeddings: this is the same encoder's output,
                # and a model whose key is deliberately empty (anima) would otherwise have nothing
                # distinguishing two runs with different llm_path. Its conditional embeddings would
                # rebuild while the unconditional one -- the embedding every CFG-dropped sample
                # trains against -- came back from the old encoder.
                identity=identity,
                require_identity=require_identity,
                regenerate_cache=regenerate_cache,
            )
        self.uncond_dict = uncond_text_embeddings_ds[0]
        for size_bucket_ds in self.get_size_bucket_datasets():
            size_bucket_ds.uncond_text_embeddings.append(uncond_text_embeddings_ds)


# Outermost dataset object that the caller uses. Contains multiple ConcatenatedBatchedDataset. Responsible
# for returning the correct batch for the process's data parallel rank. Calls model.prepare_inputs so the
# returned tuple of tensors is whatever the model needs.
class Dataset:
    def __init__(self, dataset_config, model, skip_dataset_validation=False,
                 batch_fill_defaults=None, batch_fill_overrides=None):
        super().__init__()
        self.dataset_config = dataset_config
        self.model = model
        self.model_name = self.model.name
        # TODO: remove. Doing this because Wan and Cosmos-Predict2 use the same latents.
        # if self.model_name == 'wan' and len(self.model.get_text_encoders()) == 0:
        #     self.model_name = 'cosmos_predict2'
        self.post_init_called = False
        self.eval_quantile = None
        # Checked here rather than at post_init so a typo fails before the caching run, not
        # after it. batch_fill_defaults lets the eval datasets change a default without
        # restating the table; train passes nothing and gets the shipped defaults.
        self.batch_fill = resolve_batch_fill_config(
            dataset_config, batch_fill_defaults, batch_fill_overrides)
        if not skip_dataset_validation:
            self.model.model_specific_dataset_config_validation(self.dataset_config)

        on_the_fly_media = None
        if getattr(self.model, 'load_media_on_the_fly', False):
            # The model has no latents worth caching (a pixel-space model's 'latents' are the
            # pixels), so the dataset decodes each image when it is drawn, in the DataLoader
            # worker. Both functions are required to be CPU-only for such a model.
            on_the_fly_media = (
                self.model.get_preprocess_media_file_fn(),
                self.model.get_call_vae_fn(self.model.get_vae()),
            )

        # cache_text_embeddings = false on a model that cannot encode its captions itself.
        encode_text_in_loader = encodes_text_in_loader(model)
        self.directory_datasets = []
        for directory_config in dataset_config['directory']:
            directory_dataset = DirectoryDataset(
                directory_config,
                dataset_config,
                self.model_name,
                framerate=model.framerate,
                round_to_multiple=model.pixels_round_to_multiple,
                skip_dataset_validation=skip_dataset_validation,
                # When the model caches no text embeddings, the caption string itself is what
                # gets tokenized every step, so augmentation can happen per sample instead of
                # being frozen into the metadata. The dataset cannot work this out for itself.
                caches_text_embeddings=len(model.get_text_encoders()) > 0 and not encode_text_in_loader,
                encode_text_in_loader=encode_text_in_loader,
                # What produced the latents. Declared per model via vae_config_keys; empty for
                # a model that has not declared one, which keeps its caches exactly as before.
                vae_identity=model.vae_cache_key(),
                on_the_fly_media=on_the_fly_media,
                layout_caption_format=getattr(model, 'layout_caption_format', None),
            )
            self.directory_datasets.append(directory_dataset)

    def post_init(self, data_parallel_rank, data_parallel_world_size, per_device_batch_size: dict, gradient_accumulation_steps, per_device_batch_size_image: dict):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        global_batch_size = {size: bs * gradient_accumulation_steps * self.data_parallel_world_size for size, bs in per_device_batch_size.items()}
        global_batch_size_image = {size: bs * gradient_accumulation_steps * self.data_parallel_world_size for size, bs in per_device_batch_size_image.items()}

        # group same size_bucket together
        datasets_by_size_bucket = defaultdict(list)
        for directory_dataset in self.directory_datasets:
            for size_bucket_dataset in directory_dataset.get_size_bucket_datasets():
                datasets_by_size_bucket[size_bucket_dataset.size_bucket].append(size_bucket_dataset)
        self.buckets = []
        for datasets in datasets_by_size_bucket.values():
            self.buckets.append(ConcatenatedBatchedDataset(datasets))

        for bucket in self.buckets:
            bucket.post_init(global_batch_size, global_batch_size_image, data_parallel_rank,
                             data_parallel_world_size, batch_fill=self.batch_fill)
        self._report_batch_fill()

        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            iteration_order.extend([i]*(len(bucket)))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.buckets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        if DEBUG:
            print(f'Dataset iteration_order: {self.iteration_order}')

        self.post_init_called = True

        if subsample_ratio := self.dataset_config.get('subsample_ratio', None):
            new_len = int(len(self) * subsample_ratio)
            self.iteration_order = self.iteration_order[:new_len]

    def _report_batch_fill(self):
        """Say what the fill did, per bucket, at startup.

        Batch fill changes how many steps an epoch has and can quietly turn a quarter of a
        batch into masked padding. Both are things someone reading a loss curve needs to know
        about beforehand, not deduce from it.
        """
        if self.batch_fill['batch_fill_strategy'] != 'fill' or not is_main_process():
            return
        reports = [b.fill_report for b in self.buckets if b.fill_report is not None]
        if not reports:
            return
        print(f"batch_fill_strategy = 'fill' (undersized_bucket = "
              f"{self.batch_fill['undersized_bucket']!r}, min_real_fraction = "
              f"{self.batch_fill['min_real_fraction']}):")
        for r in reports:
            line = (f"  bucket {r['size_bucket']}: {r['num_samples']} samples, global batch "
                    f"{r['global_batch_size']}, {r['batches_before']} -> {r['batches_after']} batches, "
                    f"+{r['num_added']} filled")
            if r['num_masked']:
                line += f", {r['num_masked']} of them masked out"
            print(line)

    def set_epoch(self, epoch):
        """Tell every bucket which epoch is starting, so the fill can rotate.

        A no-op unless batch_fill_strategy is 'fill' and fill_rotate_per_epoch is on, so the
        default path does no work and changes nothing.
        """
        for bucket in self.buckets:
            bucket.set_epoch(epoch)

    def is_batch_fill_static(self):
        """True when set_epoch can never change anything, so nothing has to be rebuilt for it."""
        return all(bucket._is_static() for bucket in self.buckets)

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order)

    def __getitem__(self, idx):
        assert self.post_init_called
        i, j = self.iteration_order[idx]
        bucket = self.buckets[i]
        examples_for_this_dp_rank = bucket[j]
        batch = self._collate(examples_for_this_dp_rank, mask_shape=bucket.mask_shape)
        return batch

    # Collates a list of feature dictionaries into a single dictionary of batched features.
    # Each feature can be a tensor, list, or single item.
    def _collate(self, examples, mask_shape=None):
        # Popped before the loop below, which would otherwise collate it into the batch and
        # hand it to prepare_inputs as though the model had asked for it.
        weights = [example.pop(SAMPLE_WEIGHT_KEY, 1.0) for example in examples]
        ret = {}
        for key in examples[0]:
            if key == 'mask':
                continue  # mask is handled specially below
            features = [example[key] for example in examples]
            if torch.is_tensor(features[0]):
                shape = features[0].shape
                if all(torch.is_tensor(f) and f.shape == shape for f in features):
                    # if we can form a single batched tensor, do it
                    features = torch.stack(features)
            ret[key] = features
        # Only some items in the batch might have valid mask.
        masks = [example['mask'] for example in examples]
        # See if we have any valid masks. If we do, they should all have the same shape.
        shape = None
        for mask in masks:
            if mask is not None:
                assert shape is None or mask.shape == shape
                shape = mask.shape
        if all(w == 1.0 for w in weights):
            # No batch fill, or a batch that is entirely real samples. Byte for byte the path
            # this had before batch fill existed, mask = None included, so nothing about an
            # ordinary run changes.
            if shape is not None:
                # At least one item has a mask. Need to make the None masks all 1s.
                for i, mask in enumerate(masks):
                    if mask is None:
                        masks[i] = torch.ones(shape, dtype=torch.float16)
                ret['mask'] = torch.stack(masks)
            else:
                # We can leave the batch mask as None and the loss_fn will skip masking entirely.
                ret['mask'] = None
            return ret

        # This batch holds padding, so every sample needs a mask carrying its weight -- the
        # padding's zero, and the compensating G/G_real on the rest. Reusing the existing mask
        # channel is what keeps all of this out of the eight get_loss_fn implementations: they
        # already multiply by the mask elementwise before averaging.
        #
        # A real mask's shape wins when there is one, so a synthesised mask can never disagree
        # with it. mask_shape is only the fallback for a batch where nobody supplied one, and
        # it only has to be a plausible spatial shape: every model interpolates the mask to the
        # latent size, and a constant survives that unchanged.
        if shape is None:
            shape = mask_shape
        if shape is None:
            raise RuntimeError(
                'A padded batch needs a mask shape and none could be determined. This means the '
                'size bucket had no spatial dimensions to fall back on.'
            )
        for i, (mask, weight) in enumerate(zip(masks, weights)):
            base = torch.ones(shape, dtype=torch.float16) if mask is None else mask
            # float16 to match what PreprocessMediaFile produces, so a synthesised mask and a
            # real one stack. The weight rounds to about four significant digits here, which is
            # far below the noise of bf16 training and keeps the mask at its existing size.
            masks[i] = base * weight
        ret['mask'] = torch.stack(masks)
        return ret

    def cache_metadata(self, regenerate_cache=False, trust_cache=False):
        for ds in self.directory_datasets:
            ds.cache_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)

    def cache_latents(self, map_fn, regenerate_cache=False, trust_cache=False, caching_batch_size=1, takes=None):
        for ds in self.directory_datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, trust_cache=trust_cache,
                             caching_batch_size=caching_batch_size, takes=takes)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1, takes=None, uncond=True):
        text_encoder_key = getattr(self.model, 'text_encoder_cache_key', lambda _i: '')(i)
        text_encoder_identity = getattr(
            self.model, 'text_encoder_identity', lambda _i: text_encoder_key)(i)
        require_identity = getattr(self.model, 'text_encoder_requires_identity', lambda _i: False)(i)
        for ds in self.directory_datasets:
            ds.cache_text_embeddings(
                map_fn, i, regenerate_cache=regenerate_cache,
                caching_batch_size=caching_batch_size, text_encoder_key=text_encoder_key,
                # Two different questions, so two calls. The fingerprint catches a change the
                # cache must react to by rebuilding; the manifest only says whose contents
                # these are. A model that keeps something out of its fingerprint, to avoid
                # moving every existing install's cache path, can still declare it here.
                identity=text_encoder_identity,
                require_identity=require_identity,
                takes=takes, uncond=uncond,
            )
        if uncond:
            # some techniques need access to the uncond
            self.model.uncond_dict = self.directory_datasets[0].uncond_dict


def _cache_fn(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, trust_cache, caching_batch_size,
              stage='all', node=0, text_encoder=None):
    # Dataset map() starts a bunch of processes. Make sure torch uses a limited number of threads
    # to avoid CPU contention.
    # TODO: if we ever change Datasets map to use spawn instead of fork, this might not work.
    #torch.set_num_threads(os.cpu_count() // NUM_PROC)
    # HF Datasets map can randomly hang if this is greater than one (???)
    # See https://github.com/pytorch/pytorch/issues/10996
    # Alternatively, we could try fixing this by using spawn instead of fork.
    torch.set_num_threads(1)
    try:
        _cache_datasets(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, trust_cache,
                        caching_batch_size, stage, node, text_encoder)
    finally:
        # Signal that we're done, also on an error: the GPU ranks stop waiting for work and global
        # rank 0 finds this process's exit code.
        queue.put(None)


def check_shared_legacy_caches(datasets):
    """Directories reading one source (a train and an eval dataset) share its legacy metadata
    cache, grouped datasets included, under one caption profile: other buckets would make each
    run regroup over the other's grouping."""
    seen = {}
    for ds in datasets:
        for directory in ds.directory_datasets:
            if directory.reuse_metadata_cache:
                continue
            key = (os.path.realpath(directory.cache_dir), directory.caption_cache_suffix)
            bucketing = directory._bucketing()
            other = seen.setdefault(key, bucketing)
            if other != bucketing:
                raise ValueError(
                    f'{directory.path} is read by two dataset directories with the same caption settings and other '
                    f'buckets ({other} and {bucketing}). With reuse_metadata_cache = false they share one metadata '
                    'cache. Give them the same buckets, or set reuse_metadata_cache = true, which keeps a grouping '
                    'per bucket setting.')


def metadata_units(datasets):
    """{source directory: its DirectoryDatasets}, in the order of the dataset configs: what one
    node caches the metadata of. Directories reading the same source (a train and an eval
    dataset, or two settings of one directory) share files under it, so one node caches them one
    after the other, as a single node would."""
    units = {}
    for ds in datasets:
        for directory in ds.directory_datasets:
            # The real path: two spellings of one directory (relative, through a symlink) are
            # one set of files.
            units.setdefault(os.path.realpath(directory.path), []).append(directory)
    return units


def tensor_unit(bucket):
    """The unit of work a bucket's caches belong to: its cache directory, or with
    reuse_metadata_cache that bucket in every caption profile of its snapshot, since a profile
    caches its latents from another's. Buckets of one unit (also a train and an eval dataset
    reading the same directory) are cached by one node, one after the other, as a single node
    would."""
    directory = bucket.directory_dataset
    if getattr(directory, 'reuse_metadata_cache', False):
        return os.path.realpath(directory.snapshot_dir / bucket.cache_dir.name)
    return os.path.realpath(bucket.cache_dir)


def tensor_units(datasets):
    """Units of work (see tensor_unit), largest first: what one node caches the latents, or the
    text embeddings, of. The order is the same on every node that loads the same metadata."""
    rows = defaultdict(int)
    for ds in datasets:
        for directory in ds.directory_datasets:
            for bucket in directory.bucket_datasets():
                # An aspect ratio bucket is encoded once for each of its resolutions.
                rows[tensor_unit(bucket)] += len(bucket.metadata_dataset) * len(getattr(bucket, 'resolutions', [None]))
    return [unit for unit, count in sorted(rows.items(), key=lambda item: (-item[1], item[0]))]


def _claim(queue, pool, units):
    """The index in units of the next unit of work in pool for this node, len(units) or more once
    there is none left. Every node draws from the same counter, through a GPU rank of its node
    (see _claim_work), so a node that finishes early takes more."""
    parent_conn, child_conn = mp.Pipe(duplex=False)
    queue.put(('claim', pool, hashlib.sha256(json.dumps(units).encode()).hexdigest(), child_conn))
    return parent_conn.recv()


def _work_from_pool(queue, pool, units, work):
    count = 0
    while (index := _claim(queue, pool, list(units))) < len(units):
        unit = list(units)[index]
        print(f'{socket.gethostname()}: caching {pool} of {unit} ({index + 1} of {len(units)})')
        work(unit)
        count += 1
    print(f'{socket.gethostname()}: cached {pool} of {count} of {len(units)} on this node')


def _cache_datasets(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, trust_cache,
                    caching_batch_size, stage='all', node=0, text_encoder=None):
    global _REFRESHED_SNAPSHOTS, _regenerate_flattened
    _REFRESHED_SNAPSHOTS = {}
    _regenerate_flattened = stage != 'text' or text_encoder == 0
    try:
        _cache_stage(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, trust_cache,
                     caching_batch_size, stage, node, text_encoder)
    finally:
        _REFRESHED_SNAPSHOTS = None
        _regenerate_flattened = True


def _cache_stage(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, trust_cache,
                 caching_batch_size, stage, node, text_encoder):
    """stage 'all' caches everything here, 'metadata_only' only the metadata (--metadata_only).
    With several nodes, every node runs stage 'metadata', then 'latents', then 'text' once per
    text encoder, each taking units of work from a pool all the nodes share until none is left.
    Each is a round of its own, so a GPU swaps models once, not for every unit, and no two nodes
    work on one bucket at once: the text encoders of a bucket share its flattened captions."""
    if stage == 'metadata':
        units = metadata_units(datasets)
        _work_from_pool(queue, 'metadata', units, lambda unit: [
            directory.cache_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)
            for directory in units[unit]])
        return
    if stage in ('latents', 'text'):
        # Cached in the 'metadata' round; what any rank loads once caching is done.
        for ds in datasets:
            ds.cache_metadata(trust_cache=True)
    else:
        for ds in datasets:
            ds.cache_metadata(regenerate_cache=regenerate_cache, trust_cache=trust_cache)
        if stage == 'metadata_only':
            return

    pipes = {}

    def latents_map_fn(example, rank):
        is_edit_dataset = ('control_file' in example)
        first_size_bucket = example['size_bucket'][0]
        tensors_and_masks = []
        image_specs = []
        captions = []
        control_tensors_and_masks = []
        for i, (image_spec, mask_path, size_bucket, caption) in enumerate(
            zip(example['image_spec'], example['mask_file'], example['size_bucket'], example['caption'])
        ):
            assert size_bucket == first_size_bucket
            items = preprocess_media_file_fn(image_spec, mask_path, size_bucket)
            tensors_and_masks.extend(items)
            image_specs.extend([image_spec] * len(items))
            captions.extend([caption] * len(items))
            if is_edit_dataset:
                control_file = example['control_file'][i]
                control_items = preprocess_media_file_fn((None, control_file), None, size_bucket)
                assert len(control_items) == 1
                assert len(items) == 1
                control_tensors_and_masks.append(control_items[0])
            else:
                control_tensors_and_masks.append(None)

        if len(tensors_and_masks) == 0:
            assert not is_edit_dataset
            return {'latents': [], 'mask': [], 'image_spec': [], 'caption': []}

        caching_batch_size = len(example['image_spec'])
        results = defaultdict(list)
        for i in range(0, len(tensors_and_masks), caching_batch_size):
            tensor = torch.stack([t[0] for t in tensors_and_masks[i:i+caching_batch_size]])
            c_tensor = torch.stack([t[0] for t in control_tensors_and_masks[i:i+caching_batch_size]]) if is_edit_dataset else None
            audio_list = [t[1] for t in tensors_and_masks[i:i+caching_batch_size]]  # needs to be list because any example's audio could be None
            if rank not in pipes:
                pipes[rank] = mp.Pipe(duplex=False)
            parent_conn, child_conn = pipes[rank]
            queue.put((0, tensor, c_tensor, audio_list, child_conn))
            result = parent_conn.recv()  # dict
            for k, v in result.items():
                results[k].append(v)
        # concatenate the batches
        for k, v in results.items():
            if isinstance(v[0], list):
                # some features could be lists: [[audio1, audio2], [audio3, None]]
                results[k] = sum(v, [])
            else:
                results[k] = torch.cat(v)
        results['image_spec'] = image_specs
        results['mask'] = [t[-1] for t in tensors_and_masks]
        results['caption'] = captions
        return results

    def text_embedding_map_fn(text_encoder_idx):
        def fn(example, rank):
            if rank not in pipes:
                pipes[rank] = mp.Pipe(duplex=False)
            parent_conn, child_conn = pipes[rank]
            control_file = example['control_file'] if 'control_file' in example else None
            queue.put((text_encoder_idx+1, example['caption'], example['is_video'], control_file, child_conn))
            result = parent_conn.recv()  # dict
            result['image_spec'] = example['image_spec']
            return result
        return fn

    def cache_latents(takes=None):
        for ds in datasets:
            ds.cache_latents(latents_map_fn, regenerate_cache=regenerate_cache, trust_cache=trust_cache,
                             caching_batch_size=caching_batch_size, takes=takes)

    def cache_text_embeddings(text_encoder_idx, takes=None, uncond=True):
        for ds in datasets:
            ds.cache_text_embeddings(text_embedding_map_fn(text_encoder_idx), text_encoder_idx+1,
                                     regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size,
                                     takes=takes, uncond=uncond)

    if stage == 'all':
        cache_latents()
        for text_encoder_idx in range(num_text_encoders):
            cache_text_embeddings(text_encoder_idx)
        return

    units = tensor_units(datasets)

    def taking(unit):
        return lambda bucket: tensor_unit(bucket) == unit

    if stage == 'latents':
        _work_from_pool(queue, 'latents', units, lambda unit: cache_latents(taking(unit)))
        return

    def text_embeddings(text_encoder_idx, unit):
        takes = taking(unit)
        # An aspect ratio bucket makes the size buckets its text embeddings go to while it caches
        # its latents, which may have been on another node: load those here.
        for ds in datasets:
            ds.cache_latents(None, trust_cache=True, takes=lambda bucket: (
                takes(bucket) and isinstance(bucket, ARBucketDataset) and not bucket.size_buckets))
        cache_text_embeddings(text_encoder_idx, takes, uncond=False)

    _work_from_pool(queue, f'text embeddings {text_encoder + 1}', units,
                    lambda unit: text_embeddings(text_encoder, unit))
    if node == 0:
        # The empty caption's embedding is the directory's, not a bucket's.
        cache_text_embeddings(text_encoder, takes=lambda bucket: False)


# How long ranks that do no caching wait for the ones that do: caching millions of images takes
# hours, far past the process group's timeout.
CACHING_WAIT = timedelta(days=7)
# How often the ranks taking caching work look for a failure while the queue is empty.
CACHING_POLL_SECONDS = 10
# How often each rank tells the store it is alive while caching, and how long a rank may stay
# silent before the others take it for dead (see _Liveness).
CACHING_HEARTBEAT_SECONDS = 30
CACHING_SILENCE = timedelta(minutes=15)
_caching_rounds = 0


def _share_work_queue():
    """(queue, manager, error): the caching work queue on the ranks that can reach it, None on the
    others; error is this rank's failure to connect to it.

    Global rank 0 serves the queue from a multiprocess Manager, behind a Unix socket in its own
    machine's /tmp, so only the ranks on that machine can connect; ranks on other nodes do no
    caching and wait for it to finish. That is caching on one node, or with multi_node_caching
    off (_share_node_queues has the queues of caching on every node). The manager is returned on
    rank 0 only, to be kept alive while the queue is in use.
    """
    error = None
    if is_main_process():
        try:
            manager = mp.Manager()
            queue = manager.Queue()
            shared = [(socket.gethostname(), manager.address, pickle.dumps(queue))]
        except Exception as e:
            # Told to every rank, which would otherwise wait in the broadcast until it times out.
            error = e
            shared = ['failed']
    else:
        shared = [None]
    torch.distributed.broadcast_object_list(shared, src=0, group=dist.get_world_group())
    if is_main_process():
        if error is not None:
            raise error
        return queue, manager, None
    if shared[0] == 'failed':
        raise RuntimeError('Dataset caching could not start on global rank 0: see its log')
    host, address, payload = shared[0]
    reachable = host == socket.gethostname() and (not str(address).startswith('/') or os.path.exists(address))
    if not reachable:
        return None, None, None
    try:
        return pickle.loads(payload), None, None
    except Exception as e:
        # Raised from _run_caching, after telling the ranks taking work to stop.
        return None, None, e


def _caching_key(round_):
    return f'diffusion_pipe_caching_{round_}'


def _report_caching(round_, ok):
    """Global rank 0 tells the ranks of other nodes, waiting in _wait_for_caching, how caching went."""
    torch.distributed.distributed_c10d._get_default_store().set(_caching_key(round_), 'ok' if ok else 'failed')


def _wait_for_caching(round_, liveness):
    # The rendezvous store rather than a barrier: a barrier would time out long before caching ends.
    store = torch.distributed.distributed_c10d._get_default_store()
    _wait_in_store([_caching_key(round_)], liveness)
    if store.get(_caching_key(round_)) != b'ok':
        raise RuntimeError('Dataset caching failed on global rank 0: see its log')


def _abort_key(round_):
    return _caching_key(round_) + '_abort'


def _abort_caching(round_):
    torch.distributed.distributed_c10d._get_default_store().set(_abort_key(round_), '1')


class _Liveness:
    """Every rank beats a counter in the store from a thread while a round of caching runs, and
    the ranks that wait (for a task, or for other nodes) watch the others' counters. A rank that
    dies (killed for memory, say) answers no task and reports nothing: without this, its node's
    caching process would wait for its answer forever and every rank would wait out CACHING_WAIT.
    Counters, not clocks, so the nodes' clocks need not agree."""

    def __init__(self, round_):
        self.round_ = round_
        self.store = torch.distributed.distributed_c10d._get_default_store()
        self.seen = {}
        self.checked = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._beat, daemon=True)
        self.thread.start()

    def _key(self, rank):
        return f'{_caching_key(self.round_)}_alive{rank}'

    def _beat(self):
        key = self._key(dist.get_rank())
        while True:
            try:
                self.store.add(key, 1)
            except Exception:
                # The store is out of reach: to the others this rank falls silent, as it should.
                pass
            if self.stop.wait(CACHING_HEARTBEAT_SECONDS):
                return

    def check(self):
        """Raise if a rank has been silent for CACHING_SILENCE. Looks at most once a beat."""
        now = time.monotonic()
        if self.checked is not None and now - self.checked < CACHING_HEARTBEAT_SECONDS:
            return
        self.checked = now
        for rank in range(dist.get_world_size()):
            count = self.store.add(self._key(rank), 0)
            if rank not in self.seen or self.seen[rank][0] != count:
                self.seen[rank] = (count, now)
            elif now - self.seen[rank][1] > CACHING_SILENCE.total_seconds():
                raise RuntimeError(f'Global rank {rank} has given no sign of life for {CACHING_SILENCE} during '
                                   'dataset caching: it may have died, see its log')

    def close(self):
        self.stop.set()
        self.thread.join()


def _wait_in_store(keys, liveness):
    """Wait until every key is set: a poll rather than store.wait, to notice a dead rank, which
    would never set its key."""
    store = torch.distributed.distributed_c10d._get_default_store()
    deadline = time.monotonic() + CACHING_WAIT.total_seconds()
    while not store.check(keys):
        if time.monotonic() > deadline:
            raise RuntimeError(f'Dataset caching did not finish within {CACHING_WAIT}')
        liveness.check()
        time.sleep(CACHING_POLL_SECONDS)


def _work_on_queue(queue, handle_task, process, round_, liveness):
    """Take caching tasks until the caching process is done. `process` is that process on global
    rank 0, None elsewhere.

    A rank that fails sets an abort key in the store, which the others look for between tasks: the
    caching process may be waiting forever for that rank's result, and would never end the queue.
    """
    store = torch.distributed.distributed_c10d._get_default_store()
    try:
        while True:
            if store.check([_abort_key(round_)]):
                raise RuntimeError('Dataset caching failed on another rank: see its log')
            try:
                task = queue.get(timeout=CACHING_POLL_SECONDS)
            except Empty:
                if process is not None and not process.is_alive():
                    raise RuntimeError(f'The dataset caching process ended with exit code {process.exitcode} '
                                       'before finishing: see its log')
                liveness.check()
                continue
            if task is None:
                # Propagate None so all worker processes break out of this loop.
                # This is safe because it's a FIFO queue. The first None always comes after all work items.
                queue.put(None)
                break
            handle_task(task)
        if process is not None:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f'The dataset caching process failed with exit code {process.exitcode}: see its log')
    except BaseException:
        _abort_caching(round_)
        raise


def _run_caching(queue, handle_task, process, round_, error=None):
    """Every rank's part in one round of dataset caching; raises on every rank if any failed.

    error is a failure of this rank while setting the round up: it then takes no work, and tells
    the ranks taking work to stop.
    """
    liveness = _Liveness(round_)
    try:
        if error is not None:
            _abort_caching(round_)
        elif queue is None:
            # Another node: its GPUs are not reached by the queue. The cache is read from the shared
            # filesystem once rank 0's node has written it.
            if int(os.environ.get('LOCAL_RANK', '0')) == 0:
                print(f'{socket.gethostname()}: dataset caching runs on the GPUs of global rank 0\'s node; '
                      'waiting for it')
            _wait_for_caching(round_, liveness)
        else:
            _work_on_queue(queue, handle_task, process, round_, liveness)
    except Exception as e:
        error = e
    finally:
        liveness.close()
    if is_main_process():
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join()
        _report_caching(round_, ok=error is None)
    all_ok(error)


def _node_layout():
    """(node, nodes, leader): this rank's node, numbered in the order of the nodes' lowest global
    ranks (global rank 0's node is 0), the number of nodes, and whether this rank is the lowest of
    its node. A node is a hostname."""
    hosts = [None] * dist.get_world_size()
    torch.distributed.all_gather_object(hosts, socket.gethostname(), group=dist.get_world_group())
    rank = dist.get_rank()
    nodes = list(dict.fromkeys(hosts))
    return nodes.index(hosts[rank]), len(nodes), hosts.index(hosts[rank]) == rank


def _share_node_queues(leader):
    """(queue, manager, error) as _share_work_queue, for a queue on every node: the lowest rank of
    each node serves one to the ranks of its node."""
    error = None
    entry = None
    if leader:
        try:
            manager = mp.Manager()
            queue = manager.Queue()
            entry = (socket.gethostname(), manager.address, pickle.dumps(queue))
        except Exception as e:
            error = e
            entry = 'failed'
    entries = [None] * dist.get_world_size()
    torch.distributed.all_gather_object(entries, entry, group=dist.get_world_group())
    if 'failed' in entries:
        if error is not None:
            raise error
        raise RuntimeError(f'Dataset caching could not start on global rank {entries.index("failed")}: see its log')
    if leader:
        return queue, manager, None
    host = socket.gethostname()
    _, address, payload = next(e for e in entries if e is not None and e[0] == host)
    if str(address).startswith('/') and not os.path.exists(address):
        return None, None, RuntimeError(
            f'{host} cannot reach the caching queue of its node at {address}: does another machine have '
            'the same hostname?')
    try:
        return pickle.loads(payload), None, None
    except Exception as e:
        return None, None, e


def _claim_work(round_, pool, units_digest):
    """The next index of pool's counter, for a node's caching process (see _claim). The first
    claim records the digest of the units the counter indexes; a node that sees other units (other
    metadata) fails rather than cache the wrong ones."""
    store = torch.distributed.distributed_c10d._get_default_store()
    key = f'{_caching_key(round_)}_{pool}'
    if store.compare_set(key + '_units', '', units_digest).decode() != units_digest:
        raise RuntimeError(
            f'{socket.gethostname()} sees other dataset {pool} work than another node: every node must read the '
            'same dataset configs, and cache files from a volume they share')
    return store.add(key, 1) - 1


def _with_claims(round_, handle_task):
    """handle_task for a round in which caching processes draw work from pools: claims are
    answered here, everything else goes to handle_task."""
    def handle(task):
        if task[0] == 'claim':
            _, pool, units_digest, pipe = task
            pipe.send(_claim_work(round_, pool, units_digest))
        else:
            handle_task(task)
    return handle


def _node_key(round_, node):
    return f'{_caching_key(round_)}_node{node}'


def _wait_for_nodes(round_, nodes, liveness):
    store = torch.distributed.distributed_c10d._get_default_store()
    keys = [_node_key(round_, node) for node in range(nodes)]
    _wait_in_store(keys, liveness)
    failed = [node for node, key in enumerate(keys) if store.get(key) != b'ok']
    if failed:
        raise RuntimeError(f'Dataset caching failed on node(s) {failed} (numbered by their lowest global rank): '
                           'see their logs')


def _run_caching_on_every_node(queue, handle_task, process, round_, node, nodes, leader, error=None):
    """_run_caching for a round in which every node works through a queue of its own. The lowest
    rank of each node reports how its node did, and every rank waits for every node in the store:
    one node's share takes hours, far past a collective's timeout. A rank that fails sets the
    abort key, which stops the ranks of every node."""
    liveness = _Liveness(round_)
    try:
        if error is not None:
            _abort_caching(round_)
        else:
            _work_on_queue(queue, handle_task, process, round_, liveness)
    except Exception as e:
        error = e
    if leader:
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join()
        torch.distributed.distributed_c10d._get_default_store().set(
            _node_key(round_, node), 'ok' if error is None else 'failed')
    try:
        _wait_for_nodes(round_, nodes, liveness)
    except Exception as e:
        if error is None:
            error = e
    finally:
        liveness.close()
    all_ok(error)


def describe_buckets(dataset):
    """What --metadata_only prints: the files of each directory per bucket, largest first. An
    aspect ratio bucket is encoded once for each of its resolutions."""
    lines = []
    for directory in dataset.directory_datasets:
        buckets = sorted(directory.bucket_datasets(), key=lambda bucket: -len(bucket.metadata_dataset))
        total = sum(len(bucket.metadata_dataset) for bucket in buckets)
        lines.append(f'{directory.path}: {total} files in {len(buckets)} buckets')
        for bucket in buckets:
            if isinstance(bucket, ARBucketDataset):
                ar, frames = bucket.ar_frames
                name = f'aspect ratio {ar:.3f}, resolutions {", ".join(f"{float(r):g}" for r in bucket.resolutions)}'
            else:
                width, height, frames = bucket.size_bucket
                name = f'{width}x{height}'
            if frames > 1:
                name += f', {frames} frames'
            lines.append(f'  {name}: {len(bucket.metadata_dataset)}')
    return lines


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model, regenerate_cache=False, trust_cache=False, caching_batch_size=1, keep_models_loaded=False,
                 multi_node=True):
        self.model = model
        # With several nodes, each caches its share of the buckets on its own GPUs. Off, global
        # rank 0's node caches everything while the others wait.
        self.multi_node = multi_node
        self.vae = self.model.get_vae()
        # Nothing to cache for text encoders that encode each batch during training instead.
        self.text_encoders = [] if encodes_text_in_loader(model) else self.model.get_text_encoders()
        self.submodels = [self.vae] + list(self.text_encoders)
        self.call_vae_fn = self.model.get_call_vae_fn(self.vae)
        self.vae_supports_audio = 'audio' in signature(self.call_vae_fn).parameters
        self.call_text_encoder_fns = [self.model.get_call_text_encoder_fn(text_encoder) for text_encoder in self.text_encoders]
        self.te_fn_requires_control_file = [
            len(signature(fn).parameters) == 3
            for fn in self.call_text_encoder_fns
        ]
        self.regenerate_cache = regenerate_cache
        self.trust_cache = trust_cache
        self.caching_batch_size = caching_batch_size
        self.keep_models_loaded = keep_models_loaded
        self.datasets = []
        self.recent_text_encodings = _RecentTextEncodings()

    def register(self, dataset):
        self.datasets.append(dataset)

    # Some notes for myself:
    # Use the third-party multiprocess library because HF Datasets uses it for the map() calls.
    # Mix and match native multiprocessing / torch.multiprocessing and multiprocess at your peril! Things can break.
    # In patches.py we register reductions so Tensors sent over Queues and Pipes are efficient just like in torch.multiprocessing.
    def cache(self, unload_models=True, metadata_only=False):
        """metadata_only (--metadata_only): scan the directories, group them into buckets and
        cache that, with no latents or text embeddings, so neither the VAE nor a text encoder
        runs. A later run reads this metadata as it reads any metadata cache."""
        check_shared_legacy_caches(self.datasets)
        node, nodes, leader = _node_layout() if self.multi_node else (0, 1, is_main_process())
        if nodes == 1:
            self._cache_on_rank_0s_node('metadata_only' if metadata_only else 'all')
        else:
            # Every node takes directories, then buckets, from pools they share, until none are
            # left. Each round ends on every node before the next begins: the metadata of every
            # directory is cached before any bucket is drawn, the latents of every bucket before
            # any text embedding, and one text encoder's embeddings before the next encoder's.
            self._cache_on_every_node('metadata', node, nodes, leader)
            if not metadata_only:
                self._cache_on_every_node('latents', node, nodes, leader)
                for text_encoder in range(len(self.text_encoders)):
                    self._cache_on_every_node('text', node, nodes, leader, text_encoder=text_encoder)

        if metadata_only:
            # No model was used. Every rank reads the metadata back, so one that cannot (a node
            # that does not see the volume rank 0 wrote to) fails the run here and not at training.
            dist.barrier()
            error = None
            try:
                for ds in self.datasets:
                    ds.cache_metadata(trust_cache=True)
            except Exception as e:
                error = e
            all_ok(error)
            return

        if unload_models:
            # Free memory in all unneeded submodels. This is easier than trying to delete every reference.
            # TODO: check if this is actually freeing memory.
            for model in self.submodels:
                if not isinstance(model, nn.Module):
                    continue
                if (self.model.name == 'sdxl' and model is self.vae) or self.keep_models_loaded:
                    # If full fine tuning SDXL, we need to keep the VAE weights around for saving the model.
                    model.to('cpu')
                else:
                    model.to('meta')
            # TODO: this keeps weights in RAM, which can OOM for very large text encoders. Not a big deal since you can
            # just relaunch the training script, since the data is now cached. But would be good to make it work in one pass.
            mm.unload_all_models()  # Comfy managed models

        dist.barrier()

        # Now load all datasets from cache. A rank that cannot fails every rank, rather than leave
        # the others waiting in the next collective until it times out.
        error = None
        try:
            for ds in self.datasets:
                ds.cache_metadata(trust_cache=True)
                ds.cache_latents(None, trust_cache=True)
                for i in range(1, len(self.text_encoders)+1):
                    ds.cache_text_embeddings(None, i)
        except Exception as e:
            error = e
        all_ok(error)

    def _start_caching_process(self, queue, **stage):
        # start up a process to run through the dataset caching flow
        process = mp.Process(
            target=_cache_fn,
            args=(
                self.datasets,
                queue,
                self.model.get_preprocess_media_file_fn(),
                len(self.text_encoders),
                self.regenerate_cache,
                self.trust_cache,
                self.caching_batch_size,
            ),
            kwargs=stage,
        )
        process.start()
        return process

    def _cache_on_rank_0s_node(self, stage='all'):
        """All of caching on the GPUs of global rank 0's node."""
        global _caching_rounds
        _caching_rounds += 1
        round_ = _caching_rounds
        queue, manager, error = _share_work_queue()
        process = None
        if is_main_process():
            try:
                process = self._start_caching_process(queue, stage=stage)
            except Exception as e:
                error = e
        # loop on the original processes (one per GPU) to handle tasks requiring GPU models (VAE, text encoders)
        _run_caching(queue, self._handle_task, process, round_, error)

    def _cache_on_every_node(self, stage, node, nodes, leader, **text_encoder):
        global _caching_rounds
        _caching_rounds += 1
        round_ = _caching_rounds
        queue, manager, error = _share_node_queues(leader)
        process = None
        if leader:
            try:
                process = self._start_caching_process(queue, stage=stage, node=node, **text_encoder)
            except Exception as e:
                error = e
        _run_caching_on_every_node(queue, _with_claims(round_, self._handle_task), process, round_, node, nodes,
                                   leader, error)

    @torch.no_grad()
    def _handle_task(self, task):
        id = task[0]
        # moved needed submodel to cuda, and everything else to cpu
        submodel = self.submodels[id]
        if isinstance(submodel, nn.Module):
            if next(self.submodels[id].parameters()).device.type != 'cuda':
                for i, submodel in enumerate(self.submodels):
                    if i != id:
                        submodel.to('cpu')
                self.submodels[id].to('cuda')
        else:
            # ComfyUI model in a wrapper class that delays loading until the model is needed.
            self.submodels[id].load_model_if_needed()
        if id == 0:
            tensor, control_tensor, audio_list, pipe = task[1:]
            kwargs = {}
            if self.vae_supports_audio:
                kwargs['audio'] = audio_list
            if control_tensor is not None:
                # edit dataset
                results = self.call_vae_fn(tensor, control_tensor, **kwargs)
            else:
                results = self.call_vae_fn(tensor, **kwargs)
        elif id > 0:
            caption, is_video, control_file, pipe = task[1:]
            args = [caption, is_video]
            idx = id - 1
            if self.te_fn_requires_control_file[idx]:
                args.append(control_file)
            results = self.recent_text_encodings.encode(idx, self.call_text_encoder_fns[idx], args)
        else:
            raise RuntimeError()
        # Need to move to CPU here. If we don't, we get this error:
        # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        # I think this is because HF Datasets uses the multiprocess library (different from Python multiprocessing!) so it will always use fork.
        cpu_results = {}
        for k, v in results.items():
            if isinstance(v, (list, tuple)):
                cpu_results[k] = [x.to('cpu') if torch.is_tensor(x) else x for x in v]
            else:
                cpu_results[k] = v.to('cpu')
        pipe.send(cpu_results)


class _RecentTextEncodings:
    """Text-encoder results of recently encoded rows, so a row that repeats one is not encoded
    again: the cache_shuffle_num variants of a layout caption are one text, and repeated empty
    captions are too. Those repeats sit in adjacent rows, which caching_batch_size and the
    worker processes split across batches, but every batch is encoded here, in the one
    process that serves the GPU queue. Bounded in bytes; the oldest rows go first.

    A text encoder's result has one entry per row along its first dimension (the cache stores
    result[key][i] for row i). A result that does not is used as it is, and nothing is reused.
    A column that is None (control_file for a dataset without control images) is passed on
    as None.
    """

    def __init__(self, limit_bytes=256 << 20):
        self.limit_bytes = limit_bytes
        self.rows = OrderedDict()
        self.size = 0

    @staticmethod
    def _per_row(results, count):
        rows = [{} for _ in range(count)]
        for name, value in results.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == count:
                parts = [value[i].to('cpu').clone() for i in range(count)]
            elif isinstance(value, (list, tuple)) and len(value) == count:
                parts = [v.to('cpu').clone() if torch.is_tensor(v) else v for v in value]
            else:
                return None
            for row, part in zip(rows, parts):
                row[name] = part
        return rows

    @staticmethod
    def _row_bytes(row):
        return sum(v.numel() * v.element_size() for v in row.values() if torch.is_tensor(v))

    def _remember(self, key, row):
        size = self._row_bytes(row)
        if size > self.limit_bytes:
            return
        self.rows[key] = (row, size)
        self.size += size
        while self.size > self.limit_bytes:
            self.size -= self.rows.popitem(last=False)[1][1]

    def encode(self, encoder, encode, columns):
        count = len(columns[0])
        keys = [repr((encoder, row)) for row in zip(*(c for c in columns if c is not None))]
        found, missing = {}, []
        for i, key in enumerate(keys):
            if key in found:
                continue
            if key in self.rows:
                self.rows.move_to_end(key)
                found[key] = self.rows[key][0]
            else:
                found[key] = None
                missing.append(i)
        if len(missing) == count:
            results = encode(*columns)
            rows = self._per_row(results, count)
            for key, row in zip(keys, rows or []):
                self._remember(key, row)
            return results
        if missing:
            results = encode(*[None if c is None else [c[i] for i in missing] for c in columns])
            rows = self._per_row(results, len(missing))
            if rows is None:
                return encode(*columns)
            for i, row in zip(missing, rows):
                found[keys[i]] = row
                self._remember(keys[i], row)
        names = found[keys[0]].keys()
        return {name: [found[key][name] for key in keys] for name in names}


def split_batch(batch, pieces):
    # Each of features, label is a tuple of tensors.
    features, label = batch
    split_size = features[0].size(0) // pieces
    # The tuples passed to Deepspeed need to only contain tensors. For None (e.g. mask, or optional conditioning), convert to empty tensor.
    split_features = zip(*(torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])]*pieces for tensor in features))
    split_label = zip(*(torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])]*pieces for tensor in label))
    # Deepspeed works with a tuple of (features, labels).
    return list(zip(split_features, split_label))


# Splits an example (feature dict) along the batch dimension into a list of examples.
# Keeping this code because we might want to switch to this way of doing things eventually.
# def split_batch(example, pieces):
#     key, value = example.popitem()
#     input_batch_size = len(value)
#     example[key] = value
#     split_size = input_batch_size // pieces
#     examples = [{} for _ in range(pieces)]
#     for key, value in example.items():
#         assert len(value) == input_batch_size
#         for i, j in enumerate(range(0, input_batch_size, split_size)):
#             examples[i][key] = value[j:j+split_size]
#     return examples


# DataLoader that divides batches into microbatches for gradient accumulation steps when doing
# pipeline parallel training. Iterates indefinitely (deepspeed requirement). Keeps track of epoch.
# Updates epoch as soon as the final batch is returned (notably different from qlora-pipe).
class PipelineDataLoader:
    def __init__(self, dataset, model_engine, gradient_accumulation_steps, model, num_dataloader_workers=1,
                 text_encoding=None):
        if len(dataset) == 0:
            raise RuntimeError(
                'Processed dataset was empty. Probably caused by rounding down for each size bucket.\n'
                'Try decreasing the global batch size, or increasing num_repeats.\n'
                f'The dataset config that triggered this error was:\n{dataset.dataset_config}'
            )
        self.model = model
        # A utils.text_encoding.TextEncodingOnTheFly when the captions are encoded per batch.
        self.text_encoding = text_encoding
        self.dataset = dataset
        self.model_engine = model_engine
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_dataloader_workers = num_dataloader_workers
        self.iter_called = False
        self.eval_quantile = None
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        self._set_dataset_epoch()
        # Be careful to only create the DataLoader some bounded number of times: https://github.com/pytorch/pytorch/issues/91252
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def _set_dataset_epoch(self):
        """Hand the epoch to the dataset, for batch fill's per-epoch rotation.

        A no-op for every configuration except batch_fill_strategy = 'fill' with
        fill_rotate_per_epoch on, and the dataset decides that for itself.

        With workers, this assignment happens in the parent and does not reach them:
        persistent_workers keeps a forked/spawned copy of the dataset that was made when the
        DataLoader was built, so the workers would go on serving the previous epoch's tail.
        Rebuilding the DataLoader is what carries the new order across. Once per epoch is a
        bounded number of times, which is what the pytorch issue linked above is about -- the
        thing to avoid is rebuilding inside the loop.
        """
        if not hasattr(self.dataset, 'set_epoch'):
            return
        self.dataset.set_epoch(self.epoch)
        if self.num_dataloader_workers > 0 and not self.dataset.is_batch_fill_static():
            self.recreate_dataloader = True

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self._set_dataset_epoch()
        self.data = self._pull_batches_from_dataloader()

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __iter__(self):
        self.iter_called = True
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        if self.next_micro_batch == None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            self.epoch += 1
            # Before the DataLoader is rebuilt, so a rebuild triggered here picks up the new
            # epoch's tail rather than the previous one's.
            self._set_dataset_epoch()
            if self.recreate_dataloader:
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = None
        return ret

    def _create_dataloader(self, skip_first_n_batches=None):
        if skip_first_n_batches is not None:
            sampler = SkipFirstNSampler(skip_first_n_batches, len(self.dataset))
        else:
            sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=False,
            batch_size=None,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            persistent_workers=(self.num_dataloader_workers > 0),
            prefetch_factor=2 if self.num_dataloader_workers > 0 else None,
        )

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            # For models whose training noise levels follow an epoch schedule (MOSS-Image's
            # clean_end_share); eval batches are prepared with a quantile instead.
            if self.eval_quantile is None and hasattr(self.model, 'set_training_epoch'):
                self.model.set_training_epoch(self.epoch)
            if self.text_encoding is not None:
                batch = self.text_encoding(batch)
            features, label = self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)
            *target_list, mask = label
            # The target depends on the noise, so we must broadcast it from the first stage to the last.
            # NOTE: I had to patch the pipeline parallel TrainSchedule so that the LoadMicroBatch commands
            # would line up on the first and last stage so that this doesn't deadlock.
            broadcasted_targets = []
            for t in target_list:
                broadcasted_targets.append(self._broadcast_target(t))
            label = (*broadcasted_targets, mask)
            self.num_batches_pulled += 1
            for micro_batch in split_batch((features, label), self.gradient_accumulation_steps):
                yield micro_batch

    def _broadcast_target(self, target):
        model_engine = self.model_engine
        if not model_engine.is_pipe_parallel:
            return target

        assert model_engine.is_first_stage() or model_engine.is_last_stage()
        grid = model_engine.grid

        src_rank = grid.stage_to_global(0)
        dest_rank = grid.stage_to_global(model_engine.num_stages - 1)
        assert src_rank in grid.pp_group
        assert dest_rank in grid.pp_group
        target = target.to('cuda')  # must be on GPU to broadcast

        if model_engine.is_first_stage():
            dist.send(target, dest_rank)
        else:
            dist.recv(target, src_rank)
        return target

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
            'batches_per_epoch': len(self.dataset),
        }

    def load_state_dict(self, state_dict):
        assert not self.iter_called
        self.epoch = state_dict['epoch']
        # -1 because by preloading the next micro_batch, it's always going to have one more batch
        # pulled than the actual number of batches iterated by the caller. A checkpoint saved right
        # after the epoch rolled over has pulled none, and skips none.
        self.num_batches_pulled = max(state_dict['num_batches_pulled'] - 1, 0)
        saved = state_dict.get('batches_per_epoch')
        now = len(self.dataset)
        if saved is not None and saved != now:
            # Resumed on another number of GPUs or nodes, or with another micro batch size or
            # gradient_accumulation_steps: an epoch has another number of batches, so continue after
            # the same share of it. At most saved - 1 were done, so this stays below now.
            converted = self.num_batches_pulled * now // saved
            if is_main_process():
                print(f'WARNING: an epoch had {saved} batches when the checkpoint was saved and has {now} now: '
                      f'resuming epoch {self.epoch} after {converted} batches instead of {self.num_batches_pulled}, '
                      'about where it stopped.')
            self.num_batches_pulled = converted
        # The resumed epoch decides which samples pad the final batch, so this has to happen
        # before the DataLoader is built or the first epoch after a resume walks a different
        # tail than the run that was interrupted.
        self._set_dataset_epoch()
        self._create_dataloader(skip_first_n_batches=self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
        # Recreate the dataloader after the first pass so that it won't skip
        # batches again (we only want it to skip batches the first time).
        self.recreate_dataloader = True


class SkipFirstNSampler(torch.utils.data.Sampler):
    def __init__(self, n, dataset_length):
        super().__init__()
        self.n = n
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __iter__(self):
        for i in range(self.n, self.dataset_length):
            yield i


if __name__ == '__main__':
    from utils import common
    common.is_main_process = lambda: True
    from contextlib import contextmanager
    @contextmanager
    def _zero_first():
        yield
    common.zero_first = _zero_first

    from utils import dataset as dataset_util
    dataset_util.DEBUG = True

    from models import flux
    model = flux.CustomFluxPipeline.from_pretrained('/data2/imagegen_models/FLUX.1-dev', torch_dtype=torch.bfloat16)
    model.model_config = {'guidance': 1.0, 'dtype': torch.bfloat16}

    import toml
    dataset_manager = dataset_util.DatasetManager(model)
    with open('/home/anon/code/diffusion-pipe-configs/datasets/tiny1.toml') as f:
        dataset_config = toml.load(f)
    train_data = dataset_util.Dataset(dataset_config, model)
    dataset_manager.register(train_data)
    dataset_manager.cache()

    train_data.post_init(data_parallel_rank=0, data_parallel_world_size=1, per_device_batch_size=1, gradient_accumulation_steps=2)
    print(f'Dataset length: {len(train_data)}')

    for item in train_data:
        pass
