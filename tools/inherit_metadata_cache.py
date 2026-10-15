"""Give one model's dataset cache the source metadata another model already scanned.

    python tools/inherit_metadata_cache.py --dataset dataset.toml                # anima -> iris
    python tools/inherit_metadata_cache.py --dataset dataset.toml --dry-run      # check only
    python tools/inherit_metadata_cache.py --dataset dataset.toml --check-all    # stat every file

The first run of a model on a dataset enumerates every file, reads every image header and
every caption, and freezes the result as a source metadata snapshot under
<path>/cache/<model>/profiles/<geometry>/. For millions of images that scan is the slow part of
startup. The snapshot holds nothing model specific -- file paths, masks, raw captions and the
aspect-ratio bucket of each image -- so a model with the same bucketing can start from another
model's snapshot instead of scanning again. This tool copies it across (hard links when the
filesystem allows, so it costs no extra space). Everything after the snapshot (caption profile,
grouping, iteration order) is built by the target model's own code on its first run, from its
own dataset settings, exactly as if it had scanned the files itself.

Nothing is copied unless the two snapshots would be identical:
  * The bucketing settings must match: resolutions or size_buckets, ar buckets, frame_buckets,
    the rounding multiple, mask/control paths and shuffle_metadata. The donor snapshot is looked
    up by the hash of exactly these settings, so a different setting finds no donor.
  * Image datasets only (frame_buckets = [1]). The one setting allowed to differ is the model
    framerate, which only decides how many frames a video gets.
  * A sample of the recorded files (all of them with --check-all) must still exist, and a
    sample of their sizes must still land in the recorded buckets. A dataset that changed since the donor scanned it is refused, because
    a pixel-space model like Iris reads the images themselves and a vanished file would stop
    training hours later.
The donor cache is only read. Use --dry-run to see the verdict for every directory first.

With reuse_metadata_cache = false (the legacy cache, trusted with --trust_cache) there is no
snapshot. The tool then copies the donor's legacy metadata/ instead -- the file list, the
grouped metadata and the grouping keys -- so the target's --trust_cache run loads it without
scanning. That metadata already holds the captions as processed by the caption settings, so it
is copied only when the donor's recorded caption settings equal the target's, its buckets are
the target's buckets and its masks the target's masks, on top of the file checks above. Pass
--cache-text-embeddings when the target's training config sets cache_text_embeddings = true:
whether captions are augmented per step or when cached depends on it. The target's own
per-bucket and text embedding caches are built on its first run, as always.

Defaults are for Iris inheriting from Anima / Anima refiner, which share the 'anima' cache
directory. Both round to multiples of 16. Like the snapshot itself, the result reflects the
files as they were when the donor scanned them; run training with --regenerate_cache to
rescan after changing the dataset.
"""

import argparse
import copy
import datetime
import json
import os
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# diffusion-pipe's utils/ is a namespace package; ComfyUI's utils/ is a regular one and would
# win the name wherever it sits on sys.path ("No module named 'utils.dataset'"). Importing a
# light module from ours first pins it in sys.modules, as models/base.py does.
import utils.captions  # noqa: E402,F401
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'submodules' / 'ComfyUI'))

LEGACY_COLUMNS = {'image_spec', 'mask_file', 'caption', 'ar_bucket', 'size_bucket', 'is_video'}
REQUIRED_COLUMNS = LEGACY_COLUMNS | {'_caption_missing', '_caption_sidecar'}
# Tried when --from-framerate is not given. Only the hash depends on it for an image dataset.
CANDIDATE_FRAMERATES = [None, *range(1, 241)]


def _force_comfy_cpu_if_needed():
    # utils.dataset imports ComfyUI, which asks for a CUDA device at import time unless told
    # to use the CPU. This tool never touches a GPU, so let it run on a CPU-only machine too.
    import torch
    if torch.cuda.is_available():
        return
    try:
        import comfy.cli_args
    except ImportError:
        return
    comfy.cli_args.args.cpu = True


_START = time.monotonic()


def log(message):
    """Progress on stderr, as it happens: on network storage a step can take minutes."""
    elapsed = time.monotonic() - _START
    print(f'[{int(elapsed) // 60:02d}:{elapsed % 60:04.1f}] {message}', file=sys.stderr, flush=True)


class Plan:
    def __init__(self, path):
        self.path = path
        self.status = 'error'   # 'ready', 'skip' or 'error'
        self.messages = []
        self.target = None
        self.donor_snapshot = None
        self.rows = 0
        self.legacy = False          # reuse_metadata_cache = false: copy metadata/, not a snapshot
        self.legacy_entries = []     # names in the donor's metadata/ to copy
        self.replace_existing = False

    def note(self, message):
        self.messages.append(message)
        log(f'{self.path}: {message}')

    def __str__(self):
        head = {'ready': 'READY', 'skip': 'SKIP ', 'error': 'ERROR'}[self.status]
        return '\n'.join([f'[{head}] {self.path}', *[f'        {m}' for m in self.messages]])


def _resolve_active(geometry_dir):
    snapshot_dir = geometry_dir
    active = geometry_dir / 'active_snapshot.json'
    if active.exists():
        snapshot_dir = geometry_dir / json.loads(active.read_text())['directory']
        if not snapshot_dir.resolve().is_relative_to(geometry_dir.resolve()):
            raise ValueError(f'{active} points outside its cache root')
    return snapshot_dir


def find_donor_snapshot(target, donor, donor_framerates, round_to_multiple):
    """Every source_metadata of `donor` whose geometry equals the target's but for framerate."""
    from datasets.fingerprint import Hasher
    profiles = target.path / 'cache' / donor / 'profiles'
    log(f'{target.path}: looking for the {donor} snapshot under {profiles} '
        f'({len(donor_framerates)} candidate framerate(s))')
    found = []
    for framerate in donor_framerates:
        geometry_dir = profiles / Hasher.hash(target.snapshot_geometry(
            framerate=framerate, round_to_multiple=round_to_multiple))
        if not geometry_dir.is_dir():
            continue
        source = _resolve_active(geometry_dir) / 'source_metadata'
        if source.is_dir():
            found.append((framerate, source))
    return found


def _bucket_of(target, width, height):
    import numpy as np
    log_ar = np.log(width / height)
    if target.use_size_buckets:
        bucket = target._find_closest_size_bucket(log_ar, 1, False)
        return None if bucket is None else [int(x) for x in bucket]
    bucket = target._find_closest_ar_bucket(log_ar, 1, False)
    return None if bucket is None else [float(bucket[0]), int(bucket[1])]


def missing_files(snapshot, threads=32, batch_size=10000):
    """Every recorded image (or its tar) and mask that no longer exists. Threads because on
    network storage each existence check is a round trip."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def paths(rows):
        for spec, mask in zip(rows['image_spec'], rows['mask_file']):
            yield spec[0] or spec[1]
            if mask:
                yield mask

    gone = []
    with ThreadPoolExecutor(threads) as pool:
        for rows in tqdm(snapshot.select_columns(['image_spec', 'mask_file']).iter(batch_size=batch_size),
                         total=(len(snapshot) + batch_size - 1) // batch_size, desc='Checking files'):
            batch = list(dict.fromkeys(paths(rows)))
            gone.extend(p for p, ok in zip(batch, pool.map(os.path.exists, batch)) if not ok)
    return gone


def check_snapshot(plan, snapshot, sample, header_sample, seed=0, check_all=False, threads=32,
                   required=REQUIRED_COLUMNS):
    """Refuse a snapshot that is malformed, holds videos, or no longer matches the files."""
    import pyarrow.compute as pc
    from PIL import Image
    target = plan.target
    missing = required - set(snapshot.column_names)
    if missing:
        plan.note(f'donor snapshot lacks columns {sorted(missing)}; it was written by a different version.')
        return False
    donor_controls = sorted(c for c in snapshot.column_names if c.startswith('control_file'))
    if donor_controls != sorted(target.control_columns()):
        plan.note('donor snapshot and this dataset disagree about control_path.')
        return False
    plan.rows = len(snapshot)
    if plan.rows == 0:
        plan.note('donor snapshot is empty.')
        return False
    if pc.any(snapshot.data.column('is_video')).as_py():
        plan.note('donor snapshot contains videos; only image datasets can be inherited.')
        return False

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    indices = random.Random(seed).sample(range(plan.rows), min(plan.rows, sample))
    rows = list(snapshot.select(indices))
    log(f'{plan.path}: checking {len(rows)} sampled files still exist (and up to {header_sample} image sizes)')

    def present(row):
        tar_file, image_file = row['image_spec'][0], row['image_spec'][1]
        location = Path(tar_file) if tar_file else Path(image_file)
        if not location.exists():
            return str(location)
        if row['mask_file'] and not Path(row['mask_file']).exists():
            return row['mask_file']
        return None

    with ThreadPoolExecutor(threads) as pool:
        absent = list(tqdm(pool.map(present, rows), total=len(rows), desc='Sampled files'))
    gone, moved = [], []
    headers_checked = 0
    for row, missing_path in zip(rows, absent):
        tar_file, image_file = row['image_spec'][0], row['image_spec'][1]
        if missing_path is not None:
            gone.append(missing_path)
            continue
        if tar_file or headers_checked >= header_sample:
            continue
        headers_checked += 1
        try:
            with Image.open(image_file) as image:
                width, height = image.size
        except Exception:
            moved.append(image_file)
            continue
        recorded = row['size_bucket'] if target.use_size_buckets else row['ar_bucket']
        recorded = None if recorded is None else (
            [int(x) for x in recorded] if target.use_size_buckets else [float(recorded[0]), int(recorded[1])])
        if _bucket_of(target, width, height) != recorded:
            moved.append(image_file)
    plan.note(f'{plan.rows} images in the donor snapshot. Sampled {len(indices)}: {len(gone)} missing; '
              f'{headers_checked} image sizes compared to their recorded buckets: {len(moved)} differ.')
    if check_all and not gone:
        gone = missing_files(snapshot, threads=threads)
        plan.note(f'--check-all: {len(gone)} of {plan.rows} recorded files are missing.')
    if gone:
        plan.note(f'{len(gone)} sampled files no longer exist, e.g. {gone[:3]}. The dataset changed '
                  'since the donor scanned it: let the target model scan it instead.')
        return False
    if moved:
        plan.note(f'{len(moved)} sampled images no longer match their recorded bucket, e.g. {moved[:3]}. '
                  'Images were replaced since the donor scanned them: let the target model scan instead.')
        return False
    return True


def plan_directory(directory_config, dataset_config, donor, target_model, round_to_multiple,
                   donor_framerate=None, force=False, sample=2000, header_sample=200, check_all=False,
                   threads=32, caches_text_embeddings=False):
    from utils.dataset import DirectoryDataset
    log(f"{directory_config.get('path')}: reading the directory's dataset settings")
    plan = Plan(directory_config.get('path'))
    target = DirectoryDataset(copy.deepcopy(directory_config), copy.deepcopy(dataset_config), target_model,
                              framerate=None, round_to_multiple=round_to_multiple,
                              skip_dataset_validation=True, caches_text_embeddings=caches_text_embeddings)
    plan.target = target
    if target.frame_buckets.tolist() != [1]:
        plan.note(f'frame_buckets = {target.frame_buckets.tolist()}: only image datasets can be inherited.')
        return plan
    if not target.reuse_metadata_cache:
        return plan_legacy_directory(plan, donor, target_model, force, sample, header_sample, check_all, threads)
    if target.snapshot_path.exists() and not force:
        plan.note(f'{target_model} already has a snapshot at {target.snapshot_path}. Pass --force to replace '
                  'it with a new generation.')
        plan.status = 'skip'
        return plan

    framerates = CANDIDATE_FRAMERATES if donor_framerate is None else [donor_framerate]
    found = find_donor_snapshot(target, donor, framerates, round_to_multiple)
    if not found:
        donor_dir = target.path / 'cache' / donor
        if not donor_dir.is_dir():
            plan.note(f'no {donor} cache at {donor_dir}.')
        elif not (donor_dir / 'profiles').is_dir():
            plan.note(f'{donor_dir} has no profiles/ snapshots: it was built with reuse_metadata_cache = false '
                      'or by an older version, so there is nothing to inherit.')
        else:
            plan.note(f'no {donor} snapshot with the same bucketing settings under {donor_dir / "profiles"}. '
                      'resolutions/size_buckets, ar buckets, frame_buckets, mask/control paths and '
                      'shuffle_metadata must all be the same as when the donor was cached.')
        return plan
    # Several generations would only exist if the donor was cached under more than one
    # framerate with otherwise identical settings. The newest scan is the one to trust.
    framerate, source = max(found, key=lambda item: item[1].stat().st_mtime)
    plan.donor_snapshot = source
    plan.note(f'donor snapshot: {source} (donor framerate {framerate}).')

    import datasets
    log(f'{plan.path}: loading the donor snapshot')
    if not check_snapshot(plan, datasets.load_from_disk(str(source)), sample, header_sample,
                          check_all=check_all, threads=threads):
        return plan
    plan.note(f'will write {target.geometry_dir}')
    plan.status = 'ready'
    return plan


def _grouped_name(key, suffix):
    from utils.dataset import bucket_suffix
    return f'grouped_metadata_{bucket_suffix(key)}{suffix}'


def _legacy_complete(metadata_dir, suffix):
    keys_file = metadata_dir / f'grouping_keys{suffix}.json'
    if not keys_file.is_file():
        return False
    keys = json.loads(keys_file.read_text())
    return all((metadata_dir / _grouped_name(key, suffix)).is_dir() for key in keys)


def _bucket_mismatch(target, keys):
    """Grouping keys the target's bucketing would never produce, or None when all fit."""
    import numpy as np
    bad = []
    for key in keys:
        if target.use_size_buckets:
            ok = len(key) == 3 and any(list(key) == [int(x) for x in b] for b in target.size_buckets.tolist())
        else:
            ok = (len(key) == 2 and int(key[1]) in target.frame_buckets.tolist()
                  and bool(np.isclose(target.ars, float(key[0]), rtol=0, atol=1e-6).any()))
        if not ok:
            bad.append(key)
    return bad or None


def plan_legacy_directory(plan, donor, target_model, force, sample, header_sample, check_all, threads):
    """reuse_metadata_cache = false: plan copying the donor's legacy metadata/ into the target's."""
    import re
    import datasets
    from utils.dataset import LEGACY_CAPTION_SETTINGS
    target = plan.target
    plan.legacy = True
    suffix = target.caption_cache_suffix
    target_metadata = target.cache_dir / 'metadata'
    if _legacy_complete(target_metadata, suffix) and not force:
        plan.note(f'{target_model} already has legacy metadata at {target_metadata}. Pass --force to replace it '
                  '(the old one is kept as metadata.replaced_<time>).')
        plan.status = 'skip'
        return plan
    if target_metadata.is_dir() and any(target_metadata.iterdir()):
        if not force:
            plan.note(f'{target_metadata} holds metadata from an unfinished or different scan. Pass --force to '
                      'replace it (kept as metadata.replaced_<time>), or let training rebuild it.')
            return plan
        plan.replace_existing = True

    relative = target.cache_dir.relative_to(target.path / 'cache' / target_model)
    donor_metadata = target.path / 'cache' / donor / relative / 'metadata'
    log(f'{target.path}: looking for {donor} legacy metadata in {donor_metadata}')
    keys_file = donor_metadata / f'grouping_keys{suffix}.json'
    if not keys_file.is_file():
        others = sorted(p.name for p in donor_metadata.glob('grouping_keys*.json')) if donor_metadata.is_dir() else []
        if others:
            plan.note(f'{donor_metadata} has {others} but not {keys_file.name}: the donor cached captions under '
                      'other caption settings (prefix, shuffle, dropout, multiline, non-Latin filter, or caption '
                      'augmentation, which follows cache_text_embeddings: see --cache-text-embeddings).')
        elif (target.path / 'cache' / donor / 'profiles').is_dir():
            plan.note(f'{donor} was cached with reuse_metadata_cache = true (profiles/), not legacy metadata: '
                      'nothing to copy for a reuse_metadata_cache = false target.')
        else:
            plan.note(f'no {donor} legacy metadata at {donor_metadata}.')
        return plan
    keys = json.loads(keys_file.read_text())
    missing = [key for key in keys if not (donor_metadata / _grouped_name(key, suffix)).is_dir()]
    if missing:
        plan.note(f'{donor_metadata}: grouped metadata missing for buckets {missing[:3]}; the donor scan did not '
                  'finish.')
        return plan
    bad = _bucket_mismatch(target, keys)
    if bad:
        plan.note(f'donor buckets {bad[:3]} are not among this dataset\'s buckets: the bucketing settings changed '
                  'since the donor cached it.')
        return plan

    settings_file = donor_metadata / f'caption_settings{suffix}.json'
    if not settings_file.is_file():
        plan.note(f'{settings_file.name} is missing: the donor cache predates caption-settings tracking, so it '
                  'cannot be shown to hold captions processed like the target\'s. Let training rebuild it.')
        return plan
    recorded = json.loads(settings_file.read_text())
    current = target._legacy_caption_settings()
    changed = {k: (recorded.get(k, LEGACY_CAPTION_SETTINGS.get(k)), v) for k, v in current.items()
               if recorded.get(k, LEGACY_CAPTION_SETTINGS.get(k)) != v}
    if changed:
        plan.note('the donor captions were processed with other settings: '
                  + ', '.join(f'{k} {was!r} (donor) vs {now!r}' for k, (was, now) in sorted(changed.items())))
        return plan

    log(f'{plan.path}: loading the donor\'s grouped metadata ({len(keys)} buckets)')
    grouped = datasets.concatenate_datasets(
        [datasets.load_from_disk(str(donor_metadata / _grouped_name(key, suffix))) for key in keys])
    masks = {m for m in grouped.select_columns(['mask_file'])['mask_file'] if m}
    expected = target.mask_path is not None or target.default_mask_file is not None
    foreign = [m for m in masks if Path(m).parent != target.mask_path
               and (target.default_mask_file is None or Path(m) != target.default_mask_file)]
    if bool(masks) != expected or foreign:
        plan.note('donor metadata and this dataset disagree about mask_path / default_mask_file'
                  + (f', e.g. {sorted(foreign)[:2]}' if foreign else '') + '.')
        return plan
    if not check_snapshot(plan, grouped, sample, header_sample, check_all=check_all, threads=threads,
                          required=LEGACY_COLUMNS):
        return plan

    pattern = re.compile(rf'^(metadata_intermediate|metadata_with_captions\.arrow|'
                         rf'metadata{re.escape(suffix)}(_\d+_of_\d+)?\.arrow|'
                         rf'{re.escape(keys_file.name)}|{re.escape(settings_file.name)})$')
    grouped_names = {_grouped_name(key, suffix) for key in keys}
    plan.legacy_entries = sorted(p.name for p in donor_metadata.iterdir()
                                 if p.name in grouped_names or pattern.match(p.name))
    plan.donor_snapshot = donor_metadata
    plan.note(f'will write {target_metadata} ({len(plan.legacy_entries)} entries from {donor_metadata})')
    plan.status = 'ready'
    return plan


def apply_legacy_plan(plan, donor):
    """Publish the donor's legacy metadata/ as the target's, in one rename. Returns the path, or
    None when the target's metadata appeared meanwhile (another pod, or a training run)."""
    target = plan.target
    target_metadata = target.cache_dir / 'metadata'
    target.cache_dir.mkdir(parents=True, exist_ok=True)
    staging = target.cache_dir / ('.metadata.' + uuid.uuid4().hex)
    log(f'{plan.path}: publishing legacy metadata to {target_metadata} (hard links, or a copy across filesystems)')
    try:
        staging.mkdir()
        for name in plan.legacy_entries:
            source = plan.donor_snapshot / name
            if source.is_dir():
                shutil.copytree(source, staging / name, copy_function=_link_or_copy)
            else:
                _link_or_copy(source, staging / name)
        (staging / 'inherited_from.json').write_text(json.dumps({
            'donor_model': donor, 'donor_metadata': str(plan.donor_snapshot), 'rows': plan.rows,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
        }, indent=2))
        if target_metadata.exists():
            if plan.replace_existing or _legacy_complete(target_metadata, target.caption_cache_suffix):
                stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                os.replace(target_metadata, target.cache_dir / f'metadata.replaced_{stamp}')
            elif not any(target_metadata.iterdir()):
                target_metadata.rmdir()
            else:
                return None
        try:
            os.replace(staging, target_metadata)
        except OSError:
            if target_metadata.exists():
                return None
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target_metadata


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_active(geometry_dir, snapshot_dir):
    # A unique temporary name: two pods publishing at once must not rename each other's file.
    active = geometry_dir / 'active_snapshot.json'
    temporary = geometry_dir / f'.active_snapshot.{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps({'directory': os.path.relpath(snapshot_dir, geometry_dir)}))
    os.replace(temporary, active)


def apply_plan(plan, donor):
    """Publish the donor snapshot as the target's active source metadata. Returns the published
    path, or None when another process (another pod running this tool, or the target model's
    own first scan) published a snapshot there first."""
    target = plan.target
    geometry_dir = target.geometry_dir
    if target.snapshot_path.exists():
        # --force: a new generation, as --regenerate_cache would publish. The old one stays on
        # disk because profiles built from it may still be in use by a running job.
        snapshot_dir = geometry_dir / ('inherited_' + uuid.uuid4().hex)
    else:
        # Exactly where the target's own first scan would have put it.
        snapshot_dir = geometry_dir
    snapshot_path = snapshot_dir / 'source_metadata'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    staging = snapshot_dir / ('.source_metadata.' + uuid.uuid4().hex)
    log(f'{plan.path}: publishing the snapshot to {snapshot_path} (hard links, or a copy across filesystems)')
    try:
        shutil.copytree(plan.donor_snapshot, staging, copy_function=_link_or_copy)
        if snapshot_path.exists():
            return None
        try:
            os.replace(staging, snapshot_path)
        except OSError:
            # Renaming onto a directory someone else just filled fails rather than merging.
            if snapshot_path.exists():
                return None
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    (snapshot_dir / 'inherited_from.json').write_text(json.dumps({
        'donor_model': donor, 'donor_snapshot': str(plan.donor_snapshot), 'rows': plan.rows,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }, indent=2))
    _write_active(geometry_dir, snapshot_dir)
    return snapshot_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='The dataset.toml both models train on.')
    parser.add_argument('--from', dest='donor', default='anima',
                        help="Cache directory name of the model that already scanned the dataset (default 'anima', "
                             'which Anima and Anima refiner share).')
    parser.add_argument('--to', dest='target', default='iris', help="Model to give the snapshot to (default 'iris').")
    parser.add_argument('--round-to-multiple', type=int, default=16,
                        help="Target model's pixel rounding multiple (Iris: its patch size, 16). Must equal the donor's.")
    parser.add_argument('--from-framerate', type=int, default=None,
                        help='Donor model framerate (Anima: 16). Found automatically when omitted.')
    parser.add_argument('--sample', type=int, default=2000, help='Recorded files to check still exist (default 2000).')
    parser.add_argument('--check-all', action='store_true',
                        help='Check that every recorded file still exists, not just a sample. Recommended for '
                             'pixel-space models (Iris), which open the images during training.')
    parser.add_argument('--threads', type=int, default=32,
                        help='Parallel existence checks for --check-all (default 32). Raise it on network '
                             'storage (NFS, CephFS, EFS), where each check is a round trip.')
    parser.add_argument('--dry-run', action='store_true', help='Check and report; no metadata is written (empty cache directories may be created).')
    parser.add_argument('--force', action='store_true',
                        help='Replace an existing target snapshot by publishing a new generation (legacy '
                             'metadata: the old metadata/ is kept as metadata.replaced_<time>).')
    parser.add_argument('--cache-text-embeddings', action='store_true',
                        help="The target's training config sets cache_text_embeddings = true (Iris default: false). "
                             'Only matters with reuse_metadata_cache = false.')
    args = parser.parse_args(argv)

    if args.donor == args.target:
        parser.error('--from and --to name the same cache directory')
    if not Path(args.dataset).is_file():
        parser.error(f'--dataset {args.dataset!r}: no such file; pass the dataset .toml (the file, not its directory)')
    with open(args.dataset) as f:
        dataset_config = toml.load(f)

    log('importing torch, datasets and ComfyUI (can take a minute on network storage)')
    _force_comfy_cpu_if_needed()
    import utils.dataset  # noqa: F401
    directories = dataset_config['directory']
    plans = []
    for number, d in enumerate(directories, start=1):
        log(f"directory {number}/{len(directories)}: {d.get('path')}")
        plans.append(plan_directory(d, dataset_config, args.donor, args.target, args.round_to_multiple,
                                    args.from_framerate, args.force, args.sample, check_all=args.check_all,
                                    threads=args.threads, caches_text_embeddings=args.cache_text_embeddings))
    print()
    for plan in plans:
        print(plan)
    ready = [p for p in plans if p.status == 'ready']
    errors = [p for p in plans if p.status == 'error']
    if args.dry_run:
        print(f'\nDry run: {len(ready)} ready, {len(errors)} refused, {len(plans) - len(ready) - len(errors)} skipped.')
        return 1 if errors else 0
    for plan in ready:
        path = (apply_legacy_plan if plan.legacy else apply_plan)(plan, args.donor)
        if path is None:
            print(f'[SKIP ] {plan.path}\n        another process published {args.target} metadata '
                  'while this one was copying; keeping it.')
        else:
            print(f'Wrote {path}')
    print(f'\n{len(ready)} inherited, {len(errors)} refused, {len(plans) - len(ready) - len(errors)} skipped.')
    if errors:
        print('Refused directories are scanned normally on the first training run.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
