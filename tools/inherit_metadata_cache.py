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
import uuid
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'submodules' / 'ComfyUI'))

REQUIRED_COLUMNS = {'image_spec', 'mask_file', 'caption', 'ar_bucket', 'size_bucket', 'is_video',
                    '_caption_missing', '_caption_sidecar'}
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


class Plan:
    def __init__(self, path):
        self.path = path
        self.status = 'error'   # 'ready', 'skip' or 'error'
        self.messages = []
        self.target = None
        self.donor_snapshot = None
        self.rows = 0

    def note(self, message):
        self.messages.append(message)

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


def check_snapshot(plan, snapshot, sample, header_sample, seed=0, check_all=False):
    """Refuse a snapshot that is malformed, holds videos, or no longer matches the files."""
    import pyarrow.compute as pc
    from PIL import Image
    target = plan.target
    missing = REQUIRED_COLUMNS - set(snapshot.column_names)
    if missing:
        plan.note(f'donor snapshot lacks columns {sorted(missing)}; it was written by a different version.')
        return False
    if ('control_file' in snapshot.column_names) != (target.control_path is not None):
        plan.note('donor snapshot and this dataset disagree about control_path.')
        return False
    plan.rows = len(snapshot)
    if plan.rows == 0:
        plan.note('donor snapshot is empty.')
        return False
    if pc.any(snapshot.data.column('is_video')).as_py():
        plan.note('donor snapshot contains videos; only image datasets can be inherited.')
        return False

    indices = random.Random(seed).sample(range(plan.rows), min(plan.rows, sample))
    rows = snapshot.select(indices)
    gone, moved = [], []
    headers_checked = 0
    for row in rows:
        tar_file, image_file = row['image_spec'][0], row['image_spec'][1]
        location = Path(tar_file) if tar_file else Path(image_file)
        if not location.exists():
            gone.append(str(location))
            continue
        if row['mask_file'] and not Path(row['mask_file']).exists():
            gone.append(row['mask_file'])
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
        gone = missing_files(snapshot)
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
                   donor_framerate=None, force=False, sample=2000, header_sample=200, check_all=False):
    from utils.dataset import DirectoryDataset
    plan = Plan(directory_config.get('path'))
    target = DirectoryDataset(copy.deepcopy(directory_config), copy.deepcopy(dataset_config), target_model,
                              framerate=None, round_to_multiple=round_to_multiple,
                              skip_dataset_validation=True, caches_text_embeddings=False)
    plan.target = target
    if not target.reuse_metadata_cache:
        plan.note('reuse_metadata_cache = false for this directory, so no snapshot is used. Nothing to do.')
        plan.status = 'skip'
        return plan
    if target.frame_buckets.tolist() != [1]:
        plan.note(f'frame_buckets = {target.frame_buckets.tolist()}: only image datasets can be inherited.')
        return plan
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
    if not check_snapshot(plan, datasets.load_from_disk(str(source)), sample, header_sample,
                          check_all=check_all):
        return plan
    plan.note(f'will write {target.geometry_dir}')
    plan.status = 'ready'
    return plan


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def apply_plan(plan, donor):
    """Publish the donor snapshot as the target's active source metadata."""
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
    shutil.copytree(plan.donor_snapshot, staging, copy_function=_link_or_copy)
    os.replace(staging, snapshot_path)
    (snapshot_dir / 'inherited_from.json').write_text(json.dumps({
        'donor_model': donor, 'donor_snapshot': str(plan.donor_snapshot), 'rows': plan.rows,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }, indent=2))
    active = geometry_dir / 'active_snapshot.json'
    temporary = active.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'directory': os.path.relpath(snapshot_dir, geometry_dir)}))
    os.replace(temporary, active)
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
    parser.add_argument('--dry-run', action='store_true', help='Check and report; no metadata is written (empty cache directories may be created).')
    parser.add_argument('--force', action='store_true',
                        help='Replace an existing target snapshot by publishing a new generation.')
    args = parser.parse_args(argv)

    _force_comfy_cpu_if_needed()
    if args.donor == args.target:
        parser.error('--from and --to name the same cache directory')
    with open(args.dataset) as f:
        dataset_config = toml.load(f)

    plans = [plan_directory(d, dataset_config, args.donor, args.target, args.round_to_multiple,
                            args.from_framerate, args.force, args.sample, check_all=args.check_all)
             for d in dataset_config['directory']]
    for plan in plans:
        print(plan)
    ready = [p for p in plans if p.status == 'ready']
    errors = [p for p in plans if p.status == 'error']
    if args.dry_run:
        print(f'\nDry run: {len(ready)} ready, {len(errors)} refused, {len(plans) - len(ready) - len(errors)} skipped.')
        return 1 if errors else 0
    for plan in ready:
        path = apply_plan(plan, args.donor)
        print(f'Wrote {path}')
    print(f'\n{len(ready)} inherited, {len(errors)} refused, {len(plans) - len(ready) - len(errors)} skipped.')
    if errors:
        print('Refused directories are scanned normally on the first training run.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
