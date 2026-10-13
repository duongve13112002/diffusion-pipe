"""Write the prompts file for Iris-3B Turbo distillation from diffusion-pipe dataset configs.

    python tools/iris_turbo_prompts.py --dataset dataset.toml --output /pvc/turbo/prompts.txt

Iris's distillation trainer (tools/iris_turbo_distill.py, which runs Iris's own
iris3b.distill trainer) reads no images, only `prompts_file`: UTF-8 text, one prompt per line,
read in order and cycled. This builds that file from the captions a diffusion-pipe run trained
on, so the student is distilled on the distribution the fine-tuned teacher knows.

Each prompt is the caption as training fed it to the text encoder, minus augmentation:
caption_prefix applied, the prefix_tag_caption marker stripped, enable_remove_non_latin
applied, but no tag shuffling, no tag dropout and no caption dropout (Turbo learns the
unconditional branch from the teacher's CFG, not from empty prompts). captions.json,
multiline_captions, skip_empty_caption and require_non_latin_caption resolve exactly as in
training.

Because the file is line based, a caption that spans several lines is joined with spaces;
otherwise each of its lines would become a separate, truncated prompt. Empty captions,
captions that are only their caption_prefix and captions that cannot be written as UTF-8 (a
lone surrogate escape in a captions.json) are dropped and counted. Each directory's num_repeats
weights its captions (--ignore-num-repeats turns that off). Only num_repeats: training also
repeats a caption once per cache_shuffle_num variant and per resolution, and
caption_sampling = 'random_per_epoch' draws one caption per image, none of which this mirrors.
The result is shuffled with a fixed seed: the trainer reads the file in order, and dataset
order would hand each batch prompts from a single directory.

--extra-prompts adds the lines of plain text files (one prompt per line) as they are, for
example general prompts that keep the student close to the base model's range.

The dataset is walked once and the file is written to a temporary name beside the output and
renamed into place, so a pod killed mid-write on a network volume (PVC) never leaves a
truncated prompts file for the trainer to cycle.
"""

import argparse
import copy
import os
import random
import sys
import uuid
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.captions import enumerate_captions  # noqa: E402

# Settings that augment captions per sample. Distillation should see each caption once, as
# written, so they are neutralised at both the dataset and the directory level.
AUGMENTATION_OFF = {'cache_shuffle_num': 0, 'shuffle_tags': False, 'tag_dropout_rate': 0.0}

# How many prompts the released recipe reads: each student update and each fake-network update
# draws a fresh global batch (iris3b.distill.trainer.PromptStream).
RECIPES = {
    'stage 1 (turbo_stage1_6step.yaml)': (1600, 64, 1),
    'stage 2 (turbo_stage2_4step.yaml)': (500, 16, 1),
}


def flatten(text):
    """One line per prompt: join a multi-line caption with single spaces.

    str.splitlines is what Iris splits the file with, so this removes every separator it
    would split on (\\r, \\v, \\f, \\x1c-\\x1e, \\x85, \\u2028, \\u2029 as well as \\n).
    """
    return ' '.join(part.strip() for part in text.splitlines() if part.strip())


def dataset_prompts(dataset_config, apply_num_repeats=True, stats=None, progress=False):
    """The captions of one dataset config as training builds them, without augmentation.

    A caption that is nothing but its directory's caption_prefix (an empty caption file, or
    one emptied by enable_remove_non_latin) comes back as '', counted in stats['empty_prompt'].
    """
    stats = {} if stats is None else stats
    captions = []
    for directory in dataset_config.get('directory', []):
        config = copy.deepcopy(dataset_config)
        config.update(AUGMENTATION_OFF)
        config['directory'] = [{**copy.deepcopy(directory), **AUGMENTATION_OFF}]
        prefix = directory.get('caption_prefix', dataset_config.get('caption_prefix', ''))
        part = {}
        found = enumerate_captions(config, apply_num_repeats=apply_num_repeats, apply_shuffle=True,
                                   progress=progress, stats=part)
        for key, value in part.items():
            stats[key] = stats.get(key, 0) + value
        for caption in found:
            if prefix and caption.strip() == prefix.strip():
                stats['empty_prompt'] = stats.get('empty_prompt', 0) + 1
                caption = ''
            captions.append(caption)
    return captions


def read_extra_prompts(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_prompts(captions, dedupe=False, shuffle=True, seed=0):
    """(prompts, report): flattened, non-empty, optionally deduplicated, then shuffled."""
    report = {'joined': 0, 'empty': 0, 'duplicates': 0, 'unencodable': 0}
    prompts = []
    for caption in captions:
        line = flatten(caption)
        if not line:
            report['empty'] += 1
            continue
        try:
            line.encode('utf-8')
        except UnicodeEncodeError:
            report['unencodable'] += 1
            continue
        if line != caption.strip():
            report['joined'] += 1
        prompts.append(line)
    if dedupe:
        unique = list(dict.fromkeys(prompts))
        report['duplicates'] = len(prompts) - len(unique)
        prompts = unique
    if shuffle:
        random.Random(seed).shuffle(prompts)
    return prompts, report


def write_atomic(path, prompts):
    """Write to a temporary file in the same directory, fsync, then rename over `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # uuid, not the PID: in a container that is often 1 for every pod writing to the volume
    tmp = path.with_name(f'.{path.name}.tmp-{uuid.uuid4().hex}')
    try:
        with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            for prompt in prompts:
                f.write(prompt)
                f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', action='append', default=[],
                        help='A dataset.toml of the training run; repeat for several.')
    parser.add_argument('--extra-prompts', action='append', default=[],
                        help='A text file with one prompt per line to add as is; repeat for several.')
    parser.add_argument('--output', required=True, help='The prompts file to write (prompts_file of the distillation).')
    parser.add_argument('--ignore-num-repeats', action='store_true',
                        help="Count every caption once instead of weighting it by its directory's num_repeats.")
    parser.add_argument('--dedupe', action='store_true', help='Keep one copy of identical prompts.')
    parser.add_argument('--no-shuffle', action='store_true', help='Keep dataset order (not recommended, see above).')
    parser.add_argument('--seed', type=int, default=0, help='Shuffle seed (default 0).')
    parser.add_argument('--no-progress', action='store_true', help='No progress bar (for log files).')
    args = parser.parse_args(argv)
    if not args.dataset and not args.extra_prompts:
        parser.error('give at least one --dataset or --extra-prompts')

    captions = []
    for dataset in args.dataset:
        stats = {}
        found = dataset_prompts(toml.load(dataset), apply_num_repeats=not args.ignore_num_repeats,
                                stats=stats, progress=not args.no_progress)
        print(f"{dataset}: {len(found)} captions from {stats.get('resolved', 0)} media files"
              + (f", {stats['skipped']} without a caption skipped" if stats.get('skipped') else '')
              + (f", {stats['filtered']} excluded by require_non_latin_caption" if stats.get('filtered') else ''))
        captions.extend(found)
    for extra in args.extra_prompts:
        found = read_extra_prompts(extra)
        print(f'{extra}: {len(found)} prompts')
        captions.extend(found)

    prompts, report = build_prompts(captions, dedupe=args.dedupe, shuffle=not args.no_shuffle, seed=args.seed)
    if not prompts:
        raise SystemExit('No non-empty prompts; nothing written.')
    write_atomic(args.output, prompts)

    print(f'Wrote {len(prompts)} prompts ({len(set(prompts))} unique) to {args.output}')
    if report['joined']:
        print(f"  {report['joined']} multi-line captions joined into one line")
    if report['empty']:
        print(f"  {report['empty']} empty captions dropped")
    if report['unencodable']:
        print(f"  {report['unencodable']} captions with invalid characters (lone surrogates) dropped")
    if report['duplicates']:
        print(f"  {report['duplicates']} duplicates dropped (--dedupe)")
    for name, (steps, batch, fake_updates) in RECIPES.items():
        read = steps * batch * (1 + fake_updates)
        print(f'  the released {name} reads {read:,} prompts: {read / len(prompts):.2f} passes over this file')


if __name__ == '__main__':
    main()
