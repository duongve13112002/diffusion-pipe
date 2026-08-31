"""Flatten every caption in a dataset.toml into a single corpus file.

    python tools/export_caption_corpus.py --dataset dataset.toml --output captions.jsonl

Distillation trains only the text frontend of the model, so it needs the captions and none of
the images. Reading them straight from a dataset.toml works and stays exactly in step with what
the diffusion stages see, but it walks every media file (and opens every tar) on every run. For
a few million images that is slow enough to be worth doing once.

The corpus is a faithful dump, not a preprocessed one: captions are written exactly as the
dataset holds them, tag marker and all, with no `caption_prefix` and no shuffling. Those are
training-time concerns, so one corpus serves runs configured differently and each epoch sees a
fresh variant.

The prefix in particular must not be baked in. Training builds a caption as
`caption_prefix + augment(strip_marker(raw))`, so a stored `"anime, Special: red, blue"` no
longer starts with the marker: the consumer would fail to strip it, silently skip augmentation,
and train the marker as a tag.

Output format follows the extension: .jsonl, .csv or .txt. See utils/caption_corpus.py.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.caption_corpus import FORMATS, format_for, write_corpus  # noqa: E402
from utils.captions import enumerate_captions  # noqa: E402


def build_entries(captions, dedupe=False):
    if not dedupe:
        return [{'caption': c} for c in captions]
    # Counter preserves first-seen order, so the corpus stays in dataset order rather than
    # being reshuffled into hash order.
    counts = Counter(captions)
    return [{'caption': c, 'count': n} for c, n in counts.items()]


def expand_for_txt(entries):
    """TXT has nowhere to put a count, so repeat the line instead of silently dropping it."""
    out = []
    for entry in entries:
        for _ in range(int(entry.get('count', 1))):
            out.append({'caption': entry['caption']})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='Path to the dataset.toml to read.')
    parser.add_argument('--output', required=True, help='Corpus file to write (.jsonl, .csv or .txt).')
    parser.add_argument('--format', choices=FORMATS, default=None, help='Override the format inferred from the extension.')
    parser.add_argument('--dedupe', action='store_true', help='Collapse identical captions, recording how many times each occurred.')
    parser.add_argument(
        '--apply-num-repeats', action='store_true',
        help="Bake each directory's num_repeats into the corpus by repeating captions. Off by "
             'default: it multiplies the file size. This is the only way to honour num_repeats '
             'in a corpus -- the file stores no per-caption repeat count, because the dataset '
             'applies num_repeats per directory, and flooring it per caption would delete '
             'every caption in a directory with num_repeats < 1.',
    )
    args = parser.parse_args()

    fmt = format_for(args.output, args.format)
    if args.format and fmt != format_for(args.output):
        print(
            f'WARNING: writing {fmt} content to {args.output}, whose extension says '
            f'{format_for(args.output)}. Readers infer the format from the extension, so this '
            f'file will not read back correctly unless the format is passed explicitly.'
        )

    dataset_config = toml.load(args.dataset)

    markers = set()
    captions = enumerate_captions(
        dataset_config,
        apply_num_repeats=args.apply_num_repeats,
        apply_shuffle=False,
        markers_seen=markers,
    )
    if not captions:
        raise SystemExit(f'No captions found in {args.dataset}.')

    entries = build_entries(captions, dedupe=args.dedupe)
    if fmt == 'txt' and args.dedupe:
        entries = expand_for_txt(entries)

    written = write_corpus(args.output, entries, fmt)

    unique = len({e['caption'] for e in entries})
    print(f'Read {len(captions)} captions ({unique} unique) from {args.dataset}')
    print(f'Wrote {written} {"records" if fmt != "txt" else "lines"} to {args.output} [{fmt}]')
    if args.dedupe and fmt != 'txt':
        print('Deduplicated: each record carries a count, which distillation expands back out '
              'so the sampling distribution is unchanged.')
    if markers:
        rendered = ', '.join(repr(m) for m in sorted(markers))
        print(
            f'\nThe dataset marks tag captions with {rendered}. Those markers are still in the '
            f'corpus, on purpose -- they say which captions are tag lists. Whatever reads this '
            f'file must strip them, or they will be trained as if they were tags. For '
            f'distillation, set under [distill]:\n'
            f'    prefix_tag_caption = {sorted(markers)!r}'
        )


if __name__ == '__main__':
    main()
