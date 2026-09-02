"""Read and write a flat caption corpus: every caption of a dataset in one file.

Distillation trains only the text frontend, so it needs captions and nothing else. Pointing it
at a dataset.toml works, but that walks every image (and opens every tar) purely to discover
which captions exist -- millions of stat() calls for text that fits in a few hundred MB. A
corpus file is that walk, done once.

Three formats, chosen by extension:

  .jsonl  one JSON object per line: {"caption": ..., "count": 1}
  .csv    the same fields as columns
  .txt    one caption per line, with newline and backslash escaped so a caption containing a
          line break survives the round trip

JSONL is the one to prefer: it streams, it needs no quoting rules, and it is the only format
where a corrupt line is obviously a corrupt line. TXT is here because it is the easiest thing
to eyeball and hand-edit, and CSV because spreadsheets exist.

Captions are stored exactly as the dataset has them: tag marker included, no caption_prefix, no
shuffling. Those are training-time concerns -- see utils.captions.preprocess_caption -- so that
one corpus can serve runs configured differently, and so the marker is still there to say which
captions are tag lists. The prefix in particular must not be baked in: training builds a caption
as caption_prefix + augment(strip_marker(raw)), so a stored "anime, Special: red, blue" no
longer starts with the marker and would train the marker as if it were a tag.

`num_repeats` is deliberately not a field. It is a per-directory oversampling knob whose
semantics are int(len(directory_captions) * num_repeats) over a whole directory; applied per
caption it would floor every value below 1 to zero and delete captions outright. Use the
exporter's --apply-num-repeats, which expands it with the dataset's own semantics.
"""

import csv
import json
import sys
from pathlib import Path

FORMATS = ('jsonl', 'csv', 'txt')

# csv fields can be large: a long natural-language caption is nowhere near the default 128 KiB
# limit, but a pathological one-line file would abort the whole read with an opaque error.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def format_for(path, explicit=None):
    if explicit:
        if explicit not in FORMATS:
            raise ValueError(f'Unknown corpus format {explicit!r}. Choose one of {FORMATS}.')
        return explicit
    suffix = Path(path).suffix.lower().lstrip('.')
    if suffix in ('jsonl', 'ndjson'):
        return 'jsonl'
    if suffix == 'csv':
        return 'csv'
    if suffix == 'txt':
        return 'txt'
    raise ValueError(
        f'Cannot infer a corpus format from {path!r}. Use a .jsonl, .csv or .txt extension, '
        f'or pass the format explicitly.'
    )


def _escape(caption):
    return caption.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')


def _unescape(line):
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == '\\' and i + 1 < len(line):
            nxt = line[i + 1]
            if nxt == 'n':
                out.append('\n')
                i += 2
                continue
            if nxt == 'r':
                out.append('\r')
                i += 2
                continue
            if nxt == '\\':
                out.append('\\')
                i += 2
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def write_corpus(path, entries, fmt=None):
    """Write entries, each a dict with at least 'caption'. Returns the number written."""
    path = Path(path)
    fmt = format_for(path, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, 'w', encoding='utf-8', newline='') as f:
        if fmt == 'csv':
            writer = csv.DictWriter(f, fieldnames=['caption', 'count'])
            writer.writeheader()
        for entry in entries:
            caption = entry['caption']
            if fmt == 'jsonl':
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            elif fmt == 'csv':
                writer.writerow({'caption': caption, 'count': entry.get('count', 1)})
            else:
                # TXT keeps only the caption. A count has nowhere to live in a
                # one-caption-per-line file, so it is applied by duplicating lines instead.
                f.write(_escape(caption) + '\n')
            n += 1
    return n


def read_corpus(path, fmt=None, apply_count=True):
    """Read a corpus into a flat list of caption strings.

    `count` -- how many times a deduplicated caption occurred -- is expanded into repeated list
    entries, so the caller samples from the same distribution whether or not the corpus was
    deduplicated.
    """
    path = Path(path)
    fmt = format_for(path, fmt)
    captions = []
    location = [0]

    def add(caption, count):
        if not caption:
            return
        reps = 1
        if apply_count:
            try:
                reps = int(count)
            except (TypeError, ValueError):
                raise RuntimeError(
                    f'{path}:{location[0]} has a non-integer count {count!r}'
                ) from None
            if reps < 1:
                raise RuntimeError(
                    f'{path}:{location[0]} has count {reps}, which would silently drop the '
                    f'caption. Remove the record instead.'
                )
        captions.extend([caption] * reps)

    # utf-8-sig: a corpus file is an interchange format people hand-edit, and an editor's
    # BOM would otherwise land inside the first caption.
    with open(path, encoding='utf-8-sig', newline='') as f:
        if fmt == 'jsonl':
            for lineno, line in enumerate(f, 1):
                location[0] = lineno
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f'{path}:{lineno} is not valid JSON: {e}') from e
                if isinstance(record, str):
                    add(record, 1)
                    continue
                if 'caption' not in record:
                    raise RuntimeError(f"{path}:{lineno} has no 'caption' field")
                add(record['caption'], record.get('count', 1))
        elif fmt == 'csv':
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'caption' not in reader.fieldnames:
                raise RuntimeError(f"{path} has no 'caption' column")
            for lineno, record in enumerate(reader, 2):  # 1 is the header
                location[0] = lineno
                add(record['caption'] or '', record.get('count') or 1)
        else:
            for lineno, line in enumerate(f, 1):
                location[0] = lineno
                add(_unescape(line.rstrip('\n').rstrip('\r')), 1)

    return captions
