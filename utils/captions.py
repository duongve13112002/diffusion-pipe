"""Caption resolution and augmentation, with no heavy dependencies.

Split out of utils/dataset.py so that caption-only tools -- tools/export_caption_corpus.py and
the distillation stage, which train the text frontend and never touch a VAE -- can read a
dataset's captions without importing torch, DeepSpeed and ComfyUI. utils/dataset.py re-exports
every name defined here, so existing callers are unaffected.
"""

import json
import logging
import random
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

CAPTIONS_JSON_FILE = 'captions.json'


def split_tag_prefix(caption: str, prefix_tag_caption: str = '') -> tuple[str, bool]:
    """Split a caption into (body, is_tag_caption).

    `prefix_tag_caption` marks which captions are tag lists rather than natural language, so
    that tag shuffling and tag dropout only touch the ones where a comma actually separates
    independent tags. The marker is stripped: it is a dataset annotation, not something the
    model should ever be trained on. An empty marker (the default) means the dataset is not
    annotated at all, so every caption is treated as tags -- which is the behaviour this repo
    had before the marker existed.
    """
    if not prefix_tag_caption:
        return caption, True
    if caption.startswith(prefix_tag_caption):
        return caption[len(prefix_tag_caption):], True
    return caption, False


def drop_tags(tags: list[str], tag_dropout_rate: float, rng=random) -> list[str]:
    """Drop each tag independently with probability `tag_dropout_rate`.

    At least one tag always survives. An all-dropped caption would be the empty string, which
    is the *unconditional* embedding -- the trainer already produces those deliberately via
    UNCOND_FRACTION, and silently minting more of them would change the conditioning ratio
    behind the user's back. With a handful of tags and a high rate that is not a rare event.
    """
    if tag_dropout_rate <= 0 or len(tags) <= 1:
        return tags
    kept = [t for t in tags if rng.random() >= tag_dropout_rate]
    if not kept:
        kept = [tags[rng.randrange(len(tags))]]
    return kept


def preprocess_caption(
    caption: str,
    delimiter: str = ', ',
    caption_prefix: str = '',
    prefix_tag_caption: str = '',
    shuffle: bool = False,
    tag_dropout_rate: float = 0.0,
    rng=random,
) -> str:
    """Strip the tag marker, optionally shuffle and drop tags, then apply caption_prefix.

    Order matters: the marker comes off first so it is never shuffled into the middle of the
    tag list, and caption_prefix goes on last so it stays pinned to the front the way every
    other model in this repo expects.
    """
    body, is_tag = split_tag_prefix(caption, prefix_tag_caption)
    if is_tag and (shuffle or tag_dropout_rate > 0):
        tags = body.split(delimiter)
        if shuffle:
            rng.shuffle(tags)
        tags = drop_tags(tags, tag_dropout_rate, rng)
        body = delimiter.join(tags)
    return caption_prefix + body


def shuffle_captions(
    captions: list[str],
    count: int = 0,
    delimiter: str = ', ',
    caption_prefix: str = '',
    prefix_tag_caption: str = '',
    tag_dropout_rate: float = 0.0,
) -> list[str]:
    """Expand captions into the variants that get embedded and cached.

    `count` variants per caption when shuffling, otherwise one. Tag dropout applies per
    variant, so it is baked into the cache exactly like shuffling already was -- changing
    either setting needs --regenerate_cache to take effect, which is this repo's existing
    behaviour for cache_shuffle_num.
    """
    if count == 0 and tag_dropout_rate <= 0 and not prefix_tag_caption:
        return [caption_prefix + c for c in captions]

    variants = max(count, 1)
    return [
        preprocess_caption(
            caption,
            delimiter=delimiter,
            caption_prefix=caption_prefix,
            prefix_tag_caption=prefix_tag_caption,
            shuffle=(count > 0),
            tag_dropout_rate=tag_dropout_rate,
        )
        for caption in captions
        for _ in range(variants)
    ]


def read_caption_file(path: Path, multiline_captions: bool = False) -> list[str]:
    """Read a sidecar .txt caption file.

    Historically the whole file is one caption, newlines included. `multiline_captions` opts a
    dataset into treating each non-empty line as a separate caption, matching what
    captions.json already allows. It is opt-in because flipping it changes the number of
    training samples for every existing dataset whose .txt files happen to wrap.
    """
    text = path.read_text(encoding='utf-8')
    if not multiline_captions:
        return [text.strip()]
    return [line for line in (l.strip() for l in text.splitlines()) if line]


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


def dedup_and_sort(values):
    values = set(round(x, ROUND_DECIMAL_DIGITS) for x in values)
    values = list(values)
    values.sort()
    return np.array(values)


def seed_from_hash(item):
    return int(hashlib.md5(str.encode(str(item))).hexdigest(), 16) % int(1e9)


# Extensions DirectoryDataset skips when enumerating media files.
NON_MEDIA_SUFFIXES = ('.txt', '.npz', '.json', '.parquet', '.bak', '.db')


def enumerate_captions(dataset_config, apply_num_repeats=False, apply_shuffle=True):
    """Return every caption in a dataset config, without opening any media file.

    This mirrors the caption resolution DirectoryDataset does (captions.json first, then a
    matching .txt, then skip_empty_caption) and applies the same caption_prefix and tag
    shuffling, so callers see the caption distribution training will see. It exists for tools
    that need captions but not images -- tools/distill_refiner.py trains only the text frontend
    and never touches the VAE or the DiT's image path.

    As an accommodation for that use case, a directory holding only caption files with no
    media alongside them is accepted: DirectoryDataset would assert, but for a text-only tool
    the images are genuinely not needed.
    """
    captions = []
    for directory_config in dataset_config['directory']:
        def setting(key, default):
            return directory_config.get(key, dataset_config.get(key, default))

        path = Path(directory_config['path'])
        caption_prefix = setting('caption_prefix', '')
        shuffle_num = setting('cache_shuffle_num', 0)
        if setting('shuffle_tags', False) and shuffle_num == 0:
            shuffle_num = 1  # backwards compatibility, same as DirectoryDataset
        delimiter = setting('cache_shuffle_delimiter', ', ')
        skip_empty_caption = setting('skip_empty_caption', True)
        multiline_captions = setting('multiline_captions', False)
        prefix_tag_caption = setting('prefix_tag_caption', '')
        tag_dropout_rate = setting('tag_dropout_rate', 0.0)
        num_repeats = setting('num_repeats', 1) if apply_num_repeats else 1
        if not apply_shuffle:
            # Exporting a corpus: keep the captions as they are on disk so shuffling and
            # dropout stay runtime augmentations instead of being frozen into the file.
            shuffle_num = 0
            tag_dropout_rate = 0.0
            prefix_tag_caption = ''

        caption_data = None
        captions_json = path / CAPTIONS_JSON_FILE
        if captions_json.exists():
            with open(captions_json) as f:
                caption_data = json.load(f)

        files = sorted(path.glob('*'))
        # (tar_file_or_None, path_within_or_on_disk), mirroring DirectoryDataset's image_spec.
        # The distinction matters for captions.json lookups: DirectoryDataset keys a plain file
        # by basename but a tar member by its full path inside the archive.
        media_specs = []
        for file in files:
            if not file.is_file() or file.suffix in NON_MEDIA_SUFFIXES:
                continue
            if file.suffix == '.tar':
                with tarfile.TarFile(file) as tar_f:
                    media_specs.extend((file, Path(name)) for name in tar_f.getnames())
            else:
                media_specs.append((None, file))

        if not media_specs and caption_data is None:
            # Text-only directory: take the caption files themselves as the unit of work.
            media_specs = [(None, f) for f in files if f.is_file() and f.suffix == '.txt']

        directory_captions = []
        for tar_file, media_file in media_specs:
            item = None
            if caption_data is not None:
                key = str(media_file) if tar_file is not None else media_file.name
                item = caption_data.get(key, None)
                if item is None:
                    logger.warning(f'{key} has no entry in {CAPTIONS_JSON_FILE}')
                else:
                    assert isinstance(item, list), f'{CAPTIONS_JSON_FILE} must contain lists of captions'
            elif media_file.suffix == '.txt':
                item = read_caption_file(media_file, multiline_captions)
            else:
                # DirectoryDataset disables the .txt fallback for the WHOLE directory as soon as
                # a captions.json exists (`if has_captions_json or not os.path.exists(...)`).
                # Keeping the fallback here would feed distillation captions the diffusion
                # stages never see, which is the drift this helper exists to prevent.
                caption_file = media_file.with_suffix('.txt')
                if caption_file.exists():
                    item = read_caption_file(caption_file, multiline_captions)
            if not item:
                item = None
            if item is None:
                if skip_empty_caption:
                    logger.warning(f'Could not find caption for {media_file}. Skipping.')
                    continue
                item = ['']
            directory_captions.extend(shuffle_captions(
                item, shuffle_num, delimiter, caption_prefix, prefix_tag_caption, tag_dropout_rate,
            ))

        # num_repeats may be fractional -- SizeBucketDataset accepts any value > 0 and takes
        # int(len * num_repeats), so mirror that rather than assuming an integer.
        if directory_captions:
            total = int(len(directory_captions) * num_repeats)
            captions.extend(directory_captions[i % len(directory_captions)] for i in range(total))

    return captions
