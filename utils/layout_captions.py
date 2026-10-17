"""Layout captions: a .txt caption that is a JSON object describing the image region by region.

MOSS-Image and Ideogram 4 are trained on such captions, with bounding boxes in the coordinates
of the image the model sees. A training image is resized and centre-cropped to its bucket
(ImageOps.fit in models/base.py), so a box written for the original image is wrong for the
bucket unless it is rewritten for that crop. This module does that rewrite, per bucket.

Detection is per caption: a caption is a layout caption when its whole text parses as a JSON
object. Everything else is a plain caption and keeps every caption setting. Layout captions are
never tag-shuffled, tag-dropped, split into lines or filtered line by line (those would break
the JSON), and caption_prefix goes into the format's description field instead of in front of
the text. Caption dropout still applies to them.

Formats (MossImagePipeline.layout_caption_format / Ideogram4Pipeline.layout_caption_format):

  moss_image  The prompt MOSS-Image's compiler produces (skills/moss-image-prompt-enhancer in OpenMOSS/MOSS-Image):
              {"summary": {"caption": ...}, "items": [{"bbox": [y1, x1, y2, x2], ...}],
               "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
              Boxes are integer pixels of the original image with the origin at its centre.
              The technical report (appendix C.3) trains the model on three box representations
              of such prompts, so a caption may also be written in the other ones:
              normalized_1000  boxes in 0..1000 of the image, origin at the top left (what the
                               report stores, and what Figure 6 shows: no bbox_coord_space or
                               bbox_format keys; "bbox_coord_space": "normalized_1000" and
                               "bbox_format": "yxyx" are accepted as well)
              Boxes under "items" (summary/items schema) and under "elements" (scene/element
              schema) are rewritten. The caption stays in the space it is written in.
  moss_image_centered
              The same captions, but a caption in normalized_1000 is converted to the centred
              pixel space of its bucket (what the model's USERGUIDE recommends: it generalizes
              better to resolutions not seen in training), with the bbox_coord_space and
              bbox_format keys the compiler writes. Captions already centred stay as they are.
              The rewritten caption keeps the caption's own JSON formatting.
  ideogram4   Ideogram 4's prompt (ComfyUI's BuildJsonPromptIdeogram):
              {"high_level_description": ..., "compositional_deconstruction":
               {"elements": [{"bbox": [ymin, xmin, ymax, xmax], ...}]}}
              Boxes are integers in 0..1000 of the image, origin at the top left. A zero-height
              or zero-width box (which ComfyUI's bounding box nodes can write) is allowed.

A rewritten caption keeps the caption's own JSON formatting: compact (what MOSS-Image's
compiler and json.dumps(..., separators=(",", ":"), ensure_ascii=False) write), ComfyUI's
indent=2, or another usual json.dumps style; anything else comes out compact. A caption that
needs no change is used exactly as written.

A box cut by the crop is clipped to it; an item whose box lies wholly outside the crop is
dropped. A box outside the original image is an error naming the file.
"""

import functools
import json
import math
import os
import tarfile
from collections import OrderedDict
from pathlib import Path

LAYOUT_FORMATS = ('moss_image', 'moss_image_centered', 'ideogram4')

# Bumped when the rewrite changes, so text caches built by an older rewrite are rebuilt.
LAYOUT_REWRITE_VERSION = 3
# Bumped when the metadata stage handles layout captions differently (it keeps them whole and
# out of the tag operations), so metadata cached by an older version is rebuilt.
LAYOUT_METADATA_VERSION = 1

# Pixel multiple the media preprocessing rounds a bucket to (PreprocessMediaFile's default,
# which both layout models use).
BUCKET_ROUNDING = 16


def parse_layout_caption(caption):
    """The caption's JSON object, or None for a plain caption."""
    if not isinstance(caption, str):
        return None
    text = caption.strip()
    if not (text.startswith('{') and text.endswith('}')):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def is_layout_caption(caption):
    return parse_layout_caption(caption) is not None


def fit_crop_box(src_width, src_height, dst_width, dst_height):
    """(left, top, width, height) of the source region ImageOps.fit resizes to dst, in source
    pixels. Pillow's own arithmetic (bleed 0, centering 0.5), so the boxes follow the pixels."""
    src_ratio = src_width / src_height
    dst_ratio = dst_width / dst_height
    if src_ratio == dst_ratio:
        crop_width, crop_height = src_width, src_height
    elif src_ratio >= dst_ratio:
        crop_width, crop_height = dst_ratio * src_height, src_height
    else:
        crop_width, crop_height = src_width, src_width / dst_ratio
    return (src_width - crop_width) * 0.5, (src_height - crop_height) * 0.5, crop_width, crop_height


def bucket_pixel_size(size_bucket):
    """(width, height) the media preprocessing resizes an image of this bucket to
    (PreprocessMediaFile: round_to_nearest_multiple of each side)."""
    width, height = size_bucket[0], size_bucket[1]
    return (int(round(width / BUCKET_ROUNDING) * BUCKET_ROUNDING),
            int(round(height / BUCKET_ROUNDING) * BUCKET_ROUNDING))


class _OpenTars:
    """Open tar archives with a name -> member index, a few at a time.

    tarfile finds a member by scanning the headers before it (TarFile.getmember is a linear
    search, and a fresh TarFile reads every header again), so looking members up one by one
    is quadratic in the archive size. Each archive here is indexed once. The handles belong to
    the process that opened them: a forked worker shares the file offset with its parent, so
    it opens its own.
    """

    def __init__(self, limit=8):
        self.limit = limit
        self.pid = None
        self.tars = OrderedDict()

    def member(self, tar_path, name):
        if self.pid != os.getpid():
            self.tars = OrderedDict()  # inherited handles stay with the parent
            self.pid = os.getpid()
        entry = self.tars.get(tar_path)
        if entry is None:
            tar = tarfile.TarFile(tar_path)
            entry = (tar, {info.name: info for info in tar.getmembers()})
            self.tars[tar_path] = entry
            while len(self.tars) > self.limit:
                self.tars.popitem(last=False)[1][0].close()
        else:
            self.tars.move_to_end(tar_path)
        tar, index = entry
        info = index.get(name)
        if info is None:
            raise KeyError(f'{name} is not in {tar_path}')
        return tar.extractfile(info)


_open_tars = _OpenTars()


@functools.lru_cache(maxsize=1 << 18)
def _media_size(tar_path, member):
    from PIL import Image
    if tar_path is None:
        with Image.open(member) as image:
            return image.size
    with _open_tars.member(tar_path, Path(member).as_posix()) as handle, Image.open(handle) as image:
        return image.size


def media_size(image_spec):
    """(width, height) of a dataset media file, from its header. image_spec is (tar or None, path).
    Remembered per process: every caption variant and bucket of an image asks again."""
    tar_path, member = image_spec
    return _media_size(None if tar_path is None else str(tar_path), str(member))


def _box(value, location):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in value)):
        raise ValueError(f'{location} must be four finite numbers, got {value!r}')
    y1, x1, y2, x2 = value
    if not (y1 <= y2 and x1 <= x2):
        raise ValueError(f'{location} must be [y1, x1, y2, x2] with y1 <= y2 and x1 <= x2, got {list(value)}')
    return value


def _fit_span(lo, hi, mapped, new_lo, new_hi, limit_lo, limit_hi):
    """The span [new_lo, new_hi] (the rewritten [lo, hi], rounded) clipped to [limit_lo,
    limit_hi], or None when the crop leaves nothing of it. A zero-size span stays when its
    position before rounding (mapped) is inside."""
    if hi == lo and not limit_lo <= mapped <= limit_hi:
        return None
    new_lo, new_hi = max(new_lo, limit_lo), min(new_hi, limit_hi)
    return (new_lo, new_hi) if new_lo < new_hi or hi == lo else None


class _Geometry:
    """Maps original-image pixels (top-left origin) to bucket pixels."""

    def __init__(self, src_size, dst_size):
        self.src_width, self.src_height = src_size
        self.dst_width, self.dst_height = dst_size
        self.left, self.top, crop_width, crop_height = fit_crop_box(
            self.src_width, self.src_height, self.dst_width, self.dst_height)
        self.scale_x = self.dst_width / crop_width
        self.scale_y = self.dst_height / crop_height

    def x(self, value):
        return (value - self.left) * self.scale_x

    def y(self, value):
        return (value - self.top) * self.scale_y


def _centered_span(lo_px, hi_px, to_bucket, bucket_center, bucket_size):
    """[lo_px, hi_px], in pixels of the original image (top-left origin), as a span of bucket
    pixels counted from the bucket's centre, or None when the crop leaves nothing of it. As the
    compiler does: the near edge floored, the far one ceiled, in pixels."""
    return _fit_span(lo_px, hi_px, to_bucket(lo_px) - bucket_center,
                     math.floor(round(to_bucket(lo_px), 6)) - bucket_center,
                     math.ceil(round(to_bucket(hi_px), 6)) - bucket_center,
                     -bucket_center, bucket_size - bucket_center)


def _centered_boxes(entries, g, source, label):
    """Rewrite centred pixel yxyx boxes. (the entries to keep, whether anything changed)"""
    # The origin, in top-left pixel coordinates: the compiler subtracts height // 2 and
    # width // 2 from pixel coordinates (centered_box in compile_prompt.py).
    src_cx, src_cy = g.src_width // 2, g.src_height // 2
    dst_cx, dst_cy = g.dst_width // 2, g.dst_height // 2
    kept, changed = [], False
    for index, item in enumerate(entries):
        if not (isinstance(item, dict) and 'bbox' in item):
            kept.append(item)
            continue
        location = f'{source}: {label}[{index}].bbox'
        y1, x1, y2, x2 = _box(item['bbox'], location)
        # The compiler's boxes lie within [-(side // 2), side - side // 2]; one pixel of slack
        # for hand-written ones.
        if not (-src_cy - 1 <= y1 and y2 <= g.src_height - src_cy + 1
                and -src_cx - 1 <= x1 and x2 <= g.src_width - src_cx + 1):
            raise ValueError(f'{location} {list(item["bbox"])} lies outside the {g.src_width}x{g.src_height} image')
        rows = _centered_span(y1 + src_cy, y2 + src_cy, g.y, dst_cy, g.dst_height)
        cols = _centered_span(x1 + src_cx, x2 + src_cx, g.x, dst_cx, g.dst_width)
        if rows is None or cols is None:
            changed = True  # wholly cropped out
            continue
        box = [rows[0], cols[0], rows[1], cols[1]]
        if box != list(item['bbox']):
            item = dict(item, bbox=box)
            changed = True
        kept.append(item)
    return kept, changed


def _normalized_boxes(entries, g, source, label, to_centered=False):
    """Rewrite [ymin, xmin, ymax, xmax] boxes in 0..1000 of the image (origin top left) for the
    bucket, or with to_centered as centred pixel boxes of the bucket. (the entries to keep,
    whether anything changed)"""
    kept, changed = [], False
    for index, entry in enumerate(entries):
        if not (isinstance(entry, dict) and 'bbox' in entry):
            kept.append(entry)
            continue
        location = f'{source}: {label}[{index}].bbox'
        ymin, xmin, ymax, xmax = _box(entry['bbox'], location)
        if not all(0 <= v <= 1000 for v in (ymin, xmin, ymax, xmax)):
            raise ValueError(f'{location} {list(entry["bbox"])} lies outside 0..1000')
        if to_centered:
            rows = _centered_span(ymin / 1000 * g.src_height, ymax / 1000 * g.src_height, g.y,
                                  g.dst_height // 2, g.dst_height)
            cols = _centered_span(xmin / 1000 * g.src_width, xmax / 1000 * g.src_width, g.x,
                                  g.dst_width // 2, g.dst_width)
        else:
            def grid_y(v):
                return g.y(v / 1000 * g.src_height) / g.dst_height * 1000

            def grid_x(v):
                return g.x(v / 1000 * g.src_width) / g.dst_width * 1000

            rows = _fit_span(ymin, ymax, grid_y(ymin), round(grid_y(ymin)), round(grid_y(ymax)), 0, 1000)
            cols = _fit_span(xmin, xmax, grid_x(xmin), round(grid_x(xmin)), round(grid_x(xmax)), 0, 1000)
        if rows is None or cols is None:
            changed = True  # wholly cropped out
            continue
        box = [rows[0], cols[0], rows[1], cols[1]]
        if box != list(entry['bbox']):
            entry = dict(entry, bbox=box)
            changed = True
        kept.append(entry)
    return kept, changed


# The lists of a MOSS-Image prompt that hold boxes: the summary/items schema and the
# scene/element schema of the technical report (appendix C.3).
MOSS_BOX_LISTS = ('items', 'elements')


def _moss_space(obj, source):
    """'centered' or 'normalized': the space the boxes of a MOSS-Image prompt are written in."""
    space, layout = obj.get('bbox_coord_space'), obj.get('bbox_format')
    if (space, layout) == ('abs_centralized', 'yxyx'):
        return 'centered'
    # The report's own prompts carry no markers (Figure 6) or, for half of them, an explicit one.
    if space in (None, 'normalized_1000') and layout in (None, 'yxyx'):
        return 'normalized'
    raise ValueError(
        f'{source}: MOSS-Image layout boxes must be bbox_coord_space "abs_centralized" with bbox_format '
        f'"yxyx" (what its prompt compiler writes), or normalized to 0..1000 as [y1, x1, y2, x2] '
        f'(bbox_coord_space "normalized_1000" or no marker at all), got {space!r} and {layout!r}')


def _moss_boxes(obj, geometry, source, to_centered=False):
    """Rewrite the boxes of a MOSS-Image prompt in place. True if anything changed."""
    lists = {}
    for name in MOSS_BOX_LISTS:
        entries = obj.get(name)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError(f'{source}: {name} must be an array')
        if any(isinstance(entry, dict) and 'bbox' in entry for entry in entries):
            lists[name] = entries
    if not lists:
        return False
    space = _moss_space(obj, source)
    changed = False
    for name, entries in lists.items():
        if space == 'centered':
            kept, list_changed = _centered_boxes(entries, geometry, source, name)
        else:
            kept, list_changed = _normalized_boxes(entries, geometry, source, name, to_centered)
        if list_changed:
            obj[name] = kept
            changed = True
    if to_centered and space == 'normalized':
        obj['bbox_coord_space'] = 'abs_centralized'
        obj['bbox_format'] = 'yxyx'
        changed = True
    return changed


def _ideogram_boxes(obj, geometry, source):
    """Rewrite Ideogram 4's 0..1000 [ymin, xmin, ymax, xmax] boxes in place."""
    composition = obj.get('compositional_deconstruction')
    if composition is None:
        return False
    if not isinstance(composition, dict):
        raise ValueError(f'{source}: compositional_deconstruction must be an object')
    elements = composition.get('elements')
    if elements is None:
        return False
    if not isinstance(elements, list):
        raise ValueError(f'{source}: compositional_deconstruction.elements must be an array')
    kept, changed = _normalized_boxes(elements, geometry, source, 'compositional_deconstruction.elements')
    if changed:
        obj['compositional_deconstruction'] = dict(composition, elements=kept)
    return changed


def _insert_prefix(obj, path, prefix, source):
    """caption_prefix in front of the description field (path: keys from the root)."""
    parent = obj
    for key in path[:-1]:
        parent = parent.get(key) if isinstance(parent, dict) else None
    field = '.'.join(path)
    if not isinstance(parent, dict) or not isinstance(parent.get(path[-1]), str):
        raise ValueError(f'{source}: caption_prefix is set, so this layout caption needs a {field} '
                         f'string to put it in')
    parent[path[-1]] = prefix + parent[path[-1]]


def _same_style_dump(obj, original):
    """obj in the JSON formatting the original text uses, when one of the usual styles
    reproduces it exactly (compact, json.dumps' default, an indent such as ComfyUI's
    indent=2); else compact, the form MOSS-Image's compiler writes."""
    parsed = json.loads(original)
    styles = [dict(separators=(',', ':')), dict(separators=(', ', ': '))]
    lines = original.splitlines()
    if len(lines) > 1:
        indent = len(lines[1]) - len(lines[1].lstrip(' '))
        if indent:
            styles.append(dict(indent=indent))
            styles.append(dict(indent=indent, separators=(',', ': ')))
        elif lines[1].startswith('\t'):
            styles.append(dict(indent='\t'))
    for style in styles:
        for ensure_ascii in (False, True):
            if json.dumps(parsed, ensure_ascii=ensure_ascii, **style) == original:
                return json.dumps(obj, ensure_ascii=ensure_ascii, **style)
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def rewrite_layout_caption(caption, layout_format, src_size, dst_size, caption_prefix='', source='caption'):
    """The layout caption for an image of src_size (w, h) fitted to dst_size (w, h).

    A plain caption is returned unchanged; the caller applies caption_prefix to those.
    """
    obj = parse_layout_caption(caption)
    if obj is None:
        return caption
    geometry = _Geometry(src_size, dst_size)
    if layout_format in ('moss_image', 'moss_image_centered'):
        changed = _moss_boxes(obj, geometry, source, to_centered=layout_format == 'moss_image_centered')
        if caption_prefix:
            # The scene/element schema has no summary: its description is gen_brief.
            path = ('gen_brief',) if 'summary' not in obj and isinstance(obj.get('gen_brief'), str) \
                else ('summary', 'caption')
            _insert_prefix(obj, path, caption_prefix, source)
            changed = True
    elif layout_format == 'ideogram4':
        changed = _ideogram_boxes(obj, geometry, source)
        if caption_prefix:
            if 'high_level_description' not in obj:
                # BuildJsonPromptIdeogram leaves out an empty description; the prefix becomes
                # it, in the place that node puts the key (first).
                obj = {'high_level_description': caption_prefix.strip().rstrip(',').strip(), **obj}
            else:
                _insert_prefix(obj, ('high_level_description',), caption_prefix, source)
            changed = True
    else:
        raise ValueError(f'Unknown layout caption format {layout_format!r}; expected one of {LAYOUT_FORMATS}')
    original = caption.strip()
    return _same_style_dump(obj, original) if changed else original



class LayoutCaptionRewriter:
    """Per-directory rewrite of the layout captions of one bucket's rows.

    Called with the media it describes and the bucket it is trained at, so the same caption
    becomes a different text (and embedding) in each bucket.
    """

    def __init__(self, layout_format, caption_prefix=''):
        if layout_format not in LAYOUT_FORMATS:
            raise ValueError(f'Unknown layout caption format {layout_format!r}; expected one of {LAYOUT_FORMATS}')
        self.layout_format = layout_format
        self.caption_prefix = caption_prefix or ''

    @property
    def cache_tag(self):
        """Names what the rewrite does, for cache paths and fingerprints."""
        return f'layout-v{LAYOUT_REWRITE_VERSION}-{self.layout_format}'

    def __call__(self, caption, image_spec, size_bucket, is_video=False):
        if not is_layout_caption(caption):
            return caption
        source = image_spec[1] if image_spec[0] is None else f'{image_spec[0]}:{image_spec[1]}'
        if is_video:
            raise ValueError(f'{source}: layout (JSON) captions are only supported for images')
        if size_bucket is None:
            raise ValueError(f'{source}: a layout caption needs the size bucket it is trained at')
        return rewrite_layout_caption(caption, self.layout_format, media_size(image_spec),
                                      bucket_pixel_size(size_bucket), self.caption_prefix, source)
