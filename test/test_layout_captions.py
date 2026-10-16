"""Layout (JSON) captions: utils/layout_captions.py and how the dataset uses it.

The boxes of a layout caption describe the image the model sees, so they are checked against
the pixels ImageOps.fit actually produces (models/base.py convert_crop_and_resize), not only
against arithmetic written the same way as the code under test.
"""

import functools
import io
import json
import math
import os
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import utils.dataset as dataset_module  # noqa: E402
from utils.captions import preprocess_caption, read_caption_file, shuffle_captions, split_caption_lines  # noqa: E402
from utils.dataset import CAPTION_CACHE_SETTINGS, DirectoryDataset, caption_cache_suffix  # noqa: E402
from utils.layout_captions import (  # noqa: E402
    LayoutCaptionRewriter,
    bucket_pixel_size,
    fit_crop_box,
    is_layout_caption,
    media_size,
    parse_layout_caption,
    rewrite_layout_caption,
)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setattr(dataset_module, 'is_main_process', lambda: True)


class compiler:
    """MOSS-Image's own prompt compiler, the part these tests check against: centered_box from
    skills/moss-image-prompt-enhancer/scripts/compile_prompt.py of
    https://github.com/OpenMOSS/MOSS-Image (commit 1f808b7, Apache-2.0), copied unchanged."""

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def centered_box(box, width, height, location):
        require = compiler.require
        require(isinstance(box, list) and len(box) == 4, f"{location} must be [x1,y1,x2,y2]")
        require(
            all(type(v) in (int, float) and math.isfinite(v) for v in box),
            f"{location} must contain four finite numbers",
        )
        x1, y1, x2, y2 = box
        require(x1 < x2 and y1 < y2, f"{location} must have positive width and height")
        require(
            0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
            f"{location} exceeds the {width}x{height} canvas",
        )
        # Enclose in pixel edges before centering, as in Echidna generation export.
        result = [
            math.floor(round(y1, 9)) - height // 2,
            math.floor(round(x1, 9)) - width // 2,
            math.ceil(round(y2, 9)) - height // 2,
            math.ceil(round(x2, 9)) - width // 2,
        ]
        require(result[0] < result[2] and result[1] < result[3],
                f"{location} is too small at export precision")
        return result


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def moss_caption(items, **extra):
    obj = {'summary': {'caption': 'a poster'}, 'items': items,
           'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx', **extra}
    return compact(obj)


def fitted(image, size):
    """The pixels training sees: what PreprocessMediaFile does to an RGB image."""
    return ImageOps.fit(image, size)


def dark_extent(image):
    """(top, left, bottom, right) of the dark pixels, bottom/right exclusive."""
    dark = np.asarray(image.convert('L')) < 128
    rows, cols = np.flatnonzero(dark.any(axis=1)), np.flatnonzero(dark.any(axis=0))
    return rows[0], cols[0], rows[-1] + 1, cols[-1] + 1


def image_with_rect(size, top, left, bottom, right):
    image = Image.new('RGB', size, (255, 255, 255))
    image.paste((0, 0, 0), (left, top, right, bottom))
    return image


class TestDetection:
    @pytest.mark.parametrize('caption', ['{"a": 1}', '  {"summary": {"caption": "x"}}\n', '{\n  "a": [1, 2]\n}'])
    def test_json_objects_are_layout_captions(self, caption):
        assert is_layout_caption(caption)
        assert isinstance(parse_layout_caption(caption), dict)

    @pytest.mark.parametrize('caption', ['a cat, a dog', '[1, 2]', '{not json}', '{"a": 1} trailing',
                                         'prefix {"a": 1}', '', '"{}"', None, 3])
    def test_everything_else_is_plain(self, caption):
        assert not is_layout_caption(caption)
        assert parse_layout_caption(caption) is None


class TestGeometry:
    @pytest.mark.parametrize('src,dst', [((120, 80), (64, 64)), ((80, 120), (64, 64)), ((64, 64), (64, 64)),
                                         ((101, 57), (48, 80)), ((333, 77), (96, 32)), ((50, 50), (96, 32))])
    def test_fit_crop_box_is_the_box_pillow_resizes(self, monkeypatch, src, dst):
        seen = {}
        resize = Image.Image.resize

        def spy(self, size, resample=None, box=None, reducing_gap=None):
            seen['box'] = box
            return resize(self, size, resample, box, reducing_gap)

        monkeypatch.setattr(Image.Image, 'resize', spy)
        ImageOps.fit(Image.new('RGB', src), dst)
        left, top, width, height = fit_crop_box(*src, *dst)
        assert seen['box'] == pytest.approx((left, top, left + width, top + height))

    def test_bucket_pixel_size_rounds_like_the_media_preprocessing(self):
        from utils.common import round_to_nearest_multiple
        for w, h in [(100, 70), (64, 64), (1000, 1000), (1448, 1448 * 9 // 16), (23, 41)]:
            expected = (round_to_nearest_multiple(w, 16), round_to_nearest_multiple(h, 16))
            assert bucket_pixel_size((w, h, 1)) == expected

    def test_media_size_reads_files_and_tar_members(self, tmp_path):
        path = tmp_path / 'a.png'
        Image.new('RGB', (37, 21)).save(path)
        assert media_size((None, str(path))) == (37, 21)
        tar_path = tmp_path / 'd.tar'
        with tarfile.open(tar_path, 'w') as tar:
            tar.add(path, arcname='sub/a.png')
        assert media_size((str(tar_path), 'sub/a.png')) == (37, 21)

    def test_media_size_indexes_each_tar_once(self, tmp_path, monkeypatch):
        """Members are found through an index built once per archive: opening a TarFile per
        lookup rescans every header before the member, quadratic over a tar dataset."""
        import utils.layout_captions as layout_module
        tar_path = tmp_path / 'many.tar'
        sizes = {}
        with tarfile.open(tar_path, 'w') as tar:
            for i in range(300):
                image = Image.new('RGB', (8 + i % 7, 8 + i % 5))
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                info = tarfile.TarInfo(f'img/{i}.png')
                info.size = buffer.tell()
                buffer.seek(0)
                tar.addfile(info, buffer)
                sizes[f'img/{i}.png'] = image.size
        layout_module._media_size.cache_clear()
        opened = []
        real = tarfile.TarFile.__init__

        def counting_init(self, *args, **kwargs):
            opened.append(args[0] if args else kwargs.get('name'))
            real(self, *args, **kwargs)

        monkeypatch.setattr(tarfile.TarFile, '__init__', counting_init)
        scans = []
        real_next = tarfile.TarFile.next
        monkeypatch.setattr(tarfile.TarFile, 'next', lambda self: scans.append(1) or real_next(self))
        for _ in range(3):  # cache_shuffle variants and buckets ask again
            for name, size in sizes.items():
                assert media_size((str(tar_path), name)) == size
        assert len(opened) == 1
        # One pass over the headers (plus the end-of-archive read), not one per member.
        assert len(scans) <= len(sizes) + 2

    def test_tar_handles_are_not_shared_with_a_forked_process(self, tmp_path):
        import multiprocessing
        import utils.layout_captions as layout_module
        tar_path = tmp_path / 't.tar'
        with tarfile.open(tar_path, 'w') as tar:
            for i, size in enumerate([(5, 6), (7, 8)]):
                buffer = io.BytesIO()
                Image.new('RGB', size).save(buffer, format='PNG')
                info = tarfile.TarInfo(f'{i}.png')
                info.size = buffer.tell()
                buffer.seek(0)
                tar.addfile(info, buffer)
        layout_module._media_size.cache_clear()
        assert media_size((str(tar_path), '0.png')) == (5, 6)
        parent_tar = layout_module._open_tars.tars[str(tar_path)][0]
        context = multiprocessing.get_context('fork')
        with context.Pool(1) as pool:
            child = pool.apply(_child_tar_identity, (str(tar_path),))
        # The child inherited the parent's open archive and opened its own instead.
        assert child == ((7, 8), True, True)
        assert layout_module._open_tars.tars[str(tar_path)][0] is parent_tar


def _child_tar_identity(tar_path):
    import utils.layout_captions as layout_module
    inherited = layout_module._open_tars.tars.get(tar_path)
    size = media_size((tar_path, '1.png'))
    return size, inherited is not None, layout_module._open_tars.tars[tar_path] is not inherited


class TestMossBoxes:
    def test_box_follows_the_pixels_through_the_crop(self):
        # 120x80 image, rectangle x 70..100, y 20..60; MOSS boxes are centred: x - 60, y - 40.
        image = image_with_rect((120, 80), 20, 70, 60, 100)
        caption = moss_caption([{'label': 'text', 'bbox': [-20, 10, 20, 40]}])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64)))
        y1, x1, y2, x2 = out['items'][0]['bbox']
        top, left, bottom, right = dark_extent(fitted(image, (64, 64)))
        assert (y1 + 32, x1 + 32, y2 + 32, x2 + 32) == (top, left, bottom, right) == (16, 40, 48, 64)

    @pytest.mark.parametrize('seed', range(12))
    def test_random_boxes_match_the_fitted_pixels(self, seed):
        rng = random.Random(seed)
        src = (rng.randrange(60, 300), rng.randrange(60, 300))
        dst = bucket_pixel_size((rng.randrange(32, 200), rng.randrange(32, 200), 1))
        left, top, cw, ch = fit_crop_box(*src, *dst)
        # A rectangle well inside the crop, so no clipping is involved.
        rl = rng.randrange(int(left) + 4, int(left + cw * 0.5))
        rt = rng.randrange(int(top) + 4, int(top + ch * 0.5))
        rr = rng.randrange(rl + int(cw * 0.2), int(left + cw) - 3)
        rb = rng.randrange(rt + int(ch * 0.2), int(top + ch) - 3)
        image = image_with_rect(src, rt, rl, rb, rr)
        # The compiler's origin: width // 2, height // 2.
        cx, cy = src[0] // 2, src[1] // 2
        bbox = [rt - cy, rl - cx, rb - cy, rr - cx]
        out = json.loads(rewrite_layout_caption(moss_caption([{'bbox': bbox}]), 'moss_image', src, dst))
        y1, x1, y2, x2 = out['items'][0]['bbox']
        assert all(type(v) is int for v in (y1, x1, y2, x2))
        top_px, left_px, bottom_px, right_px = dark_extent(fitted(image, dst))
        # Resampling blurs an edge across about a pixel; the box must cover the rectangle and
        # stay within a pixel of it.
        dy, dx = dst[1] // 2, dst[0] // 2
        assert y1 + dy <= top_px <= y1 + dy + 1 and bottom_px - 1 <= y2 + dy <= bottom_px + 1
        assert x1 + dx <= left_px <= x1 + dx + 1 and right_px - 1 <= x2 + dx <= right_px + 1

    def test_box_covers_every_pixel_it_touches(self):
        # 100x100 to 64x64 scales by 0.64: source 15..45 lands on 9.6..28.8, so the box is the
        # pixels 9..29 (the compiler's floor/ceil), centred at 32: -23..-3.
        caption = moss_caption([{'bbox': [-35, -35, -5, -5]}])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (100, 100), (64, 64)))
        assert out['items'][0]['bbox'] == [-23, -23, -3, -3]
        # An edge on a pixel boundary stays there despite float error: 50 to 112 maps source
        # 25 to 25 * (112 / 50) = 56.00000000000001, which a bare ceil would push to 57.
        caption = moss_caption([{'bbox': [-25, -25, 0, 0]}])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (50, 50), (112, 112)))
        assert out['items'][0]['bbox'] == [-56, -56, 0, 0]

    def test_box_cut_by_the_crop_is_clipped_and_one_outside_is_dropped(self):
        # 120x80 to 64x64 keeps x 20..100 of the source (centred: -40..40).
        caption = moss_caption([
            {'label': 'cut', 'bbox': [-10, -60, 10, -30]},
            {'label': 'gone', 'bbox': [-10, -60, 10, -45]},
            {'label': 'no box'},
        ])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64)))
        assert [item['label'] for item in out['items']] == ['cut', 'no box']
        assert out['items'][0]['bbox'] == [-8, -32, 8, -24]

    def test_identity_returns_the_caption_text(self):
        caption = '{"summary": {"caption": "x"}, "items": [{"bbox": [-8, -8, 8, 8]}], ' \
                  '"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}'
        assert rewrite_layout_caption('  ' + caption + '\n', 'moss_image', (64, 64), (64, 64)) == caption
        # Formatting no json.dumps style reproduces is kept too when nothing changes.
        odd = caption.replace('{"summary": ', '{ "summary" :  ')
        assert rewrite_layout_caption(odd, 'moss_image', (64, 64), (64, 64)) == odd

    def test_items_without_boxes_need_no_coordinate_space(self):
        caption = compact({'summary': {'caption': 'x'}, 'items': [{'label': 'a'}]})
        assert rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64)) == caption

    @pytest.mark.parametrize('style', [dict(separators=(',', ':')), dict(), dict(indent=2),
                                       dict(indent=4, separators=(',', ': ')), dict(indent='\t')])
    @pytest.mark.parametrize('ensure_ascii', [False, True])
    def test_rewrite_keeps_the_caption_formatting(self, style, ensure_ascii):
        obj = {'summary': {'caption': 'café ☕'}, 'items': [{'bbox': [-20, 10, 20, 40]}],
               'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx'}
        caption = json.dumps(obj, ensure_ascii=ensure_ascii, **style)
        out = rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64))
        expected = dict(obj, items=[{'bbox': [-16, 8, 16, 32]}])
        assert out == json.dumps(expected, ensure_ascii=ensure_ascii, **style)

    def test_unrecognised_formatting_falls_back_to_compact(self):
        caption = '{ "summary" : {"caption":"x"},"items":[{"bbox":[-20,10,20,40]}],' \
                  '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
        out = rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64))
        assert out == compact(dict(json.loads(caption), items=[{'bbox': [-16, 8, 16, 32]}]))

    def test_box_outside_the_image_names_the_file(self):
        caption = moss_caption([{'bbox': [-20, 10, 20, 62]}])
        with pytest.raises(ValueError, match=r'img/a\.png: items\[0\]\.bbox .* outside the 120x80 image'):
            rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64), source='img/a.png')

    def test_far_edge_of_an_odd_image_is_accepted(self):
        # The compiler subtracts 121 // 2 = 60 from x, so a 121-wide image runs -60..61.
        caption = moss_caption([{'bbox': [-40, -60, 40, 61]}])
        rewrite_layout_caption(caption, 'moss_image', (121, 80), (64, 64))

    @pytest.mark.parametrize('src,dst', [((121, 81), (64, 48)), ((97, 203), (112, 224)),
                                         ((333, 101), (96, 32)), ((64, 64), (64, 64))])
    def test_odd_images_follow_the_compilers_origin(self, src, dst):
        """Compile an absolute-pixel draft for the original image with MOSS-Image's own
        compiler, rewrite it, and compare with the compiler run on the same box mapped to the
        bucket: identical, also when a side is odd (the origin is side // 2, not side / 2)."""
        left, top, cw, ch = fit_crop_box(*src, *dst)
        sx, sy = dst[0] / cw, dst[1] / ch
        rng = random.Random(hash(src) & 0xffff)
        for _ in range(20):
            x1 = rng.uniform(left + 1, left + cw * 0.5)
            y1 = rng.uniform(top + 1, top + ch * 0.5)
            x2 = rng.uniform(x1 + 2, left + cw - 1)
            y2 = rng.uniform(y1 + 2, top + ch - 1)
            # A whole-pixel box, as a draft drawn on the image would be.
            x1, y1, x2, y2 = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
            box = compiler.centered_box([x1, y1, x2, y2], src[0], src[1], 'bbox')
            out = json.loads(rewrite_layout_caption(moss_caption([{'bbox': box}]), 'moss_image', src, dst))
            mapped = [(x1 - left) * sx, (y1 - top) * sy, (x2 - left) * sx, (y2 - top) * sy]
            assert out['items'][0]['bbox'] == compiler.centered_box(mapped, dst[0], dst[1], 'bbox')

    def test_zero_size_box_inside_the_crop_is_kept(self):
        # A zero-height box is a degenerate but valid region; it moves with the crop.
        caption = moss_caption([{'label': 'line', 'bbox': [0, -40, 0, 40]}])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64)))
        assert out['items'][0]['bbox'] == [0, -32, 0, 32]
        # One that the crop removes is dropped like any other, on either side: the crop keeps
        # x 20..100 of 120 (centred -40..40), so x = 41 is outside, as is x = -41.
        for x in (-55, -41, 41, 55):
            caption = moss_caption([{'label': 'line', 'bbox': [-10, x, 10, x]}])
            out = json.loads(rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64)))
            assert out['items'] == [], x
        # On the crop's edge it stays, inside the bucket.
        caption = moss_caption([{'label': 'line', 'bbox': [-10, 40, 10, 40]}])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (120, 80), (64, 64)))
        assert out['items'][0]['bbox'] == [-8, 32, 8, 32]

    @pytest.mark.parametrize('space,layout', [('abs', 'yxyx'), ('abs_centralized', 'xyxy'), (None, None)])
    def test_other_coordinate_spaces_are_refused(self, space, layout):
        obj = {'summary': {'caption': 'x'}, 'items': [{'bbox': [0, 0, 8, 8]}]}
        if space:
            obj.update(bbox_coord_space=space, bbox_format=layout)
        with pytest.raises(ValueError, match='abs_centralized'):
            rewrite_layout_caption(compact(obj), 'moss_image', (64, 64), (64, 64))

    @pytest.mark.parametrize('bbox', [[0, 0, 8], [0, 0, 8, 'x'], [0, 0, 8, True], [8, 0, 0, 8],
                                      [0, 8, 8, 0], [0, 0, 8, float('nan')], 'box'])
    def test_malformed_boxes_are_refused(self, bbox):
        caption = json.dumps({'summary': {'caption': 'x'}, 'items': [{'bbox': bbox}],
                              'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx'})
        with pytest.raises(ValueError, match=r'items\[0\]\.bbox'):
            rewrite_layout_caption(caption, 'moss_image', (64, 64), (64, 64))

    def test_caption_prefix_goes_into_the_summary_caption(self):
        caption = moss_caption([{'bbox': [-8, -8, 8, 8]}])
        out = json.loads(rewrite_layout_caption(caption, 'moss_image', (64, 64), (64, 64), caption_prefix='sks, '))
        assert out['summary']['caption'] == 'sks, a poster'
        assert out['items'][0]['bbox'] == [-8, -8, 8, 8]

    @pytest.mark.parametrize('obj', [{'items': []}, {'summary': 'x'}, {'summary': {'caption': 3}}])
    def test_caption_prefix_without_a_summary_caption_names_the_file(self, obj):
        with pytest.raises(ValueError, match=r'a\.png: caption_prefix is set.*summary\.caption'):
            rewrite_layout_caption(compact(obj), 'moss_image', (64, 64), (64, 64), 'sks, ', source='a.png')


class TestIdeogramBoxes:
    def caption(self, elements, **extra):
        return json.dumps({'high_level_description': 'a poster', 'style_description': 'flat',
                           'compositional_deconstruction': {'background': 'white', 'elements': elements},
                           **extra}, indent=2, ensure_ascii=True)

    def test_box_follows_the_pixels_and_keeps_the_formatting(self):
        # 120x80, rectangle x 70..100, y 20..60: 0..1000 grid is x * 1000/120, y * 1000/80.
        image = image_with_rect((120, 80), 20, 70, 60, 100)
        bbox = [250, round(70 / 120 * 1000), 750, round(100 / 120 * 1000)]
        caption = self.caption([{'type': 'text', 'bbox': bbox, 'text': 'Café'}])
        out = rewrite_layout_caption(caption, 'ideogram4', (120, 80), (64, 64))
        obj = json.loads(out)
        assert out == json.dumps(obj, indent=2, ensure_ascii=True)
        # The compact form (json.dumps(..., separators=(",", ":"), ensure_ascii=False)) stays compact.
        compact_caption = compact(json.loads(caption))
        out_compact = rewrite_layout_caption(compact_caption, 'ideogram4', (120, 80), (64, 64))
        assert out_compact == compact(obj) and 'Café' in out_compact
        top, left, bottom, right = dark_extent(fitted(image, (64, 64)))
        ymin, xmin, ymax, xmax = obj['compositional_deconstruction']['elements'][0]['bbox']
        # The input box is itself rounded to the grid (583 for 583.3), so allow one grid step.
        expected = [round(v / 64 * 1000) for v in (top, left, bottom, right)]
        assert all(abs(a - b) <= 1 for a, b in zip([ymin, xmin, ymax, xmax], expected))

    def test_unchanged_caption_is_used_as_written(self):
        caption = self.caption([{'type': 'obj', 'bbox': [0, 0, 1000, 1000]}])
        assert rewrite_layout_caption(caption + '\n', 'ideogram4', (64, 64), (64, 64)) == caption
        caption = json.dumps({'high_level_description': 'only text'}, indent=2)
        assert rewrite_layout_caption(caption, 'ideogram4', (64, 64), (64, 64)) == caption

    @pytest.mark.parametrize('indent', [2, 0])
    def test_comfyui_dictionary_to_string_output_keeps_its_form(self, indent):
        """ComfyUI's ConvertDictionaryToString (indent 2 by default; 0 is json.dumps' default
        one-line form): the rewritten prompt is in the form ComfyUI sends at inference."""
        obj = json.loads(self.caption([{'type': 'a', 'bbox': [250, 500, 750, 800]}]))
        caption = json.dumps(obj, ensure_ascii=False, indent=indent or None)
        out = rewrite_layout_caption(caption, 'ideogram4', (120, 80), (64, 64))
        expected = json.loads(out)
        assert expected['compositional_deconstruction']['elements'][0]['bbox'] != [250, 500, 750, 800]
        assert out == json.dumps(expected, ensure_ascii=False, indent=indent or None)

    def test_zero_size_boxes_from_comfyui_are_accepted(self):
        # nodes_bounding_boxes._norm_bbox writes ymin == ymax for a zero-height region.
        caption = self.caption([{'type': 'line', 'bbox': [500, 250, 500, 750]},
                                {'type': 'gone', 'bbox': [500, 0, 500, 100]}])
        out = json.loads(rewrite_layout_caption(caption, 'ideogram4', (120, 80), (64, 64)))
        elements = out['compositional_deconstruction']['elements']
        assert [e['type'] for e in elements] == ['line']
        assert elements[0]['bbox'] == [500, 125, 500, 875]

    def test_cut_box_is_clipped_and_outside_box_dropped(self):
        # 120x80 to 64x64 keeps x 20..100 of 120, i.e. 167..833 on the 0..1000 grid.
        caption = self.caption([{'type': 'a', 'bbox': [0, 0, 1000, 500]},
                                {'type': 'b', 'bbox': [0, 0, 1000, 100]},
                                {'type': 'c'}])
        out = json.loads(rewrite_layout_caption(caption, 'ideogram4', (120, 80), (64, 64)))
        elements = out['compositional_deconstruction']['elements']
        assert [e['type'] for e in elements] == ['a', 'c']
        # x 0..60 px -> (0 - 20) * 0.8 .. (60 - 20) * 0.8 = -16..32 of 64 -> clipped to 0..500.
        assert elements[0]['bbox'] == [0, 0, 1000, 500]
        assert out['compositional_deconstruction']['background'] == 'white'

    def test_vertical_cut_is_clipped(self):
        # 80x120 to 64x64 keeps y 20..100 of 120: y 0..60 px -> -16..32 of 64 -> 0..500.
        caption = self.caption([{'type': 'a', 'bbox': [0, 0, 500, 1000]}])
        out = json.loads(rewrite_layout_caption(caption, 'ideogram4', (80, 120), (64, 64)))
        assert out['compositional_deconstruction']['elements'][0]['bbox'] == [0, 0, 500, 1000]

    def test_box_outside_the_grid_names_the_file(self):
        caption = self.caption([{'type': 'a', 'bbox': [0, 0, 1001, 500]}])
        with pytest.raises(ValueError, match=r'b\.png: compositional_deconstruction\.elements\[0\]\.bbox .*0\.\.1000'):
            rewrite_layout_caption(caption, 'ideogram4', (64, 64), (64, 64), source='b.png')

    def test_caption_prefix_goes_into_the_high_level_description(self):
        out = json.loads(rewrite_layout_caption(self.caption([]), 'ideogram4', (64, 64), (64, 64), 'sks, '))
        assert out['high_level_description'] == 'sks, a poster'
        with pytest.raises(ValueError, match=r'c\.png: .*high_level_description'):
            rewrite_layout_caption('{"high_level_description": 3}', 'ideogram4', (64, 64), (64, 64), 'sks, ', 'c.png')

    def test_caption_prefix_becomes_a_missing_description(self):
        # BuildJsonPromptIdeogram leaves high_level_description out when it is empty.
        caption = compact({'style_description': 'flat', 'compositional_deconstruction': {'elements': []}})
        out = rewrite_layout_caption(caption, 'ideogram4', (64, 64), (64, 64), 'sks, ')
        assert out == compact({'high_level_description': 'sks', 'style_description': 'flat',
                               'compositional_deconstruction': {'elements': []}})


class TestRewriter:
    def test_plain_captions_pass_through(self, tmp_path):
        rewriter = LayoutCaptionRewriter('moss_image', 'sks, ')
        # No media is read for a plain caption: the path does not exist.
        assert rewriter('sks, a cat', (None, str(tmp_path / 'missing.png')), (64, 64, 1)) == 'sks, a cat'

    def test_rewrites_for_the_media_and_bucket(self, tmp_path):
        path = tmp_path / 'a.png'
        Image.new('RGB', (120, 80)).save(path)
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        out = LayoutCaptionRewriter('moss_image')(caption, (None, str(path)), (63, 66, 1))
        assert json.loads(out)['items'][0]['bbox'] == [-16, 8, 16, 32]

    def test_refusals(self, tmp_path):
        caption = moss_caption([])
        with pytest.raises(ValueError, match='Unknown layout caption format'):
            LayoutCaptionRewriter('flux')
        with pytest.raises(ValueError, match=r'v\.mp4: .*only supported for images'):
            LayoutCaptionRewriter('moss_image')(caption, (None, 'v.mp4'), (64, 64, 33), is_video=True)
        with pytest.raises(ValueError, match=r'a\.png: .*size bucket'):
            LayoutCaptionRewriter('moss_image')(caption, (None, 'a.png'), None)

    def test_cache_tag_names_the_format(self):
        assert LayoutCaptionRewriter('moss_image').cache_tag != LayoutCaptionRewriter('ideogram4').cache_tag


LAYOUT = '{"summary": {"caption": "a, b, c"}, "items": [], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}'


class TestCaptionOperationsSkipLayoutCaptions:
    def test_shuffle_and_tag_dropout_leave_json_alone(self):
        out = shuffle_captions([LAYOUT, 'x, y, z, w'], count=3, delimiter=', ', caption_prefix='p, ',
                               tag_dropout_rate=0.5, seed=1, layout_captions=True)
        assert out[:3] == [LAYOUT] * 3
        assert all(c.startswith('p, ') for c in out[3:])

    def test_fast_path_leaves_json_alone_but_prefixes_plain_captions(self):
        out = shuffle_captions([LAYOUT, 'x'], caption_prefix='p, ', layout_captions=True)
        assert out == [LAYOUT, 'p, x']

    def test_without_the_flag_json_is_an_ordinary_caption(self):
        assert shuffle_captions([LAYOUT], caption_prefix='p, ') == ['p, ' + LAYOUT]

    def test_non_latin_filter_and_tag_marker_leave_json_alone(self):
        caption = '{"summary": {"caption": "東京の夜"}, "items": []}'
        assert preprocess_caption(caption, caption_prefix='p, ', enable_remove_non_latin=True,
                                  layout_captions=True) == caption
        assert preprocess_caption(caption, enable_remove_non_latin=True) != caption

    def test_multiline_json_is_one_caption(self, tmp_path):
        pretty = json.dumps(json.loads(LAYOUT), indent=2)
        assert split_caption_lines(pretty, multiline_captions=True, layout_captions=True) == [pretty]
        assert len(split_caption_lines(pretty, multiline_captions=True)) > 1
        assert split_caption_lines('a\nb', multiline_captions=True, layout_captions=True) == ['a', 'b']
        path = tmp_path / 'c.txt'
        path.write_text('﻿' + pretty + '\n', encoding='utf-8')
        assert read_caption_file(path, multiline_captions=True, layout_captions=True) == [pretty]


def _stub_latents(example, rank):
    import torch
    return {'latents': torch.zeros(len(example['image_spec']), 4, 8, 8), 'image_spec': example['image_spec'],
            'caption': example['caption'], 'mask': [None] * len(example['image_spec'])}


def _stub_text_encoder(example, rank):
    """The caption text as bytes, so a test can read back exactly what was encoded."""
    import torch
    return {'embedding': [torch.tensor(list(c.encode('utf-8')) or [0], dtype=torch.uint8)
                          for c in example['caption']]}


def _logging_text_encoder(log, example, rank):
    with open(log, 'a', encoding='utf-8') as handle:
        handle.write(''.join(c + '\n' for c in example['caption']))
    return _stub_text_encoder(example, rank)


def decode(embedding):
    data = bytes(embedding.tolist())
    return '' if data == b'\x00' else data.decode('utf-8')


def leaves(ds):
    result = []
    for bucket in ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets:
        result.extend(bucket.get_size_bucket_datasets() if hasattr(bucket, 'get_size_bucket_datasets') else [bucket])
    return result


class TestDataset:
    def make(self, tmp_path, captions, layout_format='moss_image', sizes=None, dataset_config=None,
             caches_text_embeddings=True, **overrides):
        d = tmp_path / 'imgs'
        d.mkdir(exist_ok=True)
        for name, text in captions.items():
            image_with_rect((sizes or {}).get(name, (120, 80)), 20, 70, 60, 100).save(d / f'{name}.png')
            (d / f'{name}.txt').write_text(text, encoding='utf-8')
        config = {'path': str(d), 'resolutions': [64], 'frame_buckets': [1], 'num_repeats': 1,
                  'reuse_metadata_cache': False, 'caption_dropout_rate': 0.0, **overrides}
        return DirectoryDataset(config, dataset_config or {'resolutions': [64]}, 'test_model',
                                skip_dataset_validation=True, layout_caption_format=layout_format,
                                caches_text_embeddings=caches_text_embeddings)

    def cache(self, ds, regenerate=True):
        ds.cache_metadata(regenerate_cache=regenerate)
        for bucket in ds.size_bucket_datasets if ds.use_size_buckets else ds.ar_bucket_datasets:
            bucket.cache_latents(_stub_latents, regenerate_cache=regenerate)
        ds.cache_text_embeddings(_stub_text_encoder, 0, regenerate_cache=regenerate)
        return leaves(ds)

    @staticmethod
    def encoded(leaf, name):
        spec = next(tuple(s) for s in leaf.metadata_dataset['image_spec'] if Path(s[1]).stem == name)
        return decode(leaf.text_embedding_datasets[0].get_text_embeddings(spec, 0)['embedding'])

    def expected(self, caption, bucket, prefix='', layout_format='moss_image', src=(120, 80)):
        return rewrite_layout_caption(caption, layout_format, src, bucket_pixel_size(bucket), prefix)

    def test_metadata_keeps_layout_captions_raw_and_announces_them(self, tmp_path, capsys):
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        ds = self.make(tmp_path, {'a': caption, 'b': 'x, y, z'}, caption_prefix='p, ', shuffle_tags=True,
                       cache_shuffle_num=2)
        ds.cache_metadata(regenerate_cache=True)
        rows = {Path(spec[1]).stem: c for b in ds.ar_bucket_datasets
                for spec, c in zip(b.metadata_dataset['image_spec'], b.metadata_dataset['caption'])}
        assert rows['a'] == [caption, caption]
        assert all(c.startswith('p, ') for c in rows['b'])
        out = capsys.readouterr().out
        assert '1 of 2 images have moss_image layout (JSON) captions' in out
        assert 'summary.caption' in out

    def test_no_notice_without_layout_captions(self, tmp_path, capsys):
        ds = self.make(tmp_path, {'a': 'plain'})
        ds.cache_metadata(regenerate_cache=True)
        assert 'layout (JSON) captions' not in capsys.readouterr().out

    def test_caption_settings_record_the_layout_format(self, tmp_path):
        plain = DirectoryDataset({'path': str(tmp_path), 'resolutions': [64], 'frame_buckets': [1],
                                  'reuse_metadata_cache': False}, {'resolutions': [64]}, 'm',
                                 skip_dataset_validation=True)
        layout = {f: DirectoryDataset({'path': str(tmp_path), 'resolutions': [64], 'frame_buckets': [1],
                                       'reuse_metadata_cache': False}, {'resolutions': [64]}, 'm',
                                      skip_dataset_validation=True, layout_caption_format=f)
                  for f in ('moss_image', 'ideogram4')}
        # A model without a format keeps the suffix it always had.
        assert plain.caption_cache_suffix == caption_cache_suffix(dict(CAPTION_CACHE_SETTINGS))
        suffixes = {plain.caption_cache_suffix, *(d.caption_cache_suffix for d in layout.values())}
        assert len(suffixes) == 3
        assert plain.layout_rewriter is None

    @pytest.mark.parametrize('reuse', [False, True])
    def test_text_cache_holds_the_caption_rewritten_for_each_bucket(self, tmp_path, reuse):
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        ds = self.make(tmp_path, {'a': caption, 'b': 'plain'}, caption_prefix='p, ',
                       resolutions=[64, 128], reuse_metadata_cache=reuse)
        found = {}
        for leaf in self.cache(ds):
            bucket = tuple(leaf.size_bucket)[-3:]
            found[bucket] = self.encoded(leaf, 'a')
            assert found[bucket] == self.expected(caption, bucket, 'p, ')
            assert self.encoded(leaf, 'b') == 'p, plain'
        # Two resolutions of one aspect ratio: two different texts, each cached by its bucket.
        assert len(found) == 2 and len(set(found.values())) == 2
        a, b = leaves(ds)
        assert a.text_embedding_datasets[0] is not b.text_embedding_datasets[0]

    def test_plain_ar_buckets_still_share_one_text_cache(self, tmp_path):
        ds = self.make(tmp_path, {'a': 'plain', 'b': 'other'}, resolutions=[64, 128])
        a, b = self.cache(ds)
        assert a.text_embedding_datasets[0] is b.text_embedding_datasets[0]

    def test_size_bucket_mode(self, tmp_path):
        caption = json.dumps({'high_level_description': 'x', 'compositional_deconstruction': {
            'elements': [{'type': 'a', 'bbox': [250, 583, 750, 833]}]}}, indent=2)
        sizes = {'wide': (120, 80), 'square': (80, 80)}
        ds = self.make(tmp_path, {'wide': caption, 'square': caption}, layout_format='ideogram4', sizes=sizes,
                       size_buckets=[[64, 64, 1], [96, 48, 1]], multiline_captions=True)
        texts = {}
        for leaf in self.cache(ds):
            (name,) = {Path(spec[1]).stem for spec in leaf.metadata_dataset['image_spec']}
            texts[name] = self.encoded(leaf, name)
            assert texts[name] == self.expected(caption, tuple(leaf.size_bucket), layout_format='ideogram4',
                                                src=sizes[name])
        assert sorted(texts) == ['square', 'wide'] and texts['square'] != texts['wide']

    def test_missing_prefix_field_fails_naming_the_file(self, tmp_path):
        ds = self.make(tmp_path, {'a': '{"items": []}'}, caption_prefix='p, ')
        with pytest.raises(ValueError, match=r'a\.png: caption_prefix is set'):
            self.cache(ds)

    def test_runtime_caption_is_rewritten_when_nothing_is_cached(self, tmp_path):
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        ds = self.make(tmp_path, {'a': caption}, caches_text_embeddings=False, caption_prefix='p, ')
        ds.cache_metadata(regenerate_cache=True)
        for bucket in ds.ar_bucket_datasets:
            bucket.cache_latents(_stub_latents, regenerate_cache=True)
        (leaf,) = leaves(ds)
        assert leaf[0]['caption'] == self.expected(caption, tuple(leaf.size_bucket)[-3:], 'p, ')

    def _legacy_then_profile(self, tmp_path, drop_digest):
        """A reuse_metadata_cache = false run from before layout captions (the JSON encoded as
        written), then a profile run with them, which may take tensors from the first."""
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        common = dict(size_buckets=[[64, 64, 1]])
        old = self.make(tmp_path, {'a': caption, 'b': 'plain'}, layout_format=None, **common)
        self.cache(old)
        manifests = list((tmp_path / 'imgs' / 'cache').rglob('text_embeddings_0*/cache_manifest.json'))
        assert manifests
        if drop_digest:
            for manifest in manifests:
                record = json.loads(manifest.read_text(encoding='utf-8'))
                del record['content_digest']
                manifest.write_text(json.dumps(record), encoding='utf-8')
        new = self.make(tmp_path, {'a': caption, 'b': 'plain'}, reuse_metadata_cache=True, **common)
        # Not regenerated: a regenerated snapshot is a new generation, which takes no donors.
        (leaf,) = self.cache(new, regenerate=False)
        return caption, leaf

    @pytest.mark.parametrize('drop_digest', [False, True])
    def test_an_old_cache_never_serves_the_unrewritten_caption(self, tmp_path, drop_digest):
        caption, leaf = self._legacy_then_profile(tmp_path, drop_digest)
        assert self.encoded(leaf, 'a') == self.expected(caption, (64, 64, 1))
        assert self.encoded(leaf, 'a') != caption
        assert self.encoded(leaf, 'b') == 'plain'

    def test_a_cache_of_rewritten_captions_is_reused(self, tmp_path):
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        common = dict(size_buckets=[[64, 64, 1]])
        self.cache(self.make(tmp_path, {'a': caption, 'b': 'plain'}, **common))
        new = self.make(tmp_path, {'a': caption, 'b': 'plain'}, reuse_metadata_cache=True, **common)
        (leaf,) = self.cache(new, regenerate=False)
        assert self.encoded(leaf, 'a') == self.expected(caption, (64, 64, 1))
        generated = list((tmp_path / 'imgs' / 'cache').rglob('text_embeddings_0_*/generated_*'))
        assert not generated, 'every caption was in the first run\'s cache, so nothing is encoded again'

    @pytest.mark.parametrize('reuse', [False, True])
    def test_a_later_launch_reads_no_image_headers(self, tmp_path, monkeypatch, reuse):
        """Loading the caches (every rank, every launch: DatasetManager.cache's map_fn=None
        pass) uses the rewritten captions kept on disk instead of rewriting them again, which
        reads each layout caption's image header."""
        import utils.layout_captions as layout_module
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        files = {'a': caption, 'b': caption, 'c': 'plain'}
        overrides = dict(resolutions=[64, 128], cache_shuffle_num=3, reuse_metadata_cache=reuse)
        first = self.cache(self.make(tmp_path, files, **overrides))
        expected = {tuple(leaf.size_bucket): self.encoded(leaf, 'a') for leaf in first}
        calls = []
        real = layout_module.media_size
        monkeypatch.setattr(layout_module, 'media_size', lambda spec: calls.append(spec) or real(spec))
        again = self.make(tmp_path, files, **overrides)
        again.cache_metadata(trust_cache=True)
        for bucket in again.size_bucket_datasets if again.use_size_buckets else again.ar_bucket_datasets:
            bucket.cache_latents(None, trust_cache=True)
        again.cache_text_embeddings(None, 0)
        assert calls == []
        assert {tuple(leaf.size_bucket): self.encoded(leaf, 'a') for leaf in leaves(again)} == expected

    @pytest.mark.parametrize('reuse', [False, True])
    def test_regenerate_cache_rewrites_for_a_replaced_image(self, tmp_path, monkeypatch, reuse):
        """The kept rewritten captions are rebuilt by --regenerate_cache: an image replaced
        under the same name, at another size, gets boxes for its new size. The metadata need
        not change with it (it records buckets, not sizes), so this is the stale case."""
        import utils.layout_captions as layout_module
        caption = moss_caption([{'bbox': [-20, -40, 20, 40]}])
        (leaf,) = self.cache(self.make(tmp_path, {'a': caption}, size_buckets=[[64, 64, 1]],
                                       reuse_metadata_cache=reuse))
        assert self.encoded(leaf, 'a') == self.expected(caption, (64, 64, 1), src=(120, 80))
        # The replacement, as the header reads it (the pixels and the metadata stay the same).
        monkeypatch.setattr(layout_module, 'media_size', lambda spec: (120, 100))
        monkeypatch.setattr(dataset_module, '_regenerated_paths', set())  # a new launch
        (leaf,) = self.cache(self.make(tmp_path, {'a': caption}, size_buckets=[[64, 64, 1]],
                                       reuse_metadata_cache=reuse))
        assert self.encoded(leaf, 'a') == self.expected(caption, (64, 64, 1), src=(120, 100))
        assert self.expected(caption, (64, 64, 1), src=(120, 100)) != self.expected(caption, (64, 64, 1))
        assert not list((tmp_path / 'imgs' / 'cache').rglob('.flattened*'))  # no temporaries left

    def test_regenerating_for_several_text_encoders_rewrites_once(self, tmp_path, monkeypatch):
        """Each text encoder caches from the same flattened captions; --regenerate_cache
        rebuilds them for the first only (the first one's dataset still has them open)."""
        import utils.layout_captions as layout_module
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        ds = self.make(tmp_path, {'a': caption}, size_buckets=[[64, 64, 1]])
        ds.cache_metadata(regenerate_cache=True)
        for bucket in ds.size_bucket_datasets:
            bucket.cache_latents(_stub_latents, regenerate_cache=True)
        monkeypatch.setattr(dataset_module, '_regenerated_paths', set())
        calls = []
        real = layout_module.media_size
        monkeypatch.setattr(layout_module, 'media_size', lambda spec: calls.append(spec) or real(spec))
        ds.cache_text_embeddings(_stub_text_encoder, 1, regenerate_cache=True)
        ds.cache_text_embeddings(_stub_text_encoder, 2, regenerate_cache=True)
        assert len(calls) == 1
        (leaf,) = leaves(ds)
        assert [decode(t.get_text_embeddings(tuple(leaf.metadata_dataset['image_spec'][0]), 0)['embedding'])
                for t in leaf.text_embedding_datasets] == [self.expected(caption, (64, 64, 1))] * 2

    def test_leftovers_of_dead_processes_are_removed(self, tmp_path):
        import datasets as hf_datasets
        import socket
        host = socket.gethostname()
        path = tmp_path / 'flattened_captions_x'
        dead = 2 ** 22 + 12345  # above any pid_max default
        stale = [tmp_path / f'.{path.name}.partial-{host}-{dead}', tmp_path / f'.{path.name}.previous-{host}-{dead}']
        kept = [tmp_path / f'.{path.name}.partial-otherhost-{dead}',
                tmp_path / f'.{path.name}.partial-{host}-{os.getpid()}x', tmp_path / '.unrelated']
        live = tmp_path / f'.{path.name}.partial-{host}-1'  # pid 1 is alive
        for d in stale + kept + [live]:
            d.mkdir()
        dataset_module._save_dataset_atomically(hf_datasets.Dataset.from_dict({'a': [1]}), path)
        assert not any(d.exists() for d in stale)
        assert all(d.exists() for d in kept + [live])
        assert hf_datasets.load_from_disk(str(path))['a'] == [1]

    @staticmethod
    def manager(encoder, requires_control_file=False):
        manager = dataset_module.DatasetManager.__new__(dataset_module.DatasetManager)
        # Not an nn.Module: a ComfyUI model wrapper that loads itself when first needed.
        manager.submodels = [None, type('Lazy', (), {'load_model_if_needed': lambda self: None})()]
        manager.call_text_encoder_fns = [encoder]
        manager.te_fn_requires_control_file = [requires_control_file]
        manager.recent_text_encodings = dataset_module._RecentTextEncodings()
        return manager

    @staticmethod
    def send(manager, captions, control_file=None):
        sent = []

        class Pipe:
            def send(self, value):
                sent.append(value)

        manager._handle_task((1, captions, [False] * len(captions), control_file, Pipe()))
        return sent[0]

    def test_identical_variants_are_encoded_once_across_batches(self):
        """cache_shuffle_num makes N copies of a layout caption (it is never shuffled). They
        are adjacent rows, which caching_batch_size (1 by default) and the worker processes
        split into separate batches; each is still encoded once."""
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        encoded = []

        def counting_encoder(captions, is_video):
            encoded.extend(captions)
            return _stub_text_encoder({'caption': captions}, 0)

        manager = self.manager(counting_encoder)
        # caching_batch_size = 1: one task per row.
        rows = [caption] * 3 + ['a, b', 'b, a', 'a, b']
        outputs = [self.send(manager, [row]) for row in rows]
        assert encoded == [caption, 'a, b', 'b, a']
        assert [decode(o['embedding'][0]) for o in outputs] == rows
        # And within one batch.
        encoded.clear()
        out = self.send(manager, [caption + ' ', caption + ' ', 'a, b'])
        assert encoded == [caption + ' ']
        assert [decode(e) for e in out['embedding']] == [caption + ' ', caption + ' ', 'a, b']

    def test_a_dataset_without_control_files_still_encodes(self):
        """A text encoder that takes control_file (Qwen-Image) gets None for a dataset without
        control images; repeats are still found by the other columns."""
        seen = []

        def qwen_like(captions, is_video, control_file):
            seen.append((list(captions), control_file))
            return _stub_text_encoder({'caption': captions}, 0)

        manager = self.manager(qwen_like, requires_control_file=True)
        out = self.send(manager, ['x', 'x', 'y'])
        assert seen == [(['x', 'y'], None)]
        assert [decode(e) for e in out['embedding']] == ['x', 'x', 'y']
        seen.clear()
        out = self.send(manager, ['x', 'z'], control_file=['c1.png', 'c2.png'])
        assert seen == [(['x', 'z'], ['c1.png', 'c2.png'])]  # another control image: another row
        out = self.send(manager, ['x'], control_file=['c1.png'])
        assert len(seen) == 1 and decode(out['embedding'][0]) == 'x'

    def test_recent_encodings_handle_batched_tensors_bounds_and_odd_results(self):
        import torch
        calls = []

        def batched(captions, is_video):
            calls.append(list(captions))
            return {'embeds': torch.stack([torch.full((2,), float(len(c))) for c in captions]),
                    'mask': [torch.ones(1)] * len(captions)}

        recent = dataset_module._RecentTextEncodings()
        out = recent.encode(0, batched, [['aa', 'b', 'aa'], [False, False, False]])
        assert calls == [['aa', 'b']]
        assert [e[0].item() for e in out['embeds']] == [2.0, 1.0, 2.0] and len(out['mask']) == 3
        # Rows that differ in another column, or in the encoder, are different rows.
        calls.clear()
        recent.encode(0, batched, [['aa', 'aa'], [True, True]])
        recent.encode(1, batched, [['aa'], [False]])
        assert calls == [['aa'], ['aa']]
        # A batch with nothing new is not encoded; one with no repeats is returned as encoded.
        calls.clear()
        recent.encode(0, batched, [['b'], [False]])
        assert calls == []
        out = recent.encode(0, batched, [['new1', 'new2'], [False, False]])
        assert torch.is_tensor(out['embeds']) and out['embeds'].shape == (2, 2)
        # Bounded in bytes: the oldest rows are forgotten.
        small = dataset_module._RecentTextEncodings(limit_bytes=3 * (2 * 4 + 4))
        for c in ['p', 'q', 'r', 's']:
            small.encode(0, batched, [[c], [False]])
        calls.clear()
        small.encode(0, batched, [['p'], [False]])
        small.encode(0, batched, [['s'], [False]])
        assert calls == [['p']] and small.size <= small.limit_bytes

        def not_per_row(captions, is_video):
            calls.append(list(captions))
            return {'pooled': torch.zeros(5)}

        calls.clear()
        recent = dataset_module._RecentTextEncodings()
        out = recent.encode(0, not_per_row, [['x', 'x', 'y'], [False] * 3])
        assert calls == [['x', 'y'], ['x', 'x', 'y']] and out['pooled'].shape == (5,)
        recent.encode(0, not_per_row, [['x'], [False]])
        assert calls == [['x', 'y'], ['x', 'x', 'y'], ['x']]  # nothing was remembered from it

    @pytest.mark.parametrize('reuse', [False, True])
    def test_caption_text_matches_the_cached_embedding(self, tmp_path, reuse):
        """With cached embeddings the batch's caption is the text that was encoded: the layout
        caption rewritten for this bucket, not the file's text."""
        caption = moss_caption([{'bbox': [-20, 10, 20, 40]}])
        ds = self.make(tmp_path, {'a': caption, 'b': 'plain'}, caption_prefix='p, ', reuse_metadata_cache=reuse)
        (leaf,) = self.cache(ds)
        seen = {}
        for i in range(len(leaf)):
            item = leaf[i]
            seen[decode(item['embedding'])] = item['caption']
        assert seen == {self.expected(caption, tuple(leaf.size_bucket)[-3:], 'p, '): self.expected(
            caption, tuple(leaf.size_bucket)[-3:], 'p, '), 'p, plain': 'p, plain'}

    @pytest.mark.parametrize('reuse', [False, True])
    def test_plain_captions_keep_the_embeddings_of_a_model_without_layout_captions(self, tmp_path, reuse):
        """Ideogram 4 declared a layout format after people had cached plain-caption datasets
        with it; the text cache of a dataset with no JSON caption is unchanged by that."""
        files = {'a': 'plain one', 'b': 'plain two'}
        self.cache(self.make(tmp_path, files, layout_format=None, reuse_metadata_cache=reuse))
        log = tmp_path / 'encoded.txt'
        ds = self.make(tmp_path, files, layout_format='ideogram4', reuse_metadata_cache=reuse)
        ds.cache_metadata(regenerate_cache=False)
        for bucket in ds.ar_bucket_datasets:
            bucket.cache_latents(_stub_latents, regenerate_cache=False)
        # The encoder runs in a worker process, so it reports through a file.
        ds.cache_text_embeddings(functools.partial(_logging_text_encoder, log), 0, regenerate_cache=False)
        assert not log.exists() or log.read_text(encoding='utf-8') == ''
        (leaf,) = leaves(ds)
        assert self.encoded(leaf, 'a') == 'plain one'
        # The log does see an encode: a caption that is not in the old cache.
        ds = self.make(tmp_path, {'a': 'plain one', 'b': 'new text'}, layout_format='ideogram4',
                       reuse_metadata_cache=reuse)
        ds.cache_metadata(regenerate_cache=reuse)  # a profile snapshot is refreshed only on request
        for bucket in ds.ar_bucket_datasets:
            bucket.cache_latents(_stub_latents, regenerate_cache=False)
        ds.cache_text_embeddings(functools.partial(_logging_text_encoder, log), 0, regenerate_cache=False)
        assert 'new text' in log.read_text(encoding='utf-8')


class TestModelsDeclareTheirFormat:
    def test_dataset_passes_the_model_format_to_each_directory(self, monkeypatch):
        seen = []

        class Recorder:
            def __init__(self, *args, **kwargs):
                seen.append(kwargs.get('layout_caption_format'))

        class Model:
            name = 'm'
            framerate = None
            pixels_round_to_multiple = 16
            layout_caption_format = 'ideogram4'

            def get_text_encoders(self):
                return [object()]

            def vae_cache_key(self):
                return ''

        monkeypatch.setattr(dataset_module, 'DirectoryDataset', Recorder)
        config = {'directory': [{'path': 'a'}, {'path': 'b'}]}
        dataset_module.Dataset(config, Model(), skip_dataset_validation=True)
        assert seen == ['ideogram4', 'ideogram4']
        del Model.layout_caption_format
        seen.clear()
        dataset_module.Dataset(config, Model(), skip_dataset_validation=True)
        assert seen == [None, None]

    def test_layout_models(self):
        import ast
        declared = {}
        for name in ('moss_image', 'ideogram4'):
            tree = ast.parse((REPO / 'models' / f'{name}.py').read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and getattr(node.targets[0], 'id', None) == 'layout_caption_format'):
                    declared[name] = node.value.value
        assert declared == {'moss_image': 'moss_image', 'ideogram4': 'ideogram4'}
