"""Tests for the caption corpus: preprocessing, multiline .txt, export and distill wiring."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import toml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.caption_corpus import format_for, read_corpus, write_corpus  # noqa: E402
from utils.captions import (  # noqa: E402
    drop_tags,
    enumerate_captions,
    preprocess_caption,
    read_caption_file,
    shuffle_captions,
    split_tag_prefix,
)


class TestTagPrefix:
    def test_empty_marker_treats_everything_as_tags(self):
        assert split_tag_prefix('a, b', '') == ('a, b', True)

    def test_marker_is_stripped_and_flags_the_caption(self):
        assert split_tag_prefix('Special: a, b', 'Special: ') == ('a, b', True)

    def test_unmarked_caption_is_left_alone(self):
        body, is_tag = split_tag_prefix('A photo of a cat.', 'Special: ')
        assert (body, is_tag) == ('A photo of a cat.', False)

    def test_matching_ignores_case(self):
        for written in ('Special: a, b', 'special: a, b', 'SPECIAL: a, b', 'SpEcIaL: a, b'):
            body, is_tag = split_tag_prefix(written, 'Special:')
            assert (body, is_tag) == ('a, b', True), written

    def test_marker_in_the_config_may_carry_whitespace(self):
        assert split_tag_prefix('Special: a, b', '  Special:  ') == ('a, b', True)

    def test_whitespace_after_the_marker_is_stripped(self):
        assert split_tag_prefix('Special:\t  a, b', 'Special:') == ('a, b', True)
        assert split_tag_prefix('Special:a, b', 'Special: ') == ('a, b', True)

    def test_whitespace_only_marker_counts_as_unset(self):
        assert split_tag_prefix('a, b', '   ') == ('a, b', True)

    def test_a_caption_merely_containing_the_marker_is_not_matched(self):
        body, is_tag = split_tag_prefix('A photo of a special: occasion', 'Special:')
        assert (body, is_tag) == ('A photo of a special: occasion', False)

    def test_stripped_marker_survives_the_full_pipeline(self):
        assert preprocess_caption('SPECIAL:   a, b ', prefix_tag_caption='Special: ') == 'a, b'

    def test_marker_never_reaches_training(self):
        # The whole point: the model must see "a, b", never "Special: a, b".
        assert preprocess_caption('Special: a, b', prefix_tag_caption='Special: ') == 'a, b'

    def test_natural_language_is_not_shuffled_or_dropped(self):
        text = 'A cat, sitting, on a mat.'
        for _ in range(50):
            out = preprocess_caption(
                text, prefix_tag_caption='Special: ', shuffle=True, tag_dropout_rate=0.9,
            )
            assert out == text

    def test_caption_prefix_goes_on_after_stripping(self):
        out = preprocess_caption('Special: a, b', caption_prefix='anime, ', prefix_tag_caption='Special: ')
        assert out == 'anime, a, b'


class TestTagDropout:
    def test_zero_rate_is_identity(self):
        tags = ['a', 'b', 'c']
        assert drop_tags(list(tags), 0.0) == tags

    def test_single_tag_is_never_dropped(self):
        assert drop_tags(['only'], 1.0) == ['only']

    def test_never_produces_an_empty_caption(self):
        # An all-dropped caption is the unconditional embedding. Minting those by accident
        # would change the conditioning ratio the trainer sets via UNCOND_FRACTION.
        rng = random.Random(0)
        for _ in range(500):
            out = drop_tags(['a', 'b', 'c'], 1.0, rng)
            assert len(out) >= 1

    def test_rate_actually_drops_tags(self):
        rng = random.Random(1234)
        tags = [str(i) for i in range(20)]
        lengths = [len(drop_tags(list(tags), 0.5, rng)) for _ in range(200)]
        mean = sum(lengths) / len(lengths)
        assert 8 < mean < 12, f'expected ~10 surviving tags at rate 0.5, got {mean}'

    def test_order_is_preserved_when_not_shuffling(self):
        rng = random.Random(7)
        out = drop_tags(['a', 'b', 'c', 'd', 'e'], 0.3, rng)
        assert out == sorted(out, key=['a', 'b', 'c', 'd', 'e'].index)


class TestShuffleCaptionsBackCompat:
    def test_defaults_are_byte_identical_to_the_old_behaviour(self):
        captions = ['a, b, c', 'd, e']
        assert shuffle_captions(captions) == captions
        assert shuffle_captions(captions, 0, ', ', 'pre ') == ['pre a, b, c', 'pre d, e']

    def test_count_still_expands_to_count_variants(self):
        out = shuffle_captions(['a, b, c'], 3)
        assert len(out) == 3
        for variant in out:
            assert sorted(variant.split(', ')) == ['a', 'b', 'c']

    def test_dropout_alone_produces_one_variant_per_caption(self):
        out = shuffle_captions(['a, b, c, d'], 0, ', ', '', '', 0.5)
        assert len(out) == 1

    def test_marker_stripped_even_with_no_shuffle_or_dropout(self):
        assert shuffle_captions(['T: a, b'], 0, ', ', '', 'T: ', 0.0) == ['a, b']


class TestMultilineCaptionFile:
    def test_default_keeps_the_whole_file_as_one_caption(self, tmp_path):
        f = tmp_path / 'a.txt'
        f.write_text('line one\nline two\n')
        assert read_caption_file(f) == ['line one\nline two']

    def test_opt_in_splits_into_one_caption_per_line(self, tmp_path):
        f = tmp_path / 'a.txt'
        f.write_text('line one\nline two\n')
        assert read_caption_file(f, multiline_captions=True) == ['line one', 'line two']

    def test_blank_lines_are_dropped(self, tmp_path):
        f = tmp_path / 'a.txt'
        f.write_text('one\n\n  \ntwo\n')
        assert read_caption_file(f, multiline_captions=True) == ['one', 'two']

    def test_whitespace_only_file_yields_nothing(self, tmp_path):
        f = tmp_path / 'a.txt'
        f.write_text('\n\n')
        assert read_caption_file(f, multiline_captions=True) == []


def _make_dataset(tmp_path, captions_by_image, extra_directory=None, extra_dataset=None):
    d = tmp_path / 'imgs'
    d.mkdir()
    for name, text in captions_by_image.items():
        (d / f'{name}.jpg').write_bytes(b'not really a jpeg')
        (d / f'{name}.txt').write_text(text)
    directory = {'path': str(d), 'resolutions': [512]}
    directory.update(extra_directory or {})
    config = {'resolutions': [512], 'directory': [directory]}
    config.update(extra_dataset or {})
    return config


class TestEnumerateCaptions:
    def test_multiline_is_off_by_default(self, tmp_path):
        config = _make_dataset(tmp_path, {'a': 'one\ntwo'})
        assert enumerate_captions(config) == ['one\ntwo']

    def test_multiline_flag_splits(self, tmp_path):
        config = _make_dataset(tmp_path, {'a': 'one\ntwo'}, extra_directory={'multiline_captions': True})
        assert sorted(enumerate_captions(config)) == ['one', 'two']

    def test_multiline_can_be_set_dataset_wide(self, tmp_path):
        config = _make_dataset(tmp_path, {'a': 'one\ntwo'}, extra_dataset={'multiline_captions': True})
        assert sorted(enumerate_captions(config)) == ['one', 'two']

    def test_apply_shuffle_false_keeps_the_marker(self, tmp_path):
        config = _make_dataset(
            tmp_path, {'a': 'T: x, y'},
            extra_directory={'prefix_tag_caption': 'T: ', 'cache_shuffle_num': 4},
        )
        raw = enumerate_captions(config, apply_shuffle=False)
        assert raw == ['T: x, y'], 'the corpus must stay faithful to the dataset'

    def test_apply_shuffle_true_strips_and_expands(self, tmp_path):
        config = _make_dataset(
            tmp_path, {'a': 'T: x, y'},
            extra_directory={'prefix_tag_caption': 'T: ', 'cache_shuffle_num': 4},
        )
        out = enumerate_captions(config)
        assert len(out) == 4
        for variant in out:
            assert not variant.startswith('T: ')
            assert sorted(variant.split(', ')) == ['x', 'y']

    def test_empty_multiline_file_is_skipped_not_turned_into_an_empty_caption(self, tmp_path):
        config = _make_dataset(
            tmp_path, {'a': '\n\n', 'b': 'real'},
            extra_directory={'multiline_captions': True},
        )
        assert enumerate_captions(config) == ['real']


class TestCorpusRoundTrip:
    @pytest.mark.parametrize('fmt', ['jsonl', 'csv', 'txt'])
    def test_plain_captions_survive(self, tmp_path, fmt):
        captions = ['a, b, c', 'A photo of a cat.', 'T: x, y']
        path = tmp_path / f'c.{fmt}'
        write_corpus(path, [{'caption': c} for c in captions])
        assert read_corpus(path) == captions

    @pytest.mark.parametrize('fmt', ['jsonl', 'csv', 'txt'])
    def test_newlines_and_backslashes_survive(self, tmp_path, fmt):
        # A .txt sidecar read without multiline_captions is one caption containing newlines,
        # so the corpus has to carry them or it silently rewrites the dataset.
        captions = ['first\nsecond', r'back\slash', 'trailing\r\ncarriage', 'comma, "quoted"']
        path = tmp_path / f'c.{fmt}'
        write_corpus(path, [{'caption': c} for c in captions])
        assert read_corpus(path) == captions

    @pytest.mark.parametrize('fmt', ['jsonl', 'csv'])
    def test_count_expands_back_to_the_original_distribution(self, tmp_path, fmt):
        path = tmp_path / f'c.{fmt}'
        write_corpus(path, [{'caption': 'common', 'count': 5}, {'caption': 'rare', 'count': 1}])
        got = read_corpus(path)
        assert got.count('common') == 5 and got.count('rare') == 1

    def test_unicode_survives(self, tmp_path):
        captions = ['1girl, ロングヘア', 'chào bạn, dấu tiếng Việt']
        path = tmp_path / 'c.jsonl'
        write_corpus(path, [{'caption': c} for c in captions])
        assert read_corpus(path) == captions

    def test_jsonl_accepts_a_bare_string_per_line(self, tmp_path):
        path = tmp_path / 'c.jsonl'
        path.write_text(json.dumps('just a caption') + '\n', encoding='utf-8')
        assert read_corpus(path) == ['just a caption']

    def test_corrupt_jsonl_names_the_line(self, tmp_path):
        path = tmp_path / 'c.jsonl'
        path.write_text('{"caption": "ok"}\nnot json\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match=r'c\.jsonl:2'):
            read_corpus(path)

    def test_jsonl_without_a_caption_field_is_rejected(self, tmp_path):
        path = tmp_path / 'c.jsonl'
        path.write_text('{"text": "wrong field"}\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match='caption'):
            read_corpus(path)


class TestFormatDetection:
    @pytest.mark.parametrize('name,expected', [
        ('a.jsonl', 'jsonl'), ('a.ndjson', 'jsonl'), ('a.CSV', 'csv'), ('a.txt', 'txt'),
    ])
    def test_inferred_from_extension(self, name, expected):
        assert format_for(name) == expected

    def test_explicit_overrides_extension(self):
        assert format_for('a.txt', 'jsonl') == 'jsonl'

    def test_unknown_extension_is_an_error(self):
        with pytest.raises(ValueError, match='Cannot infer'):
            format_for('a.parquet')


class TestExportScript:
    def _run(self, dataset_toml, output, *args):
        return subprocess.run(
            [sys.executable, str(REPO / 'tools/export_caption_corpus.py'),
             '--dataset', str(dataset_toml), '--output', str(output), *args],
            capture_output=True, text=True, cwd=REPO,
        )

    def _dataset_toml(self, tmp_path, captions_by_image, extra_directory=None):
        config = _make_dataset(tmp_path, captions_by_image, extra_directory=extra_directory)
        path = tmp_path / 'dataset.toml'
        path.write_text(toml.dumps(config))
        return path

    def test_exports_every_caption(self, tmp_path):
        dataset = self._dataset_toml(tmp_path, {'a': 'one', 'b': 'two'})
        out = tmp_path / 'corpus.jsonl'
        result = self._run(dataset, out)
        assert result.returncode == 0, result.stderr
        assert sorted(read_corpus(out)) == ['one', 'two']

    def test_multiline_directory_yields_every_line(self, tmp_path):
        dataset = self._dataset_toml(
            tmp_path, {'a': 'first\nsecond\nthird'}, extra_directory={'multiline_captions': True},
        )
        out = tmp_path / 'corpus.jsonl'
        assert self._run(dataset, out).returncode == 0
        assert sorted(read_corpus(out)) == ['first', 'second', 'third']

    def test_dedupe_keeps_the_distribution(self, tmp_path):
        dataset = self._dataset_toml(tmp_path, {'a': 'same', 'b': 'same', 'c': 'other'})
        plain, deduped = tmp_path / 'p.jsonl', tmp_path / 'd.jsonl'
        assert self._run(dataset, plain).returncode == 0
        assert self._run(dataset, deduped, '--dedupe').returncode == 0
        assert len(deduped.read_text().splitlines()) == 2
        assert sorted(read_corpus(plain)) == sorted(read_corpus(deduped))

    def test_dedupe_to_txt_repeats_lines_rather_than_losing_the_count(self, tmp_path):
        dataset = self._dataset_toml(tmp_path, {'a': 'same', 'b': 'same', 'c': 'other'})
        out = tmp_path / 'corpus.txt'
        assert self._run(dataset, out, '--dedupe').returncode == 0
        assert sorted(read_corpus(out)) == ['other', 'same', 'same']

    def test_marker_is_not_stripped_at_export(self, tmp_path):
        dataset = self._dataset_toml(
            tmp_path, {'a': 'T: x, y'}, extra_directory={'prefix_tag_caption': 'T: '},
        )
        out = tmp_path / 'corpus.jsonl'
        assert self._run(dataset, out).returncode == 0
        assert read_corpus(out) == ['T: x, y']

    def test_shuffle_is_not_baked_in(self, tmp_path):
        dataset = self._dataset_toml(
            tmp_path, {'a': 'x, y, z'}, extra_directory={'cache_shuffle_num': 8},
        )
        out = tmp_path / 'corpus.jsonl'
        assert self._run(dataset, out).returncode == 0
        assert read_corpus(out) == ['x, y, z'], 'shuffling must stay a runtime augmentation'

    def test_unknown_extension_fails_loudly(self, tmp_path):
        dataset = self._dataset_toml(tmp_path, {'a': 'one'})
        result = self._run(dataset, tmp_path / 'corpus.parquet')
        assert result.returncode != 0
        assert 'Cannot infer' in (result.stderr + result.stdout)


class TestMultipleMarkers:
    """A flat corpus can mix directories, so several markers must be strippable at once."""

    def test_a_list_of_markers_is_accepted(self):
        assert split_tag_prefix('T1: a, b', ['T1: ', 'T2: ']) == ('a, b', True)
        assert split_tag_prefix('T2: c, d', ['T1: ', 'T2: ']) == ('c, d', True)

    def test_an_unmarked_caption_is_still_left_alone(self):
        assert split_tag_prefix('prose here', ['T1: ', 'T2: ']) == ('prose here', False)

    def test_an_empty_list_means_everything_is_tags(self):
        assert split_tag_prefix('a, b', []) == ('a, b', True)
        assert split_tag_prefix('a, b', '') == ('a, b', True)

    def test_preprocess_strips_whichever_marker_matched(self):
        assert preprocess_caption('T2: a, b', prefix_tag_caption=['T1: ', 'T2: ']) == 'a, b'


class TestRawEnumerationIsFaithful:
    """apply_shuffle=False must produce exactly what a consumer can reconstruct training from."""

    def test_caption_prefix_is_not_baked_in(self, tmp_path):
        # Baking it in would put the prefix in FRONT of the marker, so the consumer's
        # split_tag_prefix would no longer match, silently disabling augmentation and training
        # the marker as a tag.
        config = _make_dataset(
            tmp_path, {'a': 'Special: red, blue'},
            extra_dataset={'caption_prefix': 'anime, ', 'prefix_tag_caption': 'Special: '},
        )
        assert enumerate_captions(config, apply_shuffle=False) == ['Special: red, blue']

    def test_the_round_trip_reproduces_what_training_sees(self, tmp_path):
        """Rebuilding from the corpus must land in the same space training samples from.

        Equality of the two sets would be wrong: training draws cache_shuffle_num times, so
        with few tags it can easily miss an ordering that 200 draws finds. The containment
        direction that matters is that training never produces something the corpus round trip
        cannot -- an earlier version asserted equality and failed 12% of runs.
        """
        config = _make_dataset(
            tmp_path, {'a': 'Special: red, blue, green'},
            extra_dataset={
                'caption_prefix': 'anime, ', 'prefix_tag_caption': 'Special: ',
                'cache_shuffle_num': 4,
            },
        )
        raw = enumerate_captions(config, apply_shuffle=False)
        rebuilt = {
            preprocess_caption(
                raw[0], caption_prefix='anime, ', prefix_tag_caption='Special: ', shuffle=True,
            )
            for _ in range(500)
        }
        training = set(enumerate_captions(config))
        assert training <= rebuilt, f'training produced {training - rebuilt}, unreachable from the corpus'
        assert all(c.startswith('anime, ') and 'Special' not in c for c in rebuilt)
        assert all(sorted(c[len('anime, '):].split(', ')) == ['blue', 'green', 'red'] for c in rebuilt)

    def test_markers_seen_reports_every_directory(self, tmp_path):
        d1, d2 = tmp_path / 'one', tmp_path / 'two'
        for d, text in ((d1, 'T1: a, b'), (d2, 'T2: c, d')):
            d.mkdir()
            (d / 'x.jpg').write_bytes(b'x')
            (d / 'x.txt').write_text(text)
        config = {
            'resolutions': [512],
            'directory': [
                {'path': str(d1), 'prefix_tag_caption': 'T1: '},
                {'path': str(d2), 'prefix_tag_caption': 'T2: '},
            ],
        }
        markers = set()
        raw = enumerate_captions(config, apply_shuffle=False, markers_seen=markers)
        assert sorted(raw) == ['T1: a, b', 'T2: c, d']
        assert markers == {'T1:', 'T2:'}
        # And the reported markers are exactly what makes the corpus usable again.
        assert [preprocess_caption(c, prefix_tag_caption=sorted(markers)) for c in sorted(raw)] == ['a, b', 'c, d']


class TestCountValidation:
    def test_non_integer_count_names_the_line(self, tmp_path):
        path = tmp_path / 'c.jsonl'
        path.write_text('{"caption": "ok"}\n{"caption": "w", "count": "3.5"}\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match=r'c\.jsonl:2.*non-integer'):
            read_corpus(path)

    @pytest.mark.parametrize('bad', [0, -3])
    def test_non_positive_count_is_rejected_rather_than_dropping_the_caption(self, tmp_path, bad):
        path = tmp_path / 'c.jsonl'
        path.write_text(json.dumps({'caption': 'w', 'count': bad}) + '\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match='silently drop'):
            read_corpus(path)

    def test_csv_line_numbers_account_for_the_header(self, tmp_path):
        path = tmp_path / 'c.csv'
        path.write_text('caption,count\nok,1\nbad,x\n', encoding='utf-8')
        with pytest.raises(RuntimeError, match=r'c\.csv:3'):
            read_corpus(path)

    def test_apply_count_false_yields_each_caption_once(self, tmp_path):
        path = tmp_path / 'c.jsonl'
        write_corpus(path, [{'caption': 'a', 'count': 5}, {'caption': 'b', 'count': 2}])
        assert read_corpus(path, apply_count=False) == ['a', 'b']

    def test_num_repeats_is_not_a_corpus_field(self, tmp_path):
        # It is per-directory in the dataset; per caption it would floor sub-1 values to zero.
        path = tmp_path / 'c.csv'
        write_corpus(path, [{'caption': 'a'}])
        assert 'num_repeats' not in path.read_text().splitlines()[0]
