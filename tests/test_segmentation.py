"""Unit tests for the segmentation module."""
from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.char_segmenter import segment_chars
from src.segmentation.line_segmenter import segment_lines
from src.segmentation.region_filters import filter_regions
from src.segmentation.word_segmenter import segment_words


# ── synthetic image helpers ────────────────────────────────────────────────────

def _make_two_line_image() -> np.ndarray:
    """White image with two horizontal black bars (one per text line)."""
    img = np.full((80, 256), 255, dtype=np.uint8)
    img[10:22, 10:246] = 0   # line 1
    img[45:57, 10:246] = 0   # line 2
    return img


def _make_three_word_line() -> np.ndarray:
    """Single-line image with three separated word-like blobs."""
    img = np.full((32, 200), 255, dtype=np.uint8)
    img[5:25, 5:40] = 0    # word 1
    img[5:25, 60:100] = 0  # word 2
    img[5:25, 130:180] = 0 # word 3
    return img


def _make_word_with_chars() -> np.ndarray:
    """Word image containing three separate character blobs."""
    img = np.full((32, 90), 255, dtype=np.uint8)
    img[5:25, 5:20] = 0   # char 1
    img[5:25, 30:50] = 0  # char 2
    img[5:25, 60:80] = 0  # char 3
    return img


# ── line segmenter ────────────────────────────────────────────────────────────

def test_segment_lines_returns_list():
    img = _make_two_line_image()
    lines, bboxes = segment_lines(img)
    assert isinstance(lines, list)
    assert isinstance(bboxes, list)


def test_segment_lines_finds_lines():
    img = _make_two_line_image()
    lines, bboxes = segment_lines(img)
    assert len(lines) >= 1
    assert len(lines) == len(bboxes)


def test_segment_lines_crops_correct_height():
    img = _make_two_line_image()
    lines, bboxes = segment_lines(img)
    for line_img, (x, y, w, h) in zip(lines, bboxes):
        assert line_img.shape[0] == h


def test_segment_lines_requires_2d():
    with pytest.raises(ValueError):
        segment_lines(np.zeros((32, 32, 3), dtype=np.uint8))


# ── word segmenter ────────────────────────────────────────────────────────────

def test_segment_words_finds_words():
    img = _make_three_word_line()
    words, bboxes = segment_words(img)
    assert len(words) >= 1
    assert len(words) == len(bboxes)


def test_segment_words_requires_2d():
    with pytest.raises(ValueError):
        segment_words(np.zeros((32, 32, 3), dtype=np.uint8))


# ── char segmenter ────────────────────────────────────────────────────────────

def test_segment_chars_finds_chars():
    img = _make_word_with_chars()
    chars, bboxes = segment_chars(img)
    assert len(chars) >= 1
    assert len(chars) == len(bboxes)


def test_segment_chars_sorted_left_to_right():
    img = _make_word_with_chars()
    _, bboxes = segment_chars(img)
    xs = [b[0] for b in bboxes]
    assert xs == sorted(xs)


def test_segment_chars_requires_2d():
    with pytest.raises(ValueError):
        segment_chars(np.zeros((32, 32, 3), dtype=np.uint8))


# ── region filter ─────────────────────────────────────────────────────────────

def test_filter_removes_too_small():
    bboxes = [(0, 0, 2, 2), (0, 0, 10, 10), (0, 0, 20, 20)]
    result = filter_regions(bboxes, min_area=50)
    for _, _, w, h in result:
        assert w * h >= 50


def test_filter_removes_bad_aspect_ratio():
    bboxes = [
        (0, 0, 1, 100),   # very tall (aspect 0.01 → eliminated)
        (0, 0, 10, 10),   # square (aspect 1.0 → keep)
        (0, 0, 100, 1),   # very wide (aspect 100 → eliminated)
    ]
    result = filter_regions(bboxes, min_area=1, min_aspect=0.1, max_aspect=10.0)
    assert len(result) == 1
    assert result[0] == (0, 0, 10, 10)


def test_filter_empty_input():
    assert filter_regions([]) == []
