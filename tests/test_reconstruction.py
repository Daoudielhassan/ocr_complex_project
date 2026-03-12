"""Unit tests for the text reconstruction module."""
from __future__ import annotations

import pytest

from src.reconstruction.reading_order import sort_reading_order
from src.reconstruction.text_rebuilder import rebuild_text


# ── rebuild_text ──────────────────────────────────────────────────────────────

def test_rebuild_text_single_block():
    predictions = [
        [["H", "e", "l", "l", "o", " "], ["W", "o", "r", "l", "d"]],
    ]
    result = rebuild_text(predictions)
    assert "Hello" in result
    assert "World" in result


def test_rebuild_text_multiple_blocks():
    predictions = [
        [["F", "o", "o", " "]],
        [["B", "a", "r"]],
    ]
    result = rebuild_text(predictions)
    assert "Foo" in result
    assert "Bar" in result
    # Blocks separated by blank line
    assert "\n\n" in result


def test_rebuild_text_strips_trailing_spaces():
    predictions = [
        [["A", " ", "B", " ", " "]],
    ]
    result = rebuild_text(predictions)
    assert not result.endswith(" ")


def test_rebuild_text_empty_input():
    assert rebuild_text([]) == ""


def test_rebuild_text_all_empty_lines():
    predictions = [
        [[], [], []],
    ]
    assert rebuild_text(predictions) == ""


def test_rebuild_text_returns_str():
    assert isinstance(rebuild_text([[[" "]]]), str)


# ── sort_reading_order ────────────────────────────────────────────────────────

def test_sort_reading_order_empty():
    assert sort_reading_order([]) == []


def test_sort_reading_order_single_region():
    regions = [(5, 10, 30, 20)]
    assert sort_reading_order(regions) == regions


def test_sort_reading_order_top_to_bottom():
    # Three regions at different y positions
    regions = [(0, 100, 50, 20), (0, 10, 50, 20), (0, 60, 50, 20)]
    sorted_r = sort_reading_order(regions)
    ys = [r[1] for r in sorted_r]
    assert ys == sorted(ys)


def test_sort_reading_order_left_to_right_same_row():
    regions = [(100, 10, 50, 20), (10, 10, 50, 20), (200, 10, 50, 20)]
    sorted_r = sort_reading_order(regions)
    xs = [r[0] for r in sorted_r]
    assert xs == sorted(xs)


def test_sort_reading_order_n_cols_hint():
    # Provide n_cols=1 forcing a simple top-to-bottom sort
    regions = [(200, 20, 50, 20), (10, 10, 50, 20)]
    sorted_r = sort_reading_order(regions, n_cols=1)
    assert sorted_r[0][1] <= sorted_r[1][1]  # first region has smaller y
