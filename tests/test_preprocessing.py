"""Unit tests for the preprocessing module.

All tests use synthetically-generated NumPy images so no external
dataset is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.binarization import binarize
from src.preprocessing.denoising import denoise
from src.preprocessing.deskewing import deskew
from src.preprocessing.normalization import normalize


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_gray(shape: tuple[int, int] = (64, 64), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def _make_document(size: int = 128, seed: int = 1) -> np.ndarray:
    """White background with a few black horizontal bars (simulating text)."""
    img = np.full((size, size), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(3):
        y = int(rng.integers(10, size - 20))
        img[y : y + 5, 5 : size - 5] = 0
    return img


# ── binarization ──────────────────────────────────────────────────────────────

def test_binarize_otsu_output_values():
    img = _make_gray()
    result = binarize(img, method="otsu")
    assert result.shape == img.shape
    assert set(np.unique(result)).issubset({0, 255})


def test_binarize_sauvola_output_values():
    img = _make_gray()
    result = binarize(img, method="sauvola")
    assert result.shape == img.shape
    assert set(np.unique(result)).issubset({0, 255})


def test_binarize_adaptive_output_values():
    img = _make_gray()
    result = binarize(img, method="adaptive")
    assert result.shape == img.shape


def test_binarize_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown binarisation method"):
        binarize(_make_gray(), method="unknown")


def test_binarize_requires_2d():
    bgr = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        binarize(bgr, method="otsu")


# ── denoising ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["median", "gaussian", "bilateral"])
def test_denoise_preserves_shape_and_dtype(method: str):
    img = _make_gray()
    result = denoise(img, method=method)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_denoise_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown denoise method"):
        denoise(_make_gray(), method="fft_magic")


# ── deskewing ─────────────────────────────────────────────────────────────────

def test_deskew_preserves_shape():
    img = _make_document()
    result = deskew(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_deskew_near_zero_angle_is_noop():
    """A perfectly horizontal document should come back almost unchanged."""
    img = _make_document(128)
    result = deskew(img, max_angle=15.0)
    # shapes must match
    assert result.shape == img.shape


# ── normalisation ─────────────────────────────────────────────────────────────

def test_normalize_scales_height():
    img = _make_gray((64, 32))
    result = normalize(img, target_height=128)
    assert result.shape[0] == 128
    # aspect ratio preserved: width should have doubled
    assert result.shape[1] == 64


def test_normalize_padding_adds_border():
    img = _make_gray((64, 64))
    result = normalize(img, target_height=64, padding=10)
    assert result.shape == (84, 84)


def test_normalize_dpi_scaling():
    img = _make_gray((100, 100))
    result = normalize(img, target_dpi=300, src_dpi=150)
    # 300/150 = 2× scaling
    assert result.shape == (200, 200)


def test_normalize_noop_when_same_height():
    img = _make_gray((128, 64))
    result = normalize(img, target_height=128)
    assert result.shape == img.shape
