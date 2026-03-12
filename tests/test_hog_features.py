"""Unit tests for the feature extraction module (HOG + scaling)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.features.feature_pipeline import build_features
from src.features.hog_extractor import extract_hog
from src.features.scaling import fit_scaler, load_scaler, save_scaler, transform


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_char(height: int = 32, width: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width), dtype=np.uint8)


# ── HOG extractor ─────────────────────────────────────────────────────────────

def test_extract_hog_returns_1d_float32():
    img = _make_char()
    features = extract_hog(img)
    assert features.ndim == 1
    assert features.dtype == np.float32
    assert len(features) > 0


def test_extract_hog_consistent_shape_across_input_sizes():
    """Different input sizes should yield the same descriptor length after resize."""
    f1 = extract_hog(_make_char(16, 16), resize_to=(32, 32))
    f2 = extract_hog(_make_char(64, 48), resize_to=(32, 32))
    assert f1.shape == f2.shape


def test_extract_hog_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        extract_hog(np.zeros((32, 32, 3), dtype=np.uint8))


def test_extract_hog_custom_params():
    img = _make_char()
    f = extract_hog(img, resize_to=(24, 24), orientations=6,
                    pixels_per_cell=(6, 6), cells_per_block=(2, 2))
    assert f.ndim == 1
    assert len(f) > 0


# ── feature pipeline ──────────────────────────────────────────────────────────

def test_build_features_shape():
    chars = [_make_char(seed=i) for i in range(7)]
    X = build_features(chars)
    expected_dim = len(extract_hog(_make_char()))
    assert X.shape == (7, expected_dim)
    assert X.dtype == np.float32


def test_build_features_empty_list():
    X = build_features([])
    assert X.shape[0] == 0


# ── scaling ───────────────────────────────────────────────────────────────────

def test_fit_and_transform():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 36)).astype(np.float32)
    scaler = fit_scaler(X)
    X_scaled = transform(X, scaler)
    assert X_scaled.shape == X.shape
    assert X_scaled.dtype == np.float32
    # After standard scaling mean ≈ 0, std ≈ 1
    assert abs(X_scaled.mean()) < 0.05


def test_scaler_save_load_roundtrip():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 20)).astype(np.float32)
    scaler = fit_scaler(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scaler.pkl"
        save_scaler(scaler, path)
        loaded = load_scaler(path)

    X_orig = transform(X, scaler)
    X_loaded = transform(X, loaded)
    np.testing.assert_array_almost_equal(X_orig, X_loaded)


def test_load_scaler_type_check(tmp_path: Path):
    """Loading a file that is not a StandardScaler should raise TypeError."""
    import pickle
    bad_path = tmp_path / "bad.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump({"not": "a scaler"}, f)
    with pytest.raises(TypeError):
        load_scaler(bad_path)
