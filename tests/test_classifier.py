"""Unit tests for the classification module (SVM model + label encoder + training)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.classification.label_encoder import LabelManager
from src.classification.model import SVMModel
from src.classification.train import train


# ── dataset helper ────────────────────────────────────────────────────────────

def _make_dataset(
    n: int = 120,
    n_features: int = 36,
    n_classes: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Linearly separable blobs per class for fast SVM convergence
    X_parts = []
    y_parts = []
    per_class = n // n_classes
    for c in range(n_classes):
        centre = rng.standard_normal(n_features) * 5
        X_parts.append(rng.standard_normal((per_class, n_features)).astype(np.float32) + centre)
        y_parts.append(np.full(per_class, c))
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── SVMModel ──────────────────────────────────────────────────────────────────

def test_svm_fit_predict_returns_correct_length():
    X, y = _make_dataset()
    model = SVMModel(kernel="linear", C=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_svm_predict_raises_if_not_fitted():
    model = SVMModel()
    X, _ = _make_dataset(n=4)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)


def test_svm_predict_proba_shape():
    X, y = _make_dataset()
    model = SVMModel(kernel="linear", probability=True)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_svm_save_load_roundtrip(tmp_path: Path):
    X, y = _make_dataset()
    model = SVMModel(kernel="linear")
    model.fit(X, y)
    expected = model.predict(X)

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = SVMModel.load(path)
    np.testing.assert_array_equal(expected, loaded.predict(X))


def test_svm_load_type_check(tmp_path: Path):
    import pickle
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"not": "an SVMModel"}, f)
    with pytest.raises(TypeError):
        SVMModel.load(bad)


# ── LabelManager ─────────────────────────────────────────────────────────────

def test_label_manager_encode_decode():
    lm = LabelManager().fit(["A", "B", "C"])
    encoded = lm.encode(["A", "C", "B", "A"])
    decoded = lm.decode(encoded)
    assert decoded == ["A", "C", "B", "A"]


def test_label_manager_classes_property():
    lm = LabelManager().fit(["X", "Y", "Z"])
    assert sorted(lm.classes) == ["X", "Y", "Z"]
    assert lm.n_classes == 3


def test_label_manager_save_load(tmp_path: Path):
    lm = LabelManager().fit(["p", "q", "r"])
    path = tmp_path / "encoder.pkl"
    lm.save(path)
    loaded = LabelManager.load(path)
    assert loaded.classes == lm.classes


# ── train() ───────────────────────────────────────────────────────────────────

def test_train_returns_svm_model_and_metrics():
    X, y = _make_dataset(n=80, n_features=36, n_classes=3)
    model, metrics = train(X, y, kernel="linear", C=1.0, cv_folds=3)
    assert isinstance(model, SVMModel)
    assert "cv_accuracy_mean" in metrics
    assert "cv_accuracy_std" in metrics
    assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0


def test_train_on_linearly_separable_data_high_accuracy():
    X, y = _make_dataset(n=120, n_features=36, n_classes=3, seed=42)
    model, metrics = train(X, y, kernel="linear", C=10.0, cv_folds=3)
    # Expect near-perfect CV accuracy on linearly separable blobs
    assert metrics["cv_accuracy_mean"] >= 0.7
