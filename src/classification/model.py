"""SVM model wrapper: fit, predict, persist."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from src.utils.logger import get_logger

log = get_logger(__name__)


class SVMModel:
    """Thin wrapper around :class:`~sklearn.svm.SVC` with save / load support.

    Parameters
    ----------
    kernel:
        SVM kernel (``"rbf"``, ``"linear"``, ``"poly"``, ``"sigmoid"``).
    C:
        Regularisation parameter.
    gamma:
        Kernel coefficient (``"scale"``, ``"auto"``, or a float).
    probability:
        Whether to enable probability estimates (required for ``predict_proba``).
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        probability: bool = True,
        random_state: int = 42,
    ) -> None:
        self._clf = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )
        self._fitted = False

    # ── training ───────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        """Fit the SVM on feature matrix *X* with integer labels *y*."""
        log.info("Training SVM on %d samples, %d features …", len(X), X.shape[1])
        self._clf.fit(X, y)
        self._fitted = True
        log.info("SVM training complete  support vectors=%d", self._clf.support_vectors_.shape[0])
        return self

    # ── inference ──────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels for *X*."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-probability matrix for *X* (shape ``[n, n_classes]``)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first.")
        return self._clf.predict_proba(X)

    # ── persistence ────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Pickle this :class:`SVMModel` to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SVMModel":
        """Load a :class:`SVMModel` from *path*."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SVMModel, got {type(obj)}")
        log.info("Model loaded from %s", path)
        return obj
