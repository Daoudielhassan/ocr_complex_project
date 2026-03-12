"""Feature normalisation / standardisation wrappers."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)


def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit a :class:`~sklearn.preprocessing.StandardScaler` on *X* and return it."""
    scaler = StandardScaler()
    scaler.fit(X)
    log.debug("Scaler fitted: n_samples=%d  n_features=%d", X.shape[0], X.shape[1])
    return scaler


def transform(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted *scaler* to feature matrix *X*."""
    return scaler.transform(X).astype(np.float32)


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Persist *scaler* to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    log.debug("Scaler saved to %s", path)


def load_scaler(path: str | Path) -> StandardScaler:
    """Load a persisted :class:`~sklearn.preprocessing.StandardScaler` from disk."""
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler, got {type(scaler)}")
    log.debug("Scaler loaded from %s", path)
    return scaler
