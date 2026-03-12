"""Label encoder / decoder with save-load support."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

log = get_logger(__name__)


class LabelManager:
    """Thin wrapper around :class:`~sklearn.preprocessing.LabelEncoder`.

    Adds ``save`` / ``load`` helpers so the encoder can be persisted
    alongside the SVM model.
    """

    def __init__(self) -> None:
        self._le = LabelEncoder()
        self._fitted = False

    # ── fitting ────────────────────────────────────────────────
    def fit(self, labels: Sequence[str]) -> "LabelManager":
        """Fit the encoder on *labels* and return *self*."""
        self._le.fit(labels)
        self._fitted = True
        log.debug(
            "LabelManager fitted: %d classes  first=%s",
            len(self._le.classes_),
            list(self._le.classes_[:5]),
        )
        return self

    # ── encode / decode ────────────────────────────────────────
    def encode(self, labels: Sequence[str]) -> np.ndarray:
        """Convert string labels to integer indices."""
        return self._le.transform(labels)

    def decode(self, indices: Sequence[int]) -> list[str]:
        """Convert integer indices back to string labels."""
        return list(self._le.inverse_transform(indices))

    # ── properties ─────────────────────────────────────────────
    @property
    def classes(self) -> list[str]:
        return list(self._le.classes_)

    @property
    def n_classes(self) -> int:
        return len(self._le.classes_)

    # ── persistence ────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Pickle this :class:`LabelManager` to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.debug("LabelManager saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LabelManager":
        """Load a :class:`LabelManager` from *path*."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LabelManager, got {type(obj)}")
        log.debug("LabelManager loaded from %s", path)
        return obj
