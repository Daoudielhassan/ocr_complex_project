"""Batch character prediction using a pretrained SVM pipeline."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.classification.label_encoder import LabelManager
from src.classification.model import SVMModel
from src.features.feature_pipeline import build_features
from src.features.scaling import transform
from src.utils.logger import get_logger

log = get_logger(__name__)


def predict_chars(
    char_imgs: Sequence[np.ndarray],
    model: SVMModel,
    scaler: StandardScaler,
    label_manager: LabelManager,
    resize_to: tuple[int, int] = (32, 32),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> list[str]:
    """Predict character labels for a list of grayscale character images.

    Parameters
    ----------
    char_imgs:
        Grayscale ``uint8`` character crops.
    model:
        Fitted :class:`~src.classification.model.SVMModel`.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    label_manager:
        Fitted :class:`~src.classification.label_encoder.LabelManager`.
    resize_to, orientations, pixels_per_cell, cells_per_block:
        HOG parameters — must match those used during training.

    Returns
    -------
    list[str]
        Predicted character strings.
    """
    if not char_imgs:
        return []

    X = build_features(
        char_imgs,
        resize_to=resize_to,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    X_scaled = transform(X, scaler)
    y_pred = model.predict(X_scaled)
    labels = label_manager.decode(y_pred)
    # Lowercase labels are stored with a trailing '_' (Windows-safe folder name).
    # Strip it to recover the actual character: 'a_' → 'a', 'z_' → 'z'.
    chars = [lbl.rstrip("_") for lbl in labels]
    log.debug("predict_chars: %d chars → '%s'", len(chars), "".join(chars[:30]))
    return chars
