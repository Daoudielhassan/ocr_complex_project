"""Full feature-extraction pipeline: resize → HOG → 2-D matrix."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.features.hog_extractor import extract_hog
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_features(
    char_imgs: Sequence[np.ndarray],
    resize_to: tuple[int, int] = (32, 32),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """Build a HOG feature matrix from a list of character images.

    Parameters
    ----------
    char_imgs:
        List of grayscale ``uint8`` character images (any individual sizes).
    resize_to, orientations, pixels_per_cell, cells_per_block, block_norm:
        Forwarded to :func:`~src.features.hog_extractor.extract_hog`.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(n_chars, n_features)``, ``float32``.
        Returns an empty array of shape ``(0,)`` when *char_imgs* is empty.
    """
    if len(char_imgs) == 0:
        return np.empty((0,), dtype=np.float32)

    feature_list = [
        extract_hog(
            img,
            resize_to=resize_to,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
        )
        for img in char_imgs
    ]
    X = np.vstack(feature_list)
    log.debug("build_features: shape=%s", X.shape)
    return X
