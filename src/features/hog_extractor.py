"""HOG descriptor extraction for individual character images."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

from src.utils.logger import get_logger

log = get_logger(__name__)


def extract_hog(
    img: np.ndarray,
    resize_to: tuple[int, int] = (32, 32),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """Extract a HOG descriptor from a single grayscale character image.

    The image is first resized to *resize_to* (height, width) so that all
    descriptors have the same dimensionality regardless of the original
    character size.

    Parameters
    ----------
    img:
        Grayscale ``uint8`` ndarray (any size).
    resize_to:
        ``(height, width)`` to resize to before computing HOG.
    orientations:
        Number of gradient orientation bins.
    pixels_per_cell:
        ``(rows, cols)`` size of each HOG cell.
    cells_per_block:
        Number of cells per normalisation block.
    block_norm:
        Block normalisation method (passed to :func:`skimage.feature.hog`).

    Returns
    -------
    np.ndarray
        1-D ``float32`` feature vector.
    """
    if img.ndim != 2:
        raise ValueError(f"extract_hog expects a 2-D image, got shape {img.shape}")

    img_resized = cv2.resize(
        img,
        (resize_to[1], resize_to[0]),  # cv2 expects (width, height)
        interpolation=cv2.INTER_AREA,
    )
    features = hog(
        img_resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
    )
    return features.astype(np.float32)
