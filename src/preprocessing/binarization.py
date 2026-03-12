"""Binarisation methods: Otsu, Sauvola, Adaptive."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from src.utils.logger import get_logger

log = get_logger(__name__)


def binarize(img: np.ndarray, method: str = "otsu", **kwargs) -> np.ndarray:
    """Convert a grayscale image to a binary (0 / 255) image.

    Parameters
    ----------
    img:
        Grayscale ``uint8`` ndarray.
    method:
        One of ``"otsu"``, ``"sauvola"``, ``"adaptive"``.
    **kwargs:
        Extra keyword arguments forwarded to the chosen implementation.

    Returns
    -------
    np.ndarray
        Binary image (values 0 or 255), same spatial shape as *img*.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got shape {img.shape}")

    method = method.lower()
    if method == "otsu":
        return _otsu(img)
    if method == "sauvola":
        return _sauvola(img, **kwargs)
    if method == "adaptive":
        return _adaptive(img, **kwargs)
    raise ValueError(
        f"Unknown binarisation method: {method!r}. "
        "Choose from: otsu, sauvola, adaptive."
    )


def _otsu(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log.debug("Otsu binarisation applied")
    return binary


def _sauvola(
    img: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
) -> np.ndarray:
    thresh = threshold_sauvola(img, window_size=window_size, k=k)
    binary = (img > thresh).astype(np.uint8) * 255
    log.debug("Sauvola binarisation applied  window=%d  k=%.2f", window_size, k)
    return binary


def _adaptive(
    img: np.ndarray,
    block_size: int = 11,
    C: int = 2,
    method: str = "gaussian",
) -> np.ndarray:
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    binary = cv2.adaptiveThreshold(
        img, 255, adaptive_method, cv2.THRESH_BINARY, block_size, C
    )
    log.debug("Adaptive binarisation applied  block=%d  C=%d", block_size, C)
    return binary
