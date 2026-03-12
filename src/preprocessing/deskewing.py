"""Document skew correction via horizontal projection-profile variance."""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct the skew angle of a document image.

    The algorithm sweeps a range of candidate angles, computes the variance
    of the horizontal projection profile for each, and picks the angle that
    maximises variance (text rows become widest horizontal bands).

    Parameters
    ----------
    img:
        Grayscale ``uint8`` ndarray.
    max_angle:
        Absolute maximum angle (degrees) to search for skew correction.

    Returns
    -------
    np.ndarray
        Deskewed image with the same shape as *img*.
    """
    angle = _estimate_skew_angle(img, max_angle)
    log.debug("Estimated skew angle: %.2f°", angle)
    if abs(angle) < 0.1:
        return img
    return _rotate(img, angle)


def _estimate_skew_angle(img: np.ndarray, max_angle: float) -> float:
    """Sweep angles; return the one that maximises projection variance."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0
    # Resolution: 0.25° steps between −max_angle and +max_angle
    angles = np.linspace(-max_angle, max_angle, num=int(max_angle * 8) + 1)

    for angle in angles:
        rotated = _rotate(binary, float(angle))
        projection = np.sum(rotated, axis=1).astype(np.float64)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *img* by *angle* degrees around its centre (same shape)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
