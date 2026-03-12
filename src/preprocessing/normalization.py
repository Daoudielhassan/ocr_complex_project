"""Image normalisation: resolution scaling and padding."""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def normalize(
    img: np.ndarray,
    target_height: int = 1024,
    target_dpi: int | None = None,
    src_dpi: int | None = None,
    padding: int = 0,
) -> np.ndarray:
    """Rescale and optionally pad a document image.

    If *target_dpi* and *src_dpi* are both provided, the image is rescaled
    proportionally to match the target DPI.  Otherwise the image is resized
    so its height equals *target_height* while the aspect ratio is preserved.

    Parameters
    ----------
    img:
        Input ``uint8`` ndarray (grayscale or BGR).
    target_height:
        Desired output height in pixels (DPI-unaware fallback mode).
    target_dpi:
        Desired output DPI.
    src_dpi:
        Source document DPI.
    padding:
        Constant-zero padding (pixels) added to every side after rescaling.

    Returns
    -------
    np.ndarray
        Normalised image.
    """
    h, w = img.shape[:2]

    if target_dpi is not None and src_dpi is not None and src_dpi > 0:
        scale = target_dpi / src_dpi
    else:
        scale = target_height / h if h > 0 else 1.0

    if abs(scale - 1.0) > 0.01:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        log.debug(
            "normalize: resized %dx%d → %dx%d (scale=%.2f)",
            w, h, new_w, new_h, scale,
        )

    if padding > 0:
        img = cv2.copyMakeBorder(
            img, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=0,
        )

    return img
