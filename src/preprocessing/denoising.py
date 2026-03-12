"""Noise-reduction filters."""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def denoise(img: np.ndarray, method: str = "median", **kwargs) -> np.ndarray:
    """Apply a denoising filter to *img*.

    Parameters
    ----------
    img:
        Input ``uint8`` ndarray (grayscale or BGR).
    method:
        One of ``"median"``, ``"gaussian"``, ``"bilateral"``, ``"nlmeans"``.
    **kwargs:
        Extra parameters forwarded to the specific filter.

    Returns
    -------
    np.ndarray
        Filtered image, same shape and dtype as *img*.
    """
    method = method.lower()
    if method == "median":
        ksize = int(kwargs.get("ksize", 3))
        result = cv2.medianBlur(img, ksize)
        log.debug("Median blur ksize=%d applied", ksize)

    elif method == "gaussian":
        ksize = int(kwargs.get("ksize", 3))
        sigma = float(kwargs.get("sigma", 0))
        result = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        log.debug("Gaussian blur ksize=%d sigma=%.1f applied", ksize, sigma)

    elif method == "bilateral":
        d = int(kwargs.get("d", 9))
        sigma_color = float(kwargs.get("sigma_color", 75))
        sigma_space = float(kwargs.get("sigma_space", 75))
        result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        log.debug("Bilateral filter d=%d applied", d)

    elif method == "nlmeans":
        h = float(kwargs.get("h", 10))
        result = cv2.fastNlMeansDenoising(img, None, h=h)
        log.debug("NL-means denoising h=%.1f applied", h)

    else:
        raise ValueError(
            f"Unknown denoise method: {method!r}. "
            "Choose from: median, gaussian, bilateral, nlmeans."
        )

    return result
