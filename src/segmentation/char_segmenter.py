"""Character segmentation from word images using external contours."""
from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.region_filters import BBox, filter_regions
from src.utils.logger import get_logger

log = get_logger(__name__)


def segment_chars(
    word_img: np.ndarray,
    min_char_area: int = 5,
    min_aspect: float = 0.05,
    max_aspect: float = 10.0,
) -> tuple[list[np.ndarray], list[BBox]]:
    """Isolate individual characters from a word image using contour detection.

    Parameters
    ----------
    word_img:
        Grayscale ``uint8`` image of a single word.
    min_char_area:
        Minimum pixel area for a character candidate.
    min_aspect, max_aspect:
        Allowed ``w / h`` ratio range.

    Returns
    -------
    (char_images, bboxes)
        ``char_images`` — cropped character images.
        ``bboxes`` — ``(x, y, w, h)`` relative to *word_img*, sorted left-to-right.
    """
    if word_img.ndim != 2:
        raise ValueError("segment_chars expects a 2-D (grayscale) image")

    _, binary = cv2.threshold(word_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_bboxes: list[BBox] = [cv2.boundingRect(c) for c in contours]
    bboxes = filter_regions(
        raw_bboxes,
        min_area=min_char_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )
    # Sort left-to-right by x coordinate
    bboxes = sorted(bboxes, key=lambda b: b[0])

    char_images: list[np.ndarray] = []
    for x, y, cw, ch in bboxes:
        char_images.append(word_img[y : y + ch, x : x + cw])

    log.debug("segment_chars: found %d chars", len(char_images))
    return char_images, bboxes
