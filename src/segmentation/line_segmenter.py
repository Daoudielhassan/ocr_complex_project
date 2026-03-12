"""Text-line segmentation using horizontal projection profiles."""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

BBox = tuple[int, int, int, int]  # (x, y, w, h)


def segment_lines(
    img: np.ndarray,
    min_line_height: int = 5,
    gap_threshold: int = 3,
) -> tuple[list[np.ndarray], list[BBox]]:
    """Segment text lines from a document block using horizontal projection.

    Parameters
    ----------
    img:
        Grayscale ``uint8`` image of a text block (or full page).
    min_line_height:
        Minimum number of rows for a run to be considered a line.
    gap_threshold:
        Consecutive blank rows ≤ this value are bridged into the same line.

    Returns
    -------
    (line_images, bboxes)
        ``line_images`` — cropped grayscale images of each line.
        ``bboxes`` — corresponding ``(x, y, w, h)`` relative to *img*.
    """
    if img.ndim != 2:
        raise ValueError("segment_lines expects a 2-D (grayscale) image")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(binary, axis=1)  # row-wise ink pixel count
    in_line = projection > 0

    line_bands = _runs(in_line, min_length=min_line_height, gap=gap_threshold)

    h, w = img.shape
    line_images: list[np.ndarray] = []
    bboxes: list[BBox] = []
    for r0, r1 in line_bands:
        line_images.append(img[r0:r1, 0:w])
        bboxes.append((0, r0, w, r1 - r0))

    log.debug("segment_lines: found %d lines", len(line_images))
    return line_images, bboxes


def _runs(
    mask: np.ndarray,
    min_length: int = 5,
    gap: int = 3,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs for contiguous True runs.

    Runs separated by at most *gap* False values are merged together.
    Only runs of length ≥ *min_length* are kept.
    """
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n:
                if not mask[j]:
                    # Look ahead for a gap
                    gap_end = j
                    while gap_end < n and not mask[gap_end]:
                        gap_end += 1
                    if gap_end - j <= gap:
                        j = gap_end  # bridge the gap
                    else:
                        break
                else:
                    j += 1
            if j - i >= min_length:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs
