"""Word segmentation from line images using vertical projection profiles."""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

BBox = tuple[int, int, int, int]  # (x, y, w, h)


def segment_words(
    line_img: np.ndarray,
    min_word_width: int = 3,
    gap_threshold: int = 5,
) -> tuple[list[np.ndarray], list[BBox]]:
    """Segment words from a line image using a vertical projection profile.

    Parameters
    ----------
    line_img:
        Grayscale ``uint8`` image of a single text line.
    min_word_width:
        Minimum column-run width (pixels) to keep as a word candidate.
    gap_threshold:
        Column gaps wider than this (pixels) are treated as word separators.

    Returns
    -------
    (word_images, bboxes)
        ``word_images`` — cropped word images.
        ``bboxes`` — ``(x, y, w, h)`` relative to *line_img*, sorted left-to-right.
    """
    if line_img.ndim != 2:
        raise ValueError("segment_words expects a 2-D (grayscale) image")

    h, w = line_img.shape
    _, binary = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(binary, axis=0)  # column-wise ink pixel count
    in_word = projection > 0

    word_bands = _col_runs(in_word, min_length=min_word_width, gap=gap_threshold)

    word_images: list[np.ndarray] = []
    bboxes: list[BBox] = []
    for c0, c1 in word_bands:
        word_images.append(line_img[0:h, c0:c1])
        bboxes.append((c0, 0, c1 - c0, h))

    log.debug("segment_words: found %d words", len(word_images))
    return word_images, bboxes


def _col_runs(
    mask: np.ndarray,
    min_length: int = 3,
    gap: int = 5,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` column ranges for ink-containing runs."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n:
                if not mask[j]:
                    gap_end = j
                    while gap_end < n and not mask[gap_end]:
                        gap_end += 1
                    if gap_end - j <= gap:
                        j = gap_end
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
