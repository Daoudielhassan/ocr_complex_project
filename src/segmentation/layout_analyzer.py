"""Document layout analyser: detect text block regions."""
from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.region_filters import BBox, filter_regions
from src.utils.logger import get_logger

log = get_logger(__name__)


def analyze_layout(img: np.ndarray, min_block_area: int = 500) -> list[BBox]:
    """Detect text block bounding boxes in a document image.

    Uses morphological dilation to cluster nearby ink pixels into coherent
    blocks, then extracts connected-component bounding boxes.

    Parameters
    ----------
    img:
        Grayscale (or already binary) ``uint8`` document image.
    min_block_area:
        Minimum pixel area for a block to be retained.

    Returns
    -------
    list of (x, y, w, h)
        Text block bounding boxes sorted top-to-bottom, left-to-right.
    """
    # Ensure binary with ink=255
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge characters into words/lines, then
    # vertically to merge lines into blocks
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(binary, kernel_h)
    dilated = cv2.dilate(dilated, kernel_v)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    bboxes: list[BBox] = []
    for i in range(1, n_labels):  # skip background label 0
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bboxes.append((x, y, w, h))

    bboxes = filter_regions(bboxes, min_area=min_block_area)
    bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
    log.debug("analyze_layout: found %d text blocks", len(bboxes))
    return bboxes
