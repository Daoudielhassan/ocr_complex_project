"""Bounding-box filter to discard noise, stamps, and artefacts."""
from __future__ import annotations

from typing import Sequence

from src.utils.logger import get_logger

log = get_logger(__name__)

BBox = tuple[int, int, int, int]  # (x, y, w, h)


def filter_regions(
    bboxes: Sequence[BBox],
    min_area: int = 10,
    max_area: int | None = None,
    min_aspect: float = 0.05,
    max_aspect: float = 20.0,
) -> list[BBox]:
    """Remove bounding boxes that are too small, too large, or have extreme aspect ratios.

    Parameters
    ----------
    bboxes:
        Input bounding boxes as ``(x, y, w, h)``.
    min_area:
        Minimum pixel area (``w * h``) to keep.
    max_area:
        Maximum pixel area to keep. ``None`` means no upper limit.
    min_aspect:
        Minimum allowed ``w / h`` ratio.
    max_aspect:
        Maximum allowed ``w / h`` ratio.

    Returns
    -------
    list[BBox]
        Filtered list in the same order as *bboxes*.
    """
    result: list[BBox] = []
    for x, y, w, h in bboxes:
        area = w * h
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        aspect = w / h if h > 0 else 0.0
        if aspect < min_aspect or aspect > max_aspect:
            continue
        result.append((x, y, w, h))

    log.debug("filter_regions: %d → %d boxes kept", len(bboxes), len(result))
    return result
