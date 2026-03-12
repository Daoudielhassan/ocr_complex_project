"""Reading-order sorter for multi-column document layouts."""
from __future__ import annotations

from src.utils.logger import get_logger

log = get_logger(__name__)

BBox = tuple[int, int, int, int]  # (x, y, w, h)


def sort_reading_order(
    regions: list[BBox],
    n_cols: int | None = None,
) -> list[BBox]:
    """Sort bounding boxes in natural reading order (top-to-bottom, left-to-right).

    For multi-column layouts the regions are grouped by column (x-range), then
    sorted top-to-bottom within each column.

    Parameters
    ----------
    regions:
        List of ``(x, y, w, h)`` bounding boxes.
    n_cols:
        Hint for the number of columns.  ``None`` = auto-detect.

    Returns
    -------
    list[BBox]
        Regions in reading order.
    """
    if not regions:
        return []

    if n_cols is None:
        n_cols = _detect_columns(regions)

    if n_cols <= 1:
        return sorted(regions, key=lambda r: (r[1], r[0]))

    max_x = max(r[0] + r[2] for r in regions)
    min_x = min(r[0] for r in regions)
    col_width = (max_x - min_x) / n_cols

    def col_idx(r: BBox) -> int:
        mid = r[0] + r[2] // 2 - min_x
        idx = int(mid / col_width)
        return max(0, min(idx, n_cols - 1))

    sorted_regions = sorted(regions, key=lambda r: (col_idx(r), r[1]))
    log.debug("sort_reading_order: %d regions, %d column(s)", len(regions), n_cols)
    return sorted_regions


def _detect_columns(regions: list[BBox]) -> int:
    """Heuristic: infer column count from x-midpoint gap analysis."""
    if len(regions) < 4:
        return 1

    min_x = min(r[0] for r in regions)
    max_x = max(r[0] + r[2] for r in regions)
    total_width = max_x - min_x
    if total_width == 0:
        return 1

    midpoints = sorted(r[0] + r[2] // 2 for r in regions)
    page_mid = min_x + total_width / 2

    left_mids = [m for m in midpoints if m < page_mid]
    right_mids = [m for m in midpoints if m >= page_mid]

    if left_mids and right_mids:
        gap = right_mids[0] - left_mids[-1]
        if gap > total_width * 0.15:
            return 2

    return 1
