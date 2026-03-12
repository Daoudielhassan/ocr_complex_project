"""Detailed error analysis for OCR misclassifications."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def analyze_errors(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    image_paths: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of misclassified character samples.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    image_paths:
        Optional list of image paths aligned with *y_true* / *y_pred*,
        included in the output for visual inspection.

    Returns
    -------
    pd.DataFrame
        Columns: ``index``, ``true``, ``predicted``, ``error_type``
        (and optionally ``image_path``).
    """
    records = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            record: dict = {
                "index": i,
                "true": t,
                "predicted": p,
                "error_type": "substitution",
            }
            if image_paths is not None:
                record["image_path"] = image_paths[i] if i < len(image_paths) else ""
            records.append(record)

    df = pd.DataFrame(records)
    n_errors = len(df)
    n_total = len(list(y_true))
    log.info(
        "analyze_errors: %d errors / %d samples (%.1f%%)",
        n_errors, n_total, 100.0 * n_errors / max(n_total, 1),
    )
    return df
