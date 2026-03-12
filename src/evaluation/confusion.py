"""Confusion matrix computation and visualisation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from src.utils.visualization import plot_confusion_matrix
from src.utils.logger import get_logger

log = get_logger(__name__)


def plot_confusion(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    out_path: Path | None = None,
) -> np.ndarray:
    """Compute and optionally save a confusion matrix.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted labels.
    labels:
        Ordered list of class names.  Inferred from data if *None*.
    title:
        Figure title.
    out_path:
        If given, the figure is saved to this path.

    Returns
    -------
    np.ndarray
        Raw integer confusion matrix (shape ``[n, n]``).
    """
    if labels is None:
        unique = sorted(set(list(y_true) + list(y_pred)), key=str)
        labels = [str(u) for u in unique]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels=labels, title=title, out_path=out_path)

    if out_path:
        log.info("Confusion matrix saved to %s", out_path)
    return cm
