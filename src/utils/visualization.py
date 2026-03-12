"""Visualisation helpers for debugging and reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend safe for headless servers


def draw_bboxes(
    img: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes ``(x, y, w, h)`` on a copy of *img*.

    Grayscale images are converted to BGR before drawing so the colour
    is applied correctly.
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bboxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def show_segments(
    segments: Sequence[np.ndarray],
    title: str = "Segments",
    max_cols: int = 10,
    out_path: Path | None = None,
) -> None:
    """Display a grid of image segments (characters, words, etc.).

    Parameters
    ----------
    segments:
        List of grayscale or BGR images to display.
    title:
        Figure title.
    max_cols:
        Maximum number of columns in the grid.
    out_path:
        If given, save the figure to this path instead of showing it.
    """
    n = len(segments)
    if n == 0:
        return
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, seg in enumerate(segments):
        display = seg if seg.ndim == 2 else cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        axes_flat[i].imshow(display, cmap="gray")
        axes_flat[i].axis("off")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str = "Confusion Matrix",
    out_path: Path | None = None,
) -> None:
    """Plot a normalised confusion matrix using matplotlib.

    Parameters
    ----------
    cm:
        Integer confusion matrix (shape ``[n, n]``).
    labels:
        Ordered list of class names.
    title:
        Figure title.
    out_path:
        If given, save the figure to this path.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
