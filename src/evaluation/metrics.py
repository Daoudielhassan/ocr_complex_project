"""OCR evaluation metrics: accuracy, F1, CER, WER."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    average: str = "weighted",
) -> dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted labels (strings or integers).
    average:
        Averaging strategy for multi-class precision / recall / F1.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    log.info(
        "Metrics  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
        acc, prec, rec, f1,
    )
    return metrics


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate:  ``edit_distance(ref, hyp) / len(ref)``."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate:  ``edit_distance(ref.split(), hyp.split()) / len(ref.split())``."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _edit_distance(a: list, b: list) -> int:
    """Dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
