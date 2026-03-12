"""Evaluation sub-package public API."""
from src.evaluation.metrics import compute_metrics, compute_cer, compute_wer
from src.evaluation.confusion import plot_confusion
from src.evaluation.error_analysis import analyze_errors
from src.evaluation.benchmark import run_benchmark

__all__ = [
    "compute_metrics",
    "compute_cer",
    "compute_wer",
    "plot_confusion",
    "analyze_errors",
    "run_benchmark",
]
