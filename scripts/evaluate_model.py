#!/usr/bin/env python3
"""Evaluate the trained model on the test split and save a report.

Usage::

    python scripts/evaluate_model.py [--data-dir PATH] [--config-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.paths import PATHS

log = get_logger("evaluate_model")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the SVM OCR model on the test set.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Test split directory (default: data/dataset/test/).")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="Configuration directory (default: configs/).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from src.pipeline.training_pipeline import TrainingPipeline, _load_configs
    from src.classification.model import SVMModel
    from src.features.scaling import load_scaler, transform
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.confusion import plot_confusion
    from src.evaluation.error_analysis import analyze_errors
    from src.utils.io import save_json

    cfg_dir = args.config_dir or PATHS.configs
    config = _load_configs(cfg_dir)
    feat_cfg = config.get("features", {})

    pipeline = TrainingPipeline(config=config)
    test_dir = args.data_dir or PATHS.test
    X_test, y_test, label_mgr = pipeline._load_split(test_dir, feat_cfg)

    if len(X_test) == 0:
        log.warning("Test set is empty — nothing to evaluate.")
        return 0

    model = SVMModel.load(PATHS.svm_model)
    scaler = load_scaler(PATHS.scaler)
    X_test_s = transform(X_test, scaler)
    y_pred = model.predict(X_test_s)

    y_true_str = label_mgr.decode(y_test)
    y_pred_str = label_mgr.decode(y_pred)

    metrics = compute_metrics(y_true_str, y_pred_str)
    log.info("Test metrics: %s", metrics)

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    report_path = PATHS.reports / "evaluation_report.json"
    save_json({"metrics": metrics, "n_samples": len(y_test)}, report_path)
    log.info("Report saved to %s", report_path)

    PATHS.figures.mkdir(parents=True, exist_ok=True)
    cm_path = PATHS.figures / "confusion_matrix_eval.png"
    plot_confusion(y_true_str, y_pred_str, out_path=cm_path)

    errors_df = analyze_errors(y_true_str, y_pred_str)
    if len(errors_df) > 0:
        errors_path = PATHS.reports / "error_analysis.csv"
        errors_df.to_csv(errors_path, index=False)
        log.info("Error analysis saved to %s", errors_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
