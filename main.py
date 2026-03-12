#!/usr/bin/env python3
"""Main entry point for the OCR Complex Project.

Usage examples
--------------
Train a new SVM model::

    python main.py --mode train

Run OCR inference on a single document::

    python main.py --mode infer --input data/raw/document.png

Evaluate the model on the test set::

    python main.py --mode evaluate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ocr_pipeline",
        description="HOG + SVM document OCR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "evaluate"],
        required=True,
        help="Execution mode.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Input document path (required for --mode infer).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Path to the configs/ directory (default: configs/ beside this file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Dataset root directory (used in --mode train/evaluate).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from src.utils.logger import get_logger
    log = get_logger("main")

    if args.mode == "train":
        log.info("=== Mode: train ===")
        from src.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline(config_dir=args.config_dir)
        metrics = pipeline.run(data_dir=args.data_dir)
        log.info("Training finished. Metrics: %s", metrics)

    elif args.mode == "infer":
        if args.input is None:
            print("ERROR: --input is required for --mode infer", file=sys.stderr)
            return 1
        if not args.input.exists():
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            return 1
        log.info("=== Mode: infer → %s ===", args.input)
        from src.pipeline.ocr_pipeline import OCRPipeline
        pipeline = OCRPipeline(config_dir=args.config_dir)
        result = pipeline.run(args.input, output_dir=args.output_dir)
        print(result.get("text", ""))

    elif args.mode == "evaluate":
        log.info("=== Mode: evaluate ===")
        from src.pipeline.training_pipeline import TrainingPipeline, _load_configs
        from src.utils.paths import PATHS
        from src.classification.model import SVMModel
        from src.features.scaling import load_scaler, transform
        from src.evaluation.metrics import compute_metrics
        from src.evaluation.confusion import plot_confusion

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
        metrics = compute_metrics(y_test, y_pred)
        log.info("Test metrics: %s", metrics)

        PATHS.figures.mkdir(parents=True, exist_ok=True)
        plot_confusion(y_test, y_pred, out_path=PATHS.figures / "confusion_matrix_eval.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
