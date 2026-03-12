#!/usr/bin/env python3
"""Train the SVM model on the prepared dataset.

Usage::

    python scripts/train_svm.py [--data-dir PATH] [--config-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

log = get_logger("train_svm")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the OCR SVM model.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Dataset root (default: data/dataset/).")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="Configuration directory (default: configs/).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from src.pipeline.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config_dir=args.config_dir)
    metrics = pipeline.run(data_dir=args.data_dir)

    if "error" in metrics:
        log.error("Training failed: %s", metrics["error"])
        return 1

    log.info("Training finished successfully.")
    log.info("Final metrics:")
    for k, v in metrics.items():
        log.info("  %-30s %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
