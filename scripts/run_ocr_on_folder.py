#!/usr/bin/env python3
"""Batch OCR processing on a folder of document images.

For every image in *folder*, runs the full OCR pipeline and saves a JSON
result to ``artifacts/exports/``.

Usage::

    python scripts/run_ocr_on_folder.py --folder data/raw/
    python scripts/run_ocr_on_folder.py --folder data/raw/ --fmt markdown
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.paths import PATHS

log = get_logger("run_ocr_on_folder")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch OCR on a folder of documents.")
    p.add_argument("--folder", type=Path, required=True,
                   help="Directory containing document images.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: artifacts/exports/).")
    p.add_argument("--fmt", default="json",
                   choices=["json", "csv", "markdown", "xlsx"],
                   help="Export format for each result (default: json).")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="Configuration directory (default: configs/).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    folder = args.folder
    if not folder.is_dir():
        print(f"ERROR: Folder not found: {folder}", file=sys.stderr)
        return 1

    from src.pipeline.ocr_pipeline import OCRPipeline
    from src.post_processing.exporter import export

    output_dir = args.output_dir or PATHS.exports
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = OCRPipeline(config_dir=args.config_dir)

    image_files = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    if not image_files:
        log.warning("No supported images found in %s", folder)
        return 0

    log.info("Found %d document(s) to process.", len(image_files))
    success, failed = 0, 0

    for img_path in image_files:
        t0 = time.perf_counter()
        try:
            result = pipeline.run(img_path, save_result=False)
            result["processing_time_s"] = round(time.perf_counter() - t0, 3)
            out_path = output_dir / img_path.stem
            export(result, args.fmt, out_path)
            log.info("  ✓ %s  (%.2fs)", img_path.name, result["processing_time_s"])
            success += 1
        except Exception as exc:
            log.error("  ✗ %s  →  %s", img_path.name, exc)
            failed += 1

    log.info("Batch complete: %d succeeded, %d failed.", success, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
