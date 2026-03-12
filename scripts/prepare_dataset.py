#!/usr/bin/env python3
"""Prepare the labelled character dataset from raw document images.

For each image in ``data/raw/`` this script runs the full preprocessing and
segmentation stack, saves isolated character crops to ``data/dataset/<split>/``
and appends entries to ``data/annotations/labels.csv``.

Usage::

    python scripts/prepare_dataset.py [--raw-dir PATH] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.paths import PATHS
from src.utils.io import save_image

log = get_logger("prepare_dataset")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare character dataset from raw documents.")
    p.add_argument("--raw-dir", type=Path, default=None, help="Raw document directory.")
    p.add_argument("--out-dir", type=Path, default=None, help="Dataset output root.")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Which split to write to (default: train).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    raw_dir = args.raw_dir or PATHS.raw
    out_root = args.out_dir or PATHS.dataset

    from src.preprocessing.image_loader import load_document
    from src.preprocessing.binarization import binarize
    from src.preprocessing.denoising import denoise
    from src.preprocessing.deskewing import deskew
    from src.preprocessing.normalization import normalize
    from src.segmentation.layout_analyzer import analyze_layout
    from src.segmentation.line_segmenter import segment_lines
    from src.segmentation.word_segmenter import segment_words
    from src.segmentation.char_segmenter import segment_chars

    image_files = list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))
    if not image_files:
        log.warning("No images found in %s", raw_dir)
        return 0

    log.info("Processing %d images from %s", len(image_files), raw_dir)
    char_count = 0

    for img_path in sorted(image_files):
        log.info("Processing %s", img_path.name)
        try:
            img = load_document(img_path)
            img = denoise(img, method="median")
            img = deskew(img)
            img = binarize(img, method="otsu")
            img = normalize(img, target_height=1024)
        except Exception as exc:
            log.error("Failed to preprocess %s: %s", img_path, exc)
            continue

        blocks = analyze_layout(img)
        if not blocks:
            h, w = img.shape
            blocks = [(0, 0, w, h)]

        for bx, by, bw, bh in blocks:
            block_img = img[by : by + bh, bx : bx + bw]
            line_imgs, _ = segment_lines(block_img)
            for line_img in line_imgs:
                word_imgs, _ = segment_words(line_img)
                for word_img in word_imgs:
                    char_imgs, _ = segment_chars(word_img)
                    for ci, char_img in enumerate(char_imgs):
                        # Without a ground-truth label we use the document stem
                        # as a placeholder label directory.  Replace with actual
                        # labels sourced from data/annotations/labels.csv.
                        label = img_path.stem
                        out_dir = out_root / args.split / label
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{img_path.stem}_{char_count:06d}.png"
                        save_image(char_img, out_path)
                        char_count += 1

    log.info("Dataset preparation complete. %d character crops saved.", char_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
