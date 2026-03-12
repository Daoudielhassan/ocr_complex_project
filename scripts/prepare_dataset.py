#!/usr/bin/env python3
"""Convert the Chars74K-Digital-English-Font dataset into the ImageFolder layout
expected by TrainingPipeline.

Chars74K-Digital source layout
-------------------------------
    <chars74k_root>/
        English/
            Fnt/
                Sample001/   ← class "0"
                    img001-001.png
                    …
                Sample002/   ← class "1"
                …
                Sample062/   ← class "z"

Class mapping (62 classes)
--------------------------
    Sample001 – Sample010  →  '0' – '9'
    Sample011 – Sample036  →  'A' – 'Z'
    Sample037 – Sample062  →  'a' – 'z'

Output ImageFolder layout
--------------------------
    <out_dir>/
        train/<label>/<img>.png
        val/<label>/<img>.png
        test/<label>/<img>.png

Usage
-----
    python scripts/prepare_dataset.py \\
        --chars74k-dir path/to/English/Fnt \\
        --out-dir data/dataset \\
        [--val-ratio 0.15] [--test-ratio 0.15] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.paths import PATHS

log = get_logger("prepare_dataset")

# ── Chars74K class mapping ────────────────────────────────────────────────────
# 62 classes: 0-9 (Sample001-010), A-Z (Sample011-036), a-z (Sample037-062)
#
# Windows filesystems are case-insensitive: folder 'a' == folder 'A'.
# To avoid collisions, lowercase letters are stored in folders named with a
# trailing underscore: 'a' → 'a_', 'b' → 'b_', …, 'z' → 'z_'.
# The trailing underscore is stripped back to the real character at decode time
# (see src/classification/predict.py → predict_chars).
_DIGITS    = [str(d) for d in range(10)]                    # '0'..'9'
_UPPERCASE = [chr(c) for c in range(ord('A'), ord('Z') + 1)]  # 'A'..'Z'
_LOWERCASE = [chr(c) for c in range(ord('a'), ord('z') + 1)]  # 'a'..'z'

CLASS_LABELS: dict[str, str] = {}  # "Sample001" → folder name
for _i, _char in enumerate(_DIGITS + _UPPERCASE + _LOWERCASE, start=1):
    # Lowercase letters get a trailing underscore to be Windows-safe
    folder = (_char + "_") if _char.islower() else _char
    CLASS_LABELS[f"Sample{_i:03d}"] = folder

# Reverse mapping: folder name → actual character (used for reference / tests)
FOLDER_TO_CHAR: dict[str, str] = {
    folder: folder.rstrip("_") for folder in CLASS_LABELS.values()
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Chars74K-Digital-English-Font to ImageFolder layout."
    )
    p.add_argument(
        "--chars74k-dir", type=Path, default=None,
        help="Path to the English/Fnt/ directory of the Chars74K dataset "
             "(default: data/raw/English/Fnt).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output root for the ImageFolder dataset (default: data/dataset).",
    )
    p.add_argument("--val-ratio",  type=float, default=0.15, help="Fraction for validation split.")
    p.add_argument("--test-ratio", type=float, default=0.15, help="Fraction for test split.")
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed for shuffling.")
    return p.parse_args()


def _split_files(
    files: list[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle and split a file list into (train, val, test)."""
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    n = len(files)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    test  = files[:n_test]
    val   = files[n_test : n_test + n_val]
    train = files[n_test + n_val :]
    return train, val, test


def main() -> int:
    args = _parse_args()

    fnt_dir = args.chars74k_dir or (PATHS.raw / "English" / "Fnt")
    out_root = args.out_dir or PATHS.dataset

    if not fnt_dir.exists():
        log.error(
            "Chars74K Fnt directory not found: %s\n"
            "Download the dataset from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ "
            "and extract it so that English/Fnt/ exists, then re-run this script.",
            fnt_dir,
        )
        return 1

    sample_dirs = sorted(fnt_dir.glob("Sample*"))
    if not sample_dirs:
        log.error("No Sample* directories found inside %s", fnt_dir)
        return 1

    log.info("Found %d class directories in %s", len(sample_dirs), fnt_dir)

    total_copied = 0
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for sample_dir in sample_dirs:
        folder_name = sample_dir.name  # e.g. "Sample011"
        label = CLASS_LABELS.get(folder_name)
        if label is None:
            log.warning("Unknown folder %s — skipping.", folder_name)
            continue

        images = sorted(
            p for p in sample_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".bmp", ".tif", ".tiff"}
        )
        if not images:
            log.warning("%s contains no images — skipping.", folder_name)
            continue

        train_f, val_f, test_f = _split_files(
            images, args.val_ratio, args.test_ratio, args.seed
        )

        for split_name, file_list in (("train", train_f), ("val", val_f), ("test", test_f)):
            dest_dir = out_root / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in file_list:
                shutil.copy2(src, dest_dir / src.name)
                split_counts[split_name] += 1
                total_copied += 1

        log.debug(
            "%-12s label=%-3s  train=%d  val=%d  test=%d",
            folder_name, label, len(train_f), len(val_f), len(test_f),
        )

    log.info(
        "Done. Copied %d images total  |  train=%d  val=%d  test=%d",
        total_copied, split_counts["train"], split_counts["val"], split_counts["test"],
    )
    log.info("Dataset written to %s", out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
