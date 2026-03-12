"""File I/O helpers used across the project."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image file and return a NumPy array.

    Parameters
    ----------
    path:
        File system path to the image.
    grayscale:
        If *True*, return a 2-D grayscale array; otherwise keep original channels.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or OpenCV cannot read it.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    """Write a NumPy array image to disk (parent dirs created automatically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def load_json(path: str | Path) -> Any:
    """Parse a JSON file and return the Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise *data* to a JSON file (parent dirs created automatically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a :class:`pandas.DataFrame`."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Save a DataFrame as CSV (parent dirs created automatically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
