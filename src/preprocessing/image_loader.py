"""Document image loading with optional PDF support."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

_SUPPORTED_RASTER = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_document(path: str | Path, force_grayscale: bool = True) -> np.ndarray:
    """Load a document image or the first page of a PDF.

    Parameters
    ----------
    path:
        Path to an image file ({png, jpg, bmp, tiff}) or a PDF.
    force_grayscale:
        If *True* (default), return a 2-D grayscale ``uint8`` array.

    Returns
    -------
    np.ndarray
        Loaded image as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If OpenCV cannot decode the file.
    ImportError
        If PDF input is requested but ``pdf2image`` is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        img = _load_pdf_first_page(path, force_grayscale)
    else:
        flag = cv2.IMREAD_GRAYSCALE if force_grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise ValueError(f"OpenCV could not read image: {path}")

    log.debug("load_document: %s  shape=%s  dtype=%s", path.name, img.shape, img.dtype)
    return img


def _load_pdf_first_page(path: Path, grayscale: bool) -> np.ndarray:
    """Convert the first PDF page to a NumPy array via ``pdf2image``."""
    try:
        from pdf2image import convert_from_path  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for PDF support. "
            "Install it with:  pip install pdf2image"
        ) from exc

    pages = convert_from_path(str(path), dpi=300, first_page=1, last_page=1)
    pil_img = pages[0]
    img = np.array(pil_img)
    # PIL returns RGB; convert to BGR (OpenCV convention)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
