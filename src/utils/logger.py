"""Standardised application logging.

Usage::

    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Hello from %s", __name__)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    Calling this function multiple times with the same *name* is safe —
    handlers are only attached once.

    Parameters
    ----------
    name:
        Logger name (use ``__name__`` at the call site).
    level:
        Logging level string (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    log_file:
        Optional path to a file where log records are also written.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
