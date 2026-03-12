"""OCR result exporter: JSON, CSV, Markdown, XLSX."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

_FORMATS = {"json", "csv", "markdown", "xlsx"}


def export(results: dict[str, Any], fmt: str, out_path: str | Path) -> Path:
    """Export OCR results to the specified format.

    Parameters
    ----------
    results:
        OCR result dict (must contain at minimum a ``"text"`` key).
    fmt:
        Output format: ``"json"``, ``"csv"``, ``"markdown"``, or ``"xlsx"``.
    out_path:
        Destination file path.  The extension is overridden to match *fmt*.

    Returns
    -------
    Path
        Actual path of the written file.

    Raises
    ------
    ValueError
        If *fmt* is not one of the supported formats.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()

    if fmt not in _FORMATS:
        raise ValueError(
            f"Unknown export format: {fmt!r}. "
            f"Choose from: {', '.join(sorted(_FORMATS))}."
        )

    if fmt == "json":
        out_path = out_path.with_suffix(".json")
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    elif fmt == "csv":
        out_path = out_path.with_suffix(".csv")
        _export_csv(results, out_path)
    elif fmt == "markdown":
        out_path = out_path.with_suffix(".md")
        _export_markdown(results, out_path)
    elif fmt == "xlsx":
        out_path = out_path.with_suffix(".xlsx")
        _export_xlsx(results, out_path)

    log.info("Exported OCR results to %s", out_path)
    return out_path


def _export_csv(results: dict, path: Path) -> None:
    text = results.get("text", "")
    lines = text.split("\n")
    df = pd.DataFrame({"line_number": range(1, len(lines) + 1), "text": lines})
    df.to_csv(path, index=False, encoding="utf-8")


def _export_markdown(results: dict, path: Path) -> None:
    text = results.get("text", "")
    source = results.get("source", "unknown")
    lines = [
        "# OCR Result",
        "",
        f"**Source**: `{source}`",
        "",
        "## Extracted Text",
        "",
        "```",
        text,
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_xlsx(results: dict, path: Path) -> None:
    text = results.get("text", "")
    lines = text.split("\n")
    df = pd.DataFrame({"line_number": range(1, len(lines) + 1), "text": lines})
    df.to_excel(path, index=False, engine="openpyxl")
