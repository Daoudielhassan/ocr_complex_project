"""End-to-end OCR inference pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.io import save_json
from src.utils.logger import get_logger
from src.utils.paths import PATHS

log = get_logger(__name__)


def _load_configs(config_dir: Path) -> dict[str, Any]:
    """Merge all YAML files in *config_dir* into a single flat dict."""
    merged: dict[str, Any] = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged.update(data)
    return merged


class OCRPipeline:
    """End-to-end OCR inference pipeline.

    Usage::

        pipeline = OCRPipeline()
        result = pipeline.run("path/to/document.png")
        print(result["text"])
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_dir: Path | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            cfg_dir = config_dir or PATHS.configs
            self.config = _load_configs(cfg_dir)
        log.debug("OCRPipeline ready (config keys: %s)", list(self.config.keys()))

    def run(
        self,
        document_path: str | Path,
        save_result: bool = True,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run the full OCR pipeline on one document.

        Parameters
        ----------
        document_path:
            Path to the input image or PDF file.
        save_result:
            If *True*, write the result as JSON to *output_dir*.
        output_dir:
            Directory for the output file.
            Defaults to ``artifacts/predictions/``.

        Returns
        -------
        dict
            Result dict with at minimum ``"text"`` and ``"source"`` keys.
        """
        document_path = Path(document_path)
        log.info("OCRPipeline.run: %s", document_path)

        from src.classification.infer_document import infer_document

        result = infer_document(document_path, self.config)

        if save_result:
            out_dir = output_dir or PATHS.predictions
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{document_path.stem}_ocr.json"
            save_json(result, out_file)
            log.info("OCR result saved to %s", out_file)

        return result
