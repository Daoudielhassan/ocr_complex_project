"""Centralised project path management.

All modules should import ``PATHS`` from this module rather than
constructing ``Path`` objects inline so that paths remain easily
configurable via the ``OCR_PROJECT_ROOT`` environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root directory.

    Resolution order:
    1. ``OCR_PROJECT_ROOT`` environment variable.
    2. Three levels up from this file (src/utils/paths.py → src/ → root).
    """
    env_root = os.environ.get("OCR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class ProjectPaths:
    """Accessor for every well-known directory in the project."""

    root: Path = field(default_factory=_project_root)

    # ── data ────────────────────────────────────────────────────
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.interim / "processed"

    @property
    def lines(self) -> Path:
        return self.interim / "lines"

    @property
    def words(self) -> Path:
        return self.interim / "words"

    @property
    def chars(self) -> Path:
        return self.interim / "chars"

    @property
    def dataset(self) -> Path:
        return self.data / "dataset"

    @property
    def train(self) -> Path:
        return self.dataset / "train"

    @property
    def val(self) -> Path:
        return self.dataset / "val"

    @property
    def test(self) -> Path:
        return self.dataset / "test"

    @property
    def annotations(self) -> Path:
        return self.data / "annotations"

    # ── models ──────────────────────────────────────────────────
    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def svm_model(self) -> Path:
        return self.models / "svm_ocr_model.pkl"

    @property
    def scaler(self) -> Path:
        return self.models / "scaler.pkl"

    @property
    def label_encoder(self) -> Path:
        return self.models / "label_encoder.pkl"

    @property
    def model_metadata(self) -> Path:
        return self.models / "metadata.json"

    # ── artifacts ───────────────────────────────────────────────
    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.artifacts / "figures"

    @property
    def predictions(self) -> Path:
        return self.artifacts / "predictions"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    @property
    def exports(self) -> Path:
        return self.artifacts / "exports"

    # ── configs / runs ──────────────────────────────────────────
    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    # ── helpers ─────────────────────────────────────────────────
    def makedirs(self) -> None:
        """Create all project directories (idempotent)."""
        dirs = [
            self.raw, self.processed, self.lines, self.words, self.chars,
            self.train, self.val, self.test, self.annotations,
            self.models, self.figures, self.predictions, self.reports,
            self.exports, self.configs, self.runs,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


#: Module-level singleton – import and use directly.
PATHS = ProjectPaths()
