"""Full supervised training pipeline: dataset → features → SVM → evaluation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils.io import save_json
from src.utils.logger import get_logger
from src.utils.paths import PATHS
from src.utils.seed import set_seed

log = get_logger(__name__)


def _load_configs(config_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged.update(data)
    return merged


class TrainingPipeline:
    """Supervised training pipeline.

    Execution order:
    1. Load images from dataset split directories.
    2. Extract HOG features.
    3. Fit a StandardScaler on the training split.
    4. Train an SVM with stratified cross-validation.
    5. Evaluate on validation and test splits.
    6. Persist model, scaler, label encoder, and metadata.

    Usage::

        pipeline = TrainingPipeline()
        metrics = pipeline.run()
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_dir: Path | None = None,
    ) -> None:
        cfg_dir = config_dir or PATHS.configs
        self.config = config if config is not None else _load_configs(cfg_dir)
        set_seed(self.config.get("seed", 42))
        log.info("TrainingPipeline initialised (seed=%d)", self.config.get("seed", 42))

    def run(self, data_dir: str | Path | None = None) -> dict[str, Any]:
        """Execute the full training loop.

        Parameters
        ----------
        data_dir:
            Root dataset directory (must contain ``train/``, ``val/``,
            ``test/`` sub-directories with class-named sub-directories).
            Defaults to ``data/dataset/``.

        Returns
        -------
        dict
            Combined training, validation and test metrics.
        """
        from src.features.scaling import fit_scaler, transform, save_scaler
        from src.classification.train import train
        from src.classification.label_encoder import LabelManager
        from src.evaluation.metrics import compute_metrics
        from src.evaluation.confusion import plot_confusion

        data_dir = Path(data_dir) if data_dir else PATHS.dataset
        feat_cfg = self.config.get("features", {})
        svm_cfg = self.config.get("svm", {})

        log.info("Loading dataset splits from %s", data_dir)
        X_train, y_train, label_mgr = self._load_split(data_dir / "train", feat_cfg)
        X_val, y_val, _ = self._load_split(data_dir / "val", feat_cfg, label_mgr=label_mgr)
        X_test, y_test, _ = self._load_split(data_dir / "test", feat_cfg, label_mgr=label_mgr)

        if len(X_train) == 0:
            log.warning("Training set is empty — aborting.")
            return {"error": "empty_dataset"}

        log.info("Fitting scaler on %d training samples", len(X_train))
        scaler = fit_scaler(X_train)
        X_train_s = transform(X_train, scaler)
        X_val_s = transform(X_val, scaler) if len(X_val) > 0 else X_val
        X_test_s = transform(X_test, scaler) if len(X_test) > 0 else X_test

        model, cv_metrics = train(
            X_train_s,
            y_train,
            kernel=svm_cfg.get("kernel", "rbf"),
            C=float(svm_cfg.get("C", 1.0)),
            gamma=svm_cfg.get("gamma", "scale"),
            cv_folds=int(svm_cfg.get("cv_folds", 5)),
            grid_search=bool(svm_cfg.get("grid_search", False)),
        )

        val_metrics: dict[str, Any] = {}
        test_metrics: dict[str, Any] = {}

        if len(X_val_s) > 0:
            y_val_pred = model.predict(X_val_s)
            val_metrics = {"val_" + k: v for k, v in compute_metrics(y_val, y_val_pred).items()}

        if len(X_test_s) > 0:
            y_test_pred = model.predict(X_test_s)
            test_metrics = {
                "test_" + k: v for k, v in compute_metrics(y_test, y_test_pred).items()
            }
            PATHS.figures.mkdir(parents=True, exist_ok=True)
            plot_confusion(y_test, y_test_pred, out_path=PATHS.figures / "confusion_matrix_test.png")

        # Persist artefacts
        PATHS.models.mkdir(parents=True, exist_ok=True)
        model.save(PATHS.svm_model)
        save_scaler(scaler, PATHS.scaler)
        label_mgr.save(PATHS.label_encoder)

        all_metrics: dict[str, Any] = {**cv_metrics, **val_metrics, **test_metrics}
        metadata = {
            "version": "1.0.0",
            "trained_at": datetime.now().isoformat(),
            "hog_params": feat_cfg,
            "svm_params": svm_cfg,
            "scores": all_metrics,
            "n_classes": label_mgr.n_classes,
            "classes": label_mgr.classes[:50],
        }
        save_json(metadata, PATHS.model_metadata)
        log.info("Training complete.  Metrics: %s", all_metrics)
        return all_metrics

    def _load_split(
        self,
        split_dir: Path,
        feat_cfg: dict,
        label_mgr: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Load all images from *split_dir* and extract HOG features.

        *split_dir* is expected to contain one sub-directory per class,
        following the standard ImageFolder layout::

            split_dir/
                A/  img1.png  img2.png …
                B/  img1.png …

        Returns
        -------
        (X, y, label_manager)
            ``X`` — float32 feature matrix.
            ``y`` — integer label array.
            ``label_manager`` — fitted :class:`LabelManager`.
        """
        from src.features.feature_pipeline import build_features
        from src.classification.label_encoder import LabelManager
        from src.utils.io import load_image

        resize_to = tuple(feat_cfg.get("resize_to", [32, 32]))
        orientations = int(feat_cfg.get("orientations", 9))
        pixels_per_cell = tuple(feat_cfg.get("pixels_per_cell", [8, 8]))
        cells_per_block = tuple(feat_cfg.get("cells_per_block", [2, 2]))

        images: list[np.ndarray] = []
        labels_str: list[str] = []

        if not split_dir.is_dir():
            log.debug("Split dir not found: %s (skipping)", split_dir)
        else:
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                for img_file in sorted(class_dir.iterdir()):
                    if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                        continue
                    try:
                        img = load_image(img_file, grayscale=True)
                        images.append(img)
                        labels_str.append(class_name)
                    except Exception as exc:
                        log.warning("Skipping %s: %s", img_file, exc)

        if not images:
            empty_lm = label_mgr if label_mgr is not None else LabelManager()
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=object),
                empty_lm,
            )

        X = build_features(
            images,
            resize_to=resize_to,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )

        if label_mgr is None:
            label_mgr = LabelManager().fit(labels_str)

        y = label_mgr.encode(labels_str)
        return X, y, label_mgr
