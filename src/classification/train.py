"""SVM training with optional cross-validation and grid search."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.classification.model import SVMModel
from src.utils.logger import get_logger

log = get_logger(__name__)


def train(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    cv_folds: int = 5,
    grid_search: bool = False,
    param_grid: dict[str, list] | None = None,
    random_state: int = 42,
) -> tuple[SVMModel, dict[str, Any]]:
    """Train an SVM classifier and report cross-validation accuracy.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Integer class labels of shape ``(n_samples,)``.
    kernel, C, gamma:
        SVM hyperparameters (ignored when *grid_search* is True).
    cv_folds:
        Number of stratified cross-validation folds.
    grid_search:
        If True, run :class:`~sklearn.model_selection.GridSearchCV` over
        *param_grid* to find the best hyperparameters before final fitting.
    param_grid:
        Parameter grid for GridSearchCV.  Defaults to a small rbf/linear grid.
    random_state:
        Random seed passed to the SVM estimator.

    Returns
    -------
    (model, metrics)
        ``model`` — fitted :class:`SVMModel`.
        ``metrics`` — dict with ``cv_accuracy_mean``, ``cv_accuracy_std``.
    """
    if grid_search:
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear"],
            }
        gs = GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X, y)
        best = gs.best_params_
        log.info("GridSearchCV best params: %s  best_score=%.4f", best, gs.best_score_)
        model = SVMModel(
            kernel=best["kernel"],
            C=best["C"],
            gamma=best["gamma"],
            random_state=random_state,
        )
        model.fit(X, y)
        metrics: dict[str, Any] = {
            "cv_accuracy_mean": float(gs.best_score_),
            "cv_accuracy_std": 0.0,
            "best_params": best,
        }
    else:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=random_state),
            X, y, cv=skf, scoring="accuracy", n_jobs=-1,
        )
        log.info("CV accuracy: %.4f ± %.4f", scores.mean(), scores.std())
        model = SVMModel(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        model.fit(X, y)
        metrics = {
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
        }

    return model, metrics
