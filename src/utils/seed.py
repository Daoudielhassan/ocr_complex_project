"""Random-seed initialisation for reproducible experiments."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for :mod:`random` and :mod:`numpy`.

    scikit-learn estimators consume numpy's global random state when
    ``random_state`` is not explicitly set.  For full reproducibility
    pass ``random_state=seed`` to every sklearn estimator as well.
    """
    random.seed(seed)
    np.random.seed(seed)
