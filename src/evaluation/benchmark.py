"""Pipeline variant comparison benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import save_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_benchmark(
    results: list[dict[str, Any]],
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Compare multiple pipeline variant results and optionally save a report.

    Parameters
    ----------
    results:
        List of result dicts.  Each dict must contain at minimum
        ``"name"``, ``"accuracy"``, and ``"f1"`` keys.
    out_path:
        If given, save the comparison table as CSV  and as JSON
        (extension is replaced automatically).

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by ``accuracy`` descending.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        save_json(results, out_path.with_suffix(".json"))
        log.info("Benchmark report saved to %s", out_path.with_suffix(".*"))

    log.info("Benchmark results:\n%s", df.to_string())
    return df
