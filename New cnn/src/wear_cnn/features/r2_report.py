"""R2 threshold sensitivity for training records and descriptive position means."""

from pathlib import Path

import numpy as np
import pandas as pd

from .plots import analysis_position_means
from .selection import r2_scores


def write_r2_threshold_report(table, output, *, thresholds, groups=None):
    """Count every score strictly above each threshold without changing selection."""
    output = Path(output)
    thresholds = np.asarray(thresholds, dtype=float)
    if (
        thresholds.ndim != 1
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError("R2 thresholds must be finite values in [0, 1]")
    wear, values, _, _ = analysis_position_means(
        table.wear, table.values, table.record_ids, table.tool_ids
    )
    scores = {
        "cnn_training_records": r2_scores(table.values, table.wear, backend="matrix"),
        "plot_position_means": r2_scores(values, wear, backend="matrix"),
    }
    groups = groups if groups is not None else {"all": list(range(len(table.feature_names)))}
    rows = []
    for scope, values in scores.items():
        for group, indices in groups.items():
            candidates = values[indices]
            for threshold in thresholds:
                rows.append(
                    {
                        "scope": scope,
                        "group": group,
                        "threshold_exclusive": threshold,
                        "selected_count": int(np.count_nonzero(candidates > threshold)),
                        "candidate_count": len(candidates),
                        "maximum_r2": float(candidates.max()) if len(candidates) else np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": table.feature_names, **scores}).to_csv(
        output.with_suffix(".features.csv"), index=False, encoding="utf-8-sig"
    )
    return frame
