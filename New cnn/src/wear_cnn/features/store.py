from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureTable:
    values: np.ndarray
    wear: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    tool_ids: list[str]
    record_ids: list[str]
