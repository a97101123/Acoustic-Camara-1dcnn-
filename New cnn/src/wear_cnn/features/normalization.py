from __future__ import annotations

import numpy as np


class MinMax01:
    """Train-fitted 0..1 scaler with constant-column protection."""

    def fit(self, values: np.ndarray) -> "MinMax01":
        x = np.asarray(values, dtype=np.float64)
        self.minimum_ = np.nanmin(x, axis=0)
        self.maximum_ = np.nanmax(x, axis=0)
        self.scale_ = self.maximum_ - self.minimum_
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "minimum_"):
            raise RuntimeError("Scaler must be fitted on training data first")
        return (np.asarray(values, dtype=np.float64) - self.minimum_) / self.scale_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)
