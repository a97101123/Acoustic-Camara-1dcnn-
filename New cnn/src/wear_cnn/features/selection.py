from __future__ import annotations

import numpy as np

from .normalization import MinMax01


def r2_scores(features: np.ndarray, wear: np.ndarray, backend: str) -> np.ndarray:
    """R² after train-only 0..1 normalization of wear and every feature."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(wear, dtype=np.float64).reshape(-1, 1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be [records, features] and match wear")
    xn = MinMax01().fit_transform(x)
    yn = MinMax01().fit_transform(y)[:, 0]
    if backend not in ("legacy", "matrix"):
        raise ValueError("R2 score_backend must be legacy or matrix")
    if backend == "matrix":
        centered_x = xn - xn.mean(axis=0)
        centered_y = yn - yn.mean()
        numerator = np.einsum("ij,i->j", centered_x, centered_y) ** 2
        denominator = np.einsum("ij,ij->j", centered_x, centered_x) * np.dot(centered_y, centered_y)
        valid = ~np.isclose(xn, xn[0]).all(axis=0)
        valid &= not np.allclose(yn, yn[0])
        scores = np.divide(
            numerator, denominator, out=np.zeros(x.shape[1]), where=valid & (denominator > 0)
        )
        return np.clip(scores, 0.0, 1.0)
    scores = np.zeros(x.shape[1], dtype=np.float64)
    for index in range(x.shape[1]):
        column = xn[:, index]
        if np.allclose(column, column[0]) or np.allclose(yn, yn[0]):
            scores[index] = 0.0
            continue
        fit = np.polyfit(yn, column, 1)
        predicted = np.polyval(fit, yn)
        denominator = np.sum((column - column.mean()) ** 2)
        scores[index] = max(0.0, 1.0 - np.sum((column - predicted) ** 2) / denominator)
    return scores


def fisher_j_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Multiclass Fisher score on unnormalised feature values."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be [records, features] and match labels")
    overall = np.mean(x, axis=0)
    numerator = np.zeros(x.shape[1])
    denominator = np.zeros(x.shape[1])
    for label in np.unique(y):
        group = x[y == label]
        numerator += len(group) * (np.mean(group, axis=0) - overall) ** 2
        denominator += np.sum((group - np.mean(group, axis=0)) ** 2, axis=0)
    # A separated feature with zero within-class variance is informative, not zero.
    # Relative regularization keeps scores finite for selection, JSON and plots.
    denominator = np.maximum(denominator, np.finfo(np.float64).eps * (numerator + denominator))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def select_r2(scores: np.ndarray, threshold: float) -> tuple[np.ndarray, bool]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= threshold <= 1:
        raise ValueError("Invalid R2 scores or threshold")
    selected = np.flatnonzero(scores > threshold)
    if not len(selected):
        raise ValueError("No features pass the R2 threshold")
    return selected, False


def select_fisher_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    return np.argsort(np.asarray(scores))[::-1][:top_k]


def select_fisher_quotas(
    scores: np.ndarray,
    feature_names: list[str] | np.ndarray,
    quotas: dict[str, int],
) -> np.ndarray:
    """Select highest Fisher scores independently for each sensor/domain."""
    values = np.asarray(scores)
    names = [str(name) for name in feature_names]
    if len(values) != len(names):
        raise ValueError("scores and feature_names must have equal length")

    def category(name: str) -> str | None:
        if name.startswith("acc::"):
            return "acc"
        if name.startswith("mic::"):
            return "mic_frequency" if "_freq_" in name else "mic_time"
        return None

    selected: list[int] = []
    for group in ("acc", "mic_frequency", "mic_time"):
        quota = int(quotas[group])
        if quota < 0:
            raise ValueError(f"quota for {group} cannot be negative")
        candidates = [index for index, name in enumerate(names) if category(name) == group]
        candidates.sort(key=lambda index: (-float(values[index]), index))
        selected.extend(candidates[:quota])
    return np.asarray(selected, dtype=int)
