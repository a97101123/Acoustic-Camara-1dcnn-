import numpy as np
import pytest

from wear_cnn.features.normalization import MinMax01
from wear_cnn.features.selection import fisher_j_scores, r2_scores, select_fisher_quotas, select_r2


def test_minmax_constant_column_is_finite():
    output = MinMax01().fit_transform(np.array([[2.0, 1.0], [2.0, 3.0]]))
    assert np.isfinite(output).all()
    assert np.all(output[:, 0] == 0)


def test_r2_finds_linear_feature():
    wear = np.arange(10, dtype=float)
    features = np.column_stack([wear * 3 + 2, np.ones(10)])
    scores = r2_scores(features, wear, "legacy")
    assert scores[0] > 0.999
    assert scores[1] == 0


def test_r2_never_fills_below_threshold():
    with pytest.raises(ValueError, match="No features pass"):
        select_r2(np.array([0.1, 0.4, 0.5]), threshold=0.5)


def test_r2_selects_every_feature_strictly_above_threshold():
    scores = np.array([0.5, 0.500001, 0.2] + [0.9] * 30)
    selected, fallback = select_r2(scores, threshold=0.5)
    assert selected.tolist() == [1] + list(range(3, 33))
    assert fallback is False


def test_fisher_j_prefers_separated_feature():
    labels = np.array([0, 0, 1, 1, 2, 2])
    features = np.column_stack([labels * 10 + np.array([0, 1, 0, 1, 0, 1]), np.ones(6)])
    scores = fisher_j_scores(features, labels)
    assert scores[0] > scores[1]


def test_fisher_quotas_select_each_sensor_domain_independently():
    names = (
        [f"acc::x_{i}" for i in range(9)]
        + [f"mic::mic_freq_{i}" for i in range(9)]
        + [f"mic::time_{i}" for i in range(5)]
    )
    selected = select_fisher_quotas(
        np.arange(len(names), dtype=float), names, {"acc": 7, "mic_frequency": 7, "mic_time": 3}
    )
    chosen = [names[index] for index in selected]
    assert len(chosen) == 17
    assert sum(name.startswith("acc::") for name in chosen) == 7
    assert sum(name.startswith("mic::") and "_freq_" in name for name in chosen) == 7
    assert sum(name.startswith("mic::") and "_freq_" not in name for name in chosen) == 3
