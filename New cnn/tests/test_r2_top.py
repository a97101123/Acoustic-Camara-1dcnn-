from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wear_cnn.configuration import validate_training_config
from wear_cnn.domain_selection import select_domains
from wear_cnn.feature_builder import feature_catalog
from wear_cnn.features.store import FeatureTable
from wear_cnn.run_io import load_config
from wear_cnn.trainer_batch import select_job

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("condition", ["noac", "withac"])
def test_active_excel_training_profiles_select_top_22(condition):
    _, cfg = load_config(ROOT / f"configs/turning_{condition}_excel_training.yaml")
    validate_training_config(cfg)
    policy = cfg["selection"]["r2"]
    assert policy["mode"] == "top"
    assert "threshold" not in policy
    assert policy["acc_axis_quota"] == {"x": 4, "y": 4, "z": 3}
    assert policy["mic_domain_quota"] == {"mic_frequency": 7, "mic_time": 4}


def test_r2_top_ranks_within_quotas_even_when_all_scores_are_below_half():
    names = [
        "acc::x_rms",
        "acc::x_std",
        "acc::y_rms",
        "acc::z_rms",
        "mic::mic_freq_1Hz",
        "mic::mic_freq_2Hz",
        "mic::mic_rms",
        "mic::mic_peak",
    ]
    policy = dict(
        mode="top",
        acc_axis_quota={"x": 1, "y": 1, "z": 1},
        mic_domain_quota={"mic_frequency": 1, "mic_time": 2},
        require_equal_sensor_totals=True,
    )
    chosen, receipts = select_domains(
        np.array([0.1, 0.4, 0.2, 0.3, 0.1, 0.4, 0.15, 0.15]), feature_catalog(names), "r2", policy
    )
    assert chosen.tolist() == [1, 2, 3, 5, 6, 7]
    assert sum(receipt["selected"] for receipt in receipts) == 6
    policy["acc_axis_quota"]["x"] = 3
    with pytest.raises(ValueError, match="requested 3, available 2"):
        select_domains(np.zeros(len(names)), feature_catalog(names), "r2", policy)


@pytest.mark.parametrize(
    "change", ["threshold", "negative", "missing", "unknown_mode", "zero", "not_mapping"]
)
def test_invalid_top_policy_fails_before_training(change):
    _, cfg = load_config(ROOT / "configs/turning_noac_excel_training.yaml")
    policy = cfg["selection"]["r2"]
    if change == "threshold":
        policy["threshold"] = 0.5
    elif change == "negative":
        policy["acc_axis_quota"]["x"] = -1
    elif change == "missing":
        del policy["mic_domain_quota"]["mic_time"]
    elif change == "unknown_mode":
        policy["mode"] = "fallback"
    elif change == "zero":
        policy["acc_axis_quota"] = dict.fromkeys("xyz", 0)
        policy["mic_domain_quota"] = {"mic_frequency": 0, "mic_time": 0}
    else:
        cfg["selection"]["r2"] = None
    with pytest.raises(ValueError):
        validate_training_config(cfg)


def test_top_exports_policy_and_keeps_training_threshold_diagnostics(tmp_path):
    names = ["acc::x_rms", "acc::y_rms", "acc::z_rms", "mic::mic_freq_1Hz", "mic::mic_rms"]
    table = FeatureTable(
        np.ones((6, len(names))),
        np.arange(6, dtype=float),
        np.array([0, 0, 1, 1, 2, 2]),
        names,
        ["T01"] * 6,
        [f"M{i:03d}" for i in range(6)],
    )
    _, cfg = load_config(ROOT / "configs/turning_noac_excel_training.yaml")
    policy = deepcopy(cfg["selection"]["r2"])
    policy.update(
        acc_axis_quota=dict.fromkeys("xyz", 1),
        mic_domain_quota={"mic_frequency": 1, "mic_time": 1},
        require_equal_sensor_totals=False,
    )
    selected, scores = select_job(
        "unused",
        feature_catalog(names),
        "r2",
        policy,
        tmp_path,
        SimpleNamespace(record_summary=lambda _: table),
    )
    assert len(selected) == 5 and np.all(scores == 0)
    exported = pd.read_csv(tmp_path / "feature_scores.csv")
    assert exported.selection_mode.eq("top").all()
    assert exported.selection_reason.eq("axis_or_domain_policy").all()
    assert (tmp_path / "r2_threshold_counts.csv").exists()
