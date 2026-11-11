from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import settings as explicit

from wear_cnn.feature_builder import feature_catalog
from wear_cnn.features.r2_report import write_r2_threshold_report
from wear_cnn.features.store import FeatureTable
from wear_cnn.trainer_batch import select_job


def test_report_counts_both_scopes_without_changing_selection(tmp_path):
    table = FeatureTable(
        np.array([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=float),
        np.array([1, 2, 3, 4]),
        np.zeros(4),
        ["linear", "constant"],
        ["a", "a", "b", "b"],
        ["M001", "M002", "M001", "M002"],
    )
    original = table.values.copy()
    output = tmp_path / "counts.csv"
    frame = write_r2_threshold_report(table, output, thresholds=[0, 0.5, 1])
    assert set(frame.scope) == {"cnn_training_records", "plot_position_means"}
    for _, scope in frame.groupby("scope"):
        assert scope.selected_count.tolist() == [1, 1, 0]
        assert scope.candidate_count.tolist() == [2, 2, 2]
    np.testing.assert_array_equal(table.values, original)
    assert len(pd.read_csv(output.with_suffix(".features.csv"))) == 2


@pytest.mark.parametrize("thresholds", [[-0.1], [1.1], [np.nan]])
def test_report_rejects_invalid_thresholds(tmp_path, thresholds):
    with pytest.raises(ValueError, match="thresholds"):
        write_r2_threshold_report(None, tmp_path / "bad.csv", thresholds=thresholds)


def test_training_selection_failure_still_publishes_threshold_counts(tmp_path):
    table = FeatureTable(
        np.ones((4, 2)),
        np.array([1, 2, 3, 4]),
        np.zeros(4),
        ["mic::mic_rms", "mic::mic_freq_1Hz"],
        ["a", "a", "b", "b"],
        ["M001", "M002", "M001", "M002"],
    )
    reader = SimpleNamespace(record_summary=lambda _: table)
    with pytest.raises(ValueError, match="No features pass"):
        select_job(
            "unused",
            feature_catalog(table.feature_names),
            "r2",
            explicit.training_config()["selection"]["r2"],
            tmp_path,
            reader,
        )
    counts = pd.read_csv(tmp_path / "r2_threshold_counts.csv")
    assert counts.selected_count.eq(0).all()
