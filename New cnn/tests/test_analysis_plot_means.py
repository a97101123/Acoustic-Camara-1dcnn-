import matplotlib
import numpy as np
import pandas as pd
import pytest
import settings as explicit

matplotlib.use("Agg")

from wear_cnn.features.plots import analysis_position_means, plot_r2_overlay


def test_stage_cap_precedes_cross_experiment_hole_mean():
    # S1 mean is 6; S2 mean is 22. The hole means must not mix stages.
    records = ["S1_H01", "S1_H02", "S1_H01", "S1_H02", "S2_H01", "S2_H01"]
    tools = ["a", "a", "b", "b", "a", "b"]
    wear, values, positions, counts = analysis_position_means(
        [1, 2, 3, 4, 5, 7], np.array([[1], [3], [5], [15], [20], [24]]), records, tools
    )
    np.testing.assert_allclose(values[:, 0], [3, 4.5, 21])
    np.testing.assert_allclose(wear, [2, 3, 6])
    assert positions == ["S1_H01", "S1_H02", "S2_H01"]
    assert counts.tolist() == [2, 2, 2]


def test_duplicate_experiment_position_is_rejected():
    with pytest.raises(ValueError, match="one observation"):
        analysis_position_means([1, 1], [[2], [4]], ["S1_H01"] * 2, ["a"] * 2)


def test_overlay_exports_recomputed_r2_and_one_point_per_position(tmp_path):
    output = tmp_path / "overlay.png"
    plot_r2_overlay(
        [1, 2, 3, 1, 2, 3],
        [[1, 3], [2, 2], [3, 1]] * 2,
        ["a", "b"],
        [0.1, 0.2],
        output,
        dpi=explicit.training_config()["output"]["plot_dpi"],
        record_ids=["M001", "M002", "M003"] * 2,
        tool_ids=["a"] * 3 + ["b"] * 3,
    )
    assert output.is_file()
    assert output.with_name("overlay.trend.png").is_file()
    means = pd.read_csv(output.with_suffix(".means.csv"))
    assert len(means) == 3
    assert means.experiment_count.tolist() == [2, 2, 2]
    assert means.machining_progress.tolist() == [1, 2, 3]
    scores = pd.read_csv(output.with_suffix(".scores.csv"))
    np.testing.assert_allclose(scores.position_mean_r2, [1, 1])
    np.testing.assert_allclose(scores.selection_r2, [0.1, 0.2])


def test_hole_progression_keeps_stage_offsets_and_sorts_positions(tmp_path):
    output = tmp_path / "holes.png"
    plot_r2_overlay(
        [30, 10, 20],
        [[3], [1], [2]],
        ["rms"],
        [0.1],
        output,
        dpi=explicit.training_config()["output"]["plot_dpi"],
        record_ids=["S3_H05", "S1_H01", "S2_H01"],
        tool_ids=["a"] * 3,
    )
    means = pd.read_csv(output.with_suffix(".means.csv"))
    assert means.position.tolist() == ["S1_H01", "S2_H01", "S3_H05"]
    assert means.machining_progress.tolist() == [1, 6, 15]
    assert means.mean_wear.tolist() == [10, 20, 30]


def test_no_passing_features_produces_explicit_empty_plot(tmp_path):
    output = tmp_path / "empty.png"
    plot_r2_overlay(
        [1, 2, 3],
        np.empty((3, 0)),
        [],
        [],
        output,
        dpi=explicit.training_config()["output"]["plot_dpi"],
        record_ids=["M001", "M002", "M003"],
        tool_ids=["a"] * 3,
    )
    assert output.is_file()
    assert output.with_name("empty.trend.png").is_file()
    assert pd.read_csv(output.with_suffix(".scores.csv")).empty
    assert len(pd.read_csv(output.with_suffix(".means.csv"))) == 3
