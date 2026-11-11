import json

import h5py
import numpy as np
import pandas as pd
import pytest

from wear_cnn.classification import classification_from_config
from wear_cnn.splitting import assign_splits, filter_stage_records, validate_dataset
from wear_cnn.trainer_batch import split_store


def binary_config():
    return {
        "dataset": {
            "stage_ids": list(range(1, 14)),
            "exclude_stages": [7],
            "labeling": {
                "method": "stage_groups",
                "groups": [list(range(1, 7)), list(range(8, 14))],
            },
            "class_names": ["Low", "High"],
        },
        "training": {"severe_class_index": 1},
        "evaluation": {
            "strategy": "fixed_knife",
            "test_knife": 9,
            "val_knives": [7, 8],
            "prediction_rule": "argmax",
        },
    }


def test_binary_filter_preserves_original_feature_rows(tmp_path):
    feature = tmp_path / "features.h5"
    target = tmp_path / "split.h5"
    rows = [(tool, stage) for tool in range(1, 10) for stage in range(1, 15)]
    with h5py.File(feature, "w") as h:
        h.attrs["sample_unit"] = "chunk"
        group = h.create_group("samples")
        group["values"] = np.arange(len(rows), dtype=np.float32)[:, None, None]
        for key, values in {
            "tool_ids": [f"T{tool:02}" for tool, stage in rows],
            "record_ids": [f"M{stage:03}" for tool, stage in rows],
            "unit_ids": ["chunk_000000"] * len(rows),
            "feature_names": ["mic::mic_rms"],
        }.items():
            group.create_dataset(key, data=values, dtype=h5py.string_dtype())
        group["wear"] = [stage * 0.01 for tool, stage in rows]
    split_store(feature, target, 42, "synthetic", binary_config())
    with h5py.File(target) as h:
        assert len(json.loads(h.attrs["classification_json"])["names"]) == 2
        for split, knives in (("train", set(range(1, 7))), ("val", {7, 8}), ("test", {9})):
            group = h[f"splits/{split}/all"]
            expected = [
                index
                for index, (tool, stage) in enumerate(rows)
                if tool in knives and stage in set(range(1, 14)) - {7}
            ]
            np.testing.assert_array_equal(group["values"][:, 0, 0], expected)
            assert set(group["tool_ids"].asstr()[:]) == {f"T{tool:02}" for tool in knives}
            np.testing.assert_array_equal(
                group["labels"][:], [int(rows[index][1] >= 8) for index in expected]
            )
        assert "normal_moderate_threshold" not in h.attrs
    manifest = pd.read_csv(tmp_path / "split_manifest.csv")
    assert len(manifest) == 108
    assert set(manifest.Stage) == set(range(1, 14)) - {7}
    invalid = binary_config()
    invalid["evaluation"]["test_knife"] = 99
    incomplete = tmp_path / "incomplete.h5"
    with pytest.raises(ValueError, match="Unknown"):
        split_store(feature, incomplete, 42, "synthetic", invalid)
    assert not incomplete.exists()
    split_store(feature, incomplete, 42, "synthetic", binary_config())
    with h5py.File(incomplete) as h:
        assert len(h["splits/test/all/labels"]) == 12


@pytest.mark.parametrize(
    "groups", [[[1], [1]], [[], [2]], [[True], [2]], [[0], [2]], [[1.5], [2]], [[1]]]
)
def test_rejects_invalid_groups(groups):
    with pytest.raises(ValueError):
        validate_dataset({"labeling": {"method": "stage_groups", "groups": groups}})


@pytest.mark.parametrize(
    "evaluation",
    [
        {"strategy": "fixed_knife", "test_knife": 9, "val_knives": [9]},
        {"strategy": "fixed_knife", "test_knife": 10, "val_knives": [7, 8]},
        {"strategy": "fixed_knife", "test_knife": 9, "val_knives": [7, 7]},
        {"strategy": "random_tools", "test_knife": 9},
    ],
)
def test_rejects_invalid_fixed_splits(evaluation):
    frame = pd.DataFrame({"ToolID": [f"T{tool:02}" for tool in range(1, 10)]})
    with pytest.raises(ValueError):
        assign_splits(frame, evaluation, 42)


def test_stage_filters_do_not_guess_arbitrary_record_ids():
    frame = pd.DataFrame({"ToolID": ["T01"], "RecordID": ["S1_H1"]})
    with pytest.raises(ValueError, match="RecordID"):
        filter_stage_records(frame, binary_config()["dataset"])


def test_three_stage_classes_and_config_examples():
    cfg = binary_config()
    cfg["dataset"]["labeling"]["groups"] = [[1, 2], [3, 4], [8, 9]]
    cfg["dataset"]["class_names"] = ["Initial", "Normal", "Severe"]
    cfg["training"]["severe_class_index"] = 2
    assert classification_from_config(cfg).num_classes == 3
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("turning_training.yaml", "turning_binary_training.yaml", "drilling_training.yaml"):
        from wear_cnn.run_io import load_config

        config = load_config(root / "configs" / name)[1]
        validate_dataset(config.get("dataset", {}))
        classification_from_config(config)
