import json

import numpy as np
import pytest
import settings as explicit

from wear_cnn.classification import (
    ClassificationSpec,
    classification_from_config,
    classification_from_metadata,
    classify_probabilities,
)
from wear_cnn.confusion_plot import plot_confusion
from wear_cnn.inference import _prediction_result


def binary_config(rule="argmax", severe=1):
    return {
        "dataset": {
            "labeling": {"method": "stage_groups", "groups": [[1, 2], [4, 5]]},
            "class_names": ["Low", "High"],
        },
        "training": {"severe_class_index": severe},
        "evaluation": {"prediction_rule": rule},
    }


def test_binary_spec_roundtrip_and_argmax():
    spec = classification_from_config(binary_config())
    assert spec.num_classes == 2 and spec.ids == (0, 1)
    assert spec.probability_columns == ("p_class_0", "p_severe")
    assert classification_from_metadata(spec.to_metadata()) == spec
    values = np.array([[0.2, 0.8], [0.7, 0.3]])
    assert spec.classify(values).tolist() == [1, 0]
    assert _prediction_result(values[0], None, spec)["class_name"] == "High"


def test_explicit_three_class_settings_roundtrip():
    spec = explicit.classification()
    assert spec == classification_from_metadata(spec.to_metadata())
    assert spec.probability_columns == ("p_initial", "p_normal", "p_severe")
    assert classify_probabilities([[0.4, 0.1, 0.5]], 0.5, explicit.classification()).tolist() == [2]


def test_threshold_supports_nonfinal_positive_class():
    spec = classification_from_config(binary_config("validation_threshold", 0))
    assert spec.classify([[0.4, 0.6], [0.2, 0.8]], 0.3).tolist() == [0, 1]


@pytest.mark.parametrize("severe", [-1, 2, True, 0.5])
def test_invalid_positive_class_rejected(severe):
    with pytest.raises(ValueError, match="severe_class_index"):
        classification_from_config(binary_config(severe=severe))


@pytest.mark.parametrize("payload", ["null", "{}", "broken", "[]"])
def test_explicit_malformed_metadata_never_falls_back(payload):
    with pytest.raises(ValueError, match="classification_json"):
        classification_from_metadata({"classification_json": payload})


def test_invalid_shapes_and_rule_rejected():
    spec = classification_from_config(binary_config())
    with pytest.raises(ValueError, match="Probabilities"):
        spec.classify([[0.1, 0.2, 0.7]])
    with pytest.raises(ValueError, match="prediction_rule"):
        classification_from_config(binary_config("bogus"))
    payload = json.loads(spec.to_metadata()["classification_json"])
    payload["names"] = ["same", "same"]
    with pytest.raises(ValueError, match="classification_json"):
        classification_from_metadata({"classification_json": json.dumps(payload)})


def test_binary_model_and_matrix(tmp_path):
    import torch

    from wear_cnn.model import build_model

    model = build_model(torch, 4, 2, explicit.model())
    assert model(torch.ones((3, 4, 8))).shape == (3, 2)
    plot_confusion(np.array([[3, 1], [2, 4]]), tmp_path / "binary.png", ["Low", "High"])
    assert (tmp_path / "binary.png").is_file()


def test_binary_checkpoint_evaluation_and_ensemble_schema(tmp_path):
    import h5py
    import pandas as pd
    import torch

    from wear_cnn.artifact_validation import valid_checkpoint
    from wear_cnn.evaluation import evaluate_hdf5
    from wear_cnn.model import build_model

    torch.set_num_threads(1)
    spec = classification_from_config(binary_config())
    metadata = {"feature_set_identity": "binary", "sample_unit": "chunk", **spec.to_metadata()}
    store = tmp_path / "binary.h5"
    with h5py.File(store, "w") as handle:
        handle.attrs.update(metadata)
        for split in ("train", "val", "test"):
            node = handle.create_group(f"splits/{split}/all")
            node["values"] = np.ones((2, 4, 2), dtype=np.float32)
            node["labels"] = [0, 1]
            node["wear"] = [1, 9]
            for key, values in {
                "feature_names": ["a", "b"],
                "tool_ids": [split] * 2,
                "record_ids": ["0", "1"],
                "unit_ids": ["0", "1"],
            }.items():
                node.create_dataset(key, data=values, dtype=h5py.string_dtype())
    checkpoint = tmp_path / "binary.pt"
    saved = {
        "model_config": explicit.model(),
        "inference_batch_size": explicit.evaluation()["inference_batch_size"],
        "inference": explicit.evaluation()["inference"],
        "state_dict": build_model(torch, 2, 2, explicit.model()).state_dict(),
        "input_channels": 2,
        "metadata": metadata,
        "feature_names": ["a", "b"],
        "feature_indices": [0, 1],
        "feature_group": "all",
        "median": np.zeros(2),
        "scale": np.ones(2),
        "training_identity": "binary",
        "epochs": 1,
    }
    torch.save(saved, checkpoint)
    assert valid_checkpoint(checkpoint)
    report = evaluate_hdf5(
        str(store),
        str(checkpoint),
        tmp_path / "report",
        "cpu",
        batch_size=explicit.evaluation()["batch_size"],
        reader=explicit.reader(),
    )
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics["severe_threshold"] is None and metrics["class_names"] == ["Low", "High"]
    assert np.loadtxt(report / "confusion_matrix.csv", delimiter=",").shape == (2, 2)
    predictions = pd.read_csv(report / "predictions.csv")
    assert {"p_class_0", "p_severe"} <= set(predictions)
    changed = ClassificationSpec(("A", "B"), ("class_0", "severe"), 1, "argmax")
    saved["metadata"] = {**metadata, **changed.to_metadata()}
    other = tmp_path / "other.pt"
    torch.save(saved, other)
    with pytest.raises(ValueError, match="same classification"):
        evaluate_hdf5(
            str(store),
            [str(checkpoint), str(other)],
            tmp_path / "bad",
            "cpu",
            batch_size=explicit.evaluation()["batch_size"],
            reader=explicit.reader(),
        )
