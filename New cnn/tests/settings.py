"""Explicit test settings sourced from the shipped functional YAML files."""

from copy import deepcopy
from pathlib import Path

from wear_cnn.run_io import load_config

ROOT = Path(__file__).resolve().parents[1]


def merge(value, changes):
    result = deepcopy(value)
    for key, replacement in changes.items():
        if isinstance(replacement, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], replacement)
        else:
            result[key] = deepcopy(replacement)
    return result


def feature_config(changes=None):
    return merge(load_config(ROOT / "configs/turning_features.yaml")[1], changes or {})


def training_config(changes=None):
    result = merge(load_config(ROOT / "configs/turning_training.yaml")[1], changes or {})
    if result["evaluation"]["strategy"] == "fixed_knife":
        for key in (
            "test_fraction",
            "validation_fraction",
            "minimum_test_tools",
            "minimum_validation_tools",
            "minimum_total_tools",
        ):
            result["evaluation"].pop(key, None)
    return result


def sampling(changes=None):
    return merge(feature_config()["sampling"], changes or {})


def io(changes=None):
    return merge(feature_config()["io"], changes or {})


def model(changes=None):
    return merge(training_config()["model"], changes or {})


def training(changes=None):
    cfg = training_config()
    return merge(
        {
            **cfg["training"],
            "model": cfg["model"],
            "inference_batch_size": cfg["evaluation"]["inference_batch_size"],
            "inference": cfg["evaluation"]["inference"],
        },
        changes or {},
    )


def evaluation(changes=None):
    return merge(training_config()["evaluation"], changes or {})


def preprocessing(changes=None):
    return merge(training_config()["preprocessing"], changes or {})


def classification():
    from wear_cnn.classification import classification_from_config

    return classification_from_config(training_config())


def prepared(path):
    from wear_cnn.features.reader import FeatureReader
    from wear_cnn.training import prepare_training_data

    return prepare_training_data(path, None, "all", FeatureReader(io_config=io()), preprocessing())


def reader():
    from wear_cnn.features.reader import FeatureReader

    return FeatureReader(io_config=io())
