"""The class order and decision rule shared by evaluation and WAV inference."""

import json
from dataclasses import dataclass

import numpy as np

CLASS_IDS = (0, 1, 2)
CLASS_KEYS = ("initial", "normal", "severe")
CLASS_NAMES = ("初期", "正常", "嚴重")
PROBABILITY_COLUMNS = tuple(f"p_{name}" for name in CLASS_KEYS)


@dataclass(frozen=True)
class ClassificationSpec:
    names: tuple
    keys: tuple
    severe_index: int
    prediction_rule: str

    def __post_init__(self):
        for field in ("names", "keys"):
            values = tuple(getattr(self, field))
            if (
                len(values) not in (2, 3)
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"Classification {field} must contain 2 or 3 unique names")
            object.__setattr__(self, field, values)
        if len(self.names) != len(self.keys):
            raise ValueError("Classification names and keys must have equal length")
        if type(self.severe_index) is not int or self.severe_index not in self.ids:
            raise ValueError("severe_class_index must identify a configured class")
        if self.prediction_rule not in ("argmax", "validation_threshold"):
            raise ValueError("prediction_rule must be argmax or validation_threshold")

    @property
    def ids(self):
        return tuple(range(len(self.names)))

    @property
    def num_classes(self):
        return len(self.names)

    @property
    def probability_columns(self):
        return tuple(f"p_{key}" for key in self.keys)

    def to_metadata(self):
        return {
            "classification_json": json.dumps(
                {
                    "names": self.names,
                    "keys": self.keys,
                    "severe_index": self.severe_index,
                    "prediction_rule": self.prediction_rule,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        }

    def classify(self, probabilities, threshold=None):
        values = np.asarray(probabilities)
        if values.ndim < 1 or values.shape[-1] != self.num_classes or not np.isfinite(values).all():
            raise ValueError("Probabilities must match the configured classes and be finite")
        if self.prediction_rule == "argmax":
            return np.argmax(values, axis=-1)
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Severe probability threshold must be finite and within [0, 1]")
        ordinary = [index for index in self.ids if index != self.severe_index]
        best = np.asarray(ordinary)[np.argmax(values[..., ordinary], axis=-1)]
        return np.where(values[..., self.severe_index] >= threshold, self.severe_index, best)


def classification_from_config(config):
    if not isinstance(config, dict):
        raise ValueError("Classification config must be a mapping")
    for key in ("dataset", "training", "evaluation"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{key} must be a mapping")
    dataset = config["dataset"]
    labeling = dataset["labeling"]
    if not isinstance(labeling, dict):
        raise ValueError("dataset.labeling must be a mapping")
    stage_groups = labeling.get("method") == "stage_groups"
    groups = labeling.get("groups", [])
    if stage_groups and not isinstance(groups, list):
        raise ValueError("Stage groups must be a list")
    count = len(groups) if stage_groups else 3
    names = dataset["class_names"]
    if not isinstance(names, (list, tuple)) or len(names) != count:
        raise ValueError("class_names must match the labeling class count")
    severe = config["training"]["severe_class_index"]
    keys = (
        CLASS_KEYS
        if count == 3 and tuple(names) == CLASS_NAMES and severe == 2
        else tuple("severe" if index == severe else f"class_{index}" for index in range(count))
    )
    rule = config["evaluation"]["prediction_rule"]
    return ClassificationSpec(tuple(names), keys, severe, rule)


def classification_from_metadata(metadata):
    if "classification_json" not in metadata:
        raise ValueError("Missing classification_json metadata; rebuild using explicit config")
    try:
        payload = json.loads(metadata["classification_json"])
        if not isinstance(payload, dict) or set(payload) != {
            "names",
            "keys",
            "severe_index",
            "prediction_rule",
        }:
            raise ValueError("Invalid classification schema")
        if not isinstance(payload["names"], list) or not isinstance(payload["keys"], list):
            raise ValueError("Invalid class names or keys")
        return ClassificationSpec(**payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid classification_json metadata") from exc


def classify_probabilities(probabilities, severe_threshold, spec=None):
    return spec.classify(probabilities, severe_threshold)
