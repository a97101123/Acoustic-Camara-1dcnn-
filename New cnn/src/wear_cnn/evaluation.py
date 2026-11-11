from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .classification import (
    classification_from_metadata,
)
from .confusion_plot import save_confusion
from .model import build_model, resolve_device
from .training import predict_batches

_STORE_METADATA_KEYS = (
    "sensor",
    "nfft",
    "mic_nfft",
    "acc_nfft",
    "sample_rate",
    "mic_sample_rate",
    "acc_sample_rate",
    "sample_unit",
    "layout",
    "feature_profile",
    "sampling_json",
    "window_seconds",
    "hop_seconds",
    "normal_moderate_threshold",
    "moderate_severe_threshold",
    "feature_set_identity",
    "process",
    "condition",
    "class_names_json",
    "labeling_json",
    "evaluation_json",
    "classification_json",
    "labeling_json",
    "evaluation_json",
)


def _validate_checkpoint_schema(saved, feature_names, store_metadata):
    """Verify selected columns and preprocessing before reading evaluation values."""
    feature_count = len(feature_names)
    metadata = saved["metadata"]
    for key in _STORE_METADATA_KEYS:
        if key in metadata and metadata[key] != store_metadata.get(key):
            raise ValueError(f"Checkpoint/store metadata mismatch: {key}")
    indices = saved["feature_indices"]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= feature_count
            for i in indices
        )
    ):
        raise ValueError("Invalid checkpoint feature indices")
    if saved.get("feature_names") != [feature_names[i] for i in indices]:
        raise ValueError("Checkpoint feature names/order differ from evaluation schema")
    median, scale = np.asarray(saved["median"]), np.asarray(saved["scale"])
    if (
        median.shape != (len(indices),)
        or scale.shape != median.shape
        or not np.isfinite(median).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Invalid or nonfinite checkpoint scaling")


def _classification_metrics(truth, predicted, spec=None):
    """Produce the same metric schema for validation and held-out test data."""
    if spec is None:
        raise ValueError("Explicit classification spec is required")
    return {
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "classes": classification_report(
            truth,
            predicted,
            labels=spec.ids,
            target_names=spec.keys,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_hdf5(
    hdf5_path: str,
    checkpoint: str | list[str],
    output: str | Path,
    device: str,
    reader=None,
    make_plots=True,
    filenames=None,
    *,
    batch_size,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Evaluation requires PyTorch") from exc
    chosen = resolve_device(torch, device)
    from .filename_templates import EVAL_FILES

    filenames = filenames or EVAL_FILES
    checkpoints = [checkpoint] if isinstance(checkpoint, str) else checkpoint
    members = []
    spec = None
    for path in checkpoints:
        saved = torch.load(path, map_location=chosen, weights_only=False)
        member_spec = classification_from_metadata(saved["metadata"])
        if members and member_spec != spec:
            raise ValueError("Ensemble checkpoints must use the same classification")
        spec = member_spec
        model = build_model(
            torch, int(saved["input_channels"]), spec.num_classes, saved["model_config"]
        ).to(chosen)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        members.append((saved, model))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not members:
        raise ValueError("At least one checkpoint is required")
    groups = {saved["feature_group"] for saved, _ in members}
    if len(groups) != 1:
        raise ValueError("Ensemble checkpoints must use same feature group")
    feature_group = groups.pop()

    if reader is None:
        raise ValueError("An explicit configured FeatureReader is required")
    with h5py.File(hdf5_path, "r") as h:
        store_metadata = {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in h.attrs.items()
        }
        if classification_from_metadata(store_metadata) != spec:
            raise ValueError("Checkpoint/store classification metadata mismatch")
        sample_unit = store_metadata.get("sample_unit")
        if sample_unit != "chunk":
            raise ValueError("Only current chunk Feature Sets can be evaluated")
        feature_names = h[f"splits/train/{feature_group}/feature_names"].asstr()[:].tolist()
        for saved, _ in members:
            _validate_checkpoint_schema(saved, feature_names, store_metadata)
        union = sorted({i for saved, _ in members for i in saved["feature_indices"]})
        column_lookup = {original: position for position, original in enumerate(union)}

        def read_split(split):
            g = h[f"splits/{split}/{feature_group}"]
            if g["feature_names"].asstr()[:].tolist() != feature_names:
                raise ValueError(f"{split} feature schema differs from training")

            def decode(seq):
                return [v.decode() if isinstance(v, bytes) else str(v) for v in seq]

            units = decode(g["unit_ids"][:])
            values = reader.values(hdf5_path, feature_group, split, union)
            labels = g["labels"][:]
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Empty or nonfinite {split} feature values")
            if len(labels) != len(values) or not np.isin(labels, spec.ids).all():
                raise ValueError(f"Invalid {split} labels")
            return values, labels, decode(g["tool_ids"][:]), decode(g["record_ids"][:]), units

        val_x, val_y, val_tools, val_records, val_units = read_split("val")
        test_x, test_y, test_tools, test_records, test_units = read_split("test")

    def observations(tools, records, units, truth, probabilities):
        frame = pd.DataFrame({"ToolID": tools, "RecordID": records, "UnitID": units, "true": truth})
        for index, name in enumerate(spec.probability_columns):
            frame[name] = probabilities[:, index]
        keys = ["ToolID", "RecordID", "UnitID"]
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate chunk identities in evaluation")
        if (frame.groupby(keys)["true"].nunique() > 1).any():
            raise ValueError("Conflicting labels within evaluation unit")
        return frame.groupby(keys, as_index=False).agg(
            {"true": "first", **{column: "mean" for column in spec.probability_columns}}
        )

    def probabilities(x):
        outputs = []
        with torch.no_grad():
            for saved, model in members:
                indices = [column_lookup[i] for i in saved["feature_indices"]]
                selected = np.take(x, indices, axis=2)
                selected = ((selected - saved["median"]) / saved["scale"]).astype("float32")
                if not np.isfinite(selected).all():
                    raise ValueError("Nonfinite normalized evaluation values")
                selected = np.transpose(selected, (0, 2, 1))
                logits = predict_batches(torch, model, selected, chosen, batch_size)
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite model logits")
                probability = torch.softmax(torch.as_tensor(logits), 1).numpy()
                if not np.isfinite(probability).all():
                    raise ValueError("Nonfinite model probabilities")
                outputs.append(probability)
        return np.mean(outputs, axis=0)

    val_p = probabilities(val_x)
    val_frame = observations(val_tools, val_records, val_units, val_y, val_p)
    val_true = val_frame.true.to_numpy()
    threshold = None
    if spec.prediction_rule == "validation_threshold":
        val_severe = val_frame[spec.probability_columns[spec.severe_index]].to_numpy()
        thresholds = np.unique(np.concatenate(([0.0], val_severe, [1.0])))
        candidates = []
        for candidate in thresholds:
            base = spec.classify(val_frame[list(spec.probability_columns)].to_numpy(), candidate)
            candidates.append(
                (
                    float(candidate),
                    f1_score(val_true, base, average="macro", zero_division=0),
                    balanced_accuracy_score(val_true, base),
                )
            )
        threshold = max(candidates, key=lambda x: (x[1], x[2]))[0]
    val_pred = spec.classify(val_frame[list(spec.probability_columns)].to_numpy(), threshold)
    val_frame["predicted"] = val_pred
    val_frame.to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    raw_probs = probabilities(test_x)
    grouped = observations(test_tools, test_records, test_units, test_y, raw_probs)
    test_y = grouped.true.to_numpy()
    test_tools = grouped.ToolID.tolist()
    test_records = grouped.RecordID.tolist()
    probs = grouped[list(spec.probability_columns)].to_numpy()
    pred = spec.classify(probs, threshold)
    metrics = {
        **_classification_metrics(test_y, pred, spec),
        "validation": _classification_metrics(val_true, val_pred, spec),
        "severe_threshold": threshold,
        "ensemble_size": len(members),
    }
    metrics.update(
        sample_unit=sample_unit,
        test_count=len(test_y),
        validation_count=len(val_true),
        class_names=list(spec.names),
        severe_class_index=spec.severe_index,
        prediction_rule=spec.prediction_rule,
    )
    (output / filenames["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / filenames["deployment"]).write_text(
        json.dumps(
            {
                "checkpoints": [str(Path(p).resolve()) for p in checkpoints],
                "severe_probability_threshold": threshold,
                "ensemble_size": len(members),
                "store_metadata": store_metadata,
                "inference": members[0][0]["inference"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    prediction_frame = grouped.copy()
    prediction_frame["predicted"] = pred
    prediction_frame.to_csv(output / filenames["predictions"], index=False, encoding="utf-8-sig")
    if sample_unit == "chunk":
        count_frame = (
            grouped.groupby(["ToolID", "RecordID"], as_index=False)
            .size()
            .rename(columns={"size": "evaluated_chunks"})
        )
        count_frame.to_csv(output / filenames["sample_counts"], index=False, encoding="utf-8-sig")
    per_tool = []
    for tool, group in pd.DataFrame({"tool": test_tools, "y": test_y, "p": pred}).groupby("tool"):
        per_tool.append(
            {
                "ToolID": tool,
                "records": len(group),
                "accuracy": accuracy_score(group.y, group.p),
                "macro_f1": f1_score(group.y, group.p, average="macro", zero_division=0),
            }
        )
    pd.DataFrame(per_tool).to_csv(output / filenames["per_tool"], index=False, encoding="utf-8-sig")
    matrix = confusion_matrix(test_y, pred, labels=spec.ids)
    save_confusion(
        matrix, output, make_plots=make_plots, filenames=filenames, class_names=spec.names
    )
    return output
