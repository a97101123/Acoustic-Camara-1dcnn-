from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .classification import classification_from_metadata
from .features.extract import _read
from .model import build_model, resolve_device


def _predict_checkpoint(torch, saved, values, device):
    """Restore one ensemble member and predict already-scaled CNN inputs."""
    spec = classification_from_metadata(saved["metadata"])
    model = build_model(
        torch, int(saved["input_channels"]), spec.num_classes, saved["model_config"]
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        batches = [
            torch.softmax(
                model(
                    torch.tensor(
                        values[start : start + saved["inference_batch_size"]], device=device
                    )
                ),
                1,
            )
            .cpu()
            .numpy()
            for start in range(0, len(values), saved["inference_batch_size"])
        ]
    return np.concatenate(batches)


def _prediction_result(probability, threshold, spec=None):
    if spec is None:
        raise ValueError("Explicit classification spec is required")
    predicted = int(spec.classify(probability, threshold))
    return {
        "class_id": predicted,
        "class_name": spec.names[predicted],
        "probabilities": probability.tolist(),
        "severe_threshold": threshold,
    }


def predict_wav(wavs: list[str], deployment: str, device: str) -> dict:
    spec = json.loads(Path(deployment).read_text(encoding="utf-8"))
    meta = _deployment_metadata(spec)
    if meta["sensor"] == "fusion":
        raise ValueError("Use predict-fusion for a MIC+ACC deployment")
    return _predict_analysis(wavs, spec, device)


def predict_fusion(
    mic_wav: str, acc_wavs: list[str], deployment: str, device: str, *,
    mic_start=None, acc_start=None, duration=None,
) -> dict:
    if len(acc_wavs) != 3:
        raise ValueError("Fusion requires ACC X/Y/Z WAV files")
    spec = json.loads(Path(deployment).read_text(encoding="utf-8"))
    meta = _deployment_metadata(spec)
    if meta["sensor"] != "fusion":
        raise ValueError("Deployment is not a fusion model")
    if (mic_start is None) != (acc_start is None):
        raise ValueError("Supply both MIC and ACC start offsets for an explicit alignment")
    starts = None if mic_start is None else [mic_start, acc_start, acc_start, acc_start]
    return _predict_analysis([mic_wav, *acc_wavs], spec, device, starts=starts, duration=duration)


def _deployment_metadata(spec):
    meta = spec.get("store_metadata", {})
    if (
        meta.get("feature_profile") != "analysis_short_time_v1"
        or meta.get("sample_unit") != "chunk"
    ):
        raise ValueError(
            "Legacy deployments are unsupported; rebuild with build-features/train-batch"
        )
    sampling = json.loads(meta["sampling_json"])
    if sampling.get("chunk_retention") != "all_complete":
        raise ValueError("Deployment requires chunk_retention=all_complete; rebuild and retrain")
    return meta


def _predict_analysis(wavs: list[str], spec: dict, device: str, *, starts=None, duration=None) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Inference requires PyTorch") from exc

    from .features.analysis_sampling import analysis_sequence, chunk_sequence

    meta = _deployment_metadata(spec)
    sampling = json.loads(meta["sampling_json"])
    sensor = meta["sensor"]
    if sensor == "fusion":
        sources = [("mic::mic", "mic", int(meta["mic_nfft"]))] + [
            (f"acc::{axis}", "acc", int(meta["acc_nfft"])) for axis in "xyz"
        ]
    elif sensor == "mic":
        sources = [("mic", "mic", int(meta["nfft"]))]
    else:
        sources = [(axis, "acc", int(meta["nfft"])) for axis in "xyz"]
    if len(wavs) != len(sources):
        raise ValueError(f"Expected {len(sources)} WAV files")
    sequences, names = [], []
    if starts is None:
        starts = [spec["inference"]["start_seconds"]] * len(wavs)
    if duration is None:
        duration = spec["inference"]["duration_seconds"]
    cut_durations = []
    for path, (prefix, kind, nfft), start in zip(wavs, sources, starts, strict=True):
        rate, wave = _read(path, start, duration)
        cut_durations.append((len(wave) / rate, rate))
        if sampling.get("target_sample_rate") is None and rate != sampling["sample_rate"]:
            raise ValueError("Inference sample rate differs from training")
        sequence, base = analysis_sequence(rate, wave, nfft, kind, sampling)
        sequences.append(sequence)
        names.extend(f"{prefix}_{name}" for name in base)
    if max(t for t, _ in cut_durations) - min(t for t, _ in cut_durations) > max(
        1 / rate for _, rate in cut_durations
    ) + 1e-9:
        raise ValueError("Sensor cut durations differ; supply aligned starts and a shared duration")
    if len({len(values) for values in sequences}) != 1:
        raise ValueError("MIC/ACC feature lengths must match for chunk fusion")
    full, _ = chunk_sequence(np.concatenate(sequences, axis=1), sampling)
    lookup = {name: index for index, name in enumerate(names)}
    if sensor in ("mic", "acc"):
        lookup.update({sensor + "::" + name: index for name, index in list(lookup.items())})
    chosen = resolve_device(torch, device)
    classification = classification_from_metadata(meta)
    members = []
    for checkpoint in spec["checkpoints"]:
        saved = torch.load(checkpoint, map_location=chosen, weights_only=False)
        if classification_from_metadata(saved["metadata"]) != classification:
            raise ValueError("Checkpoint classification differs from deployment")
        if saved["metadata"].get("sampling_json") != meta["sampling_json"]:
            raise ValueError("Checkpoint sampling differs from deployment")
        for key in ("labeling_json", "evaluation_json"):
            if saved["metadata"].get(key) != meta.get(key):
                raise ValueError(f"Checkpoint {key} differs from deployment")
        values = full[:, :, [lookup[name] for name in saved["feature_names"]]]
        values = ((values - saved["median"]) / saved["scale"]).astype("float32").transpose(0, 2, 1)
        members.append(_predict_checkpoint(torch, saved, values, chosen))
    probabilities = np.mean(members, axis=0)
    threshold = spec["severe_probability_threshold"]
    threshold = None if threshold is None else float(threshold)
    classes = classification.classify(probabilities, threshold)
    probability = probabilities.mean(axis=0)
    return {
        **_prediction_result(probability, threshold, classification),
        "sample_unit": "chunk",
        "chunk_count": len(classes),
        "chunks": [
            {
                "unit_id": f"chunk_{index:06d}",
                "class_id": int(label),
                "class_name": classification.names[label],
                "probabilities": probabilities[index].tolist(),
            }
            for index, label in enumerate(classes)
        ],
    }
