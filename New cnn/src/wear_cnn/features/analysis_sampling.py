"""Short-time features split by duration; 48 kHz/NFFT128/75% gives 750 frames.

This is an explicit sampling profile, not a bitwise MATLAB STFT reproduction.
Labels and tool splits are supplied by the existing train-only pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, stft, windows
from scipy.stats import kurtosis, skew

from ..classification import CLASS_NAMES
from .extract import _read
from .io import compression_options
from .views import feature_view

TIME_NAMES = ["mean", "max", "rms", "variance", "std", "skewness", "kurtosis"]


def sampling_options(options: dict | None) -> dict:
    from ..configuration import fields

    cfg = dict(
        fields(
            options,
            (
                "mode",
                "overlap_fraction",
                "chunk_seconds",
                "chunk_retention",
                "segments_per_record",
                "trim_fraction",
                "clip_frequency_at_mean",
                "target_sample_rate",
                "time_backend",
                "frame_batch_size",
                "verbose",
                "acc_bands",
            ),
            "sampling",
            (
                "window_samples",
                "overlap_samples",
                "hop_samples",
                "chunk_frames",
                "sample_rate",
                "effective_chunk_seconds",
            ),
        )
    )
    if "block_size" in cfg:
        raise ValueError("block_size was removed; all complete chunks are retained")
    if cfg["chunk_retention"] != "all_complete":
        raise ValueError("chunk_retention must be all_complete; rebuild old feature sets")
    if cfg["time_backend"] not in ("scipy", "matrix"):
        raise ValueError("time_backend must be scipy or matrix")
    block = cfg["frame_batch_size"]
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError("frame_batch_size must be a positive integer")
    for key in ("segments_per_record",):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"sampling.{key} must be an integer")
        cfg[key] = int(value)
    if cfg["mode"] != "analysis_chunks":
        raise ValueError("Invalid analysis sampling mode")
    if not 0 <= float(cfg["overlap_fraction"]) < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    if not np.isfinite(float(cfg["chunk_seconds"])) or float(cfg["chunk_seconds"]) <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    if cfg["segments_per_record"] < 1:
        raise ValueError("Segment count must be positive")
    if not 0 <= float(cfg["trim_fraction"]) < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    target = cfg["target_sample_rate"]
    if target is not None and (isinstance(target, bool) or int(target) != target or target < 1):
        raise ValueError("target_sample_rate must be a positive integer or null")
    if any(type(cfg[key]) is not bool for key in ("clip_frequency_at_mean", "verbose")):
        raise ValueError("sampling.clip_frequency_at_mean and sampling.verbose must be booleans")
    bands = cfg["acc_bands"]
    if bands is not None and (
        not isinstance(bands, list)
        or not bands
        or any(
            not isinstance(band, (list, tuple))
            or len(band) != 2
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in band)
            or not 0 <= band[0] < band[1]
            for band in bands
        )
    ):
        raise ValueError("sampling.acc_bands must be null or contain finite increasing frequency pairs")
    return cfg


def resolve_sampling(options: dict, rate: int, nfft: int) -> dict:
    cfg = sampling_options(options)
    if rate <= 0 or nfft < 2 or nfft % 2:
        raise ValueError("Expected positive sample rate and even NFFT >= 2")
    rate = int(cfg["target_sample_rate"] or rate)
    overlap = int(nfft * float(cfg["overlap_fraction"]))
    hop = nfft - overlap
    frames = int(float(cfg["chunk_seconds"]) * rate / hop)
    if frames < 1:
        raise ValueError("chunk_seconds is shorter than one feature hop")
    cfg.update(
        window_samples=nfft,
        overlap_samples=overlap,
        hop_samples=hop,
        chunk_frames=frames,
        sample_rate=rate,
        effective_chunk_seconds=frames * hop / rate,
    )
    return cfg


def _time_statistics(frames, backend):
    if backend == "scipy" or frames.shape[1] < 4:
        return np.column_stack(
            [
                frames.mean(axis=1),
                frames.max(axis=1),
                np.sqrt((frames**2).mean(axis=1)),
                frames.var(axis=1, ddof=1),
                frames.std(axis=1, ddof=1),
                skew(frames, axis=1, bias=False),
                kurtosis(frames, axis=1, bias=False, fisher=False),
            ]
        )
    n = frames.shape[1]
    mean = frames.mean(axis=1)
    centered = frames - mean[:, None]
    square = centered * centered
    m2 = square.mean(axis=1)
    variance = m2 * (n / (n - 1))
    valid = m2 > (np.finfo(frames.dtype).eps * mean) ** 2
    denominator = np.where(valid, m2, 1.0)
    asymmetry = (square * centered).mean(axis=1) / denominator**1.5
    asymmetry *= np.sqrt(n * (n - 1)) / (n - 2)
    fourth = (square * square).mean(axis=1) / denominator**2
    fourth = ((n * n - 1) * fourth - 3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) + 3
    return np.column_stack(
        [
            mean,
            frames.max(axis=1),
            np.sqrt((frames * frames).mean(axis=1)),
            variance,
            np.sqrt(variance),
            np.where(valid, asymmetry, np.nan),
            np.where(valid, fourth, np.nan),
        ]
    )


def analysis_sequence(rate: int, wave: np.ndarray, nfft: int, sensor: str, options: dict):
    """Return [time, feature], keeping short-time frequency and time features."""
    cfg = resolve_sampling(options, rate, nfft)
    if rate != cfg["sample_rate"]:
        divisor = math.gcd(rate, cfg["sample_rate"])
        wave = resample_poly(wave, cfg["sample_rate"] // divisor, rate // divisor)
    rate = cfg["sample_rate"]
    window = cfg["window_samples"]
    hop = window - cfg["overlap_samples"]
    if sensor not in ("mic", "acc") or nfft < window or nfft % 2:
        raise ValueError("Expected MIC/ACC and an even NFFT >= window_samples")
    parts = np.array_split(wave, cfg["segments_per_record"])
    trimmed = []
    for part in parts:
        cut = int(len(part) * cfg["trim_fraction"])
        trimmed.append(part[cut : len(part) - cut] if cut else part)
    signal = np.concatenate(trimmed)
    if len(signal) < window:
        raise ValueError("Trimmed recording is shorter than one feature window")
    hann = windows.hann(window, sym=False)
    frequencies, _, spectrum = stft(
        signal,
        fs=rate,
        window=hann,
        nperseg=window,
        noverlap=cfg["overlap_samples"],
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    if sensor == "mic" or cfg["acc_bands"] is None:
        # MATLAB's centered indices 65:128 for NFFT=128 retain DC, omit Nyquist.
        frequency_values = power[:-1].T
        frequency_names = [f"freq_{f:.3f}Hz" for f in frequencies[:-1]]
    else:
        frequency_values = []
        frequency_names = []
        for low, high in cfg["acc_bands"]:
            indices = np.flatnonzero((frequencies >= low) & (frequencies <= high))
            if not len(indices):
                raise ValueError(f"No FFT bins for ACC band {low}-{high} Hz")
            frequency_values.append(power[indices].max(axis=0))
            frequency_names.append(f"freq_band_{low}_{high}Hz")
        frequency_values = np.stack(frequency_values, axis=1)
    if cfg["clip_frequency_at_mean"]:
        frequency_values = np.minimum(frequency_values, frequency_values.mean(axis=0))
    normalized = signal - signal.mean()
    peak = np.max(np.abs(normalized))
    if peak:
        normalized = normalized / peak
    frames = np.lib.stride_tricks.sliding_window_view(normalized, window)[::hop]
    backend = cfg["time_backend"]
    if backend == "matrix":
        time_values = np.empty((len(frames), 7), dtype=np.float64)
        block = cfg["frame_batch_size"]
        for start in range(0, len(frames), block):
            time_values[start : start + block] = _time_statistics(
                frames[start : start + block] * hann, backend
            )
    else:
        time_values = _time_statistics(frames * hann, backend)
    time_values = np.nan_to_num(time_values, nan=0.0)
    return np.concatenate([time_values, frequency_values], axis=1).astype(
        np.float32
    ), TIME_NAMES + frequency_names


def chunk_sequence(sequence: np.ndarray, options: dict):
    cfg = options
    if "chunk_frames" not in cfg:
        raise ValueError("Resolve sampling using the WAV sample rate and NFFT first")
    available = len(sequence) // cfg["chunk_frames"]
    if cfg.get("chunk_retention") != "all_complete" or "block_size" in cfg:
        raise ValueError("Only all_complete chunk retention is supported; rebuild old features")
    retained = available
    if not retained:
        raise ValueError(f"Recording has {available} full chunks; needs at least 1")
    used = retained * cfg["chunk_frames"]
    return sequence[:used].reshape(retained, cfg["chunk_frames"], sequence.shape[1]), available


def build_analysis_hdf5(
    frame: pd.DataFrame,
    sensor: str,
    nfft: int,
    output: str | Path,
    options: dict,
    independent=False,
    *,
    io_config,
):
    cfg = sampling_options(options)
    storage = compression_options(io_config)
    if independent:
        frame = frame.copy()
        frame["Split"] = "train"
        frame["Label"] = -1
    if frame.duplicated(["ToolID", "RecordID"]).any():
        raise ValueError("Analysis sampling requires unique ToolID+RecordID manifest rows")
    if (frame.groupby("ToolID")["Split"].nunique() > 1).any():
        raise ValueError("One physical tool cannot span multiple splits")
    columns = ["FileName"] if sensor == "mic" else ["AccXPath", "AccYPath", "AccZPath"]
    samples, identities, receipts = [], [], []
    names = None
    for position, (_, row) in enumerate(frame.iterrows(), 1):
        sequences, row_names, rates = [], [], []
        for axis, column in enumerate(columns):
            start = row["WAVcutstart"]
            duration = row["Duration"]
            rate, wave = _read(
                row[column],
                float(start),
                float(duration),
            )
            values, base = analysis_sequence(rate, wave, nfft, sensor, cfg)
            prefix = "mic" if sensor == "mic" else "xyz"[axis]
            sequences.append(values)
            rates.append(rate)
            row_names.extend(f"{prefix}_{name}" for name in base)
        if len(set(rates)) != 1 or len({len(v) for v in sequences}) != 1:
            raise ValueError(
                f"ACC axes must have matching rates/feature lengths: {row.ToolID}/{row.RecordID}"
            )
        if names is not None and names != row_names:
            raise ValueError("Feature schema/sample rate differs between recordings")
        names = row_names
        combined = np.concatenate(sequences, axis=1)
        resolved = resolve_sampling(cfg, rates[0], nfft)
        chunks, available = chunk_sequence(combined, resolved)
        samples.extend(chunks)
        for index in range(len(chunks)):
            identities.append(
                {
                    "ToolID": str(row.ToolID),
                    "RecordID": str(row.RecordID),
                    "UnitID": f"chunk_{index:06d}",
                    "Split": row.Split,
                    "WearValue": float(row.WearValue),
                    "Label": int(row.Label),
                }
            )
        receipts.append(
            {
                "ToolID": row.ToolID,
                "RecordID": row.RecordID,
                "Split": row.Split,
                "feature_frames": len(combined),
                "available_chunks": available,
                "retained_chunks": len(chunks),
                "dropped_full_chunks": available - len(chunks),
                "tail_frames": len(combined) % resolved["chunk_frames"],
                "source_sample_rate": rates[0],
                "sample_rate": resolved["sample_rate"],
                "chunk_frames": resolved["chunk_frames"],
                "hop_samples": resolved["hop_samples"],
                "effective_chunk_seconds": resolved["effective_chunk_seconds"],
            }
        )
        if cfg.get("verbose") and (position == 1 or position % 10 == 0 or position == len(frame)):
            print(
                f"FEATURE {sensor} NFFT={nfft} record={position}/{len(frame)} chunks={len(chunks)}",
                flush=True,
            )
    if not samples:
        raise ValueError("No analysis samples")
    matrix = np.asarray(samples, dtype=np.float32)
    identity = pd.DataFrame(identities)
    string = h5py.string_dtype("utf-8")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = {"all": np.arange(len(names))}
    if sensor == "mic":
        groups.update(mic_time=np.arange(7), mic_frequency=np.arange(7, len(names)))
    else:
        groups["acc"] = np.arange(len(names))
        groups["acc_time"] = np.array([i for i, name in enumerate(names) if "_freq_" not in name])
        groups["acc_frequency"] = np.array([i for i, name in enumerate(names) if "_freq_" in name])
    with h5py.File(output, "w") as h:
        h.attrs.update(
            sensor=sensor,
            nfft=nfft,
            layout="record,time,feature",
            sample_unit="chunk",
            sampling_json=json.dumps(resolved),
            feature_profile="analysis_short_time_v1",
            class_names_json=json.dumps(list(CLASS_NAMES), ensure_ascii=False),
        )
        for split in ("train", "val", "test"):
            mask = identity.Split.eq(split).to_numpy()
            for group, selected in groups.items():
                g = h.create_group(f"splits/{split}/{group}")
                if group == "all":
                    split_values = matrix[mask]
                    chunks = (
                        (min(io_config["write_batch_size"], len(split_values)), *matrix.shape[1:])
                        if len(split_values)
                        else None
                    )
                    g.create_dataset("values", data=split_values, chunks=chunks, **storage)
                else:
                    feature_view(g, h[f"splits/{split}/all/values"], output.name, selected)
                g.create_dataset(
                    "feature_names", data=np.array(names, dtype=object)[selected], dtype=string
                )
                for key, source in (
                    ("tool_ids", "ToolID"),
                    ("record_ids", "RecordID"),
                    ("unit_ids", "UnitID"),
                ):
                    g.create_dataset(
                        key, data=identity.loc[mask, source].to_numpy(dtype=object), dtype=string
                    )
                g["wear"] = identity.loc[mask, "WearValue"].to_numpy()
                g["labels"] = identity.loc[mask, "Label"].to_numpy()
    pd.DataFrame(receipts).to_csv(
        output.with_name("sample_counts.csv"), index=False, encoding="utf-8-sig"
    )
    identity.to_csv(output.with_name("sample_manifest.csv"), index=False, encoding="utf-8-sig")
    if independent:
        # Publish a split-independent store; labels are assigned only by Trainer.
        with h5py.File(output, "a") as h:
            h.move("splits/train/all", "samples")
            del h["splits"]
            del h["samples/labels"]
            h.attrs["schema_version"] = 1
            del h.attrs["class_names_json"]
        identity.drop(columns=["Split", "Label"]).to_csv(
            output.with_name("sample_manifest.csv"), index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(receipts).drop(columns=["Split"]).to_csv(
            output.with_name("sample_counts.csv"), index=False, encoding="utf-8-sig"
        )
    return output
