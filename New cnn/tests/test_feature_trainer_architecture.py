import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import settings as explicit
import yaml
from scipy.io import wavfile

from wear_cnn.domain_selection import select_domains
from wear_cnn.feature_builder import build_features, feature_catalog
from wear_cnn.run_io import sha256
from wear_cnn.trainer_batch import render_training_report, train_batch, training_seeds


def test_axis_and_mic_count_policies():
    names = [
        "acc::x_rms",
        "acc::x_std",
        "acc::y_rms",
        "acc::z_rms",
        "mic::mic_freq_1Hz",
        "mic::mic_freq_2Hz",
        "mic::mic_rms",
    ]
    catalog = feature_catalog(names)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7])
    options = {
        "acc_axis_top_k": {"x": 1, "y": 1, "z": 1},
        "threshold": 0.5,
        "mic_count_policy": "match_acc_total",
    }
    selected, _ = select_domains(
        scores,
        catalog,
        "r2",
        explicit.merge(explicit.training_config()["selection"]["r2"], options),
    )
    assert selected.tolist() == [4, 5, 6]
    scores[6] = 0.5
    assert select_domains(
        scores,
        catalog,
        "r2",
        explicit.merge(explicit.training_config()["selection"]["r2"], options),
    )[0].tolist() == [4, 5]
    options["mic_count_policy"] = "all_above_threshold"
    assert select_domains(
        scores,
        catalog,
        "r2",
        explicit.merge(explicit.training_config()["selection"]["r2"], options),
    )[0].tolist() == [4, 5]
    scores[:4] = [0.5, 0.8, 0.7, 0.6]
    assert select_domains(
        scores, catalog, "r2", explicit.merge(explicit.training_config()["selection"]["r2"], {})
    )[0].tolist() == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="equal ACC"):
        select_domains(
            scores,
            catalog,
            "j",
            explicit.merge(
                explicit.training_config()["selection"]["j"],
                {
                    "acc_axis_quota": {"x": 1, "y": 1, "z": 1},
                    "mic_domain_quota": {"mic_frequency": 1, "mic_time": 1},
                },
            ),
        )


def test_seed_range():
    assert training_seeds({"seed_start": 20, "runs": 3}) == [20, 21, 22]
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            training_seeds({"seed_start": 42, "runs": value})


def test_status_write_retries_transient_windows_lock(tmp_path, monkeypatch):
    from wear_cnn.run_io import atomic_json

    original = Path.replace
    attempts = []

    def locked_once(self, target):
        attempts.append(str(self))
        if len(attempts) == 1:
            raise PermissionError("Windows reader lock")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", locked_once)
    atomic_json(tmp_path / "state.json", {"status": "complete"})
    assert len(attempts) == 2
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "complete"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("custom", [False, True])
def test_builder_and_trainer_without_source_wavs(tmp_path, custom):
    import torch

    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    rate = 16000
    t = np.arange(rate * 2) / rate
    for tool in range(6):
        for record, wear in enumerate([10.0, 50.0, 90.0]):
            paths = []
            for channel in range(4):
                path = raw / f"{tool}_{record}_{channel}.wav"
                wavfile.write(
                    path,
                    rate,
                    (10000 * np.sin(2 * np.pi * (400 + wear * 3 + channel * 100) * t)).astype(
                        "int16"
                    ),
                )
                paths.append(str(path))
            rows.append(
                dict(
                    ToolID=f"T{tool}",
                    RecordID=f"R{record}",
                    Condition="turning_noac",
                    WearValue=wear,
                    FileName=paths[0],
                    AccXPath=paths[1],
                    AccYPath=paths[2],
                    AccZPath=paths[3],
                    Enabled=True,
                    MICStart=0,
                    ACCStart=0,
                    Duration=2,
                )
            )
    excel = tmp_path / "sources.xlsx"
    pd.DataFrame(rows).to_excel(excel, index=False)
    config = tmp_path / "features.yaml"
    config.write_text(
        yaml.safe_dump(
            explicit.feature_config(
                {
                    "process": "turning",
                    "conditions": ["turning_noac"],
                    "input": {"excel": str(excel), "data_root": str(raw)},
                    "output": {"root": str(tmp_path / "features"), "name": "test"},
                    "nffts": [128],
                    "sampling": {"target_sample_rate": None},
                }
            )
        )
    )
    if custom:
        cfg = yaml.safe_load(config.read_text())
        cfg["io"] = explicit.io({"compression": "lzf", "write_batch_size": 3, "read_batch_size": 2})
        cfg["sampling"]["time_backend"] = "matrix"
        cfg["output"]["filenames"] = {
            "features": "{process}_{condition}_{nfft}.h5",
            "catalog": "catalog_{nfft}.csv",
            "sample_manifest": "{sensor}_samples.csv",
        }
        config.write_text(yaml.safe_dump(cfg))
    feature_set = build_features(config)
    spec = json.loads((feature_set / "feature_set.json").read_text())
    store = feature_set / spec["entries"][0]["path"]
    before = sha256(store)
    with h5py.File(store) as h:
        assert h["samples/values"].compression == ("lzf" if custom else "gzip")
        values = h["samples/values"]
        assert values.chunks[1:] == values.shape[1:]
        from wear_cnn.features.analysis_sampling import (
            analysis_sequence,
            chunk_sequence,
            resolve_sampling,
        )
        from wear_cnn.features.extract import _read

        source_rate, source_wave = _read(rows[0]["FileName"], 0, 2)
        sampling = explicit.sampling({"target_sample_rate": None})
        sequence, names = analysis_sequence(source_rate, source_wave, 128, "mic", sampling)
        chunks, _ = chunk_sequence(sequence, resolve_sampling(sampling, source_rate, 128))
        np.testing.assert_array_equal(values[0, :, : len(names)], chunks[0])
        assert "splits" not in h and "labels" not in h["samples"]
        assert "normal_moderate_threshold" not in h.attrs
    raw.rename(tmp_path / "offline")
    train_config = tmp_path / "training.yaml"
    train_config.write_text(
        yaml.safe_dump(
            explicit.training_config(
                {
                    "features": str(feature_set),
                    "output": {"root": str(tmp_path / "results"), "name": "test"},
                    "device": "cpu",
                    "cpu_threads": 1,
                    "training": {"seed_start": 42, "runs": 2, "epochs": 1},
                    "selection": {
                        "methods": ["j"],
                        "j": {
                            "acc_axis_quota": {"x": 1, "y": 1, "z": 1},
                            "mic_domain_quota": {"mic_frequency": 2, "mic_time": 1},
                        },
                    },
                }
            )
        )
    )
    if custom:
        cfg = yaml.safe_load(train_config.read_text())
        cfg["cache_features"] = True
        cfg["io"] = explicit.io({"read_batch_size": 2})
        cfg["training"].update(
            batch_size=4, validation_batch_size=8, checkpoint_every=3, preload_device=True
        )
        cfg["evaluation"] = explicit.evaluation({"batch_size": 5})
        cfg["output"]["filenames"] = {
            "model": "{condition}_{method}_seed{seed}.pt",
            "history": "history_{seed}.csv",
            "selection": "chosen_{method}.json",
            "scores": "scores_{method}.csv",
            "top": "top_{method}.csv",
            "curves": "curves_{seed}.png",
        }
        cfg["output"]["evaluation_filenames"] = {
            "metrics": "metrics_{seed}.json",
            "predictions": "predictions_{seed}.csv",
            "confusion": "matrix_{seed}.csv",
            "confusion_plot": "matrix_{seed}.png",
        }
        cfg["output"]["report_filenames"] = {
            "summary": "{process}_summary.csv",
            "readme": "REPORT.md",
        }
        train_config.write_text(yaml.safe_dump(cfg))
    previous = torch.get_num_threads()
    try:
        run = train_batch(train_config)
        state = json.loads((run / "state.json").read_text())
        assert state["status"] == "complete", state
        job = next(run.glob("03_details/*/nfft*/j"))
        registry = json.loads((job / "artifacts.json").read_text())
        assert all((job / s["directory"] / s["files"]["model"]).exists() for s in registry["seeds"])
        selection_dir = job / registry.get("selection_directory", ".")
        top = pd.read_csv(selection_dir / registry["files"]["top"])
        assert top.sensor.value_counts().to_dict() == {"acc": 3, "mic": 3}
        checkpoint = (
            job / registry["seeds"][0]["directory"] / registry["seeds"][0]["files"]["model"]
        )
        timestamp = checkpoint.stat().st_mtime_ns
        assert train_batch(train_config, resume=run) == run
        assert checkpoint.stat().st_mtime_ns == timestamp
        render_training_report(run)
        assert list((job / "figures").glob("*_selected_top.png"))
        assert (job / "seed_42" / registry["seeds"][0]["files"]["curves"]).exists()
        assert (job / "ensemble" / registry["ensemble"]["predictions"]).exists()
        if custom:
            assert (run / "01_summary/turning_summary.csv").exists()
            assert (run / "REPORT.md").exists()
        with h5py.File(job.parent / "split.h5") as h:
            tools = [set(h[f"splits/{s}/all/tool_ids"][:]) for s in ["train", "val", "test"]]
            assert not tools[0] & tools[1] and not tools[0] & tools[2] and not tools[1] & tools[2]
        assert sha256(store) == before
        assert (run / "output_manifest.csv").is_file()
        spec["entries"][0]["sha256"] = "wrong"
        (feature_set / "feature_set.json").write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="Invalid feature artifact"):
            train_batch(train_config)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "templates",
    [
        {"model": "../model.pt"},
        {"model": "CON.pt"},
        {"model": "{unknown}.pt"},
        {"scores": "same.csv", "top": "SAME.csv"},
        {"model": "model.csv"},
        {"unknown_key": "x.csv"},
        {"r2_scatter": "same.png"},
    ],
)
def test_invalid_filename_templates(templates):
    from wear_cnn.filename_templates import TRAIN_FILES, resolve_names

    with pytest.raises(ValueError):
        resolve_names(TRAIN_FILES, templates, {"feature_index": 0})
