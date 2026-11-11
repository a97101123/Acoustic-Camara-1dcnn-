import json

import h5py
import numpy as np
import pandas as pd
import pytest
import settings as explicit
import yaml
from scipy.io import wavfile

from wear_cnn.domain_selection import select_domains
from wear_cnn.feature_builder import build_features, feature_catalog, validate_feature_store
from wear_cnn.inference import predict_wav
from wear_cnn.trainer_batch import train_batch
from wear_cnn.training import train_hdf5


def test_mic_only_ignores_acc_and_excludes_whole_tool(tmp_path):
    import torch

    torch.set_num_threads(1)
    rows = []
    for tool in range(7):
        for record, wear in enumerate([10.0, 50.0, 90.0]):
            path = tmp_path / f"{tool}_{record}.wav"
            if tool != 6:
                t = np.arange(32000) / 16000
                wavfile.write(
                    path, 16000, (10000 * np.sin(2 * np.pi * (300 + wear) * t)).astype("int16")
                )
            else:
                path.write_bytes(bytes(100))
            rows.append(
                dict(
                    ToolID=f"T{tool}",
                    RecordID=f"R{record}",
                    WearValue=wear,
                    Condition="mic_test",
                    FileName=str(path),
                    MICStart=0,
                    ACCStart=0,
                    Duration=2,
                )
            )
    source = tmp_path / "sources.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    config = tmp_path / "features.yaml"
    config.write_text(
        yaml.safe_dump(
            explicit.feature_config(
                {
                    "process": "turning",
                    "conditions": ["mic_test"],
                    "sensors": ["mic"],
                    "exclude_tools": ["T6"],
                    "input": {"excel": str(source), "data_root": str(tmp_path)},
                    "nffts": [128],
                    "output": {"root": str(tmp_path / "features"), "name": "mic"},
                    "sampling": {"target_sample_rate": None},
                }
            )
        )
    )
    features = build_features(config)
    entry = json.loads((features / "feature_set.json").read_text())["entries"][0]
    store = features / entry["path"]
    with h5py.File(store) as h:
        assert h.attrs["sensor"] == "mic"
        assert not any(v == "T6" for v in h["samples/tool_ids"].asstr()[:])
    provenance = json.loads((features / "source_manifest.json").read_text())
    assert len(provenance["files"]) == 18
    train = tmp_path / "train.yaml"
    train.write_text(
        yaml.safe_dump(
            explicit.training_config(
                {
                    "features": str(features),
                    "device": "cpu",
                    "cpu_threads": 1,
                    "output": {"root": str(tmp_path / "results"), "name": "mic"},
                    "training": {"epochs": 1, "runs": 1},
                    "selection": {
                        "methods": ["j"],
                        "j": {
                            "acc_axis_quota": {"x": 0, "y": 0, "z": 0},
                            "mic_domain_quota": {"mic_time": 2, "mic_frequency": 2},
                        },
                    },
                }
            )
        )
    )
    run = train_batch(train)
    assert json.loads((run / "state.json").read_text())["status"] == "complete"
    report = next(run.glob("03_details/*/nfft128/j/ensemble/deployment.json"))
    result = predict_wav([str(tmp_path / "0_0.wav")], str(report), "cpu")
    assert result["chunk_count"] > 0
    with h5py.File(store, "a") as h:
        h["samples/values"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_feature_store(store)


def test_mic_r2_never_fills_below_threshold():
    catalog = feature_catalog(["mic::mic_rms", "mic::mic_freq_1Hz"])
    selected, _ = select_domains(
        [0.5, 0.8], catalog, "r2", explicit.merge(explicit.training_config()["selection"]["r2"], {})
    )
    assert selected.tolist() == [1]
    with pytest.raises(ValueError, match="No features pass"):
        select_domains(
            [0.5, 0.1],
            catalog,
            "r2",
            explicit.merge(explicit.training_config()["selection"]["r2"], {}),
        )
    selected, _ = select_domains(
        [0.9, 0.8], catalog, "r2", explicit.merge(explicit.training_config()["selection"]["r2"], {})
    )
    assert selected.tolist() == [0, 1]


def test_epoch_resume_matches_uninterrupted_and_rejects_changed_data(tmp_path):
    import torch

    torch.set_num_threads(1)
    data = tmp_path / "data.h5"
    rng = np.random.default_rng(4)
    with h5py.File(data, "w") as h:
        h.attrs.update(
            feature_set_identity="test",
            sample_unit="chunk",
            sampling_json='{"chunk_retention":"all_complete"}',
            **explicit.classification().to_metadata(),
        )
        for split in ["train", "val"]:
            g = h.create_group(f"splits/{split}/all")
            g["values"] = rng.normal(size=(12, 8, 3)).astype("float32")
            g["labels"] = np.tile([0, 1, 2], 4)
            g["feature_names"] = np.array(["a", "b", "c"], dtype=h5py.string_dtype())
            g["tool_ids"] = np.array([split] * 12, dtype=h5py.string_dtype())
            g["record_ids"] = np.array([str(i) for i in range(12)], dtype=h5py.string_dtype())
            g["wear"] = np.arange(12)
            g["unit_ids"] = np.array(["chunk_000000"] * 12, dtype=h5py.string_dtype())
    uninterrupted = tmp_path / "full.pt"
    resumed = tmp_path / "resume.pt"
    train_hdf5(
        data,
        uninterrupted,
        epochs=2,
        device="cpu",
        make_plots=False,
        seed=42,
        prepared=explicit.prepared(data),
        options=explicit.training(),
    )

    def interrupt(_):
        raise RuntimeError("simulated stop")

    with pytest.raises(RuntimeError, match="simulated"):
        train_hdf5(
            data,
            resumed,
            epochs=2,
            device="cpu",
            make_plots=False,
            progress=interrupt,
            seed=42,
            prepared=explicit.prepared(data),
            options=explicit.training(),
        )
    train_hdf5(
        data,
        resumed,
        epochs=2,
        device="cpu",
        make_plots=False,
        seed=42,
        prepared=explicit.prepared(data),
        options=explicit.training(),
    )
    a = torch.load(uninterrupted, weights_only=False)
    b = torch.load(resumed, weights_only=False)
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    with h5py.File(data, "a") as h:
        h["splits/train/all/values"][0, 0, 0] += 5
    with pytest.raises(ValueError, match="identity differs"):
        train_hdf5(
            data,
            resumed,
            epochs=3,
            device="cpu",
            make_plots=False,
            seed=42,
            prepared=explicit.prepared(data),
            options=explicit.training(),
        )
