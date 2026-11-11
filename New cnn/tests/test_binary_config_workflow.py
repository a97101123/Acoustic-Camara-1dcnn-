import json

import numpy as np
import pandas as pd
import settings as explicit
import yaml
from scipy.io import wavfile

from wear_cnn.artifact_validation import valid_checkpoint
from wear_cnn.feature_builder import build_features
from wear_cnn.inference import predict_wav
from wear_cnn.trainer_batch import render_training_report, train_batch


def test_binary_config_trains_reports_resumes_and_predicts_wav(tmp_path):
    import torch

    rows = []
    rate = 4000
    time = np.arange(int(rate * 1.2)) / rate
    for tool in range(1, 10):
        for stage in (1, 7, 8, 13):
            path = tmp_path / f"T{tool:02}_M{stage:03}.wav"
            signal = (8000 * np.sin(2 * np.pi * (100 + stage * 12 + tool) * time)).astype(np.int16)
            wavfile.write(path, rate, signal)
            rows.append(
                {
                    "ToolID": f"T{tool:02}",
                    "RecordID": f"M{stage:03}",
                    "Condition": "turning_noac",
                    "WearValue": stage * 0.01,
                    "FileName": str(path),
                    "MICStart": 0,
                    "Duration": 1.2,
                }
            )
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    feature_config = tmp_path / "features.yaml"
    feature_config.write_text(
        yaml.safe_dump(
            explicit.feature_config(
                {
                    "process": "turning",
                    "sensors": ["mic"],
                    "nffts": [128],
                    "input": {"excel": str(source), "data_root": str(tmp_path)},
                    "output": {"root": str(tmp_path / "features"), "name": "binary"},
                    "sampling": {"target_sample_rate": None},
                }
            )
        ),
        encoding="utf-8",
    )
    feature_set = build_features(feature_config)
    config = explicit.training_config(
        {
            "features": str(feature_set),
            "device": "cpu",
            "cpu_threads": 1,
            "output": {"root": str(tmp_path / "results"), "name": "binary"},
            "dataset": {
                "stage_ids": list(range(1, 14)),
                "exclude_stages": [7],
                "labeling": {
                    "method": "stage_groups",
                    "groups": [list(range(1, 7)), list(range(8, 14))],
                },
                "class_names": ["Low", "High"],
            },
            "evaluation": {
                "strategy": "fixed_knife",
                "test_knife": 9,
                "val_knives": [7, 8],
                "prediction_rule": "argmax",
            },
            "training": {"runs": 1, "epochs": 1, "severe_class_index": 1},
            "selection": {
                "methods": ["j"],
                "j": {
                    "acc_axis_quota": {"x": 0, "y": 0, "z": 0},
                    "mic_domain_quota": {"mic_frequency": 2, "mic_time": 1},
                },
            },
        }
    )
    training_config = tmp_path / "training.yaml"
    training_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    previous = torch.get_num_threads()
    try:
        run = train_batch(training_config)
        state = json.loads((run / "state.json").read_text())
        assert state["status"] == "complete", state
        job = next(run.glob("03_details/*/nfft*/j"))
        registry = json.loads((job / "artifacts.json").read_text())
        member = registry["seeds"][0]
        checkpoint = job / member["directory"] / member["files"]["model"]
        assert valid_checkpoint(checkpoint)
        saved = torch.load(checkpoint, weights_only=False)
        assert saved["state_dict"]["head.weight"].shape[0] == 2
        evaluation = job / "ensemble"
        metrics = json.loads((evaluation / registry["ensemble"]["metrics"]).read_text())
        assert metrics["class_names"] == ["Low", "High"]
        assert metrics["severe_class_index"] == 1
        assert metrics["severe_threshold"] is None
        assert set(metrics["classes"]) >= {"class_0", "severe"}
        matrix = np.loadtxt(evaluation / registry["ensemble"]["confusion"], delimiter=",")
        assert matrix.shape == (2, 2)
        predictions = pd.read_csv(evaluation / registry["ensemble"]["predictions"])
        assert {"p_class_0", "p_severe"} <= set(predictions)
        assert "p_normal" not in predictions
        assert set(predictions.ToolID) == {"T09"}
        assert set(predictions.RecordID) == {"M001", "M008", "M013"}
        deployment = evaluation / registry["ensemble"]["deployment"]
        prediction = predict_wav([rows[-1]["FileName"]], str(deployment), "cpu")
        assert prediction["class_name"] in {"Low", "High"}
        assert len(prediction["probabilities"]) == 2
        assert prediction["class_id"] == int(np.argmax(prediction["probabilities"]))
        deployment_spec = json.loads(deployment.read_text())
        deployment_spec["store_metadata"]["labeling_json"] = json.dumps({"groups": [[1], [13]]})
        altered_deployment = tmp_path / "altered_deployment.json"
        altered_deployment.write_text(json.dumps(deployment_spec), encoding="utf-8")
        import pytest

        with pytest.raises(ValueError, match="labeling_json"):
            predict_wav([rows[-1]["FileName"]], str(altered_deployment), "cpu")
        render_training_report(run)
        assert (evaluation / registry["ensemble"]["confusion_plot"]).is_file()
        mtime = checkpoint.stat().st_mtime_ns
        assert train_batch(training_config, resume=run) == run
        assert checkpoint.stat().st_mtime_ns == mtime
        config["dataset"]["class_names"] = ["Early", "Late"]
        training_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="Resume config"):
            train_batch(training_config, resume=run)
    finally:
        torch.set_num_threads(previous)
