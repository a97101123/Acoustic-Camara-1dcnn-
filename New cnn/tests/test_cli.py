"""Compatibility checks for the supported command-line interface."""

import json
import subprocess
import sys

import numpy as np
import pytest
import settings as explicit

from wear_cnn.classification import classify_probabilities
from wear_cnn.cli import build_parser, run_command


@pytest.mark.parametrize(
    "command",
    [
        "prepare",
        "fuse",
        "plan-sweep",
        "run-sweep",
        "feature-analysis",
        "run-candidates",
        "run-condition",
        "run-fusion-grid",
        "run-analysis-fusion",
        "train",
        "evaluate",
        "cross-validate",
        "build-turning-manifests",
        "build-h-manifests",
        "build-h-report",
    ],
)
def test_removed_workflows_are_rejected(command):
    result = subprocess.run(
        [sys.executable, "-m", "wear_cnn.cli", command, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


@pytest.mark.parametrize(
    "command,module,function,arguments,expected",
    [
        (
            "build-features",
            "feature_builder",
            "build_features",
            ["--config", "features.yaml"],
            ("features.yaml",),
        ),
        (
            "train-batch",
            "trainer_batch",
            "train_batch",
            ["--config", "train.yaml", "--features", "features", "--resume", "run"],
            ("train.yaml", "features", "run"),
        ),
    ],
)
def test_modern_workflow_dispatch(command, module, function, arguments, expected, monkeypatch):
    import importlib

    calls = []

    def execute(*args):
        calls.append(args)
        return "result-path"

    monkeypatch.setattr(importlib.import_module(f"wear_cnn.{module}"), function, execute)
    assert run_command(build_parser().parse_args([command, *arguments])) == "result-path"
    assert calls == [expected]


def test_main_help_identifies_current_workflow():
    help_text = build_parser().format_help()
    assert "Current workflow: build-features then train-batch" in help_text
    assert "Legacy" not in help_text


def test_help_does_not_load_training_and_plotting_modules():
    script = """
import sys
from wear_cnn.cli import build_parser
build_parser().format_help()
assert 'torch' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert 'wear_cnn.training' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_fusion_prediction_arguments_use_explicit_device():
    parser = build_parser()
    fusion = parser.parse_args(
        [
            "predict-fusion",
            "--deployment",
            "deployment.json",
            "--mic-wav",
            "mic.wav",
            "--acc-wav",
            "x.wav",
            "y.wav",
            "z.wav",
            "--device",
            "auto",
        ]
    )
    assert fusion.acc_wav == ["x.wav", "y.wav", "z.wav"]
    assert fusion.device == "auto"


def test_prediction_command_still_returns_the_same_json(monkeypatch):
    from wear_cnn import inference

    expected = {"class_id": 2, "class_name": "嚴重", "probabilities": [0.1, 0.2, 0.7]}
    calls = []

    def predict(wavs, deployment, device):
        calls.append((wavs, deployment, device))
        return expected

    monkeypatch.setattr(inference, "predict_wav", predict)
    args = build_parser().parse_args(
        ["predict", "--deployment", "d.json", "--wav", "m.wav", "--device", "auto"]
    )
    assert json.loads(run_command(args)) == expected
    assert calls == [(["m.wav"], "d.json", "auto")]


def test_shared_classification_rule_preserves_threshold_and_ties():
    values = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.7, 0.2]])
    assert classify_probabilities(values, 0.5, explicit.classification()).tolist() == [2, 0, 1]
    assert int(classify_probabilities(values[0], 0.5, explicit.classification())) == 2
