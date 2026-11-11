import json
from pathlib import Path

import pytest
import yaml

import wear_cnn.workflow as workflow
from wear_cnn.gui import validate_yaml_command


def test_modern_workflow_binds_outputs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "build_features", lambda config: tmp_path / "features")

    def train(config, features, resume):
        calls.append((Path(config).name, features, resume))
        run = tmp_path / "trained"
        run.mkdir()
        (run / "run_summary.json").write_text('{"failed": 0}')
        return run

    monkeypatch.setattr(workflow, "train_batch", train)
    monkeypatch.setattr(workflow, "render_training_report", lambda run: calls.append(run) or run)
    config = tmp_path / "workflow.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": "run",
                "steps": [
                    {"command": "build-features", "config": "features.yaml"},
                    {"command": "train-batch", "config": "train.yaml"},
                    {"command": "render-training-report"},
                ],
            }
        )
    )
    records = json.loads(workflow.run_workflow(config).read_text())
    assert calls == [("train.yaml", str(tmp_path / "features"), None), str(tmp_path / "trained")]
    assert [row["status"] for row in records] == ["complete"] * 3
    assert [row["index"] for row in records] == [1, 2, 3]


@pytest.mark.parametrize("command", ["prepare", "fuse", "run-sweep", "feature-analysis"])
def test_old_commands_rejected_before_output(tmp_path, command):
    config = tmp_path / "workflow.yaml"
    config.write_text(yaml.safe_dump({"output": "run", "steps": [{"command": command}]}))
    with pytest.raises(ValueError, match="Unsupported workflow command"):
        workflow.run_workflow(config)
    assert not (tmp_path / "run").exists()


def test_schema_mismatch(tmp_path):
    config = tmp_path / "features.yaml"
    config.write_text(yaml.safe_dump({"process": "turning", "input": {}, "output": {}}))
    valid, message = validate_yaml_command(config, "run-workflow")
    assert not valid
    assert "建立特徵" in message
    with pytest.raises(ValueError, match="steps"):
        workflow.run_workflow(config)


def test_training_failure_logged(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("bad input")

    monkeypatch.setattr(workflow, "train_batch", fail)
    config = tmp_path / "workflow.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": "run",
                "steps": [
                    {
                        "command": "train-batch",
                        "config": "train.yaml",
                        "features": "existing",
                        "resume": "resume",
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="bad input"):
        workflow.run_workflow(config)
    records = json.loads((tmp_path / "run/workflow_log.json").read_text())
    assert records[0]["status"] == "failed"
    assert records[0]["error"] == "ValueError: bad input"


def test_explicit_feature_and_resume_paths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "build_features", lambda config: tmp_path / "built")
    run = tmp_path / "trained"
    run.mkdir()
    (run / "run_summary.json").write_text('{"failed": 0}')
    monkeypatch.setattr(workflow, "train_batch", lambda *args: calls.append(args) or run)
    config = tmp_path / "workflow.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "output": "run",
                "steps": [
                    {"command": "build-features", "config": "features.yaml"},
                    {
                        "command": "train-batch",
                        "config": "train.yaml",
                        "features": "existing",
                        "resume": "resume",
                    },
                ],
            }
        )
    )
    workflow.run_workflow(config)
    assert calls == [
        (str(tmp_path / "train.yaml"), str(tmp_path / "existing"), str(tmp_path / "resume"))
    ]
