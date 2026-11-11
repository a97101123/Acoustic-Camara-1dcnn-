from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .feature_builder import build_features
from .run_io import load_config
from .trainer_batch import render_training_report, train_batch


def _path(base: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def run_workflow(config_path: str | Path) -> Path:
    """Run current feature, training and report steps in order."""
    source = Path(config_path).resolve()
    _, cfg = load_config(source)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("steps"), list) or not cfg["steps"]:
        raise ValueError("Workflow YAML requires a non-empty 'steps' list")
    allowed = {"build-features", "train-batch", "render-training-report"}
    has_training = False
    for step in cfg["steps"]:
        if not isinstance(step, dict) or step.get("command") not in allowed:
            raise ValueError(f"Unsupported workflow command: {step}")
        command = step["command"]
        if command == "render-training-report":
            if not step.get("run") and not has_training:
                raise ValueError("Report step requires 'run' or a preceding train-batch step")
        elif not step.get("config"):
            raise ValueError(f"{command} step requires 'config'")
        has_training |= command == "train-batch"
    base = source.parent
    output = Path(_path(base, cfg["output"]))
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "workflow_log.json"
    records: list[dict[str, Any]] = []
    features = None
    training_run = None
    for index, step in enumerate(cfg["steps"], 1):
        command = step["command"]
        record: dict[str, Any] = {
            "index": index,
            "name": step.get("name", command),
            "command": command,
            "status": "running",
        }
        records.append(record)
        log_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            if command == "build-features":
                result = build_features(_path(base, step["config"]))
                features = str(result)
            elif command == "train-batch":
                selected = _path(base, step["features"]) if step.get("features") else features
                resume = _path(base, step["resume"]) if step.get("resume") else None
                result = train_batch(_path(base, step["config"]), selected, resume)
                training_run = str(result)
                summary_path = Path(result) / "run_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary["failed"]:
                    record["result"] = training_run
                    raise RuntimeError(f"{summary['failed']} training jobs failed; results: {result}")
            else:
                run = _path(base, step["run"]) if step.get("run") else training_run
                result = render_training_report(run)
            record.update(status="complete", result=str(result))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            raise
        log_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path
