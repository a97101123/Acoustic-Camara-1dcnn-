"""Four-condition orchestration; Builder and Trainer remain independent."""

import argparse
import os
from datetime import datetime
from pathlib import Path

from wear_cnn.configuration import fields, validate_training_config
from wear_cnn.feature_builder import build_features, validate_feature_config
from wear_cnn.run_io import atomic_json, load_config, resolve
from wear_cnn.trainer_batch import train_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    source, cfg = load_config(args.config)
    fields(cfg, ("jobs",), "batch")
    if not isinstance(cfg["jobs"], list) or not cfg["jobs"]:
        raise ValueError("batch.jobs must be a nonempty list")
    jobs = []
    total_jobs = 0
    for index, job in enumerate(cfg["jobs"]):
        fields(job, ("features", "training"), f"batch.jobs[{index}]")
        feature_path = resolve(source.parent, job["features"])
        training_path = resolve(source.parent, job["training"])
        _, feature_cfg, frame, _, _ = validate_feature_config(feature_path)
        _, training_cfg = load_config(training_path)
        validate_training_config(training_cfg)
        conditions = (
            frame.Condition.unique().tolist()
            if training_cfg["conditions"] is None
            else [c for c in frame.Condition.unique() if c in training_cfg["conditions"]]
        )
        nffts = (
            feature_cfg["nffts"]
            if training_cfg["nffts"] is None
            else [n for n in feature_cfg["nffts"] if n in training_cfg["nffts"]]
        )
        total_jobs += len(conditions) * len(nffts) * len(training_cfg["selection"]["methods"])
        jobs.append((feature_cfg["process"], feature_path, training_path))
    if len({process for process, _, _ in jobs}) != len(jobs):
        raise ValueError("batch.jobs must use distinct process names")
    state_path = Path(args.state).resolve()
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started": datetime.now().isoformat(),
        "feature_sets": {},
        "training_runs": {},
        "total_jobs": total_jobs,
        "current": {},
    }

    def publish(stage, process, run=None):
        state["current"] = {
            "stage": stage,
            "process": process,
            "run": str(run) if run else None,
            "time": datetime.now().isoformat(),
        }
        if run:
            state["feature_sets" if stage == "FEATURES" else "training_runs"][process] = str(run)
        atomic_json(state_path, state)
        print(state["current"], flush=True)

    try:
        for process, feature_path, training_path in jobs:
            publish("FEATURES", process)
            features = build_features(
                feature_path,
                on_run=lambda run, p=process: publish("FEATURES", p, run),
            )
            state["feature_sets"][process] = str(features)
        failed = 0
        for process, feature_path, training_path in jobs:
            publish("TRAINING", process)
            run = train_batch(
                training_path,
                features=state["feature_sets"][process],
                on_run=lambda run, p=process: publish("TRAINING", p, run),
            )
            state["training_runs"][process] = str(run)
            import json

            summary = json.loads((run / "run_summary.json").read_text())
            failed += summary["failed"]
        state.update(
            status="complete_with_failures" if failed else "complete",
            failed=failed,
            completed=total_jobs - failed,
        )
        publish("FINISHED", "all")
        return 1 if failed else 0
    except BaseException as exc:
        state.update(status="interrupted", error=str(exc))
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
