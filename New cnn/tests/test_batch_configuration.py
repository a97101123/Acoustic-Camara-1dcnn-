import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_batch_counts_actual_source_conditions_when_filter_is_null(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "tools" / "run_four_architecture.py"
    spec = importlib.util.spec_from_file_location("configured_batch", script)
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    batch_config = tmp_path / "batch.yaml"
    training = {"conditions": None, "nffts": None, "selection": {"methods": ["j", "r2"]}}
    feature = {"process": "turning", "conditions": None, "nffts": [128, 256]}
    source = pd.DataFrame({"Condition": ["a", "a", "b"]})
    snapshots = []

    def load(path):
        value = {"jobs": [{"features": "features.yaml", "training": "training.yaml"}]}
        return path, value if path == batch_config else training

    def stop(*args, **kwargs):
        raise RuntimeError("stop before extraction")

    monkeypatch.setattr(batch, "load_config", load)
    monkeypatch.setattr(batch, "validate_training_config", lambda cfg: None)
    monkeypatch.setattr(
        batch, "validate_feature_config", lambda path: (path, feature, source, {}, [128, 256])
    )
    monkeypatch.setattr(batch, "build_features", stop)
    monkeypatch.setattr(batch, "atomic_json", lambda path, state: snapshots.append(dict(state)))
    monkeypatch.setattr(
        batch.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(config=batch_config, state=tmp_path / "state.json"),
    )
    with pytest.raises(RuntimeError, match="before extraction"):
        batch.main()
    assert snapshots[0]["total_jobs"] == 8
