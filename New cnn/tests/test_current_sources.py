import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def source_builder(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "current_sources", Path(__file__).parents[1] / "tools/build_current_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "data/h_drive"
    directory.mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    identities = {
        "ToolID": ["T01", "T01"],
        "RecordID": ["M001", "M002"],
        "WearValue": [0.1, 0.3],
        "WAVcutstart": [2.0, 4.0],
        "Duration": [25.0, 25.0],
    }
    acc = pd.DataFrame(identities)
    for axis in "XYZ":
        paths = []
        for i in (1, 2):
            filename = f"{axis}{i}.wav"
            (raw / filename).touch()
            paths.append(f"H:/turning/{filename}")
        acc[f"Acc{axis}Path"] = paths
    acc.iloc[::-1].to_csv(directory / "turning_acc.csv", index=False)
    for condition in ("turning_noAC", "turning_withAC"):
        frame = pd.DataFrame(identities)
        frame["FileName"] = [f"H:/turning/{condition}{i}.wav" for i in (1, 2)]
        for i in (1, 2):
            (raw / f"{condition}{i}.wav").touch()
        frame.to_csv(directory / f"{condition}.csv", index=False)
    return module, raw


def test_paired_sources_preserve_identity_wear_crops_and_existing_output(source_builder, tmp_path):
    module, raw = source_builder
    output = tmp_path / "current.csv"
    module.build_sources("turning", raw, output)
    frame = pd.read_csv(output)
    assert len(frame) == 4
    assert frame.WearValue.tolist() == [0.1, 0.3, 0.1, 0.3]
    assert frame.MICStart.tolist() == frame.ACCStart.tolist() == [2.0, 4.0, 2.0, 4.0]
    assert frame.AccXPath.map(lambda p: Path(p).name).tolist() == ["X1.wav", "X2.wav"] * 2
    assert set(frame.Condition) == {"turning_noac", "turning_withac"}
    assert len(json.loads(output.with_suffix(".provenance.json").read_text())["inputs"]) == 3
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.build_sources("turning", raw, output)
    assert output.read_bytes() == original


def test_source_wear_conflict_fails_before_publication(source_builder, tmp_path):
    module, raw = source_builder
    path = module.ROOT / "data/h_drive/turning_acc.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "WearValue"] = 9.0
    frame.to_csv(path, index=False)
    output = tmp_path / "current.csv"
    with pytest.raises(ValueError, match="wear differs"):
        module.build_sources("turning", raw, output)
    assert not output.exists()
