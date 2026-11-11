from pathlib import Path

import pandas as pd
import pytest
import yaml

from wear_cnn.feature_builder import read_sources, validate_feature_config
from wear_cnn.run_io import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_nested_overrides_preserve_other_parameters_and_base(tmp_path):
    base = ROOT / "configs/turning_training.yaml"
    _, before = load_config(base)
    launch = tmp_path / "experiment.yaml"
    launch.write_text(
        yaml.safe_dump(
            {
                "includes": [str(base)],
                "overrides": {
                    "training": {"epochs": 2},
                    "nffts": [128],
                },
            }
        ),
        encoding="utf-8",
    )
    _, result = load_config(launch)
    assert result["training"] == {**before["training"], "epochs": 2}
    assert result["nffts"] == [128]
    assert load_config(base)[1] == before


@pytest.mark.parametrize("changes", [{"training": {"epohs": 2}}, {"training": 2}, None])
def test_invalid_overrides_fail(tmp_path, changes):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {"includes": [str(ROOT / "configs/turning_training.yaml")], "overrides": changes}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="override"):
        load_config(path)


def test_excel_reads_selected_sheet_and_preserves_text_ids(tmp_path):
    (tmp_path / "audio.wav").touch()
    path = tmp_path / "sources.xlsx"
    rows = pd.DataFrame(
        [
            dict(
                ToolID="001",
                RecordID="0001",
                FileName="audio.wav",
                Condition="turning_noac",
                WearValue=12,
                MICStart=0,
                Duration=1,
            )
        ]
    )
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"notes": ["not source data"]}).to_excel(
            writer, sheet_name="notes", index=False
        )
        rows.to_excel(writer, sheet_name="recordings", index=False)
    result = read_sources(path, tmp_path, None, ["mic"], [], sheet="recordings")
    assert result.ToolID.tolist() == ["001"]
    assert result.RecordID.tolist() == ["0001"]
    assert result.WearValue.tolist() == [12]
    with pytest.raises(ValueError, match="Missing source columns"):
        read_sources(path, tmp_path, None, ["mic"], [])


@pytest.mark.parametrize("sheet", [None, True, -1, [], ""])
def test_invalid_sheet_fails_before_reading_sources(tmp_path, sheet):
    _, config = load_config(ROOT / "configs/turning_features.yaml")
    config["input"]["sheet"] = sheet
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="input.sheet"):
        validate_feature_config(path)
