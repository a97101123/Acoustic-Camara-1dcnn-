import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.io import wavfile

from wear_cnn.feature_builder import build_features, validate_feature_config
from wear_cnn.run_io import load_config, sha256
from wear_cnn.source_identity import read_paired_excel


@pytest.fixture
def paired_sources(tmp_path):
    sources = {}
    for sensor in ("mic", "acc_x", "acc_y", "acc_z"):
        rows = []
        for stage in (1, 9):
            audio = tmp_path / "knife3" / f"st{stage:02d}" / f"{sensor}.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio, 48000, np.zeros(48000 * 2, dtype=np.int16))
            rows.append(
                dict(
                    FileName=str(audio),
                    WearValue=stage * 10,
                    WAVcutstart=0.1 if sensor == "mic" else 0.2,
                )
            )
        source = tmp_path / f"{sensor}.xlsx"
        pd.DataFrame(rows[::-1] if sensor == "acc_y" else rows).to_excel(source, index=False)
        sources[sensor] = source
    return sources


def test_pair_by_identity_not_excel_row_order(paired_sources):
    result = read_paired_excel(paired_sources)
    assert result.knife.tolist() == [3, 3]
    assert result.stage.tolist() == [1, 9]
    assert result.wear_um.tolist() == [10, 90]
    assert result.mic_start_sec.tolist() == [0.1, 0.1]
    assert result.acc_start_sec.tolist() == [0.2, 0.2]
    assert "st01" in result.acc_y.iloc[0]


def test_one_audio_file_cannot_be_reused_for_different_axes(paired_sources):
    paired_sources["acc_z"] = paired_sources["acc_x"]
    with pytest.raises(ValueError, match="different sensors"):
        read_paired_excel(paired_sources)


def test_paired_excel_builds_feature_set_and_hashes_every_workbook(paired_sources, tmp_path):
    _, cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/experiment_paired_excel.yaml"
    )
    cfg["input"].update(
        excel={key: str(value) for key, value in paired_sources.items()},
        data_root=str(tmp_path),
        duration_seconds=1.0,
    )
    cfg["output"].update(root=str(tmp_path / "output"), name="paired")
    cfg["nffts"] = [128]
    config = tmp_path / "features.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _, _, frame, _, _ = validate_feature_config(config)
    assert frame.ToolID.tolist() == ["T03", "T03"]
    assert frame.RecordID.tolist() == ["M001", "M009"]
    assert frame.ACCStart.tolist() == [0.2, 0.2]
    run = build_features(config)
    receipt = json.loads((Path(run) / "source_manifest.json").read_text())
    assert receipt["source_excel_sha256"] == {
        key: sha256(value) for key, value in paired_sources.items()
    }


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", "pairing is incomplete"),
        ("duplicate", "duplicate tool/stage"),
        ("wear", "WearValue differs"),
        ("start", "acc_start_sec values differ"),
        ("nan", "finite and nonnegative"),
    ],
)
def test_invalid_pairs_fail(paired_sources, change, error):
    source = paired_sources["acc_z"]
    rows = pd.read_excel(source)
    if change == "missing":
        rows = rows.iloc[:1]
    elif change == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    elif change == "wear":
        rows.loc[0, "WearValue"] = 999
    elif change == "start":
        rows.loc[0, "WAVcutstart"] = 1
    else:
        rows.loc[0, "WAVcutstart"] = float("nan")
    rows.to_excel(source, index=False)
    with pytest.raises(ValueError, match=error):
        read_paired_excel(paired_sources)
