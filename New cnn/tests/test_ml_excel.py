from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.io import wavfile

from wear_cnn.feature_builder import read_sources, validate_feature_config
from wear_cnn.run_io import load_config
from wear_cnn.source_identity import recording_identity


@pytest.mark.parametrize(
    "path,expected",
    [
        (r"D:\data\knife3\flank_12.wav", (3, 12)),
        ("tool02/audio/FLANK003.wav", (2, 3)),
        ("T09/flank-13.wav", (9, 13)),
        ("4/flank1.wav", (4, 1)),
        ("knife1/st01_cut.wav", (1, 1)),
        ("knife3/st12/accx.wav", (3, 12)),
    ],
)
def test_path_identity(path, expected):
    assert recording_identity(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "knife1/audio.wav",
        "audio/flank1.wav",
        "knife0/flank1.wav",
        "knife1/flank0.wav",
        "knife1/knife2/flank1.wav",
        None,
    ],
)
def test_ambiguous_or_missing_identity_fails(path):
    with pytest.raises(ValueError, match="FileName"):
        recording_identity(path)


def test_three_column_excel_through_feature_config(tmp_path):
    audio = tmp_path / "knife3/flank_12.wav"
    audio.parent.mkdir()
    wavfile.write(audio, 48000, np.zeros(48000 * 27, dtype=np.int16))
    workbook = tmp_path / "sources.xlsx"
    pd.DataFrame([dict(FileName="knife3/flank_12.wav", WearValue=12.5, WAVcutstart=1.25)]).to_excel(
        workbook, index=False
    )
    root = Path(__file__).resolve().parents[1]
    _, config = load_config(root / "configs/experiment_excel.yaml")
    config["input"].update(excel=str(workbook), data_root=str(tmp_path))
    launch = tmp_path / "features.yaml"
    launch.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, _, frame, _, _ = validate_feature_config(launch)
    assert frame.ToolID.tolist() == ["T03"]
    assert frame.RecordID.tolist() == ["M012"]
    assert frame.MICStart.tolist() == [1.25]
    assert frame.WearValue.tolist() == [12.5]
    assert frame.Duration.tolist() == [25.0]
    assert frame.FileName.tolist() == [str(audio)]


def test_three_columns_cannot_invent_acc_paths(tmp_path):
    path = tmp_path / "sources.xlsx"
    pd.DataFrame([dict(FileName="knife1/flank1.wav", WearValue=1, WAVcutstart=0)]).to_excel(
        path, index=False
    )
    with pytest.raises(ValueError, match="ACC requires paired paths"):
        read_sources(path, tmp_path, ["turning_noac"], ["mic", "acc"], [], duration=25)
