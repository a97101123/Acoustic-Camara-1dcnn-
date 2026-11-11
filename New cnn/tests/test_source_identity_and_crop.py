import numpy as np
import pytest
from scipy.io import wavfile

from wear_cnn.feature_builder import read_sources
from wear_cnn.features.extract import _read


def test_csv_preserves_leading_zero_ids_and_excludes_exact_tool(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"path validation only")
    csv = tmp_path / "sources.csv"
    csv.write_text(
        "ToolID,RecordID,FileName,Condition,WearValue,MICStart,ACCStart,Duration\n001,0001,audio.wav,c,1,0,0,1\n1,0001,audio.wav,c,2,0,0,1\n"
    )
    frame = read_sources(csv, tmp_path, sensors=["mic"], conditions=None, exclude_tools=[])
    assert frame.ToolID.tolist() == ["001", "1"]
    assert frame.RecordID.tolist() == ["0001", "0001"]
    filtered = read_sources(csv, tmp_path, sensors=["mic"], exclude_tools=["001"], conditions=None)
    assert filtered.ToolID.tolist() == ["1"]


@pytest.mark.parametrize("start,duration", [(0, 5), (0.5, 1), (1, None)])
def test_audio_window_cannot_silently_truncate(tmp_path, start, duration):
    path = tmp_path / "one_second.wav"
    wavfile.write(path, 16000, np.arange(16000, dtype=np.int16))
    with pytest.raises(ValueError, match="exceeds available"):
        _read(path, start, duration)


@pytest.mark.parametrize(
    "start,duration", [(-1, 1), (float("nan"), 1), (0, 0), (0, -1), (0, float("inf"))]
)
def test_audio_window_rejects_invalid_coordinates(tmp_path, start, duration):
    with pytest.raises(ValueError, match="must be finite"):
        _read(tmp_path / "not_opened.wav", start, duration)


def test_valid_audio_window_preserves_existing_samples(tmp_path):
    path = tmp_path / "one_second.wav"
    samples = np.arange(16000, dtype=np.int16)
    wavfile.write(path, 16000, samples)
    rate, values = _read(path, 0.25, 0.5)
    expected = samples[4000:12000].astype(np.float64)
    assert rate == 16000
    np.testing.assert_array_equal(values, expected - expected.mean())
