import numpy as np
import pytest
from scipy.io import wavfile

from wear_cnn.features.extract import _read


@pytest.mark.parametrize("stereo", [False, True])
def test_mapped_crop_matches_original_conversion(tmp_path, stereo):
    samples = np.random.default_rng(42).integers(
        -30000, 30000, (8000, 2) if stereo else 8000, dtype=np.int16
    )
    path = tmp_path / "input.wav"
    wavfile.write(path, 8000, samples)
    original = samples.astype(np.float64)
    if stereo:
        original = original.mean(axis=1)
    expected = original[2000:6000]
    rate, actual = _read(path, 0.25, 0.5)
    assert rate == 8000
    np.testing.assert_array_equal(actual, expected - expected.mean())


def test_packed_24bit_pcm_falls_back_without_changing_values(tmp_path):
    import wave

    path = tmp_path / "packed.wav"
    samples = np.arange(8000, dtype=np.int32) - 4000
    packed = ((samples[:, None] >> np.array([0, 8, 16])) & 255).astype(np.uint8)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 3, 8000, 0, "NONE", "not compressed"))
        output.writeframes(packed.tobytes())
    rate, actual = _read(path, 0.25, 0.5)
    _, original = wavfile.read(path)
    expected = original.astype(np.float64)[2000:6000]
    assert rate == 8000
    np.testing.assert_array_equal(actual, expected - expected.mean())
