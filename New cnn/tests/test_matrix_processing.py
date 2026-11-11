import numpy as np
import pytest
import settings as explicit

from wear_cnn.features.analysis_sampling import analysis_sequence, sampling_options


@pytest.mark.parametrize("nfft", [128, 192, 256, 320, 384, 448, 512])
@pytest.mark.parametrize("sensor", ["mic", "acc"])
@pytest.mark.parametrize("kind", ["noise", "silence", "tone"])
def test_matrix_features_match_scipy(nfft, sensor, kind):
    rate = 16000
    signal = np.random.default_rng(42).normal(size=rate)
    if kind == "silence":
        signal[:] = 0
    if kind == "tone":
        signal = np.sin(2 * np.pi * 350 * np.arange(rate) / rate)
    options = {"target_sample_rate": None, "segments_per_record": 5, "time_backend": "scipy"}
    expected, names = analysis_sequence(rate, signal, nfft, sensor, explicit.sampling(options))
    actual, fast_names = analysis_sequence(
        rate,
        signal,
        nfft,
        sensor,
        explicit.sampling({**options, "time_backend": "matrix", "frame_batch_size": 37}),
    )
    assert names == fast_names
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(
    "options", [{"time_backend": "bad"}, {"frame_batch_size": 0}, {"frame_batch_size": True}]
)
def test_invalid_matrix_options_rejected(options):
    with pytest.raises(ValueError):
        sampling_options(explicit.sampling(options))
