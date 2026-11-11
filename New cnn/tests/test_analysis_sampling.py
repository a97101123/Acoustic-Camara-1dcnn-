import numpy as np
import pytest
import settings as explicit

from wear_cnn.features.analysis_sampling import analysis_sequence, chunk_sequence, resolve_sampling


@pytest.mark.parametrize(
    "nfft,frames",
    [(128, 750), (192, 500), (256, 375), (320, 300), (384, 250), (448, 214), (512, 187)],
)
def test_duration_drives_chunk_points(nfft, frames):
    cfg = resolve_sampling(explicit.sampling({}), 48000, nfft)
    assert cfg["chunk_frames"] == frames
    assert cfg["hop_samples"] == nfft // 4
    assert cfg["effective_chunk_seconds"] <= 0.5
    assert 0.5 - cfg["effective_chunk_seconds"] < cfg["hop_samples"] / 48000


def test_all_complete_chunks_survive_block_remainder():
    cfg = resolve_sampling(explicit.sampling({}), 48000, 128)
    sequence = np.arange(750 * 29 * 2, dtype=np.float32).reshape(750 * 29, 2)
    chunks, available = chunk_sequence(sequence, cfg)
    assert available == 29
    assert chunks.shape == (29, 750, 2)
    assert np.array_equal(chunks[1], sequence[750:1500])
    assert np.array_equal(chunks[-1], sequence[-750:])


def test_short_recording_keeps_chunks_below_block_size():
    cfg = resolve_sampling(explicit.sampling({}), 48000, 128)
    sequence = np.ones((750 * 3 + 100, 2), dtype=np.float32)
    chunks, available = chunk_sequence(sequence, cfg)
    assert available == 3
    assert chunks.shape == (3, 750, 2)


@pytest.mark.parametrize("retention", [None, "legacy_blocks"])
def test_legacy_sampling_retention_is_rejected(retention):
    cfg = resolve_sampling(explicit.sampling({}), 48000, 128)
    if retention is None:
        cfg.pop("chunk_retention")
    else:
        cfg["chunk_retention"] = retention
    with pytest.raises(ValueError, match="all_complete"):
        chunk_sequence(np.ones((750 * 29, 2)), cfg)


def test_recording_without_full_chunk_is_rejected():
    cfg = resolve_sampling(explicit.sampling({}), 48000, 128)
    with pytest.raises(ValueError, match="full chunks"):
        chunk_sequence(np.ones((749, 2)), cfg)


def test_nfft_changes_feature_sampling_and_chunk_length_together():
    rate = 16000
    wave = np.sin(2 * np.pi * 1000 * np.arange(rate * 3) / rate)
    counts = []
    for nfft in (128, 256, 512):
        cfg = resolve_sampling(explicit.sampling({"target_sample_rate": None}), rate, nfft)
        values, names = analysis_sequence(rate, wave, nfft, "mic", explicit.sampling(cfg))
        chunks, _ = chunk_sequence(values, cfg)
        assert chunks.shape[1] == int(0.5 * rate / (nfft / 4))
        assert len(names) == 7 + nfft // 2
        counts.append(len(chunks))
    assert max(counts) - min(counts) <= 1


def test_mixed_source_rates_produce_aligned_feature_samples():
    cfg = {"target_sample_rate": 48000}
    counts = []
    for rate in (192000, 48000):
        wave = np.sin(2 * np.pi * 1000 * np.arange(rate * 2) / rate)
        values, names = analysis_sequence(rate, wave, 128, "mic", explicit.sampling(cfg))
        resolved = resolve_sampling(explicit.sampling(cfg), rate, 128)
        chunks, _ = chunk_sequence(values, resolved)
        assert resolved["sample_rate"] == 48000
        assert resolved["chunk_frames"] == 750
        counts.append((len(values), chunks.shape))
    assert counts[0] == counts[1]
