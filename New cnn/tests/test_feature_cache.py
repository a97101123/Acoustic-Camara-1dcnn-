import h5py
import numpy as np
import pytest
import settings as explicit

from wear_cnn.artifact_validation import valid_hdf5
from wear_cnn.features.reader import FeatureReader


def store(path, sensor, reorder=False):
    strings = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h:
        h.attrs.update(
            sensor=sensor,
            sample_unit="chunk",
            nfft=128,
            feature_set_identity="test",
            sampling_json='{"chunk_retention": "all_complete"}',
            **explicit.classification().to_metadata(),
        )
        for split in ("train", "val", "test"):
            g = h.create_group(f"splits/{split}/all")
            order = [2, 3, 0, 1] if reorder else [0, 1, 2, 3]
            g["values"] = np.arange(40, dtype=np.float32).reshape(4, 5, 2)[order]
            g["wear"] = np.array([10, 10, 90, 90])[order]
            g["labels"] = np.array([0, 0, 2, 2])[order]
            names = (
                ["mic_mean", "mic_freq_375Hz"]
                if sensor == "mic"
                else ["x_mean", "x_freq_band_0_750Hz"]
            )
            for key, values in [
                ("feature_names", names),
                ("tool_ids", np.array(["T1"] * 4)[order]),
                ("record_ids", np.array(["R1", "R1", "R2", "R2"])[order]),
                ("unit_ids", np.array(["C0", "C1", "C0", "C1"])[order]),
            ]:
                g.create_dataset(key, data=np.array(values, dtype=object), dtype=strings)


def virtual_store(source, target):
    with h5py.File(source) as src, h5py.File(target, "w") as dst:
        dst.attrs.update(dict(src.attrs))
        for split in ("train", "val", "test"):
            original = src[f"splits/{split}/all"]
            group = dst.create_group(f"splits/{split}/all")
            for name in original:
                if name != "values":
                    src.copy(original[name], group, name=name)
            values = original["values"]
            layout = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
            layout[:] = h5py.VirtualSource(str(source), values.name, shape=values.shape)
            group.create_virtual_dataset("values", layout)


def test_reader_handles_modern_virtual_split_view_and_column_order(tmp_path):
    source, view = tmp_path / "source.h5", tmp_path / "view.h5"
    store(source, "mic")
    virtual_store(source, view)
    with h5py.File(source) as h:
        expected = h["splits/train/all/values"][:]
    reader = FeatureReader(tmp_path / "cache", io_config=explicit.io())
    assert np.array_equal(reader.values(view), expected)
    assert np.array_equal(reader.values(view, columns=[1, 0]), expected[:, :, [1, 0]])


def test_validation_cache_rechecks_changed_virtual_source(tmp_path, monkeypatch):
    import os

    source, view = tmp_path / "source.h5", tmp_path / "view.h5"
    store(source, "mic")
    virtual_store(source, view)
    cache = set()
    assert valid_hdf5(view, cache=cache)
    original = h5py.Dataset.__getitem__
    reads = []

    def count_reads(dataset, key):
        if dataset.name.endswith("/values"):
            reads.append(dataset.name)
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", count_reads)
    assert valid_hdf5(view, cache=cache)
    assert not reads
    before = source.stat()
    with h5py.File(source, "r+") as handle:
        handle["splits/train/all/values"][0, 0, 0] = np.nan
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert not valid_hdf5(view, cache=cache)
    assert reads


def test_summary_cache_preserves_record_weighting_and_targets(tmp_path, monkeypatch):
    source = tmp_path / "mic.h5"
    store(source, "mic")
    reader = FeatureReader(tmp_path / "cache", io_config=explicit.io())
    table = reader.record_summary(source)
    with h5py.File(source, "r") as h:
        raw = h["splits/train/all/values"][:]
    assert np.array_equal(
        table.values, np.stack([raw[:2].mean(axis=(0, 1)), raw[2:].mean(axis=(0, 1))])
    )
    assert table.labels.tolist() == [0, 2]
    second = FeatureReader(tmp_path / "cache", io_config=explicit.io())
    monkeypatch.setattr(
        second,
        "_physical",
        lambda *a: (_ for _ in ()).throw(AssertionError("Expected cached summary")),
    )
    assert np.array_equal(second.record_summary(source).values, table.values)


def test_gpu_problem_shapes_and_checkpoint_roundtrip(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    from wear_cnn.model import build_model

    for channels in (14, 17):
        model = build_model(torch, channels, 3, explicit.model()).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for batch in (8, 32):
            x = torch.randn(batch, channels, 750, device="cuda")
            for _ in range(3):
                optimizer.zero_grad()
                loss = model(x).square().mean()
                loss.backward()
                optimizer.step()
                assert torch.isfinite(loss)
        torch.cuda.synchronize()
        path = tmp_path / f"model{channels}.pt"
        torch.save(model.state_dict(), path)
        restored = build_model(torch, channels, 3, explicit.model()).cuda()
        restored.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        model.eval()
        restored.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), restored(x))
