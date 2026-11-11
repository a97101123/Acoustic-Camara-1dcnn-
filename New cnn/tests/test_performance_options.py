import h5py
import numpy as np
import pytest
import settings as explicit

from wear_cnn.features.io import compression_options, io_options, read_rows
from wear_cnn.features.selection import r2_scores
from wear_cnn.training import train_hdf5


@pytest.mark.parametrize("rows", [[0, 1, 2], [8, 2, 5, 2], [], [4]])
def test_batched_hdf5_read_preserves_order_and_duplicates(tmp_path, rows):
    values = np.arange(120, dtype="float32").reshape(10, 3, 4)
    with h5py.File(tmp_path / "test.h5", "w") as handle:
        dataset = handle.create_dataset("values", data=values)
        assert np.array_equal(read_rows(dataset, rows), values[rows])


@pytest.mark.parametrize("compression", ["gzip", "lzf", "none"])
def test_compression_is_lossless(tmp_path, compression):
    values = np.random.default_rng(4).normal(size=(7, 11, 5)).astype("float32")
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "values", data=values, **compression_options(explicit.io({"compression": compression}))
        )
    with h5py.File(path) as handle:
        assert np.array_equal(handle["values"][:], values)


@pytest.mark.parametrize(
    "config",
    [
        {"write_batch_size": 0},
        {"read_batch_size": True},
        {"gzip_level": 10},
        {"compression": "lossy"},
        {"typo": 2},
    ],
)
def test_invalid_io_options_are_rejected(config):
    with pytest.raises(ValueError):
        io_options(explicit.io(config))


def test_matrix_r2_matches_regression_for_linear_noise_and_constant_features():
    rng = np.random.default_rng(42)
    wear = rng.uniform(0, 100, 1000)
    values = np.column_stack([wear, -wear, np.ones(1000), rng.normal(size=(1000, 50))])
    np.testing.assert_allclose(
        r2_scores(values, wear, "matrix"), r2_scores(values, wear, "legacy"), atol=1e-12
    )
    assert np.array_equal(r2_scores(values, np.ones(1000), "matrix"), np.zeros(values.shape[1]))


def make_training_store(path):
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            feature_set_identity="test",
            sample_unit="chunk",
            sampling_json='{"chunk_retention":"all_complete"}',
            **explicit.classification().to_metadata(),
        )
        for split in ["train", "val"]:
            node = handle.create_group(f"splits/{split}/all")
            node["values"] = rng.normal(size=(12, 8, 3)).astype("float32")
            node["labels"] = np.tile([0, 1, 2], 4)
            node["wear"] = np.arange(12)
            for key, values in {
                "feature_names": ["a", "b", "c"],
                "tool_ids": [split] * 12,
                "record_ids": [str(i) for i in range(12)],
                "unit_ids": ["chunk_000000"] * 12,
            }.items():
                node.create_dataset(
                    key, data=np.array(values, dtype=object), dtype=h5py.string_dtype()
                )


def test_sparse_checkpoint_and_preloading_keep_weights_and_resume_exact(tmp_path, monkeypatch):
    import torch

    import wear_cnn.training as training

    torch.set_num_threads(1)
    source = tmp_path / "data.h5"
    make_training_store(source)
    reference, accelerated = tmp_path / "reference.pt", tmp_path / "accelerated.pt"
    train_hdf5(
        source,
        reference,
        epochs=4,
        device="cpu",
        make_plots=False,
        options=explicit.training({"batch_size": 4}),
        seed=42,
        prepared=explicit.prepared(source),
    )
    real_save = training._save
    saved_epochs = []

    def save(torch_module, data, path):
        if path.name.endswith(".resume.pt"):
            saved_epochs.append(data["epoch"])
        return real_save(torch_module, data, path)

    def interrupt(record):
        if record["epoch"] == 3:
            raise RuntimeError("stop after an unsaved epoch")

    monkeypatch.setattr(training, "_save", save)
    options = {"batch_size": 4, "checkpoint_every": 2, "preload_device": True}
    with pytest.raises(RuntimeError, match="unsaved"):
        train_hdf5(
            source,
            accelerated,
            epochs=4,
            device="cpu",
            make_plots=False,
            options=explicit.training(options),
            progress=interrupt,
            seed=42,
            prepared=explicit.prepared(source),
        )
    assert saved_epochs == [2]
    train_hdf5(
        source,
        accelerated,
        epochs=4,
        device="cpu",
        make_plots=False,
        options=explicit.training(options),
        seed=42,
        prepared=explicit.prepared(source),
    )
    assert saved_epochs == [2, 4]
    before, after = [torch.load(path, weights_only=False) for path in [reference, accelerated]]
    assert all(
        torch.equal(before["state_dict"][key], after["state_dict"][key])
        for key in before["state_dict"]
    )
