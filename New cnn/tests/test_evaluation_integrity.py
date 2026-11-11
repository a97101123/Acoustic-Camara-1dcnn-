import json

import h5py
import numpy as np
import pytest
import settings as explicit
from scipy.io import wavfile

from wear_cnn.artifact_validation import valid_hdf5
from wear_cnn.evaluation import evaluate_hdf5
from wear_cnn.inference import predict_wav


@pytest.fixture
def artifacts(tmp_path):
    import torch

    from wear_cnn.model import build_model

    torch.set_num_threads(1)
    store = tmp_path / "features.h5"
    metadata = dict(sensor="mic", nfft=128, sample_rate=8000, window_seconds=0.5, hop_seconds=0.1)
    from wear_cnn.features.analysis_sampling import resolve_sampling

    metadata.update(explicit.classification().to_metadata())
    metadata.update(
        feature_set_identity="test",
        sample_unit="chunk",
        feature_profile="analysis_short_time_v1",
        sampling_json=json.dumps(
            resolve_sampling(explicit.sampling({"target_sample_rate": None}), 8000, 128)
        ),
    )
    names = ["mic_mean", "mic_max"]
    with h5py.File(store, "w") as h:
        h.attrs.update(metadata)
        for split in ["train", "val", "test"]:
            g = h.create_group(f"splits/{split}/all")
            g["values"] = np.ones((3, 4, 2), dtype="float32")
            g["labels"] = [0, 1, 2]
            g["wear"] = [1.0, 5.0, 9.0]
            for key, values in dict(
                feature_names=names,
                tool_ids=[split] * 3,
                record_ids=["0", "1", "2"],
                unit_ids=["0", "1", "2"],
            ).items():
                g.create_dataset(
                    key, data=np.array(values, dtype=object), dtype=h5py.string_dtype()
                )
    checkpoint = tmp_path / "model.pt"
    saved = dict(
        state_dict=build_model(torch, 2, 3, explicit.model()).state_dict(),
        input_channels=2,
        feature_names=names,
        feature_indices=[0, 1],
        feature_group="all",
        median=np.zeros(2),
        scale=np.ones(2),
        metadata=metadata,
        model_config=explicit.model(),
        inference_batch_size=explicit.evaluation()["inference_batch_size"],
        inference=explicit.evaluation()["inference"],
    )
    torch.save(saved, checkpoint)
    return store, checkpoint, saved


@pytest.mark.parametrize("change", ["nfft", "names", "test_names"])
def test_evaluation_rejects_incompatible_schema(tmp_path, artifacts, change):
    store, model, _ = artifacts
    with h5py.File(store, "a") as h:
        if change == "nfft":
            h.attrs["nfft"] = 256
        else:
            splits = ["test"] if change == "test_names" else ["train", "val", "test"]
            for split in splits:
                h[f"splits/{split}/all/feature_names"][0] = "wrong_feature"
    with pytest.raises(ValueError, match="metadata|feature|schema"):
        evaluate_hdf5(
            str(store),
            str(model),
            tmp_path / "report",
            "cpu",
            make_plots=False,
            batch_size=explicit.evaluation()["batch_size"],
            reader=explicit.reader(),
        )
    assert not (tmp_path / "report/metrics.json").exists()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_corrupt_test_features_rejected_by_resume_and_evaluation(tmp_path, artifacts, bad):
    store, model, _ = artifacts
    with h5py.File(store, "a") as h:
        h["splits/test/all/values"][0, :, :] = bad
    assert not valid_hdf5(store)
    with pytest.raises(ValueError, match="finite"):
        evaluate_hdf5(
            str(store),
            str(model),
            tmp_path / "report",
            "cpu",
            make_plots=False,
            batch_size=explicit.evaluation()["batch_size"],
            reader=explicit.reader(),
        )
    assert not (tmp_path / "report/metrics.json").exists()


def test_evaluation_rejects_nonfinite_model_outputs(tmp_path, artifacts):
    import torch

    store, model, saved = artifacts
    saved["state_dict"]["head.bias"][0] = float("nan")
    torch.save(saved, model)
    with pytest.raises(ValueError, match="finite"):
        evaluate_hdf5(
            str(store),
            str(model),
            tmp_path / "report",
            "cpu",
            make_plots=False,
            batch_size=explicit.evaluation()["batch_size"],
            reader=explicit.reader(),
        )
    assert not (tmp_path / "report/metrics.json").exists()


@pytest.mark.parametrize("damage", ["nan", "missing", "cycle"])
def test_resume_validates_virtual_dataset_sources(tmp_path, artifacts, damage):
    store, _, _ = artifacts
    virtual = tmp_path / "virtual.h5"
    with h5py.File(store) as source, h5py.File(virtual, "w") as target:
        target.attrs.update(dict(source.attrs))
        for split in ["train", "val", "test"]:
            original = source[f"splits/{split}/all"]
            node = target.create_group(f"splits/{split}/all")
            for name in ["labels", "wear"]:
                source.copy(original[name], node, name=name)
            values = original["values"]
            layout = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
            layout[:] = h5py.VirtualSource(str(store), values.name, shape=values.shape)
            node.create_virtual_dataset("values", layout)
    assert valid_hdf5(virtual)
    if damage == "missing":
        store.unlink()
    elif damage == "nan":
        with h5py.File(store, "a") as source:
            source["splits/test/all/values"][0, :, :] = np.nan
    else:
        with h5py.File(virtual, "a") as target:
            name = "splits/test/all/values"
            shape = target[name].shape
            del target[name]
            layout = h5py.VirtualLayout(shape=shape, dtype="float32")
            layout[:] = h5py.VirtualSource(str(virtual), "/" + name, shape=shape)
            target.create_virtual_dataset(name, layout)
    assert not valid_hdf5(virtual)


@pytest.mark.parametrize("rate", [8000, 16000])
def test_record_prediction_checks_training_sample_rate(tmp_path, artifacts, rate):
    _, model, _ = artifacts
    audio = tmp_path / "audio.wav"
    wavfile.write(audio, rate, np.sin(np.arange(rate) * 0.2).astype("float32"))
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            dict(
                checkpoints=[str(model)],
                inference=explicit.evaluation()["inference"],
                inference_batch_size=explicit.evaluation()["inference_batch_size"],
                severe_probability_threshold=0.5,
                store_metadata=artifacts[2]["metadata"],
            )
        )
    )
    if rate == 16000:
        with pytest.raises(ValueError, match="sample rate"):
            predict_wav([str(audio)], str(deployment), "cpu")
    else:
        assert predict_wav([str(audio)], str(deployment), "cpu")["class_id"] in [0, 1, 2]


def test_legacy_time_only_model_without_training_rate_is_rejected(tmp_path, artifacts):
    import torch

    _, model, saved = artifacts
    del saved["metadata"]["feature_profile"]
    torch.save(saved, model)
    audio = tmp_path / "audio.wav"
    wavfile.write(audio, 8000, np.sin(np.arange(8000) * 0.2).astype("float32"))
    deployment = tmp_path / "deployment.json"
    deployment.write_text(
        json.dumps(
            dict(
                checkpoints=[str(model)],
                inference=explicit.evaluation()["inference"],
                inference_batch_size=explicit.evaluation()["inference_batch_size"],
                severe_probability_threshold=0.5,
                store_metadata=artifacts[2]["metadata"],
            )
        )
    )
    with pytest.raises(ValueError, match="Legacy deployments"):
        predict_wav([str(audio)], str(deployment), "cpu")
