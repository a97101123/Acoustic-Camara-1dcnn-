import zipfile

import numpy as np
import pandas as pd
import pytest
import settings as explicit

from wear_cnn.artifact_validation import valid_checkpoint
from wear_cnn.feature_builder import read_sources
from wear_cnn.features.selection import fisher_j_scores
from wear_cnn.splitting import apply_labels, training_tertiles


def test_perfect_fisher_feature_outranks_noisy_feature():
    labels = np.repeat(np.arange(3), 2)
    values = np.column_stack([labels * 10, labels * 10 + np.tile([0, 1], 3), np.ones(6)])
    scores = fisher_j_scores(values, labels)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1] > scores[2] == 0


def test_manifest_keeps_exact_identifiers(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    for suffix in ["csv", "xlsx"]:
        path = tmp_path / ("source." + suffix)
        frame = pd.DataFrame(
            dict(
                ToolID=["001", "1"],
                RecordID=["0001", "0002"],
                FileName=[str(audio)] * 2,
                WearValue=[1.0, 2.0],
                Condition=["test"] * 2,
                MICStart=[0, 0],
                ACCStart=[0, 0],
                Duration=[1, 1],
            )
        )
        if suffix == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False)
        loaded = read_sources(path, tmp_path, sensors=["mic"], conditions=None, exclude_tools=[])
        assert loaded.ToolID.tolist() == ["001", "1"]
        assert loaded.RecordID.tolist() == ["0001", "0002"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0])
def test_invalid_wear_rejected_at_import_and_labeling(tmp_path, bad):
    audio = tmp_path / "audio.wav"
    audio.touch()
    frame = pd.DataFrame(
        dict(
            ToolID=["T1"],
            RecordID=["R1"],
            FileName=[str(audio)],
            WearValue=[bad],
            Condition=["test"],
            MICStart=[0],
            ACCStart=[0],
            Duration=[1],
            Split=["train"],
        )
    )
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="WearValue|missing required"):
        read_sources(path, tmp_path, sensors=["mic"], conditions=None, exclude_tools=[])
    with pytest.raises(ValueError, match="WearValue"):
        apply_labels(frame, (1.0, 2.0))
    with pytest.raises(ValueError, match="WearValue"):
        training_tertiles(frame)


def test_plain_zip_is_not_a_model(tmp_path):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hello.txt", "not a model")
    assert not valid_checkpoint(path)


def test_checkpoint_validates_weights_and_training_identity(tmp_path):
    import torch

    from wear_cnn.model import build_model

    path = tmp_path / "model.pt"
    checkpoint = dict(
        state_dict=build_model(torch, 2, 3, explicit.model()).state_dict(),
        input_channels=2,
        training_identity="expected",
        epochs=3,
        feature_indices=[0, 1],
        feature_names=["a", "b"],
        feature_group="all",
        median=np.zeros(2),
        scale=np.ones(2),
        model_config=explicit.model(),
        inference_batch_size=explicit.evaluation()["inference_batch_size"],
        inference=explicit.evaluation()["inference"],
        metadata=explicit.classification().to_metadata(),
    )
    torch.save(checkpoint, path)
    assert valid_checkpoint(path, expected_identity="expected", min_epochs=3)
    assert not valid_checkpoint(path, expected_identity="wrong", min_epochs=3)
    assert not valid_checkpoint(path, expected_identity="expected", min_epochs=4)
    checkpoint["state_dict"]["head.weight"][0, 0] = float("nan")
    torch.save(checkpoint, path)
    assert not valid_checkpoint(path)
