from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from wear_cnn.configuration import validate_training_config
from wear_cnn.feature_builder import validate_feature_config
from wear_cnn.run_io import load_config
from wear_cnn.trainer_batch import train_batch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name", ["turning_training", "drilling_training", "turning_binary_training"]
)
def test_shipped_training_manifest_contains_only_functional_references(name):
    path = ROOT / "configs" / f"{name}.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(manifest) == {"includes"}
    assert len(manifest["includes"]) >= 5
    _, cfg = load_config(path)
    validate_training_config(cfg)
    assert cfg["training"]["optimizer"]["betas"] == [0.9, 0.999]
    assert cfg["model"]["filters"] == [32, 64]


@pytest.mark.parametrize(
    "section,key",
    [
        ("model", "dropout"),
        ("training", "learning_rate"),
        ("evaluation", "batch_size"),
        ("preprocessing", "scale_epsilon"),
        ("io", "compression"),
        ("dataset", "class_names"),
    ],
)
def test_training_missing_settings_fail_before_opening_features(tmp_path, section, key):
    _, cfg = load_config(ROOT / "configs/turning_training.yaml")
    del cfg[section][key]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        train_batch(path, features=str(tmp_path / "nonexistent"))
    assert not (tmp_path / "results").exists()


def test_unknown_nested_setting_reports_full_path():
    _, cfg = load_config(ROOT / "configs/turning_training.yaml")
    cfg["training"]["optimizer"]["learn_rate"] = 1
    with pytest.raises(ValueError, match="training.optimizer.learn_rate"):
        validate_training_config(cfg)


def test_sampling_missing_setting_fails_before_opening_source(tmp_path):
    _, cfg = load_config(ROOT / "configs/turning_features.yaml")
    del cfg["sampling"]["overlap_fraction"]
    cfg["input"]["excel"] = "missing.csv"
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling.overlap_fraction"):
        validate_feature_config(path)


def test_include_cycle_and_duplicate_sections_are_rejected(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text("includes: [b.yaml]\n", encoding="utf-8")
    second.write_text("includes: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(first)
    second.write_text("training: {}\n", encoding="utf-8")
    first.write_text("includes: [b.yaml, b.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate config section: training"):
        load_config(first)


def test_model_uses_explicit_architecture_and_checkpoint_options():
    import torch

    from wear_cnn.model import build_model

    _, cfg = load_config(ROOT / "configs/turning_training.yaml")
    options = deepcopy(cfg["model"])
    options.update(filters=[8, 16], kernel_size=5, dropout=0.35, norm_groups=2)
    model = build_model(torch, 3, 2, options)
    assert model.conv1.out_channels == 8
    assert model.conv2.out_channels == 16
    assert model.conv1.kernel_size == (5,)
    assert model.dropout.p == 0.35
    assert model.norm1.num_groups == 2
    assert model(torch.ones((2, 3, 12))).shape == (2, 2)
    del options["dropout"]
    with pytest.raises(ValueError, match="model.dropout"):
        build_model(torch, 3, 2, options)
