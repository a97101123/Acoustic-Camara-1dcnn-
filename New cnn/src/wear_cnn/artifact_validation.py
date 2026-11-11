"""Validate stored model weights and feature arrays before batch reuse.

Checks are read-only. HDF5 virtual sources are resolved before their physical
values are checked in bounded blocks, avoiding native nested VDS traversal.
"""

import zipfile
from pathlib import Path

import numpy as np


def valid_checkpoint(path, expected_identity=None, min_epochs=1):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None or not any(
                name.endswith("/data.pkl") for name in archive.namelist()
            ):
                return False
        import torch

        from .classification import classification_from_metadata
        from .model import build_model

        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not saved.get("training_identity"):
            return False
        if expected_identity is not None and saved["training_identity"] != expected_identity:
            return False
        early_complete = (
            expected_identity is not None
            and saved.get("stopped_early") is True
            and isinstance(saved.get("stop_iteration"), int)
            and saved["stop_iteration"] > 0
        )
        if saved.get("epochs", 0) < min_epochs and not early_complete:
            return False
        names, indices = saved["feature_names"], saved["feature_indices"]
        if not names or len(names) != len(indices) or len(set(indices)) != len(indices):
            return False
        median, scale = np.asarray(saved["median"]), np.asarray(saved["scale"])
        if median.shape != (len(names),) or scale.shape != median.shape:
            return False
        if not np.isfinite(median).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            return False
        spec = classification_from_metadata(saved.get("metadata", {}))
        model = build_model(torch, saved["input_channels"], spec.num_classes, saved["model_config"])
        model.load_state_dict(saved["state_dict"], strict=True)
        return all(torch.isfinite(value).all().item() for value in model.state_dict().values())
    except Exception:
        return False


def valid_hdf5(path, cache=None):
    """Read physical values in bounded blocks, validating every VDS source first."""
    import h5py

    checked, active = set(), set()

    def check_values(filename, dataset_name):
        filename = Path(filename).resolve()
        key = (str(filename), dataset_name)
        if key in active:
            return False
        if key in checked:
            return True
        active.add(key)
        try:
            with h5py.File(filename, "r") as handle:
                dataset = handle[dataset_name]
                if dataset.ndim != 3 or min(dataset.shape) < 1:
                    return False
                if dataset.is_virtual:
                    sources = dataset.virtual_sources()
                    if not sources:
                        return False
                    for source in sources:
                        raw = (
                            source.file_name.decode()
                            if isinstance(source.file_name, bytes)
                            else source.file_name
                        )
                        target = filename if raw == "." else Path(raw)
                        if not target.is_absolute():
                            target = filename.parent / target
                        name = (
                            source.dset_name.decode()
                            if isinstance(source.dset_name, bytes)
                            else source.dset_name
                        )
                        if not check_values(target, name):
                            return False
                else:
                    stat = filename.stat()
                    signature = (str(filename), stat.st_size, stat.st_mtime_ns, dataset_name)
                    if cache is None or signature not in cache:
                        for offset in range(0, len(dataset), 32):
                            if not np.isfinite(dataset[offset : offset + 32]).all():
                                return False
                        if cache is not None:
                            cache.add(signature)
            checked.add(key)
            return True
        finally:
            active.remove(key)

    try:
        path = Path(path).resolve()
        with h5py.File(path, "r") as handle:
            if (
                not handle.attrs.get("feature_set_identity")
                or handle.attrs.get("sample_unit") != "chunk"
            ):
                return False
            for split in ("train", "val", "test"):
                node = handle[f"splits/{split}/all"]
                if not check_values(path, node["values"].name):
                    return False
                count = len(node["values"])
                for name in ("labels", "wear"):
                    values = node[name][:]
                    if len(values) != count or not np.isfinite(values).all():
                        return False
                if not np.isin(node["labels"][:], [0, 1, 2]).all():
                    return False
        return True
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, RecursionError):
        return False
