"""Read canonical physical arrays, with reusable disk and record-summary caches.

Modern Trainer split views use native HDF5 virtual-dataset traversal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from .io import io_options
from .store import FeatureTable


def texts(dataset):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]]


class FeatureReader:
    def __init__(self, cache_dir=None, *, io_config):
        self.read_batch_size = io_options(io_config)["read_batch_size"]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._arrays = {}
        self._summaries = {}

    def fingerprint(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"

    def _key(self, path, split, kind):
        return hashlib.sha256(f"{self.fingerprint(path)}|{split}|{kind}".encode()).hexdigest()

    def _physical(self, path, split):
        key = self._key(path, split, "values")
        if key in self._arrays:
            return self._arrays[key]
        with h5py.File(path, "r") as h:
            source = h[f"splits/{split}/all/values"]
            if source.is_virtual and not h.attrs.get("feature_set_identity"):
                raise ValueError("Expected physical source values")
            if self.cache_dir:
                target = self.cache_dir / f"{key}.npy"
                if not target.exists():
                    temporary = target.with_suffix(".npy.tmp")
                    mapped = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=source.dtype, shape=source.shape
                    )
                    for start in range(0, len(source), self.read_batch_size):
                        mapped[start : start + self.read_batch_size] = source[
                            start : start + self.read_batch_size
                        ]
                    mapped.flush()
                    del mapped
                    temporary.replace(target)
                result = np.load(target, mmap_mode="r", allow_pickle=False)
            else:
                result = source[:]
        self._arrays[key] = result
        return result

    def _identity(self, path, split, group="all"):
        with h5py.File(path, "r") as h:
            if not h.attrs.get("feature_set_identity"):
                raise ValueError(
                    "Legacy feature stores are unsupported; rebuild with build-features/train-batch"
                )
            node = h[f"splits/{split}/{group}"]
            if node["values"].ndim != 3:
                raise ValueError("Expected chunk feature values with three dimensions")
            return {
                "names": texts(node["feature_names"]),
                "tools": texts(node["tool_ids"]),
                "records": texts(node["record_ids"]),
                "units": texts(node["unit_ids"]),
                "wear": node["wear"][:],
                "labels": node["labels"][:],
            }

    def values(self, path, group="all", split="train", columns=None):
        identity = self._identity(path, split, group)
        chosen = list(range(len(identity["names"]))) if columns is None else list(columns)
        if not chosen or min(chosen) < 0 or max(chosen) >= len(identity["names"]):
            raise ValueError("Invalid feature indices")
        requested = [identity["names"][i] for i in chosen]
        canonical = self._identity(path, split)
        lookup = {name: i for i, name in enumerate(canonical["names"])}
        indices = [lookup[name] for name in requested]
        physical = self._physical(path, split)
        if indices == list(range(physical.shape[-1])):
            return physical
        return np.take(physical, indices, axis=-1)

    def table(self, path, group="all", split="train"):
        identity = self._identity(path, split, group)
        return FeatureTable(
            self.values(path, group, split),
            identity["wear"],
            identity["labels"],
            identity["names"],
            identity["tools"],
            identity["records"],
        )

    def record_summary(self, path, group="all", split="train"):
        """Preserve one equally weighted observation per ToolID+RecordID."""
        identity = self._identity(path, split, group)
        key = self._key(path, split, "record_summary")
        if key not in self._summaries:
            target = self.cache_dir / f"{key}.npz" if self.cache_dir else None
            if target and target.exists():
                with np.load(target, allow_pickle=False) as saved:
                    result = FeatureTable(
                        saved["values"],
                        saved["wear"],
                        saved["labels"],
                        saved["names"].tolist(),
                        saved["tools"].tolist(),
                        saved["records"].tolist(),
                    )
            else:
                canonical = self._identity(path, split)
                values = self.values(path, "all", split)
                grouped = {}
                for index, pair in enumerate(zip(canonical["tools"], canonical["records"])):
                    grouped.setdefault(pair, []).append(index)
                averages, wear, labels = [], [], []
                for pair, indices in grouped.items():
                    if len(set(canonical["labels"][indices])) != 1 or not np.allclose(
                        canonical["wear"][indices], canonical["wear"][indices[0]]
                    ):
                        raise ValueError(f"Targets differ within record {pair}")
                    sample = values[indices]
                    averages.append(sample.mean(axis=tuple(range(sample.ndim - 1))))
                    wear.append(canonical["wear"][indices[0]])
                    labels.append(canonical["labels"][indices[0]])
                result = FeatureTable(
                    np.asarray(averages),
                    np.asarray(wear),
                    np.asarray(labels),
                    canonical["names"],
                    [p[0] for p in grouped],
                    [p[1] for p in grouped],
                )
                if target:
                    temporary = target.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        np.savez(
                            stream,
                            values=result.values,
                            wear=result.wear,
                            labels=result.labels,
                            names=result.feature_names,
                            tools=result.tool_ids,
                            records=result.record_ids,
                        )
                    temporary.replace(target)
            self._summaries[key] = result
        summary = self._summaries[key]
        lookup = {name: i for i, name in enumerate(summary.feature_names)}
        return FeatureTable(
            summary.values[:, [lookup[n] for n in identity["names"]]],
            summary.wear,
            summary.labels,
            identity["names"],
            summary.tool_ids,
            summary.record_ids,
        )

    def clear_memory(self):
        self._arrays.clear()
        self._summaries.clear()
