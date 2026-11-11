import json

import h5py
import numpy as np
import pytest
import settings as explicit

from wear_cnn.features.reader import FeatureReader
from wear_cnn.inference import _deployment_metadata


def test_modern_split_vds_and_record_summary(tmp_path):
    source = tmp_path / "features.h5"
    split = tmp_path / "split.h5"
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    with h5py.File(source, "w") as h:
        h["samples/values"] = values
    with h5py.File(split, "w", libver="latest") as h:
        h.attrs["feature_set_identity"] = "modern-feature-set"
        node = h.create_group("splits/train/all")
        layout = h5py.VirtualLayout(values.shape, dtype=values.dtype)
        layout[:] = h5py.VirtualSource(str(source), "samples/values", shape=values.shape)
        node.create_virtual_dataset("values", layout)
        for key, data in {
            "feature_names": ["a", "b"],
            "tool_ids": ["T1", "T1"],
            "record_ids": ["R1", "R1"],
            "unit_ids": ["chunk_0", "chunk_1"],
        }.items():
            node.create_dataset(key, data=data, dtype=h5py.string_dtype())
        node["wear"] = [1.0, 1.0]
        node["labels"] = [0, 0]
    reader = FeatureReader(tmp_path / "cache", io_config=explicit.io())
    np.testing.assert_array_equal(reader.values(split), values)
    np.testing.assert_allclose(reader.record_summary(split).values, values.mean(axis=(0, 1))[None])
    np.testing.assert_array_equal(reader.values(split, columns=[1]), values[:, :, [1]])
    with h5py.File(split, "a") as h:
        del h.attrs["feature_set_identity"]
    with pytest.raises(ValueError, match="Legacy feature stores"):
        reader.values(split)


@pytest.mark.parametrize("retention", [None, "legacy_blocks"])
def test_deployment_rejects_old_chunk_policy(retention):
    sampling = {} if retention is None else {"chunk_retention": retention}
    with pytest.raises(ValueError, match="chunk_retention=all_complete"):
        _deployment_metadata(
            {
                "store_metadata": {
                    "feature_profile": "analysis_short_time_v1",
                    "sample_unit": "chunk",
                    "sampling_json": json.dumps(sampling),
                }
            }
        )


def test_deployment_accepts_modern_chunks_only():
    meta = {
        "feature_profile": "analysis_short_time_v1",
        "sample_unit": "chunk",
        "sampling_json": json.dumps({"chunk_retention": "all_complete"}),
    }
    assert _deployment_metadata({"store_metadata": meta}) is meta
    with pytest.raises(ValueError, match="Legacy deployments"):
        _deployment_metadata({})
