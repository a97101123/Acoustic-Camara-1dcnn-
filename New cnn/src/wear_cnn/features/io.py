"""Configuration and bounded HDF5 reads shared by feature-building pipelines."""

import numpy as np


def io_options(options):
    from ..configuration import fields

    config = fields(
        options, ("compression", "gzip_level", "read_batch_size", "write_batch_size"), "io"
    )
    if config["compression"] not in ("gzip", "lzf", "none"):
        raise ValueError("io.compression must be gzip, lzf or none")
    for key in ("read_batch_size", "write_batch_size"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"io.{key} must be a positive integer")
    level = config["gzip_level"]
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 9:
        raise ValueError("io.gzip_level must be an integer in [0, 9]")
    return config


def compression_options(options):
    config = io_options(options)
    if config["compression"] == "none":
        return {}
    result = {"compression": config["compression"], "shuffle": True}
    if config["compression"] == "gzip":
        result["compression_opts"] = config["gzip_level"]
    return result


def read_rows(dataset, indices):
    """Read a row block once, restoring caller order after sorted HDF5 indexing."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    if len(indices) == 1 or np.all(np.diff(indices) == 1):
        return dataset[int(indices[0]) : int(indices[-1]) + 1]
    unique, inverse = np.unique(indices, return_inverse=True)
    return dataset[unique][inverse]
