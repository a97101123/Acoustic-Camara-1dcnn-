"""Reproducible CPU microbenchmarks; writes synthetic files to the configured output."""

import argparse
import json
import statistics
from time import perf_counter

import h5py
import numpy as np

from wear_cnn.configuration import fields
from wear_cnn.features.io import compression_options, read_rows
from wear_cnn.features.selection import r2_scores
from wear_cnn.run_io import load_config, resolve


def benchmark(config_path):
    source, config = load_config(config_path)
    fields(
        config,
        (
            "records",
            "score_features",
            "samples",
            "frames",
            "features",
            "read_batch_size",
            "repeats",
            "output",
            "seed",
            "score_backends",
            "compressions",
            "io",
        ),
        "benchmark",
    )
    for key in [
        "records",
        "score_features",
        "samples",
        "frames",
        "features",
        "read_batch_size",
        "repeats",
    ]:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    root = resolve(source.parent, config["output"])
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])

    def measure(operation):
        operation()  # Warm caches before collecting the median.
        times = []
        for _ in range(config["repeats"]):
            start = perf_counter()
            operation()
            times.append(perf_counter() - start)
        return statistics.median(times)

    values = rng.normal(size=(config["records"], config["score_features"]))
    wear = rng.uniform(size=config["records"])
    result = {"config": config, "r2": {}}
    for backend in config["score_backends"]:
        result["r2"][backend] = measure(lambda: r2_scores(values, wear, backend))
    np.testing.assert_allclose(
        r2_scores(values, wear, "matrix"), r2_scores(values, wear, "legacy"), atol=1e-12
    )
    data = rng.normal(size=(config["samples"], config["frames"], config["features"])).astype(
        "float32"
    )
    result["compression"] = {}
    for compression in config["compressions"]:
        target = root / f"synthetic_{compression}.h5"

        def write():
            with h5py.File(target, "w") as handle:
                handle.create_dataset(
                    "values",
                    data=data,
                    **compression_options({**config["io"], "compression": compression}),
                )

        seconds = measure(write)
        with h5py.File(target) as handle:
            assert np.array_equal(handle["values"][:], data)
        result["compression"][compression] = {"seconds": seconds, "bytes": target.stat().st_size}
    with h5py.File(root / "synthetic_gzip.h5") as handle:
        dataset = handle["values"]

        def per_row():
            return np.stack([dataset[i] for i in range(len(data))])

        def batched():
            block = config["read_batch_size"]
            return np.concatenate(
                [
                    read_rows(dataset, np.arange(start, min(len(data), start + block)))
                    for start in range(0, len(data), block)
                ]
            )

        assert np.array_equal(per_row(), batched())
        result["hdf5_reads"] = {"per_row": measure(per_row), "batched": measure(batched)}
    output = root / "benchmark.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    print(benchmark(parser.parse_args().config))
