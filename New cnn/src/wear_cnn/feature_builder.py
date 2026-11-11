"""Publish immutable, split-independent feature sets from paired Excel sources."""

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .features.analysis_sampling import build_analysis_hdf5, sampling_options
from .features.io import compression_options, io_options, read_rows
from .filename_templates import FEATURE_FILES, resolve_names
from .run_io import atomic_json, load_config, new_run, output_manifest, resolve, sha256

PATH_COLUMNS = ["FileName", "AccXPath", "AccYPath", "AccZPath"]
IDENTITY = ["ToolID", "RecordID"]


def read_sources(path, data_root, conditions, sensors, exclude_tools, *, sheet=0, duration=None):
    if isinstance(path, dict):
        from .configuration import positive
        from .source_identity import read_paired_excel

        expected = ({"mic"} if "mic" in sensors else set()) | (
            {"acc_x", "acc_y", "acc_z"} if "acc" in sensors else set()
        )
        if set(path) != expected or not conditions or len(conditions) != 1:
            raise ValueError(
                "Paired Excel requires one condition and exactly the configured sensor sources"
            )
        positive(duration, "input.duration_seconds")
        frame = read_paired_excel(path, sheet, data_root).rename(
            columns={
                "mic": "FileName",
                "acc_x": "AccXPath",
                "acc_y": "AccYPath",
                "acc_z": "AccZPath",
                "wear_um": "WearValue",
                "mic_start_sec": "MICStart",
                "acc_start_sec": "ACCStart",
            }
        )
        frame["ToolID"] = frame.knife.map(lambda value: f"T{value:02d}")
        frame["RecordID"] = frame.stage.map(lambda value: f"M{value:03d}")
        frame["Duration"] = duration
        frame["Condition"] = conditions[0]
    else:
        path = Path(path)
        identity_types = {"ToolID": str, "RecordID": str}
        frame = (
            pd.read_csv(path, dtype=identity_types)
            if path.suffix.lower() == ".csv"
            else pd.read_excel(path, sheet_name=sheet, dtype=identity_types)
        )
    if {"FileName", "WearValue", "WAVcutstart"}.issubset(frame) and not set(IDENTITY) & set(frame):
        from .configuration import positive
        from .source_identity import recording_identity

        if sensors != ["mic"] or not conditions or len(conditions) != 1:
            raise ValueError(
                "ML Excel requires sensors: [mic] and exactly one condition; ACC requires paired paths"
            )
        positive(duration, "input.duration_seconds")
        identities = [recording_identity(value) for value in frame.FileName]
        frame["ToolID"] = [f"T{tool:02d}" for tool, _ in identities]
        frame["RecordID"] = [f"M{stage:03d}" for _, stage in identities]
        frame["MICStart"] = frame.WAVcutstart
        frame["Duration"] = duration
        frame["Condition"] = conditions[0]
    columns = (["FileName"] if "mic" in sensors else []) + (
        PATH_COLUMNS[1:] if "acc" in sensors else []
    )
    crop_columns = (
        (["MICStart"] if "mic" in sensors else [])
        + (["ACCStart"] if "acc" in sensors else [])
        + ["Duration"]
    )
    required = set(columns + IDENTITY + ["Condition", "WearValue"] + crop_columns)
    if required - set(frame):
        raise ValueError(f"Missing source columns: {sorted(required - set(frame))}")
    if "Enabled" in frame:
        enabled = frame.Enabled.astype(str).str.lower()
        if not enabled.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Enabled must contain true/false or 1/0")
        frame = frame[enabled.isin(["true", "1"])].copy()
    if conditions:
        frame = frame[frame.Condition.isin(conditions)].copy()
    if exclude_tools:
        frame = frame[~frame.ToolID.astype(str).isin(exclude_tools)].copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Sources are empty or contain missing required values")
    frame[IDENTITY] = frame[IDENTITY].astype(str)
    if frame.duplicated(["Condition", *IDENTITY]).any():
        raise ValueError("Duplicate condition/tool/record identity")
    frame["WearValue"] = pd.to_numeric(frame.WearValue, errors="raise")
    if not np.isfinite(frame.WearValue).all() or (frame.WearValue < 0).any():
        raise ValueError("WearValue must be finite and nonnegative, in micrometres")
    if "WearUnit" in frame and not frame.WearUnit.isin(["um", "µm", "μm"]).all():
        raise ValueError("This profile requires WearUnit in micrometres")
    for column in (
        (["MICStart"] if "mic" in sensors else [])
        + (["ACCStart"] if "acc" in sensors else [])
        + ["Duration"]
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if (
                not np.isfinite(frame[column]).all()
                or (frame[column] < 0).any()
                or (column == "Duration" and (frame[column] == 0).any())
            ):
                raise ValueError(
                    f"{column} requires explicit finite valid values for every enabled row; align ACC with cut MIC"
                )
    for column in columns:
        frame[column] = [str(resolve(data_root, str(p))) for p in frame[column]]
        for p in frame[column]:
            if not Path(p).is_file():
                raise FileNotFoundError(p)
    return frame


def feature_catalog(names):
    rows = []
    for index, name in enumerate(names):
        sensor, raw = name.split("::", 1)
        frequency = "_freq_" in raw
        numbers = re.findall(r"\d+(?:\.\d+)?", raw.split("freq_", 1)[-1]) if frequency else []
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "sensor": sensor,
                "axis": raw[0] if sensor == "acc" else "",
                "domain": "acc"
                if sensor == "acc"
                else ("mic_frequency" if frequency else "mic_time"),
                "frequency_low_hz": float(numbers[0]) if numbers else None,
                "frequency_high_hz": float(numbers[-1]) if numbers else None,
                "unit": "profile_defined",
            }
        )
    return pd.DataFrame(rows)


def validate_feature_config(config_path):
    source, cfg = load_config(config_path)
    from .configuration import fields

    fields(
        cfg,
        (
            "process",
            "io",
            "output_mode",
            "channel_mix",
            "sensors",
            "conditions",
            "exclude_tools",
            "sampling",
            "nffts",
            "input",
            "output",
        ),
        "config",
    )
    fields(cfg["input"], ("excel", "data_root"), "input", ("sheet", "duration_seconds"))
    sheet = cfg["input"].get("sheet", 0)
    if not ((type(sheet) is int and sheet >= 0) or (isinstance(sheet, str) and sheet)):
        raise ValueError("input.sheet must be a nonnegative index or nonempty sheet name")
    fields(cfg["output"], ("root", "name"), "output", ("filenames",))
    cfg["io"] = io_options(cfg["io"])
    sampling = sampling_options(cfg["sampling"])
    process = cfg["process"]
    if process not in ("drilling", "turning"):
        raise ValueError("process must be drilling or turning")
    if cfg["output_mode"] != "features_only":
        raise ValueError("Only features_only is implemented in this phase")
    if cfg["channel_mix"] != "mean":
        raise ValueError("Only the existing mean channel mix is implemented")
    sensors = cfg["sensors"]
    if (
        not isinstance(sensors, list)
        or not sensors
        or len(set(sensors)) != len(sensors)
        or set(sensors) - {"mic", "acc"}
    ):
        raise ValueError("sensors must contain mic and/or acc without duplicates")
    cfg["sensors"] = sensors
    excel = cfg["input"]["excel"]
    sources = (
        {key: resolve(source.parent, value) for key, value in excel.items()}
        if isinstance(excel, dict)
        else resolve(source.parent, excel)
    )
    frame = read_sources(
        sources,
        resolve(source.parent, cfg["input"]["data_root"]),
        cfg["conditions"],
        sensors,
        cfg["exclude_tools"],
        sheet=sheet,
        duration=cfg["input"].get("duration_seconds"),
    )

    nffts = cfg["nffts"]
    if (
        not nffts
        or len(set(nffts)) != len(nffts)
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2 for n in nffts)
    ):
        raise ValueError("nffts must be unique even positive integers >= 2")
    for condition in frame.Condition.unique():
        for nfft in nffts:
            resolve_names(
                FEATURE_FILES,
                cfg["output"].get("filenames"),
                {"process": process, "condition": condition, "nfft": nfft, "sensor": "fusion"},
            )
    # Validate the exact sample bounds used by the extractor and check PCM integrity.
    import wave

    for _, row in frame.iterrows():
        cut_durations = []
        for column in (["FileName"] if "mic" in sensors else []) + (
            PATH_COLUMNS[1:] if "acc" in sensors else []
        ):
            with wave.open(row[column], "rb") as wav:
                if wav.getnchannels() > 2:
                    raise ValueError(
                        "More than two channels requires explicit DAS/channel selection; not implemented here"
                    )
                start = float(row["MICStart" if column == "FileName" else "ACCStart"])
                duration = float(row["Duration"])
                rate = wav.getframerate()
                begin, count = int(start * rate), int(duration * rate)
                if start < 0 or count < 16 or begin + count > wav.getnframes():
                    raise ValueError(f"Invalid cut window: {row[column]}")
                expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
                actual = 0
                while True:
                    block = wav.readframes(262144)
                    if not block:
                        break
                    actual += len(block)
                if actual != expected:
                    raise ValueError(f"Truncated PCM data: {row[column]}")
                cut_durations.append((count / rate, rate))
        if max(t for t, _ in cut_durations) - min(t for t, _ in cut_durations) > max(
            1 / rate for _, rate in cut_durations
        ) + 1e-9:
            raise ValueError(
                f"MIC/ACC durations differ: {row.ToolID}/{row.RecordID}; supply aligned starts/duration"
            )
    return source, cfg, frame, sampling, nffts


def build_features(config_path, on_run=None):
    source, cfg, frame, sampling, nffts = validate_feature_config(config_path)
    process = cfg["process"]
    root = new_run(source.parent, cfg["output"], process)
    cfg["sampling"] = sampling
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    frame.to_csv(root / "source_rows.csv", index=False, encoding="utf-8-sig")
    state = {
        "status": "building",
        "schema_version": 1,
        "process": process,
        "entries": [],
        "pid": __import__("os").getpid(),
    }
    atomic_json(root / "state.json", state)
    if on_run:
        on_run(root)
    try:
        sensors = cfg["sensors"]
        columns = (["FileName"] if "mic" in sensors else []) + (
            PATH_COLUMNS[1:] if "acc" in sensors else []
        )
        hashes = {p: sha256(p) for p in sorted(set(frame[columns].to_numpy().ravel()))}
        provenance = {
            "source_excel_sha256": {
                key: sha256(resolve(source.parent, value))
                for key, value in cfg["input"]["excel"].items()
            }
            if isinstance(cfg["input"]["excel"], dict)
            else sha256(resolve(source.parent, cfg["input"]["excel"])),
            "files": hashes,
            "sampling": sampling,
            "sensors": sensors,
        }
        atomic_json(root / "source_manifest.json", provenance)
        for condition, batch in frame.groupby("Condition", sort=True):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", str(condition)):
                raise ValueError("Condition must use letters, numbers, underscore or hyphen")
            for nfft in nffts:
                directory = root / str(condition) / f"nfft{nfft}"
                directory.mkdir(parents=True)
                names_config = cfg["output"].get("filenames")
                filenames = resolve_names(
                    FEATURE_FILES,
                    names_config,
                    {"process": process, "condition": condition, "nfft": nfft, "sensor": "fusion"},
                )
                print(f"FEATURES {condition} NFFT={nfft}", flush=True)
                for sensor in sensors:
                    sensor_dir = directory / sensor
                    sensor_dir.mkdir()
                    rows = batch.copy()
                    start = "MICStart" if sensor == "mic" else "ACCStart"
                    if start in rows:
                        rows["WAVcutstart"] = rows[start]
                    build_analysis_hdf5(
                        rows,
                        sensor,
                        nfft,
                        sensor_dir / "features.h5",
                        sampling,
                        independent=True,
                        io_config=cfg["io"],
                    )
                    sensor_names = resolve_names(
                        FEATURE_FILES,
                        names_config,
                        {
                            "process": process,
                            "condition": condition,
                            "nfft": nfft,
                            "sensor": sensor,
                        },
                    )
                    for key in ("sample_manifest", "sample_counts"):
                        old = sensor_dir / FEATURE_FILES[key]
                        if old.name != sensor_names[key]:
                            old.replace(sensor_dir / sensor_names[key])
                target = directory / filenames["features"]
                if len(sensors) == 1:
                    sensor = sensors[0]
                    (directory / sensor / "features.h5").replace(target)
                    with h5py.File(target, "a") as out:
                        g = out["samples"]
                        names = [sensor + "::" + n for n in g["feature_names"].asstr()[:]]
                        del g["feature_names"]
                        g.create_dataset(
                            "feature_names",
                            data=np.array(names, dtype=object),
                            dtype=h5py.string_dtype(),
                        )
                        out.attrs.update(process=process, condition=condition)
                    feature_catalog(names).assign(nfft=nfft).to_csv(
                        directory / filenames["catalog"], index=False, encoding="utf-8-sig"
                    )
                    validate_feature_store(target)
                    state["entries"].append(
                        {
                            "condition": str(condition),
                            "nfft": nfft,
                            "path": str(target.relative_to(root)),
                            "catalog": str((directory / filenames["catalog"]).relative_to(root)),
                            "sha256": sha256(target),
                        }
                    )
                    atomic_json(root / "state.json", state)
                    continue
                with (
                    h5py.File(directory / "mic/features.h5") as mic,
                    h5py.File(directory / "acc/features.h5") as acc,
                    h5py.File(target, "w") as out,
                ):
                    mg, ag = mic["samples"], acc["samples"]

                    def decode(node):
                        return node.asstr()[:].tolist()

                    def keys(g):
                        return list(
                            zip(*[decode(g[k]) for k in ("tool_ids", "record_ids", "unit_ids")])
                        )

                    mkeys, akeys = keys(mg), keys(ag)
                    if len(set(mkeys)) != len(mkeys) or set(mkeys) != set(akeys):
                        raise ValueError("MIC/ACC sample identities differ")
                    lookup = {key: i for i, key in enumerate(akeys)}
                    order = np.array([lookup[key] for key in mkeys])
                    if (
                        not np.allclose(mg["wear"][:], ag["wear"][:][order])
                        or mg["values"].shape[1] != ag["values"].shape[1]
                    ):
                        raise ValueError("MIC/ACC wear or temporal lengths differ")
                    g = out.create_group("samples")
                    m, a = mg["values"], ag["values"]
                    values = g.create_dataset(
                        "values",
                        shape=(*m.shape[:-1], m.shape[-1] + a.shape[-1]),
                        dtype="float32",
                        chunks=(
                            min(cfg["io"]["write_batch_size"], len(m)),
                            m.shape[1],
                            m.shape[-1] + a.shape[-1],
                        ),
                        **compression_options(cfg["io"]),
                    )
                    print(f"FUSION {condition} NFFT={nfft} samples={len(m)}", flush=True)
                    for offset in range(0, len(m), cfg["io"]["write_batch_size"]):
                        stop = min(len(m), offset + cfg["io"]["write_batch_size"])
                        # h5py fancy indexing requires sorted indices; preserve identity order.
                        av = read_rows(a, order[offset:stop])
                        values[offset:stop] = np.concatenate([m[offset:stop], av], axis=-1)
                        if (
                            offset == 0
                            or stop == len(m)
                            or offset // cfg["io"]["write_batch_size"] % 20 == 0
                        ):
                            print(
                                f"FUSION {condition} NFFT={nfft} samples={stop}/{len(m)}",
                                flush=True,
                            )
                    names = ["mic::" + n for n in decode(mg["feature_names"])] + [
                        "acc::" + n for n in decode(ag["feature_names"])
                    ]
                    for key in ("tool_ids", "record_ids", "unit_ids", "wear"):
                        mic.copy(mg[key], g, name=key)
                    g.create_dataset(
                        "feature_names",
                        data=np.array(names, dtype=object),
                        dtype=h5py.string_dtype(),
                    )
                    out.attrs.update(dict(mic.attrs))
                    out.attrs.update(
                        sensor="fusion",
                        process=process,
                        condition=condition,
                        mic_nfft=nfft,
                        acc_nfft=nfft,
                    )
                    feature_catalog(names).assign(nfft=nfft).to_csv(
                        directory / filenames["catalog"], index=False, encoding="utf-8-sig"
                    )
                for sensor in ("mic", "acc"):
                    (directory / sensor / "features.h5").unlink()
                validate_feature_store(target)
                state["entries"].append(
                    {
                        "condition": str(condition),
                        "nfft": nfft,
                        "path": str(target.relative_to(root)),
                        "catalog": str((directory / filenames["catalog"]).relative_to(root)),
                        "sha256": sha256(target),
                    }
                )
                atomic_json(root / "state.json", state)
        # Detect changing sources rather than publishing inconsistent provenance.
        if any(sha256(p) != digest for p, digest in hashes.items()):
            raise ValueError("A source WAV changed while features were built")
        for entry in state["entries"]:
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise ValueError("A feature artifact changed before publication")
            validate_feature_store(root / entry["path"])
        state["status"] = "complete"
        state["identity"] = (
            __import__("hashlib")
            .sha256(
                json.dumps(
                    {"provenance": provenance, "entries": state["entries"]}, sort_keys=True
                ).encode()
            )
            .hexdigest()
        )
        atomic_json(root / "feature_set.json", state)
        atomic_json(root / "state.json", state)
        (root / "README.md").write_text(
            "# Feature Set\n\nSplit-independent MIC/ACC features. See feature_set.json, feature_catalog.csv and per-sensor sample CSVs.\n",
            encoding="utf-8",
        )
        output_manifest(root)
        return root
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        atomic_json(root / "state.json", state)
        raise


def validate_feature_store(path):
    """Reopen physical output and read every value before publishing its hash."""
    with h5py.File(path, "r") as h:
        node = h["samples"]
        values = node["values"]
        if values.ndim != 3 or min(values.shape) < 1 or values.is_virtual:
            raise ValueError("Invalid physical feature shape")
        if len(node["feature_names"]) != values.shape[-1]:
            raise ValueError("Feature name count differs from values")
        for key in ("tool_ids", "record_ids", "unit_ids", "wear"):
            if len(node[key]) != len(values):
                raise ValueError("Feature identity count differs from values")
            node[key][:]
        keys = list(
            zip(*(node[k].asstr()[:].tolist() for k in ("tool_ids", "record_ids", "unit_ids")))
        )
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate sample identities")
        if not np.isfinite(node["wear"][:]).all():
            raise ValueError("Nonfinite wear values")
        for start in range(0, len(values), 32):
            if not np.isfinite(values[start : start + 32]).all():
                raise ValueError("Nonfinite feature values")
