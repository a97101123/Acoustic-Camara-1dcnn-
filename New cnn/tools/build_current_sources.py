"""Build current paired CSV inputs without overwriting the original manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH_COLUMNS = ["FileName", "AccXPath", "AccYPath", "AccZPath"]


def build_sources(process, data_root, destination):
    inputs = []
    if process == "turning":
        acc_path = ROOT / "data/h_drive/turning_acc.csv"
        acc = pd.read_csv(acc_path, dtype={"ToolID": str, "RecordID": str})
        inputs.append(acc_path)
        batches = []
        for condition, filename in [
            ("turning_noac", "turning_noAC.csv"),
            ("turning_withac", "turning_withAC.csv"),
        ]:
            path = ROOT / "data/h_drive" / filename
            mic = pd.read_csv(path, dtype={"ToolID": str, "RecordID": str})
            inputs.append(path)
            paired = mic.merge(
                acc, on=["ToolID", "RecordID"], suffixes=("_mic", "_acc"), validate="one_to_one"
            )
            if len(paired) != len(mic) or len(paired) != len(acc):
                raise ValueError(f"Incomplete MIC/ACC pairing: {condition}")
            if not np.array_equal(paired.WearValue_mic, paired.WearValue_acc):
                raise ValueError(f"MIC/ACC wear differs: {condition}")
            frame = paired[["ToolID", "RecordID", *PATH_COLUMNS]].copy()
            frame["WearValue"] = paired.WearValue_mic
            frame["MICStart"] = paired.WAVcutstart_mic
            frame["ACCStart"] = paired.WAVcutstart_acc
            if not np.array_equal(paired.Duration_mic, paired.Duration_acc):
                raise ValueError(f"MIC/ACC durations differ: {condition}")
            frame["Duration"] = paired.Duration_mic
            frame["Condition"] = condition
            frame["Enabled"] = True
            batches.append(frame)
        frame = pd.concat(batches, ignore_index=True)
    else:
        path = ROOT / "data/sources/drilling_sources.csv"
        inputs.append(path)
        frame = pd.read_csv(path, dtype={"ToolID": str, "RecordID": str})

    for column in PATH_COLUMNS:

        def remap(raw):
            value = str(raw).replace("\\", "/")
            prefix = f"H:/{process}/"
            if value.lower().startswith(prefix.lower()):
                return str(data_root / value[len(prefix) :])
            path = Path(value)
            return str(path if path.is_absolute() else data_root / path)

        frame[column] = frame[column].map(remap)
        missing = [p for p in frame[column] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    if frame.duplicated(["Condition", "ToolID", "RecordID"]).any():
        raise ValueError("Duplicate paired source identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    provenance = {
        "process": process,
        "rows": len(frame),
        "data_root": str(data_root),
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "wear_policy": "Copy source WearValue unchanged; no interpolation or unit conversion",
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"{process}: {len(frame)} rows -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", choices=["turning", "drilling"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build_sources(
        args.process,
        args.data_root.resolve(),
        args.output or ROOT / f"data/sources/{args.process}_current.csv",
    )


if __name__ == "__main__":
    main()
