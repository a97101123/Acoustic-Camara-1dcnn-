"""Identify physical tools and machining stages from ML recording paths."""

import re
from pathlib import Path, PurePosixPath


def recording_identity(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid FileName: {value!r}")
    path = PurePosixPath(value.replace("\\", "/"))
    stages = re.findall(
        r"(?<![A-Za-z0-9])(?:flank|stage|st)[_ -]*([0-9]+)(?![0-9])", path.stem, re.IGNORECASE
    )
    if not stages:
        stages = [
            match.group(1)
            for part in path.parts[:-1]
            if (match := re.fullmatch(r"(?:flank|st|stage)[_ -]*([0-9]+)", part, re.IGNORECASE))
        ]
    tools = [
        match.group(1)
        for part in path.parts[:-1]
        if (match := re.fullmatch(r"(?:knife|tool|T|刀具)[_ -]*([0-9]+)", part, re.IGNORECASE))
    ]
    if not tools and re.fullmatch(r"[0-9]+", path.parent.name):
        tools = [path.parent.name]
    if len(stages) != 1 or len(tools) != 1 or int(stages[0]) < 1 or int(tools[0]) < 1:
        raise ValueError(f"Cannot identify one tool folder and flank stage from FileName: {value}")
    return int(tools[0]), int(stages[0])


def read_paired_excel(sources, sheet=0, audio_root=None):
    """Pair sensor workbooks by physical identity, never by row position."""
    import numpy as np
    import pandas as pd

    if not isinstance(sources, dict) or not sources:
        raise ValueError("Excel sources must be a nonempty sensor-to-path mapping")
    paired = None
    for sensor, source in sources.items():
        source = Path(source).resolve()
        rows = pd.read_excel(source, sheet_name=sheet)
        if not {"FileName", "WearValue", "WAVcutstart"}.issubset(rows) or rows.empty:
            raise ValueError(f"{sensor}: Excel requires FileName, WearValue, WAVcutstart")
        identities = [recording_identity(value) for value in rows.FileName]
        rows["knife"] = [tool for tool, _ in identities]
        rows["stage"] = [stage for _, stage in identities]
        if rows.duplicated(["knife", "stage"]).any():
            raise ValueError(f"{sensor}: duplicate tool/stage")
        rows = rows.set_index(["knife", "stage"]).sort_index()
        numeric = rows[["WearValue", "WAVcutstart"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
            raise ValueError(f"{sensor}: WearValue and WAVcutstart must be finite and nonnegative")
        if paired is None:
            paired = pd.DataFrame({"wear_um": numeric.WearValue})
        elif not paired.index.equals(rows.index):
            raise ValueError(f"{sensor}: tool/stage pairing is incomplete")
        elif not np.array_equal(paired.wear_um, numeric.WearValue):
            raise ValueError(f"{sensor}: WearValue differs between paired sensors")
        start = "acc_start_sec" if sensor.startswith("acc_") else "mic_start_sec"
        if start in paired and not np.array_equal(paired[start], numeric.WAVcutstart):
            raise ValueError(f"{sensor}: paired {start} values differ")
        paired[start] = numeric.WAVcutstart
        root = Path(audio_root) if audio_root is not None else source.parent
        paired[sensor] = [str((root / value).resolve()) for value in rows.FileName]
        if any(
            (paired[sensor] == paired[other]).any()
            for other in sources
            if other != sensor and other in paired
        ):
            raise ValueError(f"{sensor}: the same audio file cannot represent different sensors")
    return paired.reset_index()
