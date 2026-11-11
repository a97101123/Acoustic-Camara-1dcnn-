from __future__ import annotations

import numpy as np
from scipy.io import wavfile


def _read(path: str, start: float, duration: float | None) -> tuple[int, np.ndarray]:
    try:
        start = float(start)
        duration = None if duration is None else float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("Audio start/duration must be numeric") from exc
    if not np.isfinite(start) or start < 0:
        raise ValueError("Audio start must be finite and nonnegative")
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("Audio duration must be finite and positive")
    try:
        rate, values = wavfile.read(path, mmap=True)
    except ValueError as exc:
        # Packed 24-bit PCM cannot be memory mapped by scipy.
        if "mmap" not in str(exc).lower():
            raise
        rate, values = wavfile.read(path)
    begin = int(start * rate)
    end = len(values) if duration is None else begin + int(duration * rate)
    if begin >= len(values) or end > len(values):
        raise ValueError(
            f"Requested audio window exceeds available samples: {path} (start={begin}, end={end}, available={len(values)})"
        )
    # Own the cropped array so the returned data does not retain the WAV mapping.
    x = values[begin:end].astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < 16:
        raise ValueError(f"Audio segment too short: {path}")
    return rate, x - x.mean()
