from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .normalization import MinMax01


def plot_score_overview(
    names, scores, selected, method: str, output: Path, dpi: int, *, threshold
) -> None:
    selected_set = set(map(int, selected))
    colors = ["#e58b27" if i in selected_set else "#2e6ba5" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(np.arange(len(scores)), scores, color=colors)
    ax.set(title=f"{method.upper()} feature overview", xlabel="Feature", ylabel=method.upper())
    ax.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=7)
    if method.lower() == "r2" and threshold is not None:
        ax.axhline(threshold, color="crimson", linestyle="--", label=f"R² = {threshold:g}")
        ax.legend()
    peak = int(np.argmax(scores))
    ax.annotate(
        f"Peak: {names[peak]} | {method.upper()}={scores[peak]:.3f}",
        xy=(peak, scores[peak]),
        xytext=(0.68, 0.95),
        textcoords="axes fraction",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_r2_scatter(wear, feature, feature_name: str, score: float, output: Path, dpi: int) -> None:
    x = MinMax01().fit_transform(np.asarray(wear).reshape(-1, 1))[:, 0]
    y = MinMax01().fit_transform(np.asarray(feature).reshape(-1, 1))[:, 0]
    coefficients = np.polyfit(x, y, 1)
    line_x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=25, alpha=0.75)
    ax.plot(line_x, np.polyval(coefficients, line_x), "--", color="crimson")
    ax.set(
        title=f"{feature_name} | R²={score:.3f}",
        xlabel="Normalized wear",
        ylabel="Normalized feature",
    )
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def analysis_position_means(wear, values, record_ids, tool_ids):
    """Cap each stage at its feature mean, then average experiments per position."""
    values = np.asarray(values, dtype=float)
    records = list(map(str, record_ids))
    keys = pd.DataFrame({"tool": tool_ids, "position": records})
    if keys.duplicated().any():
        raise ValueError("Expected one observation per tool and record")
    stages = [
        match.group(1) if (match := re.fullmatch(r"(S\d+)_H\d+", record)) else record
        for record in records
    ]
    frame = pd.DataFrame(values)
    caps = frame.groupby(stages, sort=False).transform("mean")
    clipped = frame.where(frame <= caps, caps)
    means = clipped.groupby(records, sort=False).mean()
    mean_wear = pd.Series(wear).groupby(records, sort=False).mean()
    counts = keys.groupby("position", sort=False).tool.nunique()
    return mean_wear.to_numpy(), means.to_numpy(), means.index.tolist(), counts.to_numpy()


def plot_r2_overlay(
    wear,
    values,
    names,
    scores,
    output: Path,
    dpi: int,
    *,
    record_ids,
    tool_ids,
    trend_output: Path | None = None,
) -> None:
    """Plot progression trends and wear regressions from the same position means."""
    wear, values, positions, counts = analysis_position_means(wear, values, record_ids, tool_ids)
    x = MinMax01().fit_transform(np.asarray(wear).reshape(-1, 1))[:, 0]
    y = MinMax01().fit_transform(np.asarray(values))
    if all(re.fullmatch(r"S[1-3]_H0?[1-5]", position) for position in positions):
        # analysis/curvefit.m uses three stages of five holes, numbered 1..15.
        progression = np.array(
            [
                (int(re.fullmatch(r"S(\d+)_H(\d+)", position).group(1)) - 1) * 5
                + int(re.fullmatch(r"S(\d+)_H(\d+)", position).group(2))
                for position in positions
            ]
        )
        progression_label = "Machined hole count"
    elif all(re.fullmatch(r"M\d+", position) for position in positions):
        progression = np.array([int(position[1:]) for position in positions])
        progression_label = "Machining pass count"
    else:
        progression = np.arange(len(positions))
        progression_label = "Recorded machining position"
    order = np.argsort(progression)
    line_x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(names)))
    plot_scores = []
    for index, (name, color) in enumerate(zip(names, colors)):
        feature = y[:, index]
        coefficients = np.polyfit(x, feature, 1) if np.ptp(x) > 0 else [0, feature.mean()]
        residual = np.sum((feature - np.polyval(coefficients, x)) ** 2)
        total = np.sum((feature - feature.mean()) ** 2)
        score = 1 - residual / total if total > 0 else float("nan")
        plot_scores.append(score)
        fitted = np.polyval(coefficients, line_x)
        ax.scatter(x, feature, s=22, alpha=0.6, color=color)
        ax.plot(line_x, fitted, "--", color=color, label=f"{name} | R²={score:.3f}")
    ax.set(
        title="Stage-capped position means across training experiments",
        xlabel="Normalized wear",
        ylabel="Normalized feature",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )
    ax.grid(alpha=0.25)
    if len(names):
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    else:
        ax.text(0.5, 0.5, "No features with R² > 0.5", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    trend_output = trend_output or output.with_name(f"{output.stem}.trend.png")
    trend, trend_ax = plt.subplots(figsize=(12.8, 7.2))
    for index, (name, color) in enumerate(zip(names, colors)):
        trend_ax.plot(progression[order], y[order, index], "-o", color=color, label=name)
    trend_ax.plot(progression[order], x[order], "k-*", linewidth=2, label="Tool wear")
    trend_ax.set(
        title="Feature and wear trends across machining positions",
        xlabel=progression_label,
        ylabel="Normalized feature / wear",
        ylim=(-0.03, 1.03),
    )
    if progression_label == "Recorded machining position":
        trend_ax.set_xticks(progression[order], np.asarray(positions)[order], rotation=45)
    trend_ax.grid(alpha=0.25)
    if not len(names):
        trend_ax.text(
            0.5, 0.5, "No features with R² > 0.5", transform=trend_ax.transAxes, ha="center"
        )
    trend_ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    trend.tight_layout()
    trend_output.parent.mkdir(parents=True, exist_ok=True)
    trend.savefig(trend_output, dpi=dpi)
    plt.close(trend)
    means = pd.DataFrame(values, columns=names)
    means.insert(0, "machining_progress", progression)
    means.insert(0, "mean_wear", wear)
    means.insert(0, "experiment_count", counts)
    means.insert(0, "position", positions)
    means.iloc[order].to_csv(output.with_suffix(".means.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"feature": names, "selection_r2": scores, "position_mean_r2": plot_scores}
    ).to_csv(output.with_suffix(".scores.csv"), index=False, encoding="utf-8-sig")
