from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .normalization import MinMax01


@plt.rc_context({
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})
def plot_feature_groups(scores, output: Path, dpi: int, *, means=None, threshold=None):
    """Write separate score and wear plots for each MIC domain or ACC axis."""
    frequency = pd.to_numeric(scores.frequency_low_hz, errors="coerce").notna()
    groups = {
        "mic_time": scores.domain.eq("mic_time"),
        "mic_frequency": scores.domain.eq("mic_frequency"),
        **{
            f"acc_{axis}_{domain}": scores.sensor.eq("acc") & scores.axis.eq(axis) & mask
            for axis in "xyz"
            for domain, mask in (("time", ~frequency), ("frequency", frequency))
        },
    }

    def label(row):
        if pd.notna(row.frequency_low_hz):
            low, high = float(row.frequency_low_hz), float(row.frequency_high_hz)
            return f"{low:g} Hz" if low == high else f"{low:g}–{high:g} Hz"
        feature = row.feature_name.split("::", 1)[-1]
        return feature.removeprefix("mic_").removeprefix(f"{row.axis}_")

    method = scores.method.iloc[0]
    method_title = "R²" if method == "r2" else "Fisher J"
    figsize = (1920 / dpi, 1080 / dpi)
    for domain, mask in (("time", ~frequency), ("frequency", frequency)):
        acc = scores.loc[scores.sensor.eq("acc") & mask].sort_values("axis", kind="stable")
        if not acc.empty:
            plot_score_overview(
                [f"{row.axis.upper()} {label(row)}" for row in acc.itertuples()],
                acc.score.to_numpy(), np.flatnonzero(acc.selected), method,
                output / f"acc_{domain}_overview.png", dpi,
                threshold=threshold, annotate_peak=False, show_feature_labels=len(acc) <= 24,
                title=f"ACC X / Y / Z · {domain.title()} · {method_title} scores",
                feature_label_size=14, figsize=figsize,
            )
    for name, mask in groups.items():
        group = scores.loc[mask]
        if group.empty:
            continue
        group_title = name.replace("_", " ").upper()
        selected = group.loc[group.selected].sort_values("score")
        plot_score_overview(
            [label(row) for row in group.itertuples()], group.score.to_numpy(),
            np.flatnonzero(group.selected), method, output / f"{name}_score_overview.png", dpi,
            threshold=threshold, show_feature_labels=len(group) <= 12, annotate_peak=False,
            title=f"{group_title} · {method_title} scores", feature_label_size=15,
            figsize=figsize,
        )
        if selected.empty:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        labels = [label(row) for row in selected.itertuples()]
        ax.barh(labels, selected.score)
        ax.set(title=f"{group_title} · {method_title} selected features", xlabel=method_title)
        fig.tight_layout()
        fig.savefig(output / f"{name}_selected_top.png", dpi=dpi)
        plt.close(fig)
        if means is None:
            continue
        names = selected.feature_name.tolist()
        wear = MinMax01().fit_transform(means[["mean_wear"]].to_numpy())[:, 0]
        values = MinMax01().fit_transform(means[names].to_numpy())
        order = np.argsort(means.machining_progress.to_numpy())
        for kind in ("trend", "scatter"):
            fig, ax = plt.subplots(figsize=figsize)
            for index, feature in enumerate(labels):
                y = values[:, index]
                if kind == "trend":
                    ax.plot(means.machining_progress.to_numpy()[order], y[order], "-o", label=feature)
                else:
                    points = ax.scatter(wear, y, s=20, label=feature)
                    if np.ptp(wear) > 0:
                        line = np.linspace(0, 1, 100)
                        ax.plot(line, np.polyval(np.polyfit(wear, y, 1), line), "--",
                                color=points.get_facecolor()[0])
            if kind == "trend":
                ax.plot(means.machining_progress.to_numpy()[order], wear[order], "k-*", label="Tool wear")
            ax.set(
                title=f"{group_title} · {'Machining trend' if kind == 'trend' else 'Wear regression'}",
                xlabel="Machining position" if kind == "trend" else "Normalized wear",
                ylabel="Normalized feature / wear" if kind == "trend" else "Normalized feature",
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"{name}_{kind}.png", dpi=dpi)
            plt.close(fig)


def plot_score_overview(
    names, scores, selected, method: str, output: Path, dpi: int, *, threshold,
    show_feature_labels: bool = True,
    annotate_peak: bool = True,
    title: str | None = None,
    feature_label_size: int = 7,
    figsize: tuple[float, float] = (14, 7),
) -> None:
    selected_set = set(map(int, selected))
    colors = ["#e58b27" if i in selected_set else "#2e6ba5" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(scores)), scores, color=colors)
    ax.set(title=title or f"{method.upper()} feature overview", xlabel="Feature", ylabel=method.upper())
    if show_feature_labels:
        ax.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=feature_label_size)
    else:
        ax.set_xlabel("Feature index within group")
    if method.lower() == "r2" and threshold is not None:
        ax.axhline(threshold, color="crimson", linestyle="--", label=f"R² = {threshold:g}")
        ax.legend()
    if annotate_peak:
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


def wear_position_means(wear, values, names, record_ids, tool_ids):
    wear, values, positions, counts = analysis_position_means(wear, values, record_ids, tool_ids)
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
    means = pd.DataFrame(values, columns=names)
    means.insert(0, "machining_progress", progression)
    means.insert(0, "mean_wear", wear)
    means.insert(0, "experiment_count", counts)
    means.insert(0, "position", positions)
    return means.sort_values("machining_progress"), progression_label


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
    means, progression_label = wear_position_means(wear, values, names, record_ids, tool_ids)
    positions = means.position.tolist()
    progression = means.machining_progress.to_numpy()
    x = MinMax01().fit_transform(means[["mean_wear"]].to_numpy())[:, 0]
    y = MinMax01().fit_transform(means[list(names)].to_numpy())
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
    means.to_csv(output.with_suffix(".means.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"feature": names, "selection_r2": scores, "position_mean_r2": plot_scores}
    ).to_csv(output.with_suffix(".scores.csv"), index=False, encoding="utf-8-sig")
