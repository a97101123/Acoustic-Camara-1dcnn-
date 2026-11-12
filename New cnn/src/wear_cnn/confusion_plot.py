"""Count matrix with row (recall) and column (precision) summaries."""

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
import numpy as np


@lru_cache(maxsize=16)
def _label_font(names):
    required = {ord(char) for name in names for char in name if not char.isascii()}
    if not required:
        return None
    for font in font_manager.fontManager.ttflist:
        if "lastresort" in Path(font.fname).stem.lower():
            continue
        if required.issubset(FT2Font(font.fname).get_charmap()):
            return font_manager.FontProperties(fname=font.fname)
    return None


def plot_confusion(matrix, output: str | Path, class_names=None):
    matrix = np.asarray(matrix, dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) not in (2, 3):
        raise ValueError("Confusion matrix must be square with 2 or 3 classes")
    count_classes = len(matrix)
    names = (
        list(class_names)
        if class_names is not None
        else (["Initial", "Normal", "Severe"] if count_classes == 3 else ["Low", "High"])
    )
    if len(names) != count_classes:
        raise ValueError("Confusion class names must match the matrix")
    diagonal = np.diag(matrix)
    rows = matrix.sum(axis=1)
    columns = matrix.sum(axis=0)
    recall = np.divide(diagonal, rows, out=np.zeros(count_classes, float), where=rows > 0)
    precision = np.divide(diagonal, columns, out=np.zeros(count_classes, float), where=columns > 0)
    fig = plt.figure(figsize=(9, 7))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[3, 1.5], height_ratios=[3, 1.2], hspace=0.4, wspace=0.4
    )
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for i in range(count_classes):
        for j in range(count_classes):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    ax.set(
        xticks=range(count_classes),
        yticks=range(count_classes),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix (sample counts)",
    )
    row_ax = fig.add_subplot(grid[0, 1])
    row_ax.imshow(np.c_[recall, 1 - recall], cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(count_classes):
        for j, value in enumerate((recall[i], 1 - recall[i])):
            row_ax.text(j, i, f"{value:.1%}" if rows[i] else "N/A", ha="center", va="center")
    row_ax.set(
        xticks=[0, 1],
        yticks=range(count_classes),
        xticklabels=["Correct", "Incorrect"],
        yticklabels=[],
        title="Row summary (recall)",
    )
    col_ax = fig.add_subplot(grid[1, 0])
    col_ax.imshow(
        np.vstack([precision, 1 - precision]), cmap="Greens", vmin=0, vmax=1, aspect="auto"
    )
    for i in range(2):
        for j in range(count_classes):
            value = precision[j] if i == 0 else 1 - precision[j]
            col_ax.text(j, i, f"{value:.1%}" if columns[j] else "N/A", ha="center", va="center")
    col_ax.set(
        xticks=range(count_classes),
        yticks=[0, 1],
        xticklabels=names,
        yticklabels=["Correct", "Incorrect"],
        title="Column summary (precision)",
    )
    label_font = _label_font(tuple(names))
    if label_font is not None:
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels(), *col_ax.get_xticklabels()):
            label.set_fontproperties(label_font)
    summary_ax = fig.add_subplot(grid[1, 1])
    summary_ax.axis("off")
    count = int(matrix.sum())
    accuracy = f"{diagonal.sum() / count:.1%}" if count else "N/A"
    summary_ax.text(0.5, 0.5, f"Samples: {count}\nAccuracy: {accuracy}", ha="center", va="center")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion(matrix, output: str | Path, make_plots=True, filenames=None, class_names=None):
    output = Path(output)
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    names = filenames or {
        "confusion": "confusion_matrix.csv",
        "confusion_normalized": "confusion_matrix_normalized.csv",
        "confusion_plot": "confusion_matrix.png",
    }
    np.savetxt(output / names["confusion"], matrix, fmt="%d", delimiter=",")
    np.savetxt(output / names["confusion_normalized"], normalized, fmt="%.6f", delimiter=",")
    if make_plots:
        plot_confusion(matrix, output / names["confusion_plot"], class_names=class_names)
