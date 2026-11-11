"""Safe, validated artifact basenames; metadata/state paths remain stable."""

import re
from pathlib import Path
from string import Formatter

FEATURE_FILES = {
    "features": "features.h5",
    "catalog": "feature_catalog.csv",
    "sample_manifest": "sample_manifest.csv",
    "sample_counts": "sample_counts.csv",
}
TRAIN_FILES = {
    "model": "model.pt",
    "history": "model.history.csv",
    "model_metadata": "model.json",
    "curves": "model.curves.png",
    "selection": "selected_features.json",
    "scores": "feature_scores.csv",
    "top": "selected_top.csv",
    "quota": "quota_audit.csv",
    "seed_metrics": "seed_metrics.csv",
    "score_plot": "score_overview.png",
    "top_plot": "selected_top.png",
    "r2_overlay": "r2_overlay.png",
    "r2_scatter": "r2_{feature_index:04d}.png",
}
EVAL_FILES = {
    "metrics": "metrics.json",
    "deployment": "deployment.json",
    "predictions": "predictions.csv",
    "sample_counts": "evaluated_sample_counts.csv",
    "per_tool": "per_tool_metrics.csv",
    "confusion": "confusion_matrix.csv",
    "confusion_normalized": "confusion_matrix_normalized.csv",
    "confusion_plot": "confusion_matrix.png",
}
REPORT_FILES = {
    "summary": "summary.csv",
    "ranking": "ranking_by_validation.csv",
    "comparison": "{condition}_comparison.png",
    "readme": "README.md",
}
RESERVED = {
    "state.json",
    "feature_set.json",
    "source_manifest.json",
    "artifacts.json",
    "config_resolved.yaml",
    "output_manifest.csv",
    "failure.json",
    "split.h5",
}


def resolve_names(defaults, templates, context):
    templates = templates or {}
    if not isinstance(templates, dict) or set(templates) - set(defaults):
        raise ValueError("Unknown output filename template keys")
    names = {}
    for key, default in defaults.items():
        template = templates.get(key, default)
        if not isinstance(template, str):
            raise ValueError("Filename templates must be strings")
        fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
        if key == "r2_scatter" and "feature_index" not in fields:
            raise ValueError("r2_scatter requires {feature_index} to avoid overwriting features")
        if key == "comparison" and "condition" not in fields:
            raise ValueError("comparison requires {condition} to avoid overwriting conditions")
        if fields - set(context):
            raise ValueError(f"Unknown template fields: {sorted(fields - set(context))}")
        name = template.format(**context)
        stem = name.split(".")[0].upper()
        if (
            not name
            or len(name) > 200
            or name.endswith((" ", "."))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or stem
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }
            or name.lower() in RESERVED
            or Path(name).suffix.lower() != Path(default).suffix.lower()
        ):
            raise ValueError(f"Unsafe filename or incorrect extension: {name}")
        names[key] = name
    if len({n.casefold() for n in names.values()}) != len(names):
        raise ValueError("Output filenames collide")
    return names
