from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def split_tools(frame: pd.DataFrame, seed: int, options) -> pd.DataFrame:
    tools = np.array(sorted(frame["ToolID"].unique()))
    if len(tools) < options["minimum_total_tools"]:
        raise ValueError("Too few physical tools for evaluation.minimum_total_tools")
    rng = np.random.default_rng(seed)
    rng.shuffle(tools)
    test_n = max(options["minimum_test_tools"], math.ceil(len(tools) * options["test_fraction"]))
    remaining = len(tools) - test_n
    val_n = max(
        options["minimum_validation_tools"], math.ceil(remaining * options["validation_fraction"])
    )
    if test_n + val_n >= len(tools):
        raise ValueError("evaluation split settings leave no training tools")
    assignment = {tool: "test" for tool in tools[:test_n]}
    assignment.update({tool: "val" for tool in tools[test_n : test_n + val_n]})
    assignment.update({tool: "train" for tool in tools[test_n + val_n :]})
    out = frame.copy()
    out["Split"] = out["ToolID"].map(assignment)
    return out


def training_tertiles(frame: pd.DataFrame) -> tuple[float, float]:
    wear = frame.loc[frame["Split"] == "train", "WearValue"]
    if wear.empty or not np.isfinite(wear).all() or (wear < 0).any():
        raise ValueError("Training WearValue must be nonempty, finite and nonnegative")
    return float(wear.quantile(1 / 3)), float(wear.quantile(2 / 3))


def apply_labels(frame: pd.DataFrame, thresholds: tuple[float, float]) -> pd.DataFrame:
    low, high = thresholds
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Label thresholds must be finite and ordered")
    if not np.isfinite(frame.WearValue).all() or (frame.WearValue < 0).any():
        raise ValueError("WearValue must be finite and nonnegative")
    out = frame.copy()
    out["Label"] = np.select(
        [out.WearValue <= low, out.WearValue <= high], [0, 1], default=2
    ).astype(int)
    return out


def validate_dataset(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    for key in ("stage_ids", "exclude_stages"):
        stages = dataset.get(key)
        if stages is not None and (
            not isinstance(stages, list)
            or (key == "stage_ids" and not stages)
            or any(type(stage) is not int or stage < 1 for stage in stages)
            or len(set(stages)) != len(stages)
        ):
            raise ValueError(f"dataset.{key} must contain distinct positive integer stages")
    labeling = dataset["labeling"]
    if not isinstance(labeling, dict):
        raise ValueError("dataset.labeling must be a mapping")
    method = labeling["method"]
    if method not in ("training_tertiles", "stage_groups"):
        raise ValueError("dataset.labeling.method must be training_tertiles or stage_groups")
    if method != "stage_groups":
        if "groups" in labeling:
            raise ValueError("labeling.groups requires method=stage_groups")
        return
    groups = labeling.get("groups")
    if not isinstance(groups, list) or len(groups) not in (2, 3):
        raise ValueError("Stage groups require two or three groups")
    seen = set()
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("Each stage group must be nonempty")
        for stage in group:
            if type(stage) is not int or stage < 1 or stage in seen:
                raise ValueError("Stage groups require distinct, nonoverlapping positive integers")
            seen.add(stage)


def filter_stage_records(frame, dataset):
    validate_dataset(dataset)
    out = frame.copy()
    knives = dataset.get("knife_ids")
    if knives is not None:
        if not isinstance(knives, list) or not knives:
            raise ValueError("dataset.knife_ids must be a nonempty list")
        chosen = _resolve_tools(out.ToolID.unique(), knives)
        out = out[out.ToolID.isin(chosen)].copy()
    labeling = dataset["labeling"]
    grouped = labeling["method"] == "stage_groups"
    if grouped or dataset.get("stage_ids") is not None or dataset.get("exclude_stages"):
        valid = out.RecordID.astype(str).str.fullmatch(r"M[0-9]+")
        if not valid.all():
            raise ValueError("Stage filtering requires turning RecordID values M001, M002, ...")
        out["Stage"] = out.RecordID.str[1:].astype(int)
        if (out.Stage < 1).any():
            raise ValueError("Stage IDs must be positive")
        if dataset.get("stage_ids") is not None:
            out = out[out.Stage.isin(dataset["stage_ids"])].copy()
        out = out[~out.Stage.isin(dataset.get("exclude_stages") or [])].copy()
        if grouped:
            mapping = {
                stage: label for label, group in enumerate(labeling["groups"]) for stage in group
            }
            out = out[out.Stage.isin(mapping)].copy()
            out["Label"] = out.Stage.map(mapping).astype(int)
    if out.empty:
        raise ValueError("No records remain after dataset filters")
    return out


def _resolve_tools(actual, requested):
    actual = [str(tool) for tool in actual]
    if not isinstance(requested, list) or not requested:
        raise ValueError("Knife IDs must be a nonempty list")
    resolved = []
    for value in requested:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Knife IDs must be ToolID strings or positive integers")
        text = str(value)
        if isinstance(value, str) and text in actual:
            matches = [text]
        elif type(value) is int and value > 0:
            matches = [
                tool
                for tool in actual
                if (match := re.fullmatch(r"(?:T|knife)?0*([0-9]+)", tool, re.IGNORECASE))
                and int(match[1]) == value
            ]
        else:
            matches = []
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous knife ID {value!r}")
        if matches[0] in resolved:
            raise ValueError("Knife IDs must be distinct")
        resolved.append(matches[0])
    return resolved


def assign_splits(frame, evaluation, seed):
    strategy = evaluation["strategy"]
    if strategy == "random_tools":
        if any(key in evaluation for key in ("test_knife", "val_knife", "val_knives")):
            raise ValueError("Fixed knife settings require strategy=fixed_knife")
        return split_tools(frame, seed, evaluation)
    if strategy not in ("fixed_knife", "fixed_test_random_val"):
        raise ValueError("evaluation.strategy must be random_tools, fixed_knife or fixed_test_random_val")
    if "val_knife" in evaluation and "val_knives" in evaluation:
        raise ValueError("Specify only one of val_knife and val_knives")
    actual = frame.ToolID.unique()
    test = _resolve_tools(actual, [evaluation.get("test_knife")])
    if strategy == "fixed_test_random_val":
        pool = sorted(set(actual) - set(test))
        count = evaluation["validation_tool_count"]
        if type(count) is not int or not 0 < count < len(pool):
            raise ValueError("validation_tool_count must leave at least one training tool")
        validation = sorted(np.random.default_rng(seed).choice(pool, count, replace=False).tolist())
    else:
        validation = _resolve_tools(actual, evaluation["val_knives"])
    if set(test) & set(validation):
        raise ValueError("Validation and test knives must be distinct")
    out = frame.copy()
    out["Split"] = "train"
    out.loc[out.ToolID.isin(validation), "Split"] = "val"
    out.loc[out.ToolID.isin(test), "Split"] = "test"
    if set(out.Split) != {"train", "val", "test"}:
        raise ValueError("Fixed knife split requires nonempty train, val and test")
    return out


def label_training_records(frame, dataset):
    if dataset["labeling"]["method"] == "stage_groups":
        return frame, None
    thresholds = training_tertiles(frame.drop_duplicates(["ToolID", "RecordID"]))
    return apply_labels(frame, thresholds), thresholds
