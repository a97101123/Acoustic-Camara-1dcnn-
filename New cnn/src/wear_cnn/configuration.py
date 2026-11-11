"""Required experiment settings. This module never supplies parameter values."""

import math


def fields(value, required, path, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    missing = set(required) - set(value)
    if missing:
        raise ValueError(f"Missing config field: {path}.{sorted(missing)[0]}")
    unknown = set(value) - set(required) - set(optional)
    if unknown:
        raise ValueError(f"Unknown config field: {path}.{sorted(unknown)[0]}")
    return value


def positive(value, path, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{path} must be positive and finite")
    if integer and type(value) is not int:
        raise ValueError(f"{path} must be an integer")


def model_settings(value):
    fields(
        value,
        (
            "filters",
            "kernel_size",
            "dropout",
            "normalization",
            "norm_groups",
            "normalization_epsilon",
        ),
        "model",
    )
    if not isinstance(value["filters"], list) or len(value["filters"]) != 2:
        raise ValueError("model.filters must contain two channel counts")
    for count in value["filters"]:
        positive(count, "model.filters", True)
    positive(value["kernel_size"], "model.kernel_size", True)
    positive(value["norm_groups"], "model.norm_groups", True)
    positive(value["normalization_epsilon"], "model.normalization_epsilon")
    if value["normalization"] not in ("group", "layer", "none"):
        raise ValueError("model.normalization must be group, layer or none")
    if any(count % value["norm_groups"] for count in value["filters"]):
        raise ValueError("model.norm_groups must divide each model.filters value")
    if (
        isinstance(value["dropout"], bool)
        or not isinstance(value["dropout"], (int, float))
        or not 0 <= value["dropout"] < 1
    ):
        raise ValueError("model.dropout must be in [0, 1)")
    return value


def training_settings(value):
    fields(
        value,
        (
            "seed_start",
            "runs",
            "epochs",
            "batch_size",
            "validation_batch_size",
            "checkpoint_every",
            "severe_class_index",
            "learning_rate",
            "preload_device",
            "optimizer",
        ),
        "training",
        ("model", "inference_batch_size", "inference", "matlab_adam", "early_stop_patience", "label_smoothing", "lr_plateau_patience"),
    )
    for key in ("runs", "epochs", "batch_size", "validation_batch_size", "checkpoint_every"):
        positive(value[key], f"training.{key}", True)
    positive(value["learning_rate"], "training.learning_rate")
    if type(value["preload_device"]) is not bool:
        raise ValueError("training.preload_device must be a boolean")
    if type(value.get("matlab_adam", False)) is not bool:
        raise ValueError("training.matlab_adam must be a boolean")
    if "early_stop_patience" in value:
        positive(value["early_stop_patience"], "training.early_stop_patience", True)
    smoothing = value.get("label_smoothing", 0.0)
    if type(smoothing) not in (int, float) or not 0 <= smoothing < 1:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    if "lr_plateau_patience" in value:
        positive(value["lr_plateau_patience"], "training.lr_plateau_patience", True)
    opt = fields(
        value["optimizer"],
        ("name", "betas", "epsilon", "weight_decay", "amsgrad"),
        "training.optimizer",
    )
    if opt["name"] != "adam":
        raise ValueError("training.optimizer.name must be adam")
    if (
        not isinstance(opt["betas"], list)
        or len(opt["betas"]) != 2
        or any(type(v) not in (int, float) or not 0 <= v < 1 for v in opt["betas"])
    ):
        raise ValueError("training.optimizer.betas must contain two values in [0, 1)")
    positive(opt["epsilon"], "training.optimizer.epsilon")
    if (
        type(opt["weight_decay"]) not in (int, float)
        or not math.isfinite(opt["weight_decay"])
        or opt["weight_decay"] < 0
    ):
        raise ValueError("training.optimizer.weight_decay must be finite and nonnegative")
    if type(opt["amsgrad"]) is not bool:
        raise ValueError("training.optimizer.amsgrad must be a boolean")
    return value


def validate_training_config(cfg):
    fields(
        cfg,
        (
            "output",
            "device",
            "cpu_threads",
            "split_seed",
            "dataset",
            "evaluation",
            "training",
            "selection",
            "model",
            "preprocessing",
            "io",
            "cache_features",
            "conditions",
            "nffts",
        ),
        "config",
        ("features",),
    )
    fields(
        cfg["output"],
        ("root", "name", "plot_dpi"),
        "output",
        ("filenames", "evaluation_filenames", "report_filenames"),
    )
    positive(cfg["output"]["plot_dpi"], "output.plot_dpi", True)
    fields(
        cfg["dataset"],
        ("knife_ids", "stage_ids", "exclude_stages", "labeling", "class_names"),
        "dataset",
    )
    method = (
        cfg["dataset"]["labeling"]["method"]
        if isinstance(cfg["dataset"]["labeling"], dict) and "method" in cfg["dataset"]["labeling"]
        else None
    )
    fields(
        cfg["dataset"]["labeling"],
        ("method", "groups") if method == "stage_groups" else ("method",),
        "dataset.labeling",
    )
    evaluation = cfg["evaluation"]
    required = ("strategy", "prediction_rule", "batch_size", "inference_batch_size", "inference")
    if isinstance(evaluation, dict) and evaluation.get("strategy") == "fixed_knife":
        required += ("test_knife", "val_knives")
    elif isinstance(evaluation, dict) and evaluation.get("strategy") == "fixed_test_random_val":
        required += ("test_knife", "validation_tool_count")
    else:
        required += (
            "test_fraction",
            "validation_fraction",
            "minimum_test_tools",
            "minimum_validation_tools",
            "minimum_total_tools",
        )
    fields(evaluation, required, "evaluation")
    fields(evaluation["inference"], ("start_seconds", "duration_seconds"), "evaluation.inference")
    crop = evaluation["inference"]
    if (
        type(crop["start_seconds"]) not in (int, float)
        or not math.isfinite(crop["start_seconds"])
        or crop["start_seconds"] < 0
    ):
        raise ValueError("evaluation.inference.start_seconds must be finite and nonnegative")
    if crop["duration_seconds"] is not None:
        positive(crop["duration_seconds"], "evaluation.inference.duration_seconds")
    for key in ("batch_size", "inference_batch_size"):
        positive(evaluation[key], f"evaluation.{key}", True)
    if evaluation["strategy"] == "random_tools":
        for key in ("test_fraction", "validation_fraction"):
            if type(evaluation[key]) not in (int, float) or not 0 < evaluation[key] < 1:
                raise ValueError(f"evaluation.{key} must be in (0, 1)")
        for key in ("minimum_test_tools", "minimum_validation_tools", "minimum_total_tools"):
            positive(evaluation[key], f"evaluation.{key}", True)
    if evaluation["strategy"] == "fixed_test_random_val":
        positive(evaluation["validation_tool_count"], "evaluation.validation_tool_count", True)
    training_settings(cfg["training"])
    seed = cfg["training"]["seed_start"]
    if type(seed) is not int or seed < 0 or seed + cfg["training"]["runs"] - 1 >= 2**32:
        raise ValueError("training.seed_start/runs must stay within [0, 2**32)")
    if type(cfg["split_seed"]) is not int or not 0 <= cfg["split_seed"] < 2**32:
        raise ValueError("split_seed must be an integer within [0, 2**32)")
    if evaluation["strategy"] not in ("random_tools", "fixed_knife", "fixed_test_random_val"):
        raise ValueError("evaluation.strategy must be random_tools, fixed_knife or fixed_test_random_val")
    from .classification import classification_from_config
    from .splitting import validate_dataset

    validate_dataset(cfg["dataset"])
    classification_from_config(cfg)
    for key in ("conditions", "nffts"):
        values = cfg[key]
        if values is not None and (
            not isinstance(values, list) or not values or len(set(values)) != len(values)
        ):
            raise ValueError(f"{key} must be null or a nonempty unique list")
    if cfg["nffts"] is not None and any(type(n) is not int or n < 2 or n % 2 for n in cfg["nffts"]):
        raise ValueError("nffts must contain even positive integers >= 2")
    if cfg["conditions"] is not None and any(
        not isinstance(c, str) or not c for c in cfg["conditions"]
    ):
        raise ValueError("conditions must contain nonempty strings")
    model_settings(cfg["model"])
    fields(cfg["preprocessing"], ("scaler", "scale_epsilon"), "preprocessing")
    if cfg["preprocessing"]["scaler"] != "median_iqr":
        raise ValueError("preprocessing.scaler must be median_iqr")
    positive(cfg["preprocessing"]["scale_epsilon"], "preprocessing.scale_epsilon")
    positive(cfg["cpu_threads"], "cpu_threads", True)
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if type(cfg["cache_features"]) is not bool:
        raise ValueError("cache_features must be a boolean")
    selection = fields(cfg["selection"], ("methods", "j", "r2"), "selection")
    if (
        not isinstance(selection["methods"], list)
        or not selection["methods"]
        or len(set(selection["methods"])) != len(selection["methods"])
        or set(selection["methods"]) - {"j", "r2"}
    ):
        raise ValueError("selection.methods must be a unique nonempty subset of j/r2")
    fields(
        selection["j"],
        ("acc_axis_quota", "mic_domain_quota", "require_equal_sensor_totals"),
        "selection.j",
    )
    if not isinstance(selection["r2"], dict):
        raise ValueError("selection.r2 must be a mapping")
    r2_mode = selection["r2"].get("mode", "threshold")
    if r2_mode not in ("threshold", "top"):
        raise ValueError("selection.r2.mode must be threshold or top")
    if r2_mode == "top":
        fields(
            selection["r2"],
            (
                "mode",
                "score_backend",
                "report_thresholds",
                "acc_axis_quota",
                "mic_domain_quota",
                "require_equal_sensor_totals",
            ),
            "selection.r2",
        )
    else:
        fields(
            selection["r2"],
            ("threshold", "score_backend", "report_thresholds"),
            "selection.r2",
            ("mode",),
        )
        if (
            type(selection["r2"]["threshold"]) not in (int, float)
            or not 0 <= selection["r2"]["threshold"] <= 1
        ):
            raise ValueError("selection.r2.threshold must be in [0, 1]")
    if selection["r2"]["score_backend"] not in ("legacy", "matrix"):
        raise ValueError("selection.r2.score_backend must be legacy or matrix")
    thresholds = selection["r2"]["report_thresholds"]
    if (
        not isinstance(thresholds, list)
        or not thresholds
        or any(type(t) not in (int, float) or not 0 <= t <= 1 for t in thresholds)
    ):
        raise ValueError("selection.r2.report_thresholds must contain values in [0, 1]")
    for method in ["j", "r2"] if r2_mode == "top" else ["j"]:
        policy = selection[method]
        fields(policy["acc_axis_quota"], ("x", "y", "z"), f"selection.{method}.acc_axis_quota")
        fields(
            policy["mic_domain_quota"],
            ("mic_joint",) if "mic_joint" in policy["mic_domain_quota"] else ("mic_frequency", "mic_time"),
            f"selection.{method}.mic_domain_quota",
        )
        for section in ("acc_axis_quota", "mic_domain_quota"):
            if any(type(q) is not int or q < 0 for q in policy[section].values()):
                raise ValueError(f"selection.{method}.{section} must contain nonnegative integers")
        if not any(
            q
            for section in ("acc_axis_quota", "mic_domain_quota")
            for q in policy[section].values()
        ):
            raise ValueError(f"selection.{method} must select at least one feature")
        if type(policy["require_equal_sensor_totals"]) is not bool:
            raise ValueError(f"selection.{method}.require_equal_sensor_totals must be a boolean")
    from .features.io import io_options

    io_options(cfg["io"])
    return cfg
