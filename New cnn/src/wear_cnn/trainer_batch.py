"""Train from completed Feature Sets; never read or rebuild source WAVs."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .artifact_validation import valid_checkpoint
from .classification import classification_from_config, classification_from_metadata
from .domain_selection import select_domains
from .evaluation import evaluate_hdf5
from .features.plots import plot_feature_groups, wear_position_means
from .features.r2_report import write_r2_threshold_report
from .features.reader import FeatureReader
from .features.selection import fisher_j_scores, r2_scores
from .filename_templates import EVAL_FILES, REPORT_FILES, TRAIN_FILES, resolve_names
from .run_io import atomic_json, load_config, new_run, output_manifest, resolve, sha256
from .splitting import assign_splits, filter_stage_records, label_training_records, validate_dataset
from .training import PIPELINE_REVISION, prepare_training_data, train_hdf5, training_identity


def training_seeds(options):
    start, runs = options["seed_start"], options["runs"]
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (start, runs))
        or runs < 1
        or start < 0
        or start + runs - 1 >= 2**32
    ):
        raise ValueError("seed_start/runs must define positive runs within [0, 2**32)")
    return list(range(start, start + runs))


def split_store(feature, target, seed, identity, config):
    """Only tiny identity/label arrays are stored; feature values remain external VDS."""
    dataset = config["dataset"]
    evaluation = config["evaluation"]
    validate_dataset(dataset)
    spec = classification_from_config(config)
    target = Path(target)
    temporary = target.with_name(target.name + ".tmp")
    with h5py.File(feature, "r") as source, h5py.File(temporary, "w") as out:
        node = source["samples"]
        samples = pd.DataFrame(
            {
                k: node[v].asstr()[:]
                for k, v in [
                    ("ToolID", "tool_ids"),
                    ("RecordID", "record_ids"),
                    ("UnitID", "unit_ids"),
                ]
            }
        )
        samples["WearValue"] = node["wear"][:]
        samples["SourceRow"] = np.arange(len(samples))
        samples = filter_stage_records(samples, dataset)
        records = samples.drop_duplicates(["ToolID", "RecordID"])
        assignment = assign_splits(records, evaluation, seed)
        samples = samples.merge(
            assignment[["ToolID", "RecordID", "Split"]],
            on=["ToolID", "RecordID"],
            validate="many_to_one",
            sort=False,
        )
        samples, thresholds = label_training_records(samples, dataset)
        if (
            dataset.get("labeling", {}).get("method") == "stage_groups"
            or evaluation.get("strategy") in ("fixed_knife", "fixed_test_random_val")
        ):
            for split in ("train", "val", "test"):
                if set(samples.loc[samples.Split.eq(split), "Label"]) != set(spec.ids):
                    raise ValueError(f"{split} split must contain every configured class")
        out.attrs.update(dict(source.attrs))
        out.attrs.update(feature_set_identity=identity, **spec.to_metadata())
        out.attrs["labeling_json"] = json.dumps(dataset, sort_keys=True)
        out.attrs["evaluation_json"] = json.dumps(evaluation, sort_keys=True)
        if thresholds is not None:
            out.attrs.update(
                normal_moderate_threshold=thresholds[0],
                moderate_severe_threshold=thresholds[1],
            )
        values = node["values"]
        for split in ("train", "val", "test"):
            indices = np.flatnonzero(samples.Split.eq(split))
            if not len(indices):
                raise ValueError(f"Empty {split} split")
            group = out.create_group(f"splits/{split}/all")
            layout = h5py.VirtualLayout(shape=(len(indices), *values.shape[1:]), dtype=values.dtype)
            ref = h5py.VirtualSource(str(Path(feature).resolve()), values.name, shape=values.shape)
            for row, index in enumerate(samples.loc[indices, "SourceRow"]):
                layout[row] = ref[int(index)]
            group.create_virtual_dataset("values", layout)
            source.copy(node["feature_names"], group, name="feature_names")
            for column, key in [
                ("ToolID", "tool_ids"),
                ("RecordID", "record_ids"),
                ("UnitID", "unit_ids"),
            ]:
                group.create_dataset(
                    key,
                    data=samples.loc[indices, column].to_numpy(dtype=object),
                    dtype=h5py.string_dtype(),
                )
            group["wear"] = samples.loc[indices, "WearValue"].to_numpy()
            group["labels"] = samples.loc[indices, "Label"].to_numpy()
    samples.to_csv(Path(target).with_name("split_manifest.csv"), index=False, encoding="utf-8-sig")
    atomic_json(Path(target).with_name("split_tools.json"), {
        "split_seed": seed,
        "strategy": evaluation["strategy"],
        "tools": {split: sorted(samples.loc[samples.Split.eq(split), "ToolID"].unique().tolist())
                  for split in ("train", "val", "test")},
    })
    temporary.replace(target)
    return thresholds


def select_job(store, catalog, method, options, destination, reader, names=None):
    names = names or TRAIN_FILES
    table = reader.record_summary(store)
    if table.feature_names != catalog.feature_name.tolist():
        raise ValueError("Feature catalog differs from HDF5 feature order")
    if method == "r2":
        write_r2_threshold_report(
            table, destination / "r2_threshold_counts.csv", thresholds=options["report_thresholds"]
        )
    scores = (
        fisher_j_scores(table.values, table.labels)
        if method == "j"
        else r2_scores(table.values, table.wear, backend=options["score_backend"])
    )
    indices, receipts = select_domains(scores, catalog, method, options)
    frame = catalog.copy()
    frame["score"] = scores
    frame["method"] = method
    frame["selected"] = frame.feature_index.isin(indices)
    threshold_mode = method == "r2" and options.get("mode", "threshold") == "threshold"
    frame["passes_threshold"] = scores > float(options["threshold"]) if threshold_mode else True
    frame["selection_mode"] = options.get("mode", "threshold") if method == "r2" else "top"
    frame["selection_reason"] = np.where(
        frame.selected,
        "above_r2_threshold" if threshold_mode else "axis_or_domain_policy",
        np.where(frame.passes_threshold, "rank_limit", "below_threshold"),
    )
    frame["rank"] = frame.score.rank(method="first", ascending=False).astype(int)
    frame.to_csv(destination / names["scores"], index=False, encoding="utf-8-sig")
    frame[frame.selected].sort_values(
        ["domain", "axis", "score", "feature_index"], ascending=[True, True, False, True]
    ).to_csv(destination / names["top"], index=False, encoding="utf-8-sig")
    pd.DataFrame(receipts).to_csv(destination / names["quota"], index=False, encoding="utf-8-sig")
    atomic_json(
        destination / names["selection"],
        {"all": {"method": method, "selected_indices": indices.tolist(), "policy": options}},
    )
    return indices, scores


def _render_training_report(root):
    """Regenerate plots from saved selections and the linked Feature Set only."""
    import matplotlib.pyplot as plt

    from .confusion_plot import plot_confusion
    from .training import plot_training_history

    root = Path(root)
    _, report_config = load_config(root / "05_settings/config_resolved.yaml")
    run_registry = json.loads((root / "artifacts.json").read_text())
    report_names = run_registry["report_files"]
    links = json.loads((root / "04_feature_links/source.json").read_text())
    feature_root = Path(links["path"])
    manifest = json.loads((feature_root / "feature_set.json").read_text())
    if manifest.get("identity") != links["identity"]:
        raise ValueError("Report Feature Set identity changed")
    for entry in manifest["entries"]:
        if sha256(feature_root / entry["path"]) != entry["sha256"]:
            raise ValueError("Report Feature Set artifact missing or modified")
    for artifact in root.glob("03_details/*/nfft*/*/artifacts.json"):
        job = artifact.parent
        registry = json.loads(artifact.read_text())
        names = registry["files"]
        selection_directory = job / registry.get("selection_directory", ".")
        with h5py.File(job.parent / "split.h5") as store:
            class_names = list(classification_from_metadata(dict(store.attrs)).names)
        selected = selection_directory / names["selection"]
        if not selected.exists():
            continue
        scores = pd.read_csv(selection_directory / names["scores"])
        method = scores.method.iloc[0]
        means = None
        if method == "r2":
            chosen = json.loads(selected.read_text())["all"]["selected_indices"]
            table = FeatureReader(io_config=report_config["io"]).record_summary(job.parent / "split.h5")
            means, _ = wear_position_means(
                table.wear,
                table.values[:, chosen],
                [table.feature_names[index] for index in chosen],
                table.record_ids,
                table.tool_ids,
            )
            means.to_csv(
                selection_directory / "wear_position_means.csv",
                index=False,
                encoding="utf-8-sig",
            )
        plot_feature_groups(
            scores, job / "figures", report_config["output"]["plot_dpi"],
            means=means,
            threshold=report_config["selection"]["r2"].get("threshold") if method == "r2" else None,
        )
        for seed in registry["seeds"]:
            directory = job / seed["directory"]
            history = directory / seed["files"]["history"]
            if history.exists():
                plot_training_history(
                    pd.read_csv(history).to_dict("records"), directory / seed["files"]["curves"]
                )
            model_path = directory / seed["files"]["model"]
            iteration_history = model_path.with_suffix(".validation_iterations.csv")
            if iteration_history.exists():
                from .training import plot_validation_iterations

                plot_validation_iterations(
                    pd.read_csv(iteration_history), iteration_history.with_suffix(".png")
                )
            matrix = directory / "evaluation" / seed["evaluation"]["confusion"]
            if matrix.exists():
                plot_confusion(
                    np.loadtxt(matrix, delimiter=",", dtype=int),
                    matrix.parent / seed["evaluation"]["confusion_plot"],
                    class_names=class_names,
                )
        matrix = job / "ensemble" / registry["ensemble"]["confusion"]
        if matrix.exists():
            plot_confusion(
                np.loadtxt(matrix, delimiter=",", dtype=int),
                matrix.parent / registry["ensemble"]["confusion_plot"],
                class_names=class_names,
            )
    summary = pd.read_csv(root / "01_summary" / report_names["summary"])
    complete = summary[summary.status.eq("complete")]
    if not complete.empty:
        complete.sort_values("validation_macro_f1", ascending=False).to_csv(
            root / "01_summary" / report_names["ranking"], index=False, encoding="utf-8-sig"
        )
        for condition, rows in complete.groupby("condition"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for method, values in rows.groupby("method"):
                values = values.sort_values("nfft")
                ax.plot(values.nfft, values.validation_macro_f1, marker="o", label=method)
            ax.set(title=condition, xlabel="NFFT", ylabel="Validation Macro-F1", ylim=(0, 1))
            ax.legend()
            comparison = resolve_names(
                REPORT_FILES,
                run_registry["report_templates"],
                {**run_registry["context"], "condition": condition},
            )["comparison"]
            fig.tight_layout()
            fig.savefig(root / "02_figures" / comparison, dpi=160)
            plt.close(fig)
    (root / report_names["readme"]).write_text(
        f"# Training Run\n\n- [Summary](01_summary/{report_names['summary']})\n- [Comparison figures](02_figures/)\n- [Details](03_details/) by condition, NFFT, and selection method. Each method has grouped charts in `figures/`.\n- [Feature Set link](04_feature_links/source.json)\n- [Resolved config](05_settings/config_resolved.yaml)\n",
        encoding="utf-8",
    )
    output_manifest(root)
    return root


def render_training_report(root):
    root = Path(root).resolve()
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if state is not None:
        state.pop("report_error", None)
        state["report_status"] = "running"
        atomic_json(state_path, state)
    try:
        result = _render_training_report(root)
    except BaseException as exc:
        if state is not None:
            state["report_status"] = "failed" if isinstance(exc, Exception) else "interrupted"
            state["report_error"] = str(exc)
            atomic_json(state_path, state)
        raise
    if state is not None:
        state["report_status"] = "complete"
        atomic_json(state_path, state)
    return result


def train_batch(config_path, features=None, resume=None, on_run=None):
    source, cfg = load_config(config_path)
    from .configuration import validate_training_config

    validate_training_config(cfg)
    cfg["training"]["model"] = cfg["model"]
    cfg["training"]["inference_batch_size"] = cfg["evaluation"]["inference_batch_size"]
    cfg["training"]["inference"] = cfg["evaluation"]["inference"]
    validate_dataset(cfg["dataset"])
    classification_from_config(cfg)
    feature_path = features or cfg.get("features")
    if not feature_path:
        raise ValueError("Supply --features with a completed Feature Set directory")
    feature_root = resolve(source.parent, feature_path)
    manifest = json.loads((feature_root / "feature_set.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise ValueError("Incomplete or incompatible Feature Set")
    cfg["features"] = str(feature_root)
    cfg["resolved_seeds"] = training_seeds(cfg["training"])
    methods = cfg["selection"]["methods"]
    if not methods or len(set(methods)) != len(methods) or set(methods) - {"j", "r2"}:
        raise ValueError("methods must be a unique nonempty subset of j/r2")
    entries = [
        e
        for e in manifest["entries"]
        if (not cfg["conditions"] or e["condition"] in cfg["conditions"])
        and (not cfg["nffts"] or e["nfft"] in cfg["nffts"])
    ]
    if not entries:
        raise ValueError("No matching feature entries")
    report_context = {"process": manifest["process"], "condition": "all"}
    report_templates = cfg["output"].get("report_filenames")
    report_names = resolve_names(REPORT_FILES, report_templates, report_context)
    for entry in entries:
        path = resolve(feature_root, entry["path"])
        if not path.is_relative_to(feature_root) or sha256(path) != entry["sha256"]:
            raise ValueError(f"Invalid feature artifact: {path}")
        with h5py.File(path, "r") as feature_store:
            sampling = json.loads(feature_store.attrs["sampling_json"])
            if sampling.get("chunk_retention") != "all_complete" or "block_size" in sampling:
                raise ValueError("Rebuild Feature Set with all_complete chunk retention")
        for method in methods:
            for seed in [*cfg["resolved_seeds"], "shared", "ensemble"]:
                context = {
                    "process": manifest["process"],
                    "condition": entry["condition"],
                    "nfft": entry["nfft"],
                    "method": method,
                    "seed": seed,
                    "fold": "holdout",
                    "feature_index": 0,
                }
                resolve_names(TRAIN_FILES, cfg["output"].get("filenames"), context)
                resolve_names(EVAL_FILES, cfg["output"].get("evaluation_filenames"), context)
    digest = hashlib.sha256(
        json.dumps(
            {
                "config": cfg,
                "feature_identity": manifest["identity"],
                "revision": PIPELINE_REVISION,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    root = (
        Path(resume).resolve()
        if resume
        else new_run(
            source.parent,
            cfg["output"],
            manifest["process"],
            settings={
                "nffts": sorted({entry["nfft"] for entry in entries}),
                "conditions": sorted({entry["condition"] for entry in entries}),
                "class_names": cfg["dataset"]["class_names"],
                "selection_methods": methods,
            },
        )
    )
    for name in ("01_summary", "02_figures", "03_details", "04_feature_links", "05_settings"):
        (root / name).mkdir(exist_ok=True)
    state_path = root / "state.json"
    state = (
        json.loads(state_path.read_text())
        if resume
        else {"config_hash": digest, "jobs": {}, "total_jobs": len(entries) * len(methods)}
    )
    if state["config_hash"] != digest:
        raise ValueError(
            "Resume config, Feature Set or pipeline revision changed; start a new output run"
        )
    atomic_json(
        root / "artifacts.json",
        {
            "report_files": report_names,
            "report_templates": report_templates,
            "context": report_context,
        },
    )
    state.update(status="running", pid=os.getpid())
    atomic_json(state_path, state)
    if on_run:
        on_run(root)
    (root / "05_settings/config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    atomic_json(
        root / "04_feature_links/source.json",
        {"path": str(feature_root), "identity": manifest["identity"]},
    )
    import torch

    torch.set_num_threads(int(cfg["cpu_threads"]))
    try:
        for entry in entries:
            shared = root / "03_details" / entry["condition"] / f"nfft{entry['nfft']}"
            shared.mkdir(parents=True, exist_ok=True)
            store = shared / "split.h5"
            if not store.exists():
                split_store(
                    feature_root / entry["path"],
                    store,
                    int(cfg["split_seed"]),
                    manifest["identity"],
                    cfg,
                )
            cache_dir = root / ".cache" / "feature_arrays" if cfg["cache_features"] else None
            reader = FeatureReader(cache_dir, io_config=cfg["io"])
            catalog_path = (
                feature_root / entry["catalog"]
                if entry.get("catalog")
                else (feature_root / entry["path"]).with_name("feature_catalog.csv")
            )
            catalog = pd.read_csv(catalog_path, keep_default_na=False)
            for method in methods:
                name = f"{entry['condition']}_nfft{entry['nfft']}_{method}"
                job = shared / method
                job.mkdir(exist_ok=True)
                context = {
                    "process": manifest["process"],
                    "condition": entry["condition"],
                    "nfft": entry["nfft"],
                    "method": method,
                    "seed": "shared",
                    "fold": "holdout",
                    "feature_index": 0,
                }
                names = resolve_names(TRAIN_FILES, cfg["output"].get("filenames"), context)
                registry_path = job / "artifacts.json"
                if registry_path.exists():
                    selection_directory = json.loads(registry_path.read_text()).get(
                        "selection_directory", "."
                    )
                else:
                    selection_directory = "." if (job / names["selection"]).exists() else "selection"
                selection_path = job / selection_directory
                selection_path.mkdir(exist_ok=True)
                registry = {
                    "context": context,
                    "templates": cfg["output"].get("filenames"),
                    "files": names,
                    "selection_directory": selection_directory,
                    "seeds": [],
                    "ensemble": resolve_names(
                        EVAL_FILES,
                        cfg["output"].get("evaluation_filenames"),
                        {**context, "seed": "ensemble"},
                    ),
                }
                for seed in cfg["resolved_seeds"]:
                    seed_context = {**context, "seed": seed}
                    registry["seeds"].append(
                        {
                            "seed": seed,
                            "directory": f"seed_{seed}",
                            "files": resolve_names(
                                TRAIN_FILES, cfg["output"].get("filenames"), seed_context
                            ),
                            "evaluation": resolve_names(
                                EVAL_FILES, cfg["output"].get("evaluation_filenames"), seed_context
                            ),
                        }
                    )
                atomic_json(registry_path, registry)
                row = {
                    "name": name,
                    "condition": entry["condition"],
                    "nfft": entry["nfft"],
                    "method": method,
                }
                try:
                    state["current"] = {
                        "stage": "SELECTION",
                        "name": name,
                        "time": __import__("datetime").datetime.now().isoformat(),
                    }
                    atomic_json(state_path, state)
                    selection = selection_path / names["selection"]
                    if not selection.exists():
                        select_job(
                            store, catalog, method, cfg["selection"][method], selection_path, reader, names
                        )
                    prepared = prepare_training_data(
                        str(store), str(selection), "all", reader, cfg["preprocessing"]
                    )
                    checkpoints, seed_rows = [], []
                    for seed_info in registry["seeds"]:
                        seed = seed_info["seed"]
                        directory = job / seed_info["directory"]
                        directory.mkdir(exist_ok=True)
                        checkpoint = directory / seed_info["files"]["model"]
                        chosen = cfg["device"]
                        if chosen == "auto":
                            chosen = "cuda" if torch.cuda.is_available() else "cpu"
                        identity = training_identity(prepared, seed, chosen, cfg["training"])
                        if not valid_checkpoint(
                            checkpoint,
                            expected_identity=identity,
                            min_epochs=cfg["training"]["epochs"],
                        ):

                            def progress(values):
                                state["current"] = {
                                    "stage": "TRAINING",
                                    "name": name,
                                    "seed": seed,
                                    "time": __import__("datetime").datetime.now().isoformat(),
                                    **values,
                                }
                                atomic_json(state_path, state)

                            train_hdf5(
                                str(store),
                                checkpoint,
                                epochs=cfg["training"]["epochs"],
                                seed=seed,
                                device=cfg["device"],
                                selection=str(selection),
                                prepared=prepared,
                                make_plots=False,
                                progress=progress,
                                options=cfg["training"],
                                output_names=seed_info["files"],
                            )
                        report = evaluate_hdf5(
                            str(store),
                            str(checkpoint),
                            directory / "evaluation",
                            cfg["device"],
                            reader=reader,
                            make_plots=False,
                            filenames=seed_info["evaluation"],
                            batch_size=cfg["evaluation"]["batch_size"],
                        )
                        metrics = json.loads(
                            (report / seed_info["evaluation"]["metrics"]).read_text()
                        )
                        seed_rows.append(
                            {
                                "seed": seed,
                                "validation_macro_f1": metrics["validation"]["macro_f1"],
                                "test_macro_f1": metrics["macro_f1"],
                                "test_accuracy": metrics["accuracy"],
                            }
                        )
                        checkpoints.append(str(checkpoint))
                    pd.DataFrame(seed_rows).to_csv(
                        job / names["seed_metrics"], index=False, encoding="utf-8-sig"
                    )
                    report = evaluate_hdf5(
                        str(store),
                        checkpoints,
                        job / "ensemble",
                        cfg["device"],
                        reader=reader,
                        make_plots=False,
                        filenames=registry["ensemble"],
                        batch_size=cfg["evaluation"]["batch_size"],
                    )
                    metrics = json.loads((report / registry["ensemble"]["metrics"]).read_text())
                    row.update(
                        status="complete",
                        test_macro_f1=metrics["macro_f1"],
                        validation_macro_f1=metrics["validation"]["macro_f1"],
                        seed_macro_f1_mean=float(np.mean([r["test_macro_f1"] for r in seed_rows])),
                        seed_macro_f1_std=float(np.std([r["test_macro_f1"] for r in seed_rows])),
                    )
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                    atomic_json(job / "failure.json", row)
                state["jobs"][name] = row
                atomic_json(state_path, state)
                pd.DataFrame(state["jobs"].values()).to_csv(
                    root / "01_summary" / report_names["summary"], index=False, encoding="utf-8-sig"
                )
                print(f"{name}: {row['status']}", flush=True)
            reader.clear_memory()
        failed = sum(r["status"] == "failed" for r in state["jobs"].values())
        state["status"] = "complete_with_failures" if failed else "complete"
        atomic_json(state_path, state)
        atomic_json(
            root / "run_summary.json",
            {
                "total_jobs": state["total_jobs"],
                "failed": failed,
                "completed": len(state["jobs"]) - failed,
            },
        )
        try:
            render_training_report(root)
        except Exception as exc:
            print(f"Training report failed: {exc}", flush=True)
        return root
    except BaseException as exc:
        if state.get("status") not in ("complete", "complete_with_failures"):
            state.update(status="interrupted", error=str(exc))
            atomic_json(state_path, state)
        raise
