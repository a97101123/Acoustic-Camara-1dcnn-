"""Training from stored features, with train-only scaling and resumable epochs."""

import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .classification import classification_from_metadata
from .model import build_model, resolve_device

PIPELINE_REVISION = "explicit-functional-config-20261027-v6"


def training_identity(data, seed, device, options):
    digest = hashlib.sha256()
    for split in ["train", "val"]:
        digest.update(data["arrays"][split].tobytes())
        digest.update(data["labels"][split].tobytes())
    digest.update(data["median"].tobytes())
    digest.update(data["scale"].tobytes())
    digest.update(
        json.dumps(
            {
                "revision": PIPELINE_REVISION,
                "seed": seed,
                "group": data["feature_group"],
                "indices": data["feature_indices"],
                "feature_names": data["feature_names"],
                "metadata": data["metadata"],
                "device": device,
                "training": options,
                "learning_rate": float(options["learning_rate"]),
                **{
                    key: options[key]
                    for key in ("preload_device", "validation_batch_size")
                    if key in options
                },
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def predict_batches(torch, model, values, device, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    result = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = (
                values[start : start + batch_size].to(device=device, dtype=torch.float32)
                if torch.is_tensor(values)
                else torch.as_tensor(
                    np.array(values[start : start + batch_size], copy=True),
                    dtype=torch.float32,
                    device=device,
                )
            )
            result.append(model(x).detach().cpu().numpy())
    return (
        np.concatenate(result)
        if result
        else np.empty((0, model.head.out_features), dtype=np.float32)
    )


def prepare_training_data(hdf5_path, selection, group, reader, preprocessing):
    with h5py.File(hdf5_path, "r") as h:
        names = h[f"splits/train/{group}/feature_names"].asstr()[:].tolist()
        labels = {s: h[f"splits/{s}/{group}/labels"][:].astype(np.int64) for s in ["train", "val"]}
        metadata = {k: v.item() if hasattr(v, "item") else v for k, v in h.attrs.items()}
    indices = list(range(len(names)))
    if selection:
        spec = json.loads(Path(selection).read_text(encoding="utf-8"))
        indices = spec[group]["selected_indices"]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(names)
            for i in indices
        )
    ):
        raise ValueError("Invalid selected feature indices")
    arrays = {s: reader.values(hdf5_path, group, s, indices) for s in labels}
    classification = classification_from_metadata(metadata)
    for s, values in arrays.items():
        if not len(values) or len(values) != len(labels[s]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {s} feature values")
        if not np.isin(labels[s], classification.ids).all():
            raise ValueError(f"Invalid {s} labels")
    train = arrays["train"]
    flat = train.reshape(-1, train.shape[-1])
    median = np.median(flat, axis=0)
    q25, q75 = np.percentile(flat, [25, 75], axis=0)
    scale = q75 - q25
    scale[scale < preprocessing["scale_epsilon"]] = 1
    for s, x in arrays.items():
        x = ((x - median) / scale).astype(np.float32)
        arrays[s] = np.transpose(x, (0, 2, 1)).copy()
    return {
        "arrays": arrays,
        "labels": labels,
        "median": median,
        "scale": scale,
        "feature_indices": indices,
        "feature_names": [names[i] for i in indices],
        "feature_group": group,
        "metadata": metadata,
        "input_channels": arrays["train"].shape[1],
    }


def plot_training_history(history, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ["train_loss", "val_loss"]:
        if key in frame:
            axes[0].plot(frame["epoch"], frame[key], label=key)
    for key in ["train_accuracy", "val_accuracy"]:
        if key in frame:
            axes[1].plot(frame["epoch"], frame[key], label=key)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_validation_iterations(validation_frame, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    best_point = validation_frame.loc[validation_frame.val_loss.idxmin()]
    for axis, key, title in zip(
        axes, ("val_loss", "val_accuracy"), ("Validation loss", "Validation accuracy")
    ):
        axis.plot(validation_frame.iteration, validation_frame[key], marker=".")
        axis.axvline(best_point.iteration, linestyle="--", color="gray",
                     label=f"Selected checkpoint: update {int(best_point.iteration)}")
        axis.set(xlabel="Optimizer updates", ylabel=title)
        axis.legend()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _save(torch, data, path):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def train_hdf5(
    hdf5_path,
    output,
    epochs,
    seed,
    device,
    selection=None,
    group="all",
    progress=None,
    prepared=None,
    make_plots=True,
    options=None,
    output_names=None,
):
    import torch

    from .configuration import training_settings

    training_settings(options)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    chosen = resolve_device(torch, device)
    if chosen not in ["cpu", "cuda"] or chosen == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Unavailable device: {chosen}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if prepared is None:
        raise ValueError("prepared training arrays and explicit preprocessing config are required")
    data = prepared
    x, y = data["arrays"], data["labels"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = (
        output.parent / output_names["history"]
        if output_names
        else output.with_suffix(".history.csv")
    )
    curves = (
        output.parent / output_names["curves"]
        if output_names
        else output.with_suffix(".curves.png")
    )
    resume_path = output.with_name(output.name + ".resume.pt")
    batch_size = options["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    checkpoint_every = options["checkpoint_every"]
    validation_batch_size = options["validation_batch_size"]
    for key, value in [
        ("checkpoint_every", checkpoint_every),
        ("validation_batch_size", validation_batch_size),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(options["preload_device"], bool):
        raise ValueError("preload_device must be a boolean")
    rate = float(options["learning_rate"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid learning_rate")
    identity = training_identity(data, seed, chosen, options)
    device_x, device_y = None, None
    if options["preload_device"]:
        device_x = {split: torch.as_tensor(values, device=chosen) for split, values in x.items()}
        device_y = torch.as_tensor(y["train"], device=chosen)
    classification = classification_from_metadata(data["metadata"])
    model = build_model(
        torch, data["input_channels"], classification.num_classes, options["model"]
    ).to(chosen)
    optimizer_config = options["optimizer"]
    matlab_adam = options.get("matlab_adam", False)
    parameters = model.parameters()
    if matlab_adam:
        weights, other = [], []
        for parameter in model.parameters():
            (weights if parameter.ndim > 1 else other).append(parameter)
        parameters = [{"params": weights}, {"params": other, "weight_decay": 0.0}]
    optimizer = torch.optim.Adam(
        parameters,
        lr=rate,
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
        amsgrad=optimizer_config["amsgrad"],
    )
    plateau_patience = options.get("lr_plateau_patience")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-6
        )
        if plateau_patience is not None else None
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=options.get("label_smoothing", 0.0))
    start = 0
    history = []
    validation_history = []
    patience = options.get("early_stop_patience")
    stale_checks = 0
    stopped_early = False
    stop_iteration = None
    best = float("inf")
    best_state = None
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=chosen, weights_only=False)
        if saved["training_identity"] != identity:
            raise ValueError("Resume training identity differs")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"]
        history = saved["history"]
        validation_history = saved.get("validation_history", [])
        stale_checks = saved.get("stale_checks", 0)
        stopped_early = saved.get("stopped_early", False)
        stop_iteration = saved.get("stop_iteration")
        best = saved["best_loss"]
        best_state = saved["best_state"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if chosen == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
    val_input = device_x["val"] if device_x is not None else x["val"]

    def validate():
        model.eval()
        logits = predict_batches(torch, model, val_input, chosen, validation_batch_size)
        val_loss = loss_fn(torch.as_tensor(logits), torch.as_tensor(y["val"])).item()
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        return logits, val_loss

    fixed_order = None
    if matlab_adam:
        fixed_order = np.random.RandomState(seed).permutation(len(y["train"]))
        if len(fixed_order) >= batch_size:
            fixed_order = fixed_order[:len(fixed_order) // batch_size * batch_size]
    for epoch in range(start, start if stopped_early else epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        gpu_losses, gpu_correct, batch_counts = [], [], []
        order = fixed_order if matlab_adam else np.random.permutation(len(y["train"]))
        batches_per_epoch = (len(order) + batch_size - 1) // batch_size
        for offset in range(0, len(order), batch_size):
            ids = order[offset : offset + batch_size]
            if device_x is not None:
                device_ids = torch.as_tensor(ids, device=chosen)
                batch = device_x["train"].index_select(0, device_ids)
                target = device_y.index_select(0, device_ids)
            else:
                batch = torch.as_tensor(x["train"][ids], device=chosen)
                target = torch.as_tensor(y["train"][ids], device=chosen)
            optimizer.zero_grad()
            logits = model(batch)
            loss = train_loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            seen += len(ids)
            iteration = epoch * batches_per_epoch + offset // batch_size + 1
            if matlab_adam and (iteration == 1 or iteration % 50 == 0):
                iteration_logits, iteration_val_loss = validate()
                validation_history.append({
                    "iteration": iteration,
                    "epoch": iteration / batches_per_epoch,
                    "val_loss": iteration_val_loss,
                    "val_accuracy": float(np.mean(iteration_logits.argmax(1) == y["val"])),
                })
                if iteration_val_loss < best:
                    stale_checks = 0
                    best = iteration_val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale_checks += 1
                stopped_early = patience is not None and stale_checks >= patience
                model.train()
            if chosen == "cuda":
                # Collect detached scalars on the device; two host transfers per
                # epoch replace two synchronizations per mini-batch. Keep the
                # original Python-double loss accumulation order.
                gpu_losses.append(loss.detach())
                gpu_correct.append((logits.argmax(1) == target).sum())
                batch_counts.append(len(ids))
            else:
                total_loss += loss.item() * len(ids)
                correct += (logits.argmax(1) == target).sum().item()
            if stopped_early:
                stop_iteration = iteration
                break
        if chosen == "cuda":
            total_loss = sum(
                value * count
                for value, count in zip(torch.stack(gpu_losses).cpu().tolist(), batch_counts)
            )
            correct = torch.stack(gpu_correct).sum().item()
        logits, val_loss = validate()
        record = {
            "epoch": iteration / batches_per_epoch if stopped_early else epoch + 1,
            "train_loss": total_loss / seen,
            "val_loss": val_loss,
            "train_accuracy": correct / seen,
            "val_accuracy": float(np.mean(logits.argmax(1) == y["val"])),
        }
        if scheduler is not None:
            record["learning_rate"] = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
        history.append(record)
        if (not matlab_adam or epoch + 1 == epochs) and val_loss < best:
            stale_checks = 0
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif not matlab_adam:
            stale_checks += 1
        if not matlab_adam and patience is not None and stale_checks >= patience:
            stopped_early = True
            stop_iteration = iteration
        if stopped_early or (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
            snapshot = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch + 1,
                "history": history,
                "validation_history": validation_history,
                "best_loss": best,
                "stale_checks": stale_checks,
                "stopped_early": stopped_early,
                "stop_iteration": stop_iteration,
                "best_state": best_state,
                "training_identity": identity,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if chosen == "cuda" else [],
            }
            _save(torch, snapshot, resume_path)
            pd.DataFrame(history).to_csv(history_path, index=False)
        if progress:
            progress(record)
        if stopped_early:
            break
    if validation_history:
        final_iteration = stop_iteration or epochs * ((len(fixed_order) + batch_size - 1) // batch_size)
        if validation_history[-1]["iteration"] != final_iteration:
            validation_history.append({
                "iteration": final_iteration,
                "epoch": epochs,
                "val_loss": history[-1]["val_loss"],
                "val_accuracy": history[-1]["val_accuracy"],
            })
        validation_frame = pd.DataFrame(validation_history)
        validation_frame.to_csv(output.with_suffix(".validation_iterations.csv"), index=False)
        if make_plots:
            plot_validation_iterations(validation_frame, output.with_suffix(".validation_iterations.png"))
    saved = {
        k: data[k]
        for k in [
            "median",
            "scale",
            "feature_indices",
            "feature_names",
            "feature_group",
            "input_channels",
        ]
    }
    saved.update(
        model_config=options["model"],
        inference_batch_size=options["inference_batch_size"],
        inference=options["inference"],
        state_dict=best_state,
        seed=seed,
        epochs=len(history),
        stopped_early=stopped_early,
        stop_iteration=stop_iteration,
        training_identity=identity,
        store_metadata=data["metadata"],
        metadata=data["metadata"],
    )
    _save(torch, saved, output)
    metadata_path = (
        output.parent / output_names["model_metadata"]
        if output_names
        else output.with_suffix(".json")
    )
    from .run_io import atomic_json

    atomic_json(
        metadata_path,
        {
            "seed": seed,
            "epochs": len(history),
            "training_identity": identity,
            "feature_names": data["feature_names"],
            "feature_group": data["feature_group"],
            "input_channels": data["input_channels"],
            "best_validation_loss": best,
            "stopped_early": stopped_early,
            "stop_iteration": stop_iteration,
            "metadata": data["metadata"],
        },
    )
    if make_plots:
        plot_training_history(history, curves)
    return output
