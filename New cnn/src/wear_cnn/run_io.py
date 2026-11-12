"""Resolved configuration, durable status writes and unique run directories."""

import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml


class _ConfigLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate config key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    path = Path(path).resolve()

    def read(source, stack):
        if source in stack:
            raise ValueError(f"Config include cycle: {source}")
        value = yaml.load(source.read_text(encoding="utf-8"), Loader=_ConfigLoader)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a configuration mapping: {source}")
        if "includes" not in value:
            return value
        if (
            set(value) - {"includes", "overrides"}
            or not isinstance(value["includes"], list)
            or not value["includes"]
        ):
            raise ValueError(f"Launch config requires includes and optional overrides: {source}")
        merged = {}
        for relative in value["includes"]:
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"Config includes must contain paths: {source}")
            part = read((source.parent / relative).resolve(), (*stack, source))
            duplicate = set(merged) & set(part)
            if duplicate:
                raise ValueError(f"Duplicate config section: {sorted(duplicate)[0]}")
            merged.update(part)

        def override(target, changes, location=""):
            if not isinstance(changes, dict):
                raise ValueError(f"Config overrides must be a mapping: {location or source}")
            for key, replacement in changes.items():
                field = f"{location}.{key}" if location else key
                if key not in target:
                    raise ValueError(f"Unknown override field: {field}")
                if isinstance(target[key], dict):
                    override(target[key], replacement, field)
                else:
                    target[key] = replacement

        override(merged, value.get("overrides", {}))
        return merged

    return path, read(path, ())


def resolve(base, path):
    return (Path(base) / path).resolve()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def new_run(base, output, process, *, settings=None):
    name = output["name"]
    if Path(name).name != name or any(c in name for c in '<>:"/\\|?*'):
        raise ValueError("output.name must be a safe directory name")
    started = datetime.now().astimezone()
    root = resolve(base, str(output["root"]).replace("{month}", started.strftime("%Y-%m")))
    stamp = uuid.uuid4().hex[:6]
    run = root / process / f"{started:%Y%m%d_%H%M%S}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    atomic_json(
        run / "run.json",
        {
            "run_id": run.name,
            "name": name,
            "process": process,
            "created_at": started.isoformat(),
            "settings": settings or {},
        },
    )
    return run


def output_manifest(root):
    import pandas as pd

    root = Path(root)
    pd.DataFrame(
        [
            {
                "path": str(p.resolve()),
                "relative_path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
                "kind": p.suffix,
            }
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "output_manifest.csv"
        ]
    ).to_csv(root / "output_manifest.csv", index=False, encoding="utf-8-sig")
