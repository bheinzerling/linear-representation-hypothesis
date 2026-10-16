from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_").replace(".", "_")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def atomic_write_csv(path: str | Path, frame: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def atomic_write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(to_jsonable(payload), sort_keys=True))


def save_numpy(path: str | Path, array: Any) -> None:
    import numpy as np

    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    cwd = str(cwd or Path.cwd())

    def run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": (run(["git", "status", "--porcelain"]) or "") != "",
    }


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "git": get_git_metadata(),
    }


def dump_run_config(path: str | Path, config: dict[str, Any]) -> None:
    payload = {"config": config, "runtime": runtime_metadata()}
    atomic_write_json(path, payload)


def maybe_resume(path: str | Path, force: bool = False) -> bool:
    return Path(path).exists() and not force
