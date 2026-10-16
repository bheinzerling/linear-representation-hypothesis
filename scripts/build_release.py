"""Build a deterministic source archive from the standalone repository."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

from lrh.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DATA",
    "NOTICE.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    "AGENTS.md",
}
FOLDERS = {
    "src",
    "scripts",
    "data",
    "figures",
    "paper",
    "docs",
    "tests",
    "envs",
    "experiments",
    ".github",
}
EXCLUDED = {".venv", "__pycache__", "out", "results", ".pytest_cache", ".ruff_cache"}


def release_files(root: Path = ROOT) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or (rel.parts[0] not in FOLDERS and rel.as_posix() not in ROOT_FILES):
            continue
        if EXCLUDED.intersection(rel.parts) or path.suffix == ".pyc":
            continue
        if (
            rel.parts[0] == "experiments"
            and len(rel.parts) > 2
            and rel.parts[2] in {"data", "out", "results", "experiment-reports"}
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Release content must be a regular file: {rel}")
        files.append(path)
    return files


def main() -> None:
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    archive = output / "linear-representation-hypothesis-source.zip"
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in release_files():
            name = "linear-representation-hypothesis/" + path.relative_to(ROOT).as_posix()
            item = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = (0o100755 if path.name.endswith(".sh") else 0o100644) << 16
            handle.writestr(item, path.read_bytes())
    archive.write_bytes(buffer.getvalue())
    entries = [archive]
    model_inputs = output / "model-inputs.tar.gz"
    if model_inputs.is_file():
        entries.append(model_inputs)
    manifest = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in entries}
    write_json(output / "release.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
