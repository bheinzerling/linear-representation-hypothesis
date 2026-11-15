"""Extract the complete, checksum-verified model-input bundle."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from lrh.io import sha256

ROOT = Path(__file__).resolve().parents[1]


def download(archive: Path, manifest: dict) -> None:
    """Download to a temporary file and publish only after checking its hash."""
    url = manifest.get("download_url")
    if not url:
        raise ValueError("The input manifest does not specify a download URL")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        if sha256(temporary) != manifest["sha256"]:
            raise ValueError("Downloaded model-input archive checksum mismatch")
        os.replace(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract(archive: Path, destination: Path, manifest: dict) -> int:
    if sha256(archive) != manifest["sha256"]:
        raise ValueError("Model-input archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        expected = manifest["files"]
        if {m.name for m in members} != set(expected):
            raise ValueError("Archive members differ from the input manifest")
        for member in members:
            target = (destination / member.name).resolve()
            if (
                not member.isfile()
                or not target.is_relative_to(destination.resolve())
                or Path(member.name).parts[0] not in {"numerical", "properties"}
            ):
                raise ValueError(f"Invalid model-input member: {member.name}")
            if target.exists() and sha256(target) != expected[member.name]["sha256"]:
                raise FileExistsError(f"Existing model input differs: {target}")
        handle.extractall(destination, members=members, filter="data")
    for name, entry in expected.items():
        if sha256(destination / name) != entry["sha256"]:
            raise ValueError(f"Extracted model-input checksum mismatch: {name}")
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/model-inputs.tar.gz")
    parser.add_argument("--destination", type=Path, default=ROOT / "experiments")
    parser.add_argument("--download", action="store_true", help="Fetch the pinned release asset if the archive is absent")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifests/model-inputs.json").read_text())
    if not args.archive.is_file():
        if not args.download:
            parser.error("Model-input archive is absent. Add --download to fetch the checksum-pinned release asset.")
        download(args.archive, manifest)
    count = extract(args.archive, args.destination, manifest)
    print(f"Prepared {count} model-input files in {args.destination}")


if __name__ == "__main__":
    main()
