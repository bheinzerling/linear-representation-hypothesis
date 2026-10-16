"""Extract the complete, checksum-verified model-input bundle."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from lrh.io import sha256

ROOT = Path(__file__).resolve().parents[1]


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
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifests/model-inputs.json").read_text())
    count = extract(args.archive, args.destination, manifest)
    print(f"Prepared {count} model-input files in {args.destination}")


if __name__ == "__main__":
    main()
