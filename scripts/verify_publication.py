"""Verify that the public branch, latest release, and downloaded source agree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from lrh.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def api(repository: str, resource: str = "") -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}{resource}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "lrh-release-verifier"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download(asset: dict, directory: Path) -> Path:
    name = asset["name"]
    if Path(name).name != name:
        raise ValueError(f"Invalid asset filename: {name}")
    path = directory / name
    temporary = path.with_name(path.name + ".part")
    try:
        with (
            urllib.request.urlopen(asset["browser_download_url"], timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if temporary.stat().st_size != asset["size"]:
            raise ValueError(f"Published asset size mismatch: {name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def verify_source_archive(archive: Path, commit: str) -> int:
    tree = subprocess.check_output(["git", "ls-tree", "-rz", commit], cwd=ROOT)
    blobs = {}
    for record in tree.split(b"\0"):
        if record:
            metadata, name = record.split(b"\t", 1)
            blobs[name.decode()] = metadata.split()[2].decode()
    seen = set()
    prefix = "linear-representation-hypothesis/"
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            if not member.filename.startswith(prefix):
                raise ValueError(f"Unexpected source member: {member.filename}")
            name = member.filename.removeprefix(prefix)
            content = handle.read(member)
            digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if name in seen or blobs.get(name) != digest:
                raise ValueError(f"Released source differs from public commit: {name}")
            seen.add(name)
    required = {
        "paper/paper.tex", "docs/model-experiments.md", "scripts/verify.py",
        "experiments/numerical/scripts/prepare_figure7_matched_inputs.py",
        "experiments/numerical/scripts/run_figure7_matched_probes.py",
        "experiments/numerical/src/figure7_historical_comparison.py",
    }
    if not required <= seen:
        raise ValueError(f"Required publication sources absent: {sorted(required - seen)}")
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", default="bheinzerling/linear-representation-hypothesis")
    parser.add_argument("--expected-commit")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/release.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/publication-verification")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Publication verification requires a clean source checkout")
    commit = args.expected_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    repository = api(args.repository)
    if repository["private"]:
        raise ValueError("The paper artifact repository is not public")
    branch = repository["default_branch"]
    for label, ref in (("default branch", branch), ("release tag", args.tag)):
        observed = api(args.repository, "/commits/" + urllib.parse.quote(ref, safe=""))["sha"]
        if observed != commit:
            raise ValueError(f"Public {label} has {observed}; expected {commit}")
    release = api(args.repository, "/releases/latest")
    if release["tag_name"] != args.tag or release["draft"] or release["prerelease"]:
        raise ValueError(f"Latest published release does not match {args.tag}")
    assets = {entry["name"]: entry for entry in release["assets"]}
    expected = json.loads(args.manifest.read_text())
    required = set(expected) | {"release.json"}
    if not required <= assets.keys():
        raise ValueError(f"Missing published assets: {sorted(required - assets.keys())}")
    directory = args.output_dir / args.tag
    directory.mkdir(parents=True, exist_ok=True)
    published_manifest = download(assets["release.json"], directory)
    if json.loads(published_manifest.read_text()) != expected:
        raise ValueError("Published release manifest differs from the verified local manifest")
    for name, entry in expected.items():
        path = download(assets[name], directory)
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Published asset checksum mismatch: {name}")
    source_files = verify_source_archive(directory / "linear-representation-hypothesis-source.zip", commit)
    receipt = {
        "status": "published_and_verified", "repository": args.repository,
        "commit": commit, "tag": args.tag, "release_url": release["html_url"],
        "source_files_matching_commit": source_files, "assets": expected,
    }
    write_json(directory / "verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
