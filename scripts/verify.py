"""Check frozen input hashes and the completeness of local source imports."""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from lrh.io import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(root: Path = ROOT) -> int:
    manifest = root / "data/MANIFEST.sha256"
    checked = 0
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid manifest path: {name}")
        if sha256(path) != digest:
            raise ValueError(f"Input hash mismatch: {name}")
        listed.add(name)
        checked += 1
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / "data").rglob("*")
        if p.is_file() and p != manifest and "__pycache__" not in p.parts
    }
    if listed != actual:
        raise ValueError(f"Unlisted or stale data paths: {sorted(listed ^ actual)}")
    return checked


def verify_model_imports(root: Path = ROOT) -> int:
    checked = 0
    for folder, package in (("numerical", "src"), ("properties", "introlm")):
        source = root / "experiments" / folder
        for path in sorted(source.rglob("*.py")):
            if any(p in path.parts for p in (".venv", "__pycache__", "out", "data")):
                continue
            checked += 1
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = path.parent
                        for _ in range(node.level - 1):
                            base = base.parent
                        names = [node.module] if node.module else []
                    else:
                        base = source / "src" if package == "introlm" else source
                        names = [node.module] if node.module else []
                        names = [n for n in names if n.split(".")[0] == package]
                elif isinstance(node, ast.Import):
                    base = source / "src" if package == "introlm" else source
                    names = [n.name for n in node.names if n.name.split(".")[0] == package]
                else:
                    continue
                for name in names:
                    target = base.joinpath(*name.split("."))
                    if not target.is_dir() and not target.with_suffix(".py").is_file():
                        raise ValueError(
                            f"Missing local import in {path.relative_to(root)}: {name}"
                        )
    return checked


def verify_paper_tables(root: Path = ROOT) -> int:
    with (root / "data/literature/evidence_ratings.csv").open(newline="") as handle:
        ratings = {(r["paper"], r["claim"]): r["rating"] for r in csv.DictReader(handle)}
    claims = ["D", "E", "R", "I", "SC", "N", "S_C", "S_M"]
    checked = 0
    seen = set()
    source = (root / "paper/paper.tex").read_text()
    tables = re.findall(r"\\begin\{tabularx\}(.*?)\\end\{tabularx\}", source, re.DOTALL)
    for line in [row for table in tables for row in re.split(r"\\\\", table)]:
        if r"\Ev" not in line or r"\citet{" not in line:
            continue
        study = re.search(r"\\citet\{([^}]+)\}", line).group(1)
        markers = re.findall(r"\\Ev(Support|Mixed|Against)\{(\d)\}|\\Ev(Untested)", line)
        if len(markers) != len(claims):
            raise ValueError(f"Unexpected evidence-table row: {study}")
        for claim, (kind, strength, untested) in zip(claims, markers, strict=True):
            observed = (
                "U" if untested else {"Support": "+", "Mixed": "~", "Against": "-"}[kind] + strength
            )
            if ratings[study, claim] != observed:
                raise ValueError(f"Manuscript and evidence ratings differ: {study}, {claim}")
            checked += 1
        seen.add(study)
    if len(seen) != 30 or checked != (17 + 30) * len(claims):
        raise ValueError("The manuscript must contain the complete and excerpted evidence tables")
    return checked


def main() -> None:
    inputs = verify_manifest()
    sources = verify_model_imports()
    cells = verify_paper_tables()
    symlinks = [
        p.relative_to(ROOT).as_posix()
        for folder in ("src", "scripts", "data", "figures", "paper", "experiments")
        for p in (ROOT / folder).rglob("*")
        if p.is_symlink() and ".venv" not in p.parts
    ]
    if symlinks:
        raise ValueError(f"Repository content must use regular files: {symlinks}")
    print(
        f"Verified {inputs} frozen inputs, {sources} model-source files, and {cells} manuscript table cells."
    )


if __name__ == "__main__":
    main()
