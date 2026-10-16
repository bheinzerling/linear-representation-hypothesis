"""Record data checksums after an intentional, reviewed evidence update."""

from pathlib import Path

from lrh.io import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = ROOT / "data"
    manifest = data / "MANIFEST.sha256"
    files = [
        p
        for p in sorted(data.rglob("*"))
        if p.is_file() and p != manifest and "__pycache__" not in p.parts
    ]
    manifest.write_text("".join(f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files))
    print(f"Recorded {len(files)} data hashes in {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
