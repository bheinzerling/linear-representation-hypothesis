"""Render all main and appendix figures from the repository's frozen inputs."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = [
    "render_main.py",
    "render_components.py",
    "render_complexity.py",
    "render_pls_transfer.py",
    "render_transfer.py",
]


def main() -> None:
    for name in GENERATORS:
        print(f"Rendering {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "figures" / name)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
