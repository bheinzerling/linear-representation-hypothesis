"""Plan and execute CPU reproduction from immutable evidence files."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lrh.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "mentions": "lrh.analysis.mentions",
    "numerical": "lrh.analysis.numerical",
    "literature": "lrh.analysis.literature",
    "probe_mdl": "lrh.analysis.probe_mdl",
}


@dataclass(frozen=True)
class RunPlan:
    output: Path
    run_id: str
    identity: dict


def plan_run(root: Path, output: Path) -> RunPlan:
    inputs = {
        p.relative_to(root).as_posix(): sha256(p)
        for folder in ("data", "src/lrh")
        for p in sorted((root / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")
    }
    identity = {"inputs": inputs, "packages": versions, "python": platform.python_version()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return RunPlan(output=output, run_id=digest[:16], identity=identity)


def is_complete(plan: RunPlan) -> bool:
    try:
        result = json.loads((plan.output / "run.json").read_text())
        if (plan.output / "DONE").read_text().strip() != plan.run_id:
            return False
        return (
            result["run_id"] == plan.run_id
            and bool(result["outputs"])
            and all(
                (plan.output / name).is_file() and sha256(plan.output / name) == digest
                for name, digest in result["outputs"].items()
            )
        )
    except (OSError, ValueError, KeyError):
        return False


def run(plan: RunPlan, root: Path) -> bool:
    plan.output.mkdir(parents=True, exist_ok=True)
    with (plan.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if is_complete(plan):
            print(f"Verified existing reproduction {plan.run_id}: {plan.output}", flush=True)
            return False
        (plan.output / "DONE").unlink(missing_ok=True)
        write_json(plan.output / "run.json", {"run_id": plan.run_id, **plan.identity})
        for task, module in TASKS.items():
            command = [
                sys.executable,
                "-m",
                module,
                "--data-dir",
                str(root / "data" / task),
                "--output-dir",
                str(plan.output / task),
            ]
            print(f"Reproducing {task}", flush=True)
            subprocess.run(command, check=True, cwd=root)
        outputs = {
            p.relative_to(plan.output).as_posix(): sha256(p)
            for task in TASKS
            for p in sorted((plan.output / task).rglob("*"))
            if p.is_file()
        }
        write_json(
            plan.output / "run.json",
            {
                "run_id": plan.run_id,
                **plan.identity,
                "outputs": outputs,
                "command": "uv run python scripts/reproduce.py",
            },
        )
        (plan.output / "DONE").write_text(plan.run_id + "\n")
        print(f"Completed reproduction {plan.run_id}: {plan.output}", flush=True)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/reproduction")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = plan_run(ROOT, args.output_dir.resolve())
    if args.plan_only:
        print(
            json.dumps(
                {"run_id": plan.run_id, "complete": is_complete(plan), "tasks": list(TASKS)},
                indent=2,
            )
        )
    else:
        run(plan, ROOT)
