from __future__ import annotations

import argparse
import ast
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/run_number_property_list_steering.py"


def test_parity_reproduction_command_matches_script_parser(monkeypatch):
    document = (ROOT / "docs/model-experiments.md").read_text()
    examples = [
        block for block in re.findall(r"```bash\n(.*?)```", document, re.DOTALL) if SCRIPT in block
    ]
    assert len(examples) == 1
    command = shlex.split(examples[0].replace("\\\n", ""))
    source = ROOT / "experiments/properties" / SCRIPT
    parser = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    # Run the unchanged argument parser without importing the model runtime.
    namespace = {"argparse": argparse, "Path": Path}
    code = compile(ast.Module(body=[parser], type_ignores=[]), str(source), "exec")
    exec(code, namespace)  # noqa: S102 -- Only the checked-in argument parser is executed.
    monkeypatch.setattr("sys.argv", command[command.index(SCRIPT) :])
    assert namespace["parse_args"]().property == "parity"
