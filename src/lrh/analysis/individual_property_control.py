"""Reconstruct property-control scores, development choices and uncertainty."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import runpy
import tempfile
from pathlib import Path

TABLE_NAMES = {
    "parity": "Parity", "primality": "Primality",
    **{f"divisible_by_{n}": f"Divisible by {n}" for n in (3, 4, 5, 6, 8, 10, 25)},
    "residue_mod_3": "Residue mod 3", "residue_mod_4": "Residue mod 4",
    "units_digit": "Units digit", "tens_digit": "Tens digit", "hundreds_digit": "Hundreds digit",
    "digit_count": "Digit count", "distinct_digit_count": "Distinct digit count",
    "contains_zero": "Contains zero", "has_repeated_digit": "Repeated digits", "palindrome": "Palindrome",
}


def table_rows(summaries: list[dict], selected: list[str]) -> str:
    by_property = {row["property"]: row for row in summaries}
    lines = []
    for prop, name in TABLE_NAMES.items():
        label = rf"\textbf{{{name}}}" if prop in selected else name
        if prop not in by_property:
            lines.append(label + r" & \multicolumn{4}{c}{Baseline failed} & 0/3 \\")
            continue
        row = by_property[prop]
        low, high = row["target_interval"]
        control = max(row[k] for k in ("wrong_sign", "random", "location"))
        lines.append(f"{label} & {100 * row['selected']:.1f} & {100 * low:.1f}--{100 * high:.1f}"
                     f" & {100 * control:.1f} & {row['baseline_correct_n']} & {row['evaluated_seeds']}/3 " + r"\\")
    return "\n".join(lines) + "\n"


def selective_properties(cells: list[dict], properties: list[str], seeds: list[int]) -> list[str]:
    """Apply the caption's observed-rate rule to the entire property inventory."""
    selected = []
    for prop in properties:
        rows = [r for r in cells if (r["model"], r["property"], r["context"],
                r["family"], r["mode"]) == ("qwen36_27b", prop, "property", "list", "selected")]
        if len(rows) != len(seeds) or {r["seed"] for r in rows} != set(seeds):
            continue
        if all(r["dev_identity_gate"] and set(r["class_outcomes"]) == {"0", "1"}
               and all(v["baseline_correct_n"] > 0
                       and all(x is not None for x in v["conditional_success_rates"])
                       and v["conditional_success_rates"][0] > max(v["conditional_success_rates"][1:])
                       for v in r["class_outcomes"].values()) for r in rows):
            selected.append(prop)
    return selected


def unpack(records: Path, destination: Path) -> None:
    seen = set()
    cases = {}
    with gzip.open(records, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            path = Path(record["path"])
            if path.is_absolute() or ".." in path.parts or path.as_posix() in seen:
                raise ValueError("Unsafe or duplicate reconstruction path")
            seen.add(path.as_posix())
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = record["payload"]
            if path.name == "000-case-definitions.json":
                cases[path.parent.as_posix()] = payload
            if target.suffix == ".gz":
                if cases:
                    definitions = cases[Path(*path.parts[:2]).as_posix()]
                    for row in payload["rows"]:
                        row.update(definitions[row["case_id"]])
                target.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
            elif target.name == "DONE":
                target.write_text(payload)
            else:
                target.write_text(json.dumps(payload))


def verify_protocol(root: Path, config: dict, protocol: dict, source_digest: str) -> None:
    """Check deterministic partitions, baseline gates, pins and complete coverage."""
    paths = sorted(root.glob("*/*/results.json"))
    results = [json.loads(p.read_text()) for p in paths]
    actual = {(r["task"]["model"], r["task"]["property"], r["task"]["seed"])
              for r in results if not r["task"]["pilot"]}
    expected = {(m, p, s) for m in config["models"] for p in config["properties"] for s in config["seeds"]}
    if actual != expected or len(results) != 120:
        raise ValueError("Incomplete experiment roster")
    for path, result in zip(paths, results, strict=True):
        task = result["task"]
        if result["runtime"]["source_digest"] != source_digest:
            raise ValueError("Recorded execution source differs from the registered revision")
        spec = protocol["PROPERTIES"][task["property"]]
        splits = json.loads((path.parent / "splits.json").read_text())["values"]
        if splits != protocol["value_splits"](spec, task["seed"]):
            raise ValueError("Saved split differs from the deterministic builder")
        binding = config["models"][task["model"]]
        if any(result["runtime"]["model"][k] != binding[k]
               for k in ("model_id", "revision", "chat", "fixed_layer")):
            raise ValueError("Model pin or prompt/site condition differs")
        for family in result.get("families", ["joint", "list"]):
            dev = result["baseline_metrics"][family]["dev"]
            passes = (dev["macro_factual"] >= config["minimum_baseline_accuracy"]
                      and dev["macro_parsed"] >= config["minimum_parse_coverage"])
            cells = [c for c in result["results"] if c["family"] == family]
            if not cells or any((c["status"] == "evaluated") != passes for c in cells):
                raise ValueError("Baseline qualification differs from the reported gate")


def calculate(data: Path, output: Path | None = None) -> list[dict]:
    here = Path(__file__).resolve().parent
    analysis = runpy.run_path(str(here / "independent_analysis.py"))
    protocol = runpy.run_path(str(here / "task_protocol.py"))
    manifest = json.loads((data / "protocol.json").read_text())
    summaries = json.loads((data / "summary.json").read_text())
    properties = list(protocol["PROPERTIES"])
    with tempfile.TemporaryDirectory(prefix="property-replay-") as directory:
        temporary = Path(directory)
        replay = {}
        for name, config_name, cells_name in (
            ("generic", "rikyu-v1.json", "generic_label_cells.json"),
            ("explicit", "rikyu-explicit-list-v2.json", "explicit_predicate_cells.json"),
        ):
            stream = data / f"{name}-records.jsonl.gz"
            if hashlib.sha256(stream.read_bytes()).hexdigest() != manifest["records"][stream.name]["sha256"]:
                raise ValueError(f"Reconstruction stream hash mismatch: {name}")
            root, result_dir = temporary / name / "results", temporary / name / "analysis"
            unpack(stream, root)
            config = manifest["configurations"][config_name]
            verify_protocol(root, config, protocol, manifest["execution_sources"][name]["registered_source_digest"])
            audit = analysis["analyze"](root, result_dir)
            cells = json.loads((result_dir / "cells.json").read_text())
            if cells != json.loads((data / cells_name).read_text()):
                raise ValueError(f"Independent per-split replay differs: {name}")
            rows = json.loads((result_dir / "analysis.json").read_text())
            expected = rows if name == "explicit" else [r for r in rows if r["family"] == "joint"]
            if not all(r in summaries for r in expected):
                raise ValueError(f"Recomputed intervals or summaries differ: {name}")
            replay[name] = {"audit": audit, "summaries": rows, "cells": cells}
            if output is not None:
                target = output / name
                target.mkdir(parents=True, exist_ok=True)
                for file in result_dir.glob("*.json"):
                    (target / file.name).write_bytes(file.read_bytes())
        selected = selective_properties(replay["explicit"]["cells"], properties, manifest["split_seeds"])
        rows = [r for r in replay["explicit"]["summaries"]
                if (r["model"], r["context"], r["family"], r["mode"])
                == ("qwen36_27b", "property", "list", "selected")]
        if len(rows) != 15:
            raise ValueError("Expected fifteen properties with qualified Qwen list splits")
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "selective-properties.json").write_text(json.dumps(selected, indent=2) + "\n")
            (output / "table6-rows.tex").write_text(table_rows(rows, selected))
            (output / "reconstruction.json").write_text(json.dumps({
                "status": "passed", "properties_evaluated": len(properties),
                "selective_count": len(selected), "selective_properties": selected,
                "records_replayed": sum(v["audit"]["unique_saved_generation_records"] for v in replay.values()),
            }, indent=2) + "\n")
        return rows



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = calculate(args.data_dir, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Reconstructed {len(rows)} property summaries, including intervals and development selections")


if __name__ == "__main__":
    main()
