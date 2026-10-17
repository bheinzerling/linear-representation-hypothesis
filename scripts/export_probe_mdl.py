"""Export compact, per-example evidence from the completed online-code run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run: Path, paper_data: Path, output: Path) -> None:
    source = json.loads((run / "run.json").read_text())
    identity = source["identity"]
    if (run / "DONE").read_text().strip() != source["config_hash"]:
        raise ValueError("Incomplete source run")
    inventory = json.loads((run / "inventory.json").read_text())
    for name, expected in inventory.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or digest(path) != expected:
            raise ValueError(f"Invalid source artifact: {name}")
    output.mkdir(parents=True, exist_ok=True)
    settings = identity["roster"]
    orders = identity["orders"]
    coders = identity["coders"]
    boundaries = identity["boundaries"]
    n = boundaries[-1]
    shape = (len(orders), len(settings), len(coders), n)
    bits = np.full(shape, np.log2(1000), dtype=np.float64)
    predictions = np.full(shape[:2] + (n,), -1, dtype=np.int16)
    temperatures = np.full(shape[:2] + (len(boundaries) - 1, len(coders)), np.inf)
    ordered = np.empty((len(orders), n), dtype=np.int16)
    dev_predictions, test_predictions = [], []
    dev_values = test_values = None
    for j, setting in enumerate(settings):
        setting_id = setting["id"]
        with np.load(run / f"full-{setting_id}.npz", allow_pickle=False) as full:
            if dev_values is None:
                dev_values = full["dev_values"].copy()
                test_values = full["test_values"].copy()
            np.testing.assert_array_equal(dev_values, full["dev_values"])
            np.testing.assert_array_equal(test_values, full["test_values"])
            dev_predictions.append(full["dev_predictions"].copy())
            test_predictions.append(full["test_predictions"].copy())
        for i, order in enumerate(orders):
            ordered[i] = json.loads((run / f"order-{order}.json").read_text())
            for k, (start, end) in enumerate(pairwise(boundaries)):
                path = run / f"order-{order}/prefix-{start:03d}/{setting_id}.npz"
                receipt = json.loads(path.with_suffix(".json").read_text())
                with np.load(path, allow_pickle=False) as block:
                    np.testing.assert_array_equal(block["values"], ordered[i, start:end])
                    predictions[i, j, start:end] = block["predictions"]
                    for c, coder in enumerate(coders):
                        bits[i, j, c, start:end] = block[f"{coder}_bits"]
                        temperature = receipt["temperatures"][c]
                        temperatures[i, j, k, c] = np.inf if temperature is None else temperature
    np.savez_compressed(
        output / "observations.npz",
        bits=bits,
        block_predictions=predictions,
        temperatures=temperatures,
        ordered_values=ordered,
        setting_ids=np.array([s["id"] for s in settings]),
        order_seeds=np.array(orders),
        coders=np.array(coders),
        boundaries=np.array(boundaries),
        dev_values=dev_values,
        test_values=test_values,
        dev_predictions=np.array(dev_predictions, dtype=np.int16),
        test_predictions=np.array(test_predictions, dtype=np.int16),
    )
    for name in ("summary.csv", "per_order_metrics.csv"):
        shutil.copyfile(paper_data / name, output / name)
    shutil.copyfile(run / "blocks.csv", output / "blocks.csv")
    protocol = {
        "source_config_hash": source["config_hash"],
        "source_commit": source["git_commit"],
        "identity": identity,
        "observation_axes": {
            "bits": ["order", "setting", "coder", "encoded_example"],
            "temperatures": ["order", "setting", "prefix", "coder"],
        },
        "source_inventory_sha256": digest(run / "inventory.json"),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    snapshot = json.loads((paper_data / "input_snapshot.json").read_text())
    snapshot["export"] = "scripts/export_probe_mdl.py"
    snapshot["files"] = {
        name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
        for name in (
            "observations.npz", "protocol.json", "blocks.csv", "summary.csv",
            "per_order_metrics.csv",
        )
    }
    (output / "input_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Exported {bits.size:,} per-example costs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paper-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/probe_mdl")
    args = parser.parse_args()
    export(args.run.resolve(), args.paper_data.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
