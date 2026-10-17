#!/usr/bin/env python3
"""Compute prefix-only online codelengths for all 73 Figure 3 probe settings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.probe_mdl import (CODERS, UNIFORM_BITS, PrefixFitter, block_boundaries,
                           calibrate_temperature, calibration_partition, label_bits)
from src.utils import atomic_write_json, atomic_write_text

STEM = "2026-10-17-figure3-probe-mdl"
DEFAULT_REPR = ROOT / "out/runs/source_value_rhs_addition_dataset_llama_3_1_8b_layer10__2026_06_15/representations/llama_3_1_8b/layer_10/source_token_representations.npz"
DEFAULT_SPLIT = ROOT / "configs/probe_mdl_split.json"
ROSTER = ROOT / "configs/probe_mdl_roster.json"
REFERENCE_CACHE_SHA256 = "42cf4af31df100654d0efebc27e0a82d375464fd1833e54b121e85cc9418092a"
SOURCES = [Path(__file__), ROOT / "src/probe_mdl.py", ROOT / "src/probes.py",
           ROOT / "src/activation_steering.py", ROSTER, DEFAULT_SPLIT, ROOT / "uv.lock"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_centroids(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        selected = data["style_ids"].astype(str) == "addition_formula_rhs_source"
        values = data["source_values"][selected].astype(int)
        activations = data["activations"][selected].astype(np.float32)
    np.testing.assert_array_equal(np.bincount(values, minlength=1000), np.full(1000, 20))
    centroids = np.stack([activations[values == value].mean(0) for value in range(1000)])
    metadata = json.loads(path.with_name("metadata.json").read_text())
    if metadata["model_name"] != "meta-llama/Llama-3.1-8B" or metadata["layer_idx"] != 10:
        raise ValueError("Expected Llama-3.1-8B layer-10 activations")
    if centroids.shape != (1000, 4096):
        raise ValueError("Unexpected centroid shape")
    return centroids.astype(np.float64), {
        "cache_sha256": sha256(path), "collection": metadata,
        "centroid_aggregation": "float32 arithmetic mean before float64 probe fitting",
    }


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "out/figure3-probe-mdl")
    p.add_argument("--repr-npz", type=Path, default=DEFAULT_REPR)
    p.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--allow-new-activations", action="store_true",
                   help="Permit a regenerated cache with a different hash; record it as a new run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--orders", default="1729,2718,31415")
    p.add_argument("--fit-seed", type=int, default=1729)
    p.add_argument("--architectures", default="linear,pls,circular_reg,circular_cls")
    p.add_argument("--controls", help="Optional comma-separated controls, for qualification only")
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def make_roster(args: argparse.Namespace) -> list[dict]:
    table = json.loads(ROSTER.read_text())
    rows = []
    for r in table:
        arch = r["architecture"]
        value = float(r["control_value"])
        if arch not in args.architectures.split(","):
            continue
        if args.controls and value not in list(map(float, args.controls.split(","))):
            continue
        cfg = dict(r["config"])
        if arch == "circular_cls":
            cfg["seed"] = args.fit_seed
        rows.append({"id": r["id"], "architecture": arch, "config": cfg,
                     "control_name": r["control_name"], "control_value": value,
                     "old_effective_df": float(r["old_effective_df"]),
                     "archived_test_mae": float(r["archived_test_mae"])})
    if not rows:
        raise ValueError("Empty sweep")
    return rows


def plan_run(args: argparse.Namespace) -> tuple[dict, Path]:
    roster = make_roster(args)
    activation_hash = sha256(args.repr_npz) if args.repr_npz.exists() else REFERENCE_CACHE_SHA256
    if not args.plan_only and not args.repr_npz.is_file():
        raise FileNotFoundError("Provide the regenerated activation cache with --repr-npz")
    if activation_hash != REFERENCE_CACHE_SHA256 and not args.allow_new_activations:
        raise ValueError("Activation cache differs; use --allow-new-activations for a new run")
    identity = {"protocol": "voita_eq3_prefix_only_v1", "roster": roster,
                "orders": list(map(int, args.orders.split(","))), "fit_seed": args.fit_seed,
                "device": args.device, "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES},
                "activation_sha256": activation_hash,
                "split_sha256": sha256(args.split_json), "roster_sha256": sha256(ROSTER),
                "boundaries": block_boundaries(800), "calibration_fraction": .2,
                "uniform_mixture": .01, "temperature_grid": "logspace(-6,6,49) plus infinity",
                "coders": list(CODERS), "blas_threads": 4,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "packages": {k: importlib.metadata.version(k) for k in
                             ("torch", "numpy", "scipy", "scikit-learn", "pandas")}}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"config_hash": digest, "identity": identity}, args.output / digest[:16]


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def block_complete(path: Path, receipt: Path) -> bool:
    if not path.exists() or not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    if saved["artifact_sha256"] != sha256(path):
        raise ValueError(f"Corrupted completed block: {path}")
    return True


def git_metadata(root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=root, stderr=subprocess.DEVNULL
        ))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def main() -> None:
    args = arguments()
    plan, run = plan_run(args)
    print(json.dumps({"run": str(run), "settings": len(plan["identity"]["roster"]),
                      "boundaries": plan["identity"]["boundaries"], "complete": (run / "DONE").exists()}), flush=True)
    if args.plan_only:
        return
    if (run / "DONE").exists():
        if (run / "DONE").read_text().strip() != plan["config_hash"]:
            raise ValueError("Completion marker does not match the plan")
        inventory = json.loads((run / "inventory.json").read_text())
        for name, expected in inventory.items():
            if sha256(run / name) != expected:
                raise ValueError(f"Completed run artifact changed: {name}")
        return
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device.startswith("cuda"):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
    run.mkdir(parents=True, exist_ok=True)
    oom_path = Path("/proc/self/oom_score_adj")
    provenance = {**plan, **git_metadata(ROOT), "host": platform.node(),
                  "python": sys.executable, "prefix": sys.prefix, "argv": sys.argv,
                  "oom_score_adj": oom_path.read_text().strip() if oom_path.exists() else None}
    atomic_write_json(run / "run.json", provenance)
    atomic_write_text(run / "command.txt", " ".join(sys.argv) + "\n")
    centroids, source = load_centroids(args.repr_npz)
    if source["cache_sha256"] != plan["identity"]["activation_sha256"]:
        raise ValueError("Activation cache changed")
    if plan["identity"]["split_sha256"] != "e0ab1eaf6db11396291155ad12f61b49b9260da63c6302709836159132930d20":
        raise ValueError("Unexpected split")
    atomic_write_json(run / "activation_source.json", source)
    split = json.loads(args.split_json.read_text())
    train_values = np.array(split["train_source_values"], dtype=int)
    test_values = np.array(split["test_source_values"], dtype=int)
    dev_values = np.array(split["dev_source_values"], dtype=int)
    if len(set(train_values) & set(test_values)) or len(set(train_values) & set(dev_values)):
        raise ValueError("Overlapping partitions")
    roster = plan["identity"]["roster"]
    start = time.monotonic()
    for order in plan["identity"]["orders"]:
        ordered = train_values[np.random.default_rng(order).permutation(len(train_values))]
        atomic_write_json(run / f"order-{order}.json", ordered.tolist())
        boundaries = plan["identity"]["boundaries"]
        for i, stop in enumerate(boundaries):
            if stop == len(ordered):
                break
            end = boundaries[i + 1]
            prefix, following = ordered[:stop], ordered[stop:end]
            if stop >= 3:
                fit_idx, cal_idx = calibration_partition(stop)
                calibration_fitter = PrefixFitter(centroids[prefix[fit_idx]], prefix[fit_idx], device=args.device, seed=args.fit_seed)
                full_fitter = PrefixFitter(centroids[prefix], prefix, device=args.device, seed=args.fit_seed)
            stage = run / f"order-{order}" / f"prefix-{stop:03d}"
            stage.mkdir(parents=True, exist_ok=True)
            for setting in roster:
                target = stage / (setting["id"] + ".npz")
                receipt = target.with_suffix(".json")
                if block_complete(target, receipt):
                    continue
                arch, cfg = setting["architecture"], setting["config"]
                if stop < 3:
                    native_bits = ordinal_bits = np.full(len(following), UNIFORM_BITS)
                    predictions = np.full(len(following), -1)
                    temperatures = [np.inf, np.inf]
                    metadata = {"fallback": "uniform: fewer than two fit examples plus one calibration example"}
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        calibration_probe = calibration_fitter.fit(arch, cfg)
                        cal_native, cal_ordinal, _ = calibration_probe.outputs(centroids[prefix[cal_idx]])
                        temperatures = [calibrate_temperature(scores, prefix[cal_idx]) for scores in (cal_native, cal_ordinal)]
                        probe = full_fitter.fit(arch, cfg)
                        native, ordinal, predictions = probe.outputs(centroids[following])
                        native_bits = label_bits(native, following, temperatures[0])
                        ordinal_bits = label_bits(ordinal, following, temperatures[1])
                        metadata = {"fit": probe.metadata, "calibration_fit": calibration_probe.metadata,
                                    "warnings": sorted({str(w.message) for w in caught})}
                save_npz(target, values=following, native_bits=native_bits, ordinal_bits=ordinal_bits,
                         predictions=predictions)
                atomic_write_json(receipt, {"order": order, "prefix_size": stop, "end": end,
                                  "setting_id": setting["id"], "temperatures":
                                  [None if np.isinf(t) else t for t in temperatures],
                                  "metadata": metadata, "artifact_sha256": sha256(target)})
            print(f"[coded] order={order} prefix={stop} block={len(following)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    # Full-data predictions are a separate evaluation; MDL never consumes dev/test labels.
    full_fitter = PrefixFitter(centroids[train_values], train_values, device=args.device, seed=args.fit_seed)
    for setting in roster:
        target = run / ("full-" + setting["id"] + ".npz")
        if target.exists():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = full_fitter.fit(setting["architecture"], setting["config"])
            _, _, predictions = probe.outputs(centroids[np.r_[dev_values, test_values]])
        save_npz(target, dev_values=dev_values, test_values=test_values,
                 dev_predictions=predictions[:len(dev_values)], test_predictions=predictions[len(dev_values):])
    records, blocks = [], []
    for order in plan["identity"]["orders"]:
        for setting in roster:
            for coder in CODERS:
                total = boundaries[0] * UNIFORM_BITS
                blocks.append({"order": order, "setting_id": setting["id"], "coder": coder,
                               "prefix_size": 0, "block_size": boundaries[0], "block_bits": total})
                for stop, end in zip(boundaries[:-1], boundaries[1:]):
                    path = run / f"order-{order}" / f"prefix-{stop:03d}" / (setting["id"] + ".npz")
                    with np.load(path, allow_pickle=False) as data:
                        bits = data[coder + "_bits"]
                        if len(bits) != end - stop or not np.isfinite(bits).all() or np.any(bits < 0):
                            raise ValueError(f"Invalid codelengths: {path}")
                    total += float(bits.sum())
                    blocks.append({"order": order, "setting_id": setting["id"], "coder": coder,
                                   "prefix_size": stop, "block_size": end-stop, "block_bits": float(bits.sum())})
                with np.load(run / ("full-" + setting["id"] + ".npz"), allow_pickle=False) as full:
                    mae = float(np.mean(np.abs(full["test_predictions"] - full["test_values"])))
                    dev_mae = float(np.mean(np.abs(full["dev_predictions"] - full["dev_values"])))
                    accuracy = float(np.mean(full["test_predictions"] == full["test_values"]))
                records.append({k: v for k, v in setting.items() if k != "config"} | {
                    "order": order, "coder": coder, "online_bits": total, "bits_per_value": total/800,
                    "uniform_bits": 800*UNIFORM_BITS, "compression": 800*UNIFORM_BITS/total,
                    "test_mae": mae, "dev_mae": dev_mae, "test_accuracy": accuracy})
    pd.DataFrame(records).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(blocks).to_csv(run / "blocks.csv", index=False)
    inventory = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                 if p.is_file() and p.name not in {"DONE", "inventory.json"}}
    atomic_write_json(run / "inventory.json", inventory)
    atomic_write_json(run / "metrics.json", {"rows": len(records), "settings": len(roster),
                      "orders": plan["identity"]["orders"], "seconds": time.monotonic()-start,
                      "input_labels_encoded": 800, "test_values": 100, "dev_values": 100})
    atomic_write_text(run / "DONE", plan["config_hash"] + "\n")
    print(f"[complete] {run}", flush=True)


if __name__ == "__main__":
    main()
