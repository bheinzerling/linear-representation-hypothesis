#!/usr/bin/env python3
"""Validate the complete deletion grid and export paired, value-clustered results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rank41_deletion import TASKS, conditions, digest, file_sha256, stable_seed
from src.utils import atomic_write_csv, atomic_write_json

STEM = "2026-09-15-rank41-deletion"
ARTIFACTS = ROOT / "experiment-reports/artifacts" / STEM


def cluster_bootstrap(frame, columns, *, seed, n_boot=2000):
    """Resample source values, keeping all their repeated split appearances.

    The point estimate weights prompts equally. Bootstrap multiplicities apply
    to every appearance of a value, with a ratio estimator for each resample.
    This is conditional on the ten fitted bases, not a training-set bootstrap.
    """
    clustered = frame.groupby("source_value", sort=True)[columns].agg(["sum", "count"])
    n = len(clustered)
    counts = clustered[(columns[0], "count")].to_numpy(dtype=float)
    sums = np.column_stack([clustered[(c, "sum")].to_numpy(dtype=float) for c in columns])
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot)
    boot = (weights @ sums) / (weights @ counts)[:, None]
    point = sums.sum(0) / counts.sum()
    limits = np.quantile(boot, [0.025, 0.975], axis=0)
    return {name: {"mean": float(point[j]), "low": float(limits[0, j]), "high": float(limits[1, j])}
            for j, name in enumerate(columns)}


def aggregate(detail):
    metrics = ["baseline_correct", "intervention_correct", "top1_agreement", "target_logprob_delta",
               "kl_original_to_intervention", "delta_l2", "delta_energy_fraction", "centered_energy"]
    units = detail.groupby(["split", "task_id", "source_value", "family"], as_index=False)[metrics].mean()
    units["accuracy_change"] = units.intervention_correct - units.baseline_correct
    records = []
    for (task, family), frame in units.groupby(["task_id", "family"], sort=True):
        stats = cluster_bootstrap(frame, ["baseline_correct", "intervention_correct", "accuracy_change"],
                                  seed=stable_seed(task, "bootstrap"))
        by_split = frame.groupby("split").accuracy_change.mean()
        baseline_n = int(frame.baseline_correct.sum())
        row = {"task_id": task, "family": family, "n_prompts": len(frame),
               "n_unique_values": frame.source_value.nunique(), "n_splits": frame.split.nunique(),
               "baseline_correct_n": baseline_n,
               "accuracy_on_baseline_correct": float((frame.intervention_correct * frame.baseline_correct).sum() / baseline_n),
               "top1_change_rate": float(1 - frame.top1_agreement.mean()),
               "lost_correct_n": float((frame.baseline_correct * (1 - frame.intervention_correct)).sum()),
               "gained_correct_n": float(((1 - frame.baseline_correct) * frame.intervention_correct).sum()),
               "split_change_min": by_split.min(), "split_change_max": by_split.max(),
               "mean_kl": frame.kl_original_to_intervention.mean(),
               "mean_logprob_change": frame.target_logprob_delta.mean(),
               "mean_delta_l2": frame.delta_l2.mean(),
               "mean_delta_energy_fraction": frame.delta_energy_fraction.mean()}
        for key, name in (("baseline_correct", "baseline_accuracy"),
                          ("intervention_correct", "accuracy"), ("accuracy_change", "accuracy_change")):
            for stat, value in stats[key].items():
                row[name if stat == "mean" else f"{name}_{stat}"] = value
        records.append(row)
    summary = pd.DataFrame(records)
    # Paired property-minus-control differences on exactly the same prompts.
    wide = units.pivot(index=["split", "task_id", "source_value"], columns="family", values="intervention_correct").reset_index()
    contrast_rows = []
    controls = [c for c in wide.columns if c.startswith("delete_") and c != "delete_property"] + ["matched_noise"]
    for task, frame in wide.groupby("task_id"):
        frame = frame.copy()
        for control in controls:
            frame["difference"] = frame.delete_property - frame[control]
            stats = cluster_bootstrap(frame, ["difference"], seed=stable_seed(task, "contrast"))["difference"]
            contrast_rows.append({"task_id": task, "control": control, **stats})
    return summary, units, pd.DataFrame(contrast_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "out/rank41_deletion_inputs/20260915_v1")
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    # CLI examples use repository-relative paths; normalize before provenance.
    args.runs = [p.resolve() for p in args.runs]
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    frames, provenances, parts_sha = [], [], {}
    invariant_configs = []
    for run in args.runs:
        if not (run / "DONE").exists():
            raise ValueError(f"Run incomplete: {run}")
        prov = json.loads((run / "run.json").read_text())
        cfg = dict(prov["config"])
        splits = cfg.pop("splits")
        if cfg["max_examples"] is not None or cfg["bundle_hash"] != manifest["bundle_hash"]:
            raise ValueError("Smoke run or incompatible input bundle")
        invariant_configs.append(cfg)
        provenances.append(prov)
        paths = [run / "parts" / f"{split}__{task}__{c.name}.csv"
                 for split in splits for task in TASKS for c in conditions()]
        if len(list((run / "parts").glob("*.csv"))) != len(paths):
            raise ValueError("Unexpected condition files")
        for path in paths:
            frame = pd.read_csv(path, keep_default_na=False)
            if len(frame) != 100 or frame.source_value.duplicated().any():
                raise ValueError(f"Unexpected unit count: {path}")
            parts_sha[str(path.relative_to(ROOT))] = file_sha256(path)
            frames.append(frame)
    if any(c != invariant_configs[0] for c in invariant_configs[1:]):
        raise ValueError("Mismatched shard configurations")
    if any(p["packages"] != provenances[0]["packages"] for p in provenances[1:]):
        raise ValueError("Mismatched software versions")
    detail = pd.concat(frames, ignore_index=True)
    expected_n = 10 * 7 * 14 * 100
    if len(detail) != expected_n or detail.duplicated(["split", "task_id", "source_value", "condition"]).any():
        raise ValueError("Incomplete or duplicate cross-shard grid")
    expected_splits = {f"split_{i:02d}" for i in range(10)}
    if set(detail.split) != expected_splits or set(detail.task_id) != set(TASKS):
        raise ValueError("Incomplete task/split inventory")
    unit_keys = ["split", "task_id", "source_value"]
    if (detail.groupby(unit_keys).baseline_token_id.nunique() != 1).any():
        raise ValueError("Baseline drift across paired conditions")
    original = detail.loc[detail.condition == "original"]
    if not original.top1_agreement.all() or original.kl_original_to_intervention.max() != 0:
        raise ValueError("Zero-delta identity failed")
    prompt_rows = []
    for split in sorted(expected_splits):
        f = pd.read_json(args.bundle / f"{split}.prompts.jsonl", lines=True)
        prompt_rows.append(f)
    prompts = pd.concat(prompt_rows, ignore_index=True)
    check = original.merge(prompts, on=unit_keys, suffixes=("_actual", "_expected"), validate="one_to_one")
    for column in ("prompt", "token_idx", "target_token_id", "source_token_id"):
        if not (check[f"{column}_actual"] == check[f"{column}_expected"]).all():
            raise ValueError(f"Prompt identity mismatch in {column}")
    summary, units, contrasts = aggregate(detail)
    control_diagnostics = pd.concat(pd.read_csv(p) for p in sorted(args.bundle.glob("*.controls.csv")))
    atomic_write_csv(args.output / "summary.csv", summary)
    atomic_write_csv(args.output / "paired_control_contrasts.csv", contrasts)
    atomic_write_csv(args.output / "control_diagnostics.csv", control_diagnostics)
    # Small derived data boundary for plotting. Raw prompts/predictions stay in the run artifacts.
    per_split = units.groupby(["split", "task_id", "family"], as_index=False).agg(
        accuracy=("intervention_correct", "mean"), baseline_accuracy=("baseline_correct", "mean"),
        accuracy_change=("accuracy_change", "mean"), mean_kl=("kl_original_to_intervention", "mean"),
        mean_delta_energy_fraction=("delta_energy_fraction", "mean"))
    atomic_write_csv(args.output / "per_split.csv", per_split)
    # Reproducible random examples plus the first lost and first preserved correct answer per task.
    example_parts = []
    deletion = detail.loc[detail.condition == "delete_property"].copy()
    for task, frame in deletion.groupby("task_id"):
        example_parts.append(frame.sample(n=2, random_state=stable_seed(task, "examples")).assign(selection="random"))
        for correct, label in ((False, "first lost correct answer"), (True, "first preserved correct answer")):
            selected = frame.loc[frame.baseline_correct & (frame.intervention_correct == correct)]
            if len(selected):
                example_parts.append(selected.sort_values(["split", "source_value"]).head(1).assign(selection=label))
    atomic_write_csv(args.output / "examples.csv", pd.concat(example_parts, ignore_index=True))
    validation_audits = {}
    for name in ("audit_batch100", "cross_host_audit", "historical_audit_00_04"):
        path = args.bundle.parent / f"{name}.json"
        if path.exists():
            validation_audits[name] = {"sha256": file_sha256(path), "result": json.loads(path.read_text())}
    audit = {
        "schema": 1, "bundle_hash": manifest["bundle_hash"], "runs": [str(r.relative_to(ROOT)) for r in args.runs],
        "run_provenance": provenances, "part_hashes": parts_sha,
        "n_rows": len(detail), "n_prompt_units": len(original),
        "n_unique_source_values": int(original.source_value.nunique()),
        "condition_inventory": [asdict(c) for c in conditions()],
        "bootstrap": {"replicates": 2000, "cluster": "source_value", "unit_weight": "one prompt",
                      "control_seed_reduction": "mean within each split/task/value before bootstrap",
                      "interpretation": "conditional on the ten frozen fitted bases; no training-set uncertainty"},
        "checks": {"complete_grid": True, "matched_configs": True, "matched_packages": True,
                   "baseline_consistency": True, "input_identity": True, "zero_delta_identity": True},
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "validation_audits": validation_audits,
    }
    atomic_write_json(args.output / "provenance.json", audit)
    print(summary.pivot(index="task_id", columns="family", values="accuracy").round(4).to_string())
    print(json.dumps({k: v for k, v in audit.items() if k in ("n_rows", "n_prompt_units", "n_unique_source_values", "checks")}))


if __name__ == "__main__":
    main()
