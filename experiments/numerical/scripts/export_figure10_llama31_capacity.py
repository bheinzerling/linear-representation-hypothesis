#!/usr/bin/env python3
"""Validate the replacement sweep and export Figure 10's frozen plotting bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_source_value_probe_capacity_relative_layers_dev_report import (
    _prepare_baseline_reference, _prepare_mean_curves,
)
from src.utils import atomic_write_json, atomic_write_jsonl


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, source_hashes = [], {}
    for worker in manifest["workers"]:
        run = ROOT / worker["run_dir"]
        if not (run / "DONE").exists():
            raise RuntimeError(f"Incomplete worker: {worker['id']}")
        path = run / "summary_metrics.jsonl"
        rows.extend(records(path))
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    frame = pd.DataFrame(rows)
    assert set(frame.model_alias) == {"llama_3_1_8b"}
    assert set(frame.model_dim) == {4096}
    assert set(frame.split_name) == {"split_00"}
    keys = ["model_alias", "family", "relative_layer", "architecture", "step_index", "partition"]
    assert not frame.duplicated(keys).any(), "Duplicate evaluations"
    assert len(frame) == 55 * 73 * 3, f"Unexpected evaluation count: {len(frame)}"
    expected_counts = {"linear": 13, "pls": 11, "circular_reg": 24, "circular_cls": 24, "mean_baseline": 1}
    families = {family["family"] for family in manifest["prompt_families"]}
    assert set(frame.family) == families
    for family in families:
        for relative, absolute in zip(manifest["relative_layers"], manifest["absolute_layers"], strict=True):
            condition = frame[(frame.family == family) & np.isclose(frame.relative_layer, relative)]
            assert len(condition) == 219, (family, relative, len(condition))
            assert set(condition.layer_idx) == {absolute}
            for partition, count in manifest["partitions"].items():
                subset = condition[condition.partition == partition]
                assert set(subset.num_examples) == {count}
                assert subset.groupby("architecture").size().to_dict() == expected_counts
    assert np.isfinite(frame.mean_abs_error).all()
    assert frame.mean_abs_error.between(0, 999).all()
    assert frame.accuracy.between(0, 1).all()
    means = _prepare_mean_curves(frame)
    assert len(means) == 792 and set(means.num_prompt_styles) == {5}
    fields = ["model_alias", "model_dim", "architecture", "relative_layer", "effective_df", "mean_abs_error_mean"]
    # _build_frontier in the original diagnostic report exported effective_df_max.
    # Its separate mean(log2 DF) field was not used by the paper renderer.
    replacement = means.rename(columns={"effective_df_max": "effective_df"})[fields].to_dict("records")
    _, new_baseline = _prepare_baseline_reference(frame)
    original_points = records(args.original_bundle / "points.jsonl")
    original_baseline = records(args.original_bundle / "baseline_by_model.jsonl")
    assert "llama_3_8b" in {row["model_alias"] for row in original_points}
    retained = [row for row in original_points if row["model_alias"] != "llama_3_8b"]
    combined = retained + replacement
    baselines = [row for row in original_baseline if row["model_alias"] != "llama_3_8b"]
    baselines += new_baseline[["model_alias", "mean_abs_error"]].to_dict("records")
    atomic_write_jsonl(output / "points.jsonl", combined)
    atomic_write_jsonl(output / "replacement_points.jsonl", replacement)
    atomic_write_jsonl(output / "baseline_by_model.jsonl", baselines)
    # Check serialized output, including all original floating-point values.
    reloaded_retained = [row for row in records(output / "points.jsonl") if row["model_alias"] != "llama_3_1_8b"]
    assert reloaded_retained == retained
    assert len({row["model_alias"] for row in combined}) == 6
    atomic_write_jsonl(args.manifest.parent / "summary_metrics.jsonl", rows)
    atomic_write_jsonl(args.manifest.parent / "mean_curves.jsonl", means.to_dict("records"))
    combined_df = pd.DataFrame(combined)
    # Match the original first-minimum rule in ascending sweep-step order.
    # Ridge runs from unregularized (largest DF) to most regularized.
    combined_df["capacity_order"] = np.where(
        combined_df.architecture == "linear", -combined_df.effective_df, combined_df.effective_df,
    )
    best = combined_df.sort_values(["mean_abs_error_mean", "capacity_order"]).groupby(
        ["model_alias", "architecture", "relative_layer"], as_index=False
    ).head(1)
    migration_checks = {}
    for worker in manifest["workers"]:
        before_path = args.manifest.parent / f"{worker['id']}_before_migration.jsonl"
        if not before_path.exists():
            continue
        old_rows = records(before_path)
        assert sha256(before_path) == worker["segments"][0]["summary_sha256"]
        current_rows = {tuple(row[key] for key in keys): row for row in rows}
        for old_row in old_rows:
            current = current_rows[tuple(old_row[key] for key in keys)]
            assert json.dumps(current, sort_keys=True) == json.dumps(old_row, sort_keys=True), "Migration changed an existing evaluation"
        migration_checks[worker["id"]] = {"unchanged_evaluation_rows": len(old_rows)}
    optima = best.groupby("architecture").agg(
        median_best_mae=("mean_abs_error_mean", "median"),
        median_best_df=("effective_df", "median"),
    ).reset_index().to_dict("records")
    atomic_write_json(args.manifest.parent / "validation.json", {
        "status": "passed", "conditions": 55, "probe_fits": 3960, "evaluation_rows": len(rows),
        "replacement_plot_points": len(replacement), "retained_plot_points": len(retained),
        "other_model_points_identical": True, "six_model_median_optima": optima,
        "migration_checks": migration_checks,
        "llama31_median_optima": best[best.model_alias == "llama_3_1_8b"].groupby("architecture").agg(
            median_best_mae=("mean_abs_error_mean", "median"), median_best_df=("effective_df", "median"),
        ).reset_index().to_dict("records"),
    })
    atomic_write_json(output / "metadata.json", {
        "figure_id": "fig_probe_capacity_paths", "bundle_version": 2,
        "study": manifest["study"], "checkpoint": manifest["checkpoint"],
        "model_order": ["llama_3_2_1b", "llama_3_2_3b", "llama_3_1_8b", "olmo_3_7b_base", "olmo_2_13b", "gpt_oss_20b"],
        "aggregation": {"y": "arithmetic mean of test MAE over five prompt families",
                        "x": "maximum effective DF across the same five families at each matched capacity step",
                        "layers": manifest["relative_layers"], "capacity_selection": "all tested capacities retained"},
        "source_summary_sha256": source_hashes,
        "original_bundle_sha256": {name: sha256(args.original_bundle / name) for name in ["points.jsonl", "baseline_by_model.jsonl", "metadata.json"]},
        "bundle_sha256": {name: sha256(output / name) for name in ["points.jsonl", "baseline_by_model.jsonl", "replacement_points.jsonl"]},
        "comparison_notes": manifest["comparison_notes"],
        "export_command": " ".join(sys.argv),
    })
    print(json.dumps({"output": str(output), "fits": 3960, "six_model_median_optima": optima}, indent=2))


if __name__ == "__main__":
    main()
