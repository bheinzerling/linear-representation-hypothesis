#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from safetensors import safe_open


MODELS = (
    "llama_3_2_1b",
    "llama_3_2_3b",
    "llama_3_8b",
    "olmo_3_7b_base",
    "olmo_2_13b",
    "gpt_oss_20b",
)
EXPECTED_METRICS_PER_MODEL = 5203
EXPECTED_DIRECTION_PAIRS_PER_MODEL = 220
EXPECTED_LAYER_COUNT = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and bundle the six-model prompt-transfer robustness suite.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-train-examples", type=int, default=800)
    parser.add_argument("--expected-test-examples", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_complete_run(root: Path, model_alias: str) -> Path:
    matches = [path for path in root.glob(f"{model_alias}__*") if (path / "DONE").is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one complete {model_alias} run, found {len(matches)}: {matches}")
    return matches[0]


def _gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def load_context_coverage(run_dir: Path, model_alias: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family_dir in sorted((run_dir / "aggregates").iterdir()):
        if not family_dir.is_dir():
            continue
        for partition in ("train", "dev", "test"):
            path = family_dir / f"{partition}.safetensors"
            with safe_open(str(path), framework="np") as handle:
                values = handle.get_tensor("values")
                all_counts = handle.get_tensor("all_context_counts")
                paired_counts = handle.get_tensor("paired_context_counts")
            rows.append(
                pd.DataFrame(
                    {
                        "model_alias": model_alias,
                        "family": family_dir.name,
                        "partition": partition,
                        "source_value": values.astype(int),
                        "all_context_count": all_counts.astype(int),
                        "seen_context_count": paired_counts.astype(int),
                        "heldout_context_count": paired_counts.astype(int),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _question_tables(metrics: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    new_values = metrics.loc[metrics["question"].eq("new_values")].copy()
    new_values_summary = (
        new_values.groupby(["model_alias", "architecture"], as_index=False)
        .agg(
            mean_mae=("mean_abs_error", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_accuracy=("accuracy", "mean"),
            family_layer_cells=("mean_abs_error", "size"),
        )
    )
    new_values_summary.to_csv(output_dir / "q1_new_values_summary.csv", index=False)
    counts["q1_new_values_summary"] = len(new_values_summary)

    new_contexts = metrics.loc[metrics["question"].eq("new_contexts_within_family")].copy()
    q2_keys = ["model_alias", "architecture", "train_family", "relative_layer"]
    q2 = new_contexts.pivot(index=q2_keys, columns="condition", values="mean_abs_error").reset_index()
    q2["heldout_minus_seen_mae"] = q2["heldout_contexts"] - q2["seen_contexts"]
    q2.to_csv(output_dir / "q2_new_contexts_paired.csv", index=False)
    counts["q2_new_contexts_paired"] = len(q2)

    affine = metrics.loc[metrics["question"].eq("complete_affine_cross_family")].copy()
    q3 = (
        affine.groupby(["model_alias", "architecture", "train_family", "test_family"], as_index=False)
        .agg(
            mean_over_layers_mae=("mean_abs_error", "mean"),
            mean_over_layers_accuracy=("accuracy", "mean"),
            layer_count=("relative_layer", "nunique"),
        )
    )
    q3["is_native_family"] = q3["train_family"].eq(q3["test_family"])
    q3.to_csv(output_dir / "q3_complete_affine_transfer_matrix.csv", index=False)
    counts["q3_complete_affine_transfer_matrix"] = len(q3)

    mean_shift = metrics.loc[metrics["question"].eq("family_mean_shift")].copy()
    q4_keys = ["model_alias", "architecture", "train_family", "test_family", "relative_layer"]
    q4 = mean_shift.pivot(index=q4_keys, columns="condition", values="mean_abs_error").reset_index()
    q4["is_native_family"] = q4["train_family"].eq(q4["test_family"])
    q4["raw_minus_aligned_mae"] = q4["raw"] - q4["family_mean_aligned"]
    q4.to_csv(output_dir / "q4_family_mean_alignment_paired.csv", index=False)
    counts["q4_family_mean_alignment_paired"] = len(q4)

    calibration = metrics.loc[metrics["question"].eq("direction_after_calibration")].copy()
    q5_keys = ["model_alias", "architecture", "train_family", "test_family", "relative_layer"]
    q5 = calibration.pivot(index=q5_keys, columns="condition", values="mean_abs_error").reset_index()
    q5["source_minus_intercept_mae"] = q5["complete_source_affine"] - q5["target_intercept"]
    q5["source_minus_gain_intercept_mae"] = q5["complete_source_affine"] - q5["target_gain_and_intercept"]
    q5["gain_intercept_minus_native_mae"] = q5["target_gain_and_intercept"] - q5["native_target_probe"]
    q5.to_csv(output_dir / "q5_direction_calibration_ladder.csv", index=False)
    counts["q5_direction_calibration_ladder"] = len(q5)

    pooled = metrics.loc[metrics["question"].eq("balanced_pooled_leave_one_family_out")].copy()
    pooled_fit = (
        pooled.loc[pooled["condition"].eq("balanced_pooled_probe")]
        .groupby(["model_alias", "architecture", "test_family", "relative_layer"], as_index=False)
        .agg(
            balanced_pooled_mae=("mean_abs_error", "mean"),
            balanced_pooled_mae_std=("mean_abs_error", "std"),
            assignment_replicates=("replicate", "nunique"),
        )
    )
    native = (
        pooled.loc[pooled["condition"].eq("native_target_probe")]
        .rename(columns={"mean_abs_error": "native_target_mae"})[
            ["model_alias", "architecture", "test_family", "relative_layer", "native_target_mae"]
        ]
    )
    q7 = pooled_fit.merge(native, on=["model_alias", "architecture", "test_family", "relative_layer"], validate="one_to_one")
    q7["pooled_minus_native_mae"] = q7["balanced_pooled_mae"] - q7["native_target_mae"]
    q7.to_csv(output_dir / "q7_balanced_pooled_leave_one_family_out.csv", index=False)
    counts["q7_balanced_pooled_leave_one_family_out"] = len(q7)

    global_centroid = metrics.loc[metrics["question"].eq("global_value_centroid_decodability")].copy()
    q8 = (
        global_centroid.groupby(["model_alias", "architecture"], as_index=False)
        .agg(
            mean_over_layers_mae=("mean_abs_error", "mean"),
            mean_over_layers_accuracy=("accuracy", "mean"),
            layer_count=("relative_layer", "nunique"),
        )
    )
    q8.to_csv(output_dir / "q8_global_value_centroid_summary.csv", index=False)
    counts["q8_global_value_centroid_summary"] = len(q8)
    return counts


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_parts: list[pd.DataFrame] = []
    direction_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    example_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for model_alias in MODELS:
        run_dir = find_complete_run(input_root, model_alias)
        analysis_dir = run_dir / "analysis"
        metrics_path = analysis_dir / "metrics.csv"
        directions_path = analysis_dir / "direction_pairs.csv"
        bootstrap_path = analysis_dir / "direction_bootstrap.csv"
        examples_path = analysis_dir / "prediction_examples.csv"
        metrics = pd.read_csv(metrics_path)
        directions = pd.read_csv(directions_path)
        bootstrap = pd.read_csv(bootstrap_path)
        examples = pd.read_csv(examples_path)
        if len(metrics) != EXPECTED_METRICS_PER_MODEL:
            raise ValueError(f"{model_alias}: expected {EXPECTED_METRICS_PER_MODEL} metrics, found {len(metrics)}.")
        if len(directions) != EXPECTED_DIRECTION_PAIRS_PER_MODEL:
            raise ValueError(
                f"{model_alias}: expected {EXPECTED_DIRECTION_PAIRS_PER_MODEL} direction pairs, found {len(directions)}."
            )
        if metrics["relative_layer"].nunique() != EXPECTED_LAYER_COUNT:
            raise ValueError(f"{model_alias}: incomplete relative-layer coverage.")
        if set(metrics["num_train_examples"].astype(int)) != {int(args.expected_train_examples)}:
            raise ValueError(
                f"{model_alias}: training counts are not exactly {int(args.expected_train_examples)}."
            )
        if set(metrics["num_test_examples"].astype(int)) != {int(args.expected_test_examples)}:
            raise ValueError(f"{model_alias}: test counts are not exactly {int(args.expected_test_examples)}.")
        if metrics["model_revision"].nunique() != 1:
            raise ValueError(f"{model_alias}: multiple model revisions are present.")
        metric_parts.append(metrics)
        direction_parts.append(directions)
        bootstrap_parts.append(bootstrap)
        example_parts.append(examples)
        coverage_parts.append(load_context_coverage(run_dir, model_alias))
        run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        model_metadata = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
        analysis_execution_path = run_dir / "analysis_execution.json"
        analysis_execution = (
            json.loads(analysis_execution_path.read_text(encoding="utf-8"))
            if analysis_execution_path.is_file()
            else {"policy": "shared_probe_fitter_with_intervention_decoder"}
        )
        run_rows.append(
            {
                "model_alias": model_alias,
                "run_id": run_dir.name,
                "model_revision": str(metrics["model_revision"].iloc[0]),
                "collection_host": model_metadata["collection_host"],
                "torch": model_metadata["torch"],
                "cuda": model_metadata["cuda"],
                "compute_dtype": model_metadata["compute_dtype"],
                "determinism_policy": run_payload["run_payload"].get(
                    "determinism_policy",
                    {"requested": True, "warn_only": False},
                ),
                "analysis_execution": analysis_execution,
                "config_sha256": run_payload["run_payload"]["config_sha256"],
                "metrics_sha256": sha256_file(metrics_path),
                "direction_pairs_sha256": sha256_file(directions_path),
            }
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    directions = pd.concat(direction_parts, ignore_index=True)
    bootstraps = pd.concat(bootstrap_parts, ignore_index=True)
    examples = pd.concat(example_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    _gzip_csv(metrics, output_dir / "metrics.csv.gz")
    _gzip_csv(directions, output_dir / "direction_pairs.csv.gz")
    bootstraps.to_csv(output_dir / "direction_bootstrap.csv", index=False)
    _gzip_csv(examples, output_dir / "prediction_examples.csv.gz")
    coverage_summary = (
        coverage.groupby(["family", "partition"], as_index=False)
        .agg(
            independent_value_count=("source_value", "nunique"),
            minimum_all_contexts=("all_context_count", "min"),
            mean_all_contexts=("all_context_count", "mean"),
            maximum_all_contexts=("all_context_count", "max"),
            minimum_paired_contexts=("seen_context_count", "min"),
            mean_paired_contexts=("seen_context_count", "mean"),
            maximum_paired_contexts=("seen_context_count", "max"),
        )
    )
    coverage_summary.to_csv(output_dir / "context_coverage_summary.csv", index=False)
    table_counts = _question_tables(metrics, output_dir)
    metadata = {
        "schema_version": 1,
        "models": list(MODELS),
        "metric_rows": int(len(metrics)),
        "direction_pair_rows": int(len(directions)),
        "direction_bootstrap_rows": int(len(bootstraps)),
        "prediction_example_rows": int(len(examples)),
        "relative_layers": sorted(float(value) for value in metrics["relative_layer"].unique()),
        "train_examples_per_probe": int(args.expected_train_examples),
        "test_examples_per_condition": int(args.expected_test_examples),
        "target_calibration_examples": int(args.expected_test_examples),
        "capacity_source": "single-context native-development selections from the 2026-04-20 sweep",
        "table_rows": table_counts,
        "runs": run_rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
