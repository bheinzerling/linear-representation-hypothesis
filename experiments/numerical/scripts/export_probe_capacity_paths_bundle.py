#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import atomic_write_json, atomic_write_jsonl, ensure_dir, runtime_metadata


DEFAULT_SOURCE_ARTIFACT_DIR = (
    ROOT
    / "experiment-reports"
    / "artifacts"
    / "2026-04-20-source-value-probe-capacity-sweep-split00-relative-layers-capacity-dev"
)
DEFAULT_OUTPUT_DIR = ROOT.parent / "papers" / "wip" / "num-repr-steering" / "figures" / "data" / "probe_capacity_paths"
MODEL_ORDER = [
    "llama_3_2_1b",
    "llama_3_2_3b",
    "llama_3_8b",
    "olmo_3_7b_base",
    "olmo_2_13b",
    "gpt_oss_20b",
]
POINTS_FILENAME = "points.jsonl"
BASELINE_FILENAME = "baseline_by_model.jsonl"
METADATA_FILENAME = "metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a paper-facing data bundle for the probe-capacity paths figure."
    )
    parser.add_argument("--source-artifact-dir", default=str(DEFAULT_SOURCE_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _build_points_bundle(source_artifact_dir: Path) -> list[dict[str, object]]:
    points = pd.read_json(source_artifact_dir / "pareto_all_points.jsonl", lines=True)
    points = points.loc[points["model_alias"].isin(MODEL_ORDER)].copy()
    points = points.loc[
        :,
        [
            "model_alias",
            "model_dim",
            "architecture",
            "relative_layer",
            "effective_df",
            "mean_abs_error_mean",
        ],
    ].copy()
    points["mean_abs_error_mean"] = points["mean_abs_error_mean"].clip(lower=1e-3)
    points = points.sort_values(["model_alias", "architecture", "relative_layer", "effective_df"]).reset_index(drop=True)
    return points.to_dict("records")


def _build_baseline_bundle(source_artifact_dir: Path) -> list[dict[str, object]]:
    baseline = pd.read_json(source_artifact_dir / "baseline_reference_by_model.jsonl", lines=True)
    baseline = baseline.loc[baseline["model_alias"].isin(MODEL_ORDER)].copy()
    baseline = baseline.loc[:, ["model_alias", "mean_abs_error"]].copy()
    baseline["mean_abs_error"] = baseline["mean_abs_error"].clip(lower=1e-3)
    baseline = baseline.sort_values(["model_alias"]).reset_index(drop=True)
    return baseline.to_dict("records")


def main() -> None:
    args = parse_args()
    source_artifact_dir = Path(args.source_artifact_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    points_rows = _build_points_bundle(source_artifact_dir)
    baseline_rows = _build_baseline_bundle(source_artifact_dir)

    atomic_write_jsonl(output_dir / POINTS_FILENAME, points_rows)
    atomic_write_jsonl(output_dir / BASELINE_FILENAME, baseline_rows)
    atomic_write_json(
        output_dir / METADATA_FILENAME,
        {
            "bundle_version": 1,
            "figure_id": "fig_probe_capacity_paths",
            "source_artifact_dir": str(source_artifact_dir),
            "source_files": {
                "points": "pareto_all_points.jsonl",
                "baseline_by_model": "baseline_reference_by_model.jsonl",
            },
            "bundle_files": {
                "points": POINTS_FILENAME,
                "baseline_by_model": BASELINE_FILENAME,
                "metadata": METADATA_FILENAME,
            },
            "model_order": MODEL_ORDER,
            "fields": {
                "points": [
                    "model_alias",
                    "model_dim",
                    "architecture",
                    "relative_layer",
                    "effective_df",
                    "mean_abs_error_mean",
                ],
                "baseline_by_model": ["model_alias", "mean_abs_error"],
            },
            "runtime": runtime_metadata(),
        },
    )


if __name__ == "__main__":
    main()
