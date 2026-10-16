#!/usr/bin/env python3
"""Evaluate oracle-to-predicted owner-block replacements in the Figure 1 span."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_figure1_owner_block_prefix_steering import (  # noqa: E402
    BEST_SPAN_RUN,
    COMPETITIVE_RUN,
    DISCOVERY_ROWS,
    PANEL_KEY_COLUMNS,
    SOURCE_RUN,
    _array_hash,
    _evaluate_coefficients,
    _load_hidden,
    _load_model_context,
    _panel_hash,
)
from src.number_property_probes import property_labels  # noqa: E402
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.oracle_block_replacement import (  # noqa: E402
    OwnerBlock,
    benjamini_hochberg,
    build_owner_blocks,
    build_replacement_conditions,
    cluster_bootstrap_mean,
    cluster_means,
    interaction_contrast,
    replace_owner_blocks,
)
from src.property_attribution import direction_owners_from_diagnostics  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    load_yaml,
    runtime_metadata,
    set_global_seed,
)


OWNER_PREFIX_RUN = (
    ROOT / "out/figure1_owner_block_prefix_steering/llama31_8b_layer10_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "out/figure1_oracle_block_replacement/llama31_8b_layer10_v1"
)
DESIGN_PATH = (
    ROOT
    / "experiment-designs/2026-07-29-oracle-to-predicted-block-replacement.md"
)
DEFAULT_SEED = 20260729
BOOTSTRAP_DRAWS = 4000
NLL_FLOOR = 1e-12
REQUIRED_CUBLAS = ":4096:8"
COMPACT_COLUMNS = (
    "condition",
    "panel_row_idx",
    "baseline_row_idx",
    "target_slot",
    "source_value",
    "target_value",
    "delta_value",
    "distance_bucket",
    "predicted_token_id",
    "predicted_text",
    "predicted_value",
    "expected_value",
    "numeric_mass",
    "target_prob",
    "correct",
    "target_error",
    "delta_norm",
    "source_fisher_quadratic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run oracle-to-predicted owner-block replacement interactions."
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--competitive-run-dir", type=Path, default=COMPETITIVE_RUN)
    parser.add_argument("--best-span-run-dir", type=Path, default=BEST_SPAN_RUN)
    parser.add_argument("--owner-prefix-run-dir", type=Path, default=OWNER_PREFIX_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "evaluate", "analyze"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_owners(args: argparse.Namespace, rank: int) -> tuple[str, ...]:
    diagnostics = pd.read_csv(
        args.competitive_run_dir / "basis_training_diagnostics.csv"
    )
    return direction_owners_from_diagnostics(
        diagnostics,
        split="split_00",
        expected_rank=int(rank),
    )


def _block_rows(blocks: Sequence[OwnerBlock]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_index": block_index,
                "property_id": block.property_id,
                "owner_rank": block.rank,
                "coordinate_indices": ",".join(
                    str(value) for value in block.coordinate_indices
                ),
            }
            for block_index, block in enumerate(blocks)
        ]
    )


def _condition_rows(blocks: Sequence[OwnerBlock]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition_index": condition_index,
                "condition_id": condition.condition_id,
                "condition_kind": condition.condition_kind,
                "replaced_blocks": ",".join(condition.replaced_blocks),
                "num_replaced_blocks": len(condition.replaced_blocks),
                "replaced_rank": sum(
                    block.rank
                    for block in blocks
                    if block.property_id in condition.replaced_blocks
                ),
            }
            for condition_index, condition in enumerate(
                build_replacement_conditions(blocks)
            )
        ]
    )


def prepare(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    prepared_path = output_dir / "replacement_coefficients.npz"
    if prepared_path.exists() and not args.force:
        print(prepared_path)
        return
    panel = pd.read_csv(args.owner_prefix_run_dir / "panel.csv", low_memory=False)
    if args.max_rows is not None:
        panel = panel.head(int(args.max_rows)).copy()
    with np.load(args.owner_prefix_run_dir / "prepared.npz") as source:
        basis = np.asarray(source["competitive_basis"], dtype=np.float64)
        exact_patch = np.asarray(source["exact_patch"], dtype=np.float64)[: len(panel)]
        magnitude = np.asarray(
            source["magnitude_coefficients"], dtype=np.float64
        )[: len(panel)]
        source_keys = tuple(source.files)
        source_arrays = {
            key: np.asarray(source[key], dtype=np.float64)[: len(panel)]
            for key in source_keys
            if key.startswith("property__")
        }
    owners = _load_owners(args, basis.shape[1])
    blocks = build_owner_blocks(owners)
    if sum(block.rank for block in blocks) != basis.shape[1]:
        raise ValueError("Nonempty owner blocks do not partition the competitive rank.")
    oracle = exact_patch @ basis
    predicted_by_block: dict[str, np.ndarray] = {}
    for block in blocks:
        predicted = np.zeros_like(oracle)
        indices = np.asarray(block.coordinate_indices, dtype=np.int64)
        if block.property_id == "magnitude":
            predicted[:, indices] = magnitude[:, indices]
        else:
            key = f"property__{block.property_id}"
            if key not in source_arrays:
                raise KeyError(f"Missing prepared label prediction {key!r}.")
            predicted[:, indices] = source_arrays[key][:, indices]
        predicted_by_block[block.property_id] = predicted
    all_replaced = replace_owner_blocks(
        oracle,
        predicted_by_block,
        {block.property_id: block for block in blocks},
        tuple(block.property_id for block in blocks),
    )
    arrays: dict[str, np.ndarray] = {
        "competitive_basis": basis.astype(np.float32),
        "oracle": oracle.astype(np.float32),
        "all_replaced": all_replaced.astype(np.float32),
    }
    for property_id, predicted in predicted_by_block.items():
        arrays[f"predicted__{property_id}"] = predicted.astype(np.float32)
    _atomic_savez(prepared_path, **arrays)
    block_frame = _block_rows(blocks)
    condition_frame = _condition_rows(blocks)
    if len(block_frame) != 19 or len(condition_frame) != 192:
        raise ValueError(
            f"Expected 19 blocks and 192 conditions; got "
            f"{len(block_frame)} and {len(condition_frame)}."
        )
    atomic_write_csv(output_dir / "panel.csv", panel)
    atomic_write_csv(output_dir / "owner_blocks.csv", block_frame)
    atomic_write_csv(output_dir / "conditions.csv", condition_frame)
    metadata = {
        "schema_version": 1,
        "seed": DEFAULT_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "nll_floor": NLL_FLOOR,
        "required_cublas_workspace_config": REQUIRED_CUBLAS,
        "panel_rows": len(panel),
        "discovery_rows": min(DISCOVERY_ROWS, len(panel)),
        "confirmatory_rows": max(0, len(panel) - DISCOVERY_ROWS),
        "panel_hash": _panel_hash(panel),
        "competitive_basis_hash": _array_hash(basis.astype(np.float32)),
        "source_prepared_sha256": _sha256(
            args.owner_prefix_run_dir / "prepared.npz"
        ),
        "source_panel_sha256": _sha256(args.owner_prefix_run_dir / "panel.csv"),
        "design_sha256": _sha256(DESIGN_PATH) if DESIGN_PATH.exists() else None,
        "runner_sha256": _sha256(Path(__file__)),
        "source_run": str(args.source_run_dir.resolve()),
        "competitive_run": str(args.competitive_run_dir.resolve()),
        "best_span_run": str(args.best_span_run_dir.resolve()),
        "owner_prefix_run": str(args.owner_prefix_run_dir.resolve()),
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "prepare_metadata.json", metadata)
    print(prepared_path)


def _parse_blocks(raw: Any) -> tuple[str, ...]:
    text = str(raw)
    if not text or text.lower() == "nan":
        return ()
    return tuple(value for value in text.split(",") if value)


def _load_prepared(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[OwnerBlock, ...],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    panel = pd.read_csv(args.output_dir / "panel.csv", low_memory=False)
    conditions = pd.read_csv(args.output_dir / "conditions.csv")
    block_frame = pd.read_csv(args.output_dir / "owner_blocks.csv")
    blocks = tuple(
        OwnerBlock(
            property_id=str(row.property_id),
            coordinate_indices=tuple(
                int(value)
                for value in str(row.coordinate_indices).split(",")
                if value
            ),
        )
        for row in block_frame.itertuples(index=False)
    )
    with np.load(args.output_dir / "replacement_coefficients.npz") as prepared:
        basis = np.asarray(prepared["competitive_basis"], dtype=np.float64)
        oracle = np.asarray(prepared["oracle"], dtype=np.float64)
        predicted = {
            block.property_id: np.asarray(
                prepared[f"predicted__{block.property_id}"], dtype=np.float64
            )
            for block in blocks
        }
    return panel, conditions, blocks, basis, oracle, predicted


def _requested_condition_ids(
    raw: str,
    conditions: pd.DataFrame,
) -> tuple[str, ...]:
    available = tuple(conditions["condition_id"].astype(str))
    if str(raw).strip().lower() == "all":
        return available
    selected = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return selected


def _condition_coefficients(
    *,
    condition_id: str,
    conditions: pd.DataFrame,
    blocks: Sequence[OwnerBlock],
    oracle: np.ndarray,
    predicted_by_block: Mapping[str, np.ndarray],
) -> np.ndarray:
    row = conditions.loc[conditions["condition_id"].astype(str).eq(condition_id)]
    if len(row) != 1:
        raise ValueError(f"Expected one condition row for {condition_id!r}.")
    replaced = _parse_blocks(row.iloc[0]["replaced_blocks"])
    return replace_owner_blocks(
        oracle,
        predicted_by_block,
        {block.property_id: block for block in blocks},
        replaced,
    )


def evaluate(args: argparse.Namespace) -> None:
    panel, conditions, blocks, basis, oracle, predicted = _load_prepared(args)
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    requested = _requested_condition_ids(args.conditions, conditions)
    selected = tuple(
        condition_id
        for condition_id in requested
        if int(
            conditions.loc[
                conditions["condition_id"].astype(str).eq(condition_id),
                "condition_index",
            ].iloc[0]
        )
        % int(args.num_shards)
        == int(args.shard_index)
    )
    with np.load(args.best_span_run_dir / "local_metrics.npz") as metrics:
        fisher = np.asarray(metrics["fisher__competitive"], dtype=np.float64)[
            : len(panel)
        ]
    hidden, lengths, panel_to_unique = _load_hidden(
        args.best_span_run_dir, len(panel)
    )
    cfg, _, _, loaded = _load_model_context(args)
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(loaded.tokenizer)
    panel_indices = np.arange(len(panel), dtype=np.int64)
    for condition_id in selected:
        path = args.output_dir / "condition_details" / f"{condition_id}.csv"
        if path.exists() and not args.force:
            continue
        coefficients = _condition_coefficients(
            condition_id=condition_id,
            conditions=conditions,
            blocks=blocks,
            oracle=oracle,
            predicted_by_block=predicted,
        )
        detail = _evaluate_coefficients(
            condition_id=condition_id,
            panel=panel,
            panel_indices=panel_indices,
            coefficients=coefficients,
            basis=basis,
            fisher_full=fisher,
            hidden=hidden,
            lengths=lengths,
            panel_to_unique=panel_to_unique,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            layer_idx=int(cfg["intervention_layer_idx"]),
            batch_size=int(args.batch_size),
        )
        missing = sorted(set(COMPACT_COLUMNS) - set(detail.columns))
        if missing:
            raise ValueError(f"Evaluation output lacks compact columns: {missing}")
        ensure_dir(path.parent)
        atomic_write_csv(path, detail.loc[:, COMPACT_COLUMNS])
        print(f"[evaluate] {condition_id}", flush=True)
    del loaded
    torch.cuda.empty_cache()


def _loss_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    probability = pd.to_numeric(result["target_prob"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    result["target_nll"] = -np.log(np.clip(probability, NLL_FLOOR, 1.0))
    result["abs_target_error"] = np.abs(
        pd.to_numeric(result["target_error"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    )
    result["exact_failure"] = 1.0 - pd.to_numeric(
        result["correct"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    result["target_prob_numeric"] = probability
    return result


def _aligned_condition(
    by_condition: Mapping[str, pd.DataFrame],
    condition_id: str,
) -> pd.DataFrame:
    frame = by_condition[condition_id].sort_values("panel_row_idx").reset_index(
        drop=True
    )
    expected = np.arange(DISCOVERY_ROWS, DISCOVERY_ROWS + len(frame), dtype=np.int64)
    observed = frame["panel_row_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(f"Confirmatory row coverage is invalid for {condition_id}.")
    return frame


def _summary_rows(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset, frame in (
        ("full", details),
        ("discovery", details.loc[details["panel_row_idx"] < DISCOVERY_ROWS]),
        ("confirmatory", details.loc[details["panel_row_idx"] >= DISCOVERY_ROWS]),
    ):
        for condition_id, group in frame.groupby("condition", sort=False):
            numeric = _loss_columns(group)
            rows.append(
                {
                    "subset": subset,
                    "condition": condition_id,
                    "num_examples": len(group),
                    "mean_target_nll": float(numeric["target_nll"].mean()),
                    "mean_target_probability": float(
                        numeric["target_prob_numeric"].mean()
                    ),
                    "exact_accuracy": float(1.0 - numeric["exact_failure"].mean()),
                    "mean_abs_target_error": float(
                        numeric["abs_target_error"].mean()
                    ),
                    "valid_numeric_output_rate": float(
                        pd.to_numeric(group["predicted_value"], errors="coerce")
                        .notna()
                        .mean()
                    ),
                    "mean_delta_norm": float(
                        pd.to_numeric(group["delta_norm"], errors="coerce").mean()
                    ),
                    "mean_source_fisher_quadratic": float(
                        pd.to_numeric(
                            group["source_fisher_quadratic"], errors="coerce"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pair_examples(
    *,
    pair_effects: pd.DataFrame,
    by_condition: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    selected_pairs = pd.concat(
        [
            pair_effects.nlargest(3, "interaction_target_nll"),
            pair_effects.nsmallest(3, "interaction_target_nll"),
        ],
        ignore_index=True,
    ).drop_duplicates(["left_block", "right_block"])
    rng = np.random.default_rng(DEFAULT_SEED)
    rows: list[dict[str, object]] = []
    oracle = _aligned_condition(by_condition, "oracle")
    for pair in selected_pairs.itertuples(index=False):
        left_id = f"replace__{pair.left_block}"
        right_id = f"replace__{pair.right_block}"
        pair_id = f"replace__{pair.left_block}__{pair.right_block}"
        left = _aligned_condition(by_condition, left_id)
        right = _aligned_condition(by_condition, right_id)
        both = _aligned_condition(by_condition, pair_id)
        contrast = interaction_contrast(
            both["target_nll"],
            left["target_nll"],
            right["target_nll"],
            oracle["target_nll"],
        )
        extreme = int(np.argmax(np.abs(contrast)))
        random_row = int(rng.integers(0, len(contrast)))
        for selection, row_index in (("largest_absolute_J", extreme), ("seeded_random", random_row)):
            panel_row_idx = int(oracle.iloc[row_index]["panel_row_idx"])
            panel_row = panel.iloc[panel_row_idx]
            rows.append(
                {
                    "selection": selection,
                    "left_block": pair.left_block,
                    "right_block": pair.right_block,
                    "pair_mean_interaction_target_nll": pair.interaction_target_nll,
                    "panel_row_idx": panel_row_idx,
                    "baseline_row_idx": int(oracle.iloc[row_index]["baseline_row_idx"]),
                    "source_value": int(panel_row["source_value"]),
                    "target_value": int(panel_row["target_value"]),
                    "delta_value": int(panel_row["delta_value"]),
                    "row_interaction_target_nll": float(contrast[row_index]),
                    "oracle_prediction": str(oracle.iloc[row_index]["predicted_text"]),
                    "left_prediction": str(left.iloc[row_index]["predicted_text"]),
                    "right_prediction": str(right.iloc[row_index]["predicted_text"]),
                    "pair_prediction": str(both.iloc[row_index]["predicted_text"]),
                    "oracle_target_probability": float(
                        oracle.iloc[row_index]["target_prob_numeric"]
                    ),
                    "left_target_probability": float(
                        left.iloc[row_index]["target_prob_numeric"]
                    ),
                    "right_target_probability": float(
                        right.iloc[row_index]["target_prob_numeric"]
                    ),
                    "pair_target_probability": float(
                        both.iloc[row_index]["target_prob_numeric"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> None:
    panel, conditions, blocks, _, oracle_coefficients, predicted = _load_prepared(args)
    expected = tuple(conditions["condition_id"].astype(str))
    detail_paths = {
        path.stem: path
        for path in (args.output_dir / "condition_details").glob("*.csv")
    }
    missing = sorted(set(expected) - set(detail_paths))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} condition details; first: {missing[:5]}"
        )
    parts: list[pd.DataFrame] = []
    by_condition: dict[str, pd.DataFrame] = {}
    for condition_id in expected:
        frame = pd.read_csv(detail_paths[condition_id], low_memory=False)
        if len(frame) != len(panel):
            raise ValueError(
                f"{condition_id} has {len(frame)} rows; expected {len(panel)}."
            )
        if frame["panel_row_idx"].duplicated().any():
            raise ValueError(f"{condition_id} contains duplicate panel rows.")
        numeric = _loss_columns(frame)
        parts.append(numeric)
        by_condition[condition_id] = numeric.loc[
            numeric["panel_row_idx"] >= DISCOVERY_ROWS
        ].copy()
    details = pd.concat(parts, ignore_index=True)
    summary = _summary_rows(details)
    atomic_write_csv(args.output_dir / "causal_summary.csv", summary)

    block_by_property = {block.property_id: block for block in blocks}
    oracle = _aligned_condition(by_condition, "oracle")
    clusters = oracle["baseline_row_idx"].to_numpy(dtype=np.int64)
    single_rows: list[dict[str, object]] = []
    for block_index, block in enumerate(blocks):
        condition_id = f"replace__{block.property_id}"
        replacement = _aligned_condition(by_condition, condition_id)
        nll_delta = (
            replacement["target_nll"] - oracle["target_nll"]
        ).to_numpy(dtype=np.float64)
        mae_delta = (
            replacement["abs_target_error"] - oracle["abs_target_error"]
        ).to_numpy(dtype=np.float64)
        failure_delta = (
            replacement["exact_failure"] - oracle["exact_failure"]
        ).to_numpy(dtype=np.float64)
        nll_estimate, nll_low, nll_high = cluster_bootstrap_mean(
            nll_delta,
            clusters,
            draws=BOOTSTRAP_DRAWS,
            seed=DEFAULT_SEED + 100 + block_index,
        )
        mae_estimate, mae_low, mae_high = cluster_bootstrap_mean(
            mae_delta,
            clusters,
            draws=BOOTSTRAP_DRAWS,
            seed=DEFAULT_SEED + 200 + block_index,
        )
        indices = np.asarray(block.coordinate_indices, dtype=np.int64)
        truth = oracle_coefficients[DISCOVERY_ROWS:, :][:, indices]
        fitted = predicted[block.property_id][DISCOVERY_ROWS:, :][:, indices]
        residual = fitted - truth
        truth_energy = float(np.sum(np.square(truth)))
        residual_energy = float(np.sum(np.square(residual)))
        flat_truth = truth.reshape(-1)
        flat_fitted = fitted.reshape(-1)
        correlation = (
            float(np.corrcoef(flat_truth, flat_fitted)[0, 1])
            if np.std(flat_truth) > 0.0 and np.std(flat_fitted) > 0.0
            else np.nan
        )
        single_rows.append(
            {
                "block_index": block_index,
                "property_id": block.property_id,
                "owner_rank": block.rank,
                "coordinate_fve": (
                    1.0 - residual_energy / truth_energy
                    if truth_energy > 0.0
                    else np.nan
                ),
                "coordinate_correlation": correlation,
                "mean_oracle_block_norm": float(
                    np.linalg.norm(truth, axis=1).mean()
                ),
                "mean_prediction_error_norm": float(
                    np.linalg.norm(residual, axis=1).mean()
                ),
                "target_nll_degradation": nll_estimate,
                "target_nll_degradation_ci_low": nll_low,
                "target_nll_degradation_ci_high": nll_high,
                "mae_degradation": mae_estimate,
                "mae_degradation_ci_low": mae_low,
                "mae_degradation_ci_high": mae_high,
                "exact_failure_increase": float(
                    cluster_means(failure_delta, clusters)[1].mean()
                ),
            }
        )
    single_effects = pd.DataFrame(single_rows)
    atomic_write_csv(args.output_dir / "single_block_effects.csv", single_effects)

    pair_rows: list[dict[str, object]] = []
    pair_conditions = conditions.loc[conditions["condition_kind"].eq("pair")]
    for pair_index, row in enumerate(pair_conditions.itertuples(index=False)):
        left_block, right_block = _parse_blocks(row.replaced_blocks)
        left = _aligned_condition(by_condition, f"replace__{left_block}")
        right = _aligned_condition(by_condition, f"replace__{right_block}")
        pair = _aligned_condition(by_condition, str(row.condition_id))
        nll_interaction = interaction_contrast(
            pair["target_nll"],
            left["target_nll"],
            right["target_nll"],
            oracle["target_nll"],
        )
        mae_interaction = interaction_contrast(
            pair["abs_target_error"],
            left["abs_target_error"],
            right["abs_target_error"],
            oracle["abs_target_error"],
        )
        failure_interaction = interaction_contrast(
            pair["exact_failure"],
            left["exact_failure"],
            right["exact_failure"],
            oracle["exact_failure"],
        )
        estimate, low, high = cluster_bootstrap_mean(
            nll_interaction,
            clusters,
            draws=BOOTSTRAP_DRAWS,
            seed=DEFAULT_SEED + 1000 + pair_index,
        )
        _, cluster_nll = cluster_means(nll_interaction, clusters)
        test = ttest_1samp(cluster_nll, popmean=0.0)
        fisher_interaction = interaction_contrast(
            pair["source_fisher_quadratic"],
            left["source_fisher_quadratic"],
            right["source_fisher_quadratic"],
            oracle["source_fisher_quadratic"],
        )
        squared_l2_interaction = interaction_contrast(
            np.square(pair["delta_norm"]),
            np.square(left["delta_norm"]),
            np.square(right["delta_norm"]),
            np.square(oracle["delta_norm"]),
        )
        pair_rows.append(
            {
                "pair_index": pair_index,
                "left_block": left_block,
                "right_block": right_block,
                "left_rank": block_by_property[left_block].rank,
                "right_rank": block_by_property[right_block].rank,
                "interaction_target_nll": estimate,
                "interaction_target_nll_ci_low": low,
                "interaction_target_nll_ci_high": high,
                "interaction_target_nll_t_p": float(test.pvalue),
                "interaction_mae": float(
                    cluster_means(mae_interaction, clusters)[1].mean()
                ),
                "interaction_exact_failure": float(
                    cluster_means(failure_interaction, clusters)[1].mean()
                ),
                "interaction_fisher_quadratic": float(
                    cluster_means(fisher_interaction, clusters)[1].mean()
                ),
                "interaction_squared_l2": float(
                    cluster_means(squared_l2_interaction, clusters)[1].mean()
                ),
                "max_abs_row_squared_l2_interaction": float(
                    np.max(np.abs(squared_l2_interaction))
                ),
            }
        )
    pair_effects = pd.DataFrame(pair_rows)
    pair_effects["interaction_target_nll_bh_q"] = benjamini_hochberg(
        pair_effects["interaction_target_nll_t_p"].to_numpy(dtype=np.float64)
    )
    single_nll = single_effects.set_index("property_id")[
        "target_nll_degradation"
    ].to_dict()
    pair_effects["left_single_target_nll_degradation"] = pair_effects[
        "left_block"
    ].map(single_nll)
    pair_effects["right_single_target_nll_degradation"] = pair_effects[
        "right_block"
    ].map(single_nll)
    pair_effects["single_target_nll_degradation_sum"] = (
        pair_effects["left_single_target_nll_degradation"]
        + pair_effects["right_single_target_nll_degradation"]
    )
    pair_effects["single_target_nll_degradation_product"] = (
        pair_effects["left_single_target_nll_degradation"]
        * pair_effects["right_single_target_nll_degradation"]
    )
    compounding_x = np.column_stack(
        [
            np.ones(len(pair_effects), dtype=np.float64),
            pair_effects["single_target_nll_degradation_sum"].to_numpy(
                dtype=np.float64
            ),
            pair_effects["single_target_nll_degradation_product"].to_numpy(
                dtype=np.float64
            ),
        ]
    )
    compounding_y = pair_effects["interaction_target_nll"].to_numpy(
        dtype=np.float64
    )
    compounding_coefficients, _, _, _ = np.linalg.lstsq(
        compounding_x,
        compounding_y,
        rcond=None,
    )
    compounding_prediction = compounding_x @ compounding_coefficients
    pair_effects["generic_compounding_prediction"] = compounding_prediction
    pair_effects["generic_compounding_residual"] = (
        compounding_y - compounding_prediction
    )
    denominator = float(np.sum(np.square(compounding_y - compounding_y.mean())))
    in_sample_r2 = float(
        1.0
        - np.sum(np.square(compounding_y - compounding_prediction))
        / max(denominator, 1e-24)
    )
    rng = np.random.default_rng(DEFAULT_SEED + 9000)
    fold_ids = np.empty(len(pair_effects), dtype=np.int64)
    fold_ids[rng.permutation(len(pair_effects))] = (
        np.arange(len(pair_effects), dtype=np.int64) % 10
    )
    cross_validated_prediction = np.full(len(pair_effects), np.nan, dtype=np.float64)
    for fold in range(10):
        test = fold_ids == fold
        train = ~test
        fold_coefficients, _, _, _ = np.linalg.lstsq(
            compounding_x[train],
            compounding_y[train],
            rcond=None,
        )
        cross_validated_prediction[test] = (
            compounding_x[test] @ fold_coefficients
        )
    cross_validated_r2 = float(
        1.0
        - np.sum(
            np.square(compounding_y - cross_validated_prediction)
        )
        / max(denominator, 1e-24)
    )
    pair_effects["generic_compounding_fold"] = fold_ids
    pair_effects["generic_compounding_cv_prediction"] = (
        cross_validated_prediction
    )
    atomic_write_csv(args.output_dir / "pairwise_interactions.csv", pair_effects)
    atomic_write_json(
        args.output_dir / "interaction_compounding_diagnostics.json",
        {
            "status": "post_hoc_descriptive",
            "formula": (
                "J ~ intercept + (left single NLL degradation + right single "
                "NLL degradation) + (left single NLL degradation * right "
                "single NLL degradation)"
            ),
            "coefficients": {
                "intercept": float(compounding_coefficients[0]),
                "single_sum": float(compounding_coefficients[1]),
                "single_product": float(compounding_coefficients[2]),
            },
            "in_sample_r2": in_sample_r2,
            "ten_fold_pair_cv_r2": cross_validated_r2,
            "seed": DEFAULT_SEED + 9000,
        },
    )

    all_replaced = _aligned_condition(by_condition, "replace__all")
    all_rows: list[dict[str, object]] = []
    for metric in ("target_nll", "abs_target_error", "exact_failure"):
        delta = (
            all_replaced[metric] - oracle[metric]
        ).to_numpy(dtype=np.float64)
        estimate, low, high = cluster_bootstrap_mean(
            delta,
            clusters,
            draws=BOOTSTRAP_DRAWS,
            seed=DEFAULT_SEED + 3000 + len(all_rows),
        )
        all_rows.append(
            {
                "metric": metric,
                "all_replaced_minus_oracle": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    atomic_write_csv(
        args.output_dir / "all_block_contrast.csv",
        pd.DataFrame(all_rows),
    )
    examples = _pair_examples(
        pair_effects=pair_effects,
        by_condition=by_condition,
        panel=panel,
    )
    atomic_write_csv(args.output_dir / "interaction_examples.csv", examples)

    changed_rows: list[dict[str, object]] = []
    confirmatory_panel = panel.iloc[DISCOVERY_ROWS:].reset_index(drop=True)
    for block in blocks:
        if block.property_id == "magnitude":
            changed = (
                confirmatory_panel["source_value"].to_numpy(dtype=np.int64)
                != confirmatory_panel["target_value"].to_numpy(dtype=np.int64)
            )
        else:
            changed = property_labels(
                block.property_id,
                confirmatory_panel["source_value"].to_numpy(dtype=np.int64),
            ) != property_labels(
                block.property_id,
                confirmatory_panel["target_value"].to_numpy(dtype=np.int64),
            )
        replacement = _aligned_condition(
            by_condition, f"replace__{block.property_id}"
        )
        degradation = (
            replacement["target_nll"] - oracle["target_nll"]
        ).to_numpy(dtype=np.float64)
        changed_rows.append(
            {
                "property_id": block.property_id,
                "changed_prevalence": float(np.mean(changed)),
                "target_nll_degradation_changed": float(
                    np.mean(degradation[changed])
                ),
                "target_nll_degradation_unchanged": (
                    float(np.mean(degradation[~changed]))
                    if np.any(~changed)
                    else np.nan
                ),
            }
        )
    atomic_write_csv(
        args.output_dir / "changed_property_strata.csv",
        pd.DataFrame(changed_rows),
    )

    max_l2_interaction = float(
        pair_effects["max_abs_row_squared_l2_interaction"].max()
    )
    completion = {
        "status": "complete",
        "conditions_expected": len(expected),
        "conditions_observed": len(detail_paths),
        "blocks": len(blocks),
        "pairs": len(pair_effects),
        "panel_rows": len(panel),
        "confirmatory_rows": len(oracle),
        "max_abs_row_squared_l2_interaction": max_l2_interaction,
        "euclidean_disjointness_audit_passed": max_l2_interaction < 2e-4,
        "artifacts": {
            name: _sha256(args.output_dir / name)
            for name in (
                "causal_summary.csv",
                "single_block_effects.csv",
                "pairwise_interactions.csv",
                "interaction_compounding_diagnostics.json",
                "all_block_contrast.csv",
                "interaction_examples.csv",
                "changed_property_strata.csv",
            )
        },
        "runtime": runtime_metadata(),
    }
    atomic_write_json(args.output_dir / "DONE.json", completion)
    atomic_write_text(args.output_dir / "DONE", "ok\n")


def main() -> None:
    args = parse_args()
    set_global_seed(DEFAULT_SEED)
    if args.stage in {"all", "prepare"}:
        prepare(args)
    if args.stage in {"all", "evaluate"}:
        observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed != REQUIRED_CUBLAS:
            raise RuntimeError(
                f"CUBLAS_WORKSPACE_CONFIG must be {REQUIRED_CUBLAS!r}; "
                f"got {observed!r}."
            )
        evaluate(args)
    if args.stage in {"all", "analyze"}:
        analyze(args)


if __name__ == "__main__":
    main()
