#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    DEFAULT_OUTPUT as BAG_OUTPUT,
    PROPERTY_ORDER,
    SOURCE_RUN,
    activations_by_value,
    load_source_config,
    load_split_activations,
)
from src.iterative_nullspace_probes import (  # noqa: E402
    categorical_direction_candidates,
    categorical_metrics,
    permutation_max_balanced_accuracy,
    permutation_r2,
    pls_x_weight1_candidate,
    project_out_rank_one,
    regression_metrics,
    select_categorical_candidate,
)
from src.model_loader import load_tokenizer  # noqa: E402
from src.number_property_probes import PROPERTY_LABELS, property_labels  # noqa: E402
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text, ensure_dir, runtime_metadata  # noqa: E402


DEFAULT_OUTPUT = ROOT / "out/figure1_iterative_nullspace_property_probes/llama31_8b_layer10_v1"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
REGIME_ORDER = ("occurrence_heldout", "value_disjoint")
MAGNITUDE_ID = "magnitude"
EXTENDED_SUPPORTED_PROPERTY_ORDER = (
    "divisible_by_4",
    "divisible_by_6",
    "divisible_by_8",
    "divisible_by_25",
    "residue_mod_3",
    "residue_mod_4",
    "contains_zero",
    "has_repeated_digit",
    "distinct_digit_count",
    "palindrome",
)
ALL_CATEGORICAL_PROPERTY_ORDER = tuple(dict.fromkeys((*PROPERTY_ORDER, *PROPERTY_LABELS)))
SEED = 20260722


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count greedily extractable rank-one linear number-property directions with held-out stopping."
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--bag-output-dir", type=Path, default=BAG_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default="all", help="Comma-separated regimes or 'all'.")
    parser.add_argument(
        "--properties",
        default="original",
        help=(
            "Comma-separated ids, 'all', or a preset: original, extended_supported, "
            "comprehensive_supported."
        ),
    )
    parser.add_argument("--max-directions", type=int, default=256)
    parser.add_argument("--num-permutations", type=int, default=4999)
    parser.add_argument("--dev-alpha", type=float, default=0.01)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _selected(
    raw: str,
    available: Sequence[str],
    *,
    presets: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    if presets is not None and str(raw).strip() in presets:
        return tuple(presets[str(raw).strip()])
    if str(raw).strip().lower() == "all":
        return tuple(available)
    result = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    unknown = sorted(set(result) - set(available))
    if unknown:
        raise ValueError(f"Unknown selections {unknown}; available: {list(available)}")
    return result


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in (SEED, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_regime_data(
    *, source_run_dir: Path, bag_output_dir: Path
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], dict[str, Any]]:
    cfg, model_entry, resolved = load_source_config(source_run_dir)
    split_path = bag_output_dir / "split00_activations.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Run scripts/run_figure1_bag_of_features_steering.py --skip-causal-eval first."
        )
    saved = np.load(split_path)
    strict = {
        partition: (
            np.asarray(saved[f"{partition}_values"], dtype=np.int64),
            np.asarray(saved[f"{partition}_activations"], dtype=np.float64),
        )
        for partition in ("train", "dev", "test")
    }
    train_by_value = np.asarray(saved["occurrence_train_by_value"], dtype=np.float64)
    all_values = np.arange(1000, dtype=np.int64)

    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=str(resolved["model_revision"]),
    )
    occurrence_dev = activations_by_value(
        load_split_activations(
            split_name="split_08", cfg=cfg, model_entry=model_entry, resolved=resolved, tokenizer=tokenizer
        )
    ).astype(np.float64)
    occurrence_test = activations_by_value(
        load_split_activations(
            split_name="split_09", cfg=cfg, model_entry=model_entry, resolved=resolved, tokenizer=tokenizer
        )
    ).astype(np.float64)
    regimes = {
        "occurrence_heldout": {
            "train": (all_values, train_by_value),
            "dev": (all_values, occurrence_dev),
            "test": (all_values, occurrence_test),
        },
        "value_disjoint": strict,
    }
    provenance = {
        "source_run": str(source_run_dir.resolve()),
        "bag_output": str(bag_output_dir.resolve()),
        "split_artifact": str(split_path.resolve()),
        "split_artifact_sha256": _sha256(split_path),
        "model_alias": str(cfg["model"]),
        "model_id": str(model_entry["hf_id"]),
        "model_revision": str(resolved["model_revision"]),
        "layer": int(cfg["intervention_layer_idx"]),
        "family": str(cfg["family"]),
        "activation_cache_root": str(Path(cfg["activation_cache_root"]).resolve()),
        "activation_cache_policy": "require_hit",
        "occurrence_splits": {"train": "mean(split_01..split_07)", "dev": "split_08", "test": "split_09"},
        "value_disjoint_split": "split_00 train/dev/test",
    }
    return regimes, provenance


def _centered_partitions(
    partitions: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    center = np.asarray(partitions["train"][1], dtype=np.float64).mean(axis=0)
    return {
        partition: np.asarray(activations, dtype=np.float64) - center[None, :]
        for partition, (_, activations) in partitions.items()
    }, center


def _load_progress(combo_dir: Path, hidden_dim: int, force: bool) -> tuple[list[dict[str, Any]], np.ndarray]:
    if force:
        return [], np.zeros((hidden_dim, 0), dtype=np.float64)
    rows_path = combo_dir / "iterations.csv"
    directions_path = combo_dir / "directions.npz"
    if not rows_path.exists() or not directions_path.exists():
        return [], np.zeros((hidden_dim, 0), dtype=np.float64)
    frame = pd.read_csv(rows_path)
    rows = frame.to_dict(orient="records")
    basis = np.asarray(np.load(directions_path)["basis"], dtype=np.float64)
    if basis.shape[0] != hidden_dim:
        raise ValueError(f"Saved basis has shape {basis.shape}, expected first dimension {hidden_dim}.")
    accepted_rows = sum(bool(row["accepted"]) for row in rows)
    if accepted_rows != basis.shape[1]:
        raise ValueError(f"Progress mismatch: {accepted_rows} accepted rows but basis rank {basis.shape[1]}.")
    return rows, basis


def _save_progress(combo_dir: Path, rows: list[dict[str, Any]], basis: np.ndarray) -> None:
    ensure_dir(combo_dir)
    atomic_write_csv(combo_dir / "iterations.csv", pd.DataFrame(rows))
    temp = combo_dir / "directions.tmp.npz"
    np.savez_compressed(temp, basis=np.asarray(basis, dtype=np.float32))
    temp.replace(combo_dir / "directions.npz")


def run_combo(
    *,
    output_dir: Path,
    regime: str,
    property_id: str,
    partitions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    max_directions: int,
    num_permutations: int,
    dev_alpha: float,
    test_alpha: float,
    force: bool,
) -> dict[str, Any]:
    combo_dir = ensure_dir(output_dir / regime / property_id)
    done_path = combo_dir / "DONE"
    summary_path = combo_dir / "summary.json"
    if done_path.exists() and summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    residual, center = _centered_partitions(partitions)
    hidden_dim = int(residual["train"].shape[1])
    rows, basis = _load_progress(combo_dir, hidden_dim, force)
    for direction in basis.T:
        residual = {partition: project_out_rank_one(x, direction) for partition, x in residual.items()}
    if rows and not bool(rows[-1]["accepted"]):
        rows = []
        basis = np.zeros((hidden_dim, 0), dtype=np.float64)
        residual, center = _centered_partitions(partitions)

    values = {partition: np.asarray(partitions[partition][0]) for partition in ("train", "dev", "test")}
    if property_id == MAGNITUDE_ID:
        labels = {partition: np.asarray(values[partition], dtype=np.float64) for partition in values}
        metric_name = "r2"
    else:
        labels = {partition: property_labels(property_id, values[partition]) for partition in values}
        metric_name = "balanced_accuracy"

    stopped_reason = "max_directions"
    for iteration in range(int(basis.shape[1]), int(max_directions)):
        if property_id == MAGNITUDE_ID:
            candidate, pls1_cosine = pls_x_weight1_candidate(
                train_x=residual["train"],
                train_y=labels["train"],
                dev_x=residual["dev"],
                test_x=residual["test"],
                existing_basis=basis,
                n_components=16,
                scale=False,
            )
            train_metrics = regression_metrics(labels["train"], candidate.train_prediction)
            dev_metrics = regression_metrics(labels["dev"], candidate.dev_prediction)
            test_metrics = regression_metrics(labels["test"], candidate.test_prediction)
            dev_null = permutation_r2(
                truth=labels["dev"], prediction=candidate.dev_prediction, observed_r2=float(dev_metrics[metric_name]),
                num_permutations=num_permutations, seed=_seed(regime, property_id, iteration, "dev"),
            )
            test_null = permutation_r2(
                truth=labels["test"], prediction=candidate.test_prediction, observed_r2=float(test_metrics[metric_name]),
                num_permutations=num_permutations, seed=_seed(regime, property_id, iteration, "test"),
            )
            candidate_count = 1
            alpha = None
            component = 0
        else:
            candidates = categorical_direction_candidates(
                train_x=residual["train"], train_y=labels["train"], dev_x=residual["dev"],
                test_x=residual["test"], existing_basis=basis, alphas=ALPHAS,
            )
            if not candidates:
                stopped_reason = "no_candidate_direction"
                break
            selected, dev_metrics = select_categorical_candidate(candidates, labels["dev"])
            candidate = candidates[selected]
            train_metrics = categorical_metrics(labels["train"], candidate.train_prediction)
            test_metrics = categorical_metrics(labels["test"], candidate.test_prediction)
            dev_null = permutation_max_balanced_accuracy(
                truth=labels["dev"],
                candidate_predictions=np.stack([item.dev_prediction for item in candidates], axis=0),
                observed_max=float(dev_metrics[metric_name]), num_permutations=num_permutations,
                seed=_seed(regime, property_id, iteration, "dev"),
            )
            test_null = permutation_max_balanced_accuracy(
                truth=labels["test"], candidate_predictions=candidate.test_prediction,
                observed_max=float(test_metrics[metric_name]), num_permutations=num_permutations,
                seed=_seed(regime, property_id, iteration, "test"),
            )
            candidate_count = len(candidates)
            alpha = candidate.alpha
            component = candidate.coefficient_component
            pls1_cosine = None

        accepted = bool(float(dev_null["p_value"]) < float(dev_alpha))
        row: dict[str, Any] = {
            "regime": regime,
            "property_id": property_id,
            "iteration": int(iteration),
            "direction_number": int(iteration + 1),
            "accepted": accepted,
            "selection_metric": metric_name,
            "candidate_count": int(candidate_count),
            "ridge_alpha": alpha,
            "coefficient_component": int(component),
            "dev_p_value": float(dev_null["p_value"]),
            "dev_null_mean": float(dev_null["null_mean"]),
            "dev_null_q99": float(dev_null["null_q99"]),
            "test_p_value": float(test_null["p_value"]),
            "test_significant": bool(float(test_null["p_value"]) < float(test_alpha)),
            "test_null_mean": float(test_null["null_mean"]),
            "test_null_q95": float(test_null["null_q95"]),
            "pls16_vs_pls1_xweight_abs_cosine": pls1_cosine,
        }
        for partition, metrics in (("train", train_metrics), ("dev", dev_metrics), ("test", test_metrics)):
            row.update({f"{partition}_{key}": value for key, value in metrics.items()})
        rows.append(row)
        if not accepted:
            stopped_reason = "dev_no_better_than_random"
            _save_progress(combo_dir, rows, basis)
            break
        direction = np.asarray(candidate.direction, dtype=np.float64)
        if basis.shape[1]:
            direction -= basis @ (basis.T @ direction)
            direction /= np.linalg.norm(direction)
        basis = np.column_stack([basis, direction])
        residual = {partition: project_out_rank_one(x, direction) for partition, x in residual.items()}
        _save_progress(combo_dir, rows, basis)

    accepted_rows = [row for row in rows if bool(row["accepted"])]
    test_confirmed_prefix = 0
    for row in accepted_rows:
        if not bool(row["test_significant"]):
            break
        test_confirmed_prefix += 1
    summary = {
        "regime": regime,
        "property_id": property_id,
        "greedy_dev_selected_direction_count": int(len(accepted_rows)),
        "test_confirmed_direction_count": int(sum(bool(row["test_significant"]) for row in accepted_rows)),
        "test_confirmed_prefix_count": int(test_confirmed_prefix),
        "stopped_reason": stopped_reason,
        "reached_max_directions": bool(len(accepted_rows) >= int(max_directions)),
        "max_directions": int(max_directions),
        "num_permutations": int(num_permutations),
        "dev_alpha": float(dev_alpha),
        "test_alpha": float(test_alpha),
        "hidden_dim": hidden_dim,
        "center_norm": float(np.linalg.norm(center)),
        "basis_orthogonality_max_error": float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))) if basis.shape[1] else 0.0,
        "method": "PLS x_weight1 (n_components=16, scale=False)" if property_id == MAGNITUDE_ID else "dev-selected rank-one RidgeClassifier coefficient SVD direction",
    }
    atomic_write_json(summary_path, summary)
    atomic_write_text(done_path, "done\n")
    return summary


def write_overlap(output_dir: Path, properties: Sequence[str], regimes: Sequence[str]) -> None:
    rows: list[dict[str, Any]] = []
    for regime in regimes:
        bases: dict[str, np.ndarray] = {}
        for property_id in properties:
            path = output_dir / regime / property_id / "directions.npz"
            if path.exists():
                bases[property_id] = np.asarray(np.load(path)["basis"], dtype=np.float64)
        for left, left_basis in bases.items():
            for right, right_basis in bases.items():
                cross = left_basis.T @ right_basis
                singular = np.linalg.svd(cross, compute_uv=False) if cross.size else np.asarray([])
                denom = max(1, min(left_basis.shape[1], right_basis.shape[1]))
                rows.append(
                    {
                        "regime": regime,
                        "property_a": left,
                        "property_b": right,
                        "rank_a": int(left_basis.shape[1]),
                        "rank_b": int(right_basis.shape[1]),
                        "max_canonical_cosine": float(singular[0]) if singular.size else 0.0,
                        "mean_squared_overlap": float(np.sum(cross * cross) / denom),
                    }
                )
    atomic_write_csv(output_dir / "direction_overlap.csv", pd.DataFrame(rows))


def main() -> None:
    args = parse_args()
    if args.num_permutations < 100:
        raise ValueError("Use at least 100 permutations.")
    regimes = _selected(args.regimes, REGIME_ORDER)
    properties = _selected(
        args.properties,
        (*ALL_CATEGORICAL_PROPERTY_ORDER, MAGNITUDE_ID),
        presets={
            "original": (*PROPERTY_ORDER, MAGNITUDE_ID),
            "extended_supported": EXTENDED_SUPPORTED_PROPERTY_ORDER,
            "comprehensive_supported": (*PROPERTY_ORDER, MAGNITUDE_ID, *EXTENDED_SUPPORTED_PROPERTY_ORDER),
        },
    )
    ensure_dir(args.output_dir)
    regime_data, provenance = load_regime_data(source_run_dir=args.source_run_dir, bag_output_dir=args.bag_output_dir)
    summaries: list[dict[str, Any]] = []
    for regime in regimes:
        for property_id in properties:
            print(f"[{regime}/{property_id}]", flush=True)
            summary = run_combo(
                output_dir=args.output_dir, regime=regime, property_id=property_id,
                partitions=regime_data[regime], max_directions=int(args.max_directions),
                num_permutations=int(args.num_permutations), dev_alpha=float(args.dev_alpha),
                test_alpha=float(args.test_alpha), force=bool(args.force),
            )
            summaries.append(summary)
            print(
                f"  directions={summary['greedy_dev_selected_direction_count']} "
                f"test_prefix={summary['test_confirmed_prefix_count']} stop={summary['stopped_reason']}",
                flush=True,
            )
    atomic_write_csv(args.output_dir / "property_summary.csv", pd.DataFrame(summaries))
    iteration_frames = [
        pd.read_csv(args.output_dir / regime / property_id / "iterations.csv")
        for regime in regimes
        for property_id in properties
        if (args.output_dir / regime / property_id / "iterations.csv").exists()
    ]
    if iteration_frames:
        atomic_write_csv(args.output_dir / "iterations.csv", pd.concat(iteration_frames, ignore_index=True))
    write_overlap(args.output_dir, properties, regimes)
    atomic_write_json(
        args.output_dir / "metadata.json",
        {
            "design": {
                "estimand": "maximum count along a registered greedy rank-one extraction path, not a global subspace-rank proof",
                "categorical_metric": "balanced accuracy",
                "categorical_chance": "permutation null preserving held-out label counts",
                "numeric_metric": "R2",
                "candidate_selection_control": "dev permutation distribution of maximum score across all ridge/SVD candidates",
                "stopping": f"first dev iteration with p >= {float(args.dev_alpha)}",
                "confirmation": f"untouched test permutation p < {float(args.test_alpha)}",
                "centering": "training mean",
                "projection": "orthogonal rank one after every accepted iteration",
            },
            "properties": list(properties),
            "regimes": list(regimes),
            "alphas": list(ALPHAS),
            "seed": SEED,
            "provenance": provenance,
            "runtime": runtime_metadata(),
        },
    )
    atomic_write_text(args.output_dir / "DONE", "done\n")


if __name__ == "__main__":
    main()
