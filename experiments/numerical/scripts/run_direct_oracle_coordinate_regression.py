#!/usr/bin/env python3
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
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import SOURCE_RUN  # noqa: E402
from scripts.run_figure1_oracle_block_replacement import (  # noqa: E402
    COMPACT_COLUMNS,
    NLL_FLOOR,
)
from scripts.run_figure1_owner_block_prefix_steering import (  # noqa: E402
    _evaluate_coefficients,
    _load_hidden,
    _load_model_context,
)
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.oracle_block_replacement import cluster_bootstrap_mean  # noqa: E402
from src.oracle_coordinate_regression import (  # noqa: E402
    PROPERTY_IDS,
    coordinate_detail,
    coordinate_summary,
    deterministic_group_split,
    fit_joint_feature_transform,
    fit_kernel_ridge_predict,
    fit_residual_mlp_predict,
    fit_ridge_candidates,
    fit_ridge_predict,
    fit_target_transform,
    kernel_gamma_base,
    select_kernel_ridge,
    select_residual_mlp,
)
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    ensure_dir,
    runtime_metadata,
    set_global_seed,
)


BEST_SPAN_RUN = (
    ROOT / "out/figure1_best_in_span_causal_ceiling/llama31_8b_layer10_v1"
)
OWNER_PREFIX_RUN = (
    ROOT / "out/figure1_owner_block_prefix_steering/llama31_8b_layer10_v1"
)
REPLACEMENT_RUN = (
    ROOT / "out/figure1_oracle_block_replacement/llama31_8b_layer10_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "out/direct_oracle_coordinate_regression/llama31_8b_layer10_v1"
)
DESIGN_PATH = (
    ROOT
    / "experiment-designs/2026-07-29-direct-oracle-coordinate-regression.md"
)
DEFAULT_SEED = 20260729
REQUIRED_CUBLAS = ":4096:8"
BOOTSTRAP_DRAWS = 10000
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
KERNEL_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GAMMA_MULTIPLIERS = (0.25, 1.0, 4.0)
MLP_WIDTHS = (64, 128)
MLP_WEIGHT_DECAYS = (1e-4, 1e-3)
MLP_LEARNING_RATE = 1e-3
MLP_MAX_EPOCHS = 600
MLP_PATIENCE = 60
LEARNED_CONDITIONS = ("ridge", "poly2", "poly3", "rbf", "residual_mlp")
CONDITION_ORDER = (
    "no_edit",
    "independent_property_all",
    *LEARNED_CONDITIONS,
    "oracle",
)
PANEL_KEY_COLUMNS = (
    "baseline_row_idx",
    "target_slot",
    "source_value",
    "target_value",
    "token_idx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit matched linear and nonlinear 41-output predictors of the "
            "Figure 1 projected exact patch and evaluate causal steering."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--best-span-run-dir", type=Path, default=BEST_SPAN_RUN)
    parser.add_argument("--owner-prefix-run-dir", type=Path, default=OWNER_PREFIX_RUN)
    parser.add_argument("--replacement-run-dir", type=Path, default=REPLACEMENT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument(
        "--stage",
        choices=("all", "fit", "evaluate", "analyze"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _panel_hash(panel: pd.DataFrame) -> str:
    return hashlib.sha256(
        panel.loc[:, PANEL_KEY_COLUMNS].to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _selected_conditions(raw: str) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return CONDITION_ORDER
    selected = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    unknown = sorted(set(selected) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return selected


def _load_geometry(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    panel = pd.read_csv(args.owner_prefix_run_dir / "panel.csv", low_memory=False)
    with np.load(args.best_span_run_dir / "geometry.npz") as payload:
        basis = np.asarray(payload["competitive_basis"], dtype=np.float64)
        source_activations = np.asarray(
            payload["source_activations"], dtype=np.float64
        )
        exact_patch = np.asarray(payload["exact_patch"], dtype=np.float64)
    if len(panel) != 2000 or basis.shape != (4096, 41):
        raise ValueError(
            f"Unexpected panel/basis shapes: {len(panel)} and {basis.shape}."
        )
    if source_activations.shape != exact_patch.shape or len(exact_patch) != len(panel):
        raise ValueError("Geometry arrays are not aligned to the panel.")
    source_coordinates = source_activations @ basis
    oracle = exact_patch @ basis
    with np.load(args.replacement_run_dir / "replacement_coefficients.npz") as source:
        replacement_basis = np.asarray(
            source["competitive_basis"], dtype=np.float64
        )
        independent = np.asarray(source["all_replaced"], dtype=np.float64)
        replacement_oracle = np.asarray(source["oracle"], dtype=np.float64)
    if not np.allclose(basis, replacement_basis, rtol=1e-6, atol=1e-7):
        raise ValueError("Replacement and best-span competitive bases differ.")
    if not np.allclose(oracle, replacement_oracle, rtol=2e-5, atol=2e-5):
        raise ValueError("Replacement and reconstructed oracle coordinates differ.")
    return panel, basis, source_coordinates, oracle, independent


def _condition_rows() -> pd.DataFrame:
    kinds = {
        "no_edit": "negative_control",
        "independent_property_all": "prior_label_prediction",
        "ridge": "linear_joint_regression",
        "poly2": "nonlinear_joint_regression",
        "poly3": "nonlinear_joint_regression",
        "rbf": "nonlinear_joint_regression",
        "residual_mlp": "nonlinear_joint_regression",
        "oracle": "causal_ceiling",
    }
    return pd.DataFrame(
        [
            {
                "condition_index": index,
                "condition": condition,
                "condition_kind": kinds[condition],
                "selected_by": (
                    "validation_coordinate_mse"
                    if condition in LEARNED_CONDITIONS
                    else "fixed_reference"
                ),
            }
            for index, condition in enumerate(CONDITION_ORDER)
        ]
    )


def _feature_metadata(transform: Any, features: np.ndarray) -> dict[str, Any]:
    return {
        "property_ids": list(transform.property_ids),
        "property_count": len(transform.property_ids),
        "feature_dim": int(features.shape[1]),
        "source_coordinate_dim": int(len(transform.coordinate_mean)),
        "property_feature_dim": int(transform.encoder.feature_dim),
        "feature_names": list(transform.encoder.feature_names),
        "blocks": [
            {
                "property_id": block.property_id,
                "kind": block.kind,
                "classes": (
                    None
                    if block.classes is None
                    else [int(value) for value in block.classes]
                ),
                "start": int(block.start),
                "stop": int(block.stop),
                "block_scale": float(block.block_scale),
            }
            for block in transform.encoder.blocks
        ],
    }


def fit(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    prediction_path = output_dir / "predicted_coefficients.npz"
    if prediction_path.exists() and not args.force:
        print(prediction_path)
        return
    set_global_seed(DEFAULT_SEED)
    torch.set_num_threads(1)
    panel, basis, source_coordinates, oracle, independent = _load_geometry(args)
    split = deterministic_group_split(
        panel["baseline_row_idx"].to_numpy(dtype=np.int64),
        seed=DEFAULT_SEED,
        train_groups=60,
        validation_groups=20,
        test_groups=20,
    )
    train_mask = split.row_partitions == "train"
    validation_mask = split.row_partitions == "validation"
    test_mask = split.row_partitions == "test"
    source_values = panel["source_value"].to_numpy(dtype=np.int64)
    target_values = panel["target_value"].to_numpy(dtype=np.int64)
    feature_transform = fit_joint_feature_transform(
        source_coordinates=source_coordinates,
        source_values=source_values,
        target_values=target_values,
        train_mask=train_mask,
        property_ids=PROPERTY_IDS,
    )
    features = feature_transform.transform(
        source_coordinates,
        source_values,
        target_values,
    )
    target_transform = fit_target_transform(oracle, train_mask)
    normalized_target = target_transform.transform(oracle)
    train_x = features[train_mask]
    train_y = normalized_target[train_mask]
    validation_x = features[validation_mask]
    validation_y = normalized_target[validation_mask]
    fit_mask = train_mask | validation_mask
    fit_x = features[fit_mask]
    fit_y = normalized_target[fit_mask]

    ridge_alpha, ridge_trace = fit_ridge_candidates(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        alphas=RIDGE_ALPHAS,
    )
    ridge_selection_model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(
        train_x,
        train_y,
    )
    ridge_train_prediction = np.asarray(
        ridge_selection_model.predict(train_x), dtype=np.float64
    )
    ridge_validation_prediction = np.asarray(
        ridge_selection_model.predict(validation_x), dtype=np.float64
    )
    ridge_final_model, ridge_normalized = fit_ridge_predict(
        fit_x=fit_x,
        fit_y=fit_y,
        predict_x=features,
        alpha=ridge_alpha,
    )

    kernel_selections: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = [
        {"condition": "ridge", **row} for row in ridge_trace
    ]
    normalized_predictions: dict[str, np.ndarray] = {
        "ridge": ridge_normalized
    }
    for condition, kernel, degree in (
        ("poly2", "polynomial", 2),
        ("poly3", "polynomial", 3),
        ("rbf", "rbf", None),
    ):
        selection, trace = select_kernel_ridge(
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            kernel=kernel,
            degree=degree,
            gamma_multipliers=GAMMA_MULTIPLIERS,
            alphas=KERNEL_ALPHAS,
        )
        kernel_selections[condition] = selection
        trace_rows.extend({"condition": condition, **row} for row in trace)
        normalized_predictions[condition] = fit_kernel_ridge_predict(
            fit_x=fit_x,
            fit_y=fit_y,
            predict_x=features,
            selection=selection,
        )

    mlp_selection, mlp_trace = select_residual_mlp(
        train_x=train_x,
        train_y=train_y,
        train_base_prediction=ridge_train_prediction,
        validation_x=validation_x,
        validation_y=validation_y,
        validation_base_prediction=ridge_validation_prediction,
        widths=MLP_WIDTHS,
        weight_decays=MLP_WEIGHT_DECAYS,
        learning_rate=MLP_LEARNING_RATE,
        max_epochs=MLP_MAX_EPOCHS,
        patience=MLP_PATIENCE,
        seed=DEFAULT_SEED + 5000,
    )
    trace_rows.extend({"condition": "residual_mlp", **row} for row in mlp_trace)
    ridge_fit_prediction = np.asarray(
        ridge_final_model.predict(fit_x), dtype=np.float64
    )
    normalized_predictions["residual_mlp"] = fit_residual_mlp_predict(
        fit_x=fit_x,
        fit_y=fit_y,
        fit_base_prediction=ridge_fit_prediction,
        predict_x=features,
        predict_base_prediction=ridge_normalized,
        selection=mlp_selection,
        learning_rate=MLP_LEARNING_RATE,
        seed=DEFAULT_SEED + 6000,
    )

    predictions = {
        condition: target_transform.inverse_transform(values)
        for condition, values in normalized_predictions.items()
    }
    predictions.update(
        {
            "no_edit": np.zeros_like(oracle),
            "independent_property_all": independent,
            "oracle": oracle,
        }
    )
    for condition in CONDITION_ORDER:
        if predictions[condition].shape != oracle.shape:
            raise ValueError(f"Prediction shape mismatch for {condition}.")
        if not np.isfinite(predictions[condition]).all():
            raise ValueError(f"Non-finite prediction for {condition}.")

    split_frame = panel.loc[:, list(PANEL_KEY_COLUMNS)].copy()
    split_frame.insert(0, "panel_row_idx", np.arange(len(panel), dtype=np.int64))
    split_frame["regression_partition"] = split.row_partitions
    atomic_write_csv(output_dir / "panel.csv", panel)
    atomic_write_csv(output_dir / "split.csv", split_frame)
    atomic_write_csv(output_dir / "conditions.csv", _condition_rows())
    atomic_write_csv(output_dir / "selection_trace.csv", pd.DataFrame(trace_rows))
    atomic_write_json(
        output_dir / "feature_metadata.json",
        _feature_metadata(feature_transform, features),
    )
    selected_payload = {
        "ridge": {"alpha": ridge_alpha},
        **{
            condition: {
                "kernel": selection.kernel,
                "degree": selection.degree,
                "gamma": selection.gamma,
                "gamma_multiplier": selection.gamma_multiplier,
                "alpha": selection.alpha,
                "validation_mse": selection.validation_mse,
                "training_gamma_base": kernel_gamma_base(
                    train_x,
                    selection.kernel,
                ),
            }
            for condition, selection in kernel_selections.items()
        },
        "residual_mlp": {
            "width": mlp_selection.width,
            "weight_decay": mlp_selection.weight_decay,
            "best_epoch": mlp_selection.best_epoch,
            "validation_mse": mlp_selection.validation_mse,
            "learning_rate": MLP_LEARNING_RATE,
        },
    }
    atomic_write_json(
        output_dir / "selected_hyperparameters.json",
        selected_payload,
    )
    _atomic_savez(
        prediction_path,
        competitive_basis=basis.astype(np.float32),
        oracle=oracle.astype(np.float32),
        source_coordinates=source_coordinates.astype(np.float32),
        features=features.astype(np.float32),
        target_center=target_transform.center.astype(np.float32),
        target_scale=np.asarray([target_transform.scale], dtype=np.float32),
        **{
            f"prediction__{condition}": predictions[condition].astype(np.float32)
            for condition in CONDITION_ORDER
        },
    )

    with np.load(args.best_span_run_dir / "local_metrics.npz") as source:
        fisher = np.asarray(source["fisher__competitive"], dtype=np.float64)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    for partition, mask in (
        ("train", train_mask),
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        for condition in CONDITION_ORDER:
            summary_rows.append(
                {
                    "partition": partition,
                    "condition": condition,
                    **coordinate_summary(
                        truth=oracle[mask],
                        prediction=predictions[condition][mask],
                        train_center=target_transform.center,
                        fisher=fisher[mask],
                    ),
                }
            )
            if partition == "test":
                detail = coordinate_detail(
                    truth=oracle[mask],
                    prediction=predictions[condition][mask],
                    fisher=fisher[mask],
                )
                indices = np.flatnonzero(mask)
                detail_rows.append(
                    pd.DataFrame(
                        {
                            "condition": condition,
                            "panel_row_idx": indices,
                            "baseline_row_idx": panel.iloc[indices][
                                "baseline_row_idx"
                            ].to_numpy(dtype=np.int64),
                            "source_value": source_values[indices],
                            "target_value": target_values[indices],
                            **detail,
                        }
                    )
                )
    atomic_write_csv(
        output_dir / "coordinate_summary.csv",
        pd.DataFrame(summary_rows),
    )
    atomic_write_csv(
        output_dir / "coordinate_test_details.csv",
        pd.concat(detail_rows, ignore_index=True),
    )

    train_target_values = set(int(value) for value in target_values[train_mask])
    validation_target_values = set(
        int(value) for value in target_values[validation_mask]
    )
    test_target_values = set(int(value) for value in target_values[test_mask])
    atomic_write_json(
        output_dir / "fit_metadata.json",
        {
            "schema_version": 1,
            "seed": DEFAULT_SEED,
            "split": {
                "unit": "baseline_row_idx",
                "train_groups": list(split.train_groups),
                "validation_groups": list(split.validation_groups),
                "test_groups": list(split.test_groups),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "test_rows": int(test_mask.sum()),
            },
            "target_value_overlap": {
                "train_validation": len(
                    train_target_values & validation_target_values
                ),
                "train_test": len(train_target_values & test_target_values),
                "validation_test": len(
                    validation_target_values & test_target_values
                ),
            },
            "panel_hash": _panel_hash(panel),
            "competitive_basis_hash": _array_hash(basis.astype(np.float32)),
            "oracle_hash": _array_hash(oracle.astype(np.float32)),
            "design_sha256": _sha256(DESIGN_PATH) if DESIGN_PATH.exists() else None,
            "runner_sha256": _sha256(Path(__file__)),
            "source_geometry_sha256": _sha256(
                args.best_span_run_dir / "geometry.npz"
            ),
            "source_replacement_sha256": _sha256(
                args.replacement_run_dir / "replacement_coefficients.npz"
            ),
            "selection_metric": "validation mean squared normalized coordinate error",
            "runtime": runtime_metadata(),
        },
    )
    print(prediction_path)


def evaluate(args: argparse.Namespace) -> None:
    panel = pd.read_csv(args.output_dir / "panel.csv", low_memory=False)
    split = pd.read_csv(args.output_dir / "split.csv")
    panel_indices = split.loc[
        split["regression_partition"].astype(str).eq("test"),
        "panel_row_idx",
    ].to_numpy(dtype=np.int64)
    with np.load(args.output_dir / "predicted_coefficients.npz") as source:
        basis = np.asarray(source["competitive_basis"], dtype=np.float64)
        coefficients = {
            condition: np.asarray(
                source[f"prediction__{condition}"], dtype=np.float64
            )
            for condition in CONDITION_ORDER
        }
    with np.load(args.best_span_run_dir / "local_metrics.npz") as source:
        fisher = np.asarray(source["fisher__competitive"], dtype=np.float64)
    hidden, lengths, panel_to_unique = _load_hidden(
        args.best_span_run_dir,
        len(panel),
    )
    cfg, _, _, loaded = _load_model_context(args)
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(loaded.tokenizer)
    selected = _selected_conditions(args.conditions)
    for condition in selected:
        path = args.output_dir / "condition_details" / f"{condition}.csv"
        if path.exists() and not args.force:
            continue
        detail = _evaluate_coefficients(
            condition_id=condition,
            panel=panel,
            panel_indices=panel_indices,
            coefficients=coefficients[condition],
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
            raise ValueError(f"Evaluation output lacks columns: {missing}")
        ensure_dir(path.parent)
        atomic_write_csv(path, detail.loc[:, COMPACT_COLUMNS])
        print(f"[evaluate] {condition}", flush=True)
    del loaded
    torch.cuda.empty_cache()


def _behavior_columns(frame: pd.DataFrame) -> pd.DataFrame:
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
    return result


def _aligned_details(
    frames: Mapping[str, pd.DataFrame],
    condition: str,
) -> pd.DataFrame:
    return (
        frames[condition]
        .sort_values("panel_row_idx")
        .reset_index(drop=True)
    )


def _contrast_row(
    *,
    condition: str,
    reference: str,
    metric: str,
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    estimate, low, high = cluster_bootstrap_mean(
        values,
        clusters,
        draws=BOOTSTRAP_DRAWS,
        seed=seed,
    )
    return {
        "condition": condition,
        "reference": reference,
        "metric": metric,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_unit": "baseline_row_idx",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def analyze(args: argparse.Namespace) -> None:
    detail_paths = {
        path.stem: path
        for path in (args.output_dir / "condition_details").glob("*.csv")
    }
    missing = sorted(set(CONDITION_ORDER) - set(detail_paths))
    if missing:
        raise FileNotFoundError(f"Missing causal details: {missing}")
    frames = {
        condition: _behavior_columns(
            pd.read_csv(detail_paths[condition], low_memory=False)
        )
        for condition in CONDITION_ORDER
    }
    expected_indices: np.ndarray | None = None
    for condition, frame in frames.items():
        indices = np.sort(frame["panel_row_idx"].to_numpy(dtype=np.int64))
        if len(indices) != 400 or len(np.unique(indices)) != 400:
            raise ValueError(f"{condition} does not contain 400 unique test rows.")
        if expected_indices is None:
            expected_indices = indices
        elif not np.array_equal(indices, expected_indices):
            raise ValueError(f"{condition} has mismatched test coverage.")

    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        frame = frames[condition]
        summary_rows.append(
            {
                "condition": condition,
                "num_examples": len(frame),
                "num_source_contexts": int(frame["baseline_row_idx"].nunique()),
                "exact_accuracy": float(frame["correct"].mean()),
                "mean_target_nll": float(frame["target_nll"].mean()),
                "mean_target_probability": float(frame["target_prob"].mean()),
                "numeric_mae": float(frame["abs_target_error"].mean()),
                "valid_numeric_output_rate": float(
                    pd.to_numeric(frame["predicted_value"], errors="coerce")
                    .notna()
                    .mean()
                ),
                "mean_delta_norm": float(frame["delta_norm"].mean()),
                "mean_source_fisher_quadratic": float(
                    frame["source_fisher_quadratic"].mean()
                ),
            }
        )
    causal_summary = pd.DataFrame(summary_rows)
    atomic_write_csv(args.output_dir / "causal_summary.csv", causal_summary)

    ridge = _aligned_details(frames, "ridge")
    oracle = _aligned_details(frames, "oracle")
    clusters = ridge["baseline_row_idx"].to_numpy(dtype=np.int64)
    contrast_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for condition in CONDITION_ORDER:
        current = _aligned_details(frames, condition)
        for reference_name, reference in (("ridge", ridge), ("oracle", oracle)):
            if condition == reference_name:
                continue
            for metric, values in (
                (
                    "target_nll_difference",
                    current["target_nll"].to_numpy(dtype=np.float64)
                    - reference["target_nll"].to_numpy(dtype=np.float64),
                ),
                (
                    "exact_accuracy_difference",
                    current["correct"].to_numpy(dtype=np.float64)
                    - reference["correct"].to_numpy(dtype=np.float64),
                ),
                (
                    "numeric_mae_difference",
                    current["abs_target_error"].to_numpy(dtype=np.float64)
                    - reference["abs_target_error"].to_numpy(dtype=np.float64),
                ),
            ):
                contrast_rows.append(
                    _contrast_row(
                        condition=condition,
                        reference=reference_name,
                        metric=metric,
                        values=values,
                        clusters=clusters,
                        seed=DEFAULT_SEED + 10000 + seed_offset,
                    )
                )
                seed_offset += 1
    atomic_write_csv(
        args.output_dir / "causal_contrasts.csv",
        pd.DataFrame(contrast_rows),
    )

    coordinate_details = pd.read_csv(
        args.output_dir / "coordinate_test_details.csv"
    )
    ridge_coordinate = (
        coordinate_details.loc[
            coordinate_details["condition"].astype(str).eq("ridge")
        ]
        .sort_values("panel_row_idx")
        .reset_index(drop=True)
    )
    coordinate_contrasts: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITION_ORDER):
        if condition == "ridge":
            continue
        current = (
            coordinate_details.loc[
                coordinate_details["condition"].astype(str).eq(condition)
            ]
            .sort_values("panel_row_idx")
            .reset_index(drop=True)
        )
        values = (
            current["squared_error"].to_numpy(dtype=np.float64)
            - ridge_coordinate["squared_error"].to_numpy(dtype=np.float64)
        )
        coordinate_contrasts.append(
            _contrast_row(
                condition=condition,
                reference="ridge",
                metric="coordinate_squared_error_difference",
                values=values,
                clusters=ridge_coordinate["baseline_row_idx"].to_numpy(
                    dtype=np.int64
                ),
                seed=DEFAULT_SEED + 20000 + condition_index,
            )
        )
    atomic_write_csv(
        args.output_dir / "coordinate_contrasts.csv",
        pd.DataFrame(coordinate_contrasts),
    )

    example_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(DEFAULT_SEED + 30000)
    for condition in ("poly2", "poly3", "rbf", "residual_mlp"):
        current = _aligned_details(frames, condition)
        improvement = (
            ridge["target_nll"].to_numpy(dtype=np.float64)
            - current["target_nll"].to_numpy(dtype=np.float64)
        )
        for selection, row_index in (
            ("largest_nll_improvement_vs_ridge", int(np.argmax(improvement))),
            ("largest_nll_harm_vs_ridge", int(np.argmin(improvement))),
            ("seeded_random", int(rng.integers(0, len(current)))),
        ):
            example_rows.append(
                {
                    "condition": condition,
                    "selection": selection,
                    "panel_row_idx": int(current.iloc[row_index]["panel_row_idx"]),
                    "baseline_row_idx": int(
                        current.iloc[row_index]["baseline_row_idx"]
                    ),
                    "source_value": int(current.iloc[row_index]["source_value"]),
                    "target_value": int(current.iloc[row_index]["target_value"]),
                    "condition_prediction": str(
                        current.iloc[row_index]["predicted_text"]
                    ),
                    "ridge_prediction": str(ridge.iloc[row_index]["predicted_text"]),
                    "oracle_prediction": str(
                        oracle.iloc[row_index]["predicted_text"]
                    ),
                    "condition_target_probability": float(
                        current.iloc[row_index]["target_prob"]
                    ),
                    "ridge_target_probability": float(
                        ridge.iloc[row_index]["target_prob"]
                    ),
                    "oracle_target_probability": float(
                        oracle.iloc[row_index]["target_prob"]
                    ),
                    "target_nll_improvement_vs_ridge": float(
                        improvement[row_index]
                    ),
                }
            )
    atomic_write_csv(
        args.output_dir / "interaction_examples.csv",
        pd.DataFrame(example_rows),
    )

    artifacts = (
        "fit_metadata.json",
        "feature_metadata.json",
        "selected_hyperparameters.json",
        "split.csv",
        "conditions.csv",
        "selection_trace.csv",
        "coordinate_summary.csv",
        "coordinate_test_details.csv",
        "coordinate_contrasts.csv",
        "causal_summary.csv",
        "causal_contrasts.csv",
        "interaction_examples.csv",
        "predicted_coefficients.npz",
    )
    atomic_write_json(
        args.output_dir / "DONE.json",
        {
            "status": "complete",
            "schema_version": 1,
            "conditions": len(CONDITION_ORDER),
            "test_rows": 400,
            "test_source_contexts": 20,
            "required_cublas_workspace_config": REQUIRED_CUBLAS,
            "artifacts": {
                name: _sha256(args.output_dir / name) for name in artifacts
            },
            "runtime": runtime_metadata(),
        },
    )
    (args.output_dir / "DONE").write_text("done\n", encoding="utf-8")
    print(args.output_dir / "DONE.json")


def main() -> None:
    args = parse_args()
    if args.stage in {"all", "fit"}:
        fit(args)
    if args.stage in {"all", "evaluate"}:
        evaluate(args)
    if args.stage in {"all", "analyze"}:
        analyze(args)


if __name__ == "__main__":
    main()
