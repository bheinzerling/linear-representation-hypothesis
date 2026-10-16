from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.magnitude_intervention import MagnitudeProbe


SCALAR_ARCHITECTURES = ("pls", "linear")


def stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def aggregate_context_subsets(
    frame: pd.DataFrame,
    activations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate all, even-indexed, and odd-indexed contexts per value.

    Even and odd centroids use the same number of rows for each value.  The
    first k registered variants in each subset are retained, where k is the
    smaller of the two available subset counts.
    """

    matrix = np.asarray(activations)
    if matrix.ndim != 3 or matrix.shape[0] != len(frame):
        raise ValueError(f"Expected row x layer x feature activations, got {matrix.shape} for {len(frame)} rows.")
    required = {"source_value", "variant_idx"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing context columns: {sorted(missing)}")
    if frame.duplicated(["source_value", "variant_idx"]).any():
        raise ValueError("Duplicate source_value/variant_idx rows in context panel.")

    source = frame["source_value"].to_numpy(dtype=np.int64)
    variants = frame["variant_idx"].to_numpy(dtype=np.int64)
    values = np.sort(np.unique(source))
    all_centroids: list[np.ndarray] = []
    seen_centroids: list[np.ndarray] = []
    heldout_centroids: list[np.ndarray] = []
    all_counts: list[int] = []
    paired_counts: list[int] = []
    for value in values:
        idx = np.flatnonzero(source == value)
        local_order = np.argsort(variants[idx], kind="stable")
        idx = idx[local_order]
        local_variants = variants[idx]
        even = idx[(local_variants % 2) == 0]
        odd = idx[(local_variants % 2) == 1]
        if len(even) == 0 or len(odd) == 0:
            raise ValueError(f"Value {value} does not have both even and odd context variants.")
        paired = min(len(even), len(odd))
        all_centroids.append(np.asarray(matrix[idx], dtype=np.float32).mean(axis=0, dtype=np.float32))
        seen_centroids.append(np.asarray(matrix[even[:paired]], dtype=np.float32).mean(axis=0, dtype=np.float32))
        heldout_centroids.append(np.asarray(matrix[odd[:paired]], dtype=np.float32).mean(axis=0, dtype=np.float32))
        all_counts.append(len(idx))
        paired_counts.append(paired)
    return {
        "values": values.astype(np.int64),
        "all_context_centroid": np.stack(all_centroids).astype(np.float32),
        "seen_context_centroid": np.stack(seen_centroids).astype(np.float32),
        "heldout_context_centroid": np.stack(heldout_centroids).astype(np.float32),
        "all_context_counts": np.asarray(all_counts, dtype=np.int64),
        "paired_context_counts": np.asarray(paired_counts, dtype=np.int64),
    }


def assert_value_alignment(items: Iterable[tuple[str, np.ndarray]]) -> np.ndarray:
    reference_name: str | None = None
    reference: np.ndarray | None = None
    for name, values in items:
        current = np.asarray(values, dtype=np.int64).reshape(-1)
        if reference is None:
            reference_name = str(name)
            reference = current
        elif not np.array_equal(reference, current):
            raise ValueError(f"Value rows for {name} differ from {reference_name}.")
    if reference is None:
        raise ValueError("No value arrays were supplied.")
    return reference


def global_value_centroids(family_arrays: dict[str, np.ndarray]) -> np.ndarray:
    if not family_arrays:
        raise ValueError("At least one family array is required.")
    shapes = {tuple(np.asarray(value).shape) for value in family_arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"Family centroid shapes are not aligned: {sorted(shapes)}")
    return np.stack([np.asarray(value, dtype=np.float32) for value in family_arrays.values()], axis=0).mean(
        axis=0, dtype=np.float32
    )


def rounded_prediction_metrics(
    *, truth: np.ndarray, continuous_prediction: np.ndarray, value_min: int = 0, value_max: int = 999
) -> tuple[dict[str, float | int], np.ndarray]:
    y = np.asarray(truth, dtype=np.int64).reshape(-1)
    continuous = np.asarray(continuous_prediction, dtype=np.float64).reshape(-1)
    if len(y) != len(continuous):
        raise ValueError("Truth and prediction lengths differ.")
    rounded = np.clip(np.rint(continuous), int(value_min), int(value_max)).astype(np.int64)
    residual = rounded.astype(np.float64) - y.astype(np.float64)
    continuous_residual = continuous - y.astype(np.float64)
    denominator = float(np.sum((y.astype(np.float64) - float(y.mean())) ** 2))
    metrics: dict[str, float | int] = {
        "num_test_examples": int(len(y)),
        "mean_abs_error": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "accuracy": float(np.mean(rounded == y)),
        "continuous_mean_abs_error": float(np.mean(np.abs(continuous_residual))),
        "continuous_rmse": float(np.sqrt(np.mean(continuous_residual**2))),
        "continuous_r2": float(1.0 - np.sum(continuous_residual**2) / denominator) if denominator > 0.0 else float("nan"),
    }
    return metrics, rounded


def probe_prediction_metrics(
    *, probe: MagnitudeProbe, representations: np.ndarray, truth: np.ndarray
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    features = np.asarray(probe.predict_features(representations), dtype=np.float64)
    y = np.asarray(truth, dtype=np.int64).reshape(-1)
    if features.ndim == 2 and features.shape[1] == 1:
        continuous = features[:, 0]
        metrics, rounded = rounded_prediction_metrics(
            truth=y,
            continuous_prediction=continuous,
            value_min=int(probe.support_values.min()),
            value_max=int(probe.support_values.max()),
        )
        return metrics, rounded, continuous
    rounded = np.asarray(probe.predict_values(representations), dtype=np.int64)
    residual = rounded.astype(np.float64) - y.astype(np.float64)
    metrics = {
        "num_test_examples": int(len(y)),
        "mean_abs_error": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "accuracy": float(np.mean(rounded == y)),
        "continuous_mean_abs_error": float("nan"),
        "continuous_rmse": float("nan"),
        "continuous_r2": float("nan"),
    }
    return metrics, rounded, np.full(len(y), np.nan, dtype=np.float64)


def translated_probe(
    probe: MagnitudeProbe,
    *, source_family_mean: np.ndarray,
    common_mean: np.ndarray,
) -> MagnitudeProbe:
    source = np.asarray(source_family_mean, dtype=np.float64).reshape(-1)
    common = np.asarray(common_mean, dtype=np.float64).reshape(-1)
    coefficients = np.asarray(probe.coefficients, dtype=np.float64)
    if len(source) != coefficients.shape[0] or len(common) != coefficients.shape[0]:
        raise ValueError("Mean and probe feature dimensions differ.")
    adjusted_bias = np.asarray(probe.bias, dtype=np.float64) + (source - common) @ coefficients
    return replace(probe, bias=np.asarray(adjusted_bias, dtype=np.float64))


def fit_intercept_only(
    *, coefficients: np.ndarray, calibration_x: np.ndarray, calibration_y: np.ndarray
) -> tuple[np.ndarray, float]:
    weight = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    x = np.asarray(calibration_x, dtype=np.float64)
    y = np.asarray(calibration_y, dtype=np.float64).reshape(-1)
    if x.shape != (len(y), len(weight)):
        raise ValueError("Calibration matrix shape does not match coefficients and labels.")
    intercept = float(np.mean(y - x @ weight))
    return weight, intercept


def fit_gain_and_intercept(
    *, direction: np.ndarray, calibration_x: np.ndarray, calibration_y: np.ndarray
) -> tuple[np.ndarray, float, float]:
    weight = np.asarray(direction, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(weight))
    if norm <= 0.0:
        raise ValueError("Cannot calibrate a zero direction.")
    unit = weight / norm
    x = np.asarray(calibration_x, dtype=np.float64)
    y = np.asarray(calibration_y, dtype=np.float64).reshape(-1)
    scores = x @ unit
    centered_scores = scores - float(scores.mean())
    denominator = float(np.sum(centered_scores**2))
    if denominator <= 0.0:
        raise ValueError("Calibration scores have zero variance.")
    gain = float(np.sum(centered_scores * (y - float(y.mean()))) / denominator)
    intercept = float(y.mean() - gain * scores.mean())
    return unit, gain, intercept


def balanced_family_assignment(
    *, values: np.ndarray, families: list[str], seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    y = np.asarray(values, dtype=np.int64).reshape(-1)
    if not families or len(y) % len(families) != 0:
        raise ValueError("The value count must be divisible by the number of source families.")
    rng = np.random.default_rng(int(seed))
    positions = rng.permutation(len(y))
    per_family = len(y) // len(families)
    assigned: dict[str, np.ndarray] = {}
    for family_idx, family in enumerate(families):
        selected = np.sort(positions[family_idx * per_family : (family_idx + 1) * per_family])
        assigned[str(family)] = selected.astype(np.int64)
    return y, assigned


def build_balanced_pooled_matrix(
    *, values: np.ndarray, family_arrays: dict[str, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    families = list(family_arrays)
    y, assignment = balanced_family_assignment(values=values, families=families, seed=seed)
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for family in families:
        selected = assignment[family]
        matrix = np.asarray(family_arrays[family], dtype=np.float32)
        if matrix.shape[0] != len(y):
            raise ValueError(f"Family {family} has {matrix.shape[0]} rows for {len(y)} values.")
        rows.append(matrix[selected])
        labels.append(y[selected])
        audit_rows.extend(
            {"family": family, "row_position": int(position), "source_value": int(y[position])}
            for position in selected
        )
    pooled_x = np.concatenate(rows, axis=0)
    pooled_y = np.concatenate(labels, axis=0)
    audit = pd.DataFrame(audit_rows).sort_values(["family", "source_value"]).reset_index(drop=True)
    if len(pooled_y) != len(y) or len(np.unique(pooled_y)) != len(y):
        raise ValueError("Balanced pooling must use every training value exactly once.")
    return pooled_y, pooled_x, audit


def median_control_row(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        raise ValueError("Cannot select a median control from an empty table.")
    ordered = rows.assign(_distance=(rows["control_value"].astype(float) - rows["control_value"].astype(float).median()).abs())
    return ordered.sort_values(["_distance", "control_value"], kind="stable").iloc[0].drop(labels=["_distance"])


def scalar_direction(probe: MagnitudeProbe) -> np.ndarray:
    coefficients = np.asarray(probe.coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 1:
        raise ValueError(f"Probe {probe.architecture} does not define one scalar direction: {coefficients.shape}.")
    direction = coefficients[:, 0]
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("Probe direction has zero norm.")
    return direction / norm


def direction_pair_rows(
    *, probes: dict[str, MagnitudeProbe], common_x: np.ndarray
) -> list[dict[str, Any]]:
    x = np.asarray(common_x, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(probes), 2):
        left_direction = scalar_direction(probes[left])
        right_direction = scalar_direction(probes[right])
        left_scores = x @ left_direction
        right_scores = x @ right_direction
        if float(np.std(left_scores)) <= 0.0 or float(np.std(right_scores)) <= 0.0:
            agreement = float("nan")
        else:
            agreement = float(np.corrcoef(left_scores, right_scores)[0, 1])
        rows.append(
            {
                "left_family": left,
                "right_family": right,
                "direction_cosine": float(left_direction @ right_direction),
                "prediction_agreement": agreement,
                "num_common_examples": int(len(x)),
            }
        )
    return rows


def clustered_direction_bootstrap(
    frame: pd.DataFrame,
    *, group_columns: list[str],
    metrics: tuple[str, ...] = ("direction_cosine", "prediction_agreement"),
    replicates: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap registered layers as clusters and family pairs within layers."""

    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie between zero and one.")
    required = set(group_columns) | {"relative_layer", "left_family", "right_family"} | set(metrics)
    if missing := required - set(frame.columns):
        raise ValueError(f"Direction bootstrap is missing columns: {sorted(missing)}")
    alpha = (1.0 - float(confidence_level)) / 2.0
    output: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, group in frame.groupby(grouper, sort=True, dropna=False):
        key_values = (group_key,) if len(group_columns) == 1 else tuple(group_key)
        layers = np.sort(group["relative_layer"].unique())
        if len(layers) == 0:
            continue
        group_seed = int(stable_hash({"seed": int(seed), "group": [str(value) for value in key_values]}, 8), 16)
        rng = np.random.default_rng(group_seed)
        samples = {metric: np.empty(int(replicates), dtype=np.float64) for metric in metrics}
        for replicate in range(int(replicates)):
            selected_layers = rng.choice(layers, size=len(layers), replace=True)
            selected_rows: list[pd.DataFrame] = []
            for layer in selected_layers:
                local = group.loc[np.isclose(group["relative_layer"].astype(float), float(layer))]
                positions = rng.integers(0, len(local), size=len(local))
                selected_rows.append(local.iloc[positions])
            sampled = pd.concat(selected_rows, ignore_index=True)
            for metric in metrics:
                samples[metric][replicate] = float(sampled[metric].mean())
        for metric in metrics:
            values = samples[metric]
            row = {column: value for column, value in zip(group_columns, key_values, strict=True)}
            row.update(
                {
                    "metric": metric,
                    "estimate": float(group[metric].mean()),
                    "ci_low": float(np.quantile(values, alpha)),
                    "ci_high": float(np.quantile(values, 1.0 - alpha)),
                    "confidence_level": float(confidence_level),
                    "bootstrap_replicates": int(replicates),
                    "num_layers": int(len(layers)),
                    "num_family_pairs": int(len(group)),
                    "bootstrap_unit": "relative_layer cluster; family pairs within layer",
                }
            )
            output.append(row)
    return pd.DataFrame(output)
