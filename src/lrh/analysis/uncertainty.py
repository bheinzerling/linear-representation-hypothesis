"""Conditional evaluation bootstraps used by the current paper figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

REPLICATES = 10_000
SEED = 20261013
PROBE_METHODS = ("linear", "pls", "circular_reg", "circular_cls")
PROBE_COLUMNS = ("r2", "accuracy", "mean_abs_error", "median_abs_error")
STEERING_METHODS = ("linear", "circular_reg", "centroid_patch", "no_edit")
PAIR_KEYS = ("baseline_row_idx", "target_slot", "source_value", "target_value", "target_output")


def limits(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high)


def probe_uncertainty(rows: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    reference = None
    indices = np.random.default_rng(SEED).integers(0, 100, size=(REPLICATES, 100))
    for method in PROBE_METHODS:
        point = metrics.loc[method]
        part = rows.loc[rows.architecture.eq(method)].sort_values("value").reset_index(drop=True)
        if part.empty:
            if method != "circular_cls":
                raise ValueError(f"Missing prediction rows for {method}")
            for column in PROBE_COLUMNS:
                records.append(
                    {
                        "architecture": method,
                        "metric": column,
                        "estimate": float(point[column]),
                        "low": np.nan,
                        "high": np.nan,
                        "n": 100,
                        "replicates": 0,
                        "status": "per_value_predictions_unavailable",
                    }
                )
            continue
        if len(part) != 100 or part.value.nunique() != 100:
            raise ValueError(f"Expected 100 distinct held-out values for {method}")
        targets = part.value.to_numpy(dtype=float)
        if reference is None:
            reference = targets
        else:
            np.testing.assert_array_equal(targets, reference)
        errors = part.prediction.to_numpy(dtype=float) - targets
        np.testing.assert_array_equal(np.abs(errors), part.abs_error.to_numpy())
        point_checks = {
            "mean_abs_error": np.abs(errors).mean(),
            "median_abs_error": np.median(np.abs(errors)),
            "accuracy": (errors == 0).mean(),
            "r2": 1 - np.square(errors).mean() / np.var(targets),
        }
        for column, value in point_checks.items():
            np.testing.assert_allclose(value, point[column], rtol=0, atol=1e-12)
        sampled_errors, sampled_targets = errors[indices], targets[indices]
        samples = {
            "mean_abs_error": np.abs(sampled_errors).mean(axis=1),
            "median_abs_error": np.median(np.abs(sampled_errors), axis=1),
            "accuracy": (sampled_errors == 0).mean(axis=1),
            "r2": 1 - np.square(sampled_errors).mean(axis=1) / np.var(sampled_targets, axis=1),
        }
        for column in PROBE_COLUMNS:
            low, high = limits(samples[column])
            records.append(
                {
                    "architecture": method,
                    "metric": column,
                    "estimate": float(point[column]),
                    "low": low,
                    "high": high,
                    "n": 100,
                    "replicates": REPLICATES,
                    "status": "conditional_value_bootstrap",
                }
            )
    return pd.DataFrame(records)


def steering_uncertainty(rows: pd.DataFrame) -> pd.DataFrame:
    parts, reference = {}, None
    for method in STEERING_METHODS[:-1]:
        part = (
            rows.loc[rows.method.eq(method)].sort_values(list(PAIR_KEYS[:2])).reset_index(drop=True)
        )
        if len(part) != 2000 or part.duplicated(list(PAIR_KEYS[:2])).any():
            raise ValueError(f"Expected 2,000 distinct edits for {method}")
        if not part.groupby("baseline_row_idx").size().eq(20).all():
            raise ValueError(f"Expected twenty targets per source for {method}")
        if part.baseline_row_idx.nunique() != 100 or part.source_value.nunique() != 100:
            raise ValueError(f"Expected 100 source clusters for {method}")
        matched = part[list(PAIR_KEYS) + ["no_edit_output"]]
        if reference is None:
            reference = matched
        else:
            pd.testing.assert_frame_equal(reference, matched)
        parts[method] = part
    baseline = parts["linear"].copy()
    baseline["output"] = baseline.no_edit_output
    parts["no_edit"] = baseline
    indices = np.random.default_rng(SEED).integers(0, 100, size=(REPLICATES, 100))
    draws, records = {}, []
    for method, part in parts.items():
        numeric_output = pd.to_numeric(part.output, errors="coerce")
        errors = (numeric_output - part.target_output).abs()
        for metric, successes in (("exact", errors.eq(0)), ("within10", errors.le(10))):
            clusters = successes.groupby(part.baseline_row_idx).mean().sort_index().to_numpy()
            samples = clusters[indices].mean(axis=1)
            draws[method, metric] = samples
            count = int(successes.sum())
            if count == 0:
                low, high = np.nan, np.nan
                status = "zero_successes_interval_unavailable"
            else:
                low, high = limits(samples)
                status = "conditional_source_cluster_bootstrap"
            records.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": count / len(part),
                    "count": count,
                    "n": len(part),
                    "low": low,
                    "high": high,
                    "clusters": 100,
                    "targets_per_cluster": 20,
                    "replicates": REPLICATES,
                    "status": status,
                }
            )
    for record in records:
        baseline_samples = draws["no_edit", record["metric"]]
        paired = draws[record["method"], record["metric"]] - baseline_samples
        baseline_estimate = next(
            r["estimate"]
            for r in records
            if r["method"] == "no_edit" and r["metric"] == record["metric"]
        )
        record["change_from_no_edit"] = record["estimate"] - baseline_estimate
        record["change_low"], record["change_high"] = (
            (np.nan, np.nan) if np.ptp(paired) == 0 else limits(paired)
        )
    return pd.DataFrame(records)
