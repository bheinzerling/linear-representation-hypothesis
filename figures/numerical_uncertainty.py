#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/figures/conditional_uncertainty"
PROBE_ROWS = ROOT / "data/figures/probe_tests/reproduced_test_predictions.csv"
PROBE_METRICS = ROOT / "data/figures/decodability_efficiency/probe_metrics.csv"
STEERING_ROWS = ROOT / "data/figures/numerical_sufficiency_atlas/steering_accuracy_rows.csv.gz"
REPLICATES = 10000
SEED = 20261013
LEVEL = .95
PROBE_METHODS = ("linear", "pls", "circular_reg", "circular_cls")
PROBE_COLUMNS = ("r2", "accuracy", "mean_abs_error", "median_abs_error")
STEERING_METHODS = ("linear", "circular_reg", "centroid_patch", "no_edit")
PAIR_KEYS = ("baseline_row_idx", "target_slot", "source_value", "target_value", "target_output")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limits(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2], method="linear")
    return float(lo), float(hi)


def probe_uncertainty(rows: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired value bootstrap, conditional on the three available fitted probes."""
    records = []
    reference = None
    indices = np.random.default_rng(SEED).integers(0, 100, size=(REPLICATES, 100))
    for method in PROBE_METHODS:
        point = metrics.loc[method]
        part = rows.loc[rows.architecture.eq(method)].sort_values("value").reset_index(drop=True)
        if part.empty:
            assert method == "circular_cls"
            for column in PROBE_COLUMNS:
                records.append(dict(architecture=method, metric=column, estimate=float(point[column]),
                                    low=np.nan, high=np.nan, n=100, replicates=0,
                                    status="per_value_predictions_unavailable"))
            continue
        assert len(part) == 100 and part.value.nunique() == 100
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
        sampled_errors = errors[indices]
        sampled_targets = targets[indices]
        samples = {
            "mean_abs_error": np.abs(sampled_errors).mean(axis=1),
            "median_abs_error": np.median(np.abs(sampled_errors), axis=1),
            "accuracy": (sampled_errors == 0).mean(axis=1),
            "r2": 1 - np.square(sampled_errors).mean(axis=1) / np.var(sampled_targets, axis=1),
        }
        for column in PROBE_COLUMNS:
            lo, hi = limits(samples[column])
            records.append(dict(architecture=method, metric=column, estimate=float(point[column]),
                                low=lo, high=hi, n=100, replicates=REPLICATES,
                                status="conditional_value_bootstrap"))
    return pd.DataFrame(records)


def steering_uncertainty(rows: pd.DataFrame) -> pd.DataFrame:
    """Paired source bootstrap with all twenty targets retained in each cluster."""
    parts = {}
    reference = None
    for method in STEERING_METHODS[:-1]:
        part = rows.loc[rows.method.eq(method)].sort_values(list(PAIR_KEYS[:2])).reset_index(drop=True)
        assert len(part) == 2000 and not part.duplicated(list(PAIR_KEYS[:2])).any()
        assert part.groupby("baseline_row_idx").size().eq(20).all()
        assert part.baseline_row_idx.nunique() == part.source_value.nunique() == 100
        if reference is None:
            reference = part[list(PAIR_KEYS) + ["no_edit_output"]]
        else:
            pd.testing.assert_frame_equal(reference, part[list(PAIR_KEYS) + ["no_edit_output"]])
        parts[method] = part
    baseline = parts["linear"].copy()
    baseline["output"] = baseline.no_edit_output
    parts["no_edit"] = baseline
    indices = np.random.default_rng(SEED).integers(0, 100, size=(REPLICATES, 100))
    draws = {}
    records = []
    for method, part in parts.items():
        errors = (part.output - part.target_output).abs()
        for metric, successes in (("exact", errors.eq(0)), ("within10", errors.le(10))):
            clusters = successes.groupby(part.baseline_row_idx).mean().sort_index().to_numpy()
            assert len(clusters) == 100
            samples = clusters[indices].mean(axis=1)
            draws[method, metric] = samples
            count = int(successes.sum())
            if count == 0:
                # Resampling only observed failures cannot estimate an upper
                # success bound. Keep the zero count and omit that interval.
                lo, hi = np.nan, np.nan
                status = "zero_successes_interval_unavailable"
            else:
                lo, hi = limits(samples)
                status = "conditional_source_cluster_bootstrap"
            records.append(dict(method=method, metric=metric, estimate=count / len(part),
                                count=count, n=len(part), low=lo, high=hi, clusters=100,
                                targets_per_cluster=20, replicates=REPLICATES, status=status))
    for record in records:
        key = (record["method"], record["metric"])
        baseline_samples = draws["no_edit", record["metric"]]
        paired_samples = draws[key] - baseline_samples
        baseline_estimate = next(r["estimate"] for r in records
                                 if r["method"] == "no_edit" and r["metric"] == record["metric"])
        record["change_from_no_edit"] = record["estimate"] - baseline_estimate
        # A paired empirical distribution consisting only of zero changes
        # supplies no uncertainty about new source contexts.
        if np.ptp(paired_samples) == 0:
            record["change_low"], record["change_high"] = np.nan, np.nan
        else:
            record["change_low"], record["change_high"] = limits(paired_samples)
    return pd.DataFrame(records)


def load_probe_uncertainty(metrics: pd.DataFrame) -> pd.DataFrame:
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    path = OUTPUT / "probe_metrics.csv"
    assert digest(path) == provenance["outputs"][path.name]
    assert digest(PROBE_ROWS) == provenance["inputs"][PROBE_ROWS.relative_to(ROOT).as_posix()]
    assert digest(PROBE_METRICS) == provenance["inputs"][PROBE_METRICS.relative_to(ROOT).as_posix()]
    frame = pd.read_csv(path).set_index(["architecture", "metric"])
    for method in PROBE_METHODS:
        for metric in PROBE_COLUMNS:
            np.testing.assert_allclose(frame.loc[(method, metric), "estimate"],
                                       metrics.loc[method, metric], rtol=0, atol=1e-12)
    return frame


def load_steering_uncertainty() -> pd.DataFrame:
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    path = OUTPUT / "steering_accuracy.csv"
    assert digest(path) == provenance["outputs"][path.name]
    assert digest(STEERING_ROWS) == provenance["inputs"][STEERING_ROWS.relative_to(ROOT).as_posix()]
    return pd.read_csv(path).set_index(["method", "metric"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(PROBE_METRICS).set_index("architecture")
    probe = probe_uncertainty(pd.read_csv(PROBE_ROWS), metrics)
    steering = steering_uncertainty(pd.read_csv(STEERING_ROWS))
    outputs = {}
    for name, frame in (("probe_metrics.csv", probe), ("steering_accuracy.csv", steering)):
        path = OUTPUT / name
        frame.to_csv(path, index=False, float_format="%.12g")
        outputs[name] = digest(path)
    deletion_path = ROOT / "data/figures/deletion/provenance.json"
    deletion = json.loads(deletion_path.read_text())
    count_path = ROOT / "data/figures/number_property_evidence_main/contradiction_counts.csv"
    provenance = {
        "command": "uv run python figures/numerical_uncertainty.py",
        "script_sha256": digest(Path(__file__)),
        "inputs": {p.relative_to(ROOT).as_posix(): digest(p)
                   for p in (PROBE_ROWS, PROBE_METRICS, STEERING_ROWS, deletion_path, count_path)},
        "bootstrap": {"level": LEVEL, "replicates": REPLICATES, "seed": SEED,
                      "quantile_method": "linear", "numpy": np.__version__},
        "probe": {
            "unit": "heldout numerical value",
            "n": 100,
            "paired_across_methods": True,
            "conditioning": "one frozen split and fitted probe per method; fitting uncertainty excluded",
            "r2_denominator": "target variance recomputed in each resampled evaluation set",
            "classifier_exception": "Aggregate scores and a median certificate are available; per-value predictions are absent, so no classifier interval is estimated.",
        },
        "steering": {
            "unit": "baseline_row_idx, identifying one source context and value",
            "clusters": 100, "targets_per_cluster": 20, "n_per_method": 2000,
            "paired_across_methods_and_baseline": True,
            "conditioning": "frozen model, intervention procedures, and twenty targets per source; uncertainty from fitting the edits excluded",
            "zero_successes": "Display the observed count. Percentile resampling of all-zero observations cannot estimate an upper success bound; interval endpoints are unavailable.",
            "constant_paired_change": "An interval is omitted when all resampled paired changes are identical.",
        },
        "existing_deletion_intervals": deletion["bootstrap"] | {
            "prompt_instances_per_task": 1000, "distinct_source_values": 639,
            "fitted_bases": 10, "overlapping_splits": True,
        },
        "contradiction_counts": {
            "conditioning": "fixed baseline-correct sets of 100 classification queries and fifty enumeration prompts",
            "intervals": "Counts are descriptive; a sampling design for population intervals is unavailable.",
        },
        "outputs": outputs,
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(probe.to_string(index=False))
    print(steering.to_string(index=False))


if __name__ == "__main__":
    main()
