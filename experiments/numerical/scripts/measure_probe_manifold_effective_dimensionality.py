#!/usr/bin/env python3
"""Measure ambient dimensionality of source-value probe and centroid manifolds.

This is the probe-side companion to
``measure_numeric_manifold_effective_dimensionality.py``.  It keeps the same
activation locus as the CausaLab-style numeric path analysis, then fits the
project's source-value probes and measures the dimensionality of the decoded
value path each probe exposes for interventions.  It also includes the matched
CausaLab-style centroid locus rows so the low-rank probe paths can be compared
directly against the actual centroid trajectory in the same report.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.io import to_html
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_numeric_manifold_effective_dimensionality import (  # noqa: E402
    adjacent_nn_rate,
    covariance_eigenvalues,
    effective_rank_metrics,
    html_table,
    orderings,
    summarize_random_null,
    unit_rows,
)
from src.magnitude_intervention import evaluate_probe_split, fit_magnitude_probe  # noqa: E402
from src.probe_architectures import canonicalize_probe_architecture_name  # noqa: E402
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text, ensure_dir  # noqa: E402


DEFAULT_RUN_DIR = ROOT / "out/runs/source_value_rhs_addition_dataset_llama_3_1_8b_layer10"
DEFAULT_REPR_NPZ = DEFAULT_RUN_DIR / "representations/llama_3_1_8b/layer_10/source_token_representations.npz"
DEFAULT_SPLIT_JSON = ROOT / "data/prompt_cache_canonical_source_splits_t800_d100_te100__a68a297dd188__r0/split_00/source_values.json"
DEFAULT_REPORT = ROOT / "experiment-reports" / f"{date.today().isoformat()}-probe-manifold-effective-dimensionality.html"

PROBE_SPECS: dict[str, dict[str, Any]] = {
    "linear": {"ridge_lambda": 0.551881046550891},
    "pls": {"n_components": 16, "scale": False},
    "circular_reg": {"basis_dim": 10},
    "circular_cls": {
        "basis_dim": 10,
        "hidden_dim": 100,
        "lr": 0.01,
        "weight_decay": 0.001,
        "betas": [0.9, 0.999],
        "max_epochs": 1000,
        "patience": 20,
        "val_fraction": 0.1,
        "batch_size": 256,
        "seed": 1729,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repr-npz", type=Path, default=DEFAULT_REPR_NPZ)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--style-id", default="addition_formula_rhs_source")
    parser.add_argument("--fit-x-modes", default="0,all", help="Comma-separated x filters: 0 and/or all.")
    parser.add_argument("--architectures", default="linear,pls,circular_reg,circular_cls")
    parser.add_argument("--random-controls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--twonn-trim-quantile", type=float, default=0.90)
    parser.add_argument("--twonn-min-points", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def as_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_fit_x_mode(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"all", "none", "pooled"}:
        return None
    return int(text)


def fit_x_label(fit_x: int | None) -> str:
    return "all_x" if fit_x is None else f"fixed_x{fit_x}"


def load_split_values(path: Path) -> dict[str, np.ndarray]:
    with open(as_root(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out = {
        "train": np.asarray(payload["train_source_values"], dtype=np.int64),
        "dev": np.asarray(payload["dev_source_values"], dtype=np.int64),
        "test": np.asarray(payload["test_source_values"], dtype=np.int64),
    }
    for name, values in out.items():
        if len(values) == 0:
            raise ValueError(f"Split {name!r} is empty in {path}.")
    return out


def load_centroids(repr_npz: Path, *, style_id: str, fit_x: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    path = as_root(repr_npz)
    with np.load(path) as loaded:
        activations = loaded["activations"].astype(np.float32)
        source_values = loaded["source_values"].astype(np.int64)
        style_ids = loaded["style_ids"].astype(str)
        x_values = loaded["x_values"].astype(np.int64)
    mask = style_ids == str(style_id)
    if fit_x is not None:
        mask &= x_values == int(fit_x)
    selected = activations[mask]
    selected_values = source_values[mask]
    if len(selected) == 0:
        raise ValueError(f"No rows matched style_id={style_id!r}, fit_x={fit_x!r}.")

    centroids = np.zeros((1000, selected.shape[1]), dtype=np.float32)
    counts = np.zeros(1000, dtype=np.int64)
    for value in range(1000):
        rows = selected[selected_values == value]
        if len(rows) == 0:
            raise ValueError(f"Missing value {value} for style_id={style_id!r}, fit_x={fit_x!r}.")
        centroids[value] = rows.mean(axis=0)
        counts[value] = len(rows)
    metadata = {
        "repr_npz": str(path),
        "style_id": str(style_id),
        "fit_x": "all" if fit_x is None else int(fit_x),
        "n_rows": int(len(selected)),
        "hidden_dim": int(selected.shape[1]),
        "min_count_per_value": int(counts.min()),
        "max_count_per_value": int(counts.max()),
        "mean_count_per_value": float(counts.mean()),
    }
    return centroids, metadata


def twonn_estimate(points: np.ndarray, *, trim_quantile: float, min_points: int) -> dict[str, Any]:
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D points, got {x.shape}.")
    if len(x) < min_points:
        return {
            "twonn_mle_dim": np.nan,
            "twonn_slope_dim": np.nan,
            "twonn_slope_r2": np.nan,
            "twonn_n_used": 0,
            "twonn_n_removed_duplicates": 0,
        }
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(x)
    distances, _ = nn.kneighbors(x, return_distance=True)
    r1 = distances[:, 1].astype(np.float64)
    r2 = distances[:, 2].astype(np.float64)
    ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > 0) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    if len(mu) < min_points:
        return {
            "twonn_mle_dim": np.nan,
            "twonn_slope_dim": np.nan,
            "twonn_slope_r2": np.nan,
            "twonn_n_used": int(len(mu)),
            "twonn_n_removed_duplicates": int((~ok).sum()),
        }
    log_mu = np.log(mu)
    mle_dim = float(len(log_mu) / np.sum(log_mu))
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    f = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / n
    x_fit = np.log(mu_sorted)
    y_fit = -np.log(np.maximum(1.0 - f, 1e-12))
    keep = f <= float(trim_quantile)
    x_fit = x_fit[keep]
    y_fit = y_fit[keep]
    denom = float(np.dot(x_fit, x_fit))
    slope = float(np.dot(x_fit, y_fit) / denom) if denom > 0 else np.nan
    pred = slope * x_fit
    ss_res = float(np.square(y_fit - pred).sum())
    ss_tot = float(np.square(y_fit - y_fit.mean()).sum())
    return {
        "twonn_mle_dim": mle_dim,
        "twonn_slope_dim": slope,
        "twonn_slope_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "twonn_n_used": int(len(mu)),
        "twonn_n_removed_duplicates": int((~ok).sum()),
        "twonn_median_mu": float(np.median(mu)),
        "twonn_trim_quantile": float(trim_quantile),
    }


def decoded_probe_points(probe: Any) -> np.ndarray:
    values = np.arange(1000, dtype=np.int64)
    features = probe.encode_values(values)
    return np.asarray(features @ probe.decoder_matrix, dtype=np.float64)


def significant_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = eig[np.isfinite(eig) & (eig > 0)]
    if eig.size == 0:
        return eig
    # Probe paths can be exactly rank-1 or have exactly constant tangents; in
    # those cases raw covariance leaves tiny numerical dust that should not be
    # counted as hundreds of ambient PCs.
    threshold = max(1e-18, float(eig[0]) * 1e-10)
    return eig[eig > threshold]


def rank_metrics(matrix: np.ndarray, *, prefix: str) -> dict[str, float | int]:
    return effective_rank_metrics(significant_eigenvalues(covariance_eigenvalues(matrix)), prefix=prefix)


def matrix_rank(matrix: np.ndarray, *, center: bool = True) -> int:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False, full_matrices=False)
    if len(s) == 0:
        return 0
    return int(np.sum(s > max(x.shape) * np.finfo(np.float64).eps * s[0]))


def orthonormal_basis(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return np.zeros((x.shape[1], 0), dtype=np.float64)
    keep = s > max(x.shape) * np.finfo(np.float64).eps * s[0]
    return vt[keep].T


def span_explained_fraction(points: np.ndarray, basis_points: np.ndarray) -> float:
    x = np.asarray(points, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    total = float(np.square(x).sum())
    if total <= 0:
        return np.nan
    basis = orthonormal_basis(basis_points)
    if basis.shape[1] == 0:
        return 0.0
    projected = x @ basis @ basis.T
    return float(np.square(projected).sum() / total)


def order_metrics(name: str, order: np.ndarray, points: np.ndarray) -> dict[str, Any]:
    ordered = points[order].astype(np.float64)
    diffs = np.diff(ordered, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    unit_diffs = unit_rows(diffs)
    tangent_cosines = np.sum(unit_diffs[:-1] * unit_diffs[1:], axis=1)
    second = np.diff(ordered, n=2, axis=0)
    row: dict[str, Any] = {
        "order": name,
        "path_length": float(step_lengths.sum()),
        "mean_step_length": float(step_lengths.mean()),
        "median_step_length": float(np.median(step_lengths)),
        "step_length_cv": float(step_lengths.std() / step_lengths.mean()) if step_lengths.mean() > 0 else 0.0,
        "chord_length": float(np.linalg.norm(ordered[-1] - ordered[0])),
        "path_chord_ratio": float(step_lengths.sum() / max(np.linalg.norm(ordered[-1] - ordered[0]), 1e-12)),
        "adjacent_nn_rate": adjacent_nn_rate(points.astype(np.float64), order),
        "mean_adjacent_tangent_cosine": float(tangent_cosines.mean()),
        "median_adjacent_tangent_cosine": float(np.median(tangent_cosines)),
        "mean_second_difference_norm": float(np.linalg.norm(second, axis=1).mean()),
        "second_difference_energy_per_step": float(np.square(second).sum() / max(1, len(second))),
        "raw_tangent_span_rank": matrix_rank(diffs, center=False),
        "unit_tangent_span_rank": matrix_rank(unit_diffs, center=False),
    }
    row.update(rank_metrics(diffs, prefix="raw_tangent_"))
    row.update(rank_metrics(unit_diffs, prefix="unit_tangent_"))
    return row


def probe_path_summary(
    *,
    architecture: str,
    fit_mode: str,
    points: np.ndarray,
    actual_centroids: np.ndarray,
    support_features: np.ndarray,
    random_controls: int,
    seed: int,
    trim_quantile: float,
    min_points: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    point_eig = covariance_eigenvalues(points)
    centered_points = np.asarray(points, dtype=np.float64) - np.asarray(points, dtype=np.float64).mean(axis=0, keepdims=True)
    point_row: dict[str, Any] = {
        "fit_mode": fit_mode,
        "architecture": architecture,
        "locus_type": "decoded_probe",
        "decoded_path_rank": matrix_rank(points),
        "support_feature_rank": matrix_rank(support_features),
        "actual_centroid_span_explained_by_probe": span_explained_fraction(actual_centroids, points),
        "probe_span_explained_by_actual_centroids": span_explained_fraction(points, actual_centroids),
    }
    point_row.update(effective_rank_metrics(significant_eigenvalues(point_eig), prefix="point_"))
    point_row.update(twonn_estimate(centered_points, trim_quantile=trim_quantile, min_points=min_points))
    feature_metrics = effective_rank_metrics(significant_eigenvalues(covariance_eigenvalues(support_features)), prefix="feature_")
    point_row.update(feature_metrics)

    order_rows = []
    for name, order in orderings(int(random_controls), int(seed)).items():
        row = order_metrics(name, order, points)
        row.update({"fit_mode": fit_mode, "architecture": architecture, "locus_type": "decoded_probe"})
        order_rows.append(row)
    order_summary = pd.DataFrame(order_rows)
    random_summary = summarize_random_null(order_summary)
    if not random_summary.empty:
        random_summary.insert(0, "locus_type", "decoded_probe")
        random_summary.insert(0, "architecture", architecture)
        random_summary.insert(0, "fit_mode", fit_mode)
    return point_row, order_summary, random_summary


def centroid_locus_summary(
    *,
    fit_mode: str,
    centroids: np.ndarray,
    random_controls: int,
    seed: int,
    trim_quantile: float,
    min_points: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    point_eig = covariance_eigenvalues(centroids)
    centered_points = np.asarray(centroids, dtype=np.float64) - np.asarray(centroids, dtype=np.float64).mean(axis=0, keepdims=True)
    point_row: dict[str, Any] = {
        "fit_mode": fit_mode,
        "architecture": "actual_centroid_locus",
        "locus_type": "causalab_centroid",
        "decoded_path_rank": matrix_rank(centroids),
        "support_feature_rank": np.nan,
        "actual_centroid_span_explained_by_probe": 1.0,
        "probe_span_explained_by_actual_centroids": 1.0,
    }
    point_row.update(effective_rank_metrics(significant_eigenvalues(point_eig), prefix="point_"))
    point_row.update(twonn_estimate(centered_points, trim_quantile=trim_quantile, min_points=min_points))
    for key in list(point_row):
        if key.startswith("feature_"):
            point_row[key] = np.nan

    order_rows = []
    for name, order in orderings(int(random_controls), int(seed)).items():
        row = order_metrics(name, order, centroids)
        row.update({"fit_mode": fit_mode, "architecture": "actual_centroid_locus", "locus_type": "causalab_centroid"})
        order_rows.append(row)
    order_summary = pd.DataFrame(order_rows)
    random_summary = summarize_random_null(order_summary)
    if not random_summary.empty:
        random_summary.insert(0, "locus_type", "causalab_centroid")
        random_summary.insert(0, "architecture", "actual_centroid_locus")
        random_summary.insert(0, "fit_mode", fit_mode)
    return point_row, order_summary, random_summary


def probe_eval_rows(probe: Any, centroids: np.ndarray, split_values: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, values in split_values.items():
        _, summary = evaluate_probe_split(
            probe=probe,
            representations=centroids[values],
            values=values,
            split_name=split,
        )
        rows.append(summary)
    _, summary = evaluate_probe_split(
        probe=probe,
        representations=centroids,
        values=np.arange(1000, dtype=np.int64),
        split_name="all_values",
    )
    rows.append(summary)
    return rows


def main_bar(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    for fit_mode in summary["fit_mode"].drop_duplicates().tolist():
        sub = summary[summary["fit_mode"] == fit_mode]
        fig.add_trace(go.Bar(x=sub["architecture"], y=sub[metric], name=str(fit_mode)))
    fig.update_layout(title=title, xaxis_title="Locus", yaxis_title=metric, barmode="group", height=430)
    return fig


def write_report(
    *,
    report_path: Path,
    artifact_dir: Path,
    config: dict[str, Any],
    point_summary: pd.DataFrame,
    order_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    eval_summary: pd.DataFrame,
    elapsed: float,
) -> None:
    numeric_orders = order_summary[order_summary["order"] == "numeric"].copy()
    merged = point_summary.merge(
        numeric_orders[
            [
                "fit_mode",
                "architecture",
                "path_length",
                "adjacent_nn_rate",
                "raw_tangent_span_rank",
                "raw_tangent_d95",
                "raw_tangent_participation_rank",
            ]
        ],
        on=["fit_mode", "architecture"],
        how="left",
    )
    compact_cols = [
        "fit_mode",
        "locus_type",
        "architecture",
        "decoded_path_rank",
        "point_d95",
        "raw_tangent_d95",
        "twonn_slope_dim",
        "actual_centroid_span_explained_by_probe",
        "path_length",
        "adjacent_nn_rate",
        "raw_tangent_span_rank",
    ]
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="tags" content="probe-manifold,centroid-locus,effective-dimensionality,pca,twonn">
<title>Probe And Centroid Manifold Effective Dimensionality</title>
<style>
body{{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033;line-height:1.46}}
h1{{font-size:30px;margin-bottom:4px}} h2{{font-size:21px;margin-top:34px;border-top:1px solid #d8dee8;padding-top:20px}}
.note{{color:#5b6878}} .callout{{border-left:4px solid #2563eb;background:#f6f9ff;padding:12px 14px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 18px}} th,td{{border:1px solid #d8dee8;padding:6px 8px;text-align:right}} th{{background:#edf1f6}} td:first-child,th:first-child{{text-align:left}}
code{{background:#eef2f7;padding:1px 4px;border-radius:4px}}
</style>
</head>
<body>
<h1>Probe And Centroid Manifold Effective Dimensionality</h1>
<p class="note">Generated {date.today().isoformat()}. Artifacts: <code>{html.escape(str(artifact_dir.relative_to(ROOT)))}</code>. Elapsed: {elapsed:.1f}s.</p>
<h2>Interpretation</h2>
<div class="callout">Rows labeled <code>actual_centroid_locus</code> are the matched CausaLab-style value centroids at the same layer-8 source-token locus. Rows labeled as probes are decoded intervention manifolds fitted on those centroids. Linear and current PLS are scalar-feature probes, so their decoded paths are rank-1. The circular probes use the configured 10D sinusoidal support code; their decoded paths are low-rank by construction. The key comparison is the actual centroid locus's high-dimensional trajectory versus how little of that centroid cloud variance the low-rank probe paths span.</div>
<h2>Run Configuration</h2>
{html_table(pd.DataFrame([config]))}
<h2>Compact Summary</h2>
{html_table(merged[compact_cols])}
<h2>Probe Prediction Quality</h2>
{html_table(eval_summary)}
<h2>Point-Cloud Metrics</h2>
{html_table(point_summary)}
<h2>Numeric-Order Tangent Metrics</h2>
{html_table(numeric_orders)}
<h2>Random-Order Null Summary</h2>
{html_table(random_summary)}
<h2>Figures</h2>
{to_html(main_bar(point_summary, "point_d95", "Point-cloud d95"), include_plotlyjs="cdn", full_html=False)}
{to_html(main_bar(point_summary, "actual_centroid_span_explained_by_probe", "Actual centroid variance explained by probe path span"), include_plotlyjs=False, full_html=False)}
{to_html(main_bar(merged, "raw_tangent_d95", "Numeric-order tangent d95"), include_plotlyjs=False, full_html=False)}
</body>
</html>
"""
    ensure_dir(report_path.parent)
    atomic_write_text(report_path, body)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    report_path = as_root(args.report_path)
    artifact_dir = as_root(args.artifact_dir) if args.artifact_dir else report_path.parent / "artifacts" / report_path.stem
    done_path = artifact_dir / "DONE"
    if done_path.exists() and report_path.exists() and not args.force:
        print(f"[skip] {report_path} already exists; use --force to rerun")
        return
    ensure_dir(artifact_dir)
    started = time.time()

    split_values = load_split_values(as_root(args.split_json))
    fit_modes = [parse_fit_x_mode(item) for item in parse_csv(args.fit_x_modes)]
    architectures = [canonicalize_probe_architecture_name(item) for item in parse_csv(args.architectures)]
    unknown = [arch for arch in architectures if arch not in PROBE_SPECS]
    if unknown:
        raise ValueError(f"Unsupported architectures: {unknown}")

    config = {
        "repr_npz": str(as_root(args.repr_npz)),
        "split_json": str(as_root(args.split_json)),
        "style_id": str(args.style_id),
        "fit_x_modes": ",".join(fit_x_label(item) for item in fit_modes),
        "architectures": ",".join(architectures),
        "random_controls": int(args.random_controls),
        "seed": int(args.seed),
        "device": str(args.device),
        "report_path": str(report_path),
    }

    point_rows: list[dict[str, Any]] = []
    order_parts: list[pd.DataFrame] = []
    random_parts: list[pd.DataFrame] = []
    eval_rows: list[dict[str, Any]] = []
    probe_meta_rows: list[dict[str, Any]] = []

    for fit_x in fit_modes:
        mode = fit_x_label(fit_x)
        print(f"[load] {mode}", flush=True)
        centroids, centroid_meta = load_centroids(as_root(args.repr_npz), style_id=str(args.style_id), fit_x=fit_x)
        print(f"[summarize] {mode} actual_centroid_locus", flush=True)
        point_row, order_summary, random_summary = centroid_locus_summary(
            fit_mode=mode,
            centroids=centroids,
            random_controls=int(args.random_controls),
            seed=int(args.seed),
            trim_quantile=float(args.twonn_trim_quantile),
            min_points=int(args.twonn_min_points),
        )
        point_row.update({f"centroid_{key}": value for key, value in centroid_meta.items() if key not in {"repr_npz", "style_id", "fit_x"}})
        point_rows.append(point_row)
        order_parts.append(order_summary)
        random_parts.append(random_summary)
        for architecture in architectures:
            spec = dict(PROBE_SPECS[architecture])
            if architecture == "circular_cls":
                spec["device"] = str(args.device)
                spec["seed"] = int(args.seed)
            print(f"[fit] {mode} {architecture}", flush=True)
            probe = fit_magnitude_probe(
                architecture=architecture,
                train_x=centroids[split_values["train"]],
                train_values=split_values["train"],
                config=spec,
                value_min=0,
                value_max=999,
            )
            probe_points = decoded_probe_points(probe)
            point_row, order_summary, random_summary = probe_path_summary(
                architecture=architecture,
                fit_mode=mode,
                points=probe_points,
                actual_centroids=centroids,
                support_features=np.asarray(probe.support_features, dtype=np.float64),
                random_controls=int(args.random_controls),
                seed=int(args.seed),
                trim_quantile=float(args.twonn_trim_quantile),
                min_points=int(args.twonn_min_points),
            )
            point_row.update({f"centroid_{key}": value for key, value in centroid_meta.items() if key not in {"repr_npz", "style_id", "fit_x"}})
            point_rows.append(point_row)
            order_parts.append(order_summary)
            random_parts.append(random_summary)
            for row in probe_eval_rows(probe, centroids, split_values):
                row.update({"fit_mode": mode, "probe_config": json.dumps(spec, sort_keys=True)})
                eval_rows.append(row)
            meta = {
                "fit_mode": mode,
                "architecture": architecture,
                "probe_config": json.dumps(spec, sort_keys=True),
                "coefficients_shape": list(np.asarray(probe.coefficients).shape),
                "decoder_shape": list(np.asarray(probe.decoder_matrix).shape),
                "support_features_shape": list(np.asarray(probe.support_features).shape),
            }
            meta.update(probe.metadata)
            probe_meta_rows.append(meta)

    point_summary = pd.DataFrame(point_rows)
    order_summary = pd.concat(order_parts, ignore_index=True)
    random_summary = pd.concat([part for part in random_parts if not part.empty], ignore_index=True)
    eval_summary = pd.DataFrame(eval_rows)
    probe_metadata = pd.DataFrame(probe_meta_rows)

    atomic_write_json(artifact_dir / "config.json", config)
    atomic_write_csv(artifact_dir / "point_cloud_summary.csv", point_summary)
    atomic_write_csv(artifact_dir / "order_tangent_summary.csv", order_summary)
    atomic_write_csv(artifact_dir / "random_order_null_summary.csv", random_summary)
    atomic_write_csv(artifact_dir / "probe_eval_summary.csv", eval_summary)
    atomic_write_csv(artifact_dir / "probe_metadata.csv", probe_metadata)
    write_report(
        report_path=report_path,
        artifact_dir=artifact_dir,
        config=config,
        point_summary=point_summary,
        order_summary=order_summary,
        random_summary=random_summary,
        eval_summary=eval_summary,
        elapsed=time.time() - started,
    )
    atomic_write_text(done_path, "done\n")
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
