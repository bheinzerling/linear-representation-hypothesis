#!/usr/bin/env python3
"""Measure ambient effective dimensionality of numeric centroid manifolds.

The intrinsic coordinate of a causalab-style numeric curve is one-dimensional by
construction. This script measures a different object: how many ambient
residual-stream directions are needed to describe the value centroids and the
ordered tangent directions of a candidate 0..999 path.
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
from plotly.io import to_html

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text, ensure_dir  # noqa: E402


DEFAULT_RUN_DIR = ROOT / "out/runs/source_value_rhs_addition_dataset__2026_05_13"
DEFAULT_REPR_NPZ = DEFAULT_RUN_DIR / "representations/llama_3_2_3b/layer_8/source_token_representations.npz"
DEFAULT_REPORT = ROOT / "experiment-reports" / f"{date.today().isoformat()}-numeric-manifold-effective-dimensionality.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repr-npz", type=Path, default=DEFAULT_REPR_NPZ)
    parser.add_argument("--style-id", default="addition_formula_rhs_source")
    parser.add_argument(
        "--fit-x",
        default="0",
        help="Left-operand/carrier value to filter before centroiding, or 'all' to pool all x values.",
    )
    parser.add_argument("--random-controls", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def as_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_fit_x(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"", "all", "none", "pooled"}:
        return None
    return int(text)


def effective_rank_metrics(eigenvalues: np.ndarray, *, prefix: str = "") -> dict[str, float | int]:
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = eig[np.isfinite(eig) & (eig > 0)]
    out: dict[str, float | int] = {}
    if eig.size == 0:
        for key in ["n_nonzero", "total_variance", "participation_rank", "entropy_rank", "stable_rank"]:
            out[f"{prefix}{key}"] = 0
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            out[f"{prefix}d{int(threshold * 100)}"] = 0
        return out
    total = float(eig.sum())
    probs = eig / total
    cumulative = np.cumsum(probs)
    out[f"{prefix}n_nonzero"] = int(eig.size)
    out[f"{prefix}total_variance"] = total
    out[f"{prefix}participation_rank"] = float((total * total) / np.sum(eig * eig))
    out[f"{prefix}entropy_rank"] = float(np.exp(-np.sum(probs * np.log(np.maximum(probs, 1e-300)))))
    out[f"{prefix}stable_rank"] = float(total / eig[0])
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        out[f"{prefix}d{int(threshold * 100)}"] = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    out[f"{prefix}top1_var"] = float(probs[0])
    out[f"{prefix}top2_var"] = float(probs[:2].sum())
    out[f"{prefix}top5_var"] = float(probs[:5].sum())
    out[f"{prefix}top10_var"] = float(probs[:10].sum())
    return out


def covariance_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eig = (singular_values * singular_values) / max(1, x.shape[0] - 1)
    return eig[eig > 0]


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def load_centroids(repr_npz: Path, style_id: str, fit_x: int | None) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    path = as_root(repr_npz)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        activations = loaded["activations"].astype(np.float32)
        source_values = loaded["source_values"].astype(np.int64)
        style_ids = loaded["style_ids"].astype(str)
        x_values = loaded["x_values"].astype(np.int64)
    mask = style_ids == str(style_id)
    if fit_x is not None:
        mask &= x_values == int(fit_x)
    if not np.any(mask):
        raise ValueError(f"No rows matched style_id={style_id!r}, fit_x={fit_x!r}")
    selected = activations[mask]
    selected_values = source_values[mask]
    centroids = np.zeros((1000, selected.shape[1]), dtype=np.float32)
    counts = np.zeros(1000, dtype=np.int64)
    within_sse = 0.0
    for value in range(1000):
        rows = selected[selected_values == value]
        counts[value] = len(rows)
        if len(rows) == 0:
            raise ValueError(f"Missing value {value} under style_id={style_id!r}, fit_x={fit_x!r}")
        centroid = rows.mean(axis=0)
        centroids[value] = centroid
        within_sse += float(np.square(rows.astype(np.float64) - centroid.astype(np.float64)).sum())
    total_sse = float(np.square(selected.astype(np.float64) - selected.mean(axis=0, keepdims=True).astype(np.float64)).sum())
    between_sse = float(np.square(centroids.astype(np.float64) - selected.mean(axis=0, keepdims=True).astype(np.float64)).sum() * float(counts.mean()))
    centroid_rows = pd.DataFrame({"value": np.arange(1000, dtype=np.int64), "count": counts})
    metadata = {
        "repr_npz": str(path),
        "style_id": str(style_id),
        "fit_x": "all" if fit_x is None else int(fit_x),
        "n_rows": int(len(selected)),
        "hidden_dim": int(selected.shape[1]),
        "min_count_per_value": int(counts.min()),
        "max_count_per_value": int(counts.max()),
        "mean_count_per_value": float(counts.mean()),
        "within_sse": within_sse,
        "total_sse": total_sse,
        "within_fraction_of_total_sse": float(within_sse / total_sse) if total_sse > 0 else None,
        "between_sse_approx_fraction": float(between_sse / total_sse) if total_sse > 0 else None,
    }
    return centroids, centroid_rows, metadata


def orderings(random_controls: int, seed: int) -> dict[str, np.ndarray]:
    values = np.arange(1000, dtype=np.int64)
    out = {
        "numeric": values,
        "reverse_numeric": values[::-1],
        "lexical_unpadded": np.asarray(sorted(values.tolist(), key=lambda value: str(int(value))), dtype=np.int64),
        "lexical_padded": np.asarray(sorted(values.tolist(), key=lambda value: f"{int(value):03d}"), dtype=np.int64),
        "ones_then_tens_hundreds": np.asarray(
            sorted(values.tolist(), key=lambda value: (int(value) % 10, (int(value) // 10) % 10, int(value) // 100)),
            dtype=np.int64,
        ),
        "hundreds_then_tens_ones": np.asarray(
            sorted(values.tolist(), key=lambda value: (int(value) // 100, (int(value) // 10) % 10, int(value) % 10)),
            dtype=np.int64,
        ),
    }
    rng = np.random.default_rng(seed)
    for idx in range(int(random_controls)):
        out[f"random_{idx:03d}"] = rng.permutation(values)
    return out


def adjacent_nn_rate(points: np.ndarray, order: np.ndarray) -> float:
    ordered = points[order]
    gram = ordered @ ordered.T
    sq = np.sum(ordered * ordered, axis=1)
    distances = sq[:, None] + sq[None, :] - 2.0 * gram
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    ok = np.zeros(len(order), dtype=bool)
    idx = np.arange(len(order))
    ok |= nearest == np.maximum(idx - 1, 0)
    ok |= nearest == np.minimum(idx + 1, len(order) - 1)
    ok[0] = nearest[0] == 1
    ok[-1] = nearest[-1] == len(order) - 2
    return float(ok.mean())


def order_metrics(name: str, order: np.ndarray, centroids: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    points = centroids[order].astype(np.float64)
    diffs = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    unit_diffs = unit_rows(diffs)
    tangent_cosines = np.sum(unit_diffs[:-1] * unit_diffs[1:], axis=1)
    second = np.diff(points, n=2, axis=0)
    raw_tangent_eig = covariance_eigenvalues(diffs)
    unit_tangent_eig = covariance_eigenvalues(unit_diffs)
    row: dict[str, Any] = {
        "order": name,
        "path_length": float(step_lengths.sum()),
        "mean_step_length": float(step_lengths.mean()),
        "median_step_length": float(np.median(step_lengths)),
        "step_length_cv": float(step_lengths.std() / step_lengths.mean()) if step_lengths.mean() > 0 else 0.0,
        "chord_length": float(np.linalg.norm(points[-1] - points[0])),
        "path_chord_ratio": float(step_lengths.sum() / max(np.linalg.norm(points[-1] - points[0]), 1e-12)),
        "adjacent_nn_rate": adjacent_nn_rate(centroids.astype(np.float64), order),
        "mean_adjacent_tangent_cosine": float(tangent_cosines.mean()),
        "median_adjacent_tangent_cosine": float(np.median(tangent_cosines)),
        "mean_second_difference_norm": float(np.linalg.norm(second, axis=1).mean()),
        "second_difference_energy_per_step": float(np.square(second).sum() / max(1, len(second))),
    }
    row.update(effective_rank_metrics(raw_tangent_eig, prefix="raw_tangent_"))
    row.update(effective_rank_metrics(unit_tangent_eig, prefix="unit_tangent_"))
    spectra = []
    for family, eig in [("raw_tangent", raw_tangent_eig), ("unit_tangent", unit_tangent_eig)]:
        total = float(eig.sum())
        probs = eig / total if total > 0 else np.zeros_like(eig)
        cumulative = np.cumsum(probs)
        for idx in range(min(200, len(eig))):
            spectra.append(
                {
                    "order": name,
                    "spectrum": family,
                    "component": idx + 1,
                    "eigenvalue": float(eig[idx]),
                    "explained_variance_ratio": float(probs[idx]),
                    "cumulative_explained_variance": float(cumulative[idx]),
                }
            )
    return row, pd.DataFrame(spectra)


def summarize_random_null(order_summary: pd.DataFrame) -> pd.DataFrame:
    random_rows = order_summary[order_summary["order"].astype(str).str.startswith("random_")].copy()
    if random_rows.empty:
        return pd.DataFrame()
    numeric_cols = random_rows.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        values = random_rows[col].dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            continue
        rows.append(
            {
                "metric": col,
                "random_mean": float(values.mean()),
                "random_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "random_p05": float(np.quantile(values, 0.05)),
                "random_p50": float(np.quantile(values, 0.50)),
                "random_p95": float(np.quantile(values, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def spectrum_figure(spectrum: pd.DataFrame, *, title: str) -> go.Figure:
    fig = go.Figure()
    keep_orders = ["numeric", "lexical_unpadded", "ones_then_tens_hundreds", "random_000"]
    for order in keep_orders:
        for family in ["point_cloud", "raw_tangent", "unit_tangent"]:
            sub = spectrum[(spectrum["order"] == order) & (spectrum["spectrum"] == family)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["component"],
                    y=sub["cumulative_explained_variance"],
                    mode="lines",
                    name=f"{order} | {family}",
                )
            )
    fig.update_layout(title=title, xaxis_title="Component", yaxis_title="Cumulative explained variance", height=520)
    return fig


def bar_figure(order_summary: pd.DataFrame, metric: str) -> go.Figure:
    display = order_summary[~order_summary["order"].astype(str).str.startswith("random_")].copy()
    random_rows = order_summary[order_summary["order"].astype(str).str.startswith("random_")].copy()
    if not random_rows.empty:
        display = pd.concat(
            [
                display,
                pd.DataFrame(
                    [
                        {
                            "order": "random_mean",
                            metric: float(random_rows[metric].mean()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    fig = go.Figure(go.Bar(x=display["order"], y=display[metric]))
    fig.update_layout(title=metric, xaxis_title="Order", yaxis_title=metric, height=420)
    return fig


def html_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    shown = frame if max_rows is None else frame.head(max_rows)
    return shown.to_html(index=False, escape=True, float_format=lambda value: f"{value:.4g}")


def write_report(
    *,
    report_path: Path,
    artifact_dir: Path,
    metadata: dict[str, Any],
    point_summary: pd.DataFrame,
    order_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    spectrum: pd.DataFrame,
    elapsed: float,
) -> None:
    numeric = order_summary[order_summary["order"] == "numeric"].iloc[0].to_dict()
    point = point_summary.iloc[0].to_dict()
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="tags" content="numeric-manifold,effective-dimensionality,pca,tangent-dimension">
<title>Numeric Manifold Effective Dimensionality</title>
<style>
body{{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033;line-height:1.46}}
h1{{font-size:30px;margin-bottom:4px}} h2{{font-size:21px;margin-top:34px;border-top:1px solid #d8dee8;padding-top:20px}}
.note{{color:#5b6878}} .callout{{border-left:4px solid #2563eb;background:#f6f9ff;padding:12px 14px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 18px}} th,td{{border:1px solid #d8dee8;padding:6px 8px;text-align:right}} th{{background:#edf1f6}} td:first-child,th:first-child{{text-align:left}}
code{{background:#eef2f7;padding:1px 4px;border-radius:4px}}
</style>
</head>
<body>
<h1>Numeric Manifold Effective Dimensionality</h1>
<p class="note">Generated {date.today().isoformat()}. Artifacts: <code>{html.escape(str(artifact_dir.relative_to(ROOT)))}</code>. Elapsed: {elapsed:.1f}s.</p>
<h2>Interpretation</h2>
<div class="callout">The causalab-style curve is intrinsically 1D by parameterization, but the <em>ambient</em> centroid cloud is not low-dimensional in the same sense. For this run, the centroid point cloud needs d95={int(point["point_d95"])} PCs and has participation rank {float(point["point_participation_rank"]):.1f}. The numeric-order raw tangent directions need d95={int(numeric["raw_tangent_d95"])} PCs and have participation rank {float(numeric["raw_tangent_participation_rank"]):.1f}.</div>
<p>These metrics describe representation-space occupancy, not causal controllability. They are useful for testing whether a one-parameter path is merely a high-dimensional interpolating curve.</p>
<h2>Run Metadata</h2>
{html_table(pd.DataFrame([metadata]))}
<h2>Point-Cloud Dimensionality</h2>
{html_table(point_summary)}
<h2>Order And Tangent Dimensionality</h2>
{html_table(order_summary[~order_summary["order"].astype(str).str.startswith("random_")])}
<h2>Random-Order Null Summary</h2>
{html_table(random_summary)}
<h2>Figures</h2>
{to_html(spectrum_figure(spectrum, title="Cumulative variance spectra"), include_plotlyjs="cdn", full_html=False)}
{to_html(bar_figure(order_summary, "raw_tangent_participation_rank"), include_plotlyjs=False, full_html=False)}
{to_html(bar_figure(order_summary, "path_length"), include_plotlyjs=False, full_html=False)}
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

    fit_x = parse_fit_x(args.fit_x)
    print("[load] centroids", flush=True)
    centroids, centroid_rows, metadata = load_centroids(as_root(args.repr_npz), str(args.style_id), fit_x)
    point_eig = covariance_eigenvalues(centroids)
    point_summary = pd.DataFrame([{**effective_rank_metrics(point_eig, prefix="point_")}])
    point_spectrum_rows = []
    point_probs = point_eig / point_eig.sum()
    point_cum = np.cumsum(point_probs)
    for idx in range(min(200, len(point_eig))):
        point_spectrum_rows.append(
            {
                "order": "all_orders",
                "spectrum": "point_cloud",
                "component": idx + 1,
                "eigenvalue": float(point_eig[idx]),
                "explained_variance_ratio": float(point_probs[idx]),
                "cumulative_explained_variance": float(point_cum[idx]),
            }
        )
    # Duplicate point spectrum under representative orders to simplify plotting.
    point_spectrum = pd.DataFrame(point_spectrum_rows)
    point_spectrum = pd.concat(
        [
            point_spectrum.assign(order=order)
            for order in ["numeric", "lexical_unpadded", "ones_then_tens_hundreds", "random_000"]
        ],
        ignore_index=True,
    )

    order_rows = []
    spectrum_parts = [point_spectrum]
    for idx, (name, order) in enumerate(orderings(int(args.random_controls), int(args.seed)).items()):
        if idx < 12 or name.startswith("random_"):
            print(f"[order] {name}", flush=True)
        row, spec = order_metrics(name, order, centroids)
        order_rows.append(row)
        spectrum_parts.append(spec)
    order_summary = pd.DataFrame(order_rows)
    random_summary = summarize_random_null(order_summary)
    spectrum = pd.concat(spectrum_parts, ignore_index=True)

    metadata.update(
        {
            "random_controls": int(args.random_controls),
            "seed": int(args.seed),
            "report_path": str(report_path),
        }
    )
    atomic_write_json(artifact_dir / "config.json", metadata)
    atomic_write_csv(artifact_dir / "centroid_counts.csv", centroid_rows)
    atomic_write_csv(artifact_dir / "point_cloud_summary.csv", point_summary)
    atomic_write_csv(artifact_dir / "order_tangent_summary.csv", order_summary)
    atomic_write_csv(artifact_dir / "random_order_null_summary.csv", random_summary)
    atomic_write_csv(artifact_dir / "spectra.csv", spectrum)
    write_report(
        report_path=report_path,
        artifact_dir=artifact_dir,
        metadata=metadata,
        point_summary=point_summary,
        order_summary=order_summary,
        random_summary=random_summary,
        spectrum=spectrum,
        elapsed=time.time() - started,
    )
    atomic_write_text(done_path, "done\n")
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
