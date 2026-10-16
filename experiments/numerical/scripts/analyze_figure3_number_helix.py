#!/usr/bin/env python3
"""Fit and visualize numerical activation geometry for paper Figure 3.

CPU analysis of the exact activation cache used for the four-probe comparison.
All fitted maps and display bases use the original 800 training values.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.io import to_html

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "out/runs/source_value_rhs_addition_dataset_llama_3_1_8b_layer10__2026_06_15"
DEFAULT_REPR = RUN / "representations/llama_3_1_8b/layer_10/source_token_representations.npz"
DEFAULT_SPLIT = ROOT / "data/prompt_cache_canonical_source_splits_t800_d100_te100__a68a297dd188__r0/split_00/source_values.json"
STEM = "2026-09-07-figure3-number-helix"
DEFAULT_OUTPUT = ROOT / "experiment-reports/artifacts" / STEM
STYLE = "addition_formula_rhs_source"
RIDGE_LAMBDA = 0.551881046550891
PERIOD_MODELS = {
    "linear": (),
    "helix_100": (100,),
    "helix_1000_100_10": (1000, 100, 10),
    "helix_100_10_5_2": (100, 10, 5, 2),
    "helix_1000_100_10_5_2": (1000, 100, 10, 5, 2),
}
DISPLAY_MODEL = "helix_1000_100_10_5_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_centroids(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        selected = data["style_ids"].astype(str) == STYLE
        values = data["source_values"][selected].astype(int)
        contexts = data["x_values"][selected].astype(int)
        activations = data["activations"][selected].astype(np.float32)
    counts = np.bincount(values, minlength=1000)
    np.testing.assert_array_equal(counts, np.full(1000, 20))
    # Match the original loader's float32 averaging before float64 fitting.
    centroids = np.stack([activations[values == value].mean(0) for value in range(1000)])
    metadata = json.loads(path.with_name("metadata.json").read_text())
    assert metadata["model_name"] == "meta-llama/Llama-3.1-8B"
    assert metadata["layer_idx"] == 10 and centroids.shape == (1000, 4096)
    return centroids.astype(np.float64), {
        "cache": relative(path), "cache_sha256": sha256(path),
        "collection": metadata, "prompt_style": STYLE,
        "value_range": [0, 999], "selected_rows": int(selected.sum()),
        "centroids": 1000, "contexts_per_centroid": 20,
        "auxiliary_values": np.unique(contexts).tolist(),
        "centroid_aggregation": "float32 arithmetic mean, as in the original probe run",
    }


def design(values: np.ndarray, periods: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    columns = [np.ones(len(values)), (values - 499.5) / 999]
    for period in periods:
        columns.append(np.cos(2 * np.pi * values / period))
        # sin(pi*v) vanishes on the integer domain; omit its zero column.
        if period != 2:
            columns.append(np.sin(2 * np.pi * values / period))
    return np.column_stack(columns)


def reconstruction_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - np.sum((actual - predicted) ** 2) /
                 np.sum((actual - actual.mean(0)) ** 2))


def orient_rows(basis: np.ndarray) -> np.ndarray:
    basis = basis.copy()
    signs = np.sign(basis[np.arange(len(basis)), np.abs(basis).argmax(1)])
    return basis * signs[:, None]


def analyze(centroids: np.ndarray, splits: dict[str, np.ndarray]) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    train = splits["train"]
    values = np.arange(1000)
    mean = centroids[train].mean(0)
    centered = centroids - mean
    u, singular, vt = np.linalg.svd(centered[train], full_matrices=False)
    direction = (vt.T * (singular / (singular**2 + RIDGE_LAMBDA))) @ (u.T @ (train - train.mean()))
    bias = float(train.mean() - mean @ direction)
    prediction = centroids @ direction + bias
    rounded = np.clip(np.rint(prediction), 0, 999)
    test = splits["test"]
    probe_metrics = {
        "test_mae": float(np.mean(abs(rounded[test] - test))),
        "test_exact_accuracy": float(np.mean(rounded[test] == test)),
        "test_r2": float(1 - np.mean((rounded[test] - test)**2) / np.var(test)),
        "ridge_lambda": RIDGE_LAMBDA,
    }
    np.testing.assert_allclose(
        [probe_metrics["test_mae"], probe_metrics["test_exact_accuracy"], probe_metrics["test_r2"]],
        [8.96, 0.02, 0.9975901250398997], atol=1e-12,
    )
    scalar_axis = direction / np.linalg.norm(direction)
    transverse = centered - np.outer(centered @ scalar_axis, scalar_axis)
    _, _, transverse_vt = np.linalg.svd(transverse[train], full_matrices=False)
    probe_basis = np.vstack((scalar_axis, orient_rows(transverse_vt[:2])))
    pca_basis = orient_rows(vt[:3])
    np.testing.assert_allclose(probe_basis @ probe_basis.T, np.eye(3), atol=1e-12)
    probe_points = centered @ probe_basis.T
    pca_points = centered @ pca_basis.T
    np.testing.assert_allclose(
        probe_points[:, 0] * np.linalg.norm(direction) + train.mean(), prediction, atol=1e-10,
    )
    arrays = {
        "mean": mean, "ridge_direction": direction, "ridge_bias": np.array(bias),
        "probe_basis": probe_basis, "pca_basis": pca_basis,
    }
    records = []
    fit_points = {}
    for name, periods in PERIOD_MODELS.items():
        features = design(values, periods)
        coefficients, _, rank, _ = np.linalg.lstsq(features[train], centroids[train], rcond=None)
        assert rank == features.shape[1]
        fitted = features @ coefficients
        arrays[name + "_coefficients"] = coefficients
        fit_points[name] = {
            "probe": (fitted - mean) @ probe_basis.T,
            "pca": (fitted - mean) @ pca_basis.T,
        }
        for partition, indices in splits.items():
            record = {"model": name, "periods": ",".join(map(str, periods)),
                      "features": features.shape[1], "partition": partition, "n_values": len(indices)}
            for space, basis in (("full", None), ("probe_view", probe_basis), ("pca_view", pca_basis)):
                actual, predicted = centroids[indices], fitted[indices]
                if basis is not None:
                    actual, predicted = actual @ basis.T, predicted @ basis.T
                record[space + "_r2"] = reconstruction_r2(actual, predicted)
            records.append(record)
    point_data = pd.DataFrame({"value": values, "ridge_prediction": prediction})
    point_data["partition"] = ""
    for partition, indices in splits.items():
        point_data.loc[indices, "partition"] = partition
    assert point_data.partition.ne("").all()
    for space, points in (("probe", probe_points), ("pca", pca_points)):
        for dimension in range(3):
            point_data[f"{space}_{dimension+1}"] = points[:, dimension]
            point_data[f"{space}_fit_{dimension+1}"] = fit_points[DISPLAY_MODEL][space][:, dimension]
    metrics = pd.DataFrame(records)
    geometry = {
        "display_model": DISPLAY_MODEL, "periods": list(PERIOD_MODELS[DISPLAY_MODEL]),
        "fit": "ordinary least squares, 800 original training values, full 4096-dimensional centroids",
        "model_formula": "mu + a*(v-499.5)/999 + sum_p (b_p*cos(2*pi*v/p) + c_p*sin(2*pi*v/p))",
        "period_2": "cosine only on the integer domain",
        "model_comparison": "exploratory fixed period families; no optimized frequencies or test-fitted coefficients",
        "plot_basis_fit": "800 original training values, with no whitening or per-axis rescaling",
        "probe_view": "unit ridge direction and two leading PCs in its orthogonal complement",
        "pca_view": "first three principal components of centered training centroids",
        "probe_view_train_variance_fraction": float(np.sum(probe_points[train]**2) / np.sum(centered[train]**2)),
        "pca_view_train_variance_fraction": float(np.sum(pca_points[train]**2) / np.sum(centered[train]**2)),
        "ridge_pc1_absolute_cosine": float(abs(scalar_axis @ pca_basis[0])),
        "ridge_prediction_from_display": {"slope": float(np.linalg.norm(direction)), "intercept": float(train.mean())},
    }
    return {"probe_check": probe_metrics, "geometry": geometry}, arrays, metrics, point_data


def scatter_figure(points: pd.DataFrame, space: str, geometry: dict) -> go.Figure:
    labels = ["Ridge direction", "Transverse PC 1", "Transverse PC 2"] if space == "probe" else ["PC 1", "PC 2", "PC 3"]
    fig = go.Figure()
    for partition, size, symbol in (("train", 2.8, "circle"), ("dev", 3.4, "square"), ("test", 4.2, "diamond")):
        rows = points[points.partition == partition]
        fig.add_trace(go.Scatter3d(
            x=rows[f"{space}_1"], y=rows[f"{space}_2"], z=rows[f"{space}_3"],
            mode="markers", name=f"{partition} ({len(rows)})",
            marker={"size": size, "symbol": symbol, "color": rows.value, "colorscale": "Cividis",
                    "cmin": 0, "cmax": 999, "opacity": .8, "showscale": partition == "train",
                    "colorbar": {"title": "Number", "thickness": 12, "len": .65}},
            customdata=np.column_stack((rows.value, rows.ridge_prediction)),
            hovertemplate="Number %{customdata[0]}<br>Ridge prediction %{customdata[1]:.1f}<extra>%{fullData.name}</extra>",
        ))
    fig.add_trace(go.Scatter3d(
        x=points[f"{space}_fit_1"], y=points[f"{space}_fit_2"], z=points[f"{space}_fit_3"],
        mode="lines", name="Generalized helix fit", line={"color": "#ac7d00", "width": 3},
        customdata=points.value, hovertemplate="Fit at number %{customdata}<extra></extra>",
    ))
    if space == "probe":
        x = np.array([points.probe_1.min(), points.probe_1.max()])
        fig.add_trace(go.Scatter3d(
            x=x, y=[0, 0], z=[0, 0], mode="lines", name="Ridge direction",
            line={"color": "#333333", "width": 4, "dash": "dash"},
        ))
    fig.update_layout(
        height=660, margin={"l": 0, "r": 0, "t": 12, "b": 0}, paper_bgcolor="white",
        font={"family": "Space Grotesk, system-ui, sans-serif", "color": "#333333", "size": 13},
        legend={"orientation": "h", "y": 1.02},
        template="plotly_white",
        scene={"xaxis_title": labels[0], "yaxis_title": labels[1], "zaxis_title": labels[2],
               "xaxis": {"backgroundcolor": "white", "gridcolor": "#dedede"},
               "yaxis": {"backgroundcolor": "white", "gridcolor": "#dedede"},
               "zaxis": {"backgroundcolor": "white", "gridcolor": "#dedede"},
               "aspectmode": "data", "camera": {"projection": {"type": "orthographic"},
               "eye": {"x": 1.5, "y": 1.9, "z": .9}}},
    )
    return fig


def write_report(path: Path, points: pd.DataFrame, metrics: pd.DataFrame, provenance: dict) -> None:
    geometry = provenance["geometry"]
    fit = metrics[(metrics.model == DISPLAY_MODEL) & (metrics.partition == "test")].iloc[0]
    table = metrics[metrics.partition == "test"][["model", "features", "full_r2", "probe_view_r2", "pca_view_r2"]]
    table = table.rename(columns={"model": "Fitted family", "features": "Features",
                                "full_r2": "Full-space reconstruction R²",
                                "probe_view_r2": "Ridge-view reconstruction R²",
                                "pca_view_r2": "PCA-view reconstruction R²"})
    plots = [
        to_html(scatter_figure(points, space, geometry), full_html=False, include_plotlyjs=i == 0,
                config={"displaylogo": False, "responsive": True}, div_id=f"{space}-view")
        for i, space in enumerate(("probe", "pca"))
    ]
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>Number representations and generalized helix fits</title>
<style>
:root {{color-scheme:light;--primary:#ac7d00;--ink:#333333;--muted:#696969;--line:#dedede}}
* {{box-sizing:border-box}} body {{margin:0;background:#fff;color:var(--ink);font:16px/1.58 'Space Grotesk',system-ui,sans-serif}}
main {{max-width:1120px;margin:48px auto;padding:0 24px 72px}}
h1 {{font-size:2.4414rem;line-height:1.275;margin:0 0 24px}} h2 {{font-size:1.5625rem;line-height:1.275;margin:48px 0 18px}}
p {{max-width:90ch}} a {{color:#8c6000}} a:hover {{color:#694600}} :focus-visible {{outline:2px solid var(--primary);outline-offset:2px}}
.table-wrap {{overflow:auto}} table {{border-collapse:collapse;width:100%;font-size:14px}} th,td {{border-bottom:1px solid var(--line);padding:12px;text-align:left}}
th {{background:#fafafa}} .caption {{font-size:.8rem;color:var(--muted)}} details {{border-top:1px solid var(--line);padding:18px 0;margin-top:36px}} summary {{cursor:pointer;color:#8c6000}}
pre {{overflow:auto;font-size:12px;background:#fafafa;padding:18px;border-radius:12px 0 12px 0}}
@media(max-width:640px) {{main {{margin-top:24px;padding:0 12px 36px}} h1 {{font-size:1.9531rem}}}}
@media(prefers-reduced-motion:reduce) {{* {{scroll-behavior:auto}}}}
</style></head><body><main>
<h1>Number representations and generalized helix fits</h1>
<p>The paper distinguishes accurate decoding from claims about activation geometry and behavioral control.
Figure 3 needs to show how numerical values can be decoded from curved activation structure.
This analysis tests whether the activations underlying its probing results support an empirical replacement for the helix schematic.</p>
<p>The data are the exact Llama-3.1-8B layer-10 source-token cache used for panel (c).
Each of the 1,000 values from 0 to 999 is averaged over 20 addition contexts.
Fitting the original ridge probe reproduces its test MAE of 8.96, exact accuracy of 2%, and R² of 0.998.</p>
<h2>Curvature is visible; the helix fit captures part of it</h2>
<p>The generalized helix explains {fit.full_r2:.1%} of the held-out activation variance in the full space.
In the display that includes the ridge direction, it explains {fit.probe_view_r2:.1%} of the projected test variance.
That three-dimensional display retains {geometry['probe_view_train_variance_fraction']:.1%} of the training activation variance.
The visualization therefore supports a description of the displayed structure with an explicit reconstruction residual.</p>
<p>The first axis is the normalized ridge direction.
The other two axes maximize training variance in its orthogonal complement.
These three axes are orthonormal, and their units are unchanged.
Projection onto the first axis preserves the original affine ridge prediction exactly.
This direction has absolute cosine {geometry['ridge_pc1_absolute_cosine']:.2f} with PC1.</p>
{plots[0]}
<p class="caption">All 1,000 observed centroids are shown, colored by numerical value.
Marker shape identifies the original train, development, and test partitions.
The ochre line joins the generalized-helix predictions at consecutive integer values.
Drag to rotate, scroll to zoom, hover for values and predictions, and click legend entries to toggle traces.</p>
<h2>A view using ordinary PCA</h2>
<p>The first three principal components retain {geometry['pca_view_train_variance_fraction']:.1%} of training variance.
The same fit explains {fit.pca_view_r2:.1%} of the projected test variance.
PCA and the probe-containing view show the same activations under different orthogonal projections.</p>
{plots[1]}
<h2>Reconstruction in the full space and the displayed subspaces</h2>
<p>All coefficients are fitted on the original 800 training values.
The table evaluates predictions for the original 100 test values; the remaining 100 values form the development partition.
The fitted families compare a linear trend with fixed periodic terms.
The displayed family uses periods 1,000, 100, 10, 5, and 2; these choices are exploratory numerical hypotheses.
Its period-2 sine is zero on integers and is omitted.</p>
<div class="table-wrap">{table.to_html(index=False, float_format=lambda value: f"{value:.3f}", border=0)}</div>
<p class="caption">Reconstruction R² = 1 − Σ‖h − ĥ‖² / Σ‖h − mean(h)‖², evaluated over the test values in each named space.
This measures activation reconstruction; Figure 3(c)'s R² measures prediction of numerical values.
All estimates use one fixed partition, without resampling intervals.
Every fitted family, including the linear baseline, is reported.</p>
<h2>Implications for Figure 3</h2>
<p>A plot of the observed centroids with the fit keeps reconstruction error visible.
The ridge direction provides the direct link to panel (c).
Its distinction from PC1 prevents the geometric display from attributing the probe's accuracy to the principal axis.
The fit and the three-dimensional projection leave substantial activation variance unexplained.
Neither analysis tests behavioral control.</p>
<p>The next useful check would test the same fixed periods across additional layers or prompt families.
Such comparisons should keep value partitions, activation sites, and context averaging matched.
The present result supports an illustration for this single condition.</p>
<details><summary>Earlier interactive plots located</summary>
<p><a href="2026-04-23-llama-3-8b-circular-components-basis10.html">Llama 3 8B circular-component report, 23 April</a>
contains interactive PCA views of the original Llama 3 8B at layer 3 in identity prompts.
Its reported cell was selected using test MAE and differs from panel (c)'s checkpoint, layer, and prompts.</p>
<p><a href="2026-05-13-source-value-number-spline-0-999.html">Number spline report, 13 May</a>
contains interactive 3D centroid and spline views for 0–999 using Llama 3.2 3B at layer 8.
Both reports use Plotly. Their data differ from the cache analyzed here.</p></details>
<details><summary>Construction, provenance, and regeneration</summary>
<p>The same scalar-plus-Fourier model is fitted directly to 4,096-dimensional centroids, then projected for display.
The fit is not performed on a rendered curve. All 1,000 values are displayed without point selection, whitening, or axis rescaling.</p>
<pre>{html.escape(json.dumps(provenance, indent=2))}</pre></details>
</main></body></html>"""
    # Embedded Plotly shaders contain trailing spaces; keep generated diffs clean.
    path.write_text("\n".join(line.rstrip(" \t\r") for line in text.split("\n")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repr-npz", type=Path, default=DEFAULT_REPR)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "experiment-reports" / f"{STEM}.html")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centroids, source = load_centroids(args.repr_npz.resolve())
    split_json = json.loads(args.split_json.read_text())
    splits = {name: np.asarray(split_json[f"{name}_source_values"], dtype=int) for name in ("train", "dev", "test")}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(1000))
    assert [len(splits[name]) for name in ("train", "dev", "test")] == [800, 100, 100]
    summary, arrays, metrics, points = analyze(centroids, splits)
    points_path = args.output_dir / "projected_centroids.csv"
    metrics_path = args.output_dir / "reconstruction_metrics.csv"
    points.to_csv(points_path, index=False, float_format="%.10g")
    metrics.to_csv(metrics_path, index=False)
    np.savez_compressed(args.output_dir / "fitted_maps.npz", **arrays)
    provenance = {
        **summary, "source": source,
        "split": {"path": relative(args.split_json.resolve()), "sha256": sha256(args.split_json),
                  "sizes": {name: len(indices) for name, indices in splits.items()}, "seed": 1729},
        "generation": {"script": relative(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
                       "command": "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --frozen python scripts/analyze_figure3_number_helix.py",
                       "numpy_version": np.__version__, "pandas_version": pd.__version__, "plotly_version": plotly.__version__},
        "outputs": {relative(path.resolve()): sha256(path) for path in (points_path, metrics_path)},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(args.report, points, metrics, provenance)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
