#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "experiment-reports" / "artifacts" / "2026-04-18-source-value-probe-capacity-sweep-split00-relative-layers"
DEFAULT_REPORT_PATH = ROOT / "experiment-reports" / "2026-04-20-source-value-probe-capacity-sweep-split00-relative-layers-capacity-dev.html"
DEFAULT_REPORT_STEM = DEFAULT_REPORT_PATH.stem

MODEL_ORDER = ["llama_3_2_1b", "llama_3_2_3b", "llama_3_8b", "olmo_3_7b_base", "olmo_2_13b", "gpt_oss_20b"]
PROBE_ARCH_ORDER = ["circular_reg", "circular_cls", "pls", "linear"]
BASELINE_ARCHITECTURE = "mean_baseline"
BASELINE_LINE_COLOR = "#7f7f7f"
ARCH_ORDER = list(PROBE_ARCH_ORDER)
ARCH_COLORS = {"circular_reg": "#b2182b", "circular_cls": "#2166ac", "pls": "#1b7837", "linear": "#762a83"}
LAYER_DASHES = {0.0: "solid", 0.1: "dash", 0.5: "dot", 0.9: "dashdot"}
SELECTED_LAYERS = [0.0, 0.1, 0.5, 0.9]
WIN_GRID = np.linspace(0.0, 1.0, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a development-only source-value probe capacity diagnostics report.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--report-stem", default=DEFAULT_REPORT_STEM)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    atomic_write_text(path, lines + ("\n" if lines else ""))


def _ordered_labels(frame: pd.DataFrame, *, key_col: str, label_col: str, preferred_keys: list[str] | None = None) -> list[str]:
    ordered: list[str] = []
    if preferred_keys is not None and key_col in frame.columns:
        key_to_label = (
            frame.loc[:, [key_col, label_col]]
            .drop_duplicates()
            .set_index(key_col)[label_col]
            .to_dict()
        )
        for key in preferred_keys:
            if key in key_to_label and key_to_label[key] not in ordered:
                ordered.append(key_to_label[key])
    for label in frame[label_col].drop_duplicates().tolist():
        if label not in ordered:
            ordered.append(label)
    return ordered


def _subplot_dims(n_items: int, *, n_cols: int = 3) -> tuple[int, int]:
    cols = min(n_cols, max(1, n_items))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, value)):02x}" for value in rgb)


def _blend_with_white(hex_color: str, amount: float) -> str:
    base = _hex_to_rgb(hex_color)
    blended = tuple(int(round(channel * (1.0 - amount) + 255 * amount)) for channel in base)
    return _rgb_to_hex(blended)


def _effective_df_ticks(max_effective_df: float) -> tuple[list[float], list[str]]:
    capped_max = max(float(max_effective_df), 1.0)
    max_exponent = int(math.floor(math.log2(capped_max)))
    tickvals = [float(2**exponent) for exponent in range(max_exponent + 1)]
    if not math.isclose(tickvals[-1], capped_max, rel_tol=1e-9, abs_tol=1e-9):
        tickvals.append(capped_max)
    ticktext = [f"{int(round(value))}" if float(value).is_integer() else f"{value:.0f}" for value in tickvals]
    return tickvals, ticktext


def _load_inputs(artifact_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_df = pd.read_json(artifact_dir / "summary_metrics.jsonl", lines=True)
    summary_df["architecture"] = summary_df["architecture"].replace({"circular": "circular_reg"})
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    return summary_df, metadata


def _prepare_mean_curves(summary_df: pd.DataFrame) -> pd.DataFrame:
    test_df = summary_df.loc[
        (summary_df["partition"] == "test") & (summary_df["architecture"].isin(PROBE_ARCH_ORDER))
    ].copy()
    max_step_by_arch = test_df.groupby("architecture")["step_index"].max().to_dict()
    test_df["max_step_index"] = test_df["architecture"].map(max_step_by_arch)
    test_df["normalized_step"] = np.where(test_df["max_step_index"] > 0, test_df["step_index"] / test_df["max_step_index"], 0.0)
    test_df["log2_effective_df"] = np.log2(np.clip(test_df["effective_df"].astype(float), a_min=1.0, a_max=None))
    grouped = (
        test_df.groupby(
            ["model_alias", "model_label", "architecture", "relative_layer", "step_index", "max_step_index"],
            as_index=False,
        )
        .agg(
            model_dim=("model_dim", "first"),
            normalized_step=("normalized_step", "mean"),
            layer_idx_min=("layer_idx", "min"),
            layer_idx_max=("layer_idx", "max"),
            num_prompt_styles=("family_label", "nunique"),
            control_name=("control_name", "first"),
            control_value_min=("control_value", "min"),
            control_value_max=("control_value", "max"),
            effective_df_min=("effective_df", "min"),
            effective_df_max=("effective_df", "max"),
            log2_effective_df_mean=("log2_effective_df", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            mean_abs_error_mean=("mean_abs_error", "mean"),
            mean_abs_error_std=("mean_abs_error", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .sort_values(["model_alias", "architecture", "relative_layer", "step_index"])
        .reset_index(drop=True)
    )
    for col in ["accuracy_std", "mean_abs_error_std", "rmse_std"]:
        grouped[col] = grouped[col].fillna(0.0)
    return grouped


def _prepare_baseline_reference(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = summary_df.loc[
        (summary_df["partition"] == "test") & (summary_df["architecture"] == BASELINE_ARCHITECTURE)
    ].copy()
    if baseline_df.empty:
        empty = pd.DataFrame()
        return empty, empty
    by_layer = (
        baseline_df.groupby(["model_alias", "model_label", "relative_layer"], as_index=False)
        .agg(
            mean_abs_error=("mean_abs_error", "mean"),
            train_mean_source_value=("control_value", "mean"),
            num_prompt_styles=("family_label", "nunique"),
        )
        .sort_values(["model_alias", "relative_layer"])
        .reset_index(drop=True)
    )
    by_model = (
        by_layer.groupby(["model_alias", "model_label"], as_index=False)
        .agg(
            mean_abs_error=("mean_abs_error", "mean"),
            train_mean_source_value=("train_mean_source_value", "mean"),
            num_prompt_styles=("num_prompt_styles", "max"),
        )
        .sort_values(["model_alias"])
        .reset_index(drop=True)
    )
    return by_layer, by_model


def _derive_curve_metrics(mean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    group_cols = ["model_alias", "model_label", "architecture", "relative_layer"]
    for keys, curve in mean_df.groupby(group_cols, sort=False):
        curve = curve.sort_values("step_index").reset_index(drop=True)
        baseline = float(curve.loc[0, "mean_abs_error_mean"])
        best_idx = curve["mean_abs_error_mean"].idxmin()
        best_row = curve.loc[best_idx]
        best_mae = float(best_row["mean_abs_error_mean"])
        total_gain = max(0.0, baseline - best_mae)
        mae_values = curve["mean_abs_error_mean"].to_numpy(dtype=float)
        normalized_steps = curve["normalized_step"].to_numpy(dtype=float)
        if total_gain > 0.0:
            progress = np.clip((baseline - mae_values) / total_gain, 0.0, 1.0)
            reach_mask = progress >= 0.9
            step_to_90 = float(normalized_steps[np.argmax(reach_mask)]) if bool(np.any(reach_mask)) else 1.0
            efficiency_90 = 1.0 - step_to_90
            saturation_half = float(np.max(progress[normalized_steps <= 0.5])) if bool(np.any(normalized_steps <= 0.5)) else 0.0
        else:
            progress = np.zeros_like(mae_values)
            step_to_90 = 1.0
            efficiency_90 = 0.0
            saturation_half = 0.0
        marginal_gain = np.concatenate([[0.0], mae_values[:-1] - mae_values[1:]])
        for row_idx, row in curve.iterrows():
            point_rows.append(
                {
                    **{col: row[col] for col in group_cols},
                    "step_index": int(row["step_index"]),
                    "max_step_index": int(row["max_step_index"]),
                    "normalized_step": float(row["normalized_step"]),
                    "layer_label": f"rel={row['relative_layer']:.1f}",
                    "control_name": row["control_name"],
                    "control_value_min": float(row["control_value_min"]),
                    "control_value_max": float(row["control_value_max"]),
                    "effective_df_min": float(row["effective_df_min"]),
                    "effective_df_max": float(row["effective_df_max"]),
                    "num_prompt_styles": int(row["num_prompt_styles"]),
                    "mean_abs_error_mean": float(row["mean_abs_error_mean"]),
                    "mean_abs_error_std": float(row["mean_abs_error_std"]),
                    "accuracy_mean": float(row["accuracy_mean"]),
                    "rmse_mean": float(row["rmse_mean"]),
                    "gain_over_step0": float(baseline - row["mean_abs_error_mean"]),
                    "progress_fraction": float(progress[row_idx]),
                    "marginal_gain": float(marginal_gain[row_idx]),
                }
            )
        curve_rows.append(
            {
                "model_alias": keys[0],
                "model_label": keys[1],
                "architecture": keys[2],
                "relative_layer": float(keys[3]),
                "baseline_mae": baseline,
                "best_mae": best_mae,
                "best_step_index": int(best_row["step_index"]),
                "best_normalized_step": float(best_row["normalized_step"]),
                "best_control_value_min": float(best_row["control_value_min"]),
                "best_control_value_max": float(best_row["control_value_max"]),
                "best_effective_df_min": float(best_row["effective_df_min"]),
                "best_effective_df_max": float(best_row["effective_df_max"]),
                "best_minus_step0_gap": total_gain,
                "step_to_90": step_to_90,
                "capacity_efficiency_90": efficiency_90,
                "saturation_half": saturation_half,
                "capacity_sensitivity_std": float(np.std(mae_values, ddof=0)),
                "capacity_sensitivity_range": float(np.max(mae_values) - np.min(mae_values)),
                "num_steps": int(len(curve)),
            }
        )
    return pd.DataFrame(point_rows), pd.DataFrame(curve_rows)


def _build_win_share(mean_df: pd.DataFrame) -> pd.DataFrame:
    win_rows: list[dict[str, Any]] = []
    for (model_alias, model_label, relative_layer), subset in mean_df.groupby(["model_alias", "model_label", "relative_layer"], sort=False):
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for architecture, arch_df in subset.groupby("architecture", sort=False):
            arch_df = arch_df.sort_values("normalized_step")
            xs = arch_df["normalized_step"].to_numpy(dtype=float)
            ys = arch_df["mean_abs_error_mean"].to_numpy(dtype=float)
            curves[str(architecture)] = (xs, ys)
        if set(curves) != set(ARCH_ORDER):
            continue
        for q in WIN_GRID:
            values = {
                architecture: float(np.interp(q, xs, ys))
                for architecture, (xs, ys) in curves.items()
            }
            best_value = min(values.values())
            winners = [architecture for architecture, value in values.items() if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9)]
            share = 1.0 / float(len(winners))
            for architecture in winners:
                win_rows.append(
                    {
                        "model_alias": model_alias,
                        "model_label": model_label,
                        "relative_layer": float(relative_layer),
                        "architecture": architecture,
                        "normalized_step_query": float(q),
                        "win_share": share,
                    }
                )
    win_df = pd.DataFrame(win_rows)
    if win_df.empty:
        return win_df
    return (
        win_df.groupby(["model_alias", "model_label", "relative_layer", "architecture"], as_index=False)["win_share"]
        .mean()
        .rename(columns={"win_share": "win_fraction_over_capacity"})
        .sort_values(["model_alias", "relative_layer", "architecture"])
        .reset_index(drop=True)
    )


def _build_frontier(mean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_points = (
        mean_df.loc[:, ["model_alias", "model_label", "architecture", "model_dim", "relative_layer", "step_index", "normalized_step", "effective_df_max", "log2_effective_df_mean", "mean_abs_error_mean"]]
        .rename(columns={"effective_df_max": "effective_df"})
        .copy()
    )
    grouped = (
        mean_df.groupby(["model_alias", "model_label", "architecture", "model_dim", "effective_df_max"], as_index=False)
        .apply(lambda frame: frame.loc[frame["mean_abs_error_mean"].idxmin(), ["relative_layer", "step_index", "normalized_step", "log2_effective_df_mean", "mean_abs_error_mean"]])
        .reset_index()
        .rename(columns={"effective_df_max": "effective_df"})
    )
    grouped = grouped.loc[:, ["model_alias", "model_label", "architecture", "model_dim", "effective_df", "relative_layer", "step_index", "normalized_step", "log2_effective_df_mean", "mean_abs_error_mean"]]
    frontier_rows: list[dict[str, Any]] = []
    for (model_alias, model_label, architecture), subset in grouped.groupby(["model_alias", "model_label", "architecture"], sort=False):
        subset = subset.sort_values("effective_df").reset_index(drop=True)
        running_best = math.inf
        frontier_layer = math.nan
        frontier_step = math.nan
        frontier_norm = math.nan
        model_dim = float(subset["model_dim"].iloc[0])
        for _, row in subset.iterrows():
            if float(row["mean_abs_error_mean"]) < running_best:
                running_best = float(row["mean_abs_error_mean"])
                frontier_layer = float(row["relative_layer"])
                frontier_step = float(row["step_index"])
                frontier_norm = float(row["normalized_step"])
            frontier_rows.append(
                {
                    "model_alias": model_alias,
                    "model_label": model_label,
                    "architecture": architecture,
                    "model_dim": model_dim,
                    "effective_df": float(row["effective_df"]),
                    "log2_effective_df_mean": float(row["log2_effective_df_mean"]),
                    "frontier_mae": running_best,
                    "frontier_source_layer": frontier_layer,
                    "frontier_source_step": frontier_step,
                    "frontier_source_normalized_step": frontier_norm,
                }
            )
    return all_points, pd.DataFrame(frontier_rows)


def _summary_table(curve_summary_df: pd.DataFrame) -> pd.DataFrame:
    best_gap = (
        curve_summary_df.sort_values(["model_alias", "architecture", "best_minus_step0_gap"], ascending=[True, True, False])
        .groupby(["model_alias", "architecture"], as_index=False)
        .head(1)
        .rename(columns={"relative_layer": "layer_best_gap", "best_minus_step0_gap": "best_gap"})
    )
    best_eff = (
        curve_summary_df.sort_values(["model_alias", "architecture", "capacity_efficiency_90"], ascending=[True, True, False])
        .groupby(["model_alias", "architecture"], as_index=False)
        .head(1)
        .rename(columns={"relative_layer": "layer_best_efficiency", "capacity_efficiency_90": "best_efficiency_90"})
    )
    stable = (
        curve_summary_df.sort_values(["model_alias", "architecture", "capacity_sensitivity_std"], ascending=[True, True, True])
        .groupby(["model_alias", "architecture"], as_index=False)
        .head(1)
        .rename(columns={"relative_layer": "layer_most_stable", "capacity_sensitivity_std": "lowest_sensitivity_std"})
    )
    merged = best_gap.merge(
        best_eff.loc[:, ["model_alias", "architecture", "layer_best_efficiency", "best_efficiency_90"]],
        on=["model_alias", "architecture"],
        how="left",
    ).merge(
        stable.loc[:, ["model_alias", "architecture", "layer_most_stable", "lowest_sensitivity_std"]],
        on=["model_alias", "architecture"],
        how="left",
    )
    model_label_map = curve_summary_df.loc[:, ["model_alias", "model_label"]].drop_duplicates().set_index("model_alias")["model_label"].to_dict()
    merged["model_label"] = merged["model_alias"].map(model_label_map)
    return merged.loc[:, ["model_label", "architecture", "layer_best_gap", "best_gap", "layer_best_efficiency", "best_efficiency_90", "layer_most_stable", "lowest_sensitivity_std"]]


def _table_html(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame.loc[:, columns].to_html(index=False, escape=False)


def _line_grid_by_model(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    hover_extra_cols: list[str],
    include_plotlyjs: str | bool = False,
) -> str:
    model_labels = _ordered_labels(frame, key_col="model_alias", label_col="model_label", preferred_keys=MODEL_ORDER)
    rows, cols = _subplot_dims(len(model_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=model_labels, shared_xaxes=True)
    label_to_model = (
        frame.loc[:, ["model_alias", "model_label"]]
        .drop_duplicates()
        .set_index("model_label")["model_alias"]
        .to_dict()
    )
    for idx, model_label in enumerate(model_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        model_alias = label_to_model[model_label]
        cell = frame.loc[frame["model_alias"] == model_alias].copy()
        for architecture in ARCH_ORDER:
            arch = cell.loc[cell["architecture"] == architecture].sort_values(x_col)
            if arch.empty:
                continue
            customdata = arch[hover_extra_cols].to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=arch[x_col],
                    y=arch[y_col],
                    mode="lines+markers",
                    name=architecture,
                    legendgroup=architecture,
                    showlegend=(idx == 0),
                    line=dict(color=ARCH_COLORS[architecture]),
                    hovertemplate=(
                        f"{model_label}<br>"
                        "arch=%{fullData.name}<br>"
                        "rel_layer=%{x:.1f}<br>"
                        f"{y_label}=%{{y:.4f}}<br>"
                        + "".join(f"{col_name}=%{{customdata[{j}]}}<br>" for j, col_name in enumerate(hover_extra_cols))
                        + "<extra></extra>"
                    ),
                    customdata=customdata,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(tickmode="array", tickvals=sorted(cell[x_col].drop_duplicates().tolist()), row=row, col=col)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=310 * rows,
        width=430 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.04, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _selected_layer_curve_figure(
    frame: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    y_label: str,
    include_plotlyjs: str | bool = False,
) -> str:
    selected = frame.loc[frame["relative_layer"].isin(SELECTED_LAYERS)].copy()
    selected["layer_label"] = selected["relative_layer"].map(lambda value: f"rel={value:.1f}")
    model_labels = _ordered_labels(selected, key_col="model_alias", label_col="model_label", preferred_keys=MODEL_ORDER)
    rows, cols = _subplot_dims(len(model_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=model_labels, shared_xaxes=True)
    label_to_model = (
        selected.loc[:, ["model_alias", "model_label"]]
        .drop_duplicates()
        .set_index("model_label")["model_alias"]
        .to_dict()
    )
    for idx, model_label in enumerate(model_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        model_alias = label_to_model[model_label]
        cell = selected.loc[selected["model_alias"] == model_alias].copy()
        for architecture in ARCH_ORDER:
            for relative_layer in SELECTED_LAYERS:
                line = cell.loc[(cell["architecture"] == architecture) & (cell["relative_layer"] == relative_layer)].sort_values("normalized_step")
                if line.empty:
                    continue
                name = f"{architecture} | rel={relative_layer:.1f}"
                fig.add_trace(
                    go.Scatter(
                        x=line["normalized_step"],
                        y=line[y_col],
                        mode="lines+markers",
                        name=name,
                        legendgroup=name,
                        showlegend=(idx == 0),
                        line=dict(color=ARCH_COLORS[architecture], dash=LAYER_DASHES[relative_layer]),
                        hovertemplate=(
                            f"{model_label}<br>"
                            f"arch={architecture}<br>"
                            f"rel_layer={relative_layer:.1f}<br>"
                            "norm_step=%{x:.2f}<br>"
                            f"{y_label}=%{{y:.4f}}<br>"
                            "step_index=%{customdata[0]}<br>"
                            "effective_df=%{customdata[1]}..%{customdata[2]}<br>"
                            "<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                line["step_index"].to_numpy(dtype=int),
                                line["effective_df_min"].to_numpy(dtype=float),
                                line["effective_df_max"].to_numpy(dtype=float),
                            ],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(tickmode="array", tickvals=[0.0, 0.25, 0.5, 0.75, 1.0], row=row, col=col)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=310 * rows,
        width=430 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _heatmap_grid(mean_df: pd.DataFrame, *, include_plotlyjs: str | bool = False) -> str:
    ordered_model_aliases = [
        model_alias
        for model_alias in MODEL_ORDER
        if model_alias in set(mean_df["model_alias"].dropna().unique().tolist())
    ]
    ordered_models = _ordered_labels(mean_df, key_col="model_alias", label_col="model_label", preferred_keys=ordered_model_aliases)
    subplot_titles = [f"{model_label} | {architecture}" for model_label in ordered_models for architecture in ARCH_ORDER]
    rows = len(ordered_models)
    cols = len(ARCH_ORDER)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, shared_xaxes=False, shared_yaxes=False)
    model_label_map = mean_df.loc[:, ["model_alias", "model_label"]].drop_duplicates().set_index("model_alias")["model_label"].to_dict()
    zmin = float(mean_df["mean_abs_error_mean"].min())
    zmax = float(mean_df["mean_abs_error_mean"].max())
    for row_idx, model_alias in enumerate(ordered_model_aliases, start=1):
        for col_idx, architecture in enumerate(ARCH_ORDER, start=1):
            cell = mean_df.loc[(mean_df["model_alias"] == model_alias) & (mean_df["architecture"] == architecture)].copy()
            if cell.empty:
                continue
            pivot = cell.pivot(index="normalized_step", columns="relative_layer", values="mean_abs_error_mean").sort_index().sort_index(axis=1)
            text = np.full(pivot.shape, "", dtype=object)
            for i, step in enumerate(pivot.index.tolist()):
                step_rows = cell.loc[cell["normalized_step"] == step].sort_values("relative_layer")
                for j, (_, point) in enumerate(step_rows.iterrows()):
                    text[i, j] = (
                        f"model={model_label_map[model_alias]}<br>"
                        f"arch={architecture}<br>"
                        f"rel_layer={point['relative_layer']:.1f}<br>"
                        f"norm_step={point['normalized_step']:.2f}<br>"
                        f"step_index={int(point['step_index'])}<br>"
                        f"MAE={point['mean_abs_error_mean']:.4f}<br>"
                        f"effective_df={point['effective_df_min']:.0f}..{point['effective_df_max']:.0f}"
                    )
            fig.add_trace(
                go.Heatmap(
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    z=pivot.to_numpy(dtype=float),
                    zmin=zmin,
                    zmax=zmax,
                    colorscale="Viridis_r",
                    colorbar=dict(title="Mean MAE") if (row_idx == 1 and col_idx == cols) else None,
                    showscale=(row_idx == 1 and col_idx == cols),
                    text=text,
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.update_xaxes(tickmode="array", tickvals=sorted(pivot.columns.tolist()), row=row_idx, col=col_idx)
            fig.update_yaxes(tickmode="array", tickvals=sorted(pivot.index.tolist()), row=row_idx, col=col_idx)
    fig.update_layout(
        title="Mean Test MAE Heatmaps | Relative Layer x Normalized Capacity Step",
        template="plotly_white",
        height=260 * rows,
        width=420 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _win_fraction_figure(win_df: pd.DataFrame, *, include_plotlyjs: str | bool = False) -> str:
    return _line_grid_by_model(
        win_df,
        x_col="relative_layer",
        y_col="win_fraction_over_capacity",
        title="Architecture Win Fraction Across Normalized Capacity Grid",
        y_label="Win Fraction",
        hover_extra_cols=[],
        include_plotlyjs=include_plotlyjs,
    )


def _pareto_frontier_figure(
    all_points: pd.DataFrame,
    frontier_df: pd.DataFrame,
    baseline_by_model: pd.DataFrame,
    *,
    include_plotlyjs: str | bool = False,
) -> str:
    model_labels = _ordered_labels(all_points, key_col="model_alias", label_col="model_label", preferred_keys=MODEL_ORDER)
    rows, cols = _subplot_dims(len(model_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=model_labels, shared_xaxes=False, shared_yaxes=False)
    label_to_model = (
        all_points.loc[:, ["model_alias", "model_label"]]
        .drop_duplicates()
        .set_index("model_label")["model_alias"]
        .to_dict()
    )
    for idx, model_label in enumerate(model_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        model_alias = label_to_model[model_label]
        points = all_points.loc[all_points["model_alias"] == model_alias].copy()
        frontier = frontier_df.loc[frontier_df["model_alias"] == model_alias].copy()
        model_dim = float(points["model_dim"].iloc[0])
        tickvals, ticktext = _effective_df_ticks(model_dim)
        baseline_row = baseline_by_model.loc[baseline_by_model["model_alias"] == model_alias]
        if not baseline_row.empty:
            baseline_mae = float(baseline_row["mean_abs_error"].iloc[0])
            train_mean_value = float(baseline_row["train_mean_source_value"].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=[1.0, model_dim],
                    y=[max(baseline_mae, 1e-6), max(baseline_mae, 1e-6)],
                    mode="lines",
                    name="train-mean baseline",
                    legendgroup="train-mean baseline",
                    showlegend=(idx == 0),
                    line=dict(color=BASELINE_LINE_COLOR, dash="dash", width=2),
                    hovertemplate=(
                        f"{model_label}<br>"
                        "reference=train-mean baseline<br>"
                        "effective_df=%{x:.1f}<br>"
                        "MAE=%{y:.4f}<br>"
                        f"mean train source value={train_mean_value:.2f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        for architecture in ARCH_ORDER:
            arch_points = points.loc[points["architecture"] == architecture]
            arch_frontier = frontier.loc[frontier["architecture"] == architecture].sort_values("effective_df")
            if not arch_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=arch_points["effective_df"],
                        y=np.clip(arch_points["mean_abs_error_mean"].to_numpy(dtype=float), a_min=1e-6, a_max=None),
                        mode="markers",
                        name=f"{architecture} points",
                        legendgroup=architecture,
                        showlegend=False,
                        marker=dict(color=ARCH_COLORS[architecture], opacity=0.18, size=6),
                        hovertemplate=(
                            f"{model_label}<br>"
                            f"arch={architecture}<br>"
                            "rel_layer=%{customdata[0]:.1f}<br>"
                            "step_index=%{customdata[1]}<br>"
                            "effective_df=%{x:.1f}<br>"
                            "MAE=%{y:.4f}<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                arch_points["relative_layer"].to_numpy(dtype=float),
                                arch_points["step_index"].to_numpy(dtype=int),
                            ],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
            if not arch_frontier.empty:
                fig.add_trace(
                    go.Scatter(
                        x=arch_frontier["effective_df"],
                        y=np.clip(arch_frontier["frontier_mae"].to_numpy(dtype=float), a_min=1e-6, a_max=None),
                        mode="lines+markers",
                        name=architecture,
                        legendgroup=architecture,
                        showlegend=(idx == 0),
                        line=dict(color=ARCH_COLORS[architecture]),
                        hovertemplate=(
                            f"{model_label}<br>"
                            "arch=%{fullData.name}<br>"
                            "effective_df=%{x:.1f}<br>"
                            "frontier MAE=%{y:.4f}<br>"
                            "source rel_layer=%{customdata[0]:.1f}<br>"
                            "source norm_step=%{customdata[1]:.2f}<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                arch_frontier["frontier_source_layer"].to_numpy(dtype=float),
                                arch_frontier["frontier_source_normalized_step"].to_numpy(dtype=float),
                            ],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_yaxes(type="log", row=row, col=col)
        fig.update_xaxes(
            type="log",
            range=[0.0, math.log10(model_dim)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title_text="Effective DF",
            row=row,
            col=col,
        )
    fig.update_layout(
        title="Pareto-Style Frontier | Best Achievable MAE at or Below Each Effective-DF (Log MAE)",
        template="plotly_white",
        height=310 * rows,
        width=430 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.04, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _layer_connected_capacity_paths_figure(
    all_points: pd.DataFrame,
    baseline_by_model: pd.DataFrame,
    *,
    include_plotlyjs: str | bool = False,
) -> str:
    model_labels = _ordered_labels(all_points, key_col="model_alias", label_col="model_label", preferred_keys=MODEL_ORDER)
    rows, cols = _subplot_dims(len(model_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=model_labels, shared_xaxes=False, shared_yaxes=False)
    label_to_model = (
        all_points.loc[:, ["model_alias", "model_label"]]
        .drop_duplicates()
        .set_index("model_label")["model_alias"]
        .to_dict()
    )
    for idx, model_label in enumerate(model_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        model_alias = label_to_model[model_label]
        points = all_points.loc[all_points["model_alias"] == model_alias].copy()
        model_dim = float(points["model_dim"].iloc[0])
        tickvals, ticktext = _effective_df_ticks(model_dim)
        baseline_row = baseline_by_model.loc[baseline_by_model["model_alias"] == model_alias]
        if not baseline_row.empty:
            baseline_mae = float(baseline_row["mean_abs_error"].iloc[0])
            train_mean_value = float(baseline_row["train_mean_source_value"].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=[1.0, model_dim],
                    y=[max(baseline_mae, 1e-6), max(baseline_mae, 1e-6)],
                    mode="lines",
                    name="train-mean baseline",
                    legendgroup="train-mean baseline",
                    showlegend=(idx == 0),
                    line=dict(color=BASELINE_LINE_COLOR, dash="dash", width=2),
                    hovertemplate=(
                        f"{model_label}<br>"
                        "reference=train-mean baseline<br>"
                        "effective_df=%{x:.1f}<br>"
                        "MAE=%{y:.4f}<br>"
                        f"mean train source value={train_mean_value:.2f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        for architecture in ARCH_ORDER:
            arch_points = points.loc[points["architecture"] == architecture].copy()
            if arch_points.empty:
                continue
            for layer_rank, (relative_layer, layer_points) in enumerate(sorted(arch_points.groupby("relative_layer"), key=lambda item: item[0])):
                layer_points = layer_points.sort_values("effective_df")
                shade_amount = 0.60 * (1.0 - float(relative_layer))
                color = _blend_with_white(ARCH_COLORS[architecture], shade_amount)
                fig.add_trace(
                    go.Scatter(
                        x=layer_points["effective_df"],
                        y=np.clip(layer_points["mean_abs_error_mean"].to_numpy(dtype=float), a_min=1e-6, a_max=None),
                        mode="lines+markers",
                        name=architecture,
                        legendgroup=architecture,
                        showlegend=(idx == 0 and layer_rank == 0),
                        line=dict(color=color, width=1.8),
                        marker=dict(color=color, size=6),
                        hovertemplate=(
                            f"{model_label}<br>"
                            f"arch={architecture}<br>"
                            "rel_layer=%{customdata[0]:.1f}<br>"
                            "step_index=%{customdata[1]}<br>"
                            "effective_df=%{x:.1f}<br>"
                            "MAE=%{y:.4f}<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                layer_points["relative_layer"].to_numpy(dtype=float),
                                layer_points["step_index"].to_numpy(dtype=int),
                            ],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_yaxes(type="log", row=row, col=col)
        fig.update_xaxes(
            type="log",
            range=[0.0, math.log10(model_dim)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title_text="Effective DF",
            row=row,
            col=col,
        )
    fig.update_layout(
        title="Layer-Connected Capacity Paths | Same-Layer Markers Connected Across Effective-DF (Log MAE)",
        template="plotly_white",
        height=310 * rows,
        width=430 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.04, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _write_report(
    *,
    report_path: Path,
    artifact_dir: Path,
    mean_df: pd.DataFrame,
    baseline_by_layer_df: pd.DataFrame,
    baseline_by_model_df: pd.DataFrame,
    curve_points_df: pd.DataFrame,
    curve_summary_df: pd.DataFrame,
    win_df: pd.DataFrame,
    all_points_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    report_artifact_dir = ensure_dir(artifact_dir)
    atomic_write_jsonl(report_artifact_dir / "mean_curves_over_prompt_styles.jsonl", mean_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "baseline_reference_by_layer.jsonl", baseline_by_layer_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "baseline_reference_by_model.jsonl", baseline_by_model_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "curve_points_with_progress.jsonl", curve_points_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "curve_summary_by_layer.jsonl", curve_summary_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "win_fraction_by_layer.jsonl", win_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "pareto_all_points.jsonl", all_points_df.to_dict("records"))
    atomic_write_jsonl(report_artifact_dir / "pareto_frontier.jsonl", frontier_df.to_dict("records"))
    atomic_write_json(report_artifact_dir / "metadata.json", metadata)

    summary_table = _summary_table(curve_summary_df)

    plotlyjs_mode: str | bool = "cdn"
    heatmaps = _heatmap_grid(mean_df, include_plotlyjs=plotlyjs_mode)
    plotlyjs_mode = False
    gain_curves = _selected_layer_curve_figure(
        curve_points_df,
        y_col="gain_over_step0",
        title="Gain over Step 0 | Representative Layer Slices",
        y_label="MAE(step_0) - MAE(step_k)",
        include_plotlyjs=plotlyjs_mode,
    )
    progress_curves = _selected_layer_curve_figure(
        curve_points_df,
        y_col="progress_fraction",
        title="Fraction of Total Gain Achieved | Representative Layer Slices",
        y_label="Progress Fraction",
        include_plotlyjs=plotlyjs_mode,
    )
    marginal_curves = _selected_layer_curve_figure(
        curve_points_df.loc[curve_points_df["step_index"] > 0].copy(),
        y_col="marginal_gain",
        title="Marginal Gain per Additional Capacity Step | Representative Layer Slices",
        y_label="MAE(step_{k-1}) - MAE(step_k)",
        include_plotlyjs=plotlyjs_mode,
    )
    gap_by_layer = _line_grid_by_model(
        curve_summary_df,
        x_col="relative_layer",
        y_col="best_minus_step0_gap",
        title="Best-minus-Step0 Gap by Relative Layer",
        y_label="Best-minus-Step0 Gap",
        hover_extra_cols=["best_step_index", "best_normalized_step"],
        include_plotlyjs=plotlyjs_mode,
    )
    efficiency_by_layer = _line_grid_by_model(
        curve_summary_df,
        x_col="relative_layer",
        y_col="capacity_efficiency_90",
        title="Capacity Efficiency to 90% of Best Gain by Relative Layer",
        y_label="Capacity Efficiency 90",
        hover_extra_cols=["step_to_90", "best_step_index"],
        include_plotlyjs=plotlyjs_mode,
    )
    saturation_by_layer = _line_grid_by_model(
        curve_summary_df,
        x_col="relative_layer",
        y_col="saturation_half",
        title="Saturation Index by Relative Layer",
        y_label="Saturation Half",
        hover_extra_cols=["best_minus_step0_gap", "best_step_index"],
        include_plotlyjs=plotlyjs_mode,
    )
    sensitivity_by_layer = _line_grid_by_model(
        curve_summary_df,
        x_col="relative_layer",
        y_col="capacity_sensitivity_std",
        title="Capacity Sensitivity by Relative Layer",
        y_label="MAE Std Across Steps",
        hover_extra_cols=["capacity_sensitivity_range", "best_minus_step0_gap"],
        include_plotlyjs=plotlyjs_mode,
    )
    win_fraction_plot = _win_fraction_figure(win_df, include_plotlyjs=plotlyjs_mode)
    pareto_plot = _pareto_frontier_figure(all_points_df, frontier_df, baseline_by_model_df, include_plotlyjs=plotlyjs_mode)
    pareto_layer_paths_plot = _layer_connected_capacity_paths_figure(all_points_df, baseline_by_model_df, include_plotlyjs=plotlyjs_mode)

    metric_glossary_html = """
<ul>
  <li><strong>Mean test MAE</strong>: prompt-style-averaged held-out MAE at one model, architecture, layer, and capacity step. Formula: <code>mean_family(MAE_test)</code>.</li>
  <li><strong>Gain over step 0</strong>: improvement relative to the lowest-capacity setting. Formula: <code>MAE(step_0) - MAE(step_k)</code>.</li>
  <li><strong>Progress fraction</strong>: fraction of the total attainable improvement already realized by step <code>k</code>. Formula: <code>(MAE(step_0) - MAE(step_k)) / (MAE(step_0) - MAE(best_step))</code>.</li>
  <li><strong>Best-minus-step0 gap</strong>: total improvement unlocked somewhere along the capacity sweep. Formula: <code>MAE(step_0) - min_k MAE(step_k)</code>.</li>
  <li><strong>Capacity efficiency 90</strong>: transformed version of the step needed to reach 90% of the total gain. Formula: <code>1 - min(normalized_step where progress >= 0.9)</code>.</li>
  <li><strong>Saturation index</strong>: fraction of the total gain already obtained by the first half of the ladder. Formula: <code>max(progress_k for normalized_step <= 0.5)</code>.</li>
  <li><strong>Capacity sensitivity</strong>: how much the layer depends on exact capacity choice. Formula: <code>std_k(MAE(step_k))</code>.</li>
  <li><strong>Win fraction over capacity</strong>: fraction of normalized capacity queries where an architecture has the lowest interpolated MAE for a fixed model and layer.</li>
  <li><strong>Pareto frontier</strong>: best achievable MAE at or below each effective-DF after choosing the strongest layer available at that cost.</li>
</ul>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{metadata['report_title']}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #111; }}
    h1, h2, h3 {{ margin-bottom: 0.35em; }}
    p, li {{ line-height: 1.45; }}
    code {{ background: #f4f4f4; padding: 0.1em 0.25em; border-radius: 4px; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px 0; min-width: 960px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    .stacked-figs {{ display: flex; flex-direction: column; gap: 20px; }}
    .wide-fig {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{metadata['report_title']}</h1>

  <h2>Research Question</h2>
  <p>
    Which visualization best exposes the impact of probe capacity step on held-out source-value decoding quality once prompt styles are averaged, and which of those variants would be worth promoting into the main relative-layer report?
  </p>

  <h2>Scope of Evidence</h2>
  <p>
    This development-only report is derived from the existing relative-layer sweep artifact <code>{metadata['source_artifact_dir']}</code>.
    It uses the held-out <code>test</code> rows from the completed <code>5 models x 5 prompt styles x 11 relative layers</code> sweep, then aggregates over prompt styles to study how performance changes over the capacity ladder inside each architecture.
  </p>

  <h2>Executive Summary</h2>
  <p>
    The report collects ten capacity-sensitive plot variants in one place: raw MAE heatmaps, gain-over-step-0 curves, progress curves, marginal-gain curves, best-minus-step-0 gap summaries, capacity-efficiency summaries, saturation summaries, sensitivity summaries, architecture win-fraction summaries, and Pareto-style frontiers.
    The goal is not to argue for one view yet, but to let you compare how much each variant preserves the layer story versus how much it compresses the sweep into a single scalar.
  </p>

  <h2>Main Findings</h2>
  <ul>
    <li>The raw heatmaps are the most complete view of the <code>layer x capacity</code> surface, but they are also the densest and hardest to summarize in the main report.</li>
    <li>The most interpretable higher-is-better summaries are <code>best-minus-step0 gap</code>, <code>capacity efficiency 90</code>, and <code>saturation index</code>; they preserve the main capacity story without forcing the reader to mentally integrate a whole curve.</li>
    <li>The representative-layer curve slices are useful development diagnostics because they expose whether a layer improves steadily, saturates early, or gains only at the extreme high-capacity end.</li>
    <li>The win-fraction and Pareto sections are useful tie-breakers: win-fraction shows architecture dominance across the whole ladder, while the frontier view shows the best attainable MAE at a given effective-DF after layer selection.</li>
  </ul>

  <h2>Key Tables and Figures</h2>
  <p>Metric glossary:</p>
  {metric_glossary_html}

  <h3>Layer-Summary Table</h3>
  {_table_html(summary_table, ['model_label', 'architecture', 'layer_best_gap', 'best_gap', 'layer_best_efficiency', 'best_efficiency_90', 'layer_most_stable', 'lowest_sensitivity_std'])}

  <h3>Raw MAE Heatmaps</h3>
  <p>
    One panel per <code>model x architecture</code>. Color shows prompt-style-averaged held-out MAE at each relative layer and normalized capacity step.
  </p>
  {heatmaps}

  <h3>Gain over Step 0 Curves</h3>
  <p>
    Representative layer slices at <code>rel=0.0, 0.1, 0.5, 0.9</code>. Higher is better.
  </p>
  {gain_curves}

  <h3>Fraction of Total Gain Achieved</h3>
  <p>
    Same representative layer slices, but now normalized so <code>1.0</code> means the best step on that layer has already been reached.
  </p>
  {progress_curves}

  <h3>Marginal Gain per Additional Capacity Step</h3>
  <p>
    Positive values mean the next capacity increase reduced MAE; negative values mean the step made MAE worse.
  </p>
  {marginal_curves}

  <h3>Best-minus-Step0 Gap by Relative Layer</h3>
  {gap_by_layer}

  <h3>Capacity Efficiency to 90% of Best Gain</h3>
  {efficiency_by_layer}

  <h3>Saturation Index by Relative Layer</h3>
  {saturation_by_layer}

  <h3>Capacity Sensitivity by Relative Layer</h3>
  <p>
    Lower values mean a layer is more robust to the exact capacity step; higher values mean the layer is more sensitive to that choice.
  </p>
  {sensitivity_by_layer}

  <h3>Architecture Win Fraction Across the Capacity Ladder</h3>
  {win_fraction_plot}

  <h3>Pareto-Style Best-Layer Frontier</h3>
  <p>
    Each frontier line shows the best MAE reachable at or below each effective-DF after selecting the strongest layer available at that cost.
    The y-axis is logarithmic so early low-capacity errors do not visually flatten the higher-capacity regime, and the x-axis now shows actual effective-DF on a shared log scale from 1 up to the model dimension.
    The gray dashed line is the train-mean baseline.
  </p>
  {pareto_plot}

  <h3>Layer-Connected Capacity Paths</h3>
  <p>
    This companion view connects markers that come from the same relative layer across capacity steps.
    Within each architecture, shallow layers are lighter and deeper layers are darker.
    Both axes use the same effective-DF framing as the Pareto plot: actual effective-DF on a shared log x-axis and MAE on a log y-axis.
    The gray dashed line is the train-mean baseline.
  </p>
  {pareto_layer_paths_plot}

  <h2>Interpretation</h2>
  <p>
    The raw heatmaps are still the truth-preserving view. They show exactly where performance improves, stalls, or degrades as capacity grows.
    But if the main report needs a compact scalar summary, the most reader-friendly candidates are the plots that remain higher-is-better: the total gap unlocked by capacity, the efficiency with which the curve reaches most of that gain, and the saturation index that asks whether the gain arrives early.
    The win-fraction plot is stronger as an architecture-comparison diagnostic than as a standalone explanatory figure, while the Pareto frontier is strongest when the question is explicitly about performance per unit effective-DF rather than about layers.
  </p>

  <h2>Limitations and Negative Results</h2>
  <p>
    All derived views here average over prompt styles, so prompt-family disagreements are intentionally compressed away.
    They also use only the existing <code>split_00</code> sweep, so none of the scalar summaries yet separate stable structure from split-specific noise.
    Some summaries remain mildly redundant: for example, <code>best-minus-step0 gap</code> and <code>saturation index</code> answer related questions when the improvement curve is nearly monotonic.
  </p>

  <h2>Implications</h2>
  <p>
    This report should make it easier to choose one or two capacity-sensitive visuals for the main report.
    If the priority is interpretability, start from <code>best-minus-step0 gap</code> or <code>capacity efficiency 90</code>.
    If the priority is preserving the whole sweep, keep one heatmap or representative-layer curve slice in the development appendix and use the scalar summary only as the headline figure.
  </p>

  <h2>Recommended Next Experiments</h2>
  <ul>
    <li>Select one compact higher-is-better summary and one raw-surface companion plot for promotion into the main report.</li>
    <li>Repeat the same derived summaries over more CV splits to see which variants remain stable enough to deserve main-report status.</li>
    <li>If prompt-family disagreement becomes central, rebuild the same diagnostics without averaging over prompt styles so the report can show where that averaging hides real differences.</li>
  </ul>
</body>
</html>
"""
    atomic_write_text(report_path, html)


def main() -> None:
    args = parse_args()
    source_artifact_dir = Path(args.artifact_dir)
    report_path = Path(args.report_path)
    report_stem = str(args.report_stem)
    report_artifact_dir = ROOT / "experiment-reports" / "artifacts" / report_stem

    summary_df, source_metadata = _load_inputs(source_artifact_dir)
    mean_df = _prepare_mean_curves(summary_df)
    baseline_by_layer_df, baseline_by_model_df = _prepare_baseline_reference(summary_df)
    curve_points_df, curve_summary_df = _derive_curve_metrics(mean_df)
    win_df = _build_win_share(mean_df)
    all_points_df, frontier_df = _build_frontier(mean_df)

    metadata = {
        "report_title": "Source-Value Probe Capacity Sweep | split_00 | Relative Layers | Capacity Diagnostics (Dev)",
        "source_artifact_dir": str(source_artifact_dir),
        "source_report_title": source_metadata.get("report_title", ""),
        "selected_layers": SELECTED_LAYERS,
        "win_grid": WIN_GRID.tolist(),
    }
    _write_report(
        report_path=report_path,
        artifact_dir=report_artifact_dir,
        mean_df=mean_df,
        baseline_by_layer_df=baseline_by_layer_df,
        baseline_by_model_df=baseline_by_model_df,
        curve_points_df=curve_points_df,
        curve_summary_df=curve_summary_df,
        win_df=win_df,
        all_points_df=all_points_df,
        frontier_df=frontier_df,
        metadata=metadata,
    )
    print(report_path)


if __name__ == "__main__":
    main()
