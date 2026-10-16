#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import socket
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_value_probe_capacity_sweep import (
    _augment_summary,
    _build_ridge_schedule,
    _collect_partition_activations,
    _load_partition_rows,
    _power_of_two_ladder,
    default_prompt_sequence_cache_root,
)
from src.magnitude_intervention import evaluate_probe_split, fit_magnitude_probe
from src.model_loader import load_causal_lm, load_tokenizer, resolve_model_revision
from src.probe_architectures import canonicalize_probe_architecture_name, probe_architecture_display_name
from src.utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    dump_run_config,
    ensure_dir,
    load_jsonl,
    load_yaml,
    set_global_seed,
)


DEFAULT_CONFIG = ROOT / "configs" / "source_value_probe_capacity_sweep_split00_relative_layers.yaml"
FAMILY_ORDER = [
    "Addition Formula",
    "Addition Verbalized",
    "Subtraction Formula",
    "Subtraction Verbalized",
    "Identity Patchscopes",
]
PROBE_ARCH_ORDER = ["circular_reg", "circular_cls", "pls", "linear"]
BASELINE_ARCHITECTURE = "mean_baseline"
BASELINE_LINE_COLOR = "#7f7f7f"
ARCH_ORDER = list(PROBE_ARCH_ORDER)
ARCH_COLORS = {"circular_reg": "#d62728", "circular_cls": "#b54d4d", "pls": "#2ca02c", "linear": "#1f77b4"}


def _checkpoint_progress(combo_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    """Save completed fits and their RNG continuation as one atomic checkpoint."""
    path = combo_dir / "progress_checkpoint.pt"
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(
        {
            "summary_rows": summary_rows,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    os.replace(temporary, path)


def _restore_progress(combo_dir: Path) -> list[dict[str, Any]] | None:
    path = combo_dir / "progress_checkpoint.pt"
    if not path.exists():
        return None
    # This is a locally generated checkpoint in this configuration's run directory.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return list(checkpoint["summary_rows"])


def _write_progress(output_dir: Path, **fields: Any) -> None:
    atomic_write_json(
        output_dir / "progress.json",
        {"updated_at": datetime.now(timezone.utc).isoformat(), "server": socket.gethostname(),
         "pid": os.getpid(), **fields},
    )


MODEL_ORDER = ["llama_3_2_1b", "llama_3_2_3b", "llama_3_8b", "olmo_3_7b_base", "olmo_2_13b", "gpt_oss_20b"]
METRIC_SPECS = {
    "accuracy": {
        "label": "Exact Accuracy",
        "description": "Fraction of examples where the rounded probe prediction exactly matches the target source value.",
        "formula": "mean(round(prediction) == target)",
        "higher_is_better": True,
    },
    "mean_abs_error": {
        "label": "Mean Absolute Error",
        "description": "Average absolute distance between the rounded probe prediction and the target source value.",
        "formula": "mean(abs(round(prediction) - target))",
        "higher_is_better": False,
    },
    "rmse": {
        "label": "Root Mean Squared Error",
        "description": "Square root of the average squared error between the rounded probe prediction and the target source value.",
        "formula": "sqrt(mean((round(prediction) - target)^2))",
        "higher_is_better": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the source-value probe capacity sweep across relative layers.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="out/runs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backfill-mean-baseline-run-dir")
    parser.add_argument("--split-name")
    parser.add_argument("--report-stem")
    return parser.parse_args()


def _decimal_round(number: float, n_decimal_places: int = 0) -> Decimal:
    round_fmt = "1"
    if n_decimal_places > 0:
        round_fmt += "." + "0" * n_decimal_places
    return Decimal(str(number)).quantize(Decimal(round_fmt), rounding=ROUND_HALF_UP)


def _resolve_n_layers(*, model_name: str, trust_remote_code: bool) -> int:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        if hasattr(config, key):
            return int(getattr(config, key))
    raise ValueError(f"Could not resolve number of layers for {model_name}.")


def _layer_rel_to_abs(layer_rel: float, n_layers: int) -> int:
    resolved = int(_decimal_round(float(layer_rel) * int(n_layers)))
    return max(0, min(int(n_layers) - 1, resolved))


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _dense_prefix_ladder(limit: int, dense_prefix_max: int | None) -> list[int]:
    if dense_prefix_max is None:
        return _power_of_two_ladder(limit)
    capped_prefix = min(int(limit), int(dense_prefix_max))
    if capped_prefix < 1:
        return _power_of_two_ladder(limit)
    merged = set(range(1, capped_prefix + 1))
    merged.update(_power_of_two_ladder(limit))
    return sorted(int(value) for value in merged)


def _summary_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    control_value = row.get("control_value")
    if isinstance(control_value, float):
        control_repr = f"{control_value:.12g}"
    else:
        control_repr = str(control_value)
    return (
        str(row["partition"]),
        str(row["architecture"]),
        str(row["control_name"]),
        control_repr,
    )


def _load_seed_rows(
    *,
    current_combo_dir: Path,
    seed_output_dir: Path | None,
    model_alias: str,
    relative_layer: float,
    family: str,
    split_name: str,
    circular_step_index_by_value: dict[int, int],
) -> list[dict[str, Any]]:
    candidate_paths: list[Path] = []
    current_path = current_combo_dir / "probe_summary.jsonl"
    if current_path.exists():
        candidate_paths.append(current_path)
    if seed_output_dir is not None:
        seed_path = seed_output_dir / model_alias / f"layer_rel_{relative_layer:.1f}".replace(".", "_") / family / "probe_summary.jsonl"
        if seed_path.exists():
            candidate_paths.append(seed_path)

    merged_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in candidate_paths:
        for row in load_jsonl(path):
            normalized = dict(row)
            if str(normalized.get("split_name")) != str(split_name):
                continue
            normalized["architecture"] = canonicalize_probe_architecture_name(str(normalized.get("architecture")))
            if normalized.get("architecture") == "circular_reg":
                control_value = int(normalized["control_value"])
                if control_value not in circular_step_index_by_value:
                    continue
                normalized["step_index"] = int(circular_step_index_by_value[control_value])
                normalized["effective_df"] = float(control_value)
            key = _summary_key(normalized)
            if key not in merged_rows:
                merged_rows[key] = normalized
    return list(merged_rows.values())


def _constant_mean_baseline_summary(
    *,
    values: np.ndarray,
    split_name: str,
    train_mean_value: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = np.asarray(values, dtype=np.int64)
    raw_prediction = np.full(len(truth), float(train_mean_value), dtype=np.float64)
    prediction = np.rint(raw_prediction).astype(np.int64)
    prediction = np.clip(prediction, 0, 999)
    abs_error = np.abs(prediction - truth)
    detail_df = pd.DataFrame(
        {
            "split": split_name,
            "architecture": BASELINE_ARCHITECTURE,
            "value": truth,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "correct": (prediction == truth).astype(int),
            "abs_error": abs_error.astype(np.int64),
            "prediction_feature_norm": np.zeros(len(truth), dtype=np.float64),
        }
    )
    summary = {
        "split": split_name,
        "architecture": BASELINE_ARCHITECTURE,
        "num_examples": int(len(detail_df)),
        "accuracy": float(detail_df["correct"].mean()),
        "mean_abs_error": float(detail_df["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean((prediction.astype(np.float64) - truth.astype(np.float64)) ** 2))),
    }
    return detail_df, summary


def _baseline_metadata_row(
    *,
    partition: str,
    train_mean_value: float,
    model_alias: str,
    model_label: str,
    family: str,
    family_label: str,
    split_name: str,
    model_dim: int,
    relative_layer: float,
    layer_idx: int,
    n_layers: int,
    values: np.ndarray,
) -> dict[str, Any]:
    detail_df, base_summary = _constant_mean_baseline_summary(
        values=values,
        split_name=partition,
        train_mean_value=train_mean_value,
    )
    summary_row = _augment_summary(detail_df, {**base_summary, "partition": partition})
    summary_row.update(
        {
            "architecture": BASELINE_ARCHITECTURE,
            "step_index": -1,
            "control_name": "train_mean_source_value",
            "control_value": float(train_mean_value),
            "effective_df": 0.0,
            "model_alias": model_alias,
            "model_label": model_label,
            "family": family,
            "family_label": family_label,
            "split_name": split_name,
            "model_dim": int(model_dim),
            "relative_layer": float(relative_layer),
            "layer_idx": int(layer_idx),
            "n_layers": int(n_layers),
        }
    )
    return summary_row


def _seed_model_from_existing_run(
    *,
    output_dir: Path,
    seed_output_dir: Path | None,
    model_alias: str,
    split_name: str,
    relative_layers: list[float],
    family_specs: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if seed_output_dir is None:
        return None
    seed_model_dir = seed_output_dir / model_alias
    if not seed_model_dir.exists():
        return None

    seeded_rows: list[dict[str, Any]] = []
    for relative_layer in relative_layers:
        layer_rel_dir = f"layer_rel_{relative_layer:.1f}".replace(".", "_")
        for family_spec in family_specs:
            family = str(family_spec["family"])
            seed_combo_dir = seed_model_dir / layer_rel_dir / family
            seed_summary_path = seed_combo_dir / "probe_summary.jsonl"
            seed_resolved_path = seed_combo_dir / "resolved.json"
            seed_done_path = seed_combo_dir / "DONE"
            if not seed_summary_path.exists():
                return None

            combo_dir = ensure_dir(output_dir / model_alias / layer_rel_dir / family)
            combo_rows = [
                {**dict(row), "architecture": canonicalize_probe_architecture_name(str(row.get("architecture")))}
                for row in load_jsonl(seed_summary_path)
                if str(row.get("split_name")) == str(split_name)
            ]
            if not combo_rows:
                return None
            expected_architectures = {BASELINE_ARCHITECTURE, *PROBE_ARCH_ORDER}
            observed_architectures = {str(row["architecture"]) for row in combo_rows}
            if not expected_architectures.issubset(observed_architectures):
                return None
            seeded_rows.extend(combo_rows)
            atomic_write_jsonl(combo_dir / "probe_summary.jsonl", combo_rows)
            if seed_resolved_path.exists():
                resolved_payload = json.loads(seed_resolved_path.read_text(encoding="utf-8"))
                if str(resolved_payload.get("split_name")) != str(split_name):
                    return None
                atomic_write_json(combo_dir / "resolved.json", resolved_payload)
            if seed_done_path.exists():
                atomic_write_text(combo_dir / "DONE", seed_done_path.read_text(encoding="utf-8"))
    return seeded_rows


def _competitive_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df.loc[summary_df["architecture"].isin(PROBE_ARCH_ORDER)].copy()


def _baseline_summary_df(summary_df: pd.DataFrame, *, partition: str = "test") -> pd.DataFrame:
    return summary_df.loc[
        (summary_df["architecture"] == BASELINE_ARCHITECTURE) & (summary_df["partition"] == partition)
    ].copy()


def _load_saved_experiment_config(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    experiment_config = payload.get("experiment_config")
    if experiment_config is None and isinstance(payload.get("config"), dict):
        experiment_config = payload["config"].get("experiment_config")
    if experiment_config is None:
        raise ValueError(f"{run_dir / 'config.json'} does not contain an experiment_config payload.")
    return dict(experiment_config)


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


def _metric_sort_spec(metric: str) -> tuple[list[str], list[bool]]:
    if metric == "accuracy":
        return ["accuracy", "r2", "mean_abs_error", "rmse"], [False, False, True, True]
    if metric == "mean_abs_error":
        return ["mean_abs_error", "rmse", "accuracy", "r2"], [True, True, False, False]
    if metric == "rmse":
        return ["rmse", "mean_abs_error", "accuracy", "r2"], [True, True, False, False]
    raise ValueError(f"Unsupported metric {metric!r}.")


def _best_rows(summary_df: pd.DataFrame, *, partition: str, metric: str, include_architecture: bool) -> pd.DataFrame:
    metric_cols, metric_ascending = _metric_sort_spec(metric)
    group_cols = ["model_alias", "family", "relative_layer"]
    sort_cols = ["model_alias", "family", "relative_layer"]
    sort_ascending = [True, True, True]
    if include_architecture:
        group_cols.append("architecture")
        sort_cols.append("architecture")
        sort_ascending.append(True)
    return (
        summary_df.loc[summary_df["partition"] == partition]
        .sort_values(
            sort_cols + metric_cols,
            ascending=sort_ascending + metric_ascending,
        )
        .groupby(group_cols, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _table_html(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame.loc[:, columns].to_html(index=False, escape=False)


def _subplot_dims(n_items: int, *, n_cols: int = 3) -> tuple[int, int]:
    cols = min(n_cols, max(1, n_items))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def _best_metric_grid_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    facet_key_col: str,
    facet_label_col: str,
    line_key_col: str,
    line_label_col: str,
    baseline_frame: pd.DataFrame | None = None,
    preferred_facet_keys: list[str] | None = None,
    preferred_line_keys: list[str] | None = None,
    include_plotlyjs: str | bool = False,
) -> str:
    metric_label = METRIC_SPECS[metric]["label"]
    facet_labels = _ordered_labels(frame, key_col=facet_key_col, label_col=facet_label_col, preferred_keys=preferred_facet_keys)
    line_labels = _ordered_labels(frame, key_col=line_key_col, label_col=line_label_col, preferred_keys=preferred_line_keys)
    rows, cols = _subplot_dims(len(facet_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=facet_labels, shared_xaxes=True)

    label_to_facet = (
        frame.loc[:, [facet_key_col, facet_label_col]]
        .drop_duplicates()
        .set_index(facet_label_col)[facet_key_col]
        .to_dict()
    )
    label_to_line = (
        frame.loc[:, [line_key_col, line_label_col]]
        .drop_duplicates()
        .set_index(line_label_col)[line_key_col]
        .to_dict()
    )

    for idx, facet_label in enumerate(facet_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        facet_key = label_to_facet[facet_label]
        facet_frame = frame.loc[frame[facet_key_col] == facet_key].copy()
        for line_label in line_labels:
            line_key = label_to_line[line_label]
            line_frame = facet_frame.loc[facet_frame[line_key_col] == line_key].sort_values("relative_layer")
            if line_frame.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=line_frame["relative_layer"],
                    y=line_frame[metric],
                    mode="lines+markers",
                    name=line_label,
                    legendgroup=line_label,
                    showlegend=(idx == 0),
                    hovertemplate=(
                        f"{facet_label}<br>"
                        f"{line_label}<br>"
                        "rel_layer=%{x:.1f}<br>"
                        f"{metric_label}=%{{y:.4f}}<br>"
                        "layer_idx=%{customdata[0]}<br>"
                        "winner_arch=%{customdata[1]}<br>"
                        "control=%{customdata[2]}<extra></extra>"
                    ),
                    customdata=list(zip(line_frame["layer_idx"], line_frame["architecture"], line_frame["control_value"])),
                ),
                row=row,
                col=col,
            )
        if metric == "mean_abs_error" and baseline_frame is not None and not baseline_frame.empty:
            baseline_line = (
                baseline_frame.loc[baseline_frame[facet_key_col] == facet_key]
                .groupby("relative_layer", as_index=False)
                .agg(
                    mean_abs_error=("mean_abs_error", "mean"),
                    train_mean_source_value=("control_value", "mean"),
                )
                .sort_values("relative_layer")
            )
            if not baseline_line.empty:
                fig.add_trace(
                    go.Scatter(
                        x=baseline_line["relative_layer"],
                        y=baseline_line["mean_abs_error"],
                        mode="lines",
                        name="train-mean baseline",
                        legendgroup="train-mean baseline",
                        showlegend=(idx == 0),
                        line=dict(color=BASELINE_LINE_COLOR, dash="dash", width=2),
                        hovertemplate=(
                            f"{facet_label}<br>"
                            "reference=train-mean baseline<br>"
                            "rel_layer=%{x:.1f}<br>"
                            f"{metric_label}=%{{y:.4f}}<br>"
                            "mean train source value=%{customdata[0]:.2f}<extra></extra>"
                        ),
                        customdata=np.stack(
                            [baseline_line["train_mean_source_value"].to_numpy(dtype=float)],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(
            tickmode="array",
            tickvals=sorted(facet_frame["relative_layer"].drop_duplicates().tolist()),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=310 * rows,
        width=430 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.04, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _family_architecture_metric_figure(
    frame: pd.DataFrame,
    *,
    family_label: str,
    metric: str,
    baseline_frame: pd.DataFrame | None = None,
    include_plotlyjs: str | bool = False,
) -> str:
    metric_label = METRIC_SPECS[metric]["label"]
    subset = frame.loc[frame["family_label"] == family_label].copy()
    model_labels = _ordered_labels(subset, key_col="model_alias", label_col="model_label", preferred_keys=MODEL_ORDER)
    rows, cols = _subplot_dims(len(model_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=model_labels, shared_xaxes=True)
    label_to_model = (
        subset.loc[:, ["model_alias", "model_label"]]
        .drop_duplicates()
        .set_index("model_label")["model_alias"]
        .to_dict()
    )

    for idx, model_label in enumerate(model_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        model_alias = label_to_model[model_label]
        cell = subset.loc[subset["model_alias"] == model_alias].copy()
        for architecture in ARCH_ORDER:
            arch = cell.loc[cell["architecture"] == architecture].sort_values("relative_layer")
            if arch.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=arch["relative_layer"],
                    y=arch[metric],
                    mode="lines+markers",
                    name=architecture,
                    legendgroup=architecture,
                    showlegend=(idx == 0),
                    line=dict(color=ARCH_COLORS[architecture]),
                    hovertemplate=(
                        f"{family_label}<br>"
                        f"{model_label}<br>"
                        "arch=%{fullData.name}<br>"
                        "rel_layer=%{x:.1f}<br>"
                        f"{metric_label}=%{{y:.4f}}<br>"
                        "layer_idx=%{customdata[0]}<br>"
                        "control=%{customdata[1]}<extra></extra>"
                    ),
                    customdata=list(zip(arch["layer_idx"], arch["control_value"])),
                ),
                row=row,
                col=col,
            )
        if metric == "mean_abs_error" and baseline_frame is not None and not baseline_frame.empty:
            baseline_line = (
                baseline_frame.loc[
                    (baseline_frame["family_label"] == family_label) & (baseline_frame["model_alias"] == model_alias)
                ]
                .groupby("relative_layer", as_index=False)
                .agg(
                    mean_abs_error=("mean_abs_error", "mean"),
                    train_mean_source_value=("control_value", "mean"),
                )
                .sort_values("relative_layer")
            )
            if not baseline_line.empty:
                fig.add_trace(
                    go.Scatter(
                        x=baseline_line["relative_layer"],
                        y=baseline_line["mean_abs_error"],
                        mode="lines",
                        name="train-mean baseline",
                        legendgroup="train-mean baseline",
                        showlegend=(idx == 0),
                        line=dict(color=BASELINE_LINE_COLOR, dash="dash", width=2),
                        hovertemplate=(
                            f"{family_label}<br>"
                            f"{model_label}<br>"
                            "reference=train-mean baseline<br>"
                            "rel_layer=%{x:.1f}<br>"
                            f"{metric_label}=%{{y:.4f}}<br>"
                            "train source mean=%{customdata[0]:.2f}<extra></extra>"
                        ),
                        customdata=np.stack(
                            [baseline_line["train_mean_source_value"].to_numpy(dtype=float)],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(
            tickmode="array",
            tickvals=sorted(cell["relative_layer"].drop_duplicates().tolist()),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"{family_label} | Best Test {metric_label} by Relative Layer",
        template="plotly_white",
        height=310 * rows,
        width=430 * cols,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", y=1.04, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _architecture_win_figure(winners: pd.DataFrame) -> str:
    counts = (
        winners.groupby(["relative_layer", "architecture"], as_index=False)
        .size()
        .rename(columns={"size": "num_wins"})
        .sort_values(["relative_layer", "architecture"])
    )
    fig = go.Figure()
    for architecture in ARCH_ORDER:
        arch = counts.loc[counts["architecture"] == architecture].sort_values("relative_layer")
        fig.add_trace(
            go.Scatter(
                x=arch["relative_layer"],
                y=arch["num_wins"],
                mode="lines+markers",
                name=architecture,
                line=dict(color=ARCH_COLORS[architecture]),
            )
        )
    fig.update_layout(
        title="Test Winner Counts by Relative Layer",
        template="plotly_white",
        height=360,
        width=900,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.12, x=0.0),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _mean_metric_by_architecture_figure(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    baseline_frame: pd.DataFrame | None = None,
    include_plotlyjs: str | bool = False,
) -> str:
    metric_label = METRIC_SPECS[metric]["label"]
    model_labels = _ordered_labels(frame, key_col="model_alias", label_col="model_label", preferred_keys=MODEL_ORDER)
    rows, cols = _subplot_dims(len(model_labels))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=model_labels, shared_xaxes=True)
    label_to_model = (
        frame.loc[:, ["model_alias", "model_label"]]
        .drop_duplicates()
        .set_index("model_label")["model_alias"]
        .to_dict()
    )

    grouped = (
        frame.groupby(["model_alias", "architecture", "relative_layer"], as_index=False)
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            layer_idx_min=("layer_idx", "min"),
            layer_idx_max=("layer_idx", "max"),
            num_prompt_styles=("family_label", "nunique"),
        )
        .sort_values(["model_alias", "architecture", "relative_layer"])
        .reset_index(drop=True)
    )
    grouped["metric_std"] = grouped["metric_std"].fillna(0.0)

    for idx, model_label in enumerate(model_labels):
        row = idx // cols + 1
        col = idx % cols + 1
        model_alias = label_to_model[model_label]
        cell = grouped.loc[grouped["model_alias"] == model_alias].copy()
        for architecture in ARCH_ORDER:
            arch = cell.loc[cell["architecture"] == architecture].sort_values("relative_layer")
            if arch.empty:
                continue
            color = ARCH_COLORS[architecture]
            band_color = color.replace("#", "")
            rgba = f"rgba({int(band_color[0:2], 16)}, {int(band_color[2:4], 16)}, {int(band_color[4:6], 16)}, 0.16)"
            upper = arch["metric_mean"] + arch["metric_std"]
            lower = arch["metric_mean"] - arch["metric_std"]
            fig.add_trace(
                go.Scatter(
                    x=arch["relative_layer"],
                    y=upper,
                    mode="lines",
                    line=dict(width=0, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=architecture,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=arch["relative_layer"],
                    y=lower,
                    mode="lines",
                    line=dict(width=0, color=color),
                    fill="tonexty",
                    fillcolor=rgba,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=architecture,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=arch["relative_layer"],
                    y=arch["metric_mean"],
                    mode="lines+markers",
                    name=architecture,
                    legendgroup=architecture,
                    showlegend=(idx == 0),
                    line=dict(color=color),
                    hovertemplate=(
                        f"{model_label}<br>"
                        "arch=%{fullData.name}<br>"
                        "rel_layer=%{x:.1f}<br>"
                        f"mean {metric_label}=%{{y:.4f}}<br>"
                        "std across prompt styles=%{customdata[0]:.4f}<br>"
                        "layer_idx_range=%{customdata[1]}..%{customdata[2]}<br>"
                        "prompt_styles=%{customdata[3]}<extra></extra>"
                    ),
                    customdata=list(
                        zip(
                            arch["metric_std"],
                            arch["layer_idx_min"],
                            arch["layer_idx_max"],
                            arch["num_prompt_styles"],
                        )
                    ),
                ),
                row=row,
                col=col,
            )
        if metric == "mean_abs_error" and baseline_frame is not None and not baseline_frame.empty:
            baseline_line = (
                baseline_frame.loc[baseline_frame["model_alias"] == model_alias]
                .groupby("relative_layer", as_index=False)
                .agg(
                    mean_abs_error=("mean_abs_error", "mean"),
                    train_mean_source_value=("control_value", "mean"),
                    num_prompt_styles=("family_label", "nunique"),
                )
                .sort_values("relative_layer")
            )
            if not baseline_line.empty:
                fig.add_trace(
                    go.Scatter(
                        x=baseline_line["relative_layer"],
                        y=baseline_line["mean_abs_error"],
                        mode="lines",
                        name="train-mean baseline",
                        legendgroup="train-mean baseline",
                        showlegend=(idx == 0),
                        line=dict(color=BASELINE_LINE_COLOR, dash="dash", width=2),
                        hovertemplate=(
                            f"{model_label}<br>"
                            "reference=train-mean baseline<br>"
                            "rel_layer=%{x:.1f}<br>"
                            f"mean {metric_label}=%{{y:.4f}}<br>"
                            "mean train source value=%{customdata[0]:.2f}<br>"
                            "prompt_styles=%{customdata[1]}<extra></extra>"
                        ),
                        customdata=np.stack(
                            [
                                baseline_line["train_mean_source_value"].to_numpy(dtype=float),
                                baseline_line["num_prompt_styles"].to_numpy(dtype=int),
                            ],
                            axis=1,
                        ),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(
            tickmode="array",
            tickvals=sorted(cell["relative_layer"].drop_duplicates().tolist()),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
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
    summary_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    ensure_dir(report_path.parent)
    ensure_dir(artifact_dir)
    probe_summary_df = _competitive_summary_df(summary_df)
    baseline_test_df = _baseline_summary_df(summary_df, partition="test")
    best_dev_accuracy_arch = _best_rows(probe_summary_df, partition="dev", metric="accuracy", include_architecture=True)
    best_test_accuracy_arch = _best_rows(probe_summary_df, partition="test", metric="accuracy", include_architecture=True)
    best_test_mae_arch = _best_rows(probe_summary_df, partition="test", metric="mean_abs_error", include_architecture=True)
    best_test_rmse_arch = _best_rows(probe_summary_df, partition="test", metric="rmse", include_architecture=True)

    winner_accuracy = _best_rows(probe_summary_df, partition="test", metric="accuracy", include_architecture=False)
    winner_mae = _best_rows(probe_summary_df, partition="test", metric="mean_abs_error", include_architecture=False)
    winner_rmse = _best_rows(probe_summary_df, partition="test", metric="rmse", include_architecture=False)

    atomic_write_jsonl(artifact_dir / "summary_metrics.jsonl", summary_df.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_dev_accuracy_by_architecture.jsonl", best_dev_accuracy_arch.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test_accuracy_by_architecture.jsonl", best_test_accuracy_arch.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test_mae_by_architecture.jsonl", best_test_mae_arch.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test_rmse_by_architecture.jsonl", best_test_rmse_arch.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test_accuracy_overall.jsonl", winner_accuracy.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test_mae_overall.jsonl", winner_mae.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test_rmse_overall.jsonl", winner_rmse.to_dict("records"))
    atomic_write_json(artifact_dir / "metadata.json", metadata)

    winner_counts = (
        winner_accuracy.groupby("architecture", as_index=False)
        .size()
        .rename(columns={"size": "num_family_model_layer_cells_won"})
        .sort_values("num_family_model_layer_cells_won", ascending=False)
        .reset_index(drop=True)
    )
    metric_winner_table = pd.DataFrame(
        [
            {
                "metric": METRIC_SPECS["accuracy"]["label"],
                "best_mean_value": winner_accuracy["accuracy"].mean(),
                "best_median_value": winner_accuracy["accuracy"].median(),
            },
            {
                "metric": METRIC_SPECS["mean_abs_error"]["label"],
                "best_mean_value": winner_mae["mean_abs_error"].mean(),
                "best_median_value": winner_mae["mean_abs_error"].median(),
            },
            {
                "metric": METRIC_SPECS["rmse"]["label"],
                "best_mean_value": winner_rmse["rmse"].mean(),
                "best_median_value": winner_rmse["rmse"].median(),
            },
        ]
    )
    winner_arch_by_metric = []
    for metric_name, frame in [("accuracy", winner_accuracy), ("mean_abs_error", winner_mae), ("rmse", winner_rmse)]:
        counts = (
            frame.groupby("architecture", as_index=False)
            .size()
            .rename(columns={"size": "num_cells"})
            .sort_values("num_cells", ascending=False)
        )
        counts["metric"] = METRIC_SPECS[metric_name]["label"]
        winner_arch_by_metric.append(counts)
    winner_arch_by_metric_df = pd.concat(winner_arch_by_metric, ignore_index=True)

    model_summary_plots = []
    family_summary_plots = []
    family_detail_sections = []
    plotlyjs_mode: str | bool = "cdn"
    for metric_name in ("accuracy", "mean_abs_error", "rmse"):
        winner_frame = {
            "accuracy": winner_accuracy,
            "mean_abs_error": winner_mae,
            "rmse": winner_rmse,
        }[metric_name]
        model_summary_plots.append(
            _best_metric_grid_figure(
                winner_frame,
                metric=metric_name,
                title=f"Best Test {METRIC_SPECS[metric_name]['label']} by Relative Layer | One Panel per Model",
                facet_key_col="model_alias",
                facet_label_col="model_label",
                line_key_col="family",
                line_label_col="family_label",
                baseline_frame=baseline_test_df,
                preferred_facet_keys=MODEL_ORDER,
                preferred_line_keys=[item.lower().replace(" ", "_") for item in []],
                include_plotlyjs=plotlyjs_mode,
            )
        )
        plotlyjs_mode = False
        family_summary_plots.append(
            _best_metric_grid_figure(
                winner_frame,
                metric=metric_name,
                title=f"Best Test {METRIC_SPECS[metric_name]['label']} by Relative Layer | One Panel per Prompt Style",
                facet_key_col="family",
                facet_label_col="family_label",
                line_key_col="model_alias",
                line_label_col="model_label",
                baseline_frame=baseline_test_df,
                preferred_facet_keys=[label.lower().replace(" ", "_") for label in []],
                preferred_line_keys=MODEL_ORDER,
                include_plotlyjs=plotlyjs_mode,
            )
        )

    for family_label in FAMILY_ORDER:
        if family_label not in set(best_test_accuracy_arch["family_label"]):
            continue
        plots = []
        for metric_name, frame in [
            ("accuracy", best_test_accuracy_arch),
            ("mean_abs_error", best_test_mae_arch),
            ("rmse", best_test_rmse_arch),
        ]:
            plots.append(
                _family_architecture_metric_figure(
                    frame,
                    family_label=family_label,
                    metric=metric_name,
                    baseline_frame=baseline_test_df,
                    include_plotlyjs=plotlyjs_mode,
                )
            )
            plotlyjs_mode = False
        family_detail_sections.append(
            f"<h3>{family_label}</h3><div class='stacked-figs'>{''.join(f'<div class=\"wide-fig\">{plot}</div>' for plot in plots)}</div>"
        )

    mae_architecture_average_plot = _mean_metric_by_architecture_figure(
        best_test_mae_arch,
        metric="mean_abs_error",
        title="Mean Absolute Error by Relative Layer | Architecture Mean Across Prompt Styles, One Panel per Model",
        baseline_frame=baseline_test_df,
        include_plotlyjs=False,
    )

    metric_glossary_html = "".join(
        f"<li><strong>{spec['label']}</strong>: {spec['description']} Formula: <code>{spec['formula']}</code>.</li>"
        for spec in METRIC_SPECS.values()
    )
    num_models = len(list(metadata.get("models", [])))
    num_prompt_families = len(list(metadata.get("prompt_families", [])))
    num_relative_layers = len(list(metadata.get("relative_layers", [])))
    num_probe_architectures = len(PROBE_ARCH_ORDER)
    num_cells = num_models * num_prompt_families * num_relative_layers
    architecture_labels = ", ".join(
        f"<code>{probe_architecture_display_name(architecture)}</code>" for architecture in PROBE_ARCH_ORDER
    )

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
    .mini-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
    .mini-fig {{ flex: 0 0 auto; }}
    .stacked-figs {{ display: flex; flex-direction: column; gap: 20px; }}
    .wide-fig {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{metadata['report_title']}</h1>
  <h2>Research Question</h2>
  <p>
    How does source-value probe quality change with relative layer, prompt style, model, and probe architecture when capacity is swept within each architecture on canonical <code>{metadata['split_name']}</code>?
  </p>

  <h2>Scope of Evidence</h2>
  <p>
    This report covers all {num_cells} cells from the completed relative-layer sweep: {num_models} models, {num_prompt_families} prompt styles, and {num_relative_layers} relative layers from <code>0.0</code> through <code>1.0</code>.
    Relative layer mapping follows the repo standard: <code>layer_idx = round(rel * n_layers)</code>, clipped to <code>[0, n_layers-1]</code>.
    Each cell compares {architecture_labels} over a matched effective-df ladder.
  </p>

  <h2>Executive Summary</h2>
  <p>
    The updated report focuses on metric-specific best-by-layer summaries rather than only accuracy winners.
    It separates three questions: which method reaches the highest exact recovery, which minimizes absolute error, and which minimizes squared error.
    The figures below expose the layer dependence both across models and across prompt styles, then break the same trends down by architecture inside each prompt family.
  </p>

  <h2>Main Findings</h2>
  <ul>
    <li>Best-by-layer behavior is now shown separately for <code>Exact Accuracy</code>, <code>Mean Absolute Error</code>, and <code>Root Mean Squared Error</code>, instead of forcing all judgments through accuracy alone.</li>
    <li>The summary views show the same layer sweep from two directions: one panel per model with prompt-style lines, and one panel per prompt style with model lines.</li>
    <li>The detailed family sections show the architecture comparison directly, one panel per model, so the layer at which the four probe families become competitive is visible without referring back to raw JSONL.</li>
  </ul>

  <h2>Key Tables and Figures</h2>
  <p>
    Metric glossary:
  </p>
  <ul>
    {metric_glossary_html}
  </ul>

  <h3>Overall Accuracy Winner Counts</h3>
  {_table_html(winner_counts, ['architecture', 'num_family_model_layer_cells_won'])}

  <h3>Best-Overall Metric Summary</h3>
  {_table_html(metric_winner_table, ['metric', 'best_mean_value', 'best_median_value'])}

  <h3>Winning Architecture by Metric</h3>
  {_table_html(winner_arch_by_metric_df, ['metric', 'architecture', 'num_cells'])}

  <h3>Accuracy Winner Count by Relative Layer</h3>
  {_architecture_win_figure(winner_accuracy)}

  <h3>MAE by Architecture, Averaged Across Prompt Styles</h3>
  <p>
    Each line shows the mean best-test <code>Mean Absolute Error</code> for one probe architecture after averaging over the five prompt styles at a fixed model and relative layer.
    Shaded bands show <code>&plusmn;1</code> standard deviation across prompt styles, and the gray dashed line shows the train-mean baseline.
  </p>
  {mae_architecture_average_plot}

  <h3>Model-Centered Summary Plots</h3>
  {''.join(f"<div class='wide-fig'>{plot}</div>" for plot in model_summary_plots)}

  <h3>Prompt-Style-Centered Summary Plots</h3>
  {''.join(f"<div class='wide-fig'>{plot}</div>" for plot in family_summary_plots)}

  <h3>Best Test Rows by Accuracy Objective</h3>
  {_table_html(best_test_accuracy_arch.sort_values(['family_label', 'model_label', 'relative_layer', 'architecture']).reset_index(drop=True), ['family_label', 'model_label', 'relative_layer', 'layer_idx', 'architecture', 'control_name', 'control_value', 'effective_df', 'accuracy', 'mean_abs_error', 'rmse', 'r2', 'spearman'])}

  <h2>Interpretation</h2>
  <p>
    The summary plots now distinguish metric-specific layer optima from architecture-specific layer curves. This matters because exact recovery, absolute error, and squared error need not peak at the same layer or under the same probe family.
    The model-centered plots answer whether prompt-style sensitivity emerges at the same depth inside a given model, while the prompt-style-centered plots answer whether the same prompt family behaves similarly across models.
  </p>

  <h2>Limitations and Negative Results</h2>
  <p>
    These figures are all based on <code>split_00</code> only. They compare capacity sweeps within each architecture, but not cross-validation variance across data splits.
    Also, the relative layer <code>0.0</code> block still inherits the known limitation that prompt context has not yet mixed into the representation, so prompt-style curves there should be interpreted as embedding-level rather than context-sensitive.
  </p>

  <h2>Implications</h2>
  <p>
    This report is intended to make the next selection step easier: choose which model, prompt family, and relative-layer bands are worth carrying into broader cross-validation or steering experiments.
    The detailed family sections below expose whether those choices differ by probe family or stay stable across the three objectives.
  </p>

  <h2>Recommended Next Experiments</h2>
  <ul>
    <li>Repeat the same relative-layer report surface on additional CV splits to separate genuine layer effects from split-specific noise.</li>
    <li>Use the strongest layer bands from this report as candidate probe-training layers for the planned cross-prompt generalization experiments.</li>
    <li>Where exact accuracy and error metrics disagree, inspect example-level predictions to determine whether the issue is rounding failure or a genuinely distorted numeric code.</li>
  </ul>

  <h2>Detailed Family Breakdowns</h2>
  {''.join(family_detail_sections)}
</body>
</html>
"""
    atomic_write_text(report_path, html)


def _backfill_mean_baseline_run(run_dir: Path) -> None:
    cfg = _load_saved_experiment_config(run_dir)
    split_dataset_dir = ROOT / str(cfg["split_dataset_dir"])
    split_name = str(cfg["split_name"])
    models_cfg = load_yaml(cfg["models_config"])

    all_summary_rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        model_alias = model_dir.name
        model_entry = {**models_cfg.get("default", {}), **models_cfg["models"].get(model_alias, {})}
        model_label = str(model_entry.get("label", model_alias))
        for layer_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            for family_dir in sorted(path for path in layer_dir.iterdir() if path.is_dir()):
                combo_summary_path = family_dir / "probe_summary.jsonl"
                if not combo_summary_path.exists():
                    continue
                combo_summary_rows = load_jsonl(combo_summary_path)
                if not combo_summary_rows:
                    continue
                resolved = json.loads((family_dir / "resolved.json").read_text(encoding="utf-8"))
                family = str(resolved["family"])
                family_label = str(resolved["family_label"])
                rows_by_partition = _load_partition_rows(split_dataset_dir, split_name, family)
                train_values = rows_by_partition["train"]["source_value"].to_numpy(dtype=np.int64)
                train_mean_value = float(train_values.astype(np.float64).mean())
                existing_keys = {_summary_key(row) for row in combo_summary_rows}
                model_dim = int(resolved.get("model_dim", combo_summary_rows[0].get("model_dim", 0)))
                relative_layer = float(resolved["relative_layer"])
                layer_idx = int(resolved["layer_idx"])
                n_layers = int(resolved["n_layers"])
                changed = False
                for partition in ("train", "dev", "test"):
                    baseline_key = _summary_key(
                        {
                            "partition": partition,
                            "architecture": BASELINE_ARCHITECTURE,
                            "control_name": "train_mean_source_value",
                            "control_value": float(train_mean_value),
                        }
                    )
                    if baseline_key in existing_keys:
                        continue
                    values = rows_by_partition[partition]["source_value"].to_numpy(dtype=np.int64)
                    summary_row = _baseline_metadata_row(
                        partition=partition,
                        train_mean_value=train_mean_value,
                        model_alias=model_alias,
                        model_label=model_label,
                        family=family,
                        family_label=family_label,
                        split_name=split_name,
                        model_dim=model_dim,
                        relative_layer=relative_layer,
                        layer_idx=layer_idx,
                        n_layers=n_layers,
                        values=values,
                    )
                    combo_summary_rows.append(summary_row)
                    existing_keys.add(_summary_key(summary_row))
                    changed = True
                combo_summary_rows = sorted(
                    combo_summary_rows,
                    key=lambda row: (
                        str(row["architecture"]),
                        int(row["step_index"]),
                        str(row["partition"]),
                    ),
                )
                if changed:
                    atomic_write_jsonl(combo_summary_path, combo_summary_rows)
                all_summary_rows.extend(combo_summary_rows)

    all_summary_rows = sorted(
        all_summary_rows,
        key=lambda row: (
            str(row["model_alias"]),
            float(row["relative_layer"]),
            str(row["family"]),
            str(row["architecture"]),
            int(row["step_index"]),
            str(row["partition"]),
        ),
    )
    summary_df = pd.DataFrame(all_summary_rows)
    atomic_write_jsonl(run_dir / "summary_metrics.jsonl", all_summary_rows)
    report_stem = str(cfg["report_stem"])
    report_path = ROOT / "experiment-reports" / f"{report_stem}.html"
    artifact_dir = ROOT / "experiment-reports" / "artifacts" / report_stem
    _write_report(
        report_path=report_path,
        artifact_dir=artifact_dir,
        summary_df=summary_df,
        metadata={
            "report_title": "Source-Value Probe Capacity Sweep | split_00 | Relative Layers",
            "split_name": split_name,
            "representation_source": str(cfg.get("representation_source", "prompt_activations")),
            "relative_layers": [float(value) for value in cfg["relative_layers"]],
            "models": [str(item) for item in cfg["models"]],
            "prompt_families": [dict(item) for item in cfg["prompt_families"]],
            "circular_dense_low_df_max": cfg.get("circular_dense_low_df_max"),
            "seed_output_dir": cfg.get("seed_output_dir"),
        },
    )
    print(run_dir)
    print(report_path)


def main() -> None:
    args = parse_args()
    if args.backfill_mean_baseline_run_dir:
        _backfill_mean_baseline_run(Path(args.backfill_mean_baseline_run_dir))
        return

    cfg = load_yaml(args.config)
    if args.split_name:
        cfg["split_name"] = str(args.split_name)
    if args.report_stem:
        cfg["report_stem"] = str(args.report_stem)
    set_global_seed(int(cfg["seed"]))

    models_cfg = load_yaml(cfg["models_config"])
    model_aliases = [str(item) for item in cfg["models"]]
    family_specs = [dict(item) for item in cfg["prompt_families"]]
    relative_layers = [float(value) for value in cfg["relative_layers"]]
    split_dataset_dir = Path(cfg["split_dataset_dir"])
    split_name = str(cfg["split_name"])
    representation_source = str(cfg.get("representation_source", "prompt_activations"))
    circular_dense_low_df_max = cfg.get("circular_dense_low_df_max")
    circular_cls_config = dict(cfg.get("circular_cls_config", {}))
    use_activation_cache = bool(cfg.get("use_activation_cache", True))
    require_activation_cache_hit = bool(cfg.get("require_activation_cache_hit", False))
    if circular_dense_low_df_max is not None:
        circular_dense_low_df_max = int(circular_dense_low_df_max)
    seed_output_dir = cfg.get("seed_output_dir")
    seed_output_dir_path = Path(seed_output_dir) if seed_output_dir else None
    if representation_source != "prompt_activations":
        raise ValueError(f"This runner currently supports representation_source='prompt_activations' only, got {representation_source!r}.")

    hash_payload = {"task_name": "source_value_probe_capacity_sweep_relative_layers", "config": cfg}
    run_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    output_dir = Path(args.output_root) / f"source_value_probe_capacity_sweep_relative_layers__{run_hash}__r0"
    if (output_dir / "DONE").exists() and not args.force:
        print(output_dir)
        return
    ensure_dir(output_dir)
    atomic_write_text(output_dir / "command.txt", " ".join(sys.argv) + "\n")
    dump_run_config(output_dir / "config.json", {"args": vars(args), "experiment_config": cfg})

    all_summary_rows: list[dict[str, Any]] = []

    for model_alias in model_aliases:
        seeded_model_rows = _seed_model_from_existing_run(
            output_dir=output_dir,
            seed_output_dir=seed_output_dir_path,
            model_alias=model_alias,
            split_name=split_name,
            relative_layers=relative_layers,
            family_specs=family_specs,
        )
        if seeded_model_rows is not None:
            print(f"[seed] {model_alias} from {seed_output_dir_path}")
            all_summary_rows.extend(seeded_model_rows)
            continue

        model_entry = {**models_cfg.get("default", {}), **models_cfg["models"][model_alias]}
        model_name = str(model_entry["hf_id"])
        trust_remote_code = bool(model_entry.get("trust_remote_code", False))
        n_layers = _resolve_n_layers(model_name=model_name, trust_remote_code=trust_remote_code)
        device = str(model_entry.get("device", "cpu"))
        dtype = str(model_entry.get("dtype", "float32"))
        batch_size = int(cfg.get("probe_batch_size_by_model", {}).get(model_alias, 64))
        circular_cls_fit_config = {
            **circular_cls_config,
            "seed": int(circular_cls_config.get("seed", int(cfg["seed"]))),
            "device": str(circular_cls_config.get("device", device)),
        }
        print(f"[load] {model_alias} ({model_name})")
        _write_progress(output_dir, status="running", phase="model_load", model_alias=model_alias)
        model_revision = resolve_model_revision(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=model_entry.get("revision"),
        )
        if use_activation_cache and require_activation_cache_hit:
            tokenizer = load_tokenizer(
                model_name,
                trust_remote_code=trust_remote_code,
                revision=model_entry.get("revision"),
            )
            loaded_model = None
            loaded_attn_implementation = model_entry.get("attn_implementation")
        else:
            loaded = load_causal_lm(
                model_name=model_name,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=model_entry.get("attn_implementation"),
                revision=model_entry.get("revision"),
            )
            tokenizer = loaded.tokenizer
            loaded_model = loaded.model
            loaded_attn_implementation = loaded.attn_implementation
            model_revision = loaded.model_revision

        for relative_layer in relative_layers:
            layer_idx = _layer_rel_to_abs(relative_layer, n_layers=n_layers)
            print(f"[layer] {model_alias} rel={relative_layer:.1f} abs={layer_idx}")
            for family_spec in family_specs:
                family = str(family_spec["family"])
                family_label = str(family_spec["family_label"])
                print(f"[run] {model_alias} | rel={relative_layer:.1f} | {family}")
                combo_dir = ensure_dir(output_dir / model_alias / f"layer_rel_{relative_layer:.1f}".replace(".", "_") / family)
                combo_summary_path = combo_dir / "probe_summary.jsonl"
                if (combo_dir / "DONE").exists() and combo_summary_path.exists():
                    all_summary_rows.extend(load_jsonl(combo_summary_path))
                    if cfg.get("checkpoint_each_fit", False):
                        _restore_progress(combo_dir)
                    continue
                _write_progress(output_dir, status="running", phase="activations", model_alias=model_alias,
                                family=family, relative_layer=relative_layer, layer_idx=layer_idx)
                rows_by_partition = _load_partition_rows(split_dataset_dir, split_name, family)
                activations_by_partition = _collect_partition_activations(
                    rows_by_partition=rows_by_partition,
                    model=loaded_model,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    device=device,
                    model_alias=model_alias,
                    model_id=model_name,
                    model_revision=model_revision,
                    attn_implementation=loaded_attn_implementation,
                    family=family,
                    activation_cache_root=cfg.get("activation_cache_root"),
                    use_activation_cache=use_activation_cache,
                    require_activation_cache_hit=require_activation_cache_hit,
                )
                train_values, train_x = activations_by_partition["train"]
                model_dim = int(train_x.shape[1])
                base_circular_steps = _power_of_two_ladder(model_dim)
                circular_steps = _dense_prefix_ladder(model_dim, circular_dense_low_df_max)
                circular_step_index_by_value = {int(value): idx for idx, value in enumerate(circular_steps)}
                pls_max_components = min(model_dim, int(train_x.shape[0] - 1))
                pls_steps = _power_of_two_ladder(pls_max_components)
                ridge_schedule = _build_ridge_schedule(train_x, num_steps=len(base_circular_steps))

                combo_summary_rows = _load_seed_rows(
                    current_combo_dir=combo_dir,
                    seed_output_dir=seed_output_dir_path,
                    model_alias=model_alias,
                    relative_layer=float(relative_layer),
                    family=family,
                    split_name=split_name,
                    circular_step_index_by_value=circular_step_index_by_value,
                )
                if cfg.get("checkpoint_each_fit", False):
                    resumed_rows = _restore_progress(combo_dir)
                    if resumed_rows is not None:
                        combo_summary_rows = resumed_rows
                existing_keys = {_summary_key(row) for row in combo_summary_rows}
                train_mean_value = float(np.asarray(train_values, dtype=np.float64).mean())
                for partition in ("train", "dev", "test"):
                    baseline_key = _summary_key(
                        {
                            "partition": partition,
                            "architecture": BASELINE_ARCHITECTURE,
                            "control_name": "train_mean_source_value",
                            "control_value": float(train_mean_value),
                        }
                    )
                    if baseline_key in existing_keys:
                        continue
                    values, _ = activations_by_partition[partition]
                    summary_row = _baseline_metadata_row(
                        partition=partition,
                        train_mean_value=train_mean_value,
                        model_alias=model_alias,
                        model_label=str(model_entry.get("label", model_alias)),
                        family=family,
                        family_label=family_label,
                        split_name=split_name,
                        model_dim=model_dim,
                        relative_layer=float(relative_layer),
                        layer_idx=int(layer_idx),
                        n_layers=int(n_layers),
                        values=values,
                    )
                    combo_summary_rows.append(summary_row)
                    existing_keys.add(_summary_key(summary_row))
                architecture_sweeps: list[tuple[str, list[dict[str, Any]]]] = [
                    (
                        "linear",
                        [
                            {
                                "step_index": item["step_index"],
                                "control_name": "ridge_lambda",
                                "control_value": item["ridge_lambda"],
                                "effective_df": item["effective_df"],
                                "fit_config": {"ridge_lambda": item["ridge_lambda"]},
                            }
                            for item in ridge_schedule
                        ],
                    ),
                    (
                        "circular_reg",
                        [
                            {
                                "step_index": idx,
                                "control_name": "basis_dim",
                                "control_value": int(value),
                                "effective_df": float(value),
                                "fit_config": {"basis_dim": int(value)},
                            }
                            for idx, value in enumerate(circular_steps)
                        ],
                    ),
                    (
                        "circular_cls",
                        [
                            {
                                "step_index": idx,
                                "control_name": "basis_dim",
                                "control_value": int(value),
                                "effective_df": float(value),
                                "fit_config": {
                                    **circular_cls_fit_config,
                                    "basis_dim": int(value),
                                },
                            }
                            for idx, value in enumerate(circular_steps)
                        ],
                    ),
                    (
                        "pls",
                        [
                            {
                                "step_index": idx,
                                "control_name": "n_components",
                                "control_value": int(value),
                                "effective_df": float(value),
                                "fit_config": {"n_components": int(value)},
                            }
                            for idx, value in enumerate(pls_steps)
                        ],
                    ),
                ]

                for architecture, sweep_rows in architecture_sweeps:
                    for sweep_row in sweep_rows:
                        needed_partitions = []
                        for partition in ("train", "dev", "test"):
                            key = _summary_key(
                                {
                                    "partition": partition,
                                    "architecture": architecture,
                                    "control_name": sweep_row["control_name"],
                                    "control_value": sweep_row["control_value"],
                                }
                            )
                            if key not in existing_keys:
                                needed_partitions.append(partition)
                        if not needed_partitions:
                            continue
                        _write_progress(output_dir, status="running", phase="probe_fit", model_alias=model_alias,
                                        family=family, relative_layer=relative_layer, layer_idx=layer_idx,
                                        architecture=architecture, control_value=sweep_row["control_value"],
                                        completed_fits=sum(row["partition"] == "test" and row["architecture"] != BASELINE_ARCHITECTURE
                                                           for row in all_summary_rows + combo_summary_rows))
                        probe = fit_magnitude_probe(
                            architecture=architecture,
                            train_x=train_x,
                            train_values=train_values,
                            config=sweep_row["fit_config"],
                            value_min=0,
                            value_max=999,
                        )
                        for partition in needed_partitions:
                            values, split_x = activations_by_partition[partition]
                            detail_df, base_summary = evaluate_probe_split(
                                probe=probe,
                                representations=split_x,
                                values=values,
                                split_name=partition,
                            )
                            detail_df = detail_df.rename(columns={"split": "partition"})
                            summary_row = _augment_summary(detail_df, {**base_summary, "partition": partition})
                            summary_row.update(
                                {
                                    "architecture": architecture,
                                    "step_index": int(sweep_row["step_index"]),
                                    "control_name": str(sweep_row["control_name"]),
                                    "control_value": sweep_row["control_value"],
                                    "effective_df": float(sweep_row["effective_df"]),
                                    "model_alias": model_alias,
                                    "model_label": str(model_entry.get("label", model_alias)),
                                    "family": family,
                                    "family_label": family_label,
                                    "split_name": split_name,
                                    "model_dim": model_dim,
                                    "relative_layer": float(relative_layer),
                                    "layer_idx": int(layer_idx),
                                    "n_layers": int(n_layers),
                                }
                            )
                            combo_summary_rows.append(summary_row)
                            existing_keys.add(_summary_key(summary_row))
                        if cfg.get("checkpoint_each_fit", False):
                            _checkpoint_progress(combo_dir, combo_summary_rows)
                            atomic_write_jsonl(combo_summary_path, combo_summary_rows)

                combo_summary_rows = sorted(
                    combo_summary_rows,
                    key=lambda row: (
                        str(row["architecture"]),
                        int(row["step_index"]),
                        str(row["partition"]),
                    ),
                )
                all_summary_rows.extend(combo_summary_rows)

                atomic_write_jsonl(combo_dir / "probe_summary.jsonl", combo_summary_rows)
                atomic_write_json(
                    combo_dir / "resolved.json",
                    {
                        "model_alias": model_alias,
                        "model_name": model_name,
                        "family": family,
                        "family_label": family_label,
                        "split_name": split_name,
                        "representation_source": representation_source,
                        "relative_layer": float(relative_layer),
                        "layer_idx": int(layer_idx),
                        "n_layers": int(n_layers),
                        "probe_batch_size": batch_size,
                        "model_dim": model_dim,
                        "base_circular_steps": base_circular_steps,
                        "circular_steps": circular_steps,
                        "circular_dense_low_df_max": circular_dense_low_df_max,
                        "circular_cls_config": circular_cls_fit_config,
                        "pls_steps": pls_steps,
                        "ridge_schedule": ridge_schedule,
                        "seed_output_dir": str(seed_output_dir_path) if seed_output_dir_path is not None else None,
                        "activation_cache_root": str(cfg.get("activation_cache_root") or default_prompt_sequence_cache_root()),
                        "use_activation_cache": use_activation_cache,
                        "require_activation_cache_hit": require_activation_cache_hit,
                    },
                )
                atomic_write_text(combo_dir / "DONE", "ok\n")

        if not (use_activation_cache and require_activation_cache_hit):
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_summary_rows = sorted(
        all_summary_rows,
        key=lambda row: (
            str(row["model_alias"]),
            float(row["relative_layer"]),
            str(row["family"]),
            str(row["architecture"]),
            int(row["step_index"]),
            str(row["partition"]),
        ),
    )
    summary_df = pd.DataFrame(all_summary_rows)
    atomic_write_jsonl(output_dir / "summary_metrics.jsonl", all_summary_rows)
    report_stem = str(cfg["report_stem"])
    report_path = ROOT / "experiment-reports" / f"{report_stem}.html"
    artifact_dir = ROOT / "experiment-reports" / "artifacts" / report_stem
    _write_report(
        report_path=report_path,
        artifact_dir=artifact_dir,
        summary_df=summary_df,
        metadata={
            "report_title": f"Source-Value Probe Capacity Sweep | {split_name} | Relative Layers",
            "split_name": split_name,
            "representation_source": representation_source,
            "relative_layers": relative_layers,
            "models": model_aliases,
            "prompt_families": family_specs,
            "circular_dense_low_df_max": circular_dense_low_df_max,
            "circular_cls_config": circular_cls_config,
            "seed_output_dir": str(seed_output_dir_path) if seed_output_dir_path is not None else None,
        },
    )
    atomic_write_text(output_dir / "DONE", "ok\n")
    _write_progress(output_dir, status="done", phase="complete",
                    completed_fits=sum(row["partition"] == "test" and row["architecture"] != BASELINE_ARCHITECTURE
                                       for row in all_summary_rows))
    print(output_dir)
    print(report_path)


if __name__ == "__main__":
    main()
