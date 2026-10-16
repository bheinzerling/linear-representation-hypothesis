#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
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
)
from src.magnitude_intervention import evaluate_probe_split, fit_magnitude_probe
from src.model_loader import load_causal_lm
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    dump_run_config,
    ensure_dir,
    load_jsonl,
    load_yaml,
    set_global_seed,
)


DEFAULT_CONFIG = ROOT / "configs" / "source_value_probe_prompt_style_transfer_split00_relative_layers.yaml"

FAMILY_ORDER = [
    "Addition Formula",
    "Addition Verbalized",
    "Subtraction Formula",
    "Subtraction Verbalized",
    "Identity Patchscopes",
]
PROBE_ARCH_ORDER = ["circular_reg", "circular_cls", "pls", "linear"]
MODEL_ORDER = ["llama_3_2_1b", "llama_3_2_3b", "llama_3_8b", "olmo_3_7b_base", "olmo_2_13b", "gpt_oss_20b"]
ARCHITECTURE_COLORS = {
    "circular_reg": "#d62728",
    "circular_cls": "#b54d4d",
    "pls": "#2ca02c",
    "linear": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-prompt-style source-value probe transfer across relative layers."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="out/runs")
    parser.add_argument("--force", action="store_true")
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


def _family_label_map(family_specs: list[dict[str, Any]]) -> dict[str, str]:
    return {str(item["family"]): str(item["family_label"]) for item in family_specs}


def _assert_source_value_alignment(
    *,
    rows_by_family: dict[str, dict[str, pd.DataFrame]],
    family_label_by_key: dict[str, str],
) -> list[dict[str, Any]]:
    alignment_rows: list[dict[str, Any]] = []
    for partition in ("train", "dev", "test"):
        reference_values: tuple[int, ...] | None = None
        reference_family: str | None = None
        for family, rows_by_partition in rows_by_family.items():
            frame = rows_by_partition[partition]
            values = tuple(sorted(frame["source_value"].astype(int).unique().tolist()))
            alignment_rows.append(
                {
                    "partition": partition,
                    "family": family,
                    "family_label": family_label_by_key[family],
                    "num_sources": int(len(values)),
                    "source_values_hash": hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()[:16],
                    "min_source_value": int(values[0]),
                    "max_source_value": int(values[-1]),
                }
            )
            if reference_values is None:
                reference_values = values
                reference_family = family
                continue
            if values != reference_values:
                raise ValueError(
                    "Prompt styles do not share the same source_value set for partition "
                    f"{partition!r}: {family!r} differs from {reference_family!r}."
                )
    return alignment_rows


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _center_activations_by_family(
    activations_by_family: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], list[dict[str, Any]]]:
    centered: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for family, partition_map in activations_by_family.items():
        stacked = np.concatenate(
            [np.asarray(acts, dtype=np.float32) for _, acts in partition_map.values()],
            axis=0,
        )
        family_mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        mean_norm = float(np.linalg.norm(family_mean.astype(np.float64)))
        centered[family] = {}
        for partition, (values, activations) in partition_map.items():
            centered_acts = np.asarray(activations, dtype=np.float32) - family_mean[None, :]
            centered[family][partition] = (values, centered_acts.astype(np.float32, copy=False))
            residual_mean = centered_acts.mean(axis=0, dtype=np.float64)
            summary_rows.append(
                {
                    "family": family,
                    "partition": partition,
                    "num_examples": int(centered_acts.shape[0]),
                    "feature_dim": int(centered_acts.shape[1]),
                    "mean_vector_l2_norm": mean_norm,
                    "residual_mean_l2_norm": float(np.linalg.norm(residual_mean)),
                }
            )
    return centered, summary_rows


def _evaluate_probe(
    *,
    probe,
    values: np.ndarray,
    activations: np.ndarray,
    split_name: str,
) -> dict[str, Any]:
    detail_df, base_summary = evaluate_probe_split(
        probe=probe,
        representations=activations,
        values=values,
        split_name=split_name,
    )
    return _augment_summary(detail_df.rename(columns={"split": "partition"}), {**base_summary, "partition": split_name})


def _score_sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["mean_abs_error", "rmse", "accuracy", "r2"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def _architecture_sweeps(train_x: np.ndarray) -> list[tuple[str, list[dict[str, Any]]]]:
    model_dim = int(train_x.shape[1])
    circular_steps = _power_of_two_ladder(model_dim)
    pls_max_components = min(model_dim, int(train_x.shape[0] - 1))
    pls_steps = _power_of_two_ladder(pls_max_components)
    ridge_schedule = _build_ridge_schedule(train_x, num_steps=len(circular_steps))
    return [
        (
            "linear",
            [
                {
                    "step_index": int(item["step_index"]),
                    "control_name": "ridge_lambda",
                    "control_value": float(item["ridge_lambda"]),
                    "effective_df": float(item["effective_df"]),
                    "fit_config": {"ridge_lambda": float(item["ridge_lambda"])},
                }
                for item in ridge_schedule
            ],
        ),
        (
            "circular_reg",
            [
                {
                    "step_index": int(idx),
                    "control_name": "basis_dim",
                    "control_value": int(value),
                    "effective_df": float(value),
                    "fit_config": {"basis_dim": int(value)},
                }
                for idx, value in enumerate(circular_steps)
            ],
        ),
        (
            "pls",
            [
                {
                    "step_index": int(idx),
                    "control_name": "n_components",
                    "control_value": int(value),
                    "effective_df": float(value),
                    "fit_config": {"n_components": int(value)},
                }
                for idx, value in enumerate(pls_steps)
            ],
        ),
    ]


def _heatmap_matrix(
    frame: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index="train_family_label",
        columns="test_family_label",
        values=value_col,
        aggfunc="mean",
    )
    return pivot.reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER)


def _heatmap_triptych_html(
    *,
    average_matrix: pd.DataFrame,
    best_matrix: pd.DataFrame,
    worst_matrix: pd.DataFrame,
    title: str,
    best_rel_layer: float,
    best_layer_idx: int,
    best_mean_mae: float,
    worst_rel_layer: float,
    worst_layer_idx: int,
    worst_mean_mae: float,
    include_plotlyjs: str | bool = False,
) -> str:
    matrices = [average_matrix, best_matrix, worst_matrix]
    z_values = np.concatenate([matrix.to_numpy(dtype=np.float64).reshape(-1) for matrix in matrices])
    zmin = float(np.nanmin(z_values))
    zmax = float(np.nanmax(z_values))
    subplot_titles = [
        "Mean Over Relative Layers",
        f"Best Layer Mean | rel={best_rel_layer:.1f} abs={best_layer_idx} | mean MAE={best_mean_mae:.4f}",
        f"Worst Layer Mean | rel={worst_rel_layer:.1f} abs={worst_layer_idx} | mean MAE={worst_mean_mae:.4f}",
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.09,
    )
    for col_idx, matrix in enumerate(matrices, start=1):
        values = matrix.to_numpy(dtype=np.float64)
        text = np.vectorize(lambda x: f"{x:.2f}" if np.isfinite(x) else "NA")(values)
        heatmap = go.Heatmap(
            z=values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="MAE") if col_idx == 3 else None,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate=(
                "train style=%{y}<br>"
                "test style=%{x}<br>"
                "MAE=%{z:.4f}<extra></extra>"
            ),
            showscale=(col_idx == 3),
        )
        fig.add_trace(heatmap, row=1, col=col_idx)
        fig.update_xaxes(tickangle=35, row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="Train Prompt Style", row=1, col=col_idx)
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=col_idx)
    fig.update_xaxes(title_text="Test Prompt Style", row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1600,
        height=540,
        margin=dict(l=40, r=20, t=90, b=80),
    )
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _table_html(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame.loc[:, columns].to_html(index=False, escape=False)


def _write_report(
    *,
    report_path: Path,
    artifact_dir: Path,
    selected_df: pd.DataFrame,
    layer_summary_df: pd.DataFrame,
    section_summary_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    ensure_dir(report_path.parent)
    ensure_dir(artifact_dir)
    atomic_write_csv(artifact_dir / "selected_transfer_summary.csv", selected_df)
    atomic_write_csv(artifact_dir / "layer_matrix_summary.csv", layer_summary_df)
    atomic_write_csv(artifact_dir / "section_summary.csv", section_summary_df)
    atomic_write_json(artifact_dir / "metadata.json", metadata)

    test_selected = selected_df.loc[selected_df["evaluation_partition"] == "test"].copy()
    html_parts: list[str] = []
    for model_alias in MODEL_ORDER:
        for architecture in PROBE_ARCH_ORDER:
            section_row = section_summary_df.loc[
                (section_summary_df["model_alias"] == model_alias) & (section_summary_df["architecture"] == architecture)
            ]
            if section_row.empty:
                continue
            section = section_row.iloc[0]
            section_test = test_selected.loc[
                (test_selected["model_alias"] == model_alias) & (test_selected["architecture"] == architecture)
            ].copy()
            average_matrix = _heatmap_matrix(section_test, value_col="mean_abs_error")
            best_matrix = _heatmap_matrix(
                section_test.loc[section_test["relative_layer"] == float(section["best_relative_layer"])],
                value_col="mean_abs_error",
            )
            worst_matrix = _heatmap_matrix(
                section_test.loc[section_test["relative_layer"] == float(section["worst_relative_layer"])],
                value_col="mean_abs_error",
            )
            html_parts.append(
                f"""
  <section>
    <h2>{section['model_label']} | {architecture}</h2>
    <p>
      Best layer-mean relative layer: <code>{float(section['best_relative_layer']):.1f}</code>
      (abs layer <code>{int(section['best_layer_idx'])}</code>, matrix mean MAE <code>{_fmt(section['best_matrix_mean_mae'])}</code>) |
      Worst layer-mean relative layer: <code>{float(section['worst_relative_layer']):.1f}</code>
      (abs layer <code>{int(section['worst_layer_idx'])}</code>, matrix mean MAE <code>{_fmt(section['worst_matrix_mean_mae'])}</code>)
    </p>
    {_heatmap_triptych_html(
        average_matrix=average_matrix,
        best_matrix=best_matrix,
        worst_matrix=worst_matrix,
        title=f"{section['model_label']} | {architecture} | Test MAE Train-Style to Test-Style",
        best_rel_layer=float(section['best_relative_layer']),
        best_layer_idx=int(section['best_layer_idx']),
        best_mean_mae=float(section['best_matrix_mean_mae']),
        worst_rel_layer=float(section['worst_relative_layer']),
        worst_layer_idx=int(section['worst_layer_idx']),
        worst_mean_mae=float(section['worst_matrix_mean_mae']),
        include_plotlyjs='cdn' if not html_parts else False,
    )}
  </section>
"""
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
    table {{ border-collapse: collapse; margin: 16px 0 28px 0; min-width: 720px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    .caption {{ color: #444; margin-top: -8px; margin-bottom: 24px; }}
    section {{ margin-top: 44px; }}
  </style>
</head>
<body>
  <h1>{metadata['report_title']}</h1>
  <p>
    Capacity is selected separately for each <code>train prompt style × architecture × relative layer</code> cell using the native dev MAE from that train style,
    then the frozen probe is evaluated on every test prompt style. Rows are train styles, columns are test styles.
    The diagonal is prompt-style-native MAE; off-diagonals are cross-style transfer MAE.
  </p>
  <p>
    Representation centering: <code>{'family-mean centered' if metadata.get('center_representations') else 'none'}</code>.
    When enabled, each prompt style at a fixed model/layer is centered by subtracting its own mean activation vector computed over the full train+dev+test cache.
  </p>
  <p>
    Source-value matching is enforced across prompt styles before any fit/eval work begins. The run aborts if train/dev/test source sets differ across families.
  </p>

  <h2>Section Summary</h2>
  {_table_html(
      section_summary_df.sort_values(['model_alias', 'architecture']).reset_index(drop=True),
      [
          'model_label',
          'architecture',
          'best_relative_layer',
          'best_layer_idx',
          'best_matrix_mean_mae',
          'worst_relative_layer',
          'worst_layer_idx',
          'worst_matrix_mean_mae',
          'mean_over_layers_matrix_mean_mae',
      ],
  )}

  <h2>Layer Summary</h2>
  {_table_html(
      layer_summary_df.sort_values(['model_alias', 'architecture', 'relative_layer']).reset_index(drop=True),
      [
          'model_label',
          'architecture',
          'relative_layer',
          'layer_idx',
          'matrix_mean_mae',
          'diagonal_mean_mae',
          'off_diagonal_mean_mae',
      ],
  )}

  {''.join(html_parts)}
</body>
</html>
"""
    atomic_write_text(report_path, html)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["seed"]))

    models_cfg = load_yaml(cfg["models_config"])
    split_dataset_dir = Path(cfg["split_dataset_dir"])
    split_name = str(cfg["split_name"])
    representation_source = str(cfg.get("representation_source", "prompt_activations"))
    center_representations = bool(cfg.get("center_representations", False))
    progress_logging = bool(cfg.get("progress_logging", False))
    if representation_source != "prompt_activations":
        raise ValueError(
            f"This runner currently supports representation_source='prompt_activations' only, got {representation_source!r}."
        )

    family_specs = list(cfg["prompt_families"])
    family_label_by_key = _family_label_map(family_specs)
    hash_payload = {
        "task_name": "source_value_probe_prompt_style_transfer_relative_layers",
        "config": cfg,
    }
    run_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    output_dir = Path(args.output_root) / f"source_value_probe_prompt_style_transfer_relative_layers__{run_hash}__r0"
    if (output_dir / "DONE").exists() and not args.force:
        print(output_dir)
        report_path = ROOT / "experiment-reports" / f"{cfg['report_stem']}.html"
        print(report_path)
        return

    ensure_dir(output_dir)
    atomic_write_text(output_dir / "command.txt", " ".join(sys.argv) + "\n")
    dump_run_config(
        output_dir / "config.json",
        {
            "args": vars(args),
            "experiment_config": cfg,
        },
    )

    all_sweep_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_layer_summary_rows: list[dict[str, Any]] = []
    all_alignment_rows: list[dict[str, Any]] = []

    relative_layers = [float(value) for value in cfg["relative_layers"]]
    for model_alias in cfg["models"]:
        model_alias = str(model_alias)
        model_entry = {**models_cfg.get("default", {}), **models_cfg["models"][model_alias]}
        model_name = str(model_entry["hf_id"])
        device = str(cfg.get("device") or model_entry.get("device", "cpu"))
        dtype = str(cfg.get("dtype") or model_entry.get("dtype", "float32"))
        probe_batch_size = int(cfg["probe_batch_size_by_model"][model_alias])
        model_label = str(cfg.get("model_labels", {}).get(model_alias, model_alias))
        trust_remote_code = bool(model_entry.get("trust_remote_code", False))

        model_dir = ensure_dir(output_dir / model_alias)
        rows_by_family = {
            str(item["family"]): _load_partition_rows(split_dataset_dir, split_name, str(item["family"]))
            for item in family_specs
        }
        alignment_rows = _assert_source_value_alignment(
            rows_by_family=rows_by_family,
            family_label_by_key=family_label_by_key,
        )
        for row in alignment_rows:
            row["model_alias"] = model_alias
            row["model_label"] = model_label
        all_alignment_rows.extend(alignment_rows)
        atomic_write_csv(model_dir / "source_value_alignment.csv", pd.DataFrame(alignment_rows))

        n_layers = _resolve_n_layers(model_name=model_name, trust_remote_code=trust_remote_code)
        loaded = load_causal_lm(
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=model_entry.get("attn_implementation"),
        )

        for relative_layer in relative_layers:
            layer_idx = _layer_rel_to_abs(float(relative_layer), n_layers)
            layer_dir = ensure_dir(model_dir / f"layer_rel_{relative_layer:.1f}".replace(".", "_"))
            progress_log_path = layer_dir / "progress_log.jsonl"
            if (layer_dir / "DONE").exists() and not args.force:
                all_sweep_rows.extend(_load_jsonl_if_exists(layer_dir / "sweep_summary.jsonl"))
                all_selected_rows.extend(_load_jsonl_if_exists(layer_dir / "selected_transfer_summary.jsonl"))
                all_layer_summary_rows.extend(_load_jsonl_if_exists(layer_dir / "layer_summary.jsonl"))
                continue

            activations_by_family: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
            for family_spec in family_specs:
                family = str(family_spec["family"])
                collect_started = time.perf_counter()
                if progress_logging:
                    _append_jsonl(
                        progress_log_path,
                        {
                            "event": "collect_partition_activations_start",
                            "model_alias": model_alias,
                            "relative_layer": float(relative_layer),
                            "layer_idx": int(layer_idx),
                            "family": family,
                            "timestamp": time.time(),
                        },
                    )
                activations_by_family[family] = _collect_partition_activations(
                    rows_by_partition=rows_by_family[family],
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    layer_idx=layer_idx,
                    batch_size=probe_batch_size,
                    device=device,
                )
                collect_elapsed = time.perf_counter() - collect_started
                if progress_logging:
                    _append_jsonl(
                        progress_log_path,
                        {
                            "event": "collect_partition_activations_done",
                            "model_alias": model_alias,
                            "relative_layer": float(relative_layer),
                            "layer_idx": int(layer_idx),
                            "family": family,
                            "elapsed_sec": float(collect_elapsed),
                            "timestamp": time.time(),
                        },
                    )
            centering_rows: list[dict[str, Any]] = []
            if center_representations:
                centering_started = time.perf_counter()
                if progress_logging:
                    _append_jsonl(
                        progress_log_path,
                        {
                            "event": "centering_start",
                            "model_alias": model_alias,
                            "relative_layer": float(relative_layer),
                            "layer_idx": int(layer_idx),
                            "timestamp": time.time(),
                        },
                    )
                activations_by_family, centering_rows = _center_activations_by_family(activations_by_family)
                if progress_logging:
                    _append_jsonl(
                        progress_log_path,
                        {
                            "event": "centering_done",
                            "model_alias": model_alias,
                            "relative_layer": float(relative_layer),
                            "layer_idx": int(layer_idx),
                            "elapsed_sec": float(time.perf_counter() - centering_started),
                            "timestamp": time.time(),
                        },
                    )

            layer_sweep_rows: list[dict[str, Any]] = []
            layer_selected_rows: list[dict[str, Any]] = []
            for train_family_spec in family_specs:
                train_family = str(train_family_spec["family"])
                train_family_label = str(train_family_spec["family_label"])
                train_values, train_x = activations_by_family[train_family]["train"]
                sweeps = _architecture_sweeps(train_x)
                model_dim = int(train_x.shape[1])

                for architecture, sweep_rows in sweeps:
                    architecture_rows: list[dict[str, Any]] = []
                    for sweep_row in sweep_rows:
                        fit_started = time.perf_counter()
                        if progress_logging:
                            _append_jsonl(
                                progress_log_path,
                                {
                                    "event": "fit_probe_start",
                                    "model_alias": model_alias,
                                    "relative_layer": float(relative_layer),
                                    "layer_idx": int(layer_idx),
                                    "train_family": train_family,
                                    "architecture": architecture,
                                    "step_index": int(sweep_row["step_index"]),
                                    "control_name": str(sweep_row["control_name"]),
                                    "control_value": sweep_row["control_value"],
                                    "effective_df": float(sweep_row["effective_df"]),
                                    "num_train_examples": int(train_x.shape[0]),
                                    "feature_dim": int(train_x.shape[1]),
                                    "timestamp": time.time(),
                                },
                            )
                        probe = fit_magnitude_probe(
                            architecture=architecture,
                            train_x=train_x,
                            train_values=train_values,
                            config=sweep_row["fit_config"],
                            value_min=int(cfg.get("value_min", 0)),
                            value_max=int(cfg.get("value_max", 999)),
                        )
                        if progress_logging:
                            _append_jsonl(
                                progress_log_path,
                                {
                                    "event": "fit_probe_done",
                                    "model_alias": model_alias,
                                    "relative_layer": float(relative_layer),
                                    "layer_idx": int(layer_idx),
                                    "train_family": train_family,
                                    "architecture": architecture,
                                    "step_index": int(sweep_row["step_index"]),
                                    "control_name": str(sweep_row["control_name"]),
                                    "control_value": sweep_row["control_value"],
                                    "effective_df": float(sweep_row["effective_df"]),
                                    "elapsed_sec": float(time.perf_counter() - fit_started),
                                    "timestamp": time.time(),
                                },
                            )
                        dev_values, dev_x = activations_by_family[train_family]["dev"]
                        native_dev_summary = _evaluate_probe(
                            probe=probe,
                            values=dev_values,
                            activations=dev_x,
                            split_name="dev",
                        )
                        native_dev_summary.update(
                            {
                                "model_alias": model_alias,
                                "model_label": model_label,
                                "train_family": train_family,
                                "train_family_label": train_family_label,
                                "test_family": train_family,
                                "test_family_label": train_family_label,
                                "evaluation_partition": "dev",
                                "is_native_style": True,
                                "architecture": architecture,
                                "step_index": int(sweep_row["step_index"]),
                                "control_name": str(sweep_row["control_name"]),
                                "control_value": sweep_row["control_value"],
                                "effective_df": float(sweep_row["effective_df"]),
                                "relative_layer": float(relative_layer),
                                "layer_idx": int(layer_idx),
                                "n_layers": int(n_layers),
                                "split_name": split_name,
                                "model_dim": int(model_dim),
                            }
                        )
                        architecture_rows.append(native_dev_summary)

                        for test_family_spec in family_specs:
                            test_family = str(test_family_spec["family"])
                            test_family_label = str(test_family_spec["family_label"])
                            test_values, test_x = activations_by_family[test_family]["test"]
                            test_summary = _evaluate_probe(
                                probe=probe,
                                values=test_values,
                                activations=test_x,
                                split_name="test",
                            )
                            test_summary.update(
                                {
                                    "model_alias": model_alias,
                                    "model_label": model_label,
                                    "train_family": train_family,
                                    "train_family_label": train_family_label,
                                    "test_family": test_family,
                                    "test_family_label": test_family_label,
                                    "evaluation_partition": "test",
                                    "is_native_style": bool(test_family == train_family),
                                    "architecture": architecture,
                                    "step_index": int(sweep_row["step_index"]),
                                    "control_name": str(sweep_row["control_name"]),
                                    "control_value": sweep_row["control_value"],
                                    "effective_df": float(sweep_row["effective_df"]),
                                    "relative_layer": float(relative_layer),
                                    "layer_idx": int(layer_idx),
                                    "n_layers": int(n_layers),
                                    "split_name": split_name,
                                    "model_dim": int(model_dim),
                                }
                            )
                            architecture_rows.append(test_summary)

                    architecture_df = pd.DataFrame(architecture_rows)
                    native_dev_df = architecture_df.loc[architecture_df["evaluation_partition"] == "dev"].copy()
                    if native_dev_df.empty:
                        raise ValueError(
                            f"Missing native dev rows for model={model_alias}, rel_layer={relative_layer}, "
                            f"train_family={train_family}, architecture={architecture}."
                        )
                    selected = _score_sort_frame(native_dev_df).iloc[0]
                    selected_step_index = int(selected["step_index"])
                    selected_rows = architecture_df.loc[architecture_df["step_index"] == selected_step_index].copy()
                    selected_rows["selected_by_native_dev"] = True
                    selected_rows["selected_dev_mean_abs_error"] = float(selected["mean_abs_error"])
                    selected_rows["selected_dev_rmse"] = float(selected["rmse"])
                    selected_rows["selected_dev_accuracy"] = float(selected["accuracy"])
                    selected_rows["selected_dev_r2"] = float(selected["r2"])
                    layer_sweep_rows.extend(architecture_rows)
                    layer_selected_rows.extend(selected_rows.to_dict("records"))

            selected_test_df = pd.DataFrame(layer_selected_rows).loc[
                pd.DataFrame(layer_selected_rows)["evaluation_partition"] == "test"
            ].copy()
            layer_summary_rows: list[dict[str, Any]] = []
            if not selected_test_df.empty:
                for architecture in PROBE_ARCH_ORDER:
                    arch_df = selected_test_df.loc[selected_test_df["architecture"] == architecture].copy()
                    if arch_df.empty:
                        continue
                    diagonal_mask = arch_df["train_family"] == arch_df["test_family"]
                    layer_summary_rows.append(
                        {
                            "model_alias": model_alias,
                            "model_label": model_label,
                            "architecture": architecture,
                            "relative_layer": float(relative_layer),
                            "layer_idx": int(layer_idx),
                            "n_layers": int(n_layers),
                            "matrix_mean_mae": float(arch_df["mean_abs_error"].mean()),
                            "diagonal_mean_mae": float(arch_df.loc[diagonal_mask, "mean_abs_error"].mean()),
                            "off_diagonal_mean_mae": float(arch_df.loc[~diagonal_mask, "mean_abs_error"].mean()),
                            "num_cells": int(len(arch_df)),
                        }
                    )

            atomic_write_jsonl(layer_dir / "sweep_summary.jsonl", layer_sweep_rows)
            atomic_write_jsonl(layer_dir / "selected_transfer_summary.jsonl", layer_selected_rows)
            atomic_write_jsonl(layer_dir / "layer_summary.jsonl", layer_summary_rows)
            if center_representations:
                for row in centering_rows:
                    row.update(
                        {
                            "model_alias": model_alias,
                            "model_label": model_label,
                            "relative_layer": float(relative_layer),
                            "layer_idx": int(layer_idx),
                        }
                    )
                atomic_write_jsonl(layer_dir / "centering_summary.jsonl", centering_rows)
            atomic_write_json(
                layer_dir / "resolved.json",
                {
                    "model_alias": model_alias,
                    "model_name": model_name,
                    "model_label": model_label,
                    "relative_layer": float(relative_layer),
                    "layer_idx": int(layer_idx),
                    "n_layers": int(n_layers),
                    "split_name": split_name,
                    "probe_batch_size": int(probe_batch_size),
                    "center_representations": center_representations,
                    "progress_logging": progress_logging,
                },
            )
            atomic_write_text(layer_dir / "DONE", "ok\n")
            all_sweep_rows.extend(layer_sweep_rows)
            all_selected_rows.extend(layer_selected_rows)
            all_layer_summary_rows.extend(layer_summary_rows)

        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sweep_df = pd.DataFrame(all_sweep_rows)
    selected_df = pd.DataFrame(all_selected_rows)
    layer_summary_df = pd.DataFrame(all_layer_summary_rows)
    alignment_df = pd.DataFrame(all_alignment_rows)
    if sweep_df.empty or selected_df.empty or layer_summary_df.empty:
        raise ValueError("The cross-style transfer run produced no summary rows.")

    section_rows: list[dict[str, Any]] = []
    for (model_alias, architecture), group in layer_summary_df.groupby(["model_alias", "architecture"], as_index=False):
        ordered = group.sort_values(["matrix_mean_mae", "relative_layer"], ascending=[True, True]).reset_index(drop=True)
        best = ordered.iloc[0]
        worst = group.sort_values(["matrix_mean_mae", "relative_layer"], ascending=[False, True]).reset_index(drop=True).iloc[0]
        section_rows.append(
            {
                "model_alias": model_alias,
                "model_label": str(best["model_label"]),
                "architecture": architecture,
                "best_relative_layer": float(best["relative_layer"]),
                "best_layer_idx": int(best["layer_idx"]),
                "best_matrix_mean_mae": float(best["matrix_mean_mae"]),
                "worst_relative_layer": float(worst["relative_layer"]),
                "worst_layer_idx": int(worst["layer_idx"]),
                "worst_matrix_mean_mae": float(worst["matrix_mean_mae"]),
                "mean_over_layers_matrix_mean_mae": float(group["matrix_mean_mae"].mean()),
            }
        )
    section_summary_df = pd.DataFrame(section_rows)

    atomic_write_jsonl(output_dir / "sweep_summary.jsonl", sweep_df.to_dict("records"))
    atomic_write_jsonl(output_dir / "selected_transfer_summary.jsonl", selected_df.to_dict("records"))
    atomic_write_jsonl(output_dir / "layer_matrix_summary.jsonl", layer_summary_df.to_dict("records"))
    atomic_write_jsonl(output_dir / "section_summary.jsonl", section_summary_df.to_dict("records"))
    atomic_write_csv(output_dir / "sweep_summary.csv", sweep_df)
    atomic_write_csv(output_dir / "selected_transfer_summary.csv", selected_df)
    atomic_write_csv(output_dir / "layer_matrix_summary.csv", layer_summary_df)
    atomic_write_csv(output_dir / "section_summary.csv", section_summary_df)
    atomic_write_csv(output_dir / "source_value_alignment.csv", alignment_df)

    report_stem = str(cfg["report_stem"])
    report_path = ROOT / "experiment-reports" / f"{report_stem}.html"
    artifact_dir = ROOT / "experiment-reports" / "artifacts" / report_stem
    _write_report(
        report_path=report_path,
        artifact_dir=artifact_dir,
        selected_df=selected_df,
        layer_summary_df=layer_summary_df,
        section_summary_df=section_summary_df,
        metadata={
            "report_title": f"Source-Value Probe Cross-Prompt-Style Transfer | {split_name}",
            "split_name": split_name,
            "relative_layers": relative_layers,
            "models": [str(model) for model in cfg["models"]],
            "architectures": PROBE_ARCH_ORDER,
            "prompt_families": FAMILY_ORDER,
            "center_representations": center_representations,
            "progress_logging": progress_logging,
        },
    )
    atomic_write_text(output_dir / "DONE", "ok\n")
    print(output_dir)
    print(report_path)


if __name__ == "__main__":
    main()
