#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activation_cache import (
    PromptSequenceActivationCache,
    build_prompt_sequence_cache_spec,
    default_prompt_sequence_cache_root,
)
from src.activation_steering import collect_prompt_activations, collect_prompt_hidden_sequences
from src.magnitude_intervention import evaluate_probe_split, fit_magnitude_probe
from src.model_loader import load_causal_lm, load_tokenizer, resolve_model_revision
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


DEFAULT_CONFIG = ROOT / "configs" / "source_value_probe_capacity_sweep_addition_formula_split00_llama_3_2_3b.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a source-value probe capacity sweep on canonical prompt-cache splits.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="out/runs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _power_of_two_ladder(limit: int) -> list[int]:
    if int(limit) < 1:
        raise ValueError(f"limit must be positive, got {limit!r}")
    values: list[int] = []
    current = 1
    while current < int(limit):
        values.append(int(current))
        current *= 2
    if not values or values[-1] != int(limit):
        values.append(int(limit))
    return values


def _ridge_effective_df(squared_singular_values: np.ndarray, ridge_lambda: float) -> float:
    lam = float(ridge_lambda)
    if lam < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda!r}")
    if lam == 0.0:
        return float(len(squared_singular_values))
    return float(np.sum(squared_singular_values / (squared_singular_values + lam)))


def _solve_lambda_for_target_df(squared_singular_values: np.ndarray, target_df: float) -> float:
    full_rank = float(len(squared_singular_values))
    if target_df >= full_rank:
        return 0.0
    if target_df <= 0.0:
        return float(np.max(squared_singular_values) * 1e12)

    lo = 0.0
    hi = float(np.max(squared_singular_values)) if len(squared_singular_values) else 1.0
    while _ridge_effective_df(squared_singular_values, hi) > target_df:
        hi *= 2.0
        if hi > float(np.max(squared_singular_values) * 1e18):
            break
    for _ in range(96):
        mid = 0.5 * (lo + hi)
        df_mid = _ridge_effective_df(squared_singular_values, mid)
        if df_mid > target_df:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _build_ridge_schedule(train_x: np.ndarray, num_steps: int) -> list[dict[str, float]]:
    centered = np.asarray(train_x, dtype=np.float64) - np.asarray(train_x, dtype=np.float64).mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    squared = np.asarray((singular_values[singular_values > 1e-12]) ** 2, dtype=np.float64)
    full_rank = float(len(squared))
    rows: list[dict[str, float]] = []
    for step_idx in range(int(num_steps)):
        if step_idx == 0:
            ridge_lambda = 0.0
            target_df = full_rank
        else:
            target_df = full_rank / float(2**step_idx)
            ridge_lambda = _solve_lambda_for_target_df(squared, target_df)
        rows.append(
            {
                "step_index": int(step_idx),
                "ridge_lambda": float(ridge_lambda),
                "target_effective_df": float(target_df),
                "effective_df": float(_ridge_effective_df(squared, ridge_lambda)),
            }
        )
    return rows


def _load_partition_rows(dataset_dir: Path, split_name: str, family: str) -> dict[str, pd.DataFrame]:
    rows_by_partition: dict[str, pd.DataFrame] = {}
    for partition in ("train", "dev", "test"):
        path = dataset_dir / split_name / family / f"{partition}.jsonl"
        frame = pd.DataFrame(load_jsonl(path))
        frame["partition"] = partition
        sort_columns = [column for column in ("source_value", "variant_idx", "operand_mode", "context_idx", "prompt") if column in frame.columns]
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
        rows_by_partition[partition] = frame
    return rows_by_partition


def _collect_partition_activations(
    *,
    rows_by_partition: dict[str, pd.DataFrame],
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int,
    device: str,
    model_alias: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    attn_implementation: str | None = None,
    family: str | None = None,
    activation_cache_root: str | Path | None = None,
    use_activation_cache: bool = True,
    require_activation_cache_hit: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not use_activation_cache:
        activations_by_partition: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for partition, frame in rows_by_partition.items():
            prompts = frame["prompt"].astype(str).tolist()
            token_idxs = frame["token_idx"].astype(int).tolist()
            activations = collect_prompt_activations(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                token_idxs=token_idxs,
                layer_idx=int(layer_idx),
                batch_size=int(batch_size),
                device=device,
            )
            values = frame["source_value"].to_numpy(dtype=np.int64)
            activations_by_partition[partition] = (values, activations.astype(np.float32))
        return activations_by_partition

    if model_alias is None or model_id is None or model_revision is None or family is None:
        raise ValueError("Activation caching requires model_alias, model_id, model_revision, and family.")

    cache_root = Path(activation_cache_root) if activation_cache_root is not None else default_prompt_sequence_cache_root()
    cache = PromptSequenceActivationCache(
        build_prompt_sequence_cache_spec(
            cache_root=cache_root,
            model_alias=model_alias,
            model_id=model_id,
            model_revision=model_revision,
            layer_idx=layer_idx,
            attn_implementation=attn_implementation,
            family=family,
        )
    )

    requests_by_partition: dict[str, list[dict[str, Any]]] = {}
    entries_by_key: dict[str, dict[str, Any]] = {}
    for partition, frame in rows_by_partition.items():
        prompts = frame["prompt"].astype(str).tolist()
        prepared_entries = cache.prepare_entries(prompts=prompts, tokenizer=tokenizer, layer_idx=int(layer_idx))
        partition_requests: list[dict[str, Any]] = []
        for prepared_entry, token_idx, source_value in zip(
            prepared_entries,
            frame["token_idx"].astype(int).tolist(),
            frame["source_value"].astype(int).tolist(),
            strict=False,
        ):
            cache_key = str(prepared_entry["cache_key"])
            entries_by_key.setdefault(cache_key, prepared_entry)
            partition_requests.append(
                {
                    "cache_key": cache_key,
                    "token_idx": int(token_idx),
                    "source_value": int(source_value),
                }
            )
        requests_by_partition[partition] = partition_requests

    missing_entries = cache.get_missing_entries(list(entries_by_key.values()))
    if missing_entries and require_activation_cache_hit:
        sample_keys = [str(entry["cache_key"]) for entry in missing_entries[:3]]
        raise ValueError(
            f"Activation cache miss for shard {cache.spec.shard_id}: missing {len(missing_entries)} entries. "
            f"Sample cache keys: {sample_keys!r}"
        )
    if missing_entries:
        if model is None:
            raise ValueError("Cannot populate missing activation cache entries without a loaded model.")
        missing_prompts = [str(entry["prompt"]) for entry in missing_entries]
        missing_sequences = collect_prompt_hidden_sequences(
            model=model,
            tokenizer=tokenizer,
            prompts=missing_prompts,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
        )
        cache.store_missing_sequences(entries=missing_entries, sequences=missing_sequences)

    activations_by_partition: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for partition, partition_requests in requests_by_partition.items():
        values = np.asarray([int(item["source_value"]) for item in partition_requests], dtype=np.int64)
        row_activations: list[np.ndarray] = []
        for request in partition_requests:
            sequence = cache.load_sequence(str(request["cache_key"]))
            token_idx = int(request["token_idx"])
            if token_idx < 0 or token_idx >= int(sequence.shape[0]):
                raise ValueError(
                    f"Cached sequence token_idx out of range: token_idx={token_idx}, seq_len={sequence.shape[0]}, cache_key={request['cache_key']}"
                )
            row_activations.append(np.asarray(sequence[token_idx], dtype=np.float32))
        activations_by_partition[partition] = (values, np.stack(row_activations, axis=0).astype(np.float32))
    return activations_by_partition


def _r2_score(truth: np.ndarray, pred: np.ndarray) -> float:
    y_true = np.asarray(truth, dtype=np.float64)
    y_pred = np.asarray(pred, dtype=np.float64)
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))


def _spearman(truth: np.ndarray, pred: np.ndarray) -> float:
    series_true = pd.Series(np.asarray(truth, dtype=np.float64))
    series_pred = pd.Series(np.asarray(pred, dtype=np.float64))
    value = series_true.corr(series_pred, method="spearman")
    return float(value) if value is not None else float("nan")


def _augment_summary(detail_df: pd.DataFrame, base_summary: dict[str, Any]) -> dict[str, Any]:
    truth = detail_df["value"].to_numpy(dtype=np.int64)
    pred = detail_df["prediction"].to_numpy(dtype=np.int64)
    augmented = dict(base_summary)
    augmented["r2"] = _r2_score(truth, pred)
    augmented["spearman"] = _spearman(truth, pred)
    if "raw_prediction" in detail_df.columns and detail_df["raw_prediction"].notna().any():
        raw_pred = detail_df["raw_prediction"].to_numpy(dtype=np.float64)
        augmented["raw_r2"] = _r2_score(truth, raw_pred)
        augmented["raw_spearman"] = _spearman(truth, raw_pred)
    else:
        augmented["raw_r2"] = float("nan")
        augmented["raw_spearman"] = float("nan")
    return augmented


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _make_accuracy_figure(summary_df: pd.DataFrame) -> str:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Dev Exact Accuracy", "Test Exact Accuracy"),
        shared_yaxes=True,
    )
    color_map = {"linear": "#1f77b4", "circular_reg": "#d62728", "circular_cls": "#b54d4d", "pls": "#2ca02c"}
    for col_idx, partition in enumerate(("dev", "test"), start=1):
        subset = summary_df.loc[summary_df["partition"] == partition].copy()
        for architecture in ("linear", "circular_reg", "circular_cls", "pls"):
            arch = subset.loc[subset["architecture"] == architecture].copy().sort_values("step_index")
            if arch.empty:
                continue
            hover = np.stack(
                [
                    arch["control_value"].astype(str),
                    arch["effective_df"].map(lambda value: _format_float(value, 2)),
                    arch["mean_abs_error"].map(lambda value: _format_float(value, 2)),
                    arch["r2"].map(lambda value: _format_float(value, 4)),
                ],
                axis=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=arch["step_index"],
                    y=arch["accuracy"],
                    mode="lines+markers",
                    name=architecture,
                    legendgroup=architecture,
                    line=dict(color=color_map.get(architecture)),
                    marker=dict(size=8),
                    customdata=hover,
                    hovertemplate=(
                        "architecture=%{fullData.name}<br>"
                        "step=%{x}<br>"
                        "control=%{customdata[0]}<br>"
                        "effective_df=%{customdata[1]}<br>"
                        "accuracy=%{y:.4f}<br>"
                        "mean_abs_error=%{customdata[2]}<br>"
                        "r2=%{customdata[3]}<extra></extra>"
                    ),
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text="Capacity step", row=1, col=col_idx)
    fig.update_yaxes(title_text="Exact accuracy", row=1, col=1)
    fig.update_layout(height=420, width=980, margin=dict(l=40, r=20, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _make_r2_figure(summary_df: pd.DataFrame) -> str:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Dev R²", "Test R²"),
        shared_yaxes=True,
    )
    color_map = {"linear": "#1f77b4", "circular_reg": "#d62728", "circular_cls": "#b54d4d", "pls": "#2ca02c"}
    for col_idx, partition in enumerate(("dev", "test"), start=1):
        subset = summary_df.loc[summary_df["partition"] == partition].copy()
        for architecture in ("linear", "circular_reg", "circular_cls", "pls"):
            arch = subset.loc[subset["architecture"] == architecture].copy().sort_values("step_index")
            if arch.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=arch["step_index"],
                    y=arch["r2"],
                    mode="lines+markers",
                    name=architecture,
                    legendgroup=architecture,
                    line=dict(color=color_map.get(architecture)),
                    marker=dict(size=8),
                    hovertemplate="architecture=%{fullData.name}<br>step=%{x}<br>r2=%{y:.4f}<extra></extra>",
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text="Capacity step", row=1, col=col_idx)
    fig.update_yaxes(title_text="R² on decoded values", row=1, col=1)
    fig.update_layout(height=420, width=980, margin=dict(l=40, r=20, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _render_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame.loc[:, columns].copy()
    return subset.to_html(index=False, escape=False)


def _write_report(
    *,
    report_path: Path,
    artifact_dir: Path,
    summary_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    ensure_dir(report_path.parent)
    ensure_dir(artifact_dir)
    best_dev = (
        summary_df.loc[summary_df["partition"] == "dev"]
        .sort_values(["architecture", "accuracy", "r2", "mean_abs_error"], ascending=[True, False, False, True])
        .groupby("architecture", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_test = (
        summary_df.loc[summary_df["partition"] == "test"]
        .sort_values(["architecture", "accuracy", "r2", "mean_abs_error"], ascending=[True, False, False, True])
        .groupby("architecture", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    atomic_write_jsonl(artifact_dir / "summary_metrics.jsonl", summary_df.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_dev.jsonl", best_dev.to_dict("records"))
    atomic_write_jsonl(artifact_dir / "best_test.jsonl", best_test.to_dict("records"))
    atomic_write_json(artifact_dir / "metadata.json", metadata)

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
    .meta {{ margin: 18px 0 26px 0; }}
    .caption {{ color: #444; margin-top: -8px; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>{metadata['report_title']}</h1>
  <div class="meta">
    <p>
      Model: <code>{metadata['model_alias']}</code> |
      Family: <code>{metadata['family']}</code> |
      Split: <code>{metadata['split_name']}</code> |
      Layer: <code>{metadata['probe_layer_idx']}</code> |
      Representation: <code>{metadata['representation_source']}</code>
    </p>
    <p>
      Circular sweep: <code>{metadata['circular_steps']}</code><br/>
      PLS sweep: <code>{metadata['pls_steps']}</code><br/>
      Linear ridge: <code>{metadata['linear_num_steps']}</code> steps with effective-df targets halving from the full train rank.
    </p>
    <p>
      Note: PLS cannot exceed the train predictor rank. With <code>n_train={metadata['train_size']}</code>, the maximum fitted PLS component count here is <code>{metadata['pls_max_components']}</code>.
    </p>
  </div>

  <h2>Best Dev Settings</h2>
  {_render_table(best_dev, ['architecture', 'control_value', 'effective_df', 'accuracy', 'mean_abs_error', 'rmse', 'r2', 'spearman'])}

  <h2>Best Test Settings</h2>
  {_render_table(best_test, ['architecture', 'control_value', 'effective_df', 'accuracy', 'mean_abs_error', 'rmse', 'r2', 'spearman'])}

  <h2>Accuracy Curves</h2>
  {_make_accuracy_figure(summary_df)}
  <div class="caption">Each point is one fitted probe. For linear ridge, the hover text shows both <code>lambda</code> and the resulting effective degrees of freedom.</div>

  <h2>R² Curves</h2>
  {_make_r2_figure(summary_df)}
  <div class="caption">R² is computed on decoded integer predictions against the true <code>source_value</code>.</div>

  <h2>Full Sweep Summary</h2>
  {_render_table(summary_df.sort_values(['architecture', 'partition', 'step_index']).reset_index(drop=True), ['architecture', 'partition', 'step_index', 'control_value', 'effective_df', 'accuracy', 'mean_abs_error', 'rmse', 'r2', 'spearman'])}
</body>
</html>
"""
    atomic_write_text(report_path, html)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["seed"]))

    models_cfg = load_yaml(cfg["models_config"])
    model_alias = str(cfg["model"])
    model_entry = {**models_cfg.get("default", {}), **models_cfg["models"][model_alias]}
    model_name = str(model_entry["hf_id"])
    device = str(cfg.get("device") or model_entry.get("device", "cpu"))
    dtype = str(cfg.get("dtype") or model_entry.get("dtype", "float32"))
    split_dataset_dir = Path(cfg["split_dataset_dir"])
    family = str(cfg["family"])
    split_name = str(cfg["split_name"])
    representation_source = str(cfg.get("representation_source", "prompt_activations"))
    probe_layer_idx = int(cfg.get("probe_layer_idx", 0))
    use_activation_cache = bool(cfg.get("use_activation_cache", True))
    require_activation_cache_hit = bool(cfg.get("require_activation_cache_hit", False))
    if representation_source != "prompt_activations":
        raise ValueError(f"This runner currently supports representation_source='prompt_activations' only, got {representation_source!r}.")

    hash_payload = {
        "task_name": "source_value_probe_capacity_sweep",
        "config": cfg,
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
    }
    run_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    output_dir = Path(args.output_root) / f"source_value_probe_capacity_sweep__{run_hash}__r0"
    if (output_dir / "DONE").exists() and not args.force:
        print(output_dir)
        return

    rows_by_partition = _load_partition_rows(split_dataset_dir, split_name, family)
    trust_remote_code = bool(model_entry.get("trust_remote_code", False))
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
    activations_by_partition = _collect_partition_activations(
        rows_by_partition=rows_by_partition,
        model=loaded_model,
        tokenizer=tokenizer,
        layer_idx=probe_layer_idx,
        batch_size=int(cfg.get("probe_batch_size", 64)),
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
    circular_steps = _power_of_two_ladder(model_dim)
    pls_max_components = min(model_dim, int(train_x.shape[0] - 1))
    pls_steps = _power_of_two_ladder(pls_max_components)
    ridge_schedule = _build_ridge_schedule(train_x, num_steps=len(circular_steps))

    ensure_dir(output_dir)
    atomic_write_text(output_dir / "command.txt", " ".join(sys.argv) + "\n")
    dump_run_config(
        output_dir / "config.json",
        {
            "args": vars(args),
            "experiment_config": cfg,
            "model_entry": model_entry,
            "resolved": {
                "model_name": model_name,
                "model_revision": model_revision,
                "device": device,
                "dtype": dtype,
                "representation_source": representation_source,
                "probe_layer_idx": probe_layer_idx,
                "model_dim": model_dim,
                "activation_cache_root": str(cfg.get("activation_cache_root") or default_prompt_sequence_cache_root()),
                "use_activation_cache": use_activation_cache,
                "require_activation_cache_hit": require_activation_cache_hit,
                "circular_steps": circular_steps,
                "pls_steps": pls_steps,
                "ridge_schedule": ridge_schedule,
            },
        },
    )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

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
            probe = fit_magnitude_probe(
                architecture=architecture,
                train_x=train_x,
                train_values=train_values,
                config=sweep_row["fit_config"],
                value_min=int(cfg.get("value_min", 0)),
                value_max=int(cfg.get("value_max", 999)),
            )
            for partition in ("train", "dev", "test"):
                values, split_x = activations_by_partition[partition]
                detail_df, base_summary = evaluate_probe_split(
                    probe=probe,
                    representations=split_x,
                    values=values,
                    split_name=partition,
                )
                detail_df = detail_df.rename(columns={"split": "partition"})
                detail_df["architecture"] = architecture
                detail_df.insert(1, "step_index", int(sweep_row["step_index"]))
                detail_df["control_name"] = str(sweep_row["control_name"])
                detail_df["control_value"] = sweep_row["control_value"]
                detail_df["effective_df"] = float(sweep_row["effective_df"])
                detail_df["family"] = family
                detail_df["split_name"] = split_name
                detail_rows.extend(detail_df.to_dict("records"))

                summary_row = _augment_summary(detail_df, {**base_summary, "partition": partition})
                summary_row.update(
                    {
                        "architecture": architecture,
                        "step_index": int(sweep_row["step_index"]),
                        "control_name": str(sweep_row["control_name"]),
                        "control_value": sweep_row["control_value"],
                        "effective_df": float(sweep_row["effective_df"]),
                        "family": family,
                        "split_name": split_name,
                    }
                )
                summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    detail_path = output_dir / "probe_predictions.jsonl"
    summary_path = output_dir / "probe_summary.jsonl"
    atomic_write_jsonl(detail_path, detail_rows)
    atomic_write_jsonl(summary_path, summary_rows)

    report_stem = str(cfg.get("report_stem", "2026-04-18-source-value-probe-capacity-sweep"))
    report_path = ROOT / "experiment-reports" / f"{report_stem}.html"
    artifact_dir = ROOT / "experiment-reports" / "artifacts" / report_stem
    _write_report(
        report_path=report_path,
        artifact_dir=artifact_dir,
        summary_df=summary_df,
        metadata={
            "report_title": f"Source-Value Probe Capacity Sweep | {cfg['family_label']} | {model_alias} | {split_name}",
            "model_alias": model_alias,
            "family": family,
            "family_label": str(cfg["family_label"]),
            "split_name": split_name,
            "probe_layer_idx": probe_layer_idx,
            "representation_source": representation_source,
            "model_dim": model_dim,
            "circular_steps": circular_steps,
            "pls_steps": pls_steps,
            "pls_max_components": pls_max_components,
            "linear_num_steps": len(ridge_schedule),
            "train_size": int(len(train_values)),
        },
    )
    atomic_write_text(output_dir / "DONE", "ok\n")
    print(output_dir)
    print(report_path)


if __name__ == "__main__":
    main()
