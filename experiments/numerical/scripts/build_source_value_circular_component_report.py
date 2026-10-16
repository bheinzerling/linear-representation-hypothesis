#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_value_probe_capacity_sweep import (
    _collect_partition_activations,
    _load_partition_rows,
    default_prompt_sequence_cache_root,
)
from src.activation_steering import circular_sinusoidal_encode
from src.magnitude_intervention import fit_magnitude_probe, probe_from_payload
from src.model_loader import load_tokenizer, resolve_model_revision
from src.probe_architectures import (
    CIRCULAR_CLS_PROBE_ARCHITECTURE,
    CIRCULAR_REG_PROBE_ARCHITECTURE,
    canonicalize_probe_architecture_name,
    probe_architecture_display_name,
)
from src.utils import ensure_dir, load_yaml


DEFAULT_RUN_DIR = ROOT / "out" / "runs" / "source_value_probe_capacity_sweep_relative_layers__a08a6c06b96d__r0"
DEFAULT_REPORT_STEM = "2026-04-23-llama-3-2-3b-circular-components-basis10"
DEFAULT_ARCHITECTURE = CIRCULAR_REG_PROBE_ARCHITECTURE
MODEL_DISPLAY_NAMES = {
    "gpt_oss_20b": "GPT OSS 20B",
    "llama_3_2_1b": "Llama 3.2 1B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "llama_3_8b": "Llama 3 8B",
    "olmo_2_13b": "OLMo 2 13B",
    "olmo_3_7b_base": "OLMo 3 7B Base",
}
FAMILY_ORDER = [
    "addition_formula",
    "addition_verbalized",
    "subtraction_formula",
    "subtraction_verbalized",
    "identity_patchscopes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-value circular-component visualization report and save the fitted probe weights."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--report-stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument("--model-alias", default="llama_3_2_3b")
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=[CIRCULAR_REG_PROBE_ARCHITECTURE, CIRCULAR_CLS_PROBE_ARCHITECTURE, "circular"],
        help="Circular probe variant to visualize. The legacy 'circular' name maps to circular_reg.",
    )
    parser.add_argument("--split-name", default="split_00")
    parser.add_argument("--basis-dim", type=int, default=10)
    parser.add_argument("--partition", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--family", default=None)
    parser.add_argument("--relative-layer", type=float, default=None)
    parser.add_argument(
        "--classifier-support-rescale",
        default="none",
        choices=[
            "none",
            "w_out_norm",
            "w_out_single_pass",
            "w_in_weighted_over_w_out",
            "effective_input_axis",
            "train_std_match",
        ],
        help=(
            "Visualization-only rescaling for circular_cls support curves. "
            "The example coordinates and saved probe weights are unchanged."
        ),
    )
    parser.add_argument(
        "--reuse-probe-artifact",
        action="store_true",
        help="Load the existing report artifact selected_probe.json instead of refitting the probe.",
    )
    return parser.parse_args()


def _family_rank(family: str) -> tuple[int, str]:
    if family in FAMILY_ORDER:
        return (FAMILY_ORDER.index(family), family)
    return (len(FAMILY_ORDER), family)


def _relative_layer_slug(relative_layer: float) -> str:
    return f"layer_rel_{float(relative_layer):.1f}".replace(".", "_")


def _model_display_name(model_alias: str) -> str:
    if model_alias in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_alias]
    return str(model_alias).replace("_", " ")


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = float(np.sum((truth - truth.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - (np.sum((truth - pred) ** 2) / denom))


def _fit_pca(train_x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_matrix = np.asarray(train_x, dtype=np.float64)
    if train_matrix.ndim != 2:
        raise ValueError(f"Expected 2D train_x for PCA, got shape={train_matrix.shape}.")
    feature_dim = int(train_matrix.shape[1])
    component_count = min(int(n_components), feature_dim)
    train_mean = train_matrix.mean(axis=0, dtype=np.float64)
    centered_train = train_matrix - train_mean
    _, singular_values, vh = np.linalg.svd(centered_train, full_matrices=False)
    components = vh[:component_count].astype(np.float64, copy=False)
    explained_variance = ((singular_values[:component_count] ** 2) / max(train_matrix.shape[0] - 1, 1)).astype(
        np.float64, copy=False
    )
    return train_mean.astype(np.float64, copy=False), components, explained_variance


def _apply_pca(x: np.ndarray, *, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D activations for PCA projection, got shape={matrix.shape}.")
    return (matrix - mean) @ components.T


def _fig_to_div(fig: go.Figure, *, include_plotlyjs: str | bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _component_reorder_indices(basis_dim: int) -> list[int]:
    pair_starts = list(range(0, int(basis_dim), 2))
    reordered: list[int] = []
    for start in reversed(pair_starts):
        reordered.append(start)
        if start + 1 < int(basis_dim):
            reordered.append(start + 1)
    return reordered


def _component_labels(basis_dim: int) -> list[str]:
    return [f"component_{idx + 1:02d}" for idx in range(int(basis_dim))]


def _reorder_component_frame(frame: pd.DataFrame, basis_dim: int) -> pd.DataFrame:
    reorder_indices = _component_reorder_indices(int(basis_dim))
    reordered = frame.copy()
    original_labels = _component_labels(int(basis_dim))
    renamed_columns: dict[str, str] = {}
    for new_idx, old_idx in enumerate(reorder_indices):
        old_label = original_labels[old_idx]
        new_label = original_labels[new_idx]
        renamed_columns[old_label] = new_label
        support_old_label = f"support_{old_label}"
        if support_old_label in reordered.columns:
            renamed_columns[support_old_label] = f"support_{new_label}"
    return reordered.rename(columns=renamed_columns)


def _render_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p><em>No rows.</em></p>"

    def _format_cell(value: Any) -> str:
        if isinstance(value, float):
            return _fmt(value, digits=4)
        return str(value)

    return frame.to_html(
        index=False,
        classes="dataframe compact",
        border=0,
        escape=False,
        formatters={column: _format_cell for column in frame.columns},
    )


def _load_run_context(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary_df = pd.read_json(run_dir / "summary_metrics.jsonl", lines=True)
    summary_df["architecture"] = summary_df["architecture"].map(canonicalize_probe_architecture_name)
    return config_payload, summary_df


def _projection_terms(architecture: str) -> dict[str, str]:
    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        return {
            "short": "circular classifier",
            "hyphen": "circular-classifier",
            "title": "Circular Classifier",
            "probe_label": probe_architecture_display_name(architecture),
            "mapping": (
                "classifier latent coordinates projected back into the original circular codebook "
                "through the learned basis-to-latent map"
            ),
            "support_reference": (
                "the learned classifier-projected support trajectory induced by the fixed sinusoidal codebook "
                "over values <code>0..999</code>"
            ),
            "support_trace_label": "Classifier-projected support curve",
            "support_path_label": "classifier-projected support path",
            "component_summary_reference": "classifier-projected support coordinate",
            "support_component_hover_label": "classifier-projected support component",
            "support_pair_hover_label": "classifier-projected support",
            "distinction": (
                "This is the repo's classifier-style circular probe (<code>circular_cls</code>), not the older "
                "regression-style circular probe (<code>circular_reg</code>)."
            ),
            "legacy_note": "",
        }
    return {
        "short": "circular regression",
        "hyphen": "circular-regression",
        "title": "Circular Regression",
        "probe_label": probe_architecture_display_name(architecture),
        "mapping": "the learned affine regression from LM representation space into the circular feature space",
        "support_reference": "the fixed sinusoidal support code over values <code>0..999</code>",
        "support_trace_label": "Ideal support curve",
        "support_path_label": "ideal support path",
        "component_summary_reference": "ideal sinusoidal support coordinate",
        "support_component_hover_label": "ideal component",
        "support_pair_hover_label": "ideal support",
        "distinction": (
            "This is the repo's regression-style circular probe (<code>circular_reg</code>), not the newer "
            "classifier-style circular probe (<code>circular_cls</code>)."
        ),
        "legacy_note": (
            "Older source rows in this sweep used the legacy label <code>circular</code>; this report canonicalizes "
            "that label to <code>circular_reg</code> to distinguish it from <code>circular_cls</code>."
        ),
    }


def _select_cell(
    summary_df: pd.DataFrame,
    *,
    model_alias: str,
    architecture: str,
    split_name: str,
    basis_dim: int,
    partition: str,
    family: str | None,
    relative_layer: float | None,
) -> tuple[pd.Series, pd.DataFrame]:
    candidates = summary_df.loc[
        (summary_df["model_alias"] == model_alias)
        & (summary_df["split_name"] == split_name)
        & (summary_df["architecture"] == architecture)
        & (summary_df["partition"] == partition)
        & (summary_df["control_value"] == basis_dim)
    ].copy()
    if family is not None:
        candidates = candidates.loc[candidates["family"] == family].copy()
    if relative_layer is not None:
        candidates = candidates.loc[np.isclose(candidates["relative_layer"], float(relative_layer))].copy()
    if candidates.empty:
        raise ValueError(
            f"No rows found for model_alias={model_alias!r}, split_name={split_name!r}, "
            f"basis_dim={basis_dim}, partition={partition!r}, family={family!r}, relative_layer={relative_layer!r}."
        )

    candidates["family_sort"] = candidates["family"].map(lambda value: _family_rank(str(value)))
    candidates = candidates.sort_values(
        ["mean_abs_error", "accuracy", "relative_layer", "family_sort"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    selected = candidates.iloc[0].copy()
    candidates = candidates.drop(columns=["family_sort"]).reset_index(drop=True)
    return selected, candidates


def _load_model_entry(models_cfg_path: Path, model_alias: str) -> dict[str, Any]:
    models_cfg = load_yaml(models_cfg_path)
    default_cfg = dict(models_cfg.get("default", {}))
    model_cfg = dict(models_cfg["models"][model_alias])
    return {**default_cfg, **model_cfg}


def _circular_codebook(basis_dim: int) -> np.ndarray:
    return circular_sinusoidal_encode(
        np.arange(0, 1000, dtype=np.int64),
        embedding_dim=int(basis_dim),
        min_value=0,
        max_value=1000,
    ).astype(np.float64, copy=False)


def _classifier_code_score_components(*, probe: Any, basis_dim: int) -> np.ndarray:
    codebook = _circular_codebook(int(basis_dim))
    support_latents = np.asarray(probe.support_features, dtype=np.float64)
    code_to_latent = np.linalg.pinv(codebook) @ support_latents
    return support_latents @ code_to_latent.T


def _classifier_code_to_latent(*, probe: Any, basis_dim: int) -> np.ndarray:
    codebook = _circular_codebook(int(basis_dim))
    support_latents = np.asarray(probe.support_features, dtype=np.float64)
    return np.linalg.pinv(codebook) @ support_latents


def _safe_positive(value: float, *, fallback: float = 1.0) -> float:
    numeric = float(value)
    if math.isfinite(numeric) and numeric > 0.0:
        return numeric
    return float(fallback)


def _classifier_support_rescale_table(
    *,
    probe: Any,
    basis_dim: int,
    projections_by_partition: dict[str, pd.DataFrame],
    raw_support_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    component_names = [f"component_{idx + 1:02d}" for idx in range(int(basis_dim))]
    reorder_indices = _component_reorder_indices(int(basis_dim))
    code_to_latent = _classifier_code_to_latent(probe=probe, basis_dim=int(basis_dim))
    w_out = code_to_latent.T
    w_out_norm = np.linalg.norm(w_out, axis=0)
    coefficients = np.asarray(probe.coefficients, dtype=np.float64)
    effective_input_axes = coefficients @ w_out
    effective_input_axis_norm = np.linalg.norm(effective_input_axes, axis=0)
    w_in_row_norm = np.linalg.norm(coefficients.T, axis=1)
    weighted_w_in_norm: list[float] = []
    for original_component_idx in range(int(basis_dim)):
        weights = np.square(w_out[:, original_component_idx])
        denom = float(weights.sum())
        if denom <= 0.0:
            weighted_w_in_norm.append(float("nan"))
        else:
            weighted_w_in_norm.append(float(np.sqrt(np.sum(weights * np.square(w_in_row_norm)) / denom)))

    train_projection_df = projections_by_partition["train"]
    rows: list[dict[str, Any]] = []
    for component_idx, original_component_idx in enumerate(reorder_indices):
        component_name = component_names[component_idx]
        raw_support_std = float(raw_support_df[component_name].std(ddof=0))
        train_std = float(train_projection_df[component_name].std(ddof=0))
        output_norm = _safe_positive(w_out_norm[original_component_idx])
        weighted_input_norm = _safe_positive(weighted_w_in_norm[original_component_idx])
        effective_norm = _safe_positive(effective_input_axis_norm[original_component_idx])
        if method == "none":
            scale_factor = 1.0
        elif method == "w_out_norm":
            scale_factor = output_norm
        elif method == "w_out_single_pass":
            scale_factor = 1.0 / output_norm
        elif method == "w_in_weighted_over_w_out":
            scale_factor = weighted_input_norm / output_norm
        elif method == "effective_input_axis":
            scale_factor = effective_norm / (output_norm**2)
        elif method == "train_std_match":
            scale_factor = train_std / _safe_positive(raw_support_std)
        else:
            raise ValueError(f"Unsupported classifier support rescale method: {method!r}")
        rows.append(
            {
                "component": int(component_idx + 1),
                "original_component": int(original_component_idx + 1),
                "rescale_method": method,
                "scale_factor": float(scale_factor),
                "train_std": train_std,
                "raw_support_std": raw_support_std,
                "w_out_column_norm": float(w_out_norm[original_component_idx]),
                "w_in_weighted_row_norm": float(weighted_w_in_norm[original_component_idx]),
                "effective_input_axis_norm": float(effective_input_axis_norm[original_component_idx]),
            }
        )
    return pd.DataFrame(rows)


def _apply_support_rescale(
    *,
    support_df: pd.DataFrame,
    projections_by_partition: dict[str, pd.DataFrame],
    scale_table: pd.DataFrame,
    basis_dim: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scaled_support_df = support_df.copy()
    scaled_projections: dict[str, pd.DataFrame] = {}
    scale_by_component = {
        int(row["component"]): float(row["scale_factor"]) for row in scale_table.to_dict(orient="records")
    }
    for component_idx in range(int(basis_dim)):
        component_name = f"component_{component_idx + 1:02d}"
        factor = scale_by_component[component_idx + 1]
        scaled_support_df[component_name] = scaled_support_df[component_name].astype(np.float64) * factor
    for partition_name, projection_df in projections_by_partition.items():
        scaled_df = projection_df.copy()
        for component_idx in range(int(basis_dim)):
            component_name = f"component_{component_idx + 1:02d}"
            support_name = f"support_{component_name}"
            factor = scale_by_component[component_idx + 1]
            scaled_df[support_name] = scaled_df[support_name].astype(np.float64) * factor
        scaled_projections[partition_name] = scaled_df
    return scaled_support_df, scaled_projections


def _project_to_report_components(
    *,
    probe: Any,
    predicted_features: np.ndarray,
    values: np.ndarray,
    basis_dim: int,
    architecture: str,
) -> tuple[np.ndarray, np.ndarray]:
    codebook = _circular_codebook(int(basis_dim))
    value_indices = np.asarray(values, dtype=np.int64).reshape(-1)
    true_components = codebook[value_indices]
    if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
        return np.asarray(predicted_features, dtype=np.float64), true_components
    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        # circular_cls scores labels as x_latent @ (code @ W_out.T).T, so the
        # comparable basis_dim report coordinates are the effective code-score
        # coordinates x_latent @ W_out, not an unconstrained inverse projection.
        support_latents = np.asarray(probe.support_features, dtype=np.float64)
        code_to_latent = np.linalg.pinv(codebook) @ support_latents
        predicted_components = np.asarray(predicted_features, dtype=np.float64) @ code_to_latent.T
        support_components = support_latents @ code_to_latent.T
        return predicted_components, support_components[value_indices]
    raise ValueError(f"Unsupported circular architecture for component report: {architecture}")


def _prepare_projection_payload(
    *,
    run_dir: Path,
    config_payload: dict[str, Any],
    selected: pd.Series,
    architecture: str,
    basis_dim: int,
    partition: str,
    classifier_support_rescale: str,
    probe_artifact_path: Path | None,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    exp_cfg = dict(config_payload["config"]["experiment_config"])
    family = str(selected["family"])
    relative_layer = float(selected["relative_layer"])
    layer_idx = int(selected["layer_idx"])
    split_name = str(selected["split_name"])
    model_alias = str(selected["model_alias"])

    combo_dir = run_dir / model_alias / _relative_layer_slug(relative_layer) / family
    resolved_payload = json.loads((combo_dir / "resolved.json").read_text(encoding="utf-8"))

    dataset_dir = ROOT / str(exp_cfg["split_dataset_dir"])
    rows_by_partition = _load_partition_rows(dataset_dir, split_name, family)

    model_entry = _load_model_entry(ROOT / str(exp_cfg["models_config"]), model_alias)
    model_name = str(model_entry["hf_id"])
    trust_remote_code = bool(model_entry.get("trust_remote_code", False))
    attn_implementation = model_entry.get("attn_implementation")
    model_revision = resolve_model_revision(
        model_name,
        trust_remote_code=trust_remote_code,
        revision=model_entry.get("revision"),
    )
    tokenizer = load_tokenizer(
        model_name,
        trust_remote_code=trust_remote_code,
        revision=model_revision,
    )

    activations_by_partition = _collect_partition_activations(
        rows_by_partition=rows_by_partition,
        model=None,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        batch_size=int(resolved_payload["probe_batch_size"]),
        device="cpu",
        model_alias=model_alias,
        model_id=model_name,
        model_revision=model_revision,
        attn_implementation=None if attn_implementation is None else str(attn_implementation),
        family=family,
        activation_cache_root=Path(str(resolved_payload.get("activation_cache_root") or default_prompt_sequence_cache_root())),
        use_activation_cache=bool(resolved_payload.get("use_activation_cache", True)),
        require_activation_cache_hit=True,
    )

    probe_config: dict[str, Any] = {"basis_dim": int(basis_dim)}
    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        probe_config.update(dict(resolved_payload.get("circular_cls_config", {})))
    if probe_artifact_path is not None:
        probe = probe_from_payload(json.loads(probe_artifact_path.read_text(encoding="utf-8")))
        if canonicalize_probe_architecture_name(str(probe.architecture)) != architecture:
            raise ValueError(
                f"Probe artifact architecture={probe.architecture!r} does not match requested architecture={architecture!r}."
            )
        if int(probe.metadata.get("basis_dim", basis_dim)) != int(basis_dim):
            raise ValueError(f"Probe artifact basis_dim={probe.metadata.get('basis_dim')!r} does not match {basis_dim}.")
    else:
        train_values, train_x = activations_by_partition["train"]
        probe = fit_magnitude_probe(
            architecture=architecture,
            train_x=train_x,
            train_values=train_values,
            config=probe_config,
            value_min=0,
            value_max=999,
        )
    _, train_x = activations_by_partition["train"]
    pca_mean, pca_components, pca_explained_variance = _fit_pca(train_x, int(basis_dim))
    projections_by_partition: dict[str, pd.DataFrame] = {}
    pca_projections_by_partition: dict[str, pd.DataFrame] = {}
    for partition_name, (part_values, part_x) in activations_by_partition.items():
        predicted_features = probe.predict_features(part_x)
        predicted_components, true_components = _project_to_report_components(
            probe=probe,
            predicted_features=predicted_features,
            values=part_values,
            basis_dim=int(basis_dim),
            architecture=architecture,
        )
        predicted_values = probe.predict_values(part_x)
        abs_error = np.abs(predicted_values - part_values)
        projection_df = pd.DataFrame(
            {
                "partition": partition_name,
                "value": part_values.astype(np.int64),
                "prediction": predicted_values.astype(np.int64),
                "abs_error": abs_error.astype(np.int64),
            }
        )
        for component_idx in range(int(basis_dim)):
            label = f"component_{component_idx + 1:02d}"
            projection_df[label] = predicted_components[:, component_idx].astype(np.float64)
            projection_df[f"support_{label}"] = true_components[:, component_idx].astype(np.float64)
        projection_df = projection_df.sort_values("value").reset_index(drop=True)
        projections_by_partition[partition_name] = _reorder_component_frame(projection_df, int(basis_dim))

        pca_coordinates = _apply_pca(part_x, mean=pca_mean, components=pca_components)
        pca_projection_df = pd.DataFrame(
            {
                "partition": partition_name,
                "value": part_values.astype(np.int64),
                "prediction": predicted_values.astype(np.int64),
                "abs_error": abs_error.astype(np.int64),
            }
        )
        for component_idx in range(int(basis_dim)):
            label = f"pc_{component_idx + 1:02d}"
            pca_projection_df[label] = pca_coordinates[:, component_idx].astype(np.float64)
        pca_projections_by_partition[partition_name] = pca_projection_df.sort_values("value").reset_index(drop=True)

    selected_projection_df = projections_by_partition[partition]

    support_values = np.arange(0, 1000, dtype=np.int64)
    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        support_features = _classifier_code_score_components(probe=probe, basis_dim=int(basis_dim))
    else:
        support_features = _circular_codebook(int(basis_dim))
    support_df = pd.DataFrame({"value": support_values})
    for component_idx in range(int(basis_dim)):
        support_df[f"component_{component_idx + 1:02d}"] = support_features[:, component_idx].astype(np.float64)
    support_df = _reorder_component_frame(support_df, int(basis_dim))
    raw_support_df: pd.DataFrame | None = None
    support_scale_df: pd.DataFrame | None = None
    support_rescale_method = str(classifier_support_rescale)
    if architecture != CIRCULAR_CLS_PROBE_ARCHITECTURE and support_rescale_method != "none":
        raise ValueError("--classifier-support-rescale can only be used with architecture=circular_cls.")
    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        support_scale_df = _classifier_support_rescale_table(
            probe=probe,
            basis_dim=int(basis_dim),
            projections_by_partition=projections_by_partition,
            raw_support_df=support_df,
            method=support_rescale_method,
        )
        if support_rescale_method != "none":
            raw_support_df = support_df.copy()
            support_df, projections_by_partition = _apply_support_rescale(
                support_df=support_df,
                projections_by_partition=projections_by_partition,
                scale_table=support_scale_df,
                basis_dim=int(basis_dim),
            )

    even_indices = np.arange(0, int(basis_dim), 2, dtype=np.float64)
    log_term = np.log(1000.0) / float(max(int(basis_dim), 2))
    div_term = np.exp(even_indices * -log_term)
    component_rows: list[dict[str, Any]] = []
    for component_idx in range(int(basis_dim)):
        original_component_idx = _component_reorder_indices(int(basis_dim))[component_idx]
        pair_idx = original_component_idx // 2
        frequency_scale = float(div_term[pair_idx])
        predicted_component = selected_projection_df[f"component_{component_idx + 1:02d}"].to_numpy(dtype=np.float64)
        support_component = selected_projection_df[f"support_component_{component_idx + 1:02d}"].to_numpy(dtype=np.float64)
        component_rows.append(
            {
                "component": int(component_idx + 1),
                "kind": "sin" if original_component_idx % 2 == 0 else "cos",
                "pair_index": int(component_idx // 2 + 1),
                "frequency_scale": frequency_scale,
                "pred_mean": float(predicted_component.mean()),
                "pred_std": float(predicted_component.std(ddof=0)),
                "support_mean": float(support_component.mean()),
                "support_std": float(support_component.std(ddof=0)),
                "support_scale_factor": float(support_scale_df.iloc[component_idx]["scale_factor"])
                if support_scale_df is not None
                else 1.0,
                "component_r2_to_support": _r2_score(support_component, predicted_component),
                "component_pearson_to_support": float(
                    np.corrcoef(predicted_component, support_component)[0, 1]
                )
                if np.std(predicted_component) > 0 and np.std(support_component) > 0
                else float("nan"),
                "component_pearson_to_value": float(
                    np.corrcoef(predicted_component, selected_projection_df["value"].to_numpy(dtype=np.float64))[0, 1]
                )
                if np.std(predicted_component) > 0
                else float("nan"),
            }
        )
    component_summary_df = pd.DataFrame(component_rows)

    pca_rows: list[dict[str, Any]] = []
    total_explained_variance = float(np.sum(pca_explained_variance))
    explained_ratio = (
        pca_explained_variance / total_explained_variance if total_explained_variance > 0.0 else np.zeros_like(pca_explained_variance)
    )
    cumulative_ratio = np.cumsum(explained_ratio)
    for component_idx in range(int(basis_dim)):
        pca_component = pca_projections_by_partition[partition][f"pc_{component_idx + 1:02d}"].to_numpy(dtype=np.float64)
        pca_rows.append(
            {
                "component": int(component_idx + 1),
                "explained_variance": float(pca_explained_variance[component_idx]),
                "explained_variance_ratio": float(explained_ratio[component_idx]),
                "cumulative_explained_variance_ratio": float(cumulative_ratio[component_idx]),
                "projected_mean": float(pca_component.mean()),
                "projected_std": float(pca_component.std(ddof=0)),
                "component_pearson_to_value": float(
                    np.corrcoef(pca_component, pca_projections_by_partition[partition]["value"].to_numpy(dtype=np.float64))[0, 1]
                )
                if np.std(pca_component) > 0
                else float("nan"),
            }
        )
    pca_summary_df = pd.DataFrame(pca_rows)

    selection_payload = {
        "combo_dir": str(combo_dir),
        "model_alias": model_alias,
        "model_name": model_name,
        "model_revision": model_revision,
        "architecture": architecture,
        "architecture_label": probe_architecture_display_name(architecture),
        "legacy_source_architecture": "circular" if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE else None,
        "component_coordinate_space": (
            "fixed sinusoidal circular feature space"
            if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE
            else "effective classifier code-score coordinates"
        ),
        "classifier_support_rescale": support_rescale_method,
        "probe_config": probe_config,
        "family": family,
        "family_label": str(selected["family_label"]),
        "relative_layer": relative_layer,
        "layer_idx": layer_idx,
        "basis_dim": int(basis_dim),
        "partition": partition,
        "split_name": split_name,
        "summary_row": {
            "mean_abs_error": float(selected["mean_abs_error"]),
            "accuracy": float(selected["accuracy"]),
            "rmse": float(selected["rmse"]),
            "r2": float(selected["r2"]),
        },
        "pca": {
            "n_components": int(basis_dim),
            "fit_partition": "train",
            "explained_variance_ratio_sum": float(explained_ratio.sum()),
        },
        "available_partitions": ["train", "dev", "test"],
        "resolved_payload": resolved_payload,
    }
    return projections_by_partition, support_df, raw_support_df, support_scale_df, pca_projections_by_partition, component_summary_df, pca_summary_df, {
        **selection_payload,
        "probe": probe,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "pca_explained_variance": pca_explained_variance,
    }


def _component_scatter_figure(
    projections_by_partition: dict[str, pd.DataFrame],
    support_df: pd.DataFrame,
    *,
    raw_support_df: pd.DataFrame | None = None,
    basis_dim: int,
    family_label: str,
    component_title: str,
    selected_partition: str,
    support_trace_label: str,
    support_component_hover_label: str,
) -> go.Figure:
    cols = min(3, int(basis_dim))
    rows = math.ceil(int(basis_dim) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Component {idx + 1}" for idx in range(int(basis_dim))],
        shared_yaxes=True,
        vertical_spacing=0.045,
    )
    partitions = ["train", "dev", "test"]
    partition_styles = {
        "train": {"size": 4.0, "opacity": 0.52},
        "dev": {"size": 4.4, "opacity": 0.68},
        "test": {"size": 4.8, "opacity": 0.82},
    }
    for component_idx in range(int(basis_dim)):
        row = component_idx // cols + 1
        col = component_idx % cols + 1
        component_name = f"component_{component_idx + 1:02d}"
        if raw_support_df is not None:
            fig.add_trace(
                go.Scatter(
                    x=raw_support_df[component_name],
                    y=raw_support_df["value"],
                    mode="lines",
                    line=dict(color="rgba(20,20,20,0.25)", width=1.0, dash="dot"),
                    hovertemplate=(
                        f"Raw {support_component_hover_label} {component_idx + 1}<br>"
                        "numeric value=%{y}<br>"
                        "raw support coordinate=%{x:.4f}<extra></extra>"
                    ),
                    name=f"Raw {support_trace_label}",
                    legendgroup="raw_ideal_support",
                    showlegend=(component_idx == 0),
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=support_df[component_name],
                y=support_df["value"],
                mode="lines",
                line=dict(color="rgba(20,20,20,0.55)", width=1.2),
                hovertemplate=(
                    f"{support_component_hover_label} {component_idx + 1}<br>"
                    "numeric value=%{y}<br>"
                    "support coordinate=%{x:.4f}<extra></extra>"
                ),
                name=support_trace_label,
                legendgroup="ideal_support",
                showlegend=(component_idx == 0),
            ),
            row=row,
            col=col,
        )
        for partition_name in partitions:
            projection_df = projections_by_partition[partition_name]
            color_values = projection_df["value"].to_numpy(dtype=np.float64)
            style = partition_styles[partition_name]
            visible = True if partition_name == selected_partition else "legendonly"
            fig.add_trace(
                go.Scatter(
                    x=projection_df[component_name],
                    y=projection_df["value"],
                    mode="markers",
                    visible=visible,
                    marker=dict(
                        size=style["size"],
                        opacity=style["opacity"],
                        color=color_values,
                        colorscale="Viridis",
                        showscale=(component_idx == int(basis_dim) - 1),
                        colorbar=dict(
                            title="Numeric value",
                            len=min(0.34, 0.82 / max(rows, 1)),
                            y=1.0 - (min(0.34, 0.82 / max(rows, 1)) / 2.0),
                        )
                        if (component_idx == int(basis_dim) - 1)
                        else None,
                    ),
                    customdata=np.stack(
                        [
                            projection_df["prediction"].to_numpy(dtype=np.int64),
                            projection_df["abs_error"].to_numpy(dtype=np.int64),
                            projection_df[f"support_{component_name}"].to_numpy(dtype=np.float64),
                        ],
                        axis=1,
                    ),
                    hovertemplate=(
                        f"partition={partition_name}<br>"
                        f"component={component_idx + 1}<br>"
                        "numeric value=%{y}<br>"
                        "projected coordinate=%{x:.4f}<br>"
                        "support coordinate=%{customdata[2]:.4f}<br>"
                        "prediction=%{customdata[0]}<br>"
                        "abs_error=%{customdata[1]}<extra></extra>"
                    ),
                    name=f"{partition_name.capitalize()} examples",
                    legendgroup=f"{partition_name}_examples",
                    showlegend=(component_idx == 0),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text=f"Projected component {component_idx + 1}", row=row, col=col)
    for row_idx in range(rows):
        fig.update_yaxes(title_text="Numeric value", row=row_idx + 1, col=1)

    fig.update_layout(
        template="plotly_white",
        title=f"Numeric Value vs Projected {component_title} Component | {family_label}",
        height=max(int(576 * rows), 1760),
        width=1500,
        margin=dict(l=60, r=40, t=138, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            bordercolor="rgba(217,207,191,0.95)",
            borderwidth=1,
            bgcolor="rgba(255,250,243,0.92)",
        ),
    )
    return fig


def _pairwise_lissajous_figure(
    projections_by_partition: dict[str, pd.DataFrame],
    support_df: pd.DataFrame,
    *,
    raw_support_df: pd.DataFrame | None = None,
    basis_dim: int,
    family_label: str,
    component_title: str,
    selected_partition: str,
    support_trace_label: str,
    support_pair_hover_label: str,
) -> go.Figure:
    component_names = [f"component_{idx + 1:02d}" for idx in range(int(basis_dim))]
    fig = make_subplots(
        rows=int(basis_dim),
        cols=int(basis_dim),
        horizontal_spacing=0.004,
        vertical_spacing=0.004,
    )
    combined_projection_df = pd.concat([projections_by_partition[name] for name in ["train", "dev", "test"]], ignore_index=True)
    min_coord = float(min(combined_projection_df[component_names].min().min(), support_df[component_names].min().min()))
    max_coord = float(max(combined_projection_df[component_names].max().max(), support_df[component_names].max().max()))
    pad = 0.08 * max(max_coord - min_coord, 1.0)
    axis_range = [min_coord - pad, max_coord + pad]
    partition_styles = {
        "train": {"size": 2.0, "opacity": 0.34},
        "dev": {"size": 2.2, "opacity": 0.50},
        "test": {"size": 2.4, "opacity": 0.66},
    }
    for row_idx in range(int(basis_dim)):
        for col_idx in range(int(basis_dim)):
            x_name = component_names[col_idx]
            y_name = component_names[row_idx]
            if raw_support_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=raw_support_df[x_name],
                        y=raw_support_df[y_name],
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0.16)", width=0.8, dash="dot"),
                        hovertemplate=(
                            f"Raw {support_pair_hover_label}<br>x=component {col_idx + 1}<br>y=component {row_idx + 1}<br>"
                            "numeric value=%{text}<extra></extra>"
                        ),
                        text=raw_support_df["value"],
                        name=f"Raw {support_trace_label}",
                        legendgroup="raw_ideal_support",
                        showlegend=(row_idx == 0 and col_idx == 0),
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
            fig.add_trace(
                go.Scatter(
                    x=support_df[x_name],
                    y=support_df[y_name],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.28)", width=1),
                    hovertemplate=(
                        f"{support_pair_hover_label}<br>x=component {col_idx + 1}<br>y=component {row_idx + 1}<br>"
                        "numeric value=%{text}<extra></extra>"
                    ),
                    text=support_df["value"],
                    name=support_trace_label,
                    legendgroup="ideal_support",
                    showlegend=(row_idx == 0 and col_idx == 0),
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )
            for partition_name in ["train", "dev", "test"]:
                projection_df = projections_by_partition[partition_name]
                color_values = projection_df["value"].to_numpy(dtype=np.float64)
                style = partition_styles[partition_name]
                fig.add_trace(
                    go.Scattergl(
                        x=projection_df[x_name],
                        y=projection_df[y_name],
                        mode="markers",
                        visible=True if partition_name == selected_partition else "legendonly",
                        marker=dict(
                            size=style["size"],
                            opacity=style["opacity"],
                            color=color_values,
                            colorscale="Viridis",
                            showscale=(row_idx == 0 and col_idx == int(basis_dim) - 1),
                            colorbar=dict(title="Numeric value", len=0.22, y=0.92)
                            if (row_idx == 0 and col_idx == int(basis_dim) - 1)
                            else None,
                        ),
                        customdata=np.stack(
                            [
                                projection_df["prediction"].to_numpy(dtype=np.int64),
                                projection_df["abs_error"].to_numpy(dtype=np.int64),
                            ],
                            axis=1,
                        ),
                        hovertemplate=(
                            f"partition={partition_name}<br>"
                            f"x=component {col_idx + 1}, y=component {row_idx + 1}<br>"
                            "numeric value=%{marker.color}<br>"
                            "x=%{x:.4f}<br>"
                            "y=%{y:.4f}<br>"
                            "prediction=%{customdata[0]}<br>"
                            "abs_error=%{customdata[1]}<extra></extra>"
                        ),
                        name=f"{partition_name.capitalize()} examples",
                        legendgroup=f"{partition_name}_examples",
                        showlegend=(row_idx == 0 and col_idx == 0),
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
            fig.update_xaxes(range=axis_range, row=row_idx + 1, col=col_idx + 1, showgrid=False, zeroline=False)
            fig.update_yaxes(range=axis_range, row=row_idx + 1, col=col_idx + 1, showgrid=False, zeroline=False)
            if row_idx == int(basis_dim) - 1:
                fig.update_xaxes(title_text=f"C{col_idx + 1}", row=row_idx + 1, col=col_idx + 1)
            if col_idx == 0:
                fig.update_yaxes(title_text=f"C{row_idx + 1}", row=row_idx + 1, col=col_idx + 1)

    fig.update_layout(
        template="plotly_white",
        title=f"Pairwise {component_title} Component Lissajous Grid | {family_label}",
        height=2500,
        width=2500,
        margin=dict(l=70, r=40, t=138, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            bordercolor="rgba(217,207,191,0.95)",
            borderwidth=1,
            bgcolor="rgba(255,250,243,0.92)",
        ),
    )
    return fig


def _pairwise_pca_figure(
    pca_projections_by_partition: dict[str, pd.DataFrame],
    *,
    basis_dim: int,
    family_label: str,
    selected_partition: str,
) -> go.Figure:
    component_names = [f"pc_{idx + 1:02d}" for idx in range(int(basis_dim))]
    fig = make_subplots(
        rows=int(basis_dim),
        cols=int(basis_dim),
        horizontal_spacing=0.004,
        vertical_spacing=0.004,
    )
    combined_projection_df = pd.concat(
        [pca_projections_by_partition[name] for name in ["train", "dev", "test"]],
        ignore_index=True,
    )
    min_coord = float(combined_projection_df[component_names].min().min())
    max_coord = float(combined_projection_df[component_names].max().max())
    pad = 0.08 * max(max_coord - min_coord, 1.0)
    axis_range = [min_coord - pad, max_coord + pad]
    partition_styles = {
        "train": {"size": 2.0, "opacity": 0.34},
        "dev": {"size": 2.2, "opacity": 0.50},
        "test": {"size": 2.4, "opacity": 0.66},
    }
    for row_idx in range(int(basis_dim)):
        for col_idx in range(int(basis_dim)):
            x_name = component_names[col_idx]
            y_name = component_names[row_idx]
            for partition_name in ["train", "dev", "test"]:
                projection_df = pca_projections_by_partition[partition_name]
                color_values = projection_df["value"].to_numpy(dtype=np.float64)
                style = partition_styles[partition_name]
                fig.add_trace(
                    go.Scattergl(
                        x=projection_df[x_name],
                        y=projection_df[y_name],
                        mode="markers",
                        visible=True if partition_name == selected_partition else "legendonly",
                        marker=dict(
                            size=style["size"],
                            opacity=style["opacity"],
                            color=color_values,
                            colorscale="Viridis",
                            showscale=(row_idx == int(basis_dim) - 1 and col_idx == int(basis_dim) - 1),
                            colorbar=dict(title="Numeric value", len=0.22, y=0.18)
                            if (row_idx == int(basis_dim) - 1 and col_idx == int(basis_dim) - 1)
                            else None,
                        ),
                        customdata=np.stack(
                            [
                                projection_df["prediction"].to_numpy(dtype=np.int64),
                                projection_df["abs_error"].to_numpy(dtype=np.int64),
                            ],
                            axis=1,
                        ),
                        hovertemplate=(
                            f"partition={partition_name}<br>"
                            f"x=PC {col_idx + 1}, y=PC {row_idx + 1}<br>"
                            "numeric value=%{marker.color}<br>"
                            "x=%{x:.4f}<br>"
                            "y=%{y:.4f}<br>"
                            "prediction=%{customdata[0]}<br>"
                            "abs_error=%{customdata[1]}<extra></extra>"
                        ),
                        name=f"{partition_name.capitalize()} examples",
                        legendgroup=f"{partition_name}_examples",
                        showlegend=(row_idx == 0 and col_idx == 0),
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
            fig.update_xaxes(range=axis_range, row=row_idx + 1, col=col_idx + 1, showgrid=False, zeroline=False)
            fig.update_yaxes(range=axis_range, row=row_idx + 1, col=col_idx + 1, showgrid=False, zeroline=False)
            if row_idx == int(basis_dim) - 1:
                fig.update_xaxes(title_text=f"PC{col_idx + 1}", row=row_idx + 1, col=col_idx + 1)
            if col_idx == 0:
                fig.update_yaxes(title_text=f"PC{row_idx + 1}", row=row_idx + 1, col=col_idx + 1)

    fig.update_layout(
        template="plotly_white",
        title=f"Pairwise PCA Component Grid | {family_label}",
        height=2500,
        width=2500,
        margin=dict(l=70, r=40, t=138, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            bordercolor="rgba(217,207,191,0.95)",
            borderwidth=1,
            bgcolor="rgba(255,250,243,0.92)",
        ),
    )
    return fig


def build_report(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    architecture = canonicalize_probe_architecture_name(str(args.architecture))
    terms = _projection_terms(architecture)
    config_payload, summary_df = _load_run_context(run_dir)
    selected, candidates = _select_cell(
        summary_df,
        model_alias=str(args.model_alias),
        architecture=architecture,
        split_name=str(args.split_name),
        basis_dim=int(args.basis_dim),
        partition=str(args.partition),
        family=None if args.family is None else str(args.family),
        relative_layer=args.relative_layer,
    )
    report_path = ROOT / "experiment-reports" / f"{args.report_stem}.html"
    artifact_dir = ensure_dir(ROOT / "experiment-reports" / "artifacts" / str(args.report_stem))
    ensure_dir(report_path.parent)
    probe_artifact_path = artifact_dir / "selected_probe.json" if bool(args.reuse_probe_artifact) else None
    (
        projections_by_partition,
        support_df,
        raw_support_df,
        support_scale_df,
        pca_projections_by_partition,
        component_summary_df,
        pca_summary_df,
        payload,
    ) = _prepare_projection_payload(
        run_dir=run_dir,
        config_payload=config_payload,
        selected=selected,
        architecture=architecture,
        basis_dim=int(args.basis_dim),
        partition=str(args.partition),
        classifier_support_rescale=str(args.classifier_support_rescale),
        probe_artifact_path=probe_artifact_path,
    )
    probe = payload.pop("probe")

    projection_csv_names = {partition: f"{partition}_component_projections.csv" for partition in ["train", "dev", "test"]}
    pca_projection_csv_names = {partition: f"{partition}_pca_projections.csv" for partition in ["train", "dev", "test"]}
    support_csv_name = "support_component_lookup.csv"
    raw_support_csv_name = "raw_support_component_lookup.csv"
    support_scale_csv_name = "classifier_support_rescale.csv"
    component_summary_name = "component_summary.csv"
    pca_summary_name = "pca_component_summary.csv"
    candidate_csv_name = "basis_dim_candidate_rows.csv"
    probe_json_name = "selected_probe.json"
    probe_npz_name = "selected_probe_weights.npz"
    pca_npz_name = "selected_pca_weights.npz"
    selection_json_name = "selected_cell.json"

    for partition_name, projection_df in projections_by_partition.items():
        projection_df.to_csv(artifact_dir / projection_csv_names[partition_name], index=False)
    for partition_name, projection_df in pca_projections_by_partition.items():
        projection_df.to_csv(artifact_dir / pca_projection_csv_names[partition_name], index=False)
    support_df.to_csv(artifact_dir / support_csv_name, index=False)
    if raw_support_df is not None:
        raw_support_df.to_csv(artifact_dir / raw_support_csv_name, index=False)
    if support_scale_df is not None:
        support_scale_df.to_csv(artifact_dir / support_scale_csv_name, index=False)
    component_summary_df.to_csv(artifact_dir / component_summary_name, index=False)
    pca_summary_df.to_csv(artifact_dir / pca_summary_name, index=False)
    candidates.loc[
        :,
        ["family", "family_label", "relative_layer", "layer_idx", "mean_abs_error", "accuracy", "rmse", "r2"],
    ].to_csv(artifact_dir / candidate_csv_name, index=False)
    if not bool(args.reuse_probe_artifact):
        (artifact_dir / probe_json_name).write_text(json.dumps(probe.to_jsonable(), indent=2), encoding="utf-8")
        np.savez_compressed(
            artifact_dir / probe_npz_name,
            coefficients=np.asarray(probe.coefficients, dtype=np.float64),
            bias=np.asarray(probe.bias, dtype=np.float64),
            decoder_matrix=np.asarray(probe.decoder_matrix, dtype=np.float64),
            support_values=np.asarray(probe.support_values, dtype=np.int64),
            support_features=np.asarray(probe.support_features, dtype=np.float64),
        )
    np.savez_compressed(
        artifact_dir / pca_npz_name,
        mean=np.asarray(payload.pop("pca_mean"), dtype=np.float64),
        components=np.asarray(payload.pop("pca_components"), dtype=np.float64),
        explained_variance=np.asarray(payload.pop("pca_explained_variance"), dtype=np.float64),
    )
    (artifact_dir / selection_json_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    component_fig = _component_scatter_figure(
        projections_by_partition,
        support_df,
        raw_support_df=raw_support_df,
        basis_dim=int(args.basis_dim),
        family_label=f"{selected['family_label']} ({terms['title']})",
        component_title=str(terms["title"]),
        selected_partition=str(args.partition),
        support_trace_label=str(terms["support_trace_label"]),
        support_component_hover_label=str(terms["support_component_hover_label"]),
    )
    pca_fig = _pairwise_pca_figure(
        pca_projections_by_partition,
        basis_dim=int(args.basis_dim),
        family_label=str(selected["family_label"]),
        selected_partition=str(args.partition),
    )
    lissajous_fig = _pairwise_lissajous_figure(
        projections_by_partition,
        support_df,
        raw_support_df=raw_support_df,
        basis_dim=int(args.basis_dim),
        family_label=f"{selected['family_label']} ({terms['title']})",
        component_title=str(terms["title"]),
        selected_partition=str(args.partition),
        support_trace_label=str(terms["support_trace_label"]),
        support_pair_hover_label=str(terms["support_pair_hover_label"]),
    )

    candidate_table = candidates.loc[
        :,
        ["family_label", "relative_layer", "layer_idx", "mean_abs_error", "accuracy", "rmse", "r2"],
    ].rename(
        columns={
            "family_label": "Prompt family",
            "relative_layer": "Relative layer",
            "layer_idx": "Layer idx",
            "mean_abs_error": "Test MAE",
            "accuracy": "Test exact accuracy",
            "rmse": "Test RMSE",
            "r2": "Test R^2",
        }
    )
    component_table = component_summary_df.rename(
        columns={
            "component": "Component",
            "kind": "Kind",
            "pair_index": "Sin/Cos pair",
            "frequency_scale": "Frequency scale",
            "pred_mean": "Projected mean",
            "pred_std": "Projected std",
            "support_std": "Support std",
            "support_scale_factor": "Support scale factor",
            "component_r2_to_support": "Component R^2 vs support",
            "component_pearson_to_support": "Component Pearson vs support",
            "component_pearson_to_value": "Component Pearson vs numeric value",
        }
    )[
        [
            "Component",
            "Kind",
            "Sin/Cos pair",
            "Frequency scale",
            "Projected mean",
            "Projected std",
            "Support std",
            "Support scale factor",
            "Component R^2 vs support",
            "Component Pearson vs support",
            "Component Pearson vs numeric value",
        ]
    ]
    support_scale_table = (
        support_scale_df.rename(
            columns={
                "component": "Component",
                "original_component": "Original component",
                "rescale_method": "Rescale method",
                "scale_factor": "Scale factor",
                "train_std": "Train std",
                "raw_support_std": "Raw support std",
                "w_out_column_norm": "W_out column norm",
                "w_in_weighted_row_norm": "W_in weighted row norm",
                "effective_input_axis_norm": "Effective input-axis norm",
            }
        )
        if support_scale_df is not None
        else pd.DataFrame()
    )
    pca_component_table = pca_summary_df.rename(
        columns={
            "component": "PC",
            "explained_variance": "Explained variance",
            "explained_variance_ratio": "Explained variance ratio",
            "cumulative_explained_variance_ratio": "Cumulative explained variance ratio",
            "projected_mean": "Projected mean",
            "projected_std": "Projected std",
            "component_pearson_to_value": "PC Pearson vs numeric value",
        }
    )[
        [
            "PC",
            "Explained variance",
            "Explained variance ratio",
            "Cumulative explained variance ratio",
            "Projected mean",
            "Projected std",
            "PC Pearson vs numeric value",
        ]
    ]
    model_display_name = _model_display_name(str(payload["model_alias"]))
    support_rescale_method = str(payload.get("classifier_support_rescale", "none"))
    support_rescale_note = ""
    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE and support_rescale_method != "none":
        support_rescale_note = (
            f" For visualization, the classifier-projected support curves are rescaled with "
            f"<code>{support_rescale_method}</code>; dotted gray curves show the unscaled classifier support."
        )
    support_extra_artifacts = ""
    if support_scale_df is not None:
        support_extra_artifacts += f", <code>{artifact_dir / support_scale_csv_name}</code>"
    if raw_support_df is not None:
        support_extra_artifacts += f", <code>{artifact_dir / raw_support_csv_name}</code>"
    support_scale_section = ""
    if support_scale_df is not None:
        support_scale_section = f"""
  <section class="card">
    <h2>Classifier Support Scale</h2>
    <p>
      For <code>circular_cls</code>, examples are scored by an inner product between an input projection from <code>W_in</code> and a label projection from
      <code>W_out</code>. These per-component diagnostics expose the visualization scale factors and the relevant learned-vector norms.
    </p>
    {_render_table(support_scale_table)}
  </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{model_display_name} {terms["title"]} Component Report</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffaf3;
      --ink: #1f2421;
      --muted: #5f665e;
      --accent: #a63d40;
      --accent-soft: #efe0cc;
      --border: #d9cfbf;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background:
        radial-gradient(circle at top left, rgba(166, 61, 64, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(67, 122, 97, 0.08), transparent 24%),
        linear-gradient(180deg, #f9f4ec 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px 0;
      font-weight: 700;
      line-height: 1.15;
    }}
    h1 {{
      font-size: 2.2rem;
      letter-spacing: -0.02em;
    }}
    h2 {{
      font-size: 1.3rem;
      margin-top: 28px;
    }}
    p, li {{
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.55;
    }}
    .lede {{
      max-width: 980px;
      font-size: 1.05rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat, .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(48, 39, 28, 0.07);
    }}
    .stat {{
      padding: 14px 16px;
    }}
    .stat .label {{
      font-size: 0.82rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .stat .value {{
      font-size: 1.55rem;
      color: var(--ink);
      margin-top: 6px;
    }}
    .card {{
      padding: 18px 20px;
      margin-top: 18px;
    }}
    .caption {{
      font-size: 0.92rem;
      color: var(--muted);
      margin-top: 12px;
    }}
    code {{
      background: var(--accent-soft);
      padding: 0.12rem 0.36rem;
      border-radius: 6px;
      color: var(--ink);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }}
    a {{
      color: var(--accent);
    }}
    table.dataframe {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.93rem;
    }}
    table.dataframe th, table.dataframe td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    table.dataframe th {{
      color: var(--ink);
      font-size: 0.84rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
  </style>
</head>
<body>
<main>
  <h1>{model_display_name} {terms["title"]} Probe Components at Basis Dimension {payload["basis_dim"]}</h1>
  <p class="lede">
    This report visualizes the learned <strong>{terms["probe_label"]}</strong> coordinates for the best held-out <code>basis_dim=10</code> source-value cell on
    <code>{payload["model_name"]}</code>. In the notation of
    <em>Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers</em>, the plotted x and y coordinates are {terms["mapping"]}, while the thin black reference curves trace {terms["support_reference"]}.
    {terms["distinction"]}{support_rescale_note}
  </p>

  <div class="grid">
    <div class="stat"><div class="label">Selected Prompt Family</div><div class="value">{payload["family_label"]}</div></div>
    <div class="stat"><div class="label">Relative Layer</div><div class="value">{payload["relative_layer"]:.1f}</div></div>
    <div class="stat"><div class="label">Absolute Layer</div><div class="value">{payload["layer_idx"]}</div></div>
    <div class="stat"><div class="label">Held-Out Partition</div><div class="value">{payload["partition"]}</div></div>
    <div class="stat"><div class="label">Test MAE</div><div class="value">{payload["summary_row"]["mean_abs_error"]:.2f}</div></div>
    <div class="stat"><div class="label">Test Exact Accuracy</div><div class="value">{100.0 * payload["summary_row"]["accuracy"]:.1f}%</div></div>
  </div>

  <section class="card">
    <h2>Scope of Evidence</h2>
    <p>
      The underlying run is <code>{run_dir}</code>. The selected cell is determined from the dense low-dimensional circular sweep on
      <code>{payload["split_name"]}</code> by minimizing held-out <code>{payload["partition"]}</code> MAE within
      <code>model={payload["model_alias"]}</code>, <code>architecture={architecture}</code>, <code>basis_dim={payload["basis_dim"]}</code>.
      {terms["legacy_note"]}
      The reusable fitted probe is saved as <code>{artifact_dir / probe_json_name}</code> and <code>{artifact_dir / probe_npz_name}</code>.
    </p>
    <p>
      Saved artifact tables: <code>{artifact_dir / projection_csv_names["train"]}</code>,
      <code>{artifact_dir / projection_csv_names["dev"]}</code>,
      <code>{artifact_dir / projection_csv_names["test"]}</code>,
      <code>{artifact_dir / pca_projection_csv_names["train"]}</code>,
      <code>{artifact_dir / pca_projection_csv_names["dev"]}</code>,
      <code>{artifact_dir / pca_projection_csv_names["test"]}</code>,
      <code>{artifact_dir / support_csv_name}</code>,
      <code>{artifact_dir / component_summary_name}</code>, <code>{artifact_dir / pca_summary_name}</code>,
      <code>{artifact_dir / candidate_csv_name}</code>{support_extra_artifacts},
      and <code>{artifact_dir / selection_json_name}</code>.
    </p>
  </section>

  <section class="card">
    <h2>Candidate Cells</h2>
    <p>
      These are all <code>{payload["model_alias"]}</code> held-out <code>{payload["partition"]}</code> {terms["hyphen"]} rows at
      <code>basis_dim={payload["basis_dim"]}</code>, sorted by MAE. This makes the default selection rule explicit.
    </p>
    {_render_table(candidate_table)}
  </section>

  <section class="card">
      <h2>Component Summary</h2>
    <p>
      <code>Component R^2 vs support</code> compares each learned projected coordinate against the {terms["component_summary_reference"]} for the same held-out values.
      Large values mean the fitted component closely follows the report's support reference on unseen examples.
      If support rescaling is enabled, these support columns and summary statistics use the rescaled display coordinates.
    </p>
    {_render_table(component_table)}
  </section>
{support_scale_section}

  <section class="card">
    <h2>PCA Summary</h2>
    <p>
      The PCA basis is fit on the same cached <code>train</code> activations used to refit the {terms["short"]} probe, then applied unchanged to
      <code>train</code>, <code>dev</code>, and <code>test</code>. The table below reports how much variance each principal component explains
      in the train activation cloud, along with its correlation to numeric value on the selected <code>{payload["partition"]}</code> split.
    </p>
    {_render_table(pca_component_table)}
  </section>

  <section class="card">
    <h2>Numeric Value vs {terms["title"]} Component</h2>
    {_fig_to_div(component_fig, include_plotlyjs="cdn")}
    <p class="caption">
      Each panel plots numeric value on the y-axis against one learned projected {terms["hyphen"]} component on the x-axis.
      Use the Plotly legend to toggle the {terms["support_path_label"]} and the <code>train</code>, <code>dev</code>, and <code>test</code> example sets.
      Source artifacts: <code>{artifact_dir / projection_csv_names["train"]}</code>, <code>{artifact_dir / projection_csv_names["dev"]}</code>,
      <code>{artifact_dir / projection_csv_names["test"]}</code>, <code>{artifact_dir / support_csv_name}</code>.
    </p>
  </section>

  <section class="card">
    <h2>Pairwise Lissajous Grid</h2>
    {_fig_to_div(lissajous_fig, include_plotlyjs=False)}
    <p class="caption">
      Row <code>i</code> and column <code>j</code> show the learned projection onto {terms["hyphen"]} components <code>(j, i)</code>.
      Use the Plotly legend to toggle the {terms["support_path_label"]} and the <code>train</code>, <code>dev</code>, and <code>test</code> example sets.
      Marker color is the true numeric value.
      This makes it easier to see which 2D component pairs recover clean periodic structure and where the learned geometry visibly departs from the support code.
    </p>
  </section>

  <section class="card">
    <h2>Pairwise PCA Grid</h2>
    {_fig_to_div(pca_fig, include_plotlyjs=False)}
    <p class="caption">
      This grid uses the same cached activations as the {terms["hyphen"]} pairwise view, but replaces the learned {terms["hyphen"]} coordinates with the top
      <code>{payload["pca"]["n_components"]}</code> train-fit principal components. Use the Plotly legend to toggle the
      <code>train</code>, <code>dev</code>, and <code>test</code> example sets. Marker color is the true numeric value.
      Source artifacts: <code>{artifact_dir / pca_projection_csv_names["train"]}</code>, <code>{artifact_dir / pca_projection_csv_names["dev"]}</code>,
      <code>{artifact_dir / pca_projection_csv_names["test"]}</code>, <code>{artifact_dir / pca_npz_name}</code>.
    </p>
  </section>
</main>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    report_path = build_report(args)
    print(report_path)


if __name__ == "__main__":
    main()
