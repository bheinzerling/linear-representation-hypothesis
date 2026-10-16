from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.activation_steering import _softmax_numpy
from src.gradient_steering import (
    collect_gradient_directions,
    collect_numeric_logit_directional_derivatives,
    compute_counterfactual_patch_deltas,
    forward_next_token_logits_with_delta,
    objective_scores_numpy,
)
from src.magnitude_intervention import MagnitudeProbe, probe_from_payload
from src.ones_digit_counterfactual_gradients import (
    _best_scalar_projection,
    _family_projection_metrics,
    _row_cosine,
    _row_norms,
    _safe_ratio,
    fisher_numeric_logit_residual_metrics,
    functional_residual_metrics,
    make_norm_matched_random_deltas,
    numeric_logit_residual_metrics,
)
from src.ones_digit_intervention import _numeric_tracking_arrays
from src.steering_sanity import orthonormal_row_basis


def load_probe_models(path: str | Path) -> dict[str, MagnitudeProbe]:
    payload = json.loads(Path(path).read_text())
    return {str(name): probe_from_payload(dict(probe_payload)) for name, probe_payload in payload.items()}


def intervention_family_basis(probe: MagnitudeProbe) -> np.ndarray:
    if str(probe.architecture) == "pls_x_weight1" and "subspace_basis" in probe.metadata:
        return orthonormal_row_basis(np.asarray(probe.metadata["subspace_basis"], dtype=np.float64))
    return orthonormal_row_basis(np.asarray(probe.decoder_matrix, dtype=np.float64))


def sample_intervention_examples(
    *,
    intervention_df: pd.DataFrame,
    evaluation_splits: list[str],
    tasks: list[str],
    operand_modes: list[str],
    max_examples_per_condition: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for evaluation_split in evaluation_splits:
        split_df = intervention_df.loc[intervention_df["evaluation_split"] == str(evaluation_split)].copy()
        if split_df.empty:
            continue
        for task in tasks:
            task_df = split_df.loc[split_df["task"] == str(task)].copy()
            if task_df.empty:
                continue
            for operand_mode in operand_modes:
                mode_df = task_df.loc[task_df["operand_mode"] == str(operand_mode)].copy()
                if mode_df.empty:
                    continue
                rng = np.random.default_rng(
                    int(seed)
                    + (97 * len(str(task)))
                    + (11 if str(operand_mode) == "x2" else 0)
                    + (1 if str(evaluation_split) == "test" else 0)
                )
                sample_size = min(int(max_examples_per_condition), len(mode_df))
                take = np.sort(rng.choice(len(mode_df), size=sample_size, replace=False))
                sampled = mode_df.iloc[take].copy().reset_index(drop=True)
                sampled["condition_sample_rank"] = np.arange(len(sampled), dtype=np.int64)
                rows.append(sampled)
    if not rows:
        return pd.DataFrame()
    sampled_all = pd.concat(rows, ignore_index=True)
    sampled_all["analysis_example_id"] = np.arange(len(sampled_all), dtype=np.int64)
    return sampled_all


def _objective_frame(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    delta_rows: np.ndarray | None,
) -> pd.DataFrame:
    logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["prompt"].tolist(),
        token_idxs=valid_df["token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        delta_rows=None if delta_rows is None else np.asarray(delta_rows, dtype=np.float64),
        capture_selected=False,
    )
    target_token_ids = valid_df["target_token_id"].to_numpy(dtype=np.int64)
    baseline_token_ids = valid_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64)
    objective_score = objective_scores_numpy(
        logits,
        target_token_ids=target_token_ids,
        baseline_token_ids=baseline_token_ids,
        objective=objective,
    )
    probs = _softmax_numpy(logits)
    rows = np.arange(len(valid_df), dtype=np.int64)
    return pd.DataFrame(
        {
            "objective_score": objective_score.astype(np.float64),
            "target_prob": probs[rows, target_token_ids].astype(np.float64),
            "baseline_answer_prob": probs[rows, baseline_token_ids].astype(np.float64),
        }
    )


def _forward_logits(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    delta_rows: np.ndarray | None,
) -> np.ndarray:
    logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["prompt"].tolist(),
        token_idxs=valid_df["token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        delta_rows=None if delta_rows is None else np.asarray(delta_rows, dtype=np.float64),
        capture_selected=False,
    )
    return np.asarray(logits, dtype=np.float64)


def _forward_shifted_logits(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if not {"shifted_prompt", "shifted_token_idx"}.issubset(valid_df.columns):
        raise ValueError("Expected shifted_prompt and shifted_token_idx columns for source-value work-left analysis.")
    logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["shifted_prompt"].tolist(),
        token_idxs=valid_df["shifted_token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        delta_rows=None,
        capture_selected=False,
    )
    return np.asarray(logits, dtype=np.float64)


def _sigma_slug(sigma: float) -> str:
    text = f"{float(sigma):g}"
    return text.replace("-", "m").replace(".", "p")


def kernelized_numeric_utility_scores_numpy(
    logits: np.ndarray,
    *,
    target_values: np.ndarray,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    sigma: float,
) -> np.ndarray:
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma!r}.")
    probs = _softmax_numpy(np.asarray(logits, dtype=np.float64)[:, np.asarray(tracked_token_ids, dtype=np.int64)])
    target = np.asarray(target_values, dtype=np.float64).reshape(-1, 1)
    values = np.asarray(tracked_values, dtype=np.float64).reshape(1, -1)
    weights = np.exp(-np.square(values - target) / (2.0 * sigma_value * sigma_value))
    utility = np.sum(probs * weights, axis=1)
    return np.log(np.clip(utility, 1e-30, None)).astype(np.float64)


def numeric_full_vocab_probability_metrics(
    *,
    baseline_logits: np.ndarray,
    steered_logits: np.ndarray,
    target_token_ids: np.ndarray,
    tracked_token_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    tracked_ids = np.asarray(tracked_token_ids, dtype=np.int64)
    target_ids = np.asarray(target_token_ids, dtype=np.int64)
    rows = np.arange(len(target_ids), dtype=np.int64)
    baseline_probs = _softmax_numpy(np.asarray(baseline_logits, dtype=np.float64))
    steered_probs = _softmax_numpy(np.asarray(steered_logits, dtype=np.float64))
    baseline_numeric_mass = np.sum(baseline_probs[:, tracked_ids], axis=1)
    steered_numeric_mass = np.sum(steered_probs[:, tracked_ids], axis=1)
    baseline_target_prob = baseline_probs[rows, target_ids]
    steered_target_prob = steered_probs[rows, target_ids]
    baseline_target_prob_numeric = _safe_ratio(baseline_target_prob, baseline_numeric_mass)
    steered_target_prob_numeric = _safe_ratio(steered_target_prob, steered_numeric_mass)
    return {
        "numeric_mass_orig": baseline_numeric_mass.astype(np.float64),
        "numeric_mass_steered": steered_numeric_mass.astype(np.float64),
        "numeric_mass_gain": (steered_numeric_mass - baseline_numeric_mass).astype(np.float64),
        "target_prob_numeric_orig": baseline_target_prob_numeric.astype(np.float64),
        "target_prob_numeric_steered": steered_target_prob_numeric.astype(np.float64),
        "target_prob_numeric_gain": (steered_target_prob_numeric - baseline_target_prob_numeric).astype(np.float64),
    }


def kernelized_numeric_utility_logit_residual_metrics(
    *,
    pre_directional_logits: np.ndarray,
    post_fixed_directional_logits: np.ndarray,
    post_moving_directional_logits: np.ndarray,
    baseline_logits: np.ndarray,
    steered_logits: np.ndarray,
    target_values: np.ndarray,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    sigma: float,
) -> dict[str, np.ndarray]:
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma!r}.")

    target = np.asarray(target_values, dtype=np.float64).reshape(-1, 1)
    values = np.asarray(tracked_values, dtype=np.float64).reshape(1, -1)
    weights = np.exp(-np.square(values - target) / (2.0 * sigma_value * sigma_value))

    def _grad_log_utility(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tracked_logits = np.asarray(logits, dtype=np.float64)[:, np.asarray(tracked_token_ids, dtype=np.int64)]
        probs = _softmax_numpy(tracked_logits)
        utility = np.sum(probs * weights, axis=1)
        clipped = np.clip(utility, 1e-30, None)
        grad = probs * ((weights / clipped[:, None]) - 1.0)
        return np.log(clipped).astype(np.float64), grad.astype(np.float64)

    def _full_vocab_log_utility(logits: np.ndarray) -> np.ndarray:
        tracked_probs = _softmax_numpy(np.asarray(logits, dtype=np.float64))[:, np.asarray(tracked_token_ids, dtype=np.int64)]
        utility = np.sum(tracked_probs * weights, axis=1)
        return np.log(np.clip(utility, 1e-30, None)).astype(np.float64)

    utility_orig, grad_orig = _grad_log_utility(baseline_logits)
    utility_steered, grad_steered = _grad_log_utility(steered_logits)
    full_utility_orig = _full_vocab_log_utility(baseline_logits)
    full_utility_steered = _full_vocab_log_utility(steered_logits)
    pre_signed = np.sum(grad_orig * np.asarray(pre_directional_logits, dtype=np.float64), axis=1)
    post_fixed_signed = np.sum(grad_orig * np.asarray(post_fixed_directional_logits, dtype=np.float64), axis=1)
    post_moving_signed = np.sum(grad_steered * np.asarray(post_moving_directional_logits, dtype=np.float64), axis=1)
    pre_abs = np.abs(pre_signed)
    post_fixed_abs = np.abs(post_fixed_signed)
    post_moving_abs = np.abs(post_moving_signed)
    return {
        "utility_orig": utility_orig.astype(np.float64),
        "utility_steered": utility_steered.astype(np.float64),
        "utility_gain": (utility_steered - utility_orig).astype(np.float64),
        "full_utility_orig": full_utility_orig.astype(np.float64),
        "full_utility_steered": full_utility_steered.astype(np.float64),
        "full_utility_gain": (full_utility_steered - full_utility_orig).astype(np.float64),
        "pre_signed": pre_signed.astype(np.float64),
        "post_fixed_signed": post_fixed_signed.astype(np.float64),
        "post_moving_signed": post_moving_signed.astype(np.float64),
        "pre": pre_abs.astype(np.float64),
        "post_fixed": post_fixed_abs.astype(np.float64),
        "post_moving": post_moving_abs.astype(np.float64),
        "fixed_ratio": _safe_ratio(post_fixed_abs, pre_abs).astype(np.float64),
        "moving_ratio": _safe_ratio(post_moving_abs, pre_abs).astype(np.float64),
    }


def analyze_counterfactual_deltas(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    probe: MagnitudeProbe,
    architecture: str,
    delta_rows: np.ndarray,
    family_basis: np.ndarray | None,
    edit_family: str,
    matched_to_architecture: str,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    functional_objective: str,
    fisher_path_num_points: int = 17,
    kernel_sigmas: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    delta_arr = np.asarray(delta_rows, dtype=np.float64)
    source_values = valid_df["source_value"].to_numpy(dtype=np.int64)
    target_values = valid_df["target_value"].to_numpy(dtype=np.int64)
    output_target_values = (
        valid_df["shifted_target"].to_numpy(dtype=np.int64)
        if "shifted_target" in valid_df.columns
        else target_values
    )
    patch_deltas = compute_counterfactual_patch_deltas(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
    )
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
    grad_orig = collect_gradient_directions(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=functional_objective,
        delta_rows=None,
    )
    grad_steered = collect_gradient_directions(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=functional_objective,
        delta_rows=delta_arr,
    )

    pre_numeric_jvp = collect_numeric_logit_directional_derivatives(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        tracked_token_ids=tracked_token_ids,
        direction_rows=patch_deltas,
        delta_rows=None,
    )
    post_fixed_numeric_jvp = collect_numeric_logit_directional_derivatives(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        tracked_token_ids=tracked_token_ids,
        direction_rows=patch_deltas - delta_arr,
        delta_rows=None,
    )
    post_moving_numeric_jvp = collect_numeric_logit_directional_derivatives(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        tracked_token_ids=tracked_token_ids,
        direction_rows=patch_deltas - delta_arr,
        delta_rows=delta_arr,
    )

    baseline_frame = _objective_frame(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=objective,
        delta_rows=None,
    )
    steered_frame = _objective_frame(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=objective,
        delta_rows=delta_arr,
    )

    patch_norm = _row_norms(patch_deltas)
    delta_norm = _row_norms(delta_arr)
    patch_minus_probe = patch_deltas - delta_arr
    patch_minus_probe_norm = _row_norms(patch_minus_probe)
    best_alpha = _best_scalar_projection(patch_deltas, delta_arr)
    best_delta = best_alpha[:, None] * delta_arr
    patch_minus_best = patch_deltas - best_delta
    patch_minus_best_norm = _row_norms(patch_minus_best)
    family_metrics = _family_projection_metrics(
        patch_deltas=patch_deltas,
        family_basis=family_basis,
        grad_orig=grad_orig,
        grad_steered=grad_steered,
    )
    functional_metrics = functional_residual_metrics(
        patch_deltas=patch_deltas,
        probe_deltas=delta_arr,
        grad_orig=grad_orig,
        grad_steered=grad_steered,
    )
    numeric_metrics = numeric_logit_residual_metrics(
        pre_directional_logits=pre_numeric_jvp,
        post_fixed_directional_logits=post_fixed_numeric_jvp,
        post_moving_directional_logits=post_moving_numeric_jvp,
    )
    baseline_logits = _forward_logits(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        delta_rows=None,
    )
    steered_logits = _forward_logits(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        delta_rows=delta_arr,
    )
    counterfactual_logits = _forward_shifted_logits(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
    )
    fisher_metrics = fisher_numeric_logit_residual_metrics(
        pre_directional_logits=pre_numeric_jvp,
        post_fixed_directional_logits=post_fixed_numeric_jvp,
        post_moving_directional_logits=post_moving_numeric_jvp,
        baseline_logits=baseline_logits,
        counterfactual_logits=counterfactual_logits,
        tracked_token_ids=tracked_token_ids,
        path_num_points=int(fisher_path_num_points),
    )
    numeric_probability_metrics = numeric_full_vocab_probability_metrics(
        baseline_logits=baseline_logits,
        steered_logits=steered_logits,
        target_token_ids=valid_df["target_token_id"].to_numpy(dtype=np.int64),
        tracked_token_ids=tracked_token_ids,
    )
    kernel_metric_payloads: dict[str, dict[str, np.ndarray]] = {}
    for sigma in kernel_sigmas or []:
        sigma_value = float(sigma)
        slug = _sigma_slug(sigma_value)
        kernel_metrics = kernelized_numeric_utility_logit_residual_metrics(
            pre_directional_logits=pre_numeric_jvp,
            post_fixed_directional_logits=post_fixed_numeric_jvp,
            post_moving_directional_logits=post_moving_numeric_jvp,
            baseline_logits=baseline_logits,
            steered_logits=steered_logits,
            target_values=output_target_values,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            sigma=sigma_value,
        )
        kernel_metric_payloads[slug] = {
            "sigma": np.full(len(valid_df), sigma_value, dtype=np.float64),
            **kernel_metrics,
        }
    detail = valid_df[
        [column for column in [
            "analysis_example_id",
            "evaluation_split",
            "family",
            "family_label",
            "task",
            "operand_mode",
            "example_id",
            "source_value",
            "target_value",
            "delta_value",
            "abs_delta_value",
            "x1",
            "x2",
            "target",
            "shifted_x1",
            "shifted_x2",
            "shifted_target",
        ] if column in valid_df.columns]
    ].copy()
    detail["architecture"] = str(architecture)
    detail["edit_family"] = str(edit_family)
    detail["matched_to_architecture"] = str(matched_to_architecture)
    detail["probe_delta_norm"] = delta_norm.astype(np.float64)
    detail["patch_delta_norm"] = patch_norm.astype(np.float64)
    detail["patch_probe_cosine"] = _row_cosine(patch_deltas, delta_arr).astype(np.float64)
    detail["patch_explained_fraction"] = family_metrics["patch_explained_fraction"]
    detail["patch_orthogonal_fraction"] = family_metrics["patch_orthogonal_fraction"]
    detail["patch_minus_probe_norm"] = patch_minus_probe_norm.astype(np.float64)
    detail["patch_minus_probe_fraction"] = _safe_ratio(patch_minus_probe_norm, patch_norm).astype(np.float64)
    detail["patch_minus_best_probe_fraction"] = _safe_ratio(patch_minus_best_norm, patch_norm).astype(np.float64)
    detail["patch_best_alpha"] = best_alpha.astype(np.float64)
    detail["grad_orig_norm"] = _row_norms(grad_orig).astype(np.float64)
    detail["grad_steered_norm"] = _row_norms(grad_steered).astype(np.float64)
    detail["grad_norm_ratio"] = _safe_ratio(detail["grad_steered_norm"].to_numpy(dtype=np.float64), detail["grad_orig_norm"].to_numpy(dtype=np.float64)).astype(np.float64)
    detail["grad_orig_span_norm"] = family_metrics["grad_orig_span_norm"]
    detail["grad_steered_span_norm"] = family_metrics["grad_steered_span_norm"]
    detail["grad_span_ratio"] = family_metrics["grad_span_ratio"]
    detail["grad_orig_orth_norm"] = family_metrics["grad_orig_orth_norm"]
    detail["grad_steered_orth_norm"] = family_metrics["grad_steered_orth_norm"]
    detail["grad_orth_ratio"] = family_metrics["grad_orth_ratio"]
    detail["grad_orig_span_fraction"] = family_metrics["grad_orig_span_fraction"]
    detail["grad_steered_span_fraction"] = family_metrics["grad_steered_span_fraction"]
    detail["objective_score_orig"] = baseline_frame["objective_score"].to_numpy(dtype=np.float64)
    detail["objective_score_steered"] = steered_frame["objective_score"].to_numpy(dtype=np.float64)
    detail["objective_gain"] = detail["objective_score_steered"] - detail["objective_score_orig"]
    detail["target_prob_orig"] = baseline_frame["target_prob"].to_numpy(dtype=np.float64)
    detail["target_prob_steered"] = steered_frame["target_prob"].to_numpy(dtype=np.float64)
    detail["target_prob_gain"] = detail["target_prob_steered"] - detail["target_prob_orig"]
    detail["baseline_answer_prob_orig"] = baseline_frame["baseline_answer_prob"].to_numpy(dtype=np.float64)
    detail["baseline_answer_prob_steered"] = steered_frame["baseline_answer_prob"].to_numpy(dtype=np.float64)
    for key, values in numeric_probability_metrics.items():
        detail[key] = values
    for key, values in functional_metrics.items():
        detail[key] = values
    for key, values in numeric_metrics.items():
        detail[key] = values
    for key, values in fisher_metrics.items():
        detail[key] = values
    kernel_detail_columns: dict[str, np.ndarray] = {}
    for slug, payload in kernel_metric_payloads.items():
        kernel_detail_columns[f"kernel_sigma_{slug}"] = payload["sigma"]
        kernel_detail_columns[f"kernel_sigma_{slug}_utility_orig"] = payload["utility_orig"]
        kernel_detail_columns[f"kernel_sigma_{slug}_utility_steered"] = payload["utility_steered"]
        kernel_detail_columns[f"kernel_sigma_{slug}_utility_gain"] = payload["utility_gain"]
        kernel_detail_columns[f"kernel_sigma_{slug}_full_utility_orig"] = payload["full_utility_orig"]
        kernel_detail_columns[f"kernel_sigma_{slug}_full_utility_steered"] = payload["full_utility_steered"]
        kernel_detail_columns[f"kernel_sigma_{slug}_full_utility_gain"] = payload["full_utility_gain"]
        for key in ("pre_signed", "post_fixed_signed", "post_moving_signed", "pre", "post_fixed", "post_moving", "fixed_ratio", "moving_ratio"):
            kernel_detail_columns[f"kernel_sigma_{slug}_{key}"] = payload[key]
    if kernel_detail_columns:
        detail = pd.concat([detail, pd.DataFrame(kernel_detail_columns, index=detail.index)], axis=1)

    summary_group_cols = [
        column
        for column in ["family", "family_label", "architecture", "edit_family", "matched_to_architecture", "evaluation_split", "task"]
        if column in detail.columns
    ]
    summary = (
        detail.groupby(summary_group_cols, as_index=False, dropna=False)
        .agg(
            num_examples=("analysis_example_id", "size"),
            mean_objective_gain=("objective_gain", "mean"),
            mean_target_prob_gain=("target_prob_gain", "mean"),
            mean_numeric_mass_orig=("numeric_mass_orig", "mean"),
            mean_numeric_mass_steered=("numeric_mass_steered", "mean"),
            mean_numeric_mass_gain=("numeric_mass_gain", "mean"),
            mean_target_prob_numeric_gain=("target_prob_numeric_gain", "mean"),
            mean_grad_norm_ratio=("grad_norm_ratio", "mean"),
            median_grad_norm_ratio=("grad_norm_ratio", "median"),
            mean_grad_orth_ratio=("grad_orth_ratio", "mean"),
            mean_patch_probe_cosine=("patch_probe_cosine", "mean"),
            mean_patch_explained_fraction=("patch_explained_fraction", "mean"),
            mean_patch_minus_probe_fraction=("patch_minus_probe_fraction", "mean"),
            mean_patch_minus_best_probe_fraction=("patch_minus_best_probe_fraction", "mean"),
            mean_patch_best_alpha=("patch_best_alpha", "mean"),
            mean_work_l2_ratio=("patch_minus_probe_fraction", "mean"),
            mean_functional_residual_fixed_ratio=("functional_residual_fixed_ratio", "mean"),
            median_functional_residual_fixed_ratio=("functional_residual_fixed_ratio", "median"),
            mean_functional_residual_moving_ratio=("functional_residual_moving_ratio", "mean"),
            median_functional_residual_moving_ratio=("functional_residual_moving_ratio", "median"),
            mean_numeric_logit_residual_fixed_ratio=("numeric_logit_residual_fixed_ratio", "mean"),
            median_numeric_logit_residual_fixed_ratio=("numeric_logit_residual_fixed_ratio", "median"),
            mean_numeric_logit_residual_moving_ratio=("numeric_logit_residual_moving_ratio", "mean"),
            median_numeric_logit_residual_moving_ratio=("numeric_logit_residual_moving_ratio", "median"),
            mean_fisher_baseline_fixed_ratio=("fisher_baseline_fixed_ratio", "mean"),
            median_fisher_baseline_fixed_ratio=("fisher_baseline_fixed_ratio", "median"),
            mean_fisher_counterfactual_fixed_ratio=("fisher_counterfactual_fixed_ratio", "mean"),
            median_fisher_counterfactual_fixed_ratio=("fisher_counterfactual_fixed_ratio", "median"),
            mean_fisher_midpoint_fixed_ratio=("fisher_midpoint_fixed_ratio", "mean"),
            median_fisher_midpoint_fixed_ratio=("fisher_midpoint_fixed_ratio", "median"),
            mean_fisher_path_fixed_ratio=("fisher_path_fixed_ratio", "mean"),
            median_fisher_path_fixed_ratio=("fisher_path_fixed_ratio", "median"),
        )
        .sort_values([column for column in ["family", "evaluation_split", "task", "architecture", "edit_family"] if column in summary_group_cols])
        .reset_index(drop=True)
    )
    for slug in kernel_metric_payloads:
        kernel_summary = (
            detail.groupby(summary_group_cols, as_index=False, dropna=False)
            .agg(
                **{
                    f"mean_kernel_sigma_{slug}_utility_gain": (f"kernel_sigma_{slug}_utility_gain", "mean"),
                    f"mean_kernel_sigma_{slug}_full_utility_gain": (f"kernel_sigma_{slug}_full_utility_gain", "mean"),
                    f"mean_kernel_sigma_{slug}_fixed_ratio": (f"kernel_sigma_{slug}_fixed_ratio", "mean"),
                    f"median_kernel_sigma_{slug}_fixed_ratio": (f"kernel_sigma_{slug}_fixed_ratio", "median"),
                    f"mean_kernel_sigma_{slug}_moving_ratio": (f"kernel_sigma_{slug}_moving_ratio", "mean"),
                }
            )
        )
        summary = summary.merge(kernel_summary, on=summary_group_cols, how="left", validate="one_to_one")
    pair_group_cols = [
        column
        for column in ["family", "family_label", "architecture", "edit_family", "matched_to_architecture", "evaluation_split", "task", "operand_mode"]
        if column in detail.columns
    ]
    pair_summary = (
        detail.groupby(pair_group_cols, as_index=False, dropna=False)
        .agg(
            num_examples=("analysis_example_id", "size"),
            mean_objective_gain=("objective_gain", "mean"),
            mean_target_prob_gain=("target_prob_gain", "mean"),
            mean_numeric_mass_gain=("numeric_mass_gain", "mean"),
            mean_target_prob_numeric_gain=("target_prob_numeric_gain", "mean"),
            mean_grad_norm_ratio=("grad_norm_ratio", "mean"),
            mean_numeric_logit_residual_fixed_ratio=("numeric_logit_residual_fixed_ratio", "mean"),
            mean_fisher_path_fixed_ratio=("fisher_path_fixed_ratio", "mean"),
        )
        .sort_values([column for column in ["family", "evaluation_split", "task", "operand_mode", "architecture", "edit_family"] if column in pair_group_cols])
        .reset_index(drop=True)
    )
    for slug in kernel_metric_payloads:
        kernel_pair_summary = (
            detail.groupby(pair_group_cols, as_index=False, dropna=False)
            .agg(**{f"mean_kernel_sigma_{slug}_fixed_ratio": (f"kernel_sigma_{slug}_fixed_ratio", "mean")})
        )
        pair_summary = pair_summary.merge(kernel_pair_summary, on=pair_group_cols, how="left", validate="one_to_one")
    return detail, summary, pair_summary
