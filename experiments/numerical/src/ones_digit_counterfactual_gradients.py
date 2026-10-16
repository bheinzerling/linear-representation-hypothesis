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
from src.ones_digit_intervention import (
    OPERAND_MODES,
    TASKS,
    OnesDigitProbe,
    _numeric_tracking_arrays,
    build_pair_intervention_frame,
    canonical_probe_architecture_name,
)
from src.steering_sanity import build_shifted_prompts, orthonormal_row_basis, random_unit_rows


def _probe_from_payload(probe_payload: dict[str, Any]) -> OnesDigitProbe:
    return OnesDigitProbe(
        architecture=canonical_probe_architecture_name(str(probe_payload["architecture"])),
        coefficients=np.asarray(probe_payload["coefficients"], dtype=np.float64),
        bias=np.asarray(probe_payload["bias"], dtype=np.float64),
        decoder_matrix=np.asarray(probe_payload["decoder_matrix"], dtype=np.float64),
        support_digits=np.asarray(probe_payload["support_digits"], dtype=np.int64),
        support_features=np.asarray(probe_payload["support_features"], dtype=np.float64),
        metadata=dict(probe_payload.get("metadata", {})),
    )


def load_probe_models(path: str | Path) -> dict[str, OnesDigitProbe]:
    payload = json.loads(Path(path).read_text())
    return {
        canonical_probe_architecture_name(str(architecture)): _probe_from_payload(dict(probe_payload))
        for architecture, probe_payload in payload.items()
    }


def load_probe_models_by_layer(path: str | Path) -> dict[int, dict[str, OnesDigitProbe]]:
    payload = json.loads(Path(path).read_text())
    probes_by_layer: dict[int, dict[str, OnesDigitProbe]] = {}
    for layer_key, layer_payload in payload.items():
        probes_by_layer[int(layer_key)] = {
            canonical_probe_architecture_name(str(architecture)): _probe_from_payload(dict(probe_payload))
            for architecture, probe_payload in dict(layer_payload).items()
        }
    return probes_by_layer


def intervention_family_basis(probe: OnesDigitProbe) -> np.ndarray:
    support_digits = np.asarray(probe.support_digits, dtype=np.int64)
    deltas: list[np.ndarray] = []
    for source_digit in support_digits.tolist():
        for target_digit in support_digits.tolist():
            if int(source_digit) == int(target_digit):
                continue
            delta = probe.intervention_delta(
                source_digits=np.asarray([source_digit], dtype=np.int64),
                target_digits=np.asarray([target_digit], dtype=np.int64),
            )[0]
            deltas.append(np.asarray(delta, dtype=np.float64))
    return orthonormal_row_basis(np.stack(deltas, axis=0))


def _project_onto_basis(rows: np.ndarray, basis: np.ndarray) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    return (array @ basis.T) @ basis


def _row_norms(rows: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(rows, dtype=np.float64), axis=1)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    den = np.asarray(denominator, dtype=np.float64)
    num = np.asarray(numerator, dtype=np.float64)
    return num / np.where(den > 1e-12, den, np.nan)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    dots = np.sum(a_arr * b_arr, axis=1)
    denom = _row_norms(a_arr) * _row_norms(b_arr)
    return dots / np.where(denom > 1e-12, denom, np.nan)


def _best_scalar_projection(reference: np.ndarray, direction: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    dir_arr = np.asarray(direction, dtype=np.float64)
    numer = np.sum(ref * dir_arr, axis=1)
    denom = np.sum(dir_arr * dir_arr, axis=1)
    return numer / np.where(denom > 1e-12, denom, np.nan)


def make_norm_matched_random_deltas(
    reference_rows: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    ref = np.asarray(reference_rows, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    unit = random_unit_rows(rng, ref.shape)
    return unit * _row_norms(ref)[:, None]


def _family_projection_metrics(
    *,
    patch_deltas: np.ndarray,
    family_basis: np.ndarray | None,
    grad_orig: np.ndarray,
    grad_steered: np.ndarray,
) -> dict[str, np.ndarray]:
    patch = np.asarray(patch_deltas, dtype=np.float64)
    grad_pre = np.asarray(grad_orig, dtype=np.float64)
    grad_post = np.asarray(grad_steered, dtype=np.float64)
    row_count = len(patch)
    if family_basis is None:
        nan_rows = np.full(row_count, np.nan, dtype=np.float64)
        return {
            "patch_explained_fraction": nan_rows.copy(),
            "patch_orthogonal_fraction": nan_rows.copy(),
            "grad_orig_span_norm": nan_rows.copy(),
            "grad_steered_span_norm": nan_rows.copy(),
            "grad_span_ratio": nan_rows.copy(),
            "grad_orig_orth_norm": nan_rows.copy(),
            "grad_steered_orth_norm": nan_rows.copy(),
            "grad_orth_ratio": nan_rows.copy(),
            "grad_orig_span_fraction": nan_rows.copy(),
            "grad_steered_span_fraction": nan_rows.copy(),
        }

    basis = np.asarray(family_basis, dtype=np.float64)
    patch_proj = _project_onto_basis(patch, basis)
    patch_orth = patch - patch_proj
    grad_orig_proj = _project_onto_basis(grad_pre, basis)
    grad_orig_orth = grad_pre - grad_orig_proj
    grad_steered_proj = _project_onto_basis(grad_post, basis)
    grad_steered_orth = grad_post - grad_steered_proj

    patch_norm = _row_norms(patch)
    patch_proj_norm = _row_norms(patch_proj)
    patch_orth_norm = _row_norms(patch_orth)
    grad_orig_norm = _row_norms(grad_pre)
    grad_steered_norm = _row_norms(grad_post)
    grad_orig_proj_norm = _row_norms(grad_orig_proj)
    grad_orig_orth_norm = _row_norms(grad_orig_orth)
    grad_steered_proj_norm = _row_norms(grad_steered_proj)
    grad_steered_orth_norm = _row_norms(grad_steered_orth)
    return {
        "patch_explained_fraction": _safe_ratio(patch_proj_norm, patch_norm).astype(np.float64),
        "patch_orthogonal_fraction": _safe_ratio(patch_orth_norm, patch_norm).astype(np.float64),
        "grad_orig_span_norm": grad_orig_proj_norm.astype(np.float64),
        "grad_steered_span_norm": grad_steered_proj_norm.astype(np.float64),
        "grad_span_ratio": _safe_ratio(grad_steered_proj_norm, grad_orig_proj_norm).astype(np.float64),
        "grad_orig_orth_norm": grad_orig_orth_norm.astype(np.float64),
        "grad_steered_orth_norm": grad_steered_orth_norm.astype(np.float64),
        "grad_orth_ratio": _safe_ratio(grad_steered_orth_norm, grad_orig_orth_norm).astype(np.float64),
        "grad_orig_span_fraction": _safe_ratio(grad_orig_proj_norm, grad_orig_norm).astype(np.float64),
        "grad_steered_span_fraction": _safe_ratio(grad_steered_proj_norm, grad_steered_norm).astype(np.float64),
    }


def functional_residual_metrics(
    *,
    patch_deltas: np.ndarray,
    probe_deltas: np.ndarray,
    grad_orig: np.ndarray,
    grad_steered: np.ndarray,
) -> dict[str, np.ndarray]:
    patch = np.asarray(patch_deltas, dtype=np.float64)
    probe = np.asarray(probe_deltas, dtype=np.float64)
    grad_pre = np.asarray(grad_orig, dtype=np.float64)
    grad_post = np.asarray(grad_steered, dtype=np.float64)
    residual_post = patch - probe
    pre_signed = np.sum(grad_pre * patch, axis=1)
    post_fixed_signed = np.sum(grad_pre * residual_post, axis=1)
    post_moving_signed = np.sum(grad_post * residual_post, axis=1)
    pre_abs = np.abs(pre_signed)
    post_fixed_abs = np.abs(post_fixed_signed)
    post_moving_abs = np.abs(post_moving_signed)
    return {
        "functional_residual_pre_signed": pre_signed.astype(np.float64),
        "functional_residual_post_fixed_signed": post_fixed_signed.astype(np.float64),
        "functional_residual_post_moving_signed": post_moving_signed.astype(np.float64),
        "functional_residual_pre": pre_abs.astype(np.float64),
        "functional_residual_post_fixed": post_fixed_abs.astype(np.float64),
        "functional_residual_post_moving": post_moving_abs.astype(np.float64),
        "functional_residual_fixed_ratio": _safe_ratio(post_fixed_abs, pre_abs).astype(np.float64),
        "functional_residual_moving_ratio": _safe_ratio(post_moving_abs, pre_abs).astype(np.float64),
    }


def numeric_logit_residual_metrics(
    *,
    pre_directional_logits: np.ndarray,
    post_fixed_directional_logits: np.ndarray,
    post_moving_directional_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    pre = _row_norms(pre_directional_logits)
    post_fixed = _row_norms(post_fixed_directional_logits)
    post_moving = _row_norms(post_moving_directional_logits)
    return {
        "numeric_logit_residual_pre": pre.astype(np.float64),
        "numeric_logit_residual_post_fixed": post_fixed.astype(np.float64),
        "numeric_logit_residual_post_moving": post_moving.astype(np.float64),
        "numeric_logit_residual_fixed_ratio": _safe_ratio(post_fixed, pre).astype(np.float64),
        "numeric_logit_residual_moving_ratio": _safe_ratio(post_moving, pre).astype(np.float64),
    }



def build_counterfactual_prompt_frame(*, valid_df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for operand_mode, mode_df in valid_df.groupby("operand_mode", sort=False):
        shifted = build_shifted_prompts(mode_df, tokenizer=tokenizer, operand_mode=str(operand_mode))
        merged = mode_df[["analysis_example_id", "example_id"]].merge(shifted, on="example_id", how="left")
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=["analysis_example_id", "example_id", "shifted_prompt", "shifted_token_idx"])
    return pd.concat(frames, ignore_index=True).sort_values("analysis_example_id").reset_index(drop=True)


def _tracked_logits(logits: np.ndarray, tracked_token_ids: np.ndarray | None) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float64)
    if tracked_token_ids is None:
        return logits_array.astype(np.float64)
    tracked = np.asarray(tracked_token_ids, dtype=np.int64)
    return logits_array[:, tracked].astype(np.float64)


def _tracked_probs_from_logits(logits: np.ndarray, tracked_token_ids: np.ndarray | None) -> np.ndarray:
    return _softmax_numpy(_tracked_logits(logits, tracked_token_ids)).astype(np.float64)


def _fisher_quadratic_from_probs(direction_rows: np.ndarray, probs: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction_rows, dtype=np.float64)
    prob_array = np.asarray(probs, dtype=np.float64)
    second_moment = np.sum(prob_array * np.square(direction), axis=1)
    first_moment = np.square(np.sum(prob_array * direction, axis=1))
    return np.maximum(second_moment - first_moment, 0.0).astype(np.float64)


def fisher_numeric_logit_residual_metrics(
    *,
    pre_directional_logits: np.ndarray,
    post_fixed_directional_logits: np.ndarray,
    post_moving_directional_logits: np.ndarray,
    baseline_logits: np.ndarray,
    counterfactual_logits: np.ndarray,
    tracked_token_ids: np.ndarray | None,
    path_num_points: int,
) -> dict[str, np.ndarray]:
    baseline_probs = _tracked_probs_from_logits(baseline_logits, tracked_token_ids)
    counterfactual_probs = _tracked_probs_from_logits(counterfactual_logits, tracked_token_ids)
    tracked_baseline_logits = _tracked_logits(baseline_logits, tracked_token_ids)
    tracked_counterfactual_logits = _tracked_logits(counterfactual_logits, tracked_token_ids)

    variants: dict[str, np.ndarray] = {
        "baseline": _fisher_quadratic_from_probs(pre_directional_logits, baseline_probs),
        "counterfactual": _fisher_quadratic_from_probs(pre_directional_logits, counterfactual_probs),
        "midpoint": 0.5 * (
            _fisher_quadratic_from_probs(pre_directional_logits, baseline_probs)
            + _fisher_quadratic_from_probs(pre_directional_logits, counterfactual_probs)
        ),
    }

    num_points = max(int(path_num_points), 2)
    path_quadratic_pre = np.zeros(len(pre_directional_logits), dtype=np.float64)
    path_quadratic_post_fixed = np.zeros(len(pre_directional_logits), dtype=np.float64)
    path_quadratic_post_moving = np.zeros(len(pre_directional_logits), dtype=np.float64)
    for alpha in np.linspace(0.0, 1.0, num_points):
        mixed_logits = ((1.0 - float(alpha)) * tracked_baseline_logits) + (float(alpha) * tracked_counterfactual_logits)
        mixed_probs = _softmax_numpy(mixed_logits).astype(np.float64)
        path_quadratic_pre += _fisher_quadratic_from_probs(pre_directional_logits, mixed_probs)
        path_quadratic_post_fixed += _fisher_quadratic_from_probs(post_fixed_directional_logits, mixed_probs)
        path_quadratic_post_moving += _fisher_quadratic_from_probs(post_moving_directional_logits, mixed_probs)
    variants["path"] = path_quadratic_pre / float(num_points)

    metrics: dict[str, np.ndarray] = {}
    for variant_name, pre_quadratic in variants.items():
        if variant_name == "baseline":
            post_fixed_quadratic = _fisher_quadratic_from_probs(post_fixed_directional_logits, baseline_probs)
            post_moving_quadratic = _fisher_quadratic_from_probs(post_moving_directional_logits, baseline_probs)
        elif variant_name == "counterfactual":
            post_fixed_quadratic = _fisher_quadratic_from_probs(post_fixed_directional_logits, counterfactual_probs)
            post_moving_quadratic = _fisher_quadratic_from_probs(post_moving_directional_logits, counterfactual_probs)
        elif variant_name == "midpoint":
            post_fixed_quadratic = 0.5 * (
                _fisher_quadratic_from_probs(post_fixed_directional_logits, baseline_probs)
                + _fisher_quadratic_from_probs(post_fixed_directional_logits, counterfactual_probs)
            )
            post_moving_quadratic = 0.5 * (
                _fisher_quadratic_from_probs(post_moving_directional_logits, baseline_probs)
                + _fisher_quadratic_from_probs(post_moving_directional_logits, counterfactual_probs)
            )
        else:
            post_fixed_quadratic = path_quadratic_post_fixed / float(num_points)
            post_moving_quadratic = path_quadratic_post_moving / float(num_points)

        pre = np.sqrt(np.maximum(pre_quadratic, 0.0)).astype(np.float64)
        post_fixed = np.sqrt(np.maximum(post_fixed_quadratic, 0.0)).astype(np.float64)
        post_moving = np.sqrt(np.maximum(post_moving_quadratic, 0.0)).astype(np.float64)
        metrics[f"fisher_{variant_name}_pre"] = pre
        metrics[f"fisher_{variant_name}_post_fixed"] = post_fixed
        metrics[f"fisher_{variant_name}_post_moving"] = post_moving
        metrics[f"fisher_{variant_name}_fixed_ratio"] = _safe_ratio(post_fixed, pre).astype(np.float64)
        metrics[f"fisher_{variant_name}_moving_ratio"] = _safe_ratio(post_moving, pre).astype(np.float64)
    return metrics


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
    frame, _ = _objective_frame_with_logits(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        objective=objective,
        delta_rows=delta_rows,
    )
    return frame


def _objective_frame_with_logits(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    delta_rows: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    frame = pd.DataFrame(
        {
            "objective_score": objective_score.astype(np.float64),
            "target_prob": probs[rows, target_token_ids].astype(np.float64),
            "baseline_answer_prob": probs[rows, baseline_token_ids].astype(np.float64),
        }
    )
    return frame, np.asarray(logits, dtype=np.float64)


def sample_pair_intervention_examples(
    *,
    eligible_df: pd.DataFrame,
    tokenizer,
    evaluation_splits: list[str],
    tasks: list[str],
    operand_modes: list[str],
    max_examples_per_pair: int,
    seed: int,
    digit_place: str = "ones",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for evaluation_split in evaluation_splits:
        split_df = eligible_df.loc[eligible_df["evaluation_split"] == str(evaluation_split)].copy()
        if split_df.empty:
            continue
        for task in tasks:
            if task not in TASKS:
                raise ValueError(f"Unsupported task: {task!r}")
            task_df = split_df.loc[split_df["task"] == str(task)].copy()
            if task_df.empty:
                continue
            for operand_mode in operand_modes:
                if operand_mode not in OPERAND_MODES:
                    raise ValueError(f"Unsupported operand_mode: {operand_mode!r}")
                mode_df = task_df.loc[task_df["operand_mode"] == str(operand_mode)].copy()
                if mode_df.empty:
                    continue
                for source_digit in range(10):
                    source_df = mode_df.loc[mode_df["source_digit"] == int(source_digit)].copy()
                    if source_df.empty:
                        continue
                    for target_digit in range(10):
                        if target_digit == source_digit:
                            continue
                        valid_df = build_pair_intervention_frame(
                            eligible_df=source_df,
                            tokenizer=tokenizer,
                            source_digit=int(source_digit),
                            target_digit=int(target_digit),
                            digit_place=str(digit_place),
                        )
                        if valid_df.empty:
                            continue
                        pair_seed = (
                            int(seed)
                            + (10007 * int(source_digit))
                            + (1009 * int(target_digit))
                            + (97 * len(str(task)))
                            + (11 if str(operand_mode) == "x2" else 0)
                            + (1 if str(evaluation_split) == "test" else 0)
                        )
                        rng = np.random.default_rng(pair_seed)
                        sample_size = min(int(max_examples_per_pair), len(valid_df))
                        take = np.sort(rng.choice(len(valid_df), size=sample_size, replace=False))
                        sampled = valid_df.iloc[take].copy().reset_index(drop=True)
                        sampled["pair_sample_rank"] = np.arange(len(sampled), dtype=np.int64)
                        rows.append(sampled)
    if not rows:
        return pd.DataFrame()
    sampled_df = pd.concat(rows, ignore_index=True)
    sampled_df = sampled_df.sort_values(
        [
            "evaluation_split",
            "task",
            "operand_mode",
            "source_digit",
            "target_digit",
            "pair_sample_rank",
            "source_value",
            "x1",
            "x2",
        ]
    ).reset_index(drop=True)
    sampled_df["analysis_example_id"] = np.arange(len(sampled_df), dtype=np.int64)
    sampled_df["example_id"] = sampled_df["analysis_example_id"]
    return sampled_df


def analyze_counterfactual_deltas(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    architecture: str,
    delta_rows: np.ndarray,
    family_basis: np.ndarray | None,
    unscaled_direction_rows: np.ndarray,
    edit_family: str,
    matched_to_architecture: str | None = None,
    functional_objective: str = "target_margin_logit",
    fisher_path_num_points: int = 9,
    fisher_output_space: str = "tracked_numeric",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if valid_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if str(fisher_output_space) not in {"tracked_numeric", "full_vocab"}:
        raise ValueError(f"Unsupported fisher_output_space: {fisher_output_space!r}")

    _, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
    probe_deltas = np.asarray(delta_rows, dtype=np.float64)
    unscaled_probe_deltas = np.asarray(unscaled_direction_rows, dtype=np.float64)
    patch_rows = []
    for _, mode_df in valid_df.groupby("operand_mode", sort=False):
        mode_patch = compute_counterfactual_patch_deltas(
            model=model,
            tokenizer=tokenizer,
            valid_df=mode_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
        )
        patch_rows.append(
            pd.DataFrame(
                {
                    "analysis_example_id": mode_df["analysis_example_id"].to_numpy(dtype=np.int64),
                    "patch_delta": list(np.asarray(mode_patch, dtype=np.float64)),
                }
            )
        )
    patch_frame = pd.concat(patch_rows, ignore_index=True).set_index("analysis_example_id")
    patch_order = valid_df["analysis_example_id"].to_numpy(dtype=np.int64)
    patch_deltas = np.stack(patch_frame.loc[patch_order, "patch_delta"].to_numpy(), axis=0).astype(np.float64)

    analysis_grad_orig = collect_gradient_directions(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=objective,
        delta_rows=None,
    )
    analysis_grad_steered = collect_gradient_directions(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=objective,
        delta_rows=probe_deltas,
    )
    if functional_objective == objective:
        functional_grad_orig = analysis_grad_orig
        functional_grad_steered = analysis_grad_steered
    else:
        functional_grad_orig = collect_gradient_directions(
            model=model,
            tokenizer=tokenizer,
            valid_df=valid_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
            objective=functional_objective,
            delta_rows=None,
        )
        functional_grad_steered = collect_gradient_directions(
            model=model,
            tokenizer=tokenizer,
            valid_df=valid_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
            objective=functional_objective,
            delta_rows=probe_deltas,
        )

    objective_orig, baseline_logits = _objective_frame_with_logits(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=objective,
        delta_rows=None,
    )
    objective_steered, _ = _objective_frame_with_logits(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        objective=objective,
        delta_rows=probe_deltas,
    )
    counterfactual_df = build_counterfactual_prompt_frame(valid_df=valid_df, tokenizer=tokenizer)
    counterfactual_logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=counterfactual_df["shifted_prompt"].tolist(),
        token_idxs=counterfactual_df["shifted_token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        delta_rows=None,
        capture_selected=False,
    )

    patch_norm = _row_norms(patch_deltas)
    probe_norm = _row_norms(probe_deltas)
    grad_orig_norm = _row_norms(analysis_grad_orig)
    grad_steered_norm = _row_norms(analysis_grad_steered)
    family_metrics = _family_projection_metrics(
        patch_deltas=patch_deltas,
        family_basis=family_basis,
        grad_orig=analysis_grad_orig,
        grad_steered=analysis_grad_steered,
    )

    best_alpha = _best_scalar_projection(patch_deltas, unscaled_probe_deltas)
    best_alpha = np.nan_to_num(best_alpha, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_best_probe = unscaled_probe_deltas * best_alpha[:, None]
    residual_fixed = patch_deltas - probe_deltas
    residual_best = patch_deltas - scaled_best_probe
    functional_metrics = functional_residual_metrics(
        patch_deltas=patch_deltas,
        probe_deltas=probe_deltas,
        grad_orig=functional_grad_orig,
        grad_steered=functional_grad_steered,
    )
    pre_directional_logits = collect_numeric_logit_directional_derivatives(
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
    post_fixed_directional_logits = collect_numeric_logit_directional_derivatives(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        tracked_token_ids=tracked_token_ids,
        direction_rows=residual_fixed,
        delta_rows=None,
    )
    post_moving_directional_logits = collect_numeric_logit_directional_derivatives(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        tracked_token_ids=tracked_token_ids,
        direction_rows=residual_fixed,
        delta_rows=probe_deltas,
    )
    numeric_logit_metrics = numeric_logit_residual_metrics(
        pre_directional_logits=pre_directional_logits,
        post_fixed_directional_logits=post_fixed_directional_logits,
        post_moving_directional_logits=post_moving_directional_logits,
    )
    if str(fisher_output_space) == "full_vocab":
        fisher_tracked_token_ids: np.ndarray | None = None
        fisher_pre_directional_logits = collect_numeric_logit_directional_derivatives(
            model=model,
            tokenizer=tokenizer,
            valid_df=valid_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
            tracked_token_ids=None,
            direction_rows=patch_deltas,
            delta_rows=None,
        )
        fisher_post_fixed_directional_logits = collect_numeric_logit_directional_derivatives(
            model=model,
            tokenizer=tokenizer,
            valid_df=valid_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
            tracked_token_ids=None,
            direction_rows=residual_fixed,
            delta_rows=None,
        )
        fisher_post_moving_directional_logits = collect_numeric_logit_directional_derivatives(
            model=model,
            tokenizer=tokenizer,
            valid_df=valid_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
            tracked_token_ids=None,
            direction_rows=residual_fixed,
            delta_rows=probe_deltas,
        )
    else:
        fisher_tracked_token_ids = tracked_token_ids
        fisher_pre_directional_logits = pre_directional_logits
        fisher_post_fixed_directional_logits = post_fixed_directional_logits
        fisher_post_moving_directional_logits = post_moving_directional_logits
    fisher_metrics = fisher_numeric_logit_residual_metrics(
        pre_directional_logits=fisher_pre_directional_logits,
        post_fixed_directional_logits=fisher_post_fixed_directional_logits,
        post_moving_directional_logits=fisher_post_moving_directional_logits,
        baseline_logits=np.asarray(baseline_logits, dtype=np.float64),
        counterfactual_logits=np.asarray(counterfactual_logits, dtype=np.float64),
        tracked_token_ids=fisher_tracked_token_ids,
        path_num_points=int(fisher_path_num_points),
    )

    detail = valid_df.copy().reset_index(drop=True)
    detail["architecture"] = str(architecture)
    detail["edit_family"] = str(edit_family)
    detail["matched_to_architecture"] = (
        np.full(len(detail), str(matched_to_architecture), dtype=object)
        if matched_to_architecture is not None
        else np.full(len(detail), np.nan, dtype=object)
    )
    detail["objective"] = str(objective)
    detail["functional_objective"] = str(functional_objective)
    detail["fisher_output_space"] = str(fisher_output_space)
    detail["probe_alpha"] = 1.0
    detail["probe_delta_norm"] = probe_norm.astype(np.float64)
    detail["patch_delta_norm"] = patch_norm.astype(np.float64)
    detail["patch_probe_cosine"] = _row_cosine(patch_deltas, probe_deltas).astype(np.float64)
    detail["patch_explained_fraction"] = family_metrics["patch_explained_fraction"]
    detail["patch_orthogonal_fraction"] = family_metrics["patch_orthogonal_fraction"]
    detail["patch_minus_probe_norm"] = _row_norms(residual_fixed).astype(np.float64)
    detail["patch_minus_probe_fraction"] = _safe_ratio(_row_norms(residual_fixed), patch_norm).astype(np.float64)
    detail["patch_best_alpha"] = best_alpha.astype(np.float64)
    detail["patch_minus_best_probe_norm"] = _row_norms(residual_best).astype(np.float64)
    detail["patch_minus_best_probe_fraction"] = _safe_ratio(_row_norms(residual_best), patch_norm).astype(np.float64)
    detail["objective_orig"] = objective_orig["objective_score"].to_numpy(dtype=np.float64)
    detail["objective_steered"] = objective_steered["objective_score"].to_numpy(dtype=np.float64)
    detail["objective_gain"] = (detail["objective_steered"] - detail["objective_orig"]).astype(np.float64)
    detail["target_prob_orig"] = objective_orig["target_prob"].to_numpy(dtype=np.float64)
    detail["target_prob_steered"] = objective_steered["target_prob"].to_numpy(dtype=np.float64)
    detail["target_prob_gain"] = (detail["target_prob_steered"] - detail["target_prob_orig"]).astype(np.float64)
    detail["baseline_answer_prob_orig"] = objective_orig["baseline_answer_prob"].to_numpy(dtype=np.float64)
    detail["baseline_answer_prob_steered"] = objective_steered["baseline_answer_prob"].to_numpy(dtype=np.float64)
    detail["grad_orig_norm"] = grad_orig_norm.astype(np.float64)
    detail["grad_steered_norm"] = grad_steered_norm.astype(np.float64)
    detail["grad_norm_ratio"] = _safe_ratio(grad_steered_norm, grad_orig_norm).astype(np.float64)
    for column_name, values in family_metrics.items():
        if column_name.startswith("patch_"):
            continue
        detail[column_name] = values
    detail["work_l2_pre_norm"] = patch_norm.astype(np.float64)
    detail["work_l2_post_norm"] = _row_norms(residual_fixed).astype(np.float64)
    detail["work_l2_ratio"] = _safe_ratio(detail["work_l2_post_norm"], detail["work_l2_pre_norm"]).astype(np.float64)
    for column_name, values in functional_metrics.items():
        detail[column_name] = values
    for column_name, values in numeric_logit_metrics.items():
        detail[column_name] = values
    for column_name, values in fisher_metrics.items():
        detail[column_name] = values

    summary = (
        detail.groupby(["architecture", "edit_family", "matched_to_architecture", "evaluation_split", "task", "fisher_output_space"], as_index=False, dropna=False)
        .agg(
            num_examples=("analysis_example_id", "size"),
            mean_objective_gain=("objective_gain", "mean"),
            mean_target_prob_gain=("target_prob_gain", "mean"),
            mean_grad_norm_ratio=("grad_norm_ratio", "mean"),
            median_grad_norm_ratio=("grad_norm_ratio", "median"),
            mean_grad_orth_ratio=("grad_orth_ratio", "mean"),
            mean_patch_probe_cosine=("patch_probe_cosine", "mean"),
            mean_patch_explained_fraction=("patch_explained_fraction", "mean"),
            mean_patch_minus_probe_fraction=("patch_minus_probe_fraction", "mean"),
            mean_patch_minus_best_probe_fraction=("patch_minus_best_probe_fraction", "mean"),
            mean_patch_best_alpha=("patch_best_alpha", "mean"),
            mean_work_l2_ratio=("work_l2_ratio", "mean"),
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
        .sort_values(["evaluation_split", "task", "edit_family", "architecture"])
        .reset_index(drop=True)
    )

    pair_summary = (
        detail.groupby(
            ["architecture", "edit_family", "matched_to_architecture", "evaluation_split", "task", "source_digit", "target_digit", "fisher_output_space"],
            as_index=False,
            dropna=False,
        )
        .agg(
            num_examples=("analysis_example_id", "size"),
            mean_objective_gain=("objective_gain", "mean"),
            mean_target_prob_gain=("target_prob_gain", "mean"),
            mean_grad_norm_ratio=("grad_norm_ratio", "mean"),
            mean_grad_orth_ratio=("grad_orth_ratio", "mean"),
            mean_patch_probe_cosine=("patch_probe_cosine", "mean"),
            mean_patch_explained_fraction=("patch_explained_fraction", "mean"),
            mean_patch_minus_probe_fraction=("patch_minus_probe_fraction", "mean"),
            mean_patch_minus_best_probe_fraction=("patch_minus_best_probe_fraction", "mean"),
            mean_patch_best_alpha=("patch_best_alpha", "mean"),
            mean_work_l2_ratio=("work_l2_ratio", "mean"),
            mean_functional_residual_fixed_ratio=("functional_residual_fixed_ratio", "mean"),
            mean_functional_residual_moving_ratio=("functional_residual_moving_ratio", "mean"),
            mean_numeric_logit_residual_fixed_ratio=("numeric_logit_residual_fixed_ratio", "mean"),
            mean_numeric_logit_residual_moving_ratio=("numeric_logit_residual_moving_ratio", "mean"),
            mean_fisher_baseline_fixed_ratio=("fisher_baseline_fixed_ratio", "mean"),
            mean_fisher_counterfactual_fixed_ratio=("fisher_counterfactual_fixed_ratio", "mean"),
            mean_fisher_midpoint_fixed_ratio=("fisher_midpoint_fixed_ratio", "mean"),
            mean_fisher_path_fixed_ratio=("fisher_path_fixed_ratio", "mean"),
        )
        .sort_values(["evaluation_split", "task", "edit_family", "architecture", "source_digit", "target_digit"])
        .reset_index(drop=True)
    )
    return detail, summary, pair_summary


def analyze_counterfactual_gradients(
    *,
    model,
    tokenizer,
    probe: OnesDigitProbe,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    functional_objective: str = "target_margin_logit",
    probe_alpha: float = 1.0,
    fisher_path_num_points: int = 9,
    fisher_output_space: str = "tracked_numeric",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unscaled_probe_deltas = probe.intervention_delta(
        source_digits=valid_df["source_digit"].to_numpy(dtype=np.int64),
        target_digits=valid_df["target_digit"].to_numpy(dtype=np.int64),
    )
    return analyze_counterfactual_deltas(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        objective=objective,
        architecture=str(probe.architecture),
        delta_rows=np.asarray(unscaled_probe_deltas * float(probe_alpha), dtype=np.float64),
        family_basis=intervention_family_basis(probe),
        unscaled_direction_rows=np.asarray(unscaled_probe_deltas, dtype=np.float64),
        edit_family="probe",
        matched_to_architecture=str(probe.architecture),
        functional_objective=functional_objective,
        fisher_path_num_points=int(fisher_path_num_points),
        fisher_output_space=str(fisher_output_space),
    )
