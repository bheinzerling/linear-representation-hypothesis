#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.linalg import null_space

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_value_probe_capacity_sweep import _collect_partition_activations, _load_partition_rows
from src.activation_steering import _parse_optional_int, _softmax_numpy
from src.gradient_steering import (
    forward_next_token_logits_torch_with_delta,
    forward_next_token_logits_with_delta,
    freeze_model_parameters,
    objective_scores_numpy,
    objective_scores_torch,
    top_patch_subspace_basis,
)
from src.identity_prompting import render_arithmetic_prompt_from_row, render_identity_prompt
from src.magnitude_counterfactual_gradients import intervention_family_basis
from src.magnitude_intervention import fit_magnitude_probe
from src.model_loader import load_causal_lm, resolve_model_revision
from src.ones_digit_intervention import _numeric_tracking_arrays
from src.probe_manifold_paths import probe_path_deltas
from src.rfm_agop import fit_agop_steering_family
from src.source_value_steering import (
    DEFAULT_DISTANCE_BUCKETS,
    compute_projected_patch_probe_delta,
    compute_value_code_probe_delta,
    distance_bucket_label,
    sample_stratified_target_values,
    shifted_operands_and_target_for_row,
    summarize_steering_metrics,
    valid_source_value_targets_for_row,
)
from src.steering_sanity import evaluate_logits_against_shifted_targets
from src.token_utils import build_single_token_report, encode_integer
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text, dump_run_config, ensure_dir, load_yaml, set_global_seed


DEFAULT_CONFIG = ROOT / "configs" / "source_value_steering_addition_formula_llama_3_1_8b_fixed_controls.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-value probe-manifold steering on canonical prompt-cache splits.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default="out/runs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _bucket_specs_from_config(cfg: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    bucket_rows = cfg.get("distance_buckets")
    if bucket_rows is None:
        return DEFAULT_DISTANCE_BUCKETS
    return tuple((int(row[0]), int(row[1])) for row in bucket_rows)


def _resolve_model_entry(models_payload: dict[str, Any], model_alias: str) -> dict[str, Any]:
    defaults = dict(models_payload.get("default", {}))
    entry = dict(models_payload["models"][model_alias])
    resolved = {**defaults, **entry}
    if "hf_id" not in resolved:
        raise KeyError(f"Model alias {model_alias!r} is missing 'hf_id'.")
    return resolved


def _build_baseline_frame(
    *,
    tokenizer,
    base_rows: pd.DataFrame,
    logits: np.ndarray,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    probs = _softmax_numpy(np.asarray(logits, dtype=np.float64))
    pred_ids = probs.argmax(axis=-1).astype(np.int64)
    pred_text = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in pred_ids.tolist()]
    pred_value = [_parse_optional_int(text) for text in pred_text]
    numeric_probs = probs[:, tracked_token_ids]
    numeric_mass = numeric_probs.sum(axis=1)
    expected = (numeric_probs * tracked_values.astype(np.float64)[None, :]).sum(axis=1) / np.clip(numeric_mass, 1e-8, None)
    target_ids = base_rows["target_token_id"].to_numpy(dtype=np.int64)
    baseline = base_rows.copy()
    baseline["baseline_row_idx"] = np.arange(len(baseline), dtype=np.int64)
    baseline["baseline_predicted_token_id"] = pred_ids
    baseline["baseline_predicted_text"] = pred_text
    baseline["baseline_predicted_value"] = pred_value
    baseline["baseline_expected_value"] = expected.astype(np.float64)
    baseline["baseline_target_prob_orig"] = probs[np.arange(len(baseline), dtype=np.int64), target_ids].astype(np.float64)
    baseline["baseline_correct"] = (pred_ids == target_ids).astype(int)
    return baseline, probs


def _build_intervention_rows(
    *,
    base_rows: pd.DataFrame,
    tokenizer,
    n_targets_per_example: int,
    seed: int,
    distance_buckets: tuple[tuple[int, int], ...],
    value_min: int,
    value_max: int,
    candidate_values: np.ndarray | None = None,
) -> pd.DataFrame:
    target_pool = (
        np.arange(int(value_min), int(value_max) + 1, dtype=np.int64)
        if candidate_values is None
        else np.asarray(candidate_values, dtype=np.int64).reshape(-1)
    )
    if len(target_pool) == 0:
        raise ValueError("candidate_values must contain at least one value.")
    if np.any(target_pool < int(value_min)) or np.any(target_pool > int(value_max)):
        raise ValueError(f"candidate_values must lie in [{value_min}, {value_max}].")
    target_pool = np.unique(target_pool)
    rows: list[dict[str, Any]] = []
    for row in base_rows.to_dict("records"):
        source_value = int(row["source_value"])
        valid_targets = valid_source_value_targets_for_row(
            task=str(row["task"]),
            operand_mode=str(row["operand_mode"]),
            x1=int(row["x1"]),
            x2=int(row["x2"]),
            source_value=source_value,
            value_min=int(value_min),
            value_max=int(value_max),
            candidate_values=target_pool,
        )
        selected_targets = sample_stratified_target_values(
            source_value=source_value,
            valid_target_values=valid_targets,
            num_targets=int(n_targets_per_example),
            seed=int(seed) + (1000003 * int(row["baseline_row_idx"])),
            distance_buckets=distance_buckets,
        )
        for target_slot, target_value in enumerate(selected_targets):
            shifted_x1, shifted_x2, shifted_target = shifted_operands_and_target_for_row(
                task=str(row["task"]),
                operand_mode=str(row["operand_mode"]),
                x1=int(row["x1"]),
                x2=int(row["x2"]),
                target_value=int(target_value),
            )
            token_ids = encode_integer(tokenizer, shifted_target)
            if len(token_ids) != 1:
                raise ValueError(f"Shifted target {shifted_target} is not single-token.")
            if str(row["task"]) in {"addition", "subtraction"}:
                shifted_prompt, shifted_token_idx = render_arithmetic_prompt_from_row(
                    row=row,
                    x1=shifted_x1,
                    x2=shifted_x2,
                    tokenizer=tokenizer,
                )
            elif str(row["task"]) == "identity":
                shifted_prompt, shifted_token_idx = render_identity_prompt(
                    value=int(target_value),
                    tokenizer=tokenizer,
                    example_values=json.loads(str(row.get("identity_example_values", "[]"))),
                    operator=str(row.get("identity_operator", " = ")),
                    separator=str(row.get("identity_separator", "; ")),
                )
            else:
                raise ValueError(f"Unsupported task: {row['task']!r}")
            abs_delta = abs(int(target_value) - source_value)
            matched_bucket = None
            for min_abs, max_abs in distance_buckets:
                if int(min_abs) <= abs_delta <= int(max_abs):
                    matched_bucket = distance_bucket_label(int(min_abs), int(max_abs))
                    break
            rows.append(
                {
                    **row,
                    "target_slot": int(target_slot),
                    "target_value": int(target_value),
                    "delta_value": int(target_value) - source_value,
                    "abs_delta_value": abs_delta,
                    "distance_bucket": matched_bucket,
                    "shifted_x1": shifted_x1,
                    "shifted_x2": shifted_x2,
                    "shifted_target": shifted_target,
                    "target_token_id": int(token_ids[0]),
                    "shifted_prompt": shifted_prompt,
                    "shifted_token_idx": int(shifted_token_idx),
                    "expected_value_baseline": float(row["baseline_expected_value"]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["target_slot", "source_value"]).reset_index(drop=True)
    return frame


def _attach_shifted_target_baseline_probs(
    *,
    intervention_df: pd.DataFrame,
    baseline_probs: np.ndarray,
) -> pd.DataFrame:
    updated = intervention_df.copy()
    row_idx = updated["baseline_row_idx"].to_numpy(dtype=np.int64)
    target_ids = updated["target_token_id"].to_numpy(dtype=np.int64)
    updated["baseline_target_prob"] = baseline_probs[row_idx, target_ids].astype(np.float64)
    return updated


def _filter_target_slot_shard(interventions: pd.DataFrame, target_sampler_cfg: dict[str, Any]) -> pd.DataFrame:
    shard_count = target_sampler_cfg.get("target_slot_shard_count")
    shard_index = target_sampler_cfg.get("target_slot_shard_index")
    if shard_count is None and shard_index is None:
        return interventions
    if shard_count is None or shard_index is None:
        raise ValueError("target_slot_shard_count and target_slot_shard_index must be set together.")
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count <= 0:
        raise ValueError("target_slot_shard_count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("target_slot_shard_index must be in [0, target_slot_shard_count).")
    selected = interventions.loc[interventions["target_slot"].astype(int) % shard_count == shard_index].copy()
    if selected.empty:
        raise ValueError(
            f"Target-slot shard {shard_index}/{shard_count} selected no interventions "
            f"from {interventions['target_slot'].nunique()} target slots."
        )
    selected["target_slot_shard_index"] = shard_index
    selected["target_slot_shard_count"] = shard_count
    return selected.reset_index(drop=True)


def _evaluate_with_delta_rows(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    delta_rows: np.ndarray,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["prompt"].tolist(),
        token_idxs=valid_df["token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        delta_rows=np.asarray(delta_rows, dtype=np.float32),
        capture_selected=False,
    )
    return evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )


def _optimize_scale_search(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    base_deltas: np.ndarray,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
    objective: str,
    steps: int,
    lr: float,
    l2_penalty: float,
    max_scale: float,
) -> pd.DataFrame:
    final_logits_chunks: list[np.ndarray] = []
    final_score_chunks: list[np.ndarray] = []
    best_scale_chunks: list[np.ndarray] = []
    best_delta_chunks: list[np.ndarray] = []
    baseline_score_chunks: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    direction_rows = np.asarray(base_deltas, dtype=np.float32)

    for start in range(0, len(valid_df), batch_size):
        batch_df = valid_df.iloc[start : start + batch_size].reset_index(drop=True)
        prompts = batch_df["prompt"].tolist()
        token_idxs = batch_df["token_idx"].astype(int).tolist()
        target_token_ids_np = np.array(batch_df["target_token_id"].to_numpy(dtype=np.int64), copy=True)
        baseline_token_ids_np = np.array(batch_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64), copy=True)
        target_token_ids = torch.as_tensor(target_token_ids_np, dtype=torch.long, device=model_device)
        baseline_token_ids = torch.as_tensor(baseline_token_ids_np, dtype=torch.long, device=model_device)
        base_direction = torch.as_tensor(direction_rows[start : start + len(batch_df)], dtype=torch.float32, device=model_device)

        scale_param = torch.nn.Parameter(torch.ones((len(batch_df), 1), dtype=torch.float32, device=model_device))
        optimizer = torch.optim.Adam([scale_param], lr=float(lr))
        best_reg_scores = torch.full((len(batch_df),), -float("inf"), dtype=torch.float32, device=model_device)
        best_scales = torch.ones((len(batch_df),), dtype=torch.float32, device=model_device)
        best_delta = base_direction.clone()

        baseline_logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=int(layer_idx),
            batch_size=len(batch_df),
            device=device,
            delta_rows=None,
            capture_selected=False,
        )
        baseline_objective = objective_scores_numpy(
            baseline_logits,
            target_token_ids=target_token_ids_np,
            baseline_token_ids=baseline_token_ids_np,
            objective=objective,
        )

        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            delta_rows = scale_param * base_direction
            logits = forward_next_token_logits_torch_with_delta(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                token_idxs=token_idxs,
                layer_idx=int(layer_idx),
                device=device,
                delta_rows=delta_rows,
            )
            objective_scores = objective_scores_torch(
                logits,
                target_token_ids=target_token_ids,
                baseline_token_ids=baseline_token_ids,
                objective=objective,
            )
            regularized_scores = objective_scores - float(l2_penalty) * delta_rows.pow(2).sum(dim=-1)
            (-regularized_scores.sum()).backward()
            optimizer.step()
            with torch.no_grad():
                scale_param.clamp_(min=0.0, max=float(max_scale))
                delta_now = scale_param * base_direction
                update = regularized_scores > best_reg_scores
                best_reg_scores[update] = regularized_scores[update]
                best_scales[update] = scale_param.squeeze(-1)[update]
                best_delta[update] = delta_now[update]

        best_logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=int(layer_idx),
            batch_size=len(batch_df),
            device=device,
            delta_rows=best_delta.detach().cpu().numpy(),
            capture_selected=False,
        )
        final_scores = objective_scores_numpy(
            best_logits,
            target_token_ids=target_token_ids_np,
            baseline_token_ids=baseline_token_ids_np,
            objective=objective,
        )
        final_logits_chunks.append(best_logits)
        final_score_chunks.append(final_scores.astype(np.float64))
        best_scale_chunks.append(best_scales.detach().float().cpu().numpy())
        best_delta_chunks.append(best_delta.detach().float().cpu().numpy())
        baseline_score_chunks.append(baseline_objective.astype(np.float64))

    final_logits = np.concatenate(final_logits_chunks, axis=0)
    final_scores = np.concatenate(final_score_chunks, axis=0)
    best_scales = np.concatenate(best_scale_chunks, axis=0)
    best_deltas = np.concatenate(best_delta_chunks, axis=0)
    baseline_scores = np.concatenate(baseline_score_chunks, axis=0)
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=final_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    result["search_objective"] = final_scores.astype(np.float64)
    result["delta_norm"] = np.linalg.norm(best_deltas, axis=1)
    result["search_score_gain"] = final_scores.astype(np.float64) - baseline_scores
    result["selected_scale"] = best_scales.astype(np.float64)
    return result


def _optimize_basis_search(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    basis_matrix: np.ndarray | None,
    norm_budgets: np.ndarray | None,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
    objective: str,
    steps: int,
    lr: float,
    l2_penalty: float,
) -> pd.DataFrame:
    final_logits_chunks: list[np.ndarray] = []
    final_score_chunks: list[np.ndarray] = []
    best_delta_chunks: list[np.ndarray] = []
    baseline_score_chunks: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    basis = torch.as_tensor(np.asarray(basis_matrix, dtype=np.float32), dtype=torch.float32, device=model_device)
    budget_rows = None if norm_budgets is None else np.asarray(norm_budgets, dtype=np.float32)

    for start in range(0, len(valid_df), batch_size):
        batch_df = valid_df.iloc[start : start + batch_size].reset_index(drop=True)
        prompts = batch_df["prompt"].tolist()
        token_idxs = batch_df["token_idx"].astype(int).tolist()
        target_token_ids_np = np.array(batch_df["target_token_id"].to_numpy(dtype=np.int64), copy=True)
        baseline_token_ids_np = np.array(batch_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64), copy=True)
        target_token_ids = torch.as_tensor(target_token_ids_np, dtype=torch.long, device=model_device)
        baseline_token_ids = torch.as_tensor(baseline_token_ids_np, dtype=torch.long, device=model_device)
        coeffs = torch.nn.Parameter(torch.zeros((len(batch_df), basis.shape[0]), dtype=torch.float32, device=model_device))
        optimizer = torch.optim.Adam([coeffs], lr=float(lr))
        best_reg_scores = torch.full((len(batch_df),), -float("inf"), dtype=torch.float32, device=model_device)
        best_delta = torch.zeros((len(batch_df), basis.shape[1]), dtype=torch.float32, device=model_device)
        budgets = (
            None
            if budget_rows is None
            else torch.as_tensor(budget_rows[start : start + len(batch_df)], dtype=torch.float32, device=model_device).view(-1, 1)
        )

        baseline_logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=int(layer_idx),
            batch_size=len(batch_df),
            device=device,
            delta_rows=None,
            capture_selected=False,
        )
        baseline_objective = objective_scores_numpy(
            baseline_logits,
            target_token_ids=target_token_ids_np,
            baseline_token_ids=baseline_token_ids_np,
            objective=objective,
        )

        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            delta_rows = coeffs @ basis
            logits = forward_next_token_logits_torch_with_delta(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                token_idxs=token_idxs,
                layer_idx=int(layer_idx),
                device=device,
                delta_rows=delta_rows,
            )
            objective_scores = objective_scores_torch(
                logits,
                target_token_ids=target_token_ids,
                baseline_token_ids=baseline_token_ids,
                objective=objective,
            )
            regularized_scores = objective_scores - float(l2_penalty) * delta_rows.pow(2).sum(dim=-1)
            (-regularized_scores.sum()).backward()
            optimizer.step()
            with torch.no_grad():
                delta_now = coeffs @ basis
                if budgets is not None:
                    norms = delta_now.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    factor = torch.clamp(budgets / norms, max=1.0)
                    coeffs.mul_(factor)
                    delta_now = coeffs @ basis
                update = regularized_scores > best_reg_scores
                best_reg_scores[update] = regularized_scores[update]
                best_delta[update] = delta_now[update]

        best_logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=int(layer_idx),
            batch_size=len(batch_df),
            device=device,
            delta_rows=best_delta.detach().cpu().numpy(),
            capture_selected=False,
        )
        final_scores = objective_scores_numpy(
            best_logits,
            target_token_ids=target_token_ids_np,
            baseline_token_ids=baseline_token_ids_np,
            objective=objective,
        )
        final_logits_chunks.append(best_logits)
        final_score_chunks.append(final_scores.astype(np.float64))
        best_delta_chunks.append(best_delta.detach().float().cpu().numpy())
        baseline_score_chunks.append(baseline_objective.astype(np.float64))

    final_logits = np.concatenate(final_logits_chunks, axis=0)
    final_scores = np.concatenate(final_score_chunks, axis=0)
    best_deltas = np.concatenate(best_delta_chunks, axis=0)
    baseline_scores = np.concatenate(baseline_score_chunks, axis=0)
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=final_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    result["search_objective"] = final_scores.astype(np.float64)
    result["delta_norm"] = np.linalg.norm(best_deltas, axis=1)
    result["search_score_gain"] = final_scores.astype(np.float64) - baseline_scores
    result["selected_scale"] = np.nan
    result["norm_budget"] = np.nan if budget_rows is None else budget_rows.astype(np.float64)
    return result


def _orthonormal_rows_torch(raw_basis: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal row basis spanning the current raw DAS parameters."""
    if raw_basis.ndim != 2:
        raise ValueError(f"Expected a 2D basis parameter, got shape {tuple(raw_basis.shape)!r}.")
    q, _ = torch.linalg.qr(raw_basis.T, mode="reduced")
    return q.T


def _project_rows_torch(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (rows @ basis.T) @ basis


def _optimize_das_projected_patch_basis(
    *,
    model,
    tokenizer,
    train_df: pd.DataFrame,
    exact_patch_rows: np.ndarray,
    rank: int,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    steps: int,
    lr: float,
    l2_penalty: float,
    seed: int,
    init_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Learn a DAS-style rotated low-rank subspace for patch interventions.

    The learned intervention applies only the projection of each exact
    source-to-target patch into a shared rank-r subspace. This is the direct
    linear analogue of DAS: the subspace is trained on the causal steering
    objective, not on probe decodability.
    """
    if int(rank) <= 0:
        raise ValueError(f"DAS rank must be positive, got {rank!r}.")
    if len(train_df) == 0:
        raise ValueError("DAS training requires at least one intervention row.")
    patch_rows = np.asarray(exact_patch_rows, dtype=np.float32)
    if patch_rows.ndim != 2 or patch_rows.shape[0] != len(train_df):
        raise ValueError(
            f"exact_patch_rows must have shape (len(train_df), hidden_dim), got {patch_rows.shape!r} for {len(train_df)} rows."
        )

    model_device = next(model.parameters()).device
    hidden_dim = int(patch_rows.shape[1])
    rank = min(int(rank), hidden_dim)
    if init_basis is None:
        generator = torch.Generator(device=model_device)
        generator.manual_seed(int(seed))
        raw_init = torch.randn((rank, hidden_dim), dtype=torch.float32, device=model_device, generator=generator)
    else:
        init = np.asarray(init_basis, dtype=np.float32)
        if init.ndim != 2 or init.shape[1] != hidden_dim:
            raise ValueError(f"init_basis must have shape (rank, {hidden_dim}), got {init.shape!r}.")
        if init.shape[0] < rank:
            raise ValueError(f"init_basis rank {init.shape[0]} is smaller than requested DAS rank {rank}.")
        raw_init = torch.as_tensor(init[:rank], dtype=torch.float32, device=model_device)

    raw_basis = torch.nn.Parameter(raw_init.clone())
    optimizer = torch.optim.Adam([raw_basis], lr=float(lr))
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(train_df), dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []

    for step_idx in range(int(steps)):
        if step_idx % max(1, int(np.ceil(len(order) / max(1, int(batch_size))))) == 0:
            rng.shuffle(order)
        start = (step_idx * int(batch_size)) % len(order)
        batch_indices = order[start : start + int(batch_size)]
        if len(batch_indices) < int(batch_size) and len(order) > len(batch_indices):
            batch_indices = np.concatenate([batch_indices, order[: int(batch_size) - len(batch_indices)]])
        batch_df = train_df.iloc[batch_indices].reset_index(drop=True)
        patches = torch.as_tensor(patch_rows[batch_indices], dtype=torch.float32, device=model_device)
        target_token_ids = torch.as_tensor(batch_df["target_token_id"].to_numpy(dtype=np.int64), dtype=torch.long, device=model_device)
        baseline_token_ids = torch.as_tensor(
            batch_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=model_device,
        )

        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        basis = _orthonormal_rows_torch(raw_basis)
        delta_rows = _project_rows_torch(patches, basis)
        logits = forward_next_token_logits_torch_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_df["prompt"].astype(str).tolist(),
            token_idxs=batch_df["token_idx"].astype(int).tolist(),
            layer_idx=int(layer_idx),
            device=device,
            delta_rows=delta_rows,
        )
        objective_values = objective_scores_torch(
            logits,
            target_token_ids=target_token_ids,
            baseline_token_ids=baseline_token_ids,
            objective=objective,
        )
        regularized = objective_values - float(l2_penalty) * delta_rows.pow(2).sum(dim=-1)
        (-regularized.mean()).backward()
        optimizer.step()

        diagnostics.append(
            {
                "step": int(step_idx),
                "rank": int(rank),
                "train_objective_mean": float(objective_values.detach().float().mean().cpu().item()),
                "train_regularized_objective_mean": float(regularized.detach().float().mean().cpu().item()),
                "mean_delta_norm": float(delta_rows.detach().float().norm(dim=-1).mean().cpu().item()),
            }
        )

    with torch.no_grad():
        basis = _orthonormal_rows_torch(raw_basis).detach().float().cpu().numpy().astype(np.float64)
    return basis, pd.DataFrame(diagnostics)


def _evaluate_das_projected_patch(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    exact_patch_rows: np.ndarray,
    basis_matrix: np.ndarray | None,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    patches = np.asarray(exact_patch_rows, dtype=np.float64)
    if basis_matrix is None:
        projected = patches
    else:
        basis = np.asarray(basis_matrix, dtype=np.float64)
        if basis.shape[0] >= patches.shape[1]:
            projected = patches
        else:
            projected = (patches @ basis.T) @ basis
    result = _evaluate_with_delta_rows(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        delta_rows=projected,
        layer_idx=int(layer_idx),
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
        batch_size=int(batch_size),
        device=device,
    )
    result["delta_norm"] = np.linalg.norm(projected, axis=1).astype(np.float64)
    return result


def _power_of_two_rank_grid(max_rank: int) -> list[int]:
    if int(max_rank) <= 0:
        raise ValueError(f"max_rank must be positive, got {max_rank!r}.")
    ranks: list[int] = []
    value = 1
    while value < int(max_rank):
        ranks.append(value)
        value *= 2
    if not ranks or ranks[-1] != int(max_rank):
        ranks.append(int(max_rank))
    return ranks


def _complete_orthonormal_basis_rows(basis_rows: np.ndarray, hidden_dim: int) -> np.ndarray:
    base = np.asarray(basis_rows, dtype=np.float64)
    if base.size == 0:
        return np.eye(int(hidden_dim), dtype=np.float64)
    base = base[: int(hidden_dim)]
    if base.shape[0] >= int(hidden_dim):
        return base[: int(hidden_dim)]
    complement = null_space(base)
    if complement.size == 0:
        return base
    return np.concatenate([base, complement.T.astype(np.float64)], axis=0)


def _value_feature_delta_rows(probe, frame: pd.DataFrame) -> np.ndarray:
    source_features = probe.encode_values(frame["source_value"].to_numpy(dtype=np.int64))
    target_features = probe.encode_values(frame["target_value"].to_numpy(dtype=np.int64))
    return np.asarray(target_features - source_features, dtype=np.float64)


def _circular_das_delta_rows(probe, frame: pd.DataFrame, decoder_matrix: np.ndarray) -> np.ndarray:
    feature_delta = _value_feature_delta_rows(probe, frame)
    decoder = np.asarray(decoder_matrix, dtype=np.float64)
    if feature_delta.ndim != 2 or decoder.ndim != 2 or feature_delta.shape[1] != decoder.shape[0]:
        raise ValueError(f"Feature/decoder shapes do not align: {feature_delta.shape!r} @ {decoder.shape!r}.")
    return np.asarray(feature_delta @ decoder, dtype=np.float64)


def _optimize_circular_das_decoder(
    *,
    model,
    tokenizer,
    train_df: pd.DataFrame,
    feature_delta_rows: np.ndarray,
    init_decoder: np.ndarray,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    steps: int,
    lr: float,
    l2_penalty: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Train a circular-probe decoder directly on the steering objective."""
    if len(train_df) == 0:
        raise ValueError("Circular DAS training requires at least one intervention row.")
    features = np.asarray(feature_delta_rows, dtype=np.float32)
    decoder_init = np.asarray(init_decoder, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(train_df):
        raise ValueError(f"feature_delta_rows must have shape (len(train_df), feature_dim), got {features.shape!r}.")
    if decoder_init.ndim != 2 or decoder_init.shape[0] != features.shape[1]:
        raise ValueError(f"init_decoder must have shape ({features.shape[1]}, hidden_dim), got {decoder_init.shape!r}.")

    model_device = next(model.parameters()).device
    decoder = torch.nn.Parameter(torch.as_tensor(decoder_init, dtype=torch.float32, device=model_device).clone())
    optimizer = torch.optim.Adam([decoder], lr=float(lr))
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(train_df), dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []

    for step_idx in range(int(steps)):
        if step_idx % max(1, int(np.ceil(len(order) / max(1, int(batch_size))))) == 0:
            rng.shuffle(order)
        start = (step_idx * int(batch_size)) % len(order)
        batch_indices = order[start : start + int(batch_size)]
        if len(batch_indices) < int(batch_size) and len(order) > len(batch_indices):
            batch_indices = np.concatenate([batch_indices, order[: int(batch_size) - len(batch_indices)]])
        batch_df = train_df.iloc[batch_indices].reset_index(drop=True)
        feature_delta = torch.as_tensor(features[batch_indices], dtype=torch.float32, device=model_device)
        target_token_ids = torch.as_tensor(
            np.array(batch_df["target_token_id"].to_numpy(dtype=np.int64), copy=True),
            dtype=torch.long,
            device=model_device,
        )
        baseline_token_ids = torch.as_tensor(
            np.array(batch_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64), copy=True),
            dtype=torch.long,
            device=model_device,
        )

        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        delta_rows = feature_delta @ decoder
        logits = forward_next_token_logits_torch_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_df["prompt"].astype(str).tolist(),
            token_idxs=batch_df["token_idx"].astype(int).tolist(),
            layer_idx=int(layer_idx),
            device=device,
            delta_rows=delta_rows,
        )
        objective_values = objective_scores_torch(
            logits,
            target_token_ids=target_token_ids,
            baseline_token_ids=baseline_token_ids,
            objective=objective,
        )
        regularized = objective_values - float(l2_penalty) * delta_rows.pow(2).sum(dim=-1)
        (-regularized.mean()).backward()
        optimizer.step()

        diagnostics.append(
            {
                "step": int(step_idx),
                "train_objective_mean": float(objective_values.detach().float().mean().cpu().item()),
                "train_regularized_objective_mean": float(regularized.detach().float().mean().cpu().item()),
                "mean_delta_norm": float(delta_rows.detach().float().norm(dim=-1).mean().cpu().item()),
                "decoder_norm": float(decoder.detach().float().norm().cpu().item()),
            }
        )

    return decoder.detach().float().cpu().numpy().astype(np.float64), pd.DataFrame(diagnostics)


def _evaluate_circular_das(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    probe,
    decoder_matrix: np.ndarray,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    delta_rows = _circular_das_delta_rows(probe, valid_df, decoder_matrix)
    result = _evaluate_with_delta_rows(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        delta_rows=delta_rows,
        layer_idx=int(layer_idx),
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
        batch_size=int(batch_size),
        device=device,
    )
    result["delta_norm"] = np.linalg.norm(delta_rows, axis=1).astype(np.float64)
    return result


def _method_slug(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _search_methods(search_cfg: dict[str, Any]) -> list[str]:
    methods = search_cfg.get("methods")
    if methods is None:
        return ["fixed", "scale_search", "manifold_search"]
    parsed = [str(value) for value in methods]
    return ["probe_path_search" if value == "probe_manifold_path_search" else value for value in parsed]


def _path_search_positions(search_cfg: dict[str, Any]) -> np.ndarray:
    if search_cfg.get("path_positions") is not None:
        positions = [float(value) for value in search_cfg["path_positions"]]
    else:
        num_positions = int(search_cfg.get("path_num_positions", 21))
        if num_positions < 2:
            raise ValueError("path_num_positions must be at least 2.")
        min_position = float(search_cfg.get("path_min_position", 0.0))
        max_position = float(search_cfg.get("path_max_position", 1.0))
        positions = np.linspace(min_position, max_position, num_positions).tolist()
    if not positions:
        raise ValueError("Probe-manifold path search requires at least one path position.")
    return np.asarray(positions, dtype=np.float64)


def _equal_budget_patch_multipliers(search_cfg: dict[str, Any]) -> list[float]:
    values = search_cfg.get("equal_budget_patch_multipliers", search_cfg.get("equal_budget_multipliers"))
    if values is None:
        value = search_cfg.get("equal_budget_patch_multiplier", 1.0)
        return [float(value)]
    parsed = [float(value) for value in values]
    if not parsed:
        return [1.0]
    return [max(parsed)]


def _reference_methods(cfg: dict[str, Any]) -> list[str]:
    methods = cfg.get("reference_methods", [])
    return [str(value) for value in methods]


def normalize_equal_budget_search_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Collapse legacy equal-budget multiplier sweeps to one exact-patch budget."""
    updated = dict(cfg)
    search_cfg = dict(updated.get("search", {}))
    if "equal_budget_search" in _search_methods(search_cfg):
        multiplier = _equal_budget_patch_multipliers(search_cfg)[0]
        search_cfg.pop("equal_budget_patch_multipliers", None)
        search_cfg.pop("equal_budget_multipliers", None)
        if np.isclose(float(multiplier), 1.0):
            search_cfg.pop("equal_budget_patch_multiplier", None)
        else:
            search_cfg["equal_budget_patch_multiplier"] = float(multiplier)
    updated["search"] = search_cfg
    return updated


def _value_centroids(values: np.ndarray, activations: np.ndarray, *, value_min: int, value_max: int) -> np.ndarray:
    value_arr = np.asarray(values, dtype=np.int64)
    x_arr = np.asarray(activations, dtype=np.float64)
    centroids = np.full((int(value_max) - int(value_min) + 1, x_arr.shape[1]), np.nan, dtype=np.float64)
    for value in range(int(value_min), int(value_max) + 1):
        mask = value_arr == int(value)
        if not np.any(mask):
            raise ValueError(f"Cannot build value-centroid reference: no activations for value {value}.")
        centroids[int(value) - int(value_min)] = x_arr[mask].mean(axis=0)
    return centroids


def _pca_value_centroid_reference(
    values: np.ndarray,
    activations: np.ndarray,
    *,
    value_min: int,
    value_max: int,
    n_components: int,
) -> dict[str, Any]:
    value_arr = np.asarray(values, dtype=np.int64)
    x_arr = np.asarray(activations, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected a 2D activation matrix, got shape {x_arr.shape!r}.")
    if int(n_components) < 1:
        raise ValueError(f"n_components must be positive, got {n_components!r}.")
    n_fit_components = min(int(n_components), x_arr.shape[0] - 1, x_arr.shape[1])
    if n_fit_components < 1:
        raise ValueError(f"Cannot fit PCA with activation shape {x_arr.shape!r}.")
    mean = x_arr.mean(axis=0)
    centered = x_arr - mean[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_fit_components].astype(np.float64)
    projected = centered @ components.T
    centroids = _value_centroids(
        values=value_arr,
        activations=projected,
        value_min=value_min,
        value_max=value_max,
    )
    total_variance = float(np.sum(np.square(singular_values)))
    explained_variance = float(np.sum(np.square(singular_values[:n_fit_components])))
    explained_ratio = explained_variance / total_variance if total_variance > 0.0 else float("nan")
    return {
        "centroids": centroids,
        "components": components,
        "n_components": int(n_fit_components),
        "requested_n_components": int(n_components),
        "explained_variance_ratio": explained_ratio,
    }


def _collect_shifted_representations_for_slot(
    *,
    slot_df: pd.DataFrame,
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    return np.asarray(
        _collect_partition_activations(
            rows_by_partition={
                "slot": slot_df[["prompt", "shifted_prompt", "shifted_token_idx", "token_idx", "source_value"]].assign(
                    prompt=slot_df["shifted_prompt"].astype(str),
                    token_idx=slot_df["shifted_token_idx"].astype(int),
                )
            },
            model=model,
            tokenizer=tokenizer,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
            use_activation_cache=False,
        )["slot"][1],
        dtype=np.float64,
    )


def _base_deltas_for_slot(
    *,
    delta_kind: str,
    probe,
    slot_df: pd.DataFrame,
    source_representations: np.ndarray,
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int,
    device: str,
    target_representations: np.ndarray | None = None,
) -> np.ndarray:
    if delta_kind == "value_code":
        return compute_value_code_probe_delta(
            probe=probe,
            source_values=slot_df["source_value"].to_numpy(dtype=np.int64),
            target_values=slot_df["target_value"].to_numpy(dtype=np.int64),
        )
    if delta_kind == "projected_patch":
        if target_representations is None:
            target_representations = _collect_shifted_representations_for_slot(
                slot_df=slot_df,
                model=model,
                tokenizer=tokenizer,
                layer_idx=int(layer_idx),
                batch_size=int(batch_size),
                device=device,
            )
        source_rows = source_representations[slot_df["baseline_row_idx"].to_numpy(dtype=np.int64)]
        return compute_projected_patch_probe_delta(
            probe=probe,
            source_representations=source_rows,
            target_representations=target_representations,
        )
    raise ValueError(f"Unsupported delta_kind: {delta_kind!r}")


def _build_das_training_panel(
    *,
    train_rows: pd.DataFrame,
    train_values: np.ndarray,
    train_x: np.ndarray,
    model,
    tokenizer,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
    device: str,
    seed: int,
    n_targets_per_example: int,
    distance_buckets: tuple[tuple[int, int], ...],
    value_min: int,
    value_max: int,
    max_interventions: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_interventions = _build_das_training_rows(
        train_rows=train_rows,
        train_values=train_values,
        model=model,
        tokenizer=tokenizer,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        seed=seed,
        n_targets_per_example=n_targets_per_example,
        distance_buckets=distance_buckets,
        value_min=value_min,
        value_max=value_max,
        max_interventions=max_interventions,
    )

    patch_parts: list[np.ndarray] = []
    row_parts: list[pd.DataFrame] = []
    for target_slot in sorted(train_interventions["target_slot"].unique().tolist()):
        slot_df = train_interventions.loc[train_interventions["target_slot"] == int(target_slot)].copy().reset_index(drop=True)
        source_rows = np.asarray(train_x[slot_df["baseline_row_idx"].to_numpy(dtype=np.int64)], dtype=np.float64)
        target_rows = _collect_shifted_representations_for_slot(
            slot_df=slot_df,
            model=model,
            tokenizer=tokenizer,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
        )
        patch_parts.append(np.asarray(target_rows - source_rows, dtype=np.float64))
        row_parts.append(slot_df)
    if not patch_parts:
        raise ValueError("DAS training panel is empty.")
    return pd.concat(row_parts, ignore_index=True), np.concatenate(patch_parts, axis=0)


def _build_das_training_rows(
    *,
    train_rows: pd.DataFrame,
    train_values: np.ndarray,
    model,
    tokenizer,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
    device: str,
    seed: int,
    n_targets_per_example: int,
    distance_buckets: tuple[tuple[int, int], ...],
    value_min: int,
    value_max: int,
    max_interventions: int | None,
) -> pd.DataFrame:
    train_prompts = train_rows["prompt"].astype(str).tolist()
    train_token_idxs = train_rows["token_idx"].astype(int).tolist()
    train_logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=train_prompts,
        token_idxs=train_token_idxs,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        delta_rows=None,
        capture_selected=False,
    )
    baseline_train, train_probs = _build_baseline_frame(
        tokenizer=tokenizer,
        base_rows=train_rows.copy(),
        logits=train_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    if not np.array_equal(baseline_train["source_value"].to_numpy(dtype=np.int64), np.asarray(train_values, dtype=np.int64)):
        raise ValueError("Train activation order does not match train baseline row order.")
    train_interventions = _build_intervention_rows(
        base_rows=baseline_train,
        tokenizer=tokenizer,
        n_targets_per_example=int(n_targets_per_example),
        seed=int(seed),
        distance_buckets=distance_buckets,
        value_min=int(value_min),
        value_max=int(value_max),
    )
    train_interventions = _attach_shifted_target_baseline_probs(
        intervention_df=train_interventions,
        baseline_probs=train_probs,
    )
    if max_interventions is not None and int(max_interventions) > 0 and len(train_interventions) > int(max_interventions):
        train_interventions = (
            train_interventions.sample(n=int(max_interventions), random_state=int(seed))
            .sort_values(["target_slot", "source_value"])
            .reset_index(drop=True)
        )
    if train_interventions.empty:
        raise ValueError("DAS training rows are empty.")
    return train_interventions.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    cfg = normalize_equal_budget_search_config(raw_cfg)
    set_global_seed(int(cfg["seed"]))

    models_payload = load_yaml(cfg["models_config"])
    model_alias = str(cfg["model"])
    model_entry = _resolve_model_entry(models_payload, model_alias)
    model_name = str(model_entry["hf_id"])
    device = str(cfg.get("device") or model_entry.get("device", "cpu"))
    dtype = str(cfg.get("dtype") or model_entry.get("dtype", "float32"))
    layer_idx = int(cfg["intervention_layer_idx"])

    hash_payload = {
        "task_name": "source_value_probe_manifold_steering",
        "config": raw_cfg,
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
    }
    run_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    run_dir = Path(args.output_root) / f"source_value_probe_manifold_steering__{run_hash}__r0"
    if (run_dir / "DONE").exists() and not args.force:
        print(run_dir)
        return

    rows_by_partition = _load_partition_rows(
        dataset_dir=Path(cfg["split_dataset_dir"]),
        split_name=str(cfg["split_name"]),
        family=str(cfg["family"]),
    )
    distance_buckets = _bucket_specs_from_config(cfg["target_sampler"])

    trust_remote_code = bool(model_entry.get("trust_remote_code", False))
    loaded = load_causal_lm(
        model_name=model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=model_entry.get("attn_implementation"),
        revision=model_entry.get("revision"),
    )
    freeze_model_parameters(loaded.model)

    token_report = build_single_token_report(loaded.tokenizer)
    if not bool(token_report["is_single_token"].all()):
        raise RuntimeError(f"Model {model_name} does not satisfy the single-token 0..999 requirement.")

    model_revision = loaded.model_revision or resolve_model_revision(
        model_name,
        trust_remote_code=trust_remote_code,
        revision=model_entry.get("revision"),
    )

    activations_by_partition = _collect_partition_activations(
        rows_by_partition=rows_by_partition,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        layer_idx=layer_idx,
        batch_size=int(cfg["probe_batch_size"]),
        device=device,
        model_alias=model_alias,
        model_id=model_name,
        model_revision=model_revision,
        attn_implementation=loaded.attn_implementation,
        family=str(cfg["family"]),
        activation_cache_root=cfg.get("activation_cache_root"),
        use_activation_cache=bool(cfg.get("use_activation_cache", True)),
        require_activation_cache_hit=bool(cfg.get("require_activation_cache_hit", True)),
    )
    train_values, train_x = activations_by_partition["train"]
    dev_values, dev_x = activations_by_partition["dev"]

    prompts = rows_by_partition["dev"]["prompt"].astype(str).tolist()
    token_idxs = rows_by_partition["dev"]["token_idx"].astype(int).tolist()
    baseline_logits, _ = forward_next_token_logits_with_delta(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=layer_idx,
        batch_size=int(cfg["evaluation_batch_size"]),
        device=device,
        delta_rows=None,
        capture_selected=False,
    )
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(loaded.tokenizer)
    baseline_dev, baseline_probs = _build_baseline_frame(
        tokenizer=loaded.tokenizer,
        base_rows=rows_by_partition["dev"].copy(),
        logits=baseline_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    if not np.array_equal(baseline_dev["source_value"].to_numpy(dtype=np.int64), dev_values):
        raise ValueError("Dev activation order does not match dev baseline row order.")

    interventions = _build_intervention_rows(
        base_rows=baseline_dev,
        tokenizer=loaded.tokenizer,
        n_targets_per_example=int(cfg["target_sampler"]["n_targets_per_example"]),
        seed=int(cfg["seed"]),
        distance_buckets=distance_buckets,
        value_min=int(cfg.get("value_min", 0)),
        value_max=int(cfg.get("value_max", 999)),
    )
    interventions = _attach_shifted_target_baseline_probs(
        intervention_df=interventions,
        baseline_probs=baseline_probs,
    )
    interventions = _filter_target_slot_shard(interventions, dict(cfg["target_sampler"]))

    ensure_dir(run_dir)
    atomic_write_text(run_dir / "command.txt", " ".join(sys.argv) + "\n")
    atomic_write_csv(run_dir / "token_report.csv", token_report)
    atomic_write_csv(run_dir / "baseline_dev.csv", baseline_dev)
    atomic_write_csv(run_dir / "interventions.csv", interventions)

    probe_rows: list[dict[str, Any]] = []
    agop_diag_frames: list[pd.DataFrame] = []
    das_diag_frames: list[pd.DataFrame] = []
    das_basis_payload: dict[str, np.ndarray] = {}
    circular_das_diag_frames: list[pd.DataFrame] = []
    circular_das_decoder_payload: dict[str, np.ndarray] = {}
    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    overall_rows: list[dict[str, Any]] = []
    search_cfg = dict(cfg["search"])
    search_methods = _search_methods(search_cfg)
    reference_methods = _reference_methods(cfg)
    equal_budget_multipliers = _equal_budget_patch_multipliers(search_cfg)
    slot_representation_cache: dict[int, dict[str, np.ndarray]] = {}
    value_min = int(cfg.get("value_min", 0))
    value_max = int(cfg.get("value_max", 999))
    centroid_reference = None
    pca64_centroid_reference = None
    reference_method_metadata: dict[str, Any] = {}
    if "causalab_centroid_endpoint" in reference_methods or "causalab_pca64_centroid_endpoint" in reference_methods:
        all_partition_values = []
        all_partition_x = []
        for partition_values, partition_x in activations_by_partition.values():
            all_partition_values.append(np.asarray(partition_values, dtype=np.int64))
            all_partition_x.append(np.asarray(partition_x, dtype=np.float64))
        reference_values = np.concatenate(all_partition_values, axis=0)
        reference_x = np.concatenate(all_partition_x, axis=0)
        reference_method_metadata["centroid_reference_examples"] = int(len(reference_values))
        if "causalab_centroid_endpoint" in reference_methods:
            centroid_reference = _value_centroids(
                values=reference_values,
                activations=reference_x,
                value_min=value_min,
                value_max=value_max,
            )
        if "causalab_pca64_centroid_endpoint" in reference_methods:
            pca64_centroid_reference = _pca_value_centroid_reference(
                values=reference_values,
                activations=reference_x,
                value_min=value_min,
                value_max=value_max,
                n_components=int(cfg.get("centroid_pca_components", 64)),
            )
            reference_method_metadata["pca64_centroid_reference"] = {
                "requested_n_components": int(pca64_centroid_reference["requested_n_components"]),
                "n_components": int(pca64_centroid_reference["n_components"]),
                "explained_variance_ratio": float(pca64_centroid_reference["explained_variance_ratio"]),
            }

    def slot_representations(target_slot: int, slot_df: pd.DataFrame) -> dict[str, np.ndarray]:
        cached = slot_representation_cache.get(int(target_slot))
        if cached is not None:
            return cached
        source_rows = np.asarray(dev_x[slot_df["baseline_row_idx"].to_numpy(dtype=np.int64)], dtype=np.float64)
        target_rows = _collect_shifted_representations_for_slot(
            slot_df=slot_df,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            layer_idx=layer_idx,
            batch_size=int(cfg["evaluation_batch_size"]),
            device=device,
        )
        exact_patch_rows = np.asarray(target_rows - source_rows, dtype=np.float64)
        cached = {
            "source_rows": source_rows,
            "target_rows": target_rows,
            "exact_patch_rows": exact_patch_rows,
            "exact_patch_norms": np.linalg.norm(exact_patch_rows, axis=1).astype(np.float64),
        }
        slot_representation_cache[int(target_slot)] = cached
        return cached

    reference_method_parts: list[pd.DataFrame] = []
    for target_slot in sorted(interventions["target_slot"].unique().tolist()):
        slot_df = interventions.loc[interventions["target_slot"] == int(target_slot)].copy().reset_index(drop=True)
        reps = slot_representations(int(target_slot), slot_df)
        exact_patch_rows = reps["exact_patch_rows"].astype(np.float64)
        exact_patch_norms = reps["exact_patch_norms"].astype(np.float64)

        if "causalab_centroid_endpoint" in reference_methods:
            if centroid_reference is None:
                raise RuntimeError("Internal error: missing centroid_reference.")
            source_idx = slot_df["source_value"].to_numpy(dtype=np.int64) - value_min
            target_idx = slot_df["target_value"].to_numpy(dtype=np.int64) - value_min
            centroid_deltas = centroid_reference[target_idx] - centroid_reference[source_idx]
            centroid_norms = np.linalg.norm(centroid_deltas, axis=1).astype(np.float64)
            centroid = _evaluate_with_delta_rows(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                valid_df=slot_df,
                delta_rows=centroid_deltas,
                layer_idx=layer_idx,
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
                batch_size=int(cfg["evaluation_batch_size"]),
                device=device,
            )
            centroid["delta_norm"] = centroid_norms
            centroid["base_delta_norm"] = centroid_norms
            centroid["exact_patch_norm"] = exact_patch_norms
            centroid["selected_scale"] = 1.0
            centroid["search_score_gain"] = np.nan
            centroid["search_objective"] = np.nan
            centroid["norm_budget"] = np.nan
            centroid["budget_multiplier"] = np.nan
            centroid["search_tier"] = "causalab_centroid_endpoint"
            centroid["method"] = "causalab_centroid_endpoint"
            centroid["architecture"] = "causalab_1d_manifold"
            centroid["delta_kind"] = "centroid_endpoint"
            reference_method_parts.append(centroid)

        if "causalab_pca64_centroid_endpoint" in reference_methods:
            if pca64_centroid_reference is None:
                raise RuntimeError("Internal error: missing pca64_centroid_reference.")
            source_idx = slot_df["source_value"].to_numpy(dtype=np.int64) - value_min
            target_idx = slot_df["target_value"].to_numpy(dtype=np.int64) - value_min
            pca64_delta_coords = pca64_centroid_reference["centroids"][target_idx] - pca64_centroid_reference["centroids"][source_idx]
            pca64_deltas = pca64_delta_coords @ pca64_centroid_reference["components"]
            pca64_norms = np.linalg.norm(pca64_deltas, axis=1).astype(np.float64)
            pca64 = _evaluate_with_delta_rows(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                valid_df=slot_df,
                delta_rows=pca64_deltas,
                layer_idx=layer_idx,
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
                batch_size=int(cfg["evaluation_batch_size"]),
                device=device,
            )
            pca64["delta_norm"] = pca64_norms
            pca64["base_delta_norm"] = pca64_norms
            pca64["exact_patch_norm"] = exact_patch_norms
            pca64["selected_scale"] = 1.0
            pca64["search_score_gain"] = np.nan
            pca64["search_objective"] = np.nan
            pca64["norm_budget"] = np.nan
            pca64["budget_multiplier"] = np.nan
            pca64["search_tier"] = "causalab_pca64_centroid_endpoint"
            pca64["method"] = "causalab_pca64_centroid_endpoint"
            pca64["architecture"] = "causalab_pca64_1d_manifold"
            pca64["delta_kind"] = "pca64_centroid_endpoint"
            reference_method_parts.append(pca64)

        if "counterfactual_patch" in reference_methods:
            full = _evaluate_with_delta_rows(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                valid_df=slot_df,
                delta_rows=exact_patch_rows,
                layer_idx=layer_idx,
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
                batch_size=int(cfg["evaluation_batch_size"]),
                device=device,
            )
            full["delta_norm"] = exact_patch_norms
            full["base_delta_norm"] = exact_patch_norms
            full["exact_patch_norm"] = exact_patch_norms
            full["selected_scale"] = 1.0
            full["search_score_gain"] = np.nan
            full["search_objective"] = np.nan
            full["norm_budget"] = np.nan
            full["budget_multiplier"] = np.nan
            full["search_tier"] = "counterfactual_patch"
            full["method"] = "counterfactual_patch"
            full["architecture"] = "full_counterfactual"
            full["delta_kind"] = "exact_patch"
            reference_method_parts.append(full)

    if reference_method_parts:
        reference_detail = pd.concat(reference_method_parts, ignore_index=True)
        detail_frames.append(reference_detail)
        slot_summary = summarize_steering_metrics(
            reference_detail,
            group_cols=["architecture", "delta_kind", "method", "target_slot", "distance_bucket"],
        )
        overall_summary = summarize_steering_metrics(
            reference_detail,
            group_cols=["architecture", "delta_kind", "method"],
        )
        summary_frames.extend([slot_summary, overall_summary])
        overall_rows.extend(overall_summary.to_dict("records"))

    for das_name, das_cfg_raw in dict(cfg.get("das_specs", {})).items():
        das_cfg = dict(das_cfg_raw)
        ranks = das_cfg.get("ranks", das_cfg.get("rank", 1))
        if isinstance(ranks, str) and ranks == "powers_of_two_to_model_dim":
            rank_values = _power_of_two_rank_grid(int(train_x.shape[1]))
        elif isinstance(ranks, (int, float, str)):
            rank_values = [int(ranks)]
        else:
            rank_values = [int(value) for value in ranks]
        if not rank_values:
            raise ValueError(f"DAS spec {das_name!r} must include at least one rank.")
        hidden_dim = int(train_x.shape[1])
        rank_values = sorted(set(max(1, min(int(rank), hidden_dim)) for rank in rank_values))
        optimize_rank_max = das_cfg.get("optimize_rank_max")
        optimize_rank_max = hidden_dim if optimize_rank_max is None else int(optimize_rank_max)
        save_unoptimized_bases = bool(das_cfg.get("save_unoptimized_bases", False))
        das_train_df, das_train_patches = _build_das_training_panel(
            train_rows=rows_by_partition["train"],
            train_values=train_values,
            train_x=train_x,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            layer_idx=layer_idx,
            batch_size=int(das_cfg.get("batch_size", cfg["evaluation_batch_size"])),
            device=device,
            seed=int(das_cfg.get("seed", int(cfg["seed"]) + 3109)),
            n_targets_per_example=int(das_cfg.get("n_targets_per_example", cfg["target_sampler"]["n_targets_per_example"])),
            distance_buckets=distance_buckets,
            value_min=value_min,
            value_max=value_max,
            max_interventions=None if das_cfg.get("max_train_interventions") is None else int(das_cfg["max_train_interventions"]),
        )
        init_mode = str(das_cfg.get("init", "patch_pca"))
        completed_patch_basis = None
        if init_mode == "patch_pca":
            max_basis_rank = min(max(rank_values), hidden_dim)
            patch_basis = top_patch_subspace_basis(das_train_patches, rank=min(max_basis_rank, len(das_train_patches), hidden_dim))
            completed_patch_basis = _complete_orthonormal_basis_rows(patch_basis, hidden_dim=hidden_dim)
        elif init_mode != "random":
            raise ValueError(f"Unsupported DAS init mode {init_mode!r}; expected 'patch_pca' or 'random'.")
        das_method_parts: list[pd.DataFrame] = []
        for requested_rank in rank_values:
            rank = int(requested_rank)
            init_basis = None
            optimized = rank <= int(optimize_rank_max)
            if init_mode == "patch_pca" and completed_patch_basis is not None:
                init_basis = completed_patch_basis[:rank]
            if optimized:
                basis, diagnostics = _optimize_das_projected_patch_basis(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    train_df=das_train_df,
                    exact_patch_rows=das_train_patches,
                    rank=rank,
                    layer_idx=layer_idx,
                    batch_size=int(das_cfg.get("batch_size", cfg["evaluation_batch_size"])),
                    device=device,
                    objective=str(das_cfg.get("objective", search_cfg.get("objective", "target_logprob"))),
                    steps=int(das_cfg.get("steps", 200)),
                    lr=float(das_cfg.get("lr", 0.01)),
                    l2_penalty=float(das_cfg.get("l2_penalty", 0.0)),
                    seed=int(das_cfg.get("seed", int(cfg["seed"]) + 3109)) + rank,
                    init_basis=init_basis,
                )
            else:
                if completed_patch_basis is None:
                    raise ValueError("Unoptimized DAS ranks require patch_pca initialization so a completed basis is available.")
                basis = None if rank >= hidden_dim else np.asarray(completed_patch_basis[:rank], dtype=np.float64)
                diagnostics = pd.DataFrame(
                    [
                        {
                            "step": -1,
                            "rank": int(rank),
                            "train_objective_mean": np.nan,
                            "train_regularized_objective_mean": np.nan,
                            "mean_delta_norm": np.nan,
                        }
                    ]
                )
            architecture_name = f"das_{das_name}_rank{rank}"
            basis_key = f"{architecture_name}__basis"
            if basis is not None and (optimized or save_unoptimized_bases):
                das_basis_payload[basis_key] = np.asarray(basis, dtype=np.float64)
            diagnostics["architecture"] = architecture_name
            diagnostics["method"] = "das_projected_patch"
            diagnostics["delta_kind"] = "projected_patch"
            diagnostics["requested_rank"] = int(requested_rank)
            diagnostics["rank"] = int(rank)
            diagnostics["hidden_dim"] = int(hidden_dim)
            diagnostics["optimized"] = bool(optimized)
            diagnostics["basis_source"] = "optimized_das" if optimized else ("full_identity" if rank >= hidden_dim else "completed_patch_pca")
            diagnostics["train_rows"] = int(len(das_train_df))
            das_diag_frames.append(diagnostics)

            for target_slot in sorted(interventions["target_slot"].unique().tolist()):
                slot_df = interventions.loc[interventions["target_slot"] == int(target_slot)].copy().reset_index(drop=True)
                reps = slot_representations(int(target_slot), slot_df)
                exact_patch_rows = reps["exact_patch_rows"].astype(np.float64)
                exact_patch_norms = reps["exact_patch_norms"].astype(np.float64)
                das_eval = _evaluate_das_projected_patch(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    valid_df=slot_df,
                    exact_patch_rows=exact_patch_rows,
                    basis_matrix=basis,
                    layer_idx=layer_idx,
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                    batch_size=int(cfg["evaluation_batch_size"]),
                    device=device,
                )
                das_eval["base_delta_norm"] = das_eval["delta_norm"].to_numpy(dtype=np.float64)
                das_eval["exact_patch_norm"] = exact_patch_norms
                das_eval["selected_scale"] = 1.0
                das_eval["search_score_gain"] = np.nan
                das_eval["search_objective"] = np.nan
                das_eval["norm_budget"] = np.nan
                das_eval["budget_multiplier"] = np.nan
                das_eval["search_tier"] = "das_projected_patch"
                das_eval["method"] = "das_projected_patch"
                das_eval["architecture"] = architecture_name
                das_eval["delta_kind"] = "projected_patch"
                das_eval["basis_rank"] = int(rank)
                das_eval["hidden_dim"] = int(hidden_dim)
                das_eval["das_optimized"] = bool(optimized)
                das_eval["basis_source"] = "optimized_das" if optimized else ("full_identity" if rank >= hidden_dim else "completed_patch_pca")
                das_method_parts.append(das_eval)

        if das_method_parts:
            das_detail = pd.concat(das_method_parts, ignore_index=True)
            detail_frames.append(das_detail)
            slot_summary = summarize_steering_metrics(
                das_detail,
                group_cols=["architecture", "delta_kind", "method", "target_slot", "distance_bucket"],
            )
            overall_summary = summarize_steering_metrics(
                das_detail,
                group_cols=["architecture", "delta_kind", "method"],
            )
            summary_frames.extend([slot_summary, overall_summary])
            overall_rows.extend(overall_summary.to_dict("records"))

    for das_name, circular_das_cfg_raw in dict(cfg.get("circular_das_specs", {})).items():
        circular_das_cfg = dict(circular_das_cfg_raw)
        architecture = str(circular_das_cfg.get("architecture", "circular_reg"))
        probe_cfg = dict(cfg.get("probe_specs", {}).get(architecture, {}))
        probe_cfg.update(dict(circular_das_cfg.get("probe_config", {})))
        if circular_das_cfg.get("basis_dim") is not None:
            probe_cfg["basis_dim"] = int(circular_das_cfg["basis_dim"])
        probe = fit_magnitude_probe(
            architecture=architecture,
            train_x=train_x,
            train_values=train_values,
            config=probe_cfg,
            value_min=value_min,
            value_max=value_max,
        )
        feature_dim = int(probe.support_features.shape[1])
        hidden_dim = int(train_x.shape[1])
        init_mode = str(circular_das_cfg.get("init", "probe_decoder"))
        if init_mode == "probe_decoder":
            init_decoder = np.asarray(probe.decoder_matrix, dtype=np.float64)
        elif init_mode == "zero":
            init_decoder = np.zeros((feature_dim, hidden_dim), dtype=np.float64)
        elif init_mode == "random":
            rng = np.random.default_rng(int(circular_das_cfg.get("seed", int(cfg["seed"]) + 7919)))
            scale = float(circular_das_cfg.get("random_init_scale", 0.001))
            init_decoder = rng.standard_normal(size=(feature_dim, hidden_dim)).astype(np.float64) * scale
        else:
            raise ValueError(f"Unsupported circular DAS init mode {init_mode!r}; expected probe_decoder, zero, or random.")
        if init_decoder.shape != (feature_dim, hidden_dim):
            raise ValueError(
                f"Circular DAS init decoder shape {init_decoder.shape!r} does not match feature_dim x hidden_dim {(feature_dim, hidden_dim)!r}."
            )

        train_rows_for_circular_das = _build_das_training_rows(
            train_rows=rows_by_partition["train"],
            train_values=train_values,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            layer_idx=layer_idx,
            batch_size=int(circular_das_cfg.get("batch_size", cfg["evaluation_batch_size"])),
            device=device,
            seed=int(circular_das_cfg.get("seed", int(cfg["seed"]) + 7919)),
            n_targets_per_example=int(circular_das_cfg.get("n_targets_per_example", cfg["target_sampler"]["n_targets_per_example"])),
            distance_buckets=distance_buckets,
            value_min=value_min,
            value_max=value_max,
            max_interventions=(
                None
                if circular_das_cfg.get("max_train_interventions") is None
                else int(circular_das_cfg["max_train_interventions"])
            ),
        )
        feature_deltas = _value_feature_delta_rows(probe, train_rows_for_circular_das)
        decoder, diagnostics = _optimize_circular_das_decoder(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            train_df=train_rows_for_circular_das,
            feature_delta_rows=feature_deltas,
            init_decoder=init_decoder,
            layer_idx=layer_idx,
            batch_size=int(circular_das_cfg.get("batch_size", cfg["evaluation_batch_size"])),
            device=device,
            objective=str(circular_das_cfg.get("objective", search_cfg.get("objective", "target_logprob"))),
            steps=int(circular_das_cfg.get("steps", 200)),
            lr=float(circular_das_cfg.get("lr", 0.01)),
            l2_penalty=float(circular_das_cfg.get("l2_penalty", 0.0)),
            seed=int(circular_das_cfg.get("seed", int(cfg["seed"]) + 7919)),
        )
        basis_dim_label = int(probe.metadata.get("basis_dim", probe_cfg.get("basis_dim", feature_dim)))
        architecture_name = f"{architecture}_das_{das_name}_basis{basis_dim_label}_feat{feature_dim}"
        circular_das_decoder_payload[f"{architecture_name}__decoder"] = np.asarray(decoder, dtype=np.float64)
        diagnostics["architecture"] = architecture_name
        diagnostics["probe_architecture"] = architecture
        diagnostics["method"] = "circular_das"
        diagnostics["delta_kind"] = "value_code"
        diagnostics["feature_dim"] = int(feature_dim)
        diagnostics["basis_dim"] = int(basis_dim_label)
        diagnostics["hidden_dim"] = int(hidden_dim)
        diagnostics["init_mode"] = init_mode
        diagnostics["train_rows"] = int(len(train_rows_for_circular_das))
        circular_das_diag_frames.append(diagnostics)

        circular_das_method_parts: list[pd.DataFrame] = []
        for target_slot in sorted(interventions["target_slot"].unique().tolist()):
            slot_df = interventions.loc[interventions["target_slot"] == int(target_slot)].copy().reset_index(drop=True)
            reps = slot_representations(int(target_slot), slot_df)
            exact_patch_norms = reps["exact_patch_norms"].astype(np.float64)
            circular_eval = _evaluate_circular_das(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                valid_df=slot_df,
                probe=probe,
                decoder_matrix=decoder,
                layer_idx=layer_idx,
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
                batch_size=int(cfg["evaluation_batch_size"]),
                device=device,
            )
            circular_eval["base_delta_norm"] = circular_eval["delta_norm"].to_numpy(dtype=np.float64)
            circular_eval["exact_patch_norm"] = exact_patch_norms
            circular_eval["selected_scale"] = 1.0
            circular_eval["search_score_gain"] = np.nan
            circular_eval["search_objective"] = np.nan
            circular_eval["norm_budget"] = np.nan
            circular_eval["budget_multiplier"] = np.nan
            circular_eval["search_tier"] = "circular_das"
            circular_eval["method"] = "circular_das"
            circular_eval["architecture"] = architecture_name
            circular_eval["delta_kind"] = "value_code"
            circular_eval["feature_dim"] = int(feature_dim)
            circular_eval["basis_dim"] = int(basis_dim_label)
            circular_eval["probe_architecture"] = architecture
            circular_das_method_parts.append(circular_eval)

        circular_das_detail = pd.concat(circular_das_method_parts, ignore_index=True)
        detail_frames.append(circular_das_detail)
        slot_summary = summarize_steering_metrics(
            circular_das_detail,
            group_cols=["architecture", "delta_kind", "method", "target_slot", "distance_bucket"],
        )
        overall_summary = summarize_steering_metrics(
            circular_das_detail,
            group_cols=["architecture", "delta_kind", "method"],
        )
        summary_frames.extend([slot_summary, overall_summary])
        overall_rows.extend(overall_summary.to_dict("records"))

    if bool(cfg.get("evaluate_probe_specs", True)):
        probe_specs_to_evaluate = dict(cfg.get("probe_specs", {}))
        if cfg.get("probe_architectures") is not None:
            selected_architectures = {str(value) for value in cfg["probe_architectures"]}
            probe_specs_to_evaluate = {key: value for key, value in probe_specs_to_evaluate.items() if str(key) in selected_architectures}
    else:
        probe_specs_to_evaluate = {}

    for architecture, probe_cfg in probe_specs_to_evaluate.items():
        probe = fit_magnitude_probe(
            architecture=str(architecture),
            train_x=train_x,
            train_values=train_values,
            config=dict(probe_cfg),
            value_min=int(cfg.get("value_min", 0)),
            value_max=int(cfg.get("value_max", 999)),
        )
        probe_rows.append(
            {
                "architecture": str(architecture),
                "probe_config": json.dumps(probe_cfg, sort_keys=True),
                "feature_dim": int(probe.metadata.get("feature_dim", probe.support_features.shape[1])),
                "train_size": int(len(train_values)),
            }
        )

        for delta_kind in [str(value) for value in cfg["delta_kinds"]]:
            method_parts: list[pd.DataFrame] = []
            family_basis = intervention_family_basis(probe)

            for target_slot in sorted(interventions["target_slot"].unique().tolist()):
                slot_df = interventions.loc[interventions["target_slot"] == int(target_slot)].copy().reset_index(drop=True)
                reps = slot_representations(int(target_slot), slot_df)
                base_deltas = _base_deltas_for_slot(
                    delta_kind=delta_kind,
                    probe=probe,
                    slot_df=slot_df,
                    source_representations=dev_x,
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    layer_idx=layer_idx,
                    batch_size=int(cfg["evaluation_batch_size"]),
                    device=device,
                    target_representations=reps["target_rows"],
                )
                base_delta_norms = np.linalg.norm(base_deltas, axis=1).astype(np.float64)
                exact_patch_norms = reps["exact_patch_norms"].astype(np.float64)

                if "fixed" in search_methods:
                    fixed = _evaluate_with_delta_rows(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        valid_df=slot_df,
                        delta_rows=base_deltas,
                        layer_idx=layer_idx,
                        tracked_values=tracked_values,
                        tracked_token_ids=tracked_token_ids,
                        batch_size=int(cfg["evaluation_batch_size"]),
                        device=device,
                    )
                    fixed["delta_norm"] = base_delta_norms
                    fixed["base_delta_norm"] = base_delta_norms
                    fixed["exact_patch_norm"] = exact_patch_norms
                    fixed["selected_scale"] = 1.0
                    fixed["search_score_gain"] = np.nan
                    fixed["search_objective"] = np.nan
                    fixed["norm_budget"] = np.nan
                    fixed["budget_multiplier"] = np.nan
                    fixed["search_tier"] = "native_fixed"
                    fixed["method"] = "fixed"
                    fixed["architecture"] = str(architecture)
                    fixed["delta_kind"] = delta_kind
                    method_parts.append(fixed)

                if "scale_search" in search_methods:
                    scale = _optimize_scale_search(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        valid_df=slot_df,
                        base_deltas=base_deltas,
                        layer_idx=layer_idx,
                        tracked_values=tracked_values,
                        tracked_token_ids=tracked_token_ids,
                        batch_size=int(cfg["evaluation_batch_size"]),
                        device=device,
                        objective=str(search_cfg["objective"]),
                        steps=int(search_cfg["steps"]),
                        lr=float(search_cfg["lr"]),
                        l2_penalty=float(search_cfg["l2_penalty"]),
                        max_scale=float(search_cfg["max_scale"]),
                    )
                    scale["norm_budget"] = np.nan
                    scale["budget_multiplier"] = np.nan
                    scale["base_delta_norm"] = base_delta_norms
                    scale["exact_patch_norm"] = exact_patch_norms
                    scale["search_tier"] = "native_scale_search"
                    scale["method"] = "scale_search"
                    scale["architecture"] = str(architecture)
                    scale["delta_kind"] = delta_kind
                    method_parts.append(scale)

                if "probe_path_search" in search_methods:
                    path_positions = _path_search_positions(search_cfg)
                    path_deltas = probe_path_deltas(
                        probe,
                        source_values=slot_df["source_value"].to_numpy(dtype=np.int64),
                        target_values=slot_df["target_value"].to_numpy(dtype=np.int64),
                        path_positions=path_positions,
                    )
                    path_parts: list[pd.DataFrame] = []
                    for position_idx, path_position in enumerate(path_positions.tolist()):
                        path = _evaluate_with_delta_rows(
                            model=loaded.model,
                            tokenizer=loaded.tokenizer,
                            valid_df=slot_df,
                            delta_rows=path_deltas[:, position_idx, :],
                            layer_idx=layer_idx,
                            tracked_values=tracked_values,
                            tracked_token_ids=tracked_token_ids,
                            batch_size=int(cfg["evaluation_batch_size"]),
                            device=device,
                        )
                        path["_path_row_idx"] = np.arange(len(slot_df), dtype=np.int64)
                        path["path_position"] = float(path_position)
                        path["path_position_idx"] = int(position_idx)
                        path["delta_norm"] = np.linalg.norm(path_deltas[:, position_idx, :], axis=1).astype(np.float64)
                        path_parts.append(path)
                    path_all = pd.concat(path_parts, ignore_index=True)
                    path_all["_path_objective"] = path_all["target_prob"].astype(float)
                    best_idx = path_all.groupby("_path_row_idx")["_path_objective"].idxmax().to_numpy()
                    path_best = path_all.loc[best_idx].sort_values("_path_row_idx").reset_index(drop=True)
                    path_best = path_best.drop(columns=["_path_row_idx", "_path_objective"])
                    path_best["selected_scale"] = path_best["path_position"].astype(float)
                    path_best["search_score_gain"] = path_best["target_prob"].to_numpy(dtype=np.float64) - slot_df["baseline_target_prob"].to_numpy(dtype=np.float64)
                    path_best["search_objective"] = path_best["target_prob"].to_numpy(dtype=np.float64)
                    path_best["norm_budget"] = np.nan
                    path_best["budget_multiplier"] = np.nan
                    path_best["base_delta_norm"] = base_delta_norms
                    path_best["exact_patch_norm"] = exact_patch_norms
                    path_best["search_tier"] = "probe_path_search"
                    path_best["method"] = "probe_path_search"
                    path_best["architecture"] = str(architecture)
                    path_best["delta_kind"] = delta_kind
                    method_parts.append(path_best)

                if "manifold_search" in search_methods:
                    norm_budgets = base_delta_norms * float(search_cfg["manifold_norm_budget_multiplier"])
                    manifold = _optimize_basis_search(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        valid_df=slot_df,
                        basis_matrix=family_basis,
                        norm_budgets=norm_budgets,
                        layer_idx=layer_idx,
                        tracked_values=tracked_values,
                        tracked_token_ids=tracked_token_ids,
                        batch_size=int(cfg["evaluation_batch_size"]),
                        device=device,
                        objective=str(search_cfg["objective"]),
                        steps=int(search_cfg["steps"]),
                        lr=float(search_cfg["lr"]),
                        l2_penalty=float(search_cfg["l2_penalty"]),
                    )
                    manifold["budget_multiplier"] = float(search_cfg["manifold_norm_budget_multiplier"])
                    manifold["base_delta_norm"] = base_delta_norms
                    manifold["exact_patch_norm"] = exact_patch_norms
                    manifold["search_tier"] = "native_norm_budget_manifold_search"
                    manifold["method"] = "manifold_search"
                    manifold["architecture"] = str(architecture)
                    manifold["delta_kind"] = delta_kind
                    method_parts.append(manifold)

                if "equal_budget_search" in search_methods:
                    for multiplier in equal_budget_multipliers:
                        norm_budgets = exact_patch_norms * float(multiplier)
                        equal_budget = _optimize_basis_search(
                            model=loaded.model,
                            tokenizer=loaded.tokenizer,
                            valid_df=slot_df,
                            basis_matrix=family_basis,
                            norm_budgets=norm_budgets,
                            layer_idx=layer_idx,
                            tracked_values=tracked_values,
                            tracked_token_ids=tracked_token_ids,
                            batch_size=int(cfg["evaluation_batch_size"]),
                            device=device,
                            objective=str(search_cfg["objective"]),
                            steps=int(search_cfg["steps"]),
                            lr=float(search_cfg["lr"]),
                            l2_penalty=float(search_cfg["l2_penalty"]),
                        )
                        equal_budget["budget_multiplier"] = float(multiplier)
                        equal_budget["base_delta_norm"] = base_delta_norms
                        equal_budget["exact_patch_norm"] = exact_patch_norms
                        equal_budget["search_tier"] = "equal_budget_search"
                        equal_budget["method"] = "equal_budget_search"
                        equal_budget["architecture"] = str(architecture)
                        equal_budget["delta_kind"] = delta_kind
                        method_parts.append(equal_budget)

                if "unconstrained_manifold_search" in search_methods:
                    unconstrained = _optimize_basis_search(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        valid_df=slot_df,
                        basis_matrix=family_basis,
                        norm_budgets=None,
                        layer_idx=layer_idx,
                        tracked_values=tracked_values,
                        tracked_token_ids=tracked_token_ids,
                        batch_size=int(cfg["evaluation_batch_size"]),
                        device=device,
                        objective=str(search_cfg["objective"]),
                        steps=int(search_cfg["steps"]),
                        lr=float(search_cfg["lr"]),
                        l2_penalty=float(search_cfg.get("unconstrained_l2_penalty", 0.0)),
                    )
                    unconstrained["budget_multiplier"] = np.nan
                    unconstrained["base_delta_norm"] = base_delta_norms
                    unconstrained["exact_patch_norm"] = exact_patch_norms
                    unconstrained["search_tier"] = "unconstrained_manifold_search"
                    unconstrained["method"] = "unconstrained_manifold_search"
                    unconstrained["architecture"] = str(architecture)
                    unconstrained["delta_kind"] = delta_kind
                    method_parts.append(unconstrained)

            arch_detail = pd.concat(method_parts, ignore_index=True)
            detail_frames.append(arch_detail)

            slot_summary = summarize_steering_metrics(
                arch_detail,
                group_cols=["architecture", "delta_kind", "method", "target_slot", "distance_bucket"],
            )
            overall_summary = summarize_steering_metrics(
                arch_detail,
                group_cols=["architecture", "delta_kind", "method"],
            )
            summary_frames.extend([slot_summary, overall_summary])
            overall_rows.extend(overall_summary.to_dict("records"))

    reference_bases: dict[str, np.ndarray] = {}
    for architecture, probe_cfg in cfg.get("probe_specs", {}).items():
        try:
            ref_probe = fit_magnitude_probe(
                architecture=str(architecture),
                train_x=train_x,
                train_values=train_values,
                config=dict(probe_cfg),
                value_min=int(cfg.get("value_min", 0)),
                value_max=int(cfg.get("value_max", 999)),
            )
            reference_bases[str(architecture)] = intervention_family_basis(ref_probe)
        except Exception:
            pass

    for architecture, agop_cfg in dict(cfg.get("agop_specs", {})).items():
        agop_family, agop_diag = fit_agop_steering_family(
            architecture=str(architecture),
            train_x=train_x,
            train_values=train_values,
            dev_x=dev_x,
            dev_values=dev_values,
            config=dict(agop_cfg),
            value_min=int(cfg.get("value_min", 0)),
            value_max=int(cfg.get("value_max", 999)),
            reference_bases=reference_bases,
        )
        agop_diag_frames.append(agop_diag)
        family_basis = np.asarray(agop_family.basis_matrix, dtype=np.float64)
        agop_method_parts: list[pd.DataFrame] = []

        for target_slot in sorted(interventions["target_slot"].unique().tolist()):
            slot_df = interventions.loc[interventions["target_slot"] == int(target_slot)].copy().reset_index(drop=True)
            reps = slot_representations(int(target_slot), slot_df)
            base_deltas = agop_family.intervention_delta(
                source_values=slot_df["source_value"].to_numpy(dtype=np.int64),
                target_values=slot_df["target_value"].to_numpy(dtype=np.int64),
            )
            base_delta_norms = np.linalg.norm(base_deltas, axis=1).astype(np.float64)
            exact_patch_norms = reps["exact_patch_norms"].astype(np.float64)

            if "fixed" in search_methods:
                fixed = _evaluate_with_delta_rows(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    valid_df=slot_df,
                    delta_rows=base_deltas,
                    layer_idx=layer_idx,
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                    batch_size=int(cfg["evaluation_batch_size"]),
                    device=device,
                )
                fixed["delta_norm"] = base_delta_norms
                fixed["base_delta_norm"] = base_delta_norms
                fixed["exact_patch_norm"] = exact_patch_norms
                fixed["selected_scale"] = 1.0
                fixed["search_score_gain"] = np.nan
                fixed["search_objective"] = np.nan
                fixed["norm_budget"] = np.nan
                fixed["budget_multiplier"] = np.nan
                fixed["search_tier"] = "native_fixed"
                fixed["method"] = "fixed"
                fixed["architecture"] = str(architecture)
                fixed["delta_kind"] = "value_code"
                agop_method_parts.append(fixed)

            if "scale_search" in search_methods:
                scale = _optimize_scale_search(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    valid_df=slot_df,
                    base_deltas=base_deltas,
                    layer_idx=layer_idx,
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                    batch_size=int(cfg["evaluation_batch_size"]),
                    device=device,
                    objective=str(search_cfg["objective"]),
                    steps=int(search_cfg["steps"]),
                    lr=float(search_cfg["lr"]),
                    l2_penalty=float(search_cfg["l2_penalty"]),
                    max_scale=float(search_cfg["max_scale"]),
                )
                scale["norm_budget"] = np.nan
                scale["budget_multiplier"] = np.nan
                scale["base_delta_norm"] = base_delta_norms
                scale["exact_patch_norm"] = exact_patch_norms
                scale["search_tier"] = "native_scale_search"
                scale["method"] = "scale_search"
                scale["architecture"] = str(architecture)
                scale["delta_kind"] = "value_code"
                agop_method_parts.append(scale)

            if "manifold_search" in search_methods:
                norm_budgets = base_delta_norms * float(search_cfg["manifold_norm_budget_multiplier"])
                manifold = _optimize_basis_search(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    valid_df=slot_df,
                    basis_matrix=family_basis,
                    norm_budgets=norm_budgets,
                    layer_idx=layer_idx,
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                    batch_size=int(cfg["evaluation_batch_size"]),
                    device=device,
                    objective=str(search_cfg["objective"]),
                    steps=int(search_cfg["steps"]),
                    lr=float(search_cfg["lr"]),
                    l2_penalty=float(search_cfg["l2_penalty"]),
                )
                manifold["budget_multiplier"] = float(search_cfg["manifold_norm_budget_multiplier"])
                manifold["base_delta_norm"] = base_delta_norms
                manifold["exact_patch_norm"] = exact_patch_norms
                manifold["search_tier"] = "native_norm_budget_manifold_search"
                manifold["method"] = "manifold_search"
                manifold["architecture"] = str(architecture)
                manifold["delta_kind"] = "value_code"
                agop_method_parts.append(manifold)

            if "equal_budget_search" in search_methods:
                for multiplier in equal_budget_multipliers:
                    norm_budgets = exact_patch_norms * float(multiplier)
                    equal_budget = _optimize_basis_search(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        valid_df=slot_df,
                        basis_matrix=family_basis,
                        norm_budgets=norm_budgets,
                        layer_idx=layer_idx,
                        tracked_values=tracked_values,
                        tracked_token_ids=tracked_token_ids,
                        batch_size=int(cfg["evaluation_batch_size"]),
                        device=device,
                        objective=str(search_cfg["objective"]),
                        steps=int(search_cfg["steps"]),
                        lr=float(search_cfg["lr"]),
                        l2_penalty=float(search_cfg["l2_penalty"]),
                    )
                    equal_budget["budget_multiplier"] = float(multiplier)
                    equal_budget["base_delta_norm"] = base_delta_norms
                    equal_budget["exact_patch_norm"] = exact_patch_norms
                    equal_budget["search_tier"] = "equal_budget_search"
                    equal_budget["method"] = "equal_budget_search"
                    equal_budget["architecture"] = str(architecture)
                    equal_budget["delta_kind"] = "value_code"
                    agop_method_parts.append(equal_budget)

            if "unconstrained_manifold_search" in search_methods:
                unconstrained = _optimize_basis_search(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    valid_df=slot_df,
                    basis_matrix=family_basis,
                    norm_budgets=None,
                    layer_idx=layer_idx,
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                    batch_size=int(cfg["evaluation_batch_size"]),
                    device=device,
                    objective=str(search_cfg["objective"]),
                    steps=int(search_cfg["steps"]),
                    lr=float(search_cfg["lr"]),
                    l2_penalty=float(search_cfg.get("unconstrained_l2_penalty", 0.0)),
                )
                unconstrained["budget_multiplier"] = np.nan
                unconstrained["base_delta_norm"] = base_delta_norms
                unconstrained["exact_patch_norm"] = exact_patch_norms
                unconstrained["search_tier"] = "unconstrained_manifold_search"
                unconstrained["method"] = "unconstrained_manifold_search"
                unconstrained["architecture"] = str(architecture)
                unconstrained["delta_kind"] = "value_code"
                agop_method_parts.append(unconstrained)

        agop_detail = pd.concat(agop_method_parts, ignore_index=True)
        detail_frames.append(agop_detail)
        slot_summary = summarize_steering_metrics(
            agop_detail,
            group_cols=["architecture", "delta_kind", "method", "target_slot", "distance_bucket"],
        )
        overall_summary = summarize_steering_metrics(
            agop_detail,
            group_cols=["architecture", "delta_kind", "method"],
        )
        summary_frames.extend([slot_summary, overall_summary])
        overall_rows.extend(overall_summary.to_dict("records"))

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    probes_df = pd.DataFrame(probe_rows)
    agop_diagnostics = pd.concat(agop_diag_frames, ignore_index=True) if agop_diag_frames else pd.DataFrame()
    das_diagnostics = pd.concat(das_diag_frames, ignore_index=True) if das_diag_frames else pd.DataFrame()
    circular_das_diagnostics = pd.concat(circular_das_diag_frames, ignore_index=True) if circular_das_diag_frames else pd.DataFrame()

    atomic_write_csv(run_dir / "probes.csv", probes_df)
    atomic_write_csv(run_dir / "agop_diagnostics.csv", agop_diagnostics)
    atomic_write_csv(run_dir / "das_diagnostics.csv", das_diagnostics)
    atomic_write_csv(run_dir / "circular_das_diagnostics.csv", circular_das_diagnostics)
    if das_basis_payload:
        np.savez_compressed(run_dir / "das_bases.npz", **das_basis_payload)
    if circular_das_decoder_payload:
        np.savez_compressed(run_dir / "circular_das_decoders.npz", **circular_das_decoder_payload)
    atomic_write_csv(run_dir / "details.csv", details)
    atomic_write_csv(run_dir / "summary.csv", summary)
    atomic_write_json(
        run_dir / "metrics.json",
        {
            "run_id": run_dir.name,
            "model_alias": model_alias,
            "model_hf_id": model_name,
            "split_name": str(cfg["split_name"]),
            "family": str(cfg["family"]),
            "family_label": str(cfg["family_label"]),
            "intervention_layer_idx": layer_idx,
            "num_train_examples": int(len(train_values)),
            "num_dev_examples": int(len(dev_values)),
            "num_interventions": int(len(interventions)),
            "num_target_slots": int(interventions["target_slot"].nunique()),
            "reference_method_metadata": reference_method_metadata,
            "overall_summary": overall_rows,
        },
    )
    dump_run_config(
        run_dir / "config.json",
        {
            "args": vars(args),
            "experiment_config": cfg,
            "model_entry": model_entry,
            "resolved": {
                "model_name": model_name,
                "model_revision": model_revision,
                "device": device,
                "dtype": dtype,
                "train_shape": list(train_x.shape),
                "dev_shape": list(dev_x.shape),
                "distance_buckets": [[int(lo), int(hi)] for lo, hi in distance_buckets],
            },
        },
    )
    atomic_write_text(run_dir / "DONE", "ok\n")
    print(run_dir)


if __name__ == "__main__":
    main()
