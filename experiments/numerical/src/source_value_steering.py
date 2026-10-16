from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.magnitude_intervention import MagnitudeProbe


DEFAULT_DISTANCE_BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 5),
    (6, 10),
    (11, 25),
    (26, 50),
    (51, 100),
    (101, 200),
    (201, 300),
    (301, 400),
    (401, 500),
)


def distance_bucket_label(min_abs_delta: int, max_abs_delta: int) -> str:
    if int(min_abs_delta) == int(max_abs_delta):
        return f"{int(min_abs_delta)}"
    return f"{int(min_abs_delta)}-{int(max_abs_delta)}"


def sample_stratified_target_values(
    *,
    source_value: int,
    valid_target_values: Sequence[int] | np.ndarray,
    num_targets: int,
    seed: int,
    distance_buckets: Iterable[tuple[int, int]] = DEFAULT_DISTANCE_BUCKETS,
) -> list[int]:
    target_arr = np.unique(np.asarray(valid_target_values, dtype=np.int64).reshape(-1))
    target_arr = target_arr[target_arr != int(source_value)]
    if len(target_arr) < int(num_targets):
        raise ValueError(
            f"Requested {num_targets} targets for source_value={source_value}, but only {len(target_arr)} valid targets exist."
        )

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    used: set[int] = set()
    source = int(source_value)

    for min_abs_delta, max_abs_delta in distance_buckets:
        min_abs = int(min_abs_delta)
        max_abs = int(max_abs_delta)
        if max_abs < min_abs:
            raise ValueError(f"Invalid distance bucket ({min_abs_delta}, {max_abs_delta}).")
        candidates = target_arr[
            (np.abs(target_arr - source) >= min_abs)
            & (np.abs(target_arr - source) <= max_abs)
        ]
        candidates = np.asarray([value for value in candidates.tolist() if int(value) not in used], dtype=np.int64)
        if len(candidates) == 0:
            continue
        chosen = int(rng.choice(candidates))
        selected.append(chosen)
        used.add(chosen)
        if len(selected) >= int(num_targets):
            return selected

    remaining = np.asarray([value for value in target_arr.tolist() if int(value) not in used], dtype=np.int64)
    if len(remaining) < int(num_targets) - len(selected):
        raise ValueError(
            f"Unable to fill {num_targets} targets for source_value={source_value}; only {len(selected)} selected after bucketing."
        )
    fill_count = int(num_targets) - len(selected)
    fill = rng.choice(remaining, size=fill_count, replace=False)
    selected.extend(int(value) for value in np.asarray(fill, dtype=np.int64).tolist())
    return selected


def valid_source_value_targets_for_row(
    *,
    task: str,
    operand_mode: str,
    x1: int,
    x2: int,
    source_value: int,
    value_min: int,
    value_max: int,
    candidate_values: Sequence[int] | np.ndarray,
) -> np.ndarray:
    candidate_arr = np.unique(np.asarray(candidate_values, dtype=np.int64).reshape(-1))
    candidate_arr = candidate_arr[candidate_arr != int(source_value)]
    task_name = str(task)
    operand = str(operand_mode)
    if task_name == "addition":
        if operand == "x1":
            keep = (candidate_arr + int(x2) >= int(value_min)) & (candidate_arr + int(x2) <= int(value_max))
        elif operand == "x2":
            keep = (int(x1) + candidate_arr >= int(value_min)) & (int(x1) + candidate_arr <= int(value_max))
        else:
            raise ValueError(f"Unsupported addition operand_mode: {operand_mode!r}")
        return candidate_arr[keep]
    if task_name == "subtraction":
        if operand == "x1":
            shifted_target = candidate_arr - int(x2)
        elif operand == "x2":
            shifted_target = int(x1) - candidate_arr
        else:
            raise ValueError(f"Unsupported subtraction operand_mode: {operand_mode!r}")
        keep = (shifted_target >= int(value_min)) & (shifted_target <= int(value_max))
        return candidate_arr[keep]
    if task_name == "identity":
        if operand not in {"self", "x1"}:
            raise ValueError(f"Unsupported identity operand_mode: {operand_mode!r}")
        keep = (candidate_arr >= int(value_min)) & (candidate_arr <= int(value_max))
        return candidate_arr[keep]
    raise ValueError(f"Unsupported task: {task!r}")


def shifted_operands_and_target_for_row(
    *,
    task: str,
    operand_mode: str,
    x1: int,
    x2: int,
    target_value: int,
) -> tuple[int, int, int]:
    task_name = str(task)
    operand = str(operand_mode)
    target = int(target_value)
    if task_name == "addition":
        if operand == "x1":
            shifted_x1, shifted_x2 = target, int(x2)
        elif operand == "x2":
            shifted_x1, shifted_x2 = int(x1), target
        else:
            raise ValueError(f"Unsupported addition operand_mode: {operand_mode!r}")
        return shifted_x1, shifted_x2, shifted_x1 + shifted_x2
    if task_name == "subtraction":
        if operand == "x1":
            shifted_x1, shifted_x2 = target, int(x2)
        elif operand == "x2":
            shifted_x1, shifted_x2 = int(x1), target
        else:
            raise ValueError(f"Unsupported subtraction operand_mode: {operand_mode!r}")
        return shifted_x1, shifted_x2, shifted_x1 - shifted_x2
    if task_name == "identity":
        if operand not in {"self", "x1"}:
            raise ValueError(f"Unsupported identity operand_mode: {operand_mode!r}")
        return target, 0, target
    raise ValueError(f"Unsupported task: {task!r}")


def compute_value_code_probe_delta(
    *,
    probe: MagnitudeProbe,
    source_values: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    return probe.intervention_delta(
        source_values=np.asarray(source_values, dtype=np.int64),
        target_values=np.asarray(target_values, dtype=np.int64),
    ).astype(np.float64)


def compute_projected_patch_probe_delta(
    *,
    probe: MagnitudeProbe,
    source_representations: np.ndarray,
    target_representations: np.ndarray,
) -> np.ndarray:
    source_features = probe.predict_features(np.asarray(source_representations, dtype=np.float64))
    target_features = probe.predict_features(np.asarray(target_representations, dtype=np.float64))
    return np.asarray((target_features - source_features) @ probe.decoder_matrix, dtype=np.float64)


def summarize_steering_metrics(
    results: pd.DataFrame,
    *,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, subset in results.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        payload = {column: value for column, value in zip(group_cols, group_key, strict=False)}
        predicted_numeric = pd.to_numeric(subset["predicted_value"], errors="coerce")
        output_delta = pd.to_numeric(subset["predicted_numeric_delta"], errors="coerce")
        payload.update(
            {
                "num_examples": int(len(subset)),
                "valid_output_rate": float(predicted_numeric.notna().mean()),
                "argmax_accuracy": float(subset["correct"].mean()),
                "mean_target_prob_delta": float(subset["target_prob_gain"].mean()),
                "mean_output_value_delta": float(output_delta.mean()) if output_delta.notna().any() else float("nan"),
                "mean_abs_output_value_delta": float(output_delta.abs().mean()) if output_delta.notna().any() else float("nan"),
                "mean_expected_value_delta": float(subset["expected_shift"].mean()),
                "mean_abs_target_error": float(subset["target_error"].abs().mean()),
                "mean_requested_value_delta": float(subset["delta_value"].mean()),
                "mean_abs_requested_value_delta": float(subset["abs_delta_value"].mean()),
                "mean_delta_norm": float(subset["delta_norm"].mean()) if "delta_norm" in subset else float("nan"),
                "mean_base_delta_norm": float(subset["base_delta_norm"].mean()) if "base_delta_norm" in subset else float("nan"),
                "mean_exact_patch_norm": float(subset["exact_patch_norm"].mean()) if "exact_patch_norm" in subset else float("nan"),
                "mean_norm_budget": float(subset["norm_budget"].mean()) if "norm_budget" in subset else float("nan"),
                "mean_budget_multiplier": float(subset["budget_multiplier"].mean()) if "budget_multiplier" in subset else float("nan"),
                "mean_selected_scale": float(subset["selected_scale"].mean()) if "selected_scale" in subset else float("nan"),
                "mean_search_score_gain": float(subset["search_score_gain"].mean()) if "search_score_gain" in subset else float("nan"),
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)
