"""Auditable decoding/steering comparisons with one immutable probe per fit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def value_centroids(activations: np.ndarray, values: np.ndarray, n_values: int = 1000) -> np.ndarray:
    """Match the archived float32 averaging operation, without fitting on labels."""
    x = np.asarray(activations, dtype=np.float32)
    y = np.asarray(values, dtype=np.int64)
    if len(x) != len(y) or not np.array_equal(np.unique(y), np.arange(n_values)):
        raise ValueError("Centroid bank must contain every value with aligned rows.")
    return np.stack([x[y == v].mean(axis=0) for v in range(n_values)])


def validate_partitions(split: dict[str, list[int]]) -> None:
    sets = [set(split[k]) for k in ("train", "dev", "test")]
    if [len(v) for v in sets] != [800, 100, 100]:
        raise ValueError("Expected the archived 800/100/100 split.")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Fitting and evaluation values overlap.")
    if set.union(*sets) != set(range(1000)):
        raise ValueError("Split must partition 0--999.")


def numeric_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    residual = prediction - truth
    variance = float(np.mean((truth - truth.mean()) ** 2))
    return {
        "exact_accuracy": float(np.mean(prediction == truth)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.mean(residual ** 2) / variance) if variance else float("nan"),
    }


def evaluate_decoding(probe: Any, activations: np.ndarray, values: np.ndarray) -> tuple[pd.DataFrame, dict]:
    features = probe.predict_features(activations)
    decoded = probe.predict_values(activations)
    code = probe.encode_values(values)
    details = pd.DataFrame({
        "source_value": values, "probe_prediction": decoded,
        "raw_scalar_prediction": features[:, 0] if features.shape[1] == 1 else np.nan,
        "code_mse": np.mean((features - code) ** 2, axis=1),
    })
    summary = numeric_metrics(values, decoded)
    summary.update(n_rows=len(values), n_values=len(np.unique(values)), code_mse=float(details["code_mse"].mean()))
    if features.shape[1] == 1:
        summary.update({"raw_scalar_" + k: v for k, v in numeric_metrics(values, features[:, 0]).items()})
    return details, summary


def requested_edit(probe: Any, source_activations: np.ndarray, source_values: np.ndarray,
                   target_values: np.ndarray, rule: str) -> np.ndarray:
    target_code = probe.encode_values(target_values)
    if rule == "label_difference":
        source_code = probe.encode_values(source_values)
    elif rule == "prediction_corrected":
        source_code = probe.predict_features(source_activations)
    else:
        raise ValueError(f"Unknown edit rule: {rule}")
    return (target_code - source_code) @ probe.decoder_matrix


def replace_source(row: dict, target: int, tokenizer: Any) -> tuple[str, int]:
    """Replace only the query's recorded number token; keep demonstrations intact."""
    prompt = str(row["prompt"])
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    idx = int(row["token_idx"])
    start, end = encoded["offset_mapping"][idx]
    if prompt[start:end] != str(int(row["source_value"])):
        raise ValueError("Recorded intervention token is not the source numeral.")
    shifted = prompt[:start] + str(int(target)) + prompt[end:]
    tokens = tokenizer(shifted, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, span in enumerate(tokens["offset_mapping"]) if tuple(span) == (start, start + len(str(int(target))))]
    if len(positions) != 1:
        raise ValueError("Counterfactual source numeral is not one token.")
    return shifted, positions[0]


def cluster_accuracy_interval(frame: pd.DataFrame, column: str, seed: int = 1729) -> tuple[float | None, float | None]:
    scores = frame.groupby("source_prompt_id", sort=True)[column].mean().to_numpy(float)
    if not np.any(scores):
        return None, None
    rng = np.random.default_rng(seed)
    means = np.mean(scores[rng.integers(0, len(scores), size=(10000, len(scores)))], axis=1)
    return tuple(float(v) for v in np.quantile(means, [.025, .975]))


def summarize_behavior(frame: pd.DataFrame) -> dict:
    lo, hi = cluster_accuracy_interval(frame, "exact")
    valid = frame["predicted_value"].notna()
    return {
        "n_edits": len(frame), "n_source_prompts": frame["source_prompt_id"].nunique(),
        "exact_count": int(frame["exact"].sum()), "exact_accuracy": float(frame["exact"].mean()),
        "exact_ci_low": lo, "exact_ci_high": hi,
        "within10_accuracy": float(frame["within10"].mean()),
        "nonnumeric_rate": float(1 - valid.mean()),
        "numeric_output_mae": float(np.abs(frame.loc[valid, "predicted_value"] - frame.loc[valid, "shifted_target"]).mean()) if valid.any() else None,
        "source_answer_retention": float(frame["source_answer_retained"].mean()),
        "mean_edit_norm": float(frame["applied_edit_norm"].mean()),
    }
