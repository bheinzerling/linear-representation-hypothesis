from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score


@dataclass(frozen=True)
class DirectionCandidate:
    direction: np.ndarray
    train_prediction: np.ndarray
    dev_prediction: np.ndarray
    test_prediction: np.ndarray
    alpha: float | None
    coefficient_component: int


def stable_unit_direction(direction: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    result = np.asarray(direction, dtype=np.float64).reshape(-1).copy()
    norm = float(np.linalg.norm(result))
    if norm <= float(atol):
        raise ValueError("Cannot normalize a zero direction.")
    result /= norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def project_out_rank_one(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    x = np.asarray(activations, dtype=np.float64)
    unit = stable_unit_direction(direction)
    return np.asarray(x - (x @ unit)[:, None] * unit[None, :], dtype=np.float64)


def categorical_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth)
    pred = np.asarray(prediction)
    return {
        "n": int(len(y)),
        "num_classes": int(len(np.unique(y))),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    return {
        "n": int(len(y)),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0.0 and np.std(pred) > 0.0 else float("nan"),
    }


def _orthogonal_candidate(direction: np.ndarray, existing_basis: np.ndarray, *, atol: float = 1e-10) -> np.ndarray | None:
    candidate = np.asarray(direction, dtype=np.float64).reshape(-1)
    basis = np.asarray(existing_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != candidate.shape[0]:
        raise ValueError(f"Basis/direction mismatch: {basis.shape} vs {candidate.shape}.")
    if basis.shape[1]:
        candidate = candidate - basis @ (basis.T @ candidate)
    norm = float(np.linalg.norm(candidate))
    if norm <= float(atol):
        return None
    return stable_unit_direction(candidate)


def categorical_direction_candidates(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    test_x: np.ndarray,
    existing_basis: np.ndarray,
    alphas: Sequence[float],
    duplicate_cosine: float = 1.0 - 1e-8,
) -> list[DirectionCandidate]:
    train = np.asarray(train_x, dtype=np.float64)
    dev = np.asarray(dev_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    y = np.asarray(train_y)
    candidates: list[DirectionCandidate] = []
    directions: list[np.ndarray] = []
    for alpha in alphas:
        full_probe = RidgeClassifier(alpha=float(alpha)).fit(train, y)
        coefficients = np.asarray(full_probe.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        _, singular_values, right_vectors = np.linalg.svd(coefficients, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] <= 0.0:
            continue
        rank = int(np.sum(singular_values > max(1e-12, 1e-10 * float(singular_values[0]))))
        for component in range(rank):
            direction = _orthogonal_candidate(right_vectors[component], existing_basis)
            if direction is None:
                continue
            if any(abs(float(direction @ prior)) >= float(duplicate_cosine) for prior in directions):
                continue
            train_score = (train @ direction)[:, None]
            scalar_probe = RidgeClassifier(alpha=float(alpha)).fit(train_score, y)
            candidates.append(
                DirectionCandidate(
                    direction=direction,
                    train_prediction=np.asarray(scalar_probe.predict(train_score)),
                    dev_prediction=np.asarray(scalar_probe.predict((dev @ direction)[:, None])),
                    test_prediction=np.asarray(scalar_probe.predict((test @ direction)[:, None])),
                    alpha=float(alpha),
                    coefficient_component=int(component),
                )
            )
            directions.append(direction)
    return candidates


def select_categorical_candidate(candidates: Sequence[DirectionCandidate], dev_y: np.ndarray) -> tuple[int, dict[str, float | int]]:
    if not candidates:
        raise ValueError("No categorical direction candidates were provided.")
    metrics = [categorical_metrics(dev_y, candidate.dev_prediction) for candidate in candidates]
    index = max(
        range(len(candidates)),
        key=lambda i: (
            float(metrics[i]["balanced_accuracy"]),
            float(metrics[i]["accuracy"]),
            -float(candidates[i].alpha or 0.0),
            -int(candidates[i].coefficient_component),
        ),
    )
    return int(index), metrics[index]


def permutation_max_balanced_accuracy(
    *,
    truth: np.ndarray,
    candidate_predictions: np.ndarray,
    observed_max: float,
    num_permutations: int,
    seed: int,
    batch_size: int = 128,
) -> dict[str, float | int]:
    y = np.asarray(truth)
    predictions = np.asarray(candidate_predictions)
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    if predictions.ndim != 2 or predictions.shape[1] != len(y):
        raise ValueError(f"Prediction/truth mismatch: {predictions.shape} vs {y.shape}.")
    classes, counts = np.unique(y, return_counts=True)
    count_by_class = {value.item() if hasattr(value, "item") else value: int(count) for value, count in zip(classes, counts)}
    inverse_count = np.asarray([1.0 / count_by_class[value.item() if hasattr(value, "item") else value] for value in y])
    rng = np.random.default_rng(int(seed))
    null_maxima = np.empty(int(num_permutations), dtype=np.float64)
    for start in range(0, int(num_permutations), int(batch_size)):
        stop = min(start + int(batch_size), int(num_permutations))
        permuted = np.stack([rng.permutation(len(y)) for _ in range(stop - start)], axis=0)
        permuted_y = y[permuted]
        permuted_weights = inverse_count[permuted]
        correct = predictions[None, :, :] == permuted_y[:, None, :]
        scores = np.sum(correct * permuted_weights[:, None, :], axis=2) / float(len(classes))
        null_maxima[start:stop] = np.max(scores, axis=1)
    p_value = (1.0 + float(np.sum(null_maxima >= float(observed_max) - 1e-15))) / (float(num_permutations) + 1.0)
    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_maxima)),
        "null_q95": float(np.quantile(null_maxima, 0.95)),
        "null_q99": float(np.quantile(null_maxima, 0.99)),
        "num_permutations": int(num_permutations),
        "num_candidates": int(predictions.shape[0]),
    }


def pls_x_weight1_candidate(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    test_x: np.ndarray,
    existing_basis: np.ndarray,
    n_components: int,
    scale: bool = False,
) -> tuple[DirectionCandidate, float]:
    train = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    max_components = min(train.shape[0] - 1, train.shape[1])
    resolved_components = max(1, min(int(n_components), int(max_components)))
    model = PLSRegression(n_components=resolved_components, scale=bool(scale)).fit(train, y[:, None])
    direction = _orthogonal_candidate(np.asarray(model.x_weights_[:, 0], dtype=np.float64), existing_basis)
    if direction is None:
        raise ValueError("The first PLS X-weight is already contained in the removed basis.")
    train_score = train @ direction
    slope, intercept = np.polyfit(train_score, y, deg=1)

    audit_model = PLSRegression(n_components=1, scale=bool(scale)).fit(train, y[:, None])
    audit_direction = _orthogonal_candidate(np.asarray(audit_model.x_weights_[:, 0], dtype=np.float64), existing_basis)
    audit_cosine = abs(float(direction @ audit_direction)) if audit_direction is not None else float("nan")

    def predict(x: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(x, dtype=np.float64) @ direction) * slope + intercept, dtype=np.float64)

    return (
        DirectionCandidate(
            direction=direction,
            train_prediction=predict(train),
            dev_prediction=predict(dev_x),
            test_prediction=predict(test_x),
            alpha=None,
            coefficient_component=0,
        ),
        float(audit_cosine),
    )


def permutation_r2(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    observed_r2: float,
    num_permutations: int,
    seed: int,
    batch_size: int = 256,
) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError(f"Prediction/truth mismatch: {pred.shape} vs {y.shape}.")
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError("R2 permutation test requires nonconstant truth values.")
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(num_permutations), dtype=np.float64)
    for start in range(0, int(num_permutations), int(batch_size)):
        stop = min(start + int(batch_size), int(num_permutations))
        permuted = np.stack([rng.permutation(y) for _ in range(stop - start)], axis=0)
        null[start:stop] = 1.0 - np.sum((permuted - pred[None, :]) ** 2, axis=1) / denominator
    p_value = (1.0 + float(np.sum(null >= float(observed_r2) - 1e-15))) / (float(num_permutations) + 1.0)
    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
        "num_permutations": int(num_permutations),
        "num_candidates": 1,
    }
