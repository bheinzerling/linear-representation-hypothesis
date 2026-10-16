from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def is_prime(value: int) -> int:
    value = int(value)
    if value < 2:
        return 0
    if value % 2 == 0:
        return int(value == 2)
    for divisor in range(3, isqrt(value) + 1, 2):
        if value % divisor == 0:
            return 0
    return 1


def digit_sum(value: int) -> int:
    return sum(int(digit) for digit in str(abs(int(value))))


def is_perfect_square(value: int) -> int:
    value = int(value)
    return int(value >= 0 and isqrt(value) ** 2 == value)


def is_perfect_cube(value: int) -> int:
    value = int(value)
    if value < 0:
        return 0
    root = round(value ** (1.0 / 3.0))
    return int(any(candidate >= 0 and candidate**3 == value for candidate in range(max(0, root - 1), root + 2)))


def is_power_of_two(value: int) -> int:
    value = int(value)
    return int(value > 0 and value & (value - 1) == 0)


def is_power_of_ten(value: int) -> int:
    value = int(value)
    if value < 1:
        return 0
    while value % 10 == 0:
        value //= 10
    return int(value == 1)


def is_triangular(value: int) -> int:
    value = int(value)
    if value < 0:
        return 0
    discriminant_root = isqrt(8 * value + 1)
    return int(discriminant_root**2 == 8 * value + 1)


def is_fibonacci(value: int) -> int:
    value = int(value)
    if value < 0:
        return 0
    return int(isqrt(5 * value * value + 4) ** 2 == 5 * value * value + 4 or isqrt(5 * value * value - 4) ** 2 == 5 * value * value - 4)


PROPERTY_LABELS: Mapping[str, Callable[[int], int]] = {
    "parity": lambda value: int(value) % 2,
    "primality": is_prime,
    "divisible_by_3": lambda value: int(int(value) % 3 == 0),
    "divisible_by_5": lambda value: int(int(value) % 5 == 0),
    "divisible_by_10": lambda value: int(int(value) % 10 == 0),
    "digit_count": lambda value: len(str(abs(int(value)))),
    "units_digit": lambda value: int(value) % 10,
    "tens_digit": lambda value: (int(value) // 10) % 10,
    "hundreds_digit": lambda value: (int(value) // 100) % 10,
    # Common divisibility and modular arithmetic properties.
    "divisible_by_4": lambda value: int(int(value) % 4 == 0),
    "divisible_by_6": lambda value: int(int(value) % 6 == 0),
    "divisible_by_8": lambda value: int(int(value) % 8 == 0),
    "divisible_by_9": lambda value: int(int(value) % 9 == 0),
    "divisible_by_11": lambda value: int(int(value) % 11 == 0),
    "divisible_by_12": lambda value: int(int(value) % 12 == 0),
    "divisible_by_25": lambda value: int(int(value) % 25 == 0),
    "divisible_by_100": lambda value: int(int(value) % 100 == 0),
    "residue_mod_3": lambda value: int(value) % 3,
    "residue_mod_4": lambda value: int(value) % 4,
    "residue_mod_9": lambda value: int(value) % 9,
    # Decimal-composition properties not reducible to one position label.
    "digit_sum": digit_sum,
    "digital_root": lambda value: 0 if int(value) == 0 else 1 + (abs(int(value)) - 1) % 9,
    "contains_zero": lambda value: int("0" in str(abs(int(value)))),
    "has_repeated_digit": lambda value: int(len(set(str(abs(int(value))))) < len(str(abs(int(value))))),
    "distinct_digit_count": lambda value: len(set(str(abs(int(value))))),
    "palindrome": lambda value: int(str(abs(int(value))) == str(abs(int(value)))[::-1]),
    # Familiar named integer sets. These are rare in 0..999 and therefore
    # require the value-disjoint audit before being interpreted as abstractions.
    "perfect_square": is_perfect_square,
    "perfect_cube": is_perfect_cube,
    "power_of_two": is_power_of_two,
    "power_of_ten": is_power_of_ten,
    "triangular": is_triangular,
    "fibonacci": is_fibonacci,
}


def property_labels(property_id: str, values: Sequence[int] | np.ndarray) -> np.ndarray:
    try:
        label = PROPERTY_LABELS[str(property_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown number property: {property_id!r}") from exc
    return np.asarray([label(int(value)) for value in values], dtype=np.int64)


@dataclass(frozen=True)
class LinearPropertyProbe:
    property_id: str
    classes: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    class_score_centroids: np.ndarray
    alpha: float

    def scores(self, activations: np.ndarray) -> np.ndarray:
        scores = np.asarray(activations, dtype=np.float64) @ self.coefficients.T + self.intercept[None, :]
        return np.asarray(scores, dtype=np.float64)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        scores = self.scores(activations)
        if len(self.classes) == 2:
            indices = (scores[:, 0] > 0.0).astype(np.int64)
        else:
            indices = np.argmax(scores, axis=1)
        return self.classes[indices]

    def row_basis(self, *, orthogonal_to: np.ndarray | None = None, rtol: float = 1e-10) -> np.ndarray:
        rows = np.asarray(self.coefficients, dtype=np.float64)
        if orthogonal_to is not None:
            direction = np.asarray(orthogonal_to, dtype=np.float64).reshape(-1)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            rows = rows - (rows @ direction)[:, None] * direction[None, :]
        _, singular, vh = np.linalg.svd(rows, full_matrices=False)
        if singular.size == 0 or singular[0] <= 0.0:
            return np.empty((0, rows.shape[1]), dtype=np.float64)
        rank = int(np.sum(singular > float(rtol) * singular[0]))
        return np.asarray(vh[:rank], dtype=np.float64)

    def class_score_delta(self, source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
        source_labels = property_labels(self.property_id, source_values)
        target_labels = property_labels(self.property_id, target_values)
        class_to_index = {int(label): index for index, label in enumerate(self.classes.tolist())}
        source_idx = np.asarray([class_to_index[int(label)] for label in source_labels], dtype=np.int64)
        target_idx = np.asarray([class_to_index[int(label)] for label in target_labels], dtype=np.int64)
        return self.class_score_centroids[target_idx] - self.class_score_centroids[source_idx]

    def minimum_norm_delta(
        self,
        source_values: np.ndarray,
        target_values: np.ndarray,
        *,
        orthogonal_to: np.ndarray | None = None,
    ) -> np.ndarray:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if orthogonal_to is not None:
            direction = np.asarray(orthogonal_to, dtype=np.float64).reshape(-1)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            coefficients = coefficients - (coefficients @ direction)[:, None] * direction[None, :]
        score_delta = self.class_score_delta(source_values, target_values)
        return np.asarray(score_delta @ np.linalg.pinv(coefficients.T, rcond=1e-6), dtype=np.float64)


def _as_score_matrix(model: RidgeClassifier, activations: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(np.asarray(activations, dtype=np.float64)), dtype=np.float64)
    return scores[:, None] if scores.ndim == 1 else scores


def fit_property_probe(
    *,
    property_id: str,
    train_activations: np.ndarray,
    train_values: np.ndarray,
    dev_activations: np.ndarray,
    dev_values: np.ndarray,
    alphas: Sequence[float],
) -> tuple[LinearPropertyProbe, list[dict[str, float]]]:
    train_y = property_labels(property_id, train_values)
    dev_y = property_labels(property_id, dev_values)
    candidates: list[tuple[float, float, RidgeClassifier]] = []
    trace: list[dict[str, float]] = []
    for alpha in alphas:
        model = RidgeClassifier(alpha=float(alpha)).fit(np.asarray(train_activations, dtype=np.float64), train_y)
        prediction = model.predict(np.asarray(dev_activations, dtype=np.float64))
        balanced = float(balanced_accuracy_score(dev_y, prediction))
        accuracy = float(accuracy_score(dev_y, prediction))
        trace.append({"alpha": float(alpha), "dev_balanced_accuracy": balanced, "dev_accuracy": accuracy})
        candidates.append((balanced, -float(alpha), model))
    _, _, best = max(candidates, key=lambda item: (item[0], item[1]))
    coefficients = np.asarray(best.coef_, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    intercept = np.asarray(best.intercept_, dtype=np.float64).reshape(-1)
    train_scores = _as_score_matrix(best, train_activations)
    classes = np.asarray(best.classes_, dtype=np.int64)
    score_centroids = np.stack([train_scores[train_y == label].mean(axis=0) for label in classes], axis=0)
    return (
        LinearPropertyProbe(
            property_id=str(property_id),
            classes=classes,
            coefficients=coefficients,
            intercept=intercept,
            class_score_centroids=np.asarray(score_centroids, dtype=np.float64),
            alpha=float(best.alpha),
        ),
        trace,
    )


def probe_metrics(probe: LinearPropertyProbe, activations: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    truth = property_labels(probe.property_id, values)
    prediction = probe.predict(activations)
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "num_classes": int(len(np.unique(truth))),
    }


def independent_sum_delta(
    probes: Mapping[str, LinearPropertyProbe],
    property_ids: Sequence[str],
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    deltas = [
        probes[property_id].minimum_norm_delta(
            source_values,
            target_values,
            orthogonal_to=orthogonal_to,
        )
        for property_id in property_ids
    ]
    if not deltas:
        return np.zeros((len(source_values), next(iter(probes.values())).coefficients.shape[1]), dtype=np.float64)
    return np.sum(np.stack(deltas, axis=0), axis=0)


def joint_minimum_norm_delta(
    probes: Mapping[str, LinearPropertyProbe],
    property_ids: Sequence[str],
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    if not property_ids:
        return np.zeros((len(source_values), next(iter(probes.values())).coefficients.shape[1]), dtype=np.float64)
    coefficient_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    direction = None
    if orthogonal_to is not None:
        direction = np.asarray(orthogonal_to, dtype=np.float64).reshape(-1)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    for property_id in property_ids:
        probe = probes[property_id]
        coefficients = np.asarray(probe.coefficients, dtype=np.float64)
        if direction is not None:
            coefficients = coefficients - (coefficients @ direction)[:, None] * direction[None, :]
        coefficient_parts.append(coefficients)
        score_parts.append(probe.class_score_delta(source_values, target_values))
    coefficients = np.concatenate(coefficient_parts, axis=0)
    score_delta = np.concatenate(score_parts, axis=1)
    return np.asarray(score_delta @ np.linalg.pinv(coefficients.T, rcond=1e-6), dtype=np.float64)


def probe_union_basis(
    probes: Mapping[str, LinearPropertyProbe],
    property_ids: Sequence[str],
    *,
    orthogonal_to: np.ndarray | None = None,
    rtol: float = 1e-10,
) -> np.ndarray:
    """Return an orthonormal basis for the union of selected probe row spaces."""
    if not property_ids:
        hidden_dim = int(next(iter(probes.values())).coefficients.shape[1])
        return np.empty((0, hidden_dim), dtype=np.float64)
    rows = np.concatenate(
        [probes[property_id].row_basis(orthogonal_to=orthogonal_to, rtol=rtol) for property_id in property_ids],
        axis=0,
    )
    if rows.shape[0] == 0:
        return rows
    _, singular, vh = np.linalg.svd(rows, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.empty((0, rows.shape[1]), dtype=np.float64)
    rank = int(np.sum(singular > float(rtol) * singular[0]))
    return np.asarray(vh[:rank], dtype=np.float64)


def project_activation_difference(source_activations: np.ndarray, target_activations: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project each contextual source-to-target difference into an orthonormal row basis."""
    source = np.asarray(source_activations, dtype=np.float64)
    target = np.asarray(target_activations, dtype=np.float64)
    row_basis = np.asarray(basis, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError(f"Source and target activations must have the same 2D shape, got {source.shape} and {target.shape}.")
    if row_basis.ndim != 2 or row_basis.shape[1] != source.shape[1]:
        raise ValueError(f"Basis must have shape (rank, {source.shape[1]}), got {row_basis.shape}.")
    difference = target - source
    return np.asarray((difference @ row_basis.T) @ row_basis, dtype=np.float64)


def oracle_joint_projection_delta(
    probes: Mapping[str, LinearPropertyProbe],
    property_ids: Sequence[str],
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    *,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    """Match the contextual target projection in the union of probe row spaces."""
    basis = probe_union_basis(probes, property_ids, orthogonal_to=orthogonal_to)
    return project_activation_difference(source_activations, target_activations, basis)


def oracle_independent_projection_delta(
    probes: Mapping[str, LinearPropertyProbe],
    property_ids: Sequence[str],
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    *,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    """Sum contextual target projections separately, retaining overlap as the literal bag baseline."""
    source = np.asarray(source_activations, dtype=np.float64)
    target = np.asarray(target_activations, dtype=np.float64)
    if not property_ids:
        return np.zeros_like(source)
    parts = [
        project_activation_difference(
            source,
            target,
            probes[property_id].row_basis(orthogonal_to=orthogonal_to),
        )
        for property_id in property_ids
    ]
    return np.asarray(np.sum(np.stack(parts, axis=0), axis=0), dtype=np.float64)


def property_change_count(property_ids: Sequence[str], source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    changed = [property_labels(property_id, source_values) != property_labels(property_id, target_values) for property_id in property_ids]
    return np.sum(np.stack(changed, axis=1), axis=1).astype(np.int64)


def norm_matched_random_deltas(
    reference: np.ndarray,
    *,
    seed: int,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    random = rng.standard_normal(reference.shape)
    if orthogonal_to is not None:
        direction = np.asarray(orthogonal_to, dtype=np.float64).reshape(-1)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        random -= (random @ direction)[:, None] * direction[None, :]
    random_norm = np.linalg.norm(random, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    return random * (reference_norm / np.maximum(random_norm, 1e-12))


def rescale_auxiliary_to_norm(auxiliary: np.ndarray, target_norm: np.ndarray) -> np.ndarray:
    auxiliary = np.asarray(auxiliary, dtype=np.float64)
    target_norm = np.asarray(target_norm, dtype=np.float64).reshape(-1, 1)
    return auxiliary * (target_norm / np.maximum(np.linalg.norm(auxiliary, axis=1, keepdims=True), 1e-12))
