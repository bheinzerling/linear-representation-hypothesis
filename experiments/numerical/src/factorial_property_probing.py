from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss


@dataclass(frozen=True)
class DigitFactorization:
    factorization_id: str
    kind: str
    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.pairs) != 5:
            raise ValueError(f"Expected five digit pairs, got {len(self.pairs)}.")
        flat = [digit for pair in self.pairs for digit in pair]
        if sorted(flat) != list(range(10)):
            raise ValueError(f"Pairs must partition digits 0..9 exactly, got {flat}.")

    @property
    def side_by_digit(self) -> np.ndarray:
        result = np.empty(10, dtype=np.int64)
        for side_zero, side_one in self.pairs:
            result[int(side_zero)] = 0
            result[int(side_one)] = 1
        return result

    @property
    def pair_by_digit(self) -> np.ndarray:
        result = np.empty(10, dtype=np.int64)
        for pair_id, pair in enumerate(self.pairs):
            for digit in pair:
                result[int(digit)] = int(pair_id)
        return result

    def labels(self, values: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        digits = np.asarray(values, dtype=np.int64) % 10
        return self.side_by_digit[digits], self.pair_by_digit[digits]


def parity_factorization() -> DigitFactorization:
    return DigitFactorization(
        factorization_id="parity",
        kind="registered_parity",
        pairs=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
    )


def random_factorizations(
    *,
    count: int,
    seed: int,
) -> tuple[DigitFactorization, ...]:
    if int(count) < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(int(seed))
    digits = np.arange(10, dtype=np.int64)
    parity_side_zero = (0, 2, 4, 6, 8)
    observed: set[tuple[tuple[int, int], ...]] = set()
    result: list[DigitFactorization] = []
    while len(result) < int(count):
        side_zero = set(
            int(value)
            for value in rng.choice(digits, size=5, replace=False).tolist()
        )
        if 0 not in side_zero:
            side_zero = set(range(10)) - side_zero
        side_zero_tuple = tuple(sorted(side_zero))
        if side_zero_tuple == parity_side_zero:
            continue
        side_one = np.asarray(sorted(set(range(10)) - side_zero), dtype=np.int64)
        paired_one = rng.permutation(side_one)
        pairs = tuple(
            (int(left), int(right))
            for left, right in zip(side_zero_tuple, paired_one, strict=True)
        )
        if pairs in observed:
            continue
        observed.add(pairs)
        result.append(
            DigitFactorization(
                factorization_id=f"random_{len(result):03d}",
                kind="matched_random",
                pairs=pairs,
            )
        )
    return tuple(result)


def factorization_manifest_rows(
    factorizations: Sequence[DigitFactorization],
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for factorization in factorizations:
        for pair_id, (side_zero, side_one) in enumerate(factorization.pairs):
            rows.append(
                {
                    "factorization_id": factorization.factorization_id,
                    "kind": factorization.kind,
                    "pair_id": int(pair_id),
                    "side_zero_digit": int(side_zero),
                    "side_one_digit": int(side_one),
                }
            )
    return rows


def raw_feature_transform(
    train_activations: np.ndarray,
) -> tuple[np.ndarray, float]:
    values = np.asarray(train_activations, dtype=np.float64)
    center = values.mean(axis=0)
    variance = np.mean((values - center[None, :]) ** 2, axis=0)
    rms_scale = float(np.sqrt(np.mean(variance)))
    if not np.isfinite(rms_scale) or rms_scale <= 0.0:
        raise ValueError(f"Invalid raw RMS scale: {rms_scale}.")
    return np.asarray(center, dtype=np.float64), rms_scale


def apply_raw_feature_transform(
    activations: np.ndarray,
    *,
    center: np.ndarray,
    rms_scale: float,
) -> np.ndarray:
    return np.asarray(
        (np.asarray(activations, dtype=np.float64) - center[None, :])
        / float(rms_scale),
        dtype=np.float64,
    )


def fit_conditional_logistic(
    *,
    features: np.ndarray,
    side: np.ndarray,
    pair_id: np.ndarray,
    c: float,
    seed: int,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> LogisticRegression:
    values = np.asarray(features, dtype=np.float64)
    pairs = np.asarray(pair_id, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(pairs):
        raise ValueError("Feature and pair rows must align.")
    design = np.concatenate([np.eye(5, dtype=np.float64)[pairs], values], axis=1)
    model = LogisticRegression(
        C=float(c),
        solver="liblinear",
        dual=True,
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(seed),
    )
    return model.fit(design, np.asarray(side, dtype=np.int64))


def conditional_probability(
    model: LogisticRegression,
    *,
    features: np.ndarray,
    pair_id: np.ndarray,
) -> np.ndarray:
    pairs = np.asarray(pair_id, dtype=np.int64)
    design = np.concatenate(
        [
            np.eye(5, dtype=np.float64)[pairs],
            np.asarray(features, dtype=np.float64),
        ],
        axis=1,
    )
    class_one = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    return np.asarray(model.predict_proba(design)[:, class_one], dtype=np.float64)


def binary_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    baseline_probability: float = 0.5,
) -> dict[str, float | int]:
    target = np.asarray(truth, dtype=np.int64)
    prob = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    prediction = (prob >= 0.5).astype(np.int64)
    nll = float(log_loss(target, prob, labels=[0, 1]))
    baseline = np.full(len(target), float(baseline_probability), dtype=np.float64)
    baseline_nll = float(log_loss(target, baseline, labels=[0, 1]))
    return {
        "n": int(len(target)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "nll_nats": nll,
        "baseline_nll_nats": baseline_nll,
        "conditional_bits": float((baseline_nll - nll) / np.log(2.0)),
    }


def contrast_geometry(
    *,
    features: np.ndarray,
    values: np.ndarray,
    factorization: DigitFactorization,
) -> tuple[dict[str, float], np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    digits = np.asarray(values, dtype=np.int64) % 10
    centroids = np.stack(
        [matrix[digits == digit].mean(axis=0) for digit in range(10)],
        axis=0,
    )
    cells = np.empty((2, 5, matrix.shape[1]), dtype=np.float64)
    for pair_id, (side_zero, side_one) in enumerate(factorization.pairs):
        cells[0, pair_id] = centroids[side_zero]
        cells[1, pair_id] = centroids[side_one]
    contrasts = cells[1] - cells[0]
    norms = np.linalg.norm(contrasts, axis=1)
    normalized = contrasts / np.maximum(norms[:, None], 1e-12)
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(5, k=1)]
    singular = np.linalg.svd(contrasts, compute_uv=False)
    total_energy = float(np.sum(singular**2))

    grand = cells.mean(axis=(0, 1))
    side_effect = cells.mean(axis=1) - grand[None, :]
    pair_effect = cells.mean(axis=0) - grand[None, :]
    interaction = (
        cells
        - grand[None, None, :]
        - side_effect[:, None, :]
        - pair_effect[None, :, :]
    )
    side_energy = float(5.0 * np.sum(side_effect**2))
    interaction_energy = float(np.sum(interaction**2))
    denominator = side_energy + interaction_energy
    return (
        {
            "mean_pair_contrast_cosine": float(np.mean(upper)),
            "min_pair_contrast_cosine": float(np.min(upper)),
            "rank1_contrast_energy_fraction": (
                float(singular[0] ** 2 / total_energy) if total_energy > 0.0 else 0.0
            ),
            "side_vs_interaction_energy_fraction": (
                float(side_energy / denominator) if denominator > 0.0 else 0.0
            ),
            "mean_contrast_norm": float(np.mean(norms)),
        },
        np.asarray(contrasts, dtype=np.float64),
    )


def cluster_bootstrap_binary_metrics(
    *,
    truth: np.ndarray,
    probability: np.ndarray,
    clusters: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    target = np.asarray(truth, dtype=np.int64)
    prob = np.asarray(probability, dtype=np.float64)
    cluster_values = np.asarray(clusters)
    unique = np.unique(cluster_values)
    indices = {value: np.flatnonzero(cluster_values == value) for value in unique}
    rng = np.random.default_rng(int(seed))
    balanced: list[float] = []
    bits: list[float] = []
    for _ in range(int(samples)):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[value] for value in drawn])
        metrics = binary_metrics(target[selected], prob[selected])
        balanced.append(float(metrics["balanced_accuracy"]))
        bits.append(float(metrics["conditional_bits"]))
    result: dict[str, float] = {}
    for name, values in (
        ("balanced_accuracy", balanced),
        ("conditional_bits", bits),
    ):
        array = np.asarray(values, dtype=np.float64)
        result[f"{name}_lower"] = float(np.quantile(array, 0.025))
        result[f"{name}_median"] = float(np.quantile(array, 0.5))
        result[f"{name}_upper"] = float(np.quantile(array, 0.975))
    return result


def empirical_upper_tail(
    observed: float,
    controls: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(controls, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("At least one scalar control is required.")
    greater_equal = int(np.sum(values >= float(observed)))
    return {
        "control_count": int(len(values)),
        "controls_greater_equal": greater_equal,
        "empirical_upper_tail_probability": float(
            (1 + greater_equal) / (1 + len(values))
        ),
        "control_percentile_of_observed": float(
            100.0 * np.mean(values <= float(observed))
        ),
        "control_median": float(np.median(values)),
        "control_p95": float(np.quantile(values, 0.95)),
        "control_p99": float(np.quantile(values, 0.99)),
    }
