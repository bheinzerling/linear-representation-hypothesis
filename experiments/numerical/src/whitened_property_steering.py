from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)

from src.number_property_probes import property_labels


MAGNITUDE_ID = "magnitude"


@dataclass(frozen=True)
class CompactWhitening:
    mean: np.ndarray
    basis: np.ndarray
    scales: np.ndarray
    relative_tolerance: float
    singular_values: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def transform(self, activations: np.ndarray) -> np.ndarray:
        centered = np.asarray(activations, dtype=np.float64) - self.mean[None, :]
        return np.asarray(
            (centered @ self.basis) / self.scales[None, :],
            dtype=np.float64,
        )

    def inverse_delta(self, whitened_delta: np.ndarray) -> np.ndarray:
        values = np.asarray(whitened_delta, dtype=np.float64)
        return np.asarray(
            (values * self.scales[None, :]) @ self.basis.T,
            dtype=np.float64,
        )

    def covariance_error(self, train_activations: np.ndarray) -> float:
        whitened = self.transform(train_activations)
        covariance = whitened.T @ whitened / float(len(whitened) - 1)
        return float(np.max(np.abs(covariance - np.eye(self.rank))))


def fit_compact_whitening(
    train_activations: np.ndarray,
    *,
    relative_tolerance: float = 1e-8,
) -> CompactWhitening:
    values = np.asarray(train_activations, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"Expected at least two 2D training rows, got {values.shape}.")
    mean = values.mean(axis=0)
    centered = values - mean[None, :]
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("Training activations have zero centered rank.")
    keep = singular > float(relative_tolerance) * float(singular[0])
    retained_singular = singular[keep]
    basis = right[keep].T
    scales = retained_singular / np.sqrt(float(len(values) - 1))
    orthogonality_error = float(
        np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))
    )
    if orthogonality_error > 2e-7:
        raise ValueError(
            "Compact whitening basis is not orthonormal "
            f"(max error {orthogonality_error:.3e})."
        )
    return CompactWhitening(
        mean=np.asarray(mean, dtype=np.float64),
        basis=np.asarray(basis, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        relative_tolerance=float(relative_tolerance),
        singular_values=np.asarray(singular, dtype=np.float64),
    )


@dataclass(frozen=True)
class RidgePropertyProbe:
    property_id: str
    kind: str
    classes: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    class_score_centroids: np.ndarray
    alpha: float
    target_mean: float
    target_scale: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        return np.asarray(
            values @ self.coefficients.T + self.intercept[None, :],
            dtype=np.float64,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.scores(features)
        if self.kind == "magnitude":
            return scores[:, 0] * self.target_scale + self.target_mean
        return self.classes[np.argmax(scores, axis=1)]

    def score_delta(
        self,
        source_values: np.ndarray,
        target_values: np.ndarray,
    ) -> np.ndarray:
        source = np.asarray(source_values, dtype=np.int64)
        target = np.asarray(target_values, dtype=np.int64)
        if self.kind == "magnitude":
            return ((target - source) / self.target_scale)[:, None]
        source_labels = property_labels(self.property_id, source)
        target_labels = property_labels(self.property_id, target)
        lookup = {int(label): index for index, label in enumerate(self.classes)}
        source_index = np.asarray(
            [lookup[int(label)] for label in source_labels], dtype=np.int64
        )
        target_index = np.asarray(
            [lookup[int(label)] for label in target_labels], dtype=np.int64
        )
        return np.asarray(
            self.class_score_centroids[target_index]
            - self.class_score_centroids[source_index],
            dtype=np.float64,
        )

    def minimum_norm_delta(
        self,
        source_values: np.ndarray,
        target_values: np.ndarray,
        *,
        rcond: float = 1e-8,
    ) -> np.ndarray:
        desired = self.score_delta(source_values, target_values)
        return np.asarray(
            desired @ np.linalg.pinv(self.coefficients.T, rcond=float(rcond)),
            dtype=np.float64,
        )

    def row_basis(self, *, relative_tolerance: float = 1e-10) -> np.ndarray:
        _, singular, right = np.linalg.svd(
            self.coefficients, full_matrices=False
        )
        if singular.size == 0 or singular[0] <= 0.0:
            return np.empty((0, self.coefficients.shape[1]), dtype=np.float64)
        rank = int(
            np.sum(singular > float(relative_tolerance) * float(singular[0]))
        )
        return np.asarray(right[:rank], dtype=np.float64)


class _DualRidgePath:
    def __init__(self, train_features: np.ndarray):
        values = np.asarray(train_features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D features, got {values.shape}.")
        self.feature_mean = values.mean(axis=0)
        self.centered = values - self.feature_mean[None, :]
        gram = self.centered @ self.centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = np.maximum(eigenvalues[order], 0.0)
        self.eigenvectors = eigenvectors[:, order]
        self.tolerance = max(
            1e-12,
            1e-12 * float(self.eigenvalues[0]) if self.eigenvalues.size else 1e-12,
        )

    def fit(self, targets: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        target_mean = y.mean(axis=0)
        centered_target = y - target_mean[None, :]
        projected = self.eigenvectors.T @ centered_target
        denominator = self.eigenvalues + float(alpha)
        inverse = np.divide(
            1.0,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > self.tolerance,
        )
        dual = self.eigenvectors @ (inverse[:, None] * projected)
        coefficients = self.centered.T @ dual
        intercept = target_mean - self.feature_mean @ coefficients
        return (
            np.asarray(coefficients.T, dtype=np.float64),
            np.asarray(intercept, dtype=np.float64),
        )


def _categorical_targets(
    property_id: str,
    values: np.ndarray,
    *,
    classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = property_labels(property_id, values)
    resolved_classes = (
        np.unique(labels)
        if classes is None
        else np.asarray(classes, dtype=np.int64)
    )
    lookup = {int(label): index for index, label in enumerate(resolved_classes)}
    try:
        indices = np.asarray(
            [lookup[int(label)] for label in labels], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(
            f"Property {property_id!r} contains a label absent from training."
        ) from exc
    targets = np.zeros((len(labels), len(resolved_classes)), dtype=np.float64)
    targets[np.arange(len(labels)), indices] = 1.0
    return targets, resolved_classes, labels


def fit_property_probe(
    *,
    property_id: str,
    train_features: np.ndarray,
    train_values: np.ndarray,
    dev_features: np.ndarray,
    dev_values: np.ndarray,
    alphas: Sequence[float],
    path: _DualRidgePath | None = None,
) -> tuple[RidgePropertyProbe, list[dict[str, float]]]:
    train_x = np.asarray(train_features, dtype=np.float64)
    dev_x = np.asarray(dev_features, dtype=np.float64)
    train_values = np.asarray(train_values, dtype=np.int64)
    dev_values = np.asarray(dev_values, dtype=np.int64)
    ridge_path = _DualRidgePath(train_x) if path is None else path
    if property_id == MAGNITUDE_ID:
        target_mean = float(train_values.mean())
        target_scale = float(train_values.std())
        if target_scale <= 0.0:
            raise ValueError("Magnitude target has zero training standard deviation.")
        train_target = ((train_values - target_mean) / target_scale)[:, None]
        dev_truth = dev_values.astype(np.float64)
        classes = np.asarray([], dtype=np.int64)
        kind = "magnitude"
    else:
        train_target, classes, _ = _categorical_targets(property_id, train_values)
        dev_truth = property_labels(property_id, dev_values)
        target_mean = 0.0
        target_scale = 1.0
        kind = "categorical"

    candidates: list[tuple[tuple[float, float], RidgePropertyProbe]] = []
    trace: list[dict[str, float]] = []
    for alpha in alphas:
        coefficients, intercept = ridge_path.fit(train_target, alpha=float(alpha))
        train_scores = train_x @ coefficients.T + intercept[None, :]
        if kind == "magnitude":
            centroids = np.empty((0, 1), dtype=np.float64)
        else:
            train_labels = property_labels(property_id, train_values)
            centroids = np.stack(
                [
                    train_scores[train_labels == label].mean(axis=0)
                    for label in classes
                ],
                axis=0,
            )
        probe = RidgePropertyProbe(
            property_id=str(property_id),
            kind=kind,
            classes=np.asarray(classes, dtype=np.int64),
            coefficients=coefficients,
            intercept=intercept,
            class_score_centroids=np.asarray(centroids, dtype=np.float64),
            alpha=float(alpha),
            target_mean=target_mean,
            target_scale=target_scale,
        )
        prediction = probe.predict(dev_x)
        if kind == "magnitude":
            mae = float(mean_absolute_error(dev_truth, prediction))
            score = (-mae, -float(alpha))
            trace.append(
                {
                    "alpha": float(alpha),
                    "dev_mae": mae,
                    "dev_r2": float(r2_score(dev_truth, prediction)),
                }
            )
        else:
            balanced = float(balanced_accuracy_score(dev_truth, prediction))
            score = (balanced, -float(alpha))
            trace.append(
                {
                    "alpha": float(alpha),
                    "dev_balanced_accuracy": balanced,
                    "dev_accuracy": float(accuracy_score(dev_truth, prediction)),
                }
            )
        candidates.append((score, probe))
    return max(candidates, key=lambda item: item[0])[1], trace


def fit_probe_family(
    *,
    property_ids: Sequence[str],
    train_features: np.ndarray,
    train_values: np.ndarray,
    dev_features: np.ndarray,
    dev_values: np.ndarray,
    alphas: Sequence[float],
) -> tuple[dict[str, RidgePropertyProbe], list[dict[str, float]]]:
    path = _DualRidgePath(train_features)
    probes: dict[str, RidgePropertyProbe] = {}
    trace: list[dict[str, float]] = []
    for property_id in property_ids:
        probe, rows = fit_property_probe(
            property_id=property_id,
            train_features=train_features,
            train_values=train_values,
            dev_features=dev_features,
            dev_values=dev_values,
            alphas=alphas,
            path=path,
        )
        probes[property_id] = probe
        trace.extend({"property_id": property_id, **row} for row in rows)
    return probes, trace


def probe_metrics(
    probe: RidgePropertyProbe,
    features: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int]:
    truth_values = np.asarray(values, dtype=np.int64)
    prediction = probe.predict(features)
    if probe.kind == "magnitude":
        return {
            "n": int(len(values)),
            "mae": float(mean_absolute_error(truth_values, prediction)),
            "r2": float(r2_score(truth_values, prediction)),
            "normalized_mae": float(
                mean_absolute_error(truth_values, prediction) / 999.0
            ),
            "num_classes": 1000,
        }
    truth = property_labels(probe.property_id, truth_values)
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "num_classes": int(len(np.unique(truth))),
    }


def independent_feature_delta(
    probes: Mapping[str, RidgePropertyProbe],
    property_ids: Sequence[str],
    source_values: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    selected = tuple(str(value) for value in property_ids)
    if not selected:
        feature_dim = int(next(iter(probes.values())).coefficients.shape[1])
        return np.zeros((len(source_values), feature_dim), dtype=np.float64)
    return np.sum(
        np.stack(
            [
                probes[property_id].minimum_norm_delta(
                    source_values, target_values
                )
                for property_id in selected
            ],
            axis=0,
        ),
        axis=0,
    )


def greedy_low_overlap_order(
    probes: Mapping[str, RidgePropertyProbe],
    property_ids: Sequence[str],
) -> tuple[str, ...]:
    remaining = list(dict.fromkeys(str(value) for value in property_ids))
    selected: list[str] = []
    selected_basis = np.empty(
        (0, next(iter(probes.values())).coefficients.shape[1]), dtype=np.float64
    )
    while remaining:
        candidates: list[tuple[float, str, np.ndarray]] = []
        for property_id in remaining:
            basis = probes[property_id].row_basis()
            if len(selected_basis):
                residual = basis - (basis @ selected_basis.T) @ selected_basis
            else:
                residual = basis
            novelty = float(
                np.sum(residual * residual) / max(1, basis.shape[0])
            )
            candidates.append((novelty, property_id, basis))
        _, chosen, chosen_basis = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        selected.append(chosen)
        remaining.remove(chosen)
        stacked = np.concatenate([selected_basis, chosen_basis], axis=0)
        _, singular, right = np.linalg.svd(stacked, full_matrices=False)
        rank = int(
            np.sum(
                singular
                > max(float(singular[0]) if singular.size else 0.0, 1e-12)
                * 1e-10
            )
        )
        selected_basis = np.asarray(right[:rank], dtype=np.float64)
    return tuple(selected)
