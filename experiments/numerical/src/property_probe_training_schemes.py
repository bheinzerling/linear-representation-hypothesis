from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.number_property_probes import property_labels
from src.property_subspace_synthesis import PropertyFeatureBlock, PropertyFeatureEncoder


@dataclass(frozen=True)
class TrainedPropertyBasis:
    method: str
    basis: np.ndarray
    diagnostics: tuple[dict[str, float | int | str], ...]

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


@dataclass(frozen=True)
class SemanticBasisDecoder:
    method: str
    center: np.ndarray
    basis: np.ndarray
    feature_encoder: PropertyFeatureEncoder
    coefficients: np.ndarray
    alpha: float

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def predict_coordinates(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.feature_encoder.transform(values) @ self.coefficients, dtype=np.float64)

    def synthesize(self, values: np.ndarray) -> np.ndarray:
        coordinates = self.predict_coordinates(values)
        return np.asarray(self.center[None, :] + coordinates @ self.basis.T, dtype=np.float64)


def _validate_xy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_dev = np.asarray(dev_x, dtype=np.float64)
    y_dev = np.asarray(dev_y, dtype=np.float64)
    if x_train.ndim != 2 or y_train.ndim != 2 or x_dev.ndim != 2 or y_dev.ndim != 2:
        raise ValueError("Train/dev activations and targets must all be matrices.")
    if len(x_train) != len(y_train) or len(x_dev) != len(y_dev):
        raise ValueError("Train/dev activation and target row counts do not match.")
    if x_train.shape[1] != x_dev.shape[1] or y_train.shape[1] != y_dev.shape[1]:
        raise ValueError("Train/dev activation or target dimensions do not match.")
    return x_train, y_train, x_dev, y_dev


def _left_basis(matrix: np.ndarray, *, max_rank: int | None = None, rtol: float = 1e-10) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a matrix, got {values.shape}.")
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.zeros((values.shape[0], 0), dtype=np.float64)
    rank = int(np.sum(singular > float(rtol) * float(singular[0])))
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return np.asarray(left[:, :rank], dtype=np.float64)


def _activation_ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    n = max(len(features), 1)
    if alpha < 0.0:
        raise ValueError("Ridge alpha must be non-negative.")
    if alpha == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
        return np.asarray(coefficients, dtype=np.float64)
    kernel = (features @ features.T) / n
    dual = np.linalg.solve(kernel + float(alpha) * np.eye(len(features)), targets)
    return np.asarray((features.T @ dual) / n, dtype=np.float64)


def _activation_ridge_path(
    x: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
) -> list[tuple[float, np.ndarray]]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    n = max(len(features), 1)
    kernel = (features @ features.T) / n
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    projected_targets = eigenvectors.T @ targets
    result: list[tuple[float, np.ndarray]] = []
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        if alpha < 0.0:
            raise ValueError("Ridge alpha must be non-negative.")
        if alpha == 0.0:
            threshold = max(1e-12, 1e-10 * float(eigenvalues[-1]))
            inverse = np.divide(
                1.0,
                eigenvalues,
                out=np.zeros_like(eigenvalues),
                where=eigenvalues > threshold,
            )
        else:
            inverse = 1.0 / (eigenvalues + alpha)
        dual = eigenvectors @ (projected_targets * inverse[:, None])
        coefficients = (features.T @ dual) / n
        result.append((alpha, np.asarray(coefficients, dtype=np.float64)))
    return result


def _feature_ridge_coefficients(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if alpha < 0.0:
        raise ValueError("Ridge alpha must be non-negative.")
    if alpha == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return np.asarray(coefficients, dtype=np.float64)
    gram = (x.T @ x) / max(len(x), 1)
    cross = (x.T @ y) / max(len(x), 1)
    return np.asarray(
        np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), cross),
        dtype=np.float64,
    )


def _fraction_energy_explained(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    return float(1.0 - np.sum((y - pred) ** 2) / max(float(np.sum(y**2)), 1e-24))


def fit_independent_posthoc_svd_basis(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    blocks: Sequence[PropertyFeatureBlock],
    alphas: Sequence[float],
    max_rank: int,
) -> TrainedPropertyBasis:
    x_train, y_train, x_dev, y_dev = _validate_xy(train_x, train_y, dev_x, dev_y)
    property_bases: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | str]] = []
    selected_by_property: dict[str, tuple[float, np.ndarray, float]] = {}
    for alpha, all_coefficients in _activation_ridge_path(x_train, y_train, alphas):
        all_prediction = x_dev @ all_coefficients
        for block in blocks:
            target_dev = y_dev[:, block.start : block.stop]
            coefficients = all_coefficients[:, block.start : block.stop]
            score = _fraction_energy_explained(
                target_dev, all_prediction[:, block.start : block.stop]
            )
            previous = selected_by_property.get(block.property_id)
            if previous is None or (score, -alpha) > (previous[2], -previous[0]):
                selected_by_property[block.property_id] = (alpha, coefficients.copy(), score)
    for block in blocks:
        target_dev = y_dev[:, block.start : block.stop]
        del target_dev
        selected_alpha, selected_coefficients, selected_score = selected_by_property[block.property_id]
        basis = _left_basis(selected_coefficients)
        if basis.shape[1] == 0:
            continue
        property_bases.append(basis)
        diagnostics.append(
            {
                "property_id": block.property_id,
                "selected_alpha": selected_alpha,
                "dev_target_fve": selected_score,
                "property_basis_rank": int(basis.shape[1]),
            }
        )
    if not property_bases:
        raise ValueError("Independent probe training produced no property directions.")
    union_basis = _left_basis(np.concatenate(property_bases, axis=1), max_rank=max_rank)
    return TrainedPropertyBasis(
        method="independent_posthoc_svd",
        basis=union_basis,
        diagnostics=tuple(diagnostics),
    )


def fit_joint_multitask_svd_basis(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    alphas: Sequence[float],
    max_rank: int,
) -> TrainedPropertyBasis:
    x_train, y_train, x_dev, y_dev = _validate_xy(train_x, train_y, dev_x, dev_y)
    candidates: list[tuple[float, np.ndarray, float]] = []
    for alpha, coefficients in _activation_ridge_path(x_train, y_train, alphas):
        score = _fraction_energy_explained(y_dev, x_dev @ coefficients)
        candidates.append((alpha, coefficients, score))
    selected_alpha, selected_coefficients, selected_score = max(
        candidates, key=lambda item: (item[2], -item[0])
    )
    basis = _left_basis(selected_coefficients, max_rank=max_rank)
    if basis.shape[1] == 0:
        raise ValueError("Joint multitask probe training produced no directions.")
    return TrainedPropertyBasis(
        method="joint_multitask_svd",
        basis=basis,
        diagnostics=(
            {
                "selected_alpha": selected_alpha,
                "dev_target_fve": selected_score,
                "property_basis_rank": int(basis.shape[1]),
            },
        ),
    )


def fit_competitive_deflation_basis(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    fit_indices: np.ndarray,
    selection_indices: np.ndarray,
    blocks: Sequence[PropertyFeatureBlock],
    max_rank: int,
    rtol: float = 1e-10,
) -> TrainedPropertyBasis:
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError("Training activations and targets must be row-aligned matrices.")
    fit_rows = np.asarray(fit_indices, dtype=np.int64)
    selection_rows = np.asarray(selection_indices, dtype=np.int64)
    if fit_rows.ndim != 1 or selection_rows.ndim != 1 or not len(fit_rows) or not len(selection_rows):
        raise ValueError("Competitive fit and selection indices must be non-empty vectors.")
    if (
        np.any(fit_rows < 0)
        or np.any(selection_rows < 0)
        or np.any(fit_rows >= len(x_train))
        or np.any(selection_rows >= len(x_train))
    ):
        raise ValueError("Competitive fit or selection indices are out of bounds.")
    if np.intersect1d(fit_rows, selection_rows).size:
        raise ValueError("Competitive fit and selection indices must be disjoint.")

    residual_fit = x_train[fit_rows].copy()
    residual_selection = x_train[selection_rows].copy()
    y_fit = y_train[fit_rows]
    y_selection = y_train[selection_rows]
    directions: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | str]] = []
    fit_cross_covariance = (residual_fit.T @ y_fit) / max(len(residual_fit), 1)
    full_cross_covariance = (x_train.T @ y_train) / max(len(x_train), 1)
    initial_fit_strength = max(float(np.linalg.norm(fit_cross_covariance, ord=2)), 1e-24)
    initial_full_strength = max(float(np.linalg.norm(full_cross_covariance, ord=2)), 1e-24)

    def orthogonal_rank_one_direction(
        cross_covariance: np.ndarray,
        *,
        initial_strength: float,
    ) -> tuple[np.ndarray | None, float]:
        strength = float(np.linalg.norm(cross_covariance, ord=2))
        if not np.isfinite(strength) or strength <= float(rtol) * initial_strength:
            return None, strength
        candidate_basis = _left_basis(cross_covariance, max_rank=1, rtol=rtol)
        if candidate_basis.shape[1] == 0:
            return None, strength
        direction = candidate_basis[:, 0]
        if directions:
            existing = np.column_stack(directions)
            original_norm = float(np.linalg.norm(direction))
            # A second modified Gram-Schmidt pass prevents floating-point remnants
            # from being normalized into a duplicate direction at rank exhaustion.
            for _ in range(2):
                direction = direction - existing @ (existing.T @ direction)
            norm = float(np.linalg.norm(direction))
            if norm <= max(float(rtol), np.sqrt(np.finfo(np.float64).eps)) * max(
                original_norm, 1.0
            ):
                return None, strength
            direction = direction / norm
        return np.asarray(direction, dtype=np.float64), strength

    for iteration in range(int(max_rank)):
        candidates: list[tuple[float, str, np.ndarray, float]] = []
        for block in blocks:
            block_cross = fit_cross_covariance[:, block.start : block.stop]
            direction, strength = orthogonal_rank_one_direction(
                block_cross,
                initial_strength=initial_fit_strength,
            )
            if direction is None:
                continue
            train_scalar = residual_fit @ direction
            selection_scalar = residual_selection @ direction
            target_train = y_fit[:, block.start : block.stop]
            target_selection = y_selection[:, block.start : block.stop]
            denominator = max(float(train_scalar @ train_scalar), 1e-24)
            calibration = (train_scalar[:, None] * target_train).sum(axis=0) / denominator
            prediction = selection_scalar[:, None] * calibration[None, :]
            score = _fraction_energy_explained(target_selection, prediction)
            candidates.append((score, block.property_id, direction, strength))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[3], item[1]), reverse=True)
        selected = None
        for candidate_score, candidate_property, _, candidate_strength in candidates:
            block = next(item for item in blocks if item.property_id == candidate_property)
            refit_direction, refit_strength = orthogonal_rank_one_direction(
                full_cross_covariance[:, block.start : block.stop],
                initial_strength=initial_full_strength,
            )
            if refit_direction is not None:
                selected = (
                    candidate_score,
                    candidate_property,
                    refit_direction,
                    candidate_strength,
                    refit_strength,
                    block,
                )
                break
        if selected is None:
            break
        (
            selection_score_before_refit,
            selected_property,
            selected_direction,
            selected_strength,
            selected_refit_strength,
            selected_block,
        ) = selected
        fit_scalar = residual_fit @ selected_direction
        selection_scalar = residual_selection @ selected_direction
        target_fit = y_fit[:, selected_block.start : selected_block.stop]
        target_selection = y_selection[:, selected_block.start : selected_block.stop]
        denominator = max(float(fit_scalar @ fit_scalar), 1e-24)
        calibration = (fit_scalar[:, None] * target_fit).sum(axis=0) / denominator
        selected_score = _fraction_energy_explained(
            target_selection,
            selection_scalar[:, None] * calibration[None, :],
        )
        directions.append(np.asarray(selected_direction, dtype=np.float64))
        residual_fit -= (residual_fit @ selected_direction)[:, None] * selected_direction[None, :]
        residual_selection -= (
            residual_selection @ selected_direction
        )[:, None] * selected_direction[None, :]
        fit_cross_covariance -= np.outer(
            selected_direction, selected_direction @ fit_cross_covariance
        )
        full_cross_covariance -= np.outer(
            selected_direction, selected_direction @ full_cross_covariance
        )
        score_values = np.asarray([candidate[0] for candidate in candidates], dtype=np.float64)
        diagnostics.append(
            {
                "iteration": iteration,
                "selected_property": selected_property,
                "selected_inner_fve": selected_score,
                "selection_fve_before_full_refit": selection_score_before_refit,
                "selected_cross_covariance_norm": selected_strength,
                "full_refit_cross_covariance_norm": selected_refit_strength,
                "candidate_mean_fve": float(np.mean(score_values)),
                "candidate_max_fve": float(np.max(score_values)),
                "candidate_min_fve": float(np.min(score_values)),
            }
        )
    if not directions:
        raise ValueError("Competitive probe training produced no directions.")
    basis = np.column_stack(directions)
    orthogonality_error = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
    if orthogonality_error > 1e-8:
        raise ValueError(f"Competitive basis lost orthogonality: {orthogonality_error:.3e}")
    return TrainedPropertyBasis(
        method="competitive_deflation",
        basis=np.asarray(basis, dtype=np.float64),
        diagnostics=tuple(diagnostics),
    )


def select_semantic_basis_decoder(
    *,
    method: str,
    train_activations: np.ndarray,
    train_values: np.ndarray,
    dev_activations: np.ndarray,
    dev_values: np.ndarray,
    basis: np.ndarray,
    feature_encoder: PropertyFeatureEncoder,
    ranks: Sequence[int],
    alphas: Sequence[float],
) -> tuple[SemanticBasisDecoder, list[dict[str, float | int | str]]]:
    train_x = np.asarray(train_activations, dtype=np.float64)
    dev_x = np.asarray(dev_activations, dtype=np.float64)
    ordered_basis = np.asarray(basis, dtype=np.float64)
    center = train_x.mean(axis=0)
    train_centered = train_x - center[None, :]
    dev_centered = dev_x - center[None, :]
    train_features = feature_encoder.transform(np.asarray(train_values, dtype=np.int64))
    dev_features = feature_encoder.transform(np.asarray(dev_values, dtype=np.int64))
    total_dev_energy = max(float(np.sum(dev_centered**2)), 1e-24)
    candidate_ranks = sorted({min(max(1, int(rank)), ordered_basis.shape[1]) for rank in ranks})
    rows: list[dict[str, float | int | str]] = []
    candidates: list[SemanticBasisDecoder] = []
    for rank in candidate_ranks:
        selected_basis = ordered_basis[:, :rank]
        train_coordinates = train_centered @ selected_basis
        for alpha in alphas:
            coefficients = _feature_ridge_coefficients(train_features, train_coordinates, float(alpha))
            prediction = dev_features @ coefficients @ selected_basis.T
            dev_fve = float(1.0 - np.sum((dev_centered - prediction) ** 2) / total_dev_energy)
            decoder = SemanticBasisDecoder(
                method=method,
                center=np.asarray(center, dtype=np.float64),
                basis=np.asarray(selected_basis, dtype=np.float64),
                feature_encoder=feature_encoder,
                coefficients=np.asarray(coefficients, dtype=np.float64),
                alpha=float(alpha),
            )
            candidates.append(decoder)
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "alpha": float(alpha),
                    "dev_fraction_variance_explained": dev_fve,
                    "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
                }
            )
    best_index = max(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["dev_fraction_variance_explained"]),
            -int(rows[index]["rank"]),
            -float(rows[index]["alpha"]),
        ),
    )
    rows[best_index]["selected"] = 1
    for index, row in enumerate(rows):
        if index != best_index:
            row["selected"] = 0
    return candidates[best_index], rows


def property_readout_metrics(
    *,
    train_activations: np.ndarray,
    train_values: np.ndarray,
    dev_activations: np.ndarray,
    dev_values: np.ndarray,
    test_activations: np.ndarray,
    test_values: np.ndarray,
    basis: np.ndarray,
    feature_encoder: PropertyFeatureEncoder,
    alphas: Sequence[float],
) -> list[dict[str, float | int | str]]:
    selected_basis = np.asarray(basis, dtype=np.float64)
    train_z = np.asarray(train_activations, dtype=np.float64) @ selected_basis
    dev_z = np.asarray(dev_activations, dtype=np.float64) @ selected_basis
    test_z = np.asarray(test_activations, dtype=np.float64) @ selected_basis
    train_y = feature_encoder.transform(np.asarray(train_values, dtype=np.int64))
    dev_y = feature_encoder.transform(np.asarray(dev_values, dtype=np.int64))
    test_y = feature_encoder.transform(np.asarray(test_values, dtype=np.int64))
    rows: list[dict[str, float | int | str]] = []
    for block in feature_encoder.blocks:
        block_train = train_y[:, block.start : block.stop]
        block_dev = dev_y[:, block.start : block.stop]
        block_test = test_y[:, block.start : block.stop]
        candidates: list[tuple[float, np.ndarray, float]] = []
        for alpha in alphas:
            coefficients = _feature_ridge_coefficients(train_z, block_train, float(alpha))
            score = _fraction_energy_explained(block_dev, dev_z @ coefficients)
            candidates.append((float(alpha), coefficients, score))
        selected_alpha, selected_coefficients, dev_score = max(
            candidates, key=lambda item: (item[2], -item[0])
        )
        prediction = test_z @ selected_coefficients
        row: dict[str, float | int | str] = {
            "property_id": block.property_id,
            "kind": block.kind,
            "selected_alpha": selected_alpha,
            "dev_target_fve": dev_score,
            "test_target_fve": _fraction_energy_explained(block_test, prediction),
        }
        raw_prediction = prediction * block.block_scale + block.train_mean[None, :]
        if block.kind == "continuous":
            truth = np.asarray(test_values, dtype=np.float64)
            decoded = raw_prediction[:, 0]
            denominator = max(float(np.sum((truth - truth.mean()) ** 2)), 1e-24)
            row.update(
                {
                    "test_r2": float(1.0 - np.sum((truth - decoded) ** 2) / denominator),
                    "test_mae": float(np.mean(np.abs(truth - decoded))),
                    "test_accuracy": float("nan"),
                    "test_balanced_accuracy": float("nan"),
                }
            )
        else:
            if block.classes is None:
                raise ValueError(f"Categorical property {block.property_id!r} has no classes.")
            truth = property_labels(block.property_id, np.asarray(test_values, dtype=np.int64))
            decoded = block.classes[np.argmax(raw_prediction, axis=1)]
            class_recalls = [
                float(np.mean(decoded[truth == label] == label))
                for label in block.classes
                if np.any(truth == label)
            ]
            row.update(
                {
                    "test_r2": float("nan"),
                    "test_mae": float("nan"),
                    "test_accuracy": float(np.mean(decoded == truth)),
                    "test_balanced_accuracy": float(np.mean(class_recalls)),
                }
            )
        rows.append(row)
    return rows


def subspace_similarity(left_basis: np.ndarray, right_basis: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left_basis, dtype=np.float64)
    right = np.asarray(right_basis, dtype=np.float64)
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    overlap_trace = float(np.sum(singular**2))
    min_rank = min(left.shape[1], right.shape[1])
    return {
        "left_rank": int(left.shape[1]),
        "right_rank": int(right.shape[1]),
        "mean_squared_canonical_correlation": float(overlap_trace / max(min_rank, 1)),
        "symmetric_projector_similarity": float(
            overlap_trace / max(float(np.sqrt(left.shape[1] * right.shape[1])), 1.0)
        ),
        "minimum_canonical_correlation": float(singular[-1]) if singular.size else float("nan"),
        "maximum_principal_angle_degrees": float(
            np.degrees(np.arccos(np.clip(singular[-1], -1.0, 1.0)))
        )
        if singular.size
        else float("nan"),
    }
