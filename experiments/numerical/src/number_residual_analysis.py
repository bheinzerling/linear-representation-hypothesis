from __future__ import annotations

from typing import Sequence

import numpy as np

from src.number_property_probes import property_labels
from src.property_subspace_synthesis import PropertyFeatureEncoder


def _as_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix, got shape {matrix.shape}.")
    return matrix


def validate_orthonormal_basis(basis: np.ndarray, *, hidden_dim: int) -> np.ndarray:
    matrix = _as_matrix(basis, name="basis")
    if matrix.shape[0] != int(hidden_dim) or matrix.shape[1] == 0:
        raise ValueError(
            f"Basis shape {matrix.shape} is incompatible with hidden dimension {hidden_dim}."
        )
    error = float(np.max(np.abs(matrix.T @ matrix - np.eye(matrix.shape[1]))))
    if error > 1e-7:
        raise ValueError(f"Basis is not orthonormal (maximum Gram error {error:.3e}).")
    return matrix


def project_out(centered_activations: np.ndarray, basis: np.ndarray) -> np.ndarray:
    activations = _as_matrix(centered_activations, name="centered_activations")
    directions = validate_orthonormal_basis(basis, hidden_dim=activations.shape[1])
    return np.asarray(activations - (activations @ directions) @ directions.T, dtype=np.float64)


def deterministic_random_basis(*, hidden_dim: int, rank: int, seed: int) -> np.ndarray:
    if not 0 < int(rank) <= int(hidden_dim):
        raise ValueError(f"Rank {rank} must be in 1..{hidden_dim}.")
    generator = np.random.default_rng(int(seed))
    matrix = generator.standard_normal((int(hidden_dim), int(rank)))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return np.asarray(basis, dtype=np.float64)


def removed_energy(centered_activations: np.ndarray, basis: np.ndarray) -> float:
    activations = _as_matrix(centered_activations, name="centered_activations")
    directions = validate_orthonormal_basis(basis, hidden_dim=activations.shape[1])
    return float(np.sum((activations @ directions) ** 2))


def variance_matched_pca_basis(
    train_centered: np.ndarray,
    *,
    target_removed_energy: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Choose leading train PCs whose cumulative energy is closest to a target.

    The target is normally the energy removed by a trained probe basis on the same
    training rows. No dev/test activations influence either the PCs or selected rank.
    """

    activations = _as_matrix(train_centered, name="train_centered")
    target = float(target_removed_energy)
    total = float(np.sum(activations**2))
    if not np.isfinite(target) or target <= 0.0 or target > total * (1.0 + 1e-10):
        raise ValueError(f"Invalid target removed energy {target} for total energy {total}.")
    _, singular_values, right = np.linalg.svd(activations, full_matrices=False)
    energies = singular_values**2
    positive = energies > max(float(energies[0]) * 1e-12, 1e-24)
    energies = energies[positive]
    right = right[positive]
    cumulative = np.cumsum(energies)
    rank = int(np.argmin(np.abs(cumulative - target))) + 1
    basis = np.asarray(right[:rank].T, dtype=np.float64)
    matched = float(cumulative[rank - 1])
    return basis, {
        "rank": rank,
        "target_removed_energy": target,
        "matched_removed_energy": matched,
        "relative_match_error": float(abs(matched - target) / max(target, 1e-24)),
        "target_removed_fraction": float(target / max(total, 1e-24)),
        "matched_removed_fraction": float(matched / max(total, 1e-24)),
    }


def variance_matched_pca_bases(
    train_centered: np.ndarray,
    *,
    target_removed_energies: dict[str, float],
) -> dict[str, tuple[np.ndarray, dict[str, float | int]]]:
    """Select several variance-matched PCA controls from one train-only SVD."""

    activations = _as_matrix(train_centered, name="train_centered")
    total = float(np.sum(activations**2))
    _, singular_values, right = np.linalg.svd(activations, full_matrices=False)
    energies = singular_values**2
    positive = energies > max(float(energies[0]) * 1e-12, 1e-24)
    energies = energies[positive]
    right = right[positive]
    cumulative = np.cumsum(energies)
    results: dict[str, tuple[np.ndarray, dict[str, float | int]]] = {}
    for name, raw_target in target_removed_energies.items():
        target = float(raw_target)
        if not np.isfinite(target) or target <= 0.0 or target > total * (1.0 + 1e-10):
            raise ValueError(f"Invalid target removed energy {target} for total energy {total}.")
        rank = int(np.argmin(np.abs(cumulative - target))) + 1
        matched = float(cumulative[rank - 1])
        results[str(name)] = (
            np.asarray(right[:rank].T, dtype=np.float64),
            {
                "rank": rank,
                "target_removed_energy": target,
                "matched_removed_energy": matched,
                "relative_match_error": float(abs(matched - target) / max(target, 1e-24)),
                "target_removed_fraction": float(target / max(total, 1e-24)),
                "matched_removed_fraction": float(matched / max(total, 1e-24)),
            },
        )
    return results


def residual_spectrum(activations: np.ndarray) -> dict[str, float | int]:
    matrix = _as_matrix(activations, name="activations")
    # The nonzero covariance spectrum is obtained from the smaller row Gram matrix.
    gram = (matrix @ matrix.T) / max(len(matrix) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.size == 0 or float(eigenvalues.sum()) <= 0.0:
        raise ValueError("Activation spectrum has zero total energy.")
    threshold = max(float(eigenvalues[0]) * 1e-10, 1e-15)
    eigenvalues = eigenvalues[eigenvalues > threshold]
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    cumulative = np.cumsum(probabilities)

    def rank_at(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    return {
        "sample_rank": int(len(eigenvalues)),
        "total_variance": float(eigenvalues.sum()),
        "effective_rank_entropy": float(np.exp(entropy)),
        "effective_rank_participation": float(1.0 / np.sum(probabilities**2)),
        "top1_variance_fraction": float(probabilities[0]),
        "top5_variance_fraction": float(probabilities[:5].sum()),
        "rank_at_50pct": rank_at(0.50),
        "rank_at_80pct": rank_at(0.80),
        "rank_at_90pct": rank_at(0.90),
        "rank_at_95pct": rank_at(0.95),
        "rank_at_99pct": rank_at(0.99),
    }


def cosine_similarity_matrix(activations: np.ndarray) -> np.ndarray:
    matrix = _as_matrix(activations, name="activations")
    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.divide(
        matrix,
        norms[:, None],
        out=np.zeros_like(matrix),
        where=norms[:, None] > 1e-15,
    )
    return np.asarray(normalized @ normalized.T, dtype=np.float64)


def centered_kernel_alignment(left_kernel: np.ndarray, right_kernel: np.ndarray) -> float:
    left = _as_matrix(left_kernel, name="left_kernel")
    right = _as_matrix(right_kernel, name="right_kernel")
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError(f"Kernels must be same-sized square matrices, got {left.shape}/{right.shape}.")
    left_centered = left - left.mean(axis=0)[None, :] - left.mean(axis=1)[:, None] + left.mean()
    right_centered = (
        right - right.mean(axis=0)[None, :] - right.mean(axis=1)[:, None] + right.mean()
    )
    numerator = float(np.sum(left_centered * right_centered))
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    return float(numerator / denominator) if denominator > 1e-24 else float("nan")


def off_diagonal_centered_kernel_alignment(
    left_kernel: np.ndarray, right_kernel: np.ndarray
) -> float:
    """Centered kernel alignment excluding self-similarity terms.

    This avoids spuriously large finite-sample alignment between unrelated
    representational and label kernels caused only by their shared diagonals.
    """

    left = _as_matrix(left_kernel, name="left_kernel")
    right = _as_matrix(right_kernel, name="right_kernel")
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError(f"Kernels must be same-sized square matrices, got {left.shape}/{right.shape}.")
    left_centered = left - left.mean(axis=0)[None, :] - left.mean(axis=1)[:, None] + left.mean()
    right_centered = (
        right - right.mean(axis=0)[None, :] - right.mean(axis=1)[:, None] + right.mean()
    )
    upper = np.triu_indices(left.shape[0], k=1)
    left_pairs = left_centered[upper]
    right_pairs = right_centered[upper]
    numerator = float(left_pairs @ right_pairs)
    denominator = float(np.sqrt((left_pairs @ left_pairs) * (right_pairs @ right_pairs)))
    return float(numerator / denominator) if denominator > 1e-24 else float("nan")


def pairwise_structure_metrics(
    activations: np.ndarray,
    values: np.ndarray,
    *,
    property_ids: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    matrix = _as_matrix(activations, name="activations")
    value_array = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(matrix) != len(value_array) or len(value_array) < 3:
        raise ValueError("Activations and at least three values are required.")
    similarity = cosine_similarity_matrix(matrix)
    upper = np.triu_indices(len(value_array), k=1)
    similarities = similarity[upper]
    distances = np.abs(value_array[:, None] - value_array[None, :]).astype(np.float64)
    pair_distances = distances[upper]
    log_distance = np.log1p(pair_distances)
    standardized_log_distance = (log_distance - log_distance.mean()) / max(
        float(log_distance.std()), 1e-15
    )
    distance_kernel = np.exp(-0.5 * (distances / max(float(np.median(pair_distances)), 1.0)) ** 2)
    rows: list[dict[str, float | int | str]] = [
        {
            "structure": "numeric_distance",
            "pair_count": int(len(similarities)),
            "same_pair_count": 0,
            "different_pair_count": int(len(similarities)),
            "mean_similarity_same": float("nan"),
            "mean_similarity_different": float("nan"),
            "similarity_gap": float("nan"),
            "distance_adjusted_similarity_gap": float("nan"),
            "pairwise_pearson": float(np.corrcoef(similarities, pair_distances)[0, 1]),
            "kernel_alignment": off_diagonal_centered_kernel_alignment(
                similarity, distance_kernel
            ),
        }
    ]
    for property_id in property_ids:
        labels = property_labels(property_id, value_array)
        equality = labels[:, None] == labels[None, :]
        pair_equality = equality[upper].astype(np.float64)
        same = pair_equality == 1.0
        different = ~same
        if not np.any(same) or not np.any(different):
            continue
        design = np.column_stack(
            [np.ones(len(similarities)), standardized_log_distance, pair_equality]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, similarities, rcond=None)
        rows.append(
            {
                "structure": f"equal_{property_id}",
                "pair_count": int(len(similarities)),
                "same_pair_count": int(same.sum()),
                "different_pair_count": int(different.sum()),
                "mean_similarity_same": float(similarities[same].mean()),
                "mean_similarity_different": float(similarities[different].mean()),
                "similarity_gap": float(similarities[same].mean() - similarities[different].mean()),
                "distance_adjusted_similarity_gap": float(coefficients[2]),
                "pairwise_pearson": float(np.corrcoef(similarities, pair_equality)[0, 1]),
                "kernel_alignment": off_diagonal_centered_kernel_alignment(
                    similarity, equality.astype(np.float64)
                ),
            }
        )
    return similarity, rows


def _ridge_path(
    train_x: np.ndarray,
    train_y: np.ndarray,
    alphas: Sequence[float],
) -> list[tuple[float, np.ndarray]]:
    x = _as_matrix(train_x, name="train_x")
    y = _as_matrix(train_y, name="train_y")
    if len(x) != len(y):
        raise ValueError("Ridge inputs are not row-aligned.")
    n = max(len(x), 1)
    kernel = (x @ x.T) / n
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    projected = eigenvectors.T @ y
    rows: list[tuple[float, np.ndarray]] = []
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        if alpha <= 0.0:
            raise ValueError("Readout ridge alphas must be positive.")
        dual = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        rows.append((alpha, np.asarray((x.T @ dual) / n, dtype=np.float64)))
    return rows


def fresh_property_readout_metrics(
    *,
    train_activations: np.ndarray,
    train_values: np.ndarray,
    dev_activations: np.ndarray,
    dev_values: np.ndarray,
    test_activations: np.ndarray,
    test_values: np.ndarray,
    feature_encoder: PropertyFeatureEncoder,
    alphas: Sequence[float],
) -> list[dict[str, float | int | str]]:
    train_x = _as_matrix(train_activations, name="train_activations")
    dev_x = _as_matrix(dev_activations, name="dev_activations")
    test_x = _as_matrix(test_activations, name="test_activations")
    train_y = feature_encoder.transform(np.asarray(train_values, dtype=np.int64))
    dev_y = feature_encoder.transform(np.asarray(dev_values, dtype=np.int64))
    test_y = feature_encoder.transform(np.asarray(test_values, dtype=np.int64))
    candidates: dict[str, tuple[float, np.ndarray, float]] = {}
    for alpha, coefficients in _ridge_path(train_x, train_y, alphas):
        dev_prediction = dev_x @ coefficients
        for block in feature_encoder.blocks:
            truth = dev_y[:, block.start : block.stop]
            prediction = dev_prediction[:, block.start : block.stop]
            score = float(1.0 - np.sum((truth - prediction) ** 2) / max(np.sum(truth**2), 1e-24))
            previous = candidates.get(block.property_id)
            if previous is None or (score, -alpha) > (previous[2], -previous[0]):
                candidates[block.property_id] = (
                    alpha,
                    coefficients[:, block.start : block.stop].copy(),
                    score,
                )
    rows: list[dict[str, float | int | str]] = []
    for block in feature_encoder.blocks:
        alpha, coefficients, dev_fve = candidates[block.property_id]
        prediction = test_x @ coefficients
        truth_features = test_y[:, block.start : block.stop]
        test_fve = float(
            1.0 - np.sum((truth_features - prediction) ** 2) / max(np.sum(truth_features**2), 1e-24)
        )
        raw_prediction = prediction * block.block_scale + block.train_mean[None, :]
        row: dict[str, float | int | str] = {
            "property_id": block.property_id,
            "kind": block.kind,
            "selected_alpha": alpha,
            "dev_target_fve": dev_fve,
            "test_target_fve": test_fve,
        }
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
                raise ValueError(f"Categorical property {block.property_id} has no classes.")
            truth = property_labels(block.property_id, np.asarray(test_values, dtype=np.int64))
            train_truth = property_labels(
                block.property_id, np.asarray(train_values, dtype=np.int64)
            )
            decoded = block.classes[np.argmax(raw_prediction, axis=1)]
            recalls = [
                float(np.mean(decoded[truth == label] == label))
                for label in block.classes
                if np.any(truth == label)
            ]
            test_classes, test_counts = np.unique(truth, return_counts=True)
            _, train_counts = np.unique(train_truth, return_counts=True)
            row.update(
                {
                    "test_r2": float("nan"),
                    "test_mae": float("nan"),
                    "test_accuracy": float(np.mean(decoded == truth)),
                    "test_balanced_accuracy": (
                        float(np.mean(recalls)) if len(test_classes) >= 2 else float("nan")
                    ),
                    "train_observed_class_count": int(len(train_counts)),
                    "train_min_class_count": int(train_counts.min()),
                    "test_observed_class_count": int(len(test_counts)),
                    "test_min_class_count": int(test_counts.min()),
                }
            )
        rows.append(row)
    return rows


def cross_context_reliability(
    left_activations: np.ndarray,
    right_activations: np.ndarray,
) -> dict[str, float | int]:
    left = _as_matrix(left_activations, name="left_activations")
    right = _as_matrix(right_activations, name="right_activations")
    if left.shape != right.shape:
        raise ValueError(f"Cross-context matrices differ: {left.shape} versus {right.shape}.")
    left_kernel = left @ left.T
    right_kernel = right @ right.T
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    similarities = np.divide(
        left @ right.T,
        left_norm[:, None] * right_norm[None, :],
        out=np.zeros((len(left), len(right)), dtype=np.float64),
        where=(left_norm[:, None] * right_norm[None, :]) > 1e-15,
    )

    def retrieval(scores: np.ndarray) -> tuple[float, float]:
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.argmax(order == np.arange(len(scores))[:, None], axis=1) + 1
        return float(np.mean(ranks == 1)), float(np.mean(1.0 / ranks))

    forward_top1, forward_mrr = retrieval(similarities)
    reverse_top1, reverse_mrr = retrieval(similarities.T)
    return {
        "value_count": int(len(left)),
        "linear_cka": centered_kernel_alignment(left_kernel, right_kernel),
        "mean_paired_cosine": float(np.mean(np.diag(similarities))),
        "bidirectional_top1": float((forward_top1 + reverse_top1) / 2.0),
        "bidirectional_mrr": float((forward_mrr + reverse_mrr) / 2.0),
        "chance_top1": float(1.0 / len(left)),
    }
