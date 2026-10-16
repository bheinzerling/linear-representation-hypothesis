from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ResidualCondition:
    condition_id: str
    evidence_role: str
    description: str
    residual_dose: float | None = None


BASE_CONDITIONS: tuple[ResidualCondition, ...] = (
    ResidualCondition(
        "original_state",
        "exact_replacement_control",
        "Replace the selected state with the captured live state (a zero-delta control).",
    ),
    ResidualCondition(
        "center_plus_projection",
        "residual_necessity",
        "Keep the live competitive-subspace projection and remove its orthogonal residual.",
    ),
    ResidualCondition(
        "center_plus_residual",
        "residual_sufficiency",
        "Keep the live orthogonal residual and remove the competitive-subspace projection.",
    ),
    ResidualCondition(
        "competitive_semantic_synthesis",
        "semantic_decoder_sufficiency_transfer",
        "Use the addition-trained competitive semantic decoder state for the source value.",
    ),
    ResidualCondition(
        "projection_plus_shuffled_residual",
        "residual_identity_control",
        "Keep each live projection but attach another matched task row's residual.",
    ),
    ResidualCondition(
        "projection_plus_random_subspace_residual",
        "residual_orientation_control",
        "Project the residual into a fixed random complement subspace and restore its row norm.",
    ),
)


def residual_dose_conditions(doses: Sequence[float]) -> tuple[ResidualCondition, ...]:
    result: list[ResidualCondition] = []
    for raw_dose in doses:
        dose = float(raw_dose)
        if not np.isfinite(dose) or dose < 0.0:
            raise ValueError(f"Residual doses must be finite and non-negative, got {raw_dose!r}.")
        if np.isclose(dose, 0.0) or np.isclose(dose, 1.0):
            # These are already represented by center_plus_projection and original_state.
            continue
        label = f"{dose:g}".replace(".", "p")
        result.append(
            ResidualCondition(
                f"residual_dose_{label}",
                "residual_necessity_dose_response",
                f"Keep the live projection and scale the live residual by {dose:g}.",
                residual_dose=dose,
            )
        )
    return tuple(result)


def all_conditions(doses: Sequence[float]) -> tuple[ResidualCondition, ...]:
    conditions = (*BASE_CONDITIONS, *residual_dose_conditions(doses))
    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate residual condition ids: {ids}")
    return conditions


def validate_orthonormal_basis(basis: np.ndarray, *, atol: float = 1e-5) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Expected a nonempty [hidden_dim, rank] basis, got {matrix.shape}.")
    gram_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(matrix.shape[1]))))
    if gram_error > float(atol):
        raise ValueError(f"Basis is not orthonormal (max Gram error {gram_error:.3e}).")
    return matrix


def decompose_live_states(
    live_states: np.ndarray,
    *,
    center: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    live = np.asarray(live_states, dtype=np.float64)
    center_row = np.asarray(center, dtype=np.float64).reshape(-1)
    q = validate_orthonormal_basis(basis)
    if live.ndim != 2 or live.shape[1] != len(center_row) or q.shape[0] != live.shape[1]:
        raise ValueError(
            f"Incompatible live/center/basis shapes: {live.shape}, {center_row.shape}, {q.shape}."
        )
    centered = live - center_row[None, :]
    projection = (centered @ q) @ q.T
    residual = centered - projection
    return np.asarray(projection, dtype=np.float64), np.asarray(residual, dtype=np.float64)


def deterministic_derangement(size: int, *, seed: int) -> np.ndarray:
    n = int(size)
    if n < 2:
        raise ValueError("A shuffled-residual condition requires at least two examples.")
    order = np.random.default_rng(int(seed)).permutation(n)
    result = np.empty(n, dtype=np.int64)
    result[order] = np.roll(order, 1)
    if np.any(result == np.arange(n, dtype=np.int64)):
        raise RuntimeError("Internal error: residual shuffle contains a fixed point.")
    return result


def random_complement_basis(
    *,
    hidden_dim: int,
    excluded_basis: np.ndarray,
    rank: int,
    seed: int,
) -> np.ndarray:
    q_excluded = validate_orthonormal_basis(excluded_basis)
    dimension = int(hidden_dim)
    requested_rank = int(rank)
    available = dimension - q_excluded.shape[1]
    if q_excluded.shape[0] != dimension:
        raise ValueError(f"Excluded basis hidden dimension {q_excluded.shape[0]} != {dimension}.")
    if requested_rank <= 0 or requested_rank > available:
        raise ValueError(f"Random complement rank must lie in 1..{available}, got {requested_rank}.")
    rng = np.random.default_rng(int(seed))
    candidates = rng.standard_normal((dimension, requested_rank))
    for _ in range(2):
        candidates -= q_excluded @ (q_excluded.T @ candidates)
    random_basis, triangular = np.linalg.qr(candidates, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) < 1e-10:
        raise ValueError("Random complement draw was numerically rank deficient.")
    random_basis = np.asarray(random_basis[:, :requested_rank], dtype=np.float64)
    overlap = float(np.max(np.abs(q_excluded.T @ random_basis)))
    gram_error = float(np.max(np.abs(random_basis.T @ random_basis - np.eye(requested_rank))))
    if overlap > 1e-8 or gram_error > 1e-8:
        raise ValueError(
            f"Random complement validation failed: overlap={overlap:.3e}, Gram={gram_error:.3e}."
        )
    return random_basis


def random_subspace_residual(
    residual: np.ndarray,
    *,
    random_basis: np.ndarray,
    norm_epsilon: float = 1e-12,
) -> np.ndarray:
    rows = np.asarray(residual, dtype=np.float64)
    q_random = validate_orthonormal_basis(random_basis)
    if rows.ndim != 2 or rows.shape[1] != q_random.shape[0]:
        raise ValueError(f"Residual/random-basis shape mismatch: {rows.shape}, {q_random.shape}.")
    projected = (rows @ q_random) @ q_random.T
    source_norm = np.linalg.norm(rows, axis=1)
    projected_norm = np.linalg.norm(projected, axis=1)
    valid = (source_norm > float(norm_epsilon)) & (projected_norm > float(norm_epsilon))
    matched = np.zeros_like(projected)
    matched[valid] = projected[valid] * (source_norm[valid] / projected_norm[valid])[:, None]
    return matched


def condition_target_states(
    *,
    condition: ResidualCondition,
    live_states: np.ndarray,
    center: np.ndarray,
    projection: np.ndarray,
    residual: np.ndarray,
    semantic_synthesis: np.ndarray,
    shuffle_indices: np.ndarray,
    random_basis: np.ndarray,
) -> np.ndarray:
    live = np.asarray(live_states, dtype=np.float64)
    center_row = np.asarray(center, dtype=np.float64).reshape(1, -1)
    projected = np.asarray(projection, dtype=np.float64)
    residual_rows = np.asarray(residual, dtype=np.float64)
    semantic = np.asarray(semantic_synthesis, dtype=np.float64)
    expected_shape = live.shape
    if any(array.shape != expected_shape for array in (projected, residual_rows, semantic)):
        raise ValueError(
            f"State component mismatch: live={live.shape}, projection={projected.shape}, "
            f"residual={residual_rows.shape}, semantic={semantic.shape}."
        )
    if center_row.shape[1] != live.shape[1]:
        raise ValueError(f"Center dimension {center_row.shape[1]} != live dimension {live.shape[1]}.")

    condition_id = condition.condition_id
    if condition_id == "original_state":
        return live.copy()
    if condition_id == "center_plus_projection":
        return center_row + projected
    if condition_id == "center_plus_residual":
        return center_row + residual_rows
    if condition_id == "competitive_semantic_synthesis":
        return semantic.copy()
    if condition_id == "projection_plus_shuffled_residual":
        indices = np.asarray(shuffle_indices, dtype=np.int64).reshape(-1)
        if len(indices) != len(live) or sorted(indices.tolist()) != list(range(len(live))):
            raise ValueError("shuffle_indices must be a permutation with one entry per live state.")
        return center_row + projected + residual_rows[indices]
    if condition_id == "projection_plus_random_subspace_residual":
        return center_row + projected + random_subspace_residual(
            residual_rows, random_basis=random_basis
        )
    if condition.residual_dose is not None:
        return center_row + projected + float(condition.residual_dose) * residual_rows
    raise ValueError(f"Unknown residual condition: {condition_id!r}")


def activation_condition_metrics(
    *,
    live_states: np.ndarray,
    target_states: np.ndarray,
    projection: np.ndarray,
    residual: np.ndarray,
) -> dict[str, np.ndarray]:
    live = np.asarray(live_states, dtype=np.float64)
    target = np.asarray(target_states, dtype=np.float64)
    projected = np.asarray(projection, dtype=np.float64)
    residual_rows = np.asarray(residual, dtype=np.float64)
    if any(array.shape != live.shape for array in (target, projected, residual_rows)):
        raise ValueError("Activation metric inputs must have the same shape.")
    return {
        "live_state_l2": np.linalg.norm(live, axis=1),
        "target_state_l2": np.linalg.norm(target, axis=1),
        "replacement_delta_l2": np.linalg.norm(target - live, axis=1),
        "live_projection_l2": np.linalg.norm(projected, axis=1),
        "live_residual_l2": np.linalg.norm(residual_rows, axis=1),
    }
