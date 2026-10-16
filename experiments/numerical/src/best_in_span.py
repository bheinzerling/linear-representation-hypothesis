from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ConstraintDiagnostics:
    l2_norm: np.ndarray
    fisher_quadratic: np.ndarray
    l2_utilization: np.ndarray
    fisher_utilization: np.ndarray
    feasible: np.ndarray


def orthonormalize_columns(matrix: np.ndarray, *, rtol: float = 1e-9) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {values.shape}.")
    if values.shape[1] == 0:
        return np.empty_like(values)
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.empty((values.shape[0], 0), dtype=np.float64)
    rank = int(np.sum(singular > float(rtol) * float(singular[0])))
    return np.asarray(left[:, :rank], dtype=np.float64)


def project_rows_to_basis(rows: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rows, dtype=np.float64)
    columns = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or columns.ndim != 2:
        raise ValueError("rows and basis must both be matrices.")
    if values.shape[1] != columns.shape[0]:
        raise ValueError(
            f"Dimension mismatch: rows have width {values.shape[1]}, basis has height {columns.shape[0]}."
        )
    coefficients = values @ columns
    return coefficients, coefficients @ columns.T


def projected_fisher_from_logit_derivatives(
    logit_derivatives: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Construct BᵀFB from derivatives of tracked logits along basis columns.

    ``logit_derivatives`` has shape ``(rows, tracked_tokens, rank)`` and
    ``probabilities`` has shape ``(rows, tracked_tokens)``. The returned local
    categorical Fisher matrices have shape ``(rows, rank, rank)``.
    """

    derivatives = np.asarray(logit_derivatives, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if derivatives.ndim != 3 or probs.ndim != 2:
        raise ValueError("Expected derivatives (rows,tokens,rank) and probabilities (rows,tokens).")
    if derivatives.shape[:2] != probs.shape:
        raise ValueError(
            f"Derivative/probability mismatch: {derivatives.shape[:2]} versus {probs.shape}."
        )
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each probability row must have positive mass.")
    normalized = probs / row_sums
    means = np.einsum("nt,ntk->nk", normalized, derivatives, optimize=True)
    centered = derivatives - means[:, None, :]
    fisher = np.einsum("nt,ntk,ntl->nkl", normalized, centered, centered, optimize=True)
    return 0.5 * (fisher + np.swapaxes(fisher, 1, 2))


def directional_fisher_quadratic(
    directional_logit_derivatives: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    derivatives = np.asarray(directional_logit_derivatives, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if derivatives.ndim != 2 or probs.ndim != 2 or derivatives.shape != probs.shape:
        raise ValueError("Directional derivatives and probabilities must have the same 2D shape.")
    normalized = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)
    mean = np.sum(normalized * derivatives, axis=1)
    centered = derivatives - mean[:, None]
    return np.sum(normalized * centered * centered, axis=1)


def coefficient_fisher_quadratic(coefficients: np.ndarray, fisher: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    metric = np.asarray(fisher, dtype=np.float64)
    if values.ndim != 2 or metric.ndim != 3:
        raise ValueError("Expected coefficients (rows,rank) and fisher (rows,rank,rank).")
    if metric.shape != (values.shape[0], values.shape[1], values.shape[1]):
        raise ValueError(f"Coefficient/Fisher mismatch: {values.shape} and {metric.shape}.")
    result = np.einsum("nk,nkl,nl->n", values, metric, values, optimize=True)
    return np.maximum(result, 0.0)


def joint_constraint_diagnostics(
    coefficients: np.ndarray,
    fisher: np.ndarray,
    *,
    l2_budget: np.ndarray,
    fisher_budget: np.ndarray,
    relative_tolerance: float = 1e-5,
) -> ConstraintDiagnostics:
    values = np.asarray(coefficients, dtype=np.float64)
    l2_limit = np.asarray(l2_budget, dtype=np.float64).reshape(-1)
    fisher_limit = np.asarray(fisher_budget, dtype=np.float64).reshape(-1)
    if len(values) != len(l2_limit) or len(values) != len(fisher_limit):
        raise ValueError("Budget rows must align with coefficient rows.")
    l2 = np.linalg.norm(values, axis=1)
    quadratic = coefficient_fisher_quadratic(values, fisher)
    l2_util = np.divide(
        l2,
        l2_limit,
        out=np.where(l2 <= 1e-12, 0.0, np.inf),
        where=l2_limit > 1e-12,
    )
    fisher_util = np.divide(
        quadratic,
        fisher_limit,
        out=np.where(quadratic <= 1e-18, 0.0, np.inf),
        where=fisher_limit > 1e-18,
    )
    feasible = (l2_util <= 1.0 + relative_tolerance) & (
        fisher_util <= 1.0 + relative_tolerance
    )
    return ConstraintDiagnostics(
        l2_norm=l2,
        fisher_quadratic=quadratic,
        l2_utilization=l2_util,
        fisher_utilization=fisher_util,
        feasible=feasible,
    )


def radial_project_joint_constraints_numpy(
    coefficients: np.ndarray,
    fisher: np.ndarray,
    *,
    l2_budget: np.ndarray,
    fisher_budget: np.ndarray,
    eps: float = 1e-18,
) -> np.ndarray:
    """Radially contract rows until both homogeneous constraints are satisfied."""

    values = np.asarray(coefficients, dtype=np.float64)
    l2_limit = np.maximum(np.asarray(l2_budget, dtype=np.float64).reshape(-1), 0.0)
    fisher_limit = np.maximum(np.asarray(fisher_budget, dtype=np.float64).reshape(-1), 0.0)
    l2 = np.linalg.norm(values, axis=1)
    quadratic = coefficient_fisher_quadratic(values, fisher)
    l2_scale = np.divide(l2_limit, l2, out=np.ones_like(l2), where=l2 > eps)
    fisher_scale = np.sqrt(
        np.divide(fisher_limit, quadratic, out=np.ones_like(quadratic), where=quadratic > eps)
    )
    scale = np.minimum(1.0, np.minimum(l2_scale, fisher_scale))
    scale = np.where((l2_limit <= eps) & (l2 > eps), 0.0, scale)
    scale = np.where((fisher_limit <= eps) & (quadratic > eps), 0.0, scale)
    return values * scale[:, None]


def radial_project_joint_constraints_torch(
    coefficients: torch.Tensor,
    fisher: torch.Tensor,
    *,
    l2_budget: torch.Tensor,
    fisher_budget: torch.Tensor,
    eps: float = 1e-18,
) -> torch.Tensor:
    if coefficients.ndim != 2 or fisher.ndim != 3:
        raise ValueError("Expected coefficients (rows,rank) and fisher (rows,rank,rank).")
    l2_limit = l2_budget.reshape(-1).to(coefficients).clamp_min(0.0)
    fisher_limit = fisher_budget.reshape(-1).to(coefficients).clamp_min(0.0)
    l2 = torch.linalg.vector_norm(coefficients, dim=1)
    quadratic = torch.einsum("nk,nkl,nl->n", coefficients, fisher.to(coefficients), coefficients)
    quadratic = quadratic.clamp_min(0.0)
    one = torch.ones_like(l2)
    l2_scale = torch.where(l2 > eps, l2_limit / l2.clamp_min(eps), one)
    fisher_scale = torch.where(
        quadratic > eps,
        torch.sqrt(fisher_limit / quadratic.clamp_min(eps)),
        one,
    )
    scale = torch.minimum(one, torch.minimum(l2_scale, fisher_scale))
    scale = torch.where((l2_limit <= eps) & (l2 > eps), torch.zeros_like(scale), scale)
    scale = torch.where(
        (fisher_limit <= eps) & (quadratic > eps), torch.zeros_like(scale), scale
    )
    return coefficients * scale[:, None]


def maximum_feasible_ray_scale(
    direction_coefficients: np.ndarray,
    fisher: np.ndarray,
    *,
    l2_budget: np.ndarray,
    fisher_budget: np.ndarray,
    eps: float = 1e-18,
) -> np.ndarray:
    direction = np.asarray(direction_coefficients, dtype=np.float64)
    l2_limit = np.maximum(np.asarray(l2_budget, dtype=np.float64).reshape(-1), 0.0)
    fisher_limit = np.maximum(np.asarray(fisher_budget, dtype=np.float64).reshape(-1), 0.0)
    direction_l2 = np.linalg.norm(direction, axis=1)
    direction_q = coefficient_fisher_quadratic(direction, fisher)
    l2_scale = np.divide(
        l2_limit,
        direction_l2,
        out=np.full_like(l2_limit, np.inf),
        where=direction_l2 > eps,
    )
    fisher_scale = np.sqrt(
        np.divide(
            fisher_limit,
            direction_q,
            out=np.full_like(fisher_limit, np.inf),
            where=direction_q > eps,
        )
    )
    result = np.minimum(l2_scale, fisher_scale)
    result = np.where((direction_l2 <= eps) & (direction_q <= eps), 0.0, result)
    return np.maximum(result, 0.0)
