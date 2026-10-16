from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def orthonormal_column_basis(matrix: np.ndarray, *, rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D matrix, got shape {array.shape!r}.")
    if array.shape[1] == 0:
        return np.zeros((array.shape[0], 0), dtype=np.float64)

    u, singular_values, _ = np.linalg.svd(array, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((array.shape[0], 0), dtype=np.float64)
    threshold = max(float(atol), float(rtol) * float(singular_values[0]))
    rank = int(np.sum(singular_values > threshold))
    return np.asarray(u[:, :rank], dtype=np.float64)


def coefficient_column_basis(coefficients: np.ndarray, *, rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    return orthonormal_column_basis(coefficients, rtol=rtol, atol=atol)


def orthogonalize_against(candidate_basis: np.ndarray, existing_basis: np.ndarray, *, rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    candidate = np.asarray(candidate_basis, dtype=np.float64)
    existing = np.asarray(existing_basis, dtype=np.float64)
    if candidate.ndim != 2 or existing.ndim != 2:
        raise ValueError("candidate_basis and existing_basis must both be 2D arrays.")
    if candidate.shape[0] != existing.shape[0]:
        raise ValueError(f"Basis row mismatch: candidate {candidate.shape}, existing {existing.shape}.")
    if candidate.shape[1] == 0:
        return np.zeros((candidate.shape[0], 0), dtype=np.float64)
    residual = candidate
    if existing.shape[1] > 0:
        residual = residual - existing @ (existing.T @ residual)
    return orthonormal_column_basis(residual, rtol=rtol, atol=atol)


def append_orthonormal_basis(existing_basis: np.ndarray, new_basis: np.ndarray, *, rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    existing = np.asarray(existing_basis, dtype=np.float64)
    new = orthogonalize_against(new_basis, existing, rtol=rtol, atol=atol)
    if existing.shape[1] == 0:
        return new
    if new.shape[1] == 0:
        return existing
    return np.concatenate([existing, new], axis=1)


def project_out_basis(representations: np.ndarray, basis: np.ndarray, *, center: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(representations, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    if x.ndim != 2 or q.ndim != 2:
        raise ValueError("representations and basis must both be 2D arrays.")
    if x.shape[1] != q.shape[0]:
        raise ValueError(f"Representation/basis dimension mismatch: {x.shape} vs {q.shape}.")
    if q.shape[1] == 0:
        return np.asarray(x, dtype=np.float64).copy()
    if center is None:
        centered = x
        offset = 0.0
    else:
        offset = np.asarray(center, dtype=np.float64).reshape(1, -1)
        if offset.shape[1] != x.shape[1]:
            raise ValueError(f"Center dimension mismatch: {offset.shape} vs {x.shape}.")
        centered = x - offset
    projected = centered - (centered @ q) @ q.T
    return np.asarray(projected + offset, dtype=np.float64)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    projection_step: int
    partition: str
    added_rank: int
    cumulative_rank: int
    variance_removed_fraction: float
    remaining_basis_max_abs: float
    remaining_basis_rms: float
    mean_shift_norm: float

    def to_jsonable(self) -> dict[str, float | int | str]:
        return {
            "projection_step": int(self.projection_step),
            "partition": str(self.partition),
            "added_rank": int(self.added_rank),
            "cumulative_rank": int(self.cumulative_rank),
            "variance_removed_fraction": float(self.variance_removed_fraction),
            "remaining_basis_max_abs": float(self.remaining_basis_max_abs),
            "remaining_basis_rms": float(self.remaining_basis_rms),
            "mean_shift_norm": float(self.mean_shift_norm),
        }


def projection_diagnostics(
    *,
    raw_representations: np.ndarray,
    projected_representations: np.ndarray,
    basis: np.ndarray,
    center: np.ndarray | None,
    projection_step: int,
    partition: str,
    added_rank: int,
) -> ProjectionDiagnostics:
    raw = np.asarray(raw_representations, dtype=np.float64)
    projected = np.asarray(projected_representations, dtype=np.float64)
    q = np.asarray(basis, dtype=np.float64)
    if raw.shape != projected.shape:
        raise ValueError(f"Raw/projected shape mismatch: {raw.shape} vs {projected.shape}.")
    if center is None:
        raw_centered = raw
        projected_centered = projected
        center_vec = np.zeros(raw.shape[1], dtype=np.float64)
    else:
        center_vec = np.asarray(center, dtype=np.float64).reshape(-1)
        raw_centered = raw - center_vec[None, :]
        projected_centered = projected - center_vec[None, :]

    removed = raw_centered - projected_centered
    denom = float(np.sum(raw_centered**2))
    variance_removed_fraction = float(np.sum(removed**2) / denom) if denom > 0.0 else float("nan")
    if q.shape[1] > 0:
        remaining_coords = projected_centered @ q
        remaining_basis_max_abs = float(np.max(np.abs(remaining_coords)))
        remaining_basis_rms = float(np.sqrt(np.mean(remaining_coords**2)))
    else:
        remaining_basis_max_abs = 0.0
        remaining_basis_rms = 0.0
    mean_shift_norm = float(np.linalg.norm(projected.mean(axis=0) - center_vec))
    return ProjectionDiagnostics(
        projection_step=int(projection_step),
        partition=str(partition),
        added_rank=int(added_rank),
        cumulative_rank=int(q.shape[1]),
        variance_removed_fraction=variance_removed_fraction,
        remaining_basis_max_abs=remaining_basis_max_abs,
        remaining_basis_rms=remaining_basis_rms,
        mean_shift_norm=mean_shift_norm,
    )
