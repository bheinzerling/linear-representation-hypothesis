from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.steering_sanity import orthonormal_row_basis


@dataclass
class AgopSteeringFamily:
    architecture: str
    basis_matrix: np.ndarray
    readout_weight: np.ndarray
    readout_bias: float
    support_values: np.ndarray
    metadata: dict[str, Any]

    def predict_values(self, representations: np.ndarray) -> np.ndarray:
        features = np.asarray(representations, dtype=np.float64) @ self.basis_matrix.T
        return features @ self.readout_weight + float(self.readout_bias)

    def intervention_delta(self, source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
        delta_y = np.asarray(target_values, dtype=np.float64).reshape(-1) - np.asarray(source_values, dtype=np.float64).reshape(-1)
        denom = float(np.dot(self.readout_weight, self.readout_weight))
        if denom <= 1e-12:
            raise ValueError(f"AGOP family {self.architecture!r} has near-zero readout weight.")
        coeffs = delta_y[:, None] * self.readout_weight[None, :] / denom
        return np.asarray(coeffs @ self.basis_matrix, dtype=np.float64)


def _median_bandwidth(metric_sq: np.ndarray) -> float:
    upper = metric_sq[np.triu_indices_from(metric_sq, k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0)]
    if finite.size == 0:
        return 1.0
    return float(np.sqrt(np.median(finite)))


def _pairwise_metric_sq(
    x: np.ndarray,
    *,
    components: np.ndarray,
    weights: np.ndarray,
    diagonal_scale: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.zeros((x_arr.shape[0], x_arr.shape[0]), dtype=np.float64)
    if float(diagonal_scale) > 0.0:
        sq = np.sum(x_arr * x_arr, axis=1)
        result += float(diagonal_scale) * np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x_arr @ x_arr.T), 0.0)
    if components.size and weights.size:
        projected = x_arr @ components.T
        weighted = projected * np.sqrt(np.maximum(weights, 0.0))[None, :]
        sq = np.sum(weighted * weighted, axis=1)
        result += np.maximum(sq[:, None] + sq[None, :] - 2.0 * (weighted @ weighted.T), 0.0)
    return np.maximum(result, 0.0)


def _apply_metric(
    rows: np.ndarray,
    *,
    components: np.ndarray,
    weights: np.ndarray,
    diagonal_scale: float,
) -> np.ndarray:
    rows_arr = np.asarray(rows, dtype=np.float64)
    result = float(diagonal_scale) * rows_arr
    if components.size and weights.size:
        result = result + ((rows_arr @ components.T) * weights[None, :]) @ components
    return np.asarray(result, dtype=np.float64)


def _kernel_ridge_alpha(kernel: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    lam = float(ridge_lambda)
    if lam < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda!r}.")
    return np.linalg.solve(k + lam * np.eye(k.shape[0], dtype=np.float64), y_arr)


def _top_agop_components(gradients: np.ndarray, max_components: int) -> tuple[np.ndarray, np.ndarray]:
    grad = np.asarray(gradients, dtype=np.float64)
    if grad.ndim != 2:
        raise ValueError("Expected 2D gradient matrix.")
    max_rank = min(int(max_components), grad.shape[0], grad.shape[1])
    if max_rank <= 0:
        raise ValueError("max_components must be positive.")
    _, singular_values, vh = np.linalg.svd(grad, full_matrices=False)
    eigenvalues = np.square(singular_values[:max_rank]) / max(1, grad.shape[0])
    basis = orthonormal_row_basis(vh[:max_rank])
    if basis.shape[0] > max_rank:
        basis = basis[:max_rank]
    return np.asarray(basis, dtype=np.float64), np.asarray(eigenvalues[: basis.shape[0]], dtype=np.float64)


def _ridge_readout(features: np.ndarray, values: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, float]:
    z = np.asarray(features, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    z_mean = z.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    zc = z - z_mean
    yc = y - y_mean
    gram = zc.T @ zc
    weight = np.linalg.solve(gram + float(ridge_lambda) * np.eye(gram.shape[0], dtype=np.float64), zc.T @ yc)
    bias = y_mean - float((z_mean @ weight).item())
    return np.asarray(weight, dtype=np.float64), float(bias)


def _projection_mass(vectors: np.ndarray, basis: np.ndarray) -> float:
    vec = np.asarray(vectors, dtype=np.float64)
    if vec.size == 0:
        return float("nan")
    try:
        other_basis = orthonormal_row_basis(vec)
    except ValueError:
        return float("nan")
    projected = (basis @ other_basis.T) @ other_basis
    denom = np.sum(basis * basis)
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(projected * projected) / denom)


def fit_agop_steering_family(
    *,
    architecture: str,
    train_x: np.ndarray,
    train_values: np.ndarray,
    dev_x: np.ndarray | None = None,
    dev_values: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    value_min: int = 0,
    value_max: int = 999,
    reference_bases: dict[str, np.ndarray] | None = None,
) -> tuple[AgopSteeringFamily, pd.DataFrame]:
    cfg = dict(config or {})
    x_raw = np.asarray(train_x, dtype=np.float64)
    y_raw = np.asarray(train_values, dtype=np.float64).reshape(-1)
    if x_raw.ndim != 2:
        raise ValueError("train_x must be 2D.")
    if x_raw.shape[0] != y_raw.shape[0]:
        raise ValueError("train_x and train_values must have matching rows.")

    p = int(cfg.get("p", 1))
    if p <= 0:
        raise ValueError(f"AGOP p must be positive, got {p!r}.")
    n_iters = max(1, int(cfg.get("n_iters", 1)))
    ridge_lambda = float(cfg.get("ridge_lambda", 1e-2))
    calibration_ridge = float(cfg.get("calibration_ridge_lambda", 1e-3))
    metric_rank = max(p, int(cfg.get("metric_rank", max(8, p))))
    diagonal_floor = float(cfg.get("diagonal_floor", 1e-6))
    max_components = max(p, metric_rank)

    x_mean = x_raw.mean(axis=0, keepdims=True)
    x = x_raw - x_mean
    y_mean = float(y_raw.mean())
    y_std = float(np.std(y_raw))
    if y_std <= 0.0:
        raise ValueError("train_values must vary.")
    y = (y_raw - y_mean) / y_std

    components = np.zeros((0, x.shape[1]), dtype=np.float64)
    weights = np.zeros((0,), dtype=np.float64)
    diagonal_scale = 1.0 / max(1, x.shape[1])
    bandwidth_cfg = cfg.get("bandwidth", "median")
    last_kernel = None
    last_alpha = None
    last_gradients = None
    bandwidth = 1.0

    for _ in range(n_iters):
        metric_sq = _pairwise_metric_sq(
            x,
            components=components,
            weights=weights,
            diagonal_scale=diagonal_scale,
        )
        if str(bandwidth_cfg) == "median":
            bandwidth = _median_bandwidth(metric_sq)
        else:
            bandwidth = float(bandwidth_cfg)
        bandwidth = max(float(bandwidth), 1e-8)
        kernel = np.exp(-metric_sq / (2.0 * bandwidth * bandwidth))
        alpha = _kernel_ridge_alpha(kernel, y, ridge_lambda=ridge_lambda)
        coeff = kernel * alpha[None, :]
        pre_metric_grad = coeff @ x - coeff.sum(axis=1, keepdims=True) * x
        gradients = _apply_metric(
            pre_metric_grad,
            components=components,
            weights=weights,
            diagonal_scale=diagonal_scale,
        ) / (bandwidth * bandwidth)
        basis_all, eigenvalues_all = _top_agop_components(gradients, max_components=max_components)
        trace = float(np.sum(eigenvalues_all))
        if trace <= 0.0:
            raise ValueError("AGOP trace is zero; cannot extract steering basis.")
        keep = min(metric_rank, basis_all.shape[0])
        components = basis_all[:keep]
        weights = eigenvalues_all[:keep] / trace
        diagonal_scale = diagonal_floor
        last_kernel = kernel
        last_alpha = alpha
        last_gradients = gradients

    if last_gradients is None:
        raise RuntimeError("AGOP fitting did not produce gradients.")
    basis_all, eigenvalues_all = _top_agop_components(last_gradients, max_components=max_components)
    basis = np.asarray(basis_all[:p], dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues_all[:p], dtype=np.float64)

    for idx in range(basis.shape[0]):
        projection = x_raw @ basis[idx]
        corr = np.corrcoef(projection, y_raw)[0, 1] if np.std(projection) > 0.0 else 0.0
        if np.isfinite(corr) and corr < 0.0:
            basis[idx] *= -1.0

    train_features = x_raw @ basis.T
    readout_weight, readout_bias = _ridge_readout(train_features, y_raw, ridge_lambda=calibration_ridge)
    train_pred = train_features @ readout_weight + readout_bias
    dev_corr = float("nan")
    dev_rmse = float("nan")
    if dev_x is not None and dev_values is not None:
        dev_features = np.asarray(dev_x, dtype=np.float64) @ basis.T
        dev_truth = np.asarray(dev_values, dtype=np.float64).reshape(-1)
        dev_pred = dev_features @ readout_weight + readout_bias
        dev_corr = float(np.corrcoef(dev_pred, dev_truth)[0, 1]) if np.std(dev_pred) > 0 and np.std(dev_truth) > 0 else float("nan")
        dev_rmse = float(np.sqrt(np.mean(np.square(dev_pred - dev_truth))))

    metadata: dict[str, Any] = {
        "p": int(p),
        "n_iters": int(n_iters),
        "ridge_lambda": float(ridge_lambda),
        "calibration_ridge_lambda": float(calibration_ridge),
        "metric_rank": int(metric_rank),
        "diagonal_floor": float(diagonal_floor),
        "bandwidth": float(bandwidth),
        "top_eigenvalues": eigenvalues.tolist(),
        "agop_mass_top_p": float(np.sum(eigenvalues) / max(np.sum(eigenvalues_all), 1e-30)),
        "train_corr": float(np.corrcoef(train_pred, y_raw)[0, 1]) if np.std(train_pred) > 0 else float("nan"),
        "train_rmse": float(np.sqrt(np.mean(np.square(train_pred - y_raw)))),
        "dev_corr": dev_corr,
        "dev_rmse": dev_rmse,
        "kernel_condition": float(np.linalg.cond(last_kernel + ridge_lambda * np.eye(last_kernel.shape[0]))) if last_kernel is not None else float("nan"),
        "alpha_norm": float(np.linalg.norm(last_alpha)) if last_alpha is not None else float("nan"),
    }
    for name, ref_basis in dict(reference_bases or {}).items():
        metadata[f"projection_mass_on_{name}"] = _projection_mass(ref_basis, basis)

    family = AgopSteeringFamily(
        architecture=str(architecture),
        basis_matrix=basis,
        readout_weight=readout_weight,
        readout_bias=readout_bias,
        support_values=np.arange(int(value_min), int(value_max) + 1, dtype=np.int64),
        metadata=metadata,
    )
    diagnostics = pd.DataFrame([{key: value if not isinstance(value, list) else repr(value) for key, value in metadata.items()}]).assign(
        architecture=str(architecture)
    )
    return family, diagnostics
