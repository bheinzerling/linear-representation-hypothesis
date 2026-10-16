from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from src.owner_block_steering import PRIMARY_PROPERTY_ORDER
from src.property_subspace_synthesis import (
    PropertyFeatureEncoder,
    fit_property_feature_encoder,
)


PROPERTY_IDS = ("magnitude", *PRIMARY_PROPERTY_ORDER)


@dataclass(frozen=True)
class GroupSplit:
    row_partitions: np.ndarray
    group_partitions: Mapping[int, str]
    train_groups: tuple[int, ...]
    validation_groups: tuple[int, ...]
    test_groups: tuple[int, ...]


@dataclass(frozen=True)
class JointFeatureTransform:
    encoder: PropertyFeatureEncoder
    coordinate_mean: np.ndarray
    coordinate_scale: np.ndarray
    property_ids: tuple[str, ...]

    def transform(
        self,
        source_coordinates: np.ndarray,
        source_values: np.ndarray,
        target_values: np.ndarray,
    ) -> np.ndarray:
        coordinates = np.asarray(source_coordinates, dtype=np.float64)
        source = np.asarray(source_values, dtype=np.int64).reshape(-1)
        target = np.asarray(target_values, dtype=np.int64).reshape(-1)
        if coordinates.ndim != 2 or len(coordinates) != len(source):
            raise ValueError("Source coordinates and values are not row-aligned.")
        if len(target) != len(source):
            raise ValueError("Source and target values are not row-aligned.")
        coordinate_features = (
            (coordinates - self.coordinate_mean[None, :])
            / self.coordinate_scale[None, :]
            / np.sqrt(coordinates.shape[1])
        )
        property_scale = np.sqrt(len(self.property_ids))
        source_features = self.encoder.transform(source) / property_scale
        target_features = self.encoder.transform(target) / property_scale
        result = np.concatenate(
            [coordinate_features, source_features, target_features],
            axis=1,
        )
        if not np.isfinite(result).all():
            raise ValueError("Joint features contain non-finite values.")
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class TargetTransform:
    center: np.ndarray
    scale: float

    def transform(self, target: np.ndarray) -> np.ndarray:
        return (
            np.asarray(target, dtype=np.float64) - self.center[None, :]
        ) / self.scale

    def inverse_transform(self, target: np.ndarray) -> np.ndarray:
        return (
            np.asarray(target, dtype=np.float64) * self.scale
            + self.center[None, :]
        )


@dataclass(frozen=True)
class KernelSelection:
    kernel: str
    degree: int | None
    gamma: float
    gamma_multiplier: float
    alpha: float
    validation_mse: float


@dataclass(frozen=True)
class MLPSelection:
    width: int
    weight_decay: float
    best_epoch: int
    validation_mse: float


def deterministic_group_split(
    groups: Sequence[int] | np.ndarray,
    *,
    seed: int,
    train_groups: int,
    validation_groups: int,
    test_groups: int,
) -> GroupSplit:
    group_array = np.asarray(groups, dtype=np.int64).reshape(-1)
    unique = np.unique(group_array)
    expected = int(train_groups) + int(validation_groups) + int(test_groups)
    if len(unique) != expected:
        raise ValueError(f"Expected {expected} groups, found {len(unique)}.")
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(len(unique))]
    train = tuple(int(value) for value in shuffled[: int(train_groups)])
    validation = tuple(
        int(value)
        for value in shuffled[
            int(train_groups) : int(train_groups) + int(validation_groups)
        ]
    )
    test = tuple(int(value) for value in shuffled[-int(test_groups) :])
    mapping = {
        **{value: "train" for value in train},
        **{value: "validation" for value in validation},
        **{value: "test" for value in test},
    }
    row_partitions = np.asarray(
        [mapping[int(value)] for value in group_array],
        dtype="U10",
    )
    return GroupSplit(
        row_partitions=row_partitions,
        group_partitions=mapping,
        train_groups=train,
        validation_groups=validation,
        test_groups=test,
    )


def fit_joint_feature_transform(
    *,
    source_coordinates: np.ndarray,
    source_values: np.ndarray,
    target_values: np.ndarray,
    train_mask: np.ndarray,
    property_ids: Sequence[str] = PROPERTY_IDS,
) -> JointFeatureTransform:
    coordinates = np.asarray(source_coordinates, dtype=np.float64)
    source = np.asarray(source_values, dtype=np.int64).reshape(-1)
    target = np.asarray(target_values, dtype=np.int64).reshape(-1)
    selected = np.asarray(train_mask, dtype=bool).reshape(-1)
    if coordinates.ndim != 2 or len(coordinates) != len(selected):
        raise ValueError("Training mask and source coordinates are not row-aligned.")
    if not selected.any():
        raise ValueError("Training partition is empty.")
    training_values = np.unique(
        np.concatenate([source[selected], target[selected]])
    )
    encoder = fit_property_feature_encoder(
        train_values=training_values,
        property_ids=tuple(property_ids),
        continuous_properties=("magnitude",),
    )
    coordinate_mean = coordinates[selected].mean(axis=0)
    coordinate_scale = coordinates[selected].std(axis=0)
    coordinate_scale = np.where(coordinate_scale > 1e-10, coordinate_scale, 1.0)
    transform = JointFeatureTransform(
        encoder=encoder,
        coordinate_mean=np.asarray(coordinate_mean, dtype=np.float64),
        coordinate_scale=np.asarray(coordinate_scale, dtype=np.float64),
        property_ids=tuple(str(value) for value in property_ids),
    )
    # Transforming all rows is also the registered unseen-label audit.
    transform.transform(coordinates, source, target)
    return transform


def fit_target_transform(target: np.ndarray, train_mask: np.ndarray) -> TargetTransform:
    values = np.asarray(target, dtype=np.float64)
    selected = np.asarray(train_mask, dtype=bool).reshape(-1)
    if values.ndim != 2 or len(values) != len(selected) or not selected.any():
        raise ValueError("Target and training mask are invalid.")
    center = values[selected].mean(axis=0)
    centered = values[selected] - center[None, :]
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Target global RMS scale is degenerate.")
    return TargetTransform(center=np.asarray(center, dtype=np.float64), scale=scale)


def fit_ridge_candidates(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    trace: list[dict[str, float]] = []
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(train_x, train_y)
        prediction = np.asarray(model.predict(validation_x), dtype=np.float64)
        mse = float(np.mean(np.square(prediction - validation_y)))
        trace.append({"alpha": float(alpha), "validation_mse": mse})
    best = min(trace, key=lambda row: (row["validation_mse"], row["alpha"]))
    return float(best["alpha"]), trace


def fit_ridge_predict(
    *,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    predict_x: np.ndarray,
    alpha: float,
) -> tuple[Ridge, np.ndarray]:
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(fit_x, fit_y)
    return model, np.asarray(model.predict(predict_x), dtype=np.float64)


def polynomial_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    degree: int,
    gamma: float,
) -> np.ndarray:
    return np.asarray(
        np.power(
            1.0
            + float(gamma)
            * (
                np.asarray(left, dtype=np.float64)
                @ np.asarray(right, dtype=np.float64).T
            ),
            int(degree),
        ),
        dtype=np.float64,
    )


def rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    squared = (
        np.sum(x**2, axis=1)[:, None]
        + np.sum(y**2, axis=1)[None, :]
        - 2.0 * (x @ y.T)
    )
    np.maximum(squared, 0.0, out=squared)
    return np.asarray(np.exp(-float(gamma) * squared), dtype=np.float64)


def kernel_gamma_base(train_x: np.ndarray, kernel: str) -> float:
    values = np.asarray(train_x, dtype=np.float64)
    if kernel == "polynomial":
        mean_norm = float(np.mean(np.sum(values**2, axis=1)))
        return 1.0 / max(mean_norm, 1e-12)
    if kernel == "rbf":
        squared = (
            np.sum(values**2, axis=1)[:, None]
            + np.sum(values**2, axis=1)[None, :]
            - 2.0 * (values @ values.T)
        )
        positive = squared[np.triu_indices(len(values), k=1)]
        positive = positive[positive > 1e-12]
        if positive.size == 0:
            raise ValueError("RBF median-distance heuristic is degenerate.")
        return 1.0 / float(np.median(positive))
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _kernel_function(
    *,
    kernel: str,
    degree: int | None,
    gamma: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if kernel == "polynomial":
        if degree is None:
            raise ValueError("Polynomial kernel requires a degree.")
        return lambda left, right: polynomial_kernel(
            left,
            right,
            degree=int(degree),
            gamma=float(gamma),
        )
    if kernel == "rbf":
        return lambda left, right: rbf_kernel(
            left,
            right,
            gamma=float(gamma),
        )
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _kernel_eigendecomposition(kernel_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(kernel_matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues, eigenvectors


def select_kernel_ridge(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    kernel: str,
    degree: int | None,
    gamma_multipliers: Sequence[float],
    alphas: Sequence[float],
) -> tuple[KernelSelection, list[dict[str, float | int | str | None]]]:
    base = kernel_gamma_base(train_x, kernel)
    trace: list[dict[str, float | int | str | None]] = []
    for multiplier in gamma_multipliers:
        gamma = float(base * float(multiplier))
        function = _kernel_function(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
        )
        train_kernel = function(train_x, train_x)
        eigenvalues, eigenvectors = _kernel_eigendecomposition(train_kernel)
        projected_target = eigenvectors.T @ np.asarray(train_y, dtype=np.float64)
        validation_projection = function(validation_x, train_x) @ eigenvectors
        for alpha in alphas:
            dual = projected_target / (
                eigenvalues[:, None] + float(alpha)
            )
            prediction = validation_projection @ dual
            mse = float(np.mean(np.square(prediction - validation_y)))
            trace.append(
                {
                    "kernel": kernel,
                    "degree": degree,
                    "gamma": gamma,
                    "gamma_multiplier": float(multiplier),
                    "alpha": float(alpha),
                    "validation_mse": mse,
                }
            )
    best = min(
        trace,
        key=lambda row: (
            float(row["validation_mse"]),
            abs(np.log(float(row["gamma_multiplier"]))),
            float(row["alpha"]),
        ),
    )
    return (
        KernelSelection(
            kernel=str(kernel),
            degree=None if degree is None else int(degree),
            gamma=float(best["gamma"]),
            gamma_multiplier=float(best["gamma_multiplier"]),
            alpha=float(best["alpha"]),
            validation_mse=float(best["validation_mse"]),
        ),
        trace,
    )


def fit_kernel_ridge_predict(
    *,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    predict_x: np.ndarray,
    selection: KernelSelection,
) -> np.ndarray:
    function = _kernel_function(
        kernel=selection.kernel,
        degree=selection.degree,
        gamma=selection.gamma,
    )
    train_kernel = function(fit_x, fit_x)
    eigenvalues, eigenvectors = _kernel_eigendecomposition(train_kernel)
    projected_target = eigenvectors.T @ np.asarray(fit_y, dtype=np.float64)
    dual = projected_target / (eigenvalues[:, None] + selection.alpha)
    return np.asarray(
        (function(predict_x, fit_x) @ eigenvectors) @ dual,
        dtype=np.float64,
    )


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(output_dim)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def select_residual_mlp(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_base_prediction: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_base_prediction: np.ndarray,
    widths: Sequence[int],
    weight_decays: Sequence[float],
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[MLPSelection, list[dict[str, float | int]]]:
    x_train = _tensor(train_x)
    residual_train = _tensor(train_y - train_base_prediction)
    x_validation = _tensor(validation_x)
    validation_truth = _tensor(validation_y)
    validation_base = _tensor(validation_base_prediction)
    trace: list[dict[str, float | int]] = []
    candidates: list[
        tuple[float, int, float, int, dict[str, torch.Tensor]]
    ] = []
    for width in widths:
        for weight_decay in weight_decays:
            torch.manual_seed(int(seed))
            model = ResidualMLP(
                input_dim=x_train.shape[1],
                output_dim=residual_train.shape[1],
                width=int(width),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
            )
            best_loss = float("inf")
            best_epoch = 0
            best_state = deepcopy(model.state_dict())
            stale = 0
            for epoch in range(1, int(max_epochs) + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_train)
                loss = torch.mean(torch.square(prediction - residual_train))
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_prediction = validation_base + model(x_validation)
                    validation_loss = float(
                        torch.mean(
                            torch.square(
                                validation_prediction - validation_truth
                            )
                        ).item()
                    )
                if validation_loss < best_loss - 1e-10:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= int(patience):
                    break
            trace.append(
                {
                    "width": int(width),
                    "weight_decay": float(weight_decay),
                    "best_epoch": int(best_epoch),
                    "epochs_run": int(epoch),
                    "validation_mse": float(best_loss),
                }
            )
            candidates.append(
                (
                    float(best_loss),
                    int(width),
                    float(weight_decay),
                    int(best_epoch),
                    best_state,
                )
            )
    best_loss, width, weight_decay, best_epoch, _ = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return (
        MLPSelection(
            width=width,
            weight_decay=weight_decay,
            best_epoch=best_epoch,
            validation_mse=best_loss,
        ),
        trace,
    )


def fit_residual_mlp_predict(
    *,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_base_prediction: np.ndarray,
    predict_x: np.ndarray,
    predict_base_prediction: np.ndarray,
    selection: MLPSelection,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    model = ResidualMLP(
        input_dim=fit_x.shape[1],
        output_dim=fit_y.shape[1],
        width=selection.width,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=selection.weight_decay,
    )
    x_fit = _tensor(fit_x)
    residual_fit = _tensor(fit_y - fit_base_prediction)
    for _ in range(selection.best_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(x_fit) - residual_fit))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        residual = model(_tensor(predict_x)).cpu().numpy().astype(np.float64)
    return np.asarray(predict_base_prediction + residual, dtype=np.float64)


def coordinate_detail(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    fisher: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    expected = np.asarray(truth, dtype=np.float64)
    fitted = np.asarray(prediction, dtype=np.float64)
    if expected.shape != fitted.shape or expected.ndim != 2:
        raise ValueError("Truth and prediction matrices must have matching shapes.")
    error = fitted - expected
    squared_error = np.sum(error**2, axis=1)
    truth_norm = np.linalg.norm(expected, axis=1)
    prediction_norm = np.linalg.norm(fitted, axis=1)
    cosine = np.divide(
        np.sum(expected * fitted, axis=1),
        truth_norm * prediction_norm,
        out=np.zeros(len(expected), dtype=np.float64),
        where=(truth_norm * prediction_norm) > 1e-12,
    )
    result = {
        "squared_error": squared_error,
        "error_norm": np.sqrt(squared_error),
        "truth_norm": truth_norm,
        "prediction_norm": prediction_norm,
        "cosine": cosine,
    }
    if fisher is not None:
        fisher_values = np.asarray(fisher, dtype=np.float64)
        if fisher_values.shape != (len(expected), expected.shape[1], expected.shape[1]):
            raise ValueError("Fisher matrices are not aligned with coordinates.")
        result["fisher_weighted_error"] = np.einsum(
            "bi,bij,bj->b",
            error,
            fisher_values,
            error,
            optimize=True,
        )
    return result


def coordinate_summary(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    train_center: np.ndarray,
    fisher: np.ndarray | None = None,
) -> dict[str, float | int]:
    detail = coordinate_detail(truth=truth, prediction=prediction, fisher=fisher)
    expected = np.asarray(truth, dtype=np.float64)
    denominator = float(
        np.sum(
            np.square(expected - np.asarray(train_center, dtype=np.float64)[None, :])
        )
    )
    numerator = float(np.sum(detail["squared_error"]))
    result: dict[str, float | int] = {
        "num_examples": int(len(expected)),
        "fraction_variance_explained": float(
            1.0 - numerator / max(denominator, 1e-24)
        ),
        "mean_squared_error_per_row": float(np.mean(detail["squared_error"])),
        "rmse_per_row": float(np.sqrt(np.mean(detail["squared_error"]))),
        "mean_cosine": float(np.mean(detail["cosine"])),
        "mean_truth_norm": float(np.mean(detail["truth_norm"])),
        "mean_prediction_norm": float(np.mean(detail["prediction_norm"])),
        "mean_norm_ratio": float(
            np.mean(
                np.divide(
                    detail["prediction_norm"],
                    detail["truth_norm"],
                    out=np.zeros(len(expected), dtype=np.float64),
                    where=detail["truth_norm"] > 1e-12,
                )
            )
        ),
    }
    if "fisher_weighted_error" in detail:
        result["mean_fisher_weighted_error"] = float(
            np.mean(detail["fisher_weighted_error"])
        )
    return result
