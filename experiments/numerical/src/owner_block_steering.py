from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.property_subspace_synthesis import PropertyFeatureEncoder


PRIMARY_PROPERTY_ORDER = (
    "hundreds_digit",
    "tens_digit",
    "units_digit",
    "has_repeated_digit",
    "distinct_digit_count",
    "divisible_by_3",
    "residue_mod_4",
    "divisible_by_10",
    "parity",
    "divisible_by_6",
    "divisible_by_8",
    "palindrome",
    "contains_zero",
    "primality",
    "divisible_by_25",
    "digit_count",
    "divisible_by_5",
    "residue_mod_3",
    "divisible_by_4",
)

DEFAULT_RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_AUXILIARY_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)


@dataclass(frozen=True)
class OwnerBlockModel:
    property_id: str
    coordinate_indices: tuple[int, ...]
    feature_start: int
    feature_stop: int
    coefficients: np.ndarray
    alpha: float
    dev_coordinate_fve: float

    @property
    def rank(self) -> int:
        return len(self.coordinate_indices)

    def predict_owner_coordinates(
        self,
        encoder: PropertyFeatureEncoder,
        values: np.ndarray,
    ) -> np.ndarray:
        count = len(np.asarray(values).reshape(-1))
        if self.rank == 0:
            return np.zeros((count, 0), dtype=np.float64)
        features = encoder.transform(values)[:, self.feature_start : self.feature_stop]
        return np.asarray(features @ self.coefficients, dtype=np.float64)

    def delta_coefficients(
        self,
        encoder: PropertyFeatureEncoder,
        source_values: np.ndarray,
        target_values: np.ndarray,
        *,
        ambient_rank: int,
    ) -> np.ndarray:
        source = self.predict_owner_coordinates(encoder, source_values)
        target = self.predict_owner_coordinates(encoder, target_values)
        result = np.zeros((len(source), int(ambient_rank)), dtype=np.float64)
        if self.rank:
            result[:, np.asarray(self.coordinate_indices, dtype=np.int64)] = target - source
        return result


@dataclass(frozen=True)
class PrefixSpec:
    prefix_index: int
    condition_id: str
    added_property: str | None
    active_properties: tuple[str, ...]
    owner_rank_added: int
    owner_rank_total: int
    nominal_rank: int


def owner_indices(
    owners: Sequence[str],
    property_ids: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    owner_values = tuple(str(value) for value in owners)
    selected = tuple(str(value) for value in property_ids)
    unknown = sorted(set(owner_values) - set(selected) - {"magnitude"})
    if unknown:
        raise ValueError(f"Owner labels are absent from the requested properties: {unknown}")
    return {
        property_id: tuple(
            index for index, owner in enumerate(owner_values) if owner == property_id
        )
        for property_id in selected
    }


def build_prefix_specs(
    owners: Sequence[str],
    *,
    property_order: Sequence[str] = PRIMARY_PROPERTY_ORDER,
) -> tuple[PrefixSpec, ...]:
    order = tuple(str(value) for value in property_order)
    by_property = owner_indices(owners, order)
    specs = [
        PrefixSpec(
            prefix_index=0,
            condition_id="p00_magnitude",
            added_property=None,
            active_properties=(),
            owner_rank_added=0,
            owner_rank_total=0,
            nominal_rank=1,
        )
    ]
    active: list[str] = []
    owner_rank_total = 0
    for prefix_index, property_id in enumerate(order, start=1):
        active.append(property_id)
        rank_added = len(by_property[property_id])
        owner_rank_total += rank_added
        specs.append(
            PrefixSpec(
                prefix_index=prefix_index,
                condition_id=f"p{prefix_index:02d}_{property_id}",
                added_property=property_id,
                active_properties=tuple(active),
                owner_rank_added=rank_added,
                owner_rank_total=owner_rank_total,
                nominal_rank=1 + owner_rank_total,
            )
        )
    return tuple(specs)


def _ridge_coefficients(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Ridge features and targets must be aligned matrices.")
    if float(alpha) < 0.0:
        raise ValueError("Ridge alpha must be nonnegative.")
    if float(alpha) == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return np.asarray(coefficients, dtype=np.float64)
    gram = (x.T @ x) / max(len(x), 1)
    cross = (x.T @ y) / max(len(x), 1)
    return np.asarray(
        np.linalg.solve(
            gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64),
            cross,
        ),
        dtype=np.float64,
    )


def _fraction_variance_explained(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    fitted = np.asarray(prediction, dtype=np.float64)
    return float(
        1.0
        - np.sum(np.square(actual - fitted))
        / max(float(np.sum(np.square(actual))), 1e-24)
    )


def fit_owner_block_models(
    *,
    encoder: PropertyFeatureEncoder,
    owners: Sequence[str],
    basis: np.ndarray,
    train_values: np.ndarray,
    train_activations: np.ndarray,
    dev_values: np.ndarray,
    dev_activations: np.ndarray,
    property_ids: Sequence[str] = PRIMARY_PROPERTY_ORDER,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
) -> tuple[dict[str, OwnerBlockModel], list[dict[str, object]]]:
    full_basis = np.asarray(basis, dtype=np.float64)
    if full_basis.ndim != 2:
        raise ValueError("The competitive basis must be a matrix.")
    if len(owners) != full_basis.shape[1]:
        raise ValueError("Direction-owner count must match competitive basis rank.")
    train_x = np.asarray(train_activations, dtype=np.float64)
    dev_x = np.asarray(dev_activations, dtype=np.float64)
    if train_x.shape[1] != full_basis.shape[0] or dev_x.shape[1] != full_basis.shape[0]:
        raise ValueError("Activation hidden size does not match the competitive basis.")
    train_center = train_x.mean(axis=0)
    train_coordinates = (train_x - train_center[None, :]) @ full_basis
    dev_coordinates = (dev_x - train_center[None, :]) @ full_basis
    train_features = encoder.transform(train_values)
    dev_features = encoder.transform(dev_values)
    blocks = {block.property_id: block for block in encoder.blocks}
    by_property = owner_indices(owners, property_ids)
    models: dict[str, OwnerBlockModel] = {}
    diagnostics: list[dict[str, object]] = []
    for property_id in property_ids:
        block = blocks[property_id]
        indices = by_property[property_id]
        if not indices:
            model = OwnerBlockModel(
                property_id=property_id,
                coordinate_indices=(),
                feature_start=block.start,
                feature_stop=block.stop,
                coefficients=np.zeros((block.stop - block.start, 0), dtype=np.float64),
                alpha=0.0,
                dev_coordinate_fve=0.0,
            )
            models[property_id] = model
            diagnostics.append(
                {
                    "property_id": property_id,
                    "owner_rank": 0,
                    "selected_alpha": 0.0,
                    "dev_coordinate_fve": 0.0,
                    "analytic_dummy": True,
                }
            )
            continue
        column_indices = np.asarray(indices, dtype=np.int64)
        x_train = train_features[:, block.start : block.stop]
        x_dev = dev_features[:, block.start : block.stop]
        y_train = train_coordinates[:, column_indices]
        y_dev = dev_coordinates[:, column_indices]
        candidates: list[tuple[float, float, np.ndarray]] = []
        for alpha in alphas:
            coefficients = _ridge_coefficients(x_train, y_train, float(alpha))
            score = _fraction_variance_explained(y_dev, x_dev @ coefficients)
            candidates.append((score, float(alpha), coefficients))
        selected_score, selected_alpha, selected_coefficients = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        model = OwnerBlockModel(
            property_id=property_id,
            coordinate_indices=indices,
            feature_start=block.start,
            feature_stop=block.stop,
            coefficients=selected_coefficients,
            alpha=selected_alpha,
            dev_coordinate_fve=selected_score,
        )
        models[property_id] = model
        diagnostics.append(
            {
                "property_id": property_id,
                "owner_rank": len(indices),
                "selected_alpha": selected_alpha,
                "dev_coordinate_fve": selected_score,
                "analytic_dummy": False,
            }
        )
    return models, diagnostics


def cumulative_auxiliary_coefficients(
    property_coefficients: Mapping[str, np.ndarray],
    specs: Sequence[PrefixSpec],
) -> dict[str, np.ndarray]:
    first = next(iter(property_coefficients.values()))
    running = np.zeros_like(np.asarray(first, dtype=np.float64))
    result: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.added_property is not None:
            running = running + np.asarray(
                property_coefficients[spec.added_property],
                dtype=np.float64,
            )
        result[spec.condition_id] = running.copy()
    return result


def orthonormal_prefix_coordinate_basis(
    *,
    magnitude_coordinate_direction: np.ndarray,
    owners: Sequence[str],
    active_properties: Sequence[str],
    rtol: float = 1e-10,
) -> np.ndarray:
    magnitude = np.asarray(magnitude_coordinate_direction, dtype=np.float64).reshape(-1)
    magnitude_norm = float(np.linalg.norm(magnitude))
    if magnitude_norm <= 0.0:
        raise ValueError("Magnitude coordinate direction has zero norm.")
    magnitude = magnitude / magnitude_norm
    active = set(str(value) for value in active_properties)
    selected_indices = [
        index
        for index, property_id in enumerate(owners)
        if str(property_id) in active
    ]
    columns = [magnitude[:, None]]
    if selected_indices:
        selector = np.eye(len(owners), dtype=np.float64)[
            :, np.asarray(selected_indices, dtype=np.int64)
        ]
        columns.append(selector)
    candidate = np.concatenate(columns, axis=1)
    left, singular, _ = np.linalg.svd(candidate, full_matrices=False)
    if not len(singular):
        raise ValueError("Prefix coordinate candidate is empty.")
    rank = int(np.sum(singular > float(rtol) * float(singular[0])))
    return np.asarray(left[:, :rank], dtype=np.float64)


def rescale_rows_to_norm(
    rows: np.ndarray,
    target_norms: np.ndarray,
    *,
    fallback_rows: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    target = np.asarray(target_norms, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or len(values) != len(target):
        raise ValueError("Rows and target norms must align.")
    norms = np.linalg.norm(values, axis=1)
    result = values.copy()
    usable = norms > 1e-12
    result[usable] *= (target[usable] / norms[usable])[:, None]
    missing = (~usable) & (target > 1e-12)
    if np.any(missing):
        if fallback_rows is None:
            raise ValueError("Cannot norm-match zero rows without fallback rows.")
        fallback = np.asarray(fallback_rows, dtype=np.float64)
        if fallback.shape != values.shape:
            raise ValueError("Fallback rows must match the input shape.")
        fallback_norms = np.linalg.norm(fallback, axis=1)
        if np.any(fallback_norms[missing] <= 1e-12):
            raise ValueError("Fallback rows are also zero where a positive norm is required.")
        result[missing] = (
            fallback[missing]
            * (target[missing] / fallback_norms[missing])[:, None]
        )
    result[target <= 1e-12] = 0.0
    return result


def select_scale_by_mae(
    scale_to_abs_error: Mapping[float, np.ndarray],
) -> tuple[float, list[dict[str, float]]]:
    if not scale_to_abs_error:
        raise ValueError("At least one scale candidate is required.")
    rows: list[dict[str, float]] = []
    for scale, errors in scale_to_abs_error.items():
        values = np.asarray(errors, dtype=np.float64)
        finite = np.isfinite(values)
        mae = float(np.mean(values[finite])) if np.any(finite) else float("inf")
        rows.append(
            {
                "scale": float(scale),
                "mae": mae,
                "valid_fraction": float(np.mean(finite)),
            }
        )
    selected = min(rows, key=lambda row: (row["mae"], row["scale"]))
    return float(selected["scale"]), rows
