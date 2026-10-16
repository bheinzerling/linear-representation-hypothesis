from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.number_property_probes import property_labels
from src.probe_manifold_ablation import orthonormal_column_basis


@dataclass(frozen=True)
class PropertyCoordinateModel:
    property_id: str
    basis: np.ndarray
    kind: str
    classes: np.ndarray | None
    class_centroids: np.ndarray | None
    continuous_coefficients: np.ndarray | None

    def coordinates(self, values: np.ndarray) -> np.ndarray:
        value_array = np.asarray(values, dtype=np.int64).reshape(-1)
        if self.kind == "categorical":
            if self.classes is None or self.class_centroids is None:
                raise ValueError(f"Categorical model {self.property_id!r} is incomplete.")
            labels = property_labels(self.property_id, value_array)
            class_to_index = {int(label): index for index, label in enumerate(self.classes.tolist())}
            missing = sorted(set(int(label) for label in labels.tolist()) - set(class_to_index))
            if missing:
                raise ValueError(f"Property {self.property_id!r} has unseen synthesis labels: {missing}")
            indices = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)
            return np.asarray(self.class_centroids[indices], dtype=np.float64)
        if self.kind == "continuous_linear":
            if self.continuous_coefficients is None:
                raise ValueError(f"Continuous model {self.property_id!r} is incomplete.")
            design = np.column_stack([value_array.astype(np.float64), np.ones(len(value_array), dtype=np.float64)])
            return np.asarray(design @ self.continuous_coefficients, dtype=np.float64)
        raise ValueError(f"Unsupported coordinate model kind: {self.kind!r}")

    def vectors(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.coordinates(values) @ self.basis.T, dtype=np.float64)


@dataclass(frozen=True)
class PropertySynthesisModel:
    center: np.ndarray
    coordinate_models: Mapping[str, PropertyCoordinateModel]

    @property
    def hidden_dim(self) -> int:
        return int(len(self.center))

    def validate_property_ids(self, property_ids: Sequence[str]) -> tuple[str, ...]:
        selected = tuple(str(property_id) for property_id in property_ids)
        missing = sorted(set(selected) - set(self.coordinate_models))
        if missing:
            raise ValueError(f"Missing coordinate models for properties: {missing}")
        return selected


@dataclass(frozen=True)
class PropertyUnionEigenbasis:
    property_ids: tuple[str, ...]
    basis: np.ndarray
    singular_values: np.ndarray
    property_energy: np.ndarray
    property_shares: np.ndarray
    concatenated_direction_count: int

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def overlap_multiplicity(self) -> np.ndarray:
        return np.asarray(self.singular_values**2, dtype=np.float64)

    @property
    def effective_property_count(self) -> np.ndarray:
        denominator = np.sum(self.property_shares**2, axis=0)
        return np.divide(
            1.0,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 1e-24,
        )


@dataclass(frozen=True)
class PropertyFeatureBlock:
    property_id: str
    kind: str
    classes: np.ndarray | None
    train_mean: np.ndarray
    block_scale: float
    start: int
    stop: int


@dataclass(frozen=True)
class PropertyFeatureEncoder:
    property_ids: tuple[str, ...]
    blocks: tuple[PropertyFeatureBlock, ...]
    feature_names: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, values: np.ndarray) -> np.ndarray:
        value_array = np.asarray(values, dtype=np.int64).reshape(-1)
        features = np.zeros((len(value_array), self.feature_dim), dtype=np.float64)
        for block in self.blocks:
            if block.kind == "continuous":
                raw = value_array.astype(np.float64)[:, None]
            elif block.kind == "categorical":
                if block.classes is None:
                    raise ValueError(f"Categorical block {block.property_id!r} has no classes.")
                labels = property_labels(block.property_id, value_array)
                class_to_index = {int(label): index for index, label in enumerate(block.classes.tolist())}
                missing = sorted(set(int(label) for label in labels.tolist()) - set(class_to_index))
                if missing:
                    raise ValueError(f"Property {block.property_id!r} has unseen feature labels: {missing}")
                raw = np.zeros((len(value_array), len(block.classes)), dtype=np.float64)
                indices = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)
                raw[np.arange(len(value_array)), indices] = 1.0
            else:
                raise ValueError(f"Unsupported feature block kind: {block.kind!r}")
            features[:, block.start : block.stop] = (raw - block.train_mean[None, :]) / block.block_scale
        return features


@dataclass(frozen=True)
class UnionRidgeDecoder:
    center: np.ndarray
    eigenbasis: PropertyUnionEigenbasis
    feature_encoder: PropertyFeatureEncoder
    coefficients: np.ndarray
    alpha: float

    def predict_coordinates(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.feature_encoder.transform(values) @ self.coefficients, dtype=np.float64)

    def synthesize(self, values: np.ndarray) -> np.ndarray:
        coordinates = self.predict_coordinates(values)
        return np.asarray(self.center[None, :] + coordinates @ self.eigenbasis.basis.T, dtype=np.float64)


def property_union_eigenbasis(
    property_bases: Mapping[str, np.ndarray],
    *,
    property_ids: Sequence[str] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> PropertyUnionEigenbasis:
    selected = tuple(property_bases) if property_ids is None else tuple(str(value) for value in property_ids)
    if not selected:
        raise ValueError("At least one property basis is required.")
    missing = sorted(set(selected) - set(property_bases))
    if missing:
        raise ValueError(f"Missing property bases: {missing}")
    bases = [np.asarray(property_bases[property_id], dtype=np.float64) for property_id in selected]
    hidden_dims = {basis.shape[0] for basis in bases if basis.ndim == 2}
    if len(hidden_dims) != 1 or any(basis.ndim != 2 or basis.shape[1] == 0 for basis in bases):
        raise ValueError(f"Invalid property basis shapes: {[basis.shape for basis in bases]}")
    concatenated = np.concatenate(bases, axis=1)
    left, singular_values, _ = np.linalg.svd(concatenated, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("The concatenated property basis has no singular values.")
    threshold = max(float(atol), float(rtol) * float(singular_values[0]))
    rank = int(np.sum(singular_values > threshold))
    if rank <= 0:
        raise ValueError("The concatenated property basis has numerical rank zero.")
    union_basis = np.asarray(left[:, :rank], dtype=np.float64)
    kept_singular = np.asarray(singular_values[:rank], dtype=np.float64)
    property_energy = np.stack(
        [np.sum((basis.T @ union_basis) ** 2, axis=0) for basis in bases],
        axis=0,
    )
    energy_total = property_energy.sum(axis=0)
    property_shares = np.divide(
        property_energy,
        energy_total[None, :],
        out=np.zeros_like(property_energy),
        where=energy_total[None, :] > 1e-24,
    )
    if not np.allclose(energy_total, kept_singular**2, rtol=1e-7, atol=1e-9):
        raise ValueError("Property-energy reconciliation does not match the union singular spectrum.")
    return PropertyUnionEigenbasis(
        property_ids=selected,
        basis=union_basis,
        singular_values=kept_singular,
        property_energy=np.asarray(property_energy, dtype=np.float64),
        property_shares=np.asarray(property_shares, dtype=np.float64),
        concatenated_direction_count=int(concatenated.shape[1]),
    )


def truncate_property_union_eigenbasis(
    eigenbasis: PropertyUnionEigenbasis,
    *,
    rank: int,
) -> PropertyUnionEigenbasis:
    retained_rank = int(rank)
    if retained_rank <= 0 or retained_rank > eigenbasis.rank:
        raise ValueError(f"Requested rank {retained_rank} outside 1..{eigenbasis.rank}.")
    return PropertyUnionEigenbasis(
        property_ids=eigenbasis.property_ids,
        basis=np.asarray(eigenbasis.basis[:, :retained_rank], dtype=np.float64),
        singular_values=np.asarray(eigenbasis.singular_values[:retained_rank], dtype=np.float64),
        property_energy=np.asarray(eigenbasis.property_energy[:, :retained_rank], dtype=np.float64),
        property_shares=np.asarray(eigenbasis.property_shares[:, :retained_rank], dtype=np.float64),
        concatenated_direction_count=eigenbasis.concatenated_direction_count,
    )


def fit_property_feature_encoder(
    *,
    train_values: np.ndarray,
    property_ids: Sequence[str],
    continuous_properties: Sequence[str] = ("magnitude",),
) -> PropertyFeatureEncoder:
    values = np.asarray(train_values, dtype=np.int64).reshape(-1)
    selected = tuple(str(property_id) for property_id in property_ids)
    continuous = set(str(property_id) for property_id in continuous_properties)
    blocks: list[PropertyFeatureBlock] = []
    names: list[str] = []
    start = 0
    for property_id in selected:
        if property_id in continuous:
            raw = values.astype(np.float64)[:, None]
            classes = None
            kind = "continuous"
            block_names = [property_id]
        else:
            labels = property_labels(property_id, values)
            classes = np.unique(labels)
            raw = np.zeros((len(values), len(classes)), dtype=np.float64)
            class_to_index = {int(label): index for index, label in enumerate(classes.tolist())}
            indices = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)
            raw[np.arange(len(values)), indices] = 1.0
            kind = "categorical"
            block_names = [f"{property_id}={int(label)}" for label in classes]
        train_mean = raw.mean(axis=0)
        centered = raw - train_mean[None, :]
        block_scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if block_scale <= 1e-12:
            raise ValueError(f"Property feature block {property_id!r} is constant in training data.")
        stop = start + raw.shape[1]
        blocks.append(
            PropertyFeatureBlock(
                property_id=property_id,
                kind=kind,
                classes=None if classes is None else np.asarray(classes, dtype=np.int64),
                train_mean=np.asarray(train_mean, dtype=np.float64),
                block_scale=block_scale,
                start=start,
                stop=stop,
            )
        )
        names.extend(block_names)
        start = stop
    return PropertyFeatureEncoder(property_ids=selected, blocks=tuple(blocks), feature_names=tuple(names))


def _ridge_coefficients(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if float(alpha) < 0.0:
        raise ValueError("Ridge alpha must be non-negative.")
    if float(alpha) == 0.0:
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return np.asarray(coefficients, dtype=np.float64)
    gram = (x.T @ x) / max(len(x), 1)
    cross = (x.T @ y) / max(len(x), 1)
    return np.asarray(
        np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64), cross),
        dtype=np.float64,
    )


def select_union_ridge_decoder(
    *,
    train_activations: np.ndarray,
    train_values: np.ndarray,
    dev_activations: np.ndarray,
    dev_values: np.ndarray,
    property_bases: Mapping[str, np.ndarray],
    property_ids: Sequence[str],
    alphas: Sequence[float],
    retained_energy_fractions: Sequence[float] = (1.0,),
    continuous_properties: Sequence[str] = ("magnitude",),
    rtol: float = 1e-8,
    atol: float = 1e-10,
    eigenbasis: PropertyUnionEigenbasis | None = None,
) -> tuple[UnionRidgeDecoder, list[dict[str, float | int]]]:
    train_x = np.asarray(train_activations, dtype=np.float64)
    dev_x = np.asarray(dev_activations, dtype=np.float64)
    train_value_array = np.asarray(train_values, dtype=np.int64).reshape(-1)
    dev_value_array = np.asarray(dev_values, dtype=np.int64).reshape(-1)
    if train_x.ndim != 2 or dev_x.ndim != 2 or train_x.shape[1] != dev_x.shape[1]:
        raise ValueError(f"Invalid train/dev activation shapes: {train_x.shape}, {dev_x.shape}")
    if len(train_x) != len(train_value_array) or len(dev_x) != len(dev_value_array):
        raise ValueError("Activation/value row mismatch in union ridge decoder selection.")
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if not alpha_values:
        raise ValueError("At least one ridge alpha is required.")
    energy_fractions = tuple(float(value) for value in retained_energy_fractions)
    if not energy_fractions or any(not (0.0 < value <= 1.0) for value in energy_fractions):
        raise ValueError("Retained SVD energy fractions must lie in (0, 1].")
    full_eigenbasis = eigenbasis
    if full_eigenbasis is None:
        full_eigenbasis = property_union_eigenbasis(
            property_bases,
            property_ids=property_ids,
            rtol=rtol,
            atol=atol,
        )
    elif full_eigenbasis.property_ids != tuple(str(value) for value in property_ids):
        raise ValueError("Provided eigenbasis property order does not match property_ids.")
    encoder = fit_property_feature_encoder(
        train_values=train_value_array,
        property_ids=property_ids,
        continuous_properties=continuous_properties,
    )
    center = train_x.mean(axis=0)
    train_features = encoder.transform(train_value_array)
    dev_features = encoder.transform(dev_value_array)
    dev_centered = dev_x - center[None, :]
    train_total = max(float(np.sum((train_x - center[None, :]) ** 2)), 1e-24)
    dev_total = max(float(np.sum(dev_centered**2)), 1e-24)
    rows: list[dict[str, float | int]] = []
    candidates: list[UnionRidgeDecoder] = []
    gram = (train_features.T @ train_features) / max(len(train_features), 1)
    gram_eigenvalues = np.linalg.eigvalsh(gram)
    cumulative_energy = np.cumsum(full_eigenbasis.singular_values**2)
    total_direction_energy = float(cumulative_energy[-1])
    candidate_ranks = sorted(
        {
            full_eigenbasis.rank
            if fraction == 1.0
            else min(
                int(np.searchsorted(cumulative_energy / total_direction_energy, fraction, side="left") + 1),
                full_eigenbasis.rank,
            )
            for fraction in energy_fractions
        }
    )
    for retained_rank in candidate_ranks:
        retained_basis = truncate_property_union_eigenbasis(full_eigenbasis, rank=retained_rank)
        train_coordinates = (train_x - center[None, :]) @ retained_basis.basis
        dev_coordinates = dev_centered @ retained_basis.basis
        dev_union_total = max(float(np.sum(dev_coordinates**2)), 1e-24)
        retained_direction_energy_fraction = float(
            cumulative_energy[retained_rank - 1] / total_direction_energy
        )
        for alpha in alpha_values:
            coefficients = _ridge_coefficients(train_features, train_coordinates, alpha)
            decoder = UnionRidgeDecoder(
                center=np.asarray(center, dtype=np.float64),
                eigenbasis=retained_basis,
                feature_encoder=encoder,
                coefficients=coefficients,
                alpha=float(alpha),
            )
            candidates.append(decoder)
            train_prediction = train_features @ coefficients
            dev_prediction = dev_features @ coefficients
            train_residual = (train_x - center[None, :]) - train_prediction @ retained_basis.basis.T
            dev_residual = dev_centered - dev_prediction @ retained_basis.basis.T
            if alpha == 0.0:
                effective_df = float(
                    np.sum(gram_eigenvalues > 1e-10 * max(float(gram_eigenvalues[-1]), 1e-24))
                )
            else:
                effective_df = float(np.sum(gram_eigenvalues / (gram_eigenvalues + alpha)))
            rows.append(
                {
                    "alpha": float(alpha),
                    "train_fraction_variance_explained": float(
                        1.0 - np.sum(train_residual**2) / train_total
                    ),
                    "dev_fraction_variance_explained": float(1.0 - np.sum(dev_residual**2) / dev_total),
                    "dev_union_coordinate_fraction_explained": float(
                        1.0 - np.sum((dev_coordinates - dev_prediction) ** 2) / dev_union_total
                    ),
                    "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
                    "effective_df": effective_df,
                    "feature_dim": encoder.feature_dim,
                    "union_rank": retained_basis.rank,
                    "full_union_rank": full_eigenbasis.rank,
                    "retained_direction_energy_fraction": retained_direction_energy_fraction,
                }
            )
    best_index = max(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["dev_fraction_variance_explained"]),
            -int(rows[index]["union_rank"]),
            -float(rows[index]["alpha"]),
        ),
    )
    return candidates[best_index], rows


def fit_property_synthesis_model(
    *,
    train_activations: np.ndarray,
    train_values: np.ndarray,
    property_bases: Mapping[str, np.ndarray],
    continuous_properties: Sequence[str] = ("magnitude",),
) -> PropertySynthesisModel:
    train_x = np.asarray(train_activations, dtype=np.float64)
    values = np.asarray(train_values, dtype=np.int64).reshape(-1)
    if train_x.ndim != 2 or train_x.shape[0] != len(values):
        raise ValueError(f"Train activation/value mismatch: {train_x.shape} vs {values.shape}.")
    center = train_x.mean(axis=0)
    centered = train_x - center[None, :]
    continuous = set(str(property_id) for property_id in continuous_properties)
    models: dict[str, PropertyCoordinateModel] = {}
    for property_id, raw_basis in property_bases.items():
        basis = np.asarray(raw_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != train_x.shape[1] or basis.shape[1] == 0:
            raise ValueError(f"Invalid basis for {property_id!r}: {basis.shape}.")
        coordinates = centered @ basis
        if property_id in continuous:
            design = np.column_stack([values.astype(np.float64), np.ones(len(values), dtype=np.float64)])
            coefficients, _, _, _ = np.linalg.lstsq(design, coordinates, rcond=None)
            models[property_id] = PropertyCoordinateModel(
                property_id=str(property_id),
                basis=basis,
                kind="continuous_linear",
                classes=None,
                class_centroids=None,
                continuous_coefficients=np.asarray(coefficients, dtype=np.float64),
            )
        else:
            labels = property_labels(property_id, values)
            classes = np.unique(labels)
            centroids = np.stack([coordinates[labels == label].mean(axis=0) for label in classes], axis=0)
            models[property_id] = PropertyCoordinateModel(
                property_id=str(property_id),
                basis=basis,
                kind="categorical",
                classes=np.asarray(classes, dtype=np.int64),
                class_centroids=np.asarray(centroids, dtype=np.float64),
                continuous_coefficients=None,
            )
    return PropertySynthesisModel(center=np.asarray(center, dtype=np.float64), coordinate_models=models)


def independent_property_synthesis(
    model: PropertySynthesisModel,
    *,
    values: np.ndarray,
    property_ids: Sequence[str],
) -> np.ndarray:
    selected = model.validate_property_ids(property_ids)
    result = np.repeat(model.center[None, :], len(values), axis=0)
    for property_id in selected:
        result += model.coordinate_models[property_id].vectors(values)
    return np.asarray(result, dtype=np.float64)


def _constraint_system(
    model: PropertySynthesisModel,
    *,
    values: np.ndarray,
    property_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    selected = model.validate_property_ids(property_ids)
    coordinates = np.concatenate([model.coordinate_models[property_id].coordinates(values) for property_id in selected], axis=1)
    map_matrix = np.concatenate([model.coordinate_models[property_id].basis.T for property_id in selected], axis=0)
    return np.asarray(coordinates, dtype=np.float64), np.asarray(map_matrix, dtype=np.float64)


def joint_minimum_norm_property_synthesis(
    model: PropertySynthesisModel,
    *,
    values: np.ndarray,
    property_ids: Sequence[str],
    rcond: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float]]:
    target_coordinates, map_matrix = _constraint_system(model, values=values, property_ids=property_ids)
    decoder = np.linalg.pinv(map_matrix.T, rcond=float(rcond))
    centered_synthesis = target_coordinates @ decoder
    realized_coordinates = centered_synthesis @ map_matrix.T
    residual = realized_coordinates - target_coordinates
    diagnostics = {
        "constraint_rmse": float(np.sqrt(np.mean(residual**2))),
        "constraint_relative_rmse": float(
            np.sqrt(np.sum(residual**2) / max(float(np.sum(target_coordinates**2)), 1e-24))
        ),
        "constraint_count": int(map_matrix.shape[0]),
        "decoder_rank": int(np.linalg.matrix_rank(map_matrix, tol=max(1e-10, 1e-8 * float(np.linalg.svd(map_matrix, compute_uv=False)[0])))),
    }
    return np.asarray(model.center[None, :] + centered_synthesis, dtype=np.float64), diagnostics


def actual_projection_sum(
    model: PropertySynthesisModel,
    *,
    actual_activations: np.ndarray,
    property_ids: Sequence[str],
) -> np.ndarray:
    selected = model.validate_property_ids(property_ids)
    centered = np.asarray(actual_activations, dtype=np.float64) - model.center[None, :]
    result = np.zeros_like(centered)
    for property_id in selected:
        basis = model.coordinate_models[property_id].basis
        result += (centered @ basis) @ basis.T
    return np.asarray(model.center[None, :] + result, dtype=np.float64)


def property_union_basis(
    model: PropertySynthesisModel,
    *,
    property_ids: Sequence[str],
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    selected = model.validate_property_ids(property_ids)
    concatenated = np.concatenate([model.coordinate_models[property_id].basis for property_id in selected], axis=1)
    return orthonormal_column_basis(concatenated, rtol=float(rtol), atol=float(atol))


def oracle_union_projection(
    model: PropertySynthesisModel,
    *,
    actual_activations: np.ndarray,
    property_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    basis = property_union_basis(model, property_ids=property_ids)
    centered = np.asarray(actual_activations, dtype=np.float64) - model.center[None, :]
    projected = (centered @ basis) @ basis.T
    return np.asarray(model.center[None, :] + projected, dtype=np.float64), basis


def rowwise_norm_match_to_actual(
    synthesis: np.ndarray,
    *,
    actual_activations: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    synth_centered = np.asarray(synthesis, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    actual_centered = np.asarray(actual_activations, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    synth_norm = np.linalg.norm(synth_centered, axis=1)
    actual_norm = np.linalg.norm(actual_centered, axis=1)
    scales = np.divide(actual_norm, synth_norm, out=np.zeros_like(actual_norm), where=synth_norm > 1e-12)
    return np.asarray(center[None, :] + synth_centered * scales[:, None], dtype=np.float64)


def geometric_detail(
    *,
    actual_activations: np.ndarray,
    synthesis: np.ndarray,
    center: np.ndarray,
) -> dict[str, np.ndarray]:
    actual = np.asarray(actual_activations, dtype=np.float64)
    synth = np.asarray(synthesis, dtype=np.float64)
    center_row = np.asarray(center, dtype=np.float64).reshape(1, -1)
    if actual.shape != synth.shape:
        raise ValueError(f"Actual/synthesis shape mismatch: {actual.shape} vs {synth.shape}.")
    actual_centered = actual - center_row
    synth_centered = synth - center_row
    residual = actual - synth
    actual_norm = np.linalg.norm(actual_centered, axis=1)
    synth_norm = np.linalg.norm(synth_centered, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    raw_actual_norm = np.linalg.norm(actual, axis=1)
    raw_synth_norm = np.linalg.norm(synth, axis=1)
    centered_cosine = np.divide(
        np.sum(actual_centered * synth_centered, axis=1),
        actual_norm * synth_norm,
        out=np.zeros(len(actual), dtype=np.float64),
        where=(actual_norm * synth_norm) > 1e-24,
    )
    raw_cosine = np.divide(
        np.sum(actual * synth, axis=1),
        raw_actual_norm * raw_synth_norm,
        out=np.zeros(len(actual), dtype=np.float64),
        where=(raw_actual_norm * raw_synth_norm) > 1e-24,
    )
    return {
        "euclidean_distance": residual_norm,
        "squared_error": np.sum(residual**2, axis=1),
        "relative_l2_error": np.divide(
            residual_norm,
            actual_norm,
            out=np.full(len(actual), np.nan, dtype=np.float64),
            where=actual_norm > 1e-12,
        ),
        "centered_cosine_similarity": centered_cosine,
        "centered_cosine_distance": 1.0 - centered_cosine,
        "raw_cosine_similarity": raw_cosine,
        "raw_cosine_distance": 1.0 - raw_cosine,
        "actual_centered_norm": actual_norm,
        "synthesis_centered_norm": synth_norm,
        "centered_norm_ratio": np.divide(
            synth_norm,
            actual_norm,
            out=np.full(len(actual), np.nan, dtype=np.float64),
            where=actual_norm > 1e-12,
        ),
    }


def pairwise_geometry_correlations(
    *,
    actual_activations: np.ndarray,
    synthesis: np.ndarray,
    center: np.ndarray,
    max_pairs: int = 20_000,
    seed: int = 0,
) -> dict[str, float | int]:
    from scipy.stats import spearmanr

    actual = np.asarray(actual_activations, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    synth = np.asarray(synthesis, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    n = len(actual)
    left, right = np.triu_indices(n, k=1)
    if len(left) > int(max_pairs):
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(left), size=int(max_pairs), replace=False)
        left, right = left[indices], right[indices]
    actual_euclidean = np.linalg.norm(actual[left] - actual[right], axis=1)
    synth_euclidean = np.linalg.norm(synth[left] - synth[right], axis=1)

    def cosine_distance(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(cols, axis=1)
        similarity = np.divide(
            np.sum(rows * cols, axis=1), denominator,
            out=np.zeros(len(rows), dtype=np.float64), where=denominator > 1e-24,
        )
        return 1.0 - similarity

    actual_cosine = cosine_distance(actual[left], actual[right])
    synth_cosine = cosine_distance(synth[left], synth[right])
    def safe_spearman(left_values: np.ndarray, right_values: np.ndarray) -> float:
        if np.ptp(left_values) <= 1e-15 or np.ptp(right_values) <= 1e-15:
            return float("nan")
        return float(spearmanr(left_values, right_values).statistic)

    return {
        "pair_count": int(len(left)),
        "pairwise_euclidean_spearman": safe_spearman(actual_euclidean, synth_euclidean),
        "pairwise_cosine_spearman": safe_spearman(actual_cosine, synth_cosine),
    }


def summarize_geometric_fit(
    *,
    actual_activations: np.ndarray,
    synthesis: np.ndarray,
    center: np.ndarray,
    pair_seed: int = 0,
) -> dict[str, float | int]:
    detail = geometric_detail(actual_activations=actual_activations, synthesis=synthesis, center=center)
    total_squared_error = float(np.sum(detail["squared_error"]))
    total_centered_variance = float(np.sum(detail["actual_centered_norm"] ** 2))
    actual_centered = np.asarray(actual_activations, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    synth_centered = np.asarray(synthesis, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    global_cosine = float(
        np.sum(actual_centered * synth_centered)
        / max(float(np.linalg.norm(actual_centered) * np.linalg.norm(synth_centered)), 1e-24)
    )
    summary: dict[str, float | int] = {
        "n": int(len(actual_activations)),
        "fraction_variance_explained": float(1.0 - total_squared_error / max(total_centered_variance, 1e-24)),
        "global_centered_cosine_similarity": global_cosine,
        "mean_euclidean_distance": float(np.mean(detail["euclidean_distance"])),
        "mean_relative_l2_error": float(np.nanmean(detail["relative_l2_error"])),
        "mean_centered_cosine_similarity": float(np.mean(detail["centered_cosine_similarity"])),
        "mean_raw_cosine_similarity": float(np.mean(detail["raw_cosine_similarity"])),
        "mean_centered_norm_ratio": float(np.nanmean(detail["centered_norm_ratio"])),
    }
    summary.update(
        pairwise_geometry_correlations(
            actual_activations=actual_activations,
            synthesis=synthesis,
            center=center,
            seed=int(pair_seed),
        )
    )
    return summary


def distribution_shift_detail(
    *,
    baseline_logits: np.ndarray,
    synthesis_logits: np.ndarray,
    target_token_ids: np.ndarray,
    numeric_token_ids: np.ndarray,
    numeric_values: np.ndarray,
) -> dict[str, np.ndarray]:
    from scipy.special import logsumexp

    baseline = np.asarray(baseline_logits, dtype=np.float32)
    synthesis = np.asarray(synthesis_logits, dtype=np.float32)
    targets = np.asarray(target_token_ids, dtype=np.int64).reshape(-1)
    if baseline.shape != synthesis.shape or baseline.ndim != 2 or baseline.shape[0] != len(targets):
        raise ValueError(f"Logit/target mismatch: {baseline.shape}, {synthesis.shape}, {targets.shape}.")
    numeric_ids = np.asarray(numeric_token_ids, dtype=np.int64).reshape(-1)
    numeric_vals = np.asarray(numeric_values, dtype=np.float64).reshape(-1)
    if len(numeric_ids) != len(numeric_vals):
        raise ValueError("numeric_token_ids and numeric_values must have the same length.")

    baseline_logp = baseline - logsumexp(baseline, axis=1, keepdims=True)
    synthesis_logp = synthesis - logsumexp(synthesis, axis=1, keepdims=True)
    baseline_p = np.exp(baseline_logp).astype(np.float32, copy=False)
    synthesis_p = np.exp(synthesis_logp).astype(np.float32, copy=False)
    rows = np.arange(len(targets), dtype=np.int64)
    baseline_numeric = baseline_p[:, numeric_ids].astype(np.float64)
    synthesis_numeric = synthesis_p[:, numeric_ids].astype(np.float64)
    baseline_numeric_mass = baseline_numeric.sum(axis=1)
    synthesis_numeric_mass = synthesis_numeric.sum(axis=1)
    baseline_numeric_conditional = baseline_numeric / np.clip(baseline_numeric_mass[:, None], 1e-30, None)
    synthesis_numeric_conditional = synthesis_numeric / np.clip(synthesis_numeric_mass[:, None], 1e-30, None)
    numeric_kl = np.sum(
        baseline_numeric_conditional
        * (
            np.log(np.clip(baseline_numeric_conditional, 1e-30, None))
            - np.log(np.clip(synthesis_numeric_conditional, 1e-30, None))
        ),
        axis=1,
    )
    return {
        "full_kl_original_to_synthesis": np.sum(
            baseline_p * (baseline_logp - synthesis_logp), axis=1, dtype=np.float64
        ),
        "full_kl_synthesis_to_original": np.sum(
            synthesis_p * (synthesis_logp - baseline_logp), axis=1, dtype=np.float64
        ),
        "numeric_conditional_kl_original_to_synthesis": numeric_kl,
        "baseline_top1_token_id": np.argmax(baseline, axis=1).astype(np.int64),
        "synthesis_top1_token_id": np.argmax(synthesis, axis=1).astype(np.int64),
        "top1_agreement": (np.argmax(baseline, axis=1) == np.argmax(synthesis, axis=1)).astype(np.int64),
        "baseline_correct": (np.argmax(baseline, axis=1) == targets).astype(np.int64),
        "synthesis_correct": (np.argmax(synthesis, axis=1) == targets).astype(np.int64),
        "baseline_target_logprob": baseline_logp[rows, targets].astype(np.float64),
        "synthesis_target_logprob": synthesis_logp[rows, targets].astype(np.float64),
        "target_logprob_delta": (synthesis_logp[rows, targets] - baseline_logp[rows, targets]).astype(np.float64),
        "baseline_target_prob": baseline_p[rows, targets].astype(np.float64),
        "synthesis_target_prob": synthesis_p[rows, targets].astype(np.float64),
        "target_prob_delta": (synthesis_p[rows, targets] - baseline_p[rows, targets]).astype(np.float64),
        "baseline_entropy": (-np.sum(baseline_p * baseline_logp, axis=1, dtype=np.float64)),
        "synthesis_entropy": (-np.sum(synthesis_p * synthesis_logp, axis=1, dtype=np.float64)),
        "baseline_numeric_mass": baseline_numeric_mass,
        "synthesis_numeric_mass": synthesis_numeric_mass,
        "baseline_numeric_expected_value": baseline_numeric_conditional @ numeric_vals,
        "synthesis_numeric_expected_value": synthesis_numeric_conditional @ numeric_vals,
    }
