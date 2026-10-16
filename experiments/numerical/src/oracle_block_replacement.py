from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from src.owner_block_steering import PRIMARY_PROPERTY_ORDER


@dataclass(frozen=True)
class OwnerBlock:
    property_id: str
    coordinate_indices: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.coordinate_indices)


@dataclass(frozen=True)
class ReplacementCondition:
    condition_id: str
    replaced_blocks: tuple[str, ...]
    condition_kind: str


def build_owner_blocks(
    owners: Sequence[str],
    *,
    property_order: Sequence[str] = ("magnitude", *PRIMARY_PROPERTY_ORDER),
    include_empty: bool = False,
) -> tuple[OwnerBlock, ...]:
    owner_values = tuple(str(value) for value in owners)
    order = tuple(str(value) for value in property_order)
    unknown = sorted(set(owner_values) - set(order))
    if unknown:
        raise ValueError(f"Unknown owner labels: {unknown}")
    blocks = tuple(
        OwnerBlock(
            property_id=property_id,
            coordinate_indices=tuple(
                index
                for index, owner in enumerate(owner_values)
                if owner == property_id
            ),
        )
        for property_id in order
    )
    selected = blocks if include_empty else tuple(block for block in blocks if block.rank)
    covered = sorted(
        index for block in selected for index in block.coordinate_indices
    )
    if covered != list(range(len(owner_values))):
        missing = sorted(set(range(len(owner_values))) - set(covered))
        if missing:
            raise ValueError(f"Owner blocks do not cover coordinates: {missing}")
    return selected


def build_replacement_conditions(
    blocks: Sequence[OwnerBlock],
) -> tuple[ReplacementCondition, ...]:
    property_ids = tuple(block.property_id for block in blocks if block.rank)
    conditions = [
        ReplacementCondition(
            condition_id="oracle",
            replaced_blocks=(),
            condition_kind="oracle",
        )
    ]
    conditions.extend(
        ReplacementCondition(
            condition_id=f"replace__{property_id}",
            replaced_blocks=(property_id,),
            condition_kind="single",
        )
        for property_id in property_ids
    )
    conditions.extend(
        ReplacementCondition(
            condition_id=f"replace__{left}__{right}",
            replaced_blocks=(left, right),
            condition_kind="pair",
        )
        for left, right in combinations(property_ids, 2)
    )
    conditions.append(
        ReplacementCondition(
            condition_id="replace__all",
            replaced_blocks=property_ids,
            condition_kind="all",
        )
    )
    return tuple(conditions)


def replace_owner_blocks(
    oracle_coefficients: np.ndarray,
    predicted_by_block: Mapping[str, np.ndarray],
    block_by_property: Mapping[str, OwnerBlock],
    replaced_blocks: Sequence[str],
) -> np.ndarray:
    oracle = np.asarray(oracle_coefficients, dtype=np.float64)
    if oracle.ndim != 2:
        raise ValueError("Oracle coefficients must be a row-by-coordinate matrix.")
    result = oracle.copy()
    for property_id in replaced_blocks:
        if property_id not in block_by_property:
            raise KeyError(f"Unknown owner block {property_id!r}.")
        if property_id not in predicted_by_block:
            raise KeyError(f"Missing predicted coefficients for {property_id!r}.")
        predicted = np.asarray(predicted_by_block[property_id], dtype=np.float64)
        if predicted.shape != oracle.shape:
            raise ValueError(
                f"Predicted block {property_id!r} has shape {predicted.shape}; "
                f"expected {oracle.shape}."
            )
        indices = np.asarray(
            block_by_property[property_id].coordinate_indices,
            dtype=np.int64,
        )
        result[:, indices] = predicted[:, indices]
    return result


def interaction_contrast(
    pair_loss: np.ndarray,
    left_loss: np.ndarray,
    right_loss: np.ndarray,
    oracle_loss: np.ndarray,
) -> np.ndarray:
    pair = np.asarray(pair_loss, dtype=np.float64)
    left = np.asarray(left_loss, dtype=np.float64)
    right = np.asarray(right_loss, dtype=np.float64)
    oracle = np.asarray(oracle_loss, dtype=np.float64)
    if not (pair.shape == left.shape == right.shape == oracle.shape):
        raise ValueError("All interaction losses must have the same shape.")
    return pair - left - right + oracle


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("P-values must be finite values in [0, 1].")
    count = len(values)
    order = np.argsort(values, kind="stable")
    ranked = values[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def cluster_means(
    values: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(values, dtype=np.float64).reshape(-1)
    group = np.asarray(clusters).reshape(-1)
    if len(observed) != len(group):
        raise ValueError("Values and cluster labels must align.")
    labels, inverse = np.unique(group, return_inverse=True)
    sums = np.bincount(inverse, weights=observed, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels))
    return labels, sums / counts


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    _, means = cluster_means(values, clusters)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(means),
        size=(int(draws), len(means)),
        endpoint=False,
    )
    samples = means[indices].mean(axis=1)
    return (
        float(means.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )
