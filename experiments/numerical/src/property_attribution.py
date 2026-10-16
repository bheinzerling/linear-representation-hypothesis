from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttributionCondition:
    """A subspace condition expressed in coordinates of one frozen full basis."""

    condition_id: str
    family: str
    coordinate_basis: np.ndarray
    property_id: str | None = None
    control_for: str | None = None
    control_draw: int | None = None
    coalition: tuple[str, ...] = ()
    prefix_rank: int | None = None
    null_rank: int | None = None
    sampling_scheme: str | None = None
    draw_seed: str | None = None

    @property
    def rank(self) -> int:
        return int(self.coordinate_basis.shape[1])

    def record(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "family": self.family,
            "property_id": self.property_id,
            "control_for": self.control_for,
            "control_draw": self.control_draw,
            "coalition": list(self.coalition),
            "prefix_rank": self.prefix_rank,
            "null_rank": self.null_rank,
            "sampling_scheme": self.sampling_scheme,
            "draw_seed": self.draw_seed,
            "rank": self.rank,
        }


def stable_hash(payload: object, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def validate_orthonormal_columns(
    matrix: np.ndarray,
    *,
    name: str = "basis",
    atol: float = 1e-7,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got {values.shape}.")
    if values.shape[1] and not np.allclose(
        values.T @ values, np.eye(values.shape[1]), atol=atol, rtol=0.0
    ):
        error = float(np.max(np.abs(values.T @ values - np.eye(values.shape[1]))))
        raise ValueError(f"{name} is not column-orthonormal (max error {error:.3e}).")
    return values


def direction_owners_from_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    split: str,
    method: str = "competitive_deflation",
    expected_rank: int | None = None,
) -> tuple[str, ...]:
    required = {"split", "method", "iteration", "selected_property"}
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"Training diagnostics are missing columns: {missing}")
    selected = diagnostics[
        (diagnostics["split"].astype(str) == str(split))
        & (diagnostics["method"].astype(str) == str(method))
    ].copy()
    selected = selected.dropna(subset=["iteration", "selected_property"])
    selected["iteration"] = selected["iteration"].astype(int)
    selected = selected.sort_values("iteration")
    observed = selected["iteration"].to_numpy(dtype=np.int64)
    if len(observed) and not np.array_equal(observed, np.arange(len(observed))):
        raise ValueError(
            f"Competitive iterations must be contiguous from zero, got {observed.tolist()}."
        )
    owners = tuple(selected["selected_property"].astype(str).tolist())
    if expected_rank is not None and len(owners) != int(expected_rank):
        raise ValueError(
            f"Direction-owner count {len(owners)} does not match basis rank {expected_rank}."
        )
    return owners


def coordinate_selector(ambient_rank: int, indices: Sequence[int]) -> np.ndarray:
    rank = int(ambient_rank)
    selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int64)
    if selected.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("indices must be unique.")
    if np.any(selected < 0) or np.any(selected >= rank):
        raise ValueError(f"Coordinate indices must lie in 0..{rank - 1}.")
    result = np.zeros((rank, len(selected)), dtype=np.float64)
    if len(selected):
        result[selected, np.arange(len(selected))] = 1.0
    return result


def owner_indices(
    owners: Sequence[str],
    property_ids: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    selected = tuple(str(value) for value in property_ids)
    unknown = sorted(set(str(value) for value in owners) - set(selected))
    if unknown:
        raise ValueError(f"Direction owners absent from property_ids: {unknown}")
    return {
        property_id: tuple(
            index for index, owner in enumerate(owners) if str(owner) == property_id
        )
        for property_id in selected
    }


def _condition_id(family: str, suffix: str) -> str:
    return f"{family}__{suffix}"


def build_owner_conditions(
    *,
    owners: Sequence[str],
    property_ids: Sequence[str],
    random_control_draws: int,
    seed: int,
) -> list[AttributionCondition]:
    owners_tuple = tuple(str(value) for value in owners)
    properties = tuple(str(value) for value in property_ids)
    ambient_rank = len(owners_tuple)
    by_property = owner_indices(owners_tuple, properties)
    all_indices = np.arange(ambient_rank, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    conditions = [
        AttributionCondition(
            condition_id="full",
            family="full",
            coordinate_basis=np.eye(ambient_rank, dtype=np.float64),
            coalition=properties,
        )
    ]
    for property_id in properties:
        owned = np.asarray(by_property[property_id], dtype=np.int64)
        kept = np.setdiff1d(all_indices, owned, assume_unique=True)
        conditions.extend(
            [
                AttributionCondition(
                    condition_id=_condition_id("owner_only", property_id),
                    family="owner_only",
                    property_id=property_id,
                    coordinate_basis=coordinate_selector(ambient_rank, owned),
                    coalition=(property_id,),
                ),
                AttributionCondition(
                    condition_id=_condition_id("owner_loo", property_id),
                    family="owner_loo",
                    property_id=property_id,
                    coordinate_basis=coordinate_selector(ambient_rank, kept),
                    coalition=tuple(value for value in properties if value != property_id),
                ),
            ]
        )
        for draw in range(int(random_control_draws)):
            random_owned = np.sort(
                rng.choice(ambient_rank, size=len(owned), replace=False)
            )
            random_kept = np.setdiff1d(
                all_indices, random_owned, assume_unique=True
            )
            conditions.extend(
                [
                    AttributionCondition(
                        condition_id=_condition_id(
                            "matched_owner_only", f"{property_id}__draw{draw:03d}"
                        ),
                        family="matched_owner_only",
                        property_id=property_id,
                        control_for=_condition_id("owner_only", property_id),
                        control_draw=draw,
                        coordinate_basis=coordinate_selector(
                            ambient_rank, random_owned
                        ),
                    ),
                    AttributionCondition(
                        condition_id=_condition_id(
                            "matched_owner_loo", f"{property_id}__draw{draw:03d}"
                        ),
                        family="matched_owner_loo",
                        property_id=property_id,
                        control_for=_condition_id("owner_loo", property_id),
                        control_draw=draw,
                        coordinate_basis=coordinate_selector(
                            ambient_rank, random_kept
                        ),
                    ),
                ]
            )
    return conditions


def build_prefix_conditions(owners: Sequence[str]) -> list[AttributionCondition]:
    owners_tuple = tuple(str(value) for value in owners)
    rank = len(owners_tuple)
    result: list[AttributionCondition] = []
    for prefix_rank in range(rank + 1):
        result.append(
            AttributionCondition(
                condition_id=f"prefix__rank{prefix_rank:03d}",
                family="prefix",
                property_id=(
                    None if prefix_rank == 0 else owners_tuple[prefix_rank - 1]
                ),
                coordinate_basis=coordinate_selector(
                    rank, np.arange(prefix_rank, dtype=np.int64)
                ),
                coalition=tuple(dict.fromkeys(owners_tuple[:prefix_rank])),
                prefix_rank=prefix_rank,
            )
        )
    return result


def build_prefix_edges(owners: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "edge_family": "competitive_prefix",
            "property_id": str(property_id),
            "iteration": iteration,
            "before_condition_id": f"prefix__rank{iteration:03d}",
            "after_condition_id": f"prefix__rank{iteration + 1:03d}",
        }
        for iteration, property_id in enumerate(owners)
    ]


def build_shapley_conditions(
    *,
    owners: Sequence[str],
    property_ids: Sequence[str],
    permutations: int,
    seed: int,
    antithetic_pairs: int = 0,
    antithetic_pair_start: int = 0,
) -> tuple[list[AttributionCondition], list[dict[str, object]]]:
    owners_tuple = tuple(str(value) for value in owners)
    properties = tuple(str(value) for value in property_ids)
    by_property = owner_indices(owners_tuple, properties)
    ambient_rank = len(owners_tuple)
    condition_by_id: dict[str, AttributionCondition] = {}
    edges: list[dict[str, object]] = []
    active_properties = tuple(
        property_id for property_id in properties if by_property[property_id]
    )
    dummy_properties = tuple(
        property_id for property_id in properties if not by_property[property_id]
    )

    def register(coalition: Iterable[str]) -> AttributionCondition:
        # A property with no owner-assigned directions is a literal zero block:
        # adding it must reuse the same coalition condition so its Shapley edge
        # is exactly zero without scheduling a duplicate optimization.
        ordered = tuple(
            sorted(
                property_id
                for property_id in set(str(value) for value in coalition)
                if by_property[property_id]
            )
        )
        if ordered == tuple(sorted(active_properties)):
            condition_id = "full"
        else:
            suffix = "empty" if not ordered else stable_hash(ordered, length=12)
            condition_id = f"coalition__{suffix}"
        if condition_id not in condition_by_id:
            indices = sorted(
                index
                for property_id in ordered
                for index in by_property[property_id]
            )
            condition_by_id[condition_id] = AttributionCondition(
                condition_id=condition_id,
                family="full" if condition_id == "full" else "shapley_coalition",
                coordinate_basis=coordinate_selector(ambient_rank, indices),
                coalition=properties if condition_id == "full" else ordered,
            )
        return condition_by_id[condition_id]

    path_specs: list[tuple[int, int | None, str, tuple[str, ...]]] = []
    if int(antithetic_pairs) > 0:
        if int(permutations) > 0:
            raise ValueError(
                "Choose iid permutations or antithetic pairs, not both."
            )
        pair_start = int(antithetic_pair_start)
        if pair_start < 0:
            raise ValueError("antithetic_pair_start must be nonnegative.")
        for pair_id in range(pair_start, pair_start + int(antithetic_pairs)):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(pair_id)])
            )
            forward = tuple(
                active_properties[index]
                for index in rng.permutation(len(active_properties))
            )
            path_specs.extend(
                [
                    (2 * pair_id, pair_id, "forward", forward),
                    (2 * pair_id + 1, pair_id, "reverse", forward[::-1]),
                ]
            )
    else:
        rng = np.random.default_rng(int(seed))
        for permutation_index in range(int(permutations)):
            permutation = tuple(
                active_properties[index]
                for index in rng.permutation(len(active_properties))
            )
            path_specs.append((permutation_index, None, "iid", permutation))

    for permutation_index, pair_id, orientation, permutation in path_specs:
        coalition: list[str] = []
        before = register(coalition)
        for step, property_id in enumerate(permutation):
            coalition.append(property_id)
            after = register(coalition)
            edges.append(
                {
                    "edge_family": "sampled_shapley",
                    "permutation": permutation_index,
                    "pair_id": pair_id,
                    "orientation": orientation,
                    "step": step,
                    "property_id": property_id,
                    "before_condition_id": before.condition_id,
                    "after_condition_id": after.condition_id,
                }
            )
            before = after
        for property_id in dummy_properties:
            edges.append(
                {
                    "edge_family": "sampled_shapley",
                    "permutation": permutation_index,
                    "pair_id": pair_id,
                    "orientation": orientation,
                    "step": len(permutation),
                    "property_id": property_id,
                    "before_condition_id": before.condition_id,
                    "after_condition_id": before.condition_id,
                    "analytic_dummy": True,
                }
            )
    return list(condition_by_id.values()), edges


def projected_coordinate_basis(
    full_basis: np.ndarray,
    candidate_basis: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the candidate span after projection into ``span(full_basis)``.

    The first result is an orthonormal coordinate basis ``A`` such that
    ``full_basis @ A`` is the projected span. The second result contains the
    canonical correlations between the candidate and full spans.
    """

    full = validate_orthonormal_columns(full_basis, name="full_basis")
    candidate = validate_orthonormal_columns(candidate_basis, name="candidate_basis")
    cross = full.T @ candidate
    left, singular, _ = np.linalg.svd(cross, full_matrices=False)
    if singular.size == 0:
        return np.zeros((full.shape[1], 0), dtype=np.float64), singular
    threshold = max(float(atol), float(rtol) * float(singular[0]))
    retained = int(np.sum(singular > threshold))
    return np.asarray(left[:, :retained], dtype=np.float64), np.asarray(
        singular, dtype=np.float64
    )


def orthogonal_complement_coordinates(
    subspace_coordinates: np.ndarray,
    *,
    ambient_rank: int,
    rtol: float = 1e-10,
) -> np.ndarray:
    coordinates = validate_orthonormal_columns(
        subspace_coordinates, name="subspace_coordinates"
    )
    if coordinates.shape[0] != int(ambient_rank):
        raise ValueError("Subspace coordinates do not match ambient rank.")
    if coordinates.shape[1] == 0:
        return np.eye(int(ambient_rank), dtype=np.float64)
    left, singular, _ = np.linalg.svd(coordinates.T, full_matrices=True)
    del left
    threshold = max(1e-12, float(rtol) * float(singular[0]))
    rank = int(np.sum(singular > threshold))
    _, _, right = np.linalg.svd(coordinates.T, full_matrices=True)
    return np.asarray(right[rank:].T, dtype=np.float64)


def canonical_overlap_table(
    canonical_bases: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    normalized = {
        str(name): validate_orthonormal_columns(value, name=str(name))
        for name, value in canonical_bases.items()
    }
    rows: list[dict[str, object]] = []
    for left_id, left in normalized.items():
        for right_id, right in normalized.items():
            singular = np.linalg.svd(left.T @ right, compute_uv=False)
            shared_energy = float(np.sum(singular**2))
            rows.append(
                {
                    "left_property": left_id,
                    "right_property": right_id,
                    "left_rank": left.shape[1],
                    "right_rank": right.shape[1],
                    "intersection_energy": shared_energy,
                    "left_containment": shared_energy / max(left.shape[1], 1),
                    "right_containment": shared_energy / max(right.shape[1], 1),
                    "largest_canonical_correlation": (
                        float(singular[0]) if len(singular) else 0.0
                    ),
                    "smallest_canonical_correlation": (
                        float(singular[-1]) if len(singular) else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_canonical_conditions(
    *,
    full_basis: np.ndarray,
    canonical_bases: Mapping[str, np.ndarray],
    property_ids: Sequence[str],
    random_control_draws: int = 0,
    random_control_draw_start: int = 0,
    seed: int = 0,
) -> tuple[list[AttributionCondition], pd.DataFrame]:
    full = validate_orthonormal_columns(full_basis, name="full_basis")
    conditions: list[AttributionCondition] = []
    coverage_rows: list[dict[str, object]] = []
    projected_ranks: set[int] = set()
    for property_id in tuple(str(value) for value in property_ids):
        if property_id not in canonical_bases:
            continue
        candidate = validate_orthonormal_columns(
            canonical_bases[property_id], name=f"canonical[{property_id}]"
        )
        coordinates, singular = projected_coordinate_basis(full, candidate)
        projected_ranks.add(int(coordinates.shape[1]))
        complement = orthogonal_complement_coordinates(
            coordinates, ambient_rank=full.shape[1]
        )
        conditions.extend(
            [
                AttributionCondition(
                    condition_id=f"canonical_only__{property_id}",
                    family="canonical_only",
                    property_id=property_id,
                    coordinate_basis=coordinates,
                    coalition=(property_id,),
                ),
                AttributionCondition(
                    condition_id=f"canonical_loo__{property_id}",
                    family="canonical_loo",
                    property_id=property_id,
                    coordinate_basis=complement,
                ),
            ]
        )
        coverage_rows.append(
            {
                "property_id": property_id,
                "canonical_rank": candidate.shape[1],
                "projected_rank": coordinates.shape[1],
                "full_rank": full.shape[1],
                "intersection_energy": float(np.sum(singular**2)),
                "canonical_containment_in_full": float(
                    np.sum(singular**2) / max(candidate.shape[1], 1)
                ),
                "largest_canonical_correlation": (
                    float(singular[0]) if len(singular) else 0.0
                ),
                "smallest_canonical_correlation": (
                    float(singular[-1]) if len(singular) else 0.0
                ),
            }
        )
    draw_start = int(random_control_draw_start)
    if draw_start < 0:
        raise ValueError("random_control_draw_start must be nonnegative.")
    for null_rank in sorted(projected_ranks):
        for draw in range(draw_start, draw_start + int(random_control_draws)):
            draw_seed = np.random.SeedSequence(
                [int(seed), int(null_rank), int(draw)]
            )
            rng = np.random.default_rng(draw_seed)
            if null_rank == 0:
                random_coordinates = np.empty(
                    (full.shape[1], 0), dtype=np.float64
                )
            else:
                random_coordinates, _ = np.linalg.qr(
                    rng.standard_normal(
                        (full.shape[1], null_rank)
                    ),
                    mode="reduced",
                )
            random_complement = orthogonal_complement_coordinates(
                random_coordinates, ambient_rank=full.shape[1]
            )
            conditions.extend(
                [
                    AttributionCondition(
                        condition_id=(
                            f"matched_canonical_rank{null_rank:03d}"
                            f"__draw{draw:03d}__only"
                        ),
                        family="matched_canonical_only",
                        control_for=f"canonical_rank{null_rank:03d}",
                        control_draw=draw,
                        coordinate_basis=random_coordinates,
                        null_rank=null_rank,
                        sampling_scheme="haar_grassmann",
                        draw_seed=stable_hash(
                            [int(seed), int(null_rank), int(draw)], length=16
                        ),
                    ),
                    AttributionCondition(
                        condition_id=(
                            f"matched_canonical_rank{null_rank:03d}"
                            f"__draw{draw:03d}__loo"
                        ),
                        family="matched_canonical_loo",
                        control_for=f"canonical_rank{null_rank:03d}",
                        control_draw=draw,
                        coordinate_basis=random_complement,
                        null_rank=null_rank,
                        sampling_scheme="haar_grassmann",
                        draw_seed=stable_hash(
                            [int(seed), int(null_rank), int(draw)], length=16
                        ),
                    ),
                ]
            )
    return conditions, pd.DataFrame(coverage_rows)


def transform_projected_fisher(
    full_fisher: np.ndarray,
    coordinate_basis: np.ndarray,
) -> np.ndarray:
    fisher = np.asarray(full_fisher, dtype=np.float64)
    coordinates = validate_orthonormal_columns(
        coordinate_basis, name="coordinate_basis"
    )
    if fisher.ndim != 3 or fisher.shape[1:] != (
        coordinates.shape[0],
        coordinates.shape[0],
    ):
        raise ValueError(
            f"Fisher shape {fisher.shape} is incompatible with coordinates "
            f"{coordinates.shape}."
        )
    return np.einsum(
        "ri,nrs,sj->nij", coordinates, fisher, coordinates, optimize=True
    )


def transform_full_coefficients(
    full_coefficients: np.ndarray,
    coordinate_basis: np.ndarray,
) -> np.ndarray:
    values = np.asarray(full_coefficients, dtype=np.float64)
    coordinates = validate_orthonormal_columns(
        coordinate_basis, name="coordinate_basis"
    )
    if values.ndim != 2 or values.shape[1] != coordinates.shape[0]:
        raise ValueError(
            f"Coefficient shape {values.shape} is incompatible with coordinates "
            f"{coordinates.shape}."
        )
    return values @ coordinates


def unique_conditions(
    conditions: Iterable[AttributionCondition],
) -> list[AttributionCondition]:
    by_id: dict[str, AttributionCondition] = {}
    for condition in conditions:
        previous = by_id.get(condition.condition_id)
        if previous is not None and (
            previous.record() != condition.record()
            or not np.allclose(previous.coordinate_basis, condition.coordinate_basis)
        ):
            raise ValueError(f"Conflicting condition id: {condition.condition_id}")
        by_id[condition.condition_id] = condition
    return list(by_id.values())
