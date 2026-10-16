#!/usr/bin/env python3
"""Compare direct and factorial parity directions in a shared activation space."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_factorial_parity_generalization import (  # noqa: E402
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    LOGISTIC_TOL,
    SEED,
    WHITENING_RTOL,
)
from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    DEFAULT_OUTPUT as BAG_OUTPUT,
)
from src.factorial_property_probing import (  # noqa: E402
    apply_raw_feature_transform,
    conditional_probability,
    fit_conditional_logistic,
    parity_factorization,
    raw_feature_transform,
)
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    runtime_metadata,
    set_global_seed,
)
from src.whitened_property_steering import fit_compact_whitening  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "out/parity_probe_direction_contextualization/llama31_8b_layer10_v1"
)
DEFAULT_WHITENED = (
    ROOT / "out/whitened_property_steering/llama31_8b_layer10_v1"
)
DEFAULT_FACTORIAL = (
    ROOT / "out/factorial_parity_generalization/llama31_8b_layer10_v1"
)
DEFAULT_NULLSPACE = (
    ROOT / "out/figure1_iterative_nullspace_property_probes/llama31_8b_layer10_v1"
)
DEFAULT_TRANSCODER = ROOT.parent / "properties"
OLD_SOURCE_REL = (
    "out/parity_activation_steering/"
    "llama31_8b_original2shot_value_disjoint_v1_local"
)
DESIGN_PATH = (
    ROOT
    / "experiment-designs/2026-07-29-parity-probe-direction-contextualization.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bag-output-dir", type=Path, default=BAG_OUTPUT)
    parser.add_argument("--whitened-output-dir", type=Path, default=DEFAULT_WHITENED)
    parser.add_argument("--factorial-output-dir", type=Path, default=DEFAULT_FACTORIAL)
    parser.add_argument("--nullspace-output-dir", type=Path, default=DEFAULT_NULLSPACE)
    parser.add_argument("--transcoder-root", type=Path, default=DEFAULT_TRANSCODER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Direction has zero or non-finite norm.")
    return values / norm


def _orient_even(
    vector: np.ndarray,
    *,
    activations: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    direction = _unit(vector)
    parity = np.asarray(values, dtype=np.int64) % 2
    projection = np.asarray(activations, dtype=np.float64) @ direction
    even_mean = float(projection[parity == 0].mean())
    odd_mean = float(projection[parity == 1].mean())
    if even_mean < odd_mean:
        direction = -direction
    return direction


def _angle(cosine: float, *, axial: bool = False) -> float:
    value = abs(float(cosine)) if axial else float(cosine)
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _load_old_direction(path: Path) -> np.ndarray:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(payload["selected_parity"]["unit"], dtype=np.float64)


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    split_path = args.bag_output_dir / "split00_activations.npz"
    with np.load(split_path, allow_pickle=False) as payload:
        strict = {
            partition: (
                np.asarray(payload[f"{partition}_values"], dtype=np.int64),
                np.asarray(payload[f"{partition}_activations"], dtype=np.float64),
            )
            for partition in ("train", "dev", "test")
        }
        occurrence = np.asarray(
            payload["occurrence_train_by_value"], dtype=np.float64
        )
    all_values = np.arange(1000, dtype=np.int64)
    return {
        "split_path": split_path,
        "strict": strict,
        "occurrence_values": all_values,
        "occurrence_activations": occurrence,
    }


def _add_direction(
    store: dict[str, dict[str, np.ndarray]],
    manifest: list[dict[str, Any]],
    *,
    direction_id: str,
    family: str,
    protocol: str,
    training_context: str,
    decoder: np.ndarray,
    intervention: np.ndarray,
    orientation_activations: np.ndarray,
    orientation_values: np.ndarray,
    space: str,
    held_pair: int | None = None,
) -> None:
    decoder_unit = _orient_even(
        decoder,
        activations=orientation_activations,
        values=orientation_values,
    )
    intervention_unit = _unit(intervention)
    if float(np.dot(decoder_unit, intervention_unit)) < 0.0:
        intervention_unit = -intervention_unit
    store[direction_id] = {
        "decoder": decoder_unit,
        "intervention": intervention_unit,
    }
    manifest.append(
        {
            "direction_id": direction_id,
            "family": family,
            "protocol": protocol,
            "space": space,
            "held_pair": held_pair,
            "training_context": training_context,
            "orientation": "even_positive",
            "decoder_intervention_cosine": float(
                np.dot(decoder_unit, intervention_unit)
            ),
            "decoder_intervention_angle_deg": _angle(
                float(np.dot(decoder_unit, intervention_unit))
            ),
        }
    )


def _fit_factorial_directions(
    *,
    directions: dict[str, dict[str, np.ndarray]],
    manifest: list[dict[str, Any]],
    occurrence_values: np.ndarray,
    occurrence_activations: np.ndarray,
    strict_train_activations: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    factorization = parity_factorization()
    side, pair_id = factorization.labels(occurrence_values)
    center, rms = raw_feature_transform(occurrence_activations)
    raw_features = apply_raw_feature_transform(
        occurrence_activations, center=center, rms_scale=rms
    )
    reference_whitening = fit_compact_whitening(
        strict_train_activations, relative_tolerance=WHITENING_RTOL
    )
    reference_features = reference_whitening.transform(occurrence_activations)
    audit: list[dict[str, Any]] = []

    def fit(
        features: np.ndarray,
        current_side: np.ndarray,
        current_pair: np.ndarray,
        current_seed: int,
    ):
        return fit_conditional_logistic(
            features=features,
            side=current_side,
            pair_id=current_pair,
            c=LOGISTIC_C,
            seed=current_seed,
            max_iter=LOGISTIC_MAX_ITER,
            tol=LOGISTIC_TOL,
        )

    for space, features, whitening in (
        ("raw", raw_features, None),
        ("reference_whitened", reference_features, reference_whitening),
    ):
        model = fit(features, side, pair_id, seed)
        coefficient = np.asarray(model.coef_[0, 5:], dtype=np.float64)
        if whitening is None:
            decoder = coefficient / float(rms)
            intervention = decoder
        else:
            decoder = whitening.basis @ (coefficient / whitening.scales)
            intervention = whitening.basis @ (coefficient * whitening.scales)
        _add_direction(
            directions,
            manifest,
            direction_id=f"factorial_{space}_all_pairs",
            family="factorial",
            protocol="ordinary_all_pairs",
            training_context="addition_formula_source_number",
            decoder=-decoder,
            intervention=-intervention,
            orientation_activations=occurrence_activations,
            orientation_values=occurrence_values,
            space=space,
        )
        probability = conditional_probability(
            model, features=features, pair_id=pair_id
        )
        audit.append(
            {
                "direction_id": f"factorial_{space}_all_pairs",
                "n": len(side),
                "balanced_accuracy": float(
                    (
                        ((probability >= 0.5) & (side == 1)).sum()
                        / max(int((side == 1).sum()), 1)
                        + ((probability < 0.5) & (side == 0)).sum()
                        / max(int((side == 0).sum()), 1)
                    )
                    / 2.0
                ),
                "fit_iterations": int(np.max(model.n_iter_)),
            }
        )

        for held_pair in range(5):
            train_mask = pair_id != held_pair
            model = fit(
                features[train_mask],
                side[train_mask],
                pair_id[train_mask],
                seed + held_pair + 1,
            )
            coefficient = np.asarray(model.coef_[0, 5:], dtype=np.float64)
            if whitening is None:
                decoder = coefficient / float(rms)
                intervention = decoder
            else:
                decoder = whitening.basis @ (coefficient / whitening.scales)
                intervention = whitening.basis @ (coefficient * whitening.scales)
            direction_id = f"factorial_{space}_held_pair_{held_pair}"
            _add_direction(
                directions,
                manifest,
                direction_id=direction_id,
                family="factorial",
                protocol="held_pair",
                training_context="addition_formula_source_number",
                decoder=-decoder,
                intervention=-intervention,
                orientation_activations=occurrence_activations[train_mask],
                orientation_values=occurrence_values[train_mask],
                space=space,
                held_pair=held_pair,
            )
            eval_mask = pair_id == held_pair
            probability = conditional_probability(
                model,
                features=features[eval_mask],
                pair_id=pair_id[eval_mask],
            )
            truth = side[eval_mask]
            audit.append(
                {
                    "direction_id": direction_id,
                    "n": len(truth),
                    "balanced_accuracy": float(
                        (
                            ((probability >= 0.5) & (truth == 1)).sum()
                            / max(int((truth == 1).sum()), 1)
                            + ((probability < 0.5) & (truth == 0)).sum()
                            / max(int((truth == 0).sum()), 1)
                        )
                        / 2.0
                    ),
                    "fit_iterations": int(np.max(model.n_iter_)),
                }
            )

    for held_pair in range(5):
        train_mask = pair_id != held_pair
        whitening = fit_compact_whitening(
            occurrence_activations[train_mask],
            relative_tolerance=WHITENING_RTOL,
        )
        features = whitening.transform(occurrence_activations[train_mask])
        model = fit(
            features,
            side[train_mask],
            pair_id[train_mask],
            seed + held_pair + 100,
        )
        coefficient = np.asarray(model.coef_[0, 5:], dtype=np.float64)
        decoder = whitening.basis @ (coefficient / whitening.scales)
        intervention = whitening.basis @ (coefficient * whitening.scales)
        direction_id = f"factorial_fold_whitened_held_pair_{held_pair}"
        _add_direction(
            directions,
            manifest,
            direction_id=direction_id,
            family="factorial",
            protocol="held_pair",
            training_context="addition_formula_source_number",
            decoder=-decoder,
            intervention=-intervention,
            orientation_activations=occurrence_activations[train_mask],
            orientation_values=occurrence_values[train_mask],
            space="fold_whitened",
            held_pair=held_pair,
        )
        eval_mask = pair_id == held_pair
        probability = conditional_probability(
            model,
            features=whitening.transform(occurrence_activations[eval_mask]),
            pair_id=pair_id[eval_mask],
        )
        truth = side[eval_mask]
        audit.append(
            {
                "direction_id": direction_id,
                "n": len(truth),
                "balanced_accuracy": float(
                    (
                        ((probability >= 0.5) & (truth == 1)).sum()
                        / max(int((truth == 1).sum()), 1)
                        + ((probability < 0.5) & (truth == 0)).sum()
                        / max(int((truth == 0).sum()), 1)
                    )
                    / 2.0
                ),
                "fit_iterations": int(np.max(model.n_iter_)),
                "whitening_rank": int(whitening.rank),
            }
        )
    return audit


def _geometry_tables(
    directions: Mapping[str, Mapping[str, np.ndarray]],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_id = "historical_direct"
    reference = directions[reference_id]
    rows: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for surface in ("decoder", "intervention"):
        for direction_id, payload in directions.items():
            cosine = float(np.dot(payload[surface], reference[surface]))
            rows.append(
                {
                    "surface": surface,
                    "direction_id": direction_id,
                    "reference_id": reference_id,
                    "signed_cosine": cosine,
                    "signed_angle_deg": _angle(cosine),
                    "axial_cosine": abs(cosine),
                    "axial_angle_deg": _angle(cosine, axial=True),
                }
            )
        ids = list(directions)
        for left in ids:
            for right in ids:
                cosine = float(
                    np.dot(directions[left][surface], directions[right][surface])
                )
                pairwise.append(
                    {
                        "surface": surface,
                        "left_direction_id": left,
                        "right_direction_id": right,
                        "signed_cosine": cosine,
                        "signed_angle_deg": _angle(cosine),
                        "axial_angle_deg": _angle(cosine, axial=True),
                    }
                )

    stability: list[dict[str, Any]] = []
    for surface in ("decoder", "intervention"):
        for space in ("raw", "reference_whitened", "fold_whitened"):
            ids = manifest.loc[
                manifest["space"].eq(space)
                & manifest["protocol"].eq("held_pair"),
                "direction_id",
            ].tolist()
            if len(ids) != 5:
                continue
            cosines = [
                float(np.dot(directions[ids[i]][surface], directions[ids[j]][surface]))
                for i in range(len(ids))
                for j in range(i + 1, len(ids))
            ]
            stability.append(
                {
                    "surface": surface,
                    "space": space,
                    "direction_count": len(ids),
                    "pair_count": len(cosines),
                    "signed_cosine_min": float(np.min(cosines)),
                    "signed_cosine_median": float(np.median(cosines)),
                    "signed_cosine_max": float(np.max(cosines)),
                    "signed_angle_median_deg": _angle(float(np.median(cosines))),
                }
            )

    coverage: list[dict[str, Any]] = []
    direct_ids = manifest.loc[manifest["family"].ne("factorial"), "direction_id"].tolist()
    for surface in ("decoder", "intervention"):
        for space in ("raw", "reference_whitened", "fold_whitened"):
            fold_ids = manifest.loc[
                manifest["space"].eq(space)
                & manifest["protocol"].eq("held_pair"),
                "direction_id",
            ].tolist()
            if len(fold_ids) != 5:
                continue
            matrix = np.stack(
                [directions[direction_id][surface] for direction_id in fold_ids],
                axis=1,
            )
            basis, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            rank = int(np.sum(singular > 1e-10 * singular[0]))
            basis = basis[:, :rank]
            for direct_id in direct_ids:
                vector = directions[direct_id][surface]
                projection_norm = float(np.linalg.norm(basis.T @ vector))
                coverage.append(
                    {
                        "surface": surface,
                        "factorial_space": space,
                        "direct_direction_id": direct_id,
                        "span_rank": rank,
                        "projection_norm": projection_norm,
                        "angle_to_span_deg": float(
                            np.degrees(
                                np.arccos(np.clip(projection_norm, 0.0, 1.0))
                            )
                        ),
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(pairwise),
        pd.DataFrame(stability),
        pd.DataFrame(coverage),
    )


def _collect_predictive_context(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    historical_path = (
        args.transcoder_root / OLD_SOURCE_REL / "probe_layer_metrics.csv"
    )
    historical = pd.read_csv(historical_path)
    historical = historical.loc[historical["layer"].eq(9)].iloc[0]
    rows.append(
        {
            "source": "historical_direct",
            "partition": "test",
            "space": "raw",
            "protocol": "value_disjoint_neural_linear",
            "held_pair": None,
            "n": 100,
            "balanced_accuracy": float(historical["test_accuracy"]),
        }
    )

    def append_from_csv(
        path: Path,
        *,
        source: str,
        filters: Mapping[str, Any],
        partition_column: str | None = None,
    ) -> None:
        if not path.exists():
            return
        frame = pd.read_csv(path)
        for column, value in filters.items():
            frame = frame.loc[frame[column].eq(value)]
        for row in frame.to_dict(orient="records"):
            rows.append(
                {
                    "source": source,
                    "partition": (
                        row.get(partition_column)
                        if partition_column is not None
                        else row.get("partition", row.get("evaluation"))
                    ),
                    "space": row.get("space", "raw"),
                    "protocol": row.get("protocol", "direct"),
                    "held_pair": row.get("held_pair"),
                    "n": row.get("n"),
                    "balanced_accuracy": row.get(
                        "balanced_accuracy", row.get("test_balanced_accuracy")
                    ),
                }
            )

    append_from_csv(
        args.bag_output_dir / "probe_metrics.csv",
        source="occurrence_ridge",
        filters={"training_regime": "occurrence", "property_id": "parity"},
    )
    append_from_csv(
        args.whitened_output_dir / "probe_metrics.csv",
        source="value_disjoint_ridge",
        filters={"property_id": "parity"},
    )
    append_from_csv(
        args.factorial_output_dir / "factorization_summary.csv",
        source="factorial",
        filters={"kind": "registered_parity"},
    )
    append_from_csv(
        args.factorial_output_dir / "fold_metrics.csv",
        source="factorial_fold",
        filters={"kind": "registered_parity"},
    )
    for regime in ("occurrence_heldout", "value_disjoint"):
        path = args.nullspace_output_dir / regime / "parity" / "iterations.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        first = frame.loc[frame["accepted"].astype(bool)].iloc[0]
        rows.append(
            {
                "source": f"nullspace_{regime}_first",
                "partition": "test",
                "space": "raw",
                "protocol": "first_greedy_direction",
                "held_pair": None,
                "n": None,
                "balanced_accuracy": first.get("test_balanced_accuracy"),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    done_path = output_dir / "DONE"
    if done_path.exists() and not args.force:
        print(done_path)
        return
    set_global_seed(int(args.seed), deterministic=True)
    inputs = _load_inputs(args)
    strict = inputs["strict"]
    occurrence_values = inputs["occurrence_values"]
    occurrence_activations = inputs["occurrence_activations"]
    directions: dict[str, dict[str, np.ndarray]] = {}
    manifest_rows: list[dict[str, Any]] = []

    old_source = args.transcoder_root / OLD_SOURCE_REL
    old_direction_path = old_source / "selected_and_control_directions.pt"
    old_direction = _load_old_direction(old_direction_path)
    _add_direction(
        directions,
        manifest_rows,
        direction_id="historical_direct",
        family="direct",
        protocol="value_disjoint_neural_linear",
        training_context="original_two_shot_parity_qa",
        decoder=old_direction,
        intervention=old_direction,
        orientation_activations=occurrence_activations,
        orientation_values=occurrence_values,
        space="raw",
    )

    bag_probe_path = args.bag_output_dir / "property_probes.npz"
    with np.load(bag_probe_path, allow_pickle=False) as payload:
        occurrence_direction = np.asarray(
            payload["parity__coefficients"][0], dtype=np.float64
        )
    _add_direction(
        directions,
        manifest_rows,
        direction_id="occurrence_ridge",
        family="direct",
        protocol="occurrence_heldout_ridge",
        training_context="addition_formula_source_number",
        decoder=occurrence_direction,
        intervention=occurrence_direction,
        orientation_activations=occurrence_activations,
        orientation_values=occurrence_values,
        space="raw",
    )

    whitened_bundle = args.whitened_output_dir / "probe_bundle.npz"
    with np.load(whitened_bundle, allow_pickle=False) as payload:
        raw_coefficient = np.asarray(
            payload["raw__parity__coefficients"], dtype=np.float64
        )
        raw_direction = raw_coefficient[0] - raw_coefficient[1]
        white_coefficient = np.asarray(
            payload["whitened__parity__coefficients"], dtype=np.float64
        )
        white_direction = white_coefficient[0] - white_coefficient[1]
        basis = np.asarray(payload["whitening_basis"], dtype=np.float64)
        scales = np.asarray(payload["whitening_scales"], dtype=np.float64)
    _add_direction(
        directions,
        manifest_rows,
        direction_id="value_disjoint_raw_ridge",
        family="direct",
        protocol="value_disjoint_ridge",
        training_context="addition_formula_source_number",
        decoder=raw_direction,
        intervention=raw_direction,
        orientation_activations=strict["train"][1],
        orientation_values=strict["train"][0],
        space="raw",
    )
    _add_direction(
        directions,
        manifest_rows,
        direction_id="value_disjoint_whitened_ridge",
        family="direct",
        protocol="value_disjoint_whitened_ridge",
        training_context="addition_formula_source_number",
        decoder=basis @ (white_direction / scales),
        intervention=basis @ (white_direction * scales),
        orientation_activations=strict["train"][1],
        orientation_values=strict["train"][0],
        space="whitened",
    )

    for regime, orientation in (
        ("occurrence_heldout", (occurrence_values, occurrence_activations)),
        ("value_disjoint", strict["train"]),
    ):
        path = args.nullspace_output_dir / regime / "parity" / "directions.npz"
        with np.load(path, allow_pickle=False) as payload:
            first = np.asarray(payload["basis"][:, 0], dtype=np.float64)
        _add_direction(
            directions,
            manifest_rows,
            direction_id=f"nullspace_{regime}_first",
            family="iterative_nullspace",
            protocol="first_greedy_direction",
            training_context="addition_formula_source_number",
            decoder=first,
            intervention=first,
            orientation_activations=orientation[1],
            orientation_values=orientation[0],
            space="raw",
        )

    fit_audit = _fit_factorial_directions(
        directions=directions,
        manifest=manifest_rows,
        occurrence_values=occurrence_values,
        occurrence_activations=occurrence_activations,
        strict_train_activations=strict["train"][1],
        seed=int(args.seed),
    )
    manifest = pd.DataFrame(manifest_rows)
    geometry, pairwise, stability, coverage = _geometry_tables(
        directions, manifest
    )
    predictive = _collect_predictive_context(args)

    bundle: dict[str, np.ndarray] = {}
    for direction_id, payload in directions.items():
        bundle[f"{direction_id}__decoder"] = np.asarray(
            payload["decoder"], dtype=np.float32
        )
        bundle[f"{direction_id}__intervention"] = np.asarray(
            payload["intervention"], dtype=np.float32
        )
    np.savez_compressed(output_dir / "direction_bundle.npz", **bundle)
    atomic_write_csv(output_dir / "direction_manifest.csv", manifest)
    atomic_write_csv(output_dir / "reference_geometry.csv", geometry)
    atomic_write_csv(output_dir / "pairwise_geometry.csv", pairwise)
    atomic_write_csv(output_dir / "factorial_fold_stability.csv", stability)
    atomic_write_csv(output_dir / "factorial_span_coverage.csv", coverage)
    atomic_write_csv(output_dir / "fit_audit.csv", pd.DataFrame(fit_audit))
    atomic_write_csv(output_dir / "predictive_context.csv", predictive)
    metadata = {
        "seed": int(args.seed),
        "model_id": "meta-llama/Llama-3.1-8B",
        "model_revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "hidden_width": 4096,
        "accurate_module_index": 10,
        "transformer_block_index": 9,
        "hidden_state_index": 10,
        "orientation": "even_positive",
        "design_path": str(DESIGN_PATH.resolve()),
        "input_artifacts": {
            "split": {
                "path": str(inputs["split_path"].resolve()),
                "sha256": _sha256(inputs["split_path"]),
            },
            "bag_probe": {
                "path": str(bag_probe_path.resolve()),
                "sha256": _sha256(bag_probe_path),
            },
            "whitened_probe": {
                "path": str(whitened_bundle.resolve()),
                "sha256": _sha256(whitened_bundle),
            },
            "historical_direction": {
                "path": str(old_direction_path.resolve()),
                "sha256": _sha256(old_direction_path),
            },
        },
        "factorial_fit": {
            "C": LOGISTIC_C,
            "max_iter": LOGISTIC_MAX_ITER,
            "tol": LOGISTIC_TOL,
            "whitening_relative_tolerance": WHITENING_RTOL,
        },
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    atomic_write_text(done_path, "complete\n")
    print(done_path)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
