#!/usr/bin/env python3
"""Run attribution-ordered, label-predicted Figure 1 property steering.

The primary edit is the exact Figure 1 PLS x-weight1 magnitude edit plus a
cumulative sum of competitive-deflation owner-block coordinate predictions.
The owner-block mappings are fitted using property labels and activation
coordinates only; no causal outcomes or target activations enter the primary
edit construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    load_source_config,
    load_split_activations,
)
from scripts.run_figure1_best_in_span_causal_ceiling import (  # noqa: E402
    _batch_hidden,
    _edit_hidden,
    _iter_slices,
    _numeric_target_indices,
    _optimize_arbitrary_batch,
    _suffix_context,
    _suffix_logits,
)
from scripts.run_property_subspace_representation_synthesis import (  # noqa: E402
    ALL_PROPERTIES,
)
from scripts.run_source_value_probe_manifold_steering import (  # noqa: E402
    _resolve_model_entry,
)
from src.best_in_span import joint_constraint_diagnostics  # noqa: E402
from src.gradient_steering import freeze_model_parameters  # noqa: E402
from src.magnitude_intervention import fit_magnitude_probe  # noqa: E402
from src.model_loader import load_causal_lm, load_tokenizer  # noqa: E402
from src.number_property_probes import property_labels  # noqa: E402
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.owner_block_steering import (  # noqa: E402
    DEFAULT_AUXILIARY_SCALES,
    DEFAULT_RIDGE_ALPHAS,
    PRIMARY_PROPERTY_ORDER,
    build_prefix_specs,
    cumulative_auxiliary_coefficients,
    fit_owner_block_models,
    orthonormal_prefix_coordinate_basis,
    rescale_rows_to_norm,
    select_scale_by_mae,
)
from src.property_attribution import direction_owners_from_diagnostics  # noqa: E402
from src.property_subspace_synthesis import fit_property_feature_encoder  # noqa: E402
from src.source_value_steering import summarize_steering_metrics  # noqa: E402
from src.steering_sanity import evaluate_logits_against_shifted_targets  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    load_yaml,
    runtime_metadata,
    set_global_seed,
)


SOURCE_RUN = ROOT / "out/runs/source_value_probe_manifold_steering__9bbad17c01ee__r0"
PCA64_RUN = ROOT / "out/runs/source_value_probe_manifold_steering__d10de45c376a__r0"
COMPETITIVE_RUN = ROOT / "out/property_probe_training_scheme_comparison/llama31_8b_layer10_v1"
BEST_SPAN_RUN = ROOT / "out/figure1_best_in_span_causal_ceiling/llama31_8b_layer10_v1"
DEFAULT_OUTPUT = ROOT / "out/figure1_owner_block_prefix_steering/llama31_8b_layer10_v1"
DEFAULT_SEED = 20260728
DISCOVERY_ROWS = 500
RANDOM_CONTROL_DRAWS = 5
MILESTONE_PREFIXES = (1, 2, 3, 4, 18)
REQUIRED_CUBLAS = ":4096:8"
PANEL_KEY_COLUMNS = (
    "baseline_row_idx",
    "target_slot",
    "source_value",
    "target_value",
    "token_idx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Owner-block-Shapley-ordered Figure 1 property steering."
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--pca64-run-dir", type=Path, default=PCA64_RUN)
    parser.add_argument("--competitive-run-dir", type=Path, default=COMPETITIVE_RUN)
    parser.add_argument("--best-span-run-dir", type=Path, default=BEST_SPAN_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "calibrate", "evaluate", "ceiling", "analyze"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--optimizer-steps", type=int, default=60)
    parser.add_argument("--optimizer-lr", type=float, default=0.08)
    parser.add_argument(
        "--optimizer-restarts",
        choices=("natural", "zero,natural"),
        default="zero,natural",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _panel_hash(panel: pd.DataFrame) -> str:
    payload = panel.loc[:, PANEL_KEY_COLUMNS].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_condition_rows(
    run_dir: Path,
    *,
    architecture: str,
    method: str,
    delta_kind: str,
) -> pd.DataFrame:
    details = pd.read_csv(run_dir / "details.csv", low_memory=False)
    selected = details.loc[
        details["architecture"].astype(str).eq(str(architecture))
        & details["method"].astype(str).eq(str(method))
        & details["delta_kind"].astype(str).eq(str(delta_kind))
    ].copy()
    selected = selected.sort_values(
        ["target_slot", "source_value", "target_value"]
    ).reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"No rows for {architecture}/{method}/{delta_kind} in {run_dir}."
        )
    return selected


def load_panel_and_references(
    *,
    source_run_dir: Path,
    pca64_run_dir: Path,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _load_condition_rows(
        source_run_dir,
        architecture="pls_x_weight1",
        method="fixed",
        delta_kind="value_code",
    )
    counterfactual = _load_condition_rows(
        source_run_dir,
        architecture="full_counterfactual",
        method="counterfactual_patch",
        delta_kind="exact_patch",
    )
    centroid = _load_condition_rows(
        pca64_run_dir,
        architecture="causalab_pca64_1d_manifold",
        method="causalab_pca64_centroid_endpoint",
        delta_kind="pca64_centroid_endpoint",
    )
    if max_rows is not None:
        count = int(max_rows)
        panel = panel.head(count).copy()
        counterfactual = counterfactual.head(count).copy()
        centroid = centroid.head(count).copy()
    for name, reference in (
        ("counterfactual", counterfactual),
        ("centroid", centroid),
    ):
        if len(reference) != len(panel):
            raise ValueError(f"{name} row count does not match the PLS x-weight panel.")
        if not panel.loc[:, PANEL_KEY_COLUMNS].reset_index(drop=True).equals(
            reference.loc[:, PANEL_KEY_COLUMNS].reset_index(drop=True)
        ):
            raise ValueError(f"{name} rows are not aligned with the PLS x-weight panel.")
    references = pd.concat(
        [
            panel.assign(reference_condition="published_pls_x_weight1"),
            centroid.assign(reference_condition="published_pca64_centroid"),
            counterfactual.assign(reference_condition="published_counterfactual"),
        ],
        ignore_index=True,
    )
    return panel, references


def _load_competitive_geometry(
    *,
    competitive_run_dir: Path,
    best_span_run_dir: Path,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(best_span_run_dir / "geometry.npz") as payload:
        basis = np.asarray(payload["competitive_basis"], dtype=np.float64)
        source = np.asarray(payload["source_activations"], dtype=np.float64)
        target = np.asarray(payload["target_activations"], dtype=np.float64)
        exact_patch = np.asarray(payload["exact_patch"], dtype=np.float64)
    orthogonality_error = float(
        np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))
    )
    if orthogonality_error > 2e-6:
        raise ValueError(
            f"Competitive basis is not orthonormal (max error {orthogonality_error:.3e})."
        )
    diagnostics = pd.read_csv(competitive_run_dir / "basis_training_diagnostics.csv")
    owners = direction_owners_from_diagnostics(
        diagnostics,
        split="split_00",
        expected_rank=basis.shape[1],
    )
    return basis, owners, source, target, exact_patch


def _row_norm_matched_random(
    *,
    rng: np.random.Generator,
    rows: int,
    rank: int,
    target_norms: np.ndarray,
) -> np.ndarray:
    values = rng.normal(size=(int(rows), int(rank)))
    return rescale_rows_to_norm(values, target_norms)


def _random_control_auxiliary(
    *,
    rng: np.random.Generator,
    active_properties: Sequence[str],
    active_owner_counts: Mapping[str, int],
    encoder,
    train_values: np.ndarray,
    train_activations: np.ndarray,
    dev_values: np.ndarray,
    dev_activations: np.ndarray,
    source_values: np.ndarray,
    target_values: np.ndarray,
    competitive_basis: np.ndarray,
    magnitude_coordinate_direction: np.ndarray,
    correct_norms: np.ndarray,
) -> tuple[np.ndarray, float]:
    random_owners = tuple(
        property_id
        for property_id in active_properties
        for _ in range(int(active_owner_counts[property_id]))
    )
    rank = len(random_owners)
    gaussian = rng.normal(size=(competitive_basis.shape[1], rank))
    random_coordinates, _ = np.linalg.qr(gaussian, mode="reduced")
    random_hidden_basis = competitive_basis @ random_coordinates
    models, _ = fit_owner_block_models(
        encoder=encoder,
        owners=random_owners,
        basis=random_hidden_basis,
        train_values=train_values,
        train_activations=train_activations,
        dev_values=dev_values,
        dev_activations=dev_activations,
        property_ids=active_properties,
        alphas=DEFAULT_RIDGE_ALPHAS,
    )
    local = np.zeros((len(source_values), rank), dtype=np.float64)
    for property_id in active_properties:
        local += models[property_id].delta_coefficients(
            encoder,
            source_values,
            target_values,
            ambient_rank=rank,
        )
    ambient = local @ random_coordinates.T
    fallback_local = _row_norm_matched_random(
        rng=rng,
        rows=len(source_values),
        rank=rank,
        target_norms=np.ones(len(source_values), dtype=np.float64),
    )
    ambient = rescale_rows_to_norm(
        ambient,
        correct_norms,
        fallback_rows=fallback_local @ random_coordinates.T,
    )
    magnitude = np.asarray(magnitude_coordinate_direction, dtype=np.float64).reshape(-1)
    magnitude /= np.linalg.norm(magnitude)
    magnitude_overlap = float(np.linalg.norm(random_coordinates.T @ magnitude))
    return ambient, magnitude_overlap


def prepare(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    prepared_path = output_dir / "prepared.npz"
    if prepared_path.exists() and not args.force:
        print(prepared_path)
        return
    panel, references = load_panel_and_references(
        source_run_dir=args.source_run_dir,
        pca64_run_dir=args.pca64_run_dir,
        max_rows=args.max_rows,
    )
    if len(panel) <= DISCOVERY_ROWS and args.max_rows is None:
        raise ValueError("The production panel must leave rows outside the discovery subset.")
    cfg, model_entry, resolved = load_source_config(args.source_run_dir)
    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=str(resolved["model_revision"]),
    )
    partitions = load_split_activations(
        split_name="split_00",
        cfg=cfg,
        model_entry=model_entry,
        resolved=resolved,
        tokenizer=tokenizer,
    )
    train_values, train_activations = partitions["train"]
    dev_values, dev_activations = partitions["dev"]
    magnitude_probe = fit_magnitude_probe(
        architecture="pls_x_weight1",
        train_x=np.asarray(train_activations, dtype=np.float64),
        train_values=np.asarray(train_values, dtype=np.int64),
        config=dict(cfg["probe_specs"]["pls_x_weight1"]),
        value_min=int(cfg.get("value_min", 0)),
        value_max=int(cfg.get("value_max", 999)),
    )
    source_values = panel["source_value"].to_numpy(dtype=np.int64)
    target_values = panel["target_value"].to_numpy(dtype=np.int64)
    magnitude_delta = magnitude_probe.intervention_delta(source_values, target_values)
    published_norm = panel["delta_norm"].to_numpy(dtype=np.float64)
    reconstructed_norm = np.linalg.norm(magnitude_delta, axis=1)
    baseline_norm_error = float(np.max(np.abs(reconstructed_norm - published_norm)))
    if baseline_norm_error > 2e-5:
        raise ValueError(
            "Reconstructed PLS x-weight edit does not match the published edit norms "
            f"(max error {baseline_norm_error:.3e})."
        )

    basis, owners, source_activations, target_activations, exact_patch = (
        _load_competitive_geometry(
            competitive_run_dir=args.competitive_run_dir,
            best_span_run_dir=args.best_span_run_dir,
        )
    )
    if len(source_activations) < len(panel):
        raise ValueError("Cached competitive geometry is shorter than the requested panel.")
    source_activations = source_activations[: len(panel)]
    target_activations = target_activations[: len(panel)]
    exact_patch = exact_patch[: len(panel)]
    if not np.allclose(
        target_activations - source_activations,
        exact_patch,
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError("Cached exact patch is inconsistent with source/target activations.")

    magnitude_coefficients = magnitude_delta @ basis
    magnitude_residual = magnitude_delta - magnitude_coefficients @ basis.T
    magnitude_projection_residual = float(
        np.max(np.linalg.norm(magnitude_residual, axis=1))
    )
    if magnitude_projection_residual > 2e-5:
        raise ValueError(
            "The Figure 1 magnitude edit is not contained in the competitive span "
            f"(max residual {magnitude_projection_residual:.3e})."
        )
    unit_delta = magnitude_probe.intervention_delta(
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )[0]
    magnitude_direction = unit_delta / np.linalg.norm(unit_delta)
    magnitude_coordinate_direction = basis.T @ magnitude_direction

    encoder = fit_property_feature_encoder(
        train_values=np.asarray(train_values, dtype=np.int64),
        property_ids=ALL_PROPERTIES,
    )
    models, model_diagnostics = fit_owner_block_models(
        encoder=encoder,
        owners=owners,
        basis=basis,
        train_values=np.asarray(train_values, dtype=np.int64),
        train_activations=np.asarray(train_activations, dtype=np.float64),
        dev_values=np.asarray(dev_values, dtype=np.int64),
        dev_activations=np.asarray(dev_activations, dtype=np.float64),
        property_ids=PRIMARY_PROPERTY_ORDER,
        alphas=DEFAULT_RIDGE_ALPHAS,
    )
    property_coefficients = {
        property_id: model.delta_coefficients(
            encoder,
            source_values,
            target_values,
            ambient_rank=basis.shape[1],
        )
        for property_id, model in models.items()
    }
    specs = build_prefix_specs(owners)
    cumulative = cumulative_auxiliary_coefficients(property_coefficients, specs)
    if not np.array_equal(
        cumulative[specs[-1].condition_id],
        cumulative[specs[-2].condition_id],
    ):
        raise ValueError("The divisible-by-4 dummy prefix changed the auxiliary edit.")

    owner_counts = {
        property_id: sum(owner == property_id for owner in owners)
        for property_id in PRIMARY_PROPERTY_ORDER
    }
    rng = np.random.default_rng(DEFAULT_SEED)
    controls: dict[str, np.ndarray] = {}
    random_control_rows: list[dict[str, object]] = []
    for prefix_index in MILESTONE_PREFIXES:
        spec = specs[prefix_index]
        correct_aux = cumulative[spec.condition_id]
        correct_norms = np.linalg.norm(correct_aux, axis=1)
        shuffled_target = target_values[rng.permutation(len(target_values))]
        shuffled_aux = np.zeros_like(correct_aux)
        for property_id in spec.active_properties:
            shuffled_aux += models[property_id].delta_coefficients(
                encoder,
                source_values,
                shuffled_target,
                ambient_rank=basis.shape[1],
            )
        active_indices = [
            index for index, owner in enumerate(owners) if owner in spec.active_properties
        ]
        fallback = np.zeros_like(correct_aux)
        if active_indices:
            fallback[:, np.asarray(active_indices, dtype=np.int64)] = (
                _row_norm_matched_random(
                    rng=rng,
                    rows=len(panel),
                    rank=len(active_indices),
                    target_norms=np.ones(len(panel), dtype=np.float64),
                )
            )
        controls[f"shuffled__{spec.condition_id}"] = rescale_rows_to_norm(
            shuffled_aux,
            correct_norms,
            fallback_rows=fallback,
        )
        total_norms = np.linalg.norm(magnitude_coefficients + correct_aux, axis=1)
        controls[f"value_norm_matched__{spec.condition_id}"] = rescale_rows_to_norm(
            magnitude_coefficients,
            total_norms,
        )
        for draw in range(RANDOM_CONTROL_DRAWS):
            random_aux, overlap = _random_control_auxiliary(
                rng=rng,
                active_properties=spec.active_properties,
                active_owner_counts=owner_counts,
                encoder=encoder,
                train_values=np.asarray(train_values, dtype=np.int64),
                train_activations=np.asarray(train_activations, dtype=np.float64),
                dev_values=np.asarray(dev_values, dtype=np.int64),
                dev_activations=np.asarray(dev_activations, dtype=np.float64),
                source_values=source_values,
                target_values=target_values,
                competitive_basis=basis,
                magnitude_coordinate_direction=magnitude_coordinate_direction,
                correct_norms=correct_norms,
            )
            name = f"random_rank__{spec.condition_id}__draw{draw:02d}"
            controls[name] = magnitude_coefficients + random_aux
            random_control_rows.append(
                {
                    "condition_id": name,
                    "prefix_index": prefix_index,
                    "draw": draw,
                    "rank": spec.owner_rank_total,
                    "random_overlap_with_figure1_magnitude": overlap,
                }
            )
        controls[f"shuffled__{spec.condition_id}"] = (
            magnitude_coefficients + controls[f"shuffled__{spec.condition_id}"]
        )

    controls["swap_ht__p01_tens_digit"] = (
        magnitude_coefficients + property_coefficients["tens_digit"]
    )

    arrays: dict[str, np.ndarray] = {
        "competitive_basis": basis.astype(np.float32),
        "magnitude_coefficients": magnitude_coefficients.astype(np.float32),
        "magnitude_coordinate_direction": magnitude_coordinate_direction.astype(
            np.float32
        ),
        "source_activations": source_activations.astype(np.float32),
        "target_activations": target_activations.astype(np.float32),
        "exact_patch": exact_patch.astype(np.float32),
        "source_values": source_values,
        "target_values": target_values,
    }
    for property_id, coefficients in property_coefficients.items():
        arrays[f"property__{property_id}"] = coefficients.astype(np.float32)
    for spec in specs:
        arrays[f"auxiliary__{spec.condition_id}"] = cumulative[
            spec.condition_id
        ].astype(np.float32)
        coordinate_basis = orthonormal_prefix_coordinate_basis(
            magnitude_coordinate_direction=magnitude_coordinate_direction,
            owners=owners,
            active_properties=spec.active_properties,
        )
        arrays[f"span__{spec.condition_id}"] = coordinate_basis.astype(np.float32)
    for name, coefficients in controls.items():
        arrays[f"control__{name}"] = coefficients.astype(np.float32)
    _atomic_savez(prepared_path, **arrays)

    prefix_rows = pd.DataFrame(
        [
            {
                "prefix_index": spec.prefix_index,
                "condition_id": spec.condition_id,
                "added_property": spec.added_property,
                "active_properties": ",".join(spec.active_properties),
                "owner_rank_added": spec.owner_rank_added,
                "owner_rank_total": spec.owner_rank_total,
                "nominal_rank": spec.nominal_rank,
                "actual_span_rank": arrays[f"span__{spec.condition_id}"].shape[1],
            }
            for spec in specs
        ]
    )
    if not np.array_equal(
        prefix_rows["nominal_rank"].to_numpy(dtype=np.int64),
        prefix_rows["actual_span_rank"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Nominal and actual cumulative ranks do not match.")
    atomic_write_csv(output_dir / "panel.csv", panel)
    atomic_write_csv(output_dir / "published_references.csv", references)
    atomic_write_csv(output_dir / "prefixes.csv", prefix_rows)
    atomic_write_csv(output_dir / "owner_block_models.csv", pd.DataFrame(model_diagnostics))
    atomic_write_csv(
        output_dir / "random_control_diagnostics.csv",
        pd.DataFrame(random_control_rows),
    )
    atomic_write_json(
        output_dir / "prepare_metadata.json",
        {
            "schema_version": 1,
            "design": {
                "primary_estimand": (
                    "literal Figure 1 PLS x-weight1 edit plus the cumulative sum of "
                    "label-predicted competitive owner-block coordinate differences"
                ),
                "property_order": list(PRIMARY_PROPERTY_ORDER),
                "discovery_rows": min(DISCOVERY_ROWS, len(panel)),
                "confirmatory_rows": max(0, len(panel) - DISCOVERY_ROWS),
                "auxiliary_scale_grid": list(DEFAULT_AUXILIARY_SCALES),
                "ridge_alpha_grid": list(DEFAULT_RIDGE_ALPHAS),
                "random_control_draws": RANDOM_CONTROL_DRAWS,
                "milestone_prefixes": list(MILESTONE_PREFIXES),
            },
            "model": str(model_entry["hf_id"]),
            "model_revision": str(resolved["model_revision"]),
            "layer_idx": int(cfg["intervention_layer_idx"]),
            "panel_hash": _panel_hash(panel),
            "competitive_basis_hash": _array_hash(basis.astype(np.float32)),
            "magnitude_projection_max_residual": magnitude_projection_residual,
            "published_magnitude_norm_max_abs_error": baseline_norm_error,
            "source_run": str(args.source_run_dir.resolve()),
            "pca64_run": str(args.pca64_run_dir.resolve()),
            "competitive_run": str(args.competitive_run_dir.resolve()),
            "best_span_run": str(args.best_span_run_dir.resolve()),
            "runtime": runtime_metadata(),
        },
    )
    print(prepared_path)


def _load_model_context(args: argparse.Namespace):
    cfg, source_model_entry, resolved = load_source_config(args.source_run_dir)
    model_entry = _resolve_model_entry(
        load_yaml(args.models_config),
        str(cfg["model"]),
    )
    if str(model_entry["hf_id"]) != str(source_model_entry["hf_id"]):
        raise ValueError("Resolved model entry does not match the source run.")
    device = str(args.device or model_entry.get("device", "cuda"))
    loaded = load_causal_lm(
        model_name=str(model_entry["hf_id"]),
        device=device,
        dtype=str(model_entry.get("dtype", "float16")),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        attn_implementation=model_entry.get("attn_implementation"),
        revision=str(resolved["model_revision"]),
    )
    freeze_model_parameters(loaded.model)
    return cfg, model_entry, resolved, loaded


def _load_hidden(best_span_run_dir: Path, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(best_span_run_dir / "source_hidden_sequences.npz") as payload:
        hidden = np.asarray(payload["hidden"])
        lengths = np.asarray(payload["lengths"], dtype=np.int64)
        panel_to_unique = np.asarray(payload["panel_to_unique"], dtype=np.int64)
    if len(panel_to_unique) < rows:
        raise ValueError("The cached hidden-state panel is too short.")
    return hidden, lengths, panel_to_unique[:rows]


def _condition_path(output_dir: Path, condition_id: str) -> Path:
    return output_dir / "condition_details" / f"{condition_id}.csv"


def _copy_condition_detail_text(
    source_path: Path,
    destination_path: Path,
    *,
    source_condition: str,
    destination_condition: str,
) -> pd.DataFrame:
    """Copy a dummy condition without round-tripping numeric CSV fields."""
    text = source_path.read_text(encoding="utf-8")
    replaced = text.replace(source_condition, destination_condition)
    if replaced == text:
        raise ValueError(
            f"Condition {source_condition!r} was not found in {source_path}."
        )
    ensure_dir(destination_path.parent)
    atomic_write_text(destination_path, replaced)
    return pd.read_csv(destination_path, low_memory=False)


def _evaluate_coefficients(
    *,
    condition_id: str,
    panel: pd.DataFrame,
    panel_indices: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    fisher_full: np.ndarray,
    hidden: np.ndarray,
    lengths: np.ndarray,
    panel_to_unique: np.ndarray,
    model,
    tokenizer,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
) -> pd.DataFrame:
    selected_panel = panel.iloc[panel_indices].reset_index(drop=True)
    selected_coefficients = np.asarray(coefficients, dtype=np.float64)[panel_indices]
    selected_mapping = panel_to_unique[panel_indices]
    delta_norms = np.linalg.norm(selected_coefficients, axis=1)
    fisher = np.asarray(fisher_full, dtype=np.float64)[panel_indices]
    fisher_quadratic = np.einsum(
        "bi,bij,bj->b",
        selected_coefficients,
        fisher,
        selected_coefficients,
        optimize=True,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    parts: list[pd.DataFrame] = []
    with _suffix_context(model, int(layer_idx)), torch.no_grad():
        for row_slice in _iter_slices(len(selected_panel), int(batch_size)):
            states, mask = _batch_hidden(
                hidden,
                lengths,
                selected_mapping,
                row_slice,
                device=device,
                dtype=dtype,
            )
            token_idxs = torch.as_tensor(
                selected_panel["token_idx"].to_numpy(dtype=np.int64, copy=True)[
                    row_slice
                ],
                dtype=torch.long,
                device=device,
            )
            delta = torch.as_tensor(
                selected_coefficients[row_slice] @ basis.T,
                dtype=dtype,
                device=device,
            )
            logits = _suffix_logits(
                model=model,
                hidden_states=_edit_hidden(states, token_idxs, delta),
                attention_mask=mask,
                logit_token_ids=None,
            )
            result = evaluate_logits_against_shifted_targets(
                tokenizer=tokenizer,
                valid_df=selected_panel.iloc[row_slice].copy(),
                logits=logits.float().cpu().numpy(),
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
            )
            result["condition"] = condition_id
            result["panel_row_idx"] = panel_indices[row_slice]
            result["delta_norm"] = delta_norms[row_slice]
            result["source_fisher_quadratic"] = fisher_quadratic[row_slice]
            parts.append(result)
    return pd.concat(parts, ignore_index=True)


def _requested_conditions(raw: str, available: Sequence[str]) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return tuple(available)
    selected = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return selected


def _load_evaluation_state(args: argparse.Namespace):
    output_dir = args.output_dir
    panel = pd.read_csv(output_dir / "panel.csv", low_memory=False)
    prefixes = pd.read_csv(output_dir / "prefixes.csv")
    prepared = np.load(output_dir / "prepared.npz")
    basis = np.asarray(prepared["competitive_basis"], dtype=np.float64)
    with np.load(args.best_span_run_dir / "local_metrics.npz") as metrics:
        fisher = np.asarray(metrics["fisher__competitive"], dtype=np.float64)[: len(panel)]
    hidden, lengths, panel_to_unique = _load_hidden(args.best_span_run_dir, len(panel))
    cfg, model_entry, resolved, loaded = _load_model_context(args)
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(loaded.tokenizer)
    return (
        panel,
        prefixes,
        prepared,
        basis,
        fisher,
        hidden,
        lengths,
        panel_to_unique,
        cfg,
        model_entry,
        resolved,
        loaded,
        tracked_values,
        tracked_token_ids,
    )


def calibrate(args: argparse.Namespace) -> None:
    (
        panel,
        prefixes,
        prepared,
        basis,
        fisher,
        hidden,
        lengths,
        panel_to_unique,
        cfg,
        _,
        _,
        loaded,
        tracked_values,
        tracked_token_ids,
    ) = _load_evaluation_state(args)
    discovery_count = min(DISCOVERY_ROWS, len(panel))
    discovery_indices = np.arange(discovery_count, dtype=np.int64)
    available = prefixes.loc[prefixes["prefix_index"] > 0, "condition_id"].tolist()
    selected = _requested_conditions(args.conditions, available)
    scale_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for condition_id in selected:
        prefix_index = int(
            prefixes.loc[prefixes["condition_id"].eq(condition_id), "prefix_index"].iloc[0]
        )
        base = np.asarray(prepared["magnitude_coefficients"], dtype=np.float64)
        auxiliary = np.asarray(
            prepared[f"auxiliary__{condition_id}"], dtype=np.float64
        )
        scale_to_error: dict[float, np.ndarray] = {}
        for scale in DEFAULT_AUXILIARY_SCALES:
            candidate_id = f"calibration__{condition_id}__scale{str(scale).replace('.', 'p')}"
            detail_path = _condition_path(args.output_dir, candidate_id)
            if detail_path.exists() and not args.force:
                detail = pd.read_csv(detail_path, low_memory=False)
            elif condition_id == "p19_divisible_by_4":
                previous_id = (
                    "calibration__p18_residue_mod_3__scale"
                    f"{str(scale).replace('.', 'p')}"
                )
                previous_path = _condition_path(args.output_dir, previous_id)
                if not previous_path.exists():
                    raise FileNotFoundError(
                        f"Run {previous_id} before the analytic P19 calibration."
                    )
                detail = _copy_condition_detail_text(
                    previous_path,
                    detail_path,
                    source_condition=previous_id,
                    destination_condition=candidate_id,
                )
            else:
                detail = _evaluate_coefficients(
                    condition_id=candidate_id,
                    panel=panel,
                    panel_indices=discovery_indices,
                    coefficients=base + float(scale) * auxiliary,
                    basis=basis,
                    fisher_full=fisher,
                    hidden=hidden,
                    lengths=lengths,
                    panel_to_unique=panel_to_unique,
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                    layer_idx=int(cfg["intervention_layer_idx"]),
                    batch_size=int(args.batch_size),
                )
                ensure_dir(detail_path.parent)
                atomic_write_csv(detail_path, detail)
            scale_to_error[float(scale)] = np.abs(
                pd.to_numeric(detail["target_error"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            )
        selected_scale, candidates = select_scale_by_mae(scale_to_error)
        for candidate in candidates:
            scale_rows.append(
                {
                    "prefix_index": prefix_index,
                    "condition_id": condition_id,
                    **candidate,
                }
            )
        selection_rows.append(
            {
                "prefix_index": prefix_index,
                "condition_id": condition_id,
                "selected_scale": selected_scale,
                "selection_subset": f"panel_rows_0_{discovery_count - 1}",
                "selection_metric": "mean_abs_target_error",
            }
        )
        print(f"[calibrate] {condition_id}: {selected_scale}", flush=True)
    existing_scales = (
        pd.read_csv(args.output_dir / "scale_candidates.csv")
        if (args.output_dir / "scale_candidates.csv").exists() and not args.force
        else pd.DataFrame()
    )
    existing_selected = (
        pd.read_csv(args.output_dir / "selected_scales.csv")
        if (args.output_dir / "selected_scales.csv").exists() and not args.force
        else pd.DataFrame()
    )
    combined_scales = pd.concat([existing_scales, pd.DataFrame(scale_rows)], ignore_index=True)
    combined_scales = combined_scales.drop_duplicates(
        ["condition_id", "scale"], keep="last"
    ).sort_values(["prefix_index", "scale"])
    combined_selected = pd.concat(
        [existing_selected, pd.DataFrame(selection_rows)], ignore_index=True
    )
    combined_selected = combined_selected.drop_duplicates(
        ["condition_id"], keep="last"
    ).sort_values("prefix_index")
    atomic_write_csv(args.output_dir / "scale_candidates.csv", combined_scales)
    atomic_write_csv(args.output_dir / "selected_scales.csv", combined_selected)
    del loaded
    torch.cuda.empty_cache()


def evaluate(args: argparse.Namespace) -> None:
    (
        panel,
        prefixes,
        prepared,
        basis,
        fisher,
        hidden,
        lengths,
        panel_to_unique,
        cfg,
        _,
        _,
        loaded,
        tracked_values,
        tracked_token_ids,
    ) = _load_evaluation_state(args)
    selected_scales = pd.read_csv(args.output_dir / "selected_scales.csv")
    scale_by_condition = dict(
        zip(
            selected_scales["condition_id"].astype(str),
            selected_scales["selected_scale"].astype(float),
            strict=True,
        )
    )
    base = np.asarray(prepared["magnitude_coefficients"], dtype=np.float64)
    condition_coefficients: dict[str, np.ndarray] = {}
    for row in prefixes.itertuples(index=False):
        auxiliary = np.asarray(
            prepared[f"auxiliary__{row.condition_id}"], dtype=np.float64
        )
        condition_coefficients[f"natural__{row.condition_id}"] = base + auxiliary
        if int(row.prefix_index) == 0:
            condition_coefficients[f"calibrated__{row.condition_id}"] = base.copy()
        elif str(row.condition_id) in scale_by_condition:
            condition_coefficients[f"calibrated__{row.condition_id}"] = (
                base + scale_by_condition[str(row.condition_id)] * auxiliary
            )
    for key in prepared.files:
        if key.startswith("control__"):
            condition_coefficients[key.removeprefix("control__")] = np.asarray(
                prepared[key], dtype=np.float64
            )
    available = tuple(condition_coefficients)
    selected = _requested_conditions(args.conditions, available)
    panel_indices = np.arange(len(panel), dtype=np.int64)
    for condition_id in selected:
        detail_path = _condition_path(args.output_dir, condition_id)
        if detail_path.exists() and not args.force:
            continue
        if condition_id in {
            "natural__p00_magnitude",
            "calibrated__p00_magnitude",
        }:
            references = pd.read_csv(
                args.output_dir / "published_references.csv", low_memory=False
            )
            detail = references.loc[
                references["reference_condition"].eq(
                    "published_pls_x_weight1"
                )
            ].copy()
            detail = detail.drop(columns=["reference_condition"]).reset_index(
                drop=True
            )
            if len(detail) != len(panel):
                raise ValueError("The published P0 reference panel is incomplete.")
            for column in PANEL_KEY_COLUMNS:
                if not np.array_equal(
                    pd.to_numeric(detail[column], errors="coerce").to_numpy(),
                    pd.to_numeric(panel[column], errors="coerce").to_numpy(),
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"The published P0 reference is misaligned on {column}."
                    )
            coefficients = condition_coefficients[condition_id]
            detail["condition"] = condition_id
            detail["panel_row_idx"] = panel_indices
            detail["source_fisher_quadratic"] = np.einsum(
                "bi,bij,bj->b",
                coefficients,
                fisher,
                coefficients,
                optimize=True,
            )
            ensure_dir(detail_path.parent)
            atomic_write_csv(detail_path, detail)
            print(f"[evaluate] {condition_id} (published baseline)", flush=True)
            continue
        if condition_id.startswith("calibrated__"):
            prefix_condition = condition_id.split("__", 1)[1]
            if (
                prefix_condition in scale_by_condition
                and scale_by_condition[prefix_condition] == 0.0
            ):
                previous_id = "calibrated__p00_magnitude"
                previous_path = _condition_path(args.output_dir, previous_id)
                if not previous_path.exists():
                    raise FileNotFoundError(
                        "Run the calibrated P0 baseline before zero-scale prefixes."
                    )
                _copy_condition_detail_text(
                    previous_path,
                    detail_path,
                    source_condition=previous_id,
                    destination_condition=condition_id,
                )
                print(f"[evaluate] {condition_id} (zero-scale copy)", flush=True)
                continue
        if condition_id in {
            "natural__p19_divisible_by_4",
            "calibrated__p19_divisible_by_4",
        }:
            variant = condition_id.split("__", 1)[0]
            previous_id = f"{variant}__p18_residue_mod_3"
            previous_path = _condition_path(args.output_dir, previous_id)
            if not previous_path.exists():
                raise FileNotFoundError(
                    f"Run {previous_id} before the analytic P19 dummy condition."
                )
            if variant == "calibrated":
                if (
                    scale_by_condition["p18_residue_mod_3"]
                    != scale_by_condition["p19_divisible_by_4"]
                ):
                    raise ValueError("P18 and P19 selected different auxiliary scales.")
            _copy_condition_detail_text(
                previous_path,
                detail_path,
                source_condition=previous_id,
                destination_condition=condition_id,
            )
            print(f"[evaluate] {condition_id} (analytic dummy)", flush=True)
            continue
        detail = _evaluate_coefficients(
            condition_id=condition_id,
            panel=panel,
            panel_indices=panel_indices,
            coefficients=condition_coefficients[condition_id],
            basis=basis,
            fisher_full=fisher,
            hidden=hidden,
            lengths=lengths,
            panel_to_unique=panel_to_unique,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            layer_idx=int(cfg["intervention_layer_idx"]),
            batch_size=int(args.batch_size),
        )
        ensure_dir(detail_path.parent)
        atomic_write_csv(detail_path, detail)
        print(f"[evaluate] {condition_id}", flush=True)
    del loaded
    torch.cuda.empty_cache()


def ceiling(args: argparse.Namespace) -> None:
    (
        panel,
        prefixes,
        prepared,
        competitive_basis,
        fisher_full,
        hidden,
        lengths,
        panel_to_unique,
        cfg,
        _,
        _,
        loaded,
        tracked_values,
        tracked_token_ids,
    ) = _load_evaluation_state(args)
    with np.load(args.best_span_run_dir / "local_metrics.npz") as metrics:
        l2_budget = np.asarray(
            metrics["reference_l2__exact_patch"], dtype=np.float32
        )[: len(panel)]
        fisher_budget = np.asarray(
            metrics["reference_fisher_q__exact_patch"], dtype=np.float32
        )[: len(panel)]
    exact_patch = np.asarray(prepared["exact_patch"], dtype=np.float64)
    available = prefixes["condition_id"].astype(str).tolist()
    selected = _requested_conditions(args.conditions, available)
    device = next(loaded.model.parameters()).device
    dtype = next(loaded.model.parameters()).dtype
    tracked = torch.as_tensor(
        tracked_token_ids, dtype=torch.long, device=device
    )
    target_indices_all = _numeric_target_indices(panel, tracked_token_ids)
    for condition_id in selected:
        output_path = args.output_dir / "ceiling_coefficients" / f"{condition_id}.npz"
        detail_path = _condition_path(args.output_dir, f"ceiling__{condition_id}")
        if output_path.exists() and detail_path.exists() and not args.force:
            continue
        coordinate_basis = np.asarray(
            prepared[f"span__{condition_id}"], dtype=np.float64
        )
        hidden_basis = competitive_basis @ coordinate_basis
        if condition_id == "p19_divisible_by_4":
            previous_path = (
                args.output_dir
                / "ceiling_coefficients"
                / "p18_residue_mod_3.npz"
            )
            if not previous_path.exists():
                raise FileNotFoundError(
                    "Run the P18 ceiling before the analytic P19 dummy ceiling."
                )
            with np.load(previous_path) as previous:
                previous_coordinate_basis = np.asarray(
                    previous["coordinate_basis"], dtype=np.float64
                )
                if not np.array_equal(
                    previous_coordinate_basis.astype(np.float32),
                    coordinate_basis.astype(np.float32),
                ):
                    raise ValueError("P18 and P19 ceiling coordinate bases differ.")
                coefficients = np.asarray(
                    previous["coefficients"], dtype=np.float32
                )
                scores = np.asarray(previous["scores"], dtype=np.float32)
            ensure_dir(output_path.parent)
            _atomic_savez(
                output_path,
                coefficients=coefficients,
                scores=scores,
                coordinate_basis=coordinate_basis.astype(np.float32),
                hidden_basis=hidden_basis.astype(np.float32),
            )
            previous_detail_path = _condition_path(
                args.output_dir, "ceiling__p18_residue_mod_3"
            )
            if not previous_detail_path.exists():
                raise FileNotFoundError(
                    "The P18 ceiling detail is required for the P19 dummy audit."
                )
            _copy_condition_detail_text(
                previous_detail_path,
                detail_path,
                source_condition="ceiling__p18_residue_mod_3",
                destination_condition=f"ceiling__{condition_id}",
            )
            previous_diagnostics_path = (
                args.output_dir
                / "ceiling_coefficients"
                / "p18_residue_mod_3.csv"
            )
            atomic_write_text(
                args.output_dir / "ceiling_coefficients" / f"{condition_id}.csv",
                previous_diagnostics_path.read_text(encoding="utf-8"),
            )
            print(f"[ceiling] {condition_id} (analytic dummy)", flush=True)
            continue
        natural = exact_patch @ hidden_basis
        transformed_fisher = np.einsum(
            "ia,bij,jc->bac",
            coordinate_basis,
            fisher_full,
            coordinate_basis,
            optimize=True,
        ).astype(np.float32)
        coefficient_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        with _suffix_context(loaded.model, int(cfg["intervention_layer_idx"])):
            basis_tensor = torch.as_tensor(hidden_basis, dtype=dtype, device=device)
            for row_slice in _iter_slices(len(panel), int(args.batch_size)):
                states, mask = _batch_hidden(
                    hidden,
                    lengths,
                    panel_to_unique,
                    row_slice,
                    device=device,
                    dtype=dtype,
                )
                token_idxs = torch.as_tensor(
                    panel["token_idx"].to_numpy(dtype=np.int64, copy=True)[row_slice],
                    dtype=torch.long,
                    device=device,
                )
                target_indices = torch.as_tensor(
                    target_indices_all[row_slice], dtype=torch.long, device=device
                )
                best_coefficients: np.ndarray | None = None
                best_scores: np.ndarray | None = None
                initializations = (
                    ("zero", "natural")
                    if args.optimizer_restarts == "zero,natural"
                    else ("natural",)
                )
                for initialization in initializations:
                    initial = (
                        np.zeros_like(natural[row_slice])
                        if initialization == "zero"
                        else natural[row_slice]
                    )
                    coefficients, scores = _optimize_arbitrary_batch(
                        model=loaded.model,
                        states=states,
                        mask=mask,
                        token_idxs=token_idxs,
                        basis=basis_tensor,
                        initial_coefficients=initial,
                        fisher=transformed_fisher[row_slice],
                        l2_budget=l2_budget[row_slice],
                        fisher_budget=fisher_budget[row_slice],
                        target_indices=target_indices,
                        tracked=tracked,
                        steps=int(args.optimizer_steps),
                        lr=float(args.optimizer_lr),
                    )
                    if best_scores is None:
                        best_coefficients = coefficients
                        best_scores = scores
                    else:
                        improve = scores > best_scores
                        best_scores = np.where(improve, scores, best_scores)
                        best_coefficients = np.where(
                            improve[:, None], coefficients, best_coefficients
                        )
                assert best_coefficients is not None and best_scores is not None
                coefficient_parts.append(best_coefficients.astype(np.float32))
                score_parts.append(best_scores.astype(np.float32))
        coefficients = np.concatenate(coefficient_parts, axis=0)
        scores = np.concatenate(score_parts, axis=0)
        ensure_dir(output_path.parent)
        _atomic_savez(
            output_path,
            coefficients=coefficients,
            scores=scores,
            coordinate_basis=coordinate_basis.astype(np.float32),
            hidden_basis=hidden_basis.astype(np.float32),
        )
        ambient_coefficients = coefficients @ coordinate_basis.T
        detail = _evaluate_coefficients(
            condition_id=f"ceiling__{condition_id}",
            panel=panel,
            panel_indices=np.arange(len(panel), dtype=np.int64),
            coefficients=ambient_coefficients,
            basis=competitive_basis,
            fisher_full=fisher_full,
            hidden=hidden,
            lengths=lengths,
            panel_to_unique=panel_to_unique,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            layer_idx=int(cfg["intervention_layer_idx"]),
            batch_size=int(args.batch_size),
        )
        ensure_dir(detail_path.parent)
        atomic_write_csv(detail_path, detail)
        diagnostics = joint_constraint_diagnostics(
            coefficients,
            transformed_fisher,
            l2_budget=l2_budget,
            fisher_budget=fisher_budget,
        )
        atomic_write_csv(
            args.output_dir / "ceiling_coefficients" / f"{condition_id}.csv",
            pd.DataFrame(
                {
                    "panel_row_idx": np.arange(len(panel), dtype=np.int64),
                    "l2_norm": diagnostics.l2_norm,
                    "fisher_quadratic": diagnostics.fisher_quadratic,
                    "l2_budget": l2_budget,
                    "fisher_budget": fisher_budget,
                    "l2_utilization": diagnostics.l2_utilization,
                    "fisher_utilization": diagnostics.fisher_utilization,
                    "feasible": diagnostics.feasible,
                    "optimized_score": scores,
                }
            ),
        )
        print(f"[ceiling] {condition_id}", flush=True)
    del loaded
    torch.cuda.empty_cache()


def analyze(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    detail_paths = sorted((output_dir / "condition_details").glob("*.csv"))
    if not detail_paths:
        raise FileNotFoundError("No condition detail files are available for analysis.")
    details = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in detail_paths],
        ignore_index=True,
    )
    production = details.loc[
        ~details["condition"].astype(str).str.startswith("calibration__")
    ].copy()
    subset_frames: list[pd.DataFrame] = []
    for subset, frame in (
        ("full", production),
        ("discovery", production.loc[production["panel_row_idx"] < DISCOVERY_ROWS]),
        ("confirmatory", production.loc[production["panel_row_idx"] >= DISCOVERY_ROWS]),
    ):
        if frame.empty:
            continue
        summary = summarize_steering_metrics(frame, group_cols=["condition"])
        fisher_summary = (
            frame.groupby("condition", as_index=False)
            .agg(
                mean_source_fisher_quadratic=(
                    "source_fisher_quadratic",
                    "mean",
                ),
                median_source_fisher_quadratic=(
                    "source_fisher_quadratic",
                    "median",
                ),
            )
        )
        summary = summary.merge(fisher_summary, on="condition", how="left")
        summary.insert(0, "subset", subset)
        subset_frames.append(summary)
    causal_summary = pd.concat(subset_frames, ignore_index=True)
    atomic_write_csv(output_dir / "causal_details.csv", production)
    atomic_write_csv(output_dir / "causal_summary.csv", causal_summary)

    references = pd.read_csv(output_dir / "published_references.csv", low_memory=False)
    references["panel_row_idx"] = references.groupby(
        "reference_condition", sort=False
    ).cumcount()
    reference_frames: list[pd.DataFrame] = []
    for subset, frame in (
        ("full", references),
        ("discovery", references.loc[references["panel_row_idx"] < DISCOVERY_ROWS]),
        (
            "confirmatory",
            references.loc[references["panel_row_idx"] >= DISCOVERY_ROWS],
        ),
    ):
        if frame.empty:
            continue
        renamed = frame.rename(columns={"reference_condition": "condition"})
        summary = summarize_steering_metrics(renamed, group_cols=["condition"])
        summary.insert(0, "subset", subset)
        reference_frames.append(summary)
    reference_summary = pd.concat(reference_frames, ignore_index=True)
    atomic_write_csv(output_dir / "reference_summary.csv", reference_summary)

    prefixes = pd.read_csv(output_dir / "prefixes.csv")
    prefix_by_index = {
        int(row.prefix_index): str(row.condition_id)
        for row in prefixes.itertuples(index=False)
    }
    with np.load(output_dir / "prepared.npz") as prepared:
        magnitude_coordinate_direction = np.asarray(
            prepared["magnitude_coordinate_direction"], dtype=np.float64
        )
    magnitude_coordinate_direction /= np.linalg.norm(
        magnitude_coordinate_direction
    )
    diagnostics = pd.read_csv(
        args.competitive_run_dir / "basis_training_diagnostics.csv"
    )
    owners = direction_owners_from_diagnostics(
        diagnostics,
        split="split_00",
        expected_rank=len(magnitude_coordinate_direction),
    )
    overlap_rows: list[dict[str, object]] = []
    owner_order = ("magnitude",) + tuple(PRIMARY_PROPERTY_ORDER)
    for property_id in owner_order:
        indices = np.asarray(
            [
                index
                for index, owner in enumerate(owners)
                if owner == property_id
            ],
            dtype=np.int64,
        )
        coordinates = magnitude_coordinate_direction[indices]
        overlap_rows.append(
            {
                "property_id": property_id,
                "owner_rank": len(indices),
                "owner_block_projection_norm": float(
                    np.linalg.norm(coordinates)
                ),
                "owner_block_projection_energy": float(
                    np.dot(coordinates, coordinates)
                ),
                "max_abs_direction_cosine": float(
                    np.max(np.abs(coordinates), initial=0.0)
                ),
            }
        )
    overlap_frame = pd.DataFrame(overlap_rows)
    if not np.isclose(
        overlap_frame["owner_block_projection_energy"].sum(),
        1.0,
        atol=2e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "Figure 1 magnitude overlap energy does not partition across owners."
        )
    atomic_write_csv(output_dir / "magnitude_owner_overlap.csv", overlap_frame)
    confirmatory = production.loc[
        production["panel_row_idx"] >= DISCOVERY_ROWS
    ].copy()
    confirmatory_references = references.loc[
        references["panel_row_idx"] >= DISCOVERY_ROWS
    ].copy()
    if confirmatory.empty and len(pd.read_csv(output_dir / "panel.csv")) > DISCOVERY_ROWS:
        raise ValueError("The production details omit the confirmatory panel.")

    def aligned_errors(
        frame: pd.DataFrame,
        condition: str,
    ) -> pd.DataFrame:
        selected = frame.loc[frame["condition"].astype(str).eq(condition)].copy()
        selected["abs_error"] = np.abs(
            pd.to_numeric(selected["target_error"], errors="coerce")
        )
        return selected[
            ["panel_row_idx", "baseline_row_idx", "abs_error"]
        ].sort_values("panel_row_idx")

    def bootstrap_cluster_stat(
        values: np.ndarray,
        clusters: np.ndarray,
        *,
        seed_offset: int,
        draws: int = 4000,
    ) -> tuple[float, float, float, float]:
        frame = pd.DataFrame({"value": values, "cluster": clusters})
        by_cluster = frame.groupby("cluster", sort=True)["value"].mean().to_numpy(
            dtype=np.float64
        )
        estimate = float(np.mean(by_cluster))
        rng = np.random.default_rng(DEFAULT_SEED + int(seed_offset))
        indices = rng.integers(
            0, len(by_cluster), size=(int(draws), len(by_cluster)), endpoint=False
        )
        samples = by_cluster[indices].mean(axis=1)
        return (
            estimate,
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
            float(np.mean(samples > 0.0)),
        )

    increment_rows: list[dict[str, object]] = []
    for variant in ("natural", "calibrated", "ceiling"):
        for prefix_index in range(1, 20):
            before_id = f"{variant}__{prefix_by_index[prefix_index - 1]}"
            after_id = f"{variant}__{prefix_by_index[prefix_index]}"
            before = aligned_errors(confirmatory, before_id)
            after = aligned_errors(confirmatory, after_id)
            if before.empty or after.empty:
                continue
            merged = before.merge(
                after,
                on=["panel_row_idx", "baseline_row_idx"],
                suffixes=("_before", "_after"),
                validate="one_to_one",
            )
            improvement = (
                merged["abs_error_before"] - merged["abs_error_after"]
            ).to_numpy(dtype=np.float64)
            estimate, low, high, positive_probability = bootstrap_cluster_stat(
                improvement,
                merged["baseline_row_idx"].to_numpy(dtype=np.int64),
                seed_offset=1000 * (1 + ("natural", "calibrated", "ceiling").index(variant))
                + prefix_index,
            )
            prefix_row = prefixes.loc[
                prefixes["prefix_index"].eq(prefix_index)
            ].iloc[0]
            rank_added = int(prefix_row["owner_rank_added"])
            increment_rows.append(
                {
                    "subset": "confirmatory",
                    "variant": variant,
                    "prefix_index": prefix_index,
                    "before_condition": before_id,
                    "after_condition": after_id,
                    "added_property": prefix_row["added_property"],
                    "owner_rank_added": rank_added,
                    "mae_improvement": estimate,
                    "mae_improvement_ci_low": low,
                    "mae_improvement_ci_high": high,
                    "bootstrap_probability_improvement": positive_probability,
                    "mae_improvement_per_direction": (
                        estimate / rank_added if rank_added else np.nan
                    ),
                }
            )
    increment_frame = pd.DataFrame(increment_rows)
    atomic_write_csv(output_dir / "paired_prefix_increments.csv", increment_frame)

    centroid_rows = confirmatory_references.loc[
        confirmatory_references["reference_condition"].eq(
            "published_pca64_centroid"
        )
    ].copy()
    centroid_rows["centroid_abs_error"] = np.abs(
        pd.to_numeric(centroid_rows["target_error"], errors="coerce")
    )
    gap_rows: list[dict[str, object]] = []
    for variant in ("natural", "calibrated", "ceiling"):
        base_id = f"{variant}__{prefix_by_index[0]}"
        base = aligned_errors(confirmatory, base_id).rename(
            columns={"abs_error": "base_abs_error"}
        )
        if base.empty:
            continue
        for prefix_index in range(20):
            condition_id = f"{variant}__{prefix_by_index[prefix_index]}"
            current = aligned_errors(confirmatory, condition_id).rename(
                columns={"abs_error": "current_abs_error"}
            )
            if current.empty:
                continue
            merged = (
                base.merge(
                    current,
                    on=["panel_row_idx", "baseline_row_idx"],
                    validate="one_to_one",
                )
                .merge(
                    centroid_rows[
                        ["panel_row_idx", "baseline_row_idx", "centroid_abs_error"]
                    ],
                    on=["panel_row_idx", "baseline_row_idx"],
                    validate="one_to_one",
                )
            )
            grouped = (
                merged.groupby("baseline_row_idx", sort=True)[
                    ["base_abs_error", "current_abs_error", "centroid_abs_error"]
                ]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            denominator = float(
                np.mean(grouped[:, 0]) - np.mean(grouped[:, 2])
            )
            closure = float(
                (np.mean(grouped[:, 0]) - np.mean(grouped[:, 1])) / denominator
            )
            rng = np.random.default_rng(
                DEFAULT_SEED + 5000 + 100 * ("natural", "calibrated", "ceiling").index(variant)
                + prefix_index
            )
            draw_indices = rng.integers(
                0, len(grouped), size=(4000, len(grouped)), endpoint=False
            )
            sampled = grouped[draw_indices].mean(axis=1)
            sampled_closure = np.divide(
                sampled[:, 0] - sampled[:, 1],
                sampled[:, 0] - sampled[:, 2],
                out=np.full(len(sampled), np.nan, dtype=np.float64),
                where=np.abs(sampled[:, 0] - sampled[:, 2]) > 1e-12,
            )
            gap_rows.append(
                {
                    "subset": "confirmatory",
                    "variant": variant,
                    "prefix_index": prefix_index,
                    "condition": condition_id,
                    "centroid_gap_closure": closure,
                    "centroid_gap_closure_ci_low": float(
                        np.nanquantile(sampled_closure, 0.025)
                    ),
                    "centroid_gap_closure_ci_high": float(
                        np.nanquantile(sampled_closure, 0.975)
                    ),
                }
            )
    gap_frame = pd.DataFrame(gap_rows)
    atomic_write_csv(output_dir / "centroid_gap_closure.csv", gap_frame)

    changed_rows: list[dict[str, object]] = []
    panel = pd.read_csv(output_dir / "panel.csv", low_memory=False)
    confirmatory_panel = panel.iloc[DISCOVERY_ROWS:].copy().reset_index(drop=True)
    for prefix_index in range(1, 20):
        property_id = str(
            prefixes.loc[
                prefixes["prefix_index"].eq(prefix_index), "added_property"
            ].iloc[0]
        )
        before = aligned_errors(
            confirmatory, f"natural__{prefix_by_index[prefix_index - 1]}"
        )
        after = aligned_errors(
            confirmatory, f"natural__{prefix_by_index[prefix_index]}"
        )
        if before.empty or after.empty:
            continue
        merged = before.merge(
            after,
            on=["panel_row_idx", "baseline_row_idx"],
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        changed = property_labels(
            property_id,
            confirmatory_panel["source_value"].to_numpy(dtype=np.int64),
        ) != property_labels(
            property_id,
            confirmatory_panel["target_value"].to_numpy(dtype=np.int64),
        )
        changed_by_row = dict(
            zip(
                confirmatory_panel.index.to_numpy(dtype=np.int64) + DISCOVERY_ROWS,
                changed,
                strict=True,
            )
        )
        merged["property_changed"] = merged["panel_row_idx"].map(changed_by_row).astype(bool)
        merged["improvement"] = (
            merged["abs_error_before"] - merged["abs_error_after"]
        )
        changed_mean = float(
            merged.loc[merged["property_changed"], "improvement"].mean()
        )
        unchanged_mean = float(
            merged.loc[~merged["property_changed"], "improvement"].mean()
        )
        changed_rows.append(
            {
                "prefix_index": prefix_index,
                "property_id": property_id,
                "changed_prevalence": float(merged["property_changed"].mean()),
                "mae_improvement_changed": changed_mean,
                "mae_improvement_unchanged": unchanged_mean,
                "specificity_contrast": changed_mean - unchanged_mean,
                "max_auxiliary_norm_when_unchanged": float(
                    np.linalg.norm(
                        np.asarray(
                            np.load(output_dir / "prepared.npz")[
                                f"property__{property_id}"
                            ],
                            dtype=np.float64,
                        )[DISCOVERY_ROWS:][~changed],
                        axis=1,
                    ).max(initial=0.0)
                ),
            }
        )
    changed_frame = pd.DataFrame(changed_rows)
    atomic_write_csv(output_dir / "changed_property_strata.csv", changed_frame)
    if (
        not changed_frame.empty
        and changed_frame["max_auxiliary_norm_when_unchanged"].max() > 1e-10
    ):
        raise ValueError("A property contribution is nonzero on unchanged-property rows.")

    correlation_rows: list[dict[str, object]] = []
    for condition_id, frame in confirmatory.groupby("condition", sort=False):
        requested = pd.to_numeric(frame["delta_value"], errors="coerce")
        realized = pd.to_numeric(frame["predicted_numeric_delta"], errors="coerce")
        valid = requested.notna() & realized.notna()
        correlation_rows.append(
            {
                "condition": condition_id,
                "num_examples": int(valid.sum()),
                "requested_output_pearson": float(
                    requested[valid].corr(realized[valid], method="pearson")
                ),
                "requested_output_spearman": float(
                    requested[valid].corr(realized[valid], method="spearman")
                ),
                "abs_requested_abs_output_spearman": float(
                    requested[valid].abs().corr(
                        realized[valid].abs(), method="spearman"
                    )
                ),
            }
        )
    atomic_write_csv(
        output_dir / "requested_output_correlation.csv",
        pd.DataFrame(correlation_rows),
    )

    published_p0 = references.loc[
        references["reference_condition"].eq("published_pls_x_weight1")
    ].sort_values("panel_row_idx")
    reproduced_p0 = production.loc[
        production["condition"].eq("natural__p00_magnitude")
    ].sort_values("panel_row_idx")
    baseline_audit: dict[str, object] = {
        "published_rows": len(published_p0),
        "reproduced_rows": len(reproduced_p0),
    }
    if len(published_p0) == len(reproduced_p0) and len(published_p0):
        for column in ("predicted_token_id", "predicted_value", "correct"):
            left = pd.to_numeric(published_p0[column], errors="coerce").to_numpy()
            right = pd.to_numeric(reproduced_p0[column], errors="coerce").to_numpy()
            baseline_audit[f"{column}_exact_match"] = bool(
                np.array_equal(left, right, equal_nan=True)
            )
        baseline_audit["target_prob_max_abs_error"] = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(
                        published_p0["target_prob"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    - pd.to_numeric(
                        reproduced_p0["target_prob"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                )
            )
        )
        baseline_audit["delta_norm_max_abs_error"] = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(
                        published_p0["delta_norm"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    - pd.to_numeric(
                        reproduced_p0["delta_norm"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                )
            )
        )
    atomic_write_json(output_dir / "published_baseline_audit.json", baseline_audit)

    dummy_rows: list[dict[str, object]] = []
    for variant in ("natural", "calibrated", "ceiling"):
        p18 = production.loc[
            production["condition"].eq(f"{variant}__p18_residue_mod_3")
        ].sort_values("panel_row_idx")
        p19 = production.loc[
            production["condition"].eq(f"{variant}__p19_divisible_by_4")
        ].sort_values("panel_row_idx")
        if p18.empty or p19.empty:
            continue
        columns = (
            "predicted_token_id",
            "predicted_value",
            "correct",
            "target_prob",
            "delta_norm",
            "source_fisher_quadratic",
        )
        exact = all(
            np.array_equal(
                pd.to_numeric(p18[column], errors="coerce").to_numpy(),
                pd.to_numeric(p19[column], errors="coerce").to_numpy(),
                equal_nan=True,
            )
            for column in columns
        )
        dummy_rows.append(
            {
                "variant": variant,
                "rows": len(p18),
                "bit_identical": exact,
            }
        )
    dummy_frame = pd.DataFrame(
        dummy_rows,
        columns=("variant", "rows", "bit_identical"),
    )
    atomic_write_csv(output_dir / "dummy_prefix_audit.csv", dummy_frame)
    natural_dummy = dummy_frame.loc[dummy_frame["variant"].eq("natural")]
    if not natural_dummy.empty and not bool(natural_dummy["bit_identical"].all()):
        raise ValueError("The natural divisible-by-4 dummy prefix is not bit-identical.")

    atomic_write_text(output_dir / "DONE", "ok\n")


def main() -> None:
    args = parse_args()
    set_global_seed(DEFAULT_SEED)
    if args.stage in {"all", "prepare"}:
        prepare(args)
    gpu_stages = {"calibrate", "evaluate", "ceiling"}
    if args.stage == "all" or args.stage in gpu_stages:
        observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed != REQUIRED_CUBLAS:
            raise RuntimeError(
                f"CUBLAS_WORKSPACE_CONFIG must be {REQUIRED_CUBLAS!r}; got {observed!r}."
            )
    if args.stage in {"all", "calibrate"}:
        calibrate(args)
    if args.stage in {"all", "evaluate"}:
        evaluate(args)
    if args.stage in {"all", "ceiling"}:
        ceiling(args)
    if args.stage in {"all", "analyze"}:
        analyze(args)


if __name__ == "__main__":
    main()
