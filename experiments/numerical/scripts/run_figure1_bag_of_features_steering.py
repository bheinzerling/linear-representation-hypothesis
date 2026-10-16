#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_value_probe_capacity_sweep import (  # noqa: E402
    _collect_partition_activations,
    _load_partition_rows,
)
from scripts.run_source_value_probe_manifold_steering import (  # noqa: E402
    _evaluate_with_delta_rows,
    _resolve_model_entry,
)
from src.gradient_steering import freeze_model_parameters  # noqa: E402
from src.model_loader import load_causal_lm, load_tokenizer  # noqa: E402
from src.number_property_probes import (  # noqa: E402
    PROPERTY_LABELS,
    LinearPropertyProbe,
    fit_property_probe,
    independent_sum_delta,
    joint_minimum_norm_delta,
    norm_matched_random_deltas,
    probe_metrics,
    property_change_count,
    rescale_auxiliary_to_norm,
)
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.source_value_steering import summarize_steering_metrics  # noqa: E402
from src.utils import atomic_write_csv, atomic_write_json, ensure_dir, load_yaml, runtime_metadata, set_global_seed  # noqa: E402


SOURCE_RUN = ROOT / "out/runs/source_value_probe_manifold_steering__9bbad17c01ee__r0"
CENTROID_ARTIFACT = ROOT / "experiment-reports/artifacts/2026-06-16-figure1-allpair-delta-dictionary/value_representations.npz"
DEFAULT_OUTPUT = ROOT / "out/figure1_bag_of_features_steering/llama31_8b_layer10_v1"
PROPERTY_ORDER = (
    "parity",
    "primality",
    "divisible_by_3",
    "divisible_by_5",
    "divisible_by_10",
    "digit_count",
    "units_digit",
    "tens_digit",
    "hundreds_digit",
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SEED = 20260721


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train layer-10 number-property probes and evaluate bag-of-features repair of Figure 1 PLS steering."
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--centroid-artifact", type=Path, default=CENTROID_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--conditions", default="all", help="Comma-separated condition ids or 'all'.")
    parser.add_argument("--skip-causal-eval", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def load_source_config(source_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = json.loads((source_run_dir / "config.json").read_text(encoding="utf-8"))
    return (
        dict(payload["config"]["experiment_config"]),
        dict(payload["config"]["model_entry"]),
        dict(payload["config"]["resolved"]),
    )


def load_split_activations(
    *,
    split_name: str,
    cfg: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    resolved: Mapping[str, Any],
    tokenizer,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = _load_partition_rows(Path(cfg["split_dataset_dir"]), split_name, str(cfg["family"]))
    return _collect_partition_activations(
        rows_by_partition=rows,
        model=None,
        tokenizer=tokenizer,
        layer_idx=int(cfg["intervention_layer_idx"]),
        batch_size=int(cfg.get("probe_batch_size", 64)),
        device="cpu",
        model_alias=str(cfg["model"]),
        model_id=str(model_entry["hf_id"]),
        model_revision=str(resolved["model_revision"]),
        attn_implementation=model_entry.get("attn_implementation"),
        family=str(cfg["family"]),
        activation_cache_root=cfg["activation_cache_root"],
        use_activation_cache=True,
        require_activation_cache_hit=True,
    )


def activations_by_value(partitions: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    hidden_dim = int(next(iter(partitions.values()))[1].shape[1])
    result = np.empty((1000, hidden_dim), dtype=np.float32)
    observed = np.zeros(1000, dtype=bool)
    for values, activations in partitions.values():
        result[np.asarray(values, dtype=np.int64)] = np.asarray(activations, dtype=np.float32)
        observed[np.asarray(values, dtype=np.int64)] = True
    if not np.all(observed):
        raise ValueError(f"Expected all values 0..999, missing {np.flatnonzero(~observed)[:10].tolist()}")
    return result


def save_probe_bundle(path: Path, probes: Mapping[str, LinearPropertyProbe]) -> None:
    payload: dict[str, np.ndarray] = {}
    for property_id, probe in probes.items():
        payload[f"{property_id}__classes"] = probe.classes
        payload[f"{property_id}__coefficients"] = probe.coefficients.astype(np.float32)
        payload[f"{property_id}__intercept"] = probe.intercept
        payload[f"{property_id}__class_score_centroids"] = probe.class_score_centroids
        payload[f"{property_id}__alpha"] = np.asarray([probe.alpha], dtype=np.float64)
    ensure_dir(path.parent)
    np.savez_compressed(path, **payload)


def load_probe_bundle(path: Path) -> dict[str, LinearPropertyProbe]:
    bundle = np.load(path)
    return {
        property_id: LinearPropertyProbe(
            property_id=property_id,
            classes=np.asarray(bundle[f"{property_id}__classes"], dtype=np.int64),
            coefficients=np.asarray(bundle[f"{property_id}__coefficients"], dtype=np.float64),
            intercept=np.asarray(bundle[f"{property_id}__intercept"], dtype=np.float64),
            class_score_centroids=np.asarray(bundle[f"{property_id}__class_score_centroids"], dtype=np.float64),
            alpha=float(bundle[f"{property_id}__alpha"][0]),
        )
        for property_id in PROPERTY_ORDER
    }


def train_or_load_probes(
    *,
    output_dir: Path,
    cfg: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    resolved: Mapping[str, Any],
    tokenizer,
    force: bool,
) -> tuple[dict[str, LinearPropertyProbe], dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    bundle_path = output_dir / "property_probes.npz"
    split0_path = output_dir / "split00_activations.npz"
    if bundle_path.exists() and split0_path.exists() and not force:
        saved = np.load(split0_path)
        split0 = {
            partition: (
                np.asarray(saved[f"{partition}_values"], dtype=np.int64),
                np.asarray(saved[f"{partition}_activations"], dtype=np.float32),
            )
            for partition in ("train", "dev", "test")
        }
        return load_probe_bundle(bundle_path), split0, np.asarray(saved["occurrence_train_by_value"], dtype=np.float32)

    split0 = load_split_activations(
        split_name="split_00", cfg=cfg, model_entry=model_entry, resolved=resolved, tokenizer=tokenizer
    )
    occurrence_sum: np.ndarray | None = None
    for split_index in range(1, 8):
        current = activations_by_value(
            load_split_activations(
                split_name=f"split_{split_index:02d}",
                cfg=cfg,
                model_entry=model_entry,
                resolved=resolved,
                tokenizer=tokenizer,
            )
        ).astype(np.float64)
        occurrence_sum = current if occurrence_sum is None else occurrence_sum + current
    if occurrence_sum is None:
        raise RuntimeError("No occurrence-training activations were loaded.")
    occurrence_train = np.asarray(occurrence_sum / 7.0, dtype=np.float32)
    occurrence_dev = activations_by_value(
        load_split_activations(split_name="split_08", cfg=cfg, model_entry=model_entry, resolved=resolved, tokenizer=tokenizer)
    )
    occurrence_test = activations_by_value(
        load_split_activations(split_name="split_09", cfg=cfg, model_entry=model_entry, resolved=resolved, tokenizer=tokenizer)
    )
    all_values = np.arange(1000, dtype=np.int64)
    probes: dict[str, LinearPropertyProbe] = {}
    metric_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for property_id in PROPERTY_ORDER:
        probe, trace = fit_property_probe(
            property_id=property_id,
            train_activations=occurrence_train,
            train_values=all_values,
            dev_activations=occurrence_dev,
            dev_values=all_values,
            alphas=ALPHAS,
        )
        probes[property_id] = probe
        for row in trace:
            alpha_rows.append({"training_regime": "occurrence", "property_id": property_id, **row})
        for partition, activations in (("dev", occurrence_dev), ("test", occurrence_test)):
            metric_rows.append(
                {
                    "training_regime": "occurrence",
                    "partition": partition,
                    "property_id": property_id,
                    "alpha": probe.alpha,
                    **probe_metrics(probe, activations, all_values),
                }
            )

        strict_probe, strict_trace = fit_property_probe(
            property_id=property_id,
            train_activations=split0["train"][1],
            train_values=split0["train"][0],
            dev_activations=split0["dev"][1],
            dev_values=split0["dev"][0],
            alphas=ALPHAS,
        )
        for row in strict_trace:
            alpha_rows.append({"training_regime": "value_disjoint", "property_id": property_id, **row})
        for partition in ("dev", "test"):
            metric_rows.append(
                {
                    "training_regime": "value_disjoint",
                    "partition": partition,
                    "property_id": property_id,
                    "alpha": strict_probe.alpha,
                    **probe_metrics(strict_probe, split0[partition][1], split0[partition][0]),
                }
            )
    atomic_write_csv(output_dir / "probe_metrics.csv", pd.DataFrame(metric_rows))
    atomic_write_csv(output_dir / "probe_alpha_sweep.csv", pd.DataFrame(alpha_rows))
    save_probe_bundle(bundle_path, probes)
    np.savez_compressed(
        split0_path,
        occurrence_train_by_value=occurrence_train,
        **{
            f"{partition}_{kind}": value
            for partition, (values, activations) in split0.items()
            for kind, value in (("values", values), ("activations", activations))
        },
    )
    return probes, split0, occurrence_train


def orthonormal_union_basis(
    probes: Mapping[str, LinearPropertyProbe],
    property_ids: Sequence[str],
    *,
    orthogonal_to: np.ndarray | None,
) -> np.ndarray:
    rows = np.concatenate([probes[property_id].coefficients for property_id in property_ids], axis=0).astype(np.float64)
    if orthogonal_to is not None:
        direction = np.asarray(orthogonal_to, dtype=np.float64).reshape(-1)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        rows -= (rows @ direction)[:, None] * direction[None, :]
    _, singular, vh = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular[0]), 1e-12) * 1e-10))
    return np.asarray(vh[:rank], dtype=np.float64)


def write_geometry(
    *,
    output_dir: Path,
    probes: Mapping[str, LinearPropertyProbe],
    value_direction: np.ndarray,
    centroid_artifact: Path,
) -> None:
    bases = {
        property_id: probes[property_id].row_basis(orthogonal_to=value_direction)
        for property_id in PROPERTY_ORDER
    }
    overlap_rows: list[dict[str, Any]] = []
    for left in PROPERTY_ORDER:
        for right in PROPERTY_ORDER:
            cross = bases[left] @ bases[right].T
            singular = np.linalg.svd(cross, compute_uv=False)
            denom = max(1, min(len(bases[left]), len(bases[right])))
            overlap_rows.append(
                {
                    "property_a": left,
                    "property_b": right,
                    "rank_a": len(bases[left]),
                    "rank_b": len(bases[right]),
                    "max_canonical_cosine": float(singular[0]) if singular.size else 0.0,
                    "mean_squared_overlap": float(np.sum(cross * cross) / denom),
                }
            )
    atomic_write_csv(output_dir / "probe_subspace_overlap.csv", pd.DataFrame(overlap_rows))

    artifact = np.load(centroid_artifact)
    centroids = np.asarray(artifact["centroids"], dtype=np.float64)
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    total = float(np.sum(centered * centered))
    coverage_rows: list[dict[str, Any]] = []
    feature_sets = {property_id: (property_id,) for property_id in PROPERTY_ORDER}
    feature_sets["all_properties"] = PROPERTY_ORDER
    for name, property_ids in feature_sets.items():
        basis = orthonormal_union_basis(probes, property_ids, orthogonal_to=value_direction)
        projected = centered @ basis.T
        coverage_rows.append(
            {
                "feature_set": name,
                "rank": int(len(basis)),
                "centroid_variance_coverage": float(np.sum(projected * projected) / total),
            }
        )
    value_unit = np.asarray(value_direction, dtype=np.float64).reshape(1, -1)
    value_unit /= max(float(np.linalg.norm(value_unit)), 1e-12)
    all_basis = orthonormal_union_basis(probes, PROPERTY_ORDER, orthogonal_to=value_direction)
    combined = np.concatenate([value_unit, all_basis], axis=0)
    projected = centered @ combined.T
    coverage_rows.append(
        {
            "feature_set": "pls_value_plus_all_properties",
            "rank": int(len(combined)),
            "centroid_variance_coverage": float(np.sum(projected * projected) / total),
        }
    )
    atomic_write_csv(output_dir / "centroid_span_coverage.csv", pd.DataFrame(coverage_rows))


def load_panel(source_run_dir: Path, max_rows: int | None) -> pd.DataFrame:
    details = pd.read_csv(source_run_dir / "details.csv", low_memory=False)
    panel = details.loc[
        details["architecture"].eq("pls")
        & details["method"].eq("fixed")
        & details["delta_kind"].eq("value_code")
    ].copy()
    panel = panel.sort_values(["target_slot", "source_value", "target_value"]).reset_index(drop=True)
    if max_rows is not None:
        panel = panel.head(int(max_rows)).copy().reset_index(drop=True)
    if panel.empty:
        raise ValueError(f"No PLS fixed rows found in {source_run_dir}")
    return panel


def condition_deltas(
    *,
    probes: Mapping[str, LinearPropertyProbe],
    source_values: np.ndarray,
    target_values: np.ndarray,
    base_delta: np.ndarray,
    value_direction: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    cumulative = {
        "pls_plus_parity": ("parity",),
        "pls_plus_parity_primality": ("parity", "primality"),
        "pls_plus_divisibility": ("parity", "primality", "divisible_by_3", "divisible_by_5", "divisible_by_10"),
        "pls_plus_digit_count": ("parity", "primality", "divisible_by_3", "divisible_by_5", "divisible_by_10", "digit_count"),
        "pls_plus_units_digit": ("parity", "primality", "divisible_by_3", "divisible_by_5", "divisible_by_10", "digit_count", "units_digit"),
        "pls_plus_tens_digit": ("parity", "primality", "divisible_by_3", "divisible_by_5", "divisible_by_10", "digit_count", "units_digit", "tens_digit"),
        "pls_plus_all_independent": PROPERTY_ORDER,
    }
    conditions: dict[str, np.ndarray] = {"pls_value_only": np.asarray(base_delta, dtype=np.float64)}
    auxiliary_by_name: dict[str, np.ndarray] = {}
    for name, property_ids in cumulative.items():
        auxiliary = independent_sum_delta(
            probes,
            property_ids,
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        auxiliary_by_name[name] = auxiliary
        conditions[name] = base_delta + auxiliary
    for property_id in PROPERTY_ORDER:
        name = f"pls_plus_only_{property_id}"
        if property_id == "parity":
            continue
        auxiliary = independent_sum_delta(
            probes,
            (property_id,),
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        conditions[name] = base_delta + auxiliary
    for name, property_ids in {
        "pls_plus_intrinsic_independent": PROPERTY_ORDER[:5],
        "pls_plus_positional_digits_independent": PROPERTY_ORDER[6:],
        "pls_plus_decimal_form_independent": PROPERTY_ORDER[5:],
        "pls_plus_all_except_units_independent": tuple(
            property_id for property_id in PROPERTY_ORDER if property_id != "units_digit"
        ),
    }.items():
        auxiliary = independent_sum_delta(
            probes,
            property_ids,
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        conditions[name] = base_delta + auxiliary
    correct_aux = joint_minimum_norm_delta(
        probes,
        PROPERTY_ORDER,
        source_values,
        target_values,
        orthogonal_to=value_direction,
    )
    conditions["properties_only_joint"] = correct_aux
    conditions["pls_plus_all_joint"] = base_delta + correct_aux
    for scale in (0.25, 0.5, 2.0, 4.0, 6.0, 8.0):
        conditions[f"pls_plus_all_joint_scale_{str(scale).replace('.', 'p')}"] = base_delta + float(scale) * correct_aux
    conditions["pls_plus_all_wrong_sign"] = base_delta - correct_aux
    random_aux = norm_matched_random_deltas(correct_aux, seed=SEED, orthogonal_to=value_direction)
    conditions["pls_plus_norm_matched_random"] = base_delta + random_aux
    rng = np.random.default_rng(SEED)
    shuffled_targets = target_values[rng.permutation(len(target_values))]
    shuffled_aux = joint_minimum_norm_delta(
        probes,
        PROPERTY_ORDER,
        source_values,
        shuffled_targets,
        orthogonal_to=value_direction,
    )
    shuffled_aux = rescale_auxiliary_to_norm(shuffled_aux, np.linalg.norm(correct_aux, axis=1))
    conditions["pls_plus_norm_matched_shuffled"] = base_delta + shuffled_aux
    total_norm = np.linalg.norm(base_delta + correct_aux, axis=1)
    value_scaled = rescale_auxiliary_to_norm(base_delta, total_norm)
    conditions["pls_value_only_total_norm_matched"] = value_scaled
    independent_aux = auxiliary_by_name["pls_plus_all_independent"]
    for scale in (2.0, 4.0, 6.0, 8.0):
        slug = str(scale).replace(".", "p")
        conditions[f"pls_plus_all_independent_scale_{slug}"] = base_delta + float(scale) * independent_aux
        conditions[f"pls_plus_all_wrong_sign_scale_{slug}"] = base_delta - float(scale) * correct_aux
        conditions[f"pls_plus_norm_matched_random_scale_{slug}"] = base_delta + float(scale) * random_aux
        conditions[f"pls_plus_norm_matched_shuffled_scale_{slug}"] = base_delta + float(scale) * shuffled_aux
        scaled_total_norm = np.linalg.norm(base_delta + float(scale) * correct_aux, axis=1)
        conditions[f"pls_value_only_total_norm_matched_scale_{slug}"] = rescale_auxiliary_to_norm(
            base_delta,
            scaled_total_norm,
        )
    for property_id in PROPERTY_ORDER:
        property_aux = joint_minimum_norm_delta(
            probes,
            (property_id,),
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        conditions[f"pls_plus_joint4_only_{property_id}"] = base_delta + 4.0 * property_aux
        leave_one_out = tuple(candidate for candidate in PROPERTY_ORDER if candidate != property_id)
        leave_one_out_aux = joint_minimum_norm_delta(
            probes,
            leave_one_out,
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        conditions[f"pls_plus_joint4_all_except_{property_id}"] = base_delta + 4.0 * leave_one_out_aux
    for name, property_ids in cumulative.items():
        cumulative_aux = joint_minimum_norm_delta(
            probes,
            property_ids,
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        conditions[f"{name}_joint4"] = base_delta + 4.0 * cumulative_aux
    example_core = ("parity", "primality", "units_digit")
    example_core_aux = joint_minimum_norm_delta(
        probes,
        example_core,
        source_values,
        target_values,
        orthogonal_to=value_direction,
    )
    conditions["pls_plus_joint4_example_core"] = base_delta + 4.0 * example_core_aux

    diagnostics = pd.DataFrame(
        {
            "source_value": source_values,
            "target_value": target_values,
            "property_change_count": property_change_count(PROPERTY_ORDER, source_values, target_values),
            "base_delta_norm": np.linalg.norm(base_delta, axis=1),
            "correct_aux_norm": np.linalg.norm(correct_aux, axis=1),
            "independent_aux_norm": np.linalg.norm(auxiliary_by_name["pls_plus_all_independent"], axis=1),
            "correct_total_norm": np.linalg.norm(base_delta + correct_aux, axis=1),
            "value_dot_correct_aux": np.asarray(correct_aux @ (value_direction / np.linalg.norm(value_direction)), dtype=np.float64),
        }
    )
    return conditions, diagnostics


def write_intervention_probe_targeting(
    *,
    output_dir: Path,
    probes: Mapping[str, LinearPropertyProbe],
    source_activations: np.ndarray,
    source_values: np.ndarray,
    target_values: np.ndarray,
    conditions: Mapping[str, np.ndarray],
) -> None:
    rows: list[dict[str, Any]] = []
    for condition, delta in conditions.items():
        edited = np.asarray(source_activations, dtype=np.float64) + np.asarray(delta, dtype=np.float64)
        for property_id, probe in probes.items():
            source_labels = np.asarray([PROPERTY_LABELS[property_id](int(value)) for value in source_values], dtype=np.int64)
            target_labels = np.asarray([PROPERTY_LABELS[property_id](int(value)) for value in target_values], dtype=np.int64)
            changed = source_labels != target_labels
            prediction = probe.predict(edited)
            rows.append(
                {
                    "condition": condition,
                    "property_id": property_id,
                    "n": int(len(prediction)),
                    "changed_n": int(changed.sum()),
                    "target_match_rate": float(np.mean(prediction == target_labels)),
                    "changed_target_match_rate": float(np.mean(prediction[changed] == target_labels[changed])) if np.any(changed) else np.nan,
                    "unchanged_preservation_rate": float(np.mean(prediction[~changed] == source_labels[~changed])) if np.any(~changed) else np.nan,
                }
            )
    atomic_write_csv(output_dir / "intervention_probe_targeting.csv", pd.DataFrame(rows))


def evaluate_conditions(
    *,
    output_dir: Path,
    source_run_dir: Path,
    panel: pd.DataFrame,
    conditions: Mapping[str, np.ndarray],
    selected_conditions: Sequence[str],
    cfg: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    tokenizer,
    eval_device: str | None,
    eval_batch_size: int | None,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_path = output_dir / "causal_details.csv"
    summary_path = output_dir / "causal_summary.csv"
    existing_parts: dict[str, pd.DataFrame] = {}
    if detail_path.exists() and not force:
        existing = pd.read_csv(detail_path, low_memory=False)
        existing_parts = {str(name): frame.copy() for name, frame in existing.groupby("condition", sort=False)}
    todo = [name for name in selected_conditions if name not in existing_parts]
    if todo:
        # Keep this CPU-only preflight before the costly model load. It validates both
        # numeric-token coverage and the repository helper API used by causal scoring.
        tracked_values, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
        device = str(eval_device or cfg.get("device") or model_entry.get("device", "cpu"))
        dtype = str(cfg.get("dtype") or model_entry.get("dtype", "float32"))
        loaded = load_causal_lm(
            model_name=str(model_entry["hf_id"]),
            device=device,
            dtype=dtype,
            trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
            attn_implementation=model_entry.get("attn_implementation"),
            revision=model_entry.get("revision"),
        )
        freeze_model_parameters(loaded.model)
        batch_size = int(eval_batch_size or cfg.get("evaluation_batch_size", 100))
        for condition in todo:
            delta = np.asarray(conditions[condition], dtype=np.float32)
            result = _evaluate_with_delta_rows(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                valid_df=panel,
                delta_rows=delta,
                layer_idx=int(cfg["intervention_layer_idx"]),
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
                batch_size=batch_size,
                device=device,
            )
            result["condition"] = condition
            result["delta_norm"] = np.linalg.norm(delta.astype(np.float64), axis=1)
            result["property_change_count"] = property_change_count(
                PROPERTY_ORDER,
                panel["source_value"].to_numpy(dtype=np.int64),
                panel["target_value"].to_numpy(dtype=np.int64),
            )
            existing_parts[condition] = result
            combined = pd.concat([existing_parts[name] for name in selected_conditions if name in existing_parts], ignore_index=True)
            atomic_write_csv(detail_path, combined)
            print(f"[bag-of-features] evaluated {condition} ({len(panel)} rows)", flush=True)
    details = pd.concat([existing_parts[name] for name in selected_conditions if name in existing_parts], ignore_index=True)
    by_change = summarize_steering_metrics(details, group_cols=["condition", "property_change_count"])
    overall = summarize_steering_metrics(details, group_cols=["condition"])
    overall["property_change_count"] = np.nan
    summary = pd.concat([overall, by_change], ignore_index=True)
    atomic_write_csv(summary_path, summary)
    return details, summary


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    output_dir = ensure_dir(args.output_dir)
    cfg, source_model_entry, resolved = load_source_config(args.source_run_dir)
    models = load_yaml(args.models_config)
    model_entry = _resolve_model_entry(models, str(cfg["model"]))
    if str(model_entry["hf_id"]) != str(source_model_entry["hf_id"]):
        raise ValueError("Resolved model config does not match the source run model.")
    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=model_entry.get("revision"),
    )
    probes, split0, _ = train_or_load_probes(
        output_dir=output_dir,
        cfg=cfg,
        model_entry=model_entry,
        tokenizer=tokenizer,
        resolved=resolved,
        force=bool(args.force_probes),
    )
    centroid_payload = np.load(args.centroid_artifact)
    value_points = np.asarray(centroid_payload["pls_value_points"], dtype=np.float64)
    value_direction = np.asarray(centroid_payload["pls_decoder"], dtype=np.float64).reshape(-1)
    write_geometry(
        output_dir=output_dir,
        probes=probes,
        value_direction=value_direction,
        centroid_artifact=args.centroid_artifact,
    )
    panel = load_panel(args.source_run_dir, args.max_eval_rows)
    source_values = panel["source_value"].to_numpy(dtype=np.int64)
    target_values = panel["target_value"].to_numpy(dtype=np.int64)
    base_delta = value_points[target_values] - value_points[source_values]
    conditions, diagnostics = condition_deltas(
        probes=probes,
        source_values=source_values,
        target_values=target_values,
        base_delta=base_delta,
        value_direction=value_direction,
    )
    atomic_write_csv(output_dir / "pair_diagnostics.csv", diagnostics)
    split0_by_value = activations_by_value(split0)
    write_intervention_probe_targeting(
        output_dir=output_dir,
        probes=probes,
        source_activations=split0_by_value[source_values],
        source_values=source_values,
        target_values=target_values,
        conditions=conditions,
    )
    requested = [part.strip() for part in str(args.conditions).split(",") if part.strip()]
    selected = list(conditions) if requested == ["all"] else requested
    unknown = sorted(set(selected) - set(conditions))
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; available: {list(conditions)}")
    metadata = {
        "design": {
            "hypothesis": "Adding independently decoded number-property edits to the main PLS value edit improves exact arithmetic steering.",
            "unit": "Figure 1 ordered source-to-target intervention row",
            "independent_variable": "auxiliary number-property edit condition",
            "dependent_metrics": ["argmax_accuracy", "mean_abs_target_error", "mean_target_prob_delta", "valid_output_rate"],
            "matched_settings": ["model", "revision", "addition_formula prompt", "split_00 panel", "layer 10", "token locus", "target sampler", "decoder"],
            "controls": ["wrong sign", "norm-matched random", "norm-matched shuffled target properties", "value-only total-norm matched"],
        },
        "model": str(model_entry["hf_id"]),
        "model_revision": str(resolved["model_revision"]),
        "layer": int(cfg["intervention_layer_idx"]),
        "source_run": str(args.source_run_dir),
        "centroid_artifact": str(args.centroid_artifact),
        "property_order": list(PROPERTY_ORDER),
        "probe_training": "mean activation by value over split_01..split_07; alpha selected on split_08; final accuracy on split_09",
        "strict_probe_audit": "separate split_00 train/dev/test value-disjoint probes",
        "conditions": selected,
        "panel_rows": int(len(panel)),
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    if args.skip_causal_eval:
        print(output_dir)
        return
    evaluate_conditions(
        output_dir=output_dir,
        source_run_dir=args.source_run_dir,
        panel=panel,
        conditions=conditions,
        selected_conditions=selected,
        cfg=cfg,
        model_entry=model_entry,
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        eval_batch_size=args.eval_batch_size,
        force=bool(args.force_eval),
    )
    (output_dir / "DONE").write_text("done\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
