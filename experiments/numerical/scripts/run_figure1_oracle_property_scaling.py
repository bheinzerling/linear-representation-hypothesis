#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    PROPERTY_ORDER,
    SEED,
    load_panel,
    load_probe_bundle,
    load_source_config,
    write_intervention_probe_targeting,
)
from scripts.run_source_value_probe_manifold_steering import (  # noqa: E402
    _collect_shifted_representations_for_slot,
    _evaluate_with_delta_rows,
    _resolve_model_entry,
)
from src.gradient_steering import freeze_model_parameters  # noqa: E402
from src.model_loader import load_causal_lm, load_tokenizer  # noqa: E402
from src.number_property_probes import (  # noqa: E402
    LinearPropertyProbe,
    norm_matched_random_deltas,
    oracle_independent_projection_delta,
    oracle_joint_projection_delta,
    probe_union_basis,
    project_activation_difference,
    property_change_count,
    property_labels,
    rescale_auxiliary_to_norm,
)
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.source_value_steering import summarize_steering_metrics  # noqa: E402
from src.utils import atomic_write_csv, atomic_write_json, ensure_dir, load_yaml, runtime_metadata, set_global_seed  # noqa: E402


SOURCE_RUN = ROOT / "out/runs/source_value_probe_manifold_steering__9bbad17c01ee__r0"
BASE_BAG_RUN = ROOT / "out/figure1_bag_of_features_steering/llama31_8b_layer10_v1"
CENTROID_ARTIFACT = (
    ROOT / "experiment-reports/artifacts/2026-06-16-figure1-allpair-delta-dictionary/value_representations.npz"
)
DEFAULT_OUTPUT = ROOT / "out/figure1_oracle_property_scaling/llama31_8b_layer10_v1"
CONDITION_ORDER = (
    "oracle_changed_joint_plus_pls",
    "oracle_all_joint_plus_pls",
    "oracle_changed_independent_plus_pls",
    "oracle_all_independent_plus_pls",
    "oracle_changed_properties_only",
    "oracle_changed_wrong_sign_plus_pls",
    "oracle_changed_random_norm_matched_plus_pls",
    "oracle_changed_shuffled_norm_matched_plus_pls",
    "oracle_value_only_total_norm_matched",
    "oracle_full_contextual_patch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate closed-form oracle property scaling from contextual target projections."
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--base-bag-run-dir", type=Path, default=BASE_BAG_RUN)
    parser.add_argument("--centroid-artifact", type=Path, default=CENTROID_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--conditions", default="all", help="Comma-separated condition ids or 'all'.")
    parser.add_argument("--force-target-activations", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def panel_signature(panel: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in panel[["source_value", "target_value", "target_slot", "shifted_prompt", "shifted_token_idx"]].itertuples(
        index=False, name=None
    ):
        digest.update(json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_source_activations(base_bag_run_dir: Path, panel: pd.DataFrame) -> np.ndarray:
    with np.load(base_bag_run_dir / "split00_activations.npz") as payload:
        dev_values = np.asarray(payload["dev_values"], dtype=np.int64)
        dev_activations = np.asarray(payload["dev_activations"], dtype=np.float64)
    row_idx = panel["baseline_row_idx"].to_numpy(dtype=np.int64)
    source_values = panel["source_value"].to_numpy(dtype=np.int64)
    if row_idx.min(initial=0) < 0 or row_idx.max(initial=-1) >= len(dev_values):
        raise ValueError("Panel baseline row indices are outside the cached split_00 dev activation range.")
    if not np.array_equal(dev_values[row_idx], source_values):
        raise ValueError("Panel source rows do not match cached split_00 dev activations.")
    return np.asarray(dev_activations[row_idx], dtype=np.float64)


def load_cached_target_activations(path: Path, *, signature: str, expected_rows: int) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as payload:
        cached_signature = str(np.asarray(payload["panel_signature"]).reshape(-1)[0])
        activations = np.asarray(payload["target_activations"], dtype=np.float64)
    if cached_signature != signature or activations.shape[0] != int(expected_rows):
        return None
    return activations


def collect_target_activations(
    *,
    output_path: Path,
    signature: str,
    panel: pd.DataFrame,
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    activations = _collect_shifted_representations_for_slot(
        slot_df=panel,
        model=model,
        tokenizer=tokenizer,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
    )
    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        panel_signature=np.asarray([signature]),
        target_activations=np.asarray(activations, dtype=np.float32),
    )
    return np.asarray(activations, dtype=np.float64)


def changed_property_masks(source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    return np.stack(
        [property_labels(property_id, source_values) != property_labels(property_id, target_values) for property_id in PROPERTY_ORDER],
        axis=1,
    )


def oracle_changed_joint_delta(
    *,
    probes: Mapping[str, LinearPropertyProbe],
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    changed_masks: np.ndarray,
    value_direction: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(np.asarray(source_activations, dtype=np.float64))
    bit_weights = 1 << np.arange(len(PROPERTY_ORDER), dtype=np.int64)
    mask_codes = np.asarray(changed_masks, dtype=np.int64) @ bit_weights
    for code in np.unique(mask_codes):
        selected_rows = np.flatnonzero(mask_codes == int(code))
        selected_properties = [
            property_id for property_idx, property_id in enumerate(PROPERTY_ORDER) if int(code) & (1 << property_idx)
        ]
        if not selected_properties:
            continue
        result[selected_rows] = oracle_joint_projection_delta(
            probes,
            selected_properties,
            source_activations[selected_rows],
            target_activations[selected_rows],
            orthogonal_to=value_direction,
        )
    return result


def oracle_changed_independent_delta(
    *,
    probes: Mapping[str, LinearPropertyProbe],
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    changed_masks: np.ndarray,
    value_direction: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(np.asarray(source_activations, dtype=np.float64))
    for property_idx, property_id in enumerate(PROPERTY_ORDER):
        rows = np.flatnonzero(changed_masks[:, property_idx])
        if len(rows) == 0:
            continue
        result[rows] += oracle_independent_projection_delta(
            probes,
            (property_id,),
            source_activations[rows],
            target_activations[rows],
            orthogonal_to=value_direction,
        )
    return result


def oracle_conditions(
    *,
    probes: Mapping[str, LinearPropertyProbe],
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_activations: np.ndarray,
    target_activations: np.ndarray,
    base_delta: np.ndarray,
    value_direction: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    changed_masks = changed_property_masks(source_values, target_values)
    contextual_difference = np.asarray(target_activations - source_activations, dtype=np.float64)
    changed_joint = oracle_changed_joint_delta(
        probes=probes,
        source_activations=source_activations,
        target_activations=target_activations,
        changed_masks=changed_masks,
        value_direction=value_direction,
    )
    all_joint = oracle_joint_projection_delta(
        probes,
        PROPERTY_ORDER,
        source_activations,
        target_activations,
        orthogonal_to=value_direction,
    )
    changed_independent = oracle_changed_independent_delta(
        probes=probes,
        source_activations=source_activations,
        target_activations=target_activations,
        changed_masks=changed_masks,
        value_direction=value_direction,
    )
    all_independent = oracle_independent_projection_delta(
        probes,
        PROPERTY_ORDER,
        source_activations,
        target_activations,
        orthogonal_to=value_direction,
    )

    rng = np.random.default_rng(SEED)
    shuffled_targets = target_activations[rng.permutation(len(target_activations))]
    shuffled = oracle_changed_joint_delta(
        probes=probes,
        source_activations=source_activations,
        target_activations=shuffled_targets,
        changed_masks=changed_masks,
        value_direction=value_direction,
    )
    shuffled = rescale_auxiliary_to_norm(shuffled, np.linalg.norm(changed_joint, axis=1))
    random = norm_matched_random_deltas(changed_joint, seed=SEED, orthogonal_to=value_direction)
    total_norm = np.linalg.norm(base_delta + changed_joint, axis=1)
    value_only = rescale_auxiliary_to_norm(base_delta, total_norm)

    conditions = {
        "oracle_changed_joint_plus_pls": base_delta + changed_joint,
        "oracle_all_joint_plus_pls": base_delta + all_joint,
        "oracle_changed_independent_plus_pls": base_delta + changed_independent,
        "oracle_all_independent_plus_pls": base_delta + all_independent,
        "oracle_changed_properties_only": changed_joint,
        "oracle_changed_wrong_sign_plus_pls": base_delta - changed_joint,
        "oracle_changed_random_norm_matched_plus_pls": base_delta + random,
        "oracle_changed_shuffled_norm_matched_plus_pls": base_delta + shuffled,
        "oracle_value_only_total_norm_matched": value_only,
        "oracle_full_contextual_patch": contextual_difference,
    }

    pair_diagnostics = pd.DataFrame(
        {
            "source_value": source_values,
            "target_value": target_values,
            "property_change_count": changed_masks.sum(axis=1),
            "base_pls_norm": np.linalg.norm(base_delta, axis=1),
            "oracle_changed_joint_norm": np.linalg.norm(changed_joint, axis=1),
            "oracle_all_joint_norm": np.linalg.norm(all_joint, axis=1),
            "oracle_changed_independent_norm": np.linalg.norm(changed_independent, axis=1),
            "oracle_all_independent_norm": np.linalg.norm(all_independent, axis=1),
            "full_contextual_patch_norm": np.linalg.norm(contextual_difference, axis=1),
        }
    )

    property_rows: list[pd.DataFrame] = []
    for property_idx, property_id in enumerate(PROPERTY_ORDER):
        basis = probes[property_id].row_basis(orthogonal_to=value_direction)
        oracle_component = project_activation_difference(source_activations, target_activations, basis)
        nominal = probes[property_id].minimum_norm_delta(
            source_values,
            target_values,
            orthogonal_to=value_direction,
        )
        nominal_sq = np.sum(np.square(nominal), axis=1)
        equivalent_scale = np.divide(
            np.sum(oracle_component * nominal, axis=1),
            nominal_sq,
            out=np.full(len(source_values), np.nan, dtype=np.float64),
            where=nominal_sq > 1e-18,
        )
        residual = oracle_component - np.nan_to_num(equivalent_scale)[:, None] * nominal
        oracle_norm = np.linalg.norm(oracle_component, axis=1)
        residual_fraction = np.divide(
            np.linalg.norm(residual, axis=1),
            oracle_norm,
            out=np.zeros(len(source_values), dtype=np.float64),
            where=oracle_norm > 1e-12,
        )
        property_rows.append(
            pd.DataFrame(
                {
                    "row_idx": np.arange(len(source_values), dtype=np.int64),
                    "source_value": source_values,
                    "target_value": target_values,
                    "property_id": property_id,
                    "probe_rank": int(basis.shape[0]),
                    "property_changed": changed_masks[:, property_idx],
                    "oracle_projection_norm": oracle_norm,
                    "nominal_centroid_delta_norm": np.sqrt(nominal_sq),
                    "nominal_equivalent_scale": equivalent_scale,
                    "scalar_fit_residual_fraction": residual_fraction,
                }
            )
        )
    return conditions, pair_diagnostics, pd.concat(property_rows, ignore_index=True)


def summarize_projection_diagnostics(details: pd.DataFrame) -> pd.DataFrame:
    changed = details.loc[details["property_changed"]].copy()
    return (
        changed.groupby("property_id", as_index=False)
        .agg(
            n=("row_idx", "size"),
            probe_rank=("probe_rank", "first"),
            mean_oracle_projection_norm=("oracle_projection_norm", "mean"),
            median_oracle_projection_norm=("oracle_projection_norm", "median"),
            median_nominal_equivalent_scale=("nominal_equivalent_scale", "median"),
            q10_nominal_equivalent_scale=("nominal_equivalent_scale", lambda values: values.quantile(0.10)),
            q90_nominal_equivalent_scale=("nominal_equivalent_scale", lambda values: values.quantile(0.90)),
            median_scalar_fit_residual_fraction=("scalar_fit_residual_fraction", "median"),
        )
        .sort_values("property_id")
        .reset_index(drop=True)
    )


def evaluate_conditions(
    *,
    output_dir: Path,
    panel: pd.DataFrame,
    conditions: Mapping[str, np.ndarray],
    selected_conditions: Sequence[str],
    model,
    tokenizer,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
    device: str,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_path = output_dir / "causal_details.csv"
    existing_parts: dict[str, pd.DataFrame] = {}
    if detail_path.exists() and not force:
        existing = pd.read_csv(detail_path, low_memory=False)
        existing_parts = {str(name): frame.copy() for name, frame in existing.groupby("condition", sort=False)}
    for condition in [name for name in selected_conditions if name not in existing_parts]:
        delta = np.asarray(conditions[condition], dtype=np.float32)
        result = _evaluate_with_delta_rows(
            model=model,
            tokenizer=tokenizer,
            valid_df=panel,
            delta_rows=delta,
            layer_idx=int(layer_idx),
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            batch_size=int(batch_size),
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
        print(f"[oracle-scaling] evaluated {condition} ({len(panel)} rows)", flush=True)
    details = pd.concat([existing_parts[name] for name in selected_conditions if name in existing_parts], ignore_index=True)
    by_change = summarize_steering_metrics(details, group_cols=["condition", "property_change_count"])
    overall = summarize_steering_metrics(details, group_cols=["condition"])
    overall["property_change_count"] = np.nan
    summary = pd.concat([overall, by_change], ignore_index=True)
    atomic_write_csv(output_dir / "causal_summary.csv", summary)
    return details, summary


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    output_dir = ensure_dir(args.output_dir)
    cfg, source_model_entry, resolved = load_source_config(args.source_run_dir)
    model_entry = _resolve_model_entry(load_yaml(args.models_config), str(cfg["model"]))
    if str(model_entry["hf_id"]) != str(source_model_entry["hf_id"]):
        raise ValueError("Resolved model config does not match the source run model.")
    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=model_entry.get("revision"),
    )
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
    probes = load_probe_bundle(args.base_bag_run_dir / "property_probes.npz")
    panel = load_panel(args.source_run_dir, args.max_eval_rows)
    source_values = panel["source_value"].to_numpy(dtype=np.int64)
    target_values = panel["target_value"].to_numpy(dtype=np.int64)
    source_activations = load_source_activations(args.base_bag_run_dir, panel)
    centroid_payload = np.load(args.centroid_artifact)
    value_points = np.asarray(centroid_payload["pls_value_points"], dtype=np.float64)
    value_direction = np.asarray(centroid_payload["pls_decoder"], dtype=np.float64).reshape(-1)
    base_delta = value_points[target_values] - value_points[source_values]

    requested = [part.strip() for part in str(args.conditions).split(",") if part.strip()]
    selected = list(CONDITION_ORDER) if requested == ["all"] else requested
    unknown = sorted(set(selected) - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; available: {list(CONDITION_ORDER)}")

    device = str(args.eval_device or cfg.get("device") or model_entry.get("device", "cpu"))
    dtype = str(cfg.get("dtype") or model_entry.get("dtype", "float32"))
    batch_size = int(args.eval_batch_size or cfg.get("evaluation_batch_size", 100))
    target_path = output_dir / "target_activations.npz"
    signature = panel_signature(panel)
    target_activations = None if args.force_target_activations else load_cached_target_activations(
        target_path, signature=signature, expected_rows=len(panel)
    )

    loaded = load_causal_lm(
        model_name=str(model_entry["hf_id"]),
        device=device,
        dtype=dtype,
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        attn_implementation=model_entry.get("attn_implementation"),
        revision=model_entry.get("revision"),
    )
    freeze_model_parameters(loaded.model)
    if target_activations is None:
        target_activations = collect_target_activations(
            output_path=target_path,
            signature=signature,
            panel=panel,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            layer_idx=int(cfg["intervention_layer_idx"]),
            batch_size=batch_size,
            device=device,
        )

    conditions, pair_diagnostics, projection_diagnostics = oracle_conditions(
        probes=probes,
        source_values=source_values,
        target_values=target_values,
        source_activations=source_activations,
        target_activations=target_activations,
        base_delta=base_delta,
        value_direction=value_direction,
    )
    atomic_write_csv(output_dir / "pair_diagnostics.csv", pair_diagnostics)
    atomic_write_csv(output_dir / "property_projection_diagnostics.csv", projection_diagnostics)
    atomic_write_csv(output_dir / "property_projection_summary.csv", summarize_projection_diagnostics(projection_diagnostics))
    write_intervention_probe_targeting(
        output_dir=output_dir,
        probes=probes,
        source_activations=source_activations,
        source_values=source_values,
        target_values=target_values,
        conditions={name: conditions[name] for name in selected},
    )
    atomic_write_json(
        output_dir / "metadata.json",
        {
            "design": {
                "hypothesis": "Contextual target projection supplies a principled row-specific property dose without a global scale sweep.",
                "oracle_information": "Layer-10 activation of the actual shifted target prompt; no output success or target-token probability is used for scaling.",
                "primary_condition": "oracle_changed_joint_plus_pls",
                "primary_definition": "PLS delta plus the source-to-contextual-target difference projected into the union of changed property-probe row spaces after removing the PLS direction.",
                "matched_settings": [
                    "model",
                    "revision",
                    "addition_formula prompt",
                    "split_00 panel",
                    "layer 10",
                    "token locus",
                    "target sampler",
                    "decoder",
                ],
                "controls": ["wrong sign", "norm-matched random", "norm-matched shuffled target projection", "value-only total-norm matched"],
                "caveat": "This is an oracle diagnostic because obtaining the shifted target activation requires a separate forward pass of the desired target prompt.",
            },
            "model": str(model_entry["hf_id"]),
            "model_revision": str(resolved["model_revision"]),
            "layer": int(cfg["intervention_layer_idx"]),
            "panel_rows": int(len(panel)),
            "panel_signature": signature,
            "conditions": selected,
            "source_run": str(args.source_run_dir),
            "base_bag_run": str(args.base_bag_run_dir),
            "runtime": runtime_metadata(),
        },
    )
    evaluate_conditions(
        output_dir=output_dir,
        panel=panel,
        conditions=conditions,
        selected_conditions=selected,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
        layer_idx=int(cfg["intervention_layer_idx"]),
        batch_size=batch_size,
        device=device,
        force=bool(args.force_eval),
    )
    (output_dir / "DONE").write_text("done\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
