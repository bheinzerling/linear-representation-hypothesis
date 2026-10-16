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
    DEFAULT_OUTPUT as BAG_OUTPUT,
    PROPERTY_ORDER,
    SOURCE_RUN,
    load_source_config,
)
from scripts.run_figure1_iterative_nullspace_property_probes import (  # noqa: E402
    DEFAULT_OUTPUT as ORIGINAL_PROBE_OUTPUT,
    EXTENDED_SUPPORTED_PROPERTY_ORDER,
    load_regime_data,
)
from scripts.run_source_value_probe_capacity_sweep import _load_partition_rows  # noqa: E402
from src.gradient_steering import forward_next_token_logits_with_delta, freeze_model_parameters  # noqa: E402
from src.model_loader import load_causal_lm  # noqa: E402
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.property_subspace_synthesis import (  # noqa: E402
    actual_projection_sum,
    distribution_shift_detail,
    fit_property_synthesis_model,
    geometric_detail,
    independent_property_synthesis,
    joint_minimum_norm_property_synthesis,
    oracle_union_projection,
    rowwise_norm_match_to_actual,
    summarize_geometric_fit,
)
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    runtime_metadata,
    set_global_seed,
)


EXTENDED_PROBE_OUTPUT = ROOT / "out/figure1_iterative_nullspace_extended_property_probes/llama31_8b_layer10_v1"
DEFAULT_OUTPUT = ROOT / "out/property_subspace_representation_synthesis/llama31_8b_layer10_v1"
SEED = 20260722
REGIMES = ("occurrence_heldout", "value_disjoint")
ORIGINAL_PROPERTIES = ("magnitude", *PROPERTY_ORDER)
ADDED_PROPERTIES = tuple(EXTENDED_SUPPORTED_PROPERTY_ORDER)
ALL_PROPERTIES = (*ORIGINAL_PROPERTIES, *ADDED_PROPERTIES)
PROPERTY_GROUPS: dict[str, tuple[str, ...]] = {
    "magnitude": ("magnitude",),
    "positional": ("magnitude", "digit_count", "units_digit", "tens_digit", "hundreds_digit"),
    "original10": ORIGINAL_PROPERTIES,
    "added10": ADDED_PROPERTIES,
    "all20": ALL_PROPERTIES,
}
METHOD_METADATA = {
    "global_mean": ("negative_control", "none", False),
    "magnitude_label_sum": ("label_sum", "magnitude", False),
    "positional_label_sum": ("label_sum", "positional", False),
    "original10_label_sum": ("label_sum", "original10", False),
    "added10_label_sum": ("label_sum", "added10", False),
    "all20_label_sum": ("label_sum", "all20", False),
    "all20_label_sum_norm_matched": ("label_sum_norm_matched", "all20", True),
    "original10_label_joint": ("label_joint", "original10", False),
    "all20_label_joint": ("label_joint", "all20", False),
    "original10_actual_projection_sum": ("actual_projection_sum", "original10", True),
    "all20_actual_projection_sum": ("actual_projection_sum", "all20", True),
    "original10_union_oracle": ("union_projection_oracle", "original10", True),
    "all20_union_oracle": ("union_projection_oracle", "all20", True),
    "exact_cached_activation": ("exact_control", "all", True),
}
CONDITION_ORDER = tuple(METHOD_METADATA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize layer activations from iterative property subspaces and score "
            "geometric reconstruction plus causal next-token behavior."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--bag-output-dir", type=Path, default=BAG_OUTPUT)
    parser.add_argument("--original-probe-output-dir", type=Path, default=ORIGINAL_PROBE_OUTPUT)
    parser.add_argument("--extended-probe-output-dir", type=Path, default=EXTENDED_PROBE_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regimes", default="all", help="Comma-separated regimes or 'all'.")
    parser.add_argument("--conditions", default="all", help="Comma-separated conditions or 'all'.")
    parser.add_argument("--skip-behavior", action="store_true")
    parser.add_argument("--behavior-only", action="store_true")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-behavior-rows", type=int, default=None)
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--force-behavior", action="store_true")
    return parser.parse_args()


def _selected(raw: str, available: Sequence[str]) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return tuple(available)
    selected = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown selections {unknown}; available: {list(available)}")
    return selected


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_property_bases(
    *, regime: str, original_probe_output_dir: Path, extended_probe_output_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    bases: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for property_id in ALL_PROPERTIES:
        root = original_probe_output_dir if property_id in ORIGINAL_PROPERTIES else extended_probe_output_dir
        direction_path = root / regime / property_id / "directions.npz"
        summary_path = root / regime / property_id / "summary.json"
        if not direction_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing property basis or summary for {regime}/{property_id}: {direction_path}")
        saved_basis = np.asarray(np.load(direction_path)["basis"], dtype=np.float64)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dev_selected_count = int(summary["greedy_dev_selected_direction_count"])
        confirmed_prefix = int(summary["test_confirmed_prefix_count"])
        confirmed_count = int(summary["test_confirmed_direction_count"])
        if saved_basis.ndim != 2 or saved_basis.shape[1] != dev_selected_count or dev_selected_count <= 0:
            raise ValueError(
                f"Saved basis/dev-selection mismatch for {regime}/{property_id}: "
                f"saved shape={saved_basis.shape}, dev-selected={dev_selected_count}."
            )
        if not (0 <= confirmed_prefix <= confirmed_count <= dev_selected_count):
            raise ValueError(
                f"Invalid test-confirmation provenance for {regime}/{property_id}: "
                f"prefix={confirmed_prefix}, count={confirmed_count}, dev-selected={dev_selected_count}."
            )
        # The basis is selected using the development set. Test-confirmation fields
        # are evaluation metadata only; using them to select columns here would leak
        # the reconstruction test into the representation definition.
        bases[property_id] = saved_basis
        provenance[property_id] = {
            "path": str(direction_path.resolve()),
            "sha256": _sha256(direction_path),
            "saved_greedy_direction_count": int(saved_basis.shape[1]),
            "test_confirmed_direction_count": confirmed_count,
            "test_confirmed_prefix_count": confirmed_prefix,
            "included": True,
            "stopped_reason": summary.get("stopped_reason"),
        }
    return bases, provenance


def _evaluation_frame(*, cfg: Mapping[str, Any], regime: str) -> pd.DataFrame:
    split_name = "split_09" if regime == "occurrence_heldout" else "split_00"
    rows = _load_partition_rows(Path(cfg["split_dataset_dir"]), split_name, str(cfg["family"]))
    if regime == "occurrence_heldout":
        frame = pd.concat([rows[part] for part in ("train", "dev", "test")], ignore_index=True)
    else:
        frame = rows["test"].copy()
    frame = frame.sort_values("source_value").reset_index(drop=True)
    if frame["source_value"].duplicated().any():
        raise ValueError(f"Evaluation frame for {regime} does not contain one row per source value.")
    return frame


def _coordinate_diagnostics(
    *, model, test_values: np.ndarray, test_activations: np.ndarray, regime: str
) -> list[dict[str, Any]]:
    centered = np.asarray(test_activations, dtype=np.float64) - model.center[None, :]
    rows: list[dict[str, Any]] = []
    for property_id, coordinate_model in model.coordinate_models.items():
        actual = centered @ coordinate_model.basis
        predicted = coordinate_model.coordinates(test_values)
        residual = actual - predicted
        denominator = float(np.sum(actual**2))
        actual_flat = actual.reshape(-1)
        predicted_flat = predicted.reshape(-1)
        cosine_denom = float(np.linalg.norm(actual_flat) * np.linalg.norm(predicted_flat))
        rows.append(
            {
                "regime": regime,
                "property_id": property_id,
                "direction_count": int(actual.shape[1]),
                "coordinate_fraction_energy_explained": float(
                    1.0 - np.sum(residual**2) / max(denominator, 1e-24)
                ),
                "coordinate_rmse": float(np.sqrt(np.mean(residual**2))),
                "coordinate_global_cosine_similarity": float(
                    np.sum(actual_flat * predicted_flat) / max(cosine_denom, 1e-24)
                ),
            }
        )
    return rows


def _build_syntheses(
    *, model, values: np.ndarray, actual: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    n = len(values)
    syntheses: dict[str, np.ndarray] = {
        "global_mean": np.repeat(model.center[None, :], n, axis=0),
        "exact_cached_activation": np.asarray(actual, dtype=np.float64).copy(),
    }
    diagnostics: dict[str, dict[str, Any]] = {
        "global_mean": {},
        "exact_cached_activation": {},
    }
    def active_properties(group_id: str) -> tuple[str, ...]:
        return tuple(
            property_id
            for property_id in PROPERTY_GROUPS[group_id]
            if property_id in model.coordinate_models
        )

    def group_diagnostics(group_id: str) -> dict[str, Any]:
        active = active_properties(group_id)
        omitted = tuple(property_id for property_id in PROPERTY_GROUPS[group_id] if property_id not in active)
        return {
            "requested_property_count": len(PROPERTY_GROUPS[group_id]),
            "active_property_count": len(active),
            "active_direction_count": int(
                sum(model.coordinate_models[property_id].basis.shape[1] for property_id in active)
            ),
            "omitted_zero_confirmed_properties": ";".join(omitted),
        }

    for group_id in ("magnitude", "positional", "original10", "added10", "all20"):
        condition = f"{group_id}_label_sum"
        syntheses[condition] = independent_property_synthesis(
            model, values=values, property_ids=active_properties(group_id)
        )
        diagnostics[condition] = group_diagnostics(group_id)
    syntheses["all20_label_sum_norm_matched"] = rowwise_norm_match_to_actual(
        syntheses["all20_label_sum"], actual_activations=actual, center=model.center
    )
    diagnostics["all20_label_sum_norm_matched"] = group_diagnostics("all20")
    for group_id in ("original10", "all20"):
        active = active_properties(group_id)
        joint_condition = f"{group_id}_label_joint"
        syntheses[joint_condition], diagnostics[joint_condition] = joint_minimum_norm_property_synthesis(
            model, values=values, property_ids=active
        )
        diagnostics[joint_condition].update(group_diagnostics(group_id))
        sum_condition = f"{group_id}_actual_projection_sum"
        syntheses[sum_condition] = actual_projection_sum(
            model, actual_activations=actual, property_ids=active
        )
        diagnostics[sum_condition] = group_diagnostics(group_id)
        oracle_condition = f"{group_id}_union_oracle"
        syntheses[oracle_condition], union_basis = oracle_union_projection(
            model, actual_activations=actual, property_ids=active
        )
        diagnostics[oracle_condition] = {
            **group_diagnostics(group_id),
            "union_rank": int(union_basis.shape[1]),
        }
    missing = sorted(set(CONDITION_ORDER) - set(syntheses))
    if missing:
        raise RuntimeError(f"Missing synthesis conditions: {missing}")
    return syntheses, diagnostics


def run_geometry(
    *,
    output_dir: Path,
    regimes: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    selected_regimes: Sequence[str],
    cfg: Mapping[str, Any],
    original_probe_output_dir: Path,
    extended_probe_output_dir: Path,
    force: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    detail_path = output_dir / "geometry_detail.csv"
    summary_path = output_dir / "geometry_summary.csv"
    coordinate_path = output_dir / "property_coordinate_summary.csv"
    metadata_path = output_dir / "geometry_metadata.json"
    if all(path.exists() for path in (detail_path, summary_path, coordinate_path, metadata_path)) and not force:
        return {}, json.loads(metadata_path.read_text(encoding="utf-8"))

    prepared: dict[str, dict[str, Any]] = {}
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    basis_provenance: dict[str, Any] = {}
    for regime in selected_regimes:
        train_values, train_x = regimes[regime]["train"]
        test_values, test_x = regimes[regime]["test"]
        frame = _evaluation_frame(cfg=cfg, regime=regime)
        if not np.array_equal(frame["source_value"].to_numpy(dtype=np.int64), np.asarray(test_values)):
            raise ValueError(f"Prompt frame and cached activations are misaligned for {regime}.")
        bases, basis_provenance[regime] = _load_property_bases(
            regime=regime,
            original_probe_output_dir=original_probe_output_dir,
            extended_probe_output_dir=extended_probe_output_dir,
        )
        model = fit_property_synthesis_model(
            train_activations=train_x,
            train_values=train_values,
            property_bases=bases,
        )
        syntheses, diagnostics = _build_syntheses(model=model, values=test_values, actual=test_x)
        coordinate_rows.extend(
            _coordinate_diagnostics(
                model=model,
                test_values=np.asarray(test_values),
                test_activations=np.asarray(test_x),
                regime=regime,
            )
        )
        for condition in CONDITION_ORDER:
            detail = geometric_detail(
                actual_activations=test_x, synthesis=syntheses[condition], center=model.center
            )
            condition_frame = pd.DataFrame(detail)
            condition_frame.insert(0, "source_value", np.asarray(test_values, dtype=np.int64))
            condition_frame.insert(0, "condition", condition)
            condition_frame.insert(0, "regime", regime)
            detail_frames.append(condition_frame)
            summary = summarize_geometric_fit(
                actual_activations=test_x,
                synthesis=syntheses[condition],
                center=model.center,
                pair_seed=SEED,
            )
            method_family, property_group, uses_actual = METHOD_METADATA[condition]
            summary_rows.append(
                {
                    "regime": regime,
                    "condition": condition,
                    "method_family": method_family,
                    "property_group": property_group,
                    "property_count": len(PROPERTY_GROUPS.get(property_group, ())),
                    "uses_actual_test_activation": uses_actual,
                    **diagnostics[condition],
                    **summary,
                }
            )
        prepared[regime] = {
            "model": model,
            "values": np.asarray(test_values, dtype=np.int64),
            "actual": np.asarray(test_x, dtype=np.float64),
            "frame": frame,
            "syntheses": syntheses,
        }

    atomic_write_csv(detail_path, pd.concat(detail_frames, ignore_index=True))
    atomic_write_csv(summary_path, pd.DataFrame(summary_rows))
    atomic_write_csv(coordinate_path, pd.DataFrame(coordinate_rows))
    metadata = {
        "schema_version": 1,
        "seed": SEED,
        "regimes": list(selected_regimes),
        "conditions": list(CONDITION_ORDER),
        "property_groups": {key: list(value) for key, value in PROPERTY_GROUPS.items()},
        "coordinate_estimator": {
            "categorical": "training-label centroid of centered activation coordinates in each property basis",
            "magnitude": "OLS linear map from integer value to centered activation coordinates",
        },
        "basis_provenance": basis_provenance,
        "runtime": runtime_metadata(),
    }
    atomic_write_json(metadata_path, metadata)
    return prepared, metadata


def _prepare_for_behavior(
    *,
    regimes: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    selected_regimes: Sequence[str],
    cfg: Mapping[str, Any],
    original_probe_output_dir: Path,
    extended_probe_output_dir: Path,
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for regime in selected_regimes:
        train_values, train_x = regimes[regime]["train"]
        test_values, test_x = regimes[regime]["test"]
        frame = _evaluation_frame(cfg=cfg, regime=regime)
        if not np.array_equal(frame["source_value"].to_numpy(dtype=np.int64), np.asarray(test_values)):
            raise ValueError(f"Prompt frame and cached activations are misaligned for {regime}.")
        bases, _ = _load_property_bases(
            regime=regime,
            original_probe_output_dir=original_probe_output_dir,
            extended_probe_output_dir=extended_probe_output_dir,
        )
        model = fit_property_synthesis_model(
            train_activations=train_x, train_values=train_values, property_bases=bases
        )
        syntheses, _ = _build_syntheses(model=model, values=test_values, actual=test_x)
        prepared[regime] = {
            "model": model,
            "values": np.asarray(test_values, dtype=np.int64),
            "actual": np.asarray(test_x, dtype=np.float64),
            "frame": frame,
            "syntheses": syntheses,
        }
    return prepared


def _behavior_summary(frame: pd.DataFrame, *, regime: str, condition: str) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=np.float64)
    expected = frame["synthesis_numeric_expected_value"].to_numpy(dtype=np.float64)
    return {
        "regime": regime,
        "condition": condition,
        "n": int(len(frame)),
        "baseline_task_accuracy": float(frame["baseline_correct"].mean()),
        "synthesis_task_accuracy": float(frame["synthesis_correct"].mean()),
        "top1_agreement": float(frame["top1_agreement"].mean()),
        "mean_full_kl_original_to_synthesis": float(frame["full_kl_original_to_synthesis"].mean()),
        "p95_full_kl_original_to_synthesis": float(
            frame["full_kl_original_to_synthesis"].quantile(0.95)
        ),
        "mean_full_kl_synthesis_to_original": float(frame["full_kl_synthesis_to_original"].mean()),
        "mean_numeric_conditional_kl_original_to_synthesis": float(
            frame["numeric_conditional_kl_original_to_synthesis"].mean()
        ),
        "mean_target_logprob_delta": float(frame["target_logprob_delta"].mean()),
        "mean_target_prob_delta": float(frame["target_prob_delta"].mean()),
        "mean_synthesis_numeric_mass": float(frame["synthesis_numeric_mass"].mean()),
        "mean_abs_synthesis_expected_error": float(np.mean(np.abs(expected - target))),
        "synthesis_numeric_top1_rate": float(frame["synthesis_top1_numeric"].mean()),
    }


def run_behavior(
    *,
    output_dir: Path,
    prepared: Mapping[str, Mapping[str, Any]],
    selected_regimes: Sequence[str],
    selected_conditions: Sequence[str],
    cfg: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    resolved: Mapping[str, Any],
    eval_device: str | None,
    eval_batch_size: int | None,
    max_rows: int | None,
    force: bool,
) -> None:
    behavior_dir = ensure_dir(output_dir / "behavior_parts")
    todo: list[tuple[str, str]] = []
    for regime in selected_regimes:
        for condition in selected_conditions:
            path = behavior_dir / f"{regime}__{condition}.csv"
            if force or not path.exists():
                todo.append((regime, condition))
    if todo:
        device = str(eval_device or cfg.get("device") or model_entry.get("device", "cpu"))
        dtype = str(cfg.get("dtype") or model_entry.get("dtype", "float32"))
        loaded = load_causal_lm(
            model_name=str(model_entry["hf_id"]),
            device=device,
            dtype=dtype,
            trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
            attn_implementation=model_entry.get("attn_implementation"),
            revision=str(resolved["model_revision"]),
        )
        freeze_model_parameters(loaded.model)
        numeric_values, numeric_token_ids = _numeric_tracking_arrays(loaded.tokenizer)
        numeric_id_to_value = {int(token_id): int(value) for value, token_id in zip(numeric_values, numeric_token_ids)}
        batch_size = int(eval_batch_size or cfg.get("evaluation_batch_size", 100))
        alignment_rows: list[dict[str, Any]] = []
        for regime in selected_regimes:
            regime_todo = [condition for candidate_regime, condition in todo if candidate_regime == regime]
            if not regime_todo:
                continue
            payload = prepared[regime]
            frame = payload["frame"].copy()
            actual = np.asarray(payload["actual"], dtype=np.float64)
            syntheses = payload["syntheses"]
            if max_rows is not None:
                frame = frame.iloc[: int(max_rows)].reset_index(drop=True)
                actual = actual[: len(frame)]
            prompts = frame["prompt"].astype(str).tolist()
            token_idxs = frame["token_idx"].astype(int).tolist()
            baseline_logits, live_activations = forward_next_token_logits_with_delta(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                prompts=prompts,
                token_idxs=token_idxs,
                layer_idx=int(cfg["intervention_layer_idx"]),
                batch_size=batch_size,
                device=device,
                delta_rows=None,
                capture_selected=True,
            )
            if live_activations is None:
                raise RuntimeError("Baseline forward did not return selected layer activations.")
            live_residual = np.asarray(live_activations, dtype=np.float64) - actual
            alignment_rows.append(
                {
                    "regime": regime,
                    "n": int(len(frame)),
                    "cache_live_rmse": float(np.sqrt(np.mean(live_residual**2))),
                    "cache_live_mean_l2": float(np.mean(np.linalg.norm(live_residual, axis=1))),
                    "cache_live_max_abs": float(np.max(np.abs(live_residual))),
                }
            )
            targets = frame["target_token_id"].to_numpy(dtype=np.int64)
            for condition in regime_todo:
                target_activation = np.asarray(syntheses[condition][: len(frame)], dtype=np.float32)
                delta = target_activation - np.asarray(live_activations, dtype=np.float32)
                synthesis_logits, _ = forward_next_token_logits_with_delta(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompts=prompts,
                    token_idxs=token_idxs,
                    layer_idx=int(cfg["intervention_layer_idx"]),
                    batch_size=batch_size,
                    device=device,
                    delta_rows=delta,
                    capture_selected=False,
                )
                metrics = distribution_shift_detail(
                    baseline_logits=baseline_logits,
                    synthesis_logits=synthesis_logits,
                    target_token_ids=targets,
                    numeric_token_ids=numeric_token_ids,
                    numeric_values=numeric_values,
                )
                result = frame[["source_value", "target", "target_token_id", "x1", "x2", "prompt"]].copy()
                for name, values in metrics.items():
                    result[name] = values
                synthesis_top1 = result["synthesis_top1_token_id"].to_numpy(dtype=np.int64)
                baseline_top1 = result["baseline_top1_token_id"].to_numpy(dtype=np.int64)
                result["synthesis_top1_numeric"] = [int(token_id in numeric_id_to_value) for token_id in synthesis_top1]
                result["baseline_top1_numeric"] = [int(token_id in numeric_id_to_value) for token_id in baseline_top1]
                result["synthesis_top1_value"] = [numeric_id_to_value.get(int(token_id), np.nan) for token_id in synthesis_top1]
                result["baseline_top1_value"] = [numeric_id_to_value.get(int(token_id), np.nan) for token_id in baseline_top1]
                result.insert(0, "condition", condition)
                result.insert(0, "regime", regime)
                atomic_write_csv(behavior_dir / f"{regime}__{condition}.csv", result)
                del synthesis_logits, metrics, result
            del baseline_logits, live_activations
        atomic_write_csv(output_dir / "cache_live_alignment.csv", pd.DataFrame(alignment_rows))

    parts: list[pd.DataFrame] = []
    for regime in selected_regimes:
        for condition in selected_conditions:
            path = behavior_dir / f"{regime}__{condition}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing behavioral result part: {path}")
            parts.append(pd.read_csv(path, low_memory=False))
    details = pd.concat(parts, ignore_index=True)
    summaries = [
        _behavior_summary(frame, regime=str(regime), condition=str(condition))
        for (regime, condition), frame in details.groupby(["regime", "condition"], sort=False)
    ]
    atomic_write_csv(output_dir / "behavior_detail.csv", details)
    atomic_write_csv(output_dir / "behavior_summary.csv", pd.DataFrame(summaries))
    atomic_write_json(
        output_dir / "behavior_metadata.json",
        {
            "schema_version": 1,
            "regimes": list(selected_regimes),
            "conditions": list(selected_conditions),
            "max_rows": max_rows,
            "model_id": str(model_entry["hf_id"]),
            "model_revision": str(resolved["model_revision"]),
            "layer": int(cfg["intervention_layer_idx"]),
            "replacement_definition": "delta = synthesized cached state - live state at the selected token",
            "full_kl_direction": "original || synthesis",
            "runtime": runtime_metadata(),
        },
    )


def main() -> None:
    args = parse_args()
    if args.skip_behavior and args.behavior_only:
        raise ValueError("--skip-behavior and --behavior-only are mutually exclusive.")
    set_global_seed(SEED)
    selected_regimes = _selected(args.regimes, REGIMES)
    selected_conditions = _selected(args.conditions, CONDITION_ORDER)
    output_dir = ensure_dir(args.output_dir)
    cfg, model_entry, resolved = load_source_config(args.source_run_dir)
    regimes, representation_provenance = load_regime_data(
        source_run_dir=args.source_run_dir, bag_output_dir=args.bag_output_dir
    )
    prepared: dict[str, dict[str, Any]] = {}
    if not args.behavior_only:
        prepared, _ = run_geometry(
            output_dir=output_dir,
            regimes=regimes,
            selected_regimes=selected_regimes,
            cfg=cfg,
            original_probe_output_dir=args.original_probe_output_dir,
            extended_probe_output_dir=args.extended_probe_output_dir,
            force=args.force_geometry,
        )
    if not args.skip_behavior:
        if not prepared:
            prepared = _prepare_for_behavior(
                regimes=regimes,
                selected_regimes=selected_regimes,
                cfg=cfg,
                original_probe_output_dir=args.original_probe_output_dir,
                extended_probe_output_dir=args.extended_probe_output_dir,
            )
        run_behavior(
            output_dir=output_dir,
            prepared=prepared,
            selected_regimes=selected_regimes,
            selected_conditions=selected_conditions,
            cfg=cfg,
            model_entry=model_entry,
            resolved=resolved,
            eval_device=args.eval_device,
            eval_batch_size=args.eval_batch_size,
            max_rows=args.max_behavior_rows,
            force=args.force_behavior,
        )
    atomic_write_json(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "seed": SEED,
            "regimes": list(selected_regimes),
            "geometry_conditions": list(CONDITION_ORDER),
            "behavior_conditions": [] if args.skip_behavior else list(selected_conditions),
            "representation_provenance": representation_provenance,
            "source_run": str(args.source_run_dir.resolve()),
            "original_probe_output": str(args.original_probe_output_dir.resolve()),
            "extended_probe_output": str(args.extended_probe_output_dir.resolve()),
            "runtime": runtime_metadata(),
        },
    )
    atomic_write_text(output_dir / "DONE", "ok\n")
    print(output_dir)


if __name__ == "__main__":
    main()
