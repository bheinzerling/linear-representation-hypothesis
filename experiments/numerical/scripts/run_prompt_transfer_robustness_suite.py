#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file as load_safetensors_numpy
from safetensors.numpy import save_file as save_safetensors_numpy
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_prompt_style_context_centroid_relative_layers import (
    ARCHITECTURES,
    PARTITIONS,
    collect_multi_layer_activations,
    fit_config_from_selection,
    layer_rel_to_abs,
    load_partition_frame,
    selected_capacity_table,
    validate_panel,
)
from src.magnitude_intervention import fit_magnitude_probe
from src.model_loader import load_causal_lm, load_tokenizer
from src.prompt_transfer_robustness import (
    aggregate_context_subsets,
    assert_value_alignment,
    build_balanced_pooled_matrix,
    clustered_direction_bootstrap,
    direction_pair_rows,
    fit_gain_and_intercept,
    fit_intercept_only,
    global_value_centroids,
    median_control_row,
    probe_prediction_metrics,
    rounded_prediction_metrics,
    scalar_direction,
    stable_hash,
    translated_probe,
)
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    get_git_metadata,
    load_yaml,
    set_global_seed,
)


DEFAULT_CONFIG = ROOT / "configs" / "prompt_transfer_robustness_suite.yaml"
AGGREGATE_FORMAT_VERSION = 1
EXPECTED_FULL_COUNTS = {"train": 800, "dev": 100, "test": 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the matched prompt-transfer robustness suite.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("collect", "analyze", "all", "finalize"), default="all")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--force-analyze", action="store_true")
    parser.add_argument("--smoke-values-per-partition", type=int, default=None)
    parser.add_argument("--layer-position", type=int, default=None, help="Analyze one zero-based relative-layer position.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_safetensors(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_safetensors_numpy({key: np.ascontiguousarray(value) for key, value in arrays.items()}, str(tmp))
    os.replace(tmp, path)


def _resolved_model_config(cfg: dict[str, Any], model_alias: str, dtype_override: str | None) -> dict[str, Any]:
    models_cfg = load_yaml(ROOT / str(cfg["models_config"]))
    resolved = {**models_cfg.get("default", {}), **models_cfg["models"][model_alias], **cfg["models"][model_alias]}
    if dtype_override is not None:
        resolved["dtype"] = str(dtype_override)
    return resolved


def _run_identity(
    *, cfg: dict[str, Any], config_path: Path, model_alias: str, dtype_override: str | None,
    smoke_values_per_partition: int | None,
) -> tuple[str, dict[str, Any]]:
    implementation_files = [
        ROOT / "scripts" / "run_prompt_transfer_robustness_suite.py",
        ROOT / "src" / "prompt_transfer_robustness.py",
        ROOT / "scripts" / "run_prompt_style_context_centroid_relative_layers.py",
        ROOT / "src" / "magnitude_intervention.py",
        ROOT / "src" / "model_loader.py",
    ]
    payload = {
        "task": "prompt_transfer_robustness_suite",
        "config_sha256": sha256_file(config_path),
        "selection_sha256": sha256_file(ROOT / str(cfg["selection_csv"])),
        "model_alias": model_alias,
        "resolved_model": _resolved_model_config(cfg, model_alias, dtype_override),
        "smoke_values_per_partition": smoke_values_per_partition,
        "cublas_workspace_config": ":4096:8",
        "aggregate_format_version": AGGREGATE_FORMAT_VERSION,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in implementation_files
        },
    }
    return f"{model_alias}__{stable_hash(payload)}", payload


def _restrict_smoke_values(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    values = np.sort(frame["source_value"].astype(int).unique())[: int(limit)]
    return frame.loc[frame["source_value"].astype(int).isin(values)].reset_index(drop=True)


def _panel_fingerprint(
    *, tokenizer: Any, frame: pd.DataFrame, activation_payload: dict[str, Any]
) -> tuple[str, list[list[int]]]:
    prompts = frame["prompt"].astype(str).tolist()
    encoded = tokenizer(prompts, add_special_tokens=False, padding=False)["input_ids"]
    digest = hashlib.sha256(json.dumps(activation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for row, input_ids in zip(frame.itertuples(index=False), encoded, strict=True):
        token_idx = int(row.token_idx)
        if token_idx < 0 or token_idx >= len(input_ids):
            raise ValueError(f"Token index {token_idx} is invalid for value {row.source_value}.")
        decoded = tokenizer.decode([int(input_ids[token_idx])], skip_special_tokens=False).strip()
        if not decoded.isdigit() or int(decoded) != int(row.source_value):
            raise ValueError(
                f"Tokenizer/locus mismatch for value={row.source_value}, token_idx={token_idx}, decoded={decoded!r}."
            )
        row_payload = [
            str(row.prompt),
            [int(value) for value in input_ids],
            token_idx,
            int(row.source_value),
            int(row.variant_idx),
        ]
        digest.update(json.dumps(row_payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), [[int(value) for value in input_ids] for input_ids in encoded]


def collect_model_aggregates(
    *, cfg: dict[str, Any], model_alias: str, output_dir: Path, device: str,
    dtype_override: str | None, force: bool, smoke_values_per_partition: int | None,
) -> dict[str, Any]:
    expected_paths = [
        output_dir / "aggregates" / str(family) / f"{partition}.safetensors"
        for family in cfg["families"] for partition in PARTITIONS
    ]
    metadata_path = output_dir / "model_metadata.json"
    if all(path.is_file() and path.with_suffix(".manifest.json").is_file() for path in expected_paths) and metadata_path.is_file() and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    run_model = _resolved_model_config(cfg, model_alias, dtype_override)
    model_id = str(run_model["hf_id"])
    revision = str(run_model["revision"])
    trust_remote_code = bool(run_model.get("trust_remote_code", False))
    attn_implementation = run_model.get("attn_implementation")
    auto_config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    n_layers = int(auto_config.num_hidden_layers)
    relative_layers = [float(value) for value in cfg["relative_layers"]]
    layer_indices = [layer_rel_to_abs(value, n_layers) for value in relative_layers]
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"Relative layers are not unique for {model_alias}: {layer_indices}")

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code, revision=revision)
    dataset_root = ROOT / str(cfg["split_dataset_dir"])
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    shard_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for family in cfg["families"]:
        for partition in PARTITIONS:
            frame = load_partition_frame(dataset_root, str(cfg["split_name"]), str(family), partition, None)
            frame = _restrict_smoke_values(frame, smoke_values_per_partition)
            validate_panel(frame, require_complete=True)
            activation_payload = {
                "format_version": AGGREGATE_FORMAT_VERSION,
                "model_alias": model_alias,
                "model_id": model_id,
                "model_revision": revision,
                "compute_dtype": str(run_model.get("dtype", "float32")),
                "attn_implementation": str(attn_implementation),
                "relative_layers": relative_layers,
                "layer_indices": layer_indices,
                "family": str(family),
                "partition": partition,
                "context_split": "even versus odd variant_idx with paired per-value count",
            }
            panel_sha256, _ = _panel_fingerprint(tokenizer=tokenizer, frame=frame, activation_payload=activation_payload)
            frames[(str(family), partition)] = frame
            shard_payloads[(str(family), partition)] = {**activation_payload, "panel_sha256": panel_sha256}

    loaded = load_causal_lm(
        model_id,
        device=device,
        dtype=str(run_model.get("dtype", "float32")),
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        revision=revision,
    )
    coverage_rows: list[dict[str, Any]] = []
    try:
        for family in cfg["families"]:
            for partition in PARTITIONS:
                path = output_dir / "aggregates" / str(family) / f"{partition}.safetensors"
                manifest_path = path.with_suffix(".manifest.json")
                payload = shard_payloads[(str(family), partition)]
                if path.is_file() and manifest_path.is_file() and not force:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("activation_payload") != payload:
                        raise ValueError(f"Aggregate cache identity mismatch at {path}; use --force-collect to replace it.")
                    continue
                frame = frames[(str(family), partition)]
                started = time.time()
                activations = collect_multi_layer_activations(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    frame=frame,
                    layer_indices=layer_indices,
                    batch_size=int(run_model["batch_size"]),
                    device=device,
                )
                aggregated = aggregate_context_subsets(frame, activations)
                storage_dtype = np.float16 if str(cfg.get("storage_dtype", "float16")) == "float16" else np.float32
                arrays = {
                    key: value.astype(storage_dtype) if key.endswith("_centroid") else value
                    for key, value in aggregated.items()
                }
                _atomic_save_safetensors(path, arrays)
                atomic_write_json(
                    manifest_path,
                    {
                        "activation_payload": payload,
                        "artifact_sha256": sha256_file(path),
                        "tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
                        "storage_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
                        "elapsed_seconds": float(time.time() - started),
                    },
                )
                for value, all_count, paired_count in zip(
                    aggregated["values"], aggregated["all_context_counts"], aggregated["paired_context_counts"], strict=True
                ):
                    coverage_rows.append(
                        {
                            "family": str(family),
                            "partition": partition,
                            "source_value": int(value),
                            "all_context_count": int(all_count),
                            "seen_context_count": int(paired_count),
                            "heldout_context_count": int(paired_count),
                        }
                    )
                del activations, aggregated, arrays
                gc.collect()
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if coverage_rows:
        atomic_write_csv(output_dir / "context_coverage.csv", pd.DataFrame(coverage_rows))
    metadata = {
        "model_alias": model_alias,
        "model_id": model_id,
        "model_revision": revision,
        "n_layers": n_layers,
        "relative_layers": relative_layers,
        "layer_indices": layer_indices,
        "batch_size": int(run_model["batch_size"]),
        "compute_dtype": str(run_model.get("dtype", "float32")),
        "attn_implementation": str(attn_implementation),
        "collection_host": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_text(output_dir / "COLLECT_DONE", "complete\n")
    return metadata


def _load_aggregate_panel(cfg: dict[str, Any], output_dir: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    panel: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for family in cfg["families"]:
        for partition in PARTITIONS:
            path = output_dir / "aggregates" / str(family) / f"{partition}.safetensors"
            if not path.is_file() or not path.with_suffix(".manifest.json").is_file():
                raise FileNotFoundError(f"Missing validated aggregate shard: {path}")
            panel[(str(family), partition)] = {
                key: np.asarray(value, dtype=np.float32) if key.endswith("_centroid") else np.asarray(value)
                for key, value in load_safetensors_numpy(str(path)).items()
            }
    return panel


def _selection_for(
    selections: pd.DataFrame, *, relative_layer: float, family: str, architecture: str
) -> pd.Series:
    match = selections.loc[
        np.isclose(selections["relative_layer"].astype(float), float(relative_layer))
        & selections["train_family"].eq(str(family))
        & selections["architecture"].eq(str(architecture))
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one capacity for layer={relative_layer}, family={family}, architecture={architecture}.")
    return match.iloc[0]


def _fit_from_row(row: pd.Series, train_x: np.ndarray, train_values: np.ndarray):
    fit_architecture, fit_config = fit_config_from_selection(row)
    return fit_magnitude_probe(
        architecture=fit_architecture,
        train_x=train_x,
        train_values=train_values,
        config=fit_config,
        value_min=0,
        value_max=999,
    ), fit_architecture, fit_config


def _capacity_fields(row: pd.Series) -> dict[str, Any]:
    return {
        "control_name": str(row["control_name"]),
        "control_value": float(row["control_value"]),
        "capacity_source": "single_context_native_dev_2026_04_20",
    }


def _record_examples(
    examples: list[dict[str, Any]], *, base: dict[str, Any], truth: np.ndarray, rounded: np.ndarray,
    continuous: np.ndarray,
) -> None:
    y = np.asarray(truth, dtype=np.int64).reshape(-1)
    pred = np.asarray(rounded, dtype=np.int64).reshape(-1)
    if len(y) == 0:
        return
    error = np.abs(pred - y)
    worst = int(np.argmax(error))
    random_seed = int(stable_hash({key: str(value) for key, value in base.items()}, 8), 16)
    random_idx = int(np.random.default_rng(random_seed).integers(0, len(y)))
    continuous_array = np.asarray(continuous, dtype=np.float64).reshape(-1)
    for selection, idx in (("largest_absolute_error", worst), ("seeded_random", random_idx)):
        examples.append(
            {
                **base,
                "selection": selection,
                "source_value": int(y[idx]),
                "prediction": int(pred[idx]),
                "continuous_prediction": float(continuous_array[idx]) if np.isfinite(continuous_array[idx]) else float("nan"),
                "absolute_error": int(error[idx]),
            }
        )


def _record_probe_metric(
    rows: list[dict[str, Any]], examples: list[dict[str, Any]], *, base: dict[str, Any], probe: Any,
    x: np.ndarray, y: np.ndarray, save_examples: bool = False,
) -> None:
    metrics, rounded, continuous = probe_prediction_metrics(probe=probe, representations=x, truth=y)
    row = {**base, **metrics}
    rows.append(row)
    if save_examples:
        _record_examples(examples, base=base, truth=y, rounded=rounded, continuous=continuous)


def _record_continuous_metric(
    rows: list[dict[str, Any]], examples: list[dict[str, Any]], *, base: dict[str, Any], prediction: np.ndarray,
    y: np.ndarray, save_examples: bool = False,
) -> None:
    metrics, rounded = rounded_prediction_metrics(truth=y, continuous_prediction=prediction)
    row = {**base, **metrics}
    rows.append(row)
    if save_examples:
        _record_examples(examples, base=base, truth=y, rounded=rounded, continuous=prediction)


def _validate_partition_counts(
    panel: dict[tuple[str, str], dict[str, np.ndarray]], families: list[str], smoke_values: int | None
) -> None:
    expected = {partition: int(smoke_values) for partition in PARTITIONS} if smoke_values is not None else EXPECTED_FULL_COUNTS
    for partition in PARTITIONS:
        values = assert_value_alignment(
            (family, panel[(family, partition)]["values"]) for family in families
        )
        if len(values) != expected[partition]:
            raise ValueError(f"Expected {expected[partition]} {partition} values, found {len(values)}.")


def analyze_layer(
    *, cfg: dict[str, Any], model_alias: str, model_meta: dict[str, Any], output_dir: Path,
    panel: dict[tuple[str, str], dict[str, np.ndarray]], selections: pd.DataFrame,
    layer_position: int, force: bool,
) -> None:
    relative_layer = float(model_meta["relative_layers"][layer_position])
    layer_idx = int(model_meta["layer_indices"][layer_position])
    layer_dir = output_dir / "analysis" / "layers" / f"layer_{layer_position:02d}"
    if (layer_dir / "DONE").is_file() and not force:
        return
    layer_dir.mkdir(parents=True, exist_ok=True)
    families = [str(value) for value in cfg["families"]]
    architectures = [str(value) for value in cfg["architectures"]]
    direction_architectures = [str(value) for value in cfg["direction_architectures"]]
    common = {
        "model_alias": model_alias,
        "model_id": model_meta["model_id"],
        "model_revision": model_meta["model_revision"],
        "relative_layer": relative_layer,
        "layer_idx": layer_idx,
        "n_layers": int(model_meta["n_layers"]),
    }
    arrays: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for family in families:
        for partition in PARTITIONS:
            shard = panel[(family, partition)]
            values = shard["values"].astype(np.int64)
            for condition in cfg["conditions"]:
                arrays[(family, partition, str(condition))] = (
                    values,
                    shard[str(condition)][:, layer_position, :].astype(np.float32),
                )

    metrics_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    direction_tensors: dict[str, np.ndarray] = {}

    # Tests 1 and 2: matched new values, then paired held-out contexts.
    seen_probes: dict[tuple[str, str], Any] = {}
    for family in families:
        train_values, train_x = arrays[(family, "train", "seen_context_centroid")]
        test_values, test_seen_x = arrays[(family, "test", "seen_context_centroid")]
        heldout_values, test_heldout_x = arrays[(family, "test", "heldout_context_centroid")]
        if not np.array_equal(test_values, heldout_values):
            raise ValueError("Seen and held-out context evaluations use different values.")
        for architecture in architectures:
            selection = _selection_for(
                selections, relative_layer=relative_layer, family=family, architecture=architecture
            )
            probe, fit_architecture, _ = _fit_from_row(selection, train_x, train_values)
            seen_probes[(family, architecture)] = probe
            base = {
                **common,
                "question": "new_values",
                "condition": "matched_seen_context_distribution",
                "architecture": architecture,
                "fit_architecture": fit_architecture,
                "train_family": family,
                "test_family": family,
                "replicate": 0,
                "num_train_examples": int(len(train_values)),
                "num_calibration_examples": 0,
                **_capacity_fields(selection),
            }
            _record_probe_metric(metrics_rows, examples, base=base, probe=probe, x=test_seen_x, y=test_values)
            for context_condition, local_x in (
                ("seen_contexts", test_seen_x), ("heldout_contexts", test_heldout_x)
            ):
                base = {
                    **common,
                    "question": "new_contexts_within_family",
                    "condition": context_condition,
                    "architecture": architecture,
                    "fit_architecture": fit_architecture,
                    "train_family": family,
                    "test_family": family,
                    "replicate": 0,
                    "num_train_examples": int(len(train_values)),
                    "num_calibration_examples": 0,
                    **_capacity_fields(selection),
                }
                _record_probe_metric(
                    metrics_rows, examples, base=base, probe=probe, x=local_x, y=test_values, save_examples=True
                )

    # Native all-context probes are reused by tests 3--6.
    raw_probes: dict[tuple[str, str], Any] = {}
    for family in families:
        train_values, train_x = arrays[(family, "train", "all_context_centroid")]
        for architecture in architectures:
            selection = _selection_for(
                selections, relative_layer=relative_layer, family=family, architecture=architecture
            )
            probe, _, _ = _fit_from_row(selection, train_x, train_values)
            raw_probes[(family, architecture)] = probe
            if architecture in direction_architectures:
                direction_tensors[f"{architecture}__{family}"] = scalar_direction(probe).astype(np.float32)

    family_train_means = {
        family: arrays[(family, "train", "all_context_centroid")][1].mean(axis=0, dtype=np.float64)
        for family in families
    }
    common_train_mean = np.stack(list(family_train_means.values()), axis=0).mean(axis=0)

    for source_family in families:
        for architecture in architectures:
            selection = _selection_for(
                selections, relative_layer=relative_layer, family=source_family, architecture=architecture
            )
            probe = raw_probes[(source_family, architecture)]
            aligned_probe = translated_probe(
                probe, source_family_mean=family_train_means[source_family], common_mean=common_train_mean
            )
            for target_family in families:
                test_values, target_test_x = arrays[(target_family, "test", "all_context_centroid")]
                fit_architecture = str(probe.architecture)
                base = {
                    **common,
                    "question": "complete_affine_cross_family",
                    "condition": "raw",
                    "architecture": architecture,
                    "fit_architecture": fit_architecture,
                    "train_family": source_family,
                    "test_family": target_family,
                    "replicate": 0,
                    "num_train_examples": int(len(arrays[(source_family, "train", "all_context_centroid")][0])),
                    "num_calibration_examples": 0,
                    **_capacity_fields(selection),
                }
                _record_probe_metric(metrics_rows, examples, base=base, probe=probe, x=target_test_x, y=test_values)
                for mean_condition in ("raw", "family_mean_aligned"):
                    local_probe = probe if mean_condition == "raw" else aligned_probe
                    local_x = (
                        target_test_x
                        if mean_condition == "raw"
                        else target_test_x - family_train_means[target_family][None, :] + common_train_mean[None, :]
                    )
                    base = {
                        **common,
                        "question": "family_mean_shift",
                        "condition": mean_condition,
                        "architecture": architecture,
                        "fit_architecture": fit_architecture,
                        "train_family": source_family,
                        "test_family": target_family,
                        "replicate": 0,
                        "num_train_examples": int(len(arrays[(source_family, "train", "all_context_centroid")][0])),
                        "num_calibration_examples": 0,
                        "alignment_examples_per_family": int(len(arrays[(target_family, "train", "all_context_centroid")][0])),
                        **_capacity_fields(selection),
                    }
                    _record_probe_metric(metrics_rows, examples, base=base, probe=local_probe, x=local_x, y=test_values)

                if architecture not in direction_architectures or source_family == target_family:
                    continue
                dev_values, target_dev_x = arrays[(target_family, "dev", "all_context_centroid")]
                direction = np.asarray(probe.coefficients, dtype=np.float64)[:, 0]
                intercept_weight, target_intercept = fit_intercept_only(
                    coefficients=direction, calibration_x=target_dev_x, calibration_y=dev_values
                )
                unit_direction, target_gain, target_gain_intercept = fit_gain_and_intercept(
                    direction=direction, calibration_x=target_dev_x, calibration_y=dev_values
                )
                calibration_conditions = {
                    "complete_source_affine": np.asarray(probe.predict_features(target_test_x))[:, 0],
                    "target_intercept": target_test_x @ intercept_weight + target_intercept,
                    "target_gain_and_intercept": target_gain * (target_test_x @ unit_direction) + target_gain_intercept,
                    "native_target_probe": np.asarray(raw_probes[(target_family, architecture)].predict_features(target_test_x))[:, 0],
                }
                for calibration_condition, continuous_prediction in calibration_conditions.items():
                    condition_selection = (
                        _selection_for(
                            selections,
                            relative_layer=relative_layer,
                            family=target_family,
                            architecture=architecture,
                        )
                        if calibration_condition == "native_target_probe"
                        else selection
                    )
                    base = {
                        **common,
                        "question": "direction_after_calibration",
                        "condition": calibration_condition,
                        "architecture": architecture,
                        "fit_architecture": fit_architecture,
                        "train_family": source_family,
                        "test_family": target_family,
                        "replicate": 0,
                        "num_train_examples": int(len(arrays[(source_family, "train", "all_context_centroid")][0])),
                        "num_calibration_examples": 0 if calibration_condition in {"complete_source_affine", "native_target_probe"} else int(len(dev_values)),
                        "target_gain": target_gain if calibration_condition == "target_gain_and_intercept" else float("nan"),
                        **_capacity_fields(condition_selection),
                    }
                    _record_continuous_metric(
                        metrics_rows,
                        examples,
                        base=base,
                        prediction=continuous_prediction,
                        y=test_values,
                        save_examples=True,
                    )

    # Test 6: independently fitted direction recurrence on one common test set.
    global_test_x = global_value_centroids(
        {family: arrays[(family, "test", "all_context_centroid")][1] for family in families}
    )
    for architecture in direction_architectures:
        local_probes = {family: raw_probes[(family, architecture)] for family in families}
        for row in direction_pair_rows(probes=local_probes, common_x=global_test_x):
            direction_rows.append({**common, "architecture": architecture, **row})

    # Test 7: exactly 800 family-balanced rows, with the target family held out.
    train_values = assert_value_alignment(
        (family, arrays[(family, "train", "all_context_centroid")][0]) for family in families
    )
    for target_idx, target_family in enumerate(families):
        source_families = [family for family in families if family != target_family]
        target_test_values, target_test_x = arrays[(target_family, "test", "all_context_centroid")]
        for architecture in direction_architectures:
            source_selection_rows = pd.DataFrame(
                [
                    _selection_for(
                        selections, relative_layer=relative_layer, family=family, architecture=architecture
                    )
                    for family in source_families
                ]
            )
            pooled_selection = median_control_row(source_selection_rows)
            native_selection = _selection_for(
                selections, relative_layer=relative_layer, family=target_family, architecture=architecture
            )
            native_probe = raw_probes[(target_family, architecture)]
            native_base = {
                **common,
                "question": "balanced_pooled_leave_one_family_out",
                "condition": "native_target_probe",
                "architecture": architecture,
                "fit_architecture": str(native_probe.architecture),
                "train_family": target_family,
                "test_family": target_family,
                "replicate": -1,
                "num_train_examples": int(len(train_values)),
                "num_calibration_examples": 0,
                **_capacity_fields(native_selection),
            }
            _record_probe_metric(
                metrics_rows, examples, base=native_base, probe=native_probe, x=target_test_x, y=target_test_values
            )
            for replicate in range(int(cfg["pooled_assignment_replicates"])):
                assignment_seed = int(cfg["seed"]) + (1009 * target_idx) + replicate
                pooled_values, pooled_x, _ = build_balanced_pooled_matrix(
                    values=train_values,
                    family_arrays={
                        family: arrays[(family, "train", "all_context_centroid")][1]
                        for family in source_families
                    },
                    seed=assignment_seed,
                )
                pooled_probe, fit_architecture, _ = _fit_from_row(pooled_selection, pooled_x, pooled_values)
                base = {
                    **common,
                    "question": "balanced_pooled_leave_one_family_out",
                    "condition": "balanced_pooled_probe",
                    "architecture": architecture,
                    "fit_architecture": fit_architecture,
                    "train_family": "|".join(source_families),
                    "test_family": target_family,
                    "replicate": replicate,
                    "assignment_seed": assignment_seed,
                    "num_train_examples": int(len(pooled_values)),
                    "num_source_families": int(len(source_families)),
                    "examples_per_source_family": int(len(pooled_values) // len(source_families)),
                    "num_calibration_examples": 0,
                    **_capacity_fields(pooled_selection),
                }
                _record_probe_metric(
                    metrics_rows,
                    examples,
                    base=base,
                    probe=pooled_probe,
                    x=target_test_x,
                    y=target_test_values,
                    save_examples=True,
                )

    # Test 8: one equal-family global centroid row per value.
    global_train_x = global_value_centroids(
        {family: arrays[(family, "train", "all_context_centroid")][1] for family in families}
    )
    global_test_values = assert_value_alignment(
        (family, arrays[(family, "test", "all_context_centroid")][0]) for family in families
    )
    for architecture in architectures:
        all_selection_rows = pd.DataFrame(
            [
                _selection_for(selections, relative_layer=relative_layer, family=family, architecture=architecture)
                for family in families
            ]
        )
        global_selection = median_control_row(all_selection_rows)
        global_probe, fit_architecture, _ = _fit_from_row(global_selection, global_train_x, train_values)
        base = {
            **common,
            "question": "global_value_centroid_decodability",
            "condition": "equal_family_global_centroid",
            "architecture": architecture,
            "fit_architecture": fit_architecture,
            "train_family": "global",
            "test_family": "global",
            "replicate": 0,
            "num_train_examples": int(len(train_values)),
            "num_calibration_examples": 0,
            "num_averaged_families": int(len(families)),
            **_capacity_fields(global_selection),
        }
        _record_probe_metric(
            metrics_rows,
            examples,
            base=base,
            probe=global_probe,
            x=global_test_x,
            y=global_test_values,
            save_examples=True,
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    direction_frame = pd.DataFrame(direction_rows)
    atomic_write_csv(layer_dir / "metrics.csv", metrics_frame)
    atomic_write_csv(layer_dir / "direction_pairs.csv", direction_frame)
    atomic_write_csv(layer_dir / "prediction_examples.csv", pd.DataFrame(examples))
    _atomic_save_safetensors(layer_dir / "native_directions.safetensors", direction_tensors)
    atomic_write_json(
        layer_dir / "metadata.json",
        {
            **common,
            "metric_rows": int(len(metrics_frame)),
            "direction_pair_rows": int(len(direction_frame)),
            "prediction_example_rows": int(len(examples)),
        },
    )
    atomic_write_text(layer_dir / "DONE", "complete\n")


def _write_assignment_audit(cfg: dict[str, Any], output_dir: Path, panel: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    path = output_dir / "analysis" / "balanced_pooled_assignments.csv"
    if path.is_file():
        return
    families = [str(value) for value in cfg["families"]]
    values = assert_value_alignment((family, panel[(family, "train")]["values"]) for family in families)
    rows: list[pd.DataFrame] = []
    for target_idx, target_family in enumerate(families):
        source_families = [family for family in families if family != target_family]
        for replicate in range(int(cfg["pooled_assignment_replicates"])):
            assignment_seed = int(cfg["seed"]) + (1009 * target_idx) + replicate
            _, _, audit = build_balanced_pooled_matrix(
                values=values,
                family_arrays={family: np.zeros((len(values), 1), dtype=np.float32) for family in source_families},
                seed=assignment_seed,
            )
            audit.insert(0, "assignment_seed", assignment_seed)
            audit.insert(0, "replicate", replicate)
            audit.insert(0, "heldout_family", target_family)
            rows.append(audit)
    atomic_write_csv(path, pd.concat(rows, ignore_index=True))


def finalize_analysis(
    *, cfg: dict[str, Any], model_alias: str, model_meta: dict[str, Any], output_dir: Path
) -> None:
    layer_root = output_dir / "analysis" / "layers"
    expected = [layer_root / f"layer_{position:02d}" for position in range(len(cfg["relative_layers"]))]
    incomplete = [path for path in expected if not (path / "DONE").is_file()]
    if incomplete:
        raise RuntimeError(f"Cannot finalize; incomplete layer analyses: {[path.name for path in incomplete]}")
    metrics = pd.concat([pd.read_csv(path / "metrics.csv") for path in expected], ignore_index=True)
    directions = pd.concat([pd.read_csv(path / "direction_pairs.csv") for path in expected], ignore_index=True)
    examples = pd.concat([pd.read_csv(path / "prediction_examples.csv") for path in expected], ignore_index=True)
    bootstrap = clustered_direction_bootstrap(
        directions,
        group_columns=["model_alias", "architecture"],
        replicates=int(cfg["bootstrap_replicates"]),
        confidence_level=float(cfg["bootstrap_confidence_level"]),
        seed=int(cfg["seed"]),
    )
    analysis_dir = output_dir / "analysis"
    atomic_write_csv(analysis_dir / "metrics.csv", metrics)
    atomic_write_csv(analysis_dir / "direction_pairs.csv", directions)
    atomic_write_csv(analysis_dir / "direction_bootstrap.csv", bootstrap)
    atomic_write_csv(analysis_dir / "prediction_examples.csv", examples)
    atomic_write_json(
        analysis_dir / "summary.json",
        {
            "model_alias": model_alias,
            "model_revision": model_meta["model_revision"],
            "metric_rows": int(len(metrics)),
            "direction_pair_rows": int(len(directions)),
            "bootstrap_rows": int(len(bootstrap)),
            "prediction_example_rows": int(len(examples)),
            "relative_layer_count": int(metrics["relative_layer"].nunique()),
            "questions": sorted(metrics["question"].astype(str).unique().tolist()),
            "direction_question": "independently_fitted_direction_recurrence",
            "train_count_values": sorted(metrics["num_train_examples"].astype(int).unique().tolist()),
            "test_count_values": sorted(metrics["num_test_examples"].astype(int).unique().tolist()),
        },
    )
    atomic_write_text(output_dir / "DONE", "complete\n")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    if int(cfg.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported config schema version.")
    if args.model not in cfg["models"]:
        raise ValueError(f"Unknown model alias: {args.model}")
    if tuple(cfg["architectures"]) != ARCHITECTURES:
        raise ValueError(f"Registered architectures must match {ARCHITECTURES}.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" and args.mode in {"collect", "all"}:
        raise RuntimeError("Authoritative CUDA collection requires CUBLAS_WORKSPACE_CONFIG=:4096:8.")
    set_global_seed(int(cfg["seed"]), deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    output_root = Path(args.output_root or cfg["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_id, run_payload = _run_identity(
        cfg=cfg,
        config_path=config_path,
        model_alias=args.model,
        dtype_override=args.dtype,
        smoke_values_per_partition=args.smoke_values_per_partition,
    )
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "run.json",
        {
            "run_id": run_id,
            "run_payload": run_payload,
            "config": cfg,
            "git": get_git_metadata(ROOT),
            "runtime": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
            },
        },
    )

    if args.mode in {"collect", "all"}:
        model_meta = collect_model_aggregates(
            cfg=cfg,
            model_alias=args.model,
            output_dir=output_dir,
            device=args.device,
            dtype_override=args.dtype,
            force=bool(args.force_collect),
            smoke_values_per_partition=args.smoke_values_per_partition,
        )
    else:
        metadata_path = output_dir / "model_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Collection metadata is missing: {metadata_path}")
        model_meta = json.loads(metadata_path.read_text(encoding="utf-8"))

    if args.mode in {"analyze", "all"}:
        panel = _load_aggregate_panel(cfg, output_dir)
        _validate_partition_counts(panel, [str(value) for value in cfg["families"]], args.smoke_values_per_partition)
        _write_assignment_audit(cfg, output_dir, panel)
        selections = selected_capacity_table(ROOT / str(cfg["selection_csv"]), args.model)
        positions = (
            [int(args.layer_position)]
            if args.layer_position is not None
            else list(range(len(model_meta["relative_layers"])))
        )
        for position in positions:
            if position < 0 or position >= len(model_meta["relative_layers"]):
                raise ValueError(f"Invalid layer position {position}.")
            print(f"[{args.model}] analyzing layer position {position}/{len(model_meta['relative_layers']) - 1}", flush=True)
            analyze_layer(
                cfg=cfg,
                model_alias=args.model,
                model_meta=model_meta,
                output_dir=output_dir,
                panel=panel,
                selections=selections,
                layer_position=position,
                force=bool(args.force_analyze),
            )
        if args.layer_position is None:
            finalize_analysis(cfg=cfg, model_alias=args.model, model_meta=model_meta, output_dir=output_dir)
    elif args.mode == "finalize":
        finalize_analysis(cfg=cfg, model_alias=args.model, model_meta=model_meta, output_dir=output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
