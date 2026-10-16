#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file as load_safetensors_numpy
from safetensors.numpy import save_file as save_safetensors_numpy
from tqdm.auto import tqdm
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activation_steering import LayerForwardHook, hidden_states_hook
from src.magnitude_intervention import evaluate_probe_split, fit_magnitude_probe
from src.model_loader import load_causal_lm, load_tokenizer
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text, get_git_metadata, load_jsonl, load_yaml, set_global_seed


DEFAULT_CONFIG = ROOT / "configs" / "prompt_style_context_centroid_relative_layers.yaml"
PARTITIONS = ("train", "dev", "test")
ARCHITECTURES = ("circular", "pls", "linear")
CONDITIONS = ("single_context", "context_centroid")
INTEGER_TOKEN_RE = re.compile(r"^\s*([0-9]{1,3})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the matched one-context versus context-centroid prompt-style transfer sweep.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument("--limit-per-shard", type=int, default=None)
    return parser.parse_args()


def stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_rel_to_abs(layer_rel: float, n_layers: int) -> int:
    resolved = int(Decimal(str(float(layer_rel) * int(n_layers))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(0, min(int(n_layers) - 1, resolved))


def load_partition_frame(dataset_root: Path, split_name: str, family: str, partition: str, limit: int | None) -> pd.DataFrame:
    path = dataset_root / split_name / family / f"{partition}.jsonl"
    rows = load_jsonl(path)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["source_value", "variant_idx", "prompt"], kind="stable").reset_index(drop=True)
    if limit is not None:
        frame = frame.head(int(limit)).copy()
    required = {"prompt", "token_idx", "source_value", "variant_idx"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame


def validate_panel(frame: pd.DataFrame, *, require_complete: bool = True) -> None:
    if frame.duplicated(["source_value", "variant_idx"]).any():
        raise ValueError("Duplicate source_value/variant_idx rows in prompt panel.")
    counts = frame.groupby("source_value")["variant_idx"].nunique()
    if require_complete and (counts.min() < 1 or counts.max() > 20):
        raise ValueError(f"Expected 1--20 available variants per value, observed {int(counts.min())}--{int(counts.max())}.")
    if not frame.groupby("source_value")["variant_idx"].apply(lambda values: 0 in set(values.astype(int))).all():
        raise ValueError("Every independent value must contain variant 0.")


def audit_token_loci(tokenizer: Any, frame: pd.DataFrame) -> None:
    prompts = frame["prompt"].astype(str).tolist()
    encoded = tokenizer(prompts, add_special_tokens=False, padding=False)["input_ids"]
    for row, input_ids in zip(frame.itertuples(index=False), encoded, strict=True):
        token_idx = int(row.token_idx)
        if token_idx < 0 or token_idx >= len(input_ids):
            raise ValueError(f"Token index {token_idx} is out of range for source value {row.source_value}.")
        decoded = tokenizer.decode([int(input_ids[token_idx])], skip_special_tokens=False)
        match = INTEGER_TOKEN_RE.fullmatch(decoded)
        if match is None or int(match.group(1)) != int(row.source_value):
            raise ValueError(
                f"Tokenizer/locus mismatch: source={row.source_value}, token_idx={token_idx}, decoded={decoded!r}."
            )


def aggregate_context_activations(
    frame: pd.DataFrame,
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(activations)
    if matrix.ndim != 3 or matrix.shape[0] != len(frame):
        raise ValueError(f"Expected row x layer x feature activations, got {matrix.shape} for {len(frame)} rows.")
    values = np.sort(frame["source_value"].astype(int).unique())
    source = frame["source_value"].to_numpy(dtype=np.int64)
    variants = frame["variant_idx"].to_numpy(dtype=np.int64)
    singles: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    for value in values:
        idx = np.flatnonzero(source == value)
        local_variants = variants[idx]
        if len(np.unique(local_variants)) != len(local_variants):
            raise ValueError(f"Duplicate context variant for source value {value}.")
        zero = idx[local_variants == 0]
        if len(zero) != 1:
            raise ValueError(f"Expected one variant-0 row for source value {value}, got {len(zero)}.")
        singles.append(np.asarray(matrix[zero[0]], dtype=np.float32))
        centroids.append(np.asarray(matrix[idx], dtype=np.float32).mean(axis=0, dtype=np.float32))
        counts.append(len(idx))
    return values, np.stack(singles), np.stack(centroids), np.asarray(counts, dtype=np.int64)


def collect_multi_layer_activations(
    *, model: Any, tokenizer: Any, frame: pd.DataFrame, layer_indices: list[int], batch_size: int, device: str
) -> np.ndarray:
    prompts = frame["prompt"].astype(str).tolist()
    token_indices = frame["token_idx"].astype(int).tolist()
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(frame), int(batch_size)), desc="activation batches", leave=False):
        stop = min(len(frame), start + int(batch_size))
        encoded = tokenizer(
            prompts[start:stop], return_tensors="pt", padding=True, add_special_tokens=False
        )
        model_kwargs = {key: value.to(device) for key, value in encoded.items()}
        local_token_indices = token_indices[start:stop]

        class Capture:
            def __init__(self) -> None:
                self.value: np.ndarray | None = None

            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                idx = torch.arange(output.shape[0], device=output.device)
                token_idx = torch.as_tensor(local_token_indices, dtype=torch.long, device=output.device)
                self.value = output[idx, token_idx].detach().to(dtype=torch.float16, device="cpu").numpy()
                return output

        hook = LayerForwardHook(model)
        captures = [Capture() for _ in layer_indices]
        for capture, layer_idx in zip(captures, layer_indices, strict=True):
            hook.attach(capture, layer_idx=int(layer_idx))
        try:
            with torch.inference_mode():
                outputs = model(**model_kwargs, use_cache=False)
            del outputs
        finally:
            hook.remove()
        if any(capture.value is None for capture in captures):
            raise RuntimeError("At least one registered layer hook did not capture activations.")
        chunks.append(np.stack([capture.value for capture in captures], axis=1))
        del model_kwargs, captures
    return np.concatenate(chunks, axis=0)


def save_aggregate_shard(
    path: Path, *, values: np.ndarray, single: np.ndarray, centroid: np.ndarray, context_counts: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_safetensors_numpy(
        {
            "values": values.astype(np.int64),
            "single_context": single.astype(np.float16),
            "context_centroid": centroid.astype(np.float16),
            "context_counts": context_counts.astype(np.int64),
        },
        str(tmp),
    )
    os.replace(tmp, path)


def condition_arrays(path: Path, condition: str, layer_position: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_safetensors_numpy(str(path))
    return arrays["values"].astype(np.int64), arrays[condition][:, int(layer_position), :].astype(np.float32)


def selected_capacity_table(path: Path, model_alias: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["model_alias"].eq(model_alias)
        & frame["evaluation_partition"].eq("dev")
        & frame["is_native_style"].astype(bool)
        & frame["architecture"].isin(ARCHITECTURES)
    ].copy()
    keys = ["relative_layer", "train_family", "architecture"]
    if len(selected) != 11 * 5 * 3 or selected.duplicated(keys).any():
        raise ValueError(f"Expected one registered capacity for every layer/family/architecture cell; got {len(selected)}.")
    return selected.sort_values(keys).reset_index(drop=True)


def fit_config_from_selection(row: pd.Series) -> tuple[str, dict[str, Any]]:
    architecture = str(row["architecture"])
    value = row["control_value"]
    if architecture == "linear":
        return "linear", {"ridge_lambda": float(value)}
    if architecture == "circular":
        return "circular_reg", {"basis_dim": int(value)}
    if architecture == "pls":
        return "pls", {"n_components": int(value), "scale": False}
    raise ValueError(f"Unsupported architecture: {architecture}")


def collect_model_aggregates(
    *, cfg: dict[str, Any], model_alias: str, output_dir: Path, device: str, dtype_override: str | None,
    force: bool, limit_per_shard: int | None
) -> dict[str, Any]:
    models_cfg = load_yaml(cfg["models_config"])
    base_model = {**models_cfg.get("default", {}), **models_cfg["models"][model_alias]}
    run_model = {**base_model, **cfg["models"][model_alias]}
    if dtype_override is not None:
        run_model["dtype"] = str(dtype_override)
    model_id = str(run_model["hf_id"])
    revision = str(run_model["revision"])
    trust_remote_code = bool(run_model.get("trust_remote_code", False))
    auto_config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    n_layers = int(auto_config.num_hidden_layers)
    relative_layers = [float(value) for value in cfg["relative_layers"]]
    layer_indices = [layer_rel_to_abs(value, n_layers) for value in relative_layers]
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"Relative layers are not unique for {model_alias}: {layer_indices}")

    aggregate_dir = output_dir / "aggregates"
    expected = [aggregate_dir / family / f"{partition}.safetensors" for family in cfg["families"] for partition in PARTITIONS]
    if all(path.is_file() for path in expected) and not force:
        return {
            "model_id": model_id, "revision": revision, "n_layers": n_layers,
            "relative_layers": relative_layers, "layer_indices": layer_indices,
            "batch_size": int(run_model["batch_size"]), "dtype": str(run_model.get("dtype", "float32")),
        }

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code, revision=revision)
    dataset_root = ROOT / str(cfg["split_dataset_dir"])
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for family in cfg["families"]:
        for partition in PARTITIONS:
            frame = load_partition_frame(dataset_root, str(cfg["split_name"]), str(family), partition, limit_per_shard)
            validate_panel(frame, require_complete=limit_per_shard is None)
            audit_token_loci(tokenizer, frame)
            frames[(str(family), partition)] = frame

    loaded = load_causal_lm(
        model_id,
        device=device,
        dtype=str(run_model.get("dtype", "float32")),
        trust_remote_code=trust_remote_code,
        attn_implementation=run_model.get("attn_implementation"),
        revision=revision,
    )
    try:
        for family in cfg["families"]:
            for partition in PARTITIONS:
                path = aggregate_dir / str(family) / f"{partition}.safetensors"
                if path.is_file() and not force:
                    continue
                frame = frames[(str(family), partition)]
                activations = collect_multi_layer_activations(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    frame=frame,
                    layer_indices=layer_indices,
                    batch_size=int(run_model["batch_size"]),
                    device=device,
                )
                values, single, centroid, counts = aggregate_context_activations(frame, activations)
                save_aggregate_shard(path, values=values, single=single, centroid=centroid, context_counts=counts)
                coverage = pd.DataFrame(
                    {
                        "family": str(family),
                        "partition": partition,
                        "source_value": values,
                        "available_context_count": counts,
                    }
                )
                coverage_path = output_dir / "context_coverage" / str(family) / f"{partition}.csv"
                coverage_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_csv(coverage_path, coverage)
                del activations, single, centroid
                gc.collect()
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "model_id": model_id, "revision": revision, "n_layers": n_layers,
        "relative_layers": relative_layers, "layer_indices": layer_indices,
        "batch_size": int(run_model["batch_size"]), "dtype": str(run_model.get("dtype", "float32")),
    }


def fit_model_probes(*, cfg: dict[str, Any], model_alias: str, output_dir: Path, model_meta: dict[str, Any], force: bool) -> Path:
    metrics_path = output_dir / "selected_transfer_summary.csv"
    if metrics_path.is_file() and not force:
        return metrics_path
    selection_path = ROOT / str(cfg["selection_csv"])
    selections = selected_capacity_table(selection_path, model_alias)
    shard_paths = {
        (str(family), partition): output_dir / "aggregates" / str(family) / f"{partition}.safetensors"
        for family in cfg["families"] for partition in PARTITIONS
    }
    rows: list[dict[str, Any]] = []
    started = time.time()
    for layer_position, (relative_layer, layer_idx) in enumerate(zip(model_meta["relative_layers"], model_meta["layer_indices"], strict=True)):
        local_selection = selections.loc[np.isclose(selections["relative_layer"], float(relative_layer))]
        for condition in CONDITIONS:
            cached = {
                (family, partition): condition_arrays(shard_paths[(family, partition)], condition, layer_position)
                for family in cfg["families"] for partition in PARTITIONS
            }
            for train_family in cfg["families"]:
                train_values, train_x = cached[(str(train_family), "train")]
                for architecture in ARCHITECTURES:
                    match = local_selection.loc[
                        local_selection["train_family"].eq(str(train_family))
                        & local_selection["architecture"].eq(architecture)
                    ]
                    if len(match) != 1:
                        raise ValueError(f"Missing registered capacity for {model_alias}/{relative_layer}/{train_family}/{architecture}.")
                    selected = match.iloc[0]
                    fit_architecture, fit_config = fit_config_from_selection(selected)
                    fit_started = time.time()
                    probe = fit_magnitude_probe(
                        architecture=fit_architecture,
                        train_x=train_x,
                        train_values=train_values,
                        config=fit_config,
                        value_min=0,
                        value_max=999,
                    )
                    for test_family in cfg["families"]:
                        test_values, test_x = cached[(str(test_family), "test")]
                        _, summary = evaluate_probe_split(
                            probe=probe, representations=test_x, values=test_values, split_name="test"
                        )
                        rows.append(
                            {
                                "model_alias": model_alias,
                                "model_id": model_meta["model_id"],
                                "model_revision": model_meta["revision"],
                                "condition": condition,
                                "relative_layer": float(relative_layer),
                                "layer_idx": int(layer_idx),
                                "n_layers": int(model_meta["n_layers"]),
                                "train_family": str(train_family),
                                "test_family": str(test_family),
                                "is_native_style": str(train_family) == str(test_family),
                                "architecture": architecture,
                                "fit_architecture": fit_architecture,
                                "control_name": str(selected["control_name"]),
                                "control_value": float(selected["control_value"]),
                                "capacity_source": "single_context_native_dev_2026_04_20",
                                "num_train_values": int(len(train_values)),
                                "num_test_values": int(len(test_values)),
                                "mean_abs_error": float(summary["mean_abs_error"]),
                                "rmse": float(summary["rmse"]),
                                "accuracy": float(summary["accuracy"]),
                                "fit_elapsed_seconds": float(time.time() - fit_started),
                            }
                        )
                    del probe
            del cached
            gc.collect()
        partial = pd.DataFrame(rows)
        atomic_write_csv(output_dir / "selected_transfer_summary.partial.csv", partial)
    frame = pd.DataFrame(rows)
    expected_rows = len(CONDITIONS) * len(cfg["relative_layers"]) * len(cfg["families"]) * len(ARCHITECTURES) * len(cfg["families"])
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} metric rows, got {len(frame)}.")
    atomic_write_csv(metrics_path, frame)
    mean_cells = (
        frame.groupby(["model_alias", "condition", "architecture", "train_family", "test_family"], as_index=False)
        .agg(mean_over_layers_mae=("mean_abs_error", "mean"), layer_count=("relative_layer", "nunique"))
    )
    atomic_write_csv(output_dir / "mean_over_relative_layers_cells.csv", mean_cells)
    atomic_write_json(
        output_dir / "fit_summary.json",
        {
            "metric_rows": int(len(frame)),
            "mean_cell_rows": int(len(mean_cells)),
            "elapsed_seconds": float(time.time() - started),
            "conditions": list(CONDITIONS),
            "architectures": list(ARCHITECTURES),
        },
    )
    atomic_write_text(output_dir / "DONE", "complete\n")
    return metrics_path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if int(cfg.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported config schema version.")
    if tuple(cfg["conditions"]) != CONDITIONS:
        raise ValueError(f"Registered conditions must be {CONDITIONS}.")
    if args.model not in cfg["models"]:
        raise ValueError(f"Unknown model alias: {args.model}")
    set_global_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=False)
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root or cfg["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_payload = {
        "config_sha256": sha256_file(config_path),
        "model_alias": args.model,
        "limit_per_shard": args.limit_per_shard,
        "dtype_override": args.dtype,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    run_id = f"{args.model}__{stable_hash(run_payload)}"
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
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
            },
        },
    )
    model_meta = collect_model_aggregates(
        cfg=cfg,
        model_alias=args.model,
        output_dir=output_dir,
        device=args.device,
        dtype_override=args.dtype,
        force=bool(args.force_collect),
        limit_per_shard=args.limit_per_shard,
    )
    atomic_write_json(output_dir / "model_metadata.json", model_meta)
    fit_model_probes(cfg=cfg, model_alias=args.model, output_dir=output_dir, model_meta=model_meta, force=bool(args.force_fit))
    print(output_dir)


if __name__ == "__main__":
    main()
