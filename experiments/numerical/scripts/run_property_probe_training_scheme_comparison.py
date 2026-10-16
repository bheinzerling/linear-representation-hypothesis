#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    SOURCE_RUN,
    load_source_config,
    load_split_activations,
)
from scripts.run_property_subspace_representation_synthesis import (  # noqa: E402
    ALL_PROPERTIES,
    run_behavior,
)
from scripts.run_source_value_probe_capacity_sweep import _load_partition_rows  # noqa: E402
from src.model_loader import load_tokenizer  # noqa: E402
from src.property_probe_training_schemes import (  # noqa: E402
    fit_competitive_deflation_basis,
    fit_independent_posthoc_svd_basis,
    fit_joint_multitask_svd_basis,
    property_readout_metrics,
    select_semantic_basis_decoder,
    subspace_similarity,
)
from src.property_subspace_synthesis import (  # noqa: E402
    fit_property_feature_encoder,
    geometric_detail,
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


DEFAULT_OUTPUT = ROOT / "out/property_probe_training_scheme_comparison/llama31_8b_layer10_v1"
SEED = 20260722
SPLITS = tuple(f"split_{index:02d}" for index in range(10))
METHODS = ("independent_posthoc_svd", "competitive_deflation", "joint_multitask_svd")
PROBE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
DECODER_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BASE_RANKS = (1, 2, 4, 8, 12, 16, 24, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent post-hoc SVD, globally competitive deflation, and joint multitask "
            "SVD property-probe training across ten value-disjoint splits."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", default="all", help="Comma-separated split ids or 'all'.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods or 'all'.")
    parser.add_argument("--skip-behavior", action="store_true")
    parser.add_argument("--behavior-only", action="store_true")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
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


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in (SEED, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _save_state(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in payload.items()})
    temporary.replace(path)


def _load_split(
    *,
    split_name: str,
    cfg: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    resolved: Mapping[str, Any],
    tokenizer,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, pd.DataFrame]]:
    activations = load_split_activations(
        split_name=split_name,
        cfg=cfg,
        model_entry=model_entry,
        resolved=resolved,
        tokenizer=tokenizer,
    )
    rows = _load_partition_rows(Path(cfg["split_dataset_dir"]), split_name, str(cfg["family"]))
    for partition in ("train", "dev", "test"):
        values, matrix = activations[partition]
        frame_values = rows[partition]["source_value"].to_numpy(dtype=np.int64)
        if not np.array_equal(np.asarray(values, dtype=np.int64), frame_values):
            raise ValueError(f"Row/activation misalignment in {split_name}/{partition}.")
        if len(values) != len(matrix):
            raise ValueError(f"Value/activation mismatch in {split_name}/{partition}.")
    return activations, rows


def _method_training(
    *,
    method: str,
    train_centered: np.ndarray,
    train_targets: np.ndarray,
    dev_centered: np.ndarray,
    dev_targets: np.ndarray,
    encoder,
    max_rank: int,
    split_name: str,
):
    if method == "independent_posthoc_svd":
        return fit_independent_posthoc_svd_basis(
            train_x=train_centered,
            train_y=train_targets,
            dev_x=dev_centered,
            dev_y=dev_targets,
            blocks=encoder.blocks,
            alphas=PROBE_ALPHAS,
            max_rank=max_rank,
        )
    if method == "joint_multitask_svd":
        return fit_joint_multitask_svd_basis(
            train_x=train_centered,
            train_y=train_targets,
            dev_x=dev_centered,
            dev_y=dev_targets,
            alphas=PROBE_ALPHAS,
            max_rank=max_rank,
        )
    if method == "competitive_deflation":
        rng = np.random.default_rng(_seed(split_name, method, "inner_split"))
        permutation = rng.permutation(len(train_centered))
        selection_n = max(1, int(round(0.2 * len(permutation))))
        selection_indices = np.sort(permutation[:selection_n])
        fit_indices = np.sort(permutation[selection_n:])
        return fit_competitive_deflation_basis(
            train_x=train_centered,
            train_y=train_targets,
            fit_indices=fit_indices,
            selection_indices=selection_indices,
            blocks=encoder.blocks,
            max_rank=max_rank,
        )
    raise ValueError(f"Unknown probe-training method: {method}")


def run_geometry(
    *,
    output_dir: Path,
    selected_splits: Sequence[str],
    selected_methods: Sequence[str],
    cfg: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    resolved: Mapping[str, Any],
    tokenizer,
    force: bool,
) -> None:
    required = [
        output_dir / "geometry_summary.csv",
        output_dir / "geometry_detail.csv",
        output_dir / "decoder_path.csv",
        output_dir / "probe_readout.csv",
        output_dir / "basis_training_diagnostics.csv",
        output_dir / "basis_stability.csv",
        output_dir / "split_summary.csv",
    ]
    if all(path.exists() for path in required) and not force:
        return
    geometry_summary_rows: list[dict[str, Any]] = []
    geometry_detail_frames: list[pd.DataFrame] = []
    decoder_path_frames: list[pd.DataFrame] = []
    readout_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    split_summary_rows: list[dict[str, Any]] = []
    bases_by_variant: dict[tuple[str, str, str], np.ndarray] = {}
    for split_name in selected_splits:
        print(f"geometry {split_name}", flush=True)
        activations, rows = _load_split(
            split_name=split_name,
            cfg=cfg,
            model_entry=model_entry,
            resolved=resolved,
            tokenizer=tokenizer,
        )
        train_values, train_x_raw = activations["train"]
        dev_values, dev_x_raw = activations["dev"]
        test_values, test_x_raw = activations["test"]
        train_x = np.asarray(train_x_raw, dtype=np.float64)
        dev_x = np.asarray(dev_x_raw, dtype=np.float64)
        test_x = np.asarray(test_x_raw, dtype=np.float64)
        center = train_x.mean(axis=0)
        train_centered = train_x - center[None, :]
        dev_centered = dev_x - center[None, :]
        test_centered = test_x - center[None, :]
        encoder = fit_property_feature_encoder(
            train_values=np.asarray(train_values, dtype=np.int64),
            property_ids=ALL_PROPERTIES,
        )
        train_targets = encoder.transform(np.asarray(train_values, dtype=np.int64))
        dev_targets = encoder.transform(np.asarray(dev_values, dtype=np.int64))
        target_rank = int(np.linalg.matrix_rank(train_targets))
        rank_candidates = tuple(sorted({*BASE_RANKS, target_rank}))
        state_payload: dict[str, np.ndarray] = {
            "values": np.asarray(test_values, dtype=np.int64),
            "actual": np.asarray(test_x, dtype=np.float32),
            "train_center": np.asarray(center, dtype=np.float32),
        }
        for method in selected_methods:
            trained = _method_training(
                method=method,
                train_centered=train_centered,
                train_targets=train_targets,
                dev_centered=dev_centered,
                dev_targets=dev_targets,
                encoder=encoder,
                max_rank=target_rank,
                split_name=split_name,
            )
            decoder, path_rows = select_semantic_basis_decoder(
                method=method,
                train_activations=train_x,
                train_values=np.asarray(train_values, dtype=np.int64),
                dev_activations=dev_x,
                dev_values=np.asarray(dev_values, dtype=np.int64),
                basis=trained.basis,
                feature_encoder=encoder,
                ranks=rank_candidates,
                alphas=DECODER_ALPHAS,
            )
            synthesis = decoder.synthesize(np.asarray(test_values, dtype=np.int64))
            state_payload[f"synthesis__{method}"] = np.asarray(synthesis, dtype=np.float32)
            state_payload[f"basis__{method}"] = np.asarray(decoder.basis, dtype=np.float32)
            state_payload[f"full_basis__{method}"] = np.asarray(trained.basis, dtype=np.float32)
            bases_by_variant[(split_name, method, "selected_decoder_basis")] = np.asarray(
                decoder.basis, dtype=np.float64
            )
            bases_by_variant[(split_name, method, "full_trained_basis")] = np.asarray(
                trained.basis, dtype=np.float64
            )
            path = pd.DataFrame(path_rows)
            path.insert(0, "split", split_name)
            decoder_path_frames.append(path)
            for diagnostic in trained.diagnostics:
                diagnostic_rows.append(
                    {"split": split_name, "method": method, **diagnostic}
                )
            detail = pd.DataFrame(
                geometric_detail(actual_activations=test_x, synthesis=synthesis, center=decoder.center)
            )
            detail.insert(0, "source_value", np.asarray(test_values, dtype=np.int64))
            detail.insert(0, "method", method)
            detail.insert(0, "split", split_name)
            geometry_detail_frames.append(detail)
            geometry_summary_rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "full_basis_rank": trained.rank,
                    "selected_rank": decoder.rank,
                    "selected_decoder_alpha": decoder.alpha,
                    "target_feature_dim": encoder.feature_dim,
                    "target_feature_rank": target_rank,
                    **summarize_geometric_fit(
                        actual_activations=test_x,
                        synthesis=synthesis,
                        center=decoder.center,
                        pair_seed=_seed(split_name, method, "pairs"),
                    ),
                }
            )
            for basis_variant, readout_basis in (
                ("selected_decoder_basis", decoder.basis),
                ("full_trained_basis", trained.basis),
            ):
                property_rows = property_readout_metrics(
                    train_activations=train_centered,
                    train_values=np.asarray(train_values, dtype=np.int64),
                    dev_activations=dev_centered,
                    dev_values=np.asarray(dev_values, dtype=np.int64),
                    test_activations=test_centered,
                    test_values=np.asarray(test_values, dtype=np.int64),
                    basis=readout_basis,
                    feature_encoder=encoder,
                    alphas=DECODER_ALPHAS,
                )
                readout_rows.extend(
                    {
                        "split": split_name,
                        "method": method,
                        "basis_variant": basis_variant,
                        "basis_rank": int(readout_basis.shape[1]),
                        "selected_rank": decoder.rank,
                        **row,
                    }
                    for row in property_rows
                )
        _save_state(output_dir / "states" / f"{split_name}.npz", state_payload)
        split_summary_rows.append(
            {
                "split": split_name,
                "train_n": len(train_values),
                "dev_n": len(dev_values),
                "test_n": len(test_values),
                "train_dev_overlap": len(set(train_values.tolist()) & set(dev_values.tolist())),
                "train_test_overlap": len(set(train_values.tolist()) & set(test_values.tolist())),
                "dev_test_overlap": len(set(dev_values.tolist()) & set(test_values.tolist())),
                "target_feature_dim": encoder.feature_dim,
                "target_feature_rank": target_rank,
                "test_prompt_unique_n": int(rows["test"]["prompt"].nunique()),
            }
        )
    stability_rows: list[dict[str, Any]] = []
    for method in selected_methods:
        for basis_variant in ("selected_decoder_basis", "full_trained_basis"):
            for left_index, left_split in enumerate(selected_splits):
                for right_split in selected_splits[left_index + 1 :]:
                    stability_rows.append(
                        {
                            "method": method,
                            "basis_variant": basis_variant,
                            "left_split": left_split,
                            "right_split": right_split,
                            **subspace_similarity(
                                bases_by_variant[(left_split, method, basis_variant)],
                                bases_by_variant[(right_split, method, basis_variant)],
                            ),
                        }
                    )
    atomic_write_csv(output_dir / "geometry_summary.csv", pd.DataFrame(geometry_summary_rows))
    atomic_write_csv(output_dir / "geometry_detail.csv", pd.concat(geometry_detail_frames, ignore_index=True))
    atomic_write_csv(output_dir / "decoder_path.csv", pd.concat(decoder_path_frames, ignore_index=True))
    atomic_write_csv(output_dir / "probe_readout.csv", pd.DataFrame(readout_rows))
    atomic_write_csv(output_dir / "basis_training_diagnostics.csv", pd.DataFrame(diagnostic_rows))
    atomic_write_csv(output_dir / "basis_stability.csv", pd.DataFrame(stability_rows))
    atomic_write_csv(output_dir / "split_summary.csv", pd.DataFrame(split_summary_rows))


def _prepare_behavior(
    *,
    output_dir: Path,
    selected_splits: Sequence[str],
    selected_methods: Sequence[str],
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for split_name in selected_splits:
        state_path = output_dir / "states" / f"{split_name}.npz"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing geometry state: {state_path}")
        state = np.load(state_path)
        rows = _load_partition_rows(Path(cfg["split_dataset_dir"]), split_name, str(cfg["family"]))
        frame = rows["test"].sort_values("source_value").reset_index(drop=True)
        values = np.asarray(state["values"], dtype=np.int64)
        if not np.array_equal(values, frame["source_value"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Saved state and prompt rows are misaligned for {split_name}.")
        prepared[split_name] = {
            "values": values,
            "actual": np.asarray(state["actual"], dtype=np.float64),
            "frame": frame,
            "syntheses": {
                method: np.asarray(state[f"synthesis__{method}"], dtype=np.float64)
                for method in selected_methods
            },
        }
    return prepared


def write_aggregates(output_dir: Path) -> None:
    geometry = pd.read_csv(output_dir / "geometry_summary.csv")
    readout = pd.read_csv(output_dir / "probe_readout.csv")
    behavior = pd.read_csv(output_dir / "behavior_summary.csv")
    readout = readout[readout["basis_variant"] == "selected_decoder_basis"]
    categorical = readout[readout["kind"] == "categorical"]
    magnitude = readout[readout["property_id"] == "magnitude"]
    rows: list[dict[str, Any]] = []
    for method, group in geometry.groupby("method", sort=False):
        behavior_group = behavior[behavior["condition"] == method]
        categorical_group = categorical[categorical["method"] == method]
        magnitude_group = magnitude[magnitude["method"] == method]
        rows.append(
            {
                "method": method,
                "split_count": int(group["split"].nunique()),
                "mean_selected_rank": float(group["selected_rank"].mean()),
                "sd_selected_rank": float(group["selected_rank"].std(ddof=1)),
                "mean_test_fve": float(group["fraction_variance_explained"].mean()),
                "sd_test_fve": float(group["fraction_variance_explained"].std(ddof=1)),
                "mean_centered_cosine": float(group["global_centered_cosine_similarity"].mean()),
                "mean_categorical_balanced_accuracy": float(
                    categorical_group["test_balanced_accuracy"].mean()
                ),
                "mean_magnitude_r2": float(magnitude_group["test_r2"].mean()),
                "mean_task_accuracy": float(behavior_group["synthesis_task_accuracy"].mean()),
                "sd_task_accuracy": float(behavior_group["synthesis_task_accuracy"].std(ddof=1)),
                "mean_full_kl": float(
                    behavior_group["mean_full_kl_original_to_synthesis"].mean()
                ),
                "mean_target_logprob_delta": float(
                    behavior_group["mean_target_logprob_delta"].mean()
                ),
            }
        )
    atomic_write_csv(output_dir / "method_aggregate.csv", pd.DataFrame(rows))


def main() -> None:
    args = parse_args()
    if args.skip_behavior and args.behavior_only:
        raise ValueError("--skip-behavior and --behavior-only are mutually exclusive.")
    set_global_seed(SEED)
    selected_splits = _selected(args.splits, SPLITS)
    selected_methods = _selected(args.methods, METHODS)
    output_dir = ensure_dir(args.output_dir)
    cfg, model_entry, resolved = load_source_config(args.source_run_dir)
    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=str(resolved["model_revision"]),
    )
    if not args.behavior_only:
        run_geometry(
            output_dir=output_dir,
            selected_splits=selected_splits,
            selected_methods=selected_methods,
            cfg=cfg,
            model_entry=model_entry,
            resolved=resolved,
            tokenizer=tokenizer,
            force=args.force_geometry,
        )
    if not args.skip_behavior:
        prepared = _prepare_behavior(
            output_dir=output_dir,
            selected_splits=selected_splits,
            selected_methods=selected_methods,
            cfg=cfg,
        )
        run_behavior(
            output_dir=output_dir,
            prepared=prepared,
            selected_regimes=selected_splits,
            selected_conditions=selected_methods,
            cfg=cfg,
            model_entry=model_entry,
            resolved=resolved,
            eval_device=args.eval_device,
            eval_batch_size=args.eval_batch_size,
            max_rows=None,
            force=args.force_behavior,
        )
        write_aggregates(output_dir)
    atomic_write_json(
        output_dir / "metadata.json",
        {
            "schema_version": 2,
            "seed": SEED,
            "splits": list(selected_splits),
            "methods": list(selected_methods),
            "properties": list(ALL_PROPERTIES),
            "probe_alphas": list(PROBE_ALPHAS),
            "decoder_alphas": list(DECODER_ALPHAS),
            "rank_candidates": list(BASE_RANKS),
            "competitive_inner_selection_fraction": 0.2,
            "model_id": str(model_entry["hf_id"]),
            "model_revision": str(resolved["model_revision"]),
            "layer": int(cfg["intervention_layer_idx"]),
            "activation_cache_root": str(cfg["activation_cache_root"]),
            "activation_cache_policy": "require_hit",
            "runtime": runtime_metadata(),
        },
    )
    atomic_write_text(output_dir / "DONE", "ok\n")
    print(output_dir)


if __name__ == "__main__":
    main()
