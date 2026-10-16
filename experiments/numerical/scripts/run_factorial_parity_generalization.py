#!/usr/bin/env python3
"""Test whether parity geometry transfers to unseen ones-digit identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    DEFAULT_OUTPUT as BAG_OUTPUT,
    SOURCE_RUN,
)
from scripts.run_figure1_iterative_nullspace_property_probes import (  # noqa: E402
    load_regime_data,
)
from src.factorial_property_probing import (  # noqa: E402
    DigitFactorization,
    apply_raw_feature_transform,
    binary_metrics,
    cluster_bootstrap_binary_metrics,
    conditional_probability,
    contrast_geometry,
    empirical_upper_tail,
    factorization_manifest_rows,
    fit_conditional_logistic,
    parity_factorization,
    random_factorizations,
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
    ROOT / "out/factorial_parity_generalization/llama31_8b_layer10_v1"
)
DESIGN_PATH = (
    ROOT / "experiment-designs/2026-07-29-factorial-parity-generalization.md"
)
SEED = 20260729
RANDOM_CONTROL_COUNT = 200
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
LOGISTIC_TOL = 1e-8
WHITENING_RTOL = 1e-8
BOOTSTRAP_SAMPLES = 2000
EVALUATION_PARTITION = "evaluation"
EVALUATION_NAME = "value_table_0_999"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factorial parity generalization on cached activations."
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--bag-output-dir", type=Path, default=BAG_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-controls", type=int, default=RANDOM_CONTROL_COUNT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _fit_model(
    *,
    features: np.ndarray,
    side: np.ndarray,
    pair_id: np.ndarray,
    seed: int,
):
    model = fit_conditional_logistic(
        features=features,
        side=side,
        pair_id=pair_id,
        c=LOGISTIC_C,
        seed=seed,
        max_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
    )
    iterations = int(np.max(model.n_iter_))
    return model, iterations, bool(iterations < LOGISTIC_MAX_ITER)


def _evaluate_factorization(
    *,
    factorization: DigitFactorization,
    space: str,
    values: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    seed: int,
    save_predictions: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    train_side, train_pair = factorization.labels(values["train"])
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    full_model, full_iterations, full_converged = _fit_model(
        features=features["train"],
        side=train_side,
        pair_id=train_pair,
        seed=seed,
    )
    side, pair_id = factorization.labels(values[EVALUATION_PARTITION])
    probability = conditional_probability(
        full_model,
        features=features[EVALUATION_PARTITION],
        pair_id=pair_id,
    )
    summary_rows.append(
        {
            "factorization_id": factorization.factorization_id,
            "kind": factorization.kind,
            "space": space,
            "evaluation": EVALUATION_NAME,
            "protocol": "ordinary_all_pairs",
            "fold_count": 1,
            "fold_balanced_accuracy_min": np.nan,
            "fold_balanced_accuracy_mean": np.nan,
            "fold_balanced_accuracy_std": np.nan,
            "max_fit_iterations": full_iterations,
            "all_fits_converged": full_converged,
            **binary_metrics(side, probability),
        }
    )

    combined: list[dict[str, Any]] = []
    for held_pair in range(5):
        train_mask = train_pair != held_pair
        model, iterations, converged = _fit_model(
            features=features["train"][train_mask],
            side=train_side[train_mask],
            pair_id=train_pair[train_mask],
            seed=seed + held_pair + 1,
        )
        eval_side, eval_pair = factorization.labels(values[EVALUATION_PARTITION])
        eval_mask = eval_pair == held_pair
        probability = conditional_probability(
            model,
            features=features[EVALUATION_PARTITION][eval_mask],
            pair_id=eval_pair[eval_mask],
        )
        metrics = binary_metrics(eval_side[eval_mask], probability)
        fold_rows.append(
            {
                "factorization_id": factorization.factorization_id,
                "kind": factorization.kind,
                "space": space,
                "evaluation": EVALUATION_NAME,
                "protocol": "held_pair",
                "held_pair": int(held_pair),
                "side_zero_digit": int(factorization.pairs[held_pair][0]),
                "side_one_digit": int(factorization.pairs[held_pair][1]),
                "fit_iterations": iterations,
                "fit_converged": converged,
                **metrics,
            }
        )
        combined.append(
            {
                "truth": np.asarray(eval_side[eval_mask], dtype=np.int64),
                "probability": np.asarray(probability, dtype=np.float64),
                "value": np.asarray(
                    values[EVALUATION_PARTITION][eval_mask], dtype=np.int64
                ),
                "iterations": iterations,
                "converged": converged,
                "fold_balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
    truth = np.concatenate([row["truth"] for row in combined])
    probability = np.concatenate([row["probability"] for row in combined])
    value = np.concatenate([row["value"] for row in combined])
    order = np.argsort(value)
    truth, probability, value = truth[order], probability[order], value[order]
    fold_balanced = np.asarray(
        [row["fold_balanced_accuracy"] for row in combined], dtype=np.float64
    )
    summary_rows.append(
        {
            "factorization_id": factorization.factorization_id,
            "kind": factorization.kind,
            "space": space,
            "evaluation": EVALUATION_NAME,
            "protocol": "held_pair",
            "fold_count": 5,
            "fold_balanced_accuracy_min": float(np.min(fold_balanced)),
            "fold_balanced_accuracy_mean": float(np.mean(fold_balanced)),
            "fold_balanced_accuracy_std": float(np.std(fold_balanced)),
            "max_fit_iterations": int(max(row["iterations"] for row in combined)),
            "all_fits_converged": bool(
                all(row["converged"] for row in combined)
            ),
            **binary_metrics(truth, probability),
        }
    )
    if save_predictions:
        pair_by_value = factorization.labels(value)[1]
        prediction_rows.extend(
            {
                "factorization_id": factorization.factorization_id,
                "kind": factorization.kind,
                "space": space,
                "evaluation": EVALUATION_NAME,
                "protocol": "held_pair",
                "value": int(current_value),
                "units_digit": int(current_value % 10),
                "held_pair": int(current_pair),
                "truth": int(current_truth),
                "probability_side_one": float(current_probability),
                "prediction": int(current_probability >= 0.5),
                "prefix_cluster": int(current_value // 10),
            }
            for current_value, current_pair, current_truth, current_probability in zip(
                value, pair_by_value, truth, probability, strict=True
            )
        )
    return fold_rows, summary_rows, prediction_rows


def _strict_fold_whitening(
    *,
    factorization: DigitFactorization,
    values: Mapping[str, np.ndarray],
    activations: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_side, train_pair = factorization.labels(values["train"])
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for held_pair in range(5):
        train_mask = train_pair != held_pair
        whitening = fit_compact_whitening(
            activations["train"][train_mask],
            relative_tolerance=WHITENING_RTOL,
        )
        train_features = whitening.transform(activations["train"][train_mask])
        model, iterations, converged = _fit_model(
            features=train_features,
            side=train_side[train_mask],
            pair_id=train_pair[train_mask],
            seed=seed + held_pair + 100,
        )
        eval_side, eval_pair = factorization.labels(values[EVALUATION_PARTITION])
        eval_mask = eval_pair == held_pair
        probability = conditional_probability(
            model,
            features=whitening.transform(
                activations[EVALUATION_PARTITION][eval_mask]
            ),
            pair_id=eval_pair[eval_mask],
        )
        metrics = binary_metrics(eval_side[eval_mask], probability)
        fold_rows.append(
            {
                "factorization_id": factorization.factorization_id,
                "kind": factorization.kind,
                "space": "fold_whitened",
                "evaluation": EVALUATION_NAME,
                "protocol": "held_pair",
                "held_pair": int(held_pair),
                "side_zero_digit": int(factorization.pairs[held_pair][0]),
                "side_one_digit": int(factorization.pairs[held_pair][1]),
                "fit_iterations": iterations,
                "fit_converged": converged,
                "whitening_rank": int(whitening.rank),
                **metrics,
            }
        )
        combined.append(
            {
                "truth": np.asarray(eval_side[eval_mask], dtype=np.int64),
                "probability": np.asarray(probability, dtype=np.float64),
                "value": np.asarray(
                    values[EVALUATION_PARTITION][eval_mask], dtype=np.int64
                ),
                "iterations": iterations,
                "converged": converged,
                "fold_balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
    summaries: list[dict[str, Any]] = []
    truth = np.concatenate([row["truth"] for row in combined])
    probability = np.concatenate([row["probability"] for row in combined])
    value = np.concatenate([row["value"] for row in combined])
    order = np.argsort(value)
    truth, probability, value = truth[order], probability[order], value[order]
    fold_balanced = np.asarray(
        [row["fold_balanced_accuracy"] for row in combined], dtype=np.float64
    )
    summaries.append(
        {
            "factorization_id": factorization.factorization_id,
            "kind": factorization.kind,
            "space": "fold_whitened",
            "evaluation": EVALUATION_NAME,
            "protocol": "held_pair",
            "fold_count": 5,
            "fold_balanced_accuracy_min": float(np.min(fold_balanced)),
            "fold_balanced_accuracy_mean": float(np.mean(fold_balanced)),
            "fold_balanced_accuracy_std": float(np.std(fold_balanced)),
            "max_fit_iterations": int(max(row["iterations"] for row in combined)),
            "all_fits_converged": bool(
                all(row["converged"] for row in combined)
            ),
            **binary_metrics(truth, probability),
        }
    )
    pair_by_value = factorization.labels(value)[1]
    prediction_rows.extend(
        {
            "factorization_id": factorization.factorization_id,
            "kind": factorization.kind,
            "space": "fold_whitened",
            "evaluation": EVALUATION_NAME,
            "protocol": "held_pair",
            "value": int(current_value),
            "units_digit": int(current_value % 10),
            "held_pair": int(current_pair),
            "truth": int(current_truth),
            "probability_side_one": float(current_probability),
            "prediction": int(current_probability >= 0.5),
            "prefix_cluster": int(current_value // 10),
        }
        for current_value, current_pair, current_truth, current_probability in zip(
            value, pair_by_value, truth, probability, strict=True
        )
    )
    return fold_rows, summaries, prediction_rows


def _geometry_rows(
    *,
    factorization: DigitFactorization,
    space: str,
    train_values: np.ndarray,
    train_features: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, contrasts = contrast_geometry(
        features=train_features,
        values=train_values,
        factorization=factorization,
    )
    mean_direction = contrasts.mean(axis=0)
    mean_direction /= max(float(np.linalg.norm(mean_direction)), 1e-12)
    detail = [
        {
            "factorization_id": factorization.factorization_id,
            "kind": factorization.kind,
            "space": space,
            "pair_id": int(pair_id),
            "side_zero_digit": int(factorization.pairs[pair_id][0]),
            "side_one_digit": int(factorization.pairs[pair_id][1]),
            "contrast_norm": float(np.linalg.norm(contrast)),
            "cosine_to_mean_contrast": float(
                np.dot(contrast, mean_direction)
                / max(float(np.linalg.norm(contrast)), 1e-12)
            ),
        }
        for pair_id, contrast in enumerate(contrasts)
    ]
    return (
        {
            "factorization_id": factorization.factorization_id,
            "kind": factorization.kind,
            "space": space,
            **metrics,
        },
        detail,
    )


def _rank_rows(
    summary: pd.DataFrame,
    geometry: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    held = summary.loc[
        summary["protocol"].eq("held_pair")
        & summary["space"].isin(["raw", "reference_whitened"])
    ]
    for (space, evaluation), frame in held.groupby(["space", "evaluation"]):
        observed = frame.loc[frame["kind"].eq("registered_parity")].iloc[0]
        controls = frame.loc[frame["kind"].eq("matched_random")]
        for metric in ("balanced_accuracy", "conditional_bits"):
            rows.append(
                {
                    "surface": "predictive",
                    "space": space,
                    "evaluation": evaluation,
                    "metric": metric,
                    "observed": float(observed[metric]),
                    **empirical_upper_tail(
                        float(observed[metric]),
                        controls[metric].to_numpy(dtype=np.float64),
                    ),
                }
            )
    for space, frame in geometry.groupby("space"):
        observed = frame.loc[frame["kind"].eq("registered_parity")].iloc[0]
        controls = frame.loc[frame["kind"].eq("matched_random")]
        for metric in (
            "mean_pair_contrast_cosine",
            "rank1_contrast_energy_fraction",
            "side_vs_interaction_energy_fraction",
        ):
            rows.append(
                {
                    "surface": "geometry",
                    "space": space,
                    "evaluation": "fit_context_mean",
                    "metric": metric,
                    "observed": float(observed[metric]),
                    **empirical_upper_tail(
                        float(observed[metric]),
                        controls[metric].to_numpy(dtype=np.float64),
                    ),
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    done_path = output_dir / "DONE"
    if done_path.exists() and not args.force:
        print(done_path)
        return
    set_global_seed(int(args.seed), deterministic=True)
    regimes, provenance = load_regime_data(
        source_run_dir=args.source_run_dir,
        bag_output_dir=args.bag_output_dir,
    )
    occurrence = regimes["occurrence_heldout"]
    source_values = {
        partition: np.asarray(occurrence[partition][0], dtype=np.int64)
        for partition in ("train", "dev", "test")
    }
    source_activations = {
        partition: np.asarray(occurrence[partition][1], dtype=np.float64)
        for partition in ("train", "dev", "test")
    }
    for partition in ("train", "dev", "test"):
        if not np.array_equal(
            source_values[partition],
            np.arange(1000, dtype=np.int64),
        ):
            raise ValueError(f"{partition} values are not exactly 0..999.")
    if not (
        np.array_equal(source_activations["train"], source_activations["dev"])
        and np.array_equal(source_activations["train"], source_activations["test"])
    ):
        raise RuntimeError(
            "The registered design expects the cached occurrence splits to be "
            "byte-identical at the source-number token. Register a context "
            "replication analysis before using distinct arrays."
        )
    values = {
        "train": source_values["train"],
        EVALUATION_PARTITION: source_values["train"],
    }
    activations = {
        "train": source_activations["train"],
        EVALUATION_PARTITION: source_activations["train"],
    }

    raw_center, raw_rms = raw_feature_transform(activations["train"])
    reference_whitening_path = args.bag_output_dir / "split00_activations.npz"
    if not reference_whitening_path.exists():
        raise FileNotFoundError(
            f"Reference whitening artifact is missing: {reference_whitening_path}"
        )
    with np.load(reference_whitening_path, allow_pickle=False) as payload:
        reference_whitening_activations = np.asarray(
            payload["train_activations"],
            dtype=np.float64,
        )
    if (
        reference_whitening_activations.ndim != 2
        or reference_whitening_activations.shape[1]
        != activations["train"].shape[1]
    ):
        raise ValueError(
            "Reference whitening activations do not match the probe dimension: "
            f"{reference_whitening_activations.shape} versus "
            f"{activations['train'].shape}."
        )
    whitening = fit_compact_whitening(
        reference_whitening_activations,
        relative_tolerance=WHITENING_RTOL,
    )
    if whitening.rank >= reference_whitening_activations.shape[0]:
        raise RuntimeError(
            "Reference whitening retained as many dimensions as source rows; "
            "centered whitening should have rank at most n-1."
        )
    feature_spaces = {
        "raw": {
            partition: apply_raw_feature_transform(
                activations[partition],
                center=raw_center,
                rms_scale=raw_rms,
            )
            for partition in ("train", EVALUATION_PARTITION)
        },
        "reference_whitened": {
            partition: whitening.transform(activations[partition])
            for partition in ("train", EVALUATION_PARTITION)
        },
    }
    parity = parity_factorization()
    factorizations = (
        parity,
        *random_factorizations(
            count=int(args.random_controls),
            seed=int(args.seed),
        ),
    )
    manifest = pd.DataFrame(factorization_manifest_rows(factorizations))
    manifest_payload = manifest.to_csv(index=False).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for factorization_index, factorization in enumerate(factorizations):
        if factorization_index % 25 == 0:
            print(
                f"factorization {factorization_index + 1}/{len(factorizations)}",
                flush=True,
            )
        for space, features in feature_spaces.items():
            current_fold, current_summary, current_predictions = (
                _evaluate_factorization(
                    factorization=factorization,
                    space=space,
                    values=values,
                    features=features,
                    seed=int(args.seed) + factorization_index * 17,
                    save_predictions=factorization.kind == "registered_parity",
                )
            )
            fold_rows.extend(current_fold)
            summary_rows.extend(current_summary)
            prediction_rows.extend(current_predictions)
            current_geometry, current_contrasts = _geometry_rows(
                factorization=factorization,
                space=space,
                train_values=values["train"],
                train_features=features["train"],
            )
            geometry_rows.append(current_geometry)
            contrast_rows.extend(current_contrasts)

    strict_fold, strict_summary, strict_predictions = _strict_fold_whitening(
        factorization=parity,
        values=values,
        activations=activations,
        seed=int(args.seed),
    )
    fold_rows.extend(strict_fold)
    summary_rows.extend(strict_summary)
    prediction_rows.extend(strict_predictions)

    folds = pd.DataFrame(fold_rows)
    summaries = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows)
    geometry = pd.DataFrame(geometry_rows)
    contrasts = pd.DataFrame(contrast_rows)
    ranks = pd.DataFrame(_rank_rows(summaries, geometry))

    bootstrap_rows: list[dict[str, Any]] = []
    for (space, evaluation), frame in predictions.groupby(["space", "evaluation"]):
        interval = cluster_bootstrap_binary_metrics(
            truth=frame["truth"].to_numpy(dtype=np.int64),
            probability=frame["probability_side_one"].to_numpy(dtype=np.float64),
            clusters=frame["prefix_cluster"].to_numpy(dtype=np.int64),
            samples=int(args.bootstrap_samples),
            seed=int(args.seed)
            + int(hashlib.sha256(f"{space}:{evaluation}".encode()).hexdigest()[:8], 16),
        )
        bootstrap_rows.append(
            {
                "space": space,
                "evaluation": evaluation,
                "cluster_definition": "floor(value / 10)",
                "cluster_count": int(frame["prefix_cluster"].nunique()),
                "bootstrap_samples": int(args.bootstrap_samples),
                **interval,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    expected_factorizations = 1 + int(args.random_controls)
    expected_fold_rows = expected_factorizations * 2 * 5 + 5
    expected_summary_rows = expected_factorizations * 2 * 2 + 1
    audit = {
        "expected_factorizations": expected_factorizations,
        "observed_factorizations": int(
            manifest["factorization_id"].nunique()
        ),
        "expected_fold_rows": expected_fold_rows,
        "observed_fold_rows": int(len(folds)),
        "expected_summary_rows": expected_summary_rows,
        "observed_summary_rows": int(len(summaries)),
        "expected_geometry_rows": expected_factorizations * 2,
        "observed_geometry_rows": int(len(geometry)),
        "registered_prediction_rows": int(len(predictions)),
        "all_logistic_fits_converged": bool(folds["fit_converged"].all()),
        "all_numeric_summary_values_finite": bool(
            np.isfinite(
                summaries[
                    [
                        "accuracy",
                        "balanced_accuracy",
                        "nll_nats",
                        "conditional_bits",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        ),
    }
    audit["all_complete"] = bool(
        audit["observed_factorizations"] == audit["expected_factorizations"]
        and audit["observed_fold_rows"] == audit["expected_fold_rows"]
        and audit["observed_summary_rows"] == audit["expected_summary_rows"]
        and audit["observed_geometry_rows"] == audit["expected_geometry_rows"]
        and audit["all_logistic_fits_converged"]
        and audit["all_numeric_summary_values_finite"]
    )
    if not audit["all_complete"]:
        raise RuntimeError(f"Factorial generalization audit failed: {audit}")

    metadata = {
        "schema_version": 1,
        "design_path": str(DESIGN_PATH.resolve()),
        "model_id": provenance["model_id"],
        "model_revision": provenance["model_revision"],
        "layer": provenance["layer"],
        "family": provenance["family"],
        "occurrence_splits": provenance["occurrence_splits"],
        "activation_cache_root": provenance["activation_cache_root"],
        "source_split_artifact": provenance["split_artifact"],
        "source_split_artifact_sha256": provenance["split_artifact_sha256"],
        "activation_array_sha256": {
            partition: _array_sha256(source_activations[partition])
            for partition in ("train", "dev", "test")
        },
        "occurrence_arrays_byte_identical": True,
        "factorization_manifest_sha256": manifest_hash,
        "random_control_count": int(args.random_controls),
        "seed": int(args.seed),
        "logistic": {
            "C": LOGISTIC_C,
            "solver": "liblinear",
            "dual": True,
            "penalty": "l2",
            "max_iter": LOGISTIC_MAX_ITER,
            "tol": LOGISTIC_TOL,
        },
        "raw_global_rms_scale": raw_rms,
        "reference_subset_whitening": {
            "source_artifact": str(reference_whitening_path.resolve()),
            "source_artifact_sha256": _sha256(reference_whitening_path),
            "source_array_sha256": _array_sha256(
                reference_whitening_activations
            ),
            "source_shape": list(reference_whitening_activations.shape),
            "rank": int(whitening.rank),
            "relative_tolerance": WHITENING_RTOL,
            "train_covariance_max_abs_error": whitening.covariance_error(
                reference_whitening_activations
            ),
            "retained_condition_number": float(
                whitening.singular_values[0]
                / whitening.singular_values[whitening.rank - 1]
            ),
        },
        "bootstrap_samples": int(args.bootstrap_samples),
        "runtime": runtime_metadata(),
    }
    atomic_write_csv(output_dir / "factorization_manifest.csv", manifest)
    atomic_write_csv(output_dir / "fold_metrics.csv", folds)
    atomic_write_csv(output_dir / "factorization_summary.csv", summaries)
    atomic_write_csv(output_dir / "registered_parity_predictions.csv", predictions)
    atomic_write_csv(output_dir / "geometry_summary.csv", geometry)
    atomic_write_csv(output_dir / "contrast_detail.csv", contrasts)
    atomic_write_csv(output_dir / "empirical_ranks.csv", ranks)
    atomic_write_csv(output_dir / "bootstrap_intervals.csv", bootstrap)
    atomic_write_json(output_dir / "completion_audit.json", audit)
    atomic_write_json(output_dir / "metadata.json", metadata)
    atomic_write_text(done_path, "ok\n")
    print(output_dir)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
