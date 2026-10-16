from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from src.activation_steering import circular_sinusoidal_encode
from src.identity_prompting import render_arithmetic_prompt_from_row
from src.ones_digit_intervention import (
    OPERAND_MODES,
    TASKS,
    _fit_least_squares_features,
    _numeric_tracking_arrays,
    collect_number_representations,
    evaluate_baseline_arithmetic_contexts,
    render_arithmetic_contexts,
)
from src.probe_architectures import (
    CIRCULAR_CLS_PROBE_ARCHITECTURE,
    CIRCULAR_REG_PROBE_ARCHITECTURE,
    canonicalize_probe_architecture_name,
)
from src.probes import fit_classifier_probe
from src.steering_sanity import evaluate_logits_against_shifted_targets
from src.token_utils import encode_integer


PLS_X_WEIGHT1_PROBE_ARCHITECTURE = "pls_x_weight1"
MAGNITUDE_PROBE_ARCHITECTURES = (
    "linear",
    CIRCULAR_REG_PROBE_ARCHITECTURE,
    CIRCULAR_CLS_PROBE_ARCHITECTURE,
    "pls",
    PLS_X_WEIGHT1_PROBE_ARCHITECTURE,
)


def _round_clip_values(values: np.ndarray, *, value_min: int, value_max: int) -> np.ndarray:
    rounded = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    return np.clip(rounded, int(value_min), int(value_max))


def _support_values(value_min: int, value_max: int) -> np.ndarray:
    return np.arange(int(value_min), int(value_max) + 1, dtype=np.int64)


def _value_positions(support_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    support = np.asarray(support_values, dtype=np.int64).reshape(-1)
    value_arr = np.asarray(values, dtype=np.int64).reshape(-1)
    positions = np.searchsorted(support, value_arr)
    if np.any(positions >= len(support)) or not np.array_equal(support[positions], value_arr):
        raise ValueError("Values must lie inside the probe support.")
    return positions


def _fit_ridge_features(train_x: np.ndarray, train_features: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_features, dtype=np.float64)
    lam = float(ridge_lambda)
    if lam < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda!r}")
    if lam == 0.0:
        return _fit_least_squares_features(x, y)

    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    u, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    shrink = singular_values / ((singular_values**2) + lam)
    coefficients = (vt.T * shrink[None, :]) @ (u.T @ y_centered)
    bias = (y_mean - x_mean @ coefficients).reshape(-1)
    return np.asarray(coefficients, dtype=np.float64), np.asarray(bias, dtype=np.float64)


@dataclass(frozen=True)
class MagnitudeSplitConfig:
    train_size: int
    dev_size: int
    test_size: int
    value_min: int = 0
    value_max: int = 999
    seed: int = 1729

    @property
    def total_size(self) -> int:
        return int(self.train_size + self.dev_size + self.test_size)


@dataclass
class MagnitudeProbe:
    architecture: str
    coefficients: np.ndarray
    bias: np.ndarray
    decoder_matrix: np.ndarray
    support_values: np.ndarray
    support_features: np.ndarray
    metadata: dict[str, Any]

    def encode_values(self, values: np.ndarray) -> np.ndarray:
        architecture = canonicalize_probe_architecture_name(self.architecture)
        value_arr = np.asarray(values, dtype=np.int64).reshape(-1)
        if architecture in {"linear", "pls", PLS_X_WEIGHT1_PROBE_ARCHITECTURE}:
            return value_arr[:, None].astype(np.float64)
        if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
            return circular_sinusoidal_encode(
                value_arr,
                embedding_dim=int(self.support_features.shape[1]),
                min_value=int(self.support_values.min()),
                max_value=int(self.support_values.max()) + 1,
            )
        if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
            return self.support_features[_value_positions(self.support_values, value_arr)]
        raise ValueError(f"Unsupported architecture: {architecture}")

    def predict_features(self, representations: np.ndarray) -> np.ndarray:
        array = np.asarray(representations, dtype=np.float64)
        return array @ self.coefficients + self.bias

    def predict_values(self, representations: np.ndarray) -> np.ndarray:
        predicted_features = self.predict_features(representations)
        architecture = canonicalize_probe_architecture_name(self.architecture)
        if architecture in {"linear", "pls", PLS_X_WEIGHT1_PROBE_ARCHITECTURE}:
            return _round_clip_values(
                predicted_features.reshape(-1),
                value_min=int(self.support_values.min()),
                value_max=int(self.support_values.max()),
            )
        support = np.asarray(self.support_features, dtype=np.float64)
        predicted = np.asarray(predicted_features, dtype=np.float64)
        if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
            pred_norm = np.linalg.norm(predicted, axis=1, keepdims=True)
            pred_norm = np.where(pred_norm > 0, pred_norm, 1.0)
            support_norm = np.linalg.norm(support, axis=1, keepdims=True)
            support_norm = np.where(support_norm > 0, support_norm, 1.0)
            nearest_idx = np.argmax((predicted / pred_norm) @ (support / support_norm).T, axis=1)
        elif architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
            nearest_idx = np.argmax(predicted @ support.T, axis=1)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        return self.support_values[nearest_idx]

    def intervention_delta(self, source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
        source_features = self.encode_values(source_values)
        target_features = self.encode_values(target_values)
        return np.asarray((target_features - source_features) @ self.decoder_matrix, dtype=np.float64)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("coefficients", "bias", "decoder_matrix", "support_values", "support_features"):
            payload[key] = np.asarray(payload[key]).tolist()
        return payload


def probe_from_payload(probe_payload: dict[str, Any]) -> MagnitudeProbe:
    return MagnitudeProbe(
        architecture=canonicalize_probe_architecture_name(str(probe_payload["architecture"])),
        coefficients=np.asarray(probe_payload["coefficients"], dtype=np.float64),
        bias=np.asarray(probe_payload["bias"], dtype=np.float64),
        decoder_matrix=np.asarray(probe_payload["decoder_matrix"], dtype=np.float64),
        support_values=np.asarray(probe_payload["support_values"], dtype=np.int64),
        support_features=np.asarray(probe_payload["support_features"], dtype=np.float64),
        metadata=dict(probe_payload.get("metadata", {})),
    )


def build_value_splits(config: MagnitudeSplitConfig) -> pd.DataFrame:
    values = _support_values(config.value_min, config.value_max)
    if len(values) < config.total_size:
        raise ValueError(
            f"Not enough values in [{config.value_min}, {config.value_max}] for requested split size {config.total_size}."
        )
    rng = np.random.default_rng(config.seed)
    selected = rng.choice(values, size=config.total_size, replace=False)
    selected = np.asarray(selected, dtype=np.int64)
    bounds = {
        "train": slice(0, config.train_size),
        "dev": slice(config.train_size, config.train_size + config.dev_size),
        "test": slice(config.train_size + config.dev_size, config.total_size),
    }
    rows: list[dict[str, Any]] = []
    for split, split_slice in bounds.items():
        split_values = np.sort(selected[split_slice])
        for split_rank, value in enumerate(split_values.tolist()):
            rows.append({"value": int(value), "split": split, "split_rank": int(split_rank)})
    return pd.DataFrame(rows).sort_values(["split", "value"]).reset_index(drop=True)


def build_probe_dataset(
    *,
    split_df: pd.DataFrame,
    representation_df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    merged = split_df.merge(representation_df, on="value", how="inner").sort_values(["split", "value"]).reset_index(drop=True)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "dev", "test"):
        subset = merged.loc[merged["split"] == split_name].copy()
        rep_matrix = np.stack(subset["representation"].to_numpy(), axis=0).astype(np.float32)
        values = subset["value"].to_numpy(dtype=np.int64)
        datasets[split_name] = (values, rep_matrix)
    return datasets


def fit_magnitude_probe(
    *,
    architecture: str,
    train_x: np.ndarray,
    train_values: np.ndarray,
    config: dict[str, Any] | None = None,
    value_min: int = 0,
    value_max: int = 999,
) -> MagnitudeProbe:
    cfg = dict(config or {})
    architecture = canonicalize_probe_architecture_name(architecture)
    support_values = _support_values(value_min, value_max)

    if architecture == "linear":
        support_features = support_values[:, None].astype(np.float64)
        train_features = np.asarray(train_values, dtype=np.float64)[:, None]
        ridge_lambda = float(cfg.get("ridge_lambda", 0.0))
        coefficients, bias = _fit_ridge_features(train_x, train_features, ridge_lambda=ridge_lambda)
        metadata = {"feature_dim": 1, "ridge_lambda": ridge_lambda}
    elif architecture in {"pls", PLS_X_WEIGHT1_PROBE_ARCHITECTURE}:
        support_features = support_values[:, None].astype(np.float64)
        train_features = np.asarray(train_values, dtype=np.float64)[:, None]
        # For scalar targets, PLSRegression can still use multiple latent components.
        # The effective limit is set by the predictor matrix rank, not y-dim=1.
        max_components = min(train_x.shape[0] - 1, train_x.shape[1])
        n_components = max(1, min(int(cfg.get("n_components", 1)), max_components))
        model = PLSRegression(n_components=n_components, scale=bool(cfg.get("scale", False)))
        model.fit(train_x.astype(np.float64), train_features.astype(np.float64))
        if architecture == "pls":
            # Recover the effective affine map from model.predict directly; this is more stable
            # across sklearn versions than interpreting coef_/intercept_ for scalar targets.
            zero = np.zeros((1, train_x.shape[1]), dtype=np.float64)
            basis = np.eye(train_x.shape[1], dtype=np.float64)
            bias = np.asarray(model.predict(zero)[0], dtype=np.float64).reshape(-1)
            coefficients = np.asarray(model.predict(basis) - bias[None, :], dtype=np.float64)
            if coefficients.shape != (train_x.shape[1], 1):
                raise ValueError(f"Unexpected derived PLS coefficient shape {coefficients.shape}; expected {(train_x.shape[1], 1)}.")
            basis_mode = "effective_predict_coef"
        else:
            x_weight1 = np.asarray(model.x_weights_[:, :1], dtype=np.float64)
            if x_weight1.shape != (train_x.shape[1], 1):
                raise ValueError(f"Unexpected PLS x_weight1 shape {x_weight1.shape}; expected {(train_x.shape[1], 1)}.")
            scores = np.asarray(train_x, dtype=np.float64) @ x_weight1
            score_centered = scores - scores.mean(axis=0, keepdims=True)
            y_centered = train_features - train_features.mean(axis=0, keepdims=True)
            denom = float(np.sum(score_centered[:, 0] ** 2))
            if denom <= 0.0:
                raise ValueError("Cannot calibrate pls_x_weight1 probe because the first x_weight score has zero variance.")
            slope = float(np.sum(score_centered[:, 0] * y_centered[:, 0]) / denom)
            intercept = float(train_features.mean() - scores.mean() * slope)
            coefficients = np.asarray(x_weight1 * slope, dtype=np.float64)
            bias = np.asarray([intercept], dtype=np.float64)
            basis_mode = "calibrated_x_weight1"
        metadata = {
            "feature_dim": 1,
            "n_components": int(n_components),
            "scale": bool(cfg.get("scale", False)),
            "basis_mode": basis_mode,
        }
        if architecture == PLS_X_WEIGHT1_PROBE_ARCHITECTURE:
            metadata["subspace_basis_mode"] = "all_pls_x_weights"
            metadata["subspace_basis"] = np.asarray(model.x_weights_[:, :n_components], dtype=np.float64).T.tolist()
    elif architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
        basis_dim = int(cfg.get("basis_dim", 64))
        support_features = circular_sinusoidal_encode(
            support_values,
            embedding_dim=basis_dim,
            min_value=int(value_min),
            max_value=int(value_max) + 1,
        )
        train_features = support_features[np.asarray(train_values, dtype=np.int64)]
        coefficients, bias = _fit_least_squares_features(train_x, train_features)
        metadata = {"feature_dim": int(support_features.shape[1]), "basis_dim": int(basis_dim)}
    elif architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        basis_dim = int(cfg.get("basis_dim", 64))
        hidden_dim = int(cfg.get("hidden_dim", 100))
        codebook = circular_sinusoidal_encode(
            support_values,
            embedding_dim=basis_dim,
            min_value=int(value_min),
            max_value=int(value_max) + 1,
        ).astype(np.float32)
        classifier = fit_classifier_probe(
            train_x=np.asarray(train_x, dtype=np.float32),
            train_y=np.asarray(train_values, dtype=np.int64),
            test_x=np.asarray(train_x[:1], dtype=np.float32),
            codebook=codebook,
            hidden_dim=hidden_dim,
            lr=float(cfg.get("lr", 1e-2)),
            weight_decay=float(cfg.get("weight_decay", 1e-3)),
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            max_epochs=int(cfg.get("max_epochs", 1000)),
            patience=int(cfg.get("patience", 20)),
            val_fraction=float(cfg.get("val_fraction", 0.1)),
            batch_size=int(cfg.get("batch_size", 256)),
            seed=int(cfg.get("seed", 0)),
            device=str(cfg.get("device", "cpu")),
            fixed_epochs=None if cfg.get("fixed_epochs") is None else int(cfg["fixed_epochs"]),
        )
        w_in = np.asarray(classifier.best_state["w_in.weight"], dtype=np.float64)
        w_out = np.asarray(classifier.best_state["w_out.weight"], dtype=np.float64)
        coefficients = w_in.T
        bias = np.zeros(w_in.shape[0], dtype=np.float64)
        support_features = codebook.astype(np.float64) @ w_out.T
        metadata = {
            "feature_dim": int(support_features.shape[1]),
            "basis_dim": int(basis_dim),
            "hidden_dim": int(hidden_dim),
            "lr": float(cfg.get("lr", 1e-2)),
            "weight_decay": float(cfg.get("weight_decay", 1e-3)),
            "betas": [float(x) for x in tuple(cfg.get("betas", (0.9, 0.999)))],
            "max_epochs": int(cfg.get("max_epochs", 1000)),
            "patience": int(cfg.get("patience", 20)),
            "val_fraction": float(cfg.get("val_fraction", 0.1)),
            "batch_size": int(cfg.get("batch_size", 256)),
            "seed": int(cfg.get("seed", 0)),
            "device": str(cfg.get("device", "cpu")),
            "best_val_loss": float(classifier.best_val_loss),
            "epochs_trained": int(len(classifier.history)),
            "fixed_epochs": None if cfg.get("fixed_epochs") is None else int(cfg["fixed_epochs"]),
            "optimizer_examples": int(len(train_values))
            if cfg.get("fixed_epochs") is not None
            else int(len(train_values) - max(1, round(len(train_values) * float(cfg.get("val_fraction", 0.1))))),
        }
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    decoder_matrix = np.linalg.pinv(coefficients)
    return MagnitudeProbe(
        architecture=architecture,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
        decoder_matrix=np.asarray(decoder_matrix, dtype=np.float64),
        support_values=support_values,
        support_features=np.asarray(support_features, dtype=np.float64),
        metadata=metadata,
    )


def evaluate_probe_split(
    *,
    probe: MagnitudeProbe,
    representations: np.ndarray,
    values: np.ndarray,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = np.asarray(values, dtype=np.int64)
    predicted = probe.predict_values(representations)
    feature_pred = probe.predict_features(representations)
    abs_error = np.abs(predicted - truth)
    raw_pred = feature_pred.reshape(-1) if feature_pred.ndim == 2 and feature_pred.shape[1] == 1 else np.full(len(truth), np.nan)
    detail = pd.DataFrame(
        {
            "split": split_name,
            "architecture": probe.architecture,
            "value": truth,
            "prediction": predicted,
            "raw_prediction": raw_pred.astype(np.float64),
            "correct": (predicted == truth).astype(int),
            "abs_error": abs_error.astype(np.int64),
            "prediction_feature_norm": np.linalg.norm(feature_pred, axis=1).astype(np.float64),
        }
    )
    summary = {
        "split": split_name,
        "architecture": probe.architecture,
        "num_examples": int(len(detail)),
        "accuracy": float(detail["correct"].mean()),
        "mean_abs_error": float(detail["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean((predicted.astype(np.float64) - truth.astype(np.float64)) ** 2))),
    }
    return detail, summary


def _source_partner_bounds(*, source_value: int, task: str, operand_mode: str) -> tuple[int, int]:
    if task == "addition":
        return 0, 999 - int(source_value)
    if task == "subtraction" and operand_mode == "x1":
        return 0, int(source_value)
    if task == "subtraction" and operand_mode == "x2":
        return int(source_value), 999
    raise ValueError(f"Unsupported task/operand_mode combination: task={task!r}, operand_mode={operand_mode!r}")


def build_arithmetic_contexts(
    *,
    source_values: np.ndarray,
    evaluation_split: str,
    tasks: list[str],
    operand_modes: list[str],
    contexts_per_source: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
    for operand_mode in operand_modes:
        if operand_mode not in OPERAND_MODES:
            raise ValueError(f"Unsupported operand_mode: {operand_mode}")

    for source_value in np.asarray(source_values, dtype=np.int64).tolist():
        for task in tasks:
            for operand_mode in operand_modes:
                partner_min, partner_max = _source_partner_bounds(
                    source_value=int(source_value),
                    task=str(task),
                    operand_mode=str(operand_mode),
                )
                partner_values = np.arange(int(partner_min), int(partner_max) + 1, dtype=np.int64)
                rng = np.random.default_rng(
                    int(seed) + (1009 * int(source_value)) + (97 * len(str(task))) + (11 if str(operand_mode) == "x2" else 0)
                )
                if len(partner_values) >= int(contexts_per_source):
                    selected = np.sort(rng.choice(partner_values, size=int(contexts_per_source), replace=False))
                else:
                    selected = np.sort(rng.choice(partner_values, size=int(contexts_per_source), replace=True))
                for context_idx, partner_value in enumerate(selected.tolist()):
                    if str(operand_mode) == "x1":
                        x1 = int(source_value)
                        x2 = int(partner_value)
                    else:
                        x1 = int(partner_value)
                        x2 = int(source_value)
                    target = x1 + x2 if str(task) == "addition" else x1 - x2
                    if not (0 <= target <= 999):
                        raise ValueError("Generated an invalid baseline target.")
                    rows.append(
                        {
                            "evaluation_split": str(evaluation_split),
                            "source_value": int(source_value),
                            "task": str(task),
                            "operand_mode": str(operand_mode),
                            "context_idx": int(context_idx),
                            "x1": int(x1),
                            "x2": int(x2),
                            "target": int(target),
                        }
                    )
    return (
        pd.DataFrame(rows)
        .sort_values(["evaluation_split", "task", "operand_mode", "source_value", "context_idx"])
        .reset_index(drop=True)
    )


def _valid_target_mask(
    *,
    row: pd.Series,
    candidate_target_values: np.ndarray,
) -> np.ndarray:
    source_value = int(row["source_value"])
    targets = np.asarray(candidate_target_values, dtype=np.int64)
    if int(row["operand_mode"] == "x1"):
        shifted_x1 = targets
        shifted_x2 = np.full_like(targets, int(row["x2"]))
    else:
        shifted_x1 = np.full_like(targets, int(row["x1"]))
        shifted_x2 = targets
    shifted_target = shifted_x1 + shifted_x2 if str(row["task"]) == "addition" else shifted_x1 - shifted_x2
    return (targets != source_value) & (shifted_target >= 0) & (shifted_target <= 999)


def build_value_intervention_frame(
    *,
    eligible_df: pd.DataFrame,
    tokenizer,
    candidate_target_values: np.ndarray,
    targets_per_example: int,
    seed: int,
) -> pd.DataFrame:
    if eligible_df.empty:
        return pd.DataFrame()
    subset = eligible_df.copy()
    if "example_id" not in subset.columns:
        subset = subset.reset_index(drop=False).rename(columns={"index": "example_id"})

    target_pool = np.asarray(candidate_target_values, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for row in subset.to_dict("records"):
        series = pd.Series(row)
        valid_targets = target_pool[_valid_target_mask(row=series, candidate_target_values=target_pool)]
        if len(valid_targets) == 0:
            continue
        rng = np.random.default_rng(int(seed) + (1000003 * int(row["example_id"])) + (1 if str(row["evaluation_split"]) == "test" else 0))
        take = min(int(targets_per_example), len(valid_targets))
        selected_targets = np.sort(rng.choice(valid_targets, size=take, replace=False))
        for target_value in selected_targets.tolist():
            if str(row["operand_mode"]) == "x1":
                shifted_x1 = int(target_value)
                shifted_x2 = int(row["x2"])
            else:
                shifted_x1 = int(row["x1"])
                shifted_x2 = int(target_value)
            shifted_target = shifted_x1 + shifted_x2 if str(row["task"]) == "addition" else shifted_x1 - shifted_x2
            token_ids = encode_integer(tokenizer, int(shifted_target))
            if len(token_ids) != 1:
                continue
            shifted_prompt, shifted_token_idx = render_arithmetic_prompt_from_row(
                row=row,
                x1=int(shifted_x1),
                x2=int(shifted_x2),
                tokenizer=tokenizer,
            )
            rows.append(
                {
                    **row,
                    "target_value": int(target_value),
                    "delta_value": int(target_value) - int(row["source_value"]),
                    "abs_delta_value": abs(int(target_value) - int(row["source_value"])),
                    "shifted_x1": int(shifted_x1),
                    "shifted_x2": int(shifted_x2),
                    "shifted_target": int(shifted_target),
                    "target_token_id": int(token_ids[0]),
                    "shifted_prompt": shifted_prompt,
                    "shifted_token_idx": int(shifted_token_idx),
                    "expected_value_baseline": float(row["baseline_expected_value"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["evaluation_split", "task", "operand_mode", "source_value", "example_id", "target_value"]
    ).reset_index(drop=True)


def summarize_pair_results(results: pd.DataFrame) -> dict[str, Any]:
    predicted_numeric = pd.to_numeric(results["predicted_value"], errors="coerce")
    abs_distance = (predicted_numeric - results["shifted_target"].astype(float)).abs()
    return {
        "num_examples": int(len(results)),
        "number_accuracy": float(results["correct"].mean()),
        "mean_abs_distance_to_target": float(abs_distance.mean()) if abs_distance.notna().any() else float("nan"),
        "mean_target_prob_delta": float(results["target_prob_gain"].mean()),
        "numeric_prediction_rate": float(predicted_numeric.notna().mean()),
        "mean_expected_distance_to_target": float((results["expected_value"] - results["shifted_target"]).abs().mean()),
        "mean_abs_delta_value": float(results["abs_delta_value"].mean()),
    }


def run_probe_manifold_interventions(
    *,
    model,
    tokenizer,
    probe: MagnitudeProbe,
    intervention_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    method_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if intervention_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tracked_values, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
    source_values = intervention_df["source_value"].to_numpy(dtype=np.int64)
    target_values = intervention_df["target_value"].to_numpy(dtype=np.int64)
    delta_rows = probe.intervention_delta(source_values=source_values, target_values=target_values).astype(np.float32)

    from src.activation_steering import _batch_forward_with_optional_direction

    logits_chunks, _, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=intervention_df["prompt"].tolist(),
        token_idxs=intervention_df["token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        scaled_directions=delta_rows,
        capture_pre_activations=False,
    )
    logits = np.concatenate(logits_chunks, axis=0)
    results = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=intervention_df,
        logits=logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    ).assign(
        method=probe.architecture if method_name is None else str(method_name),
        edit_family="probe",
        architecture=str(probe.architecture),
    )
    summary_rows: list[dict[str, Any]] = []
    for (evaluation_split, task, operand_mode), subset in results.groupby(["evaluation_split", "task", "operand_mode"], sort=True):
        summary_rows.append(
            {
                "method": probe.architecture if method_name is None else str(method_name),
                "edit_family": "probe",
                "architecture": str(probe.architecture),
                "evaluation_split": str(evaluation_split),
                "task": str(task),
                "operand_mode": str(operand_mode),
                **summarize_pair_results(subset),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["evaluation_split", "task", "operand_mode", "architecture"]).reset_index(drop=True)
    return results.reset_index(drop=True), summary
