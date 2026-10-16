from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from src.activation_steering import (
    _batch_forward_with_optional_direction,
    _parse_optional_int,
    _softmax_numpy,
    circular_sinusoidal_encode,
    collect_prompt_activations,
    render_probe_prompt,
    render_prompt_with_operand_spans,
    resolve_token_idx,
)
from src.identity_prompting import ArithmeticPromptConfig, render_arithmetic_prompt, sample_arithmetic_example_rows
from src.counterfactual_scoring import (
    PROBABILITY_BANK_ATTR,
    CounterfactualProbabilityBank,
    attach_counterfactual_baseline,
)
from src.probe_architectures import (
    CIRCULAR_CLS_PROBE_ARCHITECTURE,
    CIRCULAR_REG_PROBE_ARCHITECTURE,
    canonicalize_probe_architecture_name as canonicalize_shared_probe_architecture_name,
)
from src.probes import fit_classifier_probe
from src.steering_sanity import evaluate_logits_against_shifted_targets
from src.token_utils import encode_integer


TASKS = ("addition", "subtraction")
OPERAND_MODES = ("x1", "x2")
CAT_PROBE_ARCHITECTURE = "digitplace_linprobe_cat"
REG_PROBE_ARCHITECTURE = "digitplace_linprobe_reg"
LEGACY_CAT_PROBE_ARCHITECTURE = "linear"
PROBE_ARCHITECTURES = (
    CAT_PROBE_ARCHITECTURE,
    REG_PROBE_ARCHITECTURE,
    "pls",
    CIRCULAR_REG_PROBE_ARCHITECTURE,
    CIRCULAR_CLS_PROBE_ARCHITECTURE,
)
DIGIT_PLACES = ("ones", "tens", "hundreds")
PLACE_VALUE = {"ones": 1, "tens": 10, "hundreds": 100}


def canonical_probe_architecture_name(architecture: str) -> str:
    name = str(architecture)
    if name == LEGACY_CAT_PROBE_ARCHITECTURE:
        return CAT_PROBE_ARCHITECTURE
    return canonicalize_shared_probe_architecture_name(name)


def _validate_digit_place(digit_place: str) -> str:
    place = str(digit_place)
    if place not in DIGIT_PLACES:
        raise ValueError(f"Unsupported digit_place: {digit_place!r}")
    return place


def _digit_column(digit_place: str) -> str:
    return f"{_validate_digit_place(digit_place)}_digit"


def _extract_digits(values: np.ndarray | list[int], digit_place: str) -> np.ndarray:
    place = _validate_digit_place(digit_place)
    arr = np.asarray(values, dtype=np.int64)
    return ((arr // int(PLACE_VALUE[place])) % 10).astype(np.int64)


def _replace_digit(values: np.ndarray | list[int], digit_place: str, digits: np.ndarray | list[int] | int) -> np.ndarray:
    place = _validate_digit_place(digit_place)
    arr = np.asarray(values, dtype=np.int64)
    digit_arr = np.asarray(digits, dtype=np.int64)
    return arr - (_extract_digits(arr, place) * int(PLACE_VALUE[place])) + (digit_arr * int(PLACE_VALUE[place]))


def _variant_bounds_for_place(source_value: int, digit_place: str) -> tuple[int, int]:
    place = _validate_digit_place(digit_place)
    source_digit = int(_extract_digits([source_value], place)[0])
    base = int(source_value) - (source_digit * int(PLACE_VALUE[place]))
    return base, base + (9 * int(PLACE_VALUE[place]))


@dataclass(frozen=True)
class OnesDigitSplitConfig:
    train_per_digit: int
    dev_per_digit: int
    test_per_digit: int
    value_min: int = 0
    value_max: int = 999
    seed: int = 1729
    digit_place: str = "ones"

    @property
    def values_per_digit(self) -> int:
        return int(self.train_per_digit + self.dev_per_digit + self.test_per_digit)


@dataclass
class OnesDigitProbe:
    architecture: str
    coefficients: np.ndarray
    bias: np.ndarray
    decoder_matrix: np.ndarray
    support_digits: np.ndarray
    support_features: np.ndarray
    metadata: dict[str, Any]

    def encode_digits(self, digits: np.ndarray) -> np.ndarray:
        architecture = canonical_probe_architecture_name(self.architecture)
        digit_arr = np.asarray(digits, dtype=np.int64).reshape(-1)
        if architecture in {CAT_PROBE_ARCHITECTURE, "pls"}:
            eye = np.eye(len(self.support_digits), dtype=np.float64)
            return eye[digit_arr]
        if architecture == REG_PROBE_ARCHITECTURE:
            return digit_arr[:, None].astype(np.float64)
        if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
            return circular_sinusoidal_encode(
                digit_arr,
                embedding_dim=int(self.support_features.shape[1]),
                min_value=int(self.support_digits.min()),
                max_value=int(self.support_digits.max()) + 1,
            )
        if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
            return self.support_features[digit_arr]
        raise ValueError(f"Unsupported architecture: {architecture}")

    def predict_features(self, representations: np.ndarray) -> np.ndarray:
        array = np.asarray(representations, dtype=np.float64)
        return array @ self.coefficients + self.bias

    def predict_digits(self, representations: np.ndarray) -> np.ndarray:
        return decode_probe_digits(
            predicted_features=self.predict_features(representations),
            support_digits=self.support_digits,
            support_features=self.support_features,
            architecture=self.architecture,
        )

    def intervention_delta(self, source_digits: np.ndarray, target_digits: np.ndarray) -> np.ndarray:
        source_features = self.encode_digits(source_digits)
        target_features = self.encode_digits(target_digits)
        return np.asarray((target_features - source_features) @ self.decoder_matrix, dtype=np.float64)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("coefficients", "bias", "decoder_matrix", "support_digits", "support_features"):
            payload[key] = np.asarray(payload[key]).tolist()
        return payload


def build_digit_splits(config: OnesDigitSplitConfig) -> pd.DataFrame:
    digit_place = _validate_digit_place(config.digit_place)
    digit_col = _digit_column(digit_place)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed)
    for digit in range(10):
        candidates = np.arange(config.value_min, config.value_max + 1, dtype=np.int64)
        candidates = candidates[_extract_digits(candidates, digit_place) == digit]
        if len(candidates) < config.values_per_digit:
            raise ValueError(
                f"Digit {digit} has only {len(candidates)} candidates in "
                f"[{config.value_min}, {config.value_max}], need {config.values_per_digit}."
            )
        selected = rng.choice(candidates, size=config.values_per_digit, replace=False)
        selected = np.asarray(selected, dtype=np.int64)
        split_bounds = {
            "train": slice(0, config.train_per_digit),
            "dev": slice(config.train_per_digit, config.train_per_digit + config.dev_per_digit),
            "test": slice(config.train_per_digit + config.dev_per_digit, config.values_per_digit),
        }
        for split, split_slice in split_bounds.items():
            split_values = np.sort(selected[split_slice])
            for split_rank, value in enumerate(split_values.tolist()):
                rows.append(
                    {
                        "value": int(value),
                        digit_col: int(digit),
                        "split": split,
                        "split_rank": int(split_rank),
                    }
                )
    return pd.DataFrame(rows).sort_values(["split", digit_col, "value"]).reset_index(drop=True)


def collect_number_representations(
    *,
    model,
    tokenizer,
    values: np.ndarray,
    layer_idx: int,
    representation_source: str,
    prompt_template: str,
    token_mode: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    value_arr = np.asarray(values, dtype=np.int64)
    if representation_source == "token_embeddings":
        if int(layer_idx) != 0:
            raise ValueError("representation_source='token_embeddings' is only valid at layer_idx=0.")
        rows = []
        weight = model.get_input_embeddings().weight.detach().float().cpu().numpy()
        for value in value_arr.tolist():
            token_ids = encode_integer(tokenizer, int(value))
            if len(token_ids) != 1:
                raise ValueError(f"Value {value} is not single-token under the current tokenizer.")
            rows.append(weight[int(token_ids[0])])
        return np.asarray(rows, dtype=np.float32)

    if representation_source != "prompt_activations":
        raise ValueError(f"Unsupported representation_source: {representation_source!r}")

    prompts: list[str] = []
    token_idxs: list[int] = []
    for value in value_arr.tolist():
        prompt, span = render_probe_prompt(prompt_template, int(value))
        token_idx = resolve_token_idx(tokenizer=tokenizer, prompt=prompt, mention_span=span, mode=token_mode)
        prompts.append(prompt)
        token_idxs.append(int(token_idx))
    return collect_prompt_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
    ).astype(np.float32)


def _fit_least_squares_features(train_x: np.ndarray, train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([train_x.astype(np.float64), np.ones((train_x.shape[0], 1), dtype=np.float64)], axis=1)
    solution, *_ = np.linalg.lstsq(design, train_features.astype(np.float64), rcond=None)
    coefficients = np.asarray(solution[:-1], dtype=np.float64)
    bias = np.asarray(solution[-1], dtype=np.float64)
    return coefficients, bias


def decode_probe_digits(
    *,
    predicted_features: np.ndarray,
    support_digits: np.ndarray,
    support_features: np.ndarray,
    architecture: str,
) -> np.ndarray:
    predicted = np.asarray(predicted_features, dtype=np.float64)
    support = np.asarray(support_features, dtype=np.float64)
    if canonical_probe_architecture_name(architecture) in {CAT_PROBE_ARCHITECTURE, "pls"}:
        scores = predicted @ support.T
        nearest_idx = np.argmax(scores, axis=1)
    elif architecture == REG_PROBE_ARCHITECTURE:
        predicted_scalar = predicted.reshape(predicted.shape[0], -1)
        if predicted_scalar.shape[1] != 1:
            raise ValueError(f"{REG_PROBE_ARCHITECTURE} expects scalar features, got shape {predicted.shape}.")
        support_scalar = support.reshape(-1)
        nearest_idx = np.argmin(np.abs(predicted_scalar[:, 0][:, None] - support_scalar[None, :]), axis=1)
    elif architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
        pred_norm = np.linalg.norm(predicted, axis=1, keepdims=True)
        pred_norm = np.where(pred_norm > 0, pred_norm, 1.0)
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support_norm = np.where(support_norm > 0, support_norm, 1.0)
        nearest_idx = np.argmax((predicted / pred_norm) @ (support / support_norm).T, axis=1)
    elif architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        nearest_idx = np.argmax(predicted @ support.T, axis=1)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return np.asarray(support_digits, dtype=np.int64)[nearest_idx]


def fit_ones_digit_probe(
    *,
    architecture: str,
    train_x: np.ndarray,
    train_digits: np.ndarray,
    config: dict[str, Any] | None = None,
) -> OnesDigitProbe:
    cfg = dict(config or {})
    architecture = canonical_probe_architecture_name(architecture)
    support_digits = np.arange(10, dtype=np.int64)

    if architecture == CAT_PROBE_ARCHITECTURE:
        support_features = np.eye(10, dtype=np.float64)
        train_features = support_features[np.asarray(train_digits, dtype=np.int64)]
        coefficients, bias = _fit_least_squares_features(train_x, train_features)
        metadata = {"feature_dim": int(support_features.shape[1])}
    elif architecture == REG_PROBE_ARCHITECTURE:
        support_features = support_digits[:, None].astype(np.float64)
        train_features = np.asarray(train_digits, dtype=np.float64)[:, None]
        coefficients, bias = _fit_least_squares_features(train_x, train_features)
        metadata = {"feature_dim": int(support_features.shape[1])}
    elif architecture == "pls":
        support_features = np.eye(10, dtype=np.float64)
        train_features = support_features[np.asarray(train_digits, dtype=np.int64)]
        max_components = min(train_x.shape[0] - 1, train_x.shape[1], train_features.shape[1])
        n_components = max(1, min(int(cfg.get("n_components", 4)), max_components))
        model = PLSRegression(n_components=n_components, scale=bool(cfg.get("scale", False)))
        model.fit(train_x.astype(np.float64), train_features.astype(np.float64))
        coefficients = np.asarray(model.coef_.T if model.coef_.shape[0] == train_features.shape[1] else model.coef_, dtype=np.float64)
        if coefficients.shape != (train_x.shape[1], train_features.shape[1]):
            raise ValueError(
                f"Unexpected PLS coefficient shape {coefficients.shape}; "
                f"expected {(train_x.shape[1], train_features.shape[1])}."
            )
        bias = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
        metadata = {
            "feature_dim": int(support_features.shape[1]),
            "n_components": int(n_components),
            "scale": bool(cfg.get("scale", False)),
        }
    elif architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
        basis_dim = int(cfg.get("basis_dim", 8))
        support_features = circular_sinusoidal_encode(support_digits, embedding_dim=basis_dim, min_value=0, max_value=10)
        train_features = support_features[np.asarray(train_digits, dtype=np.int64)]
        coefficients, bias = _fit_least_squares_features(train_x, train_features)
        metadata = {"feature_dim": int(support_features.shape[1]), "basis_dim": int(basis_dim)}
    elif architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        basis_dim = int(cfg.get("basis_dim", 8))
        hidden_dim = int(cfg.get("hidden_dim", 16))
        codebook = circular_sinusoidal_encode(support_digits, embedding_dim=basis_dim, min_value=0, max_value=10).astype(np.float32)
        classifier = fit_classifier_probe(
            train_x=np.asarray(train_x, dtype=np.float32),
            train_y=np.asarray(train_digits, dtype=np.int64),
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
        }
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    decoder_matrix = np.linalg.pinv(coefficients)
    return OnesDigitProbe(
        architecture=architecture,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
        decoder_matrix=np.asarray(decoder_matrix, dtype=np.float64),
        support_digits=support_digits,
        support_features=np.asarray(support_features, dtype=np.float64),
        metadata=metadata,
    )


def evaluate_probe_split(
    *,
    probe: OnesDigitProbe,
    representations: np.ndarray,
    digits: np.ndarray,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = np.asarray(digits, dtype=np.int64)
    predicted = probe.predict_digits(representations)
    feature_pred = probe.predict_features(representations)
    abs_distance = np.abs(predicted - truth)
    detail = pd.DataFrame(
        {
            "split": split_name,
            "architecture": probe.architecture,
            "digit": truth,
            "prediction": predicted,
            "correct": (predicted == truth).astype(int),
            "abs_digit_distance": abs_distance.astype(np.int64),
            "prediction_feature_norm": np.linalg.norm(feature_pred, axis=1).astype(np.float64),
        }
    )
    summary = {
        "split": split_name,
        "architecture": probe.architecture,
        "num_examples": int(len(detail)),
        "accuracy": float(detail["correct"].mean()),
        "mean_abs_digit_distance": float(detail["abs_digit_distance"].mean()),
    }
    return detail, summary


def build_probe_dataset(
    *,
    split_df: pd.DataFrame,
    representation_df: pd.DataFrame,
    digit_place: str = "ones",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    digit_col = _digit_column(digit_place)
    merged = split_df.merge(representation_df, on="value", how="inner").sort_values(["split", digit_col, "value"]).reset_index(drop=True)
    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "dev", "test"):
        subset = merged.loc[merged["split"] == split_name].copy()
        rep_matrix = np.stack(subset["representation"].to_numpy(), axis=0).astype(np.float32)
        digits = subset[digit_col].to_numpy(dtype=np.int64)
        values = subset["value"].to_numpy(dtype=np.int64)
        datasets[split_name] = (values, digits, rep_matrix)
    return datasets


def _partner_bounds_for_all_targets(*, source_value: int, digit_place: str, task: str, operand_mode: str) -> tuple[int, int]:
    min_variant, max_variant = _variant_bounds_for_place(int(source_value), digit_place)
    if task == "addition":
        partner_min = 0
        partner_max = 999 - max_variant
        return partner_min, partner_max
    if task == "subtraction" and operand_mode == "x1":
        return 0, min_variant
    if task == "subtraction" and operand_mode == "x2":
        return max_variant, 999
    raise ValueError(f"Unsupported task/operand_mode combination: task={task!r}, operand_mode={operand_mode!r}")


def build_arithmetic_contexts(
    *,
    source_values: np.ndarray,
    evaluation_split: str,
    tasks: list[str],
    operand_modes: list[str],
    contexts_per_source: int,
    seed: int,
    digit_place: str = "ones",
) -> pd.DataFrame:
    place = _validate_digit_place(digit_place)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
    for operand_mode in operand_modes:
        if operand_mode not in OPERAND_MODES:
            raise ValueError(f"Unsupported operand_mode: {operand_mode}")

    for source_value in np.asarray(source_values, dtype=np.int64).tolist():
        source_digit = int(_extract_digits([source_value], place)[0])
        for task in tasks:
            for operand_mode in operand_modes:
                partner_min, partner_max = _partner_bounds_for_all_targets(
                    source_value=int(source_value),
                    digit_place=place,
                    task=task,
                    operand_mode=operand_mode,
                )
                if partner_max < partner_min:
                    continue
                partner_values = np.arange(partner_min, partner_max + 1, dtype=np.int64)
                rng = np.random.default_rng(seed + (1009 * int(source_value)) + (97 * len(task)) + (11 if operand_mode == "x2" else 0))
                if len(partner_values) >= contexts_per_source:
                    selected = np.sort(rng.choice(partner_values, size=int(contexts_per_source), replace=False))
                else:
                    selected = np.sort(rng.choice(partner_values, size=int(contexts_per_source), replace=True))
                for context_idx, partner_value in enumerate(selected.tolist()):
                    if operand_mode == "x1":
                        x1 = int(source_value)
                        x2 = int(partner_value)
                    else:
                        x1 = int(partner_value)
                        x2 = int(source_value)
                    target = x1 + x2 if task == "addition" else x1 - x2
                    if not (0 <= target <= 999):
                        raise ValueError("Generated an invalid target despite partner bounds.")
                    rows.append(
                        {
                            "evaluation_split": str(evaluation_split),
                            "source_value": int(source_value),
                            "source_digit": int(source_digit),
                            "task": task,
                            "operand_mode": operand_mode,
                            "context_idx": int(context_idx),
                            "x1": int(x1),
                            "x2": int(x2),
                            "target": int(target),
                        }
                    )
    return (
        pd.DataFrame(rows)
        .sort_values(["evaluation_split", "task", "operand_mode", "source_digit", "source_value", "context_idx"])
        .reset_index(drop=True)
    )


def render_arithmetic_contexts(
    *,
    context_df: pd.DataFrame,
    templates: dict[str, str],
    tokenizer,
    arithmetic_prompt_config: ArithmeticPromptConfig | None = None,
    seed: int | None = None,
    namespace: str | None = None,
) -> pd.DataFrame:
    prompt_cfg = arithmetic_prompt_config or ArithmeticPromptConfig()
    grouped_candidates: dict[str, list[dict[str, Any]]] = {}
    if int(prompt_cfg.num_examples) > 0:
        if seed is None:
            raise ValueError("render_arithmetic_contexts requires `seed` when arithmetic few-shot prompting is enabled.")
        for task, subset in context_df.groupby("task", sort=False):
            grouped_candidates[str(task)] = subset.sort_values(
                ["evaluation_split", "source_value", "operand_mode", "context_idx", "x1", "x2", "target"]
            ).to_dict("records")
    rows: list[dict[str, Any]] = []
    for row in context_df.to_dict("records"):
        template = str(templates[row["task"]])
        example_rows: list[dict[str, int]] = []
        if int(prompt_cfg.num_examples) > 0:
            example_rows = sample_arithmetic_example_rows(
                candidate_rows=grouped_candidates[str(row["task"])],
                query_row=row,
                num_examples=int(prompt_cfg.num_examples),
                seed=int(seed),
                namespace=str(namespace or row["task"]),
            )
            prompt, token_idx = render_arithmetic_prompt(
                template=template,
                x1=int(row["x1"]),
                x2=int(row["x2"]),
                operand_mode=str(row["operand_mode"]),
                tokenizer=tokenizer,
                example_rows=example_rows,
                separator=str(prompt_cfg.separator),
            )
        else:
            prompt, spans = render_prompt_with_operand_spans(template, int(row["x1"]), int(row["x2"]))
            token_idx = resolve_token_idx(
                tokenizer=tokenizer,
                prompt=prompt,
                mention_span=spans[str(row["operand_mode"])],
                mode="mention_last",
            )
        target_ids = encode_integer(tokenizer, int(row["target"]))
        if len(target_ids) != 1:
            raise ValueError(f"Target {row['target']} is not single-token for prompt evaluation.")
        rows.append(
            {
                **row,
                "template": template,
                "prompt": prompt,
                "token_idx": int(token_idx),
                "target_token_id": int(target_ids[0]),
                "arithmetic_example_rows_json": json.dumps(example_rows),
                "arithmetic_num_examples": int(prompt_cfg.num_examples),
                "arithmetic_separator": str(prompt_cfg.separator),
            }
        )
    return pd.DataFrame(rows)


def evaluate_baseline_arithmetic_contexts(
    *,
    model,
    tokenizer,
    context_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    filter_baseline_correct_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_df = context_df.copy()
    if "example_id" not in context_df:
        context_df["example_id"] = np.arange(len(context_df), dtype=np.int64)
    logits_chunks, _, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=context_df["prompt"].tolist(),
        token_idxs=context_df["token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
        scaled_directions=None,
        capture_pre_activations=False,
    )
    logits = np.concatenate(logits_chunks, axis=0)
    probs = _softmax_numpy(logits)
    pred_ids = probs.argmax(axis=-1).astype(np.int64)
    pred_text = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in pred_ids.tolist()]
    pred_value = [_parse_optional_int(text) for text in pred_text]
    target_probs = probs[np.arange(len(context_df)), context_df["target_token_id"].to_numpy(dtype=np.int64)]
    baseline = context_df.assign(
        baseline_predicted_token_id=pred_ids,
        baseline_predicted_text=pred_text,
        baseline_predicted_value=pred_value,
        baseline_target_prob=target_probs.astype(np.float64),
        baseline_correct=(pred_ids == context_df["target_token_id"].to_numpy(dtype=np.int64)).astype(int),
        baseline_expected_value=np.nan,
    ).copy()

    numeric_token_rows = []
    for value in range(1000):
        token_ids = encode_integer(tokenizer, int(value))
        if len(token_ids) != 1:
            raise ValueError(f"Value {value} is not single-token under the current tokenizer.")
        numeric_token_rows.append(int(token_ids[0]))
    tracked_values = np.arange(1000, dtype=np.int64)
    tracked_token_ids = np.asarray(numeric_token_rows, dtype=np.int64)
    numeric_probs = probs[:, tracked_token_ids]
    numeric_mass = numeric_probs.sum(axis=1)
    baseline_expected = (numeric_probs * tracked_values.astype(np.float64)[None, :]).sum(axis=1) / np.clip(numeric_mass, 1e-8, None)
    baseline["baseline_expected_value"] = baseline_expected.astype(np.float64)
    baseline.attrs[PROBABILITY_BANK_ATTR] = CounterfactualProbabilityBank(
        baseline["example_id"].to_numpy(), tracked_token_ids, numeric_probs
    )

    eligible = baseline.loc[baseline["baseline_correct"] == 1].reset_index(drop=True) if filter_baseline_correct_only else baseline.copy()
    return baseline.reset_index(drop=True), eligible


def build_pair_intervention_frame(
    *,
    eligible_df: pd.DataFrame,
    tokenizer,
    source_digit: int,
    target_digit: int,
    digit_place: str = "ones",
) -> pd.DataFrame:
    place = _validate_digit_place(digit_place)
    subset = eligible_df.loc[eligible_df["source_digit"] == int(source_digit)].copy()
    if subset.empty:
        return subset
    if "example_id" not in subset.columns:
        subset = subset.reset_index(drop=False).rename(columns={"index": "example_id"})

    shifted_values = _replace_digit(
        subset["source_value"].to_numpy(dtype=np.int64),
        place,
        np.full(len(subset), int(target_digit), dtype=np.int64),
    )
    if np.any(shifted_values == subset["source_value"].to_numpy(dtype=np.int64)):
        raise ValueError("build_pair_intervention_frame received an identity intervention.")

    if (subset["operand_mode"] == "x1").all():
        shifted_x1 = shifted_values
        shifted_x2 = subset["x2"].to_numpy(dtype=np.int64)
    else:
        shifted_x1 = subset["x1"].to_numpy(dtype=np.int64).copy()
        shifted_x2 = subset["x2"].to_numpy(dtype=np.int64).copy()
        x1_mask = subset["operand_mode"].to_numpy() == "x1"
        shifted_x1[x1_mask] = shifted_values[x1_mask]
        shifted_x2[~x1_mask] = shifted_values[~x1_mask]

    shifted_target = np.where(
        subset["task"].to_numpy() == "addition",
        shifted_x1 + shifted_x2,
        shifted_x1 - shifted_x2,
    ).astype(np.int64)

    if np.any((shifted_target < 0) | (shifted_target > 999)):
        raise ValueError("Shifted target fell outside the supported 0..999 range.")

    target_token_ids = []
    for value in shifted_target.tolist():
        token_ids = encode_integer(tokenizer, int(value))
        if len(token_ids) != 1:
            raise ValueError(f"Shifted target {value} is not single-token.")
        target_token_ids.append(int(token_ids[0]))

    result = subset.assign(
        target_digit=int(target_digit),
        shifted_value=shifted_values.astype(np.int64),
        shifted_x1=shifted_x1.astype(np.int64),
        shifted_x2=shifted_x2.astype(np.int64),
        shifted_target=shifted_target.astype(np.int64),
        target_token_id=np.asarray(target_token_ids, dtype=np.int64),
        expected_value_baseline=subset["baseline_expected_value"].astype(np.float64),
    )
    return attach_counterfactual_baseline(result, eligible_df.attrs.get(PROBABILITY_BANK_ATTR))


def summarize_pair_results(results: pd.DataFrame) -> dict[str, Any]:
    predicted_numeric = pd.to_numeric(results["predicted_value"], errors="coerce")
    abs_distance = (predicted_numeric - results["shifted_target"].astype(float)).abs()
    return {
        "num_examples": int(len(results)),
        "number_accuracy": float(results["correct"].mean()),
        "mean_abs_distance_to_target": float(abs_distance.mean()) if abs_distance.notna().any() else float("nan"),
        "mean_target_prob_delta": float(results["target_prob_gain"].mean()),
        "counterfactual_probability_n": int(results["counterfactual_probability_available"].sum()),
        "numeric_prediction_rate": float(predicted_numeric.notna().mean()),
        "mean_expected_distance_to_target": float((results["expected_value"] - results["shifted_target"]).abs().mean()),
    }


def _numeric_tracking_arrays(tokenizer) -> tuple[np.ndarray, np.ndarray]:
    numeric_token_ids = []
    for value in range(1000):
        token_ids = encode_integer(tokenizer, int(value))
        if len(token_ids) != 1:
            raise ValueError(f"Value {value} is not single-token under the current tokenizer.")
        numeric_token_ids.append(int(token_ids[0]))
    tracked_values = np.arange(1000, dtype=np.int64)
    tracked_token_ids = np.asarray(numeric_token_ids, dtype=np.int64)
    return tracked_values, tracked_token_ids


def _run_digit_shift_interventions(
    *,
    model,
    tokenizer,
    eligible_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    delta_builder: Callable[[pd.DataFrame, int, int], np.ndarray],
    method_name: str,
    edit_family: str,
    digit_place: str = "ones",
    architecture: str | None = None,
    activation_postprocess_fn: Callable[[Any], Any] | None = None,
    source_digits: list[int] | None = None,
    target_digits: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if eligible_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tracked_values, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    allowed_source_digits = list(range(10)) if source_digits is None else [int(value) for value in source_digits]
    allowed_target_digits = list(range(10)) if target_digits is None else [int(value) for value in target_digits]

    for task in TASKS:
        task_df = eligible_df.loc[eligible_df["task"] == task].copy()
        if task_df.empty:
            continue
        for operand_mode in OPERAND_MODES:
            mode_df = task_df.loc[task_df["operand_mode"] == operand_mode].copy()
            if mode_df.empty:
                continue
            pair_frames: list[pd.DataFrame] = []
            pair_delta_rows: list[np.ndarray] = []
            for source_digit in allowed_source_digits:
                source_df = mode_df.loc[mode_df["source_digit"] == source_digit].copy()
                if source_df.empty:
                    continue
                for target_digit in allowed_target_digits:
                    if target_digit == source_digit:
                        continue
                    valid_df = build_pair_intervention_frame(
                        eligible_df=source_df,
                        tokenizer=tokenizer,
                        source_digit=source_digit,
                        target_digit=target_digit,
                        digit_place=digit_place,
                    )
                    if valid_df.empty:
                        continue
                    pair_frames.append(
                        valid_df.assign(
                            source_digit=int(source_digit),
                            target_digit=int(target_digit),
                        )
                    )
                    pair_delta_rows.append(np.asarray(delta_builder(valid_df, source_digit, target_digit), dtype=np.float32))

            if not pair_frames:
                continue

            batched_valid_df = pd.concat(pair_frames, ignore_index=True)
            batched_delta_rows = np.concatenate(pair_delta_rows, axis=0)
            logits_chunks, _, _ = _batch_forward_with_optional_direction(
                model=model,
                tokenizer=tokenizer,
                prompts=batched_valid_df["prompt"].tolist(),
                token_idxs=batched_valid_df["token_idx"].astype(int).tolist(),
                layer_idx=int(layer_idx),
                batch_size=int(batch_size),
                device=device,
                scaled_directions=batched_delta_rows,
                capture_pre_activations=False,
                activation_postprocess_fn=activation_postprocess_fn,
            )
            logits = np.concatenate(logits_chunks, axis=0)
            results = evaluate_logits_against_shifted_targets(
                tokenizer=tokenizer,
                valid_df=batched_valid_df,
                logits=logits,
                tracked_values=tracked_values,
                tracked_token_ids=tracked_token_ids,
            ).assign(
                method=str(method_name),
                edit_family=str(edit_family),
                architecture="" if architecture is None else str(architecture),
            )
            detail_frames.append(results)

            for (pair_source_digit, pair_target_digit), pair_results in results.groupby(["source_digit", "target_digit"], sort=True):
                summary_rows.append(
                    {
                        "method": str(method_name),
                        "edit_family": str(edit_family),
                        "architecture": "" if architecture is None else str(architecture),
                        "evaluation_split": str(pair_results["evaluation_split"].iloc[0]),
                        "task": task,
                        "operand_mode": operand_mode,
                        "source_digit": int(pair_source_digit),
                        "target_digit": int(pair_target_digit),
                        **summarize_pair_results(pair_results),
                    }
                )

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["evaluation_split", "method", "task", "operand_mode", "source_digit", "target_digit"])
        .reset_index(drop=True)
    )
    return detail_df, summary_df


def run_probe_manifold_interventions(
    *,
    model,
    tokenizer,
    probe: OnesDigitProbe,
    eligible_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    method_name: str | None = None,
    digit_place: str = "ones",
    activation_postprocess_fn: Callable[[Any], Any] | None = None,
    source_digits: list[int] | None = None,
    target_digits: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run_digit_shift_interventions(
        model=model,
        tokenizer=tokenizer,
        eligible_df=eligible_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        delta_builder=lambda valid_df, source_digit, target_digit: probe.intervention_delta(
            source_digits=np.full(len(valid_df), source_digit, dtype=np.int64),
            target_digits=np.full(len(valid_df), target_digit, dtype=np.int64),
        ),
        method_name=probe.architecture if method_name is None else str(method_name),
        edit_family="probe",
        digit_place=digit_place,
        architecture=probe.architecture,
        activation_postprocess_fn=activation_postprocess_fn,
        source_digits=source_digits,
        target_digits=target_digits,
    )


def run_exact_patch_interventions(
    *,
    model,
    tokenizer,
    eligible_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    method_name: str = "exact_patch",
    digit_place: str = "ones",
    activation_postprocess_fn: Callable[[Any], Any] | None = None,
    source_digits: list[int] | None = None,
    target_digits: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.gradient_steering import compute_counterfactual_patch_deltas

    return _run_digit_shift_interventions(
        model=model,
        tokenizer=tokenizer,
        eligible_df=eligible_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        delta_builder=lambda valid_df, _source_digit, _target_digit: compute_counterfactual_patch_deltas(
            model=model,
            tokenizer=tokenizer,
            valid_df=valid_df,
            layer_idx=int(layer_idx),
            batch_size=int(batch_size),
            device=device,
        ),
        method_name=str(method_name),
        edit_family="exact_patch",
        digit_place=digit_place,
        architecture=None,
        activation_postprocess_fn=activation_postprocess_fn,
        source_digits=source_digits,
        target_digits=target_digits,
    )


def aggregate_heatmap_metrics(summary_df: pd.DataFrame, *, aggregate_operand_modes: bool = True) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    group_cols = ["evaluation_split", "architecture", "task", "source_digit", "target_digit"]
    if "method" in summary_df.columns:
        group_cols.insert(1, "method")
    if "edit_family" in summary_df.columns:
        insert_at = 2 if "method" in summary_df.columns else 1
        group_cols.insert(insert_at, "edit_family")
    if not aggregate_operand_modes:
        insert_at = len(group_cols) - 2
        group_cols.insert(insert_at, "operand_mode")
    aggregated = (
        summary_df.groupby(group_cols, as_index=False)[
            ["number_accuracy", "mean_abs_distance_to_target", "mean_target_prob_delta", "numeric_prediction_rate"]
        ]
        .mean()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return aggregated
