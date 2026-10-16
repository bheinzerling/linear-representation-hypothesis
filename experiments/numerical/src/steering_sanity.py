from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.activation_steering import (
    ActivationSteeringProbe,
    _batch_forward_with_optional_direction,
    _parse_optional_int,
    _softmax_numpy,
    build_numeric_token_info,
    circular_sinusoidal_encode,
    collect_prompt_activations,
    fit_activation_circular_probe,
    get_model_layers,
    make_task_examples,
    render_prompt_with_operand_spans,
    resolve_shifted_values,
    resolve_token_idx,
    sample_examples,
)
from src.arithmetic_eval import filter_single_token_targets
from src.counterfactual_scoring import CounterfactualProbabilityBank, attach_counterfactual_baseline
from src.embedding_extract import get_input_embeddings
from src.token_utils import encode_integer
from src.utils import atomic_write_csv, ensure_dir


@dataclass(frozen=True)
class SanityTestSpec:
    test_id: str
    hypothesis_id: str
    hypothesis: str
    reason: str
    kind: str
    layer: str
    params: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedSteeringSlice:
    task: str
    operand_mode: str
    template: str
    seed: int
    layer_idx: int
    prompt_df: pd.DataFrame
    baseline_all: pd.DataFrame
    eligible: pd.DataFrame
    tracked_values: np.ndarray
    tracked_token_ids: np.ndarray
    probability_bank: CounterfactualProbabilityBank | None = None


def resolve_named_layer(model, layer: str) -> int:
    total = len(get_model_layers(model))
    if layer == "embed":
        return 0
    if layer == "early":
        return min(1, total - 1)
    if layer == "mid":
        return min(total - 1, max(1, 1 + ((total - 2) // 2)))
    if layer == "late":
        return total - 1
    return int(layer)


def get_default_sanity_tests() -> list[SanityTestSpec]:
    return [
        SanityTestSpec(
            test_id="prompt_counterfactual",
            hypothesis_id="H0",
            hypothesis="If the evaluation protocol is sound, editing the prompt text to the shifted operand should make the model produce the shifted answer.",
            reason="This is the direct textual counterfactual and acts as an upper bound for any hidden-state intervention.",
            kind="prompt_counterfactual",
            layer="none",
            params={},
        ),
        SanityTestSpec(
            test_id="embedding_delta",
            hypothesis_id="H1",
            hypothesis="If operand identity at the embedding layer is causally used, replacing the operand embedding with the target number embedding should steer the answer.",
            reason="The target token embedding is an on-manifold state the model was trained on.",
            kind="embedding_delta",
            layer="embed",
            params={"lambda": 1.0},
        ),
        SanityTestSpec(
            test_id="embedding_interp_half",
            hypothesis_id="H1",
            hypothesis="If operand identity at the embedding layer is causally used, partial interpolation toward the target token embedding should partially steer the answer.",
            reason="Interpolating toward the target token tests whether the effect grows smoothly with distance along the exact embedding manifold.",
            kind="embedding_delta",
            layer="embed",
            params={"lambda": 0.5},
        ),
        SanityTestSpec(
            test_id="projected_embedding_delta",
            hypothesis_id="H2",
            hypothesis="If the causal part of exact embedding replacement lies in the learned numeric subspace, projecting the true embedding delta into that subspace should retain some effect.",
            reason="This isolates the component of exact embedding replacement that the circular probe can represent.",
            kind="projected_embedding_delta",
            layer="embed",
            params={},
        ),
        SanityTestSpec(
            test_id="circular_embed",
            hypothesis_id="H3",
            hypothesis="If the circular probe manifold is the causal numeric subspace at the embedding layer, inverse-mapped circular steering should move answers toward the shifted target.",
            reason="This is the direct probe-manifold intervention at the embedding layer.",
            kind="circular_delta",
            layer="embed",
            params={"norm_match": None},
        ),
        SanityTestSpec(
            test_id="circular_embed_norm_match_embedding",
            hypothesis_id="H3",
            hypothesis="If circular steering currently fails because its magnitude is mismatched, norm-matching it to the true embedding replacement should improve its effect.",
            reason="This isolates magnitude as a possible failure mode.",
            kind="circular_delta",
            layer="embed",
            params={"norm_match": "embedding"},
        ),
        SanityTestSpec(
            test_id="circular_early",
            hypothesis_id="H4",
            hypothesis="If causal numeric structure emerges only after contextualization, circular steering should work better in early residual layers than at the embedding layer.",
            reason="Later residual states may contain a cleaner task-relevant number representation than raw token embeddings.",
            kind="circular_delta",
            layer="early",
            params={"norm_match": None},
        ),
        SanityTestSpec(
            test_id="circular_mid",
            hypothesis_id="H4",
            hypothesis="If causal numeric structure emerges only after contextualization, circular steering should work better in mid residual layers than at the embedding layer.",
            reason="Mid-layer residual states are a natural place for arithmetic-relevant representations to become causal.",
            kind="circular_delta",
            layer="mid",
            params={"norm_match": None},
        ),
        SanityTestSpec(
            test_id="counterfactual_patch_early",
            hypothesis_id="H5",
            hypothesis="If the exact context-conditioned counterfactual operand state is causal, patching that state from the shifted prompt should strongly steer the answer.",
            reason="This uses a real hidden state from a shifted prompt rather than an abstract decoded delta.",
            kind="counterfactual_patch",
            layer="early",
            params={},
        ),
        SanityTestSpec(
            test_id="counterfactual_patch_mid",
            hypothesis_id="H5",
            hypothesis="If the exact context-conditioned counterfactual operand state is causal, patching that state from the shifted prompt should strongly steer the answer.",
            reason="Mid-layer patching tests whether task-specific contextual representations are needed for causal control.",
            kind="counterfactual_patch",
            layer="mid",
            params={},
        ),
        SanityTestSpec(
            test_id="random_global",
            hypothesis_id="H6",
            hypothesis="If the observed steering effect is semantic rather than generic perturbation, a same-norm random direction should not reliably produce the shifted target.",
            reason="This is the simplest matched-norm negative control.",
            kind="random_global",
            layer="embed",
            params={},
        ),
        SanityTestSpec(
            test_id="random_per_example",
            hypothesis_id="H6",
            hypothesis="If the observed steering effect is semantic rather than generic perturbation, fresh random directions per example should not reliably produce the shifted target.",
            reason="This removes any accidental alignment of one fixed random vector.",
            kind="random_per_example",
            layer="embed",
            params={},
        ),
        SanityTestSpec(
            test_id="random_subspace",
            hypothesis_id="H6",
            hypothesis="If the numeric subspace itself is special, random directions inside that subspace should still fail relative to the correct circular direction.",
            reason="This controls for dimensionality and subspace membership while scrambling semantics.",
            kind="random_subspace",
            layer="embed",
            params={},
        ),
        SanityTestSpec(
            test_id="wrong_target_embedding_delta",
            hypothesis_id="H6",
            hypothesis="If exact semantic alignment matters, moving toward the wrong token embedding should not produce the intended shifted answer.",
            reason="This preserves on-manifold structure while breaking numeric semantics.",
            kind="wrong_target_embedding_delta",
            layer="embed",
            params={},
        ),
    ]


def _alpha_slug(alpha: float) -> str:
    text = f"{float(alpha):+.3f}"
    return text.replace("+", "p").replace("-", "m").replace(".", "_")


def condition_stem(*, spec: SanityTestSpec, seed: int, task: str, operand_mode: str, alpha: float) -> str:
    return f"seed{int(seed)}__{task}__{operand_mode}__{spec.test_id}__alpha{_alpha_slug(alpha)}"


def aggregate_saved_conditions(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(output_dir)
    condition_dir = base / "conditions"
    sample_dir = base / "samples"

    condition_frames = [pd.read_csv(path) for path in sorted(condition_dir.glob("*.csv"))]
    summary = pd.concat(condition_frames, ignore_index=True) if condition_frames else pd.DataFrame()
    aggregated = (
        summary.groupby(
            ["hypothesis_id", "test_id", "hypothesis", "reason", "kind", "layer", "layer_idx", "task", "operand_mode", "alpha"],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            num_examples_mean=("num_examples", "mean"),
            num_examples_std=("num_examples", "std"),
            num_sampled_examples_mean=("num_sampled_examples", "mean"),
            num_eligible_examples_mean=("num_eligible_examples", "mean"),
            baseline_full_accuracy_mean=("baseline_full_accuracy", "mean"),
            baseline_full_accuracy_std=("baseline_full_accuracy", "std"),
            argmax_accuracy_mean=("argmax_accuracy", "mean"),
            argmax_accuracy_std=("argmax_accuracy", "std"),
            mean_expected_shift_mean=("mean_expected_shift", "mean"),
            mean_expected_shift_std=("mean_expected_shift", "std"),
            mean_abs_expected_shift_mean=("mean_abs_expected_shift", "mean"),
            mean_target_error_mean=("mean_target_error", "mean"),
            mean_abs_target_error_mean=("mean_abs_target_error", "mean"),
            mean_target_prob_gain_mean=("mean_target_prob_gain", "mean"),
            baseline_answer_retention_mean=("baseline_answer_retention", "mean"),
            numeric_prediction_rate_mean=("numeric_prediction_rate", "mean"),
            mean_predicted_numeric_delta_mean=("mean_predicted_numeric_delta", "mean"),
        )
        .reset_index()
        if not summary.empty
        else pd.DataFrame()
    )

    sample_frames = [pd.read_csv(path) for path in sorted(sample_dir.glob("*.csv"))]
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return summary, aggregated, sample_df


def prepare_steering_slice(
    *,
    model,
    tokenizer,
    task: str,
    template: str,
    operand_mode: str,
    layer_idx: int,
    batch_size: int,
    device: str,
    seed: int,
    max_examples: int | None,
    filter_baseline_correct_only: bool,
) -> PreparedSteeringSlice:
    numeric_info = build_numeric_token_info(tokenizer)
    tracked_values = numeric_info["values"]
    tracked_token_ids = numeric_info["token_ids"]
    tracked_values_float = tracked_values.astype(np.float64)

    examples = filter_single_token_targets(tokenizer, make_task_examples(task))
    examples = sample_examples(examples=examples, max_examples=max_examples, seed=seed)
    prompt_rows: list[dict[str, Any]] = []
    for example_id, ex in enumerate(examples):
        prompt, spans = render_prompt_with_operand_spans(template, ex.x1, ex.x2)
        token_idx = resolve_token_idx(
            tokenizer=tokenizer,
            prompt=prompt,
            mention_span=spans[operand_mode],
            mode="mention_last",
        )
        prompt_rows.append(
            {
                "example_id": int(example_id),
                "task": task,
                "x1": int(ex.x1),
                "x2": int(ex.x2),
                "target": int(ex.target),
                "prompt": prompt,
                "token_idx": int(token_idx),
                "template": template,
                "operand_mode": operand_mode,
            }
        )
    prompt_df = pd.DataFrame(prompt_rows)
    prompts = prompt_df["prompt"].tolist()
    token_idxs = prompt_df["token_idx"].astype(int).tolist()
    baseline_target_token_ids = [encode_integer(tokenizer, int(value))[0] for value in prompt_df["target"].tolist()]

    baseline_logits_chunks, _, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        scaled_directions=None,
        capture_pre_activations=False,
    )
    baseline_logits = np.concatenate(baseline_logits_chunks, axis=0)
    baseline_probs = _softmax_numpy(baseline_logits)
    baseline_pred_ids = baseline_probs.argmax(axis=-1)
    baseline_pred_text = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in baseline_pred_ids.tolist()]
    baseline_pred_value = [_parse_optional_int(text) for text in baseline_pred_text]
    baseline_numeric_probs = baseline_probs[:, tracked_token_ids]
    baseline_numeric_mass = baseline_numeric_probs.sum(axis=1)
    baseline_expected = (baseline_numeric_probs * tracked_values_float[None, :]).sum(axis=1) / np.clip(baseline_numeric_mass, 1e-8, None)
    baseline_target_prob = baseline_probs[np.arange(len(prompt_df)), np.asarray(baseline_target_token_ids, dtype=np.int64)]
    baseline_all = prompt_df.assign(
        alpha=0.0,
        shifted_x1=prompt_df["x1"],
        shifted_x2=prompt_df["x2"],
        shifted_target=prompt_df["target"],
        predicted_token_id=baseline_pred_ids.astype(np.int64),
        predicted_text=baseline_pred_text,
        predicted_value=baseline_pred_value,
        expected_value=baseline_expected.astype(np.float64),
        numeric_mass=baseline_numeric_mass.astype(np.float64),
        target_token_id=baseline_target_token_ids,
        target_prob=baseline_target_prob.astype(np.float64),
        correct=(baseline_pred_ids == np.asarray(baseline_target_token_ids)).astype(int),
        stage="baseline",
    )
    if filter_baseline_correct_only:
        eligible = baseline_all.loc[baseline_all["correct"] == 1].reset_index(drop=True)
    else:
        eligible = baseline_all.copy()
    return PreparedSteeringSlice(
        task=task,
        operand_mode=operand_mode,
        template=template,
        seed=seed,
        layer_idx=layer_idx,
        prompt_df=prompt_df,
        baseline_all=baseline_all,
        eligible=eligible,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
        probability_bank=CounterfactualProbabilityBank(
            prompt_df["example_id"].to_numpy(), tracked_token_ids, baseline_numeric_probs
        ),
    )


def make_valid_shifted_rows(
    *,
    slice_data: PreparedSteeringSlice,
    tokenizer,
    alpha: float,
) -> pd.DataFrame:
    valid_rows: list[dict[str, Any]] = []
    for _, row in slice_data.eligible.iterrows():
        shifted_x1, shifted_x2, shifted_target = resolve_shifted_values(
            task=slice_data.task,
            operand_mode=slice_data.operand_mode,
            x1=int(row["x1"]),
            x2=int(row["x2"]),
            alpha=float(alpha),
        )
        if not (0 <= shifted_x1 <= 999 and 0 <= shifted_x2 <= 999 and 0 <= shifted_target <= 999):
            continue
        target_ids = encode_integer(tokenizer, int(shifted_target))
        if len(target_ids) != 1:
            continue
        valid_rows.append(
            {
                **row.to_dict(),
                "alpha": float(alpha),
                "shifted_x1": int(shifted_x1),
                "shifted_x2": int(shifted_x2),
                "shifted_target": int(shifted_target),
                "target_token_id": int(target_ids[0]),
            }
        )
    return pd.DataFrame(valid_rows)


def build_shifted_prompts(valid_df: pd.DataFrame, *, tokenizer, operand_mode: str) -> pd.DataFrame:
    if {"shifted_prompt", "shifted_token_idx"}.issubset(valid_df.columns):
        return valid_df.loc[:, ["example_id", "shifted_prompt", "shifted_token_idx"]].copy()
    shifted_rows: list[dict[str, Any]] = []
    for _, row in valid_df.iterrows():
        prompt, spans = render_prompt_with_operand_spans(row["template"], int(row["shifted_x1"]), int(row["shifted_x2"]))
        token_idx = resolve_token_idx(
            tokenizer=tokenizer,
            prompt=prompt,
            mention_span=spans[operand_mode],
            mode="mention_last",
        )
        shifted_rows.append({"example_id": int(row["example_id"]), "shifted_prompt": prompt, "shifted_token_idx": int(token_idx)})
    return pd.DataFrame(shifted_rows)


def random_unit_rows(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    rows = rng.standard_normal(size=shape)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return rows / norms


def row_norms(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)


def orthonormal_row_basis(matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    _, singular_values, vh = np.linalg.svd(array, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    if rank <= 0:
        raise ValueError("Subspace basis is empty.")
    return vh[:rank]


def project_rows_onto_subspace(vectors: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    basis = orthonormal_row_basis(basis_matrix)
    return (np.asarray(vectors, dtype=np.float64) @ basis.T) @ basis


def evaluate_logits_against_shifted_targets(
    *,
    tokenizer,
    valid_df: pd.DataFrame,
    logits: np.ndarray,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
) -> pd.DataFrame:
    tracked_values_float = tracked_values.astype(np.float64)
    probs = _softmax_numpy(logits)
    pred_ids = probs.argmax(axis=-1).astype(np.int64)
    pred_text = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in pred_ids.tolist()]
    pred_value = [_parse_optional_int(text) for text in pred_text]
    numeric_probs = probs[:, tracked_token_ids]
    numeric_mass = numeric_probs.sum(axis=1)
    expected = (numeric_probs * tracked_values_float[None, :]).sum(axis=1) / np.clip(numeric_mass, 1e-8, None)
    target_probs = probs[np.arange(len(valid_df)), valid_df["target_token_id"].to_numpy(dtype=np.int64)]
    result = valid_df.assign(
        predicted_token_id=pred_ids,
        predicted_text=pred_text,
        predicted_value=pred_value,
        expected_value=expected.astype(np.float64),
        numeric_mass=numeric_mass.astype(np.float64),
        target_prob=target_probs.astype(np.float64),
        correct=(pred_ids == valid_df["target_token_id"].to_numpy(dtype=np.int64)).astype(int),
        stage="intervention",
    )
    result["expected_shift"] = result["expected_value"] - result["expected_value_baseline"]
    result["target_error"] = result["expected_value"] - result["shifted_target"]
    # Older serialized frames contain only the original gold-answer probability.
    # That is not a valid baseline for a different counterfactual answer.
    before = result.get("baseline_counterfactual_prob", pd.Series(np.nan, index=result.index)).copy()
    if "target" in result:
        unchanged = result["shifted_target"] == result["target"]
        before.loc[unchanged & before.isna()] = result.loc[unchanged & before.isna(), "baseline_target_prob"]
    result["baseline_counterfactual_prob"] = before
    result["counterfactual_probability_available"] = before.notna()
    result["target_prob_gain"] = result["target_prob"] - before
    original_before = result.get("baseline_original_target_prob", result["baseline_target_prob"])
    result["target_prob_vs_original_answer"] = result["target_prob"] - original_before
    result["baseline_answer_retained"] = (result["predicted_token_id"] == result["baseline_predicted_token_id"]).astype(int)
    pred_numeric = pd.to_numeric(result["predicted_value"], errors="coerce")
    base_numeric = pd.to_numeric(result["baseline_predicted_value"], errors="coerce")
    result["predicted_numeric_delta"] = pred_numeric - base_numeric
    return result


def summarize_condition(
    *,
    spec: SanityTestSpec,
    seed: int,
    layer_idx: int | None,
    alpha: float,
    valid_df: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, Any]:
    numeric_delta = pd.to_numeric(results["predicted_numeric_delta"], errors="coerce")
    return {
        "test_id": spec.test_id,
        "hypothesis_id": spec.hypothesis_id,
        "hypothesis": spec.hypothesis,
        "reason": spec.reason,
        "kind": spec.kind,
        "layer": spec.layer,
        "layer_idx": layer_idx,
        "seed": int(seed),
        "task": str(valid_df["task"].iloc[0]),
        "operand_mode": str(valid_df["operand_mode"].iloc[0]),
        "alpha": float(alpha),
        "num_examples": int(len(results)),
        "num_sampled_examples": int(valid_df["num_sampled_examples"].iloc[0]),
        "num_eligible_examples": int(valid_df["num_eligible_examples"].iloc[0]),
        "baseline_full_accuracy": float(valid_df["baseline_full_accuracy"].iloc[0]),
        "baseline_filter_mode": str(valid_df["baseline_filter_mode"].iloc[0]),
        "argmax_accuracy": float(results["correct"].mean()),
        "mean_expected_shift": float(results["expected_shift"].mean()),
        "mean_abs_expected_shift": float(results["expected_shift"].abs().mean()),
        "mean_target_error": float(results["target_error"].mean()),
        "mean_abs_target_error": float(results["target_error"].abs().mean()),
        "mean_target_prob_gain": float(results["target_prob_gain"].mean()),
        "counterfactual_probability_n": int(results["counterfactual_probability_available"].sum()),
        "baseline_answer_retention": float(results["baseline_answer_retained"].mean()),
        "numeric_prediction_rate": float(pd.to_numeric(results["predicted_value"], errors="coerce").notna().mean()),
        "mean_predicted_numeric_delta": float(numeric_delta.mean()) if numeric_delta.notna().any() else float("nan"),
    }


def add_baseline_metadata(valid_df: pd.DataFrame, slice_data: PreparedSteeringSlice, filter_baseline_correct_only: bool) -> pd.DataFrame:
    enriched = valid_df.copy()
    enriched["num_sampled_examples"] = int(len(slice_data.baseline_all))
    enriched["num_eligible_examples"] = int(len(slice_data.eligible))
    enriched["baseline_full_accuracy"] = float(slice_data.baseline_all["correct"].mean()) if len(slice_data.baseline_all) else float("nan")
    enriched["baseline_filter_mode"] = "correct_only" if filter_baseline_correct_only else "all_sampled"
    enriched = enriched.rename(
        columns={
            "expected_value": "expected_value_baseline",
            "predicted_token_id": "baseline_predicted_token_id",
            "predicted_text": "baseline_predicted_text",
            "predicted_value": "baseline_predicted_value",
            "target_prob": "baseline_target_prob",
            "correct": "baseline_correct",
        }
    )
    return attach_counterfactual_baseline(enriched, slice_data.probability_bank)


def compute_embedding_delta(
    *,
    values_from: np.ndarray,
    values_to: np.ndarray,
    embeddings: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    delta = embeddings[values_to] - embeddings[values_from]
    return np.asarray(delta, dtype=np.float64) * float(scale)


def compute_circular_delta(
    *,
    values_from: np.ndarray,
    values_to: np.ndarray,
    probe: ActivationSteeringProbe,
) -> np.ndarray:
    source_basis = circular_sinusoidal_encode(
        values_from,
        embedding_dim=int(probe.basis_dim),
        min_value=int(probe.support_min),
        max_value=int(probe.support_max) + 1,
    )
    target_basis = circular_sinusoidal_encode(
        values_to,
        embedding_dim=int(probe.basis_dim),
        min_value=int(probe.support_min),
        max_value=int(probe.support_max) + 1,
    )
    return np.asarray((target_basis - source_basis) @ probe.decoder_matrix, dtype=np.float64)


def scale_rows_to_match(reference: np.ndarray, target_norms: np.ndarray) -> np.ndarray:
    rows = np.asarray(reference, dtype=np.float64)
    ref_norms = np.linalg.norm(rows, axis=1, keepdims=True)
    ref_norms = np.where(ref_norms > 0, ref_norms, 1.0)
    return rows / ref_norms * np.asarray(target_norms, dtype=np.float64).reshape(-1, 1)


def wrong_target_values(values_to: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shift = int(rng.integers(1, 1000))
    return (np.asarray(values_to, dtype=np.int64) + shift) % 1000


def build_test_directions(
    *,
    spec: SanityTestSpec,
    valid_df: pd.DataFrame,
    embeddings: np.ndarray,
    probe: ActivationSteeringProbe | None,
    source_activations: np.ndarray | None,
    counterfactual_activations: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    operand_values = valid_df[valid_df["operand_mode"].iloc[0]].to_numpy(dtype=np.int64)
    shifted_operand = valid_df["shifted_x1"].to_numpy(dtype=np.int64) if valid_df["operand_mode"].iloc[0] == "x1" else valid_df["shifted_x2"].to_numpy(dtype=np.int64)
    embedding_target_delta = compute_embedding_delta(values_from=operand_values, values_to=shifted_operand, embeddings=embeddings, scale=1.0)
    target_norms = row_norms(embedding_target_delta)

    if spec.kind == "embedding_delta":
        return compute_embedding_delta(
            values_from=operand_values,
            values_to=shifted_operand,
            embeddings=embeddings,
            scale=float(spec.params.get("lambda", 1.0)),
        )
    if spec.kind == "projected_embedding_delta":
        if probe is None:
            raise ValueError("projected_embedding_delta requires a probe.")
        return project_rows_onto_subspace(embedding_target_delta, probe.decoder_matrix)
    if spec.kind == "circular_delta":
        if probe is None:
            raise ValueError("circular_delta requires a probe.")
        circular_delta = compute_circular_delta(values_from=operand_values, values_to=shifted_operand, probe=probe)
        if spec.params.get("norm_match") == "embedding":
            return scale_rows_to_match(circular_delta, target_norms)
        return circular_delta
    if spec.kind == "random_global":
        direction = random_unit_rows(rng, (1, embeddings.shape[1]))[0]
        return direction[None, :] * target_norms[:, None]
    if spec.kind == "random_per_example":
        return random_unit_rows(rng, embedding_target_delta.shape) * target_norms[:, None]
    if spec.kind == "random_subspace":
        if probe is None:
            raise ValueError("random_subspace requires a probe.")
        subspace_basis = orthonormal_row_basis(probe.decoder_matrix)
        coeffs = rng.standard_normal(size=(len(valid_df), subspace_basis.shape[0]))
        directions = coeffs @ subspace_basis
        return scale_rows_to_match(directions, target_norms)
    if spec.kind == "wrong_target_embedding_delta":
        wrong_values = wrong_target_values(shifted_operand, rng)
        return compute_embedding_delta(values_from=operand_values, values_to=wrong_values, embeddings=embeddings, scale=1.0)
    if spec.kind == "counterfactual_patch":
        if source_activations is None or counterfactual_activations is None:
            raise ValueError("counterfactual_patch requires source and counterfactual activations.")
        return np.asarray(counterfactual_activations - source_activations, dtype=np.float64)
    raise ValueError(f"Unsupported test kind for direction building: {spec.kind}")


def maybe_fit_probe_for_layers(
    *,
    model,
    tokenizer,
    test_specs: list[SanityTestSpec],
    prompt_template: str,
    basis_dim: int,
    batch_size: int,
    device: str,
) -> dict[int, ActivationSteeringProbe]:
    probes: dict[int, ActivationSteeringProbe] = {}
    for spec in test_specs:
        if spec.kind not in {"circular_delta", "projected_embedding_delta", "random_subspace"}:
            continue
        layer_idx = resolve_named_layer(model, spec.layer)
        if layer_idx in probes:
            continue
        probe, _ = fit_activation_circular_probe(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            prompt_template=prompt_template,
            token_mode="mention_last",
            value_min=0,
            value_max=999,
            basis_dim=int(basis_dim),
            batch_size=batch_size,
            device=device,
        )
        probes[layer_idx] = probe
    return probes


def evaluate_prompt_counterfactual(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    shifted_prompt_df = build_shifted_prompts(valid_df, tokenizer=tokenizer, operand_mode=str(valid_df["operand_mode"].iloc[0]))
    merged = valid_df.merge(shifted_prompt_df, on="example_id", how="left")
    logits_chunks, _, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=merged["shifted_prompt"].tolist(),
        token_idxs=merged["shifted_token_idx"].astype(int).tolist(),
        layer_idx=0,
        batch_size=batch_size,
        device=device,
        scaled_directions=None,
        capture_pre_activations=False,
    )
    logits = np.concatenate(logits_chunks, axis=0)
    return evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=merged,
        logits=logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )


def run_sanity_suite(
    *,
    model,
    tokenizer,
    templates: dict[str, str],
    test_specs: list[SanityTestSpec],
    seeds: list[int],
    tasks: list[str],
    operand_modes: list[str],
    alpha_values: list[float],
    probe_prompt_template: str,
    basis_dim: int,
    probe_batch_size: int,
    evaluation_batch_size: int,
    device: str,
    max_examples: int | None,
    filter_baseline_correct_only: bool,
    details_sample_per_condition: int,
    output_dir: str | Path | None = None,
    force_conditions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    embeddings = get_input_embeddings(model).detach().float().cpu().numpy()
    probes = maybe_fit_probe_for_layers(
        model=model,
        tokenizer=tokenizer,
        test_specs=test_specs,
        prompt_template=probe_prompt_template,
        basis_dim=int(basis_dim),
        batch_size=probe_batch_size,
        device=device,
    )
    probe_rows = []
    for layer_idx, probe in probes.items():
        probe_rows.append(
            {
                "layer_idx": int(layer_idx),
                "fit_decoded_accuracy": float(probe.fit_decoded_accuracy),
                "fit_basis_mse": float(probe.fit_basis_mse),
                "fit_mae": float(probe.fit_mae),
                "fit_rmse": float(probe.fit_rmse),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    manifest = pd.DataFrame([spec.to_jsonable() for spec in test_specs])
    condition_dir = None if output_dir is None else ensure_dir(Path(output_dir) / "conditions")
    sample_dir = None if output_dir is None else ensure_dir(Path(output_dir) / "samples")

    slice_cache: dict[tuple[str, str, int, int], PreparedSteeringSlice] = {}
    for seed in seeds:
        for task in tasks:
            template = templates[task]
            for operand_mode in operand_modes:
                for spec in test_specs:
                    layer_idx = 0 if spec.layer == "none" else resolve_named_layer(model, spec.layer)
                    cache_key = (task, operand_mode, seed, layer_idx)
                    if cache_key not in slice_cache:
                        slice_cache[cache_key] = prepare_steering_slice(
                            model=model,
                            tokenizer=tokenizer,
                            task=task,
                            template=template,
                            operand_mode=operand_mode,
                            layer_idx=layer_idx,
                            batch_size=evaluation_batch_size,
                            device=device,
                            seed=seed,
                            max_examples=max_examples,
                            filter_baseline_correct_only=filter_baseline_correct_only,
                        )
                    slice_data = slice_cache[cache_key]
                    probe = None if spec.layer == "none" else probes.get(layer_idx)
                    for alpha in alpha_values:
                        stem = condition_stem(
                            spec=spec,
                            seed=seed,
                            task=task,
                            operand_mode=operand_mode,
                            alpha=float(alpha),
                        )
                        summary_part_path = None if condition_dir is None else condition_dir / f"{stem}.csv"
                        sample_part_path = None if sample_dir is None else sample_dir / f"{stem}.csv"
                        if summary_part_path is not None and summary_part_path.exists() and not force_conditions:
                            cached_summary = pd.read_csv(summary_part_path)
                            if not cached_summary.empty:
                                summary_rows.extend(cached_summary.to_dict(orient="records"))
                            if sample_part_path is not None and sample_part_path.exists():
                                cached_sample = pd.read_csv(sample_part_path)
                                if not cached_sample.empty:
                                    sample_frames.append(cached_sample)
                            continue
                        valid_df = make_valid_shifted_rows(slice_data=slice_data, tokenizer=tokenizer, alpha=alpha)
                        if valid_df.empty:
                            continue
                        valid_df = add_baseline_metadata(valid_df, slice_data, filter_baseline_correct_only)
                        seed_offset = (
                            (seed * 1315423911)
                            + sum(ord(ch) for ch in f"{spec.test_id}:{task}:{operand_mode}")
                            + int(round(alpha * 1000))
                        ) % (2**32)
                        rng = np.random.default_rng(seed_offset)

                        if spec.kind == "prompt_counterfactual":
                            results = evaluate_prompt_counterfactual(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                batch_size=evaluation_batch_size,
                                device=device,
                            )
                        else:
                            source_activations = None
                            counterfactual_activations = None
                            if spec.kind == "counterfactual_patch":
                                source_activations = collect_prompt_activations(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompts=valid_df["prompt"].tolist(),
                                    token_idxs=valid_df["token_idx"].astype(int).tolist(),
                                    layer_idx=layer_idx,
                                    batch_size=evaluation_batch_size,
                                    device=device,
                                )
                                shifted_prompt_df = build_shifted_prompts(valid_df, tokenizer=tokenizer, operand_mode=operand_mode)
                                counterfactual_activations = collect_prompt_activations(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompts=shifted_prompt_df["shifted_prompt"].tolist(),
                                    token_idxs=shifted_prompt_df["shifted_token_idx"].astype(int).tolist(),
                                    layer_idx=layer_idx,
                                    batch_size=evaluation_batch_size,
                                    device=device,
                                )
                            directions = build_test_directions(
                                spec=spec,
                                valid_df=valid_df,
                                embeddings=embeddings,
                                probe=probe,
                                source_activations=source_activations,
                                counterfactual_activations=counterfactual_activations,
                                rng=rng,
                            )
                            logits_chunks, _, _ = _batch_forward_with_optional_direction(
                                model=model,
                                tokenizer=tokenizer,
                                prompts=valid_df["prompt"].tolist(),
                                token_idxs=valid_df["token_idx"].astype(int).tolist(),
                                layer_idx=layer_idx,
                                batch_size=evaluation_batch_size,
                                device=device,
                                scaled_directions=directions,
                                capture_pre_activations=False,
                            )
                            logits = np.concatenate(logits_chunks, axis=0)
                            results = evaluate_logits_against_shifted_targets(
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                logits=logits,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                            )

                        summary_row = summarize_condition(
                            spec=spec,
                            seed=seed,
                            layer_idx=None if spec.layer == "none" else layer_idx,
                            alpha=alpha,
                            valid_df=valid_df,
                            results=results,
                        )
                        summary_rows.append(summary_row)
                        if summary_part_path is not None:
                            atomic_write_csv(summary_part_path, pd.DataFrame([summary_row]))
                        if details_sample_per_condition > 0:
                            sample_n = min(int(details_sample_per_condition), len(results))
                            sampled = results.sample(n=sample_n, random_state=seed).assign(test_id=spec.test_id, seed=seed, layer=str(spec.layer))
                            sample_frames.append(sampled)
                            if sample_part_path is not None:
                                atomic_write_csv(sample_part_path, sampled)

    summary = pd.DataFrame(summary_rows)
    aggregated = (
        summary.groupby(["hypothesis_id", "test_id", "kind", "layer", "layer_idx", "task", "operand_mode", "alpha"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            num_examples_mean=("num_examples", "mean"),
            baseline_full_accuracy_mean=("baseline_full_accuracy", "mean"),
            argmax_accuracy_mean=("argmax_accuracy", "mean"),
            argmax_accuracy_std=("argmax_accuracy", "std"),
            mean_expected_shift_mean=("mean_expected_shift", "mean"),
            mean_expected_shift_std=("mean_expected_shift", "std"),
            mean_abs_expected_shift_mean=("mean_abs_expected_shift", "mean"),
            mean_target_error_mean=("mean_target_error", "mean"),
            mean_abs_target_error_mean=("mean_abs_target_error", "mean"),
            mean_target_prob_gain_mean=("mean_target_prob_gain", "mean"),
            baseline_answer_retention_mean=("baseline_answer_retention", "mean"),
        )
        .reset_index()
    )
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    probe_df = pd.DataFrame(probe_rows).sort_values("layer_idx").reset_index(drop=True) if probe_rows else pd.DataFrame()
    return manifest, probe_df, summary, aggregated if not aggregated.empty else summary, sample_df
