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
    SOURCE_RUN,
    load_source_config,
)
from scripts.run_source_value_probe_capacity_sweep import _load_partition_rows  # noqa: E402
from src.activation_steering import resolve_token_idx  # noqa: E402
from src.gradient_steering import (  # noqa: E402
    forward_next_token_logits_with_delta,
    freeze_model_parameters,
)
from src.identity_prompting import render_arithmetic_prompt, render_identity_prompt  # noqa: E402
from src.magnitude_intervention import build_arithmetic_contexts  # noqa: E402
from src.model_loader import load_causal_lm, load_tokenizer  # noqa: E402
from src.numeric_prompt_style_pilot import PROMPT_STYLES, _format_target  # noqa: E402
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.property_residual_behavior import (  # noqa: E402
    ResidualCondition,
    activation_condition_metrics,
    all_conditions,
    condition_target_states,
    decompose_live_states,
    deterministic_derangement,
    random_complement_basis,
)
from src.property_subspace_synthesis import distribution_shift_detail  # noqa: E402
from src.source_value_prompt_style_representations import (  # noqa: E402
    _render_prompt_with_source_span,
    _style_by_id,
    _variant_by_id,
)
from src.token_utils import encode_integer  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    runtime_metadata,
    set_global_seed,
)


SEED = 20260723
DEFAULT_SCHEME_OUTPUT = ROOT / "out/property_probe_training_scheme_comparison/llama31_8b_layer10_v1"
DEFAULT_OUTPUT = ROOT / "out/property_residual_behavior/llama31_8b_layer10_v1"
SPLITS = tuple(f"split_{index:02d}" for index in range(10))
DEFAULT_DOSES = (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)
TASKS = (
    "addition_formula",
    "addition_verbalized",
    "subtraction_formula",
    "subtraction_verbalized",
    "multiplication_formula",
    "identity_patchscopes",
    "magnitude_comparison",
    "parity",
    "hundreds_digit",
    "units_digit",
)
TASK_DEFINITIONS: dict[str, dict[str, Any]] = {
    "addition_formula": {"kind": "canonical", "template": "source cached prompt"},
    "addition_verbalized": {
        "kind": "arithmetic",
        "task": "addition",
        "template": "The result of x1 plus x2 is ",
    },
    "subtraction_formula": {"kind": "arithmetic", "task": "subtraction", "template": "x1-x2="},
    "subtraction_verbalized": {
        "kind": "arithmetic",
        "task": "subtraction",
        "template": "The result of x1 minus x2 is ",
    },
    "multiplication_formula": {
        "kind": "arithmetic",
        "task": "multiplication",
        "template": "x1*x2=",
    },
    "identity_patchscopes": {"kind": "identity", "template": "identity few-shot k=4"},
    "magnitude_comparison": {
        "kind": "property_question",
        "style_id": "magnitude_comparison",
        "variant_id": "above_threshold",
        "fewshot_k": 3,
    },
    "parity": {
        "kind": "property_question",
        "style_id": "parity",
        "variant_id": "is_even",
        "fewshot_k": 3,
    },
    "hundreds_digit": {
        "kind": "property_question",
        "style_id": "hundreds_digit",
        "variant_id": "hundreds_digit",
        "fewshot_k": 3,
    },
    "units_digit": {
        "kind": "property_question",
        "style_id": "modulo_10",
        "variant_id": "remainder_mod_10",
        "fewshot_k": 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure causal necessity, sufficiency, identity, orientation, and dose-response of the "
            "activation residual orthogonal to the selected competitive property subspace."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--scheme-output-dir", type=Path, default=DEFAULT_SCHEME_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--residual-doses", default=",".join(str(value) for value in DEFAULT_DOSES))
    parser.add_argument(
        "--decomposition-basis",
        choices=("full", "selected"),
        default="full",
        help="Use the full trained competitive basis by default; selected is an explicit sensitivity run.",
    )
    parser.add_argument("--random-subspace-rank", type=int, default=None)
    parser.add_argument("--max-examples-per-task", type=int, default=None)
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _select(raw: str, available: Sequence[str]) -> tuple[str, ...]:
    if str(raw).strip().lower() == "all":
        return tuple(available)
    selected = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown selections {unknown}; available: {list(available)}")
    if not selected:
        raise ValueError("At least one selection is required.")
    return selected


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in (SEED, *parts)).encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _single_numeric_token(tokenizer, value: int) -> int | None:
    ids = encode_integer(tokenizer, int(value))
    return int(ids[0]) if len(ids) == 1 else None


def _single_text_token(tokenizer, text: str) -> int | None:
    # Boolean prompts end at "A:"; Llama's natural next token includes the word boundary.
    ids = tokenizer.encode(f" {str(text).strip()}", add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _three_demo_rows(task: str) -> list[dict[str, int]]:
    if task == "addition":
        pairs = ((17, 8), (24, 31), (105, 12))
        return [{"x1": x1, "x2": x2, "target": x1 + x2} for x1, x2 in pairs]
    if task == "subtraction":
        pairs = ((25, 8), (64, 31), (117, 12))
        return [{"x1": x1, "x2": x2, "target": x1 - x2} for x1, x2 in pairs]
    if task == "multiplication":
        pairs = ((7, 8), (12, 3), (25, 4))
        return [{"x1": x1, "x2": x2, "target": x1 * x2} for x1, x2 in pairs]
    raise ValueError(f"Unsupported demonstration task: {task!r}")


def _base_row(
    *,
    task_id: str,
    source_value: int,
    prompt: str,
    token_idx: int,
    target_text: str,
    target_token_id: int,
    x1: int | None = None,
    x2: int | None = None,
    target_numeric: float | None = None,
    output_kind: str = "int",
) -> dict[str, Any]:
    return {
        "task_id": str(task_id),
        "source_value": int(source_value),
        "prompt": str(prompt),
        "token_idx": int(token_idx),
        "target_text": str(target_text),
        "target_token_id": int(target_token_id),
        "target_numeric": np.nan if target_numeric is None else float(target_numeric),
        "output_kind": str(output_kind),
        "x1": np.nan if x1 is None else int(x1),
        "x2": np.nan if x2 is None else int(x2),
    }


def _canonical_addition_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            _base_row(
                task_id="addition_formula",
                source_value=int(row["source_value"]),
                prompt=str(row["prompt"]),
                token_idx=int(row["token_idx"]),
                target_text=str(int(row["target"])),
                target_token_id=int(row["target_token_id"]),
                x1=int(row["x1"]),
                x2=int(row["x2"]),
                target_numeric=float(row["target"]),
            )
        )
    return pd.DataFrame(rows)


def _addition_verbalized_rows(frame: pd.DataFrame, tokenizer) -> pd.DataFrame:
    definition = TASK_DEFINITIONS["addition_verbalized"]
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        target = int(row["target"])
        target_token_id = _single_numeric_token(tokenizer, target)
        if target_token_id is None:
            continue
        prompt, token_idx = render_arithmetic_prompt(
            template=str(definition["template"]),
            x1=int(row["x1"]),
            x2=int(row["x2"]),
            operand_mode=str(row["operand_mode"]),
            tokenizer=tokenizer,
            example_rows=_three_demo_rows("addition"),
        )
        rows.append(
            _base_row(
                task_id="addition_verbalized",
                source_value=int(row["source_value"]),
                prompt=prompt,
                token_idx=token_idx,
                target_text=str(target),
                target_token_id=target_token_id,
                x1=int(row["x1"]),
                x2=int(row["x2"]),
                target_numeric=float(target),
            )
        )
    return pd.DataFrame(rows)


def _subtraction_rows(values: np.ndarray, tokenizer, *, task_id: str) -> pd.DataFrame:
    definition = TASK_DEFINITIONS[task_id]
    contexts = build_arithmetic_contexts(
        source_values=np.asarray(values, dtype=np.int64),
        evaluation_split="test",
        tasks=["subtraction"],
        operand_modes=["x1"],
        contexts_per_source=1,
        seed=_seed(task_id, "contexts"),
    )
    rows: list[dict[str, Any]] = []
    for row in contexts.to_dict("records"):
        target = int(row["target"])
        target_token_id = _single_numeric_token(tokenizer, target)
        if target_token_id is None:
            continue
        prompt, token_idx = render_arithmetic_prompt(
            template=str(definition["template"]),
            x1=int(row["x1"]),
            x2=int(row["x2"]),
            operand_mode="x1",
            tokenizer=tokenizer,
            example_rows=_three_demo_rows("subtraction"),
        )
        rows.append(
            _base_row(
                task_id=task_id,
                source_value=int(row["source_value"]),
                prompt=prompt,
                token_idx=token_idx,
                target_text=str(target),
                target_token_id=target_token_id,
                x1=int(row["x1"]),
                x2=int(row["x2"]),
                target_numeric=float(target),
            )
        )
    return pd.DataFrame(rows)


def _multiplication_rows(values: np.ndarray, tokenizer) -> pd.DataFrame:
    definition = TASK_DEFINITIONS["multiplication_formula"]
    rows: list[dict[str, Any]] = []
    for source_value in np.asarray(values, dtype=np.int64).tolist():
        maximum_factor = 9 if int(source_value) == 0 else min(9, 999 // int(source_value))
        factor = max(1, int(maximum_factor))
        # Spread factors deterministically whenever more than one valid factor exists.
        factor = 1 + (_seed("multiplication", source_value) % factor)
        target = int(source_value) * int(factor)
        target_token_id = _single_numeric_token(tokenizer, target)
        if target_token_id is None:
            continue
        prompt, token_idx = render_arithmetic_prompt(
            template=str(definition["template"]),
            x1=int(source_value),
            x2=int(factor),
            operand_mode="x1",
            tokenizer=tokenizer,
            example_rows=_three_demo_rows("multiplication"),
        )
        rows.append(
            _base_row(
                task_id="multiplication_formula",
                source_value=int(source_value),
                prompt=prompt,
                token_idx=token_idx,
                target_text=str(target),
                target_token_id=target_token_id,
                x1=int(source_value),
                x2=int(factor),
                target_numeric=float(target),
            )
        )
    return pd.DataFrame(rows)


def _identity_rows(values: np.ndarray, tokenizer) -> pd.DataFrame:
    value_list = [int(value) for value in np.asarray(values, dtype=np.int64).tolist()]
    rows: list[dict[str, Any]] = []
    for index, source_value in enumerate(value_list):
        example_values = [value_list[(index + offset) % len(value_list)] for offset in range(1, 5)]
        prompt, token_idx = render_identity_prompt(
            value=source_value,
            tokenizer=tokenizer,
            example_values=example_values,
        )
        target_token_id = _single_numeric_token(tokenizer, source_value)
        if target_token_id is None:
            continue
        rows.append(
            _base_row(
                task_id="identity_patchscopes",
                source_value=source_value,
                prompt=prompt,
                token_idx=token_idx,
                target_text=str(source_value),
                target_token_id=target_token_id,
                target_numeric=float(source_value),
            )
        )
    return pd.DataFrame(rows)


def _property_question_rows(values: np.ndarray, tokenizer, *, task_id: str) -> pd.DataFrame:
    definition = TASK_DEFINITIONS[task_id]
    style = _style_by_id(str(definition["style_id"]))
    variant = _variant_by_id(style, str(definition["variant_id"]))
    rows: list[dict[str, Any]] = []
    for source_value in np.asarray(values, dtype=np.int64).tolist():
        prompt, source_span, _ = _render_prompt_with_source_span(
            style=style,
            variant=variant,
            value=int(source_value),
            fewshot_k=int(definition["fewshot_k"]),
            context_variant_idx=0,
        )
        token_idx = resolve_token_idx(
            tokenizer=tokenizer,
            prompt=prompt,
            mention_span=source_span,
            mode="mention_last",
        )
        target = style.target(int(source_value))
        target_text = _format_target(style, target)
        target_token_id = (
            _single_text_token(tokenizer, target_text)
            if style.output_kind == "bool"
            else _single_numeric_token(tokenizer, int(target))
        )
        if target_token_id is None:
            continue
        target_numeric = (1.0 if bool(target) else 0.0) if style.output_kind == "bool" else float(target)
        rows.append(
            _base_row(
                task_id=task_id,
                source_value=int(source_value),
                prompt=prompt,
                token_idx=token_idx,
                target_text=target_text,
                target_token_id=target_token_id,
                target_numeric=target_numeric,
                output_kind=style.output_kind,
            )
        )
    return pd.DataFrame(rows)


def build_task_frames(
    *,
    values: np.ndarray,
    canonical_frame: pd.DataFrame,
    tokenizer,
    selected_tasks: Sequence[str],
    max_examples_per_task: int | None,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    builders: dict[str, Any] = {
        "addition_formula": lambda: _canonical_addition_rows(canonical_frame),
        "addition_verbalized": lambda: _addition_verbalized_rows(canonical_frame, tokenizer),
        "subtraction_formula": lambda: _subtraction_rows(values, tokenizer, task_id="subtraction_formula"),
        "subtraction_verbalized": lambda: _subtraction_rows(
            values, tokenizer, task_id="subtraction_verbalized"
        ),
        "multiplication_formula": lambda: _multiplication_rows(values, tokenizer),
        "identity_patchscopes": lambda: _identity_rows(values, tokenizer),
        "magnitude_comparison": lambda: _property_question_rows(
            values, tokenizer, task_id="magnitude_comparison"
        ),
        "parity": lambda: _property_question_rows(values, tokenizer, task_id="parity"),
        "hundreds_digit": lambda: _property_question_rows(values, tokenizer, task_id="hundreds_digit"),
        "units_digit": lambda: _property_question_rows(values, tokenizer, task_id="units_digit"),
    }
    frames: dict[str, pd.DataFrame] = {}
    audit: list[dict[str, Any]] = []
    expected_values = set(int(value) for value in np.asarray(values, dtype=np.int64).tolist())
    for task_id in selected_tasks:
        frame = builders[task_id]()
        if frame.empty:
            raise ValueError(
                f"Task {task_id!r} has no single-token-valid targets for this tokenizer; "
                "the task cannot support matched next-token causal evaluation."
            )
        frame = frame.sort_values("source_value").reset_index(drop=True)
        generated_n = len(frame)
        missing_values = expected_values - set(frame["source_value"].astype(int).tolist())
        if max_examples_per_task is not None and len(frame) > int(max_examples_per_task):
            rng = np.random.default_rng(_seed(task_id, "example_cap"))
            keep = np.sort(rng.choice(len(frame), size=int(max_examples_per_task), replace=False))
            frame = frame.iloc[keep].reset_index(drop=True)
        frame.insert(0, "example_id", [f"{task_id}:{int(value):03d}" for value in frame["source_value"]])
        if frame["example_id"].duplicated().any():
            raise ValueError(f"Duplicate example ids in task {task_id!r}.")
        frames[task_id] = frame
        audit.append(
            {
                "task_id": task_id,
                "eligible_source_values": len(expected_values),
                "single_token_valid_n": generated_n,
                "single_token_dropped_n": len(missing_values),
                "evaluated_n": len(frame),
                "definition": json.dumps(TASK_DEFINITIONS[task_id], sort_keys=True),
            }
        )
    return frames, audit


def _metric_payload(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    rename = {
        "full_kl_original_to_synthesis": "kl_original_to_intervention",
        "full_kl_synthesis_to_original": "kl_intervention_to_original",
        "numeric_conditional_kl_original_to_synthesis": "numeric_kl_original_to_intervention",
        "synthesis_top1_token_id": "intervention_top1_token_id",
        "synthesis_correct": "intervention_correct",
        "synthesis_target_logprob": "intervention_target_logprob",
        "synthesis_target_prob": "intervention_target_prob",
        "synthesis_entropy": "intervention_entropy",
        "synthesis_numeric_mass": "intervention_numeric_mass",
        "synthesis_numeric_expected_value": "intervention_numeric_expected_value",
    }
    return {rename.get(key, key): np.asarray(value) for key, value in raw.items()}


def _summarize(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    detail = pd.concat(parts, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (split_name, task_id, condition_id), frame in detail.groupby(
        ["split", "task_id", "condition_id"], sort=True
    ):
        baseline_correct = frame["baseline_correct"].astype(bool)
        correct_subset = frame.loc[baseline_correct]
        rows.append(
            {
                "split": split_name,
                "task_id": task_id,
                "condition_id": condition_id,
                "evidence_role": str(frame["evidence_role"].iloc[0]),
                "decomposition_basis_mode": str(frame["decomposition_basis_mode"].iloc[0]),
                "decomposition_basis_rank": int(frame["decomposition_basis_rank"].iloc[0]),
                "n": len(frame),
                "baseline_accuracy": float(frame["baseline_correct"].mean()),
                "intervention_accuracy": float(frame["intervention_correct"].mean()),
                "top1_agreement": float(frame["top1_agreement"].mean()),
                "mean_kl_original_to_intervention": float(frame["kl_original_to_intervention"].mean()),
                "mean_target_logprob_delta": float(frame["target_logprob_delta"].mean()),
                "baseline_correct_n": int(baseline_correct.sum()),
                "baseline_correct_intervention_accuracy": (
                    float(correct_subset["intervention_correct"].mean()) if len(correct_subset) else np.nan
                ),
                "baseline_correct_mean_kl": (
                    float(correct_subset["kl_original_to_intervention"].mean()) if len(correct_subset) else np.nan
                ),
                "mean_replacement_delta_l2": float(frame["replacement_delta_l2"].mean()),
                "mean_live_projection_l2": float(frame["live_projection_l2"].mean()),
                "mean_live_residual_l2": float(frame["live_residual_l2"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_split_state(
    *, scheme_output_dir: Path, split_name: str, canonical_frame: pd.DataFrame
) -> tuple[dict[str, np.ndarray], Path]:
    state_path = scheme_output_dir / "states" / f"{split_name}.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing competitive state artifact: {state_path}")
    state_file = np.load(state_path)
    required = (
        "values",
        "actual",
        "train_center",
        "basis__competitive_deflation",
        "full_basis__competitive_deflation",
        "synthesis__competitive_deflation",
    )
    missing = [key for key in required if key not in state_file]
    if missing:
        raise ValueError(
            f"State artifact {state_path} is missing {missing}. Regenerate comparison geometry with train_center."
        )
    state = {key: np.asarray(state_file[key]) for key in required}
    frame_values = canonical_frame["source_value"].to_numpy(dtype=np.int64)
    if not np.array_equal(state["values"].astype(np.int64), frame_values):
        raise ValueError(f"State/canonical row misalignment for {split_name}.")
    return state, state_path


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)
    doses = tuple(float(value) for value in str(args.residual_doses).split(",") if value.strip())
    conditions = all_conditions(doses)
    condition_by_id = {condition.condition_id: condition for condition in conditions}
    selected_splits = _select(args.splits, SPLITS)
    selected_tasks = _select(args.tasks, TASKS)
    selected_condition_ids = _select(args.conditions, tuple(condition_by_id))
    selected_conditions = tuple(condition_by_id[condition_id] for condition_id in selected_condition_ids)
    if (
        "projection_plus_shuffled_residual" in selected_condition_ids
        and args.max_examples_per_task is not None
        and int(args.max_examples_per_task) < 2
    ):
        raise ValueError("The shuffled-residual condition requires at least two examples per task.")
    output_dir = ensure_dir(args.output_dir)
    cfg, model_entry, resolved = load_source_config(args.source_run_dir)
    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=str(resolved["model_revision"]),
    )

    prepared: dict[str, dict[str, Any]] = {}
    task_audit_rows: list[dict[str, Any]] = []
    provenance_states: list[dict[str, Any]] = []
    for split_name in selected_splits:
        partition_rows = _load_partition_rows(Path(cfg["split_dataset_dir"]), split_name, str(cfg["family"]))
        canonical_frame = partition_rows["test"].sort_values("source_value").reset_index(drop=True)
        state, state_path = _load_split_state(
            scheme_output_dir=args.scheme_output_dir,
            split_name=split_name,
            canonical_frame=canonical_frame,
        )
        task_frames, audits = build_task_frames(
            values=state["values"].astype(np.int64),
            canonical_frame=canonical_frame,
            tokenizer=tokenizer,
            selected_tasks=selected_tasks,
            max_examples_per_task=args.max_examples_per_task,
        )
        for row in audits:
            task_audit_rows.append({"split": split_name, **row})
        prepared[split_name] = {"state": state, "tasks": task_frames}
        provenance_states.append(
            {"split": split_name, "path": str(state_path.resolve()), "sha256": _sha256(state_path)}
        )

    atomic_write_csv(output_dir / "task_audit.csv", pd.DataFrame(task_audit_rows))
    design_payload = {
        "schema_version": 1,
        "seed": SEED,
        "splits": list(selected_splits),
        "tasks": list(selected_tasks),
        "conditions": [condition.__dict__ for condition in selected_conditions],
        "task_definitions": {task: TASK_DEFINITIONS[task] for task in selected_tasks},
        "invariants": {
            "unit": "one source-value prompt row within split and task",
            "model_id": str(model_entry["hf_id"]),
            "model_revision": str(resolved["model_revision"]),
            "layer": int(cfg["intervention_layer_idx"]),
            "hook_site": "selected source-number token at layer output",
            "comparison": "all conditions reuse the same prompt, token index, target, captured live state, and baseline logits",
            "replacement": "delta = condition target state - captured live state",
            "basis": (
                "full trained competitive_deflation basis from the same value-disjoint split"
                if args.decomposition_basis == "full"
                else "decoder-selected competitive_deflation basis from the same value-disjoint split"
            ),
            "decomposition_basis_mode": str(args.decomposition_basis),
            "center": "training-activation mean from the same value-disjoint split",
        },
        "claim_limits": {
            "center_plus_projection": "clean residual-necessity test at the selected intervention site",
            "center_plus_residual": (
                "residual sufficiency conditional on prompt/context information that may also lie in the residual; "
                "not evidence that residual information is exclusively numeric"
            ),
            "competitive_semantic_synthesis": (
                "addition-trained semantic-decoder transfer; prompt-style mismatch makes this descriptive outside "
                "the canonical addition task"
            ),
            "shuffled_and_random": "controls preserve residual row norm (shuffle exactly, random after norm matching) but alter identity/orientation",
            "baseline_filter": "report both all token-valid rows and the pre-intervention baseline-correct subset",
        },
        "state_provenance": provenance_states,
        "source_run": str(args.source_run_dir.resolve()),
        "scheme_output": str(args.scheme_output_dir.resolve()),
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "design_and_provenance.json", design_payload)
    if args.dry_run:
        atomic_write_text(output_dir / "DRY_RUN_DONE", "ok\n")
        print(output_dir)
        return

    parts_dir = ensure_dir(output_dir / "parts")
    todo = []
    for split_name in selected_splits:
        for task_id in selected_tasks:
            for condition in selected_conditions:
                path = parts_dir / split_name / task_id / f"{condition.condition_id}.csv"
                if args.force or not path.exists():
                    todo.append((split_name, task_id, condition, path))

    if todo:
        device = str(args.eval_device or cfg.get("device") or model_entry.get("device", "cpu"))
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
        batch_size = int(args.eval_batch_size or cfg.get("evaluation_batch_size", 100))

        for split_name in selected_splits:
            state = prepared[split_name]["state"]
            basis_key = (
                "full_basis__competitive_deflation"
                if args.decomposition_basis == "full"
                else "basis__competitive_deflation"
            )
            basis = np.asarray(state[basis_key], dtype=np.float64)
            center = np.asarray(state["train_center"], dtype=np.float64)
            values = np.asarray(state["values"], dtype=np.int64)
            synthesis_by_value = {
                int(value): np.asarray(row, dtype=np.float64)
                for value, row in zip(values.tolist(), state["synthesis__competitive_deflation"])
            }
            random_rank = int(args.random_subspace_rank or basis.shape[1])
            random_basis = random_complement_basis(
                hidden_dim=basis.shape[0],
                excluded_basis=basis,
                rank=random_rank,
                seed=_seed(split_name, "random_complement"),
            )
            for task_id in selected_tasks:
                task_todo = [entry for entry in todo if entry[0] == split_name and entry[1] == task_id]
                if not task_todo:
                    continue
                frame = prepared[split_name]["tasks"][task_id].copy()
                prompts = frame["prompt"].astype(str).tolist()
                token_idxs = frame["token_idx"].astype(int).tolist()
                baseline_logits, live_states = forward_next_token_logits_with_delta(
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
                if live_states is None:
                    raise RuntimeError(f"Failed to capture live states for {split_name}/{task_id}.")
                live = np.asarray(live_states, dtype=np.float64)
                projection, residual = decompose_live_states(live, center=center, basis=basis)
                semantic = np.stack(
                    [synthesis_by_value[int(value)] for value in frame["source_value"].astype(int)], axis=0
                )
                shuffle = (
                    deterministic_derangement(len(frame), seed=_seed(split_name, task_id, "shuffle"))
                    if len(frame) >= 2
                    else np.arange(len(frame), dtype=np.int64)
                )
                for _, _, condition, path in task_todo:
                    target_states = condition_target_states(
                        condition=condition,
                        live_states=live,
                        center=center,
                        projection=projection,
                        residual=residual,
                        semantic_synthesis=semantic,
                        shuffle_indices=shuffle,
                        random_basis=random_basis,
                    )
                    intervention_logits, _ = forward_next_token_logits_with_delta(
                        model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        prompts=prompts,
                        token_idxs=token_idxs,
                        layer_idx=int(cfg["intervention_layer_idx"]),
                        batch_size=batch_size,
                        device=device,
                        delta_rows=np.asarray(target_states - live, dtype=np.float32),
                        capture_selected=False,
                    )
                    raw_metrics = distribution_shift_detail(
                        baseline_logits=baseline_logits,
                        synthesis_logits=intervention_logits,
                        target_token_ids=frame["target_token_id"].to_numpy(dtype=np.int64),
                        numeric_token_ids=numeric_token_ids,
                        numeric_values=numeric_values,
                    )
                    result = frame.copy()
                    result.insert(0, "split", split_name)
                    result.insert(2, "condition_id", condition.condition_id)
                    result.insert(3, "evidence_role", condition.evidence_role)
                    result.insert(4, "residual_dose", condition.residual_dose)
                    result.insert(5, "decomposition_basis_mode", str(args.decomposition_basis))
                    result.insert(6, "decomposition_basis_rank", int(basis.shape[1]))
                    for key, values_array in _metric_payload(raw_metrics).items():
                        result[key] = values_array
                    for key, values_array in activation_condition_metrics(
                        live_states=live,
                        target_states=target_states,
                        projection=projection,
                        residual=residual,
                    ).items():
                        result[key] = values_array
                    ensure_dir(path.parent)
                    atomic_write_csv(path, result)
                    print(f"wrote {path}", flush=True)

    selected_parts: list[pd.DataFrame] = []
    for split_name in selected_splits:
        for task_id in selected_tasks:
            for condition in selected_conditions:
                path = parts_dir / split_name / task_id / f"{condition.condition_id}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Missing requested result part: {path}")
                selected_parts.append(pd.read_csv(path, low_memory=False))
    detail = pd.concat(selected_parts, ignore_index=True)
    atomic_write_csv(output_dir / "behavior_detail.csv", detail)
    atomic_write_csv(output_dir / "behavior_summary.csv", _summarize(selected_parts))
    atomic_write_text(output_dir / "DONE", "ok\n")
    print(output_dir)


if __name__ == "__main__":
    main()
