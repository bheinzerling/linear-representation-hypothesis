"""Utilities for value-disjoint divisibility-by-3 steering experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class ProbeExample:
    value: int
    label: int
    split: str
    text: str
    number_token_index: int
    number_token_id: int
    candidate_token_ids: tuple[int, int]
    candidate_texts: tuple[str, str]


@dataclass(frozen=True)
class JointPromptVariant:
    variant_id: str
    description: str
    pattern: str
    demonstration_values: tuple[int, ...]

    def render(self, value: int) -> str:
        if self.pattern.count("{number}") != 1:
            raise ValueError(f"{self.variant_id} must contain one query placeholder")
        return self.pattern.format(number=int(value))


@dataclass(frozen=True)
class SteeringDirection:
    unit: torch.Tensor
    projection_std: float


PROBE_PATTERN = (
    "Question: Is 42 divisible by 3? Answer: yes\n"
    "Question: Is 17 divisible by 3? Answer: no\n"
    "Question: Is {number} divisible by 3? Answer:"
)
PROBE_CANDIDATES = (" no", " yes")
INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")
DIGIT_SUM_RE = re.compile(r"digit\s*sum\s*[:=]\s*(-?\d+)", re.IGNORECASE)
DIVISIBLE_RE = re.compile(r"divisible\s+by\s+3\s*[:=]\s*(yes|no)", re.IGNORECASE)


def divisible_by_3_index(value: int) -> int:
    return int(int(value) % 3 == 0)


def divisible_by_3_word(value: int) -> str:
    return "yes" if divisible_by_3_index(value) else "no"


def decimal_digit_sum(value: int) -> int:
    return sum(int(character) for character in str(abs(int(value))))


def balanced_value_splits(seed: int = 20260717) -> dict[str, list[int]]:
    """Return an exactly balanced 660-value subset with 80/10/10 class splits."""

    rng = random.Random(int(seed))
    by_label = {
        label: [value for value in range(1000) if divisible_by_3_index(value) == label]
        for label in (0, 1)
    }
    for values in by_label.values():
        rng.shuffle(values)
    per_class = {"train": 264, "dev": 33, "test": 33}
    offsets = {0: 0, 1: 0}
    splits = {split: [] for split in per_class}
    for split, count in per_class.items():
        for label in (0, 1):
            start = offsets[label]
            stop = start + count
            splits[split].extend(by_label[label][start:stop])
            offsets[label] = stop
        random.Random(f"{seed}:{split}:divisible_by_3").shuffle(splits[split])
    validate_value_splits(splits)
    return splits


def validate_value_splits(splits: dict[str, Sequence[int]]) -> None:
    expected_per_class = {"train": 264, "dev": 33, "test": 33}
    if set(splits) != set(expected_per_class):
        raise ValueError(f"unexpected split names: {sorted(splits)}")
    observed: set[int] = set()
    for split, per_class in expected_per_class.items():
        values = list(map(int, splits[split]))
        if len(values) != 2 * per_class or len(set(values)) != len(values):
            raise ValueError(f"{split} must have {2 * per_class} unique values")
        if set(values) & observed:
            raise ValueError(f"{split} overlaps an earlier split")
        observed.update(values)
        counts = [sum(divisible_by_3_index(value) == label for value in values) for label in (0, 1)]
        if counts != [per_class, per_class]:
            raise ValueError(f"{split} is not balanced: {counts}")
    if len(observed) != 660 or not observed <= set(range(1000)):
        raise ValueError("splits must contain 660 distinct values from 0..999")


def value_split_hash(splits: dict[str, Iterable[int]]) -> str:
    payload = "|".join(
        f"{split}:{','.join(map(str, splits[split]))}" for split in ("train", "dev", "test")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {token_ids}")
    return int(token_ids[0])


def query_number_metadata(tokenizer: Any, text: str, prefix: str, value: int) -> tuple[int, int]:
    char_start = len(prefix)
    char_end = char_start + len(str(int(value)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    indices = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if max(int(start), char_start) < min(int(end), char_end)
    ]
    if len(indices) != 1:
        raise ValueError(f"query numeral {value} maps to token positions {indices}")
    index = indices[0]
    return index, int(encoded["input_ids"][index])


def build_probe_examples(
    *, values: Sequence[int], split: str, tokenizer: Any
) -> list[ProbeExample]:
    prefix, suffix = PROBE_PATTERN.split("{number}")
    candidate_ids = tuple(_single_token_id(tokenizer, text) for text in PROBE_CANDIDATES)
    examples = []
    for value in values:
        text = prefix + str(int(value)) + suffix
        token_index, token_id = query_number_metadata(tokenizer, text, prefix, int(value))
        examples.append(
            ProbeExample(
                value=int(value),
                label=divisible_by_3_index(value),
                split=split,
                text=text,
                number_token_index=token_index,
                number_token_id=token_id,
                candidate_token_ids=candidate_ids,
                candidate_texts=PROBE_CANDIDATES,
            )
        )
    return examples


def identity_prompt(value: int) -> str:
    return f"Repeat the integer exactly.\nInteger: 42\nCopy: 42\nInteger: {int(value)}\nCopy:"


def select_demonstrations(train_values: Sequence[int], per_class: int = 3) -> tuple[int, ...]:
    selected: list[int] = []
    for label in (1, 0):
        candidates = sorted(
            (int(value) for value in train_values if divisible_by_3_index(value) == label and value >= 100),
            key=lambda value: (value % 10, decimal_digit_sum(value), value),
        )
        chosen: list[int] = []
        used_units: set[int] = set()
        for value in candidates:
            if value % 10 not in used_units:
                chosen.append(value)
                used_units.add(value % 10)
            if len(chosen) == per_class:
                break
        if len(chosen) < per_class:
            remaining = [value for value in candidates if value not in chosen]
            chosen.extend(remaining[: per_class - len(chosen)])
        if len(chosen) != per_class:
            raise ValueError(f"could not select {per_class} demonstrations for label {label}")
        selected.extend(chosen)
    return tuple(selected)


def _structured(values: Sequence[int]) -> str:
    return "\n\n".join(
        f"Input: {value}\nResult after adding zero: {value}\n"
        f"Digit sum: {decimal_digit_sum(value)}\nDivisible by 3: {divisible_by_3_word(value)}"
        for value in values
    )


def _simple_structured(values: Sequence[int]) -> str:
    return "\n\n".join(
        f"Input: {value}\nResult after adding zero: {value}\n"
        f"Divisible by 3: {divisible_by_3_word(value)}"
        for value in values
    )


def _semicolon(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} + 0 = {value}; digit sum: {decimal_digit_sum(value)}; "
        f"divisible by 3: {divisible_by_3_word(value)}"
        for value in values
    )


def _fields(values: Sequence[int]) -> str:
    return "\n".join(
        f"input={value} | result={value} | digit sum={decimal_digit_sum(value)} | "
        f"divisible by 3={divisible_by_3_word(value)}"
        for value in values
    )


def build_joint_prompt_variants(train_values: Sequence[int]) -> tuple[JointPromptVariant, ...]:
    demos = select_demonstrations(train_values, per_class=3)
    short = (demos[0], demos[3])
    instruction = (
        "Add zero to each input. Report the unchanged result, its decimal digit sum, "
        "and whether it is divisible by 3."
    )
    return (
        JointPromptVariant(
            "divisible_by_3_simple_structured_6shot",
            "Parity-analogous structured identity and divisibility fields.",
            f"{_simple_structured(demos)}\n\nInput: {{number}}\nResult after adding zero:",
            demos,
        ),
        JointPromptVariant(
            "divisible_by_3_structured_6shot",
            "Structured fields with balanced demonstrations.",
            f"{_structured(demos)}\n\nInput: {{number}}\nResult after adding zero:",
            demos,
        ),
        JointPromptVariant(
            "divisible_by_3_instructed_structured_6shot",
            "Instruction and structured balanced demonstrations.",
            f"{instruction}\n\n{_structured(demos)}\n\nInput: {{number}}\nResult after adding zero:",
            demos,
        ),
        JointPromptVariant(
            "divisible_by_3_semicolon_6shot",
            "Compact equation, digit-sum, and divisibility fields.",
            f"{_semicolon(demos)}\n{{number}} + 0 =",
            demos,
        ),
        JointPromptVariant(
            "divisible_by_3_fields_2shot",
            "Short pipe-delimited prompt.",
            f"{_fields(short)}\ninput={{number}} | result=",
            short,
        ),
    )


def parse_joint_completion(text: str) -> dict[str, Any]:
    result_match = INTEGER_RE.search(text)
    digit_sum_match = DIGIT_SUM_RE.search(text)
    divisible_match = DIVISIBLE_RE.search(text)
    return {
        "parsed_result": int(result_match.group(0)) if result_match else None,
        "parsed_digit_sum": int(digit_sum_match.group(1)) if digit_sum_match else None,
        "parsed_divisible_by_3": divisible_match.group(1).lower() if divisible_match else None,
    }


def annotate_completion(*, value: int, text: str) -> dict[str, Any]:
    parsed = parse_joint_completion(text)
    result = parsed["parsed_result"]
    digit_sum = parsed["parsed_digit_sum"]
    label = parsed["parsed_divisible_by_3"]
    source_label = divisible_by_3_word(value)
    target_label = "no" if source_label == "yes" else "yes"
    joint_parsed = result is not None and label is not None
    identity_preserved = result == int(value)
    label_original = label == source_label
    label_target = label == target_label
    digit_sum_correct_for_result = result is not None and digit_sum == decimal_digit_sum(result)
    internally_consistent = joint_parsed and label == divisible_by_3_word(result)
    contradiction = identity_preserved and label_target
    triad_contradiction = contradiction and digit_sum_correct_for_result
    coherent_changed_target = (
        result is not None
        and result != int(value)
        and divisible_by_3_word(result) == target_label
        and label_target
    )
    if not joint_parsed:
        outcome = "parse_failure"
    elif contradiction:
        outcome = "identity_property_contradiction"
    elif coherent_changed_target:
        outcome = "coherent_identity_change_to_target"
    elif identity_preserved and label_original:
        outcome = "unchanged_joint_correct"
    elif identity_preserved:
        outcome = "identity_retained_other_property"
    else:
        outcome = "other_identity_change"
    return {
        **parsed,
        "source_divisible_by_3": source_label,
        "target_divisible_by_3": target_label,
        "result_present": result is not None,
        "digit_sum_present": digit_sum is not None,
        "property_present": label is not None,
        "joint_parsed": joint_parsed,
        "identity_preserved": identity_preserved,
        "property_is_original": label_original,
        "property_is_target": label_target,
        "baseline_joint_correct": identity_preserved and label_original,
        "digit_sum_correct_for_result": digit_sum_correct_for_result,
        "digit_sum_correct_for_source": digit_sum == decimal_digit_sum(value),
        "internally_consistent": internally_consistent,
        "internally_inconsistent": joint_parsed and not internally_consistent,
        "dissociated_identity_target_property": contradiction,
        "identity_digit_sum_target_property_contradiction": triad_contradiction,
        "coherent_changed_target": coherent_changed_target,
        "outcome": outcome,
    }


def summarize_completions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty completions")
    n = len(rows)
    eligible = [row for row in rows if row["joint_parsed"]]

    def rate(key: str, denominator: Sequence[dict[str, Any]] = rows) -> float:
        return sum(bool(row[key]) for row in denominator) / len(denominator)

    outcomes = sorted({str(row["outcome"]) for row in rows})
    return {
        "n": n,
        "result_coverage": rate("result_present"),
        "digit_sum_coverage": rate("digit_sum_present"),
        "property_coverage": rate("property_present"),
        "joint_coverage": rate("joint_parsed"),
        "identity_preservation_rate": rate("identity_preserved"),
        "original_property_rate": rate("property_is_original"),
        "target_property_rate": rate("property_is_target"),
        "joint_correct_rate": rate("baseline_joint_correct"),
        "dissociation_rate": rate("dissociated_identity_target_property"),
        "triad_contradiction_rate": rate("identity_digit_sum_target_property_contradiction"),
        "coherent_changed_target_rate": rate("coherent_changed_target"),
        "internal_inconsistency_rate": rate("internally_inconsistent"),
        "internal_inconsistency_when_joint_parsed": rate("internally_inconsistent", eligible) if eligible else 0.0,
        **{f"outcome_{outcome}_count": sum(row["outcome"] == outcome for row in rows) for outcome in outcomes},
    }


def summarize_paired_conditions(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired rows must be nonempty and aligned")
    for left, right in zip(baseline, candidate, strict=True):
        if int(left["value"]) != int(right["value"]):
            raise ValueError("paired rows contain different values")
    eligible = [index for index, row in enumerate(baseline) if row["baseline_joint_correct"]]
    contradictions = sum(candidate[index]["dissociated_identity_target_property"] for index in eligible)
    triad = sum(
        candidate[index]["identity_digit_sum_target_property_contradiction"] for index in eligible
    )
    return {
        "baseline_joint_correct_count": len(eligible),
        "conditional_dissociation_count": contradictions,
        "conditional_dissociation_rate": contradictions / len(eligible) if eligible else 0.0,
        "conditional_triad_contradiction_count": triad,
        "conditional_triad_contradiction_rate": triad / len(eligible) if eligible else 0.0,
    }


def select_joint_prompt(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_baseline_joint_accuracy: float = 0.90,
    minimum_joint_coverage: float = 0.95,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if float(row["baseline_joint_correct_rate"]) >= minimum_baseline_joint_accuracy
        and float(row["baseline_joint_coverage"]) >= minimum_joint_coverage
    ]
    if not eligible:
        raise RuntimeError(
            "no prompt passed the natural joint-accuracy and parse-coverage gates"
        )
    selected = max(
        eligible,
        key=lambda row: (
            float(row["conditional_dissociation_rate"]),
            float(row["conditional_triad_contradiction_rate"]),
            float(row["baseline_joint_correct_rate"]),
            -int(row["n_demonstrations"]),
            str(row["variant_id"]),
        ),
    )
    return {
        **selected,
        "selection_mode": "hard_gate_then_conditional_dissociation",
        "minimum_baseline_joint_accuracy": float(minimum_baseline_joint_accuracy),
        "minimum_joint_coverage": float(minimum_joint_coverage),
        "eligible_prompt_count": len(eligible),
    }


def normalized_direction(raw: torch.Tensor, train_activations: torch.Tensor) -> SteeringDirection:
    norm = raw.float().norm()
    if not torch.isfinite(norm) or float(norm.item()) <= 0:
        raise ValueError("direction has zero or non-finite norm")
    unit = raw.float() / norm
    projection_std = float((train_activations.float() @ unit).std(unbiased=False).item())
    if projection_std <= 0:
        raise ValueError("direction projection has zero variance")
    return SteeringDirection(unit=unit.cpu(), projection_std=projection_std)


def probe_direction_raw(weight: torch.Tensor, feature_scale: torch.Tensor) -> torch.Tensor:
    """Return the class-0 to class-1 contrast in raw activation coordinates."""

    return (weight[1].float() - weight[0].float()) / feature_scale.float()


def orthogonal_random_direction(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    reference = reference.float() / reference.float().norm()
    random_vector = torch.randn(reference.shape, generator=generator)
    random_vector -= torch.dot(random_vector, reference) * reference
    return random_vector / random_vector.norm()


def steering_sign(value: int) -> float:
    """Sign that moves a value toward its opposite divisibility class."""

    return -1.0 if divisible_by_3_index(value) else 1.0


def select_dev_condition(rows: Sequence[dict[str, Any]], identity_penalty: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty development sweep")
    scored = [
        {
            **row,
            "selection_score": float(row["conditional_flip_rate"])
            - float(identity_penalty) * (1.0 - float(row["identity_preservation_rate"])),
        }
        for row in rows
    ]
    return max(
        scored,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["conditional_flip_rate"]),
            float(row["identity_preservation_rate"]),
            float(row["mean_target_margin_gain"]),
            -float(row["alpha"]),
            -int(row["layer"]),
        ),
    )
