"""Joint identity/parity prompts and metrics for parity-steering experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from introlm.experiments.parity_prompt_gate import (
    digit_coverage_demonstrations,
    parity_word,
)


INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")
PARITY_RE = re.compile(r"\b(even|odd)\b", re.IGNORECASE)


@dataclass(frozen=True)
class JointPromptVariant:
    variant_id: str
    description: str
    pattern: str
    demonstration_values: tuple[int, ...]

    def render(self, value: int) -> str:
        if self.pattern.count("{number}") != 1:
            raise ValueError(f"{self.variant_id} must contain exactly one query placeholder")
        return self.pattern.format(number=int(value))


def _semicolon_lines(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} + 0 = {value}; {value} is {parity_word(value)}" for value in values
    )


def _comma_lines(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} + 0 = {value}, and {value} is {parity_word(value)}."
        for value in values
    )


def _two_line_examples(values: Sequence[int]) -> str:
    return "\n\n".join(
        f"{value} + 0 = {value}\n{value} is {parity_word(value)}."
        for value in values
    )


def _structured_examples(values: Sequence[int]) -> str:
    return "\n\n".join(
        f"Input: {value}\nResult after adding zero: {value}\n"
        f"Parity of result: {parity_word(value)}"
        for value in values
    )


def _field_lines(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} | plus zero: {value} | parity: {parity_word(value)}"
        for value in values
    )


def build_joint_prompt_variants(train_values: Sequence[int]) -> tuple[JointPromptVariant, ...]:
    """Build joint readout prompts with demonstrations drawn only from train values."""

    digit_demos = digit_coverage_demonstrations(train_values, minimum_value=100)
    two_shot = (digit_demos[1], digit_demos[0])
    four_shot = (digit_demos[1], digit_demos[0], digit_demos[7], digit_demos[8])
    instruction = (
        "Complete each example by adding zero, then state whether the unchanged result "
        "is even or odd."
    )
    return (
        JointPromptVariant(
            "semicolon_2shot",
            "Two compact equation-and-parity demonstrations.",
            f"{_semicolon_lines(two_shot)}\n{{number}} + 0 =",
            two_shot,
        ),
        JointPromptVariant(
            "semicolon_4shot",
            "Four compact demonstrations spanning both parities and distant final digits.",
            f"{_semicolon_lines(four_shot)}\n{{number}} + 0 =",
            four_shot,
        ),
        JointPromptVariant(
            "semicolon_10shot",
            "One compact demonstration for every final decimal digit.",
            f"{_semicolon_lines(digit_demos)}\n{{number}} + 0 =",
            digit_demos,
        ),
        JointPromptVariant(
            "instructed_semicolon_10shot",
            "Explicit task instruction plus compact digit-covering demonstrations.",
            f"{instruction}\n{_semicolon_lines(digit_demos)}\n{{number}} + 0 =",
            digit_demos,
        ),
        JointPromptVariant(
            "comma_clause_10shot",
            "Equation results and parity labels joined by a natural-language clause.",
            f"{_comma_lines(digit_demos)}\n{{number}} + 0 =",
            digit_demos,
        ),
        JointPromptVariant(
            "two_line_10shot",
            "Equation and parity claims placed on separate lines.",
            f"{_two_line_examples(digit_demos)}\n\n{{number}} + 0 =",
            digit_demos,
        ),
        JointPromptVariant(
            "structured_10shot",
            "Explicit input, arithmetic result, and result-parity fields.",
            f"{_structured_examples(digit_demos)}\n\nInput: {{number}}\nResult after adding zero:",
            digit_demos,
        ),
        JointPromptVariant(
            "compact_fields_10shot",
            "Pipe-delimited identity and parity fields.",
            f"{_field_lines(digit_demos)}\n{{number}} | plus zero:",
            digit_demos,
        ),
    )


def parse_joint_completion(text: str) -> dict[str, Any]:
    integer_match = INTEGER_RE.search(text)
    parity_match = PARITY_RE.search(text)
    return {
        "parsed_result": int(integer_match.group(0)) if integer_match else None,
        "parsed_parity": parity_match.group(1).lower() if parity_match else None,
    }


def annotate_completion(*, value: int, text: str) -> dict[str, Any]:
    parsed = parse_joint_completion(text)
    result = parsed["parsed_result"]
    parity = parsed["parsed_parity"]
    source_parity = parity_word(value)
    target_parity = "odd" if source_parity == "even" else "even"
    joint_parsed = result is not None and parity is not None
    identity_preserved = result == int(value)
    parity_is_original = parity == source_parity
    parity_is_target = parity == target_parity
    internally_consistent = (
        joint_parsed and parity == parity_word(int(result))
    )
    return {
        **parsed,
        "source_parity": source_parity,
        "target_parity": target_parity,
        "result_present": result is not None,
        "parity_present": parity is not None,
        "joint_parsed": joint_parsed,
        "identity_preserved": identity_preserved,
        "parity_is_original": parity_is_original,
        "parity_is_target": parity_is_target,
        "baseline_joint_correct": identity_preserved and parity_is_original,
        "internally_consistent": internally_consistent,
        "internally_inconsistent": joint_parsed and not internally_consistent,
        "dissociated_identity_target_parity": identity_preserved and parity_is_target,
        "coherent_changed_target": (
            result is not None
            and result != int(value)
            and int(result) % 2 == 1 - int(value) % 2
            and parity_is_target
        ),
    }


def summarize_completions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty completion rows")
    n = len(rows)
    joint = [row for row in rows if row["joint_parsed"]]

    def rate(key: str, denominator: Sequence[dict[str, Any]] = rows) -> float:
        return sum(bool(row[key]) for row in denominator) / len(denominator)

    return {
        "n": n,
        "result_coverage": rate("result_present"),
        "parity_coverage": rate("parity_present"),
        "joint_coverage": rate("joint_parsed"),
        "identity_preservation_rate": rate("identity_preserved"),
        "original_parity_rate": rate("parity_is_original"),
        "target_parity_rate": rate("parity_is_target"),
        "joint_correct_rate": rate("baseline_joint_correct"),
        "dissociation_rate": rate("dissociated_identity_target_parity"),
        "coherent_changed_target_rate": rate("coherent_changed_target"),
        "internal_inconsistency_rate": rate("internally_inconsistent"),
        "internal_inconsistency_when_joint_parsed": (
            rate("internally_inconsistent", joint) if joint else 0.0
        ),
    }


def summarize_paired_conditions(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired conditions must be nonempty and aligned")
    for left, right in zip(baseline, candidate, strict=True):
        if int(left["value"]) != int(right["value"]):
            raise ValueError("paired conditions contain different values")
    eligible = [
        index for index, row in enumerate(baseline) if row["baseline_joint_correct"]
    ]
    dissociated = sum(
        bool(candidate[index]["dissociated_identity_target_parity"])
        for index in eligible
    )
    return {
        "baseline_joint_correct_count": len(eligible),
        "conditional_dissociation_count": dissociated,
        "conditional_dissociation_rate": dissociated / len(eligible) if eligible else 0.0,
    }


def select_joint_prompt(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_baseline_joint_accuracy: float = 0.90,
    minimum_joint_coverage: float = 0.95,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from empty prompt metrics")
    eligible = [
        row
        for row in rows
        if float(row["baseline_joint_correct_rate"]) >= minimum_baseline_joint_accuracy
        and float(row["baseline_joint_coverage"]) >= minimum_joint_coverage
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["conditional_dissociation_rate"]),
                float(row["baseline_joint_correct_rate"]),
                -int(row["n_demonstrations"]),
                str(row["variant_id"]),
            ),
        )
        selection_mode = "baseline_gate_then_conditional_dissociation"
    else:
        selected = max(
            rows,
            key=lambda row: (
                float(row["baseline_joint_correct_rate"]),
                float(row["baseline_joint_coverage"]),
                float(row["conditional_dissociation_rate"]),
                -int(row["n_demonstrations"]),
                str(row["variant_id"]),
            ),
        )
        selection_mode = "fallback_maximize_baseline_competence"
    return {
        **selected,
        "selection_mode": selection_mode,
        "minimum_baseline_joint_accuracy": float(minimum_baseline_joint_accuracy),
        "minimum_joint_coverage": float(minimum_joint_coverage),
        "eligible_prompt_count": len(eligible),
    }
