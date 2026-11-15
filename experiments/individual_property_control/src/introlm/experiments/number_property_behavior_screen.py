"""Prompt construction and metrics for number-property capability screens."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Sequence

from introlm.experiments.divisible_by_3_steering import (
    balanced_value_splits as divisible_by_3_splits,
)
from introlm.experiments.primality_dissociation import (
    balanced_prime_value_splits,
    is_prime,
)


INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")
BOOLEAN_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
PRIME_STATUS_RE = re.compile(r"\b(prime|composite)\b", re.IGNORECASE)
PRIME_FIELD_RE = re.compile(r"\bPrime\s*:\s*(prime|composite)\b", re.IGNORECASE)
DIVISIBILITY_FIELD_RE = re.compile(
    r"\bDivisible by 3\s*:\s*(not\s+divisible|divisible)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class PropertySpec:
    property_id: str
    display_name: str
    field_name: str
    system_instruction: str
    rule_instruction: str
    label: Callable[[int], int]


@dataclass(frozen=True)
class PromptVariant:
    variant_id: str
    description: str
    property_id: str
    format_id: str
    demonstration_values: tuple[int, ...]
    label_style: str
    include_rule: bool = False
    chat_multiturn: bool = False


PROPERTY_SPECS = {
    "divisible_by_3": PropertySpec(
        property_id="divisible_by_3",
        display_name="Divisibility by 3",
        field_name="Divisible by 3",
        system_instruction=(
            "Copy the input as the result after adding zero, then determine whether the "
            "result is divisible by 3. Follow the requested output format exactly."
        ),
        rule_instruction=(
            "A base-10 integer is divisible by 3 exactly when the sum of its decimal "
            "digits is divisible by 3."
        ),
        label=lambda value: int(int(value) % 3 == 0),
    ),
    "primality": PropertySpec(
        property_id="primality",
        display_name="Primality",
        field_name="Prime",
        system_instruction=(
            "Copy the input as the result after adding zero, then determine whether the "
            "result is prime. Follow the requested output format exactly."
        ),
        rule_instruction=(
            "An integer greater than 1 is prime exactly when no integer from 2 through "
            "its square root divides it evenly."
        ),
        label=lambda value: int(is_prime(int(value))),
    ),
}


def value_splits(property_id: str, seed: int = 20260717) -> dict[str, list[int]]:
    if property_id == "divisible_by_3":
        return divisible_by_3_splits(seed)
    if property_id == "primality":
        return balanced_prime_value_splits(seed)
    raise ValueError(f"unknown property: {property_id}")


def range_spanning_demonstrations(
    values: Sequence[int], *, property_id: str, per_class: int
) -> tuple[int, ...]:
    spec = PROPERTY_SPECS[property_id]
    classes = {
        label: sorted(int(value) for value in values if spec.label(int(value)) == label)
        for label in (0, 1)
    }

    def spaced(items: Sequence[int], count: int) -> list[int]:
        if count == 1:
            return [items[len(items) // 2]]
        indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
        return [items[index] for index in indices]

    selected = {label: spaced(classes[label], per_class) for label in (0, 1)}
    return tuple(
        value
        for pair in zip(selected[1], selected[0], strict=True)
        for value in pair
    )


def build_prompt_variants(
    train_values: Sequence[int], property_id: str
) -> tuple[PromptVariant, ...]:
    variants: list[PromptVariant] = []
    for shots in (4, 8, 16):
        demos = range_spanning_demonstrations(
            train_values, property_id=property_id, per_class=shots // 2
        )
        variants.append(
            PromptVariant(
                f"structured_boolean_{shots}shot",
                "Structured input/result/property fields with yes/no labels.",
                property_id,
                "structured",
                demos,
                "boolean",
            )
        )
    demos = range_spanning_demonstrations(train_values, property_id=property_id, per_class=4)
    variants.extend(
        [
            PromptVariant(
                "structured_categorical_8shot",
                "Structured fields with prime/composite or divisible/not-divisible labels.",
                property_id,
                "structured",
                demos,
                "categorical",
            ),
            PromptVariant(
                "compact_fields_8shot",
                "Compact pipe-delimited fields.",
                property_id,
                "compact",
                demos,
                "boolean",
            ),
            PromptVariant(
                "table_8shot",
                "Markdown-style input/result/property table.",
                property_id,
                "table",
                demos,
                "boolean",
            ),
            PromptVariant(
                "json_8shot",
                "JSON-lines input/result/property records.",
                property_id,
                "json",
                demos,
                "boolean",
            ),
            PromptVariant(
                "question_8shot",
                "Natural-language questions with compact answers.",
                property_id,
                "question",
                demos,
                "boolean",
            ),
            PromptVariant(
                "rule_structured_8shot",
                "Explicit mathematical rule followed by structured demonstrations.",
                property_id,
                "structured",
                demos,
                "boolean",
                include_rule=True,
            ),
            PromptVariant(
                "chat_multiturn_8shot",
                "Separate user/assistant demonstration turns.",
                property_id,
                "structured",
                demos,
                "boolean",
                include_rule=True,
                chat_multiturn=True,
            ),
            PromptVariant(
                "zero_shot_structured",
                "Explicit instruction and output schema without demonstrations.",
                property_id,
                "structured",
                (),
                "boolean",
                include_rule=True,
            ),
        ]
    )
    return tuple(variants)


def label_text(spec: PropertySpec, value: int, style: str) -> str:
    positive = bool(spec.label(value))
    if style == "boolean":
        return "yes" if positive else "no"
    if style != "categorical":
        raise ValueError(f"unknown label style: {style}")
    if spec.property_id == "primality":
        return "prime" if positive else "composite"
    return "divisible" if positive else "not divisible"


def _answer(spec: PropertySpec, value: int, variant: PromptVariant) -> str:
    label = label_text(spec, value, variant.label_style)
    if variant.format_id == "structured":
        return f"Result after adding zero: {value}\n{spec.field_name}: {label}"
    if variant.format_id == "compact":
        return f"Result={value} | {spec.field_name}={label}"
    if variant.format_id == "table":
        return f"{value} | {label}"
    if variant.format_id == "json":
        key = "prime" if spec.property_id == "primality" else "divisible_by_3"
        return f'{{"result": {value}, "{key}": {str(bool(spec.label(value))).lower()}}}'
    if variant.format_id == "question":
        return f"{value}; {label}"
    raise ValueError(f"unknown format: {variant.format_id}")


def _query(spec: PropertySpec, value: int, variant: PromptVariant) -> str:
    if variant.format_id == "structured":
        return f"Input: {value}\nResult after adding zero:"
    if variant.format_id == "compact":
        return f"Input={value} | Result="
    if variant.format_id == "table":
        return f"{value} |"
    if variant.format_id == "json":
        return f'{{"input": {value}, "result":'
    if variant.format_id == "question":
        return f"Input {value}. Adding zero gives"
    raise ValueError(f"unknown format: {variant.format_id}")


def _demonstration(spec: PropertySpec, value: int, variant: PromptVariant) -> str:
    if variant.format_id == "structured":
        return f"Input: {value}\n{_answer(spec, value, variant)}"
    if variant.format_id == "compact":
        return f"Input={value} | {_answer(spec, value, variant)}"
    if variant.format_id == "table":
        return f"{value} | {_answer(spec, value, variant)}"
    if variant.format_id == "json":
        answer = _answer(spec, value, variant)
        return answer.replace('{"result":', f'{{"input": {value}, "result":', 1)
    if variant.format_id == "question":
        return f"Input {value}. Adding zero gives {_answer(spec, value, variant)}"
    raise ValueError(f"unknown format: {variant.format_id}")


def prompt_messages(variant: PromptVariant, value: int) -> list[dict[str, str]]:
    spec = PROPERTY_SPECS[variant.property_id]
    instruction = spec.system_instruction
    if variant.include_rule:
        instruction += " " + spec.rule_instruction
    if variant.format_id == "table":
        instruction += f" Use columns: input | result | {spec.field_name}."
    elif variant.format_id == "json":
        instruction += " Return one JSON object and no additional text."
    else:
        instruction += " Return only the requested result and property fields."

    if variant.chat_multiturn:
        messages = [{"role": "system", "content": instruction}]
        for demo in variant.demonstration_values:
            messages.extend(
                [
                    {"role": "user", "content": f"Input: {demo}"},
                    {"role": "assistant", "content": _answer(spec, demo, variant)},
                ]
            )
        messages.append({"role": "user", "content": f"Input: {value}"})
        return messages

    demonstrations = "\n\n".join(
        _demonstration(spec, demo, variant) for demo in variant.demonstration_values
    )
    content = instruction
    if demonstrations:
        content += "\n\n" + demonstrations
    content += "\n\n" + _query(spec, value, variant)
    return [{"role": "user", "content": content}]


def flatten_messages(messages: Sequence[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    parts = [f"{message['role'].title()}: {message['content']}" for message in messages]
    return "\n\n".join(parts) + "\n\nAssistant:"


def parse_completion(text: str, variant: PromptVariant) -> dict[str, Any]:
    integer = INTEGER_RE.search(text)
    if variant.label_style == "boolean":
        match = BOOLEAN_RE.search(text)
        label = None if match is None else int(match.group(1).lower() in {"yes", "true"})
    else:
        if variant.property_id == "primality":
            match = PRIME_FIELD_RE.search(text)
            if match is None:
                matches = PRIME_STATUS_RE.findall(text)
                status = None if not matches else matches[0]
            else:
                status = match.group(1)
            label = None if status is None else int(status.lower() == "prime")
        else:
            match = DIVISIBILITY_FIELD_RE.search(text)
            if match is None:
                matches = re.findall(
                    r"\b(not\s+divisible|divisible)\b", text, re.IGNORECASE
                )
                status = None if not matches else matches[0]
            else:
                status = match.group(1)
            label = None if status is None else int(not status.lower().startswith("not"))
    return {
        "parsed_result": None if integer is None else int(integer.group(0)),
        "parsed_label": label,
    }


def annotate_completion(
    *, value: int, text: str, variant: PromptVariant
) -> dict[str, Any]:
    parsed = parse_completion(text, variant)
    gold = PROPERTY_SPECS[variant.property_id].label(value)
    result_present = parsed["parsed_result"] is not None
    label_present = parsed["parsed_label"] is not None
    identity_correct = parsed["parsed_result"] == value
    property_correct = parsed["parsed_label"] == gold
    return {
        **parsed,
        "gold_label": gold,
        "result_present": result_present,
        "label_present": label_present,
        "joint_parsed": result_present and label_present,
        "identity_correct": identity_correct,
        "property_correct": property_correct,
        "joint_correct": identity_correct and property_correct,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    labels = (0, 1)
    class_accuracy = {}
    for label in labels:
        subset = [row for row in rows if int(row["gold_label"]) == label]
        class_accuracy[f"class_{label}_accuracy"] = sum(
            bool(row["joint_correct"]) for row in subset
        ) / len(subset)
    return {
        "n": len(rows),
        "result_coverage": sum(bool(row["result_present"]) for row in rows) / len(rows),
        "label_coverage": sum(bool(row["label_present"]) for row in rows) / len(rows),
        "joint_coverage": sum(bool(row["joint_parsed"]) for row in rows) / len(rows),
        "identity_accuracy": sum(bool(row["identity_correct"]) for row in rows) / len(rows),
        "property_accuracy": sum(bool(row["property_correct"]) for row in rows) / len(rows),
        "joint_accuracy": sum(bool(row["joint_correct"]) for row in rows) / len(rows),
        **class_accuracy,
    }


def select_prompt(
    rows: Sequence[dict[str, Any]], minimum_joint_coverage: float = 0.95
) -> dict[str, Any]:
    eligible = [
        row for row in rows if float(row["joint_coverage"]) >= minimum_joint_coverage
    ]
    if not eligible:
        raise RuntimeError("no prompt passed the parse-coverage gate")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["joint_accuracy"]),
            min(float(row["class_0_accuracy"]), float(row["class_1_accuracy"])),
            -int(row["n_demonstrations"]),
            str(row["variant_id"]),
        ),
    )
    return {
        **selected,
        "selection_mode": "coverage_gate_then_joint_accuracy",
        "minimum_joint_coverage": float(minimum_joint_coverage),
        "eligible_prompt_count": len(eligible),
    }
