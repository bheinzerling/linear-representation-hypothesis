"""Balanced list-filter prompts and metrics for numerical properties."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Callable, Sequence


INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")
BRACKETED_RE = re.compile(r"\[([^\]]*)\]", re.DOTALL)


def is_prime(value: int) -> bool:
    value = int(value)
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


@dataclass(frozen=True)
class PropertySpec:
    property_id: str
    selection_phrase: str
    rule: str
    label: Callable[[int], bool]


PROPERTY_SPECS = {
    "parity": PropertySpec(
        property_id="parity",
        selection_phrase="odd",
        rule="An integer is even exactly when it is divisible by 2.",
        # Match number_property_steering.py: class 0 is even, class 1 is odd.
        label=lambda value: int(value) % 2 == 1,
    ),
    "primality": PropertySpec(
        property_id="primality",
        selection_phrase="prime",
        rule=(
            "A prime number is an integer greater than 1 with no positive divisors "
            "other than 1 and itself."
        ),
        label=is_prime,
    ),
    "divisible_by_3": PropertySpec(
        property_id="divisible_by_3",
        selection_phrase="divisible by 3",
        rule=(
            "A number is divisible by 3 exactly when the sum of its decimal digits "
            "is divisible by 3."
        ),
        label=lambda value: int(value) % 3 == 0,
    ),
    "divisible_by_5": PropertySpec(
        property_id="divisible_by_5",
        selection_phrase="divisible by 5",
        rule=(
            "A number is divisible by 5 exactly when its final decimal digit is 0 or 5."
        ),
        label=lambda value: int(value) % 5 == 0,
    ),
    "divisible_by_10": PropertySpec(
        property_id="divisible_by_10",
        selection_phrase="divisible by 10",
        rule=(
            "A number is divisible by 10 exactly when its final decimal digit is 0."
        ),
        label=lambda value: int(value) % 10 == 0,
    ),
    "digit_count_2v3": PropertySpec(
        property_id="digit_count_2v3",
        selection_phrase="three-digit numbers",
        rule=(
            "A positive integer is a three-digit number exactly when it is between 100 "
            "and 999 inclusive."
        ),
        label=lambda value: 100 <= int(value) <= 999,
    ),
}


@dataclass(frozen=True)
class ListExample:
    example_id: str
    property_id: str
    values: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class PromptVariant:
    variant_id: str
    description: str
    output_style: str
    include_rule: bool = False
    numbered_input: bool = False
    demonstrations: tuple[ListExample, ...] = ()
    chat_multiturn: bool = False


def build_list_bank(
    property_id: str,
    *,
    seed: int,
    split: str,
    n_examples: int,
    list_size: int = 10,
) -> tuple[ListExample, ...]:
    """Build lists balanced within each occurrence and randomized in order."""
    if property_id not in PROPERTY_SPECS:
        raise ValueError(f"unknown property: {property_id}")
    if list_size <= 0 or list_size % 2:
        raise ValueError("list_size must be a positive even integer")
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    spec = PROPERTY_SPECS[property_id]
    candidates = tuple(range(10, 1000))
    by_label = {
        label: [value for value in candidates if bool(spec.label(value)) == label]
        for label in (False, True)
    }
    per_class = list_size // 2
    if any(len(values) < per_class for values in by_label.values()):
        raise ValueError("not enough values to construct a balanced list")
    property_offset = {
        "primality": 0,
        "divisible_by_3": 1_000_003,
        "parity": 2_000_033,
        "divisible_by_5": 3_000_043,
        "divisible_by_10": 4_000_037,
        "digit_count_2v3": 5_000_053,
    }[property_id]
    split_offset = sum((index + 1) * ord(char) for index, char in enumerate(split))
    rng = random.Random(int(seed) + property_offset + split_offset)
    rows: list[ListExample] = []
    for index in range(n_examples):
        values = rng.sample(by_label[False], per_class) + rng.sample(
            by_label[True], per_class
        )
        rng.shuffle(values)
        targets = tuple(value for value in values if spec.label(value))
        rows.append(
            ListExample(
                example_id=f"{property_id}-{split}-{index:04d}",
                property_id=property_id,
                values=tuple(values),
                targets=targets,
            )
        )
    return tuple(rows)


def build_prompt_variants(
    demonstrations: Sequence[ListExample],
) -> tuple[PromptVariant, ...]:
    demos = tuple(demonstrations)
    demos2 = demos[:2]
    demos4 = demos[:4]
    demos8 = demos[:8]
    return (
        PromptVariant(
            "bare_completion",
            "Literal list followed by the requested natural-language completion prefix.",
            "prefix",
        ),
        PromptVariant(
            "explicit_csv",
            "Explicit selection and source-order instructions with CSV-only output.",
            "csv",
        ),
        PromptVariant(
            "rule_csv",
            "Property rule plus explicit source-order CSV selection.",
            "csv",
            include_rule=True,
        ),
        PromptVariant(
            "numbered_csv",
            "One number per indexed line with CSV-only output.",
            "csv",
            numbered_input=True,
        ),
        PromptVariant(
            "rule_numbered_csv",
            "Property rule plus one number per indexed line and CSV-only output.",
            "csv",
            include_rule=True,
            numbered_input=True,
        ),
        PromptVariant(
            "json_array",
            "Explicit selection with a JSON-array-only response.",
            "json",
        ),
        PromptVariant(
            "fewshot_csv",
            "Two balanced list-filter demonstrations in one user message.",
            "csv",
            demonstrations=demos2,
        ),
        PromptVariant(
            "chat_multiturn_2shot",
            "Two balanced list-filter demonstrations as separate turns.",
            "csv",
            demonstrations=demos2,
            chat_multiturn=True,
        ),
        PromptVariant(
            "fewshot_numbered_csv_4",
            "Four numbered-list demonstrations in one user message.",
            "csv",
            numbered_input=True,
            demonstrations=demos4,
        ),
        PromptVariant(
            "fewshot_numbered_csv_8",
            "Eight numbered-list demonstrations in one user message.",
            "csv",
            numbered_input=True,
            demonstrations=demos8,
        ),
        PromptVariant(
            "chat_multiturn_numbered_4shot",
            "Four numbered-list demonstrations as separate turns.",
            "csv",
            numbered_input=True,
            demonstrations=demos4,
            chat_multiturn=True,
        ),
        PromptVariant(
            "chat_multiturn_numbered_8shot",
            "Eight numbered-list demonstrations as separate turns.",
            "csv",
            numbered_input=True,
            demonstrations=demos8,
            chat_multiturn=True,
        ),
    )


def _numbers_text(values: Sequence[int], *, numbered: bool) -> str:
    if numbered:
        return "\n".join(f"{index}. {value}" for index, value in enumerate(values, 1))
    return ", ".join(str(value) for value in values)


def expected_answer(example: ListExample, output_style: str) -> str:
    joined = ", ".join(str(value) for value in example.targets)
    if output_style == "json":
        return f"[{joined}]"
    if output_style in {"csv", "prefix"}:
        return joined
    raise ValueError(f"unknown output style: {output_style}")


def _query(example: ListExample, variant: PromptVariant) -> str:
    spec = PROPERTY_SPECS[example.property_id]
    numbers = _numbers_text(example.values, numbered=variant.numbered_input)
    if variant.variant_id == "bare_completion":
        return (
            f"Numbers: {numbers}\n"
            f"Among the above numbers, the following numbers are {spec.selection_phrase}:"
        )
    rule = f"{spec.rule}\n\n" if variant.include_rule else ""
    input_label = "Numbered list" if variant.numbered_input else "Numbers"
    output = (
        "Return only a JSON array of numbers."
        if variant.output_style == "json"
        else "Return only the selected numbers separated by commas."
    )
    return (
        f"{rule}Select every number that is {spec.selection_phrase}. Copy each selected "
        f"number exactly from the input and preserve the input order. {output}\n\n"
        f"{input_label}:\n{numbers}\nAnswer:"
    )


def prompt_messages(
    variant: PromptVariant, example: ListExample
) -> list[dict[str, str]]:
    if any(demo.property_id != example.property_id for demo in variant.demonstrations):
        raise ValueError("demonstrations and query must use the same property")
    if variant.chat_multiturn:
        spec = PROPERTY_SPECS[example.property_id]
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"Select all input numbers that are {spec.selection_phrase}. Copy them "
                    "exactly, preserve their input order, and return only comma-separated numbers."
                ),
            }
        ]
        demo_variant = PromptVariant(
            "numbered_csv" if variant.numbered_input else "explicit_csv",
            "",
            "csv",
            numbered_input=variant.numbered_input,
        )
        for demo in variant.demonstrations:
            messages.extend(
                [
                    {"role": "user", "content": _query(demo, demo_variant)},
                    {"role": "assistant", "content": expected_answer(demo, "csv")},
                ]
            )
        messages.append({"role": "user", "content": _query(example, demo_variant)})
        return messages

    parts: list[str] = []
    if variant.demonstrations:
        demo_variant = PromptVariant(
            "numbered_csv" if variant.numbered_input else "explicit_csv",
            "",
            "csv",
            numbered_input=variant.numbered_input,
        )
        for demo in variant.demonstrations:
            parts.append(f"{_query(demo, demo_variant)} {expected_answer(demo, 'csv')}")
    parts.append(_query(example, variant))
    return [{"role": "user", "content": "\n\n".join(parts)}]


def flatten_messages(messages: Sequence[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return (
        "\n\n".join(
            f"{message['role'].title()}: {message['content']}" for message in messages
        )
        + "\n\nAssistant:"
    )


def parse_number_sequence(text: str) -> tuple[int, ...]:
    """Parse the first answer-like numeric span without reading the prompt."""
    bracketed = BRACKETED_RE.search(text)
    if bracketed:
        return tuple(int(value) for value in INTEGER_RE.findall(bracketed.group(1)))
    for line in text.splitlines():
        values = INTEGER_RE.findall(line)
        if values:
            return tuple(int(value) for value in values)
    return ()


def annotate_completion(example: ListExample, text: str) -> dict[str, Any]:
    predicted = parse_number_sequence(text)
    target = example.targets
    predicted_set = set(predicted)
    target_set = set(target)
    source_set = set(example.values)
    true_positive = len(predicted_set & target_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(target_set)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "parsed_numbers": list(predicted),
        "target_numbers": list(target),
        "parse_coverage": float(bool(predicted)),
        "exact_sequence": float(predicted == target),
        "exact_set": float(
            predicted_set == target_set and len(predicted) == len(target)
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "out_of_input_count": len(predicted_set - source_set),
        "has_out_of_input": float(bool(predicted_set - source_set)),
        "duplicate_count": len(predicted) - len(predicted_set),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    mean_keys = (
        "parse_coverage",
        "exact_sequence",
        "exact_set",
        "precision",
        "recall",
        "f1",
        "has_out_of_input",
    )
    summary = {
        key: sum(float(row[key]) for row in rows) / len(rows) for key in mean_keys
    }
    summary.update(
        {
            "n_examples": len(rows),
            "total_out_of_input": sum(int(row["out_of_input_count"]) for row in rows),
            "total_duplicates": sum(int(row["duplicate_count"]) for row in rows),
        }
    )
    return summary


def select_prompt(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from empty prompt metrics")
    return max(
        rows,
        key=lambda row: (
            float(row["exact_set"]),
            float(row["exact_sequence"]),
            float(row["f1"]),
            -float(row["has_out_of_input"]),
            float(row["parse_coverage"]),
            str(row["variant_id"]),
        ),
    )
