"""Matched property definitions and metrics for number-representation steering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Any, Callable, Sequence

import torch

from introlm.experiments.primality_dissociation import is_prime


INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")


@dataclass(frozen=True)
class PropertySpec:
    property_id: str
    display_name: str
    field_name: str
    labels: tuple[str, ...]
    label: Callable[[int], int]
    target: Callable[[int], int]
    rule: str
    views: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class JointPromptVariant:
    variant_id: str
    property_id: str
    view_id: str
    description: str
    demonstration_values: tuple[int, ...]
    include_rule: bool = False
    chat_multiturn: bool = False
    prompt_family: str = "matched"
    label_style: str = "default"
    capability_variant_id: str | None = None


def parity(value: int) -> int:
    return int(value) % 2


def divisible_by(divisor: int) -> Callable[[int], int]:
    return lambda value: int(int(value) % divisor == 0)


def units_digit(value: int) -> int:
    return int(value) % 10


def tens_digit(value: int) -> int:
    return (int(value) // 10) % 10


def hundreds_digit(value: int) -> int:
    return (int(value) // 100) % 10


def digit_count_2v3(value: int) -> int:
    digits = len(str(abs(int(value))))
    if digits not in (2, 3):
        raise ValueError(f"digit_count_2v3 excludes {value}")
    return digits - 2


def next_class(label: Callable[[int], int], num_classes: int) -> Callable[[int], int]:
    return lambda value: (label(int(value)) + 1) % int(num_classes)


PROPERTY_SPECS = {
    "parity": PropertySpec(
        property_id="parity",
        display_name="Parity / divisibility by 2",
        field_name="Parity",
        labels=("even", "odd"),
        label=parity,
        target=lambda value: 1 - parity(value),
        rule="An integer is even exactly when it is divisible by 2; otherwise it is odd.",
        views=("parity", "divisible_by_2"),
    ),
    "divisible_by_3": PropertySpec(
        property_id="divisible_by_3",
        display_name="Divisibility by 3",
        field_name="Divisible by 3",
        labels=("no", "yes"),
        label=divisible_by(3),
        target=lambda value: 1 - divisible_by(3)(value),
        rule="A base-10 integer is divisible by 3 exactly when its digit sum is divisible by 3.",
        views=("divisible_by_3",),
    ),
    "divisible_by_5": PropertySpec(
        property_id="divisible_by_5",
        display_name="Divisibility by 5",
        field_name="Divisible by 5",
        labels=("no", "yes"),
        label=divisible_by(5),
        target=lambda value: 1 - divisible_by(5)(value),
        rule="An integer is divisible by 5 exactly when its final decimal digit is 0 or 5.",
        views=("divisible_by_5",),
    ),
    "divisible_by_10": PropertySpec(
        property_id="divisible_by_10",
        display_name="Divisibility by 10",
        field_name="Divisible by 10",
        labels=("no", "yes"),
        label=divisible_by(10),
        target=lambda value: 1 - divisible_by(10)(value),
        rule="An integer is divisible by 10 exactly when its final decimal digit is 0.",
        views=("divisible_by_10",),
    ),
    "primality": PropertySpec(
        property_id="primality",
        display_name="Primality",
        field_name="Prime status",
        labels=("composite", "prime"),
        label=lambda value: int(is_prime(int(value))),
        target=lambda value: 1 - int(is_prime(int(value))),
        rule=(
            "An integer greater than 1 is prime exactly when no integer from 2 through "
            "its square root divides it evenly; 0 and 1 are not prime."
        ),
        views=("primality",),
    ),
    "units_digit": PropertySpec(
        property_id="units_digit",
        display_name="Units digit",
        field_name="Units digit",
        labels=tuple(str(value) for value in range(10)),
        label=units_digit,
        target=next_class(units_digit, 10),
        rule="The units digit is the final digit of the base-10 representation.",
        views=("units_digit",),
    ),
    "tens_digit": PropertySpec(
        property_id="tens_digit",
        display_name="Tens digit",
        field_name="Tens digit",
        labels=tuple(str(value) for value in range(10)),
        label=tens_digit,
        target=next_class(tens_digit, 10),
        rule="The tens digit is the second digit from the right, using 0 when absent.",
        views=("tens_digit",),
    ),
    "hundreds_digit": PropertySpec(
        property_id="hundreds_digit",
        display_name="Hundreds digit",
        field_name="Hundreds digit",
        labels=tuple(str(value) for value in range(10)),
        label=hundreds_digit,
        target=next_class(hundreds_digit, 10),
        rule="The hundreds digit is the third digit from the right, using 0 when absent.",
        views=("hundreds_digit",),
    ),
    "digit_count_2v3": PropertySpec(
        property_id="digit_count_2v3",
        display_name="Decimal digit count (2 versus 3)",
        field_name="Number of decimal digits",
        labels=("2", "3"),
        label=digit_count_2v3,
        target=lambda value: 1 - digit_count_2v3(value),
        rule="Numbers from 10 through 99 have two digits; numbers from 100 through 999 have three.",
        views=("digit_count_2v3",),
    ),
}


def candidate_values(property_id: str) -> list[int]:
    if property_id == "digit_count_2v3":
        return list(range(10, 1000))
    if property_id == "primality":
        return list(range(2, 1000))
    return list(range(1000))


def balanced_value_splits(
    property_id: str, seed: int = 20260717
) -> dict[str, list[int]]:
    """Build class-balanced, value-disjoint 80/10/10 splits."""

    if property_id in {"divisible_by_3", "primality"}:
        from introlm.experiments.number_property_behavior_screen import (
            value_splits as behavior_value_splits,
        )

        result = behavior_value_splits(property_id, seed)
        validate_value_splits(property_id, result)
        return result
    spec = PROPERTY_SPECS[property_id]
    by_class = {
        class_index: [
            value
            for value in candidate_values(property_id)
            if spec.label(value) == class_index
        ]
        for class_index in range(spec.num_classes)
    }
    per_class = min(len(values) for values in by_class.values())
    dev_count = max(1, round(per_class * 0.1))
    test_count = dev_count
    train_count = per_class - dev_count - test_count
    if train_count < 1:
        raise ValueError(f"not enough values for {property_id}")
    result = {"train": [], "dev": [], "test": []}
    for class_index, values in by_class.items():
        values = list(values)
        random.Random(f"{seed}:{property_id}:{class_index}").shuffle(values)
        selected = values[:per_class]
        result["train"].extend(selected[:train_count])
        result["dev"].extend(selected[train_count : train_count + dev_count])
        result["test"].extend(selected[train_count + dev_count :])
    for split in result:
        random.Random(f"{seed}:{property_id}:{split}").shuffle(result[split])
    validate_value_splits(property_id, result)
    return result


def balanced_occurrence_splits(
    property_id: str, seed: int = 20260717
) -> dict[str, list[int]]:
    """Build class-balanced occurrence splits while allowing values across splits.

    Split sizes match :func:`balanced_value_splits`. Values remain unique within a
    split, but each split is sampled independently, so a value may occur in more
    than one split. The runner renders overlapping values in split-specific probe
    contexts, making the rows distinct occurrences rather than duplicate strings.
    """

    spec = PROPERTY_SPECS[property_id]
    reference = balanced_value_splits(property_id, seed)
    counts = {
        split: len(reference[split]) // spec.num_classes
        for split in ("train", "dev", "test")
    }
    by_class = {
        class_index: [
            value
            for value in candidate_values(property_id)
            if spec.label(value) == class_index
        ]
        for class_index in range(spec.num_classes)
    }
    result = {"train": [], "dev": [], "test": []}
    for split in result:
        for class_index, values in by_class.items():
            candidates = list(values)
            random.Random(
                f"occurrence:{seed}:{property_id}:{split}:{class_index}"
            ).shuffle(candidates)
            result[split].extend(candidates[: counts[split]])
        random.Random(f"occurrence:{seed}:{property_id}:{split}:rows").shuffle(
            result[split]
        )
    validate_occurrence_splits(property_id, result)
    return result


def validate_occurrence_splits(
    property_id: str, splits: dict[str, Sequence[int]]
) -> None:
    spec = PROPERTY_SPECS[property_id]
    if set(splits) != {"train", "dev", "test"}:
        raise ValueError("expected train/dev/test splits")
    for split in ("train", "dev", "test"):
        values = [int(value) for value in splits[split]]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{split} is empty or contains duplicate values")
        counts = [
            sum(spec.label(value) == label for value in values)
            for label in range(spec.num_classes)
        ]
        if len(set(counts)) != 1:
            raise ValueError(f"{split} is not class balanced: {counts}")


def split_overlap_counts(splits: dict[str, Sequence[int]]) -> dict[str, int]:
    sets = {split: set(map(int, values)) for split, values in splits.items()}
    return {
        "train_dev": len(sets["train"].intersection(sets["dev"])),
        "train_test": len(sets["train"].intersection(sets["test"])),
        "dev_test": len(sets["dev"].intersection(sets["test"])),
        "all_three": len(sets["train"].intersection(sets["dev"], sets["test"])),
    }


def validate_value_splits(property_id: str, splits: dict[str, Sequence[int]]) -> None:
    spec = PROPERTY_SPECS[property_id]
    if set(splits) != {"train", "dev", "test"}:
        raise ValueError("expected train/dev/test splits")
    observed: set[int] = set()
    for split in ("train", "dev", "test"):
        values = [int(value) for value in splits[split]]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{split} is empty or contains duplicate values")
        if observed.intersection(values):
            raise ValueError(f"{split} overlaps an earlier split")
        observed.update(values)
        counts = [
            sum(spec.label(value) == label for value in values)
            for label in range(spec.num_classes)
        ]
        if len(set(counts)) != 1:
            raise ValueError(f"{split} is not class balanced: {counts}")


def split_hash(splits: dict[str, Sequence[int]]) -> str:
    payload = {
        split: [int(value) for value in splits[split]] for split in sorted(splits)
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def range_spanning_demonstrations(
    train_values: Sequence[int], property_id: str, per_class: int
) -> tuple[int, ...]:
    spec = PROPERTY_SPECS[property_id]

    def spaced(values: Sequence[int], count: int) -> list[int]:
        ordered = sorted(int(value) for value in values)
        if count == 1:
            return [ordered[len(ordered) // 2]]
        indices = [
            round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
        ]
        return [ordered[index] for index in indices]

    selected = {
        class_index: spaced(
            [value for value in train_values if spec.label(int(value)) == class_index],
            per_class,
        )
        for class_index in range(spec.num_classes)
    }
    return tuple(
        selected[class_index][example_index]
        for example_index in range(per_class)
        for class_index in range(spec.num_classes)
    )


def sampled_balanced_demonstrations(
    train_values: Sequence[int],
    property_id: str,
    per_class: int,
    *,
    seed: int,
    salt: str = "default",
) -> tuple[int, ...]:
    """Sample class-balanced demonstrations with an independently recorded seed."""

    spec = PROPERTY_SPECS[property_id]
    selected: dict[int, list[int]] = {}
    for class_index in range(spec.num_classes):
        candidates = [
            int(value)
            for value in train_values
            if spec.label(int(value)) == class_index
        ]
        if len(candidates) < per_class:
            raise ValueError(
                f"not enough class-{class_index} demonstrations for {property_id}"
            )
        random.Random(
            f"demonstrations:{seed}:{property_id}:{salt}:{class_index}"
        ).shuffle(candidates)
        selected[class_index] = candidates[:per_class]
    return tuple(
        selected[class_index][example_index]
        for example_index in range(per_class)
        for class_index in range(spec.num_classes)
    )


def balanced_value_sample(
    values: Sequence[int],
    property_id: str,
    size: int | None,
    *,
    seed: int,
    salt: str = "steering",
) -> list[int]:
    """Select a deterministic class-balanced subset for steering evaluation."""

    source = [int(value) for value in values]
    if size is None or size >= len(source):
        return source
    spec = PROPERTY_SPECS[property_id]
    if size <= 0 or size % spec.num_classes:
        raise ValueError(
            f"sample size must be a positive multiple of {spec.num_classes}"
        )
    per_class = size // spec.num_classes
    selected: list[int] = []
    for class_index in range(spec.num_classes):
        candidates = [value for value in source if spec.label(value) == class_index]
        if len(candidates) < per_class:
            raise ValueError(
                f"not enough class-{class_index} values for a sample of size {size}"
            )
        random.Random(
            f"value-sample:{seed}:{property_id}:{salt}:{class_index}"
        ).shuffle(candidates)
        selected.extend(candidates[:per_class])
    random.Random(f"value-sample:{seed}:{property_id}:{salt}:rows").shuffle(selected)
    return selected


def prompt_variants(
    train_values: Sequence[int],
    property_id: str,
    view_id: str | None = None,
    *,
    demonstration_seed: int | None = None,
) -> tuple[JointPromptVariant, ...]:
    spec = PROPERTY_SPECS[property_id]
    view = spec.views[0] if view_id is None else view_id
    per_class_values = (1, 2, 4) if spec.num_classes == 2 else (1, 2)
    variants = [
        JointPromptVariant(
            "zero_shot_structured",
            property_id,
            view,
            "Explicit instruction and output schema without demonstrations.",
            (),
            include_rule=True,
        )
    ]
    for count in per_class_values:
        demonstrations = (
            range_spanning_demonstrations(train_values, property_id, per_class=count)
            if demonstration_seed is None
            else sampled_balanced_demonstrations(
                train_values,
                property_id,
                per_class=count,
                seed=demonstration_seed,
                salt=f"matched-{view}-{count}",
            )
        )
        shots = count * spec.num_classes
        variants.extend(
            [
                JointPromptVariant(
                    f"structured_{shots}shot",
                    property_id,
                    view,
                    "Structured identity and property fields.",
                    demonstrations,
                ),
                JointPromptVariant(
                    f"rule_structured_{shots}shot",
                    property_id,
                    view,
                    "Explicit rule and structured identity/property fields.",
                    demonstrations,
                    include_rule=True,
                ),
            ]
        )
    demonstrations = (
        range_spanning_demonstrations(train_values, property_id, per_class=1)
        if demonstration_seed is None
        else sampled_balanced_demonstrations(
            train_values,
            property_id,
            per_class=1,
            seed=demonstration_seed,
            salt=f"multiturn-{view}",
        )
    )
    variants.append(
        JointPromptVariant(
            f"chat_multiturn_{len(demonstrations)}shot",
            property_id,
            view,
            "Separate user/assistant demonstration turns.",
            demonstrations,
            include_rule=True,
            chat_multiturn=True,
        )
    )
    if property_id in {"divisible_by_3", "primality"}:
        from introlm.experiments.number_property_behavior_screen import (
            build_prompt_variants as build_capability_variants,
        )

        capability_ids = {
            "structured_boolean_4shot",
            "structured_categorical_8shot",
            "chat_multiturn_8shot",
            "zero_shot_structured",
        }
        for capability in build_capability_variants(train_values, property_id):
            if capability.variant_id not in capability_ids:
                continue
            capability_demonstrations = capability.demonstration_values
            if demonstration_seed is not None and capability_demonstrations:
                per_class = len(capability_demonstrations) // spec.num_classes
                capability_demonstrations = sampled_balanced_demonstrations(
                    train_values,
                    property_id,
                    per_class=per_class,
                    seed=demonstration_seed,
                    salt=f"capability-{capability.variant_id}",
                )
            variants.append(
                JointPromptVariant(
                    variant_id=f"capability_{capability.variant_id}",
                    property_id=property_id,
                    view_id=view,
                    description=("Capability-screen prompt: " + capability.description),
                    demonstration_values=capability_demonstrations,
                    include_rule=capability.include_rule,
                    chat_multiturn=capability.chat_multiturn,
                    prompt_family="capability",
                    label_style=capability.label_style,
                    capability_variant_id=capability.variant_id,
                )
            )
    return tuple(variants)


def _view_field_and_labels(
    spec: PropertySpec, view_id: str, label_style: str = "default"
) -> tuple[str, tuple[str, ...]]:
    if label_style == "boolean":
        field = "Prime" if spec.property_id == "primality" else spec.field_name
        return field, ("no", "yes")
    if label_style == "categorical":
        if spec.property_id == "primality":
            return "Prime", ("composite", "prime")
        if spec.property_id == "divisible_by_3":
            return spec.field_name, ("not divisible", "divisible")
        raise ValueError(f"categorical labels are unsupported for {spec.property_id}")
    if label_style != "default":
        raise ValueError(f"unknown label style: {label_style}")
    if view_id == "divisible_by_2":
        return "Divisible by 2", ("no", "yes")
    return spec.field_name, spec.labels


def _answer(spec: PropertySpec, variant: JointPromptVariant, value: int) -> str:
    field, labels = _view_field_and_labels(spec, variant.view_id, variant.label_style)
    class_index = spec.label(value)
    if variant.view_id == "divisible_by_2":
        class_index = int(parity(value) == 0)
    return f"Result after adding zero: {value}\n{field}: {labels[class_index]}"


def prompt_messages(variant: JointPromptVariant, value: int) -> list[dict[str, str]]:
    if variant.prompt_family == "capability":
        from introlm.experiments.number_property_behavior_screen import (
            PromptVariant as CapabilityPromptVariant,
            prompt_messages as capability_prompt_messages,
        )

        capability = CapabilityPromptVariant(
            variant_id=str(variant.capability_variant_id),
            description=variant.description,
            property_id=variant.property_id,
            format_id="structured",
            demonstration_values=variant.demonstration_values,
            label_style=variant.label_style,
            include_rule=variant.include_rule,
            chat_multiturn=variant.chat_multiturn,
        )
        return capability_prompt_messages(capability, value)
    if variant.prompt_family != "matched":
        raise ValueError(f"unknown prompt family: {variant.prompt_family}")
    spec = PROPERTY_SPECS[variant.property_id]
    field, labels = _view_field_and_labels(spec, variant.view_id, variant.label_style)
    instruction = (
        "Copy the input exactly as the result after adding zero, then report "
        f"{field.lower()}. Use one of these labels exactly: {', '.join(labels)}."
    )
    if variant.include_rule:
        instruction += " " + spec.rule
    if variant.chat_multiturn:
        messages = [{"role": "system", "content": instruction}]
        for demonstration in variant.demonstration_values:
            messages.extend(
                [
                    {"role": "user", "content": f"Input: {demonstration}"},
                    {
                        "role": "assistant",
                        "content": _answer(spec, variant, demonstration),
                    },
                ]
            )
        messages.append({"role": "user", "content": f"Input: {value}"})
        return messages
    demonstrations = "\n\n".join(
        f"Input: {demonstration}\n{_answer(spec, variant, demonstration)}"
        for demonstration in variant.demonstration_values
    )
    content = instruction
    if demonstrations:
        content += "\n\n" + demonstrations
    content += f"\n\nInput: {value}\nResult after adding zero:"
    return [{"role": "user", "content": content}]


def direct_property_messages(
    property_id: str,
    value: int,
    *,
    raw: bool = False,
    demonstration_values: Sequence[int] = (),
) -> list[dict[str, str]]:
    spec = PROPERTY_SPECS[property_id]
    if raw:
        demonstrations = "\n\n".join(
            f"Integer: {demonstration}\n{spec.field_name}: "
            f"{spec.labels[spec.label(int(demonstration))]}"
            for demonstration in demonstration_values
        )
        query = f"Integer: {value}\n{spec.field_name}:"
        return [
            {
                "role": "user",
                "content": f"{demonstrations}\n\n{query}" if demonstrations else query,
            }
        ]
    if demonstration_values:
        rule = (
            "A whole number is even when its last decimal digit is 0, 2, 4, 6, "
            "or 8. It is odd when its last decimal digit is 1, 3, 5, 7, or 9."
            if property_id == "parity"
            else spec.rule
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"Report {spec.field_name.lower()} using exactly one of these "
                    f"labels: {', '.join(spec.labels)}. {rule}"
                ),
            }
        ]
        for demonstration in demonstration_values:
            messages.extend(
                [
                    {"role": "user", "content": f"Input: {demonstration}"},
                    {
                        "role": "assistant",
                        "content": (
                            f"{spec.field_name}: "
                            f"{spec.labels[spec.label(int(demonstration))]}"
                        ),
                    },
                ]
            )
        messages.append({"role": "user", "content": f"Input: {value}"})
        return messages
    return [
        {
            "role": "system",
            "content": (
                "Return only one label from this list: " + ", ".join(spec.labels) + "."
            ),
        },
        {"role": "user", "content": f"Integer: {value}\n{spec.field_name}:"},
    ]


def direct_identity_messages(
    value: int,
    *,
    raw: bool = False,
    demonstration_values: Sequence[int] = (),
) -> list[dict[str, str]]:
    if raw:
        demonstrations = "\n\n".join(
            f"Integer: {demonstration}\nCopy: {demonstration}"
            for demonstration in demonstration_values
        )
        query = f"Integer: {value}\nCopy:"
        return [
            {
                "role": "user",
                "content": f"{demonstrations}\n\n{query}" if demonstrations else query,
            }
        ]
    return [
        {
            "role": "system",
            "content": "Repeat the integer exactly and output no other text.",
        },
        {"role": "user", "content": f"Integer: {value}\nCopy:"},
    ]


def neutral_probe_messages(
    value: int, *, raw: bool = False, context_id: int = 0
) -> list[dict[str, str]]:
    templates = (
        "Integer: {value}",
        "Number: {value}",
        "Value: {value}",
        "Read this integer: {value}",
    )
    content = templates[int(context_id) % len(templates)].format(value=int(value))
    if raw:
        return [{"role": "user", "content": content}]
    return [
        {"role": "system", "content": "Read the integer."},
        {"role": "user", "content": content},
    ]


def parse_property_label(
    text: str,
    property_id: str,
    view_id: str | None = None,
    label_style: str = "default",
) -> int | None:
    spec = PROPERTY_SPECS[property_id]
    view = spec.views[0] if view_id is None else view_id
    field, labels = _view_field_and_labels(spec, view, label_style)
    lowered = text.lower()
    alternatives = "|".join(
        re.escape(label.lower()) for label in sorted(labels, key=len, reverse=True)
    )
    field_match = re.search(
        rf"{re.escape(field.lower())}\s*:\s*({alternatives})(?![a-z0-9])",
        lowered,
    )
    if field_match:
        label_index = labels.index(field_match.group(1))
        return (
            1 - label_index
            if view == "divisible_by_2" and label_style == "default"
            else label_index
        )
    matches: list[tuple[int, int]] = []
    for class_index, label in enumerate(labels):
        match = re.search(
            rf"(?<![a-z0-9]){re.escape(label.lower())}(?![a-z0-9])", lowered
        )
        if match:
            matches.append((match.start(), class_index))
    if not matches:
        return None
    label_index = min(matches)[1]
    return (
        1 - label_index
        if view == "divisible_by_2" and label_style == "default"
        else label_index
    )


def parse_identity(text: str) -> int | None:
    match = INTEGER_RE.search(text)
    return None if match is None else int(match.group(0))


def annotate_joint(
    *,
    value: int,
    text: str,
    property_id: str,
    view_id: str | None = None,
    label_style: str = "default",
) -> dict[str, Any]:
    spec = PROPERTY_SPECS[property_id]
    result = parse_identity(text)
    parsed_label = parse_property_label(text, property_id, view_id, label_style)
    source = spec.label(value)
    target = spec.target(value)
    identity = result == int(value)
    joint_parsed = result is not None and parsed_label is not None
    result_label = None
    if result is not None:
        try:
            result_label = spec.label(result)
        except ValueError:
            pass
    consistent = result_label is not None and parsed_label == result_label
    return {
        "parsed_result": result,
        "parsed_result_in_domain": result_label is not None,
        "parsed_label": parsed_label,
        "source_label": source,
        "target_label": target,
        "result_present": result is not None,
        "label_present": parsed_label is not None,
        "joint_parsed": joint_parsed,
        "identity_preserved": identity,
        "source_label_generated": parsed_label == source,
        "target_label_generated": parsed_label == target,
        "baseline_joint_correct": identity and parsed_label == source,
        "dissociated_identity_target": identity and parsed_label == target,
        "internally_consistent": joint_parsed and consistent,
        "internally_inconsistent": joint_parsed and not consistent,
        "coherent_changed_target": (
            result is not None
            and result != int(value)
            and result_label == target
            and parsed_label == target
        ),
        "extra_record_continuation": "\nInput:" in text or "\n\nUser:" in text,
    }


def summarize_joint(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    n = len(rows)

    def rate(key: str) -> float:
        return sum(bool(row[key]) for row in rows) / n

    return {
        "n": n,
        "joint_coverage": rate("joint_parsed"),
        "identity_preservation_rate": rate("identity_preserved"),
        "source_label_rate": rate("source_label_generated"),
        "target_label_rate": rate("target_label_generated"),
        "joint_accuracy": rate("baseline_joint_correct"),
        "dissociation_rate": rate("dissociated_identity_target"),
        "coherent_changed_target_rate": rate("coherent_changed_target"),
        "internal_inconsistency_rate": rate("internally_inconsistent"),
        "extra_record_continuation_rate": rate("extra_record_continuation"),
    }


def paired_condition_summary(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired rows must be nonempty and aligned")
    eligible = []
    for index, (before, after) in enumerate(zip(baseline, candidate, strict=True)):
        if int(before["value"]) != int(after["value"]):
            raise ValueError("paired values are not aligned")
        if before["baseline_joint_correct"]:
            eligible.append(index)
    contradictions = sum(
        candidate[index]["dissociated_identity_target"] for index in eligible
    )
    return {
        "baseline_joint_correct_count": len(eligible),
        "conditional_dissociation_count": int(contradictions),
        "conditional_dissociation_rate": (
            contradictions / len(eligible) if eligible else 0.0
        ),
    }


def pairwise_probe_directions(
    *,
    weight: torch.Tensor,
    feature_scale: torch.Tensor,
    train_activations: torch.Tensor,
    spec: PropertySpec,
) -> dict[int, dict[str, Any]]:
    if weight.shape != (spec.num_classes, feature_scale.numel()):
        raise ValueError(
            "probe weight and feature scale dimensions do not match property"
        )
    directions = {}
    for source in range(spec.num_classes):
        representative = next(
            value
            for value in candidate_values(spec.property_id)
            if spec.label(value) == source
        )
        target = spec.target(representative)
        raw = (weight[target].float() - weight[source].float()) / feature_scale.float()
        norm = raw.norm()
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise ValueError(f"degenerate direction {source}->{target}")
        unit = raw / norm
        scale = float((train_activations.float() @ unit).std(unbiased=False).item())
        directions[source] = {
            "source_class": source,
            "target_class": target,
            "unit": unit.cpu(),
            "projection_std": scale,
        }
    return directions


def random_control_directions(
    directions: dict[int, dict[str, Any]], seed: int
) -> dict[int, dict[str, Any]]:
    output = {}
    for source, payload in directions.items():
        unit = payload["unit"].float()
        generator = torch.Generator().manual_seed(int(seed) + int(source))
        random_unit = torch.randn(unit.shape, generator=generator)
        random_unit -= torch.dot(random_unit, unit) * unit
        random_unit /= random_unit.norm()
        output[source] = {
            **payload,
            "unit": random_unit,
        }
    return output
