"""Utilities for value-disjoint parity activation-steering experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    split: str
    task: str
    pattern: str
    candidate_texts: tuple[str, ...]


@dataclass(frozen=True)
class PromptExample:
    value: int
    parity_index: int
    split: str
    template_id: str
    task: str
    text: str
    number_token_index: int
    number_token_id: int
    candidate_token_ids: tuple[int, ...]
    candidate_texts: tuple[str, ...]
    gold_index: int


@dataclass(frozen=True)
class SteeringDirection:
    unit: torch.Tensor
    projection_std: float


PARITY_TEMPLATES = (
    PromptTemplate(
        "parity_train_classify",
        "train",
        "parity",
        "Classify each integer as even or odd.\n42: even\n17: odd\n{number}:",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_train_question",
        "train",
        "parity",
        "Question: What is the parity of 42? Answer: even\nQuestion: What is the parity of 17? Answer: odd\nQuestion: What is the parity of {number}? Answer:",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_train_arrow",
        "train",
        "parity",
        "42 -> even\n17 -> odd\n{number} ->",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_dev_decide",
        "dev",
        "parity",
        "Decide whether each number is odd or even.\n42: even\n17: odd\n{number}:",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_dev_property",
        "dev",
        "parity",
        "The parity of 42 is even. The parity of 17 is odd. The parity of {number} is",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_test_choice",
        "test",
        "parity",
        "Is 42 even or odd? Answer: even\nIs 17 even or odd? Answer: odd\nIs {number} even or odd? Answer:",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_test_write",
        "test",
        "parity",
        "For 42, write the parity: even\nFor 17, write the parity: odd\nFor {number}, write the parity:",
        (" even", " odd"),
    ),
    PromptTemplate(
        "parity_test_divisible_by_two",
        "test",
        "divisible_by_two",
        "Is 42 divisible by two? Answer: yes\nIs 17 divisible by two? Answer: no\nIs {number} divisible by two? Answer:",
        (" yes", " no"),
    ),
)


ORIGINAL_TWO_SHOT_PATTERN = (
    "Question: What is the parity of 42? Answer: even\n"
    "Question: What is the parity of 17? Answer: odd\n"
    "Question: What is the parity of {number}? Answer:"
)


ORIGINAL_TWO_SHOT_TEMPLATES = tuple(
    PromptTemplate(
        template_id=f"parity_{split}_original_two_shot",
        split=split,
        task="parity",
        pattern=ORIGINAL_TWO_SHOT_PATTERN,
        candidate_texts=(" even", " odd"),
    )
    for split in ("train", "dev", "test")
)


CONTROL_TEMPLATES = (
    PromptTemplate(
        "control_identity",
        "test",
        "identity",
        "Repeat the integer exactly.\nInteger: 42\nCopy: 42\nInteger: {number}\nCopy:",
        (),
    ),
    PromptTemplate(
        "control_units_digit",
        "test",
        "units_digit",
        "Give the final decimal digit.\n42 -> 2\n107 -> 7\n{number} ->",
        tuple(str(value) for value in range(10)),
    ),
    PromptTemplate(
        "control_digit_count",
        "test",
        "digit_count",
        "Count the decimal digits.\n7 -> 1\n42 -> 2\n314 -> 3\n{number} ->",
        ("1", "2", "3"),
    ),
    PromptTemplate(
        "control_magnitude_range",
        "test",
        "magnitude_range",
        "Use low for 0-99, middle for 100-499, and high for 500-999.\n42 -> low\n314 -> middle\n771 -> high\n{number} ->",
        (" low", " middle", " high"),
    ),
)


def parity_index(value: int) -> int:
    """Return 0 for even and 1 for odd."""

    return int(value) % 2


def balanced_value_splits(seed: int = 20260716) -> dict[str, list[int]]:
    """Create parity-balanced 800/100/100 value-disjoint splits over 0..999."""

    rng = random.Random(int(seed))
    by_parity = {
        parity: [value for value in range(1000) if parity_index(value) == parity]
        for parity in (0, 1)
    }
    for values in by_parity.values():
        rng.shuffle(values)
    sizes = {"train": 400, "dev": 50, "test": 50}
    offsets = {0: 0, 1: 0}
    result = {split: [] for split in sizes}
    for split, size in sizes.items():
        for parity in (0, 1):
            start = offsets[parity]
            stop = start + size
            result[split].extend(by_parity[parity][start:stop])
            offsets[parity] = stop
        random.Random(f"{seed}:{split}").shuffle(result[split])
    validate_value_splits(result)
    return result


def validate_value_splits(splits: dict[str, Sequence[int]]) -> None:
    expected_sizes = {"train": 800, "dev": 100, "test": 100}
    if set(splits) != set(expected_sizes):
        raise ValueError(f"expected splits {sorted(expected_sizes)}, got {sorted(splits)}")
    observed: set[int] = set()
    for split, expected_size in expected_sizes.items():
        values = list(splits[split])
        if len(values) != expected_size or len(set(values)) != expected_size:
            raise ValueError(f"{split} must contain {expected_size} unique values")
        if set(values) & observed:
            raise ValueError(f"{split} overlaps an earlier split")
        observed.update(values)
        counts = [sum(parity_index(value) == parity for value in values) for parity in (0, 1)]
        if counts[0] != counts[1]:
            raise ValueError(f"{split} is not parity-balanced: {counts}")
    if observed != set(range(1000)):
        raise ValueError("splits do not partition 0..999")


def templates_for_split(split: str) -> tuple[PromptTemplate, ...]:
    return tuple(template for template in PARITY_TEMPLATES if template.split == split)


def templates_for_suite(split: str, suite: str = "default") -> tuple[PromptTemplate, ...]:
    if suite == "default":
        return templates_for_split(split)
    if suite == "original_two_shot":
        return tuple(
            template for template in ORIGINAL_TWO_SHOT_TEMPLATES if template.split == split
        )
    raise ValueError(f"unknown parity prompt suite: {suite}")


def template_manifest(suite: str = "default") -> list[dict[str, Any]]:
    if suite == "default":
        parity_templates = PARITY_TEMPLATES
    elif suite == "original_two_shot":
        parity_templates = ORIGINAL_TWO_SHOT_TEMPLATES
    else:
        raise ValueError(f"unknown parity prompt suite: {suite}")
    return [
        {
            "template_id": template.template_id,
            "split": template.split,
            "task": template.task,
            "pattern": template.pattern,
            "candidate_texts": list(template.candidate_texts),
        }
        for template in (*parity_templates, *CONTROL_TEMPLATES)
    ]


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {token_ids}")
    return int(token_ids[0])


def _render(pattern: str, value: int) -> tuple[str, int, int]:
    if pattern.count("{number}") != 1:
        raise ValueError("prompt pattern must contain exactly one {number} placeholder")
    prefix, suffix = pattern.split("{number}")
    numeral = str(int(value))
    return prefix + numeral + suffix, len(prefix), len(prefix) + len(numeral)


def _number_token(tokenizer: Any, text: str, char_start: int, char_end: int) -> tuple[int, int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if max(start, char_start) < min(end, char_end)
    ]
    if len(indices) != 1:
        raise ValueError(f"number span {text[char_start:char_end]!r} maps to token indices {indices}")
    index = indices[0]
    return index, int(encoded["input_ids"][index])


def numeric_candidate_texts() -> tuple[str, ...]:
    return tuple(str(value) for value in range(1000))


def control_gold_index(task: str, value: int) -> int:
    if task == "identity":
        return int(value)
    if task == "units_digit":
        return int(value) % 10
    if task == "digit_count":
        return len(str(int(value))) - 1
    if task == "magnitude_range":
        return 0 if value < 100 else 1 if value < 500 else 2
    raise ValueError(f"unknown control task: {task}")


def build_examples(
    *,
    values: Sequence[int],
    templates: Sequence[PromptTemplate],
    tokenizer: Any,
) -> list[PromptExample]:
    examples: list[PromptExample] = []
    numeric_texts: tuple[str, ...] | None = None
    numeric_ids: tuple[int, ...] | None = None
    for template in templates:
        candidate_texts = template.candidate_texts
        if template.task == "identity":
            if numeric_texts is None:
                numeric_texts = numeric_candidate_texts()
                numeric_ids = tuple(_single_token_id(tokenizer, text) for text in numeric_texts)
            candidate_texts = numeric_texts
            candidate_token_ids = numeric_ids
        else:
            candidate_token_ids = tuple(_single_token_id(tokenizer, text) for text in candidate_texts)
        for value in values:
            text, char_start, char_end = _render(template.pattern, value)
            token_index, token_id = _number_token(tokenizer, text, char_start, char_end)
            if template.task in {"parity", "divisible_by_two"}:
                gold = parity_index(value)
            else:
                gold = control_gold_index(template.task, value)
            examples.append(
                PromptExample(
                    value=int(value),
                    parity_index=parity_index(value),
                    split=template.split,
                    template_id=template.template_id,
                    task=template.task,
                    text=text,
                    number_token_index=token_index,
                    number_token_id=token_id,
                    candidate_token_ids=tuple(candidate_token_ids),
                    candidate_texts=tuple(candidate_texts),
                    gold_index=gold,
                )
            )
    return examples


def probe_direction_raw(
    *,
    weight: torch.Tensor,
    feature_scale: torch.Tensor,
    positive_class: int,
    negative_class: int,
) -> torch.Tensor:
    """Convert a standardized linear-probe contrast to raw activation coordinates."""

    if weight.ndim != 2:
        raise ValueError("probe weight must have shape [classes, hidden]")
    if feature_scale.ndim != 1 or feature_scale.shape[0] != weight.shape[1]:
        raise ValueError("feature scale does not match probe weight")
    return (weight[int(positive_class)] - weight[int(negative_class)]).float() / feature_scale.float()


def normalized_direction(raw: torch.Tensor, train_activations: torch.Tensor) -> SteeringDirection:
    norm = raw.float().norm()
    if not torch.isfinite(norm) or float(norm.item()) <= 0:
        raise ValueError("direction has zero or non-finite norm")
    unit = raw.float() / norm
    projection_std = float((train_activations.float() @ unit).std(unbiased=False).item())
    if not projection_std > 0:
        raise ValueError("direction projection has zero variance")
    return SteeringDirection(unit=unit.cpu(), projection_std=projection_std)


def orthogonal_random_direction(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    reference = reference.float() / reference.float().norm()
    random_vector = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    random_vector -= torch.dot(random_vector, reference) * reference
    return random_vector / random_vector.norm()


def value_split_hash(splits: dict[str, Iterable[int]]) -> str:
    payload = "|".join(
        f"{split}:{','.join(map(str, splits[split]))}" for split in ("train", "dev", "test")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def select_dev_condition(rows: Sequence[dict[str, Any]], identity_penalty: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty development sweep")
    scored = []
    for row in rows:
        selection_score = float(row["conditional_flip_rate"]) - float(identity_penalty) * (
            1.0 - float(row["identity_preservation_rate"])
        )
        scored.append({**row, "selection_score": selection_score})
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
