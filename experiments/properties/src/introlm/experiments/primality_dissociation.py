"""Data, prompts, and metrics for value-disjoint primality steering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import random
import re
from typing import Any, Iterable, Sequence


INTEGER_RE = re.compile(r"(?<!\d)-?\d+(?!\d)")
PRIME_STATUS_RE = re.compile(r"\b(composite|prime(?!\s+status))\b", re.IGNORECASE)


def instantiate_compatible_config(config_type: type, values: dict[str, Any]) -> Any:
    """Construct a shared helper config while tolerating absent optional fields."""

    accepted_fields = set(inspect.signature(config_type).parameters)
    return config_type(**{key: value for key, value in values.items() if key in accepted_fields})


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


def prime_status(value: int) -> str | None:
    if int(value) < 2:
        return None
    return "prime" if is_prime(value) else "composite"


def balanced_prime_value_splits(seed: int = 20260717) -> dict[str, list[int]]:
    """Use all 168 primes and 168 sampled composites in stratified disjoint splits."""

    primes = [value for value in range(2, 1000) if is_prime(value)]
    composites = [value for value in range(2, 1000) if not is_prime(value)]
    if len(primes) != 168:
        raise AssertionError(f"expected 168 primes below 1000, got {len(primes)}")
    rng = random.Random(int(seed))
    sampled_composites = rng.sample(composites, len(primes))
    rng.shuffle(primes)
    rng.shuffle(sampled_composites)
    class_sizes = {"train": 134, "dev": 17, "test": 17}
    result = {split: [] for split in class_sizes}
    offset = 0
    for split, size in class_sizes.items():
        result[split] = primes[offset : offset + size] + sampled_composites[offset : offset + size]
        random.Random(f"{seed}:{split}").shuffle(result[split])
        offset += size
    validate_prime_value_splits(result)
    return result


def validate_prime_value_splits(splits: dict[str, Sequence[int]]) -> None:
    expected = {"train": 268, "dev": 34, "test": 34}
    if set(splits) != set(expected):
        raise ValueError(f"expected splits {sorted(expected)}, got {sorted(splits)}")
    seen: set[int] = set()
    for split, size in expected.items():
        values = list(map(int, splits[split]))
        if len(values) != size or len(set(values)) != size:
            raise ValueError(f"{split} must contain {size} unique values")
        if any(value < 2 or value >= 1000 for value in values):
            raise ValueError("values must be in 2..999")
        if seen.intersection(values):
            raise ValueError(f"{split} overlaps another split")
        seen.update(values)
        prime_count = sum(is_prime(value) for value in values)
        if prime_count * 2 != len(values):
            raise ValueError(f"{split} is not class-balanced")


def value_split_hash(splits: dict[str, Iterable[int]]) -> str:
    payload = "|".join(
        f"{split}:{','.join(map(str, splits[split]))}" for split in ("train", "dev", "test")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProbeTemplate:
    template_id: str
    split: str
    pattern: str
    candidate_texts: tuple[str, str] = (" prime", " composite")


@dataclass(frozen=True)
class ProbeExample:
    value: int
    parity_index: int  # Compatibility name for the generic binary steering helpers.
    split: str
    template_id: str
    task: str
    text: str
    number_token_index: int
    number_token_id: int
    candidate_token_ids: tuple[int, ...]
    candidate_texts: tuple[str, ...]
    gold_index: int


PROBE_PATTERN = (
    "Classify each integer as prime or composite.\n"
    "67: prime\n68: composite\n97: prime\n99: composite\n{number}:"
)
PROBE_TEMPLATES = tuple(
    ProbeTemplate(f"prime_status_{split}_four_shot", split, PROBE_PATTERN)
    for split in ("train", "dev", "test")
)


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {token_ids}")
    return int(token_ids[0])


def _number_token(tokenizer: Any, text: str, start: int, end: int) -> tuple[int, int]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    indices = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if max(int(left), start) < min(int(right), end)
    ]
    if len(indices) != 1:
        raise ValueError(f"number span {text[start:end]!r} maps to {indices}")
    index = indices[0]
    return index, int(encoded["input_ids"][index])


def build_probe_examples(
    *, values: Sequence[int], split: str, tokenizer: Any
) -> list[ProbeExample]:
    template = next(template for template in PROBE_TEMPLATES if template.split == split)
    candidate_ids = tuple(_single_token_id(tokenizer, text) for text in template.candidate_texts)
    examples = []
    for value in values:
        prefix, suffix = template.pattern.split("{number}")
        numeral = str(int(value))
        text = prefix + numeral + suffix
        token_index, token_id = _number_token(
            tokenizer, text, len(prefix), len(prefix) + len(numeral)
        )
        label = 0 if is_prime(value) else 1
        examples.append(
            ProbeExample(
                value=int(value),
                parity_index=label,
                split=split,
                template_id=template.template_id,
                task="prime_status",
                text=text,
                number_token_index=token_index,
                number_token_id=token_id,
                candidate_token_ids=candidate_ids,
                candidate_texts=template.candidate_texts,
                gold_index=label,
            )
        )
    return examples


@dataclass(frozen=True)
class JointPromptVariant:
    variant_id: str
    description: str
    pattern: str
    demonstration_values: tuple[int, ...]
    includes_factor_field: bool = False

    def render(self, value: int) -> str:
        return self.pattern.format(number=int(value))


def _balanced_demos(train_values: Sequence[int], count_per_class: int) -> tuple[int, ...]:
    primes = sorted(value for value in train_values if is_prime(value))[:count_per_class]
    composites = sorted(value for value in train_values if not is_prime(value))[:count_per_class]
    values = [item for pair in zip(primes, composites, strict=True) for item in pair]
    return tuple(values)


def _structured(values: Sequence[int]) -> str:
    return "\n\n".join(
        f"Input: {value}\nResult after adding zero: {value}\nPrime status of result: {prime_status(value)}"
        for value in values
    )


def _semicolon(values: Sequence[int]) -> str:
    return "\n".join(
        f"{value} + 0 = {value}; {value} is {prime_status(value)}" for value in values
    )


def _factor(value: int) -> int:
    for divisor in range(2, int(value)):
        if value % divisor == 0:
            return divisor
    raise ValueError(f"{value} has no nontrivial factor")


def _factor_examples(values: Sequence[int]) -> str:
    rows = []
    for value in values:
        factor = "none" if is_prime(value) else str(_factor(value))
        rows.append(
            f"Input: {value}\nResult after adding zero: {value}\n"
            f"Prime status of result: {prime_status(value)}\nNontrivial factor: {factor}"
        )
    return "\n\n".join(rows)


def build_joint_prompt_variants(train_values: Sequence[int]) -> tuple[JointPromptVariant, ...]:
    variants = []
    instruction = (
        "Add zero to the input, copy the unchanged result exactly, and classify the result "
        "as prime or composite."
    )
    for shots in (4, 8, 16):
        demos = _balanced_demos(train_values, shots // 2)
        variants.append(
            JointPromptVariant(
                f"structured_{shots}shot",
                "Structured identity and prime-status fields.",
                f"{_structured(demos)}\n\nInput: {{number}}\nResult after adding zero:",
                demos,
            )
        )
        variants.append(
            JointPromptVariant(
                f"semicolon_{shots}shot",
                "Compact equation and prime-status clauses.",
                f"{_semicolon(demos)}\n{{number}} + 0 =",
                demos,
            )
        )
    demos = _balanced_demos(train_values, 4)
    variants.append(
        JointPromptVariant(
            "instructed_structured_8shot",
            "Explicit instruction with structured demonstrations.",
            f"{instruction}\n\n{_structured(demos)}\n\nInput: {{number}}\nResult after adding zero:",
            demos,
        )
    )
    variants.append(
        JointPromptVariant(
            "structured_factor_8shot",
            "Optional factor field; primary scoring still uses identity and status only.",
            f"{_factor_examples(demos)}\n\nInput: {{number}}\nResult after adding zero:",
            demos,
            includes_factor_field=True,
        )
    )
    return tuple(variants)


def parse_joint_completion(text: str) -> dict[str, Any]:
    integer_match = INTEGER_RE.search(text)
    status_matches = PRIME_STATUS_RE.findall(text)
    return {
        "parsed_result": int(integer_match.group(0)) if integer_match else None,
        "parsed_prime_status": status_matches[-1].lower() if status_matches else None,
    }


def annotate_completion(*, value: int, text: str) -> dict[str, Any]:
    parsed = parse_joint_completion(text)
    result = parsed["parsed_result"]
    status = parsed["parsed_prime_status"]
    source_status = prime_status(value)
    if source_status is None:
        raise ValueError("source values 0 and 1 are excluded")
    target_status = "composite" if source_status == "prime" else "prime"
    result_status = prime_status(result) if result is not None else None
    joint_parsed = result is not None and status is not None
    identity_preserved = result == int(value)
    target_hit = status == target_status
    source_hit = status == source_status
    coherent_changed = (
        joint_parsed
        and not identity_preserved
        and result_status == target_status
        and status == target_status
    )
    contradiction = identity_preserved and target_hit
    return {
        **parsed,
        "source_prime_status": source_status,
        "target_prime_status": target_status,
        "transition": f"{source_status}_to_{target_status}",
        "result_present": result is not None,
        "prime_status_present": status is not None,
        "joint_parsed": joint_parsed,
        "identity_preserved": identity_preserved,
        "status_is_original": source_hit,
        "status_is_target": target_hit,
        "baseline_joint_correct": identity_preserved and source_hit,
        "internally_consistent": joint_parsed and result_status == status,
        "internally_inconsistent": joint_parsed and result_status != status,
        "dissociated_identity_target_status": contradiction,
        "coherent_changed_target": coherent_changed,
        "outcome": (
            "retained_identity_target_contradiction"
            if contradiction
            else "coherent_changed_identity"
            if coherent_changed
            else "unchanged_natural"
            if identity_preserved and source_hit
            else "other_parsed"
            if joint_parsed
            else "parse_failure"
        ),
    }


def summarize_completions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    n = len(rows)
    count = lambda key: sum(bool(row[key]) for row in rows)  # noqa: E731
    outcomes = {
        name: sum(row["outcome"] == name for row in rows)
        for name in (
            "retained_identity_target_contradiction",
            "coherent_changed_identity",
            "unchanged_natural",
            "other_parsed",
            "parse_failure",
        )
    }
    return {
        "n": n,
        "joint_coverage": count("joint_parsed") / n,
        "joint_correct_rate": count("baseline_joint_correct") / n,
        "identity_preservation_rate": count("identity_preserved") / n,
        "target_status_rate": count("status_is_target") / n,
        "dissociation_rate": count("dissociated_identity_target_status") / n,
        "coherent_changed_target_rate": count("coherent_changed_target") / n,
        **{f"{key}_count": value for key, value in outcomes.items()},
    }


def summarize_paired(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired rows must be nonempty and aligned")
    eligible = [index for index, row in enumerate(baseline) if row["baseline_joint_correct"]]
    hits = sum(candidate[index]["dissociated_identity_target_status"] for index in eligible)
    return {
        "baseline_joint_correct_count": len(eligible),
        "conditional_dissociation_count": hits,
        "conditional_dissociation_rate": hits / len(eligible) if eligible else 0.0,
    }


def select_joint_prompt(
    rows: Sequence[dict[str, Any]], *, minimum_accuracy: float = 0.90,
    minimum_coverage: float = 0.95,
) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["baseline_joint_correct_rate"] >= minimum_accuracy
        and row["baseline_joint_coverage"] >= minimum_coverage
    ]
    if not eligible:
        raise RuntimeError(
            f"no prompt passed natural behavior gates accuracy>={minimum_accuracy} "
            f"and coverage>={minimum_coverage}"
        )
    selected = max(
        eligible,
        key=lambda row: (
            row["conditional_dissociation_rate"],
            row["baseline_joint_correct_rate"],
            -row["n_demonstrations"],
            row["variant_id"],
        ),
    )
    return {
        **selected,
        "eligible_prompt_count": len(eligible),
        "minimum_baseline_joint_accuracy": minimum_accuracy,
        "minimum_joint_coverage": minimum_coverage,
    }
