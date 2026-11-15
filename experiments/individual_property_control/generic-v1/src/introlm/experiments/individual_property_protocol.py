"""Pure task definitions and validation for individual number-property control."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Sequence


PROTOCOL_VERSION = "individual-property-control-v1"


@dataclass(frozen=True)
class Property:
    property_id: str
    field: str
    labels: tuple[str, ...]
    rule: str

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def label(self, value: int) -> int:
        n = int(value)
        if not 0 <= n <= 999:
            raise ValueError("The registered numeral domain is 0..999")
        p = self.property_id
        if p == "parity":
            return n % 2
        if p == "primality":
            return int(n >= 2 and all(n % k for k in range(2, math.isqrt(n) + 1)))
        if p.startswith("divisible_by_"):
            return int(n % int(p.removeprefix("divisible_by_")) == 0)
        if p.startswith("residue_mod_"):
            return n % int(p.removeprefix("residue_mod_"))
        if p == "units_digit":
            return n % 10
        if p == "tens_digit":
            return (n // 10) % 10
        if p == "hundreds_digit":
            return n // 100
        if p == "digit_count":
            return len(str(n)) - 1
        if p == "distinct_digit_count":
            return len(set(str(n))) - 1
        if p == "contains_zero":
            return int("0" in str(n))
        if p == "has_repeated_digit":
            return int(len(set(str(n))) < len(str(n)))
        if p == "palindrome":
            return int(str(n) == str(n)[::-1])
        raise ValueError(p)

    def target(self, value: int) -> int:
        return (self.label(value) + 1) % self.num_classes


PROPERTIES = {
    "parity": Property(
        "parity",
        "Parity",
        ("even", "odd"),
        "Even numbers are divisible by 2; odd numbers are not.",
    ),
    "primality": Property(
        "primality",
        "Prime",
        ("no", "yes"),
        "A prime integer is greater than 1 and has no positive divisors except 1 and itself. Zero and one are not prime.",
    ),
    "digit_count": Property(
        "digit_count",
        "Digit count",
        ("1", "2", "3"),
        "Count the decimal digits, with no leading zeros. Zero has one digit.",
    ),
    "distinct_digit_count": Property(
        "distinct_digit_count",
        "Distinct digit count",
        ("1", "2", "3"),
        "Count the different decimal digits appearing in the numeral. For example, 121 has two distinct digits.",
    ),
    "units_digit": Property(
        "units_digit",
        "Units digit",
        tuple(map(str, range(10))),
        "Report the final decimal digit.",
    ),
    "tens_digit": Property(
        "tens_digit",
        "Tens digit",
        tuple(map(str, range(10))),
        "Report the second decimal digit from the right, or 0 if absent.",
    ),
    "hundreds_digit": Property(
        "hundreds_digit",
        "Hundreds digit",
        tuple(map(str, range(10))),
        "Report the third decimal digit from the right, or 0 if absent.",
    ),
    "contains_zero": Property(
        "contains_zero",
        "Contains zero",
        ("no", "yes"),
        "Answer yes exactly when the decimal numeral contains the digit 0.",
    ),
    "has_repeated_digit": Property(
        "has_repeated_digit",
        "Repeated digit",
        ("no", "yes"),
        "Answer yes exactly when a decimal digit appears more than once in the numeral.",
    ),
    "palindrome": Property(
        "palindrome",
        "Palindrome",
        ("no", "yes"),
        "Answer yes exactly when the decimal numeral is identical when reversed. Single-digit numerals are palindromes.",
    ),
}
for _divisor in (3, 4, 5, 6, 8, 10, 25):
    _key = f"divisible_by_{_divisor}"
    PROPERTIES[_key] = Property(
        _key,
        f"Divisible by {_divisor}",
        ("no", "yes"),
        f"Answer yes exactly when division by {_divisor} leaves remainder zero.",
    )
for _modulus in (3, 4):
    _key = f"residue_mod_{_modulus}"
    PROPERTIES[_key] = Property(
        _key,
        f"Residue modulo {_modulus}",
        tuple(map(str, range(_modulus))),
        f"Report the remainder after division by {_modulus}.",
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def value_splits(spec: Property, seed: int) -> dict[str, list[int]]:
    """Stratify all 1,000 values; preserve every class without resampling values."""
    result: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    for c in range(spec.num_classes):
        values = [n for n in range(1000) if spec.label(n) == c]
        random.Random(f"{PROTOCOL_VERSION}:{seed}:{spec.property_id}:{c}").shuffle(
            values
        )
        heldout = max(1, round(len(values) * 0.1))
        if len(values) - 2 * heldout < 1:
            raise ValueError("A class has insufficient independent values")
        result["train"].extend(values[: -2 * heldout])
        result["dev"].extend(values[-2 * heldout : -heldout])
        result["test"].extend(values[-heldout:])
    for name, values in result.items():
        random.Random(f"{seed}:{spec.property_id}:{name}").shuffle(values)
    validate_splits(spec, result)
    return result


def validate_splits(spec: Property, splits: dict[str, Sequence[int]]) -> None:
    sets = {k: set(map(int, v)) for k, v in splits.items()}
    if set(sets) != {"train", "dev", "test"}:
        raise ValueError("Require train, dev, and test partitions")
    for name, values in splits.items():
        if len(sets[name]) != len(values):
            raise ValueError("Duplicate values within a split")
        if {spec.label(n) for n in values} != set(range(spec.num_classes)):
            raise ValueError("Every split must contain every class")
    if any(
        sets[a] & sets[b]
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise ValueError("Numerals overlap across value splits")
    if set.union(*sets.values()) != set(range(1000)):
        raise ValueError("The registered numeral domain is not partitioned completely")


def sample_values(
    spec: Property, values: Sequence[int], maximum: int, seed: int
) -> list[int]:
    """Cap each class, keeping rare held-out values and macro scoring explicit."""
    limit = max(1, math.ceil(maximum / spec.num_classes))
    result = []
    for c in range(spec.num_classes):
        group = [int(n) for n in values if spec.label(n) == c]
        random.Random(f"{seed}:{spec.property_id}:{c}:sample").shuffle(group)
        result.extend(group[:limit])
    random.Random(seed).shuffle(result)
    return result


@dataclass(frozen=True)
class Case:
    case_id: str
    property_id: str
    value: int
    source_class: int
    target_class: int
    family: str
    content: str
    anchor: str
    anchor_offset: int
    expected_factual: tuple[int, ...] = ()
    expected_target: tuple[int, ...] = ()
    input_values: tuple[int, ...] = ()
    filter_class: int | None = None


def demonstrations(spec: Property, train: Sequence[int], seed: int) -> list[int]:
    per_class = 4 if spec.num_classes == 2 else 2
    chosen = []
    for c in range(spec.num_classes):
        group = [int(n) for n in train if spec.label(n) == c]
        random.Random(f"demo:{seed}:{spec.property_id}:{c}").shuffle(group)
        chosen.extend(group[:per_class])
    random.Random(seed).shuffle(chosen)
    return chosen


def joint_cases(
    spec: Property, values: Sequence[int], train: Sequence[int], seed: int, split: str
) -> list[Case]:
    examples = "\n\n".join(
        f"Input: {n}\nResult after adding zero: {n}\n{spec.field}: {spec.labels[spec.label(n)]}"
        for n in demonstrations(spec, train, seed)
    )
    instruction = f"Copy the input exactly as the result after adding zero, then report {spec.field.lower()}. Use exactly these two output fields. Allowed property labels: {', '.join(spec.labels)}. {spec.rule}"
    cases = []
    for n in values:
        anchor = f"Input: {n}\nResult after adding zero:"
        content = f"{instruction}\n\n{examples}\n\n{anchor}"
        cases.append(
            Case(
                f"{split}:joint:{n}",
                spec.property_id,
                n,
                spec.label(n),
                spec.target(n),
                "joint",
                content,
                anchor,
                len("Input: "),
            )
        )
    return cases


def list_cases(
    spec: Property, values: Sequence[int], pool: Sequence[int], seed: int, split: str
) -> list[Case]:
    """One edited numeral; lists use only their declared value partition."""
    if not set(values).issubset(set(pool)):
        raise ValueError("Target values must belong to the list's partition")
    cases = []
    for index, n in enumerate(values):
        rng = random.Random(f"list:{seed}:{spec.property_id}:{split}:{n}")
        source, target = spec.label(n), spec.target(n)
        # Across classes this covers both insertion and removal, including all
        # registered multiclass next-class transitions.
        filter_class = target if index % 2 == 0 else source
        positive = [int(k) for k in pool if k != n and spec.label(k) == filter_class]
        negative = [int(k) for k in pool if k != n and spec.label(k) != filter_class]
        per_class = max(1, min(5, len(positive), len(negative)))
        source_selected = source == filter_class
        pos_n = per_class - int(source_selected)
        neg_n = per_class - int(not source_selected)
        others = rng.sample(positive, pos_n) + rng.sample(negative, neg_n)
        position = rng.randrange(len(others) + 1)
        numbers = others[:position] + [n] + others[position:]
        numbered = "\n".join(f"{j + 1}. {k}" for j, k in enumerate(numbers))
        anchor = f"{position + 1}. {n}\n"
        instruction = f"Select only the input numbers whose {spec.field.lower()} is {spec.labels[filter_class]}. {spec.rule} Return only the selected numerals as a comma-separated list, in input order. Do not output item indices, explanations, or property labels."
        content = f"{instruction}\n\nNumbers:\n{numbered}\nAnswer:"
        factual = tuple(k for k in numbers if spec.label(k) == filter_class)
        expected = tuple(
            k
            for j, k in enumerate(numbers)
            if (
                target == filter_class
                if j == position
                else spec.label(k) == filter_class
            )
        )
        cases.append(
            Case(
                f"{split}:list:{n}",
                spec.property_id,
                n,
                source,
                target,
                "list",
                content,
                anchor,
                len(f"{position + 1}. "),
                factual,
                expected,
                tuple(numbers),
                filter_class,
            )
        )
    return cases


def fit_case(spec: Property, n: int, context: str) -> Case:
    if context == "property":
        anchor = f"Input: {n}\n"
        content = f"Report {spec.field.lower()}. {spec.rule}\n{anchor}{spec.field}:"
        offset = len("Input: ")
    elif context == "addition":
        anchor = f"92+{n}="
        content, offset = anchor, len("92+")
    else:
        raise ValueError(context)
    return Case(
        f"fit:{context}:{n}",
        spec.property_id,
        n,
        spec.label(n),
        spec.target(n),
        "fit",
        content,
        anchor,
        offset,
    )


def score(case: Case, text: str) -> dict[str, Any]:
    spec = PROPERTIES[case.property_id]
    if case.family == "joint":
        if case.content.endswith("Result after adding zero:"):
            text = "Result after adding zero:" + text
        result = re.search(r"Result after adding zero\s*:\s*(-?\d+)(?!\d)", text, re.I)
        alternatives = "|".join(
            re.escape(label) for label in sorted(spec.labels, key=len, reverse=True)
        )
        attribute = re.search(
            rf"{re.escape(spec.field)}\s*:\s*({alternatives})(?![a-zA-Z0-9])",
            text,
            re.I,
        )
        identity = result is not None and int(result.group(1)) == case.value
        observed = (
            None
            if attribute is None
            else next(
                (
                    i
                    for i, label in enumerate(spec.labels)
                    if label.lower() == attribute.group(1).lower()
                ),
                None,
            )
        )
        return {
            "parsed": result is not None and attribute is not None,
            "identity_preserved": identity,
            "factual": identity and observed == case.source_class,
            "target_success": identity and observed == case.target_class,
            "parsed_property": observed,
            "parsed_numeral": None if result is None else int(result.group(1)),
        }
    if case.family == "list":
        numbers = tuple(int(n) for n in re.findall(r"(?<!\d)-?\d+(?!\d)", text))
        # A numeral-only output can be bracketed; extra prose is a format error.
        remainder = re.sub(r"(?<!\d)-?\d+(?!\d)", "", text)
        parsed = not re.sub(r"[\s,;\[\]().]", "", remainder)
        identity = set(numbers).issubset(set(case.input_values)) and len(
            numbers
        ) == len(set(numbers))
        return {
            "parsed": parsed,
            "identity_preserved": identity,
            "factual": parsed and numbers == case.expected_factual,
            "target_success": parsed and numbers == case.expected_target,
            "parsed_numbers": list(numbers),
        }
    raise ValueError(case.family)


def summary(
    rows: Sequence[dict[str, Any]], baseline: Sequence[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Empty evaluation")
    baseline_by_id = {} if baseline is None else {r["case_id"]: r for r in baseline}
    if baseline is not None and set(baseline_by_id) != {r["case_id"] for r in rows}:
        raise ValueError("Paired cohorts differ")
    classes = sorted({r["source_class"] for r in rows})
    result: dict[str, Any] = {
        "n": len(rows),
        "source_class_counts": dict(Counter(r["source_class"] for r in rows)),
    }
    for field in ("parsed", "identity_preserved", "factual", "target_success"):
        result[field] = sum(bool(r[field]) for r in rows) / len(rows)
        result[f"macro_{field}"] = sum(
            sum(bool(r[field]) for r in rows if r["source_class"] == c)
            / sum(r["source_class"] == c for r in rows)
            for c in classes
        ) / len(classes)
    eligible = (
        rows
        if baseline is None
        else [r for r in rows if baseline_by_id[r["case_id"]]["factual"]]
    )
    result["baseline_correct_n"] = len(eligible)
    result["conditional_target_success"] = (
        None
        if not eligible
        else sum(bool(r["target_success"]) for r in eligible) / len(eligible)
    )
    result["by_source_class"] = {}
    for c in classes:
        group = [r for r in rows if r["source_class"] == c]
        correct = [r for r in eligible if r["source_class"] == c]
        result["by_source_class"][str(c)] = {
            "n": len(group),
            "baseline_correct_n": len(correct),
            "target_success_count": sum(bool(r["target_success"]) for r in group),
            "target_success": sum(bool(r["target_success"]) for r in group)
            / len(group),
            "conditional_target_success": None
            if not correct
            else sum(bool(r["target_success"]) for r in correct) / len(correct),
            "identity_preserved": sum(bool(r["identity_preserved"]) for r in group)
            / len(group),
        }
    result["macro_conditional_target_success"] = sum(
        (result["by_source_class"][str(c)]["conditional_target_success"] or 0)
        for c in classes
    ) / len(classes)
    result["nominal_norm_mean"] = sum(
        float(r.get("nominal_norm", 0)) for r in rows
    ) / len(rows)
    result["applied_norm_mean"] = sum(
        float(r.get("applied_norm", 0)) for r in rows
    ) / len(rows)
    return result


def protocol_identity(config: dict[str, Any], task: dict[str, Any]) -> str:
    semantic = {
        k: v
        for k, v in config.items()
        if k not in {"output_root", "source_root", "log_root"}
    }
    return digest(
        {
            "version": PROTOCOL_VERSION,
            "config": semantic,
            "task": task,
            "properties": [asdict(p) for p in PROPERTIES.values()],
        }
    )
