"""Target selection and membership metrics for list-based property steering."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Sequence

from introlm.experiments.number_property_list_filter import (
    ListExample,
    PROPERTY_SPECS,
    parse_number_sequence,
)


def select_target_indices(
    example: ListExample,
    *,
    source_class: int,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    """Select a reproducible random subset from one property class."""
    if source_class not in (0, 1):
        raise ValueError("source_class must be 0 or 1")
    if count <= 0:
        raise ValueError("count must be positive")
    spec = PROPERTY_SPECS[example.property_id]
    candidates = [
        index
        for index, value in enumerate(example.values)
        if int(bool(spec.label(value))) == source_class
    ]
    if len(candidates) < count:
        raise ValueError(
            f"{example.example_id} has only {len(candidates)} class-{source_class} values"
        )
    digest = hashlib.sha256(
        f"{seed}:{example.example_id}:{source_class}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return tuple(sorted(rng.sample(candidates, count)))


def counterfactual_targets(
    example: ListExample, target_indices: Sequence[int], source_class: int
) -> tuple[int, ...]:
    target_index_set = set(int(index) for index in target_indices)
    if len(target_index_set) != len(target_indices):
        raise ValueError("target indices must be unique")
    if not target_index_set.issubset(range(len(example.values))):
        raise ValueError("target index out of range")
    spec = PROPERTY_SPECS[example.property_id]
    for index in target_index_set:
        observed = int(bool(spec.label(example.values[index])))
        if observed != source_class:
            raise ValueError(
                f"target index {index} has class {observed}, expected {source_class}"
            )
    desired_class = 1 - int(source_class)
    output = []
    for index, value in enumerate(example.values):
        membership = int(bool(spec.label(value)))
        if index in target_index_set:
            membership = desired_class
        if membership == 1:
            output.append(int(value))
    return tuple(output)


def annotate_membership_completion(
    *,
    example: ListExample,
    text: str,
    target_indices: Sequence[int],
    source_class: int,
) -> dict[str, Any]:
    predicted = parse_number_sequence(text)
    predicted_set = set(predicted)
    source_set = set(example.values)
    factual_set = set(example.targets)
    target_index_set = set(int(index) for index in target_indices)
    desired_class = 1 - int(source_class)
    counterfactual = counterfactual_targets(example, target_indices, source_class)
    counterfactual_set = set(counterfactual)

    target_successes = sum(
        int((example.values[index] in predicted_set) == bool(desired_class))
        for index in target_index_set
    )
    untargeted_indices = [
        index for index in range(len(example.values)) if index not in target_index_set
    ]
    spec = PROPERTY_SPECS[example.property_id]
    untargeted_correct = sum(
        int(
            (example.values[index] in predicted_set)
            == bool(spec.label(example.values[index]))
        )
        for index in untargeted_indices
    )
    changed_indices = [
        index
        for index, value in enumerate(example.values)
        if (value in predicted_set) != (value in factual_set)
    ]
    desired_changes = sum(index in target_index_set for index in changed_indices)
    change_precision = (
        desired_changes / len(changed_indices) if changed_indices else 0.0
    )
    return {
        "parsed_numbers": list(predicted),
        "factual_numbers": list(example.targets),
        "counterfactual_numbers": list(counterfactual),
        "target_indices": sorted(target_index_set),
        "target_values": [example.values[index] for index in sorted(target_index_set)],
        "source_class": int(source_class),
        "desired_class": desired_class,
        "exact_factual_sequence": float(predicted == example.targets),
        "exact_counterfactual_sequence": float(predicted == counterfactual),
        "exact_counterfactual_set": float(
            predicted_set == counterfactual_set
            and len(predicted) == len(counterfactual)
        ),
        "target_success_count": target_successes,
        "target_count": len(target_index_set),
        "target_transport_rate": target_successes / len(target_index_set),
        "complete_target_transport": float(target_successes == len(target_index_set)),
        "untargeted_correct_count": untargeted_correct,
        "untargeted_count": len(untargeted_indices),
        "untargeted_membership_accuracy": untargeted_correct / len(untargeted_indices),
        "complete_untargeted_preservation": float(
            untargeted_correct == len(untargeted_indices)
        ),
        "changed_membership_count": len(changed_indices),
        "desired_change_count": desired_changes,
        "change_precision": change_precision,
        "source_fidelity": float(predicted_set.issubset(source_set)),
        "out_of_input_count": len(predicted_set - source_set),
        "duplicate_count": len(predicted) - len(predicted_set),
        "output_count": len(predicted),
    }


def summarize_membership(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty membership rows")
    target_total = sum(int(row["target_count"]) for row in rows)
    untargeted_total = sum(int(row["untargeted_count"]) for row in rows)
    changed_total = sum(int(row["changed_membership_count"]) for row in rows)
    return {
        "n_examples": len(rows),
        "exact_counterfactual_sequence": sum(
            float(row["exact_counterfactual_sequence"]) for row in rows
        )
        / len(rows),
        "exact_counterfactual_set": sum(
            float(row["exact_counterfactual_set"]) for row in rows
        )
        / len(rows),
        "target_transport_rate": sum(int(row["target_success_count"]) for row in rows)
        / target_total,
        "complete_target_transport": sum(
            float(row["complete_target_transport"]) for row in rows
        )
        / len(rows),
        "untargeted_membership_accuracy": sum(
            int(row["untargeted_correct_count"]) for row in rows
        )
        / untargeted_total,
        "complete_untargeted_preservation": sum(
            float(row["complete_untargeted_preservation"]) for row in rows
        )
        / len(rows),
        "change_precision": (
            sum(int(row["desired_change_count"]) for row in rows) / changed_total
            if changed_total
            else 0.0
        ),
        "source_fidelity": sum(float(row["source_fidelity"]) for row in rows)
        / len(rows),
        "mean_changed_memberships": changed_total / len(rows),
        "mean_out_of_input": sum(int(row["out_of_input_count"]) for row in rows)
        / len(rows),
        "mean_duplicates": sum(int(row["duplicate_count"]) for row in rows) / len(rows),
        "mean_output_count": sum(int(row["output_count"]) for row in rows) / len(rows),
    }


def select_strength(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_untargeted: float,
    minimum_source_fidelity: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select strength from empty rows")
    eligible = [
        row
        for row in rows
        if float(row["untargeted_membership_accuracy"]) >= minimum_untargeted
        and float(row["source_fidelity"]) >= minimum_source_fidelity
    ]
    candidates = eligible or list(rows)
    selected = max(
        candidates,
        key=lambda row: (
            float(row["exact_counterfactual_set"]),
            float(row["target_transport_rate"]),
            float(row["complete_untargeted_preservation"]),
            float(row["source_fidelity"]),
            -float(row["alpha"]),
        ),
    )
    return {
        **selected,
        "eligible_strength_count": len(eligible),
        "selection_mode": (
            "preservation_gate_then_counterfactual_exactness"
            if eligible
            else "preservation_gate_failed_fallback"
        ),
        "minimum_untargeted": minimum_untargeted,
        "minimum_source_fidelity": minimum_source_fidelity,
    }
