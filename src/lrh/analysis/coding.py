from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

PURPOSES = {
    "define_position": {
        "label": "Define / position",
        "functions": {
            "definition_or_formalization",
            "background_or_attribution",
            "problem_setting",
            "literature_positioning",
        },
    },
    "motivate_apply": {
        "label": "Motivate / apply",
        "functions": {
            "theoretical_premise",
            "method_motivation",
            "application_or_interpretation",
        },
    },
    "test_support": {
        "label": "Test / support",
        "functions": {
            "theoretical_explanation",
            "empirical_test",
            "evidential_claim",
        },
    },
    "qualify_challenge": {
        "label": "Qualify / challenge",
        "functions": {
            "qualification_or_refinement",
            "extension_or_repair",
            "challenge_or_counterevidence",
        },
    },
}


def cohen_kappa(left: list[Any], right: list[Any]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right) for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def audit_pairs(
    assignments: list[dict[str, str]],
    lookups: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = []
    right = []
    for assignment in assignments:
        audit_worker = assignment["audit_worker"]
        if not audit_worker:
            continue
        episode_id = assignment["episode_id"]
        primary_worker = assignment["primary_worker"]
        left.append(lookups[primary_worker][episode_id])
        right.append(lookups[audit_worker][episode_id])
    return left, right


def purpose_families(row: dict[str, Any]) -> set[str]:
    functions = set(row["rhetorical_functions"])
    return {
        purpose_id
        for purpose_id, specification in PURPOSES.items()
        if functions & specification["functions"]
    }


def purpose_agreement(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    substantive_pairs = [
        (left, right)
        for left, right in zip(left_rows, right_rows)
        if left["episode_status"] == right["episode_status"] == "substantive"
    ]
    fine_jaccards = []
    purpose_jaccards = []
    fine_exact = 0
    purpose_exact = 0
    left_purposes = []
    right_purposes = []
    for left, right in substantive_pairs:
        left_functions = set(left["rhetorical_functions"])
        right_functions = set(right["rhetorical_functions"])
        function_union = left_functions | right_functions
        fine_jaccards.append(len(left_functions & right_functions) / len(function_union))
        fine_exact += left_functions == right_functions

        left_family = purpose_families(left)
        right_family = purpose_families(right)
        family_union = left_family | right_family
        purpose_jaccards.append(len(left_family & right_family) / len(family_union))
        purpose_exact += left_family == right_family
        left_purposes.append(left_family)
        right_purposes.append(right_family)

    count = len(substantive_pairs)
    return {
        "substantive_audit_episode_count": count,
        "fine_grained_rhetorical_function_mean_jaccard": (sum(fine_jaccards) / count),
        "fine_grained_rhetorical_function_exact_set_agreement": (fine_exact / count),
        "purpose_family_mean_jaccard": sum(purpose_jaccards) / count,
        "purpose_family_exact_set_agreement": purpose_exact / count,
        "purpose_family_binary_kappa": {
            purpose_id: cohen_kappa(
                [purpose_id in values for values in left_purposes],
                [purpose_id in values for values in right_purposes],
            )
            for purpose_id in PURPOSES
        },
    }


def paper_presence(annotations, aliases):
    """Union substantive episode purposes across verified versions of a study."""
    papers = defaultdict(set)
    for row in annotations:
        if row["episode_status"] != "substantive":
            continue
        document = row["episode_id"].rsplit("-E", 1)[0]
        families = purpose_families(row)
        if not families:
            raise ValueError(f"Substantive episode has no family: {row['episode_id']}")
        papers[aliases.get(document, document)].update(families)
    return dict(papers)


def sensitivity(primary, secondary, aliases):
    before = paper_presence(primary, aliases)
    replacement = {row["episode_id"]: row for row in secondary}
    after = paper_presence([replacement.get(row["episode_id"], row) for row in primary], aliases)
    if set(before) != set(after):
        raise ValueError("Replacement changes the substantive-paper denominator")
    n = len(before)
    return {
        family: {
            "primary_count": sum(family in values for values in before.values()),
            "replacement_count": sum(family in values for values in after.values()),
            "change_percentage_points": 100
            * (
                sum(family in values for values in after.values())
                - sum(family in values for values in before.values())
            )
            / n,
        }
        for family in PURPOSES
    }
