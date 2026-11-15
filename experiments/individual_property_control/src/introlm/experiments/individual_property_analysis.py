"""Independent mathematical replay and paired analysis of property interventions."""

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np


def gold_label(prop, n):
    if prop == "parity":
        return n % 2
    if prop == "primality":
        return int(n > 1 and not any(n % d == 0 for d in range(2, math.isqrt(n) + 1)))
    if prop.startswith("divisible_by_"):
        return int(n % int(prop.rsplit("_", 1)[1]) == 0)
    if prop.startswith("residue_mod_"):
        return n % int(prop.rsplit("_", 1)[1])
    if prop in {"units_digit", "tens_digit", "hundreds_digit"}:
        return (
            n // {"units_digit": 1, "tens_digit": 10, "hundreds_digit": 100}[prop]
        ) % 10
    if prop == "digit_count":
        return len(str(n)) - 1
    if prop == "distinct_digit_count":
        return len(set(str(n))) - 1
    if prop == "contains_zero":
        return int("0" in str(n))
    if prop == "has_repeated_digit":
        return int(len(str(n)) != len(set(str(n))))
    if prop == "palindrome":
        return int(str(n) == str(n)[::-1])
    raise ValueError(prop)


def independent_score(row, property_spec):
    """Reconstruct the targets from arithmetic, not the stored target fields."""
    case = row["case"]
    prop, value = case["property_id"], case["value"]
    labels = property_spec["labels"]
    source = gold_label(prop, value)
    target = (source + 1) % len(labels)
    if (source, target) != (case["source_class"], case["target_class"]):
        raise ValueError("Stored property labels disagree with independent arithmetic")
    start = case["content"].rfind(case["anchor"]) + case["anchor_offset"]
    if case["content"][start : start + len(str(value))] != str(value):
        raise ValueError("The declared input span does not contain the source numeral")
    text = row["generated_text"]
    if case["family"] == "list":
        inputs = case["input_values"]
        if len(inputs) != len(set(inputs)) or inputs.count(value) != 1:
            raise ValueError("List input does not identify one unique edited numeral")
        selected = case["filter_class"]
        factual = [n for n in inputs if gold_label(prop, n) == selected]
        counterfactual = [
            n
            for n in inputs
            if (target if n == value else gold_label(prop, n)) == selected
        ]
        if (
            factual != case["expected_factual"]
            or counterfactual != case["expected_target"]
        ):
            raise ValueError("Stored list targets disagree with independent arithmetic")
        found = [int(n) for n in re.findall(r"(?<!\d)-?\d+(?!\d)", text)]
        remainder = re.sub(r"(?<!\d)-?\d+(?!\d)", "", text)
        parsed = all(c.isspace() or c in ",;[]()." for c in remainder)
        return {
            "parsed": parsed,
            "identity_preserved": set(found) <= set(inputs)
            and len(found) == len(set(found)),
            "factual": parsed and found == factual,
            "target_success": parsed and found == counterfactual,
            "parsed_numbers": found,
        }
    text = "Result after adding zero:" + text
    numbers = re.findall(r"Result after adding zero\s*:\s*(-?\d+)(?!\d)", text, re.I)
    terms = "|".join(re.escape(s) for s in sorted(labels, key=len, reverse=True))
    properties = re.findall(
        re.escape(property_spec["field"]) + rf"\s*:\s*({terms})(?![a-zA-Z0-9])",
        text,
        re.I,
    )
    observed = next(
        (
            i
            for i, s in enumerate(labels)
            if properties and s.lower() == properties[0].lower()
        ),
        None,
    )
    numeral = int(numbers[0]) if numbers else None
    identity = numeral == value
    return {
        "parsed": bool(numbers and properties),
        "identity_preserved": identity,
        "factual": identity and observed == source,
        "target_success": identity and observed == target,
        "parsed_property": observed,
        "parsed_numeral": numeral,
    }


def aggregate(rows, baseline=None):
    counts = Counter(r["source_class"] for r in rows)
    classes = sorted(counts)
    by_id = {r["case_id"]: r for r in (baseline or rows)}
    if set(by_id) != {r["case_id"] for r in rows}:
        raise ValueError("Unpaired examples")
    eligible = (
        rows
        if baseline is None
        else [r for r in rows if by_id[r["case_id"]]["factual"]]
    )
    result = {"n": len(rows), "source_class_counts": dict(counts)}
    for name in ["parsed", "identity_preserved", "factual", "target_success"]:
        result[name] = sum(bool(r[name]) for r in rows) / len(rows)
        result["macro_" + name] = sum(
            sum(bool(r[name]) for r in rows if r["source_class"] == c) / counts[c]
            for c in classes
        ) / len(classes)
    result["baseline_correct_n"] = len(eligible)
    result["conditional_target_success"] = (
        sum(r["target_success"] for r in eligible) / len(eligible) if eligible else None
    )
    result["by_source_class"] = {}
    for c in classes:
        group = [r for r in rows if r["source_class"] == c]
        correct = [r for r in eligible if r["source_class"] == c]
        result["by_source_class"][str(c)] = {
            "n": len(group),
            "baseline_correct_n": len(correct),
            "target_success_count": sum(r["target_success"] for r in group),
            "target_success": sum(r["target_success"] for r in group) / len(group),
            "conditional_target_success": sum(r["target_success"] for r in correct)
            / len(correct)
            if correct
            else None,
            "identity_preserved": sum(r["identity_preserved"] for r in group)
            / len(group),
        }
    result["macro_conditional_target_success"] = sum(
        result["by_source_class"][str(c)]["conditional_target_success"] or 0
        for c in classes
    ) / len(classes)
    for name in ["nominal_norm", "applied_norm"]:
        result[name + "_mean"] = sum(float(r.get(name, 0)) for r in rows) / len(rows)
    return result


def same(actual, expected):
    # JSON object keys are strings; integer class keys are canonicalized here.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def clustered_interval(cohorts, iterations=2000):
    """Resample numeral identities within classes, shared across all value splits."""
    rng = np.random.default_rng(20261110)
    seeds = sorted({row[0] for row in cohorts})
    classes = sorted({row[2] for row in cohorts})
    draws = np.zeros((iterations, len(seeds), 4))
    valid = np.ones(iterations, dtype=bool)
    for cls in classes:
        group = [r for r in cohorts if r[2] == cls]
        values = sorted({r[1] for r in group})
        success = np.zeros((len(values), len(seeds), 4))
        total = np.zeros((len(values), len(seeds)))
        for seed, value, _, correct, outcomes in group:
            i, j = values.index(value), seeds.index(seed)
            total[i, j] += int(correct)
            success[i, j] += np.asarray(outcomes) * int(correct)
        weights = rng.multinomial(
            len(values), np.full(len(values), 1 / len(values)), size=iterations
        )
        numerator = weights @ success.reshape(len(values), -1)
        denominator = weights @ total
        # A resample with no eligible numeral in an originally observed
        # class/split has an undefined rate; it must not become a zero success.
        valid &= ((denominator > 0) | (total.sum(axis=0) == 0)).all(axis=1)
        numerator = numerator.reshape(iterations, len(seeds), 4)
        draws += np.divide(
            numerator,
            denominator[:, :, None],
            out=np.zeros_like(numerator),
            where=denominator[:, :, None] != 0,
        ) / len(classes)
    if valid.sum() < 100:
        raise ValueError("Insufficient nonempty clustered bootstrap replicates")
    draws = draws[valid].mean(axis=1)
    gap = draws[:, 0] - draws[:, 1:].max(axis=1)
    return {
        "target_interval": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "control_gap_interval": np.quantile(gap, [0.025, 0.975]).tolist(),
        "unique_test_numerals": len({r[1] for r in cohorts}),
        "bootstrap_valid_draws": int(valid.sum()),
    }


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    results = sorted(root.glob("*/*/results.json"))
    if len(results) != 120:
        raise ValueError(f"Expected 120 logical runs, found {len(results)}")
    cells, baseline_rows, examples = [], [], []
    groups = defaultdict(list)
    cohorts = defaultdict(list)
    replayed = 0
    for path in results:
        result = json.loads(path.read_text())
        task, directory = result["task"], path.parent
        if (directory / "DONE").read_text().strip() != result["run_identity"]:
            raise ValueError("DONE identity mismatch")
        partition = json.loads((directory / "splits.json").read_text())["values"]
        values = {s: set(v) for s, v in partition.items()}
        if (
            set.union(*values.values()) != set(range(1000))
            or sum(map(len, values.values())) != 1000
        ):
            raise ValueError("Value partitions overlap or omit numerals")
        data = {}
        for relative in sorted(set(result["generation_index"].values())):
            with gzip.open(directory / relative, "rt") as stream:
                payload = json.load(stream)
            if Path(relative).name != payload["cache_key"] + ".json.gz":
                raise ValueError("Cache key mismatch")
            rows = payload["rows"]
            for row in rows:
                check = independent_score(row, result["property"])
                if any(row[k] != v for k, v in check.items()):
                    raise ValueError("Independent completion replay failed")
                split = row["case_id"].split(":")[0]
                if (
                    row["case"]["value"] not in values[split]
                    or not set(row["case"]["input_values"]) <= values[split]
                ):
                    raise ValueError("Behavioral case crosses numeral partitions")
            replayed += len(rows)
            data[relative] = rows
        index = result["generation_index"]
        families = result.get("families", ["joint", "list"])
        for family in families:
            for split in ["dev", "test"]:
                rows = data[index[f"{family}/{split}/baseline"]]
                if not same(aggregate(rows), result["baseline_metrics"][family][split]):
                    raise ValueError("Independent baseline aggregation failed")
                if not task["pilot"]:
                    for cls in sorted(
                        {r["case"]["filter_class"] for r in rows}, key=str
                    ):
                        subset = [r for r in rows if r["case"]["filter_class"] == cls]
                        baseline_rows.append(
                            {
                                **task,
                                "family": family,
                                "split": split,
                                "filter_class": cls,
                                "n": len(subset),
                                "correct": sum(r["factual"] for r in subset),
                            }
                        )
        for cell in result["results"]:
            if cell["status"] != "evaluated":
                continue
            context, family, mode = (
                cell["fit_context"],
                cell["family"],
                cell["site_mode"],
            )
            development = data[index[f"{family}/dev/baseline"]]
            sweep = json.loads(
                (directory / f"selection-{context}-{family}.json").read_text()
            )
            for candidate in sweep:
                label = f"{context}/{family}/dev/l{candidate['layer']}/a{candidate['alpha']}"
                if not same(
                    aggregate(data[index[label]], development), candidate["metrics"]
                ):
                    raise ValueError("Independent development aggregation failed")
            candidates = [
                c
                for c in sweep
                if mode != "fixed"
                or c["layer"] == result["runtime"]["model"]["fixed_layer"]
            ]
            eligible = [
                c
                for c in candidates
                if c["metrics"]["macro_identity_preserved"] >= 0.95
            ]
            chosen = max(
                eligible or candidates,
                key=lambda c: (
                    c["metrics"]["macro_conditional_target_success"],
                    c["metrics"]["macro_identity_preserved"],
                    -c["alpha"],
                    -c["layer"],
                ),
            )
            if not same(
                {**chosen, "identity_gate_passed": bool(eligible)}, cell["selection"]
            ):
                raise ValueError(
                    "Saved intervention choice is not the development-only optimum"
                )
            baseline = data[index[f"{family}/test/baseline"]]
            conditions = [
                "selected_direction",
                "wrong_sign",
                "random_direction",
                "delimiter_location",
            ]
            records = []
            for condition in conditions:
                rows = data[index[f"{context}/{family}/{mode}/test/{condition}"]]
                if not same(aggregate(rows, baseline), cell["conditions"][condition]):
                    raise ValueError("Independent intervention aggregation failed")
                records.append({r["case_id"]: r for r in rows})
            if task["pilot"]:
                continue
            rates = [
                cell["conditions"][name]["macro_conditional_target_success"]
                for name in conditions
            ]
            flat = {
                **task,
                "context": context,
                "family": family,
                "mode": mode,
                "layer": cell["selection"]["layer"],
                "alpha": cell["selection"]["alpha"],
                "dev_identity_gate": cell["selection"]["identity_gate_passed"],
                "test_identity": cell["conditions"][conditions[0]][
                    "macro_identity_preserved"
                ],
                "test_baseline": cell["baseline"]["test"]["macro_factual"],
                "n": len(baseline),
                "baseline_correct_n": cell["conditions"][conditions[0]][
                    "baseline_correct_n"
                ],
                "selected": rates[0],
                "wrong_sign": rates[1],
                "random": rates[2],
                "location": rates[3],
                "worst_class": min(
                    v["conditional_target_success"] or 0
                    for v in cell["conditions"][conditions[0]][
                        "by_source_class"
                    ].values()
                ),
            }
            flat["class_outcomes"] = {
                str(c): {
                    "n": cell["conditions"][conditions[0]]["by_source_class"][str(c)][
                        "n"
                    ],
                    "baseline_correct_n": cell["conditions"][conditions[0]][
                        "by_source_class"
                    ][str(c)]["baseline_correct_n"],
                    "conditional_success_rates": [
                        cell["conditions"][condition]["by_source_class"][str(c)][
                            "conditional_target_success"
                        ]
                        for condition in conditions
                    ],
                }
                for c in sorted({r["source_class"] for r in baseline})
            }
            cells.append(flat)
            key = (task["model"], task["property"], context, family, mode)
            groups[key].append(flat)
            for base in baseline:
                outcomes = [
                    record[base["case_id"]]["target_success"] for record in records
                ]
                cohorts[key].append(
                    (
                        task["seed"],
                        base["case"]["value"],
                        base["source_class"],
                        base["factual"],
                        outcomes,
                    )
                )
            if (
                task["seed"] == 20261110
                and context == "property"
                and mode == "selected"
            ):
                for kind in ["success", "failure"]:
                    row = next(
                        (
                            r
                            for r in records[0].values()
                            if r["target_success"] == (kind == "success")
                            and next(
                                b for b in baseline if b["case_id"] == r["case_id"]
                            )["factual"]
                        ),
                        None,
                    )
                    if row:
                        base = next(
                            b for b in baseline if b["case_id"] == row["case_id"]
                        )
                        examples.append(
                            {
                                **task,
                                "family": family,
                                "kind": kind,
                                "selection": "first matching saved test case; no random sampling",
                                "case": row["case"],
                                "baseline": base["generated_text"],
                                "intervention": row["generated_text"],
                            }
                        )
        if not task["pilot"]:
            for family in families:
                for row in data[index[f"{family}/dev/baseline"]]:
                    if (
                        family == "list"
                        and not row["factual"]
                        and row["case"]["filter_class"] == 0
                        and task["seed"] == 20261110
                    ):
                        examples.append(
                            {
                                **task,
                                "family": family,
                                "kind": "baseline_failure",
                                "selection": "first saved development failure for the zero label",
                                "case": row["case"],
                                "baseline": row["generated_text"],
                            }
                        )
                        break
    summaries = []
    for key, rows in sorted(groups.items()):
        summary = dict(zip(["model", "property", "context", "family", "mode"], key))
        summary["evaluated_seeds"] = len(rows)
        for metric in [
            "selected",
            "wrong_sign",
            "random",
            "location",
            "test_identity",
            "test_baseline",
            "worst_class",
        ]:
            summary[metric] = float(np.mean([r[metric] for r in rows]))
        summary["seed_min"], summary["seed_max"] = (
            min(r["selected"] for r in rows),
            max(r["selected"] for r in rows),
        )
        summary["all_dev_identity_gates"] = all(r["dev_identity_gate"] for r in rows)
        summary["baseline_correct_n"] = sum(r["baseline_correct_n"] for r in rows)
        summary.update(clustered_interval(cohorts[key]))
        summaries.append(summary)
    audit = {
        "status": "passed",
        "logical_runs": len(results),
        "main_runs": sum(
            not json.loads(p.read_text())["task"]["pilot"] for p in results
        ),
        "unique_saved_generation_records": replayed,
        "evaluated_comparisons": len(cells),
        "baseline_cells": len(baseline_rows),
        "independent_arithmetic_and_aggregates": True,
        "uncertainty": "2000 percentile bootstrap draws; numeral clusters shared across value splits; stratified by source class; exploratory intervals without multiplicity correction",
    }
    for name, value in [
        ("analysis.json", summaries),
        ("cells.json", cells),
        ("baseline.json", baseline_rows),
        ("examples.json", examples),
        ("validation.json", audit),
    ]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    return audit
