"""Recompute the completed author audit from released, case-level judgments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

PURPOSES = ("define_position", "motivate_apply", "test_support", "qualify_challenge")
STATUSES = {"substantive", "bibliography_only", "structural_fragment", "context_insufficient"}
PHASES = ("strong_coding", "expanded_coding")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def binary_metrics(human: list[int], machine: list[int]) -> dict:
    n = len(human)
    tp = sum(h == m == 1 for h, m in zip(human, machine, strict=True))
    tn = sum(h == m == 0 for h, m in zip(human, machine, strict=True))
    fp = sum(h == 0 and m == 1 for h, m in zip(human, machine, strict=True))
    fn = sum(h == 1 and m == 0 for h, m in zip(human, machine, strict=True))
    observed = (tp + tn) / n if n else None
    hp, mp = (sum(human) / n, sum(machine) / n) if n else (0, 0)
    expected = hp * mp + (1 - hp) * (1 - mp)
    return {
        "n": n,
        "agreement": observed,
        "kappa": (observed - expected) / (1 - expected) if n and expected < 1 else None,
        "both_present": tp,
        "both_absent": tn,
        "machine_only": fp,
        "human_only": fn,
        "machine_precision": tp / (tp + fp) if tp + fp else None,
        "machine_recall": tp / (tp + fn) if tp + fn else None,
    }


def compare(humans: list[dict], machines: dict, functions: dict) -> dict:
    paired = [(row, machines[row["episode_id"]]) for row in humans]
    resolved = [(h, m) for h, m in paired if h["episode_status"] != "context_insufficient"]
    substantive = [
        (h, m) for h, m in paired if h["episode_status"] == m["episode_status"] == "substantive"
    ]

    def machine_purposes(row):
        return {key for key in PURPOSES if set(functions[key]) & set(row["rhetorical_functions"])}

    purpose_metrics = {}
    for key in PURPOSES:
        usable = [
            (int(h[key]), int(key in machine_purposes(m))) for h, m in substantive if h[key] != "U"
        ]
        purpose_metrics[key] = {
            **binary_metrics([h for h, _ in usable], [m for _, m in usable]),
            "uncertain_excluded": len(substantive) - len(usable),
        }
    complete = [(h, m) for h, m in substantive if all(h[key] != "U" for key in PURPOSES)]
    exact, jaccards = [], []
    for h, m in complete:
        hs = {key for key in PURPOSES if h[key] == "1"}
        ms = machine_purposes(m)
        exact.append(hs == ms)
        jaccards.append(len(hs & ms) / len(hs | ms) if hs | ms else 1.0)
    return {
        "paired_completed": len(paired),
        "status_resolved_n": len(resolved),
        "status_agreement": (
            sum(h["episode_status"] == m["episode_status"] for h, m in resolved) / len(resolved)
            if resolved
            else None
        ),
        "status_confusion_human_then_machine": dict(
            sorted(
                Counter(
                    f"{h['episode_status']} | {m['episode_status']}" for h, m in resolved
                ).items()
            )
        ),
        "both_substantive_n": len(substantive),
        "purpose_set_resolved_n": len(complete),
        "purpose_exact_set_agreement": sum(exact) / len(exact) if exact else None,
        "purpose_mean_jaccard": math.fsum(jaccards) / len(jaccards) if jaccards else None,
        "purpose_binary": purpose_metrics,
    }


def compute(root: Path) -> dict:
    folder = root / "human_audit"
    rows = read_csv(folder / "annotations.csv")
    cases = read_csv(folder / "case_manifest.csv")
    provenance = json.loads((folder / "provenance.json").read_text())
    summary = json.loads((root / "purpose_summary.json").read_text())
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate human audit case")
    if len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate human audit episode")
    if {r["reviewer_id"] for r in rows} != {"author_1"}:
        raise ValueError("The released audit must identify its single author reviewer")
    manifest = {r["case_id"]: r for r in cases}
    if len(manifest) != len(cases) or set(manifest) != {r["case_id"] for r in rows}:
        raise ValueError("Human audit cases differ from the released manifest")
    for row in rows:
        case = manifest[row["case_id"]]
        if any(row[key] != case[key] for key in ("episode_id", "phase")):
            raise ValueError("Human audit episode or sampling phase differs from the manifest")
        if row["episode_status"] not in STATUSES:
            raise ValueError("Invalid human episode status")
        allowed = {"0", "1", "U"} if row["episode_status"] == "substantive" else {"N/A"}
        if any(row[key] not in allowed for key in PURPOSES):
            raise ValueError("Invalid human purpose judgment")
    for name, expected in provenance["automated_source_sha256"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root.resolve()) or sha256(source) != expected:
            raise ValueError(f"Human audit machine-source hash differs: {name}")
    if sha256(folder / "label_config.xml") != provenance["label_config_sha256"]:
        raise ValueError("Human audit annotation instructions changed")
    machines = {role: {} for role in ("primary", "secondary")}
    for phase in PHASES:
        path = root / "coding" / phase
        lookups = {
            coder: {
                r["episode_id"]: r
                for r in (
                    json.loads(line)
                    for line in (path / f"{coder}_annotations.jsonl").read_text().splitlines()
                    if line.strip()
                )
            }
            for coder in ("coder_a", "coder_b")
        }
        expected_ids = set()
        for assignment in read_csv(path / "assignment_manifest.csv"):
            if assignment["audit_worker"]:
                episode = assignment["episode_id"]
                expected_ids.add(episode)
                machines["primary"][episode] = lookups[assignment["primary_worker"]][episode]
                machines["secondary"][episode] = lookups[assignment["audit_worker"]][episode]
        if {r["episode_id"] for r in rows if r["phase"] == phase} != expected_ids:
            raise ValueError("Human audit coverage differs from the frozen duplicate-coded sample")
    functions = summary["purpose_functions"]
    return {
        "reviewer_id": "author_1",
        "reviewer_role": "one author",
        "episodes": len(rows),
        "documents": len({r["paper_id"] for r in cases}),
        "episode_status_counts": dict(sorted(Counter(r["episode_status"] for r in rows).items())),
        "comparisons": {
            role: compare(rows, machine, functions) for role, machine in machines.items()
        },
        "by_audit": {
            phase: {
                role: compare([r for r in rows if r["phase"] == phase], machine, functions)
                for role, machine in machines.items()
            }
            for phase in PHASES
        },
    }


def verify(root: Path) -> dict:
    metrics = compute(root)
    stored = json.loads((root / "human_audit/metrics.json").read_text())
    if metrics != stored:
        raise ValueError("Human audit metrics differ from the released judgments")
    declared = json.loads((root / "purpose_summary.json").read_text())["human_audit"]
    if (
        declared["status"] != "completed"
        or declared["total_cases"] != metrics["episodes"]
        or declared["documents"] != metrics["documents"]
        or declared["episode_status_counts"] != metrics["episode_status_counts"]
        or declared["annotations_sha256"] != sha256(root / "human_audit/annotations.csv")
    ):
        raise ValueError("Human audit summary differs from the released judgments")
    return metrics


def verify_manuscript(source: str, metrics: dict) -> None:
    parts = source.split(r"\appendixsection{Search Procedure", 1)
    if len(parts) != 2:
        raise ValueError("Manuscript has no search-procedure appendix")
    source = parts[1]
    match = re.search(r"One author reviewed (\d+) episodes from (\d+) documents", source)
    if not match or tuple(map(int, match.groups())) != (metrics["episodes"], metrics["documents"]):
        raise ValueError("Manuscript human audit sample differs from the released judgments")
    row = re.search(r"Human\s*&\s*(\d+)\s*&\s*(\d+).*?&\s*(\d+)\s*&\s*(\d+)\s*\\\\", source)
    counts = metrics["episode_status_counts"]
    expected = (
        metrics["episodes"],
        counts.get("substantive", 0),
        counts.get("bibliography_only", 0) + counts.get("structural_fragment", 0),
        counts.get("context_insufficient", 0),
    )
    if not row or tuple(map(int, row.groups())) != expected:
        raise ValueError("Manuscript human audit table differs from the released judgments")
    agreement = re.search(
        r"each agent agreed with the author's status judgment in (\d+) of (\d+) cases", source
    )
    for comparison in metrics["comparisons"].values():
        matches = sum(
            n
            for pair, n in comparison["status_confusion_human_then_machine"].items()
            if len(set(pair.split(" | "))) == 1
        )
        if not agreement or tuple(map(int, agreement.groups())) != (
            matches,
            comparison["status_resolved_n"],
        ):
            raise ValueError(
                "Manuscript human-machine agreement differs from the released judgments"
            )
