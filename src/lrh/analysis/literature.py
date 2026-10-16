"""Recompute overlapping purpose counts, coding agreement, and evidence tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lrh.analysis.coding import audit_pairs, paper_presence, purpose_agreement, sensitivity
from lrh.analysis.human_audit import verify as verify_human_audit
from lrh.io import write_json


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def jsonl_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_dir
    summary = json.loads((root / "purpose_summary.json").read_text())
    human_audit = verify_human_audit(root)
    primary = jsonl_rows(root / "coding/expanded_coding/corpus_annotations_primary.jsonl")
    versions = csv_rows(root / "paper_versions.csv")
    aliases = {row["version_document_id"]: row["canonical_document_id"] for row in versions}
    papers = paper_presence(primary, aliases)
    expected_labels = {
        row["paper_id"]: set(row["purpose_families"].split("|"))
        for row in csv_rows(root / "paper_labels.csv")
    }
    if papers != expected_labels:
        raise ValueError("Episode-to-paper aggregation differs from the reported labels")
    left, right = [], []
    by_round = {}
    for name in ("strong_coding", "expanded_coding"):
        folder = root / "coding" / name
        lookups = {
            coder: {
                row["episode_id"]: row for row in jsonl_rows(folder / f"{coder}_annotations.jsonl")
            }
            for coder in ("coder_a", "coder_b")
        }
        a, b = audit_pairs(csv_rows(folder / "assignment_manifest.csv"), lookups)
        left.extend(a)
        right.extend(b)
        by_round[name] = purpose_agreement(a, b)
    agreement = purpose_agreement(left, right)
    changes = sensitivity(primary, right, aliases)
    if agreement != {k: summary["reliability"][k] for k in agreement}:
        raise ValueError("Coding agreement differs from the reported summary")
    if changes != summary["audited_replacement_sensitivity"]["by_family"]:
        raise ValueError("Coding replacement sensitivity differs from the reported summary")
    presence = []
    for row in csv_rows(root / "paper_presence.csv"):
        count = sum(row["purpose_id"] in labels for labels in papers.values())
        if count != int(row["paper_count"]) or len(papers) != int(row["substantive_papers"]):
            raise ValueError("Paper-purpose count mismatch")
        presence.append(
            {
                "purpose_id": row["purpose_id"],
                "papers": count,
                "percentage": 100 * count / len(papers),
            }
        )
    ratings = csv_rows(root / "evidence_ratings.csv")
    studies = json.loads((root / "evidence_studies.json").read_text())
    expected = {
        (study["key"], claim): value
        for study in studies
        for claim, value in study["ratings"].items()
    }
    observed = {(row["paper"], row["claim"]): row["rating"] for row in ratings}
    if len(ratings) != 240 or len(studies) != 30 or observed != expected:
        raise ValueError("Evidence table cells disagree with the source-checked study records")
    if any(
        not row["source_pdf_url"].startswith("https://") or not row["evidence_pages"]
        for row in ratings
    ):
        raise ValueError("Evidence rating lacks a primary-source reference")
    write_json(
        args.output_dir / "metrics.json",
        {
            "coded_episodes": len(primary),
            "human_audit": human_audit,
            "substantive_papers": len(papers),
            "presence": presence,
            "duplicated_episodes": len(left),
            "agreement": agreement,
            "agreement_by_round": by_round,
            "replacement_sensitivity": changes,
            "evidence_studies": len(studies),
            "evidence_cells": len(ratings),
            "evidence_rating_frequencies": dict(Counter(row["rating"] for row in ratings)),
            "interpretation": "Purposes overlap within a selected corpus. Coding agreement is between automated agents. Evidence ratings are source-checked judgments; calculations do not independently validate those judgments.",
        },
    )
    print(
        f"Verified {len(primary)} coded episodes, {len(left)} audit pairs, and {len(ratings)} evidence cells"
    )


if __name__ == "__main__":
    main()
