"""Export public audit labels from a saved private annotation export."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from lrh.analysis.human_audit import PURPOSES, compute, read_csv, sha256
from lrh.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(
    private_export: Path,
    study_manifest: Path,
    label_config: Path,
    original_coding_dir: Path,
    root: Path,
) -> dict:
    records = json.loads(private_export.read_text())
    study = json.loads(study_manifest.read_text())
    original_hashes, released_hashes = {}, {}
    for name, expected in study["source_hashes"].items():
        if name.startswith("private/"):
            continue
        original = original_coding_dir / name
        released = root / "coding" / name
        if sha256(original) != expected:
            raise ValueError(f"Original coding input changed since review: {name}")
        if original.suffix == ".jsonl":
            fields = ("episode_id", "episode_status", "rhetorical_functions")

            def judgments(path, selected_fields=fields):
                return {
                    row["episode_id"]: {key: row[key] for key in selected_fields}
                    for row in map(json.loads, path.read_text().splitlines())
                }

            equal = judgments(original) == judgments(released)
        else:
            fields = ("episode_id", "paper_id", "primary_worker", "audit_worker")
            equal = [{key: row[key] for key in fields} for row in read_csv(original)] == [
                {key: row[key] for key in fields} for row in read_csv(released)
            ]
        if not equal:
            raise ValueError(f"Released coding judgments differ from the review inputs: {name}")
        original_hashes["coding/" + name] = expected
        released_hashes["coding/" + name] = sha256(released)
    if len({r["reviewer"] for r in records}) != 1:
        raise ValueError("Select exactly one completed author review for this release")
    folder = root / "human_audit"
    folder.mkdir(parents=True, exist_ok=True)
    locations = {
        r["episode_id"]: r for r in json.loads((root / "coding/episode_locations.json").read_text())
    }
    assignments = {
        (phase, r["episode_id"]): r
        for phase in ("strong_coding", "expanded_coding")
        for r in read_csv(root / "coding" / phase / "assignment_manifest.csv")
    }
    study_cases = {r["case_id"]: r for r in study["cases"]}
    annotations, cases = [], []
    for record in sorted(records, key=lambda r: r["case_id"]):
        case = study_cases[record["case_id"]]
        if (case["episode_id"], case["phase"]) != (record["episode_id"], record["phase"]):
            raise ValueError("Saved judgment differs from the frozen review task")
        location = locations[record["episode_id"]]
        assignment = assignments[record["phase"], record["episode_id"]]
        annotations.append(
            {
                "reviewer_id": "author_1",
                "case_id": record["case_id"],
                "episode_id": record["episode_id"],
                "phase": record["phase"],
                "episode_status": record["episode_status"],
                **{
                    key: (
                        "U" if record["purposes"].get(key) is None else str(record["purposes"][key])
                    )
                    if record["episode_status"] == "substantive"
                    else "N/A"
                    for key in PURPOSES
                },
            }
        )
        cases.append(
            {
                "case_id": record["case_id"],
                "episode_id": record["episode_id"],
                "phase": record["phase"],
                "paper_id": assignment["paper_id"],
                "paper_title": location["paper_title"],
                "year": location["year"],
                "citation_rank": assignment["citation_rank"],
                "page_start": location["page_start"],
                "page_end": location["page_end"],
                "context_sha256": case["context_sha256"],
            }
        )
    write_csv(folder / "annotations.csv", annotations)
    write_csv(folder / "case_manifest.csv", cases)
    shutil.copyfile(label_config, folder / "label_config.xml")
    dates = {datetime.fromisoformat(r["updated_at"]).date().isoformat() for r in records}
    write_json(
        folder / "provenance.json",
        {
            "schema_version": 1,
            "reviewer_id": "author_1",
            "reviewer_role": "one author",
            "review_dates": sorted(dates),
            "source_annotations_sha256": sha256(private_export),
            "source_study_manifest_sha256": sha256(study_manifest),
            "label_config_sha256": study["label_config_sha256"],
            "display_randomization_seed": study["randomization_seed"],
            "selection": study["selection"],
            "blinding": "The interface hid machine judgments, citation ranks, and sampling groups.",
            "original_automated_source_sha256": original_hashes,
            "automated_source_sha256": released_hashes,
            "released_source_verification": (
                "Episode identifiers, statuses, rhetorical functions, and coder assignments match "
                "the original review inputs; file serialization may differ in the release."
            ),
            "passage_corpus_sha256": study["source_hashes"]["private/expanded_episodes.jsonl"],
            "earlier_targeted_packet": {
                "total_cases": 30,
                "status": "prepared separately before the completed 60-case review",
                "selection": "Twenty purpose-disagreement cases and ten controls.",
            },
            "export_fields": list(annotations[0]),
            "private_fields_excluded": [
                "reviewer account",
                "annotation identifier",
                "save timestamp",
                "free-text notes",
                "revision history",
                "credentials",
                "database",
                "source passages",
                "source PDFs",
            ],
            "interpretation": (
                "One author reviewed the selected episodes. Human-machine agreement concerns this "
                "stratified sample; human intercoder reliability and corpus-wide error are unmeasured."
            ),
        },
    )
    metrics = compute(root)
    write_json(folder / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-export", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, required=True)
    parser.add_argument("--original-coding-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/literature")
    args = parser.parse_args()
    metrics = export(
        args.private_export,
        args.study_manifest,
        args.label_config,
        args.original_coding_dir,
        args.data_dir,
    )
    print(
        f"Exported {metrics['episodes']} completed author judgments from {metrics['documents']} documents"
    )


if __name__ == "__main__":
    main()
