"""Check frozen search observations and screening identities without network access.

These checks establish consistency of the preserved packet, not independent
verification of historical database execution or result totals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", " ".join(title.split())).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def verify(data_dir: Path) -> dict:
    data_dir = data_dir.resolve()
    ledger = json.loads((data_dir / "retrieval_ledger.json").read_text())
    if ledger.get("schema_version") != 1:
        raise ValueError("Unsupported retrieval ledger schema")
    required = {
        "fulltext_comparison_counts.csv", "screening_candidates.csv",
        "source_evidence/annual_count_records.csv", "source_evidence/collector_run.json",
        "source_evidence/collector_excerpt.txt",
    }
    if not required.issubset(ledger["files"]):
        raise ValueError("Retrieval ledger omits required evidence files")
    for name, record in ledger["files"].items():
        path = (data_dir / name).resolve()
        if not path.is_relative_to(data_dir) or not path.is_file():
            raise ValueError(f"Missing or invalid retrieval evidence path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Retrieval evidence hash mismatch: {name}")

    annual = read_csv(data_dir / "fulltext_comparison_counts.csv")
    source = read_csv(data_dir / "source_evidence/annual_count_records.csv")
    columns = ("series_id", "snapshot_date", "year", "result_count", "partial_year")

    def observations(rows: list[dict[str, str]]) -> dict:
        result = {}
        for row in rows:
            key = (row["series_id"], int(row["year"]))
            if key in result:
                raise ValueError(f"Duplicate annual observation: {key}")
            result[key] = tuple(row[column] for column in columns)
        return result

    if observations(annual) != observations(source):
        raise ValueError("Plotted counts disagree with the historical source table")
    series = {row["series_id"]: row for row in ledger["annual_series"]}
    if set(series) != {row["series_id"] for row in annual}:
        raise ValueError("Annual series and retrieval ledger disagree")
    for series_id, record in series.items():
        rows = [row for row in annual if row["series_id"] == series_id]
        counts = {row["year"]: int(row["result_count"]) for row in rows}
        partial = [int(row["year"]) for row in rows if row["partial_year"].lower() == "true"]
        if counts != record["annual_counts"] or partial != record["partial_years"]:
            raise ValueError(f"Annual counts or partial-year labels disagree: {series_id}")
        if {row["snapshot_date"] for row in rows} != {record["snapshot_date"]}:
            raise ValueError(f"Snapshot dates disagree: {series_id}")
        if record["historical_execution_independently_verified"]:
            raise ValueError("This packet lacks evidence for independent historical execution")
    if ledger["historical_retrieval_independently_verified"]:
        raise ValueError("Frozen aggregates do not independently verify historical retrieval")

    candidates = read_csv(data_dir / "screening_candidates.csv")
    screening = ledger["screening"]
    ids: set[str] = set()
    dedup_keys: set[str] = set()
    aliases = []
    ownership: dict[str, str] = {}
    resolved = positive = retained = 0
    for row in candidates:
        paper_id = row["openalex_id"]
        title = normalized_title(row["title"])
        if title:
            dedup_key = f"title:{title}"
        elif row["arxiv_id"]:
            dedup_key = f"arxiv:{row['arxiv_id']}"
        elif row["doi"]:
            dedup_key = f"doi:{row['doi'].casefold()}"
        else:
            dedup_key = f"openalex:{paper_id}"
        if not re.fullmatch(r"W\d+", paper_id) or paper_id in ids:
            raise ValueError(f"Invalid or duplicate screening identity: {paper_id}")
        if dedup_key in dedup_keys:
            raise ValueError(f"Duplicate candidate deduplication key: {paper_id}")
        ids.add(paper_id)
        dedup_keys.add(dedup_key)
        members = row["source_openalex_ids"].split("|")
        if members[0] != paper_id or row["primary_field_id"] != screening["field_id"]:
            raise ValueError(f"Candidate membership or primary field disagrees: {paper_id}")
        for member in members:
            if not re.fullmatch(r"W\d+", member):
                raise ValueError(f"Invalid source membership: {member}")
            if member in ownership and ownership[member] != paper_id:
                raise ValueError(f"Source identity assigned to multiple title groups: {member}")
            ownership[member] = paper_id
        aliases.extend(members)
        matched = bool(row["semantic_scholar_paper_id"])
        cited = bool(row["citation_count"]) and int(row["citation_count"]) > 0
        included = row["retained_in_fulltext_corpus"] == "true"
        if row["retained_in_fulltext_corpus"] not in {"true", "false"}:
            raise ValueError(f"Invalid corpus inclusion value: {paper_id}")
        if cited and not matched or included and not cited:
            raise ValueError(f"Invalid screening transition: {paper_id}")
        resolved += matched
        positive += cited
        retained += included
    observed = {
        "title_groups": len(candidates), "source_id_memberships": len(aliases),
        "unique_source_ids": len(set(aliases)), "resolved_to_semantic_scholar": resolved,
        "positive_citation_candidates": positive, "retained_documents": retained,
    }
    if any(screening[key] != value for key, value in observed.items()):
        raise ValueError("Screening denominator or candidate transitions disagree with ledger")
    run = json.loads((data_dir / "source_evidence/collector_run.json").read_text())
    if (run["openalex"]["fulltext_retrieved"] != len(aliases)
            or run["openalex"]["fulltext_deduped"] != len(candidates)
            or run["software"]["collector_sha256"] != screening["recorded_collector_sha256"]
            or run["started_at"] != screening["run_started_at"]
            or run["completed_at"] != screening["run_completed_at"]):
        raise ValueError("Screening ledger disagrees with the archived collector-run record")
    if screening["exact_recorded_collector_recovered"]:
        raise ValueError("The exact historical collector executable was not recovered")
    broad_total = sum(int(row["result_count"]) for row in annual
                      if row["series_id"] == "openalex_current")
    relationship = ledger["annual_to_screening_relationship"]
    if broad_total != relationship["plotted_openalex_2022_2026_sum"]:
        raise ValueError("OpenAlex annual total disagrees with ledger")
    if relationship["screening_is_deduplicated_subset_of_annual_response_verified"]:
        raise ValueError("Annual-query response membership was not preserved")
    return {
        "annual_observations": len(annual), "plotted_openalex_total": broad_total,
        "screening": observed, "verified_evidence_files": len(ledger["files"]),
        "historical_retrieval_independently_verified": False,
        "verification": "frozen observations and candidate mapping",
        "limitations": ledger["limitations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().data_dir), indent=2))
