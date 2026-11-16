import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lrh.analysis.retrieval_provenance import verify

SOURCE = Path(__file__).resolve().parents[1] / "data/mentions"


@pytest.fixture
def packet(tmp_path):
    shutil.copytree(SOURCE, tmp_path / "mentions")
    return tmp_path / "mentions"


def update_hash(packet, name):
    path = packet / "retrieval_ledger.json"
    ledger = json.loads(path.read_text())
    ledger["files"][name]["sha256"] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(ledger))


def test_preserved_screening_chain_and_historical_limits(packet):
    result = verify(packet)
    assert result["annual_observations"] == 14
    assert result["plotted_openalex_total"] == 866
    assert result["screening"] == {
        "title_groups": 567, "source_id_memberships": 651, "unique_source_ids": 649,
        "resolved_to_semantic_scholar": 519, "positive_citation_candidates": 311,
        "retained_documents": 206,
    }
    assert result["historical_retrieval_independently_verified"] is False


def test_plot_cannot_drift_from_preserved_source_table(packet):
    path = packet / "fulltext_comparison_counts.csv"
    path.write_text(path.read_text().replace(",2026,526,", ",2026,527,"))
    update_hash(packet, path.name)
    with pytest.raises(ValueError, match="historical source table"):
        verify(packet)


def test_one_source_id_cannot_belong_to_two_title_groups(packet):
    path = packet / "screening_candidates.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    target = next(row for row in rows if "|" in row["source_openalex_ids"])
    other = next(row for row in rows if row["openalex_id"] != target["openalex_id"])
    members = target["source_openalex_ids"].split("|")
    members[1] = other["openalex_id"]
    target["source_openalex_ids"] = "|".join(members)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    update_hash(packet, path.name)
    with pytest.raises(ValueError, match="multiple title groups"):
        verify(packet)


def test_integrity_checks_cannot_certify_missing_historical_execution(packet):
    path = packet / "retrieval_ledger.json"
    ledger = json.loads(path.read_text())
    ledger["historical_retrieval_independently_verified"] = True
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="independently verify historical retrieval"):
        verify(packet)
