from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from lrh.analysis.human_audit import PURPOSES, compute, read_csv, verify, verify_manuscript

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/literature"


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_completed_audit_reproduces_counts_and_uncertainty_exclusions():
    metrics = verify(DATA)
    assert (metrics["episodes"], metrics["documents"]) == (60, 54)
    assert metrics["episode_status_counts"] == {
        "substantive": 47,
        "bibliography_only": 12,
        "structural_fragment": 1,
    }
    for role, exact in (("primary", 12), ("secondary", 7)):
        comparison = metrics["comparisons"][role]
        assert comparison["status_agreement"] == 58 / 60
        assert comparison["both_substantive_n"] == 46
        assert comparison["purpose_set_resolved_n"] == 43
        assert comparison["purpose_exact_set_agreement"] == exact / 43
        assert [comparison["purpose_binary"][key]["n"] for key in PURPOSES] == [46, 45, 44, 46]
    verify_manuscript((ROOT / "paper/paper.tex").read_text(), metrics)


def test_duplicate_or_changed_human_judgments_cannot_pass_release_check(tmp_path):
    data = tmp_path / "literature"
    shutil.copytree(DATA, data)
    path = data / "human_audit/annotations.csv"
    rows = read_csv(path)
    write_rows(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate human audit case"):
        verify(data)
    changed = next(row for row in rows if row["episode_status"] == "substantive")
    changed["episode_status"] = "bibliography_only"
    for key in PURPOSES:
        changed[key] = "N/A"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="metrics differ"):
        verify(data)


def test_machine_source_changes_require_a_new_documented_audit(tmp_path):
    data = tmp_path / "literature"
    shutil.copytree(DATA, data)
    path = data / "coding/expanded_coding/coder_a_annotations.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="machine-source hash differs"):
        compute(data)


@pytest.mark.parametrize(
    "old,new",
    [
        ("One author reviewed 60 episodes", "One author reviewed 59 episodes"),
        (r"Human & 60 & 47", r"Human & 60 & 46"),
        ("judgment in 58 of 60 cases", "judgment in 57 of 60 cases"),
    ],
)
def test_manuscript_count_changes_require_matching_released_judgments(old, new):
    source = (ROOT / "paper/paper.tex").read_text()
    assert old in source
    with pytest.raises(ValueError, match="differs from the released judgments"):
        verify_manuscript(source.replace(old, new, 1), verify(DATA))
