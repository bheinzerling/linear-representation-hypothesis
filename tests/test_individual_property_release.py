from __future__ import annotations

import gzip
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(str(ROOT / "src/lrh/analysis/individual_property_control.py"))
DATA = ROOT / "data/individual_property_control"


def test_complete_inventory_derives_seven_properties_and_exact_manuscript_rows():
    cells = json.loads((DATA / "explicit_predicate_cells.json").read_text())
    protocol = json.loads((DATA / "protocol.json").read_text())
    selected = RUNNER["selective_properties"](cells, list(RUNNER["TABLE_NAMES"]), protocol["split_seeds"])
    assert set(selected) == {"parity", "primality", "divisible_by_3", "divisible_by_5",
                             "divisible_by_10", "has_repeated_digit", "palindrome"}
    rows = [r for r in json.loads((DATA / "summary.json").read_text())
            if (r["model"], r["context"], r["family"], r["mode"])
            == ("qwen36_27b", "property", "list", "selected")]
    assert RUNNER["table_rows"](rows, selected) in (ROOT / "paper/paper.tex").read_text()
    assert "six properties" not in (ROOT / "paper/paper.tex").read_text()


def test_selection_is_derived_for_every_property_and_requires_both_class_transitions():
    cells = [{"model": "qwen36_27b", "property": "new_property", "context": "property",
              "family": "list", "mode": "selected", "seed": seed, "dev_identity_gate": True,
              "class_outcomes": {c: {"baseline_correct_n": 10,
                                    "conditional_success_rates": [0.1, 0.0, 0.0, 0.0]}
                                 for c in ("0", "1")}} for seed in (1, 2, 3)]
    assert RUNNER["selective_properties"](cells, ["new_property"], [1, 2, 3]) == ["new_property"]
    cells[0]["class_outcomes"]["1"]["conditional_success_rates"][1] = 0.1
    assert RUNNER["selective_properties"](cells, ["new_property"], [1, 2, 3]) == []


def test_record_unpack_restores_shared_case_and_rejects_escaping_paths(tmp_path):
    archive = tmp_path / "records.gz"
    definitions = {"test:1": {"case": {"value": 10}, "prompt": "10="}}
    payload = {"cache_key": "abc", "rows": [{"case_id": "test:1", "generated_text": "10"}]}
    records = [{"path": "main/run/000-case-definitions.json", "payload": definitions},
               {"path": "main/run/generations/abc.json.gz", "payload": payload}]
    archive.write_bytes(gzip.compress("\n".join(json.dumps(r) for r in records).encode()))
    RUNNER["unpack"](archive, tmp_path / "restored")
    with gzip.open(tmp_path / "restored/main/run/generations/abc.json.gz", "rt") as handle:
        actual = json.load(handle)["rows"][0]
    assert actual["case"] == {"value": 10} and actual["prompt"] == "10="
    archive.write_bytes(gzip.compress(json.dumps({"path": "../escaped", "payload": {}}).encode()))
    with pytest.raises(ValueError, match="Unsafe"):
        RUNNER["unpack"](archive, tmp_path / "restored")
