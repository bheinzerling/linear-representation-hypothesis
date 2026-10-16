from __future__ import annotations

import io
import json
import runpy
import shutil
import tarfile
from pathlib import Path

import pytest

from lrh.io import sha256, write_json
from lrh.pipeline import is_complete, plan_run

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT / "scripts/verify.py"))
PREPARE = runpy.run_path(str(ROOT / "scripts/prepare_model_inputs.py"))


def test_repository_inputs_and_model_imports():
    assert VERIFY["verify_manifest"]() > 100
    assert VERIFY["verify_model_imports"]() > 50
    assert VERIFY["verify_paper_tables"]() == 376


def test_paper_table_check_detects_a_changed_rating(tmp_path):
    (tmp_path / "paper").mkdir()
    (tmp_path / "data/literature").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "data/literature/evidence_ratings.csv",
        tmp_path / "data/literature/evidence_ratings.csv",
    )
    source = (ROOT / "paper/paper.tex").read_text()
    line = next(
        line
        for line in source.splitlines()
        if r"\citet{jobanputra2025subtract}" in line and r"\EvSupport{1}" in line
    )
    changed = source.replace(line, line.replace(r"\EvSupport{1}", r"\EvSupport{3}"), 1)
    (tmp_path / "paper/paper.tex").write_text(changed)
    with pytest.raises(ValueError, match="ratings differ"):
        VERIFY["verify_paper_tables"](tmp_path)


def test_manifest_detects_modified_and_unlisted_inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "observations.csv"
    source.write_text("target,prediction\n1,1\n")
    (data / "MANIFEST.sha256").write_text(f"{sha256(source)}  data/observations.csv\n")
    assert VERIFY["verify_manifest"](tmp_path) == 1
    (data / "unlisted.csv").write_text("2,2\n")
    with pytest.raises(ValueError, match="Unlisted"):
        VERIFY["verify_manifest"](tmp_path)
    (data / "unlisted.csv").unlink()
    source.write_text("target,prediction\n1,0\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        VERIFY["verify_manifest"](tmp_path)


def test_run_identity_and_completion_require_matching_outputs(tmp_path):
    (tmp_path / "data").mkdir()
    data = tmp_path / "data/results.csv"
    data.write_text("value\n1\n")
    output = tmp_path / "out"
    first = plan_run(tmp_path, output)
    assert first.run_id == plan_run(tmp_path, output).run_id
    assert not is_complete(first)
    output.mkdir()
    metric = output / "metrics.json"
    write_json(metric, {"accuracy": 1})
    write_json(
        output / "run.json", {"run_id": first.run_id, "outputs": {"metrics.json": sha256(metric)}}
    )
    assert not is_complete(first)
    (output / "DONE").write_text(first.run_id)
    assert is_complete(first)
    metric.write_text("{}")
    assert not is_complete(first)
    data.write_text("value\n2\n")
    assert first.run_id != plan_run(tmp_path, output).run_id


def test_model_inputs_verify_payload_and_preserve_different_existing_files(tmp_path):
    archive = tmp_path / "inputs.tar.gz"
    payload = b'{"values":[1,2,3]}\n'
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("numerical/data/prompts.json")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    target = tmp_path / "experiments"
    payload_file = tmp_path / "payload.json"
    payload_file.write_bytes(payload)
    manifest = {
        "sha256": sha256(archive),
        "files": {"numerical/data/prompts.json": {"sha256": sha256(payload_file)}},
    }
    assert PREPARE["extract"](archive, target, manifest) == 1
    extracted = target / "numerical/data/prompts.json"
    assert json.loads(extracted.read_text()) == {"values": [1, 2, 3]}
    extracted.write_text('{"values":[4]}')
    with pytest.raises(FileExistsError):
        PREPARE["extract"](archive, target, manifest)
