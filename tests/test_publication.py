from __future__ import annotations

import runpy
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT / "scripts/verify_publication.py"))["verify_source_archive"]


@pytest.mark.parametrize("change", [None, "stale_paper", "missing_runner", "untracked_source"])
def test_release_source_must_match_the_published_commit(tmp_path, monkeypatch, change):
    files = {
        "paper/paper.tex": "corrected manuscript\n",
        "docs/model-experiments.md": "portable reproduction\n",
        "scripts/verify.py": "verification\n",
        "experiments/numerical/scripts/prepare_figure7_matched_inputs.py": "preparation\n",
        "experiments/numerical/scripts/run_figure7_matched_probes.py": "execution\n",
        "experiments/numerical/src/figure7_historical_comparison.py": "optional history\n",
    }
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Release test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "fixture"], cwd=tmp_path, check=True,
    )
    if change == "stale_paper":
        files["paper/paper.tex"] = "old manuscript\n"
    elif change == "missing_runner":
        files.pop("experiments/numerical/scripts/run_figure7_matched_probes.py")
    elif change == "untracked_source":
        files["scripts/uncommitted.py"] = "unpublished change\n"
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, content in files.items():
            handle.writestr("linear-representation-hypothesis/" + name, content)
    monkeypatch.setitem(VERIFY.__globals__, "ROOT", tmp_path)
    if change is None:
        assert VERIFY(archive, "HEAD") == 6
    else:
        with pytest.raises(ValueError, match="differs from public commit|sources absent"):
            VERIFY(archive, "HEAD")
