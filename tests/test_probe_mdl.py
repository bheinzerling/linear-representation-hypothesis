from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lrh.analysis.probe_mdl import calculate, table

DATA = Path(__file__).resolve().parents[1] / "data/probe_mdl"


def refresh_hash(data: Path, name: str) -> None:
    path = data / name
    manifest = json.loads((data / "input_snapshot.json").read_text())
    manifest["files"][name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    (data / "input_snapshot.json").write_text(json.dumps(manifest))


def test_per_example_costs_reconstruct_all_released_aggregations():
    metrics, blocks, summary = calculate(DATA)
    assert len(metrics) == 438 and len(summary) == 146
    assert blocks.groupby(["order", "setting_id", "coder"]).block_size.sum().eq(800).all()
    assert summary.groupby("coder").size().eq(73).all()
    assert table(summary).count(r"\,$\pm$\,") == 10
    costs = summary.groupby(["architecture", "coder"]).bits_per_value.min()
    assert costs["circular_reg", "ordinal"] < costs["circular_cls", "ordinal"]
    assert costs["circular_cls", "native"] < costs["circular_reg", "native"]


def test_modified_codelength_cannot_pass_with_a_refreshed_hash(tmp_path):
    shutil.copytree(DATA, tmp_path / "data")
    data = tmp_path / "data"
    with np.load(data / "observations.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["bits"][0, 0, 0, 10] += .01
    np.savez_compressed(data / "observations.npz", **arrays)
    refresh_hash(data, "observations.npz")
    with pytest.raises(AssertionError):
        calculate(data)


def test_held_out_labels_cannot_enter_the_online_message(tmp_path):
    shutil.copytree(DATA, tmp_path / "data")
    data = tmp_path / "data"
    with np.load(data / "observations.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["ordered_values"][0, 0] = arrays["test_values"][0]
    np.savez_compressed(data / "observations.npz", **arrays)
    refresh_hash(data, "observations.npz")
    with pytest.raises(ValueError, match="overlap"):
        calculate(data)


def test_stale_summary_is_detected_even_with_a_refreshed_hash(tmp_path):
    shutil.copytree(DATA, tmp_path / "data")
    data = tmp_path / "data"
    summary = pd.read_csv(data / "summary.csv")
    summary.loc[0, "mean_bits"] += 1
    summary.to_csv(data / "summary.csv", index=False)
    refresh_hash(data, "summary.csv")
    with pytest.raises(AssertionError):
        calculate(data)
