#!/usr/bin/env python3
"""Check complete matched-probe outputs against frozen inputs and probe payloads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.figure7_matched_probes import content_sha256, file_sha256, validate_partitions
from src.utils import atomic_write_json


def validate(inputs: Path, run: Path) -> dict:
    manifest = json.loads((inputs / "manifest.json").read_text())
    validate_partitions(manifest["split"])
    for name, sha in manifest["files"].items():
        assert file_sha256(inputs / name) == sha, name
    completion = json.loads((run / "completion.json").read_text())
    assert completion["status"] == "passed" and (run / "DONE").read_text().strip() == completion["identity"]
    inventory = json.loads((run / "output-inventory.json").read_text())
    checked = 0
    for name, sha in inventory.items():
        if (run / name).exists():
            assert file_sha256(run / name) == sha, name
            checked += 1
    fits = json.loads((run / "fits.json").read_text())["fits"]
    assert len(fits) == 4
    fit_ids = {}
    for fit in fits:
        payload = json.loads((run / f'probe_{fit["fit_name"]}.json').read_text())
        assert content_sha256(payload) == fit["fit_id"]
        assert fit["training_rows"] == fit["training_values"] == 800
        fit_ids[fit["fit_name"]] = fit["fit_id"]
    decoding = pd.read_csv(run / "decoding_summary.csv")
    detail = pd.read_csv(run / "decoding_predictions.csv")
    assert len(decoding) == completion["decoding_conditions"] == 36
    for row in decoding.to_dict("records"):
        select = np.ones(len(detail), dtype=bool)
        for key in ("fit_name", "evaluation_bank", "partition"):
            select &= detail[key].eq(row[key]).to_numpy()
        d = detail.loc[select]
        assert len(d) == row["n_rows"] and set(d["fit_id"]) == {fit_ids[row["fit_name"]]}
        assert np.isclose(np.mean(np.abs(d["probe_prediction"] - d["source_value"])), row["mae"])
        assert np.isclose(np.mean(d["probe_prediction"] == d["source_value"]), row["exact_accuracy"])
    predictions = pd.read_csv(run / "steering_predictions.csv", keep_default_na=False)
    summary = pd.read_csv(run / "steering_summary.csv")
    assert len(summary) == completion["steering_conditions"] == 36
    assert len(predictions) == completion["steering_rows"] == 72000
    for bank in ("rhs_test", "canonical_test", "canonical_dev"):
        cases = pd.read_json(inputs / f"cases_{bank}.jsonl", lines=True)
        assert cases["case_id"].nunique() == len(cases) == 2000
        assert cases["source_prompt_id"].nunique() == 100
        part = "dev" if bank == "canonical_dev" else "test"
        assert set(cases["source_value"]) == set(manifest["split"][part])
        reference = cases.set_index("case_id")
        groups = predictions.loc[predictions["evaluation_bank"].eq(bank)].groupby(["fit_name", "method"], sort=False)
        assert len(groups) == 12
        for (fit_name, method), group in groups:
            assert len(group) == 2000 and set(group["case_id"]) == set(reference.index)
            expected = reference.loc[group["case_id"]]
            for col in ("prompt", "shifted_prompt", "source_value", "target_value", "shifted_target", "shifted_target_token_id", "token_idx", "shifted_token_idx", "source_prompt_id"):
                assert group[col].tolist() == expected[col].tolist(), (bank, fit_name, method, col)
            assert np.array_equal(group["exact"], group["predicted_token_id"].to_numpy() == group["shifted_target_token_id"].to_numpy())
            numeric = pd.to_numeric(group["predicted_value"], errors="coerce")
            assert np.array_equal(group["within10"], (np.abs(numeric - group["shifted_target"]) <= 10).fillna(False))
            row = summary.loc[summary["evaluation_bank"].eq(bank) & summary["fit_name"].eq(fit_name) & summary["method"].eq(method)].iloc[0]
            assert row["n_edits"] == 2000 and row["n_source_prompts"] == 100
            assert row["exact_count"] == group["exact"].sum()
            assert np.isclose(row["exact_accuracy"], group["exact"].mean())
            assert np.isclose(row["within10_accuracy"], group["within10"].mean())
            if fit_name != "control":
                assert set(group["fit_id"]) == {fit_ids[fit_name]}
                attained = group["post_edit_probe_target"].astype(str).str.lower().isin(["true", "1"])
                assert np.array_equal(attained,
                                      pd.to_numeric(group["post_edit_probe_prediction"]) == group["target_value"])
                assert np.isclose(row["post_edit_probe_target_accuracy"], attained.mean())
    case_decoding = pd.read_csv(run / "steering_case_decoding.csv")
    assert len(case_decoding) == 24000
    for fit_name, group in case_decoding.groupby("fit_name"):
        assert set(group["fit_id"]) == {fit_ids[fit_name]}
    return {"status": "passed", "identity": completion["identity"], "fits": 4,
            "decoding_conditions": 36, "steering_conditions": 36, "steering_rows": 72000,
            "present_output_hashes_checked": checked,
            "checks": ["input hashes and value partitions", "saved fit identities", "shared case sets and prompt bytes",
                       "counterfactual targets and intervention sites", "independent exact and lenient scoring",
                       "summary counts and rates", "post-edit probe target attainment"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.inputs, args.run_dir)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
