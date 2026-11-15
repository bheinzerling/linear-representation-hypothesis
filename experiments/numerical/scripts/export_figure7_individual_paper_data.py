#!/usr/bin/env python3
"""Export a compact, matched individual-prompt packet for Figure 7a--c."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.figure7_matched_probes import cluster_accuracy_interval, content_sha256, file_sha256, numeric_metrics
from src.magnitude_intervention import probe_from_payload
from src.utils import atomic_write_csv, atomic_write_json


def export(artifact: Path, output: Path):
    run = artifact / "run"
    verification = json.loads((artifact / "completion-validation.json").read_text())
    assert verification["status"] == "passed" and verification["outputs_validated"]
    inventory = json.loads((run / "output-inventory.json").read_text())
    source_path = run / "source_canonical_dev.npy"
    assert file_sha256(source_path) == inventory[source_path.name]
    source = np.load(source_path)
    input_manifest = json.loads((artifact / "inputs/manifest.json").read_text())
    case_path = artifact / "inputs/cases_canonical_dev.jsonl"
    assert file_sha256(case_path) == input_manifest["files"][case_path.name]
    cases = pd.read_json(case_path, lines=True)
    first = cases.drop_duplicates("source_prompt_id")
    assert len(first) == 100 and first["target_slot"].eq(0).all()
    source = source[first.index.to_numpy()]
    values = first["source_value"].to_numpy()
    fits, sources, metrics = {}, {}, []
    predictions = pd.DataFrame({"source_prompt_id": first["source_prompt_id"].to_numpy(), "value": values})
    output.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for architecture, label in (("linear", "ridge"), ("circular_reg", "circular")):
        name = "individual_" + architecture
        path = run / f"probe_{name}.json"
        payload = json.loads(path.read_text())
        probe = probe_from_payload(payload)
        fit_id = content_sha256(payload)
        payloads[name] = probe
        fits[name] = fit_id
        sources[path.name] = file_sha256(path)
        decoded = probe.predict_values(source)
        recorded = pd.read_csv(artifact / "panel_a_individual_dev_steering_sources.csv")
        recorded = recorded.loc[recorded["fit_name"].eq(name)].drop_duplicates("source_prompt_id")
        assert recorded["source_prompt_id"].tolist() == first["source_prompt_id"].tolist()
        assert set(recorded["fit_id"]) == {fit_id}
        np.testing.assert_array_equal(decoded, recorded["probe_prediction"])
        predictions[label + "_prediction"] = decoded
        predictions[label + "_fit_id"] = fit_id
        if architecture == "linear":
            predictions["ridge_raw_prediction"] = probe.predict_features(source)[:, 0]
        metrics.append({"architecture": architecture, "fit_id": fit_id, "n": 100,
                        **numeric_metrics(values, decoded)})
    atomic_write_csv(output / "decoding_predictions.csv", predictions.sort_values("value"))
    atomic_write_csv(output / "decoding_metrics.csv", pd.DataFrame(metrics))
    circular = payloads["individual_circular_reg"]
    features = circular.predict_features(source)
    components = pd.DataFrame({"value": values, "source_prompt_id": first["source_prompt_id"].to_numpy(),
                               "circular_sin_4": features[:, 8], "circular_sin_3": features[:, 6],
                               "fit_id": fits["individual_circular_reg"]})
    atomic_write_csv(output / "component_predictions.csv", components.sort_values("value"))
    support = circular.encode_values(np.arange(1000))
    atomic_write_csv(output / "code_support.csv", pd.DataFrame({"value": np.arange(1000),
                     "target_sin_4": support[:, 8], "target_sin_3": support[:, 6]}))
    summaries, snapshots = [], []
    methods = (("linear", "individual_linear", "label_difference"),
               ("circular_reg", "individual_circular_reg", "label_difference"),
               ("centroid_patch", "control", "pca64_centroid_difference"),
               ("no_edit", "control", "no_edit"),
               ("full_centroid_shift", "control", "centroid_difference"),
               ("counterfactual_patch", "control", "counterfactual_patch"))
    for method, fit_name, rule in methods:
        path = run / f"behavior_canonical_dev_{fit_name}_{rule}.csv"
        assert file_sha256(path) == inventory[path.name]
        sources[path.name] = file_sha256(path)
        rows = pd.read_csv(path)
        assert len(rows) == 2000 and rows["case_id"].tolist() == cases["case_id"].tolist()
        assert rows["source_prompt_id"].nunique() == 100
        assert rows["evaluation_bank"].eq("canonical_dev").all()
        assert rows["shifted_target"].tolist() == cases["shifted_target"].tolist()
        np.testing.assert_array_equal(rows["exact"], rows["predicted_token_id"] == rows["shifted_target_token_id"])
        np.testing.assert_array_equal(rows["within10"], (rows["predicted_value"] - rows["shifted_target"]).abs() <= 10)
        if fit_name != "control":
            assert set(rows["fit_id"]) == {fits[fit_name]}
        record = {"method": method, "fit_id": fits.get(fit_name, "control"), "n": len(rows),
                  "source_prompts": 100, "numeric_output_mae": (rows["predicted_value"]-rows["shifted_target"]).abs().mean()}
        for column in ("exact", "within10"):
            low, high = cluster_accuracy_interval(rows, column, seed=1729)
            record.update({column + "_count": int(rows[column].sum()), column + "_accuracy": rows[column].mean(),
                           column + "_low": low, column + "_high": high})
        summaries.append(record)
        snapshot = rows[["case_id", "source_prompt_id", "source_value", "target_value", "shifted_target",
                         "predicted_token_id", "shifted_target_token_id", "predicted_value", "exact", "within10", "fit_id"]].copy()
        snapshot.insert(0, "method", method)
        snapshots.append(snapshot)
    atomic_write_csv(output / "steering_summary.csv", pd.DataFrame(summaries))
    pd.concat(snapshots, ignore_index=True).to_csv(output / "steering_rows.csv.gz", index=False,
                                                compression={"method": "gzip", "mtime": 0})
    sources[source_path.name] = file_sha256(source_path)
    sources[case_path.name] = file_sha256(case_path)
    manifest = {"experiment": "2026-10-15-figure7-matched-probes", "run_identity": json.loads((run/"completion.json").read_text())["identity"],
                "cohort": "canonical_dev", "fitting_values": 800, "evaluation_source_prompts": 100,
                "target_replacements_per_prompt": 20, "evaluation_values": "Development; previously exposed replication cohort",
                "prediction_sampling": "First frozen case (target slot 0) per source prompt; individual activations, no averaging",
                "model": "Llama-3.1-8B", "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                "framework_layer": 10, "decoder_block": 9, "dtype": "float16", "attention": "eager",
                "components": {"C1": {"column": 8, "function": "sin(v * 1000**(-4/5))"},
                               "C3": {"column": 6, "function": "sin(v * 1000**(-3/5))"}},
                "fit_ids": fits, "source_sha256": sources, "bootstrap": {"clusters":100,"replicates":10000,"seed":1729},
                "script_sha256": file_sha256(__file__), "validated_result_sha256": file_sha256(artifact/"completion-validation.json"),
                "files": {p.name:file_sha256(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}}
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "files": len(manifest["files"]), "fits": fits,
                      "decoding": metrics, "steering": [{"method": r["method"], "exact_count":r["exact_count"]} for r in summaries]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT/"experiment-reports/artifacts/2026-10-15-figure7-matched-probes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.artifact_dir, args.output or args.artifact_dir/"paper_individual_dev")
