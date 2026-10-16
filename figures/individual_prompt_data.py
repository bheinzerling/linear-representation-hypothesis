from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/numerical_sufficiency_atlas/individual_prompt_fits"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["cohort"] == "canonical_dev"
    assert manifest["fitting_values"] == 800 and manifest["evaluation_source_prompts"] == 100
    for name, sha in manifest["files"].items():
        assert digest(DATA / name) == sha, name
    predictions = pd.read_csv(DATA / "decoding_predictions.csv")
    components = pd.read_csv(DATA / "component_predictions.csv")
    support = pd.read_csv(DATA / "code_support.csv")
    metrics = pd.read_csv(DATA / "decoding_metrics.csv").set_index("architecture")
    steering = pd.read_csv(DATA / "steering_summary.csv").set_index("method")
    rows = pd.read_csv(DATA / "steering_rows.csv.gz")
    assert len(predictions) == len(components) == 100
    assert predictions["value"].is_unique and predictions["source_prompt_id"].is_unique
    assert set(predictions["value"]) == set(components["value"])
    assert len(support) == 1000 and support["value"].tolist() == list(range(1000))
    np.testing.assert_allclose(support["target_sin_4"], np.sin(np.arange(1000)*1000**(-4/5)), atol=1e-12)
    np.testing.assert_allclose(support["target_sin_3"], np.sin(np.arange(1000)*1000**(-3/5)), atol=1e-12)
    for architecture, label in (("linear", "ridge"), ("circular_reg", "circular")):
        fit_id = manifest["fit_ids"]["individual_" + architecture]
        assert set(predictions[label + "_fit_id"]) == {fit_id}
        assert metrics.loc[architecture, "fit_id"] == fit_id
        np.testing.assert_allclose((predictions[label + "_prediction"]-predictions["value"]).abs().mean(),
                                   metrics.loc[architecture, "mae"])
        assert steering.loc[architecture, "fit_id"] == fit_id
    assert set(components["fit_id"]) == {manifest["fit_ids"]["individual_circular_reg"]}
    reference = None
    for method, group in rows.groupby("method", sort=False):
        assert len(group) == 2000 and group["source_prompt_id"].nunique() == 100
        keys = group[["case_id", "source_prompt_id", "source_value", "target_value", "shifted_target", "shifted_target_token_id"]].reset_index(drop=True)
        if reference is None:
            reference = keys
        else:
            pd.testing.assert_frame_equal(keys, reference)
        assert set(group["source_value"]) == set(predictions["value"])
        assert set(group["fit_id"]) == {steering.loc[method, "fit_id"]}
        for metric in ("exact", "within10"):
            expected = (group["predicted_token_id"] == group["shifted_target_token_id"] if metric == "exact" else
                        (group["predicted_value"]-group["shifted_target"]).abs() <= 10)
            np.testing.assert_array_equal(group[metric], expected)
            assert expected.sum() == steering.loc[method, metric + "_count"]
            np.testing.assert_allclose(expected.mean(), steering.loc[method, metric + "_accuracy"])
            if expected.sum() == 0:
                assert pd.isna(steering.loc[method, metric + "_low"])
                assert pd.isna(steering.loc[method, metric + "_high"])
    assert steering.loc[["linear", "circular_reg"], "exact_count"].eq(0).all()
    assert steering.loc["centroid_patch", "exact_count"] == 1995
    frame = predictions.merge(components[["value", "circular_sin_4", "circular_sin_3"]], on="value", validate="one_to_one")
    control = steering.rename(columns={"numeric_output_mae": "direct_steering_mae"}).copy()
    control["argmax_accuracy"] = control["exact_accuracy"]
    return {"test": frame, "outputs": support, "metrics": metrics.rename(columns={"mae": "mean_abs_error"}),
            "control": control, "matched_fit_manifest": manifest}


if __name__ == "__main__":
    evidence = load()
    print(json.dumps({"status": "passed", "source_prompts": len(evidence["test"]),
                      "fit_ids": evidence["matched_fit_manifest"]["fit_ids"],
                      "exact_counts": evidence["control"]["exact_count"].to_dict()}, indent=2))
