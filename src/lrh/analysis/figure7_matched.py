"""Verify the individual-prompt fits and paired interventions in Figure 7a--c."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = (
    "linear", "circular_reg", "centroid_patch", "no_edit",
    "full_centroid_shift", "counterfactual_patch",
    "linear_label_difference", "circular_reg_label_difference",
)
PROBE_METHODS = {
    "linear": ("individual_linear", "prediction_corrected"),
    "circular_reg": ("individual_circular_reg", "prediction_corrected"),
    "linear_label_difference": ("individual_linear", "label_difference"),
    "circular_reg_label_difference": ("individual_circular_reg", "label_difference"),
}
FILES = (
    "manifest.json", "decoding_predictions.csv", "decoding_metrics.csv",
    "component_predictions.csv", "code_support.csv", "steering_rows.csv.gz",
    "steering_summary.csv",
)
PAIR_KEYS = (
    "case_id", "source_prompt_id", "source_value", "target_value",
    "shifted_target", "shifted_target_token_id",
)


def calculate(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if (manifest["cohort"] != "canonical_dev"
            or manifest["fitting_values"] != 800
            or manifest["evaluation_source_prompts"] != 100
            or manifest["target_replacements_per_prompt"] != 20):
        raise ValueError("Figure 7 must use the matched individual-prompt development cohort")
    if set(manifest["files"]) != set(FILES) - {"manifest.json"}:
        raise ValueError("Incomplete Figure 7 observation inventory")
    validation = manifest.get("intervention_validation", {})
    if (validation.get("primary_edit_rule") != "prediction_corrected"
            or validation.get("comparison_edit_rule") != "label_difference"
            or validation.get("application_dtype") != "float16"):
        raise ValueError("Figure 7 must validate prediction-corrected edits after FP16 application")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed Figure 7 input: {name}")
    predictions = pd.read_csv(data_dir / "decoding_predictions.csv")
    components = pd.read_csv(data_dir / "component_predictions.csv")
    support = pd.read_csv(data_dir / "code_support.csv")
    metrics = pd.read_csv(data_dir / "decoding_metrics.csv").set_index("architecture")
    certificate = pd.read_csv(data_dir / "steering_summary.csv").set_index("method")
    rows = pd.read_csv(data_dir / "steering_rows.csv.gz")
    if (len(predictions) != 100 or not predictions.value.is_unique
            or not predictions.source_prompt_id.is_unique or len(components) != 100):
        raise ValueError("Figure 7 decoding requires 100 distinct source prompts")
    test = predictions.merge(
        components, on=["value", "source_prompt_id"], validate="one_to_one",
    )
    if len(test) != 100:
        raise ValueError("Figure 7 components and numerical predictions use different prompts")
    np.testing.assert_array_equal(support.value, np.arange(1000))
    for name, exponent in (("target_sin_4", -4 / 5), ("target_sin_3", -3 / 5)):
        np.testing.assert_allclose(support[name], np.sin(np.arange(1000) * 1000**exponent),
                                   atol=1e-12, rtol=0)
    for architecture, label in (("linear", "ridge"), ("circular_reg", "circular")):
        fit_id = manifest["fit_ids"]["individual_" + architecture]
        if (set(predictions[label + "_fit_id"]) != {fit_id}
                or metrics.loc[architecture, "fit_id"] != fit_id
                or certificate.loc[architecture, "fit_id"] != fit_id):
            raise ValueError("Figure 7 decoding and steering must share the same fits")
        errors = predictions[label + "_prediction"] - predictions.value
        computed = {
            "n": len(errors), "mae": errors.abs().mean(),
            "rmse": np.sqrt(np.square(errors).mean()),
            "r2": 1 - np.square(errors).mean() / np.var(predictions.value),
            "exact_accuracy": errors.eq(0).mean(),
        }
        for name, value in computed.items():
            np.testing.assert_allclose(value, metrics.loc[architecture, name], rtol=0, atol=1e-12)
    if set(components.fit_id) != {manifest["fit_ids"]["individual_circular_reg"]}:
        raise ValueError("Figure 7 components come from another fitted probe")
    if set(rows.method) != set(METHODS) or set(certificate.index) != set(METHODS):
        raise ValueError("Figure 7 requires both probe edit rules and all four controls")

    protocol = manifest["bootstrap"]
    if protocol["clusters"] != 100:
        raise ValueError("Figure 7 bootstrap must cluster all twenty edits per source prompt")
    indices = np.random.default_rng(protocol["seed"]).integers(
        0, 100, size=(protocol["replicates"], 100),
    )
    reference, records, draws, probe_validation = None, [], {}, []
    for method in METHODS:
        group = rows.loc[rows.method.eq(method)].sort_values("case_id").reset_index(drop=True)
        if (len(group) != 2000 or not group.case_id.is_unique
                or group.source_prompt_id.nunique() != 100
                or not group.groupby("source_prompt_id").size().eq(20).all()):
            raise ValueError(f"Unpaired Figure 7 interventions: {method}")
        if reference is None:
            reference = group[list(PAIR_KEYS)]
        else:
            pd.testing.assert_frame_equal(group[list(PAIR_KEYS)], reference)
        source_keys = group[["source_prompt_id", "source_value"]].drop_duplicates()
        pd.testing.assert_frame_equal(
            source_keys.sort_values("source_value").reset_index(drop=True),
            predictions[["source_prompt_id", "value"]].rename(columns={"value": "source_value"})
            .sort_values("source_value").reset_index(drop=True),
        )
        if set(group.fit_id) != {certificate.loc[method, "fit_id"]}:
            raise ValueError(f"Figure 7 steering fit changed: {method}")
        expected_rule = validation["methods"][method]["edit_rule"]
        if not group.edit_rule.eq(expected_rule).all() or certificate.loc[method, "edit_rule"] != expected_rule:
            raise ValueError(f"Figure 7 intervention rule changed: {method}")
        np.testing.assert_allclose(group.applied_edit_norm.mean(),
                                   certificate.loc[method, "mean_edit_norm"], rtol=0, atol=1e-12)
        if method in PROBE_METHODS:
            fit_name, rule = PROBE_METHODS[method]
            if expected_rule != rule or set(group.fit_id) != {manifest["fit_ids"][fit_name]}:
                raise ValueError("Figure 7 decoding and steering must share the same fits and edit rules")
            attained = group.post_edit_probe_prediction.eq(group.target_value)
            np.testing.assert_array_equal(group.post_edit_probe_target, attained)
            if (not np.isfinite(group.post_edit_code_mse).all()
                    or group.post_edit_code_mse.lt(0).any()):
                raise ValueError(f"Invalid Figure 7 post-edit code errors: {method}")
            count = int(attained.sum())
            if rule == "prediction_corrected" and count != len(group):
                raise ValueError(f"Figure 7 corrected edits must reach every probe target: {method}")
            code_mse = group.post_edit_code_mse.mean()
            np.testing.assert_allclose(
                [count, count / len(group), code_mse],
                certificate.loc[method, ["post_edit_probe_target_count",
                    "post_edit_probe_target_accuracy", "post_edit_code_mse"]].to_numpy(float),
                rtol=0, atol=1e-12,
            )
            probe_validation.append({
                "method": method, "edit_rule": rule, "fit_id": manifest["fit_ids"][fit_name],
                "n": len(group), "probe_target_count": count,
                "probe_target_accuracy": count / len(group), "post_edit_code_mse": code_mse,
                "mean_edit_norm": group.applied_edit_norm.mean(),
                "correct_sum_count": int(group.exact.sum()),
            })
        np.testing.assert_allclose(certificate.loc[method, ["n", "source_prompts"]]
                                   .to_numpy(float), [2000, 100])
        errors = (group.predicted_value - group.shifted_target).abs()
        np.testing.assert_allclose(errors.mean(), certificate.loc[method, "numeric_output_mae"],
                                   rtol=0, atol=1e-12)
        for metric, successes in (
            ("exact", group.predicted_token_id.eq(group.shifted_target_token_id)),
            ("within10", errors.le(10)),
        ):
            np.testing.assert_array_equal(group[metric], successes)
            clusters = successes.groupby(group.source_prompt_id, sort=True).mean().to_numpy()
            samples = clusters[indices].mean(axis=1)
            draws[method, metric] = samples
            count = int(successes.sum())
            low, high = np.quantile(samples, [0.025, 0.975]) if count else (np.nan, np.nan)
            np.testing.assert_allclose(
                [count, count / len(group), low, high],
                certificate.loc[method, [metric + "_count", metric + "_accuracy",
                                          metric + "_low", metric + "_high"]].to_numpy(float),
                rtol=0, atol=1e-12, equal_nan=True,
            )
            records.append({
                "method": method, "metric": metric, "estimate": count / len(group),
                "count": count, "n": len(group), "low": low, "high": high,
                "clusters": 100, "targets_per_cluster": 20,
                "replicates": protocol["replicates"],
                "status": ("conditional_source_cluster_bootstrap" if count else
                           "zero_successes_interval_unavailable"),
            })
    for record in records:
        baseline = next(r for r in records if r["method"] == "no_edit"
                        and r["metric"] == record["metric"])
        paired = draws[record["method"], record["metric"]] - draws["no_edit", record["metric"]]
        record["change_from_no_edit"] = record["estimate"] - baseline["estimate"]
        record["change_low"], record["change_high"] = (
            np.quantile(paired, [0.025, 0.975]) if np.ptp(paired) else (np.nan, np.nan)
        )
    return {"manifest": manifest, "test": test, "support": support,
            "decoding": metrics.reset_index(), "steering": pd.DataFrame(records),
            "probe_validation": pd.DataFrame(probe_validation)}


def verify_manuscript(text: str, result: dict) -> None:
    """Bind the release observations to the current appendix's reported results."""
    text = " ".join(text.split())
    decoding = re.search(
        r"Decoding MAE is ([\d.]+) for scalar ridge and ([\d.]+) for circular regression", text,
    )
    controls = re.search(
        r"The PCA shift reaches ([\d.]+)\\% exact accuracy, the full centroid shift "
        r"([\d.]+)\\%, and replacement with the matched counterfactual activation ([\d.]+)\\%", text,
    )
    if decoding is None or controls is None:
        raise ValueError("Cannot find the Figure 7 decoding and control results in the manuscript")
    mae = result["decoding"].set_index("architecture").loc[["linear", "circular_reg"], "mae"]
    steering = result["steering"].query("metric == 'exact'").set_index("method")
    expected = 100 * steering.loc[
        ["centroid_patch", "full_centroid_shift", "counterfactual_patch"], "estimate",
    ].to_numpy()
    try:
        np.testing.assert_allclose([float(v) for v in decoding.groups()], mae, rtol=0, atol=0.005)
        np.testing.assert_allclose([float(v) for v in controls.groups()], expected, rtol=0, atol=0.005)
    except AssertionError as exc:
        raise ValueError("Released Figure 7 results differ from the manuscript") from exc
    seed = result["manifest"]["bootstrap"]["seed"]
    if not re.search(rf"resamples of the 100 source prompts \(seed {seed}\)", text):
        raise ValueError("Figure 7 bootstrap protocol differs from the manuscript")
    probes = result["probe_validation"].set_index("method")
    corrected = probes.loc[["linear", "circular_reg"]]
    if not corrected.probe_target_count.eq(2000).all() or not corrected.correct_sum_count.eq(0).all():
        raise ValueError("Figure 7 corrected probe-target and behavioral results differ from the manuscript")
    formula = r"\Delta h=\bigl[\phi(t)-z(h_s)\bigr]W^{+}"
    targets = re.search(
        r"\bboth probes decode the requested operand in all ([\d,]+) "
        r"(?:edited activations|cases)\b", text, re.IGNORECASE,
    )
    if re.sub(r"\s+", "", formula) not in re.sub(r"\s+", "", text) or targets is None:
        raise ValueError("Figure 7 must report the corrected edit and its verified probe targets")
    if not corrected.probe_target_count.eq(int(targets.group(1).replace(",", ""))).all():
        raise ValueError("Figure 7 reported probe-target count differs from the observations")
    comparison = re.search(
        r"These label-difference edits reach the requested value under the probe in "
        r"(\d+)/2,000 scalar cases and (\d+)/2,000 circular cases", text,
    )
    if comparison is None:
        raise ValueError("Figure 7 label-difference target validation is missing")
    expected = probes.loc[["linear_label_difference", "circular_reg_label_difference"], "probe_target_count"]
    if [int(v) for v in comparison.groups()] != expected.tolist():
        raise ValueError("Figure 7 label-difference target validation differs from the manuscript")
