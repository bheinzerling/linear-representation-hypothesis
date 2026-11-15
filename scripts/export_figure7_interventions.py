"""Export both Figure 7 edit rules from a completed, hash-verified model run.

The existing packet supplies the fixed decoding observations and fit identities.
This CPU-only export retains post-edit probe predictions measured by the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {
    "linear": ("individual_linear", "prediction_corrected"),
    "circular_reg": ("individual_circular_reg", "prediction_corrected"),
    "centroid_patch": ("control", "pca64_centroid_difference"),
    "no_edit": ("control", "no_edit"),
    "full_centroid_shift": ("control", "centroid_difference"),
    "counterfactual_patch": ("control", "counterfactual_patch"),
    "linear_label_difference": ("individual_linear", "label_difference"),
    "circular_reg_label_difference": ("individual_circular_reg", "label_difference"),
}
PAIR_KEYS = [
    "case_id", "source_prompt_id", "source_value", "target_value",
    "shifted_target", "shifted_target_token_id",
]
BEHAVIOR_COLUMNS = [
    "predicted_token_id", "predicted_value", "exact", "within10", "fit_id",
    "applied_edit_norm",
]
PROBE_COLUMNS = [
    "post_edit_probe_prediction", "post_edit_probe_target", "post_edit_code_mse",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run_dir: Path, packet_dir: Path, release_dir: Path | None = None) -> None:
    manifest = json.loads((packet_dir / "manifest.json").read_text())
    inventory = json.loads((run_dir / "output-inventory.json").read_text())
    completion = json.loads((run_dir / "completion.json").read_text())
    if completion["status"] != "passed" or completion["identity"] != manifest["run_identity"]:
        raise ValueError("Interventions and decoding packet come from different runs")
    for name, expected in manifest["files"].items():
        if digest(packet_dir / name) != expected:
            raise ValueError(f"Changed decoding packet input: {name}")
    for fit_name, fit_id in manifest["fit_ids"].items():
        path = run_dir / f"probe_{fit_name}.json"
        payload = json.loads(path.read_text())
        actual_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest(path) != manifest["source_sha256"][path.name] or actual_id != fit_id:
            raise ValueError(f"Changed fitted probe: {fit_name}")

    predictions = pd.read_csv(packet_dir / "decoding_predictions.csv")
    reference, snapshots, summaries = None, [], []
    protocol = manifest["bootstrap"]
    indices = np.random.default_rng(protocol["seed"]).integers(
        0, protocol["clusters"], size=(protocol["replicates"], protocol["clusters"]),
    )
    for method, (fit_name, rule) in METHODS.items():
        path = run_dir / f"behavior_{manifest['cohort']}_{fit_name}_{rule}.csv"
        source_hash = digest(path)
        if source_hash != inventory[path.name]:
            raise ValueError(f"Changed model-run observations: {path.name}")
        manifest["source_sha256"][path.name] = source_hash
        rows = pd.read_csv(path)
        if len(rows) != 2000 or not rows.source_prompt_id.nunique() == 100:
            raise ValueError(f"Incomplete intervention condition: {method}")
        if not rows.evaluation_bank.eq(manifest["cohort"]).all():
            raise ValueError(f"Changed intervention cohort: {method}")
        keys = rows[PAIR_KEYS].sort_values("case_id").reset_index(drop=True)
        if reference is None:
            reference = keys
        else:
            pd.testing.assert_frame_equal(keys, reference)
        if set(rows.source_value) != set(predictions.value):
            raise ValueError(f"Different decoding and intervention sources: {method}")
        fit_id = manifest["fit_ids"].get(fit_name, "control")
        if set(rows.fit_id) != {fit_id}:
            raise ValueError(f"Different decoding and intervention fits: {method}")
        errors = (rows.predicted_value - rows.shifted_target).abs()
        np.testing.assert_array_equal(rows.exact, rows.predicted_token_id.eq(rows.shifted_target_token_id))
        np.testing.assert_array_equal(rows.within10, errors.le(10))
        record = {
            "method": method, "edit_rule": rule, "fit_id": fit_id,
            "n": len(rows), "source_prompts": 100,
            "numeric_output_mae": errors.mean(),
            "mean_edit_norm": rows.applied_edit_norm.mean(),
        }
        for metric in ("exact", "within10"):
            scores = rows.groupby("source_prompt_id", sort=True)[metric].mean().to_numpy()
            low, high = (
                np.quantile(scores[indices].mean(axis=1), [.025, .975])
                if rows[metric].sum() else (np.nan, np.nan)
            )
            record.update({
                metric + "_count": int(rows[metric].sum()),
                metric + "_accuracy": rows[metric].mean(),
                metric + "_low": low, metric + "_high": high,
            })
        snapshot = rows[PAIR_KEYS + BEHAVIOR_COLUMNS].copy()
        if fit_name != "control":
            attained = rows.post_edit_probe_prediction.eq(rows.target_value)
            np.testing.assert_array_equal(rows.post_edit_probe_target, attained)
            if not np.isfinite(rows.post_edit_code_mse).all() or rows.post_edit_code_mse.lt(0).any():
                raise ValueError(f"Invalid post-edit code errors: {method}")
            if rule == "prediction_corrected" and not attained.all():
                raise ValueError(f"Corrected edits fail to reach the probe target: {method}")
            record.update({
                "post_edit_probe_target_count": int(attained.sum()),
                "post_edit_probe_target_accuracy": attained.mean(),
                "post_edit_code_mse": rows.post_edit_code_mse.mean(),
            })
            snapshot[PROBE_COLUMNS] = rows[PROBE_COLUMNS]
        else:
            for column in PROBE_COLUMNS:
                snapshot[column] = np.nan
        snapshot.insert(0, "edit_rule", rule)
        snapshot.insert(0, "method", method)
        snapshots.append(snapshot)
        summaries.append(record)

    pd.DataFrame(summaries).to_csv(packet_dir / "steering_summary.csv", index=False)
    # Adjacent conditions share case identifiers; this order keeps the frozen
    # packet compact without rounding any predictions or code errors.
    pd.concat(snapshots, ignore_index=True).sort_values(["case_id", "method"]).to_csv(
        packet_dir / "steering_rows.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest["intervention_validation"] = {
        "primary_edit_rule": "prediction_corrected",
        "comparison_edit_rule": "label_difference",
        "application_dtype": manifest["dtype"],
        "probe_target_criterion": "Post-edit decoded operand equals requested operand",
        "code_error": "Mean squared error to the complete target code after FP16 application",
        "methods": {method: {"fit_name": fit, "edit_rule": rule}
                    for method, (fit, rule) in METHODS.items()},
        "exporter": "05_current_evidence/export_figure7_interventions.py",
        "exporter_sha256": digest(Path(__file__)),
    }
    manifest["files"] = {name: digest(packet_dir / name) for name in manifest["files"]}
    (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if release_dir is not None:
        release_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", *manifest["files"]):
            shutil.copyfile(packet_dir / name, release_dir / name)
    print(json.dumps({"conditions": len(summaries), "intervention_rows": sum(len(r) for r in snapshots),
                      "probe_validation": [{k: r[k] for k in
                          ("method", "post_edit_probe_target_count", "exact_count")}
                          for r in summaries if "post_edit_probe_target_count" in r]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.packet_dir, args.release_dir)
