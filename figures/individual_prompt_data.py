"""Load the matched Figure 7 packet through the released observation verifier."""
from __future__ import annotations

import json
from pathlib import Path
import runpy

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/numerical_sufficiency_atlas/individual_prompt_fits"
VERIFIER = ROOT / "src/lrh/analysis/figure7_matched.py"


def load():
    evidence = runpy.run_path(str(VERIFIER))["calculate"](DATA)
    steering = pd.read_csv(DATA / "steering_summary.csv").set_index("method")
    control = steering.rename(columns={"numeric_output_mae": "direct_steering_mae"}).copy()
    control["argmax_accuracy"] = control["exact_accuracy"]
    return {
        "test": evidence["test"], "outputs": evidence["support"],
        "metrics": evidence["decoding"].set_index("architecture")
            .rename(columns={"mae": "mean_abs_error"}),
        "control": control, "matched_fit_manifest": evidence["manifest"],
        "probe_validation": evidence["probe_validation"],
    }


if __name__ == "__main__":
    evidence = load()
    print(json.dumps({
        "status": "passed", "source_prompts": len(evidence["test"]),
        "fit_ids": evidence["matched_fit_manifest"]["fit_ids"],
        "probe_validation": evidence["probe_validation"].to_dict(orient="records"),
    }, indent=2))
