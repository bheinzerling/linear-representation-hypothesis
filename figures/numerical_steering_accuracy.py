#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from numerical_uncertainty import load_steering_uncertainty

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/numerical_sufficiency_atlas"
ROWS = DATA / "steering_accuracy_rows.csv.gz"
PROVENANCE = DATA / "steering_accuracy.provenance.json"
CONTROL = ROOT / "data/figures/coefficient_predictions/control.csv"
METHODS = ("linear", "circular_reg", "centroid_patch")
PAIR_KEYS = ["baseline_row_idx", "target_slot", "source_value", "target_value", "target_output"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    results = {}
    reference = None
    for method in METHODS:
        part = rows[rows.method == method].sort_values(PAIR_KEYS[:2]).reset_index(drop=True)
        assert len(part) == 2000 and not part.duplicated(PAIR_KEYS[:2]).any()
        if reference is None:
            reference = part
        else:
            pd.testing.assert_frame_equal(reference[PAIR_KEYS + ["no_edit_output"]],
                                          part[PAIR_KEYS + ["no_edit_output"]])
        error = (part.output - part.target_output).abs()
        results[method] = {"n": len(part), "exact_count": int(error.eq(0).sum()),
                           "within10_count": int(error.le(10).sum())}
    error = (reference.no_edit_output - reference.target_output).abs()
    results["no_edit"] = {"n": len(reference), "exact_count": int(error.eq(0).sum()),
                          "within10_count": int(error.le(10).sum())}
    for result in results.values():
        result["exact_accuracy"] = result["exact_count"] / result["n"]
        result["within10_accuracy"] = result["within10_count"] / result["n"]
    return results


def load(control):
    provenance = json.loads(PROVENANCE.read_text())
    assert digest(ROWS) == provenance["snapshot_sha256"]
    results = summarize(pd.read_csv(ROWS))
    assert results == provenance["results"]
    for method in METHODS:
        assert results[method]["n"] == control.loc[method, "n"]
        assert results[method]["exact_accuracy"] == control.loc[method, "argmax_accuracy"]
    frame = pd.DataFrame.from_dict(results, orient="index")
    uncertainty = load_steering_uncertainty()
    for metric in ("exact", "within10"):
        part = uncertainty.xs(metric, level="metric").reindex(frame.index)
        np.testing.assert_allclose(part.estimate, frame[f"{metric}_accuracy"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(part["count"], frame[f"{metric}_count"])
        for column in ("low", "high", "change_from_no_edit", "change_low", "change_high"):
            frame[f"{metric}_{column}"] = part[column]
    return frame


def draw_accuracy_intervals(axis, results, methods, *, exact_color, within10_color,
                            linewidth=.75, capsize=2.8):
    """Add source-cluster whiskers to an existing horizontal accuracy display.

    Zero exact successes have unavailable endpoints and receive no whisker.
    The observed counts are available in ``results`` for captions or tables.
    """
    for y, method in enumerate(methods):
        for metric, color in (("exact", exact_color), ("within10", within10_color)):
            row = results.loc[method]
            value, lo, hi = (100 * row[f"{metric}_{name}"]
                             for name in ("accuracy", "low", "high"))
            if np.isfinite([lo, hi]).all():
                axis.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt="none",
                              color=color, elinewidth=linewidth, capsize=capsize,
                              capthick=linewidth, zorder=2.5, clip_on=False)




