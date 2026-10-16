
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/deletion"
TASKS = ["addition_formula", "addition_verbalized", "subtraction_formula",
         "subtraction_verbalized", "identity_patchscopes", "parity", "magnitude_comparison"]
LABELS = ["a + b", "a plus b", "a − b", "a minus b", "b = ", "Is b even?", "b > 500?"]
FAMILIES = ["original", "delete_shuffled", "delete_property"]
CONDITIONS = ["Original", "Rank-matched\ncontrol", "Property"]
COLORS = ["#35618d", "#35618d", "#a64e50", "#a64e50", "#ac7d00", "#307a65", "#8064a2"]


def load_data() -> tuple[pd.DataFrame, dict]:
    source = DATA / "summary.csv"
    manifest = json.loads((DATA / "snapshot_manifest.json").read_text())
    expected = next(r["sha256"] for r in manifest["files"] if r["snapshot"] == "summary.csv")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    summary = pd.read_csv(source).set_index(["task_id", "family"])
    selected = summary.loc[pd.MultiIndex.from_product([TASKS, FAMILIES])]
    assert len(selected) == 21 and not selected.index.duplicated().any()
    assert (selected.n_prompts == 1000).all() and (selected.n_unique_values == 639).all()
    assert (selected.n_splits == 10).all()
    assert (selected.accuracy_low <= selected.accuracy).all()
    assert (selected.accuracy <= selected.accuracy_high).all()
    assert ((selected.accuracy_low >= 0) & (selected.accuracy_high <= 1)).all()

    # Match Figure 4b's arithmetic labels without importing its renderer.
    figure4 = ROOT / "figures/transfer_panels.py"
    prompt_labels = next(ast.literal_eval(node.value) for node in ast.parse(figure4.read_text()).body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "PROMPT_LABELS" for t in node.targets))
    assert LABELS[:4] == prompt_labels[:4]
    provenance = {
        "source": str(source.relative_to(ROOT)), "source_sha256": expected,
        "task_labels": dict(zip(TASKS, LABELS, strict=True)), "conditions": FAMILIES,
        "condition_labels": dict(zip(FAMILIES, [label.replace("\n", " ") for label in CONDITIONS], strict=True)),
        "metric": "Exact next-token accuracy, percent",
        "intervention": "Delete the centered projection at the source-number token in hidden-state index 10",
        "training_prompts": "Addition formulas; the fitted bases and center transfer to other prompt formats",
        "n_prompts_per_task": 1000, "n_unique_source_values": 639, "n_overlapping_splits": 10,
        "uncertainty": json.loads((DATA / "provenance.json").read_text())["bootstrap"],
        "label_reference": "Figure 4b: PROMPT_LABELS[:4] in transfer_panels.py",
        "checks": {"source_hash": True, "complete_plotted_grid": True,
                   "bounded_intervals": True, "arithmetic_labels_match_figure4b": True},
    }
    return summary, provenance


def draw(parent, summary: pd.DataFrame) -> None:
    """Preserve the selected candidate's scales, uncertainty, and task paths."""
    xs = [.115, .34, .565]
    y0, height = .19, .67
    for x, name in zip(xs, CONDITIONS, strict=True):
        parent.plot([x, x], [y0, y0 + height], color="#d8d3ca", lw=.6)
        parent.text(x, .955, name, fontsize=5.6, ha="center")
    for value in [0, 50, 100]:
        y = y0 + height * value / 100
        parent.text(0, y, str(value), fontsize=5.6, ha="left", va="center", color="#73706a")
        parent.plot([xs[0], xs[-1]], [y, y], color="#eceae5", lw=.45, zorder=0)
    markers = ["o", "s", "o", "s", "D", "^", "v"]
    linestyles = ["-", (0, (2, 1)), "-", (0, (2, 1)), "-", "-", "-"]
    property_accuracy = np.array([summary.loc[(t, "delete_property"), "accuracy"] for t in TASKS])
    label_y = y0 + height * property_accuracy
    order = np.argsort(label_y)
    for lower, upper in zip(order[:-1], order[1:]):
        label_y[upper] = max(label_y[upper], label_y[lower] + .071)
    for i, task in enumerate(TASKS):
        rows = summary.loc[task].loc[FAMILIES]
        yy = y0 + height * rows.accuracy.to_numpy()
        parent.plot(xs, yy, color=COLORS[i], ls=linestyles[i], lw=.85,
                    marker=markers[i], ms=2.3, mew=.4, zorder=2)
        parent.errorbar(xs, yy, yerr=np.array([rows.accuracy - rows.accuracy_low,
                                              rows.accuracy_high - rows.accuracy]) * height,
                        fmt="none", color=COLORS[i], lw=.65, capsize=1.2, zorder=3)
        parent.plot([xs[-1] + .014, .645], [yy[-1], label_y[i]], color=COLORS[i], lw=.5)
        parent.text(.659, label_y[i], LABELS[i], fontsize=5.75, va="center",
                    color="#8c6000" if i == 4 else COLORS[i], gid=f"numeric-task-{task}")
    parent.text(.36, .045, "Accuracy (%)", fontsize=6.4, ha="center", va="center")
