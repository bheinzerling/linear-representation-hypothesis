
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/figures/causal_use/arditi_snapshot_manifest.json"
DATA = ROOT / "data/figures/refusal"
MODELS = ["Qwen 1.8B", "Yi 6B", "Gemma 2B", "Llama 2 7B", "Llama 3 8B"]
MID, PALE, BLUE, GOLD = "#73706a", "#e9e6df", "#35618d", "#ac7d00"


def load_data() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text())
    source_path = ROOT / manifest["snapshot"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest["sha256"]
    source = json.loads(source_path.read_text())
    assert source["commit"] == manifest["source_commit"]
    selected = [r for r in source["evaluations"]
                if r["dataset"] == "jailbreakbench" and r["condition"] in ["baseline", "ablation"]]
    data = {(r["model"], r["condition"]): r for r in selected}
    assert len(data) == 10
    shared = []
    for model in MODELS:
        before, after = [data[model, condition] for condition in ["baseline", "ablation"]]
        for row in [before, after]:
            assert row["n"] == len(row["samples"]) == 100
            assert all(s["refusal"] in [0, 1] for s in row["samples"])
            assert row["refusals"] == sum(s["refusal"] for s in row["samples"])
            assert row["refusal_rate"] == row["refusals"] / 100
        hashes = [s["prompt_sha256"] for s in before["samples"]]
        assert hashes == [s["prompt_sha256"] for s in after["samples"]]
        shared.append(tuple(hashes))
    assert len(set(shared)) == 1
    provenance = {
        "source": manifest["snapshot"], "source_sha256": manifest["sha256"],
        "source_commit": source["commit"], "selection": manifest["selection"],
        "selected_candidate": manifest["selected_candidate"],
        "metric": "Phrase-based refusal, percent; exact counts out of 100 harmful instructions",
        "intervention": "Remove the projection on the selected direction at all residual-stream sites and token positions",
        "schematic": "Constructed orthogonal projection; no measured activation positions",
        "estimates": [{"model": m, "original_percent": data[m, "baseline"]["refusals"],
                       "ablated_percent": data[m, "ablation"]["refusals"]} for m in MODELS],
        "source_urls": [r["source_url"] for r in selected],
        "checks": {"frozen_source_hash": True, "counts_match_individual_labels": True,
                   "paired_prompt_sets": True, "shared_prompt_set_across_models": True},
    }
    return data, provenance


def text(parent, x, y, value, size=6.5, **kwargs):
    return parent.text(x, y, value, fontsize=size, **kwargs)


def draw(parent, data) -> None:
    """Preserve the selected candidate's composition, formula, and measured values."""
    text(parent, .5, .966, "Remove the component along r", 6.0, ha="center")
    origin, original, edited = (.22, .72), (.52, .89), (.52, .72)
    parent.plot([origin[0], .83], [origin[1], origin[1]], color=MID, lw=.55)
    parent.plot([.22, .22], [.70, .92], color=MID, lw=.55)
    for start, end, color in [(origin, original, GOLD), (origin, edited, BLUE),
                              (edited, original, MID)]:
        parent.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                        mutation_scale=6, lw=1.1, color=color))
    parent.plot([original[0], edited[0]], [original[1], edited[1]],
                color=MID, ls=(0, (2, 2)), lw=.65)
    text(parent, .55, .897, "x", 7, color=GOLD)
    text(parent, .43, .755, r"$x'$", 7, color=BLUE, ha="center")
    text(parent, .79, .74, r"$\perp r$", 6, color=MID, ha="center")
    text(parent, .55, .802, "r", 6, color=MID, ha="left")
    text(parent, .5, .621, r"$x' = x - \hat{r}\hat{r}^{\!\top}x$", 10, ha="center")
    text(parent, .5, .559, "All layers · all token positions", 5.8, ha="center", color=MID)
    parent.plot([0, 1], [.52, .52], color=PALE, lw=.6)
    text(parent, 0, .475, "Measured refusal (%)", 6.2, weight="bold")
    text(parent, .8, .477, "Original → ablated", 5.4, ha="center")
    for i, model in enumerate(MODELS):
        y = .404 - i * .074
        before, after = [data[model, condition] for condition in ["baseline", "ablation"]]
        text(parent, 0, y, model, 6.2, va="center")
        text(parent, .82, y, f"{before['refusals']} → {after['refusals']}",
             7, ha="center", va="center", color=BLUE, gid=f"refusal-result-{i}")
    text(parent, .5, .016, "Projection schematic; n = 100/model", 5.7,
         ha="center", color=MID)
