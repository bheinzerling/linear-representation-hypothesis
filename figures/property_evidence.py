#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import property_panels as base
import property_bars as bars

ROOT = base.ROOT
DATA = ROOT / "data/figures/number_property_evidence_main"
OUTPUT = ROOT / "figures/fig_number_property_evidence_main"
COMMAND = ("uv run --group figures python "
           "figures/property_evidence.py")


def accessibility(ax, properties):
    start, width = .352, .550
    base.text(ax, 0, .953, "Property", size=6.1)
    base.text(ax, .995, .953, "Mean (%)", size=5.8, ha="right", color=base.GRAY)
    bars.scale(ax, start, width)
    for i, row in enumerate(properties.sort_values("full_activation_mean", kind="stable").itertuples()):
        y = .848-i*.041
        label = row.display_name.replace("Distinct-digit count", "Distinct digits")
        base.text(ax, 0, y, label, size=5.9, va="center")
        bars.rect(ax, start, y, width, .022, bars.TRACK)
        bars.rect(ax, start, y, width*row.full_activation_mean, .022, base.BLUE)
        bars.range_mark(ax, start, width, y, row.full_activation_min,
                        row.full_activation_max, halo=True)
        bars.chance_tick(ax, start, width, y, row.chance, half=.014)
        base.text(ax, .995, y, f"{row.full_activation_mean*100:.1f}", size=5.9,
                  ha="right", va="center")
    base.text(ax, .5, .047, "Balanced accuracy (%)", size=5.5, ha="center", color=base.GRAY)
    base.text(ax, .5, .008, "Whiskers: split ranges · red ticks: chance", size=5.3,
              ha="center", color=base.GRAY)


def render(properties, counts, examples):
    base.mpl.rcParams["svg.hashsalt"] = "number-property-evidence-main"
    fig = base.plt.figure(figsize=(base.W, base.H))
    starts, widths = [.012, .434, .677], [.397, .216, .310]
    base.header(fig, starts[0], "(a) Property accessibility", "Llama-3.1-8B · full activations")
    base.header(fig, starts[1], "(b) Contradiction frequency", "Independent parity-direction tests")
    base.header(fig, starts[2], "(c) Property contradictions", "Selected test examples")
    axes = [base.parent(fig, x, w) for x, w in zip(starts, widths, strict=True)]
    accessibility(axes[0], properties)
    base.quantitative(axes[1], counts, 0)
    base.examples(axes[2], examples, 0)
    return fig

