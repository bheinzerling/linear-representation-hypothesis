#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Text3D
from mpl_toolkits.mplot3d import proj3d

import coefficient_evidence as candidates
import analytic_geometry as geometry
import numerical_steering_accuracy
import individual_prompt_data

ROOT = Path(__file__).resolve().parents[1]
STEM = "fig_numerical_sufficiency_atlas"
LOWER_ROW_SHIFT = .06
BOTTOM_TRIM = .14
HELIX_RECT = [.0125, .105, .315, .285]
COEFFICIENT_RECT = [.50, .185, .16, .23]
SURFACE_RECT = [.6775, .105, .31, .285]


def load_evidence():
    evidence = candidates.load_evidence()
    evidence.update(individual_prompt_data.load())
    return evidence


def compact_rect(rect):
    """Transform the original layout without changing physical plot dimensions."""
    x, y, width, height = rect
    if y < .5:
        y += LOWER_ROW_SHIFT
    return [x, (y - BOTTOM_TRIM) / (1 - BOTTOM_TRIM),
            width, height / (1 - BOTTOM_TRIM)]


def compact_layout(figure):
    width, height = figure.get_size_inches()
    for axis in figure.axes:
        axis.set_position(compact_rect(axis.get_position(original=True).bounds))
    for text in figure.texts:
        x, y = text.get_position()
        text.set_position(compact_rect([x, y, 0, 0])[:2])
    figure.set_size_inches(width, height * (1 - BOTTOM_TRIM))


def compact_geometry(figure, rect, circular=False):
    """Reuse the accepted constructions with labels fitted to a third-width slot."""
    first = len(figure.texts)
    if circular:
        geometry.draw_helix(figure, rect, circular=True)
        # Separate the activation-axis labels from the source and edited state.
        for label in figure.axes[-1].texts:
            if label.get_text() == "x":
                label.set_position_3d((1.35, 0, -.12))
            elif label.get_text() == "y":
                label.set_position_3d((0, 1.28, -.12))
    else:
        geometry.draw_surface(figure, rect)
    for text in list(figure.texts)[first:]:
        label = text.get_text()
        if label == "Probe-matching edit":
            text.set_text("Edited state")
        elif label == "Target activation":
            text.remove()  # F(T) names the target without a second heading.
        elif label == "Circular components match":
            text.set_text("Original states: helix")
        elif label == "Both properties match":
            text.set_text("Both properties\nmatch")
            text.set_linespacing(1.1)
        elif label.startswith(r"$P(E)=P(T)") or label.startswith(r"$h(u,v)="):
            text.remove()


def draw_steering_accuracy(axis, control):
    results = control
    assert "within10_low" in results.columns
    axis.clear()
    methods = ["linear", "circular_reg", "centroid_patch"]
    y = np.arange(len(methods))
    exact = 100 * results.loc[methods, "exact_accuracy"].to_numpy()
    lenient = 100 * results.loc[methods, "within10_accuracy"].to_numpy()
    for i, (first, second) in enumerate(zip(exact, lenient, strict=True)):
        axis.plot([first, second], [i, i], color=candidates.GRID, lw=1.2, zorder=1)
    axis.axvspan(100 * results.loc["no_edit", "within10_low"],
                 100 * results.loc["no_edit", "within10_high"],
                 color=candidates.GRAY, alpha=.12, linewidth=0, zorder=1)
    axis.axvline(100 * results.loc["no_edit", "within10_accuracy"],
                 color=candidates.GRAY, ls=(0, (2, 2)), lw=.75, zorder=1)
    numerical_steering_accuracy.draw_accuracy_intervals(
        axis, results, methods, exact_color=candidates.ORANGE, within10_color=candidates.TEAL)
    axis.scatter(exact, y, color=candidates.ORANGE, s=24, marker="o", label="Exact", clip_on=False, zorder=3)
    axis.scatter(lenient, y, facecolors="none", edgecolors=candidates.TEAL,
                 linewidths=.9, s=22, marker="D", label="Within ±10", clip_on=False, zorder=3)
    axis.set(xlim=(0, 100), ylim=(2.55, -.65), xlabel="Steering accuracy (%)")
    axis.set_yticks(y, ["Linear ridge", "Circular reg.", "Centroid patch"])
    axis.set_xticks([0, 50, 100])
    axis.tick_params(labelsize=6.5, length=2)
    axis.xaxis.label.set_size(7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=6, loc="lower left", bbox_to_anchor=(-.80, -.42),
                ncol=1, frameon=False, handletextpad=.2)
    return results


def atlas(data):
    figure = candidates.atlas(data)
    # Keep the empirical plots and rebuild the lower row in three slots.
    for axis in list(figure.axes):
        if axis.get_position().y0 < .5:
            axis.remove()
    for text in list(figure.texts):
        if text.get_position()[1] < .6:
            text.remove()
    draw_steering_accuracy(figure.axes[2], data["control"])
    for text in figure.texts:
        if text.get_text() == "Steering error":
            text.set_text("Steering accuracy")
    candidates.heading(figure, .025, .475, "d", "Circular edit leaves\nthe old axial value")
    compact_geometry(figure, HELIX_RECT, circular=True)
    candidates.heading(figure, .355, .475, "e", "20 properties, rank-41 basis")
    axis = candidates.coefficient_accuracy(figure, COEFFICIENT_RECT, data, wide=True)
    axis.set_yticks([0, 1, 2, 4.1], ["Independent\nsum", "Joint ridge", "RBF ridge",
                                  "Output-scored\nsearch"])
    axis.set_xticks([0, 50, 100])
    for text in list(axis.texts):
        text.remove()
    candidates.heading(figure, .685, .475, "f", "Two valid edits have\nan invalid sum")
    compact_geometry(figure, SURFACE_RECT)
    compact_layout(figure)
    return figure

