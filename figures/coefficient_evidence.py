#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-04-figure8-sufficiency-candidates"
OUT = ROOT / "out/figures" / STEM
INPUT = ROOT / "data/figures/coefficient_predictions"
INK, GRAY, GRID = "#25231f", "#696969", "#dedede"
BLUE, TEAL, ORANGE, PURPLE = "#35618d", "#16766f", "#b9602b", "#7b4fa3"
METHODS = ["independent_property_all", "ridge", "rbf"]
NAMES = ["Independent sum", "Joint ridge", "RBF ridge"]



def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()




def load_evidence():
    control = pd.read_csv(INPUT / "control.csv").set_index("group_key")
    metrics = pd.read_csv(INPUT / "probe_metrics.csv")
    metrics = metrics.loc[(metrics.split == "test") & (metrics.fit_mode == "all_x")]
    assert len(metrics) == 4 and metrics.num_examples.eq(100).all()
    metrics = metrics.set_index("architecture")
    outputs = pd.read_csv(INPUT / "probe_outputs.csv")
    assert outputs.partition.value_counts().to_dict() == {"train": 800, "dev": 100, "test": 100}
    test = outputs.loc[outputs.partition == "test"].copy()
    # Archived decoder metrics score integer-valued predictions; the exported
    # ridge feature is continuous and is shown directly in the scatter.
    np.testing.assert_allclose(np.abs(np.rint(test.ridge_prediction) - test.value).mean(),
                               metrics.loc["linear", "mean_abs_error"])
    np.testing.assert_allclose(np.abs(test.circular_prediction - test.value).mean(),
                               metrics.loc["circular_reg", "mean_abs_error"])
    assert (control.loc[["linear", "pls", "circular_reg", "circular_cls"], "argmax_accuracy"] == 0).all()
    np.testing.assert_allclose(metrics.loc[["linear", "circular_reg"], "mean_abs_error"], [8.96, 1.63])
    np.testing.assert_allclose(control.loc["centroid_patch", "argmax_accuracy"], .993)
    rows = pd.read_csv(INPUT / "coefficient_rows.csv.gz")
    keys = ["panel_row_idx", "baseline_row_idx", "source_value", "target_value"]
    reference = rows.loc[rows.condition == METHODS[0], keys].reset_index(drop=True)
    estimates = []
    for method in METHODS:
        part = rows.loc[rows.condition == method]
        assert len(part) == 400
        assert part[keys].reset_index(drop=True).equals(reference)
        means = part.groupby("baseline_row_idx", sort=True).correct.mean().to_numpy()
        assert len(means) == 20
        # Same estimator and seed as the production Figure 8.
        rng = np.random.default_rng(20260826)
        boots = rng.choice(means, size=(10000, 20), replace=True).mean(axis=1)
        lo, hi = np.quantile(boots, [.025, .975])
        estimates.append(dict(method=method, accuracy=part.correct.mean(), low=lo, high=hi))
    coefficient_accuracy = pd.DataFrame(estimates)
    np.testing.assert_allclose(coefficient_accuracy.accuracy, [0, .7675, .795])
    basis = pd.read_csv(INPUT / "basis_control.csv").set_index("method")
    assert basis.loc["property_coordinates", "imposed_rank"] == 41
    assert basis.loc["in_span_search", "n"] == 2000
    np.testing.assert_allclose(basis.loc["in_span_search", "exact_accuracy"], .9945)
    components = pd.read_csv(INPUT / "components.csv")
    assert len(components) == 10 and components.n_values.eq(100).all()
    np.testing.assert_allclose(components.iloc[:8].component_r2_to_support.min(), .8959084268718449)
    return dict(control=control, metrics=metrics, outputs=outputs, test=test,
                coefficient_accuracy=coefficient_accuracy, basis=basis, components=components)


def txt(ax, x, y, value, size=8, color=INK, weight="normal", **kw):
    return ax.text(x, y, value, transform=ax.transAxes, fontsize=size,
                   color=color, fontweight=weight, va="top", **kw)


def canvas(height=3.55):
    fig = plt.figure(figsize=(7.15, height), facecolor="white")
    return fig


def blank(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def heading(fig, x, y, label, title, color=INK):
    fig.text(x, y, f"({label})", fontsize=8.6, fontweight="bold", color=color, va="top")
    fig.text(x + .037, y, title, fontsize=8.6, color=INK, va="top")


def arrow(ax, start, end, color=GRAY, style="-|>", lw=1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                                lw=lw, color=color, transform=ax.transAxes))


def geometry(fig, rect, d, mono=False, small=False):
    ax = fig.add_axes(rect)
    frame, support = d["test"], d["outputs"]
    # A predeclared Figure 9 cell: low-frequency sine components 1 and 3.
    ax.plot(support.target_sin_4, support.target_sin_3, color="#bdbdbd", lw=.8)
    ax.scatter(frame.circular_sin_4, frame.circular_sin_3, s=8 if not small else 5,
               c=INK if mono else frame.value, cmap=None if mono else "viridis",
               norm=None if mono else Normalize(0, 999), marker="^", linewidths=0,
               alpha=.9, zorder=3)
    ax.set(xlim=(-1.22, 1.22), ylim=(-1.22, 1.22), aspect="equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=5.5 if small else 6, length=2, pad=1)
    ax.set_xlabel("Sine component 1", fontsize=6, labelpad=1)
    ax.set_ylabel("Sine component 3", fontsize=6, labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def predictions(fig, rect, d, mono=False):
    ax = fig.add_axes(rect)
    f = d["test"]
    ax.plot([0, 1000], [0, 1000], color="#bdbdbd", lw=.8)
    ax.scatter(f.value, f.ridge_prediction, color=GRAY if mono else BLUE,
               s=9, marker="o", linewidths=.35, facecolors="none", label="Linear ridge")
    ax.scatter(f.value, f.circular_prediction, color=INK if mono else TEAL,
               s=8, marker="+", lw=.6, label="Circular regression")
    ax.set(xlim=(0, 1000), ylim=(0, 1050), xlabel="True value", ylabel="Decoded value")
    ax.set_xticks([0, 500, 1000]); ax.set_yticks([0, 500, 1000])
    ax.tick_params(labelsize=6, length=2, pad=2)
    ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    ax.xaxis.labelpad=1; ax.yaxis.labelpad=1
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.8, loc="lower right", frameon=False, handletextpad=.2,
              borderpad=0, labelspacing=.3)
    return ax


def error_chart(fig, rect, d, mono=False):
    ax = fig.add_axes(rect)
    y = np.arange(3)
    keys = ["linear", "circular_reg", "centroid_patch"]
    decoding = [d["metrics"].loc[k, "mean_abs_error"] for k in keys[:2]]
    steering = d["control"].loc[keys, "direct_steering_mae"].to_numpy()
    for i in range(2):
        ax.plot([decoding[i], steering[i]], [i, i], color=GRID, lw=1.5)
    ax.scatter(decoding, y[:2], color=GRAY if mono else TEAL, s=20, marker="o", label="Decoding")
    ax.scatter(steering, y, color=INK if mono else ORANGE, s=24, marker="x", label="Steering")
    for i, (a, b) in enumerate(zip(decoding, steering[:2], strict=True)):
        ax.text(a+5, i+.1, f"{a:.1f}", fontsize=6, color=GRAY if mono else TEAL)
        ax.text(b-5, i-.16, f"{b:.0f}", fontsize=6, ha="right", color=INK)
    ax.text(steering[2]+6, 2, f"{steering[2]:.1f}", fontsize=6, va="center")
    ax.axvline(223.203, color=GRAY, ls=(0, (2, 2)), lw=.75)
    ax.set(xlim=(0, 250), ylim=(2.55, -.65), xlabel="Mean absolute error")
    ax.set_yticks(y, ["Linear ridge", "Circular reg.", "Centroid patch"])
    ax.set_xticks([0, 100, 200]); ax.tick_params(labelsize=6.5, length=2)
    ax.xaxis.label.set_size(7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2,
              frameon=False, handletextpad=.2, columnspacing=.8)
    ax.text(.53, .99, "No edit: 223", transform=ax.transAxes,
            va="top", fontsize=5.7, color=GRAY)
    return ax


def coefficient_accuracy(fig, rect, d, style="dots", mono=False, wide=False):
    ax = fig.add_axes(rect)
    f = d["coefficient_accuracy"]
    colors = [GRAY, INK, GRAY] if mono else [GRAY, BLUE, ORANGE]
    means = 100 * f.accuracy.to_numpy()
    errors = np.array([100 * (f.accuracy-f.low), 100 * (f.high-f.accuracy)])
    if style == "bars":
        bars = ax.bar(np.arange(3), means, width=.58, color=colors,
                      edgecolor=INK, lw=.5)
        nonzero = means > 0
        ax.errorbar(np.arange(3)[nonzero], means[nonzero], yerr=errors[:, nonzero],
                    fmt="none", color="black", elinewidth=.7, capthick=.7, capsize=2,
                    zorder=2.01)
        for i, (bar, value) in enumerate(zip(bars, means, strict=True)):
            if mono: bar.set_hatch(["", "////", "...."][i])
            ax.text(i, 7 if value == 0 else value-12, f"{value:.1f}", ha="center",
                    va="center", fontsize=7, color=INK if value==0 or i!=1 else "white")
        ax.set(ylim=(0, 100), ylabel="Exact output accuracy (%)")
        ax.set_xticks(np.arange(3), ["Independent\nsum", "Joint\nridge", "RBF\nridge"])
        ax.set_yticks([0, 50, 100]); ax.tick_params(labelsize=6.5, length=2)
        ax.yaxis.label.set_size(7); ax.grid(axis="y", color=GRID, lw=.5)
    else:
        for i, (value, color) in enumerate(zip(means, colors, strict=True)):
            if value > 0:
                ax.errorbar(value, i, xerr=errors[:, i:i+1], fmt=["o", "s", "D"][i],
                            color=color, ms=4, capsize=2, lw=.8, clip_on=False)
            else:
                ax.plot(value, i, marker=["o", "s", "D"][i], linestyle="none",
                        color=color, ms=4, clip_on=False)
            label = f"{value:.1f}%" if value else "0%"
            ax.annotate(label, (value, i), xytext=(5, 8), textcoords="offset points",
                        fontsize=7, color=INK)
        ax.set_xlim(0, 100)
        labels = NAMES.copy()
        if wide:
            val = 100 * d["basis"].loc["in_span_search", "exact_accuracy"]
            ax.scatter(val, 4.1, marker="*", s=42, color=INK if mono else PURPLE, clip_on=False)
            ax.annotate(f"{val:.1f}%", (val, 4.1), xytext=(-8, 8),
                        textcoords="offset points", ha="right", fontsize=7)
            ax.axhline(3.05, color=GRID, lw=.7)
            ax.set_yticks([0, 1, 2, 4.1], labels + ["Output-scored search"])
            ax.set_ylim(4.65, -.6)
            ax.text(.02, .095, "Separate 2,000-edit evaluation", transform=ax.transAxes,
                    fontsize=6.5, color=GRAY)
        else:
            if rect[2] < .20:
                labels[0] = "Independent\nsum"
            ax.set_yticks(range(3), labels); ax.set_ylim(2.7, -.55)
        ax.set_xticks([0, 25, 50, 75, 100] if wide else [0, 50, 100])
        ax.set_xlabel("Exact output accuracy (%)", fontsize=7)
        ax.tick_params(labelsize=6.8, length=2)
        ax.grid(axis="x", color=GRID, lw=.5)
    ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    return ax



























def atlas(d):
    fig = canvas(3.9)
    heading(fig, .025, .97, "a", "Held-out predictions")
    heading(fig, .355, .97, "b", "Circular code excerpt")
    heading(fig, .685, .97, "c", "Steering error")
    predictions(fig, [.08, .65, .205, .25], d)
    geometry(fig, [.398, .63, .18, .27], d)
    error_chart(fig, [.80, .65, .17, .23], d)
    return fig
