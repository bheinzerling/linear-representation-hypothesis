#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_cairo import RendererCairo, cairo
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/deletion"
STEM = "2026-10-03-figure5c-numerical-use-candidates"
OUT = ROOT / "out/figures" / STEM
FIG_W, FIG_H = 16 / 2.54, 2.35
PANEL_W = FIG_W * .290
TASKS = ["addition_formula", "addition_verbalized", "subtraction_formula",
         "subtraction_verbalized", "identity_patchscopes", "parity", "magnitude_comparison"]
TASK_LABELS = ["Add (formula)", "Add (verbal)", "Sub (formula)", "Sub (verbal)",
               "Copying", "Parity", "Comparison"]
FULL_LABELS = ["Addition, formula", "Addition, verbalized", "Subtraction, formula",
               "Subtraction, verbalized", "Copying", "Parity", "Comparison"]
FAMILIES = ["original", "delete_shuffled", "delete_property"]
CONDITIONS = ["Original", "Shuffled", "Property"]
INK, MID, LIGHT = "#25231f", "#73706a", "#d8d3ca"
BLUE, GOLD, PLUM = "#35618d", "#ac7d00", "#83546b"
NS = "http://www.w3.org/2000/svg"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style() -> dict:
    fonts = {}
    for style in ["Regular", "Bold", "Italic", "Bold Italic"]:
        path = Path(subprocess.check_output(
            ["fc-match", "-f", "%{file}", f"TeX Gyre Heros:style={style}"], text=True))
        font_manager.fontManager.addfont(path)
        assert font_manager.FontProperties(fname=path).get_name() == "TeX Gyre Heros"
        fonts[style] = {"path": str(path), "sha256": digest(path)}
    mpl.rcParams.update({
        "font.family": "TeX Gyre Heros", "font.size": 7.2, "mathtext.fontset": "cm",
        "axes.labelsize": 7.0, "xtick.labelsize": 6.4, "ytick.labelsize": 6.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MID,
        "xtick.color": MID, "ytick.color": MID, "axes.linewidth": .55,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path", "figure.facecolor": "white", "savefig.facecolor": "white",
        "lines.solid_capstyle": "round",
    })
    return fonts


def load_evidence() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((DATA / "snapshot_manifest.json").read_text())
    for row in manifest["files"]:
        assert digest(DATA / row["snapshot"]) == row["sha256"], row["snapshot"]
    summary = pd.read_csv(DATA / "summary.csv")
    splits = pd.read_csv(DATA / "per_split.csv")
    assert not summary.duplicated(["task_id", "family"]).any()
    assert set(summary.task_id) == set(TASKS)
    assert (summary.n_prompts == 1000).all() and (summary.n_unique_values == 639).all()
    assert (summary.n_splits == 10).all()
    assert len(summary) == 56 and len(splits) == 560
    assert (summary.accuracy_low <= summary.accuracy + 1e-12).all()
    assert (summary.accuracy <= summary.accuracy_high + 1e-12).all()
    assert ((summary.accuracy_low >= 0) & (summary.accuracy_high <= 1)).all()
    grouped = splits.groupby(["task_id", "family"])[["accuracy", "accuracy_change"]].mean()
    indexed = summary.set_index(["task_id", "family"])
    np.testing.assert_allclose(grouped, indexed.loc[grouped.index, grouped.columns], atol=1e-12)
    np.testing.assert_allclose(summary.accuracy - summary.baseline_accuracy, summary.accuracy_change)
    np.testing.assert_allclose(
        (summary.gained_correct_n - summary.lost_correct_n) / summary.n_prompts,
        summary.accuracy_change, atol=1e-12)
    selected = indexed.loc[pd.MultiIndex.from_product([TASKS, FAMILIES])].reset_index()
    selected.to_csv(OUT / "plotted_values.csv", index=False)
    return summary, {"snapshot_hashes": True, "complete_grid": True,
                     "intervals_bounded": True, "means_match_ten_splits": True,
                     "paired_changes_match_lost_and_repaired": True,
                     "plotted_rows_per_candidate": len(selected)}


def arrays(summary: pd.DataFrame, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = summary.loc[summary.family == family].set_index("task_id").loc[TASKS]
    return tuple(frame[c].to_numpy() * 100 for c in ["accuracy", "accuracy_low", "accuracy_high"])


def text(parent, x, y, value, size=6.4, **kwargs):
    return parent.text(x, y, value, fontsize=size, **kwargs)


def legend(parent, property_color=BLUE):
    for x, name, marker, color, fill in [
        (.025, "Original", "o", MID, "white"),
        (.355, "Shuffled", "s", MID, "white"),
        (.710, "Property", "o", property_color, property_color),
    ]:
        parent.plot(x, .987, marker=marker, ms=3.1, mew=.7, color=color,
                    mfc=fill, clip_on=False)
        text(parent, x + .037, .985, name, 5.9, va="center")


def intervals(parent, summary):
    legend(parent)
    ax = parent.inset_axes([.425, .19, .55, .68])
    original, _, _ = arrays(summary, "original")
    prop, _, _ = arrays(summary, "delete_property")
    for i in range(7):
        ax.plot([prop[i], original[i]], [i, i], color="#dedede", lw=1.25, zorder=1)
    for family, offset, marker, color, fill in [
        ("original", -.13, "o", MID, "white"),
        ("delete_shuffled", .13, "s", MID, "white"),
        ("delete_property", 0, "o", BLUE, BLUE),
    ]:
        value, low, high = arrays(summary, family)
        ax.errorbar(value, np.arange(7) + offset, xerr=[value-low, high-value],
                    fmt=marker, color=color, mfc=fill, ms=3.1, mew=.65,
                    lw=.7, capsize=1.3, clip_on=False, zorder=3)
    ax.set(xlim=(0, 100), ylim=(6.55, -.55), xticks=[0, 50, 100], yticks=np.arange(7))
    ax.set_yticklabels(TASK_LABELS, fontsize=6.35)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, pad=2, labelsize=6.0)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=.4)
    ax.set_axisbelow(True)
    text(parent, .70, .003, "Accuracy (%)", 6.5, ha="center", va="bottom")


def matrix(parent, summary):
    cmap = LinearSegmentedColormap.from_list("accuracy", ["#fbf8ef", "#94b4ac", "#155a4a"])
    left, width = .405, .185
    for j, name in enumerate(CONDITIONS):
        text(parent, left + (j+.5)*width, .98, name, 5.8, ha="center", va="center")
    for i, task in enumerate(TASKS):
        y = .86 - i*.097
        text(parent, 0, y+.008, TASK_LABELS[i], 6.35, va="center")
        for j, family in enumerate(FAMILIES):
            row = summary.loc[(summary.task_id == task) & (summary.family == family)].iloc[0]
            x = left + j*width
            v, lo, hi = row.accuracy*100, row.accuracy_low*100, row.accuracy_high*100
            background = cmap(v/100)
            parent.add_patch(Rectangle((x, y-.043), width-.010, .090,
                                       facecolor=background, edgecolor="none"))
            linear_rgb = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4
                          for c in background[:3]]
            luminance = np.dot([.2126, .7152, .0722], linear_rgb)
            white_contrast = 1.05/(luminance+.05)
            black_contrast = (luminance+.05)/.05
            color = "white" if white_contrast >= black_contrast else "black"
            assert max(white_contrast, black_contrast) >= 4.5
            text(parent, x+(width-.010)/2, y+.009, f"{v:.1f}", 6.35,
                 color=color, ha="center", va="center")
            # A cell-local interval strip uses exactly the same 0–100 scale.
            x0, span, yy = x+.018, width-.046, y-.029
            parent.plot([x0, x0+span], [yy, yy], color=color, alpha=.25, lw=.45)
            parent.plot([x0+span*lo/100, x0+span*hi/100], [yy, yy], color=color, lw=.9)
            parent.plot(x0+span*v/100, yy, "|", ms=2.0, mew=.65, color=color)
    for j in range(100):
        parent.add_patch(Rectangle((.405+j*.00555, .120), .0056, .028,
                                   facecolor=cmap(j/99), edgecolor="none"))
    for v in [0, 50, 100]:
        text(parent, .405+.555*v/100, .090, str(v), 5.8, ha="center", va="top")
    text(parent, .69, .003, "Accuracy (%)", 6.5, ha="center", va="bottom")


def mirrored(parent, summary):
    ax = parent.inset_axes([.405, .20, .57, .67])
    original, blo, bhi = arrays(summary, "original")
    prop, plo, phi = arrays(summary, "delete_property")
    control, clo, chi = arrays(summary, "delete_shuffled")
    yy = np.arange(7)
    ax.barh(yy, -original, height=.45, color="#b7b8bb", edgecolor="none")
    ax.barh(yy, prop, height=.45, color=PLUM, edgecolor="none")
    ax.errorbar(-original, yy, xerr=[bhi-original, original-blo], fmt="none",
                color=INK, lw=.7, capsize=1.4, clip_on=False)
    ax.errorbar(prop, yy, xerr=[prop-plo, phi-prop], fmt="none",
                color=PLUM, lw=.8, capsize=1.5, clip_on=False)
    ax.errorbar(control, yy+.24, xerr=[control-clo, chi-control], fmt="D", ms=2.6,
                color=INK, mfc="white", mew=.6, lw=.55, capsize=1, clip_on=False)
    ax.axvline(0, color=MID, lw=.5)
    ax.set(xlim=(-100, 100), ylim=(6.55, -.55), yticks=yy,
           yticklabels=TASK_LABELS, xticks=[-100, 0, 100], xticklabels=["100", "0", "100"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3, labelsize=6.35)
    ax.tick_params(axis="x", length=2, pad=2, labelsize=6)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    text(parent, .54, .965, "Original", 6.0, ha="center")
    text(parent, .835, .965, "Property", 6.0, ha="center", color=PLUM)
    parent.plot(.07, .09, "D", ms=2.8, mfc="white", color=INK, mew=.65)
    text(parent, .105, .09, "Shuffled", 5.8, va="center")
    text(parent, .70, .003, "Accuracy (%)", 6.5, ha="center", va="bottom")


def slopes(parent, summary):
    xs = [.115, .34, .565]
    y0, h = .19, .67
    for x, name in zip(xs, CONDITIONS, strict=True):
        parent.plot([x, x], [y0, y0+h], color=LIGHT, lw=.6)
        text(parent, x, .955, name, 5.6, ha="center")
    for v in [0, 50, 100]:
        y = y0+h*v/100
        text(parent, 0, y, str(v), 5.6, ha="left", va="center", color=MID)
        parent.plot([xs[0], xs[-1]], [y, y], color="#eceae5", lw=.45, zorder=0)
    colors = [BLUE, BLUE, "#a64e50", "#a64e50", GOLD, "#307a65", "#8064a2"]
    markers = ["o", "s", "o", "s", "D", "^", "v"]
    linestyles = ["-", (0,(2,1)), "-", (0,(2,1)), "-", "-", "-"]
    # Endpoint labels follow measured positions; separate labels in dense bands.
    p, _, _ = arrays(summary, "delete_property")
    label_y = y0+h*p/100
    idx = np.argsort(label_y)
    for lower, upper in zip(idx[:-1], idx[1:]):
        label_y[upper] = max(label_y[upper], label_y[lower]+.071)
    for i, task in enumerate(TASKS):
        rows = summary.loc[summary.task_id == task].set_index("family").loc[FAMILIES]
        yy = y0+h*rows.accuracy.to_numpy()
        parent.plot(xs, yy, color=colors[i], ls=linestyles[i], lw=.85,
                    marker=markers[i], ms=2.3, mew=.4, zorder=2)
        parent.errorbar(xs, yy, yerr=np.array([rows.accuracy-rows.accuracy_low,
                                              rows.accuracy_high-rows.accuracy])*h,
                        fmt="none", color=colors[i], lw=.65, capsize=1.2, zorder=3)
        parent.plot([xs[-1]+.014, .645], [yy[-1], label_y[i]], color=colors[i], lw=.5)
        text(parent, .659, label_y[i], TASK_LABELS[i], 5.75,
             va="center", color="#8c6000" if i == 4 else colors[i])
    text(parent, .36, .045, "Accuracy (%)", 6.4, ha="center", va="center")


def typographic(parent, summary):
    xx = [.48, .69, .90]
    for x, name in zip(xx, CONDITIONS, strict=True):
        text(parent, x, .965, name, 5.8, ha="center")
    parent.plot([0, 1], [.918, .918], color=INK, lw=.65)
    for i, task in enumerate(TASKS):
        y = .855-i*.113
        text(parent, 0, y-.015, TASK_LABELS[i], 6.0, va="center")
        for j, family in enumerate(FAMILIES):
            row = summary.loc[(summary.task_id == task) & (summary.family == family)].iloc[0]
            v, lo, hi = row.accuracy*100, row.accuracy_low*100, row.accuracy_high*100
            text(parent, xx[j], y, f"{v:.1f}", 7.0, ha="center", va="center",
                 weight="bold" if j == 2 else "normal")
            fmt = (lambda z: "100" if z == 100 else f"{z:.1f}")
            text(parent, xx[j], y-.046, f"{fmt(lo)}–{fmt(hi)}", 4.9,
                 ha="center", va="center", color=MID)
        if i in [1, 3, 4]:
            parent.plot([0, 1], [y-.078, y-.078], color=LIGHT, lw=.4)
    parent.plot([0, 1], [.087, .087], color=INK, lw=.6)
    text(parent, .5, .002, "Accuracy %, 95% intervals below", 5.65,
         ha="center", va="bottom")


DRAW = [intervals, matrix, mirrored, slopes, typographic]


def header(fig, left, title, subtitle):
    fig.text(left, (FIG_H-.08)/FIG_H, title, va="top", fontsize=8.8, weight="semibold")
    fig.text(left, (FIG_H-.31)/FIG_H, subtitle, va="top", fontsize=7.0, color=MID)


def parent_axes(fig, left, width):
    ax = fig.add_axes([left, .14/FIG_H, width, 1.73/FIG_H])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax






















