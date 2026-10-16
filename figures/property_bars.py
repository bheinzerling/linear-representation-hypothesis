#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

import property_panels as base

ROOT = base.ROOT
STEM = "2026-10-04-figure7-accessibility-bars"
OUT = ROOT / "out/figures" / STEM
DATA = base.DATA
FULL, RANK, CHANCE = "#9cb4c9", "#35618d", "#ae5358"
TRACK, INK, GRAY = "#eeeae2", base.INK, base.GRAY
START, WIDTH = .352, .390


def rect(ax, x, y, width, height, face, edge="none", lw=0):
    ax.add_patch(Rectangle((x, y-height/2), width, height, facecolor=face,
                           edgecolor=edge, linewidth=lw, transform=ax.transAxes))


def range_mark(ax, start, width, y, lo, hi, color=INK, cap=.0033, halo=False):
    x0, x1 = start+width*lo, start+width*hi
    effects = [pe.Stroke(linewidth=1.25, foreground="white"), pe.Normal()] if halo else []
    ax.plot([x0, x1], [y, y], color=color, lw=.46, transform=ax.transAxes,
            path_effects=effects, zorder=5)
    for x in [x0, x1]:
        ax.plot([x, x], [y-cap, y+cap], color=color, lw=.46,
                transform=ax.transAxes, path_effects=effects, zorder=5)


def chance_tick(ax, start, width, y, value, mono=False, half=.017):
    x = start+width*value
    ax.plot([x, x], [y-half, y+half], color=INK if mono else CHANCE,
            lw=.6, ls=(0, (1.5, 1)) if mono else "-", transform=ax.transAxes, zorder=7)


def scale(ax, start, width):
    for value in [0, .5, 1]:
        x = start+width*value
        ax.plot([x, x], [.890, .895], color=GRAY, lw=.45, transform=ax.transAxes)
        base.text(ax, x, .908, f"{value*100:.0f}", size=5.0, ha="center", color=GRAY)


def accessibility_bars(ax, frame, style):
    rows = frame.sort_values("full_activation_mean", kind="stable")
    base.text(ax, 0, .953, "Property", size=6.1)
    if style==2:
        for title, x, width in [("Full", .352, .213), ("Rank 41", .711, .203)]:
            base.text(ax, x+width/2, .953, title, size=6.0, ha="center")
            scale(ax, x, width)
    else:
        base.text(ax, .839, .953, "Full", size=5.8, ha="right",
                  color=RANK if style==3 else INK if style==4 else GRAY)
        base.text(ax, .995, .953, "Rank 41", size=5.8, ha="right",
                  color=INK if style in [3,4] else RANK)
        scale(ax, START, WIDTH)
    for i, r in enumerate(rows.itertuples()):
        y = .848-i*.041
        if style==4 and i%2==0:
            rect(ax, -.005, y, 1.005, .040, "#f5f4f1")
        label = r.display_name.replace("Distinct-digit count", "Distinct digits")
        base.text(ax, 0, y, label, size=5.9, va="center")
        if style==2:
            for prefix, x, width, number in [("full_activation", .352, .213, .660),
                                            ("rank41", .711, .203, .995)]:
                mean, lo, hi = [getattr(r, prefix+s) for s in ["_mean", "_min", "_max"]]
                rect(ax, x, y, width, .020, TRACK)
                rect(ax, x, y, width*mean, .020, FULL if prefix=="full_activation" else RANK)
                range_mark(ax, x, width, y, lo, hi)
                chance_tick(ax, x, width, y, r.chance, half=.013)
                base.text(ax, number, y, f"{mean*100:.1f}", size=5.7, ha="right", va="center")
            continue
        if style in [0, 4]:
            for prefix, dy in [("full_activation", .0094), ("rank41", -.0094)]:
                mean, lo, hi = [getattr(r, prefix+s) for s in ["_mean", "_min", "_max"]]
                rect(ax, START, y+dy, WIDTH, .012, TRACK)
                if style==4:
                    rect(ax, START, y+dy, WIDTH*mean, .012,
                         "white" if prefix=="full_activation" else INK,
                         edge=INK, lw=.5)
                else:
                    rect(ax, START, y+dy, WIDTH*mean, .012,
                         FULL if prefix=="full_activation" else RANK)
                range_mark(ax, START, WIDTH, y+dy, lo, hi, halo=style==4)
        else:
            rect(ax, START, y, WIDTH, .026, TRACK)
            if style==1:
                rect(ax, START, y, WIDTH*r.full_activation_mean, .026, FULL)
                rect(ax, START, y, WIDTH*r.rank41_mean, .011, RANK)
            else:
                rect(ax, START, y, WIDTH*r.full_activation_mean, .019, RANK)
                ax.plot(START+WIDTH*r.rank41_mean, y, "D", ms=3.1,
                        mfc="white", mec=INK, mew=.65, transform=ax.transAxes, zorder=9)
            range_mark(ax, START, WIDTH, y+.0162, r.full_activation_min, r.full_activation_max)
            range_mark(ax, START, WIDTH, y-.0162, r.rank41_min, r.rank41_max)
        chance_tick(ax, START, WIDTH, y, r.chance, mono=style==4)
        for prefix, x, color in [("full_activation", .839, RANK if style==3 else GRAY),
                                  ("rank41", .995, INK if style==3 else RANK)]:
            base.text(ax, x, y, f"{getattr(r, prefix+'_mean')*100:.1f}", size=5.7,
                      ha="right", va="center", color=INK if style==4 else color)
    legends = ["Top: full · bottom: rank 41",
               "Broad: full · narrow: rank 41",
               "Both bar columns use the same 0–100% scale",
               "Bar: full · diamond: rank 41",
               "Top outline: full · bottom solid: rank 41"]
    base.text(ax, .5, .061, legends[style], size=5.4, ha="center", color=GRAY)
    if style in [1,3]:
        base.text(ax, .5, .025, "Upper/lower whiskers: full/rank-41 split ranges", size=5.2,
                  ha="center", color=GRAY)
        base.text(ax, .5, -.010, "Balanced accuracy (%) · red tick: chance", size=5.2,
                  ha="center", color=GRAY)
    else:
        base.text(ax, .5, .017, "Balanced accuracy (%) · split ranges · chance ticks", size=5.2,
                  ha="center", color=GRAY)














