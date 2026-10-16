#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, FancyArrowPatch, Wedge
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-09-28-figure-2-framing"
OUT = ROOT / "out/figures/claims"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "svg.hashsalt": STEM})
INK, MUTED, HAIR = "#25231F", "#68665F", "#D9D7D1"
COLORS = ["#B64645", "#D36B24", "#B68A00", "#3F8A5B", "#3B72B9", "#5457A6", "#7B4FA3"]
NAMES = ["Decodability", "Efficiency", "Robustness", "Invariance", "Structural\ncorrespondence", "Necessity", "Sufficiency"]
SHORT = ["D", "E", "R", "I", "SC", "N", "S"]
PT = 6.8 * 72 / 1200
TEXT_AUDIT = []
FIGURES = []


def tint(c, a=.1):
    v = np.array(matplotlib.colors.to_rgb(c))
    return tuple(1 - a * (1 - v))


class Canvas:
    def __init__(self, slug, height=700, face="white"):
        self.slug, self.h = slug, height
        self.fig = plt.figure(figsize=(6.8, 6.8 * height / 1200), facecolor=face)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1200), ylim=(height, 0))
        self.ax.set_axis_off()
        self.texts = []
        self.face = face

    def text(self, x, y, s, size=21, color=INK, weight="normal", ha="left", va="top", family=None, rotation=0):
        t = self.ax.text(x, y, s, fontsize=size * PT, color=color, weight=weight,
                         ha=ha, va=va, linespacing=1.24, fontfamily=family,
                         rotation=rotation)
        self.texts.append(t)
        return t

    def line(self, pts, color=INK, lw=1.2, ls="-", alpha=1):
        p = np.array(pts)
        self.ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=ls, alpha=alpha,
                     solid_capstyle="round", clip_on=False)

    def rect(self, x, y, w, h, fill="white", edge=HAIR, lw=.7):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=lw))

    def poly(self, pts, fill="none", edge=INK, lw=1):
        self.ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, lw=lw))

    def circle(self, x, y, r, fill="white", edge=INK, lw=1.2):
        self.ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge, lw=lw, zorder=2.5))

    def arrow(self, p, q, color=INK, lw=1.1, rad=0, ls="-"):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                         linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls))

    def badge(self, x, y, i, r=16):
        self.circle(x, y, r, fill=COLORS[i], edge=COLORS[i], lw=.5)
        self.text(x, y, SHORT[i], 19, "white", "bold", "center", "center")

    def save(self, *, formats=("svg", "pdf", "png")):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        b = self.fig.bbox
        for t in self.texts:
            bb = t.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > b.width + 1 or bb.y1 > b.height + 1:
                raise ValueError(f"Text outside {self.slug}: {t.get_text()}")
        overlaps = []
        for i, t in enumerate(self.texts):
            a = t.get_window_extent(renderer)
            for u in self.texts[i+1:]:
                b2 = u.get_window_extent(renderer)
                if min(a.x1,b2.x1)-max(a.x0,b2.x0) > 1 and min(a.y1,b2.y1)-max(a.y0,b2.y0) > 1:
                    overlaps.append([t.get_text(), u.get_text()])
        for ext in formats:
            kw = {"dpi": 240} if ext == "png" else {}
            if ext == "svg":
                kw["metadata"] = {"Date": None, "Description": "Schematic conceptual diagram; geometry is illustrative."}
            self.fig.savefig(OUT / f"{self.slug}.{ext}", facecolor=self.face, **kw)
            if ext == "svg":
                target = OUT / f"{self.slug}.{ext}"
                target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
        TEXT_AUDIT.append({"variant": self.slug, "minimum_font_pt": min(t.get_fontsize() for t in self.texts),
                           "text_count": len(self.texts), "width_in": 6.8,
                           "height_in": self.fig.get_figheight(), "text_within_canvas": True,
                           "text_overlaps": overlaps})
        FIGURES.append(self)


def direction(c, x, y, w=120, h=75, color=COLORS[0]):
    c.arrow((x + 5, y + h - 5), (x + w, y + 4), color, 1.5)
    for u in np.linspace(.12, .84, 6):
        c.circle(x + u*w, y + h*(1-u), 3.7, color, color, .5)


def plane(c, x, y, w=135, h=80, color=COLORS[4]):
    c.poly([(x, y+h*.73), (x+w*.35, y), (x+w, y+h*.27), (x+w*.65, y+h)], tint(color, .08), color, .9)
    th = np.linspace(0, 2*math.pi, 100)
    c.line(list(zip(x+w*.5+w*.24*np.cos(th), y+h*.5+h*.21*np.sin(th))), color, 1.5)


def decode(c, x, y, w=145, h=80, color=COLORS[0], label=False):
    ts = np.linspace(-1.1, 1.1, 13)
    for k, t in enumerate(ts):
        xx, yy = x + (t+1.1)/2.2*w*.58, y + h*.15 + t*t*h*.32
        c.circle(xx, yy, 3.6, tint(color, .4+.6*k/12), color, .3)
        if k % 3 == 0:
            c.line([(xx, yy+6), (x+(t+1.1)/2.2*w*.58, y+h*.87)], color, .5, ":", .65)
    c.arrow((x-3, y+h*.87), (x+w*.67, y+h*.87), color, .9)
    c.arrow((x+w*.72, y+h*.5), (x+w*.9, y+h*.5), color, .9)
    c.text(x+w*.96, y+h*.5, "y", 24, color, "normal", "center", "center", family="DejaVu Serif")
    if label:
        c.text(x+w*.26, y+h+9, "affine decoder", 18, MUTED, ha="center")


def relations(c, x, y, w=150, h=70, color=COLORS[4]):
    p = [(x, y+h*.75), (x+w*.22, y), (x+w*.4, y+h*.75)]
    q = [(x+w*.66, y+h*.7), (x+w*.84, y+h*.12), (x+w, y+h*.7)]
    for pts in [p, q]:
        c.line(pts+[pts[0]], color, 1.0)
        for px, py in pts:
            c.circle(px, py, 4.8, color, color, .5)
    for a, b in zip(p, q):
        c.line([a, b], MUTED, .55, ":", .5)


def computation(c, x, y, w=160, h=65, color=COLORS[5]):
    c.rect(x, y+h*.32, w*.25, h*.5, tint(color), color)
    c.text(x+w*.125, y+h*.57, "h", 24, color, ha="center", va="center", family="DejaVu Serif")
    c.arrow((x+w*.27, y+h*.57), (x+w*.55, y+h*.57), color)
    c.circle(x+w*.7, y+h*.57, h*.27, tint(color), color)
    c.text(x+w*.7, y+h*.57, "f", 24, color, ha="center", va="center", family="DejaVu Serif")
    c.arrow((x+w*.83, y+h*.57), (x+w*1.05, y+h*.57), color)
    c.text(x+w*1.1, y+h*.57, "y", 24, color, va="center", family="DejaVu Serif")
    c.arrow((x-w*.3, y+h*.57), (x-w*.02, y+h*.57), color)


