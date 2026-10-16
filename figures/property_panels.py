#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from itertools import combinations
from pathlib import Path
import re
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import deletion_style as common

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-04-figure7-three-panel-drafts"
OUT = ROOT / "out/figures" / STEM
DATA = OUT / "inputs"
W, H = 7.15, 3.05
INK, GRAY, LINE = "#25231f", "#73706a", "#d8d3ca"
BLUE, GOLD, PLUM = "#35618d", "#8c6000", "#83546b"
CMAP = LinearSegmentedColormap.from_list("accuracy", ["#ffffff", "#e5edf4", BLUE])



def text(ax, x, y, s, size=6.3, **kw):
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=kw.pop("color", INK), **kw)


def parent(fig, x, width):
    ax = fig.add_axes([x, .045, width, .825])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax


def inset(ax, x, y, width, height):
    box = ax.get_position()
    return ax.figure.add_axes([box.x0+x*box.width, box.y0+y*box.height,
                              width*box.width, height*box.height])


def header(fig, x, title, subtitle):
    fig.text(x, .956, title, fontsize=7.6, fontweight="bold", va="top", color=INK)
    fig.text(x, .904, subtitle, fontsize=6, va="top", color=GRAY)


def family(prop):
    if prop.startswith("divisible_"):
        return 0
    if prop.startswith("residue_"):
        return 1
    if prop in {"primality", "palindrome", "parity"}:
        return 2
    return 3


def matrix_ink(mean):
    rgb=CMAP(mean/100)[:3]
    channels=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
    luminance=sum(w*v for w,v in zip([.2126,.7152,.0722],channels,strict=True))
    return "white" if luminance<.179 else "black"


def scatter(ax, frame):
    plot = inset(ax, .15, .215, .77, .72)
    plot.set(xlim=(-.015, 1.04), ylim=(-.015, 1.04), aspect="equal")
    plot.plot([0, 1], [0, 1], color=LINE, lw=.8, ls=(0, (3, 3)), zorder=0)
    plot.set_xticks([0, .5, 1], ["0", ".5", "1"])
    plot.set_yticks([0, .5, 1], ["0", ".5", "1"])
    plot.tick_params(length=2, pad=2, labelsize=6)
    plot.set_xlabel("Full activation", fontsize=6.7, labelpad=2)
    plot.set_ylabel("Rank-41 basis", fontsize=6.7, labelpad=2)
    colors = [BLUE, GOLD, INK, PLUM]
    markers = ["o", "s", "D", "^"]
    for r in frame.itertuples():
        c, m = colors[family(r.property_id)], markers[family(r.property_id)]
        plot.errorbar(r.full_activation_mean, r.rank41_mean,
                      xerr=[[max(0, r.full_activation_mean-r.full_activation_min)],
                            [max(0, r.full_activation_max-r.full_activation_mean)]],
                      yerr=[[max(0, r.rank41_mean-r.rank41_min)],
                            [max(0, r.rank41_max-r.rank41_mean)]],
                      fmt=m, ms=3.6, color=c, ecolor=c, elinewidth=.45, capsize=1,
                      alpha=.8, markeredgewidth=.5, zorder=3)
    labels = {
        "residue_mod_4": ("Residue mod 4", (.23, .42)),
        "residue_mod_3": ("Residue mod 3", (.21, .76)),
        "divisible_by_25": ("Divisible by 25", (.64, .26)),
        "palindrome": ("Palindrome", (.87, .43)),
        "primality": ("Primality", (.92, .73)),
        "parity": ("Parity", (.83, 1.005)),
    }
    for prop, (label, position) in labels.items():
        r = frame.set_index("property_id").loc[prop]
        plot.annotate(label, (r.full_activation_mean, r.rank41_mean), position,
                      fontsize=5.6, color=INK, ha="center", va="center",
                      arrowprops=dict(arrowstyle="-", lw=.45, color=GRAY),
                      bbox=dict(facecolor="white", edgecolor="none", pad=.35))
    for i, label in enumerate(["Divisibility", "Residues", "Other", "Digits"]):
        x, y = .12 + (i % 2)*.44, .078-(i//2)*.052
        ax.plot(x, y, markers[i], ms=3.5, color=colors[i], transform=ax.transAxes)
        text(ax, x+.035, y, label, size=5.7, va="center")
    text(ax, .5, -.015, "19 properties · balanced accuracy · ranges across splits", size=5.5,
         ha="center", color=GRAY)


def matrix(ax, frame, shaded=True):
    rows = frame.sort_values("full_activation_mean", kind="stable")
    text(ax, .0, .98, "Property", size=6.2)
    if shaded:
        text(ax, .408, .98, "Chance", size=5.4, ha="center")
        columns = [("Full activation", .58, "full_activation"), ("Rank 41", .863, "rank41")]
        width = .253
    else:
        columns = [("Full activation", .54, "full_activation"), ("Rank 41", .845, "rank41")]
        width = .292
    for label, x, _ in columns:
        text(ax, x, .98, label, size=6.2, ha="center", color=BLUE if shaded else INK)
    for i, r in enumerate(rows.itertuples()):
        y = .944 - i*.046
        label = r.display_name.replace("Distinct-digit count", "Distinct digits")
        text(ax, 0, y, label, size=5.9, va="center")
        if shaded:
            text(ax, .408, y, f"{r.chance*100:.0f}", size=5.6, ha="center", va="center", color=GRAY)
        elif i%2 == 0:
            ax.add_patch(Rectangle((.415, y-.0215), .568, .043, facecolor="#f5f4f1",
                                  edgecolor="none", transform=ax.transAxes, zorder=0))
        for _, x, prefix in columns:
            mean, lo, hi = [getattr(r, prefix+suffix)*100 for suffix in ["_mean", "_min", "_max"]]
            if shaded:
                ax.add_patch(Rectangle((x-width/2, y-.0215), width, .043,
                                      facecolor=CMAP(mean/100), edgecolor="white", lw=.3,
                                      transform=ax.transAxes))
            s = f"{mean:.1f} [{lo:.0f}–{hi:.0f}]"
            text(ax, x, y, s, size=5.7, ha="center", va="center",
                 color=matrix_ink(mean) if shaded else INK)
    ax.plot([0, .983], [.035, .035], color=LINE, lw=.6, transform=ax.transAxes)
    text(ax, .5, .006, "Balanced accuracy (%) · mean [split min–max]", size=5.5,
         ha="center", color=GRAY)
    if not shaded:
        text(ax, .5, -.035, "Chance: 10% digits; 25% mod 4; 33% count/mod 3; 50% binary", size=5,
             ha="center", color=GRAY)


def profile(ax, frame):
    rows = frame.sort_values("full_activation_mean", kind="stable").reset_index(drop=True)
    p = inset(ax, .11, .44, .85, .48)
    for prefix, dx, marker, face in [("full_activation", -.15, "o", "white"),
                                     ("rank41", .15, "s", INK)]:
        mean = rows[prefix+"_mean"].to_numpy()
        p.errorbar(np.arange(19)+dx, mean,
                   yerr=[np.maximum(0, mean-rows[prefix+"_min"].to_numpy()),
                         np.maximum(0, rows[prefix+"_max"].to_numpy()-mean)],
                   fmt=marker, ms=2.6, mfc=face, mec=INK, mew=.5, ecolor=GRAY,
                   elinewidth=.5, capsize=1, color=INK, zorder=3)
    p.scatter(np.arange(19), rows.chance, marker="x", s=6, color=GRAY, linewidth=.5)
    p.set(xlim=(-.7, 18.7), ylim=(-.015, 1.06))
    p.set_xticks(np.arange(19), [str(i+1) for i in range(19)])
    p.set_yticks([0, .5, 1], ["0", ".5", "1"])
    p.tick_params(axis="x", length=0, labelsize=4.8, pad=2)
    p.tick_params(axis="y", length=2, labelsize=5.6, pad=2)
    p.set_ylabel("Balanced accuracy", fontsize=6.1, labelpad=2)
    for x, m, fill, label in [(.09, "o", "white", "Full"), (.40, "s", INK, "Rank 41"), (.75, "x", GRAY, "Chance")]:
        ax.plot(x, .984, m, ms=3, mfc=fill, mec=INK, mew=.5, transform=ax.transAxes)
        text(ax, x+.035, .984, label, size=5.7, va="center")
    for i, r in enumerate(rows.itertuples()):
        column, row = i//10, i%10
        label = r.display_name.replace("Divisible by", "Div. by").replace("Distinct-digit count", "Distinct digits")
        text(ax, .015+column*.51, .346-row*.033, f"{i+1:2}. {label}", size=5.4, va="center")
    text(ax, .5, -.035, "Means and ranges across 6–10 value-disjoint splits", size=5.5,
         ha="center", color=GRAY)


def table_rows(ax, rows, top, step=.059, size=5.8):
    for i, r in enumerate(rows.itertuples()):
        y = top-i*step
        text(ax, .01, y, r.label, size=size, va="center",
             fontweight="bold" if r.label=="Parity direction" else "normal")
        text(ax, .98, y, f"{r.contradictions}/{r.n}", size=size, ha="right", va="center",
             color=GOLD if r.label=="Parity direction" else INK,
             fontweight="bold" if r.label=="Parity direction" else "normal")


def quantitative(ax, counts, style):
    llama = counts.loc[counts.experiment=="classification"]
    qwen = counts.loc[counts.experiment=="enumeration"]
    for top, title, model in [(.965, "Classification", "Llama-3.1-8B · 100 numerals"),
                              (.493, "Enumeration", "Qwen3.6-27B · 50 lists")]:
        text(ax, .01, top, title, size=6.5, fontweight="bold")
        text(ax, .01, top-.051, model, size=5.7, color=GRAY)
    ax.plot([.01, .98], [.54, .54], lw=.6, color=LINE, transform=ax.transAxes)
    if style==0:
        table_rows(ax, llama, .800, .089, 6)
        table_rows(ax, qwen, .352, .062, 6)
        text(ax, .01, .018, "Contradictions / evaluated inputs", size=5.5, color=GRAY)
    elif style==1:
        for frame, nrows, base in [(llama, 10, .626), (qwen, 5, .247)]:
            r=frame.loc[frame.label=="Parity direction"].iloc[0]
            text(ax, .01, base+.245 if nrows==10 else base+.155,
                 f"{r.contradictions}/{r.n} contradictions", size=7, fontweight="bold", color=GOLD)
            for k in range(r.n):
                x, y = .04+(k%10)*.084, base+(nrows-1-k//10)*.0185
                ax.add_patch(Rectangle((x,y), .067, .0128, transform=ax.transAxes,
                                      facecolor=GOLD if k<r.contradictions else "white",
                                      edgecolor=GOLD if k<r.contradictions else GRAY, lw=.35))
            others=frame.loc[frame.label!="Parity direction"]
            table_rows(ax, others, .607 if nrows==10 else .221, .052, 5.8)
        text(ax, .01, .011, "Filled tile = one contradiction", size=5.5, color=GRAY)
    elif style==2:
        for frame, bottom in [(llama,.655),(qwen,.195)]:
            p=inset(ax,.18,bottom,.77,.195)
            p.set(xlim=(-.6,len(frame)-.4),ylim=(-3,105))
            p.set_yticks([0,50,100],["0","50","100"])
            p.tick_params(axis="y",length=2,labelsize=5,pad=2)
            labels=["B","P","R"] if len(frame)==3 else ["B","P","W","R","D"]
            p.set_xticks(range(len(frame)), labels)
            p.tick_params(axis="x",length=0,labelsize=5.1,pad=2)
            p.grid(axis="y",color=LINE,lw=.35)
            for i,r in enumerate(frame.itertuples()):
                p.plot(i, r.rate*100, "s" if r.label=="Parity direction" else "o",
                       ms=3.5,mfc=INK if r.label=="Parity direction" else "white",mec=INK,mew=.6)
                p.annotate(f"{r.contradictions}/{r.n}",(i,r.rate*100),xytext=(0,3),
                           textcoords="offset points",fontsize=5,ha="center")
        text(ax,.01,.116,"Observed contradiction rate (%)",size=5.4,color=GRAY)
        text(ax,.01,.069,"B baseline; P parity direction",size=5.1,color=GRAY)
        text(ax,.01,.036,"R random; W wrong sign",size=5.1,color=GRAY)
        text(ax,.01,.003,"D delimiter location",size=5.1,color=GRAY)
    else:
        for frame, y in [(llama,.765),(qwen,.323)]:
            r=frame.loc[frame.label=="Parity direction"].iloc[0]
            text(ax,.01,y,f"{r.contradictions}/{r.n}",size=17,color=GOLD,va="center")
            text(ax,.64,y,"Parity\ndirection",size=5.8,va="center")
            table_rows(ax,frame.loc[frame.label!="Parity direction"],y-.110,.052,5.8)
        text(ax,.01,.010,"Contradictions on fixed test inputs",size=5.4,color=GRAY)


def prompt_title(ax, y, title, model):
    pack=HPacker(children=[TextArea(title,textprops=dict(fontfamily="TeX Gyre Heros",fontsize=6.2,
                                                         fontweight="bold",color=INK)),
                           TextArea(f" ({model})",textprops=dict(fontfamily="TeX Gyre Heros",fontsize=5.8,color=GRAY))],
                 align="baseline",pad=0,sep=0)
    ax.add_artist(AnnotationBbox(pack,(.035,y),xycoords=ax.transAxes,box_alignment=(0,.5),
                                frameon=False,pad=0))


def examples(ax, source, style):
    faces = [("#f2f6fa","#fcf8ee"),("#f5f5f5","#f5f5f5"),("white","white"),("white","white")][style]
    for y,h,face,color in [(.51,.483,faces[0],BLUE),(.005,.472,faces[1],GOLD)]:
        if style<2:
            ax.add_patch(Rectangle((0,y),1,h,facecolor=face,edgecolor=LINE,lw=.5,
                                  transform=ax.transAxes))
            if style==1:
                ax.plot([0,0],[y,y+h],color=color,lw=1.5,transform=ax.transAxes)
        else:
            ax.plot([0,1],[y+h,y+h],color=INK if style==2 else color,lw=.65,transform=ax.transAxes)
    c,e=source["classification"],source["enumeration"]
    prompt_title(ax,.956,"Classification prompt",c["model"])
    text(ax,.035,.899,"Add zero; report even or odd (10-shot).",size=5.9)
    text(ax,.035,.818,c["prompt_suffix"],size=5.8,fontfamily="monospace",va="center",linespacing=1.3)
    for x,label,out,color in [(.035,"Baseline",c["baseline_output"],INK),
                               (.55,"Parity edit",c["edited_output"],GOLD if style!=2 else INK)]:
        text(ax,x,.716,label,size=5.8,color=GRAY)
        text(ax,x,.635,out,size=7,fontfamily="monospace",va="center",linespacing=1.2,color=color)
    text(ax,.035,.538,"Same numeral, opposite parity label",size=5.7,color=GRAY)
    prompt_title(ax,.437,"Enumeration prompt",e["model"])
    text(ax,.035,.359,"Select every odd number.\nCopy exactly, preserving input order.",
         size=5.9,va="center",linespacing=1.15)
    for i,value in enumerate(e["source_numbers"]):
        text(ax,.06+(i%5)*.191,.281-(i//5)*.052,str(value),size=6.4,fontfamily="monospace",
             color=GOLD if value in e["target_values"] and style!=2 else INK,ha="left",va="center")
    for y,label,values,color in [(.185,"Baseline",e["baseline_output"],INK),
                                 (.095,"Parity edit",e["edited_output"],GOLD if style!=2 else INK)]:
        text(ax,.035,y,label,size=5.7,color=GRAY)
        text(ax,.035,y-.041,", ".join(map(str,values)),size=5.8,fontfamily="monospace",color=color)
    text(ax,.035,.007,"634 is even; all other selections are preserved.",size=5.4,color=GRAY)
















