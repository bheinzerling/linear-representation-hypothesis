#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_cairo import RendererCairo, cairo
from matplotlib.lines import Line2D
from matplotlib.path import Path as ArtistPath
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Text3D
import numpy as np
import pandas as pd

import gen_fig_source_marked as source_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/main_layout"
WORK = ROOT / "out/space-optimization/after"
FONT = "TeX Gyre Heros"
INK, GRAY, LINE = "#25231f", "#73706a", "#d8d3ca"
PAPER_WIDTH_PT = 16 / 2.54 * 72
RECORDS = {}
FONT_RECORDS = {}

# The locked figures group uses cairocffi; Matplotlib's hatch code uses the
# corresponding pycairo enum spellings.
if not hasattr(cairo, "Content"):
    cairo.Content = SimpleNamespace(COLOR_ALPHA=cairo.CONTENT_COLOR_ALPHA)
if not hasattr(cairo, "Extend"):
    cairo.Extend = SimpleNamespace(REPEAT=cairo.EXTEND_REPEAT)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    for style in ("Regular", "Bold", "Italic", "Bold Italic"):
        path = subprocess.check_output(
            ["fc-match", "-f", "%{file}", f"{FONT}:style={style}"], text=True).strip()
        font_manager.fontManager.addfont(path)
        assert font_manager.FontProperties(fname=path).get_name() == FONT
        FONT_RECORDS[style] = {"file": Path(path).name, "sha256": digest(path)}
    font_manager.fontManager.addfont(source_style.SOURCE)
    FONT_RECORDS["Condensed"] = {"file": source_style.SOURCE.name,
                                  "sha256": digest(source_style.SOURCE)}
    font_dir = Path(mpl.get_data_path())/"fonts/ttf"
    for name in ("cmr10", "cmmi10", "cmsy10", "STIXGeneral", "DejaVuSansMono"):
        path = font_dir/(name+".ttf")
        FONT_RECORDS[name] = {"file": path.name, "sha256": digest(path)}
    mpl.rcParams.update({
        "font.family": FONT, "font.size": 8.5, "mathtext.fontset": "cm",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
        "axes.linewidth": .55, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GRAY, "ytick.color": GRAY, "svg.fonttype": "none",
        "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
        "hatch.linewidth": .4,
    })


def factor(fig):
    return fig.get_figwidth() * 72 / PAPER_WIDTH_PT


def body(fig, final_pt=7.5):
    return final_pt * factor(fig)


def fonts(fig):
    for text in fig.findobj(Text):
        if not any("mono" in f.lower() for f in text.get_fontfamily()):
            if not (text.get_gid() or "").startswith("source-"):
                text.set_fontfamily(FONT)
        text.set_math_fontfamily("cm")
    for ax in all_axes(fig):
        ax.patch.set_facecolor("none")


def all_axes(fig):
    def descend(axis):
        yield axis
        for child in axis.child_axes:
            yield from descend(child)
    for axis in fig.axes:
        yield from descend(axis)


def locate(axis, rect):
    axis.set_axes_locator(None)
    axis.set_position(rect)


def inset_rect(parent, rect):
    p = parent.get_position(original=True)
    x, y, width, height = rect
    return [p.x0 + x*p.width, p.y0 + y*p.height, width*p.width, height*p.height]


def axis_fonts(axis, size, tick=None):
    tick = size if tick is None else tick
    axis.tick_params(axis="both", labelsize=tick)
    axis.xaxis.label.set_fontsize(size)
    axis.yaxis.label.set_fontsize(size)


def fill_panel_width(fig, stem):
    """Place the framed panels across the full export width."""
    if "panels" not in RECORDS.get(stem, {}):
        return
    width = fig.get_figwidth()*72
    panels = RECORDS[stem]["panels"]
    left = min(p["frame_pt"][0] for p in panels)/width
    right = max(p["frame_pt"][2] for p in panels)/width
    edge = .4/width  # Keep the complete frame stroke inside the canvas.
    scale = (1-2*edge)/(right-left)

    def move_x(x):
        return edge+(np.asarray(x)-left)*scale

    fig.canvas.draw()
    positions = [(ax, ax.get_position(original=True).bounds) for ax in all_axes(fig)]
    for ax, (x, y, w, h) in positions:
        if getattr(ax, "_preserve_display_aspect", False):
            y -= h*(scale-1)/2
            h *= scale
        locate(ax, [float(move_x(x)), y, w*scale, h])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((float(move_x(x)), y))
    for artist in fig.artists:
        if isinstance(artist, Rectangle):
            artist.set_x(float(move_x(artist.get_x())))
            artist.set_width(artist.get_width()*scale)
        elif isinstance(artist, Line2D):
            artist.set_xdata(move_x(artist.get_xdata()))
    for legend in fig.legends:
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((float(move_x(anchor.x0)), anchor.y0,
                                   anchor.width*scale, anchor.height), transform=fig.transFigure)
    for p in panels:
        p["frame_pt"][0] = float(move_x(p["frame_pt"][0]/width))*width
        p["frame_pt"][2] = float(move_x(p["frame_pt"][2]/width))*width
    RECORDS[stem]["horizontal_fill"] = {
        "original_frame_extent_pt": [left*width, right*width],
        "frame_extent_pt": [.4, width-.4], "position_scale": scale,
    }


def fill_prism_width(fig, stem):
    """Fit the conceptual illustration by moving anchors and scaling geometry."""
    labels, _ = bounds(fig)
    left = min(labels, key=lambda t: t["bbox_pt"][0])
    right = max(labels, key=lambda t: t["bbox_pt"][0]+t["bbox_pt"][2])
    width = fig.get_figwidth()*72
    left_offset = left["bbox_pt"][0]-left["anchor_pt"][0]
    right_offset = right["bbox_pt"][0]+right["bbox_pt"][2]-right["anchor_pt"][0]
    scale = (width-2-right_offset+left_offset)/(right["anchor_pt"][0]-left["anchor_pt"][0])
    shift = 1-left_offset-scale*left["anchor_pt"][0]
    axis = fig.axes[0]
    lo, hi = axis.get_xlim()
    span = (hi-lo)/scale
    axis.set_xlim(lo-shift*span/width, lo-shift*span/width+span)
    top, bottom = axis.get_ylim()
    center, half = (top+bottom)/2, (top-bottom)/scale/2
    axis.set_ylim(center+half, center-half)
    RECORDS[stem] = {"horizontal_fill": {
        "original_ink_extent_pt": [left["bbox_pt"][0], right["bbox_pt"][0]+right["bbox_pt"][2]],
        "ink_extent_pt": [1, width-1], "position_scale": scale,
    }}


def panel_text_clearance(fig, stem, labels):
    """Keep labels inside their panel, including the outermost glyphs."""
    height = fig.get_figheight()*72
    rows = []
    for panel in RECORDS.get(stem, {}).get("panels", []):
        left, top, right, bottom = panel["frame_pt"]
        low, high = height-bottom, height-top
        margins = []
        for label in labels:
            x, y, w, h = label["bbox_pt"]
            if left <= x+w/2 <= right and low <= y+h/2 <= high:
                gap = min(x-left, right-x-w, y-low, high-y-h)
                assert gap >= .8, (stem, panel["letter"], label["label"], "label close to panel edge", gap)
                margins.append(gap)
        rows.append({"letter": panel["letter"], "minimum_text_margin_pt": min(margins)})
    return rows


def projected_labels(axis, entries):
    """Place labels beside projected points with fixed, readable offsets."""
    for label in list(axis.texts):
        label.remove()
    axis._projection_labels = []
    for label, point, offset, ha, va, color in entries:
        x, y, _ = proj3d.proj_transform(*point, axis.get_proj())
        annotation = axis.annotate(label, (x, y), xytext=tuple(body(axis.figure, v) for v in offset),
                                   textcoords="offset points", ha=ha, va=va,
                                   fontsize=body(axis.figure, 7.5), color=color,
                                   annotation_clip=False, zorder=20)
        axis._projection_labels.append((annotation, point, offset))


def annotation_clearance(fig):
    renderer = fig.canvas.get_renderer()
    records = []
    for axis in fig.axes:
        entries = getattr(axis, "_projection_labels", [])
        if not entries:
            continue
        segments = []
        for line in axis.lines:
            xx, yy, zz = line.get_data_3d()
            points = np.column_stack([xx, yy, zz])
            if len(points) > 1:
                segments.append(points)
        for collection in axis.collections:
            segments.extend(np.asarray(s) for s in getattr(collection, "_segments3d", []))
        screen_paths = []
        for segment in segments:
            xx, yy, _ = proj3d.proj_transform(*segment.T, axis.get_proj())
            screen_paths.append(ArtistPath(axis.transData.transform(np.column_stack([xx, yy]))))
        for label, point, offset in entries:
            xx, yy, _ = proj3d.proj_transform(*point, axis.get_proj())
            screen = axis.transData.transform((xx, yy))
            box = label.get_window_extent(renderer)
            distance = np.linalg.norm(screen-np.clip(screen, [box.x0, box.y0], [box.x1, box.y1]))*72/fig.dpi
            crosses = sum(path.intersects_bbox(box.padded(.3*fig.dpi/72), filled=False)
                          for path in screen_paths)
            assert distance >= body(fig, 3), (label.get_text(), "label touches its node", distance)
            assert not crosses, (label.get_text(), "label crosses a diagram line", crosses)
            records.append({"label": label.get_text(), "point": list(point),
                            "offset_final_pt": list(offset), "node_clearance_pt": distance,
                            "line_intersections": crosses})
    return records


def panel_headers(fig, stem, linebreaks=None, bottom_pt=None, frame_edges=None, title_overrides=None, source_overrides=None):
    spec = copy.deepcopy(next(s for s in source_style.SPEC["figures"] if s["stem"] == stem))
    scale = factor(fig)
    w, h = fig.get_size_inches()*72
    # The same sizes at paper width for every empirical figure.
    title_size, source_size = 8.5*scale, 6*scale
    rows = []
    for p in spec["panels"]:
        if title_overrides and p["letter"] in title_overrides:
            p["title"] = title_overrides[p["letter"]]
        if source_overrides and p["letter"] in source_overrides:
            p["source"] = source_overrides[p["letter"]]
        if bottom_pt is not None:
            p["bottom"] = bottom_pt
        if frame_edges and p["letter"] in frame_edges:
            old_left = p["left"]
            p["left"], p["right"] = frame_edges[p["letter"]]
            p["title_x"] += p["left"]-old_left
        colors = source_style.SPEC["palette"][p["source"]]
        title = p["title"]
        right = p["right"]-3*scale
        left = p["title_x"]
        lines = (linebreaks or {}).get(p["letter"]) or source_style.wrap_title(
            title, right-left, right-left, title_size, 700)
        assert " ".join(lines) == title, (stem, title, lines)
        assert all(source_style.width(line, title_size, 700) <= right-left for line in lines)
        assert len(lines) <= 3, (stem, title, lines)
        rect = [p["left"]/w, (h-p["bottom"])/h,
                (p["right"]-p["left"])/w, (p["bottom"]-p["top"])/h]
        fig.add_artist(Rectangle(rect[:2], *rect[2:], transform=fig.transFigure,
                                  facecolor=colors["tint"], edgecolor="none", zorder=-10))
        fig.add_artist(Rectangle(rect[:2], *rect[2:], transform=fig.transFigure,
                                  facecolor="none", edgecolor=colors["ink"],
                                  lw=.65*scale, zorder=5))
        source_baseline = p["top"]+2.5*scale+.8*source_size
        baseline = p["top"]+2.5*scale+1.2*source_size+.8*title_size
        for i, line in enumerate(lines):
            fig.text(left/w, 1-(baseline+i*title_size*1.08)/h, line,
                     fontsize=title_size, weight="bold", color="#000000",
                     va="baseline", gid=f"header-{p['letter']}-{i}", zorder=10)
        fig.text(right/w, 1-source_baseline/h, colors["label"],
                 fontsize=source_size, fontfamily="TeX Gyre Heros Cn",
                 ha="right", va="baseline", color=colors["ink"],
                 gid=f"source-{p['letter']}", zorder=10)
        rows.append({"letter": p["letter"], "source": p["source"],
                     "title": title, "title_lines": lines,
                     "source_baseline_from_top_pt": source_baseline,
                     "title_baseline_from_top_pt": baseline,
                     "frame_pt": [p[k] for k in ("left", "top", "right", "bottom")]})
    RECORDS[stem] = {"panels": rows}


def text(ax, x, y, label, size, **kwargs):
    return ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
                   color=kwargs.pop("color", INK), **kwargs)


def verify_inputs():
    """Pin the evidence independently of changes to layout and typography."""
    import claims as claims_source
    import analytic_geometry as analytic_source
    import coefficient_evidence as sufficiency_source
    paths = [claims_source.compact.ART,
             ROOT/"data/figures/lrh_uptake_and_use/fulltext_comparison_counts.csv",
             ROOT/"data/figures/panel_sources/spec.json",
             ROOT/"data/figures/causal_use/arditi_snapshot_manifest.json",
             ROOT/"data/figures/refusal/author_evaluations.json",
             ROOT/"data/figures/literature_interventions/literature_data.json",
             analytic_source.OUT/"analytic_cases.json"]
    for directory in ["decodability_efficiency", "robustness_invariance_structure",
                      "pls_grid", "deletion",
                      "number_property_evidence_main"]:
        folder = ROOT/"data/figures"/directory
        paths.extend(p for p in folder.iterdir() if p.suffix in {".json", ".csv"})
    paths.append(ROOT/"data/figures/decodability_efficiency/reference_panels.py")
    paths.extend(ROOT/"data/figures/numerical_sufficiency_atlas"/name for name in
                 ("steering_accuracy_rows.csv.gz", "steering_accuracy.provenance.json"))
    paths.append(ROOT/"figures/individual_prompt_data.py")
    paths.extend((ROOT/"data/figures/numerical_sufficiency_atlas/individual_prompt_fits").iterdir())
    paths.extend(ROOT/"data/figures/conditional_uncertainty"/name for name in
                 ("probe_metrics.csv", "steering_accuracy.csv", "provenance.json"))
    paths.extend(p for p in sufficiency_source.INPUT.iterdir()
                 if p.suffix in {".json", ".csv", ".gz"} and p.name != "paper_figure_sizes.json")
    actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}
    path = DATA/"input_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == actual, "Frozen figure inputs changed; review the evidence manifest."
    else:
        path.write_text(json.dumps(actual, indent=2)+"\n")
    return actual


def mentions():
    import mentions_panel as old
    old.setup_style()
    # Save one 13.6 pt text line at the paper's 7.7 cm column width.
    height_pt = 174.221
    bottom_pt, top_pt = .195*189.277, .020*189.277
    fig = plt.figure(figsize=(242.537/72, height_pt/72))
    ax = fig.add_axes([.149, bottom_pt/height_pt, .842,
                      1-(bottom_pt+top_pt)/height_pt])
    rows = old.read_csv("fulltext_comparison_counts.csv")
    old.plot_broad_search(ax, rows)
    labels = ["Google Scholar (Jul. 2026)", "Google Scholar (Dec. 2025)",
              "OpenAlex (Jul. 2026)", "Partial 2026 counts"]
    legend = ax.get_legend()
    for t, label in zip(legend.get_texts(), labels, strict=True):
        t.set_text(label)
        t.set_fontsize(7.4)
    legend._legend_box.sep = 3
    axis_fonts(ax, 9.0, 8.5)
    for t in ax.texts:
        t.set_fontsize(8.1)
    return fig


def claims():
    import claims as old
    compact, base = old.compact, old.compact.base
    old.register_fonts()
    base.NAMES, base.SHORT = old.CLAIM_NAMES, old.CLAIM_KEYS
    compact.CLAIM_ROWS = old.CLAIM_ROWS
    layout = next(v for v in compact.LAYOUTS if v["id"] == "03-tighter-stack")
    canvas, geometry = compact.diagram(layout, slug="fig_claims_space_optimized",
                                       save=False, badge_family=FONT)
    for patch in canvas.ax.patches:
        if patch.get_gid() == "claim-initial-SC":
            center = (978, old.CLAIM_ROWS[4])
            patch.set_transform(patch.get_transform()-canvas.ax.transData
                                + Affine2D().translate(-center[0], -center[1])
                                .scale(.75).translate(*center)+canvas.ax.transData)
    for t in canvas.texts:
        label = t.get_text()
        if label in old.HEADINGS:
            t.set_fontsize(28*base.PT)
        elif label in {"direction", "linear subspace", "affine decoder"}:
            t.set_text("Linear decoder" if label == "affine decoder" else label.capitalize())
            t.set_fontsize(23.5*base.PT)
        elif label == "Spectrum of LRH claims":
            t.set_text("LRH claim categories")
        elif label in base.NAMES:
            t.set_weight("normal")
            t.set_fontsize(23*base.PT)
        elif label in {"Accessibility", "Structure", "Computation"}:
            if label == "Accessibility":
                t.set_text("Decodability")
            t.set_fontsize(23.5*base.PT)
        elif label == "the LRH":
            t.set_weight("normal")
            t.set_fontsize(24*base.PT)
        elif label in {"h", "f", "y"}:
            t.set_text(f"${label}$")
            t.set_fontsize(23*base.PT)
    # Uniformly enlarge the illustration into the unused canvas margins.
    # Text sizes stay fixed; the prism image keeps its physical aspect ratio.
    canvas.ax.set_xlim(27.06, 1180.34)
    # Match the previous printed float height with full-text-width inclusion.
    canvas.fig.set_figheight(canvas.fig.get_figheight()*.98)
    half_height = 490*(1180.34-27.06)/1200/2*.98
    canvas.ax.set_ylim(245+half_height, 245-half_height)
    # Preserve the prism image, all six examples and the evenly spaced claims.
    return canvas.fig


def decodability():
    import decodability_panels as old
    reference, geometry, provenance, metrics, capacity, manifest = old.load_inputs()
    height_pt = 156.8
    fig = plt.figure(figsize=(7.15, height_pt/72))
    parents = {}
    rects = {"a": [.009, .085, .164, .77], "b": [.185, .04, .225, .79],
             "c": [.422, .035, .276, .81], "d": [.717, .035, .276, .81]}
    for key, rect in rects.items():
        parents[key] = fig.add_axes(rect)
    a, b, c, d = [parents[k] for k in "abcd"]
    reference.draw_local_linearization(a)
    a.set_title("", loc="left")
    for t in a.texts:
        label = t.get_text()
        t.set_fontsize(body(fig, 7.1))
        if label == "evaluation\nregion":
            t.set_text("evaluation region")
            t.set_position((.08, .97))
            t.set_fontsize(body(fig, 6.5))
        elif label == "affine\napproximation":
            t.set_text("linear\napproximation")
            t.set_position((.38, -.13))
        elif label.startswith("activation"):
            t.set_text("activation (1D slice)")
            t.set_position((0, -1.11))
            t.set_fontsize(body(fig, 6.5))
    reference.draw_probe_outputs(b, geometry, provenance, footer_offset=.045)
    b.get_legend().remove()
    plot, cbar = b.child_axes
    locate(plot, inset_rect(b, [.005, .09, .99, .87]))
    locate(cbar, inset_rect(b, [.31, .035, .46, .030]))
    cbar.tick_params(labelsize=body(fig, 7), length=0, pad=2)
    for t in b.texts:
        if t.get_text() == "value":
            t.set_position((.20, .047))
            t.set_fontsize(body(fig, 7))
    for t in plot.texts:
        t.set_fontsize(body(fig, 7.3))
    old.draw_metrics(c, metrics)
    upper, lower = c.child_axes
    # Fixed physical plot heights leave room for the lower metric legend.
    # The lower chart sits farther down within the shorter panel.
    c_position = c.get_position(original=True)
    chart_left = c_position.x0 + .12*c_position.width
    chart_width = .85*c_position.width
    locate(upper, [chart_left, 77.6/height_pt, chart_width, 45/height_pt])
    locate(lower, [chart_left, 20.5/height_pt, chart_width, 40/height_pt])
    for axis, labels in [(upper, [r"$R^2$", "Exact accuracy"]),
                         (lower, ["MAE", "Median AE"])]:
        for t, label in zip(axis.get_legend().get_texts(), labels, strict=True):
            t.set_text(label)
            t.set_fontsize(body(fig, 7))
        axis_fonts(axis, body(fig), body(fig, 7))
        for t in axis.texts:
            t.set_fontsize(body(fig, 7))
    lower.tick_params(axis="x", labelsize=body(fig, 7), pad=2)
    capacity_axis = old.draw_capacity(d, capacity)
    # Preserve the displayed bounds of the current draft independently of any
    # concurrent edits to the older, standalone generator.
    capacity_axis.set_xlim(1, 4600)
    capacity_axis.set_xticks([1, 16, 256, 4096], ["1", "16", "256", "4096"])
    capacity_rect = inset_rect(d, [.20, 0, .745, 1])
    # Align the axis title with panel (c)'s two-line method labels.
    locate(capacity_axis, [capacity_rect[0], 27.1/height_pt,
                          capacity_rect[2], 79.12/height_pt])
    capacity_axis.set_xlabel("Method-specific complexity$^\\dagger$", fontsize=body(fig, 7), labelpad=0)
    capacity_axis.set_ylabel("MAE", fontsize=body(fig), labelpad=2)
    capacity_axis.tick_params(labelsize=body(fig, 7))
    for t in d.get_legend().get_texts():
        t.set_fontsize(body(fig, 7))
    d_position = d.get_position(original=True)
    legend_y = ((height_pt-48.48)/height_pt-d_position.y0)/d_position.height
    d.get_legend().set_bbox_to_anchor((.55, legend_y))
    panel_headers(fig, "fig_decodability_efficiency", bottom_pt=height_pt-.7)
    RECORDS["fig_decodability_efficiency"]["vertical_layout_pt"] = {
        "b_plot_bottom": plot.get_position(original=True).y0*height_pt,
        "c_upper_height": 45, "c_lower_height": 40,
        "c_chart_gap": 77.6-(20.5+40),
        "d_plot_height": 79.12,
    }
    return fig


def robustness():
    import transfer_panels as old
    fig = plt.figure(figsize=old.FIGURE_SIZE)
    sources = json.loads((old.DATA/"literature_sources.json").read_text())
    factual = json.loads((old.DATA/"factual_relation_faithfulness.json").read_text())
    assert [r["faithfulness_display_percent"] for r in factual["rows"]] == [88, 59, 6]
    old.sentiment(fig)
    old.sae_feature_matching(fig, sources["paulo_belrose"])
    old.invariance(fig)
    old.structural_hierarchy(fig)
    # A new compact factual panel preserves all three measured bars/baselines.
    left, height = old.COLUMN_LEFTS[2], fig.get_figheight()
    ax = fig.add_axes([left, .025/height, old.COLUMN_WIDTH, 1.475/height])
    wp = old.COLUMN_WIDTH*fig.get_figwidth()*72
    ax.set(xlim=(0, wp), ylim=(1.475*72, 0))
    ax.axis("off")
    ax.text(0, 11, "Factual relations", fontsize=8.5)
    ax.text(wp, 11, factual["model"], fontsize=6.5, ha="right", color=GRAY)
    ax.text(0, 30, "Miles Davis", fontsize=7.5)
    ax.text(wp, 30, "trumpet", fontsize=7.5, ha="right")
    ax.add_patch(FancyArrowPatch((45, 27.5), (100, 27.5), arrowstyle="-|>",
                                 mutation_scale=7.5, lw=.9, color=old.BLUE,
                                 shrinkA=0, shrinkB=0))
    ax.text(72.5, 22, "plays the", fontsize=7, color=old.BLUE, ha="center")
    ax.text(wp/2, 43, r"$\hat h_o=W_r h_s+b_r$", fontsize=9.2, ha="center")
    ax.text(0, 54.5, "Faithfulness (%)", fontsize=7, color=GRAY)
    for y, row in zip([65.5, 81.5, 97.5], factual["rows"], strict=True):
        value, baseline = [row[k] for k in ("faithfulness_from_vector_geometry", "random_guess_from_vector_geometry")]
        assert 0 <= baseline <= value <= 1
        assert round(value*100) == row["faithfulness_display_percent"]
        ax.text(0, y, row["label"], fontsize=7.5)
        ax.text(wp, y, f"≈{row['faithfulness_display_percent']}%", fontsize=7.5,
                color=old.BLUE, ha="right")
        ax.add_patch(Rectangle((0, y+3), wp-1, 5, color="#EEEAE2", lw=0))
        ax.add_patch(Rectangle((0, y+3), (wp-1)*value, 5, color=old.BLUE, lw=0))
        ax.plot([(wp-1)*baseline]*2, [y+2, y+9], color=old.RED, lw=.9)
    sent, sae, cos, transfer, tree, factual_ax = fig.axes
    # Uniform enlargement preserves the projected angles between directions.
    sentiment_scale = .288/.276
    locate(sent, [.025, 1.66/height, .288, .96*sentiment_scale/height])
    for t in sent.texts:
        t.set_fontsize(7.5)
    # Place the training condition beside a shorter subheading.
    for t in list(fig.texts):
        if t.get_text() == "SAE feature matching":
            t.set_text("SAE features")
            t.set_fontsize(8.5)
        elif t.get_text() == "Trained on the same data":
            t.set_text("Same data")
            t.set_position((old.COLUMN_LEFTS[0]+old.COLUMN_WIDTH, 1.35/height))
            t.set_ha("right")
            t.set_fontsize(7)
        elif t.get_text() == "Hierarchical concepts":
            t.set_position((left, 2.77/height))
        elif t.get_text() == r"Probing the numeric value $b$":
            t.set_position((old.COLUMN_LEFTS[1], 2.81/height))
        elif t.get_text() == "Probe transfer":
            t.set_position((old.COLUMN_LEFTS[1], 1.15/height))
        elif "causal inner product" in t.get_text():
            t.set_fontsize(6.7)
            t.set_position((left+old.COLUMN_WIDTH/2, 1.66/height))
    locate(sae, [.018, .07/height, .296, 1.15/height])
    for t in sae.texts:
        if t.get_text().startswith("Only ≈"):
            t.set_text("≈30% shared\nacross seeds")
            t.set_position((.205, .34))
            t.set_fontsize(7.5)
        else:
            t.set_text(t.get_text().replace("\nfeatures", ""))
            t.set_fontsize(7.5)
    locate(cos, [old.COLUMN_LEFTS[1]+.10, 1.71/height, .166, .166*fig.get_figwidth()/height])
    for t in cos.texts:
        t.set_fontsize(7 if t.get_text() != "Cosine" else 6.5)
        if t.get_text() == "Cosine":
            t.set_position((.735, .925))
    cos.tick_params(axis="both", labelsize=7)
    cos.yaxis.label.set_fontsize(7)
    cos.child_axes[0].tick_params(labelsize=5.7)
    locate(transfer, [old.COLUMN_LEFTS[1]+.086, .32/height, .197, .72/height])
    transfer.set_ylabel("")
    axis_fonts(transfer, 7.5, 7)
    prompt_labels = {"a + b": "a + b =", "a − b": "a − b =", "b = b": "b ="}
    for axis in (cos, transfer):
        axis.set_xticklabels([prompt_labels.get(t.get_text(), t.get_text())
                             for t in axis.get_xticklabels()])
        axis.set_yticklabels([prompt_labels.get(t.get_text(), t.get_text())
                             for t in axis.get_yticklabels()])
    for legend in fig.legends:
        for t, label in zip(legend.get_texts(), ["Same", "Other"], strict=True):
            t.set_text(label)
            t.set_fontsize(6.5)
        legend.set_bbox_to_anchor((old.COLUMN_LEFTS[1]+old.COLUMN_WIDTH, 1.21/height))
        legend._loc = 1  # upper right, alongside the Probe transfer heading
    sent._preserve_display_aspect = True
    locate(tree, [left, 1.69/height, old.COLUMN_WIDTH, 1.02/height])
    tree.set_xlim(-.04, 2.36)
    for t in tree.texts:
        t.set_fontsize(7 if not t.get_text().startswith("$") else 7.5)
        if t.get_text() == "mammal" and t.get_position()[0] > 1:
            x, y = t.get_position()
            t.set_position((x-.15, y))
        elif t.get_text() == "dog" and t.get_position()[0] > 1:
            x, y = t.get_position()
            t.set_position((x-.13, y))
    for column_left in old.COLUMN_LEFTS:
        divider = 1.32 if column_left == old.COLUMN_LEFTS[1] else 1.50
        fig.add_artist(Line2D([column_left, column_left+old.COLUMN_WIDTH],
                             [divider/height]*2, transform=fig.transFigure, color=LINE, lw=.5))
    panel_headers(fig, "fig_robustness_invariance_structure")
    return fig


def numerical_paths(parent, summary):
    import plot_causal_use_numbers as old
    xs, y0, height = [.115, .3625, .61], .14, .76
    for x, name in zip(xs, ["Original", "Control", "Property"], strict=True):
        parent.plot([x, x], [y0, y0+height], color=LINE, lw=.6)
        text(parent, x, .973, name, 7.6, ha="center")
    for value in [0, 50, 100]:
        y = y0+height*value/100
        text(parent, 0, y, str(value), 7.5, va="center", color=GRAY)
        parent.plot([xs[0], xs[-1]], [y, y], color="#eceae5", lw=.45, zorder=0)
    label_y = y0+height*np.array([summary.loc[(t, "delete_property"), "accuracy"] for t in old.TASKS])
    for lower, upper in zip(np.argsort(label_y)[:-1], np.argsort(label_y)[1:]):
        label_y[upper] = max(label_y[upper], label_y[lower]+.084)
    for i, task in enumerate(old.TASKS):
        rows = summary.loc[task].loc[old.FAMILIES]
        yy = y0+height*rows.accuracy.to_numpy()
        parent.plot(xs, yy, color=old.COLORS[i], ls=(0, (2, 1)) if i in (1, 3) else "-",
                    marker=["o", "s", "o", "s", "D", "^", "v"][i], ms=2.8, mew=.4, lw=.9, zorder=2)
        parent.errorbar(xs, yy, yerr=np.array([rows.accuracy-rows.accuracy_low,
                                              rows.accuracy_high-rows.accuracy])*height,
                        fmt="none", color=old.COLORS[i], lw=.65, capsize=1.2, zorder=3)
        parent.plot([xs[-1]+.014, .672], [yy[-1], label_y[i]], color=old.COLORS[i], lw=.5)
        text(parent, .686, label_y[i], old.LABELS[i], 7.8, va="center",
             color="#8c6000" if i == 4 else old.COLORS[i], gid=f"numeric-task-{task}")
    text(parent, .3625, .015, "Accuracy (%)", 8, ha="center")


def refusal_panel(parent, data, models):
    import plot_causal_use_refusal as old
    # Enlarge the constructed geometry uniformly; move the formula and evidence
    # downward into the space released by the shorter model list.
    def point(x, y):
        return .085+1.2*(x-.14), .59+1.2*(y-.68)

    origin, original, edited = point(.14, .68), point(.50, .96), point(.50, .68)
    parent.plot([origin[0], point(.82, .68)[0]], [origin[1]]*2, color=GRAY, lw=.55)
    parent.plot([origin[0]]*2, [point(.14, .66)[1], point(.14, .98)[1]], color=GRAY, lw=.55)
    for start, end, color in [(origin, original, old.GOLD), (origin, edited, old.BLUE), (edited, original, GRAY)]:
        parent.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8.4,
                                         lw=1.1, color=color))
    parent.plot([original[0]]*2, [original[1], edited[1]], color=GRAY, ls=(0, (2, 2)), lw=.65)
    text(parent, *point(.53, .958), "$x$", 8.5, color=old.GOLD)
    text(parent, *point(.37, .71), "$x'$", 8.5, color=old.BLUE, ha="center")
    text(parent, *point(.79, .70), r"$\perp r$", 8, color=GRAY, ha="center")
    text(parent, *point(.53, .81), "$r$", 8, color=GRAY)
    text(parent, *point(.68, .94), "All layers\nall tokens", 7.5, color=GRAY, va="top", linespacing=1.15)
    text(parent, .5, .478, r"$x' = x - \hat r\hat r^{\!\top}x$", 11, ha="center")
    parent.plot([0, 1], [.424]*2, color=old.PALE, lw=.6)
    text(parent, 0, .356, "Refusal (%)", 8, weight="bold")
    text(parent, 1, .356, "Original → ablated", 7.5, ha="right", color=GRAY)
    for i, model in enumerate(models):
        y = .278-i*.08
        before, after = [data[model, c]["refusals"] for c in ["baseline", "ablation"]]
        text(parent, 0, y, model, 8, va="center")
        text(parent, .82, y, f"{before} → {after}", 8.5, ha="center", va="center",
             color=old.BLUE, gid=f"refusal-result-{i}")


def necessity():
    import causal_panels as old
    data = old.candidates.load_data()
    refusal, _ = old.refusal.load_data()
    numerical, _ = old.numerical.load_data()
    fig = plt.figure(figsize=old.FIGURE_SIZE)
    height = fig.get_figheight()
    parents = []
    for left in old.candidates.PANEL_X:
        ax = fig.add_axes([left, .06/height, old.candidates.PANEL_WIDTH, 1.82/height])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        parents.append(ax)
    subtitles = ["Arditi et al. (2024)", "Tigges et al. (2024) · Pythia-2.8B", "Llama-3.1-8B · rank 41"]
    for left, subtitle in zip(old.candidates.PANEL_X, subtitles, strict=True):
        fig.text(left, (height-.30)/height, subtitle, fontsize=7.5, va="top", color=GRAY)
    fig.text(old.candidates.PANEL_X[0]+old.candidates.PANEL_WIDTH,
             (height-.30)/height, "$n=100$/model", fontsize=7, color=GRAY, ha="right", va="top")
    models = ["Qwen 1.8B", "Yi 6B", "Gemma 2B", "Llama 3 8B"]
    refusal_panel(parents[0], refusal, models)
    old.candidates.tigges_accuracy(parents[1], data["tigges"])
    sentiment = parents[1].child_axes[0]
    for t in list(parents[1].texts):
        t.remove()
    locate(sentiment, inset_rect(parents[1], [.405, .172, .55, .82]))
    axis_fonts(sentiment, 8, 7.7)
    sentiment.tick_params(axis="y", labelsize=8)
    for t in sentiment.texts:
        t.set_fontsize(7.8)
    sentiment.text(47, -.32, "chance", fontsize=7.5, color=GRAY, ha="right", va="center")
    numerical_paths(parents[2], numerical)
    panel_headers(fig, "fig_causal_use")
    RECORDS["fig_causal_use"]["refusal_model_selection"] = {
        "models": models,
        "reason": "One model per family; retain Llama 3 from the two Llama generations.",
        "rates_percent": [{"model": model,
                            "original": refusal[model, "baseline"]["refusals"],
                            "ablated": refusal[model, "ablation"]["refusals"]}
                           for model in models],
    }
    RECORDS["fig_causal_use"]["schematic_uniform_scale"] = 1.2
    return fig


def property_panel_layout(ax):
    """Keep content spacing in points within Figure 6's shorter panels."""
    reference_height = .858*219.6
    height = ax.get_position().height*ax.figure.get_figheight()*72

    def y(value, raise_pt=0):
        return 1-((1-value)*reference_height-raise_pt)/height

    return y, reference_height/height


def accessibility(ax, frame, bar_width_reduction_pt):
    import property_bars as bars
    import property_panels as old
    fig = ax.figure
    y_at, height_ratio = property_panel_layout(ax)
    reference_width = .397*7.15*72
    panel_width = ax.get_position().width*fig.get_figwidth()*72
    start, width = .355*reference_width/panel_width, (.53*reference_width-bar_width_reduction_pt)/panel_width
    text(ax, 0, y_at(.965), "Property", body(fig), weight="bold")
    text(ax, start+width/2, y_at(.965), "Balanced accuracy (%)", body(fig, 7), ha="center")
    text(ax, .995, y_at(.965), "Mean", body(fig, 7), ha="right", color=GRAY)
    # Natural 0–100% axis; all min/max and chance positions remain exact.
    for value in [0, 50, 100]:
        x = start+width*value/100
        ax.plot([x, x], [y_at(.915), y_at(.900)], color=GRAY, lw=.45, transform=ax.transAxes)
        text(ax, x, y_at(.93), str(value), body(fig, 7), ha="center", color=GRAY)
    rows = frame.sort_values("full_activation_mean", kind="stable")
    for i, row in enumerate(rows.itertuples()):
        y = y_at(.868-i*.046, raise_pt=2.4)
        label = row.display_name.replace("Distinct-digit count", "Distinct digits")
        text(ax, 0, y, label, body(fig, 7.4), va="center")
        bars.rect(ax, start, y, width, .026*height_ratio, bars.TRACK)
        bars.rect(ax, start, y, width*row.full_activation_mean, .026*height_ratio, old.BLUE)
        bars.range_mark(ax, start, width, y, row.full_activation_min, row.full_activation_max,
                        cap=.0033*height_ratio, halo=True)
        bars.chance_tick(ax, start, width, y, row.chance, half=.016*height_ratio)
        text(ax, .995, y, f"{row.full_activation_mean*100:.1f}", body(fig, 7.2), ha="right", va="center")


def contradiction_counts(ax, frame):
    fig = ax.figure
    y_at, _ = property_panel_layout(ax)
    for top, title, model, exp, row_top, step in [
        (.915, "Classification", "Llama-3.1-8B · 100 numerals", "classification", .795, .10),
        (.48, "Enumeration", "Qwen3.6-27B · 50 lists", "enumeration", .355, .074)]:
        text(ax, .01, y_at(top, raise_pt=10), title, body(fig), weight="bold")
        text(ax, .01, y_at(top-.065, raise_pt=10), model, body(fig, 6.5), color=GRAY)
        for i, r in enumerate(frame.loc[frame.experiment == exp].itertuples()):
            y = y_at(row_top-i*step, raise_pt=10)
            bold = r.label == "Parity direction"
            text(ax, .01, y, r.label, body(fig, 7.4), va="center", weight="bold" if bold else "normal")
            text(ax, .98, y, f"{r.contradictions}/{r.n}", body(fig, 7.4), ha="right", va="center",
                 color="#8c6000" if bold else INK, weight="bold" if bold else "normal")
    ax.plot([.01, .98], [y_at(.535, raise_pt=10)]*2, color=LINE, lw=.6, transform=ax.transAxes)


def contradiction_examples(ax, source):
    fig = ax.figure
    y_at, height_ratio = property_panel_layout(ax)
    c, e = source["classification"], source["enumeration"]
    upper_bottom = y_at(.535, raise_pt=10)
    for y, height, color in [(.535, .465, "#f7f9fb"), (.002, .503, "#fdfbf5")]:
        bottom = upper_bottom if y == .535 else y_at(y, raise_pt=10)
        box_height = 1-upper_bottom if y == .535 else height*height_ratio
        ax.add_patch(Rectangle((0, bottom), 1, box_height, facecolor=color, edgecolor=LINE, lw=.5,
                              transform=ax.transAxes))
    # Models occupy the unused right side of each example's title row.
    for y, title, model in [(.958, "Classification", c["model"]), (.458, "Enumeration", e["model"])]:
        y = y_at(y, raise_pt=10 if title == "Enumeration" else 0)
        text(ax, .035, y, title, body(fig, 7.4), weight="bold")
        text(ax, .965, y, model, body(fig, 6.4), color=GRAY, ha="right")
    text(ax, .035, y_at(.888), "Add zero; report even or odd (10-shot).", body(fig, 6.8))
    text(ax, .035, y_at(.789), c["prompt_suffix"], body(fig, 6.8), fontfamily="monospace",
         va="center", linespacing=1.3)
    for x, label, out, color in [(.035, "Baseline", c["baseline_output"], INK),
                               (.55, "Parity edit", c["edited_output"], "#8c6000")]:
        text(ax, x, y_at(.686), label, body(fig, 7), color=GRAY)
        text(ax, x, y_at(.635), out.replace("\n", " "), body(fig, 9), fontfamily="monospace", va="center", color=color)
    text(ax, .035, y_at(.372, raise_pt=10), "Select every odd number.\nCopy exactly, preserving input order.",
         body(fig, 6.8), va="center", linespacing=1.15)
    for i, value in enumerate(e["source_numbers"]):
        text(ax, .055+(i%5)*.191, y_at(.287-(i//5)*.069, raise_pt=10), str(value), body(fig, 8.4),
             fontfamily="monospace", color="#8c6000" if value in e["target_values"] else INK,
             va="center")
    for y, label, values, color in [(.142, "Baseline", e["baseline_output"], INK),
                                  (.059, "Parity edit", e["edited_output"], "#8c6000")]:
        y = y_at(y, raise_pt=10)
        text(ax, .035, y, label, body(fig, 7), color=GRAY)
        text(ax, .27, y, ",".join(map(str, values)), body(fig, 7.2),
             fontfamily="monospace", color=color)


def properties():
    import property_evidence as old
    manifest = json.loads((old.DATA/"input_manifest.json").read_text())
    for name, value in manifest["files"].items():
        assert digest(old.DATA/name) == value, name
    frame = pd.read_csv(old.DATA/"accessibility.csv")
    counts = pd.read_csv(old.DATA/"contradiction_counts.csv")
    examples = json.loads((old.DATA/"examples.json").read_text())
    assert len(frame) == 19 and counts.contradictions.tolist() == [0, 93, 3, 0, 48, 10, 0, 3]
    height_pt, transfer_pt = 209.6, 18
    fig = plt.figure(figsize=(7.15, height_pt/72))
    transfer = transfer_pt/(7.15*72)
    content_height = height_pt-(1-.883)*219.6-2
    parents = []
    for x, width in zip([.012, .434-transfer, .677],
                        [.397-transfer, .216+transfer, .310], strict=True):
        ax = fig.add_axes([x, 2/height_pt, width, content_height/height_pt])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        parents.append(ax)
    accessibility(parents[0], frame, transfer_pt)
    contradiction_counts(parents[1], counts)
    contradiction_examples(parents[2], examples)
    panel_headers(fig, "fig_number_property_evidence_main", bottom_pt=height_pt-.6,
                  frame_edges={"a": (2.1776, 215.4232-transfer_pt),
                               "b": (219.4232-transfer_pt, 342.5196)})
    RECORDS["fig_number_property_evidence_main"]["layout_pt"] = {
        "width_transferred_a_to_b": transfer_pt, "bar_width_reduction": transfer_pt,
        "height_reduction": 10, "property_rows_raised": 2.4,
    }
    return fig


def sufficiency():
    import numerical_panels as old
    import numerical_steering_accuracy
    analytic = old.geometry.analytic_cases()
    frozen = json.loads((old.geometry.OUT/"analytic_cases.json").read_text())
    for name in ("helix_circular", "surface"):
        assert analytic[name] == frozen[name]
    evidence = old.load_evidence()
    accuracies = evidence["control"]
    fig = old.atlas(evidence)
    height_pt = fig.get_figheight()*72
    plot_clearance = body(fig, 5)/height_pt
    geometry_annotations = {r"$d(E)=1$", r"$P(E)=(1,1)$",
                            r"$F(E)=0$", r"$F(T)=1$"}
    for t in list(fig.texts):
        if t.get_position()[1] > .87 or t.get_text().startswith(("20 properties", "Circular edit leaves", "Two valid edits")):
            t.remove()
        elif t.get_text() in {"(a)", "(b)", "(c)", "(d)", "(e)", "(f)"} or t.get_text() in {"a", "b", "c", "d", "e", "f"}:
            t.remove()
        elif t.get_text() in {"Edited state", "Both properties\nmatch"}:
            t.remove()
        elif t.get_text().startswith("Original states"):
            t.remove()
        elif t.get_text() in geometry_annotations:
            t.remove()
    prediction, circular, steering, helix, coefficient_axis, surface = fig.axes
    locate(prediction, [.087, .615, .222, .305-plot_clearance])
    axis_fonts(prediction, body(fig), body(fig, 7))
    for t in prediction.get_legend().get_texts():
        t.set_fontsize(body(fig, 6.6))
        if t.get_text() == "Linear ridge":
            t.set_text("Scalar ridge")
    locate(circular, [.413, .600, .214, .335-plot_clearance])
    axis_fonts(circular, body(fig, 7.1), body(fig, 7))
    steering.clear()
    locate(steering, [.812, .635, .164, .276-plot_clearance])
    methods = ["linear", "circular_reg", "centroid_patch"]
    exact = 100*accuracies.loc[methods, "exact_accuracy"].to_numpy()
    lenient = 100*accuracies.loc[methods, "within10_accuracy"].to_numpy()
    y = np.arange(len(methods))
    for i, (a, b) in enumerate(zip(exact, lenient, strict=True)):
        steering.plot([a, b], [i, i], color=LINE, lw=1.2, zorder=1)
    steering.axvline(100*accuracies.loc["no_edit", "within10_accuracy"],
                     color=GRAY, ls=(0, (2, 2)), lw=.75, zorder=1)
    numerical_steering_accuracy.draw_accuracy_intervals(
        steering, accuracies, methods, exact_color=old.candidates.ORANGE,
        within10_color=old.candidates.TEAL, capsize=2.8)
    steering.scatter(exact, y, color=old.candidates.ORANGE, s=24,
                     marker="o", label="Exact", clip_on=False, zorder=3)
    steering.scatter(lenient, y, facecolors="none", edgecolors=old.candidates.TEAL,
                     linewidths=.9, s=22, marker="D", label="Within ±10", clip_on=False, zorder=3)
    steering.set(xlim=(0, 100), ylim=(2.55, -.65), xlabel="Steering accuracy (%)")
    steering.set_yticks(y, ["Scalar ridge", "Circular reg.", "PCA centroid shift"])
    steering.set_xticks([0, 50, 100])
    steering.tick_params(length=2)
    steering.spines[["top", "right"]].set_visible(False)
    axis_fonts(steering, body(fig), body(fig, 7))
    steering.tick_params(axis="y", labelsize=body(fig, 7))
    steering.legend(fontsize=body(fig, 7), loc="lower left", bbox_to_anchor=(-.82, -.42),
                    ncol=1, frameon=False, handletextpad=.2)
    locate(coefficient_axis, [.805, .135-plot_clearance, .170, .285])
    coefficient_axis.set_xlabel("Exact accuracy (%)", fontsize=body(fig), labelpad=2)
    coefficient_axis.tick_params(labelsize=body(fig, 7), pad=3)
    for axis, left, bottom in [(helix, .019, -.014), (surface, .351, .037)]:
        locate(axis, [left, bottom, .225, .373])
        axis.set_box_aspect(axis.get_box_aspect(), zoom=1.12)
    blue, teal, gold = old.geometry.BLUE, old.geometry.TEAL, old.geometry.GOLD
    projected_labels(helix, [
        ("Output", (0, 0, 1.23), (-6, -6), "right", "center", GRAY),
        ("Source", (1, 0, 0), (0, -7), "center", "top", blue),
        ("Desired output", (0, 1, 1), (4, 4), "left", "bottom", teal),
        ("Code matched", (0, 1, 0), (4, 5), "left", "bottom", teal),
        ("Edit\nOriginal output", (0, 1, 0), (4, -4), "left", "top", gold),
    ])
    projected_labels(surface, [
        ("Source", (0, 0, 0), (-2, -7), "center", "top", blue),
        ("Properties\nmatched", (1, 1, 0), (4, 5), "left", "bottom", teal),
        ("Sum\nOriginal output", (1, 1, 0), (4, -4), "left", "top", gold),
        ("Desired output", (1, 1, 1), (5, 4), "left", "bottom", teal),
        ("Edit 1", (1, 0, 0), (0, -7), "center", "top", teal),
        ("Edit 2", (0, 1, 0), (-5, 2), "right", "bottom", teal),
    ])
    panel_headers(fig, "fig_numerical_sufficiency_atlas", linebreaks={
        "d": ["(d) Circular edit leaves", "the old axial value"],
        "e": ["(e) Two valid edits have", "an invalid sum"]},
        title_overrides={"c": "(c) Steering accuracy",
                         "e": "(e) Two valid edits have an invalid sum",
                         "f": "(f) 20 properties, rank-41 basis"},
        source_overrides={"e": "schematic", "f": "ours"})
    RECORDS["fig_numerical_sufficiency_atlas"]["steering_accuracy"] = {
        "methods": methods, "exact_percent": exact.tolist(), "within10_percent": lenient.tolist(),
        "no_edit_within10_percent": 100*accuracies.loc["no_edit", "within10_accuracy"],
        "n_per_method": 2000, "tolerance_inclusive": 10,
        "uncertainty": {"level": .95, "unit": "source context", "clusters": 100,
                        "replicates": 10000, "seed": 1729,
                        "conditional_on": "fixed fitted probes, intervention procedures, and target set",
                        "zero_successes": "counts only; upper success bound unavailable",
                        "intervals": accuracies.loc[methods, ["exact_low", "exact_high",
                                      "within10_low", "within10_high"]].replace({np.nan: None}).to_dict(orient="index")},
    }
    RECORDS["fig_numerical_sufficiency_atlas"]["matched_individual_fits"] = {
        "manifest": old.individual_prompt_data.DATA.relative_to(ROOT).as_posix()+"/manifest.json",
        "manifest_sha256": digest(old.individual_prompt_data.DATA/"manifest.json"),
        "fit_ids": evidence["matched_fit_manifest"]["fit_ids"],
        "evaluation_bank": "canonical_dev", "decoding_source_prompts": 100,
        "steering_cases": 2000, "averaging": False,
    }
    return fig


BUILDERS = {
    1: ("fig_lrh_fulltext_mentions", mentions), 2: ("fig_claims", claims),
    3: ("fig_decodability_efficiency", decodability),
    4: ("fig_robustness_invariance_structure", robustness),
    5: ("fig_causal_use", necessity), 6: ("fig_number_property_evidence_main", properties),
    7: ("fig_numerical_sufficiency_atlas", sufficiency),
}


def bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels, outside = [], []
    hidden_ticks = {id(t) for ax in all_axes(fig) if not ax.axison
                    for axis in (ax.xaxis, ax.yaxis) for t in axis.findobj(Text)}
    for t in fig.findobj(Text):
        if id(t) in hidden_ticks or not t.get_visible() or not t.get_text() or (t.axes and not t.axes.get_visible()):
            continue
        # Hidden ticks on an axis turned off are layout internals, not artwork.
        if t.axes and not t.axes.axison and t not in t.axes.texts:
            continue
        if isinstance(t, Text3D):
            saved = t.get_position()
            px, py, _ = proj3d.proj_transform(*t.get_position_3d(), t.axes.get_proj())
            t.set_position((px, py))
            try:
                b = Text.get_window_extent(t, renderer)
            finally:
                t.set_position(saved)
        else:
            b = Text.get_window_extent(t, renderer)
        if b.width and b.height:
            labels.append({"label": t.get_text(), "font_pt": t.get_fontsize(),
                           "bbox_pt": [v*72/fig.dpi for v in b.bounds], "gid": t.get_gid(),
                           "angle": t.get_rotation(),
                           "anchor_pt": (t.get_transform().transform(t.get_unitless_position())*72/fig.dpi).tolist()})
            if b.x0 < -1.5 or b.y0 < -1.5 or b.x1 > fig.bbox.x1+1.5 or b.y1 > fig.bbox.y1+1.5:
                outside.append(t.get_text())
    return labels, outside


def collisions(labels):
    """Check ordinary text and distinguish parallel diagonal tick labels."""
    overlaps, parallel_ticks = [], []
    for index, a in enumerate(labels):
        for b in labels[index+1:]:
            x, y, w, h = a["bbox_pt"]
            xx, yy, ww, hh = b["bbox_pt"]
            dx, dy = min(x+w, xx+ww)-max(x, xx), min(y+h, yy+hh)-max(y, yy)
            if dx <= 1.2 or dy <= 1.2:
                continue
            pair = [a["label"], b["label"]]
            if a["angle"] == b["angle"] == 45:
                ax, ay = a["anchor_pt"]
                bx, by = b["anchor_pt"]
                perpendicular = abs((bx-ax)-(by-ay))/np.sqrt(2)
                if perpendicular > 1.15*max(a["font_pt"], b["font_pt"]):
                    parallel_ticks.append({"labels": pair, "perpendicular_gap_pt": perpendicular})
                    continue
            overlaps.append(pair)
    return overlaps, parallel_ticks


def export(fig, stem):
    fonts(fig)
    if stem == "fig_claims":
        fill_prism_width(fig, stem)
    fill_panel_width(fig, stem)
    # Glyph paths keep the SVG portable without embedding seven complete font
    # files in every asset. The PDF retains embedded, selectable vector text.
    mpl.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": stem+"-space"})
    labels, outside = bounds(fig)
    assert not outside, (stem, "Text outside canvas", outside)
    overlaps, parallel_ticks = collisions(labels)
    assert not overlaps, (stem, "Text overlaps", overlaps)
    panel_margins = panel_text_clearance(fig, stem, labels)
    geometry_labels = annotation_clearance(fig)
    if stem == "fig_decodability_efficiency":
        renderer = fig.canvas.get_renderer()
        lower = fig.axes[2].child_axes[1]
        for t in lower.texts:
            if t.get_color() == "white":
                x, _ = t.get_position()
                bar = next(p for p in lower.patches if abs(p.get_x()+p.get_width()/2-x) < 1e-8)
                label_box, bar_box = t.get_window_extent(renderer), bar.get_window_extent(renderer)
                assert bar_box.x0 <= label_box.x0 and label_box.x1 <= bar_box.x1, t.get_text()
    output = ROOT/"figures"/(stem+"_space_optimized")
    width, height = fig.get_size_inches()*72
    original_dpi = fig.dpi
    surface = cairo.PDFSurface(str(output.with_suffix(".pdf")), width, height)
    surface.restrict_to_version(cairo.PDF_VERSION_1_5)
    try:
        surface.set_metadata(cairo.PDF_METADATA_CREATE_DATE, "")
        surface.set_metadata(cairo.PDF_METADATA_MOD_DATE, "")
        fig.set_dpi(300)
        context = cairo.Context(surface)
        context.scale(72/fig.dpi, 72/fig.dpi)
        renderer = RendererCairo(fig.dpi)
        renderer.set_context(context)
        fig.draw(renderer)
        context.show_page()
    finally:
        surface.finish()
        fig.set_dpi(original_dpi)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    svg_path = output.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "1700", "-png",
                    str(output.with_suffix(".pdf")), str(WORK/stem)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if stem == "fig_claims":
        shutil.copyfile((WORK/stem).with_suffix(".png"), output.with_suffix(".png"))
    pdf_fonts = subprocess.check_output(["pdffonts", str(output.with_suffix(".pdf"))], text=True)
    assert "Type 3" not in pdf_fonts, (stem, pdf_fonts)
    assert "TeXGyreHeros-Regular" in pdf_fonts, (stem, pdf_fonts)
    if stem != "fig_lrh_fulltext_mentions":
        assert "TeXGyreHeros-Bold" in pdf_fonts, (stem, pdf_fonts)
    if stem not in {"fig_lrh_fulltext_mentions", "fig_claims"}:
        assert "TeXGyreHerosCondensed-Regular" in pdf_fonts, (stem, pdf_fonts)
    record = RECORDS.setdefault(stem, {})
    record.update({"canvas_pt": [width, height], "text_outside_canvas": outside,
                   "text_overlaps": overlaps, "diagonal_tick_clearance": parallel_ticks,
                   "geometry_annotation_clearance": geometry_labels,
                   "panel_text_clearance": panel_margins,
                   "text_horizontal_extent_pt": [min(t["bbox_pt"][0] for t in labels),
                                                 max(t["bbox_pt"][0]+t["bbox_pt"][2] for t in labels)],
                   "labels": labels, "pdf_fonts": pdf_fonts.splitlines(),
                   "outputs": {output.with_suffix(ext).relative_to(ROOT).as_posix():
                               digest(output.with_suffix(ext)) for ext in (".pdf", ".svg")}})
    plt.close(fig)
    print(output.with_suffix(".pdf"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=int, nargs="+", default=list(BUILDERS))
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    for number in args.figures:
        setup()
        stem, build = BUILDERS[number]
        export(build(), stem)
    previous = ROOT/"out/figures/main.json"
    previous.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(previous.read_text()) if previous.exists() else {"figures": {}}
    record["command"] = "uv run --group figures python figures/render_main.py"
    record["script_sha256"] = digest(__file__)
    record["matplotlib"] = mpl.__version__
    record["software"] = {"matplotlib": mpl.__version__, "numpy": np.__version__,
                           "pandas": pd.__version__, "cairo": cairo.cairo_version_string()}
    record["inputs"] = inputs
    record["fonts"] = FONT_RECORDS
    record["figures"].update(RECORDS)
    previous.write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
