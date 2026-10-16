
from __future__ import annotations

from paper_figure_style import configure_fonts, savefig

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

from numerical_uncertainty import load_probe_uncertainty

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/decodability_efficiency"
OUTPUT = ROOT / "figures/fig_decodability_efficiency"
ORDER = ["linear", "pls", "circular_reg", "circular_cls"]
LABELS = {"linear": "Scalar\nridge", "pls": "Scalar\nPLS",
          "circular_reg": "Circular\nreg.", "circular_cls": "Circular\ncls."}
COLORS = {"linear": "#35618D", "pls": "#3F7F64",
          "circular_reg": "#AC7D00", "circular_cls": "#8064A2"}
MARKERS = {"linear": "o", "pls": "s", "circular_reg": "^", "circular_cls": "D"}
DARK, MID, LIGHT = "#25231F", "#73706A", "#D8D3CA"
METRIC_STYLES = {
    "r2": {"lighten": .12, "hatch": None},
    "accuracy": {"lighten": .84, "hatch": "///"},
    "mean_abs_error": {"lighten": .12, "hatch": None},
    "median_abs_error": {"lighten": .84, "hatch": "///"},
}
LAYOUT = {
    "size": (7.15, 2.55),
    "panels": {"a": [.004, .08, .174, .72], "b": [.181, .035, .226, .79],
               "c": [.415, .02, .282, .80], "d": [.709, .025, .286, .795]},
    "titles": {"a": (.01, .95), "b": (.19, .95), "c": (.423, .95), "d": (.718, .95)},
    "b_footer_offset": .045,
    "d_plot_inset": [.19, .276, .77, .604],
    "export_crop_bottom_inches": .12,
    "export_crop_top_inches": .03,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    manifest = json.loads((DATA / "input_snapshots.json").read_text())
    for name, item in manifest.items():
        assert sha256(DATA / name) == item["sha256"], name
    module_spec = importlib.util.spec_from_file_location("figure3_reference_panels", DATA / "reference_panels.py")
    reference = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reference)
    geometry = pd.read_csv(DATA / "probe_outputs.csv")
    geometry_prov = json.loads((DATA / "probe_outputs.provenance.json").read_text())
    metrics = pd.read_csv(DATA / "probe_metrics.csv").set_index("architecture").loc[ORDER].reset_index()
    capacity = pd.read_csv(DATA / "capacity_points.csv")
    np.testing.assert_array_equal(geometry.value, np.arange(1000))
    assert metrics.num_examples.eq(100).all()
    assert capacity.layer_idx.eq(10).all() and capacity.model_name.eq("meta-llama/Llama-3.1-8B").all()
    assert capacity.prompt_family.eq("addition_formula_rhs_source").all()
    assert capacity.partition.eq("test").all() and capacity.num_examples.eq(100).all()
    assert set(capacity.architecture) == set(ORDER)
    assert capacity.groupby("architecture").effective_df.nunique().gt(1).all()
    assert np.isfinite(capacity[["effective_df", "mean_abs_error"]]).all().all()
    return reference, geometry, geometry_prov, metrics, capacity, manifest


def lighten(color: str, amount=.85):
    return np.array(to_rgb(color)) * (1 - amount) + amount


def style_axis(ax):
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color(MID)
        ax.spines[edge].set_linewidth(.65)
    ax.tick_params(axis="both", colors=MID, labelsize=7.5, length=2.2, width=.6, pad=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=LIGHT, linewidth=.5)


def draw_metrics(ax, frame, uncertainty=None):
    if uncertainty is None:
        uncertainty = load_probe_uncertainty(frame.set_index("architecture"))
    ax.axis("off")
    upper = ax.inset_axes([.105, .57, .88, .34])
    lower = ax.inset_axes([.105, .18, .88, .30], sharex=upper)
    x = np.arange(4)
    for axis, specs in [
        (upper, [(-.24, "r2", r"$R^2$ $\uparrow$"),
                 (.24, "accuracy", r"Exact accuracy $\uparrow$")]),
        (lower, [(-.24, "mean_abs_error", r"MAE $\downarrow$"),
                 (.24, "median_abs_error", r"Median AE $\downarrow$")]),
    ]:
        handles = []
        for offset, column, label in specs:
            style = METRIC_STYLES[column]
            colors = [COLORS[method] for method in ORDER]
            bars = axis.bar(x + offset, frame[column], width=.42,
                            color=[lighten(c, style["lighten"]) for c in colors], edgecolor=colors,
                            linewidth=.8, hatch=style["hatch"], zorder=3)
            handles.append(Patch(facecolor=lighten(MID, style["lighten"]), edgecolor=MID,
                                 linewidth=.8, hatch=style["hatch"], label=label))
            high_for_labels = []
            for i, (method, value) in enumerate(zip(ORDER, frame[column], strict=True)):
                interval = uncertainty.loc[(method, column)]
                lo, hi = float(interval.low), float(interval.high)
                high_for_labels.append(hi if np.isfinite(hi) else value)
                if np.isfinite([lo, hi]).all():
                    axis.errorbar(i + offset, value, yerr=[[value - lo], [hi - value]],
                                  fmt="none", color=DARK, elinewidth=.65,
                                  capsize=1.25, capthick=.65, zorder=4)
            if column == "r2":
                continue
            for bar, value, high, color in zip(bars, frame[column], high_for_labels, colors, strict=True):
                is_score = column == "accuracy"
                position = max(value, high) + (.04 if is_score else .24)
                axis.text(bar.get_x() + bar.get_width()/2, position,
                          f"{value:.0%}" if is_score else f"{value:.1f}",
                          ha="center", va="bottom", color=color,
                          fontsize=7.3, zorder=5)
        axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.015),
                    frameon=False, ncol=2, fontsize=7.2, handlelength=.72,
                    handletextpad=.35, columnspacing=.7, borderaxespad=0, borderpad=0)
    upper.set(xlim=(-.55, 3.55), ylim=(0, 1), yticks=[0, .5, 1], yticklabels=["0", ".5", "1"])
    lower.set(ylim=(0, 15), yticks=[0, 5, 10, 15], xticks=x,
              xticklabels=[LABELS[name] for name in ORDER])
    for axis in (upper, lower):
        style_axis(axis)
    upper.tick_params(axis="x", bottom=False, labelbottom=False)
    lower.tick_params(axis="x", colors=DARK, labelsize=7.4, length=0, pad=3)
    for label, method in zip(lower.get_xticklabels(), ORDER, strict=True):
        label.set_color(COLORS[method])
    assert len(upper.patches) == 8
    assert len(lower.patches) == 8


def draw_capacity(ax, frame):
    ax.axis("off")
    plot = ax.inset_axes(LAYOUT["d_plot_inset"])
    for method in ORDER:
        rows = frame.loc[frame.architecture == method].sort_values("effective_df")
        plot.plot(rows.effective_df, rows.mean_abs_error, color=COLORS[method],
                  marker=MARKERS[method], markersize=2.7, linewidth=1.05,
                  markeredgewidth=.5, markerfacecolor=COLORS[method], label=LABELS[method].replace("\n", " "))
    plot.set_xscale("log", base=2)
    plot.set_yscale("log")
    plot.set(xlim=(1, 4600), ylim=(.65, 700))
    plot.set_xticks([1, 16, 256, 4096], ["1", "16", "256", "4096"])
    plot.set_yticks([1, 10, 100], ["1", "10", "100"])
    plot.xaxis.set_minor_locator(NullLocator())
    plot.yaxis.set_minor_locator(NullLocator())
    plot.set_xlabel("Method-specific\ncomplexity" + r"$^\dagger$", color=DARK, fontsize=7.5, labelpad=3)
    plot.set_ylabel(r"MAE $\downarrow$", color=DARK, fontsize=7.5, labelpad=2)
    style_axis(plot)
    handles, labels = plot.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower center", bbox_to_anchor=(.54, .91), ncol=2,
              frameon=False, fontsize=7.3, handlelength=1.2, handletextpad=.35,
              columnspacing=.75, labelspacing=.25, borderaxespad=0)
    assert len(plot.lines) == 4
    assert sum(len(line.get_xdata()) for line in plot.lines) == len(frame)
    return plot


def build(reference, geometry, geometry_prov, metrics, capacity):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
                         "font.size": 8, "text.color": DARK, "axes.titlecolor": DARK,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "fig_decodability_efficiency",
                         "hatch.linewidth": .4})
    configure_fonts()
    fig = plt.figure(figsize=LAYOUT["size"])
    axes = {key: fig.add_axes(rect) for key, rect in LAYOUT["panels"].items()}
    reference.draw_local_linearization(axes["a"])
    axes["a"].set_title("", loc="left")
    reference.draw_probe_outputs(axes["b"], geometry, geometry_prov, footer_offset=LAYOUT["b_footer_offset"])
    # Retain the original marks and projection while adapting small text to print size.
    for text in axes["a"].texts:
        text.set_fontsize(7.5)
        if text.get_text() == "affine\napproximation":
            text.set_text("linear\napproximation")
    for text in axes["b"].get_legend().get_texts():
        text.set_fontsize(6.9)
        if text.get_text() in {"Linear probe", "Ridge", "Linear ridge"}:
            text.set_text("Scalar ridge")
    draw_metrics(axes["c"], metrics)
    capacity_plot = draw_capacity(axes["d"], capacity)
    titles = {"a": "(a) Local\nlinearization", "b": "(b) Helix of\nprobe outputs",
              "c": "(c) Probe performance", "d": "(d) Probe complexity"}
    for key, (x, y) in LAYOUT["titles"].items():
        fig.text(x, y, titles[key], ha="left", va="top", fontsize=8.5,
                 fontweight="bold", linespacing=1.1)
    keys = list(LAYOUT["panels"])
    dividers = [((LAYOUT["panels"][left][0] + LAYOUT["panels"][left][2] + LAYOUT["panels"][right][0])/2, .11, .88)
                for left, right in zip(keys[:-1], keys[1:])]
    for x, lower, upper in dividers:
        fig.add_artist(Line2D([x, x], [lower, upper], transform=fig.transFigure, color=LIGHT, lw=.5))
    fig.canvas.draw()
    # No figure-level heading or legend may escape the exported canvas.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, text.get_text()
    for i, text in enumerate(fig.texts):
        for other in fig.texts[i + 1:]:
            assert not text.get_window_extent(renderer).overlaps(other.get_window_extent(renderer)), (text.get_text(), other.get_text())
    # White numeric labels must fit fully inside their dark bar fills.
    lower = axes["c"].child_axes[1]
    for text in lower.texts:
        if text.get_color() != "white":
            continue
        x, _ = text.get_position()
        bar = next(p for p in lower.patches if abs(p.get_x() + p.get_width()/2 - x) < 1e-8)
        label_box, bar_box = text.get_window_extent(renderer), bar.get_window_extent(renderer)
        assert bar_box.x0 <= label_box.x0 and label_box.x1 <= bar_box.x1, text.get_text()
    c_bottom = min(t.get_window_extent(renderer).y0 for t in lower.get_xticklabels())
    b_bottom = axes["b"].get_legend().get_window_extent(renderer).y0
    d_bottom = capacity_plot.xaxis.label.get_window_extent(renderer).y0
    assert b_bottom >= c_bottom and abs(d_bottom - c_bottom) < 1.5
    bottom_alignment = {"b_legend": b_bottom, "c_tick_labels": c_bottom, "d_axis_title": d_bottom}
    return fig, {key: value * 72 / fig.dpi for key, value in bottom_alignment.items()}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reference, geometry, geometry_prov, metrics, capacity, manifest = load_inputs()
    fig, alignment = build(reference, geometry, geometry_prov, metrics, capacity)
    crop = LAYOUT["export_crop_bottom_inches"]
    export_box = mpl.transforms.Bbox.from_bounds(
        0, crop, fig.get_figwidth(), fig.get_figheight() - crop - LAYOUT["export_crop_top_inches"])
    alignment = {key: value - crop * 72 for key, value in alignment.items()}
    assert min(alignment.values()) > 1.5
    outputs = {}
    for extension in ("pdf", "png", "svg"):
        path = OUTPUT.with_suffix(f".{extension}")
        metadata = {"pdf": {"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None},
                    "svg": {"Creator": Path(__file__).name, "Date": None}}.get(extension)
        savefig(fig, path, dpi=350, metadata=metadata, bbox_inches=export_box)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs[str(path.relative_to(ROOT))] = sha256(path)
    plt.close(fig)
    provenance = {"figure_id": "fig_decodability_efficiency", "status": "main Figure 3", "inputs": manifest,
                  "data_semantics": json.loads((DATA / "capacity.provenance.json").read_text()),
                  "layout": LAYOUT, "family_colors": COLORS, "panel_c_palette": "families",
                  "panel_c_metric_styles": METRIC_STYLES,
                  "panel_d_marker_fill": "family color", "bottom_alignment_points": alignment,
                  "conditional_uncertainty": json.loads((ROOT / "data/figures/conditional_uncertainty/provenance.json").read_text()),
                  "data_points_per_family": capacity.groupby("architecture").size().to_dict(),
                  "axis_ranges": {"c_scores": [0, 1], "c_errors": [0, 15], "d_x": [1, 4600], "d_y": [.65, 700]},
                  "generation": {"script": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__)),
                                 "command": "uv run --group figures python figures/decodability_panels.py",
                                 "numpy": np.__version__, "matplotlib": mpl.__version__}, "outputs": outputs}
    OUTPUT.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Rendered {OUTPUT.name}; bottom edges in points: {alignment}")


if __name__ == "__main__":
    main()
