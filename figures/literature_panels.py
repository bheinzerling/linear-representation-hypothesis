
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/literature_interventions"
OUTPUT = ROOT / "figures/fig_literature_interventions"
INK, MID, LIGHT = "#25231F", "#73706A", "#D8D3CA"
BLUE, GREEN, OCHRE, RED = "#35618D", "#3F7F64", "#AC7D00", "#AE5358"
PURPLE = "#8064A2"
TIGGES_COLORS = [INK, MID, BLUE, GREEN, PURPLE]
ROW_TOPS = [.985, .660, .335]
PANEL_X = [.024, .357, .690]
PANEL_WIDTH = .290
LOSS_CMAP = LinearSegmentedColormap.from_list("use_loss", ["#FAF8F3", "#7FA89A", "#225F58"])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> dict:
    # Explicit registration handles fonts installed after Matplotlib cached its list.
    for path in sorted(font_manager.findSystemFonts()):
        if Path(path).stem.startswith(("Tinos-", "NotoSans-")):
            font_manager.fontManager.addfont(path)
    fonts = {}
    for family, weight in [("Tinos", "normal"), ("Noto Sans", "semibold")]:
        path = Path(font_manager.findfont(
            font_manager.FontProperties(family=family, weight=weight),
            fallback_to_default=False))
        fonts[f"{family} {weight}"] = {"file": path.name, "sha256": digest(path)}
    mpl.rcParams.update({
        "font.family": "Tinos", "font.size": 7.2,
        "axes.labelsize": 7.0, "xtick.labelsize": 6.4, "ytick.labelsize": 6.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MID,
        "xtick.color": MID, "ytick.color": MID, "axes.linewidth": .55,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "causal-use-literature-candidates",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "mathtext.fontset": "stix", "lines.solid_capstyle": "round",
    })
    return fonts


def load_data() -> dict:
    data = json.loads((DATA / "literature_data.json").read_text())
    for row in data["tigges"]["conditions"]:
        if row["accuracy_percent"] is not None:
            assert row["accuracy_percent"] + row["loss_pp"] == 100
    assert data["tigges"]["conditions"][1]["accuracy_percent"] is None
    ls = data["lasri"]
    assert ls["layers"] == list(range(13))
    for key in ["subject_loss_pp", "target_loss_pp", "subject_random_loss_pp",
                "target_random_loss_pp", "subject_removed_rank", "target_removed_rank"]:
        assert len(ls[key]) == 13 and np.isfinite(ls[key]).all()
    assert ls["subject_loss_pp"][1] == 32 and ls["subject_loss_pp"][9:] == [0] * 4
    assert ls["target_loss_pp"][11] == 33
    assert data["hanna"]["pca_ablation_numeric_value"] is None
    np.testing.assert_allclose(
        data["hanna"]["baseline_probability_difference_percent"]
        - data["hanna"]["leace_probability_difference_percent"], 17.0)
    return data


def panel(fig: plt.Figure, index: int, title: str) -> plt.Axes:
    row, col = divmod(index, 3)
    ax = fig.add_axes([PANEL_X[col], ROW_TOPS[row] - .300, PANEL_WIDTH, .213])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0, 1.125, f"({chr(97 + index)}) {title}", va="top",
            fontfamily="Noto Sans", fontsize=7.6, fontweight="semibold")
    return ax


def axis_clean(ax: plt.Axes, grid: str | None = None) -> None:
    ax.tick_params(length=2.0, pad=2.0)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.45, alpha=.65, zorder=0)
    ax.set_axisbelow(True)


def arrow(ax: plt.Axes, start: tuple, end: tuple, color: str = MID, **kwargs) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=7, lw=.8, color=color,
                                shrinkA=1, shrinkB=1, **kwargs))


def text_box(ax: plt.Axes, xy: tuple, width: float, height: float,
             text: str, color: str = BLUE, fontsize: float = 7.2) -> None:
    ax.add_patch(FancyBboxPatch(xy, width, height,
                               boxstyle="round,pad=0.015,rounding_size=0.02",
                               facecolor="white", edgecolor=color, lw=.7))
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, linespacing=1.2, color=color)


def tigges_accuracy(parent: plt.Axes, data: dict) -> None:
    ax = parent.inset_axes([.405, .25, .55, .71])
    conditions = data["conditions"]
    labels = ["No ablation", "Random\nall tokens", "Sentiment\nall tokens",
              "Sentiment\ncommas", "Full state\ncommas"]
    for i, (row, color) in enumerate(zip(conditions, TIGGES_COLORS, strict=True)):
        if row["accuracy_percent"] is None:
            lo, hi = row["accuracy_lower_bound_open"], row["accuracy_upper_bound_closed"]
            ax.plot([lo, hi], [i, i], lw=2, color=color)
            ax.scatter([lo], [i], s=18, facecolors="white", edgecolors=color, lw=.8,
                       zorder=4, clip_on=False)
            ax.annotate(">99", (lo, i), xytext=(-5, -9), textcoords="offset points",
                        ha="right", fontsize=7, color=color)
        else:
            value = row["accuracy_percent"]
            ax.plot([0, value], [i, i], color=color, alpha=.28, lw=1)
            ax.scatter([value], [i], s=23, marker="s" if i == 0 else "o", color=color,
                       zorder=4, clip_on=False)
            ax.annotate(f"{value}", (value, i), xytext=(-5, 5), textcoords="offset points",
                        ha="right", fontsize=7.2, color=color)
    ax.axvline(50, color=MID, ls=(0, (2, 2)), lw=.6)
    ax.set(xlim=(0, 100), ylim=(4.5, -.55), yticks=range(5), yticklabels=labels,
           xticks=[0, 50, 100], xlabel="Accuracy (%)")
    ax.tick_params(axis="y", length=0, labelsize=6.8)
    ax.spines["left"].set_visible(False)
    axis_clean(ax)
    parent.text(.405, .005, "Dotted: 50% chance", fontsize=6.3, color=MID)


def tigges_loss(parent: plt.Axes, data: dict) -> None:
    ax = parent.inset_axes([.405, .25, .55, .71])
    labels = ["Random\nall tokens", "Sentiment\nall tokens", "Sentiment\ncommas", "Full state\ncommas"]
    for i, (row, color) in enumerate(zip(data["conditions"][1:], TIGGES_COLORS[1:], strict=True)):
        value = row["loss_pp"]
        if value is None:
            ax.add_patch(Rectangle((0, i - .25), 1, .5, facecolor="none", edgecolor=color, lw=.8))
            ax.text(3, i, "<1", va="center", color=color)
        else:
            ax.barh(i, value, height=.5, color=color, alpha=.86, zorder=3)
            ax.text(value + 2, i, str(value), va="center", color=color, fontsize=7.5)
    ax.set(ylim=(3.55, -.55), xlim=(0, 50), yticks=range(4), yticklabels=labels,
           xticks=[0, 25, 50], xlabel="Accuracy loss (pp)")
    ax.tick_params(axis="y", length=0, labelsize=6.8)
    ax.spines["left"].set_visible(False)
    axis_clean(ax, "x")
    parent.text(.405, .005, "No ablation: 0 pp", color=MID, fontsize=6.3)


def tigges_positions(parent: plt.Axes, data: dict) -> None:
    parent.text(.01, .91, "Ablation", fontsize=6.8, color=MID)
    parent.text(.56, .91, "Token positions", ha="center", fontsize=6.8, color=MID)
    parent.text(.995, .91, "Accuracy", ha="right", fontsize=6.8, color=MID)
    positions = np.linspace(.365, .74, 5)
    for i, (row, color) in enumerate(zip(data["conditions"], TIGGES_COLORS, strict=True)):
        y = .78 - i * .139
        parent.text(.01, y, row["direction"], va="center", fontsize=6.9)
        for j, x in enumerate(positions):
            selected = row["positions"] == "all" or (row["positions"] == "commas" and j in [1, 3])
            parent.add_patch(Rectangle((x - .025, y - .035), .05, .07,
                                       facecolor=color if selected else "#EEEAE2",
                                       edgecolor=color if selected else LIGHT, lw=.45))
            if j in [1, 3]:
                parent.text(x, y + .006, ",", ha="center", va="center",
                            fontsize=10, color="white" if selected else MID)
        value = ">99%" if row["accuracy_percent"] is None else f"{row['accuracy_percent']}%"
        parent.text(.995, y, value, ha="right", va="center", color=color, fontsize=8.3)
    parent.text(.5, .025, "Shaded = ablated; token strip schematic", ha="center", fontsize=6.1, color=MID)


def lasri_lines(parent: plt.Axes, data: dict) -> None:
    ax = parent.inset_axes([.18, .16, .78, .70])
    layers = data["layers"]
    for name, color, marker in [("subject", BLUE, "o"), ("target", OCHRE, "s")]:
        ax.plot(layers, data[f"{name}_loss_pp"], marker=marker, ms=3.2, lw=1.3,
                color=color, zorder=3, clip_on=False)
        ax.plot(layers, data[f"{name}_random_loss_pp"], marker=marker, ms=2.8,
                mfc="white", mew=.7, lw=.8, ls=(0, (2, 2)), color=color,
                zorder=4, clip_on=False)
    ax.annotate("32", (1, 32), xytext=(1, 7), textcoords="offset points", color=BLUE)
    ax.annotate("33", (11, 33), xytext=(-1, 7), textcoords="offset points", ha="center", color=OCHRE)
    ax.set(xlim=(-.2, 12.2), ylim=(0, 50), xticks=[0, 3, 6, 9, 12],
           yticks=[0, 25, 50], xlabel="Layer", ylabel="Accuracy loss (pp)")
    axis_clean(ax, "y")
    ax.legend([Line2D([], [], color=BLUE, marker="o", ms=3),
               Line2D([], [], color=OCHRE, marker="s", ms=3)],
              ["Subject", "Target"], loc="upper left", frameon=False, ncol=2,
              fontsize=6.7, handlelength=1.0, borderpad=.1, columnspacing=.8)
    parent.text(.18, .94, "Dashed / open: rank-matched random", fontsize=6.4, color=MID)


def lasri_heatmap(parent: plt.Axes, data: dict) -> None:
    ax = parent.inset_axes([.265, .29, .715, .54])
    values = np.array([data["subject_loss_pp"], data["subject_random_loss_pp"],
                       data["target_loss_pp"], data["target_random_loss_pp"]])
    im = ax.imshow(values, aspect="auto", cmap=LOSS_CMAP, vmin=0, vmax=50, interpolation="none")
    for r in range(4):
        for c in range(13):
            ax.text(c, r, str(values[r, c]), ha="center", va="center",
                    fontsize=5.5, color=INK if values[r, c] < 30 else "white")
    ax.set(yticks=range(4), yticklabels=["Subject", "Random", "Target", "Random"],
           xticks=[0, 3, 6, 9, 12], xticklabels=[0, 3, 6, 9, 12], xlabel="Layer")
    ax.axhline(1.5, color="white", lw=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=2, labelsize=6.3)
    cb_ax = parent.inset_axes([.265, .07, .715, .055])
    cb = parent.figure.colorbar(im, cax=cb_ax, orientation="horizontal", ticks=[0, 25, 50])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, pad=1, labelsize=5.8)
    parent.text(.265, .96, "Accuracy loss (pp)", fontsize=6.8, color=MID)


def lasri_profiles(parent: plt.Axes, data: dict) -> None:
    for k, (name, color, marker, label) in enumerate([
            ("subject", BLUE, "o", "Subject"), ("target", OCHRE, "s", "Target")]):
        ax = parent.inset_axes([.16 + k * .48, .18, .32, .67])
        vals = data[f"{name}_loss_pp"]
        ax.hlines(data["layers"], 0, vals, color=color, lw=.85, alpha=.65)
        ax.scatter(vals, data["layers"], s=15, color=color, marker=marker, zorder=3)
        ax.scatter(data[f"{name}_random_loss_pp"], data["layers"], s=18,
                   facecolors="white", edgecolors=MID, linewidths=.7, zorder=4)
        ax.set(xlim=(-2, 50), ylim=(12.6, -.6), xticks=[0, 25, 50],
               yticks=[0, 3, 6, 9, 12])
        ax.set_title(label, fontsize=7.2, color=color, pad=4)
        if k == 0:
            ax.set_ylabel("Layer", labelpad=1)
        else:
            ax.set_yticklabels([])
        axis_clean(ax, "x")
    parent.text(.56, .025, "Accuracy loss (pp)", ha="center", fontsize=7)
    parent.text(.16, .97, "Open circles: rank-matched random", fontsize=6.3, color=MID)


def year_curve(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    """An explicitly schematic ordered U; no measured point locations."""
    t = np.linspace(-1, 1, 11)
    xx, yy = x + width * (t + 1) / 2, y + height * t ** 2
    ax.plot(xx, yy, color=LIGHT, lw=.8, zorder=1)
    for j, (a, b) in enumerate(zip(xx, yy, strict=True)):
        ax.scatter(a, b, s=9, color=mpl.colormaps["viridis"](.12 + .76 * j / 10), zorder=2)
    ax.text(x, y + height + .035, "low", ha="center", fontsize=5.8, color=MID)
    ax.text(x + width, y + height + .035, "high", ha="center", fontsize=5.8, color=MID)


def hanna_dots(parent: plt.Axes, data: dict) -> None:
    year_curve(parent, .055, .67, .25, .19)
    parent.text(.18, .58, "Year order\n(schematic)", ha="center", va="top", fontsize=6.2, color=MID)
    arrow(parent, (.34, .76), (.48, .76))
    parent.text(.75, .79, "Ablate PCA span", ha="center", fontsize=7.1)
    parent.text(.75, .67, "Little change\nreported", ha="center", fontsize=7.7, color=GREEN, linespacing=1.2)
    ax = parent.inset_axes([.28, .13, .68, .30])
    for i, (value, color) in enumerate([
        (data["baseline_probability_difference_percent"], INK),
        (data["leace_probability_difference_percent"], OCHRE),
    ]):
        ax.plot([0, value], [i, i], color=color, alpha=.35, lw=1)
        ax.scatter([value], [i], color=color, s=23, zorder=3)
        ax.annotate(f"{value:.1f}", (value, i), xytext=(-3, 5), textcoords="offset points",
                    ha="right", fontsize=7, color=color)
    ax.axvline(0, color=LIGHT, lw=.7)
    ax.set(xlim=(-100, 100), ylim=(1.55, -.6), yticks=[0, 1], yticklabels=["Baseline", "LEACE"],
           xticks=[-100, 0, 100], xlabel="Probability difference (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    axis_clean(ax)


def hanna_bars(parent: plt.Axes, data: dict) -> None:
    parent.text(.5, .95, "Ordered year geometry in PCA", ha="center", fontsize=6.8, color=MID)
    text_box(parent, (.12, .71), .84, .14, "PCA ablation: little change reported", GREEN, fontsize=6.9)
    ax = parent.inset_axes([.23, .15, .71, .45])
    values = [data["baseline_probability_difference_percent"], data["leace_probability_difference_percent"]]
    ax.bar([0, 1], values, color=[INK, OCHRE], width=.45, alpha=.88, zorder=3)
    for i, (value, color) in enumerate(zip(values, [INK, OCHRE], strict=True)):
        ax.text(i, value + 6, f"{value:.1f}", ha="center", fontsize=7.3, color=color)
    ax.axhline(0, color=MID, lw=.6)
    ax.set(ylim=(-100, 100), xlim=(-.65, 1.65), yticks=[-100, 0, 100],
           xticks=[0, 1], xticklabels=["Baseline", "LEACE"], ylabel="Prob. difference (%)")
    ax.tick_params(axis="x", length=0)
    axis_clean(ax, "y")
    parent.text(.58, .025, "LEACE decrease: 17.0 pp", ha="center", color=OCHRE, fontsize=6.9)


def hanna_flow(parent: plt.Axes, data: dict) -> None:
    year_curve(parent, .09, .79, .22, .12)
    parent.text(.68, .88, "Ordered year\nrepresentations", ha="center", va="center", fontsize=7.6)
    parent.text(.20, .73, "schematic", ha="center", color=MID, fontsize=6)
    text_box(parent, (.05, .48), .41, .15, "Ablate PCA span", BLUE, fontsize=7)
    text_box(parent, (.62, .48), .35, .15, "Little change\nreported", GREEN, fontsize=7.3)
    arrow(parent, (.47, .555), (.605, .555))
    parent.plot([.05, .97], [.395, .395], color=LIGHT, lw=.6)
    parent.text(.51, .35, "LEACE erasure", ha="center", fontsize=7.1)
    baseline, leace = data["baseline_probability_difference_percent"], data["leace_probability_difference_percent"]
    parent.text(.20, .20, f"{baseline:.1f}%", ha="center", fontsize=12, color=INK)
    parent.text(.82, .20, f"{leace:.1f}%", ha="center", fontsize=12, color=OCHRE)
    arrow(parent, (.37, .24), (.64, .24), OCHRE)
    parent.text(.20, .11, "Baseline", ha="center", fontsize=6.3, color=MID)
    parent.text(.82, .11, "After erasure", ha="center", fontsize=6.3, color=MID)
    parent.text(.51, .01, "Probability difference; decrease 17.0 pp", ha="center", fontsize=6.5, color=MID)


def main() -> None:
    fonts = style()
    data = load_data()
    fig = plt.figure(figsize=(16 / 2.54, 7.40))
    studies = [
        ("Tigges et al. (2024) · Sentiment", "Pythia-2.8B · Originally correct SST examples with commas · All-layer ablation"),
        ("Lasri et al. (2022) · Grammatical number", "BERT-base · Subject–verb agreement · Target = masked verb; layer 0 = embeddings"),
        ("Hanna et al. (2023) · Numerical geometry", "GPT-2 small · Year-span completion · PCA ablation and LEACE are separate experiments"),
    ]
    for top, (title, subtitle) in zip(ROW_TOPS, studies, strict=True):
        fig.text(.024, top, title, fontsize=9.2, fontfamily="Noto Sans", fontweight="semibold", va="top")
        fig.text(.024, top - .027, subtitle, fontsize=7.0, color=MID, va="top")
    for y in [.675, .350]:
        fig.add_artist(Line2D([.024, .980], [y, y], color=LIGHT, lw=.6, transform=fig.transFigure))
    specs = [
        ("Accuracy after ablation", tigges_accuracy, "tigges"),
        ("Loss from the baseline", tigges_loss, "tigges"),
        ("Ablated token positions", tigges_positions, "tigges"),
        ("Effects across layers", lasri_lines, "lasri"),
        ("Layer × position matrix", lasri_heatmap, "lasri"),
        ("Paired layer profiles", lasri_profiles, "lasri"),
        ("Geometry and deletion", hanna_dots, "hanna"),
        ("Intervention comparison", hanna_bars, "hanna"),
        ("Operation and outcome", hanna_flow, "hanna"),
    ]
    containers = []
    for i, (title, render, source) in enumerate(specs):
        parent = panel(fig, i, title)
        render(parent, data[source])
        containers.append(parent)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    escaped = []
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width and bbox.height and (
            bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1 + 1 or bbox.y1 > fig.bbox.y1 + 1
        ):
            escaped.append(artist.get_text())
    assert len(containers) == 9
    assert not escaped, f"Text outside the sheet: {escaped}"
    for suffix in ["pdf", "svg", "png"]:
        kwargs = {"dpi": 300} if suffix == "png" else {}
        if suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None,
                                  "Title": "Causal use: literature visualization candidates"}
        if suffix == "svg":
            kwargs["metadata"] = {"Date": None}
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), **kwargs)
        if suffix == "svg":
            svg_path = OUTPUT.with_suffix(".svg")
            svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    provenance = {
        "figure": str(OUTPUT.relative_to(ROOT)),
        "status": "temporary nine-panel comparison sheet",
        "command": "uv run python figures/literature_panels.py",
        "script_sha256": digest(Path(__file__)),
        "inputs": {str((DATA / name).relative_to(ROOT)): digest(DATA / name)
                   for name in ["literature_data.json", "semantic_contract.md"]},
        "panel_count": len(containers),
        "panels": [{"label": chr(97 + i), "title": title, "source": source}
                   for i, (title, _, source) in enumerate(specs)],
        "width_cm": 16, "height_inches": 7.40,
        "axis_ranges": {"accuracy_percent": [0, 100], "accuracy_loss_pp": [0, 50],
                        "probability_difference_percent": [-100, 100]},
        "derived_values": {"hanna_leace_decrease_pp": round(
            data["hanna"]["baseline_probability_difference_percent"]
            - data["hanna"]["leace_probability_difference_percent"], 8)},
        "checks": {"source_value_checks": "passed", "text_outside_sheet": escaped},
        "schematics": {"c": "Five illustrative token positions, with commas at positions 2 and 4.",
                       "g_i": "Constructed U-shaped year order; points are not measured projections."},
        "software": {"matplotlib": mpl.__version__, "numpy": np.__version__},
        "fonts": fonts,
    }
    OUTPUT.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
