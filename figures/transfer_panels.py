
from __future__ import annotations

from paper_figure_style import configure_fonts, savefig

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/figures/robustness_invariance_structure"
INVARIANCE_DATA = ROOT / "data/figures/pls_grid"
OUTPUT = ROOT / "figures/fig_robustness_invariance_structure"
INK, MID, LIGHT = "#25231F", "#73706A", "#D8D3CA"
BLUE, GREEN, OCHRE, RED = "#35618D", "#3F7F64", "#AC7D00", "#AE5358"
PURPLE = "#8064A2"
COLUMN_LEFTS = (.018, .352, .686)
COLUMN_WIDTH = .296
FIGURE_SIZE = (6.30, 3.20)
ROW_DIVIDER = 1.50
SENTIMENT_ROTATION_DEGREES = -14
FAMILIES = ["addition_formula", "addition_verbalized", "subtraction_formula",
            "subtraction_verbalized", "identity_patchscopes"]
PROMPT_LABELS = ["a + b", "a plus b", "a − b", "a minus b", "b = b"]
FAMILY_COLORS = [BLUE, GREEN, OCHRE, PURPLE, RED]
COS_CMAP = LinearSegmentedColormap.from_list("direction_cosine", [RED, "#FAF8F3", GREEN])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentiment(fig: plt.Figure) -> dict:
    left = COLUMN_LEFTS[0]
    height = fig.get_figheight()
    fig.text(left, 2.77 / height, "Sentiment directions", fontsize=8.5, fontweight="medium")
    ax = fig.add_axes([left + .010, 1.65 / height, COLUMN_WIDTH - .020, .96 / height])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    # Constructed examples; the blue direction is their actual mean difference.
    rng = np.random.default_rng(731)
    negative = rng.normal([.19, .25], [.045, .045], (13, 2))
    start = negative.mean(axis=0)
    positive = rng.normal(0, [.055, .15], (13, 2))
    positive -= positive.mean(axis=0)
    base_angle = np.deg2rad(37)
    positive += start + .68 * np.array([np.cos(base_angle), np.sin(base_angle)])
    theta = np.deg2rad(SENTIMENT_ROTATION_DEGREES)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    positive = start + (positive - start) @ rotation.T
    positive_mean = positive.mean(axis=0)
    mean_difference = positive_mean - start

    ax.scatter(*negative.T, s=13, marker="_", linewidth=1, color=RED, alpha=.65)
    ax.scatter(*positive.T, s=13, marker="+", linewidth=.9, color=GREEN, alpha=.65)
    ax.text(.03, .08, "negative", fontsize=7.3, color=RED)
    ax.text(positive_mean[0], positive[:, 1].min() - .15, "positive",
            fontsize=7.3, color=GREEN, ha="center", va="top")

    for degrees, color, label in [
        (0, BLUE, "Mean diff."), (9, PURPLE, "Logistic"), (18, OCHRE, "PCA"),
    ]:
        angle = np.deg2rad(degrees)
        direction_rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]])
        end = start + direction_rotation @ mean_difference
        if degrees == 0:
            assert np.allclose(end, positive_mean)
            label_position = (start + end) / 2 + np.array([-.035, -.23])
            alignment = "center"
        elif degrees == 9:
            label_position = np.array([.995, end[1] + .10])
            alignment = "right"
        else:
            label_position = end + np.array([-.23, .09])
            alignment = "left"
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=1.4, color=color, zorder=4, shrinkA=0, shrinkB=0,
        ))
        ax.text(*label_position, label, fontsize=7, color=color, ha=alignment)
    return {
        "sentiment_rng_seed": 731,
        "sentiment_clockwise_rotation_degrees": -SENTIMENT_ROTATION_DEGREES,
        "positive_spread_before_rotation": [.055, .15],
        "negative_sample_mean": start.tolist(),
        "positive_sample_mean": positive_mean.tolist(),
        "mean_difference_arrow_start": start.tolist(),
        "mean_difference_arrow_end": positive_mean.tolist(),
        "sentiment_method_colors": {"Mean diff.": BLUE, "Logistic": PURPLE, "PCA": OCHRE},
        "sentiment_class_colors": {"negative": RED, "positive": GREEN},
        "sentiment_angles_degrees_schematic": [
            float(np.rad2deg(np.arctan2(mean_difference[1], mean_difference[0])) + x)
            for x in [0, 9, 18]
        ],
    }


def sae_feature_matching(fig: plt.Figure, source: dict) -> dict:
    left = COLUMN_LEFTS[0]
    height = fig.get_figheight()
    shared = source["shared_fraction_approximate"]
    icon_count = 10
    pairs = [(1, 7), (5, 2), (8, 4)]
    assert np.isclose(len(pairs) / icon_count, shared)
    assert len({a for a, _ in pairs}) == len({b for _, b in pairs}) == len(pairs)
    crossing_count = sum((a - c) * (b - d) < 0
                         for i, (a, b) in enumerate(pairs) for c, d in pairs[i + 1:])
    assert crossing_count > 0

    fig.text(left, 1.35 / height, "SAE feature matching", fontsize=8.5, fontweight="medium")
    fig.text(left, 1.17 / height, "Trained on the same data", fontsize=7, color=MID)
    ax = fig.add_axes([left, .20 / height, COLUMN_WIDTH, .90 / height])
    ax.set(xlim=(0, 1), ylim=(0, .77))
    ax.axis("off")
    xs = np.linspace(.28, .96, icon_count)
    row_y = [.65, .03]
    for row, (y, label) in enumerate(zip(row_y, ["SAE A\nfeatures", "SAE B\nfeatures"], strict=True)):
        ax.text(0, y, label, va="center", fontsize=7, linespacing=1.1)
        matched = {pair[row] for pair in pairs}
        for i, x in enumerate(xs):
            ax.scatter(x, y, s=25, marker="s", linewidth=.6, zorder=4,
                       facecolor=BLUE if i in matched else "#EEEAE2",
                       edgecolor=BLUE if i in matched else "#B4ACA1")
    for a, b in pairs:
        ax.plot([xs[a], xs[b]], row_y, color=BLUE, lw=.9, zorder=2)
    callout_text = f"Only ≈{shared:.0%} of\nfeatures shared\nacross seeds"
    ax.text(.245, np.mean(row_y), callout_text, ha="center", va="center",
            fontsize=7, color=BLUE, linespacing=1.15, zorder=5,
            bbox={"boxstyle": "round,pad=.3,rounding_size=.15", "facecolor": "#F3F6F9",
                  "edgecolor": "#BBCBD9", "linewidth": .55})
    return {
        "sae_icon_count_per_fit": icon_count,
        "sae_shared_fraction_approximate": shared,
        "sae_unmatched_fraction_approximate": 1 - shared,
        "sae_illustrative_pairs_zero_indexed": pairs,
        "sae_link_crossing_count": crossing_count,
        "sae_icon_positions": xs.tolist(),
        "sae_row_y_positions": row_y,
        "sae_callout_text": callout_text.replace("\n", " "),
        "sae_diagram_semantics": "Ten icons per SAE summarize rounded proportions; positions and links are schematic.",
    }


def invariance(fig: plt.Figure) -> dict:
    manifest = json.loads((INVARIANCE_DATA / "input_provenance.json").read_text())
    for name, digest in manifest["snapshots"].items():
        assert sha256(INVARIANCE_DATA / name) == digest, name
    directions = pd.read_csv(INVARIANCE_DATA / "pls_directions.csv")
    transfer = pd.read_csv(INVARIANCE_DATA / "pls_transfer.csv")
    assert len(directions) == 660 and len(transfer) == 1650
    cosines = np.eye(5)
    pairs = directions.groupby(["left_family", "right_family"]).direction_cosine.mean()
    assert len(pairs) == 10
    for (a, b), value in pairs.items():
        i, j = FAMILIES.index(a), FAMILIES.index(b)
        cosines[i, j] = cosines[j, i] = value
    errors = transfer.groupby(["train_family", "test_family"]).raw.mean().unstack().loc[FAMILIES, FAMILIES].values
    assert np.isfinite(errors).all() and ((errors >= 0) & (errors <= 100)).all()
    native = np.diag(errors)
    cross = np.array([np.delete(row, i).mean() for i, row in enumerate(errors)])
    np.testing.assert_allclose([native.mean(), cross.mean()], [13.566121212121212, 58.13117424242424])

    left = COLUMN_LEFTS[1]
    height, width = fig.get_figheight(), fig.get_figwidth()
    fig.text(left, 2.77 / height, r"Probing the numeric value $b$", fontsize=8.5, fontweight="medium")
    ax = fig.add_axes([left + .100, 1.65 / height, .153, (.153 * width) / height])
    masked = np.ma.masked_where(np.triu(np.ones((5, 5), dtype=bool), k=1), cosines)
    ax.imshow(masked, cmap=COS_CMAP, vmin=-1, vmax=1, interpolation="none")
    for i in range(5):
        for j in range(i + 1):
            value = cosines[i, j]
            label = "1" if i == j else f"{value:.2f}".removeprefix("0")
            ax.text(j, i, label, ha="center", va="center", fontsize=6.3,
                    color="white" if value > .68 else INK)
    ax.set(xticks=range(5), yticks=range(5), yticklabels=PROMPT_LABELS)
    ax.set_xticklabels(PROMPT_LABELS, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.1)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.tick_params(length=0, pad=2)
    ax.set_ylabel("Prompt variant", fontsize=6.5, labelpad=3)
    for labels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for label, color in zip(labels, FAMILY_COLORS, strict=True):
            label.set_color(color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([.51, .84, .45, .037])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=COS_CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(labelsize=5.1, length=1.5, pad=1)
    cb.outline.set_linewidth(.3)
    ax.text(.735, .94, "Cosine", transform=ax.transAxes, ha="center", fontsize=6.1)

    fig.text(left, 1.22 / height, "Probe transfer", fontsize=8.5, fontweight="medium")
    handles = [Line2D([], [], color=MID, marker=marker, markersize=3.0, lw=0, label=label)
               for marker, label in [("D", "Same variant"), ("o", "Other variants")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(left + COLUMN_WIDTH / 2, 1.14 / height),
               ncol=2, frameon=False, fontsize=5.8, handlelength=.6, handletextpad=.4,
               columnspacing=.8, borderpad=0)
    tx = fig.add_axes([left + .095, .28 / height, .192, .64 / height])
    for i, (row, color) in enumerate(zip(errors, FAMILY_COLORS, strict=True)):
        targets = np.delete(row, i)
        tx.plot([native[i], cross[i]], [i, i], color=color, lw=1.0, alpha=.65, zorder=2)
        tx.scatter(targets, i + np.linspace(-.13, .13, 4), s=7, color=color, alpha=.35, lw=0, zorder=3)
        tx.scatter(native[i], i, s=17, marker="D", color=color, zorder=4)
        tx.scatter(cross[i], i, s=18, color=color, edgecolor="white", lw=.4, zorder=4)
    tx.set(xlim=(0, 100), xticks=[0, 50, 100], ylim=(4.65, -.65),
           yticks=range(5), yticklabels=PROMPT_LABELS)
    tx.set_xlabel("Test MAE", fontsize=7.1, labelpad=2)
    tx.set_ylabel("Prompt variant", fontsize=6.5, labelpad=3)
    tx.tick_params(axis="y", labelsize=6.2, length=0, pad=2)
    tx.tick_params(axis="x", labelsize=6.4, length=2, pad=2)
    for label, color in zip(tx.get_yticklabels(), FAMILY_COLORS, strict=True):
        label.set_color(color)
    tx.grid(axis="x", color=LIGHT, lw=.4, zorder=0)
    return {
        "families": FAMILIES, "prompt_labels": PROMPT_LABELS,
        "cosine_matrix": cosines.tolist(), "mae_matrix": errors.tolist(),
        "native_mae": native.tolist(), "cross_family_mae": cross.tolist(),
        "model_count": 6, "relative_layer_count": 11,
        "axes": {"cosine": [-1, 1], "mae": [0, 100]},
        "small_points": "Mean MAE for each of four target variants, averaged over models/layers.",
    }


def structural_hierarchy(fig: plt.Figure) -> dict:
    left, height = COLUMN_LEFTS[2], fig.get_figheight()
    fig.text(left, 2.60 / height, "Hierarchical concepts", fontsize=8.5, fontweight="medium")
    ax = fig.add_axes([left, 1.64 / height, COLUMN_WIDTH, .80 / height])
    ax.set(xlim=(-.06, 2.45), ylim=(-.10, 1.05), aspect="equal")
    ax.axis("off")
    names = ["organism", "animal", "mammal", "dog"]
    colors = [INK, BLUE, OCHRE, PURPLE]
    edge_colors = colors[1:]
    tree = {
        "organism": [.58, .92], "animal": [.56, .65], "plant": [.99, .65],
        "mammal": [.54, .38], "bird": [.99, .38], "dog": [.52, .10], "cat": [.99, .10],
    }

    def arrow(start, end, color, lw=1.05, size=5.7, zorder=3):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=size,
                                     linewidth=lw, color=color, shrinkA=1.3, shrinkB=2.0,
                                     zorder=zorder))

    for parent, child in [("organism", "plant"), ("animal", "bird"), ("mammal", "cat")]:
        arrow(tree[parent], tree[child], LIGHT, lw=.75, size=4.5, zorder=1)
    for parent, child, color in zip(names[:-1], names[1:], edge_colors, strict=True):
        arrow(tree[parent], tree[child], color)
    for name, position in tree.items():
        color = colors[names.index(name)] if name in names else MID
        ax.scatter(*position, s=13, color=color, edgecolor="white", lw=.4, zorder=4)
        if name in names[1:]:
            x, y, align = position[0] - .07, position[1], "right"
        else:
            x, y, align = position[0], position[1] + (.085 if name == "organism" else -.095), "center"
        ax.text(x, y, name, fontsize=6.0, color=INK if name in names else MID,
                ha=align, va="center")
    arrow((1.06, .47), (1.29, .47), MID, lw=.7, size=5.5)

    # Idealized concept vectors have a nonzero shared parent component and
    # three orthogonal refinements. Projection only controls the illustration.
    vectors = np.array([[1, 0, 0, 0], [1, 1, 0, 0],
                        [1, 1, 1, 0], [1, 1, 1, 1]], dtype=float)
    increments = np.diff(vectors, axis=0)
    np.testing.assert_allclose(increments @ increments.T, np.eye(3))
    np.testing.assert_allclose(np.sum(vectors[:-1] * increments, axis=1), 0)
    projection = np.array([[.56, 0, -.28], [0, .45, .20]])
    origin = np.array([1.60, .14])
    points = vectors[:, 1:] @ projection.T + origin
    vertices = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    projected_vertices = vertices @ projection.T + origin
    for i in range(len(vertices)):
        for j in range(i):
            if np.abs(vertices[i] - vertices[j]).sum() == 1:
                ax.plot(*projected_vertices[[i, j]].T, color=LIGHT, lw=.45,
                        linestyle=(0, (2, 2)), zorder=0)
    for i, color in enumerate(edge_colors):
        arrow(points[i], points[i+1], color, lw=1.35, size=6.0)
    label_offsets = [(0, -.12), (0, -.12), (.11, .25), (-.11, .11)]
    for point, name, color, offset in zip(points, names, colors, label_offsets, strict=True):
        ax.scatter(*point, s=13, color=color, edgecolor="white", lw=.4, zorder=4)
        ax.text(*(point+offset), name, fontsize=6.0, ha="center", va="center")
    for i, (offset, color) in enumerate(zip([[-.08, .10], [-.15, 0], [-.14, -.04]], edge_colors, strict=True)):
        midpoint = (points[i] + points[i+1]) / 2 + offset
        ax.text(*midpoint, rf"$\Delta_{i+1}$", fontsize=6.7, ha="center", va="center", color=color)
    fig.text(left + COLUMN_WIDTH / 2, 1.56 / height,
             r"$\Delta_i\perp\Delta_j$  (causal inner product)", fontsize=6.2,
             ha="center", color=MID)
    return {
        "hierarchy_levels": 4,
        "hierarchy_chain": names,
        "hierarchy_sibling_edges": [["organism", "plant"], ["animal", "bird"], ["mammal", "cat"]],
        "hierarchy_tree_positions": tree,
        "hierarchy_idealized_concept_vectors": vectors.tolist(),
        "hierarchy_increment_gram_matrix": (increments @ increments.T).tolist(),
        "hierarchy_projection": projection.tolist(),
        "hierarchy_display_origin": origin.tolist(),
        "hierarchy_projection_type": "Oblique view of three mutually orthogonal increments; projected angles are display angles.",
        "hierarchy_level_colors": dict(zip(names, colors, strict=True)),
    }


def structural_relations(fig: plt.Figure, source: dict) -> dict:
    """Design E: one named relation and three bars on a shared 0–100% scale."""
    left, height = COLUMN_LEFTS[2], fig.get_figheight()
    panel_bottom = .025
    width_points = COLUMN_WIDTH * fig.get_figwidth() * 72
    height_points = (ROW_DIVIDER - panel_bottom) * 72
    ax = fig.add_axes([left, panel_bottom / height, COLUMN_WIDTH,
                       (ROW_DIVIDER - panel_bottom) / height])
    ax.set(xlim=(0, width_points), ylim=(height_points, 0))
    ax.axis("off")
    ax.set_label("factual_relations")
    ax.text(0, 10.8, "Factual relations", fontsize=8.5, fontweight="medium")
    ax.text(0, 23.7, "Relation-specific affine maps", fontsize=7, color=MID)
    ax.text(0, 39, "Miles Davis", fontsize=6.8)
    ax.text(103, 39, "trumpet", fontsize=6.8)
    ax.add_patch(FancyArrowPatch((43, 36.8), (97, 36.8), arrowstyle="-|>",
                                 mutation_scale=6.5, linewidth=.9, color=BLUE,
                                 shrinkA=0, shrinkB=0))
    ax.text(70, 31.4, "plays the", fontsize=6.4, color=BLUE, ha="center")
    ax.text(width_points / 2, 50.7, r"$\hat h_o=W_r h_s+b_r$", fontsize=8, ha="center")
    ax.text(0, 61, "Faithfulness (0–100%)", fontsize=6.5, color=MID)
    ax.text(width_points, 61, source["model"], fontsize=6.3, color=MID, ha="right")

    assert [row["faithfulness_display_percent"] for row in source["rows"]] == [88, 59, 6]
    bar_width = width_points - 1
    for label_y, row in zip([71, 84, 97], source["rows"], strict=True):
        value = row["faithfulness_from_vector_geometry"]
        baseline = row["random_guess_from_vector_geometry"]
        assert 0 <= baseline <= value <= 1
        assert round(value * 100) == row["faithfulness_display_percent"]
        ax.text(0, label_y, row["label"], fontsize=6.7)
        ax.text(width_points, label_y, f"≈{row['faithfulness_display_percent']}%",
                fontsize=6.7, color=BLUE, ha="right")
        bar_y = label_y + 3
        ax.add_patch(Rectangle((0, bar_y), bar_width, 3.8, facecolor="#EEEAE2", lw=0))
        ax.add_patch(Rectangle((0, bar_y), bar_width * value, 3.8, facecolor=BLUE, lw=0))
        ax.plot([bar_width * baseline] * 2, [bar_y - 1, bar_y + 4.8], color=RED, lw=.9)
    return {
        "factual_relation": "musician plays instrument",
        "factual_relation_example_pairs": [["Miles Davis", "trumpet"]],
        "factual_relation_arrow_label": "plays the",
        "factual_relation_mapping": "predicted object state h_hat_o = W_r h_s + b_r",
        "factual_relation_semantics": "Constructed subject–object association and relation-specific affine map.",
    }


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 7.8, "xtick.labelsize": 7.1,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MID,
        "xtick.color": MID, "axes.linewidth": .55,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    configure_fonts()
    sources = json.loads((DATA / "literature_sources.json").read_text())
    factual_relations = json.loads((DATA / "factual_relation_faithfulness.json").read_text())
    fig = plt.figure(figsize=FIGURE_SIZE)
    height = fig.get_figheight()
    for left, title in zip(COLUMN_LEFTS, [
        "(a) Robustness", "(b) Invariance", "(c) Structural\ncorrespondence",
    ], strict=True):
        fig.text(left, (height - .1376) / height, title, va="top", fontsize=8.8,
                 fontweight="semibold", linespacing=1.15)

    sentiment_construction = sentiment(fig)
    sae_construction = sae_feature_matching(fig, sources["paulo_belrose"])
    invariance_results = invariance(fig)
    hierarchy_construction = structural_hierarchy(fig)
    relation_construction = structural_relations(fig, factual_relations)
    for left in [COLUMN_LEFTS[0], COLUMN_LEFTS[2]]:
        fig.add_artist(Line2D(
            [left, left + COLUMN_WIDTH], [ROW_DIVIDER / height] * 2,
            transform=fig.transFigure, color=LIGHT, lw=.55,
        ))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.texts:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.x1, text.get_text()
        assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.y1, text.get_text()
    relation_ax = next(ax for ax in fig.axes if ax.get_label() == "factual_relations")
    panel_bounds = relation_ax.get_window_extent(renderer)
    relation_text_bounds = [(text, text.get_window_extent(renderer)) for text in relation_ax.texts]
    for text, bounds in relation_text_bounds:
        assert bounds.x0 >= panel_bounds.x0 - .01 and bounds.x1 <= panel_bounds.x1 + .01, text.get_text()
        assert bounds.y0 >= panel_bounds.y0 and bounds.y1 <= panel_bounds.y1, text.get_text()
    for i, (text, bounds) in enumerate(relation_text_bounds):
        for other, other_bounds in relation_text_bounds[i + 1:]:
            assert not bounds.overlaps(other_bounds), (text.get_text(), other.get_text())

    for extension in ("pdf", "png", "svg"):
        metadata = {"Creator": Path(__file__).name} if extension == "pdf" else None
        savefig(fig, OUTPUT.with_suffix("." + extension), dpi=240, metadata=metadata)
    svg_path = OUTPUT.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    provenance = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "command": "uv run --group figures python figures/transfer_panels.py",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [
            DATA / "literature_sources.json", DATA / "factual_relation_faithfulness.json",
            INVARIANCE_DATA / "input_provenance.json",
            INVARIANCE_DATA / "pls_directions.csv", INVARIANCE_DATA / "pls_transfer.csv",
        ]},
        "semantic_contract": str((DATA / "semantic_contract.md").relative_to(ROOT)),
        "conceptual_panels": ["a, top: sentiment directions",
                              "c, top: hierarchy and orthogonal refinements",
                              "c, bottom: example of a relation-specific affine map"],
        "published_panels": ["a, bottom: schematic feature matching summarizes SAE overlap across initialization seeds",
                             "c, bottom: three selected GPT-J faithfulness means and random-guess baselines digitized from published Figure 3"],
        "measured_local_panels": ["b, top: prompt-variant direction cosines", "b, bottom: frozen probe transfer"],
        "invariance": invariance_results,
        "factual_relation_faithfulness": factual_relations,
        "construction": {
            **sentiment_construction,
            **sae_construction,
            **hierarchy_construction,
            **relation_construction,
        },
        "figure_inches": list(fig.get_size_inches()),
        "layout_validation": {
            "figure_text_within_canvas": True,
            "factual_relation_text_within_panel": True,
            "factual_relation_text_bounds_disjoint": True,
        },
        "exports": {extension: sha256(OUTPUT.with_suffix("." + extension))
                    for extension in ("pdf", "png", "svg")},
    }
    OUTPUT.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.close(fig)
    print("Exported", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
