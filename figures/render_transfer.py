#!/usr/bin/env python3

from __future__ import annotations

from paper_figure_style import configure_fonts, register_fonts, savefig
from paper_heatmaps import continuous_colorbar, matrix_image

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "figures" / "prompt_style_context_centroid_relative_layers"
DEFAULT_CELLS_CSV = DEFAULT_DATA_DIR / "mean_over_relative_layers_cells.csv"
DEFAULT_OUTPUT_PDF = PAPER_ROOT / "figures" / "fig_prompt_style_centering_transfer.pdf"
DEFAULT_OUTPUT_PNG = PAPER_ROOT / "figures" / "fig_prompt_style_centering_transfer.png"
METHOD_STEMS = {
    "circular": "fig_prompt_variant_transfer_circular_reg",
    "pls": "fig_prompt_variant_transfer_scalar_pls",
    "linear": "fig_prompt_variant_transfer_scalar_ridge",
}
FIT_METHODS = {"circular": "circular_reg", "pls": "pls", "linear": "linear"}

MODEL_ORDER = [
    "llama_3_2_1b",
    "llama_3_2_3b",
    "llama_3_8b",
    "olmo_3_7b_base",
    "olmo_2_13b",
    "gpt_oss_20b",
]
MODEL_LABELS = {
    "llama_3_2_1b": "Llama 3.2 1B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "llama_3_8b": "Llama 3 8B",
    "olmo_3_7b_base": "OLMo 3 7B",
    "olmo_2_13b": "OLMo 2 13B",
    "gpt_oss_20b": "GPT-OSS 20B",
}
ARCH_ORDER = ["circular", "pls", "linear"]
ARCH_LABELS = {
    "circular": "Circular reg.",
    "pls": "Scalar PLS",
    "linear": "Scalar ridge",
}
FAMILY_ORDER = [
    "addition_formula",
    "addition_verbalized",
    "subtraction_formula",
    "subtraction_verbalized",
    "identity_patchscopes",
]
FAMILY_LABELS = {
    "addition_formula": r"$a+b=$",
    "addition_verbalized": "a plus b",
    "subtraction_formula": r"$a-b=$",
    "subtraction_verbalized": "a minus b",
    "identity_patchscopes": r"$b=$",
}
CONDITION_ORDER = ["single_context", "context_centroid"]
COLORBAR_VMIN = 0.0
COLORBAR_VMAX = 140.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matched prompt-variant transfer figures, one per probe method."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=PAPER_ROOT / "figures")
    return parser.parse_args()


def load_frame(*, data_dir: Path) -> pd.DataFrame:
    cells_csv = data_dir / DEFAULT_CELLS_CSV.name
    if not cells_csv.exists():
        raise FileNotFoundError(
            "Missing local prompt-style context-centroid transfer bundle. "
            "Refresh it from the experiment repo artifact snapshot."
        )
    frame = pd.read_csv(cells_csv)
    selected = frame.loc[
        frame["model_alias"].isin(MODEL_ORDER)
        & frame["architecture"].isin(ARCH_ORDER)
        & frame["condition"].isin(CONDITION_ORDER)
    ].copy()
    keys = ["model_alias", "architecture", "condition", "train_family", "test_family"]
    expected_rows = len(MODEL_ORDER) * len(ARCH_ORDER) * len(CONDITION_ORDER) * len(FAMILY_ORDER) ** 2
    if len(selected) != expected_rows or selected.duplicated(keys).any():
        raise ValueError(f"Expected {expected_rows} unique registered cells, found {len(selected)} rows.")
    if selected["mean_over_layers_mae"].isna().any() or set(selected["layer_count"].astype(int)) != {11}:
        raise ValueError("Every displayed cell must contain finite MAE averaged over eleven relative layers.")
    return selected


def build_split_matrix(frame: pd.DataFrame, *, model_alias: str, architecture: str) -> np.ndarray:
    subset = frame.loc[
        (frame["model_alias"] == model_alias) & (frame["architecture"] == architecture)
    ].copy()
    value_map = {
        (
            row["condition"],
            row["train_family"],
            row["test_family"],
        ): float(row["mean_over_layers_mae"])
        for _, row in subset.iterrows()
    }
    matrix = np.full((len(FAMILY_ORDER), len(FAMILY_ORDER) * 2), np.nan, dtype=float)
    for row_idx, train_family in enumerate(FAMILY_ORDER):
        for col_idx, test_family in enumerate(FAMILY_ORDER):
            for half_idx, condition in enumerate(CONDITION_ORDER):
                matrix[row_idx, 2 * col_idx + half_idx] = value_map[(condition, train_family, test_family)]
    return matrix


def format_cell_value(value: float) -> str:
    return f"{value:.1f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_color(value: float, *, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize) -> str:
    rgba = cmap(norm(min(max(value, COLORBAR_VMIN), COLORBAR_VMAX)))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if luminance > 0.62 else "white"


def draw_split_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    cell_fontsize: float,
) -> mpl.image.AxesImage:
    mesh = matrix_image(ax, matrix, cmap=cmap, norm=norm)
    for boundary in [1, 3, 5, 7, 9]:
        ax.axvline(boundary, color="white", linewidth=0.25, solid_capstyle="butt", zorder=3)
    for boundary in [2, 4, 6, 8]:
        ax.axvline(boundary, color="white", linewidth=0.55, solid_capstyle="butt", zorder=3)
    for boundary in [1, 2, 3, 4]:
        ax.axhline(boundary, color="white", linewidth=0.55, solid_capstyle="butt", zorder=3)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                format_cell_value(value),
                ha="center",
                va="center",
                fontsize=cell_fontsize,
                fontstretch="normal",
                fontweight="normal",
                color=text_color(value, cmap=cmap, norm=norm),
            )
    ax.set_xlim(0, 10)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def build_figure(frame: pd.DataFrame, architecture: str = "circular") -> plt.Figure:
    if architecture not in ARCH_ORDER:
        raise ValueError(f"Unknown probe method: {architecture}")
    mpl.rcParams.update({
        "font.size": 7.5, "font.weight": "normal",
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "svg.hashsalt": METHOD_STEMS[architecture],
    })
    configure_fonts()
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.0), constrained_layout=False)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=COLORBAR_VMIN, vmax=COLORBAR_VMAX)
    for panel_idx, (model_alias, ax) in enumerate(zip(MODEL_ORDER, axes.flat, strict=True)):
        values = build_split_matrix(frame, model_alias=model_alias, architecture=architecture)
        mesh = draw_split_heatmap(ax, values, cmap=cmap, norm=norm, cell_fontsize=7)
        # Each outlined pair compares training and test on the same variant.
        for variant_idx in range(len(FAMILY_ORDER)):
            ax.add_patch(mpl.patches.Rectangle(
                (2 * variant_idx, variant_idx), 2, 1, fill=False,
                edgecolor="#333333", linewidth=0.5, zorder=5))
        ax.set_title(MODEL_LABELS[model_alias], fontsize=8.5, fontweight="normal", pad=5)
        ax.set_yticks(np.arange(0.5, 5.0), labels=[FAMILY_LABELS[f] for f in FAMILY_ORDER])
        if panel_idx % 2:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(np.arange(1.0, 10.0, 2.0))
        if panel_idx >= 4:
            ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER],
                               rotation=25, ha="right", fontsize=7.5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=7.5, pad=2)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.90, bottom=0.13,
                        wspace=0.07, hspace=0.43)
    fig.text(0.095, 0.99, ARCH_LABELS[architecture], fontsize=8.5,
             fontweight="normal", ha="left", va="top")
    fig.text(0.99, 0.99, "One context | context mean", fontsize=7.5,
             ha="right", va="top")
    fig.text(0.095, 0.945, "Outline: within variant · other cells: transfer",
             fontsize=7.5, ha="left", va="baseline")
    fig.text(0.095, 0.028, "Rows: training variant\nColumns: test variant",
             fontsize=7.5, ha="left", va="bottom", linespacing=1.15)
    cax = fig.add_axes([0.68, 0.049, 0.30, 0.015])
    cbar = continuous_colorbar(fig, mesh, cax=cax, orientation="horizontal")
    cbar.set_label("Mean held-out MAE", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7, length=2, pad=2)
    cbar.set_ticks([0, 70, 140])
    cbar.ax.xaxis.set_label_position("top")
    return fig


def build_focus_figure(frame: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(3.35, 2.08))
    ax = fig.add_axes([0.22, 0.34, 0.75, 0.56])
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=COLORBAR_VMIN, vmax=COLORBAR_VMAX)
    matrix = build_split_matrix(frame, model_alias="llama_3_8b", architecture="linear")
    mesh = draw_split_heatmap(
        ax,
        matrix,
        cmap=cmap,
        norm=norm,
        cell_fontsize=5.7,
    )
    ax.set_yticks(np.arange(0.5, 5.0, 1.0), labels=[FAMILY_LABELS[family] for family in FAMILY_ORDER])
    ax.set_xticks(np.arange(1.0, 10.0, 2.0))
    ax.set_xticklabels(
        [FAMILY_LABELS[family] for family in FAMILY_ORDER],
        rotation=25,
        ha="right",
    )
    ax.tick_params(axis="both", labelsize=6.7)
    ax.set_xlabel("Test prompt variant", fontsize=7.5, labelpad=3)
    ax.set_ylabel("Training prompt variant", fontsize=7.5, labelpad=3)
    cax = fig.add_axes([0.78, 0.105, 0.18, 0.018])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label("MAE", fontsize=6.3, labelpad=1)
    cbar.ax.tick_params(labelsize=5.8, length=2, pad=1.5)
    cbar.set_ticks([0, 70, 140])
    cbar.ax.xaxis.set_label_position("top")
    return fig


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(data_dir=data_dir)
    input_paths = [data_dir / DEFAULT_CELLS_CSV.name, data_dir / "metadata.json"]
    for architecture in ARCH_ORDER:
        fig = build_figure(frame, architecture)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        minimum_cell_clearance = float("inf")
        for ax in fig.axes[:6]:
            for label in ax.texts:
                box = label.get_window_extent(renderer)
                x, y = label.get_position()
                left, bottom = ax.transData.transform((np.floor(x), np.floor(y)))
                right, top = ax.transData.transform((np.floor(x) + 1, np.floor(y) + 1))
                margin = min(box.x0 - min(left, right), max(left, right) - box.x1,
                             box.y0 - min(bottom, top), max(bottom, top) - box.y1)
                minimum_cell_clearance = min(minimum_cell_clearance, margin * 72 / fig.dpi)
                if margin < 0:
                    raise ValueError(f"Cell label exceeds its half: {label.get_text()}")
        outputs = []
        for ext in ("pdf", "svg", "png"):
            path = output_dir / f"{METHOD_STEMS[architecture]}.{ext}"
            metadata = ({"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None}
                        if ext == "pdf" else {"Date": None} if ext == "svg" else None)
            savefig(fig, path, bbox_inches=None, dpi=300, metadata=metadata)
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            outputs.append(path)
        plt.close(fig)
        subset = frame.loc[frame.architecture == architecture]
        value_map = {
            (row.model_alias, row.train_family, row.test_family, row.condition):
            float(row.mean_over_layers_mae) for row in subset.itertuples()
        }
        paired_cells = [{
            "model": model, "train_variant": train, "test_variant": test,
            "single_context": value_map[(model, train, test, "single_context")],
            "context_mean": value_map[(model, train, test, "context_centroid")],
        } for model in MODEL_ORDER for train in FAMILY_ORDER for test in FAMILY_ORDER]
        record = {
            "figure_id": METHOD_STEMS[architecture], "probe_method": FIT_METHODS[architecture],
            "probe_label": ARCH_LABELS[architecture], "data_values": len(subset),
            "model_order": MODEL_ORDER, "variant_order": FAMILY_ORDER,
            "variant_labels": FAMILY_LABELS, "condition_order": CONDITION_ORDER,
            "paired_cells": paired_cells, "color_range": [COLORBAR_VMIN, COLORBAR_VMAX],
            "color_values": "unrounded MAE, capped at 140 for color only",
            "display_precision": "one decimal throughout",
            "layout": {"rows": 3, "columns": 2, "canvas_pt": [453.6, 504.0],
                       "cell_font_pt": 7, "tick_font_pt": 7.5, "title_font_pt": 8.5,
                       "font_weight": "normal", "font_stretch": "normal"},
            "rendering": {"heatmap": "single nearest-neighbor image per matrix",
                          "colorbar": "single gradient image",
                          "text_and_separators": "vector",
                          "helper_sha256": sha256(Path(__file__).with_name("paper_heatmaps.py"))},
            "within_variant": "black outline; remaining cells show transfer",
            "source_method_check": {
                "source_script": "scripts/run_prompt_style_context_centroid_relative_layers.py",
                "location": "lines 221-222 map circular to circular_reg; fit_magnitude_probe from line 347",
            },
            "inputs": {path.relative_to(PAPER_ROOT).as_posix(): sha256(path) for path in input_paths},
            "generation": {
                "script": Path(__file__).relative_to(PAPER_ROOT).as_posix(),
                "script_sha256": sha256(Path(__file__)),
                "command": "uv run --group figures python figures/render_transfer.py",
                "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": mpl.__version__,
                "fonts": {style: {"file": Path(spec["path"]).name, "sha256": spec["sha256"]}
                          for style, spec in register_fonts().items()},
            },
            "validation": {"minimum_cell_text_margin_pt": minimum_cell_clearance},
            "outputs": {path.relative_to(PAPER_ROOT).as_posix(): sha256(path) for path in outputs},
        }
        (output_dir / f"{METHOD_STEMS[architecture]}.provenance.json").write_text(
            json.dumps(record, indent=2) + "\n")
        print(outputs[0])


if __name__ == "__main__":
    main()
