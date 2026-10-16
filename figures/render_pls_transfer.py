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
from matplotlib.patches import Rectangle


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "figures" / "prompt_transfer_robustness"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "pls_family_transfer.csv"
DEFAULT_PROVENANCE = DEFAULT_DATA_DIR / "provenance.json"
DEFAULT_OUTPUT_PDF = PAPER_ROOT / "figures" / "fig_prompt_family_pls_transfer.pdf"
DEFAULT_OUTPUT_PNG = PAPER_ROOT / "figures" / "fig_prompt_family_pls_transfer.png"
DEFAULT_OUTPUT_SVG = DEFAULT_OUTPUT_PDF.with_suffix(".svg")
DEFAULT_OUTPUT_METADATA = PAPER_ROOT / "out/figures/pls-transfer.json"

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
MAE_COLUMNS = ("raw_affine_mae", "family_mean_aligned_mae")
MAE_MAX = 100.0
FIGURE_SIZE = (6.3, 2.6)
PANEL_SIDE = 1.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG)
    parser.add_argument("--output-metadata", type=Path, default=DEFAULT_OUTPUT_METADATA)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = data_dir / DEFAULT_INPUT.name
    provenance_path = data_dir / DEFAULT_PROVENANCE.name
    frame = pd.read_csv(input_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {
        "train_family",
        "test_family",
        "raw_affine_mae",
        "family_mean_aligned_mae",
        "direction_cosine",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    expected = {(train, test) for train in FAMILY_ORDER for test in FAMILY_ORDER}
    observed = set(zip(frame["train_family"], frame["test_family"], strict=True))
    if len(frame) != 25 or observed != expected:
        raise ValueError("Expected one row for every ordered prompt-family pair.")
    if not np.isfinite(frame[list(MAE_COLUMNS) + ["direction_cosine"]].to_numpy()).all():
        raise ValueError("Figure inputs must be finite.")
    if (frame[list(MAE_COLUMNS)] < 0).any().any():
        raise ValueError("MAE values must be nonnegative.")
    if not frame["direction_cosine"].between(-1, 1).all():
        raise ValueError("Direction cosine must lie in [-1, 1].")
    cosine = matrix(frame, "direction_cosine")
    if not np.allclose(cosine, cosine.T, atol=1e-10) or not np.allclose(
        np.diag(cosine), 1.0, atol=1e-12
    ):
        raise ValueError("The direction-cosine matrix must be symmetric with unit diagonal.")
    validate_summaries(frame)
    return frame, provenance


def matrix(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    lookup = {
        (row.train_family, row.test_family): float(getattr(row, value_column))
        for row in frame.itertuples(index=False)
    }
    return np.array(
        [[lookup[(train, test)] for test in FAMILY_ORDER] for train in FAMILY_ORDER],
        dtype=float,
    )


def validate_summaries(frame: pd.DataFrame) -> None:
    native = frame["train_family"].eq(frame["test_family"])
    observed = {
        "native_mae": float(frame.loc[native, "raw_affine_mae"].mean()),
        "raw_transfer_mae": float(frame.loc[~native, "raw_affine_mae"].mean()),
        "aligned_transfer_mae": float(
            frame.loc[~native, "family_mean_aligned_mae"].mean()
        ),
        "mean_pairwise_cosine": float(
            frame.loc[
                frame["train_family"].map(FAMILY_ORDER.index)
                < frame["test_family"].map(FAMILY_ORDER.index),
                "direction_cosine",
            ].mean()
        ),
    }
    expected = {
        "native_mae": 13.566121212121212,
        "raw_transfer_mae": 58.131174242424244,
        "aligned_transfer_mae": 42.171174242424236,
        "mean_pairwise_cosine": 0.6351300925350061,
    }
    for key, expected_value in expected.items():
        if not np.isclose(observed[key], expected_value, atol=5e-11):
            raise ValueError(
                f"Registered summary drift for {key}: {observed[key]} != {expected_value}"
            )


def text_color(
    value: float, *, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize
) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#111111" if luminance > 0.61 else "white"


def draw_matrix(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    value_format: str,
    show_ylabels: bool,
) -> mpl.image.AxesImage:
    mesh = matrix_image(ax, values, cmap=cmap, norm=norm)
    for boundary in range(1, 5):
        ax.axhline(boundary, color="white", linewidth=0.5, solid_capstyle="butt", zorder=3)
        ax.axvline(boundary, color="white", linewidth=0.5, solid_capstyle="butt", zorder=3)
    for index in range(5):
        ax.add_patch(
            Rectangle(
                (index, index),
                1,
                1,
                fill=False,
                edgecolor="#111111",
                linewidth=0.5,
                zorder=4,
            )
        )
    for row in range(5):
        for column in range(5):
            value = float(values[row, column])
            ax.text(
                column + 0.5,
                row + 0.5,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color(value, cmap=cmap, norm=norm),
                zorder=5,
            )
    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(5) + 0.5, [FAMILY_LABELS[x] for x in FAMILY_ORDER])
    ax.set_xticklabels(
        [FAMILY_LABELS[x] for x in FAMILY_ORDER], rotation=28, ha="right"
    )
    ax.set_yticks(np.arange(5) + 0.5)
    if show_ylabels:
        ax.set_yticklabels([FAMILY_LABELS[x] for x in FAMILY_ORDER])
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=2, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def draw_cosine_triangle(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.image.AxesImage:
    """Draw each cross-family cosine once, omitting the trivial diagonal."""
    displayed = np.full((4, 4), np.nan)
    for row in range(1, 5):
        display_row = row - 1
        for column in range(row):
            value = float(values[row, column])
            displayed[display_row, column] = value
            ax.text(
                column + 0.5,
                display_row + 0.5,
                format(value, ".2f"),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color(value, cmap=cmap, norm=norm),
            )
    mesh = matrix_image(ax, np.ma.masked_invalid(displayed), cmap=cmap, norm=norm)
    for boundary in range(1, 4):
        ax.axhline(boundary, color="white", linewidth=0.5, solid_capstyle="butt", zorder=3)
        ax.axvline(boundary, color="white", linewidth=0.5, solid_capstyle="butt", zorder=3)
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_aspect("equal")
    ax.set_xticks(
        np.arange(4) + 0.5,
        [FAMILY_LABELS[x] for x in FAMILY_ORDER[:-1]],
    )
    ax.set_xticklabels(
        [FAMILY_LABELS[x] for x in FAMILY_ORDER[:-1]], rotation=28, ha="right"
    )
    ax.set_yticks(
        np.arange(4) + 0.5,
        [FAMILY_LABELS[x] for x in FAMILY_ORDER[1:]],
    )
    ax.tick_params(length=0, pad=2, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "font.weight": "normal",
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    configure_fonts()
    figure = plt.figure(figsize=FIGURE_SIZE)

    def axes_at(left: float, bottom: float, width: float, height: float) -> plt.Axes:
        """Place axes in inches so plot and type sizes match the main figures."""
        return figure.add_axes([
            left / FIGURE_SIZE[0], bottom / FIGURE_SIZE[1],
            width / FIGURE_SIZE[0], height / FIGURE_SIZE[1],
        ])

    raw_ax = axes_at(0.64, 1.00, PANEL_SIDE, PANEL_SIDE)
    aligned_ax = axes_at(2.29, 1.00, PANEL_SIDE, PANEL_SIDE)
    cosine_ax = axes_at(4.63, 1.00, PANEL_SIDE, PANEL_SIDE)

    mae_cmap = mpl.colormaps["viridis"]
    mae_norm = mpl.colors.Normalize(vmin=0.0, vmax=MAE_MAX)
    raw_mesh = draw_matrix(
        raw_ax,
        matrix(frame, "raw_affine_mae"),
        cmap=mae_cmap,
        norm=mae_norm,
        value_format=".0f",
        show_ylabels=True,
    )
    draw_matrix(
        aligned_ax,
        matrix(frame, "family_mean_aligned_mae"),
        cmap=mae_cmap,
        norm=mae_norm,
        value_format=".0f",
        show_ylabels=False,
    )

    cosine_cmap = mpl.colormaps["RdBu_r"]
    cosine_norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cosine_mesh = draw_cosine_triangle(
        cosine_ax,
        matrix(frame, "direction_cosine"),
        cmap=cosine_cmap,
        norm=cosine_norm,
    )

    raw_ax.set_title("(a) Original centroids", loc="left", pad=3)
    aligned_ax.set_title("(b) Mean alignment", loc="left", pad=3)
    cosine_ax.set_title("(c) Probe-direction cosine", loc="left", pad=3)

    mae_cax = axes_at(0.84, 0.18, 2.65, 0.07)
    mae_colorbar = continuous_colorbar(figure, raw_mesh, cax=mae_cax, orientation="horizontal")
    mae_colorbar.set_ticks([0, 50, 100])
    mae_colorbar.set_label("Held-out MAE", fontsize=8, labelpad=3)
    mae_colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    mae_colorbar.ax.xaxis.set_label_position("top")

    cosine_cax = axes_at(4.63, 0.18, PANEL_SIDE, 0.07)
    cosine_colorbar = continuous_colorbar(figure, cosine_mesh, cax=cosine_cax, orientation="horizontal")
    cosine_colorbar.set_ticks([-1, 0, 1])
    cosine_colorbar.set_label("Cosine", fontsize=8, labelpad=3)
    cosine_colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    cosine_colorbar.ax.xaxis.set_label_position("top")

    figure.text(0.03 / FIGURE_SIZE[0], 1.675 / FIGURE_SIZE[1],
                "Source prompt variant", rotation=90, ha="left", va="center", fontsize=8)
    figure.text(2.14 / FIGURE_SIZE[0], 0.49 / FIGURE_SIZE[1],
                "Target prompt variant", ha="center", va="bottom", fontsize=8)
    return figure


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    input_path = data_dir / DEFAULT_INPUT.name
    provenance_path = data_dir / DEFAULT_PROVENANCE.name
    output_pdf = args.output_pdf.resolve()
    output_png = args.output_png.resolve()
    output_svg = args.output_svg.resolve()
    output_metadata = args.output_metadata.resolve()
    frame, provenance = load_data(data_dir)
    figure = build_figure(frame)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    savefig(
        figure,
        output_pdf,
        bbox_inches=None,
        metadata={"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None},
    )
    savefig(
        figure,
        output_png,
        dpi=600,
        bbox_inches=None,
        metadata={"Software": Path(__file__).name},
    )
    savefig(figure, output_svg, bbox_inches=None, metadata={"Date": None})
    output_svg.write_text("\n".join(line.rstrip() for line in output_svg.read_text().splitlines()) + "\n")
    plt.close(figure)
    output_metadata.write_text(
        json.dumps(
            {
                "figure_id": "fig_prompt_family_pls_transfer",
                "source_file": input_path.name,
                "source_sha256": sha256(input_path),
                "provenance_file": provenance_path.name,
                "provenance_sha256": sha256(provenance_path),
                "source_repository_commit": provenance["source_commit"],
                "family_order": FAMILY_ORDER,
                "variant_display_labels": FAMILY_LABELS,
                "probe_label": "Scalar PLS",
                "rendering": {
                    "heatmap": "single nearest-neighbor image per matrix",
                    "colorbar": "single gradient image",
                    "text_and_separators": "vector",
                    "canvas_inches": figure.get_size_inches().tolist(),
                    "layout": "three panels in one row",
                    "panel_side_inches": PANEL_SIDE,
                    "colorbar_orientation": "horizontal",
                    "font_sizes_pt": {"cells": 7, "ticks": 7.5, "labels": 8, "titles": 8.5},
                    "font_weight": "normal",
                    "font_files": register_fonts(),
                    "helper_sha256": sha256(Path(__file__).with_name("paper_heatmaps.py")),
                    "style_helper_sha256": sha256(Path(__file__).with_name("paper_figure_style.py")),
                },
                "panels": [
                    "raw_affine_mae",
                    "family_mean_aligned_mae",
                    "direction_cosine",
                ],
                "mae_color_range": [0.0, MAE_MAX],
                "cosine_color_range": [-1.0, 1.0],
                "generation_script": Path(__file__).name,
                "generation_script_sha256": sha256(Path(__file__).resolve()),
                "generation_command": "uv run --group figures python figures/render_pls_transfer.py",
                "output_pdf_sha256": sha256(output_pdf),
                "output_png_sha256": sha256(output_png),
                "output_svg_sha256": sha256(output_svg),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
