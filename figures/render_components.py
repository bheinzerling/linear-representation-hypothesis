#!/usr/bin/env python3
from __future__ import annotations

from paper_figure_style import configure_fonts, register_fonts, savefig

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.ticker import MaxNLocator

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "figures" / "empirical_probe_outputs"
CALIBRATION_DIR = PAPER_ROOT / "data" / "figures" / "mainline_circular_components"
DEFAULT_OUTPUT_PDF = PAPER_ROOT / "figures" / "fig_circular_component_pairs.pdf"
DEFAULT_OUTPUT_PNG = PAPER_ROOT / "figures" / "fig_circular_component_pairs.png"

SUPPORT_COLOR = "#d0d0d0"
SPINE_COLOR = "#dddddd"
CMAP = plt.get_cmap("viridis")
NORM = Normalize(vmin=0, vmax=999)
PARTITION_STYLES = {
    "train": {"marker": "o", "size": 2.8, "alpha": 0.24, "linewidths": 0.0},
    "dev": {"marker": "x", "size": 3.9, "alpha": 0.40, "linewidths": 0.36},
    "test": {"marker": "^", "size": 4.4, "alpha": 0.48, "linewidths": 0.0},
}
PARTITION_LABELS = {
    "train": "Train examples",
    "dev": "Dev examples",
    "test": "Test examples",
}
AXIS_LIMIT = (-1.1, 1.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the mainline Llama-3.1-8B layer-10 circular-component pairs."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-pdf", default=str(DEFAULT_OUTPUT_PDF))
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG))
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(data_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    provenance = json.loads((data_dir / "provenance.json").read_text())
    expected = next(digest for name, digest in provenance["outputs"].items()
                    if Path(name).name == "probe_outputs.csv")
    assert sha256(data_dir / "probe_outputs.csv") == expected
    calibration = json.loads((CALIBRATION_DIR / "provenance.json").read_text())
    assert sha256(data_dir / "provenance.json") == calibration["source_provenance_sha256"]
    assert sha256(CALIBRATION_DIR / "metadata.json") == calibration["outputs"]["metadata.json"]
    metadata = json.loads((CALIBRATION_DIR / "metadata.json").read_text())
    assert metadata["model_name"] == "meta-llama/Llama-3.1-8B" and metadata["layer_idx"] == 10
    frame = pd.read_csv(data_dir / "probe_outputs.csv")
    np.testing.assert_array_equal(frame.value, np.arange(1000))
    assert frame.partition.value_counts().to_dict() == {"train": 800, "dev": 100, "test": 100}
    assert np.isfinite(frame.select_dtypes(include="number")).all().all()
    support = frame[["value"]].copy()
    for display_pair, source_pair in enumerate(reversed(range(5))):
        for offset, kind in enumerate(("sin", "cos")):
            name = f"component_{2 * display_pair + offset + 1:02d}"
            frame[name] = frame[f"circular_{kind}_{source_pair}"]
            support[name] = frame[f"target_{kind}_{source_pair}"]
    frames = {name: frame.loc[frame.partition == name].copy() for name in ("train", "dev", "test")}
    return frames, support, metadata


def _component_sort_key(name: str) -> int:
    match = re.fullmatch(r"component_(\d+)", name)
    if match is None:
        raise ValueError(f"Unexpected component column: {name}")
    return int(match.group(1))


def axis_limit(
    frames: dict[str, pd.DataFrame],
    support: pd.DataFrame,
    component_columns: list[str],
    metadata: dict[str, object],
) -> tuple[float, float]:
    if str(metadata.get("architecture", "circular_reg")) != "circular_cls":
        return AXIS_LIMIT
    values = [support[component_columns].to_numpy(dtype=float)]
    values.extend(frame[component_columns].to_numpy(dtype=float) for frame in frames.values())
    minimum = min(float(value.min()) for value in values)
    maximum = max(float(value.max()) for value in values)
    pad = 0.08 * max(maximum - minimum, 1.0)
    return minimum - pad, maximum + pad


def legend_item(
    label: str,
    *,
    color: str,
    marker: str | None = None,
    linestyle: str | None = None,
    markerfacecolor: str | None = None,
    markeredgewidth: float = 0.8,
) -> HPacker:
    drawing = DrawingArea(26, 12, 0, 0)
    if linestyle is not None:
        line_artist = Line2D([2, 24], [6, 6], color=color, linewidth=1.1, linestyle=linestyle)
        drawing.add_artist(line_artist)
    if marker is not None:
        marker_artist = Line2D(
            [13],
            [6],
            color=color,
            marker=marker,
            markersize=5.0 if marker != "x" else 5.5,
            linestyle="None",
            markerfacecolor=markerfacecolor if markerfacecolor is not None else color,
            markeredgewidth=markeredgewidth,
        )
        drawing.add_artist(marker_artist)
    text = TextArea(label, textprops={"size": 8.5})
    return HPacker(children=[drawing, text], align="center", pad=0, sep=4)


def build_figure(frames: dict[str, pd.DataFrame], support: pd.DataFrame, metadata: dict[str, object]) -> plt.Figure:
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    configure_fonts()
    component_columns = sorted(
        [column for column in support.columns if re.fullmatch(r"component_\d+", column)],
        key=_component_sort_key,
    )
    basis_dim = int(metadata["basis_dim"])
    if len(component_columns) != basis_dim:
        raise ValueError(f"Expected {basis_dim} component columns, found {len(component_columns)}.")
    shared_axis_limit = axis_limit(frames, support, component_columns, metadata)

    fig, axes = plt.subplots(
        basis_dim,
        basis_dim,
        figsize=(7.35, 8.85),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    scatter_ref = None

    for row_idx in range(basis_dim):
        for col_idx in range(basis_dim):
            ax = axes[row_idx, col_idx]
            x_name = component_columns[col_idx]
            y_name = component_columns[row_idx]

            ax.plot(
                support[x_name],
                support[y_name],
                color=SUPPORT_COLOR,
                linewidth=0.55,
                alpha=0.90,
                zorder=1,
            )
            for partition_name in ["train", "dev", "test"]:
                frame = frames[partition_name]
                style = PARTITION_STYLES[partition_name]
                scatter = ax.scatter(
                    frame[x_name],
                    frame[y_name],
                    c=frame["value"],
                    cmap=CMAP,
                    norm=NORM,
                    marker=style["marker"],
                    s=style["size"],
                    alpha=style["alpha"],
                    linewidths=style["linewidths"],
                    edgecolors="none" if style["marker"] != "x" else None,
                    rasterized=False,
                    zorder=2 if partition_name != "test" else 3,
                )
                if scatter_ref is None:
                    scatter_ref = scatter

            ax.set_xlim(*shared_axis_limit)
            ax.set_ylim(*shared_axis_limit)
            ax.grid(False)
            ax.tick_params(axis="both", labelsize=5.8, length=1.6, pad=1.5)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.35)
                spine.set_color(SPINE_COLOR)

            if row_idx == 0:
                ax.set_title(f"C{col_idx + 1}", fontsize=7.2, pad=2.5)
            if col_idx == 0:
                ax.set_ylabel(f"C{row_idx + 1}", fontsize=7.2, labelpad=2.5)
            if row_idx != basis_dim - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col_idx != 0:
                ax.tick_params(axis="y", labelleft=False)

    if str(metadata.get("architecture", "circular_reg")) == "circular_cls":
        for ax in axes[-1, :]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        for ax in axes[:, 0]:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        for ax in axes[-1, :]:
            ax.set_xticks([-1.0, 0.0, 1.0])
        for ax in axes[:, 0]:
            ax.set_yticks([-1.0, 0.0, 1.0])

    fig.subplots_adjust(left=0.085, right=0.985, top=0.888, bottom=0.135, wspace=0.015, hspace=0.015)
    fig.supxlabel("Predicted circular-regression component value", fontsize=9.0, y=0.102)
    fig.supylabel("Predicted circular-regression component value", fontsize=9.0, x=0.028)

    legend_row = HPacker(
        children=[
            legend_item("Target code", color=SUPPORT_COLOR, linestyle="-"),
            legend_item(
                PARTITION_LABELS["train"],
                color="#6b6b6b",
                marker=PARTITION_STYLES["train"]["marker"],
                markerfacecolor="#6b6b6b",
            ),
            legend_item(
                PARTITION_LABELS["dev"],
                color="#6b6b6b",
                marker=PARTITION_STYLES["dev"]["marker"],
                markerfacecolor="none",
                markeredgewidth=0.9,
            ),
            legend_item(
                PARTITION_LABELS["test"],
                color="#6b6b6b",
                marker=PARTITION_STYLES["test"]["marker"],
                markerfacecolor="#6b6b6b",
            ),
        ],
        align="center",
        pad=0,
        sep=9,
    )
    legend = AnchoredOffsetbox(
        loc="upper center",
        child=legend_row,
        pad=0.24,
        borderpad=0.45,
        frameon=True,
        bbox_to_anchor=(0.5, 0.985),
        bbox_transform=fig.transFigure,
    )
    legend.patch.set_edgecolor("#666666")
    legend.patch.set_linewidth(0.8)
    legend.patch.set_facecolor("white")
    fig.add_artist(legend)

    cax = fig.add_axes([0.29, 0.055, 0.42, 0.022])
    cbar = fig.colorbar(scatter_ref, cax=cax, orientation="horizontal")
    cbar.solids.set_rasterized(False)
    cbar.set_ticks([0, 250, 500, 750, 999])
    cbar.ax.tick_params(labelsize=7.5, pad=1.5)
    cbar.set_label("Numeric value", fontsize=8.5, labelpad=2.5)

    return fig


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_pdf = Path(args.output_pdf).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    frames, support, metadata = load_bundle(data_dir)
    fig = build_figure(frames, support, metadata)
    savefig(
        fig,
        output_pdf,
        compact_vector=True,
        bbox_inches="tight",
        metadata={"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None},
    )
    savefig(
        fig,
        output_png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": Path(__file__).name},
    )
    plt.close(fig)
    provenance = {
        "data": {str(path.relative_to(PAPER_ROOT)): sha256(path) for path in (
            data_dir / "probe_outputs.csv", data_dir / "provenance.json",
            CALIBRATION_DIR / "metadata.json", CALIBRATION_DIR / "component_summary.csv",
            CALIBRATION_DIR / "provenance.json",
        )},
        "generation": {
            "script": str(Path(__file__).relative_to(PAPER_ROOT)),
            "script_sha256": sha256(Path(__file__)),
            "command": "uv run --group figures python figures/render_components.py",
            "matplotlib_version": mpl.__version__,
            "pdf_backend": "matplotlib.pdf",
            "pdf_fonttype": 3,
            "font_family": "TeX Gyre Heros",
            "fonts": {face: {"file": Path(font["path"]).name,
                              "sha256": font["sha256"]}
                      for face, font in register_fonts().items()},
        },
        "display": {"component_order": metadata["component_order"], "axis_limits": list(AXIS_LIMIT),
                    "partitions": {name: len(frame) for name, frame in frames.items()},
                    "metrics_use_unclipped_predictions": True},
        "outputs": {str(path.relative_to(PAPER_ROOT)): sha256(path) for path in (output_pdf, output_png)},
    }
    output_pdf.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
