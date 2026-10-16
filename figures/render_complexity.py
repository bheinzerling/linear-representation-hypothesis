#!/usr/bin/env python3
from __future__ import annotations

from paper_figure_style import configure_fonts, savefig

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "figures" / "probe_capacity_paths"
DEFAULT_POINTS_JSONL = DEFAULT_DATA_DIR / "points.jsonl"
DEFAULT_BASELINE_JSONL = DEFAULT_DATA_DIR / "baseline_by_model.jsonl"
DEFAULT_OUTPUT_PDF = PAPER_ROOT / "figures" / "fig_probe_capacity_paths.pdf"
DEFAULT_OUTPUT_PNG = PAPER_ROOT / "figures" / "fig_probe_capacity_paths.png"

MODEL_ORDER = [
    "llama_3_2_1b",
    "llama_3_2_3b",
    "llama_3_1_8b",
    "olmo_3_7b_base",
    "olmo_2_13b",
    "gpt_oss_20b",
]
MODEL_LABELS = {
    "llama_3_2_1b": "Llama 3.2 1B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "llama_3_1_8b": "Llama 3.1 8B",
    "olmo_3_7b_base": "OLMo 3 7B",
    "olmo_2_13b": "OLMo 2 13B",
    "gpt_oss_20b": "GPT-OSS 20B",
}
ARCH_ORDER = ["circular_reg", "circular_cls", "pls", "linear"]
ARCH_COLORS = {
    "circular_reg": "#b2182b",
    "circular_cls": "#2166ac",
    "pls": "#1b7837",
    "linear": "#762a83",
}
ARCH_LABELS = {
    "circular_reg": "Circular reg.",
    "circular_cls": "Circular cls.",
    "pls": "Scalar PLS",
    "linear": "Scalar ridge",
}
ARCH_MARKERS = {
    "circular_reg": "^",
    "circular_cls": "D",
    "pls": "s",
    "linear": "o",
}
BASELINE_COLOR = "#7f7f7f"

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
FONT_RECORD = configure_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the probe-capacity paths figure from a paper-local data bundle."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-pdf", default=str(DEFAULT_OUTPUT_PDF))
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG))
    return parser.parse_args()


def load_frames(*, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    points_jsonl = data_dir / DEFAULT_POINTS_JSONL.name
    baseline_jsonl = data_dir / DEFAULT_BASELINE_JSONL.name
    if not points_jsonl.exists() or not baseline_jsonl.exists():
        raise FileNotFoundError(
            f"Missing frozen probe-complexity inputs in {data_dir}"
        )

    frame = pd.read_json(points_jsonl, lines=True)
    mask = frame["model_alias"].isin(MODEL_ORDER)
    frame = frame.loc[mask].copy()
    frame["mean_abs_error_mean"] = frame["mean_abs_error_mean"].clip(lower=1e-3)
    baseline = pd.read_json(baseline_jsonl, lines=True)
    baseline = baseline.loc[baseline["model_alias"].isin(MODEL_ORDER)].copy()
    baseline["mean_abs_error"] = baseline["mean_abs_error"].clip(lower=1e-3)
    return (
        frame.sort_values(["model_alias", "architecture", "relative_layer", "effective_df"]),
        baseline.sort_values(["model_alias"]),
    )


def blend_with_white(hex_color: str, amount: float) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    base = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(channel * (1.0 - amount) + amount for channel in base)


def legend_item(label: str, *, color: str, marker: str | None = None, linestyle: str = "-") -> HPacker:
    drawing = DrawingArea(24, 10, 0, 0)
    line = Line2D([1, 21], [5, 5], color=color, linewidth=1.0, linestyle=linestyle)
    drawing.add_artist(line)
    if marker is not None:
        marker_artist = Line2D(
            [11],
            [5],
            color=color,
            marker=marker,
            markersize=3.2,
            linestyle="None",
        )
        drawing.add_artist(marker_artist)
    text = TextArea(label, textprops={"size": 8, "weight": "normal"})
    return HPacker(children=[drawing, text], align="center", pad=0, sep=4)


def complexity_ticks(model_dim: float) -> list[int]:
    tick_candidates = [1, 4, 16, 64, 256, 1024, 4096]
    ticks = [tick for tick in tick_candidates if tick <= int(model_dim)]
    if not ticks or ticks[0] != 1:
        ticks = [1] + ticks
    if int(model_dim) not in ticks:
        ticks.append(int(model_dim))
    return ticks


def build_figure(frame: pd.DataFrame, baseline: pd.DataFrame) -> plt.Figure:
    # The ACL text width is 453.543 bp. A matching export keeps the specified
    # font sizes intact when included at full text width.
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.35), sharey=True, constrained_layout=False)
    axes_flat = axes.flatten()

    for idx, model_alias in enumerate(MODEL_ORDER):
        ax = axes_flat[idx]
        model_frame = frame.loc[frame["model_alias"] == model_alias]
        if model_frame.empty:
            continue
        model_dim = float(model_frame["model_dim"].iloc[0])
        baseline_row = baseline.loc[baseline["model_alias"] == model_alias]
        if not baseline_row.empty:
            ax.plot(
                [1.0, model_dim],
                [float(baseline_row["mean_abs_error"].iloc[0])] * 2,
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=0.8,
                alpha=0.95,
            )
        for architecture in ARCH_ORDER:
            arch_frame = model_frame.loc[model_frame["architecture"] == architecture]
            if arch_frame.empty:
                continue
            for relative_layer in sorted(arch_frame["relative_layer"].unique()):
                layer_frame = arch_frame.loc[arch_frame["relative_layer"] == relative_layer]
                if layer_frame.empty:
                    continue
                shade_amount = 0.60 * (1.0 - float(relative_layer))
                color = blend_with_white(ARCH_COLORS[architecture], shade_amount)
                ax.plot(
                    layer_frame["effective_df"],
                    layer_frame["mean_abs_error_mean"],
                    color=color,
                    linestyle="solid",
                    marker=ARCH_MARKERS[architecture],
                    markersize=2.0,
                    linewidth=0.75,
                    alpha=0.95,
                )
        ax.set_title(MODEL_LABELS[model_alias], fontsize=8.5, fontweight="normal", pad=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(1, model_dim)
        ax.set_ylim(0.7, 700)
        x_ticks = complexity_ticks(model_dim)
        x_ticklabels = [str(tick) for tick in x_ticks]
        if model_alias in {"llama_3_2_1b", "olmo_2_13b"} and len(x_ticklabels) >= 2:
            x_ticklabels[-2] = ""
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=7, width=0.5, length=3)
        ax.tick_params(axis="both", which="minor", width=0.4, length=1.5)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", which="major", color="#e5e5e5", linewidth=0.45)
        for name, spine in ax.spines.items():
            spine.set_visible(name in {"left", "bottom"})
            spine.set_linewidth(0.55)
            spine.set_color("#555555")

    fig.subplots_adjust(left=0.11, right=0.96, top=0.85, bottom=0.09, wspace=0.22, hspace=0.36)

    fig.supxlabel("Method-specific complexity", fontsize=8.5)
    fig.supylabel("Held-out MAE", fontsize=8.5)
    legend_first_row = HPacker(
        children=[
            legend_item(
                ARCH_LABELS["circular_reg"], color=ARCH_COLORS["circular_reg"], marker=ARCH_MARKERS["circular_reg"]
            ),
            legend_item(
                ARCH_LABELS["circular_cls"], color=ARCH_COLORS["circular_cls"], marker=ARCH_MARKERS["circular_cls"]
            ),
        ],
        align="center",
        pad=0,
        sep=10,
    )
    legend_second_row = HPacker(
        children=[
            legend_item(ARCH_LABELS["pls"], color=ARCH_COLORS["pls"], marker=ARCH_MARKERS["pls"]),
            legend_item(ARCH_LABELS["linear"], color=ARCH_COLORS["linear"], marker=ARCH_MARKERS["linear"]),
            legend_item("Train-mean baseline", color=BASELINE_COLOR, linestyle="--"),
        ],
        align="center",
        pad=0,
        sep=10,
    )
    legend_rows = VPacker(children=[legend_first_row, legend_second_row],
                          align="center", pad=0, sep=5)
    legend = AnchoredOffsetbox(
        loc="upper center",
        child=legend_rows,
        pad=0.25,
        borderpad=0.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        bbox_transform=fig.transFigure,
    )
    fig.add_artist(legend)
    return fig


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_pdf = Path(args.output_pdf).resolve()
    output_png = Path(args.output_png).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    frame, baseline = load_frames(data_dir=data_dir)
    fig = build_figure(frame, baseline)
    # Export this new physical canvas directly; the shared helper's historical
    # tight-bbox dimensions describe the previous wide layout.
    savefig(fig, output_pdf, bbox_inches=None,
            metadata={"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None})
    savefig(fig, output_png, dpi=300, bbox_inches=None)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path = output_pdf.with_suffix(".provenance.json")
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if "compiled_paper_check" in previous:
        previous["previous_compiled_paper_check"] = previous.pop("compiled_paper_check")
    if "generated_at" in previous:
        previous["source_recorded_at"] = previous.pop("generated_at")
    metadata = json.loads((data_dir / "metadata.json").read_text())
    provenance = previous | {
        "command": "uv run --group figures python figures/render_complexity.py",
        "script_sha256": digest(Path(__file__)),
        "inputs": {path.name: digest(path) for path in (data_dir / "points.jsonl",
                                                       data_dir / "baseline_by_model.jsonl",
                                                       data_dir / "metadata.json")},
        "aggregation": metadata["aggregation"],
        "complexity_definition": metadata["complexity_definition"],
        "canvas_inches": fig.get_size_inches().tolist(),
        "layout": {"rows": 3, "columns": 2, "model_order": MODEL_ORDER,
                   "legend_rows": 2, "export_canvas_pt": (fig.get_size_inches() * 72).tolist(),
                   "target_paper_width_pt": 453.543,
                   "font_sizes_pt": {"ticks": 7, "legend": 8, "titles_and_axis_labels": 8.5},
                   "font_weight": "normal", "grid": "major horizontal guides only"},
        "fonts": FONT_RECORD,
        "sha256": {(path.relative_to(PAPER_ROOT).as_posix() if path.is_relative_to(PAPER_ROOT) else path.name): digest(path)
                   for path in (Path(__file__), data_dir / "points.jsonl", data_dir / "baseline_by_model.jsonl",
                                output_pdf, output_png)},
        "outputs": {path.name: digest(path) for path in (output_pdf, output_png)},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
