#!/usr/bin/env python3

from __future__ import annotations

from paper_figure_style import configure_fonts, savefig

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data/figures/lrh_uptake_and_use"

OCHRE = "#ac7d00"
BLUE = "#35618d"
GREEN = "#3f7f64"
PURPLE = "#76558a"
DARK = "#25231f"
MID = "#73706a"
LIGHT = "#d8d3ca"
PARTIAL_LINESTYLE = (0, (2.5, 2))


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.edgecolor": LIGHT,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MID,
            "ytick.color": MID,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "grid.color": "#e7e2d9",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    configure_fonts()


def save(fig: plt.Figure, stem: str) -> None:
    savefig(fig, HERE / f"{stem}.pdf", bbox_inches="tight")
    savefig(fig, HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_broad_search(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    styles = {
        "google_scholar_current": (
            "Google Scholar (through July 2026)",
            OCHRE,
            "o",
            "-",
            2.0,
        ),
        "google_scholar_survey": (
            "Google Scholar (Dec. 2025 survey)",
            MID,
            "o",
            (0, (3, 2)),
            1.5,
        ),
        "openalex_current": (
            "OpenAlex (through July 2026)",
            BLUE,
            "s",
            "-",
            1.8,
        ),
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["series_id"]].append(row)

    for series_id, (label, color, marker, linestyle, linewidth) in styles.items():
        series = sorted(grouped[series_id], key=lambda row: int(row["year"]))
        complete = [row for row in series if row["partial_year"].lower() == "false"]
        partial = [row for row in series if row["partial_year"].lower() == "true"]
        ax.plot(
            [int(row["year"]) for row in complete],
            [int(row["result_count"]) for row in complete],
            color=color,
            marker=marker,
            markersize=3.8,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=3,
        )
        if partial:
            last_complete, last_partial = complete[-1], partial[-1]
            ax.plot(
                [int(last_complete["year"]), int(last_partial["year"])],
                [
                    int(last_complete["result_count"]),
                    int(last_partial["result_count"]),
                ],
                color=color,
                linewidth=linewidth,
                linestyle=PARTIAL_LINESTYLE,
                zorder=2,
            )
            ax.scatter(
                [int(last_partial["year"])],
                [int(last_partial["result_count"])],
                marker=marker,
                s=25,
                facecolors="white",
                edgecolors=color,
                linewidths=1.1,
                zorder=4,
            )

    ax.set(xlim=(2021.8, 2026.2), ylim=(0, 725))
    ax.set_xticks(range(2022, 2027))
    ax.set_yticks([0, 200, 400, 600])
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Exact-phrase search results")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color=DARK,
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.1,
            linestyle=PARTIAL_LINESTYLE,
        )
    )
    labels.append("Partial counts for 2026")
    ax.legend(
        handles,
        labels,
        loc="upper left",
        fontsize=6.7,
        handlelength=2.4,
        borderaxespad=0.3,
        labelspacing=0.35,
    )
    for year, value, label, color, offset in (
        (2026, 676, "676", OCHRE, (-4, 5)),
        (2026, 526, "526", BLUE, (-4, 5)),
        (2025, 447, "447", MID, (4, -10)),
    ):
        ax.annotate(
            label,
            (year, value),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            color=color,
            fontsize=7.2,
        )


def plot_purpose_composition(
    ax: plt.Axes,
    rows: list[dict[str, str]],
) -> None:
    purpose_order = [
        ("define_position", "Define / position", BLUE),
        ("motivate_apply", "Motivate / apply", OCHRE),
        ("test_support", "Test / support", GREEN),
        ("qualify_challenge", "Qualify / challenge", PURPLE),
    ]
    lookup = {(int(row["year"]), row["purpose_id"]): row for row in rows}
    years = sorted({int(row["year"]) for row in rows})
    shares = [
        [
            100 * float(lookup[(year, purpose_id)]["sample_share"])
            for year in years
        ]
        for purpose_id, _, _ in purpose_order
    ]
    sample_sizes = [
        int(lookup[(year, purpose_order[0][0])]["sample_substantive_papers"])
        for year in years
    ]

    bottom = [0.0] * len(years)
    for (_, label, color), values in zip(purpose_order, shares, strict=True):
        ax.bar(
            years,
            values,
            bottom=bottom,
            width=0.72,
            label=label,
            color=color,
            alpha=0.86,
            linewidth=0.45,
            edgecolor="white",
            zorder=3,
        )
        bottom = [lower + value for lower, value in zip(bottom, values, strict=True)]
    ax.set_xticks(
        years,
        [
            f"{year}{'*' if year == 2026 else ''}\n$n$={sample_size}"
            for year, sample_size in zip(years, sample_sizes)
        ],
    )
    ax.set(xlim=(2022.55, 2026.45), ylim=(0, 100))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Publication year and coded sample size")
    ax.set_ylabel("Weighted share of coded papers (%)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
        fontsize=6.6,
        handlelength=1.7,
        columnspacing=0.8,
        borderaxespad=0.3,
        labelspacing=0.35,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        choices=("all", "broad", "purpose"),
        default="all",
    )
    args = parser.parse_args()
    setup_style()
    if args.figure in {"all", "broad"}:
        broad_fig, broad_ax = plt.subplots(figsize=(3.35, 2.65))
        plot_broad_search(
            broad_ax,
            read_csv("fulltext_comparison_counts.csv"),
        )
        broad_fig.subplots_adjust(
            left=0.19,
            right=0.98,
            top=0.98,
            bottom=0.21,
        )
        save(broad_fig, "fig_lrh_fulltext_mentions")

    if args.figure in {"all", "purpose"}:
        purpose_fig, purpose_ax = plt.subplots(figsize=(3.35, 2.90))
        plot_purpose_composition(
            purpose_ax,
            read_csv("annotation_purpose_estimates.csv"),
        )
        purpose_fig.subplots_adjust(
            left=0.20,
            right=0.98,
            top=0.78,
            bottom=0.23,
        )
        save(purpose_fig, "fig_lrh_purpose_composition")


if __name__ == "__main__":
    main()
