from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lrh.io import require_files, write_json

OCHRE = "#ac7d00"
BLUE = "#35618d"
DARK = "#25231f"
MID = "#73706a"
LIGHT = "#d8d3ca"
PARTIAL_LINESTYLE = (0, (2.5, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the main-text LRH mention figure.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
            "font.size": 8.5,
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


def plot_counts(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
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

    if set(grouped) != set(styles):
        raise ValueError(f"Unexpected series: {sorted(grouped)}")
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
                [int(last_complete["result_count"]), int(last_partial["result_count"])],
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


def main() -> None:
    args = parse_args()
    data_path = args.data_dir / "fulltext_comparison_counts.csv"
    require_files([data_path, args.data_dir / "query_definitions.json"])
    rows = read_rows(data_path)
    lookup = {(row["series_id"], int(row["year"])): int(row["result_count"]) for row in rows}
    expected = {
        ("google_scholar_current", 2026): 676,
        ("openalex_current", 2026): 526,
        ("google_scholar_survey", 2025): 447,
    }
    if any(lookup.get(key) != value for key, value in expected.items()):
        raise ValueError("The frozen annual counts do not match the main-text annotations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    plot_counts(ax, rows)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.21)
    fig.savefig(
        args.output_dir / "fig_lrh_fulltext_mentions.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "Linear? Representation? Hypothesis?",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        args.output_dir / "fig_lrh_fulltext_mentions.png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "Linear? Representation? Hypothesis?"},
    )
    plt.close(fig)
    write_json(
        args.output_dir / "metrics.json",
        {
            "annual_counts": {
                f"{series_id}:{year}": count for (series_id, year), count in sorted(lookup.items())
            },
            "main_annotations": {f"{key[0]}:{key[1]}": value for key, value in expected.items()},
        },
    )


if __name__ == "__main__":
    main()
