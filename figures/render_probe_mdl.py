"""Render appendix MDL evidence from the frozen numerical experiment summary.

uv run --group figures python figures/render_probe_mdl.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figure_style import configure_fonts, savefig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/probe_mdl"
OUTPUT = ROOT / "figures/fig_probe_mdl"
ORDER = ["linear", "pls", "circular_reg", "circular_cls"]
NAMES = {"linear": "Scalar ridge", "pls": "Scalar PLS", "circular_reg": "Circular reg.", "circular_cls": "Circular cls."}
COLORS = {"linear": "#35618D", "pls": "#3F7F64", "circular_reg": "#AC7D00", "circular_cls": "#8064A2"}
MARKERS = {"linear": "o", "pls": "s", "circular_reg": "^", "circular_cls": "D"}
CODERS = ["native", "ordinal"]
UNIFORM = float(np.log2(1000))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((DATA / "input_snapshot.json").read_text())
    for name, record in manifest["files"].items():
        assert digest(DATA / name) == record["sha256"], name
    summary = pd.read_csv(DATA / "summary.csv")
    records = pd.read_csv(DATA / "per_order_metrics.csv")
    assert len(summary) == 146 and len(records) == 438
    assert set(summary["architecture"]) == set(ORDER)
    assert set(summary["coder"]) == set(CODERS)
    assert not summary.duplicated(["id", "coder"]).any()
    assert records.groupby(["id", "coder"]).size().eq(3).all()
    assert records.groupby(["id", "coder"])["order"].nunique().eq(3).all()
    assert records["order"].isin(manifest["comparison"]["order_seeds"]).all()
    assert summary.groupby("id")["test_mae"].nunique().eq(1).all()
    assert summary.groupby("coder").size().eq(73).all()
    for _, row in summary.iterrows():
        group = records[(records["id"] == row["id"]) & (records["coder"] == row["coder"])]
        np.testing.assert_allclose(row["mean_bits"], group["online_bits"].mean(), rtol=1e-12)
        np.testing.assert_allclose(row["sd_bits"], group["online_bits"].std(ddof=1), atol=1e-10)
        np.testing.assert_allclose(row["bits_per_value"], row["mean_bits"] / 800, atol=1e-12)
        np.testing.assert_allclose(row["sd_bits_per_value"], row["sd_bits"] / 800, atol=1e-12)
        np.testing.assert_allclose(row["test_mae"], group["test_mae"], atol=1e-12)
        assert group["control_value"].eq(row["control"]).all()
    assert np.isfinite(summary[["bits_per_value", "sd_bits_per_value", "test_mae"]]).all().all()
    assert summary["test_mae"].gt(0).all()
    return summary, manifest


def build(summary: pd.DataFrame):
    configure_fonts()
    mpl.rcParams.update({"font.size": 8.5, "mathtext.fontset": "cm", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True, sharey=True)
    xmax = max(10.5, float((summary["bits_per_value"] + summary["sd_bits_per_value"]).max()) + .35)
    ymin = min(.65, float(summary["test_mae"].min()) * .8)
    point_counts = {}
    for ax, coder, title in zip(axes, CODERS,
                                ["(a) Probe-specific scores", "(b) Common numerical-error model"], strict=True):
        count = 0
        for method in ORDER:
            rows = summary[(summary["coder"] == coder) & (summary["architecture"] == method)]
            ax.errorbar(rows["bits_per_value"], rows["test_mae"], xerr=rows["sd_bits_per_value"],
                        fmt=MARKERS[method], color=COLORS[method], markersize=3.8,
                        alpha=.85, elinewidth=.6, capsize=1.5, label=NAMES[method])
            count += len(rows)
        ax.axvline(UNIFORM, color="#73706A", linewidth=.7, linestyle=":")
        ax.set(xlabel="Online codelength (bits/value)", xlim=(0, xmax), ylim=(ymin, 800), yscale="log")
        ax.set_title(title, loc="left", fontsize=8.5, pad=8)
        ax.grid(axis="y", color="#D8D3CA", linewidth=.5)
        point_counts[coder] = count
    axes[0].set_ylabel("Test MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    assert point_counts == {"native": 73, "ordinal": 73}
    return fig, {"point_counts": point_counts, "x_limits": [0, xmax], "y_limits": [ymin, 800],
                 "uniform_bits_per_value": UNIFORM, "canvas_inches": [7.1, 3.15]}


def table(summary: pd.DataFrame) -> tuple[str, list[dict]]:
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Probe & Setting & \multicolumn{2}{c}{Bits/value} & Test MAE \\",
             r" & & Scores & Numerical error & \\", r"\midrule"]
    rows_out = []
    for method in ORDER:
        selected = []
        for coder in CODERS:
            candidates = summary[(summary["architecture"] == method) & (summary["coder"] == coder)]
            selected.append(candidates.sort_values(["bits_per_value", "id"]).iloc[0])
        # The two circular-regression likelihoods select different settings.
        for j, row in enumerate(selected):
            if j == 1 and row["id"] == selected[0]["id"]:
                continue
            pair = summary[summary["id"] == row["id"]].set_index("coder")
            if method == "linear":
                setting = rf"$\lambda={row['control']:.3g}$"
            else:
                symbol = "k" if method == "pls" else "d"
                setting = rf"${symbol}={int(row['control'])}$"
            costs = [f"{pair.loc[c, 'bits_per_value']:.2f}" + r"\,$\pm$\," + f"{pair.loc[c, 'sd_bits_per_value']:.2f}" for c in CODERS]
            lines.append(f"{NAMES[method]} & {setting} & {costs[0]} & {costs[1]} & {row['test_mae']:.2f}" + r" \\")
            rows_out.append({"architecture": method, "setting_id": row["id"], "control": float(row["control"])})
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    assert len(rows_out) == 5
    return "\n".join(lines) + "\n", rows_out


def main() -> None:
    summary, manifest = load_inputs()
    fig, semantics = build(summary)
    outputs = {}
    for extension in ("pdf", "png", "svg"):
        path = OUTPUT.with_suffix("." + extension)
        metadata = {"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None} if extension == "pdf" else ({"Creator": Path(__file__).name, "Date": None} if extension == "svg" else None)
        savefig(fig, path, dpi=300, metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs[str(path.relative_to(ROOT))] = digest(path)
    plt.close(fig)
    table_text, rows = table(summary)
    table_path = ROOT / "figures/table_probe_mdl.tex"
    table_path.write_text(table_text)
    outputs[str(table_path.relative_to(ROOT))] = digest(table_path)
    record = {"inputs": manifest, "input_snapshot_sha256": digest(DATA / "input_snapshot.json"),
              "figure": semantics, "selected_settings": rows, "fonts": configure_fonts(),
              "generation": {"script": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__)),
                             "command": "uv run --group figures python figures/render_probe_mdl.py",
                             "matplotlib": mpl.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "outputs": outputs}
    OUTPUT.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Rendered appendix MDL figure and five-setting comparison table; 146 summary rows verified.")


if __name__ == "__main__":
    main()
