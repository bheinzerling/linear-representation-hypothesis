"""Recalculate online codelengths and held-out errors from per-example observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["linear", "pls", "circular_reg", "circular_cls"]
NAMES = dict(zip(ORDER, ["Scalar ridge", "Scalar PLS", "Circular reg.", "Circular cls."]))
COLORS = dict(zip(ORDER, ["#35618D", "#3F7F64", "#AC7D00", "#8064A2"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D"]))
CODERS = ["native", "ordinal"]
UNIFORM_BITS = float(np.log2(1000))


def compare(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate observation identities")
    pd.testing.assert_frame_equal(
        actual.set_index(keys).sort_index()[expected.set_index(keys).columns],
        expected.set_index(keys).sort_index(),
        check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-9,
    )


def calculate(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot = json.loads((data / "input_snapshot.json").read_text())
    for name, record in snapshot["files"].items():
        path = (data / name).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError(f"Invalid evidence path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {name}")
    protocol = json.loads((data / "protocol.json").read_text())
    identity = protocol["identity"]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if digest != protocol["source_config_hash"]:
        raise ValueError("Source protocol hash mismatch")
    settings = identity["roster"]
    if len(settings) != 73 or set(identity["coders"]) != set(CODERS):
        raise ValueError("Expected all 73 settings and both probability models")
    records, blocks = [], []
    with np.load(data / "observations.npz", allow_pickle=False) as raw:
        costs = raw["bits"]
        np.testing.assert_array_equal(raw["setting_ids"], [s["id"] for s in settings])
        np.testing.assert_array_equal(raw["order_seeds"], identity["orders"])
        np.testing.assert_array_equal(raw["coders"], identity["coders"])
        np.testing.assert_array_equal(raw["boundaries"], identity["boundaries"])
        if costs.shape != (3, 73, 2, 800):
            raise ValueError("Incomplete per-example codelength array")
        if not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError("Invalid per-example codelength")
        if np.any(costs > -np.log2(.01 / 1000) + 1e-10):
            raise ValueError("Codelength exceeds the uniform-mixture bound")
        np.testing.assert_allclose(costs[..., :3], UNIFORM_BITS)
        dev, test = raw["dev_values"], raw["test_values"]
        if len(dev) != 100 or len(test) != 100 or set(dev) & set(test):
            raise ValueError("Invalid development/test partition")
        for i, order in enumerate(identity["orders"]):
            values = raw["ordered_values"][i]
            if len(set(values)) != 800 or set(values) & (set(dev) | set(test)):
                raise ValueError("Coding and evaluation labels overlap")
            if set(values) | set(dev) | set(test) != set(range(1000)):
                raise ValueError("Incomplete value partition")
            if i and set(values) != set(raw["ordered_values"][0]):
                raise ValueError("Orderings use different training labels")
            for j, setting in enumerate(settings):
                dp, tp = raw["dev_predictions"][j], raw["test_predictions"][j]
                if dp.shape != (100,) or tp.shape != (100,):
                    raise ValueError("Incomplete held-out predictions")
                if np.any(np.r_[dp, tp] < 0) or np.any(np.r_[dp, tp] > 999):
                    raise ValueError("Invalid decoded prediction")
                for c, coder in enumerate(identity["coders"]):
                    per_example = costs[i, j, c]
                    ends = [0, *identity["boundaries"]]
                    total = 0.
                    for start, end in pairwise(ends):
                        block_bits = float(per_example[start:end].sum())
                        total += block_bits
                        blocks.append({
                            "order": order, "setting_id": setting["id"], "coder": coder,
                            "prefix_size": start, "block_size": end - start,
                            "block_bits": block_bits,
                        })
                    records.append({
                        **{k: v for k, v in setting.items() if k != "config"},
                        "order": order, "coder": coder, "online_bits": total,
                        "bits_per_value": total / 800, "uniform_bits": 800 * UNIFORM_BITS,
                        "compression": 800 * UNIFORM_BITS / total,
                        "test_mae": float(np.abs(tp - test).mean()),
                        "dev_mae": float(np.abs(dp - dev).mean()),
                        "test_accuracy": float((tp == test).mean()),
                    })
    metrics, block_frame = pd.DataFrame(records), pd.DataFrame(blocks)
    summaries = []
    for (setting_id, coder), group in metrics.groupby(["id", "coder"]):
        row = group.iloc[0]
        summaries.append({
            "id": setting_id, "architecture": row.architecture, "coder": coder,
            "mean_bits": group.online_bits.mean(), "sd_bits": group.online_bits.std(ddof=1),
            "min_bits": group.online_bits.min(), "max_bits": group.online_bits.max(),
            "bits_per_value": group.online_bits.mean() / 800,
            "sd_bits_per_value": group.online_bits.std(ddof=1) / 800,
            "test_mae": row.test_mae, "dev_mae": row.dev_mae,
            "accuracy": row.test_accuracy, "control": row.control_value,
            "control_name": row.control_name, "old_mae": row.archived_test_mae,
        })
    summary = pd.DataFrame(summaries)
    compare(metrics, pd.read_csv(data / "per_order_metrics.csv"), ["id", "order", "coder"])
    compare(block_frame, pd.read_csv(data / "blocks.csv"),
            ["order", "setting_id", "coder", "prefix_size"])
    compare(summary, pd.read_csv(data / "summary.csv"), ["id", "coder"])
    return metrics, block_frame, summary


def render(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True, sharey=True)
    xmax = max(10.5, float((summary.bits_per_value + summary.sd_bits_per_value).max()) + .35)
    titles = ["(a) Probe-specific scores", "(b) Common numerical-error model"]
    for ax, coder, title in zip(axes, CODERS, titles):
        for method in ORDER:
            rows = summary[(summary.coder == coder) & (summary.architecture == method)]
            ax.errorbar(
                rows.bits_per_value, rows.test_mae, xerr=rows.sd_bits_per_value,
                fmt=MARKERS[method], color=COLORS[method], markersize=3.8,
                elinewidth=.6, capsize=1.5, alpha=.85, label=NAMES[method],
            )
        ax.axvline(UNIFORM_BITS, color="#73706A", linewidth=.7, linestyle=":")
        ax.set(xlabel="Online codelength (bits/value)", xlim=(0, xmax),
               ylim=(.65, 800), yscale="log")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", color="#D8D3CA", linewidth=.5)
    axes[0].set_ylabel("Test MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"fig_probe_mdl.{extension}", dpi=200)
    plt.close(fig)


def table(summary: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Probe & Setting & \multicolumn{2}{c}{Bits/value} & Test MAE \\",
             r" & & Scores & Numerical error & \\", r"\midrule"]
    for method in ORDER:
        selected = []
        for coder in CODERS:
            rows = summary[(summary.architecture == method) & (summary.coder == coder)]
            setting_id = rows.sort_values(["bits_per_value", "id"]).iloc[0].id
            if setting_id not in selected:
                selected.append(setting_id)
        for setting_id in selected:
            pair = summary[summary.id == setting_id].set_index("coder")
            row = pair.loc["native"]
            symbol = r"\lambda" if method == "linear" else ("k" if method == "pls" else "d")
            value = f"{row.control:.3g}" if method == "linear" else str(int(row.control))
            setting = f"${symbol}={value}$"
            costs = [f"{pair.loc[c, 'bits_per_value']:.2f}" + r"\,$\pm$\,"
                     + f"{pair.loc[c, 'sd_bits_per_value']:.2f}" for c in CODERS]
            lines.append(f"{NAMES[method]} & {setting} & {costs[0]} & {costs[1]} & "
                         + f"{row.test_mae:.2f}" + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics, blocks, summary = calculate(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "per_order_metrics.csv", index=False)
    blocks.to_csv(args.output_dir / "blocks.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "table_probe_mdl.tex").write_text(table(summary))
    render(summary, args.output_dir)
    result = {"settings": 73, "orders": 3, "coders": 2, "encoded_labels": 800,
              "per_example_costs": 3 * 73 * 2 * 800, "per_order_records": len(metrics),
              "summary_rows": len(summary), "uniform_bits": 800 * UNIFORM_BITS}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
