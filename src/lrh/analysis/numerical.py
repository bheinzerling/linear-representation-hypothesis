"""Verify and render current numerical panels from the compact frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lrh.analysis.uncertainty import probe_uncertainty, steering_uncertainty
from lrh.io import require_files, sha256, write_json

METHODS = ["linear", "pls", "circular_reg", "circular_cls"]
NAMES = ["Scalar ridge", "Scalar PLS", "Circular regression", "Circular classifier"]
COLORS = ["#AC7D00", "#8064A2", "#3F7F64", "#35618D"]
FAMILIES = [
    "addition_formula",
    "addition_verbalized",
    "subtraction_formula",
    "subtraction_verbalized",
    "identity_patchscopes",
]
FAMILY_LABELS = ["a + b", "a plus b", "a − b", "a minus b", "b = b"]
TASKS = FAMILIES + ["parity", "magnitude_comparison"]
TASK_LABELS = [
    "Addition",
    "Verbal addition",
    "Subtraction",
    "Verbal subtraction",
    "Copying",
    "Parity",
    "Comparison",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "Linear? Representation? Hypothesis?",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Linear? Representation? Hypothesis?"},
    )
    plt.close(fig)


def verify_inputs(data_dir: Path) -> dict:
    provenance = json.loads((data_dir / "input_provenance.json").read_text())
    for name, entry in provenance["files"].items():
        path = data_dir / name
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Changed current-panel input: {name}")
    return provenance


def verify_intervals(computed: pd.DataFrame, certificate: Path, keys: list[str]) -> None:
    frozen = pd.read_csv(certificate)
    observed = computed.sort_values(keys).reset_index(drop=True)
    expected = frozen.sort_values(keys).reset_index(drop=True)
    if list(observed.columns) != list(expected.columns):
        raise ValueError(f"Uncertainty schema drift: {certificate.name}")
    for column in observed.columns:
        if pd.api.types.is_numeric_dtype(observed[column]):
            np.testing.assert_allclose(
                observed[column], expected[column], atol=1e-10, rtol=0, equal_nan=True
            )
        else:
            pd.testing.assert_series_equal(observed[column], expected[column], check_names=False)


def probe_panels(data_dir: Path, output_dir: Path) -> dict:
    output = pd.read_csv(data_dir / "probe_outputs.csv.gz")
    point_metrics = pd.read_csv(data_dir / "probe_metrics.csv").set_index("architecture")
    if output.partition.value_counts().to_dict() != {"train": 800, "dev": 100, "test": 100}:
        raise ValueError("Probe-output split drift")
    np.testing.assert_array_equal(output.value, np.arange(1000))
    test = output.loc[output.partition.eq("test")]
    for name, predictions in (
        ("linear", np.rint(test.ridge_prediction)),
        ("circular_reg", test.circular_prediction),
    ):
        np.testing.assert_allclose(
            np.abs(predictions - test.value).mean(),
            point_metrics.loc[name, "mean_abs_error"],
            atol=1e-12,
        )
    rows = pd.read_csv(data_dir / "reproduced_test_predictions.csv")
    uncertainty = probe_uncertainty(rows, point_metrics)
    medians = {}
    for method, group in rows.groupby("architecture"):
        if len(group) != 100 or set(group.value) != set(test.value):
            raise ValueError(f"Mismatched test values for {method}")
        errors = (group.prediction - group.value).abs()
        if not errors.eq(group.abs_error).all():
            raise ValueError(f"Incorrect stored absolute errors for {method}")
        medians[method] = float(errors.median())
        np.testing.assert_allclose(
            medians[method], point_metrics.loc[method, "median_abs_error"], atol=1e-12
        )
    certificate = json.loads((data_dir / "classifier_median_certificate.json").read_text())
    classifier = point_metrics.loc["circular_cls"]
    n = int(classifier.num_examples)
    zero = round(classifier.accuracy * n)
    total = round(classifier.mean_abs_error * n)
    at_least_two_upper = total - (n - zero)
    lower_rank, upper_rank = (n + 1) // 2, n // 2 + 1
    if not (
        0 <= at_least_two_upper <= n and zero < lower_rank and n - at_least_two_upper >= upper_rank
    ):
        raise ValueError("Classifier totals do not imply median absolute error of one")
    expected_certificate = {
        "architecture": "circular_cls",
        "num_examples": n,
        "num_zero_errors": zero,
        "sum_abs_error": total,
        "num_errors_at_least_two_upper_bound": at_least_two_upper,
        "lower_middle_rank": lower_rank,
        "upper_middle_rank": upper_rank,
        "median_abs_error": 1.0,
    }
    if any(certificate.get(key) != value for key, value in expected_certificate.items()):
        raise ValueError("Classifier median certificate differs from the frozen totals")
    np.testing.assert_allclose(classifier.median_abs_error, 1.0, atol=0, rtol=0)
    verify_intervals(uncertainty, data_dir / "probe_uncertainty.csv", ["architecture", "metric"])
    uncertainty.to_csv(output_dir / "probe_uncertainty.csv", index=False, float_format="%.12g")

    # These axes are fitted supervised outputs, with a prescribed sinusoidal target code.
    fig = plt.figure(figsize=(6.2, 3.2))
    ax = fig.add_subplot(projection="3d")
    pair = 3
    ax.scatter(
        output.ridge_prediction,
        output[f"circular_cos_{pair}"],
        output[f"circular_sin_{pair}"],
        c=output.value,
        cmap="viridis",
        s=3,
    )
    values = np.linspace(0, 999, 2000)
    omega = 0.01584893192461114
    ax.plot(values, np.cos(omega * values), np.sin(omega * values), color="gray", lw=0.7)
    ax.set(xlabel="Ridge prediction", ylabel="Predicted cosine", zlabel="Predicted sine")
    ax.set_title("Figure 3b: supervised numerical probe outputs")
    save(fig, output_dir, "fig03_supervised_probe_outputs")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), layout="constrained")
    for ax, metric, label in zip(
        axes.flat,
        ["r2", "accuracy", "mean_abs_error", "median_abs_error"],
        ["R²", "Exact accuracy", "Mean absolute error", "Median absolute error"],
    ):
        selected = (
            uncertainty.loc[uncertainty.metric.eq(metric)].set_index("architecture").loc[METHODS]
        )
        for i, (_, row) in enumerate(selected.iterrows()):
            ax.scatter(i, row.estimate, color=COLORS[i], zorder=3)
            if pd.notna(row.low):
                ax.errorbar(
                    i,
                    row.estimate,
                    yerr=[[row.estimate - row.low], [row.high - row.estimate]],
                    color=COLORS[i],
                    fmt="none",
                    capsize=3,
                )
        ax.set_xticks(range(4), ["Ridge", "PLS", "Circular\nreg.", "Circular\ncls."])
        ax.set_ylabel(label)
    fig.suptitle("Figure 3c: fixed-fit evaluation; 95% value bootstrap where rows are available")
    save(fig, output_dir, "fig03_probe_metrics")

    capacity = pd.read_csv(data_dir / "probe_capacity.csv")
    if len(capacity) != 73 or set(capacity.architecture) != set(METHODS):
        raise ValueError("Expected all 73 predefined capacity-sweep settings")
    if not capacity.num_examples.eq(100).all() or not capacity.partition.eq("test").all():
        raise ValueError("Capacity sweep must use the fixed 100-value test set")
    fig, ax = plt.subplots(figsize=(6.8, 3.1), layout="constrained")
    minima = []
    for method, label, color in zip(METHODS, NAMES, COLORS):
        part = capacity.loc[capacity.architecture.eq(method)].sort_values("effective_df")
        ax.plot(part.effective_df, part.mean_abs_error, ".-", color=color, label=label, ms=3)
        best = part.loc[part.mean_abs_error.idxmin()]
        minima.append(
            {
                "architecture": method,
                "settings": len(part),
                "minimum_test_mae": float(best.mean_abs_error),
                "control_name": best.control_name,
                "control_value": float(best.control_value),
            }
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Family-specific complexity (different definitions)",
        ylabel="Test mean absolute error",
    )
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Figure 3d: representative within-family tuning paths")
    save(fig, output_dir, "fig03_family_complexity")
    return {
        "points": len(output),
        "split_counts": output.partition.value_counts().to_dict(),
        "supervised_display_pair_index": pair,
        "probe_metrics": records(point_metrics.reset_index()),
        "median_errors_from_predictions": medians,
        "classifier_median_certificate": certificate,
        "conditional_uncertainty": records(uncertainty),
        "capacity_minima": minima,
        "complexity_comparison": "Four family-specific quantities; cross-family efficiency is undetermined.",
    }


def appendix_capacity_panel(data_dir: Path, output_dir: Path) -> dict:
    frame = pd.read_json(data_dir / "appendix_capacity_points.jsonl.gz", lines=True)
    baseline = pd.read_json(data_dir / "appendix_capacity_baselines.jsonl", lines=True)
    models = [
        "llama_3_2_1b",
        "llama_3_2_3b",
        "llama_3_1_8b",
        "olmo_3_7b_base",
        "olmo_2_13b",
        "gpt_oss_20b",
    ]
    labels = [
        "Llama 3.2 1B",
        "Llama 3.2 3B",
        "Llama 3.1 8B",
        "OLMo 3 7B",
        "OLMo 2 13B",
        "GPT-OSS 20B",
    ]
    if len(frame) != 4752 or set(frame.model_alias) != set(models):
        raise ValueError("Expected the complete six-model appendix capacity curves")
    if frame.relative_layer.nunique() != 11 or len(baseline) != 6:
        raise ValueError("Appendix capacity depth/baseline inventory drift")
    best = frame.loc[
        frame.groupby(
            ["model_alias", "architecture", "relative_layer"]
        ).mean_abs_error_mean.idxmin()
    ]
    if len(best) != 264:
        raise ValueError("Expected six models × eleven depths × four family optima")
    medians = best.groupby("architecture")[["effective_df", "mean_abs_error_mean"]].median()
    for method, complexity, error in [
        ("linear", 399.5, 19.783),
        ("pls", 16.0, 20.298),
        ("circular_cls", 10.0, 2.452),
        ("circular_reg", 1024.0, 1.738),
    ]:
        np.testing.assert_allclose(medians.loc[method], [complexity, error], atol=1e-10, rtol=0)
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.5), sharey=True, layout="constrained")
    for ax, model, title in zip(axes.flat, models, labels):
        selected = frame.loc[frame.model_alias.eq(model)]
        reference = baseline.loc[baseline.model_alias.eq(model), "mean_abs_error"].iloc[0]
        ax.axhline(reference, color="gray", ls="--", lw=0.7)
        for method, color, label in zip(METHODS, COLORS, NAMES):
            part = selected.loc[selected.architecture.eq(method)]
            for depth, curve in part.groupby("relative_layer"):
                curve = curve.sort_values("effective_df")
                ax.plot(
                    curve.effective_df,
                    curve.mean_abs_error_mean.clip(lower=0.001),
                    color=color,
                    lw=0.7,
                    alpha=0.25 + 0.7 * depth,
                    label=label if depth == 1 else None,
                )
        ax.set(xscale="log", yscale="log", title=title, ylim=(0.7, 700))
    axes.flat[0].legend(frameon=False, fontsize=6)
    fig.supxlabel("Family-specific complexity (maximum over five prompt families)")
    fig.supylabel("Test MAE (mean over five prompt families)")
    save(fig, output_dir, "fig_appendix_capacity_paths")
    return {
        "aggregated_points": len(frame),
        "model_count": 6,
        "relative_depth_count": 11,
        "underlying_prompt_families": 5,
        "within_family_optima": len(best),
        "median_optima": records(medians.reset_index()),
        "data_level": "Frozen five-family mean errors and maximum family-specific tuning quantities; underlying family rows absent.",
        "comparison_to_figure3d": "Separate fits, checkpoints, and prompt-family aggregation.",
    }


def transfer_panel(data_dir: Path, output_dir: Path) -> dict:
    directions = pd.read_csv(data_dir / "pls_directions.csv.gz")
    transfer = pd.read_csv(data_dir / "pls_transfer.csv.gz")
    if len(directions) != 660 or len(transfer) != 1650:
        raise ValueError("Expected the complete six-model/eleven-depth PLS panel")
    if transfer.duplicated(["model_alias", "relative_layer", "train_family", "test_family"]).any():
        raise ValueError("Duplicated transfer cells")
    if transfer.model_alias.nunique() != 6 or transfer.relative_layer.nunique() != 11:
        raise ValueError("Transfer model/depth inventory drift")
    cosine = np.eye(5)
    for (left, right), value in (
        directions.groupby(["left_family", "right_family"]).direction_cosine.mean().items()
    ):
        i, j = FAMILIES.index(left), FAMILIES.index(right)
        cosine[i, j] = cosine[j, i] = value
    errors = transfer.groupby(["train_family", "test_family"])[
        ["raw", "family_mean_aligned"]
    ].mean()
    frozen = pd.read_csv(data_dir / "pls_transfer_means.csv").set_index(
        ["train_family", "test_family"]
    )
    computed = errors.reindex(frozen.index)
    np.testing.assert_allclose(computed.raw, frozen.raw_affine_mae, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        computed.family_mean_aligned, frozen.family_mean_aligned_mae, atol=1e-8, rtol=0
    )
    for (left, right), row in frozen.iterrows():
        np.testing.assert_allclose(
            cosine[FAMILIES.index(left), FAMILIES.index(right)],
            row.direction_cosine,
            atol=1e-10,
            rtol=0,
        )
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.3), layout="constrained")
    matrices = [
        cosine,
        errors.raw.unstack().loc[FAMILIES, FAMILIES].to_numpy(),
        errors.family_mean_aligned.unstack().loc[FAMILIES, FAMILIES].to_numpy(),
    ]
    for ax, values, label, cmap, bounds in zip(
        axes,
        matrices,
        ["Direction cosine", "Frozen probe MAE", "Training-mean aligned MAE"],
        ["coolwarm", "viridis_r", "viridis_r"],
        [(-1, 1), (0, 100), (0, 100)],
    ):
        image = ax.imshow(values, cmap=cmap, vmin=bounds[0], vmax=bounds[1])
        ax.set_xticks(range(5), FAMILY_LABELS, rotation=50, ha="right")
        ax.set_yticks(range(5), FAMILY_LABELS)
        ax.set_title(label)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Figure 4b and appendix: arithmetic means over six models × eleven depths")
    save(fig, output_dir, "fig04_pls_transfer")
    native = transfer.train_family.eq(transfer.test_family)
    native_mean, cross_mean = (
        transfer.loc[native, "raw"].mean(),
        transfer.loc[~native, "raw"].mean(),
    )
    np.testing.assert_allclose([native_mean, cross_mean], [13.566121212121212, 58.13117424242424])
    return {
        "model_count": 6,
        "relative_depth_count": 11,
        "direction_pairs": len(directions),
        "transfer_cells": len(transfer),
        "native_mean_mae": float(native_mean),
        "cross_family_mean_mae": float(cross_mean),
        "transfer_to_native_ratio": float(cross_mean / native_mean),
        "aligned_cross_family_mean_mae": float(transfer.loc[~native, "family_mean_aligned"].mean()),
        "cosine_matrix": cosine.tolist(),
        "mean_transfer_cells": records(frozen.reset_index()),
        "uncertainty": "Descriptive means across registered model/depth cells; no population intervals.",
    }


def deletion_seed(task: str, purpose: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"20260915::{task}::{purpose}".encode()).digest()[:4], "little"
    )


def deletion_panel(data_dir: Path, output_dir: Path) -> dict:
    units = pd.read_csv(data_dir / "deletion_units.csv.gz")
    summary = pd.read_csv(data_dir / "deletion_summary.csv")
    per_split = pd.read_csv(data_dir / "deletion_per_split.csv")
    contrasts = pd.read_csv(data_dir / "deletion_control_contrasts.csv")
    keys = ["split", "task_id", "source_value", "family"]
    if len(units) != 56000 or units.duplicated(keys).any():
        raise ValueError("Expected the complete reduced deletion grid")
    if units.source_value.nunique() != 639 or units.split.nunique() != 10:
        raise ValueError("Deletion source-value/split inventory drift")
    units["accuracy_change"] = units.intervention_correct - units.baseline_correct
    computed = (
        units.groupby(["split", "task_id", "family"])[
            ["baseline_correct", "intervention_correct", "accuracy_change"]
        ]
        .mean()
        .reset_index()
    )
    merged = per_split.merge(
        computed,
        on=["split", "task_id", "family"],
        validate="one_to_one",
        suffixes=("", "_computed"),
    )
    for old, new in [
        ("baseline_accuracy", "baseline_correct"),
        ("accuracy", "intervention_correct"),
        ("accuracy_change", "accuracy_change_computed"),
    ]:
        np.testing.assert_allclose(merged[old], merged[new], atol=1e-12, rtol=0)
    maximum_error = 0.0
    # Cluster multiplicities retain every split appearance of each numerical value.
    for task, task_units in units.groupby("task_id", sort=True):
        expected = summary.loc[summary.task_id.eq(task)].set_index("family")
        families = list(expected.index)
        wide = task_units.pivot(
            index=["split", "source_value"], columns="family", values="intervention_correct"
        ).reindex(columns=families)
        base = task_units.groupby(["split", "source_value"]).baseline_correct.first()
        if len(wide) != 1000 or wide.isna().any().any():
            raise ValueError(f"Incomplete deletion prompt grid for {task}")
        by_value = wide.groupby("source_value", sort=True).sum()
        counts = wide.groupby("source_value", sort=True).size().to_numpy(float)
        n = len(counts)
        rng = np.random.default_rng(deletion_seed(task, "bootstrap"))
        weights = rng.multinomial(n, np.full(n, 1 / n), size=2000)
        denominator = weights @ counts
        samples = weights @ by_value.to_numpy(float) / denominator[:, None]
        baseline_sums = base.groupby("source_value", sort=True).sum().to_numpy(float)
        baseline_samples = weights @ baseline_sums / denominator
        differences = samples - baseline_samples[:, None]
        point = wide.mean().to_numpy()
        for values, low_column, high_column in [
            (samples, "accuracy_low", "accuracy_high"),
            (differences, "accuracy_change_low", "accuracy_change_high"),
        ]:
            low, high = np.quantile(values, [0.025, 0.975], axis=0)
            for actual, column in [(low, low_column), (high, high_column)]:
                drift = np.max(np.abs(actual - expected[column].to_numpy()))
                maximum_error = max(maximum_error, float(drift))
                np.testing.assert_allclose(actual, expected[column], atol=1e-12, rtol=0)
        np.testing.assert_allclose(point, expected.accuracy, atol=1e-12, rtol=0)
        control_rows = contrasts.loc[contrasts.task_id.eq(task)].set_index("control")
        controls = list(control_rows.index)
        paired = wide["delete_property"].to_numpy()[:, None] - wide[controls].to_numpy()
        paired_frame = pd.DataFrame(paired, index=wide.index, columns=controls)
        paired_sums = paired_frame.groupby("source_value", sort=True).sum().to_numpy()
        paired_weights = np.random.default_rng(deletion_seed(task, "contrast")).multinomial(
            n, np.full(n, 1 / n), size=2000
        )
        paired_samples = paired_weights @ paired_sums / (paired_weights @ counts)[:, None]
        low, high = np.quantile(paired_samples, [0.025, 0.975], axis=0)
        for values, column in [(paired.mean(0), "mean"), (low, "low"), (high, "high")]:
            np.testing.assert_allclose(values, control_rows[column], atol=1e-12, rtol=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    for condition, label, color, offset in [
        ("delete_property", "Delete property subspace", "#35618D", -0.12),
        ("delete_shuffled", "Delete shuffled-label subspace", "#73706A", 0.12),
    ]:
        part = summary.loc[summary.family.eq(condition)].set_index("task_id").loc[TASKS]
        y = np.arange(len(part)) + offset
        ax.errorbar(
            100 * part.accuracy_change,
            y,
            xerr=[
                100 * (part.accuracy_change - part.accuracy_change_low),
                100 * (part.accuracy_change_high - part.accuracy_change),
            ],
            fmt="o",
            color=color,
            ms=4,
            capsize=2,
            label=label,
        )
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_yticks(range(7), TASK_LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change (percentage points)")
    ax.set_title("Figure 5c: source-value cluster intervals conditional on ten fitted bases")
    ax.legend(frameon=False, fontsize=7)
    save(fig, output_dir, "fig05_subspace_deletion")
    return {
        "reduced_source_condition_units": len(units),
        "unique_source_values": 639,
        "fitted_bases": 10,
        "prompts_per_task": 1000,
        "bootstrap_replicates": 2000,
        "summary_cells_verified": len(summary),
        "paired_contrasts_verified": len(contrasts),
        "maximum_interval_recomputation_error": maximum_error,
        "main_panel": records(
            summary.loc[summary.family.isin(["delete_property", "delete_shuffled"])]
        ),
        "conditioning": "Ten frozen fitted bases; overlapping split appearances cluster by source_value.",
    }


def property_panel(data_dir: Path, output_dir: Path) -> dict:
    counts = pd.read_csv(data_dir / "parity_contradiction_counts.csv")
    properties = pd.read_csv(data_dir / "property_accessibility.csv")
    if len(counts) != 8 or len(properties) != 19:
        raise ValueError("Property panel inventory drift")
    np.testing.assert_allclose(counts.rate, counts.contradictions / counts.n, atol=1e-12)
    if set(counts.loc[counts.experiment.eq("classification"), "n"]) != {100}:
        raise ValueError("Classification trial-count drift")
    if set(counts.loc[counts.experiment.eq("enumeration"), "n"]) != {50}:
        raise ValueError("Enumeration trial-count drift")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.7), layout="constrained")
    ax = axes[0]
    ax.errorbar(
        properties.full_activation_mean,
        np.arange(19),
        xerr=[
            properties.full_activation_mean - properties.full_activation_min,
            properties.full_activation_max - properties.full_activation_mean,
        ],
        fmt="o",
        ms=3,
        color="#35618D",
    )
    ax.scatter(properties.chance, np.arange(19), marker="|", color="gray")
    ax.set_yticks(range(19), properties.display_name)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.02), xlabel="Balanced accuracy", title="Figure 6a: mean and split range")
    for ax, experiment in zip(axes[1:], ["classification", "enumeration"]):
        part = counts.loc[counts.experiment.eq(experiment)]
        ax.barh(np.arange(len(part)), part.rate, color="#3F7F64")
        for i, (_, row) in enumerate(part.iterrows()):
            ax.text(
                min(row.rate + 0.02, 0.92),
                i,
                f"{row.contradictions}/{row.n}",
                va="center",
                fontsize=7,
            )
        ax.set_yticks(range(len(part)), part.label)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1.05), xlabel="Contradiction frequency", title=f"Figure 6: {experiment}")
    save(fig, output_dir, "fig06_property_counts")
    return {
        "categorical_properties": len(properties),
        "contradiction_counts": records(counts),
        "uncertainty": "Descriptive counts on fixed baseline-correct sets; no population intervals.",
    }


def control_panels(data_dir: Path, output_dir: Path) -> dict:
    rows = pd.read_csv(data_dir / "steering_accuracy_rows.csv.gz")
    uncertainty = steering_uncertainty(rows)
    verify_intervals(uncertainty, data_dir / "steering_uncertainty.csv", ["method", "metric"])
    uncertainty.to_csv(output_dir / "steering_uncertainty.csv", index=False, float_format="%.12g")
    fig, ax = plt.subplots(figsize=(6.3, 2.7), layout="constrained")
    keys = ["linear", "circular_reg", "centroid_patch"]
    for metric, marker, offset, label in [
        ("exact", "o", -0.1, "Exact"),
        ("within10", "s", 0.1, "Within ±10"),
    ]:
        part = uncertainty.loc[uncertainty.metric.eq(metric)].set_index("method").loc[keys]
        for i, (_, row) in enumerate(part.iterrows()):
            ax.scatter(
                100 * row.estimate,
                i + offset,
                marker=marker,
                color="#35618D" if metric == "exact" else "#3F7F64",
                label=label if i == 0 else None,
            )
            if pd.notna(row.low):
                ax.errorbar(
                    100 * row.estimate,
                    i + offset,
                    xerr=[[100 * (row.estimate - row.low)], [100 * (row.high - row.estimate)]],
                    fmt="none",
                    color="#35618D" if metric == "exact" else "#3F7F64",
                    capsize=2,
                )
            if row["count"] == 0:
                ax.text(2, i + offset, "0/2000", va="center", fontsize=7)
    ax.axvline(15.8, color="gray", ls=":", label="No-edit within ±10")
    ax.set_yticks(range(3), ["Scalar ridge edit", "Circular edit", "Centroid patch"])
    ax.invert_yaxis()
    ax.set(
        xlim=(-2, 102),
        xlabel="Accuracy (%)",
        title="Figure 7c: source-cluster uncertainty for fixed edits",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=3)
    save(fig, output_dir, "fig07_control_accuracy")

    coefficient_rows = pd.read_csv(data_dir / "coefficient_prediction_rows.csv.gz")
    conditions = ["independent_property_all", "ridge", "rbf"]
    estimates, reference = [], None
    matching = ["panel_row_idx", "baseline_row_idx", "source_value", "target_value"]
    for method in conditions:
        part = coefficient_rows.loc[coefficient_rows.condition.eq(method)]
        if len(part) != 400 or part.baseline_row_idx.nunique() != 20:
            raise ValueError(
                "Coefficient evaluation must use 400 edits from twenty source contexts"
            )
        if reference is None:
            reference = part[matching].reset_index(drop=True)
        else:
            pd.testing.assert_frame_equal(reference, part[matching].reset_index(drop=True))
        means = part.groupby("baseline_row_idx", sort=True).correct.mean().to_numpy()
        boots = (
            np.random.default_rng(20260826).choice(means, size=(10000, 20), replace=True).mean(1)
        )
        low, high = np.quantile(boots, [0.025, 0.975])
        estimates.append(
            {
                "method": method,
                "n": len(part),
                "clusters": 20,
                "accuracy": float(part.correct.mean()),
                "low": float(low),
                "high": float(high),
            }
        )
    np.testing.assert_allclose([r["accuracy"] for r in estimates], [0, 0.7675, 0.795])
    search = (
        pd.read_csv(data_dir / "dimensionality_control.csv")
        .set_index("method")
        .loc["in_span_search"]
    )
    if int(search.n) != 2000:
        raise ValueError("Output-scored search uses a separate 2,000-edit evaluation")
    np.testing.assert_allclose(search.exact_accuracy, 0.9945)
    fig, ax = plt.subplots(figsize=(6.3, 2.9), layout="constrained")
    for i, row in enumerate(estimates):
        if row["accuracy"] == 0:
            ax.scatter(0, i, color="#35618D")
            ax.text(2, i, f"0/{row['n']}", va="center", fontsize=7)
        else:
            ax.errorbar(
                100 * row["accuracy"],
                i,
                xerr=[
                    [100 * (row["accuracy"] - row["low"])],
                    [100 * (row["high"] - row["accuracy"])],
                ],
                fmt="o",
                color="#35618D",
                capsize=2,
            )
    ax.scatter(100 * search.exact_accuracy, 4, marker="D", color="#B9602B")
    ax.set_yticks(
        [0, 1, 2, 4],
        [
            "Independent property sum",
            "Joint ridge",
            "RBF ridge",
            "Output-scored search\n(separate 2,000 edits)",
        ],
    )
    ax.invert_yaxis()
    ax.set(
        xlim=(-2, 102),
        xlabel="Exact accuracy (%)",
        title="Figure 7f: coefficient prediction with 20 source clusters",
    )
    save(fig, output_dir, "fig07_coefficient_predictions")

    output = pd.read_csv(data_dir / "probe_outputs.csv.gz")
    test = output.loc[output.partition.eq("test")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    axes[0].plot([0, 999], [0, 999], color="gray", lw=0.7)
    axes[0].scatter(
        test.value,
        test.ridge_prediction,
        s=12,
        facecolors="none",
        edgecolors="#AC7D00",
        label="Scalar ridge",
    )
    axes[0].scatter(
        test.value,
        test.circular_prediction,
        s=10,
        marker="+",
        color="#3F7F64",
        label="Circular regression",
    )
    axes[0].set(
        xlabel="Target value", ylabel="Predicted value", title="Figure 7a: 100 held-out values"
    )
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(output.target_sin_4, output.target_sin_3, color="gray", lw=0.7)
    axes[1].scatter(test.circular_sin_4, test.circular_sin_3, s=9, c=test.value, cmap="viridis")
    axes[1].set(
        xlabel="Sine component 1",
        ylabel="Sine component 3",
        title="Figure 7b: lower-frequency code outputs",
        xlim=(-1.22, 1.22),
        ylim=(-1.22, 1.22),
        aspect="equal",
    )
    save(fig, output_dir, "fig07_numerical_predictions")
    return {
        "steering_accuracy": records(uncertainty),
        "coefficient_prediction": estimates,
        "coefficient_comparison": "Joint methods add source activations; evaluation reuses targets and doses differ.",
        "output_scored_search": {
            "n": int(search.n),
            "exact_accuracy": float(search.exact_accuracy),
            "comparison": "Separate evaluation with requested output used to score candidates.",
        },
    }


def main() -> None:
    args = parse_args()
    require_files(
        [
            args.data_dir / name
            for name in [
                "input_provenance.json",
                "reported_config.json",
                "probe_uncertainty.csv",
                "steering_uncertainty.csv",
            ]
        ]
    )
    verify_inputs(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 180,
        }
    )
    metrics = {
        "coverage": "Current empirical numerical panels; layout simplified for CPU verification.",
        "figure_3": probe_panels(args.data_dir, args.output_dir),
        "appendix_capacity": appendix_capacity_panel(args.data_dir, args.output_dir),
        "figure_4": transfer_panel(args.data_dir, args.output_dir),
        "figure_5": deletion_panel(args.data_dir, args.output_dir),
        "figure_6": property_panel(args.data_dir, args.output_dir),
        "figure_7": control_panels(args.data_dir, args.output_dir),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Verified current Figures 3–7 numerical panels; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
