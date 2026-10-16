"""Export matched fitted ridge and circular-regression outputs for Figure 3 views."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_figure3_number_helix import (
    DEFAULT_REPR,
    DEFAULT_SPLIT,
    load_centroids,
    relative,
    sha256,
)
from scripts.measure_probe_manifold_effective_dimensionality import PROBE_SPECS
from src.magnitude_intervention import evaluate_probe_split, fit_magnitude_probe

SUMMARY = ROOT / "experiment-reports/artifacts/2026-06-15-probe-manifold-effective-dimensionality-llama31-8b-layer10/probe_eval_summary.csv"
OUTPUT = ROOT / "experiment-reports/artifacts/2026-09-07-figure3-probe-outputs"


def main() -> None:
    print("Loading the matched Llama-3.1-8B layer-10 centroids.", flush=True)
    centroids, source = load_centroids(DEFAULT_REPR)
    assert source["cache_sha256"] == "42cf4af31df100654d0efebc27e0a82d375464fd1833e54b121e85cc9418092a"
    assert sha256(DEFAULT_SPLIT) == "e0ab1eaf6db11396291155ad12f61b49b9260da63c6302709836159132930d20"
    assert sha256(SUMMARY) == "17ad892685478387eb9ddd546856868022cf7c8eb73b9fff6e05a21ec5af14fc"
    payload = json.loads(DEFAULT_SPLIT.read_text())
    splits = {name: np.asarray(payload[name + "_source_values"], dtype=int)
              for name in ("train", "dev", "test")}
    original = pd.read_csv(SUMMARY)
    frame = pd.DataFrame({"value": np.arange(1000), "partition": ""})
    for name, values in splits.items():
        frame.loc[values, "partition"] = name
    metrics, probes = [], {}
    for architecture in ("linear", "circular_reg"):
        print(f"Reproducing {architecture} with the archived configuration.", flush=True)
        probe = fit_magnitude_probe(
            architecture=architecture, train_x=centroids[splits["train"]],
            train_values=splits["train"], config=PROBE_SPECS[architecture],
        )
        probes[architecture] = probe
        for partition, values in splits.items():
            _, summary = evaluate_probe_split(
                probe=probe, representations=centroids[values], values=values, split_name=partition,
            )
            expected = original.loc[(original.architecture == architecture)
                                    & (original.split == partition)
                                    & (original.fit_mode == "all_x")].iloc[0]
            for name in ("num_examples", "mean_abs_error", "rmse", "accuracy"):
                np.testing.assert_allclose(summary[name], expected[name], rtol=0, atol=1e-12)
            metrics.append(summary)
            if partition == "test":
                print(f"{architecture}: archived test MAE {summary['mean_abs_error']} matched.", flush=True)
    frame["ridge_prediction"] = probes["linear"].predict_features(centroids).ravel()
    circular = probes["circular_reg"]
    predicted = circular.predict_features(centroids)
    target = circular.support_features
    frame["circular_prediction"] = circular.predict_values(centroids)
    for pair in range(5):
        for offset, label in enumerate(("sin", "cos")):
            frame[f"circular_{label}_{pair}"] = predicted[:, 2 * pair + offset]
            frame[f"target_{label}_{pair}"] = target[:, 2 * pair + offset]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points_path = OUTPUT / "probe_outputs.csv"
    metrics_path = OUTPUT / "verified_metrics.csv"
    frame.to_csv(points_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    frequencies = np.exp(np.arange(0, 10, 2) * -np.log(1000) / 10)
    provenance = {
        "source": source, "split": {"path": relative(DEFAULT_SPLIT), "sha256": sha256(DEFAULT_SPLIT)},
        "probe_configs": {name: PROBE_SPECS[name] for name in probes},
        "original_summary": {"path": relative(SUMMARY), "sha256": sha256(SUMMARY)},
        "verification": {"archived_metrics_matched": 24, "absolute_tolerance": 1e-12},
        "circular_pairs": [
            {"pair_index": i, "angular_frequency": float(omega), "period": float(2 * np.pi / omega),
             "target": "sin(omega * value), cos(omega * value)"}
            for i, omega in enumerate(frequencies)
        ],
        "generation": {
            "script": relative(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
            "command": ("OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
                        "uv run --frozen --no-sync python scripts/export_figure3_probe_outputs.py"),
            "numpy_version": np.__version__, "pandas_version": pd.__version__, "device": "cpu",
            "dependencies": {relative(ROOT / name): sha256(ROOT / name) for name in (
                "scripts/analyze_figure3_number_helix.py",
                "scripts/measure_probe_manifold_effective_dimensionality.py",
                "src/magnitude_intervention.py", "src/ones_digit_intervention.py",
                "src/activation_steering.py",
            )},
        },
        "outputs": {relative(path): sha256(path) for path in (points_path, metrics_path)},
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {points_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
