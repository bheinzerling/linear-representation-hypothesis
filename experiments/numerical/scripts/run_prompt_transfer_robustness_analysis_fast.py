#!/usr/bin/env python3
"""Analyze prompt-transfer probes without constructing unused intervention decoders.

The shared magnitude-probe fitter computes ``pinv(coefficients)`` so fitted
probes can later be used for activation interventions.  This experiment uses
only ``predict_values``, ``predict_features``, and scalar coefficient
directions.  The decoder matrix cannot affect any recorded output, while a
full-rank circular probe otherwise requires an expensive square pseudoinverse.

This launcher substitutes an empty decoder matrix during fitting and records
the optimization and launcher hash in ``analysis_execution.json``.  Collection
and scientific run identities remain unchanged because the fitted
coefficients, biases, support features, predictions, and metrics are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_prompt_transfer_robustness_gpt_oss as gpt_oss
from scripts import run_prompt_transfer_robustness_suite as suite
from src.utils import atomic_write_json, load_yaml


_ORIGINAL_FIT_MAGNITUDE_PROBE = suite.fit_magnitude_probe
_ORIGINAL_PINV = np.linalg.pinv


def _unused_decoder(*args: Any, **kwargs: Any) -> np.ndarray:
    return np.empty((0, 0), dtype=np.float64)


def fit_magnitude_probe_without_decoder(*args: Any, **kwargs: Any) -> Any:
    np.linalg.pinv = _unused_decoder
    try:
        return _ORIGINAL_FIT_MAGNITUDE_PROBE(*args, **kwargs)
    finally:
        np.linalg.pinv = _ORIGINAL_PINV


def _argument_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _output_directory() -> Path:
    model_alias = _argument_value("--model")
    if model_alias is None:
        raise ValueError("--model is required.")
    config_path = Path(_argument_value("--config") or suite.DEFAULT_CONFIG).resolve()
    cfg = load_yaml(config_path)
    dtype_override = _argument_value("--dtype")
    smoke = _argument_value("--smoke-values-per-partition")
    identity_fn = (
        gpt_oss._run_identity_with_exception
        if model_alias == gpt_oss.MODEL_ALIAS
        else suite._run_identity
    )
    run_id, _ = identity_fn(
        cfg=cfg,
        config_path=config_path,
        model_alias=model_alias,
        dtype_override=dtype_override,
        smoke_values_per_partition=None if smoke is None else int(smoke),
    )
    output_root = Path(_argument_value("--output-root") or cfg["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root / run_id


def main() -> None:
    model_alias = _argument_value("--model")
    if model_alias == gpt_oss.MODEL_ALIAS:
        suite._run_identity = gpt_oss._run_identity_with_exception
        suite.torch.use_deterministic_algorithms = gpt_oss._deterministic_algorithms_warn_only
    suite.fit_magnitude_probe = fit_magnitude_probe_without_decoder
    output_dir = _output_directory()
    layer_root = output_dir / "analysis" / "layers"
    preexisting_layers = sorted(
        int(path.parent.name.removeprefix("layer_"))
        for path in layer_root.glob("layer_*/DONE")
    )
    suite.main()
    completed_layers = sorted(
        int(path.parent.name.removeprefix("layer_"))
        for path in layer_root.glob("layer_*/DONE")
    )
    optimized_layers = sorted(set(completed_layers) - set(preexisting_layers))
    atomic_write_json(
        output_dir / "analysis_execution.json",
        {
            "policy": (
                "mixed_shared_and_prediction_only_fit"
                if preexisting_layers
                else "skip_unused_intervention_decoder"
            ),
            "launcher": str(Path(__file__).resolve().relative_to(ROOT)),
            "launcher_sha256": suite.sha256_file(Path(__file__).resolve()),
            "omitted_operation": "numpy.linalg.pinv(coefficients)",
            "recorded_outputs_depend_on": [
                "probe coefficients",
                "probe bias",
                "support features",
            ],
            "recorded_outputs_depend_on_decoder_matrix": False,
            "preexisting_shared_fitter_layer_positions": preexisting_layers,
            "prediction_only_layer_positions": optimized_layers,
        },
    )


if __name__ == "__main__":
    main()
