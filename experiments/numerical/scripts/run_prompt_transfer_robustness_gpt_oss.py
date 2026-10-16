#!/usr/bin/env python3
"""Run the prompt-transfer suite for GPT-OSS with an explicit histc exception.

Transformers' grouped-MoE implementation computes integer token counts with
``torch.histc`` on CUDA.  PyTorch does not mark that operation deterministic,
so the shared strict runner stops before the first forward pass.  This wrapper
retains deterministic-algorithm checks in warning mode for GPT-OSS and adds
the exception, its reason, and this wrapper's hash to the run identity.  It is
therefore impossible to confuse these artifacts with a strict-determinism run.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_prompt_transfer_robustness_suite as suite


MODEL_ALIAS = "gpt_oss_20b"
DETERMINISM_EXCEPTION = {
    "requested": True,
    "warn_only": True,
    "operation": "torch.histc on CUDA",
    "call_site": "transformers.integrations.moe.grouped_mm_experts_forward",
    "reason": (
        "GPT-OSS grouped-MoE routing counts tokens per expert with CUDA histc, "
        "which PyTorch 2.7.1 does not provide under strict deterministic mode."
    ),
}


def _run_identity_with_exception(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    model_alias: str,
    dtype_override: str | None,
    smoke_values_per_partition: int | None,
) -> tuple[str, dict[str, Any]]:
    if model_alias != MODEL_ALIAS:
        raise ValueError(
            f"This launcher is restricted to {MODEL_ALIAS}, received {model_alias}."
        )
    _, payload = _ORIGINAL_RUN_IDENTITY(
        cfg=cfg,
        config_path=config_path,
        model_alias=model_alias,
        dtype_override=dtype_override,
        smoke_values_per_partition=smoke_values_per_partition,
    )
    payload["determinism_policy"] = {
        **DETERMINISM_EXCEPTION,
        "wrapper_sha256": suite.sha256_file(Path(__file__).resolve()),
    }
    return f"{model_alias}__{suite.stable_hash(payload)}", payload


def _deterministic_algorithms_warn_only(mode: bool, *, warn_only: bool = False) -> None:
    _ORIGINAL_USE_DETERMINISTIC_ALGORITHMS(mode, warn_only=True)


_ORIGINAL_RUN_IDENTITY = suite._run_identity
_ORIGINAL_USE_DETERMINISTIC_ALGORITHMS = torch.use_deterministic_algorithms


def main() -> None:
    suite._run_identity = _run_identity_with_exception
    suite.torch.use_deterministic_algorithms = _deterministic_algorithms_warn_only
    suite.main()


if __name__ == "__main__":
    main()
