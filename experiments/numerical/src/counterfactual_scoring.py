"""Transient baseline probabilities for scoring the same answer before/after.

The bank stays outside CSV columns. Reordered/subsampled frames look up explicit
example IDs; missing banks must produce unavailable gains, never a difference
between probabilities of two different answers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PROBABILITY_BANK_ATTR = "counterfactual_probability_bank"


@dataclass(frozen=True, eq=False)
class CounterfactualProbabilityBank:
    example_ids: np.ndarray
    token_ids: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        examples = np.array(self.example_ids, dtype=np.int64, copy=True)
        tokens = np.array(self.token_ids, dtype=np.int64, copy=True)
        probs = np.array(self.probabilities, copy=True)
        if examples.ndim != 1 or tokens.ndim != 1 or probs.shape != (len(examples), len(tokens)):
            raise ValueError("Probability bank dimensions do not match its IDs.")
        if len(np.unique(examples)) != len(examples) or len(np.unique(tokens)) != len(tokens):
            raise ValueError("Probability bank IDs must be unique.")
        if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
            raise ValueError("Invalid baseline probabilities.")
        for name, value in (("example_ids", examples), ("token_ids", tokens), ("probabilities", probs)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_example_rows", {int(v): i for i, v in enumerate(examples)})
        object.__setattr__(self, "_token_columns", {int(v): i for i, v in enumerate(tokens)})

    def __deepcopy__(self, memo):
        # Pandas propagates/deep-copies attrs when slicing. This bank is immutable.
        return self

    def lookup(self, example_ids: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        if len(example_ids) != len(token_ids):
            raise ValueError("Baseline example and target ID lengths differ.")
        try:
            rows = np.asarray([self._example_rows[int(v)] for v in example_ids], dtype=np.int64)
            cols = np.asarray([self._token_columns[int(v)] for v in token_ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Missing baseline example/target ID: {error.args[0]}") from error
        return self.probabilities[rows, cols].astype(np.float64)


def attach_counterfactual_baseline(
    frame: pd.DataFrame, bank: CounterfactualProbabilityBank | None
) -> pd.DataFrame:
    result = frame.copy()
    if "baseline_original_target_prob" not in result:
        result["baseline_original_target_prob"] = result["baseline_target_prob"]
    if bank is None:
        result["baseline_counterfactual_prob"] = np.nan
    else:
        result["baseline_counterfactual_prob"] = bank.lookup(
            result["example_id"].to_numpy(), result["target_token_id"].to_numpy()
        )
        result["baseline_target_prob"] = result["baseline_counterfactual_prob"]
    return result
