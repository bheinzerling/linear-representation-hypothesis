from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def round_predictions(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(int)


def exact_accuracy(predictions: Iterable[int], targets: Iterable[int]) -> float:
    preds = np.asarray(list(predictions))
    gold = np.asarray(list(targets))
    if preds.size == 0:
        return math.nan
    return float((preds == gold).mean())


def mae(predictions: Iterable[float], targets: Iterable[float]) -> float:
    preds = np.asarray(list(predictions), dtype=float)
    gold = np.asarray(list(targets), dtype=float)
    return float(np.abs(preds - gold).mean())


def rmse(predictions: Iterable[float], targets: Iterable[float]) -> float:
    preds = np.asarray(list(predictions), dtype=float)
    gold = np.asarray(list(targets), dtype=float)
    return float(np.sqrt(np.mean((preds - gold) ** 2)))
