"""Prequential label coding for the four numerical probes in Figure 3.

The online sum follows Voita and Titov (2020), equation (3). Probability
calibration is learned from an already transmitted prefix, never its next block.
All predictive distributions have the same integer alphabet, including labels
absent from the fitting prefix. This module does not load a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cross_decomposition import PLSRegression

ALPHABET = np.arange(1000, dtype=np.int64)
FRACTIONS = (.001, .002, .004, .008, .016, .032, .0625, .125, .25, .5, 1.)
UNIFORM_BITS = float(np.log2(len(ALPHABET)))
UNIFORM_MIX = .01
TEMPERATURES = np.r_[np.logspace(-6, 6, 49), np.inf]
CODERS = ("native", "ordinal")


def block_boundaries(n: int) -> list[int]:
    """Official floor rule, merging empty/duplicate blocks on a small dataset."""
    if n < 3:
        raise ValueError("Online coding requires at least three examples")
    return sorted({max(1, int(n * f)) for f in FRACTIONS} | {n})


def log_probabilities(scores: np.ndarray, temperature: float, mix: float = UNIFORM_MIX) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(ALPHABET) or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores over all 1,000 labels")
    if not 0 <= mix < 1 or temperature <= 0 or np.isnan(temperature):
        raise ValueError("Invalid calibration parameters")
    if np.isinf(temperature):
        return np.full_like(scores, -np.log(len(ALPHABET)))
    # Subtract before scaling to avoid loss of precision at low temperatures.
    logits = (scores - scores.max(axis=1, keepdims=True)) / temperature
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    if mix:
        logs = np.logaddexp(np.log1p(-mix) + logs, np.log(mix / len(ALPHABET)))
    return logs


def label_bits(scores: np.ndarray, targets: np.ndarray, temperature: float) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Integer targets required")
    if len(targets) != len(scores) or np.any((targets < 0) | (targets >= len(ALPHABET))):
        raise ValueError("Targets outside the common alphabet")
    return -log_probabilities(scores, temperature)[np.arange(len(targets)), targets] / np.log(2.)


def calibrate_temperature(scores: np.ndarray, targets: np.ndarray) -> float:
    """Select from a fixed shared grid using only labels in a past prefix."""
    if not len(targets):
        raise ValueError("Calibration labels are required")
    costs = [label_bits(scores, targets, float(t)).sum() for t in TEMPERATURES]
    return float(TEMPERATURES[int(np.argmin(costs))])


def calibration_partition(prefix_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 80/20 split; order was randomized before block formation."""
    if prefix_size < 3:
        raise ValueError("A fit needs two examples plus a calibration example")
    n_cal = max(1, int(round(.2 * prefix_size)))
    return np.arange(prefix_size - n_cal), np.arange(prefix_size - n_cal, prefix_size)


@dataclass
class FittedProbe:
    architecture: str
    coefficients: np.ndarray
    bias: np.ndarray
    support_features: np.ndarray
    metadata: dict[str, Any]

    def outputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(x, dtype=np.float64) @ self.coefficients + self.bias
        if self.architecture in {"linear", "pls"}:
            means = features.reshape(-1)
            predictions = np.clip(np.rint(means), 0, 999).astype(int)
            native = -(means[:, None] - ALPHABET[None, :]) ** 2
        else:
            if self.architecture == "circular_reg":
                normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-300)
                support = self.support_features / np.maximum(
                    np.linalg.norm(self.support_features, axis=1, keepdims=True), 1e-300)
                native = normalized @ support.T
            elif self.architecture == "circular_cls":
                native = features @ self.support_features.T
            else:
                raise ValueError(self.architecture)
            predictions = np.argmax(native, axis=1)
        ordinal = -np.abs(predictions[:, None] - ALPHABET[None, :]).astype(float)
        return native, ordinal, predictions


class PrefixFitter:
    """Reuse label-independent factorizations across a sweep on one prefix.

    The circular least-squares fit keeps the original augmented design's
    minimum-norm intercept. Ridge instead centers x and y, as in the source.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, *, device: str, seed: int):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)
        self.device = device
        self.seed = seed
        if len(self.y) < 2:
            raise ValueError("At least two fitting examples are required")

    @cached_property
    def design_inverse(self) -> np.ndarray:
        design = np.column_stack([self.x, np.ones(len(self.x))])
        return np.linalg.pinv(design, rcond=np.finfo(float).eps * max(design.shape))

    @cached_property
    def centered_svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.linalg.svd(self.x - self.x.mean(0), full_matrices=False)

    def fit(self, architecture: str, config: dict[str, Any]) -> FittedProbe:
        from src.activation_steering import circular_sinusoidal_encode

        if architecture == "linear":
            lam = float(config["ridge_lambda"])
            support = ALPHABET[:, None].astype(float)
            if lam == 0:
                solution = self.design_inverse @ self.y[:, None]
                coefficients, bias = solution[:-1], solution[-1]
            else:
                u, singular, vt = self.centered_svd
                coefficients = ((vt.T * (singular / (singular**2 + lam))) @
                                (u.T @ (self.y - self.y.mean())))[:, None]
                bias = np.array([self.y.mean()]) - self.x.mean(0) @ coefficients
            metadata = {"ridge_lambda": lam}
        elif architecture == "pls":
            components = min(int(config["n_components"]), len(self.y) - 1, self.x.shape[1])
            model = PLSRegression(n_components=components, scale=False).fit(self.x, self.y[:, None])
            # sklearn's coefficients act on centered x for scale=False.
            coefficients = model.coef_.T.copy()
            bias = model.predict(np.zeros((1, self.x.shape[1])))[0]
            support = ALPHABET[:, None].astype(float)
            metadata = {"requested_components": int(config["n_components"]), "fitted_components": components}
        elif architecture == "circular_reg":
            support = circular_sinusoidal_encode(ALPHABET, embedding_dim=int(config["basis_dim"]),
                                                  min_value=0, max_value=1000)
            solution = self.design_inverse @ support[self.y]
            coefficients, bias = solution[:-1], solution[-1]
            metadata = {"basis_dim": int(config["basis_dim"])}
        elif architecture == "circular_cls":
            import torch
            from src.probes import fit_classifier_probe

            torch.manual_seed(self.seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(self.seed)
            codebook = circular_sinusoidal_encode(ALPHABET, embedding_dim=int(config["basis_dim"]),
                                                   min_value=0, max_value=1000).astype(np.float32)
            result = fit_classifier_probe(
                train_x=self.x.astype(np.float32), train_y=self.y,
                test_x=self.x[:1].astype(np.float32), codebook=codebook,
                **{k: v for k, v in config.items() if k != "basis_dim"}, device=self.device)
            w_in = result.best_state["w_in.weight"].numpy().astype(float)
            w_out = result.best_state["w_out.weight"].numpy().astype(float)
            coefficients, bias = w_in.T, np.zeros(w_in.shape[0])
            support = codebook.astype(float) @ w_out.T
            metadata = {"epochs_trained": len(result.history), "hidden_dim": w_in.shape[0],
                        "basis_dim": int(config["basis_dim"])}
        else:
            raise ValueError(architecture)
        return FittedProbe(architecture, coefficients, np.asarray(bias), support, metadata)
