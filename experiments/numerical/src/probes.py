from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_binary_codebook(num_labels: int = 1000, width: int | None = None) -> np.ndarray:
    width = width or int(np.ceil(np.log2(num_labels)))
    values = np.arange(num_labels, dtype=np.int64)[:, None]
    bits = ((values >> np.arange(width, dtype=np.int64)) & 1).astype(np.float32)
    return bits


def build_sinusoidal_codebook(
    num_labels: int = 1000,
    width: int = 100,
    variant: Literal["variant_paper_literal", "variant_standard_transformer_style"] = "variant_paper_literal",
    denominator: float | None = None,
) -> np.ndarray:
    denominator = denominator or float(width)
    codebook = np.zeros((num_labels, width), dtype=np.float32)
    values = np.arange(num_labels, dtype=np.float32)
    if variant == "variant_paper_literal":
        for j in range(width):
            angle = values * np.exp(j / 1000.0) / denominator
            codebook[:, j] = np.sin(angle) if j % 2 == 0 else np.cos(angle)
        return codebook
    if variant == "variant_standard_transformer_style":
        for j in range(width):
            pair_idx = j // 2
            angle = values / (10000.0 ** (2.0 * pair_idx / width))
            codebook[:, j] = np.sin(angle) if j % 2 == 0 else np.cos(angle)
        return codebook
    raise ValueError(f"Unknown sinusoidal variant: {variant}")


@dataclass
class RegressionResult:
    predictions: np.ndarray
    weights: np.ndarray
    bias: float


def fit_linear_regression(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> RegressionResult:
    design = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)], axis=1)
    solution, *_ = np.linalg.lstsq(design, train_y.astype(np.float64), rcond=None)
    weights = solution[:-1].astype(np.float32)
    bias = float(solution[-1])
    preds = test_x @ weights + bias
    return RegressionResult(predictions=preds.astype(np.float32), weights=weights, bias=bias)


def fit_log_linear_regression(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> RegressionResult:
    transformed_y = np.log1p(train_y.astype(np.float64))
    result = fit_linear_regression(train_x, transformed_y, test_x)
    preds = np.expm1(result.predictions.astype(np.float64))
    return RegressionResult(predictions=preds.astype(np.float32), weights=result.weights, bias=result.bias)


class CodebookClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, codebook: np.ndarray):
        super().__init__()
        self.w_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(codebook.shape[1], hidden_dim, bias=False)
        self.register_buffer("codebook", torch.tensor(codebook, dtype=torch.float32))

    def projected_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_in(x)

    def projected_labels(self, label_indices: torch.Tensor | None = None) -> torch.Tensor:
        codes = self.codebook if label_indices is None else self.codebook[label_indices]
        return self.w_out(codes)

    def logits(self, x: torch.Tensor, label_indices: torch.Tensor | None = None) -> torch.Tensor:
        x_proj = self.projected_inputs(x)
        label_proj = self.projected_labels(label_indices=label_indices)
        return x_proj @ label_proj.T


@dataclass
class ClassifierTrainResult:
    predictions: np.ndarray
    best_state: dict[str, torch.Tensor]
    best_val_loss: float
    history: list[dict[str, float]]


def fit_classifier_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    codebook: np.ndarray,
    hidden_dim: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    max_epochs: int = 1000,
    patience: int = 20,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
    fixed_epochs: int | None = None,
) -> ClassifierTrainResult:
    if len(train_y) < 2:
        raise ValueError("Need at least two training labels for classifier probe.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_y))
    if fixed_epochs is None:
        val_size = max(1, int(round(len(train_y) * val_fraction)))
        val_idx = perm[:val_size]
        inner_idx = perm[val_size:]
        if inner_idx.size == 0:
            inner_idx = val_idx
            val_idx = perm[-1:]
    else:
        if int(fixed_epochs) <= 0:
            raise ValueError(f"fixed_epochs must be positive, got {fixed_epochs!r}")
        # A second-stage refit uses every supplied example in optimizer batches.
        # Reusing the full set for the logged loss is diagnostic only: the epoch
        # count must already have been selected by a separate fit.
        inner_idx = perm
        val_idx = perm

    inner_x = torch.tensor(train_x[inner_idx], dtype=torch.float32)
    inner_y = torch.tensor(train_y[inner_idx], dtype=torch.long)
    val_x = torch.tensor(train_x[val_idx], dtype=torch.float32)
    val_y = torch.tensor(train_y[val_idx], dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)

    train_labels = np.sort(np.unique(train_y))
    train_label_to_pos = {label: idx for idx, label in enumerate(train_labels.tolist())}
    inner_targets = torch.tensor([train_label_to_pos[int(y)] for y in inner_y.tolist()], dtype=torch.long)
    val_targets = torch.tensor([train_label_to_pos[int(y)] for y in val_y.tolist()], dtype=torch.long)
    train_label_tensor = torch.tensor(train_labels, dtype=torch.long, device=device)

    model = CodebookClassifier(input_dim=train_x.shape[1], hidden_dim=hidden_dim, codebook=codebook).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    loader = DataLoader(TensorDataset(inner_x, inner_targets), batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model.logits(batch_x, label_indices=train_label_tensor)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * batch_x.shape[0]
            train_count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            val_logits = model.logits(val_x.to(device), label_indices=train_label_tensor)
            val_loss = float(F.cross_entropy(val_logits, val_targets.to(device)).item())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_total / max(1, train_count),
                "val_loss": val_loss,
            }
        )
        if fixed_epochs is not None:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if fixed_epochs is None and epoch - best_epoch >= patience:
            break
        if fixed_epochs is not None and epoch + 1 >= int(fixed_epochs):
            break

    if best_state is None:
        raise RuntimeError("Classifier training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model.logits(test_x_tensor.to(device), label_indices=None)
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    return ClassifierTrainResult(
        predictions=predictions.astype(np.int64),
        best_state=best_state,
        best_val_loss=best_val_loss,
        history=history,
    )
