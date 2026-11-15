"""Linear probes over the balanced number-feature dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SPLITS = ("train", "dev", "test")
REGRESSION_TARGET_TYPES = frozenset({"regression"})


@dataclass(frozen=True)
class LayerSpec:
    name: str
    relative_layer: float
    absolute_layer: int
    hidden_state_index: int
    num_layers: int


@dataclass(frozen=True)
class ProbeConfig:
    model: str
    dataset_dir: str
    output_dir: str
    layer: str = "rel0p3"
    activation_batch_size: int = 128
    probe_batch_size: int = 512
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    seed: int = 20260714
    dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = True
    max_examples_per_split: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_relative_layer(name: str, num_layers: int) -> LayerSpec:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if not name.startswith("rel") or "p" not in name:
        raise ValueError(f"expected a relative layer such as rel0p3, got {name!r}")
    whole, fraction = name[3:].split("p", maxsplit=1)
    if not whole.isdigit() or not fraction.isdigit():
        raise ValueError(f"invalid relative layer {name!r}")
    relative = float(f"{int(whole)}.{fraction}")
    if not 0.0 <= relative < 1.0:
        raise ValueError("number-feature probes require a transformer-block relative layer in [0, 1)")
    absolute = min(num_layers - 1, max(0, round(relative * (num_layers - 1))))
    return LayerSpec(
        name=name,
        relative_layer=relative,
        absolute_layer=absolute,
        hidden_state_index=absolute + 1,
        num_layers=num_layers,
    )


def read_feature_rows(path: Path, max_examples_per_split: int = 0) -> dict[str, list[dict[str, Any]]]:
    rows = {split: [] for split in SPLITS}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["split"])
            if split not in rows:
                raise ValueError(f"unexpected split {split!r} in {path}")
            if max_examples_per_split and len(rows[split]) >= max_examples_per_split:
                continue
            rows[split].append(row)
    if any(not rows[split] for split in SPLITS):
        raise ValueError(f"all splits must be nonempty in {path}")
    return rows


def flatten_probe_inputs(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[int], list[list[int]], list[int]]:
    texts: list[str] = []
    positions: list[int] = []
    expected_token_ids: list[list[int]] = []
    inputs_per_row: list[int] = []
    for row in rows:
        row_inputs = list(row["inputs"])
        if not row_inputs:
            raise ValueError(f"row {row.get('example_id')} has no inputs")
        inputs_per_row.append(len(row_inputs))
        for input_row in row_inputs:
            locus = input_row["probe_locus"]
            token_indices = [int(value) for value in locus["token_indices"]]
            token_ids = [int(value) for value in locus["token_ids"]]
            if len(token_indices) != 1 or len(token_ids) != 1:
                raise ValueError(
                    f"simple probes require a one-token locus; {row.get('example_id')} has {token_indices}"
                )
            texts.append(str(input_row["text"]))
            positions.append(token_indices[0])
            expected_token_ids.append(token_ids)
    if len(set(inputs_per_row)) != 1:
        raise ValueError("all rows of one feature must have the same number of probe inputs")
    return texts, positions, expected_token_ids, inputs_per_row


@torch.inference_mode()
def extract_activations(
    *,
    model: nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    layer: LayerSpec,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    texts, positions, expected_token_ids, inputs_per_row = flatten_probe_inputs(rows)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        batch_texts = texts[start:stop]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        if "token_type_ids" in encoded:
            encoded.pop("token_type_ids")
        batch_positions = positions[start:stop]
        for offset, (position, expected) in enumerate(
            zip(batch_positions, expected_token_ids[start:stop], strict=True)
        ):
            observed = [int(encoded["input_ids"][offset, position].item())]
            if observed != expected:
                raise ValueError(
                    f"tokenization mismatch for row input {start + offset}: expected {expected}, observed {observed}"
                )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[layer.hidden_state_index]
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        position_tensor = torch.tensor(batch_positions, device=hidden.device)
        selected = hidden[batch_indices, position_tensor].float().cpu()
        chunks.append(selected)
    flat = torch.cat(chunks, dim=0)
    inputs_per_example = inputs_per_row[0]
    if inputs_per_example == 1:
        return flat
    return flat.reshape(len(rows), inputs_per_example * flat.shape[-1])


def fit_standardizer(train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, scale


def standardize(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() - mean) / scale


def natural_label_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def classification_targets(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    labels = sorted(
        {str(row["label_key"]) for row in rows_by_split["train"]},
        key=natural_label_sort_key,
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        unknown = sorted(
            {str(row["label_key"]) for row in rows_by_split[split]} - set(label_to_index),
            key=natural_label_sort_key,
        )
        if unknown:
            raise ValueError(f"{split} contains labels absent from train: {unknown[:10]}")
        targets[split] = torch.tensor(
            [label_to_index[str(row["label_key"])] for row in rows_by_split[split]],
            dtype=torch.long,
        )
    return targets, labels


def regression_targets(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
    return {
        split: torch.tensor([float(row["label"]) for row in rows_by_split[split]], dtype=torch.float32)
        for split in SPLITS
    }


def macro_classification_metrics(target: torch.Tensor, prediction: torch.Tensor, num_classes: int) -> dict[str, float]:
    target = target.long().cpu()
    prediction = prediction.long().cpu()
    accuracy = float((target == prediction).float().mean().item())
    recalls: list[float] = []
    f1s: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(((target == class_index) & (prediction == class_index)).sum().item())
        false_positive = int(((target != class_index) & (prediction == class_index)).sum().item())
        false_negative = int(((target == class_index) & (prediction != class_index)).sum().item())
        support = true_positive + false_negative
        if support == 0:
            continue
        recall = true_positive / support
        precision = true_positive / max(1, true_positive + false_positive)
        recalls.append(recall)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target_np = target.float().cpu().numpy().astype(np.float64)
    prediction_np = prediction.float().cpu().numpy().astype(np.float64)
    residual = prediction_np - target_np
    ss_residual = float(np.square(residual).sum())
    ss_total = float(np.square(target_np - target_np.mean()).sum())
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": 0.0 if ss_total == 0 else 1.0 - ss_residual / ss_total,
        "pearson_r": _correlation(target_np, prediction_np),
        "spearman_r": _correlation(_rankdata(target_np), _rankdata(prediction_np)),
    }


def _data_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=min(batch_size, len(x)),
        shuffle=True,
        generator=generator,
    )


@torch.inference_mode()
def _predict(probe: nn.Linear, x: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        outputs.append(probe(x[start : start + batch_size].to(device)).float().cpu())
    return torch.cat(outputs, dim=0)


def train_linear_probe(
    *,
    activations: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    target_type: str,
    config: ProbeConfig,
    device: torch.device,
    num_classes: int = 0,
) -> tuple[nn.Linear, dict[str, Any], torch.Tensor, torch.Tensor]:
    set_seed(config.seed)
    mean, scale = fit_standardizer(activations["train"])
    features = {split: standardize(activations[split], mean, scale) for split in SPLITS}
    is_regression = target_type in REGRESSION_TARGET_TYPES
    output_dim = 1 if is_regression else num_classes
    probe = nn.Linear(features["train"].shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if is_regression:
        target_mean = targets["train"].mean()
        target_scale = targets["train"].std(unbiased=False).clamp_min(1e-6)
        fit_targets = {split: (targets[split] - target_mean) / target_scale for split in SPLITS}
        criterion: nn.Module = nn.MSELoss()
    else:
        target_mean = torch.tensor(0.0)
        target_scale = torch.tensor(1.0)
        fit_targets = targets
        criterion = nn.CrossEntropyLoss()

    loader = _data_loader(
        features["train"],
        fit_targets["train"],
        batch_size=config.probe_batch_size,
        seed=config.seed,
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        probe.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = probe(batch_x)
            loss = criterion(output.squeeze(-1), batch_y) if is_regression else criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)
        probe.eval()
        dev_output = _predict(probe, features["dev"], device, config.probe_batch_size)
        dev_loss = float(
            criterion(dev_output.squeeze(-1), fit_targets["dev"]).item()
            if is_regression
            else criterion(dev_output, fit_targets["dev"]).item()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "dev_loss": dev_loss,
            }
        )
        if dev_loss < best_loss - 1e-7:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in probe.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("probe training did not produce a checkpoint")
    probe.load_state_dict(best_state)
    probe.eval()

    split_metrics: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        output = _predict(probe, features[split], device, config.probe_batch_size)
        if is_regression:
            prediction = output.squeeze(-1) * target_scale + target_mean
            split_metrics[split] = regression_metrics(targets[split], prediction)
        else:
            prediction = output.argmax(dim=-1)
            split_metrics[split] = macro_classification_metrics(targets[split], prediction, num_classes)
            if num_classes >= 5:
                top5 = output.topk(k=5, dim=-1).indices
                split_metrics[split]["top5_accuracy"] = float(
                    (top5 == targets[split][:, None]).any(dim=-1).float().mean().item()
                )

    details: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_dev_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
        "split_metrics": split_metrics,
        "input_dim": int(features["train"].shape[1]),
        "output_dim": output_dim,
        "target_mean": float(target_mean.item()),
        "target_scale": float(target_scale.item()),
    }
    return probe.cpu(), details, mean, scale


def baseline_metrics(
    *,
    rows_by_split: dict[str, list[dict[str, Any]]],
    targets: dict[str, torch.Tensor],
    target_type: str,
    num_classes: int = 0,
) -> dict[str, dict[str, float]]:
    if target_type in REGRESSION_TARGET_TYPES:
        train_mean = targets["train"].mean()
        return {
            split: regression_metrics(targets[split], torch.full_like(targets[split], train_mean))
            for split in SPLITS
        }
    train_counts = Counter(int(value) for value in targets["train"].tolist())
    majority = min((-count, label) for label, count in train_counts.items())[1]
    return {
        split: macro_classification_metrics(
            targets[split],
            torch.full_like(targets[split], majority),
            num_classes,
        )
        for split in SPLITS
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_feature(
    *,
    feature_id: str,
    data_path: Path,
    output_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    layer: LayerSpec,
    config: ProbeConfig,
    device: torch.device,
) -> dict[str, Any]:
    started = time.time()
    rows_by_split = read_feature_rows(data_path, config.max_examples_per_split)
    target_type = str(rows_by_split["train"][0]["target_type"])
    activation_started = time.time()
    activations = {
        split: extract_activations(
            model=model,
            tokenizer=tokenizer,
            rows=rows_by_split[split],
            layer=layer,
            batch_size=config.activation_batch_size,
            device=device,
        )
        for split in SPLITS
    }
    activation_seconds = time.time() - activation_started
    if target_type in REGRESSION_TARGET_TYPES:
        targets = regression_targets(rows_by_split)
        labels: list[str] = []
        num_classes = 0
    else:
        targets, labels = classification_targets(rows_by_split)
        num_classes = len(labels)
    probe, training, mean, scale = train_linear_probe(
        activations=activations,
        targets=targets,
        target_type=target_type,
        config=config,
        device=device,
        num_classes=num_classes,
    )
    baseline = baseline_metrics(
        rows_by_split=rows_by_split,
        targets=targets,
        target_type=target_type,
        num_classes=num_classes,
    )
    result = {
        "feature_id": feature_id,
        "target_type": target_type,
        "labels": labels,
        "num_classes": num_classes,
        "split_sizes": {split: len(rows_by_split[split]) for split in SPLITS},
        "input_names": [str(row["name"]) for row in rows_by_split["train"][0]["inputs"]],
        "layer": asdict(layer),
        "probe": training,
        "baseline": baseline,
        "activation_seconds": activation_seconds,
        "elapsed_seconds": time.time() - started,
    }
    feature_dir = output_dir / feature_id
    feature_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "labels": labels,
            "target_type": target_type,
            "layer": asdict(layer),
            "config": asdict(config),
            "target_mean": training["target_mean"],
            "target_scale": training["target_scale"],
        },
        feature_dir / "probe.pt",
    )
    write_json(feature_dir / "metrics.json", result)
    (feature_dir / "DONE").write_text("done\n", encoding="utf-8")
    return result


def completed_feature_result(output_dir: Path, feature_id: str) -> dict[str, Any] | None:
    feature_dir = output_dir / feature_id
    metrics_path = feature_dir / "metrics.json"
    if not (feature_dir / "DONE").exists() or not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def flatten_result_row(result: dict[str, Any]) -> dict[str, Any]:
    test_metrics = result["probe"]["split_metrics"]["test"]
    baseline = result["baseline"]["test"]
    row: dict[str, Any] = {
        "feature_id": result["feature_id"],
        "target_type": result["target_type"],
        "num_classes": result["num_classes"],
        "input_dim": result["probe"]["input_dim"],
        "best_epoch": result["probe"]["best_epoch"],
        "activation_seconds": result["activation_seconds"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    row.update({f"baseline_test_{key}": value for key, value in baseline.items()})
    return row


def validate_feature_ids(requested: Iterable[str], available: Sequence[str]) -> list[str]:
    requested_list = list(requested)
    if not requested_list or requested_list == ["all"]:
        return list(available)
    missing = [feature for feature in requested_list if feature not in available]
    if missing:
        raise ValueError(f"unknown features: {missing}")
    return requested_list
