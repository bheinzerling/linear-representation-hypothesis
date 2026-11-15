#!/usr/bin/env python3
"""Train an all-layer number-property probe and test generative dissociation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Sequence

import torch
from torch import nn
import transformers


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introlm.experiments.number_feature_linear_probe import (  # noqa: E402
    ProbeConfig,
    train_linear_probe,
)
from introlm.experiments.number_property_behavior_screen import (  # noqa: E402
    flatten_messages,
)
from introlm.experiments.number_property_steering import (  # noqa: E402
    PROPERTY_SPECS,
    PropertySpec,
    annotate_joint,
    balanced_occurrence_splits,
    balanced_value_sample,
    balanced_value_splits,
    direct_identity_messages,
    direct_property_messages,
    neutral_probe_messages,
    pairwise_probe_directions,
    paired_condition_summary,
    parse_identity,
    parse_property_label,
    prompt_messages,
    prompt_variants,
    random_control_directions,
    range_spanning_demonstrations,
    sampled_balanced_demonstrations,
    split_hash,
    split_overlap_counts,
    summarize_joint,
)


LOGGER = logging.getLogger("olmo_number_property_steering")
DEFAULT_MODEL = "allenai/OLMo-2-1124-13B-Instruct"


@dataclass(frozen=True)
class RenderedPrompt:
    value: int
    text: str
    number_token_indices: tuple[int, ...]
    number_token_ids: tuple[int, ...]
    token_count: int


@dataclass(frozen=True)
class Config:
    model: str
    property_id: str
    output_dir: str
    layers: tuple[int, ...]
    candidate_layer_count: int
    alphas: tuple[float, ...]
    activation_batch_size: int
    generation_batch_size: int
    probe_batch_size: int
    max_probe_epochs: int
    probe_patience: int
    probe_learning_rate: float
    probe_weight_decay: float
    minimum_direct_property: float
    minimum_direct_identity: float
    minimum_baseline_joint_accuracy: float
    minimum_joint_coverage: float
    direct_max_new_tokens: int
    joint_max_new_tokens: int
    seed: int
    split_mode: str
    split_seed: int
    probe_seed: int
    demonstration_seed: int
    steering_seed: int
    steering_eval_size: int | None
    dtype: str
    device: str
    local_files_only: bool
    chat_template_mode: str
    chat_template_used: bool
    probe_loci: tuple[str, ...]
    require_digit_tokenization: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property", choices=tuple(PROPERTY_SPECS), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--candidate-layer-count", type=int, default=10)
    parser.add_argument("--alphas", default="0.5,1,2,4,8,16")
    parser.add_argument("--activation-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--max-probe-epochs", type=int, default=100)
    parser.add_argument("--probe-patience", type=int, default=12)
    parser.add_argument("--probe-learning-rate", type=float, default=0.03)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-direct-property", type=float, default=0.90)
    parser.add_argument("--minimum-direct-identity", type=float, default=0.95)
    parser.add_argument("--minimum-baseline-joint-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-joint-coverage", type=float, default=0.95)
    parser.add_argument("--direct-max-new-tokens", type=int, default=4)
    parser.add_argument("--joint-max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--split-mode",
        choices=("value", "occurrence"),
        default="value",
        help="Use value-disjoint splits or occurrence-disjoint splits with value overlap.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Split construction seed; defaults to --seed.",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        help="Linear-probe fitting seed; defaults to --seed.",
    )
    parser.add_argument(
        "--demonstration-seed",
        type=int,
        help="Class-balanced ICL demonstration seed; defaults to --seed.",
    )
    parser.add_argument(
        "--steering-seed",
        type=int,
        help="Held-out steering-value and random-control seed; defaults to --seed.",
    )
    parser.add_argument(
        "--steering-eval-size",
        type=int,
        help="Optional class-balanced subset of test values used for joint steering.",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--chat-template",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument(
        "--probe-loci",
        default="number",
        help=(
            "Comma-separated numeral loci: number, first_digit, last_digit, "
            "or digit_mean. 'number' requires a single-token numeral."
        ),
    )
    parser.add_argument(
        "--require-digit-tokenization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require each numeral character to map to exactly one token.",
    )
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def parse_layers(raw: str, num_layers: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(num_layers))
    layers = tuple(
        dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip())
    )
    if not layers or any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"layers must be in [0, {num_layers - 1}]")
    return layers


def parse_alphas(raw: str) -> tuple[float, ...]:
    alphas = tuple(
        dict.fromkeys(float(item.strip()) for item in raw.split(",") if item.strip())
    )
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be positive")
    return alphas


PROBE_LOCI = ("number", "first_digit", "last_digit", "digit_mean")


def parse_probe_loci(raw: str) -> tuple[str, ...]:
    loci = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    unknown = set(loci).difference(PROBE_LOCI)
    if not loci or unknown:
        raise ValueError(
            f"probe loci must be selected from {PROBE_LOCI}; got {sorted(unknown)}"
        )
    return loci


def probe_context_id(value: int, split: str, split_mode: str, seed: int) -> int:
    if split_mode == "value":
        return 0
    if split_mode != "occurrence":
        raise ValueError(f"unknown split mode: {split_mode}")
    offsets = {"train": 0, "dev": 1, "test": 2}
    if split not in offsets:
        raise ValueError(f"unknown split: {split}")
    return (int(value) + int(seed) + offsets[split]) % 4


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def use_chat_template(tokenizer: Any, mode: str) -> bool:
    available = bool(getattr(tokenizer, "chat_template", None))
    if mode == "always" and not available:
        raise ValueError("--chat-template always requires a tokenizer chat template")
    return available and mode != "never"


def render_messages(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    value: int,
    chat: bool,
    *,
    require_digit_tokenization: bool = False,
) -> RenderedPrompt:
    text = (
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if chat
        else flatten_messages(messages)
    )
    numeral = str(int(value))
    char_start = text.rfind(numeral)
    if char_start < 0:
        raise ValueError(f"query numeral {numeral!r} absent from rendered prompt")
    char_end = char_start + len(numeral)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    indices = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if max(int(start), char_start) < min(int(end), char_end)
    ]
    if not indices:
        raise ValueError(f"query numeral {numeral!r} maps to no token indices")
    if require_digit_tokenization:
        offsets = encoded["offset_mapping"]
        expected_offsets = [
            (char_start + index, char_start + index + 1)
            for index in range(len(numeral))
        ]
        observed_offsets = [tuple(map(int, offsets[index])) for index in indices]
        if observed_offsets != expected_offsets:
            raise ValueError(
                f"query numeral {numeral!r} is not one-token-per-digit: "
                f"indices={indices}, offsets={observed_offsets}"
            )
    return RenderedPrompt(
        value=int(value),
        text=text,
        number_token_indices=tuple(indices),
        number_token_ids=tuple(int(encoded["input_ids"][index]) for index in indices),
        token_count=len(encoded["input_ids"]),
    )


def unpadded_locus_indices(prompt: RenderedPrompt, locus: str) -> tuple[int, ...]:
    if locus == "number":
        if len(prompt.number_token_indices) != 1:
            raise ValueError(
                f"number locus requires one token for {prompt.value}; "
                f"got {prompt.number_token_indices}"
            )
        return prompt.number_token_indices
    if locus == "first_digit":
        return (prompt.number_token_indices[0],)
    if locus == "last_digit":
        return (prompt.number_token_indices[-1],)
    if locus == "digit_mean":
        return prompt.number_token_indices
    raise ValueError(f"unknown probe locus: {locus}")


def encoded_batch(
    tokenizer: Any,
    prompts: Sequence[RenderedPrompt],
    device: torch.device,
    locus: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [prompt.text for prompt in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    encoded.pop("token_type_ids", None)
    sequence_length = int(encoded["input_ids"].shape[1])
    selected = [unpadded_locus_indices(prompt, locus) for prompt in prompts]
    width = max(len(indices) for indices in selected)
    positions = torch.zeros((len(prompts), width), dtype=torch.long)
    position_mask = torch.zeros((len(prompts), width), dtype=torch.bool)
    for row, (prompt, indices) in enumerate(zip(prompts, selected, strict=True)):
        left_padding = sequence_length - prompt.token_count
        positions[row, : len(indices)] = torch.tensor(
            [left_padding + index for index in indices], dtype=torch.long
        )
        position_mask[row, : len(indices)] = True
        for token_index, token_id in zip(
            prompt.number_token_indices, prompt.number_token_ids, strict=True
        ):
            observed = int(encoded["input_ids"][row, left_padding + token_index].item())
            if observed != token_id:
                raise ValueError(
                    f"left-padding token mismatch for {prompt.value}: "
                    f"expected {token_id}, got {observed}"
                )
    return (
        {key: value.to(device) for key, value in encoded.items()},
        positions.to(device),
        position_mask.to(device),
    )


def aggregate_locus_hidden(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    selected = hidden[rows, positions]
    weights = position_mask.to(selected.dtype)[:, :, None]
    return (selected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def extract_activations(
    *,
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[RenderedPrompt],
    layers: Sequence[int],
    loci: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    chunks: dict[str, dict[int, list[torch.Tensor]]] = {
        locus: {int(layer): [] for layer in layers} for locus in loci
    }
    for start in range(0, len(prompts), int(batch_size)):
        batch = prompts[start : start + int(batch_size)]
        encoded_by_locus = {
            locus: encoded_batch(tokenizer, batch, device, locus) for locus in loci
        }
        encoded = next(iter(encoded_by_locus.values()))[0]
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for layer in layers:
            hidden = outputs.hidden_states[int(layer) + 1]
            for locus, (_encoded, positions, position_mask) in encoded_by_locus.items():
                chunks[locus][int(layer)].append(
                    aggregate_locus_hidden(hidden, positions, position_mask)
                    .float()
                    .cpu()
                )
        del outputs
    return {
        locus: {layer: torch.cat(parts) for layer, parts in layer_chunks.items()}
        for locus, layer_chunks in chunks.items()
    }


def probe_config(config: Config, output_dir: Path, seed: int) -> ProbeConfig:
    return ProbeConfig(
        model=config.model,
        dataset_dir="generated-in-memory",
        output_dir=str(output_dir),
        layer="block-sweep",
        activation_batch_size=config.activation_batch_size,
        probe_batch_size=config.probe_batch_size,
        max_epochs=config.max_probe_epochs,
        patience=config.probe_patience,
        learning_rate=config.probe_learning_rate,
        weight_decay=config.probe_weight_decay,
        seed=int(seed),
        dtype=config.dtype,
        device=config.device,
        local_files_only=config.local_files_only,
    )


def target_tensors(
    values: dict[str, Sequence[int]], spec: PropertySpec
) -> dict[str, torch.Tensor]:
    return {
        split: torch.tensor(
            [spec.label(value) for value in values[split]], dtype=torch.long
        )
        for split in ("train", "dev", "test")
    }


def fit_probe_sweep(
    *,
    activations: dict[str, dict[int, torch.Tensor]],
    values: dict[str, Sequence[int]],
    layers: Sequence[int],
    spec: PropertySpec,
    config: Config,
    output_dir: Path,
    device: torch.device,
    locus: str,
) -> tuple[dict[int, dict[int, dict[str, Any]]], list[dict[str, Any]], dict[int, Any]]:
    targets = target_tensors(values, spec)
    all_directions = {}
    metrics = []
    checkpoints = {}
    for layer in layers:
        LOGGER.info("fitting %s probe block=%d", spec.property_id, layer)
        layer_activations = {
            split: activations[split][int(layer)] for split in ("train", "dev", "test")
        }
        probe, details, mean, scale = train_linear_probe(
            activations=layer_activations,
            targets=targets,
            target_type="binary" if spec.num_classes == 2 else "multiclass",
            config=probe_config(config, output_dir, config.probe_seed + int(layer)),
            device=device,
            num_classes=spec.num_classes,
        )
        directions = pairwise_probe_directions(
            weight=probe.weight.detach().cpu(),
            feature_scale=scale,
            train_activations=layer_activations["train"],
            spec=spec,
        )
        all_directions[int(layer)] = directions
        metrics.append(
            {
                "locus": locus,
                "layer": int(layer),
                "hidden_state_index": int(layer) + 1,
                "best_epoch": details["best_epoch"],
                "dev_loss": details["best_dev_loss"],
                "train_accuracy": details["split_metrics"]["train"]["accuracy"],
                "dev_accuracy": details["split_metrics"]["dev"]["accuracy"],
                "test_accuracy": details["split_metrics"]["test"]["accuracy"],
                "dev_balanced_accuracy": details["split_metrics"]["dev"][
                    "balanced_accuracy"
                ],
                "test_balanced_accuracy": details["split_metrics"]["test"][
                    "balanced_accuracy"
                ],
                "mean_projection_std": sum(
                    float(payload["projection_std"]) for payload in directions.values()
                )
                / len(directions),
            }
        )
        checkpoints[int(layer)] = {
            "locus": locus,
            "probe_state_dict": {
                key: value.detach().cpu() for key, value in probe.state_dict().items()
            },
            "feature_mean": mean,
            "feature_scale": scale,
            "directions": directions,
            "metrics": details,
        }
    return all_directions, metrics, checkpoints


def select_probe_locus(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select locus and layer from development metrics without consulting test scores."""

    locus_priority = {locus: -index for index, locus in enumerate(PROBE_LOCI)}
    selected = max(
        rows,
        key=lambda row: (
            float(row["dev_balanced_accuracy"]),
            float(row["dev_accuracy"]),
            -float(row["dev_loss"]),
            locus_priority[str(row["locus"])],
            -int(row["layer"]),
        ),
    )
    return {
        "locus": str(selected["locus"]),
        "best_dev_layer": int(selected["layer"]),
        "best_dev_accuracy": float(selected["dev_accuracy"]),
        "best_dev_balanced_accuracy": float(selected["dev_balanced_accuracy"]),
        "best_dev_loss": float(selected["dev_loss"]),
        "selection_mode": "dev_balanced_accuracy_then_accuracy_then_loss",
    }


def select_candidate_layers(
    rows: Sequence[dict[str, Any]], count: int
) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("candidate layer count must be positive")
    best_accuracy = max(float(row["dev_accuracy"]) for row in rows)
    near_best = sorted(
        int(row["layer"])
        for row in rows
        if float(row["dev_accuracy"]) >= best_accuracy - 0.02
    )
    if len(near_best) <= count:
        return tuple(near_best)
    indices = [
        round(index * (len(near_best) - 1) / (count - 1)) for index in range(count)
    ]
    return tuple(dict.fromkeys(near_best[index] for index in indices))


def transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Resolve text decoder blocks for text-only and multimodal HF wrappers."""

    candidates = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    )
    for path in candidates:
        value: Any = model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None:
            return value
    paths = ", ".join(".".join(path) for path in candidates)
    raise ValueError(f"expected text decoder blocks at one of: {paths}")


def direction_deltas(
    *,
    values: Sequence[int],
    spec: PropertySpec,
    directions: dict[int, dict[str, Any]],
    alpha: float,
    sign: float,
    device: torch.device,
) -> torch.Tensor:
    deltas = []
    for value in values:
        payload = directions[spec.label(int(value))]
        unit = payload["unit"].to(device=device, dtype=torch.float32)
        deltas.append(
            unit * float(payload["projection_std"]) * float(alpha) * float(sign)
        )
    return torch.stack(deltas)


def eos_ids(model: nn.Module, tokenizer: Any) -> set[int]:
    raw = getattr(model.generation_config, "eos_token_id", None)
    if raw is None:
        raw = tokenizer.eos_token_id
    if raw is None:
        return set()
    return {int(raw)} if isinstance(raw, int) else {int(value) for value in raw}


@torch.inference_mode()
def generate_rows(
    *,
    model: nn.Module,
    tokenizer: Any,
    values: Sequence[int],
    messages_builder: Callable[[int], Sequence[dict[str, str]]],
    spec: PropertySpec,
    split: str,
    condition: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    annotate: Callable[[int, str], dict[str, Any]],
    variant_id: str,
    view_id: str,
    chat: bool,
    locus: str,
    require_digit_tokenization: bool,
    directions: dict[int, dict[str, Any]] | None = None,
    alpha: float = 0.0,
    layer: int | None = None,
    sign: float = 1.0,
) -> list[dict[str, Any]]:
    model_layers = transformer_layers(model)
    stop_ids = eos_ids(model, tokenizer)
    rows = []
    for start in range(0, len(values), int(batch_size)):
        batch_values = [int(value) for value in values[start : start + int(batch_size)]]
        prompts = [
            render_messages(
                tokenizer,
                messages_builder(value),
                value,
                chat,
                require_digit_tokenization=require_digit_tokenization,
            )
            for value in batch_values
        ]
        encoded, positions, position_mask = encoded_batch(
            tokenizer, prompts, device, locus
        )
        handle = None
        delta_norms = torch.zeros(len(batch_values))
        if directions is not None:
            deltas = direction_deltas(
                values=batch_values,
                spec=spec,
                directions=directions,
                alpha=alpha,
                sign=sign,
                device=device,
            )
            delta_norms = deltas.norm(dim=1).cpu()

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                rows = torch.arange(len(batch_values), device=hidden.device)[:, None]
                expanded_deltas = deltas[:, None, :].expand(-1, positions.shape[1], -1)
                active_rows = rows.expand_as(positions)[position_mask]
                active_positions = positions[position_mask]
                patched[active_rows, active_positions] = (
                    patched[active_rows, active_positions].float()
                    + expanded_deltas[position_mask]
                ).to(hidden.dtype)
                return (patched, *output[1:]) if isinstance(output, tuple) else patched

            handle = model_layers[int(layer)].register_forward_hook(hook)
        try:
            outputs = model(**encoded, use_cache=True)
        finally:
            if handle is not None:
                handle.remove()
        attention_mask = encoded["attention_mask"]
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1]
        finished = torch.zeros(len(batch_values), dtype=torch.bool, device=device)
        generated: list[torch.Tensor] = []
        for step in range(int(max_new_tokens)):
            already_finished = finished.clone()
            tokens = next_logits.argmax(dim=-1)
            emitted = torch.where(
                already_finished,
                torch.full_like(tokens, tokenizer.pad_token_id),
                tokens,
            )
            generated.append(emitted.cpu())
            for eos_id in stop_ids:
                finished |= emitted == eos_id
            if step + 1 == max_new_tokens or bool(finished.all().item()):
                break
            attention_mask = torch.cat(
                [attention_mask, (~already_finished).long()[:, None]], dim=1
            )
            outputs = model(
                input_ids=emitted[:, None],
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1]
        token_rows = torch.stack(generated, dim=1)
        completions = tokenizer.batch_decode(token_rows, skip_special_tokens=True)
        for index, (value, prompt, completion) in enumerate(
            zip(batch_values, prompts, completions, strict=True)
        ):
            rows.append(
                {
                    "property_id": spec.property_id,
                    "view_id": view_id,
                    "split": split,
                    "condition": condition,
                    "variant_id": variant_id,
                    "value": value,
                    "prompt": prompt.text,
                    "generated_text": completion,
                    "query_locus": locus,
                    "query_token_indices": list(prompt.number_token_indices),
                    "query_token_ids": list(prompt.number_token_ids),
                    "patched_token_count": int(position_mask[index].sum().item()),
                    "delta_norm": float(delta_norms[index].item()),
                    **annotate(value, completion),
                }
            )
        del outputs, past
    return rows


def direct_property_annotation(
    spec: PropertySpec, value: int, text: str
) -> dict[str, Any]:
    prediction = parse_property_label(text, spec.property_id)
    source = spec.label(value)
    target = spec.target(value)
    return {
        "prediction": prediction,
        "source_label": source,
        "target_label": target,
        "parsed": prediction is not None,
        "correct": prediction == source,
        "target_hit": prediction == target,
    }


def direct_identity_annotation(
    _spec: PropertySpec, value: int, text: str
) -> dict[str, Any]:
    prediction = parse_identity(text)
    return {
        "prediction": prediction,
        "parsed": prediction is not None,
        "correct": prediction == int(value),
        "target_hit": prediction == int(value),
    }


def summarize_direct_condition(
    baseline_property: Sequence[dict[str, Any]],
    candidate_property: Sequence[dict[str, Any]],
    baseline_identity: Sequence[dict[str, Any]],
    candidate_identity: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not (
        len(baseline_property)
        == len(candidate_property)
        == len(baseline_identity)
        == len(candidate_identity)
    ):
        raise ValueError("direct-readout rows are not aligned")
    property_eligible = [
        index for index, row in enumerate(baseline_property) if row["correct"]
    ]
    identity_eligible = [
        index for index, row in enumerate(baseline_identity) if row["correct"]
    ]
    return {
        "n": len(baseline_property),
        "baseline_property_accuracy": sum(row["correct"] for row in baseline_property)
        / len(baseline_property),
        "baseline_identity_accuracy": sum(row["correct"] for row in baseline_identity)
        / len(baseline_identity),
        "conditional_target_rate": (
            sum(candidate_property[index]["target_hit"] for index in property_eligible)
            / len(property_eligible)
            if property_eligible
            else 0.0
        ),
        "identity_preservation_rate": (
            sum(candidate_identity[index]["correct"] for index in identity_eligible)
            / len(identity_eligible)
            if identity_eligible
            else 0.0
        ),
        "property_eligible_count": len(property_eligible),
        "identity_eligible_count": len(identity_eligible),
    }


def select_direct_condition(
    rows: Sequence[dict[str, Any]], minimum_property: float, minimum_identity: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from empty direct-readout rows")
    baseline_property_accuracy = float(rows[0]["baseline_property_accuracy"])
    property_gate_passed = baseline_property_accuracy >= minimum_property
    eligible = [
        row
        for row in rows
        if float(row["identity_preservation_rate"]) >= minimum_identity
    ]
    candidates = eligible or list(rows)
    selected = max(
        candidates,
        key=lambda row: (
            float(row["conditional_target_rate"]),
            float(row["identity_preservation_rate"]),
            -float(row["alpha"]),
            -int(row["layer"]),
        ),
    )
    return {
        **selected,
        "selection_mode": (
            "hard_identity_gate_then_target_rate"
            if eligible
            else "identity_gate_failed_target_rate_fallback"
        ),
        "minimum_direct_identity": minimum_identity,
        "minimum_direct_property": minimum_property,
        "direct_property_gate_passed": property_gate_passed,
        "eligible_condition_count": len(eligible),
    }


def fit_shuffled_directions(
    *,
    activations: dict[str, torch.Tensor],
    values: dict[str, Sequence[int]],
    spec: PropertySpec,
    config: Config,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    targets = target_tensors(values, spec)
    shuffled = {}
    for offset, split in enumerate(("train", "dev", "test")):
        order = torch.randperm(
            len(targets[split]),
            generator=torch.Generator().manual_seed(
                config.probe_seed + 9000 + offset
            ),
        )
        shuffled[split] = targets[split][order]
    probe, details, mean, scale = train_linear_probe(
        activations=activations,
        targets=shuffled,
        target_type="binary" if spec.num_classes == 2 else "multiclass",
        config=probe_config(config, output_dir, config.probe_seed + 9999),
        device=device,
        num_classes=spec.num_classes,
    )
    directions = pairwise_probe_directions(
        weight=probe.weight.detach().cpu(),
        feature_scale=scale,
        train_activations=activations["train"],
        spec=spec,
    )
    return directions, {
        "probe_state_dict": {
            key: value.detach().cpu() for key, value in probe.state_dict().items()
        },
        "feature_mean": mean,
        "feature_scale": scale,
        "directions": directions,
        "metrics": details,
    }


def select_joint_prompt(
    rows: Sequence[dict[str, Any]],
    minimum_accuracy: float,
    minimum_coverage: float,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if float(row["baseline_joint_accuracy"]) >= minimum_accuracy
        and float(row["baseline_joint_coverage"]) >= minimum_coverage
    ]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda row: (
            float(row["conditional_dissociation_rate"]),
            float(row["baseline_joint_accuracy"]),
            -int(row["n_demonstrations"]),
            str(row["variant_id"]),
        ),
    )
    return {
        **selected,
        "selection_mode": "baseline_gate_then_conditional_dissociation",
        "eligible_prompt_count": len(eligible),
        "minimum_baseline_joint_accuracy": minimum_accuracy,
        "minimum_joint_coverage": minimum_coverage,
    }


def prefixed(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items() if key != "n"}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    spec = PROPERTY_SPECS[args.property]
    split_seed = args.seed if args.split_seed is None else args.split_seed
    probe_seed = args.seed if args.probe_seed is None else args.probe_seed
    demonstration_seed = (
        args.seed if args.demonstration_seed is None else args.demonstration_seed
    )
    steering_seed = args.seed if args.steering_seed is None else args.steering_seed
    splits = (
        balanced_value_splits(args.property, split_seed)
        if args.split_mode == "value"
        else balanced_occurrence_splits(args.property, split_seed)
    )
    overlaps = split_overlap_counts(splits)
    steering_values = balanced_value_sample(
        splits["test"],
        args.property,
        args.steering_eval_size,
        seed=steering_seed,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "property_id": args.property,
                    "num_classes": spec.num_classes,
                    "split_sizes": {key: len(value) for key, value in splits.items()},
                    "split_hash": split_hash(splits),
                    "split_mode": args.split_mode,
                    "split_seed": split_seed,
                    "probe_seed": probe_seed,
                    "demonstration_seed": demonstration_seed,
                    "steering_seed": steering_seed,
                    "steering_eval_size": len(steering_values),
                    "value_overlap_counts": overlaps,
                },
                indent=2,
            )
        )
        return
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (args.output_dir / "DONE").exists() and not args.overwrite:
        LOGGER.info("already complete: %s", args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    chat = use_chat_template(tokenizer, args.chat_template)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = parse_layers(args.layers, len(transformer_layers(model)))
    alphas = parse_alphas(args.alphas)
    probe_loci = parse_probe_loci(args.probe_loci)
    config = Config(
        model=args.model,
        property_id=args.property,
        output_dir=str(args.output_dir),
        layers=layers,
        candidate_layer_count=args.candidate_layer_count,
        alphas=alphas,
        activation_batch_size=args.activation_batch_size,
        generation_batch_size=args.generation_batch_size,
        probe_batch_size=args.probe_batch_size,
        max_probe_epochs=args.max_probe_epochs,
        probe_patience=args.probe_patience,
        probe_learning_rate=args.probe_learning_rate,
        probe_weight_decay=args.probe_weight_decay,
        minimum_direct_property=args.minimum_direct_property,
        minimum_direct_identity=args.minimum_direct_identity,
        minimum_baseline_joint_accuracy=args.minimum_baseline_joint_accuracy,
        minimum_joint_coverage=args.minimum_joint_coverage,
        direct_max_new_tokens=args.direct_max_new_tokens,
        joint_max_new_tokens=args.joint_max_new_tokens,
        seed=args.seed,
        split_mode=args.split_mode,
        split_seed=split_seed,
        probe_seed=probe_seed,
        demonstration_seed=demonstration_seed,
        steering_seed=steering_seed,
        steering_eval_size=args.steering_eval_size,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
        chat_template_mode=args.chat_template,
        chat_template_used=chat,
        probe_loci=probe_loci,
        require_digit_tokenization=args.require_digit_tokenization,
    )
    write_json(args.output_dir / "config.json", asdict(config))
    write_json(
        args.output_dir / "splits.json",
        {
            "seed": split_seed,
            "mode": args.split_mode,
            "split_hash": split_hash(splits),
            "value_overlap_counts": overlaps,
            "values": splits,
            "steering_values": steering_values,
        },
    )
    raw_prompt = not chat
    direct_demonstrations = sampled_balanced_demonstrations(
        splits["train"],
        spec.property_id,
        per_class=1,
        seed=demonstration_seed,
        salt="direct",
    )
    identity_demonstrations = direct_demonstrations[:2]

    probe_prompts = {
        split: [
            render_messages(
                tokenizer,
                neutral_probe_messages(
                    value,
                    raw=raw_prompt,
                    context_id=probe_context_id(
                        value, split, args.split_mode, split_seed
                    ),
                ),
                value,
                chat,
                require_digit_tokenization=args.require_digit_tokenization,
            )
            for value in splits[split]
        ]
        for split in ("train", "dev", "test")
    }
    activations = {}
    for split in ("train", "dev", "test"):
        LOGGER.info("extracting %s activations for %s", split, spec.property_id)
        activations[split] = extract_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=probe_prompts[split],
            layers=layers,
            loci=probe_loci,
            batch_size=args.activation_batch_size,
            device=device,
        )
    directions_by_locus = {}
    probe_rows = []
    checkpoints = {}
    for locus in probe_loci:
        locus_directions, locus_rows, locus_checkpoints = fit_probe_sweep(
            activations={
                split: activations[split][locus] for split in ("train", "dev", "test")
            },
            values=splits,
            layers=layers,
            spec=spec,
            config=config,
            output_dir=args.output_dir,
            device=device,
            locus=locus,
        )
        directions_by_locus[locus] = locus_directions
        probe_rows.extend(locus_rows)
        checkpoints[locus] = locus_checkpoints
    write_csv(args.output_dir / "probe_layer_metrics.csv", probe_rows)
    torch.save(checkpoints, args.output_dir / "probe_checkpoints.pt")
    selected_probe_locus = select_probe_locus(probe_rows)
    selected_locus = str(selected_probe_locus["locus"])
    selected_probe_row = next(
        row
        for row in probe_rows
        if row["locus"] == selected_locus
        and int(row["layer"]) == int(selected_probe_locus["best_dev_layer"])
    )
    selected_locus_rows = [row for row in probe_rows if row["locus"] == selected_locus]
    directions = directions_by_locus[selected_locus]
    candidate_layers = select_candidate_layers(
        selected_locus_rows, args.candidate_layer_count
    )
    write_json(args.output_dir / "selected_probe_locus.json", selected_probe_locus)
    write_json(args.output_dir / "candidate_layers.json", list(candidate_layers))

    dev_values = splits["dev"]
    baseline_property = generate_rows(
        model=model,
        tokenizer=tokenizer,
        values=dev_values,
        messages_builder=lambda value: direct_property_messages(
            spec.property_id,
            value,
            raw=raw_prompt,
            demonstration_values=direct_demonstrations,
        ),
        spec=spec,
        split="dev",
        condition="baseline",
        batch_size=args.generation_batch_size,
        max_new_tokens=args.direct_max_new_tokens,
        device=device,
        annotate=lambda value, text: direct_property_annotation(spec, value, text),
        variant_id="direct_property",
        view_id=spec.views[0],
        chat=chat,
        locus=selected_locus,
        require_digit_tokenization=args.require_digit_tokenization,
    )
    baseline_identity = generate_rows(
        model=model,
        tokenizer=tokenizer,
        values=dev_values,
        messages_builder=lambda value: direct_identity_messages(
            value,
            raw=raw_prompt,
            demonstration_values=identity_demonstrations,
        ),
        spec=spec,
        split="dev",
        condition="baseline",
        batch_size=args.generation_batch_size,
        max_new_tokens=args.direct_max_new_tokens,
        device=device,
        annotate=lambda value, text: direct_identity_annotation(spec, value, text),
        variant_id="direct_identity",
        view_id="identity",
        chat=chat,
        locus=selected_locus,
        require_digit_tokenization=args.require_digit_tokenization,
    )
    direct_rows = []
    direct_examples = list(baseline_property) + list(baseline_identity)
    for layer in candidate_layers:
        for alpha in alphas:
            LOGGER.info(
                "direct sweep %s block=%d alpha=%g", spec.property_id, layer, alpha
            )
            candidate_property = generate_rows(
                model=model,
                tokenizer=tokenizer,
                values=dev_values,
                messages_builder=lambda value: direct_property_messages(
                    spec.property_id,
                    value,
                    raw=raw_prompt,
                    demonstration_values=direct_demonstrations,
                ),
                spec=spec,
                split="dev",
                condition="selected_direction",
                batch_size=args.generation_batch_size,
                max_new_tokens=args.direct_max_new_tokens,
                device=device,
                annotate=lambda value, text: direct_property_annotation(
                    spec, value, text
                ),
                variant_id="direct_property",
                view_id=spec.views[0],
                chat=chat,
                locus=selected_locus,
                require_digit_tokenization=args.require_digit_tokenization,
                directions=directions[int(layer)],
                alpha=alpha,
                layer=layer,
            )
            candidate_identity = generate_rows(
                model=model,
                tokenizer=tokenizer,
                values=dev_values,
                messages_builder=lambda value: direct_identity_messages(
                    value,
                    raw=raw_prompt,
                    demonstration_values=identity_demonstrations,
                ),
                spec=spec,
                split="dev",
                condition="selected_direction",
                batch_size=args.generation_batch_size,
                max_new_tokens=args.direct_max_new_tokens,
                device=device,
                annotate=lambda value, text: direct_identity_annotation(
                    spec, value, text
                ),
                variant_id="direct_identity",
                view_id="identity",
                chat=chat,
                locus=selected_locus,
                require_digit_tokenization=args.require_digit_tokenization,
                directions=directions[int(layer)],
                alpha=alpha,
                layer=layer,
            )
            direct_examples.extend(candidate_property + candidate_identity)
            direct_rows.append(
                {
                    "locus": selected_locus,
                    "layer": int(layer),
                    "alpha": float(alpha),
                    **summarize_direct_condition(
                        baseline_property,
                        candidate_property,
                        baseline_identity,
                        candidate_identity,
                    ),
                }
            )
    selected_condition = select_direct_condition(
        direct_rows, args.minimum_direct_property, args.minimum_direct_identity
    )
    selected_layer = int(selected_condition["layer"])
    selected_alpha = float(selected_condition["alpha"])
    selected_directions = directions[selected_layer]
    write_csv(args.output_dir / "direct_layer_alpha_metrics.csv", direct_rows)
    write_json(args.output_dir / "selected_condition.json", selected_condition)
    write_jsonl_gz(
        args.output_dir / "direct_readout_examples.jsonl.gz", direct_examples
    )

    selected_activations = {
        split: activations[split][selected_locus][selected_layer]
        for split in ("train", "dev", "test")
    }
    shuffled_directions, shuffled_checkpoint = fit_shuffled_directions(
        activations=selected_activations,
        values=splits,
        spec=spec,
        config=config,
        output_dir=args.output_dir,
        device=device,
    )
    random_directions = random_control_directions(
        selected_directions, steering_seed + 5000
    )
    torch.save(
        {
            "selected": selected_directions,
            "shuffled_label": shuffled_directions,
            "random": random_directions,
            "shuffled_checkpoint": shuffled_checkpoint,
        },
        args.output_dir / "selected_and_control_directions.pt",
    )

    all_joint_examples = []
    prompt_metric_rows = []
    view_results = []
    prompt_manifest = {}
    for view_id in spec.views:
        variants = prompt_variants(
            splits["train"],
            spec.property_id,
            view_id,
            demonstration_seed=demonstration_seed,
        )
        prompt_manifest[view_id] = [asdict(variant) for variant in variants]
        view_prompt_rows = []
        dev_by_variant = {}
        for variant in variants:
            LOGGER.info("joint dev screen %s/%s", view_id, variant.variant_id)
            baseline = generate_rows(
                model=model,
                tokenizer=tokenizer,
                values=splits["dev"],
                messages_builder=lambda value, item=variant: prompt_messages(
                    item, value
                ),
                spec=spec,
                split="dev",
                condition="baseline",
                batch_size=args.generation_batch_size,
                max_new_tokens=args.joint_max_new_tokens,
                device=device,
                annotate=lambda value, text, view=view_id, style=variant.label_style: (
                    annotate_joint(
                        value=value,
                        text=text,
                        property_id=spec.property_id,
                        view_id=view,
                        label_style=style,
                    )
                ),
                variant_id=variant.variant_id,
                view_id=view_id,
                chat=chat,
                locus=selected_locus,
                require_digit_tokenization=args.require_digit_tokenization,
            )
            steered = generate_rows(
                model=model,
                tokenizer=tokenizer,
                values=splits["dev"],
                messages_builder=lambda value, item=variant: prompt_messages(
                    item, value
                ),
                spec=spec,
                split="dev",
                condition="selected_direction",
                batch_size=args.generation_batch_size,
                max_new_tokens=args.joint_max_new_tokens,
                device=device,
                annotate=lambda value, text, view=view_id, style=variant.label_style: (
                    annotate_joint(
                        value=value,
                        text=text,
                        property_id=spec.property_id,
                        view_id=view,
                        label_style=style,
                    )
                ),
                variant_id=variant.variant_id,
                view_id=view_id,
                chat=chat,
                locus=selected_locus,
                require_digit_tokenization=args.require_digit_tokenization,
                directions=selected_directions,
                alpha=selected_alpha,
                layer=selected_layer,
            )
            all_joint_examples.extend(baseline + steered)
            baseline_summary = summarize_joint(baseline)
            steered_summary = summarize_joint(steered)
            paired = paired_condition_summary(baseline, steered)
            row = {
                "property_id": spec.property_id,
                "view_id": view_id,
                "variant_id": variant.variant_id,
                "description": variant.description,
                "n_demonstrations": len(variant.demonstration_values),
                **prefixed("baseline", baseline_summary),
                **prefixed("steered", steered_summary),
                **paired,
            }
            view_prompt_rows.append(row)
            prompt_metric_rows.append(row)
            dev_by_variant[variant.variant_id] = (baseline, steered)
        selected_prompt = select_joint_prompt(
            view_prompt_rows,
            args.minimum_baseline_joint_accuracy,
            args.minimum_joint_coverage,
        )
        if selected_prompt is None:
            view_results.append(
                {
                    "view_id": view_id,
                    "gate_passed": False,
                    "best_dev_baseline_joint_accuracy": max(
                        float(row["baseline_joint_accuracy"])
                        for row in view_prompt_rows
                    ),
                    "test_metrics": [],
                }
            )
            continue
        selected_variant = next(
            variant
            for variant in variants
            if variant.variant_id == selected_prompt["variant_id"]
        )
        test_conditions = (
            ("baseline", None, 0.0, None, 1.0),
            (
                "selected_direction",
                selected_directions,
                selected_alpha,
                selected_layer,
                1.0,
            ),
            (
                "wrong_sign",
                selected_directions,
                selected_alpha,
                selected_layer,
                -1.0,
            ),
            (
                "shuffled_label",
                shuffled_directions,
                selected_alpha,
                selected_layer,
                1.0,
            ),
            (
                "random_direction",
                random_directions,
                selected_alpha,
                selected_layer,
                1.0,
            ),
        )
        condition_examples = {}
        for condition, condition_directions, alpha, layer, sign in test_conditions:
            LOGGER.info("joint test %s/%s", view_id, condition)
            condition_examples[condition] = generate_rows(
                model=model,
                tokenizer=tokenizer,
                values=steering_values,
                messages_builder=lambda value, item=selected_variant: prompt_messages(
                    item, value
                ),
                spec=spec,
                split="test",
                condition=condition,
                batch_size=args.generation_batch_size,
                max_new_tokens=args.joint_max_new_tokens,
                device=device,
                annotate=lambda value, text, view=view_id, style=selected_variant.label_style: (
                    annotate_joint(
                        value=value,
                        text=text,
                        property_id=spec.property_id,
                        view_id=view,
                        label_style=style,
                    )
                ),
                variant_id=selected_variant.variant_id,
                view_id=view_id,
                chat=chat,
                locus=selected_locus,
                require_digit_tokenization=args.require_digit_tokenization,
                directions=condition_directions,
                alpha=alpha,
                layer=layer,
                sign=sign,
            )
            all_joint_examples.extend(condition_examples[condition])
        baseline_test = condition_examples["baseline"]
        test_metrics = []
        for condition, rows in condition_examples.items():
            test_metrics.append(
                {
                    "condition": condition,
                    **summarize_joint(rows),
                    **(
                        paired_condition_summary(baseline_test, rows)
                        if condition != "baseline"
                        else {
                            "baseline_joint_correct_count": sum(
                                row["baseline_joint_correct"] for row in rows
                            ),
                            "conditional_dissociation_count": 0,
                            "conditional_dissociation_rate": 0.0,
                        }
                    ),
                }
            )
        view_results.append(
            {
                "view_id": view_id,
                "gate_passed": True,
                "selected_prompt": selected_prompt,
                "test_metrics": test_metrics,
            }
        )

    write_json(args.output_dir / "prompts.json", prompt_manifest)
    write_csv(args.output_dir / "dev_prompt_metrics.csv", prompt_metric_rows)
    write_jsonl_gz(args.output_dir / "joint_examples.jsonl.gz", all_joint_examples)
    results = {
        "config": asdict(config),
        "property": {
            "property_id": spec.property_id,
            "display_name": spec.display_name,
            "field_name": spec.field_name,
            "labels": list(spec.labels),
            "views": list(spec.views),
        },
        "split_hash": split_hash(splits),
        "split_mode": args.split_mode,
        "value_overlap_counts": overlaps,
        "steering_value_count": len(steering_values),
        "selected_probe_locus": selected_probe_locus,
        "probe_selected": selected_probe_row,
        "selected_condition": selected_condition,
        "candidate_layers": list(candidate_layers),
        "probe_best_dev": max(probe_rows, key=lambda row: float(row["dev_accuracy"])),
        "probe_best_test": max(probe_rows, key=lambda row: float(row["test_accuracy"])),
        "view_results": view_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "results.json", results)
    (args.output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
