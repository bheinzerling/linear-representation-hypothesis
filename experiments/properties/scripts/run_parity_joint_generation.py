#!/usr/bin/env python3
"""Test whether parity steering produces joint identity/parity contradictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import torch
from torch import nn
import transformers

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introlm.experiments.parity_activation_steering import (  # noqa: E402
    balanced_value_splits,
    value_split_hash,
)
from introlm.experiments.parity_joint_generation import (  # noqa: E402
    JointPromptVariant,
    annotate_completion,
    build_joint_prompt_variants,
    select_joint_prompt,
    summarize_completions,
    summarize_paired_conditions,
)


LOGGER = logging.getLogger("parity_joint_generation")
DEFAULT_SOURCE = (
    ROOT
    / "out"
    / "parity_activation_steering"
    / "llama31_8b_original2shot_value_disjoint_v1_local"
)
DEFAULT_OUTPUT = ROOT / "out" / "parity_joint_generation" / "llama31_8b_v1"


@dataclass(frozen=True)
class JointGenerationConfig:
    model: str
    source_run: str
    output_dir: str
    selection_split: str
    evaluation_split: str
    intervention_layer: int
    alpha: float
    projection_std: float
    batch_size: int
    max_new_tokens: int
    minimum_baseline_joint_accuracy: float
    minimum_joint_coverage: float
    seed: int
    dtype: str
    device: str
    local_files_only: bool
    max_values_per_split: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-split", choices=("train", "dev"), default="dev")
    parser.add_argument("--evaluation-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--minimum-baseline-joint-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-joint-coverage", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-values-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise ValueError("expected a causal LM with model.layers transformer blocks")
    return layers


def _query_metadata(
    tokenizer: Any,
    variant: JointPromptVariant,
    value: int,
) -> tuple[str, int, int, list[int]]:
    prefix, suffix = variant.pattern.split("{number}")
    numeral = str(int(value))
    text = prefix + numeral + suffix
    char_start = len(prefix)
    char_end = char_start + len(numeral)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if max(start, char_start) < min(end, char_end)
    ]
    if len(indices) != 1:
        raise ValueError(
            f"query number {numeral!r} in {variant.variant_id} maps to token indices {indices}"
        )
    index = indices[0]
    return text, index, int(encoded["input_ids"][index]), list(encoded["input_ids"])


def _eos_ids(model: nn.Module, tokenizer: Any) -> set[int]:
    raw = getattr(model.generation_config, "eos_token_id", None)
    if raw is None:
        raw = tokenizer.eos_token_id
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {int(raw)}
    return {int(value) for value in raw}


@torch.inference_mode()
def generate_variant(
    *,
    model: nn.Module,
    tokenizer: Any,
    variant: JointPromptVariant,
    values: Sequence[int],
    split: str,
    condition: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    direction_unit: torch.Tensor | None = None,
    projection_std: float = 0.0,
    alpha: float = 0.0,
    intervention_layer: int | None = None,
    sign_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    if direction_unit is None:
        if intervention_layer is not None or alpha != 0.0:
            raise ValueError("baseline generation cannot specify an intervention")
    elif intervention_layer is None or alpha <= 0.0 or projection_std <= 0.0:
        raise ValueError("steered generation requires layer, alpha, and projection scale")

    layers = _transformer_layers(model)
    eos_ids = _eos_ids(model, tokenizer)
    pad_token_id = int(tokenizer.pad_token_id)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), int(batch_size)):
        batch_values = [int(value) for value in values[start : start + int(batch_size)]]
        metadata = [_query_metadata(tokenizer, variant, value) for value in batch_values]
        prompts = [item[0] for item in metadata]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded.pop("token_type_ids", None)
        sequence_length = int(encoded["input_ids"].shape[1])
        prompt_lengths = [len(item[3]) for item in metadata]
        positions = torch.tensor(
            [sequence_length - length + item[1] for length, item in zip(prompt_lengths, metadata, strict=True)],
            dtype=torch.long,
            device=device,
        )
        for index, item in enumerate(metadata):
            observed = int(encoded["input_ids"][index, positions[index]].item())
            if observed != item[2]:
                raise ValueError(
                    f"padded token mismatch for {variant.variant_id}:{batch_values[index]}: "
                    f"expected {item[2]}, got {observed}"
                )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        handle = None
        delta_norm = 0.0
        if direction_unit is not None:
            unit = direction_unit.to(device=device, dtype=torch.float32)
            unit = unit / unit.norm()
            signs = torch.tensor(
                [1.0 if value % 2 == 1 else -1.0 for value in batch_values],
                dtype=torch.float32,
                device=device,
            )
            deltas = (
                signs[:, None]
                * float(sign_multiplier)
                * float(alpha)
                * float(projection_std)
                * unit[None, :]
            )
            delta_norm = float(deltas[0].norm().item())

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                batch_indices = torch.arange(len(batch_values), device=hidden.device)
                patched[batch_indices, positions, :] = (
                    patched[batch_indices, positions, :].float() + deltas
                ).to(dtype=patched.dtype)
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            handle = layers[int(intervention_layer)].register_forward_hook(hook)
        try:
            outputs = model(**encoded, use_cache=True)
        finally:
            if handle is not None:
                handle.remove()

        attention_mask = encoded["attention_mask"]
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1]
        finished = torch.zeros(len(batch_values), dtype=torch.bool, device=device)
        generated: list[torch.Tensor] = []
        for step in range(int(max_new_tokens)):
            was_finished = finished.clone()
            next_tokens = next_logits.argmax(dim=-1)
            emitted = torch.where(
                was_finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )
            generated.append(emitted.cpu())
            if eos_ids:
                emitted_is_eos = torch.zeros_like(finished)
                for eos_id in eos_ids:
                    emitted_is_eos |= emitted == int(eos_id)
                finished |= emitted_is_eos
            if step + 1 == int(max_new_tokens) or bool(finished.all().item()):
                break
            attention_mask = torch.cat(
                [attention_mask, (~was_finished).long()[:, None]],
                dim=1,
            )
            outputs = model(
                input_ids=emitted[:, None],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1]

        generated_ids = torch.stack(generated, dim=1)
        completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for index, (value, completion) in enumerate(zip(batch_values, completions, strict=True)):
            rows.append(
                {
                    "split": split,
                    "variant_id": variant.variant_id,
                    "condition": condition,
                    "value": value,
                    "query_token_index": int(metadata[index][1]),
                    "query_token_id": int(metadata[index][2]),
                    "generated_text": completion,
                    "delta_norm": delta_norm,
                    **annotate_completion(value=value, text=completion),
                }
            )
        del outputs, past_key_values, generated_ids
    return rows


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if key != "n"}


def _balanced_limit(values: Sequence[int], limit: int) -> list[int]:
    if limit <= 0 or limit >= len(values):
        return list(values)
    per_class = max(1, int(limit) // 2)
    selected = [value for value in values if value % 2 == 0][:per_class]
    selected += [value for value in values if value % 2 == 1][:per_class]
    return selected[:limit]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (args.output_dir / "DONE").exists() and not args.overwrite:
        LOGGER.info("completed output already exists: %s", args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    selected_condition = json.loads(
        (args.source_run / "selected_condition.json").read_text(encoding="utf-8")
    )
    direction_payload = torch.load(
        args.source_run / "selected_and_control_directions.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected_direction = direction_payload["selected_parity"]
    intervention_layer = int(selected_condition["layer"])
    alpha = float(selected_condition["alpha"])
    projection_std = float(selected_direction["projection_std"])
    if abs(projection_std - float(selected_condition["projection_std"])) > 1e-6:
        raise ValueError("selected-condition and checkpoint projection scales disagree")

    value_splits = balanced_value_splits(args.seed)
    selection_values = _balanced_limit(
        value_splits[args.selection_split], args.max_values_per_split
    )
    evaluation_values = _balanced_limit(
        value_splits[args.evaluation_split], args.max_values_per_split
    )
    variants = build_joint_prompt_variants(value_splits["train"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = JointGenerationConfig(
        model=args.model,
        source_run=str(args.source_run),
        output_dir=str(args.output_dir),
        selection_split=args.selection_split,
        evaluation_split=args.evaluation_split,
        intervention_layer=intervention_layer,
        alpha=alpha,
        projection_std=projection_std,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        minimum_baseline_joint_accuracy=args.minimum_baseline_joint_accuracy,
        minimum_joint_coverage=args.minimum_joint_coverage,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
        max_values_per_split=args.max_values_per_split,
    )
    write_json(args.output_dir / "config.json", asdict(config))
    write_json(
        args.output_dir / "prompts.json",
        [asdict(variant) for variant in variants],
    )

    all_examples: list[dict[str, Any]] = []
    dev_metrics: list[dict[str, Any]] = []
    parity_unit = selected_direction["unit"]
    for variant in variants:
        LOGGER.info("screening prompt %s baseline", variant.variant_id)
        baseline = generate_variant(
            model=model,
            tokenizer=tokenizer,
            variant=variant,
            values=selection_values,
            split=args.selection_split,
            condition="baseline",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        LOGGER.info("screening prompt %s parity steer", variant.variant_id)
        steered = generate_variant(
            model=model,
            tokenizer=tokenizer,
            variant=variant,
            values=selection_values,
            split=args.selection_split,
            condition="parity_steer",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            direction_unit=parity_unit,
            projection_std=projection_std,
            alpha=alpha,
            intervention_layer=intervention_layer,
        )
        all_examples.extend(baseline)
        all_examples.extend(steered)
        baseline_summary = summarize_completions(baseline)
        steered_summary = summarize_completions(steered)
        dev_metrics.append(
            {
                "variant_id": variant.variant_id,
                "description": variant.description,
                "n_demonstrations": len(variant.demonstration_values),
                "n": len(selection_values),
                **_prefix_metrics("baseline", baseline_summary),
                **_prefix_metrics("steered", steered_summary),
                **summarize_paired_conditions(baseline, steered),
            }
        )
    selected = select_joint_prompt(
        dev_metrics,
        minimum_baseline_joint_accuracy=args.minimum_baseline_joint_accuracy,
        minimum_joint_coverage=args.minimum_joint_coverage,
    )
    selected_variant = next(
        variant for variant in variants if variant.variant_id == selected["variant_id"]
    )
    write_csv(args.output_dir / "selection_prompt_metrics.csv", dev_metrics)
    write_json(args.output_dir / "selected_prompt.json", selected)
    LOGGER.info(
        "selected prompt %s mode=%s baseline_joint=%.3f conditional_dissociation=%.3f",
        selected_variant.variant_id,
        selected["selection_mode"],
        selected["baseline_joint_correct_rate"],
        selected["conditional_dissociation_rate"],
    )

    condition_specs = (
        ("baseline", None, 1.0),
        ("parity_steer", parity_unit, 1.0),
        ("wrong_sign", parity_unit, -1.0),
        ("random_direction", direction_payload["random_unit"], 1.0),
    )
    evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    for condition, unit, sign_multiplier in condition_specs:
        LOGGER.info("evaluating selected prompt condition %s", condition)
        evaluation_rows[condition] = generate_variant(
            model=model,
            tokenizer=tokenizer,
            variant=selected_variant,
            values=evaluation_values,
            split=args.evaluation_split,
            condition=condition,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            direction_unit=unit,
            projection_std=projection_std if unit is not None else 0.0,
            alpha=alpha if unit is not None else 0.0,
            intervention_layer=intervention_layer if unit is not None else None,
            sign_multiplier=sign_multiplier,
        )
        all_examples.extend(evaluation_rows[condition])

    baseline = evaluation_rows["baseline"]
    evaluation_metrics: list[dict[str, Any]] = []
    for condition, rows in evaluation_rows.items():
        summary = summarize_completions(rows)
        paired = (
            summarize_paired_conditions(baseline, rows)
            if condition != "baseline"
            else {
                "baseline_joint_correct_count": sum(
                    bool(row["baseline_joint_correct"]) for row in baseline
                ),
                "conditional_dissociation_count": 0,
                "conditional_dissociation_rate": 0.0,
            }
        )
        evaluation_metrics.append(
            {
                "condition": condition,
                "variant_id": selected_variant.variant_id,
                **summary,
                **paired,
            }
        )
    write_csv(args.output_dir / "evaluation_metrics.csv", evaluation_metrics)
    write_jsonl_gz(args.output_dir / "examples.jsonl.gz", all_examples)

    result = {
        "config": asdict(config),
        "value_split_hash": value_split_hash(value_splits),
        "selection_values": selection_values,
        "evaluation_values": evaluation_values,
        "selected_prompt": selected,
        "selection_prompt_metrics": dev_metrics,
        "evaluation_metrics": evaluation_metrics,
        "elapsed_seconds": time.time() - started,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    write_json(args.output_dir / "results.json", result)
    (args.output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
