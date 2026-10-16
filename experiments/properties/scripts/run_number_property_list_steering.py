#!/usr/bin/env python3
"""Steer selected Qwen numeral occurrences and measure list-membership transport."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import json
import logging
from pathlib import Path
import random
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

from introlm.experiments.number_property_list_filter import (  # noqa: E402
    ListExample,
    annotate_completion,
    build_prompt_variants,
    prompt_messages,
    summarize,
)
from introlm.experiments.number_property_list_steering import (  # noqa: E402
    annotate_membership_completion,
    select_strength,
    select_target_indices,
    summarize_membership,
)


LOGGER = logging.getLogger("number_property_list_steering")


@dataclass(frozen=True)
class RenderedListPrompt:
    text: str
    token_count: int
    number_positions: tuple[int, ...]
    delimiter_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    numeral_token_widths: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument(
        "--property",
        choices=(
            "parity",
            "primality",
            "divisible_by_3",
            "divisible_by_5",
            "divisible_by_10",
            "digit_count_2v3",
        ),
        required=True,
    )
    parser.add_argument("--behavior-run-dir", type=Path, required=True)
    parser.add_argument("--direction-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="2,4,8,12,16")
    parser.add_argument("--subset-sizes", default="1,2,3")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-variant", default="numbered_csv")
    parser.add_argument(
        "--probe-locus", choices=("auto", "number", "last_digit"), default="auto"
    )
    parser.add_argument("--minimum-untargeted", type=float, default=0.98)
    parser.add_argument("--minimum-source-fidelity", type=float, default=0.98)
    parser.add_argument("--minimum-behavioral-exact", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--demonstration-seed",
        type=int,
        help="Seed for ordering the behavior-run demonstration examples; defaults to --seed.",
    )
    parser.add_argument(
        "--target-seed",
        type=int,
        help="Seed selecting which eligible list positions are steered; defaults to --seed.",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def comma_numbers(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def transformer_layers(model: nn.Module) -> Any:
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
    raise ValueError("could not locate transformer layers")


def eos_ids(model: nn.Module, tokenizer: Any) -> set[int]:
    raw = getattr(model.generation_config, "eos_token_id", None)
    if raw is None:
        raw = tokenizer.eos_token_id
    if raw is None:
        return set()
    return {int(raw)} if isinstance(raw, int) else {int(value) for value in raw}


def load_examples(path: Path, property_id: str, split: str) -> tuple[ListExample, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))[property_id][split]
    return tuple(
        ListExample(
            example_id=row["example_id"],
            property_id=row["property_id"],
            values=tuple(int(value) for value in row["values"]),
            targets=tuple(int(value) for value in row["targets"]),
        )
        for row in payload
    )


def apply_chat_template(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_list_prompt(
    tokenizer: Any, variant: Any, example: ListExample
) -> RenderedListPrompt:
    text = apply_chat_template(tokenizer, prompt_messages(variant, example))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    number_positions = []
    delimiter_positions = []
    token_ids = []
    numeral_token_widths = []
    for index, value in enumerate(example.values, 1):
        marker = f"{index}. {value}"
        marker_start = text.rfind(marker)
        if marker_start < 0:
            raise ValueError(f"missing query marker {marker!r}")
        char_start = marker_start + len(f"{index}. ")
        char_end = char_start + len(str(value))
        numeral_tokens = [
            token_index
            for token_index, (left, right) in enumerate(offsets)
            if max(left, char_start) < min(right, char_end)
        ]
        observed = [offsets[token_index] for token_index in numeral_tokens]
        covered = all(
            any(left <= char_index and right >= char_index + 1 for left, right in observed)
            for char_index in range(char_start, char_end)
        )
        if not numeral_tokens or not covered:
            raise ValueError(
                f"token offsets do not cover {value} at marker {index}: {observed}"
            )
        number_position = numeral_tokens[-1]
        delimiter_position = next(
            (
                token_index
                for token_index, (left, _right) in enumerate(offsets)
                if token_index > number_position and left >= char_end
            ),
            None,
        )
        if delimiter_position is None:
            raise ValueError(f"no delimiter token follows {value}")
        number_positions.append(number_position)
        delimiter_positions.append(delimiter_position)
        token_ids.append(int(encoded["input_ids"][number_position]))
        numeral_token_widths.append(len(numeral_tokens))
    return RenderedListPrompt(
        text=text,
        token_count=len(encoded["input_ids"]),
        number_positions=tuple(number_positions),
        delimiter_positions=tuple(delimiter_positions),
        token_ids=tuple(token_ids),
        numeral_token_widths=tuple(numeral_token_widths),
    )


def encode_batch(
    *,
    tokenizer: Any,
    prompts: Sequence[RenderedListPrompt],
    target_indices: Sequence[Sequence[int]],
    location: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(
        [prompt.text for prompt in prompts],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    encoded.pop("token_type_ids", None)
    sequence_length = int(encoded["input_ids"].shape[1])
    positions = []
    for row, (prompt, indices) in enumerate(zip(prompts, target_indices, strict=True)):
        source_positions = (
            prompt.number_positions
            if location == "number"
            else prompt.delimiter_positions
        )
        left_padding = sequence_length - prompt.token_count
        row_positions = [
            left_padding + source_positions[int(index)] for index in indices
        ]
        positions.append(row_positions)
        for value_index, token_id in enumerate(prompt.token_ids):
            observed = int(
                encoded["input_ids"][
                    row, left_padding + prompt.number_positions[value_index]
                ].item()
            )
            if observed != token_id:
                raise ValueError(f"left-padding token mismatch in row {row}")
    widths = {len(row) for row in positions}
    if len(widths) != 1:
        raise ValueError("all intervention plans in a batch must have equal width")
    return (
        {key: value.to(device) for key, value in encoded.items()},
        torch.tensor(positions, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def generate_text_rows(
    *,
    model: nn.Module,
    tokenizer: Any,
    variant: Any,
    examples: Sequence[ListExample],
    target_indices: Sequence[Sequence[int]],
    split: str,
    condition: str,
    layer: int,
    alpha: float,
    direction_payload: dict[str, Any] | None,
    sign: float,
    location: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    layers = transformer_layers(model)
    stop_ids = eos_ids(model, tokenizer)
    result_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_targets = target_indices[start : start + batch_size]
        prompts = [
            render_list_prompt(tokenizer, variant, example)
            for example in batch_examples
        ]
        encoded, positions = encode_batch(
            tokenizer=tokenizer,
            prompts=prompts,
            target_indices=batch_targets,
            location=location,
            device=device,
        )
        handle = None
        delta_norm = 0.0
        if direction_payload is not None:
            unit = direction_payload["unit"].to(device=device, dtype=torch.float32)
            delta = (
                unit
                * float(direction_payload["projection_std"])
                * float(alpha)
                * float(sign)
            )
            delta_norm = float(delta.norm().item())

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                rows = torch.arange(len(batch_examples), device=hidden.device)[:, None]
                patched[rows, positions] = (
                    patched[rows, positions].float() + delta
                ).to(hidden.dtype)
                return (patched, *output[1:]) if isinstance(output, tuple) else patched

            handle = layers[int(layer)].register_forward_hook(hook)
        try:
            outputs = model(**encoded, use_cache=True)
        finally:
            if handle is not None:
                handle.remove()

        attention_mask = encoded["attention_mask"]
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1]
        finished = torch.zeros(len(batch_examples), dtype=torch.bool, device=device)
        generated = []
        for step in range(max_new_tokens):
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
        for example, prompt, indices, completion in zip(
            batch_examples, prompts, batch_targets, completions, strict=True
        ):
            result_rows.append(
                {
                    "example_id": example.example_id,
                    "property_id": example.property_id,
                    "split": split,
                    "condition": condition,
                    "source_numbers": list(example.values),
                    "prompt": prompt.text,
                    "generated_text": completion,
                    "target_indices": list(indices),
                    "layer": int(layer),
                    "alpha": float(alpha),
                    "location": location,
                    "sign": float(sign),
                    "per_target_delta_norm": delta_norm,
                }
            )
    return result_rows


def annotate_rows(
    *,
    rows: Sequence[dict[str, Any]],
    examples: Sequence[ListExample],
    target_indices: Sequence[Sequence[int]],
    source_class: int,
) -> list[dict[str, Any]]:
    if len(rows) != len(examples) or len(rows) != len(target_indices):
        raise ValueError("rows, examples, and target plans must align")
    output = []
    for row, example, indices in zip(rows, examples, target_indices, strict=True):
        if row["example_id"] != example.example_id:
            raise ValueError("example alignment mismatch")
        output.append(
            {
                **row,
                **annotate_membership_completion(
                    example=example,
                    text=row["generated_text"],
                    target_indices=indices,
                    source_class=source_class,
                ),
            }
        )
    return output


def target_plans(
    examples: Sequence[ListExample], *, source_class: int, count: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        select_target_indices(
            example, source_class=source_class, count=count, seed=seed
        )
        for example in examples
    )


def baseline_metrics(
    rows: Sequence[dict[str, Any]], examples: Sequence[ListExample]
) -> dict[str, Any]:
    annotated = [
        annotate_completion(example, row["generated_text"])
        for row, example in zip(rows, examples, strict=True)
    ]
    return summarize(annotated)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    alphas = comma_numbers(args.alphas, float)
    subset_sizes = comma_numbers(args.subset_sizes, int)
    demonstration_seed = (
        args.seed if args.demonstration_seed is None else args.demonstration_seed
    )
    target_seed = args.seed if args.target_seed is None else args.target_seed
    if any(size < 1 or size > 5 for size in subset_sizes):
        raise ValueError("subset sizes must be between 1 and 5")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (args.output_dir / "DONE").exists() and not args.overwrite:
        LOGGER.info("completed output exists: %s", args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    demos = list(load_examples(
        args.behavior_run_dir / "datasets.json", args.property, "demo"
    ))
    random.Random(
        f"list-demonstrations:{demonstration_seed}:{args.property}"
    ).shuffle(demos)
    dev = load_examples(args.behavior_run_dir / "datasets.json", args.property, "dev")
    test = load_examples(args.behavior_run_dir / "datasets.json", args.property, "test")
    variants = build_prompt_variants(demos)
    try:
        variant = next(
            item for item in variants if item.variant_id == args.prompt_variant
        )
    except StopIteration as error:
        available = ", ".join(item.variant_id for item in variants)
        raise ValueError(
            f"unknown prompt variant {args.prompt_variant!r}; available: {available}"
        ) from error
    if not variant.numbered_input:
        raise ValueError("list steering requires a numbered-input prompt variant")
    selected_condition = json.loads(
        (args.direction_run_dir / "selected_condition.json").read_text(encoding="utf-8")
    )
    recorded_locus = selected_condition.get("locus")
    probe_locus = (
        str(recorded_locus)
        if args.probe_locus == "auto" and recorded_locus
        else ("number" if args.probe_locus == "auto" else args.probe_locus)
    )
    if probe_locus not in {"number", "last_digit"}:
        raise ValueError(f"unsupported list-steering probe locus: {probe_locus}")
    layer = int(selected_condition["layer"])
    direction_bundle = torch.load(
        args.direction_run_dir / "selected_and_control_directions.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected_directions = direction_bundle["selected"]
    random_directions = direction_bundle["random"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    validation_prompts = [render_list_prompt(tokenizer, variant, dev[0])]
    if probe_locus == "number" and any(
        width != 1 for width in validation_prompts[0].numeral_token_widths
    ):
        raise ValueError(
            "number probe locus requires every query numeral to be a single token; "
            f"observed widths={validation_prompts[0].numeral_token_widths}"
        )

    config = {
        "model": args.model,
        "property": args.property,
        "behavior_run_dir": str(args.behavior_run_dir),
        "direction_run_dir": str(args.direction_run_dir),
        "output_dir": str(args.output_dir),
        "alphas": list(alphas),
        "subset_sizes": list(subset_sizes),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "minimum_untargeted": args.minimum_untargeted,
        "minimum_source_fidelity": args.minimum_source_fidelity,
        "minimum_behavioral_exact": args.minimum_behavioral_exact,
        "seed": args.seed,
        "demonstration_seed": demonstration_seed,
        "target_seed": target_seed,
        "dtype": args.dtype,
        "layer": layer,
        "locus": probe_locus,
        "prompt_variant": variant.variant_id,
        "chat_template": True,
        "enable_thinking": False,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    write_json(args.output_dir / "config.json", config)

    empty_dev_plans = tuple(() for _ in dev)
    empty_test_plans = tuple(() for _ in test)
    LOGGER.info("generating unsteered baselines")
    dev_baseline = generate_text_rows(
        model=model,
        tokenizer=tokenizer,
        variant=variant,
        examples=dev,
        target_indices=empty_dev_plans,
        split="dev",
        condition="no_steer",
        layer=layer,
        alpha=0,
        direction_payload=None,
        sign=1,
        location="number",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    test_baseline = generate_text_rows(
        model=model,
        tokenizer=tokenizer,
        variant=variant,
        examples=test,
        target_indices=empty_test_plans,
        split="test",
        condition="no_steer",
        layer=layer,
        alpha=0,
        direction_payload=None,
        sign=1,
        location="number",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    baseline_summary = {
        "dev": baseline_metrics(dev_baseline, dev),
        "test": baseline_metrics(test_baseline, test),
    }
    write_json(args.output_dir / "baseline_metrics.json", baseline_summary)
    failing_splits = [
        split
        for split, metrics in baseline_summary.items()
        if float(metrics["exact_set"]) < args.minimum_behavioral_exact
    ]
    if failing_splits:
        gate_result = {
            "config": config,
            "baseline_metrics": baseline_summary,
            "behavioral_gate_passed": False,
            "failing_splits": failing_splits,
            "elapsed_seconds": time.time() - started,
        }
        write_json(args.output_dir / "results.json", gate_result)
        (args.output_dir / "GATE_FAILED").write_text("behavioral exact-set gate failed\n")
        print(json.dumps(gate_result, indent=2, sort_keys=True))
        return

    dev_metric_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for source_class in (0, 1):
        plans = target_plans(dev, source_class=source_class, count=1, seed=target_seed)
        candidate_metrics = []
        for alpha in alphas:
            LOGGER.info("dev source=%d alpha=%g", source_class, alpha)
            generated = generate_text_rows(
                model=model,
                tokenizer=tokenizer,
                variant=variant,
                examples=dev,
                target_indices=plans,
                split="dev",
                condition="selected_direction",
                layer=layer,
                alpha=alpha,
                direction_payload=selected_directions[source_class],
                sign=1,
                location="number",
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            annotated = annotate_rows(
                rows=generated,
                examples=dev,
                target_indices=plans,
                source_class=source_class,
            )
            all_rows.extend(annotated)
            metrics = {
                "property_id": args.property,
                "source_class": source_class,
                "direction": f"{source_class}_to_{1 - source_class}",
                "subset_size": 1,
                "condition": "selected_direction",
                "alpha": alpha,
                **summarize_membership(annotated),
            }
            candidate_metrics.append(metrics)
            dev_metric_rows.append(metrics)
        selected = select_strength(
            candidate_metrics,
            minimum_untargeted=args.minimum_untargeted,
            minimum_source_fidelity=args.minimum_source_fidelity,
        )
        selections.append(selected)
        LOGGER.info(
            "selected source=%d alpha=%g transport=%.3f exact=%.3f",
            source_class,
            selected["alpha"],
            selected["target_transport_rate"],
            selected["exact_counterfactual_set"],
        )

    test_metric_rows: list[dict[str, Any]] = []
    selected_by_source = {int(row["source_class"]): row for row in selections}
    for source_class in (0, 1):
        alpha = float(selected_by_source[source_class]["alpha"])
        for subset_size in subset_sizes:
            plans = target_plans(
                test,
                source_class=source_class,
                count=subset_size,
                seed=target_seed,
            )
            baseline_annotated = annotate_rows(
                rows=test_baseline,
                examples=test,
                target_indices=plans,
                source_class=source_class,
            )
            for row in baseline_annotated:
                row.update(
                    {
                        "condition": "no_steer",
                        "source_class": source_class,
                        "subset_size": subset_size,
                    }
                )
            all_rows.extend(baseline_annotated)
            condition_specs = (
                (
                    "selected_direction",
                    selected_directions[source_class],
                    1.0,
                    "number",
                ),
                (
                    "wrong_sign",
                    selected_directions[source_class],
                    -1.0,
                    "number",
                ),
                (
                    "random_direction",
                    random_directions[source_class],
                    1.0,
                    "number",
                ),
                (
                    "delimiter_location",
                    selected_directions[source_class],
                    1.0,
                    "delimiter",
                ),
            )
            condition_rows = [("no_steer", baseline_annotated)]
            for condition, payload, sign, location in condition_specs:
                LOGGER.info(
                    "test source=%d k=%d condition=%s alpha=%g",
                    source_class,
                    subset_size,
                    condition,
                    alpha,
                )
                generated = generate_text_rows(
                    model=model,
                    tokenizer=tokenizer,
                    variant=variant,
                    examples=test,
                    target_indices=plans,
                    split="test",
                    condition=condition,
                    layer=layer,
                    alpha=alpha,
                    direction_payload=payload,
                    sign=sign,
                    location=location,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                annotated = annotate_rows(
                    rows=generated,
                    examples=test,
                    target_indices=plans,
                    source_class=source_class,
                )
                all_rows.extend(annotated)
                condition_rows.append((condition, annotated))
            for condition, rows in condition_rows:
                test_metric_rows.append(
                    {
                        "property_id": args.property,
                        "source_class": source_class,
                        "direction": f"{source_class}_to_{1 - source_class}",
                        "subset_size": subset_size,
                        "condition": condition,
                        "alpha": 0.0 if condition == "no_steer" else alpha,
                        **summarize_membership(rows),
                    }
                )

    write_csv(args.output_dir / "dev_strength_metrics.csv", dev_metric_rows)
    write_json(args.output_dir / "selected_strengths.json", selections)
    write_csv(args.output_dir / "test_membership_metrics.csv", test_metric_rows)
    write_jsonl_gz(args.output_dir / "examples.jsonl.gz", all_rows)
    results = {
        "config": config,
        "baseline_metrics": baseline_summary,
        "behavioral_gate_passed": True,
        "selected_strengths": selections,
        "test_metrics": test_metric_rows,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "results.json", results)
    (args.output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
