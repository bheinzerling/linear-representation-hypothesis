#!/usr/bin/env python3
"""Screen models and prompt formats on balanced number-property list filtering."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introlm.experiments.number_property_list_filter import (  # noqa: E402
    PROPERTY_SPECS,
    ListExample,
    PromptVariant,
    annotate_completion,
    build_list_bank,
    build_prompt_variants,
    flatten_messages,
    prompt_messages,
    select_prompt,
    summarize,
)


LOGGER = logging.getLogger("number_property_list_filter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--properties", default="primality,divisible_by_3", help="Comma-separated IDs"
    )
    parser.add_argument(
        "--chat-template", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument("--dev-examples", type=int, default=30)
    parser.add_argument("--test-examples", type=int, default=50)
    parser.add_argument("--demo-examples", type=int, default=8)
    parser.add_argument("--list-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260718)
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


def use_chat_template(tokenizer: Any, mode: str) -> bool:
    available = bool(getattr(tokenizer, "chat_template", None))
    if mode == "always" and not available:
        raise ValueError("--chat-template always requires a tokenizer chat template")
    return available and mode != "never"


def render_prompt(
    tokenizer: Any, variant: PromptVariant, example: ListExample, chat: bool
) -> str:
    messages = prompt_messages(variant, example)
    if not chat:
        return flatten_messages(messages)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def generate_rows(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[ListExample],
    split: str,
    variant: PromptVariant,
    chat: bool,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            render_prompt(tokenizer, variant, example, chat) for example in batch
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded.pop("token_type_ids", None)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        new_ids = generated[:, encoded["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for example, prompt, completion in zip(
            batch, prompts, completions, strict=True
        ):
            rows.append(
                {
                    "model_label": None,
                    "property_id": example.property_id,
                    "split": split,
                    "variant_id": variant.variant_id,
                    "example_id": example.example_id,
                    "source_numbers": list(example.values),
                    "prompt": prompt,
                    "generated_text": completion,
                    **annotate_completion(example, completion),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (args.output_dir / "DONE").exists() and not args.overwrite:
        LOGGER.info("completed output exists: %s", args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    properties = tuple(
        item.strip() for item in args.properties.split(",") if item.strip()
    )
    unknown = set(properties).difference(PROPERTY_SPECS)
    if unknown:
        raise ValueError(f"unknown properties: {sorted(unknown)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    chat = use_chat_template(tokenizer, args.chat_template)
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

    config = {
        "model": args.model,
        "model_label": args.model_label,
        "properties": list(properties),
        "chat_template_mode": args.chat_template,
        "chat_template_used": chat,
        "dev_examples": args.dev_examples,
        "test_examples": args.test_examples,
        "demo_examples": args.demo_examples,
        "list_size": args.list_size,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    write_json(args.output_dir / "config.json", config)

    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    prompts: dict[str, Any] = {}
    for property_id in properties:
        demos = build_list_bank(
            property_id,
            seed=args.seed,
            split="demo",
            n_examples=args.demo_examples,
            list_size=args.list_size,
        )
        dev = build_list_bank(
            property_id,
            seed=args.seed,
            split="dev",
            n_examples=args.dev_examples,
            list_size=args.list_size,
        )
        test = build_list_bank(
            property_id,
            seed=args.seed,
            split="test",
            n_examples=args.test_examples,
            list_size=args.list_size,
        )
        variants = build_prompt_variants(demos)
        datasets[property_id] = {
            split: [asdict(row) for row in examples]
            for split, examples in (("demo", demos), ("dev", dev), ("test", test))
        }
        prompts[property_id] = [asdict(variant) for variant in variants]
        dev_metrics: list[dict[str, Any]] = []
        for variant in variants:
            LOGGER.info("%s dev prompt %s", property_id, variant.variant_id)
            rows = generate_rows(
                model=model,
                tokenizer=tokenizer,
                examples=dev,
                split="dev",
                variant=variant,
                chat=chat,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            for row in rows:
                row["model_label"] = args.model_label
            all_rows.extend(rows)
            metrics = {
                "model_label": args.model_label,
                "property_id": property_id,
                "variant_id": variant.variant_id,
                "description": variant.description,
                "split": "dev_screen",
                **summarize(rows),
            }
            metric_rows.append(metrics)
            dev_metrics.append(metrics)
        selected = dict(select_prompt(dev_metrics))
        selected["selected_on"] = "dev_exact_set_then_sequence_f1"
        selections.append(selected)
        selected_variant = next(
            variant
            for variant in variants
            if variant.variant_id == selected["variant_id"]
        )
        LOGGER.info(
            "%s selected %s dev_exact_set=%.4f",
            property_id,
            selected_variant.variant_id,
            selected["exact_set"],
        )
        rows = generate_rows(
            model=model,
            tokenizer=tokenizer,
            examples=test,
            split="test",
            variant=selected_variant,
            chat=chat,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        for row in rows:
            row["model_label"] = args.model_label
        all_rows.extend(rows)
        metric_rows.append(
            {
                "model_label": args.model_label,
                "property_id": property_id,
                "variant_id": selected_variant.variant_id,
                "description": selected_variant.description,
                "split": "test_selected",
                **summarize(rows),
            }
        )

    write_json(args.output_dir / "datasets.json", datasets)
    write_json(args.output_dir / "prompts.json", prompts)
    write_csv(args.output_dir / "prompt_metrics.csv", metric_rows)
    write_json(args.output_dir / "selected_prompts.json", selections)
    write_jsonl_gz(
        args.output_dir / "verbatim_prompts_and_completions.jsonl.gz", all_rows
    )
    results = {
        "config": config,
        "selected_prompts": selections,
        "test_metrics": [row for row in metric_rows if row["split"] == "test_selected"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "results.json", results)
    (args.output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
