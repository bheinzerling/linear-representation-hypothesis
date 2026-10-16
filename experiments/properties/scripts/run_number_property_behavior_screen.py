#!/usr/bin/env python3
"""Screen prompt formats and model competence for numerical properties."""

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

from introlm.experiments.number_property_behavior_screen import (  # noqa: E402
    PROPERTY_SPECS,
    PromptVariant,
    annotate_completion,
    build_prompt_variants,
    flatten_messages,
    prompt_messages,
    select_prompt,
    summarize,
    value_splits,
)


LOGGER = logging.getLogger("number_property_behavior_screen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--properties", default="divisible_by_3,primality", help="Comma-separated IDs"
    )
    parser.add_argument(
        "--chat-template",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--minimum-joint-coverage", type=float, default=0.95)
    parser.add_argument("--competence-gate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def render_prompt(tokenizer: Any, variant: PromptVariant, value: int, chat: bool) -> str:
    messages = prompt_messages(variant, value)
    if chat:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return flatten_messages(messages)


@torch.inference_mode()
def generate_rows(
    *,
    model: Any,
    tokenizer: Any,
    values: Sequence[int],
    split: str,
    variant: PromptVariant,
    chat: bool,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), batch_size):
        batch_values = [int(value) for value in values[start : start + batch_size]]
        prompts = [render_prompt(tokenizer, variant, value, chat) for value in batch_values]
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
        for value, prompt, completion in zip(batch_values, prompts, completions, strict=True):
            rows.append(
                {
                    "property_id": variant.property_id,
                    "split": split,
                    "variant_id": variant.variant_id,
                    "value": value,
                    "prompt": prompt,
                    "generated_text": completion,
                    **annotate_completion(value=value, text=completion, variant=variant),
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

    properties = tuple(item.strip() for item in args.properties.split(",") if item.strip())
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
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "minimum_joint_coverage": args.minimum_joint_coverage,
        "competence_gate": args.competence_gate,
        "seed": args.seed,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    write_json(args.output_dir / "config.json", config)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    split_manifest: dict[str, Any] = {}
    prompt_manifest: dict[str, Any] = {}
    for property_id in properties:
        splits = value_splits(property_id, args.seed)
        variants = build_prompt_variants(splits["train"], property_id)
        split_manifest[property_id] = splits
        prompt_manifest[property_id] = [asdict(variant) for variant in variants]
        dev_metrics: list[dict[str, Any]] = []
        for variant in variants:
            LOGGER.info("%s dev prompt %s", property_id, variant.variant_id)
            rows = generate_rows(
                model=model,
                tokenizer=tokenizer,
                values=splits["dev"],
                split="dev",
                variant=variant,
                chat=chat,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            all_rows.extend(rows)
            dev_metrics.append(
                {
                    "model_label": args.model_label,
                    "property_id": property_id,
                    "variant_id": variant.variant_id,
                    "description": variant.description,
                    "n_demonstrations": len(variant.demonstration_values),
                    "label_style": variant.label_style,
                    "format_id": variant.format_id,
                    **summarize(rows),
                }
            )
        selected = select_prompt(dev_metrics, args.minimum_joint_coverage)
        selected_rows.append(selected)
        selected_variant = next(
            variant for variant in variants if variant.variant_id == selected["variant_id"]
        )
        LOGGER.info(
            "%s selected %s dev_joint=%.4f",
            property_id,
            selected_variant.variant_id,
            selected["joint_accuracy"],
        )
        test_rows = generate_rows(
            model=model,
            tokenizer=tokenizer,
            values=splits["test"],
            split="test",
            variant=selected_variant,
            chat=chat,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        all_rows.extend(test_rows)
        summaries.extend(dev_metrics)
        summaries.append(
            {
                "model_label": args.model_label,
                "property_id": property_id,
                "variant_id": selected_variant.variant_id,
                "description": selected_variant.description,
                "n_demonstrations": len(selected_variant.demonstration_values),
                "label_style": selected_variant.label_style,
                "format_id": selected_variant.format_id,
                "split": "test_selected",
                **summarize(test_rows),
            }
        )

    for row in summaries:
        row.setdefault("split", "dev_screen")
        row["passes_competence_gate"] = (
            float(row["joint_accuracy"]) >= args.competence_gate
            and float(row["joint_coverage"]) >= args.minimum_joint_coverage
        )
    for row in selected_rows:
        row["dev_passes_competence_gate"] = (
            float(row["joint_accuracy"]) >= args.competence_gate
            and float(row["joint_coverage"]) >= args.minimum_joint_coverage
        )

    write_json(args.output_dir / "splits.json", split_manifest)
    write_json(args.output_dir / "prompts.json", prompt_manifest)
    write_csv(args.output_dir / "prompt_metrics.csv", summaries)
    write_json(args.output_dir / "selected_prompts.json", selected_rows)
    write_jsonl_gz(args.output_dir / "verbatim_prompts_and_completions.jsonl.gz", all_rows)
    results = {
        "config": config,
        "selected_prompts": selected_rows,
        "test_metrics": [row for row in summaries if row["split"] == "test_selected"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "results.json", results)
    (args.output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
