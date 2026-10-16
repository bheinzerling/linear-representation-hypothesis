#!/usr/bin/env python3
"""Screen chat prompts for direct parity classification with digit tokenizers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
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

from introlm.experiments.number_property_steering import (  # noqa: E402
    balanced_value_splits,
    range_spanning_demonstrations,
    split_hash,
)


LOGGER = logging.getLogger("digit_parity_prompt_screen")
LABEL_RE = re.compile(r"(?<![a-z0-9])(even|odd)(?![a-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class PromptVariant:
    variant_id: str
    description: str
    messages: tuple[dict[str, str], ...]
    demonstration_values: tuple[int, ...]

    def render(self, value: int) -> list[dict[str, str]]:
        marker = "{number}"
        count = sum(message["content"].count(marker) for message in self.messages)
        if count != 1:
            raise ValueError(f"{self.variant_id} has {count} query markers")
        return [
            {"role": message["role"], "content": message["content"].format(number=value)}
            for message in self.messages
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--stop-after-good", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parity_word(value: int) -> str:
    return "even" if int(value) % 2 == 0 else "odd"


def digit_demonstrations(train_values: Sequence[int]) -> tuple[int, ...]:
    train = {int(value) for value in train_values}
    return tuple(
        min(value for value in train if value >= 100 and value % 10 == digit)
        for digit in range(10)
    )


def build_prompt_variants(train_values: Sequence[int]) -> tuple[PromptVariant, ...]:
    digit_demos = digit_demonstrations(train_values)
    rule = (
        "A whole number is even when its last decimal digit is 0, 2, 4, 6, or 8. "
        "It is odd when its last decimal digit is 1, 3, 5, 7, or 9."
    )
    system = f"Report parity using exactly one of these labels: even, odd. {rule}"

    def range_multiturn(per_class: int) -> tuple[tuple[dict[str, str], ...], tuple[int, ...]]:
        demonstrations = range_spanning_demonstrations(
            train_values, "parity", per_class=per_class
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for demonstration in demonstrations:
            messages.extend(
                [
                    {"role": "user", "content": f"Input: {demonstration}"},
                    {
                        "role": "assistant",
                        "content": f"Parity: {parity_word(demonstration)}",
                    },
                ]
            )
        messages.append({"role": "user", "content": "Input: {number}"})
        return tuple(messages), demonstrations

    multiturn_2, range_demos_2 = range_multiturn(1)
    multiturn_4, range_demos_4 = range_multiturn(2)
    multiturn_8, range_demos_8 = range_multiturn(4)
    multiturn_10: list[dict[str, str]] = [{"role": "system", "content": system}]
    for demonstration in digit_demos:
        multiturn_10.extend(
            [
                {"role": "user", "content": f"Integer: {demonstration}\nParity:"},
                {"role": "assistant", "content": parity_word(demonstration)},
            ]
        )
    multiturn_10.append({"role": "user", "content": "Integer: {number}\nParity:"})
    classification_lines = "\n".join(
        f"{value}: {parity_word(value)}" for value in digit_demos
    )
    suffix_lines = "\n".join(
        f"{value} ends in {value % 10}, so its parity is {parity_word(value)}."
        for value in digit_demos
    )
    modulo_lines = "\n".join(
        f"{value} modulo 2 is {value % 2}; answer: {parity_word(value)}."
        for value in digit_demos
    )
    return (
        PromptVariant(
            "chat_multiturn_range_2shot",
            "Rule plus range-spanning even and odd demonstration turns.",
            multiturn_2,
            range_demos_2,
        ),
        PromptVariant(
            "chat_multiturn_range_4shot",
            "Rule plus four range-spanning alternating demonstration turns.",
            multiturn_4,
            range_demos_4,
        ),
        PromptVariant(
            "chat_multiturn_range_8shot",
            "Rule plus eight range-spanning alternating demonstration turns.",
            multiturn_8,
            range_demos_8,
        ),
        PromptVariant(
            "explicit_digit_rule",
            "Explicit final-digit rule with an answer-only constraint.",
            (
                {
                    "role": "user",
                    "content": f"{rule}\n\nClassify the integer {{number}}. Reply only: even or odd.",
                },
            ),
            (),
        ),
        PromptVariant(
            "chat_multiturn_digit_10shot",
            "Rule plus one separate demonstration turn for every final digit.",
            tuple(multiturn_10),
            digit_demos,
        ),
        PromptVariant(
            "singleturn_digit_10shot",
            "One-message classification table covering every final digit.",
            (
                {
                    "role": "user",
                    "content": (
                        f"{rule}\nExamples:\n{classification_lines}\n"
                        "{number}:\nReply with exactly one word: even or odd."
                    ),
                },
            ),
            digit_demos,
        ),
        PromptVariant(
            "suffix_reasoning_digit_10shot",
            "Worked examples explicitly connecting the suffix digit to parity.",
            (
                {
                    "role": "user",
                    "content": (
                        f"{rule}\nWorked examples:\n{suffix_lines}\n"
                        "Use the last digit to classify {number}. Reply only: even or odd."
                    ),
                },
            ),
            digit_demos,
        ),
        PromptVariant(
            "modulo_digit_10shot",
            "Worked modulo-two examples covering every final digit.",
            (
                {
                    "role": "user",
                    "content": (
                        f"Determine parity using modulo 2.\n{modulo_lines}\n"
                        "Classify {number}. Reply only: even or odd."
                    ),
                },
            ),
            digit_demos,
        ),
    )


def parse_label(text: str) -> str | None:
    match = LABEL_RE.search(text)
    return None if match is None else match.group(1).lower()


def is_label_only(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(even|odd)\s*[.!]?\s*", text, re.IGNORECASE))


def render_chat(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def validate_digit_tokenization(tokenizer: Any) -> dict[str, Any]:
    examples: dict[str, list[int]] = {}
    for value in range(1000):
        numeral = str(value)
        text = f"Integer: {numeral}"
        start = text.rindex(numeral)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        indices = [
            index
            for index, (left, right) in enumerate(encoded["offset_mapping"])
            if max(int(left), start) < min(int(right), start + len(numeral))
        ]
        offsets = [tuple(map(int, encoded["offset_mapping"][index])) for index in indices]
        expected = [(start + index, start + index + 1) for index in range(len(numeral))]
        if offsets != expected:
            raise ValueError(
                f"{value} is not one-token-per-digit: indices={indices}, offsets={offsets}"
            )
        if value in {0, 19, 419, 999}:
            examples[numeral] = [int(encoded["input_ids"][index]) for index in indices]
    return {"validated_values": 1000, "examples": examples}


def load_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    errors = []
    classes = [transformers.AutoModelForCausalLM]
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        candidate = getattr(transformers, name, None)
        if candidate is not None and candidate not in classes:
            classes.append(candidate)
    for model_class in classes:
        try:
            model = model_class.from_pretrained(
                args.model,
                dtype=torch.bfloat16,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                device_map="auto",
            )
            model.eval()
            return tokenizer, model, model_class.__name__
        except (KeyError, ValueError) as error:
            errors.append(f"{model_class.__name__}: {error}")
    raise RuntimeError("no compatible auto model class\n" + "\n".join(errors))


@torch.inference_mode()
def evaluate_variant(
    model: Any,
    tokenizer: Any,
    variant: PromptVariant,
    values: Sequence[int],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), batch_size):
        batch_values = [int(value) for value in values[start : start + batch_size]]
        prompts = [render_chat(tokenizer, variant.render(value)) for value in batch_values]
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
        completions = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for value, prompt, completion in zip(batch_values, prompts, completions, strict=True):
            prediction = parse_label(completion)
            gold = parity_word(value)
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "value": value,
                    "gold": gold,
                    "prediction": prediction or "",
                    "answer_present": prediction is not None,
                    "correct": prediction == gold,
                    "label_only": is_label_only(completion),
                    "prompt": prompt,
                    "generated_text": completion,
                }
            )
    return rows


def summarize(variant: PromptVariant, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "variant_id": variant.variant_id,
        "description": variant.description,
        "n_demonstrations": len(variant.demonstration_values),
        "n": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "coverage": sum(bool(row["answer_present"]) for row in rows) / len(rows),
        "label_only_rate": sum(bool(row["label_only"]) for row in rows) / len(rows),
        "even_prediction_rate": sum(row["prediction"] == "even" for row in rows)
        / len(rows),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if (args.output_dir / "DONE").exists() and not args.overwrite:
        LOGGER.info("completed output exists: %s", args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    splits = balanced_value_splits("parity", args.seed)
    variants = build_prompt_variants(splits["train"])
    tokenizer, model, model_class = load_model(args)
    tokenization = validate_digit_tokenization(tokenizer)
    metrics = []
    examples = []
    good = []
    for variant in variants:
        LOGGER.info("evaluating %s", variant.variant_id)
        rows = evaluate_variant(
            model,
            tokenizer,
            variant,
            splits["dev"],
            args.batch_size,
            args.max_new_tokens,
        )
        summary = summarize(variant, rows)
        metrics.append(summary)
        examples.extend(rows)
        if summary["accuracy"] > args.threshold:
            good.append(variant.variant_id)
        LOGGER.info("%s accuracy=%.3f", variant.variant_id, summary["accuracy"])
        if len(good) >= args.stop_after_good:
            break
    payload = {
        "model": args.model,
        "model_label": args.model_label,
        "model_class": model_class,
        "seed": args.seed,
        "split_hash": split_hash(splits),
        "evaluated_split": "dev",
        "dev_values": splits["dev"],
        "threshold_strictly_greater_than": args.threshold,
        "stop_after_good": args.stop_after_good,
        "good_variants": good,
        "tokenization": tokenization,
        "metrics": metrics,
        "variants": [asdict(variant) for variant in variants],
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "results.json", payload)
    write_json(args.output_dir / "examples.json", examples)
    (args.output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
