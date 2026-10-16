from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.token_utils import encode_integer, render_prompt


ArithmeticTask = Literal["addition", "subtraction", "multiplication"]


@dataclass(frozen=True)
class ArithmeticExample:
    task: ArithmeticTask
    x1: int
    x2: int
    target: int


def generate_addition_examples() -> list[ArithmeticExample]:
    examples: list[ArithmeticExample] = []
    for x1 in range(1, 500):
        for x2 in range(1, 500):
            target = x1 + x2
            if 0 <= target <= 999:
                examples.append(ArithmeticExample(task="addition", x1=x1, x2=x2, target=target))
    return examples


def generate_subtraction_examples() -> list[ArithmeticExample]:
    examples: list[ArithmeticExample] = []
    for x1 in range(2, 1000):
        for x2 in range(1, x1):
            target = x1 - x2
            if 1 <= target <= 998:
                examples.append(ArithmeticExample(task="subtraction", x1=x1, x2=x2, target=target))
    return examples


def generate_multiplication_examples(domain_cfg: dict[str, int]) -> list[ArithmeticExample]:
    examples: list[ArithmeticExample] = []
    for x1 in range(domain_cfg["x1_min"], domain_cfg["x1_max"] + 1):
        for x2 in range(domain_cfg["x2_min"], domain_cfg["x2_max"] + 1):
            target = x1 * x2
            if 0 <= target <= 999:
                examples.append(ArithmeticExample(task="multiplication", x1=x1, x2=x2, target=target))
    return examples


def filter_single_token_targets(tokenizer, examples: Iterable[ArithmeticExample]) -> list[ArithmeticExample]:
    filtered: list[ArithmeticExample] = []
    for ex in examples:
        if len(encode_integer(tokenizer, ex.target)) == 1:
            filtered.append(ex)
    return filtered


def prepare_prompt_sweep_models(selected_models: list[str]) -> list[str]:
    one_b = [model for model in selected_models if "1B" in model or "1b" in model]
    return one_b or selected_models[:1]


def batch_next_token_predictions(
    model,
    tokenizer,
    prompts: list[str],
    target_token_ids: list[int],
    batch_size: int = 128,
    device: str = "cpu",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="next-token", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        batch_targets = target_token_ids[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            last_positions = attention_mask.sum(dim=1) - 1
            next_token_logits = logits[torch.arange(logits.shape[0], device=device), last_positions]
            probs = torch.softmax(next_token_logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            pred_probs = probs[torch.arange(probs.shape[0], device=device), pred_ids]
            gold_probs = probs[torch.arange(probs.shape[0], device=device), torch.tensor(batch_targets, device=device)]
        for pred_id, pred_prob, gold_prob, gold_id in zip(
            pred_ids.cpu().tolist(),
            pred_probs.cpu().tolist(),
            gold_probs.cpu().tolist(),
            batch_targets,
        ):
            rows.append(
                {
                    "predicted_token_id": int(pred_id),
                    "predicted_text": tokenizer.decode([pred_id], clean_up_tokenization_spaces=False),
                    "correct": int(pred_id == gold_id),
                    "p_pred": float(pred_prob),
                    "p_gold": float(gold_prob),
                }
            )
    return pd.DataFrame(rows)


def evaluate_prompt_template(
    model,
    tokenizer,
    examples: list[ArithmeticExample],
    template: str,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    prompts = [render_prompt(template, ex.x1, ex.x2) for ex in examples]
    target_token_ids = [encode_integer(tokenizer, ex.target)[0] for ex in examples]
    predictions = batch_next_token_predictions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        target_token_ids=target_token_ids,
        batch_size=batch_size,
        device=device,
    )
    meta = pd.DataFrame(
        {
            "task": [ex.task for ex in examples],
            "x1": [ex.x1 for ex in examples],
            "x2": [ex.x2 for ex in examples],
            "target": [ex.target for ex in examples],
            "template": template,
            "target_token_id": target_token_ids,
            "prompt": prompts,
        }
    )
    return pd.concat([meta, predictions], axis=1)


def calibrate_prompt_templates(
    model,
    tokenizer,
    task: ArithmeticTask,
    templates: list[str],
    examples: list[ArithmeticExample],
    batch_size: int,
    calibration_examples: int,
    device: str,
    seed: int,
) -> tuple[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if len(examples) > calibration_examples:
        keep = np.sort(rng.choice(len(examples), size=calibration_examples, replace=False))
        examples = [examples[idx] for idx in keep.tolist()]

    sweep_rows: list[dict[str, Any]] = []
    for template in templates:
        results = evaluate_prompt_template(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            template=template,
            batch_size=batch_size,
            device=device,
        )
        sweep_rows.append(
            {
                "task": task,
                "template": template,
                "num_examples": len(results),
                "accuracy": float(results["correct"].mean()),
                "error_rate_pct": float((1.0 - results["correct"].mean()) * 100.0),
            }
        )
    sweep = pd.DataFrame(sweep_rows).sort_values(["accuracy", "template"], ascending=[False, True]).reset_index(drop=True)
    return str(sweep.iloc[0]["template"]), sweep


def evaluate_arithmetic_task(
    model,
    tokenizer,
    task: ArithmeticTask,
    template: str,
    batch_size: int,
    device: str,
    multiplication_domain: dict[str, int] | None = None,
) -> pd.DataFrame:
    if task == "addition":
        examples = generate_addition_examples()
    elif task == "subtraction":
        examples = generate_subtraction_examples()
    elif task == "multiplication":
        if multiplication_domain is None:
            raise ValueError("multiplication_domain is required for multiplication.")
        examples = generate_multiplication_examples(multiplication_domain)
    else:
        raise ValueError(f"Unknown arithmetic task: {task}")

    examples = filter_single_token_targets(tokenizer, examples)
    return evaluate_prompt_template(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        template=template,
        batch_size=batch_size,
        device=device,
    )


def summarize_arithmetic_results(results: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return (
        results.groupby(["task", "template"], as_index=False)
        .agg(
            num_examples=("correct", "size"),
            accuracy=("correct", "mean"),
        )
        .assign(model=model_name, error_rate_pct=lambda df: (1.0 - df["accuracy"]) * 100.0)
        .sort_values(["task", "template"])
        .reset_index(drop=True)
    )
