from __future__ import annotations

import argparse
import copy
import gc
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from transformers import GenerationConfig

from src.model_loader import load_causal_lm
from src.utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    dump_run_config,
    ensure_dir,
    load_yaml,
    maybe_resume,
    set_global_seed,
)


OutputKind = Literal["int", "bool", "string", "hex"]


@dataclass(frozen=True)
class PromptVariant:
    variant_id: str
    template: str


@dataclass(frozen=True)
class PromptStyle:
    style_id: str
    label: str
    output_kind: OutputKind
    target: Callable[[int], Any]
    variants: tuple[PromptVariant, ...]
    max_new_tokens: int = 8
    mae_kind: Literal["numeric", "none"] = "numeric"


DISTRACTORS = (137, 486, 812)
MODEL_LABELS = {
    "llama_3_2_1b": "Llama 3.2 1B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "llama_3_8b": "Llama 3 8B",
    "olmo_2_13b": "OLMo-2 13B",
    "olmo_3_7b_base": "OLMo-3 7B",
    "gpt_oss_20b": "gpt-oss-20b",
}


def _bool_text(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _rank_among_distractors(value: int) -> int:
    return 1 + sum(int(other) < int(value) for other in DISTRACTORS)


def _nearest_landmark(value: int) -> int:
    landmarks = (0, 500, 999)
    return min(landmarks, key=lambda anchor: (abs(anchor - int(value)), anchor))


def _round_nearest_hundred(value: int) -> int:
    return int(((int(value) + 50) // 100) * 100)


def _make_prompt_styles() -> tuple[PromptStyle, ...]:
    return (
        PromptStyle(
            style_id="magnitude_comparison",
            label="Magnitude Comparison",
            output_kind="bool",
            target=lambda v: int(v) > 500,
            mae_kind="none",
            variants=(
                PromptVariant("greater_than_500", "Is {value} greater than 500? Answer yes or no."),
                PromptVariant("above_threshold", "Does {value} exceed 500? Answer yes or no."),
                PromptVariant("larger_than_threshold", "Compared with 500, is {value} larger? Answer yes or no."),
            ),
        ),
        PromptStyle(
            style_id="interval_bucket",
            label="Interval Bucket",
            output_kind="int",
            target=lambda v: int(v) // 100,
            variants=(
                PromptVariant("bucket_index", "Which 100-wide bucket contains {value}? Use bucket 0 for 0-99, 1 for 100-199, ..., 9 for 900-999. Answer with the bucket number."),
                PromptVariant("range_number", "{value} falls in which range number: 0=0-99, 1=100-199, ..., 9=900-999? Answer with one digit."),
                PromptVariant("hundreds_band", "Classify {value} by hundreds band from 0 through 9. Answer only the band number."),
            ),
        ),
        PromptStyle(
            style_id="number_line_position",
            label="Number-Line Position",
            output_kind="int",
            target=_nearest_landmark,
            variants=(
                PromptVariant("closest_anchor", "Which landmark is {value} closest to: 0, 500, or 999? Answer with the landmark number."),
                PromptVariant("nearest_point", "On a 0 to 999 number line, is {value} nearest to 0, 500, or 999? Answer only 0, 500, or 999."),
                PromptVariant("closest_reference", "Choose the closest reference value for {value}: 0, 500, 999. Answer with the reference value."),
            ),
        ),
        PromptStyle(
            style_id="rank_among_distractors",
            label="Rank Among Distractors",
            output_kind="int",
            target=_rank_among_distractors,
            variants=(
                PromptVariant("rank_in_set", "In ascending order, what rank does {value} have among {value}, 137, 486, and 812? Answer 1, 2, 3, or 4."),
                PromptVariant("smallest_to_largest_rank", "Sort these numbers from smallest to largest: 137, {value}, 812, 486. What is the rank of {value}? Answer with a number from 1 to 4."),
                PromptVariant("position_after_sort", "After sorting 812, 137, {value}, 486 in increasing order, which position is {value}? Answer only 1, 2, 3, or 4."),
            ),
        ),
        PromptStyle(
            style_id="hundreds_digit",
            label="Hundreds Digit",
            output_kind="int",
            target=lambda v: int(v) // 100,
            variants=(
                PromptVariant("hundreds_digit", "What is the hundreds digit of {value}? Answer with one digit."),
                PromptVariant("digit_in_hundreds_place", "In the number {value}, which digit is in the hundreds place? Answer only the digit."),
                PromptVariant("hundreds_place_value", "For {value}, give the hundreds-place digit. Answer with a single digit."),
            ),
        ),
        PromptStyle(
            style_id="modulo_10",
            label="Modulo 10",
            output_kind="int",
            target=lambda v: int(v) % 10,
            variants=(
                PromptVariant("remainder_mod_10", "What is {value} modulo 10? Answer with one digit."),
                PromptVariant("remainder_after_dividing_by_10", "What remainder is left when {value} is divided by 10? Answer only the remainder."),
                PromptVariant("last_digit_remainder", "Give the remainder of {value} after division by 10. Answer with a digit from 0 to 9."),
            ),
        ),
        PromptStyle(
            style_id="parity",
            label="Parity",
            output_kind="bool",
            target=lambda v: int(v) % 2 == 0,
            mae_kind="none",
            variants=(
                PromptVariant("is_even", "Is {value} even? Answer yes or no."),
                PromptVariant("divisible_by_two", "Is {value} divisible by 2? Answer yes or no."),
                PromptVariant("even_number_check", "Does {value} have even parity? Answer yes or no."),
            ),
        ),
        PromptStyle(
            style_id="rounding_nearest_hundred",
            label="Rounding",
            output_kind="int",
            target=_round_nearest_hundred,
            variants=(
                PromptVariant("round_nearest_hundred", "Round {value} to the nearest hundred. Answer with the rounded number."),
                PromptVariant("nearest_100", "What is {value} rounded to the closest multiple of 100? Answer only the number."),
                PromptVariant("hundred_rounding", "If {value} is rounded to the nearest 100, what number do you get?"),
            ),
        ),
        PromptStyle(
            style_id="successor",
            label="Successor",
            output_kind="int",
            target=lambda v: int(v) + 1,
            variants=(
                PromptVariant("next_number", "What number comes immediately after {value}? Answer with the number."),
                PromptVariant("successor", "The successor of {value} is what number?"),
                PromptVariant("add_one", "Add 1 to {value}. Answer only the result."),
            ),
        ),
        PromptStyle(
            style_id="zero_padded",
            label="Zero-Padded Form",
            output_kind="string",
            target=lambda v: f"{int(v):03d}",
            variants=(
                PromptVariant("three_digit_form", "Write {value} as a three-digit number with leading zeros if needed."),
                PromptVariant("zero_padded", "Convert {value} to zero-padded 3-digit form. Answer only the three digits."),
                PromptVariant("fixed_width", "Format {value} using exactly three decimal digits. Include leading zeros when necessary."),
            ),
            max_new_tokens=6,
            mae_kind="none",
        ),
        PromptStyle(
            style_id="hexadecimal",
            label="Hexadecimal",
            output_kind="hex",
            target=lambda v: f"{int(v):X}",
            variants=(
                PromptVariant("hexadecimal", "Write {value} in hexadecimal. Answer only the hexadecimal digits."),
                PromptVariant("base_16", "Convert decimal {value} to base 16. Answer with the hex value."),
                PromptVariant("hex_notation", "What is the hexadecimal representation of {value}? Do not include 0x."),
            ),
            max_new_tokens=8,
        ),
        PromptStyle(
            style_id="semantic_affordability",
            label="Semantic Threshold",
            output_kind="bool",
            target=lambda v: int(v) >= 500,
            mae_kind="none",
            variants=(
                PromptVariant("can_afford", "An item costs 500 dollars. If you have {value} dollars, can you afford it? Answer yes or no."),
                PromptVariant("enough_pages", "A report must be at least 500 pages. A report has {value} pages. Does it meet the requirement? Answer yes or no."),
                PromptVariant("meets_minimum", "The minimum required score is 500. A score is {value}. Does it meet the minimum? Answer yes or no."),
            ),
        ),
    )


PROMPT_STYLES = _make_prompt_styles()


def model_label(model_alias: str) -> str:
    return MODEL_LABELS.get(str(model_alias), str(model_alias))


def select_eval_values(*, seed: int, num_values: int, value_min: int, value_max: int) -> list[int]:
    blocked = set(DISTRACTORS)
    population = [value for value in range(int(value_min), int(value_max) + 1) if value not in blocked]
    if int(num_values) > len(population):
        raise ValueError(f"Requested {num_values} values from a population of {len(population)}.")
    rng = np.random.default_rng(int(seed))
    values = np.sort(rng.choice(np.asarray(population, dtype=np.int64), size=int(num_values), replace=False))
    return [int(value) for value in values.tolist()]


def _fewshot_values(value: int, *, k: int) -> list[int]:
    if int(k) <= 0:
        return []
    anchors = [3, 58, 241, 509, 777, 964]
    selected: list[int] = []
    for anchor in anchors:
        if int(anchor) == int(value) or int(anchor) in DISTRACTORS:
            continue
        selected.append(int(anchor))
        if len(selected) >= int(k):
            break
    if len(selected) < int(k):
        raise ValueError(f"Could not build {k}-shot examples.")
    return selected


def _format_target(style: PromptStyle, target: Any) -> str:
    if style.output_kind == "bool":
        return _bool_text(bool(target))
    return str(target)


def _render_query(template: str, value: int) -> str:
    return str(template).format(value=int(value)).rstrip()


def render_prompt(style: PromptStyle, variant: PromptVariant, value: int, fewshot_k: int) -> str:
    examples: list[str] = []
    for example_value in _fewshot_values(value, k=int(fewshot_k)):
        example_query = _render_query(variant.template, example_value)
        example_target = _format_target(style, style.target(example_value))
        examples.append(f"Q: {example_query}\nA: {example_target}")
    query = _render_query(variant.template, value)
    if examples:
        return "\n\n".join(examples) + f"\n\nQ: {query}\nA:"
    return f"Q: {query}\nA:"


def build_prompt_frame(values: list[int], fewshot_k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for style in PROMPT_STYLES:
        for variant in style.variants:
            for fewshot_k in fewshot_k_values:
                for value in values:
                    target = style.target(int(value))
                    rows.append(
                        {
                            "style_id": style.style_id,
                            "style_label": style.label,
                            "variant_id": variant.variant_id,
                            "fewshot_k": int(fewshot_k),
                            "value": int(value),
                            "target": _format_target(style, target),
                            "target_numeric": target_to_numeric(style, target),
                            "output_kind": style.output_kind,
                            "mae_kind": style.mae_kind,
                            "prompt": render_prompt(style, variant, int(value), int(fewshot_k)),
                            "max_new_tokens": int(style.max_new_tokens),
                        }
                    )
    return pd.DataFrame(rows)


def _first_int(text: str) -> int | None:
    match = re.search(r"[-+]?\d+", str(text))
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _first_bool(text: str) -> str | None:
    match = re.search(r"\b(yes|no|true|false)\b", str(text).lower())
    if match is None:
        return None
    token = match.group(1)
    return "yes" if token in {"yes", "true"} else "no"


def _first_zero_padded(text: str) -> str | None:
    match = re.search(r"\b\d{3}\b", str(text))
    if match is not None:
        return match.group(0)
    value = _first_int(text)
    if value is None or value < 0 or value > 999:
        return None
    return f"{value:03d}"


def _first_hex(text: str) -> str | None:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^0x", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\b[0-9A-Fa-f]{1,4}\b", cleaned)
    if match is None:
        return None
    return match.group(0).upper()


def parse_prediction(output_text: str, output_kind: OutputKind) -> tuple[str | None, float | None]:
    if output_kind == "bool":
        parsed = _first_bool(output_text)
        if parsed is None:
            return None, None
        return parsed, 1.0 if parsed == "yes" else 0.0
    if output_kind == "int":
        value = _first_int(output_text)
        if value is None:
            return None, None
        return str(int(value)), float(value)
    if output_kind == "string":
        parsed = _first_zero_padded(output_text)
        if parsed is None:
            return None, None
        return parsed, None
    if output_kind == "hex":
        parsed = _first_hex(output_text)
        if parsed is None:
            return None, None
        try:
            return parsed, float(int(parsed, 16))
        except ValueError:
            return parsed, None
    raise ValueError(f"Unknown output kind: {output_kind}")


def target_to_numeric(style: PromptStyle, target: Any) -> float | None:
    if style.output_kind == "bool":
        return 1.0 if bool(target) else 0.0
    if style.output_kind == "int":
        return float(target)
    if style.output_kind == "hex":
        return float(int(str(target), 16))
    return None


def generate_outputs(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> list[str]:
    outputs: list[str] = []
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), int(batch_size)):
            batch_prompts = prompts[start : start + int(batch_size)]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            generation_config = copy.deepcopy(getattr(model, "generation_config", None))
            if generation_config is not None:
                generation_config.max_length = None
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.do_sample = False
                generation_config.max_new_tokens = int(max_new_tokens)
                generation_config.pad_token_id = tokenizer.pad_token_id
                generation_config.eos_token_id = tokenizer.eos_token_id
            else:
                generation_config = GenerationConfig(
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
            continuation_ids = generated[:, input_ids.shape[1] :]
            outputs.extend(tokenizer.batch_decode(continuation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    finally:
        tokenizer.padding_side = old_padding_side
    return outputs


def evaluate_model(
    *,
    model_alias: str,
    model_entry: dict[str, Any],
    prompts: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
    device: str | None,
    dtype: str | None,
    force: bool,
) -> pd.DataFrame:
    model_dir = ensure_dir(output_dir / "models" / str(model_alias))
    predictions_path = model_dir / "predictions.csv"
    summary_path = model_dir / "summary.csv"
    if maybe_resume(summary_path, force=force) and predictions_path.exists():
        return pd.read_csv(summary_path)

    resolved_device = str(device or model_entry.get("device", "cpu"))
    resolved_dtype = str(dtype or model_entry.get("dtype", "float32"))
    loaded = load_causal_lm(
        model_name=str(model_entry["hf_id"]),
        device=resolved_device,
        dtype=resolved_dtype,
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        attn_implementation=model_entry.get("attn_implementation"),
    )
    rows = prompts.copy().reset_index(drop=True)
    rows.insert(0, "model_alias", str(model_alias))
    rows.insert(1, "model_label", model_label(str(model_alias)))
    rows.insert(2, "model_name", str(model_entry["hf_id"]))
    rows["raw_completion"] = ""

    for (style_id, variant_id, fewshot_k, max_new_tokens), group in rows.groupby(
        ["style_id", "variant_id", "fewshot_k", "max_new_tokens"], sort=False
    ):
        outputs = generate_outputs(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            prompts=group["prompt"].astype(str).tolist(),
            max_new_tokens=int(max_new_tokens),
            batch_size=int(batch_size),
            device=resolved_device,
        )
        rows.loc[group.index, "raw_completion"] = outputs

    parsed_rows = rows.apply(
        lambda row: parse_prediction(str(row["raw_completion"]), str(row["output_kind"])), axis=1, result_type="expand"
    )
    rows["parsed"] = parsed_rows[0]
    rows["predicted_numeric"] = parsed_rows[1]
    rows["parsed_ok"] = rows["parsed"].notna()
    rows["correct"] = rows["parsed"].astype(str).str.lower() == rows["target"].astype(str).str.lower()
    rows.loc[~rows["parsed_ok"], "correct"] = False
    rows["abs_error"] = np.nan
    numeric_mask = rows["target_numeric"].notna() & rows["predicted_numeric"].notna() & (rows["mae_kind"] == "numeric")
    rows.loc[numeric_mask, "abs_error"] = (
        rows.loc[numeric_mask, "predicted_numeric"].astype(float) - rows.loc[numeric_mask, "target_numeric"].astype(float)
    ).abs()

    summary = summarize_predictions(rows)
    atomic_write_csv(predictions_path, rows)
    atomic_write_csv(summary_path, summary)
    dump_run_config(
        model_dir / "config.json",
        {
            "model_alias": model_alias,
            "model_entry": model_entry,
            "batch_size": int(batch_size),
            "device": resolved_device,
            "dtype": resolved_dtype,
        },
    )

    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def summarize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            [
                "model_alias",
                "model_label",
                "style_id",
                "style_label",
                "variant_id",
                "fewshot_k",
                "output_kind",
                "mae_kind",
            ],
            as_index=False,
        )
        .agg(
            num_examples=("correct", "size"),
            accuracy=("correct", "mean"),
            parse_rate=("parsed_ok", "mean"),
            mae=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
        )
        .sort_values(["style_id", "variant_id", "fewshot_k", "model_alias"])
        .reset_index(drop=True)
    )
    return grouped


def _format_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _render_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=False, classes="dataframe")


def _fig_to_div(fig: go.Figure, *, include_plotlyjs: str | bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _variant_template_lookup() -> dict[tuple[str, str], str]:
    templates: dict[tuple[str, str], str] = {}
    for style in PROMPT_STYLES:
        for variant in style.variants:
            templates[(style.style_id, variant.variant_id)] = variant.template
    return templates


def _html_snippet(value: Any, *, limit: int = 150) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > int(limit):
        text = text[: int(limit) - 1] + "..."
    return html.escape(text).replace("\n", "<br>")


def _prediction_example_cell(row: pd.Series | None) -> str:
    if row is None:
        return "<span class=\"muted\">none observed</span>"
    parsed = row.get("parsed")
    parsed_text = "" if parsed is None or (isinstance(parsed, float) and math.isnan(parsed)) else str(parsed)
    return (
        f"<b>{html.escape(str(row['model_label']))}</b>, "
        f"{int(row['fewshot_k'])}-shot, value <code>{int(row['value'])}</code>, "
        f"target <code>{html.escape(str(row['target']))}</code><br>"
        f"completion: <code>{_html_snippet(row.get('raw_completion'), limit=120)}</code><br>"
        f"parsed: <code>{html.escape(parsed_text)}</code>"
    )


def _first_or_none(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[0]


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def build_variant_examples(predictions: pd.DataFrame, *, style_order: list[str]) -> pd.DataFrame:
    templates = _variant_template_lookup()
    style_rank = {label: idx for idx, label in enumerate(style_order)}
    base_rows: list[dict[str, Any]] = []
    for style in PROMPT_STYLES:
        for variant in style.variants:
            base_rows.append(
                {
                    "style_id": style.style_id,
                    "style_label": style.label,
                    "variant_id": variant.variant_id,
                    "template": variant.template,
                    "style_rank": style_rank.get(style.label, len(style_rank)),
                }
            )
    if predictions.empty:
        return pd.DataFrame(base_rows)

    pred = predictions.copy()
    pred["correct_bool"] = _as_bool_series(pred["correct"])
    pred["parsed_ok_bool"] = _as_bool_series(pred["parsed_ok"])
    pred["fewshot_sort"] = pred["fewshot_k"].astype(int)
    pred["value_sort"] = pred["value"].astype(int)
    pred = pred.sort_values(["style_label", "variant_id", "fewshot_sort", "model_alias", "value_sort"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for base in base_rows:
        subset = pred.loc[(pred["style_id"] == base["style_id"]) & (pred["variant_id"] == base["variant_id"])].copy()
        correct = _first_or_none(subset.loc[subset["correct_bool"]])
        wrong = _first_or_none(subset.loc[subset["parsed_ok_bool"] & ~subset["correct_bool"]])
        unparsable = _first_or_none(subset.loc[~subset["parsed_ok_bool"]])
        rows.append(
            {
                **base,
                "correct_example": _prediction_example_cell(correct),
                "wrong_example": _prediction_example_cell(wrong),
                "unparsable_example": _prediction_example_cell(unparsable),
                "num_correct": int(subset["correct_bool"].sum()),
                "num_wrong": int((subset["parsed_ok_bool"] & ~subset["correct_bool"]).sum()),
                "num_unparsable": int((~subset["parsed_ok_bool"]).sum()),
                "template": templates.get((base["style_id"], base["variant_id"]), base["template"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["style_rank", "variant_id"]).reset_index(drop=True)


def build_report(*, output_dir: Path, report_stem: str) -> Path:
    summary_paths = sorted((output_dir / "models").glob("*/summary.csv"))
    prediction_paths = sorted((output_dir / "models").glob("*/predictions.csv"))
    if not summary_paths:
        raise RuntimeError(f"No model summaries found under {output_dir}")
    summary = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True) if prediction_paths else pd.DataFrame()

    report_dir = ensure_dir(Path("experiment-reports"))
    artifact_dir = ensure_dir(report_dir / "artifacts" / report_stem)
    report_path = report_dir / f"{report_stem}.html"

    setting_summary = (
        summary.groupby(["style_id", "style_label", "variant_id", "fewshot_k"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            min_model_accuracy=("accuracy", "min"),
            max_model_accuracy=("accuracy", "max"),
            mean_parse_rate=("parse_rate", "mean"),
            mean_mae=("mae", "mean"),
            num_models=("model_alias", "nunique"),
        )
        .sort_values(["style_label", "mean_accuracy"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best_setting = (
        setting_summary.sort_values(["style_id", "mean_accuracy", "mean_parse_rate"], ascending=[True, False, False])
        .groupby(["style_id", "style_label"], as_index=False)
        .head(1)
        .rename(
            columns={
                "variant_id": "best_variant_id",
                "fewshot_k": "best_fewshot_k",
                "mean_accuracy": "best_mean_accuracy",
                "mean_parse_rate": "best_mean_parse_rate",
                "mean_mae": "best_mean_mae",
            }
        )
    )
    variation = (
        setting_summary.groupby(["style_id", "style_label"], as_index=False)
        .agg(
            worst_mean_accuracy=("mean_accuracy", "min"),
            setting_accuracy_std=("mean_accuracy", "std"),
            worst_mean_mae=("mean_mae", "max"),
            num_settings=("mean_accuracy", "size"),
        )
    )
    style_summary = best_setting.merge(variation, on=["style_id", "style_label"], how="left")
    style_summary["accuracy_range"] = style_summary["best_mean_accuracy"] - style_summary["worst_mean_accuracy"]
    style_summary = style_summary.sort_values(["best_mean_accuracy", "style_label"], ascending=[False, True]).reset_index(drop=True)

    model_style_best = (
        summary.sort_values(["model_alias", "style_id", "accuracy", "parse_rate"], ascending=[True, True, False, False])
        .groupby(["model_alias", "model_label", "style_id", "style_label"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    atomic_write_csv(artifact_dir / "setting_summary.csv", setting_summary)
    atomic_write_csv(artifact_dir / "style_summary.csv", style_summary)
    atomic_write_csv(artifact_dir / "model_style_best.csv", model_style_best)

    style_order = style_summary["style_label"].tolist()
    model_order = [model_label(alias) for alias in summary["model_alias"].drop_duplicates().tolist()]
    heatmap_table = (
        model_style_best.pivot(index="model_label", columns="style_label", values="accuracy")
        .reindex(index=model_order, columns=style_order)
    )
    fig_heatmap = go.Figure(
        data=[
            go.Heatmap(
                z=heatmap_table.to_numpy(dtype=np.float64),
                x=heatmap_table.columns.tolist(),
                y=heatmap_table.index.tolist(),
                zmin=0.0,
                zmax=1.0,
                colorscale="Blues",
                text=heatmap_table.to_numpy(dtype=np.float64),
                texttemplate="%{z:.0%}",
                hovertemplate="Model=%{y}<br>Style=%{x}<br>Best accuracy=%{z:.4f}<extra></extra>",
                colorbar={"title": "Accuracy"},
            )
        ]
    )
    fig_heatmap.update_layout(
        title="Best Accuracy Per Model and Prompt Style",
        template="plotly_white",
        height=430,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )

    setting_fig_table = setting_summary.copy()
    setting_fig_table["setting"] = setting_fig_table["variant_id"] + " / " + setting_fig_table["fewshot_k"].astype(str) + "-shot"
    fig_settings = go.Figure()
    for style_label in style_order:
        subset = setting_fig_table.loc[setting_fig_table["style_label"] == style_label].sort_values("mean_accuracy", ascending=False)
        fig_settings.add_trace(
            go.Bar(
                x=subset["setting"],
                y=subset["mean_accuracy"],
                name=style_label,
                visible=(style_label == style_order[0]),
                hovertemplate="Setting=%{x}<br>Mean accuracy=%{y:.4f}<extra></extra>",
            )
        )
    buttons = []
    for idx, style_label in enumerate(style_order):
        visible = [False] * len(style_order)
        visible[idx] = True
        buttons.append({"label": style_label, "method": "update", "args": [{"visible": visible}, {"title": f"Setting Variation: {style_label}"}]})
    fig_settings.update_layout(
        title=f"Setting Variation: {style_order[0] if style_order else ''}",
        template="plotly_white",
        height=430,
        yaxis={"range": [0.0, 1.0], "title": "Mean accuracy across models"},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.0, "y": 1.18}],
        margin={"l": 10, "r": 10, "t": 90, "b": 120},
    )

    style_display = style_summary[
        [
            "style_label",
            "best_mean_accuracy",
            "best_variant_id",
            "best_fewshot_k",
            "worst_mean_accuracy",
            "accuracy_range",
            "setting_accuracy_std",
            "best_mean_mae",
            "best_mean_parse_rate",
        ]
    ].copy()
    style_display.columns = [
        "Prompt Style",
        "Best Mean Accuracy",
        "Best Variant",
        "Best k",
        "Worst Mean Accuracy",
        "Range",
        "Std Across Settings",
        "Best Mean MAE",
        "Best Parse Rate",
    ]
    for column in ["Best Mean Accuracy", "Worst Mean Accuracy", "Range", "Std Across Settings", "Best Parse Rate"]:
        style_display[column] = style_display[column].map(_format_pct)
    style_display["Best Mean MAE"] = style_display["Best Mean MAE"].map(_format_float)

    setting_display = setting_summary[
        ["style_label", "variant_id", "fewshot_k", "mean_accuracy", "min_model_accuracy", "max_model_accuracy", "mean_mae", "mean_parse_rate"]
    ].copy()
    setting_display.columns = ["Prompt Style", "Variant", "k", "Mean Accuracy", "Min Model Accuracy", "Max Model Accuracy", "Mean MAE", "Parse Rate"]
    for column in ["Mean Accuracy", "Min Model Accuracy", "Max Model Accuracy", "Parse Rate"]:
        setting_display[column] = setting_display[column].map(_format_pct)
    setting_display["Mean MAE"] = setting_display["Mean MAE"].map(_format_float)

    examples_html = ""
    variant_examples_html = ""
    if not predictions.empty:
        variant_examples = build_variant_examples(predictions, style_order=style_order)
        variant_examples_artifact = variant_examples[
            [
                "style_label",
                "variant_id",
                "template",
                "num_correct",
                "num_wrong",
                "num_unparsable",
                "correct_example",
                "wrong_example",
                "unparsable_example",
            ]
        ].copy()
        atomic_write_csv(artifact_dir / "variant_examples.csv", variant_examples_artifact)
        variant_examples_display = variant_examples_artifact.copy()
        variant_examples_display.columns = [
            "Prompt Style",
            "Variant",
            "Template",
            "# Correct",
            "# Wrong",
            "# Unparsable",
            "Correct Example",
            "Wrong Example",
            "Unparsable Example",
        ]
        variant_examples_display["Template"] = variant_examples_display["Template"].map(
            lambda text: f"<code>{html.escape(str(text))}</code>"
        )
        variant_examples_html = (
            "<h3>Prompt Variant Examples</h3>"
            "<p class=\"note\">Each row is one prompt-style paraphrase. Examples are representative completions selected from all evaluated models, both shot settings, and the shared 100 values. Counts are over all 1,200 predictions for that variant.</p>"
            f"{_render_table(variant_examples_display)}"
        )

        misses = (
            predictions.loc[~_as_bool_series(predictions["correct"])]
            .sort_values(["style_label", "model_alias", "variant_id", "fewshot_k", "value"])
            .head(24)
            [["model_label", "style_label", "variant_id", "fewshot_k", "value", "target", "raw_completion", "parsed"]]
            .copy()
        )
        if not misses.empty:
            misses.columns = ["Model", "Style", "Variant", "k", "Value", "Target", "Completion", "Parsed"]
            for col in ["Completion", "Parsed"]:
                misses[col] = misses[col].fillna("").map(lambda text: f"<code>{html.escape(str(text)[:80])}</code>")
            examples_html = f"<h2>Example Misses</h2>{_render_table(misses)}"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(report_stem)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #182026; }}
h1, h2 {{ line-height: 1.2; }}
.note {{ color: #52606d; max-width: 1000px; }}
table.dataframe {{ border-collapse: collapse; font-size: 13px; margin: 16px 0 28px; max-width: 100%; }}
table.dataframe th, table.dataframe td {{ border: 1px solid #d9e2ec; padding: 6px 8px; vertical-align: top; }}
table.dataframe th {{ background: #f0f4f8; }}
code {{ background: #f5f7fa; padding: 1px 3px; border-radius: 3px; }}
.muted {{ color: #7b8794; }}
</style>
</head>
<body>
<h1>Numeric Prompt Style Pilot</h1>
<h2>Research Question</h2>
<p class="note">Which prompt styles make 0-999 numeric input values usable for small-to-mid-sized language models, under matched zero-shot and 3-shot settings over the same 100 evaluation values?</p>
<h2>Scope of Evidence</h2>
<p class="note">The pilot evaluates {int(summary['model_alias'].nunique())} models, {len(style_order)} prompt styles, 3 paraphrases per style, zero-shot and 3-shot prompting, and {int(summary['num_examples'].max())} matched values per setting. Accuracy is exact match after lightweight parsing. MAE is reported only for styles with numeric or numeric-equivalent outputs.</p>
<h2>Executive Summary</h2>
<p class="note">Prompt styles are sorted by their best mean setting across models. The range and standard deviation columns quantify sensitivity to paraphrase and in-context examples within each style.</p>
<h2>Key Tables and Figures</h2>
<h3>Prompt Styles Sorted by Best Setting</h3>
{_render_table(style_display)}
{_fig_to_div(fig_heatmap, include_plotlyjs=False)}
{_fig_to_div(fig_settings, include_plotlyjs=False)}
<h3>All Settings</h3>
{_render_table(setting_display)}
{variant_examples_html}
{examples_html}
<h2>Metric Glossary</h2>
<p class="note"><b>Accuracy</b>: fraction of prompts where the parsed completion exactly equals the target. <b>Parse rate</b>: fraction of completions from which the expected answer type could be parsed. <b>MAE</b>: mean absolute difference between parsed numeric value and target numeric value, computed only when a numeric target and numeric parse exist.</p>
<h2>Limitations</h2>
<p class="note">This is a prompt-selection pilot, not a causal representation test. Some styles can be partly solved by digit heuristics, and generation parsing can penalize otherwise correct verbose answers. The matched 100-value design supports prompt-style screening, but not broad generalization claims.</p>
<h2>Recommended Next Experiments</h2>
<p class="note">Use the highest-accuracy, low-variation styles as candidate activation/probe families. Keep at least one scalar-comparison style and one digit/place-value style so downstream transfer can distinguish scalar value structure from digit-local shortcuts.</p>
<p class="note">Source artifacts: <code>{html.escape(str(output_dir))}</code>; derived tables: <code>{html.escape(str(artifact_dir))}</code>.</p>
</body>
</html>
"""
    atomic_write_text(report_path, html_doc)
    return report_path


def run_pilot(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("seed", 1729)))
    output_dir = ensure_dir(Path(args.output_dir or cfg.get("output_dir", "out/runs/numeric_prompt_style_pilot")))
    values = select_eval_values(
        seed=int(cfg.get("seed", 1729)),
        num_values=int(cfg.get("num_values", 100)),
        value_min=int(cfg.get("value_min", 0)),
        value_max=int(cfg.get("value_max", 998)),
    )
    fewshot_k_values = [int(k) for k in cfg.get("fewshot_k_values", [0, 3])]
    prompts = build_prompt_frame(values, fewshot_k_values)
    atomic_write_csv(output_dir / "values.csv", pd.DataFrame({"value": values}))
    atomic_write_csv(output_dir / "prompts.csv", prompts)

    model_cfg = load_yaml(cfg.get("models_config", "configs/models.yaml"))
    model_aliases = [str(alias) for alias in (args.models or cfg.get("models", []))]
    if not model_aliases:
        model_aliases = list(model_cfg["models"].keys())
    for model_alias in model_aliases:
        entry = {**model_cfg.get("default", {}), **model_cfg["models"][model_alias]}
        evaluate_model(
            model_alias=model_alias,
            model_entry=entry,
            prompts=prompts,
            output_dir=output_dir,
            batch_size=int(args.batch_size or cfg.get("batch_size", 32)),
            device=args.device or cfg.get("device"),
            dtype=args.dtype,
            force=bool(args.force),
        )
    summary_paths = sorted((output_dir / "models").glob("*/summary.csv"))
    if summary_paths:
        atomic_write_csv(output_dir / "summary.csv", pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True))
    atomic_write_json(
        output_dir / "run.json",
        {
            "config": cfg,
            "num_prompt_styles": len(PROMPT_STYLES),
            "num_values": len(values),
            "num_prompt_rows": int(len(prompts)),
            "models": model_aliases,
        },
    )
    report_path = build_report(output_dir=output_dir, report_stem=str(args.report_stem or cfg.get("report_stem", "numeric-prompt-style-pilot")))
    configured_models = {str(alias) for alias in cfg.get("models", [])}
    completed_models = {path.parent.name for path in summary_paths}
    if configured_models and configured_models.issubset(completed_models):
        atomic_write_text(output_dir / "DONE", "done\n")
    return report_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a matched numeric prompt-style pilot.")
    parser.add_argument("--config", default="configs/numeric_prompt_style_pilot.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-stem", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir or cfg.get("output_dir", "out/runs/numeric_prompt_style_pilot"))
    if args.report_only:
        report_path = build_report(output_dir=output_dir, report_stem=str(args.report_stem or cfg.get("report_stem", "numeric-prompt-style-pilot")))
    else:
        report_path = run_pilot(args)
    print(report_path)
