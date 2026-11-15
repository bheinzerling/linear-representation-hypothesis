from __future__ import annotations

import argparse
import copy
import gc
import html
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import GenerationConfig

from src.activation_steering import collect_prompt_activations
from src.activation_steering import render_prompt_with_operand_spans, resolve_token_idx
from src.model_loader import load_causal_lm, load_tokenizer
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_jsonl, atomic_write_text, ensure_dir, load_yaml, set_global_seed


@dataclass(frozen=True)
class RhsPromptStyle:
    style_id: str
    style_label: str
    template: str
    fewshot_examples: tuple[tuple[int, int], ...] = ()
    separator: str = "; "


DEFAULT_X_VALUES = (
    0,
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
    144,
    233,
    377,
    486,
    610,
    731,
    812,
    900,
    999,
)


PROMPT_STYLES = (
    RhsPromptStyle(
        style_id="addition_formula_rhs_source",
        style_label="Addition Formula RHS Source",
        template="x1+x2=",
        fewshot_examples=((123, 726), (909, 4), (52, 738)),
    ),
    RhsPromptStyle(
        style_id="addition_verbalized_rhs_source",
        style_label="Addition Verbalized RHS Source",
        template="The result of x1 plus x2 is ",
        fewshot_examples=(),
    ),
)


MODEL_LABELS = {
    "llama_3_2_1b": "Llama 3.2 1B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "llama_3_8b": "Llama 3 8B",
    "llama_3_1_8b": "Llama 3.1 8B",
    "olmo_3_7b_base": "OLMo-3 7B",
    "olmo_2_13b": "OLMo-2 13B",
    "gpt_oss_20b": "gpt-oss-20b",
}


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".npz", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def model_label(model_alias: str) -> str:
    return MODEL_LABELS.get(str(model_alias), str(model_alias))


def _render_fewshot_prefix(style: RhsPromptStyle) -> str:
    if not style.fewshot_examples:
        return ""
    examples = []
    for x1, x2 in style.fewshot_examples:
        examples.append(render_prompt_with_operand_spans(style.template, int(x1), int(x2))[0] + str(int(x1) + int(x2)))
    return style.separator.join(examples) + style.separator


def build_rhs_addition_frame(
    *,
    tokenizer: Any | None = None,
    value_min: int = 0,
    value_max: int = 999,
    x_values: tuple[int, ...] = DEFAULT_X_VALUES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for style in PROMPT_STYLES:
        prefix = _render_fewshot_prefix(style)
        for source_value in range(int(value_min), int(value_max) + 1):
            for variant_idx, x in enumerate(x_values):
                x1 = int(x)
                x2 = int(source_value)
                base_prompt, spans = render_prompt_with_operand_spans(style.template, x1, x2)
                prompt = prefix + base_prompt
                source_span = (len(prefix) + spans["x2"][0], len(prefix) + spans["x2"][1])
                token_idx = None
                if tokenizer is not None:
                    token_idx = resolve_token_idx(
                        tokenizer=tokenizer,
                        prompt=prompt,
                        mention_span=source_span,
                        mode="mention_last",
                    )
                rows.append(
                    {
                        "style_id": style.style_id,
                        "style_label": style.style_label,
                        "variant_idx": int(variant_idx),
                        "source_value": int(source_value),
                        "x": int(x),
                        "x1": int(x1),
                        "x2": int(x2),
                        "target": int(x1 + x2),
                        "template": style.template,
                        "fewshot_num_examples": int(len(style.fewshot_examples)),
                        "prompt": prompt,
                        "source_char_start": int(source_span[0]),
                        "source_char_end": int(source_span[1]),
                        "token_idx": None if token_idx is None else int(token_idx),
                    }
                )
    return pd.DataFrame(rows)


def write_dataset(frame: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    atomic_write_csv(output_dir / "prompts.csv", frame)
    atomic_write_jsonl(output_dir / "prompts.jsonl", frame.to_dict(orient="records"))
    summary = (
        frame.groupby(["style_id", "style_label"], as_index=False)
        .agg(
            num_rows=("prompt", "size"),
            num_source_values=("source_value", "nunique"),
            variants_per_source_min=("variant_idx", lambda s: int(s.groupby(frame.loc[s.index, "source_value"]).nunique().min())),
            variants_per_source_max=("variant_idx", lambda s: int(s.groupby(frame.loc[s.index, "source_value"]).nunique().max())),
            target_min=("target", "min"),
            target_max=("target", "max"),
        )
        .reset_index(drop=True)
    )
    atomic_write_csv(output_dir / "dataset_summary.csv", summary)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "value_min": int(frame["source_value"].min()),
            "value_max": int(frame["source_value"].max()),
            "num_rows": int(len(frame)),
            "x_values": [int(value) for value in sorted(frame["x"].unique().tolist())],
            "styles": summary.to_dict(orient="records"),
        },
    )


def _first_int(text: str) -> int | None:
    match = re.search(r"[-+]?\d+", str(text))
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def parse_completion_int(text: str, style_id: str | None = None) -> int | None:
    """Parse arithmetic completions that may echo an equation before the answer."""
    if str(style_id or "").endswith("formula_rhs_source"):
        return _first_int(text)
    parts = str(text).rsplit("=", 1)
    if len(parts) == 2:
        after_equals = _first_int(parts[1])
        if after_equals is not None:
            return after_equals
    return _first_int(text)


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


def select_behavior_subset(frame: pd.DataFrame, *, source_stride: int | None, max_rows_per_style: int | None, seed: int) -> pd.DataFrame:
    subset = frame.copy()
    if source_stride is not None and int(source_stride) > 1:
        subset = subset.loc[subset["source_value"].astype(int) % int(source_stride) == 0].copy()
    if max_rows_per_style is not None:
        rng = np.random.default_rng(int(seed))
        pieces = []
        for _, group in subset.groupby("style_id", sort=False):
            if len(group) <= int(max_rows_per_style):
                pieces.append(group)
            else:
                take = rng.choice(group.index.to_numpy(), size=int(max_rows_per_style), replace=False)
                pieces.append(group.loc[np.sort(take)])
        subset = pd.concat(pieces, ignore_index=True)
    return subset.sort_values(["style_id", "source_value", "variant_idx"]).reset_index(drop=True)


def evaluate_model(
    *,
    model_alias: str,
    model_entry: dict[str, Any],
    prompts: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
    device: str | None,
    dtype: str | None,
    max_new_tokens: int,
) -> pd.DataFrame:
    model_dir = ensure_dir(output_dir / "models" / str(model_alias))
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
    for (style_id,), group in rows.groupby(["style_id"], sort=False):
        del style_id
        rows.loc[group.index, "raw_completion"] = generate_outputs(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            prompts=group["prompt"].astype(str).tolist(),
            max_new_tokens=int(max_new_tokens),
            batch_size=int(batch_size),
            device=resolved_device,
        )
    rows["parsed"] = rows.apply(lambda row: parse_completion_int(str(row["raw_completion"]), str(row["style_id"])), axis=1)
    rows["parsed_ok"] = rows["parsed"].notna()
    rows["correct"] = rows["parsed"].astype("float64") == rows["target"].astype("float64")
    rows.loc[~rows["parsed_ok"], "correct"] = False
    rows["abs_error"] = np.nan
    ok = rows["parsed_ok"]
    rows.loc[ok, "abs_error"] = (rows.loc[ok, "parsed"].astype(int) - rows.loc[ok, "target"].astype(int)).abs()
    summary = summarize_predictions(rows)
    atomic_write_csv(model_dir / "predictions.csv", rows)
    atomic_write_csv(model_dir / "summary.csv", summary)
    atomic_write_json(
        model_dir / "config.json",
        {
            "model_alias": str(model_alias),
            "model_entry": model_entry,
            "device": resolved_device,
            "dtype": resolved_dtype,
            "batch_size": int(batch_size),
            "max_new_tokens": int(max_new_tokens),
        },
    )
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def summarize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["model_alias", "model_label", "style_id", "style_label"], as_index=False)
        .agg(
            num_examples=("correct", "size"),
            accuracy=("correct", "mean"),
            parse_rate=("parsed_ok", "mean"),
            mae=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            target_min=("target", "min"),
            target_max=("target", "max"),
        )
        .sort_values(["style_id", "model_alias"])
        .reset_index(drop=True)
    )


def collect_representations(
    *,
    model_alias: str,
    model_entry: dict[str, Any],
    prompts: pd.DataFrame,
    output_dir: Path,
    layer_idx: int,
    batch_size: int,
    device: str | None,
    dtype: str | None,
    storage_dtype: str,
) -> Path:
    repr_dir = ensure_dir(output_dir / "representations" / str(model_alias) / f"layer_{int(layer_idx)}")
    resolved_device = str(device or model_entry.get("device", "cpu"))
    resolved_dtype = str(dtype or model_entry.get("dtype", "float32"))
    loaded = load_causal_lm(
        model_name=str(model_entry["hf_id"]),
        device=resolved_device,
        dtype=resolved_dtype,
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        attn_implementation=model_entry.get("attn_implementation"),
    )
    activations = collect_prompt_activations(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompts=prompts["prompt"].astype(str).tolist(),
        token_idxs=prompts["token_idx"].astype(int).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=resolved_device,
    )
    if str(storage_dtype) == "float16":
        activations_to_save = activations.astype(np.float16)
    elif str(storage_dtype) == "float32":
        activations_to_save = activations.astype(np.float32)
    else:
        raise ValueError(f"Unsupported representation storage_dtype: {storage_dtype!r}")
    _atomic_save_npz(
        repr_dir / "source_token_representations.npz",
        activations=activations_to_save,
        source_values=prompts["source_value"].to_numpy(dtype=np.int64),
        variant_indices=prompts["variant_idx"].to_numpy(dtype=np.int64),
        style_ids=prompts["style_id"].to_numpy(dtype=np.str_),
        x_values=prompts["x"].to_numpy(dtype=np.int64),
        token_indices=prompts["token_idx"].to_numpy(dtype=np.int64),
    )
    atomic_write_csv(repr_dir / "rows.csv", prompts)
    atomic_write_json(
        repr_dir / "metadata.json",
        {
            "model_alias": str(model_alias),
            "model_name": str(model_entry["hf_id"]),
            "layer_idx": int(layer_idx),
            "num_rows": int(len(prompts)),
            "hidden_dim": int(activations.shape[1]),
            "activation_dtype_in_memory": str(activations.dtype),
            "storage_dtype": str(activations_to_save.dtype),
            "device": resolved_device,
            "dtype": resolved_dtype,
            "batch_size": int(batch_size),
        },
    )
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return repr_dir / "source_token_representations.npz"


def _render_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in ["accuracy", "parse_rate"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{100.0 * float(value):.1f}%")
    for column in ["mae", "median_abs_error"]:
        if column in display:
            display[column] = display[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value):.2f}")
    return display.to_html(index=False, escape=False, classes="dataframe")


def build_report(*, output_dir: Path, report_stem: str) -> Path:
    report_dir = ensure_dir(Path("experiment-reports"))
    artifact_dir = ensure_dir(report_dir / "artifacts" / report_stem)
    report_path = report_dir / f"{report_stem}.html"
    dataset_summary = pd.read_csv(output_dir / "dataset" / "dataset_summary.csv")
    summary_paths = sorted((output_dir / "behavior" / "models").glob("*/summary.csv"))
    summaries = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True) if summary_paths else pd.DataFrame()
    prediction_paths = sorted((output_dir / "behavior" / "models").glob("*/predictions.csv"))
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True) if prediction_paths else pd.DataFrame()
    atomic_write_csv(artifact_dir / "dataset_summary.csv", dataset_summary)
    if not summaries.empty:
        atomic_write_csv(artifact_dir / "behavior_summary.csv", summaries)
    repr_paths = sorted((output_dir / "representations").glob("*/layer_*/metadata.json"))
    repr_rows = []
    for path in repr_paths:
        payload = load_yaml(path)
        repr_rows.append(payload)
    repr_summary = pd.DataFrame(repr_rows)
    if not repr_summary.empty:
        atomic_write_csv(artifact_dir / "representation_summary.csv", repr_summary)
    examples = pd.DataFrame()
    if not predictions.empty:
        rows = []
        for (model_alias, style_id), group in predictions.groupby(["model_alias", "style_id"], sort=False):
            correct = group.loc[group["correct"].astype(bool)].head(1)
            wrong = group.loc[group["parsed_ok"].astype(bool) & ~group["correct"].astype(bool)].head(1)
            unparsable = group.loc[~group["parsed_ok"].astype(bool)].head(1)
            for label, part in [("correct", correct), ("wrong", wrong), ("unparsable", unparsable)]:
                if part.empty:
                    continue
                item = part.iloc[0]
                rows.append(
                    {
                        "model_alias": model_alias,
                        "style_id": style_id,
                        "example_type": label,
                        "source_value": int(item["source_value"]),
                        "x": int(item["x"]),
                        "target": int(item["target"]),
                        "completion": str(item["raw_completion"]),
                        "parsed": "" if pd.isna(item["parsed"]) else str(int(item["parsed"])),
                        "prompt": str(item["prompt"]),
                    }
                )
        examples = pd.DataFrame(rows)
        atomic_write_csv(artifact_dir / "behavior_examples.csv", examples)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(report_stem)}</title>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #182026; }}
.note {{ color: #52606d; max-width: 1080px; }}
table.dataframe {{ border-collapse: collapse; font-size: 13px; margin: 16px 0 28px; max-width: 100%; }}
table.dataframe th, table.dataframe td {{ border: 1px solid #d9e2ec; padding: 6px 8px; vertical-align: top; }}
table.dataframe th {{ background: #f0f4f8; }}
code {{ background: #f5f7fa; padding: 1px 3px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>RHS-Source Addition Prompt Dataset and Behavioral Pilot</h1>
<h2>Research Question</h2>
<p class="note">Can we use prompts of the form <code>x + source_value =</code> to collect multiple contextual representations for every one-token number 0-999 while preserving good model behavior?</p>
<h2>Scope of Evidence</h2>
<p class="note">The dataset contains two addition prompt styles with the source value in the second operand position: formula few-shot and verbalized zero-shot. Each style has 20 fixed partner values for every source value 0-999, yielding 20,000 rows per style. The behavioral pilot may use a subset configured in the run metadata.</p>
<h2>Dataset Summary</h2>
{_render_table(dataset_summary)}
<h2>Behavior Summary</h2>
{_render_table(summaries) if not summaries.empty else '<p class="note">No behavior run was executed.</p>'}
<h2>Representation Collection</h2>
{_render_table(repr_summary) if not repr_summary.empty else '<p class="note">No representation collection was executed.</p>'}
<h2>Example Predictions</h2>
{_render_table(examples) if not examples.empty else '<p class="note">No prediction examples available.</p>'}
<h2>Design Notes</h2>
<p class="note">This is not the same as the older split-based prompt-cache variants: this dataset is balanced over all 1000 source values and keeps the source number in the RHS operand position for the addition prompts. Targets range up to 1998 because the goal is source-token representation collection rather than one-token target collection.</p>
<p class="note">Artifacts: <code>{html.escape(str(output_dir))}</code> and <code>{html.escape(str(artifact_dir))}</code>.</p>
</body>
</html>
"""
    atomic_write_text(report_path, html_doc)
    return report_path


def run(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("seed", 1729)))
    output_dir = ensure_dir(Path(args.output_dir or cfg.get("output_dir", "out/runs/source_value_rhs_addition_dataset__2026_05_13")))
    report_stem = str(args.report_stem or cfg.get("report_stem", "2026-05-13-source-value-rhs-addition-behavior-pilot"))
    model_cfg = load_yaml(cfg.get("models_config", "configs/models.yaml"))
    tokenizer_model_alias = str(cfg.get("tokenizer_model", cfg.get("models", ["llama_3_2_3b"])[0]))
    tokenizer_entry = {**model_cfg.get("default", {}), **model_cfg["models"][tokenizer_model_alias]}
    tokenizer = load_tokenizer(str(tokenizer_entry["hf_id"]), trust_remote_code=bool(tokenizer_entry.get("trust_remote_code", False)))
    frame = build_rhs_addition_frame(
        tokenizer=tokenizer,
        value_min=int(cfg.get("value_min", 0)),
        value_max=int(cfg.get("value_max", 999)),
        x_values=tuple(int(value) for value in cfg.get("x_values", DEFAULT_X_VALUES)),
    )
    write_dataset(frame, output_dir / "dataset")
    behavior_cfg = cfg.get("behavior", {})
    if bool(behavior_cfg.get("enabled", True)):
        eval_subset = select_behavior_subset(
            frame,
            source_stride=behavior_cfg.get("source_stride", 20),
            max_rows_per_style=behavior_cfg.get("max_rows_per_style"),
            seed=int(cfg.get("seed", 1729)),
        )
        behavior_dir = ensure_dir(output_dir / "behavior")
        atomic_write_csv(behavior_dir / "eval_prompts.csv", eval_subset)
        for model_alias in cfg.get("models", ["llama_3_2_3b"]):
            model_entry = {**model_cfg.get("default", {}), **model_cfg["models"][str(model_alias)]}
            evaluate_model(
                model_alias=str(model_alias),
                model_entry=model_entry,
                prompts=eval_subset,
                output_dir=behavior_dir,
                batch_size=int(behavior_cfg.get("batch_size", cfg.get("batch_size", 32))),
                device=behavior_cfg.get("device", cfg.get("device")),
                dtype=behavior_cfg.get("dtype"),
                max_new_tokens=int(behavior_cfg.get("max_new_tokens", 8)),
            )
    repr_cfg = cfg.get("representations", {})
    if bool(repr_cfg.get("enabled", False)):
        repr_frame = frame
        if repr_cfg.get("styles"):
            repr_frame = repr_frame.loc[repr_frame["style_id"].isin([str(style) for style in repr_cfg["styles"]])].reset_index(drop=True)
        if repr_cfg.get("max_rows") is not None:
            repr_frame = repr_frame.head(int(repr_cfg["max_rows"])).reset_index(drop=True)
        for model_alias in repr_cfg.get("models", cfg.get("models", ["llama_3_2_3b"])):
            model_entry = {**model_cfg.get("default", {}), **model_cfg["models"][str(model_alias)]}
            collect_representations(
                model_alias=str(model_alias),
                model_entry=model_entry,
                prompts=repr_frame,
                output_dir=output_dir,
                layer_idx=int(repr_cfg.get("layer_idx", 8)),
                batch_size=int(repr_cfg.get("batch_size", cfg.get("batch_size", 64))),
                device=repr_cfg.get("device", cfg.get("device")),
                dtype=repr_cfg.get("dtype"),
                storage_dtype=str(repr_cfg.get("storage_dtype", "float16")),
            )
    atomic_write_json(
        output_dir / "run.json",
        {
            "config": cfg,
            "num_dataset_rows": int(len(frame)),
            "num_styles": int(frame["style_id"].nunique()),
            "num_source_values": int(frame["source_value"].nunique()),
        },
    )
    return build_report(output_dir=output_dir, report_stem=report_stem)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RHS-source addition prompt dataset and run a quick behavioral pilot.")
    parser.add_argument("--config", default="configs/source_value_rhs_addition_behavior_pilot.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-stem", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    print(run(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
