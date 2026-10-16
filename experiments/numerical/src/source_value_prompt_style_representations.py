from __future__ import annotations

import argparse
import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.activation_steering import collect_prompt_activations, resolve_token_idx
from src.model_loader import load_causal_lm, load_tokenizer
from src.numeric_prompt_style_pilot import (
    PROMPT_STYLES,
    PromptStyle,
    PromptVariant,
    _format_target,
    _render_query,
    target_to_numeric,
)
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_jsonl, ensure_dir, load_yaml, set_global_seed


DEFAULT_CONTEXT_VARIANTS = 20


@dataclass(frozen=True)
class SelectedPromptSetting:
    style_id: str
    variant_id: str
    fewshot_k: int


DEFAULT_SELECTED_SETTINGS = (
    SelectedPromptSetting("successor", "add_one", 0),
    SelectedPromptSetting("interval_bucket", "hundreds_band", 3),
    SelectedPromptSetting("rounding_nearest_hundred", "hundred_rounding", 3),
    SelectedPromptSetting("magnitude_comparison", "above_threshold", 3),
    SelectedPromptSetting("semantic_affordability", "meets_minimum", 3),
)


CONTEXT_PREAMBLES = (
    "",
    "Answer the numeric query directly.\n\n",
    "Complete the following number task.\n\n",
    "Use decimal notation for the answer.\n\n",
    "Read the question carefully and answer briefly.\n\n",
    "Solve this simple arithmetic or comparison item.\n\n",
    "Give only the requested answer.\n\n",
    "Consider the value in the final question.\n\n",
    "The final line is the item to answer.\n\n",
    "Use the examples only as demonstrations.\n\n",
    "Respond with the minimal answer format.\n\n",
    "Work with the stated base-10 number.\n\n",
    "Treat the value as an ordinary integer.\n\n",
    "Follow the same rule for the query.\n\n",
    "Answer according to the requested criterion.\n\n",
    "Use the target rule shown by the prompt.\n\n",
    "Focus on the number in the question.\n\n",
    "Return the answer, not an explanation.\n\n",
    "Apply the instruction to the final value.\n\n",
    "Keep the response concise.\n\n",
)


FEWSHOT_CONTEXT_VALUES = (
    (3, 58, 241),
    (14, 206, 777),
    (42, 365, 964),
    (7, 509, 812),
    (99, 500, 901),
    (128, 256, 512),
    (19, 137, 486),
    (73, 314, 888),
    (5, 404, 990),
    (250, 501, 749),
    (31, 470, 930),
    (111, 333, 666),
    (24, 420, 840),
    (68, 275, 602),
    (150, 450, 750),
    (11, 222, 999),
    (87, 543, 876),
    (1, 499, 998),
    (200, 600, 900),
    (16, 160, 610),
)


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


def _style_by_id(style_id: str) -> PromptStyle:
    for style in PROMPT_STYLES:
        if style.style_id == style_id:
            return style
    raise KeyError(f"Unknown prompt style: {style_id!r}")


def _variant_by_id(style: PromptStyle, variant_id: str) -> PromptVariant:
    for variant in style.variants:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(f"Unknown variant {variant_id!r} for style {style.style_id!r}")


def _format_selected_setting(payload: dict[str, Any]) -> SelectedPromptSetting:
    return SelectedPromptSetting(
        style_id=str(payload["style_id"]),
        variant_id=str(payload["variant_id"]),
        fewshot_k=int(payload.get("fewshot_k", 0)),
    )


def selected_settings_from_config(cfg: dict[str, Any]) -> tuple[SelectedPromptSetting, ...]:
    raw = cfg.get("selected_settings")
    if raw is None:
        return DEFAULT_SELECTED_SETTINGS
    return tuple(_format_selected_setting(item) for item in raw)


def _context_values(context_variant_idx: int, query_value: int, fewshot_k: int) -> tuple[int, ...]:
    if int(fewshot_k) <= 0:
        return ()
    values = list(FEWSHOT_CONTEXT_VALUES[int(context_variant_idx) % len(FEWSHOT_CONTEXT_VALUES)])
    cursor = 0
    while len(values) < int(fewshot_k):
        candidate = (37 * int(context_variant_idx) + 101 * cursor + 17) % 1000
        values.append(int(candidate))
        cursor += 1
    cleaned: list[int] = []
    used = {int(query_value)}
    for value in values:
        candidate = int(value)
        while candidate in used:
            candidate = (candidate + 1) % 1000
        used.add(candidate)
        cleaned.append(candidate)
        if len(cleaned) == int(fewshot_k):
            break
    return tuple(cleaned)


def _render_prompt_with_source_span(
    *,
    style: PromptStyle,
    variant: PromptVariant,
    value: int,
    fewshot_k: int,
    context_variant_idx: int,
) -> tuple[str, tuple[int, int], tuple[int, ...]]:
    preamble = CONTEXT_PREAMBLES[int(context_variant_idx) % len(CONTEXT_PREAMBLES)]
    examples: list[str] = []
    example_values = _context_values(int(context_variant_idx), int(value), int(fewshot_k))
    for example_value in example_values:
        example_query = _render_query(variant.template, int(example_value))
        example_target = _format_target(style, style.target(int(example_value)))
        examples.append(f"Q: {example_query}\nA: {example_target}")
    query = _render_query(variant.template, int(value))
    prompt_body = f"Q: {query}\nA:" if not examples else "\n\n".join(examples) + f"\n\nQ: {query}\nA:"
    prompt = preamble + prompt_body
    final_q_marker = "\n\nQ: " if examples else "Q: "
    final_q_start = prompt.rfind(final_q_marker)
    if final_q_start < 0:
        raise ValueError("Could not locate final query in rendered prompt.")
    query_start = final_q_start + len(final_q_marker)
    value_text = str(int(value))
    relative = query.find(value_text)
    if relative < 0:
        raise ValueError(f"Could not locate source value {value_text!r} in final query {query!r}.")
    source_start = query_start + relative
    source_end = source_start + len(value_text)
    return prompt, (source_start, source_end), example_values


def build_prompt_style_representation_frame(
    *,
    tokenizer: Any | None = None,
    value_min: int = 0,
    value_max: int = 999,
    selected_settings: tuple[SelectedPromptSetting, ...] = DEFAULT_SELECTED_SETTINGS,
    context_variants: int = DEFAULT_CONTEXT_VARIANTS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting in selected_settings:
        style = _style_by_id(setting.style_id)
        variant = _variant_by_id(style, setting.variant_id)
        for source_value in range(int(value_min), int(value_max) + 1):
            target = style.target(int(source_value))
            for context_variant_idx in range(int(context_variants)):
                prompt, source_span, example_values = _render_prompt_with_source_span(
                    style=style,
                    variant=variant,
                    value=int(source_value),
                    fewshot_k=int(setting.fewshot_k),
                    context_variant_idx=int(context_variant_idx),
                )
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
                        "style_label": style.label,
                        "variant_id": variant.variant_id,
                        "variant_idx": int(context_variant_idx),
                        "context_variant_idx": int(context_variant_idx),
                        "fewshot_k": int(setting.fewshot_k),
                        "source_value": int(source_value),
                        "target": _format_target(style, target),
                        "target_numeric": target_to_numeric(style, target),
                        "output_kind": style.output_kind,
                        "mae_kind": style.mae_kind,
                        "template": variant.template,
                        "fewshot_values": ",".join(str(int(v)) for v in example_values),
                        "context_preamble": CONTEXT_PREAMBLES[int(context_variant_idx) % len(CONTEXT_PREAMBLES)].strip(),
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
        frame.groupby(["style_id", "style_label", "variant_id", "fewshot_k"], as_index=False)
        .agg(
            num_rows=("prompt", "size"),
            num_source_values=("source_value", "nunique"),
            context_variants_min=("context_variant_idx", lambda s: int(s.groupby(frame.loc[s.index, "source_value"]).nunique().min())),
            context_variants_max=("context_variant_idx", lambda s: int(s.groupby(frame.loc[s.index, "source_value"]).nunique().max())),
            target_min=("target_numeric", "min"),
            target_max=("target_numeric", "max"),
        )
        .sort_values(["style_id", "variant_id", "fewshot_k"])
        .reset_index(drop=True)
    )
    atomic_write_csv(output_dir / "dataset_summary.csv", summary)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "value_min": int(frame["source_value"].min()),
            "value_max": int(frame["source_value"].max()),
            "num_rows": int(len(frame)),
            "num_styles": int(frame["style_id"].nunique()),
            "context_variants": int(frame["context_variant_idx"].nunique()),
            "settings": summary.to_dict(orient="records"),
        },
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
    npz_path = repr_dir / "source_token_representations.npz"
    _atomic_save_npz(
        npz_path,
        activations=activations_to_save,
        source_values=prompts["source_value"].to_numpy(dtype=np.int64),
        variant_indices=prompts["context_variant_idx"].to_numpy(dtype=np.int64),
        context_variant_indices=prompts["context_variant_idx"].to_numpy(dtype=np.int64),
        style_ids=prompts["style_id"].to_numpy(dtype=np.str_),
        variant_ids=prompts["variant_id"].to_numpy(dtype=np.str_),
        fewshot_ks=prompts["fewshot_k"].to_numpy(dtype=np.int64),
        token_indices=prompts["token_idx"].to_numpy(dtype=np.int64),
    )
    atomic_write_csv(repr_dir / "rows.csv", prompts)
    atomic_write_json(
        repr_dir / "metadata.json",
        {
            "model_alias": str(model_alias),
            "model_name": str(model_entry["hf_id"]),
            "model_revision": str(loaded.model_revision),
            "layer_idx": int(layer_idx),
            "num_rows": int(len(prompts)),
            "num_styles": int(prompts["style_id"].nunique()),
            "num_source_values": int(prompts["source_value"].nunique()),
            "context_variants": int(prompts["context_variant_idx"].nunique()),
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
    return npz_path


def run(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("seed", 1729)))
    output_dir = ensure_dir(Path(args.output_dir or cfg.get("output_dir", "out/runs/source_value_prompt_style_representations__2026_05_13")))
    model_cfg = load_yaml(cfg.get("models_config", "configs/models.yaml"))
    tokenizer_model_alias = str(cfg.get("tokenizer_model", cfg.get("models", ["llama_3_2_3b"])[0]))
    tokenizer_entry = {**model_cfg.get("default", {}), **model_cfg["models"][tokenizer_model_alias]}
    tokenizer = load_tokenizer(
        str(tokenizer_entry["hf_id"]),
        trust_remote_code=bool(tokenizer_entry.get("trust_remote_code", False)),
    )
    frame = build_prompt_style_representation_frame(
        tokenizer=tokenizer,
        value_min=int(cfg.get("value_min", 0)),
        value_max=int(cfg.get("value_max", 999)),
        selected_settings=selected_settings_from_config(cfg),
        context_variants=int(cfg.get("context_variants", DEFAULT_CONTEXT_VARIANTS)),
    )
    write_dataset(frame, output_dir / "dataset")
    repr_cfg = cfg.get("representations", {})
    if bool(repr_cfg.get("enabled", True)):
        repr_frame = frame
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
            "context_variants": int(frame["context_variant_idx"].nunique()),
        },
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect source-value activations for selected numeric prompt styles.")
    parser.add_argument("--config", default="configs/source_value_prompt_style_representations_5styles_0_999.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    print(run(build_arg_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
