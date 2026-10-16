from __future__ import annotations

import hashlib
import ast
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.activation_steering import render_prompt_with_operand_spans, resolve_token_idx


@dataclass(frozen=True)
class IdentityPromptConfig:
    num_examples: int = 4
    operator: str = " = "
    separator: str = "; "

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "num_examples": int(self.num_examples),
            "operator": str(self.operator),
            "separator": str(self.separator),
        }


@dataclass(frozen=True)
class ArithmeticPromptConfig:
    num_examples: int = 0
    separator: str = "; "

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "num_examples": int(self.num_examples),
            "separator": str(self.separator),
        }


def identity_prompt_config_from_mapping(spec: Mapping[str, Any] | None = None) -> IdentityPromptConfig:
    payload = dict(spec or {})
    return IdentityPromptConfig(
        num_examples=int(payload.get("identity_num_examples", 4)),
        operator=str(payload.get("identity_operator", " = ")),
        separator=str(payload.get("identity_separator", "; ")),
    )


def arithmetic_prompt_config_from_mapping(spec: Mapping[str, Any] | None = None) -> ArithmeticPromptConfig:
    payload = dict(spec or {})
    return ArithmeticPromptConfig(
        num_examples=int(payload.get("arithmetic_num_examples", 0)),
        separator=str(payload.get("arithmetic_separator", "; ")),
    )


def _stable_rng_seed(payload: dict[str, Any]) -> int:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def sample_identity_example_values(
    *,
    candidate_values: np.ndarray,
    query_value: int,
    num_examples: int,
    seed: int,
    namespace: str,
    evaluation_split: str,
    context_idx: int,
) -> list[int]:
    if int(num_examples) <= 0:
        return []
    candidate_arr = np.asarray(candidate_values, dtype=np.int64).reshape(-1)
    pool = np.unique(candidate_arr[candidate_arr != int(query_value)])
    if len(pool) == 0:
        raise ValueError(f"No candidate identity examples available for query value {query_value}.")
    rng = np.random.default_rng(
        _stable_rng_seed(
            {
                "seed": int(seed),
                "namespace": str(namespace),
                "evaluation_split": str(evaluation_split),
                "context_idx": int(context_idx),
                "query_value": int(query_value),
            }
        )
    )
    replace = len(pool) < int(num_examples)
    sampled = rng.choice(pool, size=int(num_examples), replace=replace)
    return [int(value) for value in np.asarray(sampled, dtype=np.int64).tolist()]


def render_identity_prompt(
    *,
    value: int,
    tokenizer,
    example_values: Sequence[int],
    operator: str = " = ",
    separator: str = "; ",
) -> tuple[str, int]:
    prefix = str(separator).join(f"{int(example_value)}{operator}{int(example_value)}" for example_value in example_values)
    value_text = str(int(value))
    target_fragment = f"{value_text}{operator}"
    prompt = f"{prefix}{separator}{target_fragment}" if prefix else target_fragment
    mention_start = len(prefix) + len(separator) if prefix else 0
    mention_end = mention_start + len(value_text)
    token_idx = resolve_token_idx(tokenizer=tokenizer, prompt=prompt, mention_span=(mention_start, mention_end), mode="mention_last")
    return prompt, int(token_idx)


def sample_arithmetic_example_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    query_row: Mapping[str, Any],
    num_examples: int,
    seed: int,
    namespace: str,
) -> list[dict[str, int]]:
    if int(num_examples) <= 0:
        return []

    query_source_value = int(query_row["source_value"])
    query_key = (
        int(query_row["source_value"]),
        int(query_row["context_idx"]),
        int(query_row["x1"]),
        int(query_row["x2"]),
        int(query_row["target"]),
    )
    pool: list[dict[str, int]] = []
    fallback_pool: list[dict[str, int]] = []
    for row in candidate_rows:
        compact = {
            "source_value": int(row["source_value"]),
            "context_idx": int(row["context_idx"]),
            "x1": int(row["x1"]),
            "x2": int(row["x2"]),
            "target": int(row["target"]),
        }
        row_key = (
            compact["source_value"],
            compact["context_idx"],
            compact["x1"],
            compact["x2"],
            compact["target"],
        )
        if row_key == query_key:
            continue
        fallback_pool.append(compact)
        if compact["source_value"] != query_source_value:
            pool.append(compact)
    selected_pool = pool if pool else fallback_pool
    if not selected_pool:
        raise ValueError("No candidate arithmetic examples available for few-shot prompt rendering.")
    selected_pool = sorted(
        selected_pool,
        key=lambda row: (row["source_value"], row["context_idx"], row["x1"], row["x2"], row["target"]),
    )
    rng = np.random.default_rng(
        _stable_rng_seed(
            {
                "seed": int(seed),
                "namespace": str(namespace),
                "evaluation_split": str(query_row["evaluation_split"]),
                "task": str(query_row["task"]),
                "operand_mode": str(query_row["operand_mode"]),
                "context_idx": int(query_row["context_idx"]),
                "source_value": int(query_row["source_value"]),
                "x1": int(query_row["x1"]),
                "x2": int(query_row["x2"]),
                "target": int(query_row["target"]),
            }
        )
    )
    replace = len(selected_pool) < int(num_examples)
    sampled_idx = rng.choice(len(selected_pool), size=int(num_examples), replace=replace)
    return [dict(selected_pool[int(idx)]) for idx in np.asarray(sampled_idx, dtype=np.int64).tolist()]


def render_arithmetic_prompt(
    *,
    template: str,
    x1: int,
    x2: int,
    operand_mode: str,
    tokenizer,
    example_rows: Sequence[Mapping[str, Any]] | None = None,
    separator: str = "; ",
) -> tuple[str, int]:
    rendered_examples: list[str] = []
    for row in example_rows or ():
        example_prompt, _ = render_prompt_with_operand_spans(str(template), int(row["x1"]), int(row["x2"]))
        rendered_examples.append(f"{example_prompt}{int(row['target'])}")
    query_prompt, query_spans = render_prompt_with_operand_spans(str(template), int(x1), int(x2))
    prefix = str(separator).join(rendered_examples)
    prompt = f"{prefix}{separator}{query_prompt}" if prefix else query_prompt
    mention_span = query_spans[str(operand_mode)]
    mention_offset = len(prefix) + len(str(separator)) if prefix else 0
    token_idx = resolve_token_idx(
        tokenizer=tokenizer,
        prompt=prompt,
        mention_span=(mention_span[0] + mention_offset, mention_span[1] + mention_offset),
        mode="mention_last",
    )
    return prompt, int(token_idx)


def render_arithmetic_prompt_from_row(
    *,
    row: Mapping[str, Any],
    x1: int,
    x2: int,
    tokenizer,
) -> tuple[str, int]:
    raw_example_rows = row.get("arithmetic_example_rows_json")
    if raw_example_rows is None:
        raw_example_rows = row.get("arithmetic_example_rows", "[]")
    if isinstance(raw_example_rows, str):
        try:
            example_rows = json.loads(raw_example_rows)
        except json.JSONDecodeError:
            example_rows = ast.literal_eval(raw_example_rows)
    else:
        example_rows = raw_example_rows
    return render_arithmetic_prompt(
        template=str(row["template"]),
        x1=int(x1),
        x2=int(x2),
        operand_mode=str(row["operand_mode"]),
        tokenizer=tokenizer,
        example_rows=example_rows,
        separator=str(row.get("arithmetic_separator", "; ")),
    )
