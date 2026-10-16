from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class NumberTokenRecord:
    value: int
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    is_single_token: bool


def encode_integer(tokenizer, value: int) -> list[int]:
    text = str(value)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return list(token_ids)


def decode_token_ids(tokenizer, token_ids: Iterable[int]) -> list[str]:
    return [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in token_ids]


def is_single_token_integer(tokenizer, value: int) -> bool:
    return len(encode_integer(tokenizer, value)) == 1


def build_single_token_report(tokenizer, start: int = 0, stop: int = 1000) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value in range(start, stop):
        token_ids = encode_integer(tokenizer, value)
        rows.append(
            {
                "value": value,
                "token_ids": token_ids,
                "token_texts": decode_token_ids(tokenizer, token_ids),
                "num_tokens": len(token_ids),
                "is_single_token": len(token_ids) == 1,
            }
        )
    return pd.DataFrame(rows)


def single_token_failures(tokenizer, start: int = 0, stop: int = 1000) -> list[int]:
    report = build_single_token_report(tokenizer, start=start, stop=stop)
    return report.loc[~report["is_single_token"], "value"].astype(int).tolist()


def render_prompt(template: str, x1: int, x2: int) -> str:
    return template.replace("x1", str(x1)).replace("x2", str(x2))
