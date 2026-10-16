from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class LoadedModel:
    model_name: str
    model_revision: str
    attn_implementation: str | None
    tokenizer: Any
    model: Any
    device: str


def resolve_dtype(dtype: str | None):
    if dtype is None:
        return None
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return TORCH_DTYPES[dtype]


def resolve_model_revision(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> str:
    if revision is not None:
        return str(revision)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision is None:
        resolved_revision = "unknown"
    return str(resolved_revision)


def load_tokenizer(model_name: str, trust_remote_code: bool = False, revision: str | None = None):
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    device: str = "cpu",
    dtype: str = "float32",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    revision: str | None = None,
):
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_dtype(dtype),
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if revision is not None:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code, revision=revision)
    resolved_revision = revision
    if resolved_revision is None:
        resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if resolved_revision is None:
        resolved_revision = resolve_model_revision(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    return LoadedModel(
        model_name=model_name,
        model_revision=str(resolved_revision),
        attn_implementation=attn_implementation,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
