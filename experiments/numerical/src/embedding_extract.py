from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.token_utils import build_single_token_report
from src.utils import atomic_write_csv, atomic_write_json, ensure_dir, save_numpy, slugify_model_name


def get_input_embeddings(model) -> torch.Tensor:
    weight = model.get_input_embeddings().weight.detach()
    return weight


def extract_number_embeddings(tokenizer, model, start: int = 0, stop: int = 1000) -> dict[str, Any]:
    report = build_single_token_report(tokenizer, start=start, stop=stop)
    failures = report.loc[~report["is_single_token"], "value"].astype(int).tolist()
    if failures:
        raise ValueError(f"Tokenizer failed single-token requirement for values: {failures[:10]}")

    weight = get_input_embeddings(model)
    token_ids = report["token_ids"].map(lambda ids: int(ids[0])).to_numpy(dtype=np.int64)
    embeddings = weight[token_ids].to(torch.float32).cpu().numpy()
    return {
        "values": np.arange(start, stop, dtype=np.int64),
        "token_ids": token_ids,
        "embeddings": embeddings,
        "report": report,
        "metadata": {
            "num_values": stop - start,
            "d_model": int(embeddings.shape[1]),
            "model_vocab_size": int(weight.shape[0]),
        },
    }


def save_embedding_artifacts(
    output_dir: str | Path,
    model_name: str,
    extraction: dict[str, Any],
) -> dict[str, str]:
    model_dir = ensure_dir(Path(output_dir) / slugify_model_name(model_name))
    report_path = model_dir / "token_report.csv"
    token_ids_path = model_dir / "token_ids.npy"
    embeddings_path = model_dir / "embeddings.npy"
    metadata_path = model_dir / "metadata.json"

    atomic_write_csv(report_path, pd.DataFrame(extraction["report"]))
    save_numpy(token_ids_path, extraction["token_ids"])
    save_numpy(embeddings_path, extraction["embeddings"])
    atomic_write_json(
        metadata_path,
        {
            "model_name": model_name,
            **extraction["metadata"],
            "files": {
                "token_report": str(report_path),
                "token_ids": str(token_ids_path),
                "embeddings": str(embeddings_path),
            },
        },
    )
    return {
        "report": str(report_path),
        "token_ids": str(token_ids_path),
        "embeddings": str(embeddings_path),
        "metadata": str(metadata_path),
    }
