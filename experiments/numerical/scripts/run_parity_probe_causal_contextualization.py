#!/usr/bin/env python3
"""Compare parity directions on the historical joint-inconsistency surface."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
import transformers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text, ensure_dir  # noqa: E402


DEFAULT_GEOMETRY = (
    ROOT / "out/parity_probe_direction_contextualization/llama31_8b_layer10_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "out/parity_probe_causal_contextualization/llama31_8b_layer10_v1"
)
DEFAULT_TRANSCODER = ROOT.parent / "properties"
OLD_SOURCE_REL = (
    "out/parity_activation_steering/"
    "llama31_8b_original2shot_value_disjoint_v1_local"
)
OLD_JOINT_REL = "out/parity_joint_generation/llama31_8b_v1"
MODEL_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DEFAULT_SEED = 20260716
DEFAULT_DOSE = 7.738790988922119
DEFAULT_BOOTSTRAP_SAMPLES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transcoder-root", type=Path, default=DEFAULT_TRANSCODER)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--dose", type=float, default=DEFAULT_DOSE)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_transcoder_api(transcoder_root: Path):
    source = transcoder_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from introlm.experiments.parity_joint_generation import (  # type: ignore
        annotate_completion,
        build_joint_prompt_variants,
        summarize_completions,
    )

    return annotate_completion, build_joint_prompt_variants, summarize_completions


def _transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise ValueError("Expected a causal LM with model.layers transformer blocks.")
    return layers


def _query_metadata(
    tokenizer: Any,
    variant: Any,
    value: int,
) -> tuple[str, int, int, list[int]]:
    prefix, suffix = variant.pattern.split("{number}")
    numeral = str(int(value))
    text = prefix + numeral + suffix
    char_start = len(prefix)
    char_end = char_start + len(numeral)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if max(start, char_start) < min(end, char_end)
    ]
    if len(indices) != 1:
        raise ValueError(
            f"Query number {numeral!r} maps to token indices {indices}."
        )
    index = indices[0]
    return text, index, int(encoded["input_ids"][index]), list(encoded["input_ids"])


def _eos_ids(model: nn.Module, tokenizer: Any) -> set[int]:
    raw = getattr(model.generation_config, "eos_token_id", None)
    if raw is None:
        raw = tokenizer.eos_token_id
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {int(raw)}
    return {int(value) for value in raw}


def _scaled_direction_deltas(
    units: torch.Tensor,
    signs: torch.Tensor,
    *,
    dose: float,
    normalize_direction: bool,
) -> torch.Tensor:
    values = units
    if normalize_direction:
        values = values / values.norm(dim=1, keepdim=True)
    return signs[:, None] * float(dose) * values


@torch.inference_mode()
def generate_condition(
    *,
    model: nn.Module,
    tokenizer: Any,
    variant: Any,
    values: Sequence[int],
    condition: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    dose: float,
    annotate_completion: Callable[..., dict[str, Any]],
    direction_for_value: Callable[[int], np.ndarray] | None,
    intervention_layer: int = 9,
    normalize_direction: bool = True,
) -> list[dict[str, Any]]:
    layers = _transformer_layers(model)
    eos_ids = _eos_ids(model, tokenizer)
    pad_token_id = int(tokenizer.pad_token_id)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(values), int(batch_size)):
        batch_values = [int(value) for value in values[start : start + int(batch_size)]]
        metadata = [_query_metadata(tokenizer, variant, value) for value in batch_values]
        prompts = [item[0] for item in metadata]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded.pop("token_type_ids", None)
        sequence_length = int(encoded["input_ids"].shape[1])
        prompt_lengths = [len(item[3]) for item in metadata]
        positions = torch.tensor(
            [
                sequence_length - length + item[1]
                for length, item in zip(prompt_lengths, metadata, strict=True)
            ],
            dtype=torch.long,
            device=device,
        )
        for index, item in enumerate(metadata):
            observed = int(encoded["input_ids"][index, positions[index]].item())
            if observed != item[2]:
                raise ValueError(
                    f"Padded token mismatch for value {batch_values[index]}: "
                    f"expected {item[2]}, got {observed}."
                )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        handle = None
        if direction_for_value is not None:
            units = torch.from_numpy(
                np.stack(
                    [direction_for_value(value) for value in batch_values],
                    axis=0,
                ).astype(np.float32)
            ).to(device)
            signs = torch.tensor(
                [1.0 if value % 2 == 1 else -1.0 for value in batch_values],
                dtype=torch.float32,
                device=device,
            )
            deltas = _scaled_direction_deltas(
                units,
                signs,
                dose=float(dose),
                normalize_direction=normalize_direction,
            )

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                batch_indices = torch.arange(len(batch_values), device=hidden.device)
                patched[batch_indices, positions, :] = (
                    patched[batch_indices, positions, :].float() + deltas
                ).to(dtype=patched.dtype)
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            handle = layers[int(intervention_layer)].register_forward_hook(hook)
        try:
            outputs = model(**encoded, use_cache=True)
        finally:
            if handle is not None:
                handle.remove()

        attention_mask = encoded["attention_mask"]
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1]
        finished = torch.zeros(len(batch_values), dtype=torch.bool, device=device)
        generated: list[torch.Tensor] = []
        for step in range(int(max_new_tokens)):
            was_finished = finished.clone()
            next_tokens = next_logits.argmax(dim=-1)
            emitted = torch.where(
                was_finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )
            generated.append(emitted.cpu())
            if eos_ids:
                emitted_is_eos = torch.zeros_like(finished)
                for eos_id in eos_ids:
                    emitted_is_eos |= emitted == int(eos_id)
                finished |= emitted_is_eos
            if step + 1 == int(max_new_tokens) or bool(finished.all().item()):
                break
            attention_mask = torch.cat(
                [attention_mask, (~was_finished).long()[:, None]],
                dim=1,
            )
            outputs = model(
                input_ids=emitted[:, None],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1]

        generated_ids = torch.stack(generated, dim=1)
        completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for index, (value, completion) in enumerate(
            zip(batch_values, completions, strict=True)
        ):
            rows.append(
                {
                    "split": "test",
                    "variant_id": variant.variant_id,
                    "condition": condition,
                    "value": value,
                    "query_token_index": int(metadata[index][1]),
                    "query_token_id": int(metadata[index][2]),
                    "generated_text": completion,
                    "delta_norm": (
                        float(deltas[index].norm().item())
                        if direction_for_value is not None
                        else 0.0
                    ),
                    **annotate_completion(value=value, text=completion),
                }
            )
        del outputs, past_key_values, generated_ids
    return rows


def _outcome_category(row: Mapping[str, Any]) -> str:
    if bool(row["dissociated_identity_target_parity"]):
        return "identity_plus_target_parity"
    if bool(row["coherent_changed_target"]):
        return "coherent_changed_target"
    if bool(row["baseline_joint_correct"]):
        return "identity_plus_original_parity"
    if not bool(row["joint_parsed"]):
        return "incomplete_parse"
    if bool(row["internally_inconsistent"]):
        return "other_inconsistent"
    return "other_consistent"


def _conditional_paired(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_value = {int(row["value"]): row for row in baseline}
    candidate_by_value = {int(row["value"]): row for row in candidate}
    values = sorted(set(baseline_by_value) & set(candidate_by_value))
    eligible = [
        value for value in values if bool(baseline_by_value[value]["baseline_joint_correct"])
    ]
    dissociated = sum(
        bool(candidate_by_value[value]["dissociated_identity_target_parity"])
        for value in eligible
    )
    return {
        "baseline_joint_correct_count": len(eligible),
        "conditional_dissociation_count": dissociated,
        "conditional_dissociation_rate": (
            dissociated / len(eligible) if eligible else 0.0
        ),
    }


def _paired_similarity(
    *,
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    left = {int(row["value"]): row for row in reference}
    right = {int(row["value"]): row for row in candidate}
    values = np.asarray(sorted(set(left) & set(right)), dtype=np.int64)
    left_d = np.asarray(
        [bool(left[int(value)]["dissociated_identity_target_parity"]) for value in values],
        dtype=np.int64,
    )
    right_d = np.asarray(
        [bool(right[int(value)]["dissociated_identity_target_parity"]) for value in values],
        dtype=np.int64,
    )
    intersection = int(np.sum((left_d == 1) & (right_d == 1)))
    union = int(np.sum((left_d == 1) | (right_d == 1)))
    clusters = values // 10
    unique = np.unique(clusters)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(int(seed))
    differences: list[float] = []
    for _ in range(int(bootstrap_samples)):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([cluster_indices[cluster] for cluster in drawn])
        differences.append(float(np.mean(right_d[selected] - left_d[selected])))
    return {
        "n": len(values),
        "reference_dissociation_count": int(left_d.sum()),
        "candidate_dissociation_count": int(right_d.sum()),
        "dissociation_overlap_count": intersection,
        "dissociation_union_count": union,
        "dissociation_jaccard": intersection / union if union else 1.0,
        "dissociation_event_agreement": float(np.mean(left_d == right_d)),
        "outcome_category_agreement": float(
            np.mean(
                [
                    _outcome_category(left[int(value)])
                    == _outcome_category(right[int(value)])
                    for value in values
                ]
            )
        ),
        "exact_completion_agreement": float(
            np.mean(
                [
                    str(left[int(value)]["generated_text"])
                    == str(right[int(value)]["generated_text"])
                    for value in values
                ]
            )
        ),
        "candidate_minus_reference_dissociation_rate": float(
            np.mean(right_d - left_d)
        ),
        "difference_cluster_bootstrap_lower": float(
            np.quantile(differences, 0.025)
        ),
        "difference_cluster_bootstrap_median": float(
            np.quantile(differences, 0.5)
        ),
        "difference_cluster_bootstrap_upper": float(
            np.quantile(differences, 0.975)
        ),
    }


def _write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _load_historical_examples(path: Path, condition: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "test" and row.get("condition") == condition:
                result.append(row)
    return result


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    done_path = output_dir / "DONE"
    if done_path.exists() and not args.force:
        print(done_path)
        return
    started = time.time()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.use_deterministic_algorithms(True)

    annotate_completion, build_joint_prompt_variants, summarize_completions = (
        _load_transcoder_api(args.transcoder_root)
    )
    source_dir = args.transcoder_root / OLD_SOURCE_REL
    old_joint_dir = args.transcoder_root / OLD_JOINT_REL
    splits = json.loads((source_dir / "splits.json").read_text(encoding="utf-8"))
    values = [int(value) for value in splits["values"]["test"]]
    variants = build_joint_prompt_variants(splits["values"]["train"])
    selected = json.loads(
        (old_joint_dir / "selected_prompt.json").read_text(encoding="utf-8")
    )
    selected_variant = next(
        variant for variant in variants if variant.variant_id == selected["variant_id"]
    )
    if selected_variant.variant_id != "structured_10shot":
        raise RuntimeError(
            f"Historical selected prompt changed: {selected_variant.variant_id}."
        )

    direction_bundle_path = args.geometry_dir / "direction_bundle.npz"
    with np.load(direction_bundle_path, allow_pickle=False) as payload:
        direction_bundle = {
            key.removesuffix("__intervention"): np.asarray(value, dtype=np.float64)
            for key, value in payload.items()
            if key.endswith("__intervention")
        }
    import torch as torch_module

    old_directions = torch_module.load(
        source_dir / "selected_and_control_directions.pt",
        map_location="cpu",
        weights_only=False,
    )
    historical_random = np.asarray(old_directions["random_unit"], dtype=np.float64)

    def fixed(direction_id: str) -> Callable[[int], np.ndarray]:
        return lambda _value: direction_bundle[direction_id]

    def crossfit(prefix: str) -> Callable[[int], np.ndarray]:
        return lambda value: direction_bundle[f"{prefix}_{(int(value) % 10) // 2}"]

    condition_directions: dict[str, Callable[[int], np.ndarray] | None] = {
        "baseline": None,
        "historical_direct": fixed("historical_direct"),
        "historical_random": lambda _value: historical_random,
        "occurrence_ridge": fixed("occurrence_ridge"),
        "value_disjoint_raw_ridge": fixed("value_disjoint_raw_ridge"),
        "value_disjoint_whitened_ridge": fixed(
            "value_disjoint_whitened_ridge"
        ),
        "factorial_raw_all_pairs": fixed("factorial_raw_all_pairs"),
        "factorial_raw_crossfit": crossfit("factorial_raw_held_pair"),
        "factorial_reference_whitened_all_pairs": fixed(
            "factorial_reference_whitened_all_pairs"
        ),
        "factorial_reference_whitened_crossfit": crossfit(
            "factorial_reference_whitened_held_pair"
        ),
        "factorial_fold_whitened_crossfit": crossfit(
            "factorial_fold_whitened_held_pair"
        ),
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=_torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    by_condition: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for condition, direction_for_value in condition_directions.items():
        print(f"generating {condition}", flush=True)
        rows = generate_condition(
            model=model,
            tokenizer=tokenizer,
            variant=selected_variant,
            values=values,
            condition=condition,
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_new_tokens),
            device=device,
            dose=float(args.dose),
            annotate_completion=annotate_completion,
            direction_for_value=direction_for_value,
        )
        by_condition[condition] = rows
        all_rows.extend(rows)

    baseline = by_condition["baseline"]
    metric_rows: list[dict[str, Any]] = []
    for condition, rows in by_condition.items():
        summary = summarize_completions(rows)
        paired = (
            _conditional_paired(baseline, rows)
            if condition != "baseline"
            else {
                "baseline_joint_correct_count": int(
                    sum(bool(row["baseline_joint_correct"]) for row in baseline)
                ),
                "conditional_dissociation_count": 0,
                "conditional_dissociation_rate": 0.0,
            }
        )
        metric_rows.append(
            {
                "condition": condition,
                "variant_id": selected_variant.variant_id,
                **summary,
                **paired,
            }
        )

    reference = by_condition["historical_direct"]
    similarity_rows = [
        {
            "condition": condition,
            "reference_condition": "historical_direct",
            **_paired_similarity(
                reference=reference,
                candidate=rows,
                bootstrap_samples=int(args.bootstrap_samples),
                seed=int(args.seed) + index,
            ),
        }
        for index, (condition, rows) in enumerate(by_condition.items())
        if condition != "baseline"
    ]

    historical_baseline = _load_historical_examples(
        old_joint_dir / "examples.jsonl.gz", "baseline"
    )
    historical_direct = _load_historical_examples(
        old_joint_dir / "examples.jsonl.gz", "parity_steer"
    )
    reproduction = {
        "baseline": _paired_similarity(
            reference=historical_baseline,
            candidate=by_condition["baseline"],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed) + 1000,
        ),
        "historical_direct": _paired_similarity(
            reference=historical_direct,
            candidate=by_condition["historical_direct"],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed) + 1001,
        ),
    }

    atomic_write_csv(output_dir / "causal_metrics.csv", pd.DataFrame(metric_rows))
    atomic_write_csv(
        output_dir / "causal_similarity.csv", pd.DataFrame(similarity_rows)
    )
    _write_jsonl_gz(output_dir / "examples.jsonl.gz", all_rows)
    metadata = {
        "model": args.model,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "device": args.device,
        "seed": int(args.seed),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "intervention_transformer_block": 9,
        "intervention_hidden_state_index": 10,
        "dose_l2": float(args.dose),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "test_values": values,
        "test_value_count": len(values),
        "variant_id": selected_variant.variant_id,
        "condition_order": list(condition_directions),
        "crossfit_pair_rule": "held_pair=(value % 10) // 2",
        "direction_bundle": {
            "path": str(direction_bundle_path.resolve()),
            "sha256": _sha256(direction_bundle_path),
        },
        "historical_source": str(source_dir.resolve()),
        "historical_joint_run": str(old_joint_dir.resolve()),
        "historical_reproduction": reproduction,
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "cluster": "value // 10",
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    atomic_write_text(done_path, "complete\n")
    print(done_path)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
