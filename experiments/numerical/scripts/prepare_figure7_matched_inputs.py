#!/usr/bin/env python3
"""Freeze prompts, legacy replacements, splits and checkpoint bytes before runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.figure7_matched_probes import content_sha256, file_sha256, replace_source, validate_partitions, value_centroids
from src.source_value_rhs_addition_dataset import build_rhs_addition_frame
from src.source_value_steering import sample_stratified_target_values
from src.utils import atomic_write_json, atomic_write_jsonl


def prompt_id(row):
    return content_sha256({k: row[k] for k in ("prompt", "token_idx", "source_value")})[:20]


def cases_for(frame, bank, tokenizer, legacy=None):
    rows = []
    for i, source in enumerate(frame.to_dict("records")):
        auxiliary = int(source["x1"] if source["operand_mode"] == "x2" else source["x2"])
        targets = sample_stratified_target_values(source_value=source["source_value"],
                    valid_target_values=np.arange(1000 - auxiliary), num_targets=20, seed=1729 + 1000003 * i)
        if legacy is not None:
            subset = legacy.loc[legacy["prompt"].eq(source["prompt"])].sort_values("target_slot")
            assert len(subset) == 20
            targets = subset["target_value"].astype(int).tolist()
        for slot, target in enumerate(targets):
            shifted, idx = replace_source(source, target, tokenizer)
            if legacy is not None:
                old = subset.iloc[slot]
                assert shifted == old["shifted_prompt"] and idx == old["shifted_token_idx"]
                assert auxiliary + target == old["shifted_target"]
            rows.append({**source, "evaluation_bank": bank, "target_slot": slot, "target_value": target,
                         "shifted_target": auxiliary + target, "shifted_prompt": shifted,
                         "shifted_token_idx": idx,
                         "shifted_input_ids": tokenizer.encode(shifted, add_special_tokens=False),
                         "shifted_target_token_id": tokenizer.encode(str(auxiliary + target), add_special_tokens=False)[0],
                         "case_id": content_sha256([bank, source["source_prompt_id"], target])[:24]})
    return pd.DataFrame(rows).sort_values(["target_slot", "source_value"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "out/figure7-matched-20261015/inputs")
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise ValueError("Inputs are frozen; select a fresh directory for a revised protocol.")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    split_dir = ROOT / "data/prompt_cache_canonical_source_splits_t800_d100_te100__a68a297dd188__r0/split_00"
    raw_split = json.loads((split_dir / "source_values.json").read_text())
    split = {p: raw_split[f"{p}_source_values"] for p in ("train", "dev", "test")}
    validate_partitions(split)
    fields = ["source_value", "prompt", "token_idx", "x1", "x2", "target", "operand_mode", "partition"]
    canonical = []
    for part in ("train", "dev", "test"):
        rows = pd.read_json(split_dir / "addition_formula" / f"{part}.jsonl", lines=True).sort_values("source_value")
        rows["partition"] = part
        assert set(rows["source_value"]) == set(split[part])
        canonical.append(rows[fields])
    canonical = pd.concat(canonical, ignore_index=True)
    canonical["input_ids"] = tokenizer(canonical["prompt"].tolist(), add_special_tokens=False)["input_ids"]
    rhs = build_rhs_addition_frame(tokenizer=None)
    rhs = rhs.loc[rhs["style_id"].eq("addition_formula_rhs_source")].copy().reset_index(drop=True)
    encoded = tokenizer(rhs["prompt"].tolist(), add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    for row, offsets in zip(rhs.to_dict("records"), encoded["offset_mapping"], strict=True):
        matches = [i for i, span in enumerate(offsets) if tuple(span) == (row["source_char_start"], row["source_char_end"])]
        assert len(matches) == 1
        positions.append(matches[0])
    rhs["token_idx"], rhs["operand_mode"] = positions, "x2"
    inverse = {v: p for p, values in split.items() for v in values}
    rhs["partition"] = rhs["source_value"].map(inverse)
    rhs = rhs[fields + ["variant_idx", "x"]]
    rhs["input_ids"] = encoded["input_ids"]
    for frame in (canonical, rhs):
        frame["source_prompt_id"] = [prompt_id(r) for r in frame.to_dict("records")]
    atomic_write_jsonl(args.output / "canonical_rows.jsonl", canonical.to_dict("records"))
    atomic_write_jsonl(args.output / "rhs_rows.jsonl", rhs.to_dict("records"))
    legacy_path = ROOT / "out/runs/source_value_probe_manifold_steering__9bbad17c01ee__r0/details.csv"
    legacy = pd.read_csv(legacy_path)
    legacy = legacy.loc[legacy["architecture"].eq("linear") & legacy["method"].eq("fixed") & legacy["delta_kind"].eq("value_code")]
    assert len(legacy) == 2000
    selected_rhs = []
    rng = np.random.default_rng(1729)
    for value in sorted(split["test"]):
        candidates = rhs.loc[rhs["source_value"].eq(value) & rhs["target"].le(999) & rhs["x"].le(979)]
        selected_rhs.append(candidates.iloc[int(rng.integers(len(candidates)))].to_dict())
    counts = {}
    for bank, frame in (("rhs_test", pd.DataFrame(selected_rhs)),
                        ("canonical_dev", canonical.loc[canonical["partition"].eq("dev")]),
                        ("canonical_test", canonical.loc[canonical["partition"].eq("test")])):
        cases = cases_for(frame.reset_index(drop=True), bank, tokenizer, legacy=legacy if bank == "canonical_dev" else None)
        assert len(cases) == 2000 and cases["source_prompt_id"].nunique() == 100
        assert cases["case_id"].nunique() == 2000 and cases["shifted_target"].between(0, 999).all()
        atomic_write_jsonl(args.output / f"cases_{bank}.jsonl", cases.to_dict("records"))
        counts[bank] = {"source_prompts": 100, "interventions": len(cases)}
    archive = ROOT / "out/runs/source_value_rhs_addition_dataset_llama_3_1_8b_layer10__2026_06_15/representations/llama_3_1_8b/layer_10/source_token_representations.npz"
    with np.load(archive) as old:
        mask = old["style_ids"] == "addition_formula_rhs_source"
        np.save(args.output / "archived_centroids.npy", value_centroids(old["activations"][mask], old["source_values"][mask]))
    code_files = [p for p in sorted((ROOT / "src").glob("*.py"))] + [Path(__file__), ROOT / "scripts/run_figure7_matched_probes.py"]
    manifest = {"version": 1, "seed": 1729, "split": split, "counts": counts,
                "rhs_rows": len(rhs), "canonical_rows": len(canonical), "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                "files": {p.name: file_sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},
                "model_files": {str(p.relative_to(args.model_path)): file_sha256(p) for p in sorted(args.model_path.rglob('*')) if p.is_file()},
                "code_files": {str(p.relative_to(ROOT)): file_sha256(p) for p in code_files},
                "historical_sources": {str(legacy_path.relative_to(ROOT)): file_sha256(legacy_path), str(archive.relative_to(ROOT)): file_sha256(archive)}}
    atomic_write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "passed", "rhs_rows": len(rhs), "canonical_rows": len(canonical), "counts": counts,
                      "manifest_sha256": file_sha256(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
