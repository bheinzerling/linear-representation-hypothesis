#!/usr/bin/env python3
"""Freeze prompts and train-only controls before matched GPU evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import SOURCE_RUN, load_source_config
from scripts.run_property_probe_training_scheme_comparison import ALL_PROPERTIES, _load_split, _method_training
from scripts.run_property_residual_behavior import DEFAULT_SCHEME_OUTPUT, build_task_frames
from src.model_loader import load_tokenizer
from src.number_residual_analysis import deterministic_random_basis
from src.property_subspace_synthesis import fit_property_feature_encoder
from src.rank41_deletion import TASKS, conditions, digest, file_sha256, stable_seed
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text

DEFAULT_BUNDLE = ROOT / "out/rank41_deletion_inputs/20260915_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg, entry, resolved = load_source_config(SOURCE_RUN)
    tokenizer = load_tokenizer(entry["hf_id"], revision=resolved["model_revision"])
    recipe = {
        "schema": 1, "seed": 20260915, "source_config_sha256": file_sha256(SOURCE_RUN / "config.json"),
        "model": entry["hf_id"], "revision": resolved["model_revision"],
        "dtype": "float16", "attention": "eager", "layer_idx": int(cfg["intervention_layer_idx"]),
        "basis_key": "full_basis__competitive_deflation", "tasks": TASKS,
        "conditions": [asdict(c) for c in conditions()],
        "source_files": {
            str(p.relative_to(ROOT)): file_sha256(p) for p in (
                Path(__file__), ROOT / "src/rank41_deletion.py",
                ROOT / "src/property_probe_training_schemes.py",
                ROOT / "scripts/run_property_probe_training_scheme_comparison.py",
                ROOT / "scripts/run_property_residual_behavior.py",
                ROOT / "src/source_value_prompt_style_representations.py",
                ROOT / "src/numeric_prompt_style_pilot.py",
            )
        },
    }
    recipe_hash = digest(recipe)
    for i in range(10):
        split = f"split_{i:02d}"
        state_path = DEFAULT_SCHEME_OUTPUT / "states" / f"{split}.npz"
        state_sha = file_sha256(state_path)
        done = output / f"{split}.json"
        if done.exists():
            cached = json.loads(done.read_text())
            if cached["recipe_hash"] != recipe_hash or cached["state_sha256"] != state_sha:
                raise ValueError(f"Prepared inputs changed for {split}; use a fresh output directory")
            if all(file_sha256(output / name) == sha for name, sha in cached["files"].items()):
                print(f"reuse {split}", flush=True)
                continue
            raise ValueError(f"Corrupt inputs for {split}")
        print(f"prepare {split}: load frozen training activations", flush=True)
        activations, rows = _load_split(split_name=split, cfg=cfg, model_entry=entry,
                                       resolved=resolved, tokenizer=tokenizer)
        values, train_raw = activations["train"]
        train = np.asarray(train_raw, dtype=np.float64)
        mean = train.mean(0)
        centered = train - mean
        state = np.load(state_path)
        np.testing.assert_allclose(state["train_center"], mean.astype(np.float32), atol=0, rtol=0)
        np.testing.assert_array_equal(state["values"], activations["test"][0])
        groups = {k: set(map(int, v[0])) for k, v in activations.items()}
        if any(groups[a] & groups[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError("Train/dev/test value leakage")
        if [len(groups[k]) for k in ("train", "dev", "test")] != [800, 100, 100]:
            raise ValueError("Unexpected mainline split sizes")
        property_basis = np.asarray(state["full_basis__competitive_deflation"], dtype=np.float64)
        if property_basis.shape != (4096, 41):
            raise ValueError(f"Expected frozen rank 41: {property_basis.shape}")
        bases = {"property": property_basis}
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        energy = float(np.sum((centered @ property_basis) ** 2))
        pca_rank = int(np.argmin(np.abs(np.cumsum(singular ** 2) - energy))) + 1
        bases["pca41"] = right[:41].T
        bases["pca_energy"] = right[:pca_rank].T
        encoder = fit_property_feature_encoder(train_values=values.astype(np.int64), property_ids=ALL_PROPERTIES)
        targets = encoder.transform(values.astype(np.int64))
        control_seeds = {}
        shuffled_diagnostics = []
        for seed_idx in range(3):
            random_seed = stable_seed(split, "random", seed_idx)
            bases[f"random_{seed_idx}"] = deterministic_random_basis(hidden_dim=4096, rank=41, seed=random_seed)
            permutation_seed = stable_seed(split, "shuffled", seed_idx)
            permutation = np.random.default_rng(permutation_seed).permutation(len(targets))
            print(f"prepare {split}: shuffled-label control {seed_idx}", flush=True)
            trained = _method_training(
                method="competitive_deflation", train_centered=centered, train_targets=targets[permutation],
                dev_centered=np.asarray(activations["dev"][1], dtype=np.float64) - mean,
                dev_targets=encoder.transform(activations["dev"][0].astype(np.int64)),
                encoder=encoder, max_rank=41, split_name=split,
            )
            if trained.basis.shape != property_basis.shape:
                raise ValueError("Shuffled-label control did not reach matched rank")
            bases[f"shuffled_{seed_idx}"] = trained.basis
            control_seeds[str(seed_idx)] = {"random": random_seed, "permutation": permutation_seed,
                                           "permuted_train_values": values[permutation].astype(int).tolist()}
            shuffled_diagnostics.extend({"seed_index": seed_idx, **r} for r in trained.diagnostics)
        diagnostics = []
        for name, basis in bases.items():
            gram_error = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
            if gram_error > 1e-6:
                raise ValueError(f"Nonorthogonal basis {name}: {gram_error}")
            diagnostics.append({
                "split": split, "basis": name, "rank": basis.shape[1], "gram_error": gram_error,
                "train_removed_energy_fraction": float(np.sum((centered @ basis) ** 2) / np.sum(centered ** 2)),
                "property_overlap_fraction": float(np.sum((property_basis.T @ basis) ** 2) / 41),
            })
        frames, audits = build_task_frames(
            values=state["values"].astype(np.int64),
            canonical_frame=rows["test"].sort_values("source_value").reset_index(drop=True),
            tokenizer=tokenizer, selected_tasks=TASKS, max_examples_per_task=None,
        )
        all_frames = []
        for task, frame in frames.items():
            frame = frame.copy()
            frame.insert(0, "split", split)
            token_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in frame["prompt"]]
            frame["input_ids"] = token_ids
            frame["source_token_id"] = [ids[int(idx)] for ids, idx in zip(token_ids, frame["token_idx"])]
            # Value 0..999 must occupy one source-number token in this mainline tokenizer.
            decoded = [tokenizer.decode([tok]).strip() for tok in frame["source_token_id"]]
            if decoded != frame["source_value"].astype(int).astype(str).tolist():
                raise ValueError(f"Source token mismatch in {split}/{task}")
            all_frames.append(frame)
        prompt_path = output / f"{split}.prompts.jsonl"
        atomic_write_text(prompt_path, pd.concat(all_frames).to_json(orient="records", lines=True))
        basis_path = output / f"{split}.npz"
        temp = output / f"{split}.tmp.npz"
        np.savez_compressed(temp, center=state["train_center"], train_values=values,
                            dev_values=activations["dev"][0], test_values=state["values"], **bases)
        temp.replace(basis_path)
        diagnostic_path = output / f"{split}.controls.csv"
        atomic_write_csv(diagnostic_path, pd.DataFrame(diagnostics))
        atomic_write_json(done, {
            "recipe_hash": recipe_hash, "state_sha256": state_sha, "state_path": str(state_path.relative_to(ROOT)),
            "split": split, "control_seeds": control_seeds, "task_audit": audits,
            "training_values_sha256": digest(values.astype(int).tolist()),
            "training_activations_sha256": hashlib_array(train_raw),
            "shuffled_fit_diagnostics": shuffled_diagnostics,
            "files": {p.name: file_sha256(p) for p in (basis_path, prompt_path, diagnostic_path)},
        })
        print(f"prepared {split}: {len(pd.concat(all_frames))} prompts, energy PCA rank {pca_rank}", flush=True)
    files = {p.name: file_sha256(p) for p in sorted(output.glob("split_*"))}
    manifest = {"recipe": recipe, "files": files}
    manifest["bundle_hash"] = digest(manifest)
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_text(output / "DONE", manifest["bundle_hash"] + "\n")
    print(json.dumps({"bundle": str(output), "bundle_hash": manifest["bundle_hash"]}), flush=True)


def hashlib_array(array):
    import hashlib
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


if __name__ == "__main__":
    main()
