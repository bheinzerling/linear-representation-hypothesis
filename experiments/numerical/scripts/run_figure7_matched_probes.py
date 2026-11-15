#!/usr/bin/env python3
"""Run the Figure 7 comparison using pinned inputs and shared probe fits."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.activation_steering import get_model_layers, hidden_states_hook
from src.figure7_matched_probes import (
    content_sha256, evaluate_decoding, file_sha256, requested_edit,
    summarize_behavior, validate_partitions, value_centroids,
)
from src.magnitude_intervention import fit_magnitude_probe
from src.model_loader import load_causal_lm
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text


REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
PROBE_SPECS = {"linear": {"ridge_lambda": 0.551881046550891}, "circular_reg": {"basis_dim": 10}}


@torch.inference_mode()
def forward_batch(model, tokenizer, frame, *, delta=None, replacement=None, layer_idx=10):
    if delta is not None and replacement is not None:
        raise ValueError("Choose additive edits or literal replacement.")
    encoded = tokenizer(frame["prompt"].tolist(), padding=True, add_special_tokens=False, return_tensors="pt")
    if "input_ids" in frame:
        for i, ids in enumerate(frame["input_ids"]):
            length = int(encoded["attention_mask"][i].sum())
            if encoded["input_ids"][i, :length].tolist() != list(ids):
                raise ValueError("Tokenizer output differs from the frozen token IDs.")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    positions = torch.tensor(frame["token_idx"].to_numpy(), device=device)
    if torch.any(positions < 0) or torch.any(positions >= encoded["attention_mask"].sum(1)):
        raise ValueError("Intervention token lies outside an unpadded prompt.")
    captured = {}

    class Edit:
        # The repository decorator adapts a bound hook method, including tuple outputs.
        @hidden_states_hook
        def __call__(self, module, args, kwargs, output):
            del module, args, kwargs
            rows = torch.arange(len(frame), device=output.device)
            result = output
            if delta is not None or replacement is not None:
                result = output.clone()
                if replacement is not None:
                    result[rows, positions] = torch.as_tensor(replacement, device=output.device, dtype=output.dtype)
                else:
                    result[rows, positions] += torch.as_tensor(delta, device=output.device, dtype=output.dtype)
            captured["states"] = result[rows, positions].detach().float().cpu().numpy()
            return result

    handle = get_model_layers(model)[int(layer_idx)].register_forward_hook(Edit(), with_kwargs=True)
    try:
        outputs = model(**encoded, use_cache=False)
        last = encoded["attention_mask"].sum(1) - 1
        logits = outputs.logits[torch.arange(len(frame), device=device), last].float()
        ids = logits.argmax(1).cpu().numpy()
        texts = tokenizer.batch_decode(ids[:, None], clean_up_tokenization_spaces=False)
    finally:
        handle.remove()
    return ids, texts, captured["states"]


def collect_states(model, tokenizer, frame, batch_size, *, path=None):
    if path is not None and path.exists():
        result = np.load(path)
        if result.shape != (len(frame), model.config.hidden_size):
            raise ValueError("Saved activation shape differs from the bound input.")
        return result
    parts = []
    for start in range(0, len(frame), batch_size):
        _, _, states = forward_batch(model, tokenizer, frame.iloc[start:start + batch_size])
        parts.append(states)
        if start % (batch_size * 20) == 0:
            print(f"[capture] {path.name if path else 'counterfactual'} {start}/{len(frame)}", flush=True)
    result = np.concatenate(parts)
    if path is not None:
        np.save(path, result)
    return result


def _numeric(text):
    text = str(text).strip()
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else None


def evaluate_method(model, tokenizer, cases, source_states, batch_size, *, delta=None, replacement=None):
    rows, edited_parts = [], []
    for start in range(0, len(cases), batch_size):
        batch = cases.iloc[start:start + batch_size]
        d = None if delta is None else delta[start:start + batch_size]
        r = None if replacement is None else replacement[start:start + batch_size]
        ids, texts, actual = forward_batch(model, tokenizer, batch, delta=d, replacement=r)
        edited_parts.append(actual)
        norms = np.linalg.norm(actual - source_states[start:start + len(batch)], axis=1)
        for i, row in enumerate(batch.to_dict("records")):
            value = _numeric(texts[i])
            rows.append({**row, "predicted_token_id": int(ids[i]), "predicted_text": texts[i],
                         "predicted_value": value, "exact": int(ids[i] == row["shifted_target_token_id"]),
                         "numeric_exact": int(value == row["shifted_target"]),
                         "within10": int(value is not None and abs(value - row["shifted_target"]) <= 10),
                         "source_answer_retained": int(value == row["target"]), "applied_edit_norm": float(norms[i])})
    return pd.DataFrame(rows), np.concatenate(edited_parts)


def input_audit(inputs, model_path):
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if file_sha256(inputs / name) != sha:
            raise ValueError(f"Changed input: {name}")
    for name, sha in manifest["model_files"].items():
        if file_sha256(model_path / name) != sha:
            raise ValueError(f"Changed model file: {name}")
    for name, sha in manifest["code_files"].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError(f"Changed bound code: {name}")
    split = manifest["split"]
    validate_partitions(split)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = input_audit(args.inputs, args.model_path)
    identity = content_sha256({"manifest": manifest, "batch_size": args.batch_size, "smoke": args.smoke,
                               "runner": file_sha256(__file__), "core": file_sha256(ROOT / "src/figure7_matched_probes.py")})
    args.output.mkdir(parents=True, exist_ok=True)
    binding = args.output / "identity.json"
    if binding.exists() and json.loads(binding.read_text())["identity"] != identity:
        raise ValueError("Output directory belongs to another scientific identity.")
    atomic_write_json(binding, {"identity": identity, "manifest": manifest, "revision": REVISION})
    if (args.output / "DONE").exists():
        print("[complete]", args.output)
        return
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    torch.manual_seed(1729)
    np.random.seed(1729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    test = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    assert torch.isfinite(test @ test).all()
    runtime = {"platform": platform.platform(), "architecture": platform.machine(), "python": sys.version,
               "prefix": sys.prefix, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "device_count": torch.cuda.device_count(),
               "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "pandas")},
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "batch_size": args.batch_size}
    atomic_write_json(args.output / "runtime.json", runtime)
    started = time.time()
    loaded = load_causal_lm(str(args.model_path), device="cuda", dtype="float16", attn_implementation="eager", revision=REVISION)
    model, tokenizer = loaded.model, loaded.tokenizer
    model.requires_grad_(False)
    tokenizer.padding_side = "right"
    assert get_model_layers(model)[10] is model.model.layers[9]
    assert all(len(tokenizer.encode(str(v), add_special_tokens=False)) == 1 for v in range(1000))
    rhs = pd.read_json(args.inputs / "rhs_rows.jsonl", lines=True)
    canonical = pd.read_json(args.inputs / "canonical_rows.jsonl", lines=True)
    split = manifest["split"]
    if args.smoke:
        # Qualify the complete production forward/edit path without changing the main split.
        cases = pd.read_json(args.inputs / "cases_canonical_test.jsonl", lines=True).iloc[:8].copy()
        original = collect_states(model, tokenizer, cases, args.batch_size)
        target = cases.copy()
        target["prompt"], target["token_idx"] = cases["shifted_prompt"], cases["shifted_token_idx"]
        target["input_ids"] = cases["shifted_input_ids"]
        donors = collect_states(model, tokenizer, target, args.batch_size)
        sham, actual = evaluate_method(model, tokenizer, cases, original, args.batch_size, delta=np.zeros_like(original))
        assert np.array_equal(actual.view(np.uint32), original.view(np.uint32))
        patch, actual = evaluate_method(model, tokenizer, cases, original, args.batch_size, replacement=donors)
        assert np.array_equal(actual.view(np.uint32), donors.view(np.uint32))
        atomic_write_json(args.output / "smoke.json", {"status": "passed", "cases": len(cases), "literal_patch_byte_equal": True,
                          "sham_byte_equal": True, "patch_exact": int(patch["exact"].sum()), "elapsed_seconds": time.time() - started})
        atomic_write_text(args.output / "DONE", "passed\n")
        return
    rhs_x = collect_states(model, tokenizer, rhs, args.batch_size, path=args.output / "rhs_activations.npy")
    canon_x = collect_states(model, tokenizer, canonical, args.batch_size, path=args.output / "canonical_activations.npy")
    centroids = value_centroids(rhs_x, rhs["source_value"].to_numpy())
    np.save(args.output / "centroids.npy", centroids)
    old_centroids = np.load(args.inputs / "archived_centroids.npy")
    atomic_write_json(args.output / "historical_activation_comparison.json", {
        "historical_revision_recorded": False, "new_revision": REVISION,
        "relative_l2": float(np.linalg.norm(centroids - old_centroids) / np.linalg.norm(old_centroids)),
        "max_absolute_difference": float(np.max(np.abs(centroids - old_centroids))),
        "note": "Historical revision and runtime were not fully recorded; the fresh matched comparison is primary."})
    train_mask = canonical["partition"].eq("train").to_numpy()
    fits = {}
    decoding_summaries, decoding_details, code_parts, fit_records = [], [], [], []
    for regime in ("centroid", "individual"):
        train_x = centroids[split["train"]] if regime == "centroid" else canon_x[train_mask]
        values = np.asarray(split["train"]) if regime == "centroid" else canonical.loc[train_mask, "source_value"].to_numpy()
        for architecture, spec in PROBE_SPECS.items():
            print(f"[fit] {regime} {architecture}", flush=True)
            probe = fit_magnitude_probe(architecture=architecture, train_x=train_x, train_values=values, config=spec)
            payload = probe.to_jsonable()
            fit_id = content_sha256(payload)
            fit_name = f"{regime}_{architecture}"
            atomic_write_json(args.output / f"probe_{fit_name}.json", payload)
            fits[fit_name] = (probe, fit_id)
            fit_records.append({"fit_name": fit_name, "fit_id": fit_id, "regime": regime, "architecture": architecture,
                                "training_rows": len(values), "training_values": len(np.unique(values))})
            for bank, frame, states in (("centroids", pd.DataFrame({"source_value": np.arange(1000)}), centroids),
                                       ("rhs_individual", rhs, rhs_x), ("canonical", canonical, canon_x)):
                for part in ("train", "dev", "test"):
                    mask = frame["source_value"].isin(split[part]).to_numpy()
                    ids = np.flatnonzero(mask)
                    detail, summary = evaluate_decoding(probe, states[mask], frame.loc[mask, "source_value"].to_numpy())
                    meta = {"fit_name": fit_name, "fit_id": fit_id, "regime": regime, "architecture": architecture,
                            "evaluation_bank": bank, "partition": part}
                    decoding_summaries.append({**meta, **summary})
                    detail["bank_row_idx"] = ids
                    for k, v in meta.items(): detail[k] = v
                    decoding_details.append(detail)
            for code_bank, code_x, code_values in (("centroids", centroids, np.arange(1000)),
                                                   ("canonical", canon_x, canonical["source_value"].to_numpy())):
                features = probe.predict_features(code_x)
                code_parts.append(pd.DataFrame({"fit_name": fit_name, "fit_id": fit_id, "evaluation_bank": code_bank,
                    "source_value": np.repeat(code_values, features.shape[1]),
                    "component": np.tile(np.arange(features.shape[1]), len(code_values)), "prediction": features.reshape(-1)}))
    atomic_write_json(args.output / "fits.json", {"fits": fit_records})
    atomic_write_csv(args.output / "decoding_summary.csv", pd.DataFrame(decoding_summaries))
    atomic_write_csv(args.output / "decoding_predictions.csv", pd.concat(decoding_details, ignore_index=True))
    atomic_write_csv(args.output / "predicted_codes.csv", pd.concat(code_parts, ignore_index=True))
    # The centroid controls intentionally include held-out target values.
    _, _, vh = np.linalg.svd(centroids.astype(float) - centroids.mean(0), full_matrices=False)
    pca = vh[:64]
    steering_summaries, steering_details, case_decoding = [], [], []
    for bank in ("rhs_test", "canonical_test", "canonical_dev"):
        print(f"[steer] {bank}", flush=True)
        cases = pd.read_json(args.inputs / f"cases_{bank}.jsonl", lines=True)
        source = collect_states(model, tokenizer, cases, args.batch_size, path=args.output / f"source_{bank}.npy")
        shifted = cases.copy()
        shifted["prompt"], shifted["token_idx"] = cases["shifted_prompt"], cases["shifted_token_idx"]
        shifted["input_ids"] = cases["shifted_input_ids"]
        donors = collect_states(model, tokenizer, shifted, args.batch_size, path=args.output / f"donor_{bank}.npy")
        directions = centroids[cases["target_value"].to_numpy()] - centroids[cases["source_value"].to_numpy()]
        methods = [("no_edit", "control", "control", None, None),
                   ("centroid_difference", "control", "control", directions, None),
                   ("pca64_centroid_difference", "control", "control", (directions @ pca.T) @ pca, None),
                   ("counterfactual_patch", "control", "control", None, donors)]
        for name, (probe, fit_id) in fits.items():
            decoded, summary = evaluate_decoding(probe, source, cases["source_value"].to_numpy())
            for col in ("case_id", "source_prompt_id", "target_value"): decoded[col] = cases[col].to_numpy()
            decoded["fit_name"], decoded["fit_id"], decoded["evaluation_bank"] = name, fit_id, bank
            case_decoding.append(decoded)
            for rule in ("label_difference", "prediction_corrected"):
                delta = requested_edit(probe, source, cases["source_value"].to_numpy(), cases["target_value"].to_numpy(), rule)
                methods.append((rule, name, fit_id, delta, None))
        for method, name, fit_id, delta, replacement in methods:
            print(f"[evaluate] {bank} {name} {method}", flush=True)
            part, actual = evaluate_method(model, tokenizer, cases, source, args.batch_size, delta=delta, replacement=replacement)
            meta = {"evaluation_bank": bank, "method": method, "fit_name": name, "fit_id": fit_id}
            summary = {**meta, **summarize_behavior(part)}
            if name in fits:
                probe = fits[name][0]
                features = probe.predict_features(actual)
                target_code = probe.encode_values(cases["target_value"].to_numpy())
                part["post_edit_probe_prediction"] = probe.predict_values(actual)
                part["post_edit_probe_target"] = part["post_edit_probe_prediction"].to_numpy() == cases["target_value"].to_numpy()
                part["post_edit_code_mse"] = np.mean((features - target_code) ** 2, axis=1)
                summary["post_edit_probe_target_accuracy"] = float(part["post_edit_probe_target"].mean())
                summary["post_edit_code_mse"] = float(part["post_edit_code_mse"].mean())
            for k, v in meta.items(): part[k] = v
            atomic_write_csv(args.output / f"behavior_{bank}_{name}_{method}.csv", part)
            steering_summaries.append(summary)
            steering_details.append(part)
            atomic_write_csv(args.output / "steering_summary.csv", pd.DataFrame(steering_summaries))
    atomic_write_csv(args.output / "steering_case_decoding.csv", pd.concat(case_decoding, ignore_index=True))
    atomic_write_csv(args.output / "steering_predictions.csv", pd.concat(steering_details, ignore_index=True))
    assert all(content_sha256(p.to_jsonable()) == fit_id for p, fit_id in fits.values())
    atomic_write_json(args.output / "completion.json", {"status": "passed", "identity": identity, "fits": len(fits),
                      "decoding_conditions": len(decoding_summaries), "steering_conditions": len(steering_summaries),
                      "steering_rows": sum(len(p) for p in steering_details), "elapsed_seconds": time.time() - started,
                      "checks": ["input and model SHA256", "disjoint source-value partitions", "same probe hashes before and after steering", "single-token numbers", "explicit block-9 hook"]})
    inventory = {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file() and p.name not in {"DONE", "output-inventory.json"}}
    atomic_write_json(args.output / "output-inventory.json", inventory)
    atomic_write_text(args.output / "DONE", identity + "\n")
    print("[complete]", args.output, flush=True)


if __name__ == "__main__":
    main()
