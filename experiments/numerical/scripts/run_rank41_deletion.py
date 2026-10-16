#!/usr/bin/env python3
"""Run immutable, resumable shards of the matched rank-41 deletion experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rank41_deletion import (
    TASKS, RunConfig, conditions, digest, file_sha256, forward_last_logits,
    intervention_delta, next_token_metrics, stable_seed,
)
from src.utils import atomic_write_csv, atomic_write_json, atomic_write_text


def plan(args):
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    bundle_hash = manifest.pop("bundle_hash")
    if digest(manifest) != bundle_hash or (args.bundle / "DONE").read_text().strip() != bundle_hash:
        raise ValueError("Invalid/incomplete input manifest")
    for name, sha in manifest["files"].items():
        if file_sha256(args.bundle / name) != sha:
            raise ValueError(f"Changed prepared artifact {name}")
    source_files = {
        str(p.relative_to(ROOT)): file_sha256(p) for p in (
            Path(__file__), ROOT / "src/rank41_deletion.py", ROOT / "src/model_loader.py",
            ROOT / "src/activation_steering.py", ROOT / "src/gradient_steering.py",
        )
    }
    splits = tuple(f"split_{int(x):02d}" for x in args.splits.split(","))
    tasks = TASKS if args.tasks == "all" else tuple(args.tasks.split(","))
    if len(set(splits)) != len(splits) or any(t not in TASKS for t in tasks):
        raise ValueError("Invalid split/task selection")
    cfg = RunConfig(bundle_hash=bundle_hash, code_hash=digest(source_files), splits=splits,
                    tasks=tasks, batch_size=args.batch_size, max_examples=args.max_examples)
    run_dir = args.output_root / cfg.run_id
    return cfg, manifest, source_files, run_dir


def read_frame(bundle, split, task, cap, tokenizer):
    frame = pd.read_json(bundle / f"{split}.prompts.jsonl", lines=True)
    frame = frame.loc[frame.task_id == task].sort_values("source_value").reset_index(drop=True)
    if cap is not None:
        # Deterministic, evenly spaced values; full-run inputs have no cap.
        frame = frame.iloc[np.linspace(0, len(frame) - 1, min(cap, len(frame)), dtype=int)].reset_index(drop=True)
    for row in frame.itertuples():
        ids = tokenizer(row.prompt, add_special_tokens=False)["input_ids"]
        if ids != row.input_ids or ids[row.token_idx] != row.source_token_id:
            raise ValueError(f"Tokenizer/input identity mismatch: {row.example_id}")
    return frame


def audit_forward(model, tokenizer, frame, bundle, cfg, recipe):
    import torch
    from src.gradient_steering import forward_next_token_logits_with_delta

    frame = frame.iloc[:min(4, len(frame))]
    kwargs = dict(model=model, tokenizer=tokenizer, prompts=frame.prompt.tolist(),
                  token_idxs=frame.token_idx.astype(int).tolist(), layer_idx=recipe["layer_idx"],
                  batch_size=4, device="cuda")
    fast, live = forward_last_logits(**kwargs)
    delta, _ = intervention_delta(live, bundle["center"], bundle,
                                  next(c for c in conditions() if c.name == "delete_property"), seed=0)
    records = []
    with torch.inference_mode():
        for name, edit in (("original", None), ("delete_property", delta)):
            old, old_live = forward_next_token_logits_with_delta(**kwargs, delta_rows=edit, capture_selected=True)
            new, _ = forward_last_logits(**kwargs, delta_rows=edit)
            metrics = next_token_metrics(old, new, frame.target_token_id.to_numpy(dtype=int))
            record = {"condition": name, "max_abs_logit_difference": float(np.max(np.abs(old - new))),
                      "max_kl": float(metrics["kl_original_to_intervention"].max()),
                      "top1_agreement": float(metrics["top1_agreement"].mean())}
            if name == "original":
                record["max_abs_activation_difference"] = float(np.max(np.abs(old_live - live)))
                if not np.array_equal(old_live, live):
                    raise ValueError(f"Forward audit activation mismatch: {record}")
            if record["top1_agreement"] != 1 or record["max_kl"] > 1e-5:
                raise ValueError(f"Forward audit failed: {record}")
            records.append(record)
    return records


def summarize(parts):
    records = []
    for path in sorted(parts):
        frame = pd.read_csv(path)
        selected = frame.loc[frame.baseline_correct]
        records.append({
            "split": frame.split.iloc[0], "task_id": frame.task_id.iloc[0],
            "condition": frame.condition.iloc[0], "family": frame.family.iloc[0],
            "n": len(frame), "baseline_accuracy": frame.baseline_correct.mean(),
            "accuracy": frame.intervention_correct.mean(),
            "accuracy_change": frame.intervention_correct.mean() - frame.baseline_correct.mean(),
            "baseline_correct_n": len(selected),
            "accuracy_on_baseline_correct": selected.intervention_correct.mean(),
            "mean_kl": frame.kl_original_to_intervention.mean(),
            "mean_logprob_change": frame.target_logprob_delta.mean(),
            "mean_delta_l2": frame.delta_l2.mean(),
            "mean_delta_energy_fraction": frame.delta_energy_fraction.mean(),
            "aggregate_delta_energy_fraction": (frame.delta_l2 ** 2).sum() / frame.centered_energy.sum(),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "out/rank41_deletion_inputs/20260915_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "out/runs")
    parser.add_argument("--splits", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    cfg, manifest, source_files, run_dir = plan(args)
    expected = [run_dir / "parts" / f"{split}__{task}__{c.name}.csv"
                for split in cfg.splits for task in cfg.tasks for c in conditions()]
    print(json.dumps({"run_dir": str(run_dir), "conditions": len(expected), "config": cfg.to_dict()}), flush=True)
    if args.plan_only:
        return
    if (run_dir / "DONE").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if metrics["run_id"] != cfg.run_id or not all(p.exists() for p in expected):
            raise ValueError("Invalid completed run")
        print("Complete: skip without model loading", flush=True)
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != cfg.workspace_config:
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching")
    import torch
    from src.gradient_steering import freeze_model_parameters
    from src.model_loader import load_causal_lm
    from src.utils import set_global_seed

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "parts").mkdir(exist_ok=True)
    existing = run_dir / "run.json"
    if existing.exists() and json.loads(existing.read_text())["config"] != json.loads(json.dumps(cfg.to_dict())):
        raise ValueError("Run config changed")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = "staged code; see source hashes"
    provenance = {
        "run_id": cfg.run_id, "config": cfg.to_dict(), "recipe": manifest["recipe"],
        "source_files": source_files, "git_commit": commit, "host": socket.gethostname(),
        "python": platform.python_version(), "executable": sys.executable,
        "packages": {p: importlib.metadata.version(p) for p in (
            "torch", "transformers", "numpy", "scipy", "pandas", "scikit-learn", "accelerate", "tokenizers")},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "environment": {k: os.environ.get(k) for k in (
            "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES")},
        "cgroup": Path("/proc/self/cgroup").read_text(), "started_unix": time.time(),
    }
    atomic_write_json(existing, provenance)
    atomic_write_text(run_dir / "command.txt", shlex.join(sys.argv) + "\n")
    set_global_seed(20260915)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    recipe = manifest["recipe"]
    loaded = load_causal_lm(recipe["model"], revision=recipe["revision"], device="cuda",
                           dtype=cfg.dtype, attn_implementation=cfg.attention)
    model, tokenizer = loaded.model, loaded.tokenizer
    freeze_model_parameters(model)
    first_bundle = dict(np.load(args.bundle / f"{cfg.splits[0]}.npz"))
    first_frame = read_frame(args.bundle, cfg.splits[0], cfg.tasks[0], cfg.max_examples, tokenizer)
    audit = audit_forward(model, tokenizer, first_frame, first_bundle, cfg, recipe)
    atomic_write_json(run_dir / "forward_audit.json", audit)
    print(f"forward audit passed: {audit}", flush=True)
    started = time.monotonic()
    timings = []
    for split in cfg.splits:
        bundle = dict(np.load(args.bundle / f"{split}.npz"))
        for task in cfg.tasks:
            paths = {c.name: run_dir / "parts" / f"{split}__{task}__{c.name}.csv" for c in conditions()}
            if all(p.exists() for p in paths.values()):
                continue
            frame = read_frame(args.bundle, split, task, cfg.max_examples, tokenizer)
            kwargs = dict(model=model, tokenizer=tokenizer, prompts=frame.prompt.tolist(),
                          token_idxs=frame.token_idx.astype(int).tolist(), layer_idx=recipe["layer_idx"],
                          batch_size=cfg.batch_size, device="cuda")
            baseline, live = forward_last_logits(**kwargs)
            for c in conditions():
                path = paths[c.name]
                if path.exists():
                    continue
                t0 = time.monotonic()
                delta, diagnostics = intervention_delta(live, bundle["center"], bundle, c,
                                                       seed=stable_seed(split, task, c.name))
                # Evaluate the zero-delta condition explicitly to check edit-hook identity.
                logits, _ = forward_last_logits(**kwargs, delta_rows=delta)
                metrics = next_token_metrics(baseline, logits, frame.target_token_id.to_numpy(dtype=int))
                if c.name == "original" and (not np.array_equal(logits, baseline)):
                    raise ValueError("Zero-delta identity failed")
                detail = frame.drop(columns=["input_ids"]).copy()
                detail["condition"] = c.name
                detail["family"] = c.family
                detail["operation"] = c.operation
                detail["basis_rank"] = bundle[c.basis].shape[1]
                for key, values in {**metrics, **diagnostics}.items():
                    detail[key] = values
                detail["baseline_token"] = [tokenizer.decode([int(x)]) for x in metrics["baseline_token_id"]]
                detail["intervention_token"] = [tokenizer.decode([int(x)]) for x in metrics["intervention_token_id"]]
                detail["run_id"] = cfg.run_id
                atomic_write_csv(path, detail)
                seconds = time.monotonic() - t0
                timings.append({"split": split, "task": task, "condition": c.name, "seconds": seconds, "n": len(frame)})
                complete = sum(p.exists() for p in expected)
                atomic_write_json(run_dir / "progress.json", {
                    "run_id": cfg.run_id, "complete": complete, "total": len(expected),
                    "elapsed_seconds": time.monotonic() - started, "latest": timings[-1],
                    "updated_unix": time.time(), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                })
                print(f"{split} {task} {c.name}: acc={metrics['intervention_correct'].mean():.3f} "
                      f"{seconds:.1f}s [{complete}/{len(expected)}]", flush=True)
            del baseline, live
    if not all(p.exists() for p in expected):
        raise RuntimeError("Incomplete condition grid")
    summary = summarize(expected)
    atomic_write_csv(run_dir / "summary.csv", summary)
    atomic_write_json(run_dir / "metrics.json", {
        "run_id": cfg.run_id, "condition_parts": len(expected), "rows": int(summary.n.sum()),
        "seconds_this_invocation": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "timings": timings,
    })
    atomic_write_text(run_dir / "DONE", cfg.run_id + "\n")
    print(f"DONE {run_dir}", flush=True)


if __name__ == "__main__":
    main()
