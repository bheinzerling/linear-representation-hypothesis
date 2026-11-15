"""Frozen, resumable GPU sweep for individual number-property interventions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

from introlm.experiments.individual_property_protocol import (
    PROPERTIES,
    PROTOCOL_VERSION,
    Case,
    atomic_json,
    digest,
    fit_case,
    joint_cases,
    list_cases,
    protocol_identity,
    sample_values,
    score,
    summary,
    value_splits,
)


def tasks(config: dict[str, Any], pilot: bool) -> list[dict[str, Any]]:
    properties = config["pilot_properties"] if pilot else config["properties"]
    seeds = [config["pilot_seed"]] if pilot else config["seeds"]
    return [
        {
            "model": model,
            "property": p,
            "seed": seed,
            "pilot": pilot,
            "run_id": f"{model}--{p}--s{seed}",
        }
        for model in config["models"]
        for p in properties
        for seed in seeds
    ]


def source_digest() -> str:
    here = Path(__file__)
    paths = [
        here,
        here.with_name("individual_property_protocol.py"),
        here.parents[3] / "scripts/run_olmo_number_property_steering.py",
    ]
    return digest(
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    )


def run_directory(root: Path, task: dict[str, Any]) -> Path:
    return root / ("pilot" if task["pilot"] else "main") / task["run_id"]


def render_case(tokenizer: Any, case: Case, chat: bool) -> tuple[str, list[int]]:
    if chat:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": case.content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = case.content
    anchor_start = text.rfind(case.anchor)
    if anchor_start < 0:
        raise ValueError("The query anchor is absent from the rendered prompt")
    start = anchor_start + case.anchor_offset
    end = start + len(str(case.value))
    if text[start:end] != str(case.value):
        raise ValueError("Source numeral span mismatch")
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [
        i
        for i, (a, b) in enumerate(tokenized["offset_mapping"])
        if max(int(a), start) < min(int(b), end)
    ]
    if not positions:
        raise ValueError("The source numeral has no token span")
    return text, positions


class Engine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        chat: bool,
        directory: Path,
        run_identity: str,
        batch_size: int,
    ):
        # Reuse the already qualified model-layout and EOS adapters.
        import importlib.util

        adapter_path = (
            Path(__file__).resolve().parents[3]
            / "scripts/run_olmo_number_property_steering.py"
        )
        module_spec = importlib.util.spec_from_file_location(
            "individual_property_legacy_adapter", adapter_path
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("Legacy generation adapter is unavailable")
        import sys

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        self.layers = module.transformer_layers(model)
        self.stop_ids = module.eos_ids(model, tokenizer)
        self.model, self.tokenizer, self.chat = model, tokenizer, chat
        self.directory, self.run_identity = directory, run_identity
        self.batch_size = batch_size
        self.cache_index: dict[str, str] = {}
        self.torch = __import__("torch")
        self.device = next(model.parameters()).device

    def encoded(self, cases: list[Case]):
        torch = self.torch
        rendered = [render_case(self.tokenizer, case, self.chat) for case in cases]
        texts = [item[0] for item in rendered]
        batch = self.tokenizer(
            texts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        batch.pop("token_type_ids", None)
        length = batch["input_ids"].shape[1]
        positions = []
        for case, (text, indices) in zip(cases, rendered, strict=True):
            encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            padded = length - len(encoded) + indices[-1]
            if int(batch["input_ids"][len(positions), padded]) != int(
                encoded[indices[-1]]
            ):
                raise ValueError("Source-token left-padding mismatch")
            positions.append(padded)
        return (
            {k: v.to(self.device) for k, v in batch.items()},
            torch.tensor(positions, device=self.device),
            texts,
        )

    def activations(self, cases: list[Case]):
        torch = self.torch
        chunks = {layer: [] for layer in range(len(self.layers))}
        with torch.inference_mode():
            for start in range(0, len(cases), self.batch_size):
                batch_cases = cases[start : start + self.batch_size]
                encoded, positions, _ = self.encoded(batch_cases)
                rows = torch.arange(len(batch_cases), device=self.device)
                captured = {}

                def capture(layer):
                    def hook(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        captured[layer] = hidden[rows, positions].float().cpu()

                    return hook

                handles = [
                    block.register_forward_hook(capture(layer))
                    for layer, block in enumerate(self.layers)
                ]
                try:
                    outputs = self.model(**encoded, use_cache=False)
                finally:
                    for handle in handles:
                        handle.remove()
                if set(captured) != set(chunks):
                    raise ValueError("Not every intervention site was captured")
                for layer in chunks:
                    chunks[layer].append(captured[layer])
                del outputs
        return {layer: torch.cat(parts) for layer, parts in chunks.items()}

    def generate(
        self,
        cases: list[Case],
        *,
        label: str,
        units=None,
        alpha: float = 0.0,
        layer: int | None = None,
        condition: str = "baseline",
        max_tokens: int = 40,
    ):
        torch = self.torch
        unit_hash = (
            None
            if units is None
            else hashlib.sha256(units.float().cpu().numpy().tobytes()).hexdigest()
        )
        key = digest(
            {
                "run": self.run_identity,
                "cases": [asdict(c) for c in cases],
                "units": unit_hash,
                "alpha": alpha,
                "layer": layer,
                "condition": condition,
                "max_tokens": max_tokens,
            }
        )
        path = self.directory / "generations" / f"{key}.json.gz"
        self.cache_index[label] = str(path.relative_to(self.directory))
        if path.exists():
            with gzip.open(path, "rt") as stream:
                cached = json.load(stream)
            if cached["cache_key"] != key or [r["case_id"] for r in cached["rows"]] != [
                c.case_id for c in cases
            ]:
                raise ValueError("Generation cache identity mismatch")
            return cached["rows"]
        results = []
        print(f"generate {label}: {len(cases)} examples", flush=True)
        with torch.inference_mode():
            for start in range(0, len(cases), self.batch_size):
                cohort = cases[start : start + self.batch_size]
                encoded, positions, texts = self.encoded(cohort)
                source_positions = positions.clone()
                applied = torch.zeros(len(cohort))
                handle = None
                if units is not None:
                    classes = torch.tensor(
                        [c.source_class for c in cohort], device=self.device
                    )
                    deltas = units.to(self.device)[classes].float() * float(alpha)
                    if condition == "wrong_sign":
                        deltas = -deltas
                    if condition == "delimiter_location":
                        positions = torch.full_like(
                            positions, encoded["input_ids"].shape[1] - 1
                        )

                    def hook(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        patched = hidden.clone()
                        rows = torch.arange(len(cohort), device=hidden.device)
                        original = hidden[rows, positions]
                        edited = (original.float() + deltas).to(hidden.dtype)
                        applied.copy_(
                            (edited.float() - original.float()).norm(dim=1).cpu()
                        )
                        patched[rows, positions] = edited
                        return (
                            (patched, *output[1:])
                            if isinstance(output, tuple)
                            else patched
                        )

                    handle = self.layers[int(layer)].register_forward_hook(hook)
                try:
                    outputs = self.model(**encoded, use_cache=True)
                finally:
                    if handle is not None:
                        handle.remove()
                attention = encoded["attention_mask"]
                past, logits = outputs.past_key_values, outputs.logits[:, -1]
                finished = torch.zeros(
                    len(cohort), dtype=torch.bool, device=self.device
                )
                generated = []
                for step in range(max_tokens):
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite generation logits")
                    previous_finished = finished.clone()
                    tokens = logits.argmax(dim=-1)
                    tokens = torch.where(
                        previous_finished, self.tokenizer.pad_token_id, tokens
                    )
                    generated.append(tokens.cpu())
                    for stop in self.stop_ids:
                        finished |= tokens == stop
                    if bool(finished.all()) or step + 1 == max_tokens:
                        break
                    attention = torch.cat(
                        [attention, (~previous_finished).long()[:, None]], dim=1
                    )
                    outputs = self.model(
                        input_ids=tokens[:, None],
                        attention_mask=attention,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past, logits = outputs.past_key_values, outputs.logits[:, -1]
                completions = self.tokenizer.batch_decode(
                    torch.stack(generated, dim=1), skip_special_tokens=True
                )
                for i, (case, text, completion) in enumerate(
                    zip(cohort, texts, completions, strict=True)
                ):
                    results.append(
                        {
                            "case_id": case.case_id,
                            "case": asdict(case),
                            "source_class": case.source_class,
                            "condition": condition,
                            "prompt": text,
                            "generated_text": completion,
                            "source_position": int(source_positions[i]),
                            "patched_position": None
                            if units is None
                            else int(positions[i]),
                            "source_token_id": int(
                                encoded["input_ids"][i, source_positions[i]]
                            ),
                            "patched_token_id": None
                            if units is None
                            else int(encoded["input_ids"][i, positions[i]]),
                            "nominal_norm": float(alpha) if units is not None else 0.0,
                            "applied_norm": float(applied[i]),
                            **score(case, completion),
                        }
                    )
                del outputs, past, logits
        payload = {"cache_key": key, "rows": results}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt") as stream:
            json.dump(payload, stream)
        temporary.replace(path)
        return results


def balanced_accuracy(predictions, labels, classes: int) -> float:
    return (
        sum(
            float((predictions[labels == c] == c).float().mean())
            for c in range(classes)
        )
        / classes
    )


def fit_ridge_heads(activations, values, spec, lambdas, device):
    """Fit class-balanced linear ridge probes; select regularization on dev only."""
    import torch

    labels = {
        split: torch.tensor([spec.label(n) for n in nums], device=device)
        for split, nums in values.items()
    }
    heads, metrics = {}, []
    for layer in sorted(activations["train"]):
        x = activations["train"][layer].to(device)
        y = labels["train"]
        counts = torch.bincount(y, minlength=spec.num_classes).float()
        weights = len(y) / (spec.num_classes * counts[y])
        mu = (weights[:, None] * x).sum(dim=0) / weights.sum()
        scale = (
            ((weights[:, None] * (x - mu).square()).sum(dim=0) / weights.sum())
            .sqrt()
            .clamp_min(1e-4)
        )
        z = (x - mu) / scale
        sqrt_weights = weights.sqrt()[:, None]
        a = z * sqrt_weights
        prior = torch.full((spec.num_classes,), 1 / spec.num_classes, device=device)
        target = (
            torch.nn.functional.one_hot(y, spec.num_classes).float() - prior
        ) * sqrt_weights
        gram = a @ a.T
        identity = torch.eye(len(y), device=device)
        best = None
        for ridge in lambdas:
            coefficient = a.T @ torch.linalg.solve(
                gram + float(ridge) * identity, target
            )
            raw_weight = coefficient / scale[:, None]
            bias = prior - mu @ raw_weight
            dev = activations["dev"][layer].to(device) @ raw_weight + bias
            accuracy = balanced_accuracy(
                dev.argmax(dim=1), labels["dev"], spec.num_classes
            )
            candidate = (accuracy, float(ridge))
            if best is None or candidate > best[0]:
                best = (candidate, raw_weight.detach().clone(), bias.detach().clone())
        assert best is not None
        (dev_accuracy, ridge), weight, bias = best
        test = activations["test"][layer].to(device) @ weight + bias
        test_accuracy = balanced_accuracy(
            test.argmax(dim=1), labels["test"], spec.num_classes
        )
        units = torch.stack(
            [
                weight[:, (c + 1) % spec.num_classes] - weight[:, c]
                for c in range(spec.num_classes)
            ]
        )
        units = units / units.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if not torch.isfinite(units).all():
            raise ValueError("Nonfinite probe direction")
        heads[layer] = units.cpu()
        metrics.append(
            {
                "layer": layer,
                "dev_balanced_accuracy": dev_accuracy,
                "test_balanced_accuracy": test_accuracy,
                "ridge": ridge,
            }
        )
    return heads, metrics


def candidate_layers(
    metrics: list[dict[str, Any]], fixed_layer: int, count: int
) -> list[int]:
    ordered = sorted(
        metrics, key=lambda r: (-r["dev_balanced_accuracy"], -r["ridge"], r["layer"])
    )
    return list(dict.fromkeys([fixed_layer] + [r["layer"] for r in ordered[:count]]))


def random_units(units, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    control = torch.randn(units.shape, generator=generator)
    control -= (control * units).sum(dim=1, keepdim=True) * units
    return control / control.norm(dim=1, keepdim=True).clamp_min(1e-12)


def select_condition(rows, minimum_identity):
    eligible = [
        r for r in rows if r["metrics"]["macro_identity_preserved"] >= minimum_identity
    ]
    candidates = eligible or rows
    chosen = max(
        candidates,
        key=lambda r: (
            r["metrics"]["macro_conditional_target_success"],
            r["metrics"]["macro_identity_preserved"],
            -r["alpha"],
            -r["layer"],
        ),
    )
    return {**chosen, "identity_gate_passed": bool(eligible)}


def worker(config: dict[str, Any], task: dict[str, Any], output_root: Path) -> None:
    import torch
    import transformers
    import numpy as np

    identity = digest(
        {
            "protocol": protocol_identity(config, task),
            "source": source_digest(),
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "architecture": platform.machine(),
                "attention": "sdpa-math-bfloat16",
            },
        }
    )
    directory = run_directory(output_root, task)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "DONE").exists():
        existing = json.loads((directory / "results.json").read_text())
        if existing["run_identity"] != identity:
            raise ValueError("Completed run belongs to another protocol")
        validate_run(directory)
        print(f"already complete: {task['run_id']}")
        return
    started = time.time()
    spec = PROPERTIES[task["property"]]
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    binding = config["models"][task["model"]]
    model_path = Path(binding["path"])
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model.to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Verify a real compute-node kernel, not just package import.
    probe = torch.randn((64, 64), device="cuda")
    if not torch.isfinite(probe @ probe.T).all():
        raise ValueError("CUDA matmul preflight failed")
    runtime = {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
        "model": binding,
        "source_digest": source_digest(),
        "run_identity": identity,
        "environment": {
            k: os.environ.get(k)
            for k in [
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
            ]
        },
    }
    atomic_json(directory / "runtime.json", runtime)
    splits = value_splits(spec, seed)
    atomic_json(
        directory / "splits.json",
        {
            "values": splits,
            "split_hash": digest(splits),
            "class_counts": {
                name: {
                    str(c): sum(spec.label(n) == c for n in nums)
                    for c in range(spec.num_classes)
                }
                for name, nums in splits.items()
            },
        },
    )
    parameters = config["pilot"] if task["pilot"] else config["main"]
    values = {
        name: sample_values(spec, splits[name], parameters[f"{name}_examples"], seed)
        for name in ["dev", "test"]
    }
    cases = {
        family: {
            split: (
                joint_cases(spec, nums, splits["train"], seed, split)
                if family == "joint"
                else list_cases(spec, nums, splits[split], seed, split)
            )
            for split, nums in values.items()
        }
        for family in ["joint", "list"]
    }
    engine = Engine(
        model,
        tokenizer,
        bool(binding["chat"]),
        directory,
        identity,
        config["batch_size"],
    )
    baselines = {
        family: {
            split: engine.generate(
                cohort,
                label=f"{family}/{split}/baseline",
                max_tokens=40 if family == "joint" else 64,
            )
            for split, cohort in data.items()
        }
        for family, data in cases.items()
    }
    gates = {
        family: {split: summary(rows) for split, rows in data.items()}
        for family, data in baselines.items()
    }
    atomic_json(directory / "baseline_metrics.json", gates)
    all_results = []
    fixed = int(binding["fixed_layer"])
    for context in config["fit_contexts"]:
        print(f"fit {task['run_id']} {context}", flush=True)
        activation_file = directory / f"activations-{context}.pt"
        if activation_file.exists():
            payload = torch.load(
                activation_file, map_location="cpu", weights_only=False
            )
            if payload["run_identity"] != identity:
                raise ValueError("Activation cache identity mismatch")
            activations = payload["activations"]
        else:
            activations = {
                split: engine.activations([fit_case(spec, n, context) for n in nums])
                for split, nums in splits.items()
            }
            temporary = activation_file.with_suffix(".tmp")
            torch.save(
                {"run_identity": identity, "activations": activations}, temporary
            )
            temporary.replace(activation_file)
        heads, metrics = fit_ridge_heads(
            activations, splits, spec, config["ridge"], engine.device
        )
        atomic_json(directory / f"probe-metrics-{context}.json", metrics)
        selected_layers = candidate_layers(
            metrics, fixed, parameters["candidate_layers"]
        )
        torch.save(
            {
                "run_identity": identity,
                "units": {layer: heads[layer] for layer in selected_layers},
            },
            directory / f"directions-{context}.pt",
        )
        for family in ["joint", "list"]:
            baseline_dev = baselines[family]["dev"]
            capable = (
                gates[family]["dev"]["macro_factual"]
                >= config["minimum_baseline_accuracy"]
                and gates[family]["dev"]["macro_parsed"]
                >= config["minimum_parse_coverage"]
            )
            if not capable:
                all_results.append(
                    {
                        "fit_context": context,
                        "family": family,
                        "status": "behavioral_baseline_failed",
                        "baseline": gates[family],
                    }
                )
                continue
            sweep = []
            for layer in selected_layers:
                for alpha in parameters["alphas"]:
                    label = f"{context}/{family}/dev/l{layer}/a{alpha}"
                    rows = engine.generate(
                        cases[family]["dev"],
                        label=label,
                        units=heads[layer],
                        layer=layer,
                        alpha=alpha,
                        condition="selected_direction",
                        max_tokens=40 if family == "joint" else 64,
                    )
                    sweep.append(
                        {
                            "layer": layer,
                            "alpha": alpha,
                            "metrics": summary(rows, baseline_dev),
                        }
                    )
            atomic_json(directory / f"selection-{context}-{family}.json", sweep)
            for mode in ["fixed", "selected"]:
                candidates = (
                    [r for r in sweep if r["layer"] == fixed]
                    if mode == "fixed"
                    else sweep
                )
                chosen = select_condition(candidates, config["minimum_identity"])
                layer, alpha = int(chosen["layer"]), float(chosen["alpha"])
                units = heads[layer]
                controls = random_units(units, seed + layer + 10000)
                result = {
                    "fit_context": context,
                    "family": family,
                    "site_mode": mode,
                    "status": "evaluated",
                    "selection": chosen,
                    "probe": next(r for r in metrics if r["layer"] == layer),
                    "baseline": gates[family],
                    "conditions": {},
                }
                for condition in [
                    "selected_direction",
                    "wrong_sign",
                    "random_direction",
                    "delimiter_location",
                ]:
                    rows = engine.generate(
                        cases[family]["test"],
                        label=f"{context}/{family}/{mode}/test/{condition}",
                        units=controls if condition == "random_direction" else units,
                        alpha=alpha,
                        layer=layer,
                        condition=condition,
                        max_tokens=40 if family == "joint" else 64,
                    )
                    result["conditions"][condition] = summary(
                        rows, baselines[family]["test"]
                    )
                all_results.append(result)
                atomic_json(directory / "partial-results.json", all_results)
        del activations, heads
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "run_identity": identity,
        "task": task,
        "property": asdict(spec),
        "runtime": runtime,
        "baseline_metrics": gates,
        "results": all_results,
        "generation_index": engine.cache_index,
        "elapsed_seconds": time.time() - started,
        "pilot_qualification": "application_validation_required",
    }
    atomic_json(directory / "results.json", result)
    validation = validate_run(directory)
    atomic_json(directory / "validation.json", validation)
    result["pilot_qualification"] = "passed" if task["pilot"] else "not_applicable"
    atomic_json(directory / "results.json", result)
    (directory / "DONE").write_text(identity + "\n")
    print(
        json.dumps(
            {
                "run_id": task["run_id"],
                "validation": validation,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )


def validate_run(directory: Path) -> dict[str, Any]:
    result = json.loads((directory / "results.json").read_text())
    spec = PROPERTIES[result["task"]["property"]]
    from introlm.experiments.individual_property_protocol import validate_splits

    splits = json.loads((directory / "splits.json").read_text())
    validate_splits(spec, splits["values"])
    if digest(splits["values"]) != splits["split_hash"]:
        raise ValueError("Split digest mismatch")
    observed = set()
    count = 0
    rows_by_path = {}
    for relative in set(result["generation_index"].values()):
        path = directory / relative
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        if path.name != payload["cache_key"] + ".json.gz":
            raise ValueError("Generation payload identity mismatch")
        rows_by_path[relative] = payload["rows"]
        for row in payload["rows"]:
            fields = row["case"].copy()
            for name in ["expected_factual", "expected_target", "input_values"]:
                fields[name] = tuple(fields[name])
            case = Case(**fields)
            replay = score(case, row["generated_text"])
            if any(replay[key] != row[key] for key in replay):
                raise ValueError("Completion scores do not reproduce")
            split = case.case_id.split(":", 1)[0]
            if (
                split not in {"dev", "test"}
                or case.value not in splits["values"][split]
            ):
                raise ValueError(
                    "Generation uses a value outside its declared partition"
                )
            if case.family == "list" and not set(case.input_values).issubset(
                set(splits["values"][split])
            ):
                raise ValueError("A copied list crosses value partitions")
            observed.add(case.family)
            count += 1
    index = result["generation_index"]
    for family in ["joint", "list"]:
        for split in ["dev", "test"]:
            rows = rows_by_path[index[f"{family}/{split}/baseline"]]
            if digest(summary(rows)) != digest(
                result["baseline_metrics"][family][split]
            ):
                raise ValueError("Baseline aggregate does not reproduce")
    for comparison in result["results"]:
        if comparison["status"] != "evaluated":
            continue
        context, family, mode = (
            comparison["fit_context"],
            comparison["family"],
            comparison["site_mode"],
        )
        baseline = rows_by_path[index[f"{family}/test/baseline"]]
        for condition, stored in comparison["conditions"].items():
            rows = rows_by_path[index[f"{context}/{family}/{mode}/test/{condition}"]]
            if digest(summary(rows, baseline)) != digest(stored):
                raise ValueError("Intervention aggregate does not reproduce")
    if observed != {"joint", "list"}:
        raise ValueError("Missing behavioral family")
    if {r["fit_context"] for r in result["results"]} != {"property", "addition"}:
        raise ValueError("Missing fitting context")
    if not count:
        raise ValueError("No saved completions")
    return {
        "status": "passed",
        "run_identity": result["run_identity"],
        "records_replayed": count,
        "families": sorted(observed),
        "value_disjoint": True,
        "split_hash": splits["split_hash"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["plan", "worker", "validate"], default="plan"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "validate":
        print(json.dumps(validate_run(args.run_dir), indent=2))
        return
    if args.config is None:
        parser.error("--config is required")
    config = json.loads(args.config.read_text())
    if len(config["properties"]) != 19 or set(config["properties"]) != set(PROPERTIES):
        raise ValueError(
            "The main sweep must cover the nineteen paper properties exactly"
        )
    rows = tasks(config, args.pilot)
    if args.phase == "plan":
        plan = {
            "protocol_version": PROTOCOL_VERSION,
            "source_digest": source_digest(),
            "tasks": rows,
            "task_count": len(rows),
            "pilot": args.pilot,
            "config_sha256": digest(config),
            "config": config,
        }
        if args.output:
            atomic_json(args.output, plan)
        print(
            json.dumps(
                {
                    "task_count": len(rows),
                    "pilot": args.pilot,
                    "config_sha256": plan["config_sha256"],
                },
                indent=2,
            )
        )
        return
    index = (
        args.task_index
        if args.task_index is not None
        else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    worker(config, rows[index], args.output_root)


if __name__ == "__main__":
    main()
