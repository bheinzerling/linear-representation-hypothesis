#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_figure1_bag_of_features_steering import (  # noqa: E402
    PROPERTY_ORDER,
    load_panel,
    load_probe_bundle,
    load_source_config,
    orthonormal_union_basis,
)
from scripts.run_figure1_oracle_property_scaling import load_source_activations  # noqa: E402
from scripts.run_source_value_probe_manifold_steering import _resolve_model_entry  # noqa: E402
from src.activation_steering import collect_prompt_hidden_sequences_torch  # noqa: E402
from src.best_in_span import (  # noqa: E402
    joint_constraint_diagnostics,
    maximum_feasible_ray_scale,
    orthonormalize_columns,
    project_rows_to_basis,
    projected_fisher_from_logit_derivatives,
    radial_project_joint_constraints_torch,
)
from src.gradient_steering import (  # noqa: E402
    _pad_hidden_sequences,
    _temporary_decoder_suffix,
    freeze_model_parameters,
)
from src.model_loader import load_causal_lm, load_tokenizer  # noqa: E402
from src.number_property_probes import joint_minimum_norm_delta  # noqa: E402
from src.ones_digit_intervention import _numeric_tracking_arrays  # noqa: E402
from src.source_value_steering import summarize_steering_metrics  # noqa: E402
from src.steering_sanity import evaluate_logits_against_shifted_targets  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    ensure_dir,
    load_yaml,
    runtime_metadata,
    set_global_seed,
)


SOURCE_RUN = ROOT / "out/runs/source_value_probe_manifold_steering__9bbad17c01ee__r0"
BASE_BAG_RUN = ROOT / "out/figure1_bag_of_features_steering/llama31_8b_layer10_v1"
ORACLE_RUN = ROOT / "out/figure1_oracle_property_scaling/llama31_8b_layer10_v1"
CENTROID_ARTIFACT = (
    ROOT
    / "experiment-reports/artifacts/2026-06-16-figure1-allpair-delta-dictionary"
    / "value_representations.npz"
)
COMPETITIVE_STATE = (
    ROOT
    / "out/property_probe_training_scheme_comparison/llama31_8b_layer10_v1"
    / "states/split_00.npz"
)
DEFAULT_OUTPUT = ROOT / "out/figure1_best_in_span_causal_ceiling/llama31_8b_layer10_v1"
SEED = 20260727
SPAN_NAMES = ("old", "competitive")
BUDGET_NAMES = ("exact_patch", "fixed6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-row causal ceilings inside old and competitive number-property "
            "spans under matched L2 and local numeric-Fisher budgets."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--base-bag-run-dir", type=Path, default=BASE_BAG_RUN)
    parser.add_argument("--oracle-run-dir", type=Path, default=ORACLE_RUN)
    parser.add_argument("--centroid-artifact", type=Path, default=CENTROID_ARTIFACT)
    parser.add_argument("--competitive-state", type=Path, default=COMPETITIVE_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "metric", "optimize", "evaluate"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--capture-batch-size", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--scalar-grid-size", type=int, default=33)
    parser.add_argument("--scalar-refinements", type=int, default=2)
    parser.add_argument("--scalar-refinement-size", type=int, default=9)
    parser.add_argument("--optimizer-steps", type=int, default=60)
    parser.add_argument("--optimizer-lr", type=float, default=0.08)
    parser.add_argument("--optimizer-restarts", choices=("natural", "zero,natural"), default="zero,natural")
    parser.add_argument("--force-hidden-cache", action="store_true")
    parser.add_argument("--force-metric", action="store_true")
    parser.add_argument("--force-optimize", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    return parser.parse_args()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _panel_signature(
    panel: pd.DataFrame,
    *,
    model_revision: str,
    layer_idx: int,
    attn_implementation: str | None,
    tokenizer,
) -> str:
    digest = hashlib.sha256()
    header = {
        "model_revision": model_revision,
        "layer_idx": int(layer_idx),
        "attn_implementation": attn_implementation,
        "cache_format": 1,
    }
    digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    for row in panel[
        ["baseline_row_idx", "prompt", "token_idx", "source_value", "target_value", "target_token_id"]
    ].itertuples(index=False, name=None):
        payload = list(row)
        payload.append(tokenizer(str(row[1]), add_special_tokens=False)["input_ids"])
        digest.update(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _experiment_signature(
    *,
    panel_signature: str,
    model_revision: str,
    layer_idx: int,
    cublas_workspace_config: str,
    bases: Mapping[str, np.ndarray],
    reference_deltas: Mapping[str, np.ndarray],
    tracked_token_ids: np.ndarray,
    optimizer_config: Mapping[str, Any],
) -> str:
    payload = {
        "schema": 1,
        "panel_signature": panel_signature,
        "model_revision": model_revision,
        "layer_idx": int(layer_idx),
        "cublas_workspace_config": cublas_workspace_config,
        "basis_hashes": {
            name: _array_sha256(np.asarray(bases[name])) for name in SPAN_NAMES
        },
        "reference_delta_hashes": {
            name: _array_sha256(np.asarray(reference_deltas[name]))
            for name in BUDGET_NAMES
        },
        "tracked_token_ids_hash": _array_sha256(
            np.asarray(tracked_token_ids, dtype=np.int64)
        ),
        "optimizer": dict(optimizer_config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _guard_output_signature(output_dir: Path, run_signature: str) -> None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return
    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    existing_signature = existing.get("run_signature")
    stage_artifacts = (
        "local_metrics.npz",
        "optimizations.npz",
        "causal_details.csv",
        "causal_summary.csv",
        "DONE",
    )
    if (
        existing_signature != run_signature
        and any((output_dir / name).exists() for name in stage_artifacts)
    ):
        raise ValueError(
            "Output directory contains cached stages from a different experiment "
            f"signature ({existing_signature!r} != {run_signature!r}). Use a new "
            "output directory or remove only the obsolete run artifacts explicitly."
        )


def _load_target_activations(path: Path, expected_rows: int) -> np.ndarray:
    with np.load(path) as payload:
        result = np.asarray(payload["target_activations"], dtype=np.float64)
    if result.shape[0] < expected_rows:
        raise ValueError(
            f"Need at least {expected_rows} target activations, found {result.shape}."
        )
    # load_panel(..., max_rows=N) takes the aligned prefix of the canonical panel.
    # Mirror that operation here so small smoke tests cannot silently use a
    # differently sized target cache.
    return result[:expected_rows]


def _build_bases_and_deltas(
    *,
    panel: pd.DataFrame,
    base_bag_run_dir: Path,
    oracle_run_dir: Path,
    centroid_artifact: Path,
    competitive_state: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    probes = load_probe_bundle(base_bag_run_dir / "property_probes.npz")
    with np.load(centroid_artifact) as artifact:
        value_points = np.asarray(artifact["pls_value_points"], dtype=np.float64)
        value_direction = np.asarray(artifact["pls_decoder"], dtype=np.float64).reshape(-1)
    value_unit = value_direction / max(float(np.linalg.norm(value_direction)), 1e-12)
    old_property_rows = orthonormal_union_basis(
        probes,
        PROPERTY_ORDER,
        orthogonal_to=value_direction,
    )
    old_basis = orthonormalize_columns(
        np.concatenate([value_unit[:, None], old_property_rows.T], axis=1)
    )
    with np.load(competitive_state) as state:
        competitive_basis = orthonormalize_columns(
            np.asarray(state["full_basis__competitive_deflation"], dtype=np.float64)
        )
    bases = {"old": old_basis, "competitive": competitive_basis}

    source_values = panel["source_value"].to_numpy(dtype=np.int64)
    target_values = panel["target_value"].to_numpy(dtype=np.int64)
    base_delta = value_points[target_values] - value_points[source_values]
    property_delta = joint_minimum_norm_delta(
        probes,
        PROPERTY_ORDER,
        source_values,
        target_values,
        orthogonal_to=value_direction,
    )
    fixed6 = base_delta + 6.0 * property_delta
    source_activations = load_source_activations(base_bag_run_dir, panel)
    target_activations = _load_target_activations(
        oracle_run_dir / "target_activations.npz", len(panel)
    )
    exact_patch = target_activations - source_activations
    deltas = {"fixed6": fixed6, "exact_patch": exact_patch}
    return bases, deltas, source_activations, target_activations


def _unique_source_rows(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    keys = panel["baseline_row_idx"].to_numpy(dtype=np.int64)
    unique_keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(unique_keys)
    first = first[order]
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return panel.iloc[first].reset_index(drop=True), remap[inverse]


def _prepare_hidden_cache(
    *,
    cache_path: Path,
    signature: str,
    panel: pd.DataFrame,
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int,
    device: str,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists() and not force:
        with np.load(cache_path) as saved:
            if str(saved["signature"][0]) == signature:
                return (
                    np.asarray(saved["hidden"], dtype=np.float16),
                    np.asarray(saved["lengths"], dtype=np.int64),
                    np.asarray(saved["panel_to_unique"], dtype=np.int64),
                )
    unique, panel_to_unique = _unique_source_rows(panel)
    sequences = collect_prompt_hidden_sequences_torch(
        model=model,
        tokenizer=tokenizer,
        prompts=unique["prompt"].astype(str).tolist(),
        layer_idx=int(layer_idx),
        batch_size=int(batch_size),
        device=device,
    )
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    hidden_dim = int(sequences[0].shape[-1])
    hidden = np.zeros((len(sequences), int(lengths.max()), hidden_dim), dtype=np.float16)
    for idx, sequence in enumerate(sequences):
        hidden[idx, : lengths[idx]] = sequence.numpy().astype(np.float16)
    _atomic_savez(
        cache_path,
        signature=np.asarray([signature]),
        hidden=hidden,
        lengths=lengths,
        panel_to_unique=panel_to_unique,
        unique_token_idxs=unique["token_idx"].to_numpy(dtype=np.int64),
        unique_baseline_row_idxs=unique["baseline_row_idx"].to_numpy(dtype=np.int64),
    )
    return hidden, lengths, panel_to_unique


def _batch_hidden(
    hidden: np.ndarray,
    lengths: np.ndarray,
    panel_to_unique: np.ndarray,
    row_slice: slice,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = panel_to_unique[row_slice]
    sequences = [hidden[idx, : lengths[idx]] for idx in selected]
    return _pad_hidden_sequences(sequences, device=device, dtype=dtype)


def _suffix_logits(
    *,
    model,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    logit_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    outputs = model.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=hidden_states,
        use_cache=False,
    )
    last_positions = attention_mask.sum(dim=1) - 1
    final = outputs.last_hidden_state[
        torch.arange(len(hidden_states), device=hidden_states.device), last_positions
    ]
    if logit_token_ids is None:
        return model.lm_head(final)
    weight = model.lm_head.weight.index_select(0, logit_token_ids).to(final)
    bias = getattr(model.lm_head, "bias", None)
    result = final @ weight.T
    return result if bias is None else result + bias.index_select(0, logit_token_ids).to(result)


def _edit_hidden(
    hidden_states: torch.Tensor,
    token_idxs: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    edited = hidden_states.clone()
    rows = torch.arange(len(edited), device=edited.device)
    edited[rows, token_idxs] = edited[rows, token_idxs] + delta
    return edited


def _numeric_target_indices(panel: pd.DataFrame, tracked_token_ids: np.ndarray) -> np.ndarray:
    lookup = {int(token_id): idx for idx, token_id in enumerate(tracked_token_ids.tolist())}
    missing = sorted(set(panel["target_token_id"].astype(int)) - set(lookup))
    if missing:
        raise ValueError(f"Shifted targets are absent from numeric tracking tokens: {missing[:10]}")
    return np.asarray([lookup[int(value)] for value in panel["target_token_id"]], dtype=np.int64)


def _iter_slices(length: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


@contextmanager
def _suffix_context(model, layer_idx: int):
    with _temporary_decoder_suffix(model, int(layer_idx)):
        yield


def _compute_local_metrics(
    *,
    output_path: Path,
    panel: pd.DataFrame,
    hidden: np.ndarray,
    lengths: np.ndarray,
    panel_to_unique: np.ndarray,
    bases: Mapping[str, np.ndarray],
    reference_deltas: Mapping[str, np.ndarray],
    model,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tracked = torch.as_tensor(tracked_token_ids, dtype=torch.long, device=device)
    collected: dict[str, list[np.ndarray]] = {
        "base_numeric_logits": [],
        **{f"fisher__{name}": [] for name in SPAN_NAMES},
        **{f"reference_fisher_q__{name}": [] for name in BUDGET_NAMES},
    }
    with _suffix_context(model, layer_idx):
        for batch_number, row_slice in enumerate(_iter_slices(len(panel), batch_size), start=1):
            states, mask = _batch_hidden(
                hidden,
                lengths,
                panel_to_unique,
                row_slice,
                device=device,
                dtype=dtype,
            )
            token_idxs = torch.as_tensor(
                panel["token_idx"].to_numpy(dtype=np.int64, copy=True)[row_slice],
                dtype=torch.long,
                device=device,
            )

            def logits_from_delta(delta: torch.Tensor) -> torch.Tensor:
                return _suffix_logits(
                    model=model,
                    hidden_states=_edit_hidden(states, token_idxs, delta),
                    attention_mask=mask,
                    logit_token_ids=tracked,
                )

            zero_delta = torch.zeros(
                (len(states), states.shape[-1]), dtype=dtype, device=device
            )
            with torch.no_grad():
                base_logits = logits_from_delta(zero_delta)
            collected["base_numeric_logits"].append(base_logits.float().cpu().numpy())
            probabilities = torch.softmax(base_logits.float(), dim=1).cpu().numpy()

            for span_name in SPAN_NAMES:
                basis = torch.as_tensor(bases[span_name], dtype=dtype, device=device)
                zero = torch.zeros((len(states), basis.shape[1]), dtype=dtype, device=device)

                def logits_from_coefficients(coefficients: torch.Tensor) -> torch.Tensor:
                    return logits_from_delta(coefficients @ basis.T)

                derivatives: list[np.ndarray] = []
                for coordinate in range(basis.shape[1]):
                    tangent = torch.zeros_like(zero)
                    tangent[:, coordinate] = 1
                    _, derivative = torch.func.jvp(
                        logits_from_coefficients, (zero,), (tangent,)
                    )
                    derivatives.append(derivative.float().cpu().numpy())
                jacobian = np.stack(derivatives, axis=2)
                collected[f"fisher__{span_name}"].append(
                    projected_fisher_from_logit_derivatives(jacobian, probabilities).astype(
                        np.float32
                    )
                )

            for budget_name in BUDGET_NAMES:
                direction = torch.as_tensor(
                    reference_deltas[budget_name][row_slice], dtype=dtype, device=device
                )
                _, derivative = torch.func.jvp(
                    logits_from_delta, (zero_delta,), (direction,)
                )
                derivative_np = derivative.float().cpu().numpy()
                mean = np.sum(probabilities * derivative_np, axis=1, keepdims=True)
                quadratic = np.sum(
                    probabilities * np.square(derivative_np - mean), axis=1
                )
                collected[f"reference_fisher_q__{budget_name}"].append(
                    quadratic.astype(np.float32)
                )
            print(
                f"[metric] batch {batch_number}/"
                f"{int(np.ceil(len(panel) / batch_size))}",
                flush=True,
            )
    arrays = {
        name: np.concatenate(parts, axis=0) for name, parts in collected.items()
    }
    for budget_name in BUDGET_NAMES:
        arrays[f"reference_l2__{budget_name}"] = np.linalg.norm(
            reference_deltas[budget_name], axis=1
        ).astype(np.float32)
    _atomic_savez(output_path, **arrays)
    return arrays


def _numeric_logits_for_coefficients(
    *,
    model,
    states: torch.Tensor,
    mask: torch.Tensor,
    token_idxs: torch.Tensor,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    tracked: torch.Tensor,
) -> torch.Tensor:
    return _suffix_logits(
        model=model,
        hidden_states=_edit_hidden(states, token_idxs, coefficients @ basis.T),
        attention_mask=mask,
        logit_token_ids=tracked,
    )


def _target_scores(logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=1).gather(
        1, target_indices[:, None]
    )[:, 0]


def _optimize_scalar_batch(
    *,
    model,
    states: torch.Tensor,
    mask: torch.Tensor,
    token_idxs: torch.Tensor,
    basis: torch.Tensor,
    direction_coefficients: np.ndarray,
    maximum_scale: np.ndarray,
    target_indices: torch.Tensor,
    tracked: torch.Tensor,
    grid_size: int,
    refinements: int,
    refinement_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    direction = torch.as_tensor(direction_coefficients, dtype=states.dtype, device=states.device)
    maximum = torch.as_tensor(maximum_scale, dtype=torch.float32, device=states.device)
    best_score = torch.full_like(maximum, -torch.inf)
    best_alpha = torch.zeros_like(maximum)
    low = torch.zeros_like(maximum)
    high = maximum.clone()

    def evaluate(alpha: torch.Tensor) -> None:
        nonlocal best_score, best_alpha
        logits = _numeric_logits_for_coefficients(
            model=model,
            states=states,
            mask=mask,
            token_idxs=token_idxs,
            basis=basis,
            coefficients=direction * alpha.to(direction)[:, None],
            tracked=tracked,
        )
        score = _target_scores(logits, target_indices)
        improve = score > best_score
        best_score = torch.where(improve, score, best_score)
        best_alpha = torch.where(improve, alpha, best_alpha)

    with torch.no_grad():
        for fraction in torch.linspace(0.0, 1.0, int(grid_size), device=states.device):
            evaluate(maximum * fraction)
        half_width = maximum / max(int(grid_size) - 1, 1)
        for _ in range(int(refinements)):
            low = torch.clamp(best_alpha - half_width, min=0.0)
            high = torch.minimum(maximum, best_alpha + half_width)
            for fraction in torch.linspace(
                0.0, 1.0, int(refinement_size), device=states.device
            ):
                evaluate(low + fraction * (high - low))
            half_width = (high - low) / max(int(refinement_size) - 1, 1)
    return best_alpha.cpu().numpy(), best_score.cpu().numpy()


def _optimize_arbitrary_batch(
    *,
    model,
    states: torch.Tensor,
    mask: torch.Tensor,
    token_idxs: torch.Tensor,
    basis: torch.Tensor,
    initial_coefficients: np.ndarray,
    fisher: np.ndarray,
    l2_budget: np.ndarray,
    fisher_budget: np.ndarray,
    target_indices: torch.Tensor,
    tracked: torch.Tensor,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = torch.nn.Parameter(
        torch.as_tensor(initial_coefficients, dtype=torch.float32, device=states.device)
    )
    fisher_tensor = torch.as_tensor(fisher, dtype=torch.float32, device=states.device)
    l2_tensor = torch.as_tensor(l2_budget, dtype=torch.float32, device=states.device)
    fisher_budget_tensor = torch.as_tensor(
        fisher_budget, dtype=torch.float32, device=states.device
    )
    with torch.no_grad():
        coefficients.copy_(
            radial_project_joint_constraints_torch(
                coefficients,
                fisher_tensor,
                l2_budget=l2_tensor,
                fisher_budget=fisher_budget_tensor,
            )
        )
    optimizer = torch.optim.Adam([coefficients], lr=float(lr))
    best_coefficients = coefficients.detach().clone()
    with torch.no_grad():
        best_score = _target_scores(
            _numeric_logits_for_coefficients(
                model=model,
                states=states,
                mask=mask,
                token_idxs=token_idxs,
                basis=basis,
                coefficients=coefficients.to(states.dtype),
                tracked=tracked,
            ),
            target_indices,
        )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = _numeric_logits_for_coefficients(
            model=model,
            states=states,
            mask=mask,
            token_idxs=token_idxs,
            basis=basis,
            coefficients=coefficients.to(states.dtype),
            tracked=tracked,
        )
        scores = _target_scores(logits, target_indices)
        with torch.no_grad():
            improve = scores > best_score
            best_score = torch.where(improve, scores, best_score)
            best_coefficients = torch.where(
                improve[:, None], coefficients.detach(), best_coefficients
            )
        (-scores.mean()).backward()
        optimizer.step()
        with torch.no_grad():
            coefficients.copy_(
                radial_project_joint_constraints_torch(
                    coefficients,
                    fisher_tensor,
                    l2_budget=l2_tensor,
                    fisher_budget=fisher_budget_tensor,
                )
            )
    with torch.no_grad():
        final_scores = _target_scores(
            _numeric_logits_for_coefficients(
                model=model,
                states=states,
                mask=mask,
                token_idxs=token_idxs,
                basis=basis,
                coefficients=coefficients.to(states.dtype),
                tracked=tracked,
            ),
            target_indices,
        )
        improve = final_scores > best_score
        best_score = torch.where(improve, final_scores, best_score)
        best_coefficients = torch.where(
            improve[:, None], coefficients.detach(), best_coefficients
        )
    return best_coefficients.cpu().numpy(), best_score.cpu().numpy()


def _run_optimizations(
    *,
    output_path: Path,
    panel: pd.DataFrame,
    hidden: np.ndarray,
    lengths: np.ndarray,
    panel_to_unique: np.ndarray,
    bases: Mapping[str, np.ndarray],
    exact_patch: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    model,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
    grid_size: int,
    refinements: int,
    refinement_size: int,
    optimizer_steps: int,
    optimizer_lr: float,
    restarts: str,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tracked = torch.as_tensor(tracked_token_ids, dtype=torch.long, device=device)
    target_index_all = _numeric_target_indices(panel, tracked_token_ids)
    natural_coefficients = {
        span: project_rows_to_basis(exact_patch, bases[span])[0].astype(np.float32)
        for span in SPAN_NAMES
    }
    result_parts: dict[str, list[np.ndarray]] = {}
    with _suffix_context(model, layer_idx):
        for span in SPAN_NAMES:
            basis_tensor = torch.as_tensor(bases[span], dtype=dtype, device=device)
            fisher_all = np.asarray(metrics[f"fisher__{span}"], dtype=np.float32)
            for budget in BUDGET_NAMES:
                l2_all = np.asarray(metrics[f"reference_l2__{budget}"], dtype=np.float32)
                q_all = np.asarray(
                    metrics[f"reference_fisher_q__{budget}"], dtype=np.float32
                )
                max_scale_all = maximum_feasible_ray_scale(
                    natural_coefficients[span],
                    fisher_all,
                    l2_budget=l2_all,
                    fisher_budget=q_all,
                ).astype(np.float32)
                scalar_alpha_parts: list[np.ndarray] = []
                scalar_score_parts: list[np.ndarray] = []
                arbitrary_parts: list[np.ndarray] = []
                arbitrary_score_parts: list[np.ndarray] = []
                for batch_number, row_slice in enumerate(
                    _iter_slices(len(panel), batch_size), start=1
                ):
                    states, mask = _batch_hidden(
                        hidden,
                        lengths,
                        panel_to_unique,
                        row_slice,
                        device=device,
                        dtype=dtype,
                    )
                    token_idxs = torch.as_tensor(
                        panel["token_idx"].to_numpy(dtype=np.int64, copy=True)[row_slice],
                        dtype=torch.long,
                        device=device,
                    )
                    target_indices = torch.as_tensor(
                        target_index_all[row_slice], dtype=torch.long, device=device
                    )
                    alpha, scalar_score = _optimize_scalar_batch(
                        model=model,
                        states=states,
                        mask=mask,
                        token_idxs=token_idxs,
                        basis=basis_tensor,
                        direction_coefficients=natural_coefficients[span][row_slice],
                        maximum_scale=max_scale_all[row_slice],
                        target_indices=target_indices,
                        tracked=tracked,
                        grid_size=grid_size,
                        refinements=refinements,
                        refinement_size=refinement_size,
                    )
                    scalar_alpha_parts.append(alpha)
                    scalar_score_parts.append(scalar_score)

                    best_coefficients: np.ndarray | None = None
                    best_scores: np.ndarray | None = None
                    initializations = (
                        ("zero", "natural") if restarts == "zero,natural" else ("natural",)
                    )
                    for initialization in initializations:
                        initial = (
                            np.zeros_like(natural_coefficients[span][row_slice])
                            if initialization == "zero"
                            else natural_coefficients[span][row_slice]
                        )
                        coefficients, scores = _optimize_arbitrary_batch(
                            model=model,
                            states=states,
                            mask=mask,
                            token_idxs=token_idxs,
                            basis=basis_tensor,
                            initial_coefficients=initial,
                            fisher=fisher_all[row_slice],
                            l2_budget=l2_all[row_slice],
                            fisher_budget=q_all[row_slice],
                            target_indices=target_indices,
                            tracked=tracked,
                            steps=optimizer_steps,
                            lr=optimizer_lr,
                        )
                        if best_scores is None:
                            best_coefficients, best_scores = coefficients, scores
                        else:
                            improve = scores > best_scores
                            best_scores = np.where(improve, scores, best_scores)
                            best_coefficients = np.where(
                                improve[:, None], coefficients, best_coefficients
                            )
                    assert best_coefficients is not None and best_scores is not None
                    arbitrary_parts.append(best_coefficients.astype(np.float32))
                    arbitrary_score_parts.append(best_scores.astype(np.float32))
                    print(
                        f"[optimize] {span}/{budget} batch {batch_number}/"
                        f"{int(np.ceil(len(panel) / batch_size))}",
                        flush=True,
                    )
                prefix = f"{span}__{budget}"
                result_parts[f"scalar_alpha__{prefix}"] = scalar_alpha_parts
                result_parts[f"scalar_score__{prefix}"] = scalar_score_parts
                result_parts[f"arbitrary_coefficients__{prefix}"] = arbitrary_parts
                result_parts[f"arbitrary_score__{prefix}"] = arbitrary_score_parts
    arrays = {
        name: np.concatenate(parts, axis=0) for name, parts in result_parts.items()
    }
    for span, coefficients in natural_coefficients.items():
        arrays[f"natural_coefficients__{span}"] = coefficients
    _atomic_savez(output_path, **arrays)
    return arrays


def _condition_coefficients(
    optimizations: Mapping[str, np.ndarray],
    span: str,
    budget: str,
    kind: str,
) -> np.ndarray:
    prefix = f"{span}__{budget}"
    if kind == "arbitrary":
        return np.asarray(
            optimizations[f"arbitrary_coefficients__{prefix}"], dtype=np.float64
        )
    natural = np.asarray(
        optimizations[f"natural_coefficients__{span}"], dtype=np.float64
    )
    alpha = np.asarray(optimizations[f"scalar_alpha__{prefix}"], dtype=np.float64)
    return natural * alpha[:, None]


def _all_condition_deltas(
    *,
    bases: Mapping[str, np.ndarray],
    reference_deltas: Mapping[str, np.ndarray],
    optimizations: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows, hidden_dim = reference_deltas["exact_patch"].shape
    conditions: dict[str, np.ndarray] = {
        "no_edit": np.zeros((rows, hidden_dim), dtype=np.float64),
        "historical_fixed6": np.asarray(reference_deltas["fixed6"], dtype=np.float64),
        "full_exact_patch": np.asarray(reference_deltas["exact_patch"], dtype=np.float64),
    }
    for span in SPAN_NAMES:
        natural = np.asarray(
            optimizations[f"natural_coefficients__{span}"], dtype=np.float64
        )
        conditions[f"natural_projection__{span}"] = natural @ bases[span].T
        for budget in BUDGET_NAMES:
            for kind in ("scalar", "arbitrary"):
                coefficients = _condition_coefficients(
                    optimizations, span, budget, kind
                )
                conditions[f"{kind}__{span}__budget_{budget}"] = (
                    coefficients @ bases[span].T
                )
    return conditions


def _evaluate_conditions(
    *,
    output_dir: Path,
    panel: pd.DataFrame,
    hidden: np.ndarray,
    lengths: np.ndarray,
    panel_to_unique: np.ndarray,
    conditions: Mapping[str, np.ndarray],
    model,
    tokenizer,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    layer_idx: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    detail_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in conditions}
    kl_parts: dict[str, list[np.ndarray]] = {name: [] for name in conditions}
    with _suffix_context(model, layer_idx), torch.no_grad():
        for batch_number, row_slice in enumerate(_iter_slices(len(panel), batch_size), start=1):
            states, mask = _batch_hidden(
                hidden,
                lengths,
                panel_to_unique,
                row_slice,
                device=device,
                dtype=dtype,
            )
            token_idxs = torch.as_tensor(
                panel["token_idx"].to_numpy(dtype=np.int64, copy=True)[row_slice],
                dtype=torch.long,
                device=device,
            )
            baseline_logits = _suffix_logits(
                model=model,
                hidden_states=states,
                attention_mask=mask,
                logit_token_ids=None,
            ).float()
            baseline_log_probs = torch.log_softmax(baseline_logits, dim=1)
            baseline_probs = baseline_log_probs.exp()
            for name, delta_rows in conditions.items():
                delta = torch.as_tensor(
                    delta_rows[row_slice], dtype=dtype, device=device
                )
                logits = _suffix_logits(
                    model=model,
                    hidden_states=_edit_hidden(states, token_idxs, delta),
                    attention_mask=mask,
                    logit_token_ids=None,
                ).float()
                log_probs = torch.log_softmax(logits, dim=1)
                kl = torch.sum(
                    baseline_probs * (baseline_log_probs - log_probs), dim=1
                )
                result = evaluate_logits_against_shifted_targets(
                    tokenizer=tokenizer,
                    valid_df=panel.iloc[row_slice].copy(),
                    logits=logits.cpu().numpy(),
                    tracked_values=tracked_values,
                    tracked_token_ids=tracked_token_ids,
                )
                result["condition"] = name
                result["delta_norm"] = np.linalg.norm(
                    delta_rows[row_slice], axis=1
                )
                result["source_kl"] = kl.cpu().numpy()
                detail_parts[name].append(result)
                kl_parts[name].append(kl.cpu().numpy())
            print(
                f"[evaluate] batch {batch_number}/"
                f"{int(np.ceil(len(panel) / batch_size))}",
                flush=True,
            )
    details = pd.concat(
        [pd.concat(detail_parts[name], ignore_index=True) for name in conditions],
        ignore_index=True,
    )
    summary = summarize_steering_metrics(details, group_cols=["condition"])
    kl_summary = pd.DataFrame(
        {
            "condition": list(conditions),
            "mean_source_kl": [
                float(np.concatenate(kl_parts[name]).mean()) for name in conditions
            ],
            "median_delta_norm": [
                float(np.median(np.linalg.norm(conditions[name], axis=1)))
                for name in conditions
            ],
        }
    )
    summary = summary.merge(kl_summary, on="condition", how="left")
    atomic_write_csv(output_dir / "causal_details.csv", details)
    atomic_write_csv(output_dir / "causal_summary.csv", summary)
    return details, summary


def _write_constraint_diagnostics(
    *,
    output_dir: Path,
    bases: Mapping[str, np.ndarray],
    metrics: Mapping[str, np.ndarray],
    optimizations: Mapping[str, np.ndarray],
) -> None:
    frames: list[pd.DataFrame] = []
    for span in SPAN_NAMES:
        fisher = np.asarray(metrics[f"fisher__{span}"], dtype=np.float64)
        natural = np.asarray(
            optimizations[f"natural_coefficients__{span}"], dtype=np.float64
        )
        for budget in BUDGET_NAMES:
            l2 = np.asarray(metrics[f"reference_l2__{budget}"], dtype=np.float64)
            q = np.asarray(
                metrics[f"reference_fisher_q__{budget}"], dtype=np.float64
            )
            for kind in ("scalar", "arbitrary"):
                coefficients = _condition_coefficients(
                    optimizations, span, budget, kind
                )
                diagnostics = joint_constraint_diagnostics(
                    coefficients,
                    fisher,
                    l2_budget=l2,
                    fisher_budget=q,
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "row_idx": np.arange(len(coefficients)),
                            "condition": f"{kind}__{span}__budget_{budget}",
                            "span": span,
                            "budget": budget,
                            "kind": kind,
                            "span_rank": bases[span].shape[1],
                            "l2_norm": diagnostics.l2_norm,
                            "fisher_quadratic": diagnostics.fisher_quadratic,
                            "l2_budget": l2,
                            "fisher_budget": q,
                            "l2_utilization": diagnostics.l2_utilization,
                            "fisher_utilization": diagnostics.fisher_utilization,
                            "feasible": diagnostics.feasible,
                        }
                    )
                )
            natural_diag = joint_constraint_diagnostics(
                natural, fisher, l2_budget=l2, fisher_budget=q
            )
            frames.append(
                pd.DataFrame(
                    {
                        "row_idx": np.arange(len(natural)),
                        "condition": f"natural_projection__{span}",
                        "span": span,
                        "budget": budget,
                        "kind": "natural_unconstrained",
                        "span_rank": bases[span].shape[1],
                        "l2_norm": natural_diag.l2_norm,
                        "fisher_quadratic": natural_diag.fisher_quadratic,
                        "l2_budget": l2,
                        "fisher_budget": q,
                        "l2_utilization": natural_diag.l2_utilization,
                        "fisher_utilization": natural_diag.fisher_utilization,
                        "feasible": natural_diag.feasible,
                    }
                )
            )
    atomic_write_csv(
        output_dir / "constraint_diagnostics.csv",
        pd.concat(frames, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching this deterministic CUDA run."
        )
    set_global_seed(SEED)
    torch.use_deterministic_algorithms(True)
    output_dir = ensure_dir(args.output_dir)
    cfg, source_model_entry, resolved = load_source_config(args.source_run_dir)
    models = load_yaml(args.models_config)
    model_entry = _resolve_model_entry(models, str(cfg["model"]))
    if str(model_entry["hf_id"]) != str(source_model_entry["hf_id"]):
        raise ValueError("Resolved model config does not match the source run.")
    panel = load_panel(args.source_run_dir, args.max_rows)
    tokenizer = load_tokenizer(
        str(model_entry["hf_id"]),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        revision=str(resolved["model_revision"]),
    )
    tracked_values, tracked_token_ids = _numeric_tracking_arrays(tokenizer)
    _numeric_target_indices(panel, tracked_token_ids)
    bases, reference_deltas, source_activations, target_activations = (
        _build_bases_and_deltas(
            panel=panel,
            base_bag_run_dir=args.base_bag_run_dir,
            oracle_run_dir=args.oracle_run_dir,
            centroid_artifact=args.centroid_artifact,
            competitive_state=args.competitive_state,
        )
    )
    device = str(args.device or model_entry.get("device", "cuda"))
    loaded = load_causal_lm(
        model_name=str(model_entry["hf_id"]),
        device=device,
        dtype=str(model_entry.get("dtype", "float16")),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        attn_implementation=model_entry.get("attn_implementation"),
        revision=str(resolved["model_revision"]),
    )
    freeze_model_parameters(loaded.model)
    layer_idx = int(cfg["intervention_layer_idx"])
    signature = _panel_signature(
        panel,
        model_revision=str(resolved["model_revision"]),
        layer_idx=layer_idx,
        attn_implementation=model_entry.get("attn_implementation"),
        tokenizer=tokenizer,
    )
    optimizer_config = {
        "steps": args.optimizer_steps,
        "lr": args.optimizer_lr,
        "restarts": args.optimizer_restarts,
        "scalar_grid_size": args.scalar_grid_size,
        "scalar_refinements": args.scalar_refinements,
        "scalar_refinement_size": args.scalar_refinement_size,
    }
    run_signature = _experiment_signature(
        panel_signature=signature,
        model_revision=str(resolved["model_revision"]),
        layer_idx=layer_idx,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        bases=bases,
        reference_deltas=reference_deltas,
        tracked_token_ids=tracked_token_ids,
        optimizer_config=optimizer_config,
    )
    _guard_output_signature(output_dir, run_signature)
    hidden, lengths, panel_to_unique = _prepare_hidden_cache(
        cache_path=output_dir / "source_hidden_sequences.npz",
        signature=signature,
        panel=panel,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        layer_idx=layer_idx,
        batch_size=args.capture_batch_size,
        device=device,
        force=bool(args.force_hidden_cache),
    )
    metadata = {
        "design": {
            "unit": "original Figure 1 ordered source-to-target pair",
            "panel_rows": len(panel),
            "spans": {
                "old": "PLS magnitude direction plus orthonormal union of the nine historical property-probe row spaces",
                "competitive": "full split_00 competitive-deflation property basis",
            },
            "interventions": [
                "literal natural target-patch projection",
                "historical fixed 6x property edit",
                "per-row scalar oracle along the natural projection ray",
                "per-row arbitrary coefficient oracle inside each span",
                "full exact contextual patch",
            ],
            "constraints": (
                "intersection of a row-specific L2 ball and source-local tracked-number "
                "categorical-Fisher ellipsoid"
            ),
            "budget_references": list(BUDGET_NAMES),
            "optimization_objective": "target conditional log probability among tracked numeric tokens",
            "final_evaluation": "full-vocabulary next-token logits",
            "interpretation": (
                "scalar and arbitrary searches are target-aware per-row oracle ceilings, "
                "not deployable steering methods"
            ),
        },
        "model": str(model_entry["hf_id"]),
        "model_revision": str(resolved["model_revision"]),
        "attention_implementation": model_entry.get("attn_implementation"),
        "layer_idx": layer_idx,
        "seed": SEED,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "panel_signature": signature,
        "run_signature": run_signature,
        "span_ranks": {name: basis.shape[1] for name, basis in bases.items()},
        "source_run": str(args.source_run_dir),
        "base_bag_run": str(args.base_bag_run_dir),
        "oracle_run": str(args.oracle_run_dir),
        "competitive_state": str(args.competitive_state),
        "optimizer": optimizer_config,
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    _atomic_savez(
        output_dir / "geometry.npz",
        old_basis=bases["old"].astype(np.float32),
        competitive_basis=bases["competitive"].astype(np.float32),
        source_activations=source_activations.astype(np.float32),
        target_activations=target_activations.astype(np.float32),
        exact_patch=reference_deltas["exact_patch"].astype(np.float32),
        fixed6=reference_deltas["fixed6"].astype(np.float32),
    )
    if args.stage == "prepare":
        print(output_dir)
        return

    metric_path = output_dir / "local_metrics.npz"
    if metric_path.exists() and not args.force_metric:
        with np.load(metric_path) as saved:
            metrics = {key: np.asarray(saved[key]) for key in saved.files}
    else:
        metrics = _compute_local_metrics(
            output_path=metric_path,
            panel=panel,
            hidden=hidden,
            lengths=lengths,
            panel_to_unique=panel_to_unique,
            bases=bases,
            reference_deltas=reference_deltas,
            model=loaded.model,
            tracked_token_ids=tracked_token_ids,
            layer_idx=layer_idx,
            batch_size=args.batch_size,
        )
    if args.stage == "metric":
        print(output_dir)
        return

    optimization_path = output_dir / "optimizations.npz"
    if optimization_path.exists() and not args.force_optimize:
        with np.load(optimization_path) as saved:
            optimizations = {key: np.asarray(saved[key]) for key in saved.files}
    else:
        optimizations = _run_optimizations(
            output_path=optimization_path,
            panel=panel,
            hidden=hidden,
            lengths=lengths,
            panel_to_unique=panel_to_unique,
            bases=bases,
            exact_patch=reference_deltas["exact_patch"],
            metrics=metrics,
            model=loaded.model,
            tracked_token_ids=tracked_token_ids,
            layer_idx=layer_idx,
            batch_size=args.batch_size,
            grid_size=args.scalar_grid_size,
            refinements=args.scalar_refinements,
            refinement_size=args.scalar_refinement_size,
            optimizer_steps=args.optimizer_steps,
            optimizer_lr=args.optimizer_lr,
            restarts=args.optimizer_restarts,
        )
    _write_constraint_diagnostics(
        output_dir=output_dir,
        bases=bases,
        metrics=metrics,
        optimizations=optimizations,
    )
    if args.stage == "optimize":
        print(output_dir)
        return

    conditions = _all_condition_deltas(
        bases=bases,
        reference_deltas=reference_deltas,
        optimizations=optimizations,
    )
    causal_summary_path = output_dir / "causal_summary.csv"
    if args.force_evaluate or not causal_summary_path.exists():
        _evaluate_conditions(
            output_dir=output_dir,
            panel=panel,
            hidden=hidden,
            lengths=lengths,
            panel_to_unique=panel_to_unique,
            conditions=conditions,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            tracked_values=tracked_values,
            tracked_token_ids=tracked_token_ids,
            layer_idx=layer_idx,
            batch_size=args.batch_size,
        )
    (output_dir / "DONE").write_text("done\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
