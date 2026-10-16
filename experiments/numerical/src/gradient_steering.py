from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.activation_steering import (
    _batch_forward_with_optional_direction,
    _parse_optional_int,
    _softmax_numpy,
    collect_prompt_activations,
    get_model_layers,
    hidden_states_hook,
)
from src.steering_sanity import (
    PreparedSteeringSlice,
    add_baseline_metadata,
    build_shifted_prompts,
    evaluate_logits_against_shifted_targets,
    make_valid_shifted_rows,
    orthonormal_row_basis,
    prepare_steering_slice,
)
from src.utils import atomic_write_csv, ensure_dir


@dataclass(frozen=True)
class GradientSearchSpec:
    test_id: str
    hypothesis_id: str
    hypothesis: str
    reason: str
    kind: str
    layer: str
    params: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def resolve_named_layer(model, layer: str) -> int:
    total = len(get_model_layers(model))
    if layer == "embed":
        return 0
    if layer == "early":
        return min(1, total - 1)
    if layer == "mid":
        return min(total - 1, max(1, 1 + ((total - 2) // 2)))
    if layer == "late":
        return total - 1
    return int(layer)


def get_default_gradient_search_tests() -> list[GradientSearchSpec]:
    specs = [
        GradientSearchSpec(
            test_id="prompt_counterfactual",
            hypothesis_id="G0",
            hypothesis="If the evaluation setup is valid, editing the prompt text to the shifted operand should produce the shifted answer.",
            reason="This is the direct textual upper bound for any hidden-state search method.",
            kind="prompt_counterfactual",
            layer="none",
            params={},
        ),
        GradientSearchSpec(
            test_id="counterfactual_patch_mid",
            hypothesis_id="G0",
            hypothesis="If the exact shifted hidden state is causally sufficient, patching that mid-layer state should transfer the answer.",
            reason="This is the strongest hidden-state reference intervention available without search.",
            kind="counterfactual_patch",
            layer="mid",
            params={},
        ),
        GradientSearchSpec(
            test_id="grad_target_logprob_embed",
            hypothesis_id="G1",
            hypothesis="If a local first-order steering direction exists at the embedding layer, the target-logprob gradient should increase the shifted answer probability.",
            reason="This uses the direct counterfactual loss gradient at the intervention site.",
            kind="gradient_linesearch",
            layer="embed",
            params={"objective": "target_logprob", "direction_mode": "raw"},
        ),
        GradientSearchSpec(
            test_id="grad_target_logprob_mid",
            hypothesis_id="G1",
            hypothesis="If a local first-order steering direction exists at a contextualized layer, the target-logprob gradient should increase the shifted answer probability.",
            reason="This tests whether target-seeking gradients become more causal after contextualization.",
            kind="gradient_linesearch",
            layer="mid",
            params={"objective": "target_logprob", "direction_mode": "raw"},
        ),
        GradientSearchSpec(
            test_id="grad_margin_mid",
            hypothesis_id="G2",
            hypothesis="If the right local direction is primarily about suppressing the source answer while boosting the shifted answer, a logit-margin gradient should outperform plain target-logprob gradients.",
            reason="This optimizes the target answer against the source answer directly.",
            kind="gradient_linesearch",
            layer="mid",
            params={"objective": "target_margin_logit", "direction_mode": "raw"},
        ),
        GradientSearchSpec(
            test_id="grad_sign_mid",
            hypothesis_id="G2",
            hypothesis="If gradient geometry is noisy but coordinate signs are informative, sign gradients should still steer the answer.",
            reason="This is an FGSM-style control on the gradient geometry.",
            kind="gradient_linesearch",
            layer="mid",
            params={"objective": "target_logprob", "direction_mode": "sign"},
        ),
    ]
    for budget in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        slug = str(int(budget))
        specs.append(
            GradientSearchSpec(
                test_id=f"optimize_target_logprob_mid_norm{slug}",
                hypothesis_id="G3",
                hypothesis="If a searched intervention is causally relevant, its effect should persist under explicit norm budgets.",
                reason="This sweeps the allowed intervention size for unconstrained per-example target-logprob search.",
                kind="optimize_delta",
                layer="mid",
                params={"objective": "target_logprob", "max_norm_override": float(budget)},
            )
        )
    for penalty in [0.001, 0.01]:
        penalty_slug = str(penalty).replace(".", "p")
        specs.append(
            GradientSearchSpec(
                test_id=f"optimize_target_logprob_mid_patchreg_{penalty_slug}",
                hypothesis_id="G4",
                hypothesis="If the search should stay close to the true counterfactual state, attraction to the actual patch should retain causal effect while improving naturality.",
                reason="This penalizes distance from the real counterfactual mid-layer patch while still optimizing the shifted answer.",
                kind="optimize_delta",
                layer="mid",
                params={"objective": "target_logprob", "patch_l2_penalty": float(penalty), "max_norm_override": 32.0},
            )
        )
        specs.append(
            GradientSearchSpec(
                test_id=f"optimize_margin_mid_patchreg_{penalty_slug}",
                hypothesis_id="G4",
                hypothesis="If the search should stay close to the true counterfactual state, attraction to the actual patch should retain causal effect while improving naturality.",
                reason="This combines patch attraction with a target-vs-source margin objective.",
                kind="optimize_delta",
                layer="mid",
                params={"objective": "target_margin_logit", "patch_l2_penalty": float(penalty), "max_norm_override": 32.0},
            )
        )
    specs.extend(
        [
            GradientSearchSpec(
                test_id="shared_target_logprob_mid",
                hypothesis_id="G5",
                hypothesis="If a condition-level steering direction exists, one shared vector should improve the shifted answer across many examples.",
                reason="This tests whether the search result can be compressed into a single universal delta per task/operand/alpha condition.",
                kind="optimize_shared_delta",
                layer="mid",
                params={"objective": "target_logprob", "max_norm_override": 32.0},
            ),
            GradientSearchSpec(
                test_id="shared_margin_mid",
                hypothesis_id="G5",
                hypothesis="If a condition-level steering direction exists, one shared vector should improve the shifted answer across many examples.",
                reason="This uses a shared vector but trains against the source answer explicitly.",
                kind="optimize_shared_delta",
                layer="mid",
                params={"objective": "target_margin_logit", "max_norm_override": 32.0},
            ),
            GradientSearchSpec(
                test_id="subspace_rank4_target_logprob_mid",
                hypothesis_id="G6",
                hypothesis="If searched interventions mainly live in a low-dimensional causal family, restricting optimization to a low-rank patch-derived subspace should preserve much of the effect.",
                reason="This optimizes per-example coefficients in the top-4 PCA/SVD subspace of true counterfactual patches.",
                kind="optimize_subspace_delta",
                layer="mid",
                params={"objective": "target_logprob", "rank": 4, "max_norm_override": 32.0},
            ),
            GradientSearchSpec(
                test_id="subspace_rank8_target_logprob_mid",
                hypothesis_id="G6",
                hypothesis="If searched interventions mainly live in a low-dimensional causal family, restricting optimization to a low-rank patch-derived subspace should preserve much of the effect.",
                reason="This optimizes per-example coefficients in the top-8 PCA/SVD subspace of true counterfactual patches.",
                kind="optimize_subspace_delta",
                layer="mid",
                params={"objective": "target_logprob", "rank": 8, "max_norm_override": 32.0},
            ),
            GradientSearchSpec(
                test_id="subspace_rank4_margin_mid",
                hypothesis_id="G6",
                hypothesis="If searched interventions mainly live in a low-dimensional causal family, restricting optimization to a low-rank patch-derived subspace should preserve much of the effect.",
                reason="This uses the same top-4 patch subspace but optimizes a target-vs-source margin.",
                kind="optimize_subspace_delta",
                layer="mid",
                params={"objective": "target_margin_logit", "rank": 4, "max_norm_override": 32.0},
            ),
        ]
    )
    return specs


def _alpha_slug(alpha: float) -> str:
    text = f"{float(alpha):+.3f}"
    return text.replace("+", "p").replace("-", "m").replace(".", "_")


def condition_stem(*, spec: GradientSearchSpec, seed: int, task: str, operand_mode: str, alpha: float) -> str:
    return f"seed{int(seed)}__{task}__{operand_mode}__{spec.test_id}__alpha{_alpha_slug(alpha)}"


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    normalized = arr / safe
    normalized[norms.squeeze(-1) <= 0] = 0.0
    return normalized


def top_patch_subspace_basis(patch_deltas: np.ndarray, rank: int) -> np.ndarray:
    basis = orthonormal_row_basis(np.asarray(patch_deltas, dtype=np.float64))
    keep = min(int(rank), basis.shape[0])
    if keep <= 0:
        raise ValueError("Requested patch subspace rank must be positive.")
    return np.asarray(basis[:keep], dtype=np.float64)


def _log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - logsumexp


def objective_scores_numpy(
    logits: np.ndarray,
    *,
    target_token_ids: np.ndarray,
    baseline_token_ids: np.ndarray,
    objective: str,
) -> np.ndarray:
    rows = np.arange(logits.shape[0], dtype=np.int64)
    if objective == "target_logprob":
        log_probs = _log_softmax_numpy(logits)
        return log_probs[rows, target_token_ids]
    if objective == "target_logit":
        return logits[rows, target_token_ids]
    if objective == "target_margin_logit":
        return logits[rows, target_token_ids] - logits[rows, baseline_token_ids]
    raise ValueError(f"Unsupported objective: {objective}")


def objective_scores_torch(
    logits: torch.Tensor,
    *,
    target_token_ids: torch.Tensor,
    baseline_token_ids: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    if objective == "target_logprob":
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[rows, target_token_ids]
    if objective == "target_logit":
        return logits[rows, target_token_ids]
    if objective == "target_margin_logit":
        return logits[rows, target_token_ids] - logits[rows, baseline_token_ids]
    raise ValueError(f"Unsupported objective: {objective}")


def freeze_model_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _prepare_token_tensor(token_idxs: list[int], seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(token_idxs, dtype=torch.long, device=device).clamp(min=0, max=max(seq_len - 1, 0))


def forward_next_token_logits_with_delta(
    *,
    model,
    tokenizer,
    prompts: list[str],
    token_idxs: list[int],
    layer_idx: int,
    batch_size: int,
    device: str,
    delta_rows: np.ndarray | torch.Tensor | None = None,
    capture_selected: bool = False,
    logit_token_ids: np.ndarray | torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    logits_chunks: list[np.ndarray] = []
    selected_chunks: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    logit_token_tensor = None
    if logit_token_ids is not None:
        logit_ids = logit_token_ids.detach().cpu().numpy() if torch.is_tensor(logit_token_ids) else logit_token_ids
        logit_token_tensor = torch.as_tensor(np.asarray(logit_ids, dtype=np.int64), dtype=torch.long, device=model_device)

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_token_idxs = token_idxs[start : start + batch_size]
        batch_delta = None if delta_rows is None else delta_rows[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        model_kwargs = {k: v.to(device) for k, v in encoded.items()}
        token_idx_tensor = _prepare_token_tensor(batch_token_idxs, seq_len=model_kwargs["input_ids"].shape[1], device=model_device)

        captured: dict[str, torch.Tensor] = {}

        class _DeltaHook:
            def __init__(self, delta) -> None:
                self.delta = delta

            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                if output.ndim < 3:
                    return output
                edited = output
                if self.delta is not None:
                    edited = output.clone()
                    idx = torch.arange(edited.shape[0], device=edited.device)
                    delta_tensor = self.delta
                    if not torch.is_tensor(delta_tensor):
                        delta_tensor = torch.as_tensor(delta_tensor, dtype=torch.float32, device=edited.device)
                    delta_tensor = delta_tensor.to(edited)
                    edited[idx, token_idx_tensor] = edited[idx, token_idx_tensor] + delta_tensor
                if capture_selected:
                    idx = torch.arange(edited.shape[0], device=edited.device)
                    captured["selected"] = edited[idx, token_idx_tensor]
                return edited

        hook = get_model_layers(model)[layer_idx].register_forward_hook(_DeltaHook(batch_delta), with_kwargs=True)
        try:
            outputs = model(**model_kwargs)
        finally:
            hook.remove()

        attention_mask = model_kwargs["attention_mask"]
        last_positions = attention_mask.sum(dim=1) - 1
        next_logits = outputs.logits[torch.arange(outputs.logits.shape[0], device=model_device), last_positions]
        if logit_token_tensor is not None:
            next_logits = next_logits.index_select(dim=1, index=logit_token_tensor)
        logits_chunks.append(next_logits.detach().float().cpu().numpy())
        if capture_selected and "selected" in captured:
            selected_chunks.append(captured["selected"].detach().float().cpu().numpy())

    selected = None if not selected_chunks else np.concatenate(selected_chunks, axis=0)
    return np.concatenate(logits_chunks, axis=0), selected


def _pad_hidden_sequences(
    hidden_sequences: list[np.ndarray | torch.Tensor],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden_sequences:
        raise ValueError("hidden_sequences must be non-empty.")
    lengths = [int(sequence.shape[0]) for sequence in hidden_sequences]
    hidden_dim = int(hidden_sequences[0].shape[-1])
    max_len = max(lengths)
    hidden_states = torch.zeros((len(hidden_sequences), max_len, hidden_dim), dtype=dtype, device=device)
    attention_mask = torch.zeros((len(hidden_sequences), max_len), dtype=torch.long, device=device)
    for row_idx, sequence in enumerate(hidden_sequences):
        seq_tensor = sequence if torch.is_tensor(sequence) else torch.as_tensor(sequence)
        seq_tensor = seq_tensor.to(device=device, dtype=dtype)
        seq_len = int(seq_tensor.shape[0])
        if int(seq_tensor.shape[-1]) != hidden_dim:
            raise ValueError("All hidden sequences must have the same hidden dimension.")
        hidden_states[row_idx, :seq_len] = seq_tensor
        attention_mask[row_idx, :seq_len] = 1
    return hidden_states, attention_mask


@contextmanager
def _temporary_decoder_suffix(model, layer_idx: int):
    if not hasattr(model, "model") or not hasattr(model.model, "layers") or not hasattr(model, "lm_head"):
        raise ValueError("cached_suffix requires a causal LM with model.layers and lm_head.")
    core = model.model
    original_layers = core.layers
    original_num_hidden_layers = getattr(core.config, "num_hidden_layers", None)
    original_layer_types = getattr(core.config, "layer_types", None)
    start_decoder_layer = int(layer_idx)
    if start_decoder_layer < 0 or start_decoder_layer > len(original_layers):
        raise ValueError(f"Invalid suffix start layer {start_decoder_layer} for {len(original_layers)} decoder layers.")
    suffix_layers = list(original_layers)[start_decoder_layer:]
    try:
        core.layers = torch.nn.ModuleList(suffix_layers)
        if original_num_hidden_layers is not None:
            core.config.num_hidden_layers = len(suffix_layers)
        if original_layer_types is not None:
            core.config.layer_types = list(original_layer_types)[start_decoder_layer:]
        yield
    finally:
        core.layers = original_layers
        if original_num_hidden_layers is not None:
            core.config.num_hidden_layers = original_num_hidden_layers
        if original_layer_types is not None:
            core.config.layer_types = original_layer_types


def _decoder_suffix_next_token_logits(
    *,
    model,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    logit_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    with _temporary_decoder_suffix(model, int(layer_idx)):
        outputs = model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=hidden_states,
            use_cache=False,
        )
    hidden_states = outputs.last_hidden_state
    last_positions = attention_mask.sum(dim=1) - 1
    last_hidden = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_positions]
    if logit_token_ids is None:
        return model.lm_head(last_hidden)
    if not hasattr(model.lm_head, "weight"):
        raise ValueError("numeric-only cached_suffix logits require an lm_head with a weight matrix.")
    weight = model.lm_head.weight.index_select(dim=0, index=logit_token_ids)
    bias = getattr(model.lm_head, "bias", None)
    selected_bias = None if bias is None else bias.index_select(dim=0, index=logit_token_ids)
    logits = last_hidden @ weight.to(last_hidden).T
    if selected_bias is not None:
        logits = logits + selected_bias.to(logits)
    return logits


def forward_next_token_logits_from_cached_hidden_sequences(
    *,
    model,
    hidden_sequences: list[np.ndarray | torch.Tensor],
    token_idxs: list[int],
    layer_idx: int,
    batch_size: int,
    device: str,
    delta_rows: np.ndarray | torch.Tensor | None = None,
    capture_selected: bool = False,
    logit_token_ids: np.ndarray | torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resume a decoder-only forward pass from cached hidden sequences.

    ``layer_idx`` uses the same convention as ``get_model_layers``:
    0 is the embedding output, and 1..N are decoder-layer outputs. The suffix
    therefore starts at decoder layer ``layer_idx``.
    """

    logits_chunks: list[np.ndarray] = []
    selected_chunks: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    logit_token_tensor = None
    if logit_token_ids is not None:
        logit_ids = logit_token_ids.detach().cpu().numpy() if torch.is_tensor(logit_token_ids) else logit_token_ids
        logit_token_tensor = torch.as_tensor(np.asarray(logit_ids, dtype=np.int64), dtype=torch.long, device=model_device)

    for start in range(0, len(hidden_sequences), batch_size):
        batch_sequences = hidden_sequences[start : start + batch_size]
        batch_token_idxs = token_idxs[start : start + batch_size]
        batch_delta = None if delta_rows is None else delta_rows[start : start + batch_size]
        hidden_states, attention_mask = _pad_hidden_sequences(batch_sequences, device=model_device, dtype=model_dtype)
        token_idx_tensor = _prepare_token_tensor(batch_token_idxs, seq_len=hidden_states.shape[1], device=model_device)
        if batch_delta is not None:
            delta_tensor = batch_delta if torch.is_tensor(batch_delta) else torch.as_tensor(batch_delta)
            delta_tensor = delta_tensor.to(device=model_device, dtype=model_dtype)
            edited = hidden_states.clone()
            idx = torch.arange(edited.shape[0], device=model_device)
            edited[idx, token_idx_tensor] = edited[idx, token_idx_tensor] + delta_tensor
            hidden_states = edited
        if capture_selected:
            idx = torch.arange(hidden_states.shape[0], device=model_device)
            selected_chunks.append(hidden_states[idx, token_idx_tensor].detach().float().cpu().numpy())
        with torch.no_grad():
            next_logits = _decoder_suffix_next_token_logits(
                model=model,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                logit_token_ids=logit_token_tensor,
            )
        logits_chunks.append(next_logits.detach().float().cpu().numpy())

    selected = None if not selected_chunks else np.concatenate(selected_chunks, axis=0)
    return np.concatenate(logits_chunks, axis=0), selected


def forward_next_token_logits_torch_with_delta(
    *,
    model,
    tokenizer,
    prompts: list[str],
    token_idxs: list[int],
    layer_idx: int,
    device: str,
    delta_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    model_device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    model_kwargs = {k: v.to(device) for k, v in encoded.items()}
    token_idx_tensor = _prepare_token_tensor(token_idxs, seq_len=model_kwargs["input_ids"].shape[1], device=model_device)

    class _DeltaHook:
        def __init__(self, delta) -> None:
            self.delta = delta

        @hidden_states_hook
        def __call__(self, module, args, kwargs, output):
            del module, args, kwargs
            if output.ndim < 3 or self.delta is None:
                return output
            edited = output.clone()
            idx = torch.arange(edited.shape[0], device=edited.device)
            edited[idx, token_idx_tensor] = edited[idx, token_idx_tensor] + self.delta.to(edited)
            return edited

    hook = get_model_layers(model)[layer_idx].register_forward_hook(_DeltaHook(delta_rows), with_kwargs=True)
    try:
        outputs = model(**model_kwargs)
    finally:
        hook.remove()

    attention_mask = model_kwargs["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1
    return outputs.logits[torch.arange(outputs.logits.shape[0], device=model_device), last_positions]


def collect_gradient_directions(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    objective: str,
    delta_rows: np.ndarray | None = None,
) -> np.ndarray:
    gradients: list[np.ndarray] = []
    model_device = next(model.parameters()).device

    for start in range(0, len(valid_df), batch_size):
        batch_df = valid_df.iloc[start : start + batch_size].reset_index(drop=True)
        batch_delta = None if delta_rows is None else np.asarray(delta_rows[start : start + batch_size], dtype=np.float32)
        prompts = batch_df["prompt"].tolist()
        token_idxs = batch_df["token_idx"].astype(int).tolist()
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        model_kwargs = {k: v.to(device) for k, v in encoded.items()}
        token_idx_tensor = _prepare_token_tensor(token_idxs, seq_len=model_kwargs["input_ids"].shape[1], device=model_device)
        target_token_ids = torch.tensor(batch_df["target_token_id"].to_list(), dtype=torch.long, device=model_device)
        baseline_token_ids = torch.tensor(batch_df["baseline_predicted_token_id"].to_list(), dtype=torch.long, device=model_device)
        captured: dict[str, torch.Tensor] = {}

        class _CaptureHook:
            def __init__(self, delta: np.ndarray | None) -> None:
                self.delta = delta

            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                if output.ndim < 3:
                    return output
                edited = output.clone()
                idx = torch.arange(output.shape[0], device=output.device)
                selected = edited[idx, token_idx_tensor]
                if self.delta is not None:
                    delta_tensor = torch.as_tensor(self.delta, dtype=selected.dtype, device=selected.device)
                    if delta_tensor.ndim == 1:
                        delta_tensor = delta_tensor.unsqueeze(0).expand_as(selected)
                    selected = selected + delta_tensor
                selected = selected.detach().clone().requires_grad_(True)
                edited[idx, token_idx_tensor] = selected
                captured["selected"] = selected
                return edited

        hook = get_model_layers(model)[layer_idx].register_forward_hook(_CaptureHook(batch_delta), with_kwargs=True)
        try:
            outputs = model(**model_kwargs)
            attention_mask = model_kwargs["attention_mask"]
            last_positions = attention_mask.sum(dim=1) - 1
            next_logits = outputs.logits[torch.arange(outputs.logits.shape[0], device=model_device), last_positions]
            objective_scores = objective_scores_torch(
                next_logits,
                target_token_ids=target_token_ids,
                baseline_token_ids=baseline_token_ids,
                objective=objective,
            )
            gradients_batch = torch.autograd.grad(objective_scores.sum(), captured["selected"])[0]
        finally:
            hook.remove()
            model.zero_grad(set_to_none=True)
        gradients.append(gradients_batch.detach().float().cpu().numpy())
    return np.concatenate(gradients, axis=0).astype(np.float64)


def collect_numeric_logit_directional_derivatives(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
    tracked_token_ids: np.ndarray | None,
    direction_rows: np.ndarray,
    delta_rows: np.ndarray | None = None,
) -> np.ndarray:
    directions = np.asarray(direction_rows, dtype=np.float32)
    if len(valid_df) != len(directions):
        raise ValueError("direction_rows must align one-to-one with valid_df.")

    directional_chunks: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    tracked_token_tensor = None
    if tracked_token_ids is not None:
        tracked_ids = np.asarray(tracked_token_ids, dtype=np.int64)
        tracked_token_tensor = torch.as_tensor(tracked_ids.tolist(), dtype=torch.long, device=model_device)

    for start in range(0, len(valid_df), batch_size):
        batch_df = valid_df.iloc[start : start + batch_size].reset_index(drop=True)
        batch_delta = None if delta_rows is None else np.asarray(delta_rows[start : start + batch_size], dtype=np.float32)
        batch_direction = torch.as_tensor(directions[start : start + batch_size], dtype=torch.float32, device=model_device)
        prompts = batch_df["prompt"].tolist()
        token_idxs = batch_df["token_idx"].astype(int).tolist()
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        model_kwargs = {k: v.to(device) for k, v in encoded.items()}
        token_idx_tensor = _prepare_token_tensor(token_idxs, seq_len=model_kwargs["input_ids"].shape[1], device=model_device)
        captured: dict[str, torch.Tensor] = {}

        class _CaptureHook:
            def __init__(self, delta: np.ndarray | None) -> None:
                self.delta = delta

            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                if output.ndim < 3:
                    return output
                edited = output.clone()
                idx = torch.arange(edited.shape[0], device=edited.device)
                selected = edited[idx, token_idx_tensor]
                if self.delta is not None:
                    delta_tensor = torch.as_tensor(self.delta, dtype=selected.dtype, device=selected.device)
                    if delta_tensor.ndim == 1:
                        delta_tensor = delta_tensor.unsqueeze(0).expand_as(selected)
                    selected = selected + delta_tensor
                edited[idx, token_idx_tensor] = selected
                captured["selected"] = selected.detach()
                return edited

        hook = get_model_layers(model)[layer_idx].register_forward_hook(_CaptureHook(batch_delta), with_kwargs=True)
        try:
            with torch.no_grad():
                model(**model_kwargs)
        finally:
            hook.remove()

        base_selected = captured["selected"].detach().clone().requires_grad_(True)
        tangent = batch_direction.to(base_selected)
        if tangent.ndim == 1:
            tangent = tangent.unsqueeze(0).expand_as(base_selected)

        def _forward_from_selected(selected_states: torch.Tensor) -> torch.Tensor:
            class _OverrideHook:
                @hidden_states_hook
                def __call__(self, module, args, kwargs, output):
                    del module, args, kwargs
                    if output.ndim < 3:
                        return output
                    edited = output.clone()
                    idx = torch.arange(edited.shape[0], device=edited.device)
                    edited[idx, token_idx_tensor] = selected_states.to(edited)
                    return edited

            inner_hook = get_model_layers(model)[layer_idx].register_forward_hook(_OverrideHook(), with_kwargs=True)
            try:
                outputs = model(**model_kwargs)
            finally:
                inner_hook.remove()

            attention_mask = model_kwargs["attention_mask"]
            last_positions = attention_mask.sum(dim=1) - 1
            next_logits = outputs.logits[torch.arange(outputs.logits.shape[0], device=model_device), last_positions]
            next_logits = next_logits.to(torch.float32)
            if tracked_token_tensor is not None:
                next_logits = next_logits.index_select(dim=1, index=tracked_token_tensor)
            return next_logits

        with torch.enable_grad():
            _, batch_jvp = torch.autograd.functional.jvp(
                _forward_from_selected,
                (base_selected,),
                (tangent,),
                create_graph=False,
                strict=False,
            )
        directional_chunks.append(batch_jvp.detach().float().cpu().numpy())
        model.zero_grad(set_to_none=True)

    return np.concatenate(directional_chunks, axis=0).astype(np.float64)


def evaluate_prompt_counterfactual(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    shifted_prompt_df = build_shifted_prompts(valid_df, tokenizer=tokenizer, operand_mode=str(valid_df["operand_mode"].iloc[0]))
    merged = valid_df.merge(shifted_prompt_df, on="example_id", how="left")
    logits_chunks, _, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=merged["shifted_prompt"].tolist(),
        token_idxs=merged["shifted_token_idx"].astype(int).tolist(),
        layer_idx=0,
        batch_size=batch_size,
        device=device,
        scaled_directions=None,
        capture_pre_activations=False,
    )
    logits = np.concatenate(logits_chunks, axis=0)
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=merged,
        logits=logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    result["search_objective"] = np.nan
    result["delta_norm"] = 0.0
    result["search_score_gain"] = np.nan
    result["selected_step_size"] = 0.0
    return result


def evaluate_counterfactual_patch(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    source = collect_prompt_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["prompt"].tolist(),
        token_idxs=valid_df["token_idx"].astype(int).tolist(),
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
    )
    shifted_prompt_df = build_shifted_prompts(valid_df, tokenizer=tokenizer, operand_mode=str(valid_df["operand_mode"].iloc[0]))
    counterfactual = collect_prompt_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=shifted_prompt_df["shifted_prompt"].tolist(),
        token_idxs=shifted_prompt_df["shifted_token_idx"].astype(int).tolist(),
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
    )
    deltas = counterfactual - source
    logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["prompt"].tolist(),
        token_idxs=valid_df["token_idx"].astype(int).tolist(),
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        delta_rows=deltas,
        capture_selected=False,
    )
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    result["search_objective"] = np.nan
    result["delta_norm"] = np.linalg.norm(deltas, axis=1)
    result["search_score_gain"] = np.nan
    result["selected_step_size"] = np.nan
    return result


def compute_counterfactual_patch_deltas(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    source = collect_prompt_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=valid_df["prompt"].tolist(),
        token_idxs=valid_df["token_idx"].astype(int).tolist(),
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
    )
    shifted_prompt_df = build_shifted_prompts(valid_df, tokenizer=tokenizer, operand_mode=str(valid_df["operand_mode"].iloc[0]))
    counterfactual = collect_prompt_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=shifted_prompt_df["shifted_prompt"].tolist(),
        token_idxs=shifted_prompt_df["shifted_token_idx"].astype(int).tolist(),
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
    )
    return np.asarray(counterfactual - source, dtype=np.float64)


def run_gradient_line_search(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
    objective: str,
    step_sizes: list[float],
    direction_mode: str,
) -> pd.DataFrame:
    raw_gradients = collect_gradient_directions(
        model=model,
        tokenizer=tokenizer,
        valid_df=valid_df,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        objective=objective,
    )
    if direction_mode == "sign":
        directions = np.sign(raw_gradients)
    elif direction_mode == "raw":
        directions = raw_gradients
    else:
        raise ValueError(f"Unsupported direction_mode: {direction_mode}")
    directions = normalize_rows(directions)
    target_token_ids = valid_df["target_token_id"].to_numpy(dtype=np.int64)
    baseline_token_ids = valid_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64)
    best_scores = np.full(len(valid_df), -np.inf, dtype=np.float64)
    best_logits = None
    best_steps = np.zeros(len(valid_df), dtype=np.float64)
    best_delta = np.zeros_like(directions, dtype=np.float64)
    baseline_logits = None

    for step_size in step_sizes:
        deltas = directions * float(step_size)
        logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=valid_df["prompt"].tolist(),
            token_idxs=valid_df["token_idx"].astype(int).tolist(),
            layer_idx=layer_idx,
            batch_size=batch_size,
            device=device,
            delta_rows=deltas,
            capture_selected=False,
        )
        if baseline_logits is None:
            baseline_logits, _ = forward_next_token_logits_with_delta(
                model=model,
                tokenizer=tokenizer,
                prompts=valid_df["prompt"].tolist(),
                token_idxs=valid_df["token_idx"].astype(int).tolist(),
                layer_idx=layer_idx,
                batch_size=batch_size,
                device=device,
                delta_rows=None,
                capture_selected=False,
            )
        scores = objective_scores_numpy(
            logits,
            target_token_ids=target_token_ids,
            baseline_token_ids=baseline_token_ids,
            objective=objective,
        )
        update = scores > best_scores
        if best_logits is None:
            best_logits = logits.copy()
        else:
            best_logits[update] = logits[update]
        best_scores[update] = scores[update]
        best_steps[update] = float(step_size)
        best_delta[update] = deltas[update]

    if best_logits is None:
        raise ValueError("Gradient line search did not evaluate any candidate steps.")
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=best_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    baseline_objective = objective_scores_numpy(
        baseline_logits,
        target_token_ids=target_token_ids,
        baseline_token_ids=baseline_token_ids,
        objective=objective,
    )
    result["search_objective"] = best_scores
    result["delta_norm"] = np.linalg.norm(best_delta, axis=1)
    result["search_score_gain"] = best_scores - baseline_objective
    result["selected_step_size"] = best_steps
    return result


def optimize_delta_search(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    batch_size: int,
    device: str,
    objective: str,
    steps: int,
    lr: float,
    max_norm: float,
    l2_penalty: float,
    patch_l2_penalty: float = 0.0,
    target_patch_deltas: np.ndarray | None = None,
    basis_matrix: np.ndarray | None = None,
    return_deltas: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    final_logits_chunks: list[np.ndarray] = []
    best_scores_chunks: list[np.ndarray] = []
    best_delta_chunks: list[np.ndarray] = []
    baseline_gain_chunks: list[np.ndarray] = []
    model_device = next(model.parameters()).device
    hidden_size = int(model.get_input_embeddings().weight.shape[1])

    for start in range(0, len(valid_df), batch_size):
        batch_df = valid_df.iloc[start : start + batch_size].reset_index(drop=True)
        prompts = batch_df["prompt"].tolist()
        token_idxs = batch_df["token_idx"].astype(int).tolist()
        target_token_ids = torch.tensor(batch_df["target_token_id"].to_list(), dtype=torch.long, device=model_device)
        baseline_token_ids = torch.tensor(batch_df["baseline_predicted_token_id"].to_list(), dtype=torch.long, device=model_device)
        if basis_matrix is not None:
            basis = torch.tensor(np.asarray(basis_matrix, dtype=np.float32), dtype=torch.float32, device=model_device)
            coeffs = torch.nn.Parameter(torch.zeros((len(batch_df), basis.shape[0]), dtype=torch.float32, device=model_device))
            parameters = [coeffs]

            def current_delta() -> torch.Tensor:
                return coeffs @ basis

        else:
            delta_param = torch.nn.Parameter(torch.zeros((len(batch_df), hidden_size), dtype=torch.float32, device=model_device))
            parameters = [delta_param]

            def current_delta() -> torch.Tensor:
                return delta_param

        optimizer = torch.optim.Adam(parameters, lr=float(lr))
        best_scores = torch.full((len(batch_df),), -float("inf"), dtype=torch.float32, device=model_device)
        best_delta = torch.zeros((len(batch_df), hidden_size), dtype=torch.float32, device=model_device)
        patch_target = None
        if target_patch_deltas is not None:
            patch_target = torch.tensor(
                np.asarray(target_patch_deltas[start : start + len(batch_df)], dtype=np.float32),
                dtype=torch.float32,
                device=model_device,
            )

        baseline_logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=layer_idx,
            batch_size=len(batch_df),
            device=device,
            delta_rows=None,
            capture_selected=False,
        )
        baseline_objective = objective_scores_numpy(
            baseline_logits,
            target_token_ids=batch_df["target_token_id"].to_numpy(dtype=np.int64),
            baseline_token_ids=batch_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64),
            objective=objective,
        )

        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            delta_rows = current_delta()
            logits = forward_next_token_logits_torch_with_delta(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                token_idxs=token_idxs,
                layer_idx=layer_idx,
                device=device,
                delta_rows=delta_rows,
            )
            objective_scores = objective_scores_torch(
                logits,
                target_token_ids=target_token_ids,
                baseline_token_ids=baseline_token_ids,
                objective=objective,
            )
            regularized_scores = objective_scores - float(l2_penalty) * delta_rows.pow(2).sum(dim=-1)
            if patch_target is not None and float(patch_l2_penalty) > 0:
                regularized_scores = regularized_scores - float(patch_l2_penalty) * (delta_rows - patch_target).pow(2).sum(dim=-1)
            loss = -regularized_scores.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if max_norm > 0:
                    if basis_matrix is not None:
                        deltas_now = coeffs @ basis
                        norms = deltas_now.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        factor = torch.clamp(float(max_norm) / norms, max=1.0)
                        coeffs.mul_(factor)
                        deltas_now = coeffs @ basis
                    else:
                        norms = delta_param.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        factor = torch.clamp(float(max_norm) / norms, max=1.0)
                        delta_param.mul_(factor)
                        deltas_now = delta_param
                update = regularized_scores > best_scores
                best_scores[update] = regularized_scores[update]
                best_delta[update] = deltas_now[update]

        best_logits, _ = forward_next_token_logits_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=layer_idx,
            batch_size=len(batch_df),
            device=device,
            delta_rows=best_delta,
            capture_selected=False,
        )
        final_logits_chunks.append(best_logits)
        best_scores_chunks.append(best_scores.detach().float().cpu().numpy())
        best_delta_chunks.append(best_delta.detach().float().cpu().numpy())
        baseline_gain_chunks.append(baseline_objective.astype(np.float64))

    final_logits = np.concatenate(final_logits_chunks, axis=0)
    best_scores = np.concatenate(best_scores_chunks, axis=0)
    best_deltas = np.concatenate(best_delta_chunks, axis=0)
    baseline_objective = np.concatenate(baseline_gain_chunks, axis=0)
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=final_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    result["search_objective"] = best_scores.astype(np.float64)
    result["delta_norm"] = np.linalg.norm(best_deltas, axis=1)
    result["search_score_gain"] = best_scores.astype(np.float64) - baseline_objective
    result["selected_step_size"] = np.nan
    if return_deltas:
        return result, best_deltas.astype(np.float64)
    return result


def optimize_shared_delta_search(
    *,
    model,
    tokenizer,
    valid_df: pd.DataFrame,
    layer_idx: int,
    tracked_values: np.ndarray,
    tracked_token_ids: np.ndarray,
    device: str,
    objective: str,
    steps: int,
    lr: float,
    max_norm: float,
    l2_penalty: float,
) -> pd.DataFrame:
    model_device = next(model.parameters()).device
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    prompts = valid_df["prompt"].tolist()
    token_idxs = valid_df["token_idx"].astype(int).tolist()
    target_token_ids_np = valid_df["target_token_id"].to_numpy(dtype=np.int64)
    baseline_token_ids_np = valid_df["baseline_predicted_token_id"].to_numpy(dtype=np.int64)
    target_token_ids = torch.tensor(valid_df["target_token_id"].to_list(), dtype=torch.long, device=model_device)
    baseline_token_ids = torch.tensor(valid_df["baseline_predicted_token_id"].to_list(), dtype=torch.long, device=model_device)

    shared_delta = torch.nn.Parameter(torch.zeros((hidden_size,), dtype=torch.float32, device=model_device))
    optimizer = torch.optim.Adam([shared_delta], lr=float(lr))
    best_mean_score = -float("inf")
    best_delta = torch.zeros((hidden_size,), dtype=torch.float32, device=model_device)

    baseline_logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=layer_idx,
        batch_size=len(valid_df),
        device=device,
        delta_rows=None,
        capture_selected=False,
    )
    baseline_objective = objective_scores_numpy(
        baseline_logits,
        target_token_ids=target_token_ids_np,
        baseline_token_ids=baseline_token_ids_np,
        objective=objective,
    )

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        batch_delta = shared_delta.unsqueeze(0).expand(len(valid_df), -1)
        logits = forward_next_token_logits_torch_with_delta(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            token_idxs=token_idxs,
            layer_idx=layer_idx,
            device=device,
            delta_rows=batch_delta,
        )
        objective_scores = objective_scores_torch(
            logits,
            target_token_ids=target_token_ids,
            baseline_token_ids=baseline_token_ids,
            objective=objective,
        )
        regularized_scores = objective_scores - float(l2_penalty) * shared_delta.pow(2).sum()
        loss = -regularized_scores.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if max_norm > 0:
                norm = shared_delta.norm().clamp_min(1e-8)
                factor = min(float(max_norm) / float(norm), 1.0)
                shared_delta.mul_(factor)
            mean_score = float(regularized_scores.mean())
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_delta.copy_(shared_delta)

    final_delta = best_delta.unsqueeze(0).expand(len(valid_df), -1)
    final_logits, _ = forward_next_token_logits_with_delta(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=layer_idx,
        batch_size=len(valid_df),
        device=device,
        delta_rows=final_delta,
        capture_selected=False,
    )
    result = evaluate_logits_against_shifted_targets(
        tokenizer=tokenizer,
        valid_df=valid_df,
        logits=final_logits,
        tracked_values=tracked_values,
        tracked_token_ids=tracked_token_ids,
    )
    final_scores = objective_scores_numpy(
        final_logits,
        target_token_ids=target_token_ids_np,
        baseline_token_ids=baseline_token_ids_np,
        objective=objective,
    )
    result["search_objective"] = final_scores
    result["delta_norm"] = float(best_delta.norm().detach().cpu().item())
    result["search_score_gain"] = final_scores - baseline_objective
    result["selected_step_size"] = np.nan
    return result


def summarize_condition(
    *,
    spec: GradientSearchSpec,
    seed: int,
    layer_idx: int | None,
    alpha: float,
    valid_df: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, Any]:
    numeric_delta = pd.to_numeric(results["predicted_numeric_delta"], errors="coerce")
    return {
        "test_id": spec.test_id,
        "hypothesis_id": spec.hypothesis_id,
        "hypothesis": spec.hypothesis,
        "reason": spec.reason,
        "kind": spec.kind,
        "layer": spec.layer,
        "layer_idx": layer_idx,
        "seed": int(seed),
        "task": str(valid_df["task"].iloc[0]),
        "operand_mode": str(valid_df["operand_mode"].iloc[0]),
        "alpha": float(alpha),
        "num_examples": int(len(results)),
        "num_sampled_examples": int(valid_df["num_sampled_examples"].iloc[0]),
        "num_eligible_examples": int(valid_df["num_eligible_examples"].iloc[0]),
        "baseline_full_accuracy": float(valid_df["baseline_full_accuracy"].iloc[0]),
        "baseline_filter_mode": str(valid_df["baseline_filter_mode"].iloc[0]),
        "argmax_accuracy": float(results["correct"].mean()),
        "mean_expected_shift": float(results["expected_shift"].mean()),
        "mean_abs_expected_shift": float(results["expected_shift"].abs().mean()),
        "mean_target_error": float(results["target_error"].mean()),
        "mean_abs_target_error": float(results["target_error"].abs().mean()),
        "mean_target_prob_gain": float(results["target_prob_gain"].mean()),
        "baseline_answer_retention": float(results["baseline_answer_retained"].mean()),
        "numeric_prediction_rate": float(pd.to_numeric(results["predicted_value"], errors="coerce").notna().mean()),
        "mean_predicted_numeric_delta": float(numeric_delta.mean()) if numeric_delta.notna().any() else float("nan"),
        "mean_delta_norm": float(pd.to_numeric(results["delta_norm"], errors="coerce").mean()),
        "mean_search_objective": float(pd.to_numeric(results["search_objective"], errors="coerce").mean()),
        "mean_search_score_gain": float(pd.to_numeric(results["search_score_gain"], errors="coerce").mean()),
        "mean_selected_step_size": float(pd.to_numeric(results["selected_step_size"], errors="coerce").mean()),
    }


def aggregate_saved_conditions(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(output_dir)
    condition_dir = base / "conditions"
    sample_dir = base / "samples"

    condition_frames = [pd.read_csv(path) for path in sorted(condition_dir.glob("*.csv"))]
    summary = pd.concat(condition_frames, ignore_index=True) if condition_frames else pd.DataFrame()
    aggregated = (
        summary.groupby(
            ["hypothesis_id", "test_id", "hypothesis", "reason", "kind", "layer", "layer_idx", "task", "operand_mode", "alpha"],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            num_examples_mean=("num_examples", "mean"),
            baseline_full_accuracy_mean=("baseline_full_accuracy", "mean"),
            argmax_accuracy_mean=("argmax_accuracy", "mean"),
            argmax_accuracy_std=("argmax_accuracy", "std"),
            mean_expected_shift_mean=("mean_expected_shift", "mean"),
            mean_target_prob_gain_mean=("mean_target_prob_gain", "mean"),
            mean_delta_norm_mean=("mean_delta_norm", "mean"),
            mean_search_objective_mean=("mean_search_objective", "mean"),
            mean_search_score_gain_mean=("mean_search_score_gain", "mean"),
            mean_selected_step_size_mean=("mean_selected_step_size", "mean"),
        )
        .reset_index()
        if not summary.empty
        else pd.DataFrame()
    )
    sample_frames = [pd.read_csv(path) for path in sorted(sample_dir.glob("*.csv"))]
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return summary, aggregated, sample_df


def run_gradient_search_suite(
    *,
    model,
    tokenizer,
    templates: dict[str, str],
    test_specs: list[GradientSearchSpec],
    seeds: list[int],
    tasks: list[str],
    operand_modes: list[str],
    alpha_values: list[float],
    evaluation_batch_size: int,
    device: str,
    max_examples: int | None,
    filter_baseline_correct_only: bool,
    line_search_step_sizes: list[float],
    optimization_steps: int,
    optimization_lr: float,
    optimization_max_norm: float,
    optimization_l2_penalty: float,
    details_sample_per_condition: int,
    output_dir: str | Path | None = None,
    force_conditions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    manifest = pd.DataFrame([spec.to_jsonable() for spec in test_specs])
    condition_dir = None if output_dir is None else ensure_dir(Path(output_dir) / "conditions")
    sample_dir = None if output_dir is None else ensure_dir(Path(output_dir) / "samples")

    slice_cache: dict[tuple[str, str, int, int], PreparedSteeringSlice] = {}
    patch_delta_cache: dict[tuple[int, str, str, int, float], np.ndarray] = {}
    for seed in seeds:
        for task in tasks:
            template = templates[task]
            for operand_mode in operand_modes:
                for spec in test_specs:
                    layer_idx = 0 if spec.layer == "none" else resolve_named_layer(model, spec.layer)
                    cache_key = (task, operand_mode, seed, layer_idx)
                    if cache_key not in slice_cache:
                        slice_cache[cache_key] = prepare_steering_slice(
                            model=model,
                            tokenizer=tokenizer,
                            task=task,
                            template=template,
                            operand_mode=operand_mode,
                            layer_idx=layer_idx,
                            batch_size=evaluation_batch_size,
                            device=device,
                            seed=seed,
                            max_examples=max_examples,
                            filter_baseline_correct_only=filter_baseline_correct_only,
                        )
                    slice_data = slice_cache[cache_key]
                    for alpha in alpha_values:
                        stem = condition_stem(spec=spec, seed=seed, task=task, operand_mode=operand_mode, alpha=float(alpha))
                        summary_part_path = None if condition_dir is None else condition_dir / f"{stem}.csv"
                        sample_part_path = None if sample_dir is None else sample_dir / f"{stem}.csv"
                        if summary_part_path is not None and summary_part_path.exists() and not force_conditions:
                            cached_summary = pd.read_csv(summary_part_path)
                            if not cached_summary.empty:
                                summary_rows.extend(cached_summary.to_dict(orient="records"))
                            if sample_part_path is not None and sample_part_path.exists():
                                cached_sample = pd.read_csv(sample_part_path)
                                if not cached_sample.empty:
                                    sample_frames.append(cached_sample)
                            continue

                        valid_df = make_valid_shifted_rows(slice_data=slice_data, tokenizer=tokenizer, alpha=alpha)
                        if valid_df.empty:
                            continue
                        valid_df = add_baseline_metadata(valid_df, slice_data, filter_baseline_correct_only)
                        condition_max_norm = float(spec.params.get("max_norm_override", optimization_max_norm))
                        patch_deltas = None
                        needs_patch = spec.kind in {"optimize_subspace_delta"} or float(spec.params.get("patch_l2_penalty", 0.0)) > 0
                        if needs_patch:
                            patch_key = (int(seed), str(task), str(operand_mode), int(layer_idx), float(alpha))
                            if patch_key not in patch_delta_cache:
                                patch_delta_cache[patch_key] = compute_counterfactual_patch_deltas(
                                    model=model,
                                    tokenizer=tokenizer,
                                    valid_df=valid_df,
                                    layer_idx=layer_idx,
                                    batch_size=evaluation_batch_size,
                                    device=device,
                                )
                            patch_deltas = patch_delta_cache[patch_key]

                        if spec.kind == "prompt_counterfactual":
                            results = evaluate_prompt_counterfactual(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                batch_size=evaluation_batch_size,
                                device=device,
                            )
                        elif spec.kind == "counterfactual_patch":
                            results = evaluate_counterfactual_patch(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                layer_idx=layer_idx,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                batch_size=evaluation_batch_size,
                                device=device,
                            )
                        elif spec.kind == "gradient_linesearch":
                            results = run_gradient_line_search(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                layer_idx=layer_idx,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                batch_size=evaluation_batch_size,
                                device=device,
                                objective=str(spec.params["objective"]),
                                step_sizes=[float(step) for step in line_search_step_sizes],
                                direction_mode=str(spec.params.get("direction_mode", "raw")),
                            )
                        elif spec.kind == "optimize_delta":
                            results = optimize_delta_search(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                layer_idx=layer_idx,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                batch_size=min(evaluation_batch_size, 64),
                                device=device,
                                objective=str(spec.params["objective"]),
                                steps=int(optimization_steps),
                                lr=float(optimization_lr),
                                max_norm=condition_max_norm,
                                l2_penalty=float(optimization_l2_penalty),
                                patch_l2_penalty=float(spec.params.get("patch_l2_penalty", 0.0)),
                                target_patch_deltas=patch_deltas,
                            )
                        elif spec.kind == "optimize_shared_delta":
                            results = optimize_shared_delta_search(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                layer_idx=layer_idx,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                device=device,
                                objective=str(spec.params["objective"]),
                                steps=int(optimization_steps),
                                lr=float(optimization_lr),
                                max_norm=condition_max_norm,
                                l2_penalty=float(optimization_l2_penalty),
                            )
                        elif spec.kind == "optimize_subspace_delta":
                            if patch_deltas is None:
                                raise ValueError("optimize_subspace_delta requires patch deltas.")
                            basis_matrix = top_patch_subspace_basis(patch_deltas, rank=int(spec.params["rank"]))
                            results = optimize_delta_search(
                                model=model,
                                tokenizer=tokenizer,
                                valid_df=valid_df,
                                layer_idx=layer_idx,
                                tracked_values=slice_data.tracked_values,
                                tracked_token_ids=slice_data.tracked_token_ids,
                                batch_size=min(evaluation_batch_size, 64),
                                device=device,
                                objective=str(spec.params["objective"]),
                                steps=int(optimization_steps),
                                lr=float(optimization_lr),
                                max_norm=condition_max_norm,
                                l2_penalty=float(optimization_l2_penalty),
                                basis_matrix=basis_matrix,
                            )
                        else:
                            raise ValueError(f"Unsupported gradient search kind: {spec.kind}")

                        summary_row = summarize_condition(
                            spec=spec,
                            seed=seed,
                            layer_idx=None if spec.layer == "none" else layer_idx,
                            alpha=alpha,
                            valid_df=valid_df,
                            results=results,
                        )
                        summary_rows.append(summary_row)
                        if summary_part_path is not None:
                            atomic_write_csv(summary_part_path, pd.DataFrame([summary_row]))
                        if details_sample_per_condition > 0:
                            sample_n = min(int(details_sample_per_condition), len(results))
                            sampled = results.sample(n=sample_n, random_state=seed).assign(
                                test_id=spec.test_id,
                                seed=seed,
                                layer=str(spec.layer),
                            )
                            sample_frames.append(sampled)
                            if sample_part_path is not None:
                                atomic_write_csv(sample_part_path, sampled)

    summary = pd.DataFrame(summary_rows)
    aggregated = (
        summary.groupby(["hypothesis_id", "test_id", "kind", "layer", "layer_idx", "task", "operand_mode", "alpha"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            num_examples_mean=("num_examples", "mean"),
            baseline_full_accuracy_mean=("baseline_full_accuracy", "mean"),
            argmax_accuracy_mean=("argmax_accuracy", "mean"),
            argmax_accuracy_std=("argmax_accuracy", "std"),
            mean_expected_shift_mean=("mean_expected_shift", "mean"),
            mean_target_prob_gain_mean=("mean_target_prob_gain", "mean"),
            mean_delta_norm_mean=("mean_delta_norm", "mean"),
            mean_search_objective_mean=("mean_search_objective", "mean"),
            mean_search_score_gain_mean=("mean_search_score_gain", "mean"),
            mean_selected_step_size_mean=("mean_selected_step_size", "mean"),
        )
        .reset_index()
    )
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return manifest, summary, aggregated if not aggregated.empty else summary, sample_df
