"""Matched numerical-subspace deletion operations and next-token evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.special import logsumexp


TASKS = (
    "addition_formula", "addition_verbalized", "subtraction_formula",
    "subtraction_verbalized", "identity_patchscopes", "parity", "magnitude_comparison",
)


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    operation: str
    basis: str = "property"
    seed_index: int = -1


def conditions() -> tuple[Condition, ...]:
    fixed = (
        Condition("original", "original", "original"),
        Condition("delete_property", "delete_property", "delete"),
        Condition("keep_property", "keep_property", "keep"),
        Condition("delete_pca41", "delete_pca41", "delete", "pca41"),
        Condition("delete_pca_energy", "delete_pca_energy", "delete", "pca_energy"),
    )
    repeated = tuple(
        Condition(f"{family}_{i}", family, operation, f"{basis}_{i}" if basis != "property" else basis, i)
        for family, operation, basis in (
            ("delete_random", "delete", "random"),
            ("delete_shuffled", "delete", "shuffled"),
            ("matched_noise", "noise", "property"),
        ) for i in range(3)
    )
    return fixed + repeated


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("::".join(map(str, (20260915, *parts))).encode()).digest()[:4], "little")


def digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class RunConfig:
    bundle_hash: str
    code_hash: str
    splits: tuple[str, ...]
    tasks: tuple[str, ...] = TASKS
    batch_size: int = 32
    max_examples: int | None = None
    workspace_config: str = ":4096:8"
    forward: str = "llama_core_last_hidden_v1"
    dtype: str = "float16"
    attention: str = "eager"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def run_id(self) -> str:
        return f"rank41_deletion__{digest(self.to_dict())[:12]}__r0"


def intervention_delta(live: np.ndarray, center: np.ndarray, bases: dict[str, np.ndarray],
                       condition: Condition, *, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    live = np.asarray(live, dtype=np.float64)
    centered = live - np.asarray(center, dtype=np.float64)
    basis = np.asarray(bases[condition.basis], dtype=np.float64)
    if np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))) > 1e-6:
        raise ValueError("Intervention basis is not orthonormal")
    projection = (centered @ basis) @ basis.T
    if condition.operation == "original":
        delta = np.zeros_like(live)
    elif condition.operation == "delete":
        delta = -projection
    elif condition.operation == "keep":
        delta = projection - centered
    elif condition.operation == "noise":
        direction = np.random.default_rng(seed).standard_normal(live.shape)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        delta = direction * np.linalg.norm(projection, axis=1, keepdims=True)
    else:
        raise ValueError(condition.operation)
    energy = np.sum(centered ** 2, axis=1)
    diagnostics = {
        "delta_l2": np.linalg.norm(delta, axis=1),
        "centered_energy": energy,
        "basis_projection_energy": np.sum(projection ** 2, axis=1),
        "delta_energy_fraction": np.sum(delta ** 2, axis=1) / np.maximum(energy, 1e-30),
    }
    return delta.astype(np.float32), diagnostics


def next_token_metrics(baseline: np.ndarray, edited: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    # Float64 reductions keep KL stable for nearly unchanged conditions.
    logp = np.asarray(baseline, dtype=np.float64)
    logq = np.asarray(edited, dtype=np.float64)
    logp -= logsumexp(logp, axis=1, keepdims=True)
    logq -= logsumexp(logq, axis=1, keepdims=True)
    ix = np.arange(len(targets))
    pred0, pred = logp.argmax(axis=1), logq.argmax(axis=1)
    return {
        "baseline_token_id": pred0, "intervention_token_id": pred,
        "baseline_correct": pred0 == targets, "intervention_correct": pred == targets,
        "top1_agreement": pred0 == pred,
        "baseline_target_logprob": logp[ix, targets],
        "intervention_target_logprob": logq[ix, targets],
        "target_logprob_delta": logq[ix, targets] - logp[ix, targets],
        "kl_original_to_intervention": np.maximum(0, np.sum(np.exp(logp) * (logp - logq), axis=1)),
    }


def forward_last_logits(*, model, tokenizer, prompts, token_idxs, layer_idx, batch_size,
                        device="cuda", delta_rows=None):
    """Same full decoder forward as mainline; apply lm_head only at final tokens.

    This avoids materializing a batch × sequence × vocabulary tensor. The smoke
    audit compares against the unchanged mainline forward helper at identical
    batch shapes. No decoder-prefix caching or attention-path changes are used.
    """
    import torch
    from src.activation_steering import get_model_layers, hidden_states_hook

    logits, selected = [], []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            if tokenizer.padding_side != "right":
                raise ValueError("Mainline source-token indices require right padding")
            kwargs = {k: v.to(device) for k, v in encoded.items()}
            indices = torch.as_tensor(token_idxs[start:start + batch_size], device=device)
            rows = torch.arange(len(batch), device=device)
            if torch.any(indices < 0) or torch.any(indices >= kwargs["attention_mask"].sum(1)):
                raise ValueError("Invalid source-token index")
            captured = {}

            class Edit:
                @hidden_states_hook
                def __call__(self, module, args, kw, output):
                    captured["selected"] = output[rows, indices].detach().float().cpu().numpy()
                    if delta_rows is None:
                        return output
                    edited = output.clone()
                    delta = torch.as_tensor(delta_rows[start:start + batch_size], device=device).to(output)
                    edited[rows, indices] = edited[rows, indices] + delta
                    return edited

            hook = get_model_layers(model)[layer_idx].register_forward_hook(Edit(), with_kwargs=True)
            try:
                output = model.model(**kwargs, use_cache=False)
                last = kwargs["attention_mask"].sum(1) - 1
                batch_logits = model.lm_head(output.last_hidden_state[rows, last])
            finally:
                hook.remove()
            logits.append(batch_logits.float().cpu().numpy())
            selected.append(captured["selected"])
    return np.concatenate(logits), np.concatenate(selected)
