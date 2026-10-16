from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.arithmetic_eval import (
    ArithmeticExample,
    filter_single_token_targets,
    generate_addition_examples,
    generate_multiplication_examples,
    generate_subtraction_examples,
)
from src.metrics import exact_accuracy, mae, rmse, round_predictions
from src.probes import fit_linear_regression
from src.token_utils import encode_integer


ArithmeticTask = Literal["addition", "subtraction", "multiplication"]
OperandMode = Literal["x1", "x2"]


@dataclass
class ActivationSteeringProbe:
    layer_idx: int
    prompt_template: str
    token_mode: str
    basis_dim: int
    support_min: int
    support_max: int
    coefficients: np.ndarray
    bias: np.ndarray
    decoder_matrix: np.ndarray
    fit_decoded_accuracy: float
    fit_basis_mse: float
    fit_mae: float
    fit_rmse: float

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
        payload["support_values"] = list(range(int(self.support_min), int(self.support_max) + 1))
        return payload


def normalize_vector(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("Cannot normalize a zero-norm vector")
    return arr / norm


def probe_direction_for_norm_mode(
    *,
    coefficients: list[float] | np.ndarray,
    norm_mode: str,
    option_name: str = "probe_norm_mode",
) -> dict[str, Any]:
    direction_arr = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    direction_unit = normalize_vector(direction_arr)
    coef_norm = float(np.linalg.norm(direction_arr))
    steering_direction_per_alpha = direction_unit
    value_gain_per_alpha = float(coef_norm)
    value_unit_scale = 1.0
    if norm_mode == "value_unit":
        if coef_norm <= 0:
            raise ValueError(f"Cannot use {option_name}='value_unit' with zero probe coefficient norm.")
        value_unit_scale = float(1.0 / coef_norm)
        steering_direction_per_alpha = direction_unit * value_unit_scale
        value_gain_per_alpha = float(np.dot(direction_arr, steering_direction_per_alpha))
    elif norm_mode != "l2":
        raise ValueError(f"Unsupported {option_name}: {norm_mode!r}")
    return {
        "norm_mode": norm_mode,
        "direction_dim": int(direction_unit.shape[0]),
        "direction_norm": float(coef_norm),
        "direction_unit": direction_unit,
        "value_unit_scale": float(value_unit_scale),
        "value_gain_per_alpha": float(value_gain_per_alpha),
        "steering_direction_per_alpha": steering_direction_per_alpha,
        "steering_direction_norm_per_alpha": float(np.linalg.norm(steering_direction_per_alpha)),
    }


def value_unit_calibration_from_activations(
    *,
    activations: np.ndarray,
    values: np.ndarray,
    direction_unit: np.ndarray | list[float],
) -> dict[str, float | None]:
    direction = normalize_vector(direction_unit)
    activation_matrix = np.asarray(activations, dtype=np.float64)
    if activation_matrix.ndim != 2:
        raise ValueError("Expected activations as a 2D array.")
    if activation_matrix.shape[1] != direction.shape[0]:
        raise ValueError(
            "Direction dimensionality does not match activation dimensionality for value-unit calibration. "
            f"direction_dim={direction.shape[0]}, activation_dim={activation_matrix.shape[1]}"
        )
    value_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if value_arr.shape[0] != activation_matrix.shape[0]:
        raise ValueError(
            "Values length does not match number of activation rows for value-unit calibration. "
            f"n_values={value_arr.shape[0]}, n_rows={activation_matrix.shape[0]}"
        )

    projected = activation_matrix @ direction
    projected_centered = projected - float(np.mean(projected))
    values_centered = value_arr - float(np.mean(value_arr))
    denom = float(np.dot(projected_centered, projected_centered))
    if denom <= 0:
        raise ValueError("Cannot calibrate value-unit scale: projected activations have zero variance.")
    gain = float(np.dot(projected_centered, values_centered) / denom)
    if not math.isfinite(gain) or abs(gain) <= 1e-8:
        raise ValueError("Cannot calibrate value-unit scale: directional value gain is too small or non-finite.")
    pred = float(np.mean(value_arr)) + gain * projected_centered
    residual = value_arr - pred
    sst = float(np.dot(values_centered, values_centered))
    r2 = None if sst <= 0 else float(1.0 - (float(np.dot(residual, residual)) / sst))
    return {
        "value_unit_scale": float(1.0 / gain),
        "value_gain_per_l2_alpha": float(gain),
        "value_calibration_r2": r2,
    }


def hidden_states_hook(func):
    def wrapper(self, module, args, kwargs, output):
        if isinstance(output, torch.Tensor):
            hidden_states = output
            return_tensor = True
        else:
            hidden_states = output[0]
            return_tensor = False
        new_hidden_states = func(self, module, args, kwargs, hidden_states)
        if return_tensor:
            return new_hidden_states
        return (new_hidden_states,) + output[1:]

    return wrapper


def get_model_layers(model) -> list[Any]:
    candidates: list[tuple[Any | None, list[Any] | None]] = []
    if hasattr(model, "model"):
        core = model.model
        candidates.append((getattr(core, "embed_tokens", None), list(getattr(core, "layers", [])) or None))
        if hasattr(core, "decoder"):
            decoder = core.decoder
            candidates.append((getattr(decoder, "embed_tokens", None), list(getattr(decoder, "layers", [])) or None))
    if hasattr(model, "transformer"):
        core = model.transformer
        candidates.append((getattr(core, "wte", None), list(getattr(core, "h", [])) or None))
        candidates.append((getattr(core, "word_embeddings", None), list(getattr(core, "h", [])) or None))
    candidates.append((getattr(model, "embed_tokens", None), list(getattr(model, "layers", [])) or None))
    for emb_layer, non_emb_layers in candidates:
        if emb_layer is not None and non_emb_layers:
            return [emb_layer] + list(non_emb_layers)
    raise ValueError(f"Unsupported model structure for activation steering: {type(model).__name__}")


class LayerForwardHook:
    def __init__(self, model):
        self.model = model
        self.handles: list[Any] = []

    @property
    def layers(self) -> list[Any]:
        return get_model_layers(self.model)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def attach(self, hook_func, layer_idx: int):
        handle = self.layers[layer_idx].register_forward_hook(hook_func, with_kwargs=True)
        self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def resolve_layer_idx(model, requested_idx: int | None) -> int:
    hook = LayerForwardHook(model)
    layer_idx = 0 if requested_idx is None else int(requested_idx)
    if layer_idx < 0 or layer_idx >= hook.n_layers:
        raise ValueError(f"intervention layer_idx={layer_idx} is out of range for n_layers={hook.n_layers}")
    return layer_idx


def render_probe_prompt(template: str, value: int) -> tuple[str, tuple[int, int]]:
    mention = str(int(value))
    prompt = template.format(value=mention)
    start = prompt.rfind(mention)
    if start < 0:
        raise ValueError(f"Could not locate probe value {mention!r} in prompt {prompt!r}")
    return prompt, (start, start + len(mention))


def render_prompt_with_operand_spans(template: str, x1: int, x2: int) -> tuple[str, dict[str, tuple[int, int]]]:
    spans: dict[str, tuple[int, int]] = {}
    parts: list[str] = []
    i = 0
    cursor = 0
    while i < len(template):
        if template.startswith("x1", i):
            text = str(x1)
            start = cursor
            parts.append(text)
            cursor += len(text)
            spans["x1"] = (start, cursor)
            i += 2
        elif template.startswith("x2", i):
            text = str(x2)
            start = cursor
            parts.append(text)
            cursor += len(text)
            spans["x2"] = (start, cursor)
            i += 2
        else:
            parts.append(template[i])
            cursor += 1
            i += 1
    prompt = "".join(parts)
    if "x1" not in spans or "x2" not in spans:
        raise ValueError(f"Template is missing x1/x2 placeholders: {template!r}")
    return prompt, spans


def resolve_token_idx(
    *,
    tokenizer,
    prompt: str,
    mention_span: tuple[int, int],
    mode: str,
) -> int:
    positions = mention_token_positions(tokenizer=tokenizer, prompt=prompt, mention_span=mention_span)
    if positions:
        return int(positions[-1] if mode == "mention_last" else positions[0])
    prefix = prompt[: mention_span[0]]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    fallback = len(prefix_ids)
    if fallback >= len(prompt_ids):
        fallback = len(prompt_ids) - 1
    if fallback < 0:
        raise ValueError("Could not resolve a valid token index for intervention.")
    return int(fallback)


def mention_token_positions(*, tokenizer, prompt: str, mention_span: tuple[int, int]) -> list[int]:
    mention_start, mention_end = mention_span
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    except TypeError:
        return []
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return []
    row_offsets = offsets if offsets and isinstance(offsets[0], (list, tuple)) else []
    positions: list[int] = []
    for idx, pair in enumerate(row_offsets):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_end <= token_start:
            continue
        if token_start >= mention_start and token_end <= mention_end:
            positions.append(idx)
    return positions


def build_numeric_token_info(tokenizer, value_min: int = 0, value_max: int = 999) -> dict[str, Any]:
    values = np.arange(value_min, value_max + 1, dtype=np.int64)
    token_ids: list[int] = []
    for value in values.tolist():
        ids = encode_integer(tokenizer, int(value))
        if len(ids) != 1:
            raise ValueError(f"Value {value} is not single-token under the current tokenizer.")
        token_ids.append(int(ids[0]))
    return {
        "values": values,
        "token_ids": np.asarray(token_ids, dtype=np.int64),
    }


def _model_device_dtype(model) -> tuple[Any, Any]:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device, parameter.dtype
    return torch.device("cpu"), torch.float32


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _parse_optional_int(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if stripped == "":
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _apply_direction_edit(
    *,
    output: torch.Tensor,
    token_indices: torch.Tensor,
    direction_rows: torch.Tensor,
    activation_postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    idx = torch.arange(output.shape[0], device=output.device)
    token_idx = token_indices.to(output.device).clamp(min=0, max=output.shape[1] - 1)
    if direction_rows.ndim == 1:
        direction_rows = direction_rows.unsqueeze(0).expand(output.shape[0], -1)
    edited = output.clone()
    edited_rows = edited[idx, token_idx] + direction_rows.to(output)
    if activation_postprocess_fn is not None:
        edited_rows = activation_postprocess_fn(edited_rows)
        if edited_rows.shape != edited[idx, token_idx].shape:
            raise ValueError(
                "Activation postprocess changed the selected activation shape. "
                f"expected={tuple(edited[idx, token_idx].shape)}, got={tuple(edited_rows.shape)}"
            )
        edited_rows = edited_rows.to(device=output.device, dtype=output.dtype)
    edited[idx, token_idx] = edited_rows
    return edited


def _batch_forward_with_optional_direction(
    *,
    model,
    tokenizer,
    prompts: list[str],
    token_idxs: list[int],
    layer_idx: int,
    batch_size: int,
    device: str,
    scaled_directions: np.ndarray | None = None,
    capture_pre_activations: bool = False,
    capture_full_sequences: bool = False,
    collect_logits: bool = True,
    activation_postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray] | None]:
    model_device, model_dtype = _model_device_dtype(model)
    all_logits: list[np.ndarray] = []
    captured_chunks: list[np.ndarray] = []
    captured_sequence_chunks: list[list[np.ndarray]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="next-token", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        batch_token_idxs = token_idxs[start : start + batch_size]
        batch_directions = None if scaled_directions is None else scaled_directions[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        model_kwargs = {k: v.to(device) for k, v in encoded.items()}

        class _BatchCapture:
            def __init__(self, *, token_indices: list[int], capture_sequences: bool) -> None:
                self.token_indices = torch.as_tensor(token_indices, dtype=torch.long)
                self.captured: np.ndarray | None = None
                self.capture_sequences = bool(capture_sequences)
                self.captured_sequences: list[np.ndarray] | None = None

            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                if output.ndim < 3:
                    return output
                batch_output = output.detach().float().cpu()
                if self.capture_sequences:
                    self.captured_sequences = [batch_output[row_idx].numpy() for row_idx in range(batch_output.shape[0])]
                idx = torch.arange(output.shape[0], device=output.device)
                token_idx = self.token_indices.to(output.device).clamp(min=0, max=output.shape[1] - 1)
                self.captured = output[idx, token_idx].detach().float().cpu().numpy()
                return output

        class _BatchDirectedEdit:
            def __init__(
                self,
                *,
                token_indices: list[int],
                directions: np.ndarray | None,
                postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
            ) -> None:
                self.token_indices = torch.as_tensor(token_indices, dtype=torch.long)
                self.directions = None if directions is None else torch.as_tensor(directions, dtype=model_dtype, device=model_device)
                self.postprocess_fn = postprocess_fn
                self.call_again = True

            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                if not self.call_again:
                    return output
                self.call_again = False
                if output.ndim < 3 or self.directions is None:
                    return output
                return _apply_direction_edit(
                    output=output,
                    token_indices=self.token_indices,
                    direction_rows=self.directions,
                    activation_postprocess_fn=self.postprocess_fn,
                )

        hook = LayerForwardHook(model)
        capture_hook = None
        if capture_pre_activations or capture_full_sequences:
            capture_hook = _BatchCapture(token_indices=batch_token_idxs, capture_sequences=capture_full_sequences)
            hook.attach(capture_hook, layer_idx=layer_idx)
        if batch_directions is not None:
            hook.attach(
                _BatchDirectedEdit(
                    token_indices=batch_token_idxs,
                    directions=batch_directions,
                    postprocess_fn=activation_postprocess_fn,
                ),
                layer_idx=layer_idx,
            )

        try:
            with torch.no_grad():
                outputs = model(**model_kwargs, use_cache=False)
        finally:
            hook.remove()

        attention_mask = model_kwargs["attention_mask"]
        if collect_logits:
            logits = outputs.logits
            last_positions = attention_mask.sum(dim=1) - 1
            next_token_logits = logits[torch.arange(logits.shape[0], device=device), last_positions]
            all_logits.append(next_token_logits.detach().float().cpu().numpy())
        if capture_hook is not None and capture_hook.captured is not None:
            captured_chunks.append(capture_hook.captured)
        if capture_hook is not None and capture_hook.captured_sequences is not None:
            trimmed_sequences: list[np.ndarray] = []
            lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
            for row_idx, seq_len in enumerate(lengths):
                trimmed_sequences.append(
                    np.asarray(capture_hook.captured_sequences[row_idx][: int(seq_len)], dtype=np.float32)
                )
            captured_sequence_chunks.append(trimmed_sequences)

    captured = None if not captured_chunks else np.concatenate(captured_chunks, axis=0).astype(np.float32)
    captured_sequences = None if not captured_sequence_chunks else [item for chunk in captured_sequence_chunks for item in chunk]
    return all_logits, captured, captured_sequences


def collect_prompt_activations(
    *,
    model,
    tokenizer,
    prompts: list[str],
    token_idxs: list[int],
    layer_idx: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    _, captured, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        scaled_directions=None,
        capture_pre_activations=True,
        capture_full_sequences=False,
        collect_logits=False,
        activation_postprocess_fn=None,
    )
    if captured is None:
        raise ValueError("Failed to capture activations.")
    return captured


def collect_prompt_hidden_sequences(
    *,
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    batch_size: int,
    device: str,
) -> list[np.ndarray]:
    _, _, captured_sequences = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=[0] * len(prompts),
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
        scaled_directions=None,
        capture_pre_activations=False,
        capture_full_sequences=True,
        collect_logits=False,
        activation_postprocess_fn=None,
    )
    if captured_sequences is None:
        raise ValueError("Failed to capture full prompt hidden sequences.")
    return [np.asarray(sequence, dtype=np.float32) for sequence in captured_sequences]


def collect_prompt_hidden_sequences_torch(
    *,
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    batch_size: int,
    device: str,
) -> list[torch.Tensor]:
    sequences: list[torch.Tensor] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="hidden-cache", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        model_kwargs = {key: value.to(device) for key, value in encoded.items()}
        captured: dict[str, torch.Tensor] = {}

        class _SequenceCapture:
            @hidden_states_hook
            def __call__(self, module, args, kwargs, output):
                del module, args, kwargs
                if output.ndim < 3:
                    return output
                captured["hidden_states"] = output.detach().cpu()
                return output

        hook = LayerForwardHook(model)
        hook.attach(_SequenceCapture(), layer_idx=layer_idx)
        try:
            with torch.no_grad():
                model(**model_kwargs)
        finally:
            hook.remove()
        if "hidden_states" not in captured:
            raise ValueError("Failed to capture hidden sequences.")
        lengths = model_kwargs["attention_mask"].sum(dim=1).detach().cpu().tolist()
        batch_hidden = captured["hidden_states"]
        for row_idx, seq_len in enumerate(lengths):
            sequences.append(batch_hidden[row_idx, : int(seq_len)].contiguous())
    return sequences


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return array / safe_norms


def circular_sinusoidal_encode(
    values,
    *,
    embedding_dim: int,
    min_value: int,
    max_value: int,
) -> np.ndarray:
    if embedding_dim < 1:
        raise ValueError("circular probe basis_dim must be at least 1")
    if max_value <= min_value:
        raise ValueError("circular probe requires max_value > min_value")

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.zeros((x.shape[0], int(embedding_dim)), dtype=np.float64)
    even_indices = np.arange(0, int(embedding_dim), 2, dtype=np.float64)
    log_term = np.log(float(max_value - min_value)) / float(max(embedding_dim, 2))
    div_term = np.exp(even_indices * -log_term)
    shifted = x - float(min_value)
    angles = shifted[:, None] * div_term[None, :]
    y[:, 0::2] = np.sin(angles)
    if embedding_dim > 1:
        y[:, 1::2] = np.cos(angles[:, : y[:, 1::2].shape[1]])
    return y


def decode_circular_probe_predictions(
    predictions,
    *,
    support_values,
    basis_dim: int,
    support_min: int,
    support_max: int,
) -> np.ndarray:
    support_basis = circular_sinusoidal_encode(
        support_values,
        embedding_dim=int(basis_dim),
        min_value=int(support_min),
        max_value=int(support_max) + 1,
    )
    support_basis = l2_normalize_rows(support_basis)
    pred_basis = l2_normalize_rows(np.asarray(predictions, dtype=np.float64))
    nearest_idx = np.argmax(pred_basis @ support_basis.T, axis=1)
    return np.asarray(support_values, dtype=np.int64)[nearest_idx]


def fit_activation_circular_probe(
    *,
    model,
    tokenizer,
    layer_idx: int,
    prompt_template: str,
    token_mode: str,
    value_min: int,
    value_max: int,
    basis_dim: int,
    batch_size: int,
    device: str,
) -> tuple[ActivationSteeringProbe, pd.DataFrame]:
    values = np.arange(value_min, value_max + 1, dtype=np.int64)
    prompts: list[str] = []
    token_idxs: list[int] = []
    for value in values.tolist():
        prompt, span = render_probe_prompt(prompt_template, int(value))
        prompts.append(prompt)
        token_idxs.append(resolve_token_idx(tokenizer=tokenizer, prompt=prompt, mention_span=span, mode=token_mode))

    activations = collect_prompt_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=layer_idx,
        batch_size=batch_size,
        device=device,
    )
    target_basis = circular_sinusoidal_encode(
        values,
        embedding_dim=int(basis_dim),
        min_value=int(value_min),
        max_value=int(value_max) + 1,
    )
    design = np.concatenate([activations.astype(np.float64), np.ones((activations.shape[0], 1), dtype=np.float64)], axis=1)
    solution, *_ = np.linalg.lstsq(design, target_basis.astype(np.float64), rcond=None)
    coefficients = solution[:-1]
    bias = solution[-1]
    predicted_basis = activations.astype(np.float64) @ coefficients + bias
    decoded_predictions = decode_circular_probe_predictions(
        predicted_basis,
        support_values=values,
        basis_dim=int(basis_dim),
        support_min=int(value_min),
        support_max=int(value_max),
    )
    scalar_probe = fit_linear_regression(activations, values.astype(np.float32), activations)
    decoder_matrix = np.linalg.pinv(coefficients)

    probe = ActivationSteeringProbe(
        layer_idx=int(layer_idx),
        prompt_template=prompt_template,
        token_mode=token_mode,
        basis_dim=int(basis_dim),
        support_min=int(value_min),
        support_max=int(value_max),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
        decoder_matrix=np.asarray(decoder_matrix, dtype=np.float64),
        fit_decoded_accuracy=exact_accuracy(decoded_predictions, values),
        fit_basis_mse=float(np.mean((predicted_basis - target_basis) ** 2)),
        fit_mae=mae(scalar_probe.predictions, values),
        fit_rmse=rmse(scalar_probe.predictions, values),
    )
    fit_rows = pd.DataFrame(
        {
            "value": values,
            "decoded_prediction": decoded_predictions.astype(np.int64),
            "decoded_correct": (decoded_predictions == values).astype(int),
        }
    )
    return probe, fit_rows


def resolve_shifted_values(
    *,
    task: ArithmeticTask,
    operand_mode: OperandMode,
    x1: int,
    x2: int,
    alpha: float,
) -> tuple[int, int, int]:
    alpha_int = int(round(alpha))
    if operand_mode == "x1":
        shifted_x1 = x1 + alpha_int
        shifted_x2 = x2
    else:
        shifted_x1 = x1
        shifted_x2 = x2 + alpha_int

    if task == "addition":
        shifted_target = shifted_x1 + shifted_x2
    elif task == "subtraction":
        shifted_target = shifted_x1 - shifted_x2
    elif task == "multiplication":
        shifted_target = shifted_x1 * shifted_x2
    else:
        raise ValueError(f"Unsupported task: {task}")
    return shifted_x1, shifted_x2, shifted_target


def make_task_examples(task: ArithmeticTask, multiplication_domain: dict[str, int] | None = None) -> list[ArithmeticExample]:
    if task == "addition":
        return generate_addition_examples()
    if task == "subtraction":
        return generate_subtraction_examples()
    if task == "multiplication":
        if multiplication_domain is None:
            raise ValueError("multiplication_domain is required for multiplication.")
        return generate_multiplication_examples(multiplication_domain)
    raise ValueError(f"Unsupported task: {task}")


def sample_examples(
    *,
    examples: list[ArithmeticExample],
    max_examples: int | None,
    seed: int,
) -> list[ArithmeticExample]:
    if max_examples is None or max_examples <= 0 or len(examples) <= max_examples:
        return examples
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(examples), size=int(max_examples), replace=False))
    return [examples[idx] for idx in keep.tolist()]


def evaluate_activation_steering(
    *,
    model,
    tokenizer,
    task: ArithmeticTask,
    template: str,
    operand_mode: OperandMode,
    probe: ActivationSteeringProbe,
    alpha_values: list[float],
    batch_size: int,
    device: str,
    seed: int,
    max_examples: int | None = None,
    filter_baseline_correct_only: bool = True,
    multiplication_domain: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_info = build_numeric_token_info(tokenizer)
    tracked_values = numeric_info["values"]
    tracked_token_ids = numeric_info["token_ids"]
    tracked_values_float = tracked_values.astype(np.float64)

    examples = filter_single_token_targets(tokenizer, make_task_examples(task, multiplication_domain=multiplication_domain))
    examples = sample_examples(examples=examples, max_examples=max_examples, seed=seed)
    prompt_rows: list[dict[str, Any]] = []
    for example_id, ex in enumerate(examples):
        prompt, spans = render_prompt_with_operand_spans(template, ex.x1, ex.x2)
        token_idx = resolve_token_idx(
            tokenizer=tokenizer,
            prompt=prompt,
            mention_span=spans[operand_mode],
            mode=probe.token_mode,
        )
        prompt_rows.append(
            {
                "example_id": int(example_id),
                "task": task,
                "x1": int(ex.x1),
                "x2": int(ex.x2),
                "target": int(ex.target),
                "prompt": prompt,
                "token_idx": int(token_idx),
                "template": template,
                "operand_mode": operand_mode,
            }
        )
    prompt_df = pd.DataFrame(prompt_rows)
    prompts = prompt_df["prompt"].tolist()
    token_idxs = prompt_df["token_idx"].astype(int).tolist()
    baseline_target_token_ids = [encode_integer(tokenizer, int(value))[0] for value in prompt_df["target"].tolist()]

    baseline_logits_chunks, _, _ = _batch_forward_with_optional_direction(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        token_idxs=token_idxs,
        layer_idx=probe.layer_idx,
        batch_size=batch_size,
        device=device,
        scaled_directions=None,
        capture_pre_activations=False,
    )
    baseline_logits = np.concatenate(baseline_logits_chunks, axis=0)
    baseline_probs = _softmax_numpy(baseline_logits)
    baseline_pred_ids = baseline_probs.argmax(axis=-1)
    baseline_pred_text = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in baseline_pred_ids.tolist()]
    baseline_pred_value = [_parse_optional_int(text) for text in baseline_pred_text]
    baseline_numeric_probs = baseline_probs[:, tracked_token_ids]
    baseline_numeric_mass = baseline_numeric_probs.sum(axis=1)
    baseline_expected = (baseline_numeric_probs * tracked_values_float[None, :]).sum(axis=1) / np.clip(baseline_numeric_mass, 1e-8, None)
    baseline_target_prob = baseline_probs[np.arange(len(prompt_df)), np.asarray(baseline_target_token_ids, dtype=np.int64)]
    baseline_df_all = prompt_df.assign(
        alpha=0.0,
        shifted_x1=prompt_df["x1"],
        shifted_x2=prompt_df["x2"],
        shifted_target=prompt_df["target"],
        predicted_token_id=baseline_pred_ids.astype(np.int64),
        predicted_text=baseline_pred_text,
        predicted_value=baseline_pred_value,
        expected_value=baseline_expected.astype(np.float64),
        numeric_mass=baseline_numeric_mass.astype(np.float64),
        target_token_id=baseline_target_token_ids,
        target_prob=baseline_target_prob.astype(np.float64),
        correct=(baseline_pred_ids == np.asarray(baseline_target_token_ids)).astype(int),
        stage="baseline",
    )
    num_sampled_examples = int(len(baseline_df_all))
    baseline_full_accuracy = float(baseline_df_all["correct"].mean()) if num_sampled_examples else float("nan")
    if filter_baseline_correct_only:
        eligible_mask = baseline_df_all["correct"].astype(bool).to_numpy()
    else:
        eligible_mask = np.ones(num_sampled_examples, dtype=bool)
    baseline_df = baseline_df_all.loc[eligible_mask].reset_index(drop=True)
    num_eligible_examples = int(len(baseline_df))

    steered_rows: list[pd.DataFrame] = [baseline_df]
    baseline_key = baseline_df[["example_id", "predicted_token_id", "predicted_text", "predicted_value", "expected_value", "target_prob", "correct"]].rename(
        columns={
            "predicted_token_id": "baseline_predicted_token_id",
            "predicted_text": "baseline_predicted_text",
            "predicted_value": "baseline_predicted_value",
            "expected_value": "baseline_expected_value",
            "target_prob": "baseline_target_prob",
            "correct": "baseline_correct",
        }
    )

    for alpha in alpha_values:
        valid_rows: list[dict[str, Any]] = []
        for _, row in baseline_df.iterrows():
            shifted_x1, shifted_x2, shifted_target = resolve_shifted_values(
                task=task,
                operand_mode=operand_mode,
                x1=int(row["x1"]),
                x2=int(row["x2"]),
                alpha=float(alpha),
            )
            if not (0 <= shifted_x1 <= 999 and 0 <= shifted_x2 <= 999 and 0 <= shifted_target <= 999):
                continue
            target_ids = encode_integer(tokenizer, int(shifted_target))
            if len(target_ids) != 1:
                continue
            valid_rows.append(
                {
                    **row.to_dict(),
                    "alpha": float(alpha),
                    "shifted_x1": int(shifted_x1),
                    "shifted_x2": int(shifted_x2),
                    "shifted_target": int(shifted_target),
                    "target_token_id": int(target_ids[0]),
                }
            )
        if not valid_rows:
            continue
        valid_df = pd.DataFrame(valid_rows)
        source_values = valid_df[operand_mode].to_numpy(dtype=np.int64)
        target_values = source_values + int(round(float(alpha)))
        source_basis = circular_sinusoidal_encode(
            source_values,
            embedding_dim=int(probe.basis_dim),
            min_value=int(probe.support_min),
            max_value=int(probe.support_max) + 1,
        )
        target_basis = circular_sinusoidal_encode(
            target_values,
            embedding_dim=int(probe.basis_dim),
            min_value=int(probe.support_min),
            max_value=int(probe.support_max) + 1,
        )
        basis_delta = target_basis - source_basis
        scaled_directions = basis_delta @ probe.decoder_matrix
        logits_chunks, _, _ = _batch_forward_with_optional_direction(
            model=model,
            tokenizer=tokenizer,
            prompts=valid_df["prompt"].tolist(),
            token_idxs=valid_df["token_idx"].astype(int).tolist(),
            layer_idx=probe.layer_idx,
            batch_size=batch_size,
            device=device,
            scaled_directions=scaled_directions,
            capture_pre_activations=False,
        )
        logits = np.concatenate(logits_chunks, axis=0)
        probs = _softmax_numpy(logits)
        pred_ids = probs.argmax(axis=-1).astype(np.int64)
        pred_text = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in pred_ids.tolist()]
        pred_value = [_parse_optional_int(text) for text in pred_text]
        numeric_probs = probs[:, tracked_token_ids]
        numeric_mass = numeric_probs.sum(axis=1)
        expected = (numeric_probs * tracked_values_float[None, :]).sum(axis=1) / np.clip(numeric_mass, 1e-8, None)
        target_probs = probs[np.arange(len(valid_df)), valid_df["target_token_id"].to_numpy(dtype=np.int64)]
        steered_df = valid_df.assign(
            predicted_token_id=pred_ids,
            predicted_text=pred_text,
            predicted_value=pred_value,
            expected_value=expected.astype(np.float64),
            numeric_mass=numeric_mass.astype(np.float64),
            target_prob=target_probs.astype(np.float64),
            correct=(pred_ids == valid_df["target_token_id"].to_numpy(dtype=np.int64)).astype(int),
            stage="steered",
        ).merge(baseline_key, on="example_id", how="left")
        steered_df["expected_shift"] = steered_df["expected_value"] - steered_df["baseline_expected_value"]
        steered_df["target_error"] = steered_df["expected_value"] - steered_df["shifted_target"]
        steered_df["target_prob_gain"] = steered_df["target_prob"] - steered_df["baseline_target_prob"]
        steered_rows.append(steered_df)

    details = pd.concat(steered_rows, ignore_index=True)
    steered_only = details.loc[details["stage"] == "steered"].copy()
    if steered_only.empty:
        summary = pd.DataFrame(
            [
                {
                    "task": task,
                    "operand_mode": operand_mode,
                    "alpha": float("nan"),
                    "num_examples": 0,
                    "num_sampled_examples": num_sampled_examples,
                    "num_eligible_examples": num_eligible_examples,
                    "baseline_full_accuracy": baseline_full_accuracy,
                    "baseline_filter_mode": "correct_only" if filter_baseline_correct_only else "all_sampled",
                    "argmax_accuracy": float("nan"),
                    "mean_expected_shift": float("nan"),
                    "mean_target_error": float("nan"),
                    "mean_target_prob_gain": float("nan"),
                }
            ]
        )
    else:
        summary = (
            steered_only.groupby(["task", "operand_mode", "alpha"], as_index=False)
            .agg(
                num_examples=("example_id", "size"),
                argmax_accuracy=("correct", "mean"),
                mean_expected_shift=("expected_shift", "mean"),
                mean_target_error=("target_error", "mean"),
                mean_target_prob_gain=("target_prob_gain", "mean"),
            )
            .sort_values(["task", "operand_mode", "alpha"])
            .reset_index(drop=True)
        )
        summary["num_sampled_examples"] = num_sampled_examples
        summary["num_eligible_examples"] = num_eligible_examples
        summary["baseline_full_accuracy"] = baseline_full_accuracy
        summary["baseline_filter_mode"] = "correct_only" if filter_baseline_correct_only else "all_sampled"
    return details, summary
