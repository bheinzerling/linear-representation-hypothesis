from __future__ import annotations

import numpy as np

from src.activation_steering import circular_sinusoidal_encode
from src.magnitude_intervention import MagnitudeProbe, PLS_X_WEIGHT1_PROBE_ARCHITECTURE
from src.probe_architectures import CIRCULAR_CLS_PROBE_ARCHITECTURE, CIRCULAR_REG_PROBE_ARCHITECTURE, canonicalize_probe_architecture_name


def probe_path_features(
    probe: MagnitudeProbe,
    *,
    source_values: np.ndarray,
    target_values: np.ndarray,
    path_positions: np.ndarray,
) -> np.ndarray:
    """Return feature-space points on a probe's 1D source-to-target path.

    The returned array has shape `(num_examples, num_positions, feature_dim)`.
    `path_positions=0` gives the source code and `path_positions=1` gives
    the target code. Positions outside `[0, 1]` are allowed for overshoot
    diagnostics. For circular-classifier paths, values outside the witnessed
    support are clamped to the support endpoints.
    """
    architecture = canonicalize_probe_architecture_name(probe.architecture)
    sources = np.asarray(source_values, dtype=np.float64).reshape(-1)
    targets = np.asarray(target_values, dtype=np.float64).reshape(-1)
    positions = np.asarray(path_positions, dtype=np.float64).reshape(-1)
    if sources.shape != targets.shape:
        raise ValueError(f"source_values and target_values must have the same shape, got {sources.shape} and {targets.shape}.")
    values = sources[:, None] + positions[None, :] * (targets[:, None] - sources[:, None])

    if architecture in {"linear", "pls", PLS_X_WEIGHT1_PROBE_ARCHITECTURE}:
        return values[:, :, None].astype(np.float64)

    if architecture == CIRCULAR_REG_PROBE_ARCHITECTURE:
        flat_features = circular_sinusoidal_encode(
            values.reshape(-1),
            embedding_dim=int(probe.support_features.shape[1]),
            min_value=int(np.min(probe.support_values)),
            max_value=int(np.max(probe.support_values)) + 1,
        )
        return np.asarray(flat_features, dtype=np.float64).reshape(values.shape[0], values.shape[1], -1)

    if architecture == CIRCULAR_CLS_PROBE_ARCHITECTURE:
        support_values = np.asarray(probe.support_values, dtype=np.float64).reshape(-1)
        support_features = np.asarray(probe.support_features, dtype=np.float64)
        if len(support_values) != support_features.shape[0]:
            raise ValueError("support_values and support_features disagree in length.")
        flat_values = values.reshape(-1)
        flat_features = np.empty((flat_values.shape[0], support_features.shape[1]), dtype=np.float64)
        for dim in range(support_features.shape[1]):
            flat_features[:, dim] = np.interp(
                flat_values,
                support_values,
                support_features[:, dim],
                left=support_features[0, dim],
                right=support_features[-1, dim],
            )
        return flat_features.reshape(values.shape[0], values.shape[1], -1)

    raise ValueError(f"Probe paths are not defined for architecture {architecture!r}.")


def probe_path_deltas(
    probe: MagnitudeProbe,
    *,
    source_values: np.ndarray,
    target_values: np.ndarray,
    path_positions: np.ndarray,
) -> np.ndarray:
    """Return decoded activation deltas along a probe's 1D value/support path.

    The returned array has shape `(num_examples, num_positions, activation_dim)`.
    Position `0` returns zero up to numerical precision. Position `1` matches
    `probe.intervention_delta(source_values, target_values)`.
    """
    path_features = probe_path_features(
        probe,
        source_values=source_values,
        target_values=target_values,
        path_positions=path_positions,
    )
    source_features = path_features[:, :1, :]
    decoder = np.asarray(probe.decoder_matrix, dtype=np.float64)
    return np.asarray((path_features - source_features) @ decoder, dtype=np.float64)


def probe_manifold_feature_path(
    probe: MagnitudeProbe,
    *,
    source_values: np.ndarray,
    target_values: np.ndarray,
    path_positions: np.ndarray,
) -> np.ndarray:
    """Backward-compatible alias for older circular path callers."""
    return probe_path_features(
        probe,
        source_values=source_values,
        target_values=target_values,
        path_positions=path_positions,
    )


def probe_manifold_delta_path(
    probe: MagnitudeProbe,
    *,
    source_values: np.ndarray,
    target_values: np.ndarray,
    path_positions: np.ndarray,
) -> np.ndarray:
    """Backward-compatible alias for older circular path callers."""
    return probe_path_deltas(
        probe,
        source_values=source_values,
        target_values=target_values,
        path_positions=path_positions,
    )
