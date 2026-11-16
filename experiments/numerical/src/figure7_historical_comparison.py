"""Optional comparison with historical activations, separate from the matched run."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_historical_comparison(archive: Path | None, output: Path) -> dict:
    """Add historical centroids only when an archive is explicitly supplied."""
    if archive is None:
        return {"status": "not_requested", "enabled": False}
    archive = archive.resolve(strict=True)
    with np.load(archive) as old:
        mask = old["style_ids"] == "addition_formula_rhs_source"
        values = np.asarray(old["source_values"][mask], dtype=np.int64)
        activations = np.asarray(old["activations"][mask], dtype=np.float32)
    if (activations.ndim != 2 or len(activations) != len(values)
            or not np.array_equal(np.unique(values), np.arange(1000))):
        raise ValueError("Historical activations must contain all 1,000 aligned source values.")
    if not np.isfinite(activations).all():
        raise ValueError("Historical activations must be finite.")
    centroids = np.stack([activations[values == value].mean(axis=0) for value in range(1000)])
    np.save(output / "archived_centroids.npy", centroids)
    return {"status": "prepared", "enabled": True, "source_sha256": _sha256(archive)}


def historical_activation_comparison(inputs: Path, manifest: dict,
                                    centroids: np.ndarray, revision: str) -> dict:
    """Compare only historical centroids bound by the input manifest."""
    name = "archived_centroids.npy"
    if name not in manifest["files"]:
        return {
            "status": "not_requested", "comparison_performed": False,
            "new_revision": revision,
            "note": "Historical centroids were not requested; the matched experiment uses fresh activations.",
        }
    path = inputs / name
    if _sha256(path) != manifest["files"][name]:
        raise ValueError("Historical centroids differ from the input manifest.")
    old_centroids = np.load(path)
    if old_centroids.shape != centroids.shape or not np.isfinite(old_centroids).all():
        raise ValueError("Historical centroids must be finite and match the fresh centroid shape.")
    norm = np.linalg.norm(old_centroids)
    if norm == 0:
        raise ValueError("Historical centroid norm must be nonzero for relative L2 comparison.")
    return {
        "status": "completed", "comparison_performed": True,
        "historical_revision_recorded": False, "new_revision": revision,
        "relative_l2": float(np.linalg.norm(centroids - old_centroids) / norm),
        "max_absolute_difference": float(np.max(np.abs(centroids - old_centroids))),
        "note": "Historical revision and runtime were not fully recorded; the fresh matched comparison is primary.",
    }
