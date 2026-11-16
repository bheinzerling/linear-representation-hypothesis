import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = runpy.run_path(str(ROOT / "experiments/numerical/src/figure7_historical_comparison.py"))
historical_activation_comparison = HELPER["historical_activation_comparison"]
prepare_historical_comparison = HELPER["prepare_historical_comparison"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_preparation_does_not_load_or_create_historical_inputs(tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Default preparation attempted to load historical activations")

    monkeypatch.setattr(np, "load", unexpected_load)
    assert prepare_historical_comparison(None, tmp_path) == {
        "status": "not_requested", "enabled": False,
    }
    assert not list(tmp_path.iterdir())


def test_unbound_historical_file_is_ignored(tmp_path, monkeypatch):
    (tmp_path / "archived_centroids.npy").write_bytes(b"unrelated stale file")

    def unexpected_load(*args, **kwargs):
        pytest.fail("An unbound historical file was loaded")

    monkeypatch.setattr(np, "load", unexpected_load)
    result = historical_activation_comparison(tmp_path, {"files": {}}, np.ones((1000, 2)), "revision")
    assert result["status"] == "not_requested"
    assert result["comparison_performed"] is False
    assert "relative_l2" not in result


def test_requested_archive_preserves_float32_averaging_and_source_hash(tmp_path):
    values = np.tile(np.arange(1000), 2)
    activations = np.random.default_rng(1729).normal(size=(2000, 3)).astype(np.float32)
    path = tmp_path / "history.npz"
    np.savez(path, source_values=np.append(values, 999),
             style_ids=np.array(["addition_formula_rhs_source"] * 2000 + ["other"]),
             activations=np.vstack([activations, np.ones((1, 3), dtype=np.float32)]))
    result = prepare_historical_comparison(path, tmp_path)
    expected = np.stack([activations[values == value].mean(axis=0) for value in range(1000)])
    np.testing.assert_array_equal(np.load(tmp_path / "archived_centroids.npy"), expected)
    assert result == {"status": "prepared", "enabled": True, "source_sha256": digest(path)}


def test_requested_comparison_supports_original_manifests(tmp_path):
    path = tmp_path / "archived_centroids.npy"
    old = np.ones((1000, 2), dtype=np.float32)
    np.save(path, old)
    # Original version-1 manifests bind this file without an optional-status field.
    result = historical_activation_comparison(tmp_path, {"files": {path.name: digest(path)}}, old * 2, "revision")
    assert result["comparison_performed"] is True
    assert result["relative_l2"] == pytest.approx(1.0)
    assert result["max_absolute_difference"] == pytest.approx(1.0)


def test_changed_historical_centroids_are_rejected(tmp_path):
    path = tmp_path / "archived_centroids.npy"
    np.save(path, np.ones((1000, 2), dtype=np.float32))
    manifest = {"files": {path.name: digest(path)}}
    np.save(path, np.ones((1000, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="input manifest"):
        historical_activation_comparison(tmp_path, manifest, np.ones((1000, 2)), "revision")


@pytest.mark.parametrize("kind", ["missing_value", "nonfinite"])
def test_invalid_requested_archive_is_rejected(tmp_path, kind):
    values = np.arange(999 if kind == "missing_value" else 1000)
    activations = np.ones((len(values), 2), dtype=np.float32)
    if kind == "nonfinite":
        activations[0, 0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez(path, source_values=values, activations=activations,
             style_ids=np.array(["addition_formula_rhs_source"] * len(values)))
    with pytest.raises(ValueError, match="all 1,000|finite"):
        prepare_historical_comparison(path, tmp_path)
    assert not (tmp_path / "archived_centroids.npy").exists()


@pytest.mark.parametrize("old", [np.zeros((1000, 2)), np.ones((999, 2)), np.full((1000, 2), np.nan)])
def test_invalid_requested_centroids_are_rejected(tmp_path, old):
    path = tmp_path / "archived_centroids.npy"
    np.save(path, old)
    with pytest.raises(ValueError, match="nonzero|shape"):
        historical_activation_comparison(tmp_path, {"files": {path.name: digest(path)}},
                                        np.ones((1000, 2)), "revision")
