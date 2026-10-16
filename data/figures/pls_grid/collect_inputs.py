"""Freeze the PLS records used for the numerical invariance design sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    sources = []

    def read(name: str) -> pd.DataFrame:
        path = args.suite / name
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
        return pd.read_csv(path).query("architecture == 'pls'").copy()

    directions = read("direction_pairs.csv.gz")
    transfer = read("q4_family_mean_alignment_paired.csv")
    assert len(directions) == 6 * 11 * 10
    assert len(transfer) == 6 * 11 * 25
    assert directions.groupby(["model_alias", "relative_layer"]).size().eq(10).all()
    assert transfer.groupby(["model_alias", "relative_layer"]).size().eq(25).all()
    assert directions.direction_cosine.between(-1 - 1e-9, 1 + 1e-9).all()
    assert (transfer.is_native_family == transfer.train_family.eq(transfer.test_family)).all()
    np.testing.assert_allclose(transfer.loc[transfer.is_native_family, "raw"],
                               transfer.loc[transfer.is_native_family, "family_mean_aligned"])
    assert set(directions.model_alias) == set(transfer.model_alias)
    models = directions[["model_alias", "model_id", "model_revision", "n_layers"]].drop_duplicates()
    directions[["model_alias", "relative_layer", "layer_idx", "left_family", "right_family",
                "direction_cosine"]].to_csv(OUTPUT / "pls_directions.csv", index=False)
    transfer[["model_alias", "relative_layer", "train_family", "test_family", "raw",
              "family_mean_aligned", "is_native_family"]].to_csv(OUTPUT / "pls_transfer.csv", index=False)
    manifest = {
        "sources": sources, "collector_sha256": sha256(Path(__file__)),
        "models": models.to_dict("records"),
        "relative_layers": sorted(directions.relative_layer.unique().tolist()),
        "split": {"train": 800, "development": 100, "test": 100, "value_disjoint": True},
        "probe": "PLS scalar magnitude decoder; independently fitted within each prompt family.",
        "capacity_selection": "Native-family development choices from the April 20 sweep.",
        "direction_comparison": "Signed cosine between unit scalar decoder weight vectors at the same checkpoint and layer.",
        "transfer": "Freeze source-family direction, gain, and intercept, and test the other family activations.",
        "alignment": "Translate each family's training-activation mean to the equal-family global training mean; freeze source direction and gain. Target labels are unused.",
        "averaging": "Equal weights for six models, eleven relative layers, and applicable family pairs. Compute angles before averaging.",
        "snapshots": {p.name: sha256(p) for p in sorted(OUTPUT.glob("*.csv"))},
    }
    (OUTPUT / "input_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(directions)} direction pairs and {len(transfer)} transfer records.")


if __name__ == "__main__":
    main()
