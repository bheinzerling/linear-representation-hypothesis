# Online description length of numerical probes

These observations reproduce the appendix comparison of all 73 Figure 3d
settings under two probability models and three training-example orderings.
The experiment encodes the original 800 training labels over the common
alphabet 0–999. The 100 development and 100 test labels are reserved for
separate full-training-set fits.

`observations.npz` contains all 350,400 per-example coding costs, the three
label orderings, block predictions and calibration temperatures. It also
contains the development and test predictions for each of the 73 settings.
The arrays use the setting, ordering and coder identifiers stored in the same
file. `protocol.json` records the complete fitting roster, prefix boundaries,
calibration procedure, package versions and original source hashes.

`blocks.csv`, `per_order_metrics.csv` and `summary.csv` retain the published
aggregations as comparison certificates. The CPU calculation reconstructs
all three from the per-example costs and held-out predictions, verifies their
agreement, and regenerates the figure and five-setting LaTeX table:

```bash
uv run python -m lrh.analysis.probe_mdl \
  --data-dir data/probe_mdl --output-dir out/probe_mdl
```

Costs are in bits. The first three encoded labels use uniform probabilities;
later blocks use fits and temperature calibration based on earlier labels.
The `native` code uses the probe's scores; `ordinal` uses a common distribution
over absolute numerical errors. Both mix predictions with 1% uniform
probability. Reported standard deviations describe the three orderings of one
fixed value partition. The classifier's refitting seed is fixed at 1729;
its ten-dimensional-code test MAE is 1.14, versus 1.09 in the earlier sweep.

The exporter checks every original run artifact before packing these arrays.
`input_snapshot.json` records the exported files' sizes and SHA-256 hashes;
the repository manifest covers the complete directory. Generated integers and
derived observations use the repository's CC BY 4.0 license.

Probe-fitting code and commands are in
[`docs/model-experiments.md`](../../docs/model-experiments.md#online-description-length).
New fits require regenerated activations and the numerical experiment
environment. The CPU calculation above uses the released observations.
