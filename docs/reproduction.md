# CPU reproduction

Use Python 3.12 and the root `uv.lock`. `uv sync --locked` installs NumPy,
pandas, and Matplotlib without the accelerator dependencies.

```bash
uv sync --locked
uv run python scripts/verify.py
uv run python scripts/reproduce.py
```

`verify.py` checks every frozen data file against `data/MANIFEST.sha256`, checks
local imports in the model sources, and rejects symbolic links in the supplied
content. The reproduction command recalculates the numerical results and
literature aggregations, including their comparisons with stored certificates.

Outputs are grouped under `out/reproduction/{mentions,numerical,literature,probe_mdl}/`.
The root `run.json` records input and source hashes, package versions, and output
hashes. `DONE` is written after all four analyses complete. A subsequent run
checks these records before reusing completed outputs. Changed inputs, code,
or outputs cause the calculations to run again.

Individual analyses can be run directly:

```bash
uv run python -m lrh.analysis.mentions --data-dir data/mentions --output-dir out/mentions
uv run python -m lrh.analysis.numerical --data-dir data/numerical --output-dir out/numerical
uv run python -m lrh.analysis.literature --data-dir data/literature --output-dir out/literature
uv run python -m lrh.analysis.probe_mdl --data-dir data/probe_mdl --output-dir out/probe_mdl
```

For development checks:

```bash
uv sync --locked --group dev
uv run --group dev pytest
uv run --group dev ruff check src scripts tests
```

The numerical results are recalculated from saved observations. This workflow
does not collect new activations or refit models. Bootstrap intervals condition
on the stored fitted probes, intervention bases, and experimental conditions.
The numerical data guide identifies unavailable predictions and the fixed
evaluation sets for which no population-sampling interpretation is claimed.

The MDL analysis sums per-example coding costs, recalculates held-out errors,
and verifies all 4,380 block records, 438 per-order records and 146 summary rows.
It regenerates the two-panel figure and the five-setting LaTeX table. The
recorded temperatures and label orderings expose the coding procedure;
[`data/probe_mdl/README.md`](../data/probe_mdl/README.md) describes the arrays.

To create a portable source ZIP and artifact checksums:

```bash
uv run python scripts/build_release.py
```

The ZIP includes code, compact evidence, manuscript sources, and figure assets.
Generated outputs, environments, and accelerator input bundles are kept
separately in `artifacts/`.
