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

Outputs are grouped under `out/reproduction/{mentions,numerical,literature}/`.
The root `run.json` records input and source hashes, package versions, and output
hashes. `DONE` is written after all three analyses complete. A subsequent run
checks these records before reusing completed outputs. Changed inputs, code,
or outputs cause the calculations to run again.

Individual analyses can be run directly:

```bash
uv run python -m lrh.analysis.mentions --data-dir data/mentions --output-dir out/mentions
uv run python -m lrh.analysis.numerical --data-dir data/numerical --output-dir out/numerical
uv run python -m lrh.analysis.literature --data-dir data/literature --output-dir out/literature
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

To create a portable source ZIP and artifact checksums:

```bash
uv run python scripts/build_release.py
```

The ZIP includes code, compact evidence, manuscript sources, and figure assets.
Generated outputs, environments, and accelerator input bundles are kept
separately in `artifacts/`.
