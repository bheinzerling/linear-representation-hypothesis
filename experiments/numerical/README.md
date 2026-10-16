# Numerical model experiments

This directory contains the numerical experiment source, configurations, and a
locked environment for Python 3.12. It covers numerical probing, prompt transfer,
complexity comparisons, deletion, classification, and coefficient prediction.

```bash
uv sync --locked
uv run python scripts/run_rank41_deletion.py --plan-only
```

Prepare the full input artifact using the command in
[the model experiment guide](../../docs/model-experiments.md). That guide
describes the experimental stages, model access, and replay commands.
Generated outputs belong in this directory's ignored `out/` directory.
