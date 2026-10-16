# Property model experiments

This directory contains property-direction fitting, prompt screening, and
number-list steering with their source dependencies and a locked Python 3.13
environment.

```bash
uv sync --locked
uv run python scripts/run_number_property_list_steering.py --help
```

[The model experiment guide](../../docs/model-experiments.md) describes the
complete input artifact and replay commands. `reported_experiment_config.json`
records the experimental conditions. The numerical directory's parity
classification runner imports this directory's prompt construction and scoring
functions. Generated results belong in the ignored `out/` directory.
