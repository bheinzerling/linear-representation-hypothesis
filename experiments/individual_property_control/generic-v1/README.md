# Original generic-label experiment

This directory preserves the registered execution source for the original generic-label sweep. Its source digest is `b2f76eb6…`.

Run `uv sync --locked`, then inspect the complete plan:

```bash
uv run python scripts/run_individual_property_control.py --phase plan --config configs/individual-property-control/rikyu-v1.json
```

The plan contains 114 main runs; `--pilot` selects six validation runs. Use the parent README for checkpoint acquisition and worker commands.
