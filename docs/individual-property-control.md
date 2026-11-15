# Individual property-control experiments

The CPU runner replays complete saved generations from both instruction formats.
It independently scores the outputs, checks numeral partitions and baseline
qualification, reconstructs development selection and all paired controls, and
recalculates the bootstrap intervals. It enumerates all nineteen properties when
applying the table's highlighting rule. The resulting seven-property list includes
divisibility by 10, whose mean success is 15.0% against zero for all controls.
These are observed comparisons; the exploratory intervals have no multiplicity
correction.

The `experiments/individual_property_control/` directory contains the experiment runner, deterministic
prompt and split builders, fitting code, independent analysis, generation adapter,
and their local imports. Its two configurations preserve the model revisions and
experimental settings. The `path` fields identify portable local destinations;
the reconstruction records retain the actual model IDs and revisions. Absolute
machine paths and hostnames have been removed from those records.

For new model execution, work in `experiments/individual_property_control/`, run `uv sync --locked`, and
obtain each checkpoint under its provider's access terms. Use the exact `revision`
in each configuration and place it at the corresponding `models/<revision>` path,
or change only the local `path` field. Then inspect the complete plan:

```bash
uv run python scripts/run_individual_property_control.py --phase plan \
  --config configs/individual-property-control/rikyu-explicit-list-v2.json
```

The generic-label sweep used an earlier execution revision, preserved in
`generic-v1/`. For that format, work in `generic-v1/`, install its locked
environment, and use `configs/individual-property-control/rikyu-v1.json` with
the same plan and worker commands. Its execution-source digest is
`b2f76eb6…`; the explicit-predicate revision's digest is `a0606702…`.
The CPU replay checks these identities against every saved run.

Each plan contains 114 main runs; `--pilot` selects six runtime-validation runs.
For each task index 0 through 113, execute the corresponding configuration:

```bash
uv run python scripts/run_individual_property_control.py --phase worker \
  --config configs/individual-property-control/rikyu-explicit-list-v2.json \
  --task-index 0 --output-root out/explicit-list
```

The two checkpoints use BF16, greedy generation, and math attention. Qwen uses
its chat template with thinking disabled; Llama uses plain prompts. The recorded
GPU environment used Python 3.13.12, Torch 2.12.1+cu130, Transformers 5.2.0, and
an NVIDIA GB200. The included lockfile provides the source environment; GPU,
driver, and kernel differences can change regenerated outputs. The CPU replay
does not load a model or refit a probe.
