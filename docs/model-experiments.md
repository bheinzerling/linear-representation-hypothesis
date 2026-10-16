# Model experiments

The two experiment directories have separate environments:

| Directory | Python | Experiments |
| --- | --- | --- |
| `experiments/numerical/` | 3.12 | Numerical probes, complexity, prompt transfer, deletion, parity classification, and coefficient prediction |
| `experiments/properties/` | 3.13 | Property directions, prompt screening, and number-list steering |

Run `uv sync --locked` within the selected directory. Each directory includes
its source dependencies, configuration files, and resolved `uv.lock`. Model
runs require accelerator memory appropriate for the checkpoint and batch size.
Obtain checkpoints from their providers under the relevant access terms.

## Inputs and configurations

The `model-inputs.tar.gz` artifact contains complete reusable prompt datasets,
fitted tensors, and intermediate results. Its contents and checksums are in
`data/manifests/model-inputs.json`. Extract it from the repository root:

```bash
uv run python scripts/prepare_model_inputs.py --archive artifacts/model-inputs.tar.gz
```

The command verifies the archive and each extracted file, places the inputs
under the two experiment directories, and refuses to overwrite differing
existing files. These are full inputs; no tensor axes or prompt tables are
truncated. Checkpoints and large activation caches are obtained or generated
separately.

`data/numerical/reported_config.json` records the paper's comparison conditions.
Preserve checkpoint revisions, numerical precision, prompt partitions, and
selection settings for a matched rerun. The appendix complexity sweep uses
Llama 3.1 8B; the transfer sweep uses Llama 3 8B. The complexity configurations
for this distinction are in `configs/figure10_llama31_capacity_20260909/`.

## Transfer and complexity

From `experiments/numerical/`, the suite collects activations when its cache is
absent and fits the specified probes:

```bash
uv run python scripts/run_prompt_transfer_robustness_suite.py --model llama_3_2_1b --mode all
```

The other suite model keys are `llama_3_2_3b`, `llama_3_8b`, `olmo_3_7b_base`,
and `olmo_2_13b`. GPT-oss uses
`scripts/run_prompt_transfer_robustness_gpt_oss.py --model gpt_oss_20b --mode all`,
including its recorded deterministic-kernel exception. The comparison uses
eleven relative depths and five prompt variants.

`build_prompt_transfer_robustness_bundle.py` aggregates completed runs.
`run_source_value_probe_capacity_sweep_relative_layers.py` performs complexity
sweeps; `run_source_value_probe_prompt_style_transfer_relative_layers.py`
performs native and cross-prompt evaluation. Inspect each script's `--help`
for configuration and output options.

## Deletion

The model-input artifact provides the ten prepared deletion bundles. From the
numerical directory:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run python scripts/run_rank41_deletion.py --plan-only
uv run python scripts/run_rank41_deletion.py
```

The planner checks input hashes and enumerates 980 shards: ten splits, seven
tasks, and fourteen intervention conditions. Completed shards are resumed.
`analyze_rank41_deletion.py` aggregates the results. To fit new bases and
controls from newly collected activations, use `prepare_rank41_deletion.py`.

## Parity classification and enumeration

From the numerical directory:

```bash
uv run python scripts/run_parity_probe_causal_contextualization.py --output-dir out/parity_classification_replay --no-local-files-only
```

This runner imports prompt construction and scoring from the sibling
`experiments/properties/src/` directory.

From `experiments/properties/`, replay the enumeration condition with:

```bash
uv run python scripts/run_number_property_list_steering.py \
  --model Qwen/Qwen3.6-27B \
  --behavior-run-dir out/number_property_list_filter/qwen3_6_27b_parity_corrected_v1 \
  --direction-run-dir out/digit_parity_steering_cross_model/full_v1/qwen3_6_27b \
  --output-dir out/parity_enumeration_replay --no-local-files-only
```

Direction fitting and selection use `run_olmo_number_property_steering.py`.
The recorded enumeration seed is 20260718; the value-disjoint direction-fit
seed is 1729. The included screening sources construct the evaluation lists.

## Coefficient prediction

The numerical sources include the following dependency sequence for collecting
new activations and fitting the intermediate objects:

| Stage | Script under `experiments/numerical/scripts/` |
| --- | --- |
| Numerical intervention panel | `run_source_value_probe_manifold_steering.py` |
| Property directions and split bases | `run_figure1_bag_of_features_steering.py`, `run_property_probe_training_scheme_comparison.py` |
| Target-informed patches and intervention bases | `run_figure1_oracle_property_scaling.py`, `run_figure1_best_in_span_causal_ceiling.py` |
| Ordered property blocks | `run_figure1_owner_block_prefix_steering.py`, `run_figure1_oracle_block_replacement.py` |
| Joint coefficient prediction | `run_direct_oracle_coordinate_regression.py` |

Use each entry point's `--help` for stage and output options. The CPU evidence
contains all 1,200 held-out coefficient-prediction rows and the summary from the
separate output-scored search. Their input information, doses, and reused target
values are documented in the numerical data guide and manuscript.
