# Numerical evidence

This directory contains the observations consumed by the CPU numerical runner.
`reported_config.json` records the experimental conditions; `input_provenance.json`
records input identities and transformations. `../MANIFEST.sha256` covers all
supplied data files.

The historical Figure 3 activation cache is excluded from the release and its
checkpoint revision was not recorded. Exact regeneration of that input is
unavailable. The released predictions and observations reproduce the reported
CPU results; [the model guide](../../docs/model-experiments.md#historical-figure-3-activations)
describes the distinction between those calculations and new model-forward replications.

| Files | Contents |
| --- | --- |
| `probe_outputs.csv.gz`, `probe_metrics.csv`, `reproduced_test_predictions.csv` | Supervised outputs, held-out summaries, and the available per-value predictions |
| `classifier_median_certificate.json` | Integer-error bounds that determine the classifier's median absolute error from its aggregate counts |
| `probe_capacity.csv`, `appendix_capacity_*.jsonl*` | Representative complexity settings and the complete six-model appendix curves |
| `pls_directions.csv.gz`, `pls_transfer.csv.gz`, `pls_transfer_means.csv` | Direction comparisons, model/depth transfer cells, and their pooled summaries |
| `deletion_*.csv*` | Reduced observations, split summaries, uncertainty, and paired control comparisons |
| `property_accessibility.csv`, `parity_contradiction_*` | Property-probe summaries, contradiction counts, and examples |
| `../figures/numerical_sufficiency_atlas/individual_prompt_fits/` | Figure 7 decoding and 16,000 paired interventions; both edit constructions, controls, post-edit probe predictions, code errors and edit norms |
| `coefficient_prediction_rows.csv.gz`, `dimensionality_control.csv` | Coefficient predictions and the separate output-scored search |

The field `effective_df` stores ridge effective degrees of
freedom, PLS component count, or circular-code dimension according to the probe
family. These quantities do not define a common complexity budget.

Bootstrap intervals measure evaluation variability conditional on stored fits
and intervention conditions. They do not include uncertainty from fitting new
probes, selecting other sites, or changing checkpoints. All four probes have individual
predictions and conditional evaluation intervals. Contradiction counts describe
the fixed baseline-correct evaluation sets.

Coefficient predictors use twenty held-out source prompts and share target
values across training and evaluation. The joint predictors also receive source
activations, and methods use different doses. The output-scored search uses a
separate evaluation and the requested output to score candidate edits. These
conditions constrain the comparisons reported in the paper.

Figure 7 uses prediction-corrected edits, which set each probe to the target code. Both probes decode the requested operand in all 2,000 cases but yield zero correct counterfactual sums. PCA centroid shifts succeed in 1,995 cases. The model-forward sources and commands are in `../../docs/model-experiments.md`.
