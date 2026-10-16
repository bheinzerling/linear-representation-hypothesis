# Numerical evidence

This directory contains the observations consumed by the CPU numerical runner.
`reported_config.json` records the experimental conditions; `input_provenance.json`
records input identities and transformations. `../MANIFEST.sha256` covers all
supplied data files.

| Files | Contents |
| --- | --- |
| `probe_outputs.csv.gz`, `probe_metrics.csv`, `reproduced_test_predictions.csv` | Supervised outputs, held-out summaries, and the available per-value predictions |
| `classifier_median_certificate.json` | Integer-error bounds that determine the classifier's median absolute error from its aggregate counts |
| `probe_capacity.csv`, `appendix_capacity_*.jsonl*` | Representative complexity settings and the complete six-model appendix curves |
| `pls_directions.csv.gz`, `pls_transfer.csv.gz`, `pls_transfer_means.csv` | Direction comparisons, model/depth transfer cells, and their pooled summaries |
| `deletion_*.csv*` | Reduced observations, split summaries, uncertainty, and paired control comparisons |
| `property_accessibility.csv`, `parity_contradiction_*` | Property-probe summaries, contradiction counts, and examples |
| `steering_accuracy_rows.csv.gz`, `steering_uncertainty.csv` | Matched interventions, no-edit outputs, and paired uncertainty certificates |
| `coefficient_prediction_rows.csv.gz`, `dimensionality_control.csv` | Coefficient predictions and the separate output-scored search |

The historical field name `effective_df` stores ridge effective degrees of
freedom, PLS component count, or circular-code dimension according to the probe
family. These quantities do not define a common complexity budget.

Bootstrap intervals measure evaluation variability conditional on stored fits
and intervention conditions. They do not include uncertainty from fitting new
probes, selecting other sites, or changing checkpoints. Individual classifier
predictions are unavailable; its stored aggregate metrics can be checked, but
its evaluation intervals cannot be reconstructed. Contradiction counts describe
the fixed baseline-correct evaluation sets.

Coefficient predictors use twenty held-out source prompts and share target
values across training and evaluation. The joint predictors also receive source
activations, and methods use different doses. The output-scored search uses a
separate evaluation and the requested output to score candidate edits. These
conditions constrain the comparisons reported in the paper.
