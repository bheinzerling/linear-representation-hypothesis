# Completed author audit

One author completed this review on 3 October 2026. The sixty episodes come
from 54 documents and comprise all duplicate-coded cases in the two automated
reliability samples: thirty from `strong_coding` and thirty from
`expanded_coding`. Each original sample selected ten episodes at random from
each of three citation-rank groups. Presentation order used seed 20261002.
The review interface displayed paper titles, years, passages, and links to the
original PDFs; it hid machine judgments, citation ranks, and sampling groups.

The earlier thirty-case packet targeted twenty purpose disagreements and ten
controls. That planning packet is separate from the completed sixty-case
review. The audit reported here uses the sixty-case review.

## Released records

- `annotations.csv` contains the latest completed judgment for each case.
  `author_1` denotes the single author reviewer. Episode status is
  `substantive`, `bibliography_only`, `structural_fragment`, or
  `context_insufficient`. The four purposes use `1` for present, `0` for
  absent, and `U` for uncertain; non-substantive episodes use `N/A`.
- `case_manifest.csv` joins the review case to its episode, sampling phase,
  document identifier, citation rank, and PDF pages. Context hashes identify
  the passages used in the private review without distributing source PDFs.
- `label_config.xml` preserves the instructions and choices shown to the author.
- `provenance.json` identifies the saved source export and study inputs by
  SHA-256, records the presentation procedure, and documents the public export.
- `metrics.json` records episode-status counts and comparisons with the
  designated primary and secondary automated annotations, pooled and by sample.

The export preserves the submitted labels. Reviewer accounts, free-text notes,
and revision history stay in the private annotation store. Both automated
coding sessions used `gpt-5.6-sol` with high reasoning effort. Their original
assignments and labels remain in `../coding/`.

## Reproduction and interpretation

From the repository root, run:

```bash
uv run python -m lrh.analysis.literature --data-dir data/literature --output-dir out/literature
uv run python scripts/verify.py
```

The author classified 47 episodes as substantive, twelve as bibliography
entries, and one as a structural fragment. Each coding agent agrees with the
author on 58 of sixty episode statuses. The two disagreements concern an
episode labeled structural fragment by the author and substantive by both
agents, and an episode labeled substantive by the author and insufficient
context by both agents.

Purpose comparisons use the 46 episodes classified as substantive by both
the author and the relevant agent. Binary comparisons exclude uncertainty
separately for each purpose, giving denominators of 44–46. Exact purpose-set
and Jaccard comparisons require all four author judgments to be resolved,
leaving 43 episodes. Exact purpose-set agreement is 12/43 with the primary
annotation and 7/43 with the secondary annotation. Purpose-specific Cohen's
kappa ranges from 0.17 to 0.53 for the primary annotation and from 0.11 to
0.48 for the secondary annotation; complete counts and metrics are in
`metrics.json`. Precision and recall use the author's judgment as reference.

Status agreement excludes author judgments of insufficient context; the
completed audit contains none. These comparisons characterize the stratified
audit sample. A single author review leaves human intercoder reliability and
corpus-wide error unmeasured. The automated full-corpus labels and derived
purpose counts retain their original values.

The maintainer can regenerate the public export using
`scripts/export_human_audit.py` with the saved private annotation export,
study manifest, annotation instructions, and original coding directory.
Recomputing the released metrics needs only the public files above and the
frozen coding records.
