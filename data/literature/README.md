# Literature evidence

`evidence_studies.json` describes the thirty reviewed studies and their candidate
representations. `evidence_ratings.csv` contains 240 ratings with rationales,
primary-source PDF URLs, and page references. `evidence_criteria.json` defines
the tests used to assess the seven claims; causal and mechanistic sufficiency
are recorded separately.

The `coding/` directory contains the codebook, structured annotation schema,
episode locations, primary annotations, and paired coding records. These
records support the paper-purpose counts and coding diagnostics.
`paper_versions.csv` records document-version pairs used in aggregation.

Run the consistency checks with:

```bash
uv run python -m lrh.analysis.literature --data-dir data/literature --output-dir out/literature
```

The runner recomputes counts from 376 coded episodes, agreement over sixty
duplicated episodes, and consistency between the study records and evidence
ratings. It also reproduces the completed author audit of those sixty episodes
from the [released judgments and protocol](human_audit/README.md), including
episode-status and purpose-level comparisons with both coding agents.
The author classified 47 episodes as substantive, twelve as bibliography
entries, and one as a structural fragment. Status agreement with each agent
is 58/60. These comparisons describe the selected audit sample; human
intercoder reliability and corpus-wide error are unmeasured.
The evidence ratings are interpretive judgments assessed against
primary sources; computational consistency checks do not independently
establish those judgments.

Original annotations use CC BY 4.0. Quotations retain their source terms.
