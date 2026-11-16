# Figure 1: preserved retrieval evidence

The figure is reproducible from `fulltext_comparison_counts.csv`. Recomputing
this plot verifies the frozen observations, not the execution of historical
database searches. `retrieval_ledger.json` separates recovered evidence from
unavailable request details. It was assembled on 16 November 2026; that date
is not a retrieval date for the plotted counts.

The packet contains:

- `source_evidence/annual_count_records.csv`: the original dated table with
  measurement descriptions and source URLs, preserved byte for byte.
- `screening_candidates.csv`: all 567 title groups, their OpenAlex memberships,
  Semantic Scholar identities and citation counts, and full-text inclusion.
- `source_evidence/collector_run.json` and `collector_excerpt.txt`: the archived
  collector-run metadata and relevant surviving query/deduplication code.

## Annual observations

The December 2025 Scholar estimates were transcribed from the survey. They
are also printed in Table 1 of its
[published version](https://aclanthology.org/2026.blackboxnlp-1.48/).
The July Scholar and OpenAlex observations are dated 31 July 2026 in the
original table. Its OpenAlex URL specifies
`search.exact="linear representation hypothesis"&group_by=publication_year`
across all fields. The five displayed OpenAlex annual counts sum to 866.

No raw July response, Scholar screenshot, or exact access timestamp was
recovered. Scholar's year-filter URLs and locale are unavailable. The OpenAlex
URL is evidence of the recorded query, not a complete HTTP request log.
Unknown fields are explicit nulls in the ledger. The original count table
describes the Scholar observations as a low-rate pilot stopped after the
access policy was checked; the released reproduction code does not retrieve
Scholar pages.

## Screening denominator

The screening pool comes from the separate collector run of 30 July 2026,
09:25:24–09:29:49 UTC. These timestamps bound the whole collector run; they are
not individual OpenAlex request timestamps. Its archived metadata and surviving
code identify `fulltext.search:"linear representation hypothesis"` restricted
to `primary_topic.field.id:17` (Computer Science).

The collector records 651 retrieved rows. Preserved memberships contain 649
distinct OpenAlex IDs; two IDs occur twice within their own groups. Grouping
normalized titles yields 567 candidates. Of these, 519 resolved to Semantic
Scholar, 311 had a positive citation count, and 206 were retained after PDF
acquisition and main-body phrase screening. These totals are recomputed from
the released candidate table. The 866 annual results and 567 candidates have
different query dates, search expressions, and field restrictions, and the
candidate pool undergoes title deduplication. No equality or direct set-subset
relationship between these two saved outputs has been verified.

Title normalization applies NFKC, collapses whitespace, casefolds, replaces
runs outside ASCII letters/digits with spaces, and strips outer spaces. The
collector merges identical normalized titles, retains the first ID and
earliest nonzero publication year, and concatenates ID memberships. The
fallback uses arXiv ID, DOI, then OpenAlex ID when a normalized title is empty.
Two records whose titles contain no ASCII letters or digits use distinct DOI
keys; the released table preserves both original candidates. The
candidate packet preserves the grouped metadata; complete pre-deduplication
responses and individual source-record years are unavailable.

The collector-run record's source hash does not match the surviving collector
or either recovered Git version. `collector_excerpt.txt` therefore documents
surviving code, with its source hash; it is not the exact historical executable.
The historical hash and this limitation are preserved in the ledger.

## Offline acceptance check

In the public repository, run:

```bash
uv run python -m lrh.analysis.retrieval_provenance --data-dir data/mentions
```

In an extracted ARR package, run from `code/`:

```bash
uv run python shared/retrieval_provenance.py --data-dir ../data/01_introduction
```

The check validates evidence hashes, all annual observations, candidate
identities, memberships, and screening transitions. The Figure 1 runner also
performs this check and includes its limits in `metrics.json`. Passing does
not independently verify historical retrieval; this remains false in the
machine-readable result.

For future retrievals, retain exact parameters and UTC request times together
with lawful response metadata or permitted manual page captures and hashes.
Record any newly obtained results under their actual retrieval date. A current
database response cannot validate a historical total.
