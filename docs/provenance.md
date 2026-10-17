# Evidence and provenance

`data/MANIFEST.sha256` identifies every supplied data file. Numerical experiment
conditions are recorded in `data/numerical/reported_config.json`; the adjoining
input provenance records the observation units, upstream artifact hashes, and
reductions used for the numerical calculations.

Frozen display inputs in `data/figures/` retain the measurements and analytic
definitions used by the manuscript figures. Their local manifests identify the
files consumed by each renderer. Rendering records include the source hashes,
font identities, and library versions. The source and data needed to run these
calculations are included here.

Literature records include primary-source URLs, page references, rating
rationales, and coding definitions. These records expose the basis for the
interpretive judgments. Recomputing the counts and checking table consistency
establishes agreement between the stored records and their aggregations.

The completed author audit is released under `data/literature/human_audit/`.
Its case-level labels reproduce the sixty-episode counts in Appendix A and
the human-machine comparisons. Provenance records distinguish the original
private export from the public label files, retain hashes of the study inputs,
and separate the earlier thirty-case planning packet from the completed review.
The literature runner and repository verifier check the released labels;
the verifier also checks the manuscript's audit sample and table counts.

The MDL release preserves per-example coding costs and held-out predictions
from the completed 73-setting run. Its exporter verifies every original block
before packing the observations, retaining exact cost values and recording the
source inventory hash. The CPU workflow recalculates all block sums, online
codes and held-out errors. `data/probe_mdl/protocol.json` records the original
fit settings and environment; the portable fitting runner uses those settings
with the public numerical experiment sources.

The model-input artifact has an archive checksum and per-file checksums in
`data/manifests/model-inputs.json`. The extraction command verifies both. New
model runs use the separate experiment lockfiles; the frozen CPU observations
retain the values used by the paper even when reruns differ across hardware
or software environments.
