# Repository instructions

Use the project-local uv environment. Run Python commands with `uv run` and
retain the separate lockfiles under `experiments/`. Use the root environment
for CPU calculations and figure generation. Follow `envs/environments.json`
when selecting an environment.

Describe the paper through its seven empirical claims and the evidence for
each. Keep probing distinct from intervention tests. Use decodability,
accessibility, and probing precisely. Name intervention directions, subspaces,
and coefficients directly. State the operation and measured result of
target-informed or output-scored interventions. Use plain scientific prose,
attach citations to the claims they support, and preserve consequential design
choices and limitations.

Keep this repository self-contained. Use relative paths, regular files, and
declared dependencies. Put generated calculations under `out/` and larger
distribution artifacts under `artifacts/`. Keep frozen observations immutable;
document deliberate data updates and refresh their manifest.

Verify changes with `scripts/verify.py`, the relevant reproduction command, and
the focused tests. Compile manuscript changes with `./paper/compile.sh`; every
TeX pass must enable SyncTeX. Check the final log and complete PDF before using
the build output. Do not alter vendored template files.

When testing source ZIPs, extract with `unzip` so recorded executable modes are
restored. Check that `paper/compile.sh` is executable before invoking it;
Python's `ZipFile.extractall` does not restore those modes.

This repository is public. Follow the user's publication instructions before
committing, pushing, tagging, or uploading release artifacts.
