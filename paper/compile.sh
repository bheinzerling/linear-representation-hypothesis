#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
make --always-make all

rerun='Rerun to get|rerunfilecheck Warning|There were undefined|Citation.*undefined|Reference.*undefined'
for pass in 1 2 3; do
    if ! grep -Eq "$rerun" out/paper.log; then
        break
    fi
    texfot --quiet pdflatex -halt-on-error -synctex=1 -output-directory out paper.tex
done
if grep -Eq "$rerun" out/paper.log; then
    echo 'LaTeX references did not converge; inspect paper/out/paper.log.' >&2
    exit 1
fi
test -s out/paper.synctex.gz
tail -c 32 out/paper.pdf | grep -q '%%EOF'
