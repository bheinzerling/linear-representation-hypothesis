# Manuscript

`paper.tex`, `refs.bib`, and the included caption files provide the complete
manuscript source. Figure assets are shared with the repository's `figures/`
directory. ACL template files retain their upstream notices.

Run from the repository root:

```bash
./paper/compile.sh
```

The build requires `make`, `pdflatex`, `bibtex`, and `texfot`. It writes the PDF,
LaTeX log, and nonempty SyncTeX mapping to `paper/out/`. All TeX passes enable
SyncTeX. The generated PDF should have no undefined citations or references.
