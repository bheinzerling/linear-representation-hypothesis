# Manuscript figures

Final PDF, PNG, and available SVG assets accompany the rendering code. Frozen
inputs are in `../data/figures/`. The generators use repository-relative paths.

On Debian or Ubuntu, the system dependencies are supplied by
`libcairo2`, `fontconfig`, `fonts-texgyre`, and `poppler-utils`. Poppler supplies
`pdftoppm` and `pdffonts` for previews and font checks. Confirm that
`fc-match 'TeX Gyre Heros'` finds the Heros font. Then run from the repository root:

```bash
uv sync --locked --group figures
uv run --group figures python scripts/render_figures.py
```

Individual commands:

| Command after `uv run --group figures python` | Output |
| --- | --- |
| `figures/render_main.py` | Figures 1–7 |
| `figures/render_components.py` | Appendix circular-component matrix |
| `figures/render_complexity.py` | Appendix complexity paths |
| `figures/render_pls_transfer.py` | Appendix PLS mean-alignment comparison |
| `figures/render_transfer.py` | Three appendix transfer displays |

The main renderer accepts `--figures 2 7` to select particular figures.
Shared modules define the analytic constructions, literature panels, and font
handling. The prism illustration is a fixed raster input; its scientific labels
and diagrams are drawn by the code. The figure code also checks frozen input
hashes and the numerical relationships used in the displays.
