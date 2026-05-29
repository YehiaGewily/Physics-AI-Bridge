# Phase 2 Paper - LaTeX Project

Part of the Physics-AI Bridge research project.

## Authors

- Yehia Said Gewily (yehia@nover.studio)
- Omar Hosney Mahmoud (omar@nover.studio)

## Build Instructions

Requires: TeX Live or MiKTeX with `pdflatex`, `bibtex`, and standard
academic packages.

To compile:

```
cd docs/phase2_project
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The double `pdflatex` run resolves cross-references and citations.

Output: `main.pdf`

## Contents

- `main.tex` - Paper source
- `references.bib` - Bibliography
- `figures/` - All paper figures (PNG, 300 DPI)
- `data/` - Raw experiment data (CSVs and summary)

## Related Documents

- Phase 1 paper: `../Computational_2D_Ising_Simulation_Phase1.pdf`
- Project root README: `../../README.md`
