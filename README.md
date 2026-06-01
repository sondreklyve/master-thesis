# Color-Superconducting Quark Matter in Compact Stars

**Mass-Radius Sequences and Parameter Sensitivity in the Quark-Meson-Diquark Model**

Master's thesis by Sondre Klyve, NTNU (2026).

This repository contains the full LaTeX source and all numerical code used to
produce the results and figures in the thesis.

## Structure

```
thesis/      LaTeX source, figures, and bibliography
numerics/    Python code for all numerical calculations
Makefile     Builds the thesis PDF via latexmk
```

## Thesis overview

**Part 1 – Neutron Stars** (background): TOV equations, ideal Fermi gas, and
a relativistic mean-field model for $npe\mu$ matter.

**Part 2 – Quark Stars** (main contribution): Quark-meson (QM) model, extension
to the quark-meson-diquark (QMD) model with 2SC color-superconducting pairing,
mass-radius sequences, parameter sensitivity analysis, and comparison with
observational constraints (NICER, radio timing).

## Building the thesis

```bash
make pdf
```

Requires a TeX Live installation with `latexmk`. A compiled PDF is also
produced automatically on each push via GitHub Actions.

## Numerical code

See [`numerics/`](numerics/) for setup instructions and a description of each
module. All figures in the thesis are generated from the Python code in that
directory.
