# Numerical Code

This directory contains the Python code used to generate all numerical results
and figures in the thesis. The code is written to support the analysis in the
main text and is not intended as a general library.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run any module from the repository root using the repo-local runner:

```bash
./numerics/bin/python -m numerics.<module>.<script>
```

The runner uses `.venv/bin/python` when available, sets a non-interactive
Matplotlib backend, and adds the repository root to `PYTHONPATH`.

## Directory overview

```
numerics/
├── ideal_fermi_gas/         Relativistic zero-temperature Fermi gas
├── incompressible_star/     Analytic incompressible star (TOV consistency check)
├── ideal_neutron_stars/     TOV solutions for idealised neutron-star models
├── npemu/                   npeμ relativistic mean-field model and TOV solver
├── qft/                     QFT utilities (loop integrals, renormalization)
├── quark_meson/             Quark-meson (QM) model thermodynamics
└── quark_stars/             QMD model, stellar sequences, and observational comparison
```

The first four modules support Part 1 of the thesis (neutron stars). The last
three are the main numerical contribution of Part 2 (quark stars).

## Code provenance

Parts of the code are adapted from prior work:

- The $npe\mu$ relativistic mean-field implementation is based on code by
  **Pogliano** (Master's thesis):
  [hdl.handle.net/11250/2445966](https://hdl.handle.net/11250/2445966)

- The radial stability solver and parts of the ideal neutron-star code are
  adapted from work by **Sletmoen** (Master's thesis):
  [hdl.handle.net/11250/3031031](https://hdl.handle.net/11250/3031031),
  with an accompanying public repository:
  [hersle/master-thesis (GitHub)](https://github.com/hersle/master-thesis/)

- Observational equation-of-state bands are taken from published datasets
  (e.g. Ng *et al.*) and processed locally for use in the plots.
