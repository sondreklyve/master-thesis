# Numerical Code

This directory contains the Python code used to generate all numerical
results and figures appearing in the project. The code is written
to support the analysis in the main text and is not intended as a general
library.

The numerical work focuses on neutron-star structure in general
relativity, using the Tolman–Oppenheimer–Volkoff equations together with
various equations of state.

## Directory Overview

```
numerics/
├── data/ # Precomputed numerical datasets
│
├── ideal_fermi_gas/ # Relativistic zero-temperature Fermi gas
│
├── incompressible_star/ # Analytic incompressible star solution
│
├── ideal_neutron_stars/ # Ideal neutron-star models and stability
│
└── npemu/ # npeμ matter, realistic EoS, and TOV solver
```

## Codex Sandbox Runner

Use the repo-local runner when executing numerical Python from Codex:

```bash
./numerics/bin/python -m numerics.quark_stars.run_qm_vacuum_scan
```

It uses `numerics/.venv/bin/python` when available, falls back to
`python3`, sets a non-interactive Matplotlib backend, adds the repository
root to `PYTHONPATH`, and keeps runtime caches in `numerics/.sandbox/`.
That lets Codex run checks in workspace sandbox mode without writing
Python or plotting caches outside this repository.

If the environment has not been created yet:

```bash
python3 -m venv numerics/.venv
./numerics/bin/python -m pip install -r numerics/requirements.txt
```


## Contents

- **ideal_fermi_gas/**  
  Illustrates basic properties of a relativistic Fermi gas and its
  equation of state.

- **incompressible_star/**  
  Analytic solution of the TOV equations for a uniform-density star,
  used as a consistency check and pedagogical example.

- **ideal_neutron_stars/**  
  Numerical solutions of the TOV equations for idealized neutron-star
  models, including radial oscillations and stability analysis.

- **npemu/**  
  The main numerical component of the project. Implements
  beta-equilibrated $npe\mu$ matter using a relativistic mean-field
  model, constructs composite equations of state, solves the TOV
  equations, and compares the results with observational constraints.

## Code Provenance

Parts of the numerical code are adapted from existing academic work:

- The $npe\mu$ relativistic mean-field implementation is based on code
  developed by **Pogliano** (Master’s thesis), available at:  
  [Pogliano, *Master’s thesis*](https://hdl.handle.net/11250/2445966)

- The radial stability solver and parts of the ideal neutron-star code
  are adapted from work by **Sletmoen** (Master’s thesis), available at:  
  [Sletmoen, *Master’s thesis*](https://hdl.handle.net/11250/3031031),  
  with an accompanying public repository:  
  [hersle/master-thesis (GitHub)](https://github.com/hersle/master-thesis/)

- Observational equation-of-state bands are taken from published data
  sets (e.g. Ng *et al.*) and processed locally for use in the plots.
