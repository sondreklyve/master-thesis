# Quark Stars

This directory is stripped to the two workflows used in the thesis.

## Pipelines

### 1. Simple QM model

Use `run_qm_simple.py`.

This is the pedagogical `T=0` two-flavor QM-model pipeline:

- no beta equilibrium
- no charge neutrality
- no electrons
- no TOV

It produces:

- combined `sigma_vs_mu_multi.pdf`
- combined `number_density_vs_mu_multi.pdf`
- combined `pressure_vs_mu_multi.pdf`
- combined `pressure_vs_energy_density_multi.pdf`
- combined `speed_of_sound_vs_mu_multi.pdf`

with the thesis baseline choices

- `m_sigma = 500, 600, 700 MeV`

Outputs go to:

- `output/simple/data/`
- `output/simple/plots/`

The simple data products now also include a combined
`qm_simple_speed_of_sound_multi.txt` table with the positive-pressure
branch diagnostic

- `c_s^2 = dP / d\varepsilon = (dP/d\mu_q)/(d\varepsilon/d\mu_q)`

### 2. Stellar quark matter

Use `run_qm_stellar_eos.py` for the constrained EoS only, and
`run_qm_stars.py` for the constrained EoS plus TOV.

This is the physical `T=0` baseline used for the quark-star part of the
chapter:

- beta equilibrium
- charge neutrality
- electrons from the equilibrium setup
- two-flavor QM model
- Andresen-style sigma scan and Maxwell construction
- `B0` vacuum normalization
- additional bag constant `B`
- TOV only after a genuine dense-matter `P=0` crossing exists

Outputs go to:

- `output/stellar/data/`
- `output/stellar/plots/`

## File Roles

- `constants.py`: physical constants and unit conversions
- `io.py`: table writing and output-directory helpers
- `plotting.py`: shared plotting style and save helpers
- `qm_parameters.py`: vacuum input container and parameter fit from
  `(m_q, m_pi, f_pi, m_sigma)`
- `qm_potential.py`: shared T=0 QM-model thermodynamics
- `qm_simple_eos.py`: simple unconstrained pipeline
- `qm_stellar_matter.py`: beta-equilibrated, charge-neutral stellar matter
- `bag_model.py`: `B0` and `B` logic
- `tov_interface.py`: bridge to the existing `npemu` TOV solver

## Sletmoen / Andresen Influence

The stellar path follows the same structure in spirit:

- `m_sigma`-centered scan
- vacuum parameters derived from `(m_q, m_pi, f_pi, m_sigma)`
- solve `mu_u` and `mu_d` along a sigma-mean-field scan
- keep the physical electron-present branch
- perform the Maxwell construction before the bag search
- beta equilibrium and charge neutrality solved explicitly
- separate treatment of `B0` and `B`
- additional bag shift fixed from the Andresen-style `epsilon / n_B`
  criterion on the quark branch at the true `P=0` surface

## Running

```bash
./numerics/.venv/bin/python -m numerics.quark_stars.run_qm_simple
./numerics/.venv/bin/python -m numerics.quark_stars.run_qm_stellar_eos --m-sigma-values 600
./numerics/.venv/bin/python -m numerics.quark_stars.run_qm_stars --m-sigma-values 600
```
