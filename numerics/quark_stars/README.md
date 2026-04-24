# Quark Stars

This module now follows one physics-driven pipeline:

1. Vacuum consistency scan
2. Selection of the lowest valid `m_sigma` values
3. EoS construction with and without Maxwell construction
4. Bag scan in `B^(1/4)` around `B_min`
5. TOV mass-radius sequences

## Workflows

Run:

```bash
./numerics/.venv/bin/python -m numerics.quark_stars.run_qm_vacuum_scan
./numerics/.venv/bin/python -m numerics.quark_stars.run_qm_eos_simple
./numerics/.venv/bin/python -m numerics.quark_stars.run_qm_stars
```

Outputs go to:

- `output/vacuum/`
- `output/eos/`
- `output/stellar/`

## File Roles

- `qm_potential.py`: core vacuum and medium thermodynamics
- `vacuum_scan.py`: vacuum-consistency scan and `m_sigma` selection
- `qm_stellar_matter.py`: clean beta-equilibrated, charge-neutral EoS builder
- `bag_model.py`: vacuum subtraction, `B_min`, and `B^(1/4)` conversion helpers
- `tov_interface.py`: bridge from the QM EoS to the existing TOV solver

The old pedagogical/simple pipeline, bag-scale multipliers, and mixed-summary plotting code were removed.
