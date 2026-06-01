# Quark Stars

This module implements the Quark-Meson-Diquark (QMD) model and produces all
numerical results for Part 2 of the thesis.

## Pipeline

The calculation follows six steps, each with a corresponding subdirectory:

| Directory | Contents |
|-----------|----------|
| `sec1_qm_eos/` | QM model equation of state |
| `sec2_qm_stars/` | QM model mass-radius sequences |
| `sec4_qmd_stars/` | QMD baseline stellar sequence |
| `sec5_parameter_sensitivity/` | Parameter sweeps (m_Δ, g_Δ, λ_Δ, λ₃) |
| `sec6_observational/` | Comparison with observational constraints |

Core physics modules:

| File | Contents |
|------|----------|
| `qmd_parameters.py` | Model parameter sets |
| `qmd_stellar.py` | QMD EoS builder and stellar matter |
| `qmd_simple.py` | Simplified QMD utilities |
| `qm_potential.py` | QM vacuum and medium thermodynamics |
| `qm_stellar_matter.py` | Beta-equilibrated, charge-neutral EoS |
| `solvers/` | TOV integrator |
| `thermodynamics/` | Maxwell construction and related tools |

## Running

From the repository root:

```bash
./numerics/bin/python -m numerics.quark_stars.sec4_qmd_stars.<script>
./numerics/bin/python -m numerics.quark_stars.sec5_parameter_sensitivity.<script>
./numerics/bin/python -m numerics.quark_stars.sec6_observational.plot_observational_mr
```

Pre-computed output data are stored in `output/` and tracked in the repository
so that figures can be regenerated without re-running the full pipeline.
