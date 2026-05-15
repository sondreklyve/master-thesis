"""Validate QMD stellar neutrality residuals and fixed-field neutrality solver.

Checks:
  A. Delta=0 analytic residual comparison:
     General analytic dOmega/dmu_e and dOmega/dmu_8 are compared against the
     simpler Delta=0 expressions built from individual quark number densities.

  B. Fixed-field neutrality solutions at Delta=0:
     Solve (mu_e, mu_8) for representative (phi, mu_q) at Delta=0.
     Expect: mu_8 ≈ 0 (color symmetry unbroken in normal phase),
             mu_e > 0 (charge neutrality requires positive electron chemical potential).

  C. Fixed-field neutrality solutions at Delta>0 (2SC phase):
     Solve (mu_e, mu_8) for representative (phi, mu_q, Delta).
     Expect: mu_8 may become nonzero; residual norm should be small.

Does NOT solve for (phi, Delta) equilibrium, build an EoS, or run TOV.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from .qmd_parameters import QMD_SET_A
from .qmd_stellar import (
    NeutralityResiduals,
    NeutralitySolution,
    QMDStellarModel,
    quark_chemical_potentials,
)
from .thermodynamics.fermi import (
    electron_number_density,
    fermion_number_density,
)

_STEP_MEV = 0.1  # deprecated compatibility argument for residual calls


def analytic_residuals_delta0(
    phi_mev: float,
    mu_q_mev: float,
    mu_e_mev: float,
    mu_8_mev: float,
) -> tuple[float, float]:
    """Analytic dOmega/dmu_e and dOmega/dmu_8 at Delta=0.

    At Delta=0 the Omega_trunc terms involving Delta all vanish, so
    mu_bar has no effect on Omega.  The derivatives come entirely from
    the individual Fermi-sea medium terms and the electron:

        dOmega/dmu_e = 2/3*(n_ur+n_ug+n_ub) - 1/3*(n_dr+n_dg+n_db) - n_e
        dOmega/dmu_8 = -1/3*(n_ur+n_ug+n_dr+n_dg) + 2/3*(n_ub+n_db)

    where n_fc = -dOmega_medium/dmu_fc = fermion_number_density(mu_fc, M_q, 2).
    """
    M_q = QMD_SET_A.g * phi_mev
    mus = quark_chemical_potentials(mu_q_mev, mu_e_mev, mu_8_mev)

    n = {fc: fermion_number_density(m, M_q, degeneracy=2.0) for fc, m in mus.items()}
    n_e = electron_number_density(mu_e_mev)

    r_e = (
        2.0 / 3.0 * (n["mu_ur"] + n["mu_ug"] + n["mu_ub"])
        - 1.0 / 3.0 * (n["mu_dr"] + n["mu_dg"] + n["mu_db"])
        - n_e
    )
    r_8 = (
        -1.0 / 3.0 * (n["mu_ur"] + n["mu_ug"] + n["mu_dr"] + n["mu_dg"])
        + 2.0 / 3.0 * (n["mu_ub"] + n["mu_db"])
    )
    return r_e, r_8


def check_a_analytic_residuals() -> bool:
    """Check A: general residuals match Delta=0 analytic expressions."""
    model = QMDStellarModel(QMD_SET_A)

    mu_q = 400.0
    mu_e = 50.0
    mu_8 = 10.0
    phi_vals = [93.0, 50.0, 10.0]

    print("Check A: Delta=0 analytic residual comparison")
    print(f"  mu_q={mu_q:.0f} MeV, mu_e={mu_e:.0f} MeV, mu_8={mu_8:.0f} MeV, Delta=0")
    print(
        f"  {'phi':>5}  "
        f"{'resid_e':>16}  {'ana_R_e':>16}  {'diff_e':>12}  "
        f"{'resid_8':>16}  {'ana_R_8':>16}  {'diff_8':>12}"
    )

    max_diff = 0.0
    for phi in phi_vals:
        nr = model.neutrality_residuals(phi, 0.0, mu_q, mu_e, mu_8, step_mev=_STEP_MEV)
        ar_e, ar_8 = analytic_residuals_delta0(phi, mu_q, mu_e, mu_8)
        diff_e = nr.d_omega_d_mu_e - ar_e
        diff_8 = nr.d_omega_d_mu_8 - ar_8
        max_diff = max(max_diff, abs(diff_e), abs(diff_8))
        print(
            f"  {phi:5.1f}  "
            f"{nr.d_omega_d_mu_e:16.4e}  {ar_e:16.4e}  {diff_e:12.3e}  "
            f"{nr.d_omega_d_mu_8:16.4e}  {ar_8:16.4e}  {diff_8:12.3e}"
        )

    tol = 1.0  # MeV^3; ~1e-6 relative for typical ~1e5 MeV^3 densities
    passed = max_diff < tol
    print(f"\n  Max |diff| = {max_diff:.3e} MeV^3  (tol={tol} MeV^3)")
    print(f"  Result: {'PASS' if passed else 'FAIL  <-- investigate'}")
    return passed


def check_b_neutrality_delta0() -> None:
    """Check B: solve neutrality at Delta=0 for several (phi, mu_q)."""
    model = QMDStellarModel(QMD_SET_A)

    phi_vals = [93.0, 50.0, 10.0]
    mu_q_vals = [400.0, 500.0, 600.0]

    print("\nCheck B: Fixed-field neutrality solutions at Delta=0")
    print(
        f"  {'mu_q':>5}  {'phi':>5}  {'Delta':>5}  "
        f"{'mu_e':>9}  {'mu_8':>9}  "
        f"{'|resid|':>10}  {'ok':>4}"
    )

    for mu_q in mu_q_vals:
        for phi in phi_vals:
            sol = model.solve_neutrality_fixed_fields(
                phi, 0.0, mu_q,
                initial_guess=(30.0, 0.0),
                step_mev=_STEP_MEV,
            )
            ok = "Y" if sol.success and sol.residual_norm < 1.0 else "N"
            print(
                f"  {mu_q:5.0f}  {phi:5.1f}  {0.0:5.1f}  "
                f"{sol.mu_e_mev:9.4f}  {sol.mu_8_mev:9.4f}  "
                f"{sol.residual_norm:10.3e}  {ok:>4}"
            )

    print("\n  Expected: mu_8 ≈ 0 (no color breaking at Delta=0),")
    print("            mu_e > 0 (charge neutrality requires positive electron mu)")


def check_c_neutrality_nonzero_delta() -> None:
    """Check C: solve neutrality at non-zero Delta (2SC phase)."""
    model = QMDStellarModel(QMD_SET_A)

    cases = [
        (400.0, 50.0, 20.0),
        (400.0, 50.0, 50.0),
        (400.0, 10.0, 20.0),
        (400.0, 10.0, 50.0),
        (500.0, 50.0, 20.0),
        (500.0, 50.0, 50.0),
        (600.0, 10.0, 20.0),
        (600.0, 10.0, 50.0),
    ]

    print("\nCheck C: Fixed-field neutrality at nonzero Delta (2SC phase)")
    print(
        f"  {'mu_q':>5}  {'phi':>5}  {'Delta':>5}  "
        f"{'mu_e':>9}  {'mu_8':>9}  "
        f"{'|resid|':>10}  {'ok':>4}"
    )

    for mu_q, phi, delta in cases:
        sol = model.solve_neutrality_fixed_fields(
            phi, delta, mu_q,
            initial_guess=(30.0, 0.0),
            step_mev=_STEP_MEV,
        )
        ok = "Y" if sol.success and sol.residual_norm < 1.0 else "N"
        print(
            f"  {mu_q:5.0f}  {phi:5.1f}  {delta:5.1f}  "
            f"{sol.mu_e_mev:9.4f}  {sol.mu_8_mev:9.4f}  "
            f"{sol.residual_norm:10.3e}  {ok:>4}"
        )

    print("\n  Expected: mu_8 may become nonzero in the 2SC phase.")
    print("  The residuals include the finite Omega_1_num BCS integral when enabled.")


def main() -> None:
    print("=" * 80)
    print("QMD Stellar Neutrality Validation")
    mode = "integral" if QMD_SET_A.include_omega_1_num else "disabled"
    print(f"  Parameter set: QMD_SET_A  (t_loop4_factor=4, Omega_1_num={mode})")
    print("  Neutrality residuals: analytic derivatives")
    print("=" * 80)

    passed_a = check_a_analytic_residuals()
    check_b_neutrality_delta0()
    check_c_neutrality_nonzero_delta()

    print("\n" + "=" * 80)
    if passed_a:
        print("Analytic residual check PASSED.")
    else:
        print("WARNING: analytic residual check FAILED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
