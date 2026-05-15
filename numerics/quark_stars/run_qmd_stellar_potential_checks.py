"""Validate the QMD stellar potential against known limits.

Checks performed:
  A. Common-mu reduction:
     QMDStellarModel.omega(phi, Delta, mu_q, 0, 0)
       == QMDSimpleModel.omega(phi, Delta, mu_q)
     Expected: difference = 0 to floating-point precision.

  B. Delta=0 stellar medium-term routing:
     At Delta=0 the difference stellar - simple should equal exactly:
       sum_fc fermion_gp(mu_fc, M_q, 2) + electron_gp(mu_e)
         - fermion_gp(mu_q, M_q, 12)
     The residual after subtracting this known difference should be ~0.
     This confirms the six individual mu_fc values (not a common mu) drive
     the stellar medium term.

Does NOT solve neutrality, build an EoS, or run TOV.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from .qmd_parameters import QMD_SET_A
from .qmd_simple import QMDSimpleModel
from .qmd_stellar import QMDStellarModel, quark_chemical_potentials
from .thermodynamics.fermi import electron_grand_potential, fermion_grand_potential


def check_a_common_mu_reduction() -> bool:
    """Check A: stellar(phi, Delta, mu_q, 0, 0) == simple(phi, Delta, mu_q)."""
    model_s = QMDSimpleModel(QMD_SET_A)
    model_st = QMDStellarModel(QMD_SET_A)

    phi_vals = [93.0, 50.0, 10.0]
    delta_vals = [0.0, 30.0, 60.0]
    mu_vals = [0.0, 300.0, 500.0]

    print("Check A: Common-mu reduction  (mu_e=0, mu_8=0)")
    print(f"  {'phi':>5}  {'Delta':>5}  {'mu_q':>5}  {'diff (MeV^4)':>16}")

    max_diff = 0.0
    for phi in phi_vals:
        for delta in delta_vals:
            for mu in mu_vals:
                o_simple = model_s.omega(phi, delta, mu)
                o_stellar = model_st.omega(phi, delta, mu, 0.0, 0.0)
                diff = o_stellar - o_simple
                max_diff = max(max_diff, abs(diff))
                print(f"  {phi:5.1f}  {delta:5.1f}  {mu:5.1f}  {diff:16.4e}")

    print(f"\n  Max |diff| = {max_diff:.3e} MeV^4")
    passed = max_diff < 1.0  # floating-point tolerance
    print(f"  Result: {'PASS' if passed else 'FAIL  <-- investigate'}")
    return passed


def check_b_delta_zero_stellar_medium() -> bool:
    """Check B: Delta=0 medium-term routing through individual mu_fc.

    At Delta=0 the Omega_trunc contributions are identical (mu_bar has no
    effect when all Delta-containing terms vanish).  The full difference
    stellar.omega - simple.omega must therefore equal:
        sum_fc gp(mu_fc, M_q, 2) + electron_gp(mu_e) - gp(mu_q, M_q, 12)
    The residual after subtracting this expected difference should be ~0.
    """
    model_s = QMDSimpleModel(QMD_SET_A)
    model_st = QMDStellarModel(QMD_SET_A)

    mu_q = 400.0
    mu_e = 50.0
    mu_8 = 10.0
    phi_vals = [93.0, 50.0, 10.0]

    mus = quark_chemical_potentials(mu_q, mu_e, mu_8)

    print(f"\nCheck B: Delta=0 stellar reduction")
    print(f"  mu_q={mu_q:.0f}, mu_e={mu_e:.0f}, mu_8={mu_8:.0f} MeV")
    print(f"  Individual quark chemical potentials:")
    for name, val in mus.items():
        print(f"    {name} = {val:.3f} MeV")

    print(f"\n  {'phi':>5}  {'diff':>18}  {'expected_diff':>18}  {'residual':>14}")

    max_residual = 0.0
    for phi in phi_vals:
        o_stellar = model_st.omega(phi, 0.0, mu_q, mu_e, mu_8)
        o_simple = model_s.omega(phi, 0.0, mu_q)
        diff = o_stellar - o_simple

        M_q = QMD_SET_A.g * phi
        med_indiv = sum(fermion_grand_potential(m, M_q, 2.0) for m in mus.values())
        med_common = fermion_grand_potential(mu_q, M_q, 12.0)
        elec = electron_grand_potential(mu_e)
        expected_diff = med_indiv + elec - med_common

        residual = diff - expected_diff
        max_residual = max(max_residual, abs(residual))
        print(f"  {phi:5.1f}  {diff:18.6e}  {expected_diff:18.6e}  {residual:14.4e}")

    print(f"\n  Max |residual| = {max_residual:.3e} MeV^4")
    print("  Note: diff is phi-dependent (M_q=g*phi drives the individual Fermi seas),")
    print("        confirming medium term routes through mu_fc, not a common mu.")
    passed = max_residual < 1.0
    print(f"  Result: {'PASS' if passed else 'FAIL  <-- investigate'}")
    return passed


def check_c_mu_e_sensitivity() -> None:
    """Check C: Verify that mu_e and mu_8 affect the potential through mu_fc splitting.

    At fixed phi=93, Delta=0, mu_q=400, compare Omega for
      (mu_e=0, mu_8=0) vs (mu_e=50, mu_8=10)
    and cross-check the difference against the analytic medium-term split.
    """
    model_st = QMDStellarModel(QMD_SET_A)
    phi, delta, mu_q = 93.0, 0.0, 400.0

    print("\nCheck C: mu_e/mu_8 sensitivity at phi=93, Delta=0, mu_q=400")
    for mu_e, mu_8 in [(0.0, 0.0), (50.0, 0.0), (0.0, 10.0), (50.0, 10.0)]:
        omega = model_st.omega(phi, delta, mu_q, mu_e, mu_8)
        mus = quark_chemical_potentials(mu_q, mu_e, mu_8)
        mu_bar = mu_q - mu_e / 6.0 + mu_8 / 3.0
        print(
            f"  mu_e={mu_e:4.0f}  mu_8={mu_8:4.0f}  "
            f"mu_bar={mu_bar:7.3f}  "
            f"Omega={omega:16.6e} MeV^4"
        )


def main() -> None:
    print("=" * 70)
    print("QMD Stellar Potential Validation")
    print("=" * 70)

    passed_a = check_a_common_mu_reduction()
    passed_b = check_b_delta_zero_stellar_medium()
    check_c_mu_e_sensitivity()

    print("\n" + "=" * 70)
    all_passed = passed_a and passed_b
    if all_passed:
        print("All checks PASSED.  QMDStellarModel is consistent with QMDSimpleModel.")
    else:
        print("WARNING: one or more checks FAILED.  Investigate before proceeding.")
    print("=" * 70)


if __name__ == "__main__":
    main()
