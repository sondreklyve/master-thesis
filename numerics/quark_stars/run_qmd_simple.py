"""Scan the QMD mean-field potential over mu_q for SET_A and SET_B.

Produces:
  output/qmd_simple/data/qmd_simple_set_a.txt
  output/qmd_simple/data/qmd_simple_set_b.txt
  output/qmd_simple/plots/qmd_simple_condensates.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .io import output_directories, save_table
from .plotting import apply_plot_style, save_figure
from .qmd_parameters import (
    QMD_SET_A, QMD_SET_A_REFERENCE, QMD_SET_B,
    QMDParameters, print_convention_summary,
)
from .qmd_simple import QMDSimpleModel, QMDSimpleState


MU_MIN_MEV = 0.0
MU_MAX_MEV = 900.0
NUM_POINTS = 2000
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PARAMETER_SETS = [
    ("set_a", QMD_SET_A),
    ("set_b", QMD_SET_B),
]

# House style: viridis colors for SET A (index 0) and SET B (index 1)
SET_COLORS = plt.cm.viridis(np.linspace(0.15, 0.6, 2))


def scan_qmd(
    model: QMDSimpleModel,
    mu_values_mev: np.ndarray,
) -> list[QMDSimpleState]:
    """Scan mean-field minimum over mu_q with warm-start continuation."""
    states: list[QMDSimpleState] = []
    prev_guess: tuple[float, float] | None = None
    for mu in mu_values_mev:
        state = model.solve_mean_fields(float(mu), initial_guess=prev_guess)
        states.append(state)
        prev_guess = (state.phi_mev, state.delta_mev)
    return states


def states_to_array(states: list[QMDSimpleState]) -> tuple[list[str], np.ndarray]:
    columns = ["mu_q_mev", "phi_mev", "delta_mev", "gap_mev", "omega_min_mev4", "success"]
    data = np.array(
        [
            [s.mu_q_mev, s.phi_mev, s.delta_mev, s.gap_mev, s.omega_min_mev4, float(s.success)]
            for s in states
        ]
    )
    return columns, data


def save_condensate_plot(
    states_a: list[QMDSimpleState],
    states_b: list[QMDSimpleState],
    plots_dir: Path,
) -> None:
    mu_a = np.array([s.mu_q_mev for s in states_a])
    phi_a = np.array([s.phi_mev for s in states_a])
    gap_a = np.array([s.gap_mev for s in states_a])

    mu_b = np.array([s.mu_q_mev for s in states_b])
    phi_b = np.array([s.phi_mev for s in states_b])
    gap_b = np.array([s.gap_mev for s in states_b])

    color_a = SET_COLORS[0]
    color_b = SET_COLORS[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax1.plot(mu_a, phi_a, linewidth=2.2, color=color_a, label="SET A")
    ax1.plot(mu_b, phi_b, linewidth=2.2, color=color_b, label="SET B")
    ax1.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax1.set_ylabel(r"$\phi\;(\mathrm{MeV})$")
    ax1.set_title("Chiral condensate")
    ax1.set_xlim(200.0, 600.0)
    ax1.legend()

    ax2.plot(mu_a, gap_a, linewidth=2.2, color=color_a, label="SET A")
    ax2.plot(mu_b, gap_b, linewidth=2.2, color=color_b, label="SET B")
    ax2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax2.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")
    ax2.set_title("Diquark gap")
    ax2.set_xlim(200.0, 600.0)
    ax2.legend()

    fig.tight_layout()
    save_figure(plots_dir / "qmd_simple_condensates.pdf")


def delta_zero_reduction_check(
    params: QMDParameters,
    phi_test_values_mev: tuple[float, ...] = (93.0, 50.0, 10.0),
    mu_test_values: tuple[float, ...] = (0.0, 250.0, 350.0, 450.0, 600.0),
) -> list[dict[str, float]]:
    """Compare QMD and QM grand potentials at Delta=0 for multiple (phi, mu) values.

    Both models are evaluated at the SAME field value phi.  At Delta=0, the
    QMD medium term must reduce to the ordinary free Fermi sea for all colors,
    so the difference should be mu-independent for each phi (a vacuum-level
    constant only).
    """
    from .qm_parameters import QMVacuumInputs, fit_qm_parameters
    from .qm_potential import TwoFlavorQMPotential

    vacuum = QMVacuumInputs(
        m_q_mev=params.m_q_mev,
        m_pi_mev=params.m_pi_mev,
        f_pi_mev=params.f_pi_mev,
        m_sigma_mev=params.m_sigma_mev,
    )
    qm_pot = TwoFlavorQMPotential(fit_qm_parameters(vacuum))
    qmd_model = QMDSimpleModel(params)

    rows = []
    for phi in phi_test_values_mev:
        for mu in mu_test_values:
            omega_qm = qm_pot.grand_potential_simple(phi, mu)
            omega_qmd = qmd_model.omega(phi, 0.0, mu)
            rows.append({
                "phi": phi,
                "mu_q": mu,
                "omega_qm": omega_qm,
                "omega_qmd": omega_qmd,
                "difference": omega_qmd - omega_qm,
            })
    return rows


def convention_scan_check() -> None:
    """Short scan comparing thesis-convention (SET_A) vs reference-convention (SET_A_REFERENCE).

    Prints 2SC onset mu and max gap for both sets to expose the effect of
    t_loop4_factor=4 vs t_loop4_factor=8.
    """
    mu_check = np.linspace(200.0, 700.0, 100)
    print("\n--- Convention comparison: SET_A (t_loop4=4) vs SET_A_REFERENCE (t_loop4=8) ---")
    for label, params in [("SET_A (thesis, t_loop4=4)", QMD_SET_A),
                           ("SET_A_REFERENCE (ref, t_loop4=8)", QMD_SET_A_REFERENCE)]:
        model = QMDSimpleModel(params)
        states = [model.solve_mean_fields(float(mu)) for mu in mu_check]
        onset = next((s.mu_q_mev for s in states if s.phase == "2SC"), None)
        gaps = [s.gap_mev for s in states if s.phase == "2SC"]
        max_gap = max(gaps) if gaps else float("nan")
        print(f"  {label}")
        if onset is not None:
            print(f"    2SC onset ≈ {onset:.1f} MeV,  max gap ≈ {max_gap:.1f} MeV")
        else:
            print("    No 2SC condensation found.")
    print()


def main() -> None:
    print_convention_summary()
    apply_plot_style()
    data_dir, plots_dir = output_directories(OUTPUT_DIR, "qmd_simple")
    mu_values = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)

    all_states: dict[str, list[QMDSimpleState]] = {}

    for tag, params in PARAMETER_SETS:
        print(f"\nScanning QMD {tag.upper()}  "
              f"(m_delta={params.m_delta_mev:.0f} MeV, "
              f"g_delta_factor={params.g_delta_factor:.2f}) ...")

        model = QMDSimpleModel(params)
        states = scan_qmd(model, mu_values)
        all_states[tag] = states

        columns, data = states_to_array(states)
        metadata = {
            "model": "QMD: analytic potential + T=0 2SC quasiparticle medium",
            "set": tag,
            "m_delta_mev": f"{params.m_delta_mev:.1f}",
            "g_delta_factor": f"{params.g_delta_factor:.4f}",
            "lambda_3_factor": f"{params.lambda_3_factor:.4f}",
            "lambda_delta_factor": f"{params.lambda_delta_factor:.4f}",
            "omega_1_num_mode": "integral" if params.include_omega_1_num else "disabled",
            "residual_cutoff_mev": f"{params.residual_cutoff_mev:.1f}",
        }
        save_table(data_dir / f"qmd_simple_{tag}.txt", columns, data, metadata)

        # Report onset and failure count
        onset_mu = next((s.mu_q_mev for s in states if s.phase == "2SC"), None)
        n_failed = sum(1 for s in states if not s.success)
        if onset_mu is not None:
            print(f"  2SC onset at mu_q ≈ {onset_mu:.1f} MeV")
        else:
            print("  No 2SC condensation found in scanned range.")
        if n_failed:
            print(f"  WARNING: {n_failed} minimization point(s) did not report success.")

    save_condensate_plot(all_states["set_a"], all_states["set_b"], plots_dir)

    # ------------------------------------------------------------------ #
    # Convention comparison: t_loop4_factor=4 vs =8                       #
    # ------------------------------------------------------------------ #
    convention_scan_check()

    # ------------------------------------------------------------------ #
    # Delta=0 QM reduction check at multiple (phi, mu) values             #
    # ------------------------------------------------------------------ #
    print("\nDelta=0 reduction check  (QMD omega(phi,0,mu) vs QM, SET_A):")
    print(f"  {'phi':>6}  {'mu_q':>6}  {'omega_qmd':>14}  {'omega_qm':>14}  {'diff':>14}")
    rows = delta_zero_reduction_check(QMD_SET_A)
    prev_phi = None
    for r in rows:
        if r["phi"] != prev_phi:
            if prev_phi is not None:
                print()
            prev_phi = r["phi"]
        print(
            f"  {r['phi']:6.1f}  {r['mu_q']:6.0f}  {r['omega_qmd']:14.6e}  "
            f"{r['omega_qm']:14.6e}  {r['difference']:14.6e}"
        )
    print("\n  Expected: diff is mu-independent for each phi (vacuum-level constant).")
    # Show that diff is constant per phi block
    from itertools import groupby
    print("\n  Variation of diff within each phi block (should be ~0):")
    for phi_val, group in groupby(rows, key=lambda r: r["phi"]):
        diffs = [r["difference"] for r in group]
        span = max(diffs) - min(diffs)
        print(f"    phi={phi_val:.1f} MeV: diff spread = {span:.3e} MeV^4")


if __name__ == "__main__":
    main()
