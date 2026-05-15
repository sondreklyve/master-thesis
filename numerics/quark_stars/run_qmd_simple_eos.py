"""Build and plot the QMD simple (common-mu) equation of state.

Uses the QMD zero-temperature 2SC medium split from Eq. A.22/A.23:
all quarks are free at Delta=0, while the paired phase has blue free quarks
plus Omega_1_num for the red-green sector.

Produces:
  output/qmd_simple_eos/data/qmd_simple_eos_set_a.txt
  output/qmd_simple_eos/data/qmd_simple_eos_set_b.txt
  output/qmd_simple_eos/plots/qmd_eos_p_vs_eps.pdf
  output/qmd_simple_eos/plots/qmd_eos_cs2.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import MEV4_TO_GEV_FM3, MEV3_TO_FM_MINUS3
from .io import output_directories, save_table
from .plotting import apply_plot_style, save_figure
from .qmd_parameters import QMD_SET_A, QMD_SET_A_REFERENCE, QMD_SET_B, QMDParameters  # noqa: F401
from .qmd_simple import QMDSimpleEoSPoint, QMDSimpleModel


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


def build_eos(params: QMDParameters, mu_values: np.ndarray) -> list[QMDSimpleEoSPoint]:
    model = QMDSimpleModel(params)
    return model.build_eos(mu_values)


def eos_to_array(
    points: list[QMDSimpleEoSPoint],
) -> tuple[list[str], np.ndarray]:
    columns = [
        "mu_q_mev", "phi_mev", "delta_mev", "gap_mev",
        "pressure_mev4", "n_q_fm3", "energy_density_gev_fm3", "cs2",
    ]
    data = np.array([
        [
            pt.mu_q_mev,
            pt.phi_mev,
            pt.delta_mev,
            pt.gap_mev,
            pt.pressure_mev4,
            pt.n_q_mev3 * MEV3_TO_FM_MINUS3,
            pt.energy_density_mev4 * MEV4_TO_GEV_FM3,
            pt.cs2,
        ]
        for pt in points
    ])
    return columns, data


def _p_vs_eps_on_ax(
    ax,
    eos_a: list[QMDSimpleEoSPoint],
    eos_b: list[QMDSimpleEoSPoint],
) -> None:
    """Draw P(ε) curves for SET A and SET B on the given axes."""
    for pts, label, ls, color in [
        (eos_a, "SET A", "-", SET_COLORS[0]),
        (eos_b, "SET B", "-", SET_COLORS[1]),
    ]:
        p_all = np.array([pt.pressure_mev4 * MEV4_TO_GEV_FM3 for pt in pts])
        e_all = np.array([pt.energy_density_mev4 * MEV4_TO_GEV_FM3 for pt in pts])
        ok = np.array([pt.success for pt in pts])
        # Restrict to positive pressure, converged points, then enforce monotone P
        mask = (p_all >= 0.0) & ok
        p = p_all[mask]
        e = e_all[mask]
        if p.size >= 2:
            mono = np.concatenate(([True], np.diff(p) >= 0))
            p, e = p[mono], e[mono]
        if p.size:
            ax.plot(p, e, linewidth=2.2, linestyle=ls, color=color, label=label)
    ax.set_xlabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_ylabel(r"$\varepsilon\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_xlim(-0.01, 0.05)
    ax.set_ylim(0.0, 0.45)
    ax.legend()


def _cs2_on_ax(
    ax,
    eos_a: list[QMDSimpleEoSPoint],
    eos_b: list[QMDSimpleEoSPoint],
) -> None:
    """Draw cs² vs μ_q for SET A and SET B on the given axes."""
    for pts, label, ls, color in [
        (eos_a, "SET A", "-", SET_COLORS[0]),
        (eos_b, "SET B", "-", SET_COLORS[1]),
    ]:
        mu = np.array([pt.mu_q_mev for pt in pts])
        cs2 = np.array([pt.cs2 for pt in pts])
        mask = np.isfinite(cs2) & (mu >= 250.0) & (mu <= 700.0)
        if mask.any():
            ax.plot(mu[mask], cs2[mask], linewidth=2.2, linestyle=ls,
                    color=color, label=label)
    ax.axhline(1.0 / 3.0, color="black", linewidth=1.0, linestyle="--",
               label=r"$c_s^2=1/3$", alpha=0.8)
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$c_s^2$")
    ax.set_xlim(250.0, 700.0)
    ax.set_ylim(0.2, 0.45)
    ax.legend()


def plot_p_vs_eps(
    eos_a: list[QMDSimpleEoSPoint],
    eos_b: list[QMDSimpleEoSPoint],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _p_vs_eps_on_ax(ax, eos_a, eos_b)
    ax.set_title(r"QMD simple EoS")
    fig.tight_layout()
    save_figure(plots_dir / "qmd_eos_p_vs_eps.pdf")


def plot_cs2(
    eos_a: list[QMDSimpleEoSPoint],
    eos_b: list[QMDSimpleEoSPoint],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    _cs2_on_ax(ax, eos_a, eos_b)
    ax.set_title(r"Speed of sound squared")
    fig.tight_layout()
    save_figure(plots_dir / "qmd_eos_cs2.pdf")


def plot_combined_eos(
    eos_a: list[QMDSimpleEoSPoint],
    eos_b: list[QMDSimpleEoSPoint],
    plots_dir: Path,
) -> None:
    """Side-by-side combined EoS figure: P(ε) on left, cs² on right."""
    fig, (ax_p, ax_cs) = plt.subplots(1, 2, figsize=(14.0, 5.0))
    _p_vs_eps_on_ax(ax_p, eos_a, eos_b)
    _cs2_on_ax(ax_cs, eos_a, eos_b)
    fig.tight_layout()
    save_figure(plots_dir / "qmd_simple_eos.pdf")


def main() -> None:
    apply_plot_style()
    data_dir, plots_dir = output_directories(OUTPUT_DIR, "qmd_simple_eos")
    mu_values = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)

    all_eos: dict[str, list[QMDSimpleEoSPoint]] = {}

    for tag, params in PARAMETER_SETS:
        print(f"\nBuilding QMD EoS  {tag.upper()}  "
              f"(m_delta={params.m_delta_mev:.0f} MeV, "
              f"g_delta_factor={params.g_delta_factor:.2f}) ...")

        pts = build_eos(params, mu_values)
        all_eos[tag] = pts

        # Summary statistics
        onset_mu = next((pt.mu_q_mev for pt in pts if pt.phase == "2SC"), None)
        n_failed = sum(1 for pt in pts if not pt.success)
        p_max = max(pt.pressure_mev4 * MEV4_TO_GEV_FM3 for pt in pts) if pts else 0.0
        e_max = max(pt.energy_density_mev4 * MEV4_TO_GEV_FM3 for pt in pts) if pts else 0.0
        cs2_vals = [pt.cs2 for pt in pts if np.isfinite(pt.cs2)]
        cs2_max = max(cs2_vals) if cs2_vals else float("nan")
        n_neg_cs2 = sum(1 for v in cs2_vals if v < 0.0)

        print(f"  EoS points (P>=0): {len(pts)}")
        if onset_mu is not None:
            print(f"  2SC onset at mu_q ≈ {onset_mu:.1f} MeV")
        else:
            print("  No 2SC condensation found in scanned range.")
        print(f"  P_max = {p_max:.4f} GeV/fm³,  eps_max = {e_max:.4f} GeV/fm³")
        print(f"  cs² range: [{min(cs2_vals):.4f}, {cs2_max:.4f}]")
        if n_neg_cs2:
            print(f"  WARNING: {n_neg_cs2} point(s) with cs² < 0 (thermodynamic instability)")
        if n_failed:
            print(f"  WARNING: {n_failed} minimization point(s) did not report success.")

        columns, data = eos_to_array(pts)
        metadata = {
            "model": "QMD simple EoS: Omega_trunc + 2SC T=0 medium",
            "omega_1_num_mode": "integral" if params.include_omega_1_num else "disabled",
            "residual_cutoff_mev": f"{params.residual_cutoff_mev:.1f}",
            "set": tag,
            "m_delta_mev": f"{params.m_delta_mev:.1f}",
            "g_delta_factor": f"{params.g_delta_factor:.4f}",
            "lambda_3_factor": f"{params.lambda_3_factor:.4f}",
            "lambda_delta_factor": f"{params.lambda_delta_factor:.4f}",
            "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
            "mu_min_mev": f"{MU_MIN_MEV:.1f}",
            "mu_max_mev": f"{MU_MAX_MEV:.1f}",
            "num_points_total": str(NUM_POINTS),
        }
        save_table(data_dir / f"qmd_simple_eos_{tag}.txt", columns, data, metadata)

    plot_p_vs_eps(all_eos["set_a"], all_eos["set_b"], plots_dir)
    plot_cs2(all_eos["set_a"], all_eos["set_b"], plots_dir)
    plot_combined_eos(all_eos["set_a"], all_eos["set_b"], plots_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
