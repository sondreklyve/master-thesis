"""Run TOV mass-radius sequences for the renormalized QM-model EoS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .thermodynamics.vacuum import b_mev4_from_root_mev, b_root_mev_from_b_mev4, minimum_bag_constant_mev4
from .io import output_directory
from .plotting import apply_plot_style, bag_curve_label, save_figure, sigma_colors
from .qm_parameters import DEFAULT_QM_VACUUM_INPUTS, fit_qm_parameters
from .qm_potential import TwoFlavorQMPotential
from .qm_stellar_matter import build_sigma_values, build_stellar_eos
from .solvers.tov import run_tov_sequence, run_tov_sequence_grav_bound


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_M_SIGMA_VALUES_MEV = (400.0, 500.0, 600.0)
BAG_ROOT_OFFSETS_MEV = (-10.0, 0.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-sigma-values", type=float, nargs="+", default=list(DEFAULT_M_SIGMA_VALUES_MEV))
    parser.add_argument("--bag-root-offsets-mev", type=float, nargs="+", default=list(BAG_ROOT_OFFSETS_MEV))
    parser.add_argument("--sigma-min-ratio", type=float, default=0.01)
    parser.add_argument("--sigma-max-ratio", type=float, default=0.9999)
    parser.add_argument("--num-points", type=int, default=600)
    parser.add_argument("--central-pressure-factor", type=float, default=1.08)
    parser.add_argument("--radial-step-km", type=float, default=0.01)
    parser.add_argument("--max-radius-km", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    stellar_dir = output_directory(args.output_dir, "stellar")
    selected_m_sigma_values = [float(value) for value in args.m_sigma_values]
    colors = sigma_colors(len(args.bag_root_offsets_mev))
    fig_combined, axes = plt.subplots(len(selected_m_sigma_values), 1, figsize=(7.4, 12.8), sharex=True)
    if len(selected_m_sigma_values) == 1:
        axes = [axes]

    for ax, m_sigma_mev in zip(axes, selected_m_sigma_values, strict=True):
        fitted = fit_qm_parameters(DEFAULT_QM_VACUUM_INPUTS.with_m_sigma(m_sigma_mev))
        potential = TwoFlavorQMPotential(fitted)
        sigma_values_mev = build_sigma_values(
            potential,
            sigma_min_ratio=args.sigma_min_ratio,
            sigma_max_ratio=args.sigma_max_ratio,
            num_points=args.num_points,
        )
        eos = build_stellar_eos(potential, sigma_values_mev, with_maxwell=True)
        b_min_mev4 = minimum_bag_constant_mev4(
            eos.pressure_b0_mev4,
            eos.energy_density_b0_mev4,
            eos.baryon_density_mev3,
        )
        b_min_root_mev = b_root_mev_from_b_mev4(b_min_mev4)
        bag_root_values_mev = [b_min_root_mev + offset_mev for offset_mev in args.bag_root_offsets_mev]
        subplot_lines: list[tuple[np.ndarray, np.ndarray, str, str, float]] = []
        for bag_root_mev, color in zip(bag_root_values_mev, colors, strict=True):
            bagged_eos = eos.with_bag_constant(b_mev4_from_root_mev(bag_root_mev), b_min_mev4=b_min_mev4)
            try:
                sequence = run_tov_sequence(
                    bagged_eos,
                    central_pressure_factor=args.central_pressure_factor,
                    radial_step_km=args.radial_step_km,
                    max_radius_km=args.max_radius_km,
                    integrator="rk4",
                )
            except ValueError as exc:
                print(
                    f"Skipping m_sigma={m_sigma_mev:.0f} MeV, "
                    f"B^(1/4)={bag_root_mev:.2f} MeV: {exc}"
                )
                continue
            tag = f"sigma_{int(round(m_sigma_mev))}_Broot_{int(round(bag_root_mev))}"
            bagged_eos.save(stellar_dir / f"qm_eos_{tag}.txt")
            sequence.save(stellar_dir / f"qm_stars_{tag}.txt")

            stable = sequence.stable_mask.astype(bool)
            unstable = ~stable
            stable_masses = sequence.mass_msun[stable]
            stable_radii = sequence.radius_km[stable]
            if stable_masses.size:
                max_index = int(np.argmax(stable_masses))
                print(
                    f"m_sigma={m_sigma_mev:.0f} MeV, "
                    f"B^(1/4)={bag_root_mev:.2f} MeV: "
                    f"M_max={stable_masses[max_index]:.6f} Msun at "
                    f"R={stable_radii[max_index]:.6f} km"
                )
            ax.plot(
                sequence.radius_km[stable],
                sequence.mass_msun[stable],
                color=color,
                linewidth=2.2,
                label=bag_curve_label(bag_root_mev),
            )
            subplot_lines.append(
                (
                    sequence.radius_km[stable],
                    sequence.mass_msun[stable],
                    color,
                    "-",
                    2.2,
                )
            )
            if np.any(unstable):
                ax.plot(
                    sequence.radius_km[unstable],
                    sequence.mass_msun[unstable],
                    color=color,
                    linewidth=1.6,
                    linestyle="--",
                )
                subplot_lines.append(
                    (
                        sequence.radius_km[unstable],
                        sequence.mass_msun[unstable],
                        color,
                        "--",
                        1.6,
                    )
                )

        # Gravitationally-bound curve for m_sigma = 600 MeV (B_min = 0)
        if abs(m_sigma_mev - 600.0) < 0.5:
            try:
                grav_seq = run_tov_sequence_grav_bound(
                    eos,
                    central_pressure_factor=args.central_pressure_factor,
                    radial_step_km=args.radial_step_km,
                    max_radius_km=100.0,
                    integrator="rk4",
                )
                grav_seq.save(stellar_dir / "qm_stars_sigma_600_grav_bound.txt")
                g_stable = grav_seq.stable_mask.astype(bool)
                g_m = grav_seq.mass_msun[g_stable]
                g_r = grav_seq.radius_km[g_stable]
                # Show only the astrophysically relevant range (M > 0.3 Msun, R < 25 km)
                phys = (g_m > 0.3) & (g_r < 25.0)
                ax.plot(
                    g_r[phys], g_m[phys],
                    color="black", linewidth=1.8, linestyle="--",
                    label=r"grav.\ bound ($B_{\min}=0$)",
                )
                subplot_lines.append((g_r[phys], g_m[phys], "black", "--", 1.8))
                print(
                    f"Grav-bound m_sigma=600 MeV: "
                    f"M_max={g_m[int(np.argmax(g_m))]:.4f} Msun at R={g_r[int(np.argmax(g_m))]:.4f} km"
                )
            except Exception as exc:
                print(f"Grav-bound m_sigma=600 MeV skipped: {exc}")

        ax.set_title(rf"$m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
        ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
        ax.set_xlim(7, 15)
        ax.set_ylim(0.4, 2.2)
        ax.legend()
        fig_single, ax_single = plt.subplots(figsize=(7.2, 4.8))
        for x, y, color, linestyle, linewidth in subplot_lines:
            ax_single.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        ax_single.set_title(rf"$m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
        ax_single.set_xlabel(r"Radius $R\;(\mathrm{km})$")
        ax_single.set_ylabel(r"Mass $M\;(M_\odot)$")
        ax_single.set_xlim(7, 15)
        ax_single.set_ylim(0.4, 2.2)
        handles, labels = ax.get_legend_handles_labels()
        ax_single.legend(handles, labels)
        save_figure(stellar_dir / f"mass_radius_sigma_{int(round(m_sigma_mev))}.pdf")

    axes[0].set_ylabel(r"Mass $M\;(M_\odot)$")
    fig_combined.tight_layout()
    fig_combined.savefig(stellar_dir / "mass_radius_combined.pdf")
    plt.close(fig_combined)


if __name__ == "__main__":
    main()
