"""Build the stellar QM EoS, apply bag shifts, and run TOV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .bag_model import bag_summary
from .constants import IRON_ENERGY_PER_BARYON_MEV
from .io import output_directories
from .plotting import apply_plot_style, save_figure
from .qm_parameters import DEFAULT_QM_VACUUM_INPUTS, fit_qm_parameters
from .run_qm_stellar_eos import build_shifted_stellar_table
from .tov_interface import run_tov_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-min-ratio", type=float, default=0.01)
    parser.add_argument("--sigma-max-ratio", type=float, default=0.9999)
    parser.add_argument("--num-points", type=int, default=450)
    parser.add_argument("--m-sigma-values", type=float, nargs="+", default=[DEFAULT_QM_VACUUM_INPUTS.m_sigma_mev])
    parser.add_argument("--stability-energy-per-baryon-mev", type=float, default=IRON_ENERGY_PER_BARYON_MEV)
    parser.add_argument("--bag-gev-fm3", type=float, default=None)
    parser.add_argument("--bag-scale-factor", type=float, default=1.0)
    parser.add_argument("--central-pressure-factor", type=float, default=1.08)
    parser.add_argument("--radial-step-km", type=float, default=0.01)
    parser.add_argument("--max-radius-km", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    return parser.parse_args()


def sigma_tag(m_sigma_mev: float) -> str:
    return f"sigma_{int(round(m_sigma_mev))}"


def main() -> None:
    args = parse_args()
    apply_plot_style()
    data_dir, plots_dir = output_directories(args.output_dir, "stellar")

    for m_sigma_mev in args.m_sigma_values:
        fitted = fit_qm_parameters(DEFAULT_QM_VACUUM_INPUTS.with_m_sigma(m_sigma_mev))
        sigma_values_mev = fitted.f_pi_mev * np.linspace(args.sigma_min_ratio, args.sigma_max_ratio, args.num_points)
        table = build_shifted_stellar_table(
            fitted,
            sigma_values_mev,
            args.stability_energy_per_baryon_mev,
            args.bag_gev_fm3,
            args.bag_scale_factor,
        )
        sequence = run_tov_sequence(
            table,
            central_pressure_factor=args.central_pressure_factor,
            radial_step_km=args.radial_step_km,
            max_radius_km=args.max_radius_km,
        )

        tag = sigma_tag(m_sigma_mev)
        table.save(data_dir / f"qm_stellar_eos_{tag}.txt")
        sequence.save(data_dir / f"qm_stars_{tag}.txt")

        fig, ax = plt.subplots()
        stable = sequence.stable_mask.astype(bool)
        unstable = ~stable
        ax.plot(sequence.radius_km[stable], sequence.mass_msun[stable], linewidth=2.2, label="stable")
        if np.any(unstable):
            ax.plot(sequence.radius_km[unstable], sequence.mass_msun[unstable], linestyle="--", linewidth=2.0, label="unstable")
        ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
        ax.set_ylabel(r"Mass $M\;(M_\odot)$")
        ax.set_title(rf"QM Star Mass-Radius Curve, $m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
        ax.legend()
        save_figure(plots_dir / f"mass_radius_{tag}.pdf")

        summary = bag_summary(table.b0_mev4, table.b_mev4, table.minimum_b_mev4)
        max_mass_index = int(np.argmax(sequence.mass_msun))
        print(f"[m_sigma = {m_sigma_mev:.0f} MeV] wrote {data_dir / f'qm_stars_{tag}.txt'}")
        print("  " + ", ".join(f"{key}={value}" for key, value in summary.items()))
        print(
            f"  M_max={sequence.mass_msun[max_mass_index]:.4f} Msun, "
            f"R(M_max)={sequence.radius_km[max_mass_index]:.4f} km"
        )


if __name__ == "__main__":
    main()
