"""Build the beta-equilibrated, charge-neutral stellar QM EoS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .bag_model import bag_summary, minimum_additional_bag_constant_mev4
from .constants import GEV_FM3_TO_MEV4, IRON_ENERGY_PER_BARYON_MEV
from .io import output_directories
from .plotting import apply_plot_style, line_plot, save_figure
from .qm_parameters import DEFAULT_QM_VACUUM_INPUTS, QMFittedParameters, fit_qm_parameters
from .qm_potential import TwoFlavorQMPotential
from .qm_stellar_matter import StellarMatterTable, build_stellar_matter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-min-ratio", type=float, default=0.01)
    parser.add_argument("--sigma-max-ratio", type=float, default=0.9999)
    parser.add_argument("--num-points", type=int, default=450)
    parser.add_argument("--m-sigma-values", type=float, nargs="+", default=[DEFAULT_QM_VACUUM_INPUTS.m_sigma_mev])
    parser.add_argument("--stability-energy-per-baryon-mev", type=float, default=IRON_ENERGY_PER_BARYON_MEV)
    parser.add_argument("--bag-gev-fm3", type=float, default=None)
    parser.add_argument("--bag-scale-factor", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    return parser.parse_args()


def sigma_tag(m_sigma_mev: float) -> str:
    return f"sigma_{int(round(m_sigma_mev))}"


def build_shifted_stellar_table(
    fitted: QMFittedParameters,
    sigma_values_mev: np.ndarray,
    stability_energy_per_baryon_mev: float,
    bag_gev_fm3: float | None,
    bag_scale_factor: float,
) -> StellarMatterTable:
    potential = TwoFlavorQMPotential(fitted)
    base_table = build_stellar_matter(potential, sigma_values_mev)
    minimum_b_mev4 = minimum_additional_bag_constant_mev4(
        base_table.pressure_b0_mev4,
        base_table.bag_energy_density_b0_mev4,
        base_table.baryon_density_mev3,
        target_energy_per_baryon_mev=stability_energy_per_baryon_mev,
    )
    if bag_gev_fm3 is None:
        b_mev4 = bag_scale_factor * minimum_b_mev4
        bag_source = f"B_min x {bag_scale_factor:.3f}"
    else:
        b_mev4 = bag_gev_fm3 * GEV_FM3_TO_MEV4
        bag_source = "manual"
    return StellarMatterTable(
        m_sigma_mev=base_table.m_sigma_mev,
        b0_mev4=base_table.b0_mev4,
        b_mev4=b_mev4,
        bag_source=bag_source,
        minimum_b_mev4=minimum_b_mev4,
        sigma_mev=base_table.sigma_mev,
        mu_u_mev=base_table.mu_u_mev,
        mu_d_mev=base_table.mu_d_mev,
        mu_e_mev=base_table.mu_e_mev,
        constituent_mass_mev=base_table.constituent_mass_mev,
        n_u_mev3=base_table.n_u_mev3,
        n_d_mev3=base_table.n_d_mev3,
        n_e_mev3=base_table.n_e_mev3,
        pressure_raw_mev4=base_table.pressure_raw_mev4,
        energy_density_raw_mev4=base_table.energy_density_raw_mev4,
        bag_energy_density_b0_mev4=base_table.bag_energy_density_b0_mev4,
    )


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
        tag = sigma_tag(m_sigma_mev)
        table.save(data_dir / f"qm_stellar_eos_{tag}.txt")

        line_plot(
            table.mu_q_mev,
            table.mu_e_mev,
            r"Average quark chemical potential $\mu_q\;(\mathrm{MeV})$",
            r"Electron chemical potential $\mu_e\;(\mathrm{MeV})$",
            rf"Stellar QM Matter: $\mu_e(\mu_q)$, $m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$",
            plots_dir / f"mu_e_vs_mu_q_{tag}.pdf",
            color="#2f6b3b",
        )
        fig, ax = plt.subplots()
        ax.plot(table.mu_q_mev, table.n_u_fm3, linewidth=2.0, label=r"$n_u$")
        ax.plot(table.mu_q_mev, table.n_d_fm3, linewidth=2.0, label=r"$n_d$")
        ax.plot(table.mu_q_mev, table.n_e_fm3, linewidth=2.0, label=r"$n_e$")
        ax.set_xlabel(r"Average quark chemical potential $\mu_q\;(\mathrm{MeV})$")
        ax.set_ylabel(r"Number density $(\mathrm{fm}^{-3})$")
        ax.set_title(rf"Composition of Stellar QM Matter, $m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
        ax.legend()
        save_figure(plots_dir / f"composition_vs_mu_q_{tag}.pdf")

        mask = table.pressure_mev4 >= 0.0
        fig, ax = plt.subplots()
        ax.plot(table.energy_density_gev_fm3[mask], table.pressure_gev_fm3[mask], linewidth=2.2, color="#7a2f2f")
        ax.set_xlabel(r"Energy density $\varepsilon\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
        ax.set_ylabel(r"Pressure $P\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
        ax.set_title(rf"Stellar QM Matter EoS, $m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
        save_figure(plots_dir / f"pressure_vs_energy_density_{tag}.pdf")

        try:
            surface = table.bag_surface()
            fig, ax = plt.subplots()
            ax.scatter([table.b_mev4**0.25], [surface.energy_per_baryon_mev], color="#1f4e79", zorder=3)
            ax.axhline(args.stability_energy_per_baryon_mev, color="black", linestyle="--", linewidth=1.2)
            ax.set_xlabel(r"$B^{1/4}\;(\mathrm{MeV})$")
            ax.set_ylabel(r"$\varepsilon / n_B\;(\mathrm{MeV})$ at $P=0$")
            ax.set_title(rf"Bag Summary, $m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
            save_figure(plots_dir / f"bag_summary_{tag}.pdf")
        except ValueError:
            pass

        summary = bag_summary(table.b0_mev4, table.b_mev4, table.minimum_b_mev4)
        print(f"[m_sigma = {m_sigma_mev:.0f} MeV] wrote {data_dir / f'qm_stellar_eos_{tag}.txt'}")
        print("  " + ", ".join(f"{key}={value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
