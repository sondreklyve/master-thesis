"""Run the simple pedagogical T=0 two-flavor QM-model pipeline."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import MEV3_TO_FM_MINUS3, NC, N_FLAVORS
from .io import output_directories, save_table
from .plotting import apply_plot_style, save_figure
from .qm_parameters import DEFAULT_QM_VACUUM_INPUTS, fit_qm_parameters
from .qm_potential import TwoFlavorQMPotential
from .qm_simple_eos import SimpleEOSTable, build_simple_eos


MU_MIN_MEV = 0.0
MU_MAX_MEV = 500.0
NUM_POINTS = 400
M_SIGMA_VALUES_MEV = (500.0, 600.0, 700.0)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def sigma_tag(m_sigma_mev: float) -> str:
    return f"sigma_{int(round(m_sigma_mev))}"


def sigma_label(m_sigma_mev: float) -> str:
    return rf"$m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$"


def sigma_colors(num_curves: int) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0.2, 0.85, num_curves))


def build_tables(mu_values_mev: np.ndarray) -> list[SimpleEOSTable]:
    tables: list[SimpleEOSTable] = []
    for m_sigma_mev in M_SIGMA_VALUES_MEV:
        fitted = fit_qm_parameters(DEFAULT_QM_VACUUM_INPUTS.with_m_sigma(m_sigma_mev))
        potential = TwoFlavorQMPotential(fitted)
        tables.append(build_simple_eos(potential, mu_values_mev))
    return tables


def save_sigma_plot(tables: list[SimpleEOSTable], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    for table, color in zip(tables, sigma_colors(len(tables)), strict=True):
        ax.plot(table.mu_q_mev, table.sigma_mev, linewidth=2.2, color=color, label=sigma_label(table.m_sigma_mev))
    ax.set_xlabel(r"Quark chemical potential $\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"Condensate $\sigma\;(\mathrm{MeV})$")
    ax.legend()
    save_figure(plots_dir / "sigma_vs_mu_multi.pdf")


def save_number_density_plot(tables: list[SimpleEOSTable], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    # Free massless quark reference: n_q = (N_c N_f / 3π²) μ_q³
    mu_ref = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)
    n_ref_fm3 = (NC * N_FLAVORS / (3.0 * np.pi**2)) * mu_ref**3 * MEV3_TO_FM_MINUS3
    ax.plot(
        mu_ref,
        n_ref_fm3,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=r"Free massless quarks",
        zorder=0,
    )
    for table, color in zip(tables, sigma_colors(len(tables)), strict=True):
        ax.plot(
            table.mu_q_mev,
            table.quark_number_density_fm3,
            linewidth=2.2,
            color=color,
            label=sigma_label(table.m_sigma_mev),
        )
    ax.set_xlabel(r"Quark chemical potential $\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"Quark number density $n_q\;(\mathrm{fm}^{-3})$")
    ax.legend()
    save_figure(plots_dir / "number_density_vs_mu_multi.pdf")


def save_pressure_plot(tables: list[SimpleEOSTable], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    for table, color in zip(tables, sigma_colors(len(tables)), strict=True):
        ax.plot(
            table.mu_q_mev,
            table.pressure_gev_fm3,
            linewidth=2.2,
            color=color,
            label=sigma_label(table.m_sigma_mev),
        )
    ax.set_xlabel(r"Quark chemical potential $\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"Pressure $P\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.legend()
    save_figure(plots_dir / "pressure_vs_mu_multi.pdf")


def save_pressure_energy_density_plot(tables: list[SimpleEOSTable], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    for table, color in zip(tables, sigma_colors(len(tables)), strict=True):
        mask = table.positive_pressure_mask
        ax.plot(
            table.pressure_gev_fm3[mask],
            table.energy_density_gev_fm3[mask],
            linewidth=2.2,
            color=color,
            label=sigma_label(table.m_sigma_mev),
        )
    ax.set_xlabel(r"Pressure $P\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.set_ylabel(r"Energy density $\varepsilon\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.legend()
    save_figure(plots_dir / "pressure_vs_energy_density_multi.pdf")


def save_speed_of_sound_outputs(tables: list[SimpleEOSTable], data_dir: Path, plots_dir: Path) -> None:
    curves: list[tuple[np.ndarray, np.ndarray]] = [table.speed_of_sound_squared_branch() for table in tables]
    max_points = max((mu_q_mev.size for mu_q_mev, _ in curves), default=0)
    combined = np.full((max_points, 2 * len(tables)), np.nan)
    columns: list[str] = []
    metadata: dict[str, object] = {
        "pipeline": "simple",
        "diagnostic": "speed_of_sound_squared",
        "definition": "c_s^2=(dP/dmu_q)/(depsilon/dmu_q) on the P>=0 branch",
        "units": "mu_q in MeV; c_s^2 dimensionless",
    }

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    colors = sigma_colors(len(tables))
    for index, (table, color, (mu_q_mev, cs2)) in enumerate(zip(tables, colors, curves, strict=True)):
        combined[: mu_q_mev.size, 2 * index] = mu_q_mev
        combined[: cs2.size, 2 * index + 1] = cs2
        columns.extend(
            [
                f"mu_q_mev_sigma_{int(round(table.m_sigma_mev))}",
                f"c_s2_sigma_{int(round(table.m_sigma_mev))}",
            ]
        )
        metadata[f"m_sigma_{index + 1}_mev"] = f"{table.m_sigma_mev:.6f}"
        ax.plot(mu_q_mev, cs2, linewidth=2.2, color=color, label=sigma_label(table.m_sigma_mev))

    ax.axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"Quark chemical potential $\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"Speed of sound squared $c_s^2$")
    ax.legend()
    save_figure(plots_dir / "speed_of_sound_vs_mu_multi.pdf")
    save_table(data_dir / "qm_simple_speed_of_sound_multi.txt", columns, combined, metadata)


def main() -> None:
    apply_plot_style()
    data_dir, plots_dir = output_directories(OUTPUT_DIR, "simple")
    mu_values_mev = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)
    tables = build_tables(mu_values_mev)

    for table in tables:
        tag = sigma_tag(table.m_sigma_mev)
        table.save(data_dir / f"qm_simple_{tag}.txt")

    save_sigma_plot(tables, plots_dir)
    save_number_density_plot(tables, plots_dir)
    save_pressure_plot(tables, plots_dir)
    save_pressure_energy_density_plot(tables, plots_dir)
    save_speed_of_sound_outputs(tables, data_dir, plots_dir)


if __name__ == "__main__":
    main()
