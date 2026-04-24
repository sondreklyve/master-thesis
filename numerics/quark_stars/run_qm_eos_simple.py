"""Build and plot the renormalized QM-model EoS with and without Maxwell construction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .io import output_directory
from .plotting import apply_plot_style, save_figure, sigma_colors, sigma_label
from .qm_parameters import DEFAULT_QM_VACUUM_INPUTS, fit_qm_parameters
from .qm_potential import TwoFlavorQMPotential
from .qm_stellar_matter import build_sigma_values, build_stellar_eos


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_PLOT_M_SIGMA_VALUES_MEV = (400.0, 500.0, 600.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-sigma-values", type=float, nargs="+", default=list(DEFAULT_PLOT_M_SIGMA_VALUES_MEV))
    parser.add_argument("--sigma-min-ratio", type=float, default=0.01)
    parser.add_argument("--sigma-max-ratio", type=float, default=0.9999)
    parser.add_argument("--num-points", type=int, default=450)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _insert_zero_pressure_surface(
    pressure_gev_fm3: np.ndarray,
    energy_density_gev_fm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure_gev_fm3, dtype=float)
    energy_density = np.asarray(energy_density_gev_fm3, dtype=float)
    if pressure.size == 0:
        return pressure, energy_density

    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(pressure))))
    zero_indices = np.flatnonzero(np.isclose(pressure, 0.0, atol=tolerance))
    if zero_indices.size:
        zero_index = int(zero_indices[0])
        pressure = pressure.copy()
        energy_density = energy_density.copy()
        pressure[zero_index] = 0.0
        return pressure, energy_density

    crossing_index = None
    for index in range(pressure.size - 1):
        p_left = float(pressure[index])
        p_right = float(pressure[index + 1])
        if p_left * p_right < 0.0:
            crossing_index = index
            break

    if crossing_index is None:
        anchor_index = int(np.argmin(np.abs(pressure)))
        pressure = pressure.copy()
        energy_density = energy_density.copy()
        pressure[anchor_index] = 0.0
        return pressure, energy_density
    else:
        p_left = float(pressure[crossing_index])
        p_right = float(pressure[crossing_index + 1])
        weight = -p_left / (p_right - p_left)
        surface_energy_density = float(
            energy_density[crossing_index] + weight * (energy_density[crossing_index + 1] - energy_density[crossing_index])
        )

    return (
        np.concatenate((pressure[: crossing_index + 1], [0.0], pressure[crossing_index + 1 :])),
        np.concatenate((energy_density[: crossing_index + 1], [surface_energy_density], energy_density[crossing_index + 1 :])),
    )


def main() -> None:
    args = parse_args()
    apply_plot_style()
    eos_dir = output_directory(args.output_dir, "eos")
    selected_m_sigma_values = [float(value) for value in args.m_sigma_values]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = sigma_colors(len(selected_m_sigma_values))
    for m_sigma_mev, color in zip(selected_m_sigma_values, colors, strict=True):
        fitted = fit_qm_parameters(DEFAULT_QM_VACUUM_INPUTS.with_m_sigma(m_sigma_mev))
        potential = TwoFlavorQMPotential(fitted)
        sigma_values_mev = build_sigma_values(
            potential,
            sigma_min_ratio=args.sigma_min_ratio,
            sigma_max_ratio=args.sigma_max_ratio,
            num_points=args.num_points,
        )

        eos_without_maxwell = build_stellar_eos(potential, sigma_values_mev, with_maxwell=False)
        eos_with_maxwell = build_stellar_eos(potential, sigma_values_mev, with_maxwell=True)

        tag = f"sigma_{int(round(m_sigma_mev))}"
        eos_without_maxwell.save(eos_dir / f"qm_eos_{tag}_without_maxwell.txt")
        eos_with_maxwell.save(eos_dir / f"qm_eos_{tag}_with_maxwell.txt")

        label = sigma_label(m_sigma_mev)
        no_maxwell_pressure, no_maxwell_energy_density = _insert_zero_pressure_surface(
            eos_without_maxwell.pressure_gev_fm3,
            eos_without_maxwell.energy_density_gev_fm3,
        )
        with_maxwell_pressure, with_maxwell_energy_density = _insert_zero_pressure_surface(
            eos_with_maxwell.pressure_gev_fm3,
            eos_with_maxwell.energy_density_gev_fm3,
        )

        ax.plot(
            no_maxwell_pressure,
            no_maxwell_energy_density,
            color=color,
            linewidth=2.0,
            linestyle=":",
        )
        ax.plot(
            with_maxwell_pressure,
            with_maxwell_energy_density,
            color=color,
            linewidth=2.2,
            linestyle="-",
            label=f"{label}",
        )

    ax.set_xlabel(r"Pressure $P\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.set_ylabel(r"Energy density $\varepsilon\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.set_xlim((-0.01, 0.05))
    ax.set_ylim((0.0, 0.45))
    ax.legend(ncol=1)
    save_figure(eos_dir / "pressure_vs_energy_density.pdf")


if __name__ == "__main__":
    main()
