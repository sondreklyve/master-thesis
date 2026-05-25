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


def _plot_raw_eos_by_pressure_sign(
    ax: plt.Axes,
    pressure_gev_fm3: np.ndarray,
    energy_density_gev_fm3: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    pressure = np.asarray(pressure_gev_fm3, dtype=float)
    energy_density = np.asarray(energy_density_gev_fm3, dtype=float)
    if pressure.size == 0:
        return

    split_pressure = [float(pressure[0])]
    split_energy_density = [float(energy_density[0])]

    for index in range(pressure.size - 1):
        p_left = float(pressure[index])
        p_right = float(pressure[index + 1])
        e_left = float(energy_density[index])
        e_right = float(energy_density[index + 1])

        if p_left != p_right and (p_left < 0.0) != (p_right < 0.0):
            weight = -p_left / (p_right - p_left)
            split_pressure.append(0.0)
            split_energy_density.append(e_left + weight * (e_right - e_left))

        split_pressure.append(p_right)
        split_energy_density.append(e_right)

    split_pressure_array = np.asarray(split_pressure)
    split_energy_array = np.asarray(split_energy_density)
    segment_start = 0
    first_solid_label = label

    for index in range(1, split_pressure_array.size):
        previous_negative = split_pressure_array[index - 1] < 0.0
        current_negative = split_pressure_array[index] < 0.0
        if previous_negative == current_negative:
            continue

        p_segment = split_pressure_array[segment_start : index + 1]
        e_segment = split_energy_array[segment_start : index + 1]
        is_negative = np.any(p_segment < 0.0)
        plot_label = None
        if not is_negative and first_solid_label is not None:
            plot_label = first_solid_label
            first_solid_label = None
        ax.plot(
            p_segment,
            e_segment,
            color=color,
            linewidth=2.0,
            linestyle=":" if is_negative else "-",
            label=plot_label,
        )
        segment_start = index

    p_segment = split_pressure_array[segment_start:]
    e_segment = split_energy_array[segment_start:]
    is_negative = np.any(p_segment < 0.0)
    plot_label = None
    if not is_negative and first_solid_label is not None:
        plot_label = first_solid_label
    ax.plot(
        p_segment,
        e_segment,
        color=color,
        linewidth=2.0,
        linestyle=":" if is_negative else "-",
        label=plot_label,
    )


def _surface_energy_density_after_maxwell(
    pressure_gev_fm3: np.ndarray,
    energy_density_gev_fm3: np.ndarray,
) -> float | None:
    pressure = np.asarray(pressure_gev_fm3, dtype=float)
    energy_density = np.asarray(energy_density_gev_fm3, dtype=float)

    for index in range(pressure.size - 1, 0, -1):
        p_left = float(pressure[index - 1])
        p_right = float(pressure[index])
        if p_left < 0.0 and p_right >= 0.0:
            weight = -p_left / (p_right - p_left)
            return float(energy_density[index - 1] + weight * (energy_density[index] - energy_density[index - 1]))

    return None


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
        _plot_raw_eos_by_pressure_sign(
            ax,
            eos_without_maxwell.pressure_gev_fm3,
            eos_without_maxwell.energy_density_gev_fm3,
            color=color,
            label=label,
        )
        surface_energy_density = _surface_energy_density_after_maxwell(
            eos_with_maxwell.pressure_gev_fm3,
            eos_with_maxwell.energy_density_gev_fm3,
        )
        if surface_energy_density is not None:
            ax.plot(
                [0.0, 0.0],
                [0.0, surface_energy_density],
                color=color,
                linewidth=2.0,
                linestyle="--",
            )

    ax.set_xlabel(r"Pressure $P\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.set_ylabel(r"Energy density $\varepsilon\;(\mathrm{GeV}\,\mathrm{fm}^{-3})$")
    ax.set_xlim((-0.01, 0.05))
    ax.set_ylim((0.0, 0.45))
    ax.legend(ncol=1)
    save_figure(eos_dir / "pressure_vs_energy_density.pdf")


if __name__ == "__main__":
    main()
