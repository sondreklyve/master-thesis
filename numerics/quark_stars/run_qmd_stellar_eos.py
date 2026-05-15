"""Build a neutral QMD stellar EoS from equilibrium states.

This script converts the neutral equilibrium branch
    mu_q -> phi, Delta, mu_e, mu_8, Omega_min
into thermodynamic quantities P, n_q, n_B, epsilon, and c_s^2.

It writes both the raw neutral branch and a Maxwell/stability-cleaned branch.
It does not perform bag shifting or TOV integration.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import MEV4_TO_GEV_FM3
from .io import ensure_directory, output_directories
from .plotting import apply_plot_style, save_figure
from .qmd_parameters import QMD_SET_A, QMD_SET_B, QMDParameters
from .qmd_stellar import (
    QMDStellarEoSPoint,
    QMDStellarModel,
    QMDStellarState,
    build_qmd_stellar_eos_from_states,
)
from .thermodynamics.maxwell import maxwell_construct


MU_MIN_MEV = 250.0
MU_MAX_MEV = 900.0
NUM_POINTS = 350
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
EQUILIBRIUM_DATA_DIR = OUTPUT_DIR / "qmd_stellar_equilibrium" / "data"

# House style: viridis colors for SET A (index 0) and SET B (index 1)
SET_COLORS = plt.cm.viridis(np.linspace(0.15, 0.6, 2))

EOS_COLUMNS = [
    "mu_q_mev",
    "mu_B_mev",
    "phi_mev",
    "delta_mev",
    "gap_mev",
    "mu_e_mev",
    "mu_8_mev",
    "delta_mu_mev",
    "gap_minus_delta_mu_mev",
    "pressure_mev4",
    "quark_density_mev3",
    "baryon_density_mev3",
    "energy_density_mev4",
    "cs2",
    "omega_min_mev4",
    "phase",
    "success",
    "neutrality_residual_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the neutral QMD stellar EoS from equilibrium states.",
    )
    parser.add_argument(
        "--include-set-b",
        action="store_true",
        default=True,
        help="Also build the exploratory QMD_SET_B EoS after QMD_SET_A (default: True).",
    )
    parser.add_argument("--mu-min", type=float, default=MU_MIN_MEV)
    parser.add_argument("--mu-max", type=float, default=MU_MAX_MEV)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument(
        "--no-reuse-equilibrium",
        action="store_true",
        help="Always recompute equilibrium states instead of reusing a matching saved table.",
    )
    return parser.parse_args()


def parameter_sets(include_set_b: bool) -> list[tuple[str, QMDParameters]]:
    sets = [("set_a", QMD_SET_A)]
    if include_set_b:
        sets.append(("set_b", QMD_SET_B))
    return sets


def read_equilibrium_table(path: Path) -> list[QMDStellarState]:
    """Read states saved by run_qmd_stellar_equilibrium_scan.py."""
    states: list[QMDStellarState] = []
    if not path.exists():
        return states

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 14:
            continue
        states.append(QMDStellarState(
            mu_q_mev=float(parts[0]),
            phi_mev=float(parts[1]),
            delta_mev=float(parts[2]),
            gap_mev=float(parts[3]),
            mu_e_mev=float(parts[4]),
            mu_8_mev=float(parts[5]),
            delta_mu_mev=float(parts[6]),
            gap_minus_delta_mu_mev=float(parts[7]),
            omega_min_mev4=float(parts[8]),
            neutrality_residual_e=float(parts[9]),
            neutrality_residual_8=float(parts[10]),
            neutrality_residual_norm=float(parts[11]),
            phase=parts[12],
            success=bool(int(parts[13])),
            message=f"Loaded from {path.name}",
        ))
    return states


def states_match_grid(
    states: list[QMDStellarState],
    mu_values_mev: np.ndarray,
    *,
    atol_mev: float = 1.0e-8,
) -> bool:
    if len(states) != len(mu_values_mev):
        return False
    loaded_mu = np.array([s.mu_q_mev for s in states], dtype=float)
    return bool(np.allclose(loaded_mu, mu_values_mev, rtol=0.0, atol=atol_mev))


def scan_equilibrium_states(
    model: QMDStellarModel,
    mu_values_mev: np.ndarray,
) -> list[QMDStellarState]:
    """Compute neutral equilibrium states over mu_q using continuation."""
    states: list[QMDStellarState] = []
    previous_state: QMDStellarState | None = None
    minimizer_options = {"maxiter": 80, "ftol": 1.0e-8}

    for i, mu in enumerate(mu_values_mev, start=1):
        state = model.solve_equilibrium(
            float(mu),
            previous_state=previous_state,
            initial_neutrality_guess=(0.0, 0.0),
            minimizer_options=minimizer_options,
        )
        states.append(state)
        if state.success and np.isfinite(state.mu_e_mev) and np.isfinite(state.mu_8_mev):
            previous_state = state

        print(
            f"    {i:3d}/{len(mu_values_mev)}  "
            f"mu_q={state.mu_q_mev:7.2f}  "
            f"P-state phi={state.phi_mev:8.3f}  "
            f"gap={state.gap_mev:8.3f}  "
            f"res={state.neutrality_residual_norm:9.2e}  "
            f"{state.phase:>6}  "
            f"{'ok' if state.success else 'FAIL'}",
            flush=True,
        )

    return states


def load_or_scan_equilibrium_states(
    tag: str,
    model: QMDStellarModel,
    mu_values_mev: np.ndarray,
    *,
    allow_reuse: bool,
) -> tuple[list[QMDStellarState], str]:
    table = EQUILIBRIUM_DATA_DIR / f"qmd_stellar_equilibrium_{tag}.txt"
    if allow_reuse and table.exists():
        states = read_equilibrium_table(table)
        if states_match_grid(states, mu_values_mev):
            print(f"  Reusing matching equilibrium table: {table}")
            return states, f"reused:{table}"
        print(f"  Existing equilibrium table does not match requested EoS grid: {table}")

    print("  Computing equilibrium branch for EoS grid ...")
    return scan_equilibrium_states(model, mu_values_mev), "computed"


def solve_vacuum_state(model: QMDStellarModel) -> QMDStellarState:
    """Solve the neutral vacuum reference at mu_q=0.

    Vacuum subtraction uses the same equilibrium machinery as the finite-mu
    branch.  If the minimizer cannot report a finite neutral solution, fall
    back to the canonical vacuum state (phi=f_pi, Delta=mu_e=mu_8=0), which is
    the model's fitted vacuum by construction.
    """
    state = model.solve_equilibrium(
        0.0,
        initial_fields=(model.params.f_pi_mev, 0.0),
        initial_neutrality_guess=(0.0, 0.0),
        minimizer_options={"maxiter": 80, "ftol": 1.0e-8},
    )
    if state.success and np.isfinite(state.omega_min_mev4):
        return state

    omega_vac = model.omega(model.params.f_pi_mev, 0.0, 0.0, 0.0, 0.0)
    return QMDStellarState(
        mu_q_mev=0.0,
        phi_mev=model.params.f_pi_mev,
        delta_mev=0.0,
        gap_mev=0.0,
        mu_e_mev=0.0,
        mu_8_mev=0.0,
        delta_mu_mev=0.0,
        gap_minus_delta_mu_mev=0.0,
        omega_min_mev4=omega_vac,
        neutrality_residual_e=0.0,
        neutrality_residual_8=0.0,
        neutrality_residual_norm=0.0,
        phase="normal",
        success=False,
        message="Canonical vacuum fallback after solve_equilibrium did not converge.",
    )


def filter_eos_points(points: list[QMDStellarEoSPoint]) -> list[QMDStellarEoSPoint]:
    return [
        p for p in points
        if (
            p.success
            and p.pressure_mev4 > 0.0
            and p.quark_density_mev3 > 0.0
            and p.energy_density_mev4 > 0.0
            and np.isfinite(p.cs2)
        )
    ]


def recompute_cs2_from_pressure_energy(
    points: list[QMDStellarEoSPoint],
) -> list[QMDStellarEoSPoint]:
    """Recompute c_s^2=dP/d epsilon on a cleaned P(epsilon) branch."""
    if not points:
        return []
    if len(points) < 2:
        return [replace(p, cs2=float("nan")) for p in points]

    pressure = np.array([p.pressure_mev4 for p in points], dtype=float)
    energy = np.array([p.energy_density_mev4 for p in points], dtype=float)
    edge_order = 2 if len(points) >= 3 else 1
    dpressure_denergy = np.gradient(pressure, energy, edge_order=edge_order)

    return [
        replace(
            p,
            cs2=(
                float(dpressure_denergy[i])
                if np.isfinite(dpressure_denergy[i])
                else float("nan")
            ),
        )
        for i, p in enumerate(points)
    ]


def replace_point_thermodynamics(
    point: QMDStellarEoSPoint,
    *,
    pressure_mev4: float,
    energy_density_mev4: float,
    cs2: float = float("nan"),
) -> QMDStellarEoSPoint:
    return replace(
        point,
        pressure_mev4=pressure_mev4,
        energy_density_mev4=energy_density_mev4,
        cs2=cs2,
    )


def nearest_eos_point(
    points: list[QMDStellarEoSPoint],
    pressure_mev4: float,
    energy_density_mev4: float,
) -> QMDStellarEoSPoint:
    pressure = np.array([p.pressure_mev4 for p in points], dtype=float)
    energy = np.array([p.energy_density_mev4 for p in points], dtype=float)
    p_scale = max(1.0, float(np.nanmax(np.abs(pressure))))
    e_scale = max(1.0, float(np.nanmax(np.abs(energy))))
    distance = ((pressure - pressure_mev4) / p_scale) ** 2 + (
        (energy - energy_density_mev4) / e_scale
    ) ** 2
    return points[int(np.nanargmin(distance))]


def strictly_increasing_pressure_energy(
    points: list[QMDStellarEoSPoint],
) -> tuple[list[QMDStellarEoSPoint], int]:
    """Keep a strictly increasing P(epsilon) branch."""
    ordered = sorted(points, key=lambda p: (p.pressure_mev4, p.energy_density_mev4))
    kept: list[QMDStellarEoSPoint] = []
    removed = 0
    last_p = -float("inf")
    last_e = -float("inf")
    for point in ordered:
        if not kept:
            kept.append(point)
            last_p = point.pressure_mev4
            last_e = point.energy_density_mev4
            continue
        p_tol = 1.0e-12 * max(1.0, abs(point.pressure_mev4), abs(last_p))
        e_tol = 1.0e-12 * max(1.0, abs(point.energy_density_mev4), abs(last_e))
        if point.pressure_mev4 > last_p + p_tol and point.energy_density_mev4 > last_e + e_tol:
            kept.append(point)
            last_p = point.pressure_mev4
            last_e = point.energy_density_mev4
        else:
            removed += 1
    return kept, removed


def build_stable_eos_points(
    raw_points: list[QMDStellarEoSPoint],
) -> tuple[list[QMDStellarEoSPoint], list[int], int, int]:
    """Apply Maxwell/stability cleaning while preserving QMD-local TOV columns.

    The shared Maxwell helper acts on pressure and energy-density arrays only.
    This wrapper maps the cleaned arrays back to the nearest raw QMD point to
    retain mu_q, baryon density, and phase, then recomputes c_s^2=dP/d epsilon.
    """
    if not raw_points:
        return [], [], 0, 0

    pressure = np.array([p.pressure_mev4 for p in raw_points], dtype=float)
    energy = np.array([p.energy_density_mev4 for p in raw_points], dtype=float)
    stable_pressure, stable_energy, maxwell_indices = maxwell_construct(pressure, energy)

    mapped: list[QMDStellarEoSPoint] = []
    for p_val, e_val in zip(stable_pressure, stable_energy):
        if p_val <= 0.0 or e_val <= 0.0:
            continue
        nearest = nearest_eos_point(raw_points, float(p_val), float(e_val))
        mapped.append(
            replace_point_thermodynamics(
                nearest,
                pressure_mev4=float(p_val),
                energy_density_mev4=float(e_val),
            )
        )

    stable, monotonic_removed = strictly_increasing_pressure_energy(mapped)
    stable = recompute_cs2_from_pressure_energy(stable)

    cs2_removed = 0
    for _ in range(3):
        filtered = [
            point for point in stable
            if np.isfinite(point.cs2) and point.cs2 >= -1.0e-10
        ]
        cs2_removed += len(stable) - len(filtered)
        if len(filtered) == len(stable):
            break
        stable, extra_monotonic_removed = strictly_increasing_pressure_energy(filtered)
        monotonic_removed += extra_monotonic_removed
        stable = recompute_cs2_from_pressure_energy(stable)

    return stable, maxwell_indices, monotonic_removed, cs2_removed


def save_eos_table(
    path: Path,
    points: list[QMDStellarEoSPoint],
    metadata: dict[str, object],
) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key}={value}\n")
        f.write(f"# columns={' '.join(EOS_COLUMNS)}\n")
        for p in points:
            f.write(
                f"{p.mu_q_mev:.10e} "
                f"{p.mu_B_mev:.10e} "
                f"{p.phi_mev:.10e} "
                f"{p.delta_mev:.10e} "
                f"{p.gap_mev:.10e} "
                f"{p.mu_e_mev:.10e} "
                f"{p.mu_8_mev:.10e} "
                f"{p.delta_mu_mev:.10e} "
                f"{p.gap_minus_delta_mu_mev:.10e} "
                f"{p.pressure_mev4:.10e} "
                f"{p.quark_density_mev3:.10e} "
                f"{p.baryon_density_mev3:.10e} "
                f"{p.energy_density_mev4:.10e} "
                f"{p.cs2:.10e} "
                f"{p.omega_min_mev4:.10e} "
                f"{p.phase} "
                f"{int(p.success)} "
                f"{p.neutrality_residual_norm:.10e}\n"
            )


def eos_arrays(points: list[QMDStellarEoSPoint]) -> dict[str, np.ndarray]:
    return {
        "mu": np.array([p.mu_q_mev for p in points]),
        "pressure": np.array([p.pressure_mev4 for p in points]),
        "energy": np.array([p.energy_density_mev4 for p in points]),
        "cs2": np.array([p.cs2 for p in points]),
        "phi": np.array([p.phi_mev for p in points]),
        "gap": np.array([p.gap_mev for p in points]),
        "mu_e": np.array([p.mu_e_mev for p in points]),
        "mu_8": np.array([p.mu_8_mev for p in points]),
    }


def _p_vs_eps_on_ax(
    ax,
    all_points: dict[str, list[QMDStellarEoSPoint]],
) -> None:
    """Draw P(ε) for all sets on the given axes."""
    tags = list(all_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(all_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls = "-"
        ax.plot(arr["pressure"] * MEV4_TO_GEV_FM3, arr["energy"] * MEV4_TO_GEV_FM3,
                linewidth=2.2, color=color, linestyle=ls, label=tag.upper().replace("_", " "))
    ax.set_xlabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_ylabel(r"$\varepsilon\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_xlim(-0.01, 0.05)
    ax.set_ylim(0.0, 0.45)
    ax.legend()


def _cs2_on_ax(
    ax,
    all_points: dict[str, list[QMDStellarEoSPoint]],
) -> None:
    """Draw cs² vs μ_q for all sets on the given axes."""
    tags = list(all_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(all_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls = "-"
        mask = np.isfinite(arr["cs2"]) & (arr["mu"] >= 250.0) & (arr["mu"] <= 700.0)
        if mask.any():
            ax.plot(arr["mu"][mask], arr["cs2"][mask],
                    linewidth=2.2, color=color, linestyle=ls, label=tag.upper().replace("_", " "))
    ax.axhline(1.0 / 3.0, color="black", linewidth=1.0, linestyle="--",
               label=r"$c_s^2=1/3$", alpha=0.8)
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$c_s^2$")
    ax.set_xlim(250.0, 700.0)
    ax.set_ylim(0.2, 0.45)
    ax.legend(fontsize=9)


def plot_pressure_vs_mu(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    tags = list(all_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(all_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls = "-"
        ax.plot(arr["mu"], arr["pressure"] * MEV4_TO_GEV_FM3,
                linewidth=2.2, color=color, linestyle=ls, label=tag.upper().replace("_", " "))
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_title(r"Neutral QMD stellar pressure")
    ax.legend()
    save_figure(plots_dir / "qmd_stellar_pressure_vs_mu.pdf")


def plot_pressure_vs_energy_density(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _p_vs_eps_on_ax(ax, all_points)
    ax.set_title(r"Neutral QMD stellar EoS")
    save_figure(plots_dir / "qmd_stellar_pressure_vs_energy_density.pdf")


def plot_speed_of_sound(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    _cs2_on_ax(ax, all_points)
    ax.set_title(r"Neutral QMD stellar speed of sound")
    save_figure(plots_dir / "qmd_stellar_speed_of_sound.pdf")


def plot_combined_eos(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    """Side-by-side: P(ε) on left, cs² on right."""
    fig, (ax_p, ax_cs) = plt.subplots(1, 2, figsize=(14.0, 5.0))
    _p_vs_eps_on_ax(ax_p, all_points)
    _cs2_on_ax(ax_cs, all_points)
    fig.tight_layout()
    save_figure(plots_dir / "qmd_stellar_eos.pdf")


def plot_condensates(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    tags = list(all_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(all_points[tag])
        label = tag.upper().replace("_", " ")
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls_phi = "-"
        ls_gap = "--"
        ax.plot(arr["mu"], arr["phi"], linewidth=2.2, color=color,
                linestyle=ls_phi, label=rf"$\phi$ {label}")
        ax.plot(arr["mu"], arr["gap"], linewidth=2.2, color=color,
                linestyle=ls_gap, label=rf"$g_\Delta\Delta_0$ {label}")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"field / gap $\;(\mathrm{MeV})$")
    ax.set_title(r"Neutral QMD condensates")
    ax.set_xlim(250.0, 600.0)
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_condensates_vs_mu.pdf")


def plot_neutrality_potentials(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    tags = list(all_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(all_points[tag])
        label = tag.upper().replace("_", " ")
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ax.plot(arr["mu"], arr["mu_e"], linewidth=2.2, color=color,
                linestyle="-", label=rf"$\mu_e$ {label}")
        ax.plot(arr["mu"], arr["mu_8"], linewidth=2.2, color=color,
                linestyle="--", label=rf"$\mu_8$ {label}")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"chemical potential $\;(\mathrm{MeV})$")
    ax.set_title(r"Neutrality chemical potentials")
    ax.set_xlim(250.0, 600.0)
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_neutrality_potentials.pdf")


def save_plots(
    all_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    plot_pressure_vs_mu(all_points, plots_dir)
    plot_pressure_vs_energy_density(all_points, plots_dir)
    plot_speed_of_sound(all_points, plots_dir)
    plot_combined_eos(all_points, plots_dir)
    plot_condensates(all_points, plots_dir)
    plot_neutrality_potentials(all_points, plots_dir)


def plot_eos_raw_vs_stable(
    raw_points: dict[str, list[QMDStellarEoSPoint]],
    stable_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    tags = list(raw_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(raw_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ax.plot(
            arr["pressure"] * MEV4_TO_GEV_FM3,
            arr["energy"] * MEV4_TO_GEV_FM3,
            linewidth=1.4, linestyle=":", color=color,
            label=f"{tag.upper().replace('_', ' ')} raw",
        )
    for i, tag in enumerate(tags):
        if tag not in stable_points:
            continue
        arr = eos_arrays(stable_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls = "-"
        ax.plot(
            arr["pressure"] * MEV4_TO_GEV_FM3,
            arr["energy"] * MEV4_TO_GEV_FM3,
            linewidth=2.2, color=color, linestyle=ls,
            label=f"{tag.upper().replace('_', ' ')} stable",
        )
    ax.set_xlabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_ylabel(r"$\varepsilon\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_title(r"Neutral QMD stellar EoS: raw vs stable")
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_eos_raw_vs_stable.pdf")


def plot_speed_of_sound_raw_vs_stable(
    raw_points: dict[str, list[QMDStellarEoSPoint]],
    stable_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    tags = list(raw_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(raw_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        mask = np.isfinite(arr["cs2"]) & (arr["mu"] >= 250.0) & (arr["mu"] <= 700.0)
        if mask.any():
            ax.plot(arr["mu"][mask], arr["cs2"][mask],
                    linewidth=1.4, linestyle=":", color=color, label=f"{tag.upper().replace('_', ' ')} raw")
    for i, tag in enumerate(tags):
        if tag not in stable_points:
            continue
        arr = eos_arrays(stable_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls = "-"
        mask = np.isfinite(arr["cs2"]) & (arr["mu"] >= 250.0) & (arr["mu"] <= 700.0)
        if mask.any():
            ax.plot(arr["mu"][mask], arr["cs2"][mask],
                    linewidth=2.2, color=color, linestyle=ls, label=f"{tag.upper().replace('_', ' ')} stable")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="-")
    ax.axhline(1.0 / 3.0, color="black", linewidth=1.0, linestyle="--",
               label=r"$c_s^2=1/3$", alpha=0.8)
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":",
               label=r"$c_s^2=1$", alpha=0.6)
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$c_s^2$")
    ax.set_title(r"Neutral QMD stellar speed of sound: raw vs stable")
    ax.set_xlim(250.0, 700.0)
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_speed_of_sound_raw_vs_stable.pdf")


def plot_pressure_vs_mu_raw_vs_stable(
    raw_points: dict[str, list[QMDStellarEoSPoint]],
    stable_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    tags = list(raw_points.keys())
    for i, tag in enumerate(tags):
        arr = eos_arrays(raw_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ax.plot(
            arr["mu"],
            arr["pressure"] * MEV4_TO_GEV_FM3,
            linewidth=1.4, linestyle=":", color=color,
            label=f"{tag.upper().replace('_', ' ')} raw",
        )
    for i, tag in enumerate(tags):
        if tag not in stable_points:
            continue
        arr = eos_arrays(stable_points[tag])
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls = "-"
        ax.plot(
            arr["mu"],
            arr["pressure"] * MEV4_TO_GEV_FM3,
            linewidth=2.2, color=color, linestyle=ls,
            label=f"{tag.upper().replace('_', ' ')} stable",
        )
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_title(r"Neutral QMD stellar pressure: raw vs stable")
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_pressure_vs_mu_raw_vs_stable.pdf")


def save_raw_vs_stable_plots(
    raw_points: dict[str, list[QMDStellarEoSPoint]],
    stable_points: dict[str, list[QMDStellarEoSPoint]],
    plots_dir: Path,
) -> None:
    plot_eos_raw_vs_stable(raw_points, stable_points, plots_dir)
    plot_speed_of_sound_raw_vs_stable(raw_points, stable_points, plots_dir)
    plot_pressure_vs_mu_raw_vs_stable(raw_points, stable_points, plots_dir)


def is_monotonic_non_decreasing(values: np.ndarray) -> bool:
    if values.size < 2:
        return True
    diffs = np.diff(values)
    local_scale = np.maximum(
        1.0,
        np.maximum(np.abs(values[:-1]), np.abs(values[1:])),
    )
    return bool(np.all(diffs >= -1.0e-12 * local_scale))


def summarize_eos(
    tag: str,
    raw_states: list[QMDStellarState],
    retained: list[QMDStellarEoSPoint],
) -> dict[str, object]:
    print(f"\nDiagnostics {tag.upper()}:")
    print(f"  Raw equilibrium points: {len(raw_states)}")
    print(f"  Retained EoS points: {len(retained)}")

    if not retained:
        print("  No retained points after filtering.")
        return {
            "retained": 0,
            "p_monotonic": False,
            "epsilon_monotonic": False,
            "cs2_exceeds_1": False,
            "maxwell_suggested": True,
        }

    arr = eos_arrays(retained)
    pressure_gev = arr["pressure"] * MEV4_TO_GEV_FM3
    energy_gev = arr["energy"] * MEV4_TO_GEV_FM3
    p_monotonic = is_monotonic_non_decreasing(arr["pressure"])
    epsilon_monotonic = is_monotonic_non_decreasing(arr["energy"])
    cs2_exceeds_1 = bool(np.any(arr["cs2"] > 1.0))
    maxwell_suggested = (not p_monotonic) or (not epsilon_monotonic)
    onset = next((p for p in retained if p.phase == "2SC"), None)

    print(
        f"  P range: {pressure_gev.min():.6g} to {pressure_gev.max():.6g} GeV/fm^3 "
        f"({arr['pressure'].min():.6e} to {arr['pressure'].max():.6e} MeV^4)"
    )
    print(
        f"  epsilon range: {energy_gev.min():.6g} to {energy_gev.max():.6g} GeV/fm^3 "
        f"({arr['energy'].min():.6e} to {arr['energy'].max():.6e} MeV^4)"
    )
    print(f"  cs2 range: {arr['cs2'].min():.6g} to {arr['cs2'].max():.6g}")
    print(f"  P(mu_q) monotonic: {p_monotonic}")
    print(f"  epsilon(P) monotonic: {epsilon_monotonic}")
    print(f"  Any cs2 > 1: {cs2_exceeds_1}")
    if onset is None:
        print("  2SC onset on retained branch: none")
    else:
        print(f"  2SC onset on retained branch: mu_q ≈ {onset.mu_q_mev:.2f} MeV")
    print(f"  Maxwell treatment suggested by monotonicity diagnostics: {maxwell_suggested}")

    return {
        "retained": len(retained),
        "p_min_mev4": float(arr["pressure"].min()),
        "p_max_mev4": float(arr["pressure"].max()),
        "epsilon_min_mev4": float(arr["energy"].min()),
        "epsilon_max_mev4": float(arr["energy"].max()),
        "cs2_min": float(arr["cs2"].min()),
        "cs2_max": float(arr["cs2"].max()),
        "p_monotonic": p_monotonic,
        "epsilon_monotonic": epsilon_monotonic,
        "cs2_exceeds_1": cs2_exceeds_1,
        "maxwell_suggested": maxwell_suggested,
        "onset_mu_q_mev": None if onset is None else float(onset.mu_q_mev),
    }


def summarize_stable_eos(
    tag: str,
    raw_points: list[QMDStellarEoSPoint],
    stable_points: list[QMDStellarEoSPoint],
    maxwell_indices: list[int],
    monotonic_removed: int,
    cs2_removed: int,
) -> dict[str, object]:
    print(f"\nStable diagnostics {tag.upper()}:")
    print(f"  Raw points: {len(raw_points)}")
    print(f"  Stable points: {len(stable_points)}")
    print(f"  Maxwell removed index marker(s): {maxwell_indices if maxwell_indices else 'none'}")
    print(f"  Monotonic-cleanup removed points: {monotonic_removed}")
    print(f"  Negative-cs2 cleanup removed points: {cs2_removed}")

    if not stable_points:
        print("  No stable points remain.")
        return {
            "stable_points": 0,
            "negative_cs2": True,
            "p_monotonic": False,
            "epsilon_monotonic": False,
            "tov_safe": False,
        }

    arr = eos_arrays(stable_points)
    pressure_gev = arr["pressure"] * MEV4_TO_GEV_FM3
    energy_gev = arr["energy"] * MEV4_TO_GEV_FM3
    p_positive = bool(np.all(arr["pressure"] > 0.0))
    energy_positive = bool(np.all(arr["energy"] > 0.0))
    p_monotonic = is_monotonic_non_decreasing(arr["pressure"])
    epsilon_monotonic = is_monotonic_non_decreasing(arr["energy"])
    negative_cs2 = bool(np.any(arr["cs2"] < -1.0e-10))
    finite_cs2 = bool(np.all(np.isfinite(arr["cs2"])))
    tov_safe = p_positive and energy_positive and p_monotonic and epsilon_monotonic and finite_cs2 and not negative_cs2

    print(
        f"  P range: {pressure_gev.min():.6g} to {pressure_gev.max():.6g} GeV/fm^3 "
        f"({arr['pressure'].min():.6e} to {arr['pressure'].max():.6e} MeV^4)"
    )
    print(
        f"  epsilon range: {energy_gev.min():.6g} to {energy_gev.max():.6g} GeV/fm^3 "
        f"({arr['energy'].min():.6e} to {arr['energy'].max():.6e} MeV^4)"
    )
    print(f"  stable cs2 range: {arr['cs2'].min():.6g} to {arr['cs2'].max():.6g}")
    print(f"  Any negative cs2 remains: {negative_cs2}")
    print(f"  P > 0 everywhere: {p_positive}")
    print(f"  epsilon > 0 everywhere: {energy_positive}")
    print(f"  P monotonic: {p_monotonic}")
    print(f"  epsilon(P) monotonic: {epsilon_monotonic}")
    print(f"  TOV interpolation should now be safe: {tov_safe}")

    return {
        "stable_points": len(stable_points),
        "p_min_mev4": float(arr["pressure"].min()),
        "p_max_mev4": float(arr["pressure"].max()),
        "epsilon_min_mev4": float(arr["energy"].min()),
        "epsilon_max_mev4": float(arr["energy"].max()),
        "cs2_min": float(arr["cs2"].min()),
        "cs2_max": float(arr["cs2"].max()),
        "negative_cs2": negative_cs2,
        "p_monotonic": p_monotonic,
        "epsilon_monotonic": epsilon_monotonic,
        "tov_safe": tov_safe,
    }


def main() -> None:
    args = parse_args()
    apply_plot_style()
    data_dir, plots_dir = output_directories(OUTPUT_DIR, "qmd_stellar_eos")
    flat_table_dir = ensure_directory(OUTPUT_DIR / "qmd_stellar_eos")
    mu_values = np.linspace(args.mu_min, args.mu_max, args.num_points)
    all_raw: dict[str, list[QMDStellarEoSPoint]] = {}
    all_stable: dict[str, list[QMDStellarEoSPoint]] = {}

    print("=" * 80)
    print("QMD Stellar Neutral EoS")
    print(f"  Omega_1_num = {'integral' if QMD_SET_A.include_omega_1_num else 'disabled'}")
    print("  Maxwell/stability cleaning applied; no TOV")
    print("  Pressure normalization: P = -(Omega_min(mu_q) - Omega_vac)")
    print("  Omega_vac: neutral equilibrium solve at mu_q=0; canonical vacuum fallback if needed")
    print(f"  EoS grid: {args.mu_min:.1f} to {args.mu_max:.1f} MeV, {args.num_points} points")
    print("=" * 80)

    for tag, params in parameter_sets(args.include_set_b):
        print(f"\nBuilding {tag.upper()} ...")
        model = QMDStellarModel(params)
        vacuum_state = solve_vacuum_state(model)
        print(
            f"  Omega_vac = {vacuum_state.omega_min_mev4:.8e} MeV^4 "
            f"(phi={vacuum_state.phi_mev:.6g}, Delta={vacuum_state.delta_mev:.6g}, "
            f"source={'solve_equilibrium' if vacuum_state.success else 'canonical fallback'})"
        )

        states, state_source = load_or_scan_equilibrium_states(
            tag,
            model,
            mu_values,
            allow_reuse=not args.no_reuse_equilibrium,
        )
        raw_eos = build_qmd_stellar_eos_from_states(states, vacuum_state.omega_min_mev4)
        retained = filter_eos_points(raw_eos)
        stable, maxwell_indices, monotonic_removed, cs2_removed = build_stable_eos_points(retained)
        all_raw[tag] = retained
        all_stable[tag] = stable
        diag = summarize_eos(tag, states, retained)
        stable_diag = summarize_stable_eos(
            tag,
            retained,
            stable,
            maxwell_indices,
            monotonic_removed,
            cs2_removed,
        )

        base_metadata = {
            "model": "QMDStellarModel",
            "set": tag,
            "omega_1_num": "integral" if params.include_omega_1_num else "disabled",
            "residual_cutoff_mev": f"{params.residual_cutoff_mev:.1f}",
            "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
            "pressure_normalization": "P=-(Omega_min(mu_q)-Omega_vac)",
            "omega_vac_mev4": f"{vacuum_state.omega_min_mev4:.10e}",
            "omega_vac_source": "neutral_equilibrium_mu_q_0" if vacuum_state.success else "canonical_vacuum_fallback",
            "delta_bound": "Delta <= f_pi",
            "neutrality": "true",
            "color_neutrality": "true",
            "mu_min_mev": f"{args.mu_min:.1f}",
            "mu_max_mev": f"{args.mu_max:.1f}",
            "num_raw_points": str(len(states)),
            "equilibrium_state_source": state_source,
        }

        raw_metadata = {
            **base_metadata,
            "branch": "raw",
            "num_retained_points": str(len(retained)),
            "maxwell_construction": "not_applied",
            "maxwell_suggested_by_monotonicity": str(diag["maxwell_suggested"]).lower(),
        }
        stable_metadata = {
            **base_metadata,
            "branch": "stable",
            "num_retained_points": str(len(stable)),
            "maxwell_construction": "applied",
            "maxwell_removed_indices": str(maxwell_indices),
            "monotonic_cleanup_removed_points": str(monotonic_removed),
            "negative_cs2_cleanup_removed_points": str(cs2_removed),
            "tov_interpolation_safe": str(stable_diag["tov_safe"]).lower(),
        }

        raw_filename = f"qmd_stellar_eos_{tag}_raw.txt"
        stable_filename = f"qmd_stellar_eos_{tag}_stable.txt"
        save_eos_table(data_dir / raw_filename, retained, raw_metadata)
        save_eos_table(flat_table_dir / raw_filename, retained, raw_metadata)
        save_eos_table(data_dir / stable_filename, stable, stable_metadata)
        save_eos_table(flat_table_dir / stable_filename, stable, stable_metadata)

    save_plots(all_stable, plots_dir)
    save_raw_vs_stable_plots(all_raw, all_stable, plots_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
