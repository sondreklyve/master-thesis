"""Scan neutral QMD stellar equilibrium states over fixed mu_q.

This is a diagnostic equilibrium scan only.  It does not build the neutral QMD
EoS, apply bag shifts, perform Maxwell construction, or run TOV.

The QMD stellar solver uses the Eq. A.22/A.23 2SC medium terms and a
conservative Delta <= f_pi bound inherited from the simple QMD branch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .io import ensure_directory, output_directories
from .plotting import apply_plot_style, save_figure
from .qmd_parameters import QMD_SET_A, QMD_SET_B, QMDParameters
from .qmd_stellar import QMDStellarModel, QMDStellarState


MU_MIN_MEV = 250.0
MU_MAX_MEV = 800.0
NUM_POINTS = 56
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# House style: viridis colors for SET A (index 0) and SET B (index 1)
SET_COLORS = plt.cm.viridis(np.linspace(0.15, 0.6, 2))

TABLE_COLUMNS = [
    "mu_q_mev",
    "phi_mev",
    "delta_mev",
    "gap_mev",
    "mu_e_mev",
    "mu_8_mev",
    "delta_mu_mev",
    "gap_minus_delta_mu_mev",
    "omega_min_mev4",
    "neutrality_residual_e",
    "neutrality_residual_8",
    "neutrality_residual_norm",
    "phase",
    "success",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan neutral QMD stellar equilibrium states at fixed mu_q.",
    )
    parser.add_argument(
        "--include-set-b",
        action="store_true",
        default=True,
        help="Also scan exploratory QMD_SET_B after thesis-default QMD_SET_A (default: True).",
    )
    parser.add_argument("--mu-min", type=float, default=MU_MIN_MEV)
    parser.add_argument("--mu-max", type=float, default=MU_MAX_MEV)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    return parser.parse_args()


def scan_qmd_stellar(
    model: QMDStellarModel,
    mu_values_mev: np.ndarray,
) -> list[QMDStellarState]:
    """Scan neutral equilibrium over mu_q using continuation."""
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
        if np.isfinite(state.mu_e_mev) and np.isfinite(state.mu_8_mev):
            previous_state = state

        print(
            f"    {i:3d}/{len(mu_values_mev)}  "
            f"mu_q={state.mu_q_mev:7.2f}  "
            f"phi={state.phi_mev:8.3f}  "
            f"gap={state.gap_mev:8.3f}  "
            f"mu_e={state.mu_e_mev:8.3f}  "
            f"mu_8={state.mu_8_mev:8.3f}  "
            f"res={state.neutrality_residual_norm:9.2e}  "
            f"{state.phase:>6}  "
            f"{'ok' if state.success else 'FAIL'}",
            flush=True,
        )

    return states


def save_states_table(
    path: Path,
    states: list[QMDStellarState],
    metadata: dict[str, object],
) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"# {key}={value}\n")
        f.write(f"# columns={' '.join(TABLE_COLUMNS)}\n")
        for s in states:
            f.write(
                f"{s.mu_q_mev:.10e} "
                f"{s.phi_mev:.10e} "
                f"{s.delta_mev:.10e} "
                f"{s.gap_mev:.10e} "
                f"{s.mu_e_mev:.10e} "
                f"{s.mu_8_mev:.10e} "
                f"{s.delta_mu_mev:.10e} "
                f"{s.gap_minus_delta_mu_mev:.10e} "
                f"{s.omega_min_mev4:.10e} "
                f"{s.neutrality_residual_e:.10e} "
                f"{s.neutrality_residual_8:.10e} "
                f"{s.neutrality_residual_norm:.10e} "
                f"{s.phase} "
                f"{int(s.success)}\n"
            )


def state_arrays(states: list[QMDStellarState]) -> dict[str, np.ndarray]:
    return {
        "mu": np.array([s.mu_q_mev for s in states]),
        "phi": np.array([s.phi_mev for s in states]),
        "gap": np.array([s.gap_mev for s in states]),
        "mu_e": np.array([s.mu_e_mev for s in states]),
        "mu_8": np.array([s.mu_8_mev for s in states]),
        "delta_mu": np.array([s.delta_mu_mev for s in states]),
        "gap_minus_delta_mu": np.array([s.gap_minus_delta_mu_mev for s in states]),
        "residual_norm": np.array([s.neutrality_residual_norm for s in states]),
        "success": np.array([s.success for s in states], dtype=bool),
    }


def _phi_gap_on_ax(
    ax,
    all_states: dict[str, list[QMDStellarState]],
) -> None:
    """Draw φ and gap curves for all sets on the given axes."""
    tags = list(all_states.keys())
    for i, tag in enumerate(tags):
        arr = state_arrays(all_states[tag])
        label = tag.upper().replace("_", " ")
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ls_phi = "-"
        ls_gap = "--"
        if len(tags) > 1:
            ax.plot(arr["mu"], arr["phi"], linewidth=2.2, color=color,
                    linestyle=ls_phi, label=rf"$\phi$ {label}")
            ax.plot(arr["mu"], arr["gap"], linewidth=2.2, color=color,
                    linestyle=ls_gap, label=rf"$g_\Delta\Delta_0$ {label}")
        else:
            ax.plot(arr["mu"], arr["phi"], linewidth=2.2, color=color,
                    linestyle=ls_phi, label=r"$\phi$")
            ax.plot(arr["mu"], arr["gap"], linewidth=2.2, color=color,
                    linestyle=ls_gap, label=r"$g_\Delta\Delta_0$")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"condensate / gap $\;(\mathrm{MeV})$")
    ax.set_xlim(250.0, 600.0)
    ax.legend(fontsize=9)


def _neutrality_pots_on_ax(
    ax,
    all_states: dict[str, list[QMDStellarState]],
) -> None:
    """Draw μ_e and μ_8 curves for all sets on the given axes."""
    tags = list(all_states.keys())
    for i, tag in enumerate(tags):
        arr = state_arrays(all_states[tag])
        label = tag.upper().replace("_", " ")
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        if len(tags) > 1:
            ax.plot(arr["mu"], arr["mu_e"], linewidth=2.2, color=color,
                    linestyle="-", label=rf"$\mu_e$ {label}")
            ax.plot(arr["mu"], arr["mu_8"], linewidth=2.2, color=color,
                    linestyle="--", label=rf"$\mu_8$ {label}")
        else:
            ax.plot(arr["mu"], arr["mu_e"], linewidth=2.2, color=color,
                    linestyle="-", label=r"$\mu_e$")
            ax.plot(arr["mu"], arr["mu_8"], linewidth=2.2, color=color,
                    linestyle="--", label=r"$\mu_8$")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"chemical potential $\;(\mathrm{MeV})$")
    ax.set_xlim(250.0, 600.0)
    ax.legend(fontsize=9)


def plot_phi_and_gap(
    all_states: dict[str, list[QMDStellarState]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    _phi_gap_on_ax(ax, all_states)
    ax.set_title(r"Neutral QMD stellar equilibrium fields")
    save_figure(plots_dir / "qmd_stellar_phi_gap_vs_mu.pdf")


def plot_neutrality_potentials(
    all_states: dict[str, list[QMDStellarState]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    _neutrality_pots_on_ax(ax, all_states)
    ax.set_title(r"Neutrality chemical potentials")
    save_figure(plots_dir / "qmd_stellar_mu_e_mu_8_vs_mu.pdf")


def plot_combined_condensates(
    all_states: dict[str, list[QMDStellarState]],
    plots_dir: Path,
) -> None:
    """Side-by-side: φ/gap on left, μ_e/μ_8 on right."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 5.0))
    _phi_gap_on_ax(ax_l, all_states)
    ax_l.set_title("Condensates")
    _neutrality_pots_on_ax(ax_r, all_states)
    ax_r.set_title("Neutrality potentials")
    fig.tight_layout()
    save_figure(plots_dir / "qmd_stellar_condensates.pdf")


def plot_gap_mismatch(
    all_states: dict[str, list[QMDStellarState]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    for i, (tag, states) in enumerate(all_states.items()):
        arr = state_arrays(states)
        label = tag.upper().replace("_", " ")
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        ax.plot(arr["mu"], arr["gap"], linewidth=2.2, color=color,
                label=rf"$g_\Delta\Delta_0$ {label}")
        ax.plot(arr["mu"], arr["delta_mu"], linewidth=2.2, color=color, linestyle="--",
                label=rf"$\delta\mu$ {label}")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"energy $\;(\mathrm{MeV})$")
    ax.set_title(r"2SC gap and mismatch")
    ax.set_xlim(250.0, 600.0)
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_gap_delta_mu_vs_mu.pdf")


def plot_residual_norm(
    all_states: dict[str, list[QMDStellarState]],
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    for i, (tag, states) in enumerate(all_states.items()):
        arr = state_arrays(states)
        color = SET_COLORS[min(i, len(SET_COLORS) - 1)]
        mask = np.isfinite(arr["residual_norm"]) & (arr["residual_norm"] > 0.0)
        ax.semilogy(arr["mu"][mask], arr["residual_norm"][mask],
                    linewidth=2.2, color=color, label=tag.upper().replace("_", " "))
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":", label=r"$1\,\mathrm{MeV}^3$")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$\sqrt{R_e^2+R_8^2}\;(\mathrm{MeV}^3)$")
    ax.set_title(r"Neutrality residual norm")
    ax.legend(fontsize=9)
    save_figure(plots_dir / "qmd_stellar_neutrality_residual_vs_mu.pdf")


def save_plots(
    all_states: dict[str, list[QMDStellarState]],
    plots_dir: Path,
) -> None:
    plot_phi_and_gap(all_states, plots_dir)
    plot_neutrality_potentials(all_states, plots_dir)
    plot_combined_condensates(all_states, plots_dir)
    plot_gap_mismatch(all_states, plots_dir)
    plot_residual_norm(all_states, plots_dir)


def summarize_states(tag: str, states: list[QMDStellarState]) -> None:
    onset = next((s for s in states if s.success and s.phase == "2SC"), None)
    failures = [s for s in states if not s.success]
    gapless = [
        s for s in states
        if s.success and s.phase == "2SC" and s.gap_minus_delta_mu_mev < 0.0
    ]
    normal_mu8 = [
        abs(s.mu_8_mev) for s in states
        if s.success and s.phase == "normal" and np.isfinite(s.mu_8_mev)
    ]

    print(f"\nSummary {tag.upper().replace("_", " ")}:")
    if onset is None:
        print("  No neutral 2SC onset found in scanned range.")
    else:
        print(
            f"  2SC onset: mu_q ≈ {onset.mu_q_mev:.2f} MeV, "
            f"gap ≈ {onset.gap_mev:.3f} MeV, "
            f"mu_e ≈ {onset.mu_e_mev:.3f} MeV, "
            f"mu_8 ≈ {onset.mu_8_mev:.3f} MeV"
        )
    if normal_mu8:
        print(f"  max |mu_8| in normal successful points: {max(normal_mu8):.3e} MeV")
    print(f"  gapless 2SC points: {len(gapless)}")
    print(f"  failed points: {len(failures)}")


def parameter_sets(include_set_b: bool) -> list[tuple[str, QMDParameters]]:
    sets = [("set_a", QMD_SET_A)]
    if include_set_b:
        sets.append(("set_b", QMD_SET_B))
    return sets


def main() -> None:
    args = parse_args()
    apply_plot_style()
    data_dir, plots_dir = output_directories(OUTPUT_DIR, "qmd_stellar_equilibrium")
    mu_values = np.linspace(args.mu_min, args.mu_max, args.num_points)

    all_states: dict[str, list[QMDStellarState]] = {}

    print("=" * 80)
    print("QMD Stellar Neutral Equilibrium Scan")
    print(f"  Omega_1_num = {'integral' if QMD_SET_A.include_omega_1_num else 'disabled'}")
    print("  Field bounds: phi in [1e-6, 2 f_pi], Delta in [0, f_pi]")
    print(f"  mu_q range: {args.mu_min:.1f} to {args.mu_max:.1f} MeV")
    print(f"  points: {args.num_points}")
    print("=" * 80)

    for tag, params in parameter_sets(args.include_set_b):
        print(f"\nScanning {tag.upper().replace("_", " ")} ...")
        print(params.describe())
        model = QMDStellarModel(params)
        states = scan_qmd_stellar(model, mu_values)
        all_states[tag] = states

        metadata = {
            "model": "QMD stellar neutral equilibrium scan",
            "omega_1_num_mode": "integral" if params.include_omega_1_num else "disabled",
            "residual_cutoff_mev": f"{params.residual_cutoff_mev:.1f}",
            "set": tag,
            "m_delta_mev": f"{params.m_delta_mev:.1f}",
            "g_delta_factor": f"{params.g_delta_factor:.4f}",
            "lambda_3_factor": f"{params.lambda_3_factor:.4f}",
            "lambda_delta_factor": f"{params.lambda_delta_factor:.4f}",
            "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
            "mu_min_mev": f"{args.mu_min:.1f}",
            "mu_max_mev": f"{args.mu_max:.1f}",
            "num_points": str(args.num_points),
            "delta_bound_note": "Delta <= f_pi to stay within the trusted condensate range of the truncated expansion",
        }
        save_states_table(data_dir / f"qmd_stellar_equilibrium_{tag}.txt", states, metadata)
        summarize_states(tag, states)

    save_plots(all_states, plots_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
