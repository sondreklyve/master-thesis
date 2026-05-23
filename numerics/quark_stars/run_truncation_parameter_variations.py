"""Truncation parameter variation figure — 3x1 panel.

For three parameter choices around QMD SET A, compares the full one-loop
calculation (Omega_{1,num} included) with the truncated calculation
(Omega_{1,num} = 0).  Shows diquark gap g_Delta*Delta_0 vs mu_q.

Parameter choices
-----------------
  Row 1: m_Delta = 400 MeV  (truncated window widens to ~86 MeV)
  Row 2: m_Delta = 700 MeV  (truncated 2SC phase disappears entirely)
  Row 3: lambda_Delta = lambda_0/8  (truncated window broadens moderately)

Produces
--------
  output/section2/data/section2_trunc_mdelta_400.txt
  output/section2/data/section2_trunc_mdelta_700.txt
  output/section2/data/section2_trunc_lamdelta_lam0div8.txt
  output/section2/plots/qmd_truncation_parameter_variations.pdf
  thesis/figures/quark_stars/qmd_truncation_parameter_variations.pdf

Usage
-----
  python -m numerics.quark_stars.run_truncation_parameter_variations
  python -m numerics.quark_stars.run_truncation_parameter_variations --plot-only
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

from .io import output_directories, save_table
from .plotting import apply_plot_style, save_figure
from .qmd_parameters import QMD_SET_A, QMDParameters
from .qmd_simple import QMDSimpleModel, QMDSimpleState


# ---------------------------------------------------------------------------
# Paths and scan parameters
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SECTION2_DIR = OUTPUT_DIR / "section2"
FULL_DATA_DIR = SECTION2_DIR / "data"
TRUNC_DATA_DIR = SECTION2_DIR / "data"
PLOTS_DIR = SECTION2_DIR / "plots"

THESIS_FIGURES_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "thesis" / "figures" / "quark_stars"
)

MU_MIN_MEV = 0.0
MU_MAX_MEV = 900.0
NUM_POINTS = 5000

# x-axis range for all plot panels
PLOT_MU_MIN = 200.0
PLOT_MU_MAX = 700.0

# Shared y-axis range across all panels (diquark gap in MeV)
GAP_YMIN = 0.0
GAP_YMAX = 300.0

# Colors matching qmd_benchmark_condensate_comparison.pdf
_COLOR = plt.cm.viridis(0.15)       # dark purple — full model
_COLOR_TRUNC = plt.cm.viridis(0.75)  # amber — truncated
_LABEL_FULL = r"Full ($\Omega_{1,\mathrm{num}}$ included)"
_LABEL_TRUNC = r"Truncated (no $\Omega_{1,\mathrm{num}}$)"
_LW = 2.2
_LW_TRUNC = 2.2

# 2SC phase threshold used for onset detection (MeV)
_GAP_THRESHOLD_MEV = 1.0


# ---------------------------------------------------------------------------
# Three parameter variations
# ---------------------------------------------------------------------------

_PARAMS_MDELTA_400 = replace(QMD_SET_A, m_delta_mev=400.0)
_PARAMS_MDELTA_700 = replace(QMD_SET_A, m_delta_mev=700.0)
_PARAMS_LAMDELTA_DIV8 = replace(QMD_SET_A, lambda_delta_factor=1.0 / 8.0)

RUNS: list[tuple[QMDParameters, str, str, str]] = [
    (
        _PARAMS_MDELTA_400,
        "section2_benchmark_mdelta_400.txt",
        "section2_trunc_mdelta_400.txt",
        r"$m_\Delta = 400~\mathrm{MeV}$",
    ),
    (
        _PARAMS_MDELTA_700,
        "section2_benchmark_mdelta_700.txt",
        "section2_trunc_mdelta_700.txt",
        r"$m_\Delta = 700~\mathrm{MeV}$",
    ),
    (
        _PARAMS_LAMDELTA_DIV8,
        "section2_benchmark_lamdelta_lam0div8.txt",
        "section2_trunc_lamdelta_lam0div8.txt",
        r"$\lambda_\Delta = \lambda_0/8$",
    ),
]


# ---------------------------------------------------------------------------
# Scan helpers (mirrors run_qmd_benchmark.py)
# ---------------------------------------------------------------------------


def _scan(model: QMDSimpleModel, mu_values: np.ndarray) -> list[QMDSimpleState]:
    states: list[QMDSimpleState] = []
    prev: tuple[float, float] | None = None
    for mu in mu_values:
        s = model.solve_mean_fields(float(mu), initial_guess=prev)
        states.append(s)
        prev = (s.phi_mev, s.delta_mev)
    return states


def _eos(states: list[QMDSimpleState]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omegas = np.array([s.omega_min_mev4 for s in states])
    mus = np.array([s.mu_q_mev for s in states])
    pressures = -(omegas - omegas[0])
    n_q = np.gradient(pressures, mus, edge_order=2)
    eps = -pressures + mus * n_q
    dp_dmu = np.gradient(pressures, mus, edge_order=2)
    deps_dmu = np.gradient(eps, mus, edge_order=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs2 = np.where(deps_dmu > 0.0, dp_dmu / deps_dmu, np.nan)
    return pressures, n_q, eps, cs2


def _save_scan(
    path: Path,
    states: list[QMDSimpleState],
    pressures: np.ndarray,
    n_q: np.ndarray,
    eps: np.ndarray,
    cs2: np.ndarray,
    params: QMDParameters,
    description: str,
) -> None:
    columns = [
        "mu_q_mev", "phi_mev", "delta_mev", "gap_mev", "phase_2sc",
        "omega_min_mev4", "pressure_mev4", "n_q_mev3",
        "energy_density_mev4", "cs2", "success",
    ]
    metadata = {
        "description": description,
        "include_omega_1_num": str(params.include_omega_1_num),
        "m_delta_mev": f"{params.m_delta_mev:.1f}",
        "g_delta_factor": f"{params.g_delta_factor:.4f}",
        "lambda_3_factor": f"{params.lambda_3_factor:.4f}",
        "lambda_delta_factor": f"{params.lambda_delta_factor:.4f}",
        "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
        "residual_cutoff_mev": f"{params.residual_cutoff_mev:.1f}",
        "mu_min_mev": f"{MU_MIN_MEV:.1f}",
        "mu_max_mev": f"{MU_MAX_MEV:.1f}",
        "num_points": str(NUM_POINTS),
    }
    data = np.array([
        [
            s.mu_q_mev, s.phi_mev, s.delta_mev, s.gap_mev,
            float(s.phase == "2SC"),
            s.omega_min_mev4, float(pressures[i]), float(n_q[i]),
            float(eps[i]), float(cs2[i]), float(s.success),
        ]
        for i, s in enumerate(states)
    ])
    save_table(path, columns, data, metadata)


def _load_gap_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_q_mev, gap_mev) arrays from a saved scan file."""
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 3]


def _compute_and_save_truncated(
    params_full: QMDParameters,
    trunc_path: Path,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run truncated scan and save; return (mu_q, gap) arrays."""
    params_trunc = replace(params_full, include_omega_1_num=False)
    mu = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)
    print(f"  Running truncated scan for {label} ({NUM_POINTS} pts) ...")
    model = QMDSimpleModel(params_trunc)
    states = _scan(model, mu)
    pressures, n_q, eps, cs2 = _eos(states)
    trunc_path.parent.mkdir(parents=True, exist_ok=True)
    _save_scan(
        trunc_path, states, pressures, n_q, eps, cs2,
        params_trunc,
        f"Truncated (no Omega_1_num): {label}",
    )
    print(f"  Saved {trunc_path.name}")
    onset_mu = next(
        (s.mu_q_mev for s in states if s.gap_mev >= _GAP_THRESHOLD_MEV), None
    )
    end_mu = None
    for s in reversed(states):
        if s.gap_mev >= _GAP_THRESHOLD_MEV:
            end_mu = s.mu_q_mev
            break
    if onset_mu is not None and end_mu is not None:
        print(f"  Truncated 2SC window: [{onset_mu:.1f}, {end_mu:.1f}] MeV "
              f"(width {end_mu - onset_mu:.1f} MeV)")
    else:
        print("  No truncated 2SC phase found.")
    gap = np.array([s.gap_mev for s in states])
    return mu, gap


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_variations(
    run_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    plots_dir: Path,
) -> None:
    """3x1 panel figure: diquark gap (full vs truncated) for three parameter choices."""
    n = len(run_data)
    fig, axes = plt.subplots(n, 1, figsize=(7.6, 4.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (mu_f, gap_f, mu_t, gap_t, panel_label) in enumerate(run_data):
        ax = axes[i]

        # Plot full model
        mask_f = (mu_f >= PLOT_MU_MIN) & (mu_f <= PLOT_MU_MAX)
        ax.plot(
            mu_f[mask_f], gap_f[mask_f],
            lw=_LW, color=_COLOR,
            label=_LABEL_FULL,
        )

        # Plot truncated model
        mask_t = (mu_t >= PLOT_MU_MIN) & (mu_t <= PLOT_MU_MAX)
        ax.plot(
            mu_t[mask_t], gap_t[mask_t],
            lw=_LW_TRUNC, color=_COLOR_TRUNC,
            label=_LABEL_TRUNC,
        )

        ax.set_xlim(PLOT_MU_MIN, PLOT_MU_MAX)
        ax.set_ylim(GAP_YMIN, GAP_YMAX)
        ax.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")

        # Panel label in upper-left
        ax.text(
            0.03, 0.95, panel_label,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=13,
        )

        if i == n - 1:
            ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")

        ax.legend(loc="upper right", fontsize=11)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "qmd_truncation_parameter_variations.pdf"
    save_figure(out_path)
    print(f"  Saved {out_path}")

    thesis_path = THESIS_FIGURES_DIR / "qmd_truncation_parameter_variations.pdf"
    THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(out_path, thesis_path)
    print(f"  Copied to {thesis_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip scans; reload saved data and regenerate the plot only.",
    )
    args = parser.parse_args()

    apply_plot_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TRUNC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []

    for params_full, full_file, trunc_file, panel_label in RUNS:
        print(f"\n--- {panel_label} ---")

        full_path = FULL_DATA_DIR / full_file
        trunc_path = TRUNC_DATA_DIR / trunc_file

        # Load full model data
        if not full_path.exists():
            raise FileNotFoundError(
                f"Full model data not found: {full_path}\n"
                "Run run_section2_sweep.py first."
            )
        mu_f, gap_f = _load_gap_data(full_path)
        onset_full = next(
            (mu_f[j] for j in range(len(mu_f)) if gap_f[j] >= _GAP_THRESHOLD_MEV),
            None,
        )
        if onset_full is not None:
            print(f"  Full model 2SC onset: {onset_full:.1f} MeV")

        # Load or compute truncated data
        if trunc_path.exists() and args.plot_only:
            mu_t, gap_t = _load_gap_data(trunc_path)
            print(f"  Loaded truncated data from {trunc_file}")
        elif trunc_path.exists() and not args.plot_only:
            print(f"  Truncated data already exists ({trunc_file}); loading ...")
            mu_t, gap_t = _load_gap_data(trunc_path)
        else:
            if args.plot_only:
                raise FileNotFoundError(
                    f"Truncated data not found: {trunc_path}\n"
                    "Run without --plot-only first."
                )
            mu_t, gap_t = _compute_and_save_truncated(params_full, trunc_path, panel_label)

        run_data.append((mu_f, gap_f, mu_t, gap_t, panel_label))

    print("\nGenerating figure ...")
    _plot_variations(run_data, PLOTS_DIR)


if __name__ == "__main__":
    main()
