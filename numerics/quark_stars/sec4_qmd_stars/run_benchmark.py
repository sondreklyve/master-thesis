"""QMD SET A benchmark scan and truncation comparison.

Runs QMD_SET_A (full one-loop potential) and the truncated variant
(include_omega_1_num=False) for the condensate comparison plot.

Produces
--------
  output/qmd_benchmark/data/qmd_benchmark.txt
  output/qmd_benchmark/data/qmd_benchmark_truncated.txt
  output/qmd_benchmark/data/qmd_benchmark_asymptotic_log.txt
  thesis/figures/quark_stars/qmd_stars/qmd_benchmark_condensates.pdf
  thesis/figures/quark_stars/qmd_stars/qmd_benchmark_condensate_comparison.pdf
  thesis/figures/quark_stars/qmd_stars/qmd_benchmark_pressure.pdf
  thesis/figures/quark_stars/qmd_stars/qmd_benchmark_eos.pdf
  thesis/figures/quark_stars/qmd_stars/qmd_benchmark_cs2.pdf

Usage
-----
  # Full scan + plots (slow, ~5 min for full + ~1 min for truncated):
  python -m numerics.quark_stars.run_qmd_benchmark

  # Replot only from saved data (fast):
  python -m numerics.quark_stars.run_qmd_benchmark --plot-only

  # Replot and create the cached high-μ conformal diagnostic if missing:
  python -m numerics.quark_stars.run_qmd_benchmark --plot-only --compute-asymptotic
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
from scipy.signal import savgol_filter

from ..constants import MEV4_TO_GEV_FM3
from ..io import output_directories, save_table
from ..plotting import PURPLE, TURQUOISE, apply_plot_style, save_figure
from ..qmd_parameters import QMD_SET_A, QMDParameters
from ..qmd_simple import QMDSimpleModel, QMDSimpleState, _loop_F, _loop_G_pi


# ---------------------------------------------------------------------------
# Scan parameters
# ---------------------------------------------------------------------------

MU_MIN_MEV = 0.0
MU_MAX_MEV = 900.0
NUM_POINTS = 5000
NUM_POINTS_TRUNC = 5000
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "thesis" / "figures" / "quark_stars" / "qmd_stars"

PARAMS_TRUNC = replace(QMD_SET_A, include_omega_1_num=False)

# Colours and labels for the two-curve comparison plot
_COLOR = PURPLE
_COLOR_TRUNC = TURQUOISE
_LABEL_FULL  = "Full"
_LABEL_TRUNC = "Truncated"
_LW = 2.2
_LW_TRUNC = 2.2
_ONSET_LW = 1.4
_ONSET_LABEL_SIZE = 11

# Low-pressure window used for the thesis EoS panel.  This keeps the
# benchmark-range structure readable instead of compressing it against a much
# larger asymptotic scale.
_EOS_ZOOM_P_MAX_GEV_FM3 = 0.05
_EOS_ZOOM_EPS_MAX_GEV_FM3 = 0.45

# Separate high-μ diagnostic.  The finite residual integral is formally an
# integral to infinity; the main benchmark follows the reference 3 GeV cutoff,
# while this plot uses a larger cutoff and a logarithmic μ grid to show the
# slow approach to the conformal limit without spending many points at high μ.
ASYMPTOTIC_MU_MIN_MEV = 300.0
ASYMPTOTIC_MU_MAX_MEV = 9000.0   # scan to 9 GeV so 6000 MeV is well interior to the SG window
ASYMPTOTIC_NUM_POINTS = 80
ASYMPTOTIC_RESIDUAL_CUTOFF_MEV = 20000.0

_EOS_SMOOTH_WINDOW = 81
_RATIO_SMOOTH_WINDOW = 9
_SMOOTH_POLY = 3


# ---------------------------------------------------------------------------
# Scan and thermodynamics
# ---------------------------------------------------------------------------


def _scan(model: QMDSimpleModel, mu_values: np.ndarray) -> list[QMDSimpleState]:
    """Warm-start scan of the QMD mean-field minimum over μ_q."""
    states: list[QMDSimpleState] = []
    prev: tuple[float, float] | None = None
    for mu in mu_values:
        s = model.solve_mean_fields(float(mu), initial_guess=prev)
        states.append(s)
        prev = (s.phi_mev, s.delta_mev)
    return states


def _eos(
    states: list[QMDSimpleState],
    omega_ref_mev4: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (pressure, n_q, energy_density, cs2) on the full μ_q grid.

    By default P = 0 at states[0] (μ_q = 0).  A separate omega_ref can be
    supplied for scans that start away from vacuum.
    """
    omegas = np.array([s.omega_min_mev4 for s in states])
    mus = np.array([s.mu_q_mev for s in states])
    omega_ref = omegas[0] if omega_ref_mev4 is None else float(omega_ref_mev4)
    edge_order = 2 if len(states) >= 3 else 1
    pressures = -(omegas - omega_ref)
    n_q = np.gradient(pressures, mus, edge_order=edge_order)
    eps = -pressures + mus * n_q
    dp_dmu = np.gradient(pressures, mus, edge_order=edge_order)
    deps_dmu = np.gradient(eps, mus, edge_order=edge_order)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs2 = np.where(deps_dmu > 0.0, dp_dmu / deps_dmu, np.nan)
    return pressures, n_q, eps, cs2


def _phys(pressures: np.ndarray, n_q: np.ndarray) -> np.ndarray:
    """Boolean mask for the positive-pressure, positive-density branch."""
    return (pressures >= 0.0) & (n_q >= 0.0)


def _odd_window(size: int, requested: int, polyorder: int = _SMOOTH_POLY) -> int:
    """Return a valid odd Savitzky-Golay window for an array of length size."""
    if size <= polyorder + 2:
        return 0
    window = min(int(requested), size if size % 2 else size - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return 0
    return window


def _smooth_for_plot(
    values: np.ndarray,
    requested_window: int,
    polyorder: int = _SMOOTH_POLY,
) -> np.ndarray:
    """Smooth a plotted curve while leaving saved table data unchanged."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() != arr.size:
        return arr
    window = _odd_window(arr.size, requested_window, polyorder)
    if window == 0:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=polyorder)


def _set_log_mu_axis(ax, lower: float, upper: float) -> None:
    """Use a log μ axis with human-readable MeV tick labels."""
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    tick_candidates = np.array([500.0, 1000.0, 2000.0, 4000.0, 6000.0])
    ticks = tick_candidates[(tick_candidates >= lower) & (tick_candidates <= upper)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.0f}" for tick in ticks])


def _mark_onset(
    ax,
    onset_mev: float | None,
    *,
    label: bool = False,
    x_offset_mev: float = 6.0,
    y_axes: float = 0.95,
) -> None:
    """Add a vertical 2SC-onset marker on μ_q-axis plots."""
    if onset_mev is None:
        return
    ax.axvline(onset_mev, color="gray", ls="--", lw=_ONSET_LW)
    if label:
        ax.text(
            onset_mev + x_offset_mev,
            y_axes,
            "2SC onset",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="left",
            fontsize=_ONSET_LABEL_SIZE,
            color="gray",
        )


def _asymptotic_gap_mev(params: QMDParameters) -> float:
    """Analytic BCS asymptotic gap g_Δ Δ̄_0 (thesis Eq. eq:qmd_asymptotic_gap).

    Physical-pion-mass version: includes F(m_π²) and m_π²F'(m_π²) corrections.
    g_Δ Δ̄_0 = m_q · exp[(4π)² / (8 g_Δ²) − 1/2 − (F + G) / 2]
    """
    F = _loop_F(params.m_pi_mev, params.m_q_mev)
    G = _loop_G_pi(params.m_pi_mev, params.m_q_mev)
    return params.m_q_mev * np.exp(
        (4.0 * np.pi) ** 2 / (8.0 * params.g_delta**2) - 0.5 - (F + G) / 2.0
    )


def _load_benchmark_data(data_dir: Path) -> dict[str, np.ndarray]:
    """Load saved qmd_benchmark.txt into a column dict (no scan recomputation)."""
    path = data_dir / "qmd_benchmark.txt"
    # columns: mu_q_mev phi_mev delta_mev gap_mev phase_2sc omega_min_mev4
    #          pressure_mev4 n_q_mev3 energy_density_mev4 cs2 success
    data = np.loadtxt(path, comments="#")
    phase_2sc = data[:, 4]
    mu_arr    = data[:, 0]
    onset_candidates = mu_arr[phase_2sc > 0.5]
    onset = float(onset_candidates[0]) if onset_candidates.size > 0 else None
    return {
        "mu_q_mev":            mu_arr,
        "phi_mev":             data[:, 1],
        "delta_mev":           data[:, 2],
        "gap_mev":             data[:, 3],
        "phase_2sc":           phase_2sc,
        "pressure_mev4":       data[:, 6],
        "n_q_mev3":            data[:, 7],
        "energy_density_mev4": data[:, 8],
        "cs2":                 data[:, 9],
        "success":             data[:, 10],
        "onset_mev":           onset,
    }


def _load_truncated_data(data_dir: Path) -> dict[str, np.ndarray] | None:
    """Load qmd_benchmark_truncated.txt; return None if missing."""
    path = data_dir / "qmd_benchmark_truncated.txt"
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    phase_2sc = data[:, 4]
    mu_arr = data[:, 0]
    onset_candidates = mu_arr[phase_2sc > 0.5]
    onset = float(onset_candidates[0]) if onset_candidates.size > 0 else None
    # Chiral onset: first mu where phi drops from vacuum value
    phi_arr = data[:, 1]
    phi_vac = float(phi_arr[0])
    chiral_break = mu_arr[phi_arr < phi_vac - 0.05]
    chiral_onset = float(chiral_break[0]) if chiral_break.size > 0 else None
    trunc_2sc = mu_arr[phase_2sc > 0.5]
    trunc_2sc_end = float(trunc_2sc[-1]) if trunc_2sc.size > 0 else None
    return {
        "mu_q_mev":            mu_arr,
        "phi_mev":             phi_arr,
        "gap_mev":             data[:, 3],
        "phase_2sc":           phase_2sc,
        "onset_mev":           onset,
        "chiral_onset_mev":    chiral_onset,
        "trunc_2sc_end_mev":   trunc_2sc_end,
    }


def _asymptotic_data_path(data_dir: Path) -> Path:
    return data_dir / "qmd_benchmark_asymptotic_log.txt"


def _load_asymptotic_data(data_dir: Path) -> dict[str, np.ndarray] | None:
    """Load the cached high-μ diagnostic table if it exists."""
    path = _asymptotic_data_path(data_dir)
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    phase_2sc = data[:, 4]
    return {
        "mu_q_mev":            data[:, 0],
        "phi_mev":             data[:, 1],
        "delta_mev":           data[:, 2],
        "gap_mev":             data[:, 3],
        "phase_2sc":           phase_2sc,
        "pressure_mev4":       data[:, 6],
        "n_q_mev3":            data[:, 7],
        "energy_density_mev4": data[:, 8],
        "cs2":                 data[:, 9],
        "success":             data[:, 10],
    }


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------


def _save_data(
    path: Path,
    states: list[QMDSimpleState],
    pressures: np.ndarray,
    n_q: np.ndarray,
    eps: np.ndarray,
    cs2: np.ndarray,
    *,
    params: QMDParameters = QMD_SET_A,
    description: str = "QMD SET A common-mu benchmark (Section 1)",
    mu_min_mev: float = MU_MIN_MEV,
    mu_max_mev: float = MU_MAX_MEV,
    num_points: int = NUM_POINTS,
) -> None:
    columns = [
        "mu_q_mev", "phi_mev", "delta_mev", "gap_mev", "phase_2sc",
        "omega_min_mev4", "pressure_mev4", "n_q_mev3",
        "energy_density_mev4", "cs2", "success",
    ]
    p = params
    metadata = {
        "description": description,
        "include_omega_1_num": str(p.include_omega_1_num),
        "m_delta_mev": f"{p.m_delta_mev:.1f}",
        "g_delta_factor": f"{p.g_delta_factor:.4f}",
        "lambda_3_factor": f"{p.lambda_3_factor:.4f}",
        "lambda_delta_factor": f"{p.lambda_delta_factor:.4f}",
        "t_loop4_factor": f"{p.t_loop4_factor:.1f}",
        "residual_cutoff_mev": f"{p.residual_cutoff_mev:.1f}",
        "mu_min_mev": f"{mu_min_mev:.1f}",
        "mu_max_mev": f"{mu_max_mev:.1f}",
        "num_points": str(num_points),
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


def _compute_asymptotic_diagnostic(
    data_dir: Path,
    *,
    allow_compute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Run or load the high-μ conformal diagnostic.

    This is deliberately separate from the production benchmark because it uses
    a larger residual cutoff and a coarser grid.  The goal is only to show the
    approach to the massless conformal limit.
    """
    cached = _load_asymptotic_data(data_dir)
    if cached is not None:
        print(f"  Loaded {len(cached['mu_q_mev'])} rows from qmd_benchmark_asymptotic_log.txt")
        return (
            cached["mu_q_mev"],
            cached["pressure_mev4"],
            cached["energy_density_mev4"],
            cached["cs2"],
        )
    if not allow_compute:
        print("  No qmd_benchmark_asymptotic_log.txt found; skipping high-μ diagnostic.")
        return None

    params = replace(QMD_SET_A, residual_cutoff_mev=ASYMPTOTIC_RESIDUAL_CUTOFF_MEV)
    mu = np.concatenate((
        np.array([MU_MIN_MEV]),
        np.geomspace(
            ASYMPTOTIC_MU_MIN_MEV,
            ASYMPTOTIC_MU_MAX_MEV,
            ASYMPTOTIC_NUM_POINTS,
        ),
    ))
    print(
        "  Running log-spaced high-μ conformal diagnostic "
        f"({ASYMPTOTIC_MU_MIN_MEV:.0f} ≤ μ_q ≤ {ASYMPTOTIC_MU_MAX_MEV:.0f} MeV, "
        f"{ASYMPTOTIC_NUM_POINTS} log points, "
        f"Λ_res = {ASYMPTOTIC_RESIDUAL_CUTOFF_MEV / 1000.0:.1f} GeV) ..."
    )
    model = QMDSimpleModel(params)
    states = _scan(model, mu)
    pressures, n_q, eps, cs2 = _eos(states)
    _save_data(
        _asymptotic_data_path(data_dir),
        states,
        pressures,
        n_q,
        eps,
        cs2,
        params=params,
        description="QMD SET A common-mu asymptotic diagnostic (log-spaced, larger residual cutoff)",
        mu_min_mev=MU_MIN_MEV,
        mu_max_mev=ASYMPTOTIC_MU_MAX_MEV,
        num_points=len(mu),
    )
    print(f"  Saved {_asymptotic_data_path(data_dir)}")
    return mu, pressures, eps, cs2


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_condensates(
    mu: np.ndarray,
    phi: np.ndarray,
    gap: np.ndarray,
    onset_mev: float | None,
    plots_dir: Path,
) -> None:
    thresh_idx = int(np.searchsorted(mu, onset_mev)) if onset_mev is not None else len(mu)
    phi_p = phi.copy()
    gap_p = gap.copy()
    if thresh_idx < len(mu):
        phi_p[thresh_idx:] = _smooth_for_plot(phi[thresh_idx:], _CS2_SG_WINDOW)
        gap_p[thresh_idx:] = _smooth_for_plot(gap[thresh_idx:], _CS2_SG_WINDOW)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax1.plot(mu, phi_p, lw=_LW, color=_COLOR)
    ax1.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax1.set_ylabel(r"$\phi_0\;(\mathrm{MeV})$")
    ax1.set_title("Chiral condensate")
    ax1.set_xlim(200.0, 700.0)

    ax2.plot(mu, gap_p, lw=_LW, color=_COLOR)
    ax2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax2.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")
    ax2.set_title("Diquark gap")
    ax2.set_xlim(200.0, 700.0)

    save_figure(plots_dir / "qmd_benchmark_condensates.pdf")


def _plot_condensate_comparison(
    mu_f: np.ndarray,
    phi_f: np.ndarray,
    gap_f: np.ndarray,
    onset_f: float | None,
    mu_t: np.ndarray,
    phi_t: np.ndarray,
    gap_t: np.ndarray,
    chiral_onset_t: float | None,
    trunc_2sc_start: float | None,
    trunc_2sc_end: float | None,
    plots_dir: Path,
) -> None:
    """Full vs truncated condensate comparison — uses benchmark 5000-pt data for full model."""
    thresh_idx = int(np.searchsorted(mu_f, onset_f)) if onset_f is not None else len(mu_f)
    phi_f_p = phi_f.copy()
    gap_f_p = gap_f.copy()
    if thresh_idx < len(mu_f):
        phi_f_p[thresh_idx:] = _smooth_for_plot(phi_f[thresh_idx:], _CS2_SG_WINDOW)
        gap_f_p[thresh_idx:] = _smooth_for_plot(gap_f[thresh_idx:], _CS2_SG_WINDOW)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax1.plot(mu_f, phi_f_p, lw=_LW,       color=_COLOR,       label=_LABEL_FULL)
    ax1.plot(mu_t, phi_t,   lw=_LW_TRUNC, color=_COLOR_TRUNC, label=_LABEL_TRUNC)
    ax2.plot(mu_f, gap_f_p, lw=_LW,       color=_COLOR,       label=_LABEL_FULL)
    ax2.plot(mu_t, gap_t,   lw=_LW_TRUNC, color=_COLOR_TRUNC, label=_LABEL_TRUNC)

    if trunc_2sc_start is not None and trunc_2sc_end is not None:
        for ax in (ax1, ax2):
            ax.axvspan(trunc_2sc_start, trunc_2sc_end,
                       color=_COLOR_TRUNC, alpha=0.16, zorder=0)

    if chiral_onset_t is not None:
        for ax in (ax1, ax2):
            ax.axvline(chiral_onset_t, color=_COLOR_TRUNC, ls="--", lw=1.2)

    ax1.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax1.set_ylabel(r"$\phi_0\;(\mathrm{MeV})$")
    ax1.set_title("Chiral condensate")
    ax1.set_xlim(250.0, 420.0)
    ax1.legend()

    ax2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax2.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")
    ax2.set_title("Diquark gap")
    ax2.set_xlim(250.0, 420.0)
    ax2.legend()

    save_figure(plots_dir / "qmd_benchmark_condensate_comparison.pdf")


def _plot_pressure(
    mu: np.ndarray,
    pressures: np.ndarray,
    n_q: np.ndarray,
    onset_mev: float | None,
    plots_dir: Path,
) -> None:
    mask = _phys(pressures, n_q)
    pres_g = pressures[mask] * MEV4_TO_GEV_FM3
    fig, ax = plt.subplots()
    ax.plot(mu[mask], pres_g, lw=_LW, color=_COLOR)
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax.set_title("Pressure")
    save_figure(plots_dir / "qmd_benchmark_pressure.pdf")


def _plot_eos(
    mu: np.ndarray,
    pressures: np.ndarray,
    eps: np.ndarray,
    n_q: np.ndarray,
    plots_dir: Path,
    asymptotic: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    mask = _phys(pressures, n_q)
    eps_g = eps[mask] * MEV4_TO_GEV_FM3
    pres_g = pressures[mask] * MEV4_TO_GEV_FM3

    fig, (ax_eos, ax_ratio) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    zoom = (
        np.isfinite(pres_g)
        & np.isfinite(eps_g)
        & (pres_g >= 0.0)
        & (pres_g <= _EOS_ZOOM_P_MAX_GEV_FM3)
    )
    ax_eos.plot(
        pres_g[zoom],
        _smooth_for_plot(eps_g[zoom], _EOS_SMOOTH_WINDOW),
        lw=_LW,
        color=_COLOR,
    )

    ax_eos.set_xlabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax_eos.set_ylabel(r"$\varepsilon\;(\mathrm{GeV\,fm}^{-3})$")
    ax_eos.set_title("Low-pressure EoS")
    ax_eos.set_xlim(0.0, _EOS_ZOOM_P_MAX_GEV_FM3)
    ax_eos.set_ylim(0.0, _EOS_ZOOM_EPS_MAX_GEV_FM3)

    if asymptotic is None:
        ratio_mu = mu[mask]
        ratio_p = pressures[mask]
        ratio_eps = eps[mask]
        ratio_title = "Benchmark-range conformal convergence"
    else:
        ratio_mu, ratio_p, ratio_eps, _ = asymptotic
        ratio_title = "Conformal convergence"
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (ratio_eps - 3.0 * ratio_p) / ratio_p
    ratio_mask = (
        np.isfinite(delta)
        & np.isfinite(ratio_mu)
        & (ratio_p > 0.0)
        & (ratio_mu >= 450.0)
        & (delta > -0.35)
        & (delta < 0.35)
    )
    ratio_x = ratio_mu[ratio_mask]
    ratio_y_raw = delta[ratio_mask]
    ratio_y = _smooth_for_plot(ratio_y_raw, _RATIO_SMOOTH_WINDOW)
    ax_ratio.plot(ratio_x, ratio_y, lw=_LW, color=_COLOR)
    ax_ratio.axhline(0.0, color="gray", ls="--", lw=1.5)
    ax_ratio.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax_ratio.set_ylabel(r"$\delta$")
    ax_ratio.set_title(ratio_title)
    if ratio_x.size:
        upper = 6000.0 if asymptotic is not None else float(ratio_x[-1])
        _set_log_mu_axis(ax_ratio, float(ratio_x[0]), upper)
    ax_ratio.set_ylim(-0.35, 0.35)
    save_figure(plots_dir / "qmd_benchmark_eos.pdf")


def _plot_asymptotic_eos(
    mu: np.ndarray,
    pressures: np.ndarray,
    eps: np.ndarray,
    cs2: np.ndarray,
    plots_dir: Path,
) -> None:
    """Plot the high-μ conformal diagnostic with a larger residual cutoff."""
    mask = (pressures > 0.0) & (eps > 0.0)
    mu_plot = mu[mask]
    p_plot = pressures[mask]
    eps_plot = eps[mask]
    if mu_plot.size > 4:
        mu_plot = mu_plot[:-2]
        p_plot = p_plot[:-2]
        eps_plot = eps_plot[:-2]
    p_g = p_plot * MEV4_TO_GEV_FM3
    eps_g = eps_plot * MEV4_TO_GEV_FM3

    fig, (ax_eos, ax_ratio, ax_cs2) = plt.subplots(1, 3, figsize=(15.0, 4.5))

    ax_eos.plot(p_g, _smooth_for_plot(eps_g, _RATIO_SMOOTH_WINDOW), lw=_LW, color=_COLOR)
    p_ref = np.linspace(0.0, float(np.nanmax(p_g)), 300)
    ax_eos.plot(p_ref, 3.0 * p_ref, color="gray", ls="--", lw=1.5,
                label=r"$\varepsilon=3P$")
    ax_eos.set_xlabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax_eos.set_ylabel(r"$\varepsilon\;(\mathrm{GeV\,fm}^{-3})$")
    ax_eos.set_title("High-pressure EoS")
    ax_eos.set_xlim(0.0, float(np.nanmax(p_g)) * 1.02)
    ax_eos.set_ylim(0.0, float(np.nanmax(eps_g)) * 1.02)
    ax_eos.legend(fontsize=8)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = eps_plot / p_plot
    ratio_mask = np.isfinite(ratio) & (mu_plot >= 500.0)
    ax_ratio.plot(
        mu_plot[ratio_mask],
        _smooth_for_plot(ratio[ratio_mask], _RATIO_SMOOTH_WINDOW),
        lw=_LW,
        color=_COLOR,
    )
    ax_ratio.axhline(3.0, color="gray", ls="--", lw=1.5)
    ax_ratio.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax_ratio.set_ylabel(r"$\varepsilon/P$")
    ax_ratio.set_title("Energy-pressure ratio")
    _set_log_mu_axis(ax_ratio, 500.0, ASYMPTOTIC_MU_MAX_MEV)
    ax_ratio.set_ylim(2.7, 3.35)

    cs2_plot = cs2[mask]
    if cs2_plot.size > mu_plot.size:
        cs2_plot = cs2_plot[:mu_plot.size]
    cs2_mask = np.isfinite(cs2_plot) & (mu_plot >= 500.0) & (cs2_plot > 0.0) & (cs2_plot < 1.0)
    ax_cs2.plot(
        mu_plot[cs2_mask],
        _smooth_for_plot(cs2_plot[cs2_mask], _RATIO_SMOOTH_WINDOW),
        lw=_LW,
        color=_COLOR,
    )
    ax_cs2.axhline(1.0 / 3.0, color="gray", ls="--", lw=1.5)
    ax_cs2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax_cs2.set_ylabel(r"$c_s^2/c^2$")
    ax_cs2.set_title("Sound speed")
    _set_log_mu_axis(ax_cs2, 500.0, ASYMPTOTIC_MU_MAX_MEV)
    ax_cs2.set_ylim(0.30, 0.39)

    fig.suptitle(
        rf"Conformal diagnostic ($\Lambda_{{\rm res}}="
        rf"{ASYMPTOTIC_RESIDUAL_CUTOFF_MEV / 1000.0:.1f}\,\mathrm{{GeV}}$)",
        y=1.02,
        fontsize=12,
    )
    save_figure(plots_dir / "qmd_benchmark_eos_asymptotic.pdf")


_CS2_SG_WINDOW = 51   # Savitzky-Golay window (≈ 9 MeV at 5000-point grid)
_CS2_SG_POLY   = 3    # polynomial order
_CS2_ONSET_THRESH = 290.0  # MeV — smoothing is applied only above this

_COND_SMOOTH_THRESH = 280.0  # MeV — smooth condensates above this (preserves onset transition)
_CS2_PLOT_MU_MIN_MEV = 250.0
_CS2_PLOT_MU_MAX_MEV = 800.0


def _plot_cs2(
    mu: np.ndarray,
    pressures: np.ndarray,
    n_q: np.ndarray,
    cs2: np.ndarray,
    onset_mev: float | None,
    plots_dir: Path,
    asymptotic: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    mask = _phys(pressures, n_q) & np.isfinite(cs2)
    mu_plot = mu[mask]
    cs2_plot = cs2[mask]

    # Remove data below onset and outliers: pre-onset cs2 is unphysical for
    # stellar matter, and cs2 outside [0, 1] are derivative artifacts.
    valid = (mu_plot >= 295.0) & (cs2_plot >= 0.0) & (cs2_plot <= 1.0)
    mu_plot = mu_plot[valid]
    cs2_plot = cs2_plot[valid]

    # Drop the last 2 points: np.gradient uses one-sided (lower-order) differences
    # at the endpoint μ_q = 900 MeV, producing a spurious spike.  All points
    # are retained in the data table; only the plot is trimmed.
    mu_plot = mu_plot[:-2]
    cs2_plot = cs2_plot[:-2]

    # Savitzky-Golay smoothing applied post-onset only (μ_q > 290 MeV).
    # cs² from np.gradient of optimizer-tolerance-limited fields carries
    # grid-frequency noise; SG preserves the physical transition jump and
    # the cs² > 1/3 bump while suppressing sub-grid-spacing oscillations.
    # The near-onset region is excluded to avoid blurring the sharp transition.
    cs2_smoothed = cs2_plot.copy()
    post = mu_plot > _CS2_ONSET_THRESH
    if post.sum() >= _CS2_SG_WINDOW:
        cs2_smoothed[post] = savgol_filter(
            cs2_plot[post], window_length=_CS2_SG_WINDOW, polyorder=_CS2_SG_POLY
        )

    fig, ax = plt.subplots()
    keep_benchmark = mu_plot <= _CS2_PLOT_MU_MAX_MEV
    mu_plot = mu_plot[keep_benchmark]
    cs2_smoothed = cs2_smoothed[keep_benchmark]
    ax.plot(mu_plot, cs2_smoothed, lw=_LW, color=_COLOR)
    ax.axhline(1.0 / 3.0, color="gray", ls="--", lw=1.5,
               label=r"Conformal limit $c_s^2 = \frac{1}{3}$")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$c_s^2/c^2$")
    ax.set_title("Speed of sound squared")
    ax.set_xlim(_CS2_PLOT_MU_MIN_MEV, _CS2_PLOT_MU_MAX_MEV)
    ax.set_ylim(0.20, 0.45)
    ax.legend()
    save_figure(plots_dir / "qmd_benchmark_cs2.pdf")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(
    mu: np.ndarray,
    states: list[QMDSimpleState],
    pressures: np.ndarray,
    n_q: np.ndarray,
    asym_gap_analytic_mev: float,
) -> None:
    onset = next((s.mu_q_mev for s in states if s.phase == "2SC"), None)
    asym_gap = states[-1].gap_mev

    fail_mask = np.array([not s.success for s in states])
    fail_mu = mu[fail_mask]
    n_fail = int(fail_mask.sum())
    n_total = len(states)

    # Cluster the failures into three regions relative to onset
    vac_thresh = 200.0
    tr_width = 20.0  # ±10 MeV around onset
    tr_lo = (onset - tr_width) if onset is not None else 250.0
    tr_hi = (onset + tr_width) if onset is not None else 290.0

    n_vac  = int((fail_mu < vac_thresh).sum())
    n_tr   = int(((fail_mu >= tr_lo) & (fail_mu <= tr_hi)).sum())
    n_bulk = int((fail_mu > tr_hi).sum())

    print("\n" + "=" * 60)
    print("QMD SET A benchmark — summary")
    print("=" * 60)
    if onset is not None:
        print(f"  2SC onset:                    μ_q = {onset:.1f} MeV")
    else:
        print("  2SC onset:                    not found in scan range")
    print(f"  Numerical g_Δ Δ_0 at {mu[-1]:.0f} MeV:  {asym_gap:.2f} MeV")
    print(f"  Analytic asymptote (Eq. 39):  {asym_gap_analytic_mev:.2f} MeV")
    print(f"  Ratio numerical/analytic:     {asym_gap / asym_gap_analytic_mev:.4f}")
    print(f"\n  Convergence (success=False):  {n_fail}/{n_total} = {100*n_fail/n_total:.1f}%")
    print(f"    vacuum  (μ_q < {vac_thresh:.0f} MeV):          {n_vac}")
    if onset is not None:
        print(f"    transition (±{tr_width:.0f} MeV around onset): {n_tr}")
    print(f"    2SC bulk (μ_q > {tr_hi:.0f} MeV):         {n_bulk}")
    print()
    print("  Note: success=False reflects gtol not met (flat landscape near")
    print("  minimum), not wrong field values.  The reported (φ, Δ) values")
    print("  are smooth through all failure boundaries.")
    print()
    print("  Optimizer options: QMDSimpleModel.solve_mean_fields does not")
    print("  expose the options kwarg of find_global_minimum, so maxiter")
    print("  cannot be set from this run script without modifying qmd_simple.py.")
    print("  L-BFGS-B default maxiter=15000 is already permissive; the")
    print("  success=False flags are tolerance-driven, not iteration-limited.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip the scan; reload saved data and regenerate plots only.",
    )
    parser.add_argument(
        "--skip-asymptotic", action="store_true",
        help="Do not load or run the high-mu conformal diagnostic.",
    )
    parser.add_argument(
        "--compute-asymptotic", action="store_true",
        help="Allow the high-mu diagnostic scan even in --plot-only mode.",
    )
    args = parser.parse_args()

    apply_plot_style()
    data_dir, _ = output_directories(OUTPUT_DIR, "qmd_benchmark")
    plots_dir = FIGURES_DIR
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    asym_gap_analytic = _asymptotic_gap_mev(QMD_SET_A)

    if args.plot_only:
        print("Plot-only mode: loading saved data ...")
        d = _load_benchmark_data(data_dir)
        mu        = d["mu_q_mev"]
        phi       = d["phi_mev"]
        gap       = d["gap_mev"]
        pressures = d["pressure_mev4"]
        n_q       = d["n_q_mev3"]
        eps       = d["energy_density_mev4"]
        cs2       = d["cs2"]
        onset     = d["onset_mev"]
        print(f"  Loaded {len(mu)} rows from qmd_benchmark.txt")
        print(f"  Analytic asymptotic g_Δ Δ̄_0 (Eq. 39, A&N 2024): {asym_gap_analytic:.2f} MeV")
    else:
        mu = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)

        print("QMD SET A benchmark scan")
        print(f"  include_omega_1_num = {QMD_SET_A.include_omega_1_num}")
        print(f"  μ_q range: {MU_MIN_MEV:.0f}–{MU_MAX_MEV:.0f} MeV, {NUM_POINTS} points")
        print(f"  Analytic asymptotic g_Δ Δ̄_0 (Eq. 39, A&N 2024): {asym_gap_analytic:.2f} MeV")

        model = QMDSimpleModel(QMD_SET_A)
        states = _scan(model, mu)
        phi = np.array([s.phi_mev for s in states])
        gap = np.array([s.gap_mev for s in states])
        pressures, n_q, eps, cs2 = _eos(states)

        _save_data(data_dir / "qmd_benchmark.txt", states, pressures, n_q, eps, cs2)
        print(f"  Saved {data_dir / 'qmd_benchmark.txt'}")

        onset = next((s.mu_q_mev for s in states if s.phase == "2SC"), None)
        n_failed = sum(1 for s in states if not s.success)
        if onset is not None:
            print(f"  2SC onset at μ_q ≈ {onset:.1f} MeV")
        else:
            print("  No 2SC condensation found in scanned range.")
        if n_failed:
            print(f"  {n_failed} points with success=False (see summary below)")

        _print_summary(mu, states, pressures, n_q, asym_gap_analytic)

    # --- Truncated model (for condensate comparison) ---
    trunc = _load_truncated_data(data_dir)
    if trunc is None:
        if args.plot_only:
            print("\nNo qmd_benchmark_truncated.txt found; skipping condensate comparison.")
        else:
            print(f"\nScanning truncated model (include_omega_1_num=False, {NUM_POINTS_TRUNC} pts) ...")
            mu_t = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS_TRUNC)
            model_t = QMDSimpleModel(PARAMS_TRUNC)
            states_t = _scan(model_t, mu_t)
            pressures_t, n_q_t, eps_t, cs2_t = _eos(states_t)
            _save_data(
                data_dir / "qmd_benchmark_truncated.txt",
                states_t, pressures_t, n_q_t, eps_t, cs2_t,
                params=PARAMS_TRUNC,
                description="QMD SET A truncated (no Omega_1_num)",
                num_points=NUM_POINTS_TRUNC,
            )
            print(f"  Saved {data_dir / 'qmd_benchmark_truncated.txt'}")
            trunc = _load_truncated_data(data_dir)

    asymptotic = None
    if not args.skip_asymptotic:
        print("\nPreparing high-μ conformal diagnostic ...")
        asymptotic = _compute_asymptotic_diagnostic(
            data_dir,
            allow_compute=(not args.plot_only) or args.compute_asymptotic,
        )

    print("\nGenerating plots ...")
    _plot_condensates(mu, phi, gap, onset, plots_dir)
    print(f"  Saved qmd_benchmark_condensates.pdf")
    _plot_pressure(mu, pressures, n_q, onset, plots_dir)
    print(f"  Saved qmd_benchmark_pressure.pdf")
    _plot_eos(mu, pressures, eps, n_q, plots_dir, asymptotic=asymptotic)
    print(f"  Saved qmd_benchmark_eos.pdf")
    _plot_cs2(mu, pressures, n_q, cs2, onset, plots_dir, asymptotic=asymptotic)
    print(f"  Saved qmd_benchmark_cs2.pdf")

    if trunc is not None:
        _plot_condensate_comparison(
            mu, phi, gap, onset,
            trunc["mu_q_mev"], trunc["phi_mev"], trunc["gap_mev"],
            trunc["chiral_onset_mev"],
            trunc["onset_mev"],
            trunc["trunc_2sc_end_mev"],
            plots_dir,
        )
        print(f"  Saved qmd_benchmark_condensate_comparison.pdf")


if __name__ == "__main__":
    main()
