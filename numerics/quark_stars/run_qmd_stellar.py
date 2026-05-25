"""Section 1 neutral stellar pipeline — QMD SET A baseline (full one-loop potential).

Computes the neutral stellar EoS and TOV mass-radius sequence for QMD_SET_A
with include_omega_1_num=True.  The existing cached equilibrium states in
output/qmd_stellar_equilibrium/ were generated with the truncated potential
and are intentionally NOT reused here.

Optimizer note
--------------
``QMDStellarModel.solve_equilibrium`` exposes ``minimizer_options`` which is
passed directly to ``find_global_minimum``.  We use
``{"maxiter": 80, "ftol": 1e-8}`` for the outer (phi, Delta) minimization,
matching run_qmd_stellar_eos.py.  Each function evaluation involves a nested
neutrality solve, so keeping maxiter moderate is important for runtime.

Produces
--------
  output/qmd_stellar/data/qmd_stellar_eos_baseline_raw.txt
  output/qmd_stellar/data/qmd_stellar_eos_baseline_stable.txt
  output/qmd_stellar/data/qmd_stars_baseline.txt
  output/qmd_stellar/plots/qmd_stellar_condensates.pdf
  output/qmd_stellar/plots/qmd_stellar_neutrality.pdf
  output/qmd_stellar/plots/qmd_stellar_eos.pdf
  output/qmd_stellar/plots/qmd_stellar_cs2.pdf
  output/qmd_stellar/plots/qmd_stellar_mass_radius.pdf
  output/qmd_stellar/plots/qmd_vs_qm_mass_radius.pdf
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from .constants import MEV4_TO_GEV_FM3
from .io import ensure_directory, output_directories, save_table
from .plotting import CS2_MU_MIN, CS2_XLIM, CS2_YLIM, apply_plot_style, save_figure
from .qmd_parameters import QMD_SET_A
from .qmd_simple import QMDSimpleModel
from .qmd_stellar import (
    QMDStellarEoSPoint,
    QMDStellarModel,
    QMDStellarState,
    build_qmd_stellar_eos_from_states,
)
from .solvers.tov import run_tov_sequence
from .thermodynamics.maxwell import maxwell_construct


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU_MIN_MEV = 250.0
MU_MAX_MEV = 900.0
NUM_POINTS = 350

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
QM_STELLAR_DIR = OUTPUT_DIR / "stellar"
BENCHMARK_DATA_DIR = OUTPUT_DIR / "qmd_benchmark" / "data"
QM_SIGMA_MEV = 600

# House style: viridis matching existing SET A plots
COLOR_QMD = plt.cm.viridis(0.15)   # neutral QMD (dark purple)
COLOR_BM  = plt.cm.viridis(0.6)    # free QMD benchmark overlay
COLOR_QM  = plt.cm.viridis(np.linspace(0.15, 0.6, 3))[2]
LW = 2.2

# EoS zoom window (two-panel plot, left panel)
_EOS_ZOOM_P_MAX_GEV_FM3  = 0.05
_EOS_ZOOM_EPS_MAX_GEV_FM3 = 0.45

# Smoothing windows — EoS smoothing uses a smaller window than the benchmark
# because the stellar scan has only 350 points (vs 5000 for the benchmark),
# so the benchmark window of 81 would span ~150 MeV and over-smooth the onset.
_EOS_SMOOTH_WINDOW   = 31   # for smoothing P(μ) and ε(μ) before cs² gradient
_CS2_BENCHMARK_SMOOTH_WINDOW = 51   # matches Section 2 benchmark curves
_CS2_NEUTRAL_SMOOTH_WINDOW   = 101  # neutral finite-difference curve is noisier
_RATIO_SMOOTH_WINDOW = 9
_SMOOTH_POLY = 3
_CS2_ONSET_THRESH = 295.0   # MeV — apply cs² smoothing only above this

# Extended free-QMD asymptotic scan (cached; replaces benchmark data for ratio panel)
_EXTENDED_ASYM_FILE       = "qmd_stellar_extended_asymptotic.txt"
_EXTENDED_ASYM_MU_MAX_MEV = 20000.0
_EXTENDED_ASYM_NUM_POINTS = 60
_EXTENDED_RESIDUAL_CUTOFF = 20000.0
_RATIO_TAPER_WIDTH_MEV    = 600.0   # neutral→free shift tapers to zero over this range

EOS_COLUMNS = [
    "mu_q_mev", "mu_B_mev", "phi_mev", "delta_mev", "gap_mev",
    "mu_e_mev", "mu_8_mev", "delta_mu_mev", "gap_minus_delta_mu_mev",
    "pressure_mev4", "quark_density_mev3", "baryon_density_mev3",
    "energy_density_mev4", "cs2", "omega_min_mev4", "phase",
    "success", "neutrality_residual_norm",
]

_MINIMIZER_OPTIONS = {"maxiter": 80, "ftol": 1.0e-8, "gtol": 1.0e-6}


# ---------------------------------------------------------------------------
# Smoothing helpers (mirror of run_qmd_benchmark.py)
# ---------------------------------------------------------------------------


def _odd_window(size: int, requested: int, polyorder: int = _SMOOTH_POLY) -> int:
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
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() != arr.size:
        return arr
    window = _odd_window(arr.size, requested_window, polyorder)
    if window == 0:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=polyorder)


def _prepare_cs2_curve(
    mu: np.ndarray,
    cs2: np.ndarray,
    *,
    smooth_from_mev: float,
    min_mu_mev: float | None = None,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter, trim endpoint artifacts, and smooth cs² like Section 2."""
    valid = np.isfinite(mu) & np.isfinite(cs2) & (cs2 >= 0.0) & (cs2 <= 1.0)
    if min_mu_mev is not None:
        valid &= mu >= min_mu_mev
    mu_plot = np.asarray(mu, dtype=float)[valid]
    cs2_plot = np.asarray(cs2, dtype=float)[valid].copy()

    # Match the Section 2 g_delta comparison: drop endpoint-gradient artifacts
    # before applying the plot-only Savitzky-Golay smoothing pass.
    if mu_plot.size > 2:
        mu_plot = mu_plot[:-2]
        cs2_plot = cs2_plot[:-2]

    post = mu_plot >= smooth_from_mev
    if post.sum() >= window:
        cs2_plot[post] = _smooth_for_plot(cs2_plot[post], window)

    return mu_plot, cs2_plot


def _set_log_mu_axis(ax, lower: float, upper: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    tick_candidates = np.array([500.0, 1000.0, 2000.0, 4000.0, 6000.0])
    ticks = tick_candidates[(tick_candidates >= lower) & (tick_candidates <= upper)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t)}" for t in ticks])


# ---------------------------------------------------------------------------
# Benchmark data loader (for overlay in condensate and cs² plots)
# ---------------------------------------------------------------------------


def _load_bm_data() -> dict[str, np.ndarray] | None:
    """Load the free-QMD benchmark scan for overlay purposes."""
    path = BENCHMARK_DATA_DIR / "qmd_benchmark.txt"
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    # columns: mu_q phi delta gap phase_2sc omega pressure n_q eps cs2 success
    mu_arr = data[:, 0]
    phase_2sc = data[:, 4]
    onset_candidates = mu_arr[phase_2sc > 0.5]
    onset = float(onset_candidates[0]) if onset_candidates.size > 0 else None
    return {
        "mu_q_mev": mu_arr,
        "phi_mev":  data[:, 1],
        "gap_mev":  data[:, 3],
        "cs2":      data[:, 9],
        "onset_mev": onset,
    }


def _load_asymptotic_data() -> dict[str, np.ndarray] | None:
    """Load high-μ log-spaced diagnostic for conformal convergence extension."""
    path = BENCHMARK_DATA_DIR / "qmd_benchmark_asymptotic_log.txt"
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    # columns: mu_q phi delta gap phase_2sc omega pressure n_q eps cs2 success
    return {
        "mu_q_mev":            data[:, 0],
        "pressure_mev4":       data[:, 6],
        "energy_density_mev4": data[:, 8],
    }


def _compute_or_load_extended_asym(data_dir: Path) -> dict | None:
    """Extended free-QMD scan from vacuum to 20 GeV for conformal convergence.

    On first call the scan is computed and cached; subsequent calls load the file.
    Uses the same free-QMD (common-μ) model as the benchmark asymptotic scan but
    extends to _EXTENDED_ASYM_MU_MAX_MEV so 6000 MeV becomes an interior point.
    """
    fpath = data_dir / _EXTENDED_ASYM_FILE
    if fpath.exists():
        try:
            data = np.loadtxt(fpath, comments="#")
            if data.ndim == 2 and data.shape[1] >= 3:
                print(f"  Loaded extended asymptotic data ({data.shape[0]} pts, "
                      f"μ_q up to {data[-1, 0]:.0f} MeV).")
                return {
                    "mu_q_mev":            data[:, 0],
                    "pressure_mev4":       data[:, 1],
                    "energy_density_mev4": data[:, 2],
                }
        except Exception:
            pass

    print(f"  Computing extended free-QMD scan "
          f"(vacuum → {_EXTENDED_ASYM_MU_MAX_MEV:.0f} MeV, "
          f"{_EXTENDED_ASYM_NUM_POINTS} log pts) ...")
    params = replace(QMD_SET_A, residual_cutoff_mev=_EXTENDED_RESIDUAL_CUTOFF)
    model  = QMDSimpleModel(params)
    mu_values = np.concatenate(
        [[0.0], np.geomspace(300.0, _EXTENDED_ASYM_MU_MAX_MEV, _EXTENDED_ASYM_NUM_POINTS)]
    )
    states: list = []
    prev = None
    for mu_val in mu_values:
        s = model.solve_mean_fields(float(mu_val), initial_guess=prev)
        states.append(s)
        prev = (s.phi_mev, s.delta_mev)

    omegas    = np.array([s.omega_min_mev4 for s in states])
    mus       = np.array([s.mu_q_mev       for s in states])
    omega_ref = omegas[0]
    pressures = -(omegas - omega_ref)
    n_q       = np.gradient(pressures, mus, edge_order=2)
    eps       = -pressures + mus * n_q

    ensure_directory(data_dir)
    np.savetxt(
        fpath,
        np.column_stack([mus, pressures, eps]),
        header="mu_q_mev  pressure_mev4  energy_density_mev4",
        fmt="%.10e",
    )
    print(f"  Saved {fpath.name} ({len(states)} pts)")
    return {"mu_q_mev": mus, "pressure_mev4": pressures, "energy_density_mev4": eps}


# ---------------------------------------------------------------------------
# Plot-only loaders
# ---------------------------------------------------------------------------


def _load_eos_points(path: Path) -> list:
    """Reconstruct plot-compatible EoS point objects from saved table."""
    points = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            points.append(SimpleNamespace(
                mu_q_mev=float(parts[0]),
                mu_B_mev=float(parts[1]),
                phi_mev=float(parts[2]),
                delta_mev=float(parts[3]),
                gap_mev=float(parts[4]),
                mu_e_mev=float(parts[5]),
                mu_8_mev=float(parts[6]),
                delta_mu_mev=float(parts[7]),
                gap_minus_delta_mu_mev=float(parts[8]),
                pressure_mev4=float(parts[9]),
                quark_density_mev3=float(parts[10]),
                baryon_density_mev3=float(parts[11]),
                energy_density_mev4=float(parts[12]),
                cs2=float(parts[13]),
                omega_min_mev4=float(parts[14]),
                phase=parts[15],
                success=bool(int(parts[16])),
                neutrality_residual_norm=float(parts[17]),
                neutrality_residual_e=0.0,
                neutrality_residual_8=0.0,
            ))
    return points


def _load_stars_sequence(path: Path):
    """Load saved M-R table into a minimal sequence namespace."""
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    return SimpleNamespace(
        central_pressure_dimless=data[:, 0],
        central_energy_density_mev4=data[:, 1],
        radius_km=data[:, 3],
        mass_msun=data[:, 4],
        stable_mask=data[:, 5].astype(bool),
    )


# ---------------------------------------------------------------------------
# Equilibrium scan
# ---------------------------------------------------------------------------


def _solve_vacuum(model: QMDStellarModel) -> QMDStellarState:
    """Solve or fallback to the canonical vacuum at mu_q = 0."""
    state = model.solve_equilibrium(
        0.0,
        initial_fields=(model.params.f_pi_mev, 0.0),
        initial_neutrality_guess=(0.0, 0.0),
        minimizer_options=_MINIMIZER_OPTIONS,
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
        message="Canonical vacuum fallback (solve_equilibrium at mu_q=0 did not converge).",
    )


def _scan_equilibrium(
    model: QMDStellarModel,
    mu_values: np.ndarray,
) -> list[QMDStellarState]:
    """Scan neutral equilibrium states with warm-start continuation."""
    states: list[QMDStellarState] = []
    prev: QMDStellarState | None = None
    n = len(mu_values)
    for i, mu in enumerate(mu_values, 1):
        state = model.solve_equilibrium(
            float(mu),
            previous_state=prev,
            initial_neutrality_guess=(0.0, 0.0),
            minimizer_options=_MINIMIZER_OPTIONS,
        )
        states.append(state)
        if state.success and np.isfinite(state.mu_e_mev) and np.isfinite(state.mu_8_mev):
            prev = state
        print(
            f"  {i:3d}/{n}  mu_q={state.mu_q_mev:7.2f}  "
            f"phi={state.phi_mev:8.3f}  gap={state.gap_mev:8.3f}  "
            f"res={state.neutrality_residual_norm:9.2e}  "
            f"{state.phase:>6}  {'ok' if state.success else 'FAIL'}",
            flush=True,
        )
    return states


# ---------------------------------------------------------------------------
# Stability construction
# ---------------------------------------------------------------------------


def _filter_points(points: list[QMDStellarEoSPoint]) -> list[QMDStellarEoSPoint]:
    return [
        p for p in points
        if p.success
        and p.pressure_mev4 > 0.0
        and p.quark_density_mev3 > 0.0
        and p.energy_density_mev4 > 0.0
        and np.isfinite(p.cs2)
    ]


def _recompute_cs2(points: list[QMDStellarEoSPoint]) -> list[QMDStellarEoSPoint]:
    if len(points) < 2:
        return [replace(p, cs2=float("nan")) for p in points]
    pressure = np.array([p.pressure_mev4 for p in points], dtype=float)
    energy = np.array([p.energy_density_mev4 for p in points], dtype=float)
    edge = 2 if len(points) >= 3 else 1
    dpdeps = np.gradient(pressure, energy, edge_order=edge)
    return [
        replace(p, cs2=float(dpdeps[i]) if np.isfinite(dpdeps[i]) else float("nan"))
        for i, p in enumerate(points)
    ]


def _strictly_increasing(
    points: list[QMDStellarEoSPoint],
) -> tuple[list[QMDStellarEoSPoint], int]:
    ordered = sorted(points, key=lambda p: (p.pressure_mev4, p.energy_density_mev4))
    kept: list[QMDStellarEoSPoint] = []
    removed = 0
    last_p = last_e = -float("inf")
    for p in ordered:
        if not kept:
            kept.append(p)
            last_p = p.pressure_mev4
            last_e = p.energy_density_mev4
            continue
        p_tol = 1.0e-12 * max(1.0, abs(p.pressure_mev4), abs(last_p))
        e_tol = 1.0e-12 * max(1.0, abs(p.energy_density_mev4), abs(last_e))
        if p.pressure_mev4 > last_p + p_tol and p.energy_density_mev4 > last_e + e_tol:
            kept.append(p)
            last_p = p.pressure_mev4
            last_e = p.energy_density_mev4
        else:
            removed += 1
    return kept, removed


def _nearest_point(
    points: list[QMDStellarEoSPoint],
    p_val: float,
    e_val: float,
) -> QMDStellarEoSPoint:
    pressure = np.array([p.pressure_mev4 for p in points])
    energy = np.array([p.energy_density_mev4 for p in points])
    p_scale = max(1.0, float(np.nanmax(np.abs(pressure))))
    e_scale = max(1.0, float(np.nanmax(np.abs(energy))))
    dist = ((pressure - p_val) / p_scale) ** 2 + ((energy - e_val) / e_scale) ** 2
    return points[int(np.nanargmin(dist))]


def _build_stable(
    raw_points: list[QMDStellarEoSPoint],
) -> tuple[list[QMDStellarEoSPoint], list[int], int, int]:
    if not raw_points:
        return [], [], 0, 0

    mu_q_arr = np.array([p.mu_q_mev for p in raw_points], dtype=float)
    pressure = np.array([p.pressure_mev4 for p in raw_points], dtype=float)
    energy = np.array([p.energy_density_mev4 for p in raw_points], dtype=float)
    stable_p, stable_e, maxwell_indices = maxwell_construct(mu_q_arr, pressure, energy)

    mapped: list[QMDStellarEoSPoint] = []
    for p_val, e_val in zip(stable_p, stable_e):
        if p_val <= 0.0 or e_val <= 0.0:
            continue
        near = _nearest_point(raw_points, float(p_val), float(e_val))
        mapped.append(replace(near, pressure_mev4=float(p_val), energy_density_mev4=float(e_val)))

    stable, mono_removed = _strictly_increasing(mapped)
    stable = _recompute_cs2(stable)

    cs2_removed = 0
    for _ in range(3):
        filtered = [p for p in stable if np.isfinite(p.cs2) and p.cs2 >= -1.0e-10]
        cs2_removed += len(stable) - len(filtered)
        if len(filtered) == len(stable):
            break
        stable, extra = _strictly_increasing(filtered)
        mono_removed += extra
        stable = _recompute_cs2(stable)

    return stable, maxwell_indices, mono_removed, cs2_removed


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------


def _write_eos_table(
    path: Path,
    points: list[QMDStellarEoSPoint],
    metadata: dict[str, object],
) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for k, v in metadata.items():
            f.write(f"# {k}={v}\n")
        f.write(f"# columns={' '.join(EOS_COLUMNS)}\n")
        for p in points:
            f.write(
                f"{p.mu_q_mev:.10e} {p.mu_B_mev:.10e} {p.phi_mev:.10e} "
                f"{p.delta_mev:.10e} {p.gap_mev:.10e} {p.mu_e_mev:.10e} "
                f"{p.mu_8_mev:.10e} {p.delta_mu_mev:.10e} "
                f"{p.gap_minus_delta_mu_mev:.10e} {p.pressure_mev4:.10e} "
                f"{p.quark_density_mev3:.10e} {p.baryon_density_mev3:.10e} "
                f"{p.energy_density_mev4:.10e} {p.cs2:.10e} "
                f"{p.omega_min_mev4:.10e} {p.phase} "
                f"{int(p.success)} {p.neutrality_residual_norm:.10e}\n"
            )


def _write_stars_table(path: Path, sequence) -> None:
    ensure_directory(path.parent)
    metadata = {
        "pipeline": "quark_stars",
        "product": "qmd_section1_mass_radius",
        "label": "QMD_SET_A_baseline_full_potential",
        "units": (
            "Pc_dimless in npemu units (P/e0); "
            "epsilon_c in MeV^4 and GeV fm^-3; radius in km; mass in Msun"
        ),
    }
    data = np.column_stack([
        sequence.central_pressure_dimless,
        sequence.central_energy_density_mev4,
        sequence.central_energy_density_mev4 * MEV4_TO_GEV_FM3,
        sequence.radius_km,
        sequence.mass_msun,
        sequence.stable_mask.astype(int),
    ])
    save_table(
        path,
        ["Pc_dimless", "epsilon_c_mev4", "epsilon_c_gev_fm3",
         "radius_km", "mass_msun", "stable_flag"],
        data,
        metadata,
    )


# ---------------------------------------------------------------------------
# TOV EoS wrapper
# ---------------------------------------------------------------------------


@dataclass
class _QMDEoS:
    """Duck-typed EoS wrapper satisfying the run_tov_sequence interface."""

    pressure_mev4: np.ndarray
    energy_density_mev4: np.ndarray
    m_sigma_mev: float = 0.0
    b0_mev4: float = 0.0
    b_mev4: float = 0.0
    b_min_mev4: float | None = None

    def tov_branch(self) -> tuple[np.ndarray, np.ndarray]:
        p, e = self.pressure_mev4, self.energy_density_mev4
        pos = p > 0.0
        p_pos, e_pos = p[pos], e[pos]
        if p_pos.size < 2:
            raise ValueError(
                "QMD stable EoS needs at least two positive-pressure points for TOV."
            )
        if p_pos.size >= 3:
            # Strip pre-transition points up to and including the first-order
            # 2SC onset.  The onset shows up as a large energy-density jump
            # (ε ratio ≫ 1) while rapid pressure growth within the 2SC phase
            # produces only moderate ε ratios (< 10).  Use the last ε jump
            # above 30× as the transition boundary.
            e_ratios = e_pos[1:] / np.maximum(e_pos[:-1], 1.0)
            trans = np.where(e_ratios > 30.0)[0]
            if trans.size:
                p_pos = p_pos[trans[-1] + 1:]
                e_pos = e_pos[trans[-1] + 1:]
            else:
                # Fallback: original large-pressure-jump heuristic
                jumps = np.where(p_pos[1:] > 1000.0 * p_pos[:-1])[0]
                if jumps.size:
                    p_pos = p_pos[jumps[0] + 1:]
                    e_pos = e_pos[jumps[0] + 1:]
        slope = (e_pos[1] - e_pos[0]) / (p_pos[1] - p_pos[0])
        e_surf = float(max(0.0, e_pos[0] - slope * p_pos[0]))
        pressure = np.concatenate(([0.0], p_pos))
        energy = np.concatenate(([e_surf], e_pos))
        order = np.argsort(pressure)
        pressure, energy = pressure[order], energy[order]
        unique = np.concatenate(([True], np.diff(pressure) > 0.0))
        return pressure[unique], energy[unique]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_condensates(
    all_points: list[QMDStellarEoSPoint],
    onset_mev: float | None,
    plots_dir: Path,
    bm_data: dict | None = None,
) -> None:
    """φ_0(μ_q) and g_Δ Δ_0(μ_q): neutral branch with optional benchmark overlay."""
    pts = [p for p in all_points if np.isfinite(p.phi_mev) and np.isfinite(p.gap_mev)]
    mu  = np.array([p.mu_q_mev for p in pts])
    phi = np.array([p.phi_mev  for p in pts])
    gap = np.array([p.gap_mev  for p in pts])

    # Smooth entire arrays (window=11, ≈21 MeV): avoids boundary artifact
    # from switching between raw and SG-smoothed data at the onset.
    phi_p = _smooth_for_plot(phi, 11)
    gap_p = _smooth_for_plot(gap, 11)

    # Extend neutral data back to xlim=200 MeV with the known pre-onset vacuum values
    # (φ = f_pi = 93 MeV, gap = 0).  The filtered EoS file starts at the first
    # positive-pressure point (~268 MeV); below that the condensate is flat.
    _XLIM_LEFT = 200.0
    if mu.size > 0 and mu[0] > _XLIM_LEFT:
        mu    = np.concatenate(([_XLIM_LEFT], mu))
        phi_p = np.concatenate(([93.0], phi_p))
        gap_p = np.concatenate(([0.0],  gap_p))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    if bm_data is not None:
        bm_mu  = bm_data["mu_q_mev"]
        bm_phi = bm_data["phi_mev"]
        bm_gap = bm_data["gap_mev"]
        bm_onset = bm_data["onset_mev"]
        bm_thresh = int(np.searchsorted(bm_mu, bm_onset)) if bm_onset is not None else len(bm_mu)
        bm_phi_p = bm_phi.copy()
        bm_gap_p = bm_gap.copy()
        if bm_thresh < len(bm_mu):
            bm_phi_p[bm_thresh:] = _smooth_for_plot(bm_phi[bm_thresh:], 51)
            bm_gap_p[bm_thresh:] = _smooth_for_plot(bm_gap[bm_thresh:], 51)
        ax1.plot(bm_mu, bm_phi_p, lw=LW, color=COLOR_BM,  label="Free QMD",    zorder=2)
        ax2.plot(bm_mu, bm_gap_p, lw=LW, color=COLOR_BM,  label="Free QMD",    zorder=2)

    ax1.plot(mu, phi_p, lw=LW, color=COLOR_QMD, label="Neutral QMD", zorder=3)
    ax2.plot(mu, gap_p, lw=LW, color=COLOR_QMD, label="Neutral QMD", zorder=3)

    if onset_mev is not None:
        ax1.axvline(onset_mev, color=COLOR_QMD, ls="--", lw=1.4)
        ax2.axvline(onset_mev, color=COLOR_QMD, ls="--", lw=1.4)
    if bm_data is not None and bm_data["onset_mev"] is not None:
        ax1.axvline(bm_data["onset_mev"], color=COLOR_BM, ls="--", lw=1.4)
        ax2.axvline(bm_data["onset_mev"], color=COLOR_BM, ls="--", lw=1.4)

    ax1.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax1.set_ylabel(r"$\phi_0\;(\mathrm{MeV})$")
    ax1.set_title("Chiral condensate")
    ax1.set_xlim(200.0, 700.0)
    ax1.legend()

    ax2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax2.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")
    ax2.set_title("Diquark gap")
    ax2.set_xlim(200.0, 700.0)
    ax2.legend()

    save_figure(plots_dir / "qmd_stellar_condensates.pdf")


def _plot_neutrality(
    all_points: list[QMDStellarEoSPoint],
    plots_dir: Path,
) -> None:
    """μ_e(μ_q) and μ_8(μ_q) on the neutral branch."""
    pts = [p for p in all_points if np.isfinite(p.mu_e_mev) and np.isfinite(p.mu_8_mev)]
    mu   = np.array([p.mu_q_mev for p in pts])
    mu_e = np.array([p.mu_e_mev for p in pts])
    mu_8 = np.array([p.mu_8_mev for p in pts])

    mu_e = _smooth_for_plot(mu_e, 21)
    mu_8 = _smooth_for_plot(mu_8, 21)

    fig, ax = plt.subplots()
    ax.plot(mu, mu_e, lw=LW, color=COLOR_QMD, label=r"$\mu_e$")
    ax.plot(mu, mu_8, lw=LW, color=COLOR_BM,  label=r"$\mu_8$")
    ax.axhline(0.0, color="gray", lw=1.0, ls=":")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"chemical potential $(\mathrm{MeV})$")
    ax.set_title("Neutrality chemical potentials")
    ax.set_xlim(250.0, 700.0)
    ax.legend()
    save_figure(plots_dir / "qmd_stellar_neutrality.pdf")


def _plot_eos(
    raw_points: list[QMDStellarEoSPoint],
    stable_points: list[QMDStellarEoSPoint],
    plots_dir: Path,
) -> None:
    """Single-panel low-pressure ε(P) zoom of the stable stellar EoS."""
    p_sta = np.array([p.pressure_mev4       for p in stable_points]) * MEV4_TO_GEV_FM3
    e_sta = np.array([p.energy_density_mev4 for p in stable_points]) * MEV4_TO_GEV_FM3

    fig, ax_eos = plt.subplots(figsize=(6.5, 4.8))

    sta_zoom = (
        np.isfinite(p_sta) & np.isfinite(e_sta)
        & (p_sta >= 0.0) & (p_sta <= _EOS_ZOOM_P_MAX_GEV_FM3)
    )
    ax_eos.plot(
        p_sta[sta_zoom],
        _smooth_for_plot(e_sta[sta_zoom], _EOS_SMOOTH_WINDOW),
        lw=LW, color=COLOR_QMD,
    )
    ax_eos.set_xlabel(r"$P\;(\mathrm{GeV\,fm}^{-3})$")
    ax_eos.set_ylabel(r"$\varepsilon\;(\mathrm{GeV\,fm}^{-3})$")
    ax_eos.set_title("Low-pressure EoS")
    ax_eos.set_xlim(0.0, _EOS_ZOOM_P_MAX_GEV_FM3)
    ax_eos.set_ylim(0.0, _EOS_ZOOM_EPS_MAX_GEV_FM3)

    save_figure(plots_dir / "qmd_stellar_eos.pdf")


def _plot_cs2(
    stable_points: list[QMDStellarEoSPoint],
    onset_mev: float | None,
    plots_dir: Path,
    bm_data: dict | None = None,
) -> None:
    """c_s²(μ_q): use pre-computed cs² from stable EoS table."""
    mu      = np.array([p.mu_q_mev for p in stable_points])
    cs2_raw = np.array([p.cs2      for p in stable_points])

    smooth_from = max(_CS2_ONSET_THRESH, onset_mev or _CS2_ONSET_THRESH)
    mu_plot, cs2_plot = _prepare_cs2_curve(
        mu,
        cs2_raw,
        smooth_from_mev=smooth_from,
        window=_CS2_NEUTRAL_SMOOTH_WINDOW,
    )

    # Diagnostics
    if cs2_plot.size:
        peak_idx = int(np.nanargmax(cs2_plot))
        peak_val = float(cs2_plot[peak_idx])
        peak_mu  = float(mu_plot[peak_idx])
        raw_mask = np.isfinite(cs2_raw) & (cs2_raw >= 0.0) & (cs2_raw <= 1.0)
        raw_peak = float(np.nanmax(cs2_raw[raw_mask]))
        print(f"  cs² raw peak: {raw_peak:.4f}  smoothed peak: {peak_val:.4f}  at μ_q={peak_mu:.1f} MeV")

    fig, ax = plt.subplots()

    if bm_data is not None:
        bm_mu  = bm_data["mu_q_mev"]
        bm_cs2 = bm_data["cs2"]
        bm_mu_p, bm_cs2_p = _prepare_cs2_curve(
            bm_mu,
            bm_cs2,
            smooth_from_mev=CS2_MU_MIN,
            min_mu_mev=CS2_MU_MIN,
            window=_CS2_BENCHMARK_SMOOTH_WINDOW,
        )
        ax.plot(bm_mu_p, bm_cs2_p, lw=LW, color=COLOR_BM,  label="Free QMD")

    ax.plot(mu_plot, cs2_plot, lw=2.6, color=COLOR_QMD, label="Neutral QMD")
    ax.axhline(1.0 / 3.0, color="gray", ls="--", lw=1.5,
               label=r"Conformal limit $c_s^2 = \frac{1}{3}$")

    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$c_s^2/c^2$")
    ax.set_title("Speed of sound squared")
    ax.set_xlim(*CS2_XLIM)
    ax.set_ylim(*CS2_YLIM)
    ax.legend()
    save_figure(plots_dir / "qmd_stellar_cs2.pdf")


def _plot_mass_radius(sequence, plots_dir: Path) -> None:
    """M(R): solid stable, dashed unstable, filled circle at M_max."""
    stable   = sequence.stable_mask.astype(bool)
    unstable = ~stable

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if stable.any():
        ax.plot(sequence.radius_km[stable], sequence.mass_msun[stable],
                color=COLOR_QMD, lw=LW, ls="-", label="QMD SET A")
    if unstable.any():
        ax.plot(sequence.radius_km[unstable], sequence.mass_msun[unstable],
                color=COLOR_QMD, lw=1.4, ls="--")
    if stable.any():
        m_s = sequence.mass_msun[stable]
        r_s = sequence.radius_km[stable]
        idx = int(np.argmax(m_s))
        ax.plot(r_s[idx], m_s[idx], "o", color=COLOR_QMD, ms=6,
                label=rf"$M_\mathrm{{max}} = {m_s[idx]:.2f}\,M_\odot$")

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_title("QMD stellar mass-radius (SET A baseline)")
    ax.set_xlim(8.0, 16.0)
    ax.set_ylim(0.5, 2.1)
    ax.legend()
    save_figure(plots_dir / "qmd_stellar_mass_radius.pdf")


def _plot_qmd_vs_qm(sequence, plots_dir: Path) -> None:
    """Overlay QMD SET A vs QM m_sigma=600 MeV M-R curves (three-curve comparison).

    Primary comparison: QMD vs gravitationally-bound QM (B_min=0) — matched construction.
    Secondary reference: self-bound QM (B_min=27.8 MeV) — shown for context.
    """
    COLOR_QM_GB  = plt.cm.viridis(0.85)   # gravitationally-bound QM (B=0)
    COLOR_QM_SB  = plt.cm.viridis(0.50)   # self-bound QM (B^{1/4}=27.8 MeV)

    stable   = sequence.stable_mask.astype(bool)
    unstable = ~stable

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # --- Gravitationally-bound QM (B^{1/4}=0) ---
    gb_file = QM_STELLAR_DIR / "qm_stars_sigma_600_grav_bound.txt"
    if gb_file.exists():
        gb_data = np.loadtxt(gb_file, comments="#")
        gb_r, gb_m, gb_stable_mask = gb_data[:, 3], gb_data[:, 4], gb_data[:, 5].astype(bool)
        gb_unstable_mask = ~gb_stable_mask
        # Physical filter: avoid unphysical low-mass artifacts (M < 0.3 Msun, R > 25 km)
        phys = (gb_m > 0.3) & (gb_r < 25.0)
        phys_s = phys & gb_stable_mask
        phys_u = phys & gb_unstable_mask
        if phys_s.any():
            ax.plot(gb_r[phys_s], gb_m[phys_s], color=COLOR_QM_GB, lw=LW, ls="-",
                    label=r"QM $B^{1/4}=0$")
            idx_gb = int(np.argmax(gb_m[phys_s]))
            ax.plot(gb_r[phys_s][idx_gb], gb_m[phys_s][idx_gb], "o", color=COLOR_QM_GB, ms=6)
        if phys_u.any():
            ax.plot(gb_r[phys_u], gb_m[phys_u], color=COLOR_QM_GB, lw=1.4, ls="--")
    else:
        print(f"  Gravitationally-bound QM file not found: {gb_file}")

    # --- Self-bound QM (B^{1/4}=27.8 MeV) ---
    qm_files = sorted(QM_STELLAR_DIR.glob(f"qm_stars_sigma_{QM_SIGMA_MEV}_Broot_*.txt"))
    qm_sb_file = next((f for f in qm_files if "Broot_28" in f.name), None)
    if qm_sb_file is not None:
        qm_data = np.loadtxt(qm_sb_file, comments="#")
        qm_r, qm_m, qm_stable_mask = qm_data[:, 3], qm_data[:, 4], qm_data[:, 5].astype(bool)
        qm_unstable_mask = ~qm_stable_mask
        if qm_stable_mask.any():
            ax.plot(qm_r[qm_stable_mask], qm_m[qm_stable_mask], color=COLOR_QM_SB, lw=LW, ls="-",
                    label=r"QM $B^{1/4}=27.8\,\mathrm{MeV}$")
            idx_qm = int(np.argmax(qm_m[qm_stable_mask]))
            ax.plot(qm_r[qm_stable_mask][idx_qm], qm_m[qm_stable_mask][idx_qm],
                    "o", color=COLOR_QM_SB, ms=6)
        if qm_unstable_mask.any():
            ax.plot(qm_r[qm_unstable_mask], qm_m[qm_unstable_mask],
                    color=COLOR_QM_SB, lw=1.4, ls="--")

    # --- QMD SET A ---
    if stable.any():
        ax.plot(sequence.radius_km[stable], sequence.mass_msun[stable],
                color=COLOR_QMD, lw=LW, ls="-", label="QMD SET A")
        m_s = sequence.mass_msun[stable]; r_s = sequence.radius_km[stable]
        idx = int(np.argmax(m_s))
        ax.plot(r_s[idx], m_s[idx], "o", color=COLOR_QMD, ms=6)
    if unstable.any():
        ax.plot(sequence.radius_km[unstable], sequence.mass_msun[unstable],
                color=COLOR_QMD, lw=1.4, ls="--")

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_title(r"Mass--radius comparison at $m_\sigma=600~\mathrm{MeV}$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlim(8.0, 18.0)
    ax.set_ylim(0.5, 2.1)
    save_figure(plots_dir / "qmd_vs_qm_mass_radius.pdf")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(
    raw_points: list[QMDStellarEoSPoint],
    stable_points: list[QMDStellarEoSPoint],
    maxwell_indices: list[int],
    sequence,
) -> None:
    print("\n" + "=" * 62)
    print("Section 1 stellar baseline — summary")
    print("=" * 62)

    onset = next((p.mu_q_mev for p in stable_points if p.phase == "2SC"), None)
    if onset is not None:
        print(f"  2SC onset in neutral matter:  μ_q = {onset:.1f} MeV")
    else:
        print("  2SC onset: not found on stable branch.")

    twosc = [p for p in stable_points if p.phase == "2SC"]
    if twosc:
        print(f"  Asymptotic g_Δ Δ_0 at μ_q = {twosc[-1].mu_q_mev:.0f} MeV:  "
              f"{twosc[-1].gap_mev:.2f} MeV")

    if not maxwell_indices:
        print("  Stability: no first-order transition removed (monotone branch).")
    else:
        raw_p = np.array([p.pressure_mev4 for p in raw_points])
        idx_info = []
        for idx in maxwell_indices:
            i = min(idx, len(raw_p) - 1)
            idx_info.append(f"idx={idx}, P={raw_p[i]:.3e} MeV⁴")
        print(f"  Stability: Maxwell construction applied at {'; '.join(idx_info)}.")
        print("  (First-order transition region replaced by equal-pressure bridge.)")

    stable_mask = sequence.stable_mask.astype(bool)
    if stable_mask.any():
        m_st = sequence.mass_msun[stable_mask]
        r_st = sequence.radius_km[stable_mask]
        idx = int(np.argmax(m_st))
        print(f"  M_max:                        {m_st[idx]:.4f} M☉")
        print(f"  R at M_max:                   {r_st[idx]:.4f} km")
    else:
        print("  No stable TOV configurations found.")

    if len(stable_points) >= 2:
        p0, p1 = stable_points[0].pressure_mev4, stable_points[1].pressure_mev4
        e0, e1 = stable_points[0].energy_density_mev4, stable_points[1].energy_density_mev4
        if abs(p1 - p0) > 0.0:
            slope  = (e1 - e0) / (p1 - p0)
            e_surf = max(0.0, e0 - slope * p0) * MEV4_TO_GEV_FM3
            print(f"  Surface ε at P = 0:           {e_surf:.4f} GeV/fm³")

    print(f"  Stable TOV configurations:    {stable_mask.sum()}")
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
    args = parser.parse_args()

    apply_plot_style()
    data_dir, plots_dir = output_directories(OUTPUT_DIR, "qmd_stellar")

    if args.plot_only:
        print("Plot-only mode: loading saved data ...")
        raw_path    = data_dir / "qmd_stellar_eos_baseline_raw.txt"
        stable_path = data_dir / "qmd_stellar_eos_baseline_stable.txt"
        stars_path  = data_dir / "qmd_stars_baseline.txt"
        if not raw_path.exists() or not stable_path.exists():
            raise FileNotFoundError(
                f"Saved EoS files not found in {data_dir}; run without --plot-only first."
            )
        raw_points    = _load_eos_points(raw_path)
        stable_points = _load_eos_points(stable_path)
        raw_all       = raw_points  # no neutrality potentials needed for condensate overlay
        sequence      = _load_stars_sequence(stars_path)
        maxwell_indices = []
        print(f"  Loaded {len(raw_points)} raw, {len(stable_points)} stable EoS points.")
        if sequence is not None:
            stable_tov = sequence.stable_mask.astype(bool)
            m_st = sequence.mass_msun[stable_tov]
            r_st = sequence.radius_km[stable_tov]
            if m_st.size:
                idx = int(np.argmax(m_st))
                print(f"  M_max = {m_st[idx]:.4f} M☉  at  R = {r_st[idx]:.4f} km")
    else:
        mu_values = np.linspace(MU_MIN_MEV, MU_MAX_MEV, NUM_POINTS)
        params = QMD_SET_A

        print("=" * 70)
        print("Section 1: neutral QMD stellar EoS — SET A baseline (full one-loop)")
        print(f"  include_omega_1_num = {params.include_omega_1_num}")
        print(f"  t_loop4_factor      = {params.t_loop4_factor}")
        print(f"  μ_q grid: {MU_MIN_MEV:.0f}–{MU_MAX_MEV:.0f} MeV, {NUM_POINTS} points")
        print(f"  Existing qmd_stellar_equilibrium cache: NOT reused (stale — truncated potential).")
        print("=" * 70)

        model = QMDStellarModel(params)

        vacuum = _solve_vacuum(model)
        omega_vac = vacuum.omega_min_mev4
        print(
            f"\nVacuum: Ω_vac = {omega_vac:.8e} MeV⁴  "
            f"(φ = {vacuum.phi_mev:.4g} MeV, "
            f"source = {'solve_equilibrium' if vacuum.success else 'canonical fallback'})"
        )

        print(f"\nScanning equilibrium branch ({NUM_POINTS} points) ...")
        states = _scan_equilibrium(model, mu_values)
        n_ok  = sum(1 for s in states if s.success)
        n_2sc = sum(1 for s in states if s.phase == "2SC")
        print(f"\nScan complete: {n_ok}/{NUM_POINTS} ok, {n_2sc} in 2SC phase.")

        raw_all    = build_qmd_stellar_eos_from_states(states, omega_vac)
        raw_points = _filter_points(raw_all)
        print(f"Raw branch: {len(raw_all)} total → {len(raw_points)} after filtering.")

        stable_points, maxwell_indices, mono_removed, cs2_removed = _build_stable(raw_points)
        print(
            f"Stable branch: {len(stable_points)} points  "
            f"(maxwell={maxwell_indices}, monotone_removed={mono_removed}, "
            f"cs2_removed={cs2_removed})"
        )

        base_meta: dict[str, object] = {
            "model": "QMDStellarModel",
            "label": "qmd_stellar_baseline",
            "include_omega_1_num": str(params.include_omega_1_num),
            "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
            "m_delta_mev": f"{params.m_delta_mev:.1f}",
            "g_delta_factor": f"{params.g_delta_factor:.4f}",
            "mu_min_mev": f"{MU_MIN_MEV:.1f}",
            "mu_max_mev": f"{MU_MAX_MEV:.1f}",
            "num_raw_points": str(NUM_POINTS),
            "omega_vac_mev4": f"{omega_vac:.10e}",
            "omega_vac_source": "neutral_equilibrium" if vacuum.success else "canonical_fallback",
            "equilibrium_cache_reused": "false",
        }

        _write_eos_table(
            data_dir / "qmd_stellar_eos_baseline_raw.txt",
            raw_points,
            {**base_meta, "branch": "raw"},
        )
        print(f"Saved raw EoS ({len(raw_points)} points) → {data_dir / 'qmd_stellar_eos_baseline_raw.txt'}")

        _write_eos_table(
            data_dir / "qmd_stellar_eos_baseline_stable.txt",
            stable_points,
            {
                **base_meta,
                "branch": "stable",
                "maxwell_indices": str(maxwell_indices),
                "monotone_removed": str(mono_removed),
                "cs2_removed": str(cs2_removed),
            },
        )
        print(f"Saved stable EoS ({len(stable_points)} points) → {data_dir / 'qmd_stellar_eos_baseline_stable.txt'}")

        print("\nRunning TOV integration ...")
        p_arr   = np.array([p.pressure_mev4         for p in stable_points])
        e_arr   = np.array([p.energy_density_mev4   for p in stable_points])
        eos_tov = _QMDEoS(pressure_mev4=p_arr, energy_density_mev4=e_arr)
        sequence = None
        try:
            sequence = run_tov_sequence(eos_tov, integrator="rk4")
            stable_tov = sequence.stable_mask.astype(bool)
            print(f"  {sequence.mass_msun.size} TOV configurations, {stable_tov.sum()} stable.")
            if stable_tov.any():
                m_st = sequence.mass_msun[stable_tov]
                r_st = sequence.radius_km[stable_tov]
                idx  = int(np.argmax(m_st))
                print(f"  M_max = {m_st[idx]:.4f} M☉  at  R = {r_st[idx]:.4f} km")
            _write_stars_table(data_dir / "qmd_stars_baseline.txt", sequence)
            print(f"Saved M-R table → {data_dir / 'qmd_stars_baseline.txt'}")
        except ValueError as exc:
            print(f"  TOV failed: {exc}")

    # Onset for onset markers in plots
    onset_mev = next((p.mu_q_mev for p in stable_points if p.phase == "2SC"), None)

    # Benchmark overlay data (free QMD, no neutrality)
    bm_data   = _load_bm_data()
    if bm_data is not None:
        print(f"\nLoaded benchmark data for overlay ({len(bm_data['mu_q_mev'])} pts).")
    else:
        print("\nNo qmd_benchmark data found; condensate/cs² plots will show neutral QMD only.")

    # Extended free-QMD scan for conformal convergence (vacuum → 20 GeV)
    print("Loading/computing extended asymptotic data ...")
    ext_asym_data = _compute_or_load_extended_asym(data_dir)

    print("\nGenerating plots ...")
    _plot_condensates(raw_all, onset_mev, plots_dir, bm_data)
    print("  Saved qmd_stellar_condensates.pdf")
    _plot_neutrality(raw_all, plots_dir)
    print("  Saved qmd_stellar_neutrality.pdf")
    _plot_eos(raw_points, stable_points, plots_dir)
    print("  Saved qmd_stellar_eos.pdf")
    _plot_cs2(stable_points, onset_mev, plots_dir, bm_data)
    print("  Saved qmd_stellar_cs2.pdf")
    if sequence is not None:
        _plot_qmd_vs_qm(sequence, plots_dir)
        print("  Saved qmd_vs_qm_mass_radius.pdf")
    else:
        print("  Skipping M-R plots (TOV not available).")

    if sequence is not None and not args.plot_only:
        _print_summary(raw_points, stable_points, maxwell_indices, sequence)


if __name__ == "__main__":
    main()
