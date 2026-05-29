"""Section 2 parameter sensitivity sweep around Set A.

Runs 8 single-parameter variations around the default parameter set:
  Sets B,C: g_Δ = 1.5g, 2.5g
  Sets D,E: m_Δ = 400, 700 MeV
  Sets F,G: λ_Δ = λ₀/8, λ₀/2
  Sets H,I: λ₃ = 0, 2λ₀

For each set:
  - 700-point neutral stellar pipeline (same solver settings as Section 1)
  - 5000-point common-μ_q benchmark (no neutrality)

Set A loaded from output/qmd_stellar/ (Section 1 results).

Produces
--------
  output/sec5_parameter_sensitivity/data/section2_stellar_{param}_{value_tag}.txt   (×8)
  output/sec5_parameter_sensitivity/data/section2_benchmark_{param}_{value_tag}.txt (×8)
  thesis/figures/quark_stars/parameter_sensitivity/section2_MR_gdelta.pdf
  thesis/figures/quark_stars/parameter_sensitivity/section2_MR_mdelta.pdf
  thesis/figures/quark_stars/parameter_sensitivity/section2_MR_lamdelta.pdf
  thesis/figures/quark_stars/parameter_sensitivity/section2_MR_lam3.pdf
  thesis/figures/quark_stars/parameter_sensitivity/section2_cs2_gdelta.pdf
  thesis/figures/quark_stars/parameter_sensitivity/section2_condensates_gdelta.pdf
  thesis/figures/quark_stars/parameter_sensitivity/section2_condensates_mdelta.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from dataclasses import dataclass, replace, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from ..constants import MEV4_TO_GEV_FM3
from ..io import ensure_directory
from ..plotting import (
    CS2_MU_MIN,
    CS2_XLIM,
    CS2_YLIM,
    SECTION2_MR_BASELINE_COLOR,
    SECTION2_MR_HIGH_VARIATION_COLOR,
    SECTION2_MR_LOW_VARIATION_COLOR,
    apply_plot_style,
    save_figure,
)
from ..qmd_parameters import QMD_SET_A, QMDParameters
from ..qmd_simple import QMDSimpleModel
from ..qmd_stellar import (
    QMDStellarEoSPoint,
    QMDStellarModel,
    QMDStellarState,
    build_qmd_stellar_eos_from_states,
)
from ..solvers.tov import run_tov_sequence
from ..thermodynamics.maxwell import maxwell_construct


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR   = Path(__file__).resolve().parent.parent / "output"
SECTION2_DIR = OUTPUT_DIR / "sec5_parameter_sensitivity"
DATA_DIR     = SECTION2_DIR / "data"
PLOTS_DIR    = Path(__file__).resolve().parents[3] / "thesis" / "figures" / "quark_stars" / "parameter_sensitivity"
ERRORS_LOG   = SECTION2_DIR / "errors.log"
SUMMARY_CSV  = SECTION2_DIR / "section2_summary.csv"

BASELINE_STELLAR_DIR = OUTPUT_DIR / "qmd_stellar" / "data"
BASELINE_BENCHMARK_DIR = OUTPUT_DIR / "qmd_benchmark" / "data"

# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------

NUM_STELLAR_POINTS = 700
MU_MIN_STELLAR = 250.0
MU_MAX_STELLAR = 900.0

NUM_BM_POINTS = 5000
MU_MIN_BM = 0.0
MU_MAX_BM = 900.0

_MINIMIZER_OPTIONS = {"maxiter": 80, "ftol": 1.0e-8, "gtol": 1.0e-6}

# Gap bound for run B special check (MeV)
KURKELA_BOUND_MEV = 268.0

LW_BASELINE = 2.6
LW_VARIATION = 2.0
_SMOOTH_POLY = 3
_CS2_GDELTA_SMOOTH_WINDOW = 51
_CS2_X_MAX_MEV = 800.0


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    run_id: str          # e.g. "B"
    param_name: str      # e.g. "gdelta"
    param_value: str     # e.g. "1.5g"
    value_tag: str       # e.g. "1p5g"  (used in filename)
    params: QMDParameters
    label: str           # human-readable for plots
    color: tuple         # matplotlib color


def _make_runs() -> list[RunConfig]:
    baseline = QMD_SET_A

    return [
        RunConfig("B", "gdelta", "1.5g", "1p5g",
                  replace(baseline, g_delta_factor=1.5),
                  "Set B", SECTION2_MR_LOW_VARIATION_COLOR),
        RunConfig("C", "gdelta", "2.5g", "2p5g",
                  replace(baseline, g_delta_factor=2.5),
                  "Set C", SECTION2_MR_HIGH_VARIATION_COLOR),
        RunConfig("D", "mdelta", "400",  "400",
                  replace(baseline, m_delta_mev=400.0),
                  "Set D", SECTION2_MR_LOW_VARIATION_COLOR),
        RunConfig("E", "mdelta", "700",  "700",
                  replace(baseline, m_delta_mev=700.0),
                  "Set E", SECTION2_MR_HIGH_VARIATION_COLOR),
        RunConfig("F", "lamdelta", "lam0div8",  "lam0div8",
                  replace(baseline, lambda_delta_factor=0.125),
                  "Set F", SECTION2_MR_LOW_VARIATION_COLOR),
        RunConfig("G", "lamdelta", "lam0div2",  "lam0div2",
                  replace(baseline, lambda_delta_factor=0.5),
                  "Set G", SECTION2_MR_HIGH_VARIATION_COLOR),
        RunConfig("H", "lam3", "0",     "0",
                  replace(baseline, lambda_3_factor=0.0),
                  "Set H", SECTION2_MR_LOW_VARIATION_COLOR),
        RunConfig("I", "lam3", "2lam0", "2lam0",
                  replace(baseline, lambda_3_factor=2.0),
                  "Set I", SECTION2_MR_HIGH_VARIATION_COLOR),
    ]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    param_name: str
    param_value: str
    # Stellar metrics
    M_max: Optional[float] = None
    R_at_Mmax: Optional[float] = None
    onset_muq: Optional[float] = None
    cs2_peak: Optional[float] = None
    cs2_peak_muq: Optional[float] = None
    asymptotic_gap: Optional[float] = None
    n_nonconverged_pct: Optional[float] = None
    notes: str = ""
    # Data arrays for plotting (kept in memory, not serialised to csv)
    radius_km: Optional[np.ndarray] = field(default=None, repr=False)
    mass_msun: Optional[np.ndarray] = field(default=None, repr=False)
    stable_mask: Optional[np.ndarray] = field(default=None, repr=False)
    bm_mu_q: Optional[np.ndarray] = field(default=None, repr=False)
    bm_phi: Optional[np.ndarray] = field(default=None, repr=False)
    bm_gap: Optional[np.ndarray] = field(default=None, repr=False)
    bm_cs2: Optional[np.ndarray] = field(default=None, repr=False)
    bm_onset: Optional[float] = None
    partial: bool = False
    failed: bool = False


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def _log_error(run_id: str, context: str, exc: Exception) -> None:
    ensure_directory(SECTION2_DIR)
    msg = f"[run {run_id}] {context}: {type(exc).__name__}: {exc}\n"
    msg += traceback.format_exc() + "\n"
    with ERRORS_LOG.open("a", encoding="utf-8") as f:
        f.write(msg)
    print(f"  ERROR logged to errors.log: {context}: {exc}")


# ---------------------------------------------------------------------------
# Stellar pipeline helpers (parameterised versions of run_qmd_stellar helpers)
# ---------------------------------------------------------------------------

def _solve_vacuum(model: QMDStellarModel) -> QMDStellarState:
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
        mu_q_mev=0.0, phi_mev=model.params.f_pi_mev, delta_mev=0.0,
        gap_mev=0.0, mu_e_mev=0.0, mu_8_mev=0.0, delta_mu_mev=0.0,
        gap_minus_delta_mu_mev=0.0, omega_min_mev4=omega_vac,
        neutrality_residual_e=0.0, neutrality_residual_8=0.0,
        neutrality_residual_norm=0.0, phase="normal", success=False,
        message="Canonical vacuum fallback.",
    )


def _scan_equilibrium(model: QMDStellarModel, mu_values: np.ndarray) -> list[QMDStellarState]:
    states: list[QMDStellarState] = []
    prev: Optional[QMDStellarState] = None
    n = len(mu_values)
    for i, mu in enumerate(mu_values, 1):
        state = model.solve_equilibrium(
            float(mu), previous_state=prev,
            initial_neutrality_guess=(0.0, 0.0),
            minimizer_options=_MINIMIZER_OPTIONS,
        )
        states.append(state)
        if state.success and np.isfinite(state.mu_e_mev) and np.isfinite(state.mu_8_mev):
            prev = state
        if i % 50 == 0 or i == n:
            print(
                f"    {i:3d}/{n}  μ_q={state.mu_q_mev:7.2f}  "
                f"φ={state.phi_mev:7.3f}  gap={state.gap_mev:7.3f}  "
                f"res={state.neutrality_residual_norm:8.2e}  "
                f"{state.phase:>6}  {'ok' if state.success else 'FAIL'}",
                flush=True,
            )
    return states


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
    energy   = np.array([p.energy_density_mev4 for p in points], dtype=float)
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
            kept.append(p); last_p = p.pressure_mev4; last_e = p.energy_density_mev4; continue
        p_tol = 1.0e-12 * max(1.0, abs(p.pressure_mev4), abs(last_p))
        e_tol = 1.0e-12 * max(1.0, abs(p.energy_density_mev4), abs(last_e))
        if p.pressure_mev4 > last_p + p_tol and p.energy_density_mev4 > last_e + e_tol:
            kept.append(p); last_p = p.pressure_mev4; last_e = p.energy_density_mev4
        else:
            removed += 1
    return kept, removed


def _nearest_point(points: list, p_val: float, e_val: float) -> QMDStellarEoSPoint:
    pressure = np.array([p.pressure_mev4       for p in points])
    energy   = np.array([p.energy_density_mev4 for p in points])
    p_scale  = max(1.0, float(np.nanmax(np.abs(pressure))))
    e_scale  = max(1.0, float(np.nanmax(np.abs(energy))))
    dist     = ((pressure - p_val) / p_scale) ** 2 + ((energy - e_val) / e_scale) ** 2
    return points[int(np.nanargmin(dist))]


def _build_stable(
    raw_points: list[QMDStellarEoSPoint],
) -> tuple[list[QMDStellarEoSPoint], list, int, int]:
    if not raw_points:
        return [], [], 0, 0
    mu_q_arr = np.array([p.mu_q_mev         for p in raw_points], dtype=float)
    pressure = np.array([p.pressure_mev4     for p in raw_points], dtype=float)
    energy   = np.array([p.energy_density_mev4 for p in raw_points], dtype=float)
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


@dataclass
class _QMDEoS:
    pressure_mev4: np.ndarray
    energy_density_mev4: np.ndarray
    m_sigma_mev: float = 0.0
    b0_mev4: float = 0.0
    b_mev4: float = 0.0
    b_min_mev4: Optional[float] = None

    def tov_branch(self) -> tuple[np.ndarray, np.ndarray]:
        p, e = self.pressure_mev4, self.energy_density_mev4
        pos = p > 0.0
        p_pos, e_pos = p[pos], e[pos]
        if p_pos.size < 2:
            raise ValueError("Stellar EoS needs at least two positive-pressure points.")
        if p_pos.size >= 3:
            e_ratios = e_pos[1:] / np.maximum(e_pos[:-1], 1.0)
            trans = np.where(e_ratios > 30.0)[0]
            if trans.size:
                p_pos = p_pos[trans[-1] + 1:]
                e_pos = e_pos[trans[-1] + 1:]
            else:
                jumps = np.where(p_pos[1:] > 1000.0 * p_pos[:-1])[0]
                if jumps.size:
                    p_pos = p_pos[jumps[0] + 1:]
                    e_pos = e_pos[jumps[0] + 1:]
        slope  = (e_pos[1] - e_pos[0]) / (p_pos[1] - p_pos[0])
        e_surf = float(max(0.0, e_pos[0] - slope * p_pos[0]))
        pressure = np.concatenate(([0.0], p_pos))
        energy   = np.concatenate(([e_surf], e_pos))
        order    = np.argsort(pressure)
        pressure, energy = pressure[order], energy[order]
        unique = np.concatenate(([True], np.diff(pressure) > 0.0))
        return pressure[unique], energy[unique]


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

_EOS_COLUMNS = [
    "mu_q_mev", "mu_B_mev", "phi_mev", "delta_mev", "gap_mev",
    "mu_e_mev", "mu_8_mev", "delta_mu_mev", "gap_minus_delta_mu_mev",
    "pressure_mev4", "quark_density_mev3", "baryon_density_mev3",
    "energy_density_mev4", "cs2", "omega_min_mev4", "phase",
    "success", "neutrality_residual_norm",
]


def _write_stellar_eos(
    path: Path,
    points: list[QMDStellarEoSPoint],
    metadata: dict,
) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for k, v in metadata.items():
            f.write(f"# {k}={v}\n")
        f.write(f"# columns={' '.join(_EOS_COLUMNS)}\n")
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


def _write_stellar_mr(path: Path, sequence, metadata: dict) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for k, v in metadata.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns=Pc_dimless epsilon_c_mev4 epsilon_c_gev_fm3 radius_km mass_msun stable_flag\n")
        for i in range(len(sequence.mass_msun)):
            f.write(
                f"{sequence.central_pressure_dimless[i]:.6e} "
                f"{sequence.central_energy_density_mev4[i]:.6e} "
                f"{sequence.central_energy_density_mev4[i]*MEV4_TO_GEV_FM3:.6e} "
                f"{sequence.radius_km[i]:.6e} "
                f"{sequence.mass_msun[i]:.6e} "
                f"{int(sequence.stable_mask[i])}\n"
            )


def _write_benchmark_data(
    path: Path,
    states,
    pressures: np.ndarray,
    n_q: np.ndarray,
    eps: np.ndarray,
    cs2: np.ndarray,
    params: QMDParameters,
    description: str = "",
) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# description={description}\n")
        f.write(f"# include_omega_1_num={params.include_omega_1_num}\n")
        f.write(f"# m_delta_mev={params.m_delta_mev:.1f}\n")
        f.write(f"# g_delta_factor={params.g_delta_factor:.4f}\n")
        f.write(f"# lambda_3_factor={params.lambda_3_factor:.4f}\n")
        f.write(f"# lambda_delta_factor={params.lambda_delta_factor:.4f}\n")
        f.write(f"# t_loop4_factor={params.t_loop4_factor:.1f}\n")
        f.write(f"# residual_cutoff_mev={params.residual_cutoff_mev:.1f}\n")
        f.write(f"# mu_min_mev={MU_MIN_BM:.1f}\n")
        f.write(f"# mu_max_mev={MU_MAX_BM:.1f}\n")
        f.write(f"# num_points={NUM_BM_POINTS}\n")
        f.write("# columns=mu_q_mev phi_mev delta_mev gap_mev phase_2sc omega_min_mev4 "
                "pressure_mev4 n_q_mev3 energy_density_mev4 cs2 success\n")
        for i, s in enumerate(states):
            f.write(
                f"{s.mu_q_mev:.10e} {s.phi_mev:.10e} {s.delta_mev:.10e} "
                f"{s.gap_mev:.10e} {float(s.phase == '2SC'):.1f} "
                f"{s.omega_min_mev4:.10e} {pressures[i]:.10e} {n_q[i]:.10e} "
                f"{eps[i]:.10e} {cs2[i]:.10e} {float(s.success):.1f}\n"
            )


# ---------------------------------------------------------------------------
# Stellar pipeline
# ---------------------------------------------------------------------------

def run_stellar_pipeline(cfg: RunConfig) -> tuple[Optional[SimpleNamespace], Optional[RunResult]]:
    """Run 700-point neutral stellar pipeline for the given parameter set.

    Returns (sequence_namespace, partial_result).  On total failure returns (None, None).
    """
    params = cfg.params
    tag    = f"{cfg.param_name}_{cfg.value_tag}"
    print(f"\n  [Stellar {cfg.run_id}] {tag}  "
          f"(g_Δ={params.g_delta_factor:.2f}g, m_Δ={params.m_delta_mev:.0f} MeV, "
          f"λ_Δ={params.lambda_delta_factor:.3f}λ₀, λ₃={params.lambda_3_factor:.3f}λ₀)")

    model = QMDStellarModel(params)

    # Vacuum
    try:
        vacuum = _solve_vacuum(model)
        omega_vac = vacuum.omega_min_mev4
    except Exception as exc:
        _log_error(cfg.run_id, "vacuum solve", exc)
        return None, None

    # Equilibrium scan
    mu_values = np.linspace(MU_MIN_STELLAR, MU_MAX_STELLAR, NUM_STELLAR_POINTS)
    try:
        states = _scan_equilibrium(model, mu_values)
    except Exception as exc:
        _log_error(cfg.run_id, "equilibrium scan", exc)
        return None, None

    n_ok       = sum(1 for s in states if s.success)
    n_2sc      = sum(1 for s in states if s.phase == "2SC")
    fail_pct   = 100.0 * (len(states) - n_ok) / max(1, len(states))
    partial    = fail_pct > 30.0
    notes_list = []
    if partial:
        notes_list.append(f"PARTIAL ({fail_pct:.1f}% non-converged)")
        print(f"    WARNING: {fail_pct:.1f}% non-convergence — marking PARTIAL")

    print(f"    Scan: {n_ok}/{NUM_STELLAR_POINTS} ok, {n_2sc} 2SC points, "
          f"{fail_pct:.1f}% failed")

    # EoS construction
    try:
        raw_all    = build_qmd_stellar_eos_from_states(states, omega_vac)
        raw_points = _filter_points(raw_all)
        stable_points, maxwell_indices, mono_removed, cs2_removed = _build_stable(raw_points)
    except Exception as exc:
        _log_error(cfg.run_id, "EoS construction", exc)
        return None, None

    if len(maxwell_indices) > 0:
        n_maxwell_pts = sum(1 for idx in maxwell_indices if idx < len(raw_points))
        if n_maxwell_pts > 0.5 * len(raw_points):
            notes_list.append(f"Maxwell removed >50% of points (removed at {maxwell_indices})")
            print(f"    FLAG: Maxwell construction removed many points: {maxwell_indices}")

    onset_muq = next((p.mu_q_mev for p in stable_points if p.phase == "2SC"), None)
    if n_2sc == 0:
        onset_muq = None
        notes_list.append("No 2SC phase in stellar density range")
        print("    NOTE: No 2SC phase found within μ_q ≤ 900 MeV (stellar range)")

    # Save EoS tables
    base_meta = {
        "section": "2",
        "run_id": cfg.run_id,
        "param_name": cfg.param_name,
        "param_value": cfg.param_value,
        "m_delta_mev": f"{params.m_delta_mev:.1f}",
        "g_delta_factor": f"{params.g_delta_factor:.4f}",
        "lambda_3_factor": f"{params.lambda_3_factor:.4f}",
        "lambda_delta_factor": f"{params.lambda_delta_factor:.4f}",
        "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
        "include_omega_1_num": str(params.include_omega_1_num),
        "mu_min_mev": f"{MU_MIN_STELLAR:.1f}",
        "mu_max_mev": f"{MU_MAX_STELLAR:.1f}",
        "num_raw_points": str(NUM_STELLAR_POINTS),
        "omega_vac_mev4": f"{omega_vac:.10e}",
    }
    eos_path = DATA_DIR / f"section2_stellar_{tag}_eos.txt"
    _write_stellar_eos(eos_path, stable_points, {**base_meta, "branch": "stable"})
    print(f"    Saved EoS → {eos_path.name}")

    # TOV
    sequence = None
    M_max = R_at_Mmax = None
    try:
        p_arr    = np.array([p.pressure_mev4       for p in stable_points])
        e_arr    = np.array([p.energy_density_mev4 for p in stable_points])
        eos_tov  = _QMDEoS(pressure_mev4=p_arr, energy_density_mev4=e_arr)
        sequence = run_tov_sequence(eos_tov, integrator="rk4")
        stable_tov = sequence.stable_mask.astype(bool)
        if stable_tov.any():
            m_st = sequence.mass_msun[stable_tov]
            r_st = sequence.radius_km[stable_tov]
            idx  = int(np.argmax(m_st))
            M_max       = float(m_st[idx])
            R_at_Mmax   = float(r_st[idx])
            print(f"    TOV: M_max={M_max:.4f} M☉  R={R_at_Mmax:.4f} km")

        # Save M-R table
        stellar_path = DATA_DIR / f"section2_stellar_{tag}.txt"
        _write_stellar_mr(stellar_path, sequence, base_meta)
        print(f"    Saved M-R → {stellar_path.name}")

    except Exception as exc:
        _log_error(cfg.run_id, "TOV integration", exc)
        notes_list.append(f"TOV failed: {exc}")

    # cs2 from stable EoS
    cs2_peak = cs2_peak_muq = None
    if stable_points:
        mu_arr  = np.array([p.mu_q_mev for p in stable_points])
        cs2_arr = np.array([p.cs2      for p in stable_points])
        fin = np.isfinite(cs2_arr) & (cs2_arr >= 0.0) & (cs2_arr <= 1.0)
        if fin.any():
            idx_pk   = int(np.nanargmax(cs2_arr[fin]))
            cs2_peak = float(cs2_arr[fin][idx_pk])
            cs2_peak_muq = float(mu_arr[fin][idx_pk])

    result = RunResult(
        run_id=cfg.run_id,
        param_name=cfg.param_name,
        param_value=cfg.param_value,
        M_max=M_max,
        R_at_Mmax=R_at_Mmax,
        onset_muq=onset_muq,
        cs2_peak=cs2_peak,
        cs2_peak_muq=cs2_peak_muq,
        n_nonconverged_pct=fail_pct,
        notes="; ".join(notes_list) if notes_list else "",
        partial=partial,
    )
    if sequence is not None:
        result.radius_km   = sequence.radius_km
        result.mass_msun   = sequence.mass_msun
        result.stable_mask = sequence.stable_mask

    return sequence, result


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------

def _bm_eos(states) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omegas = np.array([s.omega_min_mev4 for s in states])
    mus    = np.array([s.mu_q_mev       for s in states])
    omega_ref = omegas[0]
    pressures = -(omegas - omega_ref)
    edge = 2 if len(states) >= 3 else 1
    n_q  = np.gradient(pressures, mus, edge_order=edge)
    eps  = -pressures + mus * n_q
    dp_dmu   = np.gradient(pressures, mus, edge_order=edge)
    deps_dmu = np.gradient(eps, mus, edge_order=edge)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs2 = np.where(deps_dmu > 0.0, dp_dmu / deps_dmu, np.nan)
    return pressures, n_q, eps, cs2


def run_benchmark_pipeline(cfg: RunConfig) -> Optional[dict]:
    """Run 5000-point common-μ_q benchmark (no neutrality)."""
    params = cfg.params
    tag    = f"{cfg.param_name}_{cfg.value_tag}"
    print(f"\n  [Benchmark {cfg.run_id}] {tag}", flush=True)

    mu_values = np.linspace(MU_MIN_BM, MU_MAX_BM, NUM_BM_POINTS)
    model     = QMDSimpleModel(params)
    states    = []
    prev: Optional[tuple] = None
    try:
        for mu in mu_values:
            s = model.solve_mean_fields(float(mu), initial_guess=prev)
            states.append(s)
            prev = (s.phi_mev, s.delta_mev)
    except Exception as exc:
        _log_error(cfg.run_id, "benchmark scan", exc)
        return None

    pressures, n_q, eps, cs2 = _bm_eos(states)

    onset = next((s.mu_q_mev for s in states if s.phase == "2SC"), None)
    asym_gap = states[-1].gap_mev   # g_Δ·Δ₀ at μ_q = 900 MeV

    # cs2 peak (post-onset, physical range)
    mu_arr = np.array([s.mu_q_mev for s in states])
    phys   = (pressures >= 0.0) & (n_q >= 0.0) & np.isfinite(cs2)
    post   = phys & (mu_arr >= 295.0) & (cs2 >= 0.0) & (cs2 <= 1.0)
    cs2_peak = cs2_peak_muq = None
    if post.any():
        idx_pk      = int(np.nanargmax(cs2[post]))
        cs2_peak    = float(cs2[post][idx_pk])
        cs2_peak_muq = float(mu_arr[post][idx_pk])

    # Save
    bm_path = DATA_DIR / f"section2_benchmark_{tag}.txt"
    _write_benchmark_data(
        bm_path, states, pressures, n_q, eps, cs2, params,
        description=f"Section 2 run {cfg.run_id}: {cfg.param_name}={cfg.param_value}",
    )
    print(f"    Saved benchmark → {bm_path.name}")
    print(f"    onset={onset}  asym_gap={asym_gap:.2f} MeV  cs2_peak={cs2_peak}")

    phi_arr = np.array([s.phi_mev  for s in states])
    gap_arr = np.array([s.gap_mev  for s in states])

    return {
        "mu_q": mu_arr,
        "phi":  phi_arr,
        "gap":  gap_arr,
        "cs2":  cs2,
        "onset": onset,
        "asym_gap": asym_gap,
        "cs2_peak": cs2_peak,
        "cs2_peak_muq": cs2_peak_muq,
        "pressures": pressures,
        "n_q": n_q,
    }


# ---------------------------------------------------------------------------
# Set A loader
# ---------------------------------------------------------------------------

def load_baseline() -> RunResult:
    """Load Set A from output/qmd_stellar/ and qmd_benchmark/."""
    baseline = RunResult(
        run_id="baseline",
        param_name="baseline",
        param_value="baseline",
        notes="Section 1 Set A",
    )

    # Stellar M-R
    mr_path = BASELINE_STELLAR_DIR / "qmd_stars_baseline.txt"
    if mr_path.exists():
        data = np.loadtxt(mr_path, comments="#")
        stable_mask = data[:, 5].astype(bool)
        if stable_mask.any():
            m_st = data[stable_mask, 4]
            r_st = data[stable_mask, 3]
            idx  = int(np.argmax(m_st))
            baseline.M_max     = float(m_st[idx])
            baseline.R_at_Mmax = float(r_st[idx])
        baseline.radius_km   = data[:, 3]
        baseline.mass_msun   = data[:, 4]
        baseline.stable_mask = data[:, 5].astype(bool)
        print(f"  Set A M-R loaded: M_max={baseline.M_max:.4f} M☉")
    else:
        print(f"  WARNING: Set A M-R not found at {mr_path}")

    # Onset from stable EoS
    eos_path = BASELINE_STELLAR_DIR / "qmd_stellar_eos_baseline_stable.txt"
    if eos_path.exists():
        with eos_path.open() as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 16 and parts[15] == "2SC":
                    baseline.onset_muq = float(parts[0])
                    break

    # cs2 from stable EoS (parse numerically, skip "phase" string column)
    if eos_path.exists():
        mu_vals, cs2_vals = [], []
        with eos_path.open() as _f:
            for _line in _f:
                if _line.startswith("#"):
                    continue
                _parts = _line.split()
                if len(_parts) < 14:
                    continue
                try:
                    mu_vals.append(float(_parts[0]))
                    cs2_vals.append(float(_parts[13]))
                except ValueError:
                    pass
        if mu_vals:
            mu_col  = np.array(mu_vals)
            cs2_col = np.array(cs2_vals)
            fin = np.isfinite(cs2_col) & (cs2_col >= 0.0) & (cs2_col <= 1.0)
            if fin.any():
                idx_pk = int(np.nanargmax(cs2_col[fin]))
                baseline.cs2_peak     = float(cs2_col[fin][idx_pk])
                baseline.cs2_peak_muq = float(mu_col[fin][idx_pk])

    # Benchmark data
    bm_path = BASELINE_BENCHMARK_DIR / "qmd_benchmark.txt"
    if bm_path.exists():
        bm = np.loadtxt(bm_path, comments="#")
        baseline.bm_mu_q  = bm[:, 0]
        baseline.bm_phi   = bm[:, 1]
        baseline.bm_gap   = bm[:, 3]
        baseline.bm_cs2   = bm[:, 9]
        phase_2sc         = bm[:, 4]
        onset_cands       = baseline.bm_mu_q[phase_2sc > 0.5]
        baseline.bm_onset = float(onset_cands[0]) if onset_cands.size > 0 else None
        baseline.asymptotic_gap = float(bm[-1, 3])   # gap at mu_q=900 MeV
        # n_nonconverged_pct from benchmark
        n_fail = int(np.sum(bm[:, 10] < 0.5))
        baseline.n_nonconverged_pct = 100.0 * n_fail / max(1, bm.shape[0])

    print(f"  Set A: onset={baseline.onset_muq}  cs2_peak={baseline.cs2_peak}  "
          f"asym_gap={baseline.asymptotic_gap}")
    return baseline


def _load_cached_run(cfg: RunConfig) -> RunResult:
    """Load one Section 2 run from saved tables, without rerunning solvers."""
    tag = f"{cfg.param_name}_{cfg.value_tag}"
    result = RunResult(
        run_id=cfg.run_id,
        param_name=cfg.param_name,
        param_value=cfg.param_value,
    )

    mr_path = DATA_DIR / f"section2_stellar_{tag}.txt"
    if not mr_path.exists():
        raise FileNotFoundError(f"cached M-R table missing: {mr_path}")
    mr = np.loadtxt(mr_path, comments="#")
    if mr.ndim == 1:
        mr = mr[None, :]
    stable = mr[:, 5].astype(bool)
    result.radius_km = mr[:, 3]
    result.mass_msun = mr[:, 4]
    result.stable_mask = stable
    if stable.any():
        m_st = result.mass_msun[stable]
        r_st = result.radius_km[stable]
        idx = int(np.argmax(m_st))
        result.M_max = float(m_st[idx])
        result.R_at_Mmax = float(r_st[idx])

    eos_path = DATA_DIR / f"section2_stellar_{tag}_eos.txt"
    if eos_path.exists():
        mu_vals, cs2_vals = [], []
        with eos_path.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 16 and result.onset_muq is None and parts[15] == "2SC":
                    result.onset_muq = float(parts[0])
                if len(parts) >= 14:
                    try:
                        mu_vals.append(float(parts[0]))
                        cs2_vals.append(float(parts[13]))
                    except ValueError:
                        pass
        if mu_vals:
            mu_col = np.array(mu_vals)
            cs2_col = np.array(cs2_vals)
            valid = np.isfinite(cs2_col) & (cs2_col >= 0.0) & (cs2_col <= 1.0)
            if valid.any():
                idx = int(np.nanargmax(cs2_col[valid]))
                result.cs2_peak = float(cs2_col[valid][idx])
                result.cs2_peak_muq = float(mu_col[valid][idx])

    bm_path = DATA_DIR / f"section2_benchmark_{tag}.txt"
    if bm_path.exists():
        bm = np.loadtxt(bm_path, comments="#")
        if bm.ndim == 1:
            bm = bm[None, :]
        result.bm_mu_q = bm[:, 0]
        result.bm_phi = bm[:, 1]
        result.bm_gap = bm[:, 3]
        result.bm_cs2 = bm[:, 9]
        phase_2sc = bm[:, 4]
        onset_cands = result.bm_mu_q[phase_2sc > 0.5]
        result.bm_onset = float(onset_cands[0]) if onset_cands.size else None
        result.asymptotic_gap = float(result.bm_gap[-1])
        n_fail = int(np.sum(bm[:, 10] < 0.5))
        result.n_nonconverged_pct = 100.0 * n_fail / max(1, bm.shape[0])

        valid = (
            np.isfinite(result.bm_cs2)
            & (result.bm_cs2 >= 0.0)
            & (result.bm_cs2 <= 1.0)
            & (result.bm_mu_q >= 295.0)
        )
        if valid.any():
            idx = int(np.nanargmax(result.bm_cs2[valid]))
            result.cs2_peak = float(result.bm_cs2[valid][idx])
            result.cs2_peak_muq = float(result.bm_mu_q[valid][idx])
    else:
        print(f"  WARNING: cached benchmark table missing for run {cfg.run_id}: {bm_path}")

    return result


def _benchmark_dict(result: RunResult) -> dict:
    return {
        "mu_q": result.bm_mu_q,
        "phi": result.bm_phi,
        "gap": result.bm_gap,
        "cs2": result.bm_cs2,
    }


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, window: int, poly: int = _SMOOTH_POLY) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        return arr
    n = arr.size
    if n <= poly + 2:
        return arr
    w = min(window, n if n % 2 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return arr
    return savgol_filter(arr, window_length=w, polyorder=poly)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_COLOR_BASELINE = SECTION2_MR_BASELINE_COLOR
_DASHED_MSUN = 2.0


def _split_stable_branches(
    radius_km: np.ndarray,
    mass_msun: np.ndarray,
    stable_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stable and unstable M-R branches, attaching the turn-over point."""
    stable = stable_mask.astype(bool)
    unstable = ~stable

    r_st = radius_km[stable]
    m_st = mass_msun[stable]
    r_un = radius_km[unstable]
    m_un = mass_msun[unstable]

    if stable.any() and unstable.any():
        idx = int(np.argmax(m_st))
        r_un = np.concatenate(([r_st[idx]], r_un))
        m_un = np.concatenate(([m_st[idx]], m_un))

    return r_st, m_st, r_un, m_un


def _plot_mr(
    baseline: RunResult,
    runs: list[tuple[RunConfig, RunResult]],
    filename: str,
    title: str,
) -> None:
    """M-R comparison plot: 3 curves in increasing parameter order."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    baseline_label = "Set A"

    def _draw_mr(radius_km, mass_msun, stable_mask, color, label, zorder_base):
        r_st, m_st, r_un, m_un = _split_stable_branches(radius_km, mass_msun, stable_mask)
        ax.plot(r_st, m_st, color=color, lw=LW_VARIATION, ls="-",
                label=label, zorder=zorder_base)
        if r_un.size:
            ax.plot(r_un, m_un, color=color, lw=LW_VARIATION, ls="--", zorder=zorder_base - 1)
        if m_st.size:
            idx = int(np.argmax(m_st))
            ax.plot(r_st[idx], m_st[idx], "o", color=color, ms=6, zorder=zorder_base + 1)

    # Plot in increasing parameter order: lower variation, baseline, upper variation
    if len(runs) >= 1 and runs[0][1].radius_km is not None:
        cfg0, res0 = runs[0]
        _draw_mr(res0.radius_km, res0.mass_msun, res0.stable_mask, cfg0.color, cfg0.label, 3)

    if baseline.radius_km is not None:
        _draw_mr(baseline.radius_km, baseline.mass_msun, baseline.stable_mask,
                 _COLOR_BASELINE, baseline_label, 4)

    if len(runs) >= 2 and runs[1][1].radius_km is not None:
        cfg1, res1 = runs[1]
        _draw_mr(res1.radius_km, res1.mass_msun, res1.stable_mask, cfg1.color, cfg1.label, 3)

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_xlim(8.0, 16.0)
    ax.set_ylim(0.5, 2.4)
    ax.legend(fontsize=9)
    save_figure(PLOTS_DIR / filename)
    print(f"  Saved {filename}")


def _plot_cs2_gdelta(
    baseline: RunResult,
    run_a: tuple[RunConfig, RunResult],
    run_b: tuple[RunConfig, RunResult],
) -> None:
    """cs² comparison, g_Δ variation (benchmark data, no neutrality)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    def _draw_cs2(mu, cs2, color, lw, label):
        valid = (
            np.isfinite(mu) & np.isfinite(cs2)
            & (mu >= CS2_MU_MIN) & (cs2 >= 0.0) & (cs2 <= 1.0)
        )
        mu_p  = mu[valid][:-2]
        cs2_p = cs2[valid].copy()[:-2]
        cs2_p = _smooth(cs2_p, _CS2_GDELTA_SMOOTH_WINDOW)
        ax.plot(mu_p, cs2_p, color=color, lw=lw, label=label)

    # Plot in increasing order: lower variation, baseline, upper variation
    cfg_a, res_a = run_a
    if res_a.bm_cs2 is not None:
        _draw_cs2(res_a.bm_mu_q, res_a.bm_cs2, cfg_a.color, LW_VARIATION, cfg_a.label)

    if baseline.bm_mu_q is not None:
        _draw_cs2(baseline.bm_mu_q, baseline.bm_cs2,
                  _COLOR_BASELINE, LW_VARIATION, "Set A")

    cfg_b, res_b = run_b
    if res_b.bm_cs2 is not None:
        _draw_cs2(res_b.bm_mu_q, res_b.bm_cs2, cfg_b.color, LW_VARIATION, cfg_b.label)

    ax.axhline(1.0 / 3.0, color="gray", ls="--", lw=1.5,
               label=r"Conformal limit $c_s^2=\frac{1}{3}$")
    ax.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$c_s^2/c^2$")
    ax.set_xlim(*CS2_XLIM)
    ax.set_ylim(*CS2_YLIM)
    ax.legend(fontsize=9)
    save_figure(PLOTS_DIR / "section2_cs2_gdelta.pdf")
    print("  Saved section2_cs2_gdelta.pdf")


def _plot_condensates_gdelta(
    baseline: RunResult,
    run_a: tuple[RunConfig, RunResult],
    run_b: tuple[RunConfig, RunResult],
) -> None:
    """Dual panel: φ₀ (top) and g_Δ·Δ₀ (bottom), g_Δ variation, benchmark data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    def _draw(mu, phi, gap, color, lw, label):
        thresh = int(np.searchsorted(mu, 295.0))
        phi_p = phi.copy()
        gap_p = gap.copy()
        if thresh < len(mu):
            phi_p[thresh:] = _smooth(phi[thresh:], 51)
            gap_p[thresh:] = _smooth(gap[thresh:], 51)
        ax1.plot(mu, phi_p, color=color, lw=lw, label=label)
        ax2.plot(mu, gap_p, color=color, lw=lw, label=label)

    # Plot in increasing order: lower variation, baseline, upper variation
    cfg_a, res_a = run_a
    if res_a.bm_mu_q is not None:
        _draw(res_a.bm_mu_q, res_a.bm_phi, res_a.bm_gap, cfg_a.color, LW_VARIATION, cfg_a.label)

    if baseline.bm_mu_q is not None:
        _draw(baseline.bm_mu_q, baseline.bm_phi, baseline.bm_gap,
              _COLOR_BASELINE, LW_VARIATION, "Set A")

    cfg_b, res_b = run_b
    if res_b.bm_mu_q is not None:
        _draw(res_b.bm_mu_q, res_b.bm_phi, res_b.bm_gap, cfg_b.color, LW_VARIATION, cfg_b.label)

    ax1.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax1.set_ylabel(r"$\phi_0\;(\mathrm{MeV})$")
    ax1.set_xlim(200.0, 900.0)
    ax1.legend(fontsize=9)

    ax2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax2.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")
    ax2.set_xlim(200.0, 900.0)
    ax2.legend(fontsize=9)

    save_figure(PLOTS_DIR / "section2_condensates_gdelta.pdf")
    print("  Saved section2_condensates_gdelta.pdf")


def _plot_condensates_mdelta(
    baseline: RunResult,
    run_c: tuple[RunConfig, RunResult],
    run_d: tuple[RunConfig, RunResult],
) -> None:
    """Dual panel: φ₀ (left) and g_Δ·Δ₀ (right), m_Δ variation, benchmark data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    def _draw(mu, phi, gap, color, lw, label):
        thresh = int(np.searchsorted(mu, 295.0))
        phi_p = phi.copy()
        gap_p = gap.copy()
        if thresh < len(mu):
            phi_p[thresh:] = _smooth(phi[thresh:], 51)
            gap_p[thresh:] = _smooth(gap[thresh:], 51)
        ax1.plot(mu, phi_p, color=color, lw=lw, label=label)
        ax2.plot(mu, gap_p, color=color, lw=lw, label=label)

    # Plot in increasing order: lower variation, baseline, upper variation
    cfg_c, res_c = run_c
    if res_c.bm_mu_q is not None:
        _draw(res_c.bm_mu_q, res_c.bm_phi, res_c.bm_gap, cfg_c.color, LW_VARIATION, cfg_c.label)

    if baseline.bm_mu_q is not None:
        _draw(baseline.bm_mu_q, baseline.bm_phi, baseline.bm_gap,
              _COLOR_BASELINE, LW_VARIATION, "Set A")

    cfg_d, res_d = run_d
    if res_d.bm_mu_q is not None:
        _draw(res_d.bm_mu_q, res_d.bm_phi, res_d.bm_gap, cfg_d.color, LW_VARIATION, cfg_d.label)

    ax1.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax1.set_ylabel(r"$\phi_0\;(\mathrm{MeV})$")
    ax1.set_xlim(200.0, 900.0)
    ax1.legend(fontsize=9)

    ax2.set_xlabel(r"$\mu_q\;(\mathrm{MeV})$")
    ax2.set_ylabel(r"$g_\Delta\Delta_0\;(\mathrm{MeV})$")
    ax2.set_xlim(200.0, 900.0)
    ax2.legend(fontsize=9)

    save_figure(PLOTS_DIR / "section2_condensates_mdelta.pdf")
    print("  Saved section2_condensates_mdelta.pdf")


def _generate_section2_plots(
    baseline: RunResult,
    runs: list[RunConfig],
    run_results: list[RunResult],
    bm_data: dict[str, dict],
) -> None:
    """Generate Section 2 plots from in-memory or cached run results."""
    print("\n--- Generating plots ---")

    gdelta_runs = [(r, res) for r, res in zip(runs, run_results) if r.param_name == "gdelta"]
    mdelta_runs = [(r, res) for r, res in zip(runs, run_results) if r.param_name == "mdelta"]
    lam_delta_runs = [(r, res) for r, res in zip(runs, run_results) if r.param_name == "lamdelta"]
    lam3_runs = [(r, res) for r, res in zip(runs, run_results) if r.param_name == "lam3"]

    _plot_mr(baseline, gdelta_runs, "section2_MR_gdelta.pdf",
             r"Mass-radius: $g_\Delta$ variation")
    _plot_mr(baseline, mdelta_runs, "section2_MR_mdelta.pdf",
             r"Mass-radius: $m_\Delta$ variation")
    _plot_mr(baseline, lam_delta_runs, "section2_MR_lamdelta.pdf",
             r"Mass-radius: $\lambda_\Delta$ variation")
    _plot_mr(baseline, lam3_runs, "section2_MR_lam3.pdf",
             r"Mass-radius: $\lambda_3$ variation")

    if len(gdelta_runs) >= 2:
        run_a_pair = gdelta_runs[0]
        run_b_pair = gdelta_runs[1]
        for cfg, res in [run_a_pair, run_b_pair]:
            if cfg.run_id in bm_data and res.bm_mu_q is None:
                bd = bm_data[cfg.run_id]
                res.bm_mu_q = bd["mu_q"]
                res.bm_phi = bd["phi"]
                res.bm_gap = bd["gap"]
                res.bm_cs2 = bd["cs2"]
        _plot_cs2_gdelta(baseline, run_a_pair, run_b_pair)
        _plot_condensates_gdelta(baseline, run_a_pair, run_b_pair)
    else:
        print("  Skipping cs²/condensate plots (g_Δ runs missing).")

    if len(mdelta_runs) >= 2:
        run_c_pair = mdelta_runs[0]
        run_d_pair = mdelta_runs[1]
        for cfg, res in [run_c_pair, run_d_pair]:
            if cfg.run_id in bm_data and res.bm_mu_q is None:
                bd = bm_data[cfg.run_id]
                res.bm_mu_q = bd["mu_q"]
                res.bm_phi = bd["phi"]
                res.bm_gap = bd["gap"]
                res.bm_cs2 = bd["cs2"]
        _plot_condensates_mdelta(baseline, run_c_pair, run_d_pair)
    else:
        print("  Skipping condensate plot (m_Δ runs missing).")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "run_id", "param_name", "param_value",
    "M_max", "R_at_Mmax", "onset_muq",
    "cs2_peak", "cs2_peak_muq", "asymptotic_gap",
    "n_nonconverged_pct", "notes",
]


def write_summary_csv(results: list[RunResult]) -> None:
    ensure_directory(SECTION2_DIR)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": r.run_id,
                "param_name": r.param_name,
                "param_value": r.param_value,
                "M_max": f"{r.M_max:.4f}" if r.M_max is not None else "None",
                "R_at_Mmax": f"{r.R_at_Mmax:.4f}" if r.R_at_Mmax is not None else "None",
                "onset_muq": f"{r.onset_muq:.1f}" if r.onset_muq is not None else "None",
                "cs2_peak": f"{r.cs2_peak:.4f}" if r.cs2_peak is not None else "None",
                "cs2_peak_muq": f"{r.cs2_peak_muq:.1f}" if r.cs2_peak_muq is not None else "None",
                "asymptotic_gap": f"{r.asymptotic_gap:.2f}" if r.asymptotic_gap is not None else "None",
                "n_nonconverged_pct": f"{r.n_nonconverged_pct:.1f}" if r.n_nonconverged_pct is not None else "None",
                "notes": r.notes,
            })
    print(f"\nSaved summary CSV → {SUMMARY_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from cached Section 2 data without running solvers.",
    )
    args = parser.parse_args()

    apply_plot_style()
    ensure_directory(DATA_DIR)
    ensure_directory(PLOTS_DIR)

    t_start = time.time()

    print("=" * 70)
    print("Section 2: Parameter Sensitivity Sweep — Set A")
    print("=" * 70)

    # Load Set A
    print("\n--- Loading Section 1 Set A ---")
    baseline = load_baseline()

    runs = _make_runs()
    all_results: list[RunResult] = [baseline]

    # Store benchmark data keyed by run_id for cross-referencing
    bm_data: dict[str, dict] = {}

    if args.plot_only:
        print("\n--- Loading cached Section 2 sweep runs ---")
        for cfg in runs:
            result = _load_cached_run(cfg)
            all_results.append(result)
            if result.bm_mu_q is not None:
                bm_data[cfg.run_id] = _benchmark_dict(result)
            stable = int(np.sum(result.stable_mask)) if result.stable_mask is not None else 0
            unstable = (
                int(np.sum(~result.stable_mask.astype(bool)))
                if result.stable_mask is not None else 0
            )
            print(
                f"  Set {cfg.run_id}: loaded {stable} stable + "
                f"{unstable} unstable TOV points"
            )

        _generate_section2_plots(baseline, runs, all_results[1:], bm_data)
        print("\nSection 2 plots regenerated from cached data.")
        return

    # --- Execute 8 runs ---
    for cfg in runs:
        print(f"\n{'='*60}")
        print(f"Set {cfg.run_id}: {cfg.param_name} = {cfg.param_value}")
        print(f"{'='*60}")

        # ---- Stellar pipeline ----
        seq, stellar_result = None, None
        try:
            seq, stellar_result = run_stellar_pipeline(cfg)
        except Exception as exc:
            _log_error(cfg.run_id, "stellar pipeline (uncaught)", exc)

        if stellar_result is None:
            stellar_result = RunResult(
                run_id=cfg.run_id, param_name=cfg.param_name,
                param_value=cfg.param_value, failed=True,
                notes="Stellar pipeline failed completely",
            )
            print(f"  Run {cfg.run_id} stellar FAILED — continuing.")

        # ---- Benchmark pipeline ----
        bm_result = None
        try:
            bm_result = run_benchmark_pipeline(cfg)
        except Exception as exc:
            _log_error(cfg.run_id, "benchmark pipeline (uncaught)", exc)

        if bm_result is not None:
            stellar_result.asymptotic_gap = bm_result["asym_gap"]
            stellar_result.bm_mu_q  = bm_result["mu_q"]
            stellar_result.bm_phi   = bm_result["phi"]
            stellar_result.bm_gap   = bm_result["gap"]
            stellar_result.bm_cs2   = bm_result["cs2"]
            stellar_result.bm_onset = bm_result["onset"]
            # Prefer benchmark cs2_peak (5000 pts, better resolution)
            if bm_result["cs2_peak"] is not None:
                stellar_result.cs2_peak     = bm_result["cs2_peak"]
                stellar_result.cs2_peak_muq = bm_result["cs2_peak_muq"]
            bm_data[cfg.run_id] = bm_result

        # ---- Special checks ----
        if cfg.run_id == "C":
            gap_b = stellar_result.asymptotic_gap
            if gap_b is not None:
                exceeds = gap_b > KURKELA_BOUND_MEV
                note = (
                    f"Set C: g_Δ·Δ₀(900 MeV) = {gap_b:.2f} MeV "
                    f"{'EXCEEDS' if exceeds else 'below'} Kurkela bound {KURKELA_BOUND_MEV} MeV"
                )
                print(f"\n  *** {note} ***")
                if stellar_result.notes:
                    stellar_result.notes += "; " + note
                else:
                    stellar_result.notes = note

        if cfg.run_id == "E":
            if stellar_result.onset_muq is None:
                note = "No 2SC phase within μ_q ≤ 900 MeV; M_max approaches unpaired QM value"
                print(f"\n  *** Set E: {note} ***")
                if stellar_result.notes:
                    stellar_result.notes += "; " + note
                else:
                    stellar_result.notes = note

        all_results.append(stellar_result)

    # --- Write summary CSV ---
    write_summary_csv(all_results)

    # --- Generate plots ---
    _generate_section2_plots(baseline, runs, all_results[1:], bm_data)

    # --- Final summary ---
    t_end = time.time()
    elapsed = t_end - t_start

    print("\n" + "=" * 70)
    print("SECTION 2 SWEEP COMPLETE")
    print("=" * 70)

    n_attempted = len(runs)
    n_failed    = sum(1 for r in all_results[1:] if r.failed)
    n_partial   = sum(1 for r in all_results[1:] if r.partial and not r.failed)
    n_completed = n_attempted - n_failed

    print(f"Total runs completed / attempted: {n_completed} / {n_attempted}")

    if n_partial:
        print(f"PARTIAL runs: {[r.run_id for r in all_results[1:] if r.partial]}")
    if n_failed:
        print(f"FAILED  runs: {[r.run_id for r in all_results[1:] if r.failed]}")

    m_maxes = [r.M_max for r in all_results if r.M_max is not None]
    if m_maxes:
        print(f"M_max range across all {len(m_maxes)} runs: "
              f"{min(m_maxes):.4f} – {max(m_maxes):.4f} M☉")

    # Which parameter produces largest M_max variation
    param_groups = {}
    for r in all_results:
        if r.param_name not in ("baseline",) and r.M_max is not None:
            param_groups.setdefault(r.param_name, []).append(r.M_max)
    if param_groups:
        variations = {p: max(v) - min(v) for p, v in param_groups.items()}
        biggest = max(variations, key=variations.get)
        print(f"Largest M_max variation: {biggest} (ΔM_max = {variations[biggest]:.4f} M☉)")

    # 2.0 M_sun constraint
    all_above_2msun = all(m >= 2.0 for m in m_maxes) if m_maxes else False
    if all_above_2msun:
        print("2.0 M☉ constraint: satisfied across ALL runs")
    else:
        below = [r.run_id for r in all_results if r.M_max is not None and r.M_max < 2.0]
        if below:
            print(f"2.0 M☉ constraint: NOT satisfied for run(s) {below}")
        else:
            print("2.0 M☉ constraint: some runs have no M_max computed")

    # Set C gap check
    run_b_res = next((r for r in all_results if r.run_id == "C"), None)
    if run_b_res and run_b_res.asymptotic_gap is not None:
        gap_b = run_b_res.asymptotic_gap
        print(f"Set C (g_Δ=2.5g): g_Δ·Δ₀(900 MeV) = {gap_b:.2f} MeV  "
              f"({'EXCEEDS' if gap_b > KURKELA_BOUND_MEV else 'below'} "
              f"Kurkela bound {KURKELA_BOUND_MEV} MeV)")

    print(f"Total wall-clock time: {elapsed/60:.1f} min ({elapsed:.0f} s)")

    # Print table
    print("\n--- Set Summary ---")
    print(f"{'ID':>4}  {'param':>10}  {'value':>12}  {'M_max':>7}  {'R':>6}  "
          f"{'onset':>6}  {'cs2pk':>6}  {'gap':>6}  {'fail%':>5}  notes")
    print("-" * 110)
    for r in all_results:
        print(
            f"{r.run_id:>4}  {r.param_name:>10}  {r.param_value:>12}  "
            f"{r.M_max or 'N/A':>7}  {r.R_at_Mmax or 'N/A':>6}  "
            f"{str(r.onset_muq or 'None'):>6}  "
            f"{r.cs2_peak or 'N/A':>6}  {r.asymptotic_gap or 'N/A':>6}  "
            f"{r.n_nonconverged_pct or 'N/A':>5}  {r.notes[:60]}"
        )


if __name__ == "__main__":
    main()
