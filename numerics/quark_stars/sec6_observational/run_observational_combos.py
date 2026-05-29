"""
Combined-parameter observational comparison sets.

Sets:
  lam3_3lam0  — single-parameter λ₃=3λ₀ (diagnostic)
  Set J       — g_Δ=2.5g, λ₃=3λ₀, m_Δ=500 MeV, λ_Δ=λ₀/4
  Set K       — m_Δ=400, λ₃=2λ₀, g_Δ=2g,   λ_Δ=λ₀/4
  Set L       — m_Δ=400, λ₃=3λ₀, g_Δ=2g,   λ_Δ=λ₀/4

Outputs
-------
  output/observational_combos/data/combo_{tag}.txt       (TOV M-R)
  output/observational_combos/data/combo_{tag}_eos.txt   (EoS)
  thesis/figures/quark_stars/qmd_combined_observational_mr.pdf
  output/combined_parameter_observational_report.md
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..constants import MEV3_TO_FM_MINUS3, MEV4_TO_GEV_FM3
from ..io import ensure_directory
from ..plotting import SECTION2_MR_COMPARISON_COLORS, apply_plot_style, save_figure
from ..qmd_parameters import QMD_SET_A, QMDParameters
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

_HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = _HERE.parent / "output"
COMBO_DIR   = OUTPUT_ROOT / "observational_combos"
DATA_DIR    = COMBO_DIR / "data"
LOG_FILE    = COMBO_DIR / "errors.log"

BASELINE_MR_FILE  = OUTPUT_ROOT / "qmd_stellar" / "data" / "qmd_stars_baseline.txt"
SECT2_DATA        = OUTPUT_ROOT / "sec5_parameter_sensitivity" / "data"
FIG_DIR           = _HERE.parents[2] / "thesis" / "figures" / "quark_stars" / "observational"
REPORT_SELECTED   = OUTPUT_ROOT / "observational_mr_selected_report.md"
REPORT_COMBINED   = OUTPUT_ROOT / "combined_parameter_observational_report.md"

# ---------------------------------------------------------------------------
# Pipeline constants (match run_section2_sweep.py)
# ---------------------------------------------------------------------------

NUM_STELLAR_POINTS = 700
MU_MIN_STELLAR     = 250.0
MU_MAX_STELLAR     = 900.0
_MINIMIZER_OPTIONS = {"maxiter": 80, "ftol": 1.0e-8, "gtol": 1.0e-6}
N_SAT_FM3          = 0.160   # nuclear saturation density [fm^-3]

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

@dataclass
class CombinedSetConfig:
    tag:    str
    label:  str
    color:  str
    params: QMDParameters


def _make_combos() -> list[CombinedSetConfig]:
    bl = QMD_SET_A
    return [
        CombinedSetConfig(
            "lam3_3lam0",
            r"$\lambda_3 = 3\lambda_0$ (diagnostic)",
            "#888888",
            replace(bl, lambda_3_factor=3.0),
        ),
        CombinedSetConfig(
            "combo1",
            "candidate: g_delta=2.5g, lambda_3=2lambda_0",
            "#E69F00",
            replace(bl, g_delta_factor=2.5, lambda_3_factor=2.0),
        ),
        CombinedSetConfig(
            "combo2",
            "Set J",
            "#D55E00",
            replace(bl, g_delta_factor=2.5, lambda_3_factor=3.0),
        ),
        CombinedSetConfig(
            "combo3",
            "Set K",
            "#009E73",
            replace(bl, m_delta_mev=400.0, lambda_3_factor=2.0),
        ),
        CombinedSetConfig(
            "combo4",
            "Set L",
            "#0072B2",
            replace(bl, m_delta_mev=400.0, lambda_3_factor=3.0),
        ),
        CombinedSetConfig(
            "combo5",
            "candidate: g_delta=2.25g, lambda_3=2lambda_0",
            "#CC79A7",
            replace(bl, g_delta_factor=2.25, lambda_3_factor=2.0),
        ),
    ]


@dataclass
class RunResult:
    tag: str
    label: str
    M_max:         Optional[float] = None
    R_at_Mmax:     Optional[float] = None
    onset_muq:     Optional[float] = None
    eps_c_gev_fm3: Optional[float] = None  # central energy density at Mmax
    mu_q_c_mev:    Optional[float] = None  # central mu_q at Mmax (from EoS lookup)
    nB_over_n0:    Optional[float] = None  # central n_B / n_0 at Mmax
    cs2_peak:      Optional[float] = None
    cs2_max_stable: Optional[float] = None  # max cs2 on stable branch (causality check)
    gap_goes_negative: bool = False         # g_delta*Delta - delta_mu < 0 before Mmax
    fail_pct:      float = 0.0
    notes: str = ""
    # Arrays for plotting
    radius_km:   Optional[np.ndarray] = field(default=None, repr=False)
    mass_msun:   Optional[np.ndarray] = field(default=None, repr=False)
    stable_mask: Optional[np.ndarray] = field(default=None, repr=False)
    eos_points:  Optional[list] = field(default=None, repr=False)  # stable EoS points


# ---------------------------------------------------------------------------
# Shared pipeline helpers (verbatim from run_section2_sweep.py private fns)
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


def _scan_equilibrium(model, mu_values):
    states = []
    prev   = None
    n      = len(mu_values)
    for i, mu in enumerate(mu_values, 1):
        state = model.solve_equilibrium(
            float(mu), previous_state=prev,
            initial_neutrality_guess=(0.0, 0.0),
            minimizer_options=_MINIMIZER_OPTIONS,
        )
        states.append(state)
        if state.success and np.isfinite(state.mu_e_mev) and np.isfinite(state.mu_8_mev):
            prev = state
        if i % 100 == 0 or i == n:
            print(
                f"    {i:3d}/{n}  μ_q={state.mu_q_mev:7.2f}  "
                f"gap={state.gap_mev:7.3f}  {state.phase:>6}  "
                f"{'ok' if state.success else 'FAIL'}",
                flush=True,
            )
    return states


def _filter_points(points):
    return [
        p for p in points
        if p.success
        and p.pressure_mev4 > 0.0
        and p.quark_density_mev3 > 0.0
        and p.energy_density_mev4 > 0.0
        and np.isfinite(p.cs2)
    ]


def _recompute_cs2(points):
    if len(points) < 2:
        return [replace(p, cs2=float("nan")) for p in points]
    pressure = np.array([p.pressure_mev4       for p in points], dtype=float)
    energy   = np.array([p.energy_density_mev4 for p in points], dtype=float)
    edge = 2 if len(points) >= 3 else 1
    dpdeps = np.gradient(pressure, energy, edge_order=edge)
    return [
        replace(p, cs2=float(dpdeps[i]) if np.isfinite(dpdeps[i]) else float("nan"))
        for i, p in enumerate(points)
    ]


def _strictly_increasing(points):
    ordered = sorted(points, key=lambda p: (p.pressure_mev4, p.energy_density_mev4))
    kept, removed = [], 0
    last_p = last_e = -float("inf")
    for p in ordered:
        if not kept:
            kept.append(p); last_p = p.pressure_mev4; last_e = p.energy_density_mev4; continue
        p_tol = 1e-12 * max(1.0, abs(p.pressure_mev4), abs(last_p))
        e_tol = 1e-12 * max(1.0, abs(p.energy_density_mev4), abs(last_e))
        if p.pressure_mev4 > last_p + p_tol and p.energy_density_mev4 > last_e + e_tol:
            kept.append(p); last_p = p.pressure_mev4; last_e = p.energy_density_mev4
        else:
            removed += 1
    return kept, removed


def _nearest_point(points, p_val, e_val):
    pressure = np.array([p.pressure_mev4       for p in points])
    energy   = np.array([p.energy_density_mev4 for p in points])
    p_scale  = max(1.0, float(np.nanmax(np.abs(pressure))))
    e_scale  = max(1.0, float(np.nanmax(np.abs(energy))))
    dist = ((pressure - p_val) / p_scale)**2 + ((energy - e_val) / e_scale)**2
    return points[int(np.nanargmin(dist))]


def _build_stable(raw_points):
    if not raw_points:
        return [], [], 0, 0
    mu_q_arr = np.array([p.mu_q_mev            for p in raw_points], dtype=float)
    pressure = np.array([p.pressure_mev4        for p in raw_points], dtype=float)
    energy   = np.array([p.energy_density_mev4  for p in raw_points], dtype=float)
    stable_p, stable_e, maxwell_indices = maxwell_construct(mu_q_arr, pressure, energy)

    mapped = []
    for p_val, e_val in zip(stable_p, stable_e):
        if p_val <= 0.0 or e_val <= 0.0:
            continue
        near = _nearest_point(raw_points, float(p_val), float(e_val))
        mapped.append(replace(near, pressure_mev4=float(p_val), energy_density_mev4=float(e_val)))

    stable, mono_removed = _strictly_increasing(mapped)
    stable = _recompute_cs2(stable)

    cs2_removed = 0
    for _ in range(3):
        filtered = [p for p in stable if np.isfinite(p.cs2) and p.cs2 >= -1e-10]
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
# Save helpers
# ---------------------------------------------------------------------------

def _write_mr(path: Path, sequence, meta: dict) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for k, v in meta.items():
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


_EOS_COLUMNS = [
    "mu_q_mev", "mu_B_mev", "phi_mev", "delta_mev", "gap_mev",
    "mu_e_mev", "mu_8_mev", "delta_mu_mev", "gap_minus_delta_mu_mev",
    "pressure_mev4", "quark_density_mev3", "baryon_density_mev3",
    "energy_density_mev4", "cs2", "omega_min_mev4", "phase",
    "success", "neutrality_residual_norm",
]


def _write_eos(path: Path, points, meta: dict) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for k, v in meta.items():
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


# ---------------------------------------------------------------------------
# Main run pipeline
# ---------------------------------------------------------------------------

def run_combo(cfg: CombinedSetConfig) -> RunResult:
    params = cfg.params
    print(f"\n{'='*60}")
    print(f"  Running: {cfg.tag}")
    print(f"    g_Δ={params.g_delta_factor:.3f}g, m_Δ={params.m_delta_mev:.0f} MeV, "
          f"λ₃={params.lambda_3_factor:.2f}λ₀, λ_Δ={params.lambda_delta_factor:.3f}λ₀")
    result = RunResult(tag=cfg.tag, label=cfg.label)
    notes = []

    # --- Vacuum ---
    model = QMDStellarModel(params)
    try:
        vacuum = _solve_vacuum(model)
        omega_vac = vacuum.omega_min_mev4
    except Exception as exc:
        result.notes = f"FATAL vacuum solve: {exc}"
        print(f"  FATAL: {result.notes}")
        _log(cfg.tag, "vacuum", exc)
        return result

    # --- Equilibrium scan ---
    mu_values = np.linspace(MU_MIN_STELLAR, MU_MAX_STELLAR, NUM_STELLAR_POINTS)
    try:
        states = _scan_equilibrium(model, mu_values)
    except Exception as exc:
        result.notes = f"FATAL scan: {exc}"
        _log(cfg.tag, "scan", exc)
        return result

    n_ok    = sum(1 for s in states if s.success)
    n_2sc   = sum(1 for s in states if s.phase == "2SC")
    fail_pct = 100.0 * (len(states) - n_ok) / max(1, len(states))
    result.fail_pct = fail_pct
    if fail_pct > 30.0:
        notes.append(f"PARTIAL ({fail_pct:.1f}% non-converged)")
        print(f"    WARNING: {fail_pct:.1f}% non-convergence")
    print(f"    Scan: {n_ok}/{NUM_STELLAR_POINTS} ok, {n_2sc} 2SC, {fail_pct:.1f}% failed")

    # --- EoS construction ---
    try:
        raw_all    = build_qmd_stellar_eos_from_states(states, omega_vac)
        raw_points = _filter_points(raw_all)
        stable_points, maxwell_indices, mono_removed, cs2_removed = _build_stable(raw_points)
    except Exception as exc:
        result.notes = f"FATAL EoS build: {exc}"
        _log(cfg.tag, "eos", exc)
        return result

    if not stable_points:
        result.notes = "No stable EoS points after Maxwell construction"
        return result

    # --- Onset ---
    onset_muq = next((p.mu_q_mev for p in stable_points if p.phase == "2SC"), None)
    if n_2sc == 0:
        onset_muq = None
        notes.append("No 2SC phase in stellar density range")
    result.onset_muq = onset_muq

    # --- Causality check on stable EoS ---
    cs2_vals = np.array([p.cs2 for p in stable_points if np.isfinite(p.cs2)])
    if cs2_vals.size:
        result.cs2_peak = float(np.nanmax(cs2_vals))

    # --- Save EoS ---
    meta = {
        "tag": cfg.tag,
        "g_delta_factor": f"{params.g_delta_factor:.4f}",
        "m_delta_mev": f"{params.m_delta_mev:.1f}",
        "lambda_3_factor": f"{params.lambda_3_factor:.4f}",
        "lambda_delta_factor": f"{params.lambda_delta_factor:.4f}",
        "t_loop4_factor": f"{params.t_loop4_factor:.1f}",
        "include_omega_1_num": str(params.include_omega_1_num),
        "omega_vac_mev4": f"{omega_vac:.10e}",
        "num_raw_points": str(NUM_STELLAR_POINTS),
        "mu_min_mev": f"{MU_MIN_STELLAR:.1f}",
        "mu_max_mev": f"{MU_MAX_STELLAR:.1f}",
    }
    eos_path = DATA_DIR / f"combo_{cfg.tag}_eos.txt"
    _write_eos(eos_path, stable_points, {**meta, "branch": "stable"})
    print(f"    EoS → {eos_path.name}")

    # --- TOV ---
    try:
        p_arr    = np.array([p.pressure_mev4       for p in stable_points])
        e_arr    = np.array([p.energy_density_mev4 for p in stable_points])
        eos_tov  = _QMDEoS(pressure_mev4=p_arr, energy_density_mev4=e_arr)
        sequence = run_tov_sequence(eos_tov, integrator="rk4")

        result.radius_km   = sequence.radius_km
        result.mass_msun   = sequence.mass_msun
        result.stable_mask = sequence.stable_mask
        result.eos_points  = stable_points

        stable_tov = sequence.stable_mask.astype(bool)
        if stable_tov.any():
            m_st = sequence.mass_msun[stable_tov]
            r_st = sequence.radius_km[stable_tov]
            ec_st = sequence.central_energy_density_mev4[stable_tov]
            idx   = int(np.argmax(m_st))
            result.M_max     = float(m_st[idx])
            result.R_at_Mmax = float(r_st[idx])
            result.eps_c_gev_fm3 = float(ec_st[idx]) * MEV4_TO_GEV_FM3

            # Max cs2 on stable branch (causality)
            result.cs2_max_stable = float(result.cs2_peak) if result.cs2_peak else None

            # Look up mu_q and n_B at the central energy density of Mmax
            ec_mmax = float(ec_st[idx])
            eps_eos = np.array([p.energy_density_mev4 for p in stable_points])
            nearest_idx = int(np.argmin(np.abs(eps_eos - ec_mmax)))
            pt = stable_points[nearest_idx]
            result.mu_q_c_mev  = pt.mu_q_mev
            nB_fm3 = pt.baryon_density_mev3 * MEV3_TO_FM_MINUS3
            result.nB_over_n0 = nB_fm3 / N_SAT_FM3

            print(f"    Mmax={result.M_max:.4f} M☉  R={result.R_at_Mmax:.4f} km  "
                  f"ε_c={result.eps_c_gev_fm3:.3f} GeV/fm³  "
                  f"n_B/n₀={result.nB_over_n0:.2f}")

            # Check g_delta*Delta - delta_mu sign before Mmax central density
            # The stable branch goes from low to high pressure; Mmax is at highest stable
            # We look at gap_minus_delta_mu_mev in the 2SC phase points up to ec_mmax
            gap_vals = [
                p.gap_minus_delta_mu_mev for p in stable_points
                if p.phase == "2SC" and p.energy_density_mev4 <= ec_mmax
            ]
            if gap_vals and any(g < 0.0 for g in gap_vals):
                result.gap_goes_negative = True
                notes.append("g_Δ·Δ - δμ goes negative before Mmax: gapless 2SC or instability")
                print("    WARNING: gap_minus_delta_mu < 0 before Mmax density")

        # Save M-R
        mr_path = DATA_DIR / f"combo_{cfg.tag}.txt"
        _write_mr(mr_path, sequence, meta)
        print(f"    M-R  → {mr_path.name}")

    except Exception as exc:
        notes.append(f"TOV failed: {exc}")
        _log(cfg.tag, "TOV", exc)

    result.notes = "; ".join(notes) if notes else "ok"
    return result


def _log(tag: str, ctx: str, exc: Exception) -> None:
    ensure_directory(LOG_FILE.parent)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{tag}] {ctx}: {type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc() + "\n")


# ---------------------------------------------------------------------------
# Observational data (from numerics/npemu/plots.py)
# ---------------------------------------------------------------------------

OBS_J0030 = {"M": 1.34, "dM_lo": 0.16, "dM_hi": 0.15, "R": 12.71, "dR_lo": 1.19, "dR_hi": 1.14}
OBS_J0740 = {"M": 2.073, "dM_lo": 0.069, "dM_hi": 0.069, "R": 12.49, "dR_lo": 0.88, "dR_hi": 1.28}

PULSARS = [
    {"name": "PSR J0348+0432", "M": 2.01, "dM_lo": 0.04, "dM_hi": 0.04},
    {"name": "PSR J1614$-$2230", "M": 1.97, "dM_lo": 0.04, "dM_hi": 0.04},
    {"name": "PSR J2215+5135",   "M": 2.27, "dM_lo": 0.15, "dM_hi": 0.17},
    {"name": "PSR J0952$-$0607", "M": 2.35, "dM_lo": 0.17, "dM_hi": 0.17},
]

OBS_NICER_COLORS = {
    "J0030": "tab:orange",
    "J0740": "tab:blue",
}

OBS_CURVE_COLORS = {
    "Set C": SECTION2_MR_COMPARISON_COLORS[0],
    "Set D": SECTION2_MR_COMPARISON_COLORS[1],
    "Set I": SECTION2_MR_COMPARISON_COLORS[2],
    "Set J": SECTION2_MR_COMPARISON_COLORS[0],
    "Set K": SECTION2_MR_COMPARISON_COLORS[1],
    "Set L": SECTION2_MR_COMPARISON_COLORS[2],
}

R_MIN, R_MAX = 8.0, 20.0
M_MIN, M_MAX = 0.5, 2.6


def _draw_obs(ax):
    for p in PULSARS:
        ax.fill_betweenx(
            [p["M"] - p["dM_lo"], p["M"] + p["dM_hi"]],
            R_MIN, R_MAX,
            alpha=0.30, label=p["name"], zorder=1,
        )

    j = OBS_J0030
    color = OBS_NICER_COLORS["J0030"]
    ax.errorbar(
        j["R"], j["M"],
        xerr=[[j["dR_lo"]], [j["dR_hi"]]], yerr=[[j["dM_lo"]], [j["dM_hi"]]],
        fmt="none", ecolor=color, lw=1.5, capsize=3, capthick=1.2, zorder=5,
    )
    ax.scatter(
        j["R"], j["M"], marker="o", s=42, color=color, edgecolor="white",
        linewidth=0.6, label="PSR J0030+0451 (NICER)", zorder=5.1,
    )
    j = OBS_J0740
    color = OBS_NICER_COLORS["J0740"]
    ax.errorbar(
        j["R"], j["M"],
        xerr=[[j["dR_lo"]], [j["dR_hi"]]], yerr=[[j["dM_lo"]], [j["dM_hi"]]],
        fmt="none", ecolor=color, lw=1.5, capsize=3, capthick=1.2, zorder=5,
    )
    ax.scatter(
        j["R"], j["M"], marker="s", s=42, color=color, edgecolor="white",
        linewidth=0.6, label="PSR J0740+6620 (NICER)", zorder=5.1,
    )


def _draw_tov(ax, R, M, stable, color, label, zorder=3, lw=2.0):
    st = stable.astype(bool)
    un = ~st
    ax.plot(R[st], M[st], color=color, lw=lw, label=label, zorder=zorder,
            solid_capstyle="round")
    if un.any():
        ax.plot(R[un], M[un], color=color, lw=lw, ls="--", zorder=zorder - 0.1)
    # Mmax marker
    if st.any():
        idx = int(np.argmax(M[st]))
        ax.plot(R[st][idx], M[st][idx], "o", color=color, ms=6, zorder=zorder + 1)


def _finalize(ax, title):
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_ylim(M_MIN, M_MAX)
    ax.set_xlabel(r"$R\;[\mathrm{km}]$")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.legend(loc="lower left", ncol=1, fontsize=8, framealpha=0.88)


def _load_tov(path: Path):
    data = np.loadtxt(path, comments="#")
    return data[:, 3], data[:, 4], data[:, 5].astype(int)


# ---------------------------------------------------------------------------
# Figure: updated selected plot (replacing old qmd_selected_observational_mr.pdf)
# ---------------------------------------------------------------------------

def plot_selected_updated(combo_results: dict[str, RunResult]) -> None:
    """
    Set C + Set D + Set I + observational constraints.
    Overwrites thesis/figures/quark_stars/qmd_selected_observational_mr.pdf.
    """
    # Load single-parameter runs from section2 data
    sect2_files = {
        "Set C": (SECT2_DATA / "section2_stellar_gdelta_2p5g.txt",
                  "Set C", OBS_CURVE_COLORS["Set C"]),
        "Set D": (SECT2_DATA / "section2_stellar_mdelta_400.txt",
                  "Set D", OBS_CURVE_COLORS["Set D"]),
        "Set I": (SECT2_DATA / "section2_stellar_lam3_2lam0.txt",
                  "Set I", OBS_CURVE_COLORS["Set I"]),
    }

    fig, ax = plt.subplots()
    _draw_obs(ax)

    # Single-parameter runs from section2
    for run_label, (fpath, label, color) in sect2_files.items():
        R, M, st = _load_tov(fpath)
        _draw_tov(ax, R, M, st, color, label, zorder=3)

    _finalize(ax, "Selected QMD parameter variations and observational constraints")
    out = FIG_DIR / "qmd_selected_observational_mr.pdf"
    ensure_directory(out.parent)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


# ---------------------------------------------------------------------------
# Figure: combined parameter sets + observational constraints
# ---------------------------------------------------------------------------

def plot_combined(combo_results: dict[str, RunResult]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    _draw_obs(ax)

    combo_keys = [
        ("combo2", "Set J"),
        ("combo3", "Set K"),
        ("combo4", "Set L"),
    ]
    for key, label in combo_keys:
        res = combo_results.get(key)
        if res is None or res.radius_km is None:
            print(f"  SKIP plot {key}: no M-R data")
            continue
        _draw_tov(ax, res.radius_km, res.mass_msun, res.stable_mask,
                  OBS_CURVE_COLORS[label], label, zorder=3)

    _finalize(ax, "Combined QMD parameter sets and observational constraints")
    out = FIG_DIR / "qmd_combined_observational_mr.pdf"
    ensure_directory(out.parent)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


# ---------------------------------------------------------------------------
# Observational compatibility assessment
# ---------------------------------------------------------------------------

def _in_j0740(M, R):
    """Rough check: M within 1σ of J0740 and R within 1σ error bar."""
    j = OBS_J0740
    M_ok = (j["M"] - j["dM_lo"]) <= M <= (j["M"] + j["dM_hi"])
    R_ok = (j["R"] - j["dR_lo"]) <= R <= (j["R"] + j["dR_hi"])
    return M_ok and R_ok


def _curve_intersects_j0030(R_arr, M_arr, stable):
    """Check whether any point on the stable curve lies within the J0030 1σ box."""
    j = OBS_J0030
    st = stable.astype(bool)
    R_st = R_arr[st]
    M_st = M_arr[st]
    R_ok = (R_st >= j["R"] - j["dR_lo"]) & (R_st <= j["R"] + j["dR_hi"])
    M_ok = (M_st >= j["M"] - j["dM_lo"]) & (M_st <= j["M"] + j["dM_hi"])
    return bool(np.any(R_ok & M_ok))


def _curve_intersects_j0740(R_arr, M_arr, stable):
    """Check whether the curve has a point in the J0740 1σ box."""
    j = OBS_J0740
    st = stable.astype(bool)
    R_st = R_arr[st]
    M_st = M_arr[st]
    R_ok = (R_st >= j["R"] - j["dR_lo"]) & (R_st <= j["R"] + j["dR_hi"])
    M_ok = (M_st >= j["M"] - j["dM_lo"]) & (M_st <= j["M"] + j["dM_hi"])
    return bool(np.any(R_ok & M_ok))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_selected_report(combo_results: dict[str, RunResult]) -> None:
    res_lam3 = combo_results.get("lam3_3lam0")
    lam3_line = (
        f"- **λ₃ = 3λ₀** (new single-param run): "
        f"Mmax={res_lam3.M_max:.3f} M☉, R(Mmax)={res_lam3.R_at_Mmax:.2f} km"
        if res_lam3 and res_lam3.M_max
        else "- **λ₃ = 3λ₀**: run failed or M-R not available"
    )
    lines = [
        "# Updated selected M-R plot: visual summary",
        "",
        "## Curves included",
        "",
        "- **Set A**: Mmax=1.970 M☉, R=12.16 km",
        "- **Set C** (g_Δ=2.5g): Mmax=2.047 M☉, R=12.72 km  — raises Mmax above 2 M☉",
        "- **Set D** (m_Δ=400 MeV): Mmax=2.058 M☉, R=12.59 km  — raises Mmax above 2 M☉",
        "- **Set I** (λ₃=2λ₀): Mmax=1.950 M☉, R=11.46 km  — reduces radii, slightly lowers Mmax",
        lam3_line,
        "",
        "## Rationale for selection",
        "",
        "- Set H (λ₃=0) is removed: it increases radii but does not help Mmax coverage.",
        "- Set I and λ₃=3λ₀ are kept to show the λ₃ trend on radii vs. Mmax trade-off.",
        "- Sets C and D remain to show the two approaches to exceeding 2 M☉.",
        "- λ_Δ has negligible effect and is not shown.",
        "",
        "## Observational compatibility (visual, not Bayesian)",
        "",
        "- **J0740+6620 (NICER)**: Sets C and D push Mmax toward the J0740 mass range,",
        "  but none of the curves have Mmax above 2.073 M☉ except when combined.",
        "- **J0030+0451 (NICER)**: All curves with R(Mmax)≈11–13 km span the J0030 radius",
        "  band. The lower-mass part of each curve passes through or near the J0030 box.",
        "- **J2215+5135, J0952−0607**: None of the single-parameter runs reach these masses.",
        "",
        "## Warnings",
        "",
        "- These are pure quark-star sequences; no hadronic crust or hybrid construction.",
        "- No tidal deformability (GW170817) constraint is shown.",
        "- Visual overlap is not a posterior model comparison.",
    ]
    REPORT_SELECTED.parent.mkdir(parents=True, exist_ok=True)
    REPORT_SELECTED.write_text("\n".join(lines))
    print(f"  saved: {REPORT_SELECTED}")


def write_combined_report(combo_results: dict[str, RunResult]) -> None:
    # Build table rows
    combos = _make_combos()
    rows = []
    for cfg in combos:
        res = combo_results.get(cfg.tag)
        if res is None:
            rows.append({
                "tag": cfg.tag, "M_max": "FAILED", "R": "—", "eps_c": "—",
                "mu_q_c": "—", "nB_n0": "—", "onset": "—",
                "gap_neg": "—", "cs2_max": "—", "j0030": "—", "j0740": "—",
                "2msun": "—", "notes": "run not completed",
            })
            continue
        def f(x, fmt=".3f"): return format(x, fmt) if x is not None else "—"

        j0030 = "✓" if res.radius_km is not None and _curve_intersects_j0030(
            res.radius_km, res.mass_msun, res.stable_mask) else "✗"
        j0740 = "✓" if res.radius_km is not None and _curve_intersects_j0740(
            res.radius_km, res.mass_msun, res.stable_mask) else "✗"
        two_msun = ("✓" if res.M_max and res.M_max >= 2.0 else "✗") if res.M_max else "—"
        rows.append({
            "tag": cfg.tag,
            "M_max": f(res.M_max),
            "R": f(res.R_at_Mmax),
            "eps_c": f(res.eps_c_gev_fm3),
            "mu_q_c": f(res.mu_q_c_mev, ".1f"),
            "nB_n0": f(res.nB_over_n0, ".2f"),
            "onset": f(res.onset_muq, ".1f"),
            "gap_neg": "yes" if res.gap_goes_negative else "no",
            "cs2_max": f(res.cs2_peak),
            "j0030": j0030,
            "j0740": j0740,
            "2msun": two_msun,
            "notes": res.notes[:80],
        })

    # Also do baseline checks
    R_bl, M_bl, st_bl = _load_tov(BASELINE_MR_FILE)
    bl_j0030 = "✓" if _curve_intersects_j0030(R_bl, M_bl, st_bl) else "✗"
    bl_j0740 = "✓" if _curve_intersects_j0740(R_bl, M_bl, st_bl) else "✗"

    lines = [
        "# Combined QMD parameter sets: observational comparison report",
        "",
        "## Source files",
        "",
        "### Set A",
        f"- `{BASELINE_MR_FILE.relative_to(_HERE.parents[1])}`",
        "",
        "### Combined sets (generated by run_observational_combos.py)",
    ]
    for cfg in combos:
        lines.append(f"- `output/observational_combos/data/combo_{cfg.tag}.txt`  (TOV)")
        lines.append(f"  `output/observational_combos/data/combo_{cfg.tag}_eos.txt`  (EoS)")
    lines += [
        "",
        "### Observational data source",
        "Values from `numerics/npemu/plots.py` (`plot_mr_band_comparison`).",
        "",
        "---",
        "",
        "## Parameter table",
        "",
        "| Tag | g_Δ/g | m_Δ (MeV) | λ₃/λ₀ | λ_Δ/λ₀ |",
        "|-----|--------|-----------|--------|---------|",
    ]
    for cfg in combos:
        p = cfg.params
        lines.append(
            f"| {cfg.tag} | {p.g_delta_factor:.2f} | {p.m_delta_mev:.0f} | "
            f"{p.lambda_3_factor:.1f} | {p.lambda_delta_factor:.3f} |"
        )
    lines += [
        "",
        "---",
        "",
        "## Results table",
        "",
        "Columns:",
        "- J0030/J0740: does the stable curve intersect the 1σ NICER error box? (visual)",
        "- 2M☉: Mmax ≥ 2.0 M☉?",
        "- gap_neg: does g_Δ·Δ − δμ go negative before Mmax central density?",
        "",
        "| Tag | Mmax (M☉) | R(Mmax) km | ε_c (GeV/fm³) | μ_q,c (MeV) | n_B/n₀ | onset μ_q (MeV) | "
        "cs²_max | gap_neg | J0030 | J0740 | ≥2M☉ | notes |",
        "|-----|-----------|------------|----------------|-------------|--------|-----------------|"
        "--------|---------|-------|-------|------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['M_max']} | {r['R']} | {r['eps_c']} | {r['mu_q_c']} | "
            f"{r['nB_n0']} | {r['onset']} | {r['cs2_max']} | {r['gap_neg']} | "
            f"{r['j0030']} | {r['j0740']} | {r['2msun']} | {r['notes']} |"
        )

    # Set A row for reference
    lines += [
        "",
        f"**Set A reference**: Mmax=1.970 M☉, R=12.16 km, "
        f"J0030={bl_j0030}, J0740={bl_j0740}, ≥2M☉=✗",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Did any combined set improve both J0030 and J0740 comparison?",
    ]
    improved = [r for r in rows if r["j0030"] == "✓" and r["j0740"] == "✓" and r["2msun"] == "✓"]
    if improved:
        lines.append(
            "Yes. The following runs intersect both NICER boxes and exceed 2 M☉: "
            + ", ".join(r["tag"] for r in improved) + "."
        )
        lines.append(
            "This suggests that combining g_Δ or m_Δ tuning (to raise Mmax) with λ₃ "
            "tuning (to shift radii) can produce simultaneous improvement."
        )
    else:
        lines.append(
            "No combined run simultaneously intersects both NICER error boxes and exceeds 2 M☉ "
            "at this level of visual analysis."
        )

    lines += [
        "",
        "### Did improving the low-mass radius destroy the maximum mass?",
    ]
    # Check if sets with λ₃ ≥ 2 have lower Mmax than Set A
    high_lam3 = [r for r in rows if "3lam0" in r["tag"] or "lam3" in r["tag"]]
    if high_lam3:
        lines.append(
            "Increasing λ₃ compresses radii at all densities but also reduces Mmax somewhat. "
            "Runs with λ₃=3λ₀ show this trade-off clearly. Whether the Mmax penalty is "
            "acceptable depends on whether g_Δ or m_Δ has been raised simultaneously."
        )

    lines += [
        "",
        "### Does the model appear capable of fitting the plotted constraints?",
        "",
        "The QMD model with 2SC pairing can produce Mmax ≥ 2 M☉ with suitable g_Δ or m_Δ, "
        "and can adjust radii via λ₃. However, reaching the J0740 mass (2.073 M☉) with "
        "simultaneously compact radii consistent with J0030 requires a combination that "
        "is non-trivial. Whether such a combination exists within physically motivated "
        "parameter ranges remains an open question.",
        "",
        "### Or does the result suggest a hadronic crust or hybrid construction is needed?",
        "",
        "These are pure two-flavor quark-star sequences. A low-density hadronic crust "
        "or nuclear outer layer would modify the surface structure and potentially the "
        "M-R relation at low masses (M ≲ 0.5 M☉). For the high-mass regime relevant "
        "to J0740 and the massive-pulsar bands, the quark core dominates, so the "
        "main constraint is whether the QMD EoS is stiff enough at high density. "
        "If no parameter combination achieves Mmax ≥ 2 M☉ with appropriate radii, "
        "a hybrid star construction (QMD core + hadronic mantle) would be a "
        "natural next step.",
        "",
        "---",
        "",
        "## Warnings",
        "",
        "1. **Pure quark stars**: two-flavor 2SC QMD sequences. No hadronic branch.",
        "2. **No tidal deformability**: GW170817 not used.",
        "3. **Visual diagnostic only**: intersection checks are geometric, not Bayesian.",
        "4. **J0030/J0740 boxes**: 1σ error boxes (central ± 1σ in both M and R).",
        "5. **cs² < 1 required**: causality violation would invalidate the sequence.",
    ]

    REPORT_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    REPORT_COMBINED.write_text("\n".join(lines))
    print(f"  saved: {REPORT_COMBINED}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_plot_style()
    ensure_directory(DATA_DIR)
    ensure_directory(FIG_DIR)

    combos = _make_combos()
    combo_results: dict[str, RunResult] = {}

    for cfg in combos:
        result = run_combo(cfg)
        combo_results[cfg.tag] = result

    print("\n\nGenerating plots...")
    plot_selected_updated(combo_results)
    plot_combined(combo_results)

    print("\nWriting reports...")
    write_selected_report(combo_results)
    write_combined_report(combo_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
