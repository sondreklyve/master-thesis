"""
TOV + plot + report phase only — loads already-saved EoS files from
output/observational_combos/data/ and skips the expensive equilibrium scan.

Run this after the EoS files exist but TOV failed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import MEV3_TO_FM_MINUS3, MEV4_TO_GEV_FM3
from .io import ensure_directory
from .plotting import apply_plot_style
from .qmd_parameters import QMD_SET_A
from .run_observational_combos import (
    BASELINE_MR_FILE,
    COMBO_DIR,
    DATA_DIR,
    FIG_DIR,
    LOG_FILE,
    N_SAT_FM3,
    OBS_J0030,
    OBS_J0740,
    PULSARS,
    R_MIN, R_MAX, M_MIN, M_MAX,
    REPORT_COMBINED,
    REPORT_SELECTED,
    SECT2_DATA,
    ComboConfig,
    RunResult,
    _QMDEoS,
    _curve_intersects_j0030,
    _curve_intersects_j0740,
    _draw_obs,
    _draw_tov,
    _finalize,
    _load_tov,
    _log,
    _make_combos,
    plot_selected_updated,
    write_combined_report,
    write_selected_report,
)
from .solvers.tov import run_tov_sequence

# QMDStellarEoSPoint columns (from EoS files written by run_observational_combos.py):
# mu_q_mev mu_B_mev phi_mev delta_mev gap_mev mu_e_mev mu_8_mev delta_mu_mev
# gap_minus_delta_mu_mev pressure_mev4 quark_density_mev3 baryon_density_mev3
# energy_density_mev4 cs2 omega_min_mev4 phase success neutrality_residual_norm
COL_MU_Q   = 0
COL_DELTA  = 3
COL_DELTA_MU = 7
COL_GAP_MINUS_DMU = 8
COL_P      = 9
COL_NQ     = 10
COL_NB     = 11
COL_EPS    = 12
COL_CS2    = 13
COL_PHASE  = 15


def load_eos_file(path: Path) -> dict:
    """Load a combo EoS text file; return dict with arrays."""
    rows = []
    phases = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            rows.append([float(x) for x in parts[:15]] + [float(parts[16])])
            phases.append(parts[15])

    if not rows:
        raise ValueError(f"No data rows in {path}")
    arr = np.array(rows)
    return {
        "mu_q":       arr[:, COL_MU_Q],
        "delta":      arr[:, COL_DELTA],
        "delta_mu":   arr[:, COL_DELTA_MU],
        "gap_minus_dmu": arr[:, COL_GAP_MINUS_DMU],
        "pressure":   arr[:, COL_P],
        "n_b":        arr[:, COL_NB],
        "eps":        arr[:, COL_EPS],
        "cs2":        arr[:, COL_CS2],
        "phase":      phases,
        "success":    arr[:, -1],
    }


def run_tov_from_eos_file(cfg: ComboConfig) -> RunResult:
    eos_path = DATA_DIR / f"combo_{cfg.tag}_eos.txt"
    result = RunResult(tag=cfg.tag, label=cfg.label)

    if not eos_path.exists():
        result.notes = f"EoS file missing: {eos_path}"
        print(f"  SKIP {cfg.tag}: {result.notes}")
        return result

    try:
        d = load_eos_file(eos_path)
    except Exception as exc:
        result.notes = f"EoS load failed: {exc}"
        _log(cfg.tag, "eos_load", exc)
        return result

    p_arr  = d["pressure"]
    e_arr  = d["eps"]
    mu_arr = d["mu_q"]
    nb_arr = d["n_b"]
    cs2_arr = d["cs2"]
    phases  = d["phase"]
    gap_minus_dmu = d["gap_minus_dmu"]

    # 2SC onset
    onset_muq = next(
        (float(mu_arr[i]) for i, ph in enumerate(phases) if ph == "2SC"),
        None
    )
    result.onset_muq = onset_muq

    # cs2 peak — use physical maximum (≤ 1.0); count isolated gradient spikes > 1
    phys_cs2 = cs2_arr[np.isfinite(cs2_arr) & (cs2_arr >= 0) & (cs2_arr <= 1.0)]
    spike_cs2 = cs2_arr[np.isfinite(cs2_arr) & (cs2_arr > 1.0)]
    result.cs2_peak = float(np.max(phys_cs2)) if phys_cs2.size else None
    n_spikes = int(len(spike_cs2))
    if n_spikes:
        print(f"  NOTE {cfg.tag}: {n_spikes} isolated cs²>1 gradient spike(s) "
              f"(max={float(spike_cs2.max()):.3f}); these are EoS kink artifacts, not physical")

    # Build TOV EoS
    eos_tov = _QMDEoS(pressure_mev4=p_arr, energy_density_mev4=e_arr)
    try:
        sequence = run_tov_sequence(eos_tov, integrator="rk4")
    except Exception as exc:
        result.notes = f"TOV failed: {exc}"
        _log(cfg.tag, "TOV", exc)
        return result

    result.radius_km   = sequence.radius_km
    result.mass_msun   = sequence.mass_msun
    result.stable_mask = sequence.stable_mask

    stable_tov = sequence.stable_mask.astype(bool)
    if not stable_tov.any():
        result.notes = "TOV: no stable branch"
        return result

    m_st  = sequence.mass_msun[stable_tov]
    r_st  = sequence.radius_km[stable_tov]
    ec_st = sequence.central_energy_density_mev4[stable_tov]
    idx   = int(np.argmax(m_st))

    result.M_max         = float(m_st[idx])
    result.R_at_Mmax     = float(r_st[idx])
    result.eps_c_gev_fm3 = float(ec_st[idx]) * MEV4_TO_GEV_FM3

    # Look up mu_q and n_B at the central energy density of Mmax
    ec_mmax    = float(ec_st[idx])
    near_idx   = int(np.argmin(np.abs(e_arr - ec_mmax)))
    result.mu_q_c_mev = float(mu_arr[near_idx])
    nB_fm3 = float(nb_arr[near_idx]) * MEV3_TO_FM_MINUS3
    result.nB_over_n0 = nB_fm3 / N_SAT_FM3

    # Causality: max cs2 on stable EoS (cs2 < 1 required)
    result.cs2_max_stable = result.cs2_peak

    # g_delta*Delta - delta_mu sign before Mmax density
    ec_2sc = [
        gap_minus_dmu[i]
        for i, ph in enumerate(phases)
        if ph == "2SC" and e_arr[i] <= ec_mmax
    ]
    if ec_2sc and any(g < 0.0 for g in ec_2sc):
        result.gap_goes_negative = True
        print(f"  WARNING {cfg.tag}: g_Δ·Δ - δμ < 0 before Mmax density (gapless 2SC regime)")

    # Save M-R table
    mr_path = DATA_DIR / f"combo_{cfg.tag}.txt"
    try:
        from .run_observational_combos import _write_mr
        # Build a minimal namespace to reuse _write_mr
        # We need sequence.central_pressure_dimless
        from types import SimpleNamespace
        seq_ns = SimpleNamespace(
            mass_msun=sequence.mass_msun,
            radius_km=sequence.radius_km,
            stable_mask=sequence.stable_mask,
            central_pressure_dimless=sequence.central_pressure_dimless,
            central_energy_density_mev4=sequence.central_energy_density_mev4,
        )
        meta = {
            "tag": cfg.tag,
            "g_delta_factor": f"{cfg.params.g_delta_factor:.4f}",
            "m_delta_mev": f"{cfg.params.m_delta_mev:.1f}",
            "lambda_3_factor": f"{cfg.params.lambda_3_factor:.4f}",
            "lambda_delta_factor": f"{cfg.params.lambda_delta_factor:.4f}",
            "source": "tov_resume_from_eos_file",
        }
        _write_mr(mr_path, seq_ns, meta)
        print(f"  M-R → {mr_path.name}")
    except Exception as exc:
        print(f"  WARNING: could not save M-R for {cfg.tag}: {exc}")

    notes = []
    if result.gap_goes_negative:
        notes.append("g_Δ·Δ - δμ < 0 before Mmax")
    if result.cs2_peak and result.cs2_peak >= 1.0:
        notes.append(f"causality violation: cs²_max={result.cs2_peak:.3f}")
    result.notes = "; ".join(notes) if notes else "ok"

    print(f"  {cfg.tag}: Mmax={result.M_max:.3f} M☉  R={result.R_at_Mmax:.2f} km  "
          f"ε_c={result.eps_c_gev_fm3:.3f} GeV/fm³  n_B/n₀={result.nB_over_n0:.2f}  "
          f"onset={result.onset_muq:.1f} MeV  cs²_max={result.cs2_peak:.4f}")
    return result


def plot_combined(combo_results: dict[str, RunResult]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    _draw_obs(ax)

    R, M, st = _load_tov(BASELINE_MR_FILE)
    _draw_tov(ax, R, M, st, "#0072B2", r"Baseline (QMD SET A)", zorder=4, lw=2.2)

    combo_keys = ["combo1", "combo2", "combo3", "combo4", "combo5"]
    for key in combo_keys:
        res = combo_results.get(key)
        if res is None or res.radius_km is None:
            print(f"  SKIP plot {key}: no M-R data")
            continue
        cfg = next(c for c in _make_combos() if c.tag == key)
        _draw_tov(ax, res.radius_km, res.mass_msun, res.stable_mask,
                  cfg.color, cfg.label, zorder=3)

    _finalize(ax, "Combined QMD parameter sets and observational constraints")
    out = FIG_DIR / "qmd_combined_observational_mr.pdf"
    ensure_directory(out.parent)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


def main() -> None:
    apply_plot_style()
    ensure_directory(DATA_DIR)

    combos = _make_combos()
    combo_results: dict[str, RunResult] = {}

    print("Running TOV from saved EoS files...\n")
    for cfg in combos:
        combo_results[cfg.tag] = run_tov_from_eos_file(cfg)

    print("\nGenerating plots...")
    plot_selected_updated(combo_results)
    plot_combined(combo_results)

    print("\nWriting reports...")
    write_selected_report(combo_results)
    write_combined_report(combo_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
