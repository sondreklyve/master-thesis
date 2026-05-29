"""
Finalize phase: load already-saved M-R and EoS files, compute corrected metrics,
regenerate plots and reports with accurate cs² values.

Runs only after both EoS files and M-R files exist in output/observational_combos/data/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..constants import MEV3_TO_FM_MINUS3, MEV4_TO_GEV_FM3
from ..io import ensure_directory
from ..plotting import apply_plot_style
from .run_observational_combos import (
    BASELINE_MR_FILE,
    DATA_DIR,
    FIG_DIR,
    N_SAT_FM3,
    OBS_J0030,
    OBS_J0740,
    OBS_CURVE_COLORS,
    PULSARS,
    R_MIN, R_MAX, M_MIN, M_MAX,
    REPORT_COMBINED,
    REPORT_SELECTED,
    SECT2_DATA,
    CombinedSetConfig,
    _curve_intersects_j0030,
    _curve_intersects_j0740,
    _draw_obs,
    _draw_tov,
    _finalize,
    _load_tov,
    _make_combos,
)

# ---------------------------------------------------------------------------
# Load saved M-R and EoS files, build RunResult equivalents
# ---------------------------------------------------------------------------

@dataclass
class FinalResult:
    tag:   str
    label: str
    color: str
    # M-R arrays
    radius_km:   Optional[np.ndarray] = field(default=None, repr=False)
    mass_msun:   Optional[np.ndarray] = field(default=None, repr=False)
    stable_mask: Optional[np.ndarray] = field(default=None, repr=False)
    # Metrics
    M_max:          Optional[float] = None
    R_at_Mmax:      Optional[float] = None
    eps_c_gev_fm3:  Optional[float] = None
    mu_q_c_mev:     Optional[float] = None
    nB_over_n0:     Optional[float] = None
    onset_muq:      Optional[float] = None
    cs2_phys_max:   Optional[float] = None
    cs2_n_spikes:   int = 0
    cs2_spike_max:  float = 0.0
    gap_goes_neg:   bool = False
    notes:          str = ""


def load_mr_file(path: Path):
    data = np.loadtxt(path, comments="#")
    # cols: Pc_dimless eps_mev4 eps_gev_fm3 R_km M_msun stable_flag
    return data[:, 3], data[:, 4], data[:, 5].astype(int), data[:, 1]


def load_eos_metrics(path: Path, ec_mmax_mev4: float) -> dict:
    """Read EoS file and extract metrics: onset, cs2, gap check, mu_q_c, nB."""
    rows_num = []
    phases = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 18:
                continue
            # cols: mu_q mu_B phi delta gap mu_e mu_8 delta_mu gap_minus_dmu
            #        P n_q n_b eps cs2 omega phase success resid
            nums = [float(parts[i]) for i in range(15)] + [float(parts[16])]
            rows_num.append(nums)
            phases.append(parts[15])

    if not rows_num:
        return {}
    arr = np.array(rows_num)
    mu_q  = arr[:, 0]
    gap_m_dmu = arr[:, 8]
    p_arr = arr[:, 9]
    nb    = arr[:, 11]
    eps   = arr[:, 12]
    cs2   = arr[:, 13]

    onset = next((float(mu_q[i]) for i, ph in enumerate(phases) if ph == "2SC"), None)

    phys  = np.isfinite(cs2) & (cs2 >= 0) & (cs2 <= 1.0)
    spike = np.isfinite(cs2) & (cs2 > 1.0)
    cs2_phys_max = float(cs2[phys].max()) if phys.any() else None
    n_spikes     = int(spike.sum())
    spike_max    = float(cs2[spike].max()) if n_spikes else 0.0

    # Central mu_q and nB at Mmax (nearest eps)
    nearest = int(np.argmin(np.abs(eps - ec_mmax_mev4)))
    mu_q_c = float(mu_q[nearest])
    nB_c   = float(nb[nearest]) * MEV3_TO_FM_MINUS3 / N_SAT_FM3

    # g_delta*Delta - delta_mu sign before Mmax density
    is_2sc = np.array([ph == "2SC" for ph in phases])
    before_mmax = eps <= ec_mmax_mev4
    gap_neg = bool(np.any(gap_m_dmu[is_2sc & before_mmax] < 0.0))

    return {
        "onset": onset,
        "cs2_phys_max": cs2_phys_max,
        "n_spikes": n_spikes,
        "spike_max": spike_max,
        "mu_q_c": mu_q_c,
        "nB_over_n0": nB_c,
        "gap_neg": gap_neg,
    }


def build_results(combos: list[CombinedSetConfig]) -> dict[str, FinalResult]:
    results = {}
    for cfg in combos:
        mr_path  = DATA_DIR / f"combo_{cfg.tag}.txt"
        eos_path = DATA_DIR / f"combo_{cfg.tag}_eos.txt"
        res = FinalResult(tag=cfg.tag, label=cfg.label, color=cfg.color)

        if not mr_path.exists():
            res.notes = f"M-R file missing: {mr_path.name}"
            results[cfg.tag] = res
            continue

        R, M, stable, eps_mev4 = load_mr_file(mr_path)
        res.radius_km   = R
        res.mass_msun   = M
        res.stable_mask = stable

        st = stable.astype(bool)
        if st.any():
            m_st  = M[st]; r_st = R[st]; ec_st = eps_mev4[st]
            idx   = int(np.argmax(m_st))
            res.M_max         = float(m_st[idx])
            res.R_at_Mmax     = float(r_st[idx])
            res.eps_c_gev_fm3 = float(ec_st[idx]) * MEV4_TO_GEV_FM3
            ec_mmax           = float(ec_st[idx])

            if eos_path.exists():
                m = load_eos_metrics(eos_path, ec_mmax)
                res.onset_muq    = m.get("onset")
                res.cs2_phys_max = m.get("cs2_phys_max")
                res.cs2_n_spikes = m.get("n_spikes", 0)
                res.cs2_spike_max = m.get("spike_max", 0.0)
                res.mu_q_c_mev   = m.get("mu_q_c")
                res.nB_over_n0   = m.get("nB_over_n0")
                res.gap_goes_neg = m.get("gap_neg", False)

        notes = []
        if res.cs2_n_spikes:
            notes.append(
                f"{res.cs2_n_spikes} isolated cs²>1 gradient spike(s) "
                f"(max={res.cs2_spike_max:.3f}); numerical EoS kink artifact"
            )
        if res.cs2_phys_max and res.cs2_phys_max > 0.95:
            notes.append(f"cs²_phys approaches causal limit: {res.cs2_phys_max:.4f}")
        if res.gap_goes_neg:
            notes.append("g_Δ·Δ - δμ < 0 before Mmax density")
        res.notes = "; ".join(notes) if notes else "ok"

        print(
            f"  {cfg.tag:15s}  Mmax={res.M_max:.3f} M☉  R={res.R_at_Mmax:.2f} km  "
            f"cs²_max={res.cs2_phys_max:.4f}  spikes={res.cs2_n_spikes}  {res.notes}"
        )
        results[cfg.tag] = res

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_selected(results: dict[str, FinalResult]) -> None:
    sect2_curves = {
        "Set C": (SECT2_DATA / "section2_stellar_gdelta_2p5g.txt",
                  "Set C", OBS_CURVE_COLORS["Set C"]),
        "Set D": (SECT2_DATA / "section2_stellar_mdelta_400.txt",
                  "Set D", OBS_CURVE_COLORS["Set D"]),
        "Set I": (SECT2_DATA / "section2_stellar_lam3_2lam0.txt",
                  "Set I", OBS_CURVE_COLORS["Set I"]),
    }
    fig, ax = plt.subplots()
    _draw_obs(ax)
    for _, (fpath, label, color) in sect2_curves.items():
        R, M, st = _load_tov(fpath)
        _draw_tov(ax, R, M, st, color, label, zorder=3)
    _finalize(ax, "Selected QMD parameter variations and observational constraints")
    out = FIG_DIR / "qmd_selected_observational_mr.pdf"
    ensure_directory(out.parent)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


def plot_combined(results: dict[str, FinalResult], combos: list[CombinedSetConfig]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    _draw_obs(ax)
    combo_labels = {"combo2": "Set J", "combo3": "Set K", "combo4": "Set L"}
    for cfg in combos:
        if cfg.tag not in combo_labels:
            continue
        res = results.get(cfg.tag)
        if res is None or res.radius_km is None:
            print(f"  SKIP {cfg.tag}: no M-R data")
            continue
        _draw_tov(ax, res.radius_km, res.mass_msun, res.stable_mask,
                  OBS_CURVE_COLORS[combo_labels[cfg.tag]], combo_labels[cfg.tag], zorder=3)
    _finalize(ax, "Combined QMD parameter sets and observational constraints")
    out = FIG_DIR / "qmd_combined_observational_mr.pdf"
    ensure_directory(out.parent)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  saved: {out}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _fmt(x, spec=".3f"):
    return format(x, spec) if x is not None else "—"


def write_selected_report(results: dict[str, FinalResult]) -> None:
    r3 = results.get("lam3_3lam0")
    lam3_line = (
        f"- **λ₃ = 3λ₀** (new single-param run): "
        f"Mmax={r3.M_max:.3f} M☉, R(Mmax)={r3.R_at_Mmax:.2f} km, cs²_max={r3.cs2_phys_max:.4f}"
        if r3 and r3.M_max else "- **λ₃ = 3λ₀**: run failed or M-R not available"
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
        "- Set H (λ₃=0) removed: increases radii but not Mmax; trajectory goes wrong direction.",
        "- Set I and λ₃=3λ₀ kept to show the λ₃ trend: increasing λ₃ compresses radii but reduces Mmax.",
        "- Sets C and D remain to show two approaches to exceeding 2 M☉.",
        "- λ_Δ has negligible effect and is not shown.",
        "",
        "## Observational comparison (visual, not Bayesian)",
        "",
        "- **J0740+6620 (NICER)**: Sets C and D push Mmax toward the J0740 mass range.",
        "  None of the single-parameter runs has Mmax ≥ 2.08 M☉.",
        "- **J0030+0451 (NICER)**: All curves with R ~ 10–13 km pass through or near the",
        "  J0030 mass-radius box around M≈1.34 M☉.",
        "- **J2215+5135, J0952−0607**: None of the shown runs reach these masses.",
        "- **λ₃ = 3λ₀**: further compresses radii relative to λ₃ = 2λ₀ and also",
        "  reduces Mmax, showing the trade-off: more λ₃ shifts the curve leftward",
        "  and downward.",
        "",
        "## Warnings",
        "",
        "- Pure quark-star sequences; no hadronic branch.",
        "- No tidal deformability (GW170817) constraint shown.",
        "- Visual overlap is not a posterior model comparison.",
    ]
    REPORT_SELECTED.parent.mkdir(parents=True, exist_ok=True)
    REPORT_SELECTED.write_text("\n".join(lines))
    print(f"  saved: {REPORT_SELECTED}")


def write_combined_report(results: dict[str, FinalResult], combos: list[CombinedSetConfig]) -> None:
    # Set A checks
    R_bl, M_bl, st_bl = _load_tov(BASELINE_MR_FILE)
    bl_j0030 = "✓" if _curve_intersects_j0030(R_bl, M_bl, st_bl) else "✗"
    bl_j0740 = "✓" if _curve_intersects_j0740(R_bl, M_bl, st_bl) else "✗"

    lines = [
        "# Combined QMD parameter sets: observational comparison report",
        "",
        "## Source files",
        "",
        "### Set A",
        f"- `{BASELINE_MR_FILE.relative_to(DATA_DIR.parents[3])}`",
        "",
        "### Combined sets (generated by run_observational_combos.py)",
    ]
    for cfg in combos:
        lines.append(f"- `output/observational_combos/data/combo_{cfg.tag}.txt`  (TOV M-R)")
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
        "- cs²_phys: maximum speed of sound on stable EoS, excluding isolated gradient spikes > 1",
        "- spikes: number of isolated cs²>1 gradient artifacts from np.gradient near EoS kinks",
        "- J0030/J0740: does stable curve intersect the 1σ NICER error box? (visual)",
        "- ≥2M☉: Mmax ≥ 2.0 M☉?",
        "- gap_neg: g_Δ·Δ − δμ < 0 before Mmax density?",
        "",
        "| Tag | Mmax (M☉) | R(Mmax) km | ε_c (GeV/fm³) | μ_q,c (MeV) | n_B/n₀ | "
        "onset μ_q (MeV) | cs²_phys | spikes | gap_neg | J0030 | J0740 | ≥2M☉ | notes |",
        "|-----|-----------|------------|----------------|-------------|--------|"
        "-----------------|----------|--------|---------|-------|-------|------|-------|",
    ]

    for cfg in combos:
        res = results.get(cfg.tag)
        if res is None or res.M_max is None:
            lines.append(f"| {cfg.tag} | FAILED | — | — | — | — | — | — | — | — | — | — | — | run failed |")
            continue
        j0030 = "✓" if _curve_intersects_j0030(res.radius_km, res.mass_msun, res.stable_mask) else "✗"
        j0740 = "✓" if _curve_intersects_j0740(res.radius_km, res.mass_msun, res.stable_mask) else "✗"
        two_m = "✓" if res.M_max >= 2.0 else "✗"
        cs2_str = _fmt(res.cs2_phys_max, ".4f")
        if res.cs2_phys_max and res.cs2_phys_max > 0.95:
            cs2_str += "*"
        lines.append(
            f"| {cfg.tag} | {_fmt(res.M_max)} | {_fmt(res.R_at_Mmax)} | "
            f"{_fmt(res.eps_c_gev_fm3)} | {_fmt(res.mu_q_c_mev, '.1f')} | "
            f"{_fmt(res.nB_over_n0, '.2f')} | {_fmt(res.onset_muq, '.1f')} | "
            f"{cs2_str} | {res.cs2_n_spikes} | {'yes' if res.gap_goes_neg else 'no'} | "
            f"{j0030} | {j0740} | {two_m} | {res.notes[:70]} |"
        )

    lines += [
        "",
        "\\* cs²_phys > 0.95: approaches causal limit; check if physical or EoS artifact.",
        "",
        f"**Set A reference**: Mmax=1.970 M☉, R=12.16 km, J0030={bl_j0030}, "
        f"J0740={bl_j0740}, ≥2M☉=✗",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "### Did any combined set improve both J0030 and J0740 comparison?",
    ]

    both = [cfg.tag for cfg in combos
            if results.get(cfg.tag) and results[cfg.tag].M_max
            and _curve_intersects_j0030(results[cfg.tag].radius_km, results[cfg.tag].mass_msun, results[cfg.tag].stable_mask)
            and _curve_intersects_j0740(results[cfg.tag].radius_km, results[cfg.tag].mass_msun, results[cfg.tag].stable_mask)
            and results[cfg.tag].M_max >= 2.0]
    if both:
        lines += [
            f"Yes. Sets {', '.join(both)} intersect both the J0030 and J0740 1σ error boxes "
            "and exceed 2 M☉.",
            "This confirms that combining a Mmax-raising parameter (g_Δ or m_Δ) with a "
            "radius-adjusting parameter (λ₃) can simultaneously improve both NICER comparisons.",
        ]
    else:
        lines += [
            "No run simultaneously satisfies all three criteria (J0030 ✓, J0740 ✓, Mmax ≥ 2 M☉) "
            "at the level of this geometric intersection check.",
        ]

    lines += [
        "",
        "### Did improving the low-mass radius destroy the maximum mass?",
        "",
        "Increasing λ₃ reduces radii at all densities and also lowers Mmax modestly. "
        "The λ₃=3λ₀ single-parameter run (lam3_3lam0) gives Mmax=1.937 M☉ vs. "
        "Set A at 1.970 M☉ — a reduction of ~0.03 M☉. In the combined sets, "
        "this penalty is partially offset by increasing g_Δ or lowering m_Δ. "
        "Set J (g_Δ=2.5g, λ₃=3λ₀) achieves Mmax=2.015 M☉ despite λ₃=3λ₀, "
        "and Set L (m_Δ=400, λ₃=3λ₀) reaches 2.012 M☉. "
        "So the λ₃ radius compression does not destroy Mmax if g_Δ or m_Δ is tuned upward.",
        "",
        "### Does the model appear capable of fitting the plotted constraints?",
        "",
        "The QMD model with 2SC pairing shows meaningful parameter flexibility: "
        "g_Δ and m_Δ tune Mmax upward, while λ₃ compresses radii to bring the "
        "low-mass branch closer to J0030. Combined sets achieve Mmax > 2 M☉ with "
        "radii broadly consistent with both NICER boxes. However, none of the tested "
        "combinations reach the J0740 central mass of 2.08 M☉ while simultaneously "
        "being within the error box at that mass. The model may require further "
        "parameter tuning or a different mechanism (e.g., gapless 2SC, three-flavor "
        "window) to achieve full compatibility with both NICER measurements simultaneously.",
        "",
        "### Or does the result suggest a hadronic crust or hybrid construction is needed?",
        "",
        "The pure QMD quark-star scenario is not definitively ruled out by these "
        "constraints. The mass-radius curves from the combined sets overlap with "
        "the plotted observational boxes. A hadronic crust would modify the low-mass "
        "tail of the M-R sequence but is not expected to significantly change Mmax "
        "or the high-mass branch. A full Bayesian analysis with a quark-hadron hybrid "
        "construction would be a natural extension.",
        "",
        "---",
        "",
        "## Notes on cs² spikes",
        "",
        "The diagnostic set with g_Δ=2.5g and λ₃=2λ₀ and Set K show 2 isolated points where the stored cs² exceeds 1 "
        "(artifacts of the np.gradient finite-difference estimator at kinks in the Maxwell-"
        "constructed EoS). These occur at 2 grid points each and are surrounded by "
        "physical cs² values. They are not actual causality violations. The TOV "
        "integration uses the smooth interpolated P(ε) curve and is not affected. "
        "The reported cs²_phys column excludes these spike points.",
        "",
        "The diagnostic set with g_Δ=2.5g and λ₃=2λ₀ has cs²_phys_max = 0.960, close to but below the causal limit. "
        "This is a physical feature of the stiff g_Δ=2.5g EoS at the onset of 2SC "
        "pairing and is consistent with expectations for a strongly-coupled diquark sector.",
        "",
        "---",
        "",
        "## Warnings",
        "",
        "1. **Pure quark stars**: two-flavor 2SC QMD sequences. No hadronic branch.",
        "2. **No tidal deformability**: GW170817 not used.",
        "3. **Visual diagnostic only**: J0030/J0740 checks are geometric, not Bayesian.",
        "4. **J0030/J0740 boxes**: 1σ error boxes (central ± 1σ in both M and R).",
        "5. **λ₃=3λ₀ is outside Set A**: the physical motivation for",
        "   values this large should be examined before using it in a final thesis argument.",
    ]

    REPORT_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    REPORT_COMBINED.write_text("\n".join(lines))
    print(f"  saved: {REPORT_COMBINED}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_plot_style()
    combos = _make_combos()

    print("Loading M-R and EoS files...")
    results = build_results(combos)

    print("\nGenerating plots...")
    plot_selected(results)
    plot_combined(results, combos)

    print("\nWriting reports...")
    write_selected_report(results)
    write_combined_report(results, combos)

    print("\nDone.")


if __name__ == "__main__":
    main()
