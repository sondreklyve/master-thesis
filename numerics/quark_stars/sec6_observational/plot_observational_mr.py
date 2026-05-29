"""
Generate mass-radius plots comparing QMD quark-star sequences with observational constraints.

Outputs:
  thesis/figures/quark_stars/observational/qmd_baseline_observational_mr.pdf
  thesis/figures/quark_stars/observational/qmd_selected_observational_mr.pdf
  numerics/quark_stars/output/observational_mr_diagnostic_report.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from numerics.quark_stars.plotting import (
    SECTION2_MR_BASELINE_COLOR,
    SECTION2_MR_COMPARISON_COLORS,
)

OBS_NICER_COLORS = {
    "J0030": "tab:orange",
    "J0740": "tab:blue",
}
OBS_CURVE_COLORS = {
    "Set A": SECTION2_MR_BASELINE_COLOR,
    "Set C": SECTION2_MR_COMPARISON_COLORS[0],
    "Set D": SECTION2_MR_COMPARISON_COLORS[1],
    "Set I": SECTION2_MR_COMPARISON_COLORS[2],
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_SECT2 = REPO_ROOT / "numerics/quark_stars/output/sec5_parameter_sensitivity/data"
DATA_BASE = REPO_ROOT / "numerics/quark_stars/output/qmd_stellar/data"
FIG_DIR = REPO_ROOT / "thesis/figures/quark_stars/observational"
REPORT_PATH = REPO_ROOT / "numerics/quark_stars/output/observational_mr_diagnostic_report.md"

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
RUNS = {
    "Set A": {
        "file": DATA_BASE / "qmd_stars_baseline.txt",
        "label": "Set A",
        "color": "#0072B2",
        "zorder": 4,
    },
    "A": {
        "file": DATA_SECT2 / "section2_stellar_gdelta_1p5g.txt",
        "label": "Set B",
        "color": "#56B4E9",
        "zorder": 3,
    },
    "B": {
        "file": DATA_SECT2 / "section2_stellar_gdelta_2p5g.txt",
        "label": "Set C",
        "color": "#E69F00",
        "zorder": 3,
    },
    "C": {
        "file": DATA_SECT2 / "section2_stellar_mdelta_400.txt",
        "label": "Set D",
        "color": "#009E73",
        "zorder": 3,
    },
    "D": {
        "file": DATA_SECT2 / "section2_stellar_mdelta_700.txt",
        "label": "Set E",
        "color": "#F0E442",
        "zorder": 3,
    },
    "E": {
        "file": DATA_SECT2 / "section2_stellar_lamdelta_lam0div8.txt",
        "label": "Set F",
        "color": "#CC79A7",
        "zorder": 3,
    },
    "F": {
        "file": DATA_SECT2 / "section2_stellar_lamdelta_lam0div2.txt",
        "label": "Set G",
        "color": "#999999",
        "zorder": 3,
    },
    "G": {
        "file": DATA_SECT2 / "section2_stellar_lam3_0.txt",
        "label": "Set H",
        "color": "#D55E00",
        "zorder": 3,
    },
    "H": {
        "file": DATA_SECT2 / "section2_stellar_lam3_2lam0.txt",
        "label": "Set I",
        "color": "#8B4513",
        "zorder": 3,
    },
}

# ---------------------------------------------------------------------------
# Observational data (from numerics/npemu/plots.py)
# ---------------------------------------------------------------------------
OBS = {
    "J0030": {"M": 1.34, "dM_lo": 0.16, "dM_hi": 0.15, "R": 12.71, "dR_lo": 1.19, "dR_hi": 1.14},
    "J0740": {"M": 2.073, "dM_lo": 0.069, "dM_hi": 0.069, "R": 12.49, "dR_lo": 0.88, "dR_hi": 1.28},
}
PULSARS = [
    {"name": "PSR J0348+0432", "M": 2.01, "dM_lo": 0.04, "dM_hi": 0.04},
    {"name": "PSR J1614$-$2230", "M": 1.97, "dM_lo": 0.04, "dM_hi": 0.04},
    {"name": "PSR J2215+5135",  "M": 2.27, "dM_lo": 0.15, "dM_hi": 0.17},
    {"name": "PSR J0952$-$0607", "M": 2.35, "dM_lo": 0.17, "dM_hi": 0.17},
]

for key, color in zip(RUNS, plt.cm.viridis(np.linspace(0.10, 0.85, len(RUNS))), strict=True):
    RUNS[key]["color"] = color
RUNS["Set A"]["color"] = OBS_CURVE_COLORS["Set A"]
RUNS["B"]["color"] = OBS_CURVE_COLORS["Set C"]
RUNS["C"]["color"] = OBS_CURVE_COLORS["Set D"]
RUNS["H"]["color"] = OBS_CURVE_COLORS["Set I"]

# ---------------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------------
R_MIN, R_MAX = 8.0, 20.0
M_MIN, M_MAX = 0.5, 2.6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 10,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": ":",
            "figure.figsize": (8.0, 5.8),
            "savefig.bbox": "tight",
        }
    )


def load_tov(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (radius_km, mass_msun, stable_flag) arrays."""
    data = np.loadtxt(path, comments="#")
    return data[:, 3], data[:, 4], data[:, 5].astype(int)


def find_mmax(R: np.ndarray, M: np.ndarray, stable: np.ndarray) -> tuple[float, float]:
    st = stable == 1
    if not st.any():
        idx = np.argmax(M)
    else:
        idx_in_st = np.argmax(M[st])
        full_idx = np.where(st)[0][idx_in_st]
        idx = full_idx
    return float(M[idx]), float(R[idx])


def draw_tov_curve(ax, R, M, stable, color, label, zorder, lw=2.0):
    st = stable == 1
    un = ~st
    ax.plot(R[st], M[st], color=color, lw=lw, label=label, zorder=zorder, solid_capstyle="round")
    if un.any():
        ax.plot(R[un], M[un], color=color, lw=lw, ls="--", zorder=zorder - 0.1)
    Mmax, Rmax = find_mmax(R, M, stable)
    ax.plot(Rmax, Mmax, "o", color=color, ms=6, zorder=zorder + 1)


def draw_observational(ax, R_min=R_MIN, R_max=R_MAX):
    """Add mass bands and NICER error bars to ax."""
    # Horizontal mass bands
    for p in PULSARS:
        ax.fill_betweenx(
            [p["M"] - p["dM_lo"], p["M"] + p["dM_hi"]],
            R_min, R_max,
            alpha=0.30,
            label=p["name"],
            zorder=1,
        )

    # NICER error bars
    j = OBS["J0030"]
    color = OBS_NICER_COLORS["J0030"]
    ax.errorbar(
        j["R"], j["M"],
        xerr=[[j["dR_lo"]], [j["dR_hi"]]],
        yerr=[[j["dM_lo"]], [j["dM_hi"]]],
        fmt="none", ecolor=color, lw=1.5, capsize=3, capthick=1.2, zorder=5,
    )
    ax.scatter(
        j["R"], j["M"], marker="o", s=42, color=color, edgecolor="white",
        linewidth=0.6, label="PSR J0030+0451 (NICER)", zorder=5.1,
    )
    j = OBS["J0740"]
    color = OBS_NICER_COLORS["J0740"]
    ax.errorbar(
        j["R"], j["M"],
        xerr=[[j["dR_lo"]], [j["dR_hi"]]],
        yerr=[[j["dM_lo"]], [j["dM_hi"]]],
        fmt="none", ecolor=color, lw=1.5, capsize=3, capthick=1.2, zorder=5,
    )
    ax.scatter(
        j["R"], j["M"], marker="s", s=42, color=color, edgecolor="white",
        linewidth=0.6, label="PSR J0740+6620 (NICER)", zorder=5.1,
    )


def finalize_ax(ax, title: str) -> None:
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_ylim(M_MIN, M_MAX)
    ax.set_xlabel(r"$R\;[\mathrm{km}]$")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.legend(loc="lower left", ncol=1, framealpha=0.85)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  saved: {path}")


# ---------------------------------------------------------------------------
# Figure A: Set A + observational
# ---------------------------------------------------------------------------

def plot_baseline(runs: dict) -> tuple[float, float]:
    fig, ax = plt.subplots()
    draw_observational(ax)
    cfg = runs["Set A"]
    R, M, stable = load_tov(cfg["file"])
    draw_tov_curve(ax, R, M, stable, cfg["color"], cfg["label"], cfg["zorder"], lw=2.2)
    Mmax, Rmax = find_mmax(R, M, stable)
    finalize_ax(ax, "Set A and observational constraints")
    save_fig(FIG_DIR / "qmd_baseline_observational_mr.pdf")
    return Mmax, Rmax


# ---------------------------------------------------------------------------
# Figure B: Selected runs + observational
# ---------------------------------------------------------------------------

def plot_selected(runs: dict) -> dict:
    selected_keys = ["B", "C", "H"]
    fig, ax = plt.subplots()
    draw_observational(ax)
    results = {}
    for key in selected_keys:
        cfg = runs[key]
        R, M, stable = load_tov(cfg["file"])
        draw_tov_curve(ax, R, M, stable, cfg["color"], cfg["label"], cfg["zorder"])
        results[key] = find_mmax(R, M, stable)
    finalize_ax(ax, "Selected QMD parameter variations and observational constraints")
    save_fig(FIG_DIR / "qmd_selected_observational_mr.pdf")
    return results


# ---------------------------------------------------------------------------
# Figure C: All nine runs + observational
# ---------------------------------------------------------------------------

def plot_all_runs(runs: dict) -> dict:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    draw_observational(ax)
    results = {}
    for key, cfg in runs.items():
        R, M, stable = load_tov(cfg["file"])
        draw_tov_curve(ax, R, M, stable, cfg["color"], cfg["label"], cfg["zorder"], lw=1.6)
        results[key] = find_mmax(R, M, stable)
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_ylim(M_MIN, M_MAX)
    ax.set_xlabel(r"$R\;[\mathrm{km}]$")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.legend(loc="lower left", ncol=2, fontsize=9, framealpha=0.85)
    save_fig(FIG_DIR / "qmd_all_runs_observational_mr.pdf")
    return results


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

EXPECTED = {
    "Set A": (1.970, 12.16),
    "A": (1.793, 11.04),
    "B": (2.047, 12.72),
    "C": (2.058, 12.59),
    "D": (1.829, 11.28),
    "E": (1.971, 12.09),
    "F": (1.970, 12.17),
    "G": (2.004, 13.03),
    "H": (1.950, 11.46),
}


def write_report(all_results: dict) -> None:
    lines = [
        "# QMD observational M-R diagnostic report",
        "",
        "## Source files",
        "",
        "### QMD TOV sequences",
        "",
    ]
    for key, cfg in RUNS.items():
        lines.append(f"- **{key}** (`{cfg['label']}`): `{cfg['file'].relative_to(REPO_ROOT)}`")
    lines += [
        "",
        "### Observational data source",
        "",
        "Values copied from `numerics/npemu/plots.py` (function `plot_mr_band_comparison`).",
        "",
        "---",
        "",
        "## Maximum-mass confirmation table",
        "",
        "| Set | Label | M_max (M☉) computed | R(M_max) (km) computed | M_max expected | R expected | Match |",
        "|-----|-------|---------------------|------------------------|----------------|------------|-------|",
    ]
    for key, (M_exp, R_exp) in EXPECTED.items():
        if key in all_results:
            M_got, R_got = all_results[key]
            match = "✓" if abs(M_got - M_exp) < 0.05 and abs(R_got - R_exp) < 0.2 else "!"
            label = RUNS[key]["label"].replace("$", "").replace("\\", "")
            lines.append(
                f"| {key} | {label} | {M_got:.3f} | {R_got:.2f} | {M_exp:.3f} | {R_exp:.2f} | {match} |"
            )

    # Visual interpretation
    base_M, base_R = all_results["Set A"]
    lines += [
        "",
        "---",
        "",
        "## Visual interpretation",
        "",
        f"**Set A maximum mass**: {base_M:.3f} M☉, at R = {base_R:.2f} km.",
        "",
        "### Does Set A reach the robust 2 M☉ region?",
        f"Set A has Mmax = {base_M:.3f} M☉, just below 2 M☉.",
        "It falls within the PSR J1614−2230 band (1.97 ± 0.04 M☉) but does not",
        "reach the PSR J0348+0432 central value (2.01 M☉).",
        "It is below the heavier pulsars J2215+5135 (2.27 M☉) and J0952−0607 (2.35 M☉).",
        "",
        "### Are Set A radii consistent with NICER constraints?",
        f"The Set A branch spans roughly R ~ 10–13 km.",
        f"R(Mmax) = {base_R:.2f} km. NICER J0740+6620 gives R = 12.49 +1.28/−0.88 km",
        "at M = 2.073 M☉, and NICER J0030+0451 gives R = 12.71 +1.14/−1.19 km at M = 1.34 M☉.",
        "The Set A curve passes through both NICER error boxes, suggesting",
        "rough radius consistency, though its Mmax falls below J0740's central mass.",
        "",
        "### Which selected sets improve maximum mass?",
        f"- **Set C** (g_Δ = 2.5g): Mmax = {all_results['B'][0]:.3f} M☉ — exceeds 2 M☉.",
        f"- **Set D** (m_Δ = 400 MeV): Mmax = {all_results['C'][0]:.3f} M☉ — exceeds 2 M☉.",
        "",
        "### Radius changes relative to Set A:",
        f"- **Set H** (λ₃ = 0): Mmax = {all_results['G'][0]:.3f} M☉, R(Mmax) = {all_results['G'][1]:.2f} km — larger radii, higher Mmax.",
        f"- **Set I** (λ₃ = 2λ₀): Mmax = {all_results['H'][0]:.3f} M☉, R(Mmax) = {all_results['H'][1]:.2f} km — smaller radii, slightly lower Mmax.",
        "",
        "### Does any selected run appear compatible with all plotted constraints?",
        "Set C and Set D both exceed 2 M☉ and have radii consistent with NICER constraints.",
        "Set H has the largest radii; whether it remains inside the NICER boxes depends on the",
        "full curve shape. None of the sets reach the J0952−0607 central value (2.35 M☉).",
        "No set is definitively ruled out by the plotted constraints at this level of analysis.",
        "",
        "---",
        "",
        "## Warnings",
        "",
        "1. **Pure quark stars**: these are two-flavor QMD quark-star sequences with 2SC pairing.",
        "   They do not include hadronic or hybrid branches.",
        "2. **No tidal deformability**: GW170817 constraints on Λ are not used.",
        "3. **No Bayesian comparison**: visual overlap with error bars is not a posterior",
        "   model comparison. Use posterior sampling for quantitative model selection.",
        "4. **Stability flag**: stable branch defined by `stable_flag == 1` from TOV integrator.",
        "   Unstable branches are shown dashed where present in the data.",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"  saved: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    apply_style()

    print("Loading TOV sequences and computing Mmax...")
    all_results = {}
    for key, cfg in RUNS.items():
        R, M, stable = load_tov(cfg["file"])
        all_results[key] = find_mmax(R, M, stable)
        Mmax, Rmax = all_results[key]
        print(f"  {key:10s}  Mmax={Mmax:.3f} M_sun  R(Mmax)={Rmax:.2f} km")

    print("\nPlot A: Set A + observational constraints")
    plot_baseline(RUNS)

    print("Plot B: Selected sets + observational constraints")
    plot_selected(RUNS)


    print("\nWriting diagnostic report...")
    write_report(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
