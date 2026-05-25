"""Schematic Maxwell-construction figure (two panels, fabricated data)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# ---------------------------------------------------------------------------
# Style – matches the rest of the thesis (see plotting.py)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 15,
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "savefig.bbox": "tight",
    }
)

# ---------------------------------------------------------------------------
# Fabricated analytic model
#
# P(μ) = (μ - μ₀)³ - β(μ - μ₀) + C
# dP/dμ = 3(μ - μ₀)² - β
#
# Turning points:  μ_tp = μ₀ ± √(β/3)
# Maxwell endpoints (equal-area; by symmetry):  μ_{A,B} = μ₀ ∓ √β
# Transition pressure by symmetry:  P_t = C
# ---------------------------------------------------------------------------
MU0 = 1.0
BETA = 0.6          # controls the S-shape amplitude and width
C    = 1.5          # transition pressure P_t by symmetry

MU_A = MU0 - np.sqrt(BETA)          # ≈ 0.225 – transition start
MU_B = MU0 + np.sqrt(BETA)          # ≈ 1.775 – transition end
MU_TP_L = MU0 - np.sqrt(BETA / 3)   # ≈ 0.553 – local P maximum
MU_TP_R = MU0 + np.sqrt(BETA / 3)   # ≈ 1.447 – local P minimum
P_T = C


def P_raw(mu: np.ndarray) -> np.ndarray:
    return (mu - MU0) ** 3 - BETA * (mu - MU0) + C


def n_raw(mu: np.ndarray) -> np.ndarray:
    """n = dP/dμ"""
    return 3.0 * (mu - MU0) ** 2 - BETA


def eps_raw(mu: np.ndarray) -> np.ndarray:
    """ε = μ n - P  (thermodynamic identity at T = 0)"""
    return mu * n_raw(mu) - P_raw(mu)


# ---------------------------------------------------------------------------
# Sample μ array – dense for smooth curves
# ---------------------------------------------------------------------------
mu = np.linspace(0.05, 1.95, 2000)

P_r   = P_raw(mu)
eps_r = eps_raw(mu)

# Maxwell-constructed P(μ): flat at P_t between μ_A and μ_B
P_mx = np.where((mu >= MU_A) & (mu <= MU_B), P_T, P_r)

# Maxwell-constructed ε(P): vertical jump at P_t
# Stable portions: μ < μ_A  and  μ > μ_B
mask_left  = mu <= MU_A
mask_right = mu >= MU_B

eps_A = float(eps_raw(np.array([MU_A]))[0])   # ε at transition start
eps_B = float(eps_raw(np.array([MU_B]))[0])   # ε at transition end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
COLOR_MX  = "#1f4e79"
COLOR_RAW = "#888888"
COLOR_DOT = "#333333"
LW_RAW = 1.8
LW_MX  = 2.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0))

# ── Left panel: P(μ_q) ──────────────────────────────────────────────────────
ax1.plot(mu, P_r,  "--", color=COLOR_RAW, lw=LW_RAW, label=r"Raw",                   zorder=2)
ax1.plot(mu, P_mx, "-",  color=COLOR_MX,  lw=LW_MX,  label=r"Maxwell construction",  zorder=3)

# Transition pressure reference line
ax1.axhline(P_T, color=COLOR_DOT, lw=0.9, ls=":", zorder=1)

# Unstable-region annotation with arrow pointing to mid-S
mu_ann = MU0
P_ann  = P_raw(np.array([mu_ann]))[0]
ax1.annotate(
    r"$\dfrac{\mathrm{d}P}{\mathrm{d}\mu_q} < 0$",
    xy=(mu_ann, P_ann),
    xytext=(mu_ann - 0.62, P_ann - 0.24),
    fontsize=16,
    color=COLOR_RAW,
    arrowprops=dict(arrowstyle="->", color=COLOR_RAW, lw=0.9),
)

ax1.set_xlabel(r"Chemical potential $\mu_q$")
ax1.set_ylabel(r"Pressure $P$")

# Suppress all ticks; place P_t label inside the plot
ax1.set_xticks([])
ax1.set_yticks([])

# P_t label: x in axes coords (left edge), y in data coords (at P_T)
trans1 = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)
ax1.text(0.025, P_T, r"$P_t$", ha="left", va="bottom", fontsize=16,
         color=COLOR_DOT, transform=trans1)

ax1.legend(loc="upper left", framealpha=0.85)

# ── Right panel: ε(P) ───────────────────────────────────────────────────────
# Raw parametric curve
ax2.plot(P_r, eps_r, "--", color=COLOR_RAW, lw=LW_RAW, label=r"Raw", zorder=2)

# Maxwell-constructed curve: two stable branches + vertical jump
ax2.plot(
    P_r[mask_left],
    eps_r[mask_left],
    "-", color=COLOR_MX, lw=LW_MX, label=r"Maxwell construction", zorder=3,
)
ax2.plot(
    [P_T, P_T],
    [eps_A, eps_B],
    "-", color=COLOR_MX, lw=LW_MX, zorder=3,
)
ax2.plot(
    P_r[mask_right],
    eps_r[mask_right],
    "-", color=COLOR_MX, lw=LW_MX, zorder=3,
)

# Transition pressure reference line
ax2.axvline(P_T, color=COLOR_DOT, lw=0.9, ls=":", zorder=1)

ax2.set_xlabel(r"Pressure $P$")
ax2.set_ylabel(r"Energy density $\varepsilon$")

# Suppress all ticks; place P_t label inside the plot
ax2.set_xticks([])
ax2.set_yticks([])

# P_t label: x in data coords (at P_T), y in axes coords (near top)
trans2 = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
ax2.text(P_T - 0.01, 0.78, r"$P_t$", ha="right", va="top", fontsize=16,
         color=COLOR_DOT, transform=trans2)

ax2.legend(loc="upper left", framealpha=0.85)

# ── Save ────────────────────────────────────────────────────────────────────
fig.tight_layout()

outpath = (
    Path(__file__).resolve().parent.parent.parent
    / "thesis"
    / "figures"
    / "quark_stars"
    / "maxwell_construction_schematic.pdf"
)
outpath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outpath)
plt.close(fig)

print(f"Saved: {outpath}")
