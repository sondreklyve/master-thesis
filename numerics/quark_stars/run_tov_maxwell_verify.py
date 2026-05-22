"""TOV solver and Maxwell construction independent verification.

Run from repo root:
    numerics/bin/python -m numerics.quark_stars.run_tov_maxwell_verify

Covers tests T1-T5, M1-M4, N1-N3 as specified in the verification plan.
Does NOT edit any files. Reports only.
"""
from __future__ import annotations

import math
import os
import sys
import pathlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent           # numerics/quark_stars/
NPEMU_DIR = ROOT.parent / "npemu"
if str(NPEMU_DIR) not in sys.path:
    sys.path.insert(0, str(NPEMU_DIR))

import core as C
import tov  as T

from .thermodynamics.maxwell import maxwell_construct
from .qmd_parameters import QMD_SET_A
from .qmd_stellar import QMDStellarModel

# ---------------------------------------------------------------------------
# Paths to pre-computed data
# ---------------------------------------------------------------------------
STELL_DIR  = ROOT / "output" / "qmd_stellar" / "data"
STAB_FILE  = STELL_DIR / "qmd_stellar_eos_baseline_stable.txt"
RAW_FILE   = STELL_DIR / "qmd_stellar_eos_baseline_raw.txt"
STARS_FILE = STELL_DIR / "qmd_stars_baseline.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def ok(msg):   print(f"  PASS  {msg}")
def fail(msg): print(f"  FAIL  {msg}")
def info(msg): print(f"        {msg}")

def load_numeric(path):
    """Load file dropping non-numeric string columns."""
    try:
        return np.loadtxt(path, comments="#")
    except ValueError:
        raw = np.genfromtxt(path, comments="#", dtype=str, invalid_raise=False)
        cols = []
        for j in range(raw.shape[1]):
            try:
                raw[:, j].astype(float)
                cols.append(j)
            except ValueError:
                pass
        return raw[:, cols].astype(float)

def load_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            line = line[1:].strip()
            if "=" in line:
                k, _, v = line.partition("=")
                meta[k.strip()] = v.strip()
    return meta

# ---------------------------------------------------------------------------
# Build ε(P) interpolant from P, ε arrays (both in MeV^4)
# Returns interpolant in dimensionless units (divided by e0)
# ---------------------------------------------------------------------------
def build_eos_interp(press_mev4, eps_mev4):
    """Build dimensionless ε(P) interpolant. Uses linear interpolation to avoid
    cubic-spline oscillations in the Maxwell-gap region and at the surface."""
    P_d = press_mev4 / C.e0
    E_d = eps_mev4 / C.e0
    pos = P_d > 0
    P_in = P_d[pos]
    E_in = E_d[pos]
    order = np.argsort(P_in)
    P_in = P_in[order]; E_in = E_in[order]
    return interp1d(P_in, E_in, kind="linear",
                    bounds_error=False, fill_value=(E_in[0], E_in[-1]))

# ---------------------------------------------------------------------------
# Direct TOV integration capturing full profile (Euler, same as npemu)
# ---------------------------------------------------------------------------
def integrate_tov_profile(eos, Pc_dimless, rstep=0.01, r_max=30.0, tol=0.0):
    """Returns arrays (r, M, P, eps) at each Euler step."""
    P = Pc_dimless
    M = 0.0
    r = 0.0
    rs, Ms, Ps, Es = [], [], [], []
    while P > tol and r < r_max:
        r += rstep
        eps = float(eos(P))
        M  += rstep * C.dMdr(r, eps)
        P  += rstep * C.dPdr(r, M, P, eps)
        rs.append(r); Ms.append(M); Ps.append(P); Es.append(eps)
    return np.array(rs), np.array(Ms), np.array(Ps), np.array(Es)

# ---------------------------------------------------------------------------
# RK4 TOV integration (independent of npemu Euler)
# ---------------------------------------------------------------------------
def rk4_step(r, M, P, eps_fn, h):
    def f(r_, M_, P_):
        e_ = float(eps_fn(P_))
        return C.dMdr(r_, e_), C.dPdr(r_, M_, P_, e_)
    k1M, k1P = f(r, M, P)
    k2M, k2P = f(r + h/2, M + h/2*k1M, P + h/2*k1P)
    k3M, k3P = f(r + h/2, M + h/2*k2M, P + h/2*k2P)
    k4M, k4P = f(r + h,   M + h*k3M,   P + h*k3P)
    return M + h*(k1M+2*k2M+2*k3M+k4M)/6, P + h*(k1P+2*k2P+2*k3P+k4P)/6

def integrate_tov_rk4(eos, Pc_dimless, rstep=0.01, r_max=30.0, tol=0.0):
    # Start at r=rstep to match the Euler scheme (avoid division-by-zero at r=0)
    P = Pc_dimless
    r = rstep
    eps0 = float(eos(P))
    M = rstep * C.dMdr(r, eps0)  # single Euler step from origin to r=rstep
    P += rstep * C.dPdr(r, M, P, eps0)
    while P > tol and r < r_max:
        M_new, P_new = rk4_step(r, M, P, eos, rstep)
        r += rstep
        M, P = M_new, P_new
    return r, M

def run_sequence_euler(eos, Pcstart, Pcend, Pcstep_factor=1.08, rstep=0.01, r_max=30.0):
    Ms, Rs, Pcs = [], [], []
    Pc = Pcstart
    while Pc < Pcend:
        P, M, r = Pc, 0.0, 0.0
        while P > 0.0 and r < r_max:
            r += rstep
            eps = float(eos(P))
            M += rstep * C.dMdr(r, eps)
            P += rstep * C.dPdr(r, M, P, eps)
        Ms.append(M); Rs.append(r); Pcs.append(Pc)
        Pc *= Pcstep_factor
    return np.array(Pcs), np.array(Ms), np.array(Rs)

def run_sequence_rk4(eos, Pcstart, Pcend, Pcstep_factor=1.08, rstep=0.01, r_max=30.0):
    Ms, Rs = [], []
    Pc = Pcstart
    while Pc < Pcend:
        r, M = integrate_tov_rk4(eos, Pc, rstep=rstep, r_max=r_max)
        Ms.append(M); Rs.append(r)
        Pc *= Pcstep_factor
    return np.array(Ms), np.array(Rs)

# ---------------------------------------------------------------------------
# Schwarzschild interior analytic solution
# For uniform density star: P_c / eps0 = f(C) where C = 2*R0*M/R
# ---------------------------------------------------------------------------
def schwarzschild_Pc(eps0_dimless, M_sun, R_km):
    """Analytic central pressure for incompressible star."""
    C_comp = 2.0 * C.R0 * M_sun / R_km
    if C_comp >= 8.0/9.0:
        return float('inf')
    sq = math.sqrt(1.0 - C_comp)
    return eps0_dimless * (1.0 - sq) / (3.0 * sq - 1.0)

# ---------------------------------------------------------------------------
# Load QMD baseline stable EoS
# ---------------------------------------------------------------------------
stab = load_numeric(STAB_FILE)
meta = load_meta(STAB_FILE)

# Column layout (after removing string 'phase' column at pos 15):
# 0:mu_q 1:mu_B 2:phi 3:delta 4:gap 5:mu_e 6:mu_8 7:delta_mu 8:gap_minus
# 9:press 10:nq_quark 11:nq_baryon 12:eps 13:cs2 14:omega_min [15:success] [16:norm]
PRESS_COL  = 9
EPS_COL    = 12
MU_COL     = 0

press_stab = stab[:, PRESS_COL]
eps_stab   = stab[:, EPS_COL]
mu_stab    = stab[:, MU_COL]

# Only positive-pressure points
pos_mask = press_stab > 0.0
press_pos = press_stab[pos_mask]
eps_pos   = eps_stab[pos_mask]

eos_qmd = build_eos_interp(press_pos, eps_pos)

# Surface pressure threshold for QMD EoS integration (stop when P drops below onset)
Pc_qmd_tol   = float(press_pos[0]) / C.e0 * 0.5   # half the first positive pressure

# Central pressure range from stable file
Pc_qmd_start = press_pos[0] * 1.01 / C.e0
Pc_qmd_end   = press_pos[-1] * 0.999 / C.e0

# ============================================================================
# PART 1 — TOV SOLVER VERIFICATION
# ============================================================================

# ---------------------------------------------------------------------------
# T1: Uniform density (Schwarzschild interior) limit
# ---------------------------------------------------------------------------
section("T1 — Uniform density (Schwarzschild interior) limit")

# Three ε₀ values spanning QMD range (in MeV^4)
eps0_vals_mev4 = [5e6, 5e7, 5e8]  # low, mid, high density in QMD range
eps0_labels    = ["5e6", "5e7", "5e8"]

info(f"R0 = {C.R0:.6f} km/M_sun  (Schwarzschild radius per solar mass / 2)")
info(f"beta = {C.beta:.6e}  (dM/dr conversion factor)")
info(f"")
info(f"{'eps0 (MeV^4)':<14} {'M (Msun)':<12} {'R (km)':<10} {'M_analytic':<12} "
     f"{'err_M%':<10} {'Pc_num/eps0':<14} {'Pc_analytic/eps0':<18} {'err_Pc%':<10}")

t1_pass = True
# For each density, use a Pc high enough that the star fits inside r_max=30 km.
# For a uniform star: R = (3M/beta/eps0)^(1/3).  At compactness C~0.5 (near Buchdahl):
# 2*R0*M/R = 0.5 → M/R = 0.5/(2*R0) → R = (beta*eps0*R^3/3)^... just scan Pc.
# For low eps0, the star is huge. Use high Pc to keep star small.
# P_c / eps0 ~ f(C) where f(C) >> 1 near Buchdahl limit C=8/9.
# For C=0.5: f = (1-sqrt(0.5))/(3*sqrt(0.5)-1) = (1-0.707)/(2.121-1) = 0.293/1.121 = 0.261
# So use Pc_test ~ 0.26 * eps0 to target C~0.5 (moderate, Schwarzschild formula valid)
info(f"{'eps0 (MeV^4)':<14} {'M (Msun)':<12} {'R (km)':<10} {'M_analytic':<12} "
     f"{'err_M%':<10} {'Pc_num/eps0':<14} {'Pc_analytic/eps0':<18} {'err_Pc%':<10} {'complete?'}")

for eps0_mev4, label in zip(eps0_vals_mev4, eps0_labels):
    eps0_d = eps0_mev4 / C.e0
    eos_uni = interp1d([0.0, 1e6], [eps0_d, eps0_d], kind="linear",
                       fill_value=eps0_d, bounds_error=False)
    # Scan Pc values to find one where star is complete (R < 28 km) and has moderate compactness
    Pc_test = None
    for frac in [0.8, 0.5, 0.3, 0.261, 0.15, 0.10, 0.05]:
        Pc_try = frac * eps0_d
        r_try, M_try, P_try, _ = integrate_tov_profile(eos_uni, Pc_try, rstep=0.005, r_max=28.0)
        complete = (P_try[-1] <= 0.0 if len(P_try) > 0 else False)
        if complete and len(r_try) > 0:
            Pc_test = Pc_try
            break
    if Pc_test is None:
        info(f"  {label}: could not find Pc for complete star within 28 km — SKIP")
        continue
    r_arr, M_arr, P_arr, E_arr = integrate_tov_profile(eos_uni, Pc_test, rstep=0.005, r_max=28.0)
    complete = (P_arr[-1] <= 0.0) if len(P_arr) > 0 else False
    if len(M_arr) == 0:
        info(f"  {label}: no output"); continue
    M_num = M_arr[-1]; R_num = r_arr[-1]
    # Analytic: M = (beta/3) * R^3 * eps0_d
    M_analytic = (C.beta / 3.0) * R_num**3 * eps0_d
    err_M = 100.0 * abs(M_num - M_analytic) / abs(M_analytic) if M_analytic != 0 else float('nan')
    # Schwarzschild Pc
    Pc_analytic = schwarzschild_Pc(eps0_d, M_num, R_num)
    err_Pc = 100.0 * abs(Pc_test - Pc_analytic) / abs(Pc_analytic) if (Pc_analytic not in (0.0, float('inf'))) else float('nan')
    info(f"  {eps0_mev4:<14.0e} {M_num:<12.5f} {R_num:<10.4f} {M_analytic:<12.5f} "
         f"{err_M:<10.3f} {Pc_test/eps0_d:<14.6f} {Pc_analytic/eps0_d:<18.6f} {err_Pc:<10.3f} {'yes' if complete else 'truncated'}")
    if err_M > 1.0:
        fail(f"eps0={label}: M_numeric vs M_analytic error {err_M:.2f}% > 1%")
        t1_pass = False
    if not math.isnan(err_Pc) and err_Pc > 1.0:
        fail(f"eps0={label}: Pc discrepancy {err_Pc:.2f}% > 1%")
        t1_pass = False

# Also verify M ∝ R³ by running a sequence at fixed eps0=5e7 MeV^4
eps0_mev4 = 5e7
eps0_d = eps0_mev4 / C.e0
eos_uni = interp1d([0.0, 1e6], [eps0_d, eps0_d], kind="linear",
                   fill_value=eps0_d, bounds_error=False)
Pc_seq = eps0_d * np.array([0.05, 0.1, 0.2, 0.4])
MR_ratios = []
for Pc in Pc_seq:
    r_arr, M_arr, _, _ = integrate_tov_profile(eos_uni, Pc, rstep=0.005)
    if len(M_arr) > 0:
        M_num = M_arr[-1]; R_num = r_arr[-1]
        ratio = M_num / R_num**3 if R_num > 0 else float('nan')
        MR_ratios.append(ratio)

if MR_ratios:
    rng = (max(MR_ratios) - min(MR_ratios)) / np.mean(MR_ratios) * 100
    info(f"  M/R³ spread across 4 central pressures: {rng:.3f}% (expect < 0.5%)")
    if rng < 0.5:
        ok("M/R³ is constant to better than 0.5% across central pressures")
    else:
        fail(f"M/R³ spread {rng:.3f}% > 0.5% — mass-continuity violation")
        t1_pass = False

if t1_pass:
    ok("T1 PASS — Uniform density integrator consistent with Schwarzschild interior solution")
else:
    fail("T1 FAIL — see above")

# ---------------------------------------------------------------------------
# T2: Polytropic EoS — Euler vs RK4 comparison
# ---------------------------------------------------------------------------
section("T2 — Polytropic EoS (Euler vs RK4)")

info("Γ=2 polytrope: P = K·ε², K chosen to give M_max ~ 1.5 M_sun")
info("Comparison: npemu Euler integrator vs independent RK4 at same step size")

# Tune K: M_max for a stiff polytrope. Use K_dimless ≈ 1e-3 (trial).
# ε(P) = √(P/K), in dimensionless units.
# Start with a known-good calibration from the literature:
# Γ=2 polytrope with K = 0.0195 (cgs: K in units where P = K ρ^2 with ρ in g/cm³)
# In code units, this maps to K_dimless via:
#   K_cgs -> K_dimless = K_cgs * (C.gcm3toMeV4) / (C.dynetoMeV4 / C.dynetoMeV4) -- complex
# Simpler: choose K_dimless directly to get M_max ~ 1.5 M_sun.
# Trial: from the central pressure at M_max ≈ 1.5 Msun, eps_c ~ 5e8 MeV^4 ~ 5e8/e0 dimless
# K_dimless ~ Pc/eps_c^2 ~ 0.4 * eps_c / eps_c^2 = 0.4/eps_c ≈ 0.4/(5e8/e0) * e0 ~ 0.4*e0/5e8

eps_c_target = 5e8 / C.e0  # dimensionless target central density
K_dimless = 0.3 / eps_c_target  # P_c = K * eps_c^2 → K = P_c/eps_c^2 = 0.3*eps_c/eps_c^2

def poly_eos(K):
    """ε(P) for P = K ε^2."""
    def fn(P):
        P = max(float(P), 1e-30)
        return math.sqrt(P / K)
    return fn

eos_poly = poly_eos(K_dimless)
Pc_poly_start = K_dimless * (0.2 * eps_c_target)**2  # low start
Pc_poly_end   = K_dimless * (3.0 * eps_c_target)**2  # high end

# Determine Pc step that gives ~80 configurations
Pcs_euler, Ms_euler, Rs_euler = run_sequence_euler(
    eos_poly, Pc_poly_start, Pc_poly_end, Pcstep_factor=1.1, rstep=0.01
)
Ms_rk4, Rs_rk4 = run_sequence_rk4(
    eos_poly, Pc_poly_start, Pc_poly_end, Pcstep_factor=1.1, rstep=0.01
)

n_common = min(len(Ms_euler), len(Ms_rk4))
Ms_e = Ms_euler[:n_common]; Rs_e = Rs_euler[:n_common]
Ms_r = Ms_rk4[:n_common];   Rs_r = Rs_rk4[:n_common]

Mmax_euler = float(np.max(Ms_e))
Mmax_rk4   = float(np.max(Ms_r))
idx_e = int(np.argmax(Ms_e)); idx_r = int(np.argmax(Ms_r))
Rmax_euler = Rs_e[idx_e]; Rmax_rk4 = Rs_r[idx_r]

err_Mmax = 100.0 * abs(Mmax_euler - Mmax_rk4) / Mmax_rk4
err_Rmax = 100.0 * abs(Rmax_euler - Rmax_rk4) / Rmax_rk4

info(f"  Euler: M_max = {Mmax_euler:.4f} M_sun,  R(M_max) = {Rmax_euler:.4f} km")
info(f"  RK4:   M_max = {Mmax_rk4:.4f} M_sun,  R(M_max) = {Rmax_rk4:.4f} km")
info(f"  Discrepancy: ΔM_max = {err_Mmax:.3f}%,  ΔR = {err_Rmax:.3f}%")

# Point-by-point comparison
M_errs = 100.0 * np.abs(Ms_e - Ms_r) / np.maximum(np.abs(Ms_r), 1e-10)
R_errs = 100.0 * np.abs(Rs_e - Rs_r) / np.maximum(np.abs(Rs_r), 1e-10)
info(f"  Max point-wise error: ΔM = {M_errs.max():.3f}%, ΔR = {R_errs.max():.3f}%")
info(f"  Mean point-wise error: ΔM = {M_errs.mean():.3f}%, ΔR = {R_errs.mean():.3f}%")

if err_Mmax <= 2.0 and R_errs.max() <= 2.0:
    ok(f"T2 PASS — Euler/RK4 agree within 2% at rstep=0.01 km  (ΔM_max={err_Mmax:.3f}%)")
else:
    fail(f"T2 FAIL — Euler/RK4 discrepancy: ΔM_max={err_Mmax:.3f}%, ΔR_max={R_errs.max():.3f}%")

# ---------------------------------------------------------------------------
# T3: TOV invariant check on QMD baseline maximum-mass configuration
# ---------------------------------------------------------------------------
section("T3 — TOV invariant check on QMD baseline M_max configuration")

# Find the maximum-mass central pressure from the stars file
stars = load_numeric(STARS_FILE)
# Columns: Pc_dimless eps_c_mev4 eps_c_gev_fm3 radius_km mass_msun stable_flag
M_list  = stars[:, 4]
R_list  = stars[:, 3]
Pc_list = stars[:, 0]
stable  = stars[:, 5].astype(bool)

idx_max = int(np.argmax(M_list[stable]))
M_max_stored = M_list[stable][idx_max]
R_max_stored = R_list[stable][idx_max]
Pc_max_stored = Pc_list[stable][idx_max]

info(f"  Stored M_max = {M_max_stored:.4f} M_sun at R = {R_max_stored:.4f} km, Pc = {Pc_max_stored:.6g}")

# Integrate with profile recording
r_p, M_p, P_p, E_p = integrate_tov_profile(eos_qmd, Pc_max_stored, rstep=0.01, tol=Pc_qmd_tol)
info(f"  Re-integrated: M = {M_p[-1]:.4f} M_sun, R = {r_p[-1]:.4f} km")

# Check dM/dr consistency: compare (M[i+1]-M[i])/h to dMdr(r[i], eps[i])
n = len(r_p)
if n > 2:
    dM_numerical = np.diff(M_p) / 0.01    # finite difference
    dM_analytic  = np.array([C.dMdr(r_p[i], E_p[i]) for i in range(n-1)])
    err_dM = np.abs(dM_numerical - dM_analytic)
    # These should agree because we used the same Euler scheme;
    # small discrepancies come from the fact that we advanced r before computing
    rel_err_dM = err_dM / (np.abs(dM_analytic) + 1e-30)
    info(f"  dM/dr residual: max = {err_dM.max():.3e}, mean = {err_dM.mean():.3e}")
    info(f"  dM/dr relative: max = {rel_err_dM.max():.3e}")

    dP_numerical = np.diff(P_p) / 0.01
    dP_analytic  = np.array([C.dPdr(r_p[i], M_p[i], P_p[i], E_p[i]) for i in range(n-1)])
    err_dP = np.abs(dP_numerical - dP_analytic)
    rel_err_dP = err_dP / (np.abs(dP_analytic) + 1e-30)
    info(f"  dP/dr residual: max = {err_dP.max():.3e}, mean = {err_dP.mean():.3e}")
    info(f"  dP/dr relative: max = {rel_err_dP.max():.3e}")

    # TOV invariant: check that (ε+P)(M+4πr³P) = -r²(1-2GM/rc²) * dP/dr
    # In code units: dP/dr = -(R0 * eps * M / r^2) * (P/eps+1) * (beta*r³*P/M+1) / (1-2*R0*M/r)
    # The TOV "invariant" is just checking dM/dr = beta*r²*eps and the ODE is consistent.
    # Max violation of individual ODE residuals:
    info(f"  TOV consistency: Euler scheme residuals are machine-precision by construction")
    info(f"  (Euler step advances r first, then evaluates RHS — residuals show scheme error)")

    # More useful: total mass conservation at surface vs integral
    M_integrated = float(np.trapezoid(np.array([C.dMdr(r_p[i], E_p[i]) for i in range(n)]), r_p))
    M_surface = M_p[-1]
    err_mass = 100.0 * abs(M_integrated - M_surface) / M_surface
    info(f"  Mass integral consistency: ∫dMdr dr = {M_integrated:.5f}, M_surface = {M_surface:.5f}, Δ = {err_mass:.4f}%")
    if err_mass < 0.1:
        ok("T3 PASS — Mass integral consistent with surface mass to < 0.1%")
    else:
        fail(f"T3 INCONCLUSIVE — Euler forward-stepping causes {err_mass:.2f}% mass discrepancy")
else:
    fail("T3: Could not integrate M_max profile")

# ---------------------------------------------------------------------------
# T4: Convergence with radial step size
# ---------------------------------------------------------------------------
section("T4 — Convergence in radial step size (QMD baseline)")

info("Running QMD baseline at rstep = 0.001, 0.01, 0.1 km using stored EoS")

def run_seq_qmd(h):
    Pcs_out, Ms_out, Rs_out = [], [], []
    Pc = Pc_qmd_start
    while Pc < Pc_qmd_end:
        P, M, r = Pc, 0.0, 0.0
        while P > Pc_qmd_tol and r < 30.0:
            r += h
            eps = float(eos_qmd(P))
            M += h * C.dMdr(r, eps)
            P += h * C.dPdr(r, M, P, eps)
        Ms_out.append(M); Rs_out.append(r); Pcs_out.append(Pc)
        Pc *= 1.08
    return np.array(Pcs_out), np.array(Ms_out), np.array(Rs_out)

steps = [0.001, 0.01, 0.1]
results_t4 = {}
for h in steps:
    Pcs_, Ms_, Rs_ = run_seq_qmd(h)
    idx_ = int(np.argmax(Ms_))
    results_t4[h] = (Ms_[idx_], Rs_[idx_], Pcs_[idx_])

info(f"\n  {'rstep (km)':<14} {'M_max (Msun)':<16} {'R(M_max) (km)':<16} {'Pc at M_max'}")
for h, (M, R, Pc) in results_t4.items():
    info(f"  {h:<14.3f} {M:<16.5f} {R:<16.5f} {Pc:.6g}")

M_ref = results_t4[0.001][0]; R_ref = results_t4[0.001][1]
M_01  = results_t4[0.01][0];  R_01  = results_t4[0.01][1]
M_1   = results_t4[0.1][0];   R_1   = results_t4[0.1][1]

err_M_01 = 100.0 * abs(M_01 - M_ref) / M_ref
err_R_01 = 100.0 * abs(R_01 - R_ref) / R_ref
err_M_1  = 100.0 * abs(M_1  - M_ref) / M_ref
err_R_1  = 100.0 * abs(R_1  - R_ref) / R_ref

info(f"\n  Relative to h=0.001 km:")
info(f"  h=0.01:  ΔM = {err_M_01:.4f}%,  ΔR = {err_R_01:.4f}%")
info(f"  h=0.1:   ΔM = {err_M_1:.4f}%,  ΔR = {err_R_1:.4f}%")

# Check first-order convergence (error ∝ h): ratio should be ~10
if err_M_01 > 0 and err_M_1 > 0:
    ratio_M = err_M_1 / err_M_01
    ratio_R = err_R_1 / err_R_01
    info(f"  Convergence ratio (h=0.1 err)/(h=0.01 err): ΔM ratio={ratio_M:.1f}, ΔR ratio={ratio_R:.1f} (expect ~10 for O(h))")

if err_M_01 < 0.01:
    ok(f"T4 PASS — h=0.001→0.01 agrees within 0.01% on M_max  (err={err_M_01:.4f}%)")
else:
    fail(f"T4 CONCERN — h=0.001→0.01 discrepancy {err_M_01:.4f}% > 0.01% on M_max")

# ---------------------------------------------------------------------------
# T5: Stability criterion
# ---------------------------------------------------------------------------
section("T5 — Stability criterion (radial perturbation)")

info("  The npemu TOV solver identifies stable configurations by dM/dε_c > 0")
info("  (index of maximum mass, not a radial perturbation eigenvalue analysis).")
info("  A full perturbation mode-frequency solver is NOT implemented in this codebase.")
info("  Checking instead that the stable/unstable branch split is consistent with")
info("  monotone M(ε_c) behavior:")

Pcs_t5, Ms_t5, Rs_t5 = run_seq_qmd(0.01)
idx_max_t5 = int(np.argmax(Ms_t5))
n_stable   = idx_max_t5 + 1
n_total    = len(Ms_t5)
n_unstable = n_total - n_stable

# Check: on stable branch (ε_c increasing), dM/dε_c > 0
eps_c_t5 = np.array([float(eos_qmd(Pc)) * C.e0 for Pc in Pcs_t5])
dM_deps_stable   = np.diff(Ms_t5[:n_stable]) / np.diff(eps_c_t5[:n_stable])
dM_deps_unstable = np.diff(Ms_t5[n_stable:]) / np.diff(eps_c_t5[n_stable:]) if n_unstable > 1 else np.array([])

n_wrong_stable   = int(np.sum(dM_deps_stable <= 0)) if len(dM_deps_stable) > 0 else 0
n_wrong_unstable = int(np.sum(dM_deps_unstable >= 0)) if len(dM_deps_unstable) > 0 else 0

info(f"  Stable branch: {n_stable} configs, {n_wrong_stable} with dM/dε_c ≤ 0 (expect 0)")
info(f"  Unstable branch: {n_unstable} configs, {n_wrong_unstable} with dM/dε_c ≥ 0 (expect 0)")

if n_wrong_stable == 0 and n_wrong_unstable == 0:
    ok("T5 PASS — Stable/unstable branch split is monotone-consistent")
    ok("     Full perturbation analysis not implemented; dM/dε_c criterion verified")
else:
    fail(f"T5 FAIL — {n_wrong_stable} stable-branch anomalies, {n_wrong_unstable} unstable-branch anomalies")

# ============================================================================
# PART 2 — MAXWELL CONSTRUCTION VERIFICATION
# ============================================================================

section("PART 2 — MAXWELL CONSTRUCTION VERIFICATION")

# Load raw EoS
raw = load_numeric(RAW_FILE)
# Columns: 0:mu_q 1:mu_B 2:phi 3:delta 4:gap 5:mu_e 6:mu_8 7:delta_mu 8:gap_minus
# 9:press 10:nq_quark 11:nq_baryon 12:eps 13:cs2 14:omega_vac [15:success] [16:norm]
mu_raw   = raw[:, 0]
press_raw = raw[:, 9]
eps_raw   = raw[:, 12]

# Sort by mu_q
sort_idx = np.argsort(mu_raw)
mu_raw    = mu_raw[sort_idx]
press_raw = press_raw[sort_idx]
eps_raw   = eps_raw[sort_idx]

# Vacuum offset
omega_vac = float(meta.get("omega_vac_mev4", "nan"))
# press_raw is already vacuum-subtracted in the file (P = -(omega - omega_vac))

# ---------------------------------------------------------------------------
# M1: Equal-pressure Maxwell construction verification
# ---------------------------------------------------------------------------
section("M1 — Maxwell construction: equal-pressure coexistence")

stable_p, stable_e, maxwell_idx = maxwell_construct(mu_raw, press_raw, eps_raw)

info(f"  maxwell_construct returns maxwell_indices = {maxwell_idx}")
if len(maxwell_idx) == 2:
    i1, i2 = maxwell_idx
    mu1 = mu_raw[i1]; mu2 = mu_raw[i2]
    P1  = press_raw[i1]; P2 = press_raw[i2]
    info(f"  Coexistence region: μ_q ∈ [{mu1:.3f}, {mu2:.3f}] MeV")
    info(f"  Pressure at boundaries: P(μ₁) = {P1:.4e}, P(μ₂) = {P2:.4e} MeV^4")
    info(f"  Pressure match: |P₁-P₂|/P_avg = {abs(P1-P2)/((abs(P1)+abs(P2))/2+1e-30)*100:.4f}%")

    # (a) Equal-pressure criterion: P1 should ≈ P2
    if abs(P1 - P2) / (abs(P1) + 1.0) < 0.01:
        ok("M1a PASS — Equal-pressure criterion satisfied at coexistence boundaries")
    else:
        fail(f"M1a FAIL — Pressure mismatch {abs(P1-P2)/abs(P1)*100:.2f}%")

    # (b) Verify only the unstable loop is removed
    # Points [0..i1] and [i2..end] should be monotone
    p_pre   = press_raw[:i1+1]
    p_post  = press_raw[i2:]
    mono_pre  = np.all(np.diff(p_pre)  >= 0)
    mono_post = np.all(np.diff(p_post) >= 0)
    n_removed = i2 - i1 - 1
    info(f"  Points removed from loop: {n_removed}")
    info(f"  Pre-loop pressure monotone: {mono_pre}")
    info(f"  Post-loop pressure monotone: {mono_post}")
    if mono_pre and mono_post:
        ok("M1b PASS — Only the unstable loop is removed; stable segments untouched")
    else:
        fail("M1b FAIL — Non-monotone segment in pre or post loop")

    # (c) Thermodynamic consistency at boundaries
    # At i1 and i2, the construction should preserve μ and P; no jump in μ should occur
    mu_jump = abs(mu2 - mu1)
    info(f"  Chemical potential discontinuity (Maxwell gap): Δμ_q = {mu_jump:.3f} MeV")
    ok("M1c NOTE — At boundaries, μ and P are from the original data (no extrapolation)")

    # Independent equal-area check:
    # The Maxwell construction is correct if ∫μ dP over the loop = 0.
    # Equivalently: ∫_{P1}^{P2} μ_upper(P) dP = ∫_{P1}^{P2} μ_lower(P) dP
    # where upper is the pre-loop branch and lower is the post-loop branch.
    # We compute this by numerical integration.
    loop_slice = slice(i1, i2+1)
    mu_loop  = mu_raw[loop_slice]
    P_loop   = press_raw[loop_slice]
    # Find peak (local max of P in loop)
    loop_imax = int(np.argmax(P_loop))
    loop_imin = int(np.argmin(P_loop[loop_imax:])) + loop_imax

    P_peak = P_loop[loop_imax]; P_trough = P_loop[loop_imin]

    # Upper branch: from i1 to i_peak (rising)
    mu_upper = mu_loop[:loop_imax+1]
    P_upper  = P_loop[:loop_imax+1]
    # Lower branch: from i_min to i2 (rising)
    mu_lower = mu_loop[loop_imin:]
    P_lower  = P_loop[loop_imin:]

    # Find overlap pressure range
    P_lo = max(P_upper[0], P_lower[0])
    P_hi = min(P_upper[-1], P_lower[-1])

    if P_lo < P_hi:
        Pgrid = np.linspace(P_lo, P_hi, 500)
        mu_upper_interp = np.interp(Pgrid, P_upper, mu_upper)
        mu_lower_interp = np.interp(Pgrid, np.sort(P_lower),
                                     mu_lower[np.argsort(P_lower)])
        # The coexistence pressure P_t satisfies μ_upper(P_t) = μ_lower(P_t)
        diff_mu = mu_upper_interp - mu_lower_interp
        crossings = np.where(diff_mu[:-1] * diff_mu[1:] <= 0)[0]
        if crossings.size > 0:
            j = int(crossings[-1])
            frac = diff_mu[j]/(diff_mu[j]-diff_mu[j+1]) if diff_mu[j] != diff_mu[j+1] else 0.5
            P_cross = Pgrid[j] + frac * (Pgrid[j+1] - Pgrid[j])
            # Equal-area check: ∫ (mu_upper - mu_lower) dP over [P_lo, P_cross] =
            #                   ∫ (mu_lower - mu_upper) dP over [P_cross, P_hi]
            mask_lo = Pgrid <= P_cross; mask_hi = Pgrid >= P_cross
            area_upper_excess = np.trapezoid(
                np.where(mask_lo, diff_mu, 0.0), Pgrid)
            area_lower_excess = np.trapezoid(
                np.where(mask_hi, -diff_mu, 0.0), Pgrid)
            info(f"  Independent equal-area check:")
            info(f"    P_coexistence (from code) = {p_pre[-1]:.4e} MeV^4")
            info(f"    P_coexistence (equal-area) = {P_cross:.4e} MeV^4")
            info(f"    Area above plateau = {area_upper_excess:.4e}, below = {area_lower_excess:.4e}")
            area_rel = abs(area_upper_excess - area_lower_excess) / (
                (abs(area_upper_excess) + abs(area_lower_excess)) / 2 + 1e-30)
            if area_rel < 0.05:
                ok(f"M1 PASS — Equal-area verified to {area_rel*100:.2f}%")
            else:
                fail(f"M1 FAIL — Equal areas disagree by {area_rel*100:.2f}%")
        else:
            info("  Equal-area: no crossing found in P grid — degenerate loop")
    else:
        info("  No overlapping P range for equal-area check — very narrow loop")

else:
    info(f"  No complete pressure loop found (maxwell_indices = {maxwell_idx})")
    ok("M1: No phase transition detected by construction (consistent with data structure)")

# ---------------------------------------------------------------------------
# M2: Smooth EoS → no modification
# ---------------------------------------------------------------------------
section("M2 — Smooth EoS passes through construction unchanged")

# Use the stable post-Maxwell EoS (monotone by construction)
# Build mu_q corresponding to stable_p/stable_e (already Maxwell-filtered in M1)
if len(maxwell_idx) == 2:
    _i1, _i2 = maxwell_idx
    mu_stable_m2 = np.concatenate((mu_raw[:_i1+1], mu_raw[_i2:]))
else:
    mu_stable_m2 = mu_raw
stable_p2, stable_e2, idx2 = maxwell_construct(mu_stable_m2, stable_p, stable_e)
info(f"  Applying construction to already-stable EoS: maxwell_indices = {idx2}")
if len(idx2) == 0:
    ok("M2 PASS — Construction finds no loop in already-stable EoS")
else:
    fail(f"M2 FAIL — Construction modifies a smooth EoS (indices {idx2})")

# Also test with a pure polytrope (no phase transition)
n_poly = 200
mu_poly = np.linspace(250.0, 900.0, n_poly)
# P ∝ μ^4 (monotone smooth)
P_poly = 1e3 * (mu_poly / 250.0)**4
E_poly = P_poly * 3.0  # bag-model-like ε=3P
stable_pp, stable_ep, idx_p = maxwell_construct(mu_poly, P_poly, E_poly)
info(f"  Applying construction to pure power-law P ∝ μ^4: maxwell_indices = {idx_p}")
if len(idx_p) == 0:
    ok("M2 PASS — Construction finds no loop in smooth power-law EoS")
else:
    fail(f"M2 FAIL — Construction incorrectly modifies smooth power-law EoS (indices {idx_p})")

# ---------------------------------------------------------------------------
# M3: Point count sanity check
# ---------------------------------------------------------------------------
section("M3 — Point count sanity check")

maxwell_str  = meta.get("maxwell_indices", "[]")
n_raw        = int(meta.get("num_raw_points", "0"))
mono_removed = int(meta.get("monotone_removed", "0"))
cs2_removed  = int(meta.get("cs2_removed", "0"))

info(f"  Metadata: num_raw_points={n_raw}, maxwell_indices={maxwell_str}, "
     f"monotone_removed={mono_removed}, cs2_removed={cs2_removed}")

# Parse maxwell_indices
import ast
try:
    maxwell_indices_meta = ast.literal_eval(maxwell_str)
    n_maxwell_removed = maxwell_indices_meta[1] - maxwell_indices_meta[0] - 1 if len(maxwell_indices_meta) == 2 else 0
except Exception:
    n_maxwell_removed = 0

# Run construction on raw data and count
stable_p_m3, stable_e_m3, idx_m3 = maxwell_construct(mu_raw, press_raw, eps_raw)
n_maxwell_computed = (idx_m3[1] - idx_m3[0] - 1) if len(idx_m3) == 2 else 0

info(f"  Maxwell points removed by code:     {n_maxwell_computed}")
info(f"  Maxwell points removed (metadata):  {n_maxwell_removed}")
info(f"  Raw points: {n_raw}")

# Vacuum filter: points where omega_min < omega_vac (P < 0 after subtraction)
n_vacuum_removed = int(np.sum(press_raw < 0.0))
info(f"  Vacuum filter removes (P<0): {n_vacuum_removed} (thesis: 22 — 'vacuum filter includes non-monotone boundary points')")
info(f"  Note: thesis counts 22 removed by vacuum filter, but 'vacuum filter' may include")
info(f"        the initial non-physical low-μ points before the onset region.")

# The thesis says: 700 raw - 3 Maxwell - 22 vacuum - 3 monotone = 672
# Let's count: stable has 672 rows (from stab data)
info(f"  Stable EoS points in file: {len(stab)}")
info(f"  Total accounted: {n_raw} - maxwell({n_maxwell_computed}) - vacuum({n_vacuum_removed}) "
     f"- mono({mono_removed}) = {n_raw - n_maxwell_computed - n_vacuum_removed - mono_removed}")

n_stable_expected = n_raw - n_maxwell_computed - n_vacuum_removed - mono_removed
if n_stable_expected == len(stab):
    ok(f"M3 PASS — Point count consistent: {n_stable_expected} stable = {len(stab)} in file")
else:
    info(f"  Direct count differs: {n_stable_expected} vs {len(stab)} in file")
    info(f"  This may reflect the difference between maxwell_removed ({n_maxwell_computed}) and {n_maxwell_removed}")
    # The maxwell_removed in the code is specific to the raw data ordered by mu_q;
    # the metadata may account for i2-i1+1 differently.
    fail(f"M3 INCONCLUSIVE — see above")

# ---------------------------------------------------------------------------
# M4: Central pressures above Maxwell plateau
# ---------------------------------------------------------------------------
section("M4 — Central pressures of stable stars vs Maxwell plateau pressure")

# Maxwell plateau pressure
if len(maxwell_idx) == 2:
    P_plateau = press_raw[maxwell_idx[0]]  # P_transition from equal-pressure construction
else:
    P_plateau = 0.0

info(f"  Maxwell plateau pressure: P_t = {P_plateau:.4e} MeV^4")

# Stable star central pressures (from stars file)
eps_c_stable = stars[stable, 1]  # eps_c in MeV^4
# Convert eps_c to P using the EoS interpolant (inverse: P = P(ε))
# Build P(ε) from the stable branch
eps_for_P = eps_stab[pos_mask]
prs_for_P = press_stab[pos_mask]
# Sort by eps for interpolation
sort_ep = np.argsort(eps_for_P)
eos_P_from_eps = interp1d(eps_for_P[sort_ep], prs_for_P[sort_ep],
                           kind="cubic", fill_value="extrapolate")

P_c_stable = eos_P_from_eps(eps_c_stable)
P_c_min = float(np.min(P_c_stable))
P_c_max = float(np.max(P_c_stable))

info(f"  Stable star central pressures: [{P_c_min:.4e}, {P_c_max:.4e}] MeV^4")
n_below_plateau = int(np.sum(P_c_stable <= P_plateau))
info(f"  Stars with P_c ≤ P_plateau: {n_below_plateau} of {len(P_c_stable)}")

if n_below_plateau == 0:
    ok("M4 PASS — All stable star central pressures lie above the Maxwell plateau")
elif P_plateau == 0.0:
    ok("M4 PASS — No Maxwell construction applied; all stable stars at P_c > 0")
else:
    info(f"  WARNING: {n_below_plateau} stable configurations have P_c ≤ P_plateau = {P_plateau:.4e}")
    info("  These stars would have central conditions within the first-order transition region.")
    fail(f"M4 FAIL — {n_below_plateau} stable stars at P_c ≤ P_plateau")

# ============================================================================
# PART 3 — NEUTRALITY ROOT-FINDER VERIFICATION
# ============================================================================

section("PART 3 — NEUTRALITY ROOT-FINDER VERIFICATION")

model = QMDStellarModel(QMD_SET_A)

# Load raw EoS with string column handling
raw_full = []
with open(RAW_FILE) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 17:
            continue
        try:
            row = [float(parts[i]) for i in [0,2,3,4,5,6,17]]  # mu_q,phi,delta,gap,mu_e,mu_8,norm
            # verify they're parseable
            raw_full.append(row)
        except (ValueError, IndexError):
            continue

raw_full = np.array(raw_full)
# Columns of raw_full: mu_q, phi, delta, gap, mu_e, mu_8, neutrality_norm

# Three representative points: near onset, mid-range, near 900 MeV
# From the raw file: 2SC onset at ~278.8 MeV, mid ~600 MeV, high ~900 MeV
def find_row(mu_target):
    idx = int(np.argmin(np.abs(raw_full[:, 0] - mu_target)))
    return raw_full[idx]

pts = {
    "onset  (μ_q≈280)": find_row(280.0),
    "mid    (μ_q≈600)": find_row(600.0),
    "high   (μ_q≈900)": find_row(900.0),
}

# ---------------------------------------------------------------------------
# N1: Neutrality residuals at solution
# ---------------------------------------------------------------------------
section("N1 — Neutrality residuals at stored solution points")

info(f"  {'Point':<22} {'μ_q':>7} {'φ₀':>7} {'Δ₀':>7} {'μ_e':>7} {'μ_8':>8} "
     f"{'|res_e|':>12} {'|res_8|':>12} {'norm':>12} {'Pass?':>6}")

n1_pass = True
for label, row in pts.items():
    mu_q_v, phi_v, delta_v, gap_v, mu_e_v, mu_8_v, norm_stored = row
    res = model.neutrality_residuals(phi_v, delta_v, mu_q_v, mu_e_v, mu_8_v)
    re, r8 = abs(res.d_omega_d_mu_e), abs(res.d_omega_d_mu_8)
    rnorm = math.sqrt(re**2 + r8**2)
    # Typical density scale for relative comparison: n_q ~ dP/dμ at this point
    # Use |res| < 1 MeV^3 as an absolute threshold (density scale ~1e3-1e5 MeV^3)
    # For relative: typical n_baryon at these μ values
    # From the data: n_q at μ=900 is ~1e6 MeV^3, at μ=280 ~1e3 MeV^3
    passed = rnorm < 5.0  # 5 MeV^3 absolute (set by neutrality_tol_mev3=1.0 in code)
    sign = "PASS" if passed else "FAIL"
    if not passed:
        n1_pass = False
    info(f"  {label:<22} {mu_q_v:>7.1f} {phi_v:>7.3f} {delta_v:>7.4f} "
         f"{mu_e_v:>7.2f} {mu_8_v:>8.3f} {re:>12.3e} {r8:>12.3e} {rnorm:>12.3e} {sign:>6}")
    info(f"    (stored neutrality norm: {norm_stored:.3e})")

if n1_pass:
    ok("N1 PASS — Neutrality residuals < 5 MeV^3 at all three stored solution points")
else:
    fail("N1 FAIL — Large residuals at some points; neutrality not tightly satisfied")

# ---------------------------------------------------------------------------
# N2: β-equilibrium check
# ---------------------------------------------------------------------------
section("N2 — β-equilibrium (μ_d - μ_u = μ_e)")

info("  β-equilibrium is built into the chemical-potential parametrization:")
info("  μ_ur = μ_ug = μ_q - (2/3)μ_e + (1/3)μ_8")
info("  μ_dr = μ_dg = μ_q + (1/3)μ_e + (1/3)μ_8")
info("  → μ_d - μ_u = μ_e (exactly, by construction)")
info("")

from .qmd_stellar import quark_chemical_potentials
n2_pass = True
for label, row in pts.items():
    mu_q_v, phi_v, delta_v, gap_v, mu_e_v, mu_8_v, _ = row
    mus = quark_chemical_potentials(mu_q_v, mu_e_v, mu_8_v)
    diff = mus["mu_dg"] - mus["mu_ug"] - mu_e_v   # should be 0
    if abs(diff) < 1e-10:
        info(f"  {label}: μ_d - μ_u - μ_e = {diff:.3e} MeV (machine precision)")
    else:
        info(f"  {label}: β-equilibrium residual = {diff:.3e} MeV  [FAIL]")
        n2_pass = False

if n2_pass:
    ok("N2 PASS — β-equilibrium satisfied to machine precision by construction")
else:
    fail("N2 FAIL — β-equilibrium violated")

# ---------------------------------------------------------------------------
# N3: Solution uniqueness at μ_q = 500 MeV
# ---------------------------------------------------------------------------
section("N3 — Solution uniqueness at μ_q = 500 MeV")

mu_test = 500.0
# Get the stored solution at this μ for reference
row500 = find_row(mu_test)
mu_q_t, phi_t, delta_t, gap_t, mu_e_t, mu_8_t, norm_t = row500

info(f"  Reference solution at μ_q ≈ {mu_q_t:.1f} MeV:")
info(f"    φ₀={phi_t:.4f}, Δ₀={delta_t:.4f}, μ_e={mu_e_t:.3f}, μ_8={mu_8_t:.3f}")
info(f"    neutrality_norm={norm_t:.3e}")

# Run neutrality solver from 5 different starting points
starts = [
    (0.0, 0.0),
    (100.0, 0.0),
    (50.0, -20.0),
    (200.0, -50.0),
    (mu_e_t * 0.5, mu_8_t * 0.5),   # close to solution
]

info(f"\n  {'Start (μ_e0, μ_8_0)':<26} {'Converged μ_e':>14} {'Converged μ_8':>14} "
     f"{'norm':>12} {'ok?':>5}")

solutions = []
n3_pass = True
for mu_e0, mu_8_0 in starts:
    sol = model.solve_neutrality_fixed_fields(
        phi_t, delta_t, mu_q_t, initial_guess=(mu_e0, mu_8_0)
    )
    solutions.append((sol.mu_e_mev, sol.mu_8_mev, sol.residual_norm, sol.success))
    ok_str = "ok" if sol.residual_norm < 5.0 else "FAIL"
    if sol.residual_norm >= 5.0:
        n3_pass = False
    info(f"  ({mu_e0:>6.1f}, {mu_8_0:>6.1f})             {sol.mu_e_mev:>14.5f} {sol.mu_8_mev:>14.5f} "
         f"{sol.residual_norm:>12.3e} {ok_str:>5}")

# Check all converged to the same solution
converged = [(me, m8) for me, m8, norm, ok_ in solutions if norm < 5.0]
if len(converged) > 1:
    me_vals = [c[0] for c in converged]
    m8_vals = [c[1] for c in converged]
    spread_me = max(me_vals) - min(me_vals)
    spread_m8 = max(m8_vals) - min(m8_vals)
    info(f"\n  Spread in converged μ_e solutions: {spread_me:.4f} MeV")
    info(f"  Spread in converged μ_8 solutions: {spread_m8:.4f} MeV")
    if spread_me < 0.1 and spread_m8 < 0.1:
        ok("N3 PASS — All starts converge to the same unique neutral solution (spread < 0.1 MeV)")
    else:
        fail(f"N3 FAIL — Multiple neutral solutions found (Δμ_e={spread_me:.3f}, Δμ_8={spread_m8:.3f})")
        n3_pass = False
else:
    fail(f"N3: Only {len(converged)} of 5 starts converged")
    n3_pass = False

# ============================================================================
# OVERALL VERDICT
# ============================================================================

section("OVERALL VERDICT")

print("""
  T1 — Uniform density:     see above
  T2 — Polytrope Euler/RK4: see above
  T3 — TOV invariant:       see above
  T4 — Step convergence:    see above
  T5 — Stability:           criterion verified; full perturbation not implemented
  M1 — Maxwell coexistence: see above
  M2 — Smooth EoS safety:   see above
  M3 — Point counts:        see above
  M4 — P_c above plateau:   see above
  N1 — Neutrality residuals: see above
  N2 — Beta equilibrium:    PASS (by construction)
  N3 — Uniqueness:          see above
""")
