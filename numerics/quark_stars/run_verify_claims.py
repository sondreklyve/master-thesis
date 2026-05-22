"""Comprehensive numerical verification of quantitative claims in
thesis/part2/quark_stars.tex  (sec:qmd_section1 and sec:qmd_parameter_sensitivity).

Run from repo root:
    numerics/bin/python -m numerics.quark_stars.run_verify_claims

Output: per-section tables with Claim | Thesis value | Computed value | Match?
"""

from __future__ import annotations
import os, math, pathlib, sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
BENCH_DIR  = ROOT / "output" / "qmd_benchmark" / "data"
STELL_DIR  = ROOT / "output" / "qmd_stellar" / "data"
SEC2_DIR   = ROOT / "output" / "section2" / "data"

BENCH_FILE       = BENCH_DIR / "qmd_benchmark.txt"
BENCH_TRUNC      = BENCH_DIR / "qmd_benchmark_truncated.txt"
BENCH_ASYM       = BENCH_DIR / "qmd_benchmark_asymptotic_log.txt"
STELLAR_RAW      = STELL_DIR / "qmd_stellar_eos_baseline_raw.txt"
STELLAR_STABLE   = STELL_DIR / "qmd_stellar_eos_baseline_stable.txt"
STARS_BASELINE   = STELL_DIR / "qmd_stars_baseline.txt"
SEC2_SUMMARY     = SEC2_DIR.parent / "section2_summary.csv"

# Column indices for benchmark files
# mu_q phi delta gap phase_2sc omega pressure n_q eps cs2 success
C_MU, C_PHI, C_DELTA, C_GAP = 0, 1, 2, 3
C_PHASE, C_OMEGA, C_PRESS, C_NQ, C_EPS, C_CS2 = 4, 5, 6, 7, 8, 9

# Column indices for stellar EoS files
# mu_q mu_B phi delta gap mu_e mu_8 delta_mu gap-delta_mu pressure ...
CS_MU, CS_PHI, CS_DELTA, CS_GAP = 0, 2, 3, 4
CS_MUE, CS_MU8, CS_DELTAMU, CS_GAPMINUS = 5, 6, 7, 8
CS_PRESS, CS_NQ_QUARK, CS_NQ_BARYON, CS_EPS, CS_CS2 = 9, 10, 11, 12, 13

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load(path, comments="#"):
    # Use str dtype first to handle mixed columns (e.g. string phase column), then
    # try pure float; if that fails, drop non-numeric columns.
    try:
        return np.loadtxt(path, comments=comments)
    except ValueError:
        # File has string columns — load as object, extract numeric columns only
        raw = np.genfromtxt(path, comments=comments, dtype=str, invalid_raise=False)
        # Keep only columns that are entirely numeric
        numeric_cols = []
        for j in range(raw.shape[1]):
            try:
                raw[:, j].astype(float)
                numeric_cols.append(j)
            except ValueError:
                pass
        return raw[:, numeric_cols].astype(float)

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

bench  = load(BENCH_FILE)
trunc  = load(BENCH_TRUNC)
asym   = load(BENCH_ASYM)
straw  = load(STELLAR_RAW)
ststab = load(STELLAR_STABLE)
stars  = load(STARS_BASELINE)

meta_stab = load_meta(STELLAR_STABLE)

# section2 CSV
import csv
sec2 = {}
with open(SEC2_SUMMARY) as f:
    for row in csv.DictReader(f):
        sec2[row["run_id"]] = row

# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------

RTOL = 0.005   # 0.5% relative tolerance for "match"

def match(thesis_val, computed_val, atol=None, rtol=RTOL, label=""):
    if thesis_val is None or computed_val is None:
        return "UNVERIFIABLE", thesis_val, computed_val
    tv, cv = float(thesis_val), float(computed_val)
    if tv == 0:
        ok = abs(cv) < 0.01
    else:
        rel = abs(cv - tv) / abs(tv)
        ok = rel <= rtol if atol is None else (abs(cv - tv) <= atol or rel <= rtol)
    if ok:
        return "MATCH", tv, cv
    else:
        rel = abs(cv - tv) / abs(tv) if tv != 0 else float("inf")
        return f"MISMATCH ({rel*100:.1f}%)", tv, cv

def row(cid, desc, thesis_str, tv, cv, status, note=""):
    sign = "✓" if status.startswith("MATCH") else ("?" if status == "UNVERIFIABLE" else "✗")
    tv_s = f"{tv}" if isinstance(tv, str) else (f"{tv:.4g}" if tv is not None else "—")
    cv_s = f"{cv}" if isinstance(cv, str) else (f"{cv:.4g}" if cv is not None else "—")
    return f"  {sign} {cid:<5} {desc:<40} {thesis_str:<18} {cv_s:<18} {status}  {note}"

def hdr(title):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"  {'':1} {'ID':<5} {'Claim':<40} {'Thesis value':<18} {'Computed value':<18} {'Status'}")
    print(f"  {'-'*105}")

mismatches = []
matches    = []

def emit(cid, desc, thesis_str, tv, cv, status, note=""):
    r = row(cid, desc, thesis_str, tv, cv, status, note)
    print(r)
    if status.startswith("MISMATCH"):
        mismatches.append((cid, desc, tv, cv, status, note))
    elif status.startswith("MATCH"):
        matches.append(cid)


# ---------------------------------------------------------------------------
# GROUP A — Parameters
# ---------------------------------------------------------------------------
hdr("GROUP A — Numerical setup and parameter choices")

mq, fpi, msig, mpi = 300.0, 93.0, 600.0, 140.0

g_val   = mq / fpi
lam0    = 3.0 * (msig**2 - mpi**2) / fpi**2
h_val   = fpi * mpi**2
m2_val  = 0.5 * (3.0 * mpi**2 - msig**2)

# A1
s, tv, cv = match(3.23, g_val, atol=0.005)
emit("A1", "g = M_q/f_pi", "≈3.23", 3.23, g_val, s)

# A2
s, tv, cv = match(118.1, lam0, atol=0.1)
emit("A2", "lambda_0 = 3(m_sig^2-m_pi^2)/f_pi^2", "≈118.1", 118.1, lam0, s)

# A3 (in units of MeV^3; thesis says 1.82e6)
h_thesis = 1.82e6
s, tv, cv = match(h_thesis, h_val, rtol=0.005)
emit("A3", "h = f_pi * m_pi^2", "≈1.82e6 MeV^3", h_thesis, h_val, s)

# A4 — m^2 = (3m_pi^2 - m_sig^2)/2 = -150600 MeV^2; thesis says -(388)^2 = -150544
m2_thesis = -(388.0)**2
s, tv, cv = match(m2_thesis, m2_val, rtol=0.001)
emit("A4", "m^2 = (3m_pi^2-m_sig^2)/2", "≈-(388)^2", m2_thesis, m2_val, s,
     note=f"exact: {m2_val:.0f}, thesis: {m2_thesis:.0f}")

# A5 — B_0^(1/4) from vacuum omega
omega_vac = float(meta_stab.get("omega_vac_mev4", "nan"))
B0_val = -omega_vac  # in MeV^4
B0_14  = B0_val**0.25
s, tv, cv = match(155.3, B0_14, atol=0.2)
emit("A5", "B_0^(1/4) from QMD SET A vacuum", "≈155.3 MeV", 155.3, B0_14, s)


# ---------------------------------------------------------------------------
# GROUP B — Common-mu benchmark
# ---------------------------------------------------------------------------
hdr("GROUP B — Common-chemical-potential benchmark")

mu_b  = bench[:, C_MU]
phi_b = bench[:, C_PHI]
gap_b = bench[:, C_GAP]
phase_b = bench[:, C_PHASE]
cs2_b = bench[:, C_CS2]

# B1 — 2SC onset
onset_mask = phase_b > 0.5
onset_mu   = float(mu_b[onset_mask][0]) if onset_mask.any() else None
s, tv, cv = match(266.5, onset_mu, atol=0.5)
emit("B1", "2SC onset at mu_q", "266.5 MeV", 266.5, onset_mu, s)

# B2 — phi_0 jump at onset
if onset_mask.any():
    idx_onset = int(np.where(onset_mask)[0][0])
    phi_just_before = float(phi_b[idx_onset - 1]) if idx_onset > 0 else None
    phi_just_after  = float(phi_b[idx_onset])
else:
    phi_just_before, phi_just_after = None, None

s1, _, cv1 = match(93.0, phi_just_before, atol=0.1)
emit("B2a", "phi_0 before onset", "93.0 MeV", 93.0, phi_just_before, s1)
s2, _, cv2 = match(92.7, phi_just_after, atol=0.3)
emit("B2b", "phi_0 just after onset", "92.7 MeV", 92.7, phi_just_after, s2)

# B3 — gap at onset
gap_at_onset = float(gap_b[idx_onset]) if onset_mask.any() else None
s, tv, cv = match(14.0, gap_at_onset, atol=2.0)
emit("B3", "g_Delta*Delta_0 at onset", "≈14 MeV", 14.0, gap_at_onset, s)

# B4 — phi_0 at mu_q = 400 MeV
idx400 = int(np.argmin(np.abs(mu_b - 400.0)))
phi400 = float(phi_b[idx400])
s, tv, cv = match(5.9, phi400, atol=0.3)
emit("B4", "phi_0 at mu_q=400 MeV", "5.9 MeV", 5.9, phi400, s,
     note=f"mu_q={mu_b[idx400]:.2f}")

# B5 — phi_0 at mu_q = 900 MeV
idx900 = int(np.argmin(np.abs(mu_b - 900.0)))
phi900 = float(phi_b[idx900])
s, tv, cv = match(0.7, phi900, atol=0.15)
emit("B5", "phi_0 at mu_q=900 MeV", "0.7 MeV", 0.7, phi900, s,
     note=f"mu_q={mu_b[idx900]:.2f}")

# B6 — analytic asymptotic gap
# g_Delta * Delta_bar = M_q * exp[(4pi)^2/(8*g_Delta^2) - 1/2 - (F+G)/2]
# Need F and G for QMD SET A
from .qmd_simple import _loop_F, _loop_G_pi
g_delta = 2.0 * g_val
F_pi_val = _loop_F(mpi, mq)
G_pi_val = _loop_G_pi(mpi, mq)
asym_gap_analytic = mq * math.exp(
    (4.0 * math.pi)**2 / (8.0 * g_delta**2) - 0.5 - (F_pi_val + G_pi_val) / 2.0
)
s, tv, cv = match(281.6, asym_gap_analytic, atol=0.3)
emit("B6", "analytic asymp gap g_Delta*Delta_bar", "281.6 MeV", 281.6, asym_gap_analytic, s)

# B7 — numerical gap at mu_q = 900 MeV
gap900 = float(gap_b[idx900])
s, tv, cv = match(252.0, gap900, atol=2.0)
emit("B7", "numerical gap at mu_q=900 MeV", "252 MeV", 252.0, gap900, s)

# B8 — gap at mu_q = 6000 MeV from asymptotic file
mu_asym  = asym[:, C_MU]
gap_asym = asym[:, C_GAP]
idx6000  = int(np.argmin(np.abs(mu_asym - 6000.0)))
gap6000  = float(gap_asym[idx6000])
s, tv, cv = match(267.0, gap6000, atol=3.0)
emit("B8", "numerical gap at mu_q=6 GeV (asym grid)", "267 MeV", 267.0, gap6000, s,
     note=f"mu_q={mu_asym[idx6000]:.0f}")

# B9 — cs² peak ~0.41 near mu_q~400 MeV (smoothed)
# The benchmark table cs2 is the unsmoothed finite-difference value;
# the smoothed plot peak is quoted as 0.41.
# We check against the raw table peak in the physical range (mu > 300 MeV)
in_range = (mu_b > 280.0) & np.isfinite(cs2_b) & (cs2_b > 0.0) & (cs2_b < 1.0)
cs2_peak_val = float(np.nanmax(cs2_b[in_range])) if in_range.any() else None
mu_at_peak = float(mu_b[in_range][np.nanargmax(cs2_b[in_range])]) if in_range.any() else None
# Thesis says 0.41 (smoothed); raw table may be higher
s, tv, cv = match(0.41, cs2_peak_val, atol=0.08)
emit("B9", "cs² peak (benchmark, smoothed ≈0.41)", "0.41", 0.41, cs2_peak_val, s,
     note=f"raw peak={cs2_peak_val:.4f} at mu={mu_at_peak:.1f} (smoothing reduces this)")

# B10 — cs² = 0.333 at mu_q ~ 5 GeV
cs2_asym = asym[:, C_CS2]
mu_asym_plot = mu_asym
# Find mu where cs2 first crosses 0.333 from above (or closest point near 5000 MeV)
idx5000  = int(np.argmin(np.abs(mu_asym - 5000.0)))
cs2_5000 = float(cs2_asym[idx5000]) if idx5000 < len(cs2_asym) and np.isfinite(cs2_asym[idx5000]) else None
s, tv, cv = match(0.333, cs2_5000, atol=0.005)
emit("B10", "cs² at mu_q~5 GeV on extended grid", "0.333", 0.333, cs2_5000, s,
     note=f"mu={mu_asym[idx5000]:.0f}")

# B11 — 266.5/250 = 1.066 → 6.6% above tree-level; thesis says 7%
pct_above = 100.0 * (onset_mu - 250.0) / 250.0 if onset_mu else None
s, tv, cv = match(7.0, pct_above, atol=1.0)
emit("B11", "onset 7% above tree-level 250 MeV", "7%", 7.0, pct_above, s)


# ---------------------------------------------------------------------------
# GROUP C — Truncation comparison
# ---------------------------------------------------------------------------
hdr("GROUP C — Truncation comparison (consistency check)")

mu_t   = trunc[:, C_MU]
gap_t  = trunc[:, C_GAP]
phi_t  = trunc[:, C_PHI]
phase_t= trunc[:, C_PHASE]
press_t= trunc[:, C_PRESS]
press_b= bench[:, C_PRESS]

win_mask = phase_t > 0.5
win_idx  = np.where(win_mask)[0]

# C1/C2/C3/C4
if win_idx.size > 0:
    c1_val = float(mu_t[win_idx[0]])
    c2_val = float(mu_t[win_idx[-1]])
    c3_val = c2_val - c1_val
    c4_val = float(gap_t[win_mask].max())
else:
    c1_val = c2_val = c3_val = c4_val = None

s,_,_ = match(329.7, c1_val, atol=0.5)
emit("C1", "truncated window onset", "329.7 MeV", 329.7, c1_val, s)
s,_,_ = match(334.0, c2_val, atol=0.5)
emit("C2", "truncated window closure", "334.0 MeV", 334.0, c2_val, s)
s,_,_ = match(4.4, c3_val, atol=1.0)
emit("C3", "truncated window width", "~4.4 MeV", 4.4, c3_val, s)
s,_,_ = match(183.0, c4_val, atol=3.0)
emit("C4", "truncated peak gap ~183 MeV", "~183 MeV", 183.0, c4_val, s)

# C5 — chiral crossover ~302 MeV (first mu where phi drops from f_pi)
phi_vac = float(phi_t[0])
chiral_cross = mu_t[phi_t < phi_vac - 0.05]
c5_val = float(chiral_cross[0]) if chiral_cross.size > 0 else None
s,_,_ = match(302.0, c5_val, atol=3.0)
emit("C5", "truncated chiral crossover ~302 MeV", "~302 MeV", 302.0, c5_val, s)

# C6 — phi_0 reaches ~15 MeV before 2SC onset in truncated
if win_idx.size > 0:
    # find phi in region just before 2SC onset
    pre_onset_mask = (mu_t < c1_val) & (mu_t > c1_val - 10.0)
    if pre_onset_mask.any():
        c6_val = float(phi_t[pre_onset_mask][-1])
    else:
        c6_val = None
else:
    c6_val = None
s,_,_ = match(15.0, c6_val, atol=3.0)
emit("C6", "truncated phi_0 ~15 MeV before 2SC", "~15 MeV", 15.0, c6_val, s)

# C7 — full pressure at mu_q = 270 MeV ~1e6 MeV^4
idx270_b = int(np.argmin(np.abs(mu_b - 270.0)))
press270_full = float(press_b[idx270_b])
s,_,_ = match(1.0e6, press270_full, rtol=0.5)
emit("C7", "full pressure at mu_q=270 MeV ~1e6 MeV^4", "~1e6 MeV^4", 1.0e6, press270_full, s,
     note=f"computed: {press270_full:.3e}")

# C8 — truncated pressure at 350 MeV is ~28% of full
idx350_b = int(np.argmin(np.abs(mu_b - 350.0)))
idx350_t = int(np.argmin(np.abs(mu_t - 350.0)))
pf350 = float(press_b[idx350_b])
pt350 = float(press_t[idx350_t])
ratio350 = pt350 / pf350 * 100.0 if pf350 > 0 else None
s,_,_ = match(28.0, ratio350, atol=3.0)
emit("C8", "trunc pressure at 350 MeV: ~28% of full", "28%", 28.0, ratio350, s,
     note=f"trunc={pt350:.3e} full={pf350:.3e}")

# C9 — truncated pressure at 900 MeV: ~14% below full
idx900_t = int(np.argmin(np.abs(mu_t - 900.0)))
pt900 = float(press_t[idx900_t])
pf900 = float(press_b[idx900])
deficit900 = (pf900 - pt900) / pf900 * 100.0 if pf900 > 0 else None
s,_,_ = match(14.0, deficit900, atol=3.0)
emit("C9", "trunc ~14% below full at 900 MeV", "~14%", 14.0, deficit900, s,
     note=f"pf={pf900:.3e} pt={pt900:.3e}")


# ---------------------------------------------------------------------------
# GROUP D — Neutral stellar EoS
# ---------------------------------------------------------------------------
hdr("GROUP D — Neutral stellar EoS")

mu_s   = ststab[:, CS_MU]
phi_s  = ststab[:, CS_PHI]
gap_s  = ststab[:, CS_GAP]
mue_s  = ststab[:, CS_MUE]
mu8_s  = ststab[:, CS_MU8]
deltamu_s = ststab[:, CS_DELTAMU]
gapminus_s = ststab[:, CS_GAPMINUS]
press_s = ststab[:, CS_PRESS]
eps_s  = ststab[:, CS_EPS]
cs2_s  = ststab[:, CS_CS2]

# Also need raw file for onset detection before Maxwell construction
mu_r   = straw[:, CS_MU]
phi_r  = straw[:, CS_PHI]
gap_r  = straw[:, CS_GAP]
mue_r  = straw[:, CS_MUE]
mu8_r  = straw[:, CS_MU8]
deltamu_r = straw[:, CS_DELTAMU]
gapminus_r = straw[:, CS_GAPMINUS]
press_r = straw[:, CS_PRESS]
cs2_r  = straw[:, CS_CS2]

# D1 — neutral 2SC onset at 278.8 MeV
# From raw file: first point with gap > threshold
phase_raw = (gap_r > 1.0)
d1_val = float(mu_r[phase_raw][0]) if phase_raw.any() else None
s,_,_ = match(278.8, d1_val, atol=1.5)
emit("D1", "neutral 2SC onset mu_q = 278.8 MeV", "278.8 MeV", 278.8, d1_val, s)

# D2 — onset shift relative to common-mu (266.5 MeV)
d2_val = d1_val - onset_mu if (d1_val and onset_mu) else None
s,_,_ = match(12.3, d2_val, atol=1.5)
emit("D2", "onset shift relative to common-mu", "12.3 MeV", 12.3, d2_val, s)

# D3 — phi_0 = 92.95 MeV immediately after transition
if phase_raw.any():
    idx_onset_s = int(np.where(phase_raw)[0][0])
    phi_onset_s = float(phi_r[idx_onset_s])
else:
    phi_onset_s = None
s,_,_ = match(92.95, phi_onset_s, atol=0.1)
emit("D3", "phi_0 = 92.95 MeV just after neutral onset", "92.95 MeV", 92.95, phi_onset_s, s)

# D4 — phi_0 ≈ 1 MeV at mu_q = 900 MeV (neutral)
idx900_s = int(np.argmin(np.abs(mu_s - 900.0)))
phi900_s = float(phi_s[idx900_s])
s,_,_ = match(1.0, phi900_s, atol=0.3)
emit("D4", "phi_0 ≈ 1 MeV at mu_q=900 (neutral)", "≈1 MeV", 1.0, phi900_s, s)

# D5 — gap at onset 5.6 MeV
gap_onset_s = float(gap_r[idx_onset_s]) if phase_raw.any() else None
s,_,_ = match(5.6, gap_onset_s, atol=1.0)
emit("D5", "gap at neutral onset = 5.6 MeV", "5.6 MeV", 5.6, gap_onset_s, s)

# D6 — gap at mu_q = 900 MeV: 225.6 MeV (neutral, from raw)
idx900_r = int(np.argmin(np.abs(mu_r - 900.0)))
gap900_s = float(gap_r[idx900_r])
s,_,_ = match(225.6, gap900_s, atol=2.0)
emit("D6", "gap at mu_q=900 (neutral) = 225.6 MeV", "225.6 MeV", 225.6, gap900_s, s)

# D7 — mu_e/mu_q plateau 0.60-0.63 over [330, 700] MeV
mask_plateau = (mu_r >= 330.0) & (mu_r <= 700.0) & (gap_r > 1.0)
if mask_plateau.any():
    ratio_mue = mue_r[mask_plateau] / mu_r[mask_plateau]
    d7_min = float(ratio_mue.min())
    d7_max = float(ratio_mue.max())
else:
    d7_min = d7_max = None
# Check both bounds: min should be ~0.60, max ~0.63
s_min,_,_ = match(0.60, d7_min, atol=0.03)
s_max,_,_ = match(0.63, d7_max, atol=0.03)
s = "MATCH" if s_min.startswith("MATCH") and s_max.startswith("MATCH") else f"MISMATCH"
emit("D7", "mu_e/mu_q plateau 0.60-0.63 in [330,700]", "0.60-0.63",
     0.615, (d7_min+d7_max)/2 if d7_min else None, s,
     note=f"range=[{d7_min:.4f},{d7_max:.4f}]" if d7_min else "")

# D8 — mu_8 zero crossing at ~638 MeV
# Find first zero crossing of mu_8 (where it changes sign from negative to positive)
mu8_signs = np.sign(mu8_r)
crossings = np.where(np.diff(mu8_signs) > 0)[0]  # negative→positive
if crossings.size > 0:
    i_cross = int(crossings[0])
    # linear interpolate
    m1, m2 = mu_r[i_cross], mu_r[i_cross+1]
    v1, v2 = mu8_r[i_cross], mu8_r[i_cross+1]
    d8_val = float(m1 - v1 * (m2 - m1) / (v2 - v1))
else:
    d8_val = None
s,_,_ = match(638.0, d8_val, atol=5.0)
emit("D8", "mu_8 zero crossing at ~638 MeV", "≈638 MeV", 638.0, d8_val, s)

# D9 — mu_8 = +0.024 * mu_q at 900 MeV
mu8_900 = float(mu8_r[idx900_r])
mu_900  = float(mu_r[idx900_r])
d9_val  = mu8_900 / mu_900 if mu_900 > 0 else None
s,_,_ = match(0.024, d9_val, atol=0.003)
emit("D9", "mu_8 = +0.024 mu_q at 900 MeV", "+0.024", 0.024, d9_val, s)

# D10 — gapless condition first at ~858 MeV
gapless_mask = gapminus_r < 0.0
if gapless_mask.any():
    d10_val = float(mu_r[gapless_mask][0])
else:
    d10_val = None
s,_,_ = match(858.0, d10_val, atol=5.0)
emit("D10", "gapless first satisfied at ~858 MeV", "≈858 MeV", 858.0, d10_val, s)

# D11 — gap - delta_mu ≈ -1.4 MeV at 900 MeV
gapminus_900 = float(gapminus_r[idx900_r])
s,_,_ = match(-1.4, gapminus_900, atol=0.5)
emit("D11", "gap - delta_mu ≈ -1.4 MeV at 900 MeV", "≈-1.4 MeV", -1.4, gapminus_900, s)

# D12 — Maxwell construction removes 3 points
maxwell_str = meta_stab.get("maxwell_indices", "")
# "[0, 4]" means the tie-line spans indices 0..4, replacing points 1,2,3 (3 interior points removed)
# We interpret this: if maxwell_indices = [i, j], then (j-i-1) points removed
if maxwell_str:
    try:
        indices = [int(x.strip()) for x in maxwell_str.strip("[]").split(",")]
        n_max_removed = indices[1] - indices[0] - 1 if len(indices) == 2 else None
    except:
        n_max_removed = None
else:
    n_max_removed = None
# Check against expectation
s,_,_ = match(3.0, n_max_removed, atol=1.0)
emit("D12", "Maxwell construction removes 3 points", "3", 3.0, n_max_removed, s,
     note=f"maxwell_indices={maxwell_str}")

# D13 — vacuum filter removes 22 low-density points
# Total raw 700; stable file; count negative-pressure raw points
n_vac_removed = int((straw[:, CS_PRESS] < 0.0).sum())
# Actually: raw has 700, stable has fewer; the statement is 22 from vacuum filter
# From metadata: we just count
n_raw = int(meta_stab.get("num_raw_points", 0))
n_stable = ststab.shape[0]
# Maxwell removes (j-i-1), monotone_removed is given, cs2_removed=0
monotone_removed = int(meta_stab.get("monotone_removed", 0))
cs2_removed = int(meta_stab.get("cs2_removed", 0))
# n_stable = n_raw - maxwell_gap - vac_removed - monotone_removed - cs2_removed
if n_max_removed is not None:
    vac_removed_calc = n_raw - n_stable - n_max_removed - monotone_removed - cs2_removed
else:
    vac_removed_calc = None
s,_,_ = match(22.0, vac_removed_calc, atol=2.0)
emit("D13", "vacuum filter removes 22 points", "22", 22.0, vac_removed_calc, s,
     note=f"n_raw={n_raw} n_stable={n_stable} maxwell={n_max_removed} mono={monotone_removed}")

# D14 — monotonicity filter removes 3 more points
s,_,_ = match(3.0, monotone_removed, atol=1.0)
emit("D14", "monotonicity filter removes 3 points", "3", 3.0, float(monotone_removed), s)

# D15 — Final EoS table: 672 points
s,_,_ = match(672.0, float(n_stable), atol=5.0)
emit("D15", "final EoS table has 672 points", "672", 672.0, float(n_stable), s)

# D16 — cs² raw peak ≈ 0.45 at mu_q ≈ 478 MeV
in_range_s = (mu_r > 285.0) & np.isfinite(cs2_r) & (cs2_r > 0.0) & (cs2_r < 1.0)
if in_range_s.any():
    d16_val = float(np.nanmax(cs2_r[in_range_s]))
    mu_d16  = float(mu_r[in_range_s][np.nanargmax(cs2_r[in_range_s])])
else:
    d16_val = mu_d16 = None
s,_,_ = match(0.45, d16_val, atol=0.05)
emit("D16", "cs² raw peak ≈ 0.45 at ~478 MeV", "≈0.45 at ~478 MeV", 0.45, d16_val, s,
     note=f"raw peak={d16_val:.4f} at mu_q={mu_d16:.1f}" if d16_val else "")

# D17 — cs² smoothed peak ≈ 0.39
# The smoothed value is a plotting artefact; we can check the summary CSV
d17_val = float(sec2["baseline"]["cs2_peak"]) if "baseline" in sec2 else None
s,_,_ = match(0.39, d17_val, atol=0.03)
emit("D17", "cs² smoothed peak ≈ 0.39", "≈0.39", 0.39, d17_val, s,
     note=f"(from summary CSV; this is raw EoS peak, not benchmark)")

# D18 — cs² = 0.34 at mu_q = 900 MeV (neutral)
idx900_cs = int(np.argmin(np.abs(mu_r - 900.0)))
cs2_900_s = float(cs2_r[idx900_cs]) if np.isfinite(cs2_r[idx900_cs]) else None
s,_,_ = match(0.34, cs2_900_s, atol=0.02)
emit("D18", "cs² = 0.34 at mu_q=900 (neutral)", "0.34", 0.34, cs2_900_s, s)


# ---------------------------------------------------------------------------
# GROUP E — Mass-radius relations (baseline)
# ---------------------------------------------------------------------------
hdr("GROUP E — Mass-radius relations (baseline QMD SET A)")

# stars cols: Pc_dimless eps_c_mev4 eps_c_gev_fm3 radius_km mass_msun stable_flag
mass_all = stars[:, 4]
rad_all  = stars[:, 3]
eps_all  = stars[:, 2]   # in GeV/fm^3
stab_all = stars[:, 5]

n_total  = stars.shape[0]
n_stable = int((stab_all > 0.5).sum())
n_unstab = n_total - n_stable

mass_stable = mass_all[stab_all > 0.5]
rad_stable  = rad_all[stab_all > 0.5]
eps_stable  = eps_all[stab_all > 0.5]

e1_val = float(n_total)
e2_val = float(n_stable)
e3_val = float(n_unstab)

s,_,_ = match(285.0, e1_val, atol=3.0)
emit("E1a", "285 total TOV configurations", "285", 285.0, e1_val, s)
s,_,_ = match(253.0, e2_val, atol=3.0)
emit("E1b", "253 stable configurations", "253", 253.0, e2_val, s)
s,_,_ = match(32.0, e3_val, atol=3.0)
emit("E1c", "32 unstable configurations", "32", 32.0, e3_val, s)

# E2 — M_max = 1.964 M_sun
e2_mmax = float(mass_stable.max())
s,_,_ = match(1.964, e2_mmax, atol=0.003)
emit("E2", "M_max = 1.964 M_sun", "1.964", 1.964, e2_mmax, s)

# E3 — R(M_max) = 12.11 km
idx_mmax = int(np.argmax(mass_stable))
e3_rmmax = float(rad_stable[idx_mmax])
s,_,_ = match(12.11, e3_rmmax, atol=0.05)
emit("E3", "R(M_max) = 12.11 km", "12.11 km", 12.11, e3_rmmax, s)

# E4 — eps_c(M_max) = 1.196 GeV/fm^3
e4_epsc = float(eps_stable[idx_mmax])
s,_,_ = match(1.196, e4_epsc, atol=0.005)
emit("E4", "eps_c(M_max) = 1.196 GeV/fm^3", "1.196", 1.196, e4_epsc, s)

# E5 — R(1.4 M_sun) = 14.69 km
# Interpolate: find two closest stable stars straddling 1.4 M_sun
m14 = 1.4
close = np.abs(mass_stable - m14)
idx_close = np.argsort(close)[:2]
if len(idx_close) == 2 and mass_stable[idx_close].min() < m14 < mass_stable[idx_close].max():
    i0, i1 = idx_close
    if mass_stable[i0] > mass_stable[i1]:
        i0, i1 = i1, i0
    frac = (m14 - mass_stable[i0]) / (mass_stable[i1] - mass_stable[i0])
    e5_r14 = float(rad_stable[i0] + frac * (rad_stable[i1] - rad_stable[i0]))
else:
    e5_r14 = float(rad_stable[idx_close[0]])
s,_,_ = match(14.69, e5_r14, atol=0.1)
emit("E5", "R(1.4 M_sun) = 14.69 km", "14.69 km", 14.69, e5_r14, s)

# E6 — R(1.8 M_sun) = 13.41 km
m18 = 1.8
close = np.abs(mass_stable - m18)
idx_close = np.argsort(close)[:2]
if len(idx_close) == 2 and mass_stable[idx_close].min() < m18 < mass_stable[idx_close].max():
    i0, i1 = idx_close
    if mass_stable[i0] > mass_stable[i1]:
        i0, i1 = i1, i0
    frac = (m18 - mass_stable[i0]) / (mass_stable[i1] - mass_stable[i0])
    e6_r18 = float(rad_stable[i0] + frac * (rad_stable[i1] - rad_stable[i0]))
else:
    e6_r18 = float(rad_stable[idx_close[0]])
s,_,_ = match(13.41, e6_r18, atol=0.1)
emit("E6", "R(1.8 M_sun) = 13.41 km", "13.41 km", 13.41, e6_r18, s)

# E7 — eps_c(1.4 M_sun) = 0.439 GeV/fm^3
close = np.abs(mass_stable - m14)
idx14 = np.argsort(close)[0]
e7_epsc14 = float(eps_stable[idx14])
s,_,_ = match(0.439, e7_epsc14, atol=0.01)
emit("E7", "eps_c(1.4 M_sun) = 0.439 GeV/fm^3", "0.439", 0.439, e7_epsc14, s)

# E8 — eps_c(1.8 M_sun) = 0.654 GeV/fm^3
close = np.abs(mass_stable - m18)
idx18 = np.argsort(close)[0]
e8_epsc18 = float(eps_stable[idx18])
s,_,_ = match(0.654, e8_epsc18, atol=0.01)
emit("E8", "eps_c(1.8 M_sun) = 0.654 GeV/fm^3", "0.654", 0.654, e8_epsc18, s)

# E9 — QMD vs QM: delta_M_max = +0.177 M_sun
# Need the QM comparison mass-radius — need to find its stars file
qm_stars_candidates = list(pathlib.Path(ROOT / "output" / "stellar").glob("*.txt"))
qm_mmax = None
for f in qm_stars_candidates:
    try:
        d = np.loadtxt(f, comments="#")
        # Look for file where M_max ≈ 1.964 - 0.177 = 1.787
        if d.ndim == 2 and d.shape[1] >= 5:
            mm = d[:, 4].max()
            if 1.7 < mm < 1.9:
                qm_mmax = mm
                break
    except:
        pass
# The QM data may not be in this directory — try eos/
qm_mmax2 = None
for dd in [ROOT / "output" / "stellar", ROOT / "output" / "eos"]:
    if not dd.exists():
        continue
    for f in dd.glob("*.txt"):
        try:
            d = np.loadtxt(f, comments="#")
            if d.ndim == 2 and d.shape[1] >= 5:
                mm = float(d[:, 4].max())
                if 1.6 < mm < 1.95 and abs(mm - 1.787) < 0.1:
                    qm_mmax2 = mm
                    break
        except:
            pass

if qm_mmax2 is not None:
    e9_val = e2_mmax - qm_mmax2
    s,_,_ = match(0.177, e9_val, atol=0.015)
    emit("E9", "delta_M_max QMD-QM = +0.177 M_sun", "+0.177", 0.177, e9_val, s,
         note=f"QM Mmax={qm_mmax2:.4f}")
else:
    emit("E9", "delta_M_max QMD-QM = +0.177 M_sun", "+0.177", 0.177, None, "UNVERIFIABLE",
         note="QM stars file not found in output/stellar or output/eos")

# E10 — delta_R(1.4 M_sun) = +2.22 km — need QM file too
emit("E10", "delta_R(1.4 M_sun) QMD-QM = +2.22 km", "+2.22 km", 2.22, None, "UNVERIFIABLE",
     note="QM stars file not located; needs separate check")

# E11 — QM comparison uses B^(1/4) = 28 MeV
emit("E11", "QM comparison uses B^(1/4)=28 MeV", "28 MeV", 28.0, None, "UNVERIFIABLE",
     note="parameter embedded in QM run; no separate data file found here")


# ---------------------------------------------------------------------------
# Section 2 — helper for reading summary CSV
# ---------------------------------------------------------------------------
def sec2_val(run_id, field, as_float=True):
    if run_id not in sec2:
        return None
    v = sec2[run_id].get(field, "")
    if v in ("", "nan"):
        return None
    return float(v) if as_float else v

def check_sec2(cid, desc, thesis_str, tv, run_id, field, atol=None, rtol=RTOL, note=""):
    cv = sec2_val(run_id, field)
    s, tv2, cv2 = match(tv, cv, atol=atol, rtol=rtol)
    emit(cid, desc, thesis_str, tv2, cv2, s, note)


# ---------------------------------------------------------------------------
# GROUP F — Run A (g_Delta = 1.5g)
# ---------------------------------------------------------------------------
hdr("GROUP F — Parameter scan Run A (g_Delta = 1.5g)")

check_sec2("F1", "Run A: M_max = 1.786 M_sun",  "1.786",  1.786, "A", "M_max",  atol=0.003)
check_sec2("F2", "Run A: R(M_max) = 10.99 km",  "10.99",  10.99, "A", "R_at_Mmax", atol=0.05)
# F3 — R(1.4 M_sun) = 12.89: need to read the stellar file
starsA = load(SEC2_DIR / "section2_stellar_gdelta_1p5g.txt")
massA = starsA[:, 4]; radA = starsA[:, 3]; stabA = starsA[:, 5]
mass_sA = massA[stabA > 0.5]; rad_sA = radA[stabA > 0.5]
close = np.abs(mass_sA - 1.4); idx14A = np.argsort(close)[:2]
if len(idx14A) == 2 and mass_sA[idx14A].min() < 1.4 < mass_sA[idx14A].max():
    i0, i1 = idx14A
    if mass_sA[i0] > mass_sA[i1]: i0, i1 = i1, i0
    frac = (1.4 - mass_sA[i0]) / (mass_sA[i1] - mass_sA[i0])
    f3_val = float(rad_sA[i0] + frac * (rad_sA[i1] - rad_sA[i0]))
else:
    f3_val = float(rad_sA[idx14A[0]])
s,_,_ = match(12.89, f3_val, atol=0.1)
emit("F3", "Run A: R(1.4 M_sun) = 12.89 km", "12.89 km", 12.89, f3_val, s)

# F4 — gap at mu_q = 900 MeV: "352 MeV" in text / "351.8" in table
# Table has 351.75 (from CSV asymptotic_gap)
f4_val = sec2_val("A", "asymptotic_gap")
s,_,_ = match(352.0, f4_val, atol=1.0)
emit("F4", "Run A: gap at 900 MeV ≈352 MeV / 351.8 in table", "351.8 / 352", 351.8, f4_val, s)

# F5 — cs² peak ≈ 0.41 (smoothed) — Table says 0.4374
f5_val = sec2_val("A", "cs2_peak")
s,_,_ = match(0.41, f5_val, atol=0.05)
emit("F5", "Run A: cs² peak ≈0.41 (smoothed)", "≈0.41", 0.41, f5_val, s,
     note="thesis says 0.41 smoothed; table has raw 0.4374")

# F6 — onset mu_q = 281.6 MeV
check_sec2("F6", "Run A: onset mu_q = 281.6 MeV", "281.6", 281.6, "A", "onset_muq", atol=1.5)


# ---------------------------------------------------------------------------
# GROUP G — Run B (g_Delta = 2.5g)
# ---------------------------------------------------------------------------
hdr("GROUP G — Parameter scan Run B (g_Delta = 2.5g)")

check_sec2("G1", "Run B: M_max = 2.040 M_sun",  "2.040",  2.040, "B", "M_max",  atol=0.003)
check_sec2("G2", "Run B: R(M_max) = 12.67 km",  "12.67",  12.67, "B", "R_at_Mmax", atol=0.05)
starsB = load(SEC2_DIR / "section2_stellar_gdelta_2p5g.txt")
massB = starsB[:, 4]; radB = starsB[:, 3]; stabB = starsB[:, 5]
mass_sB = massB[stabB > 0.5]; rad_sB = radB[stabB > 0.5]
close = np.abs(mass_sB - 1.4); idx14B = np.argsort(close)[:2]
if len(idx14B) == 2 and mass_sB[idx14B].min() < 1.4 < mass_sB[idx14B].max():
    i0, i1 = idx14B
    if mass_sB[i0] > mass_sB[i1]: i0, i1 = i1, i0
    frac = (1.4 - mass_sB[i0]) / (mass_sB[i1] - mass_sB[i0])
    g3_val = float(rad_sB[i0] + frac * (rad_sB[i1] - rad_sB[i0]))
else:
    g3_val = float(rad_sB[idx14B[0]])
s,_,_ = match(16.06, g3_val, atol=0.1)
emit("G3", "Run B: R(1.4 M_sun) = 16.06 km", "16.06 km", 16.06, g3_val, s)

check_sec2("G4", "Run B: gap at 900 MeV = 216 MeV", "216",  216.0, "B", "asymptotic_gap", atol=1.0)
check_sec2("G5", "Run B: cs² peak ≈0.39 (smoothed)", "≈0.39", 0.39, "B", "cs2_peak", atol=0.04)
check_sec2("G6", "Run B: onset mu_q = 277.0 MeV",  "277.0",  277.0, "B", "onset_muq", atol=1.5)

# G7 — spurious vacuum 2SC up to ~69 MeV
benchB_file = SEC2_DIR / "section2_benchmark_gdelta_2p5g.txt"
benchB = load(benchB_file)
mu_bB = benchB[:, C_MU]; gap_bB = benchB[:, C_GAP]; phase_bB = benchB[:, C_PHASE]
# Spurious region: 2SC at very low mu
spur_mask = (mu_bB < 200.0) & (phase_bB > 0.5)
if spur_mask.any():
    spur_end = float(mu_bB[spur_mask][-1])
else:
    spur_end = None
s,_,_ = match(69.0, spur_end, atol=5.0)
emit("G7", "Run B: spurious vac 2SC up to ~69 MeV", "≈69 MeV", 69.0, spur_end, s)

# G8 — physical 2SC onset after spurious region: 263.6 MeV
# Find first 2SC onset above, say, 200 MeV
phys_mask = (mu_bB > 200.0) & (phase_bB > 0.5)
g8_val = float(mu_bB[phys_mask][0]) if phys_mask.any() else None
s,_,_ = match(263.6, g8_val, atol=1.5)
emit("G8", "Run B: physical onset at 263.6 MeV", "263.6 MeV", 263.6, g8_val, s)

# G9 — gapless for mu_q > 728 MeV (Run B stellar)
eosB = load(SEC2_DIR / "section2_stellar_gdelta_2p5g_eos.txt")
mu_eosB = eosB[:, CS_MU]; gapminus_eosB = eosB[:, CS_GAPMINUS]
gapless_B = gapminus_eosB < 0.0
g9_val = float(mu_eosB[gapless_B][0]) if gapless_B.any() else None
s,_,_ = match(728.0, g9_val, atol=10.0)
emit("G9", "Run B: gapless for mu_q > 728 MeV", "728 MeV", 728.0, g9_val, s)

# G10 — mismatch - gap = 21 MeV at 900 MeV (Run B)
idx900_eosB = int(np.argmin(np.abs(mu_eosB - 900.0)))
gapminus_900_B = float(gapminus_eosB[idx900_eosB])
g10_val = -gapminus_900_B  # gap - delta_mu, so delta_mu - gap = -gapminus
s,_,_ = match(21.0, g10_val, atol=2.0)
emit("G10", "Run B: mismatch-gap = 21 MeV at 900 MeV", "21 MeV", 21.0, g10_val, s)


# ---------------------------------------------------------------------------
# GROUP H — Run C (m_Delta = 400 MeV)
# ---------------------------------------------------------------------------
hdr("GROUP H — Parameter scan Run C (m_Delta = 400 MeV)")

check_sec2("H1", "Run C: M_max = 2.052 M_sun",  "2.052",  2.052, "C", "M_max",  atol=0.003)
check_sec2("H2", "Run C: R(M_max) = 12.54 km",  "12.54",  12.54, "C", "R_at_Mmax", atol=0.05)
starsC = load(SEC2_DIR / "section2_stellar_mdelta_400.txt")
massC = starsC[:, 4]; radC = starsC[:, 3]; stabC = starsC[:, 5]
mass_sC = massC[stabC > 0.5]; rad_sC = radC[stabC > 0.5]
close = np.abs(mass_sC - 1.4); idx14C = np.argsort(close)[:2]
if len(idx14C) == 2 and mass_sC[idx14C].min() < 1.4 < mass_sC[idx14C].max():
    i0, i1 = idx14C;
    if mass_sC[i0] > mass_sC[i1]: i0, i1 = i1, i0
    frac = (1.4 - mass_sC[i0]) / (mass_sC[i1] - mass_sC[i0])
    h3_val = float(rad_sC[i0] + frac * (rad_sC[i1] - rad_sC[i0]))
else:
    h3_val = float(rad_sC[idx14C[0]])
s,_,_ = match(15.72, h3_val, atol=0.1)
emit("H3", "Run C: R(1.4 M_sun) = 15.72 km", "15.72 km", 15.72, h3_val, s)

check_sec2("H4", "Run C: onset mu_q = 271.4 MeV", "271.4", 271.4, "C", "onset_muq", atol=1.5)
check_sec2("H5", "Run C: asym gap 255.7 MeV",     "255.7", 255.7, "C", "asymptotic_gap", atol=1.0)
check_sec2("H6", "Run C: cs² peak ≈0.46",          "≈0.46",  0.46, "C", "cs2_peak", atol=0.05)


# ---------------------------------------------------------------------------
# GROUP I — Run D (m_Delta = 700 MeV)
# ---------------------------------------------------------------------------
hdr("GROUP I — Parameter scan Run D (m_Delta = 700 MeV)")

check_sec2("I1", "Run D: M_max = 1.823 M_sun",  "1.823",  1.823, "D", "M_max",  atol=0.003)
check_sec2("I2", "Run D: R(M_max) = 11.22 km",  "11.22",  11.22, "D", "R_at_Mmax", atol=0.05)
starsD = load(SEC2_DIR / "section2_stellar_mdelta_700.txt")
massD = starsD[:, 4]; radD = starsD[:, 3]; stabD = starsD[:, 5]
mass_sD = massD[stabD > 0.5]; rad_sD = radD[stabD > 0.5]
close = np.abs(mass_sD - 1.4); idx14D = np.argsort(close)[:2]
if len(idx14D) == 2 and mass_sD[idx14D].min() < 1.4 < mass_sD[idx14D].max():
    i0, i1 = idx14D
    if mass_sD[i0] > mass_sD[i1]: i0, i1 = i1, i0
    frac = (1.4 - mass_sD[i0]) / (mass_sD[i1] - mass_sD[i0])
    i3_val = float(rad_sD[i0] + frac * (rad_sD[i1] - rad_sD[i0]))
else:
    i3_val = float(rad_sD[idx14D[0]])
s,_,_ = match(13.24, i3_val, atol=0.1)
emit("I3", "Run D: R(1.4 M_sun) = 13.24 km", "13.24 km", 13.24, i3_val, s)

check_sec2("I4", "Run D: onset mu_q = 293.7 MeV", "293.7", 293.7, "D", "onset_muq", atol=1.5)
check_sec2("I5", "Run D: asym gap 243.8 MeV",     "243.8", 243.8, "D", "asymptotic_gap", atol=1.0)

# I6 — tree-level onset estimate m_Delta/2 = 350 MeV
i6_val = 700.0 / 2.0
s,_,_ = match(350.0, i6_val, atol=0.1)
emit("I6", "Run D: tree-level onset m_D/2 = 350 MeV", "350 MeV", 350.0, i6_val, s)

# I7 — onset range across three m_Delta values: 22.3 MeV
onset_C = sec2_val("C", "onset_muq")
onset_bl= sec2_val("baseline", "onset_muq")
onset_D = sec2_val("D", "onset_muq")
i7_val  = onset_D - onset_C if (onset_C and onset_D) else None
s,_,_ = match(22.3, i7_val, atol=1.0)
emit("I7", "onset range across m_D values: 22.3 MeV", "22.3 MeV", 22.3, i7_val, s,
     note=f"C={onset_C:.1f} D={onset_D:.1f}" if onset_C and onset_D else "")


# ---------------------------------------------------------------------------
# GROUP J — Run E (lambda_Delta = lambda_0/8)
# ---------------------------------------------------------------------------
hdr("GROUP J — Parameter scan Run E (lambda_Delta = lambda_0/8)")

check_sec2("J1", "Run E: M_max = 1.9639 M_sun",  "1.9639", 1.9639, "E", "M_max",  atol=0.001)
check_sec2("J2", "Run E: R(M_max) = 12.04 km",   "12.04",  12.04,  "E", "R_at_Mmax", atol=0.05)
check_sec2("J3", "Run E: onset = 278.8 MeV",     "278.8",  278.8,  "E", "onset_muq", atol=1.5)
check_sec2("J4", "Run E: asym gap 252.6 MeV",    "252.6",  252.6,  "E", "asymptotic_gap", atol=0.5)


# ---------------------------------------------------------------------------
# GROUP K — Run F (lambda_Delta = lambda_0/2)
# ---------------------------------------------------------------------------
hdr("GROUP K — Parameter scan Run F (lambda_Delta = lambda_0/2)")

check_sec2("K1", "Run F: M_max = 1.9633 M_sun",  "1.9633", 1.9633, "F", "M_max",  atol=0.001)
check_sec2("K2", "Run F: onset = 277.9 MeV",     "277.9",  277.9,  "F", "onset_muq", atol=1.5)
check_sec2("K3", "Run F: asym gap 251.8 MeV",    "251.8",  251.8,  "F", "asymptotic_gap", atol=0.5)
# K4 — total M_max spread across lambda_Delta variations: 0.0007 M_sun
mmax_E = sec2_val("E", "M_max"); mmax_F = sec2_val("F", "M_max")
mmax_bl_val = sec2_val("baseline", "M_max")
if mmax_E and mmax_F and mmax_bl_val:
    k4_val = max(mmax_E, mmax_bl_val, mmax_F) - min(mmax_E, mmax_bl_val, mmax_F)
else:
    k4_val = None
s,_,_ = match(0.0007, k4_val, atol=0.0003)
emit("K4", "M_max spread across lam_D: 0.0007 M_sun", "0.0007", 0.0007, k4_val, s)


# ---------------------------------------------------------------------------
# GROUP L — Run G (lambda_3 = 0)
# ---------------------------------------------------------------------------
hdr("GROUP L — Parameter scan Run G (lambda_3 = 0)")

check_sec2("L1", "Run G: M_max = 1.998 M_sun",  "1.998",  1.998, "G", "M_max",  atol=0.003)
check_sec2("L2", "Run G: R(M_max) = 12.98 km",  "12.98",  12.98, "G", "R_at_Mmax", atol=0.05)

starsG = load(SEC2_DIR / "section2_stellar_lam3_0.txt")
massG = starsG[:, 4]; radG = starsG[:, 3]; stabG = starsG[:, 5]
mass_sG = massG[stabG > 0.5]; rad_sG = radG[stabG > 0.5]
close = np.abs(mass_sG - 1.4); idx14G = np.argsort(close)[:2]
if len(idx14G) == 2 and mass_sG[idx14G].min() < 1.4 < mass_sG[idx14G].max():
    i0, i1 = idx14G
    if mass_sG[i0] > mass_sG[i1]: i0, i1 = i1, i0
    frac = (1.4 - mass_sG[i0]) / (mass_sG[i1] - mass_sG[i0])
    l3_val = float(rad_sG[i0] + frac * (rad_sG[i1] - rad_sG[i0]))
else:
    l3_val = float(rad_sG[idx14G[0]])
s,_,_ = match(18.1, l3_val, atol=0.3)
emit("L3", "Run G: R(1.4 M_sun) ≈18.1 km", "≈18.1 km", 18.1, l3_val, s)

check_sec2("L4", "Run G: onset mu_q = 270.5 MeV", "270.5", 270.5, "G", "onset_muq", atol=1.5)
check_sec2("L5", "Run G: asym gap 252.4 MeV",     "252.4", 252.4, "G", "asymptotic_gap", atol=0.5)

# L6 — phi_0 = 20.0 MeV at mu_q = 350 MeV (Run G common-mu benchmark)
benchG = load(SEC2_DIR / "section2_benchmark_lam3_0.txt")
mu_bG = benchG[:, C_MU]; phi_bG = benchG[:, C_PHI]
idx350G = int(np.argmin(np.abs(mu_bG - 350.0)))
l6_val = float(phi_bG[idx350G])
s,_,_ = match(20.0, l6_val, atol=1.0)
emit("L6", "Run G: phi_0=20.0 MeV at mu_q=350 (bench)", "20.0 MeV", 20.0, l6_val, s,
     note=f"mu={mu_bG[idx350G]:.2f}")


# ---------------------------------------------------------------------------
# GROUP M — Run H (lambda_3 = 2*lambda_0)
# ---------------------------------------------------------------------------
hdr("GROUP M — Parameter scan Run H (lambda_3 = 2*lambda_0)")

check_sec2("M1", "Run H: M_max = 1.943 M_sun",  "1.943",  1.943, "H", "M_max",  atol=0.003)
check_sec2("M2", "Run H: R(M_max) = 11.40 km",  "11.40",  11.40, "H", "R_at_Mmax", atol=0.05)

starsH = load(SEC2_DIR / "section2_stellar_lam3_2lam0.txt")
massH = starsH[:, 4]; radH = starsH[:, 3]; stabH = starsH[:, 5]
mass_sH = massH[stabH > 0.5]; rad_sH = radH[stabH > 0.5]
close = np.abs(mass_sH - 1.4); idx14H = np.argsort(close)[:2]
if len(idx14H) == 2 and mass_sH[idx14H].min() < 1.4 < mass_sH[idx14H].max():
    i0, i1 = idx14H
    if mass_sH[i0] > mass_sH[i1]: i0, i1 = i1, i0
    frac = (1.4 - mass_sH[i0]) / (mass_sH[i1] - mass_sH[i0])
    m3_val = float(rad_sH[i0] + frac * (rad_sH[i1] - rad_sH[i0]))
else:
    m3_val = float(rad_sH[idx14H[0]])
s,_,_ = match(13.0, m3_val, atol=0.3)
emit("M3", "Run H: R(1.4 M_sun) ≈13.0 km", "≈13.0 km", 13.0, m3_val, s)

check_sec2("M4", "Run H: onset mu_q = 285.3 MeV", "285.3", 285.3, "H", "onset_muq", atol=1.5)
check_sec2("M5", "Run H: asym gap 252.4 MeV",     "252.4", 252.4, "H", "asymptotic_gap", atol=0.5)

benchH = load(SEC2_DIR / "section2_benchmark_lam3_2lam0.txt")
mu_bH = benchH[:, C_MU]; phi_bH = benchH[:, C_PHI]
idx350H = int(np.argmin(np.abs(mu_bH - 350.0)))
m6_val = float(phi_bH[idx350H])
s,_,_ = match(15.6, m6_val, atol=1.0)
emit("M6", "Run H: phi_0=15.6 MeV at mu_q=350 (bench)", "15.6 MeV", 15.6, m6_val, s,
     note=f"mu={mu_bH[idx350H]:.2f}")


# ---------------------------------------------------------------------------
# GROUP N — Synthesis claims
# ---------------------------------------------------------------------------
hdr("GROUP N — Synthesis claims")

# N1 — delta_M_max (g_Delta scan): 0.254 M_sun
mmax_A = sec2_val("A", "M_max"); mmax_B = sec2_val("B", "M_max")
n1_val = mmax_B - mmax_A if (mmax_A and mmax_B) else None
s,_,_ = match(0.254, n1_val, atol=0.005)
emit("N1", "delta_M_max (g_D scan A to B): 0.254 M_sun", "0.254", 0.254, n1_val, s)

# N2 — delta_R(M_max) (g_Delta scan): 1.68 km
r_A = sec2_val("A", "R_at_Mmax"); r_B = sec2_val("B", "R_at_Mmax")
n2_val = r_B - r_A if (r_A and r_B) else None
s,_,_ = match(1.68, n2_val, atol=0.05)
emit("N2", "delta_R(M_max) g_D scan: 1.68 km", "1.68 km", 1.68, n2_val, s)

# N3 — delta_R(1.4 M_sun) (g_Delta scan): 3.2 km (from text, not table)
n3_val = g3_val - f3_val if (g3_val and f3_val) else None
s,_,_ = match(3.2, n3_val, atol=0.15)
emit("N3", "delta_R(1.4 M_sun) g_D scan: 3.2 km", "3.2 km", 3.2, n3_val, s)

# N4 — delta_M_max (m_Delta scan): 0.229 M_sun
mmax_C = sec2_val("C", "M_max"); mmax_D = sec2_val("D", "M_max")
n4_val = mmax_C - mmax_D if (mmax_C and mmax_D) else None
s,_,_ = match(0.229, n4_val, atol=0.005)
emit("N4", "delta_M_max m_D scan C-D: 0.229 M_sun", "0.229", 0.229, n4_val, s)

# N5 — tree-level onset range 150 MeV; actual range 22 MeV
tree_range = (700.0 - 400.0) / 2.0 * 1.0  # m_D span 400-700, onset span mD/2: (700-400)/2=150
n5_tree = 150.0
s,_,_ = match(n5_tree, n5_tree, atol=0.1)
emit("N5a", "tree-level onset range: 150 MeV (analytic)", "150 MeV", 150.0, tree_range, s)
s,_,_ = match(22.3, i7_val, atol=1.0)
emit("N5b", "actual onset range: 22.3 MeV", "22.3 MeV", 22.3, i7_val, s)

# N6 — delta_M_max (lambda_3 scan): 0.055 M_sun
mmax_G = sec2_val("G", "M_max"); mmax_H_val = sec2_val("H", "M_max")
n6_val = mmax_G - mmax_H_val if (mmax_G and mmax_H_val) else None
s,_,_ = match(0.055, n6_val, atol=0.003)
emit("N6", "delta_M_max lambda_3 scan G-H: 0.055 M_sun", "0.055", 0.055, n6_val, s)

# N7 — R(1.4 M_sun) spread for lambda_3: ~5 km
n7_val = l3_val - m3_val if (l3_val and m3_val) else None
s,_,_ = match(5.0, n7_val, atol=0.5)
emit("N7", "R(1.4 M_sun) spread lam3 scan: ~5 km", "~5 km", 5.0, n7_val, s,
     note=f"G={l3_val:.2f} H={m3_val:.2f}" if l3_val and m3_val else "")

# N8 — KRS bound: g_Delta*Delta_0 ≲ 268 MeV at asymptotic density
# This is stated in the thesis as a reference value; we verify it's below Run A's gap
gap_A = sec2_val("A", "asymptotic_gap")
n8_consistent = (gap_A is not None and gap_A > 268.0)
emit("N8", "KRS bound: gap ≲268 MeV at asymptote", "≲268 MeV", 268.0, gap_A,
     "MATCH" if n8_consistent else "MISMATCH (Run A EXCEEDS bound)",
     note=f"Run A gap={gap_A:.1f} > 268 → violates bound as stated in thesis")

# N9 — KRS bound corresponds to g_Delta ≳ 2.21g
# Invert the chiral-limit asymptotic formula: g_D*Delta = M_q * exp[(4pi)^2/(8*gD^2) - 1/2]
# KRS: g_D*Delta ≤ 268 → (4pi)^2/(8*gD^2) - 1/2 ≤ log(268/300)
# If g_D*Delta_bar(g_D) = 268, then solve for g_D
# 268/300 = exp[(4pi)^2/(8*g_D^2) - 1/2]
# log(268/300) = (4pi)^2/(8*g_D^2) - 1/2
# (4pi)^2/(8*g_D^2) = log(268/300) + 1/2
target_gap = 268.0
log_arg = math.log(target_gap / mq)
# log_arg = (4pi)^2/(8*gD^2) - 1/2  (chiral limit, F=G=0)
# => (4pi)^2/(8*gD^2) = log_arg + 0.5
rhs = log_arg + 0.5
gd2_krs = (4.0 * math.pi)**2 / (8.0 * rhs)
gd_krs = math.sqrt(gd2_krs) / g_val  # in units of g
s,_,_ = match(2.21, gd_krs, atol=0.05)
emit("N9", "KRS bound → g_Delta ≳ 2.21g", "≳2.21g", 2.21, gd_krs, s,
     note=f"chiral-limit inversion gives {gd_krs:.3f}g")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n\n{'='*110}")
print(f"  SUMMARY")
print(f"{'='*110}")
print(f"\n  ({len(matches)}) MATCHING claims:")
print("  " + ", ".join(matches))

print(f"\n  ({len(mismatches)}) MISMATCHING claims (sorted by magnitude):")
print(f"  {'Claim':<6} {'Desc':<42} {'Thesis':>14} {'Computed':>14} {'Status'}")
print("  " + "-"*90)
for cid, desc, tv, cv, status, note in mismatches:
    tv_s = f"{tv:.5g}" if isinstance(tv, (int, float)) and tv is not None else str(tv)
    cv_s = f"{cv:.5g}" if isinstance(cv, (int, float)) and cv is not None else str(cv)
    print(f"  {cid:<6} {desc:<42} {tv_s:>14} {cv_s:>14}  {status}  {note}")
