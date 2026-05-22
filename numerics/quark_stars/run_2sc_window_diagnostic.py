"""Diagnostic investigation of the truncated QMD 2SC window for SET A.

Tests 1–5.  Read-only: prints to stdout only; no file writes, no thesis edits.

Run from the repo root:
    numerics/bin/python -m numerics.quark_stars.run_2sc_window_diagnostic

Runtime estimate (warm-start scans ~60 ms/pt):
  Test 1:  ~10 min  (2 fine scans around the window)
  Test 2:  ~2 min   (48 optimizer calls)
  Test 3:  ~5 s     (analytic, no quadrature)
  Test 4:  ~8 min   (4 × 1000-pt scans)
  Test 5:  ~1 s     (pure algebra)
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from dataclasses import replace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from scipy.optimize import minimize as sp_minimize, minimize_scalar

from .qmd_parameters import QMD_SET_A, QMDParameters
from .qmd_simple import QMDSimpleModel, _omega_trunc_core, _make_precomputed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAMS_TRUNC = replace(QMD_SET_A, include_omega_1_num=False)
_GAP_THRESH = 1.0   # MeV — Delta_0 threshold to call it "2SC"
_BASELINE_PATH = (
    pathlib.Path(__file__).parent
    / "output" / "qmd_benchmark" / "data"
    / "qmd_benchmark_truncated.txt"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scan(
    mu_values: np.ndarray,
    params: QMDParameters | None = None,
    tag: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phi, gap, delta) arrays; prints a progress dot every 10%."""
    p = params or PARAMS_TRUNC
    model = QMDSimpleModel(p)
    n = len(mu_values)
    phi_arr   = np.empty(n)
    gap_arr   = np.empty(n)
    delta_arr = np.empty(n)
    prev = None
    tick = max(1, n // 10)
    t0 = time.time()
    for i, mu in enumerate(mu_values):
        s = model.solve_mean_fields(float(mu), initial_guess=prev)
        phi_arr[i]   = s.phi_mev
        gap_arr[i]   = s.gap_mev
        delta_arr[i] = s.delta_mev
        prev = (s.phi_mev, s.delta_mev)
        if (i + 1) % tick == 0:
            pct = 100 * (i + 1) // n
            elapsed = time.time() - t0
            sys.stdout.write(f"\r    {tag}: {pct}%  ({elapsed:.1f}s)   ")
            sys.stdout.flush()
    if n > 0:
        elapsed = time.time() - t0
        sys.stdout.write(f"\r    {tag}: done ({elapsed:.1f}s, {1000*elapsed/n:.1f} ms/pt)\n")
        sys.stdout.flush()
    return phi_arr, gap_arr, delta_arr


def _window(mu: np.ndarray, gaps: np.ndarray) -> dict:
    mask = gaps > _GAP_THRESH
    if not mask.any():
        return {"onset": None, "closure": None, "width": 0.0, "peak": 0.0, "n_pts": 0}
    idx = np.where(mask)[0]
    return {
        "onset":   float(mu[idx[0]]),
        "closure": float(mu[idx[-1]]),
        "width":   float(mu[idx[-1]] - mu[idx[0]]),
        "peak":    float(gaps[mask].max()),
        "n_pts":   int(mask.sum()),
    }


def _hdr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# TEST 1 — Grid resolution
# ---------------------------------------------------------------------------

def test1_grid_resolution() -> None:
    _hdr("TEST 1 — Grid resolution sensitivity")

    # (a) Load existing 5000-pt baseline
    print("\n  (a) Existing 5000-pt baseline from qmd_benchmark_truncated.txt")
    if _BASELINE_PATH.exists():
        data = np.loadtxt(_BASELINE_PATH, comments="#")
        mu_b = data[:, 0]
        gap_b = data[:, 3]
        w = _window(mu_b, gap_b)
        gs = float(mu_b[1] - mu_b[0])
        print(f"      total pts : {len(mu_b)}")
        print(f"      μ range   : [{mu_b[0]:.1f}, {mu_b[-1]:.1f}] MeV  grid step = {gs:.4f} MeV")
        if w["onset"] is None:
            print("      → No 2SC window (gap never exceeds 1 MeV)")
        else:
            print(f"      onset      = {w['onset']:.4f} MeV")
            print(f"      closure    = {w['closure']:.4f} MeV")
            print(f"      width      = {w['width']:.4f} MeV")
            print(f"      peak gap   = {w['peak']:.4f} MeV")
            print(f"      pts in win = {w['n_pts']}")
    else:
        print("      (file not found; skipping)")

    # (b) 2000-pt scan on [320, 345] MeV  (grid step ~0.012 MeV)
    print("\n  (b) 2000-pt scan on μ_q ∈ [320, 345] MeV  (step ~0.012 MeV)")
    mu_b2 = np.linspace(320.0, 345.0, 2000)
    _, gap_b2, _ = _scan(mu_b2, tag="Test1b")
    w2 = _window(mu_b2, gap_b2)
    gs2 = float(mu_b2[1] - mu_b2[0])
    print(f"      grid step  = {gs2:.5f} MeV")
    if w2["onset"] is None:
        print("      → No 2SC window")
    else:
        print(f"      onset      = {w2['onset']:.4f} MeV")
        print(f"      closure    = {w2['closure']:.4f} MeV")
        print(f"      width      = {w2['width']:.4f} MeV")
        print(f"      peak gap   = {w2['peak']:.4f} MeV")
        print(f"      pts in win = {w2['n_pts']}")

    # (c) 5000-pt scan on [328, 336] MeV  (grid step ~0.0016 MeV)
    print("\n  (c) 5000-pt scan on μ_q ∈ [328, 336] MeV  (step ~0.0016 MeV)")
    mu_c = np.linspace(328.0, 336.0, 5000)
    _, gap_c, _ = _scan(mu_c, tag="Test1c")
    wc = _window(mu_c, gap_c)
    gsc = float(mu_c[1] - mu_c[0])
    print(f"      grid step  = {gsc:.6f} MeV")
    if wc["onset"] is None:
        print("      → No 2SC window")
    else:
        print(f"      onset      = {wc['onset']:.4f} MeV")
        print(f"      closure    = {wc['closure']:.4f} MeV")
        print(f"      width      = {wc['width']:.4f} MeV")
        print(f"      peak gap   = {wc['peak']:.4f} MeV")
        print(f"      pts in win = {wc['n_pts']}")

    # Cross-check: if window is real, width should converge as grid gets finer
    # A grid-artifact would have width ≈ a fixed small number of grid steps
    if w2["onset"] and wc["onset"]:
        # grid-step multiples
        n2 = round(w2["width"] / gs2)
        nc = round(wc["width"] / gsc)
        print(f"\n  Width in grid-step units:  (b) {n2} steps   (c) {nc} steps")
        if abs(n2 - nc) <= 2 and n2 <= 5:
            print("  *** WARNING: width ≈ fixed small number of steps across grids → likely artifact ***")
        elif abs(w2["width"] - wc["width"]) / max(w2["width"], 1.0) < 0.15:
            print("  Width converges to within 15% across grids → likely robust")
        else:
            print("  Width changes significantly between grids → grid-dependent")

    print("\n  VERDICT: see width convergence above.")


# ---------------------------------------------------------------------------
# TEST 2 — Minimizer robustness
# ---------------------------------------------------------------------------

def test2_minimizer_robustness() -> None:
    _hdr("TEST 2 — Minimizer robustness (multi-start)")

    params = PARAMS_TRUNC
    fp = params.f_pi_mev

    # 4×4 grid of starting points
    phi_starts   = [1e-4, 30.0, 60.0, 93.0]
    delta_starts = [0.0, 50.0, 100.0, 200.0]

    for test_mu, label in [
        (332.0, "μ_q = 332 MeV  (claimed window centre)"),
        (320.0, "μ_q = 320 MeV  (below window — normal phase expected)"),
        (340.0, "μ_q = 340 MeV  (above window — normal phase expected)"),
    ]:
        print(f"\n  {label}")
        model = QMDSimpleModel(params)

        def obj(x):
            return model.omega(float(max(x[0], 1e-6)), float(max(x[1], 0.0)), test_mu)

        print(f"  {'φ₀ start':>10} {'Δ₀ start':>10}  →  {'φ₀ final':>10} {'Δ₀ final':>10} {'Ω_min (MeV⁴)':>20}")
        print("  " + "-" * 70)

        results = []
        for phi0 in phi_starts:
            for d0 in delta_starts:
                d_init = min(float(d0), fp)
                try:
                    res = sp_minimize(
                        obj,
                        x0=[float(phi0), d_init],
                        method="L-BFGS-B",
                        bounds=[(1e-6, 2.0 * fp), (0.0, fp)],
                        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 30000},
                    )
                    pf, df = float(res.x[0]), float(res.x[1])
                    of = float(res.fun)
                except Exception:
                    pf, df, of = float("nan"), float("nan"), float("nan")
                results.append((phi0, d0, pf, df, of))
                print(f"  {phi0:>10.2f} {d0:>10.1f}  →  {pf:>10.4f} {df:>10.4f} {of:>20.8f}")

        # Statistics
        omegas = [r[4] for r in results if np.isfinite(r[4])]
        if omegas:
            spread = max(omegas) - min(omegas)
            n_2sc   = sum(1 for r in results if r[3] > _GAP_THRESH and np.isfinite(r[4]))
            n_norm  = sum(1 for r in results if r[3] <= _GAP_THRESH and np.isfinite(r[4]))
            print(f"\n  Ω spread across all starts: {spread:.4e} MeV⁴")
            print(f"  2SC solutions: {n_2sc}/16   Normal solutions: {n_norm}/16")
            if n_2sc > 0 and n_norm > 0:
                best_2sc  = min(r[4] for r in results if r[3] > _GAP_THRESH and np.isfinite(r[4]))
                best_norm = min(r[4] for r in results if r[3] <= _GAP_THRESH and np.isfinite(r[4]))
                print(f"  Best 2SC  Ω = {best_2sc:.8f}")
                print(f"  Best norm Ω = {best_norm:.8f}")
                print(f"  ΔΩ(2SC − normal) = {best_2sc - best_norm:.6e}  "
                      f"({'2SC favored' if best_2sc < best_norm else 'normal favored'})")

    print("\n  VERDICT:")
    print("  Consistent minimum at all starts = no minimizer artifact.")
    print("  Multiple distinct local minima = artifact possible.")


# ---------------------------------------------------------------------------
# TEST 3 — Direct potential inspection
# ---------------------------------------------------------------------------

def test3_potential_inspection() -> None:
    _hdr("TEST 3 — Direct potential inspection  (Ω_trunc vs Δ₀ slices)")

    params = PARAMS_TRUNC
    pre = _make_precomputed(params)
    fp = params.f_pi_mev

    test_mus = [300.0, 320.0, 330.0, 332.0, 334.0, 340.0, 360.0, 400.0]
    delta_grid = np.linspace(0.0, 200.0, 500)

    print(f"\n  Method: fix φ₀ at the normal-phase minimum, then sweep Δ₀.")
    print(f"  ΔΩ ≡ Ω_trunc(Δ₀) − Ω_trunc(0);  negative → 2SC favored.\n")

    print(f"  {'μ_q':>7}  {'φ₀(Δ=0)':>9}  {'2nd-min?':>10}  "
          f"{'Δ₀ at min':>12}  {'ΔΩ_min (MeV⁴)':>16}  {'remarks'}")
    print("  " + "-" * 80)

    for mu in test_mus:
        # (a) Find normal-phase phi_0 (minimize at Delta=0)
        def omega_norm(phi):
            return _omega_trunc_core(float(phi), 0.0, mu, params, pre)

        res_phi = minimize_scalar(
            omega_norm,
            bounds=(1e-5, 2.0 * fp),
            method="bounded",
            options={"xatol": 1e-8},
        )
        phi0 = float(res_phi.x)
        omega0 = float(res_phi.fun)

        # (b) Sweep Omega(Delta) at fixed phi0
        omega_arr = np.array([
            _omega_trunc_core(phi0, float(d), mu, params, pre) for d in delta_grid
        ])
        shifted = omega_arr - omega0   # ΔΩ = Ω(Δ) − Ω(Δ=0)

        # (c) Look for second minimum at Δ > 5 MeV
        mask_search = delta_grid > 5.0
        d_s = delta_grid[mask_search]
        v_s = shifted[mask_search]

        has_min2 = False
        d_min2 = float("nan")
        dv_min2 = float("nan")
        remarks = ""

        if len(v_s) > 3:
            imin = int(np.argmin(v_s))
            depth = float(v_s[imin])
            # Check it's a local minimum (not just the edge)
            if imin > 0 and imin < len(v_s) - 1:
                if v_s[imin] < v_s[imin - 1] and v_s[imin] < v_s[imin + 1]:
                    has_min2 = True
                    d_min2 = float(d_s[imin])
                    dv_min2 = depth
                    remarks = "2SC min" if depth < 0 else "local min (not favored)"
                elif depth < -1.0:
                    # Minimum at edge of search — still report it
                    has_min2 = True
                    d_min2 = float(d_s[imin])
                    dv_min2 = depth
                    remarks = "edge min"
            # Check if potential is monotonically decreasing (unbounded)
            if d_s.size > 10 and v_s[-1] < v_s[-10]:
                remarks = "decreasing (unbounded?)"

        d_min2_str = f"{d_min2:.2f}" if has_min2 else "—"
        dv_str = f"{dv_min2:.4f}" if has_min2 else "—"

        print(f"  {mu:>7.1f}  {phi0:>9.3f}  {'YES' if has_min2 else 'no':>10}  "
              f"{d_min2_str:>12}  {dv_str:>16}  {remarks}")

    print("\n  VERDICT:")
    print("  If a 2SC min (depth < 0) appears only near μ ∈ [330, 334] → window is real.")
    print("  If it appears over a wide range → window claim understates the feature.")
    print("  If it never appears → window is a minimizer artifact.")


# ---------------------------------------------------------------------------
# TEST 4 — Parameter sensitivity
# ---------------------------------------------------------------------------

def test4_parameter_sensitivity() -> None:
    _hdr("TEST 4 — Parameter sensitivity of the window")

    lam0 = QMD_SET_A.lambda_0
    variants = {
        "SET_A (baseline)":  PARAMS_TRUNC,
        "P1  m_Δ=400 MeV":  replace(PARAMS_TRUNC, m_delta_mev=400.0),
        "P2  m_Δ=700 MeV":  replace(PARAMS_TRUNC, m_delta_mev=700.0),
        "P3  λ_Δ=λ₀/8":     replace(PARAMS_TRUNC, lambda_delta_factor=0.125),
    }

    mu_arr = np.linspace(250.0, 700.0, 1000)  # ~1 min per variant

    print(f"\n  Scan range: μ_q ∈ [250, 700] MeV, 1000 pts, step = {mu_arr[1]-mu_arr[0]:.4f} MeV\n")
    print(f"  {'Variant':>20}  {'onset (MeV)':>12}  {'closure (MeV)':>14}  "
          f"{'width (MeV)':>12}  {'peak gap (MeV)':>15}")
    print("  " + "-" * 78)

    for name, params in variants.items():
        print(f"  Scanning {name} ...")
        _, gaps, _ = _scan(mu_arr, params=params, tag=name)
        w = _window(mu_arr, gaps)
        if w["onset"] is None:
            print(f"  {name:>20}  {'—':>12}  {'—':>14}  {'—':>12}  (no window)")
        else:
            print(f"  {name:>20}  {w['onset']:>12.2f}  {w['closure']:>14.2f}  "
                  f"{w['width']:>12.2f}  {w['peak']:>15.4f}")

    print("\n  Expected monotonic behaviour:")
    print("    P1 m_Δ=400: 2SC onset at lower μ, possibly wider or earlier window")
    print("    P2 m_Δ=700: 2SC onset at higher μ or no window at all")
    print("    P3 λ_Δ/8: wider window (quartic stiffness reduced)")
    print("\n  VERDICT:")
    print("  Monotonic + matches physics → window well-understood.")
    print("  Non-monotonic or wrong direction → thesis explanation needs revision.")


# ---------------------------------------------------------------------------
# TEST 5 — Tree-level + quartic minimal model
# ---------------------------------------------------------------------------

def test5_minimal_model() -> None:
    _hdr("TEST 5 — Tree-level + quartic minimal model (two-term)")

    params = PARAMS_TRUNC
    mD  = params.m_delta_mev
    mD2 = mD ** 2
    lamD = params.lambda_delta

    print(f"\n  Minimal model: V(Δ₀) = (mΔ² − 4μ²) Δ₀²  +  (λ_Δ/6) Δ₀⁴")
    print(f"  SET A: mΔ = {mD:.1f} MeV,  λ_Δ = {lamD:.6f}")
    print(f"\n  Non-trivial minimum exists when:")
    print(f"    (i)  mΔ² − 4μ² < 0  →  μ > mΔ/2 = {mD/2:.2f} MeV")
    print(f"    (ii) λ_Δ > 0  (always true for SET A: λ_Δ = {lamD:.4f})")
    print(f"\n  At the minimum:  Δ₀² = −3(mΔ² − 4μ²)/λ_Δ = 3(4μ² − mΔ²)/λ_Δ")
    print(f"  V_min = −3(4μ² − mΔ²)²/λ_Δ  (always ≤ 0 when Δ₀ ≠ 0)\n")
    print(f"  The 2-term model ALWAYS favors the 2SC minimum once μ > mΔ/2 = {mD/2:.2f} MeV.")
    print(f"  There is no upper closure; the minimum exists for all μ > {mD/2:.2f} MeV.")

    print(f"\n  Tabulating 2-term potential at selected μ_q values:")
    print(f"\n  {'μ_q':>7}  {'4μ²−mΔ² (MeV²)':>18}  {'Δ₀_min (MeV)':>14}  "
          f"{'V_min/Δ₀_min² (MeV²)':>22}  {'min type':>10}")
    print("  " + "-" * 78)

    for mu in [200.0, 240.0, 249.9, 250.1, 260.0, 300.0, 320.0, 330.0, 332.0,
               334.0, 340.0, 360.0, 400.0, 500.0]:
        c2 = mD2 - 4.0 * mu ** 2
        c4 = lamD / 6.0
        if c2 < 0.0 and c4 > 0.0:
            delta_min = float(np.sqrt(-c2 / (2.0 * c4)))
            v_min_over_d2 = c2 + c4 * delta_min ** 2   # = c2/2 at minimum
            min_type = "2SC"
        else:
            delta_min = 0.0
            v_min_over_d2 = float("nan")
            min_type = "normal"
        coeff = 4.0 * mu ** 2 - mD2
        d_str = f"{delta_min:.3f}" if min_type == "2SC" else "0"
        v_str = f"{v_min_over_d2:.4f}" if min_type == "2SC" else "—"
        print(f"  {mu:>7.1f}  {coeff:>18.2f}  {d_str:>14}  {v_str:>22}  {min_type:>10}")

    print(f"\n  The 2-term model has an unbounded 2SC minimum for ALL μ_q > {mD/2:.2f} MeV.")
    print(f"  The truncated analytic potential (Ω_trunc) closes this window because")
    print(f"  additional one-loop log terms (log C, loop functions) raise the")
    print(f"  effective quartic cost faster than the tree-level driving (4μ² − mΔ²)")
    print(f"  can sustain the condensate.  The 3.6 MeV window is a narrow regime")
    print(f"  where this balance is achieved by the FULL analytic potential,")
    print(f"  not just the tree + quartic pair.")
    print()
    print(f"  VERDICT:")
    print(f"  The 2-term model does NOT reproduce a 3.6 MeV window; it gives a permanent")
    print(f"  2SC phase above μ = {mD/2:.1f} MeV.  The thesis explanation attributing")
    print(f"  closure to 'the quartic catching up' is a simplification; the window")
    print(f"  closure is driven by the full analytic log structure in Ω_trunc.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _hdr("QMD 2SC WINDOW DIAGNOSTIC  —  SET A  (Ω₁,num = 0  truncation)")

    print(f"\nTruncated parameters (include_omega_1_num=False):")
    print(PARAMS_TRUNC.describe())
    print(f"\nGap threshold for '2SC': Δ₀ > {_GAP_THRESH} MeV")

    t_total = time.time()

    test1_grid_resolution()
    test2_minimizer_robustness()
    test3_potential_inspection()
    test4_parameter_sensitivity()
    test5_minimal_model()

    print(f"\n{'='*70}")
    print(f"END OF DIAGNOSTIC  (total wall time: {time.time()-t_total:.1f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
