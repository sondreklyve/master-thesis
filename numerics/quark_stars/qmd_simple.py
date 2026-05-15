"""QMD mean-field potential at a single common quark chemical potential.

Implements Omega_trunc(phi, Delta; mu_q) for the simple case
    mu_e = mu_8 = delta_mu = 0  (common mu_q for all quarks).

Decomposition of the full potential (Eq. A.22 in thesis, Andersen 2024):
    Omega_{0+1} = Omega_trunc  +  Omega_1_num  +  Omega^{mu,T}

  Omega_trunc   — the analytic renormalized vacuum contribution (implemented here).
  Omega_1_num   — finite BCS residual integral, Eq. A.23.
  Omega^{mu,T}  — T=0 finite-density term for blue quarks plus gapless
                  paired quasiparticle modes.

At Delta=0 the T=0 medium term reduces to the free Fermi sea for all 12 quark
DOF (2 spins x 2 flavors x 3 colors), treated as free quarks with constituent
mass M_q=g*phi:

    Omega^{mu,T=0} = fermion_grand_potential(mu_q, g*phi, degeneracy=12)

For Delta>0, only the blue quarks remain as free Fermi seas.  The red-green
paired sector is supplied by Omega_1_num and, when a mismatch is present, by
the gapless quasiparticle medium term.  In the common-mu benchmark the
mismatch is zero, so the gapless term vanishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad

from .constants import PI
from .qmd_parameters import QMDParameters
from .solvers.minimization import find_global_minimum
from .thermodynamics.fermi import fermion_grand_potential


_LOG_FLOOR = 1.0e-30  # lower bound for log arguments to prevent log(0)
_FIELD_ZERO_TOL = 1.0e-10  # MeV -- numerical zero for switching exact branches
_DELTA_PHASE_TOL = 1.0e-2  # MeV — Delta below this threshold is "normal" phase
_QUAD_EPSABS = 1.0e-3
_QUAD_EPSREL = 1.0e-7
_QUAD_LIMIT = 100
_CACHE_DIGITS = 8


# ---------------------------------------------------------------------------
# Standalone loop integrals (mirror TwoFlavorQMPotential._loop_F / _loop_G_pi
# but decoupled from QMFittedParameters)
# ---------------------------------------------------------------------------
#
# These functions and the shared-core helpers below (_QMDPrecomputed,
# _make_precomputed, _omega_trunc_core) are also imported by qmd_stellar.py.
# The underscore prefix signals internal use; they are not part of the public API.


def _loop_r(mass_mev: float, m_q_mev: float) -> complex:
    return np.sqrt(4.0 * m_q_mev**2 / mass_mev**2 - 1.0 + 0.0j)


def _loop_F(mass_mev: float, m_q_mev: float) -> float:
    """F(mass^2) = Re[2 - 2r·arctan(1/r)] with r = sqrt(4mq²/m² - 1)."""
    r = _loop_r(mass_mev, m_q_mev)
    if abs(r) < 1.0e-12:
        return 2.0
    return float(np.real(2.0 - 2.0 * r * np.arctan(1.0 / r)))


def _loop_G_pi(m_pi_mev: float, m_q_mev: float) -> float:
    """Return m_pi² F'(m_pi²) = Re[4mq²/(m_pi²·r_pi)·arctan(1/r_pi) - 1]."""
    r_pi = _loop_r(m_pi_mev, m_q_mev)
    prefactor = 4.0 * m_q_mev**2 / (m_pi_mev**2 * r_pi)
    return float(np.real(prefactor * np.arctan(1.0 / r_pi) - 1.0))


# ---------------------------------------------------------------------------
# Shared analytic-potential core (used by QMDSimpleModel and QMDStellarModel)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QMDPrecomputed:
    """Vacuum-level loop integrals precomputed once from QMDParameters.

    Shared by QMDSimpleModel and QMDStellarModel to avoid redundant computation.
    Create via _make_precomputed(params).
    """

    pi16sq: float            # (4π)²
    F_pi: float              # F(m_pi²)
    G_pi: float              # m_pi² F'(m_pi²)
    F_sigma: float           # F(m_sigma²)
    loop_pf: float           # 12 m_q² / ((4π)² f_π²)
    ratio_4mq2_msig2: float  # 4 m_q² / m_sigma²


def _make_precomputed(params: QMDParameters) -> _QMDPrecomputed:
    """Precompute vacuum-level loop integrals from QMDParameters."""
    pi16sq = (4.0 * PI) ** 2
    return _QMDPrecomputed(
        pi16sq=pi16sq,
        F_pi=_loop_F(params.m_pi_mev, params.m_q_mev),
        G_pi=_loop_G_pi(params.m_pi_mev, params.m_q_mev),
        F_sigma=_loop_F(params.m_sigma_mev, params.m_q_mev),
        loop_pf=12.0 * params.m_q_mev**2 / (pi16sq * params.f_pi_mev**2),
        ratio_4mq2_msig2=4.0 * params.m_q_mev**2 / params.m_sigma_mev**2,
    )


def _omega_trunc_core(
    phi_mev: float,
    delta_mev: float,
    mu_bar_mev: float,
    params: QMDParameters,
    pre: _QMDPrecomputed,
) -> float:
    """Analytic renormalized 1-loop QMD potential Omega_trunc (MeV^4).

    Implements the full meson+diquark analytic truncated potential.
    The chemical potential enters only in the Delta^2 coefficient through mu_bar_mev:

        delta2_coeff = m_Delta^2 - 4*mu_bar^2 * (1 + 4*g_Delta^2/(4π)^2 * (...))

    Caller conventions:
      QMDSimpleModel (common-mu):    mu_bar_mev = mu_q_mev
      QMDStellarModel (general):     mu_bar_mev = mu_q - mu_e/6 + mu_8/3

    At Delta=0 the mu_bar_mev argument has no effect; all Delta-containing terms vanish.
    """
    p = params
    fp = p.f_pi_mev
    mq = p.m_q_mev
    mpi = p.m_pi_mev
    msigma = p.m_sigma_mev
    g = p.g
    gd = p.g_delta
    mD = p.m_delta_mev
    lam3 = p.lambda_3
    lamD = p.lambda_delta
    mu = mu_bar_mev

    x = phi_mev / fp
    pi16sq = pre.pi16sq
    loop_pf = pre.loop_pf
    F_pi = pre.F_pi
    F_sigma = pre.F_sigma
    G_pi = pre.G_pi

    A_safe = max(g**2 * phi_mev**2, _LOG_FLOOR)
    C_safe = max(g**2 * phi_mev**2 + gd**2 * delta_mev**2, _LOG_FLOOR)
    log_A = np.log(mq**2 / A_safe)
    log_C = np.log(mq**2 / C_safe)

    # ------------------------------------------------------------------ #
    # Meson (phi) sector                                                   #
    # ------------------------------------------------------------------ #
    t1 = 0.75 * mpi**2 * fp**2 * (1.0 - loop_pf * G_pi) * x**2
    t2 = (2.0 * mq**4 / pi16sq) * (4.5 + log_A + 2.0 * log_C) * x**4
    bracket3 = (1.0 - pre.ratio_4mq2_msig2) * F_sigma + pre.ratio_4mq2_msig2 - F_pi - G_pi
    t3 = -0.25 * msigma**2 * fp**2 * (1.0 + loop_pf * bracket3) * x**2
    bracket4 = (1.0 - pre.ratio_4mq2_msig2) * F_sigma - F_pi - G_pi
    t4 = 0.125 * msigma**2 * fp**2 * (1.0 + loop_pf * bracket4) * x**4
    t5 = -0.125 * mpi**2 * fp**2 * (1.0 - loop_pf * G_pi) * x**4
    t6 = -mpi**2 * fp**2 * (1.0 - loop_pf * G_pi) * x

    # ------------------------------------------------------------------ #
    # Diquark (Delta) sector                                               #
    # ------------------------------------------------------------------ #
    delta2_coeff = mD**2 - 4.0 * mu**2 * (
        1.0 + 4.0 * gd**2 / pi16sq * (log_C - F_pi - G_pi)
    )
    t_delta2 = delta2_coeff * delta_mev**2
    t_lam3 = lam3 / 12.0 * phi_mev**2 * delta_mev**2
    t_lamD = lamD / 6.0 * delta_mev**4
    t_loop1 = 12.0 * g**2 * gd**2 / pi16sq * phi_mev**2 * delta_mev**2
    t_loop2 = 6.0 * gd**4 / pi16sq * delta_mev**4
    t_loop3 = 4.0 * gd**4 / pi16sq * (log_C - F_pi - G_pi) * delta_mev**4
    t_loop4 = p.t_loop4_factor * g**2 * gd**2 / pi16sq * (log_C - F_pi - G_pi) * phi_mev**2 * delta_mev**2

    return float(
        t1 + t2 + t3 + t4 + t5 + t6
        + t_delta2 + t_lam3 + t_lamD
        + t_loop1 + t_loop2 + t_loop3 + t_loop4
    )


# ---------------------------------------------------------------------------
# T=0 2SC quasiparticle medium terms
# ---------------------------------------------------------------------------


def _cache_value(value: float) -> float:
    """Round quadrature cache keys without changing physics-scale accuracy."""
    return round(float(value), _CACHE_DIGITS)


def _mass_and_gap(
    phi_mev: float,
    delta_mev: float,
    params: QMDParameters,
) -> tuple[float, float]:
    return params.g * phi_mev, params.g_delta * delta_mev


def _quasiparticle_energy(
    momentum_mev: float,
    mass_mev: float,
    gap_mev: float,
    mu_shift_mev: float,
) -> float:
    energy = np.sqrt(momentum_mev * momentum_mev + mass_mev * mass_mev)
    return float(np.sqrt((energy + mu_shift_mev) ** 2 + gap_mev * gap_mev))


def _integrate_quad(function, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    value, _ = quad(
        function,
        lower,
        upper,
        epsabs=_QUAD_EPSABS,
        epsrel=_QUAD_EPSREL,
        limit=_QUAD_LIMIT,
    )
    return float(value)


@lru_cache(maxsize=65536)
def _omega_1_num_cached(
    mass_mev: float,
    gap_mev: float,
    mu_bar_mev: float,
    cutoff_mev: float,
) -> float:
    """Cached Eq. A.23 quadrature in physical mass/gap variables."""
    mass = float(mass_mev)
    gap = float(gap_mev)
    mu = float(mu_bar_mev)
    cutoff = float(cutoff_mev)
    if abs(mu) <= _FIELD_ZERO_TOL:
        return 0.0

    def integrand(momentum: float) -> float:
        energy = np.sqrt(momentum * momentum + mass * mass)
        paired_scale = np.sqrt(momentum * momentum + mass * mass + gap * gap)
        e_plus = np.sqrt((energy + mu) ** 2 + gap * gap)
        e_minus = np.sqrt((energy - mu) ** 2 + gap * gap)
        subtraction = mu * mu * gap * gap / (paired_scale**3)
        return momentum * momentum * (e_plus + e_minus - 2.0 * paired_scale - subtraction)

    integral = _integrate_quad(integrand, 0.0, cutoff)
    return -2.0 * integral / (PI**2)


@lru_cache(maxsize=65536)
def _omega_1_num_mu_derivative_cached(
    mass_mev: float,
    gap_mev: float,
    mu_bar_mev: float,
    cutoff_mev: float,
) -> float:
    """d Omega_1_num / d mu_bar for the cached Eq. A.23 quadrature."""
    mass = float(mass_mev)
    gap = float(gap_mev)
    mu = float(mu_bar_mev)
    cutoff = float(cutoff_mev)
    if abs(mu) <= _FIELD_ZERO_TOL:
        return 0.0

    def integrand(momentum: float) -> float:
        energy = np.sqrt(momentum * momentum + mass * mass)
        paired_scale = np.sqrt(momentum * momentum + mass * mass + gap * gap)
        e_plus = np.sqrt((energy + mu) ** 2 + gap * gap)
        e_minus = np.sqrt((energy - mu) ** 2 + gap * gap)
        return momentum * momentum * (
            (energy + mu) / e_plus
            - (energy - mu) / e_minus
            - 2.0 * mu * gap * gap / (paired_scale**3)
        )

    integral = _integrate_quad(integrand, 0.0, cutoff)
    return -2.0 * integral / (PI**2)


def _gapless_energy_intervals(
    mass_mev: float,
    gap_mev: float,
    mu_shift_mev: float,
    delta_mu_abs_mev: float,
    cutoff_mev: float,
) -> list[tuple[float, float]]:
    """Momentum intervals where delta_mu_abs > E_delta(p).

    For E_delta(p)=sqrt((E(p)+mu_shift)^2+gap^2), the inequality can be
    solved analytically as an interval in E(p), avoiding fragile root scans.
    """
    if delta_mu_abs_mev <= gap_mev:
        return []

    q = float(np.sqrt(delta_mu_abs_mev * delta_mu_abs_mev - gap_mev * gap_mev))
    energy_low = max(mass_mev, -q - mu_shift_mev)
    energy_high = q - mu_shift_mev
    if energy_high <= energy_low:
        return []

    p_low = float(np.sqrt(max(energy_low * energy_low - mass_mev * mass_mev, 0.0)))
    p_high = float(np.sqrt(max(energy_high * energy_high - mass_mev * mass_mev, 0.0)))
    p_high = min(p_high, cutoff_mev)
    if p_high <= p_low:
        return []
    return [(p_low, p_high)]


@lru_cache(maxsize=65536)
def _gapless_integral_cached(
    mass_mev: float,
    gap_mev: float,
    mu_shift_mev: float,
    delta_mu_abs_mev: float,
    cutoff_mev: float,
) -> float:
    intervals = _gapless_energy_intervals(
        float(mass_mev),
        float(gap_mev),
        float(mu_shift_mev),
        float(delta_mu_abs_mev),
        float(cutoff_mev),
    )
    if not intervals:
        return 0.0

    def integrand(momentum: float) -> float:
        return momentum * momentum * (
            delta_mu_abs_mev
            - _quasiparticle_energy(momentum, mass_mev, gap_mev, mu_shift_mev)
        )

    return sum(_integrate_quad(integrand, lower, upper) for lower, upper in intervals)


@lru_cache(maxsize=65536)
def _gapless_mu_shift_derivative_integral_cached(
    mass_mev: float,
    gap_mev: float,
    mu_shift_mev: float,
    delta_mu_abs_mev: float,
    cutoff_mev: float,
) -> float:
    """Integral of p^2 (E+shift)/E_delta over gapless intervals."""
    intervals = _gapless_energy_intervals(
        float(mass_mev),
        float(gap_mev),
        float(mu_shift_mev),
        float(delta_mu_abs_mev),
        float(cutoff_mev),
    )
    if not intervals:
        return 0.0

    def integrand(momentum: float) -> float:
        energy = np.sqrt(momentum * momentum + mass_mev * mass_mev)
        e_delta = _quasiparticle_energy(momentum, mass_mev, gap_mev, mu_shift_mev)
        return momentum * momentum * (energy + mu_shift_mev) / e_delta

    return sum(_integrate_quad(integrand, lower, upper) for lower, upper in intervals)


@lru_cache(maxsize=65536)
def _gapless_volume_integral_cached(
    mass_mev: float,
    gap_mev: float,
    mu_shift_mev: float,
    delta_mu_abs_mev: float,
    cutoff_mev: float,
) -> float:
    """Integral of p^2 over gapless intervals."""
    intervals = _gapless_energy_intervals(
        float(mass_mev),
        float(gap_mev),
        float(mu_shift_mev),
        float(delta_mu_abs_mev),
        float(cutoff_mev),
    )
    return sum((upper**3 - lower**3) / 3.0 for lower, upper in intervals)


def omega_1_num(
    phi_mev: float,
    delta_mev: float,
    mu_bar_mev: float,
    params: QMDParameters,
) -> float:
    """BCS pairing correction to the Fermi sea (MeV^4), Eq. A.23 of thesis.

    The quadrature is written in terms of M=g*phi and gap=g_delta*Delta,
    matching GeneralFunctions_2SC.py in git-clones.  It is only evaluated for
    Delta>0; the exact Delta=0 branch is handled by the free normal-phase
    medium term to avoid numerical integration through the Fermi-surface kink.
    """
    if (not params.include_medium_term) or (not params.include_omega_1_num):
        return 0.0
    if delta_mev <= _FIELD_ZERO_TOL:
        return 0.0

    mass, gap = _mass_and_gap(phi_mev, delta_mev, params)
    return _omega_1_num_cached(
        _cache_value(mass),
        _cache_value(gap),
        _cache_value(mu_bar_mev),
        _cache_value(params.residual_cutoff_mev),
    )


def omega_1_num_mu_derivative(
    phi_mev: float,
    delta_mev: float,
    mu_bar_mev: float,
    params: QMDParameters,
) -> float:
    """Return d Omega_1_num / d mu_bar in MeV^3."""
    if (not params.include_medium_term) or (not params.include_omega_1_num):
        return 0.0
    if delta_mev <= _FIELD_ZERO_TOL:
        return 0.0

    mass, gap = _mass_and_gap(phi_mev, delta_mev, params)
    return _omega_1_num_mu_derivative_cached(
        _cache_value(mass),
        _cache_value(gap),
        _cache_value(mu_bar_mev),
        _cache_value(params.residual_cutoff_mev),
    )


def omega_qmd_paired_gapless_t0(
    phi_mev: float,
    delta_mev: float,
    mu_bar_mev: float,
    delta_mu_mev: float,
    params: QMDParameters,
) -> float:
    """T=0 contribution from gapless paired 2SC quasiparticle modes.

    This term vanishes in the common-mu case and whenever
    |delta_mu| <= gap.  The factor matches the reference-code expression
    -2/pi^2 * integral p^2 (|delta_mu|-E_delta) over gapless intervals.
    """
    if delta_mev <= _FIELD_ZERO_TOL:
        return 0.0

    mass, gap = _mass_and_gap(phi_mev, delta_mev, params)
    delta_abs = abs(float(delta_mu_mev))
    cutoff = params.residual_cutoff_mev
    if delta_abs <= gap:
        return 0.0

    plus = _gapless_integral_cached(
        _cache_value(mass),
        _cache_value(gap),
        _cache_value(mu_bar_mev),
        _cache_value(delta_abs),
        _cache_value(cutoff),
    )
    minus = _gapless_integral_cached(
        _cache_value(mass),
        _cache_value(gap),
        _cache_value(-mu_bar_mev),
        _cache_value(delta_abs),
        _cache_value(cutoff),
    )
    return -2.0 * (plus + minus) / (PI**2)


def omega_qmd_paired_gapless_derivatives_t0(
    phi_mev: float,
    delta_mev: float,
    mu_bar_mev: float,
    delta_mu_mev: float,
    params: QMDParameters,
) -> tuple[float, float]:
    """Return derivatives of the gapless medium term.

    Returns ``(dOmega_gapless/dmu_bar, dOmega_gapless/ddelta_mu)`` in MeV^3.
    Both derivatives vanish in the fully gapped regime.
    """
    if delta_mev <= _FIELD_ZERO_TOL:
        return 0.0, 0.0

    mass, gap = _mass_and_gap(phi_mev, delta_mev, params)
    delta_abs = abs(float(delta_mu_mev))
    cutoff = params.residual_cutoff_mev
    if delta_abs <= gap:
        return 0.0, 0.0

    mass_key = _cache_value(mass)
    gap_key = _cache_value(gap)
    mu_key = _cache_value(mu_bar_mev)
    minus_mu_key = _cache_value(-mu_bar_mev)
    delta_key = _cache_value(delta_abs)
    cutoff_key = _cache_value(cutoff)

    plus_mu = _gapless_mu_shift_derivative_integral_cached(
        mass_key, gap_key, mu_key, delta_key, cutoff_key
    )
    minus_mu = _gapless_mu_shift_derivative_integral_cached(
        mass_key, gap_key, minus_mu_key, delta_key, cutoff_key
    )
    d_mu_bar = 2.0 * (plus_mu - minus_mu) / (PI**2)

    plus_volume = _gapless_volume_integral_cached(
        mass_key, gap_key, mu_key, delta_key, cutoff_key
    )
    minus_volume = _gapless_volume_integral_cached(
        mass_key, gap_key, minus_mu_key, delta_key, cutoff_key
    )
    sign = 1.0 if delta_mu_mev > 0.0 else -1.0 if delta_mu_mev < 0.0 else 0.0
    d_delta_mu = sign * (-2.0) * (plus_volume + minus_volume) / (PI**2)
    return d_mu_bar, d_delta_mu


def omega_qmd_medium_t0_common(
    phi_mev: float,
    delta_mev: float,
    mu_q_mev: float,
    params: QMDParameters,
) -> float:
    """T=0 finite-density contribution in the common-mu QMD case.

    At Delta=0 this is the free Fermi sea for all 12 quark DOF.  In the
    2SC phase the only free Fermi seas are the unpaired blue quarks
    (degeneracy 4); the red-green paired sector is supplied by omega_1_num
    and gapless quasiparticle terms.

    The common-mu mismatch is zero, so the gapless contribution vanishes.
    """
    M_q = params.g * phi_mev
    if delta_mev <= _FIELD_ZERO_TOL:
        return fermion_grand_potential(mu_q_mev, M_q, degeneracy=12.0)
    return fermion_grand_potential(mu_q_mev, M_q, degeneracy=4.0)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QMDSimpleState:
    """Solution of the QMD mean-field equations at a fixed mu_q."""

    mu_q_mev: float
    phi_mev: float
    delta_mev: float
    gap_mev: float        # g_delta * delta_mev
    omega_min_mev4: float
    phase: str            # "normal" or "2SC"
    success: bool


@dataclass(frozen=True)
class QMDSimpleEoSPoint:
    """Single thermodynamic point on the QMD simple EoS."""

    mu_q_mev: float
    phi_mev: float
    delta_mev: float
    gap_mev: float
    phase: str
    omega_min_mev4: float
    pressure_mev4: float
    n_q_mev3: float
    energy_density_mev4: float
    cs2: float            # speed of sound squared (c_s/c)^2
    success: bool


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class QMDSimpleModel:
    """Quark-Meson-Diquark model at a single common quark chemical potential.

    Implements Eq. A.22/A.23 for mu_e = mu_8 = delta_mu = 0.  The Delta=0
    branch reduces to the ordinary two-flavor QM grand potential.  In the
    2SC branch, the medium term contains the unpaired blue Fermi sea and the
    paired red-green contribution from Omega_1_num.
    """

    def __init__(self, params: QMDParameters) -> None:
        self.params = params
        self._pre = _make_precomputed(params)

    def omega(self, phi_mev: float, delta_mev: float, mu_q_mev: float) -> float:
        """Full QMD grand potential in MeV^4.

        Combines:
          1. Omega_trunc: analytic renormalized 1-loop potential (phi + Delta sectors)
             — delegated to _omega_trunc_core with mu_bar = mu_q (common-mu case)
          2. Omega^{mu,T=0}: all free quarks at Delta=0, blue quarks in 2SC
          3. Omega_1_num: finite BCS residual integral, Eq. A.23

        At Delta=0 this reproduces the ordinary QM grand potential up to a
        mu-independent vacuum-level constant (exact to floating-point precision).
        """
        trunc = _omega_trunc_core(phi_mev, delta_mev, mu_q_mev, self.params, self._pre)
        omega1 = omega_1_num(phi_mev, delta_mev, mu_q_mev, self.params)
        omega_med = (
            omega_qmd_medium_t0_common(phi_mev, delta_mev, mu_q_mev, self.params)
            if self.params.include_medium_term
            else 0.0
        )
        return float(trunc + omega1 + omega_med)

    def solve_mean_fields(
        self,
        mu_q_mev: float,
        initial_guess: tuple[float, float] | None = None,
    ) -> QMDSimpleState:
        """Minimize the QMD grand potential over (phi, Delta) at fixed mu_q.

        Uses multi-start L-BFGS-B.  A coarse grid is used only when no
        continuation guess is available; scans pass the previous state as the
        first guess and avoid redoing the expensive grid at every point.
        """
        p = self.params
        fp = p.f_pi_mev

        # Upper bound for Delta: the chiral scale f_pi keeps the search inside
        # the condensate range where the truncated analytic expansion is meant
        # to be trusted.  The physical gap g_delta*Delta is still allowed to be
        # O(100 MeV), matching the 2SC scale explored in the reference notebook.
        delta_max = fp
        bounds = [(1.0e-6, 2.0 * fp), (0.0, delta_max)]

        guesses: list[tuple[float, float]] = [
            (fp, 0.0),
            (0.5 * fp, 0.0),
            (1.0e-3, 0.0),
            (0.5 * fp, 0.3 * delta_max),
            (1.0e-3, 0.3 * delta_max),
            (1.0e-3, 0.8 * delta_max),
        ]
        if initial_guess is not None:
            guesses.insert(0, initial_guess)

        result = find_global_minimum(
            lambda x: self.omega(float(x[0]), float(x[1]), mu_q_mev),
            bounds=bounds,
            initial_guesses=guesses,
            grid_shape=(40, 40) if initial_guess is None else None,
        )

        phi_min = float(result.x[0])
        delta_min = float(result.x[1])
        phase = "normal" if delta_min <= _DELTA_PHASE_TOL else "2SC"

        return QMDSimpleState(
            mu_q_mev=mu_q_mev,
            phi_mev=phi_min,
            delta_mev=delta_min,
            gap_mev=p.g_delta * delta_min,
            omega_min_mev4=float(result.fun),
            phase=phase,
            success=result.success,
        )

    def pressure_from_state(self, state: "QMDSimpleState", p_ref_mev4: float = 0.0) -> float:
        """Pressure P = -(Omega_min - Omega_ref) in MeV^4."""
        return -(state.omega_min_mev4 - p_ref_mev4)

    def build_eos(
        self,
        mu_q_values_mev: "np.ndarray",
        dmu_mev: float = 0.5,
    ) -> "list[QMDSimpleEoSPoint]":
        """Build the T=0 EoS over a scan of mu_q values.

        Pressure is normalised so that P=0 at the lowest mu_q where Omega first
        reaches its minimum (i.e., at the onset of the non-trivial branch).
        Number density and energy density are derived via:
            n_q  = dP/dmu_q  (numerical derivative, central differences)
            eps  = -P + mu_q * n_q
            cs2  = (dP/dmu_q) / (deps/dmu_q)

        Only points with P >= 0 are returned (positive-pressure branch).
        """
        states = []
        prev_guess: "tuple[float, float] | None" = None
        for mu in mu_q_values_mev:
            s = self.solve_mean_fields(float(mu), initial_guess=prev_guess)
            states.append(s)
            prev_guess = (s.phi_mev, s.delta_mev)

        omegas = np.array([s.omega_min_mev4 for s in states])
        mus = np.array([s.mu_q_mev for s in states])

        # Normalise: P = 0 at the lowest physical mu (vacuum onset)
        omega_ref = omegas[0]
        pressures = -(omegas - omega_ref)

        # Number density: dP/dmu_q via central differences
        n_q = np.gradient(pressures, mus)

        # Energy density: eps = -P + mu * n
        eps = -pressures + mus * n_q

        # Speed of sound squared: cs² = dP/deps = (dP/dmu)/(deps/dmu)
        deps_dmu = np.gradient(eps, mus)
        dp_dmu = np.gradient(pressures, mus)
        with np.errstate(invalid="ignore", divide="ignore"):
            cs2 = np.where(deps_dmu > 0.0, dp_dmu / deps_dmu, np.nan)

        points = []
        for i, s in enumerate(states):
            # Require positive pressure and non-negative number density
            # (filters out vacuum region and numerical instabilities)
            if pressures[i] < 0.0 or n_q[i] < 0.0:
                continue
            points.append(QMDSimpleEoSPoint(
                mu_q_mev=s.mu_q_mev,
                phi_mev=s.phi_mev,
                delta_mev=s.delta_mev,
                gap_mev=s.gap_mev,
                phase=s.phase,
                omega_min_mev4=s.omega_min_mev4,
                pressure_mev4=float(pressures[i]),
                n_q_mev3=float(n_q[i]),
                energy_density_mev4=float(eps[i]),
                cs2=float(cs2[i]),
                success=s.success,
            ))
        return points
