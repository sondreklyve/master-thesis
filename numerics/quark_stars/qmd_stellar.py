"""QMD stellar-matter potential and chemical-potential helpers.

Contains:
  - Chemical-potential mapping functions (pure arithmetic, no potential)
  - QMDStellarModel: grand potential at general (mu_q, mu_e, mu_8)

The stellar potential uses the same analytic Omega_trunc as QMDSimpleModel
(shared via _omega_trunc_core / _make_precomputed from qmd_simple.py), but:
  - replaces mu_q -> mu_bar = mu_q - mu_e/6 + mu_8/3 in the Delta^2 coefficient
  - uses individual color-flavor Fermi seas in the normal phase
  - uses blue-quark Fermi seas plus paired quasiparticle terms in the 2SC phase
  - adds the electron grand potential Omega_e(mu_e)
  - includes Omega_1_num from Eq. A.23 when enabled by QMDParameters

Reduction properties:
  - At mu_e=mu_8=0: QMDStellarModel.omega == QMDSimpleModel.omega  (exact)
  - At Delta=0: medium term correctly routes through per-species mu_fc
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .qmd_parameters import QMDParameters
from .qmd_simple import (
    _FIELD_ZERO_TOL,
    _QMDPrecomputed,
    _make_precomputed,
    _omega_trunc_core,
    omega_1_num,
    omega_1_num_mu_derivative,
    omega_qmd_paired_gapless_derivatives_t0,
    omega_qmd_paired_gapless_t0,
)
from .solvers.minimization import find_global_minimum
from .thermodynamics.fermi import (
    electron_grand_potential,
    electron_number_density,
    fermion_grand_potential,
    fermion_number_density,
)


_DELTA_PHASE_TOL = 1.0e-2  # MeV -- Delta below this threshold is "normal" phase
_OBJECTIVE_FAILURE_PENALTY = 1.0e80


def quark_chemical_potentials(
    mu_q_mev: float,
    mu_e_mev: float,
    mu_8_mev: float,
) -> dict[str, float]:
    """Individual color-flavor quark chemical potentials in the 2SC phase.

    In the 2SC phase the red and green quarks of both flavors are paired,
    leaving blue quarks unpaired. Charge and color neutrality are enforced
    via mu_e (electromagnetic) and mu_8 (8th SU(3)_c generator).

    Convention (mu_q is the average baryon chemical potential / 3):
      mu_{f,r} = mu_{f,g} = mu_q + Q_f * mu_e + T8_f * mu_8
      mu_{f,b}             = mu_q + Q_f * mu_e + T8_b * mu_8

    with electric charges Q_u = -2/3, Q_d = +1/3 (sign convention:
    mu_f -> mu_f + Q_f * mu_e shifts potential up for negative charges down)
    and color generator eigenvalues T8_rg = +1/3, T8_b = -2/3.

    Parameters
    ----------
    mu_q_mev : float
        Average quark chemical potential (= mu_B / 3) in MeV.
    mu_e_mev : float
        Electron chemical potential in MeV.
    mu_8_mev : float
        Color-8 chemical potential in MeV.

    Returns
    -------
    dict with keys mu_ur, mu_ug, mu_dr, mu_dg, mu_ub, mu_db (all in MeV).
    """
    mu_rg_up = mu_q_mev - 2.0 / 3.0 * mu_e_mev + 1.0 / 3.0 * mu_8_mev
    mu_rg_down = mu_q_mev + 1.0 / 3.0 * mu_e_mev + 1.0 / 3.0 * mu_8_mev
    mu_b_up = mu_q_mev - 2.0 / 3.0 * mu_e_mev - 2.0 / 3.0 * mu_8_mev
    mu_b_down = mu_q_mev + 1.0 / 3.0 * mu_e_mev - 2.0 / 3.0 * mu_8_mev
    return {
        "mu_ur": mu_rg_up,
        "mu_ug": mu_rg_up,
        "mu_dr": mu_rg_down,
        "mu_dg": mu_rg_down,
        "mu_ub": mu_b_up,
        "mu_db": mu_b_down,
    }


def pairing_chemical_potentials(
    mu_q_mev: float,
    mu_e_mev: float,
    mu_8_mev: float,
) -> dict[str, float]:
    """Average and mismatch chemical potentials for the 2SC ud diquark pair.

    The 2SC condensate pairs (ur, dg) and (ug, dr). The relevant combinations
    for the quasiparticle spectrum are:
      mu_bar  = average chemical potential of the paired quarks
      delta_mu = chemical potential mismatch between the paired species

    Convention:
      mu_bar   = mu_q - mu_e / 6 + mu_8 / 3
      delta_mu = mu_e / 2

    Parameters
    ----------
    mu_q_mev : float
        Average quark chemical potential in MeV.
    mu_e_mev : float
        Electron chemical potential in MeV.
    mu_8_mev : float
        Color-8 chemical potential in MeV.

    Returns
    -------
    dict with keys mu_bar and delta_mu (both in MeV).
    """
    mu_bar = mu_q_mev - mu_e_mev / 6.0 + mu_8_mev / 3.0
    delta_mu = mu_e_mev / 2.0
    return {"mu_bar": mu_bar, "delta_mu": delta_mu}


def gapless_diagnostic(
    delta_mev: float,
    params: QMDParameters,
    delta_mu_mev: float,
) -> float:
    """Return g_delta * Delta - delta_mu.

    Positive: BCS regime (fully gapped quasiparticle spectrum).
    Zero:     gapless onset (g_delta * Delta = delta_mu).
    Negative: gapless 2SC regime.
    """
    return params.g_delta * delta_mev - delta_mu_mev


# ---------------------------------------------------------------------------
# Neutrality result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutralityResiduals:
    """Neutrality derivatives dOmega/dmu_e and dOmega/dmu_8.

    Both residuals are in MeV^3.  The neutrality conditions are:
        d_omega_d_mu_e = 0   (charge neutrality)
        d_omega_d_mu_8 = 0   (color-8 neutrality)

    At Delta=0 these equal the analytic expressions:
        d_omega_d_mu_e = 2/3*(n_ur+n_ug+n_ub) - 1/3*(n_dr+n_dg+n_db) - n_e
        d_omega_d_mu_8 = -1/3*(n_ur+n_ug+n_dr+n_dg) + 2/3*(n_ub+n_db)
    """

    d_omega_d_mu_e: float  # MeV^3; zero at charge neutrality
    d_omega_d_mu_8: float  # MeV^3; zero at color neutrality


@dataclass(frozen=True)
class NeutralitySolution:
    """Result of solve_neutrality_fixed_fields."""

    mu_e_mev: float
    mu_8_mev: float
    residual_e: float    # MeV^3; should be ~0
    residual_8: float    # MeV^3; should be ~0
    residual_norm: float # sqrt(re^2 + r8^2), MeV^3
    success: bool
    message: str


@dataclass(frozen=True)
class QMDStellarState:
    """Neutral stellar QMD equilibrium state at fixed mu_q.

    The state is obtained by minimizing the neutral grand potential over
    (phi, Delta), with (mu_e, mu_8) solved by fixed-field neutrality at each
    trial point.
    """

    mu_q_mev: float
    phi_mev: float
    delta_mev: float
    gap_mev: float
    mu_e_mev: float
    mu_8_mev: float
    delta_mu_mev: float
    gap_minus_delta_mu_mev: float
    omega_min_mev4: float
    neutrality_residual_e: float
    neutrality_residual_8: float
    neutrality_residual_norm: float
    phase: str
    success: bool
    message: str


@dataclass(frozen=True)
class QMDStellarEoSPoint:
    """Single point on the neutral QMD stellar EoS branch.

    Pressure is vacuum-subtracted from the neutral equilibrium grand potential:
        P(mu_q) = -(Omega_min(mu_q) - Omega_vac).
    Densities and speed of sound are numerical derivatives along the neutral
    branch with mu_q as the independent variable.
    """

    mu_q_mev: float
    mu_B_mev: float
    phi_mev: float
    delta_mev: float
    gap_mev: float
    mu_e_mev: float
    mu_8_mev: float
    delta_mu_mev: float
    gap_minus_delta_mu_mev: float
    pressure_mev4: float
    quark_density_mev3: float
    baryon_density_mev3: float
    energy_density_mev4: float
    cs2: float
    omega_min_mev4: float
    phase: str
    success: bool
    neutrality_residual_norm: float


def build_qmd_stellar_eos_from_states(
    states: list[QMDStellarState],
    omega_vac_mev4: float,
) -> list[QMDStellarEoSPoint]:
    """Build raw neutral QMD stellar EoS points from equilibrium states.

    The returned list has the same length/order as the input states.  It does
    not filter the branch; callers can remove non-physical or derivative-bad
    points according to their workflow.
    """
    if not states:
        return []

    ordered = sorted(states, key=lambda s: s.mu_q_mev)
    mu = np.array([s.mu_q_mev for s in ordered], dtype=float)
    omega = np.array([s.omega_min_mev4 for s in ordered], dtype=float)
    pressure = -(omega - float(omega_vac_mev4))

    if len(ordered) >= 2:
        edge_order = 2 if len(ordered) >= 3 else 1
        n_q = np.gradient(pressure, mu, edge_order=edge_order)
    else:
        n_q = np.array([float("nan")])

    n_b = n_q / 3.0
    epsilon = -pressure + mu * n_q

    if len(ordered) >= 2:
        edge_order = 2 if len(ordered) >= 3 else 1
        dpressure_dmu = np.gradient(pressure, mu, edge_order=edge_order)
        depsilon_dmu = np.gradient(epsilon, mu, edge_order=edge_order)
        with np.errstate(divide="ignore", invalid="ignore"):
            cs2 = dpressure_dmu / depsilon_dmu
        bad = ~np.isfinite(dpressure_dmu) | ~np.isfinite(depsilon_dmu) | (depsilon_dmu == 0.0)
        cs2 = np.where(bad, np.nan, cs2)
    else:
        cs2 = np.array([float("nan")])

    return [
        QMDStellarEoSPoint(
            mu_q_mev=s.mu_q_mev,
            mu_B_mev=3.0 * s.mu_q_mev,
            phi_mev=s.phi_mev,
            delta_mev=s.delta_mev,
            gap_mev=s.gap_mev,
            mu_e_mev=s.mu_e_mev,
            mu_8_mev=s.mu_8_mev,
            delta_mu_mev=s.delta_mu_mev,
            gap_minus_delta_mu_mev=s.gap_minus_delta_mu_mev,
            pressure_mev4=float(pressure[i]),
            quark_density_mev3=float(n_q[i]),
            baryon_density_mev3=float(n_b[i]),
            energy_density_mev4=float(epsilon[i]),
            cs2=float(cs2[i]),
            omega_min_mev4=s.omega_min_mev4,
            phase=s.phase,
            success=s.success,
            neutrality_residual_norm=s.neutrality_residual_norm,
        )
        for i, s in enumerate(ordered)
    ]


# ---------------------------------------------------------------------------
# QMD stellar grand potential
# ---------------------------------------------------------------------------


class QMDStellarModel:
    """QMD grand potential for stellar matter at general (mu_q, mu_e, mu_8).

    Grand potential:
        Omega = Omega_trunc(phi, Delta; mu_bar)
              + Omega^{mu,T=0}
              + Omega_1_num
              + electron_gp(mu_e)

    where mu_bar = mu_q - mu_e/6 + mu_8/3 enters only the Delta^2 coefficient
    of Omega_trunc.  At Delta=0, Omega^{mu,T=0} is the sum over all six
    color-flavor Fermi seas.  At Delta>0, it is the two blue-quark Fermi
    seas plus possible gapless paired quasiparticle modes; the finite
    red-green BCS residual is Omega_1_num.

    Reduction guarantees:
      mu_e = mu_8 = 0   →  omega(phi, Delta, mu_q, 0, 0) == QMDSimpleModel.omega(phi, Delta, mu_q)
      Delta = 0          →  medium term = sum_fc gp(mu_fc, M_q, 2) + electron_gp(mu_e)
                            (correct individual chemical-potential routing, not a common mu)
    """

    def __init__(self, params: QMDParameters) -> None:
        self.params = params
        self._pre: _QMDPrecomputed = _make_precomputed(params)

    def omega(
        self,
        phi_mev: float,
        delta_mev: float,
        mu_q_mev: float,
        mu_e_mev: float,
        mu_8_mev: float,
    ) -> float:
        """Full QMD stellar grand potential in MeV^4.

        Parameters
        ----------
        phi_mev : float
            Sigma condensate field in MeV.
        delta_mev : float
            Diquark condensate field in MeV.
        mu_q_mev : float
            Average quark chemical potential (= mu_B / 3) in MeV.
        mu_e_mev : float
            Electron chemical potential in MeV.
        mu_8_mev : float
            Color-8 chemical potential in MeV.
        """
        # mu_bar enters only the Delta^2 coefficient of Omega_trunc
        mu_bar = mu_q_mev - mu_e_mev / 6.0 + mu_8_mev / 3.0
        trunc = _omega_trunc_core(phi_mev, delta_mev, mu_bar, self.params, self._pre)

        M_q = self.params.g * phi_mev
        mus = quark_chemical_potentials(mu_q_mev, mu_e_mev, mu_8_mev)

        if not self.params.include_medium_term:
            med = 0.0
        elif delta_mev <= _FIELD_ZERO_TOL:
            # Normal phase: all colors/flavors are ordinary free Fermi seas.
            med = sum(
                fermion_grand_potential(mu_fc, M_q, degeneracy=2.0)
                for mu_fc in mus.values()
            )
        else:
            pair = pairing_chemical_potentials(mu_q_mev, mu_e_mev, mu_8_mev)
            med = (
                fermion_grand_potential(mus["mu_ub"], M_q, degeneracy=2.0)
                + fermion_grand_potential(mus["mu_db"], M_q, degeneracy=2.0)
                + omega_qmd_paired_gapless_t0(
                    phi_mev,
                    delta_mev,
                    pair["mu_bar"],
                    pair["delta_mu"],
                    self.params,
                )
            )

        elec = electron_grand_potential(mu_e_mev)
        omega1 = omega_1_num(phi_mev, delta_mev, mu_bar, self.params)

        return float(trunc + med + omega1 + elec)

    def neutrality_residuals(
        self,
        phi_mev: float,
        delta_mev: float,
        mu_q_mev: float,
        mu_e_mev: float,
        mu_8_mev: float,
        *,
        step_mev: float = 0.1,
    ) -> NeutralityResiduals:
        """Analytic dOmega/dmu_e and dOmega/dmu_8 in MeV^3.

        ``step_mev`` is accepted for backward-compatible call sites but is no
        longer used.  The residuals are built from the same Fermi-gas,
        Omega_1_num, and gapless-mode derivatives that define omega().

        The residuals vanish at charge- and color-neutral configurations:
            d_omega_d_mu_e = 0  →  charge neutrality
            d_omega_d_mu_8 = 0  →  color-8 neutrality
        """
        del step_mev

        mus = quark_chemical_potentials(mu_q_mev, mu_e_mev, mu_8_mev)
        M_q = self.params.g * phi_mev
        n_e = electron_number_density(mu_e_mev)

        if delta_mev <= _FIELD_ZERO_TOL:
            if self.params.include_medium_term:
                n = {
                    key: fermion_number_density(mu_fc, M_q, degeneracy=2.0)
                    for key, mu_fc in mus.items()
                }
            else:
                n = {key: 0.0 for key in mus}

            d_mu_e = (
                2.0 / 3.0 * (n["mu_ur"] + n["mu_ug"] + n["mu_ub"])
                - 1.0 / 3.0 * (n["mu_dr"] + n["mu_dg"] + n["mu_db"])
                - n_e
            )
            d_mu_8 = (
                -1.0 / 3.0 * (n["mu_ur"] + n["mu_ug"] + n["mu_dr"] + n["mu_dg"])
                + 2.0 / 3.0 * (n["mu_ub"] + n["mu_db"])
            )
            return NeutralityResiduals(d_omega_d_mu_e=d_mu_e, d_omega_d_mu_8=d_mu_8)

        pair = pairing_chemical_potentials(mu_q_mev, mu_e_mev, mu_8_mev)
        mu_bar = pair["mu_bar"]
        delta_mu = pair["delta_mu"]

        C_safe = max(
            self.params.g**2 * phi_mev**2 + self.params.g_delta**2 * delta_mev**2,
            1.0e-30,
        )
        log_C = np.log(self.params.m_q_mev**2 / C_safe)
        loop_bracket = log_C - self._pre.F_pi - self._pre.G_pi
        d_mubar = -8.0 * mu_bar * (
            1.0 + 4.0 * self.params.g_delta**2 / self._pre.pi16sq * loop_bracket
        ) * delta_mev**2
        d_mubar += omega_1_num_mu_derivative(
            phi_mev,
            delta_mev,
            mu_bar,
            self.params,
        )

        if self.params.include_medium_term:
            n_ub = fermion_number_density(mus["mu_ub"], M_q, degeneracy=2.0)
            n_db = fermion_number_density(mus["mu_db"], M_q, degeneracy=2.0)
            d_gap_mubar, d_delta_mu = omega_qmd_paired_gapless_derivatives_t0(
                phi_mev,
                delta_mev,
                mu_bar,
                delta_mu,
                self.params,
            )
            d_mubar += d_gap_mubar
        else:
            n_ub = 0.0
            n_db = 0.0
            d_delta_mu = 0.0

        d_mu_e = (
            2.0 / 3.0 * n_ub
            - 1.0 / 3.0 * n_db
            - d_mubar / 6.0
            + d_delta_mu / 2.0
            - n_e
        )
        d_mu_8 = 2.0 / 3.0 * (n_ub + n_db) + d_mubar / 3.0
        return NeutralityResiduals(d_omega_d_mu_e=d_mu_e, d_omega_d_mu_8=d_mu_8)

    def solve_neutrality_fixed_fields(
        self,
        phi_mev: float,
        delta_mev: float,
        mu_q_mev: float,
        initial_guess: tuple[float, float] = (0.0, 0.0),
        step_mev: float = 0.1,
    ) -> NeutralitySolution:
        """Solve for (mu_e, mu_8) that enforce charge and color neutrality.

        Fixes phi and Delta and finds the root of
            F(mu_e, mu_8) = [dOmega/dmu_e, dOmega/dmu_8] = [0, 0]

        using scipy.optimize.root (hybr method).  The Jacobian is estimated
        internally by the solver.

        Parameters
        ----------
        phi_mev, delta_mev : float
            Fixed mean fields.
        mu_q_mev : float
            Fixed average quark chemical potential.
        initial_guess : (mu_e_0, mu_8_0)
            Starting point in MeV.  (0, 0) works for the normal phase;
            for the 2SC phase a non-zero mu_e seed (e.g. 50 MeV) may help.
        step_mev : float
            Deprecated compatibility argument; residuals are analytic.
        """
        def residual_vec(x: np.ndarray) -> np.ndarray:
            r = self.neutrality_residuals(
                phi_mev, delta_mev, mu_q_mev, float(x[0]), float(x[1]),
                step_mev=step_mev,
            )
            return np.array([r.d_omega_d_mu_e, r.d_omega_d_mu_8])

        result = root(residual_vec, np.array(initial_guess, dtype=float), method="hybr")

        mu_e_sol = float(result.x[0])
        mu_8_sol = float(result.x[1])
        res = self.neutrality_residuals(
            phi_mev, delta_mev, mu_q_mev, mu_e_sol, mu_8_sol,
            step_mev=step_mev,
        )
        re = res.d_omega_d_mu_e
        r8 = res.d_omega_d_mu_8
        rnorm = math.sqrt(re**2 + r8**2)

        return NeutralitySolution(
            mu_e_mev=mu_e_sol,
            mu_8_mev=mu_8_sol,
            residual_e=re,
            residual_8=r8,
            residual_norm=rnorm,
            success=bool(result.success),
            message=result.message,
        )

    def solve_equilibrium(
        self,
        mu_q_mev: float,
        initial_fields: tuple[float, float] | None = None,
        initial_neutrality_guess: tuple[float, float] = (0.0, 0.0),
        previous_state: QMDStellarState | None = None,
        *,
        step_mev: float = 0.1,
        neutrality_tol_mev3: float = 1.0,
        delta_phase_tol_mev: float = _DELTA_PHASE_TOL,
        minimizer_options: dict | None = None,
    ) -> QMDStellarState:
        """Solve neutral stellar equilibrium at fixed average quark mu_q.

        This uses a nested strategy:
          1. Outer minimization over the mean fields (phi, Delta).
          2. Inner solve of electric and color neutrality for (mu_e, mu_8).
          3. Objective value Omega(phi, Delta, mu_q, mu_e*, mu_8*).

        The field bounds are intentionally conservative:
            phi in [1e-6, 2 f_pi], Delta in [0, f_pi].

        The Delta ceiling keeps the search in the same condensate range used
        by the reference notebook and avoids spurious large-field minima from
        the truncated analytic expansion.
        """
        p = self.params
        fp = p.f_pi_mev
        bounds = [(1.0e-6, 2.0 * fp), (0.0, fp)]

        field_guesses: list[tuple[float, float]] = []
        if previous_state is not None:
            field_guesses.append((previous_state.phi_mev, previous_state.delta_mev))
        if initial_fields is not None:
            field_guesses.append(initial_fields)
        field_guesses.extend([
            (fp, 0.0),
            (0.5 * fp, 0.0),
            (1.0e-3, 0.0),
            (0.5 * fp, 20.0),
            (0.5 * fp, 50.0),
            (1.0e-3, 50.0),
        ])

        if previous_state is not None:
            base_neutrality_guess = (previous_state.mu_e_mev, previous_state.mu_8_mev)
        else:
            base_neutrality_guess = initial_neutrality_guess

        fallback_neutrality_guesses = [
            initial_neutrality_guess,
            (0.0, 0.0),
            (50.0, 0.0),
            (50.0, -20.0),
            (100.0, -50.0),
        ]

        last_neutrality_guess: list[tuple[float, float]] = [base_neutrality_guess]
        objective_cache: dict[tuple[float, float], tuple[float, NeutralitySolution]] = {}

        def unique_guesses(
            guesses: list[tuple[float, float]],
        ) -> list[tuple[float, float]]:
            seen: set[tuple[float, float]] = set()
            out: list[tuple[float, float]] = []
            for mu_e, mu_8 in guesses:
                key = (round(float(mu_e), 10), round(float(mu_8), 10))
                if key in seen:
                    continue
                seen.add(key)
                out.append((float(mu_e), float(mu_8)))
            return out

        def solve_neutrality_with_fallbacks(
            phi_mev: float,
            delta_mev: float,
            preferred_guess: tuple[float, float],
        ) -> NeutralitySolution | None:
            if delta_mev <= delta_phase_tol_mev:
                # At Delta=0 the normal phase is color-symmetric.  In vacuum-like
                # regions the color residual can be flat, so try the canonical
                # mu_8=0 branch before any carried-over 2SC color potential.
                zero_color_guesses = [
                    (preferred_guess[0], 0.0),
                    (base_neutrality_guess[0], 0.0),
                    (initial_neutrality_guess[0], 0.0),
                    (0.0, 0.0),
                    (50.0, 0.0),
                    (100.0, 0.0),
                ]
                raw_guesses = (
                    zero_color_guesses
                    + [preferred_guess, last_neutrality_guess[0], base_neutrality_guess]
                    + fallback_neutrality_guesses
                )
            else:
                raw_guesses = (
                    [preferred_guess, last_neutrality_guess[0], base_neutrality_guess]
                    + fallback_neutrality_guesses
                )

            guesses = unique_guesses(raw_guesses)
            best_sol: NeutralitySolution | None = None
            for guess in guesses:
                try:
                    res0 = self.neutrality_residuals(
                        phi_mev,
                        delta_mev,
                        mu_q_mev,
                        guess[0],
                        guess[1],
                        step_mev=step_mev,
                    )
                    rnorm0 = math.sqrt(
                        res0.d_omega_d_mu_e**2 + res0.d_omega_d_mu_8**2
                    )
                except Exception:
                    rnorm0 = float("inf")

                if delta_mev <= delta_phase_tol_mev and rnorm0 <= neutrality_tol_mev3:
                    sol = NeutralitySolution(
                        mu_e_mev=guess[0],
                        mu_8_mev=guess[1],
                        residual_e=res0.d_omega_d_mu_e,
                        residual_8=res0.d_omega_d_mu_8,
                        residual_norm=rnorm0,
                        success=True,
                        message="Initial neutrality guess satisfies residual tolerance.",
                    )
                    last_neutrality_guess[0] = (sol.mu_e_mev, sol.mu_8_mev)
                    return sol

                try:
                    sol = self.solve_neutrality_fixed_fields(
                        phi_mev,
                        delta_mev,
                        mu_q_mev,
                        initial_guess=guess,
                        step_mev=step_mev,
                    )
                except Exception:
                    continue

                if best_sol is None or sol.residual_norm < best_sol.residual_norm:
                    best_sol = sol

                if sol.residual_norm <= neutrality_tol_mev3:
                    last_neutrality_guess[0] = (sol.mu_e_mev, sol.mu_8_mev)
                    return sol

            if best_sol is not None and best_sol.success:
                last_neutrality_guess[0] = (best_sol.mu_e_mev, best_sol.mu_8_mev)
                return best_sol
            return best_sol

        def neutral_objective(x: np.ndarray) -> float:
            phi = float(x[0])
            delta = float(x[1])
            key = (round(phi, 8), round(delta, 8))
            cached = objective_cache.get(key)
            if cached is not None:
                return cached[0]

            sol = solve_neutrality_with_fallbacks(phi, delta, last_neutrality_guess[0])
            if sol is None or sol.residual_norm > neutrality_tol_mev3:
                penalty = _OBJECTIVE_FAILURE_PENALTY + phi * phi + delta * delta
                if sol is not None:
                    objective_cache[key] = (penalty, sol)
                return penalty

            try:
                omega = self.omega(phi, delta, mu_q_mev, sol.mu_e_mev, sol.mu_8_mev)
            except Exception:
                omega = _OBJECTIVE_FAILURE_PENALTY + phi * phi + delta * delta

            if not np.isfinite(omega):
                omega = _OBJECTIVE_FAILURE_PENALTY + phi * phi + delta * delta

            objective_cache[key] = (float(omega), sol)
            return float(omega)

        result = find_global_minimum(
            neutral_objective,
            bounds=bounds,
            initial_guesses=field_guesses,
            grid_shape=None,
            options=minimizer_options,
        )

        phi_min = float(result.x[0])
        delta_min = float(result.x[1])
        final_sol = solve_neutrality_with_fallbacks(
            phi_min,
            delta_min,
            last_neutrality_guess[0],
        )

        if final_sol is None:
            final_sol = NeutralitySolution(
                mu_e_mev=float("nan"),
                mu_8_mev=float("nan"),
                residual_e=float("nan"),
                residual_8=float("nan"),
                residual_norm=float("inf"),
                success=False,
                message="No neutral solution found at minimized fields.",
            )
            omega_min = float("inf")
        else:
            omega_min = self.omega(
                phi_min,
                delta_min,
                mu_q_mev,
                final_sol.mu_e_mev,
                final_sol.mu_8_mev,
            )

        gap_mev = p.g_delta * delta_min
        delta_mu_mev = pairing_chemical_potentials(
            mu_q_mev,
            final_sol.mu_e_mev,
            final_sol.mu_8_mev,
        )["delta_mu"]
        gap_minus_delta_mu = gapless_diagnostic(delta_min, p, delta_mu_mev)
        phase = "normal" if delta_min <= delta_phase_tol_mev else "2SC"
        neutral_ok = final_sol.residual_norm <= neutrality_tol_mev3
        minimization_ok = result.success or result.n_successful > 0
        success = bool(minimization_ok and neutral_ok and np.isfinite(omega_min))
        message = (
            f"minimizer: {result.message}; "
            f"attempts={result.n_attempts}, successful={result.n_successful}; "
            f"neutrality: {final_sol.message}"
        )

        return QMDStellarState(
            mu_q_mev=mu_q_mev,
            phi_mev=phi_min,
            delta_mev=delta_min,
            gap_mev=gap_mev,
            mu_e_mev=final_sol.mu_e_mev,
            mu_8_mev=final_sol.mu_8_mev,
            delta_mu_mev=delta_mu_mev,
            gap_minus_delta_mu_mev=gap_minus_delta_mu,
            omega_min_mev4=float(omega_min),
            neutrality_residual_e=final_sol.residual_e,
            neutrality_residual_8=final_sol.residual_8,
            neutrality_residual_norm=final_sol.residual_norm,
            phase=phase,
            success=success,
            message=message,
        )
