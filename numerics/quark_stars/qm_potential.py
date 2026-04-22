"""Shared T=0 two-flavor QM-model thermodynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import DEFAULT_NUMERIC_EPS, ELECTRON_MASS_MEV, NC, PI
from .qm_parameters import QMFittedParameters


def _fermi_momentum(mu_mev: float, mass_mev: float) -> float:
    mu_array = np.asarray(mu_mev)
    mass_array = np.asarray(mass_mev)
    squared = np.clip(mu_array**2 - mass_array**2, 0.0, None)
    momentum = np.where(mu_array > mass_array, np.sqrt(squared), 0.0)
    if momentum.ndim == 0:
        return float(momentum)
    return momentum


def fermion_number_density(mu_mev: float, mass_mev: float, degeneracy: float) -> float:
    p_f = _fermi_momentum(mu_mev, mass_mev)
    density = degeneracy * np.asarray(p_f) ** 3 / (6.0 * PI**2)
    if np.ndim(density) == 0:
        return float(density)
    return density


def fermion_grand_potential(mu_mev: float, mass_mev: float, degeneracy: float) -> float:
    mu_array = np.asarray(mu_mev)
    mass_array = np.asarray(mass_mev)
    p_f = np.asarray(_fermi_momentum(mu_array, mass_array))
    ratio = np.divide(p_f, mass_array, out=np.zeros_like(p_f, dtype=float), where=mass_array > 0.0)
    log_term = np.where((p_f > 0.0) & (mass_array > 0.0), np.arcsinh(ratio), 0.0)
    omega = -degeneracy / (48.0 * PI**2) * (
        (2.0 * mu_array**2 - 5.0 * mass_array**2) * mu_array * p_f + 3.0 * mass_array**4 * log_term
    )
    omega = np.where(p_f > 0.0, omega, 0.0)
    if omega.ndim == 0:
        return float(omega)
    return omega


def fermion_energy_density(mu_mev: float, mass_mev: float, degeneracy: float) -> float:
    omega = fermion_grand_potential(mu_mev, mass_mev, degeneracy)
    density = fermion_number_density(mu_mev, mass_mev, degeneracy)
    energy = np.asarray(omega) + np.asarray(mu_mev) * np.asarray(density)
    if energy.ndim == 0:
        return float(energy)
    return energy


def electron_number_density(mu_e_mev: float) -> float:
    return fermion_number_density(mu_e_mev, ELECTRON_MASS_MEV, degeneracy=2.0)


def electron_grand_potential(mu_e_mev: float) -> float:
    return fermion_grand_potential(mu_e_mev, ELECTRON_MASS_MEV, degeneracy=2.0)


def electron_energy_density(mu_e_mev: float) -> float:
    return fermion_energy_density(mu_e_mev, ELECTRON_MASS_MEV, degeneracy=2.0)


@dataclass(frozen=True)
class VacuumState:
    sigma_mev: float
    pressure_mev4: float
    grand_potential_mev4: float


class TwoFlavorQMPotential:
    """Shared vacuum and medium thermodynamics for the two-flavor QM model."""

    def __init__(self, parameters: QMFittedParameters) -> None:
        self.parameters = parameters

    def constituent_mass(self, sigma_mev: float) -> float:
        return self.parameters.g * sigma_mev

    def _loop_r(self, mass_mev: float) -> complex:
        return np.sqrt(4.0 * self.parameters.m_q_mev**2 / mass_mev**2 - 1.0 + 0.0j)

    def _loop_F(self, mass_mev: float) -> float:
        r_value = self._loop_r(mass_mev)
        if abs(r_value) < 1.0e-12:
            return 2.0
        return float(np.real(2.0 - 2.0 * r_value * np.arctan(1.0 / r_value)))

    def _loop_G_pi(self) -> float:
        r_pi = self._loop_r(self.parameters.m_pi_mev)
        prefactor = 4.0 * self.parameters.m_q_mev**2 / (self.parameters.m_pi_mev**2 * r_pi)
        return float(np.real(prefactor * np.arctan(1.0 / r_pi) - 1.0))

    def vacuum_potential(self, sigma_mev: float) -> float:
        p = self.parameters
        x = sigma_mev / p.f_pi_mev
        f_pi = p.f_pi_mev
        f_pi_term = self._loop_F(p.m_pi_mev)
        f_sigma_term = self._loop_F(p.m_sigma_mev)
        g_pi_term = self._loop_G_pi()
        loop_prefactor = p.loop_prefactor
        omega_bar = (
            0.75 * p.m_pi_bar**2 * (1.0 - loop_prefactor * g_pi_term) * x**2
            - 0.25
            * p.m_sigma_bar**2
            * (
                1.0
                + loop_prefactor
                * ((1.0 - 4.0 * p.g**2 / p.m_sigma_bar**2) * f_sigma_term + 4.0 * p.g**2 / p.m_sigma_bar**2 - f_pi_term - g_pi_term)
            )
            * x**2
            + 0.125
            * p.m_sigma_bar**2
            * (
                1.0
                - loop_prefactor
                * (
                    4.0 * p.g**2 / p.m_sigma_bar**2 * np.log(x**2 + DEFAULT_NUMERIC_EPS)
                    - (1.0 - 4.0 * p.g**2 / p.m_sigma_bar**2) * f_sigma_term
                    + f_pi_term
                    + g_pi_term
                )
            )
            * x**4
            - 0.125 * p.m_pi_bar**2 * (1.0 - loop_prefactor * g_pi_term) * x**4
            + 0.75 * p.g**2 * loop_prefactor * x**4
            - p.m_pi_bar**2 * (1.0 - loop_prefactor * g_pi_term) * x
        )
        return float(np.real(omega_bar * f_pi**4))

    def vacuum_derivative(self, sigma_mev: float) -> float:
        p = self.parameters
        x = sigma_mev / p.f_pi_mev
        f_sigma_term = self._loop_F(p.m_sigma_mev)
        f_pi_term = self._loop_F(p.m_pi_mev)
        g_pi_term = self._loop_G_pi()
        loop_prefactor = p.loop_prefactor
        derivative_bar = (
            1.5 * p.m_pi_bar**2 * (1.0 - loop_prefactor * g_pi_term) * x
            - 0.5
            * p.m_sigma_bar**2
            * (
                1.0
                + loop_prefactor
                * ((1.0 - 4.0 * p.g**2 / p.m_sigma_bar**2) * f_sigma_term + 4.0 * p.g**2 / p.m_sigma_bar**2 - f_pi_term - g_pi_term)
            )
            * x
            + 0.5
            * p.m_sigma_bar**2
            * (
                1.0
                - loop_prefactor
                * (
                    4.0 * p.g**2 / p.m_sigma_bar**2 * np.log(x**2 + DEFAULT_NUMERIC_EPS)
                    - (1.0 - 4.0 * p.g**2 / p.m_sigma_bar**2) * f_sigma_term
                    + f_pi_term
                    + g_pi_term
                )
            )
            * x**3
            - 0.5 * p.m_pi_bar**2 * (1.0 - loop_prefactor * g_pi_term) * x**3
            + 2.0 * p.g**2 * loop_prefactor * x**3
            - p.m_pi_bar**2 * (1.0 - loop_prefactor * g_pi_term)
        )
        return float(np.real(derivative_bar * p.f_pi_mev**3))

    def quark_grand_potential(self, sigma_mev: float, mu_mev: float) -> float:
        return fermion_grand_potential(mu_mev, self.constituent_mass(sigma_mev), degeneracy=2.0 * NC)

    def gap_residual(self, sigma_mev: float, mu_u_mev: float, mu_d_mev: float) -> float:
        p = self.parameters
        x = sigma_mev / p.f_pi_mev
        x = max(x, 1.0e-8)

        def medium_term(mu_mev: float) -> float:
            mu_bar = mu_mev / p.f_pi_mev
            threshold = p.g * x
            if mu_bar <= threshold:
                return 0.0
            root_term = np.sqrt(mu_bar**2 - threshold**2)
            log_term = np.log(np.sqrt(mu_bar**2 / threshold**2 - 1.0) + mu_bar / threshold)
            return (p.g**2 * x * p.nc) / (2.0 * PI**2) * (root_term * mu_bar - threshold**2 * log_term)

        residual_bar = self.vacuum_derivative(sigma_mev) / p.f_pi_mev**3
        residual_bar += medium_term(mu_u_mev) + medium_term(mu_d_mev)
        return float(np.real(residual_bar * p.f_pi_mev**3))

    def grand_potential_simple(self, sigma_mev: float, mu_q_mev: float) -> float:
        return self.vacuum_potential(sigma_mev) + 2.0 * self.quark_grand_potential(sigma_mev, mu_q_mev)

    def grand_potential_stellar(
        self,
        sigma_mev: float,
        mu_u_mev: float,
        mu_d_mev: float,
        mu_e_mev: float,
    ) -> float:
        return (
            self.vacuum_potential(sigma_mev)
            + self.quark_grand_potential(sigma_mev, mu_u_mev)
            + self.quark_grand_potential(sigma_mev, mu_d_mev)
            + electron_grand_potential(mu_e_mev)
        )

    def charge_density(self, sigma_mev: float, mu_u_mev: float, mu_d_mev: float, mu_e_mev: float) -> float:
        mass_mev = self.constituent_mass(sigma_mev)
        n_u = fermion_number_density(mu_u_mev, mass_mev, degeneracy=2.0 * NC)
        n_d = fermion_number_density(mu_d_mev, mass_mev, degeneracy=2.0 * NC)
        n_e = electron_number_density(mu_e_mev)
        return 2.0 / 3.0 * n_u - 1.0 / 3.0 * n_d - n_e

    def find_vacuum_state(self) -> VacuumState:
        result = minimize_scalar(
            self.vacuum_potential,
            bounds=(1.0e-6, self.parameters.sigma_max_mev),
            method="bounded",
        )
        sigma_mev = float(result.x)
        omega_mev4 = self.vacuum_potential(sigma_mev)
        return VacuumState(
            sigma_mev=sigma_mev,
            pressure_mev4=-omega_mev4,
            grand_potential_mev4=omega_mev4,
        )

    def minimize_simple_sigma(self, mu_q_mev: float, initial_sigma_mev: float | None = None) -> float:
        result = minimize_scalar(
            lambda sigma: self.grand_potential_simple(float(sigma), mu_q_mev),
            bounds=(1.0e-6, self.parameters.sigma_max_mev),
            method="bounded",
        )
        return float(result.x)
