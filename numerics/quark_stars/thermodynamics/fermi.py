"""Model-independent T=0 Fermi-gas thermodynamic utilities."""

from __future__ import annotations

import numpy as np

from ..constants import ELECTRON_MASS_MEV, PI


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


def chem_potential_pressure_term(mu_mev: float, mass_mev: float) -> float:
    mu_array = np.asarray(mu_mev)
    mass_array = np.asarray(mass_mev)
    mask = mu_array > mass_array
    p_f = np.where(mask, np.sqrt(mu_array**2 - mass_array**2), 0.0)
    result = (
        (p_f**3 * mu_mev) / 3.0
        + 0.5 * mass_array**4 * np.where(mask, np.log(p_f / mass_array + mu_array / mass_array), 0.0)
        - 0.5 * mass_array**2 * p_f * mu_array
    )
    result = np.where(mask, result, 0.0)
    if np.ndim(result) == 0:
        return float(result)
    return result


def chem_potential_energy_term(mu_mev: float, mass_mev: float) -> float:
    mu_array = np.asarray(mu_mev)
    mass_array = np.asarray(mass_mev)
    mask = mu_array > mass_array
    p_f = np.where(mask, np.sqrt(mu_array**2 - mass_array**2), 0.0)
    result = (
        p_f**3 * mu_mev
        - 0.5 * mass_array**4 * np.where(mask, np.log(p_f / mass_array + mu_array / mass_array), 0.0)
        + 0.5 * mass_array**2 * p_f * mu_array
    )
    result = np.where(mask, result, 0.0)
    if np.ndim(result) == 0:
        return float(result)
    return result
