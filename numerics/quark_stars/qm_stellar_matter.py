"""Charge-neutral, beta-equilibrated QM equation-of-state builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import root

from .thermodynamics.interpolation import interpolate_zero_pressure_surface
from .thermodynamics.vacuum import bag_metadata, b_root_mev_from_b_mev4, vacuum_subtraction_b0_mev4
from .constants import ELECTRON_MASS_MEV, MEV3_TO_FM_MINUS3, MEV4_TO_GEV_FM3, NC, PI
from .io import save_table
from .qm_potential import TwoFlavorQMPotential, fermion_number_density
from .thermodynamics.fermi import chem_potential_energy_term, chem_potential_pressure_term
from .thermodynamics.maxwell import _arr_slice, maxwell_construct


@dataclass(frozen=True)
class QuarkMatterEOS:
    m_sigma_mev: float
    construction: str
    b0_mev4: float
    b_mev4: float
    b_min_mev4: float | None
    sigma_mev: np.ndarray
    mu_u_mev: np.ndarray
    mu_d_mev: np.ndarray
    mu_e_mev: np.ndarray
    constituent_mass_mev: np.ndarray
    n_u_mev3: np.ndarray
    n_d_mev3: np.ndarray
    n_e_mev3: np.ndarray
    pressure_b0_mev4: np.ndarray
    energy_density_b0_mev4: np.ndarray

    @property
    def baryon_density_mev3(self) -> np.ndarray:
        return (self.n_u_mev3 + self.n_d_mev3) / 3.0

    @property
    def baryon_density_fm3(self) -> np.ndarray:
        return self.baryon_density_mev3 * MEV3_TO_FM_MINUS3

    @property
    def pressure_mev4(self) -> np.ndarray:
        return self.pressure_b0_mev4 - self.b_mev4

    @property
    def energy_density_mev4(self) -> np.ndarray:
        return self.energy_density_b0_mev4 + self.b_mev4

    @property
    def pressure_gev_fm3(self) -> np.ndarray:
        return self.pressure_mev4 * MEV4_TO_GEV_FM3

    @property
    def energy_density_gev_fm3(self) -> np.ndarray:
        return self.energy_density_mev4 * MEV4_TO_GEV_FM3

    @property
    def b_root_mev(self) -> float:
        return b_root_mev_from_b_mev4(self.b_mev4)

    def with_bag_constant(self, b_mev4: float, *, b_min_mev4: float | None = None) -> "QuarkMatterEOS":
        return replace(self, b_mev4=b_mev4, b_min_mev4=b_min_mev4)

    def zero_pressure_surface(self):
        return interpolate_zero_pressure_surface(self.pressure_mev4, self.energy_density_mev4, self.baryon_density_mev3)

    def tov_branch(self) -> tuple[np.ndarray, np.ndarray]:
        surface = self.zero_pressure_surface()
        positive_mask = self.pressure_mev4 > 0.0
        if not np.any(positive_mask):
            raise ValueError("The EoS has no positive-pressure branch for TOV.")
        pressure = np.concatenate(([0.0], self.pressure_mev4[positive_mask]))
        energy = np.concatenate(([surface.energy_density_mev4], self.energy_density_mev4[positive_mask]))
        order = np.argsort(pressure)
        pressure = pressure[order]
        energy = energy[order]
        unique = np.concatenate(([True], np.diff(pressure) > 0.0))
        return pressure[unique], energy[unique]

    def save(self, path: Path) -> None:
        metadata = {
            "pipeline": "quark_stars",
            "construction": self.construction,
            "m_sigma_mev": f"{self.m_sigma_mev:.6f}",
            "units": "chemical potentials and sigma in MeV; densities in fm^-3; pressure and energy density in GeV fm^-3",
        }
        metadata.update(bag_metadata(b0_mev4=self.b0_mev4, b_mev4=self.b_mev4, b_min_mev4=self.b_min_mev4))
        data = np.column_stack(
            [
                self.mu_u_mev,
                self.mu_d_mev,
                self.mu_e_mev,
                self.sigma_mev,
                self.constituent_mass_mev,
                self.n_u_mev3 * MEV3_TO_FM_MINUS3,
                self.n_d_mev3 * MEV3_TO_FM_MINUS3,
                self.n_e_mev3 * MEV3_TO_FM_MINUS3,
                self.baryon_density_fm3,
                self.pressure_gev_fm3,
                self.energy_density_gev_fm3,
            ]
        )
        save_table(
            path,
            [
                "mu_u_mev",
                "mu_d_mev",
                "mu_e_mev",
                "sigma_mev",
                "constituent_mass_mev",
                "n_u_fm-3",
                "n_d_fm-3",
                "n_e_fm-3",
                "n_B_fm-3",
                "pressure_gev_fm-3",
                "energy_density_gev_fm-3",
            ],
            data,
            metadata,
        )


def _solve_mu_pair_for_sigma(
    potential: TwoFlavorQMPotential,
    sigma_mev: float,
    initial_guess_mev: tuple[float, float] | None,
) -> tuple[float, float]:
    constituent_mass_mev = potential.constituent_mass(sigma_mev)
    if initial_guess_mev is None:
        guess = np.array([1.5 * constituent_mass_mev, 2.0 * constituent_mass_mev])
    else:
        guess = np.array(initial_guess_mev)

    def system(mu_pair_mev: np.ndarray) -> np.ndarray:
        mu_u_mev = float(mu_pair_mev[0])
        mu_d_mev = float(mu_pair_mev[1])
        mu_e_mev = mu_d_mev - mu_u_mev
        return np.array(
            [
                potential.gap_residual(sigma_mev, mu_u_mev, mu_d_mev) / potential.parameters.f_pi_mev**3,
                potential.charge_density(sigma_mev, mu_u_mev, mu_d_mev, mu_e_mev) / potential.parameters.f_pi_mev**3,
            ]
        )

    solution = root(system, guess, method="lm")
    if not solution.success:
        raise ValueError(f"Failed to solve beta-equilibrated matter at sigma={sigma_mev:.6f} MeV.")
    return float(solution.x[0]), float(solution.x[1])


def _build_sigma_scan(
    potential: TwoFlavorQMPotential,
    sigma_values_mev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_u_values = np.zeros_like(sigma_values_mev)
    mu_d_values = np.zeros_like(sigma_values_mev)
    guess: tuple[float, float] | None = None
    for index, sigma_mev in enumerate(sigma_values_mev):
        mu_u_mev, mu_d_mev = _solve_mu_pair_for_sigma(potential, float(sigma_mev), guess)
        mu_u_values[index] = mu_u_mev
        mu_d_values[index] = mu_d_mev
        guess = (mu_u_mev, mu_d_mev)
    return sigma_values_mev, mu_u_values, mu_d_values


def _electron_present_mask(
    sigma_mev: np.ndarray,
    mu_u_mev: np.ndarray,
    mu_d_mev: np.ndarray,
    potential: TwoFlavorQMPotential,
) -> np.ndarray:
    constituent_mass = potential.constituent_mass(sigma_mev)
    return (mu_u_mev >= constituent_mass) & (mu_d_mev >= constituent_mass) & ((mu_d_mev - mu_u_mev) >= ELECTRON_MASS_MEV)


def _pressure_b0(
    potential: TwoFlavorQMPotential,
    sigma_mev: np.ndarray,
    mu_u_mev: np.ndarray,
    mu_d_mev: np.ndarray,
) -> np.ndarray:
    constituent_mass = potential.constituent_mass(sigma_mev)
    mu_e_mev = mu_d_mev - mu_u_mev
    omega_vacuum = potential.vacuum_potential(potential.parameters.f_pi_mev)
    omega_sigma = np.array([potential.vacuum_potential(float(sigma_value)) for sigma_value in sigma_mev])
    return (
        -omega_sigma
        + omega_vacuum
        + NC / (4.0 * PI**2) * chem_potential_pressure_term(mu_u_mev, constituent_mass)
        + NC / (4.0 * PI**2) * chem_potential_pressure_term(mu_d_mev, constituent_mass)
        + 1.0 / (4.0 * PI**2) * chem_potential_pressure_term(mu_e_mev, ELECTRON_MASS_MEV)
    )


def _energy_b0(
    potential: TwoFlavorQMPotential,
    sigma_mev: np.ndarray,
    mu_u_mev: np.ndarray,
    mu_d_mev: np.ndarray,
) -> np.ndarray:
    constituent_mass = potential.constituent_mass(sigma_mev)
    mu_e_mev = mu_d_mev - mu_u_mev
    omega_vacuum = potential.vacuum_potential(potential.parameters.f_pi_mev)
    omega_sigma = np.array([potential.vacuum_potential(float(sigma_value)) for sigma_value in sigma_mev])
    return (
        omega_sigma
        - omega_vacuum
        + NC / (4.0 * PI**2) * chem_potential_energy_term(mu_u_mev, constituent_mass)
        + NC / (4.0 * PI**2) * chem_potential_energy_term(mu_d_mev, constituent_mass)
        + 1.0 / (4.0 * PI**2) * chem_potential_energy_term(mu_e_mev, ELECTRON_MASS_MEV)
    )


def build_sigma_values(
    potential: TwoFlavorQMPotential,
    *,
    sigma_min_ratio: float = 0.01,
    sigma_max_ratio: float = 0.9999,
    num_points: int = 450,
) -> np.ndarray:
    f_pi = potential.parameters.f_pi_mev
    return f_pi * np.linspace(sigma_min_ratio, sigma_max_ratio, num_points)


def build_stellar_eos(
    potential: TwoFlavorQMPotential,
    sigma_values_mev: np.ndarray,
    *,
    with_maxwell: bool,
) -> QuarkMatterEOS:
    sigma_scan, mu_u_scan, mu_d_scan = _build_sigma_scan(potential, sigma_values_mev)
    sigma_scan = np.flip(sigma_scan)
    mu_u_scan = np.flip(mu_u_scan)
    mu_d_scan = np.flip(mu_d_scan)

    valid_mask = _electron_present_mask(sigma_scan, mu_u_scan, mu_d_scan, potential)
    sigma = sigma_scan[valid_mask]
    mu_u = mu_u_scan[valid_mask]
    mu_d = mu_d_scan[valid_mask]
    mu_e = mu_d - mu_u
    constituent_mass = potential.constituent_mass(sigma)

    pressure_b0 = _pressure_b0(potential, sigma, mu_u, mu_d)
    energy_b0 = _energy_b0(potential, sigma, mu_u, mu_d)

    if with_maxwell:
        pressure_b0, energy_b0, indices = maxwell_construct(pressure_b0, energy_b0)
        sigma = _arr_slice(sigma, indices)
        mu_u = _arr_slice(mu_u, indices)
        mu_d = _arr_slice(mu_d, indices)
        mu_e = _arr_slice(mu_e, indices)
        constituent_mass = _arr_slice(constituent_mass, indices)

    n_u = fermion_number_density(mu_u, constituent_mass, degeneracy=2.0 * NC)
    n_d = fermion_number_density(mu_d, constituent_mass, degeneracy=2.0 * NC)
    n_e = fermion_number_density(mu_e, ELECTRON_MASS_MEV, degeneracy=2.0)

    vacuum_state = potential.find_vacuum_state()
    b0_mev4 = vacuum_subtraction_b0_mev4(vacuum_state.pressure_mev4)

    return QuarkMatterEOS(
        m_sigma_mev=potential.parameters.m_sigma_mev,
        construction="with_maxwell" if with_maxwell else "without_maxwell",
        b0_mev4=b0_mev4,
        b_mev4=0.0,
        b_min_mev4=None,
        sigma_mev=sigma,
        mu_u_mev=mu_u,
        mu_d_mev=mu_d,
        mu_e_mev=mu_e,
        constituent_mass_mev=constituent_mass,
        n_u_mev3=n_u,
        n_d_mev3=n_d,
        n_e_mev3=n_e,
        pressure_b0_mev4=pressure_b0,
        energy_density_b0_mev4=energy_b0,
    )
