"""Andresen-style beta-equilibrated, charge-neutral T=0 stellar quark matter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import root, root_scalar

from .bag_model import interpolate_zero_pressure_surface, vacuum_subtraction_b0_mev4
from .constants import ELECTRON_MASS_MEV, MEV3_TO_FM_MINUS3, MEV4_TO_GEV_FM3, NC, PI
from .io import save_table
from .qm_parameters import QMFittedParameters
from .qm_potential import TwoFlavorQMPotential, fermion_number_density


def _chem_potential_pressure_term(mu_mev: float, mass_mev: float) -> float:
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


def _chem_potential_energy_term(mu_mev: float, mass_mev: float) -> float:
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


def _arr_slice(array: np.ndarray, index_list: list[int]) -> np.ndarray:
    if not index_list:
        return array
    if len(index_list) == 1:
        return array[index_list[0] :]
    return np.concatenate((array[: index_list[0] + 1], array[index_list[1] :]))


def _maxwell_construct(pressure: np.ndarray, energy_density: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    diff_pressure = np.gradient(pressure)
    turning_points: list[int] = []
    for index in range(1, len(diff_pressure)):
        if diff_pressure[index - 1] * diff_pressure[index] < 0.0:
            turning_points.append(index)

    if len(turning_points) == 0:
        return pressure, energy_density, []

    if len(turning_points) == 1:
        index = 1
        while pressure[index] < 0.0:
            index += 1
        surface_energy = np.interp(0.0, pressure[turning_points[0] :], energy_density[turning_points[0] :])
        return (
            np.concatenate((np.array([0.0]), pressure[index:])),
            np.concatenate((np.array([surface_energy]), energy_density[index:])),
            [index - 1],
        )

    def gibbs_area(transition_pressure: float) -> tuple[float, tuple[int, int]]:
        index1 = int(np.argmin(np.abs(transition_pressure - pressure[: turning_points[0]])))
        index2 = int(np.argmin(np.abs(transition_pressure - pressure[turning_points[1] :]))) + turning_points[1]
        area1 = np.trapz(1.0 / energy_density[index1 : turning_points[0]], pressure[index1 : turning_points[0]])
        area2 = np.trapz(
            1.0 / energy_density[turning_points[0] : turning_points[1]],
            pressure[turning_points[0] : turning_points[1]],
        )
        area3 = np.trapz(
            1.0 / energy_density[turning_points[1] : index2 + 1],
            pressure[turning_points[1] : index2 + 1],
        )
        return area1 + area2 + area3, (index1, index2)

    if gibbs_area(0.0)[0] < 0.0:
        index = turning_points[1]
        while pressure[index] < 0.0:
            index += 1
        surface_energy = np.interp(0.0, pressure[turning_points[1] :], energy_density[turning_points[1] :])
        return (
            np.concatenate((np.array([0.0]), pressure[index:])),
            np.concatenate((np.array([surface_energy]), energy_density[index:])),
            [index - 1],
        )

    solution = root_scalar(
        lambda transition_pressure: gibbs_area(transition_pressure)[0],
        method="bisect",
        bracket=(0.0, pressure[turning_points[0]]),
    )
    indices = list(gibbs_area(float(solution.root))[1])
    if pressure[indices[1]] < pressure[indices[0]]:
        return (
            np.concatenate((pressure[: indices[0] + 1], pressure[indices[1] + 1 :])),
            np.concatenate((energy_density[: indices[0] + 1], energy_density[indices[1] + 1 :])),
            [indices[0], indices[1] + 1],
        )
    return (
        np.concatenate((pressure[: indices[0] + 1], pressure[indices[1] :])),
        np.concatenate((energy_density[: indices[0] + 1], energy_density[indices[1] :])),
        indices,
    )


@dataclass(frozen=True)
class StellarMatterTable:
    m_sigma_mev: float
    b0_mev4: float
    b_mev4: float
    bag_source: str
    minimum_b_mev4: float | None
    sigma_mev: np.ndarray
    mu_u_mev: np.ndarray
    mu_d_mev: np.ndarray
    mu_e_mev: np.ndarray
    constituent_mass_mev: np.ndarray
    n_u_mev3: np.ndarray
    n_d_mev3: np.ndarray
    n_e_mev3: np.ndarray
    pressure_raw_mev4: np.ndarray
    energy_density_raw_mev4: np.ndarray
    bag_energy_density_b0_mev4: np.ndarray

    @property
    def mu_q_mev(self) -> np.ndarray:
        return 0.5 * (self.mu_u_mev + self.mu_d_mev)

    @property
    def baryon_density_mev3(self) -> np.ndarray:
        return (self.n_u_mev3 + self.n_d_mev3) / 3.0

    @property
    def pressure_b0_mev4(self) -> np.ndarray:
        return self.pressure_raw_mev4 - self.b0_mev4

    @property
    def energy_density_b0_mev4(self) -> np.ndarray:
        return self.energy_density_raw_mev4 + self.b0_mev4

    @property
    def pressure_mev4(self) -> np.ndarray:
        return self.pressure_b0_mev4 - self.b_mev4

    @property
    def energy_density_mev4(self) -> np.ndarray:
        return self.energy_density_b0_mev4 + self.b_mev4

    @property
    def bag_energy_density_mev4(self) -> np.ndarray:
        return self.bag_energy_density_b0_mev4 + self.b_mev4

    @property
    def pressure_gev_fm3(self) -> np.ndarray:
        return self.pressure_mev4 * MEV4_TO_GEV_FM3

    @property
    def energy_density_gev_fm3(self) -> np.ndarray:
        return self.energy_density_mev4 * MEV4_TO_GEV_FM3

    @property
    def baryon_density_fm3(self) -> np.ndarray:
        return self.baryon_density_mev3 * MEV3_TO_FM_MINUS3

    @property
    def n_u_fm3(self) -> np.ndarray:
        return self.n_u_mev3 * MEV3_TO_FM_MINUS3

    @property
    def n_d_fm3(self) -> np.ndarray:
        return self.n_d_mev3 * MEV3_TO_FM_MINUS3

    @property
    def n_e_fm3(self) -> np.ndarray:
        return self.n_e_mev3 * MEV3_TO_FM_MINUS3

    def genuine_surface(self):
        return interpolate_zero_pressure_surface(self.pressure_mev4, self.energy_density_mev4, self.baryon_density_mev3)

    def bag_surface(self):
        return interpolate_zero_pressure_surface(
            self.pressure_mev4,
            self.bag_energy_density_mev4,
            self.baryon_density_mev3,
        )

    def tov_branch(self) -> tuple[np.ndarray, np.ndarray]:
        surface = self.genuine_surface()
        positive_mask = self.pressure_mev4 > 0.0
        if not np.any(positive_mask):
            raise ValueError("The bag-shifted stellar EoS has no positive-pressure dense branch for TOV.")
        pressure = np.concatenate(([0.0], self.pressure_mev4[positive_mask]))
        energy = np.concatenate(([surface.energy_density_mev4], self.energy_density_mev4[positive_mask]))
        order = np.argsort(pressure)
        pressure = pressure[order]
        energy = energy[order]
        unique = np.concatenate(([True], np.diff(pressure) > 0.0))
        return pressure[unique], energy[unique]

    def save(self, path: Path) -> None:
        data = np.column_stack(
            [
                self.mu_q_mev,
                self.mu_u_mev,
                self.mu_d_mev,
                self.mu_e_mev,
                self.sigma_mev,
                self.constituent_mass_mev,
                self.n_u_fm3,
                self.n_d_fm3,
                self.n_e_fm3,
                self.baryon_density_fm3,
                self.pressure_b0_mev4,
                self.pressure_mev4,
                self.pressure_gev_fm3,
                self.energy_density_b0_mev4,
                self.energy_density_mev4,
                self.energy_density_gev_fm3,
            ]
        )
        metadata = {
            "pipeline": "stellar",
            "m_sigma_mev": f"{self.m_sigma_mev:.6f}",
            "beta_equilibrium": "true",
            "charge_neutrality": "true",
            "construction": "Andresen-style sigma scan plus Maxwell construction",
            "B0_applied": "true",
            "B0_mev4": f"{self.b0_mev4:.12e}",
            "B_mev4": f"{self.b_mev4:.12e}",
            "B_source": self.bag_source,
            "B_min_mev4": "None" if self.minimum_b_mev4 is None else f"{self.minimum_b_mev4:.12e}",
            "units": "chemical potentials and sigma in MeV; densities in fm^-3; pressure/energy in MeV^4 and GeV fm^-3",
        }
        save_table(
            path,
            [
                "mu_q_mev",
                "mu_u_mev",
                "mu_d_mev",
                "mu_e_mev",
                "sigma_mev",
                "constituent_mass_mev",
                "n_u_fm-3",
                "n_d_fm-3",
                "n_e_fm-3",
                "n_B_fm-3",
                "pressure_B0_mev4",
                "pressure_total_mev4",
                "pressure_total_gev_fm-3",
                "energy_B0_mev4",
                "energy_total_mev4",
                "energy_total_gev_fm-3",
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
        + NC / (4.0 * PI**2) * _chem_potential_pressure_term(mu_u_mev, constituent_mass)
        + NC / (4.0 * PI**2) * _chem_potential_pressure_term(mu_d_mev, constituent_mass)
        + 1.0 / (4.0 * PI**2) * _chem_potential_pressure_term(mu_e_mev, ELECTRON_MASS_MEV)
    )


def _energy_b0(
    potential: TwoFlavorQMPotential,
    sigma_mev: np.ndarray,
    mu_u_mev: np.ndarray,
    mu_d_mev: np.ndarray,
    include_electrons: bool,
) -> np.ndarray:
    constituent_mass = potential.constituent_mass(sigma_mev)
    mu_e_mev = mu_d_mev - mu_u_mev
    omega_vacuum = potential.vacuum_potential(potential.parameters.f_pi_mev)
    omega_sigma = np.array([potential.vacuum_potential(float(sigma_value)) for sigma_value in sigma_mev])
    electron_term = 0.0
    if include_electrons:
        electron_term = 1.0 / (4.0 * PI**2) * _chem_potential_energy_term(mu_e_mev, ELECTRON_MASS_MEV)
    return (
        omega_sigma
        - omega_vacuum
        + NC / (4.0 * PI**2) * _chem_potential_energy_term(mu_u_mev, constituent_mass)
        + NC / (4.0 * PI**2) * _chem_potential_energy_term(mu_d_mev, constituent_mass)
        + electron_term
    )


def build_stellar_matter(
    potential: TwoFlavorQMPotential,
    sigma_values_mev: np.ndarray,
) -> StellarMatterTable:
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
    energy_b0 = _energy_b0(potential, sigma, mu_u, mu_d, include_electrons=True)
    bag_energy_b0 = _energy_b0(potential, sigma, mu_u, mu_d, include_electrons=False)

    pressure_b0, energy_b0, indices = _maxwell_construct(pressure_b0, energy_b0)
    bag_energy_b0 = _arr_slice(bag_energy_b0, indices)
    sigma = _arr_slice(sigma, indices)
    mu_u = _arr_slice(mu_u, indices)
    mu_d = _arr_slice(mu_d, indices)
    mu_e = _arr_slice(mu_e, indices)
    constituent_mass = _arr_slice(constituent_mass, indices)

    n_u = fermion_number_density(mu_u, constituent_mass, degeneracy=6.0)
    n_d = fermion_number_density(mu_d, constituent_mass, degeneracy=6.0)
    n_e = fermion_number_density(mu_e, ELECTRON_MASS_MEV, degeneracy=2.0)

    omega_vacuum = potential.vacuum_potential(potential.parameters.f_pi_mev)
    b0_mev4 = vacuum_subtraction_b0_mev4(-omega_vacuum)

    return StellarMatterTable(
        m_sigma_mev=potential.parameters.m_sigma_mev,
        b0_mev4=b0_mev4,
        b_mev4=0.0,
        bag_source="B=0",
        minimum_b_mev4=None,
        sigma_mev=sigma,
        mu_u_mev=mu_u,
        mu_d_mev=mu_d,
        mu_e_mev=mu_e,
        constituent_mass_mev=constituent_mass,
        n_u_mev3=n_u,
        n_d_mev3=n_d,
        n_e_mev3=n_e,
        pressure_raw_mev4=pressure_b0 + b0_mev4,
        energy_density_raw_mev4=energy_b0 - b0_mev4,
        bag_energy_density_b0_mev4=bag_energy_b0,
    )
