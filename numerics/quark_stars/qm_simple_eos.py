"""Simple pedagogical T=0 two-flavor QM-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .thermodynamics.derivatives import speed_of_sound_squared
from .thermodynamics.vacuum import vacuum_subtraction_b0_mev4
from .constants import MEV3_TO_FM_MINUS3, MEV4_TO_GEV_FM3
from .io import save_table
from .qm_potential import TwoFlavorQMPotential, fermion_number_density


@dataclass(frozen=True)
class SimpleEOSState:
    mu_q_mev: float
    sigma_mev: float
    constituent_mass_mev: float
    quark_number_density_mev3: float
    pressure_mev4: float
    energy_density_mev4: float


@dataclass(frozen=True)
class SimpleEOSTable:
    m_sigma_mev: float
    b0_mev4: float
    mu_q_mev: np.ndarray
    sigma_mev: np.ndarray
    constituent_mass_mev: np.ndarray
    quark_number_density_mev3: np.ndarray
    pressure_mev4: np.ndarray
    energy_density_mev4: np.ndarray

    @property
    def baryon_density_mev3(self) -> np.ndarray:
        return self.quark_number_density_mev3 / 3.0

    @property
    def quark_number_density_fm3(self) -> np.ndarray:
        return self.quark_number_density_mev3 * MEV3_TO_FM_MINUS3

    @property
    def baryon_density_fm3(self) -> np.ndarray:
        return self.baryon_density_mev3 * MEV3_TO_FM_MINUS3

    @property
    def pressure_gev_fm3(self) -> np.ndarray:
        return self.pressure_mev4 * MEV4_TO_GEV_FM3

    @property
    def energy_density_gev_fm3(self) -> np.ndarray:
        return self.energy_density_mev4 * MEV4_TO_GEV_FM3

    @property
    def positive_pressure_mask(self) -> np.ndarray:
        return self.pressure_mev4 >= 0.0

    def speed_of_sound_squared_branch(self, derivative_floor_relative: float = 1.0e-10) -> tuple[np.ndarray, np.ndarray]:
        mask = self.positive_pressure_mask
        return speed_of_sound_squared(
            self.pressure_mev4[mask],
            self.energy_density_mev4[mask],
            self.mu_q_mev[mask],
            derivative_floor_relative=derivative_floor_relative,
        )

    def save(self, path: Path) -> None:
        data = np.column_stack(
            [
                self.mu_q_mev,
                self.sigma_mev,
                self.constituent_mass_mev,
                self.quark_number_density_mev3,
                self.quark_number_density_fm3,
                self.pressure_mev4,
                self.pressure_gev_fm3,
                self.energy_density_mev4,
                self.energy_density_gev_fm3,
            ]
        )
        metadata = {
            "pipeline": "simple",
            "m_sigma_mev": f"{self.m_sigma_mev:.6f}",
            "B0_applied": "true",
            "B0_mev4": f"{self.b0_mev4:.12e}",
            "B_used_mev4": "0.000000000000e+00",
            "units": "mu/sigma/mass in MeV; densities in MeV^3 and fm^-3; pressure/energy in MeV^4 and GeV fm^-3",
        }
        save_table(
            path,
            [
                "mu_q_mev",
                "sigma_mev",
                "constituent_mass_mev",
                "n_q_mev3",
                "n_q_fm-3",
                "pressure_mev4",
                "pressure_gev_fm-3",
                "energy_density_mev4",
                "energy_density_gev_fm-3",
            ],
            data,
            metadata,
        )


def solve_simple_state(
    potential: TwoFlavorQMPotential,
    mu_q_mev: float,
    initial_sigma_mev: float | None = None,
    b0_mev4: float = 0.0,
) -> SimpleEOSState:
    sigma_mev = potential.minimize_simple_sigma(mu_q_mev, initial_sigma_mev=initial_sigma_mev)
    constituent_mass_mev = potential.constituent_mass(sigma_mev)
    quark_number_density_mev3 = 2.0 * fermion_number_density(mu_q_mev, constituent_mass_mev, degeneracy=2.0 * 3.0)
    omega_mev4 = potential.grand_potential_simple(sigma_mev, mu_q_mev)
    pressure_mev4 = -omega_mev4 - b0_mev4
    energy_density_mev4 = omega_mev4 + mu_q_mev * quark_number_density_mev3 + b0_mev4
    return SimpleEOSState(
        mu_q_mev=mu_q_mev,
        sigma_mev=sigma_mev,
        constituent_mass_mev=constituent_mass_mev,
        quark_number_density_mev3=quark_number_density_mev3,
        pressure_mev4=pressure_mev4,
        energy_density_mev4=energy_density_mev4,
    )


def build_simple_eos(
    potential: TwoFlavorQMPotential,
    mu_q_values_mev: np.ndarray,
) -> SimpleEOSTable:
    vacuum_state = potential.find_vacuum_state()
    b0_mev4 = vacuum_subtraction_b0_mev4(vacuum_state.pressure_mev4)

    states: list[SimpleEOSState] = []
    sigma_guess = vacuum_state.sigma_mev
    for mu_q_mev in mu_q_values_mev:
        state = solve_simple_state(potential, float(mu_q_mev), initial_sigma_mev=sigma_guess, b0_mev4=b0_mev4)
        sigma_guess = state.sigma_mev
        states.append(state)

    return SimpleEOSTable(
        m_sigma_mev=potential.parameters.m_sigma_mev,
        b0_mev4=b0_mev4,
        mu_q_mev=np.array([state.mu_q_mev for state in states]),
        sigma_mev=np.array([state.sigma_mev for state in states]),
        constituent_mass_mev=np.array([state.constituent_mass_mev for state in states]),
        quark_number_density_mev3=np.array([state.quark_number_density_mev3 for state in states]),
        pressure_mev4=np.array([state.pressure_mev4 for state in states]),
        energy_density_mev4=np.array([state.energy_density_mev4 for state in states]),
    )
