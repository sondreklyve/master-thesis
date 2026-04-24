"""Bag-constant handling for the stellar quark-matter pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root_scalar

from .constants import IRON_ENERGY_PER_BARYON_MEV, MEV4_TO_GEV_FM3


@dataclass(frozen=True)
class ZeroPressureSurface:
    pressure_mev4: float
    energy_density_mev4: float
    baryon_density_mev3: float

    @property
    def energy_per_baryon_mev(self) -> float:
        return self.energy_density_mev4 / self.baryon_density_mev3


def _finite_density_mask(baryon_density_mev3: np.ndarray) -> np.ndarray:
    return baryon_density_mev3 > 0.0


def interpolate_zero_pressure_surface(
    pressure_mev4: np.ndarray,
    energy_density_mev4: np.ndarray,
    baryon_density_mev3: np.ndarray,
) -> ZeroPressureSurface:
    mask = _finite_density_mask(baryon_density_mev3)
    pressure = pressure_mev4[mask]
    energy = energy_density_mev4[mask]
    density = baryon_density_mev3[mask]
    if pressure.size == 0:
        raise ValueError("The EoS has no finite-density matter branch.")

    zero_indices = np.flatnonzero(np.isclose(pressure, 0.0, atol=1.0e-12 * max(1.0, np.max(np.abs(pressure)))))
    if zero_indices.size:
        index = int(zero_indices[0])
        return ZeroPressureSurface(0.0, float(energy[index]), float(density[index]))

    for index in range(pressure.size - 1):
        p_left = float(pressure[index])
        p_right = float(pressure[index + 1])
        if p_left * p_right < 0.0:
            weight = -p_left / (p_right - p_left)
            energy_surface = float(energy[index] + weight * (energy[index + 1] - energy[index]))
            density_surface = float(density[index] + weight * (density[index + 1] - density[index]))
            return ZeroPressureSurface(0.0, energy_surface, density_surface)

    raise ValueError("No genuine dense-matter zero-pressure crossing was found.")


def vacuum_subtraction_b0_mev4(vacuum_pressure_mev4: float) -> float:
    """Return B0, the vacuum subtraction needed to normalize the vacuum pressure to zero."""
    return vacuum_pressure_mev4


def b_root_mev_from_b_mev4(b_mev4: float) -> float:
    if b_mev4 < 0.0:
        raise ValueError("The bag constant must be non-negative.")
    return b_mev4**0.25


def b_mev4_from_root_mev(b_root_mev: float) -> float:
    if b_root_mev < 0.0:
        raise ValueError("B^(1/4) must be non-negative.")
    return b_root_mev**4


def minimum_bag_constant_mev4(
    pressure_b0_mev4: np.ndarray,
    energy_b0_mev4: np.ndarray,
    baryon_density_mev3: np.ndarray,
    target_energy_per_baryon_mev: float = IRON_ENERGY_PER_BARYON_MEV,
) -> float:
    mask = _finite_density_mask(baryon_density_mev3)
    pressure = pressure_b0_mev4[mask]
    energy = energy_b0_mev4[mask]
    density = baryon_density_mev3[mask]
    if pressure.size == 0:
        raise ValueError("Cannot determine the bag constant without finite-density matter states.")

    lower_bound = max(0.0, float(np.min(pressure)))
    upper_bound = float(np.max(pressure)) * (1.0 - 1.0e-12)
    if upper_bound <= lower_bound:
        raise ValueError("Could not bracket a zero-pressure surface when solving for the physical bag constant.")

    def residual(bag_mev4: float) -> float:
        surface = interpolate_zero_pressure_surface(pressure - bag_mev4, energy + bag_mev4, density)
        return surface.energy_per_baryon_mev - target_energy_per_baryon_mev

    lower_value = residual(lower_bound)
    if lower_value >= 0.0:
        return lower_bound

    upper_value = residual(upper_bound)
    if upper_value < 0.0:
        raise ValueError("The matter-stability criterion was not reached before the dense branch lost positive pressure.")

    solution = root_scalar(residual, bracket=(lower_bound, upper_bound), method="brentq")
    if not solution.converged:
        raise ValueError("Failed to determine the additional bag constant from the physical surface criterion.")
    return float(solution.root)


def bag_metadata(
    *,
    b0_mev4: float,
    b_mev4: float,
    b_min_mev4: float | None,
) -> dict[str, str]:
    metadata = {
        "B0_mev4": f"{b0_mev4:.12e}",
        "B0_gev_fm3": f"{b0_mev4 * MEV4_TO_GEV_FM3:.12e}",
        "B_mev4": f"{b_mev4:.12e}",
        "B_gev_fm3": f"{b_mev4 * MEV4_TO_GEV_FM3:.12e}",
        "B_1_4_mev": f"{b_root_mev_from_b_mev4(b_mev4):.6f}",
    }
    if b_min_mev4 is not None:
        metadata["B_min_mev4"] = f"{b_min_mev4:.12e}"
        metadata["B_min_gev_fm3"] = f"{b_min_mev4 * MEV4_TO_GEV_FM3:.12e}"
        metadata["B_min_1_4_mev"] = f"{b_root_mev_from_b_mev4(b_min_mev4):.6f}"
    return metadata
