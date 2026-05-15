"""Zero-pressure surface interpolation for generic EoS arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
