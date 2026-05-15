"""Thermodynamic derivative utilities shared across EoS models."""

from __future__ import annotations

import numpy as np


def speed_of_sound_squared(
    pressure_mev4: np.ndarray,
    energy_density_mev4: np.ndarray,
    mu_mev: np.ndarray,
    *,
    derivative_floor_relative: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_mev, cs²) on the supplied branch.

    Computes cs² = (dP/dμ) / (dε/dμ) using numpy.gradient.
    Points where dε/dμ is below the floor or non-finite are excluded.

    Parameters
    ----------
    pressure_mev4:
        Pressure array in MeV^4, already filtered to the desired branch.
    energy_density_mev4:
        Energy density array in MeV^4, same ordering as pressure.
    mu_mev:
        Chemical potential array in MeV used as the differentiation variable.
    derivative_floor_relative:
        Fraction of max|dε/dμ| below which points are discarded.
    """
    if mu_mev.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    edge_order = 2 if mu_mev.size >= 3 else 1
    dpressure_dmu = np.gradient(pressure_mev4, mu_mev, edge_order=edge_order)
    denergy_dmu = np.gradient(energy_density_mev4, mu_mev, edge_order=edge_order)

    derivative_floor = derivative_floor_relative * max(1.0, float(np.max(np.abs(denergy_dmu))))
    valid = np.isfinite(dpressure_dmu) & np.isfinite(denergy_dmu) & (np.abs(denergy_dmu) > derivative_floor)
    return mu_mev[valid], dpressure_dmu[valid] / denergy_dmu[valid]
