"""Maxwell equal-area construction for first-order phase transitions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar


def _arr_slice(array: np.ndarray, index_list: list[int]) -> np.ndarray:
    if not index_list:
        return array
    if len(index_list) == 1:
        return array[index_list[0] :]
    return np.concatenate((array[: index_list[0] + 1], array[index_list[1] :]))


def maxwell_construct(pressure: np.ndarray, energy_density: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    diff_pressure = np.gradient(pressure)
    turning_points: list[int] = []
    for index in range(1, len(diff_pressure)):
        if diff_pressure[index - 1] * diff_pressure[index] < 0.0:
            turning_points.append(index)

    if len(turning_points) == 0:
        return pressure, energy_density, []

    if len(turning_points) == 1:
        index = 1
        while index < len(pressure) and pressure[index] < 0.0:
            index += 1
        if index == len(pressure):
            raise ValueError("The Maxwell construction removed the entire positive-pressure branch.")
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
        while index < len(pressure) and pressure[index] < 0.0:
            index += 1
        if index == len(pressure):
            raise ValueError("The Maxwell construction removed the entire positive-pressure branch.")
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
