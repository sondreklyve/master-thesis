"""Vacuum-consistency scan for the QM model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qm_parameters import DEFAULT_QM_VACUUM_INPUTS, fit_qm_parameters
from .qm_potential import TwoFlavorQMPotential


DEFAULT_M_SIGMA_VALUES_MEV = (300.0, 350.0, 400.0, 500.0, 600.0, 700.0)
DEFAULT_SELECTION_COUNT = 3


@dataclass(frozen=True)
class VacuumScanPoint:
    m_sigma_mev: float
    sigma_grid_mev: np.ndarray
    omega_mev4: np.ndarray
    sigma_min_mev: float
    omega_min_mev4: float
    curvature_mev2: float
    left_rise_mev4: float
    right_rise_mev4: float
    valid: bool
    reason: str


def scan_vacuum_point(
    m_sigma_mev: float,
    *,
    sigma_max_ratio: float = 1.75,
    num_sigma_points: int = 600,
    rise_fraction_threshold: float = 0.01,
    local_window_ratio: float = 0.25,
) -> VacuumScanPoint:
    potential = TwoFlavorQMPotential(fit_qm_parameters(DEFAULT_QM_VACUUM_INPUTS.with_m_sigma(m_sigma_mev)))
    sigma_max_mev = sigma_max_ratio * potential.parameters.f_pi_mev
    sigma_grid_mev = np.linspace(1.0e-3, sigma_max_mev, num_sigma_points)
    omega_mev4 = np.array([potential.vacuum_potential(float(sigma_mev)) for sigma_mev in sigma_grid_mev])

    vacuum_state = potential.find_vacuum_state()
    sigma_min_mev = float(vacuum_state.sigma_mev)
    omega_min_mev4 = float(vacuum_state.grand_potential_mev4)
    sigma_step_mev = float(sigma_grid_mev[1] - sigma_grid_mev[0])
    sigma_min_index = int(np.argmin(np.abs(sigma_grid_mev - sigma_min_mev)))

    local_window_mev = local_window_ratio * potential.parameters.f_pi_mev
    left_sigma_mev = max(sigma_grid_mev[0], sigma_min_mev - local_window_mev)
    right_sigma_mev = min(sigma_grid_mev[-1], sigma_min_mev + local_window_mev)
    left_index = min(len(sigma_grid_mev) - 1, max(0, int(np.searchsorted(sigma_grid_mev, left_sigma_mev, side="left"))))
    right_index = min(len(sigma_grid_mev) - 1, max(0, int(np.searchsorted(sigma_grid_mev, right_sigma_mev, side="right")) - 1))

    if 0 < sigma_min_index < len(sigma_grid_mev) - 1:
        curvature_mev2 = float(
            (omega_mev4[sigma_min_index + 1] - 2.0 * omega_mev4[sigma_min_index] + omega_mev4[sigma_min_index - 1])
            / sigma_step_mev**2
        )
    else:
        curvature_mev2 = float("nan")

    left_rise_mev4 = float(omega_mev4[left_index] - omega_min_mev4)
    right_rise_mev4 = float(omega_mev4[right_index] - omega_min_mev4)
    reference_scale = max(abs(omega_min_mev4), 1.0)
    rise_threshold = rise_fraction_threshold * reference_scale

    if sigma_min_index == 0 or sigma_min_index == len(sigma_grid_mev) - 1:
        valid = False
        reason = "minimum_on_scan_boundary"
    elif sigma_min_mev <= sigma_grid_mev[0] or sigma_min_mev >= sigma_grid_mev[-1]:
        valid = False
        reason = "vacuum_minimum_outside_scan_window"
    elif not np.isfinite(curvature_mev2) or curvature_mev2 <= 0.0:
        valid = False
        reason = "non_positive_curvature"
    elif left_rise_mev4 <= rise_threshold or right_rise_mev4 <= rise_threshold:
        valid = False
        reason = "insufficient_local_rise"
    else:
        f_pi = potential.parameters.f_pi_mev
        if abs(sigma_min_mev - f_pi) > 0.2 * f_pi:
            valid = False
            reason = "minimum_not_near_f_pi"
        else:
            valid = True
            reason = "valid_physical_vacuum_minimum"

    return VacuumScanPoint(
        m_sigma_mev=m_sigma_mev,
        sigma_grid_mev=sigma_grid_mev,
        omega_mev4=omega_mev4,
        sigma_min_mev=sigma_min_mev,
        omega_min_mev4=omega_min_mev4,
        curvature_mev2=curvature_mev2,
        left_rise_mev4=left_rise_mev4,
        right_rise_mev4=right_rise_mev4,
        valid=valid,
        reason=reason,
    )


def scan_vacuum_stability(
    m_sigma_values_mev: list[float] | tuple[float, ...] = DEFAULT_M_SIGMA_VALUES_MEV,
    *,
    sigma_max_ratio: float = 1.75,
    num_sigma_points: int = 600,
    rise_fraction_threshold: float = 0.01,
    local_window_ratio: float = 0.25,
) -> list[VacuumScanPoint]:
    return [
        scan_vacuum_point(
            float(m_sigma_mev),
            sigma_max_ratio=sigma_max_ratio,
            num_sigma_points=num_sigma_points,
            rise_fraction_threshold=rise_fraction_threshold,
            local_window_ratio=local_window_ratio,
        )
        for m_sigma_mev in m_sigma_values_mev
    ]


def select_lowest_valid_m_sigma_values(
    m_sigma_values_mev: list[float] | tuple[float, ...] = DEFAULT_M_SIGMA_VALUES_MEV,
    *,
    selection_count: int = DEFAULT_SELECTION_COUNT,
    sigma_max_ratio: float = 1.75,
    num_sigma_points: int = 600,
    rise_fraction_threshold: float = 0.01,
    local_window_ratio: float = 0.25,
) -> list[float]:
    results = scan_vacuum_stability(
        m_sigma_values_mev,
        sigma_max_ratio=sigma_max_ratio,
        num_sigma_points=num_sigma_points,
        rise_fraction_threshold=rise_fraction_threshold,
        local_window_ratio=local_window_ratio,
    )
    valid_values = [result.m_sigma_mev for result in results if result.valid]
    if len(valid_values) < selection_count:
        raise ValueError(
            f"Vacuum scan found only {len(valid_values)} valid m_sigma values, fewer than the requested {selection_count}."
        )
    return valid_values[:selection_count]
