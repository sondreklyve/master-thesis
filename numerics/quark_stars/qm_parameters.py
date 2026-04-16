"""Vacuum-input handling and parameter fits for the two-flavor QM model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import DEFAULT_SIGMA_MAX_MEV, NC, N_FLAVORS, PI


@dataclass(frozen=True)
class QMVacuumInputs:
    """Vacuum inputs for the two-flavor QM model."""

    m_q_mev: float = 300.0
    m_pi_mev: float = 138.0
    f_pi_mev: float = 93.0
    m_sigma_mev: float = 600.0
    sigma_max_mev: float = DEFAULT_SIGMA_MAX_MEV

    def with_m_sigma(self, m_sigma_mev: float) -> "QMVacuumInputs":
        return replace(self, m_sigma_mev=m_sigma_mev)


@dataclass(frozen=True)
class QMFittedParameters:
    """Fitted QM-model parameters derived from the vacuum inputs."""

    vacuum: QMVacuumInputs
    nc: int
    n_flavors: int
    g: float
    lambda_parameter: float
    h_mev3: float
    v_squared: float
    mass_squared_mev2: float
    renormalization_scale_mev: float

    @property
    def m_q_mev(self) -> float:
        return self.vacuum.m_q_mev

    @property
    def m_pi_mev(self) -> float:
        return self.vacuum.m_pi_mev

    @property
    def f_pi_mev(self) -> float:
        return self.vacuum.f_pi_mev

    @property
    def m_sigma_mev(self) -> float:
        return self.vacuum.m_sigma_mev

    @property
    def sigma_max_mev(self) -> float:
        return self.vacuum.sigma_max_mev

    @property
    def m_pi_bar(self) -> float:
        return self.m_pi_mev / self.f_pi_mev

    @property
    def m_sigma_bar(self) -> float:
        return self.m_sigma_mev / self.f_pi_mev

    @property
    def loop_prefactor(self) -> float:
        return self.g**2 * self.nc / (4.0 * PI**2)


def fit_qm_parameters(vacuum: QMVacuumInputs) -> QMFittedParameters:
    """Fit the two-flavor QM-model parameters from the vacuum inputs."""
    g = vacuum.m_q_mev / vacuum.f_pi_mev
    lambda_parameter = 3.0 * (vacuum.m_sigma_mev**2 - vacuum.m_pi_mev**2) / vacuum.f_pi_mev**2
    h_mev3 = vacuum.f_pi_mev * vacuum.m_pi_mev**2
    mass_squared_mev2 = 0.5 * (3.0 * vacuum.m_pi_mev**2 - vacuum.m_sigma_mev**2)
    v_squared = (vacuum.m_sigma_mev**2 - 3.0 * vacuum.m_pi_mev**2) / (
        vacuum.m_sigma_mev**2 - vacuum.m_pi_mev**2
    )
    renormalization_scale_mev = vacuum.m_q_mev / np.sqrt(np.e)
    return QMFittedParameters(
        vacuum=vacuum,
        nc=NC,
        n_flavors=N_FLAVORS,
        g=g,
        lambda_parameter=lambda_parameter,
        h_mev3=h_mev3,
        v_squared=v_squared,
        mass_squared_mev2=mass_squared_mev2,
        renormalization_scale_mev=renormalization_scale_mev,
    )


DEFAULT_QM_VACUUM_INPUTS = QMVacuumInputs()
