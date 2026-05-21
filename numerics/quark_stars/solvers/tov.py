"""Bridge a valid stellar EoS to the existing npemu TOV solver."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from ..constants import MEV4_TO_GEV_FM3
from ..io import save_table
from ..qm_stellar_matter import QuarkMatterEOS
from ..thermodynamics.vacuum import bag_metadata


def _npemu_directory() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "npemu"


def _load_npemu_modules():
    npemu_dir = _npemu_directory()
    if str(npemu_dir) not in sys.path:
        sys.path.insert(0, str(npemu_dir))
    import core as npemu_core  # type: ignore
    import tov as npemu_tov  # type: ignore

    return npemu_core, npemu_tov


@dataclass(frozen=True)
class MassRadiusSequence:
    m_sigma_mev: float
    b0_mev4: float
    b_mev4: float
    b_min_mev4: float | None
    central_pressure_dimless: np.ndarray
    central_energy_density_mev4: np.ndarray
    radius_km: np.ndarray
    mass_msun: np.ndarray
    stable_mask: np.ndarray

    @property
    def central_energy_density_gev_fm3(self) -> np.ndarray:
        return self.central_energy_density_mev4 * MEV4_TO_GEV_FM3

    def save(self, path: Path) -> None:
        metadata = {
            "pipeline": "quark_stars",
            "product": "mass_radius",
            "m_sigma_mev": f"{self.m_sigma_mev:.6f}",
            "units": "central pressure is npemu-dimensionless; central energy density in MeV^4/GeV fm^-3; radius in km; mass in Msun",
        }
        metadata.update(bag_metadata(b0_mev4=self.b0_mev4, b_mev4=self.b_mev4, b_min_mev4=self.b_min_mev4))
        data = np.column_stack(
            [
                self.central_pressure_dimless,
                self.central_energy_density_mev4,
                self.central_energy_density_gev_fm3,
                self.radius_km,
                self.mass_msun,
                self.stable_mask.astype(int),
            ]
        )
        save_table(
            path,
            [
                "Pc_dimless",
                "epsilon_c_mev4",
                "epsilon_c_gev_fm-3",
                "radius_km",
                "mass_msun",
                "stable_flag",
            ],
            data,
            metadata,
        )


def build_npemu_energy_from_pressure(eos: QuarkMatterEOS) -> tuple[interp1d, np.ndarray, np.ndarray]:
    npemu_core, _ = _load_npemu_modules()
    pressure_mev4, energy_mev4 = eos.tov_branch()
    pressure_dimless = pressure_mev4 / npemu_core.e0
    energy_dimless = energy_mev4 / npemu_core.e0
    interpolator = interp1d(
        pressure_dimless,
        energy_dimless,
        kind="cubic",
        assume_sorted=True,
        fill_value="extrapolate",
    )
    # Verify monotonicity of cubic spline on the tabulated grid
    e_check = interpolator(pressure_dimless)
    if not np.all(np.diff(e_check) > 0.0):
        n_nonmono = int(np.sum(np.diff(e_check) <= 0.0))
        print(f"  WARNING: cubic EoS interpolant has {n_nonmono} non-monotone step(s); "
              "consider using a finer EoS grid.")
    return interpolator, pressure_dimless, energy_dimless


def run_tov_sequence(
    eos: QuarkMatterEOS,
    central_pressure_factor: float = 1.08,
    radial_step_km: float = 0.01,
    max_radius_km: float = 30.0,
) -> MassRadiusSequence:
    npemu_core, npemu_tov = _load_npemu_modules()
    eos_interp, pressure_dimless, _ = build_npemu_energy_from_pressure(eos)
    positive_pressures = pressure_dimless[pressure_dimless > 0.0]
    if positive_pressures.size < 2:
        raise ValueError("The stellar EoS must contain at least two positive-pressure points for TOV.")

    result = npemu_tov.run_tov(
        eos_interp,
        Pcstart=float(positive_pressures[0] * 1.01),
        Pcend=float(positive_pressures[-1] * 0.999),
        Pcstep=central_pressure_factor,
        tol=0.0,
        r_max=max_radius_km,
        rstep=radial_step_km,
    )
    central_pressure_dimless = np.asarray(result["centralpressures"])
    central_energy_density_mev4 = np.array([float(eos_interp(pc)) * npemu_core.e0 for pc in central_pressure_dimless])
    return MassRadiusSequence(
        m_sigma_mev=eos.m_sigma_mev,
        b0_mev4=eos.b0_mev4,
        b_mev4=eos.b_mev4,
        b_min_mev4=eos.b_min_mev4,
        central_pressure_dimless=central_pressure_dimless,
        central_energy_density_mev4=central_energy_density_mev4,
        radius_km=np.asarray(result["Rlist"]),
        mass_msun=np.asarray(result["Mlist"]),
        stable_mask=np.asarray(result["stable_mask"]),
    )
