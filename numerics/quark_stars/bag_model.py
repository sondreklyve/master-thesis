"""Compatibility re-exports from thermodynamics.interpolation and thermodynamics.vacuum.

This module is kept so that existing imports continue to work unchanged.
New code should import directly from the thermodynamics sub-packages.
"""

from .thermodynamics.interpolation import (  # noqa: F401
    ZeroPressureSurface,
    _finite_density_mask,
    interpolate_zero_pressure_surface,
)
from .thermodynamics.vacuum import (  # noqa: F401
    b_mev4_from_root_mev,
    b_root_mev_from_b_mev4,
    bag_metadata,
    minimum_bag_constant_mev4,
    vacuum_subtraction_b0_mev4,
)
