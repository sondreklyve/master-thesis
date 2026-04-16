"""Physical constants and unit conversions used by the quark-star module."""

from __future__ import annotations

import scipy.constants as sc

NC = 3
N_FLAVORS = 2

PI = sc.pi
HBAR = sc.hbar
C = sc.c
FM = 1.0e-15

MEV_TO_J = 1.0e6 * sc.eV
GEV_TO_J = 1.0e9 * sc.eV

MEV4_TO_GEV_FM3 = MEV_TO_J**4 * FM**3 / (HBAR * C) ** 3 / GEV_TO_J
GEV_FM3_TO_MEV4 = 1.0 / MEV4_TO_GEV_FM3

MEV3_TO_FM_MINUS3 = MEV_TO_J**3 * FM**3 / (HBAR * C) ** 3
FM_MINUS3_TO_MEV3 = 1.0 / MEV3_TO_FM_MINUS3

IRON_ENERGY_PER_BARYON_MEV = 931.0
ELECTRON_MASS_MEV = 0.511

DEFAULT_SIGMA_MAX_MEV = 200.0
DEFAULT_NUMERIC_EPS = 1.0e-10
