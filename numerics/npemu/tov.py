# tov.py

import numpy as np

from eos_models import (
    build_eos_uniform,
    build_eos_rmf_plus_crust,
    build_eos_polytrope_crust,
)
from core import (
    solve_composition,
    make_eos,
    dMdr,
    dPdr,
    e0,
    MeV4togcm3,
    dynetoMeV4,
)


def run_tov(EoS, Pcstart, Pcend, Pcstep, tol, r_max=30.0, rstep=0.01,
            integrator="euler"):
    """
    Integrate the TOV equations for a sequence of central pressures.

    Parameters
    ----------
    EoS : callable
        Interpolator epsilon(P) in units of e0 (dimensionless).
    Pcstart, Pcend : float
        Minimum and maximum central pressures (dimensionless).
    Pcstep : float
        Multiplicative step factor for Pc (e.g. 1.02).
    tol : float
        Surface pressure cutoff (dimensionless).
    r_max : float
        Maximum radius in km.
    rstep : float
        Radial step in km.
    integrator : str
        "euler" (default, forward Euler) or "rk4" (classical 4th-order
        Runge-Kutta with one Euler bootstrap step from the origin).

    Returns
    -------
    result : dict
        Keys:
          - 'Mlist', 'Rlist'              : full MR sequence
          - 'centralpressures'            : Pc (dimensionless)
          - 'centraldensities'            : eps_c (g/cm^3)
          - 'StableM', 'StableR'          : stable branch
          - 'UnstableM', 'UnstableR'      : unstable branch
          - 'Stablecd', 'Unstablecd'      : central densities on branches
          - 'stable_mask', 'unstable_mask': boolean masks on full arrays
          - 'idx_max'                     : index of maximum mass
    """
    Mlist = []
    Rlist = []
    centralpressures = []

    Pc = Pcstart
    while Pc < Pcend:
        P = Pc
        M = 0.0
        r = 0.0
        centralpressures.append(Pc)

        if integrator == "rk4":
            # Bootstrap: one Euler step from r=0 to r=rstep (avoids 1/r² singularity)
            r += rstep
            _e = float(EoS(P))
            M += rstep * dMdr(r, _e)
            P += rstep * dPdr(r, M, P, _e)
            # Classical RK4
            while P > tol and r < r_max:
                e1 = float(EoS(P))
                k1M = dMdr(r, e1)
                k1P = dPdr(r, M, P, e1)

                _P2 = P + 0.5 * rstep * k1P
                _M2 = M + 0.5 * rstep * k1M
                e2 = float(EoS(max(_P2, 0.0)))
                k2M = dMdr(r + 0.5 * rstep, e2)
                k2P = dPdr(r + 0.5 * rstep, _M2, max(_P2, 0.0), e2)

                _P3 = P + 0.5 * rstep * k2P
                _M3 = M + 0.5 * rstep * k2M
                e3 = float(EoS(max(_P3, 0.0)))
                k3M = dMdr(r + 0.5 * rstep, e3)
                k3P = dPdr(r + 0.5 * rstep, _M3, max(_P3, 0.0), e3)

                _P4 = P + rstep * k3P
                _M4 = M + rstep * k3M
                e4 = float(EoS(max(_P4, 0.0)))
                k4M = dMdr(r + rstep, e4)
                k4P = dPdr(r + rstep, _M4, max(_P4, 0.0), e4)

                M += rstep * (k1M + 2.0 * k2M + 2.0 * k3M + k4M) / 6.0
                P += rstep * (k1P + 2.0 * k2P + 2.0 * k3P + k4P) / 6.0
                r += rstep
        else:
            # Forward Euler (original scheme)
            while P > tol and r < r_max:
                r += rstep
                eps = float(EoS(P))  # dimensionless epsilon/e0
                M += rstep * dMdr(r, eps)
                P += rstep * dPdr(r, M, P, eps)

        Rlist.append(r)
        Mlist.append(M)
        Pc *= Pcstep

    Mlist = np.array(Mlist)
    Rlist = np.array(Rlist)
    centralpressures = np.array(centralpressures)

    # Maximum mass index
    idx_max = int(np.argmax(Mlist))

    StableM = Mlist[: idx_max + 1]
    StableR = Rlist[: idx_max + 1]
    UnstableM = Mlist[idx_max + 1 :]
    UnstableR = Rlist[idx_max + 1 :]

    # Central densities in cgs
    centraldensities = np.array([float(EoS(Pc)) * e0 * MeV4togcm3 for Pc in centralpressures])
    Stablecd = centraldensities[: idx_max + 1]
    Unstablecd = centraldensities[idx_max + 1 :]

    # Boolean masks for plotting convenience
    stable_mask = np.zeros_like(Mlist, dtype=bool)
    stable_mask[: idx_max + 1] = True
    unstable_mask = ~stable_mask

    return dict(
        Mlist=Mlist,
        Rlist=Rlist,
        centralpressures=centralpressures,
        centraldensities=centraldensities,
        StableM=StableM,
        StableR=StableR,
        UnstableM=UnstableM,
        UnstableR=UnstableR,
        Stablecd=Stablecd,
        Unstablecd=Unstablecd,
        stable_mask=stable_mask,
        unstable_mask=unstable_mask,
        idx_max=idx_max,
    )


def _augment_eos_dict(eos_dict):
    """
    Take an EoS dict from eos_models.* and attach unified fields:

        eps_dimless : epsilon/e0
        P_dimless   : P/e0
        rho         : epsilon in g/cm^3
        P           : pressure in dyn/cm^2

    """
    out = dict(eos_dict)  # shallow copy

    eps = np.asarray(out["joined_energy"])
    P = np.asarray(out["joined_pressure"])

    rho_cgs = eps * e0 * MeV4togcm3
    P_cgs = P * e0 / dynetoMeV4

    out["eps_dimless"] = eps
    out["P_dimless"] = P
    out["rho"] = rho_cgs
    out["P"] = P_cgs

    return out


def run_model(
    model_name: str,
    gamma_crust: float = 1.2,
):
    """
    Driver for npeμ RMF models.
    """
    comp = solve_composition()
    eps_dimless, P_dimless = make_eos(
        comp["rhos"],
        comp["rhons"],
        comp["gsigmas"],
        comp["kes"],
        comp["kmuons"],
    )

    # Uniform npeμ EoS (no crust)
    eos_uniform_raw = build_eos_uniform(eps_dimless, P_dimless)
    eos_uniform = _augment_eos_dict(eos_uniform_raw)

    name = model_name.lower()
    if name in ("rmf_crust", "npemu", "crust"):
        eos_main_raw = build_eos_rmf_plus_crust(eps_dimless, P_dimless)
    elif name in ("polytrope", "poly_tail", "rmf_polytrope"):
        eos_main_raw = build_eos_polytrope_crust(eps_dimless, P_dimless, gamma_crust=gamma_crust)
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    eos_main = _augment_eos_dict(eos_main_raw)

    joined_P = eos_main["P_dimless"]

    Pcstart = 4e33 * dynetoMeV4 / e0
    Pcend   = joined_P[-2]
    Pcstep  = 1.05
    tol     = joined_P[0]

    tov_main = run_tov(
        eos_main["EoS"],
        Pcstart=Pcstart,
        Pcend=Pcend,
        Pcstep=Pcstep,
        tol=tol,
        r_max=30.0,
        rstep=0.01,
    )

    res = {
        "eos_main": eos_main,
        "eos_uniform": eos_uniform,
        "R_main": tov_main["Rlist"],
        "M_main": tov_main["Mlist"],
        "stable_main": tov_main["stable_mask"],
        "unstable_main": tov_main["unstable_mask"],
        "ec_main": tov_main["centraldensities"],
        "M_ec_main": tov_main["Mlist"],
        "model": model_name,
        "gamma_crust": gamma_crust,
        "comp": comp,
    }

    return res
