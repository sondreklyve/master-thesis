"""Generate gravitationally-bound QM stellar sequences (B=0) for m_sigma = 400, 500, 600 MeV.

Produces:
  - output/stellar/qm_stars_sigma_{400,500,600}_grav_bound.txt
  - output/stellar/qm_grav_bound_mass_radius.pdf  (new single-panel figure)
  - updates output/stellar/mass_radius_combined.pdf (adds grav-bound overlay)
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..io import output_directory
from ..plotting import apply_plot_style, save_figure, sigma_colors, sigma_label
from ..qm_parameters import DEFAULT_QM_VACUUM_INPUTS, fit_qm_parameters
from ..qm_potential import TwoFlavorQMPotential
from ..qm_stellar_matter import build_sigma_values, build_stellar_eos
from ..solvers.tov import run_tov_sequence, run_tov_sequence_grav_bound
from ..thermodynamics.vacuum import (
    b_mev4_from_root_mev,
    b_root_mev_from_b_mev4,
    minimum_bag_constant_mev4,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
M_SIGMA_VALUES = (400.0, 500.0, 600.0)
BAG_ROOT_OFFSETS_MEV = (-10.0, 0.0, 10.0)

GRAV_COLORS = dict(zip(M_SIGMA_VALUES, sigma_colors(len(M_SIGMA_VALUES))))
GRAV_LABELS = {
    400.0: r"$m_\sigma = 400\,\mathrm{MeV}$",
    500.0: r"$m_\sigma = 500\,\mathrm{MeV}$",
    600.0: r"$m_\sigma = 600\,\mathrm{MeV}$",
}


def _build_eos(m_sigma_mev: float, num_points: int = 600):
    fitted = fit_qm_parameters(DEFAULT_QM_VACUUM_INPUTS.with_m_sigma(m_sigma_mev))
    potential = TwoFlavorQMPotential(fitted)
    sigma_values = build_sigma_values(
        potential,
        sigma_min_ratio=0.01,
        sigma_max_ratio=0.9999,
        num_points=num_points,
    )
    return potential, build_stellar_eos(potential, sigma_values, with_maxwell=True)


def _run_grav_bound(stellar_dir: Path, m_sigma_mev: float) -> dict:
    tag = f"sigma_{int(round(m_sigma_mev))}_grav_bound"
    out_path = stellar_dir / f"qm_stars_{tag}.txt"

    _, eos = _build_eos(m_sigma_mev)

    seq = run_tov_sequence_grav_bound(
        eos,
        central_pressure_factor=1.08,
        radial_step_km=0.01,
        max_radius_km=100.0,
        integrator="rk4",
    )
    seq.save(out_path)

    stable = seq.stable_mask.astype(bool)
    M_s = seq.mass_msun[stable]
    R_s = seq.radius_km[stable]
    M_u = seq.mass_msun[~stable]
    R_u = seq.radius_km[~stable]

    result = {"m_sigma": m_sigma_mev, "M_max": None, "R_max": None, "R_14": None,
              "mass_msun_unstable": M_u, "radius_km_unstable": R_u}
    if M_s.size:
        imax = int(np.argmax(M_s))
        result["M_max"] = float(M_s[imax])
        result["R_max"] = float(R_s[imax])
        print(
            f"m_sigma={m_sigma_mev:.0f} MeV (grav-bound): "
            f"M_max={M_s[imax]:.4f} Msun at R={R_s[imax]:.4f} km"
        )
        if np.any(M_s >= 1.4):
            idx = int(np.argmin(np.abs(M_s - 1.4)))
            result["R_14"] = float(R_s[idx])
            print(f"  R(1.4 Msun) = {R_s[idx]:.4f} km")
        else:
            print("  1.4 Msun not reached on stable branch")

    result["radius_km"] = R_s
    result["mass_msun"] = M_s
    result["mass_msun_unstable"] = M_u
    result["radius_km_unstable"] = R_u
    return result


def _make_single_panel_figure(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for res in results:
        m = res["m_sigma"]
        M_s = res["mass_msun"]
        R_s = res["radius_km"]
        if M_s.size == 0:
            continue
        phys = (M_s > 0.3) & (R_s < 20.0)
        ax.plot(R_s[phys], M_s[phys], color=GRAV_COLORS[m], linewidth=2.2, label=GRAV_LABELS[m])
        if res["M_max"] is not None:
            imax = int(np.argmax(M_s[phys]))
            ax.plot(R_s[phys][imax], M_s[phys][imax], "o", color=GRAV_COLORS[m], ms=6)

        M_u = res["mass_msun_unstable"]
        R_u = res["radius_km_unstable"]
        if M_u.size > 0:
            phys_u = (M_u > 0.3) & (R_u < 20.0)
            ax.plot(R_u[phys_u], M_u[phys_u], color=GRAV_COLORS[m], linewidth=1.6, linestyle="--")

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_xlim(7, 17)
    ax.set_ylim(0.4, 2.5)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved single-panel figure: {out_path}")


def _make_updated_combined_figure(
    stellar_dir: Path,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(M_SIGMA_VALUES), 1, figsize=(7.4, 12.8), sharex=True)
    from ..plotting import sigma_colors, bag_curve_label

    colors = sigma_colors(len(BAG_ROOT_OFFSETS_MEV))

    for ax, m_sigma_mev in zip(axes, M_SIGMA_VALUES):
        _, eos = _build_eos(m_sigma_mev)
        b_min_mev4 = minimum_bag_constant_mev4(
            eos.pressure_b0_mev4,
            eos.energy_density_b0_mev4,
            eos.baryon_density_mev3,
        )
        b_min_root = b_root_mev_from_b_mev4(b_min_mev4)
        bag_roots = [b_min_root + offset for offset in BAG_ROOT_OFFSETS_MEV]

        for bag_root_mev, color in zip(bag_roots, colors):
            bagged_eos = eos.with_bag_constant(
                b_mev4_from_root_mev(bag_root_mev), b_min_mev4=b_min_mev4
            )
            try:
                seq = run_tov_sequence(
                    bagged_eos,
                    central_pressure_factor=1.08,
                    radial_step_km=0.01,
                    max_radius_km=30.0,
                    integrator="rk4",
                )
            except ValueError as exc:
                print(f"  Skipping m_sigma={m_sigma_mev:.0f}, B^1/4={bag_root_mev:.1f}: {exc}")
                continue
            stable = seq.stable_mask.astype(bool)
            unstable = ~stable
            ax.plot(
                seq.radius_km[stable],
                seq.mass_msun[stable],
                color=color,
                linewidth=2.2,
                label=bag_curve_label(bag_root_mev),
            )
            if np.any(stable):
                m_st = seq.mass_msun[stable]
                r_st = seq.radius_km[stable]
                idx = int(np.argmax(m_st))
                ax.plot(r_st[idx], m_st[idx], "o", color=color, ms=6, zorder=5)
            if np.any(unstable):
                ax.plot(
                    seq.radius_km[unstable],
                    seq.mass_msun[unstable],
                    color=color,
                    linewidth=1.6,
                    linestyle="--",
                )

        ax.set_title(rf"$m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$")
        ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
        ax.set_xlim(7, 15)
        ax.set_ylim(0.4, 2.2)
        ax.legend()

    axes[0].set_ylabel(r"Mass $M\;(M_\odot)$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved updated combined figure: {out_path}")


def main() -> None:
    apply_plot_style()
    stellar_dir = output_directory(OUTPUT_DIR, "stellar")

    print("=== Generating gravitationally-bound QM sequences (B=0) ===\n")
    results = []
    for m_sigma_mev in M_SIGMA_VALUES:
        res = _run_grav_bound(stellar_dir, m_sigma_mev)
        results.append(res)

    print("\n=== Building single-panel grav-bound figure ===")
    single_panel_path = stellar_dir / "qm_grav_bound_mass_radius.pdf"
    _make_single_panel_figure(results, single_panel_path)

    print("\n=== Rebuilding clean Bodmer-Witten combined figure ===")
    combined_path = stellar_dir / "mass_radius_combined.pdf"
    _make_updated_combined_figure(stellar_dir, combined_path)

    print("\n=== Summary ===")
    for res in results:
        m = res["m_sigma"]
        Mmax = res["M_max"]
        Rmax = res["R_max"]
        R14 = res["R_14"]
        r14_str = f"{R14:.2f} km" if R14 is not None else "not reached"
        print(f"  m_sigma={m:.0f} MeV: M_max={Mmax:.4f} Msun, R(M_max)={Rmax:.4f} km, R(1.4 Msun)={r14_str}")


if __name__ == "__main__":
    main()
