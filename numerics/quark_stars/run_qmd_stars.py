"""Generate QMD stellar mass-radius curves from the stable neutral EoS.

Reads the stable EoS produced by run_qmd_stellar_eos.py for SET A and
(if available) SET B, passes them to the TOV solver, and writes:
  output/qmd_stellar/qmd_stars_set_a.txt
  output/qmd_stellar/qmd_stars_set_b.txt  (if SET B EoS exists)
  output/qmd_stellar/qmd_mass_radius_set_a.pdf  (old name kept; SET A only)
  output/qmd_stellar/qmd_stellar_mass_radius.pdf  (both sets, new file)
  output/qmd_stellar/qmd_vs_qm_mass_radius.pdf  (QMD SET A vs QM sigma=600 only)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import MEV4_TO_GEV_FM3
from .io import output_directory, save_table
from .plotting import apply_plot_style, save_figure, sigma_colors
from .solvers.tov import run_tov_sequence


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
QM_STELLAR_DIR = OUTPUT_DIR / "stellar"

_COL_PRESSURE = 9   # pressure_mev4 (0-indexed)
_COL_ENERGY = 12    # energy_density_mev4
_COL_BARYON = 11    # baryon_density_mev3

# House style: viridis colors for SET A and SET B
SET_COLORS = plt.cm.viridis(np.linspace(0.15, 0.6, 2))

# QMD SET A uses m_sigma = 600 MeV (from qmd_parameters.py).
# Use the same color as the QM sigma_600 viridis entry (3 curves → index 2).
QM_SIGMA_600_COLOR = plt.cm.viridis(np.linspace(0.15, 0.6, 3))[2]
QM_MATCH_SIGMA_MEV = 600  # sigma mass matching QMD meson sector


def _load_stable_eos(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, comments="#", usecols=(_COL_PRESSURE, _COL_ENERGY, _COL_BARYON))
    return data[:, 0], data[:, 1], data[:, 2]


def _surface_energy_density(pressure: np.ndarray, energy: np.ndarray) -> float:
    """Extrapolate linearly from the first two EoS points to find ε at P=0."""
    slope = (energy[1] - energy[0]) / (pressure[1] - pressure[0])
    return float(max(0.0, energy[0] - slope * pressure[0]))


@dataclass
class QMDStellarEOS:
    """Duck-typed EoS wrapper providing the tov_branch() interface for run_tov_sequence."""

    pressure_mev4: np.ndarray
    energy_density_mev4: np.ndarray
    baryon_density_mev3: np.ndarray
    m_sigma_mev: float = 0.0    # not applicable; kept for interface compatibility
    b0_mev4: float = 0.0
    b_mev4: float = 0.0
    b_min_mev4: float | None = None

    def tov_branch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pressure_mev4, energy_mev4) starting from P=0 for TOV."""
        p = self.pressure_mev4
        e = self.energy_density_mev4
        pos = p > 0.0
        p_pos, e_pos = p[pos], e[pos]
        if p_pos.size < 2:
            raise ValueError("QMD EoS needs at least two positive-pressure points for TOV.")
        # Strip vacuum-like leading artifacts: Maxwell construction at low mu_q can leave
        # a few points with P,ε ≈ 0 before the self-bound surface, causing a >100x jump
        # in P to the physical quark matter region. Detect and remove such prefixes.
        if p_pos.size >= 3:
            jumps = np.where(p_pos[1:] > 1000.0 * p_pos[:-1])[0]
            if jumps.size > 0:
                p_pos = p_pos[jumps[0] + 1:]
                e_pos = e_pos[jumps[0] + 1:]
        e_surface = _surface_energy_density(p_pos, e_pos)
        pressure = np.concatenate(([0.0], p_pos))
        energy = np.concatenate(([e_surface], e_pos))
        order = np.argsort(pressure)
        pressure, energy = pressure[order], energy[order]
        unique = np.concatenate(([True], np.diff(pressure) > 0.0))
        return pressure[unique], energy[unique]


def _save_mass_radius(
    path: Path,
    sequence,
    tag: str,
    eos_path: Path,
) -> None:
    metadata = {
        "pipeline": "quark_stars",
        "product": "qmd_mass_radius",
        "tag": tag,
        "eos_file": eos_path.name,
        "units": (
            "Pc_dimless in npemu units (P/e0); "
            "epsilon_c in MeV^4 and GeV fm^-3; "
            "radius in km; mass in Msun"
        ),
    }
    data = np.column_stack([
        sequence.central_pressure_dimless,
        sequence.central_energy_density_mev4,
        sequence.central_energy_density_mev4 * MEV4_TO_GEV_FM3,
        sequence.radius_km,
        sequence.mass_msun,
        sequence.stable_mask.astype(int),
    ])
    save_table(
        path,
        ["Pc_dimless", "epsilon_c_mev4", "epsilon_c_gev_fm3", "radius_km", "mass_msun", "stable_flag"],
        data,
        metadata,
    )


def _print_diagnostics(sequence, tag: str, eos: QMDStellarEOS) -> None:
    stable = sequence.stable_mask.astype(bool)
    p_branch, e_branch = eos.tov_branch()

    print(f"\nDiagnostics {tag.upper()}:")
    print(f"  EoS points loaded: {eos.pressure_mev4.size}")
    print(
        f"  P range: {eos.pressure_mev4.min():.4e} – {eos.pressure_mev4.max():.4e} MeV^4"
    )
    print(
        f"  ε range: {eos.energy_density_mev4.min():.4e} – {eos.energy_density_mev4.max():.4e} MeV^4"
    )
    print(
        f"  P=0 surface ε (extrapolated): {e_branch[0]:.4e} MeV^4"
        f"  ({e_branch[0] * MEV4_TO_GEV_FM3:.4f} GeV fm^-3)"
    )
    print(f"  TOV stars computed: {sequence.mass_msun.size}")
    print(f"  Stable TOV stars:   {stable.sum()}")

    if stable.any():
        m_s = sequence.mass_msun[stable]
        r_s = sequence.radius_km[stable]
        idx = int(np.argmax(m_s))
        print(f"  M_max = {m_s[idx]:.4f} Msun  at  R = {r_s[idx]:.4f} km")
    else:
        print("  No stable stars found.")


def _plot_single_mass_radius(sequence, path: Path, tag: str, color) -> None:
    """Plot a single-set mass-radius curve (kept for backwards compatibility)."""
    stable = sequence.stable_mask.astype(bool)
    unstable = ~stable

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if stable.any():
        ax.plot(
            sequence.radius_km[stable],
            sequence.mass_msun[stable],
            color=color,
            linewidth=2.2,
            label="Stable",
        )
    if unstable.any():
        ax.plot(
            sequence.radius_km[unstable],
            sequence.mass_msun[unstable],
            color=color,
            linewidth=1.6,
            linestyle="--",
            label="Unstable",
        )
    if stable.any():
        m_s = sequence.mass_msun[stable]
        r_s = sequence.radius_km[stable]
        idx = int(np.argmax(m_s))
        ax.plot(
            r_s[idx],
            m_s[idx],
            "o",
            color=color,
            markersize=6,
            label=rf"$M_\mathrm{{max}} = {m_s[idx]:.2f}\,M_\odot$",
        )

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_title(rf"QMD mass-radius ({tag.upper()})")
    ax.legend()
    save_figure(path)


def _plot_combined_mass_radius(
    sequences: dict[str, object],
    path: Path,
) -> None:
    """Plot mass-radius curves for all available sets in one figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    set_styles = [
        ("set_a", "SET A", "-"),
        ("set_b", "SET B", "-"),
    ]

    for i, (tag, label, ls) in enumerate(set_styles):
        if tag not in sequences:
            continue
        seq = sequences[tag]
        # Skip non-self-bound sets: their surface P=0 energy density is ~0,
        # so the TOV integration extends to the boundary (R > 25 km).
        if np.min(seq.radius_km) > 25.0:
            print(f"  {label}: not self-bound (min R > 25 km), skipping from combined M-R plot.")
            continue
        stable = seq.stable_mask.astype(bool)
        unstable = ~stable
        color = SET_COLORS[i]

        if stable.any():
            ax.plot(
                seq.radius_km[stable],
                seq.mass_msun[stable],
                color=color,
                linewidth=2.2,
                linestyle=ls,
                label=label,
            )
        if unstable.any():
            ax.plot(
                seq.radius_km[unstable],
                seq.mass_msun[unstable],
                color=color,
                linewidth=1.4,
                linestyle="--",
            )
        if stable.any():
            m_s = seq.mass_msun[stable]
            r_s = seq.radius_km[stable]
            idx = int(np.argmax(m_s))
            ax.plot(r_s[idx], m_s[idx], "o", color=color, markersize=6)

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_title(r"QMD mass-radius")
    ax.set_xlim(5, 15)
    ax.set_ylim(0.45, 2.0)
    ax.legend()
    save_figure(path)


def _load_qm_mr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (radius_km, mass_msun, stable_mask) from a qm_stars_*.txt file."""
    data = np.loadtxt(path, comments="#")
    return data[:, 3], data[:, 4], data[:, 5].astype(bool)


def _plot_qmd_vs_qm(sequence, qmd_path: Path, qm_dir: Path) -> None:
    """Overlay QMD SET A stable branch against QM sigma=600 MeV curves only.

    QMD SET A uses m_sigma=600 MeV in its meson sector (see qmd_parameters.py),
    so the directly comparable QM sequence is the sigma=600 one.
    """
    # Middle bag constant B^{1/4}=27.8 MeV for QM reference
    qm_file = qm_dir / f"qm_stars_sigma_{QM_MATCH_SIGMA_MEV}_Broot_28.txt"
    if not qm_file.exists():
        all_files = sorted(qm_dir.glob(f"qm_stars_sigma_{QM_MATCH_SIGMA_MEV}_*.txt"))
        if not all_files:
            print(
                f"  No QM M-R files found for sigma={QM_MATCH_SIGMA_MEV} MeV; "
                "skipping comparison plot."
            )
            return
        qm_file = all_files[0]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Draw single QM sigma=600 reference curve (stable + unstable)
    qm_color = QM_SIGMA_600_COLOR
    r_km, m_msun, stable_qm = _load_qm_mr(qm_file)
    unstable_qm = ~stable_qm
    ax.plot(
        r_km[stable_qm], m_msun[stable_qm],
        color=qm_color, linewidth=1.8, linestyle="-",
        label=rf"QM $m_\sigma=600\,\mathrm{{MeV}}$",
    )
    if unstable_qm.any():
        ax.plot(
            r_km[unstable_qm], m_msun[unstable_qm],
            color=qm_color, linewidth=1.2, linestyle="--",
        )
    if stable_qm.any():
        m_qm_s = m_msun[stable_qm]
        r_qm_s = r_km[stable_qm]
        idx_qm = int(np.argmax(m_qm_s))
        ax.plot(r_qm_s[idx_qm], m_qm_s[idx_qm], "o", color=qm_color, markersize=6)

    # QMD SET A curve
    qmd_color = SET_COLORS[0]
    stable = sequence.stable_mask.astype(bool)
    unstable = ~stable
    ax.plot(
        sequence.radius_km[stable], sequence.mass_msun[stable],
        color=qmd_color, linewidth=2.4, linestyle="-",
        label=r"QMD SET A",
    )
    if unstable.any():
        ax.plot(
            sequence.radius_km[unstable], sequence.mass_msun[unstable],
            color=qmd_color, linewidth=1.4, linestyle="--",
        )
    if stable.any():
        m_s = sequence.mass_msun[stable]
        r_s = sequence.radius_km[stable]
        idx = int(np.argmax(m_s))
        ax.plot(r_s[idx], m_s[idx], "o", color=qmd_color, markersize=6)

    ax.set_xlabel(r"Radius $R\;(\mathrm{km})$")
    ax.set_ylabel(r"Mass $M\;(M_\odot)$")
    ax.set_title(
        rf"QMD vs QM ($m_\sigma={QM_MATCH_SIGMA_MEV}\,\mathrm{{MeV}}$) mass-radius"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=9)
    ax.set_xlim(5, 14)
    ax.set_ylim(0.45, 2.0)
    save_figure(qmd_path)


def main() -> None:
    apply_plot_style()
    stellar_dir = output_directory(OUTPUT_DIR, "qmd_stellar")

    print("=" * 70)
    print("QMD Stellar Mass-Radius Sequences")
    print("  Omega_1_num follows the QMD EoS table metadata.")
    print("=" * 70)

    eos_dir = OUTPUT_DIR / "qmd_stellar_eos"
    set_configs = [
        ("set_a", eos_dir / "qmd_stellar_eos_set_a_stable.txt"),
        ("set_b", eos_dir / "qmd_stellar_eos_set_b_stable.txt"),
    ]

    sequences: dict[str, object] = {}
    seq_a = None

    for tag, eos_path in set_configs:
        if not eos_path.exists():
            print(f"\nSkipping {tag.upper()}: stable EoS file not found: {eos_path}")
            continue

        print(f"\nLoading {tag.upper()} EoS from {eos_path.name} ...")
        pressure, energy, baryon = _load_stable_eos(eos_path)
        print(f"  Loaded {pressure.size} stable EoS points.")

        eos = QMDStellarEOS(
            pressure_mev4=pressure,
            energy_density_mev4=energy,
            baryon_density_mev3=baryon,
        )

        print(f"  Running TOV integration for {tag.upper()} ...")
        try:
            sequence = run_tov_sequence(eos)
        except ValueError as exc:
            print(f"  TOV failed for {tag.upper()}: {exc}")
            continue

        _print_diagnostics(sequence, tag, eos)
        sequences[tag] = sequence

        mr_path = stellar_dir / f"qmd_stars_{tag}.txt"
        _save_mass_radius(mr_path, sequence, tag, eos_path)
        print(f"  Saved M-R table: {mr_path}")

        # Single-set plot (old filename kept for backwards compat)
        color_idx = 0 if tag == "set_a" else 1
        pdf_path = stellar_dir / f"qmd_mass_radius_{tag}.pdf"
        _plot_single_mass_radius(sequence, pdf_path, tag, SET_COLORS[color_idx])
        print(f"  Saved M-R plot:  {pdf_path}")

        if tag == "set_a":
            seq_a = sequence

    # Combined mass-radius plot (both sets)
    if sequences:
        combined_path = stellar_dir / "qmd_stellar_mass_radius.pdf"
        _plot_combined_mass_radius(sequences, combined_path)
        print(f"\nSaved combined M-R plot: {combined_path}")

    # QMD vs QM comparison (SET A vs sigma=600 only)
    if seq_a is not None:
        cmp_path = stellar_dir / "qmd_vs_qm_mass_radius.pdf"
        _plot_qmd_vs_qm(seq_a, cmp_path, QM_STELLAR_DIR)
        print(f"Saved comparison plot: {cmp_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
