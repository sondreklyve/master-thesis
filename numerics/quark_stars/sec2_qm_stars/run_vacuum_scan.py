"""Scan the QM vacuum potential and select the lowest valid m_sigma values."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..io import ensure_directory, save_table
from ..plotting import PALETTE_6, apply_plot_style, save_figure, sigma_label
from ..qm_parameters import DEFAULT_QM_VACUUM_INPUTS
from ..vacuum_scan import DEFAULT_M_SIGMA_VALUES_MEV, scan_vacuum_stability


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "thesis" / "figures" / "quark_stars" / "qm_stars"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-sigma-values", type=float, nargs="+", default=list(DEFAULT_M_SIGMA_VALUES_MEV))
    parser.add_argument("--sigma-max-ratio", type=float, default=1.6)
    parser.add_argument("--num-sigma-points", type=int, default=600)
    parser.add_argument("--rise-fraction-threshold", type=float, default=0.01)
    parser.add_argument("--local-window-ratio", type=float, default=0.25)
    parser.add_argument("--selection-count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def save_vacuum_plot(results, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axvline(DEFAULT_QM_VACUUM_INPUTS.f_pi_mev, color="black", linestyle="--", linewidth=1.0)
    for result, color in zip(results, PALETTE_6, strict=True):
        f_pi_mev = DEFAULT_QM_VACUUM_INPUTS.f_pi_mev
        omega_scaled = (result.omega_mev4) / f_pi_mev**4
        style = "-" if result.valid else "--"
        ax.plot(
            result.sigma_grid_mev,
            omega_scaled,
            color=color,
            linewidth=2.2,
            linestyle=style,
            label=f"{sigma_label(result.m_sigma_mev)} ({'valid' if result.valid else 'invalid'})",
        )
        if result.valid:
            ax.plot(
                result.sigma_min_mev,
                result.omega_min_mev4 / f_pi_mev**4,
                marker="o",
                color=color,
                markersize=5,
            )
    ax.set_xlabel(r"Condensate $\sigma\;(\mathrm{MeV})$")
    ax.set_ylabel(r"$\Omega(\sigma)/f_\pi^4$")
    ax.set_xlim(0.0, max(result.sigma_grid_mev[-1] for result in results))
    ax.legend(ncol=2)
    save_figure(output_dir / "vacuum_potential_sigma_scan.pdf")


def main() -> None:
    args = parse_args()
    apply_plot_style()
    vacuum_dir = ensure_directory(args.output_dir / "sec2_vacuum_scan")

    results = scan_vacuum_stability(
        args.m_sigma_values,
        sigma_max_ratio=args.sigma_max_ratio,
        num_sigma_points=args.num_sigma_points,
        rise_fraction_threshold=args.rise_fraction_threshold,
        local_window_ratio=args.local_window_ratio,
    )
    valid_results = [result for result in results if result.valid]
    selected_results = valid_results[: args.selection_count]

    summary_data = np.array(
        [
            [
                result.m_sigma_mev,
                result.sigma_min_mev,
                result.curvature_mev2,
                result.left_rise_mev4,
                result.right_rise_mev4,
                float(result.valid),
            ]
            for result in results
        ]
    )
    save_table(
        vacuum_dir / "vacuum_scan_summary.txt",
        [
            "m_sigma_mev",
            "sigma_min_mev",
            "curvature_mev2",
            "left_rise_mev4",
            "right_rise_mev4",
            "valid_flag",
        ],
        summary_data,
        {
            "pipeline": "vacuum_scan",
            "criterion": "interior minimum with positive curvature and finite rise on both sides in the local vacuum window",
            "sigma_max_ratio": f"{args.sigma_max_ratio:.6f}",
            "rise_fraction_threshold": f"{args.rise_fraction_threshold:.6f}",
            "local_window_ratio": f"{args.local_window_ratio:.6f}",
        },
    )

    valid_data = np.array([[result.m_sigma_mev] for result in valid_results], dtype=float)
    save_table(
        vacuum_dir / "valid_m_sigma_values.txt",
        ["m_sigma_mev"],
        valid_data,
        {"pipeline": "vacuum_scan", "selection": "all_valid"},
    )

    selected_data = np.array([[result.m_sigma_mev] for result in selected_results], dtype=float)
    save_table(
        vacuum_dir / "selected_m_sigma_values.txt",
        ["m_sigma_mev"],
        selected_data,
        {"pipeline": "vacuum_scan", "selection": f"lowest_{args.selection_count}_valid"},
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_vacuum_plot(results, FIGURES_DIR)


if __name__ == "__main__":
    main()
