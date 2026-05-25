"""Shared plotting helpers for the quark-star workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Shared axis limits for c_s² plots — used by both run_qmd_stellar and run_section2_sweep
# so both plots stay in sync.
CS2_XLIM: tuple[float, float] = (250.0, 800.0)
CS2_YLIM: tuple[float, float] = (0.15, 0.55)
CS2_MU_MIN: float = 250.0  # lower μ_q bound for cs² curves (matches xlim left edge)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "figure.figsize": (7.6, 5.0),
            "savefig.bbox": "tight",
        }
    )


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def sigma_label(m_sigma_mev: float) -> str:
    return rf"$m_\sigma = {m_sigma_mev:.0f}\,\mathrm{{MeV}}$"


def sigma_colors(num_curves: int) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0.2, 0.85, num_curves))


def bag_curve_label(b_root_mev: float) -> str:
    return rf"$B^{{1/4}} = {b_root_mev:.1f}\,\mathrm{{MeV}}$"


def line_plot(x, y, xlabel: str, ylabel: str, title: str, path: Path, *, color: str = "#1f4e79") -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, linewidth=2.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(path)
