"""Shared plotting helpers for the quark-star workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


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


def line_plot(x, y, xlabel: str, ylabel: str, title: str, path: Path, *, color: str = "#1f4e79") -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, linewidth=2.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(path)
